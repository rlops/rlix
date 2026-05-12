"""RLix multi-pipeline launcher for NeMo RL async-GRPO pipelines.

Mirrors examples/start_multi_pipeline_test.py (ROLL path) but loads NeMo RL
configs and creates NemoRLFullFinetunePipeline actors via PipelineCoordinator.

Usage:
  python examples/start_nemo_rl_multi_pipeline.py \\
    --config_name nemo_rl_pipeline1_2gpu,nemo_rl_pipeline2_2gpu

Wrapper yaml schema (see examples/nemo_rl_test/*.yaml):
  pipeline_cls            — dotted path; must be NemoRLFullFinetunePipeline
  nemo_config_path        — path to NeMo RL master yaml
  nemo_config_overrides   — list[str] hydra-style overrides
  train_device_mapping    — list[int] GPUs for Megatron training
  infer_device_mapping    — list[int] GPUs for vLLM inference (must be a superset of train)
  num_gpus_per_node       — int
  verify_model_after_sync — bool
  actor_train / actor_infer / reference — structural stubs read by the rlix
                                          PipelineCoordinator schema validators
                                          (sleep_level=2, offload_nccl=True, etc.)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ray
from omegaconf import OmegaConf

from rlix.pipeline import COORDINATOR_MAX_CONCURRENCY
from rlix.pipeline.nemo_rl_config_bridge import register_nemo_rl_pipeline
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX, RLIX_NAMESPACE
from rlix.utils.env import pipeline_identity_env_vars, thread_limit_env_vars


def _load_nemo_master_config(*, nemo_config_path: str, overrides: List[str]) -> Any:
    """Load a NeMo RL master config + apply hydra overrides.

    Returned object supports both attribute access (used by
    register_nemo_rl_pipeline -> extract_topology_validation_inputs) and
    dict-style traversal (used downstream).
    """
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    cfg = load_config(nemo_config_path)
    if overrides:
        cfg = parse_hydra_overrides(cfg, list(overrides))
    return cfg  # OmegaConf DictConfig — supports cfg.policy.generation.vllm_cfg.* attribute access


def _build_pipeline_config(*, wrapper_cfg: Any) -> Any:
    """Resolve the wrapper Hydra config into a DictConfig pipeline_config.

    PipelineCoordinator validators mix attribute and dict access:
      getattr(actor_infer, "strategy_args").strategy_config.get("sleep_level")
    OmegaConf DictConfig satisfies both surfaces, so we keep the structured
    config and only resolve interpolations.

    Required fields consumed downstream:
      - pipeline_cls (str)
      - nemo_config_path (str)
      - nemo_config_overrides (list[str])
      - train_device_mapping / infer_device_mapping (list[int])
      - num_gpus_per_node (int)
      - verify_model_after_sync (bool)
      - actor_train / actor_infer (structural — schema validators read
        offload_nccl + strategy_args.strategy_name + sleep_level)
    """
    OmegaConf.resolve(wrapper_cfg)
    return wrapper_cfg


def _resolve_wrapper_path(*, config_path: str, config_name: str) -> Path:
    """Resolve a wrapper yaml relative to examples/{config_path}/{config_name}.yaml."""
    script_dir = Path(__file__).resolve().parent
    base = Path(config_path)
    if not base.is_absolute():
        base = script_dir / base
    target = base / f"{config_name}.yaml"
    if not target.exists():
        raise FileNotFoundError(f"Wrapper config not found: {target}")
    return target


def main() -> None:
    from rlix.pipeline.coordinator import PipelineCoordinator
    import rlix

    parser = argparse.ArgumentParser(
        description="RLix multi-pipeline launcher for NeMo RL async GRPO"
    )
    parser.add_argument(
        "--config_path",
        default="nemo_rl_test",
        help="Wrapper yaml directory (relative to examples/, default nemo_rl_test/)",
    )
    parser.add_argument(
        "--config_name",
        default="nemo_rl_pipeline1_2gpu",
        help="Comma-separated wrapper yaml names (no .yaml suffix)",
    )
    parser.add_argument(
        "--admit-delay-s",
        type=float,
        default=0.0,
        help="Sleep between admit_pipeline calls (except after the last one).",
    )
    args = parser.parse_args()

    config_names = [s.strip() for s in args.config_name.split(",") if s.strip()]
    if not config_names:
        raise ValueError("--config_name must be non-empty")

    wrapper_paths = [
        _resolve_wrapper_path(config_path=args.config_path, config_name=cn)
        for cn in config_names
    ]

    # Parse wrapper configs and corresponding NeMo master configs up front, before ray.init().
    wrapper_configs: List[Any] = []   # SimpleNamespace pipeline_configs (for coordinator + pipeline actor)
    nemo_configs: List[Any] = []      # OmegaConf master configs (for orchestrator topology validation)
    for idx, (cn, wp) in enumerate(zip(config_names, wrapper_paths), start=1):
        wrapper_cfg = OmegaConf.load(wp)
        suffix = f"mp{idx}"
        if hasattr(wrapper_cfg, "exp_name") and wrapper_cfg.exp_name:
            wrapper_cfg.exp_name = f"{wrapper_cfg.exp_name}-{suffix}"
        else:
            wrapper_cfg.exp_name = f"{cn}-{suffix}"

        nemo_path = OmegaConf.select(wrapper_cfg, "nemo_config_path")
        if not nemo_path:
            raise RuntimeError(f"{wp}: missing nemo_config_path")
        overrides = list(OmegaConf.select(wrapper_cfg, "nemo_config_overrides") or [])
        nemo_cfg = _load_nemo_master_config(
            nemo_config_path=str(nemo_path), overrides=overrides
        )

        pipeline_config = _build_pipeline_config(wrapper_cfg=wrapper_cfg)
        wrapper_configs.append(pipeline_config)
        nemo_configs.append(nemo_cfg)

    # Bring up local Ray + RLix control plane.
    _thread_env = thread_limit_env_vars()
    # debug #53 (v47): cgroup pids.max=3840 cap. Each Ray actor's CoreWorker
    # boots ~30-50 boost::asio threads by default. Verified Ray env keys
    # (grep'd from ray/_raylet.so binary strings) reduce per-actor thread
    # pool sizes; same defaults inherited by every child actor via ray.init
    # runtime_env. Combined with OMP/MKL/RAYON/OPENBLAS/TOKENIZERS=1 we
    # save ~25 threads/actor × ~14 actors = ~350 pids.
    for _ray_thread_var in (
        "RAY_num_server_call_thread",
        "RAY_num_grpc_internal_threads",
        "RAY_worker_num_grpc_internal_threads",
        "RAY_object_manager_rpc_threads_num",
        "RAY_gcs_server_rpc_server_thread_num",
        "RAY_gcs_server_rpc_client_thread_num",
    ):
        _thread_env[_ray_thread_var] = os.environ.get(_ray_thread_var, "1")
    for _misc_thread_var in (
        "TOKENIZERS_PARALLELISM",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ):
        _thread_env[_misc_thread_var] = os.environ.get(
            _misc_thread_var,
            "false" if _misc_thread_var == "TOKENIZERS_PARALLELISM" else "1",
        )
    # Pass through NCCL_DEBUG / NCCL_DEBUG_SUBSYS from driver shell so workers emit diagnostic logs.
    # debug #56 (v49 segfault): TORCH_NCCL_ENABLE_MONITORING also needed in
    # passthrough — otherwise child Ray actor venvs spawn HeartbeatMonitor
    # threads that segfault during getenv lookup under cgroup pids pressure.
    for _passthrough in (
        "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE",
        "TORCH_NCCL_ENABLE_MONITORING",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        "RLIX_BUCKET_SIZE_BYTES",
    ):
        if _passthrough in os.environ:
            _thread_env[_passthrough] = os.environ[_passthrough]
    # Force-disable HeartbeatMonitor in child actor venvs even if shell didn't
    # set it (defensive). cgroup pids pressure + watchdog thread = segfault.
    _thread_env.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    if not ray.is_initialized():
        ray.init(
            namespace=RLIX_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env={"env_vars": _thread_env},
        )

    orchestrator = rlix.init(create_if_missing=True)
    if orchestrator is None:
        raise RuntimeError("rlix.init returned None")

    CoordinatorActor = ray.remote(PipelineCoordinator)

    coordinators: List[Any] = []
    pipeline_actors: List[Any] = []
    pipeline_ids: List[str] = []
    run_refs: List[Any] = []

    admit_delay_s = float(args.admit_delay_s)

    for i, (pipeline_config, nemo_cfg) in enumerate(zip(wrapper_configs, nemo_configs)):
        train_dm = list(getattr(pipeline_config, "train_device_mapping"))
        infer_dm = list(getattr(pipeline_config, "infer_device_mapping"))
        registration = register_nemo_rl_pipeline(
            orchestrator=orchestrator,
            nemo_config=nemo_cfg,
            train_device_mapping=train_dm,
            infer_device_mapping=infer_dm,
        )

        coordinator_actor = CoordinatorActor.options(
            name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{registration.pipeline_id}",
            namespace=registration.ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            max_concurrency=COORDINATOR_MAX_CONCURRENCY,
            runtime_env={"env_vars": {
                **pipeline_identity_env_vars(
                    pipeline_id=registration.pipeline_id,
                    ray_namespace=registration.ray_namespace,
                ),
                **thread_limit_env_vars(),
            }},
        ).remote(
            pipeline_id=registration.pipeline_id,
            pipeline_config=pipeline_config,
        )
        coordinators.append(coordinator_actor)

        pipeline_actor = ray.get(
            coordinator_actor.create_pipeline_actor.remote(pipeline_config=pipeline_config)
        )
        pipeline_actors.append(pipeline_actor)
        pipeline_ids.append(registration.pipeline_id)

        # Arm pair-init barrier BEFORE pipeline.run starts (debug #48). Has to
        # be armed before the actor reaches its first _after_training so the
        # check there blocks until paired pipeline's vLLM init completes.
        # Arming after wait_for_first_after_training would race the check.
        if i < len(wrapper_configs) - 1:
            try:
                ray.get(pipeline_actor.arm_pair_setup_barrier.remote())
                print(
                    f"pair-init barrier armed on {registration.pipeline_id} "
                    f"before run.remote()",
                    flush=True,
                )
            except Exception as e:
                print(f"pair-init barrier arm failed: {e!r}", flush=True)

        run_refs.append(pipeline_actor.run.remote())

        # Step-boundary admission + pair-init barrier: don't admit ppl_{i+1}
        # until ppl_i has done one full step cycle (debug #44). Then ARM the
        # pair-init barrier on ppl_i so its next _after_training waits until
        # we've signaled ppl_{i+1}'s vLLM is ready (debug #48). This prevents
        # ppl_{i+1} vLLM init from racing with ppl_i step 1+ Megatron train,
        # which would steal GPU memory and fail KV cache check.
        leading_actor = pipeline_actor  # capture before next iter overwrites
        if i < len(wrapper_configs) - 1 and pipeline_actors:
            print(
                f"step-boundary admission: waiting for {registration.pipeline_id} "
                f"first after_training (timeout={admit_delay_s}s)",
                flush=True,
            )
            try:
                ok = ray.get(
                    leading_actor.wait_for_first_after_training.remote(
                        timeout_s=max(admit_delay_s, 30.0),
                    ),
                    timeout=max(admit_delay_s + 30.0, 60.0),
                )
                print(
                    f"step-boundary admission: {registration.pipeline_id} reached "
                    f"first after_training (signaled={ok}) — admitting next pipeline",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"step-boundary admission: wait failed ({e!r}); falling back to "
                    f"admit_delay_s={admit_delay_s}s",
                    flush=True,
                )
                if admit_delay_s > 0:
                    import time
                    time.sleep(admit_delay_s)

            # Arm pair-init barrier so ppl_i pauses on its NEXT after_training
            # until we signal ppl_{i+1}'s vLLM init is done.
            try:
                ray.get(leading_actor.arm_pair_setup_barrier.remote())
            except Exception as e:
                print(
                    f"pair-init barrier arm failed ({e!r}); ppl_{{i+1}} vLLM init "
                    f"may race ppl_{{i}} train → memory error",
                    flush=True,
                )

    # Pair-init barrier release (debug #48): once each trailing pipeline's
    # _setup_nemo_rl_objects (incl. vLLM init) reports complete, signal the
    # leading pipeline to drop its pair-init barrier. Spawn a daemon thread
    # for each leading→trailing pair so ray.get(run_refs) below isn't blocked
    # on these signals.
    import threading as _threading
    def _release_pair_barrier(leading_actor, trailing_actor, pair_label: str):
        try:
            ok = ray.get(
                trailing_actor.wait_for_setup_complete.remote(timeout_s=600.0),
                timeout=660.0,
            )
            print(
                f"pair-init signal: {pair_label} trailing setup complete "
                f"(signaled={ok}) — releasing leading barrier",
                flush=True,
            )
        except Exception as e:
            print(
                f"pair-init signal: {pair_label} wait failed ({e!r}); "
                f"releasing leading barrier anyway",
                flush=True,
            )
        try:
            ray.get(leading_actor.signal_pair_setup_complete.remote())
        except Exception as e:
            print(f"pair-init signal: {pair_label} release failed: {e!r}", flush=True)

    for idx in range(len(pipeline_actors) - 1):
        leading = pipeline_actors[idx]
        trailing = pipeline_actors[idx + 1]
        t = _threading.Thread(
            target=_release_pair_barrier,
            args=(leading, trailing, f"ppl{idx}→ppl{idx + 1}"),
            daemon=True,
        )
        t.start()

    # Per-pipeline outcome handling (debug #51): ray.get(run_refs) is fail-fast
    # so one pipeline crashing tears down the rest. Wait for each individually
    # and collect results so a single ppl crash doesn't kill the others.
    pending = list(run_refs)
    successes = 0
    failures = 0
    pipeline_id_for_ref = {ref: pid for ref, pid in zip(run_refs, pipeline_ids)}
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=None)
        for ref in ready:
            pid = pipeline_id_for_ref.get(ref, "<unknown>")
            try:
                ray.get(ref)
                print(f"pipeline {pid} run() returned successfully", flush=True)
                successes += 1
            except Exception as e:
                print(f"pipeline {pid} run() raised: {type(e).__name__}: {e}", flush=True)
                failures += 1
            # v75 (debug #66): explicit graceful unregister AFTER pipeline.run()
            # has fully returned (including its post-loop _await_release_actor_infer).
            # Without this, the scheduler eventually hits ActorDiedError on the
            # coordinator and routes through debug #50 _gather_resize_tolerate_dead,
            # which races with the still-pending await_release on a peer pipeline
            # (cosmetic warning; potential cross-GPU cleanup risk on slower runs).
            try:
                # orchestrator is a Ray actor handle (rlix.client.client returns
                # ray.remote(Orchestrator)...remote()); must use .remote() + ray.get.
                ray.get(orchestrator.unregister_pipeline.remote(pid))
                print(f"pipeline {pid} unregistered", flush=True)
            except Exception as e:
                print(
                    f"pipeline {pid} unregister failed (continuing): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
    print(f"done!!! successes={successes} failures={failures}")
    if failures and not successes:
        # All failed: surface non-zero exit so CI catches it.
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
