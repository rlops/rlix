"""
RLix multi-pipeline example (ENG-123).

This ports the fork reference configs (`pipeline1_sokoban_grpo.yaml`, `pipeline2_sokoban_grpo.yaml`) and provides a
driver that runs 1+ pipelines concurrently under the RLix control plane.

Usage (from repo root):
  python external/ROLL_rlix/examples/multi_pipeline/start_multi_pipeline_test.py --config_name pipeline1_sokoban_grpo
  python external/ROLL_rlix/examples/multi_pipeline/start_multi_pipeline_test.py --config_name pipeline1_sokoban_grpo,pipeline2_sokoban_grpo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import ray
from dacite import from_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from rlix.pipeline import COORDINATOR_MAX_CONCURRENCY
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX, RLIX_NAMESPACE


def _repo_root() -> Path:
    # Resolve the mono-repo root regardless of where this example is vendored.
    #
    # We intentionally avoid relying on a fixed `parents[N]` depth because this file
    # lives under `external/ROLL_rlix/...` in this workspace (vs `third_party/ROLL/...`
    # in other layouts).
    start = Path(__file__).resolve()
    for parent in start.parents:
        git_dir = parent / ".git"
        if git_dir.exists() and git_dir.is_dir():
            return parent
        if (parent / "AGENTS.md").exists() and (parent / "rlix").is_dir():
            return parent
    raise RuntimeError(f"Failed to locate repo root from {start}")


def _resolve_roll_root(*, repo_root: Path) -> Path:
    # Prefer the in-repo ROLL+RLix fork used by ENG-123.
    candidates = [
        repo_root / "external" / "ROLL_rlix",
        repo_root / "third_party" / "ROLL",
        repo_root / "external" / "ROLL",
    ]
    for candidate in candidates:
        if (candidate / "roll").is_dir():
            return candidate.resolve()
    raise RuntimeError(f"Failed to locate ROLL root under repo_root={repo_root} (tried {candidates})")


def _ensure_import_paths() -> tuple[Path, Path]:
    repo_root = _repo_root()
    roll_root = _resolve_roll_root(repo_root=repo_root)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(roll_root))
    return repo_root, roll_root


def _resolve_hydra_config_path(*, roll_root: Path, arg_config_path: str) -> tuple[str, Path]:
    script_dir = Path(__file__).resolve().parent
    examples_dir = (roll_root / "examples").resolve()
    config_path = Path(arg_config_path)

    if config_path.is_absolute():
        return str(config_path), config_path

    script_relative_dir = (script_dir / config_path).resolve()
    if script_relative_dir.is_dir():
        return str(config_path), script_relative_dir

    examples_relative_dir = (examples_dir / config_path).resolve()
    if examples_relative_dir.is_dir():
        hydra_config_path = os.path.relpath(examples_relative_dir, script_dir)
        return hydra_config_path, examples_relative_dir

    roll_relative_dir = (roll_root / config_path).resolve()
    if roll_relative_dir.is_dir():
        hydra_config_path = os.path.relpath(roll_relative_dir, script_dir)
        return hydra_config_path, roll_relative_dir

    raise FileNotFoundError(
        f"Config directory not found. Received --config_path={arg_config_path!r} "
        f"(tried {script_relative_dir}, {examples_relative_dir}, {roll_relative_dir})"
    )


def _cluster_registry_inputs(*, pipeline_config: Any) -> tuple[Dict[str, int], Dict[str, List[int]]]:
    cluster_tp_configs: Dict[str, int] = {}
    cluster_device_mappings: Dict[str, List[int]] = {}

    for key in ("actor_train", "actor_infer", "reference", "critic", "reward"):
        # Only register clusters that will actually be constructed by the pipeline.
        if key == "reference" and hasattr(pipeline_config, "enable_reference") and not pipeline_config.enable_reference:
            continue
        cfg = getattr(pipeline_config, key, None)
        if cfg is None:
            continue
        mapping = getattr(cfg, "device_mapping", None)
        if mapping is None:
            continue
        cluster_device_mappings[key] = list(mapping)
        cluster_tp_configs[key] = int(getattr(cfg, "num_gpus_per_worker", 1))

    if "actor_infer" not in cluster_tp_configs:
        raise RuntimeError("pipeline_config must include actor_infer device_mapping for RLix mode")
    return cluster_tp_configs, cluster_device_mappings


def _pipeline_type(pipeline_config: Any) -> str:
    """Return 'lora' if the config has LoRA adapters configured, else 'ft'.

    Mirrors the same lora detection used in RLixCoordinator.create_pipeline_actor().
    Source: rlix/pipeline/coordinator.py
    """
    adapters = getattr(getattr(pipeline_config, "actor_train", None), "model_args", None)
    adapters = getattr(adapters, "adapters", None) if adapters is not None else None
    return "lora" if adapters else "ft"


def main() -> None:
    repo_root, roll_root = _ensure_import_paths()

    from roll.pipeline.agentic.agentic_config import AgenticConfig
    from rlix.pipeline.coordinator import RLixCoordinator, _get_pipeline_namespace

    import rlix

    parser = argparse.ArgumentParser(description="RLix multi-pipeline example")
    parser.add_argument(
        "--config_path",
        default="multi_pipeline",
        help="Path to config directory (relative to third_party/ROLL/examples/)",
    )
    parser.add_argument(
        "--config_name",
        default="pipeline1_sokoban_grpo",
        help="Comma-separated config file names (without .yaml)",
    )
    parser.add_argument(
        "--admit-delay-s",
        type=float,
        default=0.0,
        help="Seconds to sleep after admitting each pipeline (except the last).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the fully resolved Hydra config to logs (can be very large).",
    )
    args = parser.parse_args()

    config_names = [name.strip() for name in args.config_name.split(",") if name.strip()]
    if not config_names:
        raise ValueError("--config_name must be non-empty")

    # Make the driver + all Ray workers able to import `roll` and `rlix`.
    # (Ray workers do not inherit the driver's `sys.path` mutations.)
    pythonpath_parts = [str(repo_root), str(roll_root)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    worker_pythonpath = os.pathsep.join(pythonpath_parts)

    # This example is often run in a single-process "smoke test" setup without a pre-existing Ray cluster.
    # Initialize a local Ray runtime so rlix.init() does not require an external `ray start --head`.
    # Log before ray.init() — this is when the head node gRPC pool size is fixed.
    _grpc_pool = os.environ.get("RAY_grpc_server_thread_pool_size", "4")
    _omp = os.environ.get("OMP_NUM_THREADS", "1")
    print(f"[ENV] RAY_grpc_server_thread_pool_size={_grpc_pool}")
    print(f"[ENV] OMP_NUM_THREADS={_omp}")
    if not ray.is_initialized():
        # Pass thread-limiting vars as the Ray-side global default runtime_env.
        # Actors that specify their own runtime_env override this, but it catches
        # any actor that does not set an explicit runtime_env.
        ray.init(
            namespace=RLIX_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env={"env_vars": {
                "OMP_NUM_THREADS": _omp,
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                "RAY_grpc_server_thread_pool_size": _grpc_pool,
            }},
        )

    hydra_config_path, _ = _resolve_hydra_config_path(roll_root=roll_root, arg_config_path=args.config_path)
    GlobalHydra.instance().clear()
    initialize(config_path=hydra_config_path, job_name="rlix_multi_pipeline", version_base=None)

    pipeline_configs: List[AgenticConfig] = []
    for idx, cn in enumerate(config_names, start=1):
        cfg = compose(config_name=cn)
        suffix = f"mp{idx}"
        if hasattr(cfg, "exp_name") and cfg.exp_name:
            cfg.exp_name = f"{cfg.exp_name}-{suffix}"
        else:
            cfg.exp_name = f"{cn}-{suffix}"

        for key in ("model_name", "base_dir", "log_dir", "profiler_output_dir"):
            if hasattr(cfg, key):
                value = getattr(cfg, key)
                if isinstance(value, str) and value:
                    setattr(cfg, key, f"{value}-{suffix}")

        if args.print_config or os.environ.get("ROLL_PRINT_CONFIG", "0") == "1":
            print(OmegaConf.to_yaml(cfg, resolve=True))

        pipeline_config = from_dict(
            data_class=AgenticConfig,
            data=OmegaConf.to_container(cfg, resolve=True),
        )
        pipeline_configs.append(pipeline_config)

    # Ensure RLix control plane is up (creates orchestrator + scheduler actors).
    orchestrator = rlix.init(create_if_missing=True)
    if orchestrator is None:
        raise RuntimeError("rlix.init returned None (expected orchestrator actor handle on rank 0)")

    CoordinatorActor = ray.remote(RLixCoordinator)

    coordinators = []
    pipeline_actors = []
    run_refs = []

    admit_delay_s = float(args.admit_delay_s)

    pipeline_ids: List[str] = []
    for pipeline_config in pipeline_configs:
        # Pass the pipeline type so the id is prefixed "ft_" or "lora_" for trace readability.
        pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote(_pipeline_type(pipeline_config)))
        pipeline_ids.append(str(pipeline_id))

    for i, (pipeline_id, pipeline_config) in enumerate(zip(pipeline_ids, pipeline_configs)):
        ray_namespace = _get_pipeline_namespace(str(pipeline_id))
        cluster_tp_configs, cluster_device_mappings = _cluster_registry_inputs(pipeline_config=pipeline_config)

        ray.get(
            orchestrator.register_pipeline.remote(
                pipeline_id=str(pipeline_id),
                ray_namespace=ray_namespace,
                cluster_tp_configs=cluster_tp_configs,
                cluster_device_mappings=cluster_device_mappings,
            )
        )
        ray.get(orchestrator.admit_pipeline.remote(pipeline_id=str(pipeline_id)))

        coordinator_actor = CoordinatorActor.options(
            name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
            namespace=ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            max_concurrency=COORDINATOR_MAX_CONCURRENCY,
            # Ray does not reliably propagate env vars from parent actors. Explicitly inject the
            # per-pipeline namespace + control-plane contract for this pipeline actor process.
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": worker_pythonpath,
                    "PIPELINE_ID": str(pipeline_id),
                    "ROLL_RAY_NAMESPACE": ray_namespace,
                    "RLIX_CONTROL_PLANE": "rlix",
                    # Propagate thread-limiting vars so coordinator + pipeline actors
                    # stay within container pids.max. Falls back to safe defaults if
                    # not set in the shell.
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                    "RAY_grpc_server_thread_pool_size": os.environ.get("RAY_grpc_server_thread_pool_size", "4"),
                }
            },
        ).remote(
            pipeline_id=pipeline_id,
            pipeline_config=pipeline_config,
        )
        coordinators.append(coordinator_actor)

        pipeline_actor = ray.get(coordinator_actor.create_pipeline_actor.remote(pipeline_config=pipeline_config))
        pipeline_actors.append(pipeline_actor)
        run_refs.append(pipeline_actor.run.remote())

        if admit_delay_s > 0 and i < len(pipeline_ids) - 1:
            print(f"admit_delay_s: sleep {admit_delay_s=}")
            import time
            time.sleep(admit_delay_s)

    # Block until all pipelines complete (fail-fast if any crashes).
    ray.get(run_refs)
    print("done!!!")

if __name__ == "__main__":
    main()
