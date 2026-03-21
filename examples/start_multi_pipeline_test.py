"""
RLix multi-pipeline example.

Runs 1+ RL training pipelines concurrently under the RLix control plane.

Usage:
  python examples/start_multi_pipeline_test.py --config_name full_finetune_pipeline1
  python examples/start_multi_pipeline_test.py --config_name full_finetune_pipeline1,full_finetune_pipeline2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import ray
from dacite import from_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from rlix.pipeline import COORDINATOR_MAX_CONCURRENCY
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX, RLIX_NAMESPACE
from rlix.utils.env import pipeline_identity_env_vars, thread_limit_env_vars


def _set_launcher_logging_env(*, config_names: List[str]) -> str:
    """Pin the launcher process to one stable log directory.

    Pipeline configs mutate process env during config parsing, including ROLL_LOG_DIR.
    In a multi-pipeline launcher, that can cause the shared driver logger to keep
    rebinding to the most recently parsed pipeline's log directory. We give the
    launcher its own fixed log target and restore it after each config compose.
    """
    launcher_name = "__".join(config_names) if config_names else "launcher"
    launcher_log_dir = str((Path("./output/multi_pipeline_driver/logs") / launcher_name).resolve())
    os.environ["ROLL_LOG_DIR"] = launcher_log_dir
    return launcher_log_dir


def _resolve_hydra_config_path(arg_config_path: str) -> tuple[str, Path]:
    """Resolve the Hydra config directory relative to this script's location."""
    script_dir = Path(__file__).resolve().parent
    config_path = Path(arg_config_path)

    # Absolute path — use as-is.
    if config_path.is_absolute():
        return str(config_path), config_path

    # Relative to script directory (e.g. "rlix_test" -> examples/rlix_test/).
    resolved = (script_dir / config_path).resolve()
    if resolved.is_dir():
        return str(config_path), resolved

    raise FileNotFoundError(
        f"Config directory not found. Received --config_path={arg_config_path!r} "
        f"(tried {resolved})"
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

    Mirrors the same lora detection used in PipelineCoordinator.create_pipeline_actor().
    Source: rlix/pipeline/coordinator.py
    """
    adapters = getattr(getattr(pipeline_config, "actor_train", None), "model_args", None)
    adapters = getattr(adapters, "adapters", None) if adapters is not None else None
    return "lora" if adapters else "ft"


def main() -> None:
    from roll.pipeline.agentic.agentic_config import AgenticConfig
    from rlix.pipeline.coordinator import PipelineCoordinator, get_pipeline_namespace

    import rlix

    parser = argparse.ArgumentParser(description="RLix multi-pipeline example")
    parser.add_argument(
        "--config_path",
        default="rlix_test",
        help="Path to config directory (relative to examples/)",
    )
    parser.add_argument(
        "--config_name",
        default="full_finetune_pipeline1",
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

    launcher_log_dir = _set_launcher_logging_env(config_names=config_names)

    hydra_config_path, _ = _resolve_hydra_config_path(arg_config_path=args.config_path)
    GlobalHydra.instance().clear()
    initialize(config_path=hydra_config_path, job_name="rlix_multi_pipeline", version_base=None)

    # Parse all configs before ray.init() so that BaseConfig.__post_init__ sets
    # env vars (e.g. MODEL_DOWNLOAD_TYPE) that Ray workers will inherit.
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

        pipeline_cls_path = getattr(cfg, "pipeline_cls", None)
        pipeline_config = from_dict(
            data_class=AgenticConfig,
            data=OmegaConf.to_container(cfg, resolve=True),
        )
        if pipeline_cls_path:
            pipeline_config.pipeline_cls = pipeline_cls_path
        pipeline_configs.append(pipeline_config)

        # Config parsing mutates process-global logging env (for example ROLL_LOG_DIR).
        # Restore the launcher's fixed log target before parsing the next config and before
        # the shared driver starts orchestrating all pipelines.
        os.environ["ROLL_LOG_DIR"] = launcher_log_dir

    # Initialize a local Ray runtime if one is not already running.
    _thread_env = thread_limit_env_vars()
    if not ray.is_initialized():
        ray.init(
            namespace=RLIX_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env={"env_vars": _thread_env},
        )

    # Ensure RLix control plane is up (creates orchestrator + scheduler actors).
    orchestrator = rlix.init(create_if_missing=True)
    if orchestrator is None:
        raise RuntimeError("rlix.init returned None (expected orchestrator actor handle on rank 0)")

    CoordinatorActor = ray.remote(PipelineCoordinator)

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
        ray_namespace = get_pipeline_namespace(str(pipeline_id))
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
            runtime_env={"env_vars": {
                **pipeline_identity_env_vars(pipeline_id=str(pipeline_id), ray_namespace=ray_namespace),
                **thread_limit_env_vars(),
            }},
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
