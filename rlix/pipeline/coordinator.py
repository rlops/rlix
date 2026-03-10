from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

from rlix.protocol.coordinator import Coordinator
from rlix.protocol.request_id import validate_pipeline_id
from rlix.protocol.types import (
    ActionResponse,
    get_pipeline_namespace,
    PIPELINE_ACTOR_NAME_PREFIX,
    ProgressReport,
    SCHEDULER_ACTOR_NAME,
    RLIX_NAMESPACE,
)


def _build_pipeline_env_vars(*, pipeline_id: str, ray_namespace: str) -> Dict[str, str]:
    job_id = ray.get_runtime_context().get_job_id()
    scratch_root = f"/tmp/rlix/{pipeline_id}/{job_id}"
    shared_root = "/tmp/rlix/shared"

    # Ensure Ray worker processes can import both `rlix` (repo root) and `roll` (ROLL root)
    # even when started from non-repo working directories.
    this_file = Path(__file__).resolve()
    repo_root = str(this_file.parents[3])   # .../RLix
    roll_root = str(this_file.parents[2])   # .../RLix/external/ROLL_rlix
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [repo_root, roll_root]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    pythonpath = os.pathsep.join(pythonpath_parts)

    env_vars = {
        "PIPELINE_ID": pipeline_id,
        "ROLL_RAY_NAMESPACE": ray_namespace,
        "RLIX_CONTROL_PLANE": "rlix",
        "PYTHONPATH": pythonpath,
        # Shared weights/cache (big, reusable).
        "HF_HOME": f"{shared_root}/hf",
        "HUGGINGFACE_HUB_CACHE": f"{shared_root}/hf/hub",
        "TRANSFORMERS_CACHE": f"{shared_root}/hf/transformers",
        "HF_DATASETS_CACHE": f"{shared_root}/hf/datasets",
        # Job/pipeline-scoped scratch (write-hot / collision-prone).
        "HUGGINGFACE_AUTOMAP_CACHE": f"{scratch_root}/hf/automap",
        "VLLM_CACHE_ROOT": f"{scratch_root}/vllm",
        "FLASHINFER_WORKSPACE_DIR": f"{scratch_root}/flashinfer",
        # Limit thread counts to avoid hitting container pids.max.
        # Read from env so shell export overrides; defaults are safe minimums.
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        "RAY_grpc_server_thread_pool_size": os.environ.get("RAY_grpc_server_thread_pool_size", "4"),
    }
    import logging as _logging
    _logging.getLogger(__name__).info(
        "[_build_pipeline_env_vars] pid=%d pipeline_id=%s OMP_NUM_THREADS=%s RAY_grpc_server_thread_pool_size=%s",
        os.getpid(), pipeline_id,
        env_vars["OMP_NUM_THREADS"], env_vars["RAY_grpc_server_thread_pool_size"],
    )
    return env_vars


def _validate_cpu_only_reward(*, pipeline_config: Any) -> None:
    reward_cfg = getattr(pipeline_config, "reward", None)
    if reward_cfg is None:
        return
    device_mapping = getattr(reward_cfg, "device_mapping", None)
    if device_mapping is None:
        return
    if isinstance(device_mapping, list) and len(device_mapping) == 0:
        return
    if isinstance(device_mapping, str) and device_mapping.strip() in {"", "[]"}:
        return
    # TODO(ENG-123): lift this restriction to support GPU reward clusters.
    raise RuntimeError("ENG-123 Phase 3 only supports CPU-only reward (reward.device_mapping must be empty/None).")


def _validate_vllm_sleep_level(*, pipeline_config: Any) -> None:
    actor_infer = getattr(pipeline_config, "actor_infer", None)
    if actor_infer is None:
        return
    strategy_args = getattr(actor_infer, "strategy_args", None)
    if strategy_args is None:
        return
    strategy_name = getattr(strategy_args, "strategy_name", None)
    if strategy_name != "vllm":
        return
    strategy_config = getattr(strategy_args, "strategy_config", None) or {}
    sleep_level = strategy_config.get("sleep_level", 1)
    if int(sleep_level) != 2:
        raise RuntimeError("ENG-123 Phase 3 requires actor_infer vLLM sleep_level=2 (drop model weights on offload).")


def _validate_offload_nccl(*, pipeline_config: Any) -> None:
    """Enforce offload_nccl=True on all clusters when sleep_level=2 is active.

    sleep_level=2 is the RLix multi-pipeline mode where GPU VRAM is shared across
    co-tenant pipelines. NCCL communicator buffers (~400-500 MB per process) accumulate
    on the GPU even when a cluster is sleeping. With 10+ co-tenant processes this can
    consume 4-5 GB of baseline VRAM, preventing KV-cache wake-up.

    offload_nccl=True destroys process groups on offload and rebuilds them on load,
    which is the only way to reclaim that memory.
    """
    # Clusters present in an agentic pipeline config.
    cluster_names = ("actor_train", "actor_infer", "reference", "critic")
    bad_clusters = []
    for name in cluster_names:
        worker_config = getattr(pipeline_config, name, None)
        if worker_config is None:
            continue
        # Skip clusters that are inactive (no GPUs assigned — e.g. default critic).
        device_mapping = getattr(worker_config, "device_mapping", None)
        if not device_mapping:
            continue
        if not getattr(worker_config, "offload_nccl", False):
            bad_clusters.append(name)
    if bad_clusters:
        raise RuntimeError(
            f"ENG-123 sleep_level=2 requires offload_nccl=True on all clusters to reclaim NCCL "
            f"buffer VRAM between cycles. Missing on: {bad_clusters}. "
            f"Add 'offload_nccl: ${{offload_nccl}}' under each cluster in your pipeline YAML."
        )


# Default Ray actor concurrency for RlixCoordinator. 4 slots let progress reports
# (fire-and-forget, fast) run concurrently with resize_infer RPCs without blocking each other.
# _progress_lock guards the shared progress state under concurrent calls.
COORDINATOR_MAX_CONCURRENCY: int = 4


class RlixCoordinator(Coordinator):
    """Per-pipeline coordinator actor (ENG-123 Phase 3).

    Contract:
    - Aggregates per-scheduler progress reports from GroupQueueManager (train + val + all LoRAs)
      and forwards a single aggregated ProgressReport (mode="aggregated") to the rlix scheduler.
    - Exposes shrink/expand RPCs for the Rlix scheduler (fail-fast).
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        pipeline_config: Any,
    ):
        validate_pipeline_id(pipeline_id)
        self._pipeline_id = pipeline_id
        self._ray_namespace = get_pipeline_namespace(pipeline_id)
        self._pipeline_env_vars = _build_pipeline_env_vars(pipeline_id=pipeline_id, ray_namespace=self._ray_namespace)

        _validate_cpu_only_reward(pipeline_config=pipeline_config)
        _validate_vllm_sleep_level(pipeline_config=pipeline_config)
        _validate_offload_nccl(pipeline_config=pipeline_config)

        # Create the cluster-wide singleton ResourceManager actor before any pipeline actor.
        # The coordinator actor holds 0 GPU so the PG bundle ({GPU: N}) can always be satisfied.
        # The actor is a namespace singleton (rlix:roll_resource_manager) shared across
        # all concurrent pipeline actors.  We also capture node-0's placement group
        # and base GPU rank here to pin pipeline actors to a GPU node for CUDA visibility.
        from roll.distributed.scheduler.resource_manager import get_or_create_roll_resource_manager_actor
        self._rm_actor = get_or_create_roll_resource_manager_actor(pipeline_config.num_gpus_per_node)
        _rm_state = ray.get(self._rm_actor.get_state.remote())
        # Node 0's placement group is used to schedule the pipeline actor on a GPU node so
        # that Ray sets CUDA_VISIBLE_DEVICES (needed for platform detection + RNG state).
        self._rm_node0_pg = _rm_state["node2pg"].get(0)

        self._pipeline_actor = None
        # Serializes resize_infer and sync_lora_weights: prevents a weight sync from
        # racing with a concurrent shrink/expand triggered by the central scheduler.
        self._resize_sync_lock = threading.Lock()

        # Serializes concurrent report_progress_from_scheduler calls when max_concurrency > 1.
        # Separate from _resize_sync_lock — no shared state between progress and resize paths.
        self._progress_lock = threading.Lock()
        # Bookkeep latest snapshot per scheduler stream for aggregated progress reporting.
        # Key: "{mode}:{adapter_id or '__fft__'}". Invariant: mode+adapter_id is unique per GQM instance;
        # two GQMs sharing the same key is a misconfiguration (last-write-wins without error).
        self._scheduler_reports: Dict[str, ProgressReport] = {}
        self._coord_progress_last_bucket: Optional[int] = None
        # Resolve rlix scheduler handle for forwarding aggregated progress.
        try:
            self._rlix_scheduler = ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=RLIX_NAMESPACE)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve {SCHEDULER_ACTOR_NAME!r} in namespace {RLIX_NAMESPACE!r}. "
                "RlixCoordinator requires the central scheduler actor to exist at startup."
            ) from exc

        # Driver is responsible for:
        # - orchestrator.allocate_pipeline_id()
        # - orchestrator.register_pipeline(...)
        # - orchestrator.admit_pipeline(...)
        # before creating this coordinator actor.

    def create_pipeline_actor(self, *, pipeline_config: Any) -> Any:
        if self._pipeline_actor is not None:
            return self._pipeline_actor

        adapters = getattr(getattr(pipeline_config, "actor_train", None), "model_args", None)
        adapters = getattr(adapters, "adapters", None) if adapters is not None else None
        if adapters:
            from rlix.pipeline.multi_lora_pipeline import RlixMultiLoraPipeline
            PipelineClass = RlixMultiLoraPipeline
        else:
            from rlix.pipeline.full_finetune_pipeline import RlixFullFinetunePipeline
            PipelineClass = RlixFullFinetunePipeline

        PipelineActor = ray.remote(PipelineClass)
        # Safety: always inject env vars before constructing the pipeline actor, so callers can't
        # accidentally create a pipeline with missing system_envs.
        self._inject_pipeline_env_vars(pipeline_config=pipeline_config)

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        self._pipeline_actor = PipelineActor.options(
            name=f"{PIPELINE_ACTOR_NAME_PREFIX}{self._pipeline_id}",
            namespace=self._ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            # Critical: allow resize RPCs to run while `run()` is in-flight.
            # Keep this small: Ray uses a thread pool for sync actors; huge values can hit thread limits.
            max_concurrency=32,
            runtime_env={"env_vars": self._pipeline_env_vars},
            # Schedule pipeline actor inside node-0's placement group bundle so that Ray
            # sets CUDA_VISIBLE_DEVICES correctly (needed for checkpoint RNG state saving).
            # num_gpus=0.01: drawn from the bundle's GPU pool (not the global pool), so
            # the singleton RM can still hold all integer GPUs in its placement group.
            num_gpus=0.01,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self._rm_node0_pg,
            ),
        ).remote(pipeline_id=self._pipeline_id, pipeline_config=pipeline_config)
        # Do not block pipeline actor creation on initialize_pipeline.
        # Initialization is executed lazily by pipeline.run() via _ensure_initialized(),
        # allowing multi-pipeline startup/admission to proceed concurrently.
        return self._pipeline_actor

    def report_progress_from_scheduler(self, report: ProgressReport) -> None:
        """Aggregate per-scheduler progress and forward to the rlix scheduler.

        Called by each GroupQueueManager (train / val / all LoRAs) at 2% cadence.
        Aggregates all streams for this pipeline and emits one report to the central
        rlix scheduler at 2% cadence of the aggregate.
        """
        metrics = report.metrics if isinstance(report.metrics, dict) else {}
        mode = str(metrics.get("mode", "train"))
        adapter_id = metrics.get("adapter_id")
        scheduler_key = f"{mode}:{adapter_id if adapter_id is not None else '__fft__'}"

        with self._progress_lock:
            # Last-write-wins: same key from a newer step overwrites the stale entry naturally.
            self._scheduler_reports[scheduler_key] = report

            # Aggregate remaining/required across all known scheduler streams.
            total_required = sum(r.step_target_trajectories for r in self._scheduler_reports.values())
            total_remaining = sum(
                float(r.metrics.get("remaining", 0)) if isinstance(r.metrics, dict) else 0.0
                for r in self._scheduler_reports.values()
            )
            if total_required <= 0:
                return

            total_collected = max(total_required - total_remaining, 0.0)
            percent_completed = total_collected / float(total_required)
            bucket = math.floor(percent_completed * 50)

            is_new_batch = bool(metrics.get("new_batch", False))
            # new_batch forces an immediate emit; otherwise only emit when the 2% bucket changes.
            if bucket == self._coord_progress_last_bucket and not is_new_batch:
                return
            self._coord_progress_last_bucket = bucket

            # Use the oldest unfinished timestamp across all streams for an accurate FIFO signal.
            oldest_ts: Optional[float] = min(
                (r.oldest_unfinished_creation_ts for r in self._scheduler_reports.values()
                 if r.oldest_unfinished_creation_ts is not None),
                default=None,
            )

        aggregated = ProgressReport(
            pipeline_id=str(self._pipeline_id),
            queued_trajectories=0,
            inflight_trajectories=0,
            step_target_trajectories=int(total_required),
            percent_completed=percent_completed,
            oldest_unfinished_creation_ts=oldest_ts,
            fifo_timestamp=time.time(),
            metrics={
                "mode": "aggregated",
                "remaining": int(total_remaining),
                "bucket": int(bucket),
                "new_batch": is_new_batch,
            },
        )
        # Fire-and-forget: progress is a background signal; same pattern as existing direct reports.
        self._rlix_scheduler.report_progress.remote(aggregated)

    def _inject_pipeline_env_vars(self, *, pipeline_config: Any) -> None:
        envs = dict(self._pipeline_env_vars)

        def _update_system_envs(obj: Any) -> None:
            if obj is None:
                return
            system_envs = getattr(obj, "system_envs", None)
            if system_envs is None:
                setattr(obj, "system_envs", dict(envs))
                return
            if not isinstance(system_envs, dict):
                raise RuntimeError(f"Expected system_envs to be dict, got {type(system_envs).__name__}")
            system_envs.update(envs)

        # Worker clusters
        _update_system_envs(getattr(pipeline_config, "actor_train", None))
        _update_system_envs(getattr(pipeline_config, "actor_infer", None))
        _update_system_envs(getattr(pipeline_config, "reference", None))
        _update_system_envs(getattr(pipeline_config, "critic", None))
        _update_system_envs(getattr(pipeline_config, "reward", None))

        # Env managers (spawn env actors/workers)
        _update_system_envs(getattr(pipeline_config, "train_env_manager", None))
        _update_system_envs(getattr(pipeline_config, "val_env_manager", None))

    def sync_lora_weights(self, *, loras_to_sync: List[str]) -> None:
        """Push trained LoRA weights to currently-awake infer workers.

        Ranks are queried INSIDE _resize_sync_lock by looking up the generate_scheduler
        actor directly, so the set cannot change between query and use (resize_infer also
        acquires this lock before shrinking/expanding).
        If all infer workers are sleeping (preempted by concurrent pipelines), sync is
        skipped — sleeping workers receive the updated LoRA via expand_worker on wake.
        """
        with self._resize_sync_lock:
            # Look up generate_scheduler by its well-known name and query ranks atomically.
            from roll.utils.constants import RAY_NAMESPACE
            generate_scheduler = ray.get_actor(
                f"RequestScheduler-{self._pipeline_id}", namespace=RAY_NAMESPACE
            )
            active_ranks = sorted(ray.get(generate_scheduler.get_active_dp_ranks.remote()))
            if not active_ranks:
                # All infer workers preempted/sleeping; expand_worker syncs on next wake.
                return
            model_update_service_name = f"{self._pipeline_id}_model_update_service"
            try:
                model_update_service = ray.get_actor(
                    model_update_service_name, namespace=self._ray_namespace
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resolve ModelUpdateService {model_update_service_name!r} "
                    f"in namespace {self._ray_namespace!r}"
                ) from e
            ray.get(model_update_service.sync_selected_workers.remote(
                active_ranks, loras_to_sync=list(loras_to_sync)
            ))

    def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        """Pipeline-scoped resize for actor_infer (ENG-123).

        Serialized with sync_lora_weights via _resize_sync_lock.

        Contract: exactly one of {dp_ranks_to_remove, dp_ranks_to_add} must be non-empty.
        Applies to both train+val RequestSchedulers (shared infer cluster):
        - Shrink: train offloads; val routing-only (skip_offload=True).
        - Expand: train loads + optional selective update; val routing-only (skip_load=True).

        NOTE: This intentionally does NOT call suspend()/resume() globally. Upstream RequestScheduler.shrink_workers()
        removes shrinking ranks from active_dp_ranks under routing_lock and aborts/drains only impacted ranks; new
        requests continue on remaining ranks. Shrink-to-zero and expand-from-zero are handled internally via
        need_suspend/resume().
        """
        if not isinstance(dp_ranks_to_remove, list):
            raise ValueError("dp_ranks_to_remove must be list[int]")
        if not isinstance(dp_ranks_to_add, list):
            raise ValueError("dp_ranks_to_add must be list[int]")
        if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
            raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")

        with self._resize_sync_lock:
            # NOTE: coordinator does not coordinate train/val request schedulers directly; it delegates to the
            # per-pipeline pipeline actor (single serialization boundary owned by pipeline runtime).
            resize_actor_name = f"{PIPELINE_ACTOR_NAME_PREFIX}{self._pipeline_id}"
            try:
                resize_actor = ray.get_actor(resize_actor_name, namespace=self._ray_namespace)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resolve pipeline actor {resize_actor_name!r} in namespace {self._ray_namespace!r} "
                    f"for pipeline_id={self._pipeline_id!r}"
                ) from e

            ref = resize_actor.resize_infer.remote(
                dp_ranks_to_remove=list(dp_ranks_to_remove),
                dp_ranks_to_add=list(dp_ranks_to_add),
            )
            ray.get(ref)
        return ActionResponse(success=True)