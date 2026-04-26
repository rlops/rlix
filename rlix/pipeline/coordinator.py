from __future__ import annotations

import copy
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, TypeVar

import ray

from rlix.pipeline.utils import validate_resize_params
from rlix.protocol.coordinator import Coordinator
from rlix.protocol.types import (
    GPU_CLUSTER_NAMES,
    PIPELINE_ACTOR_NAME_PREFIX,
    RLIX_NAMESPACE,
    SCHEDULER_ACTOR_NAME,
    ActionResponse,
    ProgressReport,
    get_pipeline_namespace,
)
from rlix.protocol.validation import validate_pipeline_id
from rlix.utils.env import parse_env_timeout_s, pipeline_identity_env_vars, thread_limit_env_vars
from rlix.utils.ray import get_actor_or_raise

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

# Max concurrent RPCs on the pipeline actor (resize + run can overlap).
# Keep small: Ray uses a thread pool for sync actors; huge values hit thread limits.
_PIPELINE_ACTOR_MAX_CONCURRENCY: int = 32

# Timeout for acquiring _resize_sync_lock in resize_infer (scheduler path).
# Limits how long the scheduler is stalled when sync_lora_weights holds the lock
# during an NCCL weight sync. Default 180s covers the ModelUpdateService default
# timeout (150s) plus headroom; override via env var for tighter SLOs.
_RESIZE_LOCK_TIMEOUT_S: Optional[float] = parse_env_timeout_s("RLIX_RESIZE_LOCK_TIMEOUT_S", default_s=180.0)


def _build_pipeline_env_vars(*, pipeline_id: str, ray_namespace: str) -> Dict[str, str]:
    """Build environment variables injected into every worker actor for a pipeline.

    Sets HF/vLLM cache paths (shared root for weights, per-job scratch for collisions),
    thread-count limits to stay under container pids.max, and pipeline identity vars.
    """
    job_id = ray.get_runtime_context().get_job_id()
    scratch_root = f"/tmp/rlix/{pipeline_id}/{job_id}"
    shared_root = "/tmp/rlix/shared"

    # NOTE: Requires pip-installed rlix and roll packages.
    # Ray workers inherit the driver's Python environment, so packages
    # installed via pip are automatically available without PYTHONPATH.

    env_vars = {
        **pipeline_identity_env_vars(pipeline_id=pipeline_id, ray_namespace=ray_namespace),
        **thread_limit_env_vars(),
        # --- Shared weights/cache (big, reusable across pipelines) ---
        "HF_HOME": f"{shared_root}/hf",
        "HF_HUB_CACHE": f"{shared_root}/hf/hub",
        "HUGGINGFACE_HUB_CACHE": f"{shared_root}/hf/hub",
        "TRANSFORMERS_CACHE": f"{shared_root}/hf/transformers",
        "HF_DATASETS_CACHE": f"{shared_root}/hf/datasets",
        # --- Job/pipeline-scoped scratch (write-hot / collision-prone) ---
        "HUGGINGFACE_AUTOMAP_CACHE": f"{scratch_root}/hf/automap",
        "VLLM_CACHE_ROOT": f"{scratch_root}/vllm",
        "FLASHINFER_WORKSPACE_DIR": f"{scratch_root}/flashinfer",
    }
    logger.info(
        "[_build_pipeline_env_vars] pid=%d pipeline_id=%s OMP_NUM_THREADS=%s RAY_num_server_call_thread=%s",
        os.getpid(),
        pipeline_id,
        env_vars["OMP_NUM_THREADS"],
        env_vars["RAY_num_server_call_thread"],
    )
    return env_vars


def _validate_config_schema(*, pipeline_config: Any) -> None:
    """Validate that pipeline_config has all required attributes.

    Raises ValueError if any required attribute is missing.
    This prevents silent failures when config keys are renamed or missing.
    """
    required_attrs = ["actor_train", "actor_infer"]
    for attr in required_attrs:
        if not hasattr(pipeline_config, attr):
            raise ValueError(f"pipeline_config missing required attribute: {attr!r}")


def _validate_cpu_only_reward(*, pipeline_config: Any) -> None:
    """Reject pipeline configs that assign GPUs to the reward cluster.

    RLix currently supports CPU-only reward workers; GPU reward clusters
    would require scheduler-managed allocation that is not yet implemented.
    """
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
    # TODO: lift this restriction to support GPU reward clusters.
    raise RuntimeError("reward cluster only supports CPU-only mode (reward.device_mapping must be empty/None).")


def _validate_vllm_sleep_level(*, pipeline_config: Any) -> None:
    """Require vLLM sleep_level=2 for multi-pipeline GPU time-sharing.

    sleep_level=2 drops model weights on offload, freeing VRAM for co-tenant
    pipelines. Lower levels retain weights and prevent effective sharing.
    """
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
    sleep_level = strategy_config.get("sleep_level", None)
    if sleep_level is None:
        strategy_config["sleep_level"] = 2
    elif int(sleep_level) != 2:
        raise RuntimeError("actor_infer vLLM sleep_level=2 required (drop model weights on offload).")


def _validate_offload_nccl(*, pipeline_config: Any) -> None:
    """Enforce offload_nccl=True on all GPU-active clusters.

    NCCL communicator buffers accumulate on the GPU even when a cluster is sleeping.
    With many co-tenant processes this can consume significant baseline VRAM,
    preventing KV-cache wake-up.

    offload_nccl=True destroys process groups on offload and rebuilds them on load,
    which is the only way to reclaim that memory.
    """
    cluster_names = GPU_CLUSTER_NAMES
    bad_clusters = []
    for name in cluster_names:
        worker_config = getattr(pipeline_config, name, None)
        if worker_config is None:
            continue
        # Skip clusters that are inactive (no GPUs assigned — e.g. reward/env clusters).
        device_mapping = getattr(worker_config, "device_mapping", None)
        if not device_mapping:
            continue
        # DeepSpeed strategies manage their own process groups and are incompatible with
        # ROLL's ReloadableProcessGroup monkey-patch. Skip enforcement for deepspeed clusters.
        strategy_name = getattr(getattr(worker_config, "strategy_args", None), "strategy_name", "")
        if strategy_name.startswith("deepspeed"):
            continue
        offload_nccl = getattr(worker_config, "offload_nccl", None)
        if offload_nccl is None:
            worker_config.offload_nccl = True
        elif not offload_nccl:
            bad_clusters.append(name)
    if bad_clusters:
        raise RuntimeError(
            f"offload_nccl=True is required on all GPU-active clusters to reclaim NCCL "
            f"buffer VRAM between cycles. Explicitly set to False on: {bad_clusters}."
        )


# Default Ray actor concurrency for PipelineCoordinator. 4 slots let progress reports
# (fire-and-forget, fast) run concurrently with resize_infer RPCs without blocking each other.
# _progress_lock guards the shared progress state under concurrent calls.
COORDINATOR_MAX_CONCURRENCY: int = 4


class PipelineCoordinator(Coordinator):
    """Per-pipeline coordinator actor.

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
        # Precondition: the driver must have already called
        # orchestrator.allocate_pipeline_id(), register_pipeline(), and
        # admit_pipeline() before creating this coordinator actor.
        validate_pipeline_id(pipeline_id)
        self._pipeline_id = pipeline_id
        self._ray_namespace = get_pipeline_namespace(pipeline_id)
        self._pipeline_env_vars = _build_pipeline_env_vars(pipeline_id=pipeline_id, ray_namespace=self._ray_namespace)

        _validate_config_schema(pipeline_config=pipeline_config)
        _validate_cpu_only_reward(pipeline_config=pipeline_config)
        _validate_vllm_sleep_level(pipeline_config=pipeline_config)
        _validate_offload_nccl(pipeline_config=pipeline_config)

        # Config flag for post-sync weight verification (disabled by default).
        self._verify_model_after_sync: bool = bool(pipeline_config.verify_model_after_sync)

        # Singleton ResourceManager (rlix:roll_resource_manager) shared across all pipelines.
        # Created before any pipeline actor so placement groups are ready.
        from roll.distributed.scheduler.resource_manager import RollResourceManagerProxy

        self._resource_manager_proxy = RollResourceManagerProxy(num_gpus_per_node=pipeline_config.num_gpus_per_node)
        # Pin pipeline actor to node-0's placement group so Ray sets
        # CUDA_VISIBLE_DEVICES (needed for platform detection + checkpoint RNG state).
        # The actor requests num_gpus=0.01 from the PG's bundle.
        self._resource_manager_node0_pg = self._resource_manager_proxy.node2pg.get(0)

        self._pipeline_actor = None
        # Lazily resolved on first sync_lora_weights call; created by the pipeline actor during init.
        self._model_update_service = None
        # Serializes resize_infer and sync_lora_weights: prevents a weight sync from
        # racing with a concurrent shrink/expand triggered by the central scheduler.
        self._resize_sync_lock = threading.Lock()
        # Bookkeep active infer dp ranks locally; updated atomically under _resize_sync_lock
        # by resize_infer (shrink/expand). Used by sync_lora_weights to avoid remote actor lookup.
        self._active_infer_dp_ranks: Set[int] = set()

        # Serializes concurrent report_progress_from_scheduler calls when max_concurrency > 1.
        # Separate from _resize_sync_lock — no shared state between progress and resize paths.
        self._progress_lock = threading.Lock()
        # Bookkeep latest snapshot per scheduler stream for aggregated progress reporting.
        # Key: "{mode}:{adapter_id or '__fft__'}". Invariant: mode+adapter_id is unique per GQM instance;
        # two GQMs sharing the same key is a misconfiguration (last-write-wins without error).
        self._scheduler_reports: Dict[str, ProgressReport] = {}
        # Last progress bucket (0–50, 2% granularity) sent to the scheduler.
        # Used to deduplicate: skip the RPC if the bucket hasn't changed.
        self._coord_progress_last_bucket: Optional[int] = None
        # Resolve rlix scheduler handle for forwarding aggregated progress.
        self._rlix_scheduler = get_actor_or_raise(
            SCHEDULER_ACTOR_NAME,
            RLIX_NAMESPACE,
            error_context="PipelineCoordinator requires the central scheduler actor to exist at startup.",
        )

    def create_pipeline_actor(self, *, pipeline_config: Any) -> Any:
        """Create and return the per-pipeline Ray actor (full-finetune or multi-LoRA).

        Selects the pipeline class based on whether adapters are configured,
        injects env vars, and schedules the actor on node-0's placement group.
        Returns the existing actor handle if already created (idempotent).
        """
        if self._pipeline_actor is not None:
            return self._pipeline_actor

        pipeline_cls_path = getattr(pipeline_config, "pipeline_cls", None)
        if not pipeline_cls_path:
            raise RuntimeError(
                "pipeline_cls is required in the pipeline YAML config. "
                "Example: 'pipeline_cls: rlix.pipeline.full_finetune_pipeline.RollFullFinetunePipeline'"
            )
        import importlib

        module_path, class_name = pipeline_cls_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"pipeline_cls module '{module_path}' not found: {exc}") from exc
        pipeline_class = getattr(module, class_name, None)
        if pipeline_class is None:
            raise RuntimeError(f"pipeline_cls class '{class_name}' not found in module '{module_path}'.")

        PipelineActor = ray.remote(pipeline_class)
        # Safety: always inject env vars before constructing the pipeline actor, so callers can't
        # accidentally create a pipeline with missing system_envs.
        # Deep copy to prevent env var leaks when same config object is reused across pipelines.
        pipeline_config = self._inject_pipeline_env_vars(pipeline_config=pipeline_config)

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        self._pipeline_actor = PipelineActor.options(
            name=f"{PIPELINE_ACTOR_NAME_PREFIX}{self._pipeline_id}",
            namespace=self._ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            max_concurrency=_PIPELINE_ACTOR_MAX_CONCURRENCY,
            runtime_env={"env_vars": self._pipeline_env_vars},
            # Schedule inside node-0's placement group so Ray sets CUDA_VISIBLE_DEVICES
            # (needed for checkpoint RNG state saving). num_gpus=0.01 is drawn from the
            # placement group's bundle, not the global pool — otherwise the ResourceManager
            # couldn't reserve all integer GPU slots in its placement group.
            num_gpus=0.01,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self._resource_manager_node0_pg,
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
        if "collected" not in metrics:
            raise ValueError(
                f"ProgressReport from pipeline {report.pipeline_id!r} missing required 'collected' metric"
            )
        if "remaining" in metrics:
            raise ValueError(
                f"ProgressReport from pipeline {report.pipeline_id!r} contains wire-level 'remaining'; "
                "send 'collected' instead (hard cutover)"
            )
        mode = str(metrics["mode"])
        adapter_id = metrics.get("adapter_id")
        scheduler_key = f"{mode}:{adapter_id if adapter_id is not None else '__fft__'}"
        is_new_batch = bool(metrics["new_batch"])

        with self._progress_lock:
            # Last-write-wins: same key from a newer step overwrites the stale entry naturally.
            self._scheduler_reports[scheduler_key] = report
            self._aggregate_and_emit(force=is_new_batch)

    def clear_progress_stream(self, *, mode: str, adapter_id: Optional[str]) -> None:
        """Remove a scheduler stream from progress aggregation.

        Called by GroupQueueManager.end_progress_batch() after get_batch() returns
        to indicate this stream no longer contributes demand. If other streams
        remain, re-aggregates and force-emits so the scheduler immediately sees
        reduced demand. If all streams are gone, clears the scheduler's stored
        progress entry entirely.

        Args:
            mode: Stream mode ("train" or "val").
            adapter_id: LoRA adapter ID, or None for full-finetune.
        """
        scheduler_key = f"{mode}:{adapter_id if adapter_id is not None else '__fft__'}"

        with self._progress_lock:
            removed = self._scheduler_reports.pop(scheduler_key, None)
            if removed is None:
                return  # Already cleared or never reported; idempotent.

            if self._scheduler_reports:
                # Other streams still active: re-aggregate and force-emit so
                # the scheduler sees reduced demand immediately (not on next
                # bucket transition from a remaining stream).
                self._aggregate_and_emit(force=True)
            else:
                # All streams gone: reset emission state and tell the scheduler
                # to drop this pipeline's progress entry entirely.
                self._coord_progress_last_bucket = None
                self._rlix_scheduler.clear_progress.remote(
                    pipeline_id=self._pipeline_id,
                )

    def _aggregate_and_emit(self, *, force: bool) -> None:
        """Recompute aggregate progress from _scheduler_reports and emit to scheduler.

        Sums clamped completion across all active streams. Each stream's completed
        count is derived locally as min(collected, step_target). Retired streams are
        removed by clear_progress_stream() so only live demand is counted.

        Must be called with _progress_lock held.

        Args:
            force: If True, emit regardless of bucket throttling (lifecycle edge).
        """
        if not self._scheduler_reports:
            return

        # Sum clamped completion across all active streams.
        # Per-stream clamping: completed = min(collected, step_target).
        total_required = 0.0
        total_completed = 0.0
        total_collected = 0.0
        for rpt in self._scheduler_reports.values():
            rpt_metrics = rpt.metrics if isinstance(rpt.metrics, dict) else {}
            step_target = float(max(int(rpt.step_target_trajectories), 1))
            collected_raw = max(0.0, float(rpt_metrics["collected"]))
            completed_clamped = min(collected_raw, step_target)
            total_required += step_target
            total_completed += completed_clamped
            total_collected += collected_raw
        if total_required <= 0:
            return

        percent_completed = min(total_completed / float(total_required), 1.0) if total_required > 0 else 0.0
        bucket = math.floor(percent_completed * 50)

        if not force and bucket == self._coord_progress_last_bucket:
            return
        self._coord_progress_last_bucket = bucket

        aggregated = ProgressReport(
            pipeline_id=str(self._pipeline_id),
            step_target_trajectories=int(total_required),
            fifo_timestamp=time.time(),
            metrics={
                "mode": "aggregated",
                "collected": int(total_collected),
                "completed": int(total_completed),
                "bucket": int(bucket),
                "new_batch": force,
            },
        )
        # Fire-and-forget: progress is a background signal; same pattern as existing direct reports.
        self._rlix_scheduler.report_progress.remote(aggregated)

    def _inject_pipeline_env_vars(self, *, pipeline_config: _T) -> _T:
        """Deep copy pipeline_config and inject env vars into the copy.

        Returns a modified deep copy to prevent env var leaks when the same config
        object is reused across multiple pipelines.
        """
        config = copy.deepcopy(pipeline_config)
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

        # All clusters that spawn Ray actors need pipeline env vars:
        # GPU clusters + CPU-only clusters (reward, env managers).
        all_cluster_names = GPU_CLUSTER_NAMES + ("reward", "train_env_manager", "val_env_manager")
        for cluster_name in all_cluster_names:
            _update_system_envs(getattr(config, cluster_name, None))

        return config

    def sync_lora_weights(self, *, loras_to_sync: List[str]) -> None:
        """Push trained LoRA weights to currently-awake infer workers.

        Active ranks come from local _active_infer_dp_ranks bookkeeping (updated by
        resize_infer under the same _resize_sync_lock), so the set cannot change between
        query and use.
        If all infer workers are sleeping (preempted by concurrent pipelines), sync is
        skipped — sleeping workers receive the updated LoRA via expand_worker on wake.

        Design tradeoff — lock held across NCCL broadcast (G11-RULE-48.2):
        _resize_sync_lock is intentionally held for the entire sync_selected_workers RPC
        (NCCL all-gather, seconds to minutes). This serializes weight sync with resize_infer
        so the active_dp_ranks set cannot change mid-broadcast (which would corrupt the
        collective). The cost is that scheduler-initiated shrink/expand is blocked during
        the broadcast window. Mitigations:
          - resize_infer acquires the lock with _RESIZE_LOCK_TIMEOUT_S (default 180s),
            so the scheduler is never blocked indefinitely.
          - sync only targets already-active ranks; if all workers are sleeping, the lock
            is released immediately (early return below).
          - On NCCL hang, the lock timeout on the resize_infer side surfaces the failure
            as a RuntimeError rather than a silent deadlock.
        Alternatives considered and rejected:
          - Releasing the lock before sync and re-checking ranks after: risks sending
            weights to a worker that was shrunk mid-broadcast (NCCL rank mismatch → crash).
          - Adding a per-sync timeout: NCCL collectives are not cancellable; a timeout
            would leave the communicator in a broken state.
        """
        acquired = self._resize_sync_lock.acquire(
            timeout=_RESIZE_LOCK_TIMEOUT_S if _RESIZE_LOCK_TIMEOUT_S is not None else -1
        )
        if not acquired:
            raise RuntimeError(
                f"sync_lora_weights timed out waiting for _resize_sync_lock after {_RESIZE_LOCK_TIMEOUT_S}s "
                f"(likely blocked by a long-running resize_infer). "
                f"pipeline_id={self._pipeline_id!r}"
            )
        try:
            # Use locally bookkept active dp ranks (updated by resize_infer under same lock).
            active_ranks = sorted(self._active_infer_dp_ranks)
            if not active_ranks:
                # All infer workers preempted/sleeping; expand_worker syncs on next wake.
                return
            # Created by the pipeline actor during init; lazy-resolve here.
            if self._model_update_service is None:
                model_update_service_name = f"{self._pipeline_id}_model_update_service"
                self._model_update_service = get_actor_or_raise(
                    model_update_service_name,
                    self._ray_namespace,
                    error_context=f"ModelUpdateService required for pipeline_id={self._pipeline_id!r}.",
                )
            model_update_service = self._model_update_service
            assert model_update_service is not None
            ray.get(
                model_update_service.sync_selected_workers.remote(
                    active_ranks,
                    adapters_to_sync=list(loras_to_sync),
                    verify=self._verify_model_after_sync,
                )
            )
        finally:
            self._resize_sync_lock.release()

    def sync_base_weights_to_active(self) -> List[int]:
        """Push trained base model weights to all currently-awake infer workers.

        Called by the pipeline after train_step + promote + offload, before releasing
        actor_train GPUs.  Mirrors sync_lora_weights() but syncs the full base model
        (adapters_to_sync=None) instead of only LoRA adapters.

        Same lock/skip semantics as sync_lora_weights: holds _resize_sync_lock for the
        entire NCCL broadcast to prevent active_dp_ranks from changing mid-flight.
        If all infer workers are sleeping (active_infer_dp_ranks is empty), sync is
        skipped — sleeping workers receive the base weights via expand_worker on wake.
        """
        acquired = self._resize_sync_lock.acquire(
            timeout=_RESIZE_LOCK_TIMEOUT_S if _RESIZE_LOCK_TIMEOUT_S is not None else -1
        )
        if not acquired:
            raise RuntimeError(
                f"sync_base_weights_to_active timed out waiting for _resize_sync_lock after {_RESIZE_LOCK_TIMEOUT_S}s "
                f"(likely blocked by a long-running resize_infer). "
                f"pipeline_id={self._pipeline_id!r}"
            )
        try:
            active_ranks = sorted(self._active_infer_dp_ranks)
            if not active_ranks:
                return []
            if self._model_update_service is None:
                model_update_service_name = f"{self._pipeline_id}_model_update_service"
                self._model_update_service = get_actor_or_raise(
                    model_update_service_name,
                    self._ray_namespace,
                    error_context=f"ModelUpdateService required for pipeline_id={self._pipeline_id!r}.",
                )
            model_update_service = self._model_update_service
            assert model_update_service is not None
            ray.get(
                model_update_service.sync_selected_workers.remote(
                    active_ranks,
                    adapters_to_sync=None,
                    verify=self._verify_model_after_sync,
                )
            )
            return active_ranks
        finally:
            self._resize_sync_lock.release()

    def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        """Pipeline-scoped resize for actor_infer.

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
        validate_resize_params(dp_ranks_to_remove, dp_ranks_to_add)

        acquired = self._resize_sync_lock.acquire(
            timeout=_RESIZE_LOCK_TIMEOUT_S if _RESIZE_LOCK_TIMEOUT_S is not None else -1
        )
        if not acquired:
            raise RuntimeError(
                f"resize_infer timed out waiting for _resize_sync_lock after {_RESIZE_LOCK_TIMEOUT_S}s "
                f"(likely blocked by a long-running sync_lora_weights NCCL sync). "
                f"pipeline_id={self._pipeline_id!r}"
            )
        try:
            # NOTE: coordinator does not coordinate train/val request schedulers directly; it delegates to the
            # per-pipeline pipeline actor (single serialization boundary owned by pipeline runtime).
            if self._pipeline_actor is None:
                raise RuntimeError(
                    f"Pipeline actor not created yet for resize_infer, pipeline_id={self._pipeline_id!r}."
                )
            ref = self._pipeline_actor.resize_infer.remote(
                dp_ranks_to_remove=list(dp_ranks_to_remove),
                dp_ranks_to_add=list(dp_ranks_to_add),
            )
            ray.get(ref)
            # Update active dp ranks bookkeeping after successful resize.
            if dp_ranks_to_remove:
                self._active_infer_dp_ranks -= set(dp_ranks_to_remove)
            else:
                self._active_infer_dp_ranks |= set(dp_ranks_to_add)
        finally:
            self._resize_sync_lock.release()
        return ActionResponse(success=True)
