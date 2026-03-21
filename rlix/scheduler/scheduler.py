"""Rlix Scheduler.

Operational policy: fail-fast only. No recovery or rehydration is provided; on any
scheduler restart, pipelines are expected to re-register and be re-admitted.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import ray

from rlix.protocol.types import (
    COORDINATOR_ACTOR_NAME_PREFIX,
    GENERATION_CLUSTER_NAME,
    ORCHESTRATOR_ACTOR_NAME,
    REWARD_CLUSTER_NAME,
    RLIX_NAMESPACE,
    Priority,
    ProgressReport,
    get_pipeline_namespace,
)
from rlix.protocol.validation import validate_pipeline_id
from rlix.scheduler import planner
from rlix.scheduler.state import SchedulerState
from rlix.scheduler.tracer import GPUTraceInfo, SchedulerTracer
from rlix.scheduler.types import (
    ClusterAllocation,
    ExecutionPlan,
    PendingPlannedReleaseRequest,
    PendingRequest,
    Request,
    SchedGuidedShrinkOp,
    SignalPendingAllocationOp,
    build_dp_rank_mapping,
    is_generation_cluster,
    parse_cluster_id,
    validate_cluster_id,
)
from rlix.scheduler.validation import ValidationInputs, validate_execution_plan
from rlix.utils.ray import get_actor_or_raise

logger = logging.getLogger(__name__)

_TOPOLOGY_READY_TIMEOUT_S: float = float(os.environ.get("RLIX_TOPOLOGY_READY_TIMEOUT_S", "120"))
_FAIL_FAST_SHUTDOWN_TIMEOUT_S: float = float(os.environ.get("RLIX_FAIL_FAST_SHUTDOWN_TIMEOUT_S", "5"))

# Progress reporting: sentinel stream key for full-finetune pipelines (no adapter_id).
# LoRA pipelines use adapter_id as stream key; full-finetune uses this reserved sentinel.
_FULL_FINETUNE_STREAM_KEY: str = "__full_finetune__"


def _validate_and_canonicalize_device_mapping(
    *,
    cluster_name: str,
    tp_size: int,
    device_mapping: List[int],
    required_gpus_per_node: int,
) -> List[int]:
    """Validate + canonicalize device_mapping.

    Canonical form: sorted GPU ids.

    Topology contract:
    - For tp_size in {2,4,8}: each TP group must be contiguous and within a single node boundary.
    - For tp_size that is a multiple of 8: each TP group must be contiguous and aligned to node boundaries,
      spanning whole nodes (required_gpus_per_node GPUs per node assumed by global GPU id layout).
    """
    if not device_mapping:
        return []
    canonical = sorted(int(x) for x in device_mapping)

    if tp_size <= 0:
        raise ValueError(f"tp_size must be > 0 for cluster {cluster_name!r}, got {tp_size!r}")
    if tp_size == 1:
        return canonical

    if required_gpus_per_node <= 0:
        raise ValueError(f"required_gpus_per_node must be > 0, got {required_gpus_per_node!r}")
    if tp_size not in (2, 4, 8) and (tp_size % required_gpus_per_node != 0):
        raise ValueError(
            f"Invalid tp_size={tp_size} for cluster {cluster_name!r}: expected 1,2,4,8 or a multiple of {required_gpus_per_node}"
        )
    if len(canonical) % tp_size != 0:
        raise ValueError(
            f"cluster {cluster_name!r} has len(device_mapping)={len(canonical)} not divisible by tp_size={tp_size}"
        )

    for i in range(0, len(canonical), tp_size):
        group = canonical[i : i + tp_size]
        if not group:
            continue
        expected = list(range(group[0], group[0] + tp_size))
        if group != expected:
            raise ValueError(
                f"Non-contiguous TP group for cluster {cluster_name!r}: got {group}, expected contiguous {expected}"
            )
        if tp_size <= required_gpus_per_node:
            start_node = group[0] // required_gpus_per_node
            end_node = group[-1] // required_gpus_per_node
            if start_node != end_node:
                raise ValueError(
                    f"TP group crosses node boundary for cluster {cluster_name!r}: group={group} "
                    f"(gpus_per_node={required_gpus_per_node})"
                )
        else:
            if group[0] % required_gpus_per_node != 0 or group[-1] % required_gpus_per_node != (
                required_gpus_per_node - 1
            ):
                raise ValueError(
                    f"TP group must align to node boundaries for cluster {cluster_name!r}: group={group} "
                    f"(gpus_per_node={required_gpus_per_node})"
                )
    return canonical


@dataclass(slots=True)
class SchedulerImpl:
    """Priority-based GPU scheduler for concurrent RL pipelines.

    Manages a central scheduling loop that allocates and reclaims GPUs across
    registered pipelines.  Non-generation requests (training, ref-log-probs, etc.)
    use fixed device mappings and preempt generation workers when necessary.
    Generation allocation uses a gap-ratio algorithm that distributes DP workers
    proportionally to each pipeline's remaining demand.

    Lifecycle::

        initialize  ->  register_pipeline  ->  admit_pipeline
                                                    |
                        request_gpus / notify_release_gpus / notify_release_then_request_gpus
                                                    |
                        unregister_pipeline  ->  shutdown

    Thread-safety: all mutable state is guarded by ``_lock``.  The central
    scheduling loop and public APIs acquire the lock independently; coordinator
    resize RPCs are executed *outside* the lock to avoid deadlocks.
    """

    _state: SchedulerState = field(init=False)
    _lock: asyncio.Lock = field(init=False)
    _wakeup_event: asyncio.Event = field(init=False)
    _topology_ready: asyncio.Event = field(init=False)
    _loop_task: Optional[asyncio.Task] = field(init=False)
    _resource_manager: Any = field(init=False)
    _cycle_counter: int = field(init=False)
    _request_seq: int = field(init=False)
    _num_gpus: Optional[int] = field(init=False)
    _required_gpus_per_node: Optional[int] = field(init=False)
    _coordinator_handle_cache: Dict[str, Tuple[str, Any]] = field(init=False)
    _tracer: SchedulerTracer = field(init=False)

    def __post_init__(self) -> None:
        """Initialize mutable internal state.

        Topology fields (``_num_gpus``, ``_required_gpus_per_node``) remain ``None``
        until ``initialize()`` queries the ResourceManager.
        """
        self._state = SchedulerState()
        self._lock = asyncio.Lock()
        self._wakeup_event = asyncio.Event()
        self._topology_ready = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None
        self._resource_manager = None
        self._cycle_counter = 0
        self._request_seq = 0
        self._num_gpus: Optional[int] = None
        self._required_gpus_per_node: Optional[int] = None
        self._coordinator_handle_cache = {}
        self._tracer = SchedulerTracer()

    async def _wait_topology_ready(self) -> None:
        """Wait for topology initialization with a bounded timeout.

        Raises RuntimeError if initialize() has not completed within the deadline.
        """
        try:
            await asyncio.wait_for(self._topology_ready.wait(), timeout=_TOPOLOGY_READY_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Scheduler topology not ready after {_TOPOLOGY_READY_TIMEOUT_S}s — "
                f"initialize() was never called or failed. "
                f"Set RLIX_TOPOLOGY_READY_TIMEOUT_S to adjust."
            ) from None

    async def shutdown(self) -> None:
        """Explicit shutdown - call from orchestrator for clean termination.

        Acquires scheduler lock to ensure no concurrent tracing writes.
        Note: This is bounded best-effort - orchestrator uses 0.5s timeout.
        """
        async with self._lock:
            self._tracer.shutdown_tracing()

    async def register_pipeline(
        self,
        *,
        pipeline_id: str,
        ray_namespace: str,
        cluster_tp_configs: Dict[str, int],
        cluster_device_mappings: Dict[str, List[int]],
    ) -> None:
        """Register a pipeline's cluster topology with the scheduler.

        Must be called before ``admit_pipeline``.  Delegates to
        ``register_pipeline_topology`` after validating the pipeline ID.
        """
        validate_pipeline_id(pipeline_id)
        await self._wait_topology_ready()
        await self.register_pipeline_topology(
            pipeline_id=pipeline_id,
            ray_namespace=ray_namespace,
            cluster_tp_configs=cluster_tp_configs,
            cluster_device_mappings=cluster_device_mappings,
        )

    async def admit_pipeline(self, *, pipeline_id: str) -> None:
        """Mark a registered pipeline as admitted so it can issue GPU requests."""
        validate_pipeline_id(pipeline_id)
        async with self._lock:
            info = self._state.pipeline_registry.get(pipeline_id)
            if info is None:
                raise RuntimeError(f"Pipeline {pipeline_id!r} must be registered before admission")
            info["admitted"] = True

    async def unregister_pipeline(self, *, pipeline_id: str) -> None:
        """Remove a pipeline and release all of its resources.

        Two-phase approach under the lock:
          1. **Validate** — parse every cluster_id; fail-fast on corruption.
          2. **Mutate** — remove allocations, pending requests, and planned releases.
        Pending waiters are woken with an error so callers do not hang.
        """
        validate_pipeline_id(pipeline_id)
        async with self._lock:
            # ============================================================
            # PHASE 1: VALIDATE - Parse all cluster_ids, fail-fast if any malformed
            # ============================================================
            allocations_to_remove: List[str] = []
            for cluster_id in self._state.active_allocations:
                try:
                    parsed_pipeline_id, _ = parse_cluster_id(cluster_id)
                except ValueError as e:
                    # CRITICAL: Malformed cluster_id indicates system corruption
                    # Signal all waiters and trigger global fail-fast shutdown
                    error_msg = f"Malformed cluster_id in active_allocations: {cluster_id!r}"
                    self._signal_all_waiters_with_error(error=error_msg)
                    await self._fail_fast_shutdown(reason=f"unregister_pipeline_invalid_cluster_id: {cluster_id!r}")
                    raise RuntimeError(error_msg) from e
                if parsed_pipeline_id == pipeline_id:
                    allocations_to_remove.append(cluster_id)

            pending_to_remove: Dict[Priority, List[PendingRequest]] = {}
            for priority in Priority:
                pending_to_remove[priority] = []
                for pending in self._state.pending_bucket(priority):
                    try:
                        parsed_pipeline_id, _ = parse_cluster_id(pending.request.cluster_id)
                    except ValueError as e:
                        error_msg = f"Malformed cluster_id in pending bucket: {pending.request.cluster_id!r}"
                        self._signal_all_waiters_with_error(error=error_msg)
                        await self._fail_fast_shutdown(
                            reason=f"unregister_pipeline_invalid_cluster_id: {pending.request.cluster_id!r}"
                        )
                        raise RuntimeError(error_msg) from e
                    if parsed_pipeline_id == pipeline_id:
                        pending_to_remove[priority].append(pending)

            planned_releases_to_remove: List[str] = []
            for cluster_id in self._state.pending_planned_release_requests:
                try:
                    parsed_pipeline_id, _ = parse_cluster_id(cluster_id)
                except ValueError as e:
                    error_msg = f"Malformed cluster_id in pending_planned_release_requests: {cluster_id!r}"
                    self._signal_all_waiters_with_error(error=error_msg)
                    await self._fail_fast_shutdown(reason=f"unregister_pipeline_invalid_cluster_id: {cluster_id!r}")
                    raise RuntimeError(error_msg) from e
                if parsed_pipeline_id == pipeline_id:
                    planned_releases_to_remove.append(cluster_id)

            # ============================================================
            # PHASE 2: MUTATE - Non-throwing operations only
            # ============================================================
            self._state.pipeline_registry.pop(pipeline_id, None)
            self._state.latest_progress_by_pipeline.pop(pipeline_id, None)
            self._coordinator_handle_cache.pop(pipeline_id, None)

            # Remove allocations
            for cluster_id in allocations_to_remove:
                alloc = self._state.active_allocations.pop(cluster_id, None)
                if alloc is not None:
                    self._tracer.end_traces_for_gpu_ids(alloc.gpu_ids)
                    self._state.idle_gpus |= set(alloc.gpu_ids)
                    self._tracer.trace_active_gpus_update(
                        num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus)
                    )

            # Remove pending requests and close queue slices
            affected_priorities: Set[Priority] = set()
            for priority, pendings in pending_to_remove.items():
                bucket = self._state.pending_bucket(priority)
                for pending in pendings:
                    # Queue Tracing: Close slice before removing
                    self._tracer.trace_queue_slice_close(pending.request.cluster_id)
                    pending.error = f"Pipeline {pipeline_id!r} unregistered"
                    pending.event.set()
                    if pending in bucket:
                        bucket.remove(pending)
                    affected_priorities.add(priority)
                # Queue Tracing: Update counter for affected priorities
                if priority in affected_priorities:
                    self._tracer.trace_queue_counter_update(priority, len(bucket))

            # Remove planned releases
            for cluster_id in planned_releases_to_remove:
                req = self._state.pending_planned_release_requests.pop(cluster_id, None)
                if req is not None:
                    req.error = f"Pipeline {pipeline_id!r} unregistered"
                    req.event.set()

            self._wakeup_event.set()

    async def initialize(
        self,
        *,
        resource_manager: Any | None = None,
        enable_gpu_tracing: bool = False,
        trace_output_dir: Optional[str] = None,
    ) -> None:
        """Bootstrap the scheduler: query GPU topology, seed idle pool, start loop.

        Idempotent — returns immediately if already initialized.  Must be called
        after the ResourceManager has completed ``init_topology()``.  Optionally
        enables Perfetto GPU tracing via parameter or ``RLIX_ENABLE_GPU_TRACING``
        env var.
        """
        if self._topology_ready.is_set() and self._loop_task is not None:
            return

        if resource_manager is not None:
            self._resource_manager = resource_manager
        if self._resource_manager is None:
            raise RuntimeError("SchedulerImpl.initialize requires a ResourceManager actor (created by orchestrator)")

        try:
            required_gpus_per_node = int(await self._resource_manager.get_required_gpus_per_node.remote())
        except RuntimeError as e:
            msg = str(e)
            if "init_topology" in msg or "not initialized" in msg:
                raise RuntimeError(
                    "ResourceManager topology not initialized; orchestrator must call ResourceManager.init_topology() first"
                ) from e
            raise
        if required_gpus_per_node <= 0:
            raise RuntimeError(f"Invalid required_gpus_per_node={required_gpus_per_node}, expected > 0")
        self._required_gpus_per_node = required_gpus_per_node

        num_gpus = int(await self._resource_manager.get_num_gpus.remote())
        if num_gpus <= 0:
            raise RuntimeError(f"ResourceManager reported num_gpus={num_gpus}, expected > 0")

        async with self._lock:
            self._state.idle_gpus = set(range(num_gpus))
            self._num_gpus = num_gpus
            self._topology_ready.set()
            if self._loop_task is None:
                self._loop_task = asyncio.create_task(self._central_scheduling_loop())

        # GPU Tracing: Enable tracing if parameter or env var is set
        # NOTE: Both env vars are read here (in scheduler actor) for consistency
        env_tracing = os.environ.get("RLIX_ENABLE_GPU_TRACING", "").lower() in ("1", "true")
        env_trace_dir = os.environ.get("RLIX_TRACE_OUTPUT_DIR")
        self._tracer.init_tracing(
            enable=enable_gpu_tracing or env_tracing, trace_output_dir=trace_output_dir or env_trace_dir
        )

        if self._tracer.enabled:
            # Eagerly create all tracks. Perfetto sorts tracks alphabetically by name, so
            # numeric prefixes ("01_", "02_", ...) in the names control the display order —
            # not the creation order here.
            # Desired UI order (top→bottom): 01_enqueue → 02_exec → 03_release →
            #   04_active_gpus → GPU* → Queue_0_INIT … Queue_6_GEN
            self._tracer.init_enqueue_marker_track()
            self._tracer.init_exec_marker_track()
            self._tracer.init_release_marker_track()
            self._tracer.init_active_gpus_counter()
            assert self._num_gpus is not None
            assert self._required_gpus_per_node is not None
            self._tracer.init_gpu_tracks(num_gpus=self._num_gpus, required_gpus_per_node=self._required_gpus_per_node)
            self._tracer.init_queue_tracks()
            # Active GPU counter: emit initial value (all GPUs idle = 0 active)
            self._tracer.trace_active_gpus_update(num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus))

    def _has_any_pending_request_locked(self, *, cluster_id: str) -> bool:
        """Return True if *any* priority bucket contains a pending request for ``cluster_id``."""
        for priority in Priority:
            for pending in self._state.pending_bucket(priority):
                if pending.request.cluster_id == cluster_id:
                    return True
        return False

    def _has_pending_request_locked(self, *, cluster_id: str, priority: Priority) -> bool:
        """Return True if the bucket for ``priority`` contains a pending request for ``cluster_id``."""
        return any(pending.request.cluster_id == cluster_id for pending in self._state.pending_bucket(priority))

    async def register_pipeline_topology(
        self,
        *,
        pipeline_id: str,
        ray_namespace: str,
        cluster_tp_configs: Dict[str, int],
        cluster_device_mappings: Dict[str, List[int]],
    ) -> None:
        """Validate and store per-cluster TP sizes and device mappings for a pipeline.

        Enforces topology constraints (contiguous TP groups, node-boundary alignment)
        and records the cluster configs in ``pipeline_registry`` under the lock.

        Args:
            pipeline_id: Unique pipeline identifier (``{type}_{12_hex}``).
            ray_namespace: Ray namespace for the pipeline's coordinator actor.
            cluster_tp_configs: Mapping of cluster name to tensor-parallel size.
            cluster_device_mappings: Mapping of cluster name to ordered GPU id list.
        """
        await self._wait_topology_ready()
        validate_pipeline_id(pipeline_id)
        if not isinstance(ray_namespace, str) or ray_namespace == "":
            raise ValueError("ray_namespace must be non-empty str")
        if not cluster_tp_configs:
            raise ValueError("cluster_tp_configs must be non-empty")
        if not cluster_device_mappings:
            raise ValueError("cluster_device_mappings must be non-empty")
        if set(cluster_tp_configs.keys()) != set(cluster_device_mappings.keys()):
            missing_tp = sorted(set(cluster_device_mappings.keys()) - set(cluster_tp_configs.keys()))
            missing_map = sorted(set(cluster_tp_configs.keys()) - set(cluster_device_mappings.keys()))
            raise ValueError(
                f"cluster config mismatch: missing tp_size for {missing_tp}, missing device_mapping for {missing_map}"
            )
        if GENERATION_CLUSTER_NAME not in cluster_tp_configs:
            raise ValueError(f"{GENERATION_CLUSTER_NAME} cluster must be registered")

        cluster_configs: Dict[str, Dict[str, Any]] = {}
        used_gpus_by_cluster: Dict[str, Set[int]] = {}
        for cluster_name, tp_size_raw in cluster_tp_configs.items():
            tp_size = int(tp_size_raw)
            if tp_size <= 0:
                raise ValueError(f"tp_size must be > 0 for cluster {cluster_name!r}, got {tp_size!r}")
            device_mapping = list(cluster_device_mappings.get(cluster_name) or [])
            if not device_mapping and cluster_name != REWARD_CLUSTER_NAME:
                raise ValueError(f"device_mapping must be non-empty for cluster {cluster_name!r}")
            if cluster_name == REWARD_CLUSTER_NAME and device_mapping:
                # TODO: support GPU reward clusters (currently restricted to CPU-only).
                raise ValueError("reward cluster only supports CPU-only mode: reward.device_mapping must be empty")
            if device_mapping and len(device_mapping) != len(set(device_mapping)):
                raise ValueError(f"device_mapping has duplicates for cluster {cluster_name!r}")
            num_gpus = self._num_gpus
            if num_gpus is None or num_gpus <= 0:
                raise RuntimeError("Scheduler GPU topology is not initialized (num_gpus unknown)")
            for gpu in device_mapping:
                if not isinstance(gpu, int):
                    raise ValueError(
                        f"device_mapping must be list[int], got {type(gpu).__name__} for cluster {cluster_name!r}"
                    )
                if gpu < 0 or gpu >= num_gpus:
                    raise ValueError(
                        f"device_mapping GPU id out of range for cluster {cluster_name!r}: gpu={gpu} not in [0,{num_gpus - 1}]"
                    )
            if device_mapping:
                device_mapping = _validate_and_canonicalize_device_mapping(
                    cluster_name=cluster_name,
                    tp_size=tp_size,
                    device_mapping=device_mapping,
                    required_gpus_per_node=int(self._required_gpus_per_node),
                )
            is_gen = cluster_name == GENERATION_CLUSTER_NAME
            cfg: Dict[str, Any] = {"tp_size": tp_size, "is_generation": is_gen, "device_mapping": device_mapping}
            if is_gen:
                cfg["max_dp_workers"] = len(device_mapping) // tp_size
            cluster_configs[cluster_name] = cfg
            if device_mapping:
                used_gpus_by_cluster[cluster_name] = set(int(x) for x in device_mapping)

        async with self._lock:
            self._state.pipeline_registry[pipeline_id] = {
                "namespace": ray_namespace,
                "cluster_configs": cluster_configs,
                "scheduler_cache": {},
                "group_queue_cache": {},
                "admitted": False,
            }

    async def get_pipeline_namespace(self, *, pipeline_id: str) -> str:
        """Return the Ray namespace for a pipeline (deterministic from pipeline_id)."""
        validate_pipeline_id(pipeline_id)
        return get_pipeline_namespace(pipeline_id)

    async def report_progress(self, report: ProgressReport) -> None:
        """Accept a progress report from a coordinator and wake the scheduling loop.

        Reports are keyed by (pipeline_id, mode, stream_key) where stream_key is
        the adapter_id for LoRA pipelines or a reserved sentinel for full-finetune.
        Source-type mixing (LoRA vs full-finetune) within a pipeline is rejected.
        """
        validate_pipeline_id(report.pipeline_id)
        if report.step_target_trajectories <= 0:
            raise ValueError("step_target_trajectories must be > 0")
        async with self._lock:
            if report.pipeline_id not in self._state.pipeline_registry:
                raise RuntimeError(f"pipeline_id {report.pipeline_id!r} not registered")
            # Keep latest nested by pipeline->mode->stream in one store.
            # stream_key is lora_id for LoRA streams, or reserved for full-finetune.
            metrics = report.metrics if isinstance(report.metrics, dict) else {}
            if "completed" not in metrics:
                raise ValueError(
                    f"ProgressReport from pipeline {report.pipeline_id!r} missing required 'completed' metric"
                )
            if "remaining" in metrics:
                raise ValueError(
                    f"ProgressReport from pipeline {report.pipeline_id!r} contains wire-level 'remaining' "
                    "at scheduler ingress; only 'completed' is accepted"
                )
            mode = str(metrics.get("mode", "train"))
            lora_id = metrics.get("adapter_id")
            stream_key_full_ft = _FULL_FINETUNE_STREAM_KEY
            pipeline_bucket = self._state.latest_progress_by_pipeline.setdefault(report.pipeline_id, {})
            has_full_ft = any(stream_key_full_ft in mode_bucket for mode_bucket in pipeline_bucket.values())
            has_lora = any(
                any(stream_key != stream_key_full_ft for stream_key in mode_bucket.keys())
                for mode_bucket in pipeline_bucket.values()
            )
            if lora_id is None:
                # Source type (full-ft vs LoRA) is fixed at caller init and must not flip mid-run.
                if has_lora:
                    # Report exact conflicting modes to avoid misleading mode-specific errors.
                    conflicting_lora_modes = sorted(
                        mode_name
                        for mode_name, mode_bucket in pipeline_bucket.items()
                        if any(stream_key != stream_key_full_ft for stream_key in mode_bucket.keys())
                    )
                    raise RuntimeError(
                        f"pipeline_id {report.pipeline_id!r} already has LoRA streams in modes {conflicting_lora_modes}; "
                        "full-finetune report (adapter_id=None) is a source-type mismatch"
                    )
                mode_bucket = pipeline_bucket.setdefault(mode, {})
                mode_bucket[stream_key_full_ft] = report
            else:
                lora_key = str(lora_id)
                if lora_key == stream_key_full_ft:
                    raise ValueError(f"adapter_id {lora_key!r} is reserved for full-finetune stream")
                # Source type (full-ft vs LoRA) is fixed at caller init and must not flip mid-run.
                if has_full_ft:
                    # Report exact conflicting modes to avoid misleading mode-specific errors.
                    conflicting_full_ft_modes = sorted(
                        mode_name
                        for mode_name, mode_bucket in pipeline_bucket.items()
                        if stream_key_full_ft in mode_bucket
                    )
                    raise RuntimeError(
                        f"pipeline_id {report.pipeline_id!r} already has a full-finetune stream in modes {conflicting_full_ft_modes}; "
                        f"LoRA report (adapter_id={lora_key!r}) is a source-type mismatch"
                    )
                mode_bucket = pipeline_bucket.setdefault(mode, {})
                mode_bucket[lora_key] = report
            self._wakeup_event.set()

    async def clear_progress(self, *, pipeline_id: str) -> None:
        """Remove all stored progress for a pipeline.

        Called by the coordinator when all streams are cleared (no active
        get_batch requests). Removes the pipeline from latest_progress_by_pipeline
        so the planner sees zero demand.
        """
        validate_pipeline_id(pipeline_id)
        async with self._lock:
            self._state.latest_progress_by_pipeline.pop(pipeline_id, None)
            self._wakeup_event.set()

    async def request_gpus(
        self,
        *,
        cluster_id: str,
        priority: Priority,
        global_step: Optional[int] = None,
        lora_name: Optional[str] = None,  # GPU Tracing: LoRA name for non-generation clusters
    ) -> List[int]:
        """Block until GPUs are allocated for ``cluster_id`` at ``priority``.

        Returns the list of allocated GPU ids.  If the cluster already has a
        matching allocation, the existing GPU list is returned immediately.
        Duplicate pending requests for the same cluster_id are rejected.
        """
        await self._wait_topology_ready()
        validate_cluster_id(cluster_id)
        event = asyncio.Event()
        pending: PendingRequest | None = None
        async with self._lock:
            pipeline_id, _ = parse_cluster_id(cluster_id)
            info = self._state.pipeline_registry.get(pipeline_id)
            if info is None:
                raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
            if not bool(info.get("admitted", False)):
                raise RuntimeError(f"pipeline_id {pipeline_id!r} not admitted; call orchestrator.admit_pipeline first")
            existing = self._state.active_allocations.get(cluster_id)
            if existing is not None:
                if existing.priority != priority:
                    raise RuntimeError(
                        f"cluster_id {cluster_id!r} already allocated with priority={existing.priority}, requested priority={priority}"
                    )
                if priority == Priority.GENERATION and not existing.active_dp_ranks:
                    pass
                elif priority != Priority.GENERATION and not existing.gpu_ids:
                    pass
                else:
                    return list(existing.gpu_ids)
            if self._has_any_pending_request_locked(cluster_id=cluster_id):
                raise RuntimeError(f"Duplicate pending request for cluster_id={cluster_id!r} is not supported")
            self._request_seq += 1
            pending = PendingRequest(
                request=Request(cluster_id=cluster_id, priority=priority, timestamp=float(self._request_seq)),
                event=event,
                global_step=global_step,
                lora_name=lora_name,  # GPU Tracing: pass lora_name to pending request
            )
            self._state.pending_bucket(priority).append(pending)
            # Queue Tracing: Track enqueue AFTER append (depth is correct)
            self._tracer.trace_queue_enqueue(
                cluster_id, priority, lora_name, bucket_depth=len(self._state.pending_bucket(priority))
            )
            # GPU Tracing: Instant marker for successful enqueue
            self._tracer.trace_enqueue_marker(cluster_id, priority)
            self._wakeup_event.set()
        await event.wait()
        if pending is None:
            raise RuntimeError("request_gpus internal error: pending request not created")
        if pending.error is not None:
            raise RuntimeError(pending.error)
        return list(pending.result)

    async def notify_release_gpus(self, *, cluster_id: str, global_step: Optional[int] = None) -> None:
        """Release all GPUs held by ``cluster_id`` back to the idle pool."""
        await self._wait_topology_ready()
        async with self._lock:
            alloc = self._state.active_allocations.pop(cluster_id, None)
            if alloc is None:
                raise RuntimeError(f"cluster_id {cluster_id!r} not found in active_allocations")
            # GPU Tracing: End traces for released GPUs
            self._tracer.end_traces_for_gpu_ids(alloc.gpu_ids)
            self._state.idle_gpus |= set(alloc.gpu_ids)
            self._tracer.trace_active_gpus_update(num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus))
            # GPU Tracing: Instant marker for release
            self._tracer.trace_release_marker(cluster_id, alloc.gpu_ids)
            self._wakeup_event.set()

    async def notify_release_then_request_gpus(
        self,
        *,
        release_cluster_id: str,
        release_global_step: int,
        request_cluster_id: str,
        request_priority: Priority,
        request_global_step: Optional[int] = None,
        request_lora_name: Optional[str] = None,  # GPU Tracing: LoRA name for non-generation clusters
    ) -> List[int]:
        """Atomically release one cluster's GPUs and enqueue a request for another.

        Both operations happen under a single lock acquisition so the freed GPUs
        are immediately visible to the scheduler when planning the new request.
        Rejects same-priority release-and-request (would be a no-op).
        """
        await self._wait_topology_ready()
        event = asyncio.Event()
        pending: PendingRequest | None = None
        async with self._lock:
            pipeline_id, _ = parse_cluster_id(request_cluster_id)
            info = self._state.pipeline_registry.get(pipeline_id)
            if info is None:
                raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
            if not bool(info.get("admitted", False)):
                raise RuntimeError(f"pipeline_id {pipeline_id!r} not admitted; call orchestrator.admit_pipeline first")
            if release_cluster_id not in self._state.active_allocations:
                raise RuntimeError(f"release_cluster_id {release_cluster_id!r} is not currently allocated")
            existing_to_release = self._state.active_allocations.get(release_cluster_id)
            # Redundant guard: the in-check above already ensures this, but kept explicit.
            if existing_to_release is None:
                raise RuntimeError(f"release_cluster_id {release_cluster_id!r} not found in active_allocations")
            # Releasing a cluster whose priority already matches the incoming request priority
            # indicates a caller bug (e.g. releasing GENERATION to re-request GENERATION).
            if existing_to_release.priority == request_priority:
                raise RuntimeError(
                    f"release_cluster_id {release_cluster_id!r} priority is already {existing_to_release.priority}; "
                    f"releasing and immediately re-requesting the same priority {request_priority!r} is a no-op"
                )

            alloc = self._state.active_allocations.pop(release_cluster_id, None)
            if alloc is None:
                raise RuntimeError(f"release_cluster_id {release_cluster_id!r} not found")
            # GPU Tracing: End traces for released GPUs
            self._tracer.end_traces_for_gpu_ids(alloc.gpu_ids)
            self._state.idle_gpus |= set(alloc.gpu_ids)
            self._tracer.trace_active_gpus_update(num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus))
            # GPU Tracing: Instant marker for release
            self._tracer.trace_release_marker(release_cluster_id, alloc.gpu_ids)
            if self._has_any_pending_request_locked(cluster_id=request_cluster_id):
                raise RuntimeError(f"Duplicate pending request for cluster_id={request_cluster_id!r} is not supported")
            self._request_seq += 1
            pending = PendingRequest(
                request=Request(
                    cluster_id=request_cluster_id, priority=request_priority, timestamp=float(self._request_seq)
                ),
                event=event,
                global_step=request_global_step,
                lora_name=request_lora_name,  # GPU Tracing: pass lora_name to pending request
            )
            self._state.pending_bucket(request_priority).append(pending)
            # Queue Tracing: Track enqueue AFTER append (depth is correct)
            self._tracer.trace_queue_enqueue(
                request_cluster_id,
                request_priority,
                request_lora_name,
                bucket_depth=len(self._state.pending_bucket(request_priority)),
            )
            # GPU Tracing: Instant marker for successful enqueue
            self._tracer.trace_enqueue_marker(request_cluster_id, request_priority)
            self._wakeup_event.set()
        await event.wait()
        if pending is None:
            raise RuntimeError("notify_release_then_request_gpus internal error: pending request not created")
        if pending.error is not None:
            raise RuntimeError(pending.error)
        return list(pending.result)

    def _signal_all_waiters_with_error(self, *, error: str) -> None:
        """Wake every pending request and planned release with an error message.

        Used during fail-fast shutdown to unblock all waiters so they can propagate
        the error instead of hanging indefinitely.
        """
        # Queue Tracing: Close all queue slices (track from stored state)
        for priority in Priority:
            for pending in list(self._state.pending_bucket(priority)):
                # Close slice before clearing
                self._tracer.trace_queue_slice_close(pending.request.cluster_id)
                pending.error = error
                pending.event.set()
            self._state.pending_bucket(priority).clear()
            # Queue Tracing: Single counter update to 0 after clear
            self._tracer.trace_queue_counter_update(priority, 0)
        for _, req in list(self._state.pending_planned_release_requests.items()):
            req.error = error
            req.event.set()
        self._state.pending_planned_release_requests.clear()

    async def _central_scheduling_loop(self) -> None:
        """Async event loop that drives scheduling cycles.

        Two trigger modes:
          1. **Event-driven** — woken by ``_wakeup_event`` (request, progress, release).
          2. **Background poll** — 1 s timeout triggers ``_should_background_rebalance_locked``
             to detect demand-vs-GPU imbalance without an explicit wakeup.

        On any unhandled exception the loop signals all waiters with an error and
        initiates fail-fast shutdown via the orchestrator.
        """
        while True:
            # Case 1: event-driven scheduling (request/progress/release wakeups) runs immediately.
            # Case 2: if there is no wakeup, use a lightweight poll and run only when
            # background rebalance triggers are met (see _should_background_rebalance_locked()).
            poll_interval_s = 1.0
            woke_by_event = False
            try:
                await asyncio.wait_for(self._wakeup_event.wait(), timeout=poll_interval_s)
                woke_by_event = True
            except asyncio.TimeoutError:
                woke_by_event = False

            should_schedule = woke_by_event
            if not should_schedule:
                async with self._lock:
                    should_schedule = self._should_background_rebalance_locked()

            if not should_schedule:
                continue

            # Only consume the wakeup edge when this cycle was actually event-driven.
            # For timeout-triggered background cycles, keep any concurrently set event.
            if woke_by_event:
                self._wakeup_event.clear()
            try:
                await self.scheduling_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                async with self._lock:
                    self._signal_all_waiters_with_error(
                        error=f"scheduler_loop_failed: {type(e).__name__}: {e}",
                    )
                await self._fail_fast_shutdown(reason=f"central_scheduling_loop_failed: {type(e).__name__}: {e}")
                raise

    def _has_waiting_requests_locked(self) -> bool:
        """Return True if any pending request or planned release exists."""
        # Any pending request/release means scheduler should remain purely event-driven.
        for priority in Priority:
            if self._state.pending_bucket(priority):
                return True
        if self._state.pending_planned_release_requests:
            return True
        return False

    def _iter_pipeline_reports_locked(self, *, pipeline_id: str) -> List[ProgressReport]:
        """Return all stored progress reports for ``pipeline_id`` across modes and streams."""
        reports: List[ProgressReport] = []
        for mode_bucket in self._state.latest_progress_by_pipeline.get(pipeline_id, {}).values():
            reports.extend(mode_bucket.values())
        return reports

    @staticmethod
    def _derive_remaining_from_report(progress: ProgressReport) -> float:
        """Derive remaining demand from a progress report's completed metric.

        Returns max(step_target - completed_clamped, 0). Clamping ensures
        overshoot never produces negative remaining.

        Guarantees:
            completed_clamped in [0, step_target]
            remaining_derived in [0, step_target]
            percent_remaining in [0, 1]
        """
        metrics = progress.metrics if isinstance(progress.metrics, dict) else {}
        completed = max(0.0, float(metrics.get("completed", 0)))
        step_target = float(max(int(progress.step_target_trajectories), 1))
        completed_clamped = min(completed, step_target)
        return max(step_target - completed_clamped, 0.0)

    def _pipeline_progress_totals_locked(self, *, pipeline_id: str) -> Tuple[float, float]:
        """Compute (total_remaining, total_required) for ``pipeline_id``.

        Derives remaining from each report's completed metric rather than
        reading wire-level remaining.
        """
        total_required = 0.0
        total_remaining = 0.0
        for progress in self._iter_pipeline_reports_locked(pipeline_id=pipeline_id):
            total_remaining += self._derive_remaining_from_report(progress)
            total_required += float(max(int(progress.step_target_trajectories), 1))
        return max(0.0, total_remaining), total_required

    def _should_background_rebalance_locked(self) -> bool:
        """Heuristic: should the scheduler run a background rebalance cycle?

        Returns True when no explicit requests are pending AND either:
          - A suspended generation cluster has non-zero remaining demand (Case 2.2), or
          - The worst-50%-completion pipelines' demand fraction deviates from their GPU
            fraction by more than 10 percentage points (Case 2.1).
        """
        # Case 2 applies only when there is no waiting request.
        if self._has_waiting_requests_locked():
            return False

        # Row schema:
        #   (completion, weighted_remaining, active_gpu_count)
        # where weighted_remaining = remaining * tp_size so the demand fraction matches
        # the same weighting used by generation gap-ratio planning.
        generation_rows: List[Tuple[float, float, int]] = []

        for cluster_id, alloc in self._state.active_allocations.items():
            if alloc.priority != Priority.GENERATION:
                continue

            pipeline_id, cluster_name = parse_cluster_id(cluster_id)
            if cluster_name != "actor_infer":
                continue

            infer_cfg = (
                self._state.pipeline_registry.get(pipeline_id, {}).get("cluster_configs", {}).get("actor_infer")
            )
            if infer_cfg is None:
                continue

            tp_size = int(infer_cfg.get("tp_size", 1))
            if tp_size <= 0:
                continue

            # Derive both counters in one pass for this pipeline.
            remaining, total_required = self._pipeline_progress_totals_locked(pipeline_id=pipeline_id)

            # Case 2.2: suspended actor_infer with non-zero remaining demand.
            if not alloc.active_dp_ranks and remaining > 0.0:
                return True

            weighted_remaining = remaining * float(tp_size)
            active_gpu_count = len(alloc.active_dp_ranks) * tp_size
            completion = 0.0 if total_required <= 0.0 else max(0.0, 1.0 - (remaining / total_required))
            generation_rows.append((completion, weighted_remaining, active_gpu_count))

        # Case 2.1: demand-vs-GPU imbalance in the worst 50% completion pipelines.
        if len(generation_rows) < 2:
            return False

        total_weighted_remaining = sum(row[1] for row in generation_rows)
        total_active_gpus = sum(row[2] for row in generation_rows)
        if total_weighted_remaining <= 0.0 or total_active_gpus <= 0:
            return False

        generation_rows.sort(key=lambda row: row[0])  # low completion first
        # Include the boundary cluster in the worst half.
        worst_k = max(1, math.ceil(len(generation_rows) * 0.5))
        worst_rows = generation_rows[:worst_k]

        remaining_fraction = sum(row[1] for row in worst_rows) / total_weighted_remaining
        active_gpu_fraction = sum(row[2] for row in worst_rows) / float(total_active_gpus)
        deviation_percent_points = abs(remaining_fraction - active_gpu_fraction) * 100.0
        return deviation_percent_points > 10.0

    async def scheduling_cycle(self) -> None:
        """Execute one full scheduling cycle: plan, validate, resize, commit.

        Phases (under lock):
          0.5. Process planned release requests (generation shrinks from coordinators).
          2. Non-generation allocation (priorities 0–5) with generation preemption.
          3. Generation gap-ratio planning (proportional DP worker distribution).
          4. Validate the execution plan against current state.
          5. Prepare resize RPC calls (shrink/expand dp_ranks per pipeline).

        Outside lock:
          5. Execute resize RPCs: shrinks first, then expands.

        Under lock again:
          6. Commit state mutations and signal pending waiters.
        """
        await self._wait_topology_ready()
        plan = ExecutionPlan()
        planned_allocation_targets: Set[str] = set()
        resize_calls: List[Tuple[Any, List[int], List[int]]] = []

        try:
            async with self._lock:
                self._cycle_counter += 1

                # Shadow of the real idle pool used for planning-time lookahead: GPUs freed
                # and re-allocated within this cycle are tracked here before any state is committed.
                planned_available_gpus = set(self._state.idle_gpus)

                # Phase 0.5: planned release requests (blocking release hint from pipeline/coordinator).
                for cluster_id, req in list(self._state.pending_planned_release_requests.items()):
                    if cluster_id in plan.clusters_to_remove:
                        req.event.set()
                        self._state.pending_planned_release_requests.pop(cluster_id, None)
                        continue
                    alloc = self._state.active_allocations.get(cluster_id)
                    if alloc is None:
                        raise RuntimeError(f"await_release_gpus for unknown cluster_id {cluster_id!r}")
                    if alloc.priority != Priority.GENERATION:
                        raise RuntimeError(
                            f"await_release_gpus is only supported for GENERATION clusters, got {cluster_id!r}"
                        )
                    if not req.dp_ranks_to_remove:
                        req.event.set()
                        self._state.pending_planned_release_requests.pop(cluster_id, None)
                        continue
                    # Make planned release GPUs available for planning in this cycle.
                    freed: Set[int] = set()
                    for dp_rank in req.dp_ranks_to_remove:
                        bundle = alloc.dp_rank_to_gpus.get(dp_rank)
                        if bundle is None:
                            pipeline_id, _ = parse_cluster_id(cluster_id)
                            infer_cfg = self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"]
                            tp_size = int(infer_cfg.get("tp_size", 1))
                            device_mapping = list(infer_cfg.get("device_mapping") or [])
                            start = dp_rank * tp_size
                            bundle = device_mapping[start : start + tp_size]
                        freed |= set(bundle or [])
                    planned_available_gpus |= freed
                    for existing in plan.sched_guided_shrink_ops:
                        if existing.cluster_id != cluster_id:
                            continue
                        for dp_rank in req.dp_ranks_to_remove:
                            if dp_rank not in existing.dp_ranks_to_remove:
                                existing.dp_ranks_to_remove.append(dp_rank)
                        break
                    else:
                        plan.sched_guided_shrink_ops.append(
                            SchedGuidedShrinkOp(cluster_id=cluster_id, dp_ranks_to_remove=list(req.dp_ranks_to_remove))
                        )

                # Phase 2: non-generation planning (priorities 0-5).
                # Track GPUs held by non-GEN allocations from prior cycles so they can be
                # explicitly excluded from the generation budget in Phase 3.
                non_gen_reserved_gpus: Set[int] = set()
                for alloc in self._state.active_allocations.values():
                    if alloc.priority != Priority.GENERATION:
                        non_gen_reserved_gpus |= set(alloc.gpu_ids)

                for prio_value in range(int(Priority.INITIALIZATION), int(Priority.GENERATION)):
                    prio = Priority(prio_value)
                    bucket = list(self._state.pending_bucket(prio))
                    if not bucket:
                        continue
                    for pending in bucket:
                        cluster_id = pending.request.cluster_id
                        if cluster_id in planned_allocation_targets:
                            raise RuntimeError(f"Duplicate pending allocation planned for cluster_id={cluster_id!r}")
                        if cluster_id in self._state.active_allocations:
                            continue
                        _, cluster_name = parse_cluster_id(cluster_id)
                        device_mapping = (
                            self._state.pipeline_registry.get(parse_cluster_id(cluster_id)[0], {})
                            .get("cluster_configs", {})
                            .get(cluster_name, {})
                            .get("device_mapping")
                        )
                        if device_mapping is None:
                            raise RuntimeError(
                                f"Unknown cluster_id {cluster_id!r}; register_pipeline_topology must run first"
                            )
                        needed = set(device_mapping)
                        missing = needed - planned_available_gpus
                        if missing:
                            # Try to free by shrinking generation donors that hold missing GPUs.
                            # Intentional behavior: non-GEN requests may preempt/suspend GEN workers.
                            # Recovery is handled by background rebalance triggers in the central loop.
                            for donor_cid, donor_alloc in list(self._state.active_allocations.items()):
                                if donor_alloc.priority != Priority.GENERATION:
                                    continue
                                if donor_cid in plan.clusters_to_remove:
                                    continue
                                if not (set(donor_alloc.gpu_ids) & missing):
                                    continue
                                tp_size = int(
                                    self._state.pipeline_registry[parse_cluster_id(donor_cid)[0]]["cluster_configs"][
                                        "actor_infer"
                                    ]["tp_size"]
                                )
                                active_ranks = sorted(donor_alloc.active_dp_ranks)
                                for dp_rank in active_ranks:
                                    bundle = set(donor_alloc.dp_rank_to_gpus.get(dp_rank) or [])
                                    if not (bundle & missing):
                                        continue
                                    # Track whether this dp_rank was already freed for a prior
                                    # allocation in this cycle. Re-adding to planned_available would
                                    # double-count the GPU, letting two allocations claim the same
                                    # physical GPU.
                                    already_in_shrink = False
                                    for existing in plan.sched_guided_shrink_ops:
                                        if existing.cluster_id != donor_cid:
                                            continue
                                        if dp_rank not in existing.dp_ranks_to_remove:
                                            existing.dp_ranks_to_remove.append(dp_rank)
                                        else:
                                            already_in_shrink = True
                                        break
                                    else:
                                        plan.sched_guided_shrink_ops.append(
                                            SchedGuidedShrinkOp(cluster_id=donor_cid, dp_ranks_to_remove=[dp_rank])
                                        )
                                    if not already_in_shrink:
                                        # Newly freed: make the GPU available for planning.
                                        planned_available_gpus |= bundle
                                    # Only subtract GPUs that are actually still unclaimed.
                                    missing -= bundle & planned_available_gpus
                                    if not missing:
                                        break
                                if not missing:
                                    break
                        if needed.issubset(planned_available_gpus):
                            planned_available_gpus -= needed
                            non_gen_reserved_gpus |= needed
                            planned_allocation_targets.add(cluster_id)
                            pipeline_id_for_op, cluster_name_for_op = parse_cluster_id(cluster_id)
                            snapshot_tp_size = int(
                                self._state.pipeline_registry[pipeline_id_for_op]["cluster_configs"][
                                    cluster_name_for_op
                                ].get("tp_size", 1)
                            )
                            plan.signal_pending_allocation_ops.append(
                                SignalPendingAllocationOp(
                                    cluster_id=cluster_id,
                                    gpus_to_allocate=sorted(needed),
                                    priority=prio,
                                    # Carry lora_name directly from the pending so _apply_plan_and_signal
                                    # does not need to re-search the bucket for it.
                                    lora_name=pending.lora_name,
                                    tp_size=snapshot_tp_size,
                                )
                            )

                # Phase 3: generation gap-ratio planning.
                # Re-exclude non-GEN GPUs: planned_available_gpus was seeded from idle_gpus (which
                # already omits them), but Phase 2 may have added GPUs freed from GEN donors that
                # overlap with non-GEN reservations from prior cycles.
                planned_available_gpus -= non_gen_reserved_gpus

                # Phase 3a: snapshot generation workers
                active_dp_workers, inactive_dp_workers, idle_for_gen = planner.snapshot_generation_dp_workers(
                    plan=plan,
                    idle_gpus=set(planned_available_gpus),
                    pipeline_registry=self._state.pipeline_registry,
                    active_allocations=self._state.active_allocations,
                )

                # Wake-only signaling: unblock pending generation requests when any generation
                # worker is active. Previously this required ALL workers to be active, which
                # caused a deadlock: when another pipeline held a GPU needed by a dp worker,
                # the GEN request was never signaled, blocking the pipeline at
                # notify_release_then_request_gpus.
                pending_gen = list(self._state.pending_bucket(Priority.GENERATION))
                for pending in pending_gen:
                    cluster_id = pending.request.cluster_id
                    pipeline_id, cluster_name = parse_cluster_id(cluster_id)
                    if cluster_name != "actor_infer":
                        continue
                    # Signal when any dp worker is active (partial allocation is valid).
                    if not active_dp_workers.get(pipeline_id):
                        continue
                    plan.signal_pending_allocation_ops.append(
                        SignalPendingAllocationOp(
                            cluster_id=cluster_id,
                            gpus_to_allocate=[],
                            priority=Priority.GENERATION,
                            tp_size=0,  # Not used at commit (gpus_to_allocate is empty)
                        )
                    )

                # Phase 3b: gap-ratio planning
                idle_for_gen = planner.plan_generation_gap_ratio(
                    plan,
                    active_dp_workers=active_dp_workers,
                    inactive_dp_workers=inactive_dp_workers,
                    non_gen_reserved_gpus=set(non_gen_reserved_gpus),
                    idle_gpus=idle_for_gen,
                    pipeline_registry=self._state.pipeline_registry,
                    active_allocations=self._state.active_allocations,
                    pending_bucket_gen=list(self._state.pending_bucket(Priority.GENERATION)),
                    progress_totals_fn=self._pipeline_progress_totals_locked,
                )

                # Phase 4: validation
                validate_execution_plan(
                    plan,
                    inputs=ValidationInputs(
                        pipeline_registry=self._state.pipeline_registry,
                        active_allocations=self._state.active_allocations,
                        idle_gpus=set(self._state.idle_gpus),
                    ),
                )

                # Phase 5: prepare execution (Phase 3: propagate shrink to pipeline runtime).
                # IMPORTANT: do not await coordinator RPCs while holding scheduler lock.
                resize_calls = self._prepare_resize_calls_locked(plan)
                # GPU Tracing: snapshot trace infos and plan details before releasing the lock.
                # dp_rank_to_gpus and pipeline_registry must be read while the lock is held.
                shrink_trace_infos = self._collect_shrink_trace_infos_locked(plan)
                expand_trace_infos = self._collect_expand_trace_infos_locked(plan)
                exec_details = self._tracer.plan_to_exec_details(plan)

            # GPU Tracing: Emit execution marker right after planning, before resize RPCs.
            # Guard: skip no-op cycles to avoid thousands of empty markers in the Perfetto timeline.
            if any(
                [
                    exec_details.get("shrinks"),
                    exec_details.get("removes"),
                    exec_details.get("allocates"),
                    exec_details.get("expands"),
                ]
            ):
                self._tracer.trace_execution_marker(exec_details, cycle_counter=self._cycle_counter)
            self._tracer.maybe_flush_trace()

            # Phase 5: execute outside the scheduler lock (avoid deadlocking progress/reporting paths).
            await self._execute_resize_calls(
                resize_calls,
                shrink_trace_infos=shrink_trace_infos,
                expand_trace_infos=expand_trace_infos,
            )

            # Phase 6: commit (Phase 2 simulation: state-only).
            async with self._lock:
                self._apply_plan_and_signal(plan)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fail-fast rule (Issue 107): any execution-phase failure triggers controlled shutdown.
            await self._fail_fast_shutdown(reason=f"scheduler_cycle_failed: {type(e).__name__}: {e}")
            raise

    def _get_or_lookup_coordinator_handle_locked(self, *, pipeline_id: str) -> Any:
        """Return the cached Ray actor handle for a pipeline's coordinator, or look it up."""
        info = self._state.pipeline_registry.get(pipeline_id)
        if info is None:
            raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
        coordinator_namespace = info.get("namespace")
        if not isinstance(coordinator_namespace, str) or coordinator_namespace == "":
            raise RuntimeError(
                f"pipeline_id {pipeline_id!r} has invalid registered namespace {coordinator_namespace!r}"
            )

        cached = self._coordinator_handle_cache.get(pipeline_id)
        if cached is not None:
            cached_namespace, cached_handle = cached
            if cached_namespace == coordinator_namespace:
                return cached_handle

        coordinator_name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"
        handle = get_actor_or_raise(
            coordinator_name,
            coordinator_namespace,
            error_context=f"Coordinator required for pipeline_id={pipeline_id!r}.",
        )

        self._coordinator_handle_cache[pipeline_id] = (coordinator_namespace, handle)
        return handle

    def _reconstruct_bundle_for_dp_rank(self, *, cluster_id: str, dp_rank: int) -> Set[int]:
        """Reconstruct the GPU bundle for a dp_rank when dp_rank_to_gpus mapping is absent.

        Extracted from the nested _reconstruct_bundle fn in _apply_plan_and_signal so it
        can be reused by _collect_shrink_trace_infos_locked without duplicating logic.
        """
        pipeline_id, cluster_name = parse_cluster_id(cluster_id)
        if cluster_name != "actor_infer":
            return set()
        infer_cfg = self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"]
        tp_size = int(infer_cfg.get("tp_size", 1))
        device_mapping = list(infer_cfg.get("device_mapping") or [])
        start = dp_rank * tp_size
        return set(device_mapping[start : start + tp_size])

    def _collect_shrink_trace_infos_locked(self, plan: "ExecutionPlan") -> List[GPUTraceInfo]:
        """Pre-collect GPU trace info for GPUs freed by shrink ops and cluster removals.

        Called under the scheduler lock so dp_rank_to_gpus is still intact.
        Only gpu_id is used at close time; the other fields satisfy the shared GPUTraceInfo type.
        """
        infos: List[GPUTraceInfo] = []
        for op in plan.sched_guided_shrink_ops:
            alloc = self._state.active_allocations.get(op.cluster_id)
            if alloc is None:
                continue
            pipeline_id, _ = parse_cluster_id(op.cluster_id)
            for dp_rank in op.dp_ranks_to_remove:
                bundle = set(alloc.dp_rank_to_gpus.get(dp_rank) or [])
                if not bundle:
                    bundle = self._reconstruct_bundle_for_dp_rank(cluster_id=op.cluster_id, dp_rank=dp_rank)
                for gpu_id in bundle:
                    infos.append(
                        GPUTraceInfo(
                            gpu_id=gpu_id,
                            cluster_id=op.cluster_id,
                            pipeline_id=pipeline_id,
                            dp_rank=dp_rank,
                        )
                    )
        for cluster_id in plan.clusters_to_remove:
            alloc = self._state.active_allocations.get(cluster_id)
            if alloc is None:
                continue
            pipeline_id, _ = parse_cluster_id(cluster_id)
            for gpu_id in alloc.gpu_ids:
                infos.append(
                    GPUTraceInfo(
                        gpu_id=gpu_id,
                        cluster_id=cluster_id,
                        pipeline_id=pipeline_id,
                        dp_rank=0,
                    )
                )
        return infos

    def _collect_expand_trace_infos_locked(self, plan: "ExecutionPlan") -> List[GPUTraceInfo]:
        """Pre-collect GPU trace info for proactive expand allocations.

        Called under the scheduler lock so pipeline_registry is stable.
        Mirrors the start_gpu_trace loop in _apply_plan_and_signal (sched_guided_allocation_ops).
        """
        infos: List[GPUTraceInfo] = []
        for op in plan.sched_guided_allocation_ops:
            if not op.gpus_to_allocate or not op.dp_ranks_to_add:
                continue
            pipeline_id, _ = parse_cluster_id(op.cluster_id)
            tp_size = int(
                self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"].get("tp_size", 1)
            )
            sorted_gpus = sorted(op.gpus_to_allocate)
            for i, dp_rank in enumerate(sorted(op.dp_ranks_to_add)):
                bundle = sorted_gpus[i * tp_size : (i + 1) * tp_size]
                for gpu_id in bundle:
                    infos.append(
                        GPUTraceInfo(
                            gpu_id=gpu_id,
                            cluster_id=op.cluster_id,
                            pipeline_id=pipeline_id,
                            dp_rank=dp_rank,
                        )
                    )
        return infos

    def _prepare_resize_calls_locked(self, plan: ExecutionPlan) -> List[Tuple[Any, List[int], List[int]]]:
        """Prepare resize RPC calls under the scheduler lock.

        Contract: per pipeline per cycle, exactly one of {dp_ranks_to_remove, dp_ranks_to_add} may be non-empty.
        """
        pipeline_to_remove: Dict[str, Set[int]] = {}
        pipeline_to_add: Dict[str, Set[int]] = {}

        def _add_remove(cluster_id: str, dp_ranks: List[int]) -> None:
            if not dp_ranks:
                return
            pipeline_id, cluster_name = parse_cluster_id(cluster_id)
            if cluster_name != "actor_infer":
                return
            s = pipeline_to_remove.setdefault(pipeline_id, set())
            for r in dp_ranks:
                s.add(int(r))

        def _add_add(cluster_id: str, dp_ranks: List[int]) -> None:
            if not dp_ranks:
                return
            pipeline_id, cluster_name = parse_cluster_id(cluster_id)
            if cluster_name != "actor_infer":
                return
            s = pipeline_to_add.setdefault(pipeline_id, set())
            for r in dp_ranks:
                s.add(int(r))

        for op in plan.sched_guided_shrink_ops:
            _add_remove(op.cluster_id, list(op.dp_ranks_to_remove))
        for op in plan.sched_guided_allocation_ops:
            _add_add(op.cluster_id, list(op.dp_ranks_to_add))

        calls: List[Tuple[Any, List[int], List[int]]] = []
        for pipeline_id in sorted(set(pipeline_to_remove.keys()) | set(pipeline_to_add.keys())):
            removes = sorted(pipeline_to_remove.get(pipeline_id, set()))
            adds = sorted(pipeline_to_add.get(pipeline_id, set()))
            # A pipeline may legally shrink one dp_rank while expanding a different one in the same
            # cycle (e.g. training preempts GPU 1 so infer moves to GPU 0). What is never valid is
            # removing and re-adding the *same* dp_rank in one cycle.
            overlapping = set(removes) & set(adds)
            if overlapping:
                raise RuntimeError(
                    "resize_infer dp_rank overlap in a single scheduling cycle: "
                    f"pipeline_id={pipeline_id!r} overlapping_dp_ranks={sorted(overlapping)} "
                    f"dp_ranks_to_remove={removes} dp_ranks_to_add={adds}"
                )
            if not removes and not adds:
                continue
            coordinator = self._get_or_lookup_coordinator_handle_locked(pipeline_id=pipeline_id)
            calls.append((coordinator, removes, adds))
        return calls

    async def _execute_resize_calls(
        self,
        calls: List[Tuple[Any, List[int], List[int]]],
        *,
        shrink_trace_infos: List[GPUTraceInfo],
        expand_trace_infos: List[GPUTraceInfo],
    ) -> None:
        """Execute pipeline resizes outside the scheduler lock.

        Order: shrinks → close GPU traces → expands → open GPU traces.
        Tracing happens here so timestamps reflect actual RPC completion, not state-commit time.
        Order matches ROLL's centralized_gpu_scheduler Phase 5 convention.

        # TODO: only block an expand on the shrinks it actually depends on (i.e. its target GPUs
        # overlap with GPUs being freed). Expands targeting already-idle GPUs can run concurrently
        # with shrinks instead of waiting for all shrinks to finish first.
        """
        # Phase 5.2: execute all shrinks (dp_ranks_to_remove) concurrently and wait for all to complete
        shrink_tasks = [
            coordinator.resize_infer.remote(dp_ranks_to_remove=list(removes), dp_ranks_to_add=[])
            for coordinator, removes, adds in calls
            if removes
        ]
        if shrink_tasks:
            await asyncio.gather(*shrink_tasks)
        # GPU Tracing: close slices right after shrinks complete, before expands start
        if shrink_trace_infos:
            self._tracer.end_traces_for_gpu_ids([info.gpu_id for info in shrink_trace_infos])

        # Phase 5.4: execute all expands (dp_ranks_to_add) concurrently after all shrinks complete
        expand_tasks = [
            coordinator.resize_infer.remote(dp_ranks_to_remove=[], dp_ranks_to_add=list(adds))
            for coordinator, removes, adds in calls
            if adds
        ]
        if expand_tasks:
            await asyncio.gather(*expand_tasks)
        # GPU Tracing: open slices right after expands complete, before state commit
        for info in expand_trace_infos:
            self._tracer.start_gpu_trace(
                info.gpu_id,
                info.cluster_id,
                info.pipeline_id,
                Priority.GENERATION,
                "proactive",
                [info.dp_rank],
                required_gpus_per_node=self._required_gpus_per_node,
                cycle_counter=self._cycle_counter,
            )

    async def _fail_fast_shutdown(self, *, reason: str) -> None:
        """Trigger a forced orchestrator shutdown on unrecoverable scheduler error."""
        try:
            orchestrator = ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=RLIX_NAMESPACE)
        except Exception as e:
            logger.error("Failed to resolve orchestrator actor for shutdown: %s: %s", type(e).__name__, e)
            return
        try:
            ref = orchestrator.shutdown.remote(force=True, reason=reason, source="scheduler")
            await asyncio.wait_for(ref, timeout=_FAIL_FAST_SHUTDOWN_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error("orchestrator.shutdown timed out after %ss", _FAIL_FAST_SHUTDOWN_TIMEOUT_S)
        except Exception as e:
            logger.error("Failed to call orchestrator.shutdown: %s: %s", type(e).__name__, e)

    def _apply_plan_and_signal(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Apply execution plan and return operation details for tracing."""
        # Collect operation details for execution marker
        exec_shrinks: List[Dict[str, Any]] = []
        exec_removes: List[Dict[str, Any]] = []
        exec_allocates: List[Dict[str, Any]] = []
        exec_expands: List[Dict[str, Any]] = []

        # GPU Tracing: shrink/remove trace closes already happened in _execute_resize_calls
        # (right after shrink RPCs completed). Only state mutations happen here.
        for op in plan.sched_guided_shrink_ops:
            if not op.dp_ranks_to_remove:
                continue
            alloc = self._state.active_allocations.get(op.cluster_id)
            if alloc is None:
                continue
            for dp_rank in op.dp_ranks_to_remove:
                bundle = set(alloc.dp_rank_to_gpus.get(dp_rank) or [])
                if not bundle:
                    bundle = self._reconstruct_bundle_for_dp_rank(cluster_id=op.cluster_id, dp_rank=dp_rank)
                alloc.active_dp_ranks.discard(dp_rank)
                alloc.dp_rank_to_gpus.pop(dp_rank, None)
                alloc.gpu_ids = [g for g in alloc.gpu_ids if g not in bundle]
                self._state.idle_gpus |= bundle
                self._tracer.trace_active_gpus_update(
                    num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus)
                )
                # Collect shrink detail for execution marker
                exec_shrinks.append(
                    {
                        "cluster_id": op.cluster_id,
                        "gpus_freed": sorted(bundle),
                        "dp_rank": dp_rank,
                    }
                )

        for cluster_id in plan.clusters_to_remove:
            alloc = self._state.active_allocations.pop(cluster_id, None)
            if alloc is not None:
                self._state.idle_gpus |= set(alloc.gpu_ids)
                self._tracer.trace_active_gpus_update(
                    num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus)
                )
                # Collect remove detail for execution marker
                exec_removes.append(
                    {
                        "cluster_id": cluster_id,
                        "gpus_freed": alloc.gpu_ids,
                    }
                )

        # Apply allocations.
        for op in plan.signal_pending_allocation_ops:
            if not op.gpus_to_allocate:
                if op.priority is None:
                    raise RuntimeError(
                        f"signal_pending_allocation_ops missing priority for cluster_id={op.cluster_id!r}"
                    )
                priority = Priority(op.priority)
                existing = self._state.active_allocations.get(op.cluster_id)
                # If this is a wake-only signal (no new GPUs needed), return the existing allocation
                # so callers don't misinterpret [] as "no allocation".
                if existing is not None and existing.priority == priority and existing.gpu_ids:
                    self._signal_pending_request(
                        cluster_id=op.cluster_id, priority=priority, result=list(existing.gpu_ids)
                    )
                else:
                    self._signal_pending_request(cluster_id=op.cluster_id, priority=priority, result=[])
                continue
            if op.priority is None:
                raise RuntimeError(f"signal_pending_allocation_ops missing priority for cluster_id={op.cluster_id!r}")
            priority = Priority(op.priority)
            pipeline_id, _ = parse_cluster_id(op.cluster_id)
            # Stale-plan tolerance: if unregister_pipeline ran during the lock gap,
            # the pipeline is gone and its waiters are already cleaned up.
            if pipeline_id not in self._state.pipeline_registry:
                if self._has_pending_request_locked(cluster_id=op.cluster_id, priority=priority):
                    raise RuntimeError(
                        f"Pipeline {pipeline_id!r} removed from registry but pending waiter still present "
                        f"for cluster_id={op.cluster_id!r} — cleanup corruption"
                    )
                logger.warning(
                    "Skipping stale signal_pending_allocation_op for unregistered pipeline %r "
                    "(cluster_id=%r); unregister_pipeline already cleaned up",
                    pipeline_id,
                    op.cluster_id,
                )
                continue
            if not self._has_pending_request_locked(cluster_id=op.cluster_id, priority=priority):
                raise RuntimeError(
                    f"Planned allocation has no pending waiter: cluster_id={op.cluster_id!r} priority={priority!r}"
                )
            gpu_set = set(op.gpus_to_allocate)
            tp_size = op.tp_size
            sorted_gpus = sorted(op.gpus_to_allocate)
            dp_rank_to_gpus = build_dp_rank_mapping(sorted_gpus, tp_size)
            active_dp_ranks = set(dp_rank_to_gpus.keys()) if is_generation_cluster(op.cluster_id) else set()
            allocation = ClusterAllocation(
                cluster_id=op.cluster_id,
                gpu_ids=sorted_gpus,
                priority=priority,
                active_dp_ranks=active_dp_ranks,
                dp_rank_to_gpus=dp_rank_to_gpus,
            )
            self._state.idle_gpus -= gpu_set
            self._state.active_allocations[op.cluster_id] = allocation
            self._tracer.trace_active_gpus_update(num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus))
            # GPU Tracing: Start traces for initial allocation
            if self._tracer.enabled:
                # lora_name was stamped onto the op at planning time (from PendingRequest.lora_name),
                # so no bucket search is needed here.
                lora_name = op.lora_name
                for gpu_id in sorted_gpus:
                    # Extract DP rank for generation clusters
                    dp_ranks = None
                    if is_generation_cluster(op.cluster_id):
                        for dp_rank, bundle in dp_rank_to_gpus.items():
                            if gpu_id in bundle:
                                dp_ranks = [dp_rank]
                                break
                    self._tracer.start_gpu_trace(
                        gpu_id,
                        op.cluster_id,
                        pipeline_id,
                        priority,
                        "initial",
                        dp_ranks,
                        lora_name,
                        required_gpus_per_node=self._required_gpus_per_node,
                        cycle_counter=self._cycle_counter,
                    )
            self._signal_pending_request(
                cluster_id=op.cluster_id, priority=priority, result=sorted(op.gpus_to_allocate)
            )
            # Collect allocate detail for execution marker
            exec_allocates.append(
                {
                    "cluster_id": op.cluster_id,
                    "gpus_allocated": sorted(op.gpus_to_allocate),
                    "priority": priority.name,
                }
            )

        # Apply expansions (state commit; RequestScheduler.expand_workers executed in scheduling_cycle before commit).
        # State commit is unconditional; signaling is deferred to a set-based pass to handle
        # the case where signal_pending_allocation_ops already consumed the pending request, or
        # multiple ops target the same cluster (merged by _try_activate_one but guarded here too).
        cluster_ids_to_signal: Set[str] = set()
        for op in plan.sched_guided_allocation_ops:
            if not op.gpus_to_allocate:
                continue
            gpu_set = set(op.gpus_to_allocate)
            pipeline_id, _ = parse_cluster_id(op.cluster_id)
            # Stale-plan tolerance: if unregister_pipeline ran during the lock gap,
            # the pipeline is gone and GPUs were already returned to idle.
            if pipeline_id not in self._state.pipeline_registry:
                logger.warning(
                    "Skipping stale sched_guided_allocation_op for unregistered pipeline %r "
                    "(cluster_id=%r); unregister_pipeline already cleaned up",
                    pipeline_id,
                    op.cluster_id,
                )
                continue
            tp_size = op.tp_size
            sorted_needed = sorted(op.gpus_to_allocate)
            dp_rank_to_gpus_to_add = {
                dp_rank: sorted_needed[i * tp_size : (i + 1) * tp_size]
                for i, dp_rank in enumerate(sorted(op.dp_ranks_to_add))
            }
            alloc = self._state.active_allocations.get(op.cluster_id)
            if alloc is None:
                updated_alloc = ClusterAllocation(
                    cluster_id=op.cluster_id,
                    gpu_ids=sorted_needed,
                    priority=Priority.GENERATION,
                    active_dp_ranks=set(dp_rank_to_gpus_to_add.keys()),
                    dp_rank_to_gpus=dict(dp_rank_to_gpus_to_add),
                )
                self._state.idle_gpus -= gpu_set
                self._state.active_allocations[op.cluster_id] = updated_alloc
            else:
                updated_dp_rank_to_gpus = dict(alloc.dp_rank_to_gpus)
                updated_dp_rank_to_gpus.update(dp_rank_to_gpus_to_add)
                updated_active_dp_ranks = set(alloc.active_dp_ranks) | set(dp_rank_to_gpus_to_add.keys())
                updated_gpu_ids = sorted(set(alloc.gpu_ids) | gpu_set)
                self._state.idle_gpus -= gpu_set
                alloc.dp_rank_to_gpus = updated_dp_rank_to_gpus
                alloc.active_dp_ranks = updated_active_dp_ranks
                alloc.gpu_ids = updated_gpu_ids
            self._tracer.trace_active_gpus_update(num_gpus=self._num_gpus, idle_gpu_count=len(self._state.idle_gpus))
            # GPU Tracing: proactive expand trace opens already happened in _execute_resize_calls
            # (right after expand RPCs completed, before this state commit).
            if op.has_pending_request:
                cluster_ids_to_signal.add(op.cluster_id)
            # Collect expand detail for execution marker
            exec_expands.append(
                {
                    "cluster_id": op.cluster_id,
                    "gpus_allocated": sorted(op.gpus_to_allocate),
                    "dp_ranks_added": sorted(op.dp_ranks_to_add),
                }
            )
        for cluster_id in cluster_ids_to_signal:
            if self._has_pending_request_locked(cluster_id=cluster_id, priority=Priority.GENERATION):
                alloc = self._state.active_allocations[cluster_id]
                self._signal_pending_request(
                    cluster_id=cluster_id, priority=Priority.GENERATION, result=sorted(alloc.gpu_ids)
                )

        # Planned release requests: signal unconditionally — by this point the shrink RPCs
        # have already executed in _execute_resize_calls, so the release is committed.
        for cluster_id, req in list(self._state.pending_planned_release_requests.items()):
            req.event.set()
            self._state.pending_planned_release_requests.pop(cluster_id, None)

        return {
            "shrinks": exec_shrinks,
            "removes": exec_removes,
            "allocates": exec_allocates,
            "expands": exec_expands,
        }

    def _signal_pending_request(
        self, *, cluster_id: str, priority: Priority, result: Optional[List[int]] = None
    ) -> None:
        """Find and fulfill a pending request in ``priority`` bucket for ``cluster_id``.

        Closes the queue trace slice, pops the request from the bucket, and wakes
        the waiting coroutine.  Tolerates a missing request when the pipeline was
        concurrently unregistered (benign race).
        """
        bucket = self._state.pending_bucket(priority)
        for idx, pending in enumerate(bucket):
            if pending.request.cluster_id != cluster_id:
                continue
            # Queue Tracing: Close slice BEFORE pop (track from stored state)
            self._tracer.trace_queue_slice_close(cluster_id)
            # Pop from bucket
            bucket.pop(idx)
            # Queue Tracing: Update counter AFTER pop with correct depth
            self._tracer.trace_queue_counter_update(priority, len(bucket))
            pending.result = list(result or [])
            pending.event.set()
            return
        # FIX: Check if pipeline was unregistered - benign race
        pipeline_id, _ = parse_cluster_id(cluster_id)
        if pipeline_id not in self._state.pipeline_registry:
            # Pipeline was unregistered, pending was already removed by unregister_pipeline
            # This is a benign race, not an error
            return
        # Pipeline still registered but no pending found - actual error
        raise RuntimeError(f"No pending request found for cluster_id={cluster_id!r} priority={priority!r}")

    async def await_release_gpus(
        self,
        *,
        pipeline_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        global_step: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Blocking planned release API (RLix checklist).

        Phase 2 implementation is state-only: scheduler selects dp_ranks_to_remove for the generation cluster from
        allocation state and blocks until the scheduler commits a shrink in its next cycle.
        """

        await self._wait_topology_ready()
        if (pipeline_id is None) == (cluster_id is None):
            raise ValueError("Exactly one of pipeline_id or cluster_id must be provided")
        if cluster_id is None:
            validate_pipeline_id(str(pipeline_id))
            cluster_id = f"{pipeline_id}_{GENERATION_CLUSTER_NAME}"
        if timeout_s is not None and (not isinstance(timeout_s, (int, float)) or timeout_s <= 0):
            raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")

        event = asyncio.Event()
        async with self._lock:
            alloc = self._state.active_allocations.get(cluster_id)
            if alloc is None:
                raise RuntimeError(f"cluster_id {cluster_id!r} not found in active_allocations")
            if alloc.priority != Priority.GENERATION:
                raise RuntimeError(f"await_release_gpus only supports GENERATION clusters, got {cluster_id!r}")
            if not alloc.active_dp_ranks:
                return

            # Idempotency: if already pending, wait on the existing request.
            existing = self._state.pending_planned_release_requests.get(cluster_id)
            if existing is not None:
                event = existing.event
                req = existing
            else:
                pipeline_id, cluster_name = parse_cluster_id(cluster_id)
                if cluster_name != "actor_infer":
                    raise RuntimeError(
                        f"await_release_gpus only supports actor_infer generation clusters, got {cluster_id!r}"
                    )

                dp_ranks_to_remove = sorted(alloc.active_dp_ranks)

                req = PendingPlannedReleaseRequest(
                    cluster_id=cluster_id,
                    dp_ranks_to_remove=dp_ranks_to_remove,
                    event=event,
                    global_step=global_step,
                )
                self._state.pending_planned_release_requests[cluster_id] = req
                self._wakeup_event.set()

        try:
            if timeout_s is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=float(timeout_s))
        except asyncio.TimeoutError:
            await self._fail_fast_shutdown(reason=f"await_release_gpus_timeout: cluster_id={cluster_id!r}")
            raise
        if req.error is not None:
            raise RuntimeError(req.error)


def scheduler_actor_class():
    """Return a Ray remote actor class wrapping ``SchedulerImpl`` with no restarts."""
    return ray.remote(max_restarts=0, max_task_retries=0)(SchedulerImpl)
