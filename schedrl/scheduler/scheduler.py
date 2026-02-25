from __future__ import annotations

"""SchedRL Scheduler (ENG-123 Phase 2).

Operational policy (ENG-123): fail-fast only. No recovery or rehydration is provided; on any
scheduler restart, pipelines are expected to re-register and be re-admitted.
"""

import atexit
import asyncio
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from schedrl.protocol.request_id import validate_pipeline_id
from schedrl.protocol.types import Priority, ProgressReport
from schedrl.scheduler.state import SchedulerState
from schedrl.scheduler.types import (
    ClusterAllocation,
    ExecutionPlan,
    PendingPlannedReleaseRequest,
    PendingRequest,
    Request,
    SchedGuidedAllocationOp,
    SchedGuidedShrinkOp,
    SignalPendingAllocationOp,
    is_generation_cluster,
    parse_cluster_id,
    validate_cluster_id,
)
from schedrl.scheduler.validation import ValidationInputs, normalize_progress_oldest_ts, validate_execution_plan

import ray

# GPU Tracing: Conditional import for tg4perfetto (may not be installed)
try:
    from tg4perfetto import TraceGenerator
    _TG4PERFETTO_AVAILABLE = True
except ImportError:
    TraceGenerator = None  # type: ignore[misc,assignment]
    _TG4PERFETTO_AVAILABLE = False

# GPU Tracing: Type-only imports for static type checking
if TYPE_CHECKING:
    from tg4perfetto import Group, NormalTrack

# GPU Tracing: TypeVar for safe trace call helper
T = TypeVar("T")

# GPU Tracing: Short names for GPU trace labels - matches actual Priority enum values
_PRIORITY_SHORT = {
    Priority.INITIALIZATION: "INIT",
    Priority.ACTOR_TRAINING: "ACT",
    Priority.CRITIC_TRAINING: "CRT",
    Priority.OLD_LOG_PROBS: "OLD",
    Priority.REF_LOG_PROBS: "REF",
    Priority.VALUE_COMPUTE: "VAL",
    Priority.GENERATION: "GEN",
}


def _validate_and_canonicalize_device_mapping(
    *,
    cluster_name: str,
    tp_size: int,
    device_mapping: List[int],
    required_gpus_per_node: int,
) -> List[int]:
    """Validate + canonicalize device_mapping.

    Canonical form: sorted GPU ids.

    ENG-123 topology contract:
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
            if group[0] % required_gpus_per_node != 0 or group[-1] % required_gpus_per_node != (required_gpus_per_node - 1):
                raise ValueError(
                    f"TP group must align to node boundaries for cluster {cluster_name!r}: group={group} "
                    f"(gpus_per_node={required_gpus_per_node})"
                )
    return canonical


@dataclass(frozen=True, slots=True)
class _GapRatioDPWorker:
    pipeline_id: str
    dp_rank: int
    gpu_ids: List[int]


@dataclass(slots=True)
class _GapRatioPipelineState:
    pipeline_id: str
    cluster_id: str
    remaining: float
    percent_remaining: float
    tp_size: int
    active_dp_workers: List[_GapRatioDPWorker]
    inactive_dp_workers: List[_GapRatioDPWorker]
    target_ratio: float = 0.0
    existing_ratio: float = 0.0
    gap: float = 0.0
    target_gpu_count: int = 0


@dataclass(slots=True)
class _GPUAllocTraceContext:
    """Context for an active GPU allocation trace slice.

    Stored in _gpu_contexts dict keyed by gpu_id.
    Used for debugging and future features (e.g., duration tracking).

    NOTE: Currently stored but not read - reserved for:
    - Duration calculation in _end_gpu_trace
    - Validation of end/start matching
    - Label updates mid-slice (if needed)
    """

    pipeline_id: str
    cluster_id: str
    priority: Priority
    alloc_type: str  # "initial" | "proactive"
    dp_ranks: Optional[List[int]] = None  # Generation clusters only
    lora_name: Optional[str] = None  # Training clusters only


@dataclass(slots=True)
class SchedulerImpl:
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
    _adapter_handle_cache: Dict[str, Tuple[str, Any]] = field(init=False)
    # GPU Tracing: State fields (MUST be declared at class level for slots=True)
    _enable_gpu_tracing: bool = field(init=False, default=False)
    _trace_gen: Optional["TraceGenerator"] = field(init=False, default=None)
    _trace_file_path: Optional[str] = field(init=False, default=None)
    _scheduler_group: Optional["Group"] = field(init=False, default=None)
    _gpu_tracks: Dict[int, "NormalTrack"] = field(init=False, default_factory=dict)
    _gpu_contexts: Dict[int, _GPUAllocTraceContext] = field(init=False, default_factory=dict)
    _trace_last_flush_ns: int = field(init=False, default=0)  # Throttled flush state
    _trace_flush_interval_ns: int = field(init=False, default=1_000_000_000)  # 1 second
    _trace_shutdown_started: bool = field(init=False, default=False)  # Shutdown guard for idempotency

    def __post_init__(self):
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
        self._adapter_handle_cache = {}

    # =========================================================================
    # GPU Tracing: Core Methods
    # =========================================================================

    def _safe_trace_call(self, func: Callable[..., T], *args, **kwargs) -> Tuple[bool, Optional[T]]:
        """Execute tracing call with fail-safe guard.

        Returns:
            tuple[bool, Optional[T]]: (success, result) where success indicates if call completed.

        On first unexpected error, disables all tracing to prevent scheduler crash.
        I/O errors are logged at debug level and don't disable tracing.
        """
        if not self._enable_gpu_tracing:
            return False, None

        try:
            return True, func(*args, **kwargs)
        except (IOError, OSError) as e:
            # I/O errors are expected - don't disable, don't spam logs
            logging.getLogger(__name__).debug(f"Trace I/O error: {e}")
            return False, None
        except Exception as e:
            # Unexpected errors - disable tracing immediately to prevent crash
            logging.getLogger(__name__).warning(f"Tracing disabled due to unexpected error: {e}")
            self._enable_gpu_tracing = False
            return False, None

    def _safe_final_flush(self) -> None:
        """Guarded flush for shutdown - not dependent on _enable_gpu_tracing.

        Called only from _shutdown_tracing during explicit shutdown.
        Safe to call even when tracing was already disabled.
        """
        if self._trace_gen is None:
            return
        try:
            self._trace_gen.flush()
        except Exception as e:
            logging.getLogger(__name__).debug(f"Final trace flush failed: {e}")

    # -------------------------------------------------------------------------
    # GPU Tracing: Extracted Helper Methods
    # -------------------------------------------------------------------------

    def _safe_trace(self, func: Callable[..., Any], *args, **kwargs) -> bool:
        """Fire-and-forget trace call. Returns success status."""
        ok, _ = self._safe_trace_call(func, *args, **kwargs)
        return ok

    def _safe_trace_get(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        """Trace call with return value. Returns result or None on failure."""
        _, result = self._safe_trace_call(func, *args, **kwargs)
        return result

    def _get_or_create_gpu_track(self, gpu_id: int) -> Optional["NormalTrack"]:
        """Get existing track or create new one. Returns None on failure."""
        if gpu_id in self._gpu_tracks:
            return self._gpu_tracks[gpu_id]

        if self._required_gpus_per_node is None:
            return None

        node_id = gpu_id // int(self._required_gpus_per_node)
        local_id = gpu_id % int(self._required_gpus_per_node)
        track = self._safe_trace_get(
            self._scheduler_group.create_track,
            f"GPU{gpu_id}_{node_id}_{local_id}",
        )
        if track is not None:
            self._gpu_tracks[gpu_id] = track
        return track

    def _build_trace_label(
        self,
        cluster_id: str,
        pipeline_id: str,
        priority: Priority,
        alloc_type: str,
        dp_ranks: Optional[List[int]] = None,
        lora_name: Optional[str] = None,
    ) -> str:
        """Build trace label string. Testable in isolation.

        Label format:
        - Training with LoRA:    [ACT] pipeline-1_actor_train | job:pipeline-1 | lora:adapter-0 | initial | C5
        - Training without LoRA: [ACT] pipeline-1_actor_train | job:pipeline-1 | initial | C5
        - Generation:            [GEN] pipeline-1_actor_infer | job:pipeline-1 | initial | C10 | DP:[0,1]
        """
        p = _PRIORITY_SHORT.get(priority, priority.name[:3])
        parts = [f"[{p}]", cluster_id, f"job:{pipeline_id}"]

        # LoRA only for non-generation clusters (with sanitization)
        if lora_name and priority != Priority.GENERATION:
            safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:64]
            parts.append(f"lora:{safe_lora}")

        parts.extend([alloc_type, f"C{self._cycle_counter}"])
        label = " | ".join(parts)

        # DP only for generation clusters
        if dp_ranks and priority == Priority.GENERATION:
            label += f" | DP:{dp_ranks}"

        return label

    def _end_traces_for_gpu_ids(self, gpu_ids: List[int]) -> None:
        """End trace slices for multiple GPUs. Reusable across release paths."""
        for gpu_id in gpu_ids:
            self._end_gpu_trace(gpu_id)

    def _trace_cycle_marker(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Record a cycle marker instant event."""
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return
        if payload:
            self._safe_trace(self._scheduler_group.instant, time.time_ns(), name, kwargs=payload)
        else:
            self._safe_trace(self._scheduler_group.instant, time.time_ns(), name)

    def _maybe_flush_trace(self) -> None:
        """Throttled flush - only flush if interval elapsed."""
        if not self._trace_gen:
            return
        now_ns = time.time_ns()
        if now_ns - self._trace_last_flush_ns >= self._trace_flush_interval_ns:
            ok, _ = self._safe_trace_call(self._trace_gen.flush)
            if ok:
                self._trace_last_flush_ns = now_ns

    def _init_tracing(self, trace_output_dir: Optional[str]) -> None:
        """Initialize trace generator.

        Called once from initialize() when enable_gpu_tracing=True.

        Fail-fast policy:
        - If tg4perfetto not installed when tracing enabled: log warning, disable tracing
        - If trace file creation fails: log warning, disable tracing
        - Runtime I/O errors during tracing: degrade gracefully (don't crash scheduler)

        NOTE: atexit handler is a best-effort fallback for non-Ray scenarios only.
        Ray terminates workers with SIGTERM/SIGKILL, which bypasses Python's atexit.
        The reliable path is explicit shutdown() via orchestrator integration.
        """
        # Guard: tg4perfetto must be available
        if not _TG4PERFETTO_AVAILABLE:
            logging.getLogger(__name__).warning("GPU tracing requested but tg4perfetto not installed; tracing disabled")
            self._enable_gpu_tracing = False
            return

        # Use correct extension: perfetto-trace (protobuf binary, NOT JSON)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._trace_file_path = os.path.join(
            trace_output_dir or os.getcwd(),
            f"schedrl_gpu_timeline_{ts}.perfetto-trace",
        )

        try:
            self._trace_gen = TraceGenerator(self._trace_file_path)
            self._scheduler_group = self._trace_gen.create_group("SCHEDULER")
            # Best-effort fallback for non-Ray scenarios; unreliable in Ray deployments
            atexit.register(self._shutdown_tracing)
            logging.getLogger(__name__).info(f"GPU tracing enabled: {self._trace_file_path}")
        except (IOError, OSError) as e:
            # I/O errors during init are expected - disable tracing gracefully
            logging.getLogger(__name__).warning(f"GPU tracing disabled (cannot create trace file): {e}")
            self._enable_gpu_tracing = False
            self._trace_gen = None
            self._scheduler_group = None
        except Exception as e:
            # Unexpected init errors - disable tracing but don't crash scheduler
            logging.getLogger(__name__).error(f"GPU tracing disabled (unexpected init error): {e}")
            self._enable_gpu_tracing = False
            self._trace_gen = None
            self._scheduler_group = None

    def _shutdown_tracing(self) -> None:
        """Idempotent shutdown. Safe to call multiple times.

        Called by:
        - atexit handler on normal process exit (unreliable in Ray - SIGTERM/SIGKILL bypass atexit)
        - Explicit shutdown() method from orchestrator (reliable path)

        Order of operations:
        1. Check idempotency guard
        2. Disable tracing flag (stops new trace calls)
        3. Clear all trace state (prevent orphaned references)
        4. Final flush (write remaining data)
        """
        if self._trace_shutdown_started:
            return
        self._trace_shutdown_started = True

        if self._trace_gen is None:
            return

        # Step 1: Disable tracing to stop new calls
        self._enable_gpu_tracing = False

        # Step 2: Clear all trace state before flush
        # NormalTrack holds parent generator refs - must clear to prevent writes after flush
        self._gpu_tracks.clear()
        self._gpu_contexts.clear()
        self._scheduler_group = None

        # Step 3: Final flush via guarded helper
        self._safe_final_flush()

        self._trace_gen = None
        # Keep _trace_file_path for post-shutdown assertions
        # Unregister atexit to prevent double-call
        try:
            atexit.unregister(self._shutdown_tracing)
        except Exception:
            pass

    async def shutdown(self) -> None:
        """Explicit shutdown - call from orchestrator for clean termination.

        Acquires scheduler lock to ensure no concurrent tracing writes.
        Note: This is bounded best-effort - orchestrator uses 0.5s timeout.
        """
        async with self._lock:
            self._shutdown_tracing()

    def _start_gpu_trace(
        self,
        gpu_id: int,
        cluster_id: str,
        pipeline_id: str,
        priority: Priority,
        alloc_type: str,
        dp_ranks: Optional[List[int]] = None,
        lora_name: Optional[str] = None,
    ) -> None:
        """Start a trace slice for GPU allocation.

        Creates track if needed, builds label, opens slice, stores context.
        Silently returns if tracing disabled or on error (best-effort).
        """
        if not self._enable_gpu_tracing:
            return

        if self._trace_gen is None or self._scheduler_group is None:
            return

        # Use extracted helper for track creation
        track = self._get_or_create_gpu_track(gpu_id)
        if track is None:
            return

        # Use extracted helper for label building
        label = self._build_trace_label(
            cluster_id, pipeline_id, priority, alloc_type, dp_ranks, lora_name
        )

        # Open slice - ONLY store context on success
        ok, _ = self._safe_trace_call(track.open, time.time_ns(), label)
        if not ok:
            return

        self._gpu_contexts[gpu_id] = _GPUAllocTraceContext(
            pipeline_id=pipeline_id,
            cluster_id=cluster_id,
            priority=priority,
            alloc_type=alloc_type,
            dp_ranks=dp_ranks,
            lora_name=lora_name,
        )

    def _end_gpu_trace(self, gpu_id: int) -> None:
        """End a trace slice for GPU release.

        Removes context and closes the slice on the track.
        Silently handles missing context (may happen in edge cases).
        """
        if not self._enable_gpu_tracing:
            return

        self._gpu_contexts.pop(gpu_id, None)
        track = self._gpu_tracks.get(gpu_id)
        if track:
            self._safe_trace(track.close, time.time_ns())

    async def register_pipeline(
        self,
        *,
        pipeline_id: str,
        ray_namespace: str,
        cluster_tp_configs: Dict[str, int],
        cluster_device_mappings: Dict[str, List[int]],
    ) -> None:
        validate_pipeline_id(pipeline_id)
        await self._topology_ready.wait()
        await self.register_pipeline_topology(
            pipeline_id=pipeline_id,
            ray_namespace=ray_namespace,
            cluster_tp_configs=cluster_tp_configs,
            cluster_device_mappings=cluster_device_mappings,
        )

    async def admit_pipeline(self, *, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        async with self._lock:
            info = self._state.pipeline_registry.get(pipeline_id)
            if info is None:
                raise RuntimeError(f"Pipeline {pipeline_id!r} must be registered before admission")
            info["admitted"] = True

    async def unregister_pipeline(self, *, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        async with self._lock:
            self._state.pipeline_registry.pop(pipeline_id, None)
            self._state.latest_progress_by_pipeline.pop(pipeline_id, None)
            self._adapter_handle_cache.pop(pipeline_id, None)

            # Single iteration: end traces + remove allocations for this pipeline
            for cluster_id in list(self._state.active_allocations.keys()):
                if not cluster_id.startswith(f"{pipeline_id}_"):
                    continue
                alloc = self._state.active_allocations.pop(cluster_id, None)
                if alloc is not None:
                    # GPU Tracing: End traces for all GPUs held by this cluster
                    self._end_traces_for_gpu_ids(alloc.gpu_ids)
                    self._state.idle_gpus |= set(alloc.gpu_ids)

            for priority in Priority:
                bucket = self._state.pending_bucket(priority)
                remaining: List[PendingRequest] = []
                for pending in bucket:
                    if not pending.request.cluster_id.startswith(f"{pipeline_id}_"):
                        remaining.append(pending)
                        continue
                    pending.error = f"Pipeline {pipeline_id!r} unregistered"
                    pending.event.set()
                bucket[:] = remaining

            for cluster_id, req in list(self._state.pending_planned_release_requests.items()):
                if not cluster_id.startswith(f"{pipeline_id}_"):
                    continue
                req.error = f"Pipeline {pipeline_id!r} unregistered"
                req.event.set()
                self._state.pending_planned_release_requests.pop(cluster_id, None)

            self._wakeup_event.set()

    async def initialize(
        self,
        *,
        resource_manager: Any | None = None,
        enable_gpu_tracing: bool = False,
        trace_output_dir: Optional[str] = None,
    ) -> None:
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
        env_tracing = os.environ.get("SCHEDRL_ENABLE_GPU_TRACING", "").lower() in ("1", "true")
        env_trace_dir = os.environ.get("SCHEDRL_TRACE_OUTPUT_DIR")
        self._enable_gpu_tracing = enable_gpu_tracing or env_tracing

        if self._enable_gpu_tracing:
            # Prefer explicit parameter, fall back to env var, then cwd
            self._init_tracing(trace_output_dir or env_trace_dir)

    def _has_any_pending_request_locked(self, *, cluster_id: str) -> bool:
        for priority in Priority:
            for pending in self._state.pending_bucket(priority):
                if pending.request.cluster_id == cluster_id:
                    return True
        return False

    def _has_pending_request_locked(self, *, cluster_id: str, priority: Priority) -> bool:
        return any(pending.request.cluster_id == cluster_id for pending in self._state.pending_bucket(priority))

    async def register_pipeline_topology(
        self,
        *,
        pipeline_id: str,
        ray_namespace: str,
        cluster_tp_configs: Dict[str, int],
        cluster_device_mappings: Dict[str, List[int]],
    ) -> None:
        await self._topology_ready.wait()
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
            raise ValueError(f"cluster config mismatch: missing tp_size for {missing_tp}, missing device_mapping for {missing_map}")
        if "actor_infer" not in cluster_tp_configs:
            raise ValueError("actor_infer cluster must be registered")

        cluster_configs: Dict[str, Dict[str, Any]] = {}
        used_gpus_by_cluster: Dict[str, Set[int]] = {}
        for cluster_name, tp_size_raw in cluster_tp_configs.items():
            tp_size = int(tp_size_raw)
            if tp_size <= 0:
                raise ValueError(f"tp_size must be > 0 for cluster {cluster_name!r}, got {tp_size!r}")
            device_mapping = list(cluster_device_mappings.get(cluster_name) or [])
            if not device_mapping and cluster_name != "reward":
                raise ValueError(f"device_mapping must be non-empty for cluster {cluster_name!r}")
            if cluster_name == "reward" and device_mapping:
                # TODO(ENG-123): support GPU reward clusters (Phase 3 restricts reward to CPU-only).
                raise ValueError("ENG-123 Phase 3 only supports CPU-only reward: reward.device_mapping must be empty")
            if device_mapping and len(device_mapping) != len(set(device_mapping)):
                raise ValueError(f"device_mapping has duplicates for cluster {cluster_name!r}")
            num_gpus = self._num_gpus
            if num_gpus is None or num_gpus <= 0:
                raise RuntimeError("Scheduler GPU topology is not initialized (num_gpus unknown)")
            for gpu in device_mapping:
                if not isinstance(gpu, int):
                    raise ValueError(f"device_mapping must be list[int], got {type(gpu).__name__} for cluster {cluster_name!r}")
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
            is_gen = cluster_name == "actor_infer"
            cfg: Dict[str, Any] = {"tp_size": tp_size, "is_generation": is_gen, "device_mapping": device_mapping}
            if is_gen:
                cfg["max_dp_workers"] = len(device_mapping) // tp_size
            cluster_configs[cluster_name] = cfg
            if device_mapping:
                used_gpus_by_cluster[cluster_name] = set(int(x) for x in device_mapping)

        # Phase 3: fail-fast on overlapping GPU assignments across clusters within a pipeline.
        # Policy: allow `actor_infer` to overlap with other clusters (optional colocation), but disallow overlaps
        # among non-actor_infer clusters.
        used_non_infer: Set[int] = set()
        for cluster_name, used in sorted(used_gpus_by_cluster.items()):
            if cluster_name == "actor_infer":
                continue
            overlap = used_non_infer & used
            if overlap:
                raise ValueError(
                    f"device_mapping overlaps across non-actor_infer clusters within pipeline {pipeline_id!r}: "
                    f"{cluster_name!r} overlaps GPUs {sorted(overlap)}"
                )
            used_non_infer |= used

        async with self._lock:
            self._state.pipeline_registry[pipeline_id] = {
                "namespace": ray_namespace,
                "cluster_configs": cluster_configs,
                "scheduler_cache": {},
                "group_queue_cache": {},
                "admitted": False,
            }

    async def get_pipeline_namespace(self, *, pipeline_id: str) -> str:
        validate_pipeline_id(pipeline_id)
        async with self._lock:
            info = self._state.pipeline_registry.get(pipeline_id)
            if info is None:
                raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
            ray_namespace = info.get("namespace")
            if not isinstance(ray_namespace, str) or ray_namespace == "":
                raise RuntimeError(f"pipeline_id {pipeline_id!r} has invalid registered namespace {ray_namespace!r}")
            return ray_namespace

    async def report_progress(self, report: ProgressReport) -> None:
        validate_pipeline_id(report.pipeline_id)
        # Fail fast with a clear message if caller violates ProgressReport counter contract.
        if report.queued_trajectories is None or report.inflight_trajectories is None:
            raise ValueError("queued_trajectories and inflight_trajectories must not be None")
        if report.step_target_trajectories <= 0:
            raise ValueError("step_target_trajectories must be > 0")
        # Extraction plan: do not clamp percent_completed; allow > 1.0 (overshoot is tolerated).
        # We only validate non-negativity here.
        if float(report.percent_completed) < 0.0:
            raise ValueError(f"percent_completed must be >= 0, got {report.percent_completed!r}")
        oldest_ts = normalize_progress_oldest_ts(report.oldest_unfinished_creation_ts, report.fifo_timestamp)
        if report.oldest_unfinished_creation_ts is None and oldest_ts is not None:
            report = ProgressReport(
                pipeline_id=report.pipeline_id,
                queued_trajectories=report.queued_trajectories,
                inflight_trajectories=report.inflight_trajectories,
                step_target_trajectories=report.step_target_trajectories,
                percent_completed=report.percent_completed,
                oldest_unfinished_creation_ts=oldest_ts,
                active_base_version=report.active_base_version,
                fifo_timestamp=report.fifo_timestamp,
                metrics=report.metrics,
            )
        async with self._lock:
            if report.pipeline_id not in self._state.pipeline_registry:
                raise RuntimeError(f"pipeline_id {report.pipeline_id!r} not registered")
            # Keep latest nested by pipeline->mode->stream in one store.
            # stream_key is adapter_id for adapter streams, or reserved for full-finetune.
            metrics = report.metrics if isinstance(report.metrics, dict) else {}
            mode = str(metrics.get("mode", "train"))
            adapter_id = metrics.get("adapter_id")
            stream_key_full_ft = "__full_finetune__"
            pipeline_bucket = self._state.latest_progress_by_pipeline.setdefault(report.pipeline_id, {})
            has_full_ft = any(stream_key_full_ft in mode_bucket for mode_bucket in pipeline_bucket.values())
            has_adapter = any(
                any(stream_key != stream_key_full_ft for stream_key in mode_bucket.keys())
                for mode_bucket in pipeline_bucket.values()
            )
            if adapter_id is None:
                # Source type (full-ft vs adapter) is fixed at caller init and must not flip mid-run.
                if has_adapter:
                    # Report exact conflicting modes to avoid misleading mode-specific errors.
                    conflicting_adapter_modes = sorted(
                        mode_name
                        for mode_name, mode_bucket in pipeline_bucket.items()
                        if any(stream_key != stream_key_full_ft for stream_key in mode_bucket.keys())
                    )
                    raise RuntimeError(
                        f"pipeline_id {report.pipeline_id!r} already has adapter streams in modes {conflicting_adapter_modes}; "
                        "full-finetune report (adapter_id=None) is a source-type mismatch"
                    )
                mode_bucket = pipeline_bucket.setdefault(mode, {})
                mode_bucket[stream_key_full_ft] = report
            else:
                adapter_key = str(adapter_id)
                if adapter_key == stream_key_full_ft:
                    raise ValueError(f"adapter_id {adapter_key!r} is reserved for full-finetune stream")
                # Source type (full-ft vs adapter) is fixed at caller init and must not flip mid-run.
                if has_full_ft:
                    # Report exact conflicting modes to avoid misleading mode-specific errors.
                    conflicting_full_ft_modes = sorted(
                        mode_name
                        for mode_name, mode_bucket in pipeline_bucket.items()
                        if stream_key_full_ft in mode_bucket
                    )
                    raise RuntimeError(
                        f"pipeline_id {report.pipeline_id!r} already has a full-finetune stream in modes {conflicting_full_ft_modes}; "
                        f"adapter report (adapter_id={adapter_key!r}) is a source-type mismatch"
                    )
                mode_bucket = pipeline_bucket.setdefault(mode, {})
                mode_bucket[adapter_key] = report
            self._wakeup_event.set()

    async def request_gpus(
        self,
        *,
        cluster_id: str,
        priority: Priority,
        global_step: Optional[int] = None,
        lora_name: Optional[str] = None,  # GPU Tracing: LoRA adapter name for non-generation clusters
    ) -> List[int]:
        await self._topology_ready.wait()
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
            self._wakeup_event.set()
        await event.wait()
        if pending is None:
            raise RuntimeError("request_gpus internal error: pending request not created")
        if pending.error is not None:
            raise RuntimeError(pending.error)
        return list(pending.result)

    async def release_gpus(self, *, cluster_id: str, global_step: Optional[int] = None) -> None:
        await self._topology_ready.wait()
        async with self._lock:
            alloc = self._state.active_allocations.pop(cluster_id, None)
            if alloc is None:
                raise RuntimeError(f"cluster_id {cluster_id!r} not found in active_allocations")
            # GPU Tracing: End traces for released GPUs
            self._end_traces_for_gpu_ids(alloc.gpu_ids)
            self._state.idle_gpus |= set(alloc.gpu_ids)
            self._wakeup_event.set()

    async def release_and_request_gpus(
        self,
        *,
        release_cluster_id: Optional[str],
        release_global_step: Optional[int],
        request_cluster_id: str,
        request_priority: Priority,
        request_global_step: Optional[int] = None,
        request_lora_name: Optional[str] = None,  # GPU Tracing: LoRA adapter name for non-generation clusters
    ) -> List[int]:
        await self._topology_ready.wait()
        event = asyncio.Event()
        pending: PendingRequest | None = None
        async with self._lock:
            pipeline_id, _ = parse_cluster_id(request_cluster_id)
            info = self._state.pipeline_registry.get(pipeline_id)
            if info is None:
                raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
            if not bool(info.get("admitted", False)):
                raise RuntimeError(f"pipeline_id {pipeline_id!r} not admitted; call orchestrator.admit_pipeline first")
            existing = self._state.active_allocations.get(request_cluster_id)
            if existing is not None:
                if existing.priority != request_priority:
                    raise RuntimeError(
                        f"cluster_id {request_cluster_id!r} already allocated with priority={existing.priority}, requested priority={request_priority}"
                    )
                if request_priority == Priority.GENERATION and not existing.active_dp_ranks:
                    pass
                elif request_priority != Priority.GENERATION and not existing.gpu_ids:
                    pass
                else:
                    return list(existing.gpu_ids)
            if release_cluster_id is not None:
                alloc = self._state.active_allocations.pop(release_cluster_id, None)
                if alloc is None:
                    raise RuntimeError(f"release_cluster_id {release_cluster_id!r} not found")
                # GPU Tracing: End traces for released GPUs
                self._end_traces_for_gpu_ids(alloc.gpu_ids)
                self._state.idle_gpus |= set(alloc.gpu_ids)
            if self._has_any_pending_request_locked(cluster_id=request_cluster_id):
                raise RuntimeError(f"Duplicate pending request for cluster_id={request_cluster_id!r} is not supported")
            self._request_seq += 1
            pending = PendingRequest(
                request=Request(cluster_id=request_cluster_id, priority=request_priority, timestamp=float(self._request_seq)),
                event=event,
                global_step=request_global_step,
                lora_name=request_lora_name,  # GPU Tracing: pass lora_name to pending request
            )
            self._state.pending_bucket(request_priority).append(pending)
            self._wakeup_event.set()
        await event.wait()
        if pending is None:
            raise RuntimeError("release_and_request_gpus internal error: pending request not created")
        if pending.error is not None:
            raise RuntimeError(pending.error)
        return list(pending.result)

    def _signal_all_waiters_with_error(self, *, error: str) -> None:
        for priority in Priority:
            for pending in list(self._state.pending_bucket(priority)):
                pending.error = error
                pending.event.set()
            self._state.pending_bucket(priority).clear()
        for _, req in list(self._state.pending_planned_release_requests.items()):
            req.error = error
            req.event.set()
        self._state.pending_planned_release_requests.clear()

    async def _central_scheduling_loop(self) -> None:
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
        # Any pending request/release means scheduler should remain purely event-driven.
        for priority in Priority:
            if self._state.pending_bucket(priority):
                return True
        if self._state.pending_planned_release_requests:
            return True
        return False

    def _iter_pipeline_reports_locked(self, *, pipeline_id: str) -> List[ProgressReport]:
        # Aggregate all stored reports across mode+stream for this pipeline.
        reports: List[ProgressReport] = []
        for mode_bucket in self._state.latest_progress_by_pipeline.get(pipeline_id, {}).values():
            reports.extend(mode_bucket.values())
        return reports

    def _pipeline_progress_totals_locked(self, *, pipeline_id: str) -> Tuple[float, float]:
        # Compute remaining/required together so callers do one pass over reports.
        total_required = 0.0
        total_remaining = 0.0
        for progress in self._iter_pipeline_reports_locked(pipeline_id=pipeline_id):
            metrics = progress.metrics if isinstance(progress.metrics, dict) else {}
            if "remaining" in metrics:
                remaining = float(metrics["remaining"])
            else:
                remaining = float(int(progress.queued_trajectories) + int(progress.inflight_trajectories))
            total_remaining += max(0.0, remaining)
            total_required += float(max(int(progress.step_target_trajectories), 1))
        return max(0.0, total_remaining), total_required

    def _should_background_rebalance_locked(self) -> bool:
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

            infer_cfg = self._state.pipeline_registry.get(pipeline_id, {}).get("cluster_configs", {}).get("actor_infer")
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
        await self._topology_ready.wait()
        plan = ExecutionPlan()
        planned_allocation_targets: Set[str] = set()
        resize_calls: List[Tuple[Any, List[int], List[int]]] = []

        try:
            async with self._lock:
                self._cycle_counter += 1

                # GPU Tracing: Cycle start marker
                self._trace_cycle_marker(
                    f"C{self._cycle_counter} Start",
                    payload={
                        "idle_gpus": len(self._state.idle_gpus),
                        "active": len(self._state.active_allocations),
                    },
                )

                planned_available_gpus = set(self._state.idle_gpus)

                # Phase 0.5: planned release requests (blocking release hint from pipeline/coordinator).
                for cluster_id, req in list(self._state.pending_planned_release_requests.items()):
                    if cluster_id in plan.clusters_to_remove:
                        req.event.set()
                        self._state.pending_planned_release_requests.pop(cluster_id, None)
                        continue
                    alloc = self._state.active_allocations.get(cluster_id)
                    if alloc is None:
                        raise RuntimeError(f"notify_ready_to_release for unknown cluster_id {cluster_id!r}")
                    if alloc.priority != Priority.GENERATION:
                        raise RuntimeError(f"notify_ready_to_release is only supported for GENERATION clusters, got {cluster_id!r}")
                    if not req.dp_ranks_to_remove:
                        req.result_released_gpu_ids = []
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
                        device_mapping = self._state.pipeline_registry.get(parse_cluster_id(cluster_id)[0], {}).get("cluster_configs", {}).get(cluster_name, {}).get("device_mapping")
                        if device_mapping is None:
                            raise RuntimeError(f"Unknown cluster_id {cluster_id!r}; register_pipeline_topology must run first")
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
                                    self._state.pipeline_registry[parse_cluster_id(donor_cid)[0]]["cluster_configs"]["actor_infer"]["tp_size"]
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
                            plan.signal_pending_allocation_ops.append(
                                SignalPendingAllocationOp(cluster_id=cluster_id, gpus_to_allocate=sorted(needed), priority=prio)
                            )

                # Phase 3: generation gap-ratio planning.
                planned_available_gpus -= non_gen_reserved_gpus

                active_dp_workers, inactive_dp_workers, idle_for_gen = self._snapshot_generation_dp_workers(
                    plan=plan, idle_gpus=set(planned_available_gpus)
                )

                # Unblock pending generation requests only when all generation workers are already active.
                # This must be mutually exclusive with expansion planning to avoid conflicting plan ops.
                pending_gen = list(self._state.pending_bucket(Priority.GENERATION))
                for pending in pending_gen:
                    cluster_id = pending.request.cluster_id
                    pipeline_id, cluster_name = parse_cluster_id(cluster_id)
                    if cluster_name != "actor_infer":
                        continue
                    if inactive_dp_workers.get(pipeline_id):
                        continue
                    if not active_dp_workers.get(pipeline_id):
                        continue
                    plan.signal_pending_allocation_ops.append(
                        SignalPendingAllocationOp(
                            cluster_id=cluster_id,
                            gpus_to_allocate=[],
                            priority=Priority.GENERATION,
                        )
                    )

                idle_for_gen = self._plan_generation_gap_ratio(
                    plan,
                    active_dp_workers=active_dp_workers,
                    inactive_dp_workers=inactive_dp_workers,
                    non_gen_reserved_gpus=set(non_gen_reserved_gpus),
                    idle_gpus=idle_for_gen,
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
                # IMPORTANT: do not await adapter RPCs while holding scheduler lock.
                resize_calls = self._prepare_resize_calls_locked(plan)

            # Phase 5: execute outside the scheduler lock (avoid deadlocking progress/reporting paths).
            await self._execute_resize_calls(resize_calls)

            # Phase 6: commit (Phase 2 simulation: state-only).
            async with self._lock:
                self._apply_plan_and_signal(plan)

            # GPU Tracing: Cycle end marker + throttled flush (OUTSIDE lock to avoid I/O latency)
            self._trace_cycle_marker(f"C{self._cycle_counter} End")
            self._maybe_flush_trace()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fail-fast rule (Issue 107): any execution-phase failure triggers controlled shutdown.
            await self._fail_fast_shutdown(reason=f"scheduler_cycle_failed: {type(e).__name__}: {e}")
            raise

    def _get_or_lookup_adapter_handle_locked(self, *, pipeline_id: str) -> Any:
        info = self._state.pipeline_registry.get(pipeline_id)
        if info is None:
            raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
        adapter_namespace = info.get("namespace")
        if not isinstance(adapter_namespace, str) or adapter_namespace == "":
            raise RuntimeError(f"pipeline_id {pipeline_id!r} has invalid registered namespace {adapter_namespace!r}")

        cached = self._adapter_handle_cache.get(pipeline_id)
        if cached is not None:
            cached_namespace, cached_handle = cached
            if cached_namespace == adapter_namespace:
                return cached_handle

        adapter_name = f"schedrl:adapter:{pipeline_id}"
        try:
            handle = ray.get_actor(adapter_name, namespace=adapter_namespace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve adapter actor {adapter_name!r} in namespace {adapter_namespace!r} for pipeline_id {pipeline_id!r}"
            ) from e

        self._adapter_handle_cache[pipeline_id] = (adapter_namespace, handle)
        return handle

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
            adapter = self._get_or_lookup_adapter_handle_locked(pipeline_id=pipeline_id)
            calls.append((adapter, removes, adds))
        return calls

    async def _execute_resize_calls(self, calls: List[Tuple[Any, List[int], List[int]]]) -> None:
        """Execute pipeline resizes outside the scheduler lock.

        Order: all shrinks first, then all expands. This ensures GPU memory is released
        before expansion broadcasts attempt to allocate, preventing OOM during transitions.
        Matches fork behavior in external/ROLL_multi_pipeline/.../centralized_gpu_scheduler.py Phase 5.
        """
        # Phase 5.2: execute all shrinks (dp_ranks_to_remove) concurrently and wait for all to complete
        shrink_tasks = [
            adapter.resize_infer.remote(dp_ranks_to_remove=list(removes), dp_ranks_to_add=[])
            for adapter, removes, adds in calls
            if removes
        ]
        if shrink_tasks:
            await asyncio.gather(*shrink_tasks)

        # Phase 5.4: execute all expands (dp_ranks_to_add) concurrently after all shrinks complete
        expand_tasks = [
            adapter.resize_infer.remote(dp_ranks_to_remove=[], dp_ranks_to_add=list(adds))
            for adapter, removes, adds in calls
            if adds
        ]
        if expand_tasks:
            await asyncio.gather(*expand_tasks)

    async def _fail_fast_shutdown(self, *, reason: str) -> None:
        try:
            orchestrator = ray.get_actor("schedrl:orchestrator", namespace="schedrl")
        except Exception as e:
            sys.stderr.write(f"[schedrl][ERROR] Failed to resolve orchestrator actor for shutdown: {type(e).__name__}: {e}\n")
            return
        try:
            orchestrator.shutdown.remote(force=True, reason=reason, source="scheduler")
        except Exception as e:
            sys.stderr.write(f"[schedrl][ERROR] Failed to call orchestrator.shutdown: {type(e).__name__}: {e}\n")
            return

    def _snapshot_generation_dp_workers(
        self, *, plan: ExecutionPlan, idle_gpus: Set[int]
    ) -> Tuple[Dict[str, List[_GapRatioDPWorker]], Dict[str, List[_GapRatioDPWorker]], Set[int]]:
        active_dp_workers: Dict[str, List[_GapRatioDPWorker]] = {}
        inactive_dp_workers: Dict[str, List[_GapRatioDPWorker]] = {}

        planned_removed_ranks: Dict[str, Set[int]] = {}
        for pipeline_id in self._state.pipeline_registry:
            cluster_id = f"{pipeline_id}_actor_infer"
            planned_removed_ranks[cluster_id] = set()
        for op in plan.sched_guided_shrink_ops:
            if not is_generation_cluster(op.cluster_id):
                continue
            planned_removed_ranks.setdefault(op.cluster_id, set()).update(op.dp_ranks_to_remove)

        non_gen_reserved_gpus: Set[int] = set()
        for cluster_id, alloc in self._state.active_allocations.items():
            if alloc.priority != Priority.GENERATION:
                non_gen_reserved_gpus |= set(alloc.gpu_ids)

        for pipeline_id, pipeline_info in self._state.pipeline_registry.items():
            cluster_configs = pipeline_info.get("cluster_configs") or {}
            infer_cfg = cluster_configs.get("actor_infer")
            if infer_cfg is None:
                continue
            tp_size = int(infer_cfg.get("tp_size", 1))
            device_mapping = list(infer_cfg.get("device_mapping") or [])
            if tp_size <= 0 or not device_mapping:
                continue

            cluster_id = f"{pipeline_id}_actor_infer"
            all_dp_ranks = list(range(len(device_mapping) // tp_size))
            removed_ranks = planned_removed_ranks.get(cluster_id, set())

            current_active_ranks: Set[int] = set()
            if cluster_id in self._state.active_allocations:
                alloc = self._state.active_allocations[cluster_id]
                if alloc.priority == Priority.GENERATION:
                    current_active_ranks = set(alloc.active_dp_ranks)

            effective_active_ranks = current_active_ranks - removed_ranks
            active_list: List[_GapRatioDPWorker] = []
            for dp_rank in sorted(effective_active_ranks):
                start_idx = dp_rank * tp_size
                gpus = device_mapping[start_idx : start_idx + tp_size]
                active_list.append(_GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=dp_rank, gpu_ids=list(gpus)))

            inactive_list: List[_GapRatioDPWorker] = []
            for dp_rank in all_dp_ranks:
                if dp_rank in effective_active_ranks:
                    continue
                start_idx = dp_rank * tp_size
                gpus = device_mapping[start_idx : start_idx + tp_size]
                inactive_list.append(_GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=dp_rank, gpu_ids=list(gpus)))

            active_dp_workers[pipeline_id] = active_list
            inactive_dp_workers[pipeline_id] = inactive_list

        assert idle_gpus.isdisjoint(non_gen_reserved_gpus), "idle_gpus must exclude non-GEN reserved GPUs"
        return active_dp_workers, inactive_dp_workers, idle_gpus

    def _has_pending_generation_request(self, cluster_id: str) -> bool:
        return any(p.request.cluster_id == cluster_id for p in self._state.pending_bucket(Priority.GENERATION))

    def _plan_generation_gap_ratio(
        self,
        plan: ExecutionPlan,
        *,
        active_dp_workers: Dict[str, List[_GapRatioDPWorker]],
        inactive_dp_workers: Dict[str, List[_GapRatioDPWorker]],
        non_gen_reserved_gpus: Set[int],
        idle_gpus: Set[int],
        epsilon: float = 0.0,
    ) -> Set[int]:
        # Ported from ROLL_multi_pipeline CentralizedGPUSchedulerImpl._plan_generation_gap_ratio_alternative,
        # adapted to SchedRL-standard progress reporting (percent_completed / step_target_trajectories).

        def _round_half_up(value: float) -> int:
            return int(math.floor(value + 0.5))

        def _remove_worker(worker: _GapRatioDPWorker) -> None:
            donor_pipeline_id = worker.pipeline_id
            donor_active = active_dp_workers.setdefault(donor_pipeline_id, [])
            donor_active[:] = [w for w in donor_active if w.dp_rank != worker.dp_rank]
            inactive_dp_workers.setdefault(donor_pipeline_id, []).append(worker)

        def _append_shrink_dp_rank(*, cluster_id: str, dp_rank: int) -> None:
            for op in plan.sched_guided_shrink_ops:
                if op.cluster_id == cluster_id:
                    if dp_rank not in op.dp_ranks_to_remove:
                        op.dp_ranks_to_remove.append(dp_rank)
                    return
            plan.sched_guided_shrink_ops.append(SchedGuidedShrinkOp(cluster_id=cluster_id, dp_ranks_to_remove=[dp_rank]))

        def _receiver_eligible(state: _GapRatioPipelineState) -> bool:
            if state.cluster_id in plan.clusters_to_remove:
                return False
            if self._has_pending_generation_request(state.cluster_id):
                return True
            return bool(state.active_dp_workers) or state.cluster_id in self._state.active_allocations

        pipeline_states: List[_GapRatioPipelineState] = []
        for pipeline_id in self._state.pipeline_registry:
            cluster_id = f"{pipeline_id}_actor_infer"
            infer_cfg = self._state.pipeline_registry[pipeline_id].get("cluster_configs", {}).get("actor_infer")
            if infer_cfg is None:
                raise KeyError(f"pipeline_id={pipeline_id!r} missing actor_infer cluster config")
            tp_size = int(infer_cfg.get("tp_size", 1))
            if tp_size <= 0:
                raise ValueError(f"pipeline_id={pipeline_id!r} has invalid actor_infer tp_size={tp_size}")

            # Reuse the same per-stream clamping path as background rebalance to keep
            # remaining-demand semantics consistent across planning paths.
            remaining, step_target = self._pipeline_progress_totals_locked(pipeline_id=pipeline_id)
            if step_target <= 0.0:
                continue
            percent_remaining = 0.0 if step_target <= 0 else remaining / step_target

            has_pending = self._has_pending_generation_request(cluster_id)
            if has_pending:
                remaining += step_target
                percent_remaining = remaining / step_target if step_target > 0 else 0.0

            active_list = active_dp_workers.setdefault(pipeline_id, [])
            inactive_list = inactive_dp_workers.setdefault(pipeline_id, [])
            pipeline_states.append(
                _GapRatioPipelineState(
                    pipeline_id=pipeline_id,
                    cluster_id=cluster_id,
                    remaining=remaining,
                    percent_remaining=percent_remaining,
                    tp_size=tp_size,
                    active_dp_workers=active_list,
                    inactive_dp_workers=inactive_list,
                )
            )

        assert idle_gpus.isdisjoint(non_gen_reserved_gpus), "idle_gpus must exclude non-GEN reserved GPUs"

        protected: Set[Tuple[str, int]] = set()
        for op in plan.sched_guided_shrink_ops:
            pipeline_id, cluster_name = parse_cluster_id(op.cluster_id)
            if cluster_name != "actor_infer":
                continue
            for dp_rank in op.dp_ranks_to_remove:
                protected.add((pipeline_id, dp_rank))

        eligible_for_target = [p for p in pipeline_states if _receiver_eligible(p)]
        total_target_weight = sum(p.remaining * p.tp_size for p in eligible_for_target)
        total_gen_budget_gpus = len(idle_gpus) + sum(len(p.active_dp_workers) * p.tp_size for p in pipeline_states)
        if total_gen_budget_gpus == 0:
            return idle_gpus

        for p in pipeline_states:
            if not _receiver_eligible(p) or total_target_weight == 0:
                p.target_ratio = 0.0
                p.target_gpu_count = 0
            else:
                p.target_ratio = (p.remaining * p.tp_size) / total_target_weight
                raw_target_bundles = (p.target_ratio * total_gen_budget_gpus) / p.tp_size
                rounded_bundles = _round_half_up(raw_target_bundles)
                p.target_gpu_count = max(rounded_bundles * p.tp_size, p.tp_size)

        def _update_gaps() -> None:
            for state in pipeline_states:
                active_gpus = len(state.active_dp_workers) * state.tp_size
                state.existing_ratio = 0.0 if total_gen_budget_gpus == 0 else active_gpus / total_gen_budget_gpus
                state.gap = state.target_ratio - state.existing_ratio

        def _compute_shrink_budget_by_pipeline_id() -> Dict[str, int]:
            shrink_budget: Dict[str, int] = {}
            for state in pipeline_states:
                if state.cluster_id in plan.clusters_to_remove:
                    min_bundles = 0
                elif _receiver_eligible(state):
                    if state.target_gpu_count <= 0:
                        min_bundles = 0
                    else:
                        min_bundles = max(1, state.target_gpu_count // state.tp_size)
                else:
                    min_bundles = 0
                shrink_budget[state.pipeline_id] = max(0, len(state.active_dp_workers) - min_bundles)
            return shrink_budget

        def _try_activate_one(
            state: _GapRatioPipelineState,
            *,
            shrink_budget_by_pipeline_id: Dict[str, int],
            percent_remaining_by_pipeline_id: Dict[str, float],
        ) -> bool:
            nonlocal idle_gpus, activations

            if state.cluster_id in plan.clusters_to_remove:
                return False

            available_inactive = [w for w in state.inactive_dp_workers if (state.pipeline_id, w.dp_rank) not in protected]
            if not available_inactive:
                return False

            candidates: List[Tuple[_GapRatioDPWorker, List[Tuple[float, _GapRatioDPWorker, Set[int]]], Tuple[int, Tuple[float, ...], int]]] = []
            for inactive in sorted(available_inactive, key=lambda w: w.dp_rank):
                needed_bundle = set(inactive.gpu_ids)
                if needed_bundle & non_gen_reserved_gpus:
                    continue

                missing = needed_bundle - idle_gpus
                donor_plan: List[Tuple[float, _GapRatioDPWorker, Set[int]]] = []

                if missing:
                    donors: List[Tuple[float, _GapRatioDPWorker, Set[int]]] = []
                    for donor_state in sorted(pipeline_states, key=lambda x: x.gap):
                        if donor_state.gap >= -epsilon:
                            continue
                        if shrink_budget_by_pipeline_id[donor_state.pipeline_id] <= 0:
                            continue
                        for worker in donor_state.active_dp_workers:
                            if (worker.pipeline_id, worker.dp_rank) in protected:
                                continue
                            worker_bundle = set(worker.gpu_ids)
                            if not (worker_bundle & missing):
                                continue
                            donors.append((donor_state.gap, worker, worker_bundle))

                    planned_shrinks_per_pipeline_id: Dict[str, int] = defaultdict(int)
                    picked: List[Tuple[float, _GapRatioDPWorker, Set[int]]] = []
                    for gap_value, worker, worker_bundle in donors:
                        if not missing:
                            break
                        if planned_shrinks_per_pipeline_id[worker.pipeline_id] >= shrink_budget_by_pipeline_id[worker.pipeline_id]:
                            continue
                        picked.append((gap_value, worker, worker_bundle))
                        planned_shrinks_per_pipeline_id[worker.pipeline_id] += 1
                        missing -= worker_bundle

                    if missing:
                        continue
                    donor_plan.extend(picked)

                needs_shrink = 0 if not donor_plan else 1
                donor_percents = sorted([percent_remaining_by_pipeline_id[donor_worker.pipeline_id] for _, donor_worker, _ in donor_plan])
                score = (needs_shrink, tuple([-p for p in donor_percents]), inactive.dp_rank)
                candidates.append((inactive, donor_plan, score))

            if not candidates:
                return False

            inactive, donor_plan, _ = sorted(candidates, key=lambda c: c[2])[0]
            needed_bundle = set(inactive.gpu_ids)
            if needed_bundle & non_gen_reserved_gpus:
                return False

            planned_available = set(idle_gpus)
            for _, _, donor_gpus in donor_plan:
                planned_available |= set(donor_gpus)
            if not needed_bundle.issubset(planned_available):
                return False

            new_idle_gpus = planned_available - needed_bundle

            for _, donor_worker, _ in donor_plan:
                _append_shrink_dp_rank(cluster_id=f"{donor_worker.pipeline_id}_actor_infer", dp_rank=donor_worker.dp_rank)
                _remove_worker(donor_worker)
                protected.add((donor_worker.pipeline_id, donor_worker.dp_rank))

            has_pending_request = self._has_pending_generation_request(state.cluster_id)
            if not has_pending_request and state.cluster_id not in self._state.active_allocations:
                return False
            if state.cluster_id in plan.clusters_to_remove:
                return False
            existing_alloc_op = next(
                (op for op in plan.sched_guided_allocation_ops if op.cluster_id == state.cluster_id),
                None,
            )
            if existing_alloc_op is not None:
                existing_alloc_op.dp_ranks_to_add.append(inactive.dp_rank)
                existing_alloc_op.gpus_to_allocate.extend(sorted(needed_bundle))
                existing_alloc_op.has_pending_request = existing_alloc_op.has_pending_request or has_pending_request
            else:
                plan.sched_guided_allocation_ops.append(
                    SchedGuidedAllocationOp(
                        cluster_id=state.cluster_id,
                        dp_ranks_to_add=[inactive.dp_rank],
                        gpus_to_allocate=sorted(needed_bundle),
                        has_pending_request=has_pending_request,
                    )
                )
            active_dp_workers.setdefault(state.pipeline_id, []).append(inactive)
            receiver_inactive = inactive_dp_workers.setdefault(state.pipeline_id, [])
            receiver_inactive[:] = [w for w in receiver_inactive if w.dp_rank != inactive.dp_rank]
            protected.add((state.pipeline_id, inactive.dp_rank))
            activations += 1
            idle_gpus = new_idle_gpus
            return True

        iterations = 0
        activations = 0
        while True:
            iterations += 1
            if iterations > 10_000 or activations > 1_000:
                raise RuntimeError("gap_ratio_generation_planning_exceeded_limits")

            _update_gaps()
            percent_remaining_by_pipeline_id = {s.pipeline_id: s.percent_remaining for s in pipeline_states}
            shrink_budget_by_pipeline_id = _compute_shrink_budget_by_pipeline_id()

            def _normalized_gap(state: _GapRatioPipelineState) -> Optional[float]:
                if state.target_ratio <= 0:
                    return None
                return state.gap / state.target_ratio

            acceptors: List[_GapRatioPipelineState] = [
                p
                for p in pipeline_states
                if p.gap > epsilon and _receiver_eligible(p) and (len(p.active_dp_workers) * p.tp_size) < p.target_gpu_count
            ]
            acceptors_with_norm_gap = [(_normalized_gap(p), p) for p in acceptors]
            acceptors_with_norm_gap = [(ng, p) for ng, p in acceptors_with_norm_gap if ng is not None]
            acceptors = [p for _, p in sorted(acceptors_with_norm_gap, key=lambda x: (-x[0], -x[1].gap, x[1].pipeline_id))]
            if not acceptors:
                break

            any_activation = False
            for acceptor in acceptors:
                if _try_activate_one(
                    acceptor,
                    shrink_budget_by_pipeline_id=shrink_budget_by_pipeline_id,
                    percent_remaining_by_pipeline_id=percent_remaining_by_pipeline_id,
                ):
                    any_activation = True
                    break

            if not any_activation:
                break

        return idle_gpus

    def _apply_plan_and_signal(self, plan: ExecutionPlan) -> None:
        def _reconstruct_bundle(*, cluster_id: str, dp_rank: int) -> Set[int]:
            pipeline_id, cluster_name = parse_cluster_id(cluster_id)
            if cluster_name != "actor_infer":
                return set()
            infer_cfg = self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"]
            tp_size = int(infer_cfg.get("tp_size", 1))
            device_mapping = list(infer_cfg.get("device_mapping") or [])
            start = dp_rank * tp_size
            return set(device_mapping[start : start + tp_size])

        for op in plan.sched_guided_shrink_ops:
            if not op.dp_ranks_to_remove:
                continue
            alloc = self._state.active_allocations.get(op.cluster_id)
            if alloc is None:
                continue
            for dp_rank in op.dp_ranks_to_remove:
                bundle = set(alloc.dp_rank_to_gpus.get(dp_rank) or [])
                if not bundle:
                    bundle = _reconstruct_bundle(cluster_id=op.cluster_id, dp_rank=dp_rank)
                alloc.active_dp_ranks.discard(dp_rank)
                alloc.dp_rank_to_gpus.pop(dp_rank, None)
                alloc.gpu_ids = [g for g in alloc.gpu_ids if g not in bundle]
                self._state.idle_gpus |= bundle
                # GPU Tracing: End traces for shrunk GPUs
                self._end_traces_for_gpu_ids(list(bundle))

        for cluster_id in plan.clusters_to_remove:
            alloc = self._state.active_allocations.pop(cluster_id, None)
            if alloc is not None:
                # GPU Tracing: End traces for removed cluster
                self._end_traces_for_gpu_ids(alloc.gpu_ids)
                self._state.idle_gpus |= set(alloc.gpu_ids)

        # Apply allocations.
        for op in plan.signal_pending_allocation_ops:
            if not op.gpus_to_allocate:
                if op.priority is None:
                    raise RuntimeError(f"signal_pending_allocation_ops missing priority for cluster_id={op.cluster_id!r}")
                priority = Priority(op.priority)
                existing = self._state.active_allocations.get(op.cluster_id)
                # If this is a wake-only signal (no new GPUs needed), return the existing allocation
                # so callers don't misinterpret [] as "no allocation".
                if existing is not None and existing.priority == priority and existing.gpu_ids:
                    self._signal_pending_request(cluster_id=op.cluster_id, priority=priority, result=list(existing.gpu_ids))
                else:
                    self._signal_pending_request(cluster_id=op.cluster_id, priority=priority, result=[])
                continue
            if op.priority is None:
                raise RuntimeError(f"signal_pending_allocation_ops missing priority for cluster_id={op.cluster_id!r}")
            priority = Priority(op.priority)
            if not self._has_pending_request_locked(cluster_id=op.cluster_id, priority=priority):
                raise RuntimeError(f"Planned allocation has no pending waiter: cluster_id={op.cluster_id!r} priority={priority!r}")
            gpu_set = set(op.gpus_to_allocate)
            self._state.idle_gpus -= gpu_set
            pipeline_id, cluster_name = parse_cluster_id(op.cluster_id)
            tp_size = int(self._state.pipeline_registry[pipeline_id]["cluster_configs"][cluster_name].get("tp_size", 1))
            dp_rank_to_gpus = {}
            if tp_size > 0:
                sorted_gpus = sorted(op.gpus_to_allocate)
                for i in range(0, len(sorted_gpus), tp_size):
                    dp_rank_to_gpus[i // tp_size] = sorted_gpus[i : i + tp_size]
            active_dp_ranks = set(dp_rank_to_gpus.keys()) if is_generation_cluster(op.cluster_id) else set()
            self._state.active_allocations[op.cluster_id] = ClusterAllocation(
                cluster_id=op.cluster_id,
                gpu_ids=sorted(op.gpus_to_allocate),
                priority=priority,
                active_dp_ranks=active_dp_ranks,
                dp_rank_to_gpus=dp_rank_to_gpus,
            )
            # GPU Tracing: Start traces for initial allocation
            if self._enable_gpu_tracing:
                # Get lora_name from pending request
                lora_name = None
                for p in self._state.pending_bucket(priority):
                    if p.request.cluster_id == op.cluster_id:
                        lora_name = p.lora_name
                        break
                # Store in allocation for proactive allocation lookup
                if lora_name:
                    self._state.active_allocations[op.cluster_id].lora_name = lora_name
                for gpu_id in sorted(op.gpus_to_allocate):
                    # Extract DP rank for generation clusters
                    dp_ranks = None
                    if is_generation_cluster(op.cluster_id):
                        for dp_rank, bundle in dp_rank_to_gpus.items():
                            if gpu_id in bundle:
                                dp_ranks = [dp_rank]
                                break
                    self._start_gpu_trace(
                        gpu_id, op.cluster_id, pipeline_id, priority,
                        "initial", dp_ranks, lora_name,
                    )
            self._signal_pending_request(cluster_id=op.cluster_id, priority=priority, result=sorted(op.gpus_to_allocate))

        # Apply expansions (state commit; adapter.expand_workers executed in scheduling_cycle before commit).
        # State commit is unconditional; signaling is deferred to a set-based pass to handle
        # the case where signal_pending_allocation_ops already consumed the pending request, or
        # multiple ops target the same cluster (merged by _try_activate_one but guarded here too).
        cluster_ids_to_signal: Set[str] = set()
        for op in plan.sched_guided_allocation_ops:
            if not op.gpus_to_allocate:
                continue
            gpu_set = set(op.gpus_to_allocate)
            self._state.idle_gpus -= gpu_set
            alloc = self._state.active_allocations.get(op.cluster_id)
            if alloc is None:
                alloc = ClusterAllocation(cluster_id=op.cluster_id, gpu_ids=[], priority=Priority.GENERATION)
                self._state.active_allocations[op.cluster_id] = alloc
            pipeline_id, _ = parse_cluster_id(op.cluster_id)
            tp_size = int(self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"].get("tp_size", 1))
            sorted_needed = sorted(op.gpus_to_allocate)
            for i, dp_rank in enumerate(sorted(op.dp_ranks_to_add)):
                alloc.dp_rank_to_gpus[dp_rank] = sorted_needed[i * tp_size : (i + 1) * tp_size]
                alloc.active_dp_ranks.add(dp_rank)
            alloc.gpu_ids = sorted(set(alloc.gpu_ids) | gpu_set)
            # GPU Tracing: Start traces for proactive allocation
            if self._enable_gpu_tracing:
                for dp_rank in sorted(op.dp_ranks_to_add):
                    bundle = alloc.dp_rank_to_gpus.get(dp_rank, [])
                    for gpu_id in bundle:
                        self._start_gpu_trace(
                            gpu_id, op.cluster_id, pipeline_id,
                            Priority.GENERATION, "proactive", [dp_rank],
                        )
            if op.has_pending_request:
                cluster_ids_to_signal.add(op.cluster_id)
        for cluster_id in cluster_ids_to_signal:
            if self._has_pending_request_locked(cluster_id=cluster_id, priority=Priority.GENERATION):
                alloc = self._state.active_allocations[cluster_id]
                self._signal_pending_request(cluster_id=cluster_id, priority=Priority.GENERATION, result=sorted(alloc.gpu_ids))

        # Planned release requests: signal after shrink commit.
        for cluster_id, req in list(self._state.pending_planned_release_requests.items()):
            # If the cluster still exists, we assume shrink commit applied.
            req.event.set()
            self._state.pending_planned_release_requests.pop(cluster_id, None)

    def _signal_pending_request(self, *, cluster_id: str, priority: Priority, result: Optional[List[int]] = None) -> None:
        bucket = self._state.pending_bucket(priority)
        for idx, pending in enumerate(bucket):
            if pending.request.cluster_id != cluster_id:
                continue
            bucket.pop(idx)
            pending.result = list(result or [])
            pending.event.set()
            return
        raise RuntimeError(f"No pending request found for cluster_id={cluster_id!r} priority={priority!r}")

    async def notify_ready_to_release(
        self,
        *,
        pipeline_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        global_step: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> List[int]:
        """Blocking planned release API (SchedRL checklist).

        Phase 2 implementation is state-only: scheduler selects dp_ranks_to_remove for the generation cluster from
        allocation state and blocks until the scheduler commits a shrink in its next cycle.
        """

        await self._topology_ready.wait()
        if (pipeline_id is None) == (cluster_id is None):
            raise ValueError("Exactly one of pipeline_id or cluster_id must be provided")
        if cluster_id is None:
            validate_pipeline_id(str(pipeline_id))
            cluster_id = f"{pipeline_id}_actor_infer"
        if timeout_s is not None and (not isinstance(timeout_s, (int, float)) or timeout_s <= 0):
            raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")

        event = asyncio.Event()
        async with self._lock:
            alloc = self._state.active_allocations.get(cluster_id)
            if alloc is None:
                raise RuntimeError(f"cluster_id {cluster_id!r} not found in active_allocations")
            if alloc.priority != Priority.GENERATION:
                raise RuntimeError(f"notify_ready_to_release only supports GENERATION clusters, got {cluster_id!r}")
            if not alloc.active_dp_ranks:
                return []

            # Idempotency: if already pending, wait on the existing request.
            existing = self._state.pending_planned_release_requests.get(cluster_id)
            if existing is not None:
                event = existing.event
                req = existing
            else:
                pipeline_id, cluster_name = parse_cluster_id(cluster_id)
                if cluster_name != "actor_infer":
                    raise RuntimeError(f"notify_ready_to_release only supports actor_infer generation clusters, got {cluster_id!r}")
                infer_cfg = self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"]
                tp_size = int(infer_cfg.get("tp_size", 1))
                if tp_size <= 0:
                    raise RuntimeError(f"Invalid tp_size={tp_size} for cluster_id {cluster_id!r}")
                device_mapping = list(infer_cfg.get("device_mapping") or [])
                if not device_mapping:
                    raise RuntimeError(f"Missing device_mapping for cluster_id {cluster_id!r}")

                dp_ranks_to_remove = sorted(alloc.active_dp_ranks)
                released_gpu_ids: List[int] = []
                for dp_rank in sorted(alloc.active_dp_ranks):
                    bundle = alloc.dp_rank_to_gpus.get(dp_rank)
                    if bundle is None:
                        start = dp_rank * tp_size
                        bundle = device_mapping[start : start + tp_size]
                    released_gpu_ids.extend(list(bundle or []))

                req = PendingPlannedReleaseRequest(
                    cluster_id=cluster_id,
                    dp_ranks_to_remove=dp_ranks_to_remove,
                    event=event,
                    global_step=global_step,
                )
                req.result_released_gpu_ids = sorted(set(released_gpu_ids))
                self._state.pending_planned_release_requests[cluster_id] = req
                self._wakeup_event.set()

        try:
            if timeout_s is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=float(timeout_s))
        except asyncio.TimeoutError:
            await self._fail_fast_shutdown(reason=f"notify_ready_to_release_timeout: cluster_id={cluster_id!r}")
            raise
        if req.error is not None:
            raise RuntimeError(req.error)

        return list(req.result_released_gpu_ids)


def scheduler_actor_class():
    return ray.remote(max_restarts=0, max_task_retries=0)(SchedulerImpl)
