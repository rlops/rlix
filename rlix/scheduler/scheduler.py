"""Rlix Scheduler.

Operational policy: fail-fast only. No recovery or rehydration is provided; on any
scheduler restart, pipelines are expected to re-register and be re-admitted.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

import ray

from rlix.protocol.request_id import validate_pipeline_id
from rlix.protocol.types import (
    COORDINATOR_ACTOR_NAME_PREFIX,
    ORCHESTRATOR_ACTOR_NAME,
    RLIX_NAMESPACE,
    Priority,
    ProgressReport,
    get_pipeline_namespace,
)
from rlix.scheduler.state import SchedulerState
from rlix.scheduler.types import (
    ClusterAllocation,
    ExecutionPlan,
    PendingPlannedReleaseRequest,
    PendingRequest,
    Request,
    SchedGuidedAllocationOp,
    SchedGuidedShrinkOp,
    SignalPendingAllocationOp,
    build_dp_rank_mapping,
    is_generation_cluster,
    parse_cluster_id,
    validate_cluster_id,
)
from rlix.scheduler.validation import ValidationInputs, validate_execution_plan
from rlix.utils.ray_actors import get_actor_or_raise

logger = logging.getLogger(__name__)

# GPU Tracing: Conditional import for tg4perfetto (may not be installed)
try:
    from tg4perfetto import TraceGenerator
    _TG4PERFETTO_AVAILABLE = True
except ImportError:
    TraceGenerator = None  # type: ignore[misc,assignment]
    _TG4PERFETTO_AVAILABLE = False

# GPU Tracing: Type-only imports for static type checking
if TYPE_CHECKING:
    from tg4perfetto import CounterTrack, Group, NormalTrack
    from tg4perfetto._tgen import GroupTrack  # GroupTrack not re-exported in __init__

# GPU Tracing: TypeVar for safe trace call helper
T = TypeVar("T")

# GPU Tracing: Short names for GPU trace labels - matches actual Priority enum values
_PRIORITY_SHORT = {
    Priority.INITIALIZATION: "INIT",
    Priority.ACTOR_TRAINING: "TRN",
    Priority.CRITIC_TRAINING: "CRT",
    Priority.OLD_LOG_PROBS: "OLD",
    Priority.REF_LOG_PROBS: "REF",
    Priority.VALUE_COMPUTE: "VAL",
    Priority.GENERATION: "GEN",
}

# Queue tracing: max slice tracks before disabling new queue slice creation.
_QUEUE_TRACK_CAP: int = 1000
# Gap-ratio generation planning iteration limits (safety bounds).
_MAX_GAP_ITERATIONS: int = 10_000
_MAX_GAP_ACTIVATIONS: int = 1_000


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


@dataclass(frozen=True, slots=True)
class _GPUTraceInfo:
    """Per-GPU context snapshot needed to open or close a Perfetto trace slice.

    Collected under the scheduler lock so the data is stable when the RPC runs outside it.
    Shrink close only uses gpu_id; expand open uses all four fields.
    """

    gpu_id: int
    cluster_id: str
    pipeline_id: str
    dp_rank: int


@dataclass(slots=True)
class _QueueSubGroup:
    """Named-track factory wrapping a tg4perfetto GroupTrack sub-group.

    GroupTrack.create_track() creates tracks named after the group, not a caller-supplied name.
    This wrapper holds GroupTrack's internal uuid and parent handles and calls _create_track
    directly (same private method Group.create_track uses) to pass an explicit name.

    Source pattern: _QueueSubGroup wraps GroupTrack for named track creation
    """

    # GroupTrack internal handles — accessed via gt._uuid and gt._parent after create_group()
    _uuid: int
    _parent: Any  # tg4perfetto.TraceGenerator at runtime; Any to avoid runtime import

    def create_track(self, track_name: str) -> "NormalTrack":
        """Create a named slice track under this sub-group."""
        return self._parent._create_track(self._uuid, track_name, 0)

    def create_counter_track(self, track_name: str) -> "CounterTrack":
        """Create a named counter track under this sub-group."""
        return self._parent._create_track(self._uuid, track_name, 1)

    @classmethod
    def from_group_track(cls, gt: "GroupTrack") -> "_QueueSubGroup":
        """Extract handles from a freshly created GroupTrack."""
        return cls(gt._uuid, gt._parent)


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
    _coordinator_handle_cache: Dict[str, Tuple[str, Any]] = field(init=False)
    # GPU Tracing: State fields (MUST be declared at class level for slots=True)
    _enable_gpu_tracing: bool = field(init=False, default=False)
    _trace_gen: Optional["TraceGenerator"] = field(init=False, default=None)
    _trace_file_path: Optional[str] = field(init=False, default=None)
    _scheduler_group: Optional["Group"] = field(init=False, default=None)
    _gpu_tracks: Dict[int, "NormalTrack"] = field(init=False, default_factory=dict)
    _trace_last_flush_ns: int = field(init=False, default=0)  # Throttled flush state
    _trace_flush_interval_ns: int = field(init=False, default=1_000_000_000)  # 1 second
    _trace_shutdown_started: bool = field(init=False, default=False)  # Shutdown guard for idempotency
    # Queue Tracing: State fields for queue visualization
    _pending_queue_trace_state: Dict[str, Tuple["NormalTrack", int, Priority]] = field(
        init=False, default_factory=dict
    )  # cluster_id -> (track, start_ns, priority)
    _queue_counter_tracks: Dict[str, "CounterTrack"] = field(
        init=False, default_factory=dict
    )  # priority_key -> counter track
    _queue_track_count: int = field(
        init=False, default=0
    )  # Total slice tracks created (for cap enforcement)
    # Queue Tracing: Maps priority short-key (e.g. "TRN") to its Perfetto queue sub-group wrapper
    _queue_groups: Dict[str, _QueueSubGroup] = field(init=False, default_factory=dict)
    # Active GPU counter track for utilization visualization
    _active_gpus_counter: Optional["CounterTrack"] = field(init=False, default=None)
    # Marker track for exec instant events
    _exec_marker_track: Optional["NormalTrack"] = field(init=False, default=None)
    # Marker track for enqueue events (separate from exec_markers)
    _enqueue_marker_track: Optional["NormalTrack"] = field(init=False, default=None)
    # Marker track for release events (separate from exec_markers)
    _release_marker_track: Optional["NormalTrack"] = field(init=False, default=None)

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
        self._coordinator_handle_cache = {}

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

    def _init_gpu_tracks(self) -> None:
        """Eagerly create all GPU tracks for correct ordering in Perfetto UI.

        MUST be called after _num_gpus and _required_gpus_per_node are set.
        Track order in Perfetto is determined by creation order.
        """
        if not self._enable_gpu_tracing:
            return
        if self._num_gpus is None or self._required_gpus_per_node is None:
            return

        for gpu_id in range(self._num_gpus):
            self._get_or_create_gpu_track(gpu_id)

    def _get_or_create_queue_group(self, priority: Priority) -> Optional[_QueueSubGroup]:
        """Get or create the Queue_<KEY> sub-group wrapper for a priority tier.

        Returns None if tracing is disabled or scheduler group is not initialized.
        """
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return None
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])
        if key in self._queue_groups:
            return self._queue_groups[key]
        # Perfetto sorts alphabetically: prefix with priority value so INIT(0) < TRN(1) < ... < GEN(6).
        # "Queue_N_KEY" sorts after GPU tracks (uppercase G < Q) and in priority order by digit.
        raw_group = self._safe_trace_get(
            self._scheduler_group.create_group,
            f"Queue_{priority.value}_{key}",
        )
        if raw_group is None:
            logging.getLogger(__name__).debug(
                f"Failed to create queue sub-group for priority {key}"
            )
            return None
        sub_group = _QueueSubGroup.from_group_track(raw_group)
        self._queue_groups[key] = sub_group
        return sub_group

    def _get_or_create_queue_counter_track(self, priority: Priority) -> Optional["CounterTrack"]:
        """Get or create counter track for queue depth. Returns None on failure."""
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])
        if key in self._queue_counter_tracks:
            return self._queue_counter_tracks[key]

        queue_group = self._get_or_create_queue_group(priority)
        if queue_group is None:
            return None
        # create_counter_track is a method on _QueueSubGroup (our wrapper) — always exists
        track = self._safe_trace_get(
            queue_group.create_counter_track,
            "depth",
        )
        if track is not None:
            self._queue_counter_tracks[key] = track
        return track

    def _get_or_create_active_gpus_counter(self) -> Optional["CounterTrack"]:
        """Get or create counter track for active GPU count. Returns None on failure."""
        if self._active_gpus_counter is not None:
            return self._active_gpus_counter

        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return None

        # Prefix "04_" forces alphabetical sort below marker tracks (01-03) and above GPU/Queue
        track = self._safe_trace_get(
            self._scheduler_group.create_counter_track,
            "04_active_gpus",
        )
        if track is not None:
            self._active_gpus_counter = track
        return track

    def _init_active_gpus_counter(self) -> None:
        """Eagerly create active_gpus counter track for correct ordering in Perfetto UI.

        Perfetto sorts tracks alphabetically by name. The "04_" prefix places this counter
        below marker tracks ("01_"-"03_") and above GPU/Queue tracks (uppercase letters).
        """
        if not self._enable_gpu_tracing:
            return
        self._get_or_create_active_gpus_counter()

    def _create_marker_track(self, name: str) -> Optional["NormalTrack"]:
        """Create a marker track with given name. Returns None on failure."""
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return None
        return self._safe_trace_get(self._scheduler_group.create_track, name)

    def _init_enqueue_marker_track(self) -> None:
        """Eagerly create marker track for enqueue instant events.

        Perfetto sorts tracks alphabetically by name. The "01_" prefix places this track
        at the very top (first) in the Perfetto UI.
        """
        self._enqueue_marker_track = self._create_marker_track("01_enqueue_markers")

    def _init_exec_marker_track(self) -> None:
        """Eagerly create marker track for exec instant events.

        Perfetto sorts tracks alphabetically by name. The "02_" prefix places this track
        second in the Perfetto UI, below enqueue_markers.
        """
        self._exec_marker_track = self._create_marker_track("02_exec_markers")

    def _init_release_marker_track(self) -> None:
        """Eagerly create marker track for release instant events.

        Perfetto sorts tracks alphabetically by name. The "03_" prefix places this track
        third in the Perfetto UI, below exec_markers.
        """
        self._release_marker_track = self._create_marker_track("03_release_markers")

    def _init_queue_tracks(self) -> None:
        """Eagerly create queue groups and counter tracks in priority order.

        Creates Queue_<N>_<KEY> groups. Perfetto sorts alphabetically, so the numeric
        prefix N (priority.value) ensures INIT(0) < TRN(1) < ... < GEN(6) in the UI.
        """
        if not self._enable_gpu_tracing:
            return
        # Create queue groups in priority order (lower value = higher priority)
        for priority in Priority:
            self._get_or_create_queue_group(priority)
            self._get_or_create_queue_counter_track(priority)

    def _create_queue_slice_track(
        self, cluster_id: str, priority: Priority, lora_name: Optional[str] = None
    ) -> Optional["NormalTrack"]:
        """Create a per-cluster slice track for queue visualization.

        Returns None on failure OR if track cap exceeded.
        When cap is exceeded, counter tracks still work but slice tracks are skipped.
        """
        # Check cap BEFORE creating
        if self._queue_track_count >= _QUEUE_TRACK_CAP:
            # Log once when cap first hit
            if self._queue_track_count == _QUEUE_TRACK_CAP:
                logging.getLogger(__name__).warning(
                    f"Queue slice track cap ({_QUEUE_TRACK_CAP}) reached. "
                    "Subsequent queue slices will not be traced (counters continue)."
                )
            self._queue_track_count += 1  # Increment to avoid repeated warnings
            return None

        queue_group = self._get_or_create_queue_group(priority)
        if queue_group is None:
            return None

        # Build "[KEY] lora_name pipeline_id" track name so Perfetto rows are human-readable.
        # Lora-relevant priorities (TRN, OLD, REF) pass lora_name; others leave it None.
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])
        pipeline_id, _ = parse_cluster_id(cluster_id)
        safe_pid = pipeline_id[:16]
        if lora_name:
            safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:24]
            track_name = f"[{key}] {safe_lora} {safe_pid}"
        else:
            track_name = f"[{key}] {safe_pid}"

        # create_track is a method on _QueueSubGroup (our wrapper) — always exists
        track = self._safe_trace_get(
            queue_group.create_track,
            track_name,
        )
        if track is not None:
            self._queue_track_count += 1
        return track

    def _trace_queue_enqueue(self, cluster_id: str, priority: Priority, lora_name: Optional[str] = None) -> None:
        """Start queue slice and increment counter when request is enqueued.

        CRITICAL: Call AFTER pending_bucket().append() so len() reflects correct depth.
        CRITICAL: Track handle stored AFTER successful open to avoid orphan state.
        """
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return

        now_ns = time.time_ns()
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])

        # Create per-cluster track (lora_name flows into the track row label)
        slice_track = self._create_queue_slice_track(cluster_id, priority, lora_name)

        # Start slice FIRST
        if slice_track:
            label = f"[{key}] {cluster_id}"
            if lora_name:
                safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:32]
                label += f" | lora:{safe_lora}"
            ok = self._safe_trace(slice_track.open, now_ns, label)
            # CRITICAL: Only store state AFTER successful open
            if ok:
                # Fail fast: duplicate cluster_id means scheduler violated one-request-per-cluster invariant
                assert cluster_id not in self._pending_queue_trace_state, (
                    f"Duplicate queue trace enqueue for cluster_id={cluster_id!r}"
                )
                self._pending_queue_trace_state[cluster_id] = (slice_track, now_ns, priority)

        # Counter: depth = current bucket size (AFTER append, so correct)
        counter_track = self._get_or_create_queue_counter_track(priority)
        if counter_track:
            depth = len(self._state.pending_bucket(priority))
            self._safe_trace(counter_track.count, now_ns, depth)

    def _trace_queue_slice_close(self, cluster_id: str) -> None:
        """Close queue slice when request is fulfilled.

        CRITICAL:
        - Call BEFORE bucket.pop()
        - ALWAYS pops pending state (prevents leaks)
        - Uses stored track handle (no track creation on close)
        - Uses DIRECT close (not _safe_trace) to work even when tracing disabled

        Note: Direct close is safe here because we have a valid track handle from
        successful open. On unexpected errors, disables tracing (consistent with _safe_trace_call).
        """
        # ALWAYS pop entry first to prevent state leaks
        entry = self._pending_queue_trace_state.pop(cluster_id, None)
        if entry is None:
            return  # No pending trace state for this cluster

        stored_track, _, stored_priority = entry

        # Direct close - NOT via _safe_trace, so works even if tracing disabled
        now_ns = time.time_ns()
        if stored_track is not None:
            try:
                stored_track.close(now_ns)
            except (IOError, OSError):
                # I/O errors are expected - ignore
                pass
            except Exception as e:
                # Unexpected error - log and disable tracing (consistent with _safe_trace_call)
                logging.getLogger(__name__).warning(f"Queue trace close error, disabling tracing: {e}")
                self._enable_gpu_tracing = False

    def _trace_queue_counter_update(self, priority: Priority, depth: int) -> None:
        """Update queue depth counter with explicit depth value.

        CRITICAL:
        - Call AFTER bucket.pop() with len(bucket) as depth
        - Depth comes from real bucket length, not cache
        - Separating this from slice_close allows correct ordering
        """
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return

        now_ns = time.time_ns()
        counter_track = self._get_or_create_queue_counter_track(priority)
        if counter_track:
            self._safe_trace(counter_track.count, now_ns, depth)

    def _trace_active_gpus_update(self) -> None:
        """Update active GPUs counter track with current utilization."""
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return

        if self._num_gpus is None:
            return  # Not initialized yet

        counter = self._get_or_create_active_gpus_counter()
        if counter:
            active_count = self._num_gpus - len(self._state.idle_gpus)
            now_ns = time.time_ns()
            self._safe_trace(counter.count, now_ns, active_count)

    def _shutdown_close_queue_slices(self) -> None:
        """Close all open queue slices during shutdown.

        CRITICAL:
        - Does NOT gate on _enable_gpu_tracing (called after it's False)
        - Uses stored track handles directly
        - Called BEFORE _safe_final_flush()
        """
        if not self._pending_queue_trace_state:
            return

        now_ns = time.time_ns()
        for cluster_id, (track, _, _) in list(self._pending_queue_trace_state.items()):
            if track is not None:
                # Direct call, no gating - we're in shutdown
                try:
                    track.close(now_ns)
                except Exception:
                    pass  # Ignore errors during shutdown
        self._pending_queue_trace_state.clear()

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
        - Training with LoRA:    [TRN] | lora:lora-0 | pipeline-1_actor_train | job:pipeline-1 | initial | C5
        - Training without LoRA: [TRN] | pipeline-1_actor_train | job:pipeline-1 | initial | C5
        - Generation:            [GEN] | pipeline-1_actor_infer | job:pipeline-1 | initial | C10 | DP:[0,1]
        """
        p = _PRIORITY_SHORT.get(priority, priority.name[:3])

        # LoRA comes right after [TRN] so the lora name is visible at a glance
        if lora_name and priority != Priority.GENERATION:
            safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:64]
            parts = [f"[{p}]", f"lora:{safe_lora}", cluster_id, f"job:{pipeline_id}"]
        else:
            parts = [f"[{p}]", cluster_id, f"job:{pipeline_id}"]

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

    def _plan_to_exec_details(self, plan: "ExecutionPlan") -> Dict[str, Any]:
        """Convert a finalized ExecutionPlan to the exec_details dict used by the execution marker.

        Called right after planning (before resize RPCs) so the marker fires at decision time.
        """
        return {
            "shrinks": [
                {"cluster_id": op.cluster_id, "dp_ranks": sorted(op.dp_ranks_to_remove)}
                for op in plan.sched_guided_shrink_ops
                if op.dp_ranks_to_remove
            ],
            "removes": [{"cluster_id": cid} for cid in sorted(plan.clusters_to_remove)],
            "allocates": [
                {
                    "cluster_id": op.cluster_id,
                    "gpus_allocated": sorted(op.gpus_to_allocate),
                    "priority": op.priority.name if op.priority else None,
                }
                for op in plan.signal_pending_allocation_ops
                if op.gpus_to_allocate
            ],
            "expands": [
                {
                    "cluster_id": op.cluster_id,
                    "gpus_allocated": sorted(op.gpus_to_allocate),
                    "dp_ranks": sorted(op.dp_ranks_to_add),
                }
                for op in plan.sched_guided_allocation_ops
                if op.gpus_to_allocate
            ],
        }

    def _trace_execution_marker(self, payload: Dict[str, Any]) -> None:
        """Record a per-cycle execution marker summarizing all GPU allocation changes."""
        if not self._enable_gpu_tracing or self._exec_marker_track is None:
            return
        self._safe_trace(
            self._exec_marker_track.instant,
            time.time_ns(),
            f"Exec C{self._cycle_counter}",
            kwargs=payload,
        )

    def _trace_enqueue_marker(self, cluster_id: str, priority: Priority) -> None:
        """Record an instant marker when request is successfully enqueued."""
        if not self._enable_gpu_tracing or self._enqueue_marker_track is None:
            return
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])
        self._safe_trace(
            self._enqueue_marker_track.instant,
            time.time_ns(),
            f"Enqueue: {key}",
            kwargs={"cluster_id": cluster_id, "priority": priority.name},
        )

    def _trace_release_marker(self, cluster_id: str, gpus_released: List[int]) -> None:
        """Record an instant marker when GPUs are released via release_gpus()."""
        if not self._enable_gpu_tracing or self._release_marker_track is None:
            return
        self._safe_trace(
            self._release_marker_track.instant,
            time.time_ns(),
            f"Release: {cluster_id}",
            kwargs={"cluster_id": cluster_id, "gpus_released": gpus_released},
        )

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

        # Guard: tg4perfetto must be available when tracing is explicitly requested
        if not _TG4PERFETTO_AVAILABLE:
            raise RuntimeError(
                "RLIX_ENABLE_GPU_TRACING is set but tg4perfetto is not installed."
                " Install it with: pip install tg4perfetto"
            )

        # Use correct extension: perfetto-trace (protobuf binary, NOT JSON)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._trace_file_path = os.path.join(
            trace_output_dir or os.getcwd(),
            f"rlix_gpu_timeline_{ts}.perfetto-trace",
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
        2. Close all open queue slices FIRST (before disabling tracing)
        3. Disable tracing flag (stops new trace calls)
        4. Clear all trace state (prevent orphaned references)
        5. Final flush (write remaining data)
        """
        if self._trace_shutdown_started:
            return
        self._trace_shutdown_started = True

        if self._trace_gen is None:
            return

        # Step 1: Close all open queue slices FIRST (before disabling tracing)
        # This uses stored track handles, works even if we proceed to disable
        self._shutdown_close_queue_slices()

        # Step 2: Disable tracing to stop new calls
        self._enable_gpu_tracing = False

        # Step 3: Clear all trace state before flush
        # NormalTrack holds parent generator refs - must clear to prevent writes after flush
        self._gpu_tracks.clear()
        self._queue_counter_tracks.clear()
        self._queue_groups.clear()  # wrapper refs hold _parent (TraceGenerator) — must clear
        self._active_gpus_counter = None
        self._scheduler_group = None

        # Step 4: Final flush via guarded helper
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

        # Open slice
        self._safe_trace_call(track.open, time.time_ns(), label)

    def _end_gpu_trace(self, gpu_id: int) -> None:
        """End a trace slice for GPU release.

        Removes context and closes the slice on the track.
        Silently handles missing context (may happen in edge cases).
        """
        if not self._enable_gpu_tracing:
            return

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
                        await self._fail_fast_shutdown(reason=f"unregister_pipeline_invalid_cluster_id: {pending.request.cluster_id!r}")
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
                    self._end_traces_for_gpu_ids(alloc.gpu_ids)
                    self._state.idle_gpus |= set(alloc.gpu_ids)
                    self._trace_active_gpus_update()

            # Remove pending requests and close queue slices
            affected_priorities: Set[Priority] = set()
            for priority, pendings in pending_to_remove.items():
                bucket = self._state.pending_bucket(priority)
                for pending in pendings:
                    # Queue Tracing: Close slice before removing
                    self._trace_queue_slice_close(pending.request.cluster_id)
                    pending.error = f"Pipeline {pipeline_id!r} unregistered"
                    pending.event.set()
                    if pending in bucket:
                        bucket.remove(pending)
                    affected_priorities.add(priority)
                # Queue Tracing: Update counter for affected priorities
                if priority in affected_priorities:
                    self._trace_queue_counter_update(priority, len(bucket))

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
        self._enable_gpu_tracing = enable_gpu_tracing or env_tracing

        if self._enable_gpu_tracing:
            # Prefer explicit parameter, fall back to env var, then cwd
            self._init_tracing(trace_output_dir or env_trace_dir)
            # Eagerly create all tracks. Perfetto sorts tracks alphabetically by name, so
            # numeric prefixes ("01_", "02_", ...) in the names control the display order —
            # not the creation order here.
            # Desired UI order (top→bottom): 01_enqueue → 02_exec → 03_release →
            #   04_active_gpus → GPU* → Queue_0_INIT … Queue_6_GEN
            self._init_enqueue_marker_track()
            self._init_exec_marker_track()
            self._init_release_marker_track()
            self._init_active_gpus_counter()
            self._init_gpu_tracks()
            self._init_queue_tracks()
            # Active GPU counter: emit initial value (all GPUs idle = 0 active)
            self._trace_active_gpus_update()

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
                # TODO: support GPU reward clusters (currently restricted to CPU-only).
                raise ValueError("reward cluster only supports CPU-only mode: reward.device_mapping must be empty")
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
        return get_pipeline_namespace(pipeline_id)

    async def report_progress(self, report: ProgressReport) -> None:
        validate_pipeline_id(report.pipeline_id)
        if report.step_target_trajectories <= 0:
            raise ValueError("step_target_trajectories must be > 0")
        async with self._lock:
            if report.pipeline_id not in self._state.pipeline_registry:
                raise RuntimeError(f"pipeline_id {report.pipeline_id!r} not registered")
            # Keep latest nested by pipeline->mode->stream in one store.
            # stream_key is lora_id for LoRA streams, or reserved for full-finetune.
            metrics = report.metrics if isinstance(report.metrics, dict) else {}
            mode = str(metrics.get("mode", "train"))
            lora_id = metrics.get("adapter_id")
            stream_key_full_ft = "__full_finetune__"
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
            # Queue Tracing: Track enqueue AFTER append (depth is correct)
            self._trace_queue_enqueue(cluster_id, priority, lora_name)
            # GPU Tracing: Instant marker for successful enqueue
            self._trace_enqueue_marker(cluster_id, priority)
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
            self._trace_active_gpus_update()
            # GPU Tracing: Instant marker for release
            self._trace_release_marker(cluster_id, alloc.gpu_ids)
            self._wakeup_event.set()

    async def release_then_request_gpus(
        self,
        *,
        release_cluster_id: str,
        release_global_step: int,
        request_cluster_id: str,
        request_priority: Priority,
        request_global_step: Optional[int] = None,
        request_lora_name: Optional[str] = None,  # GPU Tracing: LoRA name for non-generation clusters
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
            assert release_cluster_id in self._state.active_allocations, (
                f"release_cluster_id {release_cluster_id!r} is not currently allocated"
            )
            existing_to_release = self._state.active_allocations.get(release_cluster_id)
            # Redundant guard: the in-check above already ensures this, but kept explicit.
            assert existing_to_release is not None, (
                f"release_cluster_id {release_cluster_id!r} not found in active_allocations"
            )
            # Releasing a cluster whose priority already matches the incoming request priority
            # indicates a caller bug (e.g. releasing GENERATION to re-request GENERATION).
            assert existing_to_release.priority != request_priority, (
                f"release_cluster_id {release_cluster_id!r} priority is already {existing_to_release.priority}; "
                f"releasing and immediately re-requesting the same priority {request_priority!r} is a no-op"
            )

            alloc = self._state.active_allocations.pop(release_cluster_id, None)
            if alloc is None:
                raise RuntimeError(f"release_cluster_id {release_cluster_id!r} not found")
            # GPU Tracing: End traces for released GPUs
            self._end_traces_for_gpu_ids(alloc.gpu_ids)
            self._state.idle_gpus |= set(alloc.gpu_ids)
            self._trace_active_gpus_update()
            # GPU Tracing: Instant marker for release
            self._trace_release_marker(release_cluster_id, alloc.gpu_ids)
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
            # Queue Tracing: Track enqueue AFTER append (depth is correct)
            self._trace_queue_enqueue(request_cluster_id, request_priority, request_lora_name)
            # GPU Tracing: Instant marker for successful enqueue
            self._trace_enqueue_marker(request_cluster_id, request_priority)
            self._wakeup_event.set()
        await event.wait()
        if pending is None:
            raise RuntimeError("release_then_request_gpus internal error: pending request not created")
        if pending.error is not None:
            raise RuntimeError(pending.error)
        return list(pending.result)

    def _signal_all_waiters_with_error(self, *, error: str) -> None:
        # Queue Tracing: Close all queue slices (track from stored state)
        for priority in Priority:
            for pending in list(self._state.pending_bucket(priority)):
                # Close slice before clearing
                self._trace_queue_slice_close(pending.request.cluster_id)
                pending.error = error
                pending.event.set()
            self._state.pending_bucket(priority).clear()
            # Queue Tracing: Single counter update to 0 after clear
            self._trace_queue_counter_update(priority, 0)
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
            remaining = float(metrics.get("remaining", 0))
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
                                SignalPendingAllocationOp(
                                    cluster_id=cluster_id,
                                    gpus_to_allocate=sorted(needed),
                                    priority=prio,
                                    # Carry lora_name directly from the pending so _apply_plan_and_signal
                                    # does not need to re-search the bucket for it.
                                    lora_name=pending.lora_name,
                                )
                            )

                # Phase 3: generation gap-ratio planning.
                planned_available_gpus -= non_gen_reserved_gpus

                active_dp_workers, inactive_dp_workers, idle_for_gen = self._snapshot_generation_dp_workers(
                    plan=plan, idle_gpus=set(planned_available_gpus)
                )

                # Unblock pending generation requests when any generation worker is active.
                # Previously this required ALL workers to be active, which caused a deadlock:
                # when another pipeline held a GPU needed by a dp worker, the GEN request
                # was never signaled, blocking the pipeline at release_then_request_gpus.
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
                # IMPORTANT: do not await coordinator RPCs while holding scheduler lock.
                resize_calls = self._prepare_resize_calls_locked(plan)
                # GPU Tracing: snapshot trace infos and plan details before releasing the lock.
                # dp_rank_to_gpus and pipeline_registry must be read while the lock is held.
                shrink_trace_infos = self._collect_shrink_trace_infos_locked(plan)
                expand_trace_infos = self._collect_expand_trace_infos_locked(plan)
                exec_details = self._plan_to_exec_details(plan)

            # GPU Tracing: Emit execution marker right after planning, before resize RPCs.
            if any([exec_details.get("shrinks"), exec_details.get("removes"),
                    exec_details.get("allocates"), exec_details.get("expands")]):
                self._trace_execution_marker(exec_details)
            self._maybe_flush_trace()

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
        info = self._state.pipeline_registry.get(pipeline_id)
        if info is None:
            raise RuntimeError(f"pipeline_id {pipeline_id!r} not registered")
        coordinator_namespace = info.get("namespace")
        if not isinstance(coordinator_namespace, str) or coordinator_namespace == "":
            raise RuntimeError(f"pipeline_id {pipeline_id!r} has invalid registered namespace {coordinator_namespace!r}")

        cached = self._coordinator_handle_cache.get(pipeline_id)
        if cached is not None:
            cached_namespace, cached_handle = cached
            if cached_namespace == coordinator_namespace:
                return cached_handle

        coordinator_name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"
        handle = get_actor_or_raise(
            coordinator_name, coordinator_namespace,
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

    def _collect_shrink_trace_infos_locked(self, plan: "ExecutionPlan") -> List[_GPUTraceInfo]:
        """Pre-collect GPU trace info for GPUs freed by shrink ops and cluster removals.

        Called under the scheduler lock so dp_rank_to_gpus is still intact.
        Only gpu_id is used at close time; the other fields satisfy the shared _GPUTraceInfo type.
        """
        infos: List[_GPUTraceInfo] = []
        for op in plan.sched_guided_shrink_ops:
            alloc = self._state.active_allocations.get(op.cluster_id)
            if alloc is None:
                continue
            pipeline_id, _ = parse_cluster_id(op.cluster_id)
            for dp_rank in op.dp_ranks_to_remove:
                bundle = set(alloc.dp_rank_to_gpus.get(dp_rank) or [])
                if not bundle:
                    bundle = self._reconstruct_bundle_for_dp_rank(
                        cluster_id=op.cluster_id, dp_rank=dp_rank
                    )
                for gpu_id in bundle:
                    infos.append(_GPUTraceInfo(
                        gpu_id=gpu_id, cluster_id=op.cluster_id,
                        pipeline_id=pipeline_id, dp_rank=dp_rank,
                    ))
        for cluster_id in plan.clusters_to_remove:
            alloc = self._state.active_allocations.get(cluster_id)
            if alloc is None:
                continue
            pipeline_id, _ = parse_cluster_id(cluster_id)
            for gpu_id in alloc.gpu_ids:
                infos.append(_GPUTraceInfo(
                    gpu_id=gpu_id, cluster_id=cluster_id, pipeline_id=pipeline_id, dp_rank=0,
                ))
        return infos

    def _collect_expand_trace_infos_locked(self, plan: "ExecutionPlan") -> List[_GPUTraceInfo]:
        """Pre-collect GPU trace info for proactive expand allocations.

        Called under the scheduler lock so pipeline_registry is stable.
        Mirrors the _start_gpu_trace loop in _apply_plan_and_signal (sched_guided_allocation_ops).
        """
        infos: List[_GPUTraceInfo] = []
        for op in plan.sched_guided_allocation_ops:
            if not op.gpus_to_allocate or not op.dp_ranks_to_add:
                continue
            pipeline_id, _ = parse_cluster_id(op.cluster_id)
            tp_size = int(
                self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"]
                .get("tp_size", 1)
            )
            sorted_gpus = sorted(op.gpus_to_allocate)
            for i, dp_rank in enumerate(sorted(op.dp_ranks_to_add)):
                bundle = sorted_gpus[i * tp_size : (i + 1) * tp_size]
                for gpu_id in bundle:
                    infos.append(_GPUTraceInfo(
                        gpu_id=gpu_id, cluster_id=op.cluster_id,
                        pipeline_id=pipeline_id, dp_rank=dp_rank,
                    ))
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
        shrink_trace_infos: List[_GPUTraceInfo],
        expand_trace_infos: List[_GPUTraceInfo],
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
            self._end_traces_for_gpu_ids([info.gpu_id for info in shrink_trace_infos])

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
            self._start_gpu_trace(
                info.gpu_id, info.cluster_id, info.pipeline_id,
                Priority.GENERATION, "proactive", [info.dp_rank],
            )

    async def _fail_fast_shutdown(self, *, reason: str) -> None:
        try:
            orchestrator = ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=RLIX_NAMESPACE)
        except Exception as e:
            sys.stderr.write(f"[rlix][ERROR] Failed to resolve orchestrator actor for shutdown: {type(e).__name__}: {e}\n")
            return
        try:
            orchestrator.shutdown.remote(force=True, reason=reason, source="scheduler")
        except Exception as e:
            sys.stderr.write(f"[rlix][ERROR] Failed to call orchestrator.shutdown: {type(e).__name__}: {e}\n")
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
        # adapted to RLix-standard progress reporting (percent_completed / step_target_trajectories).

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
            if iterations > _MAX_GAP_ITERATIONS or activations > _MAX_GAP_ACTIVATIONS:
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
                    bundle = self._reconstruct_bundle_for_dp_rank(
                        cluster_id=op.cluster_id, dp_rank=dp_rank
                    )
                alloc.active_dp_ranks.discard(dp_rank)
                alloc.dp_rank_to_gpus.pop(dp_rank, None)
                alloc.gpu_ids = [g for g in alloc.gpu_ids if g not in bundle]
                self._state.idle_gpus |= bundle
                self._trace_active_gpus_update()
                # Collect shrink detail for execution marker
                exec_shrinks.append({
                    "cluster_id": op.cluster_id,
                    "gpus_freed": sorted(bundle),
                    "dp_rank": dp_rank,
                })

        for cluster_id in plan.clusters_to_remove:
            alloc = self._state.active_allocations.pop(cluster_id, None)
            if alloc is not None:
                self._state.idle_gpus |= set(alloc.gpu_ids)
                self._trace_active_gpus_update()
                # Collect remove detail for execution marker
                exec_removes.append({
                    "cluster_id": cluster_id,
                    "gpus_freed": alloc.gpu_ids,
                })

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
            pipeline_id, cluster_name = parse_cluster_id(op.cluster_id)
            tp_size = int(self._state.pipeline_registry[pipeline_id]["cluster_configs"][cluster_name].get("tp_size", 1))
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
            self._trace_active_gpus_update()
            # GPU Tracing: Start traces for initial allocation
            if self._enable_gpu_tracing:
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
                    self._start_gpu_trace(
                        gpu_id, op.cluster_id, pipeline_id, priority,
                        "initial", dp_ranks, lora_name,
                    )
            self._signal_pending_request(cluster_id=op.cluster_id, priority=priority, result=sorted(op.gpus_to_allocate))
            # Collect allocate detail for execution marker
            exec_allocates.append({
                "cluster_id": op.cluster_id,
                "gpus_allocated": sorted(op.gpus_to_allocate),
                "priority": priority.name,
            })

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
            tp_size = int(self._state.pipeline_registry[pipeline_id]["cluster_configs"]["actor_infer"].get("tp_size", 1))
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
            self._trace_active_gpus_update()
            # GPU Tracing: proactive expand trace opens already happened in _execute_resize_calls
            # (right after expand RPCs completed, before this state commit).
            if op.has_pending_request:
                cluster_ids_to_signal.add(op.cluster_id)
            # Collect expand detail for execution marker
            exec_expands.append({
                "cluster_id": op.cluster_id,
                "gpus_allocated": sorted(op.gpus_to_allocate),
                "dp_ranks_added": sorted(op.dp_ranks_to_add),
            })
        for cluster_id in cluster_ids_to_signal:
            if self._has_pending_request_locked(cluster_id=cluster_id, priority=Priority.GENERATION):
                alloc = self._state.active_allocations[cluster_id]
                self._signal_pending_request(cluster_id=cluster_id, priority=Priority.GENERATION, result=sorted(alloc.gpu_ids))

        # Planned release requests: signal after shrink commit.
        for cluster_id, req in list(self._state.pending_planned_release_requests.items()):
            # If the cluster still exists, we assume shrink commit applied.
            req.event.set()
            self._state.pending_planned_release_requests.pop(cluster_id, None)

        return {
            "shrinks": exec_shrinks,
            "removes": exec_removes,
            "allocates": exec_allocates,
            "expands": exec_expands,
        }

    def _signal_pending_request(self, *, cluster_id: str, priority: Priority, result: Optional[List[int]] = None) -> None:
        bucket = self._state.pending_bucket(priority)
        for idx, pending in enumerate(bucket):
            if pending.request.cluster_id != cluster_id:
                continue
            # Queue Tracing: Close slice BEFORE pop (track from stored state)
            self._trace_queue_slice_close(cluster_id)
            # Pop from bucket
            bucket.pop(idx)
            # Queue Tracing: Update counter AFTER pop with correct depth
            self._trace_queue_counter_update(priority, len(bucket))
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

    async def notify_ready_to_release(
        self,
        *,
        pipeline_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        global_step: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> List[int]:
        """Blocking planned release API (RLix checklist).

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
