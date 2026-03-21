"""GPU timeline tracing subsystem for the RLix scheduler.

Extracted from SchedulerImpl to reduce class bloat.  SchedulerTracer owns all
Perfetto trace state (tracks, generator, flags) and never accesses scheduler
state directly — all scheduler-owned values are passed as method parameters.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar

from rlix.protocol.types import Priority
from rlix.scheduler.types import ExecutionPlan, parse_cluster_id

# GPU Tracing: Conditional import for tg4perfetto (may not be installed)
try:
    from tg4perfetto import TraceGenerator  # type: ignore[import-untyped]

    _TG4PERFETTO_AVAILABLE = True
except ImportError:
    TraceGenerator = None  # type: ignore[assignment,unused-ignore]
    _TG4PERFETTO_AVAILABLE = False

# GPU Tracing: Type-only imports for static type checking
if TYPE_CHECKING:
    from tg4perfetto import CounterTrack, Group, NormalTrack  # type: ignore[import-untyped,unused-ignore]
    from tg4perfetto._tgen import GroupTrack  # type: ignore[import-untyped]

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


@dataclass(frozen=True, slots=True)
class GPUTraceInfo:
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
class SchedulerTracer:
    """GPU timeline tracing subsystem for the scheduler.

    Owns all Perfetto trace state (tracks, generator, flags).  Never acquires the
    scheduler lock and never stores references to scheduler state.  All scheduler-owned
    values are passed explicitly as method parameters at each call site.
    """

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

    @property
    def enabled(self) -> bool:
        """Read-only access to the tracing-enabled flag."""
        return self._enable_gpu_tracing

    # =========================================================================
    # GPU Tracing: Core Methods
    # =========================================================================

    def safe_trace_call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Tuple[bool, Optional[T]]:
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

    def safe_final_flush(self) -> None:
        """Guarded flush for shutdown - not dependent on _enable_gpu_tracing.

        Called only from shutdown_tracing during explicit shutdown.
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

    def safe_trace(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Fire-and-forget trace call. Returns success status."""
        ok, _ = self.safe_trace_call(func, *args, **kwargs)
        return ok

    def safe_trace_get(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Optional[T]:
        """Trace call with return value. Returns result or None on failure."""
        _, result = self.safe_trace_call(func, *args, **kwargs)
        return result

    def get_or_create_gpu_track(self, gpu_id: int, *, required_gpus_per_node: Optional[int]) -> Optional["NormalTrack"]:
        """Get existing track or create new one. Returns None on failure."""
        if gpu_id in self._gpu_tracks:
            return self._gpu_tracks[gpu_id]

        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return None

        if required_gpus_per_node is None:
            return None

        node_id = gpu_id // required_gpus_per_node
        local_id = gpu_id % required_gpus_per_node
        track = self.safe_trace_get(
            self._scheduler_group.create_track,
            f"GPU{gpu_id}_{node_id}_{local_id}",
        )
        if track is not None:
            self._gpu_tracks[gpu_id] = track
        return track

    def init_gpu_tracks(self, *, num_gpus: int, required_gpus_per_node: int) -> None:
        """Eagerly create all GPU tracks for correct ordering in Perfetto UI.

        Track order in Perfetto is determined by creation order.
        """
        if not self._enable_gpu_tracing:
            return

        for gpu_id in range(num_gpus):
            self.get_or_create_gpu_track(gpu_id, required_gpus_per_node=required_gpus_per_node)

    def get_or_create_queue_group(self, priority: Priority) -> Optional[_QueueSubGroup]:
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
        raw_group = self.safe_trace_get(
            self._scheduler_group.create_group,
            f"Queue_{priority.value}_{key}",
        )
        if raw_group is None:
            logging.getLogger(__name__).debug(f"Failed to create queue sub-group for priority {key}")
            return None
        sub_group = _QueueSubGroup.from_group_track(raw_group)
        self._queue_groups[key] = sub_group
        return sub_group

    def get_or_create_queue_counter_track(self, priority: Priority) -> Optional["CounterTrack"]:
        """Get or create counter track for queue depth. Returns None on failure."""
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])
        if key in self._queue_counter_tracks:
            return self._queue_counter_tracks[key]

        queue_group = self.get_or_create_queue_group(priority)
        if queue_group is None:
            return None
        # create_counter_track is a method on _QueueSubGroup (our wrapper) — always exists
        track = self.safe_trace_get(
            queue_group.create_counter_track,
            "depth",
        )
        if track is not None:
            self._queue_counter_tracks[key] = track
        return track

    def get_or_create_active_gpus_counter(self) -> Optional["CounterTrack"]:
        """Get or create counter track for active GPU count. Returns None on failure."""
        if self._active_gpus_counter is not None:
            return self._active_gpus_counter

        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return None

        # Prefix "04_" forces alphabetical sort below marker tracks (01-03) and above GPU/Queue
        track = self.safe_trace_get(
            self._scheduler_group.create_counter_track,
            "04_active_gpus",
        )
        if track is not None:
            self._active_gpus_counter = track
        return track

    def init_active_gpus_counter(self) -> None:
        """Eagerly create active_gpus counter track for correct ordering in Perfetto UI.

        Perfetto sorts tracks alphabetically by name. The "04_" prefix places this counter
        below marker tracks ("01_"-"03_") and above GPU/Queue tracks (uppercase letters).
        """
        if not self._enable_gpu_tracing:
            return
        self.get_or_create_active_gpus_counter()

    def create_marker_track(self, name: str) -> Optional["NormalTrack"]:
        """Create a marker track with given name. Returns None on failure."""
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return None
        return self.safe_trace_get(self._scheduler_group.create_track, name)

    def init_enqueue_marker_track(self) -> None:
        """Eagerly create marker track for enqueue instant events.

        Perfetto sorts tracks alphabetically by name. The "01_" prefix places this track
        at the very top (first) in the Perfetto UI.
        """
        self._enqueue_marker_track = self.create_marker_track("01_enqueue_markers")

    def init_exec_marker_track(self) -> None:
        """Eagerly create marker track for exec instant events.

        Perfetto sorts tracks alphabetically by name. The "02_" prefix places this track
        second in the Perfetto UI, below enqueue_markers.
        """
        self._exec_marker_track = self.create_marker_track("02_exec_markers")

    def init_release_marker_track(self) -> None:
        """Eagerly create marker track for release instant events.

        Perfetto sorts tracks alphabetically by name. The "03_" prefix places this track
        third in the Perfetto UI, below exec_markers.
        """
        self._release_marker_track = self.create_marker_track("03_release_markers")

    def init_queue_tracks(self) -> None:
        """Eagerly create queue groups and counter tracks in priority order.

        Creates Queue_<N>_<KEY> groups. Perfetto sorts alphabetically, so the numeric
        prefix N (priority.value) ensures INIT(0) < TRN(1) < ... < GEN(6) in the UI.
        """
        if not self._enable_gpu_tracing:
            return
        # Create queue groups in priority order (lower value = higher priority)
        for priority in Priority:
            self.get_or_create_queue_group(priority)
            self.get_or_create_queue_counter_track(priority)

    def create_queue_slice_track(
        self, cluster_id: str, priority: Priority, lora_name: Optional[str] = None
    ) -> Optional["NormalTrack"]:
        """Create a per-cluster slice track for queue visualization.

        Returns None on failure.
        """
        queue_group = self.get_or_create_queue_group(priority)
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
        return self.safe_trace_get(
            queue_group.create_track,
            track_name,
        )

    def trace_queue_enqueue(
        self, cluster_id: str, priority: Priority, lora_name: Optional[str] = None, *, bucket_depth: int
    ) -> None:
        """Start queue slice and increment counter when request is enqueued.

        CRITICAL: Call AFTER pending_bucket().append() so bucket_depth reflects correct depth.
        CRITICAL: Track handle stored AFTER successful open to avoid orphan state.
        """
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return

        now_ns = time.time_ns()
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])

        # Create per-cluster track (lora_name flows into the track row label)
        slice_track = self.create_queue_slice_track(cluster_id, priority, lora_name)

        # Start slice FIRST
        if slice_track:
            label = f"[{key}] {cluster_id}"
            if lora_name:
                safe_lora = lora_name.replace("|", "_").replace(" ", "_")[:32]
                label += f" | lora:{safe_lora}"
            ok = self.safe_trace(slice_track.open, now_ns, label)
            # CRITICAL: Only store state AFTER successful open
            if ok:
                # Fail fast: duplicate cluster_id means scheduler violated one-request-per-cluster invariant
                if cluster_id in self._pending_queue_trace_state:
                    raise RuntimeError(f"Duplicate queue trace enqueue for cluster_id={cluster_id!r}")
                self._pending_queue_trace_state[cluster_id] = (slice_track, now_ns, priority)

        # Counter: depth = current bucket size (AFTER append, so correct)
        counter_track = self.get_or_create_queue_counter_track(priority)
        if counter_track:
            self.safe_trace(counter_track.count, now_ns, bucket_depth)

    def trace_queue_slice_close(self, cluster_id: str) -> None:
        """Close queue slice when request is fulfilled.

        CRITICAL:
        - Call BEFORE bucket.pop()
        - ALWAYS pops pending state (prevents leaks)
        - Uses stored track handle (no track creation on close)
        - Uses DIRECT close (not safe_trace) to work even when tracing disabled

        Note: Direct close is safe here because we have a valid track handle from
        successful open. On unexpected errors, disables tracing (consistent with safe_trace_call).
        """
        # ALWAYS pop entry first to prevent state leaks
        entry = self._pending_queue_trace_state.pop(cluster_id, None)
        if entry is None:
            return  # No pending trace state for this cluster

        stored_track, _, stored_priority = entry

        # Direct close - NOT via safe_trace, so works even if tracing disabled
        now_ns = time.time_ns()
        if stored_track is not None:
            try:
                stored_track.close(now_ns)
            except (IOError, OSError):
                # I/O errors are expected - ignore
                pass
            except Exception as e:
                # Unexpected error - log and disable tracing (consistent with safe_trace_call)
                logging.getLogger(__name__).warning(f"Queue trace close error, disabling tracing: {e}")
                self._enable_gpu_tracing = False

    def trace_queue_counter_update(self, priority: Priority, depth: int) -> None:
        """Update queue depth counter with explicit depth value.

        CRITICAL:
        - Call AFTER bucket.pop() with len(bucket) as depth
        - Depth comes from real bucket length, not cache
        - Separating this from slice_close allows correct ordering
        """
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return

        now_ns = time.time_ns()
        counter_track = self.get_or_create_queue_counter_track(priority)
        if counter_track:
            self.safe_trace(counter_track.count, now_ns, depth)

    def trace_active_gpus_update(self, *, num_gpus: Optional[int], idle_gpu_count: int) -> None:
        """Update active GPUs counter track with current utilization."""
        if not self._enable_gpu_tracing or self._scheduler_group is None:
            return

        if num_gpus is None:
            return  # Not initialized yet

        counter = self.get_or_create_active_gpus_counter()
        if counter:
            active_count = num_gpus - idle_gpu_count
            now_ns = time.time_ns()
            self.safe_trace(counter.count, now_ns, active_count)

    def shutdown_close_queue_slices(self) -> None:
        """Close all open queue slices during shutdown.

        CRITICAL:
        - Does NOT gate on _enable_gpu_tracing (called after it's False)
        - Uses stored track handles directly
        - Called BEFORE safe_final_flush()
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

    def build_trace_label(
        self,
        cluster_id: str,
        pipeline_id: str,
        priority: Priority,
        alloc_type: str,
        dp_ranks: Optional[List[int]] = None,
        lora_name: Optional[str] = None,
        *,
        cycle_counter: int,
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

        parts.extend([alloc_type, f"C{cycle_counter}"])
        label = " | ".join(parts)

        # DP only for generation clusters
        if dp_ranks and priority == Priority.GENERATION:
            label += f" | DP:{dp_ranks}"

        return label

    def end_traces_for_gpu_ids(self, gpu_ids: List[int]) -> None:
        """End trace slices for multiple GPUs. Reusable across release paths."""
        for gpu_id in gpu_ids:
            self.end_gpu_trace(gpu_id)

    def plan_to_exec_details(self, plan: ExecutionPlan) -> Dict[str, Any]:
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

    def trace_execution_marker(self, payload: Dict[str, Any], *, cycle_counter: int) -> None:
        """Record a per-cycle execution marker summarizing all GPU allocation changes."""
        if not self._enable_gpu_tracing or self._exec_marker_track is None:
            return
        self.safe_trace(
            self._exec_marker_track.instant,
            time.time_ns(),
            f"Exec C{cycle_counter}",
            kwargs=payload,
        )

    def trace_enqueue_marker(self, cluster_id: str, priority: Priority) -> None:
        """Record an instant marker when request is successfully enqueued."""
        if not self._enable_gpu_tracing or self._enqueue_marker_track is None:
            return
        key = _PRIORITY_SHORT.get(priority, priority.name[:3])
        self.safe_trace(
            self._enqueue_marker_track.instant,
            time.time_ns(),
            f"Enqueue: {key}",
            kwargs={"cluster_id": cluster_id, "priority": priority.name},
        )

    def trace_release_marker(self, cluster_id: str, gpus_released: List[int]) -> None:
        """Record an instant marker when GPUs are released via notify_release_gpus()."""
        if not self._enable_gpu_tracing or self._release_marker_track is None:
            return
        self.safe_trace(
            self._release_marker_track.instant,
            time.time_ns(),
            f"Release: {cluster_id}",
            kwargs={"cluster_id": cluster_id, "gpus_released": gpus_released},
        )

    def maybe_flush_trace(self) -> None:
        """Throttled flush - only flush if interval elapsed."""
        if not self._trace_gen:
            return
        now_ns = time.time_ns()
        if now_ns - self._trace_last_flush_ns >= self._trace_flush_interval_ns:
            ok, _ = self.safe_trace_call(self._trace_gen.flush)
            if ok:
                self._trace_last_flush_ns = now_ns

    def init_tracing(self, *, enable: bool, trace_output_dir: Optional[str]) -> None:
        """Initialize trace generator.

        Called once from initialize() when tracing should be enabled.

        Fail-fast policy:
        - If tg4perfetto not installed when tracing enabled: raise RuntimeError
        - If trace file creation fails: log warning, disable tracing
        - Runtime I/O errors during tracing: degrade gracefully (don't crash scheduler)

        NOTE: atexit handler is a best-effort fallback for non-Ray scenarios only.
        Ray terminates workers with SIGTERM/SIGKILL, which bypasses Python's atexit.
        The reliable path is explicit shutdown() via orchestrator integration.
        """
        self._enable_gpu_tracing = enable
        if not self._enable_gpu_tracing:
            return

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
            atexit.register(self.shutdown_tracing)
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

    def shutdown_tracing(self) -> None:
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
        self.shutdown_close_queue_slices()

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
        self.safe_final_flush()

        self._trace_gen = None
        # Keep _trace_file_path for post-shutdown assertions
        # Unregister atexit to prevent double-call
        try:
            atexit.unregister(self.shutdown_tracing)
        except Exception:
            pass

    def start_gpu_trace(
        self,
        gpu_id: int,
        cluster_id: str,
        pipeline_id: str,
        priority: Priority,
        alloc_type: str,
        dp_ranks: Optional[List[int]] = None,
        lora_name: Optional[str] = None,
        *,
        required_gpus_per_node: Optional[int],
        cycle_counter: int,
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
        track = self.get_or_create_gpu_track(gpu_id, required_gpus_per_node=required_gpus_per_node)
        if track is None:
            return

        # Use extracted helper for label building
        label = self.build_trace_label(
            cluster_id,
            pipeline_id,
            priority,
            alloc_type,
            dp_ranks,
            lora_name,
            cycle_counter=cycle_counter,
        )

        # Open slice
        self.safe_trace_call(track.open, time.time_ns(), label)

    def end_gpu_trace(self, gpu_id: int) -> None:
        """End a trace slice for GPU release.

        Removes context and closes the slice on the track.
        Silently handles missing context (may happen in edge cases).
        """
        if not self._enable_gpu_tracing:
            return

        track = self._gpu_tracks.get(gpu_id)
        if track:
            self.safe_trace(track.close, time.time_ns())
