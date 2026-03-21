from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from rlix.protocol.types import Priority, ProgressReport
from rlix.scheduler.types import ClusterAllocation, PendingPlannedReleaseRequest, PendingRequest


@dataclass(slots=True)
class SchedulerState:
    pending_requests: Dict[Priority, List[PendingRequest]] = field(default_factory=dict)
    active_allocations: Dict[str, ClusterAllocation] = field(default_factory=dict)  # cluster_id -> allocation
    idle_gpus: Set[int] = field(default_factory=set)

    pending_planned_release_requests: Dict[str, PendingPlannedReleaseRequest] = field(
        default_factory=dict
    )  # cluster_id -> request

    pipeline_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # pipeline_id -> info

    # Keep latest snapshot per pipeline/mode/stream:
    #   latest_progress_by_pipeline[pipeline_id][mode][stream_key] = ProgressReport
    # where stream_key is lora_id for LoRA streams, or a reserved key for full-finetune.
    latest_progress_by_pipeline: Dict[str, Dict[str, Dict[str, ProgressReport]]] = field(default_factory=dict)

    def pending_bucket(self, priority: Priority) -> List[PendingRequest]:
        bucket = self.pending_requests.get(priority)
        if bucket is None:
            bucket = []
            self.pending_requests[priority] = bucket
        return bucket
