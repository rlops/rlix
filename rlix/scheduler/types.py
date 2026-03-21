"""Scheduler-internal types and cluster_id utilities.

Defines the data structures used within the scheduler's planning and execution loop:
request/response types, execution plan operations, and cluster_id parsing.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from rlix.protocol.types import GENERATION_CLUSTER_NAME, GPU_CLUSTER_NAMES, Priority
from rlix.protocol.validation import validate_pipeline_id


@dataclass(slots=True)
class ClusterAllocation:
    """Active GPU allocation for a cluster (a pipeline's named compute group).

    A cluster_id has the format ``{pipeline_id}_{cluster_name}`` where cluster_name
    is one of: ``actor_train``, ``actor_infer``, ``critic``, ``reference``.
    """

    cluster_id: str
    gpu_ids: List[int]
    priority: Priority
    active_dp_ranks: Set[int] = field(default_factory=set)
    dp_rank_to_gpus: Dict[int, List[int]] = field(default_factory=dict)
    global_step: Optional[int] = None
    timestamp: Optional[float] = None


class ValidationError(RuntimeError):
    """Raised when the execution plan violates a scheduler invariant.

    Carries a ``condition`` number (which validation check failed) and a ``context``
    dict for debugging. Both are included in the string representation.
    """

    def __init__(
        self,
        message: str,
        *,
        condition: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.condition = condition
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


@dataclass(slots=True)
class SchedGuidedShrinkOp:
    """Instruction to remove DP ranks from a cluster (scheduler-initiated shrink)."""

    cluster_id: str
    dp_ranks_to_remove: List[int]


@dataclass(slots=True)
class SignalPendingAllocationOp:
    """Instruction to satisfy a pending allocation request with specific GPUs."""

    cluster_id: str
    gpus_to_allocate: List[int]
    priority: Optional[Any] = None
    lora_name: Optional[str] = None  # GPU Tracing: carried from PendingRequest at planning time
    tp_size: int = 1  # Snapshotted at planning time; avoids registry lookup at commit


@dataclass(slots=True)
class SchedGuidedAllocationOp:
    """Instruction to expand a cluster by adding DP ranks onto specific GPUs."""

    cluster_id: str
    dp_ranks_to_add: List[int]
    gpus_to_allocate: List[int]
    has_pending_request: bool = False
    tp_size: int = 1  # Snapshotted at planning time; avoids registry lookup at commit


@dataclass(slots=True)
class ExecutionPlan:
    """The complete set of operations for one scheduling cycle."""

    sched_guided_shrink_ops: List[SchedGuidedShrinkOp] = field(default_factory=list)
    signal_pending_allocation_ops: List[SignalPendingAllocationOp] = field(default_factory=list)
    sched_guided_allocation_ops: List[SchedGuidedAllocationOp] = field(default_factory=list)
    clusters_to_remove: Set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class Request:
    """Immutable descriptor for a GPU allocation or release request."""

    cluster_id: str
    priority: Priority
    timestamp: float


@dataclass(slots=True)
class PendingRequest:
    """A GPU allocation request awaiting fulfillment by the scheduler loop.

    The caller awaits ``event``; the scheduler sets ``result`` (allocated GPU IDs)
    or ``error`` before signaling.
    """

    request: Request
    event: asyncio.Event
    global_step: Optional[int] = None
    lora_name: Optional[str] = None  # GPU tracing: LoRA name for non-generation clusters
    result: List[int] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(slots=True)
class PendingPlannedReleaseRequest:
    """A planned GPU release awaiting execution by the scheduler loop.

    The caller awaits ``event``; the scheduler sets ``error`` before signaling
    on failure.
    """

    cluster_id: str
    dp_ranks_to_remove: List[int]
    event: asyncio.Event
    global_step: Optional[int] = None
    error: Optional[str] = None


def is_generation_cluster(cluster_id: str) -> bool:
    """Return True if the cluster is a generation (rollout) cluster."""
    return cluster_id.endswith(f"_{GENERATION_CLUSTER_NAME}")


_MAX_CLUSTER_ID_LEN = 256
_CLUSTER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_cluster_id(cluster_id: str) -> None:
    """Validate cluster_id format: non-empty, bounded length, alphanumeric with ``_-``."""
    if not isinstance(cluster_id, str) or cluster_id == "":
        raise ValueError("cluster_id must be non-empty str")
    if len(cluster_id) > _MAX_CLUSTER_ID_LEN:
        raise ValueError(f"cluster_id too long: {len(cluster_id)} > {_MAX_CLUSTER_ID_LEN}")
    if _CLUSTER_ID_PATTERN.match(cluster_id) is None:
        raise ValueError(f"cluster_id contains invalid characters: {cluster_id!r}")


def parse_cluster_id(cluster_id: str) -> tuple[str, str]:
    """Suffix-aware parser for cluster_id.

    Fail-fast (H2): do not fall back to parsing by `rsplit("_", 1)`, since pipeline_id may
    contain underscores and cluster_name may evolve. Only accept known suffixes.
    """
    validate_cluster_id(cluster_id)

    for cluster_name in GPU_CLUSTER_NAMES:
        suffix = f"_{cluster_name}"
        if cluster_id.endswith(suffix):
            pipeline_id = cluster_id[: -len(suffix)]
            validate_pipeline_id(pipeline_id)
            return pipeline_id, cluster_name

    raise ValueError(
        f"Unrecognized cluster_id {cluster_id!r}. Expected suffix _<cluster_name> where cluster_name is one of "
        f"{sorted(GPU_CLUSTER_NAMES)!r}."
    )


def build_dp_rank_mapping(gpu_ids: List[int], tp_size: int) -> Dict[int, List[int]]:
    """Map DP ranks to GPU ID groups based on tensor-parallel size.

    Given ``gpu_ids=[0,1,2,3]`` and ``tp_size=2``, returns ``{0: [0,1], 1: [2,3]}``.
    """
    if tp_size <= 0:
        return {}
    sorted_gpus = sorted(gpu_ids)
    mapping: Dict[int, List[int]] = {}
    for i in range(0, len(sorted_gpus), tp_size):
        dp_rank = i // tp_size
        mapping[dp_rank] = sorted_gpus[i : i + tp_size]
    return mapping
