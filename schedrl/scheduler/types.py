from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from schedrl.protocol.request_id import validate_pipeline_id
from schedrl.protocol.types import Priority


@dataclass(slots=True)
class ClusterAllocation:
    """Active GPU allocation for a cluster_id (format: '{pipeline_id}_{cluster_name}')."""

    cluster_id: str
    gpu_ids: List[int]
    priority: Priority
    active_dp_ranks: Set[int] = field(default_factory=set)
    dp_rank_to_gpus: Dict[int, List[int]] = field(default_factory=dict)
    global_step: Optional[int] = None
    timestamp: Optional[float] = None
    lora_name: Optional[str] = None  # GPU tracing: LoRA adapter name for non-generation clusters


class ValidationError(RuntimeError):
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
    cluster_id: str
    dp_ranks_to_remove: List[int]


@dataclass(slots=True)
class SignalPendingAllocationOp:
    cluster_id: str
    gpus_to_allocate: List[int]
    priority: Optional[Any] = None


@dataclass(slots=True)
class SchedGuidedAllocationOp:
    cluster_id: str
    dp_ranks_to_add: List[int]
    gpus_to_allocate: List[int]
    has_pending_request: bool = False


@dataclass(slots=True)
class ExecutionPlan:
    sched_guided_shrink_ops: List[SchedGuidedShrinkOp] = field(default_factory=list)
    signal_pending_allocation_ops: List[SignalPendingAllocationOp] = field(default_factory=list)
    sched_guided_allocation_ops: List[SchedGuidedAllocationOp] = field(default_factory=list)
    clusters_to_remove: Set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class Request:
    cluster_id: str
    priority: Priority
    timestamp: float


@dataclass(slots=True)
class PendingRequest:
    request: Request
    event: asyncio.Event
    global_step: Optional[int] = None
    lora_name: Optional[str] = None  # GPU tracing: LoRA adapter name for non-generation clusters
    result: List[int] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(slots=True)
class PendingPlannedReleaseRequest:
    cluster_id: str
    dp_ranks_to_remove: List[int]
    event: asyncio.Event
    global_step: Optional[int] = None
    result_released_gpu_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None


def is_generation_cluster(cluster_id: str) -> bool:
    return cluster_id.endswith("_actor_infer")

_MAX_CLUSTER_ID_LEN = 256
_CLUSTER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_cluster_id(cluster_id: str) -> None:
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

    known_clusters = {"actor_train", "actor_infer", "critic", "reference"}
    for cluster_name in known_clusters:
        suffix = f"_{cluster_name}"
        if cluster_id.endswith(suffix):
            pipeline_id = cluster_id[: -len(suffix)]
            validate_pipeline_id(pipeline_id)
            return pipeline_id, cluster_name

    raise ValueError(
        f"Unrecognized cluster_id {cluster_id!r}. Expected suffix _<cluster_name> where cluster_name is one of "
        f"{sorted(known_clusters)!r}."
    )


def build_dp_rank_mapping(gpu_ids: List[int], tp_size: int) -> Dict[int, List[int]]:
    if tp_size <= 0:
        return {}
    sorted_gpus = sorted(gpu_ids)
    mapping: Dict[int, List[int]] = {}
    for i in range(0, len(sorted_gpus), tp_size):
        dp_rank = i // tp_size
        mapping[dp_rank] = sorted_gpus[i : i + tp_size]
    return mapping
