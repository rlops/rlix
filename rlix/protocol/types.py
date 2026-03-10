from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Ray namespace and actor name protocol constants shared across rlix modules.
RLIX_NAMESPACE: str = "rlix"
SCHEDULER_ACTOR_NAME: str = "rlix:scheduler"
ORCHESTRATOR_ACTOR_NAME: str = "rlix:orchestrator"
RESOURCE_MANAGER_ACTOR_NAME: str = "rlix:resource_manager"
# Prefix for per-pipeline coordinator actors: full name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"
COORDINATOR_ACTOR_NAME_PREFIX: str = "rlix:coordinator:"
# Prefix for per-pipeline coordinator actors: full name = f"{PIPELINE_ACTOR_NAME_PREFIX}{pipeline_id}"
PIPELINE_ACTOR_NAME_PREFIX: str = "rlix:pipeline:"
# Name for the ROLL-specific ResourceManager singleton actor (used when RLIX_CONTROL_PLANE=rlix)
ROLL_RESOURCE_MANAGER_ACTOR_NAME: str = "rlix:roll_resource_manager"


def get_pipeline_namespace(pipeline_id: str) -> str:
    """Canonical Ray namespace for a per-pipeline coordinator actor."""
    return f"pipeline_{pipeline_id}_NS"


@dataclass(frozen=True, slots=True)
class ActionResponse:
    success: bool
    error: Optional[str] = None


class Priority(enum.IntEnum):
    """7-tier priority system for GPU allocation (lower numeric value = higher priority)."""

    INITIALIZATION = 0
    ACTOR_TRAINING = 1
    CRITIC_TRAINING = 2
    OLD_LOG_PROBS = 3
    REF_LOG_PROBS = 4
    VALUE_COMPUTE = 5
    GENERATION = 6


@dataclass(frozen=True, slots=True)
class ProgressReport:
    pipeline_id: str
    queued_trajectories: int
    inflight_trajectories: int
    step_target_trajectories: int
    percent_completed: float = 0.0
    oldest_unfinished_creation_ts: Optional[float] = None
    active_base_version: int = 0
    fifo_timestamp: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
