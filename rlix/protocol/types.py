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

# Cluster name constants.
# "reward" is CPU-only: valid in registration configs but never appears in cluster_ids
# or GPU scheduling.
ACTOR_TRAIN_CLUSTER_NAME: str = "actor_train"
GENERATION_CLUSTER_NAME: str = "actor_infer"
CRITIC_CLUSTER_NAME: str = "critic"
REFERENCE_CLUSTER_NAME: str = "reference"
REWARD_CLUSTER_NAME: str = "reward"
ALL_CLUSTER_NAMES: tuple[str, ...] = (
    ACTOR_TRAIN_CLUSTER_NAME,
    GENERATION_CLUSTER_NAME,
    CRITIC_CLUSTER_NAME,
    REFERENCE_CLUSTER_NAME,
    REWARD_CLUSTER_NAME,
)
GPU_CLUSTER_NAMES: tuple[str, ...] = (
    ACTOR_TRAIN_CLUSTER_NAME,
    GENERATION_CLUSTER_NAME,
    CRITIC_CLUSTER_NAME,
    REFERENCE_CLUSTER_NAME,
)


def get_pipeline_namespace(pipeline_id: str) -> str:
    """Canonical Ray namespace for a per-pipeline coordinator actor."""
    return f"pipeline_{pipeline_id}_NS"


@dataclass(frozen=True, slots=True)
class ActionResponse:
    success: bool


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
    step_target_trajectories: int
    fifo_timestamp: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
