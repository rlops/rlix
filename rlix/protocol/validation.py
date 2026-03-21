from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from rlix.protocol.types import ALL_CLUSTER_NAMES, GENERATION_CLUSTER_NAME, REWARD_CLUSTER_NAME

REQUEST_ID_DELIMITER = ":"


def validate_pipeline_id(pipeline_id: str) -> None:
    if not isinstance(pipeline_id, str):
        raise ValueError(f"pipeline_id must be str, got {type(pipeline_id).__name__}")
    if pipeline_id == "":
        raise ValueError("pipeline_id must be non-empty")
    if REQUEST_ID_DELIMITER in pipeline_id:
        raise ValueError(f"pipeline_id must not contain {REQUEST_ID_DELIMITER!r}: {pipeline_id!r}")


@dataclass(frozen=True, slots=True)
class RegisterValidationInput:
    pipeline_id: str
    ray_namespace: str
    cluster_tp_configs: Dict[str, int]
    cluster_device_mappings: Dict[str, List[int]]


def validate_register_pipeline(inp: RegisterValidationInput) -> None:
    """Validate all inputs for pipeline registration.

    Checks are ordered so that structural issues (types, emptiness, key mismatches) are
    caught before semantic issues (tp_size divisibility). This means the first error a
    caller sees is always the most fundamental one.
    """
    # --- Top-level shape checks ---

    validate_pipeline_id(inp.pipeline_id)

    # Each pipeline runs in its own Ray namespace for actor isolation.
    if not isinstance(inp.ray_namespace, str) or inp.ray_namespace == "":
        raise ValueError("ray_namespace must be non-empty str")

    # A pipeline with zero clusters is meaningless.
    if not isinstance(inp.cluster_tp_configs, dict) or not inp.cluster_tp_configs:
        raise ValueError("cluster_tp_configs must be non-empty dict[str,int]")

    # The scheduler indexes into ["actor_infer"] without None guards in ~15 places,
    # so every pipeline must declare a generation cluster.
    if GENERATION_CLUSTER_NAME not in inp.cluster_tp_configs:
        raise ValueError(f"{GENERATION_CLUSTER_NAME} cluster must be registered")

    # Every cluster needs a GPU mapping entry (empty list is valid for CPU-only clusters like reward).
    if not isinstance(inp.cluster_device_mappings, dict) or not inp.cluster_device_mappings:
        raise ValueError("cluster_device_mappings must be non-empty dict[str,list[int]]")

    # The two dicts must have identical key sets — a tp_size without a device_mapping
    # (or vice versa) would cause silent misconfigurations downstream.
    #   OK:  tp_configs={"actor_infer", "actor_train"}, device_mappings={"actor_infer", "actor_train"}
    #   BAD: tp_configs={"actor_infer", "actor_train"}, device_mappings={"actor_infer"}  (missing actor_train mapping)
    if set(inp.cluster_tp_configs.keys()) != set(inp.cluster_device_mappings.keys()):
        missing_tp = sorted(set(inp.cluster_device_mappings.keys()) - set(inp.cluster_tp_configs.keys()))
        missing_map = sorted(set(inp.cluster_tp_configs.keys()) - set(inp.cluster_device_mappings.keys()))
        raise ValueError(
            f"cluster config mismatch: missing tp_size for {missing_tp}, missing device_mapping for {missing_map}"
        )

    # --- Per-cluster checks ---
    for cluster_name, tp_size_raw in inp.cluster_tp_configs.items():
        if cluster_name not in ALL_CLUSTER_NAMES:
            raise ValueError(f"Unknown cluster name {cluster_name!r}. Must be one of {sorted(ALL_CLUSTER_NAMES)!r}.")
        # Coerce to int to handle numeric strings; fail fast on non-numeric values.
        try:
            tp_size = int(tp_size_raw)
        except Exception as e:
            raise ValueError(f"tp_size must be int for cluster {cluster_name!r}, got {tp_size_raw!r}") from e
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0 for cluster {cluster_name!r}, got {tp_size!r}")

        device_mapping = list(inp.cluster_device_mappings.get(cluster_name) or [])

        # All clusters require GPUs except reward, which is CPU-only.
        if not device_mapping and cluster_name != REWARD_CLUSTER_NAME:
            raise ValueError(f"device_mapping must be non-empty for cluster {cluster_name!r}")
        if cluster_name == REWARD_CLUSTER_NAME and device_mapping:
            # TODO: support GPU reward clusters (currently restricted to CPU-only).
            raise ValueError("reward cluster only supports CPU-only mode: reward.device_mapping must be empty")

        # No duplicate GPU IDs within a single cluster.
        if device_mapping and len(device_mapping) != len(set(device_mapping)):
            raise ValueError(f"device_mapping has duplicates for cluster {cluster_name!r}")

        for gpu in device_mapping:
            if not isinstance(gpu, int):
                raise ValueError(
                    f"device_mapping must be list[int] for cluster {cluster_name!r}, got {type(gpu).__name__}"
                )

        # GPUs must split evenly into DP workers — each worker gets exactly tp_size GPUs
        # for tensor parallelism.
        #   OK:  device_mapping=[0,1,2,3], tp_size=2  → 2 DP workers, each with 2 GPUs
        #   BAD: device_mapping=[0,1,2],   tp_size=2  → 1.5 workers, cannot split evenly
        if device_mapping and len(device_mapping) % tp_size != 0:
            raise ValueError(
                f"cluster {cluster_name!r} has len(device_mapping)={len(device_mapping)} not divisible by tp_size={tp_size}"
            )
