from __future__ import annotations

from typing import List


def validate_partial_overlap_topology(
    train_devices: List[int],
    infer_devices: List[int],
    vllm_tp_size: int,
    megatron_tp: int,
    megatron_pp: int,
    megatron_cp: int,
    megatron_ep: int,
    async_grpo_enabled: bool,
) -> None:
    """Fail-fast validation of GPU topology for NeMo RL partial-overlap async GRPO.

    Raises AssertionError when the declared train/infer device mappings and
    Megatron/vLLM parallelism sizes cannot support partial overlap. Intended
    to run from NemoRLFullFinetunePipeline.initialize_pipeline() before RLix
    registration, so invalid topologies surface at pipeline startup rather
    than later during rollout.
    """
    train_set = set(train_devices)
    infer_set = set(infer_devices)
    infer_dp_size = len(infer_devices) // vllm_tp_size
    megatron_parallelism_product = megatron_tp * megatron_pp * megatron_cp * megatron_ep

    assert train_set.issubset(infer_set), "partial overlap requires train ⊂ infer"
    assert infer_dp_size >= 2, "partial overlap requires dp >= 2"
    assert async_grpo_enabled, "partial overlap requires async GRPO"
    assert len(train_devices) % megatron_parallelism_product == 0, (
        f"train device_mapping ({len(train_devices)}) must divide evenly by "
        f"tp*pp*cp*ep ({megatron_parallelism_product})"
    )
    assert len(infer_devices) % vllm_tp_size == 0, (
        f"infer device_mapping ({len(infer_devices)}) must divide evenly by vllm_tp_size ({vllm_tp_size})"
    )
    assert len(infer_set - train_set) >= vllm_tp_size, (
        "at least 1 full inference DP rank must stay active after shrink"
    )
