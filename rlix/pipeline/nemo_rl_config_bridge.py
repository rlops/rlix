from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ray

from rlix.protocol.types import get_pipeline_namespace


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


def _require_nemo_field(nemo_config: Any, dotted_path: str) -> Any:
    """Walk a dotted attribute path on *nemo_config*; raise ValueError if missing.

    Kept local to this module — unlike ``getattr(..., default)``, missing
    fields are a hard error because topology validation cannot proceed with
    silently-defaulted parallelism sizes.
    """
    current: Any = nemo_config
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            raise ValueError(
                f"nemo_config missing required field: {dotted_path}"
            )
        current = getattr(current, part)
    return current


def extract_topology_validation_inputs(*, nemo_config: Any) -> Dict[str, Any]:
    """Extract the 6 non-device inputs for :func:`validate_partial_overlap_topology`.

    Returned dict is meant to be ``**``-unpacked alongside ``train_devices``
    and ``infer_devices`` at the call site. Values are passed through as-is
    from the NeMo RL config — no type coercion, because the downstream
    validator's arithmetic will surface type errors naturally.

    Raises ValueError when any required field is absent from *nemo_config*,
    with the dotted path echoed in the message so stack-trace readers can
    locate the misconfigured key.
    """
    return {
        "vllm_tp_size": _require_nemo_field(
            nemo_config, "policy.generation.vllm_cfg.tensor_parallel_size"
        ),
        "megatron_tp": _require_nemo_field(
            nemo_config, "policy.megatron_cfg.tensor_model_parallel_size"
        ),
        "megatron_pp": _require_nemo_field(
            nemo_config, "policy.megatron_cfg.pipeline_model_parallel_size"
        ),
        "megatron_cp": _require_nemo_field(
            nemo_config, "policy.megatron_cfg.context_parallel_size"
        ),
        "megatron_ep": _require_nemo_field(
            nemo_config, "policy.megatron_cfg.expert_model_parallel_size"
        ),
        "async_grpo_enabled": _require_nemo_field(
            nemo_config, "grpo.async_grpo.enabled"
        ),
    }


def build_cluster_registry_inputs(
    *,
    nemo_config: Any,
    train_device_mapping: Optional[List[int]] = None,
    infer_device_mapping: Optional[List[int]] = None,
) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    """Build ``(cluster_tp_configs, cluster_device_mappings)`` for RLix pipeline registration.

    ``actor_train`` tp is canonicalized to 1 because Megatron workers each
    occupy a single GPU — intra-train parallelism is expressed via NCCL
    groups, not via RLix's tp field.

    Device mappings can be supplied two ways:

    1. As explicit ``train_device_mapping`` / ``infer_device_mapping``
       kwargs — this is the preferred path when the pipeline driver is
       the source of truth.
    2. As ``nemo_config.rlix.train_device_mapping`` /
       ``nemo_config.rlix.infer_device_mapping`` — a fallback for configs
       that carry the device lists directly.

    kwargs take precedence: if a kwarg is not ``None``, it is used
    verbatim and the config subtree is ignored. A kwarg of ``None``
    triggers the config fallback; an absent or ``None`` config value
    then raises. An explicitly empty list (``[]``) on either source is
    rejected by the non-empty check below — use ``None`` to mean "not
    provided, try the other source".

    Raises ValueError on empty device mappings, non-positive vllm tp,
    an infer-count not divisible by vllm tp, or when both the kwarg and
    the config fallback are absent for either device mapping.
    """
    vllm_tp = _require_nemo_field(
        nemo_config, "policy.generation.vllm_cfg.tensor_parallel_size"
    )
    if train_device_mapping is None:
        train_device_mapping = getattr(
            getattr(nemo_config, "rlix", None), "train_device_mapping", None
        )
        if train_device_mapping is None:
            raise ValueError(
                "train_device_mapping must be provided via kwarg or "
                "nemo_config.rlix.train_device_mapping"
            )
    if infer_device_mapping is None:
        infer_device_mapping = getattr(
            getattr(nemo_config, "rlix", None), "infer_device_mapping", None
        )
        if infer_device_mapping is None:
            raise ValueError(
                "infer_device_mapping must be provided via kwarg or "
                "nemo_config.rlix.infer_device_mapping"
            )
    if not train_device_mapping:
        raise ValueError("nemo_config train_device_mapping must be non-empty")
    if not infer_device_mapping:
        raise ValueError("nemo_config infer_device_mapping must be non-empty")
    if vllm_tp <= 0:
        raise ValueError(
            f"NeMo RL vllm tensor_parallel_size must be positive, got: {vllm_tp}"
        )
    if len(infer_device_mapping) % vllm_tp != 0:
        raise ValueError(
            f"NeMo RL infer_device_mapping length must divide evenly by vllm "
            f"tensor_parallel_size, got: len(infer)={len(infer_device_mapping)} "
            f"vllm_tp={vllm_tp}"
        )
    cluster_tp_configs: Dict[str, int] = {
        "actor_train": 1,
        "actor_infer": vllm_tp,
    }
    cluster_device_mappings: Dict[str, List[int]] = {
        "actor_train": list(train_device_mapping),
        "actor_infer": list(infer_device_mapping),
    }
    return cluster_tp_configs, cluster_device_mappings


def detect_pipeline_type(*, nemo_config: Any) -> str:
    """Return ``"lora"`` when NeMo RL PEFT is enabled, else ``"ft"``.

    Uses chained :func:`getattr` with ``None`` defaults so absent
    ``policy`` / ``megatron_cfg`` / ``peft`` nodes fall through to the
    full-finetune branch without raising — matches ROLL-side behavior in
    :mod:`examples.start_multi_pipeline_test`.

    Truthy-coerces ``peft.enabled`` rather than identity-checking against
    ``True``, so YAML-derived non-bool truthy values still map to
    ``"lora"``.
    """
    policy = getattr(nemo_config, "policy", None)
    megatron_cfg = getattr(policy, "megatron_cfg", None)
    peft = getattr(megatron_cfg, "peft", None)
    enabled = getattr(peft, "enabled", False)
    return "lora" if bool(enabled) else "ft"


@dataclass(frozen=True)
class NemoRlRegistrationResult:
    """Result of :func:`register_nemo_rl_pipeline`'s 3-step orchestrator dance.

    ``scheduler`` is the Ray actor handle returned by the orchestrator's
    ``AdmitResponse``, required by NeMo RL child actors (e.g.
    ``AsyncTrajectoryCollector``, ``ReplayBuffer``, ``ModelUpdateService``)
    to issue GPU allocation requests.
    """

    pipeline_id: str
    ray_namespace: str
    scheduler: Any


def register_nemo_rl_pipeline(
    *,
    orchestrator: Any,
    nemo_config: Any,
    train_device_mapping: Optional[List[int]] = None,
    infer_device_mapping: Optional[List[int]] = None,
) -> NemoRlRegistrationResult:
    """Run the RLix 3-step pipeline registration dance for a NeMo RL pipeline.

    ``train_device_mapping`` and ``infer_device_mapping`` are optional;
    when ``None`` they fall back to ``nemo_config.rlix.train_device_mapping``
    / ``nemo_config.rlix.infer_device_mapping`` — see
    :func:`build_cluster_registry_inputs` for the full precedence rules.

    Flow:
      1. Detect pipeline type (``"ft"``/``"lora"``) via
         :func:`detect_pipeline_type`.
      2. ``orchestrator.allocate_pipeline_id.remote(pipeline_type)`` → id.
      3. Build ``cluster_tp_configs`` / ``cluster_device_mappings`` via
         :func:`build_cluster_registry_inputs`.
      4. ``orchestrator.register_pipeline.remote(...)``.
      5. ``orchestrator.admit_pipeline.remote(pipeline_id=...)`` →
         ``AdmitResponse`` whose ``scheduler`` handle is propagated to the
         caller.

    Errors from any of the three orchestrator calls propagate unchanged —
    matches ROLL's ``examples/start_multi_pipeline_test.py`` fail-fast
    pattern and leaves any partial orchestrator state for post-mortem.

    Raises RuntimeError when ``admit_pipeline`` returns ``scheduler=None``:
    that only happens when the pipeline is not registered on the
    orchestrator side, which should be impossible immediately after a
    successful ``register_pipeline`` — indicates orchestrator-state
    corruption worth surfacing loudly.
    """
    pipeline_type = detect_pipeline_type(nemo_config=nemo_config)
    pipeline_id: str = ray.get(
        orchestrator.allocate_pipeline_id.remote(pipeline_type)
    )

    ray_namespace = get_pipeline_namespace(pipeline_id)
    cluster_tp_configs, cluster_device_mappings = build_cluster_registry_inputs(
        nemo_config=nemo_config,
        train_device_mapping=train_device_mapping,
        infer_device_mapping=infer_device_mapping,
    )
    ray.get(
        orchestrator.register_pipeline.remote(
            pipeline_id=pipeline_id,
            ray_namespace=ray_namespace,
            cluster_tp_configs=cluster_tp_configs,
            cluster_device_mappings=cluster_device_mappings,
        )
    )
    admit_response = ray.get(
        orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id)
    )
    if admit_response.scheduler is None:
        raise RuntimeError(
            f"NeMo RL pipeline registration: orchestrator.admit_pipeline "
            f"returned scheduler=None for pipeline_id={pipeline_id!r}; "
            f"indicates the pipeline is not registered on the orchestrator "
            f"side despite a successful register_pipeline call (possible "
            f"orchestrator-state corruption)."
        )
    return NemoRlRegistrationResult(
        pipeline_id=pipeline_id,
        ray_namespace=ray_namespace,
        scheduler=admit_response.scheduler,
    )
