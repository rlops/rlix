"""Tests for rlix.pipeline.nemo_rl_config_bridge builder functions.

Covers extract_topology_validation_inputs, build_cluster_registry_inputs,
and detect_pipeline_type. Topology *validator* tests live in
test_nemo_rl_config_bridge.py.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.pipeline": RLIX_ROOT / "pipeline",
        "rlix.protocol": RLIX_ROOT / "protocol",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_bridge(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    return importlib.import_module("rlix.pipeline.nemo_rl_config_bridge")


_SENTINEL = object()


def _make_nemo_config(
    *,
    vllm_tp: object = 2,
    meg_tp: object = 1,
    meg_pp: object = 1,
    meg_cp: object = 1,
    meg_ep: object = 1,
    async_grpo: object = True,
    peft_enabled: object = _SENTINEL,
    drop_peft: bool = False,
    drop_megatron_cfg: bool = False,
    drop_policy: bool = False,
    drop_grpo: bool = False,
    drop_async_grpo: bool = False,
    rlix_train_device_mapping: object = _SENTINEL,
    rlix_infer_device_mapping: object = _SENTINEL,
) -> SimpleNamespace:
    """Construct a minimal nested SimpleNamespace mimicking a NeMo RL config.

    drop_* flags remove whole branches so missing-field error paths can be
    exercised without pulling in omegaconf.
    """
    cfg = SimpleNamespace()
    if not drop_policy:
        vllm_cfg = SimpleNamespace(tensor_parallel_size=vllm_tp)
        generation = SimpleNamespace(vllm_cfg=vllm_cfg)
        if drop_megatron_cfg:
            cfg.policy = SimpleNamespace(generation=generation)
        else:
            megatron_cfg = SimpleNamespace(
                tensor_model_parallel_size=meg_tp,
                pipeline_model_parallel_size=meg_pp,
                context_parallel_size=meg_cp,
                expert_model_parallel_size=meg_ep,
            )
            if not drop_peft and peft_enabled is not _SENTINEL:
                megatron_cfg.peft = SimpleNamespace(enabled=peft_enabled)
            cfg.policy = SimpleNamespace(
                generation=generation, megatron_cfg=megatron_cfg
            )
    if not drop_grpo:
        if drop_async_grpo:
            cfg.grpo = SimpleNamespace()
        else:
            cfg.grpo = SimpleNamespace(
                async_grpo=SimpleNamespace(enabled=async_grpo)
            )
    rlix_fields: dict = {}
    if rlix_train_device_mapping is not _SENTINEL:
        rlix_fields["train_device_mapping"] = rlix_train_device_mapping
    if rlix_infer_device_mapping is not _SENTINEL:
        rlix_fields["infer_device_mapping"] = rlix_infer_device_mapping
    if rlix_fields:
        cfg.rlix = SimpleNamespace(**rlix_fields)
    return cfg


class TestBuildClusterRegistryInputs:
    def test_happy_path_returns_expected_structure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        tp, devs = bridge.build_cluster_registry_inputs(
            nemo_config=_make_nemo_config(vllm_tp=2),
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        assert tp == {"actor_train": 1, "actor_infer": 2}
        assert devs == {"actor_train": [0, 1], "actor_infer": [0, 1, 2, 3]}

    def test_actor_train_tp_hardcoded_to_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        tp, _ = bridge.build_cluster_registry_inputs(
            nemo_config=_make_nemo_config(vllm_tp=4),
            train_device_mapping=[0, 1, 2, 3],
            infer_device_mapping=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        assert tp["actor_train"] == 1

    def test_infer_tp_sourced_from_vllm_cfg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        tp, _ = bridge.build_cluster_registry_inputs(
            nemo_config=_make_nemo_config(vllm_tp=8),
            train_device_mapping=[0],
            infer_device_mapping=list(range(8)),
        )
        assert tp["actor_infer"] == 8

    def test_device_mappings_are_copied_not_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        train = [0, 1]
        infer = [0, 1, 2, 3]
        _, devs = bridge.build_cluster_registry_inputs(
            nemo_config=_make_nemo_config(vllm_tp=2),
            train_device_mapping=train,
            infer_device_mapping=infer,
        )
        train.append(99)
        infer.append(99)
        assert devs["actor_train"] == [0, 1]
        assert devs["actor_infer"] == [0, 1, 2, 3]

    def test_empty_train_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bridge = _load_bridge(monkeypatch)
        with pytest.raises(
            ValueError, match=r"nemo_config train_device_mapping must be non-empty"
        ):
            bridge.build_cluster_registry_inputs(
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[],
                infer_device_mapping=[0, 1],
            )

    def test_empty_infer_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bridge = _load_bridge(monkeypatch)
        with pytest.raises(
            ValueError, match=r"nemo_config infer_device_mapping must be non-empty"
        ):
            bridge.build_cluster_registry_inputs(
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[0],
                infer_device_mapping=[],
            )

    @pytest.mark.parametrize("bad_tp", [0, -1, -4])
    def test_non_positive_vllm_tp_raises(
        self, monkeypatch: pytest.MonkeyPatch, bad_tp: int
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        with pytest.raises(
            ValueError,
            match=r"NeMo RL vllm tensor_parallel_size must be positive",
        ):
            bridge.build_cluster_registry_inputs(
                nemo_config=_make_nemo_config(vllm_tp=bad_tp),
                train_device_mapping=[0],
                infer_device_mapping=[0, 1],
            )

    def test_infer_not_divisible_by_vllm_tp_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        with pytest.raises(
            ValueError,
            match=r"NeMo RL infer_device_mapping length must divide evenly",
        ):
            bridge.build_cluster_registry_inputs(
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[0],
                infer_device_mapping=[0, 1, 2],
            )

    def test_fallback_to_config_when_kwargs_not_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config(
            vllm_tp=2,
            rlix_train_device_mapping=[0, 1],
            rlix_infer_device_mapping=[0, 1, 2, 3],
        )
        tp, devs = bridge.build_cluster_registry_inputs(nemo_config=cfg)
        assert tp == {"actor_train": 1, "actor_infer": 2}
        assert devs == {"actor_train": [0, 1], "actor_infer": [0, 1, 2, 3]}

    def test_kwargs_precedence_over_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config(
            vllm_tp=2,
            rlix_train_device_mapping=[99, 99],
            rlix_infer_device_mapping=[99, 99, 99, 99],
        )
        _, devs = bridge.build_cluster_registry_inputs(
            nemo_config=cfg,
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        assert devs == {"actor_train": [0, 1], "actor_infer": [0, 1, 2, 3]}

    def test_both_missing_raises_for_train(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        with pytest.raises(
            ValueError,
            match=(
                r"train_device_mapping must be provided via kwarg or "
                r"nemo_config\.rlix\.train_device_mapping"
            ),
        ):
            bridge.build_cluster_registry_inputs(
                nemo_config=_make_nemo_config(vllm_tp=2),
                infer_device_mapping=[0, 1, 2, 3],
            )

    def test_both_missing_raises_for_infer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        with pytest.raises(
            ValueError,
            match=(
                r"infer_device_mapping must be provided via kwarg or "
                r"nemo_config\.rlix\.infer_device_mapping"
            ),
        ):
            bridge.build_cluster_registry_inputs(
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[0, 1],
            )

    def test_config_fallback_with_partial_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config(
            vllm_tp=2,
            rlix_infer_device_mapping=[0, 1, 2, 3],
        )
        _, devs = bridge.build_cluster_registry_inputs(
            nemo_config=cfg,
            train_device_mapping=[0, 1],
        )
        assert devs == {"actor_train": [0, 1], "actor_infer": [0, 1, 2, 3]}


class TestDetectPipelineType:
    def test_peft_enabled_true_returns_lora(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        assert (
            bridge.detect_pipeline_type(
                nemo_config=_make_nemo_config(peft_enabled=True)
            )
            == "lora"
        )

    def test_peft_enabled_false_returns_ft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        assert (
            bridge.detect_pipeline_type(
                nemo_config=_make_nemo_config(peft_enabled=False)
            )
            == "ft"
        )

    def test_missing_peft_node_returns_ft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        assert (
            bridge.detect_pipeline_type(nemo_config=_make_nemo_config(drop_peft=True))
            == "ft"
        )

    def test_missing_megatron_cfg_returns_ft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        assert (
            bridge.detect_pipeline_type(
                nemo_config=_make_nemo_config(drop_megatron_cfg=True)
            )
            == "ft"
        )

    def test_missing_policy_returns_ft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        assert (
            bridge.detect_pipeline_type(
                nemo_config=_make_nemo_config(drop_policy=True)
            )
            == "ft"
        )

    def test_truthy_non_bool_peft_enabled_returns_lora(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        assert (
            bridge.detect_pipeline_type(
                nemo_config=_make_nemo_config(peft_enabled="yes")
            )
            == "lora"
        )


class TestExtractTopologyValidationInputs:
    def test_happy_path_returns_all_six_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        result = bridge.extract_topology_validation_inputs(
            nemo_config=_make_nemo_config()
        )
        assert set(result) == {
            "vllm_tp_size",
            "megatron_tp",
            "megatron_pp",
            "megatron_cp",
            "megatron_ep",
            "async_grpo_enabled",
        }

    def test_values_passthrough_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        result = bridge.extract_topology_validation_inputs(
            nemo_config=_make_nemo_config(
                vllm_tp=4,
                meg_tp=2,
                meg_pp=3,
                meg_cp=5,
                meg_ep=7,
                async_grpo=False,
            )
        )
        assert result == {
            "vllm_tp_size": 4,
            "megatron_tp": 2,
            "megatron_pp": 3,
            "megatron_cp": 5,
            "megatron_ep": 7,
            "async_grpo_enabled": False,
        }

    def test_output_kwargs_match_validator_signature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        result = bridge.extract_topology_validation_inputs(
            nemo_config=_make_nemo_config()
        )
        bridge.validate_partial_overlap_topology(
            train_devices=[0, 1],
            infer_devices=[0, 1, 2, 3],
            **result,
        )

    def test_missing_vllm_tp_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config()
        del cfg.policy.generation.vllm_cfg.tensor_parallel_size
        with pytest.raises(
            ValueError,
            match=r"nemo_config missing required field: "
            r"policy\.generation\.vllm_cfg\.tensor_parallel_size",
        ):
            bridge.extract_topology_validation_inputs(nemo_config=cfg)

    def test_missing_megatron_pp_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config()
        del cfg.policy.megatron_cfg.pipeline_model_parallel_size
        with pytest.raises(
            ValueError,
            match=r"nemo_config missing required field: "
            r"policy\.megatron_cfg\.pipeline_model_parallel_size",
        ):
            bridge.extract_topology_validation_inputs(nemo_config=cfg)

    def test_missing_async_grpo_enabled_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config(drop_async_grpo=True)
        with pytest.raises(
            ValueError,
            match=r"nemo_config missing required field: grpo\.async_grpo\.enabled",
        ):
            bridge.extract_topology_validation_inputs(nemo_config=cfg)

    def test_missing_policy_node_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config(drop_policy=True)
        with pytest.raises(
            ValueError,
            match=r"nemo_config missing required field: "
            r"policy\.generation\.vllm_cfg\.tensor_parallel_size",
        ):
            bridge.extract_topology_validation_inputs(nemo_config=cfg)

    def test_missing_grpo_node_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        cfg = _make_nemo_config(drop_grpo=True)
        with pytest.raises(
            ValueError,
            match=r"nemo_config missing required field: grpo\.async_grpo\.enabled",
        ):
            bridge.extract_topology_validation_inputs(nemo_config=cfg)
