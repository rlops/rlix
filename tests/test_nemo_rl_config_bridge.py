"""Tests for rlix.pipeline.nemo_rl_config_bridge topology validation."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

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


def _valid_kwargs() -> dict:
    return {
        "train_devices": [0, 1],
        "infer_devices": [0, 1, 2, 3],
        "vllm_tp_size": 1,
        "megatron_tp": 1,
        "megatron_pp": 1,
        "megatron_cp": 1,
        "megatron_ep": 1,
        "async_grpo_enabled": True,
    }


def test_happy_path_accepts_valid_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _load_bridge(monkeypatch)
    bridge.validate_partial_overlap_topology(**_valid_kwargs())


def test_rejects_train_not_subset_of_infer(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _load_bridge(monkeypatch)
    kwargs = _valid_kwargs()
    kwargs["train_devices"] = [0, 4]
    kwargs["infer_devices"] = [0, 1, 2, 3]
    with pytest.raises(AssertionError, match=r"partial overlap requires train"):
        bridge.validate_partial_overlap_topology(**kwargs)


def test_rejects_infer_dp_size_less_than_two(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _load_bridge(monkeypatch)
    kwargs = _valid_kwargs()
    kwargs["train_devices"] = [0]
    kwargs["infer_devices"] = [0, 1]
    kwargs["vllm_tp_size"] = 2
    with pytest.raises(AssertionError, match=r"partial overlap requires dp >= 2"):
        bridge.validate_partial_overlap_topology(**kwargs)


def test_rejects_async_grpo_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _load_bridge(monkeypatch)
    kwargs = _valid_kwargs()
    kwargs["async_grpo_enabled"] = False
    with pytest.raises(AssertionError, match=r"partial overlap requires async GRPO"):
        bridge.validate_partial_overlap_topology(**kwargs)


def test_rejects_train_not_divisible_by_megatron_parallelism_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _load_bridge(monkeypatch)
    kwargs = _valid_kwargs()
    kwargs["train_devices"] = [0, 1]
    kwargs["infer_devices"] = [0, 1, 2, 3]
    kwargs["megatron_pp"] = 3
    with pytest.raises(AssertionError, match=r"must divide evenly by tp\*pp\*cp\*ep"):
        bridge.validate_partial_overlap_topology(**kwargs)


def test_rejects_infer_not_divisible_by_vllm_tp_size(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _load_bridge(monkeypatch)
    kwargs = _valid_kwargs()
    kwargs["train_devices"] = [0]
    kwargs["infer_devices"] = [0, 1, 2, 3, 4]
    kwargs["vllm_tp_size"] = 2
    with pytest.raises(AssertionError, match=r"must divide evenly by vllm_tp_size"):
        bridge.validate_partial_overlap_topology(**kwargs)


def test_rejects_no_full_infer_dp_rank_after_shrink(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _load_bridge(monkeypatch)
    kwargs = _valid_kwargs()
    kwargs["train_devices"] = [0, 1, 2]
    kwargs["infer_devices"] = [0, 1, 2, 3]
    kwargs["vllm_tp_size"] = 2
    with pytest.raises(
        AssertionError,
        match=r"at least 1 full inference DP rank must stay active after shrink",
    ):
        bridge.validate_partial_overlap_topology(**kwargs)
