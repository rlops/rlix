"""Tests for cluster name constants and their usage in validation/parsing."""
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

    def _remote(*args, **kwargs):
        def _decorate(obj):
            return obj

        return _decorate

    ray_stub.remote = _remote
    ray_stub.get_actor = lambda *args, **kwargs: None
    ray_stub.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.scheduler": RLIX_ROOT / "scheduler",
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_modules(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    protocol_types = importlib.import_module("rlix.protocol.types")
    protocol_validation = importlib.import_module("rlix.protocol.validation")
    scheduler_types = importlib.import_module("rlix.scheduler.types")
    return protocol_types, protocol_validation, scheduler_types


# --- Constant relationship tests ---


def test_gpu_cluster_names_is_subset_of_all(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, _, _ = _load_modules(monkeypatch)
    assert set(pt.GPU_CLUSTER_NAMES) | {pt.REWARD_CLUSTER_NAME} == set(pt.ALL_CLUSTER_NAMES)


def test_generation_cluster_name_in_gpu_set(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, _, _ = _load_modules(monkeypatch)
    assert pt.GENERATION_CLUSTER_NAME in pt.GPU_CLUSTER_NAMES


def test_reward_not_in_gpu_set(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, _, _ = _load_modules(monkeypatch)
    assert pt.REWARD_CLUSTER_NAME not in pt.GPU_CLUSTER_NAMES


# --- Registration validation tests ---


def _make_register_input(pv, cluster_tp_configs, cluster_device_mappings=None):
    if cluster_device_mappings is None:
        cluster_device_mappings = {name: [0, 1] for name in cluster_tp_configs}
    return pv.RegisterValidationInput(
        pipeline_id="ft_abc123def456",
        ray_namespace="pipeline_ft_abc123def456_NS",
        cluster_tp_configs=cluster_tp_configs,
        cluster_device_mappings=cluster_device_mappings,
    )


def test_validate_register_rejects_unknown_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, pv, _ = _load_modules(monkeypatch)
    inp = _make_register_input(pv, {pt.GENERATION_CLUSTER_NAME: 1, "banana": 1})
    with pytest.raises(ValueError, match="Unknown cluster name"):
        pv.validate_register_pipeline(inp)


def test_validate_register_accepts_all_known_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, pv, _ = _load_modules(monkeypatch)
    tp_configs = {name: 1 for name in pt.ALL_CLUSTER_NAMES}
    device_mappings = {name: [0, 1] for name in pt.ALL_CLUSTER_NAMES}
    # reward is CPU-only: empty device mapping
    device_mappings[pt.REWARD_CLUSTER_NAME] = []
    inp = _make_register_input(pv, tp_configs, device_mappings)
    pv.validate_register_pipeline(inp)  # should not raise


# --- parse_cluster_id tests ---


def test_parse_cluster_id_roundtrips_gpu_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, _, st = _load_modules(monkeypatch)
    pipeline_id = "ft_abc123def456"
    for cluster_name in pt.GPU_CLUSTER_NAMES:
        cluster_id = f"{pipeline_id}_{cluster_name}"
        parsed_pid, parsed_name = st.parse_cluster_id(cluster_id)
        assert parsed_pid == pipeline_id
        assert parsed_name == cluster_name


def test_parse_cluster_id_rejects_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, _, st = _load_modules(monkeypatch)
    cluster_id = f"ft_abc123def456_{pt.REWARD_CLUSTER_NAME}"
    with pytest.raises(ValueError, match="Unrecognized cluster_id"):
        st.parse_cluster_id(cluster_id)


# --- is_generation_cluster tests ---


def test_is_generation_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    pt, _, st = _load_modules(monkeypatch)
    assert st.is_generation_cluster(f"ft_abc123def456_{pt.GENERATION_CLUSTER_NAME}") is True
    assert st.is_generation_cluster("ft_abc123def456_actor_train") is False
    assert st.is_generation_cluster("ft_abc123def456_critic") is False
    assert st.is_generation_cluster("ft_abc123def456_reference") is False
