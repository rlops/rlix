"""Tests for rlix.utils.env helpers (pipeline identity + namespace resolution)."""
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
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_env(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    return importlib.import_module("rlix.utils.env")


class TestPipelineIdentityEnvVars:
    def test_returns_three_expected_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
        result = env.pipeline_identity_env_vars(
            pipeline_id="ft_abc123def456",
            ray_namespace="pipeline_ft_abc123def456_NS",
        )
        assert set(result) == {"PIPELINE_ID", "ROLL_RAY_NAMESPACE", "RLIX_CONTROL_PLANE"}

    def test_maps_args_to_env_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
        result = env.pipeline_identity_env_vars(
            pipeline_id="ft_xyz",
            ray_namespace="pipeline_ft_xyz_NS",
        )
        assert result["PIPELINE_ID"] == "ft_xyz"
        assert result["ROLL_RAY_NAMESPACE"] == "pipeline_ft_xyz_NS"

    def test_control_plane_defaults_to_rlix_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
        result = env.pipeline_identity_env_vars(pipeline_id="ft_x", ray_namespace="ns")
        assert result["RLIX_CONTROL_PLANE"] == "rlix"

    def test_control_plane_passthrough_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.setenv("RLIX_CONTROL_PLANE", "custom_plane")
        result = env.pipeline_identity_env_vars(pipeline_id="ft_x", ray_namespace="ns")
        assert result["RLIX_CONTROL_PLANE"] == "custom_plane"


class TestResolveNemoRlPipelineNamespace:
    def test_rlix_with_namespace_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
        monkeypatch.setenv("ROLL_RAY_NAMESPACE", "pipeline_ft_abc_NS")
        assert env.resolve_nemo_rl_pipeline_namespace() == "pipeline_ft_abc_NS"

    def test_rlix_missing_namespace_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
        monkeypatch.delenv("ROLL_RAY_NAMESPACE", raising=False)
        with pytest.raises(ValueError, match="NeMo RL.*ROLL_RAY_NAMESPACE"):
            env.resolve_nemo_rl_pipeline_namespace()

    def test_rlix_empty_namespace_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
        monkeypatch.setenv("ROLL_RAY_NAMESPACE", "")
        with pytest.raises(ValueError, match="NeMo RL.*ROLL_RAY_NAMESPACE"):
            env.resolve_nemo_rl_pipeline_namespace()

    def test_standalone_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
        monkeypatch.delenv("ROLL_RAY_NAMESPACE", raising=False)
        assert env.resolve_nemo_rl_pipeline_namespace() == "roll"

    def test_standalone_custom_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
        monkeypatch.delenv("ROLL_RAY_NAMESPACE", raising=False)
        assert env.resolve_nemo_rl_pipeline_namespace(default="custom_ns") == "custom_ns"

    def test_standalone_uses_namespace_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
        monkeypatch.setenv("ROLL_RAY_NAMESPACE", "manually_set_ns")
        assert env.resolve_nemo_rl_pipeline_namespace(default="roll") == "manually_set_ns"

    def test_non_rlix_control_plane_missing_namespace_no_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _load_env(monkeypatch)
        monkeypatch.setenv("RLIX_CONTROL_PLANE", "standalone")
        monkeypatch.delenv("ROLL_RAY_NAMESPACE", raising=False)
        assert env.resolve_nemo_rl_pipeline_namespace(default="roll") == "roll"
