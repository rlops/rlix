import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


def _load_env_module(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "rlix" or module_name.startswith("rlix."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)

    return importlib.import_module("rlix.utils.env")


def test_parse_env_positive_float_uses_default_when_unset(monkeypatch):
    env = _load_env_module(monkeypatch)
    monkeypatch.delenv("MILES_MAX_RESIDUAL_GPU_MEM_GB", raising=False)

    assert env.parse_env_positive_float("MILES_MAX_RESIDUAL_GPU_MEM_GB", 10.0) == 10.0


def test_parse_env_positive_float_reads_override(monkeypatch):
    env = _load_env_module(monkeypatch)
    monkeypatch.setenv("MILES_MAX_RESIDUAL_GPU_MEM_GB", "40.5")

    assert env.parse_env_positive_float("MILES_MAX_RESIDUAL_GPU_MEM_GB", 10.0) == 40.5


def test_parse_env_positive_float_rejects_non_positive(monkeypatch):
    env = _load_env_module(monkeypatch)
    monkeypatch.setenv("MILES_MAX_RESIDUAL_GPU_MEM_GB", "0")

    with pytest.raises(RuntimeError, match="must be > 0"):
        env.parse_env_positive_float("MILES_MAX_RESIDUAL_GPU_MEM_GB", 10.0)


def test_parse_env_positive_float_rejects_non_numeric(monkeypatch):
    env = _load_env_module(monkeypatch)
    monkeypatch.setenv("MILES_MAX_RESIDUAL_GPU_MEM_GB", "not-a-number")

    with pytest.raises(RuntimeError, match="must be a number"):
        env.parse_env_positive_float("MILES_MAX_RESIDUAL_GPU_MEM_GB", 10.0)
