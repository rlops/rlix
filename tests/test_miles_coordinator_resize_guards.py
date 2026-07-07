"""PR resize-guard tests for MilesCoordinator.

Covers the resize hardening:
  1. ``resize_infer`` validates its params via ``validate_resize_params``
     (exactly one of remove/add non-empty — same contract as
     ``PipelineCoordinator.resize_infer``).
  2. ``_resize_locked`` bounds ``_resize_sync_lock`` acquisition and raises
     instead of blocking forever when the lock is wedged.
"""

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
    ray_stub.get = lambda ref, timeout=None: ref
    # MilesCoordinator.__init__ resolves the central scheduler via
    # get_actor_or_raise; returning None keeps _rlix_scheduler unset without
    # raising, which the constructor tolerates (fail-open by design).
    ray_stub.get_actor = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.pipeline": RLIX_ROOT / "pipeline",
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _make_coordinator(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    module = importlib.import_module("rlix.pipeline.miles_coordinator")
    coordinator = module.MilesCoordinator(
        pipeline_id="miles_test", pipeline_config=types.SimpleNamespace()
    )
    return module, coordinator


def test_resize_infer_rejects_mixed_shrink_and_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    _, coordinator = _make_coordinator(monkeypatch)
    with pytest.raises(ValueError, match="Exactly one"):
        coordinator.resize_infer(dp_ranks_to_remove=[0], dp_ranks_to_add=[1])


def test_resize_infer_rejects_empty_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _, coordinator = _make_coordinator(monkeypatch)
    with pytest.raises(ValueError, match="Exactly one"):
        coordinator.resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[])


def test_resize_lock_acquisition_times_out_instead_of_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, coordinator = _make_coordinator(monkeypatch)
    monkeypatch.setattr(module, "_RESIZE_LOCK_TIMEOUT_S", 0.05)

    # Simulate a wedged holder: the lock is held by "someone else" while a
    # read-side caller comes in. Without the bounded acquire this would
    # block forever.
    assert coordinator._resize_sync_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="_resize_sync_lock"):
            coordinator.get_active_engines()
    finally:
        coordinator._resize_sync_lock.release()

    # After the holder releases, the same call succeeds.
    assert coordinator.get_active_engines() == frozenset()
