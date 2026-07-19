"""Regression test for _cancel_inflight's Ray actor-task cancel semantics.

Every ref in ``inflight_refs`` is an actor-method ObjectRef, and Ray only
allows ``force=False`` for actor tasks — ``force=True`` raises ValueError
before cancelling anything, which made the R06-F1 fan-out a silent no-op
(each ValueError was swallowed by the per-ref try/except). Pin that every
cancel goes out with ``force=False`` and actually reaches ray.cancel.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[list, types.ModuleType]:
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")
    cancel_calls: list = []

    def _remote(*args, **kwargs):
        # Support both bare ``@ray.remote`` (class passed directly) and
        # ``@ray.remote(...)`` / ``ray.remote(cls)`` forms.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorate(obj):
            return obj

        return _decorate

    def _cancel(ref, *, force=False, recursive=True):
        # Mirror Ray's actor-task contract: force=True raises ValueError
        # before any cancellation happens.
        if force:
            raise ValueError("force=True is not allowed for an Actor Task")
        cancel_calls.append((ref, force))

    ray_stub.remote = _remote
    ray_stub.cancel = _cancel
    ray_stub.get = lambda ref, timeout=None: ref
    ray_stub.get_actor = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.pipeline": RLIX_ROOT / "pipeline",
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)

    return cancel_calls, ray_stub


def test_cancel_inflight_uses_force_false_and_cancels_every_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_calls, _ = _install_import_stubs(monkeypatch)
    module = importlib.import_module("rlix.pipeline.miles_model_update_service")

    service = module.MilesModelUpdateService(
        pipeline_id="miles_test",
        cache_owner_actor=object(),
        rollout_manager=object(),
    )

    refs = ["ref_a", "ref_b", "ref_c"]
    inflight = list(refs)
    service._cancel_inflight(inflight, reason="unit-test")

    # Every ref reached ray.cancel with force=False (force=True would have
    # raised inside the stub and never recorded the call — the pre-fix
    # silent-no-op behavior).
    assert cancel_calls == [(ref, False) for ref in refs]
    # The list is drained so a second timeout cannot double-cancel.
    assert inflight == []
