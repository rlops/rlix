"""Tests for MilesPipeline._after_training cleanup contract.

Locks the contract:
    - The scheduler-side release runs IFF train-side offload completed
      successfully. Releasing while train weights may still be resident on
      GPU could let the scheduler hand the same GPUs to another pipeline
      and OOM it.
    - sync_base_weights_to_active() failure → release runs (offload already
      completed, train side is clean).
    - build_cpu_bucket_cache() failure → offload is still attempted (inner
      finally); if offload succeeds, release runs.
    - offload() failure → release does NOT run; _actor_train_allocated stays
      True so shutdown_hard / manual recovery handles it.
    - On successful release, _actor_train_allocated flips to False.
    - On release RPC failure, flag stays True so shutdown_hard can retry.

Pattern: stub `ray` and `rlix` package roots before importing
`rlix.pipeline.miles_pipeline` (matches existing tests/test_*.py style).
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
    """Stub `ray` and rlix package roots so we can import miles_pipeline
    without a live Ray runtime.
    """
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
    # Identity ray.get: real flow returns the resolved value of an ObjectRef;
    # in tests we let mock .remote() return the value directly so identity works.
    ray_stub.get = lambda value, *args, **kwargs: value
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.utils": RLIX_ROOT / "utils",
        "rlix.pipeline": RLIX_ROOT / "pipeline",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_miles_pipeline(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    return importlib.import_module("rlix.pipeline.miles_pipeline")


# --- Helpers ---------------------------------------------------------------


class _RemoteFn:
    """Records every call; raises if `side_effect` is set."""

    def __init__(self, side_effect=None, return_value=None):
        self._side_effect = side_effect
        self._return_value = return_value
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._side_effect is not None:
            raise self._side_effect
        return self._return_value


def _build_pipeline(
    miles_pipeline_module,
    *,
    build_raises: bool = False,
    offload_raises: bool = False,
    sync_raises: bool = False,
    release_succeeds: bool = True,
):
    """Construct a MilesPipeline with bypassed __init__ + injected mocks.

    Bypasses __init__ to avoid pipeline_id validation, namespace lookup,
    scheduler actor resolution, and topology validation — none of which
    are needed to exercise _after_training in isolation.

    Returns: (pipe, notify_remote, sync_remote, call_log)
        call_log: list[str] tracking ["build", "offload", ...] in invocation order.
    """
    MilesPipeline = miles_pipeline_module.MilesPipeline
    pipe = MilesPipeline.__new__(MilesPipeline)

    # Minimal attrs touched by _after_training + _notify_release_cluster_gpus.
    pipe._initialized = True
    pipe._actor_train_allocated = True
    pipe._actor_train_cluster_id = "test_pid_actor_train"
    pipe._pipeline_id = "test_pid"

    call_log: list[str] = []

    async def _build(step):
        call_log.append("build")
        if build_raises:
            raise RuntimeError("build boom")

    async def _offload():
        call_log.append("offload")
        if offload_raises:
            raise RuntimeError("offload boom")

    pipe._train_group = types.SimpleNamespace(
        build_cpu_bucket_cache=_build,
        offload=_offload,
    )

    sync_remote = _RemoteFn(
        side_effect=RuntimeError("sync boom") if sync_raises else None,
        return_value="synced",
    )
    pipe._coordinator_handle = types.SimpleNamespace(
        sync_base_weights_to_active=types.SimpleNamespace(remote=sync_remote),
    )

    notify_remote = _RemoteFn(
        side_effect=None if release_succeeds else RuntimeError("release boom"),
        return_value="released",
    )
    pipe._scheduler = types.SimpleNamespace(
        notify_release_gpus=types.SimpleNamespace(remote=notify_remote),
    )

    return pipe, notify_remote, sync_remote, call_log


# --- Tests -----------------------------------------------------------------


def test_after_training_happy_path_releases(monkeypatch):
    """Regression: happy path runs build → offload → sync → release."""
    mp = _load_miles_pipeline(monkeypatch)
    pipe, notify_remote, sync_remote, call_log = _build_pipeline(mp)

    pipe._after_training(step=0)

    assert call_log == ["build", "offload"]
    assert len(sync_remote.calls) == 1
    assert len(notify_remote.calls) == 1
    assert pipe._actor_train_allocated is False


def test_after_training_releases_when_sync_raises(monkeypatch):
    """sync_base_weights_to_active raises → release still runs (offload completed)."""
    mp = _load_miles_pipeline(monkeypatch)
    pipe, notify_remote, sync_remote, call_log = _build_pipeline(mp, sync_raises=True)

    with pytest.raises(RuntimeError, match="sync boom"):
        pipe._after_training(step=0)

    assert call_log == ["build", "offload"]
    assert len(sync_remote.calls) == 1
    assert len(notify_remote.calls) == 1, (
        "release MUST run when sync raises — train is already offloaded, "
        "scheduler ledger must reflect that"
    )
    assert pipe._actor_train_allocated is False


def test_after_training_attempts_offload_and_releases_when_build_raises(monkeypatch):
    """build_cpu_bucket_cache raises → inner finally still attempts offload;
    if offload succeeds, release runs."""
    mp = _load_miles_pipeline(monkeypatch)
    pipe, notify_remote, sync_remote, call_log = _build_pipeline(mp, build_raises=True)

    with pytest.raises(RuntimeError, match="build boom"):
        pipe._after_training(step=0)

    assert call_log == ["build", "offload"], (
        "offload MUST be attempted even when build_cache raises — otherwise "
        "train weights stay on GPU"
    )
    assert len(sync_remote.calls) == 0, "sync should not run when build raises before it"
    assert len(notify_remote.calls) == 1, "release runs because offload completed"
    assert pipe._actor_train_allocated is False


def test_after_training_does_NOT_release_when_offload_raises(monkeypatch):
    """offload raises → release does NOT run, flag stays True.

    This is the safety boundary: if offload failed, train weights may still
    be on GPU. Telling the scheduler 'released' would let the next pipeline
    be scheduled onto the same physical GPUs and OOM. shutdown_hard /
    manual ray-stop is the recovery surface.
    """
    mp = _load_miles_pipeline(monkeypatch)
    pipe, notify_remote, sync_remote, call_log = _build_pipeline(mp, offload_raises=True)

    with pytest.raises(RuntimeError, match="offload boom"):
        pipe._after_training(step=0)

    assert call_log == ["build", "offload"]
    assert len(sync_remote.calls) == 0, "sync must not run after offload raises"
    assert len(notify_remote.calls) == 0, (
        "release MUST NOT run when offload raises — GPU may still hold train weights"
    )
    assert pipe._actor_train_allocated is True, (
        "flag must stay True so shutdown_hard can retry; manual ray stop is "
        "the recovery surface for unknown GPU state"
    )


def test_after_training_keeps_flag_set_when_release_rpc_fails(monkeypatch):
    """Boundary: release RPC itself fails → flag stays True so shutdown_hard retries."""
    mp = _load_miles_pipeline(monkeypatch)
    pipe, notify_remote, sync_remote, call_log = _build_pipeline(mp, release_succeeds=False)

    # _after_training does not raise here — _notify_release_cluster_gpus
    # catches the RPC error internally and returns False.
    pipe._after_training(step=0)

    assert call_log == ["build", "offload"]
    assert len(sync_remote.calls) == 1
    assert len(notify_remote.calls) == 1
    assert pipe._actor_train_allocated is True, (
        "flag must stay True when release RPC fails — shutdown_hard relies on "
        "this to retry the release"
    )
