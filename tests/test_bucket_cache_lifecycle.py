"""Unit tests for BucketCacheLifecycle.

Covers version tracking, promote(), is_ready_for_version(), reset(),
error propagation, and thread-safety. No Ray or GPU required — all
worker calls are replaced with synchronous fakes.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(sys.modules):
        if key.startswith("rlix") or key == "ray":
            monkeypatch.delitem(sys.modules, key, raising=False)

    # ray stub
    ray_stub = types.ModuleType("ray")
    ray_stub.get = lambda refs, **kw: [r() if callable(r) else r for r in (refs if isinstance(refs, list) else [refs])]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    # ROLL stubs
    for mod_name in ("roll", "roll.utils", "roll.utils.logging"):
        m = types.ModuleType(mod_name)
        monkeypatch.setitem(sys.modules, mod_name, m)
    sys.modules["roll.utils.logging"].get_logger = lambda: MagicMock()  # type: ignore[attr-defined]

    # rlix package stubs
    rlix_root = REPO_ROOT / "rlix"
    for pkg in ("rlix", "rlix.pipeline"):
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(rlix_root / pkg.replace("rlix.", "").replace(".", "/"))]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, pkg, mod)

    sys.path.insert(0, str(REPO_ROOT))


def _load(monkeypatch: pytest.MonkeyPatch):
    import importlib
    _install_stubs(monkeypatch)
    return importlib.import_module("rlix.pipeline.bucket_cache_lifecycle")


# ---------------------------------------------------------------------------
# Fake worker
# ---------------------------------------------------------------------------


class _FakeWorker:
    """Synchronous fake for a NeMo RL training worker Ray actor."""

    def __init__(self, *, fail_on_version: int | None = None):
        self.promoted_versions: list[int] = []
        self.built_versions: list[int] = []
        self._fail_on = fail_on_version

    def build_latest_bucket_cache(self, version: int) -> None:
        self.built_versions.append(version)

    def promote_active_checkpoint(self, version: int) -> None:
        if self._fail_on is not None and version == self._fail_on:
            raise RuntimeError(f"promote_active_checkpoint missing cache_key={version}")
        self.promoted_versions.append(version)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mod(monkeypatch):
    return _load(monkeypatch)


@pytest.fixture()
def workers():
    return [_FakeWorker(), _FakeWorker(), _FakeWorker()]


@pytest.fixture()
def lifecycle(mod, workers):
    return mod.BucketCacheLifecycle(pipeline_id="pipe-test", workers=workers)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_defaults(lifecycle, mod):
    assert lifecycle.pipeline_id == "pipe-test"
    assert lifecycle.cache_ready_step is None
    assert lifecycle.is_ready() is False


def test_construction_rejects_empty_pipeline_id(mod, workers):
    with pytest.raises(ValueError, match="pipeline_id"):
        mod.BucketCacheLifecycle(pipeline_id="", workers=workers)


def test_construction_rejects_empty_workers(mod):
    with pytest.raises(ValueError, match="workers"):
        mod.BucketCacheLifecycle(pipeline_id="pipe", workers=[])


# ---------------------------------------------------------------------------
# promote()
# ---------------------------------------------------------------------------


def test_promote_updates_cache_ready_step(lifecycle):
    lifecycle.promote(1)
    assert lifecycle.cache_ready_step == 1


def test_promote_calls_all_workers(lifecycle, workers):
    lifecycle.promote(5)
    for w in workers:
        assert 5 in w.promoted_versions


def test_promote_sequential_versions(lifecycle):
    for v in [1, 2, 3]:
        lifecycle.promote(v)
    assert lifecycle.cache_ready_step == 3


def test_promote_base_uses_minus_one(lifecycle, workers):
    lifecycle.promote_base()
    assert lifecycle.cache_ready_step == -1
    for w in workers:
        assert -1 in w.promoted_versions


def test_promote_with_custom_base_version(mod, workers):
    lc = mod.BucketCacheLifecycle(pipeline_id="pipe", workers=workers, base_version=0)
    lc.promote_base()
    assert lc.cache_ready_step == 0


def test_promote_marks_ready(lifecycle):
    assert lifecycle.is_ready() is False
    lifecycle.promote(1)
    assert lifecycle.is_ready() is True


def test_promote_failure_propagates(mod):
    """RuntimeError from a worker must propagate — don't silently ignore."""
    bad_worker = _FakeWorker(fail_on_version=3)
    lc = mod.BucketCacheLifecycle(pipeline_id="pipe", workers=[bad_worker])
    with pytest.raises(RuntimeError, match="cache_key=3"):
        lc.promote(3)


def test_promote_failure_does_not_update_ready_step(mod):
    bad_worker = _FakeWorker(fail_on_version=3)
    lc = mod.BucketCacheLifecycle(pipeline_id="pipe", workers=[bad_worker])
    lc.promote(1)  # succeeds
    with pytest.raises(RuntimeError):
        lc.promote(3)  # fails
    # cache_ready_step must still reflect the last SUCCESSFUL promote
    assert lc.cache_ready_step == 1


# ---------------------------------------------------------------------------
# is_ready_for_version()
# ---------------------------------------------------------------------------


def test_not_ready_before_any_promote(lifecycle):
    assert lifecycle.is_ready_for_version(0) is False
    assert lifecycle.is_ready_for_version(-1) is False


def test_ready_for_exact_version(lifecycle):
    lifecycle.promote(5)
    assert lifecycle.is_ready_for_version(5) is True


def test_ready_for_older_version(lifecycle):
    lifecycle.promote(5)
    assert lifecycle.is_ready_for_version(3) is True


def test_not_ready_for_newer_version(lifecycle):
    lifecycle.promote(5)
    assert lifecycle.is_ready_for_version(6) is False


def test_ready_for_base_version(lifecycle):
    lifecycle.promote_base()
    assert lifecycle.is_ready_for_version(-1) is True


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_ready_step(lifecycle):
    lifecycle.promote(10)
    lifecycle.reset()
    assert lifecycle.cache_ready_step is None
    assert lifecycle.is_ready() is False


def test_reset_then_promote_works(lifecycle):
    lifecycle.promote(10)
    lifecycle.reset()
    lifecycle.promote(1)
    assert lifecycle.cache_ready_step == 1


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_concurrent_promotes_are_safe(mod):
    """Multiple threads calling promote() with different versions must not corrupt state."""
    workers = [_FakeWorker()]
    lc = mod.BucketCacheLifecycle(pipeline_id="pipe", workers=workers)
    errors: list[Exception] = []
    n_threads = 10

    def _promote(v: int):
        try:
            lc.promote(v)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=_promote, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # cache_ready_step must be one of the valid promoted values
    assert lc.cache_ready_step in list(range(n_threads))


def test_concurrent_is_ready_for_version_safe(lifecycle):
    """is_ready_for_version() during concurrent promote() must not crash."""
    errors: list[Exception] = []

    def _promoter():
        try:
            for v in range(50):
                lifecycle.promote(v)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def _checker():
        try:
            for _ in range(200):
                lifecycle.is_ready_for_version(25)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t1 = threading.Thread(target=_promoter)
    t2 = threading.Thread(target=_checker)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert errors == []


# ---------------------------------------------------------------------------
# Integration: full lifecycle round-trip (init → train steps → ready check)
# ---------------------------------------------------------------------------


def test_mark_promoted_updates_version_without_calling_workers(mod):
    """mark_promoted() records the version but does NOT call any worker methods."""
    worker = _FakeWorker()
    lc = mod.BucketCacheLifecycle(pipeline_id="pipe-mark", workers=[worker])

    assert lc.cache_ready_step is None

    lc.mark_promoted(-1)

    assert lc.cache_ready_step == -1
    assert lc.is_ready_for_version(-1) is True
    # Worker should NOT have been called — the pipeline layer handles real promotion
    assert worker.promoted_versions == []


def test_mark_promoted_then_promote_continues_tracking(mod):
    """mark_promoted() for init then promote() for training steps work together."""
    workers = [_FakeWorker()]
    lc = mod.BucketCacheLifecycle(pipeline_id="pipe-seq", workers=workers)

    lc.mark_promoted(-1)
    assert lc.cache_ready_step == -1

    lc.promote(1)
    assert lc.cache_ready_step == 1
    # Only the promote() call goes to workers, not mark_promoted()
    assert workers[0].promoted_versions == [1]


def test_full_lifecycle_roundtrip(mod):
    """Simulate pipeline init + 3 train steps + expand readiness check."""
    workers = [_FakeWorker(), _FakeWorker()]
    lc = mod.BucketCacheLifecycle(pipeline_id="pipe-roundtrip", workers=workers)

    # Pipeline init: build_latest_bucket_cache(-1) is called externally,
    # then promote_base() commits it.
    lc.promote_base()
    assert lc.is_ready_for_version(-1) is True
    assert lc.is_ready_for_version(0) is False

    # Step 1: train_step internally builds cache(1), then pipeline promotes.
    lc.promote(1)
    assert lc.is_ready_for_version(1) is True
    assert lc.is_ready_for_version(2) is False

    # Step 2
    lc.promote(2)
    assert lc.is_ready_for_version(2) is True
    assert lc.is_ready_for_version(3) is False

    # Step 3
    lc.promote(3)
    assert lc.is_ready_for_version(3) is True

    # All workers received all promotions in order
    for w in workers:
        assert w.promoted_versions == [-1, 1, 2, 3]
