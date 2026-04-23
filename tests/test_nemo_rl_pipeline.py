"""Unit tests for BucketCacheLifecycle.promote_base() NeMo RL integration.

Verifies:
- promote_base() calls build_latest_bucket_cache(-1) before promote_active_checkpoint(-1)
- promote() calls promote_active_checkpoint(version) and updates _cache_ready_step
- is_ready() and is_ready_for_version() reflect version state correctly
- Version accounting: _cache_ready_step is set to promoted version

All tests run without Ray or GPU — workers are simple Python fakes.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Fake worker (no Ray, no GPU)
# ---------------------------------------------------------------------------


class FakeTrainingWorker:
    """Minimal synchronous fake for a training worker actor."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.build_calls: list = []
        self.promote_calls: list = []

    def build_latest_bucket_cache(self, version: int) -> None:
        self.build_calls.append(version)

    def promote_active_checkpoint(self, version: int) -> None:
        self.promote_calls.append(version)


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _load_lifecycle(monkeypatch):
    # Remove cached modules
    for key in list(sys.modules):
        if "bucket_cache_lifecycle" in key or "rlix.pipeline" in key:
            monkeypatch.delitem(sys.modules, key, raising=False)

    # Stub roll.utils.logging
    roll_utils = types.ModuleType("roll.utils")
    roll_utils_logging = types.ModuleType("roll.utils.logging")
    roll_utils_logging.get_logger = lambda: MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roll", types.ModuleType("roll"))
    monkeypatch.setitem(sys.modules, "roll.utils", roll_utils)
    monkeypatch.setitem(sys.modules, "roll.utils.logging", roll_utils_logging)

    # Ensure rlix is importable
    rlix_root = REPO_ROOT / "rlix"
    rlix_mod = types.ModuleType("rlix")
    rlix_mod.__path__ = [str(rlix_root)]  # type: ignore[attr-defined]
    rlix_pipeline_mod = types.ModuleType("rlix.pipeline")
    rlix_pipeline_mod.__path__ = [str(rlix_root / "pipeline")]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlix", rlix_mod)
    monkeypatch.setitem(sys.modules, "rlix.pipeline", rlix_pipeline_mod)

    import importlib
    return importlib.import_module("rlix.pipeline.bucket_cache_lifecycle")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lifecycle_mod(monkeypatch):
    return _load_lifecycle(monkeypatch)


@pytest.fixture()
def workers():
    return [FakeTrainingWorker(i) for i in range(3)]


@pytest.fixture()
def lifecycle(lifecycle_mod, workers):
    return lifecycle_mod.BucketCacheLifecycle(
        pipeline_id="test_pipeline",
        workers=workers,
    )


# ---------------------------------------------------------------------------
# promote_base — calls build_latest_bucket_cache(-1) then promote(-1)
# ---------------------------------------------------------------------------


def test_promote_base_calls_build_then_promote(lifecycle, workers):
    """promote_base() must call build_latest_bucket_cache(-1) on all workers
    BEFORE calling promote_active_checkpoint(-1)."""
    lifecycle.promote_base()

    for w in workers:
        assert w.build_calls == [-1], f"worker {w.worker_id} missing build_latest_bucket_cache(-1)"
        assert w.promote_calls == [-1], f"worker {w.worker_id} missing promote_active_checkpoint(-1)"


def test_promote_base_sets_cache_ready_step(lifecycle):
    lifecycle.promote_base()
    assert lifecycle.cache_ready_step == -1


def test_promote_base_marks_ready(lifecycle):
    assert not lifecycle.is_ready()
    lifecycle.promote_base()
    assert lifecycle.is_ready()


# ---------------------------------------------------------------------------
# promote — calls promote_active_checkpoint(version)
# ---------------------------------------------------------------------------


def test_promote_calls_promote_active_checkpoint(lifecycle, workers):
    lifecycle.promote(5)
    for w in workers:
        assert 5 in w.promote_calls


def test_promote_updates_cache_ready_step(lifecycle):
    lifecycle.promote(42)
    assert lifecycle.cache_ready_step == 42


def test_promote_successive_versions(lifecycle):
    for v in [0, 1, 2, 3]:
        lifecycle.promote(v)
    assert lifecycle.cache_ready_step == 3


# ---------------------------------------------------------------------------
# Version accounting invariants
# ---------------------------------------------------------------------------


def test_promote_does_not_call_build(lifecycle, workers):
    """promote() must NOT call build_latest_bucket_cache — that's the pipeline's job."""
    lifecycle.promote(10)
    for w in workers:
        assert w.build_calls == [], (
            f"worker {w.worker_id} incorrectly called build_latest_bucket_cache in promote()"
        )


def test_is_ready_for_version_false_before_any_promote(lifecycle):
    assert not lifecycle.is_ready_for_version(0)


def test_is_ready_for_version_true_when_promoted(lifecycle):
    lifecycle.promote(5)
    assert lifecycle.is_ready_for_version(5)
    assert lifecycle.is_ready_for_version(3)


def test_is_ready_for_version_false_for_future(lifecycle):
    lifecycle.promote(2)
    assert not lifecycle.is_ready_for_version(3)


# ---------------------------------------------------------------------------
# cache_ready_step property
# ---------------------------------------------------------------------------


def test_cache_ready_step_none_before_promote(lifecycle):
    assert lifecycle.cache_ready_step is None


def test_cache_ready_step_after_promote_base(lifecycle):
    lifecycle.promote_base()
    assert lifecycle.cache_ready_step == -1


def test_reset_clears_version(lifecycle):
    lifecycle.promote(7)
    lifecycle.reset()
    assert lifecycle.cache_ready_step is None
    assert not lifecycle.is_ready()


# ---------------------------------------------------------------------------
# promote_base order: build before promote (strict ordering test)
# ---------------------------------------------------------------------------


def test_promote_base_build_before_promote_strict_order(lifecycle_mod):
    """Build call on each worker must precede any promote call on that worker."""
    call_order = []

    class OrderedWorker:
        def __init__(self, wid):
            self.worker_id = wid

        def build_latest_bucket_cache(self, version):
            call_order.append(("build", self.worker_id, version))

        def promote_active_checkpoint(self, version):
            call_order.append(("promote", self.worker_id, version))

    workers = [OrderedWorker(i) for i in range(2)]
    lc = lifecycle_mod.BucketCacheLifecycle(
        pipeline_id="ordered_test",
        workers=workers,
    )
    lc.promote_base()

    # All build calls must come before any promote calls
    build_indices = [i for i, e in enumerate(call_order) if e[0] == "build"]
    promote_indices = [i for i, e in enumerate(call_order) if e[0] == "promote"]

    assert build_indices, "No build calls recorded"
    assert promote_indices, "No promote calls recorded"
    assert max(build_indices) < min(promote_indices), (
        f"promote called before all builds completed: {call_order}"
    )
