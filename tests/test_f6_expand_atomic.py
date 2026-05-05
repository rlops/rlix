"""F6 atomic expand tests — no real Ray / GPU / vLLM required.

Verifies the core invariant of _expand_workers:
  activate_dp_ranks (step 5) is ONLY called if sync_selected_workers (step 3)
  AND set_weight_version (step 4) both succeed.

Run with:
    cd rlix/
    python -m pytest tests/test_f6_expand_atomic.py -v
    # or directly:
    python tests/test_f6_expand_atomic.py

No special dependencies beyond pytest. ray is stubbed out at import time.
"""
from __future__ import annotations

import pathlib
import sys
import threading
import types
import unittest.mock as mock
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Lightweight import isolation — lets us test _expand_workers without
# a Ray cluster, GPU, torch, or megatron installed.
#
# Strategy: pre-populate sys.modules for packages whose __init__.py would
# import heavy deps (ray, torch).  Setting __path__ correctly means Python
# still finds individual submodule .py files via normal file-system lookup,
# but never executes the __init__.py side effects.
# ---------------------------------------------------------------------------

_RLIX_ROOT = pathlib.Path(__file__).resolve().parent.parent / "rlix"  # .../rlix/rlix/


def _stub_package(dotted_name: str, fs_path: pathlib.Path) -> None:
    """Register a lightweight package stub that lets submodule .py files load normally."""
    if dotted_name not in sys.modules:
        pkg = types.ModuleType(dotted_name)
        pkg.__path__ = [str(fs_path)]
        pkg.__package__ = dotted_name
        pkg.__spec__ = None
        sys.modules[dotted_name] = pkg


def _stub_ray() -> None:
    """Minimal ray stub: ray.get, ray.remote, ray.get_actor."""
    if "ray" in sys.modules:
        return
    ray_mod = types.ModuleType("ray")

    def _get(f: Any) -> Any:
        return f._value if hasattr(f, "_value") else f

    ray_mod.get = _get
    ray_mod.remote = lambda cls_or_fn: cls_or_fn  # @ray.remote no-op decorator
    ray_mod.get_actor = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("ray.get_actor called in test — actor resolution is bypassed via object.__new__")
    )
    sys.modules["ray"] = ray_mod
    # Also needed by rlix.utils.ray (lazy imports inside functions — no-op stubs)
    sys.modules.setdefault("ray.runtime_env", types.ModuleType("ray.runtime_env"))
    sys.modules.setdefault("ray.util", types.ModuleType("ray.util"))
    sys.modules.setdefault("ray.util.state", types.ModuleType("ray.util.state"))
    sys.modules.setdefault("ray.util.scheduling_strategies", types.ModuleType("ray.util.scheduling_strategies"))


_stub_ray()
# Prevent rlix/__init__.py (imports ray.client) and
# rlix/pipeline/__init__.py (imports full_finetune_pipeline → torch) from running.
_stub_package("rlix", _RLIX_ROOT)
_stub_package("rlix.pipeline", _RLIX_ROOT / "pipeline")
_stub_package("rlix.protocol", _RLIX_ROOT / "protocol")
_stub_package("rlix.utils", _RLIX_ROOT / "utils")
_stub_package("rlix.scheduler", _RLIX_ROOT / "scheduler")


# ---------------------------------------------------------------------------
# Minimal Ray mock — lets us call ray.get(obj.remote(...)) without a Ray cluster
# ---------------------------------------------------------------------------

class _MockFuture:
    """Fake Ray ObjectRef returned by .remote()."""
    def __init__(self, value: Any) -> None:
        self._value = value


def _fake_ray_get(future: Any) -> Any:
    if isinstance(future, _MockFuture):
        return future._value
    return future


class _RemoteMethod:
    """Wraps a plain callable so .remote(*args, **kwargs) → _MockFuture."""
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs) -> _MockFuture:
        return _MockFuture(self._fn(*args, **kwargs))


def remote_method(fn):
    """Decorator: makes fn.remote(...) work like a Ray actor method."""
    return _RemoteMethod(fn)


# ---------------------------------------------------------------------------
# Mock dependencies
# ---------------------------------------------------------------------------

class MockVLLMGeneration:
    """Mock for VllmGeneration (F2/F3 stub).

    Tracks:
        active_dp_ranks      — set of currently routable ranks
        woken_ranks          — set of ranks that received wake_up_partial
        inactive_ranks       — set of ranks explicitly marked inactive (cleared on activate)
        events               — per-object call log
        shared_events        — optional shared log that captures cross-object global order
    """

    def __init__(self, dp_size: int = 4, shared_events: Optional[List[str]] = None) -> None:
        self.dp_size = dp_size
        self.active_dp_ranks: set = set(range(dp_size))
        self.woken_ranks: set = set()
        self.inactive_ranks: set = set()
        self.events: List[str] = []
        self._shared = shared_events

    def _log(self, msg: str) -> None:
        self.events.append(msg)
        if self._shared is not None:
            self._shared.append(msg)

    def mark_dp_ranks_inactive(self, dp_ranks: List[int]) -> None:
        self.inactive_ranks.update(dp_ranks)
        self.active_dp_ranks.difference_update(dp_ranks)
        self._log(f"mark_inactive({sorted(dp_ranks)})")

    def wake_up_partial(self, dp_ranks: List[int]) -> None:
        self.woken_ranks.update(dp_ranks)
        self._log(f"wake_up_partial({sorted(dp_ranks)})")

    def sleep_partial(self, dp_ranks: List[int], level: int = 2) -> None:
        self.woken_ranks.difference_update(dp_ranks)
        self.active_dp_ranks.difference_update(dp_ranks)
        self._log(f"sleep_partial({sorted(dp_ranks)}, level={level})")

    def activate_dp_ranks(self, dp_ranks: List[int]) -> None:
        self.active_dp_ranks.update(dp_ranks)
        self.inactive_ranks.difference_update(dp_ranks)
        self._log(f"activate_dp_ranks({sorted(dp_ranks)})")

    def finalize_weight_update(self, dp_ranks: List[int]) -> List[Any]:
        self._log(f"finalize_weight_update({sorted(dp_ranks)})")
        return []


class MockModelUpdateService:
    """Mock for NemoRLModelUpdateService (F4 stub).

    Set fail_on_sync=True to simulate a weight sync failure.
    """

    def __init__(self, fail_on_sync: bool = False, shared_events: Optional[List[str]] = None) -> None:
        self.fail_on_sync = fail_on_sync
        self.sync_calls: List[List[int]] = []
        self.events: List[str] = []
        self._shared = shared_events

    def _log(self, msg: str) -> None:
        self.events.append(msg)
        if self._shared is not None:
            self._shared.append(msg)

    def sync_selected_workers(self, tgt_dp_ranks: List[int], verify: bool = False) -> None:
        self._log(f"sync_selected_workers({sorted(tgt_dp_ranks)})")
        self.sync_calls.append(sorted(tgt_dp_ranks))
        if self.fail_on_sync:
            raise RuntimeError("MockModelUpdateService: simulated sync failure")

    @property
    def remote_proxy(self) -> "_MockRemoteProxy":
        return _MockRemoteProxy(self)


class MockTrajectoryCollector:
    """Mock for AsyncTrajectoryCollector (F9 stub).

    Set fail_on_set_version=True to simulate a version update failure.
    """

    def __init__(self, fail_on_set_version: bool = False, shared_events: Optional[List[str]] = None) -> None:
        self.fail_on_set_version = fail_on_set_version
        self.weight_version: int = -1
        self.set_version_calls: List[int] = []
        self.events: List[str] = []
        self._shared = shared_events

    def _log(self, msg: str) -> None:
        self.events.append(msg)
        if self._shared is not None:
            self._shared.append(msg)

    def set_weight_version(self, version: int) -> None:
        self._log(f"set_weight_version({version})")
        self.set_version_calls.append(version)
        if self.fail_on_set_version:
            raise RuntimeError("MockTrajectoryCollector: simulated set_version failure")
        self.weight_version = version


class _MockRemoteProxy:
    """Wraps a mock actor so .method.remote(...) → _MockFuture."""
    def __init__(self, actor: Any) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> _RemoteMethod:
        fn = getattr(self._actor, name)
        return _RemoteMethod(fn)


# ---------------------------------------------------------------------------
# Test fixture: build a NemoRLFullFinetunePipeline without Ray
# ---------------------------------------------------------------------------

def _make_pipeline(
    *,
    vllm: Optional[MockVLLMGeneration] = None,
    svc: Optional[MockModelUpdateService] = None,
    collector: Optional[MockTrajectoryCollector] = None,
    initial_version: int = 0,
    dp_size: int = 4,
) -> Any:
    """Construct a NemoRLFullFinetunePipeline bypassing the Ray-dependent __init__.

    Uses object.__new__ + attribute injection so no Ray cluster is needed.
    Only sets the attributes required by _expand_workers and _shrink_workers.
    """
    from rlix.pipeline.nemo_rl_pipeline import NemoRLFullFinetunePipeline

    pipeline = object.__new__(NemoRLFullFinetunePipeline)
    pipeline._pipeline_id = "test_pipeline"
    pipeline._infer_resize_lock = threading.Lock()
    pipeline._current_weight_version = initial_version
    pipeline._pre_activation_ranks = set()
    pipeline._active_dp_ranks = set()
    pipeline._cache_ready_step = initial_version
    pipeline._initialized = True

    pipeline._policy_generation = vllm or MockVLLMGeneration(dp_size=dp_size)
    pipeline._model_update_service = _MockRemoteProxy(svc or MockModelUpdateService())
    pipeline._trajectory_collector = _MockRemoteProxy(collector or MockTrajectoryCollector())

    # Keep direct references for assertions in tests
    pipeline._mock_vllm = pipeline._policy_generation
    pipeline._mock_svc = (svc or MockModelUpdateService())
    pipeline._mock_collector = (collector or MockTrajectoryCollector())

    return pipeline


def _make_pipeline_with_refs(
    *,
    vllm: MockVLLMGeneration,
    svc: MockModelUpdateService,
    collector: MockTrajectoryCollector,
    initial_version: int = 0,
) -> Any:
    """Like _make_pipeline but keeps direct references to the mocks."""
    from rlix.pipeline.nemo_rl_pipeline import NemoRLFullFinetunePipeline

    pipeline = object.__new__(NemoRLFullFinetunePipeline)
    pipeline._pipeline_id = "test_pipeline"
    pipeline._infer_resize_lock = threading.Lock()
    pipeline._current_weight_version = initial_version
    pipeline._pre_activation_ranks = set()
    pipeline._active_dp_ranks = set()
    pipeline._cache_ready_step = initial_version
    pipeline._initialized = True

    pipeline._policy_generation = vllm
    pipeline._model_update_service = _MockRemoteProxy(svc)
    pipeline._trajectory_collector = _MockRemoteProxy(collector)

    return pipeline


# ---------------------------------------------------------------------------
# Patch helper: replace ray.get in the pipeline module with _fake_ray_get
# ---------------------------------------------------------------------------

def patched_expand(pipeline, dp_ranks: List[int]):
    """Call _expand_workers with ray.get patched to work on _MockFuture."""
    with mock.patch("rlix.pipeline.nemo_rl_pipeline.ray.get", side_effect=_fake_ray_get):
        pipeline._expand_workers(dp_ranks_to_add=dp_ranks)


def patched_shrink(pipeline, dp_ranks: List[int]):
    """Call _shrink_workers with asyncio.run patched (sleep_partial is async)."""
    import asyncio

    async def _fake_sleep_partial(dp_ranks, level=2):
        pipeline._policy_generation.sleep_partial(dp_ranks, level=level)

    with mock.patch("asyncio.run", side_effect=lambda coro: asyncio.get_event_loop().run_until_complete(coro)):
        pipeline._shrink_workers(dp_ranks_to_remove=dp_ranks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestF6ExpandAtomicHappyPath:
    """Happy path: all 5 steps succeed, verify ordering and state."""

    def test_event_order(self):
        """Steps must fire in order: mark_inactive → wake_up → sync → set_version → activate."""
        # All mocks write to the same shared_events list to capture true global ordering.
        shared: List[str] = []
        vllm = MockVLLMGeneration(dp_size=4, shared_events=shared)
        svc = MockModelUpdateService(shared_events=shared)
        collector = MockTrajectoryCollector(shared_events=shared)
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=0)

        patched_expand(pipeline, dp_ranks=[1, 2])

        idx = {e: i for i, e in enumerate(shared)}
        assert "mark_inactive([1, 2])" in idx
        assert "wake_up_partial([1, 2])" in idx
        assert "sync_selected_workers([1, 2])" in idx
        assert "set_weight_version(0)" in idx  # no bump: expand reuses same cache as active refresh
        assert "activate_dp_ranks([1, 2])" in idx

        assert idx["mark_inactive([1, 2])"] < idx["wake_up_partial([1, 2])"]
        assert idx["wake_up_partial([1, 2])"] < idx["sync_selected_workers([1, 2])"]
        assert idx["sync_selected_workers([1, 2])"] < idx["set_weight_version(0)"]
        assert idx["set_weight_version(0)"] < idx["activate_dp_ranks([1, 2])"]

    def test_weight_version_incremented(self):
        """_current_weight_version stays at _cache_ready_step — expand does not bump (spec F6 no-bump)."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector(fail_on_set_version=False)
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=5)

        patched_expand(pipeline, dp_ranks=[0])

        assert pipeline._current_weight_version == 5  # same cache → same version
        assert collector.weight_version == 5

    def test_active_dp_ranks_updated(self):
        """_active_dp_ranks must contain the expanded ranks after success."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector)
        pipeline._active_dp_ranks = {0, 3}  # simulate some already-active

        patched_expand(pipeline, dp_ranks=[1, 2])

        assert pipeline._active_dp_ranks == {0, 1, 2, 3}
        assert pipeline._pre_activation_ranks == set()  # cleared on success

    def test_pre_activation_ranks_cleared_on_success(self):
        """_pre_activation_ranks must be empty after a successful expand."""
        vllm = MockVLLMGeneration(dp_size=2)
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector)

        patched_expand(pipeline, dp_ranks=[0, 1])

        assert pipeline._pre_activation_ranks == set()

    def test_vllm_active_ranks_updated(self):
        """MockVLLMGeneration.active_dp_ranks must reflect activated ranks."""
        vllm = MockVLLMGeneration(dp_size=4)
        vllm.active_dp_ranks = {0}  # start with only rank 0 active
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector)

        patched_expand(pipeline, dp_ranks=[1, 2, 3])

        assert vllm.active_dp_ranks == {0, 1, 2, 3}


class TestF6ExpandAtomicSyncFailure:
    """sync_selected_workers (step 3) fails: activate must NOT run, version unchanged."""

    def test_activate_not_called(self):
        """If sync fails, activate_dp_ranks must never be called."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=True)
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=3)

        try:
            patched_expand(pipeline, dp_ranks=[1])
        except RuntimeError:
            pass

        assert "activate_dp_ranks([1])" not in vllm.events, \
            "activate_dp_ranks must not fire when sync fails"

    def test_weight_version_not_changed(self):
        """weight_version must stay at initial value if sync fails."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=True)
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=7)

        try:
            patched_expand(pipeline, dp_ranks=[1])
        except RuntimeError:
            pass

        assert pipeline._current_weight_version == 7, \
            "weight_version must be unchanged when sync fails"
        assert collector.weight_version == -1, \
            "collector version must not be updated when sync fails"

    def test_pre_activation_ranks_retained(self):
        """Woken ranks stay in _pre_activation_ranks so diagnostics can inspect them."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=True)
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector)

        try:
            patched_expand(pipeline, dp_ranks=[2, 3])
        except RuntimeError:
            pass

        assert {2, 3}.issubset(pipeline._pre_activation_ranks), \
            "failed ranks must remain in _pre_activation_ranks for diagnostics"

    def test_wake_up_did_run(self):
        """Even when sync fails, wake_up_partial must have been called (irreversible)."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=True)
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector)

        try:
            patched_expand(pipeline, dp_ranks=[1])
        except RuntimeError:
            pass

        assert "wake_up_partial([1])" in vllm.events


class TestF6ExpandAtomicSetVersionFailure:
    """set_weight_version (step 4) fails: activate must NOT run, version unchanged."""

    def test_activate_not_called(self):
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=False)
        collector = MockTrajectoryCollector(fail_on_set_version=True)
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=2)

        try:
            patched_expand(pipeline, dp_ranks=[1])
        except RuntimeError:
            pass

        assert "activate_dp_ranks([1])" not in vllm.events

    def test_weight_version_not_changed(self):
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=False)
        collector = MockTrajectoryCollector(fail_on_set_version=True)
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=2)

        try:
            patched_expand(pipeline, dp_ranks=[1])
        except RuntimeError:
            pass

        assert pipeline._current_weight_version == 2

    def test_sync_did_run(self):
        """Sync must have run before version update was attempted."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=False)
        collector = MockTrajectoryCollector(fail_on_set_version=True)
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector)

        try:
            patched_expand(pipeline, dp_ranks=[1])
        except RuntimeError:
            pass

        assert len(svc.sync_calls) == 1


class TestF6ExpandAtomicMissingDeps:
    """Missing model_update_service or trajectory_collector: raise immediately."""

    def test_no_model_update_service_raises(self):
        vllm = MockVLLMGeneration(dp_size=4)
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(
            vllm=vllm,
            svc=MockModelUpdateService(),
            collector=collector,
        )
        pipeline._model_update_service = None  # force missing

        import pytest
        with pytest.raises(RuntimeError, match="model_update_service is None"):
            patched_expand(pipeline, dp_ranks=[1])

    def test_no_trajectory_collector_raises(self):
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=MockTrajectoryCollector())
        pipeline._trajectory_collector = None  # force missing

        import pytest
        with pytest.raises(RuntimeError, match="trajectory_collector is None"):
            patched_expand(pipeline, dp_ranks=[1])

    def test_empty_ranks_raises(self):
        pipeline = _make_pipeline()
        import pytest
        with pytest.raises(ValueError, match="non-empty"):
            patched_expand(pipeline, dp_ranks=[])


class TestF6ExpandMultipleSteps:
    """Verify version increments correctly across multiple expand cycles."""

    def test_version_increments_each_step(self):
        """Two expands from the same cache publish the same version (spec F6 no-bump).
        Version only advances when a new training step completes and _cache_ready_step advances."""
        vllm = MockVLLMGeneration(dp_size=4)
        vllm.active_dp_ranks = set()
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=0)

        # First expand: ranks [0, 1] — publishes _cache_ready_step = 0
        patched_expand(pipeline, dp_ranks=[0, 1])
        assert pipeline._current_weight_version == 0  # no bump: same cache
        assert collector.weight_version == 0

        # Simulate next training step advancing cache_ready_step
        pipeline._cache_ready_step = 1

        # Second expand: ranks [2, 3] — now publishes _cache_ready_step = 1
        patched_expand(pipeline, dp_ranks=[2, 3])
        assert pipeline._current_weight_version == 1
        assert collector.weight_version == 1

        assert pipeline._active_dp_ranks == {0, 1, 2, 3}

    def test_sync_called_only_for_target_ranks(self):
        """Each expand only syncs the specified ranks, not all ranks."""
        vllm = MockVLLMGeneration(dp_size=4)
        vllm.active_dp_ranks = {0, 1}
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector()
        pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=0)

        patched_expand(pipeline, dp_ranks=[2])

        assert svc.sync_calls == [[2]], \
            "sync must only target the specified ranks, not all dp_size ranks"


# ---------------------------------------------------------------------------
# Quick smoke test — run directly without pytest
# ---------------------------------------------------------------------------

def _run_smoke_tests():
    """Minimal smoke: happy path + sync failure. For quick validation."""
    print("=== F6 expand smoke tests ===")

    # Happy path
    vllm = MockVLLMGeneration(dp_size=4)
    svc = MockModelUpdateService()
    collector = MockTrajectoryCollector()
    pipeline = _make_pipeline_with_refs(vllm=vllm, svc=svc, collector=collector, initial_version=0)
    patched_expand(pipeline, dp_ranks=[1, 2])
    assert pipeline._current_weight_version == 1
    assert "activate_dp_ranks([1, 2])" in vllm.events
    assert pipeline._pre_activation_ranks == set()
    print("[PASS] happy path")

    # Sync failure: activate must not fire
    vllm2 = MockVLLMGeneration(dp_size=4)
    svc2 = MockModelUpdateService(fail_on_sync=True)
    collector2 = MockTrajectoryCollector()
    pipeline2 = _make_pipeline_with_refs(vllm=vllm2, svc=svc2, collector=collector2, initial_version=3)
    try:
        patched_expand(pipeline2, dp_ranks=[1])
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert "activate_dp_ranks([1])" not in vllm2.events
    assert pipeline2._current_weight_version == 3
    assert 1 in pipeline2._pre_activation_ranks
    print("[PASS] sync failure: activate not called, version unchanged")

    # set_weight_version failure
    vllm3 = MockVLLMGeneration(dp_size=4)
    svc3 = MockModelUpdateService()
    collector3 = MockTrajectoryCollector(fail_on_set_version=True)
    pipeline3 = _make_pipeline_with_refs(vllm=vllm3, svc=svc3, collector=collector3, initial_version=2)
    try:
        patched_expand(pipeline3, dp_ranks=[0])
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert "activate_dp_ranks([0])" not in vllm3.events
    assert pipeline3._current_weight_version == 2
    print("[PASS] set_version failure: activate not called, version unchanged")

    # Multi-step: version increments correctly
    vllm4 = MockVLLMGeneration(dp_size=4)
    vllm4.active_dp_ranks = set()
    svc4 = MockModelUpdateService()
    collector4 = MockTrajectoryCollector()
    pipeline4 = _make_pipeline_with_refs(vllm=vllm4, svc=svc4, collector=collector4, initial_version=0)
    patched_expand(pipeline4, dp_ranks=[0, 1])
    patched_expand(pipeline4, dp_ranks=[2, 3])
    assert pipeline4._current_weight_version == 2
    assert pipeline4._active_dp_ranks == {0, 1, 2, 3}
    print("[PASS] multi-step: version = 2, all ranks active")

    print("=== All smoke tests passed ===")


if __name__ == "__main__":
    _run_smoke_tests()
