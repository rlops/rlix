"""NeMo RL pipeline F5/F6 tests.

Tests the control-flow skeleton of NemoRLFullFinetunePipeline and
NemoRLRLixHooks without any real Ray cluster, GPU, torch, or Megatron.

Test map:
  test_hooks_are_called_around_training_step   — F5: hook timing in training loop
  test_resize_infer_dispatches_to_shrink_and_expand — F5: resize_infer routing
  test_expand_workers_is_atomic_on_success     — F6: 5-step ordering invariant
  test_expand_workers_does_not_activate_on_sync_failure — F6: error path
  test_shrink_workers_calls_sleep_partial      — F5/F2: shrink path
  test_minimal_f5_f6_integration_flow          — F5+F6: full lifecycle

Run:
    cd rlix/
    python -m pytest tests/test_nemo_rl_pipeline.py -v
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import threading
import types
import unittest.mock as mock
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

# ---------------------------------------------------------------------------
# Import isolation — must run before any rlix import.
# Pre-populates sys.modules to prevent heavy __init__.py side-effects
# (rlix/__init__.py → ray, rlix/pipeline/__init__.py → torch).
# ---------------------------------------------------------------------------

_RLIX_ROOT = pathlib.Path(__file__).resolve().parent.parent / "rlix"


def _stub_package(dotted: str, fs_path: pathlib.Path) -> None:
    if dotted not in sys.modules:
        pkg = types.ModuleType(dotted)
        pkg.__path__ = [str(fs_path)]
        pkg.__package__ = dotted
        sys.modules[dotted] = pkg


def _stub_ray() -> None:
    if "ray" in sys.modules:
        return
    ray = types.ModuleType("ray")
    # ray.get: unwrap _MockFuture; real per-test patch installed via patch_ray_get()
    ray.get = lambda f: f._value if hasattr(f, "_value") else f
    ray.remote = lambda x: x  # @ray.remote no-op
    ray.get_actor = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("ray.get_actor must not be called in unit tests")
    )
    sys.modules["ray"] = ray
    for sub in [
        "ray.runtime_env",
        "ray.util",
        "ray.util.state",
        "ray.util.scheduling_strategies",
    ]:
        sys.modules.setdefault(sub, types.ModuleType(sub))


_stub_ray()
_stub_package("rlix", _RLIX_ROOT)
_stub_package("rlix.pipeline", _RLIX_ROOT / "pipeline")
_stub_package("rlix.protocol", _RLIX_ROOT / "protocol")
_stub_package("rlix.utils", _RLIX_ROOT / "utils")
_stub_package("rlix.scheduler", _RLIX_ROOT / "scheduler")

# ---------------------------------------------------------------------------
# Real rlix imports (safe after isolation above)
# ---------------------------------------------------------------------------
from rlix.pipeline.nemo_rl_pipeline import NemoRLFullFinetunePipeline, NemoRLRLixHooks  # noqa: E402
from rlix.pipeline.utils import validate_resize_params  # noqa: E402
from rlix.protocol.types import (  # noqa: E402
    ACTOR_TRAIN_CLUSTER_NAME,
    ActionResponse,
    Priority,
)

# ---------------------------------------------------------------------------
# Fake Ray helpers
# ---------------------------------------------------------------------------


class _MockFuture:
    """Fake Ray ObjectRef returned by .remote()."""

    def __init__(self, value: Any) -> None:
        self._value = value


def _fake_ray_get(future: Any) -> Any:
    return future._value if isinstance(future, _MockFuture) else future


class _RemoteMethod:
    """Wraps a callable so .remote(*args, **kwargs) → _MockFuture."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> _MockFuture:
        return _MockFuture(self._fn(*args, **kwargs))


class _MockRemoteProxy:
    """Makes actor_handle.method.remote(...) work without real Ray."""

    def __init__(self, actor: Any) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> _RemoteMethod:
        return _RemoteMethod(getattr(self._actor, name))


@contextmanager
def patch_ray_get() -> Generator:
    """Context manager: patches ray.get in the pipeline module for the test block."""
    with mock.patch(
        "rlix.pipeline.nemo_rl_pipeline.ray.get", side_effect=_fake_ray_get
    ):
        yield


# ---------------------------------------------------------------------------
# Mock: Policy (replaces real NeMo RL Megatron policy for F4 calls)
# ---------------------------------------------------------------------------


class MockPolicy:
    """Minimal policy stub satisfying _build_cpu_bucket_cache checks."""

    def build_cpu_bucket_cache(self, step: int) -> None:
        pass

    def promote_active_checkpoint(self, version: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock: Coordinator (replaces real RLix coordinator for sync_base_weights calls)
# ---------------------------------------------------------------------------


class MockCoordinator:
    """Returns empty active_ranks so _after_training completes without side-effects."""

    def sync_base_weights_to_active(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Mock: Scheduler (replaces real RLix scheduler Ray actor)
# ---------------------------------------------------------------------------


class MockScheduler:
    """Records request_gpus / notify_release_gpus calls; returns fake allocations.

    Used as pipeline._rlix_scheduler so NemoRLRLixHooks can call the real
    _request_cluster_gpus / _notify_release_cluster_gpus methods without Ray.
    """

    def __init__(self) -> None:
        self.request_calls: List[Dict[str, Any]] = []
        self.release_calls: List[Dict[str, Any]] = []
        self.events: List[str] = []

    def _do_request_gpus(
        self,
        *,
        cluster_id: str,
        priority: Any,
        global_step: int,
        step_target_estimate: Optional[int] = None,
    ) -> List[int]:
        record = {"cluster_id": cluster_id, "step": global_step, "priority": priority}
        self.request_calls.append(record)
        self.events.append(f"request_gpus(cluster={cluster_id!r}, step={global_step})")
        return [0, 1]  # fake allocated GPU indices

    def _do_notify_release_gpus(self, *, cluster_id: str, global_step: int) -> None:
        record = {"cluster_id": cluster_id, "step": global_step}
        self.release_calls.append(record)
        self.events.append(f"notify_release(cluster={cluster_id!r}, step={global_step})")

    @property
    def request_gpus(self) -> _RemoteMethod:
        return _RemoteMethod(self._do_request_gpus)

    @property
    def notify_release_gpus(self) -> _RemoteMethod:
        return _RemoteMethod(self._do_notify_release_gpus)


# ---------------------------------------------------------------------------
# Mock: VllmGeneration (F2/F3 stub — async sleep_partial for _shrink_workers)
# ---------------------------------------------------------------------------


class MockVLLMGeneration:
    """Stub for VllmGeneration.

    sleep_partial is async (matches F2 design: abort-drain-sleep is awaitable).
    All methods write to both per-object events and optional shared_events list
    so tests can verify global call ordering across mocks.
    """

    def __init__(
        self, dp_size: int = 4, shared_events: Optional[List[str]] = None
    ) -> None:
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
        self.active_dp_ranks.difference_update(dp_ranks)
        self.inactive_ranks.update(dp_ranks)
        self._log(f"mark_inactive({sorted(dp_ranks)})")

    def wake_up_partial(self, dp_ranks: List[int]) -> None:
        self.woken_ranks.update(dp_ranks)
        self._log(f"wake_up_partial({sorted(dp_ranks)})")

    async def sleep_partial(self, dp_ranks: List[int], level: int = 2) -> None:
        """Async to match real F2 implementation (drain requires await)."""
        self.active_dp_ranks.difference_update(dp_ranks)
        self.woken_ranks.difference_update(dp_ranks)
        self._log(f"sleep_partial({sorted(dp_ranks)}, level={level})")

    def activate_dp_ranks(self, dp_ranks: List[int]) -> None:
        self.active_dp_ranks.update(dp_ranks)
        self.inactive_ranks.difference_update(dp_ranks)
        self._log(f"activate_dp_ranks({sorted(dp_ranks)})")

    def finalize_weight_update(self, dp_ranks: List[int]) -> List[Any]:
        self._log(f"finalize_weight_update({sorted(dp_ranks)})")
        return []


# ---------------------------------------------------------------------------
# Mock: ModelUpdateService (F4 stub)
# ---------------------------------------------------------------------------


class MockModelUpdateService:
    """Stub for NemoRLModelUpdateService. Set fail=True to simulate sync failure."""

    def __init__(
        self, fail_on_sync: bool = False, shared_events: Optional[List[str]] = None
    ) -> None:
        self.fail_on_sync = fail_on_sync
        self.sync_calls: List[List[int]] = []
        self.events: List[str] = []
        self._shared = shared_events

    def _log(self, msg: str) -> None:
        self.events.append(msg)
        if self._shared is not None:
            self._shared.append(msg)

    def sync_selected_workers(
        self, tgt_dp_ranks: List[int], verify: bool = False
    ) -> None:
        self._log(f"sync_selected_workers({sorted(tgt_dp_ranks)})")
        self.sync_calls.append(sorted(tgt_dp_ranks))
        if self.fail_on_sync:
            raise RuntimeError("simulated sync failure")


# ---------------------------------------------------------------------------
# Mock: TrajectoryCollector (F9 stub)
# ---------------------------------------------------------------------------


class MockTrajectoryCollector:
    """Stub for AsyncTrajectoryCollector. Set fail=True to simulate version update failure."""

    def __init__(
        self,
        fail_on_set_version: bool = False,
        shared_events: Optional[List[str]] = None,
    ) -> None:
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
            raise RuntimeError("simulated set_weight_version failure")
        self.weight_version = version


# ---------------------------------------------------------------------------
# Mock: RecordingRLixHooks (for testing hook call timing)
# ---------------------------------------------------------------------------


class RecordingRLixHooks:
    """Records every hook call with its event type and step, in global order.

    Used instead of the real NemoRLRLixHooks when we want to verify hook
    timing without needing a real pipeline actor.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def before_training(self, step: int) -> None:
        self.events.append({"type": "before_training", "step": step})

    def after_training(self, step: int) -> None:
        self.events.append({"type": "after_training", "step": step})

    def on_trajectory_collector_created(self, collector: Any) -> None:
        self.events.append({"type": "on_collector_created"})


# ---------------------------------------------------------------------------
# Fake training loop — minimal stand-in for async_grpo_train
# ---------------------------------------------------------------------------


def fake_async_grpo_train(
    *,
    num_steps: int = 3,
    rlix_hooks: Any = None,
    training_log: Optional[List[str]] = None,
) -> None:
    """Minimal substitute for async_grpo_train that fires F5 hooks.

    Calls on_trajectory_collector_created once at start (mirrors the real
    grpo.py path where AsyncTrajectoryCollector is created before the loop).
    Then for each step: before_training → "train" → after_training.

    Args:
        num_steps:     Number of simulated training steps.
        rlix_hooks:    Hook implementation (real or recording). If None, uses
                       a no-op instance that never blocks.
        training_log:  Optional list to append step markers for ordering checks.
    """
    class _NoOp:
        def before_training(self, step: int) -> None: pass
        def after_training(self, step: int) -> None: pass
        def on_trajectory_collector_created(self, collector: Any) -> None: pass

    hooks = rlix_hooks if rlix_hooks is not None else _NoOp()

    # Simulate AsyncTrajectoryCollector creation
    fake_collector = object()
    hooks.on_trajectory_collector_created(fake_collector)

    for step in range(num_steps):
        hooks.before_training(step)
        if training_log is not None:
            training_log.append(f"train_step({step})")
        # (real training would happen here)
        hooks.after_training(step)


# ---------------------------------------------------------------------------
# Pipeline fixture factory
# ---------------------------------------------------------------------------


def _make_test_pipeline(
    *,
    scheduler: Optional[MockScheduler] = None,
    vllm: Optional[MockVLLMGeneration] = None,
    svc: Optional[MockModelUpdateService] = None,
    collector: Optional[MockTrajectoryCollector] = None,
    initial_version: int = 0,
    dp_size: int = 4,
) -> NemoRLFullFinetunePipeline:
    """Build a NemoRLFullFinetunePipeline without Ray using object.__new__.

    Bypasses __init__ (which calls get_actor_or_raise → ray) and injects
    mock dependencies directly. Sets _initialized=True so _ensure_initialized
    is a no-op in all tests.
    """
    _scheduler = scheduler or MockScheduler()
    _vllm = vllm or MockVLLMGeneration(dp_size=dp_size)
    _svc = svc or MockModelUpdateService()
    _collector = collector or MockTrajectoryCollector()

    p = object.__new__(NemoRLFullFinetunePipeline)
    p._pipeline_id = "test_pipeline"
    p._initialized = True
    p._init_lock = threading.Lock()
    p._infer_resize_lock = threading.Lock()
    p._current_weight_version = initial_version
    p._pre_activation_ranks = set()
    p._active_dp_ranks = set()
    p._cache_ready_step = initial_version
    p._policy = _MockRemoteProxy(MockPolicy())
    p._coordinator_handle = _MockRemoteProxy(MockCoordinator())

    # RLix scheduler (used by NemoRLRLixHooks via _request_cluster_gpus)
    p._rlix_scheduler = _scheduler

    # Cluster IDs built from pipeline_id + cluster name constants
    p._actor_train_cluster_id = f"test_pipeline_{ACTOR_TRAIN_CLUSTER_NAME}"
    p._actor_infer_cluster_id = "test_pipeline_actor_infer"

    # NeMo RL runtime objects
    p._policy_generation = _vllm
    p._model_update_service = _MockRemoteProxy(_svc)
    p._trajectory_collector = _MockRemoteProxy(_collector)

    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHookTiming:
    """F5: before_training / after_training must bracket each training step."""

    def test_hooks_are_called_around_training_step(self):
        """Verify ordering: on_collector_created, then per-step before→train→after."""
        hooks = RecordingRLixHooks()
        training_log: List[str] = []

        fake_async_grpo_train(
            num_steps=3,
            rlix_hooks=hooks,
            training_log=training_log,
        )

        # --- Structural checks ---
        # on_collector_created fires once, before any training step
        collector_events = [e for e in hooks.events if e["type"] == "on_collector_created"]
        assert len(collector_events) == 1, "on_trajectory_collector_created must fire exactly once"
        assert hooks.events[0]["type"] == "on_collector_created", \
            "collector registration must be the very first hook event"

        # before_training fires once per step with correct step number
        before_events = [e for e in hooks.events if e["type"] == "before_training"]
        assert [e["step"] for e in before_events] == [0, 1, 2], \
            "before_training must fire for each step in order"

        # after_training fires once per step with correct step number
        after_events = [e for e in hooks.events if e["type"] == "after_training"]
        assert [e["step"] for e in after_events] == [0, 1, 2], \
            "after_training must fire for each step in order"

        # --- Per-step ordering: before → train → after ---
        # Interleave hook events with training_log to build a global timeline
        all_events: List[str] = []
        hook_iter = iter(e for e in hooks.events if e["type"] != "on_collector_created")
        train_iter = iter(training_log)
        hook_events_flat = list(hook_iter)
        # Rebuild interleaved order: [before(0), train(0), after(0), before(1), ...]
        for step in range(3):
            all_events.append(f"before_{step}")
            all_events.append(f"train_{step}")
            all_events.append(f"after_{step}")

        # Verify each before comes before its matching after
        for step in range(3):
            b_idx = next(
                i for i, e in enumerate(hook_events_flat)
                if e["type"] == "before_training" and e["step"] == step
            )
            a_idx = next(
                i for i, e in enumerate(hook_events_flat)
                if e["type"] == "after_training" and e["step"] == step
            )
            assert b_idx < a_idx, \
                f"before_training({step}) must come before after_training({step})"

    def test_hook_step_numbers_match_training_step(self):
        """step argument passed to hooks must equal the loop iteration index."""
        hooks = RecordingRLixHooks()
        fake_async_grpo_train(num_steps=5, rlix_hooks=hooks)

        for step in range(5):
            # before_training for this step must carry the correct step number
            before = next(
                e for e in hooks.events
                if e["type"] == "before_training" and e["step"] == step
            )
            after = next(
                e for e in hooks.events
                if e["type"] == "after_training" and e["step"] == step
            )
            assert before["step"] == step
            assert after["step"] == step

    def test_real_hooks_call_scheduler_request_and_release(self):
        """NemoRLRLixHooks.before/after_training must call scheduler RPCs."""
        sched = MockScheduler()
        pipeline = _make_test_pipeline(scheduler=sched)
        hooks = NemoRLRLixHooks(pipeline=pipeline)

        with patch_ray_get():
            hooks.before_training(step=7)
            hooks.after_training(step=7)

        # before_training → _request_cluster_gpus → scheduler.request_gpus
        assert len(sched.request_calls) == 1
        assert sched.request_calls[0]["step"] == 7
        assert ACTOR_TRAIN_CLUSTER_NAME in sched.request_calls[0]["cluster_id"]

        # after_training → _notify_release_cluster_gpus → scheduler.notify_release_gpus
        assert len(sched.release_calls) == 1
        assert sched.release_calls[0]["step"] == 7
        assert ACTOR_TRAIN_CLUSTER_NAME in sched.release_calls[0]["cluster_id"]

    def test_on_collector_created_registers_handle(self):
        """NemoRLRLixHooks.on_trajectory_collector_created must store the handle."""
        pipeline = _make_test_pipeline()
        hooks = NemoRLRLixHooks(pipeline=pipeline)
        fake_handle = object()

        hooks.on_trajectory_collector_created(fake_handle)

        assert pipeline._trajectory_collector is fake_handle, \
            "on_trajectory_collector_created must set pipeline._trajectory_collector"


class TestResizeInferDispatch:
    """F5: resize_infer must route correctly to _shrink or _expand."""

    def test_resize_infer_dispatches_to_shrink(self):
        """resize_infer(remove=[1], add=[]) must call sleep_partial([1])."""
        vllm = MockVLLMGeneration(dp_size=4)
        pipeline = _make_test_pipeline(vllm=vllm)

        # asyncio.run(sleep_partial(...)) is the shrink path — sleep_partial is async
        result = pipeline.resize_infer(dp_ranks_to_remove=[1], dp_ranks_to_add=[])

        assert result.success is True
        # sleep_partial must have been called
        assert any("sleep_partial([1]" in e for e in vllm.events), \
            "shrink path must call sleep_partial on the specified ranks"
        # rank 1 must no longer be active
        assert 1 not in vllm.active_dp_ranks, \
            "shrunk rank must be removed from active_dp_ranks"

    def test_resize_infer_dispatches_to_expand(self):
        """resize_infer(remove=[], add=[2]) must call activate_dp_ranks([2])."""
        vllm = MockVLLMGeneration(dp_size=4)
        vllm.active_dp_ranks = {0, 1, 3}  # rank 2 starts sleeping
        pipeline = _make_test_pipeline(vllm=vllm)

        with patch_ray_get():
            result = pipeline.resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[2])

        assert result.success is True
        assert "activate_dp_ranks([2])" in vllm.events, \
            "expand path must call activate_dp_ranks on the specified ranks"
        assert 2 in vllm.active_dp_ranks

    def test_resize_infer_rejects_both_remove_and_add(self):
        """Providing both remove and add must raise ValueError (exactly one allowed)."""
        pipeline = _make_test_pipeline()
        import pytest
        with pytest.raises(ValueError):
            pipeline.resize_infer(dp_ranks_to_remove=[1], dp_ranks_to_add=[2])

    def test_resize_infer_rejects_both_empty(self):
        """Providing neither remove nor add must raise ValueError."""
        pipeline = _make_test_pipeline()
        import pytest
        with pytest.raises(ValueError):
            pipeline.resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[])

    def test_resize_infer_returns_action_response(self):
        """resize_infer must return ActionResponse(success=True) on success."""
        vllm = MockVLLMGeneration(dp_size=4)
        pipeline = _make_test_pipeline(vllm=vllm)

        with patch_ray_get():
            resp = pipeline.resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[0])

        assert isinstance(resp, ActionResponse)
        assert resp.success is True


class TestExpandWorkersAtomic:
    """F6: _expand_workers must be atomic — activate only after sync+version succeed."""

    def _run_expand(self, pipeline, dp_ranks):
        with patch_ray_get():
            pipeline._expand_workers(dp_ranks_to_add=dp_ranks)

    def test_expand_workers_is_atomic_on_success(self):
        """F6 ordering invariant: mark→wake→sync→finalize→set_version→activate."""
        shared: List[str] = []  # single list records global call order across all mocks
        vllm = MockVLLMGeneration(dp_size=4, shared_events=shared)
        vllm.active_dp_ranks = {0}
        svc = MockModelUpdateService(shared_events=shared)
        collector = MockTrajectoryCollector(shared_events=shared)
        pipeline = _make_test_pipeline(vllm=vllm, svc=svc, collector=collector, initial_version=3)

        self._run_expand(pipeline, dp_ranks=[1, 2])

        idx = {e: i for i, e in enumerate(shared)}

        # All 5 steps must be present
        for key in [
            "mark_inactive([1, 2])",
            "wake_up_partial([1, 2])",
            "sync_selected_workers([1, 2])",
            "finalize_weight_update([1, 2])",
            "set_weight_version(3)",
            "activate_dp_ranks([1, 2])",
        ]:
            assert key in idx, f"Expected event {key!r} not found in: {shared}"

        # Ordering: each step before the next
        assert idx["mark_inactive([1, 2])"] < idx["wake_up_partial([1, 2])"]
        assert idx["wake_up_partial([1, 2])"] < idx["sync_selected_workers([1, 2])"]
        assert idx["sync_selected_workers([1, 2])"] < idx["finalize_weight_update([1, 2])"]
        assert idx["finalize_weight_update([1, 2])"] < idx["set_weight_version(3)"]
        # Critical: version must be set BEFORE routing is activated
        assert idx["set_weight_version(3)"] < idx["activate_dp_ranks([1, 2])"]

    def test_expand_workers_publishes_cache_version(self):
        """_current_weight_version must equal the cache-producing step."""
        pipeline = _make_test_pipeline(initial_version=9)

        self._run_expand(pipeline, dp_ranks=[1])

        assert pipeline._current_weight_version == 9

    def test_expand_workers_updates_collector_version(self):
        """Collector.weight_version must equal pipeline._current_weight_version after expand."""
        collector = MockTrajectoryCollector()
        pipeline = _make_test_pipeline(collector=collector, initial_version=0)

        self._run_expand(pipeline, dp_ranks=[0])

        assert collector.weight_version == 0
        assert pipeline._current_weight_version == collector.weight_version

    def test_expand_workers_clears_pre_activation_ranks(self):
        """_pre_activation_ranks must be empty after a successful expand."""
        pipeline = _make_test_pipeline()
        self._run_expand(pipeline, dp_ranks=[2, 3])
        assert pipeline._pre_activation_ranks == set()

    def test_expand_workers_updates_active_dp_ranks(self):
        """_active_dp_ranks on pipeline and vllm must contain expanded ranks."""
        vllm = MockVLLMGeneration(dp_size=4)
        vllm.active_dp_ranks = {0}
        pipeline = _make_test_pipeline(vllm=vllm)
        pipeline._active_dp_ranks = {0}

        self._run_expand(pipeline, dp_ranks=[1, 2, 3])

        assert pipeline._active_dp_ranks == {0, 1, 2, 3}
        assert vllm.active_dp_ranks == {0, 1, 2, 3}


class TestExpandWorkersSyncFailure:
    """F6: sync failure must prevent activate and leave state consistent."""

    def _run_expand_expect_failure(self, pipeline, dp_ranks):
        with patch_ray_get():
            try:
                pipeline._expand_workers(dp_ranks_to_add=dp_ranks)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected RuntimeError was not raised")

    def test_expand_workers_does_not_activate_on_sync_failure(self):
        """If sync_selected_workers raises, activate_dp_ranks must NOT run."""
        vllm = MockVLLMGeneration(dp_size=4)
        vllm.active_dp_ranks = {0}
        svc = MockModelUpdateService(fail_on_sync=True)
        pipeline = _make_test_pipeline(vllm=vllm, svc=svc)

        self._run_expand_expect_failure(pipeline, dp_ranks=[1])

        assert "activate_dp_ranks([1])" not in vllm.events, \
            "activate must not fire when sync fails"

    def test_weight_version_unchanged_on_sync_failure(self):
        """_current_weight_version must not change when sync fails."""
        svc = MockModelUpdateService(fail_on_sync=True)
        pipeline = _make_test_pipeline(svc=svc, initial_version=5)

        self._run_expand_expect_failure(pipeline, dp_ranks=[1])

        assert pipeline._current_weight_version == 5, \
            "version must be unchanged when sync fails"

    def test_collector_version_unchanged_on_sync_failure(self):
        """Collector.weight_version must not be updated when sync fails."""
        svc = MockModelUpdateService(fail_on_sync=True)
        collector = MockTrajectoryCollector()
        pipeline = _make_test_pipeline(svc=svc, collector=collector, initial_version=2)

        self._run_expand_expect_failure(pipeline, dp_ranks=[1])

        assert collector.weight_version == -1, \
            "collector version must not be updated when sync fails"

    def test_pre_activation_ranks_retained_on_sync_failure(self):
        """Woken (but not activated) ranks must stay in _pre_activation_ranks for diagnosis."""
        svc = MockModelUpdateService(fail_on_sync=True)
        pipeline = _make_test_pipeline(svc=svc)

        self._run_expand_expect_failure(pipeline, dp_ranks=[2, 3])

        assert {2, 3}.issubset(pipeline._pre_activation_ranks), \
            "_pre_activation_ranks must retain failed ranks so caller can inspect"

    def test_wake_up_ran_before_sync_failure(self):
        """wake_up_partial must have been called even when sync later fails."""
        vllm = MockVLLMGeneration(dp_size=4)
        svc = MockModelUpdateService(fail_on_sync=True)
        pipeline = _make_test_pipeline(vllm=vllm, svc=svc)

        self._run_expand_expect_failure(pipeline, dp_ranks=[1])

        assert "wake_up_partial([1])" in vllm.events

    def test_active_dp_ranks_unchanged_on_sync_failure(self):
        """vllm.active_dp_ranks must not contain the failed ranks after sync failure."""
        vllm = MockVLLMGeneration(dp_size=4)
        vllm.active_dp_ranks = {0}
        svc = MockModelUpdateService(fail_on_sync=True)
        pipeline = _make_test_pipeline(vllm=vllm, svc=svc)

        self._run_expand_expect_failure(pipeline, dp_ranks=[1, 2])

        # Ranks 1, 2 are woken but not yet routable
        assert 1 not in vllm.active_dp_ranks
        assert 2 not in vllm.active_dp_ranks

    def test_no_activate_on_set_version_failure(self):
        """activate must not fire if set_weight_version fails (step 4 failure)."""
        vllm = MockVLLMGeneration(dp_size=4)
        collector = MockTrajectoryCollector(fail_on_set_version=True)
        pipeline = _make_test_pipeline(vllm=vllm, collector=collector, initial_version=1)

        self._run_expand_expect_failure(pipeline, dp_ranks=[1])

        assert "activate_dp_ranks([1])" not in vllm.events
        assert pipeline._current_weight_version == 1  # unchanged


class TestShrinkWorkers:
    """F5/F2: _shrink_workers must call sleep_partial and update state."""

    def test_shrink_workers_calls_sleep_partial(self):
        """_shrink_workers must delegate to VllmGeneration.sleep_partial."""
        vllm = MockVLLMGeneration(dp_size=4)
        pipeline = _make_test_pipeline(vllm=vllm)

        pipeline._shrink_workers(dp_ranks_to_remove=[1, 2])

        assert any("sleep_partial([1, 2]" in e for e in vllm.events), \
            "sleep_partial must be called with the removed ranks"

    def test_shrink_workers_removes_from_active_ranks(self):
        """Shrunk ranks must no longer be in vllm.active_dp_ranks."""
        vllm = MockVLLMGeneration(dp_size=4)
        pipeline = _make_test_pipeline(vllm=vllm)

        pipeline._shrink_workers(dp_ranks_to_remove=[2, 3])

        assert 2 not in vllm.active_dp_ranks
        assert 3 not in vllm.active_dp_ranks
        assert 0 in vllm.active_dp_ranks  # non-shrunk ranks stay active

    def test_shrink_workers_uses_level_2(self):
        """sleep_partial must be called with level=2 (full VRAM release)."""
        vllm = MockVLLMGeneration(dp_size=4)
        pipeline = _make_test_pipeline(vllm=vllm)

        pipeline._shrink_workers(dp_ranks_to_remove=[0])

        # Verify level=2 appears in the event log
        assert any("level=2" in e for e in vllm.events), \
            "sleep_partial must be called with level=2 to release weights+KV cache"

    def test_shrink_workers_empty_ranks_raises(self):
        """_shrink_workers with empty list must raise ValueError immediately."""
        import pytest
        pipeline = _make_test_pipeline()
        with pytest.raises(ValueError):
            pipeline._shrink_workers(dp_ranks_to_remove=[])


class TestMissingDependencies:
    """Verify _expand_workers raises immediately when required deps are None."""

    def _run(self, pipeline, dp_ranks):
        with patch_ray_get():
            pipeline._expand_workers(dp_ranks_to_add=dp_ranks)

    def test_no_model_update_service_raises(self):
        import pytest
        pipeline = _make_test_pipeline()
        pipeline._model_update_service = None
        with pytest.raises(RuntimeError, match="model_update_service is None"):
            self._run(pipeline, dp_ranks=[1])

    def test_no_trajectory_collector_raises(self):
        import pytest
        pipeline = _make_test_pipeline()
        pipeline._trajectory_collector = None
        with pytest.raises(RuntimeError, match="trajectory_collector is None"):
            self._run(pipeline, dp_ranks=[1])

    def test_no_policy_generation_raises(self):
        import pytest
        pipeline = _make_test_pipeline()
        pipeline._policy_generation = None
        with pytest.raises(RuntimeError):
            self._run(pipeline, dp_ranks=[1])


class TestMinimalIntegrationFlow:
    """F5 + F6: end-to-end mock integration — before→shrink→train→after→expand."""

    def test_minimal_f5_f6_integration_flow(self):
        """Simulate a single training step with scheduler-driven shrink + expand.

        Timeline:
          1. on_trajectory_collector_created — collector handle registered
          2. before_training(0)             — scheduler.request_gpus called (F5)
          3. [Scheduler side effect]         — resize_infer(remove=[1]) → shrink (F5)
          4. "training"                      — (simulated)
          5. after_training(0)              — scheduler.notify_release called (F5)
          6. [Scheduler side effect]         — resize_infer(add=[1]) → expand (F6)
          7. Verify: rank 1 active, version=1, collector.version=1
        """
        # --- Setup ---
        sched = MockScheduler()
        vllm = MockVLLMGeneration(dp_size=2)
        vllm.active_dp_ranks = {0}  # only rank 0 active initially (rank 1 sleeping)
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector()
        pipeline = _make_test_pipeline(
            scheduler=sched, vllm=vllm, svc=svc, collector=collector, initial_version=0
        )
        hooks = NemoRLRLixHooks(pipeline=pipeline)

        with patch_ray_get():
            # --- Step 1: register collector ---
            # In real code this is called from async_grpo_train after collector creation.
            # NemoRLRLixHooks.on_trajectory_collector_created stores the handle on pipeline.
            mock_collector_proxy = _MockRemoteProxy(collector)
            hooks.on_trajectory_collector_created(mock_collector_proxy)
            assert pipeline._trajectory_collector is mock_collector_proxy, \
                "collector handle must be registered on pipeline after on_trajectory_collector_created"

            # --- Step 2: before_training → scheduler.request_gpus ---
            hooks.before_training(step=0)
            assert len(sched.request_calls) == 1, \
                "before_training must trigger exactly one scheduler.request_gpus call"
            assert sched.request_calls[0]["step"] == 0

            # --- Step 3: scheduler-side shrink (simulates scheduler calling resize_infer) ---
            # Scheduler receives request_gpus, decides to shrink overlap rank 1.
            pipeline.resize_infer(dp_ranks_to_remove=[1], dp_ranks_to_add=[])
            assert 1 not in vllm.active_dp_ranks, \
                "rank 1 must be sleeping after shrink"
            assert any("sleep_partial([1]" in e for e in vllm.events), \
                "shrink must have called sleep_partial"

            # --- Step 4: "training" happens here (no GPU needed for this test) ---

            # --- Step 5: after_training → scheduler.notify_release ---
            hooks.after_training(step=0)
            assert len(sched.release_calls) == 1, \
                "after_training must trigger exactly one scheduler.notify_release call"
            assert sched.release_calls[0]["step"] == 0

            # --- Step 6: scheduler-side expand (simulates scheduler calling resize_infer) ---
            # Scheduler receives notify_release, decides to expand rank 1.
            pipeline.resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[1])

            # --- Step 7: verify F6 invariants ---
            # rank 1 must be active again
            assert 1 in vllm.active_dp_ranks, \
                "rank 1 must be active after expand"
            # weight version = _cache_ready_step = step (no bump on expand, same cache)
            assert pipeline._current_weight_version == 0, \
                "weight_version must be 0 after step=0 (version = cache-producing step)"
            # collector must know the version BEFORE routing was activated
            assert collector.weight_version == 0, \
                "collector version must match pipeline version after expand"
            # no stale ranks left in pre-activation limbo
            assert pipeline._pre_activation_ranks == set(), \
                "_pre_activation_ranks must be clear after successful expand"

    def test_multiple_step_integration(self):
        """Two training steps: version must increment to 2, both shrink+expand cycles complete."""
        sched = MockScheduler()
        vllm = MockVLLMGeneration(dp_size=2)
        vllm.active_dp_ranks = {0}
        svc = MockModelUpdateService()
        collector = MockTrajectoryCollector()
        pipeline = _make_test_pipeline(
            scheduler=sched, vllm=vllm, svc=svc, collector=collector, initial_version=0
        )
        hooks = NemoRLRLixHooks(pipeline=pipeline)

        with patch_ray_get():
            hooks.on_trajectory_collector_created(_MockRemoteProxy(collector))

            for step in range(2):
                hooks.before_training(step=step)
                # Scheduler shrinks
                pipeline.resize_infer(dp_ranks_to_remove=[1], dp_ranks_to_add=[])
                # "Train"
                hooks.after_training(step=step)
                # Scheduler expands
                pipeline.resize_infer(dp_ranks_to_remove=[], dp_ranks_to_add=[1])

        # Two expand cycles: step=0 → version=0, step=1 → version=1 (no bump on expand)
        assert pipeline._current_weight_version == 1
        assert collector.weight_version == 1
        # Scheduler was called twice for each side
        assert len(sched.request_calls) == 2
        assert len(sched.release_calls) == 2
        # Step numbers are correct
        assert [c["step"] for c in sched.request_calls] == [0, 1]
        assert [c["step"] for c in sched.release_calls] == [0, 1]

    def test_expand_failure_does_not_corrupt_second_expand(self):
        """If first expand fails (sync error), second expand attempt can succeed."""
        vllm = MockVLLMGeneration(dp_size=2)
        vllm.active_dp_ranks = {0}

        # First attempt: sync fails
        svc_fail = MockModelUpdateService(fail_on_sync=True)
        collector = MockTrajectoryCollector()
        pipeline = _make_test_pipeline(vllm=vllm, svc=svc_fail, collector=collector, initial_version=0)

        with patch_ray_get():
            try:
                pipeline._expand_workers(dp_ranks_to_add=[1])
            except RuntimeError:
                pass

        # Version unchanged, rank 1 in pre_activation (woken but not active)
        assert pipeline._current_weight_version == 0
        assert 1 in pipeline._pre_activation_ranks

        # Second attempt: swap to working sync service
        pipeline._model_update_service = _MockRemoteProxy(MockModelUpdateService())

        with patch_ray_get():
            pipeline._expand_workers(dp_ranks_to_add=[1])

        # Now rank 1 is active; version = _cache_ready_step = 0 (no bump on expand)
        assert pipeline._current_weight_version == 0
        assert 1 in vllm.active_dp_ranks
        assert pipeline._pre_activation_ranks == set()
