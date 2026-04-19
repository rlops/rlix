"""Unit tests for Task 5 (RLixHooks protocol + grpo stub) and Task 6 (progress reporting).

Tests are self-contained: no Ray, no GPU, no NeMo RL runtime required.
The nemo-rl package directory is added to sys.path so imports resolve
without a pip install.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup: make nemo-rl importable without installation
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
NEMO_RL_ROOT = REPO_ROOT / "nemo-rl"

if str(NEMO_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEMO_RL_ROOT))


from nemo_rl.algorithms.rlix_hooks import NemoRLRLixHooks, NoOpRLixHooks, RLixHooks
from nemo_rl.algorithms.grpo import DO_TIME_SHARING, grpo_train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hooks(
    *,
    scheduler=None,
    pipeline_id: str = "ft_000000000000",
    cluster_ids: dict | None = None,
) -> NemoRLRLixHooks:
    if scheduler is None:
        scheduler = MagicMock()
    if cluster_ids is None:
        cluster_ids = {
            "actor_train": f"{pipeline_id}_actor_train",
            "actor_infer": f"{pipeline_id}_actor_infer",
        }
    return NemoRLRLixHooks(
        scheduler=scheduler,
        pipeline_id=pipeline_id,
        cluster_ids=cluster_ids,
    )


class _RecordingHooks:
    """Hook implementation that records every call for ordering assertions."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, int]] = []

    def before_generation(self, step: int) -> None:
        self.calls.append(("before_generation", step))

    def after_generation(self, step: int) -> None:
        self.calls.append(("after_generation", step))

    def before_training(self, step: int) -> None:
        self.calls.append(("before_training", step))

    def after_training(self, step: int) -> None:
        self.calls.append(("after_training", step))

    def before_weight_sync(self, step: int) -> None:
        self.calls.append(("before_weight_sync", step))

    def begin_progress_batch(self, step: int, count_intended: int) -> None:
        self.calls.append(("begin_progress_batch", step))

    def end_progress_batch(self, step: int, trajectories_collected: int) -> None:
        self.calls.append(("end_progress_batch", step))


# ---------------------------------------------------------------------------
# Task 5: Protocol + NoOpRLixHooks
# ---------------------------------------------------------------------------


class TestNoOpRLixHooks:
    def test_all_methods_callable_without_error(self) -> None:
        h = NoOpRLixHooks()
        h.before_generation(0)
        h.after_generation(0)
        h.before_training(0)
        h.after_training(0)
        h.before_weight_sync(0)
        h.begin_progress_batch(0, count_intended=10)
        h.end_progress_batch(0, trajectories_collected=5)

    def test_satisfies_rlix_hooks_protocol(self) -> None:
        assert isinstance(NoOpRLixHooks(), RLixHooks)

    def test_returns_none_for_all_methods(self) -> None:
        h = NoOpRLixHooks()
        assert h.before_generation(0) is None
        assert h.after_generation(0) is None
        assert h.before_training(0) is None
        assert h.after_training(0) is None
        assert h.before_weight_sync(0) is None
        assert h.begin_progress_batch(0, count_intended=1) is None
        assert h.end_progress_batch(0, trajectories_collected=1) is None


class TestNemoRLRLixHooksProtocol:
    def test_satisfies_rlix_hooks_protocol(self) -> None:
        assert isinstance(_make_hooks(), RLixHooks)

    def test_gpu_hooks_are_no_ops_until_task7(self) -> None:
        h = _make_hooks()
        # Should not raise and should return None (placeholders for Task 7)
        assert h.before_generation(0) is None
        assert h.after_generation(0) is None
        assert h.before_training(0) is None
        assert h.after_training(0) is None
        assert h.before_weight_sync(0) is None


# ---------------------------------------------------------------------------
# Task 5: DO_TIME_SHARING flag + grpo_train hook call ordering
# ---------------------------------------------------------------------------


class TestDoTimeSharingFlag:
    def test_flag_exists_and_is_bool(self) -> None:
        assert isinstance(DO_TIME_SHARING, bool)

    def test_flag_defaults_to_false(self) -> None:
        assert DO_TIME_SHARING is False


class TestGrpoTrainHookOrdering:
    def _run(self, hooks, num_steps: int = 1) -> None:
        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.num_steps = num_steps
        grpo_train(cfg, hooks=hooks)

    def test_all_five_hooks_called_each_step(self) -> None:
        rec = _RecordingHooks()
        self._run(rec, num_steps=1)
        method_names = [name for name, _ in rec.calls]
        assert "before_generation" in method_names
        assert "after_generation" in method_names
        assert "before_training" in method_names
        assert "after_training" in method_names
        assert "before_weight_sync" in method_names

    def test_hook_order_within_step(self) -> None:
        rec = _RecordingHooks()
        self._run(rec, num_steps=1)
        method_names = [name for name, _ in rec.calls]
        # Only the five Task-5 hooks are called in grpo_train (not begin/end_progress_batch)
        task5_hooks = [
            n
            for n in method_names
            if n in {
                "before_generation",
                "after_generation",
                "before_training",
                "after_training",
                "before_weight_sync",
            }
        ]
        assert task5_hooks == [
            "before_generation",
            "after_generation",
            "before_training",
            "after_training",
            "before_weight_sync",
        ], f"Wrong order: {task5_hooks}"

    def test_hooks_called_once_per_step(self) -> None:
        rec = _RecordingHooks()
        self._run(rec, num_steps=3)
        for hook_name in (
            "before_generation",
            "after_generation",
            "before_training",
            "after_training",
            "before_weight_sync",
        ):
            count = sum(1 for name, _ in rec.calls if name == hook_name)
            assert count == 3, f"{hook_name} called {count} times, expected 3"

    def test_step_index_passed_correctly(self) -> None:
        rec = _RecordingHooks()
        self._run(rec, num_steps=2)
        steps_for = {
            name: [s for n, s in rec.calls if n == name]
            for name in (
                "before_generation",
                "after_generation",
                "before_training",
                "after_training",
                "before_weight_sync",
            )
        }
        for name, steps in steps_for.items():
            assert steps == [0, 1], f"{name} got steps {steps}"

    def test_noop_hooks_used_when_none_passed(self) -> None:
        class _Cfg:
            num_steps = 1

        # Should complete without error even with hooks=None
        grpo_train(_Cfg(), hooks=None)


# ---------------------------------------------------------------------------
# Task 6: begin_progress_batch / end_progress_batch
# ---------------------------------------------------------------------------


class TestBeginProgressBatch:
    def test_resets_counter_and_bucket(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=100)
        assert h._collected_so_far == 0
        assert h._last_emitted_bucket == -1
        assert h._count_intended_for_step == 100
        assert h._current_step == 0

    def test_re_init_between_steps(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=50)
        h.end_progress_batch(0, trajectories_collected=50)
        h.begin_progress_batch(1, count_intended=200)
        assert h._collected_so_far == 0
        assert h._last_emitted_bucket == -1
        assert h._count_intended_for_step == 200
        assert h._current_step == 1

    def test_raises_on_zero_count_intended(self) -> None:
        h = _make_hooks()
        with pytest.raises(ValueError, match="count_intended must be > 0"):
            h.begin_progress_batch(0, count_intended=0)

    def test_raises_on_negative_count_intended(self) -> None:
        h = _make_hooks()
        with pytest.raises(ValueError, match="count_intended must be > 0"):
            h.begin_progress_batch(0, count_intended=-1)


class TestEndProgressBatch:
    def test_raises_without_begin(self) -> None:
        h = _make_hooks()
        with pytest.raises(RuntimeError, match="before begin_progress_batch"):
            h.end_progress_batch(0, trajectories_collected=1)

    def test_raises_on_step_mismatch(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=100)
        with pytest.raises(ValueError, match="step mismatch"):
            h.end_progress_batch(1, trajectories_collected=10)

    def test_raises_on_negative_trajectories(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=100)
        with pytest.raises(ValueError, match="trajectories_collected must be >= 0"):
            h.end_progress_batch(0, trajectories_collected=-1)

    def test_zero_trajectories_does_not_raise(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=100)
        h.end_progress_batch(0, trajectories_collected=0)

    def test_accumulates_collected_count(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=100)
        h.end_progress_batch(0, trajectories_collected=30)
        h.end_progress_batch(0, trajectories_collected=20)
        assert h._collected_so_far == 50

    def test_bucket_does_not_exceed_max(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=10)
        h.end_progress_batch(0, trajectories_collected=999)
        assert h._last_emitted_bucket == h._BUCKET_COUNT


class TestBucketDeduplication:
    """Verify emit fires at bucket boundaries and is deduplicated."""

    def _count_emits(self, h: NemoRLRLixHooks, batches: list[int], count_intended: int) -> int:
        emit_count = 0
        original_emit = h._emit_progress

        def counting_emit(step: int) -> None:
            nonlocal emit_count
            emit_count += 1

        h._emit_progress = counting_emit  # type: ignore[method-assign]
        h.begin_progress_batch(0, count_intended=count_intended)
        for n in batches:
            h.end_progress_batch(0, trajectories_collected=n)
        return emit_count

    def test_single_full_batch_emits_once(self) -> None:
        h = _make_hooks()
        count = self._count_emits(h, batches=[100], count_intended=100)
        assert count == 1

    def test_repeated_zero_batches_emit_once(self) -> None:
        h = _make_hooks()
        count = self._count_emits(h, batches=[0, 0, 0, 0], count_intended=100)
        # All batches land in bucket 0 → emit happens on first call, deduped after
        assert count == 1

    def test_two_percent_granularity(self) -> None:
        # 100 trajectories intended, deliver 1 at a time → at most 50 emits
        h = _make_hooks()
        count = self._count_emits(h, batches=[1] * 100, count_intended=100)
        assert count <= NemoRLRLixHooks._BUCKET_COUNT + 1  # +1 for bucket-0 on first emit

    def test_bucket_advances_at_correct_threshold(self) -> None:
        # With 50 intended, bucket ideally advances every 1 trajectory (1/50 * 50 = 1.0 per traj).
        # Allow one floating-point collision: expect at least 49 of the 50 possible bucket changes.
        h = _make_hooks()
        emitted_buckets: list[int] = []

        def record_emit(step: int) -> None:
            emitted_buckets.append(h._last_emitted_bucket)

        h._emit_progress = record_emit  # type: ignore[method-assign]
        h.begin_progress_batch(0, count_intended=50)

        for i in range(50):
            h.end_progress_batch(0, trajectories_collected=1)

        assert len(emitted_buckets) >= 49
        assert emitted_buckets == sorted(set(emitted_buckets))  # strictly increasing
        assert emitted_buckets[-1] == 50  # always reaches max

    def test_no_duplicate_emits_for_same_bucket(self) -> None:
        h = _make_hooks()
        emitted_buckets: list[int] = []

        def record_emit(step: int) -> None:
            emitted_buckets.append(h._last_emitted_bucket)

        h._emit_progress = record_emit  # type: ignore[method-assign]
        h.begin_progress_batch(0, count_intended=100)
        # Deliver 10 trajectories one at a time.
        # Buckets visited (floor(k/100*50) for k=1..10): 0,1,1,2,2,3,3,4,4,5
        # Distinct: 0,1,2,3,4,5 → 6 emits; no bucket emitted twice.
        for _ in range(10):
            h.end_progress_batch(0, trajectories_collected=1)
        assert emitted_buckets == sorted(set(emitted_buckets))  # strictly increasing → no duplicates
        assert len(emitted_buckets) == 6

    def test_complete_collection_reaches_max_bucket(self) -> None:
        h = _make_hooks()
        h.begin_progress_batch(0, count_intended=100)
        h.end_progress_batch(0, trajectories_collected=100)
        assert h._last_emitted_bucket == NemoRLRLixHooks._BUCKET_COUNT

    def test_overcollection_clamps_to_max_bucket(self) -> None:
        h = _make_hooks()
        emit_count = 0

        def counting_emit(step: int) -> None:
            nonlocal emit_count
            emit_count += 1

        h._emit_progress = counting_emit  # type: ignore[method-assign]
        h.begin_progress_batch(0, count_intended=10)
        h.end_progress_batch(0, trajectories_collected=5)
        h.end_progress_batch(0, trajectories_collected=5)
        pre_count = emit_count
        # Further overcollection should not advance the bucket past _BUCKET_COUNT
        h.end_progress_batch(0, trajectories_collected=100)
        assert emit_count == pre_count  # Bucket already at max, no new emit

    def test_emit_progress_called_with_correct_step(self) -> None:
        h = _make_hooks()
        emitted_steps: list[int] = []

        def record_step(step: int) -> None:
            emitted_steps.append(step)

        h._emit_progress = record_step  # type: ignore[method-assign]
        h.begin_progress_batch(7, count_intended=100)
        h.end_progress_batch(7, trajectories_collected=100)
        assert emitted_steps == [7]
