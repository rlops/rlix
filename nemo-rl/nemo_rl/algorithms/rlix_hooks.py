"""RLix scheduling hooks for NeMo RL GRPO training loop integration.

Provides:
  - RLixHooks: Protocol defining the hook interface (Task 5)
  - NoOpRLixHooks: No-op implementation for standalone NeMo RL runs (Task 5)
  - NemoRLRLixHooks: Actual RLix scheduler integration (Tasks 5 + 6)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RLixHooks(Protocol):
    """Protocol for RLix scheduling hooks injected into grpo_train().

    All methods receive the current global training step so the scheduler can
    correlate requests with the step they belong to.

    Task 5 hooks (GPU request/release):
      before_generation / after_generation   — inference GPU lifecycle
      before_training   / after_training     — training GPU lifecycle
      before_weight_sync                     — wake sleeping inference workers
                                               before refit (depends on Task 3)

    Task 6 hooks (progress reporting):
      begin_progress_batch  — record how many trajectories this step targets
      end_progress_batch    — accumulate collected count, emit at 2% granularity
    """

    def before_generation(self, step: int) -> None: ...
    def after_generation(self, step: int) -> None: ...
    def before_training(self, step: int) -> None: ...
    def after_training(self, step: int) -> None: ...
    def before_weight_sync(self, step: int) -> None: ...
    def begin_progress_batch(self, step: int, count_intended: int) -> None: ...
    def end_progress_batch(self, step: int, trajectories_collected: int) -> None: ...


class NoOpRLixHooks:
    """No-op hook implementation — used when RLix scheduler is not enabled.

    Satisfies the RLixHooks protocol so grpo_train() callers need not
    branch on whether hooks is None.
    """

    def before_generation(self, step: int) -> None:
        pass

    def after_generation(self, step: int) -> None:
        pass

    def before_training(self, step: int) -> None:
        pass

    def after_training(self, step: int) -> None:
        pass

    def before_weight_sync(self, step: int) -> None:
        pass

    def begin_progress_batch(self, step: int, count_intended: int) -> None:
        pass

    def end_progress_batch(self, step: int, trajectories_collected: int) -> None:
        pass


class NemoRLRLixHooks:
    """RLix scheduler integration hooks for NeMo RL GRPO.

    Wires grpo_train() into the RLix scheduler for multi-pipeline GPU
    time-sharing.  GPU request/release calls (before/after_generation,
    before/after_training, before_weight_sync) are placeholders pending
    Task 7 (pipeline actor) and Task 3 (NCCL destroy/reload).

    Progress reporting (Task 6) is fully implemented: begin_progress_batch /
    end_progress_batch maintain a cumulative counter and emit a ProgressReport
    to the scheduler at 2% granularity (at most 50 RPCs per step).
    """

    # Emit once every 2% of intended trajectories (50 buckets across 0-100%).
    _BUCKET_COUNT: int = 50

    def __init__(
        self,
        *,
        scheduler,  # Ray actor handle for rlix:scheduler
        pipeline_id: str,
        cluster_ids: dict[str, str],
    ) -> None:
        """
        Args:
            scheduler: Ray actor handle for rlix:scheduler.
            pipeline_id: RLix pipeline ID (e.g. "ft_000000000000").
            cluster_ids: Mapping of cluster name → cluster_id string,
                e.g. {"actor_train": "ft_xxx_actor_train",
                       "actor_infer": "ft_xxx_actor_infer"}.
        """
        self._scheduler = scheduler
        self._pipeline_id = pipeline_id
        self._cluster_ids = cluster_ids

        # Task 6: progress tracking state (reset per step in begin_progress_batch)
        self._count_intended_for_step: int = 0
        self._collected_so_far: int = 0
        self._last_emitted_bucket: int = -1
        self._current_step: int = -1

    # ------------------------------------------------------------------
    # Task 5: GPU request / release hooks
    # ------------------------------------------------------------------

    def before_generation(self, step: int) -> None:
        """Request inference GPU allocation from the RLix scheduler.

        Blocks until the scheduler grants the allocation (or times out).
        TODO(Task 7): replace pass with ray.get(scheduler.request_gpus.remote(...))
        """
        # from rlix.protocol.types import Priority
        # ray.get(self._scheduler.request_gpus.remote(
        #     cluster_id=self._cluster_ids["actor_infer"],
        #     priority=Priority.GENERATION,
        #     global_step=step,
        # ))
        pass

    def after_generation(self, step: int) -> None:
        """Notify scheduler that generation is done; triggers async shrink.

        Fire-and-forget — does not block the training loop.
        TODO(Task 7): replace pass with scheduler.notify_release_gpus.remote(...)
        """
        # self._scheduler.notify_release_gpus.remote(
        #     cluster_id=self._cluster_ids["actor_infer"],
        #     global_step=step,
        # )
        pass

    def before_training(self, step: int) -> None:
        """Request training GPU allocation.

        TODO(Task 7): replace pass with ray.get(scheduler.request_gpus.remote(...))
        """
        # from rlix.protocol.types import Priority
        # ray.get(self._scheduler.request_gpus.remote(
        #     cluster_id=self._cluster_ids["actor_train"],
        #     priority=Priority.ACTOR_TRAINING,
        #     global_step=step,
        # ))
        pass

    def after_training(self, step: int) -> None:
        """Notify scheduler that training is done.

        TODO(Task 7): replace pass with scheduler.notify_release_gpus.remote(...)
        """
        # self._scheduler.notify_release_gpus.remote(
        #     cluster_id=self._cluster_ids["actor_train"],
        #     global_step=step,
        # )
        pass

    def before_weight_sync(self, step: int) -> None:
        """Wake sleeping inference workers before refit.

        Any inference DP ranks that were sleeping (released during training)
        must be expanded and their NCCL communicators rebuilt before
        refit_policy_generation() can broadcast updated weights to them.

        TODO(Task 3): destroy_megatron_nccl_communicators() before sleep,
                      rebuild them here after wake.
        TODO(Task 7): call coordinator.resize_infer() to expand sleeping ranks.
        """
        pass

    # ------------------------------------------------------------------
    # Task 6: progress reporting
    # ------------------------------------------------------------------

    def begin_progress_batch(self, step: int, count_intended: int) -> None:
        """Start progress tracking for a generation step.

        Must be called once before the first end_progress_batch for that step.
        Resets accumulated counter and bucket state.

        Args:
            step: Current global training step.
            count_intended: Total number of trajectories grpo_train() will
                collect during this step's generation phase. Must be > 0.
        """
        if count_intended <= 0:
            raise ValueError(f"count_intended must be > 0, got {count_intended!r}")
        self._current_step = step
        self._count_intended_for_step = count_intended
        self._collected_so_far = 0
        self._last_emitted_bucket = -1

    def end_progress_batch(self, step: int, trajectories_collected: int) -> None:
        """Accumulate collected trajectories and emit a progress report if the bucket advances.

        Designed to be called each time a mini-batch of trajectories is produced
        inside the generation loop.  Emits to the scheduler at most once per 2%
        of the intended count (50 buckets total) so the scheduler is not flooded.
        Emission is fire-and-forget and does not block the caller.

        Args:
            step: Current global training step. Must match the step passed to
                the preceding begin_progress_batch call.
            trajectories_collected: Number of trajectories produced in this batch.
                Must be >= 0.

        Raises:
            RuntimeError: If called without a preceding begin_progress_batch.
            ValueError: If step does not match the current step, or if
                trajectories_collected is negative.
        """
        if self._current_step == -1:
            raise RuntimeError(
                "end_progress_batch called before begin_progress_batch"
            )
        if step != self._current_step:
            raise ValueError(
                f"end_progress_batch step mismatch: expected {self._current_step}, got {step}"
            )
        if trajectories_collected < 0:
            raise ValueError(
                f"trajectories_collected must be >= 0, got {trajectories_collected!r}"
            )

        self._collected_so_far += trajectories_collected
        bucket = min(
            int(self._collected_so_far / self._count_intended_for_step * self._BUCKET_COUNT),
            self._BUCKET_COUNT,
        )

        if bucket != self._last_emitted_bucket:
            self._last_emitted_bucket = bucket
            self._emit_progress(step)

    def _emit_progress(self, step: int) -> None:
        """Fire-and-forget ProgressReport to the RLix scheduler.

        Separated into its own method so tests can patch or override it without
        touching the bucket logic.

        TODO(Task 7): uncomment once scheduler actor is wired up.
        """
        # import time
        # from rlix.protocol.types import ProgressReport
        # self._scheduler.report_progress.remote(
        #     ProgressReport(
        #         pipeline_id=self._pipeline_id,
        #         step_target_trajectories=self._count_intended_for_step,
        #         fifo_timestamp=time.monotonic(),
        #         metrics={
        #             "completed": float(self._collected_so_far),
        #             "mode": "train",
        #         },
        #     )
        # )
        pass
