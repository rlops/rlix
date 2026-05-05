"""RLix-side implementation of the MILES :class:`RLixHooks` protocol.

MILES code (``examples/fully_async/fully_async_rollout.py``) holds an
``RLixHooks`` reference whose concrete type is opaque. The RLix entry
driver injects an instance of :class:`MilesRLixHooks` whose methods
turn the plain-kwarg progress events into RLix-coordinator RPCs.

Wire contract: this hook talks to ``Coordinator.report_progress_from_scheduler``
which expects the canonical ``rlix.protocol.types.ProgressReport`` *positional*
argument, NOT a ``metrics=`` kwarg. ``end_progress_batch`` is published via
``coordinator.clear_progress_stream(mode=..., adapter_id=...)`` (the canonical
stream-retire RPC), not as another progress report event.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rlix.protocol.types import ProgressReport

logger = logging.getLogger(__name__)

# Default mode label used when the MILES caller doesn't pass ``mode=``.
# Coordinator-side stream identity is (mode, adapter_id); both sides must
# agree on the default so the matching ``clear_progress_stream`` call
# retires the same stream.
_DEFAULT_MODE = "actor_train"


class MilesRLixHooks:
    """RLix-side implementation of the MILES :class:`RLixHooks` protocol.

    Constructed by ``examples/rlix/run_miles_rlix.py`` with the
    coordinator handle. Each protocol call publishes a canonical
    :class:`ProgressReport` (or ``clear_progress_stream`` for the
    end-of-batch event); the resulting Ray ObjectRef is discarded
    fire-and-forget so rollout-side correctness does not depend on
    coordinator reachability.
    """

    def __init__(self, coordinator_handle, *, pipeline_id: str | None = None):
        self._coordinator = coordinator_handle
        self._pipeline_id = str(pipeline_id) if pipeline_id is not None else "miles_default"
        # Track the (mode, adapter_id, target_weight_version) of the
        # currently-open batch so end_progress_batch can pick the
        # right clear_progress_stream call.
        self._open_mode: str | None = None
        self._open_adapter_id: str | None = None
        self._open_target_weight_version: int | None = None
        self._step_target_groups: int = 0
        self._collected: int = 0

    # -- RLixHooks protocol --------------------------------------------

    def begin_progress_batch(
        self,
        target_weight_version: int,
        step_target_groups: int,
        initial_completed: int,
        *,
        mode: str | None = None,
        adapter_id: str | None = None,
    ) -> None:
        # Coordinator stream identity = (mode, adapter_id). Default mode
        # avoids publishing reports under None / mismatched keys.
        self._open_mode = mode if mode is not None else _DEFAULT_MODE
        self._open_adapter_id = adapter_id
        self._open_target_weight_version = int(target_weight_version)
        self._step_target_groups = int(step_target_groups)
        self._collected = int(initial_completed)
        self._publish_report(new_batch=True)

    def bump_completed(self, *, target_weight_version: int) -> None:
        self._collected += 1
        # The active engine version may have advanced; thread it into
        # the metrics envelope so the coordinator can correlate
        # collected counts with the version they were rolled out
        # against.
        self._open_target_weight_version = int(target_weight_version)
        self._publish_report(new_batch=False)

    def end_progress_batch(self) -> None:
        # Retire the stream via the canonical RPC. Iter 21's
        # MilesCoordinator follows the abstract Coordinator contract,
        # so this matches whether or not MilesCoordinator subclasses
        # PipelineCoordinator.
        mode = self._open_mode if self._open_mode is not None else _DEFAULT_MODE
        adapter_id = self._open_adapter_id
        self._open_mode = None
        self._open_adapter_id = None
        self._open_target_weight_version = None
        self._step_target_groups = 0
        self._collected = 0
        self._fire_clear(mode=mode, adapter_id=adapter_id)

    def report_preempt(self, engine_index: int, worker_url: str) -> None:
        # Preempt notifications ride the same progress channel for
        # observability; the coordinator may demote them into metrics
        # if it doesn't act on them directly. Field shape uses metrics
        # so the canonical ProgressReport contract is preserved.
        if self._open_mode is None:
            # Preempt before begin_progress_batch — log only.
            logger.debug(
                "MilesRLixHooks: report_preempt before begin_progress_batch; "
                "engine_index=%s worker_url=%s",
                engine_index,
                worker_url,
            )
            return
        report = ProgressReport(
            pipeline_id=self._pipeline_id,
            step_target_trajectories=int(self._step_target_groups),
            fifo_timestamp=time.time(),
            metrics={
                "kind": "report_preempt",
                "mode": self._open_mode,
                "adapter_id": self._open_adapter_id,
                "engine_index": int(engine_index),
                "worker_url": str(worker_url),
            },
        )
        self._fire_report(report)

    # -- internal ------------------------------------------------------

    def _publish_report(self, *, new_batch: bool) -> None:
        """Build a ProgressReport from the current batch state and publish."""
        if self._open_mode is None:
            return
        report = ProgressReport(
            pipeline_id=self._pipeline_id,
            step_target_trajectories=int(self._step_target_groups),
            fifo_timestamp=time.time(),
            metrics={
                "kind": "begin_progress_batch" if new_batch else "bump_completed",
                "mode": self._open_mode,
                "adapter_id": self._open_adapter_id,
                "target_weight_version": int(self._open_target_weight_version or 0),
                "collected": int(self._collected),
                "new_batch": bool(new_batch),
            },
        )
        self._fire_report(report)

    def _fire_report(self, report: ProgressReport) -> None:
        try:
            handle = getattr(self._coordinator, "report_progress_from_scheduler", None)
            if handle is None:
                logger.debug(
                    "MilesRLixHooks: coordinator has no "
                    "report_progress_from_scheduler; skipping"
                )
                return
            remote = getattr(handle, "remote", None)
            if remote is not None:
                remote(report)
            else:
                handle(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MilesRLixHooks: report_progress_from_scheduler failed: %r", exc
            )

    def _fire_clear(self, *, mode: str, adapter_id: str | None) -> None:
        try:
            handle = getattr(self._coordinator, "clear_progress_stream", None)
            if handle is None:
                return
            remote = getattr(handle, "remote", None)
            if remote is not None:
                remote(mode=mode, adapter_id=adapter_id)
            else:
                handle(mode=mode, adapter_id=adapter_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MilesRLixHooks: clear_progress_stream failed: %r", exc
            )

    def clear_progress(self) -> None:
        """Tear down any per-batch coordinator-side state on shutdown.

        Public alias for :meth:`end_progress_batch` when the caller
        wants to retire the stream without the matching begin/bump
        bookkeeping.
        """
        if self._open_mode is None:
            return
        self.end_progress_batch()


__all__ = ["MilesRLixHooks"]
