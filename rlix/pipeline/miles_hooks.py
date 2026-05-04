"""RLix-side implementation of the MILES :class:`RLixHooks` protocol.

MILES code (``examples/fully_async/fully_async_rollout.py``) holds an
``RLixHooks`` reference whose concrete type is opaque. The RLix entry
driver injects an instance of :class:`MilesRLixHooks` whose methods
turn the plain-kwarg progress events into RLix-coordinator RPCs.

Design notes
------------
- The RLix side imports the protocol from ``miles.utils.rlix_hooks``
  (the seam is unidirectional: MILES does not import from rlix.*).
- Coordinator RPC calls are fire-and-forget: the hook does not block
  the rollout-worker thread on coordinator state. Failures are logged
  and swallowed (rollout-side correctness must not depend on the
  coordinator being reachable).
- The :class:`ProgressReport` envelope is constructed here so RLix
  protocol changes (new metric keys, mode/adapter_id additions) stay
  inside the RLix package without breaking MILES callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressReport:
    """Envelope for one progress event published from MILES → coordinator.

    The wire shape is intentionally a single ``metrics: dict[str, Any]``
    so the coordinator side (``MilesCoordinator.report_progress_from_scheduler``)
    can evolve the schema without breaking MILES-side callers.
    """

    metrics: dict[str, Any] = field(default_factory=dict)


class MilesRLixHooks:
    """RLix-side implementation of MILES's :class:`RLixHooks` protocol.

    Constructed by ``examples/rlix/run_miles_rlix.py`` with the
    coordinator handle. The protocol methods return ``None``; this
    implementation translates each call into a ``coordinator.<method>.remote(...)``
    RPC and discards the resulting ObjectRef (fire-and-forget).
    """

    def __init__(self, coordinator_handle, *, pipeline_id: str | None = None):
        self._coordinator = coordinator_handle
        self._pipeline_id = pipeline_id

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
        report = ProgressReport(
            metrics={
                "kind": "begin_progress_batch",
                "pipeline_id": self._pipeline_id,
                "target_weight_version": int(target_weight_version),
                "step_target_groups": int(step_target_groups),
                "initial_completed": int(initial_completed),
                "mode": mode,
                "adapter_id": adapter_id,
            }
        )
        self._fire_and_forget(report)

    def bump_completed(self, *, target_weight_version: int) -> None:
        report = ProgressReport(
            metrics={
                "kind": "bump_completed",
                "pipeline_id": self._pipeline_id,
                "target_weight_version": int(target_weight_version),
                "collected": 1,
            }
        )
        self._fire_and_forget(report)

    def end_progress_batch(self) -> None:
        report = ProgressReport(
            metrics={
                "kind": "end_progress_batch",
                "pipeline_id": self._pipeline_id,
            }
        )
        self._fire_and_forget(report)

    def report_preempt(self, engine_index: int, worker_url: str) -> None:
        report = ProgressReport(
            metrics={
                "kind": "report_preempt",
                "pipeline_id": self._pipeline_id,
                "engine_index": int(engine_index),
                "worker_url": str(worker_url),
            }
        )
        self._fire_and_forget(report)

    # -- internal ------------------------------------------------------

    def _fire_and_forget(self, report: ProgressReport) -> None:
        """Drive ``coordinator.report_progress_from_scheduler.remote(metrics=...)``.

        Failures are logged at ``warning`` and swallowed. We intentionally
        do not block the rollout thread on coordinator state.
        """
        try:
            # Ray-actor handles expose ``.remote`` for each method; the
            # explicit ``.remote`` invocation lets us forward the
            # ObjectRef to a future without awaiting it.
            handle = getattr(self._coordinator, "report_progress_from_scheduler", None)
            if handle is None:
                logger.debug(
                    "MilesRLixHooks: coordinator handle has no "
                    "report_progress_from_scheduler attribute; skipping"
                )
                return
            remote = getattr(handle, "remote", None)
            if remote is not None:
                remote(metrics=report.metrics)
            else:
                # Test fixtures may pass a plain (non-actor) coordinator;
                # still call the method directly for forward-compat.
                handle(metrics=report.metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MilesRLixHooks: report_progress_from_scheduler failed: %r", exc
            )

    def clear_progress(self) -> None:
        """Tear down any per-batch coordinator-side state on shutdown.

        Mirrors ``coordinator.clear_progress_stream`` (iter 21).
        """
        try:
            handle = getattr(self._coordinator, "clear_progress_stream", None)
            if handle is None:
                return
            remote = getattr(handle, "remote", None)
            if remote is not None:
                remote()
            else:
                handle()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MilesRLixHooks: clear_progress_stream failed: %r", exc)


__all__ = ["ProgressReport", "MilesRLixHooks"]
