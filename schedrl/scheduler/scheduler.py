from __future__ import annotations

"""SchedRL Scheduler (Phase 1).

Operational policy (ENG-123): fail-fast only. No recovery or rehydration is provided; on any
scheduler restart, pipelines are expected to re-register and be re-admitted.
"""

from dataclasses import dataclass
from typing import Any, Optional

from schedrl.protocol.request_id import validate_pipeline_id
from schedrl.protocol.types import ProgressReport, ReleaseAck
from schedrl.protocol.validation import validate_optional_timeout_s
from schedrl.scheduler.state import SchedulerState


def _require_ray():
    try:
        import ray  # noqa: F401
    except Exception as e:
        raise RuntimeError("schedrl.scheduler requires ray") from e


@dataclass(slots=True)
class SchedulerImpl:
    def __post_init__(self):
        _require_ray()
        self._state = SchedulerState()

    def register_pipeline(self, *, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        pipe = self._state.get_or_create_pipeline(pipeline_id)
        pipe.registered = True

    def admit_pipeline(self, *, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        pipe = self._state.get_or_create_pipeline(pipeline_id)
        if not pipe.registered:
            raise RuntimeError(f"Pipeline {pipeline_id!r} must be registered before admission")
        pipe.admitted = True

    def unregister_pipeline(self, *, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        self._state.pipelines.pop(pipeline_id, None)

    def report_progress(self, report: ProgressReport) -> None:
        validate_pipeline_id(report.pipeline_id)
        pipe = self._state.get_or_create_pipeline(report.pipeline_id)
        if not pipe.admitted:
            raise RuntimeError(f"Pipeline {report.pipeline_id!r} is not admitted")
        pipe.last_progress_step_target = report.step_target_trajectories

    def request_gpus(self, *, pipeline_id: str, gpu_ids: list[int], timeout_s: Optional[float] = None) -> None:
        validate_pipeline_id(pipeline_id)
        validate_optional_timeout_s(timeout_s)
        pipe = self._state.get_or_create_pipeline(pipeline_id)
        if not pipe.admitted:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is not admitted")
        if pipe.busy:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is busy")
        raise NotImplementedError("Phase 1 provides scheduler API skeleton only")

    def release_gpus(self, *, pipeline_id: str, gpu_ids: list[int]) -> ReleaseAck:
        validate_pipeline_id(pipeline_id)
        pipe = self._state.get_or_create_pipeline(pipeline_id)
        if not pipe.admitted:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is not admitted")
        if pipe.busy:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is busy")
        raise NotImplementedError("Phase 1 provides scheduler API skeleton only")

    def release_and_request(
        self,
        *,
        pipeline_id: str,
        release_gpu_ids: list[int],
        request_gpu_ids: list[int],
        timeout_s: Optional[float] = None,
    ) -> ReleaseAck:
        validate_pipeline_id(pipeline_id)
        validate_optional_timeout_s(timeout_s)
        pipe = self._state.get_or_create_pipeline(pipeline_id)
        if not pipe.admitted:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is not admitted")
        if pipe.busy:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is busy")
        raise NotImplementedError("Phase 1 provides scheduler API skeleton only")

    def notify_ready_to_release(
        self,
        *,
        pipeline_id: str,
        planned_release_gpu_ids: list[int],
        timeout_s: Optional[float] = None,
    ) -> ReleaseAck:
        validate_pipeline_id(pipeline_id)
        validate_optional_timeout_s(timeout_s)
        pipe = self._state.get_or_create_pipeline(pipeline_id)
        if not pipe.admitted:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is not admitted")
        if pipe.busy:
            raise RuntimeError(f"Pipeline {pipeline_id!r} is busy")
        raise NotImplementedError("Phase 1 provides scheduler API skeleton only")

    def get_debug_state(self) -> Any:
        return self._state


def scheduler_actor_class():
    _require_ray()
    import ray

    return ray.remote(max_restarts=0, max_task_retries=0)(SchedulerImpl)
