from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class PipelineRuntimeState:
    pipeline_id: str
    registered: bool = False
    admitted: bool = False
    busy: bool = False
    active_action: Optional[str] = None
    last_progress_step_target: Optional[int] = None


@dataclass(slots=True)
class SchedulerState:
    pipelines: Dict[str, PipelineRuntimeState] = field(default_factory=dict)

    def get_or_create_pipeline(self, pipeline_id: str) -> PipelineRuntimeState:
        state = self.pipelines.get(pipeline_id)
        if state is None:
            state = PipelineRuntimeState(pipeline_id=pipeline_id)
            self.pipelines[pipeline_id] = state
        return state
