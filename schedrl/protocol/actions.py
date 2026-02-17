from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class RegisterPipelineAction:
    pipeline_id: str


@dataclass(frozen=True, slots=True)
class AdmitPipelineAction:
    pipeline_id: str


@dataclass(frozen=True, slots=True)
class RequestGpusAction:
    pipeline_id: str
    requested_gpu_ids: List[int]
    timeout_s: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ReleaseGpusAction:
    pipeline_id: str
    released_gpu_ids: List[int]


@dataclass(frozen=True, slots=True)
class ReleaseAndRequestAction:
    pipeline_id: str
    released_gpu_ids: List[int]
    requested_gpu_ids: List[int]
    timeout_s: Optional[float] = None


@dataclass(frozen=True, slots=True)
class NotifyReadyToReleaseAction:
    pipeline_id: str
    timeout_s: Optional[float] = None
