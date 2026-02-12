from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SchedulerRunLoop:
    scheduler: Any

    def tick(self) -> None:
        raise NotImplementedError("Phase 1 provides run loop skeleton only")

