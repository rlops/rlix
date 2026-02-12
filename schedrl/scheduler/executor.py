from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Executor:
    scheduler: Any

    def execute(self, action: Any) -> Any:
        raise NotImplementedError("Phase 1 provides executor skeleton only")

