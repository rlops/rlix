from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from rlix.protocol.types import ActionResponse, ProgressReport


class Coordinator(ABC):
    @abstractmethod
    def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        raise NotImplementedError

    @abstractmethod
    def create_pipeline_actor(self, *, pipeline_config: Any) -> Any:
        """Create the pipeline actor for training/inference.

        Args:
            pipeline_config: Pipeline configuration object.

        Returns:
            Ray actor reference to the pipeline actor.
        """
        raise NotImplementedError

    @abstractmethod
    def report_progress_from_scheduler(self, report: ProgressReport) -> None:
        """Receive a per-scheduler progress snapshot and forward aggregated progress to the rlix scheduler."""
        raise NotImplementedError

    @abstractmethod
    def sync_lora_weights(self, *, loras_to_sync: List[str]) -> None:
        """Push trained LoRA weights to infer workers.

        Args:
            loras_to_sync: List of LoRA names to sync.
        """
        raise NotImplementedError