from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from schedrl.protocol.types import ActionResponse, ProgressReport, ReleaseAck


class Adapter(ABC):
    @abstractmethod
    def get_pipeline_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def report_progress(self, report: ProgressReport) -> None:
        raise NotImplementedError

    @abstractmethod
    def abort_requests(self, request_ids: List[str]) -> None:
        """Initiate abort of the given request_ids.

        Abort ACK semantics (ENG-123): an ACK means the targeted request IDs are no longer in-flight
        (removed from the running set). A request may also finish successfully during abort; that
        is tolerated as long as it is no longer in-flight.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_abort_ack(self, request_ids: List[str], timeout_s: float) -> None:
        """Block until abort ACK for request_ids or timeout.

        See `abort_requests` for the canonical ACK definition.
        """
        raise NotImplementedError

    @abstractmethod
    def release_gpus(self, gpu_ids: List[int]) -> ReleaseAck:
        raise NotImplementedError

    @abstractmethod
    def request_gpus(self, gpu_ids: List[int], timeout_s: Optional[float] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        raise NotImplementedError

    @abstractmethod
    def get_state_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError
