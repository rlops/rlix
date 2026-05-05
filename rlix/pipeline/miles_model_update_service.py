"""MILES-specific selective weight sync service.

Mirrors :class:`rlix.pipeline.model_update_service.ModelUpdateService` but
drives the MILES F4 / F5+6 atomic-unit contract: a single
``cache_owner_actor.run_sync_session(plan)`` Ray RPC carries every
per-bucket transport (cpu_serialize tmpfs + dynamic NCCL broadcast),
followed by a finalize fan-out and a single ``manager.set_weight_version``
publish, all under one ``asyncio.wait_for`` timeout.

Iter 18 lands the constructor + :class:`SyncSessionPlan` frozen dataclass +
:meth:`sync_selected_workers` signature stub. Iter 19 wires the cpu_serialize
+ NCCL broadcast transport phase. Iter 20 wires the finalize fan-out +
version publish + atomic timeout.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import Any

import ray

from rlix.utils.env import parse_env_timeout_s

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SyncSessionPlan:
    """Frozen RLix-side construction shape; serializes to a plain dict before
    crossing the Ray boundary into MILES.

    Per the cozy-plan §Shared protocol contract, MILES
    ``run_sync_session`` reads this as a mapping (no MILES → RLix
    import dependency). RLix builds the dataclass for validation
    + clarity, then ``dataclasses.asdict`` before
    ``cache_owner_actor.run_sync_session.remote(plan_dict)``.

    Required fields mirror MILES's
    :meth:`MegatronTrainRayActor.run_sync_session` plan-key validator
    so any drift fails fast on either side.
    """

    sync_id: str
    version: int
    group_name: str
    master_addr: str
    master_port: int
    timeout_s: float
    target_handles: dict[int, Any]  # engine_index → SGLangEngineActor handle
    cpu_serialize_local_ranks: frozenset[int]
    broadcast_local_ranks: frozenset[int]
    comm_ranks: dict[int, int]  # engine_index → NCCL rank within the group
    world_size: int

    def as_wire_dict(self) -> dict[str, Any]:
        """Convert to plain dict for the Ray boundary.

        Preserves ``target_handles`` as a dict (Ray pickles actor
        handles fine across actors); ``frozenset`` -> ``list`` so
        the receiving side does not depend on a frozenset import.
        """
        return {
            "sync_id": self.sync_id,
            "version": int(self.version),
            "group_name": self.group_name,
            "master_addr": self.master_addr,
            "master_port": int(self.master_port),
            "timeout_s": float(self.timeout_s),
            "target_handles": dict(self.target_handles),
            "cpu_serialize_local_ranks": list(self.cpu_serialize_local_ranks),
            "broadcast_local_ranks": list(self.broadcast_local_ranks),
            "comm_ranks": dict(self.comm_ranks),
            "world_size": int(self.world_size),
        }


@ray.remote
class MilesModelUpdateService:
    """Per-pipeline MILES selective sync service.

    Constructor (per scope F107 / X2 ctor handle injection):

    - ``pipeline_id``: unique str per pipeline.
    - ``cache_owner_actor``: handle to the MILES ``MegatronTrainRayActor``
      that has been classified as the F18 cache_owner. Service drives
      the single composite ``run_sync_session`` RPC against this handle.
    - ``rollout_manager``: handle to the MILES ``RolloutManager`` Ray
      actor. Service calls only its read-only entry
      :meth:`get_engine_handles` at sync entry, and the single write
      entry :meth:`set_weight_version` at sync exit.

    The service NEVER calls ``ray.get_actor`` at runtime to look up
    these handles; the driver / coordinator passes them in (Layer 1
    forbidden in the unified plan).
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        cache_owner_actor,
        rollout_manager,
    ):
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise ValueError("pipeline_id must be a non-empty str")
        if cache_owner_actor is None:
            raise ValueError("cache_owner_actor handle is required (F107 / X2 injection)")
        if rollout_manager is None:
            raise ValueError("rollout_manager handle is required")
        self._pipeline_id = pipeline_id
        self._cache_owner_actor = cache_owner_actor
        self._rollout_manager = rollout_manager
        # ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S env override (default
        # 150s mirrors the existing rlix.pipeline.model_update_service
        # default). Iter 20 enforces this as a single
        # asyncio.wait_for around the whole atomic unit.
        self._timeout_s: float | None = parse_env_timeout_s(
            "ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S", 150.0
        )

    def get_pipeline_id(self) -> str:
        return self._pipeline_id

    async def sync_selected_workers(
        self,
        sync_id: str | None,
        target_engine_indices: set[int] | frozenset[int],
        version: int,
    ) -> int:
        """F15 / F20 atomic sync unit — sender + finalize + publish.

        Iter 18 establishes the signature only; iter 19 wires the
        transport phase (cpu_serialize per-engine + NCCL broadcast),
        iter 20 wires the finalize fan-out + ``manager.set_weight_version``
        + single ``asyncio.wait_for(self._timeout_s)``.

        Returns the published version (echoed for caller logging).
        """
        sync_id = sync_id or f"miles-sync-{uuid.uuid4().hex}"
        if not target_engine_indices:
            logger.info(
                "[MilesModelUpdateService] sync_selected_workers no-op "
                "pipeline_id=%s sync_id=%s (empty target set)",
                self._pipeline_id,
                sync_id,
            )
            return int(version)
        logger.info(
            "[MilesModelUpdateService] sync_selected_workers_stub pipeline_id=%s "
            "sync_id=%s version=%s targets=%s",
            self._pipeline_id,
            sync_id,
            version,
            sorted(target_engine_indices),
        )
        # Iter 19/20 replaces this stub with the real transport +
        # finalize + publish + timeout flow.
        return int(version)


__all__ = ["SyncSessionPlan", "MilesModelUpdateService"]
