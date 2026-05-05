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

import asyncio
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
        *,
        broadcast_local_ranks: set[int] | frozenset[int] | None = None,
    ) -> int:
        """F15 / F20 atomic sync unit — sender + finalize + publish.

        Single ``asyncio.wait_for(self._timeout_s)`` covers:
          (a) ``cache_owner_actor.run_sync_session(plan)`` — single composite RPC
              that drives every per-bucket transport (cpu_serialize tmpfs +
              NCCL broadcast, classified via the ``broadcast_local_ranks``
              kwarg below).
          (b) ``finalize_weight_update`` fan-out across receivers.
          (c) ``manager.set_weight_version(version, engine_indices=target)``
              single publish per sync (F21).

        ``version == -1`` is the F40 base path: the cache_owner has already
        built buckets at init Step 4; the transport flow is identical (no
        special-case branch).

        ``broadcast_local_ranks`` is the subset of ``target_engine_indices``
        whose engines need the NCCL broadcast path (non-colocate engines
        whose GPU is outside the train pool). M11.1 RLix mode default is
        all-cpu_serialize, so the kwarg defaults to ``None`` (empty set).

        Returns the published version (echoed for caller logging).
        """
        import asyncio

        sync_id = sync_id or f"miles-sync-{uuid.uuid4().hex}"
        target = frozenset(int(i) for i in target_engine_indices)
        if not target:
            logger.info(
                "[MilesModelUpdateService] sync_selected_workers no-op "
                "pipeline_id=%s sync_id=%s (empty target set)",
                self._pipeline_id,
                sync_id,
            )
            return int(version)

        broadcast_set = frozenset(int(i) for i in (broadcast_local_ranks or ()))
        if not broadcast_set.issubset(target):
            raise ValueError(
                f"broadcast_local_ranks={sorted(broadcast_set)} must be a subset "
                f"of target_engine_indices={sorted(target)}"
            )
        if broadcast_set:
            # The cache-owner sender NCCL path (init_process_group +
            # per-bucket dist.broadcast + dist.destroy_process_group)
            # is not yet implemented inside MILES run_sync_session
            # (iter 12 wires only the receiver-side fan-out). Fail
            # fast rather than hang the receivers waiting for an
            # absent sender.
            raise NotImplementedError(
                "broadcast_local_ranks transport requires sender-side NCCL "
                "(init_process_group + dist.broadcast on the cache_owner). "
                "MILES iter 12 only wired the receiver-side fan-out. "
                "Until the sender-side path lands, route every target "
                "through cpu_serialize."
            )
        cpu_serialize_set = target - broadcast_set

        # R06-F1 fix: track every Ray ObjectRef issued by this atomic
        # unit so an asyncio.wait_for timeout (or any other cancellation)
        # can fan out ray.cancel(force=True) on them. Without this, the
        # local coroutine cancels but the Ray actor methods on
        # cache_owner / rollout_manager keep running and continue to
        # hold cache_owner._cache_lock — the next sync_selected_workers
        # call would queue behind the still-executing prior method.
        inflight_refs: list = []

        async def _run() -> int:
            return await self._run_atomic_unit(
                sync_id=sync_id,
                target=target,
                version=int(version),
                cpu_serialize_set=cpu_serialize_set,
                broadcast_set=broadcast_set,
                inflight_refs=inflight_refs,
            )

        try:
            if self._timeout_s is None or self._timeout_s <= 0:
                return await _run()
            return await asyncio.wait_for(_run(), timeout=float(self._timeout_s))
        except asyncio.TimeoutError:
            self._cancel_inflight(inflight_refs, reason="wait_for timeout")
            raise
        except asyncio.CancelledError:
            # Outer cancellation (caller cancelled sync_selected_workers
            # task) — propagate after firing ray.cancel so we don't leak
            # inflight Ray work either.
            self._cancel_inflight(inflight_refs, reason="task cancelled")
            raise

    def _cancel_inflight(self, inflight_refs: list, *, reason: str) -> None:
        if not inflight_refs:
            return
        logger.warning(
            "[MilesModelUpdateService] cancelling %d inflight ObjectRefs "
            "pipeline_id=%s reason=%s",
            len(inflight_refs),
            self._pipeline_id,
            reason,
        )
        for ref in inflight_refs:
            try:
                ray.cancel(ref, force=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ray.cancel failed pipeline_id=%s reason=%s exc=%r",
                    self._pipeline_id,
                    reason,
                    exc,
                )
        inflight_refs.clear()

    async def _run_atomic_unit(
        self,
        *,
        sync_id: str,
        target: frozenset[int],
        version: int,
        cpu_serialize_set: frozenset[int],
        broadcast_set: frozenset[int],
        inflight_refs: list,
    ) -> int:
        # Each .remote() is captured into inflight_refs BEFORE we await
        # so the outer cancellation handler can fire ray.cancel(force=True)
        # on every dispatched ref (R06-F1).
        # (1) Read-only entry: fetch per-engine handles from manager.
        handles_ref = self._rollout_manager.get_engine_handles.remote(sorted(target))
        inflight_refs.append(handles_ref)
        handles: dict[int, Any] = await _ray_get(handles_ref)
        if set(handles.keys()) != set(target):
            raise RuntimeError(
                f"manager returned handles {sorted(handles.keys())}; "
                f"requested {sorted(target)} (lossy snapshot)"
            )

        # (2) Build the wire-format plan dict.
        plan = await self._build_plan(
            sync_id=sync_id,
            version=version,
            target_handles=handles,
            cpu_serialize_set=cpu_serialize_set,
            broadcast_set=broadcast_set,
            inflight_refs=inflight_refs,
        )

        # (3) Single composite RPC into the cache_owner. F04 invariant
        #     means run_sync_session is the ONLY top-level Ray method
        #     used for transport.
        sync_ref = self._cache_owner_actor.run_sync_session.remote(plan)
        inflight_refs.append(sync_ref)
        await _ray_get(sync_ref)

        # (4) Finalize fan-out. F21: per-bucket payload had no version;
        #     finalize_weight_update on every receiver flushes pending
        #     work so update_weight_version below is observed at the
        #     next prefill.
        finalize_refs = [h.finalize_weight_update.remote() for h in handles.values()]
        inflight_refs.extend(finalize_refs)
        await _ray_get(finalize_refs)

        # (5) Single version publish (F21). Pipeline / coordinator MUST
        #     NOT call this directly — only the service does.
        publish_ref = self._rollout_manager.set_weight_version.remote(
            int(version), engine_indices=sorted(target)
        )
        inflight_refs.append(publish_ref)
        await _ray_get(publish_ref)
        logger.info(
            "[MilesModelUpdateService] sync_selected_workers_done pipeline_id=%s "
            "sync_id=%s version=%s targets=%s cpu_serialize=%s broadcast=%s",
            self._pipeline_id,
            sync_id,
            version,
            sorted(target),
            sorted(cpu_serialize_set),
            sorted(broadcast_set),
        )
        return int(version)

    async def _build_plan(
        self,
        *,
        sync_id: str,
        version: int,
        target_handles: dict[int, Any],
        cpu_serialize_set: frozenset[int],
        broadcast_set: frozenset[int],
        inflight_refs: list,
    ) -> dict[str, Any]:
        # F26 / C16: master_port != 0. Claim a free port from the
        # cache_owner actor (not 0; not picked by the receiver) and
        # publish it via the SharedStorage singleton so concurrent
        # service instances can avoid collisions.
        master_addr_ref = self._cache_owner_actor.get_node_ip.remote()
        master_port_ref = self._cache_owner_actor.get_free_port.remote()
        inflight_refs.append(master_addr_ref)
        inflight_refs.append(master_port_ref)
        master_addr_raw, master_port_raw = await _ray_get(
            [master_addr_ref, master_port_ref]
        )
        master_addr = str(master_addr_raw)
        master_port = int(master_port_raw)
        if master_port <= 0:
            raise RuntimeError(
                f"cache_owner.get_free_port returned non-positive {master_port}"
            )
        # SharedStorage atomic port claim — mirrors the pattern in
        # rlix.pipeline.model_update_service. ``try_put`` returns
        # False if the key already exists, which we treat as a
        # collision and re-pick. Bound the retry budget so a stuck
        # claim can't hang the sync.
        from roll.distributed.scheduler.storage import (  # type: ignore[import-not-found]
            STORAGE_NAME,
        )
        from roll.utils.constants import GLOBAL_STORAGE_NAMESPACE  # type: ignore[import-not-found]

        try:
            shared_storage = ray.get_actor(STORAGE_NAME, namespace=GLOBAL_STORAGE_NAMESPACE)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MilesModelUpdateService: SharedStorage actor unavailable; "
                "skipping port claim: %r",
                exc,
            )
            shared_storage = None

        if shared_storage is not None:
            for attempt in range(8):
                claim_key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"
                try_put_ref = shared_storage.try_put.remote(claim_key, self._pipeline_id)
                inflight_refs.append(try_put_ref)
                claimed = bool(await _ray_get(try_put_ref))
                if claimed:
                    break
                # Collision — pick another free port and re-try.
                next_port_ref = self._cache_owner_actor.get_free_port.remote()
                inflight_refs.append(next_port_ref)
                next_port = int(await _ray_get(next_port_ref))
                # Always advance master_port to the new pick to avoid
                # burning a budget slot on the same key (R06-F4).
                if next_port == master_port:
                    # get_free_port returned the same value; tiny
                    # back-off + retry on a fresh try_put round.
                    await asyncio.sleep(0.05)
                master_port = next_port
            else:
                raise RuntimeError(
                    "SharedStorage port-claim retry budget exhausted "
                    f"(pipeline_id={self._pipeline_id} addr={master_addr} "
                    f"last_port={master_port} attempts=8); release stale "
                    "claims before retrying."
                )

        # Per-engine NCCL rank within the dynamic broadcast group.
        # cache_owner is rank 0; receivers are 1..N.
        comm_ranks: dict[int, int] = {}
        for r, idx in enumerate(sorted(broadcast_set), start=1):
            comm_ranks[idx] = r
        # cpu_serialize engines are not in the broadcast group; they
        # never enter setup_collective_group. Comm-rank assignment is
        # ignored for them but kept in the dict for plan-shape
        # uniformity (run_sync_session reads it conditionally).
        for idx in cpu_serialize_set:
            comm_ranks.setdefault(idx, 0)

        world_size = 1 + len(broadcast_set) if broadcast_set else 1

        plan = SyncSessionPlan(
            sync_id=sync_id,
            version=int(version),
            group_name=f"miles_{self._pipeline_id}_{sync_id}",
            master_addr=master_addr,
            master_port=master_port,
            timeout_s=float(self._timeout_s or 0.0),
            target_handles=dict(target_handles),
            cpu_serialize_local_ranks=cpu_serialize_set,
            broadcast_local_ranks=broadcast_set,
            comm_ranks=comm_ranks,
            world_size=int(world_size),
        )
        return plan.as_wire_dict()


async def _ray_get(refs):
    """``await``able wrapper around ``ray.get`` so the timeout-bounded
    ``asyncio.wait_for`` in :meth:`sync_selected_workers` can cancel
    the surrounding task.

    Ray's ObjectRefs are already awaitable in newer versions, so we
    just delegate.
    """
    if isinstance(refs, list):
        return await asyncio.gather(*[_await_ref(r) for r in refs])
    return await _await_ref(refs)


async def _await_ref(ref):
    import asyncio  # noqa: F401  -- module-level import handled below

    return await ref


__all__ = ["SyncSessionPlan", "MilesModelUpdateService"]
