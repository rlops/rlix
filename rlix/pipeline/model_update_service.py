from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import ray

from roll.distributed.executor.cluster import Cluster
from roll.utils.logging import get_logger

logger = get_logger()


@ray.remote
class ModelUpdateService:
    """Per-pipeline service for selective sync on expand (ENG-123 Phase 4).

    Contract:
    - Scheduler-side trigger only: no promotion forwarding, no validation, no coalescing.
    - Calls into sender-side sync, which serializes via sender cache_lock.
    """

    def __init__(self, *, pipeline_id: str, src_cluster: Cluster, tgt_cluster: Cluster):
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        self.pipeline_id = pipeline_id
        self.src_cluster: Any = src_cluster
        self.tgt_cluster: Any = tgt_cluster

        self._sync_nonce = uuid.uuid4().hex[:8]
        self._timeout_s: Optional[float] = self._parse_timeout_s("ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S", default=150.0)
        self._pg_timeout_s: Optional[float] = self._parse_timeout_s("ROLL_SELECTIVE_MODEL_UPDATE_PG_TIMEOUT_S", default=120.0)

    @staticmethod
    def _parse_timeout_s(env_key: str, *, default: float) -> Optional[float]:
        raw = os.environ.get(env_key)
        if raw is None:
            return float(default)
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{env_key} must be a number, got: {raw!r}") from exc
        return None if value <= 0 else value

    @staticmethod
    def _ray_get_with_timeout(refs: Any, *, timeout_s: Optional[float], desc: str) -> Any:
        if timeout_s is None:
            return ray.get(refs)
        try:
            return ray.get(refs, timeout=float(timeout_s))
        except ray.exceptions.GetTimeoutError as exc:
            raise TimeoutError(f"{desc} timed out after {timeout_s}s") from exc

    def _select_global_sender_rank(self) -> int:
        """Return the single global cache owner: pp_rank==0, dp_rank==0, tp_rank==0, cp_rank==0."""
        for rank, info in enumerate(self.src_cluster.worker_rank_info):
            if (
                int(info.pp_rank) == 0
                and int(info.dp_rank) == 0
                and int(info.tp_rank) == 0
                and int(info.cp_rank) == 0
            ):
                return int(rank)
        raise RuntimeError(
            "No global cache owner found for selective sync "
            "(expected exactly one rank with pp_rank==0, dp_rank==0, tp_rank==0, cp_rank==0)"
        )

    def _build_comm_plan_for_sender(
        self,
        *,
        sync_id: str,
        src_rank: int,
        tgt_dp_ranks: List[int],
    ) -> Tuple[dict, str, List[int]]:
        """Build comm plan for the single global cache owner.

        Classifies each target worker's local ranks as IPC (same physical GPU as sender)
        or broadcast (different GPU, needs NCCL). Returns:
        - comm_plan: dict keyed by src_rank with all routing data the owner needs
        - group_name: NCCL group name for broadcast-path setup
        - tgt_ranks_in_group: sorted list of target dp_ranks that need broadcast setup
        """
        src_rank = int(src_rank)
        src_worker = self.src_cluster.rank2worker[src_rank]
        master_addr = ray.get(src_worker.get_node_ip.remote())
        master_port = int(ray.get(src_worker.get_free_port.remote()))

        src_devices = self.src_cluster.rank2devices.get(src_rank, [])
        if not src_devices:
            raise RuntimeError(f"Missing src devices for src_rank={src_rank}")
        src_gpu_keys: Set[Tuple[int, int]] = {
            (int(d["node_rank"]), int(d["gpu_rank"]))
            for d in src_devices
            if d.get("node_rank") is not None and d.get("gpu_rank") is not None
        }
        if not src_gpu_keys:
            raise RuntimeError(f"Missing src gpu keys for src_rank={src_rank}: {src_devices}")

        # Classify each device of each target worker as IPC or broadcast.
        tgt_devices: List[Dict[str, Any]] = []  # broadcast-only devices (for NCCL group setup)
        tgt_ranks_in_group: Set[int] = set()
        ipc_targets: List[Dict[str, Any]] = []  # [{dp_rank, local_ranks}]
        broadcast_local_ranks_by_dp_rank: Dict[int, List[int]] = {}

        for tgt_rank in tgt_dp_ranks:
            ipc_local_ranks: List[int] = []
            broadcast_local_ranks: List[int] = []
            for device in self.tgt_cluster.rank2devices[int(tgt_rank)]:
                tgt_gpu_key = (int(device["node_rank"]), int(device["gpu_rank"]))
                local_rank = int(device["rank"])
                if tgt_gpu_key in src_gpu_keys:
                    # Same physical GPU → CUDA IPC path; NCCL cannot form group with duplicate GPUs.
                    ipc_local_ranks.append(local_rank)
                else:
                    broadcast_local_ranks.append(local_rank)
                    tgt_devices.append({"rank": int(tgt_rank), "device": device})
                    tgt_ranks_in_group.add(int(tgt_rank))

            if ipc_local_ranks:
                ipc_targets.append({"dp_rank": int(tgt_rank), "local_ranks": sorted(ipc_local_ranks)})
            if broadcast_local_ranks:
                broadcast_local_ranks_by_dp_rank[int(tgt_rank)] = sorted(broadcast_local_ranks)

        safe_sync_id = str(sync_id).replace("/", "_")
        group_name = f"selective_model_update_{safe_sync_id}_src{src_rank}"

        comm_plan_args: Dict[str, Any] = dict(
            group_name=group_name,
            master_addr=master_addr,
            master_port=master_port,
            tgt_devices=tgt_devices,
            src_rank=src_rank,
            # Bookkeeping key for strategy._setup_collective_group_impl() dict indexing.
            # Selective sync uses a single global sender (all PP layers gathered), so always 0.
            src_pp_rank=0,
            # IPC routing: list of {dp_rank, local_ranks} for colocated workers.
            ipc_targets=ipc_targets,
            # Per-worker broadcast local rank masks so owner knows which ranks joined NCCL.
            broadcast_local_ranks_by_dp_rank=broadcast_local_ranks_by_dp_rank,
        )
        comm_plan = {src_rank: comm_plan_args}
        return comm_plan, group_name, sorted(tgt_ranks_in_group)

    def sync_selected_workers(self, tgt_dp_ranks: List[int], adapters_to_sync: list[str] | None = None) -> None:
        tgt_dp_ranks = sorted(set(int(r) for r in tgt_dp_ranks))
        if not tgt_dp_ranks:
            raise ValueError("tgt_dp_ranks must be non-empty")

        infer_world_size = int(self.tgt_cluster.world_size)
        invalid = [r for r in tgt_dp_ranks if r < 0 or r >= infer_world_size]
        if invalid:
            raise ValueError(f"Invalid tgt_dp_ranks={invalid}; infer_world_size={infer_world_size}")

        tgt_device_mapping = getattr(self.tgt_cluster.worker_config, "device_mapping", None)
        tgt_num_gpus_per_worker = getattr(self.tgt_cluster.worker_config, "num_gpus_per_worker", None)

        if not tgt_device_mapping:
            raise RuntimeError("tgt_cluster device_mapping is empty; selective sync requires GPU infer workers")

        if not isinstance(tgt_num_gpus_per_worker, int) or int(tgt_num_gpus_per_worker) <= 0:
            raise RuntimeError("tgt_cluster.worker_config.num_gpus_per_worker must be positive int")

        tgt_device_mapping = [int(x) for x in tgt_device_mapping]

        sync_id = f"selective_sync/{self.pipeline_id}/{self._sync_nonce}/{uuid.uuid4().hex[:8]}"
        logger.info(
            f"[ModelUpdateService] sync_selected_workers_enter pipeline_id={self.pipeline_id} "
            f"sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks}"
        )

        # Single global owner: one sender for the whole model (all PP layers gathered by owner).
        src_rank = self._select_global_sender_rank()
        comm_plan, group_name, tgt_ranks_in_group = self._build_comm_plan_for_sender(
            sync_id=sync_id,
            src_rank=src_rank,
            tgt_dp_ranks=tgt_dp_ranks,
        )
        logger.info(
            "[ModelUpdateService] selective_sync_plan "
            f"pipeline_id={self.pipeline_id} sync_id={sync_id} src_rank={src_rank} "
            f"broadcast_tgt_ranks={tgt_ranks_in_group} "
            f"ipc_targets={[e['dp_rank'] for e in comm_plan[src_rank].get('ipc_targets', [])]} "
            f"pg_timeout_s={self._pg_timeout_s}"
        )

        setup_refs = []
        if tgt_ranks_in_group:
            # Sender joins as rank 0; receivers join as ranks 1..N (dynamic comm_plan pattern).
            # Only broadcast-path workers call setup_collective_group; IPC-only ranks skip it.
            for tgt_rank in tgt_ranks_in_group:
                setup_refs.append(
                    self.tgt_cluster.rank2worker[int(tgt_rank)].setup_collective_group.remote(
                        model_update_name=sync_id,
                        comm_plan=comm_plan,
                        mode="receiver",
                        timeout_s=self._pg_timeout_s,
                    )
                )
            setup_refs.append(
                self.src_cluster.rank2worker[int(src_rank)].setup_collective_group.remote(
                    model_update_name=sync_id,
                    comm_plan=comm_plan,
                    mode="sender",
                    timeout_s=self._pg_timeout_s,
                )
            )

        try:
            if setup_refs:
                self._ray_get_with_timeout(
                    setup_refs,
                    timeout_s=self._timeout_s,
                    desc=(
                        "[ModelUpdateService] setup_collective_groups "
                        f"pipeline_id={self.pipeline_id} sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks}"
                    ),
                )

            # Dispatch sync RPC to all train workers. Only the global owner does transport;
            # non-owners return immediately. ray.get(sync_refs) provides the sync barrier.
            sync_refs = []
            for rank, worker in enumerate(self.src_cluster.workers):
                is_owner = int(rank) == src_rank
                # comm_plan is always non-None for the owner (carries ipc_targets even when
                # tgt_ranks_in_group is empty). Non-owners receive None and return immediately.
                rank_comm_plan = comm_plan if is_owner else None
                sync_refs.append(
                    worker.selective_sync_active_cache.remote(
                        sync_id=sync_id,
                        comm_plan=rank_comm_plan,
                        tgt_dp_ranks=tgt_dp_ranks,
                        tgt_workers=self.tgt_cluster.workers,
                        tgt_device_mapping=tgt_device_mapping,
                        tgt_num_gpus_per_worker=int(tgt_num_gpus_per_worker),
                        adapters_to_sync=adapters_to_sync,
                    )
                )
            self._ray_get_with_timeout(
                sync_refs,
                timeout_s=self._timeout_s,
                desc=(
                    "[ModelUpdateService] sync_selected_workers "
                    f"pipeline_id={self.pipeline_id} sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks}"
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                "[ModelUpdateService] selective sync failed. "
                f"pipeline_id={self.pipeline_id} sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks} "
                f"timeout_s={self._timeout_s}. "
                "This is a fail-fast guard to avoid indefinite hangs in sync_selected_workers."
            ) from exc
        # NCCL groups are destroyed inside selective_sync_active_cache (owner side) before returning.
        # ray.get(sync_refs) above confirms teardown is complete.

        logger.info(
            f"[ModelUpdateService] sync_selected_workers_exit pipeline_id={self.pipeline_id} sync_id={sync_id}"
        )
