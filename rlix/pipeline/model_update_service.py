"""Per-pipeline service for selective model weight synchronization on pipeline expand.

When the scheduler expands a pipeline (adds infer workers), the new workers need
up-to-date model weights from the training cluster. This service orchestrates that
transfer using two transport paths:

- **CUDA IPC**: zero-copy transfer when sender and receiver share the same physical GPU.
- **NCCL broadcast**: cross-GPU transfer via a temporary collective group.

The service is a Ray actor, one per pipeline, created by the coordinator on expand.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import ray
from roll.distributed.executor.cluster import Cluster
from roll.utils.constants import GLOBAL_STORAGE_NAMESPACE, STORAGE_NAME
from roll.utils.logging import get_logger

from rlix.utils.env import parse_env_timeout_s

logger = get_logger()


@ray.remote
class ModelUpdateService:
    """Per-pipeline service for selective sync on expand.

    Contract:
    - Scheduler-side trigger only: no promotion forwarding, no validation, no coalescing.
    - Calls into sender-side sync, which serializes via sender cache_lock.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        src_cluster: Cluster,
        tgt_cluster: Cluster,
        model_update_transport: str = "cpu_serialize",
        bucket_size_bytes: Optional[int] = None,
    ):
        """Initialize the model update service for a single pipeline.

        Args:
            pipeline_id: Unique identifier for the pipeline this service belongs to.
            src_cluster: Training cluster that holds the authoritative model weights.
            tgt_cluster: Inference cluster whose workers will receive weight updates.
            model_update_transport: Transport mode for colocated (IPC) weight transfer.
                ``"cpu_serialize"`` — DMA to pinned CPU tensor, send via ZMQ multipart
                (default; avoids GPU memory for the staging buffer).
                ``"cuda_ipc"`` — CUDA IPC handle zero-copy (lower latency, requires
                sender and receiver on the same physical GPU).
                Non-colocated (cross-GPU) transfers always use the dynamic NCCL
                broadcast path regardless of this setting.
            bucket_size_bytes: Maximum bytes per bucket when staging CPU→GPU during
                sync.  Must be set explicitly in production; ``None`` skips the VRAM
                budget guard (acceptable only in tests / single-GPU setups).
                Spec: nemorl-port-plan.md line 343.
        """
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        _valid_transports = {"cpu_serialize", "cuda_ipc"}
        if model_update_transport not in _valid_transports:
            raise ValueError(
                f"model_update_transport={model_update_transport!r} is not valid; "
                f"choose one of {sorted(_valid_transports)}"
            )
        if bucket_size_bytes is not None and (not isinstance(bucket_size_bytes, int) or bucket_size_bytes <= 0):
            raise ValueError("bucket_size_bytes must be a positive int or None")

        self.pipeline_id = pipeline_id
        self.src_cluster: Any = src_cluster
        self.tgt_cluster: Any = tgt_cluster
        self.model_update_transport: str = model_update_transport
        self.bucket_size_bytes: Optional[int] = bucket_size_bytes

        # Nonce scopes NCCL group names to this service instance, avoiding collisions
        # when multiple services coexist (e.g. after a coordinator restart).
        self._sync_nonce = uuid.uuid4().hex[:8]
        self._master_addr_by_src_rank: Dict[int, str] = {}
        self._timeout_s: Optional[float] = parse_env_timeout_s("ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S", 150.0)
        self._pg_timeout_s: Optional[float] = parse_env_timeout_s("ROLL_SELECTIVE_MODEL_UPDATE_PG_TIMEOUT_S", 120.0)

    @staticmethod
    def _ray_get_with_timeout(refs: Any, *, timeout_s: Optional[float], desc: str) -> Any:
        """Wrapper around ``ray.get`` that raises ``TimeoutError`` with *desc* on timeout.

        If *timeout_s* is ``None``, waits indefinitely.
        """
        if timeout_s is None:
            return ray.get(refs)
        try:
            return ray.get(refs, timeout=float(timeout_s))
        except ray.exceptions.GetTimeoutError as exc:
            raise TimeoutError(f"{desc} timed out after {timeout_s}s") from exc

    @staticmethod
    def _release_master_port_claim(*, master_addr: str, master_port: int) -> None:
        """Release a previously claimed rendezvous port after sync teardown completes."""
        if master_addr == "" or master_port <= 0:
            return
        shared_storage = ray.get_actor(STORAGE_NAME, namespace=GLOBAL_STORAGE_NAMESPACE)
        master_addr_port_key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"
        ray.get(shared_storage.delete.remote(master_addr_port_key))

    def _get_master_addr(self, *, src_rank: int) -> str:
        """Return the cached sender IP for *src_rank*, fetching it once on first use."""
        cached = self._master_addr_by_src_rank.get(int(src_rank))
        if cached is not None:
            return cached
        src_worker = self.src_cluster.rank2worker[int(src_rank)]
        master_addr = str(ray.get(src_worker.get_node_ip.remote()))
        self._master_addr_by_src_rank[int(src_rank)] = master_addr
        return master_addr

    def _select_global_sender_rank(self) -> int:
        """Return the single global cache owner: pp_rank==0, dp_rank==0, tp_rank==0, cp_rank==0."""
        for rank, info in enumerate(self.src_cluster.worker_rank_info):
            if int(info.pp_rank) == 0 and int(info.dp_rank) == 0 and int(info.tp_rank) == 0 and int(info.cp_rank) == 0:
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
    ) -> Tuple[dict[int, Any], str, List[int]]:
        """Build a communication plan for the single global cache owner.

        The plan decides, for every device on every target worker, which transport
        path to use:

        - **IPC path** — the target device sits on the same physical GPU as the
          sender (identified by matching ``(node_rank, gpu_rank)``).  CUDA IPC
          gives zero-copy access so no NCCL group is needed.  NCCL *cannot* form
          a group when two ranks share a GPU, so IPC is not just faster — it is
          the only correct path for colocated devices.
        - **Broadcast path** — the target device is on a different GPU.  A
          temporary NCCL collective group is created (sender as rank 0, receivers
          as ranks 1..N) and weights are broadcast over it.

        A single target worker may have a mix of IPC and broadcast devices (e.g.
        TP across 4 GPUs where 1 is colocated with the sender and 3 are not).

        The method also queries the sender worker for a free port and IP to use
        as the NCCL rendezvous master.

        Args:
            sync_id: Unique identifier for this sync operation, embedded in the
                NCCL group name to avoid collisions with concurrent syncs.
            src_rank: Global rank of the cache owner in the training cluster.
            tgt_dp_ranks: Data-parallel ranks in the inference cluster to sync.

        Returns:
            A 3-tuple of ``(comm_plan, group_name, tgt_ranks_in_group)``:

            - **comm_plan**: ``{src_rank: plan_dict}`` — keyed by the owner's
              rank. ``plan_dict`` contains:

              - ``group_name`` / ``master_addr`` / ``master_port``: NCCL
                rendezvous coordinates for the broadcast path.
              - ``src_rank``, ``src_pp_rank``: bookkeeping for
                ``_setup_collective_group_impl()``; ``src_pp_rank`` is always 0
                because selective sync gathers all PP layers into one sender.
              - ``ipc_targets``: ``[{dp_rank, local_ranks}]`` — **IPC path**
                targets: for each target worker with colocated devices, lists
                which of its device ranks share a physical GPU with the sender.
              - ``tgt_devices``: ``[{rank, device}]`` — **broadcast path**
                targets: the complement of ``ipc_targets``; devices on different
                GPUs from the sender that will join the NCCL collective group.
              - ``broadcast_local_ranks_by_dp_rank``: ``{dp_rank: [local_ranks]}``
                — tells the owner which local ranks on each target worker joined
                the NCCL group, so it broadcasts to the right subset.

            - **group_name**: NCCL group name for broadcast-path setup.

            - **tgt_ranks_in_group**: sorted dp_ranks that have at least one
              broadcast-path device. Drives which workers call
              ``setup_collective_group`` before the sync. Empty when all targets
              are IPC-only (no NCCL setup needed).
        """
        src_rank = int(src_rank)
        src_worker = self.src_cluster.rank2worker[src_rank]

        src_devices = self.src_cluster.rank2devices.get(src_rank, [])
        if not src_devices:
            raise RuntimeError(f"Missing src devices for src_rank={src_rank}")
        for device in src_devices:
            if device.get("node_rank") is None or device.get("gpu_rank") is None:
                raise RuntimeError(
                    f"Incomplete device metadata for src_rank={src_rank}: "
                    f"node_rank={device.get('node_rank')}, gpu_rank={device.get('gpu_rank')}"
                )
        src_gpu_keys: Set[Tuple[int, int]] = {(int(d["node_rank"]), int(d["gpu_rank"])) for d in src_devices}

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

        # Only fetch NCCL rendezvous coordinates when broadcast-path workers exist.
        if tgt_ranks_in_group:
            master_addr = self._get_master_addr(src_rank=src_rank)
            master_port = int(ray.get(src_worker.get_free_port.remote()))
        else:
            master_addr = ""
            master_port = 0

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

    def sync_selected_workers(
        self,
        tgt_dp_ranks: List[int],
        adapters_to_sync: list[str] | None = None,
        verify: bool = True,
    ) -> None:
        """Push model weights from the training cluster to specific infer workers.

        High-level flow:
        1. Validate target ranks and read cluster topology.
        2. Select the single global cache owner on the training side.
        3. Build a comm plan that classifies each target device as IPC or broadcast.
        4. Stand up temporary NCCL groups for broadcast-path workers.
        5. Dispatch ``selective_sync_active_cache`` to all training workers
           (only the owner actually transfers; others return immediately).
        6. Optionally verify transferred weights against sender-side checksums.

        Args:
            tgt_dp_ranks: Data-parallel ranks in the inference cluster to update.
            adapters_to_sync: If provided, only sync these LoRA adapter names
                instead of the full model weights.
            verify: When ``True``, run a post-sync weight verification pass.
        """
        tgt_dp_ranks = sorted(set(int(r) for r in tgt_dp_ranks))
        if not tgt_dp_ranks:
            raise ValueError("tgt_dp_ranks must be non-empty")

        infer_world_size = int(self.tgt_cluster.world_size)
        invalid = [r for r in tgt_dp_ranks if r < 0 or r >= infer_world_size]
        if invalid:
            raise ValueError(f"Invalid tgt_dp_ranks={invalid}; infer_world_size={infer_world_size}")

        # device_mapping tells us which physical GPUs each infer worker occupies,
        # needed to decide IPC vs broadcast for each target device.
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
        master_addr = str(comm_plan[src_rank]["master_addr"])
        master_port = int(comm_plan[src_rank]["master_port"])
        sync_completed = False

        # --- Phase 1: Set up temporary NCCL collective groups for broadcast-path workers ---
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

            # --- Phase 2: Dispatch sync to all training workers ---
            # Only the global cache owner actually transfers weights; non-owners return
            # immediately. ray.get(sync_refs) acts as the sync barrier.
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
                        model_update_transport=self.model_update_transport,
                    )
                )
            sync_results = self._ray_get_with_timeout(
                sync_refs,
                timeout_s=self._timeout_s,
                desc=(
                    "[ModelUpdateService] sync_selected_workers "
                    f"pipeline_id={self.pipeline_id} sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks}"
                ),
            )
            sync_completed = True
        except Exception as exc:
            raise RuntimeError(
                "[ModelUpdateService] selective sync failed. "
                f"pipeline_id={self.pipeline_id} sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks} "
                f"timeout_s={self._timeout_s}. "
                "This is a fail-fast guard to avoid indefinite hangs in sync_selected_workers."
            ) from exc
        finally:
            # On failure: intentionally leak the port claim — remote workers may still hold
            # the port and releasing it would risk collision on a future sync.
            # On success: release is deferred to AFTER receiver teardown (Phase 4 below),
            # so the claim covers the full sync+teardown cycle per spec (lines 380-389).
            pass

        # --- Phase 4: Receiver-side NCCL group teardown ---
        # The sender destroys its group inside selective_sync_active_cache before returning.
        # Receivers must also destroy their side — the group_name is shared.
        # Port claim is released AFTER teardown so it covers the full cycle.
        # Spec: nemorl-port-plan.md lines 380-389.
        if tgt_ranks_in_group:
            teardown_refs = [
                self.tgt_cluster.rank2worker[int(tgt_rank)].destroy_collective_group.remote(group_name)
                for tgt_rank in tgt_ranks_in_group
            ]
            self._ray_get_with_timeout(
                teardown_refs,
                timeout_s=self._timeout_s,
                desc=(
                    "[ModelUpdateService] destroy_collective_group (receivers) "
                    f"pipeline_id={self.pipeline_id} sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks}"
                ),
            )
            logger.info(
                "[ModelUpdateService] receiver_nccl_teardown_ok "
                f"pipeline_id={self.pipeline_id} sync_id={sync_id}"
            )

        # Release port claim after full teardown cycle (spec: nemorl-port-plan.md lines 380-389).
        if sync_completed:
            self._release_master_port_claim(master_addr=master_addr, master_port=master_port)

        # --- Phase 5: Post-sync verification ---
        # Spec (nemorl-port-plan.md line 624-632): finalize_weight_update() is owned
        # by the PIPELINE, not ModelUpdateService — the pipeline calls it after
        # sync_selected_workers() returns, because the pipeline controls the full
        # expand sequence (sync → finalize → version_publish → activate_routing).
        # ModelUpdateService does NOT call finalize here.
        # The cache owner returns weight_stats (checksums / norms) alongside the sync result.
        # We forward these to each target worker's verify_model to confirm weights landed correctly.
        if verify:
            sender_stats: dict[str, Any] = {}
            for result in sync_results:
                if isinstance(result, dict) and result.get("weight_stats"):
                    sender_stats = result["weight_stats"]
                    break
            if sender_stats:
                verify_refs = [
                    self.tgt_cluster.rank2worker[int(dp_rank)].verify_model.remote(expected_stats=sender_stats)
                    for dp_rank in tgt_dp_ranks
                ]
                self._ray_get_with_timeout(
                    verify_refs,
                    timeout_s=self._timeout_s,
                    desc=(
                        "[ModelUpdateService] verify_model "
                        f"pipeline_id={self.pipeline_id} sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks}"
                    ),
                )
                logger.info(f"[ModelUpdateService] verify_model_ok pipeline_id={self.pipeline_id} sync_id={sync_id}")

        logger.info(
            f"[ModelUpdateService] sync_selected_workers_exit pipeline_id={self.pipeline_id} sync_id={sync_id}"
        )
