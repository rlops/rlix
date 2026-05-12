"""Selective model weight sync for NeMo RL pipelines on scheduler-driven expand.

When the scheduler expands a NeMo RL pipeline (adds sleeping inference shards),
this service pushes the latest training weights from the CPU bucket cache to the
woken inference workers.

Transport paths:
  - cpu_serialize — CPU uint8 bucket DMA-copied to each receiver GPU.
                    Default; works across all GPU topologies.
  - cuda_ipc      — Zero-copy CUDA IPC handle; only when sender and receiver
                    share the same physical GPU (colocated overlap shards).
  - NCCL bcast    — Broadcast via StatelessProcessGroup; cross-GPU non-colocated.

This service is a Ray actor; one instance per pipeline, created by
NemoRLFullFinetunePipeline.initialize_pipeline().
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

import ray

logger = logging.getLogger(__name__)


@ray.remote
class NemoRLModelUpdateService:
    """Per-pipeline selective weight sync service for NeMo RL.

    Holds references to the Megatron training policy and the vLLM generation
    interface. sync_selected_workers is called in two scenarios:
      - expand path: DP ranks that just woke up (scheduler-driven expand).
      - active refresh path: DP ranks currently serving requests.

    Args:
        pipeline_id:       Unique identifier for this pipeline.
        policy:            NeMo RL policy object. Must expose worker actors that
                           implement selective_sync_active_cache (MegatronPolicyWorkerImpl).
                           Supported patterns: .src_cluster.workers, .workers, list, single actor.
        policy_generation: VllmGeneration Python object (not a Ray actor).
        policy_workers:    Optional pre-resolved training worker actor handles.
        model_update_receiver:
                           Optional pre-resolved inference receiver surface.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        policy: Any,
        policy_generation: Any,
        policy_workers: Optional[List[Any]] = None,
        model_update_receiver: Optional[Any] = None,
    ) -> None:
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise ValueError("pipeline_id must be a non-empty str")
        self._pipeline_id = pipeline_id
        self._policy = policy
        self._policy_generation = policy_generation
        self._policy_workers = list(policy_workers or [])
        self._model_update_receiver = model_update_receiver

        logger.info("[NemoRLModelUpdateService] init pipeline_id=%s", pipeline_id)

    def sync_selected_workers(
        self,
        tgt_dp_ranks: List[int],
        verify: bool = False,
    ) -> None:
        """Push active CPU bucket cache to the specified inference DP shards.

        Flow:
          1. Get inference receiver surface from VllmGeneration.
          2. Build comm plan (cpu_serialize, no NCCL topology analysis).
          3. Call selective_sync_active_cache on ALL training workers;
             only the cache owner (pp0/dp0/tp0) does actual transport.
          4. Finalize post-load hooks on inference workers.
          5. Optionally verify weight checksums.

        Args:
            tgt_dp_ranks: Inference DP ranks to update.
            verify:       When True, run post-sync checksum verification.
        """
        if not tgt_dp_ranks:
            raise ValueError("tgt_dp_ranks must be non-empty")

        logger.info(
            "[NemoRLModelUpdateService] sync_selected_workers start "
            "pipeline_id=%s tgt_dp_ranks=%s",
            self._pipeline_id,
            tgt_dp_ranks,
        )

        # --- Step 1: inference receiver surface ---
        # VllmGeneration is a plain Python class (not a Ray actor). Prefer the
        # pre-resolved receiver surface so this Ray actor only stores actor
        # handles and small config objects.
        if self._model_update_receiver is not None:
            receiver = self._model_update_receiver
        else:
            receiver = self._policy_generation.get_model_update_receiver()
        num_gpus_per_worker: int = int(receiver.worker_config.num_gpus_per_worker)
        device_mapping: List[int] = list(receiver.worker_config.device_mapping or [])
        dp_size: int = len(receiver.rank2worker)

        # Build tgt_workers as a list indexed by dp_rank (required by
        # selective_sync_active_cache: tgt_workers[dp_rank] → leader actor).
        tgt_workers_indexed = [receiver.rank2worker[r] for r in range(dp_size)]

        # --- Step 2: comm plan (cpu_serialize — no NCCL group needed) ---
        sync_id = f"{self._pipeline_id}_{uuid.uuid4().hex[:8]}"
        comm_plan = {
            sync_id: {
                "group_name": sync_id,
                "master_addr": "127.0.0.1",
                "master_port": 0,          # unused for cpu_serialize
                "tgt_devices": [],         # unused for cpu_serialize
                "ipc_targets": [
                    {
                        "dp_rank": dp_rank,
                        "local_ranks": list(range(num_gpus_per_worker)),
                    }
                    for dp_rank in tgt_dp_ranks
                ],
                "broadcast_local_ranks_by_dp_rank": {},   # no NCCL
            }
        }

        # --- Step 3: run selective sync on all training workers ---
        # selective_sync_active_cache is a no-op on non-owner ranks.
        policy_workers = self._get_policy_workers()
        sync_refs = [
            w.selective_sync_active_cache.remote(
                sync_id=sync_id,
                comm_plan=comm_plan,
                tgt_dp_ranks=tgt_dp_ranks,
                tgt_workers=tgt_workers_indexed,
                tgt_device_mapping=device_mapping or list(range(dp_size)),
                tgt_num_gpus_per_worker=num_gpus_per_worker,
                model_update_transport="cpu_serialize",
            )
            for w in policy_workers
        ]
        results = ray.get(sync_refs)

        # --- Step 4: finalize post-load hooks on all inference workers ---
        # VllmGeneration.finalize_weight_update() is a pass-through that calls
        # process_weights_after_loading on all workers (idempotent).
        if self._policy_generation is not None:
            self._policy_generation.finalize_weight_update()
        else:
            ray.get(
                [
                    receiver.rank2worker[int(dp_rank)].finalize_weight_update.remote()
                    for dp_rank in range(dp_size)
                ]
            )

        # --- Step 5: optional weight verification ---
        if verify:
            weight_stats: Optional[dict] = None
            for r in results:
                if isinstance(r, dict) and "weight_stats" in r:
                    weight_stats = r["weight_stats"]
                    break
            if weight_stats:
                if self._policy_generation is not None:
                    self._policy_generation.verify_model(weight_stats)
                else:
                    ray.get(
                        [
                            receiver.rank2worker[int(dp_rank)].verify_model.remote(
                                weight_stats
                            )
                            for dp_rank in range(dp_size)
                        ]
                    )

        logger.info(
            "[NemoRLModelUpdateService] sync_selected_workers done "
            "pipeline_id=%s tgt_dp_ranks=%s",
            self._pipeline_id,
            tgt_dp_ranks,
        )

    def _get_policy_workers(self) -> List[Any]:
        """Resolve list of training worker Ray actor handles from self._policy.

        Tries common NeMo RL policy API patterns in priority order:
          1. policy.worker_group.workers (NeMo RL Policy pattern)
          2. policy.src_cluster.workers  (NeMo RL ClusterSpec pattern)
          3. policy.workers              (direct cluster with .workers list)
          4. policy itself is a list/tuple of Ray actor handles
        """
        if self._policy_workers:
            return list(self._policy_workers)

        # Pattern 1: policy.worker_group.workers
        worker_group = getattr(self._policy, "worker_group", None)
        if worker_group is not None:
            workers = getattr(worker_group, "workers", None)
            if workers:
                return list(workers)

        # Pattern 2: policy.src_cluster.workers
        src_cluster = getattr(self._policy, "src_cluster", None)
        if src_cluster is not None:
            workers = getattr(src_cluster, "workers", None)
            if workers:
                return list(workers)

        # Pattern 3: policy.workers
        workers = getattr(self._policy, "workers", None)
        if workers:
            return list(workers)

        # Pattern 4: policy is a list/tuple of actor handles
        if isinstance(self._policy, (list, tuple)) and self._policy:
            return list(self._policy)

        raise RuntimeError(
            f"[NemoRLModelUpdateService] Cannot resolve training workers from policy "
            f"(type={type(self._policy).__name__}). Policy must expose "
            ".worker_group.workers, .src_cluster.workers, .workers, or be a list "
            "of Ray actor handles."
        )

    def __repr__(self) -> str:
        return f"NemoRLModelUpdateService(pipeline_id={self._pipeline_id!r})"
