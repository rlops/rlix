"""Selective model weight sync for NeMo RL pipelines on scheduler-driven expand.

When the scheduler expands a NeMo RL pipeline (adds sleeping inference shards),
this service pushes the latest training weights from the CPU bucket cache to the
woken inference workers.

Transport paths (mirroring NeMo RL's existing transports):
  - CUDA IPC   — sender and receiver share the same physical GPU (overlap shards).
                 Zero-copy; only correct path when two ranks are on the same GPU.
  - NCCL bcast — receiver is on a different GPU. Uses NeMo RL's packed_broadcast
                 producer/consumer pattern (model_update.py collective group).

This service is a Ray actor; one instance per pipeline, created by
NemoRLFullFinetunePipeline.initialize_pipeline().

NOTE (Feature 4 dependency):
    sync_selected_workers currently raises NotImplementedError until the CPU
    bucket cache (Feature 4) and selective transport routing (Feature 4/6) are
    implemented in the NeMo RL repo. The interface is complete so F5/F6 wiring
    compiles and can be tested end-to-end once F4 lands.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import ray

logger = logging.getLogger(__name__)


@ray.remote
class NemoRLModelUpdateService:
    """Per-pipeline selective weight sync service for NeMo RL.

    Holds references to the Megatron training policy and the vLLM generation
    interface. sync_selected_workers is called in two scenarios:
      - expand path: DP ranks that just woke up (scheduler-driven expand).
      - active refresh path: DP ranks currently serving requests (partial-overlap
        ranks that did not shrink during training and will not pass through expand).
    In both cases, untargeted shards are not contacted and continue generation.

    Args:
        pipeline_id:       Unique identifier for this pipeline.
        policy:            NeMo RL ColocatablePolicyInterface (Megatron backend).
                           Must expose build_cpu_bucket_cache / cache_ready_step
                           once Feature 4 is implemented.
        policy_generation: NeMo RL VllmGeneration instance owning the vLLM workers.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        policy: Any,
        policy_generation: Any,
    ) -> None:
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise ValueError("pipeline_id must be a non-empty str")
        self._pipeline_id = pipeline_id
        self._policy = policy
        self._policy_generation = policy_generation

        logger.info(
            "[NemoRLModelUpdateService] init pipeline_id=%s", pipeline_id
        )

    def sync_selected_workers(
        self,
        tgt_dp_ranks: List[int],
        verify: bool = False,
    ) -> None:
        """Push latest training weights to the specified inference DP shards.

        High-level flow (once Feature 4 is implemented):
          1. Assert CPU bucket cache is ready (_cache_ready_step >= 0).
          2. Determine transport per target device:
               - Same physical GPU as cache owner → CUDA IPC (zero-copy).
               - Different GPU                    → NCCL broadcast.
          3. For each bucket in the CPU cache:
               a. Stage CPU → GPU (sender side, controlled staging buffer).
               b. Send via IPC handle (colocated) or NCCL broadcast (remote).
               c. Receiver calls model_runner.model.load_weights() to apply.
               d. Release staging buffer before next bucket.
          4. Optionally verify weights via checksum comparison.

        Non-targeted shards (non-overlap GPUs) are NOT contacted; they continue
        generation without pause.

        Args:
            tgt_dp_ranks: DP ranks to push weights to. Two callers:
                          - expand path: ranks that just woke up, not yet routing.
                          - active refresh path: ranks currently serving requests;
                            implementation must synchronize CUDA streams after
                            load_weights() to avoid mid-inference weight switching.
            verify:       When True, run post-sync weight verification checksums.

        Raises:
            NotImplementedError: Until Feature 4 (CPU bucket cache) is implemented.
        """
        if not tgt_dp_ranks:
            raise ValueError("tgt_dp_ranks must be non-empty")

        logger.info(
            "[NemoRLModelUpdateService] sync_selected_workers "
            "pipeline_id=%s tgt_dp_ranks=%s",
            self._pipeline_id,
            tgt_dp_ranks,
        )

        raise NotImplementedError(
            "NeMo RL selective base-weight sync requires the Feature 4 sender "
            "implementation (CPU bucket cache transport). Refusing to mark stale "
            "inference workers as synced."
        )

    def __repr__(self) -> str:
        return f"NemoRLModelUpdateService(pipeline_id={self._pipeline_id!r})"
