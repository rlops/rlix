"""RLix pipeline adapter for NeMo RL async GRPO training.

NemoRLFullFinetunePipeline is a Ray actor created by PipelineCoordinator and
managed by the RLix scheduler. It implements the same resize_infer interface as
RollFullFinetunePipeline so the coordinator can drive shrink/expand without
knowing which backend is running.

Key design choices vs RollFullFinetunePipeline:
  - Training loop is NeMo RL's async_grpo_train() (not ROLL AgenticPipeline).
  - Weight sync is selective (NemoRLModelUpdateService), not full NCCL broadcast.
  - Inference routing state is owned by VllmGeneration._active_dp_ranks (F2).
  - Weight version is the training step that produced the CPU cache. Active
    refresh and later expand of the same cache publish the same version (F6).

Feature dependencies in this file:
  F5  — scheduler-driven shrink/expand, hooks, bootstrap lifecycle
  F6  — _expand_workers atomic wake+sync+version+activate
  F2  — VllmGeneration.sleep_partial / wake_up_partial / mark_dp_ranks_inactive
         / activate_dp_ranks  (called here, implemented in NeMo RL repo)
  F4  — NemoRLModelUpdateService.sync_selected_workers (CPU bucket cache)
  F11 — policy.offload_training_gpu / destroy_nccl_groups (called in after_training)
  F12 — shared PlacementGroup from RollResourceManagerProxy (called in initialize)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, List, Optional

import ray

from rlix.pipeline.nemo_rl_model_update_service import NemoRLModelUpdateService
from rlix.pipeline.utils import validate_resize_params
from rlix.protocol.types import (
    ACTOR_TRAIN_CLUSTER_NAME,
    COORDINATOR_ACTOR_NAME_PREFIX,
    GENERATION_CLUSTER_NAME,
    RLIX_NAMESPACE,
    SCHEDULER_ACTOR_NAME,
    ActionResponse,
    Priority,
    get_pipeline_namespace,
)
from rlix.utils.ray import get_actor_or_raise

logger = logging.getLogger(__name__)

_BOOTSTRAP_CACHE_VERSION = -1


# ---------------------------------------------------------------------------
# RLix hooks — real implementation injected into async_grpo_train
# ---------------------------------------------------------------------------

class NemoRLRLixHooks:
    """Real RLix hooks for NemoRLFullFinetunePipeline.

    Injected into async_grpo_train as the rlix_hooks parameter. Holds a direct
    reference to the pipeline actor (same Ray actor execution context, so no
    remote call needed).
    """

    def __init__(self, pipeline: "NemoRLFullFinetunePipeline") -> None:
        self._pipeline = pipeline

    def before_training(self, step: int) -> None:
        """Block until the scheduler grants the training GPU allocation.

        Scheduler asynchronously shrinks overlap inference workers before
        granting this request, freeing VRAM for the training phase.
        """
        logger.info(
            "[NemoRLRLixHooks] before_training step=%d — requesting actor_train GPUs",
            step,
        )
        self._pipeline._request_cluster_gpus(
            cluster_id=self._pipeline._actor_train_cluster_id,
            priority=Priority.ACTOR_TRAINING,
            global_step=step,
        )
        logger.info(
            "[NemoRLRLixHooks] before_training step=%d — actor_train GPUs granted", step
        )

    def after_training(self, step: int) -> int:
        """Refresh active inference ranks, then release the training GPU.

        Non-overlap inference ranks may keep serving throughout training and
        therefore will not pass through expand. They must receive the latest
        base weights before the scheduler is told actor_train GPUs are free.
        """
        logger.info(
            "[NemoRLRLixHooks] after_training step=%d — syncing active base weights",
            step,
        )
        version = self._pipeline._after_training(step=step)
        self._pipeline._notify_release_cluster_gpus(
            cluster_id=self._pipeline._actor_train_cluster_id,
            global_step=step,
        )
        return version

    def on_trajectory_collector_created(self, collector: Any) -> None:
        """Register the trajectory collector handle with the pipeline actor.

        _expand_workers() uses this handle to call set_weight_version after
        each selective sync, ensuring routing activation only happens after
        the collector has been told about the new weight version.
        """
        logger.info(
            "[NemoRLRLixHooks] on_trajectory_collector_created — registering collector"
        )
        self._pipeline._trajectory_collector = collector


# ---------------------------------------------------------------------------
# Pipeline actor
# ---------------------------------------------------------------------------

class NemoRLFullFinetunePipeline:
    """RLix-controlled pipeline adapter for NeMo RL async GRPO training.

    Lifecycle managed by PipelineCoordinator:
      coordinator.create_pipeline_actor()  →  __init__
      coordinator.resize_infer(remove=..)  →  _shrink_workers
      coordinator.resize_infer(add=..)     →  _expand_workers (F6 atomic)
      pipeline_actor.run()                 →  async_grpo_train with hooks

    Register with orchestrator using NemoRLConfigBridge.cluster_tp_configs and
    cluster_device_mappings. Set pipeline_cls in the config to the dotted path
    of this class so PipelineCoordinator can dynamically load it.
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any) -> None:
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise ValueError("pipeline_id must be a non-empty str")
        self._pipeline_id = pipeline_id
        self._pipeline_config = pipeline_config
        self._initialized = False
        # Guard initialize_pipeline() so resize_infer() cannot race it.
        self._init_lock = threading.Lock()
        # Serialize scheduler-driven resize_infer calls.
        self._infer_resize_lock = threading.Lock()

        self._rlix_scheduler = get_actor_or_raise(
            SCHEDULER_ACTOR_NAME,
            RLIX_NAMESPACE,
            error_context=(
                "NemoRLFullFinetunePipeline requires the central RLix scheduler "
                "actor to exist before startup."
            ),
        )

        self._actor_train_cluster_id = f"{pipeline_id}_{ACTOR_TRAIN_CLUSTER_NAME}"
        self._actor_infer_cluster_id = f"{pipeline_id}_{GENERATION_CLUSTER_NAME}"

        # State owned exclusively by this actor (single writer).
        self._trajectory_collector: Optional[Any] = None  # set by on_trajectory_collector_created
        self._current_weight_version: int = -1  # equals _cache_ready_step after publish
        self._cache_ready_step: int = -1  # updated in after_training (F4/F11 path)

        # Introspectable state — read-only externally, written only by expand/shrink.
        # active_dp_ranks mirrors VllmGeneration._active_dp_ranks (F2 owns ground truth).
        # pre_activation_ranks tracks ranks between wake_up and activate (F6 atomic window).
        self._active_dp_ranks: set = set()
        self._pre_activation_ranks: set = set()  # woken but not yet in routing

        # NeMo RL runtime objects — created during initialize_pipeline().
        self._policy: Optional[Any] = None
        self._policy_generation: Optional[Any] = None
        self._model_update_service: Optional[Any] = None

        self._coordinator_handle: Optional[Any] = None

    # ------------------------------------------------------------------
    # Coordinator handle
    # ------------------------------------------------------------------

    def _get_coordinator_handle(self) -> Any:
        if self._coordinator_handle is not None:
            return self._coordinator_handle
        namespace = get_pipeline_namespace(self._pipeline_id)
        actor_name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{self._pipeline_id}"
        self._coordinator_handle = get_actor_or_raise(
            actor_name,
            namespace,
            error_context=f"Coordinator required for pipeline_id={self._pipeline_id!r}.",
        )
        return self._coordinator_handle

    # ------------------------------------------------------------------
    # Scheduler RPC helpers
    # ------------------------------------------------------------------

    def _request_cluster_gpus(
        self,
        *,
        cluster_id: str,
        priority: Any,
        global_step: int,
        step_target_estimate: Optional[int] = None,
    ) -> List[int]:
        """Block until scheduler allocates GPUs; return allocated GPU IDs."""
        allocated = ray.get(
            self._rlix_scheduler.request_gpus.remote(
                cluster_id=str(cluster_id),
                priority=priority,
                global_step=global_step,
                step_target_estimate=step_target_estimate,
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(
                f"scheduler.request_gpus returned non-list: {type(allocated).__name__}"
            )
        return [int(x) for x in allocated]

    def _notify_release_cluster_gpus(
        self, *, cluster_id: str, global_step: int
    ) -> None:
        """Notify scheduler that a cluster's GPUs are released to the idle pool."""
        ray.get(
            self._rlix_scheduler.notify_release_gpus.remote(
                cluster_id=str(cluster_id),
                global_step=global_step,
            )
        )

    # ------------------------------------------------------------------
    # Bootstrap — Feature 5
    # ------------------------------------------------------------------

    def initialize_pipeline(self) -> ActionResponse:
        """Bootstrap NeMo RL workers under INITIALIZATION scheduler priority.

        Sequence (must not be reordered — each phase depends on the previous):

          Phase 1 — Training init (INITIALIZATION):
            Request actor_train GPUs → initialize Megatron policy
            → build_cpu_bucket_cache(-1)  [F4 stub]
            → offload_training_gpu()      [F11 stub]
            → destroy_nccl_groups()       [F11 stub]
            → release actor_train

          Phase 2 — Inference init (INITIALIZATION):
            Request actor_infer GPUs → initialize vLLM policy_generation
            → vLLM sleep(level=2)         [F1]
            → release actor_infer

          Phase 3 — Service + routing:
            Create NemoRLModelUpdateService [F4/F6]
            Shrink all DP ranks to zero     [F2 stub — routing disabled until
                                             scheduler grants GENERATION GPUs]

        Returns ActionResponse(success=True) on completion.
        """
        with self._init_lock:
            if self._initialized:
                return ActionResponse(success=True)

            logger.info(
                "[%s] initialize_pipeline start", self._pipeline_id
            )

            # ----------------------------------------------------------------
            # Phase 1: Training init
            # ----------------------------------------------------------------
            init_step = _BOOTSTRAP_CACHE_VERSION
            self._request_cluster_gpus(
                cluster_id=self._actor_train_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_step,
            )
            logger.info("[%s] actor_train GPUs granted", self._pipeline_id)

            try:
                self._init_training_workers()

                # F4 stub: build CPU bucket cache for base model weights.
                # Full implementation in Feature 4 (megatron_policy_worker.py).
                self._build_cpu_bucket_cache(step=init_step, is_bootstrap=True)
                self._cache_ready_step = init_step

                # F11 stubs: offload training GPU VRAM + destroy NCCL groups.
                # Needed so inference workers can wake_up on overlap GPUs without OOM.
                self._offload_training_gpu()
                self._destroy_nccl_groups()

            finally:
                self._notify_release_cluster_gpus(
                    cluster_id=self._actor_train_cluster_id,
                    global_step=init_step,
                )
            logger.info("[%s] actor_train released", self._pipeline_id)

            # ----------------------------------------------------------------
            # Phase 2: Inference init
            # ----------------------------------------------------------------
            self._request_cluster_gpus(
                cluster_id=self._actor_infer_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_step,
            )
            logger.info("[%s] actor_infer GPUs granted", self._pipeline_id)

            try:
                self._init_inference_workers()

                # F1: vLLM sleep(level=2) — drop weights + KV cache, free VRAM.
                # F2: after this, all DP ranks are sleeping.
                self._sleep_all_inference_workers()

            finally:
                self._notify_release_cluster_gpus(
                    cluster_id=self._actor_infer_cluster_id,
                    global_step=init_step,
                )
            logger.info("[%s] actor_infer released", self._pipeline_id)

            # ----------------------------------------------------------------
            # Phase 3: Service creation + routing disabled
            # ----------------------------------------------------------------
            self._create_model_update_service()

            # All DP ranks sleeping; routing disabled until scheduler expand.
            # F2: VllmGeneration._active_dp_ranks starts as empty set after
            # _sleep_all_inference_workers() calls finish_generation() on all ranks.
            logger.info(
                "[%s] initialize_pipeline complete — waiting for scheduler grant",
                self._pipeline_id,
            )
            self._initialized = True
            return ActionResponse(success=True)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            resp = self.initialize_pipeline()
            if not getattr(resp, "success", False):
                raise RuntimeError(f"initialize_pipeline failed: {resp!r}")

    # ------------------------------------------------------------------
    # Shrink — Feature 5 / Feature 2
    # ------------------------------------------------------------------

    def _shrink_workers(self, *, dp_ranks_to_remove: List[int]) -> None:
        """Abort-drain-sleep selected DP shards.

        Delegates to VllmGeneration.sleep_partial() which implements the
        abort → drain (poll engine idle) → sleep sequence (Feature 2).
        sleep_partial is an async method; we run it in a fresh event loop to
        keep this sync Ray actor method unblocked.
        """
        if not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be non-empty")

        logger.info(
            "[%s] _shrink_workers dp_ranks=%s", self._pipeline_id, dp_ranks_to_remove
        )

        if self._policy_generation is None:
            logger.warning(
                "[%s] _shrink_workers: policy_generation not initialized yet; skipping",
                self._pipeline_id,
            )
            return

        # Feature 2: VllmGeneration.sleep_partial(dp_ranks, level=2)
        # Implements: mark _preempted_shards → abort_all_requests → drain → sleep.
        # It's an async method because drain needs to poll engine idle.
        asyncio.run(
            self._policy_generation.sleep_partial(dp_ranks_to_remove, level=2)
        )

    # ------------------------------------------------------------------
    # Expand — Feature 6 (atomic wake + selective sync + version + routing)
    # ------------------------------------------------------------------

    def _expand_workers(self, *, dp_ranks_to_add: List[int]) -> None:
        """Atomic expand: wake → selective sync → version update → activate routing.

        F6 correctness invariant: activate_dp_ranks (step 5) is ONLY reached if
        sync_selected_workers (step 3) AND set_weight_version (step 4) both succeed.
        A failure in steps 3-5 leaves the ranks in a "woken-but-inactive" state —
        they will not serve generation requests with stale weights.

        State transitions:
          Before: ranks in sleeping set (not in _active_dp_ranks)
          Step 1: marks ranks as pre-activation (mark_dp_ranks_inactive is a no-op
                  here since they are already inactive, but makes intent explicit)
          Step 2: ranks wake up (GPU VRAM restored); _pre_activation_ranks updated
          Steps 3-4: weight sync + version update (atomic block, no routing yet)
          Step 5: ranks move from _pre_activation_ranks → _active_dp_ranks

        If any of steps 3-5 raise, _pre_activation_ranks retains the stale entries
        so callers / tests can inspect the failed state.

        Called inside coordinator._resize_sync_lock (coordinator.resize_infer holds
        the lock for the full duration, preventing concurrent expand/shrink races).
        """
        if not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be non-empty")

        ranks = list(dp_ranks_to_add)
        logger.info("[%s] _expand_workers start dp_ranks=%s", self._pipeline_id, ranks)

        if self._policy_generation is None:
            raise RuntimeError(
                f"[{self._pipeline_id}] _expand_workers: policy_generation is None; "
                "cannot expand — call initialize_pipeline() first"
            )
        if self._model_update_service is None:
            raise RuntimeError(
                f"[{self._pipeline_id}] _expand_workers: model_update_service is None; "
                "cannot expand without weight sync (would activate stale weights)"
            )
        if self._trajectory_collector is None:
            raise RuntimeError(
                f"[{self._pipeline_id}] _expand_workers: trajectory_collector is None; "
                "cannot expand without version update (register via on_trajectory_collector_created)"
            )

        # Step 1: Explicitly keep ranks out of routing before wake-up.
        # F2: VllmGeneration.mark_dp_ranks_inactive — idempotent for sleeping ranks,
        # but documents intent and sets _preempted_shards to block new dispatches.
        self._policy_generation.mark_dp_ranks_inactive(ranks)

        # Step 2: Wake sleeping workers (training already offloaded — no OOM risk).
        # F2: VllmGeneration.wake_up_partial(dp_ranks)
        self._policy_generation.wake_up_partial(ranks)
        self._pre_activation_ranks.update(ranks)

        # Steps 3-5: atomic block.
        # Any exception here means activate_dp_ranks is NOT called.
        # Ranks remain in _pre_activation_ranks (woken but not in routing).
        try:
            # Step 3: Selective weight sync — only woken shards, no global pause.
            # F4: NemoRLModelUpdateService.sync_selected_workers (CPU bucket → GPU)
            ray.get(
                self._model_update_service.sync_selected_workers.remote(
                    tgt_dp_ranks=ranks,
                )
            )
            logger.info(
                "[%s] _expand_workers: sync_selected_workers done", self._pipeline_id
            )

            self._finalize_weight_update(ranks)

            # Step 4: publish the cache version BEFORE routing activation.
            # Expand reuses the same CPU cache as active refresh, so it must not
            # bump the version for the same weights.
            new_version = self._publish_weight_version()
            logger.info(
                "[%s] _expand_workers: weight_version → %d",
                self._pipeline_id,
                new_version,
            )

            # Step 5: Activate routing — reached only if steps 3+4 succeeded.
            # F3: VllmGeneration.activate_dp_ranks adds ranks to _active_dp_ranks.
            self._policy_generation.activate_dp_ranks(ranks)
            self._active_dp_ranks.update(ranks)
            self._pre_activation_ranks.difference_update(ranks)

            logger.info(
                "[%s] _expand_workers complete — dp_ranks=%s now active, "
                "weight_version=%d",
                self._pipeline_id,
                ranks,
                self._current_weight_version,
            )

        except Exception:
            # Ranks are awake but NOT in routing. Weights may be stale.
            # _pre_activation_ranks still contains these ranks for diagnostic inspection.
            logger.error(
                "[%s] _expand_workers FAILED during sync/version/activate. "
                "Ranks %s are woken but inactive (not in routing). "
                "Inspect _pre_activation_ranks. weight_version unchanged at %d.",
                self._pipeline_id,
                ranks,
                self._current_weight_version,
            )
            raise

    # ------------------------------------------------------------------
    # resize_infer — coordinator entry point (Feature 5)
    # ------------------------------------------------------------------

    def resize_infer(
        self,
        *,
        dp_ranks_to_remove: List[int],
        dp_ranks_to_add: List[int],
    ) -> ActionResponse:
        """Scheduler-driven shrink or expand of the inference cluster.

        Called by PipelineCoordinator.resize_infer() which holds
        _resize_sync_lock for the duration, serializing with sync_lora_weights.
        Exactly one of dp_ranks_to_remove / dp_ranks_to_add must be non-empty.
        """
        self._ensure_initialized()
        validate_resize_params(dp_ranks_to_remove, dp_ranks_to_add)

        with self._infer_resize_lock:
            if dp_ranks_to_remove:
                self._shrink_workers(dp_ranks_to_remove=list(dp_ranks_to_remove))
            else:
                self._expand_workers(dp_ranks_to_add=list(dp_ranks_to_add))

        return ActionResponse(success=True)

    # ------------------------------------------------------------------
    # Training loop — Feature 5
    # ------------------------------------------------------------------

    def _after_training(self, *, step: int) -> int:
        """Post-train critical path: cache, offload, active sync, version publish."""
        self._build_cpu_bucket_cache(step=step)
        self._cache_ready_step = int(step)

        self._offload_training_gpu()
        self._destroy_nccl_groups()

        coordinator = self._get_coordinator_handle()
        active_ranks = ray.get(coordinator.sync_base_weights_to_active.remote())
        active_ranks = [int(rank) for rank in (active_ranks or [])]
        if active_ranks:
            self._finalize_weight_update(active_ranks)

        return self._publish_weight_version()

    def run(self) -> None:
        """Start async GRPO training with RLix hooks injected.

        Creates NemoRLRLixHooks (which holds a reference back to this actor),
        then calls async_grpo_train(). The hooks fire scheduler RPCs at
        before_training / after_training boundaries, which drives the
        scheduler-controlled shrink/expand cycle.

        NOTE: The actual NeMo RL object setup (policy, policy_generation,
        dataloader, tokenizer, etc.) requires Feature 12 shared PG support
        and is handled by _setup_nemo_rl_objects(). See that method for the
        full initialization sequence.
        """
        self._ensure_initialized()

        from nemo_rl.algorithms.grpo import async_grpo_train

        hooks = NemoRLRLixHooks(pipeline=self)

        # Set up NeMo RL runtime objects from pipeline_config.
        (
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            nemo_logger,
            checkpointer,
            grpo_save_state,
            master_config,
            max_trajectory_age_steps,
        ) = self._setup_nemo_rl_objects()

        logger.info("[%s] Starting async_grpo_train with RLix hooks", self._pipeline_id)
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=nemo_logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_save_state,
            master_config=master_config,
            max_trajectory_age_steps=max_trajectory_age_steps,
            rlix_hooks=hooks,
        )

    # ------------------------------------------------------------------
    # NeMo RL object setup — Feature 12 dependency
    # ------------------------------------------------------------------

    def _setup_nemo_rl_objects(self) -> tuple:
        """Create NeMo RL runtime objects from pipeline_config.

        In the full implementation this mirrors examples/run_grpo.py:
          - Create Policy (Megatron backend) on shared PG from F12.
          - Create VllmGeneration on shared PG from F12.
          - Build dataloader, tokenizer, loss_fn, checkpointer.
          - Return them for async_grpo_train.

        Feature 12 dependency: Policy and VllmGeneration must be initialized
        on placement groups obtained from RollResourceManagerProxy (shared PG),
        not via RayVirtualCluster.create() which would conflict with ROLL workers
        in mixed-deployment mode.

        Raises:
            NotImplementedError: Until Feature 12 (shared PG) is implemented
                and wired into this method.
        """
        raise NotImplementedError(
            "_setup_nemo_rl_objects requires Feature 12 (shared PlacementGroup) "
            "to be implemented. In the meantime, call async_grpo_train directly "
            "from your training script and pass rlix_hooks=NemoRLRLixHooks(pipeline)."
        )

    # ------------------------------------------------------------------
    # Phase helpers — stubs for other Features
    # ------------------------------------------------------------------

    def _init_training_workers(self) -> None:
        """Initialize Megatron training workers on shared PG.

        Feature 12 dependency: uses RollResourceManagerProxy placement group.
        Feature 4  dependency: workers must expose build_cpu_bucket_cache().
        """
        if self._policy is None:
            logger.warning(
                "[%s] _init_training_workers: policy not set; "
                "skipping (F12 stub)",
                self._pipeline_id,
            )
            return
        logger.info("[%s] Initializing Megatron training workers", self._pipeline_id)

    def _init_inference_workers(self) -> None:
        """Initialize vLLM inference workers on shared PG.

        Feature 12 dependency: uses RollResourceManagerProxy placement group.
        Feature 1  dependency: workers must accept sleep_level=2.
        """
        if self._policy_generation is None:
            logger.warning(
                "[%s] _init_inference_workers: policy_generation not set; "
                "skipping (F12 stub)",
                self._pipeline_id,
            )
            return
        logger.info("[%s] Initializing vLLM inference workers", self._pipeline_id)

    def _sleep_all_inference_workers(self) -> None:
        """Put all vLLM DP shards to sleep (level=2) after initialization.

        After this call, all inference workers have released GPU VRAM.
        Routing is effectively disabled (all DP ranks sleeping).
        Scheduler expand will wake the required shards before training.
        """
        if self._policy_generation is None:
            logger.warning(
                "[%s] _sleep_all_inference_workers: policy_generation not set; "
                "skipping",
                self._pipeline_id,
            )
            return
        # Feature 1: finish_generation() calls vLLM sleep(level=self._sleep_level).
        # Feature 2: marks all DP ranks as inactive via sleep_partial path.
        if hasattr(self._policy_generation, "finish_generation"):
            self._policy_generation.finish_generation()
        logger.info(
            "[%s] All inference workers sleeping (level=2)", self._pipeline_id
        )

    def _build_cpu_bucket_cache(self, step: int, *, is_bootstrap: bool = False) -> None:
        """Build CPU bucket cache snapshot of current training weights.

        Feature 4 dependency: implemented in megatron_policy_worker.py.
        If the policy has no cache builder yet, fail fast rather than letting
        inference serve stale weights under a new version.
        """
        if self._policy is None or not hasattr(self._policy, "build_cpu_bucket_cache"):
            if is_bootstrap:
                logger.info(
                    "[%s] _build_cpu_bucket_cache bootstrap version=%d skipped; policy cache builder unavailable",
                    self._pipeline_id,
                    step,
                )
                return
            raise NotImplementedError(
                "NeMo RL policy must implement build_cpu_bucket_cache(step) before "
                "Feature 5+6 weight refresh can run safely."
            )
        ray.get(self._policy.build_cpu_bucket_cache.remote(step))

    def _offload_training_gpu(self) -> None:
        """Release training GPU VRAM so inference can wake_up on overlap GPUs.

        Feature 11 dependency: implemented as policy.offload_training_gpu().
        """
        if self._policy is not None and hasattr(self._policy, "offload_training_gpu"):
            ray.get(self._policy.offload_training_gpu.remote())
            return
        logger.warning("[%s] policy.offload_training_gpu unavailable", self._pipeline_id)

    def _destroy_nccl_groups(self) -> None:
        """Destroy Megatron NCCL communicator groups to release their VRAM.

        Feature 11 dependency: implemented in nccl_offload.py (NeMo RL repo).
        NCCL communicator buffers can use hundreds of MB on the GPU even when
        training is idle. Without this, inference wake_up on overlap GPUs may OOM.
        """
        if self._policy is not None and hasattr(self._policy, "destroy_nccl_groups"):
            ray.get(self._policy.destroy_nccl_groups.remote())
            return
        logger.warning("[%s] policy.destroy_nccl_groups unavailable", self._pipeline_id)

    def _finalize_weight_update(self, dp_ranks: List[int]) -> None:
        """Run one post-load finalization on each target vLLM worker."""
        ranks = sorted(set(int(rank) for rank in dp_ranks))
        if not ranks:
            return
        if self._policy_generation is None:
            raise RuntimeError("policy_generation is required for finalize_weight_update")

        if not hasattr(self._policy_generation, "finalize_weight_update"):
            raise RuntimeError("policy_generation must expose finalize_weight_update(dp_ranks)")
        ray.get(self._policy_generation.finalize_weight_update(ranks))

    def _publish_weight_version(self) -> int:
        """Publish the cache-producing step as the current collector version."""
        if self._trajectory_collector is None:
            raise RuntimeError("trajectory_collector is required before publishing weight version")
        version = int(self._cache_ready_step)
        ray.get(self._trajectory_collector.set_weight_version.remote(version))
        self._current_weight_version = version
        return version

    def _create_model_update_service(self) -> None:
        """Create NemoRLModelUpdateService Ray actor in the pipeline namespace."""
        namespace = get_pipeline_namespace(self._pipeline_id)
        svc_name = f"{self._pipeline_id}_nemo_rl_model_update_service"

        from rlix.utils.env import pipeline_identity_env_vars

        runtime_env = {
            "env_vars": {
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                **pipeline_identity_env_vars(
                    pipeline_id=self._pipeline_id,
                    ray_namespace=namespace,
                ),
            }
        }

        svc = NemoRLModelUpdateService.options(
            name=svc_name,
            namespace=namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            runtime_env=runtime_env,
            lifetime="detached",
        ).remote(
            pipeline_id=self._pipeline_id,
            policy=self._policy,
            policy_generation=self._policy_generation,
        )
        ray.get(svc.__ray_ready__.remote())
        self._model_update_service = svc
        logger.info(
            "[%s] NemoRLModelUpdateService created (name=%s namespace=%s)",
            self._pipeline_id,
            svc_name,
            namespace,
        )
