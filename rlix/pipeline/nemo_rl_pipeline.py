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

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


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
        print(
            f"[RLIX_HOOK {self._pipeline._pipeline_id}] before_training step={step} "
            f"— requesting actor_train GPUs",
            flush=True,
        )
        self._pipeline._request_cluster_gpus(
            cluster_id=self._pipeline._actor_train_cluster_id,
            priority=Priority.ACTOR_TRAINING,
            global_step=step,
        )
        print(
            f"[RLIX_HOOK {self._pipeline._pipeline_id}] before_training step={step} "
            f"— actor_train GPUs granted",
            flush=True,
        )

    def before_weight_sync(self, step: int) -> None:
        """Build the CPU bucket cache while parameters are still on GPU.

        grpo.py's weight_sync block calls ``policy.offload_after_refit()`` then
        ``destroy_megatron_nccl_groups()`` before invoking ``after_training``.
        Both swap the parameters' .data with empty storage, so we have to
        snapshot the freshly-trained weights here (cf. debug_log #34).
        """
        print(
            f"[RLIX_HOOK {self._pipeline._pipeline_id}] before_weight_sync step={step} "
            f"— building CPU bucket cache",
            flush=True,
        )
        self._pipeline._before_weight_sync(step=step)

    def after_training(self, step: int) -> int:
        """Refresh active inference ranks, then release the training GPU.

        Non-overlap inference ranks may keep serving throughout training and
        therefore will not pass through expand. They must receive the latest
        base weights before the scheduler is told actor_train GPUs are free.
        """
        print(
            f"[RLIX_HOOK {self._pipeline._pipeline_id}] after_training step={step} "
            f"— syncing active base weights",
            flush=True,
        )
        version = self._pipeline._after_training(step=step)
        self._pipeline._notify_release_cluster_gpus(
            cluster_id=self._pipeline._actor_train_cluster_id,
            global_step=step,
        )
        print(
            f"[RLIX_HOOK {self._pipeline._pipeline_id}] after_training step={step} "
            f"— version={version}, actor_train released",
            flush=True,
        )
        return version

    def on_trajectory_collector_created(self, collector: Any) -> None:
        """Register the trajectory collector handle with the pipeline actor,
        then issue the initial Priority.GENERATION request so the scheduler
        wakes vLLM before ATC starts generating.

        Why GENERATION here: the NeMo RL pipeline never tells the scheduler
        about generation demand on its own (unlike full_finetune_pipeline.py
        which requests per-step in run()). Without this signal the scheduler
        sees zero demand → never expands → vLLM stays asleep → ATC.generate()
        routes to no active rank → infinite stall.

        Order matters: register the collector first so the scheduler-triggered
        _expand_workers() can call set_weight_version on it (line 467-471
        gate). Then block on the GENERATION grant; the scheduler will plan
        an expand which calls coordinator.resize_infer → _expand_workers →
        wake_up_partial → sync_selected_workers → activate_dp_ranks. Returns
        only when at least one inference dp_rank is active and routing is on.
        """
        pid = self._pipeline._pipeline_id
        print(
            f"[RLIX_HOOK {pid}] on_trajectory_collector_created — registering collector",
            flush=True,
        )
        self._pipeline._trajectory_collector = collector

        print(
            f"[RLIX_HOOK {pid}] on_trajectory_collector_created — requesting GENERATION GPUs",
            flush=True,
        )
        # step_target_estimate must be > 0 so planner.plan_generation_gap_ratio
        # doesn't `continue` past us when no progress reports exist yet
        # (planner.py:226-231). Use 1 as a minimal positive estimate; the actual
        # number doesn't affect routing for a single-pipeline-per-GPU layout —
        # planner just needs non-zero demand to assign at least one DP worker.
        allocated = self._pipeline._request_cluster_gpus(
            cluster_id=self._pipeline._actor_infer_cluster_id,
            priority=Priority.GENERATION,
            global_step=0,
            step_target_estimate=1,
        )
        print(
            f"[RLIX_HOOK {pid}] on_trajectory_collector_created — GENERATION granted "
            f"gpus={allocated}, active_dp_ranks={sorted(self._pipeline._active_dp_ranks)}",
            flush=True,
        )

        # Start the generation-grant watchdog. Once ATC is registered, it has
        # ongoing demand the scheduler doesn't know about — the watchdog
        # re-requests GENERATION whenever the cluster has been shrunk to 0
        # (e.g. by another pipeline's actor_train INITIALIZATION preempting
        # an overlapping GPU). See _generation_watchdog_loop for details.
        self._pipeline._start_generation_watchdog()

    def begin_progress_batch(self, step: int, count_intended: int) -> None:
        pass

    def end_progress_batch(self, step: int, trajectories_collected: int) -> None:
        pass

    def __reduce__(self):
        # AsyncTrajectoryCollector (a separate Ray actor) takes rlix_hooks as a ctor
        # arg and only invokes begin/end_progress_batch (both no-ops above). The
        # pipeline ref carries threading.Lock and a NeMo RL policy → not picklable.
        # Reconstruct on the ATC side as a state-less stub that satisfies the
        # protocol; pipeline-side calls (before/after_training, on_trajectory_collector_created)
        # all run in the pipeline actor and never go through pickle.
        return (_NemoRLRLixHooksATCStub, ())


class _NemoRLRLixHooksATCStub:
    """No-op stub used in AsyncTrajectoryCollector after pickling."""

    def before_training(self, step: int) -> None:
        pass

    def before_weight_sync(self, step: int) -> None:
        pass

    def after_training(self, step: int) -> int:
        return -1

    def on_trajectory_collector_created(self, collector: Any) -> None:
        pass

    def begin_progress_batch(self, step: int, count_intended: int) -> None:
        pass

    def end_progress_batch(self, step: int, trajectories_collected: int) -> None:
        pass


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
        self._nemo_setup_result: Optional[tuple] = None

        self._coordinator_handle: Optional[Any] = None

        # debug #58: configurable vLLM sleep level (default 2 = drop weights;
        # 1 = retain weight pool VAs, bypass _sleep_saved_buffers restore path).
        self._vllm_sleep_level: int = self._read_vllm_sleep_level()

        # Step-boundary admission signal: launcher uses this to defer admitting
        # the next pipeline until this one has done one full step (init + ATC +
        # step 0 + after_training). Set inside _after_training after the first
        # successful version publish. Avoids cgroup pids.max=3840 burst when
        # both pipelines try to spawn vLLM EngineCore concurrently (debug #44).
        self._first_after_training_event = threading.Event()

        # Pair-init barrier (debug #48): ppl_i's first _after_training blocks
        # until launcher signals that ppl_{i+1} has finished vLLM init. This
        # prevents ppl_{i+1} vLLM init from racing with ppl_i's step 1+ train,
        # which would steal GPU memory and cause "No available KV cache" errors
        # in vLLM's _check_enough_kv_cache_memory. Set by external setter.
        self._pair_setup_complete_event = threading.Event()
        # Initially set so single-ppl mode and pipelines without a pair
        # don't block waiting for a signal that will never come.
        self._pair_setup_complete_event.set()

        # Setup-complete signal (debug #48): set inside initialize_pipeline
        # after _setup_nemo_rl_objects returns so the launcher can detect when
        # this pipeline's vLLM is ready and unblock the paired pipeline's
        # _after_training pair-init barrier.
        self._setup_complete_event = threading.Event()

        # Generation-grant watchdog. The scheduler treats GENERATION as a one-shot
        # request: once granted, it is not automatically re-issued if the cluster
        # is later shrunk to make room for a higher-priority cluster (e.g. another
        # pipeline's actor_train INITIALIZATION on an overlapping GPU). Without a
        # persistent demand signal this leaves ATC stuck waiting for ranks that
        # the scheduler has no reason to re-expand. The watchdog re-requests
        # Priority.GENERATION whenever active+pre_activation ranks are empty
        # while ATC is alive, so the scheduler restores ranks once the
        # higher-priority work releases.
        self._gen_watchdog_thread: Optional[threading.Thread] = None
        self._gen_watchdog_stop = threading.Event()
        self._gen_watchdog_interval_s = 2.0

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

    def _await_release_actor_infer(self, *, global_step: int) -> None:
        """Block until scheduler commits the actor_infer shrink-to-zero for this pipeline.

        Mirrors ROLL's full_finetune_pipeline._await_release_actor_infer (line 645).
        Used at end of run() so that GENERATION cluster is released through the
        scheduler's planned-release path rather than via ActorDiedError cascade
        (debug #50 / debug #64 cleanup race). The scheduler's await_release_gpus
        only supports GENERATION priority clusters (scheduler.py:1801).
        """
        # Read timeout from env, fall back to 300s. Same env knob ROLL uses.
        import os as _os
        try:
            timeout_s = float(_os.environ.get("RLIX_NOTIFY_READY_TIMEOUT_S", "300"))
        except (TypeError, ValueError):
            timeout_s = 300.0
        ray.get(
            self._rlix_scheduler.await_release_gpus.remote(
                cluster_id=self._actor_infer_cluster_id,
                global_step=global_step,
                timeout_s=timeout_s,
            )
        )
        logger.info(
            "[rlix][%s] await_release_gpus done: step=%s",
            self._pipeline_id, global_step,
        )

    def _read_vllm_sleep_level(self) -> int:
        """Read actor_infer.strategy_args.strategy_config.sleep_level (default 2).

        debug #58: level 1 bypasses vLLM `_sleep_saved_buffers` restore path
        (cross-tenant CuMemAllocator VA poisoning). Level 2 is the rlix default
        (max VRAM freed for co-tenant).
        """
        actor_infer = _config_get(self._pipeline_config, "actor_infer", None)
        if actor_infer is None:
            return 2
        strategy_args = _config_get(actor_infer, "strategy_args", None)
        if strategy_args is None:
            return 2
        strategy_config = _config_get(strategy_args, "strategy_config", None)
        if strategy_config is None:
            return 2
        level = _config_get(strategy_config, "sleep_level", 2)
        try:
            return int(level)
        except (TypeError, ValueError):
            return 2

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

            # Build the NeMo Policy/VllmGeneration objects once. They are plain
            # Python handles that own Ray worker groups, so all later lifecycle
            # calls must run after this setup has populated self._policy and
            # self._policy_generation.
            self._setup_nemo_rl_objects()

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

                # F11: offload training GPU VRAM so inference workers can wake_up
                # on overlap GPUs without OOM. Disjoint topology: own train/infer
                # are on different physical GPUs and cross-pipeline overlap is
                # mediated by the scheduler's shrink/expand (sleep_partial), so
                # we keep the pp_group alive — destroying it here breaks
                # grpo.py:get_logprobs() at step 0 (cf. debug_log #24, #34).
                # grpo.py's own weight_sync block destroys + snapshots pp_group
                # at step boundaries.
                self._offload_training_gpu()

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

                # debug #68: pre-warm every DP rank by exercising one
                # wake_up_partial → sleep_partial cycle while we still hold
                # actor_infer GPUs and before any other pipeline's Megatron
                # touches the overlapping GPU. Without prewarm, the FIRST
                # wake of a previously-inactive rank on a GPU recently used
                # by another pipeline's training fails with CUDA illegal
                # memory access (v75/v76 regression of v74 milestone).
                self._prewarm_inference_ranks()

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
            # Signal launcher that NeMo RL setup (incl. vLLM init) is done so
            # the paired pipeline's pair-init barrier can be released (debug #48).
            self._setup_complete_event.set()
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

        Always delegates to VllmGeneration.sleep_partial() — debug #57 (v50):
        the empty-active-set guard in sleep_partial was lifted upstream so this
        is a single code path. sleep_all is no longer reachable from rlix
        scheduler-driven shrinks (cf. debug #55 — sleep_all → wake →
        finalize_weight_update CUDA crash on stale _k_scale buffer).
        """
        if not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be non-empty")

        print(
            f"[RLIX_PPL {self._pipeline_id}] _shrink_workers START dp_ranks={dp_ranks_to_remove} "
            f"active_before={sorted(self._active_dp_ranks)}",
            flush=True,
        )

        if self._policy_generation is None:
            logger.warning(
                "[%s] _shrink_workers: policy_generation not initialized yet; skipping",
                self._pipeline_id,
            )
            return

        target_set = set(int(r) for r in dp_ranks_to_remove)
        ok = self._policy_generation.sleep_partial(
            dp_ranks_to_remove, level=self._vllm_sleep_level, mode="abort"
        )
        if not ok:
            raise RuntimeError(
                f"[{self._pipeline_id}] sleep_partial failed for dp_ranks="
                f"{dp_ranks_to_remove}"
            )
        self._active_dp_ranks.difference_update(target_set)
        self._push_active_dp_ranks_to_collector()

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
        print(
            f"[RLIX_PPL {self._pipeline_id}] _expand_workers START dp_ranks={ranks}",
            flush=True,
        )

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
        # skip_activate=True: keep ranks off routing until weight sync finishes (Step 5).
        self._policy_generation.wake_up_partial(ranks, skip_activate=True)
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
            print(
                f"[RLIX_PPL {self._pipeline_id}] _expand_workers: sync_selected_workers done",
                flush=True,
            )

            # Step 4: publish the cache version BEFORE routing activation.
            # Expand reuses the same CPU cache as active refresh, so it must not
            # bump the version for the same weights.
            new_version = self._publish_weight_version()
            print(
                f"[RLIX_PPL {self._pipeline_id}] _expand_workers: weight_version -> {new_version}",
                flush=True,
            )

            # Step 5: Activate routing — reached only if steps 3+4 succeeded.
            # F3: VllmGeneration.activate_dp_ranks adds ranks to _active_dp_ranks.
            self._policy_generation.activate_dp_ranks(ranks)
            self._active_dp_ranks.update(ranks)
            self._pre_activation_ranks.difference_update(ranks)
            self._push_active_dp_ranks_to_collector()

            print(
                f"[RLIX_PPL {self._pipeline_id}] _expand_workers DONE dp_ranks={ranks} "
                f"now active; active_dp_ranks={sorted(self._active_dp_ranks)} "
                f"weight_version={self._current_weight_version}",
                flush=True,
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

    def _before_weight_sync(self, *, step: int) -> None:
        """Snapshot freshly-trained weights into the CPU bucket cache.

        Runs in grpo.py's weight_sync block BEFORE policy.offload_after_refit /
        destroy_megatron_nccl_groups, so the parameters are still on GPU with
        live storage. Doing the cache rebuild in _after_training (the previous
        order) saw zero-storage tensors and crashed (debug #34).

        Convention (cf. debug #36): _cache_ready_step is the weight_version
        the cached weights belong to. After training step N, the new weights
        are version N+1 — that is what ATC must see so it generates target
        weights in [N+1, N+1+max_age]. Storing N here would make ATC believe
        target [0..max_age] is already buffered and pause forever.
        """
        self._build_cpu_bucket_cache(step=step)
        self._cache_ready_step = int(step) + 1

    def _after_training(self, *, step: int) -> int:
        """Post-train critical path: active sync + version publish.

        grpo.py's weight_sync block has already done offload_after_refit +
        destroy_megatron_nccl_groups by the time this runs. Cache was built in
        _before_weight_sync. Here we only push the cached weights to active
        inference workers and publish the new version to ATC.
        """
        coordinator = self._get_coordinator_handle()
        ray.get(coordinator.sync_base_weights_to_active.remote())

        version = self._publish_weight_version()

        # Signal launcher that the first full step cycle has completed so the
        # next pipeline can be admitted (debug #44 step-boundary admission).
        first_after_training = not self._first_after_training_event.is_set()
        if first_after_training:
            self._first_after_training_event.set()

        # Pair-init barrier (debug #48): on first _after_training only, block
        # until launcher signals the paired pipeline's vLLM init is done. This
        # lets ppl_{i+1}'s vLLM init see GPU memory free of ppl_i Megatron
        # train. NoOp when no paired pipeline (event stays set).
        if first_after_training and not self._pair_setup_complete_event.is_set():
            print(
                f"[RLIX_PPL {self._pipeline_id}] _after_training step={step}: "
                f"holding pair-init barrier — waiting for paired pipeline vLLM ready",
                flush=True,
            )
            # Bounded wait so we don't hang forever if launcher never signals.
            ok = self._pair_setup_complete_event.wait(timeout=600.0)
            print(
                f"[RLIX_PPL {self._pipeline_id}] _after_training step={step}: "
                f"pair-init barrier released (signaled={ok})",
                flush=True,
            )

        return version

    def wait_for_first_after_training(self, timeout_s: Optional[float] = None) -> bool:
        """Block until ``_after_training`` has fired at least once.

        Used by the multi-pipeline launcher to serialize pipeline admission so
        ppl_{i+1}'s ray-actor / vLLM-EngineCore spawn does not collide with
        ppl_i's still-active init/offload thread peak (debug #44).

        Returns True if signaled within timeout, False on timeout.
        """
        return self._first_after_training_event.wait(timeout=timeout_s)

    def signal_pair_setup_complete(self) -> None:
        """Launcher signals that the paired pipeline's vLLM init is done.

        Unblocks this pipeline's first-after-training pair-init barrier so it
        can proceed to step 1+ training. See _pair_setup_complete_event docs
        and debug #48 for the cross-pipeline GPU memory race this avoids.
        """
        if not self._pair_setup_complete_event.is_set():
            self._pair_setup_complete_event.set()
            print(
                f"[RLIX_PPL {self._pipeline_id}] pair-init barrier released "
                f"— resuming step 1+ training",
                flush=True,
            )

    def arm_pair_setup_barrier(self) -> None:
        """Launcher arms (clears) the pair-init barrier on the leading pipeline.

        Must be called before launcher admits the second pipeline; ppl_i's
        ``_after_training`` will then block on this event until launcher calls
        ``signal_pair_setup_complete`` after ppl_{i+1}'s vLLM init reports done.
        """
        if self._pair_setup_complete_event.is_set():
            self._pair_setup_complete_event.clear()
            print(
                f"[RLIX_PPL {self._pipeline_id}] pair-init barrier armed "
                f"— first _after_training will block until paired vLLM ready",
                flush=True,
            )

    def wait_for_setup_complete(self, timeout_s: Optional[float] = None) -> bool:
        """Block until this pipeline's NeMo RL setup (incl. vLLM init) is done.

        Used by the launcher on the *trailing* pipeline so it can detect when
        ppl_{i+1}'s vLLM has finished init and unblock ppl_i's pair-init
        barrier. Set inside ``initialize_pipeline`` after _setup_nemo_rl_objects
        returns successfully.
        """
        return self._setup_complete_event.wait(timeout=timeout_s)

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
        try:
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
        finally:
            # Post-loop cleanup mirroring ROLL full_finetune_pipeline.run() lines 1170-1182.
            # Critical for multi-pipeline correctness (debug #64 cleanup cascade):
            # without explicit shrink-to-zero through the scheduler, this pipeline's
            # coordinator dies on Ray GC, scheduler triggers _gather_resize_tolerate_dead
            # auto-unregister, which races with peer pipelines' weight sync on shared GPU.
            #
            # Order matters:
            #  1. Stop the watchdog daemon FIRST so it cannot re-request GENERATION
            #     between our await_release and the actor's GC.
            #  2. await_release_actor_infer drives the scheduler's planned shrink-to-zero,
            #     committing the release before this actor dies.
            try:
                self._gen_watchdog_stop.set()
                if self._gen_watchdog_thread is not None and self._gen_watchdog_thread.is_alive():
                    self._gen_watchdog_thread.join(timeout=5.0)
                logger.info(
                    "[%s] post-run cleanup: watchdog stopped",
                    self._pipeline_id,
                )
            except Exception as exc:  # noqa: BLE001 — cleanup must not raise
                logger.warning(
                    "[%s] post-run watchdog stop failed: %s", self._pipeline_id, exc,
                )
            try:
                # Use _cache_ready_step as the final global_step (matches the version
                # we last published to ATC).
                last_step = max(int(self._cache_ready_step), 0)
                self._await_release_actor_infer(global_step=last_step)
            except Exception as exc:  # noqa: BLE001 — cleanup must not raise
                logger.warning(
                    "[%s] post-run await_release_actor_infer failed: %s",
                    self._pipeline_id, exc,
                )
            try:
                # Mirror ROLL gap 3: kill ATC so ppl1 does not busy-print
                # "All target weights already generated, pausing" and interfere
                # with peer ppl2's actor_infer rank scheduling.
                if self._trajectory_collector is not None:
                    ray.kill(self._trajectory_collector)
                    self._trajectory_collector = None
                    logger.info(
                        "[%s] post-run cleanup: ATC killed",
                        self._pipeline_id,
                    )
            except Exception as exc:  # noqa: BLE001 — cleanup must not raise
                logger.warning(
                    "[%s] post-run ray.kill(ATC) failed: %s",
                    self._pipeline_id, exc,
                )

    # ------------------------------------------------------------------
    # NeMo RL object setup — Feature 12 dependency
    # ------------------------------------------------------------------

    def _setup_nemo_rl_objects(self) -> tuple:
        """Create NeMo RL runtime objects from pipeline_config.

        Mirrors ``examples/run_grpo.py`` through tokenizer, generation config,
        response data, and ``grpo.setup()``. The only RLix-specific difference is
        that training and inference clusters are injected as shared-PG backed
        ``RLixVirtualClusterAdapter`` instances instead of letting NeMo RL create
        standalone ``RayVirtualCluster`` placement groups.
        """
        if self._nemo_setup_result is not None:
            return self._nemo_setup_result

        from omegaconf import OmegaConf

        from nemo_rl.algorithms.grpo import setup as grpo_setup
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.data.utils import setup_response_data
        from nemo_rl.models.generation import configure_generation_config
        from nemo_rl.utils.config import (
            load_config,
            parse_hydra_overrides,
            register_omegaconf_resolvers,
        )
        from nemo_rl.utils.logger import get_next_experiment_dir

        # Each NeMo RL pipeline shares the cluster's singleton PG, so the default
        # ``vllm_policy``/``lm_policy`` name prefixes collide across pipelines
        # (Ray actor names live in a single namespace per the IsolatedWorkerInitializer
        # spawn path). Suffix the prefix with this pipeline's id so worker actor
        # names like ``vllm_policy_ft_<id>-0-0`` stay unique.
        # Patch is process-local (each pipeline actor is its own Ray actor process),
        # so two pipelines patch their own module copies independently.
        import nemo_rl.distributed.worker_groups as _wg
        if not getattr(_wg.RayWorkerGroup.__init__, "_rlix_patched", False):
            _orig_rwg_init = _wg.RayWorkerGroup.__init__
            _pipeline_id_for_patch = self._pipeline_id

            def _patched_rwg_init(rwg_self, *args, name_prefix: str = "", **kwargs):
                if name_prefix and not name_prefix.endswith(_pipeline_id_for_patch):
                    name_prefix = f"{name_prefix}_{_pipeline_id_for_patch}"
                return _orig_rwg_init(rwg_self, *args, name_prefix=name_prefix, **kwargs)

            _patched_rwg_init._rlix_patched = True  # type: ignore[attr-defined]
            _wg.RayWorkerGroup.__init__ = _patched_rwg_init

        nemo_config_path = self._resolve_nemo_config_path()
        register_omegaconf_resolvers()
        cfg = load_config(nemo_config_path)

        overrides = _config_get(self._pipeline_config, "nemo_config_overrides", None)
        if overrides:
            cfg = parse_hydra_overrides(cfg, list(overrides))

        master_config = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(master_config, dict):
            raise RuntimeError(
                f"NeMo config {nemo_config_path!s} did not resolve to a dict"
            )

        logger.info("[%s] Loaded NeMo RL config from %s", self._pipeline_id, nemo_config_path)

        if bool(_config_get(self._pipeline_config, "nemo_increment_log_dir", True)):
            master_config["logger"]["log_dir"] = get_next_experiment_dir(
                master_config["logger"]["log_dir"]
            )

        tokenizer = get_tokenizer(master_config["policy"]["tokenizer"])
        if master_config["policy"]["generation"] is None:
            raise RuntimeError("NeMo RL GRPO requires policy.generation config")
        has_refit_draft_weights = bool(master_config["policy"]["draft"]["enabled"])
        master_config["policy"]["generation"] = configure_generation_config(
            master_config["policy"]["generation"],
            tokenizer,
            has_refit_draft_weights=has_refit_draft_weights,
        )

        dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
            tokenizer,
            master_config["data"],
            master_config["env"],
        )

        train_device_mapping = self._resolve_device_mapping(
            master_config, "train_device_mapping"
        )
        infer_device_mapping = self._resolve_device_mapping(
            master_config, "infer_device_mapping"
        )
        # Colocated mode: NeMo RL grpo.py:setup() requires
        # `train_cluster is inference_cluster` (literally the same Python
        # object). Build one shared cluster and alias both refs to it.
        # Disjoint mode: separate clusters per device_mapping.
        colocated_inference = bool(
            master_config.get("policy", {})
            .get("generation", {})
            .get("colocated", {})
            .get("enabled", False)
        )
        if colocated_inference:
            if list(train_device_mapping) != list(infer_device_mapping):
                raise ValueError(
                    f"colocated.enabled=true requires train_device_mapping == "
                    f"infer_device_mapping; got {train_device_mapping=} "
                    f"{infer_device_mapping=}"
                )
            train_cluster = self._make_rlix_virtual_cluster(
                name=f"{self._pipeline_id}_nemo_colocated",
                device_mapping=train_device_mapping,
                max_colocated_worker_groups=2,
                sorted_bundle_indices=train_device_mapping,
            )
            infer_cluster = train_cluster
        else:
            train_cluster = self._make_rlix_virtual_cluster(
                name=f"{self._pipeline_id}_nemo_train",
                device_mapping=train_device_mapping,
                max_colocated_worker_groups=1,
                sorted_bundle_indices=train_device_mapping,
            )
            infer_cluster = self._make_rlix_virtual_cluster(
                name=f"{self._pipeline_id}_nemo_infer",
                device_mapping=infer_device_mapping,
                max_colocated_worker_groups=1,
                sorted_bundle_indices=None,
            )

        (
            policy,
            policy_generation,
            _clusters,
            dataloader,
            val_dataloader,
            loss_fn,
            nemo_logger,
            checkpointer,
            grpo_save_state,
            master_config,
        ) = grpo_setup(
            master_config,
            tokenizer,
            dataset,
            val_dataset,
            external_train_cluster=train_cluster,
            external_inference_cluster=infer_cluster,
        )

        if policy_generation is not None:
            setattr(policy_generation, "_rlix_device_mapping", list(infer_device_mapping))

        self._policy = policy
        self._policy_generation = policy_generation
        if self._model_update_service is None:
            self._create_model_update_service()

        async_cfg = master_config["grpo"]["async_grpo"]
        self._nemo_setup_result = (
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
            int(async_cfg["max_trajectory_age_steps"]),
        )
        return self._nemo_setup_result

    def _resolve_nemo_config_path(self) -> Path:
        raw_path = (
            _config_get(self._pipeline_config, "nemo_config_path")
            or _config_get(self._pipeline_config, "nemo_rl_config_path")
            or _config_get(self._pipeline_config, "config")
        )
        if not raw_path:
            raise RuntimeError(
                "NemoRLFullFinetunePipeline requires pipeline_config.nemo_config_path"
            )
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"NeMo RL config not found: {path}")
        return path

    def _resolve_device_mapping(self, master_config: Dict[str, Any], key: str) -> List[int]:
        explicit = _config_get(self._pipeline_config, key)
        if explicit is None:
            explicit = (
                master_config.get("rlix", {}).get(key)
                if isinstance(master_config.get("rlix"), dict)
                else None
            )
        if explicit is None:
            raise RuntimeError(
                f"Missing {key}; provide pipeline_config.{key} or "
                f"nemo_config.rlix.{key}"
            )
        mapping = [int(x) for x in explicit]
        if not mapping:
            raise RuntimeError(f"{key} must be non-empty")
        return mapping

    def _make_rlix_virtual_cluster(
        self,
        *,
        name: str,
        device_mapping: List[int],
        max_colocated_worker_groups: int,
        sorted_bundle_indices: Optional[List[int]],
    ) -> Any:
        from rlix.pipeline.nemo_rl_virtual_cluster_adapter import RLixVirtualClusterAdapter

        pg_alloc = self._allocate_shared_pg(device_mapping=device_mapping)
        placement_groups = self._extract_placement_groups(pg_alloc)
        bundle_ct_per_node_list = self._extract_bundle_counts(
            pg_alloc=pg_alloc,
            placement_groups=placement_groups,
            device_mapping=device_mapping,
        )
        # Override max_colocated_worker_groups when running co-tenants on the
        # shared singleton PG: NeMo RL's RayWorkerGroup computes
        #   num_gpus = 1 / max_colocated_worker_groups
        # so a value of N lets up to N worker groups co-locate on each bundle.
        # The pipeline_config ``rlix_max_colocated_worker_groups`` overrides
        # the call-site default; default of 4 leaves headroom for 2 pipelines
        # × 2 worker types (train + infer) per shared bundle.
        override = _config_get(self._pipeline_config, "rlix_max_colocated_worker_groups")
        if override is not None:
            max_colocated_worker_groups = int(override)
        return RLixVirtualClusterAdapter(
            placement_groups=placement_groups,
            bundle_ct_per_node_list=bundle_ct_per_node_list,
            num_gpus_per_node=int(_config_get(self._pipeline_config, "num_gpus_per_node", 1)),
            use_gpus=True,
            max_colocated_worker_groups=max_colocated_worker_groups,
            name=name,
            sorted_bundle_indices=sorted_bundle_indices,
            device_mapping=device_mapping,
        )

    def _allocate_shared_pg(self, *, device_mapping: List[int]) -> Any:
        # Cluster-wide singleton placement group with one bundle per physical
        # GPU. All pipelines share this PG; per-pipeline / per-cluster device
        # routing is handled by NeMo RL via its ``cluster.device_mapping``-aware
        # bundle index selection (worker_groups.py RLix mode patch).
        #
        # Each bundle reserves a full GPU so the PG fits the host's actual
        # capacity. Workers individually request num_gpus=0.01 (RLix mode in
        # NeMo RL's worker_groups.py) so multiple workers from different
        # pipelines can colocated on the same bundle without exhausting Ray's
        # GPU accounting. CUDA_VISIBLE_DEVICES is pinned per worker to the
        # right physical GPU.
        from types import SimpleNamespace

        if len(device_mapping) <= 0:
            raise RuntimeError("device_mapping must be non-empty")

        ngpn = int(_config_get(self._pipeline_config, "num_gpus_per_node", 1))
        if ngpn <= 0:
            raise RuntimeError("num_gpus_per_node must be positive for GPU PG allocation")

        pg_name = "rlix-shared-gpu-pg"
        try:
            shared_pg = ray.util.get_placement_group(pg_name)
        except ValueError:
            bundles = [{"GPU": 1, "CPU": 4} for _ in range(ngpn)]
            shared_pg = ray.util.placement_group(
                bundles, strategy="PACK", name=pg_name
            )
        ray.get(shared_pg.ready())

        return SimpleNamespace(
            node_placement_groups=[shared_pg],
            bundle_ct_per_node_list=[len(device_mapping)],
        )

    def _extract_placement_groups(self, pg_alloc: Any) -> List[Any]:
        for attr in ("placement_groups", "pgs", "node_placement_groups"):
            value = getattr(pg_alloc, attr, None)
            if value:
                return list(value.values()) if isinstance(value, dict) else list(value)
        node2pg = getattr(pg_alloc, "node2pg", None)
        if node2pg:
            return [node2pg[k] for k in sorted(node2pg)]
        if isinstance(pg_alloc, (list, tuple)):
            # ROLL ResourceManager.allocate_placement_group returns List[List[Dict]]:
            # outer = workers, inner = per-GPU dicts {node_rank, gpu_rank, placement_group, ...}.
            # Collapse to unique PG objects ordered by first-seen node_rank.
            seen: Dict[int, Any] = {}
            for outer in pg_alloc:
                for entry in outer if isinstance(outer, (list, tuple)) else [outer]:
                    if isinstance(entry, dict) and "placement_group" in entry:
                        node_rank = int(entry.get("node_rank", 0))
                        seen.setdefault(node_rank, entry["placement_group"])
                    else:
                        # Allow direct PG / unknown entries too.
                        seen.setdefault(len(seen), entry)
            if seen:
                return [seen[k] for k in sorted(seen)]
            return list(pg_alloc)
        raise RuntimeError(
            "Unable to extract placement groups from RollResourceManagerProxy allocation"
        )

    def _extract_bundle_counts(
        self,
        *,
        pg_alloc: Any,
        placement_groups: List[Any],
        device_mapping: List[int],
    ) -> List[int]:
        for attr in ("bundle_ct_per_node_list", "bundle_counts", "workers_per_node"):
            value = getattr(pg_alloc, attr, None)
            if value:
                return [int(x) for x in value]
        if len(placement_groups) == 1:
            return [len(device_mapping)]
        # ROLL List[List[Dict]] case — count GPU dicts per node_rank, ordered by node.
        if isinstance(pg_alloc, (list, tuple)) and pg_alloc and isinstance(pg_alloc[0], (list, tuple)):
            counts: Dict[int, int] = {}
            for outer in pg_alloc:
                for entry in outer:
                    if isinstance(entry, dict):
                        node_rank = int(entry.get("node_rank", 0))
                        counts[node_rank] = counts.get(node_rank, 0) + 1
            if counts:
                return [counts[k] for k in sorted(counts)]
        return [int(getattr(pg, "bundle_count")) for pg in placement_groups]

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
        # Feature 1/2: sleep every DP rank and remove all ranks from routing.
        level = self._vllm_sleep_level
        if hasattr(self._policy_generation, "sleep_all"):
            ok = self._policy_generation.sleep_all(level=level, mode="abort")
        elif hasattr(self._policy_generation, "finish_generation"):
            ok = self._policy_generation.finish_generation()
        else:
            ok = False
        if not ok:
            raise RuntimeError(f"[{self._pipeline_id}] failed to sleep inference workers")
        logger.info(
            "[%s] All inference workers sleeping (level=%d)", self._pipeline_id, level
        )

    def _prewarm_inference_ranks(self) -> None:
        """Exercise the wake_up_partial → sleep_partial cycle once per DP rank.

        debug #68: the FIRST wake_up_partial of a DP rank that has not been
        activated this run hits CUDA illegal memory access when another
        pipeline's Megatron has touched the same physical GPU between
        construction and first activation. Pre-warm establishes the
        per-rank CuMemAllocator / CUDA-graph state immediately after
        construction (Phase 2) so subsequent wakes are second-time-or-later
        (less fragile under cross-process residual state).

        Called from ``initialize_pipeline`` right after
        ``_sleep_all_inference_workers``. The actor_infer GPUs are still held
        by this pipeline at this point (Phase 2), so other pipelines cannot
        interfere with the wake/sleep cycle.

        Implementation: ``wake_up_partial([rank], skip_activate=False)``
        wakes + adds to ``_active_dp_ranks`` + clears preempted; then
        ``sleep_partial([rank], level=L, mode="abort")`` reverses both. End
        state matches the post-``sleep_all`` invariant: ``_active_dp_ranks``
        empty + all ranks marked preempted.
        """
        if self._policy_generation is None:
            logger.warning(
                "[%s] _prewarm_inference_ranks: policy_generation not set; skipping",
                self._pipeline_id,
            )
            return
        try:
            dp_size = int(self._policy_generation.worker_group.dp_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] _prewarm_inference_ranks: cannot read dp_size (%s); skipping",
                self._pipeline_id, exc,
            )
            return
        if dp_size <= 0:
            return
        level = self._vllm_sleep_level
        for rank in range(dp_size):
            try:
                ok_wake = self._policy_generation.wake_up_partial(
                    [rank], skip_activate=False
                )
                ok_sleep = self._policy_generation.sleep_partial(
                    [rank], level=level, mode="abort"
                )
                print(
                    f"[RLIX_PPL {self._pipeline_id}] prewarm rank={rank} "
                    f"wake={ok_wake} sleep={ok_sleep}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort prewarm
                logger.warning(
                    "[%s] _prewarm_inference_ranks rank=%d failed: %s",
                    self._pipeline_id, rank, exc,
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
        self._policy.build_cpu_bucket_cache(step)

    def _offload_training_gpu(self) -> None:
        """Release training GPU VRAM so inference can wake_up on overlap GPUs.

        Feature 11 dependency: implemented as policy.offload_training_gpu().
        """
        if self._policy is not None and hasattr(self._policy, "offload_training_gpu"):
            self._policy.offload_training_gpu()
            return
        if self._policy is not None and hasattr(self._policy, "offload_after_refit"):
            self._policy.offload_after_refit()
            return
        logger.warning("[%s] policy.offload_training_gpu unavailable", self._pipeline_id)

    def _destroy_nccl_groups(self) -> None:
        """Destroy Megatron NCCL communicator groups to release their VRAM.

        Feature 11 dependency: implemented in nccl_offload.py (NeMo RL repo).
        NCCL communicator buffers can use hundreds of MB on the GPU even when
        training is idle. Without this, inference wake_up on overlap GPUs may OOM.
        """
        if self._policy is not None and hasattr(self._policy, "destroy_nccl_groups"):
            self._policy.destroy_nccl_groups()
            return
        logger.warning("[%s] policy.destroy_nccl_groups unavailable", self._pipeline_id)

    def _start_generation_watchdog(self) -> None:
        """Spawn the daemon thread that re-requests GENERATION when the cluster is empty.

        Idempotent: starts at most one thread per pipeline actor lifetime.
        """
        if self._gen_watchdog_thread is not None and self._gen_watchdog_thread.is_alive():
            return
        self._gen_watchdog_stop.clear()
        t = threading.Thread(
            target=self._generation_watchdog_loop,
            name=f"rlix-gen-watchdog-{self._pipeline_id}",
            daemon=True,
        )
        self._gen_watchdog_thread = t
        t.start()
        print(
            f"[RLIX_PPL {self._pipeline_id}] generation watchdog started "
            f"(interval={self._gen_watchdog_interval_s}s)",
            flush=True,
        )

    def _generation_watchdog_loop(self) -> None:
        """Re-request GENERATION whenever the inference cluster has been shrunk to 0.

        Runs in a daemon thread spawned by ``_start_generation_watchdog``. Each
        iteration takes a short snapshot under ``_infer_resize_lock`` to decide
        whether ranks are missing, then drops the lock before issuing the
        re-request (which itself triggers an ``_expand_workers`` callback that
        re-acquires the lock).
        """
        while not self._gen_watchdog_stop.is_set():
            if self._gen_watchdog_stop.wait(self._gen_watchdog_interval_s):
                break

            should_request = False
            with self._infer_resize_lock:
                # ATC must be alive, otherwise there is no demand to satisfy.
                if self._trajectory_collector is None:
                    continue
                # Skip if pipeline is sleeping by design (e.g. during
                # before_training shrink while training holds the GPU).
                # We only re-request when both sets are empty AND there is no
                # in-flight transition. _pre_activation_ranks being non-empty
                # means an expand is still mid-flight and will populate soon.
                if self._active_dp_ranks or self._pre_activation_ranks:
                    continue
                should_request = True

            if not should_request:
                continue

            try:
                allocated = self._request_cluster_gpus(
                    cluster_id=self._actor_infer_cluster_id,
                    priority=Priority.GENERATION,
                    global_step=max(int(self._cache_ready_step), 0),
                    step_target_estimate=1,
                )
                print(
                    f"[RLIX_PPL {self._pipeline_id}] watchdog re-requested GENERATION "
                    f"-> gpus={allocated}, active_dp_ranks={sorted(self._active_dp_ranks)}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — log and keep polling
                logger.warning(
                    "[%s] generation watchdog re-request failed: %s",
                    self._pipeline_id,
                    exc,
                )

    def _publish_weight_version(self) -> int:
        """Publish the cache's weight_version to ATC.

        ``_cache_ready_step`` is the weight_version the cache belongs to:
          - bootstrap (init weights, never trained): ``_BOOTSTRAP_CACHE_VERSION = -1``,
            clamped to ``0`` here so we agree with grpo.py's
            ``set_weight_version(weight_version=step=0)`` (the initial value
            grpo.py writes to ATC at line 2587).
          - after training step N: ``N + 1`` (set in ``_before_weight_sync``;
            cf. debug #36). Publishing ``N`` would make ATC's
            ``_calculate_target_weights`` return targets ``[N..N+max_age]``,
            which is exactly the set already buffered → ATC pauses forever.
        """
        if self._trajectory_collector is None:
            raise RuntimeError("trajectory_collector is required before publishing weight version")
        version = max(int(self._cache_ready_step), 0)
        ray.get(self._trajectory_collector.set_weight_version.remote(version))
        self._current_weight_version = version
        return version

    def _push_active_dp_ranks_to_collector(self) -> None:
        # ATC has its own pickled VllmGeneration; routing decisions read
        # _active_dp_ranks locally. Pipeline-side activate_dp_ranks/sleep_*
        # updates do not propagate, so we mirror the current set onto the
        # collector after every expand/shrink. NoOp when the collector is not
        # yet registered (bootstrap path).
        if self._trajectory_collector is None:
            return
        ranks = sorted(int(r) for r in self._active_dp_ranks)
        try:
            ray.get(self._trajectory_collector.set_active_dp_ranks.remote(ranks))
        except AttributeError:
            logger.warning(
                "[%s] trajectory_collector.set_active_dp_ranks unavailable; "
                "ATC routing may stall (active=%s)",
                self._pipeline_id,
                ranks,
            )

    def _create_model_update_service(self) -> None:
        """Create NemoRLModelUpdateService Ray actor in the pipeline namespace."""
        if self._model_update_service is not None:
            return
        if self._policy is None or self._policy_generation is None:
            raise RuntimeError(
                "policy and policy_generation must be initialized before creating "
                "NemoRLModelUpdateService"
            )
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
            policy=None,
            policy_generation=None,
            policy_workers=list(self._policy.worker_group.workers),
            model_update_receiver=self._policy_generation.get_model_update_receiver(),
        )
        ray.get(svc.__ray_ready__.remote())
        self._model_update_service = svc
        logger.info(
            "[%s] NemoRLModelUpdateService created (name=%s namespace=%s)",
            self._pipeline_id,
            svc_name,
            namespace,
        )
