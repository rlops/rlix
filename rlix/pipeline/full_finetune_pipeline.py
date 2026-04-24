from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, cast

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import (
    AgenticPipeline,
    compute_rollout_traj_metrics,
    compute_train_data_metrics,
)
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
    get_agentic_response_level_mask,
)
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    agg_loss,
    batch_balance,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
)
from roll.utils.logging import get_logger
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch

from rlix.pipeline.utils import validate_resize_params
from rlix.protocol.types import (
    ACTOR_TRAIN_CLUSTER_NAME,
    COORDINATOR_ACTOR_NAME_PREFIX,
    CRITIC_CLUSTER_NAME,
    GENERATION_CLUSTER_NAME,
    REFERENCE_CLUSTER_NAME,
    RLIX_NAMESPACE,
    SCHEDULER_ACTOR_NAME,
    ActionResponse,
    Priority,
    get_pipeline_namespace,
)
from rlix.utils.env import parse_env_timeout_s
from rlix.utils.ray import get_actor_or_raise

logger = get_logger()

_F = TypeVar("_F", bound=Callable[..., Any])

if TYPE_CHECKING:
    def no_grad(func: _F) -> _F: ...
else:
    no_grad = torch.no_grad()


class RollFullFinetunePipeline(AgenticPipeline):  # type: ignore[misc]
    """Rlix-controlled variant of ROLL AgenticPipeline.

    Key differences from upstream AgenticPipeline.run():
    - Before each rollout, request generation GPUs from Rlix (scheduler drives expand via coordinator).
    - After each rollout, shrink actor_infer to zero and release allocation back to Rlix.
    - Validation runs synchronously to avoid racing with shrink/release.
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        # In Rlix mode we should follow the ConcurrentAgenticPipeline semantics:
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        self._pipeline_id = pipeline_id
        self._pipeline_config = pipeline_config
        self._initialized = False
        # Ray actor can run with max_concurrency>1; guard init so resize/run can't race it.
        self._init_lock = threading.Lock()
        self._rlix_scheduler = get_actor_or_raise(
            SCHEDULER_ACTOR_NAME,
            RLIX_NAMESPACE,
            error_context="The pipeline expects the central scheduler actor to be present before startup; "
            "ensure the orchestrator created it earlier or that startup ordering is correct.",
        )
        # FIXME: actor naming is inconsistent. Unify to "{cluster_name}-{pipeline_id}" everywhere.
        # 1. Cluster IDs here use constants; Cluster() constructors use pipeline_config.*.name — can diverge.
        # 2. Separators mixed: "_" for clusters, "-" for schedulers/services.
        self._actor_infer_cluster_id = f"{self._pipeline_id}_{GENERATION_CLUSTER_NAME}"
        self._actor_train_cluster_id = f"{self._pipeline_id}_{ACTOR_TRAIN_CLUSTER_NAME}"
        self._critic_cluster_id = f"{self._pipeline_id}_{CRITIC_CLUSTER_NAME}"
        self._reference_cluster_id = f"{self._pipeline_id}_{REFERENCE_CLUSTER_NAME}"
        # Lazily resolved and cached on first use by _get_coordinator_handle().
        self._coordinator_handle: Any = None
        # Lifecycle tracker for ROLL's CPU bucket cache (Feature 4).
        self._lifecycle: Any = None  # BucketCacheLifecycle, set during initialize_pipeline
        # Version of the last committed base-model checkpoint (= _lifecycle.cache_ready_step).
        self._current_weight_version: Optional[int] = None
        # ModelUpdateService Ray actor handle (Feature 6), set during initialize_pipeline.
        self._model_update_service: Any = None
        # AsyncTrajectoryCollector Ray actor handle for set_weight_version (Feature 6).
        # Injected by the training loop (grpo.py) via set_trajectory_collector().
        self._trajectory_collector: Any = None

    def set_trajectory_collector(self, collector: Any) -> None:
        """Inject the AsyncTrajectoryCollector Ray actor handle (injection path).

        Called by the training loop (grpo.py) after the collector is created.
        The pipeline also lazily resolves the collector by name via
        _get_trajectory_collector() when PIPELINE_ID and ROLL_RAY_NAMESPACE are set.
        Spec: nemorl-port-plan.md lines 490, 538, 603.
        """
        self._trajectory_collector = collector

    def _get_trajectory_collector(self) -> Any:
        """Return the trajectory collector, lazily resolved by named Ray actor if needed."""
        if self._trajectory_collector is not None:
            return self._trajectory_collector
        import os as _os
        pipeline_id = _os.environ.get("PIPELINE_ID", "")
        namespace = _os.environ.get("ROLL_RAY_NAMESPACE", "")
        if not pipeline_id or not namespace:
            return None
        try:
            self._trajectory_collector = ray.get_actor(
                f"rlix:trajectory_collector:{pipeline_id}",
                namespace=namespace,
            )
        except Exception:
            pass
        return self._trajectory_collector

    def _get_coordinator_handle(self) -> Any:
        """Resolve and cache the per-pipeline PipelineCoordinator actor handle.

        Named 'rlix:coordinator:{pipeline_id}' in the pipeline namespace.
        The coordinator serializes resize_infer and sync_lora_weights via _resize_sync_lock.
        """
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

    def initialize_pipeline(self) -> ActionResponse:
        # In RLix mode we should follow the ConcurrentAgenticPipeline semantics:
        """Initialize pipeline clusters/schedulers and prepare selective sync cache before first rollout."""
        with self._init_lock:
            if self._initialized:
                return ActionResponse(success=True)

            # Inline the heavy init logic (based on ConcurrentAgenticPipeline + AgenticPipeline init).
            # Do not call AgenticPipeline.__init__ here: we need explicit ordering + central scheduler interaction.
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            from roll.distributed.executor.cluster import Cluster
            from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
            from roll.models.model_providers import default_tokenizer_provider
            from roll.pipeline.base_pipeline import BasePipeline
            from roll.utils.constants import RAY_NAMESPACE, rlix_env_vars
            from roll.utils.functionals import RunningMoments
            from roll.utils.kl_controller import get_kl_controller

            pipeline_config = self._pipeline_config
            BasePipeline.__init__(self, pipeline_config)
            self.pipeline_config = pipeline_config

            self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
            actor_lora_target = getattr(self.pipeline_config.actor_train.model_args, "lora_target", None)
            self.use_ref_model = bool(self.pipeline_config.enable_reference and (actor_lora_target is None))
            # partial_gpu_mode is ROLL's single-pipeline overlapping GPU mode.
            # In RLix mode, shrink/expand is handled by the central scheduler — always disabled.
            self.partial_gpu_mode = False

            self.kl_ctrl = get_kl_controller(
                init_kl_coef=self.pipeline_config.init_kl_coef,
                target_kl=self.pipeline_config.target_kl,
                kl_horizon=self.pipeline_config.kl_horizon,
            )

            # INIT PHASE: Create clusters (use pipeline_id prefix to keep names readable in logs).
            self.actor_train = Cluster(
                name=f"{self._pipeline_id}_{self.pipeline_config.actor_train.name}",
                worker_cls=self.pipeline_config.actor_train.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.actor_train,
            )
            self.actor_infer = Cluster(
                name=f"{self._pipeline_id}_{self.pipeline_config.actor_infer.name}",
                worker_cls=self.pipeline_config.actor_infer.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.actor_infer,
            )

            download_clusters = [self.actor_train, self.actor_infer]

            if self.use_ref_model:
                self.reference = Cluster(
                    name=f"{self._pipeline_id}_{self.pipeline_config.reference.name}",
                    worker_cls=self.pipeline_config.reference.worker_cls,
                    resource_manager=self.resource_manager,
                    worker_config=self.pipeline_config.reference,
                )
                download_clusters.append(self.reference)

            if self.pipeline_config.adv_estimator == "gae":
                self.critic = Cluster(
                    name=f"{self._pipeline_id}_{self.pipeline_config.critic.name}",
                    worker_cls=self.pipeline_config.critic.worker_cls,
                    resource_manager=self.resource_manager,
                    worker_config=self.pipeline_config.critic,
                )
                download_clusters.append(self.critic)

            # Reward cluster is optional; keep consistent with AgenticPipeline behavior.
            self.reward = None
            self.reward_scheduler = None
            if self.pipeline_config.reward is not None and len(self.pipeline_config.reward.device_mapping) > 0:
                self.reward = Cluster(
                    name=f"{self._pipeline_id}_{self.pipeline_config.reward.name}",
                    worker_cls=self.pipeline_config.reward.worker_cls,
                    resource_manager=self.resource_manager,
                    worker_config=self.pipeline_config.reward,
                )
                download_clusters.append(self.reward)

            # INIT PHASE: Download models once per node/PG before strategy initialization.
            self.download_models(*download_clusters)
            self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

            # Reward scheduler (named actor for env managers) if reward cluster exists.
            if self.reward:
                from roll.distributed.scheduler.generate_scheduler import RequestScheduler

                reward_name = f"RewardScheduler-{self._pipeline_id}"
                self.reward_scheduler = RequestScheduler.options(
                    name=reward_name,
                    get_if_exists=True,
                    namespace=RAY_NAMESPACE,
                    runtime_env={"env_vars": rlix_env_vars()},
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False,
                    ),
                ).remote(
                    infer_cluster=self.reward,
                    pipeline_config=self.pipeline_config,
                    resource_manager=self.resource_manager,
                )

            # Rollout schedulers (named actors).
            self.train_rollout_scheduler = (
                ray.remote(RolloutScheduler)
                .options(
                    name=f"RolloutScheduler-{self._pipeline_id}-train",
                    namespace=RAY_NAMESPACE,
                    runtime_env={"env_vars": rlix_env_vars()},
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False,
                    ),
                )
                .remote(
                    config=self.pipeline_config,
                    env_manager_config=self.pipeline_config.train_env_manager,
                    resource_manager=self.resource_manager,
                    infer_cluster=self.actor_infer,
                    mode="train",
                )
            )
            self.val_rollout_scheduler = (
                ray.remote(RolloutScheduler)
                .options(
                    name=f"RolloutScheduler-{self._pipeline_id}-val",
                    namespace=RAY_NAMESPACE,
                    runtime_env={"env_vars": rlix_env_vars()},
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False,
                    ),
                )
                .remote(
                    config=self.pipeline_config,
                    env_manager_config=self.pipeline_config.val_env_manager,
                    resource_manager=self.resource_manager,
                    infer_cluster=self.actor_infer,
                    mode="val",
                )
            )

            # Create val dataset manager as in AgenticPipeline.
            from roll.datasets.global_dataset import GlobalDatasetManager

            self.val_dataset_manager = GlobalDatasetManager.options(
                name="val_dataset_manager",
                get_if_exists=True,
                namespace=RAY_NAMESPACE,
                runtime_env={"env_vars": rlix_env_vars()},
            ).remote()

            self._infer_resize_lock = threading.Lock()

            # INIT PHASE: Initialize clusters with central scheduler coordination and strict offload ordering.
            from rlix.protocol.types import Priority

            init_global_step = -1
            self._request_cluster_gpus(
                cluster_id=self._actor_train_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_global_step,
            )
            try:
                self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)

                # Build and promote the initial base-model cache (-1/-1) before offload.
                # Under sleep_level=2 this cache must stay active so expand can rehydrate infer workers.
                # Megatron-only: DeepSpeed strategies do not implement bucket cache / checkpoint promotion.
                init_checkpoint_version = -1
                self.actor_train.load_states(blocking=True)
                try:
                    ray.get(
                        [
                            w.build_latest_bucket_cache.remote(
                                checkpoint_version=int(init_checkpoint_version),
                            )
                            for w in self.actor_train.workers
                        ]
                    )
                    ray.get(
                        [
                            w.promote_active_checkpoint.remote(
                                version=int(init_checkpoint_version),
                            )
                            for w in self.actor_train.workers
                        ]
                    )
                except RuntimeError as e:
                    if "does not support" in str(e):
                        logger.info("[init][%s] skipping bucket cache/checkpoint promotion: %s", self._pipeline_id, e)
                    else:
                        raise

                # Offload training-side clusters before initializing actor_infer (avoid transient OOM).
                logger.info("[init][%s] offloading actor_train before actor_infer init", self._pipeline_id)
                self.actor_train.offload_states(blocking=True)
                logger.info("[init][%s] actor_train offload done", self._pipeline_id)
            finally:
                self._notify_release_cluster_gpus(
                    cluster_id=self._actor_train_cluster_id, global_step=init_global_step
                )
                logger.info("[init][%s] released actor_train cluster", self._pipeline_id)

            logger.info("[init][%s] requesting actor_infer cluster (INITIALIZATION)", self._pipeline_id)
            self._request_cluster_gpus(
                cluster_id=self._actor_infer_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_global_step,
            )
            logger.info("[init][%s] actor_infer cluster granted — starting init", self._pipeline_id)
            try:
                if self.reward:
                    self.reward.initialize(pipeline_config=self.pipeline_config, blocking=True)
                self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
                logger.info(
                    "[init][%s] actor_infer initialized — offloading (sleep_level=2: destroy weights+KV)",
                    self._pipeline_id,
                )
                if self.reward:
                    self.reward.offload_states(blocking=True)
                self.actor_infer.offload_states(blocking=True)
                logger.info("[init][%s] actor_infer offload done — GPU memory freed", self._pipeline_id)
            finally:
                self._notify_release_cluster_gpus(
                    cluster_id=self._actor_infer_cluster_id, global_step=init_global_step
                )
                logger.info("[init][%s] released actor_infer cluster", self._pipeline_id)

            if self.pipeline_config.adv_estimator == "gae":
                self._request_cluster_gpus(
                    cluster_id=self._critic_cluster_id,
                    priority=Priority.INITIALIZATION,
                    global_step=init_global_step,
                )
                try:
                    self.critic.initialize(pipeline_config=self.pipeline_config, blocking=True)
                    self.critic.offload_states(blocking=True)
                finally:
                    self._notify_release_cluster_gpus(cluster_id=self._critic_cluster_id, global_step=init_global_step)

            if self.use_ref_model:
                self._request_cluster_gpus(
                    cluster_id=self._reference_cluster_id,
                    priority=Priority.INITIALIZATION,
                    global_step=init_global_step,
                )
                try:
                    self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)
                    self.reference.offload_states(blocking=True)
                finally:
                    self._notify_release_cluster_gpus(
                        cluster_id=self._reference_cluster_id, global_step=init_global_step
                    )

            # RLix mode: skip set_model_update_pair (old ROLL broadcast-based sync).
            # Weight sync is handled by ModelUpdateService.sync_selected_workers (selective sync).
            # Creating the old ModelUpdateGroup would set up a persistent NCCL broadcast group
            # that interferes with the selective sync path.
            if self.pipeline_config.adv_estimator == "gae":
                self.set_checkpoint_clusters(self.actor_train, self.critic)
            else:
                self.set_checkpoint_clusters(self.actor_train)

            self.running = RunningMoments()

            # Namespace contract: in Rlix mode, require explicit per-pipeline env vars (fail fast).
            ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE", "roll")
            if os.environ.get("RLIX_CONTROL_PLANE", "") == "rlix":
                env_namespace = os.environ.get("ROLL_RAY_NAMESPACE")
                pipeline_id_env = os.environ.get("PIPELINE_ID")
                if not env_namespace:
                    raise RuntimeError("RLIX_CONTROL_PLANE=rlix requires ROLL_RAY_NAMESPACE to be set")
                if not pipeline_id_env:
                    raise RuntimeError("RLIX_CONTROL_PLANE=rlix requires PIPELINE_ID to be set")
                if pipeline_id_env != self._pipeline_id:
                    raise RuntimeError(
                        f"PIPELINE_ID mismatch for coordinator: env PIPELINE_ID={pipeline_id_env!r} "
                        f"!= coordinator pipeline_id={self._pipeline_id!r}"
                    )
                ray_namespace = env_namespace

            # Align with ConcurrentAgenticPipeline: interact with central scheduler during init.
            # The initial (-1) cache bucket is built during actor_train init above under INITIALIZATION allocation.

            # Create ModelUpdateService in the per-pipeline namespace. This is used by
            # RequestScheduler.expand_workers() in Rlix mode to sync selected dp ranks after load.
            from rlix.pipeline.model_update_service import ModelUpdateService

            from rlix.utils.env import pipeline_identity_env_vars
            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    **pipeline_identity_env_vars(
                        pipeline_id=os.environ.get("PIPELINE_ID", self._pipeline_id),
                        ray_namespace=ray_namespace,
                    ),
                }
            }
            svc = ModelUpdateService.options(  # type: ignore[attr-defined]
                name=f"{self._pipeline_id}_model_update_service",
                namespace=ray_namespace,
                get_if_exists=True,
                max_restarts=0,
                max_task_retries=0,
                runtime_env=runtime_env,
                lifetime="detached",
            ).remote(
                pipeline_id=self._pipeline_id,
                src_cluster=self.actor_train,
                tgt_cluster=self.actor_infer,
                model_update_transport=os.environ.get("RLIX_MODEL_UPDATE_TRANSPORT", "cpu_serialize"),
                bucket_size_bytes=int(os.environ["RLIX_BUCKET_SIZE_BYTES"]) if os.environ.get("RLIX_BUCKET_SIZE_BYTES") else None,
            )
            # Block until actor init completes.
            ray.get(svc.__ray_ready__.remote())
            self._model_update_service = svc
            # Start from a well-defined state:
            # - disable routing until we request GPUs from RLix.
            # NOTE: avoid local suspend()/resume() state transitions; shrink-to-zero is the single
            # source of truth for pausing generation traffic, and expand-from-zero resumes internally.
            dp_ranks = list(range(self.actor_infer.dp_size))
            ray.get(self.train_rollout_scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))
            ray.get(self.val_rollout_scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))

            # Feature 4: create lifecycle tracker. The initial base-model cache (version=-1)
            # was already built and promoted above (before actor_infer init). Record the
            # version in the lifecycle without re-calling workers.
            from rlix.pipeline.bucket_cache_lifecycle import BucketCacheLifecycle

            self._lifecycle = BucketCacheLifecycle(
                pipeline_id=self._pipeline_id,
                workers=list(self.actor_train.workers),
            )
            self._lifecycle.mark_promoted(BucketCacheLifecycle._BASE_VERSION)
            self._current_weight_version = self._lifecycle.cache_ready_step
            _tc = self._get_trajectory_collector()
            if _tc is not None:
                ray.get(_tc.set_weight_version.remote(self._current_weight_version))

            self._initialized = True
            return ActionResponse(success=True)

    def _shrink_workers(self, *, dp_ranks_to_remove: List[int]) -> Dict[str, Any]:
        """Pipeline-local shrink helper.

        Val scheduler does routing-only shrink; train scheduler does routing + physical offload.
        """
        if not isinstance(dp_ranks_to_remove, list) or not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be a non-empty list[int]")
        with self._infer_resize_lock:
            # Val: routing-only (skip_offload=True) — shared infer cluster, no physical offload.
            ray.get(self.val_rollout_scheduler.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=True))
            # Train: routing + physical offload (skip_offload=False).
            return cast(
                Dict[str, Any],
                ray.get(self.train_rollout_scheduler.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=False)),
            )

    def _expand_workers(self, *, dp_ranks_to_add: List[int]) -> Dict[str, Any]:
        """Pipeline-local expand helper.

        Atomic expand sequence (spec: nemorl-port-plan.md lines 589-609):
          1. Wake overlap ranks (skip_load=True — weights come from CPU bucket cache, not ROLL load).
          2. Sync weights from CPU bucket cache via ModelUpdateService (Feature 6 path).
          3. Val scheduler routing update (skip_load=True always).
          4. Publish _current_weight_version so newly-woken workers are consistent.
        """
        if not isinstance(dp_ranks_to_add, list) or not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be a non-empty list[int]")
        with self._infer_resize_lock:
            # Step 1: Sync weights from CPU bucket cache to the woken workers BEFORE
            # routing is enabled.  Workers are Ray actors that accept remote calls even
            # while shrunk; syncing here ensures weights land before rebalance_on_expand
            # adds the ranks to active_dp_ranks (spec: nemorl-port-plan.md lines 589-609).
            if hasattr(self, "_model_update_service") and self._model_update_service is not None:
                ray.get(
                    self._model_update_service.sync_selected_workers.remote(
                        tgt_dp_ranks=dp_ranks_to_add,
                    )
                )

            # Step 1b: finalize_weight_update — pipeline-owned per spec line 624-632.
            # Must run after all buckets land (sync_selected_workers returned) and before
            # routing is activated so inference workers are fully ready.
            finalize_refs = [
                self.actor_infer.rank2worker[int(r)].finalize_weight_update.remote()
                for r in dp_ranks_to_add
            ]
            ray.get(finalize_refs)

            # Step 2: Publish version BEFORE activating routing.
            # Spec (nemorl-port-plan.md lines 602-608): version must be published before
            # activate_dp_ranks so the collector sees the correct weight version as soon
            # as newly expanded ranks start serving requests.
            if self._lifecycle is not None:
                self._current_weight_version = self._lifecycle.cache_ready_step
                _tc = self._get_trajectory_collector()
                if _tc is not None:
                    ray.get(_tc.set_weight_version.remote(self._current_weight_version))

            # Step 3: Activate routing AFTER version is published.
            # skip_load=True — weights already synced in step 1.
            result = ray.get(self.train_rollout_scheduler.expand_sampler.remote(dp_ranks_to_add, skip_load=True))
            ray.get(self.val_rollout_scheduler.expand_sampler.remote(dp_ranks_to_add, skip_load=True))
            return cast(Dict[str, Any], result)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            resp = self.initialize_pipeline()
            if not getattr(resp, "success", False):
                raise RuntimeError(f"initialize_pipeline failed: {resp}")

    def _request_cluster_gpus(
        self,
        *,
        cluster_id: str,
        priority: Any,
        global_step: int,
        step_target_estimate: Optional[int] = None,
        lora_name: Optional[str] = None,
    ) -> List[int]:
        """Block until the scheduler allocates GPUs for a cluster. Returns allocated GPU IDs."""
        allocated = ray.get(
            self._rlix_scheduler.request_gpus.remote(
                cluster_id=str(cluster_id),
                priority=priority,
                global_step=global_step,
                step_target_estimate=step_target_estimate,
                lora_name=lora_name,  # GPU tracing: pass LoRA name for training clusters
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(f"rlix:scheduler.request_gpus returned non-list: {type(allocated).__name__}")
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(f"rlix:scheduler allocated empty GPU list for cluster_id={cluster_id!r}")
        return allocated

    def _notify_release_cluster_gpus(self, *, cluster_id: str, global_step: int) -> None:
        """Notify the scheduler that a cluster's GPUs are released back to the idle pool."""
        ray.get(self._rlix_scheduler.notify_release_gpus.remote(cluster_id=str(cluster_id), global_step=global_step))

    def _notify_release_then_request_cluster_gpus(
        self,
        *,
        release_cluster_id: str,
        release_global_step: int,
        request_cluster_id: str,
        request_priority: Any,
        request_global_step: int,
        request_step_target_estimate: Optional[int] = None,
        request_lora_name: Optional[str] = None,
    ) -> List[int]:
        """Atomically release one cluster's GPUs and block until another cluster is allocated."""
        allocated = ray.get(
            self._rlix_scheduler.notify_release_then_request_gpus.remote(
                release_cluster_id=str(release_cluster_id),
                release_global_step=int(release_global_step),
                request_cluster_id=str(request_cluster_id),
                request_priority=request_priority,
                request_global_step=int(request_global_step),
                request_step_target_estimate=request_step_target_estimate,
                request_lora_name=request_lora_name,  # GPU tracing: pass LoRA name for training clusters
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(
                f"rlix:scheduler.notify_release_then_request_gpus returned non-list: {type(allocated).__name__}"
            )
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(f"rlix:scheduler allocated empty GPU list for cluster_id={request_cluster_id!r}")
        return allocated

    def _generation_num_return_sequences(self) -> int:
        """Return the rollout scheduler's effective num_return_sequences."""
        raw = getattr(self.pipeline_config.actor_infer.generating_args, "num_return_sequences", None)
        n = 1 if raw is None else int(raw)
        if n <= 0:
            raise RuntimeError(f"Invalid num_return_sequences={raw!r}; expected > 0")
        return n

    def _estimate_generation_step_target(self, *, train_batch_size: int, include_val: bool) -> int:
        """Estimate total trajectory demand for a held GENERATION allocation."""
        num_return_sequences = self._generation_num_return_sequences()
        total = int(train_batch_size) * num_return_sequences
        if include_val:
            total += int(self.pipeline_config.val_batch_size) * num_return_sequences
        return total

    def _await_release_actor_infer(self, *, global_step: int) -> None:
        """Block until the scheduler commits the actor_infer shrink for this pipeline."""
        timeout_s = parse_env_timeout_s("RLIX_NOTIFY_READY_TIMEOUT_S", 300.0)

        ray.get(
            self._rlix_scheduler.await_release_gpus.remote(
                cluster_id=self._actor_infer_cluster_id,
                global_step=global_step,
                timeout_s=timeout_s,
            )
        )
        logger.info(f"[rlix][{self._pipeline_id}] await_release_gpus done: step={global_step}")

    def val(self, global_step: int) -> dict[str, Any]:
        """Validation with bounded timeouts and no trajectory dump.

        Overrides AgenticPipeline.val() to:
        1. Bound ray.get() calls with self._rollout_get_batch_timeout_s (set by run()).
        2. Skip dump_rollout_trajectories to avoid race with concurrent train rollout dump.
        """
        from roll.pipeline.agentic.agentic_pipeline import get_episode_scores

        batch = DataProto()
        metrics = {}
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["global_step"] = global_step
        timeout_s = getattr(self, "_rollout_get_batch_timeout_s", None)
        ray.get(self.val_dataset_manager.reset.remote(), timeout=timeout_s)
        eval_batch = ray.get(
            self.val_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.val_batch_size),
            timeout=timeout_s,
        )

        if "get_batch_return_start_time" in eval_batch.meta_info:
            metrics["time/get_batch_cost_val"] = time.time() - eval_batch.meta_info.pop("get_batch_return_start_time")

        # Intentionally skip dump_rollout_trajectories: val runs concurrently with train rollout
        # on the same GENERATION allocation, and concurrent dumps for the same global_step race.
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()

        batch_grouped = eval_batch.group_by(keys="tags")
        for group_name, group_batch in batch_grouped.items():
            traj_group_scores = []
            batch_traj_grouped = group_batch.group_by(keys="traj_group_id")
            for batch_traj_group_name, batch_traj_group in batch_traj_grouped.items():
                traj_group_score = get_episode_scores(batch_traj_group)
                traj_group_scores.append(traj_group_score.mean().item())
            eval_score = torch.tensor(traj_group_scores, dtype=torch.float)
            eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
        logger.info(f"val_batch_size: {len(eval_batch)}")
        logger.info(f"val metrics: {metrics}")

        return metrics

    @no_grad
    def run(self) -> None:
        """RLix-controlled training loop aligned with agentic pipeline.

        Implements individual blocking GPU cycles with request -> execute -> release
        pattern for each cluster. Key design choices:
        - Overlapped val + train rollout on a single GENERATION allocation.
        - sample_uuids, batch_balance, compute_train_data_metrics, TPS metrics, entropy.
        - Unconditional state persistence (state.step/log_history/do_checkpoint) even during
          critic warmup steps.
        - No reference log probs phase: ref_log_probs = old_log_probs.clone() (KL penalty = 0).
        - No decoded trajectory logging (tokenizer-decode + JSON log block).
        - No val trajectory dump (val() override skips dump_rollout_trajectories).

        GPU allocation state machine per step (priority in parens, lower = higher):
            Phase 4.5:  request actor_infer (GENERATION=6)
            Phase 11:   request critic (VALUE_COMPUTE=5)          [GAE only]
            Phase 13:   release critic → request actor_train (OLD_LOG_PROBS=3)  [GAE]
                        request actor_train (OLD_LOG_PROBS=3)     [non-GAE]
            Phase 15:   release actor_train → request critic (CRITIC_TRAINING=2)  [GAE]
              warmup:   release critic immediately
            Phase 16:   release critic → request actor_train (ACTOR_TRAINING=1)  [GAE, non-warmup]
                        release actor_train → request actor_train (ACTOR_TRAINING=1)  [non-GAE]
            Next step Phase 1: await_release actor_infer
            Next step Phase 4.5: release actor_train → request actor_infer

        Invariants:
        - At most 2 clusters allocated at any time (actor_infer + one train/critic).
        - Every release_then_request has distinct priorities (scheduler rejects same-priority).
        - Warmup steps release all train clusters; Phase 4.5 detects this via
          ``critic_warmup <= (global_step - 1)`` to avoid releasing an unallocated cluster.
        """
        self._ensure_initialized()
        logger.info("Starting reorganized concurrent agentic pipeline")

        rollout_get_batch_timeout_s = parse_env_timeout_s("ROLL_ROLLOUT_GET_BATCH_TIMEOUT_S", default_s=None)
        self._rollout_get_batch_timeout_s = rollout_get_batch_timeout_s

        tps_timer = _Timer(window_size=5)

        ran_any_step = False
        last_train_cluster_allocated = None

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue

            ran_any_step = True
            metrics = {}

            logger.info(f"pipeline {self._pipeline_id} rollout global step {global_step} start...")

            with Timer(name="pipeline_step_total", logger=None) as step_timer:
                with tps_timer:

                    # ============================================================
                    # RLix Phase 1: Notify release (replaces agentic Phases 1-2)
                    # ============================================================
                    if global_step > 0:
                        self._await_release_actor_infer(global_step=global_step - 1)
                        logger.info(f"run() {self._pipeline_id=} Phase 1: Notified scheduler")

                    # ============================================================
                    # RLix Phase 4.5: Request generation GPUs (replaces agentic Phases 3-5)
                    # ============================================================
                    allocated_actor_infer_gpus = None
                    actor_infer_num_gpus = len(getattr(self.actor_infer.worker_config, "device_mapping", []))
                    assert actor_infer_num_gpus > 0
                    expected_gpus = list(self.actor_infer.worker_config.device_mapping)
                    eval_this_step = (
                        self.pipeline_config.eval_steps > 0 and global_step % self.pipeline_config.eval_steps == 0
                    )
                    generation_step_target_estimate = self._estimate_generation_step_target(
                        train_batch_size=self.pipeline_config.rollout_batch_size,
                        include_val=bool(eval_this_step),
                    )
                    # actor_train GPUs are released immediately at end of each training step (Feature 4/5/6),
                    # so there is never a deferred release to perform here — always use plain request.
                    allocated_actor_infer_gpus = self._request_cluster_gpus(
                        cluster_id=self._actor_infer_cluster_id,
                        priority=Priority.GENERATION,
                        global_step=global_step,
                        step_target_estimate=generation_step_target_estimate,
                    )
                    assert len(allocated_actor_infer_gpus) > 0
                    is_partial_allocation = len(allocated_actor_infer_gpus) < len(expected_gpus)
                    logger.info(
                        f"run() {self._pipeline_id=} Phase 4.5: Actor infer GPU allocation completed - "
                        f"expected={expected_gpus}, allocated={allocated_actor_infer_gpus}, "
                        f"is_partial_allocation={is_partial_allocation}"
                    )
                    if is_partial_allocation:
                        logger.warning(
                            f"run() {self._pipeline_id=} Phase 4.5: PARTIAL allocation detected for actor_infer - "
                            f"got {len(allocated_actor_infer_gpus)}/{len(expected_gpus)} GPUs. "
                            f"Missing GPUs: {set(expected_gpus) - set(allocated_actor_infer_gpus)}"
                        )
                    assert (
                        len(allocated_actor_infer_gpus) != 0
                    ), "shall not be empty for sched logic as we just released all gpus"

                    # ============================================================
                    # Phase 6/7/8: Overlapped validation + train rollout
                    # One GENERATION request per step. Both run on the same held allocation.
                    # ============================================================
                    batch: DataProto = DataProto()
                    batch.meta_info = {"global_step": global_step}

                    val_future = None
                    val_metrics = {}
                    with Timer(name="val", logger=None) as val_timer:
                        if eval_this_step:
                            val_future = self.executor.submit(self.val, global_step)

                        # Train rollout runs immediately on the same held actor_infer allocation
                        try:
                            with Timer(name="rollout", logger=None) as rollout_timer:
                                batch = ray.get(
                                    self.train_rollout_scheduler.get_batch.remote(
                                        batch, self.pipeline_config.rollout_batch_size
                                    ),
                                    timeout=rollout_get_batch_timeout_s,
                                )

                            # Train rollout bookkeeping (outside rollout timer, inside step)
                            sample_uuids = [
                                f"{traj_id}_{i}" for i, traj_id in enumerate(batch.non_tensor_batch["traj_id"])
                            ]
                            batch.non_tensor_batch["sample_uuid"] = np.array(sample_uuids, dtype=object)
                            if "get_batch_return_start_time" in batch.meta_info:
                                metrics["time/get_batch_cost_train"] = time.time() - batch.meta_info.pop(
                                    "get_batch_return_start_time"
                                )
                            actor_infer_metrics = self.actor_infer.get_metrics()
                            metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))
                            metrics.update(compute_rollout_traj_metrics(batch))

                            dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                        finally:
                            # Always join val before unwinding, even on rollout failure
                            if val_future is not None:
                                try:
                                    val_metrics = val_future.result()
                                except Exception:
                                    logger.warning(
                                        f"run() {self._pipeline_id=}: val() failed during rollout error unwind"
                                    )

                    metrics["time/step_rollout"] = rollout_timer.last
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    batch.meta_info["global_step"] = global_step
                    batch.meta_info["_broadcast_non_tensor_batch"] = True
                    batch.meta_info["loss_mask_keys"] = ["response_mask"]

                    # Critical barrier: join val before any GPU release/reallocation
                    if val_metrics:
                        metrics.update(val_metrics)
                        metrics["time/step_val"] = val_timer.last

                    # ============================================================
                    # Phase 10: Batch Processing (CPU)
                    # ============================================================
                    batch = compute_discounted_returns(
                        batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma
                    )
                    batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # ============================================================
                    # RLix Phase 11: Critic value compute GPU cycle (if GAE)
                    # Extracted from agentic Phase 12 into its own GPU cycle.
                    # ============================================================
                    if self.pipeline_config.adv_estimator == "gae":
                        # Plain request while actor_infer is held: the scheduler triggers
                        # resize_infer (via coordinator) to free shared GPUs if needed.
                        # Pipeline actor max_concurrency > 1 allows resize_infer to run concurrently.
                        self._request_cluster_gpus(
                            cluster_id=self._critic_cluster_id,
                            priority=Priority.VALUE_COMPUTE,
                            global_step=global_step,
                        )
                        values_refs = self.critic.compute_values(batch, blocking=False)
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))

                    # ============================================================
                    # RLix Phase 13: Old Log Probs GPU request
                    # ============================================================
                    if self.pipeline_config.adv_estimator != "gae":
                        self._request_cluster_gpus(
                            cluster_id=self._actor_train_cluster_id,
                            priority=Priority.OLD_LOG_PROBS,
                            global_step=global_step,
                        )
                    else:
                        self._notify_release_then_request_cluster_gpus(
                            release_cluster_id=self._critic_cluster_id,
                            release_global_step=global_step,
                            request_cluster_id=self._actor_train_cluster_id,
                            request_priority=Priority.OLD_LOG_PROBS,
                            request_global_step=global_step,
                        )

                    # Phase 12: Old Log Probs computation
                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        if self.pipeline_config.enable_reference and not self.use_ref_model:
                            batch.meta_info["disable_adapter"] = False
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.enable_old_logprobs_recompute:
                            batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                    self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "pipeline_model_parallel_size", 1
                                    ),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "virtual_pipeline_model_parallel_size", None
                                    ),
                                    "actor_train/compute_log_probs",
                                )
                                metrics.update(dynamic_batching_metrics)
                            old_log_probs: DataProto = self.actor_train.compute_log_probs(batch, blocking=True)
                            batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                            avg_old_log_prob = masked_mean(
                                batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:]
                            )
                            metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})
                            metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                            agg_entropy = agg_loss(
                                loss_mat=old_log_probs.batch["entropy"],
                                loss_mask=batch.batch["response_mask"][:, 1:],
                                loss_agg_mode="token-mean",
                            )
                            metrics.update({"critic/entropy/mean": agg_entropy.item()})
                        else:
                            batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                        # No reference log probs phase: use old_log_probs as ref unconditionally.
                        # .clone() makes KL divergence zero, effectively disabling the penalty.
                        ref_log_probs = batch.batch["old_log_probs"].clone()
                        batch.batch["ref_log_probs"] = ref_log_probs
                        avg_ref_log_prob = masked_mean(
                            batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:]
                        )
                        metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})

                    metrics["time/step_old_log_probs_values"] = cal_old_logpb_timer.last

                    # Response level mask
                    with Timer(name="cal_response_level_mask", logger=None) as timer:
                        batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
                        metrics.update(mask_metrics)
                    metrics["time/step_cal_response_level_mask"] = timer.last

                    # ============================================================
                    # Phase 13b: Advantage Computation (CPU)
                    # ============================================================
                    with Timer(name="cal_response_norm_rewards", logger=None) as timer:
                        batch, reward_metrics = compute_response_level_rewards(
                            batch=batch, pipeline_config=self.pipeline_config
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        metrics.update(reward_metrics)
                    metrics["time/step_cal_norm_rewards"] = timer.last

                    with Timer(name="cal_token_reward", logger=None) as timer:
                        batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
                        metrics.update(token_level_metrics)
                    metrics["time/step_cal_token_reward"] = timer.last

                    with Timer(name="compute_advantage", logger=None) as timer:
                        batch = agentic_compute_advantage(
                            data=batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    metrics["time/step_adv"] = timer.last

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        batch, corr_metrics = apply_train_infer_correction_to_batch(
                            self.pipeline_config,
                            batch,
                            update_mask_keys=batch.meta_info["loss_mask_keys"],
                        )
                        metrics.update(corr_metrics)

                    # ============================================================
                    # RLix Phase 15: Critic Training GPU cycle (if GAE)
                    # ============================================================
                    if self.pipeline_config.adv_estimator == "gae":
                        # Release actor_train (OLD_LOG_PROBS) → request critic (CRITIC_TRAINING)
                        self._notify_release_then_request_cluster_gpus(
                            release_cluster_id=self._actor_train_cluster_id,
                            release_global_step=global_step,
                            request_cluster_id=self._critic_cluster_id,
                            request_priority=Priority.CRITIC_TRAINING,
                            request_global_step=global_step,
                        )

                        with Timer(name="critic_train_step", logger=None) as critic_train_timer:
                            critic_train_metrics = DataProto.materialize_concat(
                                data_refs=self.critic.train_step(batch, blocking=False),
                            )
                            metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                        metrics["time/critic_train_step"] = critic_train_timer.last

                        # Warmup: release critic now (no actor training this step).
                        # Non-warmup: critic stays allocated — Phase 16 releases it.
                        if self.pipeline_config.critic_warmup > global_step:
                            self._notify_release_cluster_gpus(
                                cluster_id=self._critic_cluster_id, global_step=global_step
                            )

                    # ============================================================
                    # RLix Phase 16: Actor Training GPU cycle
                    # ============================================================
                    if self.pipeline_config.critic_warmup <= global_step:
                        if self.pipeline_config.adv_estimator == "gae":
                            # Release critic (CRITIC_TRAINING) → request actor_train (ACTOR_TRAINING)
                            self._notify_release_then_request_cluster_gpus(
                                release_cluster_id=self._critic_cluster_id,
                                release_global_step=global_step,
                                request_cluster_id=self._actor_train_cluster_id,
                                request_priority=Priority.ACTOR_TRAINING,
                                request_global_step=global_step,
                            )
                        else:
                            # Same cluster, priority transition: OLD_LOG_PROBS → ACTOR_TRAINING
                            self._notify_release_then_request_cluster_gpus(
                                release_cluster_id=self._actor_train_cluster_id,
                                release_global_step=global_step,
                                request_cluster_id=self._actor_train_cluster_id,
                                request_priority=Priority.ACTOR_TRAINING,
                                request_global_step=global_step,
                            )

                        with Timer(name="actor_train_step", logger=None) as actor_train_timer:
                            batch_balance_metrics = batch_balance(
                                batch,
                                dp_size=self.actor_train.dp_size,
                                minibatch_size=self.actor_train.dp_size
                                * self.pipeline_config.actor_train.training_args.per_device_train_batch_size
                                * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps,
                                logging_prefix="global_seqlen/actor_train",
                            )
                            metrics.update(batch_balance_metrics)
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                    self.pipeline_config.actor_train.sequence_length_round_in_train,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "pipeline_model_parallel_size", 1
                                    ),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "virtual_pipeline_model_parallel_size", None
                                    ),
                                    "actor_train/train_step",
                                )
                                metrics.update(dynamic_batching_metrics)
                            batch.meta_info["checkpoint_version"] = global_step
                            actor_train_metrics = DataProto.materialize_concat(
                                data_refs=self.actor_train.train_step(batch, blocking=False),
                            )
                            metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
                        metrics["time/train_step"] = actor_train_timer.last

                        # Feature 4: build CPU bucket cache, then promote to active.
                        # Build must precede promote (spec: nemorl-port-plan.md:332-338).
                        # Megatron-only: DeepSpeed strategies do not implement these methods.
                        checkpoint_version = int(batch.meta_info.get("checkpoint_version", global_step))
                        try:
                            ray.get(
                                [
                                    worker.build_latest_bucket_cache.remote(checkpoint_version)
                                    for worker in self.actor_train.workers
                                ]
                            )
                            ray.get(
                                [
                                    worker.promote_active_checkpoint.remote(checkpoint_version)
                                    for worker in self.actor_train.workers
                                ]
                            )
                            assert self._lifecycle is not None
                            self._lifecycle.mark_promoted(checkpoint_version)
                        except RuntimeError as e:
                            if "does not support" in str(e):
                                logger.info("[train][%s] skipping bucket cache build/promote: %s", self._pipeline_id, e)
                            else:
                                raise

                        # Offload training weights to CPU before syncing to active infer workers.
                        self.actor_train.offload_states(blocking=True)

                        # Feature 5/6: sync base weights to all currently-active infer dp ranks.
                        # sync_selected_workers handles transport; finalize is pipeline-owned (spec line 624).
                        # Coordinator returns the exact ranks that were synced (may be [] if all sleeping).
                        coordinator = self._get_coordinator_handle()
                        synced_ranks: List[int] = ray.get(coordinator.sync_base_weights_to_active.remote())

                        # finalize_weight_update: pipeline-owned, only for the synced ranks (spec line 488-490).
                        if synced_ranks:
                            finalize_refs = [
                                self.actor_infer.rank2worker[int(r)].finalize_weight_update.remote()
                                for r in synced_ranks
                            ]
                            ray.get(finalize_refs)

                        # Publish version after sync+finalize completes.
                        self._current_weight_version = self._lifecycle.cache_ready_step
                        _tc = self._get_trajectory_collector()
                        if _tc is not None:
                            ray.get(_tc.set_weight_version.remote(self._current_weight_version))
                        # Spec: nemorl-port-plan.md lines 489-490, 536-538.

                        # Release actor_train GPUs immediately (not deferred to next step).
                        self._notify_release_cluster_gpus(
                            cluster_id=self._actor_train_cluster_id,
                            global_step=global_step,
                        )
                        last_train_cluster_allocated = None
                    else:
                        # Warmup: Phase 15 released actor_train → critic, then critic was released above.
                        # No train cluster remains allocated.
                        last_train_cluster_allocated = None

                # Unconditional — runs even during critic warmup
                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

                with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                    data_metrics = compute_train_data_metrics(batch=batch)
                metrics["time/step_compute_data_metrics"] = data_metrics_timer.last
                metrics.update(data_metrics)
                metrics["system/tps"] = tps_timer.mean_throughput
                metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

                self.state.step = global_step
                self.state.log_history.append(metrics)
                if self.pipeline_config.critic_warmup <= global_step:
                    self.do_checkpoint(global_step=global_step, offload_after_checkpoint=True)
                else:
                    self.do_checkpoint(global_step=global_step)

            metrics["time/step_total"] = step_timer.last
            self.tracker.log(values=metrics, step=global_step)

            logger.info(f"pipeline {self._pipeline_id} step {global_step} finished")
            global_step += 1

        # Post-loop cleanup: release only clusters that are actually allocated.
        if last_train_cluster_allocated is not None:
            self._notify_release_cluster_gpus(cluster_id=last_train_cluster_allocated, global_step=global_step)
        if ran_any_step:
            self._await_release_actor_infer(global_step=global_step)

        ray.get(
            [
                self.train_rollout_scheduler.shutdown.remote(),
                self.val_rollout_scheduler.shutdown.remote(),
            ]
        )
        logger.info(f"pipeline {self._pipeline_id} complete!")

    def resize_infer(self, *, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> ActionResponse:
        self._ensure_initialized()
        validate_resize_params(dp_ranks_to_remove, dp_ranks_to_add)

        if dp_ranks_to_remove:
            self._shrink_workers(dp_ranks_to_remove=list(dp_ranks_to_remove))
        else:
            self._expand_workers(dp_ranks_to_add=list(dp_ranks_to_add))

        return ActionResponse(success=True)
