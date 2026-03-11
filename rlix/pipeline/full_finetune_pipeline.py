from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from codetiming import Timer

from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX, ActionResponse, get_pipeline_namespace, Priority, SCHEDULER_ACTOR_NAME, RLIX_NAMESPACE

from rlix.pipeline.utils import _get_env_timeout_s

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.pipeline.agentic.agentic_pipeline import compute_rollout_traj_metrics
import threading
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

logger = get_logger()


class RlixFullFinetunePipeline(AgenticPipeline):
    """Rlix-controlled variant of ROLL AgenticPipeline (ENG-123 Phase 3).

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
        try:
            self._rlix_scheduler = ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=RLIX_NAMESPACE)
        except Exception as e:
            # Expectation: the central rlix scheduler actor ('rlix:scheduler')
            # must already be created before the pipeline is instantiated.
            # Fail loudly with a clear message to aid debugging of startup ordering.
            raise RuntimeError(
                f"Failed to resolve {SCHEDULER_ACTOR_NAME} in namespace '{RLIX_NAMESPACE}'. "
                "The pipeline expects the central scheduler actor to be present before startup; "
                "ensure the orchestrator created it earlier or that startup ordering is correct."
            ) from e
        self._actor_infer_cluster_id = f"{self._pipeline_id}_actor_infer"
        self._actor_train_cluster_id = f"{self._pipeline_id}_actor_train"
        self._critic_cluster_id = f"{self._pipeline_id}_critic"
        self._reference_cluster_id = f"{self._pipeline_id}_reference"
        # Lazily resolved and cached on first use by _get_coordinator_handle().
        self._coordinator_handle: Any = None

    def _get_coordinator_handle(self) -> Any:
        """Resolve and cache the per-pipeline RlixCoordinator actor handle.

        Named 'rlix:coordinator:{pipeline_id}' in the pipeline namespace.
        The coordinator serializes resize_infer and sync_lora_weights via _resize_sync_lock.
        """
        if self._coordinator_handle is not None:
            return self._coordinator_handle
        namespace = get_pipeline_namespace(self._pipeline_id)
        actor_name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{self._pipeline_id}"
        try:
            self._coordinator_handle = ray.get_actor(actor_name, namespace=namespace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve coordinator actor {actor_name!r} in namespace {namespace!r}"
            ) from e
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
            from roll.distributed.scheduler.generate_scheduler import RequestScheduler
            from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
            from roll.models.model_providers import default_tokenizer_provider
            from roll.pipeline.base_pipeline import BasePipeline
            from roll.utils.functionals import RunningMoments
            from roll.utils.kl_controller import get_kl_controller
            from roll.utils.constants import RAY_NAMESPACE, rlix_env_vars

            pipeline_config = self._pipeline_config
            BasePipeline.__init__(self, pipeline_config)
            self.pipeline_config = pipeline_config

            self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
            actor_lora_target = getattr(self.pipeline_config.actor_train.model_args, "lora_target", None)
            self.use_ref_model = bool(self.pipeline_config.enable_reference and (actor_lora_target is None))
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

            # shared RequestScheduler (named actor).
            request_scheduler_name = f"RequestScheduler-{self._pipeline_id}"
            self.generate_scheduler = RequestScheduler.options(
                name=request_scheduler_name,
                namespace=RAY_NAMESPACE,
                get_if_exists=True,
                runtime_env={"env_vars": rlix_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency=1024, # Large enough for shared use
            ).remote(
                infer_cluster=self.actor_infer,
                pipeline_config=self.pipeline_config,
                resource_manager=self.resource_manager,
            )

            # Rollout schedulers (named actors).
            self.train_rollout_scheduler = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-{self._pipeline_id}-train",
                namespace=RAY_NAMESPACE,
                runtime_env={"env_vars": rlix_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=self.pipeline_config,
                env_manager_config=self.pipeline_config.train_env_manager,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="train",
            )
            self.val_rollout_scheduler = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-{self._pipeline_id}-val",
                namespace=RAY_NAMESPACE,
                runtime_env={"env_vars": rlix_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=self.pipeline_config,
                env_manager_config=self.pipeline_config.val_env_manager,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="val",
            )

            # Create val dataset manager as in AgenticPipeline.
            from roll.datasets.global_dataset import GlobalDatasetManager

            self.val_dataset_manager = GlobalDatasetManager.options(
                name="val_dataset_manager",
                get_if_exists=True,
                namespace=RAY_NAMESPACE,
                runtime_env={"env_vars": rlix_env_vars()},
            ).remote()

            # Infer resize serialization boundary (ENG-123).
            infer_strategy_config = self.actor_infer.worker_config.strategy_args.strategy_config
            tp_size = int(infer_strategy_config.get("tensor_parallel_size", 1))
            pp_size = int(infer_strategy_config.get("pipeline_parallel_size", 1))
            self._infer_gpus_per_dp_rank = tp_size * pp_size
            self._infer_device_mapping = list(getattr(self.pipeline_config.actor_infer, "device_mapping", None) or [])
            if not self._infer_device_mapping:
                raise RuntimeError("actor_infer.device_mapping must be set")
            self._infer_resize_lock = threading.Lock()

            # INIT PHASE: Initialize clusters with central scheduler coordination and strict offload ordering.
            from rlix.protocol.types import Priority

            init_global_step = -1
            self._request_static_cluster(
                cluster_id=self._actor_train_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_global_step,
            )
            try:
                refs: List[ray.ObjectRef] = []
                refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
                ray.get(refs)

                # Build and promote the initial base-model cache (-1/-1) before offload.
                # Under sleep_level=2 this cache must stay active so expand can rehydrate infer workers.
                init_checkpoint_version = -1
                self.actor_train.load_states(blocking=True)
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
                            checkpoint_version=int(init_checkpoint_version),
                        )
                        for w in self.actor_train.workers
                    ]
                )

                # Offload training-side clusters before initializing actor_infer (avoid transient OOM).
                logger.info("[init][%s] offloading actor_train before actor_infer init", self._pipeline_id)
                self.actor_train.offload_states(blocking=True)
                logger.info("[init][%s] actor_train offload done", self._pipeline_id)
            finally:
                self._release_static_cluster(cluster_id=self._actor_train_cluster_id, global_step=init_global_step)
                logger.info("[init][%s] released actor_train cluster", self._pipeline_id)

            logger.info("[init][%s] requesting actor_infer cluster (INITIALIZATION)", self._pipeline_id)
            self._request_static_cluster(
                cluster_id=self._actor_infer_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_global_step,
            )
            logger.info("[init][%s] actor_infer cluster granted — starting init", self._pipeline_id)
            try:
                refs = []
                if self.reward:
                    refs.extend(self.reward.initialize(pipeline_config=self.pipeline_config, blocking=False))
                refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
                ray.get(refs)
                logger.info("[init][%s] actor_infer initialized — offloading (sleep_level=2: destroy weights+KV)", self._pipeline_id)
                if self.reward:
                    self.reward.offload_states(blocking=True)
                self.actor_infer.offload_states(blocking=True)
                logger.info("[init][%s] actor_infer offload done — GPU memory freed", self._pipeline_id)
            finally:
                self._release_static_cluster(cluster_id=self._actor_infer_cluster_id, global_step=init_global_step)
                logger.info("[init][%s] released actor_infer cluster", self._pipeline_id)

            if self.pipeline_config.adv_estimator == "gae":
                self._request_static_cluster(
                    cluster_id=self._critic_cluster_id,
                    priority=Priority.INITIALIZATION,
                    global_step=init_global_step,
                )
                try:
                    self.critic.initialize(pipeline_config=self.pipeline_config, blocking=True)
                    self.critic.offload_states(blocking=True)
                finally:
                    self._release_static_cluster(cluster_id=self._critic_cluster_id, global_step=init_global_step)

            if self.use_ref_model:
                self._request_static_cluster(
                    cluster_id=self._reference_cluster_id,
                    priority=Priority.INITIALIZATION,
                    global_step=init_global_step,
                )
                try:
                    self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)
                    self.reference.offload_states(blocking=True)
                finally:
                    self._release_static_cluster(cluster_id=self._reference_cluster_id, global_step=init_global_step)

            # RLix mode: skip set_model_update_pair (old ROLL broadcast-based sync).
            # Weight sync is handled by ModelUpdateService.sync_selected_workers (selective sync).
            # Creating the old ModelUpdateGroup would set up a persistent NCCL broadcast group
            # that interferes with the selective sync path.
            if self.pipeline_config.adv_estimator == "gae":
                self.set_checkpoint_clusters(self.actor_train, self.critic)
            else:
                self.set_checkpoint_clusters(self.actor_train)

            self.running = RunningMoments()

            # Validate partial GPU mode configuration and set self.partial_gpu_mode
            if getattr(self.pipeline_config, "partial_gpu_mode", False):
                self.partial_gpu_mode = self._validate_partial_gpu_config()
            else:
                self.partial_gpu_mode = False

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

            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "PIPELINE_ID": os.environ.get("PIPELINE_ID", self._pipeline_id),
                    "ROLL_RAY_NAMESPACE": ray_namespace,
                    "RLIX_CONTROL_PLANE": os.environ.get("RLIX_CONTROL_PLANE", "rlix"),
                }
            }
            svc = ModelUpdateService.options(
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
            )
            ray.get(svc.__ray_ready__.remote())

            # Start from a well-defined state (ENG-123):
            # - disable routing until we request GPUs from RLix.
            # NOTE: avoid local suspend()/resume() state transitions; shrink-to-zero is the single
            # source of truth for pausing generation traffic, and expand-from-zero resumes internally.
            dp_ranks = self._actor_infer_all_dp_ranks()
            ray.get(self.train_rollout_scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))
            ray.get(self.val_rollout_scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))

            self._initialized = True
            return ActionResponse(success=True)

    def _shrink_workers(self, *, dp_ranks_to_remove: List[int]) -> Dict[str, Any]:
        """Pipeline-local shrink helper (ENG-123).

        In RLix mode with shared RequestScheduler, a single call performs:
        - routing-only shrink (updates shared active_dp_ranks)
        - physical offload (skip_offload=False)
        """
        if not isinstance(dp_ranks_to_remove, list) or not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be a non-empty list[int]")
        with self._infer_resize_lock:
            # Both train and val share self.generate_scheduler.
            # One call with skip_offload=False is sufficient.
            return ray.get(
                self.train_rollout_scheduler.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=False)
            )

    def _expand_workers(self, *, dp_ranks_to_add: List[int], train_skip_load: bool) -> Dict[str, Any]:
        """Pipeline-local expand helper (ENG-123).

        In RLix mode with shared RequestScheduler, a single call performs:
        - weight load (skip_load=train_skip_load)
        - routing-only expand (updates shared active_dp_ranks)
        """
        if not isinstance(dp_ranks_to_add, list) or not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be a non-empty list[int]")
        with self._infer_resize_lock:
            # Both train and val share self.generate_scheduler.
            return ray.get(
                self.train_rollout_scheduler.expand_sampler.remote(
                    dp_ranks_to_add, skip_load=bool(train_skip_load)
                )
            )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            resp = self.initialize_pipeline()
            if not getattr(resp, "success", False):
                raise RuntimeError(f"initialize_pipeline failed: {resp}")

    def _actor_infer_device_mapping(self) -> List[int]:
        mapping = getattr(self.pipeline_config.actor_infer, "device_mapping", None)
        if mapping is None:
            raise RuntimeError("actor_infer.device_mapping must be set for Rlix mode")
        if not isinstance(mapping, list):
            raise RuntimeError(f"actor_infer.device_mapping must be list[int], got {type(mapping).__name__}")
        if not mapping:
            raise RuntimeError("actor_infer.device_mapping must be non-empty for Rlix mode")
        if not all(isinstance(x, int) and x >= 0 for x in mapping):
            raise RuntimeError("actor_infer.device_mapping must be list[int>=0]")
        return list(mapping)

    def _actor_infer_all_dp_ranks(self) -> List[int]:
        infer_strategy_config = self.actor_infer.worker_config.strategy_args.strategy_config
        tp_size = int(infer_strategy_config.get("tensor_parallel_size", 1))
        pp_size = int(infer_strategy_config.get("pipeline_parallel_size", 1))
        gpus_per_dp_rank = tp_size * pp_size
        device_mapping = self._actor_infer_device_mapping()
        if len(device_mapping) % int(gpus_per_dp_rank) != 0:
            raise RuntimeError("actor_infer.device_mapping length must be divisible by gpus_per_dp_rank")
        max_dp = len(device_mapping) // int(gpus_per_dp_rank)
        return list(range(int(max_dp)))


    def _request_static_cluster(
        self, *, cluster_id: str, priority: Any, global_step: int, lora_name: Optional[str] = None
    ) -> List[int]:
        allocated = ray.get(
            self._rlix_scheduler.request_gpus.remote(
                cluster_id=str(cluster_id),
                priority=priority,
                global_step=global_step,
                lora_name=lora_name,  # GPU tracing: pass LoRA name for training clusters
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(f"rlix:scheduler.request_gpus returned non-list: {type(allocated).__name__}")
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(f"rlix:scheduler allocated empty GPU list for cluster_id={cluster_id!r}")
        return allocated

    def _release_static_cluster(self, *, cluster_id: str, global_step: int) -> None:
        ray.get(self._rlix_scheduler.release_gpus.remote(cluster_id=str(cluster_id), global_step=global_step))

    def _release_and_request_static_cluster(
        self,
        *,
        release_cluster_id: str,
        release_global_step: int,
        request_cluster_id: str,
        request_priority: Any,
        request_global_step: int,
        request_lora_name: Optional[str] = None,
    ) -> List[int]:
        allocated = ray.get(
            self._rlix_scheduler.release_and_request_gpus.remote(
                release_cluster_id=str(release_cluster_id),
                release_global_step=int(release_global_step),
                request_cluster_id=str(request_cluster_id),
                request_priority=request_priority,
                request_global_step=int(request_global_step),
                request_lora_name=request_lora_name,  # GPU tracing: pass LoRA name for training clusters
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(f"rlix:scheduler.release_and_request_gpus returned non-list: {type(allocated).__name__}")
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(f"rlix:scheduler allocated empty GPU list for cluster_id={request_cluster_id!r}")
        return allocated

    def _notify_ready_to_release_actor_infer(self, *, global_step: int) -> List[int]:
        timeout_s_raw = os.environ.get("RLIX_NOTIFY_READY_TIMEOUT_S", "300")
        try:
            timeout_s = float(timeout_s_raw)
        except ValueError as e:
            raise RuntimeError(f"Invalid RLIX_NOTIFY_READY_TIMEOUT_S={timeout_s_raw!r}") from e
        if timeout_s <= 0:
            raise RuntimeError(f"RLIX_NOTIFY_READY_TIMEOUT_S must be > 0, got {timeout_s!r}")

        released = ray.get(
            self._rlix_scheduler.notify_ready_to_release.remote(
                cluster_id=self._actor_infer_cluster_id,
                global_step=global_step,
                timeout_s=timeout_s,
            )
        )
        if not isinstance(released, list):
            raise RuntimeError(f"notify_ready_to_release returned non-list: {type(released).__name__}")
        released = [int(x) for x in released]
        logger.info(
            f"[rlix][{self._pipeline_id}] notify_ready_to_release done: step={global_step} released={sorted(released)}"
        )
        return released


    @torch.no_grad()
    def run(self):
        """
        Reorganized run method following concurrent_agentic_pipeline_workflow.md.

        Implements individual blocking cycles with request → execute → release pattern
        for each cluster (reference, actor_train, critic). Only actor_infer (rollout)
        uses async/partial allocation.

        Key differences from run():
        - Phase 1: Conditional suspend with atomic try_set_offload_notified()
        - Phase 5: Uses expand_workers() instead of start_server()
        - Phases 11-16: Individual blocking cycles (not merged)
        - Worker methods handle load/offload internally via state_offload_manager
        """
        # Ensure pipeline is initialized before running the training loop.
        self._ensure_initialized()

        logger.info("Starting reorganized concurrent agentic pipeline")

        # RLix: timeouts for notify/gpu-request are managed internally by RLix methods.
        # RLix: model_update() removed — weights are promoted via promote_active_checkpoint after actor training.
        rollout_get_batch_timeout_s = _get_env_timeout_s("ROLL_ROLLOUT_GET_BATCH_TIMEOUT_S", 1800.0)

        
        batch = DataProto()
        batch.meta_info["global_step"] = 0
        # RLix: has_active_allocation not available on RLix scheduler; skip assertion.

        for global_step in range(self.pipeline_config.max_steps):
            # Resume from checkpoint: skip steps already completed (mirrors AgenticPipeline.run()).
            if global_step <= self.state.step:
                global_step += 1
                continue

            batch.meta_info["global_step"] = global_step
            # Offload model states to CPU after every worker call this step (applies to all clusters).
            batch.meta_info["is_offload_states"] = True
            metrics = {}

            logger.info(f"=========={self._pipeline_id} Step {global_step} ==========")  # RLix: use _pipeline_id

            with Timer(name="per_step", logger=None) as step_timer:
                # ============================================================
                # Phase 1: Conditional Suspend & Notify Release
                # Reference: concurrent_agentic_pipeline_workflow.md lines 58-78
                # ============================================================
                if global_step > 0:
                    # Suspend rollout generation (async mode only)
                    # notify_ready_to_release() is idempotent internally, so safe to call always
                    # ray.get(self.train_rollout_scheduler.suspend.remote(), timeout=10)

                    # Notify CentralScheduler that we're ready to release generation GPUs.
                    # RLix: _notify_ready_to_release_actor_infer() wraps ray.get + internal timeout.
                    self._notify_ready_to_release_actor_infer(global_step=global_step - 1)
                    logger.info(f"run() {self._pipeline_id=} Phase 1: Suspended rollout and notified scheduler")

                # RLix: Phase 3 model_update() removed.
                # Weights are promoted to infer workers via promote_active_checkpoint in Phase 16
                # after actor training completes. expand_sampler loads promoted weights on next expand.

                # ============================================================
                # Phase 4.5: Request Generation GPUs, this triggers model update and gpu provisioning
                # Reference: concurrent_agentic_pipeline_workflow.md lines 87-98
                # ============================================================
                # RLix: gpu_scheduler check removed — RLix scheduler is always present.
                allocated_actor_infer_gpus = None
                actor_infer_num_gpus = len(
                    getattr(self.actor_infer.worker_config, 'device_mapping', [])
                )
                assert actor_infer_num_gpus > 0
                expected_gpus = list(self.actor_infer.worker_config.device_mapping)
                if global_step > 0 and (self.pipeline_config.adv_estimator != "gae" or (
                        self.pipeline_config.adv_estimator == "gae" and self.pipeline_config.critic_warmup <= (global_step - 1))):
                    # Offload is enforced in _release_and_request_static_cluster().
                    # RLix: no timeout param.
                    allocated_actor_infer_gpus = self._release_and_request_static_cluster(
                        release_cluster_id=self._actor_train_cluster_id,
                        release_global_step=global_step - 1,
                        request_cluster_id=self._actor_infer_cluster_id,
                        request_priority=Priority.GENERATION,
                        request_global_step=global_step,
                    )
                else:
                    # RLix: no timeout param.
                    allocated_actor_infer_gpus = self._request_static_cluster(
                        cluster_id=self._actor_infer_cluster_id,
                        priority=Priority.GENERATION,
                        global_step=global_step,
                    )
                assert len(allocated_actor_infer_gpus) > 0
                # Log allocation details
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
                        f"This will trigger partial worker expansion. "
                        f"Missing GPUs: {set(expected_gpus) - set(allocated_actor_infer_gpus)}"
                    )
                # RLix: _validate_gpu_allocation() not defined; skip.
                assert len(allocated_actor_infer_gpus) != 0, 'shall not be empty for sched logic as we just released all gpus'

                # ============================================================
                # Phase 5: Expand Workers (Load & Resume)
                # Reference: concurrent_agentic_pipeline_workflow.md lines 102-114
                # ============================================================
                # Phase 5: Central scheduler drives worker expansion via resize_infer() callback.
                # No explicit expand_workers() call needed here.
                # TODO: add val() call here (after GPU allocation, before rollout) for eval_steps > 0.
                # HEAD: if eval_steps > 0 and step % eval_steps == 0: self.val(global_step)

                # ============================================================
                # Phase 7: Rollout Get Batch
                # Reference: concurrent_agentic_pipeline_workflow.md lines 118-124
                # ============================================================
                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch = ray.get(self.train_rollout_scheduler.get_batch.remote(
                        batch, self.pipeline_config.rollout_batch_size
                    ), timeout=rollout_get_batch_timeout_s)
                    dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                batch.meta_info["global_step"] = global_step
                # Required by strategy._get_batch_num_tokens() to identify valid token masks.
                # Mirrors agentic_pipeline.py:441. Source: roll/pipeline/agentic/agentic_pipeline.py
                batch.meta_info["loss_mask_keys"] = ["response_mask"]
                # Required for workers to broadcast non_tensor_batch (traj_id, scores, etc.) across DP ranks.
                batch.meta_info["_broadcast_non_tensor_batch"] = True
                logger.info(f"run() {self._pipeline_id=} Phase 7: Rollout Get Batch")

                # ============================================================
                # Phase 10: Batch Processing (CPU)
                # Reference: concurrent_agentic_pipeline_workflow.md lines 111-115
                # ============================================================
                batch = compute_discounted_returns(
                    batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma
                )
                batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                # Get response level mask
                with Timer(name="cal_response_level_mask", logger=None) as timer:
                    batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
                    metrics.update(mask_metrics)
                metrics["time/cal_response_level_mask"] = timer.last
                logger.info(f"run() {self._pipeline_id=} Phase 10: Batch processing (CPU) completed")

                # ============================================================
                # Phase 11: Value Compute Cycle (Priority.VALUE_COMPUTE, if GAE)
                # Reference: concurrent_agentic_pipeline_workflow.md lines 133-151
                # ============================================================
                if self.pipeline_config.adv_estimator == "gae":
                    # 1. Request GPUs (blocking). RLix: no timeout param.
                    allocated_critic_gpus = self._request_static_cluster(
                        cluster_id=self._critic_cluster_id,
                        priority=Priority.VALUE_COMPUTE,
                        global_step=global_step,
                    )

                    # 2. Compute values (BLOCKING) - internally handles load/offload
                    values_refs = self.critic.compute_values(batch, blocking=False)
                    values = DataProto.materialize_concat(data_refs=values_refs)
                    batch.batch["values"] = values.batch["values"]
                    # Offload is enforced in the upcoming GPU release/transfer call.

                # ============================================================
                # Phase 13: Old Log Probs Cycle (Priority.OLD_LOG_PROBS)
                # Reference: concurrent_agentic_pipeline_workflow.md lines 176-193
                # ============================================================
                # 1. Request GPUs (blocking via PendingRequest). RLix: no timeout param.
                if self.pipeline_config.adv_estimator != "gae":
                     allocated_actor_train_gpus = self._request_static_cluster(
                        cluster_id=self._actor_train_cluster_id,
                        priority=Priority.OLD_LOG_PROBS,
                        global_step=global_step,
                    )
                else:
                    allocated_actor_train_gpus = self._release_and_request_static_cluster(
                        release_cluster_id=self._critic_cluster_id,
                        release_global_step=global_step,
                        request_cluster_id=self._actor_train_cluster_id,
                        request_priority=Priority.OLD_LOG_PROBS,
                        request_global_step=global_step,
                    )

                # 2. Compute log probs (BLOCKING) - internally handles load/offload
                with Timer(name="cal_old_log_probs_values", logger=None) as old_logpb_timer:
                    old_log_probs_refs = self.actor_train.compute_log_probs(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    # TODO: support true ref_log_probs for enable_reference=True configs via a
                    # dedicated reference cluster GPU cycle (mirrors HEAD Phase 11). Simplified
                    # for now: old_log_probs used as ref, correct only when enable_reference=False.
                    batch.batch["ref_log_probs"] = batch.batch["old_log_probs"]
                metrics["time/old_log_probs_values"] = old_logpb_timer.last
                # Offload is enforced in the upcoming GPU release/transfer call.
                logger.info(f"run() {self._pipeline_id=} Phase 13: Old log probs cycle completed")

                # ============================================================
                # Phase 14: Advantage Computation (CPU)
                # Reference: concurrent_agentic_pipeline_workflow.md lines 197-204
                # ============================================================
                with Timer(name="cal_norm_rewards", logger=None) as timer:
                    batch, reward_metrics = compute_response_level_rewards(
                        batch=batch, pipeline_config=self.pipeline_config
                    )
                    metrics.update(reward_metrics)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                metrics["time/cal_norm_rewards"] = timer.last

                with Timer(name="cal_token_reward", logger=None) as timer:
                    batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
                    metrics.update(token_level_metrics)
                metrics["time/cal_token_reward"] = timer.last

                with Timer(name="compute_advantage", logger=None) as timer:
                    # RLix: use agentic_compute_advantage (consistent with agentic_pipeline.py).
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
                metrics["time/adv"] = timer.last
                logger.info(f"run() {self._pipeline_id=} Phase 14: Advantage computation (CPU) completed")

                # When recomputing old log-probs at train time, precompute train-infer IS weights
                # into batch.batch["train_infer_is_weight"] so agentic_actor_worker.loss_func can read it.
                # Mirrors agentic_pipeline.py:613-616. Source: roll/pipeline/agentic/agentic_pipeline.py
                if self.pipeline_config.enable_old_logprobs_recompute:
                    batch, corr_metrics = apply_train_infer_correction_to_batch(
                        self.pipeline_config, batch,
                        update_mask_keys=batch.meta_info['loss_mask_keys'],
                    )
                    metrics.update(corr_metrics)

                # ============================================================
                # Phase 15: Critic Training Cycle (Priority.CRITIC_TRAINING, if GAE)
                # Reference: concurrent_agentic_pipeline_workflow.md lines 207-225
                # ============================================================
                if self.pipeline_config.adv_estimator == "gae":
                    # 1. Request GPUs (blocking). RLix: no timeout param.
                    allocated_critic_gpus = self._release_and_request_static_cluster(
                        release_cluster_id=self._actor_train_cluster_id,
                        release_global_step=global_step,
                        request_cluster_id=self._critic_cluster_id,
                        request_priority=Priority.CRITIC_TRAINING,
                        request_global_step=global_step,
                    )

                    # 2. Train step (BLOCKING) - internally handles load/offload
                    with Timer(name="critic_train_step", logger=None) as critic_train_timer:
                        critic_train_metrics_refs = self.critic.train_step(batch, blocking=False)
                        critic_train_metrics = DataProto.materialize_concat(
                            data_refs=critic_train_metrics_refs
                        )
                        metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                    metrics["time/critic_train_step"] = critic_train_timer.last
                    # Offload is enforced in the upcoming GPU release/transfer call.

                    if self.pipeline_config.critic_warmup > global_step:
                        # RLix: _release_static_cluster instead of _release_gpu.
                        self._release_static_cluster(cluster_id=self._critic_cluster_id, global_step=global_step)
                    logger.info(f"run() {self._pipeline_id=} Phase 15: Critic training cycle completed")

                # ============================================================
                # Phase 16: Actor Training Cycle (Priority.ACTOR_TRAINING)
                # Reference: concurrent_agentic_pipeline_workflow.md lines 229-247
                # ============================================================
                if self.pipeline_config.critic_warmup <= global_step:
                    # 1. Request GPUs (blocking). RLix: no timeout param.
                    if self.pipeline_config.adv_estimator == "gae":
                        allocated_actor_train_gpus = self._release_and_request_static_cluster(
                            release_cluster_id=self._critic_cluster_id,
                            release_global_step=global_step,
                            request_cluster_id=self._actor_train_cluster_id,
                            request_priority=Priority.ACTOR_TRAINING,
                            request_global_step=global_step,
                        )
                    else:
                        # Switch actor_train from OLD_LOG_PROBS -> ACTOR_TRAINING priority (same cluster, different task).
                        allocated_actor_train_gpus = self._release_and_request_static_cluster(
                            release_cluster_id=self._actor_train_cluster_id,
                            release_global_step=global_step,
                            request_cluster_id=self._actor_train_cluster_id,
                            request_priority=Priority.ACTOR_TRAINING,
                            request_global_step=global_step,
                        )

                    # TODO: add batch_balance() here to equalize token counts across DP ranks
                    # before training (mirrors HEAD). Skipped for simplification; restore if
                    # distributed training hangs on uneven shards.
                    # 2. Train step (BLOCKING) - internally handles load/offload
                    with Timer(name="actor_train_step", logger=None) as actor_train_timer:
                        # Shard batch into dynamic micro-batches if enabled; sets global_micro_batch_indices
                        # required by make_mini_batch_iter_for_dynamic_batching() in base_worker.train_step().
                        # Mirrors agentic_pipeline.py:631-641. Source: roll/pipeline/agentic/agentic_pipeline.py
                        if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                            batch, dynamic_batching_metrics = dynamic_batching_shard(
                                batch,
                                self.actor_train.dp_size,
                                self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                self.pipeline_config.actor_train.sequence_length_round_in_train,
                                self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                "actor_train/train_step",
                            )
                            metrics.update(dynamic_batching_metrics)
                        # Time-sharing: tag batch with version for strategy-level cache build.
                        batch.meta_info["checkpoint_version"] = global_step
                        actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                        actor_train_metrics = DataProto.materialize_concat(
                            data_refs=actor_train_metrics_refs
                        )
                        metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
                    metrics["time/train_step"] = actor_train_timer.last

                    # Promote trained weights so expand_sampler can rehydrate infer workers on the next step.
                    # Replaces Phase 3 model_update(): expand_sampler loads from the promoted checkpoint.
                    checkpoint_version = int(batch.meta_info.get("checkpoint_version", global_step))
                    ray.get([
                        worker.promote_active_checkpoint.remote(checkpoint_version)
                        for worker in self.actor_train.workers
                    ])
                    # Append metrics before do_checkpoint so log_history[-1] exists.
                    # metrics is a mutable dict, so Phase 17 updates are visible via the same reference.
                    self.state.step = global_step
                    self.state.log_history.append(metrics)
                    # offload_after_checkpoint=True frees model + optimizer from GPU.
                    # _release_static_cluster runs post-loop, so GPU is still held here.
                    self.do_checkpoint(global_step=global_step, offload_after_checkpoint=True)
                    logger.info(f"run() {self._pipeline_id=} Phase 16: Actor training cycle completed")

                # ============================================================
                # Phase 17: Metrics & Logging
                # Reference: concurrent_agentic_pipeline_workflow.md lines 251-256
                # ============================================================
                # RLix: compute_rollout_traj_metrics replaces compute_data_metrics.
                data_metrics = compute_rollout_traj_metrics(batch)
                metrics.update(data_metrics)
                logger.info(f"run() {self._pipeline_id=} Phase 17: Metrics computation completed")

            # End of Timer block — record per-step wall time before checkpointing.
            metrics["time/per_step_e2e"] = step_timer.last

            # State was already set and log_history was already appended in Phase 16.
            self.tracker.log(values=metrics, step=global_step)
            logger.info(f"=========={self._pipeline_id} Step {global_step} completed ==========")

        # Release train, generation GPUs after the final step (only if any steps ran).
        if self.pipeline_config.max_steps > 0:
            self._release_static_cluster(cluster_id=self._actor_train_cluster_id, global_step=global_step)
            self._notify_ready_to_release_actor_infer(global_step=global_step)
            logger.info(f"run() {self._pipeline_id=} end-of-loop cleanup: actor_train GPU released, scheduler notified")

        # Shut down rollout schedulers to clean up their Ray actors after training completes.
        ray.get([
            self.train_rollout_scheduler.shutdown.remote(),
            self.val_rollout_scheduler.shutdown.remote(),
        ])
        logger.info(f"{self._pipeline_id} pipeline run() completed")

    def resize_infer(self, *, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]):
        self._ensure_initialized()
        if not isinstance(dp_ranks_to_remove, list):
            raise ValueError("dp_ranks_to_remove must be list[int]")
        if not isinstance(dp_ranks_to_add, list):
            raise ValueError("dp_ranks_to_add must be list[int]")
        if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
            raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")

        if dp_ranks_to_remove:
            self._shrink_workers(dp_ranks_to_remove=list(dp_ranks_to_remove))
        else:
            self._expand_workers(dp_ranks_to_add=list(dp_ranks_to_add), train_skip_load=False)

        return ActionResponse(success=True)
