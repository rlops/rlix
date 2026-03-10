"""Rlix Multi-LoRA Pipeline.

Sequential cycle for lora-aware agentic training under Rlix's sleep_level=2:
  Expand -> Rollout (all tags) -> Shrink -> Train (dirty loras) -> Repeat

Key constraints vs AgenticMultiLoraPipeline:
  - sleep_level=2 (GPU weights released; actors stay alive in CPU RAM)
  - No partial_gpu_mode (sequential, not overlapping)
  - megatron_train strategy required
  - lora_optimizer_mode='per_adapter' required
  - Per-tag RolloutSchedulers (one per env tag / lora)
"""
from __future__ import annotations

import json
import os
import time
import threading
from collections import deque
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer

from rlix.protocol.types import ActionResponse, Priority

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import compute_rollout_traj_metrics, compute_train_data_metrics
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
    get_agentic_response_level_mask,
)
from rlix.pipeline.full_finetune_pipeline import RlixFullFinetunePipeline
from rlix.pipeline.utils import _get_env_timeout_s
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    agg_loss,
    batch_balance,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
)
from roll.utils.logging import get_logger
from roll.utils.lora_routing import normalize_domain
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch

logger = get_logger()


class RlixMultiLoraPipeline(RlixFullFinetunePipeline):
    """Rlix-controlled multi-LoRA agentic pipeline.

    Cycle: Expand → Rollout (all tags) → Shrink → Train (dirty loras) → Repeat.

    Constraints:
    - actor_infer.strategy_args.strategy_config.sleep_level == 2
    - actor_train.strategy_args.strategy_name == 'megatron_train'
    - actor_train.strategy_args.strategy_config.lora_optimizer_mode == 'per_adapter'
    - actor_train.model_args.adapters is not None
    """

    def initialize_pipeline(self) -> ActionResponse:
        """Initialize pipeline with per-tag rollout schedulers and multi-LoRA validation."""
        # super() owns _init_lock + _initialized guard; do not re-acquire here (not reentrant).
        result = super().initialize_pipeline()
        if not getattr(result, "success", False):
            return result

        # Guard child-specific init (idempotent: Ray may call twice if actor restarts are enabled).
        if getattr(self, "_rollout_schedulers_initialized", False):
            return ActionResponse(success=True)

        pipeline_config = self._pipeline_config

        # --- Multi-LoRA validation ---
        train_strategy_name = (
            getattr(getattr(pipeline_config.actor_train, "strategy_args", None), "strategy_name", None)
        )
        if train_strategy_name != "megatron_train":
            raise RuntimeError(
                f"RlixMultiLoraPipeline requires actor_train strategy_name='megatron_train', "
                f"got {train_strategy_name!r}"
            )
        train_strategy_config = (
            getattr(getattr(pipeline_config.actor_train, "strategy_args", None), "strategy_config", None) or {}
        )
        lora_optimizer_mode = train_strategy_config.get("lora_optimizer_mode", "shared")
        if lora_optimizer_mode != "per_adapter":
            raise RuntimeError(
                "RlixMultiLoraPipeline requires actor_train strategy_config.lora_optimizer_mode='per_adapter', "
                f"got {lora_optimizer_mode!r}"
            )
        adapters = getattr(pipeline_config.actor_train.model_args, "adapters", None) or {}
        if not adapters:
            raise RuntimeError(
                "RlixMultiLoraPipeline requires actor_train.model_args.adapters to be non-empty"
            )

        # --- Static VRAM cap (Phase 2) ---
        max_resident = getattr(pipeline_config, "max_resident_adapters", None)
        if max_resident is not None and len(adapters) > int(max_resident):
            raise RuntimeError(
                f"RlixMultiLoraPipeline: number of loras ({len(adapters)}) exceeds "
                f"max_resident_adapters ({max_resident}). Reduce the lora count or raise the cap."
            )

        # --- Build tag → lora mapping ---
        base_env = pipeline_config.train_env_manager
        tags = list(base_env.tags) if getattr(base_env, "tags", None) else []
        if not tags:
            raise RuntimeError("train_env_manager.tags must be non-empty for RlixMultiLoraPipeline")
        self._tag_to_lora: Dict[str, str] = {tag: normalize_domain(tag) for tag in tags}
        unknown = sorted({a for a in self._tag_to_lora.values() if a not in adapters})
        if unknown:
            raise RuntimeError(
                f"RlixMultiLoraPipeline: env tags map to unknown loras: {unknown}. "
                f"Configured loras: {sorted(adapters.keys())}"
            )

        # --- Per-tag rollout schedulers ---
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
        from roll.utils.constants import rlix_env_vars

        ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE", "roll")
        num_groups_partition = list(getattr(base_env, "num_groups_partition", []) or [])
        if len(num_groups_partition) != len(tags):
            # Fall back: equal partition
            num_groups_partition = [getattr(base_env, "num_env_groups", 1)] * len(tags)

        self.rollout_schedulers: Dict[str, Any] = {}
        for tag, n_group in zip(tags, num_groups_partition):
            env_cfg = replace(base_env)
            env_cfg.tags = [tag]
            env_cfg.num_groups_partition = [n_group]
            env_cfg.num_env_groups = n_group
            env_cfg.name = f"train_env_{tag}"
            env_cfg.__post_init__()
            # Ensure each per-tag scheduler can produce rollout_batch_size trajectories per step.
            train_env_num = env_cfg.num_env_groups * env_cfg.group_size
            traj_per_env = (pipeline_config.rollout_batch_size + train_env_num - 1) // train_env_num
            if env_cfg.max_traj_per_env < traj_per_env:
                env_cfg.max_traj_per_env = traj_per_env
            pipeline_config.make_env_configs(env_cfg)

            self.rollout_schedulers[tag] = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-{self._pipeline_id}-{tag}",
                namespace=ray_namespace,
                runtime_env={"env_vars": rlix_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=pipeline_config,
                env_manager_config=env_cfg,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="train",
                request_scheduler=self.generate_scheduler,
            )

        # Build and promote initial per-lora caches so first expand can sync all loras.
        all_loras = list(dict.fromkeys(self._tag_to_lora.values()))
        for lora_name in all_loras:
            ray.get([
                worker.build_latest_bucket_cache.remote(0, lora_name)
                for worker in self.actor_train.workers
            ])
            ray.get([
                worker.promote_active_adapter_checkpoint.remote(lora_name, 0)
                for worker in self.actor_train.workers
            ])

        # Shrink all per-tag schedulers to zero (initial state, before first expand).
        dp_ranks = self._actor_infer_all_dp_ranks()
        for scheduler in self.rollout_schedulers.values():
            ray.get(scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))

        self._rollout_schedulers_initialized = True
        logger.info(
            f"[init][{self._pipeline_id}] RlixMultiLoraPipeline ready: "
            f"loras={sorted(adapters.keys())} tags={tags}"
        )
        return ActionResponse(success=True)


    @torch.no_grad()
    def run(self) -> None:
        """Multi-LoRA training loop.

        Per-lora step tracking with first-ready (barrier_mode=False) dispatch:
        each lora trains independently and terminates when its lora_step reaches max_steps.

        Cycle per tick (one ready tag):
          Phase 1 → Phase 4.5 → Phase 7 (async get_batch) → Phase 10 → Phase 13 → Phase 14
          → Phase 15 (GAE only) → Phase 16 (train_step_lora + promote + sync) → Phase 17
        """
        self._ensure_initialized()
        logger.info(f"Starting RlixMultiLoraPipeline run: {self._pipeline_id}")

        rollout_get_batch_timeout_s = _get_env_timeout_s("ROLL_ROLLOUT_GET_BATCH_TIMEOUT_S", 1800.0)

        # Build ordered lora + tag lists (insertion-order dedup via dict.fromkeys).
        loras: List[str] = list(dict.fromkeys(self._tag_to_lora.values()))
        max_steps_per_lora: int = self.pipeline_config.max_steps
        # Per-lora step counters — each terminates independently.
        # TODO: checkpoint resume — restore per-lora lora_step from saved state.
        lora_step: Dict[str, int] = {name: 0 for name in loras}
        tags: List[str] = list(self.rollout_schedulers.keys())

        # Phase-1 / Phase-4.5 state: track whether any tick has completed to know
        # when it is safe to call _notify_ready_to_release_actor_infer.
        any_tick_completed: bool = False
        prev_trained_step: int = 0

        # ============================================================
        # Kick off initial get_batch for all active tags (mirrors agentic_multi_lora_pipeline.py:532-545).
        # ============================================================
        # Track in-flight refs as a single FIFO queue to keep fair wait order.
        # Each item is (tag, get_batch_ref); tags are unique in the queue.
        in_flight: deque[tuple[str, Any]] = deque()
        for tag in tags:
            lora = self._tag_to_lora[tag]
            if lora_step[lora] < max_steps_per_lora:
                ref = self.rollout_schedulers[tag].get_batch.remote(
                    DataProto(meta_info={"global_step": lora_step[lora]}),
                    self.pipeline_config.rollout_batch_size,
                )
                in_flight.append((tag, ref))

        while any(lora_step[name] < max_steps_per_lora for name in loras):
            metrics: Dict[str, Any] = {}

            with Timer(name="per_step", logger=None) as step_timer:
  
                # ============================================================
                # Phase 4.5: Request generation GPUs.
                # On the first tick there is no cluster to release; on subsequent ticks
                # release actor_train (from previous training) and request actor_infer.
                # ============================================================
                expected_gpus = list(self.actor_infer.worker_config.device_mapping)
                assert len(expected_gpus) > 0
                if any_tick_completed and (
                    self.pipeline_config.adv_estimator != "gae"
                    or self.pipeline_config.critic_warmup <= prev_trained_step
                ):
                    # Release actor_train GPUs from last tick and request actor_infer GPUs.
                    allocated_actor_infer_gpus = self._release_and_request_static_cluster(
                        release_cluster_id=self._actor_train_cluster_id,
                        release_global_step=prev_trained_step,
                        request_cluster_id=self._actor_infer_cluster_id,
                        request_priority=Priority.GENERATION,
                        request_global_step=prev_trained_step + 1,
                    )
                else:
                    allocated_actor_infer_gpus = self._request_static_cluster(
                        cluster_id=self._actor_infer_cluster_id,
                        priority=Priority.GENERATION,
                        global_step=prev_trained_step,
                    )
                assert len(allocated_actor_infer_gpus) > 0
                is_partial_allocation = len(allocated_actor_infer_gpus) < len(expected_gpus)
                logger.info(
                    f"run() {self._pipeline_id=} Phase 4.5: infer GPU alloc "
                    f"expected={expected_gpus} allocated={allocated_actor_infer_gpus} "
                    f"partial={is_partial_allocation}"
                )

                # ============================================================
                # Phase 7: First-ready get_batch (barrier_mode=False).
                # Fill any gaps for active tags, then wait for the first ready ref.
                # Pattern copied from agentic_multi_lora_pipeline.py:556-639.
                # ============================================================
                for tag in tags:
                    lora = self._tag_to_lora[tag]
                    # Keep at most one in-flight request per tag.
                    if lora_step[lora] < max_steps_per_lora and all(t != tag for t, _ in in_flight):
                        ref = self.rollout_schedulers[tag].get_batch.remote(
                            DataProto(meta_info={"global_step": lora_step[lora]}),
                            self.pipeline_config.rollout_batch_size,
                        )
                        in_flight.append((tag, ref))

                # Build wait inputs using queue order (head first) to avoid fixed tag-order bias.
                active_refs = [ref for _, ref in in_flight]
                assert active_refs, f"no in-flight get_batch refs; lora_step={lora_step}"
                ready, _ = ray.wait(active_refs, num_returns=1, timeout=rollout_get_batch_timeout_s)
                if not ready:
                    raise RuntimeError(
                        f"get_batch timed out ({rollout_get_batch_timeout_s}s) "
                        f"in_flight={sorted(tag for tag, _ in in_flight)}"
                    )
                ready_ref = ready[0]
                ready_tag = next(tag for tag, ref in in_flight if ref == ready_ref)
                batch = ray.get(ready_ref)
                in_flight = deque((tag, ref) for tag, ref in in_flight if tag != ready_tag)
                lora_name = self._tag_to_lora[ready_tag]

                dump_rollout_trajectories(
                    self.pipeline_config.rollout_dump_dir, lora_step[lora_name], batch
                )
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                # Required by strategy._get_batch_num_tokens() to identify valid token masks.
                batch.meta_info["loss_mask_keys"] = ["response_mask"]
                # Required for workers to broadcast non_tensor_batch across DP ranks.
                batch.meta_info["_broadcast_non_tensor_batch"] = True
                # Pass per-lora step so base_worker.train_step_lora can build bucket cache.
                batch.meta_info["global_step"] = lora_step[lora_name]
                batch.meta_info["is_offload_states"] = True
                logger.info(
                    f"run() {self._pipeline_id=} Phase 7: ready tag={ready_tag!r} "
                    f"lora={lora_name!r} lora_step={lora_step[lora_name]}"
                )

                # ============================================================
                # Phase 10: Batch processing (CPU).
                # ============================================================
                batch = compute_discounted_returns(
                    batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma
                )
                batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                with Timer(name="cal_response_level_mask", logger=None) as timer:
                    batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
                    metrics.update(mask_metrics)
                metrics["time/cal_response_level_mask"] = timer.last
                logger.info(f"run() {self._pipeline_id=} Phase 10: batch processing completed")

                # ============================================================
                # Phase 11: Value compute (GAE only).
                # ============================================================
                if self.pipeline_config.adv_estimator == "gae":
                    self._request_static_cluster(
                        cluster_id=self._critic_cluster_id,
                        priority=Priority.VALUE_COMPUTE,
                        global_step=lora_step[lora_name],
                    )
                    values_refs = self.critic.compute_values(batch, blocking=False)
                    values = DataProto.materialize_concat(data_refs=values_refs)
                    batch.batch["values"] = values.batch["values"]

                # ============================================================
                # Phase 13: Old log probs.
                # ============================================================
                if self.pipeline_config.adv_estimator != "gae":
                    # Do NOT call _notify_ready_to_release_actor_infer here. In multi-lora, we
                    # sync dirty lora weights directly to active infer workers at Phase 16.
                    # The scheduler's preemption path frees only the GPUs that actor_train needs
                    # (a partial shrink), so active_dp_ranks stays non-empty through Phase 16.
                    # After actor_train releases, the scheduler calls expand_worker to sync
                    # loras to any workers that were preempted (now idle).
                    allocated_actor_train_gpus = self._request_static_cluster(
                        cluster_id=self._actor_train_cluster_id,
                        priority=Priority.OLD_LOG_PROBS,
                        global_step=lora_step[lora_name],
                        lora_name=lora_name,
                    )
                else:
                    allocated_actor_train_gpus = self._release_and_request_static_cluster(
                        release_cluster_id=self._critic_cluster_id,
                        release_global_step=lora_step[lora_name],
                        request_cluster_id=self._actor_train_cluster_id,
                        request_priority=Priority.OLD_LOG_PROBS,
                        request_global_step=lora_step[lora_name],
                        request_lora_name=lora_name,
                    )
                with Timer(name="cal_old_log_probs_values", logger=None) as old_logpb_timer:
                    old_log_probs_refs = self.actor_train.compute_log_probs(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    # TODO: support true ref_log_probs for enable_reference=True via dedicated
                    # reference cluster GPU cycle. Simplified: old_log_probs used as ref.
                    batch.batch["ref_log_probs"] = batch.batch["old_log_probs"]
                metrics["time/old_log_probs_values"] = old_logpb_timer.last
                logger.info(f"run() {self._pipeline_id=} Phase 13: old log probs completed")

                # ============================================================
                # Phase 14: Advantage computation (CPU).
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
                logger.info(f"run() {self._pipeline_id=} Phase 14: advantage computation completed")

                if self.pipeline_config.enable_old_logprobs_recompute:
                    batch, corr_metrics = apply_train_infer_correction_to_batch(
                        self.pipeline_config, batch,
                        update_mask_keys=batch.meta_info["loss_mask_keys"],
                    )
                    metrics.update(corr_metrics)

                # ============================================================
                # Phase 15: Critic training (GAE only).
                # ============================================================
                if self.pipeline_config.adv_estimator == "gae":
                    self._release_and_request_static_cluster(
                        release_cluster_id=self._actor_train_cluster_id,
                        release_global_step=lora_step[lora_name],
                        request_cluster_id=self._critic_cluster_id,
                        request_priority=Priority.CRITIC_TRAINING,
                        request_global_step=lora_step[lora_name],
                    )
                    with Timer(name="critic_train_step", logger=None) as critic_train_timer:
                        critic_train_metrics_refs = self.critic.train_step(batch, blocking=False)
                        critic_train_metrics = DataProto.materialize_concat(
                            data_refs=critic_train_metrics_refs
                        )
                        metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                    metrics["time/critic_train_step"] = critic_train_timer.last

                    if self.pipeline_config.critic_warmup > lora_step[lora_name]:
                        self._release_static_cluster(
                            cluster_id=self._critic_cluster_id,
                            global_step=lora_step[lora_name],
                        )
                    logger.info(f"run() {self._pipeline_id=} Phase 15: critic training completed")

                # ============================================================
                # Phase 16: Actor training (train_step_lora) + promote + scheduler sync.
                # Pattern copied from full_finetune_pipeline.py Phase 16 + HEAD multi_lora_pipeline.py:534-568.
                # ============================================================
                if self.pipeline_config.critic_warmup <= lora_step[lora_name]:
                    # Request actor_train GPUs (release critic if GAE, else re-request actor_train).
                    if self.pipeline_config.adv_estimator == "gae":
                        self._release_and_request_static_cluster(
                            release_cluster_id=self._critic_cluster_id,
                            release_global_step=lora_step[lora_name],
                            request_cluster_id=self._actor_train_cluster_id,
                            request_priority=Priority.ACTOR_TRAINING,
                            request_global_step=lora_step[lora_name],
                            request_lora_name=lora_name,
                        )
                    else:
                        # Switch actor_train from OLD_LOG_PROBS → ACTOR_TRAINING.
                        self._release_and_request_static_cluster(
                            release_cluster_id=self._actor_train_cluster_id,
                            release_global_step=lora_step[lora_name],
                            request_cluster_id=self._actor_train_cluster_id,
                            request_priority=Priority.ACTOR_TRAINING,
                            request_global_step=lora_step[lora_name],
                            request_lora_name=lora_name,
                        )

                    with Timer(name="actor_train_step", logger=None) as actor_train_timer:
                        # (a) Train using per-lora optimizer step.
                        actor_train_metrics_refs = self.actor_train.train_step_lora(batch, blocking=False)
                        actor_train_metrics = DataProto.materialize_concat(
                            data_refs=actor_train_metrics_refs
                        )
                        metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
                    metrics["time/train_step"] = actor_train_timer.last

                    # (b) Extract trained loras from lora_name; fail fast if missing or unknown.
                    if "lora_name" not in batch.non_tensor_batch:
                        raise RuntimeError("missing non_tensor_batch['lora_name']")
                    valid_loras = set(self._tag_to_lora.values())
                    trained_loras: List[str] = list(dict.fromkeys(
                        str(n) for n in batch.non_tensor_batch["lora_name"].tolist()
                        if str(n) in valid_loras
                    ))
                    if not trained_loras:
                        raise RuntimeError(
                            f"no recognized loras in lora_name: "
                            f"{batch.non_tensor_batch['lora_name'].tolist()!r}"
                        )

                    # (c) Promote per-lora checkpoint — enables expand_sampler to load on next expand.
                    checkpoint_version = int(
                        batch.meta_info.get("checkpoint_version", lora_step[lora_name])
                    )
                    for lora in trained_loras:
                        ray.get([
                            worker.promote_active_adapter_checkpoint.remote(
                                lora, checkpoint_version
                            )
                            for worker in self.actor_train.workers
                        ])

                    # (d) Push updated lora weights to active infer workers directly via
                    # the coordinator actor. The coordinator looks up generate_scheduler itself and
                    # queries active_dp_ranks inside _resize_sync_lock to avoid race conditions.
                    # If all workers are sleeping (preempted by concurrent pipelines),
                    # the coordinator skips sync and expand_worker handles it on next wake.
                    ray.get(self._get_coordinator_handle().sync_lora_weights.remote(
                        loras_to_sync=trained_loras,
                    ))
                    # Append metrics before do_checkpoint so log_history[-1] exists.
                    # metrics is a mutable dict, so Phase 17 updates are visible via the same reference.
                    self.state.step = lora_step[lora_name]
                    self.state.log_history.append(metrics)
                    # Checkpoint while actor_train GPU is still held, then offload all states
                    # so the GPU is clean when Phase 4.5 of the next tick releases actor_train
                    # and requests actor_infer (preventing OOM on the infer expand).
                    self.do_checkpoint(global_step=lora_step[lora_name], offload_after_checkpoint=True)
                    # actor_train GPU is released at Phase 4.5 of the next while-loop tick
                    # via _release_and_request_static_cluster; GPU is clean (offloaded) by then.
                    logger.info(f"run() {self._pipeline_id=} Phase 16: actor training + sync + checkpoint completed")
                # ============================================================
                # Phase 17: Per-lora step tracking and metrics.
                # ============================================================
                prev_trained_step = lora_step[lora_name]  # capture before increment
                lora_step[lora_name] += 1
                any_tick_completed = True

                metrics.update(compute_rollout_traj_metrics(batch))
                metrics["system/lora_step"] = lora_step[lora_name]
                for name, step in lora_step.items():
                    metrics[f"system/lora_step/{name}"] = step
                logger.info(f"run() {self._pipeline_id=} Phase 17: metrics computed lora_step={lora_step}")

            # End of Timer block — record per-tick wall time before checkpointing.
            metrics["time/per_step_e2e"] = step_timer.last

            # state.step and log_history were already set in Phase 16.
            self.tracker.log(values=metrics, step=lora_step[lora_name], lora_name=lora_name)
            logger.info(f"===== {self._pipeline_id} tick completed lora={lora_name!r} step={lora_step[lora_name]} =====")

            # Re-kick in-flight get_batch for the consumed tag if lora has more steps.
            if lora_step[lora_name] < max_steps_per_lora:
                ref = self.rollout_schedulers[ready_tag].get_batch.remote(
                    DataProto(meta_info={"global_step": lora_step[lora_name]}),
                    self.pipeline_config.rollout_batch_size,
                )
                in_flight.append((ready_tag, ref))

        # ============================================================
        # End-of-loop cleanup: release GPUs and shut down schedulers.
        # ============================================================
        max_lora_step = max(lora_step.values()) if lora_step else 0
        if max_lora_step > 0:
            self._notify_ready_to_release_actor_infer(global_step=max_lora_step - 1)
            self._release_static_cluster(
                cluster_id=self._actor_train_cluster_id, global_step=max_lora_step - 1
            )
        ray.get([sched.shutdown.remote() for sched in self.rollout_schedulers.values()])
        ray.get(self.val_rollout_scheduler.shutdown.remote())
        logger.info(f"{self._pipeline_id} pipeline run() completed")

    def resize_infer(self, *, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]):
        """Rlix hook for per-tag scheduler shrink/expand."""
        self._ensure_initialized()
        if not isinstance(dp_ranks_to_remove, list):
            raise ValueError("dp_ranks_to_remove must be list[int]")
        if not isinstance(dp_ranks_to_add, list):
            raise ValueError("dp_ranks_to_add must be list[int]")
        if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
            raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")

        if dp_ranks_to_remove:
            self._shrink_all_schedulers(dp_ranks_to_remove=list(dp_ranks_to_remove))
        else:
            try:
                self._expand_all_schedulers(dp_ranks_to_add=list(dp_ranks_to_add))
            except Exception as e:
                error_msg = str(e)
                logger.fatal(
                    f"[rlix][{self._pipeline_id}] expand failed (possible partial TP group failure): {error_msg}"
                )
                raise RuntimeError(f"PARTIAL_TP_GROUP_FAILURE: {error_msg}") from e

        return ActionResponse(success=True)

    def _shrink_all_schedulers(self, *, dp_ranks_to_remove: List[int]) -> None:
        """Shrink all per-tag rollout schedulers (atomically via shared RequestScheduler)."""
        if not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be non-empty")
        with self._infer_resize_lock:
            # All per-tag schedulers and val_rollout_scheduler share the same RequestScheduler actor.
            # A single call with skip_offload=False updates routing state and performs physical offload.
            # We use val_rollout_scheduler as the handle, but any would work.
            ray.get(self.val_rollout_scheduler.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=False))

    def _expand_all_schedulers(self, *, dp_ranks_to_add: List[int]) -> None:
        """Expand all per-tag rollout schedulers (atomically via shared RequestScheduler)."""
        if not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be non-empty")
        with self._infer_resize_lock:
            # All per-tag schedulers and val_rollout_scheduler share the same RequestScheduler actor.
            # A single call with skip_load=False performs weight load/selection sync and updates routing.
            expand_metrics = ray.get(self.val_rollout_scheduler.expand_sampler.remote(dp_ranks_to_add, skip_load=False))
            # TODO(item-6): Run a dummy forward pass (batch_size=1) on newly expanded workers to
            # initialize CUDA kernels before exposing them to the scheduler (prevents first-request
            # timeout). Not implemented yet — monitor expand latency before adding.
