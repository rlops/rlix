"""Rlix Multi-LoRA Pipeline.

Sequential cycle for lora-aware agentic training under Rlix's sleep_level=2:
  Expand -> Rollout (all tags) -> Shrink -> Train (dirty loras) -> Repeat

Key constraints vs AgenticMultiLoraPipeline:
  - sleep_level=2 (GPU weights released; actors stay alive in CPU RAM)
  - No partial_gpu_mode (sequential, not overlapping)
  - megatron_train strategy required
  - is_lora_optimizer_isolated=true required
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
from roll.pipeline.agentic.agentic_pipeline import compute_rollout_traj_metrics, compute_train_data_metrics, get_episode_scores
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
    - actor_train.model_args.adapters is not None (per-adapter optimizer)
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
        # Isolated multi-adapter config validation (is_lora_optimizer_isolated, use_distributed_optimizer,
        # overlap_grad_reduce) is handled in MegatronTrainStrategy.initialize() to cover
        # all init paths, not just this pipeline.
        adapters = getattr(pipeline_config.actor_train.model_args, "adapters", None) or {}
        if not adapters:
            raise RuntimeError(
                "RlixMultiLoraPipeline requires actor_train.model_args.adapters to be non-empty"
            )

        # Create per-LoRA trackers for independent per-adapter metric logging.
        self._create_lora_trackers()

        # TODO: support GAE with per-LoRA critics: frozen backbone + per-LoRA adapters + per-LoRA value heads.
        if self.pipeline_config.adv_estimator == "gae":
            raise NotImplementedError(
                "RlixMultiLoraPipeline does not support adv_estimator='gae'. "
                "A single shared critic cannot produce accurate advantages across different LoRA tasks. "
                "Requires per-LoRA critic adapters and per-LoRA value heads on a shared backbone "
                "(not yet implemented). Use 'grpo' or 'reinforce_plus_plus' instead."
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
            # Shallow-copy the base config so per-tag mutations don't affect other tags.
            env_cfg = replace(base_env)
            # Narrow the config to this single tag's env subset (one tag, one partition).
            env_cfg.tags = [tag]
            env_cfg.num_groups_partition = [n_group]
            env_cfg.num_env_groups = n_group
            env_cfg.name = f"train_env_{tag}"
            # Recompute derived fields (world_size, max_env_num_per_worker, etc.) for the reduced env count.
            env_cfg.__post_init__()
            # Ensure per-tag max_traj_per_env is sufficient after narrowing to this tag's env subset.
            pipeline_config.ensure_min_traj_per_env(env_cfg, pipeline_config.rollout_batch_size)
            # Rebuild env_configs so worker_rank → env_id mapping reflects only this tag's envs.
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
                # RolloutScheduler discovers RequestScheduler via named actor lookup internally.
            )

        # Shut down parent's single global val_rollout_scheduler (replaced by per-tag schedulers below).
        ray.get(self.val_rollout_scheduler.shutdown.remote())

        # Per-tag val rollout schedulers (mirrors train schedulers for per-adapter eval).
        from roll.pipeline.agentic.agentic_config import EnvManagerConfig
        val_env: EnvManagerConfig = pipeline_config.val_env_manager
        val_tags = list(val_env.tags) if getattr(val_env, "tags", None) else []
        # Val tags must match train tags exactly for correct per-adapter eval.
        assert val_tags == tags, (
            f"val_env_manager.tags must match train_env_manager.tags: "
            f"val={val_tags} train={tags}"
        )
        num_tags = len(tags)

        # Validate val partition: no fallback, require explicit valid config.
        val_num_groups_partition = list(getattr(val_env, "num_groups_partition", []) or [])
        assert len(val_num_groups_partition) == num_tags, (
            f"val_env_manager.num_groups_partition length ({len(val_num_groups_partition)}) "
            f"must match num_tags ({num_tags})"
        )
        assert all(n_group > 0 for n_group in val_num_groups_partition), (
            f"val_env_manager.num_groups_partition entries must all be > 0: {val_num_groups_partition}"
        )
        assert sum(val_num_groups_partition) == val_env.num_env_groups, (
            f"sum(val_env_manager.num_groups_partition) = {sum(val_num_groups_partition)} "
            f"must equal val_env_manager.num_env_groups = {val_env.num_env_groups}"
        )

        # Per-tag val_batch_size: equal split, validated per-tag.
        assert pipeline_config.val_batch_size % num_tags == 0, (
            f"val_batch_size ({pipeline_config.val_batch_size}) must be divisible by "
            f"num_tags ({num_tags})"
        )
        val_batch_size_per_tag = pipeline_config.val_batch_size // num_tags
        self._val_batch_size_per_tag: Dict[str, int] = {}
        for tag, val_n_group in zip(tags, val_num_groups_partition):
            tag_val_env_num = val_n_group * val_env.group_size
            assert val_batch_size_per_tag % tag_val_env_num == 0, (
                f"per-tag val_batch_size ({val_batch_size_per_tag}) must be divisible by "
                f"tag {tag!r} val_env_num ({tag_val_env_num} = {val_n_group} * {val_env.group_size})"
            )
            self._val_batch_size_per_tag[tag] = val_batch_size_per_tag

        self.val_rollout_schedulers: Dict[str, Any] = {}
        for tag, val_n_group in zip(tags, val_num_groups_partition):
            val_env_cfg = replace(val_env)
            val_env_cfg.tags = [tag]
            val_env_cfg.num_groups_partition = [val_n_group]
            val_env_cfg.num_env_groups = val_n_group
            val_env_cfg.name = f"val_env_{tag}"
            val_env_cfg.__post_init__()
            # Ensure per-tag max_traj_per_env is sufficient for the proportional val batch.
            pipeline_config.ensure_min_traj_per_env(val_env_cfg, self._val_batch_size_per_tag[tag])
            pipeline_config.make_env_configs(val_env_cfg)
            self.val_rollout_schedulers[tag] = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-{self._pipeline_id}-val-{tag}",
                namespace=ray_namespace,
                runtime_env={"env_vars": rlix_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=pipeline_config,
                env_manager_config=val_env_cfg,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="val",
                # RolloutScheduler discovers RequestScheduler via named actor lookup internally.
            )

        # Point parent's handle to first per-tag scheduler for parent code compatibility.
        self.val_rollout_scheduler = list(self.val_rollout_schedulers.values())[0]

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
        # Shrink val schedulers to zero as well (routing-only, same initial state).
        for scheduler in self.val_rollout_schedulers.values():
            ray.get(scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))

        self._rollout_schedulers_initialized = True
        logger.info(
            f"[init][{self._pipeline_id}] RlixMultiLoraPipeline ready: "
            f"loras={sorted(adapters.keys())} tags={tags}"
        )
        return ActionResponse(success=True)

    def _create_lora_trackers(self) -> None:
        """Create one metrics tracker per LoRA adapter for independent per-adapter tracking."""
        from roll.utils.tracking import create_lora_tracker

        pipeline_config = self._pipeline_config
        adapters = getattr(pipeline_config.actor_train.model_args, "adapters", None) or {}
        if not adapters:
            return
        adapter_names = sorted(adapters.keys())
        tracker_name = pipeline_config.track_with

        self.lora_trackers: Dict[str, Any] = {}
        for name in adapter_names:
            self.lora_trackers[name] = create_lora_tracker(
                tracker_name=tracker_name,
                lora_name=name,
                config=pipeline_config.to_dict(),
                **pipeline_config.tracker_kwargs,
            )
        logger.info("Created per-LoRA trackers for adapters: %s", adapter_names)


    def val_single(self, lora_name: str, global_step: int) -> dict:
        """Validate a single adapter by running only its matching tag's val scheduler."""
        metrics: dict = {}

        for tag, val_scheduler in self.val_rollout_schedulers.items():
            # Only validate the tag that maps to the given adapter.
            if self._tag_to_lora[tag] != lora_name:
                continue
            metrics.update(self._val_tag(tag, val_scheduler, global_step))

        logger.info(f"val_single lora={lora_name} metrics: {metrics}")
        return metrics

    def _val_tag(self, tag: str, val_scheduler: Any, global_step: int) -> dict:
        """Run validation for a single tag and return prefixed metrics."""
        metrics: dict = {}
        batch = DataProto(meta_info={"is_offload_states": False, "global_step": global_step})
        eval_batch = ray.get(val_scheduler.get_batch.remote(batch, self._val_batch_size_per_tag[tag]))

        if "get_batch_return_start_time" in eval_batch.meta_info:
            metrics[f"time/get_batch_cost_val/{tag}"] = (
                time.time() - eval_batch.meta_info.pop("get_batch_return_start_time")
            )

        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, eval_batch)
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{tag}/{k}": v for k, v in eval_metrics.items()})
        return metrics

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
        lora_step: Dict[str, int] = {name: 0 for name in loras}
        # Monotonic global tick for checkpoint ids and eval cadence.
        global_tick: int = 0

        # Resume per-lora state from checkpoint if available.
        if "lora_step_by_adapter" in self.state.kv:
            saved_mapping = self.state.kv["tag_to_adapter"]
            if saved_mapping != self._tag_to_lora:
                raise RuntimeError(
                    f"Checkpoint tag_to_adapter mismatch: saved={saved_mapping} current={self._tag_to_lora}"
                )
            lora_step = dict(self.state.kv["lora_step_by_adapter"])
            global_tick = int(self.state.kv["global_tick"])
            logger.info(f"Resumed from checkpoint: global_tick={global_tick} lora_step={lora_step}")
        tags: List[str] = list(self.rollout_schedulers.keys())

        # Phase-1 / Phase-4.5 state: track whether any tick has completed to know
        # when it is safe to call _notify_ready_to_release_actor_infer.
        any_tick_completed: bool = False
        prev_trained_step: int = 0
        # Tokens-per-second throughput tracker.
        tps_timer = _Timer(window_size=5)
        # Track submission time per tag for rollout wait_s computation
        # (pattern from agentic_multi_lora_pipeline.py:680-683).
        submitted_at_mono: Dict[str, float] = {}

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
                submitted_at_mono[tag] = time.monotonic()

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
                if any_tick_completed:
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
                        submitted_at_mono[tag] = time.monotonic()

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
                # Compute rollout wait time (pattern from agentic_multi_lora_pipeline.py:680-683).
                wait_s = time.monotonic() - submitted_at_mono.pop(ready_tag)

                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch = ray.get(ready_ref)
                    in_flight = deque((tag, ref) for tag, ref in in_flight if tag != ready_tag)
                    lora_name = self._tag_to_lora[ready_tag]

                    # Measure scheduler-to-pipeline transfer latency (pattern from agentic_pipeline.py:340-341).
                    if "get_batch_return_start_time" in batch.meta_info:
                        metrics["time/get_batch_cost_train"] = (
                            time.time() - batch.meta_info.pop("get_batch_return_start_time")
                        )

                    dump_rollout_trajectories(
                        self.pipeline_config.rollout_dump_dir, lora_step[lora_name], batch
                    )
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    # Collect actor inference metrics (pattern from agentic_multi_lora_pipeline.py:688-691).
                    actor_infer_metrics = self.actor_infer.get_metrics()
                    if "metrics" in actor_infer_metrics.meta_info:
                        metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))
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
                # Rollout time = ray.wait duration + post-wait processing (matching ROLL line 801).
                metrics["time/step_rollout"] = rollout_timer.last + wait_s

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
                metrics["time/step_cal_response_level_mask"] = timer.last
                logger.info(f"run() {self._pipeline_id=} Phase 10: batch processing completed")

                # ============================================================
                # Phase 13: Old log probs.
                # ============================================================
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
                # Balance batch for old log-prob compute (production pattern: agentic_pipeline.py:442).
                batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                # Dynamic batching for old log-prob path (matching agentic_pipeline.py:445-455).
                if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                    batch, dynamic_batching_metrics = dynamic_batching_shard(
                        batch,
                        self.actor_train.dp_size,
                        self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                        self.pipeline_config.actor_train.sequence_length_round_in_infer,
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                        "actor_train/compute_log_probs",
                    )
                    metrics.update(dynamic_batching_metrics)
                with Timer(name="cal_old_log_probs_values", logger=None) as old_logpb_timer:
                    old_log_probs_refs = self.actor_train.compute_log_probs(batch, blocking=False)
                    old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                    # TODO: support true ref_log_probs for enable_reference=True via dedicated
                    # reference cluster GPU cycle. Simplified: old_log_probs used as ref.
                    batch.batch["ref_log_probs"] = batch.batch["old_log_probs"]
                    # Log old/ref log-prob and entropy metrics (pattern from agentic_pipeline.py:458-482).
                    avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics["critic/old_log_prob/mean"] = avg_old_log_prob.item()
                    avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                    metrics["critic/ref_log_prob/mean"] = avg_ref_log_prob.item()
                    agg_entropy_val = agg_loss(
                        loss_mat=old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    metrics["critic/entropy/mean"] = agg_entropy_val.item()
                metrics["time/step_old_log_probs_values"] = old_logpb_timer.last
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
                # Compute training data metrics (pattern from agentic_pipeline.py:562).
                # Timer matches base agentic_pipeline.py:564-566.
                with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                    metrics.update(compute_train_data_metrics(batch=batch))
                metrics["time/step_compute_data_metrics"] = data_metrics_timer.last
                logger.info(f"run() {self._pipeline_id=} Phase 14: advantage computation completed")

                if self.pipeline_config.enable_old_logprobs_recompute:
                    batch, corr_metrics = apply_train_infer_correction_to_batch(
                        self.pipeline_config, batch,
                        update_mask_keys=batch.meta_info["loss_mask_keys"],
                    )
                    metrics.update(corr_metrics)

                # ============================================================
                # Phase 16: Actor training (train_step_lora) + promote + scheduler sync.
                # Pattern copied from full_finetune_pipeline.py Phase 16 + HEAD multi_lora_pipeline.py:534-568.
                # ============================================================
                # Switch actor_train from OLD_LOG_PROBS → ACTOR_TRAINING.
                self._release_and_request_static_cluster(
                    release_cluster_id=self._actor_train_cluster_id,
                    release_global_step=lora_step[lora_name],
                    request_cluster_id=self._actor_train_cluster_id,
                    request_priority=Priority.ACTOR_TRAINING,
                    request_global_step=lora_step[lora_name],
                    request_lora_name=lora_name,
                )

                # Derive sample UUIDs from traj_id (same contract as agentic_pipeline.py:338-339).
                sample_uuids = [f"{traj_id}_{idx}" for idx, traj_id in enumerate(batch.non_tensor_batch["traj_id"])]
                batch.non_tensor_batch["sample_uuid"] = np.array(sample_uuids, dtype=object)

                # Balance batch for training (production pattern: agentic_pipeline.py:534-537).
                batch_balance_metrics = batch_balance(
                    batch=batch,
                    dp_size=self.actor_train.dp_size,
                    minibatch_size=self.actor_train.dp_size
                    * self.pipeline_config.actor_train.training_args.per_device_train_batch_size
                    * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps,
                    logging_prefix="global_seqlen/actor_train",
                )
                metrics.update(batch_balance_metrics)

                # Dynamic batching: shard batch before training (pattern from agentic_multi_lora_pipeline.py:747-761).
                if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                    batch, dynamic_batching_metrics = dynamic_batching_shard(
                        batch,
                        self.actor_train.dp_size,
                        self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                        self.pipeline_config.actor_train.sequence_length_round_in_train,
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                        "actor_train/train_step_lora",
                    )
                    metrics.update(dynamic_batching_metrics)

                with Timer(name="actor_train_step", logger=None) as actor_train_timer:
                    # Time-sharing: tag batch with version for strategy-level cache build.
                    batch.meta_info["checkpoint_version"] = lora_step[lora_name]
                    # (a) Train using per-lora optimizer step.
                    actor_train_metrics_refs = self.actor_train.train_step_lora(batch, blocking=False)
                    actor_train_metrics = DataProto.materialize_concat(
                        data_refs=actor_train_metrics_refs
                    )
                    metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
                metrics["time/step_train"] = actor_train_timer.last
                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                metrics["system/tps"] = tps_timer.mean_throughput

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
                logger.info(f"run() {self._pipeline_id=} Phase 16: actor training + sync completed")
                # ============================================================
                # Phase 17: Per-lora step tracking and metrics.
                # ============================================================
                prev_trained_step = lora_step[lora_name]  # capture before increment
                lora_step[lora_name] += 1
                global_tick += 1
                any_tick_completed = True

                # Per-adapter validation: run after sync so inference weights are current.
                if (self.pipeline_config.eval_steps > 0
                        and lora_step[lora_name] % self.pipeline_config.eval_steps == 0):
                    with Timer(name="val", logger=None) as val_timer:
                        val_metrics = self.val_single(lora_name=lora_name, global_step=lora_step[lora_name])
                    metrics.update(val_metrics)
                    metrics["time/step_val"] = val_timer.last

                metrics.update(compute_rollout_traj_metrics(batch))
                metrics["system/lora_step"] = lora_step[lora_name]
                metrics["system/global_tick"] = global_tick
                for name, step in lora_step.items():
                    metrics[f"system/lora_step/{name}"] = step
                # Cumulative sample count (pattern from agentic_pipeline.py:569).
                metrics["system/samples"] = global_tick * self.pipeline_config.rollout_batch_size
                metrics["system/lora_name"] = lora_name
                logger.info(f"run() {self._pipeline_id=} Phase 17: metrics computed lora_step={lora_step}")

            # End of Timer block — record per-tick wall time before checkpointing.
            metrics["time/step_total"] = step_timer.last

            # Per-LoRA tracker: per-adapter step counter (same view as full-finetune).
            if hasattr(self, "lora_trackers"):
                self.lora_trackers[lora_name].log(values=metrics, step=lora_step[lora_name])
            # Shared tracker: global tick for pipeline-wide overview (no overlapping steps).
            self.tracker.log(values=metrics, step=global_tick)

            # Persist per-lora state for checkpoint resume.
            all_done = all(lora_step[name] >= max_steps_per_lora for name in loras)
            self.state.kv["lora_step_by_adapter"] = dict(lora_step)
            self.state.kv["global_tick"] = global_tick
            self.state.kv["tag_to_adapter"] = dict(self._tag_to_lora)
            self.state.step = global_tick
            # Minimal log_history entry for do_checkpoint (reads log_history[-1] for system/step).
            # Do not persist full tick_metrics: base resume replay lacks lora_name context.
            self.state.log_history.append({"system/step": global_tick})
            self.do_checkpoint(global_step=global_tick, is_last_step=all_done, offload_after_checkpoint=True)
            logger.info(f"===== {self._pipeline_id} tick completed lora={lora_name!r} step={lora_step[lora_name]} =====")

            # Re-kick in-flight get_batch for the consumed tag if lora has more steps.
            if lora_step[lora_name] < max_steps_per_lora:
                ref = self.rollout_schedulers[ready_tag].get_batch.remote(
                    DataProto(meta_info={"global_step": lora_step[lora_name]}),
                    self.pipeline_config.rollout_batch_size,
                )
                in_flight.append((ready_tag, ref))
                submitted_at_mono[ready_tag] = time.monotonic()

        # ============================================================
        # End-of-loop cleanup: release GPUs, shut down schedulers, close trackers.
        # Each cleanup step isolated so one failure doesn't skip the rest.
        # ============================================================
        try:
            max_lora_step = max(lora_step.values()) if lora_step else 0
            if max_lora_step > 0:
                self._notify_ready_to_release_actor_infer(global_step=max_lora_step - 1)
                self._release_static_cluster(
                    cluster_id=self._actor_train_cluster_id, global_step=max_lora_step - 1
                )
        except Exception:
            logger.exception("Failed to release GPU clusters")
        try:
            ray.get(
                [sched.shutdown.remote() for sched in self.rollout_schedulers.values()]
                + [sched.shutdown.remote() for sched in self.val_rollout_schedulers.values()]
            )
        except Exception:
            logger.exception("Failed to shutdown rollout schedulers")
        try:
            if hasattr(self, "lora_trackers"):
                for lora_tracker in self.lora_trackers.values():
                    lora_tracker.finish()
            self.tracker.finish()
        except Exception:
            logger.exception("tracker.finish failed")
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
        """Shrink all per-tag rollout schedulers (train + val) atomically.

        2-phase pattern: routing-only shrink on all except last, then physical offload on last.
        """
        if not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be non-empty")
        with self._infer_resize_lock:
            # Include both train and val schedulers so shrink covers all dispatchers.
            all_schedulers = list(self.rollout_schedulers.values()) + list(self.val_rollout_schedulers.values())
            # Phase 1: routing-only shrink on all except last.
            for scheduler in all_schedulers[:-1]:
                ray.get(scheduler.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=True))
            # Phase 2: last scheduler does routing + physical offload.
            ray.get(all_schedulers[-1].shrink_sampler.remote(dp_ranks_to_remove, skip_offload=False))

    def _expand_all_schedulers(self, *, dp_ranks_to_add: List[int]) -> None:
        """Expand all per-tag rollout schedulers (train + val) atomically.

        Sequential pattern: first scheduler does physical load, rest do routing-only expand.
        """
        if not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be non-empty")
        with self._infer_resize_lock:
            # Include both train and val schedulers so expand covers all dispatchers.
            all_schedulers = list(self.rollout_schedulers.values()) + list(self.val_rollout_schedulers.values())
            # First scheduler loads model states.
            ray.get(all_schedulers[0].expand_sampler.remote(dp_ranks_to_add, skip_load=False))
            # Rest do routing-only expand.
            for scheduler in all_schedulers[1:]:
                ray.get(scheduler.expand_sampler.remote(dp_ranks_to_add, skip_load=True))
            # TODO(item-6): Run a dummy forward pass (batch_size=1) on newly expanded workers to
            # initialize CUDA kernels before exposing them to the scheduler (prevents first-request
            # timeout). Not implemented yet — monitor expand latency before adding.
