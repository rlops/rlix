"""MilesPipeline — RLix-side per-pipeline actor for the MILES backend.

Drives the init bootstrap, runtime ``_before_training`` / ``_after_training``
hooks, and the M4 minimal hard cleanup. Constructed by
:meth:`MilesCoordinator.create_pipeline_actor`.

Cross-cutting review (post-iter-30) fix: the prior Step 7 attempted an
all-shell init pattern (RolloutManager constructed with ``pg=None`` +
``active_engine_indices=frozenset()``, then ``manager.expand_engines(full)``
to lift shell→loading) — but the receiver-side shell-creation code was
never wired into RolloutManager (iter 5's expand_engines only handles
offloaded→loading). The pragmatic M11.1-single-pipeline fix below
constructs RolloutManager AFTER actor_infer is granted with a real
placement_group via the existing standalone ``_create_placement_group``
flow; engines come up ``active`` and the F22 dual-pipeline-shell-init
contract is deferred to a future iter.

Sequence:
  Phase A (Steps 1–6): request actor_train → RayTrainGroup(worker_placements)
    → init() → build_cpu_bucket_cache(-1) → train.offload() → collect
    cache_owner role → publish_cache_ready_step(-1) (RolloutManager not yet
    created; register/bootstrap deferred to phase B).
  Phase A.5 (P1-7 fix): release actor_train BEFORE requesting actor_infer.
  Phase B (Step 7): request actor_infer → create infer placement_group →
    RolloutManager(args, pg=...) → register_model_update_resources →
    bootstrap_active_engines(full set since engines come up active).

The Step 6.6 pre-registration BEFORE Step 7 INIT contract relaxes here:
publish_cache_ready_step(-1) still fires before Step 7, but the
register_model_update_resources + bootstrap_active_engines pair is now
post-Step-7 because the manager handle is constructed there. F22 in its
strict form ties to the M11.2 dual-pipeline shell-init pattern, which is
deferred along with the receiver-side shell-creation code.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import ray

from rlix.protocol.types import (
    ACTOR_TRAIN_CLUSTER_NAME,
    GENERATION_CLUSTER_NAME,
    Priority,
    RLIX_NAMESPACE,
    SCHEDULER_ACTOR_NAME,
    get_pipeline_namespace,
)
from rlix.utils.env import parse_env_positive_float
from rlix.utils.ray import get_actor_or_raise

logger = logging.getLogger(__name__)


class MilesPipeline:
    """Per-pipeline actor created by :class:`MilesCoordinator`."""

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        self._pipeline_id = str(pipeline_id)
        self._pipeline_config = pipeline_config
        # Derive namespace from pipeline_id directly (P1-4 fix). Reading
        # _ray_namespace via getattr on a Ray ActorHandle returns None.
        self._ray_namespace = get_pipeline_namespace(self._pipeline_id)
        # Per-role cluster_id strings — match the canonical pattern in
        # rlix.pipeline.full_finetune_pipeline (P1-3 fix).
        self._actor_train_cluster_id = f"{self._pipeline_id}_{ACTOR_TRAIN_CLUSTER_NAME}"
        self._actor_infer_cluster_id = f"{self._pipeline_id}_{GENERATION_CLUSTER_NAME}"

        # Resolved during initialize_pipeline; None means "not yet
        # constructed" so cleanup paths can skip nicely on early failures.
        self._coordinator_handle = None
        self._train_group = None
        self._rollout_manager = None
        self._cache_owner_actor = None
        self._scheduler = None  # central rlix scheduler handle
        self._initialized: bool = False
        self._declared_engine_count: int = 0
        # Per-cluster_id allocation state for shutdown_hard's ledger
        # release. None means "not allocated"; set to True after a
        # successful request_gpus.
        self._actor_train_allocated: bool = False
        self._actor_infer_allocated: bool = False

        # F35 startup fail-fast — run BEFORE any GPU allocation.
        self._validate_topology(pipeline_config)

    # ------------------------------------------------------------------
    # F10 startup validation
    # ------------------------------------------------------------------

    def _validate_topology(self, pipeline_config: Any) -> None:
        miles_args = getattr(pipeline_config, "miles_args", None)
        if miles_args is None:
            logger.debug(
                "MilesPipeline: pipeline_config has no miles_args; skipping "
                "F10 startup validation (test-fixture path)"
            )
            return
        try:
            from miles.utils.rlix_validation import assert_rlix_topology
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MilesPipeline: miles.utils.rlix_validation unavailable; "
                "skipping startup validation: %r",
                exc,
            )
            return
        sglang_config = getattr(pipeline_config, "sglang_config", None)
        assert_rlix_topology(miles_args, sglang_config=sglang_config)

    # ------------------------------------------------------------------
    # Init bootstrap
    # ------------------------------------------------------------------

    def initialize_pipeline(self, *, coordinator_handle) -> None:
        self._coordinator_handle = coordinator_handle
        try:
            self._init_phase_a_train()  # Steps 1-6 (actor_train side)
            self._init_phase_b_infer()  # Step 7 (actor_infer side)
            self._initialized = True
            logger.info(
                "[MilesPipeline] initialize_pipeline complete pipeline_id=%s engines=%d",
                self._pipeline_id,
                self._declared_engine_count,
            )
        except Exception:
            self.shutdown_hard()
            raise

    def _init_phase_a_train(self) -> None:
        """Steps 1–6 — actor_train side. Step 6.6 publishes cache_ready_step
        but defers register_model_update_resources + bootstrap_active_engines
        to phase B (after the manager exists).
        """
        miles_args = getattr(self._pipeline_config, "miles_args", None)
        if miles_args is None:
            raise RuntimeError(
                "initialize_pipeline requires pipeline_config.miles_args"
            )

        # Step 1: request actor_train allocation. P1-3: real scheduler API
        # is request_gpus(*, cluster_id, priority, global_step), not
        # request_cluster_gpus(pipeline_id=, role=, num_gpus=).
        self._request_cluster_gpus(
            cluster_id=self._actor_train_cluster_id,
            priority=Priority.INITIALIZATION,
            global_step=-1,
        )
        self._actor_train_allocated = True

        # Steps 2/3: construct RayTrainGroup via worker_placements.
        placement_provider = self._build_placement_provider(miles_args)
        train_workers = placement_provider.get_train_workers()
        # Stash the provider so phase B can reuse it for the infer-pool
        # placement group without re-constructing a fresh proxy.
        self._placement_provider = placement_provider

        from miles.ray.actor_group import RayTrainGroup

        self._train_group = RayTrainGroup(
            miles_args,
            num_nodes=int(miles_args.actor_num_nodes),
            num_gpus_per_node=int(miles_args.actor_num_gpus_per_node),
            worker_placements=train_workers,
            # num_gpus=0 (not 0.01): partial-overlap rlix-mode. The ROLL
            # node-PG bundle has GPU=N capacity reserved for the whole
            # cluster; train actors use CUDA_VISIBLE_DEVICES set from
            # WorkerPlacement.gpu_ids and do NOT take a Ray bundle
            # reservation, leaving full GPU capacity for the inference
            # engines to also schedule on the same bundle.
            num_gpus_per_actor=0,
            role="actor",
            # Match miles standalone (placement_group.py:192): ref model
            # required when KL loss / KL coefficient are active. Without
            # this, train_actor's ref_log_probs computation is skipped
            # (gated on `"ref" in self.weights_backuper.backup_tags`)
            # and policy_loss_function crashes with
            # ``torch.cat(None, dim=0)`` because batch["ref_log_probs"]
            # is None.
            with_ref=bool(
                getattr(miles_args, "use_kl_loss", False)
                or float(getattr(miles_args, "kl_coef", 0.0)) != 0.0
            ),
        )
        self._run_async(self._train_group.init())

        # Step 3.5: actor.init() ends with sleep() when offload_train=True,
        # which pauses tms and unmaps the GPU weights. The HF weight gather
        # in build_cpu_bucket_cache needs weights on GPU, so wake them up
        # before the bucket build.
        if bool(getattr(miles_args, "offload_train", False)):
            logger.info("[MilesPipeline] phaseA step3.5: wake_up (post-init pre-bucket-build)")
            self._run_async(self._train_group.onload())

        # Step 4: build CPU bucket cache for the BASE step (-1).
        self._run_async(self._train_group.build_cpu_bucket_cache(step=-1))

        # Step 5: train offload — release overlap GPU.
        logger.info("[MilesPipeline] phaseA step5: offload start")
        self._run_async(self._train_group.offload())
        logger.info("[MilesPipeline] phaseA step5: offload done")

        # Step 6.5: collect cache_owner role.
        logger.info("[MilesPipeline] phaseA step6.5: collect_cache_owner_roles start")
        roles = self._train_group.collect_cache_owner_roles()
        logger.info("[MilesPipeline] phaseA step6.5: collect_cache_owner_roles done roles=%s", roles)
        owners = [(r, h) for (r, is_owner, h) in roles if is_owner]
        if len(owners) != 1:
            raise RuntimeError(
                f"F18 cache_owner uniqueness violated: got {len(owners)} owners, "
                f"expected exactly 1"
            )
        self._cache_owner_actor = owners[0][1]

        # Step 6.6 (relaxed): publish cache_ready_step BEFORE phase B so
        # any concurrent runtime expand against this pipeline's coordinator
        # finds the base version available. register_model_update_resources
        # + bootstrap_active_engines run in phase B once the manager is
        # constructed.
        logger.info("[MilesPipeline] phaseA step6.6: publish_cache_ready_step(-1) start")
        ray.get(self._coordinator_handle.publish_cache_ready_step.remote(-1))
        logger.info("[MilesPipeline] phaseA step6.6: publish_cache_ready_step(-1) done")

        # P1-7: release actor_train BEFORE requesting actor_infer so the
        # scheduler can satisfy actor_infer when the GPU pool is sized
        # for one role at a time. Mirrors the canonical pattern in
        # rlix.pipeline.full_finetune_pipeline (init L308-312).
        # R11-F1: only flip the ledger flag after a SUCCESSFUL release —
        # otherwise shutdown_hard would skip the cluster_id and leak the
        # allocation server-side.
        logger.info("[MilesPipeline] phaseA step7: release actor_train start")
        released = self._notify_release_cluster_gpus(
            cluster_id=self._actor_train_cluster_id, global_step=-1
        )
        logger.info("[MilesPipeline] phaseA step7: release actor_train done released=%s", released)
        if released:
            self._actor_train_allocated = False

    def _init_phase_b_infer(self) -> None:
        """Step 7 — actor_infer side.

        Constructs the rollout manager via the legacy ``pg`` shape (engines
        come up ``active``). The all-shell M11.2 init pattern is deferred:
        receiver-side shell-creation code is not yet wired into
        RolloutManager.expand_engines.
        """
        miles_args = getattr(self._pipeline_config, "miles_args")

        logger.info("[MilesPipeline] phaseB step1: request actor_infer start")
        infer_allocated = self._request_cluster_gpus(
            cluster_id=self._actor_infer_cluster_id,
            priority=Priority.INITIALIZATION,
            global_step=-1,
        )
        logger.info("[MilesPipeline] phaseB step1: request actor_infer done allocated=%s", infer_allocated)
        self._actor_infer_allocated = True
        # R11-F2 fail-fast: confirm the rlix scheduler granted the full
        # infer pool. The downstream _create_placement_group bypasses the
        # scheduler ledger — accepting a subset grant here would let Ray
        # oversubscribe in dual-pipeline.
        expected_infer_count = int(miles_args.rollout_num_gpus)
        if len(set(infer_allocated)) < expected_infer_count:
            raise RuntimeError(
                "scheduler granted %d actor_infer GPUs but pipeline requires "
                "%d (cluster_id=%s); _create_placement_group would bypass the "
                "scheduler ledger and double-count under multi-pipeline. "
                "Wait for full grant or wire the placement_provider path "
                "(M11.2 follow-up)."
                % (len(set(infer_allocated)), expected_infer_count,
                   self._actor_infer_cluster_id)
            )

        # Construct legacy `pg` for the inference pool. The standalone
        # MILES path uses _create_placement_group directly; we mirror it
        # here so start_rollout_servers can do its normal thing.
        #
        # R11-F2 (M11.1 -> M11.2 deferred): _create_placement_group calls
        # ``ray.util.placement_group(...)`` against Ray's free GPU pool
        # directly, independent of the rlix scheduler ledger. In M11.1
        # single-pipeline this is benign because the scheduler grant for
        # actor_infer exhausts the pool, so Ray's PG and the rlix ledger
        # reference the same physical GPUs. For M11.2 dual-pipeline this
        # bypass would let Ray oversubscribe a pool the rlix scheduler is
        # already serving to a peer pipeline. The clean fix is to route
        # through ``self._placement_provider.get_all_rollout_engine_placements()``
        # but that requires shaping the returned WorkerPlacement list back
        # into the (pg, reordered_bundle_indices, reordered_gpu_ids) tuple
        # ``RolloutManager.__init__`` consumes — work tracked as M11.2
        # follow-up. For now, fail-fast if the scheduler grant returns a
        # subset that does NOT cover the full infer pool, so we can't
        # silently mis-route under multi-pipeline contention.
        # Reuse ROLL's node-PG via the placement_provider for the inference
        # engines. The provider returns per-engine WorkerPlacement entries
        # all sharing the same node-PG (single bundle, GPU=N capacity);
        # train actors with num_gpus=0 leave the bundle capacity free for
        # the engines. NO new PG is created and the node-PG is NOT
        # removed; train actors stay alive for the loop.
        logger.info("[MilesPipeline] phaseB step2: get_all_rollout_engine_placements start")
        rollout_workers = self._placement_provider.get_all_rollout_engine_placements()
        self._placement_provider.assert_structural(rollout_workers)
        node_pg = rollout_workers[0].placement_group
        reordered_bundle_indices = [
            wp.bundle_index for wp in rollout_workers for _ in wp.gpu_ids
        ]
        reordered_gpu_ids = [g for wp in rollout_workers for g in wp.gpu_ids]
        legacy_pg = (node_pg, reordered_bundle_indices, reordered_gpu_ids)
        logger.info(
            "[MilesPipeline] phaseB step2: get_all_rollout_engine_placements done "
            "engines=%d bundle_indices=%s gpu_ids=%s",
            len(rollout_workers), reordered_bundle_indices, reordered_gpu_ids,
        )

        # RolloutManager construction. Engines come up `active` via the
        # standard start_rollout_servers flow inside __init__.
        from miles.ray.rollout import RolloutManager

        logger.info("[MilesPipeline] phaseB step3: RolloutManager.remote start")
        self._rollout_manager = RolloutManager.options(
            namespace=self._ray_namespace,
            name=f"miles_rollout_manager_{self._pipeline_id}",
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
        ).remote(miles_args, legacy_pg)
        logger.info("[MilesPipeline] phaseB step3: RolloutManager.remote done")

        logger.info("[MilesPipeline] phaseB step4: get_engine_count start")
        engine_count = int(ray.get(self._rollout_manager.get_engine_count.remote()))
        logger.info("[MilesPipeline] phaseB step4: get_engine_count done count=%d", engine_count)
        self._declared_engine_count = engine_count

        # Wire the train group to the rollout manager so rank 0 can push
        # train_parallel_config into RolloutManager during setup. Standalone
        # does this in create_training_models; rlix mode runs it here, after
        # the rollout manager exists.
        logger.info("[MilesPipeline] phaseB step4b: set_rollout_manager start")
        self._run_async(self._train_group.set_rollout_manager(self._rollout_manager))
        logger.info("[MilesPipeline] phaseB step4b: set_rollout_manager done")

        # F107 / X2: register handles. F22 (relaxed): in M11.1 single-pipeline
        # this happens after the manager exists; the dual-pipeline-shell-init
        # F22 ordering is deferred.
        logger.info("[MilesPipeline] phaseB step5: register_model_update_resources start")
        ray.get(
            self._coordinator_handle.register_model_update_resources.remote(
                cache_owner_actor=self._cache_owner_actor,
                rollout_manager=self._rollout_manager,
            )
        )
        logger.info("[MilesPipeline] phaseB step5: register_model_update_resources done")
        # All engines came up active via start_rollout_servers; bootstrap
        # the full set as the active group. X3 / F19 still applies — this
        # is the SINGLE bootstrap call.
        full_engine_indices = list(range(engine_count))
        logger.info("[MilesPipeline] phaseB step6: bootstrap_active_engines start")
        ray.get(
            self._coordinator_handle.bootstrap_active_engines.remote(
                frozenset(full_engine_indices)
            )
        )
        logger.info("[MilesPipeline] phaseB step6: bootstrap_active_engines done")

        active = ray.get(self._coordinator_handle.get_active_engines.remote())
        if set(active) != set(full_engine_indices):
            raise RuntimeError(
                f"post-INIT active set mismatch: declared={full_engine_indices}, "
                f"got={sorted(active)}"
            )

        # Push base v=-1 weights to the active engines now that the cache
        # is built (Phase A Step 4) and the active set is known. Driver no
        # longer needs to call this — keeps init self-contained.
        logger.info("[MilesPipeline] phaseB step7: sync_base_weights_to_active(-1) start")
        ray.get(self._coordinator_handle.sync_base_weights_to_active.remote(-1))
        logger.info("[MilesPipeline] phaseB step7: sync_base_weights_to_active(-1) done")

        # Step 8: transition actor_infer from INITIALIZATION → GENERATION
        # priority. INITIALIZATION is the highest priority and the
        # scheduler will NEVER preempt it; subsequent ACTOR_TRAINING
        # requests would block forever. Release at INITIALIZATION then
        # re-request at GENERATION (preemptable). Mirrors the canonical
        # full_finetune_pipeline pattern (init → release → runtime
        # re-request at GENERATION).
        logger.info("[MilesPipeline] phaseB step8: actor_infer init→GENERATION transition start")
        released_infer = self._notify_release_cluster_gpus(
            cluster_id=self._actor_infer_cluster_id, global_step=-1
        )
        if released_infer:
            self._actor_infer_allocated = False
        # The scheduler's gap-ratio planner skips GENERATION clusters with
        # both progress=0 AND step_target_estimate=None, hanging the request.
        # Estimate per-rollout trajectory demand from miles_args:
        #   rollout_batch_size × n_samples_per_prompt
        gen_step_target_estimate = max(
            int(getattr(miles_args, "rollout_batch_size", 1))
            * int(getattr(miles_args, "n_samples_per_prompt", 1)),
            1,
        )
        regranted = self._request_cluster_gpus(
            cluster_id=self._actor_infer_cluster_id,
            priority=Priority.GENERATION,
            global_step=-1,
            step_target_estimate=gen_step_target_estimate,
        )
        self._actor_infer_allocated = True
        logger.info(
            "[MilesPipeline] phaseB step8: actor_infer init→GENERATION done released=%s allocated=%s step_target_estimate=%d",
            released_infer, regranted, gen_step_target_estimate,
        )

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------

    def _wait_for_overlap_engines_offloaded(self, allocated_train_gpus, *, timeout_s: float = 60.0) -> None:
        """After scheduler grants actor_train, poll the rollout manager
        until the engines on overlap GPUs have transitioned to ``offloaded``
        AND the OS-reported GPU memory is actually free. SGLang's HTTP
        ``/release_memory_occupation`` 200 OK + state="offloaded" do not
        by themselves guarantee the CUDA driver has returned the memory
        to the OS pool — the wake_up in the next-process train actor
        would then OOM. Verify actual GPU mem free by parsing
        ``nvidia-smi --query-gpu=memory.free`` on the same node, since
        miles' single-node smoke topology has driver+actors+engines all
        on the head node and ``CUDA_VISIBLE_DEVICES`` is the per-actor
        slice of the shared physical pool.
        """
        rollout_manager = getattr(self, "_rollout_manager", None)
        if rollout_manager is None:
            return
        miles_args = self._pipeline_config.miles_args
        per_engine = max(int(getattr(miles_args, "rollout_num_gpus_per_engine", 1)), 1)
        # M11.2 multi-pipeline fix: physical GPU IDs are absolute machine
        # indices, but the RolloutManager uses LOCAL engine indices
        # (0..N-1) within this pipeline's infer pool. Convert physical →
        # local by subtracting the infer pool's first physical GPU.
        # M11.1 single-pipeline pool was [0..rollout_num_gpus-1], so
        # ``g // per_engine`` happened to equal the local engine index;
        # for M11.2 P2 (pool [2,3]), ``2 // 1 = 2`` is wrong (no engine
        # at local index 2). Read the infer mapping from
        # cluster_device_mappings, fall back to range(rollout_num_gpus)
        # for backward compat.
        cluster_mappings = (
            getattr(self._pipeline_config, "cluster_device_mappings", None) or {}
        )
        infer_mapping = list(
            cluster_mappings.get(
                "actor_infer", list(range(int(miles_args.rollout_num_gpus)))
            )
        )
        infer_first = min(infer_mapping) if infer_mapping else 0
        target_indices = sorted(
            {(int(g) - infer_first) // per_engine for g in allocated_train_gpus}
        )
        target_gpu_ids = sorted(set(int(g) for g in allocated_train_gpus))
        if not target_indices:
            return

        # Phase 1: wait for engine state transitions to "offloaded" / "shell".
        deadline = time.time() + float(timeout_s)
        uniq: set = set()
        while time.time() < deadline:
            try:
                states = ray.get(
                    rollout_manager.get_engine_states.remote(target_indices)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_wait_for_overlap_engines_offloaded: get_engine_states failed: %r", exc
                )
                return
            uniq = {states.get(i, "?") for i in target_indices}
            if uniq.issubset({"offloaded", "shell"}):
                logger.info(
                    "_wait_for_overlap_engines_offloaded: engines %s reached state=%s",
                    target_indices, uniq,
                )
                break
            time.sleep(0.1)
        else:
            logger.warning(
                "_wait_for_overlap_engines_offloaded: state timeout after %.1fs; "
                "engines %s still in state=%r",
                timeout_s, target_indices, uniq,
            )

        # Phase 2: probe nvidia-smi for OS-level free memory on the
        # overlap GPU IDs. The train actor needs model weights plus
        # activation headroom before wake_up. The default 20 GB threshold
        # is the validated Qwen2.5-0.5B smoke setting; larger models can
        # override it with MILES_MIN_FREE_GPU_MEM_GB without changing the
        # driver CLI surface.
        target_free_gb = parse_env_positive_float("MILES_MIN_FREE_GPU_MEM_GB", 20.0)
        deadline2 = time.time() + float(timeout_s)
        last_min_free_gb: Optional[float] = None
        while time.time() < deadline2:
            min_free_gb = self._probe_min_free_gpu_mem_gb(target_gpu_ids)
            if min_free_gb is None:
                # nvidia-smi unavailable or unparseable — fall back to a
                # short grace sleep so we don't spin forever.
                logger.warning(
                    "_wait_for_overlap_engines_offloaded: nvidia-smi probe unavailable; "
                    "falling back to 3s grace sleep"
                )
                time.sleep(3.0)
                return
            last_min_free_gb = min_free_gb
            if min_free_gb >= target_free_gb:
                logger.info(
                    "_wait_for_overlap_engines_offloaded: OS-level GPU mem free "
                    "min=%.2f GB across overlap GPUs %s (target=%.1f GB)",
                    min_free_gb, target_gpu_ids, target_free_gb,
                )
                return
            time.sleep(0.5)
        logger.warning(
            "_wait_for_overlap_engines_offloaded: free-mem timeout after %.1fs; "
            "min_free_gb=%.2f below %.1f GB target on GPUs %s — wake_up may OOM",
            timeout_s,
            last_min_free_gb if last_min_free_gb is not None else float("nan"),
            target_free_gb,
            target_gpu_ids,
        )

    @staticmethod
    def _probe_min_free_gpu_mem_gb(gpu_ids: list[int]) -> Optional[float]:
        """Return the minimum free GPU memory (GB) across ``gpu_ids`` as
        reported by ``nvidia-smi``. Returns ``None`` if nvidia-smi is
        not available or output cannot be parsed.
        """
        if not gpu_ids:
            return None
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={','.join(str(g) for g in gpu_ids)}",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                timeout=5.0,
            ).decode("utf-8", errors="replace")
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug("nvidia-smi probe failed: %r", exc)
            return None
        free_mibs: list[float] = []
        for line in out.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                free_mibs.append(float(line))
            except ValueError:
                continue
        if not free_mibs:
            return None
        return min(free_mibs) / 1024.0

    def _before_training(self, step: int) -> None:
        if not self._initialized:
            raise RuntimeError("MilesPipeline._before_training before initialize")
        miles_args = self._pipeline_config.miles_args
        train_count = int(miles_args.actor_num_nodes) * int(
            miles_args.actor_num_gpus_per_node
        )
        allocated = self._request_cluster_gpus(
            cluster_id=self._actor_train_cluster_id,
            priority=Priority.ACTOR_TRAINING,
            global_step=int(step),
        )
        self._actor_train_allocated = True
        # Race-free wait: scheduler awaits resize_infer (shrink_engines)
        # but SGLang's release_memory_occupation 200 OK is async wrt the
        # CUDA caching allocator. Poll the engine states so wake_up never
        # sees the old occupation (Rank 0 saw 29.39 GB used previously).
        self._wait_for_overlap_engines_offloaded(allocated)
        if not allocated or len(set(allocated)) < train_count:
            raise RuntimeError(
                f"_before_training: scheduler returned undersized actor_train "
                f"allocation ({allocated}); expected {train_count} GPUs"
            )
        # Do NOT call train_group.onload() here: when args.offload_train is
        # set, MegatronTrainRayActor.train() already wakes up internally
        # (see miles/backends/megatron_utils/actor.py:357). Adding an
        # onload here causes a double wake_up — the second call hits
        # `[torch_memory_saver.cpp] Cannot resume allocation that is not
        # paused` because tms regions are already resumed. Leave the
        # actor offloaded; rlix_train_loop will dispatch train() which
        # wakes up safely.

    def _after_training(self, step: int) -> None:
        if not self._initialized:
            raise RuntimeError("MilesPipeline._after_training before initialize")
        # Build the new CPU bucket cache for this step BEFORE offload —
        # named_params_and_buffers needs GPU weights resident.
        self._run_async(self._train_group.build_cpu_bucket_cache(step=int(step)))
        self._run_async(self._train_group.offload())
        # Coordinator drives sync_base_weights_to_active under its
        # resize lock; pipeline NEVER calls service / finalize /
        # set_weight_version directly (F05).
        ray.get(
            self._coordinator_handle.sync_base_weights_to_active.remote(int(step))
        )
        # Release the actor_train allocation back to the scheduler so
        # other pipelines can step. R11-F1: only flip the ledger flag
        # after a SUCCESSFUL release.
        released = self._notify_release_cluster_gpus(
            cluster_id=self._actor_train_cluster_id, global_step=int(step)
        )
        if released:
            self._actor_train_allocated = False

    # ------------------------------------------------------------------
    # M4 minimal hard cleanup
    # ------------------------------------------------------------------

    def shutdown_hard(self) -> None:
        # Order: kill SGLang server trees first (CUDA context), then train
        # actors, then release per-cluster_id ledger.
        if self._rollout_manager is not None:
            try:
                # R11-F3: bound the outer ray.get so a wedged manager
                # cannot block ledger release. Inner per-engine
                # shutdown.remote already has its own 10s timeout
                # (rollout.py:1013); 30s here is the worst-case bound
                # for N parallel engine shutdowns plus monitor stop.
                ray.get(
                    self._rollout_manager.shutdown_hard.remote(),
                    timeout=30.0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("shutdown_hard: rollout_manager failed: %r", exc)
            try:
                ray.kill(self._rollout_manager, no_restart=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "shutdown_hard: rollout_manager ray.kill failed: %r", exc
                )
            self._rollout_manager = None

        if self._train_group is not None:
            for h in getattr(self._train_group, "_actor_handles", []):
                if h is None:
                    continue
                try:
                    ray.kill(h, no_restart=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "shutdown_hard: train actor ray.kill failed: %r", exc
                    )
            self._train_group = None

        # P1-3 + P1-7: release each allocated cluster_id explicitly via
        # the canonical notify_release_gpus RPC. Use ray.get so the
        # release is committed before this method returns (P2-13 from
        # the cross-cutting review — fire-and-forget could be lost if
        # the actor is ray.kill'd immediately after).
        scheduler = self._get_scheduler_handle(silent_on_missing=True)
        if scheduler is not None:
            for cluster_id, was_allocated in (
                (self._actor_train_cluster_id, self._actor_train_allocated),
                (self._actor_infer_cluster_id, self._actor_infer_allocated),
            ):
                if not was_allocated:
                    continue
                try:
                    ray.get(
                        scheduler.notify_release_gpus.remote(
                            cluster_id=cluster_id, global_step=None
                        ),
                        timeout=10.0,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "shutdown_hard: notify_release_gpus(%s) failed: %r",
                        cluster_id,
                        exc,
                    )
            self._actor_train_allocated = False
            self._actor_infer_allocated = False

    def dispose(self) -> None:
        self.shutdown_hard()

    # ------------------------------------------------------------------
    # Driver-side accessors (post-iter-27 wiring)
    # ------------------------------------------------------------------

    def get_train_group(self):
        if not self._initialized:
            raise RuntimeError("MilesPipeline.get_train_group before initialize")
        return self._train_group

    def get_rollout_manager(self):
        if not self._initialized:
            raise RuntimeError("MilesPipeline.get_rollout_manager before initialize")
        return self._rollout_manager

    def get_declared_engine_count(self) -> int:
        if not self._initialized:
            raise RuntimeError("MilesPipeline.get_declared_engine_count before initialize")
        return int(self._declared_engine_count)

    def is_initialized(self) -> bool:
        return bool(self._initialized)

    # Public aliases for the underscore-prefixed step hooks. Drivers SHOULD
    # call these instead of the underscored variants so the public Ray
    # surface stays stable.
    def before_training(self, step: int) -> None:
        return self._before_training(step)

    def after_training(self, step: int) -> None:
        return self._after_training(step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _request_cluster_gpus(
        self,
        *,
        cluster_id: str,
        priority,
        global_step: int,
        step_target_estimate: int | None = None,
    ) -> list[int]:
        """Block until scheduler allocates GPUs for ``cluster_id``.

        Mirrors rlix.pipeline.full_finetune_pipeline._request_cluster_gpus
        (P1-3 fix — real scheduler API is request_gpus, not the prior
        request_cluster_gpus(pipeline_id=, role=, num_gpus=) signature).

        ``step_target_estimate`` is required for ``priority=GENERATION``
        requests when no per-pipeline progress metric has been published
        yet — the planner's gap-ratio path skips clusters with both
        progress=0 AND step_target_estimate=None, which would block the
        request forever.
        """
        scheduler = self._get_scheduler_handle()
        kwargs = dict(
            cluster_id=str(cluster_id),
            priority=priority,
            global_step=int(global_step),
        )
        if step_target_estimate is not None:
            kwargs["step_target_estimate"] = int(step_target_estimate)
        allocated = ray.get(
            scheduler.request_gpus.remote(**kwargs)
        )
        if not isinstance(allocated, list):
            raise RuntimeError(
                f"scheduler.request_gpus returned non-list: {type(allocated).__name__}"
            )
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(
                f"scheduler allocated empty GPU list for cluster_id={cluster_id!r}"
            )
        return allocated

    def _notify_release_cluster_gpus(self, *, cluster_id: str, global_step) -> bool:
        """Synchronous release. Uses ray.get so the release is committed
        before this method returns (P2-13 fix).

        Returns True iff the scheduler RPC completed successfully — the
        caller MUST consult the return value before flipping any
        per-cluster_id ledger flag (R11-F1 fix). Returning False (or
        receiving None scheduler handle) means the cluster_id may still
        be allocated server-side; ``shutdown_hard`` will retry the
        release if the flag stays True.
        """
        scheduler = self._get_scheduler_handle(silent_on_missing=True)
        if scheduler is None:
            # No scheduler reachable — treat as failure so the caller
            # leaves the ledger flag set and ``shutdown_hard`` can
            # retry once the scheduler is available again.
            return False
        try:
            ray.get(
                scheduler.notify_release_gpus.remote(
                    cluster_id=str(cluster_id),
                    global_step=(int(global_step) if global_step is not None else None),
                ),
                timeout=10.0,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_notify_release_cluster_gpus(%s) failed: %r", cluster_id, exc
            )
            return False

    def _run_async(self, coro):
        """Drive a coroutine to completion from a sync method.

        Per the cross-cutting review (P2-11): MilesPipeline runs as a
        Ray actor (max_concurrency=2). Sync methods are dispatched on a
        threadpool; reused worker threads see closed loops on subsequent
        calls. Use a fresh event loop each invocation to avoid
        DeprecationWarning on 3.12+ and RuntimeError on 3.14+.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _get_scheduler_handle(self, *, silent_on_missing: bool = False):
        if self._scheduler is not None:
            return self._scheduler
        try:
            self._scheduler = get_actor_or_raise(
                SCHEDULER_ACTOR_NAME,
                RLIX_NAMESPACE,
                error_context=(
                    "MilesPipeline requires the central scheduler actor"
                ),
            )
        except Exception:
            if silent_on_missing:
                return None
            raise
        return self._scheduler

    def _build_placement_provider(self, miles_args):
        """Construct a fresh :class:`MilesPlacementProvider`.

        For M11.2 dual-pipeline: prefer ``cluster_device_mappings`` from
        the pipeline_config (forwarded by the driver from the
        ``orchestrator.register_pipeline`` call) over recomputing
        ``range(actor_count)`` / ``range(rollout_num_gpus)``. The fallback
        keeps M11.1 single-pipeline behavior unchanged when the driver
        doesn't set the field.

        Per the cross-cutting review (P1-5): ``__ray_call__.remote(lambda)``
        is not a Ray API. RollResourceManagerProxy is documented as a
        read-only proxy to a shared singleton ResourceManager actor —
        constructing multiple proxies pointing at the same singleton is
        the intended usage, so we just construct one here.
        """
        from miles.ray.placement_provider import MilesPlacementProvider
        from roll.distributed.scheduler.resource_manager import (
            RollResourceManagerProxy,
        )

        proxy = RollResourceManagerProxy(
            num_gpus_per_node=int(miles_args.num_gpus_per_node)
        )

        cluster_mappings = getattr(
            self._pipeline_config, "cluster_device_mappings", None
        ) or {}
        train_mapping = cluster_mappings.get("actor_train")
        infer_mapping = cluster_mappings.get("actor_infer")
        if train_mapping is None:
            train_mapping = list(
                range(
                    int(miles_args.actor_num_nodes)
                    * int(miles_args.actor_num_gpus_per_node)
                )
            )
        if infer_mapping is None:
            infer_mapping = list(range(int(miles_args.rollout_num_gpus)))

        logger.info(
            "[MilesPipeline] _build_placement_provider train=%s infer=%s "
            "(from pipeline_config.cluster_device_mappings: %s)",
            train_mapping, infer_mapping, bool(cluster_mappings),
        )
        return MilesPlacementProvider(
            resource_manager_proxy=proxy,
            train_device_mapping=list(train_mapping),
            infer_device_mapping=list(infer_mapping),
            rollout_num_gpus_per_engine=int(miles_args.rollout_num_gpus_per_engine),
            num_gpus_per_node=int(miles_args.num_gpus_per_node),
        )


__all__ = ["MilesPipeline"]
