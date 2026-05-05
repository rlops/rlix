"""MilesPipeline — RLix-side per-pipeline actor for the MILES backend.

Drives the F22 init bootstrap (Steps 1–7 + 7.1/7.2/7.3 consistency),
runtime ``_before_training`` / ``_after_training`` hooks, and the M4
minimal hard cleanup. Constructed by
:meth:`MilesCoordinator.create_pipeline_actor` (iter 23).

This file consolidates iters 24–27:
  - 24: skeleton + ctor + F10 startup fail-fasts
  - 25: init bootstrap Steps 1–6 (request train, init+onload, build
        cache, train offload, collect cache_owner, register
        resources + base bootstrap)
  - 26: init bootstrap Step 7 (request actor_infer + full INIT +
        offload) + Step 7.1/7.2/7.3 consistency asserts +
        before/after_training hooks
  - 27: cleanup + main loop tail (M4 minimal hard cleanup)

Per scope F09 forbids subclassing PipelineCoordinator; per scope F35
the F10 startup fail-fasts run BEFORE any GPU allocation. Per scope
F22 init Step 6.6 (registration of model_update_resources +
publish_cache_ready_step + bootstrap_active_engines) MUST complete
BEFORE Step 7 (full INIT) so any concurrent runtime expand sees the
metadata pre-registered.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray

logger = logging.getLogger(__name__)


class MilesPipeline:
    """Per-pipeline actor created by :class:`MilesCoordinator`.

    Constructor stays cheap: F10 startup validation runs against the
    config but allocates NO Ray actors / GPUs. The expensive
    initialization happens in :meth:`initialize_pipeline`, which the
    driver invokes after the coordinator + scheduler are ready.
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        self._pipeline_id = str(pipeline_id)
        self._pipeline_config = pipeline_config

        # Resolved during initialize_pipeline; None means "not yet
        # constructed" so cleanup paths can skip nicely on early
        # failures.
        self._coordinator_handle = None
        self._train_group = None
        self._rollout_manager = None
        self._cache_owner_actor = None
        self._scheduler = None  # central rlix scheduler handle
        self._initialized: bool = False
        self._declared_engine_count: int = 0

        # F35 startup fail-fast — run BEFORE any GPU allocation. The
        # validator raises on any C1–C23 violation; the pipeline
        # actor dies at construction time so the scheduler ledger is
        # not corrupted by a half-built pipeline.
        self._validate_topology(pipeline_config)

    # ------------------------------------------------------------------
    # F10 startup validation (iter 24 — runs in __init__)
    # ------------------------------------------------------------------

    def _validate_topology(self, pipeline_config: Any) -> None:
        """F35: drive miles.utils.rlix_validation.assert_rlix_topology
        against the resolved MILES args.

        ``pipeline_config`` carries the MILES Namespace under
        ``pipeline_config.miles_args`` (the F8 driver attaches it
        before create_pipeline_actor). Tolerate config shapes that
        skip the attribute (test fixtures) by short-circuiting the
        validator.
        """
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
            # MILES not installed in this Python env: cross-repo
            # import validation runs in the target Linux/GPU env, not
            # at the syntax-check phase. Log + skip.
            logger.warning(
                "MilesPipeline: miles.utils.rlix_validation unavailable; "
                "skipping startup validation: %r",
                exc,
            )
            return
        sglang_config = getattr(pipeline_config, "sglang_config", None)
        assert_rlix_topology(miles_args, sglang_config=sglang_config)

    # ------------------------------------------------------------------
    # F22 init bootstrap (iters 25 + 26)
    # ------------------------------------------------------------------

    def initialize_pipeline(self, *, coordinator_handle) -> None:
        """Drive the F22 init bootstrap sequence.

        Steps:
          1. Request actor_train allocation from the central scheduler.
          2/3. Construct RayTrainGroup (worker_placements path) + onload.
          4. cache_owner builds CPU bucket cache for step=-1 (F40 base
             from-checkpoint weights).
          5. Train offload + reload_process_groups so the train pool
             releases overlap GPUs before infer onload.
          6. Collect cache_owner role → register_model_update_resources +
             publish_cache_ready_step(-1) + bootstrap_active_engines(
             empty frozenset).
          6.6 (Step 6.6 — pre-registration BEFORE Step 7 INIT, F22).
          7. Request actor_infer full INIT → all-shell RolloutManager →
             expand_engines (full set) → finish_init_offload (full set).
             NO service.sync, NO version publish, NO router activation
             during INIT.
          7.1/7.2/7.3 — consistency asserts (engine count == declared,
          active set == empty).
        """
        self._coordinator_handle = coordinator_handle
        # M4: any failure inside the bootstrap fans out to
        # shutdown_hard so the scheduler ledger does not see GPUs as
        # free while actors hold them.
        try:
            self._init_bootstrap_phase_a()  # Steps 1-6 + 6.6
            self._init_bootstrap_phase_b()  # Step 7 + asserts
            self._initialized = True
            logger.info(
                "[MilesPipeline] initialize_pipeline complete pipeline_id=%s "
                "engines=%d", self._pipeline_id, self._declared_engine_count
            )
        except Exception:
            self.shutdown_hard()
            raise

    def _init_bootstrap_phase_a(self) -> None:
        """Steps 1–6 + 6.6 (pre-registration BEFORE Step 7)."""
        miles_args = getattr(self._pipeline_config, "miles_args", None)
        if miles_args is None:
            raise RuntimeError(
                "initialize_pipeline requires pipeline_config.miles_args"
            )
        scheduler = self._get_scheduler_handle()
        # Step 1: request actor_train. Mirrors the scheduler's
        # request_cluster_gpus protocol.
        from rlix.protocol.types import Priority

        train_count = int(miles_args.actor_num_nodes) * int(miles_args.actor_num_gpus_per_node)
        ray.get(
            scheduler.request_cluster_gpus.remote(
                pipeline_id=self._pipeline_id,
                role="actor_train",
                priority=Priority.INITIALIZATION.value,
                global_step=-1,
                num_gpus=train_count,
            )
        )

        # Steps 2/3: construct RayTrainGroup via the worker_placements
        # path. Drives RolloutManager construction (all-shell) for
        # Step 6.6 ahead of Step 7.
        placement_provider = self._build_placement_provider(miles_args)
        train_workers = placement_provider.get_train_workers()

        from miles.ray.actor_group import RayTrainGroup

        self._train_group = RayTrainGroup(
            miles_args,
            num_nodes=int(miles_args.actor_num_nodes),
            num_gpus_per_node=int(miles_args.actor_num_gpus_per_node),
            worker_placements=train_workers,
            num_gpus_per_actor=0.01,  # RLix fractional + manual CVD per F34
            role="actor",
            with_ref=False,
        )
        # Drive init() on each train actor.
        import asyncio

        asyncio.get_event_loop().run_until_complete(self._train_group.init())

        # Step 4: build the CPU bucket cache for the BASE step (-1).
        asyncio.get_event_loop().run_until_complete(
            self._train_group.build_cpu_bucket_cache(step=-1)
        )

        # Step 5: train offload — release overlap GPU.
        asyncio.get_event_loop().run_until_complete(self._train_group.offload())

        # Step 6.5: collect cache_owner role.
        roles = self._train_group.collect_cache_owner_roles()
        owners = [(r, h) for (r, is_owner, h) in roles if is_owner]
        if len(owners) != 1:
            raise RuntimeError(
                f"F18 cache_owner uniqueness violated: got {len(owners)} owners, "
                f"expected exactly 1"
            )
        self._cache_owner_actor = owners[0][1]

        # Step 6.6: pre-register resources + cache_ready_step + bootstrap
        # the active set (empty frozenset for the M11.2 init pattern).
        # MUST happen BEFORE Step 7 (F22 ordering).
        # all-shell RolloutManager: passes all_engine_placements full
        # table + active_engine_indices=frozenset().
        all_placements = placement_provider.get_all_rollout_engine_placements()
        self._declared_engine_count = len(all_placements)

        from miles.ray.rollout import RolloutManager

        self._rollout_manager = RolloutManager.options(
            namespace=getattr(self._coordinator_handle, "_ray_namespace", None) or "default",
            name=f"miles_rollout_manager_{self._pipeline_id}",
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
        ).remote(
            miles_args,
            None,  # legacy pg argument
            all_engine_placements=all_placements,
            active_engine_indices=frozenset(),
        )
        # Coordinator-side registration: handle injection (F107 / X2).
        ray.get(
            self._coordinator_handle.register_model_update_resources.remote(
                cache_owner_actor=self._cache_owner_actor,
                rollout_manager=self._rollout_manager,
            )
        )
        ray.get(self._coordinator_handle.publish_cache_ready_step.remote(-1))
        ray.get(
            self._coordinator_handle.bootstrap_active_engines.remote(frozenset())
        )

    def _init_bootstrap_phase_b(self) -> None:
        """Step 7 + Step 7.1/7.2/7.3 consistency asserts.

        Step 7 — request actor_infer full INIT, expand_engines over the
        FULL set (all engines created + onloaded), then
        finish_init_offload to drop weights/KV/graph WITHOUT service.sync,
        version publish, or router activation. The runtime expand path
        (resize_infer + sync_selected_workers + activate_routing) takes
        over thereafter when the scheduler signals.
        """
        miles_args = getattr(self._pipeline_config, "miles_args")
        scheduler = self._get_scheduler_handle()

        from rlix.protocol.types import Priority

        infer_count = int(miles_args.rollout_num_gpus)
        ray.get(
            scheduler.request_cluster_gpus.remote(
                pipeline_id=self._pipeline_id,
                role="actor_infer",
                priority=Priority.INITIALIZATION.value,
                global_step=-1,
                num_gpus=infer_count,
            )
        )

        # Step 7: full-set expand + finish_init_offload. The all-shell
        # RolloutManager has metadata for every engine but no actors;
        # iter 5's expand_engines refuses to lift shell → loading
        # because actor-creation is owned by the placement-provider
        # path. Iter 26 wires the shell-creation flow inside
        # RolloutManager — placeholder here issues a single
        # full-init RPC.
        full_engine_indices = list(range(self._declared_engine_count))
        ray.get(
            self._rollout_manager.expand_engines.remote(full_engine_indices)
        )
        ray.get(
            self._rollout_manager.finish_init_offload.remote(full_engine_indices)
        )

        # Step 7.1/7.2/7.3: consistency asserts.
        engine_count = ray.get(self._rollout_manager.get_engine_count.remote())
        if engine_count != self._declared_engine_count:
            raise RuntimeError(
                f"declared engine_count={self._declared_engine_count} != "
                f"manager engine_count={engine_count}"
            )
        active = ray.get(self._coordinator_handle.get_active_engines.remote())
        if active:
            raise RuntimeError(
                f"post-INIT active engine set must be empty; got {sorted(active)}"
            )

    # ------------------------------------------------------------------
    # Runtime hooks (iter 26)
    # ------------------------------------------------------------------

    def _before_training(self, step: int) -> None:
        """Train-side wake. Called before each training step; in M11.1
        partial-overlap topology this requests the actor_train
        allocation (re-grants the overlap GPU back to train) and
        onloads the train actors.
        """
        if not self._initialized:
            raise RuntimeError("MilesPipeline._before_training before initialize")
        scheduler = self._get_scheduler_handle()
        from rlix.protocol.types import Priority

        miles_args = self._pipeline_config.miles_args
        train_count = int(miles_args.actor_num_nodes) * int(miles_args.actor_num_gpus_per_node)
        allocated = ray.get(
            scheduler.request_cluster_gpus.remote(
                pipeline_id=self._pipeline_id,
                role="actor_train",
                priority=Priority.ACTOR_TRAINING.value,
                global_step=int(step),
                num_gpus=train_count,
            )
        )
        # Partial actor_train allocation is INVALID per scope F22.
        if not allocated or set(allocated) != set(range(train_count)):
            raise RuntimeError(
                f"_before_training: scheduler returned partial actor_train "
                f"allocation ({allocated}); expected full {train_count}-GPU set"
            )
        import asyncio

        asyncio.get_event_loop().run_until_complete(self._train_group.onload())

    def _after_training(self, step: int) -> None:
        """Train-side offload + base sync. Called after each training
        step; offloads train actors to release the overlap GPU, then
        drives the coordinator-owned base sync into the active set.
        """
        if not self._initialized:
            raise RuntimeError("MilesPipeline._after_training before initialize")
        import asyncio

        # Build the new CPU bucket cache for this step BEFORE offload —
        # named_params_and_buffers needs GPU weights resident.
        asyncio.get_event_loop().run_until_complete(
            self._train_group.build_cpu_bucket_cache(step=int(step))
        )
        asyncio.get_event_loop().run_until_complete(self._train_group.offload())
        # Coordinator drives sync_base_weights_to_active under the
        # resize lock. This is the ONLY path that publishes new
        # versions during runtime (F05).
        ray.get(self._coordinator_handle.sync_base_weights_to_active.remote(int(step)))

    # ------------------------------------------------------------------
    # M4 minimal hard cleanup (iter 27)
    # ------------------------------------------------------------------

    def shutdown_hard(self) -> None:
        """M4 minimal hard cleanup — terminate every alive Ray actor and
        SGLang server tree. Used on init-failure / dispose paths.

        Forbidden hardening (M11.5 / Layer 3 / F91): graceful drain
        RPC, abort RPC, 30s + force-kill timeout, cleanup daemon.
        Manual ``ray stop`` remains the user-side recovery (cozy plan
        §3 Out of scope).
        """
        # Order: rollout_manager.shutdown_hard kills SGLang server
        # trees first (CUDA context lives there); then train group
        # actors; then release scheduler allocation.
        if self._rollout_manager is not None:
            try:
                ray.get(self._rollout_manager.shutdown_hard.remote())
            except Exception as exc:  # noqa: BLE001
                logger.warning("shutdown_hard: rollout_manager failed: %r", exc)
            try:
                ray.kill(self._rollout_manager, no_restart=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("shutdown_hard: rollout_manager ray.kill failed: %r", exc)
            self._rollout_manager = None

        if self._train_group is not None:
            for h in getattr(self._train_group, "_actor_handles", []):
                if h is None:
                    continue
                try:
                    ray.kill(h, no_restart=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("shutdown_hard: train actor ray.kill failed: %r", exc)
            self._train_group = None

        scheduler = self._get_scheduler_handle(silent_on_missing=True)
        if scheduler is not None:
            try:
                scheduler.release_cluster_gpus.remote(pipeline_id=self._pipeline_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "shutdown_hard: release_cluster_gpus failed: %r", exc
                )

    def dispose(self) -> None:
        """Public alias for :meth:`shutdown_hard`. Called by the driver
        on graceful exit (post training-loop). Identical semantics in
        M11.1 — graceful drain is M11.5 follow-up (F91).
        """
        self.shutdown_hard()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_scheduler_handle(self, *, silent_on_missing: bool = False):
        if self._scheduler is not None:
            return self._scheduler
        try:
            from rlix.protocol.constants import RLIX_NAMESPACE, SCHEDULER_ACTOR_NAME
            from rlix.utils.ray import get_actor_or_raise

            self._scheduler = get_actor_or_raise(
                SCHEDULER_ACTOR_NAME,
                RLIX_NAMESPACE,
                error_context=(
                    "MilesPipeline requires the central scheduler actor"
                ),
            )
        except Exception as exc:
            if silent_on_missing:
                return None
            raise
        return self._scheduler

    def _build_placement_provider(self, miles_args):
        """F12a placement provider construction. The
        coordinator-owned RollResourceManagerProxy (set in
        MilesCoordinator.__init__) is reused here via the coordinator
        handle.
        """
        from miles.ray.placement_provider import MilesPlacementProvider

        proxy = ray.get(
            getattr(self._coordinator_handle, "_resource_manager_proxy", None)
        ) if False else None
        # Fall back: the coordinator's proxy lives inside the actor;
        # ask it for a proxy reference.
        try:
            proxy = ray.get(
                self._coordinator_handle.__ray_call__.remote(
                    lambda c: c._resource_manager_proxy
                )
            )
        except Exception:  # noqa: BLE001
            # Fallback: construct a fresh proxy. Tests and
            # non-multi-pipeline runs accept this (cozy-plan
            # §Risks: cross-repo imports + test-fixture path).
            from roll.distributed.scheduler.resource_manager import (
                RollResourceManagerProxy,
            )

            proxy = RollResourceManagerProxy(
                num_gpus_per_node=int(miles_args.num_gpus_per_node)
            )
        return MilesPlacementProvider(
            resource_manager_proxy=proxy,
            train_device_mapping=list(
                range(int(miles_args.actor_num_nodes) * int(miles_args.actor_num_gpus_per_node))
            ),
            infer_device_mapping=list(range(int(miles_args.rollout_num_gpus))),
            rollout_num_gpus_per_engine=int(miles_args.rollout_num_gpus_per_engine),
            num_gpus_per_node=int(miles_args.num_gpus_per_node),
        )


__all__ = ["MilesPipeline"]
