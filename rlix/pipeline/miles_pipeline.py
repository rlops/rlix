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
            num_gpus_per_actor=0.01,
            role="actor",
            with_ref=False,
        )
        self._run_async(self._train_group.init())

        # Step 4: build CPU bucket cache for the BASE step (-1).
        self._run_async(self._train_group.build_cpu_bucket_cache(step=-1))

        # Step 5: train offload — release overlap GPU.
        self._run_async(self._train_group.offload())

        # Step 6.5: collect cache_owner role.
        roles = self._train_group.collect_cache_owner_roles()
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
        ray.get(self._coordinator_handle.publish_cache_ready_step.remote(-1))

        # P1-7: release actor_train BEFORE requesting actor_infer so the
        # scheduler can satisfy actor_infer when the GPU pool is sized
        # for one role at a time. Mirrors the canonical pattern in
        # rlix.pipeline.full_finetune_pipeline (init L308-312).
        self._notify_release_cluster_gpus(
            cluster_id=self._actor_train_cluster_id, global_step=-1
        )
        self._actor_train_allocated = False

    def _init_phase_b_infer(self) -> None:
        """Step 7 — actor_infer side.

        Constructs the rollout manager via the legacy ``pg`` shape (engines
        come up ``active``). The all-shell M11.2 init pattern is deferred:
        receiver-side shell-creation code is not yet wired into
        RolloutManager.expand_engines.
        """
        miles_args = getattr(self._pipeline_config, "miles_args")

        self._request_cluster_gpus(
            cluster_id=self._actor_infer_cluster_id,
            priority=Priority.INITIALIZATION,
            global_step=-1,
        )
        self._actor_infer_allocated = True

        # Construct legacy `pg` for the inference pool. The standalone
        # MILES path uses _create_placement_group directly; we mirror it
        # here so start_rollout_servers can do its normal thing.
        from miles.ray.placement_group import _create_placement_group

        pg, reordered_bundle_indices, reordered_gpu_ids = _create_placement_group(
            int(miles_args.rollout_num_gpus)
        )
        legacy_pg = (pg, reordered_bundle_indices, reordered_gpu_ids)

        # RolloutManager construction. Engines come up `active` via the
        # standard start_rollout_servers flow inside __init__.
        from miles.ray.rollout import RolloutManager

        self._rollout_manager = RolloutManager.options(
            namespace=self._ray_namespace,
            name=f"miles_rollout_manager_{self._pipeline_id}",
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
        ).remote(miles_args, legacy_pg)

        engine_count = int(ray.get(self._rollout_manager.get_engine_count.remote()))
        self._declared_engine_count = engine_count

        # F107 / X2: register handles. F22 (relaxed): in M11.1 single-pipeline
        # this happens after the manager exists; the dual-pipeline-shell-init
        # F22 ordering is deferred.
        ray.get(
            self._coordinator_handle.register_model_update_resources.remote(
                cache_owner_actor=self._cache_owner_actor,
                rollout_manager=self._rollout_manager,
            )
        )
        # All engines came up active via start_rollout_servers; bootstrap
        # the full set as the active group. X3 / F19 still applies — this
        # is the SINGLE bootstrap call.
        full_engine_indices = list(range(engine_count))
        ray.get(
            self._coordinator_handle.bootstrap_active_engines.remote(
                frozenset(full_engine_indices)
            )
        )

        active = ray.get(self._coordinator_handle.get_active_engines.remote())
        if set(active) != set(full_engine_indices):
            raise RuntimeError(
                f"post-INIT active set mismatch: declared={full_engine_indices}, "
                f"got={sorted(active)}"
            )

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------

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
        if not allocated or len(set(allocated)) < train_count:
            raise RuntimeError(
                f"_before_training: scheduler returned undersized actor_train "
                f"allocation ({allocated}); expected {train_count} GPUs"
            )
        self._run_async(self._train_group.onload())

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
        # other pipelines can step.
        self._notify_release_cluster_gpus(
            cluster_id=self._actor_train_cluster_id, global_step=int(step)
        )
        self._actor_train_allocated = False

    # ------------------------------------------------------------------
    # M4 minimal hard cleanup
    # ------------------------------------------------------------------

    def shutdown_hard(self) -> None:
        # Order: kill SGLang server trees first (CUDA context), then train
        # actors, then release per-cluster_id ledger.
        if self._rollout_manager is not None:
            try:
                ray.get(self._rollout_manager.shutdown_hard.remote())
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
    # Helpers
    # ------------------------------------------------------------------

    def _request_cluster_gpus(
        self,
        *,
        cluster_id: str,
        priority,
        global_step: int,
    ) -> list[int]:
        """Block until scheduler allocates GPUs for ``cluster_id``.

        Mirrors rlix.pipeline.full_finetune_pipeline._request_cluster_gpus
        (P1-3 fix — real scheduler API is request_gpus, not the prior
        request_cluster_gpus(pipeline_id=, role=, num_gpus=) signature).
        """
        scheduler = self._get_scheduler_handle()
        allocated = ray.get(
            scheduler.request_gpus.remote(
                cluster_id=str(cluster_id),
                priority=priority,
                global_step=int(global_step),
            )
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

    def _notify_release_cluster_gpus(self, *, cluster_id: str, global_step) -> None:
        """Synchronous release. Uses ray.get so the release is committed
        before this method returns (P2-13 fix)."""
        scheduler = self._get_scheduler_handle(silent_on_missing=True)
        if scheduler is None:
            return
        try:
            ray.get(
                scheduler.notify_release_gpus.remote(
                    cluster_id=str(cluster_id),
                    global_step=(int(global_step) if global_step is not None else None),
                ),
                timeout=10.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_notify_release_cluster_gpus(%s) failed: %r", cluster_id, exc
            )

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

        Per the cross-cutting review (P1-5): `__ray_call__.remote(lambda)`
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
        return MilesPlacementProvider(
            resource_manager_proxy=proxy,
            train_device_mapping=list(
                range(
                    int(miles_args.actor_num_nodes)
                    * int(miles_args.actor_num_gpus_per_node)
                )
            ),
            infer_device_mapping=list(range(int(miles_args.rollout_num_gpus))),
            rollout_num_gpus_per_engine=int(miles_args.rollout_num_gpus_per_engine),
            num_gpus_per_node=int(miles_args.num_gpus_per_node),
        )


__all__ = ["MilesPipeline"]
