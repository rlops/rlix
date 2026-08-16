"""MILES-specific Coordinator implementation.

Per scope F09 (Layer 1 forbidden) MilesCoordinator does NOT subclass
:class:`PipelineCoordinator`: that base's ``__init__`` runs four config
validators (schema / cpu_only_reward / vllm_sleep_level / offload_nccl)
keyed off ROLL config fields that MILES args don't expose. We inherit
from :class:`Coordinator` (abstract) instead and write the body manually.

Iter 21 lands the ctor + state fields + ABC stubs + backend-neutral
methods (report_progress_from_scheduler, clear_progress_stream,
_aggregate_and_emit, _inject_pipeline_env_vars). Iter 22 adds the runtime
methods that drive selective sync. Iter 23 implements resize_infer +
_expand_workers + create_pipeline_actor.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, TypeVar

import ray

from rlix.protocol.coordinator import Coordinator
from rlix.protocol.types import (
    ACTOR_TRAIN_CLUSTER_NAME,
    ActionResponse,
    GENERATION_CLUSTER_NAME,
    PIPELINE_ACTOR_NAME_PREFIX,
    ProgressReport,
    RLIX_NAMESPACE,
    SCHEDULER_ACTOR_NAME,
    get_pipeline_namespace,
)
from rlix.protocol.validation import validate_pipeline_id
from rlix.utils.env import parse_env_positive_float, pipeline_identity_env_vars
from rlix.utils.ray import get_actor_or_raise

logger = logging.getLogger(__name__)

# --- rlix#42 user-selected weight-update transport mode -----------------
# One rlix-side control (plan D4): `cpu_serialize` (default — today's
# behavior, no classification), `broadcast` (strict: ALL engines
# broadcast; fail-fast if any target engine shares a GPU with the train
# pool — NCCL cannot form a group containing the same physical GPU
# twice), `auto` (topology split; the only mode that mixes transports).
_TRANSPORT_MODE_ENV = "RLIX_MILES_UPDATE_TRANSPORT"
_VALID_TRANSPORT_MODES = ("cpu_serialize", "broadcast", "auto")
# Smoke-validity guard (plan C9): when set to "1", registration asserts
# that the pipeline's FULL engine set classifies into BOTH transport
# classes under `auto` — an overlap smoke that cannot produce mixed
# sessions is INVALID and must fail fast rather than pass vacuously.
_ASSERT_MIXED_ENV = "RLIX_ASSERT_MIXED_TRANSPORT"


def _assert_mixed_transport_startup(
    *,
    mode: str,
    pipeline_id: str,
    train_gpu_ids: Optional[List[int]],
    infer_gpu_ids: Optional[List[int]],
    per_engine: int,
) -> None:
    """C9 startup assertion, gated by ``RLIX_ASSERT_MIXED_TRANSPORT=1``.

    Only meaningful for ``auto`` (the mixing mode): classify the full
    engine set and require ≥1 broadcast-classified AND ≥1
    cpu_serialize-classified engine; log the resulting split. Any other
    mode with the env set is a smoke misconfiguration — fail fast.
    """
    if os.environ.get(_ASSERT_MIXED_ENV, "").strip() != "1":
        return
    if mode != "auto":
        raise RuntimeError(
            f"{_ASSERT_MIXED_ENV}=1 requires {_TRANSPORT_MODE_ENV}=auto "
            f"(the only mixing mode); got mode={mode!r}"
        )
    if not infer_gpu_ids:
        raise RuntimeError(
            f"{_ASSERT_MIXED_ENV}=1 requires registered topology (infer_gpu_ids)"
        )
    engine_count = len(infer_gpu_ids) // max(per_engine, 1)
    full_set = set(range(engine_count))
    broadcast = _classify_broadcast_engines(
        target_engine_indices=full_set,
        mode="auto",
        train_gpu_ids=train_gpu_ids,
        infer_gpu_ids=infer_gpu_ids,
        per_engine=per_engine,
    )
    cpu_serialize = full_set - set(broadcast)
    logger.info(
        "[MilesCoordinator] C9 mixed-transport startup assertion pipeline_id=%s "
        "engines=%s broadcast=%s cpu_serialize=%s",
        pipeline_id,
        sorted(full_set),
        sorted(broadcast),
        sorted(cpu_serialize),
    )
    if not broadcast or not cpu_serialize:
        raise RuntimeError(
            f"C9 mixed-transport startup assertion FAILED pipeline_id={pipeline_id}: "
            f"engines={sorted(full_set)} broadcast={sorted(broadcast)} "
            f"cpu_serialize={sorted(cpu_serialize)} — the overlap smoke requires "
            "≥1 engine of EACH class per pipeline; this topology cannot produce "
            "mixed sessions, so the smoke would pass vacuously (INVALID)."
        )


def _resolve_transport_mode() -> str:
    """Parse the transport-mode env once; invalid value = fail-fast
    (plan C5: no silent default on a typo)."""
    raw = os.environ.get(_TRANSPORT_MODE_ENV, "").strip().lower()
    if not raw:
        return "cpu_serialize"
    if raw not in _VALID_TRANSPORT_MODES:
        raise RuntimeError(
            f"{_TRANSPORT_MODE_ENV}={raw!r} is not a valid transport mode; "
            f"expected one of {list(_VALID_TRANSPORT_MODES)}"
        )
    return raw


def _classify_broadcast_engines(
    *,
    target_engine_indices: Set[int],
    mode: str,
    train_gpu_ids: Optional[List[int]],
    infer_gpu_ids: Optional[List[int]],
    per_engine: int,
) -> frozenset:
    """Classify a sync's target engines into the broadcast subset (AC4).

    Engine → physical-GPU mapping follows the manager's contiguous
    layout (cf. ``MilesPipeline._wait_for_overlap_engines_offloaded``):
    local engine ``e`` occupies the ``per_engine`` consecutive GPUs
    starting at ``sorted(infer_gpu_ids)[e * per_engine]``.

    - ``cpu_serialize``: empty set (no classification, no topology
      required — backward-compatible default).
    - ``auto``: engines whose GPU set is disjoint from the train pool →
      broadcast; overlapping engines → cpu_serialize (the only mixing
      mode).
    - ``broadcast``: ALL engines must be broadcast-eligible; any
      colocate target = RuntimeError (plan C5: honored exactly or the
      run refuses to proceed, never a quiet downgrade).
    """
    target = {int(i) for i in target_engine_indices}
    if mode == "cpu_serialize" or not target:
        return frozenset()
    if mode not in _VALID_TRANSPORT_MODES:
        raise RuntimeError(f"unknown transport mode {mode!r}")
    if train_gpu_ids is None or infer_gpu_ids is None:
        raise RuntimeError(
            f"{_TRANSPORT_MODE_ENV}={mode} requires topology at "
            "register_model_update_resources (train_gpu_ids + infer_gpu_ids); "
            "registration did not provide it"
        )
    train = {int(g) for g in train_gpu_ids}
    infer_sorted = sorted(int(g) for g in infer_gpu_ids)
    broadcast: Set[int] = set()
    colocate: Set[int] = set()
    for engine_index in sorted(target):
        lo = engine_index * per_engine
        hi = lo + per_engine
        engine_gpus = infer_sorted[lo:hi]
        if len(engine_gpus) != per_engine:
            raise RuntimeError(
                f"engine {engine_index} maps outside the infer pool "
                f"(pool={infer_sorted}, per_engine={per_engine})"
            )
        if train & set(engine_gpus):
            colocate.add(engine_index)
        else:
            broadcast.add(engine_index)
    if mode == "broadcast" and colocate:
        raise RuntimeError(
            f"{_TRANSPORT_MODE_ENV}=broadcast cannot serve colocate engines "
            f"{sorted(colocate)} (their GPUs intersect the train pool "
            f"{sorted(train)}); NCCL cannot form a group containing the same "
            "physical GPU twice. Use 'auto' (mixed transports) or "
            "'cpu_serialize'."
        )
    return frozenset(broadcast)

_T = TypeVar("_T")

# Default max_concurrency for the MILES pipeline actor (F108 Special C1).
# Plan §Audit Checkpoints leaves the value an evidence-based decision; 2
# matches "before training (initialize_pipeline) + concurrent runtime
# resize_infer requests" without admitting unbounded concurrency. The
# audit checkboxes (Phase 1 / Phase 2) start unchecked.
_MILES_PIPELINE_ACTOR_MAX_CONCURRENCY: int = 2


def _build_pipeline_env_vars(*, pipeline_id: str, ray_namespace: str) -> Dict[str, str]:
    """Mirror PipelineCoordinator's pipeline_identity_env_vars for MILES.

    Sets the env vars the pipeline actor needs to inject into every
    child Ray actor it creates (cluster, RolloutManager, etc.) via
    runtime_env. Reads ``RLIX_CONTROL_PLANE`` from the environment so
    actors inside an existing pipeline preserve the inherited value.
    """
    env_vars = pipeline_identity_env_vars(
        pipeline_id=str(pipeline_id), ray_namespace=str(ray_namespace)
    )
    for key in ("MILES_MAX_RESIDUAL_GPU_MEM_GB",):
        if (value := os.environ.get(key)) is not None:
            env_vars[key] = value
    return env_vars


class MilesCoordinator(Coordinator):
    """RLix coordinator for the MILES backend.

    Constructed by :func:`create_pipeline_actor` upstream (driver). The
    MILES variant injects the cache_owner actor handle + RolloutManager
    handle (F107 / X2) so :class:`MilesModelUpdateService` can be
    constructed lazily without ``ray.get_actor`` lookups.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        pipeline_config: Any,
    ):
        # Manual init — NO super().__init__() (F09 forbidden). The
        # PipelineCoordinator base validates ROLL-shaped config fields
        # MILES args do not expose; copying just the bookkeeping fields
        # we need is cheaper than carrying a config-shape adapter.
        validate_pipeline_id(pipeline_id)
        self._pipeline_id = str(pipeline_id)
        self._ray_namespace = get_pipeline_namespace(pipeline_id)
        self._pipeline_env_vars = _build_pipeline_env_vars(
            pipeline_id=self._pipeline_id, ray_namespace=self._ray_namespace
        )
        self._pipeline_config = pipeline_config
        self._pipeline_actor = None
        self._verify_model_after_sync: bool = bool(
            getattr(pipeline_config, "verify_model_after_sync", False)
        )

        # F107 / X2: model-update resources injected by
        # register_model_update_resources (iter 22). Lazy
        # MilesModelUpdateService construction reads these.
        self._model_update_service = None
        self._model_update_resources: dict[str, Any] = {}

        # Resource manager proxy for placement-group allocation.
        # Constructed once and shared with the placement provider in
        # MilesPipeline (iter 24+).
        try:
            from roll.distributed.scheduler.resource_manager import RollResourceManagerProxy

            self._resource_manager_proxy = RollResourceManagerProxy(
                num_gpus_per_node=getattr(pipeline_config, "num_gpus_per_node", 8)
            )
            self._resource_manager_node0_pg = self._resource_manager_proxy.node2pg.get(0)
        except Exception as exc:  # noqa: BLE001
            # Tests / non-Ray environments may construct the
            # coordinator without a live ResourceManager. Defer hard
            # failure to actual usage.
            logger.warning("MilesCoordinator: ResourceManager init skipped: %r", exc)
            self._resource_manager_proxy = None
            self._resource_manager_node0_pg = None

        # F19 active-set bootstrap. The flag distinguishes "first call
        # with empty set" from "second call with empty set" — without
        # it, set truthiness alone cannot detect double-bootstrap.
        self._active_engine_indices: Set[int] = set()
        self._active_engines_bootstrapped: bool = False

        # F40 base v=-1: published once at init Step 6.6 by
        # publish_cache_ready_step. Value None until init bootstrap
        # completes.
        self._cache_ready_step: Optional[int] = None

        # F20 lock discipline: single-method-single-critical-section.
        # _resize_sync_lock covers sync_base_weights_to_active +
        # resize_infer + _expand_workers / _shrink_workers atomic
        # mutation of _active_engine_indices and _cache_ready_step.
        self._resize_sync_lock = threading.Lock()
        # Progress aggregation lock — distinct from resize, separate
        # state.
        self._progress_lock = threading.Lock()
        self._scheduler_reports: Dict[str, ProgressReport] = {}
        self._coord_progress_last_bucket: Optional[int] = None

        # Resolve central rlix scheduler handle.
        try:
            self._rlix_scheduler = get_actor_or_raise(
                SCHEDULER_ACTOR_NAME,
                RLIX_NAMESPACE,
                error_context="MilesCoordinator requires the central scheduler actor.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MilesCoordinator: scheduler actor unavailable: %r", exc)
            self._rlix_scheduler = None

    # ------------------------------------------------------------------
    # ABC abstractmethod impls
    # ------------------------------------------------------------------

    def sync_lora_weights(self, *, loras_to_sync: List[str]) -> None:
        """LoRA support is M11.4 follow-up (F100). Stub raises so the
        ABC instance check does not fail at construction.
        """
        raise NotImplementedError(
            "sync_lora_weights is M11.4 follow-up (F100); MilesCoordinator "
            "first build supports only the actor / single SGLang server "
            "group path (F10 C19)."
        )

    def report_progress_from_scheduler(self, report: ProgressReport) -> None:
        """Aggregate per-stream progress and forward to the central
        scheduler. Mirrors PipelineCoordinator behavior with MILES-
        specific defaults (mode key from MilesRLixHooks default).
        """
        metrics = report.metrics if isinstance(report.metrics, dict) else {}
        # MilesRLixHooks bumps may carry "kind"="bump_completed" or
        # "begin_progress_batch"; both contribute to the same
        # aggregation. Tolerate missing "collected" by treating the
        # event as a 0-collected open (the new batch publishes the
        # demand).
        if "collected" not in metrics:
            metrics = dict(metrics)
            metrics["collected"] = 0
            metrics.setdefault("new_batch", True)
            report = ProgressReport(
                pipeline_id=report.pipeline_id,
                step_target_trajectories=report.step_target_trajectories,
                fifo_timestamp=report.fifo_timestamp,
                metrics=metrics,
            )
        mode = str(metrics.get("mode", "actor_train"))
        adapter_id = metrics.get("adapter_id")
        scheduler_key = f"{mode}:{adapter_id if adapter_id is not None else '__fft__'}"
        is_new_batch = bool(metrics.get("new_batch", False))
        with self._progress_lock:
            self._scheduler_reports[scheduler_key] = report
            self._aggregate_and_emit(force=is_new_batch)

    def clear_progress_stream(self, *, mode: str, adapter_id: Optional[str]) -> None:
        scheduler_key = f"{mode}:{adapter_id if adapter_id is not None else '__fft__'}"
        with self._progress_lock:
            removed = self._scheduler_reports.pop(scheduler_key, None)
            if removed is None:
                return
            if self._scheduler_reports:
                self._aggregate_and_emit(force=True)
            else:
                self._coord_progress_last_bucket = None
                if self._rlix_scheduler is not None:
                    try:
                        self._rlix_scheduler.clear_progress.remote(
                            pipeline_id=self._pipeline_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("clear_progress.remote failed: %r", exc)

    def _aggregate_and_emit(self, *, force: bool) -> None:
        if not self._scheduler_reports:
            return
        total_required = 0.0
        total_completed = 0.0
        total_collected = 0.0
        for rpt in self._scheduler_reports.values():
            rpt_metrics = rpt.metrics if isinstance(rpt.metrics, dict) else {}
            step_target = float(max(int(rpt.step_target_trajectories), 1))
            collected_raw = max(0.0, float(rpt_metrics.get("collected", 0)))
            completed_clamped = min(collected_raw, step_target)
            total_required += step_target
            total_completed += completed_clamped
            total_collected += collected_raw
        if total_required <= 0:
            return
        percent_completed = (
            min(total_completed / float(total_required), 1.0) if total_required > 0 else 0.0
        )
        bucket = math.floor(percent_completed * 50)
        if not force and bucket == self._coord_progress_last_bucket:
            return
        self._coord_progress_last_bucket = bucket
        aggregated = ProgressReport(
            pipeline_id=self._pipeline_id,
            step_target_trajectories=int(total_required),
            fifo_timestamp=time.time(),
            metrics={
                "mode": "aggregated",
                "collected": int(total_collected),
                "completed": int(total_completed),
                "bucket": int(bucket),
                "new_batch": force,
            },
        )
        if self._rlix_scheduler is not None:
            try:
                self._rlix_scheduler.report_progress.remote(aggregated)
            except Exception as exc:  # noqa: BLE001
                logger.warning("report_progress.remote failed: %r", exc)

    def _inject_pipeline_env_vars(self, *, pipeline_config: _T) -> _T:
        """Deep copy pipeline_config and inject system_envs into the copy."""
        injected = deepcopy(pipeline_config)
        env_vars = getattr(injected, "system_envs", None)
        if env_vars is None:
            try:
                injected.system_envs = dict(self._pipeline_env_vars)
            except Exception:  # noqa: BLE001
                # Frozen / unsupported config; caller must inject.
                logger.debug("Could not set system_envs on pipeline_config")
        else:
            try:
                env_vars.update(self._pipeline_env_vars)
            except Exception:  # noqa: BLE001
                logger.debug("Could not merge system_envs into pipeline_config")
        return injected

    # ------------------------------------------------------------------
    # F19 active-set bootstrap + resource registration (iter 22)
    # ------------------------------------------------------------------

    def bootstrap_active_engines(self, engine_indices: Set[int]) -> None:
        """F19 / scope X3: legitimate first call may pass an empty set.

        ``_active_engines_bootstrapped`` flag detects double-bootstrap
        even when the second call also uses an empty set; without it,
        set truthiness alone cannot distinguish the two cases.
        """
        with self._resize_sync_lock:
            if self._active_engines_bootstrapped:
                raise RuntimeError(
                    "bootstrap_active_engines called twice on the same "
                    "MilesCoordinator instance"
                )
            self._active_engine_indices = set(int(i) for i in engine_indices)
            self._active_engines_bootstrapped = True
        logger.info(
            "[MilesCoordinator] bootstrap_active_engines pipeline_id=%s indices=%s",
            self._pipeline_id,
            sorted(self._active_engine_indices),
        )

    def get_active_engines(self) -> frozenset[int]:
        """Read-only snapshot used by the pipeline init bootstrap
        consistency check (Step 7.3)."""
        with self._resize_sync_lock:
            return frozenset(self._active_engine_indices)

    def register_model_update_resources(
        self,
        *,
        cache_owner_actor,
        rollout_manager,
        train_gpu_ids: Optional[List[int]] = None,
        infer_gpu_ids: Optional[List[int]] = None,
        rollout_num_gpus_per_engine: int = 1,
    ) -> None:
        """X2 ctor handle injection — capture handles for lazy
        :class:`MilesModelUpdateService` construction. Pipeline calls
        this at init Step 6.6 BEFORE Step 7 INIT, so the service can
        be lazily built whenever the coordinator first needs to push
        a sync.

        Topology kwargs (rlix#42): ``train_gpu_ids`` / ``infer_gpu_ids``
        are this pipeline's physical GPU mappings
        (``cluster_device_mappings`` actor_train / actor_infer);
        ``rollout_num_gpus_per_engine`` is the uniform per-engine GPU
        count (v7 uniformity guard runs in MilesPipeline before this
        call). They feed transport classification. Optional for
        backward compatibility: with the default ``cpu_serialize``
        transport mode no classification runs; any other mode
        fail-fasts at sync time if topology was not registered.
        """
        if cache_owner_actor is None or rollout_manager is None:
            raise ValueError(
                "register_model_update_resources requires both handles"
            )
        per_engine = int(rollout_num_gpus_per_engine)
        if per_engine < 1:
            raise ValueError(
                f"rollout_num_gpus_per_engine must be >= 1; got {rollout_num_gpus_per_engine}"
            )
        # Resolve the transport mode ONCE per pipeline (invalid value =
        # fail-fast here, not at first sync) and log it (plan AC4).
        mode = _resolve_transport_mode()
        logger.info(
            "[MilesCoordinator] transport mode=%s pipeline_id=%s "
            "(env %s; train_gpus=%s infer_gpus=%s per_engine=%d)",
            mode,
            self._pipeline_id,
            _TRANSPORT_MODE_ENV,
            sorted(int(g) for g in train_gpu_ids) if train_gpu_ids else None,
            sorted(int(g) for g in infer_gpu_ids) if infer_gpu_ids else None,
            per_engine,
        )
        _assert_mixed_transport_startup(
            mode=mode,
            pipeline_id=self._pipeline_id,
            train_gpu_ids=list(train_gpu_ids) if train_gpu_ids is not None else None,
            infer_gpu_ids=list(infer_gpu_ids) if infer_gpu_ids is not None else None,
            per_engine=per_engine,
        )
        # Capability probe (codex impl-r15): the sync-under-load bracket
        # depends on the manager-side atomic register_router_if_active
        # RPC. On a version-skewed deployment (new rlix + old miles) the
        # missing method would only surface INSIDE the first sync's
        # finally — after engines were already quiesced. Probe with an
        # empty list at registration time so skew fails loud and early.
        try:
            ray.get(rollout_manager.register_router_if_active.remote([]))
        except AttributeError as exc:
            raise RuntimeError(
                "rollout_manager lacks register_router_if_active — the miles "
                "side of this deployment predates the rlix#42 sync-under-load "
                "bracket; update miles (miles/ray/rollout.py) before running "
                "RLix-mode weight sync"
            ) from exc
        self._model_update_resources = {
            "cache_owner_actor": cache_owner_actor,
            "rollout_manager": rollout_manager,
            "train_gpu_ids": (
                sorted(int(g) for g in train_gpu_ids) if train_gpu_ids is not None else None
            ),
            "infer_gpu_ids": (
                sorted(int(g) for g in infer_gpu_ids) if infer_gpu_ids is not None else None
            ),
            "rollout_num_gpus_per_engine": per_engine,
            "transport_mode": mode,
        }

    def _broadcast_set_for(self, target_engine_indices: Set[int]) -> frozenset:
        """Transport classification for one sync's target set (plan AC4).

        Reads the mode + topology captured at registration; delegates to
        the module-level :func:`_classify_broadcast_engines` (unit-tested
        directly). Default ``cpu_serialize`` mode returns an empty set
        without touching topology (backward compatible)."""
        resources = self._model_update_resources
        return _classify_broadcast_engines(
            target_engine_indices=target_engine_indices,
            mode=resources.get("transport_mode", "cpu_serialize"),
            train_gpu_ids=resources.get("train_gpu_ids"),
            infer_gpu_ids=resources.get("infer_gpu_ids"),
            per_engine=int(resources.get("rollout_num_gpus_per_engine", 1)),
        )

    def publish_cache_ready_step(self, step: int) -> int:
        """Init-bootstrap-only. Sets ``_cache_ready_step`` so any later
        runtime expand fired before the first ``after_training`` hook
        can publish base v=-1 weights via
        :meth:`sync_base_weights_to_active`.

        Active-refresh paths (iter 23 ``_after_training`` hook) update
        ``_cache_ready_step`` inside ``sync_base_weights_to_active``
        already, so this method is NOT used after init.
        """
        with self._resize_sync_lock:
            self._cache_ready_step = int(step)
        return int(step)

    def _ensure_model_update_service(self):
        """Lazy-construct :class:`MilesModelUpdateService` from the
        injected ctor handles."""
        if self._model_update_service is not None:
            return self._model_update_service
        if not self._model_update_resources:
            raise RuntimeError(
                "register_model_update_resources must be called before "
                "any sync_base_weights_to_active / resize_infer that "
                "exercises the runtime expand path"
            )
        from rlix.pipeline.miles_model_update_service import MilesModelUpdateService

        self._model_update_service = MilesModelUpdateService.options(
            namespace=self._ray_namespace,
            name=f"miles_model_update_service_{self._pipeline_id}",
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
        ).remote(
            pipeline_id=self._pipeline_id,
            cache_owner_actor=self._model_update_resources["cache_owner_actor"],
            rollout_manager=self._model_update_resources["rollout_manager"],
            rollout_num_gpus_per_engine=int(
                self._model_update_resources.get("rollout_num_gpus_per_engine", 1)
            ),
        )
        return self._model_update_service

    def sync_base_weights_to_active(self, step: int) -> int:
        """Drive a service.sync_selected_workers against the current
        active set, atomically updating ``_cache_ready_step`` inside
        the resize lock so concurrent runtime expand sees a consistent
        view. F40-compatible: ``step == -1`` is the init bootstrap
        path (cache_owner has built base buckets at init Step 4).
        """
        with self._resize_sync_lock:
            self._cache_ready_step = int(step)
            target = frozenset(self._active_engine_indices)
            if not target:
                # No active engines — only state advance, no sync.
                return int(step)
            service = self._ensure_model_update_service()
        # Run the sync OUTSIDE the resize lock so concurrent
        # report_progress_from_scheduler / clear_progress_stream
        # callers do not block on weight transport.
        broadcast_set = self._broadcast_set_for(set(target))
        # rlix#42 sync-under-load bracket (flush-timeout root cause):
        # under the fully-async rollout, generation for the NEXT step is
        # already live on these engines, and the router keeps dispatching
        # new requests while finalize's /flush_cache waits for the queue
        # to drain — a guaranteed 60 s timeout under real load. Mirror
        # shrink_engines' proven ordering BEFORE the sync: close router
        # admission (unregister raises on non-2xx, aborting the sync
        # early rather than hanging in flush), then abort in-flight work
        # (the router re-dispatches aborted requests afterwards — same
        # contract the donor-shrink path relies on). Re-open admission in
        # a finally: serving briefly-stale weights is recoverable,
        # unrouted-forever engines are not.
        rollout_manager = self._model_update_resources.get("rollout_manager")
        handles = ray.get(rollout_manager.get_engine_handles.remote(sorted(target)))
        sync_error: Exception | None = None
        try:
            # Quiesce INSIDE the guarded region so any failure — including
            # a partial unregister or an abort error — still reaches the
            # re-register finally (codex impl-r11 high #1).
            ray.get([h.unregister_from_router.remote() for h in handles.values()])
            ray.get(rollout_manager._abort_engines.remote(sorted(target)))
            return int(
                ray.get(
                    service.sync_selected_workers.remote(
                        sync_id=None,
                        target_engine_indices=target,
                        version=int(step),
                        broadcast_local_ranks=broadcast_set,
                    )
                )
            )
        except Exception as exc:
            sync_error = exc
            logger.error("sync_base_weights_to_active failed: %r", exc)
            raise
        finally:
            # Re-open router admission via the per-engine
            # ``register_with_router`` — idempotent at the router (re-add
            # discards from dead_workers), safe even for engines whose
            # unregister never happened. NOT ``activate_routing``: that
            # is the loading→active INIT transition and raises for these
            # still-active engines (codex impl-r11 high #2).
            # Concurrent-shrink guard (codex impl-r12/r13): a resize can
            # shrink one of these engines while the sync is in flight
            # (both flows RPC outside the lock by design, R10-F1). The
            # state-check + register must be ATOMIC w.r.t. shrink, so
            # both happen inside a single serialized manager call
            # (``register_router_if_active`` — the manager actor is
            # single-threaded, so no shrink can interleave between its
            # state read and the /add_worker). The coordinator-side
            # intent intersection under the lock is a cheap pre-filter
            # only; the manager call is the authority.
            reregister_error: Exception | None = None
            try:
                with self._resize_sync_lock:
                    still_intended = sorted(
                        set(self._active_engine_indices) & set(target)
                    )
                reregistered = (
                    ray.get(
                        rollout_manager.register_router_if_active.remote(
                            still_intended
                        )
                    )
                    if still_intended
                    else []
                )
                skipped = sorted(set(int(i) for i in target) - set(reregistered))
                if skipped:
                    logger.warning(
                        "sync-under-load bracket: NOT re-registering %s "
                        "(concurrently shrunk/offloaded or no longer intended "
                        "active)",
                        skipped,
                    )
            except Exception as exc:  # noqa: BLE001
                reregister_error = exc
                logger.error(
                    "sync-under-load bracket: failed to re-register router "
                    "workers for %s: %r",
                    sorted(target),
                    exc,
                )
            # Independently best-effort: reset the abort-idempotency cache
            # so a future genuine shrink re-aborts any in-flights that
            # arrive from now on (these engines stay ACTIVE — shrink's
            # release-on-offload path never fires for this bracket).
            try:
                ray.get(
                    rollout_manager._reset_abort_idempotency_for.remote(sorted(target))
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "sync-under-load bracket: failed to reset abort "
                    "idempotency for %s: %r",
                    sorted(target),
                    exc,
                )
            # codex impl-r14: a re-registration FAILURE (exception — not
            # the legitimate skipped-because-shrunk case) must never let a
            # successful sync report success: the engines would stay
            # intended-active yet unroutable, silently suspending
            # generation. Escalate unless an earlier exception is already
            # propagating (then keep the original as the primary error).
            if reregister_error is not None and sync_error is None:
                raise RuntimeError(
                    "sync_base_weights_to_active: sync succeeded but router "
                    f"re-registration failed for targets {sorted(target)}; "
                    "engines may be unroutable — failing fast rather than "
                    "reporting a healthy sync"
                ) from reregister_error

    # ------------------------------------------------------------------
    # F22 / F37 / F40 resize_infer + _expand_workers (iter 23)
    # ------------------------------------------------------------------

    def resize_infer(
        self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]
    ) -> ActionResponse:
        """Scheduler-driven resize. ``dp_ranks_*`` are the scheduler's
        DP-rank view; first-build contiguous-mapping invariant (F35
        C6) makes them == MILES engine_index.

        R10-F1: ``_shrink_workers`` / ``_expand_workers`` issue Ray RPCs
        WITHOUT holding ``_resize_sync_lock``. Snapshot-under-lock /
        RPC-without-lock / commit-under-lock matches the pattern in
        :meth:`sync_base_weights_to_active`. Concurrent callers that
        only need ``_active_engine_indices`` snapshots
        (``get_active_engines``, ``publish_cache_ready_step``,
        ``sync_base_weights_to_active``) no longer queue behind the
        full RPC chain.
        """
        if dp_ranks_to_remove:
            self._shrink_workers(set(int(i) for i in dp_ranks_to_remove))
        if dp_ranks_to_add:
            self._expand_workers(set(int(i) for i in dp_ranks_to_add))
        # rlix.protocol.types.ActionResponse is frozen=True/slots=True with
        # one field (success: bool); do not pass message/payload kwargs.
        return ActionResponse(success=True)

    def _shrink_workers(self, engine_indices: Set[int]) -> None:
        # Snapshot under lock — only to read shared state; the Ray RPC
        # runs without the lock held (R10-F1).
        with self._resize_sync_lock:
            rollout_manager = self._model_update_resources.get("rollout_manager")
            if rollout_manager is None:
                raise RuntimeError("resource registration missing for shrink")
        # RPC outside the lock.
        # Whole-GPU residual threshold (GiB). Miles shrink_engines logs
        # SGLang per-process/server_info attribution diagnostics with this
        # value; the hard whole-GPU gate runs in MilesPipeline after the
        # engines report offloaded. Default 13.0 is a temporary smoke-safe
        # value based on observed whole-GPU residuals with the known Megatron
        # train-offload gap; lower it after that follow-up is fixed and
        # re-measured.
        residual_threshold_gb = parse_env_positive_float(
            "MILES_MAX_RESIDUAL_GPU_MEM_GB", 13.0
        )
        shrunk = ray.get(
            rollout_manager.shrink_engines.remote(
                sorted(engine_indices),
                post_sleep_vram_threshold_gb=residual_threshold_gb,
            )
        )
        logger.info(
            "[MilesCoordinator] shrink_engines complete pipeline_id=%s "
            "engine_indices=%s whole_gpu_residual_threshold=%.1f GB "
            "(whole-GPU hard gate runs in MilesPipeline; Miles shrink logs SGLang diagnostics)",
            self._pipeline_id,
            sorted(shrunk),
            residual_threshold_gb,
        )
        # Commit under lock.
        with self._resize_sync_lock:
            self._active_engine_indices -= engine_indices

    def _expand_workers(self, engine_indices: Set[int]) -> None:
        # Phase 1 — Snapshot under lock (R10-F1).
        with self._resize_sync_lock:
            rollout_manager = self._model_update_resources.get("rollout_manager")
            if rollout_manager is None:
                raise RuntimeError("resource registration missing for expand")
            # F22 ordering: pipeline must have called publish_cache_ready_step
            # (init Step 6.6) before the FIRST runtime expand.
            if self._cache_ready_step is None:
                raise RuntimeError(
                    "resize_infer expand fired before publish_cache_ready_step; "
                    "MilesPipeline init bootstrap Step 6.6 must complete first"
                )
            cached_step = int(self._cache_ready_step)
            service = self._ensure_model_update_service()

        # Phase 2 — RPC chain WITHOUT the lock held. Read entry-state
        # snapshot so we dispatch INIT vs Runtime; the rollout manager
        # is itself a Ray actor with internal serialization, so racing
        # this read against another resize_infer is safe at the
        # manager layer.
        states = ray.get(
            rollout_manager.get_engine_states.remote(sorted(engine_indices))
        )
        unique_states = {states[idx] for idx in engine_indices}
        if unique_states == {"shell"}:
            # M11.2 INIT branch: full INIT happens through the pipeline's
            # init bootstrap Step 7. resize_infer should not see a shell
            # transition outside that path.
            raise RuntimeError(
                "resize_infer reached engines in 'shell' state; full INIT "
                "must come from MilesPipeline.initialize_pipeline Step 7, "
                "not from resize_infer."
            )
        if unique_states == {"active"}:
            # M11.2 Option β: with MILES_INIT_DEFER_ADD_WORKER=1 engines
            # come up state="loading", finish_init_offload lands them at
            # "offloaded", and the F40 Runtime branch handles every
            # subsequent expand. Hitting the "already-active" hatch under
            # Option β implies a regression (somebody activated routing
            # behind the coordinator's back). Raise so the regression
            # surfaces in the smoke log. Standalone miles (env unset)
            # still gets the hatch as the documented happy path.
            import os as _os
            if _os.environ.get("MILES_INIT_DEFER_ADD_WORKER") == "1":
                raise RuntimeError(
                    "_expand_workers: engines already 'active' under "
                    "MILES_INIT_DEFER_ADD_WORKER=1 (Option β); expected "
                    "'offloaded' (finish_init_offload should have landed "
                    "them there). engine_indices=%s" % sorted(engine_indices)
                )
            # Smoke-time hatch: miles' RolloutManager brings engines up
            # in "active" state directly (no shell→offloaded→active
            # transition). When the scheduler's gap-ratio planner fires
            # the first GEN expand after init, the engines are already
            # active; treat it as a no-op so the cycle can commit the
            # allocation bookkeeping. Engines already serve traffic, the
            # base v=-1 sync was driven from MilesPipeline init step7,
            # and routing is alive (RolloutManager.add_worker fired
            # during start_rollout_servers).
            with self._resize_sync_lock:
                self._active_engine_indices |= set(engine_indices)
            logger.info(
                "[MilesCoordinator] _expand_workers: engines already 'active' "
                "(M11.x init-comes-up-active branch); skipping wake/sync/activate_routing "
                "engine_indices=%s",
                sorted(engine_indices),
            )
            return
        if unique_states != {"offloaded"}:
            # Heterogeneous entry states are an upstream error per scope F37.
            raise RuntimeError(
                f"_expand_workers got heterogeneous engine states: {unique_states}; "
                f"upstream must dispatch a uniform set."
            )
        # F40 Runtime branch: wake → service.sync_selected_workers →
        # activate_routing.
        broadcast_set = self._broadcast_set_for(set(engine_indices))
        ray.get(rollout_manager.expand_engines.remote(sorted(engine_indices)))
        ray.get(
            service.sync_selected_workers.remote(
                sync_id=None,
                target_engine_indices=frozenset(engine_indices),
                version=cached_step,
                broadcast_local_ranks=broadcast_set,
            )
        )
        ray.get(rollout_manager.activate_routing.remote(sorted(engine_indices)))

        # Phase 3 — Commit under lock.
        with self._resize_sync_lock:
            self._active_engine_indices |= set(engine_indices)

    def remove_resource_manager_node_pg(self) -> bool:
        """Best-effort remove the ROLL ResourceManagerProxy's node-PG.

        ROLL pins all GPUs on the node into a single-bundle PG at startup so
        Phase A train-actor placements can be derived from it. After Phase A
        releases its actors, the PG keeps holding the GPUs from Ray's view,
        which blocks miles' Phase B `_create_placement_group` (which builds
        a fresh per-GPU bundle PG against the "free" pool). Removing the
        node-PG here unblocks Phase B; safe because train actors are gone.

        Returns True if a PG was actually removed.
        """
        pg = self._resource_manager_node0_pg
        if pg is None:
            return False
        try:
            ray.util.remove_placement_group(pg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "remove_resource_manager_node_pg: ray.util.remove_placement_group "
                "failed: %r",
                exc,
            )
            return False
        self._resource_manager_node0_pg = None
        if self._resource_manager_proxy is not None:
            try:
                self._resource_manager_proxy.node2pg.pop(0, None)
            except Exception:  # noqa: BLE001
                pass
        return True

    def create_pipeline_actor(self, *, pipeline_config: Any) -> Any:
        """Create / return the per-pipeline MilesPipeline actor."""
        if self._pipeline_actor is not None:
            return self._pipeline_actor
        from rlix.pipeline.miles_pipeline import MilesPipeline

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        injected_config = self._inject_pipeline_env_vars(pipeline_config=pipeline_config)
        # MILES_SKIP_NODE_PG_PIN=1 (smoke escape): when the ROLL ResourceManager
        # owns the only node-PG and Phase B needs to remove it to free GPUs for
        # the inference PG, the MilesPipeline must NOT be pinned to that PG —
        # otherwise removing the PG would kill the pipeline mid-init. We also
        # drop num_gpus to 0 in this mode because the node-PG is holding all
        # GPUs and an unpinned actor with num_gpus=0.01 would block forever
        # waiting for a free fractional slice.
        import os as _os

        scheduling_strategy = None
        skip_pg_pin = _os.environ.get("MILES_SKIP_NODE_PG_PIN") == "1"
        if (
            self._resource_manager_node0_pg is not None
            and not skip_pg_pin
        ):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self._resource_manager_node0_pg,
            )
        options = dict(
            name=f"{PIPELINE_ACTOR_NAME_PREFIX}{self._pipeline_id}",
            namespace=self._ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            max_concurrency=_MILES_PIPELINE_ACTOR_MAX_CONCURRENCY,
            runtime_env={"env_vars": self._pipeline_env_vars},
            num_gpus=0 if skip_pg_pin else 0.01,
        )
        if scheduling_strategy is not None:
            options["scheduling_strategy"] = scheduling_strategy
        self._pipeline_actor = ray.remote(MilesPipeline).options(**options).remote(
            pipeline_id=self._pipeline_id,
            pipeline_config=injected_config,
        )
        return self._pipeline_actor


__all__ = ["MilesCoordinator"]
