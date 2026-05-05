"""Version-tracked lifecycle manager for ROLL's CPU bucket cache.

ROLL's CPU bucket cache is split across two calls:
  1. ``_build_latest_bucket_cache(version)`` — called *inside* ``train_step``
     when ``DO_TIME_SHARING=True``. Gathers weights from all PP ranks into
     the cache owner's CPU memory.
  2. ``promote_active_checkpoint(version)`` — called by the *pipeline* after
     ``train_step`` returns. Atomically commits the just-built version as the
     one that ``selective_sync_active_cache`` (expand path) will use.

This split allows a new version to be built concurrently while the previous
active version is being broadcast to inference workers.

``BucketCacheLifecycle`` wraps these two calls with:
  - ``_cache_ready_step``: the version number of the last successfully
    promoted cache. ``-1`` = base model (pre-training).
  - ``promote(version)``: calls ``promote_active_checkpoint`` on all training
    workers and updates ``_cache_ready_step``.
  - ``is_ready_for_version(version)``: fast check used by the scheduler to
    decide whether expand is safe.

Why a separate class?
    The NeMo RL port (see ``plans/nemorl-port-plan.md`` Feature 4) needs to
    re-implement the same lifecycle without ROLL's internal ``train_step``
    hook.  Encapsulating the version accounting here makes it easy to swap
    the ROLL-backed implementation for a NeMo-backed one without touching
    the pipeline orchestration layer.

Thread / Ray safety:
    ``_cache_ready_step`` is written only by the pipeline actor (single
    writer), so no locking is needed at this level.  ROLL's
    ``promote_active_checkpoint`` acquires ``_cache_lock`` internally.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional

try:
    import ray
    _HAS_RAY = True
except ImportError:
    _HAS_RAY = False

try:
    from roll.utils.logging import get_logger
    logger = get_logger()
except Exception:  # pragma: no cover
    import logging as _logging
    logger = _logging.getLogger(__name__)  # type: ignore[assignment]


_UNINITIALIZED = object()  # sentinel


class BucketCacheLifecycle:
    """Version-tracking wrapper around ROLL's promote_active_checkpoint.

    One instance per pipeline.  Tracks which version of the CPU bucket cache
    is currently active and ready to be broadcast to inference workers.

    Args:
        pipeline_id: Human-readable identifier for the owning pipeline.
        workers: List of training worker Ray actor handles (``src_cluster.workers``).
        base_version: Version number assigned to the initial base-model cache
            built at pipeline init time.  Default is ``-1`` (ROLL convention).
    """

    _BASE_VERSION = -1  # init cache version (pre-training)

    def __init__(
        self,
        *,
        pipeline_id: str,
        workers: List[Any],
        base_version: int = -1,
    ) -> None:
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise ValueError("pipeline_id must be a non-empty string")
        if not workers:
            raise ValueError("workers must be a non-empty list")

        self.pipeline_id = pipeline_id
        self._workers = list(workers)
        self._base_version = int(base_version)

        # Tracks the last successfully promoted version.
        # Starts as sentinel (promote() has never been called).
        self._cache_ready_step: int | object = _UNINITIALIZED

        # Guards _cache_ready_step writes (single pipeline actor, but defensive).
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cache_ready_step(self) -> Optional[int]:
        """Last promoted version, or ``None`` if ``promote()`` has never run."""
        with self._lock:
            if self._cache_ready_step is _UNINITIALIZED:
                return None
            return int(self._cache_ready_step)  # type: ignore[arg-type]

    def promote(self, version: int) -> None:
        """Commit *version* as the active cache for selective sync.

        Calls ``promote_active_checkpoint(version)`` on every training worker.
        Workers are called directly (synchronous pattern compatible with both
        real Ray actors when wrapped by the caller and fake workers in tests).
        Only the cache owner (pp_rank==0, dp_rank==0, tp_rank==0) does
        meaningful work inside ROLL; non-owners return immediately.

        On success, ``_cache_ready_step`` is updated to *version*.

        Pipeline integration note:
            In the actual Ray cluster, wrap each worker call with
            ``ray.get([w.promote_active_checkpoint.remote(v) for w in workers])``
            from the pipeline layer.  Use ``BucketCacheLifecycle`` via
            ``promote_fn`` or call the internal ``_promote_workers()``
            after that ``ray.get`` completes.

        Args:
            version: Checkpoint version to promote.  Must equal the
                ``checkpoint_version`` passed to ``_build_latest_bucket_cache``
                (called by ``train_step`` internally when DO_TIME_SHARING=True).

        Raises:
            RuntimeError: If ``promote_active_checkpoint`` fails on any worker
                (e.g. cache_key not found, which means train_step did not build
                the cache for this version).
        """
        version = int(version)
        logger.info(
            "[BucketCacheLifecycle] promote_start pipeline_id=%s version=%d",
            self.pipeline_id, version,
        )

        for worker in self._workers:
            worker.promote_active_checkpoint(version)

        with self._lock:
            self._cache_ready_step = version

        logger.info(
            "[BucketCacheLifecycle] promote_done pipeline_id=%s version=%d",
            self.pipeline_id, version,
        )

    def promote_base(self) -> None:
        """Build and promote the initial base-model cache (version=-1).

        Called once during pipeline initialisation.  This method first calls
        ``build_latest_bucket_cache(-1)`` on all training workers so that the
        PP collective gather completes, then promotes version -1 to active.
        Equivalent to the init sequence in NeMo RL megatron_policy_worker:
            ray.get([w.build_latest_bucket_cache.remote(-1) for w in workers])
            ray.get([w.promote_active_checkpoint.remote(-1) for w in workers])
        """
        version = self._base_version
        logger.info(
            "[BucketCacheLifecycle] promote_base_build pipeline_id=%s version=%d",
            self.pipeline_id, version,
        )
        for worker in self._workers:
            worker.build_latest_bucket_cache(version)

        self.promote(version)

    def is_ready(self) -> bool:
        """Return ``True`` if at least one cache version has been promoted."""
        return self._cache_ready_step is not _UNINITIALIZED

    def is_ready_for_version(self, version: int) -> bool:
        """Return ``True`` if the active cache is at or beyond *version*.

        Used by the scheduler to decide whether expand is safe before
        calling ``ModelUpdateService.sync_selected_workers``.

        Returns ``False`` when ``promote()`` has never been called.
        """
        with self._lock:
            if self._cache_ready_step is _UNINITIALIZED:
                return False
            return int(self._cache_ready_step) >= int(version)  # type: ignore[arg-type]

    def mark_promoted(self, version: int) -> None:
        """Record *version* as active without calling any workers.

        Use this when the pipeline layer has already performed build and promote
        via ``ray.get([w.build_latest_bucket_cache.remote(v) ...])`` and
        ``ray.get([w.promote_active_checkpoint.remote(v) ...])`` directly,
        and only needs the lifecycle tracker to reflect the new version.

        Args:
            version: Checkpoint version that was already built and promoted externally.
        """
        version = int(version)
        with self._lock:
            self._cache_ready_step = version
        logger.info(
            "[BucketCacheLifecycle] mark_promoted pipeline_id=%s version=%d",
            self.pipeline_id, version,
        )

    def reset(self) -> None:
        """Reset version tracking (e.g. after a pipeline restart).

        Does NOT touch the ROLL worker caches — callers must rebuild the
        cache if needed.
        """
        with self._lock:
            self._cache_ready_step = _UNINITIALIZED

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        step = self.cache_ready_step
        step_str = str(step) if step is not None else "uninitialized"
        return (
            f"BucketCacheLifecycle("
            f"pipeline_id={self.pipeline_id!r}, "
            f"workers={len(self._workers)}, "
            f"cache_ready_step={step_str})"
        )
