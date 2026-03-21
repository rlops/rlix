"""Gap-ratio generation planning algorithm for the RLix scheduler.

Extracted from SchedulerImpl to reduce class bloat.  All functions are
stateless — they read state snapshots passed as parameters and mutate the
ExecutionPlan in place.  No imports from rlix.scheduler.scheduler.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rlix.protocol.types import GENERATION_CLUSTER_NAME, Priority
from rlix.scheduler.types import (
    ClusterAllocation,
    ExecutionPlan,
    PendingRequest,
    SchedGuidedAllocationOp,
    SchedGuidedShrinkOp,
    is_generation_cluster,
    parse_cluster_id,
)

# Gap-ratio generation planning iteration limits (safety bounds).
_MAX_GAP_ITERATIONS: int = 10_000
_MAX_GAP_ACTIVATIONS: int = 1_000


@dataclass(frozen=True, slots=True)
class _GapRatioDPWorker:
    """Immutable snapshot of one data-parallel worker for gap-ratio planning.

    Each worker maps to exactly one TP-sized bundle of GPUs on a single pipeline's
    generation (actor_infer) cluster.
    """

    pipeline_id: str
    dp_rank: int
    gpu_ids: List[int]


@dataclass(slots=True)
class _GapRatioPipelineState:
    """Mutable per-pipeline bookkeeping used during a single gap-ratio iteration.

    Fields are recomputed each iteration by ``_update_gaps`` and
    ``_compute_shrink_budget_by_pipeline_id``; the dataclass avoids passing many
    loose locals through the nested helpers.
    """

    pipeline_id: str
    cluster_id: str
    remaining: float
    percent_remaining: float
    tp_size: int
    active_dp_workers: List[_GapRatioDPWorker]
    inactive_dp_workers: List[_GapRatioDPWorker]
    target_ratio: float = 0.0
    existing_ratio: float = 0.0
    gap: float = 0.0
    target_gpu_count: int = 0


def has_pending_generation_request(
    pending_bucket_gen: List[PendingRequest],
    cluster_id: str,
) -> bool:
    """Return True if the GENERATION priority bucket has a pending request for ``cluster_id``."""
    return any(p.request.cluster_id == cluster_id for p in pending_bucket_gen)


def snapshot_generation_dp_workers(
    *,
    plan: ExecutionPlan,
    idle_gpus: Set[int],
    pipeline_registry: Dict[str, Dict[str, Any]],
    active_allocations: Dict[str, ClusterAllocation],
) -> Tuple[Dict[str, List[_GapRatioDPWorker]], Dict[str, List[_GapRatioDPWorker]], Set[int]]:
    """Snapshot active and inactive generation DP workers for gap-ratio planning.

    Accounts for shrink ops already in the plan (treats those ranks as inactive).
    Returns (active_by_pipeline, inactive_by_pipeline, idle_gpus_for_gen).
    """
    active_dp_workers: Dict[str, List[_GapRatioDPWorker]] = {}
    inactive_dp_workers: Dict[str, List[_GapRatioDPWorker]] = {}

    planned_removed_ranks: Dict[str, Set[int]] = {}
    for pipeline_id in pipeline_registry:
        cluster_id = f"{pipeline_id}_{GENERATION_CLUSTER_NAME}"
        planned_removed_ranks[cluster_id] = set()
    for op in plan.sched_guided_shrink_ops:
        if not is_generation_cluster(op.cluster_id):
            continue
        planned_removed_ranks.setdefault(op.cluster_id, set()).update(op.dp_ranks_to_remove)

    non_gen_reserved_gpus: Set[int] = set()
    for cluster_id, alloc in active_allocations.items():
        if alloc.priority != Priority.GENERATION:
            non_gen_reserved_gpus |= set(alloc.gpu_ids)

    for pipeline_id, pipeline_info in pipeline_registry.items():
        cluster_configs = pipeline_info.get("cluster_configs") or {}
        infer_cfg = cluster_configs.get("actor_infer")
        if infer_cfg is None:
            raise KeyError(f"pipeline_id={pipeline_id!r} missing actor_infer cluster config")
        tp_size = int(infer_cfg.get("tp_size", 1))
        device_mapping = list(infer_cfg.get("device_mapping") or [])
        if tp_size <= 0 or not device_mapping:
            continue

        cluster_id = f"{pipeline_id}_{GENERATION_CLUSTER_NAME}"
        all_dp_ranks = list(range(len(device_mapping) // tp_size))
        removed_ranks = planned_removed_ranks.get(cluster_id, set())

        current_active_ranks: Set[int] = set()
        if cluster_id in active_allocations:
            alloc = active_allocations[cluster_id]
            if alloc.priority == Priority.GENERATION:
                current_active_ranks = set(alloc.active_dp_ranks)

        effective_active_ranks = current_active_ranks - removed_ranks
        active_list: List[_GapRatioDPWorker] = []
        for dp_rank in sorted(effective_active_ranks):
            start_idx = dp_rank * tp_size
            gpus = device_mapping[start_idx : start_idx + tp_size]
            active_list.append(_GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=dp_rank, gpu_ids=list(gpus)))

        inactive_list: List[_GapRatioDPWorker] = []
        for dp_rank in all_dp_ranks:
            if dp_rank in effective_active_ranks:
                continue
            start_idx = dp_rank * tp_size
            gpus = device_mapping[start_idx : start_idx + tp_size]
            inactive_list.append(_GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=dp_rank, gpu_ids=list(gpus)))

        active_dp_workers[pipeline_id] = active_list
        inactive_dp_workers[pipeline_id] = inactive_list

    if not idle_gpus.isdisjoint(non_gen_reserved_gpus):
        raise RuntimeError("idle_gpus must exclude non-GEN reserved GPUs")
    return active_dp_workers, inactive_dp_workers, idle_gpus


def plan_generation_gap_ratio(
    plan: ExecutionPlan,
    *,
    active_dp_workers: Dict[str, List[_GapRatioDPWorker]],
    inactive_dp_workers: Dict[str, List[_GapRatioDPWorker]],
    non_gen_reserved_gpus: Set[int],
    idle_gpus: Set[int],
    pipeline_registry: Dict[str, Dict[str, Any]],
    active_allocations: Dict[str, ClusterAllocation],
    pending_bucket_gen: List[PendingRequest],
    progress_totals_fn: Callable[..., Tuple[float, float]],
    epsilon: float = 0.0,
) -> Set[int]:
    """Distribute generation GPU budget across pipelines proportionally to remaining demand.

    Iteratively activates inactive DP workers on the pipeline with the largest
    normalized gap (target_ratio - existing_ratio) / target_ratio.  When idle GPUs
    are insufficient, workers are donated from over-provisioned pipelines.

    Returns the set of GPU ids that remain idle after planning.
    """
    # Ported from ROLL_multi_pipeline CentralizedGPUSchedulerImpl._plan_generation_gap_ratio_alternative,
    # adapted to RLix-standard progress reporting (percent_completed / step_target_trajectories).

    def _round_half_up(value: float) -> int:
        return int(math.floor(value + 0.5))

    def _remove_worker(worker: _GapRatioDPWorker) -> None:
        donor_pipeline_id = worker.pipeline_id
        donor_active = active_dp_workers.setdefault(donor_pipeline_id, [])
        donor_active[:] = [w for w in donor_active if w.dp_rank != worker.dp_rank]
        inactive_dp_workers.setdefault(donor_pipeline_id, []).append(worker)

    def _append_shrink_dp_rank(*, cluster_id: str, dp_rank: int) -> None:
        for op in plan.sched_guided_shrink_ops:
            if op.cluster_id == cluster_id:
                if dp_rank not in op.dp_ranks_to_remove:
                    op.dp_ranks_to_remove.append(dp_rank)
                return
        plan.sched_guided_shrink_ops.append(SchedGuidedShrinkOp(cluster_id=cluster_id, dp_ranks_to_remove=[dp_rank]))

    def _receiver_eligible(state: _GapRatioPipelineState) -> bool:
        if state.cluster_id in plan.clusters_to_remove:
            return False
        if has_pending_generation_request(pending_bucket_gen, state.cluster_id):
            return True
        return bool(state.active_dp_workers) or state.cluster_id in active_allocations

    pipeline_states: List[_GapRatioPipelineState] = []
    for pipeline_id in pipeline_registry:
        cluster_id = f"{pipeline_id}_{GENERATION_CLUSTER_NAME}"
        infer_cfg = pipeline_registry[pipeline_id].get("cluster_configs", {}).get("actor_infer")
        if infer_cfg is None:
            raise KeyError(f"pipeline_id={pipeline_id!r} missing actor_infer cluster config")
        tp_size = int(infer_cfg.get("tp_size", 1))
        if tp_size <= 0:
            raise ValueError(f"pipeline_id={pipeline_id!r} has invalid actor_infer tp_size={tp_size}")

        # Derive remaining from completed metric; same derivation path as
        # background rebalance to keep demand semantics consistent.
        remaining, step_target = progress_totals_fn(pipeline_id=pipeline_id)
        if step_target <= 0.0:
            continue
        percent_remaining = 0.0 if step_target <= 0 else remaining / step_target

        has_pending = has_pending_generation_request(pending_bucket_gen, cluster_id)
        if has_pending:
            # Inflate demand so a pipeline that hasn't started generating yet (remaining == 0)
            # still receives a non-zero weight and gets allocated at least one DP worker.
            remaining += step_target
            percent_remaining = remaining / step_target if step_target > 0 else 0.0

        active_list = active_dp_workers.setdefault(pipeline_id, [])
        inactive_list = inactive_dp_workers.setdefault(pipeline_id, [])
        pipeline_states.append(
            _GapRatioPipelineState(
                pipeline_id=pipeline_id,
                cluster_id=cluster_id,
                remaining=remaining,
                percent_remaining=percent_remaining,
                tp_size=tp_size,
                active_dp_workers=active_list,
                inactive_dp_workers=inactive_list,
            )
        )

    if not idle_gpus.isdisjoint(non_gen_reserved_gpus):
        raise RuntimeError("idle_gpus must exclude non-GEN reserved GPUs")

    protected: Set[Tuple[str, int]] = set()
    for op in plan.sched_guided_shrink_ops:
        pipeline_id, cluster_name = parse_cluster_id(op.cluster_id)
        if cluster_name != "actor_infer":
            continue
        for dp_rank in op.dp_ranks_to_remove:
            protected.add((pipeline_id, dp_rank))

    eligible_for_target = [p for p in pipeline_states if _receiver_eligible(p)]
    # target_weight sums only eligible pipelines, but budget includes all pipelines' active
    # GPUs — ineligible pipelines' GPUs are redistributed, not kept.
    total_target_weight = sum(p.remaining * p.tp_size for p in eligible_for_target)
    total_gen_budget_gpus = len(idle_gpus) + sum(len(p.active_dp_workers) * p.tp_size for p in pipeline_states)
    if total_gen_budget_gpus == 0:
        return idle_gpus

    for p in pipeline_states:
        if not _receiver_eligible(p) or total_target_weight == 0:
            p.target_ratio = 0.0
            p.target_gpu_count = 0
        else:
            p.target_ratio = (p.remaining * p.tp_size) / total_target_weight
            raw_target_bundles = (p.target_ratio * total_gen_budget_gpus) / p.tp_size
            rounded_bundles = _round_half_up(raw_target_bundles)
            # Floor: every pipeline with non-zero demand gets at least one TP bundle,
            # otherwise the gap never closes and the pipeline stays starved.
            p.target_gpu_count = max(rounded_bundles * p.tp_size, p.tp_size)

    def _update_gaps() -> None:
        for state in pipeline_states:
            active_gpus = len(state.active_dp_workers) * state.tp_size
            state.existing_ratio = 0.0 if total_gen_budget_gpus == 0 else active_gpus / total_gen_budget_gpus
            state.gap = state.target_ratio - state.existing_ratio

    def _compute_shrink_budget_by_pipeline_id() -> Dict[str, int]:
        """Max workers each pipeline can donate without dropping below its target allocation."""
        shrink_budget: Dict[str, int] = {}
        for state in pipeline_states:
            # Only protect bundles for pipelines that are eligible receivers with non-zero demand.
            # _receiver_eligible already excludes clusters_to_remove.
            if _receiver_eligible(state) and state.target_gpu_count > 0:
                min_bundles = max(1, state.target_gpu_count // state.tp_size)
            else:
                # Pipeline is being removed, has zero demand, or is not actively participating —
                # all its workers are available for donation.
                min_bundles = 0
            shrink_budget[state.pipeline_id] = max(0, len(state.active_dp_workers) - min_bundles)
        return shrink_budget

    def _try_activate_one(
        state: _GapRatioPipelineState,
        *,
        shrink_budget_by_pipeline_id: Dict[str, int],
        percent_remaining_by_pipeline_id: Dict[str, float],
    ) -> bool:
        nonlocal idle_gpus, activations  # type: ignore[misc]  # assigned after def but in enclosing scope

        if state.cluster_id in plan.clusters_to_remove:
            return False

        available_inactive = [w for w in state.inactive_dp_workers if (state.pipeline_id, w.dp_rank) not in protected]
        if not available_inactive:
            return False

        candidates: List[
            Tuple[
                _GapRatioDPWorker,
                List[Tuple[float, _GapRatioDPWorker, Set[int]]],
                Tuple[int, Tuple[float, ...], int],
            ]
        ] = []
        for inactive in sorted(available_inactive, key=lambda w: w.dp_rank):
            needed_bundle = set(inactive.gpu_ids)
            if needed_bundle & non_gen_reserved_gpus:
                continue

            missing = needed_bundle - idle_gpus
            donor_plan: List[Tuple[float, _GapRatioDPWorker, Set[int]]] = []

            if missing:
                donors: List[Tuple[float, _GapRatioDPWorker, Set[int]]] = []
                for donor_state in sorted(pipeline_states, key=lambda x: x.gap):
                    if donor_state.gap >= -epsilon:
                        continue
                    if shrink_budget_by_pipeline_id[donor_state.pipeline_id] <= 0:
                        continue
                    for worker in donor_state.active_dp_workers:
                        if (worker.pipeline_id, worker.dp_rank) in protected:
                            continue
                        worker_bundle = set(worker.gpu_ids)
                        if not (worker_bundle & missing):
                            continue
                        donors.append((donor_state.gap, worker, worker_bundle))

                planned_shrinks_per_pipeline_id: Dict[str, int] = defaultdict(int)
                picked: List[Tuple[float, _GapRatioDPWorker, Set[int]]] = []
                for gap_value, worker, worker_bundle in donors:
                    if not missing:
                        break
                    if (
                        planned_shrinks_per_pipeline_id[worker.pipeline_id]
                        >= shrink_budget_by_pipeline_id[worker.pipeline_id]
                    ):
                        continue
                    picked.append((gap_value, worker, worker_bundle))
                    planned_shrinks_per_pipeline_id[worker.pipeline_id] += 1
                    missing -= worker_bundle

                if missing:
                    continue
                donor_plan.extend(picked)

            # Score prefers: (1) free idle GPUs over donor shrinks, (2) donors with most
            # remaining work (protects near-completion pipelines), (3) lower dp_rank for determinism.
            needs_shrink = 0 if not donor_plan else 1
            donor_percents = sorted(
                [percent_remaining_by_pipeline_id[donor_worker.pipeline_id] for _, donor_worker, _ in donor_plan]
            )
            score = (needs_shrink, tuple([-p for p in donor_percents]), inactive.dp_rank)
            candidates.append((inactive, donor_plan, score))

        if not candidates:
            return False

        inactive, donor_plan, _ = sorted(candidates, key=lambda c: c[2])[0]
        needed_bundle = set(inactive.gpu_ids)
        if needed_bundle & non_gen_reserved_gpus:
            return False

        planned_available = set(idle_gpus)
        for _, _, donor_gpus in donor_plan:
            planned_available |= set(donor_gpus)
        if not needed_bundle.issubset(planned_available):
            return False

        new_idle_gpus = planned_available - needed_bundle

        # Validate receiver eligibility BEFORE committing donor mutations.
        # Previous ordering applied donor shrinks first, leaving them unrolled if this guard fired.
        has_pending_request = has_pending_generation_request(pending_bucket_gen, state.cluster_id)
        if state.cluster_id in plan.clusters_to_remove:
            return False
        if not has_pending_request and state.cluster_id not in active_allocations:
            return False

        for _, donor_worker, _ in donor_plan:
            _append_shrink_dp_rank(
                cluster_id=f"{donor_worker.pipeline_id}_{GENERATION_CLUSTER_NAME}", dp_rank=donor_worker.dp_rank
            )
            _remove_worker(donor_worker)
            protected.add((donor_worker.pipeline_id, donor_worker.dp_rank))
        existing_alloc_op = next(
            (op for op in plan.sched_guided_allocation_ops if op.cluster_id == state.cluster_id),
            None,
        )
        if existing_alloc_op is not None:
            existing_alloc_op.dp_ranks_to_add.append(inactive.dp_rank)
            existing_alloc_op.gpus_to_allocate.extend(sorted(needed_bundle))
            existing_alloc_op.has_pending_request = existing_alloc_op.has_pending_request or has_pending_request
        else:
            plan.sched_guided_allocation_ops.append(
                SchedGuidedAllocationOp(
                    cluster_id=state.cluster_id,
                    dp_ranks_to_add=[inactive.dp_rank],
                    gpus_to_allocate=sorted(needed_bundle),
                    has_pending_request=has_pending_request,
                    tp_size=state.tp_size,
                )
            )
        active_dp_workers.setdefault(state.pipeline_id, []).append(inactive)
        receiver_inactive = inactive_dp_workers.setdefault(state.pipeline_id, [])
        receiver_inactive[:] = [w for w in receiver_inactive if w.dp_rank != inactive.dp_rank]
        protected.add((state.pipeline_id, inactive.dp_rank))
        activations += 1
        idle_gpus = new_idle_gpus
        return True

    iterations = 0
    activations = 0
    while True:
        iterations += 1
        if iterations > _MAX_GAP_ITERATIONS or activations > _MAX_GAP_ACTIVATIONS:
            raise RuntimeError("gap_ratio_generation_planning_exceeded_limits")

        _update_gaps()
        percent_remaining_by_pipeline_id = {s.pipeline_id: s.percent_remaining for s in pipeline_states}
        shrink_budget_by_pipeline_id = _compute_shrink_budget_by_pipeline_id()

        def _normalized_gap(state: _GapRatioPipelineState) -> Optional[float]:
            if state.target_ratio <= 0:
                return None
            return state.gap / state.target_ratio

        acceptors: List[_GapRatioPipelineState] = [
            p
            for p in pipeline_states
            if p.gap > epsilon
            and _receiver_eligible(p)
            and (len(p.active_dp_workers) * p.tp_size) < p.target_gpu_count
        ]
        acceptors_with_norm_gap = [(_normalized_gap(p), p) for p in acceptors]
        acceptors_with_norm_gap = [(ng, p) for ng, p in acceptors_with_norm_gap if ng is not None]
        # Sort by normalized gap desc (scale-invariant underservice), absolute gap desc as
        # tiebreaker, then pipeline_id for determinism.
        acceptors = [p for _, p in sorted(acceptors_with_norm_gap, key=lambda x: (-x[0], -x[1].gap, x[1].pipeline_id))]  # type: ignore[operator]  # None filtered on prior line
        if not acceptors:
            break

        # Activate at most one worker per iteration: _update_gaps() must recompute
        # existing_ratio after each activation to avoid using stale ratios.
        any_activation = False
        for acceptor in acceptors:
            if _try_activate_one(
                acceptor,
                shrink_budget_by_pipeline_id=shrink_budget_by_pipeline_id,
                percent_remaining_by_pipeline_id=percent_remaining_by_pipeline_id,
            ):
                any_activation = True
                break

        if not any_activation:
            break

    return idle_gpus
