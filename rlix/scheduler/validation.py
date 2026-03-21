"""Fail-fast validation for scheduler execution plans.

Validates an ``ExecutionPlan`` against 11 numbered invariant conditions before the
scheduler applies any mutations to cluster state.  The conditions fall into four
categories:

* **GPU accounting** (conditions 4, 8, 9): idle/allocated sets are disjoint, the
  GPU universe is conserved after simulating the plan, and no GPU is freed twice.
* **Operation integrity** (conditions 1, 3): each cluster appears in at most one
  operation type per cycle (allocate vs. remove).
* **State-transition legality** (conditions 2, 5, 6, 7): DP-rank shrink/expand ops
  target only generation clusters, operate on valid DP ranks within device-mapping
  bounds, and respect ``max_dp_workers``.
* **Overlap prevention** (condition 10): no DP rank or GPU is claimed by two
  clusters simultaneously.
* **Referential integrity** (condition 11): clusters scheduled for removal must
  exist in active allocations.

All checks raise ``ValidationError`` with the condition number so callers can
programmatically identify the failing invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from rlix.protocol.types import Priority
from rlix.scheduler.types import (
    ClusterAllocation,
    ExecutionPlan,
    ValidationError,
    build_dp_rank_mapping,
    is_generation_cluster,
    parse_cluster_id,
)


@dataclass(frozen=True, slots=True)
class ValidationInputs:
    """Snapshot of scheduler state needed for plan validation.

    Captures the three pieces of global state that the validator must inspect:
    the pipeline registry (cluster configs with device mappings and TP sizes),
    the current GPU allocations, and the set of unassigned GPUs.

    Attributes:
        pipeline_registry: Pipeline ID -> pipeline config dict.  Each config
            contains a ``"cluster_configs"`` mapping of cluster name to its
            ``device_mapping``, ``tp_size``, and optional ``max_dp_workers``.
        active_allocations: Cluster ID -> live ``ClusterAllocation`` at the
            start of the scheduling cycle.
        idle_gpus: GPU device indices not currently assigned to any cluster.
    """

    pipeline_registry: Dict[str, Dict[str, Any]]
    active_allocations: Dict[str, ClusterAllocation]
    idle_gpus: Set[int]


def _cluster_config(inputs: ValidationInputs, cluster_id: str) -> Dict[str, Any]:
    """Look up the cluster config dict for *cluster_id* from the pipeline registry.

    Args:
        inputs: Current scheduler state snapshot.
        cluster_id: Composite ``"{pipeline_id}/{cluster_name}"`` identifier.

    Returns:
        The cluster configuration dict containing ``device_mapping``,
        ``tp_size``, and optionally ``max_dp_workers``.

    Raises:
        KeyError: If the pipeline or cluster name is not registered.
    """
    pipeline_id, cluster_name = parse_cluster_id(cluster_id)
    pipeline = inputs.pipeline_registry.get(pipeline_id)
    if pipeline is None:
        raise KeyError(f"pipeline_id {pipeline_id!r} not registered")
    cluster_configs = pipeline.get("cluster_configs") or {}
    config: Any = cluster_configs.get(cluster_name)
    if config is None:
        raise KeyError(f"cluster_name {cluster_name!r} not registered for pipeline_id {pipeline_id!r}")
    result: Dict[str, Any] = config
    return result


def _cluster_tp_size(inputs: ValidationInputs, cluster_id: str) -> int:
    """Return the tensor-parallel size for *cluster_id* (defaults to 1).

    Raises:
        ValueError: If ``tp_size`` is non-positive.
    """
    tp_size = int(_cluster_config(inputs, cluster_id).get("tp_size", 1))
    if tp_size <= 0:
        raise ValueError(f"Invalid tp_size={tp_size} for cluster_id {cluster_id!r}")
    return tp_size


def _cluster_device_mapping(inputs: ValidationInputs, cluster_id: str) -> List[int]:
    """Return the ordered list of GPU device indices assigned to *cluster_id*.

    Raises:
        ValueError: If ``device_mapping`` is missing or empty.
    """
    mapping = list(_cluster_config(inputs, cluster_id).get("device_mapping") or [])
    if not mapping:
        raise ValueError(f"Missing device_mapping for cluster_id {cluster_id!r}")
    return mapping


def _max_dp_workers(inputs: ValidationInputs, cluster_id: str) -> int:
    """Derive the maximum number of data-parallel workers for *cluster_id*.

    If the cluster config provides an explicit ``max_dp_workers``, that value is
    returned.  Otherwise the cap is inferred as
    ``len(device_mapping) // tp_size``.

    Returns:
        Maximum DP worker count (0 when device_mapping is empty).
    """
    cfg = _cluster_config(inputs, cluster_id)
    tp_size = int(cfg.get("tp_size", 1))
    device_mapping = list(cfg.get("device_mapping") or [])
    if not device_mapping:
        return 0
    max_dp = cfg.get("max_dp_workers")
    if max_dp is None:
        return len(device_mapping) // tp_size if tp_size > 0 else 0
    return int(max_dp)


def validate_dp_ranks_to_add(*, dp_ranks_to_add: List[int], max_dp_ranks: int) -> None:
    """Validate dp_ranks_to_add for type and range bounds.

    Args:
        dp_ranks_to_add: List of DP ranks to validate.
        max_dp_ranks: Maximum allowed DP rank value (exclusive upper bound).

    Raises:
        TypeError: If dp_ranks_to_add is not a list.
        ValueError: If any rank is invalid (negative or exceeds max_dp_ranks).
    """
    if not isinstance(dp_ranks_to_add, list):
        raise TypeError(f"dp_ranks_to_add must be list[int], got {type(dp_ranks_to_add).__name__}")
    for rank in dp_ranks_to_add:
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"dp_ranks_to_add must contain non-negative integers, got {rank!r}")
        if rank >= max_dp_ranks:
            raise ValueError(f"dp_ranks_to_add rank {rank} exceeds max_dp_ranks {max_dp_ranks}")


def validate_execution_plan(plan: ExecutionPlan, *, inputs: ValidationInputs) -> None:
    """Validate an execution plan against 11 invariant conditions before it is applied.

    The validator simulates the plan's effects (shrinks, removals, allocations,
    expansions) on a copy of the current state and checks every invariant at
    each step.  If any condition is violated a ``ValidationError`` is raised
    with the condition number so the caller can identify the failure
    programmatically.

    Checked conditions:
        1. **Operation uniqueness** — each cluster appears at most once per op list.
        2. **State-transition legality** — shrink/expand ops target generation clusters only.
        3. **Mutual exclusivity** — a cluster cannot be both allocated and removed.
        4. **GPU state consistency** — idle and allocated sets stay disjoint throughout.
        5. **DP-rank activity** — shrinks remove only active ranks; expansions add only inactive ranks.
        6. **Device-mapping bounds** — allocated GPUs fall within the cluster's device mapping
           and bundle sizes match ``tp_size``.
        7. **Capacity limits** — expansions do not exceed ``max_dp_workers``.
        8. **Conservation** — after simulation, idle + allocated equals the full GPU universe.
        9. **No double-free** — no GPU is freed more than once within one plan.
        10. **No overlap** — no GPU or DP rank is claimed by two clusters simultaneously.
        11. **Existence** — clusters scheduled for removal must exist in active allocations.

    Args:
        plan: The proposed execution plan to validate.
        inputs: Scheduler state snapshot (registry, allocations, idle GPUs).

    Raises:
        ValidationError: On the first condition violation found.
    """

    # Condition 4 (GPU state consistency): idle and allocated disjoint at entry.
    allocated_gpus = {gpu for alloc in inputs.active_allocations.values() for gpu in alloc.gpu_ids}
    if not allocated_gpus.isdisjoint(inputs.idle_gpus):
        raise ValidationError(
            "idle_gpus overlaps allocated GPUs",
            condition=4,
            context={"overlap": sorted(allocated_gpus & inputs.idle_gpus)},
        )

    # Condition 1 (operation uniqueness): each cluster appears in at most one op per list.
    shrink_cluster_ids = [op.cluster_id for op in plan.sched_guided_shrink_ops]
    if len(shrink_cluster_ids) != len(set(shrink_cluster_ids)):
        raise ValidationError(
            "duplicate cluster_id in sched_guided_shrink_ops",
            condition=1,
            context={
                "cluster_ids": sorted(cid for cid in set(shrink_cluster_ids) if shrink_cluster_ids.count(cid) > 1)
            },
        )
    alloc_cluster_ids = [op.cluster_id for op in plan.sched_guided_allocation_ops]
    if len(alloc_cluster_ids) != len(set(alloc_cluster_ids)):
        raise ValidationError(
            "duplicate cluster_id in sched_guided_allocation_ops",
            condition=1,
            context={"cluster_ids": sorted(cid for cid in set(alloc_cluster_ids) if alloc_cluster_ids.count(cid) > 1)},
        )
    pending_cluster_ids = [op.cluster_id for op in plan.signal_pending_allocation_ops]
    if len(pending_cluster_ids) != len(set(pending_cluster_ids)):
        raise ValidationError(
            "duplicate cluster_id in signal_pending_allocation_ops",
            condition=1,
            context={
                "cluster_ids": sorted(cid for cid in set(pending_cluster_ids) if pending_cluster_ids.count(cid) > 1)
            },
        )

    # Condition 3 (mutual exclusivity): cannot both allocate and remove the same cluster in one cycle.
    alloc_targets = {op.cluster_id for op in plan.signal_pending_allocation_ops} | {
        op.cluster_id for op in plan.sched_guided_allocation_ops
    }
    alloc_remove_overlap = alloc_targets & set(plan.clusters_to_remove)
    if alloc_remove_overlap:
        raise ValidationError(
            "cluster_id appears in both allocation ops and clusters_to_remove",
            condition=3,
            context={"cluster_ids": sorted(alloc_remove_overlap)},
        )

    # Condition 2 (state transitions): non-generation clusters must not be shrunk/expanded via DP-rank ops.
    for shrink_op in list(plan.sched_guided_shrink_ops):
        if not is_generation_cluster(shrink_op.cluster_id):
            raise ValidationError(
                "shrink op applied to non-generation cluster",
                condition=2,
                context={"cluster_id": shrink_op.cluster_id},
            )
    for alloc_op_c2 in plan.sched_guided_allocation_ops:
        if not is_generation_cluster(alloc_op_c2.cluster_id):
            raise ValidationError(
                "generation allocation op applied to non-generation cluster",
                condition=2,
                context={"cluster_id": alloc_op_c2.cluster_id},
            )

    # Condition 11 (consistency): clusters_to_remove must only contain clusters that exist.
    missing = [cid for cid in plan.clusters_to_remove if cid not in inputs.active_allocations]
    if missing:
        raise ValidationError(
            "clusters_to_remove contains unknown cluster_id", condition=11, context={"cluster_ids": missing}
        )

    # Condition 6 (device mapping boundaries) and Condition 7 (capacity limits) and Condition 5 (dp-rank activity).
    for signal_op_c6 in plan.signal_pending_allocation_ops:
        device_mapping = set(_cluster_device_mapping(inputs, signal_op_c6.cluster_id))
        if not set(signal_op_c6.gpus_to_allocate).issubset(device_mapping):
            raise ValidationError(
                "allocation uses GPU IDs outside device_mapping",
                condition=6,
                context={
                    "cluster_id": signal_op_c6.cluster_id,
                    "gpus": sorted(set(signal_op_c6.gpus_to_allocate) - device_mapping),
                },
            )

    # --- Simulation phase ---
    # Deep-copy allocations and idle set so mutations during simulation
    # don't affect the caller's state.  The plan is applied in order:
    # shrinks -> removals -> pending allocations -> expansions.
    sim_allocations: Dict[str, ClusterAllocation] = {}
    for cid, alloc in inputs.active_allocations.items():
        sim_allocations[cid] = ClusterAllocation(
            cluster_id=alloc.cluster_id,
            gpu_ids=list(alloc.gpu_ids),
            priority=alloc.priority,
            active_dp_ranks=set(alloc.active_dp_ranks),
            dp_rank_to_gpus={k: list(v) for k, v in alloc.dp_rank_to_gpus.items()},
            global_step=alloc.global_step,
            timestamp=alloc.timestamp,
        )
    sim_idle = set(inputs.idle_gpus)

    # Condition 9: no GPU is freed twice within this plan.
    freed_gpus: Set[int] = set()

    def _free_bundle(cluster_id: str, bundle: Set[int]) -> None:
        """Return *bundle* GPUs to the idle pool; raise on double-free (condition 9)."""
        nonlocal freed_gpus, sim_idle
        if freed_gpus & bundle:
            raise ValidationError(
                "double-free detected",
                condition=9,
                context={"cluster_id": cluster_id, "gpus": sorted(freed_gpus & bundle)},
            )
        freed_gpus |= set(bundle)
        sim_idle |= set(bundle)

    def _ensure_dp_rank_mapping(cluster_id: str) -> None:
        """Lazily build dp_rank_to_gpus for a simulated allocation if missing."""
        alloc = sim_allocations.get(cluster_id)
        if alloc is None:
            return
        if alloc.dp_rank_to_gpus:
            return
        tp_size = _cluster_tp_size(inputs, cluster_id)
        alloc.dp_rank_to_gpus = build_dp_rank_mapping(alloc.gpu_ids, tp_size)

    # Apply shrinks.
    for sim_shrink_op in plan.sched_guided_shrink_ops:
        if not sim_shrink_op.dp_ranks_to_remove:
            continue
        sim_shrink_alloc = sim_allocations.get(sim_shrink_op.cluster_id)
        if sim_shrink_alloc is None:
            raise ValidationError(
                "shrink op targets unregistered cluster",
                condition=11,
                context={"cluster_id": sim_shrink_op.cluster_id},
            )
        if sim_shrink_alloc.priority != Priority.GENERATION:
            raise ValidationError(
                "shrink on non-generation allocation",
                condition=2,
                context={"cluster_id": sim_shrink_op.cluster_id},
            )
        _ensure_dp_rank_mapping(sim_shrink_op.cluster_id)
        active = set(sim_shrink_alloc.active_dp_ranks)
        if not set(sim_shrink_op.dp_ranks_to_remove).issubset(active):
            raise ValidationError(
                "shrink removes inactive dp ranks",
                condition=5,
                context={
                    "cluster_id": sim_shrink_op.cluster_id,
                    "active": sorted(active),
                    "remove": sim_shrink_op.dp_ranks_to_remove,
                },
            )
        for dp_rank in sim_shrink_op.dp_ranks_to_remove:
            bundle = set(sim_shrink_alloc.dp_rank_to_gpus.get(dp_rank) or [])
            if not bundle:
                raise ValidationError(
                    "missing dp_rank_to_gpus bundle for dp_rank",
                    condition=6,
                    context={"cluster_id": sim_shrink_op.cluster_id, "dp_rank": dp_rank},
                )
            _free_bundle(sim_shrink_op.cluster_id, bundle)
            sim_shrink_alloc.active_dp_ranks.discard(dp_rank)
            sim_shrink_alloc.dp_rank_to_gpus.pop(dp_rank, None)
            sim_shrink_alloc.gpu_ids = [g for g in sim_shrink_alloc.gpu_ids if g not in bundle]

    # Apply clusters_to_remove (any remaining GPUs are freed).
    for remove_cluster_id in sorted(plan.clusters_to_remove):
        remove_alloc = sim_allocations.pop(remove_cluster_id, None)
        if remove_alloc is None:
            continue
        _free_bundle(remove_cluster_id, set(remove_alloc.gpu_ids))

    # Apply pending allocations (initial allocations).
    for sim_signal_op in plan.signal_pending_allocation_ops:
        if not sim_signal_op.gpus_to_allocate:
            continue
        needed = set(sim_signal_op.gpus_to_allocate)
        if not needed.issubset(sim_idle):
            raise ValidationError(
                "allocation consumes non-idle GPUs",
                condition=4,
                context={"cluster_id": sim_signal_op.cluster_id, "gpus": sorted(needed - sim_idle)},
            )
        sim_idle -= needed
        tp_size = _cluster_tp_size(inputs, sim_signal_op.cluster_id)
        dp_rank_to_gpus = build_dp_rank_mapping(sorted(needed), tp_size)
        active_dp_ranks = set(dp_rank_to_gpus.keys()) if is_generation_cluster(sim_signal_op.cluster_id) else set()
        sim_allocations[sim_signal_op.cluster_id] = ClusterAllocation(
            cluster_id=sim_signal_op.cluster_id,
            gpu_ids=sorted(needed),
            priority=Priority(sim_signal_op.priority) if sim_signal_op.priority is not None else Priority.GENERATION,
            active_dp_ranks=active_dp_ranks,
            dp_rank_to_gpus=dp_rank_to_gpus,
        )

    # Apply expansions (generation dp-rank add).
    for sim_alloc_op in plan.sched_guided_allocation_ops:
        tp_size = _cluster_tp_size(inputs, sim_alloc_op.cluster_id)
        if len(sim_alloc_op.gpus_to_allocate) % tp_size != 0:
            raise ValidationError(
                "expansion gpus_to_allocate not divisible by tp_size",
                condition=6,
                context={
                    "cluster_id": sim_alloc_op.cluster_id,
                    "tp_size": tp_size,
                    "len": len(sim_alloc_op.gpus_to_allocate),
                },
            )
        needed = set(sim_alloc_op.gpus_to_allocate)
        if not needed.issubset(sim_idle):
            raise ValidationError(
                "expansion consumes non-idle GPUs",
                condition=4,
                context={"cluster_id": sim_alloc_op.cluster_id, "gpus": sorted(needed - sim_idle)},
            )
        sim_idle -= needed

        sim_expand_alloc = sim_allocations.get(sim_alloc_op.cluster_id)
        if sim_expand_alloc is None:
            sim_expand_alloc = ClusterAllocation(
                cluster_id=sim_alloc_op.cluster_id,
                gpu_ids=[],
                priority=Priority.GENERATION,
                active_dp_ranks=set(),
                dp_rank_to_gpus={},
            )
            sim_allocations[sim_alloc_op.cluster_id] = sim_expand_alloc

        max_dp = _max_dp_workers(inputs, sim_alloc_op.cluster_id)
        validate_dp_ranks_to_add(dp_ranks_to_add=sim_alloc_op.dp_ranks_to_add, max_dp_ranks=max_dp)
        if len(set(sim_expand_alloc.active_dp_ranks) | set(sim_alloc_op.dp_ranks_to_add)) > max_dp:
            raise ValidationError(
                "expansion exceeds max_dp_workers",
                condition=7,
                context={
                    "cluster_id": sim_alloc_op.cluster_id,
                    "max_dp_workers": max_dp,
                    "add": sim_alloc_op.dp_ranks_to_add,
                },
            )
        if set(sim_alloc_op.dp_ranks_to_add) & set(sim_expand_alloc.active_dp_ranks):
            raise ValidationError(
                "expansion adds already-active dp ranks",
                condition=5,
                context={
                    "cluster_id": sim_alloc_op.cluster_id,
                    "active": sorted(sim_expand_alloc.active_dp_ranks),
                    "add": sim_alloc_op.dp_ranks_to_add,
                },
            )

        # Condition 10: DP-rank overlap within a cluster.
        for dp_rank in sim_alloc_op.dp_ranks_to_add:
            if dp_rank in sim_expand_alloc.dp_rank_to_gpus:
                raise ValidationError(
                    "dp_rank already exists in dp_rank_to_gpus",
                    condition=10,
                    context={"cluster_id": sim_alloc_op.cluster_id, "dp_rank": dp_rank},
                )

        # Assign bundles by deterministic sort: dp_ranks_to_add order must match provided gpus_to_allocate bundles.
        sorted_needed = sorted(needed)
        if len(sorted_needed) != len(sim_alloc_op.dp_ranks_to_add) * tp_size:
            raise ValidationError(
                "expansion bundle size mismatch",
                condition=6,
                context={"cluster_id": sim_alloc_op.cluster_id},
            )
        for i, dp_rank in enumerate(sorted(sim_alloc_op.dp_ranks_to_add)):
            gpu_slice = sorted_needed[i * tp_size : (i + 1) * tp_size]
            sim_expand_alloc.dp_rank_to_gpus[dp_rank] = list(gpu_slice)
            sim_expand_alloc.active_dp_ranks.add(dp_rank)
            sim_expand_alloc.gpu_ids.extend(gpu_slice)
        sim_expand_alloc.gpu_ids = sorted(set(sim_expand_alloc.gpu_ids))

    # Condition 8 (conservation): idle + allocated cover a consistent GPU universe.
    # Universe is derived from all device_mappings across registered clusters.
    universe: Set[int] = set()
    for pipeline in inputs.pipeline_registry.values():
        for cfg in (pipeline.get("cluster_configs") or {}).values():
            universe |= set(cfg.get("device_mapping") or [])

    # Include the scheduler's currently known GPU pool so subset device_mappings do not
    # trip false positives when some GPUs are intentionally left unmapped by a pipeline.
    known_gpu_pool = set(inputs.idle_gpus) | {g for a in inputs.active_allocations.values() for g in a.gpu_ids}
    if universe:
        universe |= known_gpu_pool
    else:
        universe = known_gpu_pool
    final_allocated = {g for a in sim_allocations.values() for g in a.gpu_ids}
    if (final_allocated | sim_idle) != universe:
        missing_from_accounting = sorted(universe - (final_allocated | sim_idle))
        raise ValidationError(
            "GPU accounting does not cover device_mapping universe",
            condition=8,
            context={"missing": missing_from_accounting},
        )
    if final_allocated & sim_idle:
        raise ValidationError(
            "final idle_gpus overlaps final allocated GPUs",
            condition=4,
            context={"overlap": sorted(final_allocated & sim_idle)},
        )

    # Condition 10 (GPU overlap across clusters): each GPU appears in at most one allocation.
    owners: Dict[int, str] = {}
    for cid, alloc in sim_allocations.items():
        for gpu in alloc.gpu_ids:
            prev = owners.get(gpu)
            if prev is not None:
                raise ValidationError(
                    "GPU overlap across allocations",
                    condition=10,
                    context={"gpu_id": gpu, "cluster_a": prev, "cluster_b": cid},
                )
            owners[gpu] = cid
