from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from schedrl.protocol.types import Priority
from schedrl.scheduler.types import (
    ClusterAllocation,
    ExecutionPlan,
    ValidationError,
    build_dp_rank_mapping,
    is_generation_cluster,
    parse_cluster_id,
)


@dataclass(frozen=True, slots=True)
class ValidationInputs:
    pipeline_registry: Dict[str, Dict[str, Any]]
    active_allocations: Dict[str, ClusterAllocation]
    idle_gpus: Set[int]


def _cluster_config(inputs: ValidationInputs, cluster_id: str) -> Dict[str, Any]:
    pipeline_id, cluster_name = parse_cluster_id(cluster_id)
    pipeline = inputs.pipeline_registry.get(pipeline_id)
    if pipeline is None:
        raise KeyError(f"pipeline_id {pipeline_id!r} not registered")
    cluster_configs = pipeline.get("cluster_configs") or {}
    config = cluster_configs.get(cluster_name)
    if config is None:
        raise KeyError(f"cluster_name {cluster_name!r} not registered for pipeline_id {pipeline_id!r}")
    return config


def _cluster_tp_size(inputs: ValidationInputs, cluster_id: str) -> int:
    tp_size = int(_cluster_config(inputs, cluster_id).get("tp_size", 1))
    if tp_size <= 0:
        raise ValueError(f"Invalid tp_size={tp_size} for cluster_id {cluster_id!r}")
    return tp_size


def _cluster_device_mapping(inputs: ValidationInputs, cluster_id: str) -> List[int]:
    mapping = list(_cluster_config(inputs, cluster_id).get("device_mapping") or [])
    if not mapping:
        raise ValueError(f"Missing device_mapping for cluster_id {cluster_id!r}")
    return mapping


def _max_dp_workers(inputs: ValidationInputs, cluster_id: str) -> int:
    cfg = _cluster_config(inputs, cluster_id)
    tp_size = int(cfg.get("tp_size", 1))
    device_mapping = list(cfg.get("device_mapping") or [])
    if not device_mapping:
        return 0
    max_dp = cfg.get("max_dp_workers")
    if max_dp is None:
        return len(device_mapping) // tp_size if tp_size > 0 else 0
    return int(max_dp)


def validate_execution_plan(plan: ExecutionPlan, *, inputs: ValidationInputs) -> None:
    """Fail-fast validation for the scheduler execution plan (11 critical conditions).

    This is intentionally strict and is designed to catch plan/state inconsistencies before execution.
    """

    # Condition 4 (GPU state consistency): idle and allocated disjoint at entry.
    allocated_gpus = {gpu for alloc in inputs.active_allocations.values() for gpu in alloc.gpu_ids}
    if not allocated_gpus.isdisjoint(inputs.idle_gpus):
        raise ValidationError(
            "idle_gpus overlaps allocated GPUs",
            condition=4,
            context={"overlap": sorted(allocated_gpus & inputs.idle_gpus)},
        )

    # Condition 1 (operation uniqueness): handled by construction for sched_guided_shrink_ops.

    # Condition 3 (mutual exclusivity): cannot both allocate and remove the same cluster in one cycle.
    alloc_targets = {op.cluster_id for op in plan.signal_pending_allocation_ops} | {op.cluster_id for op in plan.sched_guided_allocation_ops}
    alloc_remove_overlap = alloc_targets & set(plan.clusters_to_remove)
    if alloc_remove_overlap:
        raise ValidationError(
            "cluster_id appears in both allocation ops and clusters_to_remove",
            condition=3,
            context={"cluster_ids": sorted(alloc_remove_overlap)},
        )

    # Condition 2 (state transitions): non-generation clusters must not be shrunk/expanded via DP-rank ops.
    for op in list(plan.sched_guided_shrink_ops):
        if not is_generation_cluster(op.cluster_id):
            raise ValidationError("shrink op applied to non-generation cluster", condition=2, context={"cluster_id": op.cluster_id})
    for op in plan.sched_guided_allocation_ops:
        if not is_generation_cluster(op.cluster_id):
            raise ValidationError(
                "generation allocation op applied to non-generation cluster",
                condition=2,
                context={"cluster_id": op.cluster_id},
            )

    # Condition 11 (consistency): clusters_to_remove must only contain clusters that exist.
    missing = [cid for cid in plan.clusters_to_remove if cid not in inputs.active_allocations]
    if missing:
        raise ValidationError("clusters_to_remove contains unknown cluster_id", condition=11, context={"cluster_ids": missing})

    # Condition 6 (device mapping boundaries) and Condition 7 (capacity limits) and Condition 5 (dp-rank activity).
    for op in plan.signal_pending_allocation_ops:
        device_mapping = set(_cluster_device_mapping(inputs, op.cluster_id))
        if not set(op.gpus_to_allocate).issubset(device_mapping):
            raise ValidationError(
                "allocation uses GPU IDs outside device_mapping",
                condition=6,
                context={"cluster_id": op.cluster_id, "gpus": sorted(set(op.gpus_to_allocate) - device_mapping)},
            )

    # Simulate plan effects to validate global GPU accounting and overlaps.
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
        alloc = sim_allocations.get(cluster_id)
        if alloc is None:
            return
        if alloc.dp_rank_to_gpus:
            return
        tp_size = _cluster_tp_size(inputs, cluster_id)
        alloc.dp_rank_to_gpus = build_dp_rank_mapping(alloc.gpu_ids, tp_size)

    # Apply shrinks.
    for op in plan.sched_guided_shrink_ops:
        if not op.dp_ranks_to_remove:
            continue
        alloc = sim_allocations.get(op.cluster_id)
        if alloc is None:
            continue
        if alloc.priority != Priority.GENERATION:
            raise ValidationError("shrink on non-generation allocation", condition=2, context={"cluster_id": op.cluster_id})
        _ensure_dp_rank_mapping(op.cluster_id)
        active = set(alloc.active_dp_ranks)
        if not set(op.dp_ranks_to_remove).issubset(active):
            raise ValidationError(
                "shrink removes inactive dp ranks",
                condition=5,
                context={"cluster_id": op.cluster_id, "active": sorted(active), "remove": op.dp_ranks_to_remove},
            )
        for dp_rank in op.dp_ranks_to_remove:
            bundle = set(alloc.dp_rank_to_gpus.get(dp_rank) or [])
            if not bundle:
                raise ValidationError(
                    "missing dp_rank_to_gpus bundle for dp_rank",
                    condition=6,
                    context={"cluster_id": op.cluster_id, "dp_rank": dp_rank},
                )
            _free_bundle(op.cluster_id, bundle)
            alloc.active_dp_ranks.discard(dp_rank)
            alloc.dp_rank_to_gpus.pop(dp_rank, None)
            alloc.gpu_ids = [g for g in alloc.gpu_ids if g not in bundle]

    # Apply clusters_to_remove (any remaining GPUs are freed).
    for cluster_id in sorted(plan.clusters_to_remove):
        alloc = sim_allocations.pop(cluster_id, None)
        if alloc is None:
            continue
        _free_bundle(cluster_id, set(alloc.gpu_ids))

    # Apply pending allocations (initial allocations).
    for op in plan.signal_pending_allocation_ops:
        if not op.gpus_to_allocate:
            continue
        needed = set(op.gpus_to_allocate)
        if not needed.issubset(sim_idle):
            raise ValidationError(
                "allocation consumes non-idle GPUs",
                condition=4,
                context={"cluster_id": op.cluster_id, "gpus": sorted(needed - sim_idle)},
            )
        sim_idle -= needed
        tp_size = _cluster_tp_size(inputs, op.cluster_id)
        dp_rank_to_gpus = build_dp_rank_mapping(sorted(needed), tp_size)
        active_dp_ranks = set(dp_rank_to_gpus.keys()) if is_generation_cluster(op.cluster_id) else set()
        sim_allocations[op.cluster_id] = ClusterAllocation(
            cluster_id=op.cluster_id,
            gpu_ids=sorted(needed),
            priority=Priority(op.priority) if op.priority is not None else Priority.GENERATION,
            active_dp_ranks=active_dp_ranks,
            dp_rank_to_gpus=dp_rank_to_gpus,
        )

    # Apply expansions (generation dp-rank add).
    for op in plan.sched_guided_allocation_ops:
        tp_size = _cluster_tp_size(inputs, op.cluster_id)
        if len(op.gpus_to_allocate) % tp_size != 0:
            raise ValidationError(
                "expansion gpus_to_allocate not divisible by tp_size",
                condition=6,
                context={"cluster_id": op.cluster_id, "tp_size": tp_size, "len": len(op.gpus_to_allocate)},
            )
        needed = set(op.gpus_to_allocate)
        if not needed.issubset(sim_idle):
            raise ValidationError(
                "expansion consumes non-idle GPUs",
                condition=4,
                context={"cluster_id": op.cluster_id, "gpus": sorted(needed - sim_idle)},
            )
        sim_idle -= needed

        alloc = sim_allocations.get(op.cluster_id)
        if alloc is None:
            alloc = ClusterAllocation(
                cluster_id=op.cluster_id,
                gpu_ids=[],
                priority=Priority.GENERATION,
                active_dp_ranks=set(),
                dp_rank_to_gpus={},
            )
            sim_allocations[op.cluster_id] = alloc

        max_dp = _max_dp_workers(inputs, op.cluster_id)
        if len(set(alloc.active_dp_ranks) | set(op.dp_ranks_to_add)) > max_dp:
            raise ValidationError(
                "expansion exceeds max_dp_workers",
                condition=7,
                context={"cluster_id": op.cluster_id, "max_dp_workers": max_dp, "add": op.dp_ranks_to_add},
            )
        if set(op.dp_ranks_to_add) & set(alloc.active_dp_ranks):
            raise ValidationError(
                "expansion adds already-active dp ranks",
                condition=5,
                context={"cluster_id": op.cluster_id, "active": sorted(alloc.active_dp_ranks), "add": op.dp_ranks_to_add},
            )

        # Condition 10: DP-rank overlap within a cluster.
        for dp_rank in op.dp_ranks_to_add:
            if dp_rank in alloc.dp_rank_to_gpus:
                raise ValidationError(
                    "dp_rank already exists in dp_rank_to_gpus",
                    condition=10,
                    context={"cluster_id": op.cluster_id, "dp_rank": dp_rank},
                )

        # Assign bundles by deterministic sort: dp_ranks_to_add order must match provided gpus_to_allocate bundles.
        sorted_needed = sorted(needed)
        if len(sorted_needed) != len(op.dp_ranks_to_add) * tp_size:
            raise ValidationError(
                "expansion bundle size mismatch",
                condition=6,
                context={"cluster_id": op.cluster_id},
            )
        for i, dp_rank in enumerate(sorted(op.dp_ranks_to_add)):
            bundle = sorted_needed[i * tp_size : (i + 1) * tp_size]
            alloc.dp_rank_to_gpus[dp_rank] = list(bundle)
            alloc.active_dp_ranks.add(dp_rank)
            alloc.gpu_ids.extend(bundle)
        alloc.gpu_ids = sorted(set(alloc.gpu_ids))

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


def normalize_progress_oldest_ts(oldest_unfinished_creation_ts: Optional[float], fifo_timestamp: Optional[float]) -> Optional[float]:
    if oldest_unfinished_creation_ts is not None:
        return oldest_unfinished_creation_ts
    return fifo_timestamp
