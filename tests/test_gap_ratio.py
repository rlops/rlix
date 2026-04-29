"""Behavioral tests for the extracted gap-ratio planning module.

Test 1: single-pipeline idle GPU activation (free-GPU path).
Test 2: two-pipeline donor shrink (donor-search path).
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same stub pattern as test_scheduler_apply_plan_invariants.py."""
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")

    def _remote(*args, **kwargs):
        def _decorate(obj):
            return obj
        return _decorate

    ray_stub.remote = _remote
    ray_stub.get_actor = lambda *args, **kwargs: None
    ray_stub.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.scheduler": RLIX_ROOT / "scheduler",
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_gap_ratio_modules(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    gap_ratio_mod = importlib.import_module("rlix.scheduler.planner")
    scheduler_types = importlib.import_module("rlix.scheduler.types")
    protocol_types = importlib.import_module("rlix.protocol.types")
    return gap_ratio_mod, scheduler_types, protocol_types


def test_single_pipeline_idle_gpus_activated(monkeypatch: pytest.MonkeyPatch) -> None:
    """One pipeline with 2 idle generation GPUs -> gap-ratio activates both dp workers."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_000000000000"
    cluster_id = f"{pipeline_id}_actor_infer"

    # Construct inputs: 0 active dp workers, 2 inactive, 2 idle GPUs
    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[2]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[3]),
        ]
    }

    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [2, 3],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }

    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
        )
    ]

    # 50% remaining -> nonzero demand weight
    def progress_totals_fn(*, pipeline_id):
        return (50.0, 100.0)

    remaining_idle = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={2, 3},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    # Assert: both dp workers activated, all idle GPUs consumed
    assert len(plan.sched_guided_allocation_ops) == 1
    op = plan.sched_guided_allocation_ops[0]
    assert op.cluster_id == cluster_id
    assert {gpu_id for gpus in op.dp_rank_to_gpus_to_add.values() for gpu_id in gpus} == {2, 3}
    assert remaining_idle == set()
    # No shrink ops needed (GPUs were free, no donors)
    assert len(plan.sched_guided_shrink_ops) == 0


def test_pending_request_uses_step_target_estimate_without_progress_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pending GENERATION request can bootstrap demand from its explicit estimate."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_222222222222"
    cluster_id = f"{pipeline_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[1]),
        ]
    }
    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
            step_target_estimate=4,
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (0.0, 0.0)

    remaining_idle = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    assert len(plan.sched_guided_allocation_ops) == 1
    op = plan.sched_guided_allocation_ops[0]
    assert op.cluster_id == cluster_id
    all_gpus = {g for gpus in op.dp_rank_to_gpus_to_add.values() for g in gpus}
    assert all_gpus
    assert all_gpus.issubset({0, 1})
    assert op.dp_rank_to_gpus_to_add
    assert remaining_idle != {0, 1}


def test_pending_request_without_progress_or_estimate_does_not_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """No progress snapshot and no estimate means no synthetic demand is invented."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_333333333333"
    cluster_id = f"{pipeline_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[1]),
        ]
    }
    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (0.0, 0.0)

    remaining_idle = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    assert plan.sched_guided_allocation_ops == []
    assert remaining_idle == {0, 1}


def test_real_progress_overrides_pending_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reported progress, when present, takes precedence over request-carried estimates."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    plan = ExecutionPlan()
    pipeline_id = "ft_444444444444"
    cluster_id = f"{pipeline_id}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker
    active_dp_workers = {pipeline_id: []}
    inactive_dp_workers = {
        pipeline_id: [
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=0, gpu_ids=[0]),
            _GapRatioDPWorker(pipeline_id=pipeline_id, dp_rank=1, gpu_ids=[1]),
        ]
    }
    pipeline_registry = {
        pipeline_id: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1],
                    "max_dp_workers": 2,
                },
            },
            "admitted": True,
        }
    }
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_id, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
            step_target_estimate=1000,
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        return (5.0, 10.0)

    gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus={0, 1},
        pipeline_registry=pipeline_registry,
        active_allocations={},
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    assert len(plan.sched_guided_allocation_ops) == 1
    op = plan.sched_guided_allocation_ops[0]
    assert {g for gpus in op.dp_rank_to_gpus_to_add.values() for g in gpus} == {0, 1}


def test_two_pipelines_donor_shrink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over-provisioned pipeline donates GPUs to under-provisioned pipeline."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest
    ClusterAllocation = scheduler_types.ClusterAllocation

    plan = ExecutionPlan()
    pipeline_a = "ft_000000000000"  # Nearly complete -> low demand -> over-provisioned
    pipeline_b = "ft_111111111111"  # Just started -> high demand -> under-provisioned
    cluster_a = f"{pipeline_a}_actor_infer"
    cluster_b = f"{pipeline_b}_actor_infer"

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker

    # Pipeline A: 4 active generation workers, no inactive
    # Pipeline B: 0 active, 4 inactive (same GPU pool — time-shared)
    active_dp_workers = {
        pipeline_a: [
            _GapRatioDPWorker(pipeline_id=pipeline_a, dp_rank=rank, gpu_ids=[rank]) for rank in range(4)
        ],
        pipeline_b: [],
    }
    inactive_dp_workers = {
        pipeline_a: [],
        pipeline_b: [
            _GapRatioDPWorker(pipeline_id=pipeline_b, dp_rank=rank, gpu_ids=[rank]) for rank in range(4)
        ],
    }

    pipeline_registry = {
        pid: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0, 1, 2, 3],
                    "max_dp_workers": 4,
                },
            },
            "admitted": True,
        }
        for pid in [pipeline_a, pipeline_b]
    }

    active_allocations = {
        cluster_a: ClusterAllocation(
            cluster_id=cluster_a,
            gpu_ids=[0, 1, 2, 3],
            priority=Priority.GENERATION,
            active_dp_ranks={0, 1, 2, 3},
            dp_rank_to_gpus={0: [0], 1: [1], 2: [2], 3: [3]},
        ),
    }

    # Only Pipeline B has a pending request (drives demand inflation)
    pending_bucket_gen = [
        PendingRequest(
            request=Request(cluster_id=cluster_b, priority=Priority.GENERATION, timestamp=0.0),
            event=asyncio.Event(),
        )
    ]

    def progress_totals_fn(*, pipeline_id):
        if pipeline_id == pipeline_a:
            return (10.0, 100.0)  # 10% remaining -> low demand weight
        return (90.0, 100.0)  # 90% remaining -> high demand weight (+ inflation)

    remaining = gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus=set(),  # No free GPUs — must donate from Pipeline A
        pipeline_registry=pipeline_registry,
        active_allocations=active_allocations,
        pending_bucket_gen=pending_bucket_gen,
        progress_totals_fn=progress_totals_fn,
    )

    # Assert: Pipeline A shrunk (donor), Pipeline B expanded (receiver)
    assert any(op.cluster_id == cluster_a for op in plan.sched_guided_shrink_ops)
    assert any(op.cluster_id == cluster_b for op in plan.sched_guided_allocation_ops)

    # Pipeline B got at least one GPU
    b_gpus = set()
    for op in plan.sched_guided_allocation_ops:
        if op.cluster_id == cluster_b:
            for gpus in op.dp_rank_to_gpus_to_add.values():
                b_gpus.update(gpus)
    assert len(b_gpus) >= 1


def test_no_donor_mutation_when_receiver_ineligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Receiver with no pending request and no active allocation must not trigger donor shrinks.

    Regression test: previously _try_activate_one committed donor shrink mutations before
    checking receiver eligibility, leaving orphaned shrink ops if the guard fired.
    """
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    Priority = protocol_types.Priority
    ClusterAllocation = scheduler_types.ClusterAllocation

    plan = ExecutionPlan()
    donor_id = "ft_000000000000"  # Has active workers, can donate
    receiver_id = "ft_111111111111"  # No pending request, no active allocation -> ineligible

    _GapRatioDPWorker = gap_ratio_mod._GapRatioDPWorker

    active_dp_workers = {
        donor_id: [_GapRatioDPWorker(pipeline_id=donor_id, dp_rank=0, gpu_ids=[0])],
        receiver_id: [],
    }
    inactive_dp_workers = {
        donor_id: [],
        receiver_id: [_GapRatioDPWorker(pipeline_id=receiver_id, dp_rank=0, gpu_ids=[0])],
    }

    pipeline_registry = {
        pid: {
            "cluster_configs": {
                "actor_infer": {
                    "tp_size": 1,
                    "is_generation": True,
                    "device_mapping": [0],
                    "max_dp_workers": 1,
                },
            },
        }
        for pid in [donor_id, receiver_id]
    }

    active_allocations = {
        f"{donor_id}_actor_infer": ClusterAllocation(
            cluster_id=f"{donor_id}_actor_infer",
            gpu_ids=[0],
            priority=Priority.GENERATION,
            active_dp_ranks={0},
            dp_rank_to_gpus={0: [0]},
        ),
    }

    def progress_totals_fn(*, pipeline_id):
        return (50.0, 100.0)

    gap_ratio_mod.plan_generation_gap_ratio(
        plan,
        active_dp_workers=active_dp_workers,
        inactive_dp_workers=inactive_dp_workers,
        non_gen_reserved_gpus=set(),
        idle_gpus=set(),  # No free GPUs — would need to donate
        pipeline_registry=pipeline_registry,
        active_allocations=active_allocations,
        pending_bucket_gen=[],  # No pending request for receiver
        progress_totals_fn=progress_totals_fn,
    )

    # No shrink ops should have been added — donor must not be mutated for an ineligible receiver
    assert len(plan.sched_guided_shrink_ops) == 0
    assert len(plan.sched_guided_allocation_ops) == 0


def test_snapshot_fails_fast_when_actor_infer_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """snapshot_generation_dp_workers must raise KeyError for a registered pipeline missing actor_infer."""
    gap_ratio_mod, scheduler_types, protocol_types = _load_gap_ratio_modules(monkeypatch)

    ExecutionPlan = scheduler_types.ExecutionPlan
    plan = ExecutionPlan()

    pipeline_registry = {
        "ft_000000000000": {
            "cluster_configs": {
                "actor_train": {"tp_size": 1, "device_mapping": [0, 1]},
                # actor_infer intentionally missing
            },
        }
    }

    with pytest.raises(KeyError, match="missing actor_infer"):
        gap_ratio_mod.snapshot_generation_dp_workers(
            plan=plan,
            idle_gpus={0, 1},
            pipeline_registry=pipeline_registry,
            active_allocations={},
        )
