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
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_scheduler_modules(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    scheduler_module = importlib.import_module("rlix.scheduler.scheduler")
    protocol_types = importlib.import_module("rlix.protocol.types")
    scheduler_types = importlib.import_module("rlix.scheduler.types")
    return scheduler_module, protocol_types, scheduler_types


def test_initial_allocation_raises_on_corruption_when_waiter_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline gone from registry BUT pending waiter still present → RuntimeError (corruption)."""
    scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

    scheduler = scheduler_module.SchedulerImpl()
    cluster_id = "ft_abc123def456_actor_train"
    priority = protocol_types.Priority.ACTOR_TRAINING
    scheduler._state.idle_gpus = {0, 1, 2, 3}
    scheduler._state.pending_bucket(priority).append(
        scheduler_types.PendingRequest(
            request=scheduler_types.Request(cluster_id=cluster_id, priority=priority, timestamp=0.0),
            event=asyncio.Event(),
        )
    )
    plan = scheduler_types.ExecutionPlan(
        signal_pending_allocation_ops=[
            scheduler_types.SignalPendingAllocationOp(
                cluster_id=cluster_id,
                gpus_to_allocate=[0, 1],
                priority=priority,
                tp_size=1,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="cleanup corruption"):
        scheduler._apply_plan_and_signal(plan)

    assert scheduler._state.idle_gpus == {0, 1, 2, 3}
    assert scheduler._state.active_allocations == {}
    assert len(scheduler._state.pending_bucket(priority)) == 1


def test_generation_expand_skips_stale_op_after_unregister(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline gone from registry, no waiter → skip gracefully, state unchanged."""
    scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

    scheduler = scheduler_module.SchedulerImpl()
    cluster_id = "ft_abc123def456_actor_infer"
    scheduler._state.idle_gpus = {0, 1, 2, 3}
    scheduler._state.active_allocations[cluster_id] = scheduler_types.ClusterAllocation(
        cluster_id=cluster_id,
        gpu_ids=[4, 5],
        priority=protocol_types.Priority.GENERATION,
        active_dp_ranks={0},
        dp_rank_to_gpus={0: [4, 5]},
    )
    plan = scheduler_types.ExecutionPlan(
        sched_guided_allocation_ops=[
            scheduler_types.SchedGuidedAllocationOp(
                cluster_id=cluster_id,
                dp_ranks_to_add=[1],
                gpus_to_allocate=[0, 1],
                tp_size=2,
            )
        ]
    )

    # No crash — op is skipped because pipeline_id is not in registry
    scheduler._apply_plan_and_signal(plan)

    allocation = scheduler._state.active_allocations[cluster_id]
    assert scheduler._state.idle_gpus == {0, 1, 2, 3}
    assert allocation.gpu_ids == [4, 5]
    assert allocation.active_dp_ranks == {0}
    assert allocation.dp_rank_to_gpus == {0: [4, 5]}


def test_commit_skips_stale_signal_op_after_unregister(monkeypatch: pytest.MonkeyPatch) -> None:
    """True unregister race: pipeline gone, waiter already removed → skip with warning."""
    scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

    scheduler = scheduler_module.SchedulerImpl()
    pipeline_id = "ft_abc123def456"
    cluster_id = "ft_abc123def456_actor_train"
    priority = protocol_types.Priority.ACTOR_TRAINING

    # Set up: pipeline in registry with a pending waiter
    scheduler._state.pipeline_registry[pipeline_id] = {
        "cluster_configs": {"actor_train": {"tp_size": 1, "device_mapping": [0, 1]}},
    }
    scheduler._state.idle_gpus = {0, 1, 2, 3}
    pending = scheduler_types.PendingRequest(
        request=scheduler_types.Request(cluster_id=cluster_id, priority=priority, timestamp=0.0),
        event=asyncio.Event(),
    )
    scheduler._state.pending_bucket(priority).append(pending)

    plan = scheduler_types.ExecutionPlan(
        signal_pending_allocation_ops=[
            scheduler_types.SignalPendingAllocationOp(
                cluster_id=cluster_id,
                gpus_to_allocate=[0, 1],
                priority=priority,
                tp_size=1,
            )
        ]
    )

    # Simulate unregister_pipeline cleanup: remove from registry, remove waiter, signal error
    scheduler._state.pipeline_registry.pop(pipeline_id)
    bucket = scheduler._state.pending_bucket(priority)
    pending.error = f"Pipeline {pipeline_id!r} unregistered"
    pending.event.set()
    bucket.remove(pending)

    # Call _apply_plan_and_signal — should skip stale op, no crash
    scheduler._apply_plan_and_signal(plan)

    assert scheduler._state.idle_gpus == {0, 1, 2, 3}
    assert scheduler._state.active_allocations == {}
    assert len(scheduler._state.pending_bucket(priority)) == 0


def test_unregister_during_lock_gap_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates the lock-gap race: plan built with pipeline present, commit runs after unregister.

    This mirrors what happens when unregister_pipeline runs between _execute_resize_calls
    (outside lock) and _apply_plan_and_signal (re-acquires lock): the plan references a
    pipeline that no longer exists in the registry.
    """
    scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

    scheduler = scheduler_module.SchedulerImpl()
    pipeline_id = "ft_abc123def456"
    cluster_id_train = "ft_abc123def456_actor_train"
    cluster_id_infer = "ft_abc123def456_actor_infer"
    priority = protocol_types.Priority.ACTOR_TRAINING

    # State as it was when planning happened (pipeline registered, waiter present)
    scheduler._state.pipeline_registry[pipeline_id] = {
        "cluster_configs": {
            "actor_train": {"tp_size": 1, "device_mapping": [0, 1]},
            "actor_infer": {"tp_size": 2, "device_mapping": [2, 3]},
        },
    }
    scheduler._state.idle_gpus = {0, 1, 2, 3}

    # Plan was built while pipeline was registered
    plan = scheduler_types.ExecutionPlan(
        signal_pending_allocation_ops=[
            scheduler_types.SignalPendingAllocationOp(
                cluster_id=cluster_id_train,
                gpus_to_allocate=[0, 1],
                priority=priority,
                tp_size=1,
            )
        ],
        sched_guided_allocation_ops=[
            scheduler_types.SchedGuidedAllocationOp(
                cluster_id=cluster_id_infer,
                dp_ranks_to_add=[0],
                gpus_to_allocate=[2, 3],
                tp_size=2,
            )
        ],
    )

    # Simulate unregister_pipeline cleanup (happens during lock gap)
    scheduler._state.pipeline_registry.pop(pipeline_id)
    # Waiter was already removed by unregister_pipeline (no pending requests remain)

    # Commit phase: _apply_plan_and_signal should skip both stale ops, no crash
    scheduler._apply_plan_and_signal(plan)

    # All GPUs remain idle — no stale allocation was committed
    assert scheduler._state.idle_gpus == {0, 1, 2, 3}
    assert scheduler._state.active_allocations == {}
