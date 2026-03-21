"""Pre-refactor regression tests for scheduling_cycle().

Two tests exercise scheduling_cycle() end-to-end against the CURRENT scheduler.
Both must pass before AND after each extraction step (tracer, gap-ratio).
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


def _load_scheduler_modules(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    scheduler_module = importlib.import_module("rlix.scheduler.scheduler")
    protocol_types = importlib.import_module("rlix.protocol.types")
    scheduler_types = importlib.import_module("rlix.scheduler.types")
    return scheduler_module, protocol_types, scheduler_types


class _FakeRemote:
    """Stub for coordinator.resize_infer — returns an awaitable from .remote()."""

    def remote(self, **kwargs):
        async def _noop():
            pass

        return _noop()


class _FakeCoordinator:
    """Stub coordinator with no-op resize_infer.remote()."""

    resize_infer = _FakeRemote()


def test_non_gen_allocation_through_scheduling_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 2 non-GEN allocation + Phase 6 signal through scheduling_cycle().

    Exercises: Phase 0.5 (skip) -> Phase 2 (non-GEN alloc) -> Phase 3 (skip, no progress)
    -> Phase 4 (validation) -> Phase 5 (no resize calls) -> Phase 6 (commit + signal).
    """
    scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

    Priority = protocol_types.Priority
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest

    scheduler = scheduler_module.SchedulerImpl()
    scheduler._topology_ready.set()
    scheduler._num_gpus = 4
    scheduler._required_gpus_per_node = 4
    scheduler._state.idle_gpus = {0, 1, 2, 3}

    pipeline_id = "ft_abc123def456"
    cluster_id = f"{pipeline_id}_actor_train"
    scheduler._state.pipeline_registry[pipeline_id] = {
        "namespace": "test_ns",
        "cluster_configs": {
            "actor_train": {
                "tp_size": 1,
                "is_generation": False,
                "device_mapping": [0, 1],
            },
            "actor_infer": {
                "tp_size": 1,
                "is_generation": True,
                "device_mapping": [2, 3],
                "max_dp_workers": 2,
            },
        },
        "admitted": True,
    }

    pending = PendingRequest(
        request=Request(cluster_id=cluster_id, priority=Priority.ACTOR_TRAINING, timestamp=0.0),
        event=asyncio.Event(),
    )
    scheduler._state.pending_bucket(Priority.ACTOR_TRAINING).append(pending)

    asyncio.run(scheduler.scheduling_cycle())

    assert pending.event.is_set()
    assert pending.result == [0, 1]
    assert scheduler._state.idle_gpus == {2, 3}
    assert cluster_id in scheduler._state.active_allocations
    alloc = scheduler._state.active_allocations[cluster_id]
    assert alloc.gpu_ids == [0, 1]
    assert alloc.priority == Priority.ACTOR_TRAINING


def test_gen_allocation_through_scheduling_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 3 gap-ratio planning + Phase 5 coordinator resize + Phase 6 commit.

    Proves scheduling_cycle() correctly calls snapshot_generation_dp_workers ->
    plan_generation_gap_ratio -> Phase 4 validation -> Phase 5 coordinator resize
    (via stubs) -> Phase 6 commit.
    """
    scheduler_module, protocol_types, scheduler_types = _load_scheduler_modules(monkeypatch)

    Priority = protocol_types.Priority
    ProgressReport = protocol_types.ProgressReport
    Request = scheduler_types.Request
    PendingRequest = scheduler_types.PendingRequest
    ClusterAllocation = scheduler_types.ClusterAllocation

    scheduler = scheduler_module.SchedulerImpl()
    scheduler._topology_ready.set()
    scheduler._num_gpus = 4
    scheduler._required_gpus_per_node = 4
    scheduler._state.idle_gpus = {2, 3}

    pipeline_id = "ft_000000000000"
    train_cluster_id = f"{pipeline_id}_actor_train"
    infer_cluster_id = f"{pipeline_id}_actor_infer"

    scheduler._state.pipeline_registry[pipeline_id] = {
        "namespace": "test_ns",
        "cluster_configs": {
            "actor_train": {
                "tp_size": 2,
                "is_generation": False,
                "device_mapping": [0, 1],
            },
            "actor_infer": {
                "tp_size": 1,
                "is_generation": True,
                "device_mapping": [2, 3],
                "max_dp_workers": 2,
            },
        },
        "admitted": True,
    }

    # Active allocation for actor_train (GPUs 0,1 already in use)
    scheduler._state.active_allocations[train_cluster_id] = ClusterAllocation(
        cluster_id=train_cluster_id,
        gpu_ids=[0, 1],
        priority=Priority.ACTOR_TRAINING,
    )

    # Progress: 50% remaining -> nonzero demand weight for gap-ratio planning
    scheduler._state.latest_progress_by_pipeline[pipeline_id] = {
        "default": {
            "default": ProgressReport(
                pipeline_id=pipeline_id,
                step_target_trajectories=100,
                metrics={"completed": 50},
            )
        }
    }

    # Pending GENERATION request
    pending = PendingRequest(
        request=Request(cluster_id=infer_cluster_id, priority=Priority.GENERATION, timestamp=0.0),
        event=asyncio.Event(),
    )
    scheduler._state.pending_bucket(Priority.GENERATION).append(pending)

    # Coordinator handle cache (needed for Phase 5 resize RPCs)
    scheduler._coordinator_handle_cache[pipeline_id] = ("test_ns", _FakeCoordinator())

    asyncio.run(scheduler.scheduling_cycle())

    # Phase 3 produced allocation ops, Phase 6 committed them
    assert infer_cluster_id in scheduler._state.active_allocations
    gen_alloc = scheduler._state.active_allocations[infer_cluster_id]
    assert set(gen_alloc.gpu_ids) == {2, 3}
    assert gen_alloc.priority == Priority.GENERATION
    # Phase 6 signaled the GENERATION waiter
    assert pending.event.is_set()
    # All GPUs now allocated
    assert scheduler._state.idle_gpus == set()
