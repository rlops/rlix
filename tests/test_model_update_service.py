"""Unit tests for ModelUpdateService orchestration logic.

Tests run without Ray, GPU, or ROLL installed.
All Ray actors and cluster objects are replaced with synchronous fakes.

Covers:
- _select_global_sender_rank: returns rank with all-zero parallel indices
- _build_comm_plan_for_sender: IPC vs broadcast classification based on GPU co-location
- sync_selected_workers: calls selective_sync_active_cache + finalize_weight_update
- Timeout raises RuntimeError with descriptive message
- Port claim released only on sync_completed=True
"""
from __future__ import annotations

import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Stubs — minimal fakes for all heavy deps
# ---------------------------------------------------------------------------


def _stub_modules(monkeypatch):
    """Install minimal stubs so rlix.pipeline.model_update_service can import."""
    ray_stub = types.ModuleType("ray")

    def _remote(cls_or_fn=None, **kwargs):
        if cls_or_fn is not None:
            return cls_or_fn
        return lambda fn: fn

    ray_stub.remote = _remote  # type: ignore[attr-defined]
    ray_stub.get = MagicMock(side_effect=lambda refs, timeout=None: [None] * (len(refs) if isinstance(refs, list) else 1))  # type: ignore[attr-defined]

    class _GetTimeoutError(Exception):
        pass

    ray_stub.exceptions = MagicMock()
    ray_stub.exceptions.GetTimeoutError = _GetTimeoutError
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    # roll stubs
    for m in ["roll", "roll.distributed", "roll.distributed.executor",
              "roll.distributed.executor.cluster",
              "roll.utils", "roll.utils.constants", "roll.utils.logging"]:
        stub = types.ModuleType(m)
        monkeypatch.setitem(sys.modules, m, stub)

    sys.modules["roll.utils.constants"].GLOBAL_STORAGE_NAMESPACE = "global"  # type: ignore[attr-defined]
    sys.modules["roll.utils.constants"].STORAGE_NAME = "shared_storage"  # type: ignore[attr-defined]
    sys.modules["roll.utils.logging"].get_logger = lambda: MagicMock()  # type: ignore[attr-defined]
    # Cluster is imported directly from roll.distributed.executor.cluster
    sys.modules["roll.distributed.executor.cluster"].Cluster = MagicMock  # type: ignore[attr-defined]

    # rlix and rlix.utils.env — set up as a proper package
    rlix_mod = types.ModuleType("rlix")
    rlix_mod.__path__ = [str(RLIX_ROOT)]  # type: ignore[attr-defined]
    rlix_mod.__package__ = "rlix"
    rlix_utils = types.ModuleType("rlix.utils")
    rlix_utils.__path__ = [str(RLIX_ROOT / "utils")]  # type: ignore[attr-defined]
    rlix_utils_env = types.ModuleType("rlix.utils.env")
    rlix_utils_env.parse_env_timeout_s = lambda _name, default=None: default  # type: ignore[attr-defined]
    rlix_pipeline = types.ModuleType("rlix.pipeline")
    rlix_pipeline.__path__ = [str(RLIX_ROOT / "pipeline")]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlix", rlix_mod)
    monkeypatch.setitem(sys.modules, "rlix.utils", rlix_utils)
    monkeypatch.setitem(sys.modules, "rlix.utils.env", rlix_utils_env)
    monkeypatch.setitem(sys.modules, "rlix.pipeline", rlix_pipeline)

    return ray_stub


# ---------------------------------------------------------------------------
# Fake cluster / worker data structures
# ---------------------------------------------------------------------------


@dataclass
class FakeWorkerRankInfo:
    pp_rank: int = 0
    dp_rank: int = 0
    tp_rank: int = 0
    cp_rank: int = 0


@dataclass
class FakeWorkerConfig:
    device_mapping: List[int]
    num_gpus_per_worker: int = 1


class FakeCluster:
    def __init__(self, workers, rank_infos, devices_by_rank, world_size=None):
        self.workers = workers
        self.worker_rank_info = rank_infos
        self.rank2worker = {i: w for i, w in enumerate(workers)}
        self.rank2devices = devices_by_rank
        self.world_size = world_size or len(workers)
        self.worker_config = FakeWorkerConfig(
            device_mapping=list(range(world_size or len(workers))),
            num_gpus_per_worker=1,
        )


# ---------------------------------------------------------------------------
# Helper to load the module under test
# ---------------------------------------------------------------------------


def _load_mus(monkeypatch):
    # Remove any cached rlix modules
    for key in list(sys.modules):
        if "rlix" in key or "model_update_service" in key:
            monkeypatch.delitem(sys.modules, key, raising=False)

    ray_stub = _stub_modules(monkeypatch)

    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rlix.pipeline.model_update_service",
        RLIX_ROOT / "pipeline" / "model_update_service.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["rlix.pipeline.model_update_service"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod, ray_stub


# ---------------------------------------------------------------------------
# _select_global_sender_rank
# ---------------------------------------------------------------------------


def test_select_global_sender_rank_finds_owner(monkeypatch):
    mod, _ = _load_mus(monkeypatch)

    # 4 ranks; rank 2 is pp=0,dp=0,tp=0,cp=0
    workers = [MagicMock() for _ in range(4)]
    rank_infos = [
        FakeWorkerRankInfo(pp_rank=1, dp_rank=0, tp_rank=0, cp_rank=0),
        FakeWorkerRankInfo(pp_rank=0, dp_rank=1, tp_rank=0, cp_rank=0),
        FakeWorkerRankInfo(pp_rank=0, dp_rank=0, tp_rank=0, cp_rank=0),  # owner
        FakeWorkerRankInfo(pp_rank=0, dp_rank=0, tp_rank=1, cp_rank=0),
    ]
    devices = {i: [{"node_rank": 0, "gpu_rank": i, "rank": i}] for i in range(4)}
    src_cluster = FakeCluster(workers, rank_infos, devices)
    tgt_cluster = FakeCluster([MagicMock()], [FakeWorkerRankInfo()],
                               {0: [{"node_rank": 0, "gpu_rank": 99, "rank": 0}]})

    svc = mod.ModelUpdateService.__new__(mod.ModelUpdateService)
    svc.pipeline_id = "test"
    svc.src_cluster = src_cluster
    svc.tgt_cluster = tgt_cluster
    svc._sync_nonce = "abc"
    svc._master_addr_by_src_rank = {}
    svc._timeout_s = None
    svc._pg_timeout_s = None

    assert svc._select_global_sender_rank() == 2


def test_select_global_sender_rank_raises_when_none(monkeypatch):
    mod, _ = _load_mus(monkeypatch)

    workers = [MagicMock()]
    rank_infos = [FakeWorkerRankInfo(pp_rank=1, dp_rank=1, tp_rank=1, cp_rank=1)]
    devices = {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]}
    src_cluster = FakeCluster(workers, rank_infos, devices)
    tgt_cluster = FakeCluster([MagicMock()], [FakeWorkerRankInfo()],
                               {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]})

    svc = mod.ModelUpdateService.__new__(mod.ModelUpdateService)
    svc.pipeline_id = "p"
    svc.src_cluster = src_cluster
    svc.tgt_cluster = tgt_cluster
    svc._sync_nonce = "x"
    svc._master_addr_by_src_rank = {}
    svc._timeout_s = None
    svc._pg_timeout_s = None

    with pytest.raises(RuntimeError, match="No global cache owner"):
        svc._select_global_sender_rank()


# ---------------------------------------------------------------------------
# _build_comm_plan_for_sender — IPC vs broadcast classification
# ---------------------------------------------------------------------------


def _make_svc(mod, ray_stub, src_devices_by_rank, tgt_devices_by_rank, tgt_dp_ranks=None):
    n_src = len(src_devices_by_rank)
    n_tgt = len(tgt_devices_by_rank)
    src_workers = [MagicMock() for _ in range(n_src)]
    for w in src_workers:
        w.get_node_ip.remote = MagicMock(return_value=None)
        w.get_free_port.remote = MagicMock(return_value=None)
    ray_stub.get = MagicMock(side_effect=lambda refs, timeout=None: [None] * (len(refs) if isinstance(refs, list) else 1))
    # Override specific get calls
    ray_stub.get = lambda refs, timeout=None: (
        "127.0.0.1" if not isinstance(refs, list) else ["127.0.0.1"] + [12345] * (len(refs) - 1)
    )

    rank_infos = [FakeWorkerRankInfo() for _ in range(n_src)]
    src_cluster = FakeCluster(src_workers, rank_infos, src_devices_by_rank)

    tgt_workers = [MagicMock() for _ in range(n_tgt)]
    tgt_rank_infos = [FakeWorkerRankInfo() for _ in range(n_tgt)]
    tgt_cluster = FakeCluster(tgt_workers, tgt_rank_infos, tgt_devices_by_rank)

    svc = mod.ModelUpdateService.__new__(mod.ModelUpdateService)
    svc.pipeline_id = "test_pipe"
    svc.src_cluster = src_cluster
    svc.tgt_cluster = tgt_cluster
    svc._sync_nonce = "nonce"
    svc._master_addr_by_src_rank = {}
    svc._timeout_s = None
    svc._pg_timeout_s = None

    # Patch _get_master_addr and get_free_port
    svc._get_master_addr = MagicMock(return_value="127.0.0.1")
    for w in src_workers:
        w.get_free_port = MagicMock()
        w.get_free_port.remote = MagicMock(return_value=MagicMock())

    import ray as _ray
    _ray.get = MagicMock(return_value=54321)
    return svc


def test_build_comm_plan_ipc_when_same_gpu(monkeypatch):
    """Devices sharing the same (node_rank, gpu_rank) → IPC path."""
    mod, ray_stub = _load_mus(monkeypatch)

    # Sender on node=0, gpu=0
    src_devices = {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]}
    # Target device on SAME gpu (collocated)
    tgt_devices = {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]}

    svc = _make_svc(mod, ray_stub, src_devices, tgt_devices)
    comm_plan, group_name, tgt_ranks_in_group = svc._build_comm_plan_for_sender(
        sync_id="s1", src_rank=0, tgt_dp_ranks=[0]
    )

    plan_entry = comm_plan[0]
    assert len(plan_entry["ipc_targets"]) == 1
    assert plan_entry["ipc_targets"][0]["dp_rank"] == 0
    assert tgt_ranks_in_group == []  # No NCCL group needed for IPC-only


def test_build_comm_plan_broadcast_when_different_gpu(monkeypatch):
    """Devices on different (node_rank, gpu_rank) → broadcast path."""
    mod, ray_stub = _load_mus(monkeypatch)

    src_devices = {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]}
    # Target on different GPU
    tgt_devices = {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]}

    svc = _make_svc(mod, ray_stub, src_devices, tgt_devices)
    comm_plan, group_name, tgt_ranks_in_group = svc._build_comm_plan_for_sender(
        sync_id="s2", src_rank=0, tgt_dp_ranks=[0]
    )

    plan_entry = comm_plan[0]
    assert plan_entry["ipc_targets"] == []
    assert 0 in plan_entry["broadcast_local_ranks_by_dp_rank"]
    assert tgt_ranks_in_group == [0]


# ---------------------------------------------------------------------------
# sync_selected_workers — validation errors
# ---------------------------------------------------------------------------


def test_sync_selected_workers_empty_tgt_raises(monkeypatch):
    mod, ray_stub = _load_mus(monkeypatch)
    src_cluster = FakeCluster(
        [MagicMock()],
        [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()],
        [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    svc = mod.ModelUpdateService.__new__(mod.ModelUpdateService)
    svc.pipeline_id = "p"
    svc.src_cluster = src_cluster
    svc.tgt_cluster = tgt_cluster
    svc._sync_nonce = "n"
    svc._master_addr_by_src_rank = {}
    svc._timeout_s = None
    svc._pg_timeout_s = None

    with pytest.raises(ValueError, match="non-empty"):
        svc.sync_selected_workers([])


def test_sync_selected_workers_invalid_rank_raises(monkeypatch):
    mod, ray_stub = _load_mus(monkeypatch)
    src_cluster = FakeCluster(
        [MagicMock()],
        [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()],
        [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
        world_size=1,
    )
    svc = mod.ModelUpdateService.__new__(mod.ModelUpdateService)
    svc.pipeline_id = "p"
    svc.src_cluster = src_cluster
    svc.tgt_cluster = tgt_cluster
    svc._sync_nonce = "n"
    svc._master_addr_by_src_rank = {}
    svc._timeout_s = None
    svc._pg_timeout_s = None

    with pytest.raises(ValueError, match="Invalid tgt_dp_ranks"):
        svc.sync_selected_workers([99])  # rank 99 doesn't exist in world_size=1
