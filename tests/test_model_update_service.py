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


# ---------------------------------------------------------------------------
# sync_selected_workers — finalize_weight_update is NOT called (pipeline-owned)
# ---------------------------------------------------------------------------


def test_sync_selected_workers_does_not_call_finalize_weight_update(monkeypatch):
    """ModelUpdateService must NOT call finalize_weight_update — ownership belongs
    to the pipeline (spec: nemorl-port-plan.md line 624-632).
    The pipeline calls finalize_weight_update.remote() after sync_selected_workers returns."""
    mod, ray_stub = _load_mus(monkeypatch)

    finalize_called_ranks = []

    class FakeWorkerTrackFinalize(MagicMock):
        def __init__(self, dp_rank, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dp_rank = dp_rank
            self.finalize_weight_update = MagicMock()
            self.finalize_weight_update.remote = MagicMock(
                side_effect=lambda: finalize_called_ranks.append(self._dp_rank)
            )
            self.selective_sync_active_cache = MagicMock()
            self.selective_sync_active_cache.remote = MagicMock(return_value=MagicMock())
            self.setup_collective_group = MagicMock()
            self.setup_collective_group.remote = MagicMock(return_value=MagicMock())
            self.get_node_ip = MagicMock()
            self.get_node_ip.remote = MagicMock(return_value=MagicMock())
            self.get_free_port = MagicMock()
            self.get_free_port.remote = MagicMock(return_value=MagicMock())

    src_worker = FakeWorkerTrackFinalize(dp_rank=0)
    src_worker.selective_sync_active_cache.remote.return_value = MagicMock()
    tgt_worker0 = FakeWorkerTrackFinalize(dp_rank=0)
    tgt_worker1 = FakeWorkerTrackFinalize(dp_rank=1)

    src_rank_info = FakeWorkerRankInfo(pp_rank=0, dp_rank=0, tp_rank=0, cp_rank=0)
    src_cluster = FakeCluster(
        [src_worker],
        [src_rank_info],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [tgt_worker0, tgt_worker1],
        [FakeWorkerRankInfo(), FakeWorkerRankInfo()],
        {
            0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}],
            1: [{"node_rank": 0, "gpu_rank": 2, "rank": 1}],
        },
        world_size=2,
    )

    svc = mod.ModelUpdateService.__new__(mod.ModelUpdateService)
    svc.pipeline_id = "test_no_finalize"
    svc.src_cluster = src_cluster
    svc.tgt_cluster = tgt_cluster
    svc._sync_nonce = "nfin"
    svc._master_addr_by_src_rank = {}
    svc._timeout_s = None
    svc._pg_timeout_s = None
    svc.model_update_transport = "cpu_serialize"
    svc.bucket_size_bytes = None
    svc._get_master_addr = MagicMock(return_value="127.0.0.1")
    svc._build_comm_plan_for_sender = MagicMock(
        return_value=(
            {0: {"master_addr": "127.0.0.1", "master_port": 12345, "ipc_targets": [], "broadcast_tgt_local_ranks": []}},
            "group_nfin",
            [],
        )
    )
    svc._release_master_port_claim = MagicMock()

    import ray as _ray
    _ray.get = MagicMock(return_value=[None])

    svc.sync_selected_workers([0, 1], verify=False)

    # ModelUpdateService must NOT call finalize_weight_update — that is the pipeline's job.
    assert finalize_called_ranks == [], (
        f"ModelUpdateService incorrectly called finalize_weight_update on ranks "
        f"{finalize_called_ranks} — this must be done by the pipeline (spec line 624)"
    )


def test_sync_selected_workers_calls_receiver_destroy_collective_group(monkeypatch):
    """destroy_collective_group must be called on each broadcast-path target worker
    after sync completes (spec: nemorl-port-plan.md lines 380, 385)."""
    mod, ray_stub = _load_mus(monkeypatch)

    destroy_called_ranks: list = []

    class FakeWorkerWithDestroy(MagicMock):
        def __init__(self, dp_rank, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dp_rank = dp_rank
            self.finalize_weight_update = MagicMock()
            self.finalize_weight_update.remote = MagicMock(return_value=MagicMock())
            self.selective_sync_active_cache = MagicMock()
            self.selective_sync_active_cache.remote = MagicMock(return_value=MagicMock())
            self.setup_collective_group = MagicMock()
            self.setup_collective_group.remote = MagicMock(return_value=MagicMock())
            self.destroy_collective_group = MagicMock()
            self.destroy_collective_group.remote = MagicMock(
                side_effect=lambda gn: destroy_called_ranks.append(self._dp_rank)
            )
            self.get_node_ip = MagicMock()
            self.get_node_ip.remote = MagicMock(return_value=MagicMock())
            self.get_free_port = MagicMock()
            self.get_free_port.remote = MagicMock(return_value=MagicMock())

    src_worker = FakeWorkerWithDestroy(dp_rank=0)
    src_worker.selective_sync_active_cache.remote.return_value = MagicMock()
    tgt_worker0 = FakeWorkerWithDestroy(dp_rank=0)
    tgt_worker1 = FakeWorkerWithDestroy(dp_rank=1)

    src_rank_info = FakeWorkerRankInfo(pp_rank=0, dp_rank=0, tp_rank=0, cp_rank=0)
    src_cluster = FakeCluster(
        [src_worker],
        [src_rank_info],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [tgt_worker0, tgt_worker1],
        [FakeWorkerRankInfo(), FakeWorkerRankInfo()],
        {
            0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}],  # different GPU → broadcast
            1: [{"node_rank": 0, "gpu_rank": 2, "rank": 1}],  # different GPU → broadcast
        },
        world_size=2,
    )

    svc = mod.ModelUpdateService.__new__(mod.ModelUpdateService)
    svc.pipeline_id = "test_rcv_destroy"
    svc.src_cluster = src_cluster
    svc.tgt_cluster = tgt_cluster
    svc._sync_nonce = "rcv"
    svc._master_addr_by_src_rank = {}
    svc._timeout_s = None
    svc._pg_timeout_s = None
    svc.model_update_transport = "cpu_serialize"
    svc.bucket_size_bytes = None
    svc._get_master_addr = MagicMock(return_value="127.0.0.1")
    # Both target ranks are broadcast-path (tgt_ranks_in_group = [0, 1])
    svc._build_comm_plan_for_sender = MagicMock(
        return_value=(
            {0: {"master_addr": "127.0.0.1", "master_port": 12346, "ipc_targets": [], "broadcast_tgt_local_ranks": []}},
            "group_rcv_test",
            [0, 1],  # broadcast-path ranks → setup AND destroy must be called
        )
    )
    svc._release_master_port_claim = MagicMock()

    import ray as _ray
    _ray.get = MagicMock(return_value=[None])

    svc.sync_selected_workers([0, 1], verify=False)

    assert sorted(destroy_called_ranks) == [0, 1], (
        f"Expected destroy_collective_group on receiver ranks [0, 1], got {destroy_called_ranks}"
    )


# ---------------------------------------------------------------------------
# model_update_transport — validation and wiring
# ---------------------------------------------------------------------------


def test_model_update_transport_invalid_value_raises(monkeypatch):
    """Invalid transport name must raise ValueError at construction time."""
    mod, _ = _load_mus(monkeypatch)

    src_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    with pytest.raises(ValueError, match="model_update_transport"):
        mod.ModelUpdateService(
            pipeline_id="p",
            src_cluster=src_cluster,
            tgt_cluster=tgt_cluster,
            model_update_transport="nccl_only",  # not a valid value
        )


def test_model_update_transport_defaults_to_cpu_serialize(monkeypatch):
    """Default transport must be 'cpu_serialize'."""
    mod, _ = _load_mus(monkeypatch)

    src_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    svc = mod.ModelUpdateService(
        pipeline_id="p",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
    )
    assert svc.model_update_transport == "cpu_serialize"


def test_model_update_transport_cuda_ipc_accepted(monkeypatch):
    """'cuda_ipc' is a valid transport value."""
    mod, _ = _load_mus(monkeypatch)

    src_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    svc = mod.ModelUpdateService(
        pipeline_id="p",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
        model_update_transport="cuda_ipc",
    )
    assert svc.model_update_transport == "cuda_ipc"


# ---------------------------------------------------------------------------
# bucket_size_bytes — validation and RAM guard
# ---------------------------------------------------------------------------


def test_bucket_size_bytes_none_skips_guard(monkeypatch):
    """bucket_size_bytes=None must not raise even without psutil."""
    mod, _ = _load_mus(monkeypatch)

    src_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    # Should not raise regardless of psutil availability
    svc = mod.ModelUpdateService(
        pipeline_id="p",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
        bucket_size_bytes=None,
    )
    assert svc.bucket_size_bytes is None


def test_bucket_size_bytes_negative_raises(monkeypatch):
    """Negative bucket_size_bytes must raise ValueError."""
    mod, _ = _load_mus(monkeypatch)

    src_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    with pytest.raises(ValueError, match="bucket_size_bytes"):
        mod.ModelUpdateService(
            pipeline_id="p",
            src_cluster=src_cluster,
            tgt_cluster=tgt_cluster,
            bucket_size_bytes=-1,
        )


def test_bucket_size_bytes_ram_guard_not_in_model_update_service(monkeypatch):
    """ModelUpdateService.__init__ must NOT perform the host-RAM guard.
    The guard moved to build_latest_bucket_cache() where the actual total model
    size is known (spec: nemorl-port-plan.md line 337 — check full packed model,
    not per-bucket size)."""
    mod, _ = _load_mus(monkeypatch)

    # Patch psutil to report tiny available RAM — would fail if guard were present
    psutil_stub = types.ModuleType("psutil")

    class _FakeVMem:
        available = 100 * 1024 * 1024  # 100 MB

    psutil_stub.virtual_memory = lambda: _FakeVMem()
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    src_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    # bucket_size_bytes=90 MB on 100 MB available would have triggered the old guard.
    # Now ModelUpdateService must NOT raise — the guard is in build_latest_bucket_cache.
    svc = mod.ModelUpdateService(
        pipeline_id="p",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
        bucket_size_bytes=90 * 1024 * 1024,
    )
    assert svc.bucket_size_bytes == 90 * 1024 * 1024


def test_bucket_size_bytes_ram_guard_passes(monkeypatch):
    """bucket_size_bytes within RAM budget must not raise."""
    mod, _ = _load_mus(monkeypatch)

    psutil_stub = types.ModuleType("psutil")

    class _FakeVMem:
        available = 10 * 1024 * 1024 * 1024  # 10 GB

    psutil_stub.virtual_memory = lambda: _FakeVMem()
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    src_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 0, "rank": 0}]},
    )
    tgt_cluster = FakeCluster(
        [MagicMock()], [FakeWorkerRankInfo()],
        {0: [{"node_rank": 0, "gpu_rank": 1, "rank": 0}]},
    )
    # 2 × 1 GB < 80% × 10 GB (= 8 GB) → should pass
    svc = mod.ModelUpdateService(
        pipeline_id="p",
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
        bucket_size_bytes=1 * 1024 * 1024 * 1024,
    )
    assert svc.bucket_size_bytes == 1 * 1024 * 1024 * 1024
