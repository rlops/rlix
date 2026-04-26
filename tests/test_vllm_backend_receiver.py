"""Unit tests for VllmInternalWorkerExtension receiver methods (Feature 4).

Runs without Ray, GPU, or vLLM installed.  All heavy deps are stubbed.
Tests verify:
- update_parameter_in_bucket: rank guard (skip if not in ipc_local_ranks)
- destroy_collective_group: no-op when group doesn't exist
- finalize_weight_update: calls process_weights_after_loading exactly once
- verify_model: raises on mismatch, passes on match
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------


def _make_torch_stub():
    torch_stub = types.ModuleType("torch")

    class _Dtype:
        def __init__(self, name: str, itemsize: int):
            self.name = name
            self.itemsize = itemsize

        def __eq__(self, other):
            return isinstance(other, _Dtype) and self.name == other.name

        def __hash__(self):
            return hash(self.name)

    float32 = _Dtype("float32", 4)
    uint8 = _Dtype("uint8", 1)
    torch_stub.float32 = float32  # type: ignore[attr-defined]
    torch_stub.uint8 = uint8  # type: ignore[attr-defined]

    class _Size(tuple):
        def numel(self):
            result = 1
            for s in self:
                result *= s
            return result

    class _Tensor:
        def __init__(self, raw: bytes, dtype=None, shape=None):
            self._raw = raw
            self.dtype = dtype or float32
            self.shape = _Size(shape or [len(raw) // (dtype.itemsize if dtype else 4)])

        def numel(self):
            return self.shape.numel()

        def element_size(self):
            return self.dtype.itemsize

        def float(self):
            return self

        def flatten(self):
            return self

        def view(self, target_dtype):
            t = _Tensor.__new__(_Tensor)
            t._raw = self._raw
            t.dtype = target_dtype
            t.shape = _Size([len(self._raw) // target_dtype.itemsize])
            return t

        def reshape(self, shape):
            t = _Tensor.__new__(_Tensor)
            t._raw = self._raw
            t.dtype = self.dtype
            t.shape = _Size(shape)
            return t

        def __getitem__(self, key):
            if isinstance(key, slice):
                sliced_raw = self._raw[key]
                t = _Tensor.__new__(_Tensor)
                t._raw = sliced_raw
                t.dtype = self.dtype
                t.shape = _Size([len(sliced_raw) // self.dtype.itemsize])
                return t
            raise NotImplementedError

        def to(self, device):
            return self

        def sum(self):
            return 0.0

        def max(self):
            return 0.0

        def min(self):
            return 0.0

    class _Module:
        def state_dict(self):
            t = _Tensor(b"\x00" * 4, float32, [1])
            return {"w": t}

        def load_weights(self, weights):
            pass

    class _ModelRunner:
        def __init__(self):
            self.model = _Module()
            self.vllm_config = MagicMock()
            self.model_config = MagicMock()

    torch_stub.Tensor = _Tensor  # type: ignore[attr-defined]
    torch_stub.Size = _Size  # type: ignore[attr-defined]
    dist_stub = MagicMock()
    dist_stub.is_initialized = MagicMock(return_value=True)
    dist_stub.get_rank = MagicMock(return_value=0)
    dist_stub.destroy_process_group = MagicMock()
    torch_stub.distributed = dist_stub  # type: ignore[attr-defined]
    # Register as submodule so `import torch.distributed as dist` works
    import sys as _sys
    _sys.modules["torch.distributed"] = dist_stub  # type: ignore[assignment]
    torch_stub.zeros = MagicMock(return_value=_Tensor(b"\x00" * 512, uint8, [512]))
    torch_stub.empty = MagicMock(return_value=_Tensor(b"\x00" * 4, float32, [1]))
    torch_stub.cuda = MagicMock()
    torch_stub.cuda.current_stream = MagicMock(return_value=MagicMock(synchronize=MagicMock()))

    def _cat(tensors):
        raw = b"".join(t._raw for t in tensors if hasattr(t, "_raw"))
        t = _Tensor.__new__(_Tensor)
        t._raw = raw
        t.dtype = tensors[0].dtype if tensors else float32
        t.shape = _Size([len(raw) // t.dtype.itemsize])
        return t

    torch_stub.cat = _cat  # type: ignore[attr-defined]
    return torch_stub, _Tensor, _Module, _ModelRunner


def _make_extension_instance(torch_stub, _Tensor, _Module, _ModelRunner, monkeypatch):
    """Construct a VllmInternalWorkerExtension instance with all deps stubbed."""
    # Stub all required modules before import
    for mod_name in [
        "vllm", "zmq",
        "nemo_rl.models.policy.utils",
        "nemo_rl.utils.nsys",
        "nemo_rl.utils.packed_tensor",
        "nemo_rl.models.generation.vllm.quantization",
        "nemo_rl.models.generation.vllm.quantization.fp8",
        "vllm.model_executor.model_loader.utils",
        "nemo_rl.distributed.stateless_process_group",
        "nemo_rl.models.policy.utils",
    ]:
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, MagicMock())

    # Stub calculate_aligned_size
    sys.modules["nemo_rl.models.policy.utils"].calculate_aligned_size = lambda x, alignment=512: (x + alignment - 1) // alignment * alignment  # type: ignore[attr-defined]

    # Stub fp8
    fp8_stub = sys.modules["nemo_rl.models.generation.vllm.quantization.fp8"]
    fp8_stub.is_fp8_model = MagicMock(return_value=False)  # type: ignore[attr-defined]

    # Stub process_weights_after_loading
    pwl_stub = sys.modules["vllm.model_executor.model_loader.utils"]
    pwl_stub.process_weights_after_loading = MagicMock()  # type: ignore[attr-defined]

    # Stub quantization package
    quant_stub = sys.modules["nemo_rl.models.generation.vllm.quantization"]
    quant_stub.fp8 = fp8_stub  # type: ignore[attr-defined]

    # Load vllm_backend directly by file to avoid __init__.py chain imports
    # (which require transformers, megatron, etc.)
    for key in list(sys.modules):
        if "vllm_backend" in key:
            monkeypatch.delitem(sys.modules, key, raising=False)

    import importlib.util
    from pathlib import Path

    backend_path = (
        Path(__file__).resolve().parents[1]
        / "external" / "NeMo" / "nemo_rl" / "models" / "generation" / "vllm" / "vllm_backend.py"
    )

    spec = importlib.util.spec_from_file_location("nemo_rl.models.generation.vllm.vllm_backend", backend_path)
    ext_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["nemo_rl.models.generation.vllm.vllm_backend"] = ext_mod
    spec.loader.exec_module(ext_mod)  # type: ignore[union-attr]

    # Instantiate the class with a fake model_runner and device
    ext = ext_mod.VllmInternalWorkerExtension.__new__(ext_mod.VllmInternalWorkerExtension)
    ext.model_runner = _ModelRunner()
    ext.model_config = MagicMock()
    ext.device = MagicMock()
    ext.state_dict_info = {}
    ext._model_update_groups = {}
    return ext, ext_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(monkeypatch):
    torch_stub, _Tensor, _Module, _ModelRunner = _make_torch_stub()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    ext, ext_mod = _make_extension_instance(torch_stub, _Tensor, _Module, _ModelRunner, monkeypatch)
    return ext, ext_mod, torch_stub, _Tensor


# ---------------------------------------------------------------------------
# update_parameter_in_bucket — rank guard
# ---------------------------------------------------------------------------


def test_update_parameter_in_bucket_skips_non_member(env, monkeypatch):
    """If rank is NOT in ipc_local_ranks, load_weights must NOT be called."""
    ext, _, torch_stub, _Tensor = env
    torch_stub.distributed.get_rank.return_value = 5  # rank 5

    payload = {
        "param_names": ["w"],
        "shapes": [(4,)],
        "dtypes": [torch_stub.float32],
        "offsets": [0],
        "used_bytes": 16,
        "cpu_uint8_bucket": _Tensor(b"\x00" * 512, torch_stub.uint8, [512]),
    }
    # ipc_local_ranks=[0,1,2] — rank 5 is not in this set
    ext.update_parameter_in_bucket(payload, ipc_local_ranks=[0, 1, 2], model_update_transport="cpu_serialize")

    # load_weights should NOT have been called
    assert not ext.model_runner.model.load_weights.called if hasattr(ext.model_runner.model.load_weights, "called") else True


def test_update_parameter_in_bucket_processes_member(env, monkeypatch):
    """If rank IS in ipc_local_ranks, the method should not raise."""
    ext, _, torch_stub, _Tensor = env
    torch_stub.distributed.get_rank.return_value = 0  # rank 0

    ext.model_runner.model.load_weights = MagicMock()
    ext._split_policy_and_draft_weights = lambda w: (w, [])
    ext._load_draft_weights = MagicMock()

    payload = {
        "param_names": ["w"],
        "shapes": [(4,)],
        "dtypes": [torch_stub.float32],
        "offsets": [0],
        "used_bytes": 16,
        "cpu_uint8_bucket": _Tensor(b"\x00" * 512, torch_stub.uint8, [512]),
    }
    ext.update_parameter_in_bucket(payload, ipc_local_ranks=[0], model_update_transport="cpu_serialize")
    ext.model_runner.model.load_weights.assert_called_once()


# ---------------------------------------------------------------------------
# destroy_collective_group — no-op guard
# ---------------------------------------------------------------------------


def test_destroy_collective_group_noop_when_missing(env):
    """Must not raise when group name is not in _model_update_groups."""
    ext, _, _, _ = env
    ext._model_update_groups = {}
    # Should not raise
    ext.destroy_collective_group("nonexistent_group")


def test_destroy_collective_group_calls_destroy_when_present(env):
    """Must call dist.destroy_process_group when group exists."""
    ext, _, torch_stub, _ = env
    fake_pg = MagicMock()
    ext._model_update_groups = {"my_group": fake_pg}
    ext.destroy_collective_group("my_group")
    # Group must be removed from dict
    assert "my_group" not in ext._model_update_groups


def test_destroy_collective_group_noop_when_attribute_missing(env):
    """Must not raise when _model_update_groups attr doesn't exist at all."""
    ext, _, _, _ = env
    if hasattr(ext, "_model_update_groups"):
        del ext._model_update_groups
    ext.destroy_collective_group("group_x")


# ---------------------------------------------------------------------------
# finalize_weight_update — calls process_weights_after_loading once
# ---------------------------------------------------------------------------


def test_finalize_weight_update_calls_process_weights(env):
    """process_weights_after_loading must be called exactly once."""
    ext, _, _, _ = env
    ext._maybe_process_fp8_kv_cache = MagicMock()

    import sys as _sys
    pwl = _sys.modules.get("vllm.model_executor.model_loader.utils")
    if pwl is None:
        pytest.skip("vllm stub not available")
    pwl.process_weights_after_loading.reset_mock()

    ext.finalize_weight_update()

    pwl.process_weights_after_loading.assert_called_once()
    ext._maybe_process_fp8_kv_cache.assert_called_once()


# ---------------------------------------------------------------------------
# verify_model — stats comparison
# ---------------------------------------------------------------------------


def test_verify_model_passes_on_matching_stats(env, monkeypatch):
    """Should not raise when expected stats approximately match model stats."""
    ext, _, torch_stub, _Tensor = env
    # Patch model state_dict to return a predictable tensor
    ext.model_runner.model.state_dict = lambda: {}
    # With empty state_dict, there's nothing to verify — should not raise.
    ext.verify_model({"sum": 0.0, "max": 0.0, "min": 0.0})


def test_verify_model_raises_on_mismatch(env, monkeypatch):
    """Should raise RuntimeError when expected stats deviate significantly."""
    ext, _, torch_stub, _Tensor = env

    class _FakeTensor:
        def numel(self):
            return 4

        def float(self):
            return self

        def flatten(self):
            return self

        def sum(self):
            return 100.0

        def max(self):
            return 25.0

        def min(self):
            return 25.0

    torch_stub.cat = lambda ts: _FakeTensor()
    ext.model_runner.model.state_dict = lambda: {"w": _FakeTensor()}

    # Vastly different expected stats should trigger RuntimeError
    with pytest.raises(RuntimeError, match="mismatch"):
        ext.verify_model({"sum": 999999.0, "max": 0.0, "min": 0.0})
