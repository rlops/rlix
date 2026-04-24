"""Tests for rlix.pipeline.nemo_rl_config_bridge.register_nemo_rl_pipeline.

Uses an in-process FakeOrchestrator to simulate the three ``.method.remote``
calls on a real Ray actor handle. ``ray.get`` is stubbed as the identity
function, so actor-method returns pass through unchanged.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")
    ray_stub.get = lambda ref: ref  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.pipeline": RLIX_ROOT / "pipeline",
        "rlix.protocol": RLIX_ROOT / "protocol",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_bridge(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    return importlib.import_module("rlix.pipeline.nemo_rl_config_bridge")


# --------------------------------------------------------------------------
# Fake NeMo RL config factory (trimmed copy of the builder-test helper; kept
# self-contained so this file can run in isolation).
# --------------------------------------------------------------------------

_SENTINEL = object()


def _make_nemo_config(
    *,
    vllm_tp: object = 2,
    meg_tp: object = 1,
    meg_pp: object = 1,
    meg_cp: object = 1,
    meg_ep: object = 1,
    async_grpo: object = True,
    peft_enabled: object = _SENTINEL,
    rlix_train_device_mapping: object = _SENTINEL,
    rlix_infer_device_mapping: object = _SENTINEL,
) -> SimpleNamespace:
    megatron_cfg = SimpleNamespace(
        tensor_model_parallel_size=meg_tp,
        pipeline_model_parallel_size=meg_pp,
        context_parallel_size=meg_cp,
        expert_model_parallel_size=meg_ep,
    )
    if peft_enabled is not _SENTINEL:
        megatron_cfg.peft = SimpleNamespace(enabled=peft_enabled)
    vllm_cfg = SimpleNamespace(tensor_parallel_size=vllm_tp)
    generation = SimpleNamespace(vllm_cfg=vllm_cfg)
    policy = SimpleNamespace(generation=generation, megatron_cfg=megatron_cfg)
    grpo = SimpleNamespace(async_grpo=SimpleNamespace(enabled=async_grpo))
    cfg = SimpleNamespace(policy=policy, grpo=grpo)
    rlix_fields: dict = {}
    if rlix_train_device_mapping is not _SENTINEL:
        rlix_fields["train_device_mapping"] = rlix_train_device_mapping
    if rlix_infer_device_mapping is not _SENTINEL:
        rlix_fields["infer_device_mapping"] = rlix_infer_device_mapping
    if rlix_fields:
        cfg.rlix = SimpleNamespace(**rlix_fields)
    return cfg


# --------------------------------------------------------------------------
# Ray-actor fakes: each ``method.remote(*a, **kw)`` just invokes a Python
# callable and returns its value. Combined with ``ray.get = identity`` in
# the test stub, this exercises the helper's control flow without Ray.
# --------------------------------------------------------------------------


class _FakeActorMethod:
    def __init__(self, impl):
        self._impl = impl

    def remote(self, *args, **kwargs):
        return self._impl(*args, **kwargs)


class _FakeAdmitResponse:
    def __init__(self, *, pipeline_id: str, scheduler: object) -> None:
        self.pipeline_id = pipeline_id
        self.scheduler = scheduler


class _FakeRegisterResponse:
    def __init__(self, *, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id


class FakeOrchestrator:
    def __init__(
        self,
        *,
        pipeline_id: str = "ft_abc123def456",
        scheduler: object = None,
        admit_scheduler_none: bool = False,
        raise_on: dict | None = None,
    ) -> None:
        self._allocated_pipeline_id = pipeline_id
        self._scheduler = object() if scheduler is None else scheduler
        self._admit_scheduler_none = admit_scheduler_none
        self._raise_on = raise_on or {}
        self.calls: list[tuple[str, dict]] = []
        self.allocate_pipeline_id = _FakeActorMethod(self._allocate)
        self.register_pipeline = _FakeActorMethod(self._register)
        self.admit_pipeline = _FakeActorMethod(self._admit)

    def _maybe_raise(self, op: str) -> None:
        if op in self._raise_on:
            raise self._raise_on[op]

    def _allocate(self, pipeline_type):
        self.calls.append(("allocate", {"pipeline_type": pipeline_type}))
        self._maybe_raise("allocate")
        return self._allocated_pipeline_id

    def _register(
        self,
        *,
        pipeline_id,
        ray_namespace,
        cluster_tp_configs,
        cluster_device_mappings,
    ):
        self.calls.append(
            (
                "register",
                {
                    "pipeline_id": pipeline_id,
                    "ray_namespace": ray_namespace,
                    "cluster_tp_configs": cluster_tp_configs,
                    "cluster_device_mappings": cluster_device_mappings,
                },
            )
        )
        self._maybe_raise("register")
        return _FakeRegisterResponse(pipeline_id=pipeline_id)

    def _admit(self, *, pipeline_id):
        self.calls.append(("admit", {"pipeline_id": pipeline_id}))
        self._maybe_raise("admit")
        scheduler = None if self._admit_scheduler_none else self._scheduler
        return _FakeAdmitResponse(pipeline_id=pipeline_id, scheduler=scheduler)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestRegisterNemoRlPipeline:
    def test_returns_allocated_id_and_namespace_and_scheduler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        sched_handle = object()
        orch = FakeOrchestrator(
            pipeline_id="ft_abc123def456", scheduler=sched_handle
        )
        result = bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(vllm_tp=2),
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        assert isinstance(result, bridge.NemoRlRegistrationResult)
        assert result.pipeline_id == "ft_abc123def456"
        assert result.ray_namespace == "pipeline_ft_abc123def456_NS"
        assert result.scheduler is sched_handle

    def test_calls_three_orchestrator_methods_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator()
        bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(vllm_tp=2),
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        assert [op for op, _ in orch.calls] == ["allocate", "register", "admit"]

    def test_register_kwargs_match_nemo_config_and_device_mappings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(pipeline_id="ft_xxx111222333")
        bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(vllm_tp=4),
            train_device_mapping=[0, 1, 2, 3],
            infer_device_mapping=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        _, register_kwargs = orch.calls[1]
        assert register_kwargs["pipeline_id"] == "ft_xxx111222333"
        assert register_kwargs["ray_namespace"] == "pipeline_ft_xxx111222333_NS"
        assert register_kwargs["cluster_tp_configs"] == {
            "actor_train": 1,
            "actor_infer": 4,
        }
        assert register_kwargs["cluster_device_mappings"] == {
            "actor_train": [0, 1, 2, 3],
            "actor_infer": [0, 1, 2, 3, 4, 5, 6, 7],
        }

    def test_admit_receives_allocated_pipeline_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(pipeline_id="ft_abc123def456")
        bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(vllm_tp=2),
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        _, admit_kwargs = orch.calls[2]
        assert admit_kwargs == {"pipeline_id": "ft_abc123def456"}

    def test_lora_config_passes_lora_to_allocate_pipeline_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(pipeline_id="lora_aaa000bbb111")
        bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(vllm_tp=2, peft_enabled=True),
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        assert orch.calls[0] == ("allocate", {"pipeline_type": "lora"})

    def test_ft_config_passes_ft_to_allocate_pipeline_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(pipeline_id="ft_abc123def456")
        bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(vllm_tp=2, peft_enabled=False),
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        assert orch.calls[0] == ("allocate", {"pipeline_type": "ft"})

    def test_allocate_raises_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(raise_on={"allocate": RuntimeError("alloc-boom")})
        with pytest.raises(RuntimeError, match="alloc-boom"):
            bridge.register_nemo_rl_pipeline(
                orchestrator=orch,
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[0, 1],
                infer_device_mapping=[0, 1, 2, 3],
            )
        assert [op for op, _ in orch.calls] == ["allocate"]

    def test_register_raises_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(raise_on={"register": RuntimeError("reg-boom")})
        with pytest.raises(RuntimeError, match="reg-boom"):
            bridge.register_nemo_rl_pipeline(
                orchestrator=orch,
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[0, 1],
                infer_device_mapping=[0, 1, 2, 3],
            )
        assert [op for op, _ in orch.calls] == ["allocate", "register"]

    def test_admit_raises_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(raise_on={"admit": RuntimeError("admit-boom")})
        with pytest.raises(RuntimeError, match="admit-boom"):
            bridge.register_nemo_rl_pipeline(
                orchestrator=orch,
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[0, 1],
                infer_device_mapping=[0, 1, 2, 3],
            )
        assert [op for op, _ in orch.calls] == ["allocate", "register", "admit"]

    def test_admit_returns_none_scheduler_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(
            pipeline_id="ft_abc123def456", admit_scheduler_none=True
        )
        with pytest.raises(
            RuntimeError,
            match=r"scheduler=None for pipeline_id='ft_abc123def456'",
        ):
            bridge.register_nemo_rl_pipeline(
                orchestrator=orch,
                nemo_config=_make_nemo_config(vllm_tp=2),
                train_device_mapping=[0, 1],
                infer_device_mapping=[0, 1, 2, 3],
            )

    def test_register_with_kwargs_direct_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(pipeline_id="ft_kwargs000000")
        bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(
                vllm_tp=2,
                rlix_train_device_mapping=[99, 99],
                rlix_infer_device_mapping=[99, 99, 99, 99],
            ),
            train_device_mapping=[0, 1],
            infer_device_mapping=[0, 1, 2, 3],
        )
        _, register_kwargs = orch.calls[1]
        assert register_kwargs["cluster_device_mappings"] == {
            "actor_train": [0, 1],
            "actor_infer": [0, 1, 2, 3],
        }

    def test_register_with_config_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bridge = _load_bridge(monkeypatch)
        orch = FakeOrchestrator(pipeline_id="ft_fallback00000")
        bridge.register_nemo_rl_pipeline(
            orchestrator=orch,
            nemo_config=_make_nemo_config(
                vllm_tp=2,
                rlix_train_device_mapping=[0, 1],
                rlix_infer_device_mapping=[0, 1, 2, 3],
            ),
        )
        _, register_kwargs = orch.calls[1]
        assert register_kwargs["cluster_device_mappings"] == {
            "actor_train": [0, 1],
            "actor_infer": [0, 1, 2, 3],
        }
