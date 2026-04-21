"""Tests for rlix.pipeline.nemo_rl_virtual_cluster_adapter."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> Any:
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")
    util_stub = types.ModuleType("ray.util")
    sched_stub = types.ModuleType("ray.util.scheduling_strategies")

    class _FakePGSchedulingStrategy:
        def __init__(self, placement_group: Any, placement_group_bundle_index: int) -> None:
            self.placement_group = placement_group
            self.placement_group_bundle_index = placement_group_bundle_index

    sched_stub.PlacementGroupSchedulingStrategy = _FakePGSchedulingStrategy
    util_stub.scheduling_strategies = sched_stub
    ray_stub.util = util_stub

    def _fake_remote(*decorator_args: Any, **decorator_kwargs: Any):
        def _wrap(func: Any) -> Any:
            class _RemoteFn:
                def __init__(self, fn: Any) -> None:
                    self._fn = fn
                    self.decorator_kwargs = decorator_kwargs

                def remote(self, *args: Any, **kwargs: Any) -> Any:
                    return self._fn(*args, **kwargs)

            return _RemoteFn(func)

        return _wrap

    ray_stub.remote = _fake_remote
    ray_stub.get = lambda ref: ref

    monkeypatch.setitem(sys.modules, "ray", ray_stub)
    monkeypatch.setitem(sys.modules, "ray.util", util_stub)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", sched_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.pipeline": RLIX_ROOT / "pipeline",
        "rlix.protocol": RLIX_ROOT / "protocol",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)

    return ray_stub


def _load_adapter_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_import_stubs(monkeypatch)
    return importlib.import_module("rlix.pipeline.nemo_rl_virtual_cluster_adapter")


def _fake_pg(tag: str) -> Any:
    return SimpleNamespace(tag=tag)


def _default_kwargs(pgs: List[Any]) -> dict:
    return {
        "placement_groups": pgs,
        "bundle_ct_per_node_list": [2, 2],
        "num_gpus_per_node": 8,
        "use_gpus": True,
        "max_colocated_worker_groups": 1,
        "name": "test-cluster",
    }


def test_world_size_sums_bundle_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("n0"), _fake_pg("n1")]
    adapter = mod.RLixVirtualClusterAdapter(**_default_kwargs(pgs))
    assert adapter.world_size() == 4


def test_node_count_matches_bundle_list_length(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("a"), _fake_pg("b"), _fake_pg("c")]
    kwargs = _default_kwargs(pgs)
    kwargs["bundle_ct_per_node_list"] = [8, 8, 4]
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)
    assert adapter.node_count() == 3


def test_get_placement_groups_returns_injected_pgs(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("x"), _fake_pg("y")]
    adapter = mod.RLixVirtualClusterAdapter(**_default_kwargs(pgs))
    result = adapter.get_placement_groups()
    assert [p.tag for p in result] == ["x", "y"]


def test_get_placement_groups_returns_fresh_list(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("x"), _fake_pg("y")]
    adapter = mod.RLixVirtualClusterAdapter(**_default_kwargs(pgs))
    first = adapter.get_placement_groups()
    first.clear()
    assert len(adapter.get_placement_groups()) == 2


def test_init_placement_groups_is_idempotent_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("p0"), _fake_pg("p1")]
    adapter = mod.RLixVirtualClusterAdapter(**_default_kwargs(pgs))
    first = adapter._init_placement_groups()
    second = adapter._init_placement_groups(strategy="SPREAD", use_unified_pg=True)
    assert [p.tag for p in first] == ["p0", "p1"]
    assert [p.tag for p in second] == ["p0", "p1"]
    assert adapter.get_placement_groups()[0] is pgs[0]


def test_shutdown_is_noop_and_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("alive")]
    kwargs = _default_kwargs(pgs)
    kwargs["bundle_ct_per_node_list"] = [1]
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)
    assert adapter.shutdown() is True
    assert adapter.get_placement_groups()[0].tag == "alive"


def test_shutdown_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("alive")]
    kwargs = _default_kwargs(pgs)
    kwargs["bundle_ct_per_node_list"] = [1]
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)
    assert adapter.shutdown() is True
    assert adapter.shutdown() is True


def test_public_attributes_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("p")]
    kwargs = _default_kwargs(pgs)
    kwargs["bundle_ct_per_node_list"] = [1]
    kwargs["num_gpus_per_node"] = 4
    kwargs["max_colocated_worker_groups"] = 3
    kwargs["use_gpus"] = False
    kwargs["name"] = "my-cluster"
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)
    assert adapter.num_gpus_per_node == 4
    assert adapter.max_colocated_worker_groups == 3
    assert adapter.use_gpus is False
    assert adapter.name == "my-cluster"


def test_constructor_requires_keyword_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    with pytest.raises(TypeError):
        mod.RLixVirtualClusterAdapter([_fake_pg("x")], [1], 8)  # type: ignore[misc]


def test_get_available_address_and_port_targets_requested_pg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_stub = _install_import_stubs(monkeypatch)
    captured: dict = {}

    def _capture_remote(*decorator_args: Any, **decorator_kwargs: Any):
        def _wrap(func: Any) -> Any:
            class _RemoteFn:
                def __init__(self, fn: Any) -> None:
                    self._fn = fn
                    captured["decorator_kwargs"] = decorator_kwargs

                def remote(self, *args: Any, **kwargs: Any) -> Any:
                    return ("10.0.0.1", 51234)

            return _RemoteFn(func)

        return _wrap

    ray_stub.remote = _capture_remote
    mod = importlib.import_module("rlix.pipeline.nemo_rl_virtual_cluster_adapter")

    pg_a = _fake_pg("a")
    pg_b = _fake_pg("b")
    kwargs = _default_kwargs([pg_a, pg_b])
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)

    addr, port = adapter.get_available_address_and_port(pg_idx=1, bundle_idx=1)
    assert addr == "10.0.0.1"
    assert port == 51234

    strategy = captured["decorator_kwargs"]["scheduling_strategy"]
    assert strategy.placement_group is pg_b
    assert strategy.placement_group_bundle_index == 1


def test_get_master_address_and_port_uses_first_pg_bundle_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_stub = _install_import_stubs(monkeypatch)
    captured: dict = {}

    def _capture_remote(*decorator_args: Any, **decorator_kwargs: Any):
        def _wrap(func: Any) -> Any:
            class _RemoteFn:
                def __init__(self, fn: Any) -> None:
                    self._fn = fn
                    captured["decorator_kwargs"] = decorator_kwargs

                def remote(self, *args: Any, **kwargs: Any) -> Any:
                    return ("master-host", 7777)

            return _RemoteFn(func)

        return _wrap

    ray_stub.remote = _capture_remote
    mod = importlib.import_module("rlix.pipeline.nemo_rl_virtual_cluster_adapter")

    pg_a = _fake_pg("a")
    pg_b = _fake_pg("b")
    kwargs = _default_kwargs([pg_a, pg_b])
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)

    addr, port = adapter.get_master_address_and_port()
    assert (addr, port) == ("master-host", 7777)

    strategy = captured["decorator_kwargs"]["scheduling_strategy"]
    assert strategy.placement_group is pg_a
    assert strategy.placement_group_bundle_index == 0


def test_no_nemo_rl_import(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    for module_name in sys.modules:
        assert not module_name.startswith("nemo_rl"), (
            f"adapter must not import nemo_rl, got {module_name}"
        )
    assert hasattr(mod, "RLixVirtualClusterAdapter")


def test_unimplemented_method_raises_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("p")]
    kwargs = _default_kwargs(pgs)
    kwargs["bundle_ct_per_node_list"] = [1]
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)
    with pytest.raises(AttributeError):
        _ = adapter.some_method_that_does_not_exist  # noqa: B018


def test_bundle_list_is_defensively_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("p"), _fake_pg("q")]
    source = [2, 2]
    kwargs = _default_kwargs(pgs)
    kwargs["bundle_ct_per_node_list"] = source
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)
    source.append(99)
    assert adapter.world_size() == 4
    assert adapter.node_count() == 2


def test_placement_group_list_is_defensively_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_adapter_module(monkeypatch)
    pgs = [_fake_pg("p"), _fake_pg("q")]
    kwargs = _default_kwargs(pgs)
    adapter = mod.RLixVirtualClusterAdapter(**kwargs)
    pgs.append(_fake_pg("rogue"))
    assert len(adapter.get_placement_groups()) == 2
