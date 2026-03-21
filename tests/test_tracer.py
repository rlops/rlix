"""Smoke test for SchedulerTracer composition boundary.

Exercises the most order-sensitive tracer path (enqueue -> close -> shutdown)
through the SchedulerTracer with real tg4perfetto writing to a tmpdir.
"""
from __future__ import annotations

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


def test_tracer_queue_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Queue enqueue -> close -> shutdown through SchedulerTracer with real tg4perfetto."""
    pytest.importorskip("tg4perfetto")
    _install_import_stubs(monkeypatch)
    tracer_module = importlib.import_module("rlix.scheduler.tracer")
    protocol_types = importlib.import_module("rlix.protocol.types")

    SchedulerTracer = tracer_module.SchedulerTracer
    Priority = protocol_types.Priority

    tracer = SchedulerTracer()
    tracer.init_tracing(enable=True, trace_output_dir=str(tmp_path))
    assert tracer.enabled

    # Init queue tracks (required before enqueue)
    tracer.init_queue_tracks()

    cluster_id = "ft_000000000000_actor_train"

    # Enqueue
    tracer.trace_queue_enqueue(cluster_id, Priority.ACTOR_TRAINING, bucket_depth=1)
    assert cluster_id in tracer._pending_queue_trace_state

    # Close
    tracer.trace_queue_slice_close(cluster_id)
    assert cluster_id not in tracer._pending_queue_trace_state

    # Shutdown
    tracer.shutdown_tracing()
    assert not tracer.enabled
