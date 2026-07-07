"""PR ingress-validation test for the scheduler.

A GENERATION request with neither a positive ``step_target_estimate`` nor any
stored progress is skipped by the gap-ratio planner (bootstrap-estimate
branch) and blocks its caller indefinitely. The scheduler now warns at
enqueue time; these tests pin the warn / no-warn matrix.
"""

from __future__ import annotations

import importlib
import logging
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


def _make_scheduler(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    scheduler_module = importlib.import_module("rlix.scheduler.scheduler")
    protocol_types = importlib.import_module("rlix.protocol.types")
    return scheduler_module, protocol_types, scheduler_module.SchedulerImpl()


_PIPELINE_ID = "miles_test"
_CLUSTER_ID = f"{_PIPELINE_ID}_actor_infer"


def test_warns_when_no_estimate_and_no_progress(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, scheduler = _make_scheduler(monkeypatch)
    with caplog.at_level(logging.WARNING):
        scheduler._warn_gen_request_without_demand_locked(
            cluster_id=_CLUSTER_ID, pipeline_id=_PIPELINE_ID, step_target_estimate=None
        )
    assert "may block indefinitely" in caplog.text


def test_warns_on_non_positive_estimate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, scheduler = _make_scheduler(monkeypatch)
    with caplog.at_level(logging.WARNING):
        scheduler._warn_gen_request_without_demand_locked(
            cluster_id=_CLUSTER_ID, pipeline_id=_PIPELINE_ID, step_target_estimate=0
        )
    assert "may block indefinitely" in caplog.text


def test_silent_with_positive_estimate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, scheduler = _make_scheduler(monkeypatch)
    with caplog.at_level(logging.WARNING):
        scheduler._warn_gen_request_without_demand_locked(
            cluster_id=_CLUSTER_ID, pipeline_id=_PIPELINE_ID, step_target_estimate=8
        )
    assert caplog.text == ""


def test_silent_when_progress_already_stored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scheduler_module, protocol_types, scheduler = _make_scheduler(monkeypatch)
    report = protocol_types.ProgressReport(
        pipeline_id=_PIPELINE_ID,
        step_target_trajectories=8,
        metrics={"completed": 2},
    )
    scheduler._state.latest_progress_by_pipeline[_PIPELINE_ID] = {
        "train": {"__full_finetune__": report}
    }
    with caplog.at_level(logging.WARNING):
        scheduler._warn_gen_request_without_demand_locked(
            cluster_id=_CLUSTER_ID, pipeline_id=_PIPELINE_ID, step_target_estimate=None
        )
    assert caplog.text == ""
