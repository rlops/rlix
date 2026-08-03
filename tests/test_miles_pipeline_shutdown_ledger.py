"""PR ledger-contract tests (R11-F1 family).

Covers the miles_pipeline ledger-flag contract:
  1. ``shutdown_hard`` keeps a cluster's ledger flag when its release RPC
     fails, so a later ``shutdown_hard`` / ``dispose`` call retries instead of
     silently leaking the server-side allocation (behavioral, ray stubbed).
  2. Init-phase release failures fail fast instead of letting the next
     blocking ``request_gpus`` deadlock (AST asserts, mirroring
     ``test_miles_pipeline_after_training_cleanup.py``).
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RLIX_ROOT = REPO_ROOT / "rlix"


# ---------------------------------------------------------------------------
# Import stubs — mirror tests/test_scheduler_apply_plan_invariants.py so
# rlix.pipeline.miles_pipeline imports without a real ray installation.
# ---------------------------------------------------------------------------


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("rlix"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    ray_stub = types.ModuleType("ray")

    def _remote(*args, **kwargs):
        def _decorate(obj):
            return obj

        return _decorate

    def _get(ref, timeout=None):
        # Fake remote methods return zero-arg callables so tests can inject
        # per-call success / failure at resolve time.
        return ref() if callable(ref) else ref

    ray_stub.remote = _remote
    ray_stub.get = _get
    ray_stub.get_actor = lambda *args, **kwargs: None
    ray_stub.kill = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    package_roots = {
        "rlix": RLIX_ROOT,
        "rlix.protocol": RLIX_ROOT / "protocol",
        "rlix.pipeline": RLIX_ROOT / "pipeline",
        "rlix.utils": RLIX_ROOT / "utils",
    }
    for module_name, module_path in package_roots.items():
        package_module = types.ModuleType(module_name)
        package_module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, package_module)


def _load_miles_pipeline_module(monkeypatch: pytest.MonkeyPatch):
    _install_import_stubs(monkeypatch)
    return importlib.import_module("rlix.pipeline.miles_pipeline")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeReleaseRpc:
    """Stands in for ``scheduler.notify_release_gpus``.

    ``remote`` records the call and returns a resolver the ray stub's ``get``
    invokes; clusters listed in ``fail_cluster_ids`` raise at resolve time.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_cluster_ids: set[str] = set()

    def remote(self, *, cluster_id: str, global_step=None):
        self.calls.append(cluster_id)

        def _resolve():
            if cluster_id in self.fail_cluster_ids:
                raise RuntimeError(f"injected release failure for {cluster_id}")
            return None

        return _resolve


class _FakeScheduler:
    def __init__(self) -> None:
        self.notify_release_gpus = _FakeReleaseRpc()


# ---------------------------------------------------------------------------
# Behavioral: shutdown_hard keeps flags on failure, clears on success
# ---------------------------------------------------------------------------


def test_shutdown_hard_keeps_ledger_flag_on_release_failure_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    miles_pipeline = _load_miles_pipeline_module(monkeypatch)
    pipeline = miles_pipeline.MilesPipeline(
        pipeline_id="miles_test", pipeline_config=types.SimpleNamespace(miles_args=None)
    )
    fake = _FakeScheduler()
    pipeline._scheduler = fake
    pipeline._actor_train_allocated = True
    pipeline._actor_infer_allocated = True
    fake.notify_release_gpus.fail_cluster_ids = {pipeline._actor_train_cluster_id}

    pipeline.shutdown_hard()

    # R11-F1: the failed release keeps its flag for retry; the successful
    # release clears its flag.
    assert pipeline._actor_train_allocated is True
    assert pipeline._actor_infer_allocated is False
    assert fake.notify_release_gpus.calls == [
        pipeline._actor_train_cluster_id,
        pipeline._actor_infer_cluster_id,
    ]

    # Once the scheduler recovers, a second shutdown_hard retries ONLY the
    # leaked cluster and clears the remaining flag.
    fake.notify_release_gpus.fail_cluster_ids = set()
    pipeline.shutdown_hard()

    assert pipeline._actor_train_allocated is False
    assert pipeline._actor_infer_allocated is False
    assert fake.notify_release_gpus.calls[-1] == pipeline._actor_train_cluster_id
    assert fake.notify_release_gpus.calls.count(pipeline._actor_infer_cluster_id) == 1


# ---------------------------------------------------------------------------
# AST: init-phase release failures must fail fast (raise), not fall through
# ---------------------------------------------------------------------------


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _raise_messages(fn: ast.FunctionDef) -> list[str]:
    messages: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Raise):
            continue
        parts = [
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        ]
        messages.append("".join(parts))
    return messages


def test_init_phase_a_raises_when_train_release_fails() -> None:
    source = (REPO_ROOT / "rlix" / "pipeline" / "miles_pipeline.py").read_text(encoding="utf-8")
    fn = _function_def(ast.parse(source), "_init_phase_a_train")
    assert any(
        "failed to release actor_train" in message for message in _raise_messages(fn)
    ), "_init_phase_a_train must raise when the actor_train release fails (fail-fast, not fall through to phase B)"


def test_init_phase_b_raises_when_infer_release_fails() -> None:
    source = (REPO_ROOT / "rlix" / "pipeline" / "miles_pipeline.py").read_text(encoding="utf-8")
    fn = _function_def(ast.parse(source), "_init_phase_b_infer")
    assert any(
        "failed to release actor_infer" in message for message in _raise_messages(fn)
    ), "_init_phase_b_infer must raise when the INIT release fails (fail-fast before the GENERATION re-request)"
