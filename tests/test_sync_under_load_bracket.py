"""AST-level ordering assertions for the rlix#42 sync-under-load bracket
in ``MilesCoordinator.sync_base_weights_to_active``.

The bracket mirrors shrink_engines' proven quiesce ordering so
finalize's /flush_cache cannot race live router dispatch: router
admission closes (unregister_from_router) and in-flight work aborts
(_abort_engines) BEFORE the sync RPC; routing re-activates + the abort
idempotency cache resets in a ``finally``.

Follows the repo's AST-test pattern (cf.
test_miles_model_update_service_cleanup.py) so it runs without ray.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_sync_base_fn() -> ast.FunctionDef:
    source = (REPO_ROOT / "rlix" / "pipeline" / "miles_coordinator.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sync_base_weights_to_active":
            return node
    raise AssertionError("sync_base_weights_to_active not found")


def _calls_attr(node: ast.AST, attr: str) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == attr
        for child in ast.walk(node)
    )


def _first_line_calling(fn: ast.FunctionDef, attr: str) -> int:
    lines = [
        child.lineno
        for child in ast.walk(fn)
        if isinstance(child, ast.Attribute) and child.attr == attr
    ]
    assert lines, f"{attr} not called inside sync_base_weights_to_active"
    return min(lines)


def test_quiesce_happens_before_the_sync_rpc() -> None:
    fn = _load_sync_base_fn()
    unregister_line = _first_line_calling(fn, "unregister_from_router")
    abort_line = _first_line_calling(fn, "_abort_engines")
    sync_line = _first_line_calling(fn, "sync_selected_workers")
    assert unregister_line < abort_line < sync_line, (
        "bracket order must be unregister_from_router -> _abort_engines -> "
        f"sync_selected_workers (got lines {unregister_line}, {abort_line}, "
        f"{sync_line})"
    )


def test_quiesce_is_inside_the_guarded_region() -> None:
    # codex impl-r11 high #1: a failure in unregister/abort must still
    # reach the re-register finally — so BOTH must live in the try body.
    fn = _load_sync_base_fn()
    try_nodes = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Try)
        and any(_calls_attr(stmt, "sync_selected_workers") for stmt in node.body)
    ]
    assert try_nodes, "sync RPC must be wrapped in try/finally"
    outer = try_nodes[0]
    assert any(
        _calls_attr(stmt, "unregister_from_router") for stmt in outer.body
    ), "unregister_from_router must be inside the try body"
    assert any(
        _calls_attr(stmt, "_abort_engines") for stmt in outer.body
    ), "_abort_engines must be inside the try body"


def test_reregister_lives_in_finally_and_is_not_activate_routing() -> None:
    fn = _load_sync_base_fn()
    try_nodes = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Try)
        and any(_calls_attr(stmt, "sync_selected_workers") for stmt in node.body)
    ]
    assert try_nodes, "sync RPC must be wrapped in try/finally"
    outer = try_nodes[0]
    # codex impl-r11 high #2 + impl-r13 TOCTOU: the re-register must be
    # the manager-side ATOMIC register_router_if_active (state check and
    # /add_worker in one serialized manager call) — never activate_routing
    # (loading-only INIT transition) nor a raw per-engine register after a
    # separate state read.
    assert any(
        _calls_attr(stmt, "register_router_if_active") for stmt in outer.finalbody
    ), "register_router_if_active must run from the finally block"
    assert not any(
        _calls_attr(stmt, "activate_routing") for stmt in outer.finalbody
    ), "activate_routing must NOT be used for still-active engines"
    assert not any(
        _calls_attr(stmt, "register_with_router") for stmt in outer.finalbody
    ), "raw per-engine register_with_router in the finally is a TOCTOU (impl-r13)"
    assert any(
        _calls_attr(stmt, "_reset_abort_idempotency_for") for stmt in outer.finalbody
    ), "_reset_abort_idempotency_for must run from the finally block"
