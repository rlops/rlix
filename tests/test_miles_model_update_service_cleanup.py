from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _awaits_ray_get_finalize_refs(node: ast.AST) -> bool:
    if not isinstance(node, ast.Await):
        return False
    call = node.value
    return (
        isinstance(call, ast.Call)
        and _is_name(call.func, "_ray_get")
        and len(call.args) == 1
        and _is_name(call.args[0], "finalize_refs")
    )


def test_finalize_weight_update_resumes_generation_in_finally() -> None:
    source = (
        REPO_ROOT / "rlix" / "pipeline" / "miles_model_update_service.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    guarded_finalize = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            _awaits_ray_get_finalize_refs(child)
            for stmt in node.body
            for child in ast.walk(stmt)
        )
    ]

    assert guarded_finalize, "finalize_refs await must be wrapped in try/finally"
    assert any(
        any(
            isinstance(child, ast.Attribute) and child.attr == "continue_generation"
            for stmt in node.finalbody
            for child in ast.walk(stmt)
        )
        for node in guarded_finalize
    ), "continue_generation must run from the finalize finally block"
