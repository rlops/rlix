from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _calls_attr(node: ast.AST, attr: str) -> bool:
    return any(isinstance(child, ast.Attribute) and child.attr == attr for child in ast.walk(node))


def test_after_training_releases_train_gpus_when_weight_sync_fails() -> None:
    source = (REPO_ROOT / "rlix" / "pipeline" / "miles_pipeline.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    after_training = _function_def(tree, "_after_training")

    guarded_sync = [
        node
        for node in ast.walk(after_training)
        if isinstance(node, ast.Try)
        and any(_calls_attr(stmt, "sync_base_weights_to_active") for stmt in node.body)
    ]

    assert guarded_sync, "sync_base_weights_to_active must be guarded by try/finally"
    assert any(
        any(_calls_attr(stmt, "_notify_release_cluster_gpus") for stmt in node.finalbody)
        for node in guarded_sync
    ), "actor_train release must run from the sync finally block"
