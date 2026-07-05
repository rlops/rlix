from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_attr(node: ast.AST, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == attr


def test_miles_uses_whole_gpu_residual_threshold() -> None:
    source = (REPO_ROOT / "rlix" / "pipeline" / "miles_coordinator.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    shrink_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_shrink_workers"
    )

    assert any(
        isinstance(node, ast.Call)
        and _is_name(node.func, "parse_env_positive_float")
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "MILES_MAX_RESIDUAL_GPU_MEM_GB"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 7.0
        for node in ast.walk(shrink_fn)
    ), "_shrink_workers must parse the residual threshold env var with 7GB default"

    assert any(
        isinstance(node, ast.Call)
        and _is_attr(node.func, "remote")
        and any(
            kw.arg == "post_sleep_vram_threshold_gb"
            and _is_name(kw.value, "residual_threshold_gb")
            for kw in node.keywords
        )
        for node in ast.walk(shrink_fn)
    ), "shrink_engines must receive post_sleep_vram_threshold_gb"


def test_miles_coordinator_forwards_residual_threshold_env_var() -> None:
    source = (REPO_ROOT / "rlix" / "pipeline" / "miles_coordinator.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    build_env_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_pipeline_env_vars"
    )

    assert any(
        isinstance(node, ast.Constant)
        and node.value == "MILES_MAX_RESIDUAL_GPU_MEM_GB"
        for node in ast.walk(build_env_fn)
    ), "_build_pipeline_env_vars must forward MILES_MAX_RESIDUAL_GPU_MEM_GB"


def test_miles_pipeline_enforces_whole_gpu_residual_gate() -> None:
    source = (REPO_ROOT / "rlix" / "pipeline" / "miles_pipeline.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    wait_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_wait_for_overlap_engines_offloaded"
    )

    assert "_probe_max_used_gpu_mem_gb" in source
    assert "whole-GPU residual hard gate" in source
    assert "non-SGLang co-tenants" in source
    assert any(
        isinstance(node, ast.Call)
        and _is_name(node.func, "parse_env_positive_float")
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "MILES_MAX_RESIDUAL_GPU_MEM_GB"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 7.0
        for node in ast.walk(wait_fn)
    ), "MilesPipeline hard gate must parse MILES_MAX_RESIDUAL_GPU_MEM_GB with 7GB default"
    assert any(isinstance(node, ast.Raise) for node in ast.walk(wait_fn)), (
        "MilesPipeline must raise when whole-GPU residual exceeds the threshold"
    )
