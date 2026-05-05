"""Environment installation test.

Catches hash/version mismatches between setup.py CACHED_DEPENDENCIES and
pyproject.toml before they block a fresh uv sync.

Run BEFORE any other test on a clean machine:
    python tests/test_env_install.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEMO_ROOT = REPO_ROOT / "external" / "NeMo"


def _extract_vcs_pins(path: Path) -> dict[str, str]:
    """Return {package_name: commit_hash} for all git+ VCS deps in a file."""
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    text = path.read_text()
    for m in re.finditer(
        r'([A-Za-z0-9_\-]+)\s*(?:@|==)\s*git\+https?://[^\s"\']+@([0-9a-f]{7,40})',
        text,
    ):
        pkg = m.group(1).lower().replace("-", "_").replace(".", "_")
        pins[pkg] = m.group(2)
    return pins


def test_no_vcs_hash_mismatch_between_setup_and_pyproject() -> None:
    """All git+ VCS deps in 3rdparty/*/setup.py must use the same commit hash
    as pyproject.toml.  A mismatch causes uv sync to fail on a fresh install."""

    pyproject = _extract_vcs_pins(NEMO_ROOT / "pyproject.toml")
    mismatches: list[str] = []

    for setup_py in sorted((NEMO_ROOT / "3rdparty").glob("*/setup.py")):
        setup_pins = _extract_vcs_pins(setup_py)
        for pkg, hash_in_setup in setup_pins.items():
            if pkg in pyproject and pyproject[pkg] != hash_in_setup:
                mismatches.append(
                    f"{setup_py.relative_to(REPO_ROOT)}: {pkg} "
                    f"setup={hash_in_setup[:12]} pyproject={pyproject[pkg][:12]}"
                )

    assert not mismatches, (
        "VCS dependency hash mismatch (uv sync will fail on fresh install):\n"
        + "\n".join(f"  {m}" for m in mismatches)
    )
    print(f"PASS: no VCS hash mismatches found (checked {len(pyproject)} pins)")


def test_nemo_submodule_initialized() -> None:
    """Verify the NeMo submodule has been checked out (not empty)."""
    assert (NEMO_ROOT / "pyproject.toml").exists(), (
        "external/NeMo is empty — run: git submodule update --init --recursive"
    )
    print("PASS: NeMo submodule is initialized")


def test_rlix_bucket_cache_importable() -> None:
    """Verify core rlix module loads without the full NeMo/Ray stack."""
    import importlib.util
    path = REPO_ROOT / "rlix" / "pipeline" / "bucket_cache.py"
    spec = importlib.util.spec_from_file_location("rlix.pipeline.bucket_cache", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rlix.pipeline.bucket_cache"] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "BucketRecord")
    assert hasattr(mod, "VersionedBucketCache")
    assert hasattr(mod, "_bucket_named_tensors")
    assert hasattr(mod, "unpack_bucket_record")
    print("PASS: rlix.pipeline.bucket_cache importable")


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                print(f"FAIL {name}: {e}", file=sys.stderr)
                failed += 1
    if failed:
        sys.exit(1)
    print(f"\nAll environment checks passed.")
