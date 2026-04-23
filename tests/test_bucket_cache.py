"""Unit tests for BucketRecord, VersionedBucketCache, and _bucket_named_tensors.

Uses REAL torch when installed (e.g. on Vast GPU instances), which is the
only way to correctly validate data integrity through pack/unpack round-trips.

When torch is not available (e.g. CI without GPU deps), torch-dependent tests
are skipped via pytest.importorskip, and structural/threading tests still run.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Real torch — mandatory for data-integrity tests
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="real torch required for bucket_cache tests")

import importlib.util  # noqa: E402
import sys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
_BUCKET_CACHE_PATH = REPO_ROOT / "rlix" / "pipeline" / "bucket_cache.py"

# Import bucket_cache.py directly by file path to bypass rlix/pipeline/__init__.py,
# which eagerly imports full_finetune_pipeline (requires codetiming, roll, etc.)
_spec = importlib.util.spec_from_file_location("rlix.pipeline.bucket_cache", _BUCKET_CACHE_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["rlix.pipeline.bucket_cache"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

BucketRecord = _mod.BucketRecord
VersionedBucketCache = _mod.VersionedBucketCache
_aligned_offset = _mod._aligned_offset
_bucket_named_tensors = _mod._bucket_named_tensors
unpack_bucket_record = _mod.unpack_bucket_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(*values, dtype=None) -> torch.Tensor:
    """Create a CPU float32 (or specified dtype) tensor from values."""
    return torch.tensor(list(values), dtype=dtype or torch.float32)


def _assert_tensors_equal(a: torch.Tensor, b: torch.Tensor, msg: str = "") -> None:
    """Assert two tensors have identical dtype, shape, and values."""
    assert a.dtype == b.dtype, f"{msg} dtype mismatch: {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{msg} shape mismatch: {a.shape} vs {b.shape}"
    assert torch.allclose(a.float(), b.float()), f"{msg} value mismatch:\n{a}\nvs\n{b}"


# ---------------------------------------------------------------------------
# _aligned_offset
# ---------------------------------------------------------------------------


def test_aligned_offset_zero():
    assert _aligned_offset(0) == 0


def test_aligned_offset_boundary():
    assert _aligned_offset(512) == 512


def test_aligned_offset_one_over():
    assert _aligned_offset(513) == 1024


def test_aligned_offset_arbitrary():
    assert _aligned_offset(1) == 512
    assert _aligned_offset(511) == 512
    assert _aligned_offset(1023) == 1024
    assert _aligned_offset(1024) == 1024
    assert _aligned_offset(1025) == 1536


# ---------------------------------------------------------------------------
# _bucket_named_tensors — structure
# ---------------------------------------------------------------------------


def test_bucket_named_tensors_single_structure():
    t = _t(1.0, 2.0, 3.0, 4.0)
    record = _bucket_named_tensors([("w", t)])
    assert record.param_names == ["w"]
    assert len(record.shapes) == 1
    assert len(record.dtypes) == 1
    assert record.offsets == [0]
    assert record.used_bytes == t.numel() * t.element_size()
    assert record.cpu_uint8_bucket.numel() >= record.used_bytes
    assert record.cpu_uint8_bucket.dtype == torch.uint8


def test_bucket_named_tensors_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        _bucket_named_tensors([])


def test_bucket_named_tensors_second_param_aligned():
    """Second param must start at 512-byte-aligned offset regardless of first param size."""
    t1 = _t(*[1.0] * 10)  # 10 × 4 = 40 bytes → first aligned boundary is 512
    t2 = _t(*[2.0] * 5)
    record = _bucket_named_tensors([("a", t1), ("b", t2)])
    assert record.offsets[0] == 0
    assert record.offsets[1] == 512


def test_bucket_named_tensors_used_bytes_excludes_padding():
    """used_bytes = raw element bytes only, without alignment padding."""
    t = _t(1.0, 2.0)  # 2 × 4 = 8 bytes
    record = _bucket_named_tensors([("w", t)])
    assert record.used_bytes == 8
    # But total buffer is at least 512 (one aligned slot)
    assert record.cpu_uint8_bucket.numel() >= 512


def test_bucket_named_tensors_multi_field_count():
    t1 = _t(1.0, 2.0)
    t2 = _t(3.0, 4.0, 5.0)
    t3 = _t(6.0)
    record = _bucket_named_tensors([("a", t1), ("b", t2), ("c", t3)])
    assert record.param_names == ["a", "b", "c"]
    assert len(record.offsets) == 3
    assert len(record.shapes) == 3
    assert len(record.dtypes) == 3


# ---------------------------------------------------------------------------
# _bucket_named_tensors + unpack_bucket_record — DATA INTEGRITY round-trip
# ---------------------------------------------------------------------------
# These tests verify that actual float values survive the pack → unpack cycle.
# This is the critical check the stub-based tests cannot provide.


def test_round_trip_single_float32():
    original = _t(1.5, -2.7, 3.14, 0.0)
    record = _bucket_named_tensors([("layer.weight", original)])
    unpacked = unpack_bucket_record(record)
    assert len(unpacked) == 1
    name, recovered = unpacked[0]
    assert name == "layer.weight"
    _assert_tensors_equal(recovered, original, msg="float32 round-trip")


def test_round_trip_multi_params():
    a = _t(1.0, 2.0, 3.0)
    b = _t(-1.0, -2.0)
    c = _t(100.0, 200.0, 300.0, 400.0)
    record = _bucket_named_tensors([("a", a), ("b", b), ("c", c)])
    unpacked = unpack_bucket_record(record)
    assert [n for n, _ in unpacked] == ["a", "b", "c"]
    _assert_tensors_equal(unpacked[0][1], a, msg="param a")
    _assert_tensors_equal(unpacked[1][1], b, msg="param b")
    _assert_tensors_equal(unpacked[2][1], c, msg="param c")


def test_round_trip_preserves_negative_values():
    t = _t(-999.5, -0.001, -1e6)
    record = _bucket_named_tensors([("w", t)])
    name, recovered = unpack_bucket_record(record)[0]
    _assert_tensors_equal(recovered, t, msg="negative values")


def test_round_trip_preserves_zero():
    t = torch.zeros(8, dtype=torch.float32)
    record = _bucket_named_tensors([("w", t)])
    _, recovered = unpack_bucket_record(record)[0]
    _assert_tensors_equal(recovered, t, msg="all-zeros")


def test_round_trip_2d_shape():
    """Shape must be preserved through pack/unpack."""
    original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    record = _bucket_named_tensors([("mat", original)])
    _, recovered = unpack_bucket_record(record)[0]
    assert recovered.shape == original.shape, f"shape mismatch: {recovered.shape}"
    _assert_tensors_equal(recovered, original, msg="2D shape")


def test_round_trip_float16():
    """float16 tensors must survive byte reinterpretation correctly."""
    original = _t(1.0, 2.0, 3.0, 4.0, dtype=torch.float16)
    record = _bucket_named_tensors([("w", original)])
    _, recovered = unpack_bucket_record(record)[0]
    assert recovered.dtype == torch.float16
    _assert_tensors_equal(recovered, original, msg="float16 round-trip")


def test_round_trip_large_param():
    """Large tensor (>512 bytes) must not corrupt data across the alignment boundary."""
    original = torch.arange(256, dtype=torch.float32)  # 256 × 4 = 1024 bytes
    record = _bucket_named_tensors([("big", original)])
    _, recovered = unpack_bucket_record(record)[0]
    _assert_tensors_equal(recovered, original, msg="large param")


def test_round_trip_mixed_dtypes():
    """float32 and float16 params in the same bucket must both recover correctly."""
    a = _t(1.0, 2.0, dtype=torch.float32)
    b = _t(3.0, 4.0, dtype=torch.float16)
    record = _bucket_named_tensors([("a", a), ("b", b)])
    unpacked = {n: t for n, t in unpack_bucket_record(record)}
    _assert_tensors_equal(unpacked["a"], a, msg="float32 in mixed")
    _assert_tensors_equal(unpacked["b"], b, msg="float16 in mixed")


def test_round_trip_many_small_params():
    """Many small params (each << 512 bytes) must all recover correctly."""
    originals = {f"w{i}": _t(float(i)) for i in range(20)}
    record = _bucket_named_tensors(list(originals.items()))
    unpacked = {n: t for n, t in unpack_bucket_record(record)}
    for name, original in originals.items():
        _assert_tensors_equal(unpacked[name], original, msg=f"param {name}")


# ---------------------------------------------------------------------------
# _bucket_named_tensors — buffer is CPU uint8, contiguous
# ---------------------------------------------------------------------------


def test_bucket_buffer_is_cpu():
    t = _t(1.0)
    record = _bucket_named_tensors([("w", t)])
    assert record.cpu_uint8_bucket.device.type == "cpu"


def test_bucket_buffer_is_contiguous():
    t = _t(1.0, 2.0, 3.0)
    record = _bucket_named_tensors([("w", t)])
    assert record.cpu_uint8_bucket.is_contiguous()


def test_bucket_buffer_dtype_is_uint8():
    t = _t(1.0)
    record = _bucket_named_tensors([("w", t)])
    assert record.cpu_uint8_bucket.dtype == torch.uint8


# ---------------------------------------------------------------------------
# unpack_bucket_record — element_size via torch.empty (not buf slice)
# ---------------------------------------------------------------------------


def test_unpack_element_size_does_not_read_buf_slice():
    """Verify unpack works even when offset+1 < dtype.itemsize (float32 needs 4 bytes).

    Previously buggy: buf[offset:offset+1].view(float32) would raise RuntimeError
    in real torch because 1 uint8 byte cannot be reinterpreted as float32.
    """
    t = _t(42.0)  # 1-element float32 = 4 bytes; offset=0, buf[0:1] has 1 byte
    record = _bucket_named_tensors([("w", t)])
    # This must not raise RuntimeError
    unpacked = unpack_bucket_record(record)
    _, recovered = unpacked[0]
    _assert_tensors_equal(recovered, t, msg="single element float32 unpack")


# ---------------------------------------------------------------------------
# VersionedBucketCache — two-pointer versioning
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache():
    return VersionedBucketCache()


@pytest.fixture()
def sample_buckets():
    t = _t(1.0, 2.0, 3.0, 4.0)
    return [_bucket_named_tensors([("w", t)])]


def test_cache_ready_step_none_before_promote(cache):
    assert cache.cache_ready_step is None


def test_latest_version_none_before_build(cache):
    assert cache.latest_version is None


def test_build_latest_sets_latest_not_active(cache, sample_buckets):
    cache.build_latest(0, sample_buckets)
    assert cache.latest_version == 0
    assert cache.cache_ready_step is None  # active not set yet


def test_promote_sets_active(cache, sample_buckets):
    cache.build_latest(0, sample_buckets)
    cache.promote(0)
    assert cache.cache_ready_step == 0


def test_get_active_buckets_raises_before_promote(cache, sample_buckets):
    cache.build_latest(0, sample_buckets)
    with pytest.raises(RuntimeError, match="promote"):
        with cache._cache_lock:
            cache.get_active_buckets()


def test_promote_unknown_version_raises(cache):
    with pytest.raises(KeyError):
        cache.promote(99)


def test_base_version_minus_one(cache, sample_buckets):
    cache.build_latest(-1, sample_buckets)
    cache.promote(-1)
    assert cache.cache_ready_step == -1


# ---------------------------------------------------------------------------
# GC invariant — only latest + active kept
# ---------------------------------------------------------------------------


def test_gc_keeps_only_latest_and_active(cache):
    def _make(val):
        return [_bucket_named_tensors([("w", _t(float(val)))])]

    for step in range(5):
        cache.build_latest(step, _make(step))
        cache.promote(step)

    with cache._cache_lock:
        # After promote(4): active=4, latest=4 → only 4 kept
        assert set(cache._cache_map.keys()) == {4}


def test_gc_keeps_latest_and_active_when_different(cache):
    def _make(val):
        return [_bucket_named_tensors([("w", _t(float(val)))])]

    cache.build_latest(0, _make(0))
    cache.promote(0)
    cache.build_latest(1, _make(1))
    # Not promoted yet — active=0, latest=1
    with cache._cache_lock:
        assert set(cache._cache_map.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# Active buckets contain the correct data after promote
# ---------------------------------------------------------------------------


def test_get_active_buckets_returns_correct_version_data(cache):
    """The data returned by get_active_buckets() must match what was built for that version."""
    v0_data = _t(10.0, 20.0)
    v1_data = _t(30.0, 40.0)

    cache.build_latest(0, [_bucket_named_tensors([("w", v0_data)])])
    cache.promote(0)
    cache.build_latest(1, [_bucket_named_tensors([("w", v1_data)])])
    cache.promote(1)

    with cache._cache_lock:
        buckets = cache.get_active_buckets()

    assert len(buckets) == 1
    _, recovered = unpack_bucket_record(buckets[0])[0]
    _assert_tensors_equal(recovered, v1_data, msg="active buckets after promote(1)")


def test_get_active_buckets_does_not_return_stale_version(cache):
    """After promote(1), active data must be v1, not v0."""
    v0_data = _t(1.0, 2.0)
    v1_data = _t(99.0, 88.0)

    cache.build_latest(0, [_bucket_named_tensors([("w", v0_data)])])
    cache.promote(0)
    cache.build_latest(1, [_bucket_named_tensors([("w", v1_data)])])
    cache.promote(1)

    with cache._cache_lock:
        buckets = cache.get_active_buckets()

    _, recovered = unpack_bucket_record(buckets[0])[0]
    # Must NOT match v0 data
    assert not torch.allclose(recovered.float(), v0_data.float()), (
        "get_active_buckets returned stale v0 data after promote(1)"
    )
    _assert_tensors_equal(recovered, v1_data, msg="active must be v1")


# ---------------------------------------------------------------------------
# Version tracking across multiple steps
# ---------------------------------------------------------------------------


def test_sequential_step_promotion(cache):
    for step in range(5):
        t = _t(float(step))
        cache.build_latest(step, [_bucket_named_tensors([("w", t)])])
        cache.promote(step)
        assert cache.cache_ready_step == step


def test_is_version_built(cache, sample_buckets):
    assert not cache.is_version_built(0)
    cache.build_latest(0, sample_buckets)
    assert cache.is_version_built(0)
    cache.promote(0)
    assert cache.is_version_built(0)


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_concurrent_build_latest_safe(cache):
    errors: list[Exception] = []

    def _writer(version: int):
        try:
            t = _t(float(version))
            cache.build_latest(version, [_bucket_named_tensors([("w", t)])])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(16)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == [], f"Thread errors: {errors}"
