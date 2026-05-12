"""GPU integration tests for the CPU bucket cache pipeline.

Tests the full weight caching round-trip on a real GPU using a tiny model:
  1. GPU memory is actually released after offloading weights to CPU.
  2. Weights packed into BucketRecord match the original model parameters
     bit-for-bit (no dtype promotion, no data corruption).
  3. unpack_bucket_record correctly reconstructs the source state_dict so it
     matches the source (simulates pushing weights to an inference worker).
  4. No shape or dtype mismatch survives the full cache → push pipeline.
  5. VersionedBucketCache version tracking works correctly across build/promote.

Run on Vast.ai with a real GPU:
    pytest tests/integration/test_bucket_cache_gpu.py -v

Requirements:
    pip install torch transformers
    (No NeMo or Ray needed — uses HuggingFace Qwen2.5-0.5B directly)
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

# ---------------------------------------------------------------------------
# Import pipeline modules directly by file path to avoid pulling in the full
# rlix package (which requires ray, codetiming, and other heavy deps).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "rlix" / "pipeline"

import importlib.util as _ilu

def _load(name: str, file: Path):
    spec = _ilu.spec_from_file_location(name, file)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_bucket_cache_mod = _load("rlix.pipeline.bucket_cache", PIPELINE_DIR / "bucket_cache.py")

BucketRecord = _bucket_cache_mod.BucketRecord
VersionedBucketCache = _bucket_cache_mod.VersionedBucketCache
_bucket_named_tensors = _bucket_cache_mod._bucket_named_tensors
unpack_bucket_record = _bucket_cache_mod.unpack_bucket_record

# ---------------------------------------------------------------------------
# Skip entire module if no CUDA GPU available
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU integration tests require CUDA",
)

# Tiny model — fast to load, fits on any GPU
MODEL_NAME = "Qwen/Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_allocated_mb() -> float:
    return torch.cuda.memory_allocated() / (1024**2)


def _gpu_reserved_mb() -> float:
    return torch.cuda.memory_reserved() / (1024**2)


def _load_tiny_model() -> tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    """Load Qwen2.5-0.5B onto GPU. Returns (model, original_state_dict_cpu)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()

    # snapshot original weights on CPU for comparison
    original = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return model, original


def _model_to_bucket_records(
    model: torch.nn.Module,
    bucket_size: int = 128,
) -> List[BucketRecord]:
    """Pack model parameters into BucketRecord list using new Feature 4 API.

    Partitions params into groups of up to bucket_size names and packs each
    group into one BucketRecord via _bucket_named_tensors.
    """
    items = [
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in model.state_dict().items()
    ]
    records = []
    for i in range(0, len(items), bucket_size):
        chunk = items[i : i + bucket_size]
        records.append(_bucket_named_tensors(chunk))
    return records


def _apply_records_to_state_dict(
    records: List[BucketRecord],
    target_sd: Dict[str, torch.Tensor],
) -> None:
    """Unpack all bucket records and copy weights into target_sd."""
    for record in records:
        for name, tensor in unpack_bucket_record(record):
            if name in target_sd:
                target_sd[name].copy_(tensor.to(target_sd[name].device))


# ---------------------------------------------------------------------------
# Test 1 — GPU memory is released after offloading to CPU
# ---------------------------------------------------------------------------


class TestGPUMemoryRelease:
    def test_offload_reduces_allocated_memory(self):
        """Moving model to CPU + empty_cache must drop GPU allocated MB."""
        model, _ = _load_tiny_model()

        before_mb = _gpu_allocated_mb()
        assert before_mb > 100, (
            f"Expected model to occupy >100 MB on GPU, got {before_mb:.1f} MB"
        )

        # offload
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        after_mb = _gpu_allocated_mb()
        released_pct = (before_mb - after_mb) / before_mb * 100
        assert released_pct >= 90, (
            f"Expected >=90% GPU memory released, "
            f"before={before_mb:.1f}MB after={after_mb:.1f}MB "
            f"released={released_pct:.1f}%"
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def test_cache_does_not_hold_gpu_tensors(self):
        """BucketRecord must store CPU tensors only — no GPU residue."""
        model, _ = _load_tiny_model()
        records = _model_to_bucket_records(model)

        # move model off GPU
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        before_mb = _gpu_allocated_mb()

        # iterating the records must not re-allocate GPU memory
        for record in records:
            assert record.cpu_uint8_bucket.device.type == "cpu", (
                f"BucketRecord has GPU tensor: device={record.cpu_uint8_bucket.device}"
            )

        after_mb = _gpu_allocated_mb()
        assert after_mb <= before_mb, (
            f"Reading cache increased GPU memory: {before_mb:.1f}MB → {after_mb:.1f}MB"
        )

        del model, records
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 2 — Weight correctness: packed bucket matches original model
# ---------------------------------------------------------------------------


class TestWeightCorrectnessInCache:
    def test_cached_weights_match_original_bit_for_bit(self):
        """Every parameter in BucketRecord must equal the original GPU tensor."""
        model, original_cpu = _load_tiny_model()
        records = _model_to_bucket_records(model)

        assert len(records) > 0, "No records produced — nothing was packed"

        # Unpack all records and build a flat name→tensor dict
        unpacked: Dict[str, torch.Tensor] = {}
        for record in records:
            for name, tensor in unpack_bucket_record(record):
                unpacked[name] = tensor

        mismatches: list[str] = []
        for name, original_tensor in original_cpu.items():
            if name not in unpacked:
                mismatches.append(f"{name}: missing from unpacked records")
                continue
            cached = unpacked[name]
            if cached.shape != original_tensor.shape:
                mismatches.append(
                    f"{name}: shape {cached.shape} != {original_tensor.shape}"
                )
            elif cached.dtype != original_tensor.dtype:
                mismatches.append(
                    f"{name}: dtype {cached.dtype} != {original_tensor.dtype}"
                )
            elif not torch.equal(cached, original_tensor):
                max_diff = (cached.float() - original_tensor.float()).abs().max().item()
                mismatches.append(f"{name}: values differ, max_diff={max_diff:.6f}")

        assert not mismatches, (
            f"{len(mismatches)} weight mismatches found:\n" + "\n".join(mismatches[:10])
        )

        del model, records
        gc.collect()
        torch.cuda.empty_cache()

    def test_cached_dtypes_preserved(self):
        """bfloat16 model → packed uint8 buffer → unpacked tensors must be bfloat16."""
        model, _ = _load_tiny_model()  # loaded as bfloat16
        records = _model_to_bucket_records(model)

        wrong_dtype: list[str] = []
        for record in records:
            for name, tensor in unpack_bucket_record(record):
                if tensor.dtype != torch.bfloat16:
                    wrong_dtype.append(f"{name}: {tensor.dtype}")

        assert not wrong_dtype, (
            "Some tensors were upcast from bfloat16:\n" + "\n".join(wrong_dtype[:5])
        )

        del model, records
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 3 — Push weights to a target state_dict via unpack_bucket_record
# ---------------------------------------------------------------------------


class TestBucketRecordPush:
    def _make_zero_state_dict(
        self, reference: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Create a state_dict of zeros with same shapes/dtypes as reference."""
        return {
            name: torch.zeros_like(tensor)
            for name, tensor in reference.items()
        }

    def test_push_updates_all_parameters(self):
        """After _apply_records_to_state_dict, every parameter must match source."""
        model, original_cpu = _load_tiny_model()
        records = _model_to_bucket_records(model)

        target_sd = self._make_zero_state_dict(original_cpu)
        _apply_records_to_state_dict(records, target_sd)

        mismatches: list[str] = []
        for name, original_tensor in original_cpu.items():
            received = target_sd[name]
            if not torch.equal(received, original_tensor):
                max_diff = (
                    received.float() - original_tensor.float()
                ).abs().max().item()
                mismatches.append(f"{name}: max_diff={max_diff:.6f}")

        assert not mismatches, (
            f"{len(mismatches)} parameters differ after push:\n"
            + "\n".join(mismatches[:10])
        )

        del model, records
        gc.collect()
        torch.cuda.empty_cache()

    def test_push_no_shape_mismatch(self):
        """Shapes in target state_dict must not change after push."""
        model, original_cpu = _load_tiny_model()
        records = _model_to_bucket_records(model)
        target_sd = self._make_zero_state_dict(original_cpu)

        _apply_records_to_state_dict(records, target_sd)

        shape_errors: list[str] = []
        for name, original_tensor in original_cpu.items():
            if target_sd[name].shape != original_tensor.shape:
                shape_errors.append(
                    f"{name}: {target_sd[name].shape} != {original_tensor.shape}"
                )

        assert not shape_errors, "\n".join(shape_errors)

        del model, records
        gc.collect()
        torch.cuda.empty_cache()

    def test_push_to_gpu_target(self):
        """Push from CPU cache to GPU state_dict — copy_ must handle cross-device."""
        model, original_cpu = _load_tiny_model()
        records = _model_to_bucket_records(model)

        # target lives on GPU (simulates actual vLLM inference worker)
        target_sd = {
            name: torch.zeros_like(tensor, device="cuda")
            for name, tensor in original_cpu.items()
        }

        _apply_records_to_state_dict(records, target_sd)

        mismatches: list[str] = []
        for name, original_tensor in original_cpu.items():
            received_cpu = target_sd[name].cpu()
            if not torch.equal(received_cpu, original_tensor):
                max_diff = (
                    received_cpu.float() - original_tensor.float()
                ).abs().max().item()
                mismatches.append(f"{name}: max_diff={max_diff:.6f}")

        assert not mismatches, (
            f"{len(mismatches)} parameters differ after GPU push:\n"
            + "\n".join(mismatches[:10])
        )

        del model, records
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 4 — VersionedBucketCache version tracking
# ---------------------------------------------------------------------------


class TestVersionedBucketCache:
    def test_build_and_promote_version(self):
        """build_latest + promote makes the version accessible via get_active_buckets."""
        model, original_cpu = _load_tiny_model()
        records = _model_to_bucket_records(model)

        cache = VersionedBucketCache()
        assert cache.cache_ready_step is None

        cache.build_latest(version=1, buckets=records)
        assert cache.latest_version == 1
        assert cache.cache_ready_step is None  # not promoted yet

        cache.promote(version=1)
        assert cache.cache_ready_step == 1

        active = cache.get_active_buckets()
        assert len(active) == len(records)

        # verify active buckets still match original
        unpacked: Dict[str, torch.Tensor] = {}
        for record in active:
            for name, tensor in unpack_bucket_record(record):
                unpacked[name] = tensor

        mismatches = [
            name for name, orig in original_cpu.items()
            if name not in unpacked or not torch.equal(unpacked[name], orig)
        ]
        assert not mismatches, f"Active buckets differ from original: {mismatches[:5]}"

        del model, records, cache
        gc.collect()
        torch.cuda.empty_cache()

    def test_gc_drops_old_version(self):
        """After building v2, v0 must be GC'd (only v1=latest and v2=active kept)."""
        model, _ = _load_tiny_model()
        records = _model_to_bucket_records(model)

        cache = VersionedBucketCache()
        cache.build_latest(version=0, buckets=records)
        cache.promote(version=0)
        cache.build_latest(version=1, buckets=records)
        cache.promote(version=1)
        cache.build_latest(version=2, buckets=records)  # v0 should be GC'd now

        assert not cache.is_version_built(0), "Version 0 should have been GC'd"
        assert cache.is_version_built(1)
        assert cache.is_version_built(2)

        del model, records, cache
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 5 — Full round-trip: GPU model → VersionedBucketCache → infer worker
# ---------------------------------------------------------------------------


class TestFullRoundTrip:
    def test_full_cache_roundtrip_matches_source(self):
        """End-to-end: train model (GPU) → VersionedBucketCache (CPU) → offload → push → verify."""
        model, original_cpu = _load_tiny_model()

        # Step 1: build CPU cache (simulates build_latest_bucket_cache)
        records = _model_to_bucket_records(model)
        cache = VersionedBucketCache()
        cache.build_latest(version=0, buckets=records)
        cache.promote(version=0)

        gpu_before_offload_mb = _gpu_allocated_mb()

        # Step 2: offload training model (simulates NCCL destroy + GPU release)
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        gpu_after_offload_mb = _gpu_allocated_mb()
        released_pct = (
            (gpu_before_offload_mb - gpu_after_offload_mb) / gpu_before_offload_mb * 100
            if gpu_before_offload_mb > 0
            else 100.0
        )
        assert released_pct >= 80, (
            f"GPU not sufficiently released after offload: {released_pct:.1f}%"
        )

        # Step 3: simulate inference worker wake_up — empty GPU model
        infer_sd = {
            name: torch.zeros_like(tensor, device="cuda")
            for name, tensor in original_cpu.items()
        }

        # Step 4: push active cache to inference worker (Feature 6)
        active_buckets = cache.get_active_buckets()
        _apply_records_to_state_dict(active_buckets, infer_sd)

        # Step 5: verify weights are correct on inference side
        mismatches: list[str] = []
        for name, original_tensor in original_cpu.items():
            received = infer_sd[name].cpu()
            if not torch.equal(received, original_tensor):
                max_diff = (
                    received.float() - original_tensor.float()
                ).abs().max().item()
                mismatches.append(f"{name}: max_diff={max_diff:.6f}")

        assert not mismatches, (
            f"Full round-trip: {len(mismatches)} mismatches:\n"
            + "\n".join(mismatches[:10])
        )

        del model, records, cache, infer_sd
        gc.collect()
        torch.cuda.empty_cache()
