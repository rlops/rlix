"""GPU integration tests for the CPU bucket cache pipeline.

Tests the full weight caching round-trip on a real GPU using a tiny model:
  1. GPU memory is actually released after offloading weights to CPU.
  2. Weights stored in CPUBucketCache match the original model parameters
     bit-for-bit (no dtype promotion, no data corruption).
  3. BucketReceiver correctly patches a target state_dict so it matches
     the source (simulates pushing weights to an inference worker).
  4. No shape or dtype mismatch survives the full cache → push pipeline.

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
from typing import Dict

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
_bucket_receiver_mod = _load("rlix.pipeline.bucket_receiver", PIPELINE_DIR / "bucket_receiver.py")

CPUBucketCache = _bucket_cache_mod.CPUBucketCache
Bucket = _bucket_cache_mod.Bucket
BucketUpdateRequest = _bucket_receiver_mod.BucketUpdateRequest
apply_bucket_update = _bucket_receiver_mod.apply_bucket_update

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


def _model_to_cpu_cache(model: torch.nn.Module) -> CPUBucketCache:
    """Copy all model parameters into a CPUBucketCache (shard_id=0 for all)."""
    cache = CPUBucketCache()
    with torch.no_grad():
        for name, param in model.named_parameters():
            cache.store((name, 0), param.detach().cpu().contiguous())
    return cache


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
        """CPUBucketCache must store CPU tensors only — no GPU residue."""
        model, _ = _load_tiny_model()
        cache = _model_to_cpu_cache(model)

        # move model off GPU
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        before_mb = _gpu_allocated_mb()

        # iterating the cache must not re-allocate GPU memory
        dirty = cache.get_dirty_buckets()
        for (name, shard_id), tensor in dirty.items():
            assert tensor.device.type == "cpu", (
                f"Cache stored GPU tensor for {name!r}: device={tensor.device}"
            )

        after_mb = _gpu_allocated_mb()
        assert after_mb <= before_mb, (
            f"Reading cache increased GPU memory: {before_mb:.1f}MB → {after_mb:.1f}MB"
        )

        del model, cache
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 2 — Weight correctness: cache stores exactly what the model has
# ---------------------------------------------------------------------------


class TestWeightCorrectnessInCache:
    def test_cached_weights_match_original_bit_for_bit(self):
        """Every parameter in CPUBucketCache must equal the original GPU tensor."""
        model, original_cpu = _load_tiny_model()
        cache = _model_to_cpu_cache(model)

        dirty = cache.get_dirty_buckets()
        assert len(dirty) > 0, "Cache is empty — nothing was stored"

        mismatches: list[str] = []
        for name, original_tensor in original_cpu.items():
            key = (name, 0)
            if key not in dirty:
                mismatches.append(f"{name}: missing from cache")
                continue

            cached = dirty[key]
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

        del model, cache
        gc.collect()
        torch.cuda.empty_cache()

    def test_cached_dtypes_preserved(self):
        """bfloat16 model → cache tensors must be bfloat16, not upcast."""
        model, _ = _load_tiny_model()  # loaded as bfloat16
        cache = _model_to_cpu_cache(model)

        wrong_dtype: list[str] = []
        for (name, _), tensor in cache.get_dirty_buckets().items():
            if tensor.dtype != torch.bfloat16:
                wrong_dtype.append(f"{name}: {tensor.dtype}")

        assert not wrong_dtype, (
            "Some tensors were upcast from bfloat16:\n" + "\n".join(wrong_dtype[:5])
        )

        del model, cache
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 3 — BucketReceiver: pushing weights to a target state_dict
# ---------------------------------------------------------------------------


class TestBucketReceiverPush:
    def _make_zero_state_dict(
        self, reference: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Create a state_dict of zeros with same shapes/dtypes as reference."""
        return {
            name: torch.zeros_like(tensor)
            for name, tensor in reference.items()
        }

    def test_push_updates_all_parameters(self):
        """After apply_bucket_update, every parameter in target must match source."""
        model, original_cpu = _load_tiny_model()
        cache = _model_to_cpu_cache(model)

        # target = zero-initialised inference model (simulated)
        target_sd = self._make_zero_state_dict(original_cpu)

        # build BucketUpdateRequest from dirty cache
        dirty = cache.get_dirty_buckets()
        buckets: list[Bucket] = [
            Bucket(param_name=name, shard_id=shard_id, data=tensor)
            for (name, shard_id), tensor in dirty.items()
        ]
        request = BucketUpdateRequest(sync_id=1, buckets=buckets)

        result = apply_bucket_update(target_sd, request)
        assert result.ok, f"apply_bucket_update failed: {result.errors}"

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

        del model, cache
        gc.collect()
        torch.cuda.empty_cache()

    def test_push_no_shape_mismatch(self):
        """Shapes in target state_dict must not change after push."""
        model, original_cpu = _load_tiny_model()
        cache = _model_to_cpu_cache(model)
        target_sd = self._make_zero_state_dict(original_cpu)

        dirty = cache.get_dirty_buckets()
        buckets = [
            Bucket(param_name=n, shard_id=s, data=t)
            for (n, s), t in dirty.items()
        ]
        apply_bucket_update(target_sd, BucketUpdateRequest(sync_id=2, buckets=buckets))

        shape_errors: list[str] = []
        for name, original_tensor in original_cpu.items():
            if target_sd[name].shape != original_tensor.shape:
                shape_errors.append(
                    f"{name}: {target_sd[name].shape} != {original_tensor.shape}"
                )

        assert not shape_errors, "\n".join(shape_errors)

        del model, cache
        gc.collect()
        torch.cuda.empty_cache()

    def test_push_to_gpu_target(self):
        """Push from CPU cache to GPU state_dict — tensor.copy_ must handle cross-device."""
        model, original_cpu = _load_tiny_model()
        cache = _model_to_cpu_cache(model)

        # target lives on GPU (simulates actual vLLM inference worker)
        target_sd = {
            name: torch.zeros_like(tensor, device="cuda")
            for name, tensor in original_cpu.items()
        }

        dirty = cache.get_dirty_buckets()
        buckets = [
            Bucket(param_name=n, shard_id=s, data=t)
            for (n, s), t in dirty.items()
        ]
        result = apply_bucket_update(target_sd, BucketUpdateRequest(sync_id=3, buckets=buckets))
        assert result.ok, f"apply_bucket_update to GPU target failed: {result.errors}"

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

        del model, cache
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 4 — Full round-trip: GPU model → CPU cache → zero inference model → verify
# ---------------------------------------------------------------------------


class TestFullRoundTrip:
    def test_full_cache_roundtrip_matches_source(self):
        """End-to-end: train model (GPU) → cache (CPU) → offload → push → verify."""
        model, original_cpu = _load_tiny_model()

        # Step 1: build CPU cache (simulates build_cpu_bucket_cache)
        cache = _model_to_cpu_cache(model)

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

        # Step 4: push dirty cache to inference worker
        dirty = cache.get_dirty_buckets()
        buckets = [
            Bucket(param_name=n, shard_id=s, data=t)
            for (n, s), t in dirty.items()
        ]
        result = apply_bucket_update(
            infer_sd, BucketUpdateRequest(sync_id=99, buckets=buckets)
        )
        assert result.ok, f"Weight push failed: {result.errors}"

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

        del model, cache, infer_sd
        gc.collect()
        torch.cuda.empty_cache()
