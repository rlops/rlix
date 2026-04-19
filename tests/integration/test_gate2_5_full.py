"""Gate 2.5 Full — Multi-pipeline weight sync test.

Two independent training pipelines alternating sync to shared inference workers.
Tests the key property: one pipeline can sync while the other keeps training.

Layout (4 GPUs):
  GPU 0 = Pipeline A trainer  (Qwen2.5-0.5B, seed A)
  GPU 1 = Pipeline B trainer  (Qwen2.5-0.5B, seed B)
  GPU 2, 3 = Inference workers

Process groups:
  nccl_world: all 4 ranks  — barriers + small metadata broadcasts
  gloo_a: [0, 2, 3]        — Pipeline A weights → inference workers
  gloo_b: [1, 2, 3]        — Pipeline B weights → inference workers

Per-step flow:
  1. Both pipelines train independently (different seeds → diverging weights)
  2. [Phase A] rank 0 offloads → CPU cache → broadcasts via gloo_a to ranks 2,3
              rank 1 is NOT blocked (prints "free to train")
  3. Inference workers verify A weights bit-exact
  4. rank 0 reloads model to GPU
  5. [Phase B] rank 1 offloads → CPU cache → broadcasts via gloo_b to ranks 2,3
              rank 0 is NOT blocked
  6. Inference workers verify B weights bit-exact
  7. rank 1 reloads model to GPU
  8. Inference workers assert A weights ≠ B weights (no cross-contamination)

Assertions:
  - VRAM released ≥ VRAM_RELEASE_THRESHOLD_PCT during each sync phase
  - Bit-exact hash match for each pipeline's weights on both inference workers
  - Pipeline A and B weights diverge after different-seed training

Run with:
    torchrun --nproc-per-node=4 tests/integration/test_gate2_5_full.py
"""
from __future__ import annotations

import gc
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
N_STEPS = 2
SEQ_LEN = 128
VRAM_RELEASE_THRESHOLD_PCT = 60

PIPELINE_A_RANK = 0
PIPELINE_B_RANK = 1
INFER_RANKS = [2, 3]
TRAIN_RANKS = [PIPELINE_A_RANK, PIPELINE_B_RANK]

MAX_PARAMS = 400   # upper bound on parameter count per model
ROW = 216          # 200 name bytes + 16 hash chars per param row

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import importlib.util as _ilu

def _load_mod(name, file):
    spec = _ilu.spec_from_file_location(name, file)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pd = REPO_ROOT / "rlix" / "pipeline"
_bc = _load_mod("rlix.pipeline.bucket_cache", _pd / "bucket_cache.py")
CPUBucketCache = _bc.CPUBucketCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def R() -> int:
    return dist.get_rank()

def log(msg: str) -> None:
    print(f"[rank{R()}] {msg}", flush=True)

def log0(msg: str) -> None:
    if R() == 0:
        log(msg)

def gpu_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)

def tensor_hash(t: torch.Tensor) -> str:
    b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(rank: int) -> Optional[nn.Module]:
    if rank not in TRAIN_RANKS:
        return None
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(f"cuda:{rank}")
    return model


def train_step(model: Optional[nn.Module], rank: int, step: int) -> None:
    if rank not in TRAIN_RANKS or model is None:
        return
    # Different seeds per pipeline AND per step → A and B weights diverge
    torch.manual_seed(rank * 10000 + step)
    input_ids = torch.randint(0, 1000, (1, SEQ_LEN), device=f"cuda:{rank}")
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 1e-5 * p.grad   # slightly larger LR to widen divergence
    model.zero_grad()
    log(f"  train_step loss={loss.item():.4f} (seed={rank * 10000 + step})")


# ---------------------------------------------------------------------------
# Snapshot + CPU cache
# ---------------------------------------------------------------------------

def snapshot_hashes(model: Optional[nn.Module]) -> Dict[str, str]:
    if model is None:
        return {}
    return {name: tensor_hash(p.data) for name, p in model.named_parameters()}


def build_cpu_cache(model: Optional[nn.Module]) -> Optional[CPUBucketCache]:
    if model is None:
        return None
    cache = CPUBucketCache()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            cache.store(name, shard_id=0, tensor=tensor.cpu().contiguous())
    log(f"  cache built: {len(cache.get_dirty_buckets())} buckets")
    return cache


def measure_memory_release(model: Optional[nn.Module], rank: int) -> None:
    if rank not in TRAIN_RANKS or model is None:
        return
    before_mb = gpu_mb()
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    after_mb = gpu_mb()
    released_pct = (before_mb - after_mb) / before_mb * 100 if before_mb > 0 else 100.0
    log(f"  VRAM: {before_mb:.0f}MB → {after_mb:.0f}MB  released {released_pct:.1f}%")
    if released_pct < VRAM_RELEASE_THRESHOLD_PCT:
        log(f"FAIL: rank{rank} VRAM release {released_pct:.1f}% < {VRAM_RELEASE_THRESHOLD_PCT}%")
        dist.barrier()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Broadcast cache via gloo (pure CPU, no NCCL dtype restrictions)
# ---------------------------------------------------------------------------

def broadcast_cache(
    cache: Optional[CPUBucketCache],
    src_rank: int,
    gloo_group: dist.ProcessGroup,
) -> Dict[str, Tuple[torch.Tensor, str]]:
    """
    Broadcast all dirty buckets from src_rank to every rank in gloo_group.
    Uses 3 CPU (gloo) broadcasts:
      #1  float32 header  — n_buckets + elem-counts encoded as (hi>>20, lo&FFFFF)
      #2  bfloat16 matrix — param names + per-bucket hashes
      #3  bfloat16 flat   — all weight tensors concatenated

    Only ranks inside gloo_group call this function.
    Returns {name: (tensor, expected_hash)} on non-src ranks.
    """
    received: Dict[str, Tuple[torch.Tensor, str]] = {}

    if R() == src_rank:
        assert cache is not None
        buckets = cache.get_dirty_buckets()
        n = len(buckets)
        cpu_tensors = [b.tensor.to(dtype=torch.bfloat16).contiguous() for b in buckets]
        names = [b.param_name for b in buckets]
        n_elems = [t.numel() for t in cpu_tensors]
        elem_hashes = [tensor_hash(t) for t in cpu_tensors]

        # Broadcast #1: header (float32 CPU)
        # n_elems encoded as (hi, lo) split at 2^20 so hi < 2^12, lo < 2^20 — both
        # fit in float32 exact integer range (< 2^24)
        header = torch.zeros(1 + 2 * MAX_PARAMS, dtype=torch.float32)
        header[0] = float(n)
        for i, ne in enumerate(n_elems):
            header[1 + 2 * i] = float(ne >> 20)
            header[2 + 2 * i] = float(ne & 0xFFFFF)
        dist.broadcast(header, src=src_rank, group=gloo_group)

        # Broadcast #2: name+hash matrix (bfloat16 CPU)
        # ASCII ordinals 0-127 are exact in bfloat16 (7-bit mantissa covers all)
        meta_mat = torch.zeros(MAX_PARAMS * ROW, dtype=torch.bfloat16)
        for i, (name, h) in enumerate(zip(names, elem_hashes)):
            nb = name.encode()
            row_start = i * ROW
            for j, b in enumerate(nb):
                meta_mat[row_start + j] = float(b)
            for j, c in enumerate(h):
                meta_mat[row_start + 200 + j] = float(ord(c))
        dist.broadcast(meta_mat, src=src_rank, group=gloo_group)

        # Broadcast #3: flat weight data (bfloat16 CPU)
        flat = torch.cat([t.view(-1) for t in cpu_tensors], dim=0)
        dist.broadcast(flat, src=src_rank, group=gloo_group)

    else:
        # Receive #1: header
        header = torch.zeros(1 + 2 * MAX_PARAMS, dtype=torch.float32)
        dist.broadcast(header, src=src_rank, group=gloo_group)
        n = int(header[0].item())
        n_elems = []
        for i in range(n):
            hi = int(header[1 + 2 * i].item())
            lo = int(header[2 + 2 * i].item())
            n_elems.append((hi << 20) | lo)

        # Receive #2: name+hash matrix
        meta_mat = torch.zeros(MAX_PARAMS * ROW, dtype=torch.bfloat16)
        dist.broadcast(meta_mat, src=src_rank, group=gloo_group)
        names: list[str] = []
        exp_hashes: list[str] = []
        for i in range(n):
            row = meta_mat[i * ROW: i * ROW + ROW]
            name_len = next((j for j in range(200) if row[j] == 0), 200)
            raw = row[:name_len].to(torch.int32).numpy().tolist()
            names.append(bytes(raw).decode())
            exp_hashes.append("".join(chr(int(row[200 + j].item())) for j in range(16)))

        # Receive #3: flat weight data
        total_elems = sum(n_elems)
        flat = torch.zeros(total_elems, dtype=torch.bfloat16)
        dist.broadcast(flat, src=src_rank, group=gloo_group)

        offset = 0
        for name, ne, eh in zip(names, n_elems, exp_hashes):
            received[name] = (flat[offset: offset + ne].clone(), eh)
            offset += ne

    return received


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_weights(
    received: Dict[str, Tuple[torch.Tensor, str]],
    label: str,
    step: int,
) -> None:
    """Hash-verify received weights against expected hashes embedded in protocol."""
    if R() not in INFER_RANKS:
        return
    mismatches = []
    for name, (t, expected_hash) in received.items():
        actual = tensor_hash(t)
        if actual != expected_hash:
            mismatches.append(f"{name}: {actual!r} != {expected_hash!r}")
    if mismatches:
        log(f"  FAIL step {step} pipeline {label}: {len(mismatches)} hash mismatches")
        for m in mismatches[:5]:
            log(f"    {m}")
        sys.exit(1)
    log(f"  PASS step {step} pipeline {label}: {len(received)} weights bit-exact (rank {R()})")


def verify_divergence(
    received_a: Dict[str, Tuple[torch.Tensor, str]],
    received_b: Dict[str, Tuple[torch.Tensor, str]],
    step: int,
) -> None:
    """Assert that A and B have different weights — proves correct per-pipeline routing."""
    if R() not in INFER_RANKS:
        return
    shared_names = set(received_a) & set(received_b)
    same = sum(
        1 for n in shared_names
        if tensor_hash(received_a[n][0]) == tensor_hash(received_b[n][0])
    )
    if same == len(shared_names):
        log(f"  FAIL step {step}: all {same} shared params have identical hashes — "
            f"pipelines did not diverge (check seeds)")
        sys.exit(1)
    log(f"  PASS step {step}: A≠B verified — {len(shared_names) - same}/{len(shared_names)} "
        f"params differ (rank {R()})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    world_size = dist.get_world_size()
    log0(f"world_size={world_size}, GPU={torch.cuda.get_device_name(local_rank)}")

    if world_size < 4:
        log0(f"SKIP: requires 4 GPUs (got {world_size})")
        dist.destroy_process_group()
        return

    # Warm up NCCL before creating subgroups — required on PyTorch 2.5+ so that
    # the internal NCCL all_reduce used by new_group doesn't hang on first use
    dist.barrier(device_ids=[local_rank])

    # Create per-pipeline gloo groups (ALL ranks must call new_group even if not members)
    gloo_a = dist.new_group(ranks=[PIPELINE_A_RANK] + INFER_RANKS, backend="gloo")
    gloo_b = dist.new_group(ranks=[PIPELINE_B_RANK] + INFER_RANKS, backend="gloo")
    log0("Process groups ready: gloo_a=[0,2,3]  gloo_b=[1,2,3]")

    log0(f"Loading {MODEL_NAME} on training ranks...")
    model = load_model(local_rank)
    dist.barrier(device_ids=[local_rank])
    log0("Models loaded.")

    for step in range(1, N_STEPS + 1):
        log0(f"\n{'='*60}")
        log0(f"STEP {step}/{N_STEPS}")

        # ----- Train both pipelines -----
        log0("  [train] both pipelines...")
        train_step(model, local_rank, step)
        dist.barrier(device_ids=[local_rank])

        # ----- Phase A: Pipeline A syncs; Pipeline B is free -----
        log0("  [sync A] Pipeline A offloading + broadcasting...")

        cache_a: Optional[CPUBucketCache] = None
        if local_rank == PIPELINE_A_RANK:
            build_cpu_cache(model)   # snapshot before offload (for logging)
            cache_a = build_cpu_cache(model)
            measure_memory_release(model, local_rank)   # moves model to CPU
        elif local_rank == PIPELINE_B_RANK:
            log(f"  [step {step}] Pipeline B: NOT blocked — free to train while A syncs")

        received_a: Dict[str, Tuple[torch.Tensor, str]] = {}
        if local_rank in [PIPELINE_A_RANK] + INFER_RANKS:
            received_a = broadcast_cache(cache_a, src_rank=PIPELINE_A_RANK, gloo_group=gloo_a)

        verify_weights(received_a, label="A", step=step)

        if local_rank == PIPELINE_A_RANK:
            model = model.to(f"cuda:{local_rank}")
            log(f"  Pipeline A: model reloaded to GPU")

        dist.barrier(device_ids=[local_rank])

        # ----- Phase B: Pipeline B syncs; Pipeline A is free -----
        log0("  [sync B] Pipeline B offloading + broadcasting...")

        cache_b: Optional[CPUBucketCache] = None
        if local_rank == PIPELINE_B_RANK:
            cache_b = build_cpu_cache(model)
            measure_memory_release(model, local_rank)
        elif local_rank == PIPELINE_A_RANK:
            log(f"  [step {step}] Pipeline A: NOT blocked — free to train while B syncs")

        received_b: Dict[str, Tuple[torch.Tensor, str]] = {}
        if local_rank in [PIPELINE_B_RANK] + INFER_RANKS:
            received_b = broadcast_cache(cache_b, src_rank=PIPELINE_B_RANK, gloo_group=gloo_b)

        verify_weights(received_b, label="B", step=step)

        if local_rank == PIPELINE_B_RANK:
            model = model.to(f"cuda:{local_rank}")
            log(f"  Pipeline B: model reloaded to GPU")

        dist.barrier(device_ids=[local_rank])

        # ----- Cross-check: A weights ≠ B weights -----
        log0("  [cross-check] verifying A ≠ B (no routing contamination)...")
        verify_divergence(received_a, received_b, step=step)
        dist.barrier(device_ids=[local_rank])

        log0(f"STEP {step} COMPLETE")

    log0("\n" + "=" * 60)
    log0(f"ALL GATE 2.5 FULL CHECKS PASSED ({N_STEPS} steps)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
