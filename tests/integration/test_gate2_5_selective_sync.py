"""Gate 2.5 — Part 2: Selective sync via dynamic NCCL group.

Validates the CPU-cache → dynamic-NCCL-group → target-rank weight transfer
that ModelUpdateServiceCached uses during expand.

Design (2 GPUs):
  - rank 0 = training worker / cache owner (sender)
  - rank 1 = inference worker (receiver)

Both ranks create identical weights from the same seed so there is no
need to broadcast Python objects over NCCL (which is unreliable for
control-plane messages).

Flow per cycle:
  1. rank 0 builds CPUBucketCache from in-memory weights.
  2. rank 1 has a zeroed "inference" state dict on GPU.
  3. A dynamic NCCL group is created for [0, 1].
  4. rank 0 stages each bucket CPU→GPU and broadcasts it.
  5. rank 1 receives each tensor and writes it to its state dict.
  6. Dynamic group is destroyed.
  7. rank 1 verifies bit-exact match vs. the known ground-truth weights.
  8. Repeat N_SYNC_CYCLES times to test group create/destroy stability.

Run with:
    torchrun --nproc-per-node=2 tests/integration/test_gate2_5_selective_sync.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SYNC_CYCLES = 3
TENSOR_ELEMENTS = 512 * 1024          # ~1 MB per param at bfloat16
N_PARAMS = 8
SEED = 42
VRAM_LEAK_LIMIT_MB = 200              # max acceptable growth across cycles

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
_bc_mod = _load_mod("rlix.pipeline.bucket_cache", _pd / "bucket_cache.py")
CPUBucketCache = _bc_mod.CPUBucketCache

SENDER = 0
RECEIVER = 1
PARAM_NAMES = [f"layer_{i}.weight" for i in range(N_PARAMS)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def R() -> int:
    return dist.get_rank()

def log(msg: str) -> None:
    print(f"[rank{R()}] {msg}", flush=True)

def gpu_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)

def tensor_hash(t: torch.Tensor) -> str:
    b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]

def make_weights(step: int = 0) -> Dict[str, torch.Tensor]:
    """Deterministic weights — same on both ranks for ground-truth comparison."""
    torch.manual_seed(SEED + step)
    return {
        name: torch.randn(TENSOR_ELEMENTS, dtype=torch.bfloat16)
        for name in PARAM_NAMES
    }


# ---------------------------------------------------------------------------
# One selective sync cycle
# ---------------------------------------------------------------------------

def run_cycle(
    cycle: int,
    weights: Dict[str, torch.Tensor],
    infer_sd: Dict[str, torch.Tensor],
) -> None:
    """
    rank 0: build cache, create group, broadcast each bucket CPU→GPU.
    rank 1: create group, receive each broadcast, write to infer_sd.
    All ranks in world must call new_group.
    """
    # Both ranks call new_group — required even if not in the group.
    # Here world_size=2 so [SENDER, RECEIVER] = [0, 1] = all ranks.
    dynamic_group = dist.new_group(ranks=[SENDER, RECEIVER], backend="nccl")

    if R() == SENDER:
        cache = CPUBucketCache()
        for name, tensor in weights.items():
            cache.store(name, shard_id=0, tensor=tensor.contiguous())
        buckets = cache.get_dirty_buckets()

        for bucket in buckets:
            gpu_t = bucket.tensor.cuda().contiguous()
            dist.broadcast(gpu_t, src=SENDER, group=dynamic_group)

    elif R() == RECEIVER:
        for name in PARAM_NAMES:
            buf = torch.zeros(TENSOR_ELEMENTS, dtype=torch.bfloat16, device="cuda")
            dist.broadcast(buf, src=SENDER, group=dynamic_group)
            infer_sd[name].copy_(buf)

    dist.destroy_process_group(dynamic_group)
    dist.barrier()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(
    weights: Dict[str, torch.Tensor],
    infer_sd: Dict[str, torch.Tensor],
    cycle: int,
) -> None:
    """rank 1 only — compare hashes of received vs. ground-truth."""
    if R() != RECEIVER:
        return

    mismatches: List[str] = []
    for name, original in weights.items():
        received = infer_sd[name].cpu()
        if not torch.equal(received, original):
            max_diff = (received.float() - original.float()).abs().max().item()
            h_recv = tensor_hash(received)
            h_orig = tensor_hash(original)
            mismatches.append(
                f"{name}: max_diff={max_diff:.6f} "
                f"hash_recv={h_recv} hash_orig={h_orig}"
            )

    if mismatches:
        log(f"FAIL cycle {cycle}: {len(mismatches)} weight mismatches:")
        for m in mismatches[:5]:
            log(f"  {m}")
        dist.barrier()
        sys.exit(1)
    else:
        total = len(weights)
        log(f"PASS cycle {cycle}: {total}/{total} weights bit-exact")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # device_id required in PyTorch 2.5+ for NCCL barrier to not hang
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    dist.barrier(device_ids=[local_rank])

    world_size = dist.get_world_size()
    log(f"world_size={world_size}, GPU={torch.cuda.get_device_name(local_rank)}")

    if world_size < 2:
        log("SKIP: requires 2 GPUs")
        dist.destroy_process_group()
        return

    # Ground-truth weights — same on both ranks (deterministic seed)
    weights = make_weights(step=0)

    # Inference state dict on GPU (receiver only, but both ranks allocate for simplicity)
    infer_sd = {
        name: torch.zeros(TENSOR_ELEMENTS, dtype=torch.bfloat16, device="cuda")
        for name in PARAM_NAMES
    }

    before_mb = gpu_mb()
    log(f"GPU before cycles: {before_mb:.1f} MB")

    for cycle in range(1, N_SYNC_CYCLES + 1):
        log(f"=== cycle {cycle}/{N_SYNC_CYCLES} ===")

        # Update weights each cycle to simulate a new training step
        weights = make_weights(step=cycle)

        run_cycle(cycle, weights, infer_sd)
        verify(weights, infer_sd, cycle)
        dist.barrier()

        # Reset infer_sd for next cycle
        for t in infer_sd.values():
            t.zero_()

    after_mb = gpu_mb()
    vram_growth = after_mb - before_mb
    log(f"GPU after cycles: {after_mb:.1f} MB, growth={vram_growth:.1f} MB")

    if R() == 0:
        if vram_growth > VRAM_LEAK_LIMIT_MB:
            log(f"FAIL: VRAM grew {vram_growth:.1f} MB > {VRAM_LEAK_LIMIT_MB} MB (leak)")
            sys.exit(1)
        else:
            log(f"PASS: VRAM stable across {N_SYNC_CYCLES} cycles (growth={vram_growth:.1f} MB)")

    dist.barrier()
    log(f"ALL PART 2 CHECKS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
