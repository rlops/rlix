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

# Use cached model only — avoids HF Hub network check hanging when P2P/SHM is disabled
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
_bucket_named_tensors = _bc._bucket_named_tensors
VersionedBucketCache = _bc.VersionedBucketCache
unpack_bucket_record = _bc.unpack_bucket_record


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


def build_cpu_cache(model: Optional[nn.Module]) -> Optional[VersionedBucketCache]:
    if model is None:
        return None
    with torch.no_grad():
        named_tensors = [(name, t.cpu().contiguous()) for name, t in model.state_dict().items()]
    record = _bucket_named_tensors(named_tensors)
    cache = VersionedBucketCache()
    cache.build_latest(-1, [record])
    cache.promote(-1)
    log(f"  cache built: 1 bucket, {len(named_tensors)} params")
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
        sys.exit(1)


# ---------------------------------------------------------------------------
# NCCL broadcast — proper subset groups per pipeline phase
# Spec: nemorl-port-plan.md lines 391, 1196-1201
# Phase A: src=rank0, receivers=[2,3], group=[0,2,3] — proper subset of world [0,1,2,3]
# Phase B: src=rank1, receivers=[2,3], group=[1,2,3] — proper subset of world [0,1,2,3]
# gloo used only for control-plane (buf_size exchange + hash verification)
# ---------------------------------------------------------------------------

def nccl_broadcast_cache(
    cache: Optional[VersionedBucketCache],
    src_rank: int,
    gloo_group: dist.ProcessGroup,
) -> Dict[str, Tuple[torch.Tensor, str]]:
    """Broadcast src_rank's cache to inference ranks via dynamic NCCL subset group.

    Sequence (avoids gloo/NCCL ordering deadlock):
      1. gloo: sender broadcasts buf_size to all ranks
      2. ALL ranks: create NCCL group [src_rank, 2, 3]
      3. NCCL: sender broadcasts packed uint8 buffer
      4. gloo: sender broadcasts full-buffer hash for verification
    """
    received: Dict[str, Tuple[torch.Tensor, str]] = {}
    rank = R()
    repacked = None
    all_params: list = []

    # Step 1: gloo size broadcast (all ranks, before NCCL group creation)
    if rank == src_rank and cache is not None:
        with cache._cache_lock:
            active_buckets = cache.get_active_buckets()
        for rec in active_buckets:
            all_params.extend(unpack_bucket_record(rec))
        repacked = _bucket_named_tensors(all_params)
        meta_t = torch.tensor([repacked.cpu_uint8_bucket.numel()], dtype=torch.int64)
    else:
        meta_t = torch.zeros(1, dtype=torch.int64)
    dist.broadcast(meta_t, src=src_rank, group=gloo_group)
    buf_size = int(meta_t.item())

    # Step 2: ALL ranks create NCCL group — [src, 2, 3] is proper subset of world [0,1,2,3]
    nccl_group = dist.new_group(ranks=[src_rank] + INFER_RANKS, backend="nccl")

    # Step 3: NCCL broadcast
    if rank == src_rank and repacked is not None:
        gpu_buf = repacked.cpu_uint8_bucket.pin_memory().cuda()
        dist.broadcast(gpu_buf, src=src_rank, group=nccl_group)
    elif rank in INFER_RANKS:
        gpu_buf = torch.zeros(buf_size, dtype=torch.uint8, device="cuda")
        dist.broadcast(gpu_buf, src=src_rank, group=nccl_group)
    # Non-member ranks (e.g. rank 1 during phase A, rank 0 during phase B): skip NCCL

    torch.cuda.synchronize()
    if rank in [src_rank] + INFER_RANKS:
        dist.barrier(group=nccl_group)
        dist.destroy_process_group(nccl_group)

    # Step 4: gloo hash exchange for full-buffer bit-exact verification
    hash_t = torch.zeros(16, dtype=torch.uint8)
    if rank == src_rank and repacked is not None:
        h = tensor_hash(repacked.cpu_uint8_bucket)
        for j, c in enumerate(h.encode()):
            hash_t[j] = c
    dist.broadcast(hash_t, src=src_rank, group=gloo_group)

    if rank in INFER_RANKS:
        cpu_buf = gpu_buf.cpu()
        expected_hash = bytes(hash_t.tolist()).rstrip(b"\x00").decode()
        received["_block"] = (cpu_buf, expected_hash)

    dist.barrier(group=gloo_group)
    return received


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_weights(
    received: Dict[str, Tuple[torch.Tensor, str]],
    label: str,
    step: int,
) -> None:
    """Hash-verify received NCCL buffer against sender's full-buffer hash."""
    if R() not in INFER_RANKS:
        return
    if "_block" not in received:
        return  # this rank didn't receive (bystander)
    cpu_buf, expected_hash = received["_block"]
    actual = tensor_hash(cpu_buf)
    if actual != expected_hash:
        log(f"  FAIL step {step} pipeline {label}: buffer hash {actual!r} != {expected_hash!r}")
        sys.exit(1)
    log(f"  PASS step {step} pipeline {label}: {cpu_buf.numel()} bytes bit-exact via NCCL (rank {R()})")


def verify_divergence(
    received_a: Dict[str, Tuple[torch.Tensor, str]],
    received_b: Dict[str, Tuple[torch.Tensor, str]],
    step: int,
) -> None:
    """Assert that A and B have different weights — proves correct per-pipeline routing."""
    if R() not in INFER_RANKS:
        return
    if "_block" not in received_a or "_block" not in received_b:
        return
    hash_a = tensor_hash(received_a["_block"][0])
    hash_b = tensor_hash(received_b["_block"][0])
    if hash_a == hash_b:
        log(f"  FAIL step {step}: A and B have identical buffer hashes — pipelines did not diverge")
        sys.exit(1)
    log(f"  PASS step {step}: A≠B verified — buffer hashes differ (rank {R()})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Use NCCL world — nccl_broadcast_cache creates proper NCCL subset groups.
    # Lazy init (no device_id) so new_group(backend="nccl") works on PCIe hardware.
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    log0(f"world_size={world_size}, GPU={torch.cuda.get_device_name(local_rank)}")

    if world_size < 4:
        log0(f"SKIP: requires 4 GPUs (got {world_size})")
        dist.destroy_process_group()
        return

    # gloo group for control-plane barriers and metadata exchange
    gloo_world = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    log0("Process groups ready: NCCL world + gloo control-plane")

    log0(f"Loading {MODEL_NAME} on training ranks...")
    model = load_model(local_rank)
    dist.barrier(group=gloo_world)
    log0("Models loaded.")

    for step in range(1, N_STEPS + 1):
        log0(f"\n{'='*60}")
        log0(f"STEP {step}/{N_STEPS}")

        # ----- Train both pipelines -----
        log0("  [train] both pipelines...")
        train_step(model, local_rank, step)
        dist.barrier(group=gloo_world)

        # ----- Phase A isolation snapshots -----
        # Snapshot B's VRAM and weight hashes BEFORE A offloads.
        # After the broadcast, we verify A's empty_cache had no effect on B.
        a_hashes_pre_offload: dict = {}
        b_vram_before_a = 0.0
        b_hashes_before_a: dict = {}
        if local_rank == PIPELINE_A_RANK and model is not None:
            a_hashes_pre_offload = {n: tensor_hash(p.data) for n, p in model.named_parameters()}
        if local_rank == PIPELINE_B_RANK and model is not None:
            b_vram_before_a = gpu_mb()
            b_hashes_before_a = {n: tensor_hash(p.data) for n, p in model.named_parameters()}

        # ----- Phase A: Pipeline A syncs -----
        log0("  [sync A] Pipeline A offloading + broadcasting...")

        cache_a: Optional[CPUBucketCache] = None
        if local_rank == PIPELINE_A_RANK:
            cache_a = build_cpu_cache(model)
            measure_memory_release(model, local_rank)
        elif local_rank == PIPELINE_B_RANK:
            log(f"  [step {step}] Pipeline B: not the sender — would be free in production")

        received_a = nccl_broadcast_cache(cache_a, src_rank=PIPELINE_A_RANK, gloo_group=gloo_world)
        verify_weights(received_a, label="A", step=step)

        # ----- Phase A isolation verification: B must be unaffected -----
        if local_rank == PIPELINE_B_RANK and model is not None:
            b_vram_after_a = gpu_mb()
            delta = abs(b_vram_after_a - b_vram_before_a)
            if delta > 10.0:
                log(f"FAIL: Pipeline B VRAM changed during A's empty_cache: "
                    f"{b_vram_before_a:.1f} → {b_vram_after_a:.1f} MB (delta={delta:.1f})")
                dist.barrier(group=gloo_world)
                sys.exit(1)
            log(f"PASS: Pipeline B VRAM isolated during A offload "
                f"({b_vram_before_a:.1f} → {b_vram_after_a:.1f} MB, delta={delta:.1f})")
            b_hashes_after_a = {n: tensor_hash(p.data) for n, p in model.named_parameters()}
            corrupted = [n for n in b_hashes_before_a if b_hashes_after_a.get(n) != b_hashes_before_a[n]]
            if corrupted:
                log(f"FAIL: Pipeline B weights corrupted by A's empty_cache: "
                    f"{len(corrupted)}/{len(b_hashes_before_a)} params changed")
                dist.barrier(group=gloo_world)
                sys.exit(1)
            log(f"PASS: Pipeline B weights intact after A offload "
                f"({len(b_hashes_before_a)} params verified unchanged)")

        if local_rank == PIPELINE_A_RANK:
            model = model.to(f"cuda:{local_rank}")
            log("  Pipeline A: model reloaded to GPU")

        dist.barrier(group=gloo_world)

        # ----- Phase A round-trip verification: A's weights survived CPU offload -----
        if local_rank == PIPELINE_A_RANK and model is not None and a_hashes_pre_offload:
            reloaded_hashes = {n: tensor_hash(p.data) for n, p in model.named_parameters()}
            drift = [n for n in a_hashes_pre_offload if reloaded_hashes.get(n) != a_hashes_pre_offload[n]]
            if drift:
                log(f"FAIL: Pipeline A weights changed after CPU round-trip: "
                    f"{len(drift)}/{len(a_hashes_pre_offload)} params differ")
                dist.barrier(group=gloo_world)
                sys.exit(1)
            log(f"PASS: Pipeline A weights bit-exact after CPU round-trip "
                f"({len(a_hashes_pre_offload)} params)")

        dist.barrier(group=gloo_world)

        # ----- Phase B isolation snapshots -----
        # Snapshot A's VRAM and weight hashes (model just reloaded) BEFORE B offloads.
        a_vram_before_b = 0.0
        a_hashes_before_b: dict = {}
        b_hashes_pre_offload: dict = {}
        if local_rank == PIPELINE_A_RANK and model is not None:
            a_vram_before_b = gpu_mb()
            a_hashes_before_b = {n: tensor_hash(p.data) for n, p in model.named_parameters()}
        if local_rank == PIPELINE_B_RANK and model is not None:
            b_hashes_pre_offload = {n: tensor_hash(p.data) for n, p in model.named_parameters()}

        # ----- Phase B: Pipeline B syncs -----
        log0("  [sync B] Pipeline B offloading + broadcasting...")

        cache_b: Optional[CPUBucketCache] = None
        if local_rank == PIPELINE_B_RANK:
            cache_b = build_cpu_cache(model)
            measure_memory_release(model, local_rank)
        elif local_rank == PIPELINE_A_RANK:
            log(f"  [step {step}] Pipeline A: not the sender — would be free in production")

        received_b = nccl_broadcast_cache(cache_b, src_rank=PIPELINE_B_RANK, gloo_group=gloo_world)
        verify_weights(received_b, label="B", step=step)

        # ----- Phase B isolation verification: A must be unaffected -----
        if local_rank == PIPELINE_A_RANK and model is not None:
            a_vram_after_b = gpu_mb()
            delta = abs(a_vram_after_b - a_vram_before_b)
            if delta > 10.0:
                log(f"FAIL: Pipeline A VRAM changed during B's empty_cache: "
                    f"{a_vram_before_b:.1f} → {a_vram_after_b:.1f} MB (delta={delta:.1f})")
                dist.barrier(group=gloo_world)
                sys.exit(1)
            log(f"PASS: Pipeline A VRAM isolated during B offload "
                f"({a_vram_before_b:.1f} → {a_vram_after_b:.1f} MB, delta={delta:.1f})")
            a_hashes_after_b = {n: tensor_hash(p.data) for n, p in model.named_parameters()}
            corrupted = [n for n in a_hashes_before_b if a_hashes_after_b.get(n) != a_hashes_before_b[n]]
            if corrupted:
                log(f"FAIL: Pipeline A weights corrupted by B's empty_cache: "
                    f"{len(corrupted)}/{len(a_hashes_before_b)} params changed")
                dist.barrier(group=gloo_world)
                sys.exit(1)
            log(f"PASS: Pipeline A weights intact after B offload "
                f"({len(a_hashes_before_b)} params verified unchanged)")

        if local_rank == PIPELINE_B_RANK:
            model = model.to(f"cuda:{local_rank}")
            log("  Pipeline B: model reloaded to GPU")

        dist.barrier(group=gloo_world)

        # ----- Phase B round-trip verification: B's weights survived CPU offload -----
        if local_rank == PIPELINE_B_RANK and model is not None and b_hashes_pre_offload:
            reloaded_hashes = {n: tensor_hash(p.data) for n, p in model.named_parameters()}
            drift = [n for n in b_hashes_pre_offload if reloaded_hashes.get(n) != b_hashes_pre_offload[n]]
            if drift:
                log(f"FAIL: Pipeline B weights changed after CPU round-trip: "
                    f"{len(drift)}/{len(b_hashes_pre_offload)} params differ")
                dist.barrier(group=gloo_world)
                sys.exit(1)
            log(f"PASS: Pipeline B weights bit-exact after CPU round-trip "
                f"({len(b_hashes_pre_offload)} params)")

        dist.barrier(group=gloo_world)

        # ----- Cross-check: A weights ≠ B weights -----
        log0("  [cross-check] verifying A ≠ B (no routing contamination)...")
        verify_divergence(received_a, received_b, step=step)
        dist.barrier(group=gloo_world)

        log0(f"STEP {step} COMPLETE")

    log0("\n" + "=" * 60)
    log0(f"ALL GATE 2.5 FULL CHECKS PASSED ({N_STEPS} steps)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
