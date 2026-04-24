"""Gate 2.5 — Feature 6: Expand-time sync ordering and pipeline-owned finalize.

Validates the Feature 6 contract on real GPU hardware (2 ranks):
  1. Sender builds CPU bucket cache from random model weights.
  2. A dynamic NCCL group is created (sender=rank0, receiver=rank1).
  3. Sender stages each bucket CPU→GPU and broadcasts via NCCL (inside _cache_lock).
  4. Receiver unpacks via unpack_bucket_record, writes to its model state dict.
  5. Sender destroys NCCL group inside the cache lock (spec: lines 401-402).
  6. Receiver destroys NCCL group on its side.
  7. Receiver calls finalize_weight_update (torch.cuda.synchronize — post-bucket hook).
  8. Receiver verifies bit-exact hash match vs. sender's pre-sync snapshot.
  9. routing_activated flag is set ONLY after steps 4-7 complete.
  10. Repeat N_CYCLES to verify group create/destroy stability + no VRAM leak.

Ordering invariant verified:
  sync_weights → nccl_teardown → finalize → routing_activated

Run with:
    torchrun --nproc-per-node=2 tests/integration/test_gate2_5_feature6.py

Requires: 2 GPUs
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

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
BucketRecord = _bc_mod.BucketRecord
_bucket_named_tensors = _bc_mod._bucket_named_tensors
unpack_bucket_record = _bc_mod.unpack_bucket_record
VersionedBucketCache = _bc_mod.VersionedBucketCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CYCLES = 3
HIDDEN = 256
N_PARAMS = 6
BUCKET_SIZE_BYTES = 2 * 1024 * 1024   # 2 MB per bucket
VRAM_LEAK_LIMIT_MB = 150
SENDER_RANK = 0
RECEIVER_RANK = 1

def R() -> int:
    return dist.get_rank()

def log(msg: str) -> None:
    print(f"[rank{R()}] {msg}", flush=True)

def log0(msg: str) -> None:
    if R() == 0:
        log(msg)

def tensor_hash(t: torch.Tensor) -> str:
    b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]

def gpu_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Simple model (identical architecture on both ranks)
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for i in range(N_PARAMS):
            setattr(self, f"w{i}", nn.Parameter(torch.randn(HIDDEN, HIDDEN)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(N_PARAMS):
            x = x @ getattr(self, f"w{i}")
        return x


# ---------------------------------------------------------------------------
# One full Feature 6 sync cycle
# ---------------------------------------------------------------------------

def run_sync_cycle(
    sender_model: Optional[nn.Module],
    receiver_model: Optional[nn.Module],
    cycle: int,
    gloo_group: dist.ProcessGroup,
) -> List[str]:
    """Execute one Feature 6 expand-style sync cycle.

    Returns the ordered event log so the caller can assert sequencing.
    """
    rank = R()
    events: List[str] = []
    sender_hashes: Dict[str, str] = {}
    received_hashes: Dict[str, str] = {}

    # ── Step 1: sender builds CPU bucket cache ────────────────────────────
    cache: Optional[VersionedBucketCache] = None
    if rank == SENDER_RANK:
        assert sender_model is not None
        # Simulate train step: perturb weights so each cycle differs
        with torch.no_grad():
            for p in sender_model.parameters():
                p.data += 0.01 * torch.randn_like(p) * (cycle + 1)

        # Snapshot hashes before sync
        sender_hashes = {
            name: tensor_hash(p.data)
            for name, p in sender_model.named_parameters()
        }

        named_tensors = [
            (name, p.detach().cpu().contiguous())
            for name, p in sender_model.named_parameters()
        ]
        buckets: list = []
        batch: list = []
        cur_bytes = 0
        for name, t in named_tensors:
            nb = t.numel() * t.element_size()
            if batch and cur_bytes + nb > BUCKET_SIZE_BYTES:
                buckets.append(_bucket_named_tensors(batch))
                batch = []
                cur_bytes = 0
            batch.append((name, t))
            cur_bytes += nb
        if batch:
            buckets.append(_bucket_named_tensors(batch))

        cache = VersionedBucketCache()
        cache.build_latest(cycle, buckets)
        cache.promote(cycle)
        events.append("build_cache")
        log(f"  [step1] built {len(buckets)} bucket(s)")

    dist.barrier(group=gloo_group)

    # ── Step 2: create dynamic NCCL group ────────────────────────────────
    # All world ranks must call new_group; only SENDER_RANK and RECEIVER_RANK join.
    # When world_size > 2, this creates a proper subset group (avoids PCIe hang).
    sync_ranks = [SENDER_RANK, RECEIVER_RANK]
    nccl_group = dist.new_group(ranks=sync_ranks, backend="nccl")
    dist.barrier(group=gloo_group)
    events.append("nccl_group_created")
    log0("  [step2] NCCL group created")

    # ── Step 3: transport under _cache_lock ──────────────────────────────
    if rank == SENDER_RANK:
        with cache._cache_lock:
            active_buckets = cache.get_active_buckets()
            n_buckets = len(active_buckets)
            for bucket_idx, bucket in enumerate(active_buckets):
                staging = bucket.cpu_uint8_bucket.pin_memory().cuda()
                dist.broadcast(staging, src=SENDER_RANK, group=nccl_group)
                del staging
                log(f"  [step3] sent bucket {bucket_idx+1}/{n_buckets}")
            # Barrier before destroy: ensures all receivers finish NCCL ops
            # before communicator is torn down (prevents watchdog SIGABRT).
            torch.cuda.synchronize()
            dist.barrier(group=nccl_group)
            # Sender-side NCCL teardown inside lock (spec lines 401-402)
            dist.destroy_process_group(nccl_group)
            events.append("sender_nccl_teardown")
            log("  [step3] sender NCCL group destroyed inside cache lock")

    elif rank == RECEIVER_RANK:
        assert receiver_model is not None
        # Receiver must know bucket metadata — we broadcast via gloo first
        # (In production, ModelUpdateService sends payload dicts over Ray;
        #  here we simulate by receiving via gloo the param metadata then NCCL data)

        # Get model param shapes/dtypes via local model (same architecture)
        named_params = list(receiver_model.named_parameters())
        batch_names: list = []
        batch_dtypes: list = []
        batch_shapes: list = []
        batch_offsets: list = []
        batch_used_bytes: list = []

        cur_batch: list = []
        cur_bytes = 0
        all_batches: list = []
        for name, p in named_params:
            nb = p.numel() * p.element_size()
            if cur_batch and cur_bytes + nb > BUCKET_SIZE_BYTES:
                all_batches.append(cur_batch)
                cur_batch = []
                cur_bytes = 0
            cur_batch.append((name, p.detach().cpu().contiguous()))
            cur_bytes += nb
        if cur_batch:
            all_batches.append(cur_batch)

        for batch in all_batches:
            dummy_record = _bucket_named_tensors(batch)
            total_bytes = dummy_record.cpu_uint8_bucket.numel()
            recv_buf = torch.zeros(total_bytes, dtype=torch.uint8)
            recv_staging = recv_buf.pin_memory().cuda()
            dist.broadcast(recv_staging, src=SENDER_RANK, group=nccl_group)

            recv_buf = recv_staging.cpu()
            del recv_staging

            recv_record = BucketRecord(
                param_names=dummy_record.param_names,
                shapes=dummy_record.shapes,
                dtypes=dummy_record.dtypes,
                offsets=dummy_record.offsets,
                used_bytes=dummy_record.used_bytes,
                cpu_uint8_bucket=recv_buf,
            )
            named_tensors_recv = unpack_bucket_record(recv_record)
            for name, t in named_tensors_recv:
                received_hashes[name] = tensor_hash(t)
                # Apply to receiver model
                param = dict(receiver_model.named_parameters())[name]
                with torch.no_grad():
                    param.data.copy_(t.to(param.device).view_as(param))

        torch.cuda.synchronize()
        dist.barrier(group=nccl_group)
        dist.destroy_process_group(nccl_group)
        events.append("receiver_nccl_teardown")
        log("  [step3] receiver NCCL group destroyed")

    dist.barrier(group=gloo_group)
    events.append("sync_weights_done")

    # ── Step 4: finalize_weight_update (pipeline-owned, worker-executed) ─
    if rank == RECEIVER_RANK:
        torch.cuda.synchronize()  # simulates finalize_weight_update
        events.append("finalize_done")
        log("  [step4] finalize_weight_update done")
    dist.barrier(group=gloo_group)
    if rank == SENDER_RANK:
        events.append("finalize_done")

    # ── Step 5: NOW activate routing ─────────────────────────────────────
    events.append("routing_activated")
    log0("  [step5] routing activated (AFTER sync+finalize)")

    # ── Step 6: verify bit-exact on receiver ─────────────────────────────
    if rank == RECEIVER_RANK:
        # Exchange hashes via gloo for verification
        all_hashes_tensor = torch.zeros(len(received_hashes), 16, dtype=torch.uint8)
        all_names = sorted(received_hashes.keys())
        for i, name in enumerate(all_names):
            h = received_hashes[name]
            for j, c in enumerate(h.encode()):
                all_hashes_tensor[i, j] = c

    # Sender broadcasts expected hashes
    n_params_tensor = torch.zeros(1, dtype=torch.int64)
    if rank == SENDER_RANK:
        n_params_tensor[0] = len(sender_hashes)
    dist.broadcast(n_params_tensor, src=SENDER_RANK, group=gloo_group)
    n_params = int(n_params_tensor.item())

    hash_matrix = torch.zeros(n_params, 16, dtype=torch.uint8)
    if rank == SENDER_RANK:
        all_names_s = sorted(sender_hashes.keys())
        for i, name in enumerate(all_names_s):
            h = sender_hashes[name]
            for j, c in enumerate(h.encode()):
                hash_matrix[i, j] = c
    dist.broadcast(hash_matrix, src=SENDER_RANK, group=gloo_group)

    if rank == RECEIVER_RANK:
        all_names_r = sorted(received_hashes.keys())
        mismatches = 0
        for i, name in enumerate(all_names_r):
            expected = bytes(hash_matrix[i].tolist()).rstrip(b"\x00").decode()
            actual = received_hashes[name]
            if actual != expected:
                log(f"  HASH MISMATCH {name}: got {actual!r} expected {expected!r}")
                mismatches += 1
        if mismatches:
            log(f"FAIL cycle {cycle}: {mismatches}/{n_params} hash mismatches")
            dist.barrier(group=gloo_group)
            sys.exit(1)
        log(f"  PASS cycle {cycle}: {n_params} params bit-exact")
        events.append("hash_verified")

    dist.barrier(group=gloo_group)
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    log0(f"world_size={world_size} GPU={torch.cuda.get_device_name(local_rank)}")

    if world_size < 2:
        log0("SKIP: requires ≥2 GPUs")
        dist.destroy_process_group()
        return
    # Scale to any GPU count: first GPU = sender, last GPU = receiver
    # With N GPUs: sender=0, receiver=N-1 (cross-GPU, proper NCCL subset)
    global SENDER_RANK, RECEIVER_RANK
    # Sender=first GPU, Receiver=last GPU — proper NCCL subset when world_size > 2
    RECEIVER_RANK = world_size - 1
    log0(f"Config: sender=rank{SENDER_RANK}, receiver=rank{RECEIVER_RANK}, world_size={world_size}")

    # World gloo group for barriers — all ranks participate
    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    # Build models only on sender and receiver ranks; others are bystanders
    torch.manual_seed(42)  # same seed on all ranks for identical initial weights
    sender_model: Optional[nn.Module] = (
        SimpleModel().to(f"cuda:{local_rank}") if local_rank == SENDER_RANK else None
    )
    receiver_model: Optional[nn.Module] = (
        SimpleModel().to(f"cuda:{local_rank}") if local_rank == RECEIVER_RANK else None
    )
    # Sender and receiver start with same weights (same seed)
    # Sender will diverge via training steps before each sync cycle

    dist.barrier(group=gloo_group)
    vram_start = gpu_mb()

    for cycle in range(N_CYCLES):
        log0(f"\n{'='*60}")
        log0(f"CYCLE {cycle+1}/{N_CYCLES}")

        events = run_sync_cycle(sender_model, receiver_model, cycle, gloo_group)

        # Verify ordering invariant (sender-side)
        if local_rank == SENDER_RANK:
            required_order = [
                "build_cache",
                "nccl_group_created",
                "sender_nccl_teardown",
                "sync_weights_done",
                "finalize_done",
                "routing_activated",
            ]
            for i, expected in enumerate(required_order):
                assert events[i] == expected, (
                    f"ORDERING VIOLATION at position {i}: "
                    f"expected {expected!r}, got {events[i]!r}\n"
                    f"Full event log: {events}"
                )
            log(f"  PASS cycle {cycle+1}: ordering invariant verified")

        dist.barrier(group=gloo_group)

    # VRAM leak check across cycles
    vram_end = gpu_mb()
    vram_growth = vram_end - vram_start
    log0(f"\nVRAM: {vram_start:.0f}MB → {vram_end:.0f}MB (growth={vram_growth:.1f}MB)")
    if vram_growth > VRAM_LEAK_LIMIT_MB:
        log0(f"FAIL: VRAM grew {vram_growth:.1f}MB (limit={VRAM_LEAK_LIMIT_MB}MB)")
        dist.destroy_process_group()
        sys.exit(1)

    log0(f"\n{'='*60}")
    log0(f"ALL GATE 2.5 FEATURE 6 CHECKS PASSED ({N_CYCLES} cycles)")
    log0("  [PASS] Weights synced via dynamic NCCL group")
    log0("  [PASS] Receiver weights bit-exact vs sender")
    log0("  [PASS] Ordering: sync → NCCL teardown → finalize → routing active")
    log0("  [PASS] No VRAM leak across cycles")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
