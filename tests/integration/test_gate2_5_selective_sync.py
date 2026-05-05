"""Gate 2.5 — Part 2: Selective sync via dynamic NCCL group (cross-GPU TP).

Validates the CPU-cache → dynamic-NCCL-group → target-rank weight transfer
that ModelUpdateService uses during expand for non-colocated (cross-GPU) targets.

Spec: nemorl-port-plan.md lines 316, 322, 391:
  - tp=2 with cross-GPU TP peers requires the NCCL broadcast path
  - Dynamic NCCL group must be a PROPER SUBSET of the world group
    (world=[0,1,2,3], dynamic=[0,2] or [0,2,3])
  - NCCL CANNOT form a group when sender and receiver share the same GPU
    (that case uses CUDA IPC; not tested here)
  - Gate 2.5 verifies the NCCL broadcast transport lifecycle

Layout (4 GPUs):
  rank 0 = training / cache owner (sender)
  rank 1 = training non-owner (participates in collective, no cache storage)
  rank 2 = inference worker TP rank 0 (receiver)
  rank 3 = inference worker TP rank 1 (receiver)

Flow per cycle:
  1. rank 0 packs weights into BucketRecord(s) (Feature 4 CPU bucket cache).
  2. A dynamic NCCL group is created for [0, 2, 3] (proper subset of world).
     rank 1 calls new_group too but stays outside the collective.
  3. rank 0 stages the packed uint8 bucket CPU→GPU and broadcasts.
  4. ranks 2, 3 receive the buffer, unpack via unpack_bucket_record.
  5. Dynamic group is destroyed on all members.
  6. ranks 2, 3 verify bit-exact match vs. rank 0's ground-truth.
  7. Check VRAM stability across N_SYNC_CYCLES (no leaks).

Note: rank 1 calls dist.new_group on the NCCL world group as required by PyTorch
(all ranks must call new_group), but does NOT participate in the dynamic group's
broadcasts (it is not in sync_ranks).

Run with:
    torchrun --nproc-per-node=4 tests/integration/test_gate2_5_selective_sync.py

Requires: 4 GPUs (NCCL broadcast path needs cross-GPU ranks in a proper subset group)
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

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
N_SYNC_CYCLES = 3
TENSOR_ELEMENTS = 256 * 1024          # ~512 KB per param at bfloat16
N_PARAMS = 6
SEED = 42
VRAM_LEAK_LIMIT_MB = 200

SENDER_RANK = 0
NON_OWNER_RANK = 1
INFER_RANKS = [2, 3]                  # proper subset: ranks 2 and 3 are receivers
SYNC_RANKS = [SENDER_RANK] + INFER_RANKS  # NCCL group: [0, 2, 3]

PARAM_NAMES = [f"layer_{i}.weight" for i in range(N_PARAMS)]


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

def make_weights(step: int = 0) -> Dict[str, torch.Tensor]:
    """Deterministic weights — same on all ranks for ground-truth comparison."""
    torch.manual_seed(SEED + step)
    return {
        name: torch.randn(TENSOR_ELEMENTS, dtype=torch.bfloat16)
        for name in PARAM_NAMES
    }


# ---------------------------------------------------------------------------
# One selective sync cycle via dynamic NCCL group
# ---------------------------------------------------------------------------

def run_cycle(
    cycle: int,
    weights: Dict[str, torch.Tensor],
    infer_sd: Dict[str, torch.Tensor],
    gloo_group: dist.ProcessGroup,
) -> None:
    """
    Feature 6 transport: CPU bucket → dynamic NCCL group → receiver GPU.

    Dynamic group [0, 2, 3] is a proper subset of world [0, 1, 2, 3].
    rank 1 calls new_group (required by PyTorch) but stays outside the collective.
    """
    rank = R()

    # ALL ranks must call new_group (PyTorch requirement).
    # The dynamic group covers [SENDER, INFER_0, INFER_1] = [0, 2, 3].
    # rank 1 is NOT in SYNC_RANKS but must still call new_group.
    dynamic_group = dist.new_group(ranks=SYNC_RANKS, backend="nccl")
    dist.barrier(group=gloo_group)

    if rank == SENDER_RANK:
        # Pack weights into BucketRecord — Feature 4 CPU bucket cache format
        named_tensors = [(name, t.cpu().contiguous()) for name, t in weights.items()]
        record = _bucket_named_tensors(named_tensors)

        # Stage CPU→GPU and broadcast to inference ranks via dynamic NCCL group
        gpu_buf = record.cpu_uint8_bucket.pin_memory().cuda()
        size_tensor = torch.tensor([gpu_buf.numel()], dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=SENDER_RANK, group=dynamic_group)
        dist.broadcast(gpu_buf, src=SENDER_RANK, group=dynamic_group)
        torch.cuda.synchronize()
        del gpu_buf
        log(f"  cycle {cycle}: sent {len(named_tensors)} params in 1 bucket")

    elif rank in INFER_RANKS:
        # Receive buffer size, then the packed uint8 bucket
        size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=SENDER_RANK, group=dynamic_group)
        buf_size = int(size_tensor.item())

        gpu_buf = torch.zeros(buf_size, dtype=torch.uint8, device="cuda")
        dist.broadcast(gpu_buf, src=SENDER_RANK, group=dynamic_group)
        torch.cuda.synchronize()

        # Reconstruct BucketRecord using known metadata (deterministic seed → same on all ranks)
        shapes_list = [weights[n].shape for n in PARAM_NAMES]
        dtypes_list = [weights[n].dtype for n in PARAM_NAMES]
        offsets_list: List[int] = []
        cur = 0
        for n in PARAM_NAMES:
            offsets_list.append(cur)
            ne = 1
            for s in weights[n].shape:
                ne *= s
            nb = ne * torch.empty(0, dtype=weights[n].dtype).element_size()
            cur = (cur + nb + 511) // 512 * 512

        record = BucketRecord(
            param_names=PARAM_NAMES,
            shapes=shapes_list,
            dtypes=dtypes_list,
            offsets=offsets_list,
            used_bytes=buf_size,
            cpu_uint8_bucket=gpu_buf.cpu(),
        )
        unpacked = unpack_bucket_record(record)
        for name, tensor in unpacked:
            if name in infer_sd:
                infer_sd[name].copy_(tensor.to(infer_sd[name].device))
        del gpu_buf
        log(f"  cycle {cycle}: received and applied {len(unpacked)} params")

    # rank 1: not in dynamic group, skips all collectives above
    # Spec: non-sync ranks must not call group collectives (guard is by not including in group)

    # Synchronize before destroying: barrier on the dynamic group ensures ALL
    # receivers have finished their NCCL operations before the communicator is torn down.
    # Without this, rank 0 (sender) can destroy the group while rank 2/3 are still
    # processing the received GPU buffer, causing NCCL watchdog SIGABRT.
    torch.cuda.synchronize()
    if rank in SYNC_RANKS:
        dist.barrier(group=dynamic_group)
        dist.destroy_process_group(dynamic_group)
    dist.barrier(group=gloo_group)
    log0(f"  cycle {cycle}: NCCL group destroyed")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(weights: Dict[str, torch.Tensor], infer_sd: Dict[str, torch.Tensor], cycle: int) -> None:
    """Verify received weights on inference ranks are bit-exact vs. sender's ground truth."""
    if R() not in INFER_RANKS:
        return

    mismatches = []
    for name, expected_cpu in weights.items():
        if name not in infer_sd:
            mismatches.append(f"{name}: missing from infer_sd")
            continue
        actual = infer_sd[name].cpu()
        eh = tensor_hash(expected_cpu)
        ah = tensor_hash(actual)
        if eh != ah:
            mismatches.append(f"{name}: expected {eh!r} got {ah!r}")

    if mismatches:
        log(f"FAIL cycle {cycle}: {len(mismatches)} hash mismatches:")
        for m in mismatches[:3]:
            log(f"  {m}")
        sys.exit(1)

    log(f"  PASS cycle {cycle}: {len(weights)} params bit-exact (rank {R()})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Use lazy NCCL init (no device_id) so dist.new_group(backend="nccl") works
    # with proper subset groups on PCIe-only hardware.
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()

    log(f"world_size={world_size}, GPU={torch.cuda.get_device_name(local_rank)}")

    if world_size < 4:
        log(f"SKIP: requires ≥4 GPUs for proper subset NCCL group test (got {world_size})")
        log("NOTE: dist.new_group([0,1], backend=nccl) when world=[0,1] hangs on PCIe hardware.")
        log("      Need ≥4 GPUs so dynamic group is a proper subset of world group.")
        dist.destroy_process_group()
        return

    # With N GPUs: first half = training ranks, second half = inference ranks
    # Dynamic NCCL group = sender + all inference ranks (proper subset of world)
    half = world_size // 2
    global SENDER_RANK, NON_OWNER_RANK, INFER_RANKS, SYNC_RANKS
    SENDER_RANK = 0
    NON_OWNER_RANK = 1 if half > 1 else None
    INFER_RANKS = list(range(half, world_size))
    SYNC_RANKS = [SENDER_RANK] + INFER_RANKS
    log0(f"Config: training=[0..{half-1}], inference=[{half}..{world_size-1}], sync_group={SYNC_RANKS}")

    gloo_world = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    # Ground-truth weights — deterministic, same on all ranks
    weights = make_weights(step=0)

    # Inference state dict on GPU (receivers 2,3 use; others allocate zeros)
    infer_sd: Dict[str, torch.Tensor] = {}
    if local_rank in INFER_RANKS:
        infer_sd = {
            name: torch.zeros(TENSOR_ELEMENTS, dtype=torch.bfloat16, device="cuda")
            for name in PARAM_NAMES
        }

    before_mb = gpu_mb()
    log0(f"GPU before cycles: {before_mb:.1f} MB")

    for cycle in range(1, N_SYNC_CYCLES + 1):
        log0(f"\n=== cycle {cycle}/{N_SYNC_CYCLES} ===")

        weights = make_weights(step=cycle)
        run_cycle(cycle, weights, infer_sd, gloo_world)
        verify(weights, infer_sd, cycle)
        dist.barrier(group=gloo_world)

        if local_rank in INFER_RANKS:
            for t in infer_sd.values():
                t.zero_()

    after_mb = gpu_mb()
    vram_growth = after_mb - before_mb
    log0(f"\nVRAM: {before_mb:.0f}MB → {after_mb:.0f}MB (growth={vram_growth:.1f}MB)")

    if vram_growth > VRAM_LEAK_LIMIT_MB:
        log0(f"FAIL: VRAM grew {vram_growth:.1f}MB > {VRAM_LEAK_LIMIT_MB}MB limit")
        dist.destroy_process_group()
        sys.exit(1)

    log0(f"PASS: VRAM stable across {N_SYNC_CYCLES} cycles (growth={vram_growth:.1f} MB)")
    dist.barrier(group=gloo_world)
    log(f"ALL PART 2 CHECKS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
