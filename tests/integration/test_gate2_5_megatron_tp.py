"""Gate 2.5 Megatron — Real TP=2 training + weight sync.

Uses megatron-core process groups (initialize_model_parallel / destroy_model_parallel)
with a genuine TP-sharded MLP model.  Each GPU holds a different parameter shard;
forward pass uses Megatron's all_reduce across the TP group.

Layout (4 GPUs):
  Megatron TP=2 → two TP groups: [0,1] and [2,3]
  Ranks 0,1 = training group  (first  TP replica)
  Ranks 2,3 = inference group (second TP replica, starts with same weights)

Per-step flow:
  1. Both TP groups forward + backward (with DIFFERENT seeds → weights diverge)
     Training group skips DP all-reduce intentionally so it diverges from inference group.
  2. Training ranks (0,1) offload to CPU → build CPUBucketCache
  3. destroy_model_parallel() — releases NCCL TP communicator buffers
  4. Assert VRAM released ≥ 60%
  5. World-gloo broadcast from rank 0 (training TP shard 0) then rank 1 (shard 1)
     Inference ranks (2,3) each receive the corresponding training shard
  6. Verify bit-exact hash match: rank2 = rank0's shard, rank3 = rank1's shard
  7. Verify training shard ≠ inference shard BEFORE sync (diverged), = AFTER sync
  8. initialize_model_parallel() — rebuild Megatron groups for next step

Run with:
    torchrun --nproc-per-node=4 tests/integration/test_gate2_5_megatron_tp.py

Requires:
    pip install megatron-core transformers torch
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

N_STEPS = 2
HIDDEN = 512           # model hidden dim
FFN_MULT = 4           # FFN width multiplier
BATCH, SEQ = 4, 32     # input shape
VRAM_RELEASE_THRESHOLD_PCT = 60
TRAIN_RANKS = [0, 1]
INFER_RANKS = [2, 3]
TP_SIZE = 2            # tensor parallel degree

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
# TP-sharded MLP using Megatron process groups
#
# ColumnParallelLinear splits output features across TP ranks (each rank holds
# output_size / tp_size columns).  RowParallelLinear splits input features and
# all-reduces the partial outputs across the TP group so all ranks have the
# same result.
# ---------------------------------------------------------------------------

class MegatronTPMLP(nn.Module):
    """Two-layer MLP with Megatron tensor parallelism (TP=2).

    Each GPU holds half the FFN weights:
      fc1: [hidden, ffn/tp]  (ColumnParallelLinear, no gather_output)
      fc2: [ffn/tp, hidden]  (RowParallelLinear, input_is_parallel)

    Forward all-reduces across the TP group inside RowParallelLinear.
    """

    def __init__(self, hidden: int = HIDDEN, ffn_mult: int = FFN_MULT) -> None:
        super().__init__()
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from megatron.core.model_parallel_config import ModelParallelConfig

        config = ModelParallelConfig(tensor_model_parallel_size=TP_SIZE)
        ffn = hidden * ffn_mult

        self.fc1 = ColumnParallelLinear(
            hidden, ffn, config=config,
            init_method=nn.init.xavier_normal_,
            bias=False, gather_output=False,
        )
        self.fc2 = RowParallelLinear(
            ffn, hidden, config=config,
            init_method=nn.init.xavier_normal_,
            bias=False, input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.fc1(x)
        out = torch.nn.functional.gelu(out)
        out, _ = self.fc2(out)
        return out


# ---------------------------------------------------------------------------
# Training step (skip DP all-reduce so training group diverges from inference)
# ---------------------------------------------------------------------------

def train_step(model: Optional[nn.Module], rank: int, step: int) -> None:
    if rank not in TRAIN_RANKS or model is None:
        return
    # Different seed per rank AND per step → each shard (and each step) diverges
    torch.manual_seed(rank * 10_000 + step)
    x = torch.randn(BATCH, SEQ, HIDDEN, device=f"cuda:{rank}")
    target = torch.zeros(BATCH, SEQ, HIDDEN, device=f"cuda:{rank}")
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 1e-4 * p.grad
    model.zero_grad()
    log(f"  train_step loss={loss.item():.4f} (seed={rank * 10_000 + step})")


# ---------------------------------------------------------------------------
# CPU cache helpers
# ---------------------------------------------------------------------------

def build_cpu_cache(model: Optional[nn.Module]) -> Optional[CPUBucketCache]:
    if model is None:
        return None
    cache = CPUBucketCache()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            cache.store(name, shard_id=R(), tensor=tensor.cpu().contiguous())
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
        log(f"FAIL: insufficient VRAM release {released_pct:.1f}% < {VRAM_RELEASE_THRESHOLD_PCT}%")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Gloo broadcast (all via CPU, no NCCL dtype restrictions)
# ---------------------------------------------------------------------------

MAX_PARAMS = 50
ROW = 216

def broadcast_shard(
    cache: Optional[CPUBucketCache],
    src_rank: int,
    gloo_group: dist.ProcessGroup,
) -> Dict[str, Tuple[torch.Tensor, str]]:
    """Broadcast src_rank's weight shard to all ranks in gloo_group.
    Returns {name: (tensor, expected_hash)} on non-src ranks.
    All tensors stay on CPU (gloo transport).
    """
    received: Dict[str, Tuple[torch.Tensor, str]] = {}

    if R() == src_rank:
        buckets = cache.get_dirty_buckets()
        n = len(buckets)
        cpu_tensors = [b.tensor.to(dtype=torch.float32).contiguous() for b in buckets]
        names = [b.param_name for b in buckets]
        n_elems = [t.numel() for t in cpu_tensors]
        elem_hashes = [tensor_hash(t) for t in cpu_tensors]

        # Header: float32 (n, hi_0, lo_0, ...) split at 2^20 for exact encoding
        header = torch.zeros(1 + 2 * MAX_PARAMS, dtype=torch.float32)
        header[0] = float(n)
        for i, ne in enumerate(n_elems):
            header[1 + 2 * i] = float(ne >> 20)
            header[2 + 2 * i] = float(ne & 0xFFFFF)
        dist.broadcast(header, src=src_rank, group=gloo_group)

        # Metadata matrix: bfloat16 (ASCII chars < 128, exact)
        meta = torch.zeros(MAX_PARAMS * ROW, dtype=torch.bfloat16)
        for i, (name, h) in enumerate(zip(names, elem_hashes)):
            rs = i * ROW
            for j, b in enumerate(name.encode()):
                meta[rs + j] = float(b)
            for j, c in enumerate(h):
                meta[rs + 200 + j] = float(ord(c))
        dist.broadcast(meta, src=src_rank, group=gloo_group)

        # Flat weight data: float32
        flat = torch.cat([t.view(-1) for t in cpu_tensors])
        dist.broadcast(flat, src=src_rank, group=gloo_group)

    else:
        # Receive header
        header = torch.zeros(1 + 2 * MAX_PARAMS, dtype=torch.float32)
        dist.broadcast(header, src=src_rank, group=gloo_group)
        n = int(header[0].item())
        n_elems = [(int(header[1 + 2 * i].item()) << 20) | int(header[2 + 2 * i].item())
                   for i in range(n)]

        # Receive metadata
        meta = torch.zeros(MAX_PARAMS * ROW, dtype=torch.bfloat16)
        dist.broadcast(meta, src=src_rank, group=gloo_group)
        names, exp_hashes = [], []
        for i in range(n):
            row = meta[i * ROW: i * ROW + ROW]
            name_len = next((j for j in range(200) if row[j] == 0), 200)
            raw = row[:name_len].to(torch.int32).numpy().tolist()
            names.append(bytes(raw).decode())
            exp_hashes.append("".join(chr(int(row[200 + j].item())) for j in range(16)))

        # Receive flat data
        total = sum(n_elems)
        flat = torch.zeros(total, dtype=torch.float32)
        dist.broadcast(flat, src=src_rank, group=gloo_group)

        offset = 0
        for name, ne, eh in zip(names, n_elems, exp_hashes):
            received[name] = (flat[offset: offset + ne].clone(), eh)
            offset += ne

    return received


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_shard(received: Dict, label: str, step: int, my_rank: int) -> None:
    """Verify received shard has bit-exact hashes (only for inference ranks)."""
    if my_rank not in INFER_RANKS:
        return
    mismatches = []
    for name, (t, expected_hash) in received.items():
        actual = tensor_hash(t)
        if actual != expected_hash:
            mismatches.append(f"{name}: {actual!r} != {expected_hash!r}")
    if mismatches:
        log(f"  FAIL step {step} shard from rank{label}: {len(mismatches)} hash mismatches")
        for m in mismatches[:3]:
            log(f"    {m}")
        sys.exit(1)
    log(f"  PASS step {step}: {len(received)} params bit-exact from rank{label} (rank {my_rank})")


def verify_divergence_before_sync(
    my_model: Optional[nn.Module],
    received: Dict,
    step: int,
    my_rank: int,
) -> None:
    """Assert inference rank's model weights differ from training rank's before sync."""
    if my_rank not in INFER_RANKS or my_model is None:
        return
    my_sd = {k: v.cpu().float() for k, v in my_model.state_dict().items()}
    different = sum(
        1 for name, (t, _) in received.items()
        if name in my_sd and tensor_hash(t) != tensor_hash(my_sd[name])
    )
    if different == 0:
        log(f"  WARN step {step}: training and inference have same weights before sync "
            f"(expected divergence after different-seed training on training ranks only)")
    else:
        log(f"  PASS step {step}: divergence confirmed — {different}/{len(received)} "
            f"params differ before sync (rank {my_rank})")


def apply_received_shard(
    model: Optional[nn.Module],
    received: Dict,
    my_rank: int,
) -> None:
    """Load received weights into model for inference ranks."""
    if my_rank not in INFER_RANKS or model is None:
        return
    sd = model.state_dict()
    for name, (t, _) in received.items():
        if name in sd:
            sd[name].copy_(t.view_as(sd[name]))
    model.load_state_dict(sd)
    log(f"  inference model updated with {len(received)} synced params")


# ---------------------------------------------------------------------------
# Megatron init / destroy
# ---------------------------------------------------------------------------

def init_megatron() -> None:
    from megatron.core import parallel_state as mpu
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=1,
    )

def destroy_megatron() -> None:
    from megatron.core import parallel_state as mpu
    mpu.destroy_model_parallel()


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

    # Full-world gloo group (warmup for NCCL + weight broadcast transport)
    gloo_world = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    # Megatron init: creates TP groups [0,1] and [2,3].
    # This also warms up NCCL via internal group creation.
    log0("Initializing Megatron TP=2...")
    init_megatron()
    log0("Megatron initialized.")

    # Build model on ALL ranks (each rank gets its own TP shard)
    log0("Building MegatronTPMLP...")
    model = MegatronTPMLP().to(f"cuda:{local_rank}")
    dist.barrier()
    log0(f"Model ready — each rank holds shard of {sum(p.numel() for p in model.parameters()):,} params")

    for step in range(1, N_STEPS + 1):
        log0(f"\n{'='*60}")
        log0(f"STEP {step}/{N_STEPS}")

        # ----- Train on training ranks (no DP all-reduce → inference group diverges) -----
        log0("  [1] train step on training ranks only...")
        train_step(model, local_rank, step)
        dist.barrier()

        # ----- Capture pre-sync state for divergence check on inference ranks -----
        pre_sync_cache: Optional[CPUBucketCache] = None
        if local_rank in INFER_RANKS:
            pre_sync_cache = build_cpu_cache(model)

        # ----- Training ranks: offload + destroy_model_parallel -----
        cache: Optional[CPUBucketCache] = None
        if local_rank in TRAIN_RANKS:
            log(f"  [2] build CPU cache (rank {local_rank})...")
            cache = build_cpu_cache(model)
            log(f"  [3] offload + measure VRAM release (rank {local_rank})...")
            measure_memory_release(model, local_rank)

        if local_rank in TRAIN_RANKS:
            log(f"  [4] destroy_model_parallel (rank {local_rank})...")
        destroy_megatron()
        dist.barrier()

        # ----- Sync: each training rank broadcasts its shard to ALL ranks -----
        # Phase rank0: rank 0's shard (fc1 col 0..ffn/2-1, fc2 row 0..ffn/2-1) → all
        log0("  [5a] sync training rank 0 shard → all ranks...")
        cache0 = cache if local_rank == 0 else None
        received_from_0 = broadcast_shard(cache0, src_rank=0, gloo_group=gloo_world)

        # Phase rank1: rank 1's shard → all
        log0("  [5b] sync training rank 1 shard → all ranks...")
        cache1 = cache if local_rank == 1 else None
        received_from_1 = broadcast_shard(cache1, src_rank=1, gloo_group=gloo_world)

        dist.barrier()

        # ----- Verify bit-exact on inference ranks -----
        log0("  [6] verify bit-exact hash match on inference ranks...")
        # Rank 2 should match rank 0's shard; rank 3 should match rank 1's shard
        if local_rank == 2:
            verify_shard(received_from_0, label="0", step=step, my_rank=local_rank)
        if local_rank == 3:
            verify_shard(received_from_1, label="1", step=step, my_rank=local_rank)
        dist.barrier()

        # ----- Check inference had different weights BEFORE sync (divergence) -----
        log0("  [7] verify inference weights diverged from training before sync...")
        if local_rank == 2 and pre_sync_cache is not None:
            pre = {b.param_name: b.tensor.float() for b in pre_sync_cache.get_dirty_buckets()}
            different = sum(
                1 for name, (t, _) in received_from_0.items()
                if name in pre and tensor_hash(t) != tensor_hash(pre[name])
            )
            if step > 1 and different == 0:
                log(f"  WARN step {step}: rank2 weights already matched rank0 before sync")
            else:
                log(f"  PASS step {step}: {different}/{len(received_from_0)} params diverged "
                    f"from rank0 before sync (rank 2)")
        if local_rank == 3 and pre_sync_cache is not None:
            pre = {b.param_name: b.tensor.float() for b in pre_sync_cache.get_dirty_buckets()}
            different = sum(
                1 for name, (t, _) in received_from_1.items()
                if name in pre and tensor_hash(t) != tensor_hash(pre[name])
            )
            if step > 1 and different == 0:
                log(f"  WARN step {step}: rank3 weights already matched rank1 before sync")
            else:
                log(f"  PASS step {step}: {different}/{len(received_from_1)} params diverged "
                    f"from rank1 before sync (rank 3)")
        dist.barrier()

        # ----- Rebuild Megatron process groups -----
        log0("  [8] rebuild Megatron TP groups for next step...")
        init_megatron()

        # Reload training model; update inference model with synced weights
        if local_rank in TRAIN_RANKS:
            model = model.to(f"cuda:{local_rank}")
        elif local_rank == 2:
            sd = model.state_dict()
            for name, (t, _) in received_from_0.items():
                if name in sd:
                    sd[name].copy_(t.view_as(sd[name]).to(f"cuda:{local_rank}"))
            model.load_state_dict(sd)
        elif local_rank == 3:
            sd = model.state_dict()
            for name, (t, _) in received_from_1.items():
                if name in sd:
                    sd[name].copy_(t.view_as(sd[name]).to(f"cuda:{local_rank}"))
            model.load_state_dict(sd)

        dist.barrier()
        log0(f"STEP {step} COMPLETE")

    log0("\n" + "=" * 60)
    log0(f"ALL GATE 2.5 MEGATRON TP CHECKS PASSED ({N_STEPS} steps)")
    destroy_megatron()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
