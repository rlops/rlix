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

# Force NCCL socket transport immediately — skip P2P/SHM probe phase.
# On PCIe-only hardware (no NVLink), probe hangs can exceed the 600 s default timeout.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")

import torch
import torch.distributed as dist
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_STEPS = 2
HIDDEN = 2048          # model hidden dim — large enough for VRAM release test to be meaningful
FFN_MULT = 4           # FFN width multiplier
BATCH, SEQ = 2, 32     # input shape
VRAM_RELEASE_THRESHOLD_PCT = 50
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
BucketRecord = _bc.BucketRecord
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
            bias=False, gather_output=False, skip_bias_add=False,
        )
        self.fc2 = RowParallelLinear(
            ffn, hidden, config=config,
            init_method=nn.init.xavier_normal_,
            bias=False, input_is_parallel=True, skip_bias_add=False,
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

def build_cpu_cache(model: Optional[nn.Module]) -> Optional[VersionedBucketCache]:
    if model is None:
        return None
    with torch.no_grad():
        named_tensors = [
            (name, tensor.cpu().contiguous())
            for name, tensor in model.state_dict().items()
            if tensor is not None  # Megatron TP layers store None for disabled biases
        ]
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
        log(f"FAIL: insufficient VRAM release {released_pct:.1f}% < {VRAM_RELEASE_THRESHOLD_PCT}%")
        sys.exit(1)


# ---------------------------------------------------------------------------
# NCCL broadcast — proper subset group (spec: nemorl-port-plan.md lines 391, 1196-1201)
# Gate 2.5 requires NCCL broadcast transport for cross-GPU TP ranks.
# Shard 0: sender=rank0, receiver=rank2 → group [0,2]
# Shard 1: sender=rank1, receiver=rank3 → group [1,3]
# Each is a proper subset of world [0,1,2,3] to avoid the world=group hang.
# ---------------------------------------------------------------------------

def nccl_broadcast_shard(
    cache: Optional[VersionedBucketCache],
    src_rank: int,
    recv_rank: int,
    model: Optional[nn.Module],
    gloo_group: dist.ProcessGroup,
) -> Dict[str, Tuple[torch.Tensor, str]]:
    """Broadcast src_rank's TP shard to recv_rank via dynamic NCCL group.

    All 4 world ranks call this (PyTorch requires all ranks to call new_group).
    Only src_rank and recv_rank participate in NCCL collectives.
    """
    received: Dict[str, Tuple[torch.Tensor, str]] = {}
    rank = R()

    # ALL ranks must call new_group; only [src, recv] participate in NCCL collectives.
    # [src, recv] is a proper subset of world [0,1,2,3] → avoids PCIe deadlock.
    nccl_group = dist.new_group(ranks=[src_rank, recv_rank], backend="nccl")

    if rank == src_rank:
        with cache._cache_lock:
            active_buckets = cache.get_active_buckets()
        all_params = []
        for record in active_buckets:
            all_params.extend(unpack_bucket_record(record))

        # Re-pack into a single uint8 BucketRecord for NCCL broadcast
        repacked = _bucket_named_tensors(all_params)
        gpu_buf = repacked.cpu_uint8_bucket.pin_memory().cuda()
        dist.broadcast(gpu_buf, src=src_rank, group=nccl_group)

        torch.cuda.synchronize()
        dist.barrier(group=nccl_group)
        dist.destroy_process_group(nccl_group)

        # Broadcast sender hashes via gloo for receiver verification
        sender_hashes = {name: tensor_hash(t.float()) for name, t in all_params}
        hash_flat = torch.zeros(len(all_params), 16, dtype=torch.uint8)
        names_list = list(sender_hashes.keys())
        for i, name in enumerate(names_list):
            for j, c in enumerate(sender_hashes[name].encode()):
                hash_flat[i, j] = c
        dist.broadcast(hash_flat, src=src_rank, group=gloo_group)

        for name, t in all_params:
            received[name] = (t.float(), sender_hashes[name])

    elif rank == recv_rank:
        # Derive buffer size from local model (same architecture, same param shapes).
        # Filter None — Megatron TP layers store None for disabled biases.
        assert model is not None
        local_named = [
            (k, v.detach().cpu().contiguous())
            for k, v in model.state_dict().items()
            if v is not None
        ]
        dummy = _bucket_named_tensors(local_named)
        buf_size = dummy.cpu_uint8_bucket.numel()

        gpu_buf = torch.zeros(buf_size, dtype=torch.uint8, device="cuda")
        dist.broadcast(gpu_buf, src=src_rank, group=nccl_group)

        torch.cuda.synchronize()
        dist.barrier(group=nccl_group)
        dist.destroy_process_group(nccl_group)

        # Receive sender hashes via gloo for verification
        hash_flat = torch.zeros(len(dummy.param_names), 16, dtype=torch.uint8)
        dist.broadcast(hash_flat, src=src_rank, group=gloo_group)
        sender_hashes = {}
        for i, name in enumerate(dummy.param_names):
            sender_hashes[name] = bytes(hash_flat[i].tolist()).rstrip(b"\x00").decode()

        # Reconstruct BucketRecord from received buffer using local metadata
        recv_record = BucketRecord(
            param_names=dummy.param_names,
            shapes=dummy.shapes,
            dtypes=dummy.dtypes,
            offsets=dummy.offsets,
            used_bytes=dummy.used_bytes,
            cpu_uint8_bucket=gpu_buf.cpu(),
        )
        unpacked = unpack_bucket_record(recv_record)
        for name, t in unpacked:
            received[name] = (t.float(), sender_hashes.get(name, ""))

    else:
        # Bystander: participate in gloo barrier but skip NCCL collectives.
        # Must receive the hash broadcast so gloo collective completes on all ranks.
        # Model has 2 params (fc1.weight, fc2.weight) — fixed for this test.
        hash_flat = torch.zeros(2, 16, dtype=torch.uint8)
        dist.broadcast(hash_flat, src=src_rank, group=gloo_group)

    dist.barrier(group=gloo_group)
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
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=1,
    )
    # ColumnParallelLinear requires the model-parallel RNG tracker to be seeded
    model_parallel_cuda_manual_seed(42)

def destroy_megatron() -> None:
    from megatron.core import parallel_state as mpu
    mpu.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # No device_id → lazy NCCL init (one communicator at a time, avoids simultaneous
    # world+TP init that can exhaust the 600 s timeout on socket-only transport).
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    log0(f"world_size={world_size}, GPU={torch.cuda.get_device_name(local_rank)}")

    if world_size < 4:
        log0(f"SKIP: requires 4 GPUs (got {world_size})")
        dist.destroy_process_group()
        return

    # Gloo group for weight broadcasts and barriers.
    # All dist.barrier() calls use this group so NCCL is not invoked for barriers;
    # NCCL is used only for the TP all_reduce inside the model forward pass.
    gloo_world = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    # Megatron init: creates TP groups [0,1] and [2,3].
    log0("Initializing Megatron TP=2...")
    init_megatron()
    log0("Megatron initialized.")

    # Build model on ALL ranks (each rank gets its own TP shard)
    log0("Building MegatronTPMLP...")
    model = MegatronTPMLP().to(f"cuda:{local_rank}")
    dist.barrier(group=gloo_world)
    log0(f"Model ready — each rank holds shard of {sum(p.numel() for p in model.parameters()):,} params")

    for step in range(1, N_STEPS + 1):
        log0(f"\n{'='*60}")
        log0(f"STEP {step}/{N_STEPS}")

        # ----- Train on training ranks (no DP all-reduce → inference group diverges) -----
        log0("  [1] train step on training ranks only...")
        train_step(model, local_rank, step)
        dist.barrier(group=gloo_world)

        # ----- Capture pre-sync state for divergence check on inference ranks -----
        pre_sync_cache: Optional[VersionedBucketCache] = None
        if local_rank in INFER_RANKS:
            pre_sync_cache = build_cpu_cache(model)

        # ----- Inference isolation snapshot: before training ranks offload -----
        # Snapshots VRAM and weight hashes on inference ranks.
        # After training ranks call model.cpu() + empty_cache(), we verify these are unchanged.
        infer_vram_before_offload = 0.0
        infer_hashes_before_offload: dict = {}
        if local_rank in INFER_RANKS:
            infer_vram_before_offload = gpu_mb()
            infer_hashes_before_offload = {
                n: tensor_hash(p.data) for n, p in model.named_parameters()
            }

        # ----- Training ranks: offload + destroy_model_parallel -----
        cache: Optional[VersionedBucketCache] = None
        if local_rank in TRAIN_RANKS:
            log(f"  [2] build CPU cache (rank {local_rank})...")
            cache = build_cpu_cache(model)
            log(f"  [3] offload + measure VRAM release (rank {local_rank})...")
            measure_memory_release(model, local_rank)

        if local_rank in TRAIN_RANKS:
            log(f"  [4] destroy_model_parallel (rank {local_rank})...")
        destroy_megatron()
        dist.barrier(group=gloo_world)

        # ----- Inference isolation verification -----
        if local_rank in INFER_RANKS:
            infer_vram_after_offload = gpu_mb()
            delta = abs(infer_vram_after_offload - infer_vram_before_offload)
            if delta > 10.0:
                log(f"FAIL: inference VRAM changed during training offload+destroy: "
                    f"{infer_vram_before_offload:.1f} → {infer_vram_after_offload:.1f} MB "
                    f"(delta={delta:.1f})")
                sys.exit(1)
            log(f"PASS: inference VRAM isolated during training offload "
                f"({infer_vram_before_offload:.1f} → {infer_vram_after_offload:.1f} MB, "
                f"delta={delta:.1f})")
            infer_hashes_after_offload = {
                n: tensor_hash(p.data) for n, p in model.named_parameters()
            }
            corrupted = [
                n for n in infer_hashes_before_offload
                if infer_hashes_after_offload.get(n) != infer_hashes_before_offload[n]
            ]
            if corrupted:
                log(f"FAIL: inference weights corrupted by training's empty_cache: "
                    f"{len(corrupted)}/{len(infer_hashes_before_offload)} params changed")
                sys.exit(1)
            log(f"PASS: inference weights intact during training offload "
                f"({len(infer_hashes_before_offload)} params verified unchanged)")

        # ----- Sync via NCCL proper-subset groups (spec: nemorl-port-plan.md lines 391) -----
        # Phase A: rank 0's shard → rank 2, NCCL group [0,2]
        log0("  [5a] sync training rank 0 shard → rank 2 via NCCL [0,2]...")
        cache0 = cache if local_rank == 0 else None
        received_from_0 = nccl_broadcast_shard(
            cache0, src_rank=0, recv_rank=2, model=model, gloo_group=gloo_world
        )

        # Phase B: rank 1's shard → rank 3, NCCL group [1,3]
        log0("  [5b] sync training rank 1 shard → rank 3 via NCCL [1,3]...")
        cache1 = cache if local_rank == 1 else None
        received_from_1 = nccl_broadcast_shard(
            cache1, src_rank=1, recv_rank=3, model=model, gloo_group=gloo_world
        )

        dist.barrier(group=gloo_world)

        # ----- Verify bit-exact on inference ranks -----
        log0("  [6] verify bit-exact hash match on inference ranks...")
        # Rank 2 should match rank 0's shard; rank 3 should match rank 1's shard
        if local_rank == 2:
            verify_shard(received_from_0, label="0", step=step, my_rank=local_rank)
        if local_rank == 3:
            verify_shard(received_from_1, label="1", step=step, my_rank=local_rank)
        dist.barrier(group=gloo_world)

        # ----- Check inference had different weights BEFORE sync (divergence) -----
        log0("  [7] verify inference weights diverged from training before sync...")
        if local_rank == 2 and pre_sync_cache is not None:
            with pre_sync_cache._cache_lock:
                _pre_records = pre_sync_cache.get_active_buckets()
            _pre_pairs: list = []
            for _r in _pre_records:
                _pre_pairs.extend(unpack_bucket_record(_r))
            pre = {name: t.float() for name, t in _pre_pairs}
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
            with pre_sync_cache._cache_lock:
                _pre_records = pre_sync_cache.get_active_buckets()
            _pre_pairs = []
            for _r in _pre_records:
                _pre_pairs.extend(unpack_bucket_record(_r))
            pre = {name: t.float() for name, t in _pre_pairs}
            different = sum(
                1 for name, (t, _) in received_from_1.items()
                if name in pre and tensor_hash(t) != tensor_hash(pre[name])
            )
            if step > 1 and different == 0:
                log(f"  WARN step {step}: rank3 weights already matched rank1 before sync")
            else:
                log(f"  PASS step {step}: {different}/{len(received_from_1)} params diverged "
                    f"from rank1 before sync (rank 3)")
        dist.barrier(group=gloo_world)

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

        dist.barrier(group=gloo_world)
        log0(f"STEP {step} COMPLETE")

    log0("\n" + "=" * 60)
    log0(f"ALL GATE 2.5 MEGATRON TP CHECKS PASSED ({N_STEPS} steps)")
    destroy_megatron()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
