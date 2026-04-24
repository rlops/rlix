# Task 2 — CPU Bucket Cache + Selective Weight Sync

**Branch**: `task2-bucket-cache` (rlix) · `rlix-task2` / `main` (NeMo submodule)  
**Gate**: 2.5 — all 6 GPU integration tests pass on 4× RTX A5000  
**Spec**: `plans/nemorl-port-plan.md` — Feature 4 (F4) + Feature 6-transport (F6)

---

## What this implements

GPU time-sharing between training and inference workers requires weights to be transferred after each training step without holding GPU memory on both sides simultaneously. Task 2 implements the two core primitives:

| Feature | What it does |
|---------|-------------|
| **F4 — CPU bucket cache** | After each train step, all model weights are packed into CPU-resident `BucketRecord` objects (512-byte-aligned uint8 tensors) and held in a `VersionedBucketCache`. Only the cache owner (pp=0/dp=0/tp=0/cp=0) stores the full model; non-owners drain the collective without storing. |
| **F6 — Selective sync** | `ModelUpdateService` transfers the active cache to specific inference workers: CUDA IPC for same-GPU colocated workers, dynamic NCCL broadcast for cross-GPU. The pipeline owns finalize and version publication. |

---

## Repo layout

```
rlix/                                   ← zhenyulincs/rlix (this repo)
├── rlix/pipeline/
│   ├── bucket_cache.py                 ← BucketRecord, VersionedBucketCache, pack/unpack
│   ├── bucket_cache_lifecycle.py       ← BucketCacheLifecycle (version tracking)
│   ├── model_update_service.py         ← 6-phase sync orchestrator (Ray actor)
│   ├── coordinator.py                  ← sync_base_weights_to_active()
│   └── full_finetune_pipeline.py       ← _expand_workers, finalize, version publish
├── rlix/protocol/coordinator.py        ← abstract coordinator interface
├── tests/
│   ├── test_bucket_cache.py
│   ├── test_bucket_cache_lifecycle.py
│   ├── test_model_update_service.py
│   ├── test_nemo_rl_pipeline.py
│   └── integration/
│       ├── test_gate2_5_nccl_destroy.py        ← NCCL lifecycle stability
│       ├── test_gate2_5_selective_sync.py      ← NCCL proper-subset broadcast
│       ├── test_gate2_5_megatron_tp.py         ← TP=2 training + weight sync
│       ├── test_gate2_5_qwen_train_sync.py     ← Qwen2.5-0.5B real model sync
│       ├── test_gate2_5_full.py                ← 2-pipeline isolation
│       ├── test_gate2_5_feature6.py            ← F6 sync→finalize→activate ordering
│       ├── test_gate2_5_cuda_ipc.py            ← CUDA IPC cross-process
│       ├── test_gate2_5_bucket_size_guard.py   ← bucket_size_bytes guards
│       └── test_gate2_5_trajectory_collector.py← version publish ordering
└── external/
    ├── NeMo/   ← zhenyulincs/RL.git (rlix-task2 / main)
    └── ROLL/   ← rlops/ROLL.git (rlix)

external/NeMo key files:
  nemo_rl/models/policy/workers/megatron_policy_worker.py  ← sender (build cache, sync)
  nemo_rl/models/generation/vllm/vllm_backend.py           ← receiver (CUDA IPC / cpu_serialize)
  nemo_rl/models/generation/vllm/vllm_generation.py        ← Ray actor pass-throughs + barriers
  nemo_rl/algorithms/grpo.py                               ← trajectory collector registration
```

---

## Setup

```bash
# 1. Clone with submodules
git clone https://github.com/zhenyulincs/rlix.git --recurse-submodules
cd rlix

# 2. Install deps
pip install uv && uv sync

# 3. Required env vars (no implicit defaults)
export RLIX_BUCKET_SIZE_BYTES=$((256 * 1024 * 1024))   # 256 MB per bucket
export RLIX_MODEL_UPDATE_TRANSPORT=cpu_serialize         # or cuda_ipc for same-GPU
```

---

## Running tests

### Unit tests (CPU only, no Ray)

```bash
python -m pytest tests/test_bucket_cache.py \
                  tests/test_bucket_cache_lifecycle.py \
                  tests/test_model_update_service.py \
                  tests/test_nemo_rl_pipeline.py -v
# Expected: 53 passed
```

### Gate 2.5 integration tests (4× GPU, torchrun)

```bash
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1   # PCIe hardware (no NVLink)

torchrun --nproc-per-node=2 tests/integration/test_gate2_5_nccl_destroy.py
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_selective_sync.py
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_megatron_tp.py
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=4 tests/integration/test_gate2_5_qwen_train_sync.py
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=4 tests/integration/test_gate2_5_full.py
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_feature6.py
```

All 6 should print `ALL GATE 2.5 * CHECKS PASSED` and exit 0.

### F6.3 / F4.4 / F6.6 targeted tests (single GPU)

```bash
python tests/integration/test_gate2_5_cuda_ipc.py        # CUDA IPC zero-copy
python tests/integration/test_gate2_5_bucket_size_guard.py
python tests/integration/test_gate2_5_trajectory_collector.py
```

---

## How it works

### F4 — Cache build after each train step

```
build_latest_bucket_cache(step)
  ├─ all PP/TP/CP/EP ranks participate (collective gather)
  ├─ only cache owner stores buckets
  ├─ packs params → BucketRecord (512-byte-aligned uint8, CPU)
  ├─ fail-fast: single param > bucket_size_bytes → RuntimeError
  └─ fail-fast: 2 × total_model_bytes > 80% available RAM → RuntimeError

promote_active_checkpoint(step)
  └─ VersionedBucketCache: atomically switch active pointer, GC old versions
```

### F6 — Selective sync (ModelUpdateService, 6 phases)

```
Phase 1:  Setup dynamic NCCL groups for cross-GPU targets
Phase 2:  selective_sync_active_cache on all training workers
          ├─ sender holds _cache_lock: cache lookup → transport → NCCL teardown
          ├─ CUDA IPC: get_handle_from_tensor() → rebuild_cuda_tensor() (zero-copy)
          └─ NCCL broadcast: stage CPU→GPU → dist.broadcast() on subset group
Phase 3:  Receiver-side NCCL group teardown (port claim released after)
Phase 4:  Post-sync verification (optional)

Pipeline (after sync_selected_workers returns):
  ├─ finalize_weight_update() on synced ranks  ← pipeline-owned
  ├─ set_weight_version() on trajectory collector  ← BEFORE routing activation
  └─ expand_sampler(skip_load=True)  ← activate routing
```

### Transport modes

| Mode | When | Mechanism |
|------|------|-----------|
| `cuda_ipc` | Same physical GPU (colocated) | `get_handle_from_tensor()` → IPC handle → `rebuild_cuda_tensor()` on receiver |
| `cpu_serialize` | Cross-GPU (default) | CPU uint8 bucket → Ray RPC → `pin_memory().to(device)` DMA |
| NCCL broadcast | Cross-GPU, tp > 1 | Stage CPU→GPU → `dist.broadcast()` on dynamic group `[sender] + [infer_ranks]` |

> **Key spec constraint** (line 316): NCCL cannot form a group between two processes on the **same physical GPU**. CUDA IPC is required for colocated workers — it is a correctness requirement, not just a performance optimization.

---

## Known deferred items

| Item | Reason |
|------|--------|
| `wake_up_partial()` / `activate_dp_ranks()` in expand | Feature 2 (VllmGeneration sleep/wake API) not yet built |
| ZMQ ping-pong buffering for IPC | `zmq` not in NeMo RL environment; Ray RPC achieves same result |
| `_cache_ready_step` under sender `_cache_lock` | Cross-actor Ray architecture: training worker lock ≠ pipeline lifecycle lock |
