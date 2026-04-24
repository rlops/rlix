# Task 2 — CPU Bucket Cache + Selective Weight Sync (F4, F6-transport)

> **Branch**: `task2-bucket-cache` (rlix) + `rlix-task2` (NeMo submodule)  
> **Gate**: 2.5 — all 6 integration tests pass on 4× RTX A5000

---

## What Task 2 implements

Task 2 ports two features from ROLL's `megatron_strategy.py` to the NeMo RL training stack, enabling GPU time-sharing between training and inference workers:

| Feature | Description |
|---------|-------------|
| **F4** | Training-side CPU bucket cache: after each train step, model weights are packed into `BucketRecord` (512-byte-aligned uint8 CPU tensor) and stored in a `VersionedBucketCache`. Inference workers receive weights from this cache instead of live GPU tensors. |
| **F6-transport** | Selective sync: `ModelUpdateService` transfers the active CPU cache to specific inference workers using two paths — **CUDA IPC** for same-GPU colocated workers, **dynamic NCCL group broadcast** for cross-GPU workers. |

---

## Repository layout

```
rlix/                              ← this repo (task2-bucket-cache branch)
├── rlix/pipeline/
│   ├── bucket_cache.py            ← BucketRecord, VersionedBucketCache, unpack_bucket_record
│   ├── bucket_cache_lifecycle.py  ← BucketCacheLifecycle (version tracking)
│   ├── model_update_service.py    ← ModelUpdateService (6-phase sync orchestrator)
│   ├── coordinator.py             ← sync_base_weights_to_active()
│   └── full_finetune_pipeline.py  ← _expand_workers(), version publish, finalize
├── rlix/protocol/
│   └── coordinator.py             ← abstract protocol interface
├── tests/
│   ├── test_bucket_cache.py
│   ├── test_bucket_cache_lifecycle.py
│   ├── test_model_update_service.py
│   ├── test_nemo_rl_pipeline.py
│   └── integration/
│       ├── test_gate2_5_nccl_destroy.py       ← Gate 2.5: NCCL lifecycle
│       ├── test_gate2_5_selective_sync.py     ← Gate 2.5: NCCL subset broadcast
│       ├── test_gate2_5_megatron_tp.py        ← Gate 2.5: TP=2 training + sync
│       ├── test_gate2_5_qwen_train_sync.py    ← Gate 2.5: Qwen2.5-0.5B sync
│       ├── test_gate2_5_full.py               ← Gate 2.5: 2-pipeline isolation
│       ├── test_gate2_5_feature6.py           ← F6 ordering: sync→finalize→activate
│       ├── test_gate2_5_cuda_ipc.py           ← F6.3: CUDA IPC cross-process
│       ├── test_gate2_5_bucket_size_guard.py  ← F4.4: bucket_size_bytes guards
│       └── test_gate2_5_trajectory_collector.py  ← F6.6: version publish ordering
└── external/
    ├── NeMo/    ← submodule: zhenyulincs/RL.git @ rlix-task2
    └── ROLL/    ← submodule: rlops/ROLL.git @ rlix
```

The NeMo submodule (`external/NeMo`, branch `rlix-task2`) contains the changes to:
- `nemo_rl/models/policy/workers/megatron_policy_worker.py` — `build_latest_bucket_cache`, `selective_sync_active_cache` (sender)
- `nemo_rl/models/generation/vllm/vllm_backend.py` — `update_parameter_in_bucket` (receiver, CUDA IPC + cpu_serialize)
- `nemo_rl/models/generation/vllm/vllm_generation.py` — pass-through actor methods with phase barriers
- `nemo_rl/algorithms/grpo.py` — trajectory collector named-actor registration

---

## Setup

### 1. Clone with submodules

```bash
git clone https://github.com/zhenyulincs/rlix.git --recurse-submodules
cd rlix
git checkout task2-bucket-cache
git submodule update --init --recursive
```

### 2. Python environment

```bash
# The project uses uv for env management
pip install uv
uv sync
```

### 3. Required environment variables

```bash
# Bucket size for CPU cache staging (no implicit default)
export RLIX_BUCKET_SIZE_BYTES=$((256 * 1024 * 1024))   # 256 MB

# Transport mode: cpu_serialize (default) or cuda_ipc (same-GPU colocated)
export RLIX_MODEL_UPDATE_TRANSPORT=cpu_serialize

# Vast.ai / GPU instance access (for integration tests)
# See .env file — never commit secrets
```

---

## Running the tests

### Unit tests (no GPU required)

```bash
cd rlix
python -m pytest tests/test_bucket_cache.py \
                  tests/test_bucket_cache_lifecycle.py \
                  tests/test_model_update_service.py \
                  tests/test_nemo_rl_pipeline.py -v
```

Expected: **53 passed**

### Gate 2.5 integration tests (requires 4× GPU)

All tests use `torchrun` and `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` for PCIe hardware (no NVLink).

```bash
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1

# 1. NCCL destroy/re-init stability (2 GPUs)
torchrun --nproc-per-node=2 tests/integration/test_gate2_5_nccl_destroy.py

# 2. Selective sync via NCCL proper-subset group (4 GPUs)
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_selective_sync.py

# 3. Megatron TP=2 training + NCCL weight sync per shard (4 GPUs)
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_megatron_tp.py

# 4. Qwen2.5-0.5B real model training + sync (4 GPUs)
#    Requires HF model cached: Qwen/Qwen2.5-0.5B
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_qwen_train_sync.py

# 5. Two-pipeline alternating sync, A≠B isolation (4 GPUs)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_full.py

# 6. Feature 6 ordering: sync→finalize→version_publish→activate (4 GPUs)
torchrun --nproc-per-node=4 tests/integration/test_gate2_5_feature6.py
```

All 6 should print `ALL GATE 2.5 * CHECKS PASSED` and exit 0.

### F6.3 / F4.4 / F6.6 targeted tests

```bash
# CUDA IPC cross-process (same GPU, 2 spawned processes)
python tests/integration/test_gate2_5_cuda_ipc.py

# Bucket-size configuration guards
python tests/integration/test_gate2_5_bucket_size_guard.py

# Trajectory collector version publish ordering
python tests/integration/test_gate2_5_trajectory_collector.py
```

---

## Architecture — how it works

### F4: CPU bucket cache

```
TrainStep → build_latest_bucket_cache(step)
              └─ all PP/TP/CP/EP ranks participate in gather
              └─ only cache owner (pp0/dp0/tp0/cp0) stores buckets
              └─ packs params into BucketRecord (512-byte-aligned uint8)
              └─ checks bucket_size_bytes (fail fast if oversized param)
              └─ checks host-RAM budget (2 × model_bytes < 80% available)
          → promote_active_checkpoint(step)
              └─ atomically switches VersionedBucketCache active pointer
              └─ GC old versions (keeps at most 2 copies in host RAM)
```

### F6: Selective sync (6-phase flow in ModelUpdateService)

```
Phase 1: Setup dynamic NCCL groups for broadcast-path targets
Phase 2: selective_sync_active_cache on all training workers
         └─ sender (cache owner) holds _cache_lock throughout
         └─ CUDA IPC path: get_handle_from_tensor() → IPC handle to receiver
         └─ NCCL broadcast path: stage CPU→GPU → dist.broadcast()
         └─ sender destroys NCCL group inside _cache_lock (spec line 402)
Phase 3: Receiver-side NCCL group teardown
         └─ Port claim released after teardown (not before)
Phase 4: Post-sync verification (optional)
---
Pipeline (after sync_selected_workers returns):
         └─ finalize_weight_update() on each synced rank (FP8 hooks etc.)
         └─ set_weight_version() on trajectory collector (BEFORE routing)
         └─ expand_sampler(skip_load=True) → activate routing
```

### Transport modes

| Mode | When | How |
|------|------|-----|
| `cuda_ipc` | Same physical GPU (colocated training+inference) | `get_handle_from_tensor()` → IPC handle → `rebuild_cuda_tensor()` on receiver (zero-copy) |
| `cpu_serialize` | Cross-GPU | CPU uint8 bucket dict → Ray RPC → `pin_memory().to(device)` DMA on receiver |
| NCCL broadcast | Cross-GPU, TP > 1 | Stage CPU→GPU → `dist.broadcast()` on dynamic group `[sender] + [infer_ranks]` |

---

## Key spec references

All requirements come from `plans/nemorl-port-plan.md`:

- **F4 cache owner**: lines 332–335
- **bucket_size_bytes explicit**: line 343
- **host-RAM fail-fast**: line 337
- **`_cache_lock` scope**: lines 401–402
- **IPC vs NCCL routing**: lines 316–322, 391
- **finalize_weight_update ownership**: lines 624–632
- **version publish before activate**: lines 602–608
- **port claim after teardown**: lines 380–389

---

## Known deferred items

| Item | Reason |
|------|--------|
| `wake_up_partial()` / `activate_dp_ranks()` in expand | Feature 2 (VllmGeneration sleep/wake API not yet built) |
| ZMQ ping-pong IPC buffering | `zmq` not in NeMo RL env; Ray RPC achieves equivalent result |
| `_cache_ready_step` under `_cache_lock` | Cross-actor Ray architecture constraint; separate lock by design |

---

## Documents

| File | Purpose |
|------|---------|
| `IMPLEMENTATION.md` | What was implemented and how, with file:line citations |
| `DESIGN_F4_F6.md` | Spec requirement → code mapping, Gate 2.5 coverage table |
| `ROLL_VS_NEMO_ANALYSIS.md` | How NeMo port differs from ROLL's original implementation |
| `FINAL_CODEX_REVIEW.md` | Latest Codex compliance review results |
