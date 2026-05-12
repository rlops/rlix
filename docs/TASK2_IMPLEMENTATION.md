# TASK 2: CPU Bucket Cache + vLLM Receiver Methods

Branch: `task2-bucket-cache`
Commit: `99fd9e2`

---

## What Was Built

Task 2 ports ROLL's two-pointer CPU bucket cache into the NeMo RL architecture,
replacing an incorrect PP-shard-pull implementation with the correct collective-based
approach.

### Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `rlix/pipeline/bucket_cache.py` | Rewrite | `BucketRecord` + `VersionedBucketCache` |
| `rlix/pipeline/bucket_cache_lifecycle.py` | Update | `promote_base()` calls `build_latest_bucket_cache` first |
| `rlix/pipeline/coordinator.py` | Update | `sync_base_weights_to_active()` implementation |
| `rlix/pipeline/bucket_receiver.py` | **Delete** | PP shard-pull incompatible with distributed collectives |
| `rlix/pipeline/model_update_service_cached.py` | **Delete** | Wrong serial shard-pull orchestration |
| `external/NeMo/.../vllm_backend.py` | Add 6 methods | Receiver API on `VllmInternalWorkerExtension` |
| `tests/test_bucket_cache.py` | Rewrite | Real-torch data-integrity round-trip tests |
| `tests/test_bucket_cache_lifecycle.py` | Update | `_FakeWorker` gains `build_latest_bucket_cache` |
| `tests/test_vllm_backend_receiver.py` | New | Receiver method guards |
| `tests/test_model_update_service.py` | New | MUS validation guards |
| `tests/test_nemo_rl_pipeline.py` | New | Pipeline lifecycle ordering |
| `tests/test_bucket_receiver.py` | **Delete** | Tests for deleted module |
| `tests/test_model_update_service_cache.py` | **Delete** | Tests for deleted module |

---

## Architecture

```
train_step()
    ↓
build_cpu_bucket_cache(step)          ← ALL PP/TP/CP/EP ranks participate in collective
  cache owner (pp0/dp0/tp0/cp0)       ← packs List[BucketRecord], calls build_latest(step, buckets)
  non-owners                          ← drain generator (keeps collective alive)
    ↓
promote_active_checkpoint(step)       ← switches _active_cached pointer; GC old versions
    ↓
ModelUpdateService.sync_selected_workers(tgt_dp_ranks)
  per bucket:
    staging_buf = bucket.cpu_uint8_bucket.pin_memory().cuda()  ← CPU→GPU, one bucket at a time
    → IPC path: update_parameter_in_bucket() on vllm workers
    → NCCL path: broadcast_parameter() on vllm workers
    ray.get(recv_refs)                ← barrier
    finally: del staging_buf          ← immediate release, controls peak VRAM
  finalize_weight_update() per target worker
```

---

## Module Details

### `BucketRecord` (`bucket_cache.py`)

```python
@dataclass
class BucketRecord:
    param_names: List[str]       # HF param names packed in this buffer, in order
    shapes: List                 # per-param original shapes
    dtypes: List                 # per-param original dtypes
    offsets: List[int]           # byte offsets into cpu_uint8_bucket for each param
    used_bytes: int              # total bytes actually written (no alignment padding)
    cpu_uint8_bucket: Tensor     # contiguous uint8 CPU tensor
```

All params are packed with 512-byte alignment between them (mirrors NeMo RL's
`calculate_aligned_size` and ROLL's `serialize_named_weights`).

### `_bucket_named_tensors(named_tensors)` (`bucket_cache.py`)

Packs `[(name, tensor), ...]` into a `BucketRecord`:
1. For each tensor: `.detach().cpu().contiguous().flatten().view(torch.uint8)` — flatten is required for tensors with ndim > 1
2. Computes 512-byte-aligned offsets
3. Allocates `torch.zeros(total_bytes, dtype=torch.uint8)` and `copy_` each param into its slot
4. Returns `BucketRecord` with all metadata

### `unpack_bucket_record(record)` (`bucket_cache.py`)

Inverse of `_bucket_named_tensors`. Critical: element size is obtained via
`torch.empty(0, dtype=dtype).element_size()` — **not** by slicing the buffer.
Slicing 1 uint8 byte and calling `.view(float32)` crashes real PyTorch because
4-byte alignment is not satisfied.

### `VersionedBucketCache` (`bucket_cache.py`)

Two-pointer version tracking (mirrors ROLL `megatron_strategy.py:1049-1065`):

```python
cache.build_latest(version, buckets)  # store new version, does NOT make it active
cache.promote(version)                # switch active pointer; GC all except latest+active
cache.get_active_buckets()            # read active (caller holds _cache_lock)
cache.cache_ready_step                # currently active version or None
```

GC invariant: after each `promote(v)`, all versions except `_latest_cached` and
`_active_cached` are deleted from `_cache_map`. Peak memory ≤ 2× model.

### `BucketCacheLifecycle` (`bucket_cache_lifecycle.py`)

`promote_base()` now correctly calls `build_latest_bucket_cache(-1)` on all
workers **before** `promote_active_checkpoint(-1)`. Previously it only promoted,
leaving workers without a built cache to promote.

### vLLM Receiver Methods (`vllm_backend.py` — `VllmInternalWorkerExtension`)

| Method | Guard | Purpose |
|--------|-------|---------|
| `update_parameter_in_bucket(payload, ipc_local_ranks, transport)` | `rank not in ipc_local_ranks → return` | IPC weight injection |
| `broadcast_parameter(group_name, names, dtypes, shapes, local_ranks)` | `rank not in local_ranks → return` | NCCL weight injection |
| `destroy_collective_group(group_name)` | `group_name not in _model_update_groups → return` | NCCL PG teardown |
| `setup_collective_group(name, comm_plan, mode, timeout_s)` | — | NCCL PG creation |
| `verify_model(expected_stats)` | — | Weight stats comparison |
| `finalize_weight_update()` | — | `process_weights_after_loading` + FP8 cache |

---

## Test Results

All 65 unit tests pass on Vast.ai A5000 GPU instance with **real PyTorch**
(Python 3.12.3, pytest 9.0.3).

```
platform linux -- Python 3.12.3, pytest-9.0.3
Instance: 213.181.122.2:45678 (A5000 4x)
Venv: /root/rlix/.venv/bin/python

tests/test_bucket_cache.py            36 passed
tests/test_bucket_cache_lifecycle.py  21 passed
tests/test_vllm_backend_receiver.py    8 passed
──────────────────────────────────────────────
TOTAL                                 65 passed  in 1.11s
```

Run on Vast:
```bash
ssh -p 45678 root@213.181.122.2
cd /root/rlix
/root/rlix/.venv/bin/python -m pytest \
    tests/test_bucket_cache.py \
    tests/test_bucket_cache_lifecycle.py \
    tests/test_vllm_backend_receiver.py -v
```

### Key tests

| Test | What it validates |
|------|-------------------|
| `test_round_trip_single_float32` | float32 values survive pack→unpack byte-exact |
| `test_round_trip_multi_params` | multiple params in one bucket all recover correctly |
| `test_round_trip_mixed_dtypes` | float32 and float16 in same bucket both correct |
| `test_round_trip_2d_shape` | 2D tensor shape preserved through pack/unpack |
| `test_round_trip_many_small_params` | 20 scalar params (each << 512B) all recover |
| `test_unpack_element_size_does_not_read_buf_slice` | the element_size bug fix under real torch |
| `test_gc_keeps_only_latest_and_active` | GC invariant: only 2 versions kept |
| `test_destroy_collective_group_noop_when_missing` | no-op guard when group absent |
| `test_finalize_weight_update_calls_process_weights` | called exactly once |

---

## Bugs Fixed

### 1. `unpack_bucket_record` — buffer slice view crash (real torch)

**Error:**
```
RuntimeError: unsupported operation: more than one element of the written-to tensor
refers to a single memory location
```

**Cause:** Original code computed element size as:
```python
element_bytes = buf[offset:offset+1].view(dtype).element_size()
```
In real PyTorch, 1 uint8 byte cannot be reinterpreted as float32 (needs 4 bytes).
This works in stub-based tests but crashes with real torch.

**Fix:**
```python
element_bytes = torch.empty(0, dtype=dtype).element_size()
```

### 2. 2D tensor pack — shape mismatch in `copy_` (real torch)

**Error:**
```
RuntimeError: The size of tensor a (24) must match the size of tensor b (12) at non-singleton dimension 1
```

**Cause:** `.view(torch.uint8)` on a 2D tensor preserves the 2D shape. For a
`(2, 3)` float32 tensor, `view(uint8)` gives `(2, 12)`. Then
`bucket_buf[start:start+nbytes]` is 1D `(24,)`, and `copy_((2, 12))` fails.

**Fix:** Added `.flatten()` before `.view(torch.uint8)`:
```python
uint8_view = tensor.detach().cpu().contiguous().flatten().view(torch.uint8)
```

### 3. Wrong architecture — PP shard-pull incompatible with distributed collectives

**Problem:** The prior implementation called `worker.get_pp_weight_shards(pp_rank)`
serially on each PP rank. PP gather uses NCCL all-gather — all ranks must
participate simultaneously. Serial pulls deadlock.

**Fix:** Deleted `bucket_receiver.py` and `model_update_service_cached.py`.
All ranks call `gather_all_hf_weights()` together; only the cache owner
(pp0/dp0/tp0/cp0) stores results.

### 4. `codetiming` import via `rlix/pipeline/__init__.py`

**Error:**
```
ModuleNotFoundError: No module named 'codetiming'
```

**Cause:** Test imports `from rlix.pipeline.bucket_cache import ...` which
triggers `rlix/pipeline/__init__.py`, which eagerly imports
`full_finetune_pipeline` → `codetiming`. Not installed in test environments.

**Fix:** Tests import `bucket_cache.py` directly via `importlib.util.spec_from_file_location`,
bypassing `__init__.py`. `codetiming` was also installed in the Vast venv via `uv`.

---

## What Remains (Gate 2.5)

The integration test (`Gate 2.5`) requires 2 GPU with tp=2 and validates:
1. `build_cpu_bucket_cache(step)` collective gather with all TP ranks
2. NCCL broadcast transport path (cross-GPU selective sync)
3. `destroy_megatron_nccl_groups()` → `initialize_model_parallel()` stability over 3+ steps

This gate has not been run in this session. The unit test layer above is complete.
