# TASK 2: CPU Bucket Cache + Lifecycle Version Tracking

Branch: `task2-bucket-cache`

## What Was Built

TASK 2 from the NeMo port plan implements the **CPU bucket cache** abstraction
that decouples weight serialisation from weight broadcasting.  In ROLL's
Megatron strategy, trained weights are gathered from all PP ranks into a CPU
buffer (`_build_latest_bucket_cache`, called inside `train_step` when
`DO_TIME_SHARING=True`) and then atomically committed (`promote_active_checkpoint`)
so the inference workers can pull them without racing against the next train step.

Four modules were ported/created:

| File | Origin | Purpose |
|------|--------|---------|
| `rlix/pipeline/bucket_cache.py` | ported from nemo-integration | Thread-safe in-process cache keyed by `(param_name, shard_id)` |
| `rlix/pipeline/bucket_receiver.py` | ported | PP-shard merging + state-dict patching on inference workers |
| `rlix/pipeline/model_update_service_cached.py` | ported | Orchestrates populate-from-PP + dirty-sync-to-inference |
| `rlix/pipeline/bucket_cache_lifecycle.py` | **new** | Wraps ROLL's `promote_active_checkpoint` with version tracking |

## Architecture

```
train_step (inside ROLL megatron_strategy.py)
  └─ _build_latest_bucket_cache(version)   ← PP gather → CPU bytes

pipeline (after train_step returns)
  └─ BucketCacheLifecycle.promote(version)
       ├─ worker.promote_active_checkpoint(version)  ← atomically commits in ROLL
       └─ _cache_ready_step = version

scheduler (before expand)
  └─ lifecycle.is_ready_for_version(v)  → True/False

ModelUpdateServiceCached.sync_from_cache(tgt_dp_ranks)
  ├─ get dirty buckets from CPUBucketCache
  ├─ send BucketUpdateRequest to each inference worker
  └─ mark buckets clean after ACK
```

## Module Details

### CPUBucketCache (`bucket_cache.py`)

Thread-safe dict keyed by `(param_name: str, shard_id: int)`.

- `store(key, data)` — marks key dirty
- `get_dirty_buckets()` — returns `{key: data}` for all dirty entries
- `mark_synced(keys)` / `mark_all_synced()` — clears dirty flags
- `mark_all_dirty()` — re-marks everything dirty (used after populate)
- `evict(key)` / `evict_param(param_name)` / `clear()` — memory management

`shard_id` maps to PP rank so that multi-rank PP gathers can be stored as
separate shards and reassembled on the receiver side.

### BucketReceiver (`bucket_receiver.py`)

- `BucketUpdateRequest(sync_id, buckets)` — list of `(param_name, shard_id, data)` tuples
- `BucketUpdateResult(sync_id, applied, failed, errors)` — fail-partial: one bad param
  doesn't abort the rest; `.ok` property = `len(failed) == 0`
- `merge_pp_shards(buckets)` — validates contiguous shard_ids `[0, 1, ..., N-1]`,
  concatenates along dim=0
- `apply_bucket_update(state_dict, request)` — groups by param_name, merges PP
  shards, copies tensor data into state_dict in-place

### ModelUpdateServiceCached (`model_update_service_cached.py`)

Owns a `CPUBucketCache`.

- `populate_cache_from_workers(workers)` — calls `get_pp_weight_shards(pp_rank)` on
  each worker, stores with `shard_id=pp_rank`, then `mark_all_dirty()`
- `sync_from_cache(tgt_workers)` — sends dirty buckets as `BucketUpdateRequest`,
  marks clean on success

### BucketCacheLifecycle (`bucket_cache_lifecycle.py`)

Standalone version tracker for `promote_active_checkpoint`.

```python
lifecycle = BucketCacheLifecycle(pipeline_id="p0", workers=train_workers)
lifecycle.promote_base()              # version=-1, after init
lifecycle.promote(step)              # after each train_step
lifecycle.is_ready_for_version(v)    # scheduler check before expand
lifecycle.reset()                    # after pipeline restart
```

Key design: `promote()` calls `worker.promote_active_checkpoint(version)` as a
**direct Python call** (not `.remote()`).  The pipeline layer is responsible for
wrapping in `ray.get([w.promote_active_checkpoint.remote(v) for w in workers])`
before calling the lifecycle.  This keeps the class testable without Ray.

`_cache_ready_step` uses a sentinel object (`_UNINITIALIZED`) so that version `0`
is distinguishable from "never promoted".

## Tests

64 unit tests across 4 files — all pass without Ray or GPU:

```
tests/test_bucket_cache.py           22 tests
tests/test_bucket_receiver.py        12 tests
tests/test_model_update_service_cache.py   9 tests
tests/test_bucket_cache_lifecycle.py 21 tests
```

Run with:
```bash
cd rlix
python3 -m pytest tests/test_bucket_cache*.py tests/test_bucket_receiver.py tests/test_model_update_service_cache.py -v
```

## Bugs Encountered

### 1. A5000 `setup_env.sh` apt lock race (instance setup)

**Error:**
```
E: Could not get lock /var/lib/apt/lists/lock. It is held by process 3105 (apt-get)
```

**Cause:** `unattended-upgrades` was running concurrently with `setup_env.sh`'s
`apt-get update`, holding the apt lock.  A subsequent `tg4perfetto` pip install
failed because `protoc` wasn't installed yet.

**Fix:** Waited for background apt to finish, then re-ran the affected pip
installs manually:
```bash
uv pip install --no-deps tg4perfetto>=0.0.6
uv pip install /root/rlix
```

**Lesson:** On fresh GPU instances, wait ~60s after first SSH before running
`apt-get` commands to let cloud-init / unattended-upgrades settle.

### 2. `BucketCacheLifecycle.promote()` — `.remote()` AttributeError

**Error (17 test failures):**
```
AttributeError: 'function' object has no attribute 'remote'
```

**Cause:** The initial implementation called
`worker.promote_active_checkpoint.remote(version)`, expecting a Ray actor.
Test fake workers are plain Python objects — their methods have no `.remote()` attribute.

**Fix:** Changed to direct call:
```python
# Before (broken in tests)
refs = [w.promote_active_checkpoint.remote(version) for w in self._workers]
ray.get(refs)

# After (testable)
for worker in self._workers:
    worker.promote_active_checkpoint(version)
```

The pipeline layer handles Ray scheduling; `BucketCacheLifecycle` stays
framework-agnostic.

**Lesson:** Any class that may need unit testing without Ray should use direct
method calls.  Keep Ray `.remote()` calls at the pipeline orchestration boundary.

### 3. Do NOT call `destroy_model_parallel()` between train steps

**Trap:** It might seem sensible to call `mpu.destroy_model_parallel()` (or
`torch.distributed.destroy_process_group()`) after training to "free GPU memory"
before handing the GPU to inference.

**Why it's wrong:** `destroy_model_parallel()` only tears down NCCL process groups —
it does **not** free tensor memory.  More critically, the time-sharing design
keeps the Megatron worker process alive across steps (it just sleeps while
inference runs).  Destroying the process group means the next `train_step` has
no communication backend → immediate crash.

**How time-sharing actually frees the GPU:**
The Megatron worker calls `_build_latest_bucket_cache` (copies weights to CPU),
then signals vLLM to wake up.  vLLM reuses the same physical GPU via IPC or
NCCL weight injection.  No process restart, no destroy — just sleep/wake.

To free GPU memory legitimately between train and infer, use:
```python
torch.cuda.empty_cache()   # flush PyTorch allocator cache
# (only after del model if you're truly done with training)
```
But in normal time-sharing, this isn't needed either — the GPU is shared in time,
not released.
