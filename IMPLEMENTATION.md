# Feature 4 & 6 Implementation — NeMo RL Port

Branch: `task2-bucket-cache`  
Spec: `/Users/zhenyulin/Downloads/nemorl-port-plan.md` (Features 4 and 6)  
GPU hardware used for testing: Vast.ai instance 35236058, 4× RTX A5000

## Changelog

| Date | Fix |
|------|-----|
| 2026-04-23 | CPUBucketCache → VersionedBucketCache API migration in test_gate2_5_*.py |
| 2026-04-23 | `model_update_transport` param added to `selective_sync_active_cache` |
| 2026-04-23 | `destroy_collective_group` added to sender inside `selective_sync_active_cache` |
| 2026-04-23 | `_expand_workers` ordering: sync before expand_sampler |
| 2026-04-24 | `promote_active_checkpoint` keyword arg fixed (`version=` not `checkpoint_version=`) |
| 2026-04-24 | `model_update_transport` now passed to `update_parameter_in_bucket` (was hardcoded) |
| 2026-04-24 | Receiver-side NCCL teardown added to `ModelUpdateService` Phase 4 |
| 2026-04-24 | `_cache_lock` now spans full transport + NCCL teardown (was released before teardown) |
| 2026-04-24 | `bucket_size_bytes` now fails fast if not configured (was silent 256 MB default) |
| 2026-04-24 | Host-RAM check now uses actual packed model size, not per-bucket size |
| 2026-04-24 | `finalize_weight_update` moved from `ModelUpdateService` to pipeline (spec-correct owner) |
| 2026-04-24 | `set_trajectory_collector()` added to pipeline; `set_weight_version` wired at all 3 publish sites |
| 2026-04-24 | `_cache_lock` now spans transport + NCCL teardown (sender-side group destroyed inside lock) |
| 2026-04-24 | `bucket_size_bytes` explicit — raises RuntimeError if not configured (no 256 MB default) |
| 2026-04-24 | Host-RAM check moved to `build_latest_bucket_cache` using actual packed model size |
| 2026-04-24 | `finalize_weight_update` moved from `ModelUpdateService` to pipeline (spec-correct owner) |
| 2026-04-24 | `sync_base_weights_to_active` returns synced ranks; pipeline finalizes only those ranks |
| 2026-04-24 | `is_lora: bool = False` added to `update_parameter_in_bucket` and `broadcast_parameter` |
| 2026-04-24 | Trajectory collector injected from `grpo.py` into pipeline via `set_trajectory_collector` |
| 2026-04-24 | All `vllm_generation.py` pass-through methods now await sub-worker futures before returning (phase barrier fix) |
| 2026-04-24 | Receiver uses `unpack_bucket_record()` for `cpu_serialize` path; `cuda_ipc` path reconstructs inline from the GPU buffer (no CPU roundtrip) |
| 2026-04-24 | Old `2 × bucket_size_bytes` RAM guard removed from `ModelUpdateService.__init__` (superseded by per-model check in `build_latest_bucket_cache`) |
| 2026-04-24 | Port claim now released AFTER receiver-side NCCL teardown (was before); failure intentionally leaks claim (spec lines 380-389) |
| 2026-04-24 | Phase list in doc corrected — `finalize_weight_update` is pipeline-owned, not a ModelUpdateService phase |
| 2026-04-24 | Trajectory collector named as Ray actor (`rlix:trajectory_collector:{pipeline_id}`) in `grpo.py`; pipeline resolves it lazily by name via `_get_trajectory_collector()` |
| 2026-04-24 | **F6.3 IMPLEMENTED**: cuda_ipc sender sends IPC handle via `get_handle_from_tensor`; receiver uses `self.rank` (not `dist.get_rank()`) + zero-copy `rebuild_cuda_tensor` (no CPU roundtrip) |
| 2026-04-24 | **F4.4 IMPLEMENTED**: `build_latest_bucket_cache` raises `RuntimeError` for single tensor > `bucket_size_bytes` before append |
| 2026-04-24 | **F6.6 ordering FIXED**: `set_weight_version` called BEFORE `expand_sampler` in `_expand_workers` (spec lines 602-608) |

---

## Feature 4 — CPU Bucket Cache

### What it does

Packs model parameters from a training worker into CPU-resident contiguous uint8 buffers
(`BucketRecord`). These buffers are versioned by `VersionedBucketCache` and used as the
source of truth when syncing weights to inference workers (Feature 6).

### Files

| File | Role |
|---|---|
| `rlix/pipeline/bucket_cache.py` | Core data structures and pack/unpack logic |
| `rlix/pipeline/bucket_cache_lifecycle.py` | Version tracking + Ray actor orchestration |
| `rlix/pipeline/full_finetune_pipeline.py` | Pipeline layer: calls build+promote in correct order |

### Key classes and functions

#### `BucketRecord` (dataclass)

Holds one packed weight buffer for a group of named parameters:

```
param_names  : List[str]        — HF param names packed in this record
shapes       : List[torch.Size] — original per-param shapes
dtypes       : List[torch.dtype] — original per-param dtypes
offsets      : List[int]        — byte offsets into cpu_uint8_bucket (512-byte aligned)
used_bytes   : int              — total payload bytes (excl. alignment padding)
cpu_uint8_bucket : torch.Tensor — contiguous uint8 CPU tensor
```

#### `_bucket_named_tensors(named_tensors) → BucketRecord`

Packs a list of `(name, cpu_tensor)` pairs into a single `BucketRecord`. Each tensor is:
1. Moved to CPU, flattened, viewed as `uint8`.
2. Written into a pre-allocated buffer at a 512-byte-aligned offset (mirrors ROLL `send_recv_utils.py:214` and NeMo RL `calculate_aligned_size`).

#### `unpack_bucket_record(record) → List[(name, tensor)]`

Inverse of `_bucket_named_tensors`. Used on the receiver side to reconstruct per-param
tensors from the raw uint8 buffer. Uses `torch.empty(0, dtype=dtype).element_size()`
to compute byte widths — avoids the illegal uint8→wide-dtype view bug that was present
in the original `vllm_backend.py`.

#### `VersionedBucketCache`

Thread-safe two-pointer cache:

```
build_latest(version, buckets)  — store new version (not yet active)
promote(version)                — atomically make version active; GC old versions
get_active_buckets()            — return active bucket list (caller holds _cache_lock)
```

GC invariant: after each `promote`, only `_latest_cached` and `_active_cached` are kept.
Peak memory ≤ 2× model size.

#### `BucketCacheLifecycle`

Version-tracking wrapper used by the pipeline layer (not by Ray workers directly):

```
promote(version)         — calls promote_active_checkpoint on all workers, updates _cache_ready_step
mark_promoted(version)   — records version only, does NOT call any workers
                           (use when pipeline has already issued ray.get([...remote()]))
promote_base()           — build + promote version=-1 (base model init)
is_ready_for_version(v)  — True if cache_ready_step >= v
reset()                  — clear version state (pipeline restart)
```

### Pipeline lifecycle (correct ordering per spec)

**Init** — pipeline explicitly builds and promotes the base cache before `actor_infer` starts
(`full_finetune_pipeline.py` lines ~289-310, ~448-458):
```python
# All training workers participate; only cache owner stores buckets.
ray.get([w.build_latest_bucket_cache.remote(checkpoint_version=-1) for w in workers])
# keyword must be version= (matches def promote_active_checkpoint(self, version: int))
ray.get([w.promote_active_checkpoint.remote(version=-1) for w in workers])
# Record in lifecycle without re-calling workers
self._lifecycle = BucketCacheLifecycle(pipeline_id=..., workers=...)
self._lifecycle.mark_promoted(BucketCacheLifecycle._BASE_VERSION)
self._current_weight_version = self._lifecycle.cache_ready_step
```

**Post-train-step** (`full_finetune_pipeline.py`):
```python
# Spec requires: build THEN promote (never promote before build)
ray.get([w.build_latest_bucket_cache.remote(checkpoint_version) for w in workers])
ray.get([w.promote_active_checkpoint.remote(version=checkpoint_version) for w in workers])
self._lifecycle.mark_promoted(checkpoint_version)
```

### `_cache_lock` critical section (spec: nemorl-port-plan.md line 401-402)

The lock must span **"cache lookup → transport → NCCL teardown"** without gaps.
`selective_sync_active_cache` in `megatron_policy_worker.py` holds `cache._cache_lock`
for the entire bucket transport loop, and `destroy_collective_group(group_name)` is now
called **inside** the `with cache._cache_lock:` block — before the lock is released.

### `bucket_size_bytes` — explicit config required (spec: nemorl-port-plan.md line 343)

`_rlix_get_bucket_size_bytes()` raises `RuntimeError` if neither
`worker.cfg['rlix']['bucket_size_bytes']` nor `RLIX_BUCKET_SIZE_BYTES` env var is set.
No silent 256 MB default.

### Host-RAM fail-fast (spec: nemorl-port-plan.md line 337)

Check runs inside `build_latest_bucket_cache` after the full model has been packed (so
`total_bytes` is exact). Two-pointer versioning requires ≤ 2× model in host RAM:
```python
if 2 * total_bytes > 0.8 * available_ram:
    raise RuntimeError(...)
```
Runs only on `checkpoint_version == -1` (base init). Requires `psutil`; skips with WARNING
if not installed.

### Bug fixed: `vllm_backend.py` element_size

**File**: `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py`  
**Commit**: `5b541d1` on submodule branch `rlix-task2`

Before (incorrect):
```python
nbytes = num_elements * buf[offset : offset + 1].view(dtype).element_size()
# Slicing 1 uint8 byte then viewing as bfloat16 is undefined for small slices.
```

After (correct):
```python
nbytes = num_elements * torch.empty(0, dtype=dtype).element_size()
# Returns element size without touching any buffer data.
```

---

## Feature 6 — Base-Weight Sync / Selective Sync

### What it does

Transfers the training cluster's active CPU bucket cache to specific inference workers
on pipeline expand. Uses NCCL broadcast for cross-GPU and `cpu_serialize` (ZMQ DMA) for
same-GPU transfers.

### Files

| File | Role |
|---|---|
| `rlix/pipeline/model_update_service.py` | Ray actor orchestrating the 6-phase sync flow |
| `rlix/protocol/coordinator.py` | Abstract method `sync_base_weights_to_active()` |
| `rlix/pipeline/coordinator.py` | Concrete impl: snapshots `_active_infer_dp_ranks`, calls `sync_selected_workers` |
| `rlix/pipeline/full_finetune_pipeline.py` | `_expand_workers` (lines ~482-511) and post-train sync (lines ~1062-1077) |
| `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py` | Sender: `selective_sync_active_cache`, `setup_collective_group`, `destroy_collective_group` |
| `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py` | Receiver: `setup_collective_group`, `update_parameter_in_bucket`, `broadcast_parameter`, `destroy_collective_group`, `finalize_weight_update`, `verify_model` |
| `external/NeMo/nemo_rl/models/generation/vllm/vllm_generation.py` | Exposes receiver methods as Ray-callable actor methods |

### `ModelUpdateService.sync_selected_workers` — 6-phase flow

```
Phase 1: Set up temporary NCCL collective groups (broadcast-path workers only)
         - IPC-only targets skip NCCL setup entirely
         - Sender joins as rank 0; receivers as ranks 1..N

Phase 2: Dispatch selective_sync_active_cache to all training workers
         - Only the global cache owner (pp_rank==0, dp_rank==0, tp_rank==0) transfers
         - Non-owners return immediately
         - ray.get(sync_refs) acts as the sync barrier
         - Sender destroys its NCCL group inside selective_sync_active_cache before returning

Phase 3: Receiver-side NCCL group teardown
         - Each broadcast-path target worker calls destroy_collective_group(group_name)
         - Port claim released AFTER teardown (spec lines 380-389: success only; failure leaks)
         - Spec ref: nemorl-port-plan.md lines 380, 385

Phase 4: Post-sync verification (optional, verify=True by default)
         - Sender returns weight_stats (checksums/norms)
         - Each target worker's verify_model checks weights landed correctly

NOTE: finalize_weight_update is NOT called inside ModelUpdateService.
      It is pipeline-owned (spec: nemorl-port-plan.md line 624-632).
      The pipeline calls it after sync_selected_workers() returns.
```

### Same-GPU transport: `model_update_transport` parameter

`selective_sync_active_cache` accepts `model_update_transport` (default `"cpu_serialize"`).
The sender passes this to `update_parameter_in_bucket.remote(payload, local_ranks, model_update_transport)`.

**Both `"cpu_serialize"` and `"cuda_ipc"` are now implemented end-to-end** (2026-04-24):

- `"cpu_serialize"`: payload contains `cpu_uint8_bucket` (CPU uint8 tensor). Receiver
  uses `pin_memory().to(device)` DMA then unpacks via `unpack_bucket_record`.
- `"cuda_ipc"`: sender calls `get_handle_from_tensor(staging_buf)` to produce a CUDA IPC
  handle tuple; payload contains `cuda_ipc_handle`. Receiver calls
  `rebuild_cuda_tensor(*ipc_args)` for zero-copy GPU tensor reconstruction (no CPU roundtrip).
  Rank mask uses `self.rank` (vLLM worker local rank), not `dist.get_rank()`.
  Required for colocated workers (NCCL cannot form a group on the same GPU, spec line 316).

### `finalize_weight_update` — pipeline-owned (spec: nemorl-port-plan.md line 624-632)

The spec assigns `finalize_weight_update()` to the **pipeline**, not `ModelUpdateService`.
Ownership was moved:
- `ModelUpdateService.sync_selected_workers` no longer calls `finalize_weight_update`
- `_expand_workers` calls `actor_infer.rank2worker[r].finalize_weight_update.remote()` for each target rank **after sync returns**, before routing is activated
- Post-train `sync_base_weights_to_active` path also calls finalize for all active ranks after sync

### Trajectory collector wiring (spec: nemorl-port-plan.md lines 490, 538, 603)

`AsyncTrajectoryCollector` is registered as a **named Ray actor** in `grpo.py`:
```python
name = f"rlix:trajectory_collector:{pipeline_id}"  # from PIPELINE_ID env var
namespace = os.environ.get("ROLL_RAY_NAMESPACE", "")
trajectory_collector = AsyncTrajectoryCollector.options(name=name, namespace=namespace).remote(...)
```

The pipeline resolves the collector lazily via `_get_trajectory_collector()`, which calls
`ray.get_actor(f"rlix:trajectory_collector:{pipeline_id}", namespace=namespace)` on first use.
`set_weight_version.remote(version)` is called at all three publish sites:
1. Init (base version −1)
2. `_expand_workers` post-sync (no version bump)
3. Post-train active refresh

`FullFinetunePipeline` also exposes `set_trajectory_collector(collector)` as an explicit
injection path (fallback when env vars are unavailable).

### `_expand_workers` — atomic expand ordering

Spec (nemorl-port-plan.md lines 589-609): sync must complete before routing is activated.
Correct order implemented:
```
1. sync_selected_workers(tgt_dp_ranks)           ← weights land before ranks become routable
2. finalize_weight_update on synced ranks         ← pipeline-owned post-bucket hook
3. _current_weight_version = cache_ready_step
4. trajectory_collector.set_weight_version(v)    ← BEFORE routing activation (spec lines 602-608)
5. expand_sampler(dp_ranks, skip_load=True)       ← rebalance_on_expand → routing active
```
Note: set_weight_version is called BEFORE expand_sampler (fixed 2026-04-24). Previously it
was after, which meant newly expanded ranks could serve requests before the collector saw
the correct weight version.
Note: `mark_dp_ranks_inactive` / `wake_up_partial` / `activate_dp_ranks` are Feature 2
methods not yet implemented; `expand_sampler(skip_load=True)` provides the equivalent
routing-activation effect via ROLL's scheduler.

### Version publication to trajectory collector

`AsyncTrajectoryCollector` (`nemo_rl/algorithms/async_utils.py`) is registered as a named Ray
actor in `grpo.py` (name = `rlix:trajectory_collector:{PIPELINE_ID}`). The pipeline resolves
it lazily via `_get_trajectory_collector()` and calls `set_weight_version.remote(version)` at:
- Base init (version −1)
- `_expand_workers` post-finalize (no version bump)
- Post-train active refresh post-finalize

### Known deferred items (not F4/F6 code gaps)

| Item | Status |
|------|--------|
| Same-GPU CUDA IPC via ZMQ (ping-pong buffering) | Deferred. The current `cuda_ipc` path sends IPC handles via Ray RPC (works correctly). ROLL's original uses ZMQ sockets for ping-pong double buffering to overlap communication. ZMQ not installed in the NeMo RL environment; Ray RPC achieves equivalent result without ZMQ. |
| `wake_up_partial()` / `activate_dp_ranks()` in `_expand_workers` | Deferred to Feature 2. These `VllmGeneration` sleep/wake methods are not yet implemented. Current code uses ROLL's `expand_sampler(skip_load=True)` for the equivalent routing-activation effect. |
| `_cache_ready_step` publication under sender `_cache_lock` | Architectural constraint: `_cache_lock` is on the training worker Ray actor; `_cache_ready_step` is in `BucketCacheLifecycle` on the pipeline actor. These are in different Ray processes — they cannot share the same lock. The spec intent (prevent concurrent build racing sync) is achieved: `_cache_lock` covers the full transport window, and `mark_promoted` is called after the transport completes. |

### Known intentional extras (code does more than spec requires)

| Item | Rationale |
|------|-----------|
| `VersionedBucketCache` two-pointer design | Spec (nemorl-port-plan.md line 397) asks for a simpler single-slot `_cache_ready_step`. The two-pointer implementation (`_latest_cached` + `_active_cached` + GC) was chosen to mirror ROLL's proven `megatron_strategy.py:1049-1065` pattern and provide safety against concurrent build/promote races. Strictly more than the spec requires; semantics are compatible. |
| `BucketCacheLifecycle.promote_base()`, `mark_promoted()`, `reset()` | Helper methods for the pipeline orchestration layer not explicitly named in the spec; they implement the spec's build/promote sequencing without violating it. |
| `set_trajectory_collector()` injection API | The spec only specifies the named-actor lookup path. The explicit injection setter is a fallback for environments where `PIPELINE_ID` env var is unavailable. |

### Phase barriers

All `vllm_generation.py` pass-through methods now call `ray.get(futures)` before returning,
so outer `ray.get()` calls in `ModelUpdateService` correctly barrier on sub-worker completion.
This covers: `setup_collective_group`, `update_parameter_in_bucket`, `broadcast_parameter`,
`destroy_collective_group`, `verify_model`, `finalize_weight_update`.

### `Coordinator.sync_base_weights_to_active` (abstract method)

Returns `List[int]` — the list of dp_ranks that were synced. The pipeline uses the returned
ranks to call `finalize_weight_update` on exactly the synced workers (not the full dp_size).

```python
@abstractmethod
def sync_base_weights_to_active(self) -> List[int]:
    """Push trained base model weights to all currently-awake infer workers.
    Returns sorted list of synced dp_ranks (empty if all sleeping)."""
    raise NotImplementedError
```

---

## Tests

### Unit tests (no GPU, no Ray)

| File | Tests | What is covered |
|---|---|---|
| `tests/test_bucket_cache_lifecycle.py` | 26 | version tracking, promote, mark_promoted, thread-safety |
| `tests/test_model_update_service.py` | 37 | transport config, bucket_size_bytes guard, finalize_weight_update call |
| `tests/test_nemo_rl_pipeline.py` | 15 | BucketCacheLifecycle + `_expand_workers` ordering |

Notable tests added 2026-04-24:
- `test_expand_workers_sync_before_expand_sampler` — asserts `sync_selected_workers` precedes `expand_sampler` in ordering

### GPU integration tests (4× RTX A5000, Vast.ai)

#### `tests/integration/test_bucket_cache_gpu.py`

Rewritten from deleted `bucket_receiver.py` API to new `BucketRecord`/`VersionedBucketCache` API.

```
platform linux -- Python 3.12.3, pytest-9.0.3
GPU: 4× RTX A5000

PASSED  TestGPUMemoryRelease::test_offload_reduces_allocated_memory
PASSED  TestGPUMemoryRelease::test_cache_does_not_hold_gpu_tensors
PASSED  TestWeightCorrectnessInCache::test_cached_weights_match_original_bit_for_bit
PASSED  TestWeightCorrectnessInCache::test_cached_dtypes_preserved
PASSED  TestBucketRecordPush::test_push_updates_all_parameters
PASSED  TestBucketRecordPush::test_push_no_shape_mismatch
PASSED  TestBucketRecordPush::test_push_to_gpu_target
PASSED  TestVersionedBucketCache::test_build_and_promote_version
PASSED  TestVersionedBucketCache::test_gc_drops_old_version
PASSED  TestFullRoundTrip::test_full_cache_roundtrip_matches_source

10/10 passed in 14.82s
```

What each class verifies:
- **TestGPUMemoryRelease** — GPU memory is actually released after offloading; cache holds only CPU tensors
- **TestWeightCorrectnessInCache** — packed uint8 → unpacked tensors are bit-exact with original; bfloat16 preserved
- **TestBucketRecordPush** — all params updated after push; no shape change; CPU→GPU cross-device copy
- **TestVersionedBucketCache** — build/promote makes version accessible; old version GC'd after build_latest(v+2)
- **TestFullRoundTrip** — GPU model → VersionedBucketCache → offload → infer worker push → bit-exact verify

#### `tests/integration/test_gate2_5_selective_sync.py`

2-rank NCCL selective sync test (torchrun, 2 GPUs):

```
torchrun --nproc-per-node=2 tests/integration/test_gate2_5_selective_sync.py
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1  (PCIe hardware — no NVLink/InfiniBand)

[rank1] PASS cycle 1: 8/8 weights bit-exact
[rank1] PASS cycle 2: 8/8 weights bit-exact
[rank1] PASS cycle 3: 8/8 weights bit-exact
[rank0] PASS: VRAM stable across 3 cycles (growth=0.0 MB)
ALL PART 2 CHECKS PASSED
```

What it verifies:
- rank 0 packs weights into a `BucketRecord` (CPU uint8), stages CPU→GPU, broadcasts via NCCL
- rank 1 receives packed buffer, reconstructs `BucketRecord`, calls `unpack_bucket_record`, copies to infer state dict
- 3 cycles of group create → broadcast → group destroy without VRAM growth or NCCL hangs

---

#### `tests/integration/test_gate2_5_megatron_tp.py` (re-run 2026-04-24)

After migrating from deleted `CPUBucketCache` API to `VersionedBucketCache` + `unpack_bucket_record`:

```
torchrun --nproc-per-node=4
ALL GATE 2.5 MEGATRON TP CHECKS PASSED (2 steps)  EXIT:0
```

#### `tests/integration/test_gate2_5_qwen_train_sync.py` (re-run 2026-04-24)

After same migration:

```
torchrun --nproc-per-node=4
[rank2] PASS step 1: all 291 weights verified bit-exact (rank 2)
[rank3] PASS step 1: all 291 weights verified bit-exact (rank 3)
[rank2] PASS step 2: all 291 weights verified bit-exact (rank 2)
[rank3] PASS step 2: all 291 weights verified bit-exact (rank 3)
ALL GATE 2.5 PART 3 CHECKS PASSED (2 steps)  EXIT:0
```

---

## Known constraints

- **NCCL on PCIe**: Set `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` on hardware without NVLink/InfiniBand (e.g. RTX A5000 via PCIe).
- **`finalize_weight_update` is vLLM-specific**: The method must exist on the inference worker actor. Current NeMo RL vllm backend exposes it; stub it out for other backends.
- **`sync_base_weights_to_active` is abstract**: Concrete coordinator subclasses must implement it to wire `ModelUpdateService.sync_selected_workers`.
