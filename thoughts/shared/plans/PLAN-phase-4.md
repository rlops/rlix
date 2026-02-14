## Phase 4 (Final) — Sender `cache_lock` is the Single Source of Serialization (Option A) (ENG-123)

### Summary
Selective sync on expand uses the sender’s **promoted active cached buckets**. Cache is built in `train_step` before offload; sender maintains **two pointers** (`latest_cached`, `active_cached`) and performs **GC on promotion only**. Promotion bypasses ModelUpdateService and updates sender-owned pointers. **Option A:** the sender’s single `cache_lock` provides all mutual exclusion: promotion/GC and the entire selective sync execution. Receiver target is upstream vLLM `update_parameter_in_bucket(serialized_named_tensors)` using rank-indexed payloads; no staging flag dependency and no fork receiver-shard APIs.

---

## 1) Sender-side cache/promotion/sync serialization (single lock)

### 1.1 Single `cache_lock`
Sender component that owns cache pointers/buckets has exactly one mutex:
- `cache_lock` protects:
  - `cache_map[(checkpoint_version:int, global_step:int)]`
  - `latest_cached`, `active_cached`
  - GC deletions
  - the entire selective sync execution (see 1.5)

### 1.2 Cache build in `train_step` (no GC here)
At end of `train_step`, under `cache_lock`:
1) bucketize + serialize weights using upstream `roll.utils.send_recv_utils` format
2) store into `cache_map[(checkpoint_version, global_step)]`
3) set `latest_cached = (checkpoint_version, global_step)`
4) **do not GC here**
5) offload only after cache exists

### 1.3 Promotion target is implementation-defined (strategy method or minimal Worker RPC)
Coordinator calls the sender-side component that owns pointers:
- `promote_active_checkpoint(checkpoint_version:int, global_step:int)` (strategy method **or** minimal Worker RPC)

### 1.4 Promotion + GC is atomic under `cache_lock`
`promote_active_checkpoint(...)` runs under `cache_lock`:
- verify key exists
- set sender-owned `active_cached = (checkpoint_version, global_step)`
- GC delete all caches except `{latest_cached, active_cached}`
- fail fast on invariant/deletion errors  
**Promotion atomicity requirement:** updating `active_cached` and performing GC deletions must occur within the same `cache_lock` critical section.

### 1.5 Critical Option A rule: selective sync holds `cache_lock` start→finish
Sender-side selective sync entrypoint (invoked by scheduler-side service) must:
- acquire `cache_lock`
- read `active_cached`
- use buckets from `cache_map[active_cached]`
- perform all IPC/NCCL sends/applies for `tgt_dp_ranks`
- teardown groups
- release `cache_lock` only after the sync fully completes (including teardown)

Implication:
- promotion/GC cannot run concurrently with sync and cannot delete buckets mid-sync, with no sync_inflight/leases/refcounts.

---

## 2) Scheduler-side ModelUpdateService (trigger only; correctness serialized by sender)

### 2.1 API surface
- `ModelUpdateService.sync_selected_workers(tgt_dp_ranks: List[int]) -> None`

### 2.2 No validation / no coalescing
- No requested_global_step.
- No monotonic/future checks.
- No coalescing logic.

### 2.3 Concurrency behavior (explicit nit)
- Service may be called concurrently; calls block on the sender `cache_lock` and execute one-at-a-time.
- Execution order follows lock acquisition timing; treat as arrival-ordered best effort (FIFO not guaranteed).
- Scheduler must `await` `sync_selected_workers(...)` end-to-end before reopening admission/routing.

---

## 3) Colocation + path selection (reuse upstream behavior)
- Colocation detection uses upstream overlap mechanism `is_actor_infer_overlapping_with_any_cluster(...)`.
- Colocated mode uses IPC for overlapped workers and may still broadcast to mismatch/non-overlapped workers if required by mapping (upstream Megatron behavior).

---

## 4) Receiver target (upstream vLLM only; no staging flag; no fork APIs)
- Receiver apply contract is upstream `roll/third_party/vllm/worker.py::update_parameter_in_bucket(serialized_named_tensors, ...)`:
  - indexes `serialized_named_tensors[self.rank]`
  - deserializes via `MultiprocessingSerializer.deserialize(...)`
  - applies weights
- Sender payload must be a rank-indexed list-like payload in the receiver engine-local TP/PP rank space (not `dp_rank`), using upstream `roll.utils.send_recv_utils` format.
- Do not rely on fork-only receiver-shard APIs (`meta_infos/buffer/ranks_in_worker`).
- Do not depend on any “disable CPU staging” flag.

