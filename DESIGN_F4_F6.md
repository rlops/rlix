# Task 2 Design Mapping — Feature 4 and Feature 6 Transport

This document maps the repo-local Task 2 requirements from `IMPLEMENTATION.md:39-317`, `docs/TASK2_IMPLEMENTATION.md:34-120`, and `TASK2_REVIEW.md:13-22` to the current `rlix` source tree, then summarizes Gate 2.5 coverage across every `tests/integration/test_gate2_5_*.py` file.

## Feature 4 — CPU bucket cache

### F4.1 Requirement: canonical CPU bucket record and byte-exact pack/unpack

Requirement source: `IMPLEMENTATION.md:43-94`, `docs/TASK2_IMPLEMENTATION.md:59-103`.

Implementation mapping:
- `rlix/pipeline/bucket_cache.py:69-93` defines `BucketRecord` with `param_names`, `shapes`, `dtypes`, `offsets`, `used_bytes`, and `cpu_uint8_bucket`.
- `rlix/pipeline/bucket_cache.py:96-160` implements `_bucket_named_tensors()`, which allocates a contiguous `torch.uint8` CPU buffer, aligns offsets to 512 bytes, and copies flattened CPU tensors into the bucket.
- `rlix/pipeline/bucket_cache.py:164-193` implements `unpack_bucket_record()`, reconstructing typed tensors from the byte buffer with `torch.empty(0, dtype=dtype).element_size()` rather than a buffer-slice `view()`.

Data structure / lifecycle notes:
- Allocate: `torch.zeros(total_bytes, dtype=torch.uint8)` in `rlix/pipeline/bucket_cache.py:147-152`.
- Fill: per-parameter copy into aligned offsets in `rlix/pipeline/bucket_cache.py:149-152`.
- Reconstruct: dtype-aware slicing and reshape in `rlix/pipeline/bucket_cache.py:178-193`.

Gaps:
- No functional gap found in the canonical record itself; the format is implemented and reused by sender/receiver code in the current tree (`rlix/pipeline/bucket_cache.py:1-16`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:388-412`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:451-485`).

### F4.2 Requirement: versioned cache lifecycle with active/latest pointers, eviction, and `_cache_ready_step`

Requirement source: `IMPLEMENTATION.md:83-107`, `IMPLEMENTATION.md:295-296`, `docs/TASK2_IMPLEMENTATION.md:90-108`.

Implementation mapping:
- `rlix/pipeline/bucket_cache.py:196-305` implements `VersionedBucketCache` with `_cache_map`, `_latest_cached`, `_active_cached`, `_cache_lock`, `build_latest()`, `promote()`, `get_active_buckets()`, and `_gc_unlocked()`.
- `rlix/pipeline/bucket_cache.py:296-305` performs reclaim/eviction by deleting every version except `_latest_cached` and `_active_cached`.
- `rlix/pipeline/bucket_cache_lifecycle.py:57-229` implements the pipeline-facing `BucketCacheLifecycle`, including `_cache_ready_step`, `promote_base()`, `promote()`, `mark_promoted()`, `is_ready_for_version()`, and `reset()`.

Lifecycle notes:
- Allocate/fill new version: `build_latest()` stores `List[BucketRecord]` at `rlix/pipeline/bucket_cache.py:223-238`.
- Publish active version: `promote()` flips the active pointer at `rlix/pipeline/bucket_cache.py:239-256`.
- Reclaim stale versions: `_gc_unlocked()` removes old entries at `rlix/pipeline/bucket_cache.py:296-305`.
- Publish `_cache_ready_step`: `BucketCacheLifecycle.promote()` and `mark_promoted()` update the lifecycle tracker at `rlix/pipeline/bucket_cache_lifecycle.py:107-150` and `rlix/pipeline/bucket_cache_lifecycle.py:189-206`.

Gaps:
- The repo intentionally uses a richer two-pointer cache plus a separate lifecycle tracker instead of a single-slot `_cache_ready_step` cache object; that is documented as a deliberate divergence rather than a missing implementation (`IMPLEMENTATION.md:291-297`, `rlix/pipeline/bucket_cache.py:196-305`, `rlix/pipeline/bucket_cache_lifecycle.py:57-229`).

### F4.3 Requirement: training-worker hooks for build/promote, owner-only storage, and init/post-train sequencing

Requirement source: `IMPLEMENTATION.md:109-130`, `docs/TASK2_IMPLEMENTATION.md:36-53`, `TASK2_REVIEW.md:15-22`.

Implementation mapping:
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1131-1138` defines `_rlix_is_cache_owner()`, selecting the single owner by PP/DP/TP/CP rank.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1153-1244` implements `build_latest_bucket_cache()`.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1192-1196` drains the iterator on non-owner ranks instead of storing buckets.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1213-1216` stores built buckets only on the owner.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1245-1268` implements `promote_active_checkpoint()`.
- `rlix/pipeline/full_finetune_pipeline.py:320-341` performs init-time build/promote for version `-1`.
- `rlix/pipeline/full_finetune_pipeline.py:484-492` records the promoted base version in `BucketCacheLifecycle` and publishes the initial collector version.
- `rlix/pipeline/full_finetune_pipeline.py:1084-1102` performs post-train build-then-promote ordering before offload.

Lifecycle notes:
- Init sequence: build base cache, promote base cache, mark lifecycle, publish collector version (`rlix/pipeline/full_finetune_pipeline.py:320-341`, `rlix/pipeline/full_finetune_pipeline.py:484-492`).
- Post-train sequence: build latest cache, promote active checkpoint, mark lifecycle, then offload training workers (`rlix/pipeline/full_finetune_pipeline.py:1084-1110`).

Gaps:
- The repo does not expose a local `gather_all_hf_weights()` symbol or explicit EP-aware group-split logic in the reviewed files; the collective gather is indirect through `self.megatron_bridge.export_hf_weights(...)` inside `_iter_params_with_optional_kv_scales()` (`external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1012-1033`, `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1153-1196`). This is implemented behavior, but the exact TP/PP/EP gather primitive is not visible in repo-local code.

### F4.4 Requirement: explicit capacity guards for bucket size, staging VRAM, and host RAM

Requirement source: `IMPLEMENTATION.md:139-154`, `docs/TASK2_IMPLEMENTATION.md:39-52`, `TASK2_REVIEW.md:17-21`.

Status: IMPLEMENTED.

Implementation mapping:
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2040-2092` implements `_rlix_get_bucket_size_bytes()`, resolving `worker.cfg['rlix']['bucket_size_bytes']` or `RLIX_BUCKET_SIZE_BYTES` and raising if neither is set.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2095-2127` implements `_rlix_check_vram()`, checking `bucket_size_bytes + scratch` against available VRAM.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1201-1209` now raises `RuntimeError` when a single tensor exceeds `bucket_size_bytes` before appending it to the current bucket batch.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1223-1252` performs the host-RAM fail-fast check from the actual packed `total_bytes`.
- `tests/integration/test_gate2_5_bucket_size_guard.py:117-182` covers the oversized-tensor guard and asserts that the production-source guard appears before `current_batch.append(...)`.

Gaps:
- `ModelUpdateService.__init__` still accepts `bucket_size_bytes=None` for tests or single-GPU setups, and the pipeline still passes `None` when `RLIX_BUCKET_SIZE_BYTES` is unset (`rlix/pipeline/model_update_service.py:43-79`, `rlix/pipeline/full_finetune_pipeline.py:453-467`). The sender-side build path now enforces explicit bucket sizing, but the service constructor itself remains looser than the repo docs describe.

### F4.5 Requirement: sender-side `_cache_lock` must span cache lookup, per-bucket transport, and teardown

Requirement source: `IMPLEMENTATION.md:132-154`, `IMPLEMENTATION.md:195-220`, `TASK2_REVIEW.md:20-22`.

Implementation mapping:
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1326-1403` holds `cache._cache_lock` across `get_active_buckets()`, every per-bucket send, sender-side `torch.cuda.synchronize()`, and sender-side `destroy_collective_group(group_name)`.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1332-1391` stages one bucket at a time from pinned CPU to GPU and deletes `staging_buf` immediately after the receiver barrier.
- `rlix/pipeline/model_update_service.py:405-430` performs receiver-side NCCL teardown and releases the master-port claim only after teardown completes.

Lifecycle notes:
- Allocate staging buffer: `bucket.cpu_uint8_bucket.pin_memory().cuda()` at `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1333-1337`.
- Reclaim staging buffer: `del staging_buf` in `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1388-1391`.
- Reclaim NCCL communicator: sender destroy in `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1393-1403`; receiver destroy in `rlix/pipeline/model_update_service.py:405-430`.

Gaps:
- `_cache_ready_step` publication is not updated under the same sender `_cache_lock`; the lifecycle tracker uses its own lock and is updated from the pipeline actor after the worker RPCs complete (`rlix/pipeline/bucket_cache_lifecycle.py:92-105`, `rlix/pipeline/bucket_cache_lifecycle.py:189-206`, `rlix/pipeline/full_finetune_pipeline.py:1101-1102`). The transport critical section is implemented; the version-publish critical section is separate.

### F4.6 Requirement: training GPUs must be offloaded after cache build/promote and before sync/expand reuse

Requirement source: `IMPLEMENTATION.md:109-130`, `docs/TASK2_IMPLEMENTATION.md:36-53`.

Implementation mapping:
- Init-time offload occurs after base-cache build/promote in `rlix/pipeline/full_finetune_pipeline.py:348-351`.
- Post-train offload occurs after build/promote and before active-rank sync in `rlix/pipeline/full_finetune_pipeline.py:1109-1116`.

Gaps:
- No additional code gap found for the training-side offload hook itself.

## Feature 6 Transport

### F6.1 Requirement: selective sync must target only the requested DP ranks and skip when no ranks are active

Requirement source: `IMPLEMENTATION.md:175-220`, `IMPLEMENTATION.md:260-317`, `TASK2_REVIEW.md:18-22`.

Implementation mapping:
- `rlix/pipeline/model_update_service.py:258-463` implements `ModelUpdateService.sync_selected_workers(tgt_dp_ranks, ...)` as the selective transport entrypoint.
- `rlix/pipeline/coordinator.py:507-550` implements `sync_base_weights_to_active()`, snapshots `_active_infer_dp_ranks`, skips with `[]` when no ranks are active, and calls `sync_selected_workers()` otherwise.
- `rlix/protocol/coordinator.py:55-66` exposes the abstract `sync_base_weights_to_active()` contract.
- `rlix/pipeline/full_finetune_pipeline.py:513-556` calls `sync_selected_workers()` for `dp_ranks_to_add` during expand.
- `rlix/pipeline/full_finetune_pipeline.py:1112-1137` calls `sync_base_weights_to_active()`, finalizes only the returned ranks, and publishes the synced version before releasing training GPUs.

Gaps:
- The live expand path still relies on `expand_sampler(skip_load=True)` for routing activation rather than explicit `wake_up_partial()` / `activate_dp_ranks()` calls; the current implementation is selective-sync-first, then ROLL-side expand/routing, not a native NeMo wake API (`rlix/pipeline/full_finetune_pipeline.py:525-555`).

### F6.2 Requirement: dynamic NCCL routing table must classify per-device IPC vs broadcast targets

Requirement source: `IMPLEMENTATION.md:195-220`, `TASK2_REVIEW.md:18-22`.

Implementation mapping:
- `rlix/pipeline/model_update_service.py:120-128` implements `_select_global_sender_rank()`.
- `rlix/pipeline/model_update_service.py:130-256` implements `_build_comm_plan_for_sender()`, classifying each target device by `(node_rank, gpu_rank)` into `ipc_targets`, `tgt_devices`, and `broadcast_local_ranks_by_dp_rank`, then creating the per-sync `group_name`, `master_addr`, and `master_port`.
- `rlix/pipeline/model_update_service.py:327-349` creates temporary NCCL groups only for `tgt_ranks_in_group`.
- `rlix/pipeline/model_update_service.py:405-430` tears down receiver-side groups and releases the port claim after teardown.

Routing / routing-table notes:
- Sender/receiver rank mapping is encoded in `comm_plan[src_rank]` at `rlix/pipeline/model_update_service.py:241-255`.
- The planning layer explicitly distinguishes same-GPU IPC targets from cross-GPU broadcast targets at `rlix/pipeline/model_update_service.py:205-228`.

Gaps:
- No repo-local gap remains in the route-classification table or in the IPC-vs-broadcast split itself (`rlix/pipeline/model_update_service.py:130-256`).

### F6.3 Requirement: same-GPU IPC transport must support producer/consumer protocol for `cpu_serialize` and `cuda_ipc`

Requirement source: `IMPLEMENTATION.md:222-231`, `IMPLEMENTATION.md:284-289`, `TASK2_REVIEW.md:7-10`, `TASK2_REVIEW.md:20-22`.

Status: IMPLEMENTED.

Existing producer/consumer primitives:
- `external/NeMo/nemo_rl/models/policy/utils.py:250-340` implements `stream_weights_via_ipc_zmq_impl()`, which builds a ping-pong IPC stream and emits `(cuda_ipc_handle, param_names, used_bytes)` payloads.
- `external/NeMo/nemo_rl/models/policy/utils.py:386-393` implements `rebuild_cuda_tensor_from_ipc()`.
- `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:163-249` implements the native ZMQ IPC consumer `update_weights_via_ipc_zmq()`.

Selective-sync implementation:
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1355-1392` now branches on `model_update_transport` in the sender. For `cuda_ipc`, it synchronizes the staging stream, calls `get_handle_from_tensor(staging_buf)`, and sends a `cuda_ipc_handle` payload; for `cpu_serialize`, it still sends the packed `cpu_uint8_bucket`.
- `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:390-412` uses `self.rank` for the IPC local-rank mask, branches on `model_update_transport`, patches the CUDA IPC device index for the local worker, and rebuilds the staged GPU buffer via `rebuild_cuda_tensor` with no CPU roundtrip.
- `tests/integration/test_gate2_5_cuda_ipc.py:1-25`, `tests/integration/test_gate2_5_cuda_ipc.py:77-207`, and `tests/integration/test_gate2_5_cuda_ipc.py:221-340` cover CUDA IPC handle generation, same-GPU tensor rebuild, and the receiver-side bucket update path.

Gaps:
- No repo-local implementation gap remains for selective-sync `cuda_ipc`; the same-GPU sender and receiver branches now support both `cpu_serialize` and `cuda_ipc` payloads (`external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1355-1392`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361-430`).

### F6.4 Requirement: cross-GPU transport must create, use, and destroy a dynamic NCCL group per sync

Requirement source: `IMPLEMENTATION.md:195-220`, `TASK2_REVIEW.md:20-22`.

Implementation mapping:
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1421-1470` implements sender-side `setup_collective_group()` with `StatelessProcessGroup`.
- `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:316-359` implements receiver-side `setup_collective_group()`.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1365-1379` sends per-bucket NCCL broadcasts on the sender.
- `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:414-485` implements `broadcast_parameter()`, receives the packed buffer, reconstructs typed tensors, and loads them into the model.
- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1472-1492` and `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:487-507` implement sender/receiver `destroy_collective_group()` with no-op guards.
- `external/NeMo/nemo_rl/models/generation/vllm/vllm_generation.py:858-962` exposes the receiver lifecycle methods as pass-through actor calls and blocks on `ray.get(futures)` for barrier semantics.
- `external/NeMo/nemo_rl/utils/packed_tensor.py:39-95` and `external/NeMo/nemo_rl/utils/packed_tensor.py:98-203` define the native packed broadcast producer/consumer format that `update_weights_from_collective()` reuses in the non-selective path.

Gaps:
- The selective sender currently uses raw `dist.broadcast(staging_buf, src=0, group=nccl_group)` rather than the higher-level `packed_broadcast_producer()` path (`external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1365-1369`, `external/NeMo/nemo_rl/utils/packed_tensor.py:39-95`). That is an implementation choice, not a missing stub, but it means the selective path is similar to rather than identical with the native packed-broadcast helper path.

### F6.5 Requirement: `vllm_backend` must expose the receiver API surface and request schema

Requirement source: `IMPLEMENTATION.md:185-193`, `IMPLEMENTATION.md:233-257`, `TASK2_REVIEW.md:20-22`.

Implementation mapping:
- `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:316-549` defines all six receiver methods: `setup_collective_group`, `update_parameter_in_bucket`, `broadcast_parameter`, `destroy_collective_group`, `verify_model`, and `finalize_weight_update`.
- `external/NeMo/nemo_rl/models/generation/vllm/vllm_generation.py:858-962` exposes matching pass-through methods on the generation actor and awaits inner worker futures.

Request / response schema:
- `update_parameter_in_bucket(payload, ipc_local_ranks, model_update_transport, is_lora=False)` expects a dict with `param_names`, `shapes`, `dtypes`, `offsets`, and `used_bytes`, plus `cpu_uint8_bucket` for `cpu_serialize` or `cuda_ipc_handle` for `cuda_ipc`, and returns via side effect / `None` after weight load (`external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361-430`).
- `broadcast_parameter(group_name, names, dtypes, shapes, broadcast_local_ranks, is_lora=False)` expects group metadata plus tensor metadata and returns via side effect / `None` after load (`external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:414-485`).
- `verify_model(expected_stats)` expects `sum`, `max`, and `min` statistics and raises on mismatch (`external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:508-537`).
- `finalize_weight_update()` runs `process_weights_after_loading(...)` and FP8 cache processing on the worker (`external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:538-549`).

Gaps:
- No repo-local API-surface gap remains; `update_parameter_in_bucket()` now implements both the `cpu_serialize` and `cuda_ipc` branches described by the request schema (`external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361-430`).

### F6.6 Requirement: pipeline-owned finalize and version publication after transport

Requirement source: `IMPLEMENTATION.md:233-257`, `IMPLEMENTATION.md:260-317`, `docs/TASK2_IMPLEMENTATION.md:45-53`.

Status: FIXED / IMPLEMENTED.

Implementation mapping:
- `rlix/pipeline/full_finetune_pipeline.py:536-543` calls `finalize_weight_update.remote()` for each expanded infer rank after `sync_selected_workers()` returns.
- `rlix/pipeline/full_finetune_pipeline.py:545-558` now calls `set_weight_version` before `expand_sampler`, so version publication happens before routing activation, matching spec lines 602-608.
- `rlix/pipeline/full_finetune_pipeline.py:1118-1133` finalizes the active-refresh ranks returned by `sync_base_weights_to_active()` and publishes the updated version before releasing training GPUs.
- `external/NeMo/nemo_rl/algorithms/grpo.py:2518-2546` registers the named `AsyncTrajectoryCollector` actor.
- `external/NeMo/nemo_rl/algorithms/async_utils.py:344-353` implements `set_weight_version()`.
- `tests/integration/test_gate2_5_trajectory_collector.py:112-216` covers the expand-time publish path and asserts `set_weight_version.remote(...)` appears before `expand_sampler.remote(...)` in `_expand_workers()`.

Gaps:
- No repo-local finalize/version-publish gap remains; the expand-time ordering bug is fixed and covered by Gate 2.5 targeted tests (`rlix/pipeline/full_finetune_pipeline.py:545-558`, `tests/integration/test_gate2_5_trajectory_collector.py:148-216`).

## Gate 2.5 Test Coverage Matrix

The repo currently contains nine Gate 2.5 integration files: `tests/integration/test_gate2_5_feature6.py`, `tests/integration/test_gate2_5_full.py`, `tests/integration/test_gate2_5_selective_sync.py`, `tests/integration/test_gate2_5_nccl_destroy.py`, `tests/integration/test_gate2_5_megatron_tp.py`, `tests/integration/test_gate2_5_qwen_train_sync.py`, `tests/integration/test_gate2_5_cuda_ipc.py`, `tests/integration/test_gate2_5_bucket_size_guard.py`, and `tests/integration/test_gate2_5_trajectory_collector.py`.

| test file | spec requirement | status |
|---|---|---|
| `tests/integration/test_gate2_5_feature6.py` | F4.1 canonical bucket format and F6.6 ordering/finalize after sync (`tests/integration/test_gate2_5_feature6.py:1-22`, `tests/integration/test_gate2_5_feature6.py:121-189`, `tests/integration/test_gate2_5_feature6.py:253-309`, `tests/integration/test_gate2_5_feature6.py:357-390`) | `partial` — validates bucket packing, per-cycle NCCL teardown, finalize ordering, and routing activation, but uses hand-written NCCL/GPU test logic instead of `ModelUpdateService` or `vllm_backend` receiver RPCs (`tests/integration/test_gate2_5_feature6.py:171-247`). |
| `tests/integration/test_gate2_5_cuda_ipc.py` | F6.3 same-GPU `cuda_ipc` producer/consumer transport (`tests/integration/test_gate2_5_cuda_ipc.py:1-25`, `tests/integration/test_gate2_5_cuda_ipc.py:77-207`, `tests/integration/test_gate2_5_cuda_ipc.py:221-340`) | `partial` — validates CUDA IPC handle generation, same-GPU zero-copy reconstruction, and the receiver-side bucket update path, but does not drive the full `ModelUpdateService` selective-sync stack end-to-end. |
| `tests/integration/test_gate2_5_bucket_size_guard.py` | F4.4 bucket-size configuration, oversized-tensor fail-fast, and host-RAM guard (`tests/integration/test_gate2_5_bucket_size_guard.py:1-16`, `tests/integration/test_gate2_5_bucket_size_guard.py:54-182`, `tests/integration/test_gate2_5_bucket_size_guard.py:185-253`) | `partial` — covers explicit bucket-size configuration, the oversized single-tensor `RuntimeError`, and host-RAM fail-fast behavior, but does not execute the live VRAM guard through a full worker init path. |
| `tests/integration/test_gate2_5_trajectory_collector.py` | F6.6 trajectory-collector version publication and expand-time ordering (`tests/integration/test_gate2_5_trajectory_collector.py:1-19`, `tests/integration/test_gate2_5_trajectory_collector.py:93-141`, `tests/integration/test_gate2_5_trajectory_collector.py:148-216`) | `partial` — covers init/expand/post-train version publication and verifies `set_weight_version` occurs before `expand_sampler`, but does not run a full Ray pipeline + coordinator integration path. |
| `tests/integration/test_gate2_5_selective_sync.py` | F4.1 bucket format and F6.4 proper-subset NCCL broadcast lifecycle (`tests/integration/test_gate2_5_selective_sync.py:1-38`, `tests/integration/test_gate2_5_selective_sync.py:133-202`, `tests/integration/test_gate2_5_selective_sync.py:210-233`) | `partial` — exercises raw NCCL subgroup broadcast plus `BucketRecord` reconstruction, but does not call `ModelUpdateService`, `setup_collective_group()`, `broadcast_parameter()`, or `destroy_collective_group()` from the live transport stack (`tests/integration/test_gate2_5_selective_sync.py:65-70`, `tests/integration/test_gate2_5_selective_sync.py:136-202`). |
| `tests/integration/test_gate2_5_nccl_destroy.py` | Gate 2.5 NCCL destroy/re-init stability prerequisite for F4/F6 transport reuse (`tests/integration/test_gate2_5_nccl_destroy.py:1-16`, `tests/integration/test_gate2_5_nccl_destroy.py:66-76`, `tests/integration/test_gate2_5_nccl_destroy.py:82-143`, `tests/integration/test_gate2_5_nccl_destroy.py:150-211`) | `covered` — directly validates `destroy_model_parallel()` / `initialize_model_parallel()` loops, VRAM release, stale-handle behavior, and repeated-cycle stability. |
| `tests/integration/test_gate2_5_megatron_tp.py` | F4.3 owner-side CPU cache build and Gate 2.5 TP-shard offload/re-init (`tests/integration/test_gate2_5_megatron_tp.py:1-29`, `tests/integration/test_gate2_5_megatron_tp.py:171-185`, `tests/integration/test_gate2_5_megatron_tp.py:424-472`) | `partial` — covers real TP-sharded training, CPU cache build, VRAM release, and Megatron re-init; weight transfer now uses NCCL dynamic subset groups [0,2] and [1,3] per TP shard (shard 0: rank0→rank2, shard 1: rank1→rank3), migrated from gloo; does not yet call the live `ModelUpdateService` or `vllm_backend` receiver path (`tests/integration/test_gate2_5_megatron_tp.py:205-209`, `tests/integration/test_gate2_5_megatron_tp.py:203-253`). |
| `tests/integration/test_gate2_5_qwen_train_sync.py` | F4.3 build CPU cache on a real model and Gate 2.5 end-to-end hash verification (`tests/integration/test_gate2_5_qwen_train_sync.py:1-25`, `tests/integration/test_gate2_5_qwen_train_sync.py:166-177`, `tests/integration/test_gate2_5_qwen_train_sync.py:372-388`) | `partial` — uses a real Qwen model and verifies CPU-cache-driven transmission; transfer path now uses NCCL dynamic subset group [0,2,3] (rank 0 broadcasts to inference ranks 2,3), migrated from gloo; does not call the live `vllm_backend` receiver API (`tests/integration/test_gate2_5_qwen_train_sync.py:205-262`, `tests/integration/test_gate2_5_qwen_train_sync.py:321-383`). |
| `tests/integration/test_gate2_5_full.py` | Multi-pipeline isolation around F4 cache build/offload and repeated inference updates (`tests/integration/test_gate2_5_full.py:1-35`, `tests/integration/test_gate2_5_full.py:151-161`, `tests/integration/test_gate2_5_full.py:363-500`) | `partial` — validates offload/isolation and bit-exact pipeline A/B transfers; both weight-transfer phases now use NCCL dynamic subset groups: phase-A uses group [0,2,3] (rank0→ranks 2,3) and phase-B uses group [1,2,3] (rank1→ranks 2,3), migrated from gloo; gloo is retained for control-plane barriers and metadata exchange only; does not call the live selective transport stack (`tests/integration/test_gate2_5_full.py:180-248`, `tests/integration/test_gate2_5_full.py:181-278`, `tests/integration/test_gate2_5_full.py:299-313`). |

Uncovered or not fully covered requirements:
- The live selective transport stack is still not covered in one end-to-end run that goes through `ModelUpdateService`, the sender worker RPCs, and the receiver RPCs together; current Gate 2.5 coverage is split across targeted IPC, bucket-guard, trajectory-collector, and NCCL subgroup tests (`rlix/pipeline/model_update_service.py:258-463`, `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1280-1492`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361-507`).
