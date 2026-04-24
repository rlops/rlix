# Implementation Review: F6.3 / F4.4 / F6.6

Note: the task's repo-local spec path `rlix/external/NeMo/nemo_rl/docs/nemorl-port-plan.md` is not present in this checkout. The spec line citations below therefore use the available local copy at `/Users/zhenyulin/Downloads/nemorl-port-plan.md`.

## 6.3

### Spec Compliance

- The sender does hold the cache lock across active-cache lookup, per-bucket transport, sender-side stream sync, and sender-side NCCL teardown, which matches the plan's lock-span invariant for selective sync (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:397-402`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1326-1422`).
- The implementation does have distinct `cpu_serialize` and `cuda_ipc` branches on both sender and receiver sides (`rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1345-1377`; `rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:399-408`).
- It is not fully spec-compliant in two places. First, the plan says the colocated path should reuse the existing ZMQ IPC path (`stream_weights_via_ipc_zmq` / `update_weights_via_ipc_zmq`) (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:318-321,344-345`), but the current sender bypasses those functions and pushes Python payload dicts directly over Ray RPC via `update_parameter_in_bucket.remote(...)` (`rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1379-1383`). Second, the plan describes CUDA IPC as rebuilding the CUDA tensor and slicing/views from that GPU buffer (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:320,410-411`), while the current receiver immediately copies the rebuilt CUDA buffer back to CPU with `buf_gpu.cpu()` and then copies the unpacked tensors back to GPU (`rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:399-420`).

### Correctness Bugs

- `update_parameter_in_bucket()` applies the IPC mask against `torch.distributed.get_rank()` (or `0` when distributed is uninitialized) instead of the local-rank identity that the comm plan carries. The plan's contract is explicitly `self.rank in ipc_local_ranks` (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:406-412`), but the code uses `local_rank = torch.distributed.get_rank() if ... else 0; if local_rank not in ipc_local_ranks: return` (`rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:390-394`). This is only correct if those two rank notions coincide [INFERRED]; otherwise a mixed IPC/broadcast worker can skip or double-apply a bucket.

### Test Coverage

- `test_gate2_5_cuda_ipc.py` does validate the low-level CUDA IPC primitives: it calls `get_handle_from_tensor()` in the sender, `rebuild_cuda_tensor_from_ipc()` in the receiver, rebuilds a `BucketRecord`, and checks hashes across three cycles (`rlix/tests/integration/test_gate2_5_cuda_ipc.py:84-103`; `rlix/tests/integration/test_gate2_5_cuda_ipc.py:142-170`; `rlix/tests/integration/test_gate2_5_cuda_ipc.py:183-203`).
- It does not exercise the production selective-sync path. The test never calls `selective_sync_active_cache()` or `update_parameter_in_bucket()`, and it does not involve `ModelUpdateService`, comm-plan masks, Ray RPC dispatch, `_cache_lock`, or NCCL teardown (`rlix/tests/integration/test_gate2_5_cuda_ipc.py:76-124`; `rlix/tests/integration/test_gate2_5_cuda_ipc.py:135-191`; compare `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1271-1423` and `rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361-431`). It therefore verifies only a subset of the spec and would not catch the live receiver-mask bug above.

### Verdict

FAIL. The transport branches exist, but the receiver-side rank mask does not follow the plan's `self.rank` contract and the integration test does not execute the production sender/receiver path (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:406-412`; `rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:390-394`; `rlix/tests/integration/test_gate2_5_cuda_ipc.py:76-191`).

## 4.4

### Spec Compliance

- The explicit-configuration requirement is implemented: `_rlix_get_bucket_size_bytes()` reads `worker.cfg["rlix"]["bucket_size_bytes"]` or `RLIX_BUCKET_SIZE_BYTES`, and raises if neither is set (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:337,343`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2030-2082`).
- Init-time capacity guards also exist in the worker. `build_latest_bucket_cache()` calls `_rlix_check_vram()` during the base-cache build and performs a host-RAM check against `2 * total_bytes` after building the base cache (`rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1183-1186`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1215-1243`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2085-2115`).
- The requested `vllm_backend.py` change is not the F4.4 guard implementation. The F4.4 capacity logic lives in `megatron_policy_worker.py`, while `vllm_backend.py:update_parameter_in_bucket()` is part of the receiver transport path for weight application (`rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361-431`).

### Correctness Bugs

- The configured bucket-size cap can be violated by a single oversized tensor. `build_latest_bucket_cache()` flushes only when `current_batch` is already non-empty and `current_bytes + nbytes > bucket_size_bytes`; if the first tensor in a new bucket is itself larger than the configured limit, it is appended anyway and the resulting bucket exceeds `bucket_size_bytes` with no error (`rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1198-1208`). That contradicts the plan's explicit staging-capacity guard for `bucket_size_bytes` (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:342-343`).

### Test Coverage

- Tests 1 and 2 really do exercise `_rlix_get_bucket_size_bytes()` for the missing-config and env-var paths (`rlix/tests/integration/test_gate2_5_bucket_size_guard.py:54-80`; `rlix/tests/integration/test_gate2_5_bucket_size_guard.py:90-108`).
- The host-RAM "trigger" test does not execute a production guard. It tries to import `_rlix_host_ram_check`, but no such symbol exists in `megatron_policy_worker.py`, so the test logs `SKIP` and returns early (`rlix/tests/integration/test_gate2_5_bucket_size_guard.py:150-159`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1153-1243`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2030-2115`). Even if that import succeeded, the asserted failure is manually reimplemented arithmetic in the test body, not a call into production code (`rlix/tests/integration/test_gate2_5_bucket_size_guard.py:161-179`).
- The chosen synthetic model is too small for the claimed failure case. `torch.randn(256, 256 * 6)` uses the default `float32` dtype, so it is about 1.5 MiB, and `2 * total_bytes` stays below the mocked 8 MiB budget; the test therefore only prints a note instead of asserting that the guard fired (`rlix/tests/integration/test_gate2_5_bucket_size_guard.py:139-181`).
- The file never exercises `_rlix_check_vram()`, even though the spec requires a staging-VRAM guard and the test docstring claims that check is in scope (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:343`; `rlix/tests/integration/test_gate2_5_bucket_size_guard.py:3-12`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1184-1186`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2085-2115`).

### Verdict

FAIL. The explicit-config pieces exist, but a single large tensor can bypass the configured bucket-size limit, and the integration test does not actually execute the host-RAM or VRAM guard paths (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:342-343`; `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1198-1208`; `rlix/tests/integration/test_gate2_5_bucket_size_guard.py:150-181`).

## 6.6

### Spec Compliance

- The post-train active-refresh path is aligned with the plan. After coordinator sync returns, the pipeline finalizes the synced workers, updates `_current_weight_version`, publishes it to the trajectory collector, and only then releases training GPUs (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:477-491`; `/Users/zhenyulin/Downloads/nemorl-port-plan.md:536-543`; `rlix/rlix/pipeline/coordinator.py:507-550`; `rlix/rlix/pipeline/full_finetune_pipeline.py:1112-1137`).
- The expand path is not aligned with the plan. The plan says `_expand_workers()` should wake the target ranks, sync them, finalize them, publish the version, and only then activate routing (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:588-609`). The current implementation instead syncs first, finalizes second, calls `expand_sampler(...)` third, and only after that publishes the trajectory-collector version (`rlix/rlix/pipeline/full_finetune_pipeline.py:529-555`). The local docstring also documents routing update before version publication (`rlix/rlix/pipeline/full_finetune_pipeline.py:516-520`).

### Correctness Bugs

- In the expand path, trajectory-collector publication happens after `expand_sampler()` rather than before activation. The plan makes version publication part of the pre-activation sequence (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:602-608`), but the current code publishes only after the train/val schedulers have already expanded (`rlix/rlix/pipeline/full_finetune_pipeline.py:545-555`). That means newly expanded ranks can be exposed before the collector sees the corresponding weight version [INFERRED].

### Test Coverage

- `test_gate2_5_trajectory_collector.py` never imports or calls the production pipeline/coordinator code. Each check is a local simulation with a fake collector or a hand-written `events` list (`rlix/tests/integration/test_gate2_5_trajectory_collector.py:35-58`; `rlix/tests/integration/test_gate2_5_trajectory_collector.py:69-180`).
- The ordering test is trivially true: it appends `["sync", "finalize", "set_version"]` in that order and then asserts that exact literal list (`rlix/tests/integration/test_gate2_5_trajectory_collector.py:148-165`). It does not touch `_expand_workers()` or the post-train hook, so it cannot detect the live expand-path ordering mismatch in `full_finetune_pipeline.py` (`rlix/rlix/pipeline/full_finetune_pipeline.py:529-555`).
- The publish-site tests likewise call `proxy.remote(...)` on local variables instead of exercising `_get_trajectory_collector()`, `sync_base_weights_to_active()`, or `_expand_workers()` in the actual pipeline (`rlix/tests/integration/test_gate2_5_trajectory_collector.py:93-141`; compare `rlix/rlix/pipeline/full_finetune_pipeline.py:488-492`; `rlix/rlix/pipeline/full_finetune_pipeline.py:550-555`; `rlix/rlix/pipeline/full_finetune_pipeline.py:1126-1130`).

### Verdict

FAIL. The post-train path is good, but the expand path does not follow the specified publish-before-activate ordering, and the provided test is almost entirely synthetic so it would pass even with that live mismatch (`/Users/zhenyulin/Downloads/nemorl-port-plan.md:588-609`; `rlix/rlix/pipeline/full_finetune_pipeline.py:529-555`; `rlix/tests/integration/test_gate2_5_trajectory_collector.py:93-165`).
