# Implementation Review Round 2 — 2026-04-24

## 1. update_parameter_in_bucket (vllm_backend.py)
### 1a. Rank mask
Verdict: PASS
Evidence: `rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:390-399`
Detail: The rank filter now reads `local_rank = getattr(self, "rank", None)` and checks that value against `ipc_local_ranks`, so it no longer uses a constant rank in this code path.

### 1b. Zero-copy cuda_ipc path
Verdict: PASS
Evidence: `rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:404-426`
Detail: The `cuda_ipc` branch rebuilds a GPU buffer from the IPC handle and slices/views that GPU buffer directly into per-parameter tensors, with no intermediate CPU tensor and no `.cpu()` or `.numpy()` call in that branch.

## 2. build_latest_bucket_cache (megatron_policy_worker.py)
### 2a. Oversized-tensor guard
Verdict: PASS
Evidence: `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1201-1209`
Detail: The oversized-tensor check runs before the `current_batch` flush condition, so it also fires for the first tensor in a new bucket instead of silently bypassing the limit.

### 2b. Guard correctness
Verdict: PASS
Evidence: `rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1204-1218`
Detail: Oversized tensors raise before append, while valid tensors either append directly or trigger a flush-then-append sequence, so the guard does not drop tensors or split them incorrectly.

## 3. _expand_workers ordering (full_finetune_pipeline.py)
### 3a. set_weight_version before expand_sampler
Verdict: PASS
Evidence: `rlix/rlix/pipeline/full_finetune_pipeline.py:549-557`
Detail: `_expand_workers` performs `ray.get(_tc.set_weight_version.remote(...))` before calling `expand_sampler.remote(...)`, so version publication happens before routing activation.

### 3b. Async gap risk
Verdict: PASS
Evidence: `rlix/rlix/pipeline/full_finetune_pipeline.py:553-557`
Detail: There is no intervening `await` or fire-and-forget async step between the blocking `ray.get` on `set_weight_version` and the subsequent `expand_sampler` call, so this path does not leave a gap where routing could start first.

## 4. Test files
### 4a. test_gate2_5_cuda_ipc.py
Verdict: FAIL
Evidence: `rlix/tests/integration/test_gate2_5_cuda_ipc.py:49-53,81-85,140-146,166-174`
Detail: The requested path `rlix/tests/test_gate2_5_cuda_ipc.py` is missing; the corresponding integration test only loads `bucket_cache`, defines inline CUDA IPC helper shims, and reconstructs/unpacks the buffer directly, so it never imports or invokes the real `update_parameter_in_bucket` function.

### 4b. test_gate2_5_bucket_size_guard.py
Verdict: FAIL
Evidence: `rlix/tests/integration/test_gate2_5_bucket_size_guard.py:132-137,149-160,166-193`
Detail: The requested path `rlix/tests/test_gate2_5_bucket_size_guard.py` is missing; the corresponding integration test never calls `build_latest_bucket_cache` and instead reimplements the oversized-tensor and host-RAM checks inline.

### 4c. test_gate2_5_trajectory_collector.py
Verdict: FAIL
Evidence: `rlix/tests/integration/test_gate2_5_trajectory_collector.py:35-58,73-80,187-216`
Detail: The requested path `rlix/tests/test_gate2_5_trajectory_collector.py` is missing; the corresponding integration test uses fake collector/pipeline stand-ins and a source-text ordering check, so it does not execute the real trajectory collection code path.

## Summary
- Clean: `update_parameter_in_bucket` now masks with the worker’s own `self.rank`-based identity and its `cuda_ipc` branch stays GPU-only.
- Clean: `build_latest_bucket_cache` now fails fast on a single oversized tensor before bucket assembly and preserves correct flush/append behavior for valid tensors.
- Clean: `_expand_workers` publishes the weight version synchronously before `expand_sampler`, with no async gap in between.
- Needs a fix: the requested test paths under `rlix/tests/` do not exist; the actual files live under `rlix/tests/integration/`.
- Needs a fix: the CUDA IPC test does not call the real `update_parameter_in_bucket` path.
- Needs a fix: the bucket-size guard test does not call the real `build_latest_bucket_cache` path.
- Needs a fix: the trajectory-collector test does not execute the real production pipeline/collector path.
