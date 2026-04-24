# ROLL vs NeMo RL Port Analysis for Feature 4 and Feature 6

The requested source paths were given as `rlix/external/...`, but in this workspace the repo root is already `.../rlix`, so the files that exist and were read are the corresponding `external/...` paths listed in Section 5. No requested source file was missing.

## (a) ROLL's exact serialization format for `cpu_serialize` vs `cuda_ipc`

### Shared bucket layout before transport

ROLL first converts each named tensor bucket into a single flat `torch.int8` buffer plus per-tensor metadata. `_bucket_named_tensors()` flattens every tensor with `tensor.flatten().view(torch.int8)`, concatenates those byte views with `torch.cat(..., dim=0)`, and emits one metadata dict per tensor with these exact fields:

- `name`: `str`
- `shape`: `list[int]`
- `dtype`: original `torch.dtype`
- `start_idx`: `int`
- `end_idx`: `int`
- `numel`: `int`, where this is the length of the flattened `torch.int8` slice for that tensor

This means the shared bucket itself is a 1-D `torch.int8` tensor whose length is the sum of all `meta["numel"]` values in the bucket. The cache builder stores each bucket as `(tensors_meta, bucket)` after first converting gathered weights to contiguous CPU tensors. (Observed at `external/ROLL/roll/utils/send_recv_utils.py:214-247` and `external/ROLL/roll/distributed/strategy/megatron_strategy.py:1966-1974`.)

### `cpu_serialize` path

On the sender side, ROLL serializes one Python dict with exactly two top-level fields:

- `bucket`: the flat 1-D `torch.int8` CPU tensor, made contiguous with `cpu_bucket.contiguous()`
- `tensors_meta`: the metadata list described above

That dict is serialized with `torch.save(..., io.BytesIO())`, and the resulting `bytes` blob is sent to colocated inference workers. (Observed at `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2218-2225`.)

On the receiver side, ROLL deserializes the bytes with `torch.load(io.BytesIO(raw), weights_only=True)`. If the recovered `bucket` is not already CUDA, it pins the CPU buffer, copies the whole flat bucket to GPU once with `bucket.to(device=self.device, non_blocking=True)`, synchronizes the CUDA stream, and then reconstructs tensors by slicing the flat byte bucket and reinterpreting each slice as:

- bytes range: `bucket[meta["start_idx"]:meta["end_idx"]]`
- dtype cast: `.view(meta["dtype"])`
- shape restore: `.reshape(torch.Size(meta["shape"]))`

That reconstruction is performed by `named_tensors_from_bucket()`, which returns the recovered `(name, tensor)` pairs. (Observed at `external/ROLL/roll/third_party/vllm/worker.py:748-780` and `external/ROLL/roll/utils/send_recv_utils.py:242-247`.)

### `cuda_ipc` path

The logical payload shape is the same as `cpu_serialize`: ROLL still serializes a dict with exactly:

- `bucket`
- `tensors_meta`

The difference is that `bucket` is first staged to GPU with `gpu_bucket = cpu_bucket.to(current_platform.device_type).contiguous()`, then serialized with `MultiprocessingSerializer.serialize(...)` after `monkey_patch_torch_reductions()`. So the payload is a pickled dict whose `bucket` entry is a CUDA tensor exported through PyTorch multiprocessing/CUDA-IPC reducers, not a CPU tensor serialized by `torch.save`. (Observed at `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2199-2205` and `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2226-2234`.)

`monkey_patch_torch_reductions()` is part of the format contract here: it overrides PyTorch's CUDA tensor reducers so the serialized tensor reducer stores a GPU UUID instead of a raw device index, and the rebuild path maps that UUID back to the local device index on the receiver. (Observed at `external/ROLL/roll/utils/send_recv_utils.py:160-207`.)

On the receiver side, ROLL calls `monkey_patch_torch_reductions()` again, then `pickle.loads(raw)`. If the imported `bucket` is already CUDA, the CPU-to-GPU copy path is skipped. Reconstruction of individual tensors is otherwise identical to the `cpu_serialize` path: slice by `start_idx/end_idx`, cast with `.view(meta["dtype"])`, then reshape with `meta["shape"]`. (Observed at `external/ROLL/roll/third_party/vllm/worker.py:760-780` and `external/ROLL/roll/utils/send_recv_utils.py:242-247`.)

## (b) How the NeMo port differs structurally from ROLL's pattern

1. The IPC wire format is different. ROLL sends serialized bytes for a two-field dict `{"bucket": ..., "tensors_meta": ...}` and the receiver deserializes those bytes; the NeMo port sends a Python dict with `param_names`, `shapes`, `dtypes`, `offsets`, `used_bytes`, and `cpu_uint8_bucket`, then rebuilds a `BucketRecord` from those fields. That is a different transport contract, not just a different implementation detail. (ROLL sender/receiver: `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2218-2234`, `external/ROLL/roll/third_party/vllm/worker.py:748-780`, `external/ROLL/roll/utils/send_recv_utils.py:214-247`; NeMo sender/receiver: `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1351-1363`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:374-399`.)

2. The NeMo port does not implement the `cuda_ipc` branch that ROLL uses. In the NeMo sender, `model_update_transport` is accepted and even commented as selecting `cpu_serialize` vs `cuda_ipc`, but the code always builds the same CPU-bucket payload and never branches into a CUDA-IPC serializer. In the NeMo receiver, the docstring explicitly says only `cpu_serialize` is supported, and the implementation never does the ROLL-style `torch.load` vs `pickle.loads` split. (Observed at `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1345-1363` and `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:378-413`.)

3. ROLL uses one shared bucket schema across cache build, IPC, and reconstruction; the NeMo port splits transport formats. ROLL's cache stores `(tensors_meta, bucket)` and its receiver reconstructs tensors with the shared `named_tensors_from_bucket()` helper. The NeMo port stores `BucketRecord` objects for the IPC path, but its NCCL receive path reconstructs a separate aligned packed layout by recomputing `total_bytes` and slicing a monolithic `recv_buf` using `calculate_aligned_size()`. That means the port does not have the single shared "bucket + tensors_meta" data model that ROLL uses end to end. (ROLL: `external/ROLL/roll/distributed/strategy/megatron_strategy.py:1968-1974`, `external/ROLL/roll/utils/send_recv_utils.py:214-247`; NeMo cache build and broadcast receive: `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1178-1215`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:451-485`.)

4. The sender-side verification point is different. ROLL computes sender stats while the gathered tensors are still in the cache-building path and stores those stats by cache version, then returns the stats associated with the active cached version after replay. The NeMo port does not store per-version sender stats in the cache path shown here; instead it computes one flat `{sum,max,min}` dict after transport by re-iterating the current exporter state. That is structurally weaker than ROLL's "stats are attached to the exact cached payload version" pattern. (ROLL: `external/ROLL/roll/distributed/strategy/megatron_strategy.py:1940-1965`, `external/ROLL/roll/distributed/strategy/megatron_strategy.py:1979-1992`, `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2359-2378`; NeMo: `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1404-1419`.)

5. ROLL's selective-sync path includes an explicit LoRA stage; the NeMo port leaves that path unimplemented in the inspected code. ROLL replays base buckets, then adapter buckets, then calls `add_lora` on each target worker. In the NeMo port, `adapters_to_sync` is marked unused/reserved in the sender, and `is_lora` is marked reserved/not yet used in both receive methods. (ROLL: `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2106-2140`, `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2307-2349`, `external/ROLL/roll/third_party/vllm/worker.py:141-222`; NeMo: `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1279-1300`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:379-380`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:435-436`.)

6. ROLL's base-weight NCCL receive path is streaming; the NeMo port materializes a full bucket buffer before loading. ROLL reloads the model and then receives one tensor at a time into a generator passed to `load_weights()`. The NeMo port allocates one full `recv_buf`, reconstructs the full bucket into a Python `weights` list, and only then calls `load_weights`. That is a real structural departure from ROLL's memory-capped receive path. (ROLL: `external/ROLL/roll/third_party/vllm/worker.py:696-721`; NeMo: `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:451-485`.)

## (c) What needs to change in the NeMo port to match ROLL's battle-tested pattern

### `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py`

- `MegatronPolicyWorkerImpl.build_latest_bucket_cache`
  - Stop caching `BucketRecord`-style payloads as the transport source of truth.
  - Cache the ROLL-style pair `(tensors_meta, cpu_bucket)` where `cpu_bucket` is the flat 1-D `torch.int8` buffer and `tensors_meta` uses the same field set ROLL uses: `name`, `shape`, `dtype`, `start_idx`, `end_idx`, `numel`.
  - Compute sender verification stats during cache build and store them by cache version, the same way ROLL stores `_cache_stats` keyed to the cached version. The current post-transport stats block should not be the primary source of truth if the goal is ROLL parity. (Reference pattern: `external/ROLL/roll/distributed/strategy/megatron_strategy.py:1915-1992` and `external/ROLL/roll/utils/send_recv_utils.py:214-271`.)

- `MegatronPolicyWorkerImpl.selective_sync_active_cache`
  - Replace the current `BucketRecord` payload construction with ROLL's per-bucket transport loop.
  - For IPC targets, serialize exactly one payload per bucket with:
    - `cpu_serialize`: `torch.save({"bucket": cpu_bucket.contiguous(), "tensors_meta": tensors_meta}, buf)`
    - `cuda_ipc`: stage `gpu_bucket`, call `monkey_patch_torch_reductions()`, then `MultiprocessingSerializer.serialize({"bucket": gpu_bucket, "tensors_meta": tensors_meta})`
  - Send a rank-indexed payload list sized to `tgt_num_gpus_per_worker`, matching ROLL's receiver contract.
  - Derive NCCL broadcast metadata from `named_tensors_from_bucket(gpu_bucket, tensors_meta)` rather than the current custom `BucketRecord` field set.
  - Return cached version stats, not only a post-hoc flattened state dict.
  - If full ROLL parity is required, stop leaving `adapters_to_sync` unused and port the adapter replay plus `add_lora` registration stage. (Reference pattern: `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2047-2378`.)

- `MegatronPolicyWorkerImpl.setup_collective_group` and `MegatronPolicyWorkerImpl.destroy_collective_group`
  - Align the sender-side group lifecycle with the transport loop above so teardown stays inside the same lock scope as cache lookup and bucket replay, matching ROLL's sequencing. The current code already tears down under the lock; this should remain coupled to the new transport contract. (Reference pattern: `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2095-2100` and `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2351-2378`.)

### `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py`

- `VllmInternalWorkerExtension.update_parameter_in_bucket`
  - Change the IPC receive contract to match ROLL: accept the serialized bytes payload list, select `raw = serialized_named_tensors[self.rank]`, and branch on `model_update_transport`.
  - For `cpu_serialize`, use `torch.load(io.BytesIO(raw), weights_only=True)`.
  - For `cuda_ipc`, call `monkey_patch_torch_reductions()` and `pickle.loads(raw)`.
  - Reconstruct tensors with the same `named_tensors_from_bucket(bucket, tensors_meta)` logic ROLL uses.
  - Keep the current CPU-bucket whole-copy-to-GPU optimization only as the fallback when the recovered bucket is not already CUDA.
  - If LoRA parity is required, actually use `is_lora` to stage adapter tensors instead of treating it as reserved. (Reference pattern: `external/ROLL/roll/third_party/vllm/worker.py:732-780` and `external/ROLL/roll/utils/send_recv_utils.py:160-247`.)

- `VllmInternalWorkerExtension.broadcast_parameter`
  - Rework the base-weight path to follow ROLL's streaming receive pattern: reload model memory first, then receive one tensor at a time and pass a generator to `load_weights()`.
  - Reserve the batched async receive path for the LoRA case, matching ROLL's split between base weights and LoRA payloads.
  - If the port keeps the current packed-bucket NCCL receive path instead, it will remain structurally different from ROLL even after IPC parity is fixed. (Reference pattern: `external/ROLL/roll/third_party/vllm/worker.py:649-730`.)

- `VllmInternalWorkerExtension.verify_model`
  - Match ROLL's verification structure by accepting and comparing the versioned sender stats schema that distinguishes at least base vs LoRA stages, instead of flattening the whole live state dict into a single flat stats dict. (Reference pattern: `external/ROLL/roll/third_party/vllm/worker.py:279-334` and `external/ROLL/roll/distributed/strategy/megatron_strategy.py:2359-2378`.)

### What does not appear to need a new runtime implementation first

- The bucket-size guard itself already exists in the NeMo port via `_rlix_get_bucket_size_bytes()` and `_rlix_check_vram()`. What is missing from the inspected code is test coverage, not the guard implementation. (Observed at `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2004-2101`.)

## (d) Which uncovered item is most critical to implement first

`F6.3 cuda_ipc` is the most critical item to implement first.

The reason is simple: in the inspected NeMo selective-sync path, `model_update_transport` already exists as a runtime parameter, the sender comments claim it selects `cpu_serialize` vs `cuda_ipc`, and the receiver takes the parameter too, but there is no actual CUDA-IPC sender branch and no ROLL-style CUDA-IPC receiver branch. That means the transport contract is incomplete at runtime right now, not merely undertested. By contrast, `F4.4 bucket-size guard test` targets guard code that already exists, `ModelUpdateService` end-to-end coverage is important but still only validates whatever transport path exists, and I do not see trajectory-collector logic in these selective-sync entry points at all, so `F6.6 trajectory collector` is less immediate than fixing the missing transport branch itself. (Observed at `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1271-1419`, `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361-413`, and `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:2004-2101`.)

## (e) File paths cited and exact line ranges read

- `external/ROLL/roll/distributed/strategy/megatron_strategy.py`
  - Read ranges: `1-300`, `301-600`, `601-900`, `901-1200`, `1201-1500`, `1501-1800`, `1801-2100`, `2101-2400`, `2401-2654`

- `external/ROLL/roll/third_party/vllm/worker.py`
  - Read ranges: `1-300`, `301-600`, `601-811`

- `external/ROLL/roll/utils/send_recv_utils.py`
  - Read ranges: `1-220`, `221-362`

- `external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py`
  - Read ranges: `1-300`, `301-564`

- `external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py`
  - Read ranges: `1-300`, `301-600`, `601-900`, `901-1200`, `1201-1500`, `1501-1800`, `1801-2108`
