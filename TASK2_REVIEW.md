# Task 2 Review: Gate 2.5 (F4, F6-transport)

## Verdict

**No. Task 2 is not done to the Gate 2.5 bar described in the plan.**

The strongest blockers I found are:

1. The planned same-GPU IPC transport is not implemented end-to-end. The plan treats CUDA IPC as a correctness requirement for overlap cases in [nemorl-port-plan.md](/Users/zhenyulin/Downloads/nemorl-port-plan.md:316), but [IMPLEMENTATION.md](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/IMPLEMENTATION.md:224) and [vllm_backend.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361) both say the receiver only supports `cpu_serialize`, and `update_parameter_in_bucket()` never branches on `model_update_transport`.
2. The Gate 2.5 tests do not validate the actual `ModelUpdateService` + `vllm_backend` NCCL broadcast path. The closest test, [test_gate2_5_selective_sync.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:58), imports only `bucket_cache.py` helpers and then hand-rolls `dist.new_group()` / `dist.broadcast()` directly in the test body at [136-198](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:136).
3. The gate artifacts disagree with each other. [IMPLEMENTATION.md](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/IMPLEMENTATION.md:365) still describes Part 2 as a 2-rank / 2-GPU test, but the current [test_gate2_5_selective_sync.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:245) skips when `world_size < 4`. The scripted gate runner still invokes it with `--nproc-per-node=2` in [run_gate2_5.sh](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/run_gate2_5.sh:42).

## Scope Items

| Scope item | Status | Findings |
|---|---|---|
| 1. PP collective gather | **Mostly yes, but indirect** | [build_latest_bucket_cache()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1153) explicitly says all PP/TP/EP ranks must participate and non-owners must drain the generator, and the code does call the iterator on every worker at [1192-1196](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1192). The actual gather primitive is indirect through `self.megatron_bridge.export_hf_weights(...)` in [_iter_params_with_optional_kv_scales()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1012), so I did not find a direct `gather_all_hf_weights()` call in the files reviewed. |
| 2. Cache owner storage | **Yes** | The cache-owner predicate is explicit in [_rlix_is_cache_owner()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1131). Only the owner stores buckets in [build_latest_bucket_cache()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1213), and only the owner promotes in [promote_active_checkpoint()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1245). `ModelUpdateService` also selects one global sender by `(pp, dp, tp, cp) == 0` in [_select_global_sender_rank()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/model_update_service.py:120). |
| 3. Bucket format | **Yes** | The canonical cache record exists as `BucketRecord` with `param_names`, `shapes`, `dtypes`, `offsets`, `used_bytes`, and `cpu_uint8_bucket` in [bucket_cache.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/bucket_cache.py:69). Packing is centralized in [_bucket_named_tensors()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/bucket_cache.py:96) and unpacking in [unpack_bucket_record()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/bucket_cache.py:164). The sender reuses the same record fields for both IPC payloads and NCCL metadata in [selective_sync_active_cache()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1348). |
| 4. Selective sync | **Yes** | The service targets explicit DP ranks in [sync_selected_workers()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/model_update_service.py:258), and the pipeline calls it only for the ranks being expanded in [_expand_workers()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/full_finetune_pipeline.py:513) or for the current active ranks in [sync_base_weights_to_active()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/coordinator.py:507). Only the global owner actually transfers; non-owners return immediately in [selective_sync_active_cache()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1312). |
| 5. IPC + dynamic NCCL group routing | **Partial / no** | Dynamic NCCL routing is implemented: [_build_comm_plan_for_sender()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/model_update_service.py:130) classifies each target device into IPC or broadcast, builds `ipc_targets` plus `broadcast_local_ranks_by_dp_rank`, and [sync_selected_workers()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/rlix/pipeline/model_update_service.py:327) only sets up temporary NCCL groups for `tgt_ranks_in_group`. But the IPC half is not the planned transport. The sender passes a Python `payload` dict by Ray RPC in [selective_sync_active_cache()](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/policy/workers/megatron_policy_worker.py:1351), not a ZMQ/CUDA-IPC transport object. The receiver method documents `cpu_serialize` as the only supported mode in [vllm_backend.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:378), and its implementation at [361-412](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:361) never branches on `model_update_transport`. [IMPLEMENTATION.md](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/IMPLEMENTATION.md:227) also states that `"cuda_ipc"` is not implemented on the receiver, and repeats that deferral at [288](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/IMPLEMENTATION.md:288). |
| 6. Receiver API on `vllm_backend` | **Yes for API surface; incomplete for transport parity** | The six receiver methods required by the plan exist in [vllm_backend.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_backend.py:316): `setup_collective_group`, `update_parameter_in_bucket`, `broadcast_parameter`, `destroy_collective_group`, `verify_model`, and `finalize_weight_update`. They are exposed as Ray-callable pass-throughs in [vllm_generation.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/external/NeMo/nemo_rl/models/generation/vllm/vllm_generation.py:858). The API surface is there; the transport gap is the missing receiver-side CUDA IPC behavior from item 5. |

## `test_gate2_5_selective_sync.py`

### Does it use a proper subset NCCL group?

**Yes.**

The file defines `SYNC_RANKS = [SENDER_RANK] + INFER_RANKS` at [84](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:84), creates the subgroup with `dist.new_group(ranks=SYNC_RANKS, backend="nccl")` at [136](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:136), and skips entirely when `world_size < 4` at [245-250](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:245). So it does avoid the `world == group` 2-GPU case that the user called out.

### Does it correctly test the real NCCL broadcast transport path used by Task 2?

**No. It tests a subgroup-NCCL smoke path, not the actual Task 2 implementation path.**

What it does test:

- Raw NCCL subgroup creation and raw `dist.broadcast()` on the packed uint8 bucket at [136-148](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:136).
- Receiver-side `BucketRecord` reconstruction and `unpack_bucket_record()` usage at [176-189](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:176).
- Repeated create/broadcast/destroy cycles with a proper subset group.

What it does **not** test:

- It does not import or call `ModelUpdateService`; the only dynamically loaded module is `bucket_cache.py` at [65-70](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:65).
- It does not call `setup_collective_group`, `broadcast_parameter`, `update_parameter_in_bucket`, or `destroy_collective_group` from the actual sender/receiver code.
- It reconstructs metadata locally from deterministic weights at [163-183](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:163) instead of exercising the real receiver API contract.

So the subset-group topology is correct, but the test is not an end-to-end verification of the implemented Task 2 NCCL transport path.

## Gate 2.5 Evidence Gaps

These files make the gate evidence weaker than the plan requires:

- [test_gate2_5_megatron_tp.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_megatron_tp.py:18) explicitly says the weight transfer is a **world-gloo broadcast**, and its broadcast helper is gloo/CPU-only at [204-217](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_megatron_tp.py:204).
- [test_gate2_5_qwen_train_sync.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_qwen_train_sync.py:13) claims dynamic NCCL in the header, but its `selective_sync()` docstring says the buckets are broadcast via **gloo (CPU)** at [214-217](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_qwen_train_sync.py:214), and the test initializes the default process group with `backend="gloo"` at [338-343](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_qwen_train_sync.py:338).
- [test_gate2_5_full.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_full.py:12) is also gloo-only for weight transfer, with gloo broadcast helpers at [181-197](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_full.py:181) and a gloo default process group at [329-345](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_full.py:329).
- [run_gate2_5.sh](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/run_gate2_5.sh:48) still runs Part 2 with 2 processes, which conflicts with the current 4-GPU requirement in [test_gate2_5_selective_sync.py](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/tests/integration/test_gate2_5_selective_sync.py:245).
- [IMPLEMENTATION.md](/Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rilk/rlix/IMPLEMENTATION.md:367) still describes that same test as a 2-rank / 2-GPU NCCL test, which no longer matches the file.

## Final Call

**Task 2 should be treated as not complete for Gate 2.5.**

What is present:

- CPU bucket cache ownership/versioning is in place.
- Selective target-worker sync orchestration exists.
- Dynamic NCCL subgroup routing exists.
- Receiver API surface exists on `vllm_backend` / `vllm_generation`.

What is still missing for a true Gate 2.5 pass:

- The planned same-GPU CUDA IPC path is still deferred on the receiver side.
- The gate tests do not prove the actual `ModelUpdateService` + `vllm_backend.broadcast_parameter()` NCCL broadcast path.
- The automated gate runner and implementation notes are out of sync with the current subset-group test requirements.
