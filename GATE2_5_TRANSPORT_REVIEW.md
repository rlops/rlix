# Gate 2.5 Transport Review

Reviewed files:

- `rlix/tests/integration/test_gate2_5_feature6.py`
- `rlix/tests/integration/test_gate2_5_full.py`
- `rlix/tests/integration/test_gate2_5_megatron_tp.py`
- `rlix/tests/integration/test_gate2_5_nccl_destroy.py`
- `rlix/tests/integration/test_gate2_5_qwen_train_sync.py`
- `rlix/tests/integration/test_gate2_5_selective_sync.py`

Spec anchors used for compliance judgments:

- `/Users/zhenyulin/Downloads/nemorl-port-plan.md:391`
  - "`tp=2` and overlap with at least one TP rank on a different GPU requires the broadcast path and therefore a dynamic NCCL group."
- `/Users/zhenyulin/Downloads/nemorl-port-plan.md:1196-1201`
  - `sync_selected_workers` must verify the NCCL broadcast transport path for cross-GPU TP ranks, then run 3+ steps with no NCCL errors, no VRAM leak, and correct weights.

## Summary Table

| Test file | Transport used for weight broadcast | Spec requires NCCL? | Compliant? |
| --- | --- | --- | --- |
| `test_gate2_5_selective_sync.py` | NCCL dynamic subgroup `[0,2,3]` with gloo-only barriers | Yes | Yes |
| `test_gate2_5_feature6.py` | NCCL dynamic group `[0,1]` on 2 GPUs, or `[0,last]` on larger worlds | No for the cited Gate 2.5 `tp>1` cross-GPU TP case | No as a Gate 2.5 transport proxy |
| `test_gate2_5_megatron_tp.py` | Gloo world-group CPU broadcasts from rank 0, then rank 1 | Yes | No |
| `test_gate2_5_qwen_train_sync.py` | Gloo world/default-group CPU broadcasts from rank 0 | Yes | No |
| `test_gate2_5_full.py` | Gloo world/default-group CPU broadcasts in both sync phases | Yes | No |
| `test_gate2_5_nccl_destroy.py` | No weight broadcast in file | No transport step in this file | N/A |

## Per-File Review

### `rlix/tests/integration/test_gate2_5_selective_sync.py`

- Transport used:
  - `dist.new_group(ranks=SYNC_RANKS, backend="nccl")` at `136`.
  - Sender and receivers use `dist.broadcast(..., group=dynamic_group)` on CUDA tensors at `147-148` and `155-160`.
  - World/barrier coordination is split to `gloo_world = dist.new_group(..., backend="gloo")` at `266`.
- Group structure:
  - File-level constants define `SYNC_RANKS = [0,2,3]` at `81-85`.
  - The runtime config rebuilds the same 4-GPU shape as `world=[0,1,2,3]`, `sync_group=[0,2,3]` at `256-264`.
  - This is the correct Gate 2.5 pattern: sender plus the off-GPU TP receiver ranks, and it is a proper subset of the world group.
- Compliance notes:
  - Compliant with the cited spec. It directly exercises the NCCL broadcast transport path required by `/Users/zhenyulin/Downloads/nemorl-port-plan.md:391` and repeats the cycle `N_SYNC_CYCLES = 3` at `75`, which matches the 3+ cycle stability requirement in `1198-1200`.
- Recommended fix if non-compliant:
  - None. This is the canonical transport pattern.

### `rlix/tests/integration/test_gate2_5_feature6.py`

- Transport used:
  - The file creates `nccl_group = dist.new_group(ranks=sync_ranks, backend="nccl")` at `165-166`.
  - Buckets are broadcast over NCCL with `dist.broadcast(staging, src=SENDER_RANK, group=nccl_group)` at `177-178` and `223`.
- Group structure:
  - The actual sync group is `sync_ranks = [SENDER_RANK, RECEIVER_RANK]` at `165`.
  - With the default 2-rank run described in the docstring (`torchrun --nproc-per-node=2` at `18-19`), that means world `[0,1]` and sync group `[0,1]`, which is not a proper subset.
  - When `world_size > 2`, the file moves the receiver to `world_size - 1` at `327-332`, so the sync group becomes `[0,last]`, which is a proper subset but still only covers one receiver rank.
  - The correct Gate 2.5 group for the cited `tp=2` cross-GPU transport case would be sender plus all off-GPU TP receiver ranks, e.g. `[0,2,3]` out of world `[0,1,2,3]`.
- Compliance notes:
  - This file does use NCCL correctly as a transport primitive, but it does not model the Gate 2.5 topology in the cited spec. `/Users/zhenyulin/Downloads/nemorl-port-plan.md:391` makes NCCL mandatory only for `tp_size > 1` with off-GPU TP peers; this file uses a single receiver rank, so it does not prove the required cross-GPU TP-rank transport shape.
  - On the default 2-GPU invocation it also misses the proper-subset requirement.
- Recommended fix if non-compliant:
  - If this file is intended to count toward Gate 2.5 transport coverage, move it to a 4-rank topology and build the NCCL sync group as sender plus all off-GPU TP receiver ranks, not just one receiver.
  - Reuse the `test_gate2_5_selective_sync.py` pattern: separate world gloo barriers from the NCCL transport subgroup, and keep the sync group a proper subset of world.

### `rlix/tests/integration/test_gate2_5_megatron_tp.py`

- Transport used:
  - The file explicitly documents `# Gloo broadcast (all via CPU, no NCCL dtype restrictions)` at `203-205`.
  - `broadcast_shard()` says all tensors stay on CPU with gloo transport at `215-218`.
  - The process group for weight sync is `gloo_world = dist.new_group(ranks=list(range(world_size)), backend="gloo")` at `383-386`.
  - Both shard sync phases call `broadcast_shard(..., gloo_group=gloo_world)` at `465-472`.
- Group structure:
  - Actual transport group: world `[0,1,2,3]` over gloo.
  - Topology modeled by the file: training TP group `[0,1]`, inference TP group `[2,3]` at `7-10` and `388-390`.
  - Correct NCCL transport for this sharded layout should be proper-subset shard groups:
    - shard 0: `[0,2]` out of world `[0,1,2,3]`
    - shard 1: `[1,3]` out of world `[0,1,2,3]`
  - Those groups match the file's own verification logic, where rank 2 validates rank 0's shard and rank 3 validates rank 1's shard at `476-483`.
- Compliance notes:
  - Non-compliant for Gate 2.5 transport. This file does have `tp=2` with off-GPU TP peers, so `/Users/zhenyulin/Downloads/nemorl-port-plan.md:391` and `1196-1201` require the NCCL broadcast path for the sync step.
  - It currently exercises NCCL only for Megatron TP all-reduce (`dist.init_process_group(backend="nccl")` at `374` and TP collectives inside the model), not for weight broadcast.
- Recommended fix if non-compliant:
  - Replace the gloo shard broadcasts with dynamic NCCL subgroup broadcasts.
  - For the file's current per-shard sender model, create one proper-subset NCCL group per shard phase: `[0,2]` for rank 0's shard and `[1,3]` for rank 1's shard.
  - Keep gloo only for world barriers and metadata if needed, and apply the same synchronize-plus-barrier teardown pattern used in `test_gate2_5_selective_sync.py`.

### `rlix/tests/integration/test_gate2_5_qwen_train_sync.py`

- Transport used:
  - `selective_sync()` states: `Broadcast all buckets from rank 0 to all ranks via gloo (CPU)` and `All 3 broadcasts use gloo` at `213-219`.
  - The three broadcast legs are all `dist.broadcast(..., group=gloo_group)` at `246-247`, `257-258`, and `261`, with matching receive-side gloo broadcasts at `266`, `273`, and `286`.
  - `main()` initializes `dist.init_process_group(backend="gloo")` and aliases `gloo_group = None` at `338-343`.
- Group structure:
  - Actual transport group: world/default group `[0,1,2,3]` over gloo.
  - File topology: training ranks `[0,1]`, inference ranks `[2,3]`, sender rank `0` at `51-53`.
  - Correct NCCL group for this file's sync step is `[0,2,3]` out of world `[0,1,2,3]`. Rank 1 should still call `dist.new_group`, but it should remain outside the NCCL collectives.
- Compliance notes:
  - Non-compliant for Gate 2.5 transport. The file claims `TP=2` layout across training and inference workers in the docstring at `4-6`, and the target inference side is split across ranks 2 and 3, so the spec requires the dynamic NCCL broadcast path.
  - Because the broadcast stays entirely on CPU/gloo, it does not verify the transport path named in `/Users/zhenyulin/Downloads/nemorl-port-plan.md:391` and `1196-1201`.
- Recommended fix if non-compliant:
  - Migrate `selective_sync()` to the canonical selective NCCL subgroup used in `test_gate2_5_selective_sync.py`.
  - Create `nccl_group = dist.new_group(ranks=[0,2,3], backend="nccl")`, stage the packed bucket from CPU to GPU on rank 0, receive into CUDA buffers on ranks 2 and 3, and use gloo only for outer barriers.

### `rlix/tests/integration/test_gate2_5_full.py`

- Transport used:
  - `broadcast_cache()` says `Uses 3 CPU (gloo) broadcasts` at `189-198`.
  - `main()` initializes `dist.init_process_group(backend="gloo")` at `329-332`.
  - The file then sets `gloo_world = None` and uses the default group for sync at `342-345`.
  - Phase A and phase B both call `broadcast_cache(..., gloo_group=gloo_world)` at `383` and `448`.
- Group structure:
  - Actual transport group: world/default group `[0,1,2,3]` over gloo.
  - Intended topology in the docstring is selective:
    - `gloo_a: [0,2,3]`
    - `gloo_b: [1,2,3]`
    - documented at `11-14`
  - The correct Gate 2.5 NCCL transport should follow that selective shape, but with NCCL instead of gloo:
    - phase A: `[0,2,3]` out of world `[0,1,2,3]`
    - phase B: `[1,2,3]` out of world `[0,1,2,3]`
- Compliance notes:
  - Non-compliant for Gate 2.5 transport. The modeled target inference side is ranks 2 and 3, so the cited spec requires NCCL for the cross-GPU TP broadcast step.
  - There is also a docstring/code mismatch: the docstring describes selective per-pipeline groups, but the implementation uses the gloo world/default group for both phases.
- Recommended fix if non-compliant:
  - Create explicit dynamic NCCL groups per pipeline phase instead of broadcasting on the gloo world group.
  - Phase A should sync only `[0,2,3]`; phase B should sync only `[1,2,3]`.
  - Reuse the `test_gate2_5_selective_sync.py` teardown pattern so each phase synchronizes CUDA work, barriers on the NCCL subgroup, then destroys the subgroup cleanly.

### `rlix/tests/integration/test_gate2_5_nccl_destroy.py`

- Transport used:
  - No weight broadcast transport is present in this file.
  - The file uses NCCL for top-level init at `262-265` and for TP all-reduce checks at `99`, `135`, `167`, `175`, and `232`.
- Group structure:
  - The only modeled process group is the Megatron TP group inside a 2-rank world.
  - There is no sender-plus-selected-receivers broadcast subgroup to review here.
  - If this file were extended to cover Gate 2.5 transport, it would need a larger world and a proper-subset NCCL sync group such as `[0,2,3]` out of `[0,1,2,3]`.
- Compliance notes:
  - This file is relevant to the `1198-1200` destroy/re-init stability requirement, but not to the step-5 NCCL weight-broadcast transport requirement.
  - It should stay classified as lifecycle-only, not as transport coverage.
- Recommended fix if non-compliant:
  - No transport migration needed. Keep it as the lifecycle test.

## Reference Fix

`rlix/tests/integration/test_gate2_5_selective_sync.py` is the canonical fix pattern for any Gate 2.5 test that still uses gloo for cross-GPU TP weight sync.

- Proper subset NCCL group:
  - `SYNC_RANKS = [0,2,3]` at `81-85`
  - `dynamic_group = dist.new_group(ranks=SYNC_RANKS, backend="nccl")` at `136`
  - For 4 GPUs this gives sync group `[0,2,3]` as a proper subset of world `[0,1,2,3]`.
- Correct transport path:
  - Sender stages the packed bucket CPU to GPU with `record.cpu_uint8_bucket.pin_memory().cuda()` at `145`.
  - Sender broadcasts CUDA tensors on the NCCL group at `147-148`.
  - Receivers allocate CUDA buffers and receive on the same NCCL group at `155-160`.
- Required teardown hardening:
  - `torch.cuda.synchronize()` before subgroup teardown at `198`.
  - `dist.barrier(group=dynamic_group)` before destroy at `199-200`.
  - `dist.destroy_process_group(dynamic_group)` after the subgroup barrier at `201`.
  - This is the already-applied watchdog fix: it prevents rank 0 from destroying the NCCL communicator while ranks 2 and 3 are still finishing the transport.

## Conclusion

Priority order for transport migration from gloo to NCCL:

1. `test_gate2_5_megatron_tp.py`
   - Highest priority because it already models the full `tp=2` cross-GPU training/inference layout, but the actual sync step is still gloo.
2. `test_gate2_5_qwen_train_sync.py`
   - Same core issue: the file claims Gate 2.5 selective sync semantics, but the transport stays on gloo world/default group instead of `[0,2,3]`.
3. `test_gate2_5_full.py`
   - Also still gloo. It needs per-phase NCCL subset groups (`[0,2,3]` then `[1,2,3]`) and currently has a docstring/code mismatch on group shape.

Files that do not need gloo-to-NCCL migration:

- `test_gate2_5_selective_sync.py`
  - Already implements the correct NCCL transport pattern and teardown hardening.
- `test_gate2_5_feature6.py`
  - Already uses NCCL, but should not be treated as complete Gate 2.5 transport coverage until it models sender plus all off-GPU TP receiver ranks in a proper-subset group.
- `test_gate2_5_nccl_destroy.py`
  - Lifecycle-only test; no weight-broadcast transport in scope.
