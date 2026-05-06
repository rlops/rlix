# M11 RLix-mode E2E Test Log

Append-only iteration log for the Qwen2.5-0.5B GRPO `--num-rollout 2` smoke
test on vast.ai 4xGPU under the rlix-controlled path.

Plan reference: `/Users/zhenyulin/.claude/plans/humming-sleeping-pumpkin.md`.

---

## Attempt 0 — pre-test wiring landed (no run yet)

- Date: 2026-05-04
- Branch heads (pre-push):
  - rlix: `zhenyu/miles-mvp-e2e` — adds `MilesPipeline` accessors + public `before_training`/`after_training` aliases.
  - miles: `zhenyu/m11-mvp-test` — replaces iter-16 stub with full driver
    (`examples/rlix/run_miles_rlix.py`) and adds reusable async loop helper
    (`miles/utils/rlix_train_loop.py`).
- Vast instance: not yet provisioned for this attempt.
- Outcome: wiring committed. Driver flow:
  1. `rlix.init(create_if_missing=True)` → orchestrator + auto scheduler
  2. `allocate_pipeline_id` → `register_pipeline` → `admit_pipeline`
  3. Build `MilesPipelineConfig` dataclass (writeable `system_envs`)
  4. Create `MilesCoordinator` named actor in `RLIX_NAMESPACE`
  5. `coord.create_pipeline_actor` → `MilesPipeline` actor
  6. `pipeline.initialize_pipeline(coord)` → Phase A + Phase B
  7. Pull `train_group` / `rollout_manager` via accessors
  8. `train_group.set_rollout_manager(rollout_manager)` (single asyncio.run)
  9. `coord.sync_base_weights_to_active(-1)` for v=-1 base push
  10. `run_async_train_loop` with rlix-corrected ordering
      (rollout dispatch AFTER `after_step`)
  11. `pipeline.shutdown_hard`
- Codex KT + plan review: applied feedback on
  - loop ordering (dispatch next rollout AFTER `after_step`, not before)
  - base sync via `coord.sync_base_weights_to_active(-1)` (not direct `train_group.update_weights`)
  - `--save ""` to disable save (final-rollout trigger fires regardless of `--save-interval`)
  - public hook aliases on `MilesPipeline`
  - missing CUDA / NCCL env vars in vast runbook
- Known limitations / deferred:
  - `args.start_rollout_id` defaults to 0 in rlix loop (Phase A discards init() return); fine for clean checkpoint smoke.
  - Save / eval at final rollout disabled by `--save ""` and skipping eval inside the loop helper.
  - Single-pipeline only (F22 dual-pipeline deferred).
- Next: commit, push, rsync to vast, run smoke.

---

## Attempt 1 — vast.ai 4xRTX5090 / CUDA 12.9 — partial pass, instance went offline
- Date: 2026-05-05
- Branch heads after debug:
  - rlix `zhenyu/miles-mvp-e2e` @ 9efe21c (lazy pipeline init + Phase A/B bisect logs)
  - miles `zhenyu/m11-mvp-test` @ 7899122 (vast.ai compat fixes)
- Vast instance: ssh9.vast.ai:36579, 4× RTX 5090 (32 GB each), torch 2.9.1+cu129, ray 2.55.1, sglang 0.5.10, py3.12, ROLL@main, transformers 5.3.0
- Outcome: **Phase A passing, Phase B step1 passing, instance went offline during step2**
- Achieved (in order):
  1. F10 topology validation passed (engines=4, per_engine=1, train=2, infer=4, overlap=2)
  2. orchestrator allocate / register / admit pipeline_id ✓
  3. MilesCoordinator named actor created ✓
  4. MilesPipeline.initialize_pipeline started ✓
  5. Phase A step5 offload ✓ (sleep no-op via MILES_SKIP_TMS_PAUSE=1)
  6. Phase A step6.5 collect_cache_owner_roles done — rank 0 = cache_owner ✓
  7. Phase A step6.6 publish_cache_ready_step(-1) ✓
  8. Phase A step7 release actor_train ✓
  9. Phase B step1 request actor_infer ✓ — allocated [0,1,2,3]
  10. Phase B step2 `_create_placement_group` running when SSH connection refused at vast
- Failure: SSH connection refused at ~11:28 UTC; instance unreachable. Cause unknown — possibly the user paused it, vast auto-paused, or VM crashed.

### Issues fixed during this attempt (chronological)
1. **codetiming missing** — `pip install codetiming`
2. **tensordict missing** — `pip install tensordict`
3. **ROLL py3.12 dataclass field** — `worker_config.SequencePackingConfig` had `default= SequencePackingConfig()` (mutable default). Cloned ROLL to /root/external_ROLL, patched to `default_factory=SequencePackingConfig`, `pip install -e .`
4. **peft missing** — `pip install peft`
5. **transformers `AutoModelForVision2Seq` missing** — `roll.models.model_providers` imports it; transformers 5.3.0 dropped it. Workaround: lazy-load `RollFullFinetunePipeline`/`RollMultiLoraPipeline` in `rlix/pipeline/__init__.py` so the MILES path doesn't pull roll.agentic.
6. **wandb protobuf 3.20 incompatibility** — `wandb.proto.wandb_telemetry_pb2.Imports` symbol missing under the rlix-pinned protobuf<3.21. Lazy-imported wandb in `tracking_utils.py` and `wandb_utils.py`.
7. **eval-datasets validator** — `--eval-interval 100` requires `--eval-prompt-data`. Added `--eval-prompt-data aime ...` to satisfy the validator; loop helper skips eval entirely.
8. **C2 partial overlap requires ≥2 engines** — bumped `--rollout-num-gpus-per-engine` from 2 to 1 (now 4 engines × 1 GPU).
9. **C3 fully-async rollout required** — added `--rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async`.
10. **C4 worst-case shrink survival** — bumped `--rollout-num-gpus` to 4 so non-overlap = 2 GPUs.
11. **C5 offload_train required** — added `--offload-train`.
12. **C8 MoE forbidden** — `--moe-router-topk 0` (was defaulting to 2 from megatron).
13. **C11 cpu_serialize transport required** — `--model-update-transport cpu_serialize`.
14. **`--num-gpus-per-node` default 8 vs actual 4** — set to 4 so RollResourceManagerProxy filters our node correctly. Without this, ResourceManager registers `gpu_ranks=[]`, leading to the cryptic `device_mapping used gpus are more than num_nodes×num_gpus_per_node=0×8`.
15. **ROLL_RAY_NAMESPACE / PIPELINE_ID not propagated** — ROLL constants.py asserts both at import time when `RLIX_CONTROL_PLANE=rlix`. Driver now sets them on its env and forwards via Ray runtime_env to MilesCoordinator.
16. **MILES placement_provider unpacking ROLL's `List[List[Dict]]` as a single PG** — confirmed by codex review. Patched `get_train_workers` and `get_inference_engines` to extract `allocated[idx][0]["placement_group"]` and pin `bundle_index=0`.
17. **torch_memory_saver "preload" hook segfault** — LD_PRELOAD-based malloc hook segfaults during `_get_megatron_full_params` -> `tensor.to(GPU)` on Blackwell + tms 0.0.9 + CUDA 12.9. Workaround: env-var-gated switch to "torch" hook mode (CUDAPluggableAllocator-based) via `MILES_TMS_HOOK_MODE=torch`.
18. **torch_memory_saver "torch" hook also crashes during pause/offload** — pause/resume go through the binary wrapper which dies. Workaround: `MILES_SKIP_TMS_PAUSE=1` makes sleep/wake_up near-no-ops (clear_memory + empty_cache only). 0.5B fits 32 GB without aggressive offload.
19. **Phase A wedge after offload** — destroy_process_groups was being called inside `actor.sleep()`, which killed the Gloo group used by subsequent `report_cache_owner_role` (calls `dist.get_rank()`). Patched to skip destroy_process_groups when `MILES_SKIP_TMS_PAUSE=1`.

### Codex review log
- Codex KT + plan review (pre-implementation): identified loop-ordering issue (next-rollout dispatch must come AFTER `after_step` on partial-overlap GPUs), save-at-final-rollout trigger fires regardless of `--save-interval`, base v=-1 sync via `coord.sync_base_weights_to_active(-1)` not direct `train_group.update_weights()`, missing CUDA/NCCL env vars (CUDA_DEVICE_MAX_CONNECTIONS=1, NCCL_NVLS_ENABLE auto, CUBLAS_WORKSPACE_CONFIG, NVTE_ALLOW_NONDETERMINISTIC_ALGO).
- Codex rescue (mid-debug, miles ↔ ROLL PG-shape mismatch): confirmed `ResourceManager.allocate_placement_group` returns `List[List[Dict]]` (resource_manager.py:122-178), proxy doesn't override (line 247-248); miles `placement_provider.py:252-267` and `:170` were treating return as a single PG. Recommended unpack pattern + `bundle_index=0` (PG has one bundle per node, line 56-65). All applied.

### Known issues for next attempt
- Vast instance offline; needs to be brought back. The driver wiring + env are now stable; once the instance is back, just re-run `bash /root/rlix/scripts/run_smoke_e2e.sh` — Phase A and Phase B step1 should pass on first try.
- After Phase B clears: the driver does `set_rollout_manager`, `coord.sync_base_weights_to_active(-1)`, then the loop. None of these have been exercised yet on Blackwell.
- `args.start_rollout_id` still defaults to 0 (Phase A discards `init()` return). Acceptable for clean-checkpoint smoke; needs a `MilesPipeline` accessor for resume.

### Files changed in this iteration
- rlix repo:
  - `rlix/pipeline/__init__.py` — lazy `RollFullFinetunePipeline`/`RollMultiLoraPipeline`
  - `rlix/pipeline/miles_pipeline.py` — Phase A + Phase B bisect logs
- miles repo:
  - `examples/rlix/run_miles_rlix.py` — ROLL_RAY_NAMESPACE / PIPELINE_ID forwarding, lazy init_tracking
  - `miles/utils/rlix_train_loop.py` — drop in-loop eval call
  - `miles/ray/placement_provider.py` — ROLL List[List[Dict]] unpacking, bundle_index=0
  - `miles/ray/actor_group.py` — MILES_TMS_HOOK_MODE env switch
  - `miles/backends/megatron_utils/actor.py` — MILES_TMS_HOOK_MODE + MILES_SKIP_TMS_PAUSE handling
  - `miles/utils/tracking_utils.py` + `miles/utils/wandb_utils.py` — lazy wandb import
- vast-only patches (not version-controlled):
  - ROLL `worker_config.py` SequencePackingConfig field default → default_factory

## Attempt 2 — vast.ai resumed — full Phase A+B init pass, terminal blocker on partial-overlap GPU contention
- Date: 2026-05-05
- Branch heads:
  - rlix `zhenyu/miles-mvp-e2e` @ b333143
  - miles `zhenyu/m11-mvp-test` @ 7b83be5
- Vast instance: ssh9.vast.ai:36579, 4× RTX 5090 (32 GB each), CUDA 12.9, py3.12

### Achieved (extending Attempt 1)
1. F10 topology validation passed (engines=4, per_engine=1, train=2, infer=4, overlap=2)
2. orchestrator allocate / register / admit ✓
3. MilesCoordinator named actor created ✓
4. MilesPipeline.initialize_pipeline scheduled (with `num_gpus=0` after MILES_SKIP_NODE_PG_PIN=1) ✓
5. **Phase A complete**: train init → bucket cache (988 MB, step=-1) → offload → cache_owner=rank0 → publish_cache_ready_step → release actor_train ✓
6. **Phase B step1**: request actor_infer → allocated [0,1,2,3] ✓
7. **Phase B step2a (new)**: `coord.remove_resource_manager_node_pg()` → removed=True ✓
8. **Phase B step2**: `_create_placement_group(4)` succeeded — 4 bundles on node 172.17.0.10, GPUs 0–3 ✓
9. **Phase B step3**: `RolloutManager.remote(...)` ✓
10. **Phase B step4**: `get_engine_count` → 4 engines ✓
11. **Phase B step5**: `register_model_update_resources` ✓
12. **Phase B step6**: `bootstrap_active_engines` ✓
13. `MilesPipeline.initialize_pipeline complete pipeline_id=miles_58beb7a7bc5f engines=4` ✓
14. Driver pulled handles: `train_group=ok rollout_manager=ok engines=4` ✓
15. SGLang router up at 172.17.0.10:4077; 4 SGLang engines registered (POST /add_worker for ports 15000/15003/15006/15009 returned 200) ✓

### Terminal failure
`train_group.set_rollout_manager(rollout_manager)` raised `ActorDiedError`:
```
Worker exit type: INTENDED_SYSTEM_EXIT
Worker exit detail: Destroying worker since its placement group was removed.
Placement group id: 6ae74ed32dd95b3b6fd78e088ba201000000, bundle index: 0
```
The train actors were pinned to ROLL's node-PG (via miles `placement_provider.get_train_workers` → `allocated[idx][0]["placement_group"]`). When Phase B step2a removed the node-PG to free GPUs for the inference PG, Ray killed all actors pinned to it — including the train actors that the loop still needs.

### Root cause: partial-overlap GPU contention without working tms.pause
With train [0,1] / infer [0,1,2,3] on a 4-GPU machine:
- ROLL ResourceManagerProxy's node-PG reserves all 4 GPUs.
- Train actors take 2 of those (2 GPUs reserved).
- For inference, miles Phase B wants 4 more GPUs from the same physical 4 — needs train to release its physical GPUs first.
- `torch_memory_saver.pause()` is the standard release mechanism; it crashes on this hardware (CUDA 12.9 / Blackwell / tms 0.0.9) regardless of hook mode (preload segfaults on bucket build, torch crashes on pause).
- Without `tms.pause`, train actors hold their 2 GPUs throughout. There are 4 GPUs total; engines need 4 → 2-GPU shortfall.

### Why neither escape hatch closes the loop on this hardware
- **Remove ROLL node-PG (current)**: kills train actors → set_rollout_manager fails on dead actors.
- **Don't remove ROLL node-PG**: Phase B step2 deadlocks waiting for free GPUs.
- **Disjoint pools (train=[0,1] / infer=[2,3])**: blocked by `assert_rlix_topology` C1 ("train ⊂ infer required") — rlix-mode is hardcoded for partial overlap.

### Remaining paths to a green smoke (NOT in this attempt)
- (a) Wait for `torch_memory_saver` >0.0.9 with Blackwell+CUDA12.9 fix.
- (b) Run on a machine with ≥6 GPUs so train + infer fit without offload.
- (c) Run on a smaller model (Qwen2.5-0.5B already smallest in the family; can't shrink further).
- (d) Implement standalone-style placement in placement_provider so train actors aren't pinned to ROLL's node-PG, then remove the node-PG safely.

### Smoke summary
- Lines that hit "**phaseB step6: bootstrap_active_engines done**" and "**initialize_pipeline complete ... engines=4**" — full Phase A + Phase B init + 4 SGLang engines under rlix-mode driver: **PASS** ✓
- The actual `--num-rollout 2` training loop (set_rollout_manager → coord.sync_base_weights_to_active(-1) → run_async_train_loop with before_step/after_step around train.train()): **BLOCKED** by tms.pause hardware incompatibility.

### Watchdog parameters in final wrapper
`/root/rlix/scripts/run_smoke_with_watchdog.sh`:
- SILENCE_LIMIT default 300s (raised to 600–900s for production runs since import phase can take 80–120s)
- RUN_LIMIT default 1800s (raised to 2400s for the final attempts)

### Files / commits in Attempt 2
- rlix `b333143`: MILES_SKIP_NODE_PG_PIN + remove_resource_manager_node_pg + Phase B bisect logs
- miles `7b83be5`: forward MILES_* env vars to coordinator runtime_env

## Attempt 3 — vast.ai + tms commit dc68769 + nested-patch — same terminal blocker
- Date: 2026-05-06
- Branch heads (unchanged from attempt 2):
  - rlix `zhenyu/miles-mvp-e2e` @ d48c759
  - miles `zhenyu/m11-mvp-test` @ 7b83be5
- Vast instance: ssh9.vast.ai:36579, 4× RTX 5090, **CUDA 12.9** (driver 575.51.03; image is `cuda:12.9.x`, NOT 13.0 as expected — `nvidia-smi`/`nvcc`/`/usr/local/cuda-12.9` all confirm)

### tms experiments
- User-specified commit `dc68769` (built from src): nested-region assertion fires inside Megatron DDP init.
- `0.0.9.post1` (PyPI): same nested-region assertion.
- `0.0.9` (PyPI re-pull): same assertion (PyPI 0.0.9 was apparently re-uploaded with the strict assertion; the originally-installed 0.0.9 on the vast template was a custom build).
- Patched `entrypoint.py:_with_region_config` in-place: when already in an interesting region, no-op the inner region (binary doesn't export `tms_get_current_tag`, so save/restore impossible).
- After the patch: Phase A `init()` succeeded (DDP nested regions OK). With LD_PRELOAD preload mode, segfault returned at `Update weights` (same as attempt 1). With `MILES_TMS_HOOK_MODE=torch` + `MILES_SKIP_TMS_PAUSE=1` + `MILES_SKIP_NODE_PG_PIN=1`, full Phase A + Phase B + 4 SGLang engines passed (same achievement as attempt 2), then identical `set_rollout_manager` `ActorDiedError` because the train actors are pinned to ROLL's removed node-PG.

### Conclusion
The terminal blocker is **not** torch_memory_saver. It's an architectural conflict:

- `MilesCoordinator.__init__` calls `RollResourceManagerProxy(num_gpus_per_node=4)` which creates a single-bundle PG holding all 4 node GPUs.
- `MilesPlacementProvider.get_train_workers` uses that PG for the train actors (via `wp.placement_group` from `allocate_placement_group`).
- `MilesPipeline._init_phase_b_infer` calls miles standalone `_create_placement_group(rollout_num_gpus)` which asks Ray's free GPU pool for a separate PG.
- The two layers can't coexist on a 4-GPU machine: ROLL's PG holds all 4 GPUs, leaving Ray's free pool empty for the inference PG.
- Removing ROLL's PG kills the train actors that the loop still needs for `set_rollout_manager`/`coord.sync_base_weights_to_active`.

### What would unblock this on the same hardware
None of the following are smoke-scoped one-liners; each is a real code change:
- (a) `MilesPlacementProvider.get_train_workers` should use miles' own `_create_placement_group(2)` for train actors (Ray free pool, separate from ROLL's PG). Then ROLL's RM doesn't need to own the train GPUs at all and the inference PG can coexist; BUT total reservation 2+4=6 GPU on 4-GPU hardware → still won't fit without offload.
- (b) Phase B reuses ROLL's existing node-PG for inference engines (single bundle, all 4 GPUs, with 4 engines fractionating). Train actors stay alive; train+infer fit if `num_gpus_per_actor` for train drops below 0 (i.e. zero Ray GPU reservation, just CUDA_VISIBLE_DEVICES). Single-PG multi-tenant approach, untested in rlix codebase.
- (c) Run on hardware with ≥6 GPUs so 2 train + 4 infer fits without overlap.

### Final smoke status
- **Driver wiring**: validated end-to-end through MilesPipeline `initialize_pipeline complete engines=4`.
- **rlix-mode E2E training loop**: BLOCKED by the architectural placement-group conflict above.
- **Confidence**: high that the wiring itself is correct; low that the smoke can complete on this 4-GPU box without one of the (a)/(b)/(c) refactors.

## Attempt 4 — Codex-prescribed PG-sharing fix + F4d receiver impl — END-TO-END
- Date: 2026-05-06
- Branch heads:
  - rlix `zhenyu/miles-mvp-e2e` @ ce5c5f8
  - miles `zhenyu/m11-mvp-test` @ 9040763
- Vast: ssh9.vast.ai 4× RTX 5090, CUDA 12.9 (image is `cuda:12.9.x` — driver 575.51.03 caps reported CUDA at 12.9; not 13.0), torch 2.9.1+cu129
- torch_memory_saver: 0.0.9 from PyPI, in-place patched `entrypoint.py:_with_region_config` to no-op when already in an interesting region (Megatron DDP init nests; the 0.0.9 binary doesn't export `tms_get_current_tag` so save/restore not possible).

### What now passes end-to-end
1. F10 topology validation
2. orchestrator allocate / register / admit
3. MilesCoordinator + MilesPipeline construction (with `num_gpus_per_actor=0` for train so ROLL's node-PG bundle has GPU capacity left for engines)
4. Phase A: train init → onload (post-init wake_up) → build_cpu_bucket_cache(-1) → offload (real `tms.pause`, memory drops 4.57 → 0.81 GB) → cache_owner role → publish_cache_ready_step → release actor_train
5. Phase B step1: request actor_infer [0,1,2,3]
6. Phase B step2: `placement_provider.get_all_rollout_engine_placements()` reuses ROLL's node-PG (bundle_indices=[0,0,0,0], gpu_ids=[0,1,2,3]) — NO `_create_placement_group` and NO PG removal
7. Phase B step3: RolloutManager + 4 SGLang engines spawned and registered with the router (POST /add_worker → 200 OK ×4)
8. Phase B step4: get_engine_count=4
9. Phase B step4b: train_group.set_rollout_manager — train actors stay alive
10. Phase B step5: register_model_update_resources
11. Phase B step6: bootstrap_active_engines indices=[0,1,2,3]
12. **Phase B step7: sync_base_weights_to_active(-1)** — F4d cpu_serialize sender writes torch-pickled bucket to /dev/shm; receiver reads, deserializes, re-serializes via SGLang's `MultiprocessingSerializer`, calls `tokenizer_manager.update_weights_from_tensor`. **All 4 engines hit `/update_weights_from_cpu_bucket → 200 OK`.**
13. `initialize_pipeline complete pipeline_id=... engines=4` ✓
14. Driver pulled handles ✓
15. **Loop pre-loop: `rollout_manager.generate.remote(rollout_id=0)` dispatched** ✓
16. **Loop step1: rollout_data ready** — async rollout produced 100s of valid Qwen2.5-0.5B GRPO responses with chain-of-thought reasoning, ~2.4k tok/s gen throughput, prefix-cache hit ~9.3%, response_len mean=537 / max=1386 ✓
17. **Loop step2: `before_step(0)` start** ✓ (entered _before_training in MilesPipeline)

### Remaining wedge
`_before_training(rollout_id=0)` blocks indefinitely at:
```python
allocated = self._request_cluster_gpus(
    cluster_id=self._actor_train_cluster_id,
    priority=Priority.ACTOR_TRAINING,
    global_step=int(step),
)
```

The rlix scheduler should detect priority inversion (train priority=1 vs infer priority=6) and emit `coordinator.resize_infer.remote(dp_ranks_to_remove=[overlap])` to free actor_train's GPUs. The wiring exists (rlix/scheduler/scheduler.py line 1374 + miles_coordinator.resize_infer + _shrink_workers + rollout_manager.shrink_engines), but the scheduler doesn't fire it for our setup. SGLang engines keep generating in the background; eventually the watchdog kills the run after the 25-min silence cap.

### What would unblock the loop
Diagnose why scheduler doesn't issue `resize_infer` in this configuration (single-pipeline, partial overlap, ACTOR_TRAINING request after sustained ROLLOUT activity). Likely a missing trigger on `request_gpus` priority-inversion path or a state machine that only preempts on the FIRST rollout request, not subsequent train requests. Out of scope for the smoke iteration; tracked for the rlix scheduler team.

### Bug counts (commits)
- rlix:
  - 376bec7 — accessors + public hook aliases
  - 9efe21c — lazy pipeline `__init__` + Phase A/B bisect logs
  - 5602563 — attempt 1 log
  - b333143 — MILES_SKIP_NODE_PG_PIN + remove_resource_manager_node_pg (later reverted in favor of PG-sharing)
  - d48c759, 3b0bdd7 — attempt 2/3 logs
  - **ce5c5f8** — codex-prescribed PG-sharing + STORAGE_NAME compat + num_gpus=0 + step3.5 wake_up + step4b set_rollout_manager + step7 sync(-1)
- miles:
  - e29396a — driver wiring + reusable async loop
  - 7899122, 7b83be5 — vast.ai compat (lazy wandb + placement_provider PG-shape unpack + MILES_TMS_HOOK_MODE/MILES_SKIP_TMS_PAUSE/MILES_SKIP_NODE_PG_PIN env hatches + MILES_* runtime_env forwarding)
  - **9040763** — real F4d /update_weights_from_cpu_bucket receiver (Pydantic body model, torch.load → MultiprocessingSerializer → tokenizer_manager.update_weights_from_tensor) + [loop] bisect logs

### Bottom line
This iteration validated that the rlix-mode driver, partial-overlap PG sharing, F4d cpu_serialize transport, and one full GRPO rollout all work end-to-end on a 4-GPU machine — exactly as rlix is designed to. The only missing piece is the rlix scheduler's preemption signal, which is one targeted fix away from a fully-green num-rollout=2 smoke.

---

## Attempt 5 — `enable_memory_saver` was off → `release_memory_occupation` no-op (in-progress)

- Date: 2026-05-06
- Branch heads: rlix=zhenyu/miles-mvp-e2e (rsync; nvidia-smi probe in `_wait_for_overlap_engines_offloaded`), miles=zhenyu/m11-mvp-test (rsync)
- Vast instance: 36156578 (4xRTX5090, 32 GB each)

### What changed since attempt 4
1. `_wait_for_overlap_engines_offloaded` Phase 2 now polls `nvidia-smi --query-gpu=memory.free --id=...` instead of calling `flush_engines`/`get_engine_free_gpu_mem` (which don't exist on miles' RolloutManager). Threshold ≥20 GB free; timeout 60 s.
2. Added `--offload-rollout` to `scripts/run_smoke_e2e.sh` (was missing). Without it, `args.offload_rollout=False` → SGLang launched with `enable_memory_saver=False` → `release_memory_occupation` is a NO-OP and torch_memory_saver doesn't track engine allocations.

### Smoking gun (run 4 log, before fix)
```
SGLangEngine ServerArgs: ... enable_memory_saver=False ...
[Rank 0] Memory-Usage before wake_up model: free_GB=1.93, used_GB=29.43
[torch_memory_saver.cpp] CUresult error: 2 (out of memory) file=csrc/utils.h func=cu_mem_create line=194
```
With state="offloaded" but 30 GB still occupied by SGLang's process, the 0.5B train wake_up's `cuMemCreate` for ~3.7 GB of weights runs out of OS-level memory.

### What we expect with `--offload-rollout`
- SGLang launches with `enable_memory_saver=True` → torch_memory_saver wraps engine allocations.
- `shrink_engines` calls `release_memory_occupation` which now actually returns memory to the OS pool.
- nvidia-smi probe sees ≥20 GB free within seconds; train wake_up succeeds; rollout 1 dispatches.


### Attempt 5 outcome — memory_saver works, new SGLang race exposed
- **Memory_saver fix verified**: with `enable_memory_saver=True`, `_wait_for_overlap_engines_offloaded` Phase 2 nvidia-smi probe immediately reports `min=28.20 GB` free (was 1.93 GB stuck). Train wake_up succeeds: `[Rank 0] free_GB=26.28 used_GB=5.08`.
- **But also**: SGLang engines on GPUs 0,1 crash during `release_memory_occupation` with:
  ```
  write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
  ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
  Scheduler hit an exception: ...
  SIGQUIT received. signum=None, frame=None. It usually means one child failed.
  ```
  + Megatron train actor 288645 dies with `[torch_memory_saver.cpp] Cannot resume allocation that is not paused. tag=default ptr=12918456320`.
- **Race**: `is_idle` returns true means request queue drained, but SGLang's tokenizer/scheduler may still have an in-flight Triton kernel launched for the last decode pass. `release_memory_occupation` synchronously moves persistent token-pool buffers to CPU; if the Triton kernel runs after the move, it sees a CPU tensor. The Megatron crash is downstream — once one engine's process dies, ROLL's PG-coupled actors lose connectivity.

### Attempt 6 — drain sync sleep before release_memory_occupation
- Patch (`miles/miles/ray/rollout.py:911`): under `RLIX_CONTROL_PLANE=rlix`, sleep 0.5 s after the `is_idle` drain before `release_memory_occupation`. Lets SGLang's scheduler reach a safe checkpoint where no Triton kernels are dispatched against persistent buffers.
- Cost: ~0.5 s per shrink (twice per smoke run = 1 s overhead). Smoke-only kludge; M11.5 follow-up = SGLang-side `pause_generation + synchronize_cuda` for a deterministic fix.


### Attempt 6 outcome — 0.5s sleep made flush_cache hang for 60s
- Before before_step ran, the rollout 0 result returned. Phase A→B init succeeded, `enable_memory_saver=True` confirmed.
- before_step → resize_infer → shrink_engines → drain (is_idle=True) → 0.5s sleep → release_memory_occupation → flush_cache TIMEOUT after 60s ("Timeout while flushing cache").
- The sleep didn't help; SGLang's `/flush_cache` returns non-200 for the entire 60s window — engine still has pending state from rollout 0 even after `is_idle` returned True. The sleep let the scheduler process some of that pending state but not all.
- Cascading downstream: scheduler RayTaskError → before_training fails → driver exits.

### Attempt 7 — replace sleep with `pause_generation(mode="retract")`
- Patch (`miles/miles/ray/rollout.py:911`): replace 0.5s sleep with `pause_generation` HTTP call (rlix-mode only). SGLang's `pause_generation` blocks until the scheduler reaches a safe checkpoint and retracts in-flight decode iterations. This should give `release_memory_occupation` (and its inner flush_cache) a quiescent engine to work with.
- Wrapped in try/except: pause_generation may reject if engine is already paused; we fall through to flush+release rather than blocking.


### Attempt 7 outcome — pause_generation works, but double wake_up exposed
- ✅ `pause_generation(mode="retract")` returns 200 OK from both engines
- ✅ `release_memory_occupation` returns 200 OK (no Triton crash)
- ✅ Free GPU mem reaches `min=28.16 GB` cleanly
- ✅ wake_up #1 succeeds: `[Rank 0] free=24.39, used=6.97, elapsed=0.9s`
- ❌ wake_up #2 fires immediately: `[torch_memory_saver.cpp] Cannot resume allocation that is not paused. tag=default ptr=13287555072`
- Train actor exits SYSTEM_ERROR; before_step returns OK but train_group.train() raises ActorDiedError.

#### Why two wake_ups
1. `MilesPipeline._before_training` calls `train_group.onload()` → wake_up.
2. `MegatronTrainRayActor.train()` calls `self.wake_up()` when `args.offload_train=True`.

The standalone `train_async.train` only calls #2 (no explicit onload before train). Our rlix-mode added #1 redundantly.

### Attempt 8 — remove redundant onload in `_before_training`
- Patch (`rlix/pipeline/miles_pipeline.py:558`): drop `self._run_async(self._train_group.onload())`. Leave actor offloaded; rlix_train_loop dispatches train() which wakes up safely (matching standalone).


### Attempt 8 outcome — full integration path runs cleanly until train_actor needs ref_log_probs
- ✅ before_step + shrink_engines (pause_generation + release) clean — no Triton crash, no flush_cache timeout
- ✅ train wake_up succeeds (single call only, 0.7 s elapsed)
- ✅ Generation continues on engines 2, 3 in parallel
- ❌ `train_actor` crashes: `policy_loss_function: ref_log_probs = torch.cat(ref_log_probs, dim=0); TypeError: cat() received an invalid combination of arguments - got (NoneType, dim=int)`
- Root cause: `miles_pipeline.py:176` hardcodes `with_ref=False` when constructing `RayTrainGroup`. Standalone `placement_group.py:192` derives it as `args.kl_coef != 0 or args.use_kl_loss`. Without the ref model loaded, train_actor's pre-loss `compute_log_prob(store_prefix="ref_")` is gated off (line 419 `if "ref" in self.weights_backuper.backup_tags`), and `batch["ref_log_probs"]` stays `None`.

### Attempt 9 — derive `with_ref` from miles_args.use_kl_loss / kl_coef
- Patch (`rlix/pipeline/miles_pipeline.py:163`): `with_ref=bool(args.use_kl_loss or args.kl_coef != 0.0)`. Matches miles standalone semantics. With `--use-kl-loss` set in the smoke, this becomes True; ref model is loaded; ref_log_probs computation runs each train step.


### Attempt 9 outcome — full train step succeeded; flush_cache wedge moved to weight-sync path
- ✅ ref model loaded; train_actor's compute_log_prob(store_prefix="ref_") populates `batch["ref_log_probs"]`
- ✅ rollout_id=0 step3: `train_group.train done` (full GRPO update with KL loss, ~11 s elapsed)
- ✅ step4: `after_step start` → train_group.build_cpu_bucket_cache → train_group.offload (135.70 GB CPU usage at peak)
- ❌ **`MilesCoordinator.sync_base_weights_to_active(0)`** → `MilesModelUpdateService.sync_selected_workers` → `SGLangEngine.finalize_weight_update` → `Timeout while flushing cache`
- Same SGLang stuck-flush as the shrink path (attempt 6), now in weight-sync. The fully-async rollout function returned data but the engine queue was not synchronously quiesced, so flush_cache loops 60 s for non-200 status.

### Attempt 10 — pause_generation before finalize_weight_update + continue_generation after
- Patch (`rlix/pipeline/miles_model_update_service.py:288`): in `sync_selected_workers` step (4), before the `finalize_weight_update` fan-out, call `pause_generation(mode="retract")` on each handle to retract in-flight batches and reach a quiescent state. After `finalize_weight_update` returns, call `continue_generation()` to resume.
- Rationale: same fix pattern as the shrink path (attempt 7) — SGLang's `flush_cache` only succeeds against a paused/quiescent scheduler.


### Attempt 10 outcome — ✅ FULL E2E PASS (both rollouts complete + clean shutdown_hard)
- 11:43:52 init complete (engines=4)
- 11:43:52 [loop] pre-loop generate dispatch rollout_id=0
- 11:43:57 [loop] rollout_id=0 step1: await rollout_data done (5 s)
- 11:43:57 [loop] rollout_id=0 step2: before_step done (engines [0,1] offloaded; OS-level mem free 28.16 GB)
- 11:44:07 [loop] rollout_id=0 step3: train_group.train done (10 s)
- 11:44:19 [loop] rollout_id=0 step4: after_step done (12 s, includes sync_base_weights)
- 11:44:19 [loop] rollout_id=1 step1: await rollout_data done (~0 s — already in flight)
- 11:44:19 [loop] rollout_id=1 step2: before_step done
- 11:44:22 [loop] rollout_id=1 step3: train_group.train done (3 s — warm cuda graphs)
- 11:44:31 [loop] rollout_id=1 step4: after_step done (9 s)
- 11:44:31 [run_miles_rlix] shutdown_hard complete pipeline_id=miles_cd235d4fa85a — exiting
- **EXIT_CODE=0**

#### Final commit chain (rlix branch zhenyu/miles-mvp-e2e):
1. accessors + public hook aliases
2. lazy MilesPipeline.__init__ + Phase A/B bisect logs
3. PG-sharing + STORAGE_NAME compat + num_gpus=0 + step3.5 wake_up + step4b set_rollout_manager + step7 sync(-1)
4. nvidia-smi probe in `_wait_for_overlap_engines_offloaded`
5. drop redundant `train_group.onload()` in `_before_training` (avoid double wake_up — train() already does it)
6. derive `with_ref` from `args.use_kl_loss / kl_coef` so ref model is loaded for KL-loss runs
7. `pause_generation`/`continue_generation` wrap around `finalize_weight_update` in MilesModelUpdateService.sync_selected_workers

#### Final commit chain (miles branch zhenyu/m11-mvp-test):
1. driver + reusable async loop wired
2. lazy wandb + placement_provider PG-shape unpack + MILES_TMS_HOOK_MODE/MILES_SKIP_TMS_PAUSE/MILES_SKIP_NODE_PG_PIN env hatches + MILES_* runtime_env forwarding
3. real F4d /update_weights_from_cpu_bucket receiver + [loop] bisect logs
4. rlix-mode override in `rollout.py` to force `needs_offload=True` (ensures `enable_memory_saver=True` in SGLang ServerArgs)
5. `pause_generation(mode="retract")` wrap around `release_memory_occupation` in shrink_engines (rlix-mode)
6. Smoke script: add `--offload-rollout` (so SGLang's release_memory_occupation actually frees memory back to OS)

#### Total iterations to green: 10 attempts.

#### Key lessons
1. **SGLang's `release_memory_occupation` is a no-op without `enable_memory_saver=True`**, which requires `--offload-rollout` in miles. Without this, `is_idle` returns true but the CUDA caching allocator still holds the memory; partial-overlap topology becomes impossible because the next train wake_up OOMs.
2. **`flush_cache` blocks on a non-quiescent SGLang scheduler.** With a fully-async rollout function, the rollout-data return does not synchronously quiesce the engine — `pause_generation(mode="retract")` is required before any flush_cache to retract in-flight batches and reach a safe checkpoint.
3. **`with_ref=True` is required when `args.use_kl_loss`.** Otherwise `train_actor`'s ref_log_probs computation is gated off and `policy_loss_function` crashes on `torch.cat(None, dim=0)`.
4. **Avoid double wake_up**: `MegatronTrainRayActor.train()` already wakes up internally when `args.offload_train` is set; do not call `train_group.onload()` from the runtime hook before train(). Otherwise the second resume hits `[torch_memory_saver.cpp] Cannot resume allocation that is not paused`.

#### Total wall-clock time on vast.ai instance 36156578 across all 10 attempts: ~3.5 hours (instance now stopped).

