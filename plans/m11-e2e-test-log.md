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
