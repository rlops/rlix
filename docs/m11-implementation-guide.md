# M11 Implementation Guide

> **Audience: code reviewer who has not opened this repo before.** Goes
> top-down: what the system does, the architecture in 30 seconds, every
> feature with the files and tests, then function-level references.

---

## §1 What this is

### 1.1 The problem

RL training for LLMs has two GPU-heavy stages:
1. **Rollout** (inference) — generate trajectories with the current
   policy on an inference engine like SGLang.
2. **Train** (back-prop) — update the policy with PPO/GRPO via Megatron.

A single experiment alternates between them. If train and rollout run
on **disjoint GPUs**, the rollout GPUs idle during training and vice
versa — wasted capacity.

### 1.2 What rlix does

**rlix** (this repo's `rlix/` package) is a Ray-based GPU time-sharing
controller. Multiple RL pipelines register their cluster GPU mappings
with a central scheduler (`rlix.scheduler.SchedulerImpl`); the scheduler
arbitrates GPU allocations by priority:

```
INITIALIZATION (0) > ACTOR_TRAINING (1) > … > GENERATION (6)
```

When an `ACTOR_TRAINING` request arrives for a GPU currently held by a
preemptable `GENERATION` worker, the scheduler tells the coordinator to
**resize the inference engine** (shrink), frees the overlap GPUs, and
grants them to the trainer. After training, the GPU is released and
inference can re-expand.

The scheduler addresses each pipeline by `pipeline_id` (a string from
`orchestrator.allocate_pipeline_id`); each pipeline's actor_train and
actor_infer GPUs are declared via `cluster_device_mappings`
(`{"actor_train": [0,1], "actor_infer": [0,1,2,3]}` for partial
overlap, `{"actor_train": [0], "actor_infer": [0,1]}` for one
pipeline of an M11.2 dual disjoint topology). A
`cluster_id` string of the form `f"{pipeline_id}_{cluster_name}"`
(e.g. `miles_c2dda50d955a_actor_train`,
`miles_c2dda50d955a_actor_infer_generation`) identifies the cluster
the scheduler grants/preempts on each `request_cluster_gpus` call. See
`rlix/pipeline/miles_pipeline.py:67-70` for how the strings are
assembled. Drilling into the per-pipeline namespace
+ cluster name details is in Features 6 and 16 below.

This lets **train and rollout share physical GPUs** (partial overlap)
or run as **disjoint per-pipeline pools** (multi-pipeline) on the same
machine.

### 1.3 What miles is

**miles** (sibling repo) is the actual RL training stack: SGLang for
rollout, Megatron-LM for training, Ray actors for orchestration. It
already has a standalone (non-rlix) entry point in `train_async.py`.

**rlix-mode** is when miles runs *under* rlix's scheduler instead of
standalone. The integration glue lives in:
- `rlix/pipeline/miles_pipeline.py` — per-pipeline init + runtime hooks
- `rlix/pipeline/miles_coordinator.py` — per-pipeline coordinator (resize, sync)
- `rlix/pipeline/miles_model_update_service.py` — weight-sync transport
- `miles/examples/rlix/run_miles_rlix.py` — single-pipeline driver
- `miles/examples/rlix/run_miles_dual.py` — dual-pipeline driver

### 1.4 What M11 is

M11 is the milestone for **first end-to-end working rlix-mode**. It
has three sub-milestones:

| ID | Scope | Status |
|---|---|---|
| **M11.1** | Single MilesPipeline, partial-overlap GPU sharing (train ⊂ infer), full Qwen2.5-0.5B GRPO loop with `--num-rollout 2` | ✅ GREEN |
| **M11.2** | Two concurrent MilesPipelines on disjoint GPU pools (P1=[0,1], P2=[2,3]), each running its own GRPO loop via `asyncio.gather` | ✅ GREEN |
| **M11.3** | Multi-pipeline (3+) with cross-pipeline GPU contention | Punted (needs F22 shell-init contract) |

The verification rig is **vast.ai 4xGPU instances** running Qwen2.5-0.5B
GRPO with 2 rollouts. EXIT_CODE=0 with both rollouts trained + clean
shutdown is the bar.

### 1.5 Reading order if you only have 5 minutes

1. §2 architecture diagram below
2. §3 Feature 1 (memory_saver gating) — most representative tms fix
3. §3 Feature 7 (dual driver) — most representative M11.2 change
4. §3 Feature 11 (engine-index conversion) — the M11.2 follow-up bug
5. §5 deferred / out-of-scope (so you know what NOT to ask about)

---

## §2 Architecture at a glance

```
                    ┌────────────────────────────────────────────────────┐
                    │ Driver process (run_miles_rlix.py / _dual.py)      │
                    │   1. rlix.init() → orchestrator                    │
                    │   2. allocate_pipeline_id, register, admit         │
                    │   3. spawn MilesCoordinator (named Ray actor)      │
                    │   4. coordinator.create_pipeline_actor → MilesPipe │
                    │   5. pipeline.initialize_pipeline (Phase A + B)    │
                    │   6. asyncio.run(run_async_train_loop)             │
                    └────────────────┬──────────────────────┬────────────┘
                                     │                      │
                                     ▼                      ▼
            ┌────────────────────────────────┐  ┌─────────────────────┐
            │ rlix Orchestrator + Scheduler  │  │ MilesCoordinator    │
            │ (one per Ray cluster)          │  │ (one per pipeline)  │
            │                                │  │                     │
            │ Priority queue:                │  │ - resize_infer      │
            │ INIT(0) > TRAIN(1) > … > GEN(6)│◀─┤ - sync_base_weights │
            └────────────────────────────────┘  │ - shutdown_hard     │
                                                └──────┬──────────────┘
                                                       │
                                          ┌────────────┴───────────────┐
                                          ▼                            ▼
                       ┌────────────────────────────┐ ┌──────────────────────────┐
                       │ MilesPipeline (Ray actor)  │ │ MilesModelUpdateService  │
                       │ - init Phase A (train)     │ │ (Ray actor)              │
                       │ - init Phase B (infer)     │ │ - sync_selected_workers  │
                       │ - _before_training         │ │ - finalize_weight_update │
                       │ - _after_training          │ └──────────────────────────┘
                       └────────────┬───────────────┘
                                    │
                  ┌─────────────────┴──────────────────┐
                  ▼                                    ▼
       ┌────────────────────┐            ┌──────────────────────────────┐
       │ RayTrainGroup      │            │ RolloutManager (Ray actor)   │
       │ (Megatron actors)  │            │ - shrink_engines             │
       │ - actor_train_     │            │ - expand_engines             │
       │   workers[N_GPUs]  │            │ - SGLangEngine actors[N]     │
       └────────────────────┘            └──────────────────────────────┘
```

**3-bullet legend:**
- **Driver** is the user-facing Python process; it stays single-threaded
  except for the `asyncio.run` train loop. F13 contract: no top-level
  try/except, no `ray.shutdown()` — failures propagate.
- **MilesCoordinator + MilesPipeline + MilesModelUpdateService** are
  one Ray actor each. The orchestrator + scheduler live in
  `RLIX_NAMESPACE`; the per-pipeline coordinator MUST live in the
  per-pipeline namespace `pipeline_<pipeline_id>_NS` with the name
  `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` so the scheduler's
  `resize_infer` / `shrink_engines` RPCs can resolve it. See Feature 16.
- **RayTrainGroup + RolloutManager** are miles' existing Ray actor
  groups for Megatron train workers and SGLang rollout engines.

---

## §3 Feature catalog

15 features, grouped by sub-milestone. Each entry follows the same
shape: **Problem → Files → Implementation logic → Tests / verification
→ Commit**.

### M11.1 single-pipeline tms fixes (5)

These five fixes were what took M11.1 from "infinite hang or OOM" to
green. All are documented in detail at `docs/tms-fixes.md`; this
section summarizes them for the reviewer.

#### Feature 1 — SGLang `enable_memory_saver=True` gating

- **Problem**: SGLang's `release_memory_occupation` returns 200 OK and
  the rollout-manager state machine flips to `offloaded`, but
  `nvidia-smi` shows the engine process still holds ~30 GB. The next
  train wake_up OOMs.
- **Root cause**: SGLang only releases memory if launched with
  `enable_memory_saver=True`. miles gates that on `args.offload_rollout`
  (sglang_engine.py:1057). Plus `miles/ray/rollout.py:1586`'s static
  `rollout_pg_offset` math sets `needs_offload=False` for the
  partial-overlap case (group_abs_start=2, megatron_num_gpus=2 → 2<2
  false), forcing `enable_memory_saver=False`.
- **Files**:
  - `miles/miles/ray/rollout.py:1616-1628` — rlix-mode override forces
    `needs_offload=True` when `RLIX_CONTROL_PLANE=rlix` (default math
    is at line 1616; the override condition is at line 1627-1628)
  - `rlix_miles/scripts/run_smoke_e2e.sh` — added `--offload-rollout`
- **Implementation logic**: under `RLIX_CONTROL_PLANE=rlix`, ignore
  miles' static disjoint-pool calculation and force
  `needs_offload=True` so the SGLang ServerArgs end up with
  `enable_memory_saver=True`. The smoke script must also pass
  `--offload-rollout` to flip miles' `args.offload_rollout=True`.
- **Tests / verification**: M11.1 attempt 5 log
  (`plans/m11-e2e-test-log.md`): `OS-level GPU mem free min=28.20 GB`
  immediately after `state="offloaded"` (was 1.93 GB before the fix).
- **Commit**: rlix `e0a6b27`, miles `f58b365`.

#### Feature 2 — `pause_generation` before `release_memory_occupation` in shrink

- **Problem A**: `Pointer argument cannot be accessed from Triton (cpu
  tensor?)` — SGLang scheduler launches a Triton kernel against
  persistent token-pool buffers AFTER `is_idle=True` but BEFORE the
  release moves them to CPU.
- **Problem B**: `Timeout while flushing cache` — naive `time.sleep(0.5)`
  workaround doesn't quiesce the scheduler; flush_cache loops 60s.
- **Root cause**: `is_idle=True` only means request queue is empty. The
  scheduler thread can still be mid-decode-iteration. SGLang's
  `pause_generation(mode="retract")` is the proper API to retract
  in-flight batches and reach a quiescent state.
- **Files**: `miles/miles/ray/rollout.py:911-942` (block; the actual
  `pause_generation.remote` call is at line 933)
- **Implementation logic**: in `RolloutManager.shrink_engines`, between
  the `is_idle` drain (step 3) and `release_memory_occupation` (step 4),
  fan-out `pause_generation(mode="retract")` to all target engines
  (rlix-mode only; gated on `RLIX_CONTROL_PLANE=rlix` env var). Wrapped
  in try/except so a "already paused" rejection doesn't block the loop.
- **Tests / verification**: M11.1 attempt 7 log: `POST
  /pause_generation HTTP/1.1 200 OK` precedes
  `POST /release_memory_occupation HTTP/1.1 200 OK`; train wake_up no
  longer hits the Triton crash.
- **Commit**: miles `f58b365`.

#### Feature 3 — `pause_generation` around `finalize_weight_update`

- **Problem**: After the FIRST successful train step, the post-step
  `MilesCoordinator.sync_base_weights_to_active(0)` →
  `MilesModelUpdateService.sync_selected_workers` →
  `SGLangEngine.finalize_weight_update` hits the same flush_cache
  60s-timeout — but now during weight sync, not engine release.
- **Root cause**: With a fully-async rollout function, the rollout-data
  return does not synchronously quiesce the engine. Pending decode
  batches keep the queue non-empty, so flush_cache never returns 200.
- **Files**: `rlix_miles/rlix/pipeline/miles_model_update_service.py:288-318`
- **Implementation logic**: in `sync_selected_workers` step 4, before
  the `finalize_weight_update` fan-out, call `pause_generation(mode="retract")`
  on each handle; after finalize completes, call `continue_generation()`
  to resume. Mirrors the shrink-path fix.
- **Tests / verification**: M11.1 attempt 10 log: full first iteration
  (rollout 0 train + after_step + sync_base_weights) completes; rollout 1
  trained; `EXIT_CODE=0`.
- **Commit**: rlix `e0a6b27`.

#### Feature 4 — Drop redundant `train_group.onload()` in `_before_training`

- **Problem**: Second `tms.resume` crashes with `[torch_memory_saver.cpp]
  Cannot resume allocation that is not paused. tag=default ptr=...`
- **Root cause**: `MilesPipeline._before_training` was calling
  `self._run_async(self._train_group.onload())` (which broadcasts
  `wake_up`). But `MegatronTrainRayActor.train()` ALSO calls
  `self.wake_up()` when `args.offload_train=True`
  (`miles/backends/megatron_utils/actor.py:357`). Two resumes back-to-back
  → second one finds the region already resumed.
- **Files**: `rlix_miles/rlix/pipeline/miles_pipeline.py:566-596` (the
  `_before_training` body — the redundant `onload()` call previously
  sat near the end of this block, removed in commit `e0a6b27`)
- **Implementation logic**: removed the explicit `train_group.onload()`
  call. Train wake_up now happens exactly once, inside `train()`. Matches
  standalone `train_async.train` behavior.
- **Tests / verification**: M11.1 attempt 8 log: exactly one
  `Timer wake_up end` per train step (~0.7s elapsed); no "Cannot resume"
  error.
- **Commit**: rlix `e0a6b27`.

#### Feature 5 — `MILES_TMS_HOOK_MODE=torch` for CUDA 12.9 / Blackwell

- **Problem**: Default tms 0.0.9 hook_mode (`preload` — LD_PRELOAD libc
  malloc) segfaults during `build_cpu_bucket_cache` (Phase A step 4) on
  CUDA 12.9 / Blackwell.
- **Root cause**: `hook_mode="preload"` intercepts every libc malloc in
  the process; on Blackwell + CUDA 12.9 the new mempool internals trip
  unsafe interception paths.
- **Files**: `miles/miles/backends/megatron_utils/actor.py:96-111`
  (already existed; M11 just sets the env var in the smoke script);
  `rlix_miles/scripts/run_smoke_e2e.sh` — exports
  `MILES_TMS_HOOK_MODE=torch`.
- **Implementation logic**: switch tms to PyTorch's
  `CUDAPluggableAllocator` path via `hook_mode="torch"` (set BEFORE
  any tms call that triggers `_ensure_initialized`). Narrower
  catchment; safe on Blackwell.
- **Tests / verification**: M11.1 single-pipeline run completes
  Phase A `build_cpu_bucket_cache` without segfault.
- **Commit**: miles `f58b365` (smoke script env var).

### M11.1 driver and runtime hooks (3)

#### Feature 6 — MilesPipeline init bootstrap (Phase A + Phase B)

- **Problem**: rlix's per-pipeline actor needs to set up Megatron train
  + SGLang infer + register weight-sync transport, in the right order,
  with the right GPU allocation calls into the rlix scheduler.
- **Files**: `rlix_miles/rlix/pipeline/miles_pipeline.py:120-403`
- **Implementation logic**:
  - **Phase A** (train side, steps 1–7):
    1–2. Request actor_train GPUs from the rlix scheduler at priority
         `INITIALIZATION`.
    3. Construct `RayTrainGroup` over those GPUs; init Megatron actors.
    3.5. `train_group.onload()` so HF→Megatron weight gather has GPU
         residence (fixes a downstream offloaded-weight bug).
    4. `build_cpu_bucket_cache(-1)` — Megatron exports torch_dist
       weights to a CPU bucket cache for the cpu_serialize transport.
    5. `train_group.offload()` — release train GPUs.
    6.5. `collect_cache_owner_roles` to identify which Megatron actor
         owns the cache.
    6.6. `publish_cache_ready_step(-1)` — coordinator broadcasts that
         the cache is ready for v=-1 (initial weight sync).
    7. Release the actor_train allocation back to the scheduler.
  - **Phase B** (infer side, steps 1–8):
    1. Request actor_infer GPUs at priority `INITIALIZATION`.
    2. `placement_provider.get_all_rollout_engine_placements()` returns
       per-engine `WorkerPlacement` objects pinned to the inference
       pool's physical GPUs.
    3. Construct `RolloutManager` over the placement group.
    4. Wait for engines to come up; collect engine count.
    4b. `train_group.set_rollout_manager(rollout_manager)` — wires
        the train side to the infer side for weight sync.
    5. `register_model_update_resources` — coordinator registers
       handles for the weight-sync transport.
    6. `bootstrap_active_engines` — coordinator marks the freshly-up
       engines as already active (M11.1 init-comes-up-active branch).
    7. `sync_base_weights_to_active(-1)` — base v=-1 sync via the
       cpu_serialize transport (CPU bucket → /dev/shm → SGLang
       update_weights_from_tensor).
    8. Transition actor_infer cluster from `INITIALIZATION` to
       `GENERATION` priority so the gap-ratio planner can preempt it
       later for actor_train.
- **Tests / verification**: M11.1 attempt 4 log: every step logged in
  order; init time ~5–7 min for the 0.5B model. Each step has
  bisect-quality log lines (`phaseA step5: offload start/done`).
- **Commit**: rlix `18f296d` (initial), refined through
  `2b73aef`, `b241af4`, `f00584c`, `62c154a`, `c64a92d`, `9efe21c`,
  `b333143`, `ce5c5f8`, `e0a6b27`.

#### Feature 7 — `_before_training` / `_after_training` runtime hooks

- **Problem**: For each rollout iteration, the train loop needs to (a)
  reclaim train GPUs from infer, (b) wake train weights, (c) run
  `train()`, (d) export new weights to CPU cache, (e) push them to
  infer engines, (f) release train GPUs.
- **Files**:
  - `rlix_miles/rlix/pipeline/miles_pipeline.py:535-620` — the two hook
    methods
  - `miles/miles/utils/rlix_train_loop.py` — the `run_async_train_loop`
    helper that calls them
- **Implementation logic**:
  - `_before_training(step)`: request actor_train at priority
    `ACTOR_TRAINING` → scheduler preempts overlap GENERATION engines →
    `_wait_for_overlap_engines_offloaded` blocks until OS-level GPU mem
    is actually free (see Feature 8) → return.
  - `_after_training(step)`: build the new CPU bucket cache for this
    step → `train_group.offload()` → drive
    `coord.sync_base_weights_to_active(step)` → release actor_train.
- **Tests / verification**: M11.1 attempt 10 log: `step2: before_step
  done` … `step3: train_group.train done` … `step4: after_step done` —
  all four loop steps complete cleanly for both rollouts.
- **Commit**: rlix `18f296d`.

#### Feature 8 — `_wait_for_overlap_engines_offloaded` nvidia-smi probe

- **Problem**: SGLang's `release_memory_occupation` returns 200 OK
  before the CUDA driver has actually returned memory to the OS pool.
  The next `wake_up` would OOM.
- **Files**: `rlix_miles/rlix/pipeline/miles_pipeline.py:418-505`
- **Implementation logic**: two-phase wait:
  - **Phase 1**: poll `RolloutManager.get_engine_states` until the
    overlap engines reach state="offloaded" (60s timeout).
  - **Phase 2**: probe `nvidia-smi --query-gpu=memory.free --id=...`
    on the overlap GPUs every 0.5s until min free ≥ 20 GB
    (60s timeout). This is the actual "memory back to OS" signal.
- **Tests / verification**: M11.1 attempt 5 log:
  `_wait_for_overlap_engines_offloaded: OS-level GPU mem free min=28.20 GB`.
- **Commit**: rlix `e0a6b27`. (Engine-index conversion fix in
  Feature 11 below.)

### M11.1 — sub-fixes (3)

#### Feature 9 — `with_ref` from `args.use_kl_loss` / `kl_coef`

- **Problem**: `train_actor` crashes with `torch.cat(NoneType, dim=int)`
  in `policy_loss_function` (`miles/backends/training_utils/loss.py:735`).
  `batch["ref_log_probs"]` is None.
- **Root cause**: `MilesPipeline._init_phase_a_train` was constructing
  `RayTrainGroup(..., with_ref=False)`. Without ref-model loaded,
  `train_actor`'s `compute_log_prob(store_prefix="ref_")` is gated off
  (line 419: `if "ref" in self.weights_backuper.backup_tags`). Standalone
  miles derives `with_ref` from `args.kl_coef != 0 or args.use_kl_loss`
  (`miles/ray/placement_group.py:192`).
- **Files**: `rlix_miles/rlix/pipeline/miles_pipeline.py:163-180`
- **Implementation logic**: derive `with_ref` from args:
  ```python
  with_ref=bool(getattr(args, "use_kl_loss", False)
              or float(getattr(args, "kl_coef", 0.0)) != 0.0)
  ```
- **Tests / verification**: M11.1 attempt 9 log: full GRPO update
  with KL loss completes (~11s for the 0.5B model on 4xRTX5090).
- **Commit**: rlix `e0a6b27`.

#### Feature 10 — `_expand_workers` no-op for already-active engines

- **Problem**: After Phase B init, the rlix scheduler's gap-ratio
  planner fires the first GEN expand. But miles' RolloutManager brings
  engines up directly in `active` state (no shell→offloaded→active),
  so the coordinator's `_expand_workers` finds engines already active
  and would crash on the strict state-machine check.
- **Files**: `rlix_miles/rlix/pipeline/miles_coordinator.py` (around
  line 470)
- **Implementation logic**: in `_expand_workers`, if all target engines
  report `state="active"`, log a M11.x note and **skip** the wake/sync/
  activate_routing steps (still update the local `_active_engine_indices`
  bookkeeping). M11.1 single-pipeline ergonomic hatch; the F22
  shell-init contract is the proper M11.3 fix.
- **Tests / verification**: M11.1 attempt 10 log: scheduler INIT→GEN
  transition succeeds without state-machine error.
- **Commit**: rlix `e0a6b27` (around earlier iter 23 commit `2b73aef`).

#### Feature 11 — `_wait_for_overlap_engines_offloaded` engine-index conversion

- **Problem**: M11.2 P2 (pool=[2,3]) crashes with
  `KeyError('unknown engine_index 2')` from `RolloutManager.get_engine_states`.
- **Root cause**: the function converts physical GPU IDs to engine
  indices via `g // per_engine`. For M11.1 single-pipeline (pool starts
  at GPU 0) this happens to give the right local index. For M11.2 P2
  (pool starts at GPU 2), `2 // 1 = 2` — but the RolloutManager has
  engines at LOCAL indices [0, 1], not [2, 3].
- **Files**: `rlix_miles/rlix/pipeline/miles_pipeline.py:431-450`
- **Implementation logic**: read the infer pool's first physical GPU
  from `self._pipeline_config.cluster_device_mappings["actor_infer"]`,
  use it as the offset:
  ```python
  infer_first = min(infer_mapping)
  target_indices = sorted({(int(g) - infer_first) // per_engine
                          for g in allocated_train_gpus})
  ```
  Falls back to `range(rollout_num_gpus)` when `cluster_device_mappings`
  is empty (M11.1 backward compat).
- **Tests / verification**: M11.2 attempt 4 log: MP2's
  `_wait_for_overlap_engines_offloaded` no longer logs the KeyError;
  full dual-pipeline run completes EXIT_CODE=0.
- **Commit**: rlix `549cfbd`.

### M11.2 dual-pipeline (4)

#### Feature 12 — Dual-pipeline driver (`run_miles_dual.py`)

- **Problem**: M11.1's `run_miles_rlix.py` only spawns one pipeline.
  M11.2 needs two MilesCoordinator + MilesPipeline pairs running
  concurrently on disjoint GPU pools.
- **Files**: `miles/examples/rlix/run_miles_dual.py` (new, ~370 lines)
- **Implementation logic**:
  - `_split_pools_for_dual(num_gpus_per_node, infer_pool_size)`:
    splits the machine GPU range into 2 disjoint per-pipeline pools
    (e.g., `[0,1]` + `[2,3]`).
  - `_per_pipeline_args(base_args, pipeline_index)`: deep-copies args,
    suffixes `exp_name` with `mp{idx}`, resets `sglang_router_port` to
    None so each pipeline auto-allocates a fresh router port.
  - `_build_pipeline(...)`: per pipeline, computes
    `train_mapping=pool[:train_size]` and `infer_mapping=pool` (full
    overlap within the per-pipeline pool to satisfy the C4 topology
    check), registers `cluster_device_mappings` with the orchestrator,
    creates the MilesCoordinator (with per-pipeline runtime_env_vars
    including `MILES_ROLLOUT_BASE_PORT=15000+pipeline_index*1000`),
    creates the MilesPipeline, calls `initialize_pipeline`.
  - Main: builds P1 then P2 sequentially (init), then runs both train
    loops concurrently via `asyncio.gather(_run_one_pipeline(P1),
    _run_one_pipeline(P2))`.
- **Tests / verification**: M11.2 attempt 4 log: both pipelines
  initialize, train both rollouts, clean shutdown. Total run time after
  init: ~60s (warm caches).
- **Commit**: miles `992fa26`, `6126e01`.

#### Feature 13 — `cluster_device_mappings` threaded into placement_provider

- **Problem**: `MilesPipeline._build_placement_provider` was hardcoded
  to `train_device_mapping=range(actor_count)`,
  `infer_device_mapping=range(rollout_num_gpus)`. For M11.2 P2 (physical
  pool [2,3]), this would still pass [0,1], so the placement provider
  would reference the wrong physical GPUs.
- **Files**: `rlix_miles/rlix/pipeline/miles_pipeline.py:812-855`
- **Implementation logic**: read
  `self._pipeline_config.cluster_device_mappings` (dict[str,
  list[int]]) when present; pass `train_mapping` /
  `infer_mapping` from there to the `MilesPlacementProvider`. Falls
  back to the M11.1 `range(...)` default when the dict is empty.
- **Tests / verification**: M11.2 driver passes `cluster_device_mappings={
  "actor_train": [2], "actor_infer": [2,3]}` for MP2; placement provider
  log:
  `_build_placement_provider train=[2] infer=[2, 3]
   (from pipeline_config.cluster_device_mappings: True)`.
- **Commit**: rlix `d97178e`.

#### Feature 14 — `MILES_ROLLOUT_BASE_PORT` env var for port collision

- **Problem**: Two MilesPipeline actors on the same Ray cluster both
  call `find_available_port` concurrently with the same start port
  (15000) → race for the first probe → both bind, one crashes.
- **Files**: `miles/miles/ray/rollout.py:220-235`
- **Implementation logic**: `start_engines` reads
  `MILES_ROLLOUT_BASE_PORT` env var; falls back to 15000. The dual
  driver sets distinct values per pipeline (15000 for P1, 16000 for
  P2) via Ray `runtime_env`.
- **Tests / verification**: M11.2 attempt 4 SGLang ServerArgs log
  shows P1 engines on ports 15000-15007 range and P2 engines on
  16000-16007 range; no port collisions.
- **Commit**: miles `992fa26`.

#### Feature 15 — `ulimit -n 65536` for raylet under 2 pipelines

- **Problem**: `Failed to register worker to Raylet: IOError: Failed
  to read data from the socket: End of file` — raylet crashes with
  SIGABRT (`errno=24`, EMFILE).
- **Root cause**: default soft fd limit of 1024 is exhausted by 2
  MilesPipelines + ~20 SGLang sub-processes opening sockets.
- **Files**: `rlix_miles/scripts/run_smoke_dual.sh` (top of file)
- **Implementation logic**: `ulimit -n 65536` before any Ray
  operation in the smoke script.
- **Tests / verification**: M11.2 attempt 2→3 transition: raylet no
  longer hits SIGABRT once the smoke script bumps the limit.
- **Commit**: scripts/ is gitignored; documented in
  `plans/m11-2-dual-pipeline-log.md`.

### M11.1 — additional driver-side fixes (2)

#### Feature 16 — Coordinator name + namespace must match scheduler RPC lookup

- **Problem**: `resize_infer` / `shrink_engines` RPCs from the rlix
  scheduler fail with `Failed to resolve actor`, and the central
  scheduling loop crashes when it tries to preempt GENERATION workers
  for an `ACTOR_TRAINING` request.
- **Root cause**: rlix's scheduler resolves the per-pipeline coordinator
  using `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` in the
  registered `ray_namespace` (`pipeline_<pipeline_id>_NS`). The original
  driver was naming the coordinator `f"miles_coordinator_{pipeline_id}"`
  in `RLIX_NAMESPACE` — wrong on both axes.
- **Files**:
  - `miles/examples/rlix/run_miles_rlix.py:191-208` — coordinator
    `.options(name=..., namespace=...)` block
  - `miles/examples/rlix/run_miles_dual.py:158-170` — same per-pipeline
- **Implementation logic**:
  ```python
  coordinator = (
      ray.remote(MilesCoordinator)
      .options(
          name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
          namespace=pipeline_namespace,
          lifetime="detached",
          ...
      )
      .remote(...)
  )
  ```
- **Tests / verification**: M11.1 attempt 4→5 transition: scheduler
  preempt path fires `coordinator.resize_infer` successfully (was
  raising "Failed to resolve actor" before).
- **Commit**: miles `7b83be5` (and forwarded into `run_miles_dual.py`
  in `992fa26`).

#### Feature 17 — `step_target_estimate` forwarding for ACTOR_TRAINING priority transition

- **Problem**: After Phase B step 8 transitions actor_infer from
  `INITIALIZATION` to `GENERATION`, the rlix scheduler's gap-ratio
  planner needs a target estimate of how many steps the GEN cluster
  should run before the next ACTOR_TRAINING request. Without it, the
  planner can hang on the first preempt because it doesn't know when
  the train wants the GPUs back.
- **Files**:
  - `rlix_miles/rlix/pipeline/miles_pipeline.py:389-411` — Phase B step
    8 re-request: line 397 derives `gen_step_target_estimate`, line 406
    passes it on `_request_cluster_gpus(...)` for the GEN cluster
  - `rlix_miles/rlix/pipeline/miles_pipeline.py:574` — `_before_training`
    forwards `global_step=int(step)` so the scheduler tracks per-step
    cadence on subsequent ACTOR_TRAINING requests
- **Implementation logic**: pass `step_target_estimate` (derived from
  the rollout's `cluster_tp_configs` × dynamic batch size) on the
  `_request_cluster_gpus` call when transitioning actor_infer
  INIT→GEN. The scheduler then knows to plan the next ACTOR_TRAINING
  preempt at that step boundary.
- **Tests / verification**: M11.1 attempt 4 log: scheduler GEN expand
  + ACTOR_TRAINING preempt fire on cadence; no hang on first train
  step.
- **Commit**: rlix `2b73aef` + refinements through `e0a6b27`.

---

## §4 Code paths (function-by-function)

### `rlix/pipeline/miles_pipeline.py`

| Line range | Symbol | Used by |
|---|---|---|
| 61-88 | `MilesPipeline.__init__` | `MilesCoordinator.create_pipeline_actor` |
| 133-241 | `_init_phase_a_train` (incl. `RayTrainGroup(..., with_ref=...)` construction at 163-180) | `initialize_pipeline` |
| 242-403 | `_init_phase_b_infer` (incl. step_target_estimate INIT→GENERATION at 390+) | `initialize_pipeline` |
| 418-505 | `_wait_for_overlap_engines_offloaded` (Features 8 + 11) | `_before_training` |
| 566-596 | `_before_training` (Feature 7) | `before_training` Ray method |
| 598-660 | `_after_training` (Feature 7) | `after_training` Ray method |
| 717-721 | `before_training` / `after_training` public Ray methods | rlix_train_loop |
| 389-411 | step_target_estimate derivation + forwarding (Phase B step 8 INIT→GEN) | scheduler gap-ratio planner |
| 574 | `global_step=int(step)` forwarded on each ACTOR_TRAINING `_request_cluster_gpus` | scheduler step cadence |
| 833-884 | `_build_placement_provider` (Feature 13) | `_init_phase_a_train` step 1 |

### `rlix/pipeline/miles_coordinator.py`

| Line range | Symbol | Note |
|---|---|---|
| 80-175 | `MilesCoordinator.__init__` | |
| 258-275 | `_inject_pipeline_env_vars` | for runtime_env propagation |
| 399-545 | `_expand_workers` (Feature 10) | called by scheduler's GEN expand |
| 462-505 | already-active no-op branch (M11.1 hatch) | |

### `rlix/pipeline/miles_model_update_service.py`

| Line range | Symbol | Note |
|---|---|---|
| 128-313 | `sync_selected_workers` | weight-sync entry point |
| 288-318 | pause_generation / finalize / continue_generation (Feature 3) | |

### `miles/miles/ray/rollout.py`

| Line range | Symbol | Note |
|---|---|---|
| 220-235 | `start_engines` MILES_ROLLOUT_BASE_PORT (Feature 14) | |
| 863-940 | `RolloutManager.shrink_engines` (Feature 2) | |
| 911-928 | pause_generation pre-release | |
| 1605-1622 | `start_rollout_servers` rlix-mode override (Feature 1) | |

### `miles/examples/rlix/run_miles_rlix.py`

Single-pipeline driver (M11.1). Key blocks:

| Line range | Block |
|---|---|
| 22-28 | F08/F41 RLIX_CONTROL_PLANE env-var guard before heavy imports |
| 31-46 | `_build_cluster_device_mappings` |
| 49-262 | `main()` (lazy heavy imports + driver flow steps 1-10) |

### `miles/examples/rlix/run_miles_dual.py` (Feature 12)

Dual-pipeline driver (M11.2). Key blocks:

| Line range | Block |
|---|---|
| 50-66 | `_split_pools_for_dual` |
| 75-97 | `_per_pipeline_args` |
| 99-200 | `_build_pipeline` |
| 220-377 | `main()` — topology calc, sequential init of P1+P2, asyncio.gather train loops |

---

## §5 Known limitations / deferred work

| ID | What | Why deferred | Impact |
|---|---|---|---|
| **F22** | Strict shell→offloaded→active state machine for engine init | Receiver-side shell-creation code never wired into `RolloutManager.expand_engines`. M11.1 uses a pragmatic "engines come up active" hatch (Feature 10). | Multi-pipeline (3+) needs F22 so the scheduler can manage engines that come up shell. M11.2 disjoint-pool topology works without it. |
| **M11.3** | Multi-pipeline (3+) on small GPU pool with cross-pipeline contention | Topologically requires F22 + 1-GPU train shape. Not exercised by current smoke. | Three concurrent pipelines on 4 GPUs not validated. |
| **F19/F20 (saving)** | Final-rollout save trigger | Megatron `save_model` doesn't onload weights internally; `_after_training` already offloaded. | Smoke runs pass `--save ""`. Real fix: explicit onload→save→offload bracket. |
| **Eval at final rollout** | `should_run_periodic_action` fires at the last rollout regardless of `--eval-interval` | Same offloaded-weight problem as save | Smoke skips eval. |
| **Pytest coverage of M11 changes** | The 6 critical files have NO direct unit/integration tests | Time pressure; smoke run is the main verification | Future iteration should add unit tests for `_wait_for_overlap_engines_offloaded`, `pause_generation` wrapping, `_split_pools_for_dual`, etc. |
| **Vast-only `_with_region_config` patch** | Nested-region clobber in tms 0.0.9 entrypoint.py | Upstream tms bug | Hotpatched on the vast image only; not in repo. Tracked in tms-fixes.md §5. |

---

## Appendix A — Verification log map

The two append-only iteration logs are the source of truth for what was
exercised end-to-end on real GPUs:

- `plans/m11-e2e-test-log.md` — M11.1 attempts 0–10 (10 fix-retest
  cycles, ~3.5 hr vast time). Final attempt 10 = EXIT_CODE=0 with both
  rollouts trained on 4xRTX5090.
- `plans/m11-2-dual-pipeline-log.md` — M11.2 attempts 0–4 (4 fix-retest
  cycles, ~33 min vast time). Final attempt 4 = EXIT_CODE=0 with both
  pipelines reaching shutdown_hard on 4xA40.

## Appendix B — Commit map

| Branch | HEAD | Contents |
|---|---|---|
| rlix `zhenyu/miles-mvp-e2e` | `5dc4e43` | Features 1–11 (rlix-side), tms-fixes doc, both test logs |
| miles `zhenyu/m11-mvp-test` | `6126e01` | Features 1, 2, 12, 14, miles-side rlix-mode wiring |

Both branches are pushed to GitHub:
- https://github.com/rlops/rlix
- https://github.com/rlops/miles
