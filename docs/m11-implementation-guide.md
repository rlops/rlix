# M11 Implementation Guide — F1–F12 walkthrough

> **Audience: code reviewer who has not opened this repo before.**
>
> Goes top-down: what the system does (§1), the architecture in 30 seconds (§2), then walks every feature **F1–F12 from the unified port plan** (§3) — for each one: what it does, where it lives now, what bugs M11.1 / M11.2 surfaced, how each was fixed (with file:line), and what we ran end-to-end to confirm. §4 indexes every M11.1+M11.2 fix back to its F1–F12 owner. §5 lists deferred work.
>
> Source plan: [`plans/miles-port-unified-plan.md`](../plans/miles-port-unified-plan.md). Each §3 entry below mirrors the same F-number used there.

---

## §1 What this is

### 1.1 The problem

RL training for LLMs has two GPU-heavy stages:
1. **Rollout** (inference): generate trajectories with the current policy on an inference engine like SGLang.
2. **Train** (back-prop): update the policy with PPO/GRPO via Megatron.

A single experiment alternates between them. If train and rollout run on **disjoint GPUs**, one pool idles while the other works — wasted capacity. The original ROLL framework handled this by introducing a per-cluster **time-share scheduler**; this port brings MILES (SGLang + Megatron) under that same scheduler.

### 1.2 What rlix does (the new control plane)

`rlix/` is a Ray-based GPU time-share controller. Per-pipeline actors register their `cluster_device_mappings` (e.g. `{"actor_train": [0,1], "actor_infer": [0,1,2,3]}`) with a central scheduler (`rlix.scheduler.SchedulerImpl`). The scheduler arbitrates GPU allocations by priority:

```
INITIALIZATION (0) > ACTOR_TRAINING (1) > … > GENERATION (6)
```

When an `ACTOR_TRAINING` request arrives for a GPU currently held by a preemptable `GENERATION` worker, the scheduler tells the per-pipeline coordinator to **shrink the inference engine** (release overlap GPUs), then grants them to the trainer. After training, the GPU is released and inference can re-expand.

`cluster_id` strings have the form `f"{pipeline_id}_{cluster_name}"` (e.g. `miles_c2dda50d955a_actor_train`). See `rlix/pipeline/miles_pipeline.py:67-70`. Per-pipeline namespace + coordinator name details are in F7 below.

### 1.3 What miles is (the runtime under the new control plane)

`miles/` is the actual training stack: SGLang (`miles/backends/sglang_utils/`), Megatron-LM (`miles/backends/megatron_utils/`), Ray actors gluing them together (`miles/ray/`). It already has a standalone (non-rlix) entry point in `miles/examples/fully_async/train_async.py`.

**rlix-mode** is when miles runs *under* rlix's scheduler instead of standalone. The integration glue lives in:
- `rlix/pipeline/miles_pipeline.py` — per-pipeline init + runtime hooks (Phase A train / Phase B infer)
- `rlix/pipeline/miles_coordinator.py` — per-pipeline coordinator (resize, sync, register-resources)
- `rlix/pipeline/miles_model_update_service.py` — atomic weight-sync transport
- `miles/examples/rlix/run_miles_rlix.py` — single-pipeline driver (M11.1)
- `miles/examples/rlix/run_miles_dual.py` — dual-pipeline driver (M11.2)

### 1.4 What M11 is

M11 is the milestone for **first end-to-end working rlix-mode**.

| ID | Scope | Status |
|---|---|---|
| **M11.1** | Single MilesPipeline, partial-overlap topology (`actor_train ⊂ actor_infer`), full Qwen2.5-0.5B GRPO loop with `--num-rollout 2` | ✅ GREEN |
| **M11.2** | Two concurrent MilesPipelines on disjoint pools (P1=[0,1], P2=[2,3]), each running its own GRPO loop | ✅ GREEN |
| **M11.3+** | 3+ pipelines, runtime preempt under contention, production hardening | Deferred |

Verification rig: vast.ai 4xGPU instances (RTX5090 for M11.1; A40 for M11.2), Qwen2.5-0.5B GRPO with 2 rollouts. EXIT_CODE=0 + clean shutdown is the pass bar. Iteration logs:
- `plans/m11-e2e-test-log.md` — M11.1 attempts 0–10
- `plans/m11-2-dual-pipeline-log.md` — M11.2 attempts 0–4

### 1.5 5-minute reading order

1. §2 architecture diagram below.
2. §3 F1 (memory_saver gating) — most representative tms / SGLang fix.
3. §3 F2 (selective sleep/wake) — the abort-drain-sleep ordering.
4. §3 F4 (CPU bucket cache) — the hotpath for weight updates.
5. §3 F11 (RLix conditional flag) — explains the `RLIX_CONTROL_PLANE=rlix` gating you'll see in many fixes.
6. §3 F8 (pipeline registration) — the M11.2 dual-pipeline driver.
7. §5 deferred work — so you know what NOT to ask about.

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
- **Driver** is the user-facing Python process; it stays single-threaded except for the `asyncio.run` train loop. F13 contract: no top-level `try/except`, no `ray.shutdown()` — failures propagate (`ray stop` is the recovery surface).
- **MilesCoordinator + MilesPipeline + MilesModelUpdateService** are one Ray actor each. The orchestrator + scheduler live in `RLIX_NAMESPACE` (`"rlix"`); the per-pipeline coordinator MUST live in the per-pipeline namespace `pipeline_<pipeline_id>_NS` with the name `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` so the scheduler's `resize_infer` / `shrink_engines` RPCs can resolve it. See F7.
- **RayTrainGroup + RolloutManager** are miles' existing Ray actor groups; `RayTrainGroup` wraps Megatron train actors (one per train GPU), `RolloutManager` wraps SGLang rollout engines. M11 changes do not modify their public APIs.

---

## §3 Feature catalog (F1–F12)

For each feature: **Goal** (1-line summary from port plan) → **Where it lives** (current file:line refs) → **M11 issues found** (bugs the smoke surfaced) → **How fixed** (file:line + commit) → **Verification** (smoke evidence). Ports that landed before M11 are noted but not re-opened — only items that broke during M11.1 / M11.2 are detailed.

---

### F1 — SGLang sleep/wake with `release_memory_occupation`

**Source plan**: §Feature 1 (`miles-port-unified-plan.md:366`).

**Goal.** Release SGLang inference engine VRAM (weights + KV cache) so train can use those GPUs; wake_up to reload. SGLang's API is tag-based (`release_memory_occupation(tags=None)` ≡ vLLM `sleep(level=2)`); call site requires `is_fully_idle()`.

**Where it lives now.**
- `miles/miles/backends/sglang_utils/sglang_engine.py:1057` — `enable_memory_saver` is gated on `args.offload_rollout` at SGLang `ServerArgs` build time.
- `miles/miles/ray/rollout.py:872-942` — `RolloutManager.shrink_engines` (selective release) calls `release_memory_occupation`.
- `miles/miles/ray/rollout.py:1616-1628` — `start_rollout_servers` decides per-engine `needs_offload`.
- `miles/miles/backends/megatron_utils/actor.py:96-111` — `MILES_TMS_HOOK_MODE` env var picks between torch_memory_saver `preload` (default) vs `torch` (CUDAPluggableAllocator) hook modes.

**M11 issues found.**

1. **Issue (M11.1 attempt 5)** — `release_memory_occupation` returned 200 OK and the rollout-manager flipped to `state="offloaded"`, but `nvidia-smi` showed the engine process still holding ~30 GB. Next train wake_up OOMed.
   - **Root cause.** SGLang only releases memory when launched with `enable_memory_saver=True`. miles gates that on `args.offload_rollout`. But for the partial-overlap topology, miles' `start_rollout_servers` math (`miles/ray/rollout.py:1586`) computed `group_abs_start=2, megatron_num_gpus=2 → 2<2 false → needs_offload=False`, forcing `enable_memory_saver=False` regardless of `--offload-rollout`.
   - **Fix.** rlix-mode override at `miles/miles/ray/rollout.py:1627-1628`: when `RLIX_CONTROL_PLANE=rlix`, force `needs_offload=True`. The smoke script (`scripts/run_smoke_e2e.sh`) also passes `--offload-rollout`. Commits: rlix `e0a6b27`, miles `f58b365`.
   - **Verification.** M11.1 attempt 5 log: `OS-level GPU mem free min=28.20 GB` immediately after `state="offloaded"` (was 1.93 GB before).

2. **Issue (M11.1 attempt 1)** — torch_memory_saver default `hook_mode="preload"` (LD_PRELOAD libc malloc) segfaulted during Phase A `build_cpu_bucket_cache` (F4) on CUDA 12.9 / Blackwell.
   - **Root cause.** `preload` intercepts every libc malloc; Blackwell + CUDA 12.9 mempool internals trip unsafe interception paths.
   - **Fix.** Smoke script (`scripts/run_smoke_e2e.sh`) exports `MILES_TMS_HOOK_MODE=torch` BEFORE any tms call; switches tms to PyTorch's `CUDAPluggableAllocator` path (narrower catchment). Commit: miles `f58b365` (the env var was already supported at `miles/backends/megatron_utils/actor.py:96-111`; M11 just sets it).
   - **Verification.** Phase A `build_cpu_bucket_cache` completes without segfault on the vast 4xRTX5090 image.

---

### F2 — Selective engine sleep/wake (partial sleep/wake)

**Source plan**: §Feature 2 (`miles-port-unified-plan.md:455`).

**Goal.** Sleep ONLY the engines on overlap GPUs; non-overlap engines keep serving. Requires (a) abort in-flight requests on target engines, (b) drain via SGLang `/v1/loads`, (c) call `release_memory_occupation`, (d) maintain a 5-state `EngineInfo` machine (`shell` / `active` / `disabling` / `offloaded` / `loading`) as the single source of truth.

**Where it lives now.**
- `miles/miles/ray/rollout.py:84-100` — `EngineInfo` dataclass (5-state).
- `miles/miles/ray/rollout.py:872-942` — `RolloutManager.shrink_engines` (selective release with abort + drain + SGLang release).
- `miles/miles/ray/rollout.py:968-1008` — `RolloutManager.expand_engines` (load + state→loading then `activate_routing`).
- `miles/miles/ray/rollout.py:1009-1035` — `RolloutManager.finish_init_offload` (M11.2 path: drop weights immediately after init for Pipeline B).
- `miles/miles/ray/rollout.py:1036-end` — `RolloutManager.activate_routing`.

**M11 issues found.**

1. **Issue (M11.1 attempts 5–7)** — `release_memory_occupation` failed with `Pointer argument cannot be accessed from Triton (cpu tensor?)`, then `Timeout while flushing cache` (60s loop).
   - **Root cause.** `is_idle=True` only means request queue is empty; the SGLang scheduler thread can still be mid-decode-iteration. The scheduler launches a Triton kernel against persistent token-pool buffers AFTER `is_idle` but BEFORE release moves them to CPU. A naive `time.sleep(0.5)` workaround does not quiesce the scheduler.
   - **Fix.** In `RolloutManager.shrink_engines`, between the `is_idle` drain and `release_memory_occupation`, fan-out `pause_generation(mode="retract")` to all target engines (rlix-mode only; gated on `RLIX_CONTROL_PLANE=rlix`). Wrapped in `try/except` so an "already paused" rejection does not block the loop. File: `miles/miles/ray/rollout.py:911-942` (the call is at line 933). Commit: miles `f58b365`.
   - **Verification.** M11.1 attempt 7 log: `POST /pause_generation HTTP/1.1 200 OK` precedes `POST /release_memory_occupation HTTP/1.1 200 OK`; train wake_up no longer hits the Triton crash.

2. **Issue (M11.2 attempt 4)** — Pipeline-2's `_wait_for_overlap_engines_offloaded` crashed with `KeyError('unknown engine_index 2')`.
   - **Root cause.** The Phase 2 GPU-mem probe converts physical GPU IDs to engine indices via `gpu // per_engine`. For M11.1 single-pipeline (pool starts at GPU 0) this was an identity. For M11.2 P2 (pool starts at GPU 2), `2 // 1 = 2` — but the per-pipeline `RolloutManager` has engines at LOCAL indices `[0, 1]`, not `[2, 3]`.
   - **Fix.** In `_wait_for_overlap_engines_offloaded` (`rlix/pipeline/miles_pipeline.py:418-505`, conversion at 431-450), read the infer pool's first physical GPU from `cluster_device_mappings["actor_infer"]` and subtract it as an offset:
     ```python
     infer_first = min(infer_mapping)
     target_indices = sorted({(int(g) - infer_first) // per_engine for g in allocated_train_gpus})
     ```
     Falls back to `range(rollout_num_gpus)` when `cluster_device_mappings` is empty (M11.1 backward compat). Commit: rlix `549cfbd`.
   - **Verification.** Fresh M11.2 dual smoke at HEAD `5dc4e43`/`6126e01` (2026-05-08) — `EXIT_CODE=0`, `0` `KeyError`, `0` `engine_index` warnings (see `plans/m11-review.debug.md` Run 1).

---

### F3 — Generation routing skip sleeping engines (router admission + 0-active suspend)

**Source plan**: §Feature 3 (`miles-port-unified-plan.md:643`); see also AC13, C20.

**Goal.** Router must **not** dispatch generation to engines that have been shrunk. M11.2 adds the harder case: between scheduler's `resize_infer(remove)` and the next `resize_infer(add)`, the active set can be empty. Per C20 the router must **suspend** (block-with-notify on `asyncio.Condition`), NOT raise `RuntimeError`.

**Where it lives now.**
- `miles/miles/router/router.py:43-67` — `_workers_changed: asyncio.Condition` + 4-dict state (enabled / disabled / dead / preempted).
- `miles/miles/router/router.py:514-558` — `_use_url` (sync, test-only) and `_use_url_async` (async, production proxy path) waiting on `_workers_changed.wait_for(predicate)` until candidate set non-empty.
- `miles/miles/router/router.py:333-385` — every state-mutating endpoint (`add_worker`, `enable_worker`, `disable_worker`, `remove_worker`) ends with `await self._notify_workers_changed()` to wake suspended dispatch coroutines.

**M11 issues found.** None unique to M11.1 / M11.2 happy path on the vast smoke. The 0-active suspend window is only reached in the runtime preempt cycle (after init `Priority.INIT` full allocation drains all engines to `offloaded`); on the M11.1 smoke this transition rides through cleanly. R01 review (`plans/m11-review.review-report/R01.md`) flagged a NOTE: the broad `except Exception` catch in shrink masks SGLang API contract violations — non-blocking, deferred.

---

### F4 — Training-side weight caching (CPU bucket cache + `_cache_ready_step`)

**Source plan**: §Feature 4 (`miles-port-unified-plan.md:1256`); see also AC4, AC5, AC11, F18 (cache-owner uniqueness).

**Goal.** After each train step, the cache_owner Megatron actor (= `pp0+dp0+tp0+cp0`) exports new weights to a CPU bucket cache; subsequent `sync_selected_workers` reads from this cache and pushes to SGLang via `cpu_serialize` (tmpfs `/dev/shm/miles_cpu_bucket_{uuid}.pt`) or NCCL broadcast. Two-stage mental model: **build → publish** (sender side) / **read → finalize → publish version** (receiver side).

**Where it lives now.**
- `miles/miles/backends/megatron_utils/update_weight/cpu_bucket_cache.py` — the CPU bucket cache module (build + read).
- `miles/miles/ray/actor_group.py:182-195` — `RayTrainGroup.build_cpu_bucket_cache(step)` fan-out.
- `rlix/pipeline/miles_coordinator.py:327-340` — `MilesCoordinator.publish_cache_ready_step` (publishes the new step to the model-update service).
- `rlix/pipeline/miles_coordinator.py:367-401` — `MilesCoordinator.sync_base_weights_to_active(step)` — drives the atomic-unit sync.
- `rlix/pipeline/miles_model_update_service.py:128-225` — `MilesModelUpdateService.sync_selected_workers` (the atomic unit: transport + finalize + version publish under one timeout).

**M11 issues found.**

1. **Issue (M11.1 attempt 8)** — `[torch_memory_saver.cpp] Cannot resume allocation that is not paused. tag=default ptr=...`
   - **Root cause.** `MilesPipeline._before_training` was calling `self._run_async(self._train_group.onload())` (broadcast `wake_up`). But `MegatronTrainRayActor.train()` ALSO calls `self.wake_up()` when `args.offload_train=True` (`miles/backends/megatron_utils/actor.py:357`). Two resumes back-to-back → the second finds the region already resumed and crashes.
   - **Fix.** Remove the explicit `train_group.onload()` in `_before_training` (`rlix/pipeline/miles_pipeline.py:566-596`). Train wake_up now happens exactly once, inside `train()`. Matches standalone `train_async.train` behavior. Commit: rlix `e0a6b27`.
   - **Verification.** M11.1 attempt 8 log: exactly one `Timer wake_up end` per train step (~0.7s elapsed); no "Cannot resume" error.

2. **Issue (M11.1 attempt 9)** — `train_actor` crashed with `torch.cat(NoneType, dim=int)` in `policy_loss_function` (`miles/backends/training_utils/loss.py:735`); `batch["ref_log_probs"]` was None.
   - **Root cause.** `MilesPipeline._init_phase_a_train` was constructing `RayTrainGroup(..., with_ref=False)`. Without ref-model loaded, `train_actor.compute_log_prob(store_prefix="ref_")` is gated off (`miles/backends/training_utils/train_actor.py:419` checks `if "ref" in self.weights_backuper.backup_tags`). Standalone miles derives `with_ref` from KL flags (`miles/ray/placement_group.py:192`).
   - **Fix.** Derive `with_ref` from args (`rlix/pipeline/miles_pipeline.py:163-180`):
     ```python
     with_ref=bool(getattr(args, "use_kl_loss", False)
                 or float(getattr(args, "kl_coef", 0.0)) != 0.0)
     ```
     Commit: rlix `e0a6b27`.
   - **Verification.** M11.1 attempt 9 log: full GRPO update with KL loss completes (~11s for 0.5B on 4xRTX5090).

---

### F5 + F6 — Two-path weight refresh (active in-flight + expand sync) + version accounting

**Source plan**: §Feature 5+6 (`miles-port-unified-plan.md:2198`); see also AC4, AC11, C13, C16, F15 (atomic-unit), F19 (active-set bootstrap), F20 (`_resize_sync_lock`), F21 (no per-bucket version).

**Goal.** Two refresh paths share the same atomic unit (`MilesModelUpdateService.sync_selected_workers(sync_id, targets, version)`):
- **Active in-flight refresh.** Engines stay serving; `pause_generation` brackets `finalize_weight_update`; engine reports new version even while in-flight requests may finish on old weights (A8 — bounded mis-attribution).
- **Expand sync.** A previously-offloaded engine wakes, gets a base sync (possibly `version=-1` from init-built CPU bucket per C13), then opens routing.

Version is published exactly once per sync (`manager.set_weight_version`), NOT per bucket (F21).

**Where it lives now.**
- `rlix/pipeline/miles_model_update_service.py:32-77` — `SyncSessionPlan` frozen dataclass.
- `rlix/pipeline/miles_model_update_service.py:101-225` — `MilesModelUpdateService.__init__` + `sync_selected_workers` (atomic unit: cancel inflight on timeout, single `_run_atomic_unit` body covers transport + finalize + version publish).
- `rlix/pipeline/miles_model_update_service.py:248-338` — `_run_atomic_unit` (executes the plan).
- `rlix/pipeline/miles_model_update_service.py:288-318` — pause_generation/finalize/continue_generation wrap.
- `rlix/pipeline/miles_coordinator.py:439-513` — `_expand_workers` (calls `sync_base_weights_to_active(-1)` for `version=-1` bootstrap path).

**M11 issues found.**

1. **Issue (M11.1 attempt 7)** — After the FIRST successful train step, the post-step `MilesCoordinator.sync_base_weights_to_active(0)` → `MilesModelUpdateService.sync_selected_workers` → `SGLangEngine.finalize_weight_update` hit a `flush_cache` 60s timeout (now during weight sync, not engine release).
   - **Root cause.** Same class as F2 issue 1: with a fully-async rollout function, the rollout-data return does not synchronously quiesce the engine. Pending decode batches keep the queue non-empty; `flush_cache` never returns 200.
   - **Fix.** In `sync_selected_workers` step 4, before the `finalize_weight_update` fan-out, call `pause_generation(mode="retract")` on each handle; after finalize completes, call `continue_generation` to resume. Mirrors the shrink-path fix. File: `rlix/pipeline/miles_model_update_service.py:288-318`. Commit: rlix `e0a6b27`.
   - **Verification.** M11.1 attempt 10 log: full first iteration (rollout 0 train + after_step + sync_base_weights) completes; rollout 1 trained; `EXIT_CODE=0`.

2. **Issue (M11.1 attempt 4)** — `_expand_workers` raised on the first scheduler INIT→GENERATION transition because miles' `RolloutManager` brings engines up directly in `state="active"` (no shell→offloaded→active path).
   - **Root cause.** F22 (the strict shell-init contract) was deferred — the receiver-side shell-creation code was never wired into `RolloutManager.expand_engines`.
   - **Fix.** `_expand_workers` no-op for already-active engines (`rlix/pipeline/miles_coordinator.py:439-513`, branch around line 470): if all target engines report `state="active"`, log a M11.x note and **skip** the wake/sync/activate_routing steps (still update local `_active_engine_indices` bookkeeping). Commit: rlix `2b73aef`. Pragmatic M11.1 hatch; F22 is the proper M11.3 fix (see §5).
   - **Verification.** M11.1 attempt 10 log: scheduler INIT→GEN transition succeeds without state-machine error.

---

### F7 — Per-pipeline Ray namespace isolation

**Source plan**: §Feature 7 (`miles-port-unified-plan.md:2796`); see also AC9.

**Goal.** Each pipeline's actors live in their own Ray namespace `pipeline_<pipeline_id>_NS`; the central scheduler resolves them via `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"`. Without namespace isolation, dual-pipeline runs would collide on actor names.

**Where it lives now.**
- `rlix/protocol/types.py:1-70` — `COORDINATOR_ACTOR_NAME_PREFIX`, `RLIX_NAMESPACE`, `get_pipeline_namespace(pipeline_id)`.
- `rlix/scheduler/scheduler.py:1196-1221` — scheduler's per-pipeline coordinator resolution path.
- `miles/examples/rlix/run_miles_rlix.py:191-208` — driver-side coordinator `.options(name=..., namespace=...)` block.
- `miles/examples/rlix/run_miles_dual.py:158-170` — same per-pipeline in the dual driver.

**M11 issues found.**

1. **Issue (M11.1 attempt 4)** — `resize_infer` / `shrink_engines` RPCs from the rlix scheduler failed with `Failed to resolve actor`; the central scheduling loop crashed when it tried to preempt GENERATION workers for an `ACTOR_TRAINING` request.
   - **Root cause.** rlix's scheduler resolves the per-pipeline coordinator using `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` in `get_pipeline_namespace(pipeline_id)`. The original driver was naming the coordinator `f"miles_coordinator_{pipeline_id}"` in `RLIX_NAMESPACE` — wrong on both axes.
   - **Fix.** Use the canonical name + namespace (`miles/examples/rlix/run_miles_rlix.py:191-208` and `_dual.py:158-170`):
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
     Commit: miles `7b83be5` (forwarded into `run_miles_dual.py` in `992fa26`).
   - **Verification.** M11.1 attempt 4→5 transition: scheduler preempt path fires `coordinator.resize_infer` successfully (was raising `Failed to resolve actor` before).

---

### F8 — Pipeline registration lifecycle (M11.1 single + M11.2 dual)

**Source plan**: §Feature 8 (`miles-port-unified-plan.md:2840`); see also AC1, AC9, AC10, F22 init bootstrap pre-registration.

**Goal.** Driver entry point (a) calls `rlix.init()`, (b) `orchestrator.allocate_pipeline_id`, (c) `orchestrator.register(pipeline_id, cluster_device_mappings, ...)`, (d) `orchestrator.admit(pipeline_id, priority=INIT)`, (e) spawns a `MilesCoordinator` Ray actor in the per-pipeline namespace, (f) calls `coordinator.create_pipeline_actor` which itself calls `pipeline.initialize_pipeline` running Phase A (train) + Phase B (infer) in sequence. M11.2 runs steps (a)-(f) for two pipelines, then runs both train loops via `asyncio.gather`.

**Where it lives now.**
- `rlix/orchestrator/orchestrator.py` — `allocate_pipeline_id` (UUID-based, collision-checked).
- `rlix/pipeline/miles_pipeline.py:118-130` — `initialize_pipeline` orchestration.
- `rlix/pipeline/miles_pipeline.py:133-241` — `_init_phase_a_train` (steps 1–7: request train GPUs, build `RayTrainGroup`, `build_cpu_bucket_cache(-1)`, offload, publish `_cache_ready_step=-1`, release).
- `rlix/pipeline/miles_pipeline.py:242-411` — `_init_phase_b_infer` (steps 1–8: request infer GPUs, placement provider, `RolloutManager`, `register_model_update_resources`, `bootstrap_active_engines`, `sync_base_weights_to_active(-1)`, INIT→GEN transition with `step_target_estimate`).
- `miles/examples/rlix/run_miles_rlix.py` — single-pipeline driver (M11.1).
- `miles/examples/rlix/run_miles_dual.py` — dual-pipeline driver (M11.2): `_split_pools_for_dual` (50–66), `_per_pipeline_args` (75–97, deep-copy + per-pipeline exp_name/router-port reset), `_build_pipeline` (99–200), `main` (220–377; sequential init, concurrent train via `asyncio.gather`).

**M11 issues found.**

1. **Issue (M11.1 attempt 4)** — Phase B step 8 hung on first ACTOR_TRAINING preempt because the rlix scheduler's gap-ratio planner did not know how many steps to plan for between INIT→GEN transitions.
   - **Root cause.** `step_target_estimate` was missing on the INIT→GEN re-request.
   - **Fix.** `rlix/pipeline/miles_pipeline.py:389-411` — Phase B step 8 derives `gen_step_target_estimate` (line 397) and passes it on `_request_cluster_gpus(...)` for the GEN cluster (line 406). Per-step cadence is also forwarded on `_before_training` via `global_step=int(step)` (`rlix/pipeline/miles_pipeline.py:574`). Commit: rlix `2b73aef` + refinements through `e0a6b27`.
   - **Verification.** M11.1 attempt 4 log: scheduler GEN expand + ACTOR_TRAINING preempt fire on cadence; no hang on first train step.

2. **Issue (M11.2 attempt 1)** — `Failed to register worker to Raylet: IOError: ... End of file` — raylet crashed with SIGABRT (`errno=24`, EMFILE).
   - **Root cause.** Default soft fd limit of 1024 exhausted by 2 MilesPipelines + ~20 SGLang sub-processes.
   - **Fix.** `scripts/run_smoke_dual.sh` raises `ulimit -n 65536` before any Ray operation. (Scripts dir is workspace-only, gitignored; documented in `plans/m11-2-dual-pipeline-log.md` attempt 2→3.)
   - **Verification.** M11.2 attempt 3 onward: raylet stable across both pipelines.

3. **Issue (M11.2 attempt 1)** — Two `MilesPipeline` actors on the same Ray cluster both called `find_available_port` concurrently with the same start port (15000) → race for the first probe → both bound, one crashed.
   - **Root cause.** Hardcoded base port; no per-pipeline offset.
   - **Fix.** `miles/miles/ray/rollout.py:220-235` — `start_engines` reads `MILES_ROLLOUT_BASE_PORT` env var (falls back to 15000). The dual driver sets distinct values per pipeline (15000 / 16000) via Ray `runtime_env`. Commit: miles `992fa26`.
   - **Verification.** M11.2 attempt 4 SGLang ServerArgs log shows P1 engines on 15000-15007 range and P2 engines on 16000-16007; no port collisions.

4. **Issue (M11.2 attempt 4 — non-fatal)** — `_build_placement_provider` was hardcoded to `train_device_mapping=range(actor_count), infer_device_mapping=range(rollout_num_gpus)`. For M11.2 P2 (physical pool [2,3]), this would still pass [0,1].
   - **Root cause.** Provider had no source for the per-pipeline pool offset.
   - **Fix.** `rlix/pipeline/miles_pipeline.py:833-884` — read `self._pipeline_config.cluster_device_mappings` (dict of `{actor_train: [...], actor_infer: [...]}`) when present; pass `train_mapping` / `infer_mapping` from there to `MilesPlacementProvider`. Falls back to `range(...)` when the dict is empty. Commit: rlix `d97178e`.
   - **Verification.** M11.2 attempt 4 log: `_build_placement_provider train=[2] infer=[2, 3] (from pipeline_config.cluster_device_mappings: True)`.

---

### F9 — Progress reporting

**Source plan**: §Feature 9 (`miles-port-unified-plan.md:2911`).

**Goal.** Per-rollout aggregated reporter to the scheduler so the gap-ratio planner has live cadence; group-unit + 2% bucket gate (per source plan §Feature 9 §3); `RLixHooks` protocol + `NoOpRLixHooks` stub for standalone path.

**Where it lives now.**
- `rlix/pipeline/miles_coordinator.py:170-256` — `MilesCoordinator.report_progress_from_scheduler` + `_aggregate_and_emit`.
- `rlix/pipeline/miles_hooks.py` — `RLixHooks` protocol + `NoOpRLixHooks`.
- `miles/miles/utils/tracking_utils.py` and `miles/miles/utils/wandb_utils.py` — tracking/wandb forwarding (NoOp when no rlix coordinator handle).

**M11 issues found.** None on the M11.1/M11.2 happy path. The reporter aggregation is exercised end-to-end every rollout in both smoke runs; no review finding flagged it.

---

### F10 — Partial GPU topology validation (startup fail-fasts C1–C23)

**Source plan**: §Feature 10 (`miles-port-unified-plan.md:3118`); see also AC7, C1–C23.

**Goal.** Single startup function asserts every topology / config precondition (partial-overlap subset, infer engine count ≥ 2, cpu_serialize transport, no MoE/EP, fullasync generation, Megatron parallelism divisibility, etc.). Misconfiguration must fail-fast with a clear message rather than silently OOM mid-init.

**Where it lives now.**
- `miles/miles/utils/rlix_validation.py:155-381` — `assert_rlix_topology(args, sglang_config)` (the F10 entry).
- `miles/miles/utils/rlix_validation.py:42-49` — `is_rlix_mode()` predicate (env var `RLIX_CONTROL_PLANE`).
- `miles/miles/utils/rlix_validation.py:65-87` — `train_devices_subset_of_infer` (used by F11 standalone guard).
- `miles/miles/utils/rlix_validation.py:88-114` — `async_generation_enabled`, `single_updateable_model_and_server`.
- `miles/miles/utils/rlix_validation.py:155-381` — `assert_rlix_topology` body (raises `ValueError` with per-constraint message; logs `F10 startup validation passed` on success).
- `miles/examples/rlix/run_miles_rlix.py:78-96` — driver call site.

**M11 issues found.** None. F10 fired correctly on every smoke run; M11.1 attempt 10 log confirms `F10 startup validation passed (engines=4, per_engine=1, train=2, infer=4, overlap=2)`.

---

### F11 — Conditional RLix behavior flag (`RLIX_CONTROL_PLANE=rlix`)

**Source plan**: §Feature 11 (`miles-port-unified-plan.md:3356`).

**Goal.** A single env var (`RLIX_CONTROL_PLANE=rlix`) gates all rlix-mode-specific behavior. Standalone `train_async.py` continues unchanged when the var is unset (AC8). Mirrors ROLL's `DO_TIME_SHARING` flag.

**Where it lives now.**
- `miles/miles/utils/rlix_validation.py:42-49` — `is_rlix_mode()` predicate.
- `miles/miles/utils/rlix_validation.py:383-end` — `assert_partial_overlap_standalone_safe` (F11 standalone fail-fast).
- All rlix-mode-specific branches gate on `os.environ.get("RLIX_CONTROL_PLANE") == "rlix"`. The active call sites in M11:
  - `miles/miles/ray/rollout.py:931` (F2 pause_generation in shrink)
  - `miles/miles/ray/rollout.py:1627` (F1 enable_memory_saver override)
  - `miles/miles/router/middleware_hub/radix_tree_middleware.py:69` (F3 disable RadixTreeMiddleware)
  - `miles/miles/rollout/generate_hub/multi_turn.py:45` (F3 turn-level redispatch path)
  - `miles/examples/rlix/run_miles_rlix.py:22-28` — F8 driver entry asserts `RLIX_CONTROL_PLANE=rlix` before any heavy import.

**M11 issues found.** None unique to M11. The flag was load-bearing through every smoke run and never produced a false positive / negative.

---

### F12 — Shared PG cluster (`MilesPlacementProvider`, `LOCAL_RANK=0`, SGLang `base_gpu_id=0`)

**Source plan**: §Feature 12 (`miles-port-unified-plan.md:3464`); see also F33 (SGLang `base_gpu_id=0`), F34 (`LOCAL_RANK=0` injection).

**Goal.** Replace MILES standalone's `_create_placement_group(rollout_num_gpus)` with an rlix-driven adapter (`MilesPlacementProvider`) that returns `WorkerPlacement` objects pinned to scheduler-allocated physical GPUs. Train side uses `RayTrainGroup(num_gpus_per_actor=0.01)` with manual `CUDA_VISIBLE_DEVICES`; SGLang gets `base_gpu_id=0` (post-CVD local view); `LOCAL_RANK=0` is explicitly injected.

**Where it lives now.**
- `miles/miles/ray/placement_provider.py:41-71` — `WorkerPlacement` dataclass.
- `miles/miles/ray/placement_provider.py:72-322` — `MilesPlacementProvider` class.
- `rlix/pipeline/miles_pipeline.py:833-884` — `_build_placement_provider` (Phase A step 1). Note: F95.1 ("infer-pool PG via `get_all_rollout_engine_placements()`") is currently a fail-fast assert (`len(set(infer_allocated)) >= rollout_num_gpus`); the proper end-to-end translation back into `(pg, reordered_bundle_indices, reordered_gpu_ids)` is deferred to a follow-up.
- `miles/miles/ray/actor_group.py` — `RayTrainGroup` constructor with `worker_placements=` keyword and `num_gpus_per_actor=0.01`.

**M11 issues found.** Issue 4 of F8 above (M11.2 `cluster_device_mappings` threading) is the F12 issue — `_build_placement_provider` was using `range(actor_count)` instead of the scheduler-allocated mapping. Fixed at `rlix/pipeline/miles_pipeline.py:833-884` per commit `d97178e`.

---

## §4 M11.1 + M11.2 fix index (back-pointer to F1–F12)

15 distinct fixes landed across M11.1 and M11.2; this index points each one back to its F1–F12 owner so a reviewer reading the source plan can locate every change.

| Fix | M11 attempt where surfaced | F-owner | File:line | Commit |
|---|---|---|---|---|
| 1. `enable_memory_saver=True` rlix-mode override | M11.1 attempt 5 | F1 | `miles/miles/ray/rollout.py:1627-1628` | miles `f58b365` |
| 2. `MILES_TMS_HOOK_MODE=torch` env var (CUDA 12.9 segfault) | M11.1 attempt 1 | F1 | `scripts/run_smoke_e2e.sh` (env var); supported at `miles/backends/megatron_utils/actor.py:96-111` | miles `f58b365` (script) |
| 3. `pause_generation` before `release_memory_occupation` in shrink | M11.1 attempts 5–7 | F2 | `miles/miles/ray/rollout.py:911-942` (call at 933) | miles `f58b365` |
| 4. `_wait_for_overlap_engines_offloaded` engine-index conversion | M11.2 attempt 4 | F2 | `rlix/pipeline/miles_pipeline.py:418-505` (conv at 431-450) | rlix `549cfbd` |
| 5. nvidia-smi probe for OS-level GPU mem free (Phase 2 wait) | M11.1 attempt 5 | F2 | `rlix/pipeline/miles_pipeline.py:418-505`, `_probe_min_free_gpu_mem_gb` at 527-564 | rlix `e0a6b27` |
| 6. `pause_generation` around `finalize_weight_update` (sync path) | M11.1 attempt 7 | F5+F6 | `rlix/pipeline/miles_model_update_service.py:288-318` | rlix `e0a6b27` |
| 7. Drop redundant `train_group.onload()` in `_before_training` | M11.1 attempt 8 | F4 | `rlix/pipeline/miles_pipeline.py:566-596` | rlix `e0a6b27` |
| 8. `with_ref` from KL flags (KL loss derivation) | M11.1 attempt 9 | F4 | `rlix/pipeline/miles_pipeline.py:163-180` | rlix `e0a6b27` |
| 9. `_expand_workers` no-op for already-active engines | M11.1 attempt 4 | F5+F6 | `rlix/pipeline/miles_coordinator.py:439-513` (branch ~470) | rlix `2b73aef` |
| 10. Coordinator name + namespace match scheduler RPC lookup | M11.1 attempt 4 | F7 | `miles/examples/rlix/run_miles_rlix.py:191-208`; `run_miles_dual.py:158-170` | miles `7b83be5`, `992fa26` |
| 11. `step_target_estimate` + `global_step` forwarding (Phase B INIT→GEN + per-step) | M11.1 attempt 4 | F8 | `rlix/pipeline/miles_pipeline.py:389-411`, `:574` | rlix `2b73aef` + `e0a6b27` |
| 12. `MILES_ROLLOUT_BASE_PORT` env var (per-pipeline offset) | M11.2 attempt 1 | F8 | `miles/miles/ray/rollout.py:220-235`; dual driver at `run_miles_dual.py:99-200` | miles `992fa26` |
| 13. `ulimit -n 65536` for 2-pipeline raylet | M11.2 attempt 1 | F8 | `scripts/run_smoke_dual.sh` (workspace-only) | n/a |
| 14. Dual driver — disjoint pools, deep-copy args, asyncio.gather | M11.2 attempts 1–4 | F8 | `miles/examples/rlix/run_miles_dual.py:50-377` | miles `992fa26`, `6126e01` |
| 15. `cluster_device_mappings` threaded into placement provider | M11.2 attempt 4 | F8 + F12 | `rlix/pipeline/miles_pipeline.py:833-884` | rlix `d97178e` |

---

## §5 Code paths (function-by-function reference)

For a reviewer already inside a file, here is the symbol → line-range index for the 6 modified files.

### `rlix/pipeline/miles_pipeline.py`

| Lines | Symbol | F-owner |
|---|---|---|
| 58 | `class MilesPipeline` | — |
| 61-92 | `__init__` | F8 |
| 94-117 | `_validate_topology` | F10 |
| 118-131 | `initialize_pipeline` | F8 |
| 133-241 | `_init_phase_a_train` (incl. `with_ref` derivation 163-180) | F4, F8 |
| 242-411 | `_init_phase_b_infer` (incl. INIT→GEN re-request at 389-411) | F8 |
| 418-505 | `_wait_for_overlap_engines_offloaded` (Phase 1 + Phase 2 with engine-index conv) | F2 |
| 527-564 | `_probe_min_free_gpu_mem_gb` | F2 |
| 566-596 | `_before_training` | F4 |
| 598-622 | `_after_training` | F5+F6 |
| 624-687 | `shutdown_hard` | F8 |
| 717-721 | `before_training` / `after_training` Ray method aliases | rlix_train_loop |
| 727-799 | `_request_cluster_gpus` / `_notify_release_cluster_gpus` | F8 |
| 833-884 | `_build_placement_provider` | F12 |

### `rlix/pipeline/miles_coordinator.py`

| Lines | Symbol | F-owner |
|---|---|---|
| 55-66 | `_build_pipeline_env_vars` | F7 |
| 68 | `class MilesCoordinator(Coordinator)` | F7 |
| 77-159 | `__init__` (manual init; NO `super().__init__()` per F09 forbidden) | F7 |
| 170-256 | `report_progress_from_scheduler` + `_aggregate_and_emit` | F9 |
| 258-298 | `_inject_pipeline_env_vars`, `bootstrap_active_engines` | F8 |
| 306-340 | `register_model_update_resources`, `publish_cache_ready_step` | F4 |
| 367-401 | `sync_base_weights_to_active` | F5+F6 |
| 402-438 | `resize_infer`, `_shrink_workers` | F2, F5+F6 |
| 439-513 | `_expand_workers` (incl. already-active no-op branch) | F5+F6 |
| 514-545 | `remove_resource_manager_node_pg` | F12 |
| 546-end | `create_pipeline_actor` | F8 |

### `rlix/pipeline/miles_model_update_service.py`

| Lines | Symbol | F-owner |
|---|---|---|
| 32-77 | `class SyncSessionPlan` (frozen dataclass) | F5+F6 |
| 82 | `class MilesModelUpdateService` | F5+F6 |
| 101-127 | `__init__`, `get_pipeline_id` | F5+F6 |
| 128-225 | `sync_selected_workers` (atomic-unit entry; cancel inflight on timeout) | F5+F6 |
| 248-338 | `_run_atomic_unit` | F5+F6 |
| 288-318 | pause/finalize/continue wrap | F5+F6 |
| 339-451 | `_build_plan` | F5+F6 |

### `miles/miles/ray/rollout.py`

| Lines | Symbol | F-owner |
|---|---|---|
| 84-100 | `class EngineInfo` (5-state dataclass) | F2 |
| 115 | `class ServerGroup` | F2 |
| 146-253 | `start_engines` (reads `MILES_ROLLOUT_BASE_PORT`) | F8 |
| 220-235 | `MILES_ROLLOUT_BASE_PORT` block | F8 |
| 274 | `class RolloutServer` | F2 |
| 396 | `class RolloutManager` | F2 |
| 638-742 | `offload`, `onload`, `onload_weights`, `onload_kv` (selective) | F2 |
| 774-788 | `get_engine_states` | F2 |
| 872-942 | `shrink_engines` (incl. pause_generation 911-942) | F2 |
| 968-1008 | `expand_engines` | F2 |
| 1009-1035 | `finish_init_offload` (M11.2 init-then-drop path) | F2 |
| 1036-end | `activate_routing` | F2 |
| 1605-1622 | `start_rollout_servers` rlix-mode override | F1 |
| 1627-1628 | `enable_memory_saver=True` force-on | F1 |

### `miles/examples/rlix/run_miles_rlix.py` (single-pipeline driver)

| Lines | Block | F-owner |
|---|---|---|
| 22-28 | F11 / F08 — assert `RLIX_CONTROL_PLANE=rlix` before heavy imports | F11 |
| 31-46 | `_build_cluster_device_mappings` | F8 |
| 49-262 | `main()` — orchestrator init, allocate_pipeline_id, register, admit, coordinator spawn (191-208), `create_pipeline_actor`, train loop | F8 |
| 191-208 | Coordinator `.options(name=COORDINATOR_ACTOR_NAME_PREFIX..., namespace=pipeline_namespace)` | F7 |

### `miles/examples/rlix/run_miles_dual.py` (dual-pipeline driver, M11.2)

| Lines | Block | F-owner |
|---|---|---|
| 50-66 | `_split_pools_for_dual` | F8 |
| 75-97 | `_per_pipeline_args` (deep-copy + per-pipeline exp_name + router-port reset) | F8 |
| 99-200 | `_build_pipeline` (per-pipeline coordinator + pipeline + initialize) | F7, F8 |
| 158-170 | Coordinator namespace per pipeline | F7 |
| 220-377 | `main` (sequential init of P1+P2; concurrent train via `asyncio.gather`) | F8 |

---

## §6 Known limitations / deferred work

| ID | What | Why deferred | Impact |
|---|---|---|---|
| **F22 (relaxed)** | Strict `shell → offloaded → active` engine-init state machine | Receiver-side shell-creation never wired into `RolloutManager.expand_engines`. M11.1 uses a pragmatic "engines come up active" hatch in `_expand_workers`. | Multi-pipeline (3+) needs F22 so the scheduler can manage engines that come up shell. M11.2 disjoint-pool topology works without it. |
| **M11.3** | 3+ concurrent pipelines on small GPU pool with cross-pipeline contention | Topologically requires F22 + 1-GPU train shape. Not exercised by current smoke. | Three concurrent pipelines on 4 GPUs not validated. |
| **F95.1** | Infer-pool PG via `MilesPlacementProvider.get_all_rollout_engine_placements()` (replace standalone `_create_placement_group` bypass) | Defer to dual-pipeline contention smoke; current code has a fail-fast assert that blocks silent oversubscription. | M11.2 disjoint-pool case unaffected (each pipeline's bypass owns its own PG). |
| **F19/F20 (saving)** | Final-rollout save trigger | Megatron `save_model` does not onload weights internally; `_after_training` already offloaded. | Smoke runs pass `--save ""`. Real fix: explicit onload→save→offload bracket. |
| **Eval at final rollout** | `should_run_periodic_action` fires at the last rollout regardless of `--eval-interval` | Same offloaded-weight problem as save. | Smoke skips eval. |
| **Pytest coverage of M11 changes** | 6 critical files have no direct unit/integration tests | Time pressure; smoke run is the main verification. | Future iteration should add unit tests for `_wait_for_overlap_engines_offloaded`, pause_generation wrapping, `_split_pools_for_dual`, etc. |
| **R04-F1 (HIGH from review-report)** | `rlix_train_loop.run_async_train_loop` lacks `try/finally` around `before_step / train / after_step` | Static-analysis finding; happy-path smoke does not exercise failure path. | Train() raise leaks rlix scheduler GPU allocation. Production hardening tracked into M11.x. |
| **Vast-only `_with_region_config` patch** | Nested-region clobber in tms 0.0.9 entrypoint.py | Upstream tms bug. | Hot-patched on the vast image only; not in repo. Tracked in `docs/tms-fixes.md` §5. |

Production hardening items (deferred to M11.5 per scope plan): `F79`–`F91` — bounded `_use_url` timeout, `admission_epoch` race defense, multi-pipeline cleanup daemon, `master_port` cooldown queue, dead-recovery, etc.

---

## §7 How to reproduce

The three scripts under `scripts/` are the verification rig. **A reviewer cloning the repo can re-run M11.1 single + M11.2 dual end-to-end and confirm `EXIT_CODE=0` themselves.** No hidden setup.

### 7.1 Prerequisites

A vast.ai (or equivalent) GPU instance with:

| Component | Tested with | Notes |
|---|---|---|
| GPUs | 4× (RTX5090 for M11.1; A40 for M11.2) | Any 4× of ≥40 GB memory will do for Qwen2.5-0.5B GRPO. |
| CUDA | 12.9 | Smoke is hardcoded to `MILES_TMS_HOOK_MODE=torch` (CUDAPluggableAllocator) which is required for CUDA 12.9 / Blackwell — F1 issue 2. |
| Image | `radixark/miles_latest` (vast.ai) or any image with miles + rlix + Megatron-LM checked out under `/root/{miles,rlix,Megatron-LM}` | The smoke `cd /root/miles` and uses absolute paths under `/root`. |
| Model checkpoint | Qwen2.5-0.5B at `/root/Qwen2.5-0.5B`, torch_dist at `/root/Qwen2.5-0.5B_torch_dist`, miles bundle at `/root/Qwen2.5-0.5B_miles/` | Any HF Megatron-compatible Qwen2.5-0.5B mirror works. |
| Datasets | `/root/dapo-math-17k/dapo-math-17k.jsonl`, `/root/aime-2024/aime-2024.jsonl` | Eval is gated and skipped in smoke; the file just needs to exist. |
| Branches | rlix on `zhenyu/miles-mvp-e2e` (or main once merged), miles on `zhenyu/m11-mvp-test` | See Appendix C for HEADs. |

### 7.2 Smoke scripts

Three scripts ship in `scripts/`:

| Script | What it does | Smoke target | Expected runtime on 4×RTX5090 |
|---|---|---|---|
| `scripts/run_smoke_e2e.sh` | M11.1 single-pipeline. Qwen2.5-0.5B GRPO, partial-overlap topology (train=`[0,1]` ⊂ infer=`[0,1,2,3]`), `--num-rollout 2`. | AC1, AC2, AC3, AC4, AC5 | ~5–7 min (init + 2 rollouts) |
| `scripts/run_smoke_dual.sh` | M11.2 dual-pipeline. Two MilesPipelines on disjoint pools (P1=`[0,1]`, P2=`[2,3]`), each running its own GRPO loop concurrently via `asyncio.gather`, `--num-rollout 2` each. | AC9, AC10, AC11, AC12 | ~10 min total (init phase is sequential P1→P2 then both train concurrently) |
| `scripts/run_smoke_with_watchdog.sh` | Wrapper. Launches either of the above, monitors `/root/logs/run.log` for silence (default 5 min) and total runtime (default 40 min); kills the run on either limit. Always emits `EXIT_CODE=<n>` as the last log line for grep-friendly success detection. | — | n/a |

The scripts are **rsync'd to `/root/rlix/scripts/`** on vast and executed there. They use `/root/miles/scripts/models/qwen2.5-0.5B.sh` (a miles-side helper) to source `MODEL_ARGS`.

### 7.3 Run M11.1 single-pipeline

On the vast instance:

```bash
# Use the watchdog wrapper so a hung smoke is killed automatically:
SCRIPT=/root/rlix/scripts/run_smoke_e2e.sh \
SILENCE_LIMIT=300 RUN_LIMIT=2400 \
bash /root/rlix/scripts/run_smoke_with_watchdog.sh

# Or run the smoke directly without the watchdog:
bash /root/rlix/scripts/run_smoke_e2e.sh 2>&1 | tee /root/logs/run.log
```

**Success criteria** (all must hold):

- Last line: `EXIT_CODE=0`
- `[run_miles_rlix] training loop complete`
- `[run_miles_rlix] shutdown_hard complete`
- For both `rollout_id=0` and `rollout_id=1`: lines `step1: await rollout_data done`, `step2: before_step done`, `step3: train_group.train done`, `step4: after_step done` are all present (run_async_train_loop bracketing — F4)
- `_wait_for_overlap_engines_offloaded: OS-level GPU mem free min=*` shows ≥20 GB free (F2 issue 1 + F1 issue 1 working)
- No `Traceback`, no `KeyError`, no `Pointer argument cannot be accessed from Triton`, no `Cannot resume allocation that is not paused`, no `Failed to resolve actor`

### 7.4 Run M11.2 dual-pipeline

```bash
SCRIPT=/root/rlix/scripts/run_smoke_dual.sh \
SILENCE_LIMIT=900 RUN_LIMIT=3600 \
bash /root/rlix/scripts/run_smoke_with_watchdog.sh
```

**Success criteria**:

- Last line: `EXIT_CODE=0`
- `[run_miles_dual] mp1 training loop complete pipeline_id=miles_<uuid>`
- `[run_miles_dual] mp2 training loop complete pipeline_id=miles_<uuid>`
- `[run_miles_dual] shutdown_hard complete pipeline_id=miles_<uuid>` × 2
- **Zero** `KeyError('unknown engine_index ...')` warnings (F2 issue 2 fix at HEAD `5dc4e43+` is non-regressing)
- SGLang ServerArgs log shows P1 engines on ports 15000–15007 and P2 engines on ports 16000–16007 (F8 issue 3 — `MILES_ROLLOUT_BASE_PORT` per-pipeline offset working)
- `_build_placement_provider train=[2] infer=[2, 3] (from pipeline_config.cluster_device_mappings: True)` for the second pipeline (F8 issue 4 — `cluster_device_mappings` threading working)

### 7.5 What to do if the smoke fails

1. Check `/root/logs/run.log` for the failure mode (Traceback / KeyError / hang).
2. Compare against the failure modes catalogued in §3 features F1–F12 — every smoke failure observed during M11.1 / M11.2 development is documented under "M11 issues found" with attempt number, root cause, and fix commit.
3. If the failure is **new** (not in §3), check `plans/m11-e2e-test-log.md` (M11.1) or `plans/m11-2-dual-pipeline-log.md` (M11.2) for any later append-only entries that might document a regression.
4. Network/cluster-state issues (`Failed to register worker to Raylet: IOError`) are typically vast image / Ray version issues — try `ray stop --force; sleep 3; ray start --head` manually before re-running.

### 7.6 Customizing for non-vast

The scripts hardcode `/root/{miles,rlix,Megatron-LM}` paths. To run elsewhere:

- Edit lines `cd /root/miles` and `export PYTHONPATH=...` to match your layout.
- Edit `--hf-checkpoint` / `--ref-load` / `--load` / `--prompt-data` to your model + dataset paths.
- The watchdog wrapper writes `/root/logs/run.log` — override with `LOG=/path/to/your.log`.
- Everything else (port ranges, GRPO hyperparameters, `--num-rollout 2`) should match the smoke as a baseline; deviations make the success criteria above unreliable.

---

## Appendix A — Verification log map

The two append-only iteration logs are the source of truth for what was exercised end-to-end on real GPUs.

- `plans/m11-e2e-test-log.md` — M11.1 attempts 0–10 (10 fix-retest cycles, ~3.5 hr vast time). Final attempt 10 = `EXIT_CODE=0` with both rollouts trained on 4xRTX5090.
- `plans/m11-2-dual-pipeline-log.md` — M11.2 attempts 0–4 (4 fix-retest cycles, ~33 min vast time). Final attempt 4 = `EXIT_CODE=0` with both pipelines reaching `shutdown_hard` on 4xA40.
- `plans/m11-review.debug.md` — post-review verification: M11.2 dual smoke RERUN at HEAD `5dc4e43` / `6126e01` on 2026-05-08 (Run 1) confirms Feature 11/F2 engine-index fix is non-regressing (`EXIT_CODE=0`, 0 `KeyError`, 0 `engine_index` warnings).

## Appendix B — Code review artifacts

- `plans/m11-review.plan.md` — review plan (scope, decision trace D0–D5, AC #1–#6).
- `plans/m11-review.review.md` — code-review brief (6 parallel review shards R01–R06, 35+ adversarial prompts, executable test-command annex).
- `plans/m11-review.review-report.md` — aggregated findings; final recommendation `NEEDS_FIX` driven by R04-F1 HIGH (see §5 limitations).
- `plans/m11-review.review-report/R01..R06.md` — per-shard sidecar reports.

## Appendix C — Branches at time of writing

| Branch | HEAD | Contents |
|---|---|---|
| rlix `zhenyu/miles-mvp-e2e` | `03cbeb7` | F1–F12 rlix-side; `docs/m11-implementation-guide.md`; review artifacts |
| miles `zhenyu/m11-mvp-test` | `6126e01` | F1, F2, F8, F11 miles-side rlix-mode wiring |

Both branches are pushed to GitHub:
- https://github.com/rlops/rlix
- https://github.com/radixark/miles
