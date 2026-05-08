# Code Review Plan

Feature: M11 (single + dual pipeline rlix-mode E2E for miles)
Source plan: plans/m11-review.plan.md
TLDR artifact: missing, optional
Scope artifact: missing, optional
Debug log: missing, optional (equivalents at plans/m11-e2e-test-log.md + plans/m11-2-dual-pipeline-log.md)
Code diff / changed files: present (cited in source plan; rlix HEAD `5dc4e43`, miles HEAD `6126e01`)
Review readiness: READY

## 0. Review Dashboard

- Review readiness: READY
- Blocking issues: none
- Warnings:
  - `.tldr.md` missing (optional but recommended) — TLDR projection of the source plan would sharpen the audit. Reviewers should treat the source plan + `docs/m11-implementation-guide.md` as the audit-projection substitute.
  - `.scope.md` missing (optional but recommended) — boundary classification not formalized. Reviewers should treat the "Hard constraints" + "Out of scope" sections of `m11-review.plan.md` as the implicit scope rails.
  - `.debug.md` missing — recurrence-risk shards reconstruct bug history from `plans/m11-e2e-test-log.md` (M11.1 attempts 0–10) and `plans/m11-2-dual-pipeline-log.md` (M11.2 attempts 0–4). These are append-only execution logs, so functionally equivalent to a `debug-log` artifact.
  - No standalone unit tests for the 6 critical files — smoke run is the verification surface. Reviewers must not file `TEST_GAP` findings against this gap (it is acknowledged in the source plan as out-of-scope for this milestone). They MAY file `TEST_GAP` against scenarios within the smoke that aren't actually exercised.
  - SGLang `pause_generation` semantics are an external dependency invariant — reviewers should consult upstream SGLang docs before flagging `RACE` against the new wrapping in `RolloutManager.shrink_engines` and `MilesModelUpdateService.sync_selected_workers`.
- Tests-passing evidence: present — `m11-e2e-test-log.md` attempt 10 (`EXIT_CODE=0`, both rollouts trained on 4xRTX5090) + `m11-2-dual-pipeline-log.md` attempt 4 (`EXIT_CODE=0`, both pipelines reached shutdown_hard on 4xA40)
- Code diff / changed files: present (rlix `zhenyu/miles-mvp-e2e` HEAD `5dc4e43`; miles `zhenyu/m11-mvp-test` HEAD `6126e01`; per-feature line refs in source plan)
- Author clarification: 0 questions in §1.1 (0 blocking)

## 1. Author Clarification

### 1.1 Author Clarification Questions

| QID | Question | Options | Affects | Default if unanswered | Why it matters |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

Author clarification: none required.

### 1.2 Review Assumptions

| Assumption ID | Assumption | Used for | Risk if wrong |
|---|---|---|---|
| A1 | The source plan's "Hard constraints" list (F22 deferred, multi-pipeline 3+ out of scope, save/eval at final rollout broken-by-design) is authoritative; reviewers do NOT file findings against those items. | All shards | Reviewer effort wasted on gaps the human already triaged out |
| A2 | The smoke run on a real GPU instance (M11.1 4xRTX5090, M11.2 4xA40) is the canonical correctness signal for this milestone. Static-analysis findings need a smoke-test counterexample to escalate above MEDIUM. | R01, R02, R03, R04, R05 | Static finding without a real-GPU repro might be a false alarm |
| A3 | `pause_generation(mode="retract")` is the proper SGLang API for reaching a quiescent scheduler before `release_memory_occupation` / `flush_cache` (per upstream SGLang). Reviewers MAY question miles-side correctness of how it is wrapped, but NOT whether `pause_generation` itself does what we claim. | R01 | Misclassifying a SGLang-side quiescence bug as a miles bug |
| A4 | The rlix scheduler's resolution path for the per-pipeline coordinator uses `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` in `pipeline_<pipeline_id>_NS` (`scheduler.py:1213`). | R06 | Driver-side naming convention mismatch would produce silent "Failed to resolve actor" errors |

## 2. Review Input Index

| Artifact | Path | Role | Required? | Status | Freshness notes |
|---|---|---|---|---|---|
| Source plan | `plans/m11-review.plan.md` | Authoritative scope/decision/files-changed list | required | present-current | Authored alongside this review brief |
| Implementation guide | `docs/m11-implementation-guide.md` | 17-feature catalog (high → low level), reviewer-friendly walkthrough | required (audit projection substitute for missing `.tldr.md`) | present-current | v2.1 — Codex re-review found 0 CRITICAL / 0 HIGH after fixes |
| TMS deep-dive | `docs/tms-fixes.md` | 5-fix detailed record for Features 1–5 | required for R01, R02 | present-current | Authored during M11.1 |
| M11.1 test log | `plans/m11-e2e-test-log.md` | Append-only attempts 0–10 — functions as `.debug.md` substitute | required for R01, R02, R03 | present-current | Final attempt 10 = PASS |
| M11.2 test log | `plans/m11-2-dual-pipeline-log.md` | Append-only attempts 0–4 — functions as `.debug.md` substitute | required for R05, R06 | present-current | Final attempt 4 = PASS |
| rlix diff | `git log -p 5dc4e43..` on `zhenyu/miles-mvp-e2e` (4 commits since branch base) | Code under review (rlix side) | required | present-current | Branch pushed |
| miles diff | `git log -p 6126e01..` on `zhenyu/m11-mvp-test` (4 commits since branch base) | Code under review (miles side) | required | present-current | Branch pushed |
| Smoke success archives | `~/Downloads/m11-e2e-success-1778067894.tar.gz`, `~/Downloads/m11-2-dual-success-1778123778.tar.gz` | Tests-passing evidence | optional but referenced | present-current | Archived from vast |

## 3. Feature Review Matrix

| Review ID | Feature / area | Plan anchors | Scope categories | Debug bugs | Review angles | Priority | Suggested reviewer |
|---|---|---|---|---|---|---|---|
| R01 | SGLang quiescence — `enable_memory_saver`, `pause_generation` wrapping, hook mode | Plan §"Hard constraints" 3, D3, D4; AC #2, #3 | implicit P0-CORRECTNESS-FLOOR | M11.1 attempts 4–7 (Triton-CPU race, flush_cache hang) | CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, RESOURCE_LIFECYCLE_REVIEW, ADVERSARIAL_INVARIANT_REVIEW | high | sglang-aware reviewer |
| R02 | tms region lifecycle — wake_up correctness, nvidia-smi probe, engine-index conversion | D4, AC #1, #4 | implicit P0-CORRECTNESS-FLOOR | M11.1 attempt 5 (nvidia-smi probe miss), M11.2 attempt 4 (engine-index KeyError) | CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, RESOURCE_LIFECYCLE_REVIEW, ADVERSARIAL_INVARIANT_REVIEW | high | rlix internals reviewer |
| R03 | Phase A/B init bootstrap + scheduler hand-off | D0, AC #1, #5 | implicit P0-CORRECTNESS-FLOOR | M11.1 attempts 1–3 (init order, with_ref None, already-active), M11.1 attempt 9 (with_ref=False) | PLAN_ADHERENCE_REVIEW, CORRECTNESS_REVIEW, ERROR_HANDLING_FAIL_FAST_REVIEW, ADVERSARIAL_INVARIANT_REVIEW | high | rlix scheduler reviewer |
| R04 | Runtime hooks + train loop ordering (`_before_training`, `_after_training`, `rlix_train_loop`) | D5, AC #1 | implicit P1-HAPPY-PATH | M11.1 attempt 8 (double wake_up) | CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, ERROR_HANDLING_FAIL_FAST_REVIEW, MAINTAINABILITY_REVIEW | high | rlix runtime reviewer |
| R05 | Multi-pipeline driver — disjoint pools, port collision, ulimit, deep-copy args | D2, AC #4, #5 | implicit P1-HAPPY-PATH | M11.2 attempts 1–3 (ulimit SIGABRT, ROLL not installed, base port race) | CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, RESOURCE_LIFECYCLE_REVIEW, ADVERSARIAL_INVARIANT_REVIEW | high | dual-pipeline / Ray-actor reviewer |
| R06 | Coordinator naming + namespace + scheduler RPC lookup | D0, AC #5 | implicit P0-CORRECTNESS-FLOOR | M11.1 attempt 4 (Failed to resolve actor) | CORRECTNESS_REVIEW, ADVERSARIAL_INVARIANT_REVIEW, OBSERVABILITY_REVIEW | medium | rlix scheduler reviewer |

## 4. Parallel Review Shards

### R01 — SGLang quiescence (`pause_generation` wrapping + `enable_memory_saver` gating + hook mode)

- **Goal**: Prove that the `pause_generation` wrapping in `RolloutManager.shrink_engines` and `MilesModelUpdateService.sync_selected_workers` correctly quiesces SGLang's scheduler so subsequent `release_memory_occupation` / `flush_cache` calls succeed without crashing on Triton-CPU pointer errors or timing out for 60s. Verify the rlix-mode override that forces `enable_memory_saver=True` in SGLang `ServerArgs` is correct and exhaustive across the four miles-side gates.
- **Must-read anchors**: m11-review.plan.md "D3", "D4", "Hard constraints" 3
- **Scope categories / scope notes**: implicit P0-CORRECTNESS-FLOOR (no `.scope.md` artifact; reviewer derives from "Hard constraints" + "Out of scope" sections of source plan)
- **Code areas to inspect**:
  - `miles/miles/ray/rollout.py:911-942` (shrink_engines pause_generation wrap)
  - `miles/miles/ray/rollout.py:1616-1628` (rlix-mode override forcing `needs_offload=True`)
  - `rlix/pipeline/miles_model_update_service.py:288-318` (finalize_weight_update wrap)
  - `miles/miles/backends/sglang_utils/sglang_engine.py:1057` (enable_memory_saver gate on `args.offload_rollout`)
  - `miles/miles/backends/megatron_utils/actor.py:96-111` (MILES_TMS_HOOK_MODE)
- **Review angles**: CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, RESOURCE_LIFECYCLE_REVIEW, ADVERSARIAL_INVARIANT_REVIEW, ERROR_HANDLING_FAIL_FAST_REVIEW
- **Adversarial questions**: see §5 R01
- **Required tests / evidence**: M11.1 attempts 5–7 + 10 from `plans/m11-e2e-test-log.md`; smoke archive `m11-e2e-success-1778067894.tar.gz`
- **Known bugs from debug-log**:
  - M11.1 attempt 4: `Pointer argument cannot be accessed from Triton (cpu tensor?)` — fixed by adding `pause_generation` before `release_memory_occupation`
  - M11.1 attempt 5: `Timeout while flushing cache` — `pause_generation(mode="retract")` is the right API; `time.sleep(0.5)` workaround failed
  - M11.1 attempt 6: `Cannot resume allocation that is not paused` — separate from R01 (covered in R02/R04)
- **Expected reviewer output**: subagent report at `<plan-stem>.review-report/R01.md` per §7 template, focused on quiescence invariants
- **Stop conditions**: every adversarial question answered (yes/no/out-of-scope); evidence cited from M11.1 attempt logs; verdict declared

### R02 — tms region lifecycle (wake_up, nvidia-smi probe, engine-index conversion)

- **Goal**: Prove `_wait_for_overlap_engines_offloaded`'s 2-phase wait (state → nvidia-smi free) actually catches the OS-level memory release timing. Prove the Feature 11 engine-index conversion is correct for arbitrary infer-pool starts (not just M11.1 pool=[0..N]). Prove there are no double `tms.resume` paths remaining anywhere in the rlix runtime hooks.
- **Must-read anchors**: m11-review.plan.md "D4", AC #1, AC #4
- **Scope categories / scope notes**: implicit P0-CORRECTNESS-FLOOR
- **Code areas to inspect**:
  - `rlix/pipeline/miles_pipeline.py:418-505` (`_wait_for_overlap_engines_offloaded` Phase 1 + 2)
  - `rlix/pipeline/miles_pipeline.py:431-450` (engine-index conversion using `min(infer_mapping)` offset)
  - `rlix/pipeline/miles_pipeline.py:566-596` (`_before_training` body — confirm no `train_group.onload()` call remains)
  - `miles/miles/backends/megatron_utils/actor.py:357` (the train()-internal `wake_up` call that supersedes the removed onload)
- **Review angles**: CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, RESOURCE_LIFECYCLE_REVIEW, ADVERSARIAL_INVARIANT_REVIEW
- **Adversarial questions**: see §5 R02
- **Required tests / evidence**: M11.1 attempt 5 (nvidia-smi probe success); M11.2 attempt 4 (engine-index KeyError gone after fix); commit `549cfbd` engine-index fix.
- **Known bugs from debug-log**:
  - M11.1 attempt 5: state="offloaded" but free_GB=1.97 — fixed by adding nvidia-smi Phase 2
  - M11.1 attempt 8: double wake_up (`Cannot resume allocation that is not paused`) — fixed by removing redundant onload
  - M11.2 attempt 4 warning: `KeyError('unknown engine_index 2')` — fixed by `(g - infer_first) // per_engine`
- **Expected reviewer output**: subagent report at R02.md
- **Stop conditions**: invariant arithmetic (offset, divisibility, ordering) reviewed for arbitrary `infer_mapping` (not just contiguous from 0); evidence-backed verdict

### R03 — Phase A/B init bootstrap + scheduler hand-off

- **Goal**: Walk Phase A (steps 1–7) + Phase B (steps 1–8) sequence and prove (a) the order is correct for a fresh cluster, (b) every scheduler `_request_cluster_gpus` / release pair is balanced, (c) `with_ref` derivation matches standalone, (d) `step_target_estimate` derivation at Phase B step 8 produces a sane value, (e) the M11.1 "engines come up active" hatch in `_expand_workers` is safe given the F22 deferral.
- **Must-read anchors**: m11-review.plan.md "D0", AC #1, AC #5
- **Scope categories / scope notes**: implicit P0-CORRECTNESS-FLOOR
- **Code areas to inspect**:
  - `rlix/pipeline/miles_pipeline.py:133-241` (Phase A)
  - `rlix/pipeline/miles_pipeline.py:163-180` (`with_ref` from kl args)
  - `rlix/pipeline/miles_pipeline.py:242-411` (Phase B)
  - `rlix/pipeline/miles_pipeline.py:389-411` (`step_target_estimate` derivation)
  - `rlix/pipeline/miles_coordinator.py:~470` (`_expand_workers` already-active no-op)
  - `miles/miles/ray/placement_group.py:192` (standalone `with_ref` reference)
- **Review angles**: PLAN_ADHERENCE_REVIEW, CORRECTNESS_REVIEW, ERROR_HANDLING_FAIL_FAST_REVIEW, ADVERSARIAL_INVARIANT_REVIEW
- **Adversarial questions**: see §5 R03
- **Required tests / evidence**: M11.1 attempt 4 + 9 + 10
- **Known bugs from debug-log**:
  - M11.1 attempt 9: `torch.cat(NoneType, dim=int)` — `with_ref=False` was hardcoded; fixed by reading `args.use_kl_loss / kl_coef`
  - M11.1 attempt 10: scheduler INIT→GEN transition succeeded only after `step_target_estimate` was forwarded (commit `2b73aef`)
  - M11.1 attempt 7: `_expand_workers` strict state-machine crash on already-active engines — fixed with the no-op hatch (commit reference around `2b73aef` / `e0a6b27`)
- **Expected reviewer output**: subagent report at R03.md, with explicit yes/no on whether F10's M11.1 hatch is safe for M11.2 disjoint-pool dual-pipeline (not just single-pipeline)
- **Stop conditions**: every adversarial question answered; init-step ordering audit complete; verdict declared

### R04 — Runtime hooks + train loop ordering

- **Goal**: Prove `_before_training` / `_after_training` correctly bracket each rollout iteration: GPU reclaim → wake → train → cache build → offload → weight sync → release. Confirm the rlix-mode train loop helper `rlix_train_loop.run_async_train_loop` doesn't leak state across iterations or break on rollout failures.
- **Must-read anchors**: m11-review.plan.md "D5", AC #1
- **Scope categories / scope notes**: implicit P1-HAPPY-PATH
- **Code areas to inspect**:
  - `rlix/pipeline/miles_pipeline.py:566-596` (`_before_training`)
  - `rlix/pipeline/miles_pipeline.py:598-660` (`_after_training`)
  - `rlix/pipeline/miles_pipeline.py:574` (`global_step=int(step)` forwarding)
  - `rlix/pipeline/miles_pipeline.py:717-721` (public Ray methods `before_training`, `after_training`)
  - `miles/miles/utils/rlix_train_loop.py` (full file — async loop helper)
- **Review angles**: CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, ERROR_HANDLING_FAIL_FAST_REVIEW, MAINTAINABILITY_REVIEW
- **Adversarial questions**: see §5 R04
- **Required tests / evidence**: M11.1 attempts 8 + 10 + M11.2 attempt 4
- **Known bugs from debug-log**:
  - M11.1 attempt 8: double wake_up — fixed by removing the redundant onload (also a R02 concern; reviewer should treat as cross-shard)
- **Expected reviewer output**: subagent report at R04.md
- **Stop conditions**: rollout-failure path traced (what happens if train() raises mid-iteration?); verdict declared

### R05 — Multi-pipeline driver (`run_miles_dual.py`)

- **Goal**: Prove the dual driver correctly isolates per-pipeline state: deep-copied args, distinct exp_names, distinct `MILES_ROLLOUT_BASE_PORT`, distinct `cluster_device_mappings` registered with the orchestrator, distinct Ray namespaces. Prove `_split_pools_for_dual` handles edge cases (odd GPU counts, single-GPU machines). Prove the `asyncio.gather` train loops don't have cross-pipeline awaiting / accidental shared state.
- **Must-read anchors**: m11-review.plan.md "D2", AC #4, AC #5
- **Scope categories / scope notes**: implicit P1-HAPPY-PATH
- **Code areas to inspect**:
  - `miles/examples/rlix/run_miles_dual.py:50-66` (`_split_pools_for_dual`)
  - `miles/examples/rlix/run_miles_dual.py:75-97` (`_per_pipeline_args`)
  - `miles/examples/rlix/run_miles_dual.py:99-200` (`_build_pipeline`)
  - `miles/examples/rlix/run_miles_dual.py:220-377` (`main` — sequential init + asyncio.gather)
  - `miles/miles/ray/rollout.py:220-235` (`MILES_ROLLOUT_BASE_PORT` env var read)
  - `rlix/pipeline/miles_pipeline.py:833-884` (`_build_placement_provider` reading `cluster_device_mappings`)
- **Review angles**: CORRECTNESS_REVIEW, CONCURRENCY_RACE_REVIEW, RESOURCE_LIFECYCLE_REVIEW, ADVERSARIAL_INVARIANT_REVIEW
- **Adversarial questions**: see §5 R05
- **Required tests / evidence**: M11.2 attempts 1–4
- **Known bugs from debug-log**:
  - M11.2 attempt 1: `_split_devices` (old name) not on remote — config drift between local and pushed; recurrence-risk = `_split_pools_for_dual` rename should hold
  - M11.2 attempt 2: raylet SIGABRT errno=24 — fixed by `ulimit -n 65536` in smoke script
  - M11.2 attempt 3: `ModuleNotFoundError: roll` — env setup gap; recurrence-risk = ROLL must be installed
  - M11.2 attempt 4 warning: KeyError engine_index 2 (also a R02 concern; cross-shard)
- **Expected reviewer output**: subagent report at R05.md, including `_split_pools_for_dual` edge-case analysis
- **Stop conditions**: cross-pipeline state isolation traced for: (a) wandb tracking, (b) ports, (c) actor names, (d) namespaces, (e) shared file globals; verdict declared

### R06 — Coordinator naming + namespace + scheduler RPC lookup

- **Goal**: Prove the coordinator naming + namespace convention used by the driver (`f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` in `pipeline_<pipeline_id>_NS`) matches the rlix scheduler's lookup path. Prove this convention is consistent across the single-pipeline driver and the dual-pipeline driver. Prove there are no other Ray named actors in the per-pipeline namespace that could collide.
- **Must-read anchors**: m11-review.plan.md "D0", AC #5
- **Scope categories / scope notes**: implicit P0-CORRECTNESS-FLOOR
- **Code areas to inspect**:
  - `miles/examples/rlix/run_miles_rlix.py:191-208` (single-pipeline naming)
  - `miles/examples/rlix/run_miles_dual.py:158-170` (dual-pipeline naming, per-pipeline)
  - `rlix/protocol/types.py` — `COORDINATOR_ACTOR_NAME_PREFIX`, `RLIX_NAMESPACE`, `get_pipeline_namespace` (verify these are stable contract symbols and not test-time constants)
  - `rlix/scheduler/scheduler.py` (around line 1213 — `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` resolution)
- **Review angles**: CORRECTNESS_REVIEW, ADVERSARIAL_INVARIANT_REVIEW, OBSERVABILITY_REVIEW
- **Adversarial questions**: see §5 R06
- **Required tests / evidence**: M11.1 attempt 4 (`Failed to resolve actor` → fix); M11.2 attempts 1+ (per-pipeline namespace not colliding)
- **Known bugs from debug-log**:
  - M11.1 attempt 4: `Failed to resolve actor` due to wrong name + wrong namespace — fixed in commit `7b83be5`
- **Expected reviewer output**: subagent report at R06.md
- **Stop conditions**: every per-pipeline named actor enumerated (coordinator, model_update_service, rollout_manager); collision possibility evaluated; verdict declared

## 5. Adversarial Review Prompts

### R01 — SGLang quiescence

1. If `pause_generation` returns 200 OK but the SGLang scheduler thread had ALREADY launched a Triton kernel before the pause request was processed, what state does the engine end in by the time `release_memory_occupation` runs?
2. The shrink-path wraps `pause_generation` in a try/except (line 933, with broad `except Exception` and a log warning). What if SGLang returns 4xx for "already paused"? Does the subsequent `release_memory_occupation` still succeed, or does it now race against an already-running scheduler thread that the broad except just hid?
3. The sync_selected_workers wrap (`miles_model_update_service.py:288-318`) calls `continue_generation` AFTER `finalize_weight_update`. What happens if `finalize_weight_update` itself raises? Does the engine stay paused indefinitely (resource leak)? Walk the exception path.
4. Does the rlix-mode override at `miles/miles/ray/rollout.py:1627-1628` cover ALL miles internal callers of `start_rollout_servers` that compute `needs_offload`? Or are there miles standalone code paths that still hit the M11.1 single-pipeline disjoint-pool path with `needs_offload=False`?
5. `MILES_TMS_HOOK_MODE=torch` is set in the smoke script. What if it's NOT set in production? Does `actor.py:96-111` fall back safely, or does Blackwell+CUDA12.9 segfault on first build_cpu_bucket_cache?
6. SGLang's `pause_generation(mode="retract")` retracts in-flight decode iterations. Are retracted samples LOST (not seen by the train_actor as rollout data)? Or does the rollout function catch and re-issue them on `continue_generation`? Trace the contract.
7. Multiple pipelines call `pause_generation` concurrently on disjoint engines. SGLang's HTTP server is per-engine, so they should be independent — but what about Ray's `flush_cache` HTTP client connection pool? Could one pipeline's flush blocked by 60s starve another's?

### R02 — tms region lifecycle

1. `_wait_for_overlap_engines_offloaded` Phase 2 calls `nvidia-smi --query-gpu=memory.free --id={target_gpu_ids}` every 0.5s. What if `nvidia-smi` itself is slow or hangs (driver issue)? Does the function spin or fall back?
2. The Phase 2 threshold is `min_free_gb >= 20.0`. What if the train footprint is larger than 20 GB (e.g., a 7B model)? Should the threshold be derived from the train weights' size, not hardcoded?
3. Engine-index conversion (`(g - infer_first) // per_engine`) assumes `infer_mapping` is contiguous. What if a future user passes a non-contiguous mapping like `[0, 2, 4, 6]`? Does the floor-division still produce a valid local index?
4. `_before_training` no longer calls `train_group.onload()`. What if the user has `args.offload_train=False`? Does `MegatronTrainRayActor.train()` still self-wake? Or does the train start on still-resident weights (no-op wake)?
5. Phase 1 of `_wait_for_overlap_engines_offloaded` polls `get_engine_states` with a 60s deadline. What if the rlix scheduler's resize_infer hasn't started yet when this poll begins (race)? Does the function correctly wait for the state transition, or does it return early on a stale "active" state?
6. Phase 2 fall-back path: `if shutil.which("nvidia-smi") is None: return None` → outer code logs warning and skips Phase 2. In a containerized environment without nvidia-smi, the wait would skip OS-level verification entirely. What's the regression risk in production?

### R03 — Phase A/B init bootstrap

1. Phase A step 7 releases actor_train BEFORE Phase B step 1 requests actor_infer. What if the rlix scheduler grants a competing pipeline's request between those two calls? Could the current pipeline's Phase B then fail to get its expected GPUs? Walk the multi-pipeline collision case.
2. `with_ref=bool(args.use_kl_loss or args.kl_coef != 0.0)` — what if both flags are set inconsistently (e.g., `use_kl_loss=True, kl_coef=0.0`)? The current logic loads ref. Is that the standalone behavior?
3. Phase B step 6 `bootstrap_active_engines` immediately followed by step 7 `sync_base_weights_to_active(-1)`. If the engines just came up and step 6 marks them active without waiting for warmup, does the v=-1 base sync race against ongoing engine init? Trace.
4. `_expand_workers` no-op when engines are already-active: this is M11.1's "init-comes-up-active" hatch. What if the rlix scheduler later issues a real expand for an engine that has BEEN offloaded (post-train cycle)? Does the hatch swallow that, leaving the engine in "offloaded" but the scheduler bookkeeping in "active"?
5. `step_target_estimate` derivation at line 397 — what if `cluster_tp_configs` returns 0 (uninitialized) or if the dynamic batch size is 0? Does the estimate become 0 / NaN / negative and confuse the gap-ratio planner?
6. Phase A step 4 (build_cpu_bucket_cache(-1)) requires GPU-resident weights. Step 3.5 wakes them up. What if the wake fails (tms.resume error)? Does step 4 OOM with a confusing trace, or does it fail-fast with a clear error?

### R04 — Runtime hooks + train loop ordering

1. `_after_training` does: `build_cpu_bucket_cache(step)` → `train_group.offload()` → `coord.sync_base_weights_to_active(step)` → release actor_train. If `sync_base_weights_to_active` raises mid-way (e.g., a SGLang engine crashes), is actor_train still released? Could this leak the actor_train allocation in the rlix scheduler ledger?
2. `_before_training` requests actor_train at priority `ACTOR_TRAINING`. What if no GENERATION engines are currently active (e.g., between rollouts)? Does the scheduler grant immediately, or does it block waiting for a preempt that never happens?
3. The async train loop in `rlix_train_loop.run_async_train_loop` uses `await pipe.before_training.remote(step)`. If the pipeline actor dies between iterations, does the next iteration's await hang forever, or does it raise `RayActorError`?
4. `_after_training` line 574 forwards `global_step=int(step)`. Does the scheduler use this for cadence tracking? Could a non-monotonic step (e.g., resume from checkpoint with skipped steps) confuse it?
5. The dual driver's `asyncio.gather` runs two train loops. If one pipeline raises a fatal error, does `gather` cancel the other? Should the other be allowed to finish for partial progress (current behavior)?

### R05 — Multi-pipeline driver

1. `_per_pipeline_args` sets `args.sglang_router_port = None` so each pipeline auto-allocates. What if two pipelines call `find_available_port` concurrently and both bind 5001? (One should fail bind; what's the recovery path?)
2. `_split_pools_for_dual` returns `physical[:infer_pool_size]` and `physical[infer_pool_size:2*infer_pool_size]`. What about the remaining GPUs (`physical[2*infer_pool_size:]`) on a 6-GPU machine with infer_pool_size=2? Are they leaked or used by a third pipeline?
3. `_build_pipeline` creates the MilesCoordinator with `lifetime="detached"`. What happens on driver crash? Do the coordinators survive and accumulate as zombie actors across runs? Does the next driver run get a stale `RLIX_NAMESPACE`?
4. `MILES_ROLLOUT_BASE_PORT` is set per pipeline (15000 / 16000). Does miles' port allocation respect a 1000-port window, or could pipeline 1 expand into pipeline 2's range?
5. Dual driver builds P1 then P2 sequentially via `_build_pipeline`. What if P2's init fails halfway (e.g., GPU memory shortage)? Is P1 cleaned up, or does it run alone with no partner?
6. Each pipeline gets its own `cluster_device_mappings`. What if a user passes overlapping mappings by mistake (P1 train=[0], P2 train=[0])? Does the rlix scheduler reject the second `register_pipeline`, or does it silently allow the overlap?
7. `asyncio.gather` runs both train loops. Are there any process-globals (logging dirs, wandb run names, file caches) that the two pipelines share? `ROLL_LOG_DIR` is forwarded via `runtime_env`, but what about ad-hoc state in module-level vars?

### R06 — Coordinator naming + namespace

1. `f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"` — what if `pipeline_id` is itself unstable across allocator calls (e.g., a UUID with embedded timestamps)? Could two `register_pipeline` calls in the same run collide?
2. Per-pipeline namespace `pipeline_<pipeline_id>_NS` — is `RLIX_NAMESPACE` a parent or a sibling? If a child, can the scheduler in `RLIX_NAMESPACE` actually resolve actors in the per-pipeline namespace, or does it require explicit cross-namespace lookup?
3. The dual driver builds P1 and P2 sequentially. If P1's coordinator is named first and P2's coordinator collides on `pipeline_id` (allocate_pipeline_id should be sequential, but what if it's not?), does the second `.options(name=..., get_if_exists=True)` silently return P1's coordinator?
4. Coordinator `lifetime="detached"` survives driver exit. If a user restarts the driver, does the new run see the old coordinator (named conflict) and fail, or does the orchestrator clean up?
5. The MilesPipeline + MilesModelUpdateService + RolloutManager are all named per-pipeline. What if one of them is named the same as a Ray internal (e.g., `pipeline_master`)? Are there reserved name prefixes in Ray we should be avoiding?

## 6. Subagent Assignment Plan

| Subagent | Shards | Required inputs | Output file | Independence notes | Estimated effort | Status |
|---|---|---|---|---|---|---|
| reviewer-A | R01 | rlix HEAD `5dc4e43` checkout, miles HEAD `6126e01` checkout, plans/m11-e2e-test-log.md, docs/m11-implementation-guide.md | `<plan-stem>.review-report/R01.md` | Independent of R02–R06 | 60 min | READY |
| reviewer-B | R02 | rlix HEAD `5dc4e43` checkout, plans/m11-e2e-test-log.md, plans/m11-2-dual-pipeline-log.md | `<plan-stem>.review-report/R02.md` | Independent | 45 min | READY |
| reviewer-C | R03 | rlix HEAD `5dc4e43` checkout, miles standalone reference (`miles/ray/placement_group.py:192`) | `<plan-stem>.review-report/R03.md` | Independent | 60 min | READY |
| reviewer-D | R04 | rlix HEAD `5dc4e43` checkout, `miles/miles/utils/rlix_train_loop.py` | `<plan-stem>.review-report/R04.md` | Independent | 30 min | READY |
| reviewer-E | R05 | miles HEAD `6126e01` checkout, plans/m11-2-dual-pipeline-log.md | `<plan-stem>.review-report/R05.md` | Independent | 60 min | READY |
| reviewer-F | R06 | rlix HEAD `5dc4e43` checkout, miles HEAD `6126e01` checkout, rlix/protocol/types.py, rlix/scheduler/scheduler.py:~1213 | `<plan-stem>.review-report/R06.md` | Independent | 30 min | READY |

Total: 6 parallel subagents, 285 min sequential / 60 min wall-clock if fully parallel.

## 7. Subagent Report Template

```markdown
# Review Shard Report

Shard ID:
Reviewer:
Inputs reviewed:
Code areas reviewed:
Plan anchors:
Debug-log bugs reviewed:
Review angles:

## Verdict

`PASS | PASS_WITH_NOTES | NEEDS_FIX | BLOCKED_BY_MISSING_EVIDENCE | PLAN_REVISION_NEEDED`

## Findings

| Finding ID | Severity | Type | Evidence | Recommendation |
|---|---|---|---|---|

## Adversarial Checks

| Check | Result | Evidence |
|---|---|---|

## Tests / Evidence Reviewed

| Test / command | Result | Notes |
|---|---|---|

## Plan Impact

- Does PLAN.md need revision?
- Does debug-log need update?
- Should tldr-plan / scope-triage rerun?

## Reviewer Notes
```

Severity values: `BLOCKER`, `HIGH`, `MEDIUM`, `LOW`, `NOTE`.
Finding-type values: `CORRECTNESS`, `RACE`, `RESOURCE_LEAK`, `ERROR_HANDLING`, `TEST_GAP`, `PLAN_DRIFT`, `REGRESSION`, `OVERENGINEERING`, `OBSERVABILITY_GAP`, `PERFORMANCE`, `MAINTAINABILITY`.

## 8. Final Aggregation Contract

Rules for merging the 6 subagent shard reports into a final review report.

**v1 does NOT execute aggregation.** This contract is consumed by a human aggregator (or the future `review-report` skill).

### Step 1 — Collect

Index every shard report by `Shard ID`. Detect missing reports (a shard in §6 with no corresponding report file).

If any required shard report is missing, set `Final recommendation: BLOCKED` with `aggregation incomplete: <list of missing shard IDs>` until the human resolves.

### Step 2 — Deduplicate findings

Merge findings across shards by these equivalence rules (in order):

1. **Same root cause**: canonical signature is `(plan anchor, code area, invariant)`. If two findings share all three, they are duplicates.
2. **Same code area + same `Finding type`**: weaker dedup; surfaces in §3 of the final report as "potential duplicates" rather than auto-merged.
3. **Same failing test / evidence reference**: if two findings cite the same failing assertion, merge.

When merging:
- Preserve **every reviewer's evidence excerpt** verbatim.
- Preserve every reviewer's recommendation.
- Use the highest severity across the merged set.

### Step 3 — Group findings

Within the deduplicated set, group findings:
- Primary grouping: by `Severity` (`BLOCKER` → `HIGH` → `MEDIUM` → `LOW` → `NOTE`).
- Secondary grouping within severity: by `Finding type`.
- Tertiary: by code area.

### Step 4 — Extract Plan Revision Suggestions

For any shard report with `Verdict: PLAN_REVISION_NEEDED` OR `Plan Impact: Does PLAN.md need revision? = yes`:
- Open a corresponding `BUG-*` entry in `<plan-stem>.debug.md` with `Class: BUG-PLAN-GAP` or `BUG-PLAN-WRONG`.
- The aggregator does NOT assign `BUG-*` IDs directly; defer to the human or the `debug-log` skill.

### Step 5 — Extract debug-log updates needed

Any shard finding that would constitute a new `BUG-*` is enumerated in the final report's "Debug-log updates required" section.

### Step 6 — Compute final recommendation

| Condition | `Final recommendation` |
|---|---|
| Any `BLOCKER` severity finding OR any shard verdict `BLOCKED_BY_MISSING_EVIDENCE` (without explicit human waiver) | `BLOCKED` |
| Any shard verdict `NEEDS_FIX`, OR any `HIGH` severity finding | `NEEDS_FIX` |
| All shard verdicts `PASS` / `PASS_WITH_NOTES`, no `BLOCKER` / `HIGH` findings, but `MEDIUM` / `LOW` findings present | `APPROVE_WITH_FOLLOWUPS` |
| All shard verdicts `PASS` / `PASS_WITH_NOTES`, only `NOTE` findings (or no findings) | `APPROVE` |
| Any shard verdict `PLAN_REVISION_NEEDED` (without already counting toward another tier) | `NEEDS_FIX` |

### Step 7 — Final report shape

The aggregator produces a final report (separate file: `plans/m11-review.review-report.md`).

Suggested layout:

```markdown
# Code Review Report

Source plan:
Code review plan:
Reports collected:
Aggregation date:

## 0. Executive Summary
- Final recommendation:
- Shard reports: <count> collected, <count> missing
- Findings: <count BLOCKER>, <count HIGH>, <count MEDIUM>, <count LOW>, <count NOTE>
- Plan revisions surfaced:
- Debug-log updates required:

## 1. Blocking Findings (BLOCKER + HIGH)
## 2. Non-blocking Findings (MEDIUM + LOW + NOTE)
## 3. Evidence Gaps
## 4. Adversarial Review Results
## 5. Debug-log Recurrence Risks
## 6. Plan Revisions Surfaced
## 7. Debug-log Updates Required
## 8. Test Coverage Assessment
## 9. Final Recommendation
```

### What the aggregator MUST NOT do

- Do not edit `PLAN.md`, `m11-review.plan.md`, sibling artifacts.
- Do not change shard verdicts.
- Do not invent findings the reviewers did not report.
- Do not silently drop low-severity findings.
- Do not assign `BUG-*` IDs unilaterally.

## 9. Review Execution Checklist

- [ ] Author clarification answers (none required this run) folded in.
- [ ] All §6 rows have `Status: READY`.
- [ ] Each subagent (reviewer-A through reviewer-F) has been briefed with their assigned shard.
- [ ] Aggregator has the §8 contract.
- [ ] Plan and code diff are at the commits cited in §0 (rlix `5dc4e43`, miles `6126e01`).
- [ ] Reviewers know to surface `PLAN_REVISION_NEEDED` findings via the report contract.
- [ ] Reviewers know that the M11 deferred items (F22, multi-pipeline 3+, save/eval at final rollout, vast image hotpatch, no pytest tests) are **out of scope**.
- [ ] **CRITICAL**: a FRESH dual smoke has been run on a vast instance with rlix HEAD ≥ `549cfbd` (the engine-index conversion fix landed AFTER M11.2 attempt 4's PASS log, which therefore showed the KeyError warning). R02 + R05 verdicts depend on this fresh smoke producing a clean `_wait_for_overlap_engines_offloaded` log for MP2 (no `KeyError('unknown engine_index')`).

## 10. Annex A — Executable test commands per shard (revised after Codex review)

Vast target for live runs: `ssh4.vast.ai:27893` (instance 36267892, currently stopped — restart before §10.3+).

### §10.1 Static review (R01–R04, no GPU needed)

Walk all the named anchors in one pass:

```bash
cd /Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rlix_miles && \
rg -n "pause_generation|enable_memory_saver|MILES_TMS_HOOK_MODE|_wait_for_overlap_engines_offloaded|with_ref|step_target_estimate|def _before_training|def _after_training|run_async_train_loop" \
  rlix /Users/zhenyulin/Library/CloudStorage/Dropbox/Python/miles
```

Sanity-check each shard's `Code areas to inspect` line refs match the actual file content.

### §10.2 Static review (R05 + R06)

```bash
cd /Users/zhenyulin/Library/CloudStorage/Dropbox/Python/rlix_miles && \
sed -n '50,200p;220,377p' /Users/zhenyulin/Library/CloudStorage/Dropbox/Python/miles/examples/rlix/run_miles_dual.py && \
sed -n '1196,1220p' rlix/scheduler/scheduler.py && \
sed -n '1,50p' rlix/protocol/types.py
```

Verify per-pipeline pools / port env / namespace / coordinator lookup symbols.

### §10.3 Fresh dual smoke (R02 + R05; required for sign-off)

> Required because M11.2 attempt 4's PASS log preceded the engine-index fix (`549cfbd`). Run on the latest pushed branches.

```bash
# Restart vast (locally)
~/.venv/vast/bin/vastai start instance 36267892
# Wait for state=running, then SSH in:
ssh -i ~/.ssh/general_private_key -p 27893 root@ssh4.vast.ai

# On vast: pull latest rlix + miles to ensure 549cfbd / 6126e01 are present
cd /root/rlix && git pull
cd /root/miles && git pull

# Run the dual smoke under the watchdog
tmux new-session -d -s smoke 'SCRIPT=/root/rlix/scripts/run_smoke_dual.sh \
  SILENCE_LIMIT=900 RUN_LIMIT=3600 \
  bash /root/rlix/scripts/run_smoke_with_watchdog.sh; \
  echo === DONE ===; sleep 120'

# Watch for green / regress
tail -F /root/logs/run.log | grep -E "rollout_id=|EXIT_CODE=|KeyError|Traceback|shutdown_hard"
```

**Success criteria** (all must hold):
- `EXIT_CODE=0`
- Both pipelines log `[run_miles_dual] mp{1,2} training loop complete`
- Both log `shutdown_hard complete`
- **No** `KeyError('unknown engine_index ...')` warning (confirms Feature 11 fix landed)

### §10.4 R06 destructive naming test (deliberately bad coordinator name)

> Run AFTER §10.3. Corrupts the single-pipeline driver to verify the scheduler catches the failure.

```bash
ssh -i ~/.ssh/general_private_key -p 27893 root@ssh4.vast.ai

# On vast: backup, corrupt the name, run smoke
cd /root/miles
cp examples/rlix/run_miles_rlix.py /tmp/run_miles_rlix.py.bak
perl -0pi -e 's/name=f"\{COORDINATOR_ACTOR_NAME_PREFIX\}\{pipeline_id\}"/name=f"bad_coordinator_\{pipeline_id\}"/' \
  examples/rlix/run_miles_rlix.py

# Run single smoke and confirm scheduler logs the resolve failure
bash /root/rlix/scripts/run_smoke_e2e.sh 2>&1 | tee /root/m11-bad-name.log

# Restore + verify
cp /tmp/run_miles_rlix.py.bak examples/rlix/run_miles_rlix.py
grep -c "Failed to resolve actor" /root/m11-bad-name.log
# Expected: count > 0 (scheduler tried to resolve coordinator and failed)
```

**Success criteria**: scheduler logs at least one `Failed to resolve actor` event when the coordinator is mis-named. Confirms Feature 16's invariant is load-bearing.

### §10.5 After all tests: stop vast

```bash
~/.venv/vast/bin/vastai stop instance 36267892
```

## Annex B — Codex revisions applied (provenance)

This brief was revised once after Codex review (codex:rescue agent). Applied changes:
- Added §10 Annex A with executable test commands per shard (HIGH finding: test-case concreteness)
- Added §9 checklist item requiring a fresh post-`549cfbd` dual smoke (HIGH finding: R02 + R05 evidence gap)
- Acknowledged but did NOT split R02 (MEDIUM finding: shard sharpness): R02's three concerns (TMS wake/offload, nvidia-smi probe, engine-index arithmetic) all stem from `_wait_for_overlap_engines_offloaded` and share the same evidence trail; splitting would inflate review surface without sharpening findings.
- Acknowledged but did NOT prune adversarial prompts (LOW finding): the abstract prompts (R06.1 pipeline-id collision, R06.5 Ray reserved names, R04.2 no GEN engines) MAY surface theoretical bugs even though they are not tied to observed M11 failures; reviewer free to skip.
