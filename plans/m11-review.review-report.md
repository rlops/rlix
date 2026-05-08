# Code Review Report — M11.1 + M11.2

Source plan: `plans/m11-review.plan.md`
Code review plan: `plans/m11-review.review.md`
Reports collected: `plans/m11-review.review-report/{R01,R02,R03,R04,R05,R06}.md`
Aggregation date: 2026-05-07
Code under review: rlix `zhenyu/miles-mvp-e2e@5dc4e43` + miles `zhenyu/m11-mvp-test@6126e01`

## 0. Executive Summary

- **Final recommendation: NEEDS_FIX**
- Shard reports: 6 collected, 0 missing
- Findings: 0 BLOCKER, 1 HIGH, 3 MEDIUM, 2 LOW, 4 NOTE
- Shard verdicts: R01 PASS, R02 PASS_WITH_NOTES, R03 PASS, R04 PASS_WITH_NOTES, R05 PASS_WITH_NOTES, R06 PASS_WITH_NOTES
- Tests-passing evidence: M11.1 attempt 10 (`EXIT_CODE=0`, 4xRTX5090) + M11.2 attempt 4 (`EXIT_CODE=0`, 4xA40)
- Plan revisions surfaced: 2 (M11.2 failure-recovery section; rlix_train_loop exception-safety contract)
- Debug-log updates required: 0 new bugs found by review; 3 advisory entries (F13 design choice, F13 edge case, error-path documentation)

**One-line bottom line**: All 6 shards reach `PASS` or `PASS_WITH_NOTES` on the happy path. The single HIGH finding (R04-F1) is a **failure-path resource leak** — the async train loop has no `try/finally` around `before_step / train / after_step`, so a `train()` exception leaks the rlix scheduler's GPU allocation. Happy-path E2E (M11.1 attempt 10, M11.2 attempt 4) is unaffected; production hardening is required before multi-pipeline 3+ scale-out.

## 1. Blocking Findings (BLOCKER + HIGH)

### F1 [HIGH] R04-F1 — GPU allocation not released if `train()` raises mid-iteration

- **Severity**: HIGH
- **Type**: ERROR_HANDLING_FAIL_FAST
- **Plan anchors**: D5, AC #1
- **Code area**: `miles/miles/utils/rlix_train_loop.py` (async loop), `rlix/pipeline/miles_pipeline.py:566-596` (`_before_training`), `miles_pipeline.py:598-660` (`_after_training`)
- **Reporting shard**: R04

#### Background
The `_before_training` hook claims a GPU allocation from the rlix scheduler via `_request_cluster_gpus`, sets `_actor_train_allocated=True` (miles_pipeline.py:578). The matching `_after_training` hook releases the allocation via `_notify_release_cluster_gpus` and then flips the flag back to False (lines 614-618). Per AC #1, the sequence must bracket each rollout iteration.

#### Expected behavior / invariant
The release must occur regardless of whether training succeeds or fails. Otherwise the rlix scheduler ledger holds the allocation indefinitely and subsequent pipelines (or restarted pipelines) cannot acquire the same GPU pool.

#### Finding
The async training loop in `rlix_train_loop.run_async_train_loop` does NOT wrap `before_step / train_group.train / after_step` in `try/finally`. If `train_group.train()` raises:
1. Exception propagates to the caller's `asyncio.run`.
2. `after_step()` is never invoked.
3. `_notify_release_cluster_gpus` never fires.
4. The scheduler's ledger keeps the allocation.
5. `_actor_train_allocated` stays True. (`shutdown_hard` would retry, but only if reached.)
6. Driver exits with unclean ledger.

#### Evidence
- Code: `miles/miles/utils/rlix_train_loop.py` — for-loop has no exception handling around the train step.
- Code: `miles_pipeline.py:608-614` — `sync_base_weights_to_active` is also outside any `try/finally`; if it raises, release never fires either.
- M11.1 attempt 8 demonstrates that mid-step errors crash the actor and the next iteration cannot proceed.
- M11.1 attempt 10, M11.2 attempt 4 both show clean sequential completion (happy path only).

#### Impact
HIGH for production / multi-pipeline 3+:
- Any `train()` raise (SGLang OOM during sync, network partition, CUDA OOM, gradient explosion, megatron checkpoint corruption) leaks the GPU allocation indefinitely.
- The rlix scheduler cannot recover the GPU for other pipelines without manual intervention (`ray stop`).
- M11.1 / M11.2 happy-path tests do not exercise this path; the issue is undetected by current smoke coverage.

#### Recommended action
Add explicit `try/finally` bracketing in `rlix_train_loop.run_async_train_loop`:

```python
try:
    await before_step(rollout_id)
    await train_group.train(rollout_id, rollout_data_curr_ref)
finally:
    await after_step(rollout_id)
```

Optionally wrap the driver's `asyncio.run(_async_main())` in a `try/finally` that calls `pipeline.shutdown_hard()` to cover the driver-crash path (R04-F2 below).

#### Routing
- **Debug-log**: Document error-path behavior as new BUG-* entry once the human-aggregator approves; mark as POST_M11_HARDENING.
- **PLAN.md revision**: Add an exception-safety contract for `rlix_train_loop` to the M11.1 / M11.2 "Known Limitations" section (or carry into M11.3 design).
- **Rerun needed**: Manual smoke test with injected failure (raise inside `train_group.train` after first rollout) — confirm the fix releases GPUs cleanly.

## 2. Non-blocking Findings (MEDIUM + LOW + NOTE)

Grouped by severity → finding type → code area.

### MEDIUM (3)

| Finding ID | Type | Title | Plan anchors | Code area | Shard | Recommendation |
|---|---|---|---|---|---|---|
| F2 / R02-01 | CORRECTNESS | 20 GB free threshold not derived from model size | D4, AC#1 | `miles_pipeline.py:494` | R02 | Document threshold rationale; consider deriving from `miles_args.model_size`; add `--min-free-gpu-mem-gb` flag for arbitrary models. |
| F3 / R04-F2 | ERROR_HANDLING / RACE | Driver crash before `shutdown_hard` skips scheduler release | D5 | `rlix_train_loop.py`, `miles_pipeline.py:769-799` | R04 | Wrap `asyncio.run(_async_main())` in driver-side `try/finally` calling `shutdown_hard`. |
| F4 / R04-F3 | MAINTAINABILITY | Dual driver `asyncio.gather` does not cancel peer coroutine on failure | D5 | `run_miles_dual.py:355-365` | R04 | Pass `return_exceptions=True` and inspect results, OR wrap gather in `try/finally` calling `shutdown_hard` for both pipelines. |

#### F2 / R02-01 — 20 GB free threshold hardcoded
Phase 2 of `_wait_for_overlap_engines_offloaded` polls `nvidia-smi --query-gpu=memory.free` and continues training once `min_free_gb >= 20.0` (literal constant). Safe for the 0.5B model (~3.7 GB weights) used in M11.1; not principled for 7B+ models. Users running larger models could face OOM if the driver releases memory asynchronously in parallel with train wake_up.

#### F3 / R04-F2 — Driver crash skips shutdown_hard
`_notify_release_cluster_gpus` only retries cleanup via `shutdown_hard` (miles_pipeline.py:660-680) if `shutdown_hard` is reached. If the driver crashes before that (or `asyncio.run` reraises before the driver's outer `try/finally` — which currently does not exist), cleanup is skipped entirely.

#### F4 / R04-F3 — `asyncio.gather` peer not cancelled on failure
`run_miles_dual.py:355-365` uses `asyncio.gather(...)` without `return_exceptions=True`. With default semantics, gather propagates the first exception immediately but does NOT cancel peer coroutines; pipeline B's coroutine continues running in its Ray actor (orphaned) while the driver process tears down.

### LOW (2)

| Finding ID | Type | Title | Plan anchors | Code area | Shard | Recommendation |
|---|---|---|---|---|---|---|
| F5 / R02-02 | OBSERVABILITY_GAP | nvidia-smi timeout logged at debug level only | D4, AC#1 | `miles_pipeline.py:540-552` | R02 | Promote timeout log to INFO; consider counter/metric. |
| F6 / R03-001 | CORRECTNESS (design comment) | `_after_training` does not hold lock when calling `sync_base_weights_to_active`; coordinator-side serialization handles this | D5, AC #1 | `miles_pipeline.py:608-610` | R03 | No fix required — design is sound (documented in code). |

### NOTE (4)

| Finding ID | Type | Title | Plan anchors | Code area | Shard | Recommendation |
|---|---|---|---|---|---|---|
| F7 / R01-001 | OBSERVABILITY_GAP | Broad `except Exception` catch in `shrink_engines` masks SGLang API contract | D3, AC #2 | `miles/ray/rollout.py:934` | R01 | Document SGLang `pause_generation` API contract (idempotency, error codes); add conditional logging if pause-count exceeds threshold. |
| F8 / R05-01 | RESOURCE_LIFECYCLE | Zombie `MilesCoordinator` (`lifetime="detached"`) on driver crash; P1 survives if P2 init fails halfway | D2, AC #5 | `run_miles_dual.py:202-212` (no top-level try/except per F13 hard constraint) | R05 | Document P2 init failure recovery (user runs `ray stop`); add cleanup logic in M11.3. |
| F9 / R05-02 | CORRECTNESS | `_split_pools_for_dual` does not validate odd GPU counts | AC #5 | `run_miles_dual.py:54-72` | R05 | Add explicit validation with warning or rejection; document design choice. |
| F10 / R06-F1 | OBSERVABILITY_GAP | Unnamed `lifetime="detached"` coordinator may persist after driver crash | AC #5 | `run_miles_rlix.py:203`, `run_miles_dual.py:207` | R06 | Document cleanup workflow; add `orchestrator.cleanup_stale_pipelines()` RPC in M11.3. |

**Note on potential dedup (Step 2.2 of §8 contract)**: F8 (R05-01) and F10 (R06-F1) both describe `lifetime="detached"` orphaned coordinators on driver crash. They share `Finding type` and code area but cite different invariants (R05 frames it as a per-pipeline init-failure leak; R06 frames it as a long-lived cluster observability gap). Per the conservative "≥2-signal" rule they are NOT auto-merged but are flagged here as partial duplicates — both are resolved by the same M11.3 cleanup RPC.

## 3. Evidence Gaps

| Gap ID | Source shard | Missing evidence | Why it matters | Suggested next action |
|---|---|---|---|---|
| E1 | R04 | No injected-failure tests (train() raise, sync raise, RPC failure) | Failure paths are R04-F1's blast radius; all happy-path-only today | Add fault-injection smoke in M11.x (raise inside `train_group.train` after first rollout, assert scheduler ledger empty after teardown). |
| E2 | R05 | No P2 init-failure case exercised in M11.2 smoke | Behavior under partial init failure unknown | Run injected-failure test in M11.3. |
| E3 | R05 | No odd GPU count edge case tested | Silent GPU leak on 5/6/7-GPU machines | Add validation + smoke. |
| E4 | R06 | No explicit test of stale-coordinator cleanup after driver crash + restart | Hard constraint defers cleanup to M11.3 | Out of scope for this shard; pick up in M11.3. |
| E5 | R06 | No test of Ray reserved-name collision | Ray's internal naming opaque | Out of scope; documented Ray guarantees. |
| E6 | §0 dashboard | M11.2 attempt 4 PASSED **before** the engine-index conversion fix landed (commit 549cfbd, Feature 13) | The PASS log is not technically post-fix evidence for R02 / R05 | Run a fresh M11.2 dual smoke on vast at HEAD `5dc4e43` / `6126e01` per `m11-review.review.md` §10 Annex A.3. |

E6 is the most actionable: it determines whether F2 / F3 / F4 / F8 / F9 actually need post-fix verification or remain advisory. The `m11-review.review.md` §9 checklist already requires this rerun.

## 4. Adversarial Review Results

35+ adversarial prompts were issued across the 6 shards. Aggregate result:

| Shard | Adversarial checks | PASS | PASS_WITH_NOTES | OUT_OF_SCOPE | FAIL |
|---|---|---|---|---|---|
| R01 | 7 | 4 | 2 | 2 | 0 |
| R02 | 6 | 5 | 0 | 0 | 1 (R02.2 → F2) |
| R03 | 6 | 6 | 0 | 0 | 0 |
| R04 | 5 | 2 | 0 | 2 | 1 (gather → F4) |
| R05 | 7 | 4 | 1 | 1 | 1 (R05-Q5 → F8 documented) |
| R06 | 6 | 6 | 0 | 0 | 0 |

No adversarial check escalated above the severity already captured in §1–§2. Notable PASS results worth recording for the audit trail:

- R02.3 (non-contiguous infer mapping `[0,2,4,6]`) → engine-index conversion correct after Feature 13 fix.
- R03.4 (`_expand_workers` already-active hatch is sticky?) → SAFE; phase-specific, not sticky. Engines transition to "offloaded" after first shrink.
- R05-Q1 (`find_available_port` collision P1+P2) → race avoided by sequential `_build_pipeline` + `MILES_ROLLOUT_BASE_PORT` 15000/16000 disjoint ranges.
- R06.2 (RLIX_NAMESPACE parent vs sibling?) → SIBLING, safe; Ray supports cross-namespace lookup via explicit namespace param.

## 5. Debug-log Recurrence Risks

| Risk | Source bug history | Recurrence vector | Severity if recurs | Guard now in place? |
|---|---|---|---|---|
| Failed to resolve actor (M11.1 attempt 4 → commit 7b83be5) | M11.1 attempt 4 | Driver and scheduler use different name format | HIGH | YES — `COORDINATOR_ACTOR_NAME_PREFIX` constant + `get_pipeline_namespace()` helper centralized in `rlix/protocol/types.py`; R06 confirms |
| Triton-CPU pointer race during `release_memory_occupation` (M11.1 attempts 5–7) | M11.1 attempts 5–7 | Calling `release_memory_occupation` while SGLang scheduler thread is mid-kernel | HIGH | YES — `pause_generation(mode="retract")` wrapping; R01 confirms |
| Double `wake_up` (M11.1 attempt 8) | M11.1 attempt 8 | `_before_training` calls onload AND `train()` self-wakes when `offload_train=True` | MEDIUM | YES — `_before_training` documented to NOT call onload (miles_pipeline.py:589-596); R02.4 + R04 confirm |
| `with_ref=False` torch.cat(NoneType) crash (M11.1 attempt 9 → commit 90f47c4) | M11.1 attempt 9 | Inconsistent KL flag derivation between standalone and rlix paths | HIGH | YES — `with_ref=bool(args.use_kl_loss or args.kl_coef != 0.0)` matches standalone (placement_group.py:192); R03 confirms |
| Engine-index KeyError on non-contiguous infer mapping (M11.2 attempt 4 → commit 549cfbd) | M11.2 attempt 4 | Phase 2 wait function used physical GPU index where local engine index was expected | MEDIUM | YES — engine-index conversion subtracts `min(infer_mapping)` offset; R02.3 confirms. **E6 caveat applies**: needs fresh post-fix smoke. |
| ulimit SIGABRT (M11.2 attempts 1–2) | M11.2 attempts 1–2 | Ray actor processes hit nofile cap | MEDIUM | YES — driver script raises ulimit; smoke confirms |

## 6. Plan Revisions Surfaced

| Revision ID | Source shard | Title | Class | Rationale | Suggested target |
|---|---|---|---|---|---|
| PR-01 | R04 | Document exception-safety contract for `rlix_train_loop` (`try/finally` around `before_step / train / after_step`) | PLAN-GAP | F1 (HIGH) is the blast-radius of an unstated invariant; PLAN currently treats happy-path bracketing as sufficient | M11.1 PLAN "Known Limitations" + carry into M11.3 design as "exception-safety contract" |
| PR-02 | R05 | Add "M11.2 Failure Semantics" section to PLAN | PLAN-GAP | F8 (NOTE) calls out that the F13 hard constraint (no top-level try/except) requires user discipline (`ray stop` between runs); PLAN should record this contract | M11.2 PLAN |

Neither PR-01 nor PR-02 is auto-applied. Per §8 contract Step 4, the human aggregator (or the `debug-log` skill) opens the corresponding `BUG-PLAN-GAP` entries.

## 7. Debug-log Updates Required

The 6 shards collectively did NOT identify any new bugs requiring fresh `BUG-*` IDs in `<plan-stem>.debug.md`. Three advisory entries are recommended:

| Update ID | Source shard | Suggested entry | Class |
|---|---|---|---|
| DBG-01 | R04 | Document the `train()` exception path leaving GPU allocation in scheduler ledger as a known limitation; reference R04-F1 / F1 of this report | POST_M11_HARDENING |
| DBG-02 | R05 | Record F13 design choice ("no top-level try/except in dual driver per F13 scope") to make recovery procedure (`ray stop`) explicit | F13-DESIGN-CHOICE |
| DBG-03 | R05 | Record F13 edge case ("`_split_pools_for_dual` accepts odd GPU counts silently") | F13-EDGE-CASE |

Aggregator does NOT assign `BUG-*` IDs. The human or `debug-log` skill assigns them.

## 8. Test Coverage Assessment

### Coverage by shard

| Shard | Happy-path E2E | Failure-path | Static-analysis-only |
|---|---|---|---|
| R01 (SGLang quiescence) | YES (M11.1 attempt 10, M11.2 attempt 4) | NO (no SGLang 4xx / 5xx induced) | NO |
| R02 (tms region lifecycle) | YES (M11.1 attempt 5–10) | PARTIAL (nvidia-smi-fails branch via fallback) | NO |
| R03 (Phase A/B init) | YES (M11.1 attempt 9, 10; M11.2 attempt 4) | NO | NO |
| R04 (Runtime hooks) | YES (M11.1 attempt 10) | NO (this is the F1 gap) | YES on `try/finally` analysis |
| R05 (Multi-pipeline driver) | YES (M11.2 attempt 4 — pre-engine-index-fix; needs E6 rerun) | NO (no P2 init-failure injected) | NO |
| R06 (Coordinator naming) | YES (M11.1 attempt 4 fail → 10 pass; M11.2 attempt 4 dual-pipeline collision-free) | NO (no Ray reserved-name collision injected) | NO |

### Coverage gaps

- **Failure-path coverage is the single biggest gap.** F1 (HIGH), E1, E2, E3, E4 all surface here.
- **Post-engine-index-fix smoke (E6)**: M11.2 attempt 4 PASSED on commit 549cfbd^ (parent), so it is **not** definitive evidence for R02 / R05 / Feature 13. The §10 Annex A.3 rerun on vast (after 549cfbd / 6126e01) is required to close this evidence gap.
- **Unit tests for the 6 critical files**: PLAN explicitly accepts this gap; aggregator does NOT file `TEST_GAP` against it.

### Coverage strengths

- M11.1 iterated 10 attempts — strong evidence the fixes are causally load-bearing (each fix has a matched fail/pass log pair).
- M11.2 dual-pipeline cross-validates per-pipeline isolation (namespaces, RolloutManager port ranges, deep-copy args).
- Feature 13 (engine-index conversion) has a clear regression marker: M11.2 attempt 4 warning → commit 549cfbd → rerun would prove fix.

## 9. Final Recommendation

**Final recommendation: NEEDS_FIX**

### Rationale

Per §8 Step 6 of `m11-review.review.md`:
- 0 BLOCKER → not BLOCKED.
- 1 HIGH (F1 / R04-F1) → **NEEDS_FIX** (any HIGH triggers this tier).
- All shard verdicts are PASS or PASS_WITH_NOTES — no shard is BLOCKED.

### Required for ship

1. **Fix F1 / R04-F1**: Wrap `before_step / train / after_step` in `try/finally` in `rlix_train_loop.run_async_train_loop`. Verify with injected-failure smoke (raise inside `train_group.train` after first rollout; assert rlix scheduler ledger empty after teardown).
2. **Run §10 Annex A.3 (post-engine-index-fix M11.2 smoke)** on vast at `5dc4e43` / `6126e01` to close E6.

### Recommended for follow-up (M11.3)

- F2: derive 20 GB threshold from model size (or expose `--min-free-gpu-mem-gb`).
- F3: wrap driver `asyncio.run(_async_main())` in `try/finally` calling `shutdown_hard`.
- F4: pass `return_exceptions=True` to `asyncio.gather` in `run_miles_dual.py` and inspect results.
- F8 + F10: implement `orchestrator.cleanup_stale_pipelines()` RPC.
- F9: validate odd GPU counts in `_split_pools_for_dual`.

### Out-of-scope per hard constraints (do NOT block ship)

- Multi-pipeline 3+ E2E (out of scope per source plan).
- Ray reserved-name collision testing (Ray-internal).
- Save/eval at final rollout (broken-by-design per source plan).
- F22 shell-init torch_memory_saver path (deferred per source plan).

## 10. Appendix

### 10.A — Per-shard report files

| Shard | File | Verdict | Findings |
|---|---|---|---|
| R01 | `plans/m11-review.review-report/R01.md` | PASS | 1 NOTE |
| R02 | `plans/m11-review.review-report/R02.md` | PASS_WITH_NOTES | 1 MEDIUM, 1 LOW |
| R03 | `plans/m11-review.review-report/R03.md` | PASS | 1 LOW |
| R04 | `plans/m11-review.review-report/R04.md` | PASS_WITH_NOTES | 1 HIGH, 2 MEDIUM |
| R05 | `plans/m11-review.review-report/R05.md` | PASS_WITH_NOTES | 2 NOTE |
| R06 | `plans/m11-review.review-report/R06.md` | PASS_WITH_NOTES | 1 NOTE |

### 10.B — Aggregator did NOT (per §8 contract)

- Edit `PLAN.md`, `m11-review.plan.md`, or sibling artifacts.
- Change shard verdicts.
- Invent findings the reviewers did not report.
- Drop low-severity findings silently.
- Assign `BUG-*` IDs unilaterally (suggestions in §7 only).

Report readiness: COMPLETE
