# Debug Log

Feature: MILES → RLix port (M11.1 + M11.2 code-complete pass)
Source plan: /Users/tao/Projects/miles_port/plans/miles-port-unified-plan.md
TLDR context: /Users/tao/Projects/miles_port/plans/miles-port-unified-plan.tldr.md
Scope context: /Users/tao/Projects/miles_port/plans/miles-port-unified-plan.scope.md (optional)
Current smoke target: M11.1 Gate 1/2/2.5/3 + M11.2 Gate 4 (deferred — code-complete only, no GPU smoke this run)
Current status: BLOCKED (smoke deferred — Mac-local syntax-only environment cannot exercise GPU paths)

## 0. Session Summary

| Session | Date | Test target | Result | Bugs opened | Bugs resolved | Remaining blockers |
|---|---|---|---|---:|---:|---|
| SESSION-001 | 2026-05-04 | post-/review-report HIGH-finding fix sweep | IN_PROGRESS | 14 | 0 | All 14 BUG-* `FIXED_PENDING_VERIFY` — awaiting GPU smoke run |

## 1. Active Bugs

| Bug ID | Status | Class | Symptom | Root cause confidence | Blocking test | Plan anchors |
|---|---|---|---|---|---|---|
| BUG-001 | FIXED_PENDING_VERIFY | BUG-IMPLEMENTATION | `_preempted_engines.discard` mutates outside `_abort_engines` (Layer-1 invariant scope leak) | HIGH | M11.1 Gate 4 (router C20) regression grep | A19, C-grep §3.1 line 112 |
| BUG-002 | FIXED_PENDING_VERIFY | BUG-IMPLEMENTATION | `set_weight_version` filter diverges between `engine_indices=None` and explicit-list branches | HIGH | M11.2 Gate 4 (a) runtime-expand smoke | C21 (F21), engine table line 4181 |
| BUG-003 | FIXED_PENDING_VERIFY | BUG-IMPLEMENTATION | `_abort_engines` idempotency cache survives `shrink_engines` failure → retry skips abort | MEDIUM | manual abort-drain-failure injection test | C23 (F23) |
| BUG-004 | FIXED_PENDING_VERIFY | BUG-REGRESSION | `partial_rollout` assert relocated under `rlix_mode` breaks standalone pytest contract | HIGH | `pytest tests/fast/rollout/generate_hub/test_multi_turn.py::TestExitConditions::test_partial_rollout_not_supported` | C29 (F29), C17 |
| BUG-005 | FIXED_PENDING_VERIFY | BUG-INTEGRATION | `destroy_collective_group` only catches HTTP 404; SGLang returns HTTP 400 BAD_REQUEST | HIGH | M11.2 Gate 4 second-pass NCCL teardown | A3 (Anti-regression invariant #3) |
| BUG-006 | FIXED_PENDING_VERIFY | BUG-RACE-CONDITION | `asyncio.wait_for` cancellation does not abort inflight Ray RPCs → cache_owner `_cache_lock` leak (P2-12) | HIGH | dual-pipeline timeout-then-next-call sequencing test | C15 (F15), A1 |
| BUG-007 | FIXED_PENDING_VERIFY | BUG-IMPLEMENTATION | SharedStorage port-claim retry can burn budget against same port without advancing | LOW | manual port-claim retry probe | C26 (F26) |
| BUG-008 | FIXED_PENDING_VERIFY | BUG-PLAN-GAP | `examples/rlix/run_miles_rlix.py` iter-16 stub never wires orchestrator + coordinator + pipeline | HIGH | `python -m examples.rlix.run_miles_rlix` end-to-end | C11 (F11), C13 |
| BUG-009 | FIXED_PENDING_VERIFY | BUG-IMPLEMENTATION | C9 PD-disaggregation guard silently skipped at entry path because caller passes `sglang_config=None` | HIGH | unit test where `args.sglang_config.has_pd_disaggregation == True` raises RuntimeError | C9 (F10) |
| BUG-010 | FIXED_PENDING_VERIFY | BUG-INTEGRATION | `WorkerPlacement.gpu_ids` stores global GPU ids despite docstring/plan declaring node-local; multi-node CVD breaks on node_rank > 0 | HIGH | 2-node smoke test | F12 (a), C-multi-node line 3502 |
| BUG-011 | FIXED_PENDING_VERIFY | BUG-RACE-CONDITION | `_resize_sync_lock` held across full 4-RPC chain in `_expand_workers` blocks all concurrent coordinator RPCs (P2-9) | HIGH | concurrent `report_progress_from_scheduler` during long `resize_infer` | C20, A20 |
| BUG-012 | FIXED_PENDING_VERIFY | BUG-IMPLEMENTATION | `_actor_train_allocated` flipped to False before scheduler release confirmed → ledger desync if release RPC swallows error | HIGH | fault-injection unit test (raise inside `_train_group.init()`) | A10 (Anti-regression invariant #10), C36 (F36) |
| BUG-013 | FIXED_PENDING_VERIFY | BUG-PLAN-GAP | `_init_phase_b_infer` self-allocates fresh Ray PG via `_create_placement_group`, double-counting against rlix scheduler grant | HIGH | M11.2 dual-pipeline GPU oversubscription smoke | C35 (F35), C22 (F22 deferred) |
| BUG-014 | FIXED_PENDING_VERIFY | BUG-IMPLEMENTATION | Outer `ray.get(rollout_manager.shutdown_hard.remote())` has no timeout → wedged manager blocks ledger release | MEDIUM | manual ray.get-hang injection probe | A10 (Anti-regression invariant #10), C36 (F36) |

## 2. Resolved Bugs

(none yet — all 14 BUG-* await GPU smoke verification)

## 3. Bug Entries

### BUG-001: `_preempted_engines.discard` mutates outside `_abort_engines` (Layer-1 invariant scope leak)

- Status: FIXED_PENDING_VERIFY
- Class: BUG-IMPLEMENTATION
- First observed: 2026-05-04 (R01 + R12 review subagents)
- Repro / test command: `grep -rn "_preempted_engines" external/miles/miles/` returns 7 hits including `shrink_engines:893-894` outside `_abort_engines`
- Environment: Mac-local code review; no runtime exercise
- Observed behavior: `shrink_engines` mutates `_preempted_engines` directly via `discard(idx)` after offload state flip
- Expected behavior: PLAN.md §3.1 line 112 mandates "ONLY as `RolloutManager._abort_engines` internal abort-idempotency cache" — every read AND write must reside lexically inside `_abort_engines`
- Failure excerpt: `external/miles/miles/ray/rollout.py:893-894` `if hasattr(self, "_preempted_engines"): self._preempted_engines.discard(idx)`
- Root cause: when iter-5 wrote shrink_engines's Step-6 cleanup loop, the cache discard was inlined for proximity to the state flip. Spirit-of-rule preserved (still abort-idempotency-cache scope, not routing/attribution/resize-safety) but literal "ONLY in `_abort_engines`" wording violated.
- Root cause confidence: HIGH
- Fix: added `_release_abort_idempotency_for(indices)` private helper colocated with `_abort_engines`; `shrink_engines` calls helper after Step-6 state flips. Added complementary `_reset_abort_idempotency_for(indices)` helper that `shrink_engines`'s try/except calls on failure (closes BUG-003).
- Files changed: `external/miles/miles/ray/rollout.py`
- Verification: pending — planned command: `grep -n "_preempted_engines" external/miles/miles/ray/rollout.py` (expect mutations only inside `_abort_engines` / `_release_abort_idempotency_for` / `_reset_abort_idempotency_for`); plus M11.2 Gate 4 router C20 regression grep
- Regression coverage: none yet — F78 unit test scaffolding could exercise abort-cache discipline but is dev-only
- Plan anchors: A19 (`_preempted_engines` re-promotion forbidden), PLAN.md §3.1 line 112
- Related artifacts: review-report/R01.md (R01-F1), review-report/R12.md (R12-F1)
- Related bugs: BUG-003 (same fix's try/except path)
- Plan impact: PLAN_CLARIFICATION
- Follow-up: PLAN.md §3.1 line 112 wording — permit "abort-idempotency cache lifecycle (init / update / discard) within RolloutManager private surface"

### BUG-002: `set_weight_version` filter diverges between `engine_indices=None` and explicit-list branches

- Status: FIXED_PENDING_VERIFY
- Class: BUG-IMPLEMENTATION
- First observed: 2026-05-04 (R01 review subagent; cross-referenced cross-cutting subagent review)
- Repro / test command: would surface at first runtime expand — `manager.expand_engines → state=loading → set_weight_version(version, engine_indices=None)` would no-op because `state == "active"` filter excludes loading engines
- Environment: Mac-local code review
- Observed behavior: None branch filtered `state == "active" and handle is not None` (line 800); explicit-list branch passed through `_resolve_engine_indices` which uses `is_alive()` (admits disabling/loading/offloaded)
- Expected behavior: same logical engine set must produce same fan-out regardless of branch
- Failure excerpt: `rollout.py:797-803` (pre-fix) — two branches with different predicates
- Root cause: when iter-5 wrote `set_weight_version`, the None path was meant as a convenience (publish to all currently-serving engines) but the explicit-list path serves the runtime-expand contract (publish to alive engines including `loading`). The convention drifted apart silently.
- Root cause confidence: HIGH
- Fix: collapsed both branches to `indices = self._resolve_engine_indices(engine_indices)` so both use is_alive predicate; updated docstring to document the unified semantics + cite F21
- Files changed: `external/miles/miles/ray/rollout.py`
- Verification: pending — planned command: M11.2 Gate 4 (a) runtime-expand smoke + unit test asserting None and explicit-list with same logical set produce same handle list
- Regression coverage: none yet
- Plan anchors: C21 (F21 single version publish per sync); PLAN.md engine table line 4181
- Related artifacts: review-report/R01.md (R01-F2)
- Related bugs: none
- Plan impact: PLAN_CLARIFICATION
- Follow-up: PLAN.md §F21 / engine-table line 4181 — clarify "active engines" wording → "all alive engines with handles"

### BUG-003: `_abort_engines` idempotency cache survives `shrink_engines` failure → retry skips abort

- Status: FIXED_PENDING_VERIFY
- Class: BUG-IMPLEMENTATION
- First observed: 2026-05-04 (R01 review subagent)
- Repro / test command: synthetic `shrink_engines` drain-timeout injection
- Environment: Mac-local code review
- Observed behavior: pre-fix, the cache drop only happened on success path after state flip to `offloaded`. Failures at drain (rollout.py:873), release (877), or VRAM assert (880-887) left indices in `disabling` with abort cache populated. Retry's `_abort_engines` saw empty `new_targets`, skipped abort RPC, drain re-stalled.
- Expected behavior: retry-correctness — failed shrink should leave the system in a recoverable state where re-issuing aborts re-aborts
- Failure excerpt: rollout.py:824-825 + 860 + 873-887 + 889-894 (pre-fix)
- Root cause: cache-drop logic placed at end of success path; failure paths bypassed it
- Root cause confidence: HIGH
- Fix: wrapped Steps 2-5 (abort + drain + release + post-sleep VRAM assert) in try/except that calls `_reset_abort_idempotency_for(indices)` on failure before re-raising; success path moved to call `_release_abort_idempotency_for(indices)` after Step 6
- Files changed: `external/miles/miles/ray/rollout.py`
- Verification: pending — planned command: synthetic drain-timeout test
- Regression coverage: none yet
- Plan anchors: C23 (F23 abort-drain ordering)
- Related artifacts: review-report/R01.md (R01-F3)
- Related bugs: BUG-001 (shared cache helpers)
- Plan impact: NO_PLAN_CHANGE
- Follow-up: none

### BUG-004: `partial_rollout` assert relocated under `rlix_mode` breaks standalone pytest

- Status: FIXED_PENDING_VERIFY
- Class: BUG-REGRESSION
- First observed: 2026-05-04 (R03 review subagent)
- Repro / test command: `pytest tests/fast/rollout/generate_hub/test_multi_turn.py::TestExitConditions::test_partial_rollout_not_supported`
- Environment: Mac-local code review; test fixture does not set `RLIX_CONTROL_PLANE`
- Observed behavior: post-iter-9 the assert was gated on `if rlix_mode:`, so the standalone test fixture never triggered it; `multi_turn.generate` ran to completion with `partial_rollout=True` set
- Expected behavior: pytest expects `AssertionError` matching "Partial rollout is not supported" because `multi_turn.generate` never implemented partial-rollout resume semantics
- Failure excerpt: pre-fix `multi_turn.py:82-85` assert under `if rlix_mode:`; test_multi_turn.py:274-281 expects unconditional raise
- Root cause: iter-9 commit (dee9ab93c) docstring claimed "Standalone is left untouched (legacy partial_rollout configurations remain valid)" — premise was wrong; multi_turn.py never supported partial_rollout in either mode, so the unconditional assert was the long-standing safety guard
- Root cause confidence: HIGH
- Fix: restored unconditional `assert not args.partial_rollout`; updated comment to document that `multi_turn.generate` never implemented partial-rollout resume semantics (no `len(sample.response) > 0` short-circuit like single_turn.py)
- Files changed: `external/miles/miles/rollout/generate_hub/multi_turn.py`
- Verification: pending — planned command: `pytest tests/fast/rollout/generate_hub/test_multi_turn.py::TestExitConditions::test_partial_rollout_not_supported`
- Regression coverage: existing pytest test
- Plan anchors: C29 (F29), C17
- Related artifacts: review-report/R03.md (R03-F1)
- Related bugs: none
- Plan impact: PLAN_CLARIFICATION
- Follow-up: PLAN.md §F29 — clarify partial_rollout forbidden in BOTH modes (not just RLix)

### BUG-005: `destroy_collective_group` only catches HTTP 404; SGLang returns HTTP 400 BAD_REQUEST

- Status: FIXED_PENDING_VERIFY
- Class: BUG-INTEGRATION
- First observed: 2026-05-04 (R05 review subagent)
- Repro / test command: any duplicate destroy on a non-existent group; load-bearing once iter 19/20 wires the actual broadcast path
- Environment: Mac-local code review of pinned SGLang version
- Observed behavior: pre-fix, `destroy_collective_group` caught `HTTPError` only when `status_code == 404`. SGLang's actual handler at `external/sglang/python/sglang/srt/entrypoints/http_server.py:1153-1165` returns HTTP 200 (success) or HTTP 400 BAD_REQUEST (failure including non-existent group). 404 never appears.
- Expected behavior: Anti-regression invariant #3 — duplicate or post-failure destroys must NOT raise (idempotent destroy)
- Failure excerpt: pre-fix `sglang_engine.py:857-862` only handles 404
- Root cause: invariant #3 was specified at API level but the receiver-side wire-handling code paired it with the wrong HTTP status code; SGLang's actual response shape was not consulted at iter-13 implementation time
- Root cause confidence: HIGH (cross-checked against SGLang source)
- Fix: catch HTTPError on status `(400, 404)`; substring-match response body for "group does not exist" / "does not exist" before treating as no-op success; preserve raise on unrelated 400 / 404 errors
- Files changed: `external/miles/miles/backends/sglang_utils/sglang_engine.py`
- Verification: pending — planned command: M11.2 Gate 4 second-pass NCCL teardown smoke (will exercise destroy on non-existent group)
- Regression coverage: none yet
- Plan anchors: A3 (Anti-regression invariant #3)
- Related artifacts: review-report/R05.md (R05-F1)
- Related bugs: none
- Plan impact: NO_PLAN_CHANGE (plan invariant correct; impl drifted)
- Follow-up: optional — add SGLang version pinning note in plan if upstream changes status code

### BUG-006: `asyncio.wait_for` cancellation does not abort inflight Ray RPCs (P2-12)

- Status: FIXED_PENDING_VERIFY
- Class: BUG-RACE-CONDITION
- First observed: 2026-05-04 (R06 review subagent confirms cross-cutting P2-12 unresolved)
- Repro / test command: force contrived timeout on `cache_owner_actor.run_sync_session.remote(plan)`; observe whether second `sync_selected_workers` queues behind still-executing prior method
- Environment: dual-pipeline M11.2 timeout scenario; theoretical until smoke
- Observed behavior: pre-fix, `asyncio.wait_for(_run(), timeout=ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S)` cancelled the local coroutine but Ray actor methods continued running on the cache_owner side, holding `_cache_lock` for the full natural duration
- Expected behavior: F15 atomic-unit + F20 lock discipline — on timeout, inflight remote work should be aborted so cache_owner releases `_cache_lock` before next sync arrives
- Failure excerpt: pre-fix `miles_model_update_service.py:202-204` (wait_for); `grep "ray.cancel" rlix/` returned zero matches
- Root cause: the iter-20 atomic-unit wrapper followed the asyncio convention (wait_for cancels the awaiter) but did not bridge to Ray's separate cancellation API (`ray.cancel(ref, force=True)`)
- Root cause confidence: HIGH (cross-cutting subagent review explicitly traced)
- Fix: thread `inflight_refs: list` through `_run_atomic_unit` + `_build_plan`; capture each `.remote()` ObjectRef before await; on `asyncio.TimeoutError` / `asyncio.CancelledError` call new `_cancel_inflight` helper which fires `ray.cancel(ref, force=True)` on every captured ref
- Files changed: `rlix/pipeline/miles_model_update_service.py`
- Verification: pending — planned command: timeout-injection test on real cache_owner; assert second `sync_selected_workers` does not queue behind cancelled prior call
- Regression coverage: none yet
- Plan anchors: C15 (F15 atomic sync unit), A1 (Anti-regression invariant #1 dynamic NCCL)
- Related artifacts: review-report/R06.md (R06-F1)
- Related bugs: BUG-011 (parallel lock-across-RPC issue in coordinator)
- Plan impact: NO_PLAN_CHANGE
- Follow-up: smoke evidence to convert `FIXED_PENDING_VERIFY` → `RESOLVED CONFIRMED_BY_TEST`

### BUG-007: SharedStorage port-claim retry can burn budget against same port without advancing

- Status: FIXED_PENDING_VERIFY
- Class: BUG-IMPLEMENTATION
- First observed: 2026-05-04 (R06 review subagent)
- Repro / test command: manual probe with `cache_owner.get_free_port` returning duplicate value
- Environment: Mac-local code review
- Observed behavior: pre-fix, when `next_port == master_port` the retry slept 0.05s and `continue`d WITHOUT updating `master_port`, so next iteration claimed same key → guaranteed `try_put=False`, burning a budget slot
- Expected behavior: every retry round should attempt a fresh key
- Failure excerpt: pre-fix `miles_model_update_service.py:319-323` `if next_port == master_port: await asyncio.sleep(0.05); continue`
- Root cause: minor logic bug in iter-19/20 retry loop; advance-on-collision branch wasn't hit when get_free_port returned same port
- Root cause confidence: HIGH
- Fix: drop the `continue`; always assign `master_port = next_port` after the optional 0.05s back-off; keep the retry budget meaningful. Also enriched retry-exhaustion error with `pipeline_id` + last port + attempts (R06-F3 bonus).
- Files changed: `rlix/pipeline/miles_model_update_service.py`
- Verification: pending — planned command: manual port-claim retry probe
- Regression coverage: none yet
- Plan anchors: C26 (F26 master_port != 0)
- Related artifacts: review-report/R06.md (R06-F3, R06-F4)
- Related bugs: none
- Plan impact: NO_PLAN_CHANGE
- Follow-up: none

### BUG-008: `examples/rlix/run_miles_rlix.py` iter-16 stub never wires orchestrator/coordinator/pipeline

- Status: FIXED_PENDING_VERIFY (documentation-only; full wiring deferred to follow-up iter)
- Class: BUG-PLAN-GAP
- First observed: 2026-05-04 (R07 review subagent)
- Repro / test command: `RLIX_CONTROL_PLANE=rlix python -m examples.rlix.run_miles_rlix ...` — script prints stub message and exits without allocating GPUs / creating actors
- Environment: any
- Observed behavior: pre-fix, file ended at `print("[run_miles_rlix] Iter 16 entry stub. RLix coordinator + pipeline wiring lands in iters 21-27. Re-run after those iters land.")` despite F11 plan section requiring full orchestrator + coordinator + pipeline + main loop construction. The plan TLDR did not call out this gap.
- Expected behavior: F11 plan section requires orchestrator allocate / register / admit + MilesCoordinator + create_pipeline_actor + initialize_pipeline + main loop
- Failure excerpt: pre-fix `run_miles_rlix.py:80-83`
- Root cause: scope decision during cozy-plan execution — iter 16 deliberately landed entry surface only, with the assumption that iters 17-27 would close the gap. Iter completed in spirit but the file remained a stub; plan TLDR / status box did not surface the gap to downstream readers.
- Root cause confidence: HIGH
- Fix: added "KNOWN-GAP STUB (R07-F1)" callout to module docstring naming the missing components (orchestrator.request_gpus + MilesCoordinator + coordinator.create_pipeline_actor + pipeline.initialize_pipeline + main loop) and stating that M11.1+M11.2 GPU smoke must wait for the wiring iter. Also passed `args.sglang_config` to `assert_rlix_topology` (closes BUG-009).
- Files changed: `external/miles/examples/rlix/run_miles_rlix.py`
- Verification: pending — true verification requires the wiring iter to land; documentation-fix verification is `grep "KNOWN-GAP STUB" examples/rlix/run_miles_rlix.py` returning the new callout
- Regression coverage: none — file remains a stub
- Plan anchors: C11 (F11 RLix dual-entry), C13 (F13 driver minimalism)
- Related artifacts: review-report/R07.md (R07-F1)
- Related bugs: BUG-009 (entry-path C9 silent skip)
- Plan impact: PLAN_CLARIFICATION
- Follow-up: PLAN.md M11.1 status box / TLDR — distinguish "iter 16 entry surface only" vs "wiring-iter end-to-end driver"

### BUG-009: C9 PD-disaggregation guard silently skipped at entry path

- Status: FIXED_PENDING_VERIFY
- Class: BUG-IMPLEMENTATION
- First observed: 2026-05-04 (R08 review subagent)
- Repro / test command: unit test where `args.sglang_config.has_pd_disaggregation == True` invokes `run_miles_rlix.main()` and asserts RuntimeError("C9: ...") raises
- Environment: Mac-local code review
- Observed behavior: pre-fix, `examples/rlix/run_miles_rlix.py:68` called `assert_rlix_topology(args, sglang_config=None)`; validator at `rlix_validation.py:247` short-circuited on `sglang_config is not None`. C9 was a silent no-op for the only wired RLix entry today.
- Expected behavior: C9 must fire on RLix-mode entry whenever PD disaggregation is configured (out-of-scope per F10 / scope §6)
- Failure excerpt: pre-fix `run_miles_rlix.py:68` and `rlix_validation.py:247`
- Root cause: F10 validator was specified to take an optional `sglang_config` for callers that already had a config object; iter-16 driver was specified BEFORE the M11.1 entry's sglang_config plumbing was settled, so the entry passed None. Cross-iter dependency surfaced post-implementation.
- Root cause confidence: HIGH
- Fix: validator falls back to `args.sglang_config` (set by `arguments.py:2207-2213`) when explicit kwarg is None; entry passes `getattr(args, "sglang_config", None)` explicitly as belt-and-suspenders. Both layers ensure C9 fires.
- Files changed: `external/miles/miles/utils/rlix_validation.py`, `external/miles/examples/rlix/run_miles_rlix.py`
- Verification: pending — planned command: unit test with PD-disaggregation-enabled SglangConfig invokes entry path; assert RuntimeError raises
- Regression coverage: none yet
- Plan anchors: C9 (F10 startup validation)
- Related artifacts: review-report/R08.md (R08-F1)
- Related bugs: BUG-008 (same entry file)
- Plan impact: NO_PLAN_CHANGE
- Follow-up: smoke evidence

### BUG-010: `WorkerPlacement.gpu_ids` stores global GPU ids despite docstring/plan declaring node-local

- Status: FIXED_PENDING_VERIFY
- Class: BUG-INTEGRATION
- First observed: 2026-05-04 (R09 review subagent)
- Repro / test command: 2-node mock test constructing `MilesPlacementProvider` with `num_gpus_per_node=2`, `infer_device_mapping=[0,1,2,3]`; assert `placements[1].gpu_ids == (0,1)` and `placements[1].node_rank == 1`
- Environment: Mac-local code review (multi-node scenario unobservable on single-node dev gate)
- Observed behavior: pre-fix, provider derived `node_rank = slice_gpu_ids[0] // num_gpus_per_node` (treating ids as global), then stored same global slice into `WorkerPlacement.gpu_ids` despite docstring lines 13-15 / 47-49 declaring "tuple of node-local GPU ids". CVD construction at `actor_group.py:141` `cvd = ",".join(str(g) for g in wp.gpu_ids)` would write "4,5,6,7" on node-1 Ray worker whose CUDA driver only enumerates 0..3 → CUDA init failure.
- Expected behavior: PLAN.md §F12 (a) line 3502-3520 declares "Multi-node-compatible structural invariant (M11.1 Cut 1') ... 节点本地 gpu_ids ... 实现不能把多机跑通堵死"
- Failure excerpt: pre-fix `placement_provider.py:174,186,247` store global ids; `actor_group.py:141` writes verbatim into CVD
- Root cause: when iter-14 wrote the placement provider, the `node_rank` derivation correctly used global ids (you cannot derive node_rank from a node-local id without keeping the global mapping somewhere), but the storage step used the same un-translated tuple. The docstring/plan invariant was specified but not enforced by the construction.
- Root cause confidence: HIGH (single-node dev gate happens to work because global == node-local when num_total_gpus == num_gpus_per_node, masking the bug)
- Fix: in `get_all_rollout_engine_placements`, compute `node_rank` from the global slice's first id, verify all ids in slice share the same node_rank (multi-node-spanning slice rejected fail-fast), then store `tuple(g % num_gpus_per_node for g in global_slice)` as `WorkerPlacement.gpu_ids`. Same change in `get_train_workers`. `assert_structural` updated to compare against node-local projection of the declared global slice; also asserts `node_rank` matches expected.
- Files changed: `external/miles/miles/ray/placement_provider.py`
- Verification: pending — planned command: 2-node mock test (see Repro)
- Regression coverage: F78 dev-only scaffolding doesn't cover this; M11.5 follow-up should add a multi-node fixture
- Plan anchors: F12 (a) line 3502, C33 (F33 base_gpu_id=0)
- Related artifacts: review-report/R09.md (R09-F1)
- Related bugs: none
- Plan impact: NO_PLAN_CHANGE (plan invariant correct; impl drifted)
- Follow-up: M11.5 follow-up — add a multi-node mock fixture to dev-only test scaffolding

### BUG-011: `_resize_sync_lock` held across full Ray RPC chain in `_expand_workers` (P2-9)

- Status: FIXED_PENDING_VERIFY
- Class: BUG-RACE-CONDITION
- First observed: 2026-05-04 (R10 review subagent confirms cross-cutting P2-9 unresolved)
- Repro / test command: concurrent `report_progress_from_scheduler` while `_expand_workers` is running its 4-RPC chain; observe whether progress reports queue
- Environment: dual-pipeline M11.2 contention scenario; theoretical until smoke
- Observed behavior: pre-fix, `resize_infer` took `_resize_sync_lock`, called `_expand_workers` which executed 4 sequential `ray.get(...)` (`get_engine_states`, `expand_engines`, `service.sync_selected_workers`, `activate_routing`) all under the held lock. `_shrink_workers` likewise. Lock could be held for tens of seconds.
- Expected behavior: F20 / Anti-regression spirit — `threading.Lock` should not be held across multi-step Ray RPC chains. Plan's `sync_base_weights_to_active` (`miles_coordinator.py:374-396`) shows the release-before-RPC pattern.
- Failure excerpt: pre-fix `miles_coordinator.py:409-416` (resize_infer lock scope) + `425-469` (_expand_workers body, 4 RPCs under lock)
- Root cause: PLAN.md §coordinator call-graph contradicts itself between PLAN.md:2384-2412 (release-before-RPC) and PLAN.md:2415-2453 (chain-under-lock). The code followed the latter prescriptive snippet. No deadlock today (call-graph acyclic) but progress reports / get_active_engines / publish_cache_ready_step all blocked on the lock.
- Root cause confidence: HIGH
- Fix: refactored `_expand_workers` to snapshot-under-lock / RPC-without-lock / commit-under-lock pattern matching `sync_base_weights_to_active`. Phase 1 acquires lock to capture `cached_step` + `service` handle; Phase 2 issues 4 RPCs WITHOUT lock; Phase 3 re-acquires lock to commit `_active_engine_indices |= engine_indices`. Same pattern applied to `_shrink_workers`.
- Files changed: `rlix/pipeline/miles_coordinator.py`
- Verification: pending — planned command: concurrent progress-report-during-resize_infer test
- Regression coverage: none yet
- Plan anchors: C20 (F20 lock discipline), A20 (Anti-regression invariant #20)
- Related artifacts: review-report/R10.md (R10-F1)
- Related bugs: BUG-006 (parallel lock-across-RPC issue in MILES `run_sync_session`)
- Plan impact: PLAN_CLARIFICATION
- Follow-up: PLAN.md §coordinator call-graph — reconcile inconsistency between lines 2384-2412 (release-before-RPC) and 2415-2453 (chain-under-lock). Adopt release-before-RPC as canonical.

### BUG-012: `_actor_train_allocated` flipped to False before scheduler release confirmed → ledger desync

- Status: FIXED_PENDING_VERIFY
- Class: BUG-IMPLEMENTATION
- First observed: 2026-05-04 (R11 review subagent)
- Repro / test command: fault-injection unit test — raise inside `scheduler.notify_release_gpus` during `_init_phase_a_train`; assert `shutdown_hard` re-attempts release
- Environment: transient scheduler RPC failure during init; theoretical until smoke
- Observed behavior: pre-fix, `_notify_release_cluster_gpus` swallowed exceptions (logged WARN, returned). Caller then unconditionally set `self._actor_train_allocated = False`. If release RPC raised, the scheduler kept the allocation pinned but the client believed it was released; subsequent `shutdown_hard` skipped the cluster_id (line 360 `if not was_allocated: continue`).
- Expected behavior: F36 / Anti-regression invariant #10 — cleanup ledger flag MUST be False if and only if scheduler is known to have released the cluster_id allocation
- Failure excerpt: pre-fix `miles_pipeline.py:200-203` (release-then-flag-flip) + `417-434` (helper swallows exception)
- Root cause: post-iter-30 b241af4 commit moved the actor_train release earlier in init (P1-7 fix to avoid scheduler deadlock when actor_infer was requested), but the helper's swallow-on-error semantics (originally appropriate for shutdown_hard's best-effort cleanup path) became incorrect for the success-path-init usage. The two callers needed different error semantics.
- Root cause confidence: HIGH
- Fix: `_notify_release_cluster_gpus` now returns `bool` indicating whether the scheduler RPC completed successfully; phase A and `_after_training` both flip `_actor_train_allocated = False` only when `released == True`. The helper still logs WARN on failure (preserving observability) but the caller sees the failure signal.
- Files changed: `rlix/pipeline/miles_pipeline.py`
- Verification: pending — planned command: fault-injection unit test
- Regression coverage: none yet
- Plan anchors: A10 (Anti-regression invariant #10 minimal cleanup), C36 (F36)
- Related artifacts: review-report/R11.md (R11-F1)
- Related bugs: BUG-014 (same file)
- Plan impact: NO_PLAN_CHANGE
- Follow-up: smoke evidence

### BUG-013: `_init_phase_b_infer` self-allocates fresh PG via `_create_placement_group`, double-counting against scheduler grant

- Status: FIXED_PENDING_VERIFY (M11.1 single-pipeline contract; M11.2 dual-pipeline behavior deferred)
- Class: BUG-PLAN-GAP
- First observed: 2026-05-04 (R11 review subagent; b241af4 commit message acknowledges as P1-2 pragmatic fix)
- Repro / test command: M11.2 dual-pipeline GPU-pool oversubscription smoke
- Environment: dual-pipeline M11.2 contention scenario
- Observed behavior: pre-fix, after `request_gpus(actor_infer_cluster_id, INITIALIZATION)` returned the rlix-scheduled GPU ids, the pipeline called `_create_placement_group(num_gpus)` (`external/miles/miles/ray/placement_group.py:44`) which invokes `ray.util.placement_group(...)` against Ray's free pool, independent of the rlix scheduler ledger. `MilesPlacementProvider` was constructed in phase A but never queried for the infer PG.
- Expected behavior: F35 startup contract — GPU resources flow through rlix scheduler's ledger so multi-pipeline sharing is safe
- Failure excerpt: pre-fix `miles_pipeline.py:225-230` (direct `_create_placement_group` call); `placement_provider.py:149-190` (`get_all_rollout_engine_placements` is the rlix-aware allocator) was unused for infer pool
- Root cause: b241af4 P1-2 pragmatic fix abandoned the all-shell M11.2 init pattern in favor of using legacy `pg=` shape so RolloutManager could come up active via standard `start_rollout_servers`. The legacy shape `(pg, reordered_bundle_indices, reordered_gpu_ids)` was easier to mint via `_create_placement_group` than to translate from `WorkerPlacement` list. The deferral was acknowledged in the commit message but not surfaced in plan TLDR.
- Root cause confidence: HIGH
- Fix: chose option (1) cheap-fix per finding — added explicit comment naming the deferred work + fail-fast assert that scheduler grant covers full infer pool. M11.1 single-pipeline contract: scheduler grant exhausts the pool, so Ray's PG and rlix ledger reference same physical GPUs (benign). Multi-pipeline (M11.2): partial grant would now raise RuntimeError before bypass, blocking silent oversubscription.
- Files changed: `rlix/pipeline/miles_pipeline.py`
- Verification: pending — M11.2 dual-pipeline smoke when wiring iter lands
- Regression coverage: fail-fast assert provides upper bound; no positive coverage yet
- Plan anchors: C35 (F35 startup), C22 (F22 RELAXED), A22 (Anti-regression #22 implicit — no oversubscription)
- Related artifacts: review-report/R11.md (R11-F2); commit b241af4 message
- Related bugs: BUG-008 (same M11.1→M11.2 deferred work surface)
- Plan impact: PLAN_SCOPE_CHANGE
- Follow-up: PLAN.md M11.1→M11.2 follow-up section + plan TLDR — add explicit "infer-pool placement provider routing" deferred-work entry

### BUG-014: outer `ray.get(rollout_manager.shutdown_hard.remote())` has no timeout

- Status: FIXED_PENDING_VERIFY
- Class: BUG-IMPLEMENTATION
- First observed: 2026-05-04 (R11 review subagent)
- Repro / test command: manual `ray.get` hang injection probe — make `rollout_manager.shutdown_hard` block indefinitely; assert `MilesPipeline.shutdown_hard` proceeds to `ray.kill` after bounded wait
- Environment: degraded manager scenario
- Observed behavior: pre-fix, `miles_pipeline.py:326` called `ray.get(self._rollout_manager.shutdown_hard.remote())` with no timeout. If the manager actor was stuck (alive but wedged inside a foreign call), the outer ray.get would block forever; the subsequent `ray.kill` and `notify_release_gpus` would never run.
- Expected behavior: F36 / Anti-regression invariant #10 — bounded wait on every cleanup RPC; raise on timeout. Plan canonical at PLAN.md:1593 wraps the equivalent call in `ray.get(self.actor_infer.shutdown_hard.remote(), timeout=10)`.
- Failure excerpt: pre-fix `miles_pipeline.py:325-328`
- Root cause: outer ray.get inherited the unbounded shape from earlier iter-27 draft; inner per-engine ray.get already had a 10s timeout (`rollout.py:1013`) but the outer wrapper was overlooked.
- Root cause confidence: HIGH
- Fix: added `timeout=30.0` to outer `ray.get`; on timeout, log + proceed to `ray.kill` (existing try/except already handles the TimeoutError path)
- Files changed: `rlix/pipeline/miles_pipeline.py`
- Verification: pending — planned command: manual ray.get-hang injection probe
- Regression coverage: none yet
- Plan anchors: A10 (Anti-regression invariant #10), C36 (F36)
- Related artifacts: review-report/R11.md (R11-F3)
- Related bugs: BUG-012 (same file's cleanup discipline)
- Plan impact: NO_PLAN_CHANGE
- Follow-up: smoke evidence

## 4. Retest Timeline

| Run | Command | Result | Related bugs | Notes |
|---|---|---|---|---|
| RUN-001 | `python3 -m py_compile <9 touched files>` | PASS | BUG-001..BUG-014 | Mac-local syntax-only validation; all 9 files (rollout.py, placement_provider.py, multi_turn.py, sglang_engine.py, rlix_validation.py, run_miles_rlix.py, miles_pipeline.py, miles_coordinator.py, miles_model_update_service.py) parse cleanly. Smoke verification deferred — Mac-local environment cannot exercise GPU paths. |

## 5. Plan Revision Suggestions

| Revision ID | Status | Triggered by bug | Source-plan gap | Suggested source-plan location | Rerun needed |
|---|---|---|---|---|---|
| REV-001 | APPLIED | BUG-001 | §3.1 line 112 wording too literal — permits only `_abort_engines` to mutate cache; spirit-of-rule allows `RolloutManager` private surface (init/update/discard) | PLAN.md §3.1 line 112 (Layer 1 Permanently Forbidden, A19); also patched in tldr.md (3 spots) + scope.md F03 wording | tldr-plan (deferred — sibling artifacts patched in place; full regeneration deferred) |
| REV-002 | APPLIED | BUG-002 | §F21 / engine-table line 4181 wording "active engines" reads imprecisely; intent is "all alive engines with handles (admits loading during runtime expand)" | PLAN.md §F21 / engine table line 4181 | tldr-plan (deferred) |
| REV-003 | APPLIED | BUG-004 | §F29 partial_rollout invariant should explicitly state forbidden in BOTH modes (not only RLix) — `multi_turn.generate` never implemented partial-rollout resume semantics | PLAN.md §F29 (partial_rollout discipline); also patched in tldr.md row for multi_turn.py | tldr-plan (deferred) |
| REV-004 | APPLIED | BUG-008 | M11.1 status box / TLDR does not distinguish "iter 16 entry surface only" from "wiring-iter end-to-end driver"; downstream readers may mistake stub presence for working entry | PLAN.md M11.1 in-scope section ("Driver wiring status" callout); also patched in tldr.md row for run_miles_rlix.py | tldr-plan (deferred) |
| REV-005 | APPLIED | BUG-011 | Coordinator call-graph contradicts itself between release-before-RPC (lines 2384-2412) and chain-under-lock (lines 2415-2453); pick one canonical pattern | PLAN.md §coordinator call-graph (both blocks updated to release-before-RPC); scope.md F20 row updated; tldr.md unchanged (already release-before-RPC compatible) | both (deferred) |
| REV-006 | APPLIED | BUG-013 | M11.1→M11.2 deferred-work list is missing "infer-pool placement provider routing" — the b241af4 P1-2 pragmatic fix bypasses the rlix scheduler ledger and the gap is invisible at plan TLDR level | PLAN.md Implementation follow-up table (new row); tldr.md Out-of-scope-now bullet; scope.md F95.1 row | both (deferred) |

## 6. Remaining Risks

| Risk | Evidence | Suggested next action |
|---|---|---|
| All 14 BUG-* are `FIXED_PENDING_VERIFY` — verification depends on a GPU smoke run that is explicitly out of scope per cozy plan §Verification | RUN-001 PASS confirms only Mac-local `py_compile`; M11.1 Gate 1/2/2.5/3 + M11.2 Gate 4 (a)–(f) runtime evidence is deferred | Schedule a target Linux/GPU env run; convert each `FIXED_PENDING_VERIFY` → `RESOLVED CONFIRMED_BY_TEST` once smoke evidence lands |
| Multi-node CVD breakage (BUG-010) cannot be exercised on single-node dev gate; latent until first multi-node smoke | R09-G1 evidence gap — no 2-node mock test in F78 dev-only scaffolding | Add 2-node mock fixture to dev-only test scaffolding (F78 follow-up); covers `MilesPlacementProvider` node-local invariant + CVD construction |
| P2-12 (BUG-006) and P2-9 (BUG-011) lock-across-RPC fixes are theoretically correct but observable only under timeout / contention scenarios that single-pipeline M11.1 smoke does not exercise | R06-G1 + R10-G1 evidence gaps | Dual-pipeline M11.2 timeout-injection + concurrent-progress-report tests |
| Iter-16 entry stub (BUG-008) means `python -m examples.rlix.run_miles_rlix` is non-functional even with all 14 fixes applied; the M11.1+M11.2 GPU smoke cannot run from this driver until the wiring iter lands | R07-F1, R07-G1 | Schedule a wiring iter that constructs orchestrator + MilesCoordinator + create_pipeline_actor + initialize_pipeline + main loop, then re-run all 14 verifications |
| `_init_phase_b_infer` self-allocated PG (BUG-013) is M11.1-safe but blocks M11.2 dual-pipeline; the fail-fast assert prevents silent mis-routing but does not enable dual-pipeline | R11-G2 | Route infer PG through `self._placement_provider.get_all_rollout_engine_placements()` once the WorkerPlacement → legacy `pg` tuple translation lands (M11.2 follow-up) |
| Per-iter codex review backlog: iters 21–30 land WITHOUT codex review (codex daily quota cap during execution). Cross-cutting subagent review caught 8 P1 + 5 P2 issues; 8 P1 + 2 P2 fixed earlier; this session closed the remaining HIGH-severity gaps but residual codex findings on those iters may surface when quota resets | review.md §0 codex backlog warning | Re-run `codex review --commit <SHA>` on iters 21–30 commits when daily quota resets; aggregate findings into a follow-up debug-log session |
| TLDR / scope artifacts were patched in place rather than regenerated by `tldr-plan` / `scope-triage` (per user instruction "leave the tldr and scope stale for now"); REV-001..006 are marked APPLIED to PLAN.md + reflected in tldr/scope inline edits, but the artifacts are not a clean projection of the post-patch plan. Subsequent debug-log entries that touch the same anchors may carry forward stale phrasing until the skills are rerun | REV-001..006 Status: APPLIED rows; sibling artifact mtimes do not reflect a fresh regeneration | Schedule a `tldr-plan` + `scope-triage` rerun before the next milestone gate, so the audit / boundary projections align with the patched plan |
