# Scope Triage

Feature: **MILES → RLix port (fullasync GRPO under partial overlap)**
Source plan: [miles-port-unified-plan.md](miles-port-unified-plan.md)
TLDR context: [miles-port-unified-plan.tldr.md](miles-port-unified-plan.tldr.md) (current — same source revision)
Scope review state: `SCOPE_REVIEW_READY_WITH_DEFAULTS`

## 0. Executive Summary

- MVP boundary: M11.1 single-pipeline (4-GPU tp=2 partial-overlap fullasync GRPO under RLix; cpu_serialize colocate + NCCL broadcast non-colocate; turn-level redispatch; F10 fail-fasts) + M11.2 happy path (dual-pipeline shared PG; full INIT+offload; runtime base v=-1 expand sync; donor-shrink ordering; router 0-active MVP unbounded suspend).
- P0 risks: D3.D atomic sync unit (transport+finalize+set_weight_version under one timeout); A19 `_preempted_engines` re-promotion to routing/dispatch; C20 router `_use_url` raising `RuntimeError` instead of suspending.
- Main correctness floor: Init Step 6.6 pre-registration (manager+service resources+`_cache_ready_step=-1`+`_active_engines_bootstrapped=True`) MUST land BEFORE Step 7 INIT; A5 says scheduler runs Phase-5 `resize_infer(add)` BEFORE Phase-6 waiter signal.
- Main overengineering risk: pulling A1 (bounded `_use_url` timeout + 503 sentinel + client `EnginePreemptedError` translation) from M11.5 forward into M11.2 — forks the same machinery twice.
- Blocking questions: None. Plan's §3.1 8-Layer taxonomy maps cleanly onto scope-triage categories; no MVP-affecting deltas surfaced.
- Scope delta divergent rows: 5 non-`ALIGNED` (F66 / F76 / F107 / F108 = `MISSING_SOURCE_SCOPE`; F77 = `CATEGORY_REFRAMED`); all `MVP set impact: no` → §11.

## 1. Blocking Questions

> None. Non-blocking deltas and low-confidence defaults are listed in §11 (three populating sources: `Confidence: Low` rows, non-`ALIGNED` rows with `MVP set impact: no`, and `CATEGORY_REFRAMED` rows).

## 2. Scope Delta Matrix

Plan has 108 work items > 20-row threshold. Using the divergent-rows + aligned-summary split. Canonical 9-column format.

### Divergent rows (require human review)

`Delta != ALIGNED` rows only (per hard rule).

| ID | Feature / Item | Source plan scope | TLDR mirrored scope | Triage classification | Delta | MVP set impact | Review action | Anchors |
|---|---|---|---|---|---|---|---|---|
| F66 | tmpfs file naming convention `miles_cpu_bucket_{uuid}.pt` for grep-friendly leak detection | `SOURCE-UNKNOWN` (described in §F4 §B but not classified in §3.1) | TLDR Appendix D | P1-OBSERVABILITY-FLOOR | `MISSING_SOURCE_SCOPE` | no | Non-blocking (NQ11.1) — patch source plan §3.1 to enumerate, OR accept implicit Layer 7 hygiene | §F4 §B Case B tmpfs lifecycle |
| F76 | `GET /admission_state` test diagnostic endpoint, gated by `MILES_ROUTER_TEST_HOOKS=1` | `SOURCE-UNKNOWN` (introduced in §Gate 4 (f) but not enumerated in §3.1 Layer 8 alongside `MILES_DEBUG_VERIFY_MODEL=1`) | TLDR §6.1 E14 | DEV-ONLY | `MISSING_SOURCE_SCOPE` | no | Non-blocking (NQ11.2) — patch source plan §3.1 Layer 8 | source plan §Gate 4 (f) Step P1 |
| F77 | B1 test-side `asyncio.wait_for(test_coro, timeout=N)` wrapper for Gate 4 (f) sub-tests (60s positive / 30s counter); production router stays unbounded | `SOURCE-MVP` (§3.1 Layer 7 ✅ B1, "MVP Lightweight Defensive — active in M11.1/M11.2") | TLDR §F.5: "test-side bound only; production stays unbounded" | DEV-ONLY (test-only wrapper; plan explicitly says "not a production safeguard") | `CATEGORY_REFRAMED` | no | Non-blocking (NQ11.3) | §3.1 Layer 7 ✅ B1; §Gate 4 (f) |
| F107 | X2 `MilesCoordinator.register_model_update_resources(*, cache_owner_actor, rollout_manager)` ctor handle injection (M2 P0-1) | `SOURCE-UNKNOWN` (§3.1 Special "Boundary kept" — explicitly NOT classified Layer 1-8) | TLDR Appendix D row | P0-CORRECTNESS-FLOOR (architectural ownership boundary; `ray.get_actor` lookup in service body is Layer 1 Forbidden — F05) | `MISSING_SOURCE_SCOPE` | no | Non-blocking (NQ11.4) — accept plan's Special-class treatment | §3.1 Special X2 |
| F108 | C1 `MilesPipeline` actor `max_concurrency` (tentatively `=2`); §Audit Checkpoints defer decision to execution phase | `SOURCE-UNKNOWN` (§3.1 Special "Evidence-based pending"; not Layer 1-8) | TLDR §7 references "M8 actor concurrency" checkboxes | P1-LIGHTWEIGHT-DEFENSIVE (with explicit evidence trigger; default keeps `=2`) | `MISSING_SOURCE_SCOPE` | no | Non-blocking (NQ11.5) — accept plan's evidence-deferred treatment | §3.1 Special C1; §Audit Checkpoints |

### Aligned but noteworthy

| ID | Item | Why notable | Anchors |
|---|---|---|---|
| F39 | M11.2 router 0-active MVP unbounded suspend (`_use_url` async + `asyncio.Condition` + endpoint `notify_all` + load increment INSIDE lock) | Carries the M11.2 happy path; bounded timeout / 503 sentinel / client translation are explicitly deferred to M11.5 (F79 A1). Reviewer should confirm "manual `ray stop` is acceptable recovery in M11.2 MVP" before agent handoff. | C20; AC13; D3.G; D4.B; E14; §3.1 Layer 3 A1 (deferred half) |
| F22 | Init bootstrap Step 6.6 pre-registration BEFORE Step 7 INIT request | Hardest-to-verify ordering invariant in the plan; depends on A5 (scheduler Phase-5-before-Phase-6). Plan describes 6 sub-steps (6.5/6.6a-f) — all must land. | A5; D3.E; AC10/AC11/AC12; E12/E13 |

### Aligned summary

102 of 108 items map directly onto plan's §3.1 8-Layer taxonomy and need no per-row review:

- **`ALIGNED` Layer 1 → P0-FORBIDDEN** (14 items, F01–F14): A6 fatal flag, A11 router queue, A19 `_preempted_engines` re-promotion, split RPCs, finalize outside service, lock-violating counter, `RuntimeError` from `_use_url`, post-import CVD mutation, `PipelineCoordinator` subclass, device-mapping CLI args, per-request tracking, `flattened_bucket` cpu_serialize, driver `ray.shutdown`, async admission helpers.
- **`ALIGNED` Layer 6 → P0-CORRECTNESS-FLOOR** (27 items, F15–F41): atomic sync unit, base v=-1, F4 sequence, cache_owner uniqueness, active-set bootstrap, locks, NCCL TCP rendezvous, master_port != 0, writer_lock, tmpfs lifecycle, turn redispatch MAX, `EnginePreemptedError` + `RLixRouterMetadataError`, router metadata injection, base_gpu_id=0, LOCAL_RANK=0, F10 startup asserts C1–C23, M4 minimal cleanup, M11.2 Pipeline B full INIT, donor-shrink T1<T2<T3, router 0-active MVP suspend, base v=-1 runtime expand, DO_TIME_SHARING dual entry.
- **`ALIGNED` P1-HAPPY-PATH** (23 items, F42–F64): F1 sleep/wake tag-based, F2 shrink/expand/activate, F3 router admission API + 4-dict lifecycle, multi_turn snapshot/restore, fully_async `_FatalError`, `build_cpu_bucket_cache`, cpu_serialize transport + 4 SGLang patches, NCCL broadcast path, `run_sync_session` composite RPC, `SyncSessionPlan`, dual-mask receivers, `sync_base_weights_to_active`, `_expand_workers`, namespace isolation, registration lifecycle, progress reporting, RLixHooks protocol, MilesPlacementProvider, RayTrainGroup `worker_placements`, all-shell ctor, `abort_all_requests`, `is_idle()`.
- **`ALIGNED` P1-OBSERVABILITY-FLOOR** (2 items): F65 (post-sleep VRAM SGLang `/server_info`), F67 (instrumented atomic-unit call counters).
- **`ALIGNED` Layer 7 → P1-LIGHTWEIGHT-DEFENSIVE** (7 items): F68 (A16-cheap `dead_workers.discard` + failure_count reset), F69 (X3 `_active_engines_bootstrapped` flag, cross-ref F19), F70 (B3 `master_port` claim path), F71 (standalone partial-overlap fail-fast), F72 (header hardening), F73 (`/dev/shm` capacity check), F74 (F12 round-trip structural validation). Plus F106 X1 `mode/adapter_id` nullable forward-compat.
- **`ALIGNED` DEV-ONLY** (2 items): F75 (`MILES_DEBUG_VERIFY_MODEL=1` validation gated, plan Layer 8), F78 (unit/integration tests).
- **`ALIGNED` Layer 3 → P3-DEFER-HARDENING M11.5** (15 items, F79–F93): A1 bounded timeout/sentinel/client translation, A2 admission_epoch race, A3 multi-pipeline cleanup, A4 receiver crash tolerance, A5 NCCL port cooldown, A10 SGLang ingress 503/5xx synthesis, A12 plasma zero-copy, A13 async_save, A14 train-side VRAM assert, A15 engine-overlap assert, A16-recovery dead-resurrect, A17 except broadening, graceful drain, M11.6 cuda_ipc, M11.4 LoRA.
- **`ALIGNED` Layer 4 → P3-DEFER-HARDENING trigger-bound** (2 items): F94 (A7 round-trip identity), F95 (A18 non-contiguous mapping).
- **`ALIGNED` out-of-scope → NO-OVERENGINEERING** (10 items, F96–F105): PD disaggregation, vLLM, MoE/EP, sglang_dp>1, multi-LoRA/DPO/SFT, request-level migration, cross-node single rollout engine, `RadixTreeMiddleware` in RLix mode, subclass `PipelineCoordinator` (cross-ref F09 P0), branching inside `train_async.py`.

Total ALIGNED: 102. Divergent: 5. Special note (ALIGNED but noteworthy): 2.

## 3. Category Table

Plan has 108 extracted items (> 50). Using **large-plan display strategy**: visible table holds (a) all P0, (b) all non-ALIGNED, (c) all `Confidence: Low/medium`, (d) all `Action != implement`. Routine ALIGNED P1-* / DEV-ONLY-test-scaffold rows with `Confidence: high` are summarized after the table; full table is in Appendix I.

| ID | Item | Category | Priority | Action | Anchors | Confidence | Reason |
|---|---|---|---|---|---|---|---|
| F01 | A6 — `_fatal_error` flag dual-path in fully_async | P0-FORBIDDEN | M11.1 guard | assert / not implement | §3.1 Layer 1 A6; Anti-regression invariant #9 | high | queue↔flag dual-source = state-divergence bug surface; queue `_FatalError` sentinel single-path is the final design |
| F02 | A11 — Router internal request queue / retry / spillover / synthetic fallback | P0-FORBIDDEN | M11.2 guard | reject in code review | §3.1 Layer 1 A11; C20 | high | C20 model = `asyncio.Condition` suspend ONLY |
| F03 | A19 — Promoting `_preempted_engines` to routing / dispatch / attribution / resize-safety state | P0-FORBIDDEN | M11.1 guard | assert / not implement | §3.1 Layer 1 A19 | high | responsibility permanently re-scoped to `router.enabled_workers` |
| F04 | Splitting `cache_owner.run_sync_session(plan)` into multiple top-level Ray RPCs | P0-FORBIDDEN | M11.1 guard | not implement; single composite RPC | §3.1 Layer 1; D3.D; D4.E | high | `_cache_lock` cannot span Ray RPCs |
| F05 | Calling `service.sync_selected_workers` / `finalize_weight_update` / `manager.set_weight_version` outside the documented atomic-unit / bootstrap entry points | P0-FORBIDDEN | M11.1 guard | code review | §3.1 Layer 1; D3.D | high | atomic-unit invariant is service-owned; pipeline/coordinator direct calls fragment timeout boundary |
| F06 | `worker_request_counts[url] += 1` outside the `asyncio.Condition` lock | P0-FORBIDDEN | M11.2 guard | code review | §3.1 Layer 1; C20; D3.G | high | concurrent suspend/resume races with `_finish_url` decrement → negative-count assert |
| F07 | Raising `RuntimeError("no enabled live workers")` from `_use_url` | P0-FORBIDDEN | M11.2 guard | replace with suspend | §3.1 Layer 1; C20 | high | fully_async sees generic abort → silent group-recycle attribution |
| F08 | Mutating `os.environ['CUDA_VISIBLE_DEVICES']` after `import torch` / `import sglang` | P0-FORBIDDEN | M11.1 guard | code review | §3.1 Layer 1; M6 | high | `cuInit()` lock timing — env mutation after import is silently ineffective |
| F09 | Subclassing `PipelineCoordinator` for `MilesCoordinator` | P0-FORBIDDEN | M11.1 guard | manual init only | §3.1 Layer 1; D1.A | high | `super().__init__()` triggers ROLL config validators on MILES args lacking those fields |
| F10 | Adding device-mapping CLI args | P0-FORBIDDEN | M11.1 guard | derive only | §3.1 Layer 1 | high | mapping must derive from existing args; new args drift from RLix declared mapping |
| F11 | Re-introducing per-request tracking (`_inflight_requests`) or worker-side `is_engine_resident_on_gpu` | P0-FORBIDDEN | M11.1 guard | use `/v1/loads` idle bool | §3.1 Layer 1 | high | source of truth = `EngineInfo.state` + `router.enabled_workers` |
| F12 | Re-using `load_format="flattened_bucket"` for `cpu_serialize` | P0-FORBIDDEN | M11.1 guard | use ROLL wire format | §3.1 Layer 1; D5.A | high | incompatible with ROLL CPU bucket schema |
| F13 | Driver-level `ray.shutdown()` or top-level `try/except` in `run_miles_rlix.py` | P0-FORBIDDEN | M11.1 guard | not implement | §3.1 Layer 1; F12 (c) | high | `ray stop` CLI is the recovery surface |
| F14 | Making sync admission helpers (`_*_internal`) `async def` | P0-FORBIDDEN | M11.2 guard | helpers stay sync | §3.1 Layer 1; F3 §(b.4) | high | only async endpoints + `_health_check_loop` carry `notify_all` |
| F15 | D3.D atomic sync unit `service.sync_selected_workers(sync_id, targets, version)` covering (a)+(b)+(c) under one `ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S` | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D3.D, AC1/AC4/AC11, E2/E5/E6 | high | splitting reintroduces lock-across-RPC; partial timeout boundaries leave receiver inconsistent |
| F16 | C13 base `version=-1` sync = full transport + finalize + `set_weight_version(-1)` from init-built CPU bucket; NO checkpoint shortcut | P0-CORRECTNESS-FLOOR | M11.2 P1 | implement | D5.C, AC11, C13, E5 | high | label-only path serves stale rollout weights under resume / non-equivalent `args.load` |
| F17 | D3.F F4 sequence: build CPU cache → offload → destroy NCCL → sync (in this order) | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D3.F, AC4/AC5, E9 | high | reordering deadlocks gather (build after offload) or serves stale (sync before offload) |
| F18 | C15 cache_owner uniqueness: `collect_cache_owner_roles` returns exactly one True (= pp0+dp0+tp0+cp0) at init Step 6.5 | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | A6, C15, D2.B | high | multiple owners → duplicate writes; zero owners → empty cache |
| F19 | Active-set bootstrap with `_active_engines_bootstrapped: bool` flag (X3) — distinguishes "first call with empty set" from "second call with empty set" | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | §3.1 Layer 7 ✅ X3; D3.E; AC10 | high | empty set bootstrap is legit at M11.2 init; without flag, repeat bootstrap silently overwrites |
| F20 | `_resize_sync_lock` (coordinator) — release-before-RPC pattern: `_cache_ready_step` write + state snapshot inside the lock, then ALL Ray RPCs (`get_engine_states`, `expand_engines`, `service.sync_selected_workers`, `activate_routing`, `shrink_engines`) execute OUTSIDE the lock, and `_active_engine_indices` mutations commit back inside the lock; `_cache_lock` (cache_owner) held inside single `run_sync_session` method body | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D3.D, D4.C, D4.E | high | label inversion if `_cache_ready_step` updated outside `_resize_sync_lock`; lock-across-RPC anti-pattern if `_resize_sync_lock` held during `service.sync_selected_workers` (would block all concurrent coordinator RPCs for the full ~150 s atomic-unit timeout); `_cache_lock` cannot be split across multiple top-level Ray methods |
| F21 | Per-bucket payload carries NO `weight_version` field; version published once per sync via `manager.set_weight_version` | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D4.C, E6, Fix #3 | high | per-bucket version causes engine to publish new label after first bucket while others still loading old → version inversion |
| F22 | Init bootstrap Step 6.6 pre-registration BEFORE Step 7 INIT (manager + service resources + base `_cache_ready_step=-1` + `bootstrap_active_engines(frozenset())`) | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | A5, D3.E, AC10/AC11/AC12 | high | scheduler runs `coordinator.resize_infer(add)` Phase 5 BEFORE Phase 6 waiter signal |
| F23 | F2 abort-drain-sleep ordering: admission close → `_abort_engines` → drain via `/v1/loads` `num_total_reqs == 0` → `release_memory_occupation` → post-sleep VRAM assert | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | A7, D3.A, AC3 | high | SGLang `release_memory_occupation` requires `is_fully_idle()`; partial drain → assert fires |
| F24 | F2 EngineInfo 5-state machine (`shell` / `active` / `disabling` / `offloaded` / `loading`); single `Dict[int, EngineInfo]` source of truth | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D4.A, AC3/AC10 | high | parallel maps drift; state-vs-flag dual source breaks dispatch invariant |
| F25 | F4 NCCL broadcast TCP rendezvous (NOT `dist.new_group`) + warmup allreduce on every CREATE; cache_owner global-rank as `src=0`; receivers ≥ 1 | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D5.B, AC4/AC5 | high | cross-process: `dist.new_group` cannot subgroup; warmup catches dead groups before USE |
| F26 | C16 NCCL `master_port != 0`; `get_free_port()` + SharedStorage claim per sync | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D5.E, C16, AC9 | high | `tcp://addr:0` makes other ranks unable to discover ephemeral port |
| F27 | SGLang HTTP `/update_weights_from_cpu_bucket` route enters `tokenizer_manager.model_update_lock.writer_lock`; `try/finally` release | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | F5+6 active-refresh-safety; D5.D; Fix #4 | high | active refresh allows engine to keep serving but new admission MUST pause during weight copy |
| F28 | tmpfs file lifecycle: cleanup owner = wrapper `try/finally os.unlink`; SGLang server only reads; per-bucket receiver invocation **serial** (peak = 1× bucket_size) | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D5.A, A4 | high | parallel receivers exhaust `/dev/shm` (Docker default 64 MB) |
| F29 | F3 multi_turn turn-level redispatch with `MAX_TURN_REDISPATCH_ATTEMPTS = args.rollout_num_gpus // args.rollout_num_gpus_per_engine` (total engines, NOT active count); exhaustion raises `EnginePreemptedError` | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D3.B, AC6, E4 | high | active-count cap collapses to 1 in shrink-to-1 cases |
| F30 | NEW exceptions `EnginePreemptedError` + `RLixRouterMetadataError` in `miles/rollout/base_types.py`; fully_async catches ONLY these two for fatal sentinel path | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D3.B, AC6 | high | catching generic Exception expands fatal surface to tool errors / OOM |
| F31 | `_is_scheduler_preempt(output, *, rlix_mode)` — RLix mode missing `miles_admission_disabled` raises `RLixRouterMetadataError`; standalone returns False | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | A19 cascade; D3.B; AC6 | high | silent degrade in RLix mode breaks turn-level redispatch |
| F32 | F3 router `do_proxy` mutates JSON body of `/generate` ONLY; injects `meta_info["miles_engine_index", "miles_admission_disabled"]`; strips `Content-Encoding`; re-serializes | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D4.B, AC6 | high | path-broad mutation breaks `/model_info` / `/v1/loads` / `/health` |
| F33 | F12 SGLang `base_gpu_id=0` in RLix path (post-CVD local view); NEVER `wp.gpu_ids[0]` | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D4.D, M9 | high | post-CVD process sees only `cuda:0..tp-1`; physical id breaks SGLang validation |
| F34 | F12 `LOCAL_RANK=0` explicit injection in RLix path (`TrainRayActor` accepts `local_rank` kwarg; `RayTrainGroup` passes `0`) | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | Fix #9, D5.F | high | RLix `num_gpus_per_actor=0.01` + manual CVD makes `ray.get_gpu_ids()` not in CVD list → ValueError |
| F35 | F10 startup fail-fasts assert C1–C23 (topology, `offload_train`, `async_save`, transport, mapping, parallelism divisibility, fullasync, etc.) | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement (`MilesPipeline.initialize_pipeline`) | C1–C23, AC7, E7/E8 | high | misconfiguration → silent OOM / mid-init crash with no diagnostic |
| F36 | M4 minimal hard cleanup: terminate Ray actors AND SGLang server process tree; bounded wait; raise on timeout; gated on `actor_infer_allocated` | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | §3.1 Layer 6 B2; Anti-regression invariant #10 | high | scheduler/Ray ledger split → next pipeline OOM |
| F37 | M11.2 Pipeline B full INIT + offload (AC10): `Priority.INIT` full allocation; all engines create+onload+offload; NO sync, NO version publish, NO router activation during INIT | P0-CORRECTNESS-FLOOR | M11.2 P1 | implement | AC10, C14, E12, M1 | high | partial init = invalid path; init must be uniform full-allocation high-priority |
| F38 | M11.2 donor-shrink-before-recipient-expand T1 < T2 < T3 (AC12): A.shrink completes BEFORE B.expand starts BEFORE B's pending GENERATION returns | P0-CORRECTNESS-FLOOR | M11.2 P1 | implement | AC12, A5, E13 | high | B waking SGLang on GPUs A still holds = double-occupy → OOM |
| F39 | M11.2 router 0-active MVP suspend: `_use_url` async + `_workers_changed: asyncio.Condition` + unbounded `wait_for(predicate)`; load increment INSIDE lock; every state-mutating endpoint ends `notify_all`; helpers stay sync | P0-CORRECTNESS-FLOOR | M11.2 P1 | implement | AC13, C20, D3.G, D4.B, E14 | high | M11.2 RUNTIME GENERATION shrink/expand reaches active-set-empty as legit runtime status |
| F40 | M11.2 base `version=-1` runtime expand path (AC11): `manager.expand_engines(target)` → `service.sync_selected_workers(sync_id, target, -1)` → `manager.activate_routing(target)`; routing opens AFTER sync | P0-CORRECTNESS-FLOOR | M11.2 P1 | implement | AC11, C13, D5.C, E5 | high | Pipeline B's first runtime expand may fire BEFORE first `after_training`; without base v=-1, serves stale weights |
| F41 | F11 DO_TIME_SHARING dual entry: `train_async.py` standalone unchanged + new `examples/rlix/run_miles_rlix.py`; standalone fail-fast on `RLIX_CONTROL_PLANE=rlix` AND on partial-overlap topology without flag | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | D2.C, AC8, E10 | high | partial-overlap in standalone mode = silent OOM via full-broadcast |
| F107 | X2 `MilesCoordinator.register_model_update_resources(*, cache_owner_actor, rollout_manager)` ctor handle injection (M2 P0-1) | P0-CORRECTNESS-FLOOR | M11.1 P1 | implement | §3.1 Special X2 | medium | architectural ownership boundary; alternative `ray.get_actor` is Layer 1 forbidden (cross-ref F05); plan classifies as Special, not Layer 1-8 — see §11 NQ11.4 |
| F66 | tmpfs file naming `miles_cpu_bucket_{uuid}.pt` for grep-friendly leak detection | P1-OBSERVABILITY-FLOOR | M11.1 | implement | §F4 §B tmpfs lifecycle | medium | low cost; surfaces leaks operationally; plan doesn't classify in §3.1 — see §11 NQ11.1 |
| F75 | A9 `verify_model` debug validation gated by `MILES_DEBUG_VERIFY_MODEL=1` | DEV-ONLY | M11.1 | keep behind flag (do NOT add to production receiver API; do NOT make Gate 2.5 pass criterion) | §3.1 Layer 8 ✅ A9 | high | per-bucket barrier + warmup allreduce is the actual MVP verification |
| F76 | `GET /admission_state` test diagnostic endpoint; gated `MILES_ROUTER_TEST_HOOKS=1` | DEV-ONLY | M11.2 (Gate 4 (f)) | keep behind flag | source plan §Gate 4 (f) Step P1 | medium | needed because router runs as separate FastAPI process; plan §3.1 Layer 8 doesn't enumerate — see §11 NQ11.2 |
| F77 | B1 test-side `asyncio.wait_for` wrapper for Gate 4 (f) sub-tests (60s positive / 30s counter); production router stays unbounded | DEV-ONLY (secondary: P1-LIGHTWEIGHT-DEFENSIVE per plan Layer 7) | M11.2 (Gate 4 (f)) | implement test-side; do NOT add timeout to production `_use_url` | §3.1 Layer 7 ✅ B1; §Gate 4 (f) | medium | plan classifies as Layer 7 (lightweight defensive); plan also says "not a production safeguard" → triage prefers DEV-ONLY; both in MVP — see §11 NQ11.3 |
| F108 | C1 `MilesPipeline` actor `max_concurrency` (tentatively `=2`); §Audit Checkpoints defer decision | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 (decision) / M11.1 (validation) | implement default `=2`; document audit checkboxes | §3.1 Special C1; §Audit Checkpoints | medium | plan defers decision to execution phase — see §11 NQ11.5 |
| F79 | A1 — bounded `_use_url` timeout + 503 sentinel + client `EnginePreemptedError` translation + `_RouterDispatchTimeout` + `RayTaskError` unwrap | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A1 | high | revisit trigger: M11.5 production SLA; MVP keeps unbounded suspend; manual `ray stop` is recovery |
| F80 | A2 — `admission_epoch` start/end race defense | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A2 | high | revisit: production multi-pipeline high-frequency shrink/expand triggers false-negative race |
| F81 | A3 — Multi-pipeline orchestrator-driven selective namespace cleanup | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A3 | high | revisit: shared-cluster pipeline crash kills healthy pipelines via `ray stop` |
| F82 | A4 — Receiver crash tolerance / conditional port leak / periodic port GC | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A4 | high | revisit: production observation of `master_port` collision |
| F83 | A5 — NCCL `master_port` TIME_WAIT cooldown queue / port pool | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A5 | high | revisit: EADDRINUSE retry exhaustion exceeds budget |
| F84 | A10 — SGLang ingress 503 + 5xx preempt sentinel synthesis (Case B 5xx-residual classification) | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A10 | high | revisit: production 5xx-residual rate justifies Case B classification |
| F85 | A12 — Plasma true zero-copy `cpu_serialize` adapter (memoryview-backed reader) | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A12 | medium | revisit: receiver-side bucket copy becomes RAM bottleneck |
| F86 | A13 — `args.async_save` support (`maybe_finalize_async_save(blocking=True)` + `cuda.synchronize()` in `actor.sleep()` prologue) | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A13 | high | revisit: RLix-mode workflow needs train-step async ckpt; MVP F10 fail-fasts |
| F87 | A14 — Train-side post-offload VRAM assert via `torch.cuda.memory_allocated()` | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A14 | high | revisit: torch_memory_saver leak observed; DISTINCT from server-side `/server_info` assert (kept as #8 invariant) |
| F88 | A15 — Engine-level overlap `non_overlap_engines >= 1` set-intersection assert | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A15 | medium | F10 already has C2 (≥2 engines); set-intersection is redundant in MVP |
| F89 | A16-recovery — `enable_worker` resurrects dead worker + Gate 4 dead-recovery as pass criterion | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A16-recovery | high | revisit: production multi-pipeline dead-worker recovery rate |
| F90 | A17 — Router `do_proxy except` broadening to `(KeyError, AttributeError, TypeError, JSONDecodeError)` | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3 A17 | medium | revisit: production router metadata parse exceptions |
| F91 | Graceful actor drain replacing `ray.kill` (`actor.shutdown()` RPC + 30s+force-kill timeout + cleanup daemon) | P3-DEFER-HARDENING | M11.5 | defer | §3.1 Layer 3; §F12 (c) follow-up | high | revisit: production multi-pipeline cleanup race observed |
| F92 | M11.6 `cuda_ipc` colocate adapter (CPU cache → H2D staging → IPC handle, ~50–80 lines) + smoke-test capability check | P3-DEFER-HARDENING | M11.6 | defer | §3.5 Strategy table; §3.1 Layer 3 (M11.6 line) | high | revisit: M11.6 production cluster (with `--ipc=host` / `CAP_SYS_PTRACE`) |
| F93 | M11.4 LoRA + multi-stream aggregation impl | P3-DEFER-HARDENING | M11.4 | defer | §3.4 M11.4 row | high | revisit: M11.4 milestone; M11.1 hook signature already preserves `mode/adapter_id` nullable (F106 forward-compat) |
| F94 | A7 — F12 `infer_device_mapping` round-trip identity self-check / scheduler allocation cross-check | P3-DEFER-HARDENING | trigger-bound | defer | §3.1 Layer 4 A7 | high | revisit: A18 unblock; first-build contiguous makes it dead assert; needs `scheduler.get_allocation()` public API |
| F95 | A18 — Non-contiguous / custom-ordered `infer_device_mapping` adapter / `scheduler_dp_rank → engine_index → gpu_ids` lookup | P3-DEFER-HARDENING | trigger-bound | defer | §3.1 Layer 4 A18 | high | revisit: cross-node engine or custom GPU ordering use case |
| F95.1 | M11.2 — `MilesPipeline._init_phase_b_infer` infer-pool PG via `MilesPlacementProvider.get_all_rollout_engine_placements()` (replace standalone `_create_placement_group(rollout_num_gpus)` bypass) | P3-DEFER-HARDENING | M11.2 | defer | §F22 RELAXED; b241af4 P1-2 commit; debug log REV-006 | high | revisit: dual-pipeline contention smoke; current code has a fail-fast assert (`len(set(infer_allocated)) >= rollout_num_gpus`) that blocks silent oversubscription, but the proper fix translates `WorkerPlacement` list back into `(pg, reordered_bundle_indices, reordered_gpu_ids)` tuple `RolloutManager.__init__` consumes |
| F96 | PD disaggregation | NO-OVERENGINEERING | — | not implement; F10 fail-fast | §3.1 Out of Scope; C9 | high | not supported in any milestone |
| F97 | vLLM backend | NO-OVERENGINEERING | — | not implement | §3.1 Out of Scope | high | only SGLang in scope |
| F98 | MoE / EP support in F4 cache | NO-OVERENGINEERING | — | not implement; F10 fail-fast | §3.1 Out of Scope; C8 | high | F4 covers dense Megatron only; ~200-400 lines + dedicated parity gate (separate plan) |
| F99 | `sglang_data_parallel_size > 1` | NO-OVERENGINEERING | — | not implement; F10 fail-fast | §3.1 Out of Scope; C4 | high | adds DP axis port doesn't handle |
| F100 | Multi-LoRA / DPO / SFT | NO-OVERENGINEERING | — | not implement | §3.1 Out of Scope | high | DPO/SFT entirely out; LoRA = M11.4 (F93) |
| F101 | Request-level deterministic migration (ROLL `RequestScheduler`) | NO-OVERENGINEERING | — | not implement | §3.1 Out of Scope | high | turn-level redispatch is the substitute |
| F102 | Cross-node single rollout engine | NO-OVERENGINEERING | — | not implement; F10 fail-fast (C7) | §3.1 Out of Scope | high | per-node `WorkerPlacement.placement_group`; no later milestone re-enables |
| F103 | `RadixTreeMiddleware` + `partial_rollout + radix_tree` | NO-OVERENGINEERING | — | not implement; F10 fail-fast (C17) | §3.1 Out of Scope | high | hides scheduler-preempt; follow-up after turn-level redispatch is stable |
| F104 | Subclassing `PipelineCoordinator` (cross-ref F09 P0) | NO-OVERENGINEERING | — | not implement (also forbidden) | §3.1 Out of Scope; Layer 1 | high | secondary label of F09 |
| F105 | Branching RLix vs standalone inside `train_async.py` | NO-OVERENGINEERING | — | not implement | §3.1 Out of Scope | high | dual-entry decouples paths |

### Summarized rows (all `Confidence: high`, all `Action: implement`, all `Delta: ALIGNED`)

For full per-row detail see Appendix I.

- **F42–F64** (23 items): P1-HAPPY-PATH for M11.1/M11.2 main path — F1 sleep/wake tag-based, F2 `shrink_engines/expand_engines/activate_routing/finish_init_offload`, F3 router admission API (`/disable_worker` etc.) + 4-dict lifecycle, multi_turn snapshot/restore + force `stream=False`, fully_async `_FatalError` queue sentinel, `build_cpu_bucket_cache` HF-format gather, cpu_serialize transport (4 SGLang patches), NCCL broadcast non-colocate path, `cache_owner_actor.run_sync_session(plan)` single composite RPC (M2), `SyncSessionPlan` frozen dataclass, dual-mask receivers, `coordinator.sync_base_weights_to_active(step)`, `coordinator._expand_workers`, F7 namespace isolation, F8 orchestrator allocate/register/admit, F9 progress reporting (group-unit + 2% bucket gate), F9 `RLixHooks` protocol + `NoOpRLixHooks`, `MilesPlacementProvider` real adapter, `RayTrainGroup`/`RolloutManager` `worker_placements` path with `num_gpus_per_actor=0.01`, `RolloutManager` all-shell ctor with `all_engine_placements + active_engine_indices=frozenset()`, `abort_all_requests`, `is_idle()` polling `/v1/loads`.
- **F65, F67** (2 items): P1-OBSERVABILITY-FLOOR — post-sleep VRAM via SGLang `/server_info memory_usage` GB; instrumented atomic-unit per-bucket / per-engine call counters.
- **F68, F69, F70, F71, F72, F73, F74, F106** (8 items): P1-LIGHTWEIGHT-DEFENSIVE Layer 7 — A16-cheap (`dead_workers.discard` + failure_count reset); X3 `_active_engines_bootstrapped` flag (cross-ref F19); B3 `master_port` `get_free_port()` + SharedStorage claim + EADDRINUSE retry-then-fail-fast; standalone partial-overlap fail-fast; router header hardening (strip `Content-Encoding`); `/dev/shm` capacity check; F12 round-trip structural validation; X1 `mode/adapter_id` nullable forward-compat for M11.4.
- **F78** (1 item): DEV-ONLY — `tests/test_partial_sleep_wake.py` (F1-F3) + `tests/test_miles_pipeline.py` (F4-F6) test scaffolding.

Total summarized: 34 items.
Total visible in §3 above: 74 items.
Grand total: 108 items.

## 4. AC Coverage Check

| AC | Source milestone | Covered by scope items | Delta warnings | Missing? |
|---|---|---|---|---|
| AC1 | M11.1 | F15, F17, F22, F35, F41, F42, F43, F44, F46, F48, F49, F50, F51, F52, F53, F54, F58, F60, F62, F65 | None on covering items | no |
| AC2 | M11.1 | F15, F20, F21, F54, F58, F65 | None | no |
| AC3 | M11.1 | F23, F24, F42, F43, F44, F45, F46, F63, F64, F65, F68 | None | no |
| AC4 | M11.1 | F15, F25, F49, F50, F51, F52, F53, F73 | None | no |
| AC5 | M11.1 | F17, F25, F50 | None | no |
| AC6 | M11.1 | F29, F30, F31, F32, F44, F46, F47 | None | no |
| AC7 | M11.1 | F35, F71 | None (F35 covers all C1–C23 enumerated in AC7) | no |
| AC8 | M11.1 | F41, F71 | None | no |
| AC9 | M11.2 | F26, F56, F57, F60, F61, F62 | None | no |
| AC10 | M11.2 | F19, F22, F24, F37, F62 | None | no |
| AC11 | M11.2 | F16, F22, F40, F55, F67 | None | no |
| AC12 | M11.2 | F22, F38 | None | no |
| AC13 | M11.2 | F02 (forbid queue), F06 (lock invariant), F07 (forbid raise), F14 (helpers stay sync), F39 (suspend impl), F45 (4-dict lifecycle drives notify_all), F76 (test diagnostic), F77 (test wrapper) | F76 = `MISSING_SOURCE_SCOPE` (NQ11.2); F77 = `CATEGORY_REFRAMED` (NQ11.3); both `MVP set impact: no`, non-blocking | no |

All ACs covered. AC13's two non-`ALIGNED` covering items are non-MVP-affecting deltas → §11 routing.

## 5. P0 Must-Address

(Full detail blocks for each `P0-FORBIDDEN` and `P0-CORRECTNESS-FLOOR` item. Restated from source plan; not invented.)

### F01 A6 — `_fatal_error` flag dual-path in fully_async

- Category: P0-FORBIDDEN
- Why it matters: queue ↔ flag dual-source = state-divergence bug surface; `output_queue` already covers all triggers via `_FatalError` sentinel.
- Required implementation: Anti-regression invariant #9 — fully_async `_FatalError` queue sentinel single-path. Do NOT add an `_fatal_error` flag attribute to `AsyncRolloutWorker`.
- Required test/evidence: code review on `examples/fully_async/fully_async_rollout.py` confirms no flag attribute; `task_done_callback` only `output_queue.put`s `_FatalError`.
- Anchors: §3.1 Layer 1 A6; Anti-regression invariant #9; F47.

### F02 A11 — Router internal request queue / retry / spillover / synthetic fallback

- Category: P0-FORBIDDEN
- Why it matters: queue is unobservable; spillover breaks namespace + scheduler accounting; retry inside router hides scheduler preempt from multi_turn turn-redispatch.
- Required implementation: C20 model = empty active set `asyncio.Condition` suspend ONLY (F39). No request queue, no retry loop, no spillover, no synthetic worker fallback in `miles/router/router.py`.
- Required test/evidence: code review of router; absence of queue / retry counters / "fallback worker" selection.
- Anchors: §3.1 Layer 1 A11; C20.

### F03 A19 — Promoting `_preempted_engines` to routing/dispatch/attribution/resize-safety state

- Category: P0-FORBIDDEN
- Why it matters: re-promotion breaks A19 cascade across F2 + F11 + Gate 1 — silent group-recycle attribution returns; manager-side state diverges from `router.enabled_workers`.
- Required implementation: keep `_preempted_engines` only as `RolloutManager`'s abort-RPC idempotency cache, with the cache lifecycle (lazy init / update on abort / discard on `offloaded` transition / reset on shrink failure) confined to `_abort_engines` and its private helpers (`_release_abort_idempotency_for`, `_reset_abort_idempotency_for`); never read from routing / dispatch / multi_turn / resize-safety code; never mutated from any other RolloutManager method or any caller outside the manager.
- Required test/evidence: grep — `_preempted_engines` references occur only inside `_abort_engines`, `_release_abort_idempotency_for`, or `_reset_abort_idempotency_for` bodies (plus their callers passing indices in).
- Anchors: §3.1 Layer 1 A19.

### F04 Splitting `cache_owner.run_sync_session(plan)` into multiple top-level Ray RPCs

- Category: P0-FORBIDDEN
- Why it matters: `_cache_lock` cannot span Ray RPCs (lock-across-RPC anti-pattern); split RPCs fragment the timeout boundary.
- Required implementation: cache_owner exposes ONE top-level Ray method `run_sync_session(plan: SyncSessionPlan) -> None` that holds `_cache_lock` for the whole transport phase; sender helpers (cpu_serialize per-engine RPC + NCCL group setup/teardown) are private in-method helpers, NOT Ray methods.
- Required test/evidence: grep `external/miles/miles/backends/megatron_utils/actor.py` — only `build_cpu_bucket_cache`, `report_cache_owner_role`, `run_sync_session` exposed as `@ray.method`.
- Anchors: §3.1 Layer 1; D3.D; D4.E; F51.

### F05 Calling `service.sync_selected_workers` / `finalize_weight_update` / `manager.set_weight_version` outside atomic-unit / bootstrap entry points

- Category: P0-FORBIDDEN
- Why it matters: atomic-unit invariant is service-owned (D3.D); pipeline/coordinator direct calls fragment timeout boundary and bypass per-bucket-no-version invariant (F21).
- Required implementation: pipeline calls only `coordinator.sync_base_weights_to_active(step)`; coordinator calls `service.sync_selected_workers(sync_id, target, version)` (atomic unit); service internally drives finalize fan-out + `manager.set_weight_version`; cache_owner internally drives transport.
- Required test/evidence: grep `rlix/pipeline/miles_pipeline.py` for `finalize_weight_update.remote` / `set_weight_version.remote` — should be ZERO direct calls.
- Anchors: §3.1 Layer 1; D3.D; F15.

### F06 `worker_request_counts[url] += 1` outside the `asyncio.Condition` lock

- Category: P0-FORBIDDEN
- Why it matters: concurrent suspend/resume races with `_finish_url` decrement → negative-count assert.
- Required implementation: in `_use_url`, the increment MUST be the last statement INSIDE `async with self._workers_changed:` block, BEFORE returning the URL.
- Required test/evidence: counter sub-test of Gate 4 (f) — two concurrent suspended dispatchers resume on same `notify_all`, counter goes net-zero; no negative-count assert.
- Anchors: §3.1 Layer 1; C20; D3.G; E14.

### F07 Raising `RuntimeError("no enabled live workers")` from `_use_url`

- Category: P0-FORBIDDEN
- Why it matters: fully_async sees generic RuntimeError as transient abort → silent group-recycle attribution → bugs hide; M11.2 0-active is legit runtime status (A11).
- Required implementation: `_use_url` MUST `await self._workers_changed.wait_for(predicate)` instead of raise; helpers stay sync; only async endpoints + `_health_check_loop` carry `notify_all`.
- Required test/evidence: Gate 4 (f) positive sub-test (P1→P2→P3); negative-test grep confirming no `RuntimeError("no enabled live workers")` literal in router.
- Anchors: §3.1 Layer 1; C20; F39.

### F08 Mutating `os.environ['CUDA_VISIBLE_DEVICES']` after `import torch` / `import sglang`

- Category: P0-FORBIDDEN
- Why it matters: `cuInit()` lock timing — env mutation after import is silently ineffective; engine sees only Ray-bootstrap-allocated cards (typically 1), TP init crashes.
- Required implementation: CVD set via `ray.remote(...).options(runtime_env={'env_vars': {'CUDA_VISIBLE_DEVICES': csv, 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1'}})` at `.options(runtime_env=...)` time; actor body never touches `os.environ['CUDA_VISIBLE_DEVICES']`.
- Required test/evidence: grep `miles/` for `os.environ['CUDA_VISIBLE_DEVICES'] =` — only in `runtime_env={'env_vars': ...}` constructions.
- Anchors: §3.1 Layer 1; M6; F12 §(b).

### F09 Subclassing `PipelineCoordinator` for `MilesCoordinator`

- Category: P0-FORBIDDEN
- Why it matters: `super().__init__()` invokes ROLL `_validate_config_schema` / `_validate_cpu_only_reward` / `_validate_vllm_sleep_level` / `_validate_offload_nccl` validators on MILES args lacking those fields → init crash.
- Required implementation: `MilesCoordinator(Coordinator)` — manual init (no `super`); copy backend-neutral methods (`report_progress_from_scheduler`, `clear_progress_stream`, `_aggregate_and_emit`, `_inject_pipeline_env_vars`); implement ABC `sync_lora_weights` as raise stub.
- Required test/evidence: code review `rlix/pipeline/miles_coordinator.py` — class extends `Coordinator` ABC, NOT `PipelineCoordinator`; no `super().__init__` call.
- Anchors: §3.1 Layer 1; D1.A.

### F10 Adding device-mapping CLI args

- Category: P0-FORBIDDEN
- Why it matters: new args drift from RLix declared `cluster_device_mappings`; provider must consume injected `train_device_mapping` / `infer_device_mapping` from F8 driver register call (F60).
- Required implementation: derive from `args.actor_num_nodes * args.actor_num_gpus_per_node` and `args.rollout_num_gpus`; do NOT add `--actor-train-device-mapping` / `--infer-device-mapping`.
- Required test/evidence: `miles/utils/arguments.py` diff — only `miles_model_update_bucket_size_mb`, `miles_post_sleep_vram_threshold_gb`, `model_update_transport` added.
- Anchors: §3.1 Layer 1.

### F11 Re-introducing per-request tracking (`_inflight_requests`) or worker-side `is_engine_resident_on_gpu`

- Category: P0-FORBIDDEN
- Why it matters: source of truth is `EngineInfo.state` (manager) + `router.enabled_workers`; per-request map = state divergence; worker-side flag = manager↔worker dual source.
- Required implementation: `RolloutManager._engines: Dict[int, EngineInfo]` is single source of truth; drain reads `/v1/loads` (A7); never instantiate `Dict[int, Set[str]]` for in-flight tracking.
- Required test/evidence: grep `RolloutManager` — no `_inflight_requests` or `is_engine_resident_on_gpu` attribute.
- Anchors: §3.1 Layer 1.

### F12 Re-using `load_format="flattened_bucket"` for `cpu_serialize`

- Category: P0-FORBIDDEN
- Why it matters: `flattened_bucket` is SGLang-internal `FlattenedTensorBucket` protocol; incompatible with ROLL CPU bucket schema (`{"bucket": pinned_cpu_uint8_tensor, "tensors_meta": list[dict]}`).
- Required implementation: cpu_serialize wire payload uses ROLL schema; `load_format` argument values are SGLang receiver-side dispatch tags, NOT `"flattened_bucket"`.
- Required test/evidence: receiver patch (`cpu_serialize_weight_sync_utils.patch`) loads `{"bucket": ..., "tensors_meta": ...}` — not `FlattenedTensorBucket`.
- Anchors: §3.1 Layer 1; D5.A.

### F13 Driver-level `ray.shutdown()` or top-level `try/except` in `run_miles_rlix.py`

- Category: P0-FORBIDDEN
- Why it matters: hides actor-lifetime bugs; `ray stop` CLI is the recovery surface (per-node, kills all actors regardless of detached state).
- Required implementation: `examples/rlix/run_miles_rlix.py` `main()` has NO top-level `try/except` and NO `ray.shutdown()`; failures raise → driver exit → user `ray stop && ray start ...`.
- Required test/evidence: code review of `examples/rlix/run_miles_rlix.py` — `main()` body is straight-line orchestrator + coordinator + pipeline + main loop; no exception handler at top level.
- Anchors: §3.1 Layer 1; F12 §(c).

### F14 Making sync admission helpers (`_*_internal`) `async def`

- Category: P0-FORBIDDEN
- Why it matters: F3 §(b.4) defines helpers as sync; only async endpoints + `_health_check_loop` carry `notify_all`; making helpers async fragments the F3 admission state machine.
- Required implementation: `_add_worker_internal` / `_remove_worker_internal` / `_disable_worker_internal` / `_enable_worker_internal` stay `def` (sync); the async endpoints (`async def add_worker(self, request)` etc.) call them, then issue `async with self._workers_changed: self._workers_changed.notify_all()`.
- Required test/evidence: grep `miles/router/router.py` — `_*_internal` methods are `def`, not `async def`; only public endpoints + `_health_check_loop` await condition lock.
- Anchors: §3.1 Layer 1; F3 §(b.4).

### F15 D3.D atomic sync unit

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: splitting (a)+(b)+(c) reintroduces lock-across-RPC; partial timeout boundaries leave receiver in inconsistent state; per-bucket version label causes inversion.
- Required implementation: `MilesModelUpdateService.sync_selected_workers(sync_id, target_engines: Set[int], version: int) -> int` builds `SyncSessionPlan` once at entry (with `target_handles = ray.get(rollout_manager.get_engine_handles.remote(target_engines))`), then in single `asyncio.wait_for(timeout=ROLL_SELECTIVE_MODEL_UPDATE_TIMEOUT_S)`: (a) `cache_owner_actor.run_sync_session(plan).remote()` ack — single composite RPC; (b) `[h.finalize_weight_update.remote() for h in plan.target_handles.values()]` fan-out; (c) `manager.set_weight_version.remote(version, engine_indices=target_engines)`.
- Required test/evidence: E2 (mixed-mask Gate 2.5); E5 (Gate 4 (d) instrumented counters); E6 (per-bucket payload audit, finalize call counter, `set_weight_version` single call).
- Anchors: D3.D; AC1, AC4, AC11; F51, F52, F67.

### F16 C13 base `version=-1` full sync (no checkpoint shortcut)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: label-only path serves stale rollout weights under resume / non-equivalent `args.load`; routing must NOT open until base sync completes.
- Required implementation: `service.sync_selected_workers(sync_id, target_engines, -1)` runs the full atomic unit (transport from init-built CPU bucket via `cache_owner_actor.run_sync_session(plan)` + per-engine `finalize_weight_update` + `set_weight_version(-1)`); coordinator `_expand_workers` calls this BEFORE `manager.activate_routing(target_engines)`.
- Required test/evidence: Gate 4 (d) — instrumented assertions that `cache_owner_actor.run_sync_session` and per-engine `finalize_weight_update` each run for the expanded engine during `version=-1` base sync, before routing opens; engine reported version after sync = -1.
- Anchors: C13; D5.C; AC11; E5.

### F17 D3.F F4 sequence (build → offload → destroy NCCL → sync)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: build cache after offload deadlocks gather (rank already left NCCL group); sync before offload serves stale weights and leaks GPU VRAM.
- Required implementation: `MilesPipeline._after_training(step)` runs in fixed order: (1) `actor_train.build_cpu_bucket_cache(step)` (all ranks gather, cache_owner stores), (2) `actor_train.offload()` (sleep + ReloadableProcessGroup destroy), (3) `coordinator.sync_base_weights_to_active(step)` triggers expand sync.
- Required test/evidence: E9 (Gate 2.5 destroy/reload cycle ≥3 steps; verify VRAM after offload + no NCCL group leak); ordering assertion in `MilesPipeline._after_training`.
- Anchors: D3.F; AC4, AC5; F4 §1.

### F18 C15 cache_owner uniqueness (init Step 6.5)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: multiple owners → duplicate writes; zero owners → empty cache; sync-hot-path query violates the "owner discovered at init, not sync" invariant.
- Required implementation: init Step 6.5 — `roles = run(self.actor_train.collect_cache_owner_roles())`; `cache_owner_actor = next(h for r, owner, h in roles if owner)`; `assert cache_owner_actor is not None`; cached on `MilesPipeline._cache_owner_actor` for entire pipeline lifetime; passed to coordinator via `register_model_update_resources(cache_owner_actor=...)` (F107).
- Required test/evidence: E12 (Gate 4 (c) Pipeline B init); fail-fast on any non-1 owner count.
- Anchors: A6; C15; D2.B.

### F19 X3 active-set bootstrap with `_active_engines_bootstrapped` flag

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: M11.2 init bootstrap input is `frozenset()` (empty set); without explicit flag, repeated bootstrap silently overwrites an already-populated set (the real ordering bug is hidden).
- Required implementation: `MilesCoordinator._active_engines_bootstrapped: bool = False` (manual init); `bootstrap_active_engines(engine_indices)` checks flag under `_resize_sync_lock`, raises if already True, then sets `_active_engine_indices = set(engine_indices)` and `_active_engines_bootstrapped = True`. Empty set is legitimate first call.
- Required test/evidence: E12 (Gate 4 (c) Step 7.3 consistency assert); unit test for double-call raising RuntimeError.
- Anchors: §3.1 Layer 7 ✅ X3; D3.E; AC10.

### F20 `_resize_sync_lock` + `_cache_lock` single-method-single-critical-section

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: label inversion if `_cache_ready_step` updated outside `_resize_sync_lock`; lock-across-RPC if `_cache_lock` split across multiple Ray methods.
- Required implementation: coordinator `sync_base_weights_to_active(step)` body holds `_resize_sync_lock` for the entire `_cache_ready_step = step` + `service.sync_selected_workers(...)` call. cache_owner `run_sync_session` body holds `_cache_lock` for the entire transport phase. `_cache_lock` shared with `build_cpu_bucket_cache` (mutual exclusion).
- Required test/evidence: E2 (Gate 2.5); E5 (Gate 4 (d) instrumented counters); concurrency test of expand+after_training under load.
- Anchors: D3.D; D4.C; D4.E; F4 §6 invariant 4.

### F21 Per-bucket payload NO `weight_version` field

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: per-bucket version label causes engine to publish new version after first bucket while others still loading old → version inversion.
- Required implementation: `update_weights_from_cpu_bucket` HTTP body schema has fields `payload_path, load_format, flush_cache, cpu_serialize_local_ranks` — no `weight_version`. Single per-sync version publish via `manager.set_weight_version(version, engine_indices)` (atomic-unit step c).
- Required test/evidence: E6 (per-bucket payload audit); SGLang fork patch (`cpu_serialize_http_route.patch`) review.
- Anchors: D4.C; Fix #3; E6; F49.

### F22 Init bootstrap Step 6.6 pre-registration BEFORE Step 7

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: scheduler runs `coordinator.resize_infer(add)` Phase 5 BEFORE Phase 6 waiter signal (A5); `_expand_workers` must find manager + service resources + base `_cache_ready_step` + bootstrap'd active set ALREADY in place, otherwise mid-init raise.
- Required implementation: in `MilesPipeline.initialize_pipeline()`: Step 6.5 collect cache_owner; Step 6.6a get all rollout placements; Step 6.6b create all-shell `RolloutManager`; Step 6.6c sanity engine_count; Step 6.6d `coordinator.register_model_update_resources(cache_owner_actor, rollout_manager)`; Step 6.6e `coordinator.publish_cache_ready_step(-1)`; Step 6.6f `coordinator.bootstrap_active_engines(frozenset())`. THEN Step 7 `_request_cluster_gpus(actor_infer, INIT)`.
- Required test/evidence: E12 (Gate 4 (c)); E13 (Gate 4 (e) timestamp); pre-condition checks in Step 6.6 are blocking before Step 7.
- Anchors: A5; D3.E; AC10, AC11, AC12.

### F23 F2 abort-drain-sleep ordering

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: SGLang `release_memory_occupation` requires `assert is_fully_idle()`; partial drain leaves in-flight requests → assert fires → silent crash.
- Required implementation: `RolloutManager.sleep_partial(engine_indices)` order: (1) under `_routing_lock`: `state ∈ {active} → disabling`, `_active_engine_indices -= ...`; (2) outside lock: `_abort_engines(engine_indices)` (POST `/abort_request {abort_all: True}` per F63); (3) `_wait_engine_idle(idx)` polls `is_idle()` (F64); (4) `release_memory_occupation(tags=None)`; (5) post-sleep VRAM assert (F65); (6) under `_routing_lock`: `state → offloaded`.
- Required test/evidence: E1 (Gate 1 a/b — engine_index not in router.enabled_workers after sleep_partial; post-sleep VRAM under threshold).
- Anchors: A7; D3.A; AC3.

### F24 EngineInfo 5-state machine

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: parallel maps drift; state-vs-flag dual source breaks dispatch invariant.
- Required implementation: single `RolloutManager._engines: Dict[int, EngineInfo]` where `EngineInfo` has `state: Literal["shell", "active", "disabling", "offloaded", "loading"]`; documented transitions per Appendix F.2 of TLDR.
- Required test/evidence: E1 (Gate 1); F2 unit test of state transitions.
- Anchors: D4.A; AC3, AC10.

### F25 NCCL broadcast TCP rendezvous + warmup allreduce

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: cache_owner Megatron actor + SGLang TP workers are independent processes — `dist.new_group` cannot subgroup across processes; warmup allreduce catches dead groups before USE phase.
- Required implementation: per-sync CLASSIFY/CREATE/USE/DESTROY: cache_owner `init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}', world_size=N, rank=0)`; receivers via SGLang `init_custom_process_group` with `comm_ranks[engine_idx]` ≥ 1; CREATE step issues `dist.all_reduce(torch.zeros(1, device='cuda'), group=tmp)` and on failure immediately destroys + raises.
- Required test/evidence: Gate 2.5 (c) — fault-injected dead group raises before USE; sender + receiver TCP rendezvous protocol.
- Anchors: D5.B; AC4, AC5; Anti-regression invariant #2.

### F26 NCCL `master_port != 0` (C16)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: `tcp://addr:0` makes other ranks unable to discover ephemeral port → silent rendezvous failure on multi-rank.
- Required implementation: `MilesModelUpdateService` in sync entry: `master_port = get_free_port()`; `SharedStorage.try_put(f"MASTER_ADDR_PORT:{addr}:{port}", pipeline_id)` atomic claim; same port to sender + all receivers.
- Required test/evidence: F10 fail-fast asserts `master_port != 0`; SharedStorage claim release on sync end.
- Anchors: D5.E; C16; AC9; F70.

### F27 SGLang HTTP route under `tokenizer_manager.model_update_lock.writer_lock`

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: active refresh allows engine to keep serving but new admission MUST pause during weight copy; otherwise half-old/half-new weights observed by in-flight requests = structurally wrong.
- Required implementation: `cpu_serialize_http_route.patch` body acquires `tokenizer_manager.model_update_lock.writer_lock` (already exists in SGLang `tokenizer_control_mixin.py`); `try: dispatch / finally: release`; route does NOT return until all TP workers ack.
- Required test/evidence: SGLang fork patch review; Gate 3 (active in-flight refresh measures bounded mis-attribution).
- Anchors: F5+6 active refresh safety; D5.D; Fix #4.

### F28 tmpfs file lifecycle (wrapper-owned cleanup, serial per-bucket)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: parallel receivers exhaust `/dev/shm` (Docker default 64 MB); cleanup-on-server side leaks on receiver crash; per-bucket parallel-N exceeds tmpfs budget.
- Required implementation: wrapper `try: write /dev/shm/miles_cpu_bucket_{uuid}.pt + HTTP POST + receive ack / finally: try: os.unlink; except FileNotFoundError: pass`; `MilesModelUpdateService` calls per-target receivers SERIALLY for the same bucket (peak tmpfs = 1× bucket_size).
- Required test/evidence: Gate 2.5 — `ls /dev/shm/miles_cpu_bucket_*` empty after sync; concurrency stress test confirms serial pattern.
- Anchors: D5.A; A4; F4 §B Critical Invariant.

### F29 Multi-turn redispatch `MAX = total_engine_count`

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: using active engine count caps retry at 1 in shrink-to-1 cases (= no retry at all); silent fallback to group-recycle hides scheduler bugs.
- Required implementation: `MAX_TURN_REDISPATCH_ATTEMPTS = args.rollout_num_gpus // args.rollout_num_gpus_per_engine` (computed at function entry from `input.args`); on exhaustion raises `EnginePreemptedError`.
- Required test/evidence: E4 (Gate 3 (d) — inject sleep_partial mid-turn; verify redispatch picks new engine; exhaustion raises EnginePreemptedError, NOT silent group recycle).
- Anchors: D3.B; AC6; E4.

### F30 EnginePreemptedError + RLixRouterMetadataError exceptions

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: catching generic Exception expands fatal surface to tool errors / framework errors / OOM; missing both exception classes means turn redispatch silently fails.
- Required implementation: NEW classes in `miles/rollout/base_types.py` (same module as `GenerateFnInput / GenerateFnOutput` — multi_turn already imports it, zero extra import seam).
- Required test/evidence: code review; fully_async `_FatalError` callback catches ONLY `(EnginePreemptedError, RLixRouterMetadataError)` per F47.
- Anchors: D3.B; AC6.

### F31 `_is_scheduler_preempt(output, rlix_mode)` classification

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: silent degrade in RLix mode (returning False on missing metadata) breaks turn-level redispatch — never triggers; mis-classifies router-bypassed responses.
- Required implementation: function signature `_is_scheduler_preempt(output, *, rlix_mode: bool) -> bool`; `finish_reason["type"] != "abort"` → False; missing `meta_info["miles_admission_disabled"]` AND rlix_mode → raise `RLixRouterMetadataError`; standalone returns False; otherwise `bool(admission_disabled)`.
- Required test/evidence: unit test for each branch (rlix mode + missing metadata raises; standalone tolerant; `admission_disabled = True` returns True).
- Anchors: A19 cascade; D3.B; AC6.

### F32 Router `do_proxy` JSON metadata injection (path-guarded)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: path-broad mutation breaks `/model_info` / `/v1/loads` / `/health` schemas; missing Content-Encoding strip corrupts client decode of mutated body.
- Required implementation: in `do_proxy`, ONLY when `path == "generate"`: parse `content` as JSON, set `data["meta_info"]["miles_engine_index", "miles_admission_disabled"]`, re-serialize; pop `Content-Encoding` (upper + lower) from `upstream_headers`; on `(json.JSONDecodeError, KeyError)` skip injection (multi_turn RLix mode will fail-fast on missing metadata via F31).
- Required test/evidence: unit tests for `/generate` (mutated), `/model_info` (unchanged), gzip body (gracefully skips injection).
- Anchors: D4.B; AC6; F72.

### F33 SGLang `base_gpu_id=0` (post-CVD local view)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: post-CVD process sees only `cuda:0..tp-1` (local indices); passing physical id breaks SGLang `physical_gpu_id ∈ CVD list` validation; silent miscount of usable GPUs.
- Required implementation: in RLix path of `start_rollout_servers_from_worker_placements`, explicitly pass `base_gpu_id=0` to `SGLangEngine`; do NOT use `wp.gpu_ids[0]`; do NOT fall back to `get_base_gpu_id(args, rank)` (standalone heuristic).
- Required test/evidence: Gate 1 — engine starts on multi-GPU correctly; check `args.base_gpu_id == 0` in spawn config.
- Anchors: D4.D; M9; F12 §(b).

### F34 LOCAL_RANK explicit injection (Fix #9)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: RLix mode `num_gpus_per_actor=0.01` + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` makes `ray.get_gpu_ids()` not in manual CVD list → `cvd.split(",").index(...)` ValueError or wrong LOCAL_RANK.
- Required implementation: `TrainRayActor.__init__(*, local_rank: int | None = None)`; sets `os.environ["LOCAL_RANK"] = str(local_rank if local_rank is not None else get_local_gpu_id())`; RLix `RayTrainGroup` passes `local_rank=0` (1 actor 1 GPU under fractional + manual CVD); standalone path doesn't pass (uses fallback).
- Required test/evidence: standalone path regression unchanged; RLix path LOCAL_RANK=0 verified.
- Anchors: Fix #9; D5.F; F12 §(a).

### F35 F10 startup fail-fasts (C1–C23)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: misconfiguration → silent OOM / mid-init crash with no diagnostic; F10 is the only enforcement surface for hard constraints.
- Required implementation: `MilesPipeline.initialize_pipeline()` runs all asserts in §F10 of source plan (full enumeration of C1–C23 from §3.2).
- Required test/evidence: E7 + E8 — fail-fast on each enumerated misconfig (non-contiguous mapping; async_save; RadixTreeMiddleware; streaming; EP/MoE>1; sglang_dp>1; transport mismatch; non-fullasync; Megatron parallelism non-divisible; infer mapping non-divisible).
- Anchors: C1–C23; AC7; F71.

### F36 M4 minimal hard cleanup

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: scheduler/Ray ledger split → next pipeline OOM; CUDA context lives in SGLang server child processes; killing only the Ray actor leaks CUDA context.
- Required implementation: on `MilesPipeline.initialize_pipeline()` failure path: best-effort `ray.kill(h, no_restart=True)` for `actor_train._actor_handles` + `actor_infer.shutdown_hard.remote(timeout=10)` (stops monitors + `ray.kill`s engine actors + terminates SGLang server process tree) + `ray.kill(actor_infer)`; then `_notify_release_cluster_gpus` (gated on `actor_infer_allocated`); finally raise. NO graceful drain RPC, NO 30s force-fallback dual-protocol, NO cleanup daemon.
- Required test/evidence: M4 self-cleanup unit test — failed init leaves no zombie actors AND no scheduler allocation drift.
- Anchors: §3.1 Layer 6 B2; Anti-regression invariant #10.

### F37 M11.2 Pipeline B full INIT + offload (AC10)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: partial init allocation = invalid path; init must be uniform full-allocation high-priority for cross-pipeline contention guarantees and Pipeline B router-empty post-condition.
- Required implementation: scheduler treats init as `Priority.INIT` full allocation; `coordinator._expand_workers([all dps])` Phase 5 calls `manager.expand_engines(all)` (shell→loading lazy ctor + onload) → `manager.finish_init_offload(all)` (loading→offloaded, drop weights+KV+graph, NO sync, NO `set_weight_version`, NO `add_worker`); after init `_active_engine_indices` empty; router worker list empty; M1 `set(allocated) == set(declared)` assert fires fail-fast.
- Required test/evidence: E12 (Gate 4 (c) — Pipeline B `initialize_pipeline` returns SUCCESS; all SGLang handles exist; states `offloaded`; no trainer CPU-bucket sync ran during INIT; router worker list empty; consistency assert passed).
- Anchors: AC10; C14; D3.E; F62.

### F38 Donor-shrink-before-recipient-expand T1 < T2 < T3 (AC12)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: B waking SGLang on GPUs A still holds = double-occupy → OOM; depends on scheduler's A5 Phase-5-before-Phase-6 ordering.
- Required implementation: scheduler enforces serialization (single `_resize_sync_lock` across coordinator); A.shrink completes (engine `offloaded`, GPU released from A) BEFORE B.expand starts; B.expand starts BEFORE B's pending GENERATION request returns to driver.
- Required test/evidence: E13 (Gate 4 (e) — timestamp test on coordinator + driver verifies T1 < T2 < T3 strict ordering).
- Anchors: AC12; A5; F22.

### F39 Router 0-active MVP suspend (C20)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: M11.2 RUNTIME GENERATION shrink/expand reaches active-set-empty as legit runtime status (after full INIT+offload, before scheduler grants `resize_infer(add)`); raise → fully_async retry loop accumulates wall-clock or silent group-recycle attribution.
- Required implementation: `class MilesRouter` ctor: `self._workers_changed = asyncio.Condition()`. `async def _use_url(self)`: `async with self._workers_changed: await self._workers_changed.wait_for(lambda: bool(set(self.worker_request_counts) & self.enabled_workers - self.dead_workers)); valid = ...; url = min(valid, key=...); self.worker_request_counts[url] += 1; return url`. `async def add_worker(...)` etc.: call sync internal helper, then `async with self._workers_changed: self._workers_changed.notify_all()`. `_health_check_loop` notifies on dead/recovered transitions. `do_proxy` MUST `await self._use_url()`. **Unbounded** `wait_for(predicate)` — bounded timeout / 503 sentinel / client translation deferred to M11.5.
- Required test/evidence: E14 (Gate 4 (f) MVP P1→P2→P3 + counter sub-test, both wrapped in test-side `asyncio.wait_for`).
- Anchors: AC13; C20; D3.G; D4.B; E14; F02, F06, F07, F14 (forbids).

### F40 M11.2 base `version=-1` runtime expand path (AC11)

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: Pipeline B's first runtime expand may fire BEFORE its first `after_training`; without base v=-1 sync, Pipeline B serves stale weights or crashes on uninitialized cache_owner state.
- Required implementation: in `coordinator._expand_workers(target_engines)` under `_resize_sync_lock`: assert `_cache_ready_step is not None` (init bootstrap published -1); `manager.expand_engines(target)` (offloaded→loading); `service.sync_selected_workers(sync_id, target, _cache_ready_step)` (atomic unit including base v=-1); `manager.activate_routing(target)`.
- Required test/evidence: E5 (Gate 4 (d) — instrumented call counters increase by exactly one sync/finalize cycle for the expanded engine; engine version after sync = -1).
- Anchors: AC11; C13; D5.C; E5; F16.

### F41 DO_TIME_SHARING dual entry + standalone fail-fast guards

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: mixing entry paths breaks standalone path regression; partial-overlap topology in standalone mode = silent OOM via full-broadcast.
- Required implementation: `train_async.py` top: `if os.environ.get("RLIX_CONTROL_PLANE") == "rlix": raise RuntimeError("...use examples/rlix/run_miles_rlix.py instead.")`; AND `if not DO_TIME_SHARING and train_devices_subset_of_infer(args): raise RuntimeError("...partial overlap detected without RLIX_CONTROL_PLANE...")`. New `examples/rlix/run_miles_rlix.py` is the ONLY entry for RLix mode.
- Required test/evidence: E10 (standalone smoke test passes when `RLIX_CONTROL_PLANE` unset); explicit fail-fast on misuse.
- Anchors: D2.C; AC8; F71; F11.

### F107 X2 — `MilesCoordinator.register_model_update_resources` ctor handle injection

- Category: P0-CORRECTNESS-FLOOR
- Why it matters: architectural ownership boundary — service body doing `ray.get_actor("rlix:rollout_manager:...")` is a Layer 1 forbidden anti-pattern (cross-ref F05); handles must be passed in at construction.
- Required implementation: `MilesCoordinator.register_model_update_resources(*, cache_owner_actor, rollout_manager)` — keyword-only; caches on `_model_update_resources` dict; `MilesModelUpdateService.__init__(*, cache_owner_actor, rollout_manager)` accepts both via ctor injection from coordinator (lazy-init on first sync).
- Required test/evidence: code review of `MilesModelUpdateService` body — no `ray.get_actor` call sites.
- Anchors: §3.1 Special X2; M2 P0-1.

## 6. MVP Implementation Set

The MVP set for **M11.1** (single-pipeline) + **M11.2 happy path** (dual-pipeline + 0-active suspend):

- **P0-CORRECTNESS-FLOOR (cross-ref §5)**: F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F31, F32, F33, F34, F35, F36, F37, F38, F39, F40, F41, F107 — 28 items.
- **P1-HAPPY-PATH**: F42–F64 (summarized) — 23 items.
- **P1-OBSERVABILITY-FLOOR**: F65, F66, F67 — 3 items.
- **P1-LIGHTWEIGHT-DEFENSIVE**: F68, F69 (cross-ref F19), F70, F71, F72, F73, F74, F106, F108 — 9 items.
- **DEV-ONLY (P1, needed for MVP test)**: F76, F77, F78 — 3 items (F75 is developer convenience; not needed for Gate pass).
- **P0-FORBIDDEN guards (cross-ref §5)**: F01–F14 — must be enforced via code review + grep + asserts (14 items).

Total MVP item count: 28 + 23 + 3 + 9 + 3 + 14 = **80 items** for M11.1 + M11.2.

## 7. Dev-Only Scaffolding

### F75 — `verify_model` debug validation

- Purpose: hash-based weight equality check after sync (debug aid for receiver-load bugs).
- How it accelerates: surfaces silent weight corruption that bypasses warmup allreduce.
- Required isolation: gated by `MILES_DEBUG_VERIFY_MODEL=1` env flag; no production receiver API surface change.
- Remove/keep policy: keep behind flag indefinitely; do NOT promote to Gate 2.5 pass criterion (Anti-regression invariant #2 covers production verification).

### F76 — `GET /admission_state` test diagnostic endpoint

- Purpose: test harness can read router admission state cross-process (router runs as separate FastAPI process).
- How it accelerates: precise barrier in Gate 4 (f) Step P1 — eliminates "wait some seconds and hope" race.
- Required isolation: gated by `MILES_ROUTER_TEST_HOOKS=1` env flag; production deployments leave it off.
- Remove/keep policy: keep behind flag for future test scaffolding (Gate 4 hardening sub-tests at M11.5).

### F77 — Test-side `asyncio.wait_for` wrapper

- Purpose: bound CI runtime when buggy `notify_all` path causes `_use_url` to hang indefinitely.
- How it accelerates: fails CI fast at the awaited boundary (60s for positive sub-test, 30s for counter sub-test) instead of silent CI hang.
- Required isolation: test code only; production `_use_url` stays unbounded (per C20 MVP).
- Remove/keep policy: keep with each Gate 4 (f) sub-test; remove only when M11.5 bounded production timeout (F79) makes test-side wrapper redundant.

### F78 — `tests/test_partial_sleep_wake.py` + `tests/test_miles_pipeline.py`

- Purpose: F1-F3 unit + F4-F6 integration coverage.
- How it accelerates: catches state-machine + atomic-unit regressions before Gate 1/2.5/3 runs.
- Required isolation: pytest-only; never imported by production code.
- Remove/keep policy: keep permanently; extend with each Feature.

## 8. Release Blockers

None classified. The plan's release semantics are bundled into M11.5 production hardening (`P3-DEFER-HARDENING`), not into M11.1/M11.2 MVP. Manual `ray stop` is the accepted recovery during MVP per source plan §6.2.

If a future M11.1 → M11.2 sign-off introduces release-time gates beyond Gate 1/2/2.5/3/4, those rows would land here.

## 9. Deferred Hardening

Layer 3 (M11.5) — bound to milestone M11.5 (production deployment / SLA / observed failure-rate):

- **F79 A1** — bounded `_use_url` timeout + 503 sentinel + client `EnginePreemptedError` translation. Revisit trigger: M11.5 production SLA. Risk if never: indefinite `_use_url` suspend hangs production until manual `ray stop`.
- **F80 A2** — admission_epoch race defense. Revisit trigger: production multi-pipeline high-frequency shrink/expand triggers false-negative race. Risk: turn-redispatch picks engine that just got disabled, wastes retry budget.
- **F81 A3** — multi-pipeline orchestrator-driven cleanup. Revisit trigger: shared-cluster pipeline crash kills healthy pipelines via `ray stop`. Risk: orchestrator-level isolation gap.
- **F82 A4** — receiver crash tolerance. Revisit trigger: production observation of `master_port` collision. Risk: NCCL group leak across receiver-crash pipeline restart.
- **F83 A5** — NCCL `master_port` cooldown queue. Revisit trigger: EADDRINUSE retry exhaustion exceeds budget. Risk: occasional sync failure after rapid teardown.
- **F84 A10** — SGLang ingress 503 + 5xx preempt synthesis. Revisit trigger: production 5xx-residual rate justifies Case B classification. Risk: 5xx-residual currently goes to group recycle (B4) — wastes work.
- **F85 A12** — plasma true zero-copy adapter. Revisit trigger: receiver-side bucket copy becomes RAM bottleneck (multi-pipeline accumulation). Risk: F4 spec doesn't promise zero-copy in M11.1; perf, not correctness.
- **F86 A13** — `args.async_save` support. Revisit trigger: RLix-mode workflow needs train-step async ckpt. Risk: forces sync ckpt path in M11.1 (F10 fail-fasts).
- **F87 A14** — train-side post-offload VRAM assert. Revisit trigger: torch_memory_saver leak observed. Risk: silent VRAM retention on train side.
- **F88 A15** — engine-overlap set-intersection assert. Revisit trigger: colocate/non-colocate boundary bug surfaces. Risk: F10 already has C2 (≥2 engines) so this is redundant in MVP.
- **F89 A16-recovery** — `enable_worker` resurrects dead worker + Gate 4 dead-recovery. Revisit trigger: production multi-pipeline dead-worker recovery rate. Risk: dead worker stays dead until pipeline restart.
- **F90 A17** — router `do_proxy except` broadening. Revisit trigger: production router metadata parse exceptions (e.g., `data` not dict → AttributeError). Risk: metadata injection silently drops on AttributeError, multi_turn raises `RLixRouterMetadataError` instead.
- **F91 graceful actor drain** — replacing `ray.kill` with `actor.shutdown()` RPC + 30s+force-kill. Revisit trigger: multi-pipeline cleanup race observed. Risk: `ray.kill` interrupts in-flight requests.
- **F92 M11.6 cuda_ipc adapter** — NEW adapter (CPU cache → H2D staging → IPC handle, ~50–80 lines) + smoke-test capability check. Revisit trigger: M11.6 production cluster (with `--ipc=host` / `CAP_SYS_PTRACE`). Risk: MVP locked to cpu_serialize; cuda_ipc is baseline performance.
- **F93 M11.4 LoRA + multi-stream** — LoRA cpu_serialize/cuda_ipc adapter + F9 multi-stream impl. Revisit trigger: M11.4 milestone. Risk: M11.1 hook signature already preserves nullable fields (F106) for forward-compat.

Layer 4 (trigger-bound, non-time-bound):

- **F94 A7** — F12 round-trip identity self-check / scheduler allocation cross-check. Revisit trigger: A18 unblock (non-contiguous mapping activated). Risk: under contiguous mapping it's a dead assert; needs `scheduler.get_allocation()` public API which doesn't exist yet.
- **F95 A18** — non-contiguous / custom-ordered `infer_device_mapping` adapter. Revisit trigger: cross-node engine or custom GPU ordering use case. Risk: F12 keeps C6 contiguous-mapping fail-fast in MVP.

## 10. Explicitly Not Doing

### F96 — PD disaggregation

- Why not now: not supported in any milestone; F10 fail-fast `not has_pd_disaggregation` (C9).
- Evidence to revisit: separate plan for PD support (large scope).
- Prevent accidental implementation: F10 startup assert.

### F97 — vLLM backend

- Why not now: only SGLang in scope.
- Evidence to revisit: separate plan if MILES adopts vLLM for inference.
- Prevent: code review.

### F98 — MoE / EP support in F4 cache

- Why not now: F4 cache covers dense Megatron only; full MoE port needs expert TP/EP gather, EP-aware metadata, EP-aware NCCL groups, receiver-side MoE loader, dedicated parity gate (~200-400 lines, separate plan).
- Evidence to revisit: business need for MoE model under RLix.
- Prevent: F10 fail-fast `expert_model_parallel_size == 1 AND moe_router_topk == 0` (C8).

### F99 — `sglang_data_parallel_size > 1`

- Why not now: adds DP axis the port doesn't handle.
- Evidence to revisit: future-bound plan for DP routing.
- Prevent: F10 fail-fast `sglang_data_parallel_size == 1` (C4).

### F100 — Multi-LoRA / DPO / SFT

- Why not now: DPO/SFT entirely out; LoRA = M11.4 (F93).
- Evidence to revisit: M11.4 milestone (LoRA only).
- Prevent: code review.

### F101 — Request-level deterministic migration

- Why not now: MILES uses turn-level redispatch (NeMo F3 form); request-level (ROLL `RequestScheduler`) is mismatched complexity.
- Evidence to revisit: turn-level retry insufficient for some agentic workload.
- Prevent: §3.1 Out of Scope.

### F102 — Cross-node single rollout engine

- Why not now: `WorkerPlacement.placement_group` is per-node; no later milestone re-enables cross-node engine.
- Evidence to revisit: never planned to revisit (§关键风险).
- Prevent: F10 M3 assert `rollout_num_gpus_per_engine <= num_gpus_per_node` (C7).

### F103 — `RadixTreeMiddleware` + `partial_rollout + radix_tree`

- Why not now: radix middleware does internal abort retry / sample state rewrite that hides scheduler-preempt; `partial_rollout + radix_tree` is follow-up after turn-level redispatch is stable.
- Evidence to revisit: production stability of M11.1/M11.2; demand for radix-tree prefix cache reuse with partial rollout.
- Prevent: F10 fail-fast `'RadixTreeMiddleware' not in args.miles_router_middleware_paths` (C17).

### F104 — Subclassing `PipelineCoordinator`

- Why not now: secondary label of F09 P0-FORBIDDEN; manual init only.
- Prevent: F09 detail block.

### F105 — Branching RLix vs standalone inside `train_async.py`

- Why not now: dual-entry (`train_async.py` standalone + `examples/rlix/run_miles_rlix.py` RLix) decouples paths cleanly.
- Prevent: F41 + F71 fail-fast guards.

(No `P4-POLISH` items separately enumerated; the plan is well-scoped and contains no purely cosmetic work.)

## 11. Non-blocking Questions / Low-confidence Defaults

Non-blocking deltas and low-confidence defaults. Three populating sources:

1. Items with `Confidence: Low` in §3 Category Table.
2. Non-`ALIGNED` Scope Delta rows where `MVP set impact: no`.
3. `CATEGORY_REFRAMED` rows (always non-blocking by definition).

| QID | Feature / Item | Question | Suggested default | Affects | Reason |
|---|---|---|---|---|---|
| NQ11.1 | F66 tmpfs naming convention | Source plan describes `miles_cpu_bucket_{uuid}.pt` naming in §F4 §B but doesn't classify in §3.1. Triage: P1-OBSERVABILITY-FLOOR. Patch source plan §3.1 Layer 7 to add it, or accept implicit Layer 7 hygiene? | Accept implicit Layer 7 hygiene; optionally patch source plan §3.1 in a future revision. Implementation already specified. | F66 row in §3 + §2 (`MISSING_SOURCE_SCOPE`) | Doesn't affect MVP membership; cheap observability hint. |
| NQ11.2 | F76 `GET /admission_state` test diagnostic | Source plan introduces this endpoint in §Gate 4 (f) Step P1 but does NOT enumerate it in §3.1 Layer 8 (alongside `MILES_DEBUG_VERIFY_MODEL=1`). Patch source plan §3.1 Layer 8 to enumerate it explicitly? | Yes — patch source plan §3.1 Layer 8 to list `MILES_ROUTER_TEST_HOOKS=1` alongside `MILES_DEBUG_VERIFY_MODEL=1`. Implementation continues unchanged either way. | F76 row in §3 + §2 (`MISSING_SOURCE_SCOPE`) | Plan inconsistency — endpoint is dev-only by intent but not catalogued where dev-only is supposed to live. |
| NQ11.3 | F77 B1 test-side `asyncio.wait_for` wrapper | Plan classifies as §3.1 Layer 7 ✅ B1 ("MVP Lightweight Defensive"); plan also explicitly says "not a production safeguard". Triage classifies as `DEV-ONLY` (test-only wrapper). Both place the item in MVP — the disagreement is about category framing, not membership. Accept the `CATEGORY_REFRAMED` reframing? | Accept — both classifications agree the wrapper ships in MVP for Gate 4 (f); the `DEV-ONLY` label captures that it's test scaffolding rather than production receiver-API surface. | F77 row in §3 + §2 (`CATEGORY_REFRAMED`) | `CATEGORY_REFRAMED` delta by definition; non-blocking; recorded for audit visibility. |
| NQ11.4 | F107 X2 ctor handle injection | Plan classifies as §3.1 Special "Boundary kept" (NOT Layer 1-8); triage classifies as P0-CORRECTNESS-FLOOR (architectural ownership boundary; alternative `ray.get_actor` is Layer 1 Forbidden). Same MVP set, different category framing. Accept Special-class treatment? | Accept Special-class — plan's Boundary-kept rationale (architectural, not Layer 1-8) is sound. Triage P0-CORRECTNESS-FLOOR cross-listing in §5 is for traceability, not relabeling. | F107 row in §3 + §2 + §5 | Confidence medium because Special-class is plan's deliberate design choice; non-blocking. |
| NQ11.5 | F108 C1 `MilesPipeline` actor `max_concurrency` | Plan tentatively sets `=2` and defers final decision via §Audit Checkpoints (Phase 1 + Phase 2 audit). Default value if implementation lands before audit checkboxes are checked off? | `=2` (per plan §3.1 Special C1 tentative); audit checkpoints SHOULD be checked off before implementation. | F108 row in §3 + §2 + §Audit Checkpoints | Plan explicitly defers decision; `=2` is conservative (allows inbound monitor RPC alongside main loop); non-blocking. |

In `SCOPE_REVIEW_READY_WITH_DEFAULTS` state (this artifact's state), the human MUST explicitly accept these defaults (or patch them into the source plan) before invoking a coding agent — see §14 precondition callout.

## 12. Drift / Missing Constraints

- **Missing source-plan classification (NQ11.2)**: F76 `GET /admission_state` test diagnostic introduced in §Gate 4 (f) but not enumerated in §3.1 Layer 8 (alongside `MILES_DEBUG_VERIFY_MODEL=1`). Action: patch source plan §3.1 Layer 8 to list explicitly.
- **Missing source-plan classification (NQ11.1)**: F66 tmpfs naming convention (`miles_cpu_bucket_{uuid}.pt`) is defined in §F4 §B as operational hygiene but isn't classified in §3.1. Triage assigned P1-OBSERVABILITY-FLOOR. Action: optionally add §3.1 Layer 7 row.
- **No TLDR/source drift detected**: TLDR was regenerated immediately before this scope-triage run (current revision = source plan). All AC IDs (AC1–AC13), constraint IDs (C1–C23), assumption IDs (A1–A11), decision IDs (D0–D6.X), and evidence IDs (E1–E14) match across both artifacts.
- **No missing AC coverage**: every AC1–AC13 has at least one covering scope item per §4.
- **No unanchored P0 items**: every P0-FORBIDDEN and P0-CORRECTNESS-FLOOR row has explicit anchor (§3.1 Layer 1 / Cn / ACn / Dn / En cite).
- **Stop conditions gap (carried from TLDR §6.2)**: source plan defines no stop condition for M11.2 multi-pipeline race scenarios (admission_epoch race, donor-shrink race) — those are tagged "M11.5 follow-up, not Gate 4 pass criterion". Action: human reviewer confirms accepting "M11.5 follow-up; manual `ray stop` recovery" as the agent's policy for M11.2 MVP, OR adds explicit stop conditions to source plan §6.2.

## 13. Recommended Implementation Order

1. **P0 forbidden guards** (assert / reject / deny — F01–F14): set up code-review checklist; add asserts where applicable (F09, F10) at startup; document in CLAUDE.md / contributor guide.
2. **P0 correctness floor** (F15–F41, F107): implement in feature dependency order:
   - First: F35 (F10 startup asserts) + F41 (DO_TIME_SHARING dual entry) — establish startup gates.
   - F23, F24 (F2 abort-drain-sleep + EngineInfo state machine) — sleep/wake correctness foundation.
   - F18, F22 (cache_owner discovery + init Step 6.6 pre-registration) — bootstrap ordering.
   - F17 (F4 sequence) — cache build order.
   - F25, F26, F27, F28 (NCCL TCP rendezvous + master_port + writer_lock + tmpfs lifecycle) — transport correctness.
   - F19, F20, F21, F107 (active set bootstrap + locks + per-bucket no-version + service register) — sync atomicity.
   - F15, F16, F40 (D3.D atomic unit + base v=-1 + M11.2 expand path).
   - F29, F30, F31, F32 (multi-turn turn-redispatch).
   - F33, F34 (RLix path placement details).
   - F36 (M4 cleanup).
   - F37, F38, F39 (M11.2 happy path).
3. **P1 happy path** (F42–F64): implement after P0 correctness floor; can parallel-track per Feature (F42–F47 = F1-F3 surface; F48–F53 = F4 surface; F54, F55 = F5+6; F56, F57 = F7-F8; F58, F59 = F9; F60, F61, F62 = F12; F63, F64 = F2 helpers).
4. **P1 observability floor** (F65, F66, F67): add as each transport/cleanup feature lands.
5. **P1 lightweight defensive** (F68–F74, F106, F108): add inline with corresponding production code (e.g., F70 master_port goes with F26).
6. **DEV-ONLY** (F76, F77, F78): only what's needed to gate-validate (Gate 4 (f) needs F76 + F77; Gates 1/2.5/3 covered by F78). F75 is purely developer convenience; defer until first time it's needed.
7. **P2 release blockers**: none in this plan; M11.5 production hardening is the release-stage equivalent.
8. **P3 deferred hardening** (F79–F95): do NOT pull into MVP without explicit user approval. Each item documented with revisit trigger in §9.

Sequence rationale: P0 correctness floor cannot be skipped; P1 happy path has concrete dependency ordering driven by F22 (init bootstrap pre-registration is the only path through Phase 5/6 ordering).

## 14. Post-Review Handoff Draft (copy-paste, conditional on human review)

**This handoff is conditional on human review.** It does NOT auto-fire when the artifact is written.

> **Precondition for `SCOPE_REVIEW_READY_WITH_DEFAULTS`**: the human reviewer MUST accept §11 defaults (or patch them into the source plan) before passing this block to a coding agent. The state name signals "human review can proceed" — NOT "agent ready". Defaults that have NOT been explicitly accepted leave this artifact in an unbound state — the agent has no authority to act on them.

```text
PRECONDITION (HUMAN ONLY) — before passing this block to a coding agent, confirm:
- §1 Blocking Questions: none in this artifact (resolved)
- §2 Scope Delta Matrix: 5 non-`ALIGNED` rows, all `MVP set impact: no` —
  F66 (`MISSING_SOURCE_SCOPE`, tmpfs naming not in §3.1; NQ11.1)
  F76 (`MISSING_SOURCE_SCOPE`, /admission_state not in §3.1 Layer 8; NQ11.2)
  F77 (`CATEGORY_REFRAMED`, plan-Layer-7 vs triage-DEV-ONLY; NQ11.3)
  F107 (`MISSING_SOURCE_SCOPE`, X2 plan-Special vs triage-P0-CORRECTNESS-FLOOR; NQ11.4)
  F108 (`MISSING_SOURCE_SCOPE`, C1 plan-evidence-deferred; NQ11.5)
  All 5 accepted via §11 defaults; none affect MVP membership.
- §4 AC Coverage Check: all 13 ACs (AC1–AC13) covered; AC13 has 2 covering
  items with non-`ALIGNED` deltas (F76, F77) but both `MVP set impact: no`.
- §11 Non-blocking defaults: NQ11.1 → accept implicit Layer 7,
  NQ11.2 → patch source plan §3.1 Layer 8 (recommended) OR accept,
  NQ11.3 → accept reframing,
  NQ11.4 → accept Special-class,
  NQ11.5 → accept tentative `=2`.

If any of the above is unresolved, do NOT use this handoff. Resolve in source plan and rerun scope-triage.

You are implementing the feature described in plans/miles-port-unified-plan.md.

Source of truth: the source plan above.
Boundary control: plans/miles-port-unified-plan.scope.md

Before writing code:
1. Read the source plan in full (plans/miles-port-unified-plan.md).
2. Read the scope file's §3 Category Table and §5 P0 Must-Address.
3. Implement only items classified P0-CORRECTNESS-FLOOR (28 items, §5) and P1-* (35 items: 23 happy-path + 3 observability + 9 lightweight-defensive) in the scope file.
4. Add P0-FORBIDDEN guards (asserts / rejects) at the boundaries the scope file identifies (14 items, F01–F14).
5. Add the minimum observability listed under P1-OBSERVABILITY-FLOOR (F65 SGLang server-side VRAM read; F66 tmpfs naming; F67 atomic-unit call counters).
6. Add only the P1-LIGHTWEIGHT-DEFENSIVE checks listed; do not invent additional defenses.
7. Add DEV-ONLY scaffolding only if §7 lists it as needed for MVP (F76 + F77 for Gate 4 (f); F78 unit tests; F75 only if developer enables MILES_DEBUG_VERIFY_MODEL=1).
8. `Scope review state:` is SCOPE_REVIEW_READY_WITH_DEFAULTS — you MUST have explicitly accepted §11 defaults (NQ11.1 through NQ11.5) before this handoff is valid. Defaults do NOT auto-apply.

Implementation order: follow §13 (P0 forbidden guards → P0 correctness floor in dependency order → P1 happy path per-Feature → P1 observability/defensive inline → DEV-ONLY only as needed for gates).

Do NOT:
- Implement any item classified NO-OVERENGINEERING (F96–F105, plus F104 cross-ref to F09).
- Pull P3-DEFER-HARDENING items into MVP without explicit approval from the user (F79–F95). In particular, do NOT add bounded `_use_url` timeout / 503 sentinel / client `EnginePreemptedError` translation to M11.2 — that's M11.5 (F79).
- Generalize beyond the explicit source-plan scope.

Stop and ask if:
- Implementation requires crossing a P0-FORBIDDEN boundary (especially F03 `_preempted_engines` re-promotion, F04 split run_sync_session RPC, F05 calling finalize/set_weight_version outside service, F07 raise from _use_url, F09 subclass PipelineCoordinator).
- A required item is missing both source plan and scope file (§12 flags F66/F76 tmpfs/admission_state classification gaps; verify these don't escalate to MVP-affecting before implementation).
- PLAN.md (source plan) and PLAN.scope.md (this scope file) disagree on any item's scope/category. Do NOT let PLAN.scope.md silently override PLAN.md — Scope Delta Matrix entries with non-`ALIGNED` deltas surface real plan disagreement that the human must resolve in PLAN.md (or explicitly accept).
- Implementation pulls a P3 or NO-OVERENGINEERING item into MVP path.
- Assumptions A1 / A5 / A6 / A11 fail during implementation (see source plan §6.2 stop conditions).
- Atomic-unit invariant (F15) cannot be preserved without splitting RPCs.
- Base `version=-1` (F16) requires hf-checkpoint equivalence shortcut.
- C20 router suspend (F39) cannot be implemented without making sync helpers async (F14 forbids).
```

## Appendix I: Full Category Table

Every extracted item; rows summarized in §3 are listed here in full per the large-plan strategy.

| ID | Item | Category | Priority | Action | Anchors | Confidence | Reason |
|---|---|---|---|---|---|---|---|
| F01–F14 | (See §3 — P0-FORBIDDEN, 14 items, all Confidence: high) | P0-FORBIDDEN | M11.1/M11.2 guard | assert / not implement / code review | §3.1 Layer 1 | high | (per-row in §3) |
| F15–F41, F107 | (See §3 — P0-CORRECTNESS-FLOOR, 28 items, F107 medium, others high) | P0-CORRECTNESS-FLOOR | M11.1/M11.2 P1 | implement | (per-row in §3) | high (medium for F107) | (per-row in §3) |
| F42 | F1 SGLang sleep/wake tag-based API (`release_memory_occupation(tags=None)` = release weights+kv_cache+cuda_graph; vLLM `level=2` equivalent) | P1-HAPPY-PATH | M11.1 | implement | AC3, F1 | high | F1 plan's primary feature; tag API is SGLang's vLLM-equivalent |
| F43 | F2 `shrink_engines` / `expand_engines` / `activate_routing` / `finish_init_offload` composite ops on `RolloutManager` | P1-HAPPY-PATH | M11.1 | implement | AC3, AC10, F2 | high | composite ops encapsulate compound operation invariant |
| F44 | F3 router admission API: `/disable_worker?url=` / `/enable_worker?url=` / `/remove_worker?url=` / `/add_worker?url=&engine_index=` | P1-HAPPY-PATH | M11.1 | implement | AC3, AC6, F3 | high | per-engine admission is routing source of truth |
| F45 | F3 router 4-dict worker lifecycle (`enabled_workers` / `dead_workers` / `worker_request_counts` / `worker_failure_counts` / `worker_engine_index_map`); `_use_url` selects from `enabled - dead` | P1-HAPPY-PATH | M11.1 | implement | D4.A, F3 §(b.4) | high | URL is the natural key; `_use_url` MUST really filter |
| F46 | F3 multi_turn `_snapshot_turn_state` / `_restore_turn_state` (5-field length record + truncate); `payload["stream"] = False` forced | P1-HAPPY-PATH | M11.1 | implement | D3.B, AC6 | high | length-only is `O(increment)`; deep-copy is quadratic |
| F47 | F3 fully_async `_FatalError` queue sentinel single-path: `task_done_callback` catches `(EnginePreemptedError, RLixRouterMetadataError)` → `output_queue.put((gid, _FatalError(exc)))`; main loop dequeue checks `isinstance(result, _FatalError)` → `raise result.exc` | P1-HAPPY-PATH | M11.1 | implement | A6 anti-regression; D3.B | high | `_FatalError` is sentinel CLASS (not Exception) so outer try/except cannot swallow |
| F48 | F4 `build_cpu_bucket_cache(step)` HF-format gather: pinned CPU buckets; `named_tensors` HF-converted; single slot covering writes; first-call lazy buffer alloc | P1-HAPPY-PATH | M11.1 | implement | M5, D2.B, F4 | high | raw Megatron state_dict won't load on SGLang receiver |
| F49 | F4 cpu_serialize transport: wrapper `ray.put(payload_bytes) → ref → engine.update_weights_from_cpu_bucket.remote(ref, ...)` → Ray auto-deref → wrapper writes `/dev/shm/miles_cpu_bucket_{uuid}.pt` → HTTP POST `payload_path` to SGLang server | P1-HAPPY-PATH | M11.1 | implement | D5.A, AC4, F4 §B | high | tmpfs path avoids base64 HTTP body bloat + subprocess `ray.init` for ObjectRef ID lookup |
| F50 | F4 NCCL broadcast non-colocate path: per-bucket H2D staging on cache_owner GPU → `dist.broadcast(staging, src=cache_owner_global_rank, group=tmp)` → free staging; per-bucket barrier before staging free | P1-HAPPY-PATH | M11.1 | implement | D5.B, AC4 | high | NCCL doesn't support CPU tensors; per-bucket free keeps peak VRAM = 1× bucket_size |
| F51 | F4 `cache_owner_actor.run_sync_session(plan)` single composite Ray RPC (M2): cache_owner holds `_cache_lock` for whole transport phase | P1-HAPPY-PATH | M11.1 | implement | D3.D, D4.E, M2 | high | cache_owner is independent Ray actor — service must RPC; single composite dissolves lock-across-RPC matrix |
| F52 | F4 `SyncSessionPlan` frozen dataclass (sync_id / version / group_name / master_addr/port / timeout / target_handles / cpu_serialize_local_ranks / broadcast_local_ranks / comm_ranks); built once at sync entry | P1-HAPPY-PATH | M11.1 | implement | D4.E, F.1 | high | frozen + plain types = trivially serializable; cache_owner has everything to drive both transports |
| F53 | F4 dual-mask receivers: `cpu_serialize_local_ranks` (M11.1 colocate) + `broadcast_local_ranks` (always) + `ipc_local_ranks` (M11.6); `is_group_exist` no-op guard on `destroy_collective_group` for colocate-only ranks | P1-HAPPY-PATH | M11.1 | implement | F.3, AC4 | high | mixed-mask sync is M11.1 load-bearing for tp>1 partial-overlap |
| F54 | F5+6 `coordinator.sync_base_weights_to_active(step)` — under `_resize_sync_lock` updates `_cache_ready_step = step` then calls `service.sync_selected_workers(sync_id, _active_engine_indices, step)` atomic unit | P1-HAPPY-PATH | M11.1 | implement | D3.C, D4.C, F5+6 | high | step MUST flow through coordinator within single critical section |
| F55 | F5+6 `coordinator._expand_workers(target_engines)` — under `_resize_sync_lock` with `_cache_ready_step is not None` guard; calls `manager.expand_engines` → `service.sync` → `manager.activate_routing` | P1-HAPPY-PATH | M11.1+M11.2 | implement | D3.E, AC11, AC13 | high | coordinator is sole orchestrator of expand; pipeline doesn't drive expand path |
| F56 | F7 Per-pipeline Ray namespace isolation — 3 named RLix actors (MilesCoordinator / MilesPipeline / MilesModelUpdateService) carry `namespace=` and `name=`; `runtime_env` propagates to anonymous MILES child actors | P1-HAPPY-PATH | M11.1 | implement | F7, AC9 | high | named actors must be explicit; MILES framework actors stay anonymous |
| F57 | F8 Driver lifecycle: `orchestrator.allocate_pipeline_id("ft")` → `register_pipeline(pipeline_id, ray_namespace, cluster_tp_configs, cluster_device_mappings)` → `admit_pipeline(pipeline_id)` → `MilesCoordinator.options(...).remote(pipeline_id=, pipeline_config=)` (keyword-only) → `coordinator.create_pipeline_actor.remote(pipeline_config=)` (keyword-only) | P1-HAPPY-PATH | M11.1 | implement | F8, AC9 | high | RLix orchestrator API is keyword-only; positional calls TypeError mid-init |
| F58 | F9 Progress reporting: reporter holds `_local_completed` (group counter, NOT trajectory) + `_progress_target_step` + `_progress_last_bucket`; 2% bucket gate at reporter; `begin_progress_batch(target_weight_version, step_target_groups, initial_completed=0, mode=None, adapter_id=None)` opens window with first `new_batch=True` snapshot; `bump_completed` only emits on bucket change; `end_progress_batch` in `finally` | P1-HAPPY-PATH | M11.1 | implement | F9 | high | trajectory-unit reports under-count by `n_samples_per_prompt` → premature shrink |
| F59 | F9 `RLixHooks` protocol + `NoOpRLixHooks` import seam in `miles/utils/rlix_hooks.py`; fully_async only depends on hook protocol (NEVER imports `ProgressReport` or `ray.get_actor` for coordinator) | P1-HAPPY-PATH | M11.1 | implement | F9 hooks decoupling | high | direct import of RLix wire types breaks standalone build |
| F60 | F12 `MilesPlacementProvider` real adapter: receives injected `RollResourceManagerProxy` + injected `train_device_mapping` / `infer_device_mapping` (both same-source as F8 driver); calls `proxy.allocate_placement_group(world_size, device_mapping=declared)` → `WorkerPlacement` per-worker view | P1-HAPPY-PATH | M11.1 | implement | D1.B, AC9, F12 | high | self-instantiating proxy double-inits and breaks shared PG |
| F61 | F12 `RayTrainGroup` / `RolloutManager` `worker_placements` path: `num_gpus_per_actor=0.01` (RLix) — actual GPU isolation via `PlacementGroupSchedulingStrategy + capture_child_tasks + CUDA_VISIBLE_DEVICES + NOSET_VISIBLE_DEVICES_ENV_VARS_LIST` at `.options(runtime_env={'env_vars': {...}})` | P1-HAPPY-PATH | M11.1 | implement | D5.F, F12 | high | standalone fractional `0.4` deadlocks RLix Gate 4 |
| F62 | F12 `RolloutManager.__init__` extended with `all_engine_placements: list[WorkerPlacement]` (length=engine_count, declared full table) + `active_engine_indices: frozenset[int]` (M11.2 init = `frozenset()`); ctor builds shell metadata slots + starts empty Miles router; SGLang actors NOT spawned by ctor in RLix mode | P1-HAPPY-PATH | M11.1+M11.2 | implement | D4.A, AC10, F12 | high | shell ctor + lazy `expand_engines` is M11.2 init pattern; `activate_routing` is sole `add_worker` entry |
| F63 | F2 `SGLangEngine.abort_all_requests()` — POST `/abort_request {"abort_all": true}` (`http_server.py:1402`) | P1-HAPPY-PATH | M11.1 | implement | D3.A, F2 | high | SGLang `assert is_fully_idle()` requires preceding abort + drain |
| F64 | F2 `SGLangEngine.is_idle()` — GET `/v1/loads`; checks `all(slot["num_total_reqs"] == 0)`; timeout=5.0 + `raise_for_status()` | P1-HAPPY-PATH | M11.1 | implement | A7, D3.A | high | `/server_info.internal_states` lacks `num_running_reqs`; silent treat-as-idle on 5xx → release while in-flight → assert fires |
| F65 | F1/F2 post-sleep VRAM assert via SGLang `/server_info` `memory_usage` GB (server-side, server child process); threshold `args.miles_post_sleep_vram_threshold_gb` (default 1.0) | P1-OBSERVABILITY-FLOOR | M11.1 | implement | F1, AC3, Anti-regression invariant #8 | high | actor `torch.cuda.memory_allocated()` always ~0; `/server_info` GB units (NOT MB) is ground truth |
| F66 | tmpfs file naming convention `miles_cpu_bucket_{uuid}.pt` (grep-friendly leak detection) | P1-OBSERVABILITY-FLOOR | M11.1 | implement | §F4 §B tmpfs lifecycle | medium | low cost; surfaces leaks operationally — but plan doesn't classify it under §3.1 (see §11 NQ11.1) |
| F67 | Per-bucket / per-engine instrumented call counters for atomic-unit audit: `cache_owner_actor.run_sync_session` count, per-engine `finalize_weight_update` count, `set_weight_version` count (E6) | P1-OBSERVABILITY-FLOOR | M11.1+M11.2 | implement | E6, Gate 4 (d) | high | atomic-unit invariant audit requires concrete counters |
| F68 | Layer 7 ✅ A16-cheap: `dead_workers.discard(url)` in `_add_worker_internal` (1 line) + `worker_failure_counts[url] = 0` in `_disable_worker_internal` (1 line) | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement | §3.1 Layer 7 ✅ A16-cheap | high | 2 lines total; covers re-registration / sleep invariant |
| F69 | Layer 7 ✅ X3 boundary: `_active_engines_bootstrapped` flag (cross-ref F19) | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement (cross-ref F19) | §3.1 Layer 7 ✅ X3 | high | one bool's cost for one detectable bug |
| F70 | Layer 7 ✅ B3 default path: `master_port != 0` + SharedStorage `MASTER_ADDR_PORT:*` claim per sync + EADDRINUSE retry exhausted → fail fast | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement | §3.1 Layer 7 ✅ B3 | high | concrete port + atomic claim + retry-then-fail-fast; cooldown/pool deferred (A5) |
| F71 | F11 standalone partial-overlap fail-fast guard at top of `train_async.py`: `if train_devices_subset_of_infer(args) and not DO_TIME_SHARING: raise` | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement | F11, AC8 | high | standalone path is unchanged; this guard catches user misuse without modifying standalone semantics |
| F72 | F3 router header hardening: `do_proxy` mutating `/generate` body MUST `pop` `Content-Encoding` (upper + lower); `Content-Length` recomputed by `build_proxy_response` JSONResponse | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement | D4.B | high | gzip body + mutated body with stale `Content-Encoding: gzip` header → client decode crash |
| F73 | F4/F10 startup `/dev/shm` capacity check: `os.path.isdir + os.access(W_OK)` + `shutil.disk_usage("/dev/shm").free >= bucket_size + 256 MB` | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement | A4, F10 §S3a-2 | high | Docker default `/dev/shm = 64 MB` will silently OOM mid-sync |
| F74 | F12 round-trip structural validation (NOT identity self-check): `len(all_placements) == engine_count`, `len(wp.gpu_ids) == tp`, `wp.gpu_ids == sorted(wp.gpu_ids)` | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement | F12 (d), §3.1 Layer 4 A7 (deferred identity check) | high | structural assert catches `WorkerPlacement` shape bugs at startup |
| F75 | A9 `verify_model` hash-based weight validation gated by `MILES_DEBUG_VERIFY_MODEL=1` | DEV-ONLY | M11.1 | keep behind flag | §3.1 Layer 8 ✅ A9 | high | per-bucket barrier + warmup allreduce is the actual MVP verification |
| F76 | `GET /admission_state` test diagnostic — gated `MILES_ROUTER_TEST_HOOKS=1` | DEV-ONLY | M11.2 (Gate 4 (f)) | implement; isolate via env flag; document plan-source location | source plan §Gate 4 (f) Step P1 | medium | router runs as separate FastAPI process; see §11 Q11.2 |
| F77 | B1 test-side `asyncio.wait_for(test_coro, timeout=N)` wrapper for Gate 4 (f) sub-tests | DEV-ONLY (secondary: P1-LIGHTWEIGHT-DEFENSIVE per plan Layer 7) | M11.2 (Gate 4 (f)) | implement test-side only | §3.1 Layer 7 ✅ B1; §Gate 4 (f) | medium | plan classifies as Layer 7 lightweight defensive but explicitly notes "not a production safeguard"; see §11 NQ11.3 |
| F78 | New tests: `tests/test_partial_sleep_wake.py` (F1-F3 unit) + `tests/test_miles_pipeline.py` (F4-F6 integration) | DEV-ONLY | M11.1 | implement | §文件改动总清单 §测试 | high | unit test scaffolding; not production code |
| F79–F95 | (See §3 — P3-DEFER-HARDENING, 17 items) | P3-DEFER-HARDENING | M11.5 / M11.6 / M11.4 / trigger-bound | defer | (per-row in §3) | high (medium F85, F88, F90) | (per-row in §3) |
| F96–F105 | (See §3 — NO-OVERENGINEERING, 10 items) | NO-OVERENGINEERING | — | not implement | (per-row in §3) | high | (per-row in §3) |
| F106 | X1 — `mode` / `adapter_id` nullable fields in `begin_progress_batch` (forward-compat for M11.4 LoRA multi-stream) | P1-LIGHTWEIGHT-DEFENSIVE (secondary: P4-POLISH) | M11.1 | keep | §3.1 Special X1 | high | wire-protocol form; nullable parameter is zero runtime cost |
| F107 | (See §3 — X2 ctor handle injection) | P0-CORRECTNESS-FLOOR | M11.1 | implement | §3.1 Special X2 | medium | (per-row in §3 + §5) |
| F108 | (See §3 — C1 max_concurrency) | P1-LIGHTWEIGHT-DEFENSIVE | M11.1 | implement default `=2` | §3.1 Special C1; §Audit Checkpoints | medium | (per-row in §3) |

Total rows: 108.
