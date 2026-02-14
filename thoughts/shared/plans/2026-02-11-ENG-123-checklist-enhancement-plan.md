# ENG-123 Checklist Enhancement Plan

**Date**: 2026-02-11  
**Source Documents**:
- Extraction Plan: `thoughts/shared/plans/2026-02-05-ENG-123-roll-multipipeline-extraction.md`
- Gap Analysis: `thoughts/shared/plans/2026-02-11-ENG-123-checklist-gap-analysis.md`

## Executive Summary

This document provides a **comprehensive checklist** that includes:
1. **All original checklist items** from lines 1316-1412 (preserved exactly)
2. **All new items** identified in the gap analysis (marked with 🆕)

Items are organized by execution phase for practical implementation. The original checklist covers approximately **12%** of detailed requirements. This enhanced checklist adds **148 new items** to achieve comprehensive coverage.

---

## Phase 1 Checklist — SchedRL Core Package Skeleton

**Primary goals**: create job-scoped Orchestrator/Scheduler singletons; define canonical protocol + fail-fast semantics; define correct Ray retry knobs; establish namespace strategy hooks (Option A) at the contract layer.

### Original Items (Preserved)

- [ ] **P0 Issue 21 & 215 (ownership)**: Orchestrator owns core components; enforce singleton actor creation.
- [ ] **P1 Issue 25 & 31 (fail-fast)**: implement global fail-fast shutdown path (already by-design; ensure it is enforced, not bypassed).
- [ ] **P1 Issue 29 & 68 (timeouts)**: define phase-specific timeout constants (register/admit/shrink/expand/abort-ACK/offload).
- [ ] **P0 Issue 51 (delimiter collision)**: implement and enforce `validate_pipeline_id` (reject `:`) and canonical `request_id` helpers.
- [ ] **P1 Issue 62, 204 & 206 (config resolution/validation)**: define validation entrypoints (even if full Hydra resolution is implemented later, the error boundaries must be clear in Phase 1).
- [ ] **P1 (namespace override plumbing)**: define `ROLL_RAY_NAMESPACE` contract + `PIPELINE_ID` propagation via Ray `runtime_env.env_vars` (implementation occurs in Phase 3, but the contract must be explicit here).
- [ ] **P1 (Ray retry semantics; correctness)**: remove any mention/usage of non-Ray knobs like `max_retries`; use Ray-supported knobs for "no retry/no replay" (`max_restarts=0`, tasks `max_task_retries=0`, avoid implicit retries).

### 🆕 Module/File Creation

- [ ] 🆕 **Create `schedrl/protocol/types.py`**: Dataclasses for IDs, ModelMode enum (Lines 1429-1441)
- [ ] 🆕 **Create `schedrl/protocol/actions.py`**: Action dataclasses for lifecycle RPCs (Lines 1429-1441)
- [ ] 🆕 **Create `schedrl/protocol/validation.py`**: Validation entrypoints (Lines 1429-1441)
- [ ] 🆕 **Create `schedrl/protocol/request_id.py`**: `build_request_id`, `parse_request_id`, `validate_request_id` helpers (Lines 1443-1450)
- [ ] 🆕 **Create `schedrl/protocol/adapter.py`**: Adapter protocol/base class (Lines 1463-1466)
- [ ] 🆕 **Create `schedrl/client/client.py`**: Client connect/get-or-create logic (Lines 1463-1466)
- [ ] 🆕 **Create `schedrl/scheduler/scheduler.py`**: Scheduler actor implementation (Lines 1602-1607)
- [ ] 🆕 **Create `schedrl/scheduler/state.py`**: Scheduler state management (Lines 1602-1607)
- [ ] 🆕 **Create `schedrl/scheduler/executor.py`**: Action executor (Lines 1602-1607)
- [ ] 🆕 **Create `schedrl/scheduler/run.py`**: Scheduler run loop (Lines 1602-1607)
- [ ] 🆕 **Create `schedrl/scheduler/resource_manager.py`**: Resource manager (actor or module) (Lines 1609-1610)

### 🆕 Functional Requirements

- [ ] 🆕 **Strict Affinity**: Pin ALL SchedRL system actors to Ray head node using `NodeAffinitySchedulingStrategy` with **soft=False** to ensure fail-fast behavior if head node is unreachable (Lines 1474-1511)
- [ ] 🆕 **Client connection flow**: `connect(create_if_missing=True)` returns orchestrator handle; `admit_pipeline(...)` is blocking and returns scheduler handle (Lines 1513-1522)
- [ ] 🆕 **Orchestrator RPC surface**: Implement full list: `register_pipeline`, `admit_pipeline`, `get_pipeline_state`, `monitor_pipelines`, `cleanup_pipeline`, `kill_pipeline`, `shutdown` (Lines 1524-1535)
- [ ] 🆕 **Shutdown semantics**: Execute `ray stop --force` on ALL nodes; zombie process prevention env vars; idempotent behavior (Lines 1536-1600)
- [ ] 🆕 **Platform independence**: Do not import framework-specific platform helpers in schedrl; inject `PlatformConfig` if needed (Lines 1612-1613)
- [ ] 🆕 **Issue 80 validation**: `register_pipeline` must validate GPU IDs are subset of `total_gpus` (Lines 1615-1616)
- [ ] 🆕 **Minimum scheduler behavior**: In-memory state per pipeline; execution path for Adapter RPCs in strict order (Lines 1621-1626)
- [ ] 🆕 **Timeout sentinel values**: Set `schedrl.abort_timeout_secs = -1` (disabled for ENG-123); all `schedrl.*timeout*` fields use `-1` sentinel (Lines 1451-1461)
- [ ] 🆕 **ModelMode enum**: Define enum with `FULL_FT` and `MULTI_LORA` values (Line 1450)
- [ ] 🆕 **PlatformConfig dataclass**: Inject platform details (`ray_device_key`, `device_control_env_var`) (Line 1626)
- [ ] 🆕 **Rank-Aware Initialization**: In ROLL framework, `schedrl.init()` must check `rank == 0` before executing; entry script runs on ALL nodes via torchrun (Lines 941-947)

### 🆕 Client APIs (Define in Phase 1)

- [ ] 🆕 **release_and_request API**: Atomic Release+Request in SchedRL Scheduler; single scheduler transaction (Lines 363-379)
- [ ] 🆕 **notify_ready_to_release API**: Blocking Planned Release in SchedRL Scheduler; blocks until Phase 0-6 planning loop reclaims resources (Lines 363-379)
- [ ] 🆕 **Library Mode connect race handling**: Implement connect semantics: get-then-create with backoff (Lines 402-408)
- [ ] 🆕 **Client exposure**: Expose APIs via `schedrl.client` for Coordinator to call (Line 373)

### 🆕 Schemas & Contracts (Define in Phase 1)

- [ ] 🆕 **Release ACK payload schema**: JSON schema with `aborted`, `remapped`, `release_reports` containing `dp_rank`, `gpu_map`, `free_bytes_by_gpu`, `total_bytes_by_gpu`; use `-1` sentinel values for memory fields (Lines 213-233)
- [ ] 🆕 **report_progress schema**: Full schema: `pipeline_id`, `queued_trajectories`, `inflight_trajectories`, `step_target_trajectories`, `metrics` (Lines 234-242)
- [ ] 🆕 **Progress reporting cadence**: Emit at: (1) batch start, (2) 2% band crossings, (3) exactly once when >= 1.0, (4) when sample buffer dequeues (Lines 238-243, 393)
- [ ] 🆕 **Non-monotonic progress handling**: If percent completion resets between steps, treat each step independently (Lines 395-396)
- [ ] 🆕 **Abort ACK semantics**: Define ACK as "targeted request IDs are no longer in-flight (removed from running_requests)"; tolerate "Success" finishes during abort (Lines 115-121)
- [ ] 🆕 **7-Step Shrink-to-Zero Sequence**: Mandatory ordering: (1) set need_suspend=True, (2) close admission, (3) abort/drain, (4) wait ACK, (5) offload/stop, (6) clear routing, (7) return success (Lines 92-99)
- [ ] 🆕 **request_id Helper Logic**: Implement `build_request_id`, `parse_request_id`, `validate_request_id` with format `{pipeline_id}:{traj_id}:{turn_id}:{attempt}`; validator MUST raise ValueError if parsing is ambiguous (Lines 198-202)
- [ ] 🆕 **traj_id validation**: Verify `traj_id` validation logic rejects the delimiter character `:` (Line 207)

### 🆕 Protocol/API Requirements (Define in Phase 1)

- [ ] 🆕 **Implicit sequencing**: Requests define dependencies; scheduler executes one atomic action per pipeline at a time (Line 1289)
- [ ] 🆕 **Strict sequencing**: Scheduler must not issue new lifecycle batch until previous batch ACKed; coordinator/adapter must execute sequentially (Line 1290)
- [ ] 🆕 **register() API**: Registers pipeline's static resource topology (TP sizes, device mappings) before pipeline actor starts (Lines 1293-1294)
- [ ] 🆕 **request_gpus() API**: Blocking Allocation; blocks until requested GPUs are allocated (Lines 1296-1297)
- [ ] 🆕 **release_gpus() API**: Immediate Release; used by stateful clusters (Training, Critic) (Lines 1298-1299)
- [ ] 🆕 **release_and_request() API**: Atomic Release+Request; single scheduler transaction (Lines 1300-1301)
- [ ] 🆕 **notify_ready_to_release() API**: Planned Release (Blocking); blocks until Phase 0-6 planning loop reclaims resources (Lines 1302-1303)
- [ ] 🆕 **report_progress() with fifo_timestamp**: Fairness Hook; passes creation timestamp of oldest waiting episode (Lines 1304-1305)
- [ ] 🆕 **unregister_pipeline() API**: Full pipeline unregistration (Line 1307)
- [ ] 🆕 **Adapter lifecycle RPCs (extraction-plan parity)**: Add `close_admission/open_admission/shrink_workers/expand_workers -> ActionResponse` to `schedrl.protocol.adapter.Adapter`. This is not required by the original Phase 1 checklist text, but is a known divergence from `2026-02-05-ENG-123-roll-multipipeline-extraction.md` (around Line 1262) and should be tracked explicitly here.

### 🆕 Design Decisions / Negative Constraints (Check Early)

- [ ] 🆕 **Do NOT refactor to heapq-based queues**: Keep existing fork per-priority FIFO lists + wakeup loop (Line 130)
- [ ] 🆕 **Do NOT compress priority taxonomy**: Keep existing 7-tier Priority enum; do not compress to 4 tiers (Line 131)
- [ ] 🆕 **Do NOT change API naming**: Keep fork API naming (`register_pipeline`/`admit_pipeline`/`request_gpus`) for ENG-123 (Line 132)
- [ ] 🆕 **Do NOT add min-dp constraints**: Do not add `min_dp_workers_per_pipeline` enforcement in ENG-123 (Line 133)
- [ ] 🆕 **Do NOT add priority boosting**: Keep fork FIFO semantics; do not add priority boosting (Line 140)
- [ ] 🆕 **Do NOT pursue heapq/lock-free refactor**: Keep fork queue implementation; defer refactors (Line 139)
- [ ] 🆕 **No idempotency tokens**: ENG-123 intentionally does NOT implement `activation_epoch`/`action_id`/idempotency tokens (Lines 266-269)
- [ ] 🆕 **No centralized timeout config**: Keep env-var timeouts; centralization + propagation is post-ENG-123 (Line 1217)
- [ ] 🆕 **No post-release GPU memory measurement**: Return -1 sentinel fields in release_reports; add measurement after ENG-123 (Line 1218)
- [ ] 🆕 **Negative Constraint**: Do NOT port fork `RequestScheduler.get_suspend_state()` into upstream. (Lines 1792-1794)
- [ ] 🆕 **Negative Constraint**: Do NOT port fork `check_gen_active_allocation_with_no_dp_workers()` into upstream.  (Lines 1792-1794)

### 🆕 What We're NOT Doing (Reminders)

- [ ] 🆕 **Not multi-framework arbitration**: Only ROLL wired end-to-end (Lines 1209-1222)
- [ ] 🆕 **Not migrating NeMo-RL/Miles**: Deferred to backlog (Lines 1209-1222)
- [ ] 🆕 **Not deleting ROLL_multi_pipeline**: Remains as reference (Lines 1209-1222)
- [ ] 🆕 **Not adding new dependencies**: Third-party Python dependencies (Lines 1209-1222)
- [ ] 🆕 **Not building new tests**: Outside existing suites (Lines 1209-1222)
- [ ] 🆕 **Not MULTI_LORA specifics**: Only reserve protocol/types with TODO (Lines 1209-1222)
- [ ] 🆕 **Not oldest_unfinished_creation_ts**: Requires persistent tracking (Lines 1209-1222)
- [ ] 🆕 **Not implementing scheduler policy refactors**: No heapq-based queues, priority taxonomy changes, or min-dp constraints beyond fork behavior (Line 1216)
- [ ] 🆕 **Negative Constraint**: Do NOT implement Service Mode connect-only semantics or detached actor lifecycle in ENG-123; deferred to backlog (Line 1533)

### 🆕 Infrastructure & Examples (P2 - Setup in Phase 1)

- [ ] 🆕 **P2 Issue 23: SchedRL Launcher Utility**: Two-component approach: Launcher manages Ray cluster lifecycle (`ray stop --force`, `ray start`); Orchestrator connects to existing cluster (Lines 592-605)
- [ ] 🆕 **P2 Issue 101 & 103: Debugging env vars**: Orchestrator accepts `env_vars` dict and propagates via `runtime_env`; Day 1 priority for debugging NCCL issues (Lines 890-898)
- [ ] 🆕 **P2 Issue 223: Legacy Initialization**: Refactor `initialize.init()` to "Connect-First, Fallback-Local" pattern; port entry point; framework integration guide (Lines 925-949)
- [ ] 🆕 **P2 Issue 229: Port working examples**: Port `pipeline1_sokoban_grpo.yaml`, `pipeline2_sokoban_grpo.yaml`, `start_multi_pipeline_test.py` (Lines 994-1000)

### 🆕 Scheduler Recovery (Define in Phase 1)

- [ ] 🆕 **Fail-fast behavior**: No recovery or rehydration; fresh pipeline starts only (Lines 1953-1965)
- [ ] 🆕 **Operational policy**: Full reset on scheduler restart; pipelines re-register (Lines 1953-1965)
- [ ] 🆕 **Config flag**: Implement `schedrl.fail_fast_on_restart = true` config flag (Line 1963)
- [ ] 🆕 **Test requirement**: Simulate scheduler restart; verify re-registration (Lines 1953-1965)

### Validation Milestone (Phase 1)

- [ ] Start a fresh Ray job; create `schedrl:orchestrator` + `schedrl:scheduler`; verify they start once and are discoverable by name.
- [ ] Negative test: invalid `pipeline_id` (contains `:`) fails fast at registration/admission.
- [ ] 🆕 **Manual verification**: ROLL driver connects in library mode
- [ ] 🆕 **Ray Retry Semantics (P1)**: Verify that non-Ray knobs (e.g., `max_retries`) are removed and only Ray-supported knobs (`max_restarts=0`, `max_task_retries=0`) are used

---

## Phase 2 Checklist — Port Scheduler Model into `schedrl` (Policy + State)

**Primary goals**: port the fork scheduling model (planning loop, fairness, progress ingestion) into SchedRL; keep ROLL as mechanics.

### Original Items (Preserved)

- [ ] **H1 (dead assertions)**: replace fork "no-op tuple assertions" with real `assert` statements or explicit `ValueError`/`RuntimeError` (fail-fast). Do not port silent tuple expressions like `(cond, "msg")`.
- [ ] **H2 (cluster_id parsing)**: implement a suffix-aware parser for cluster IDs; never parse `pipeline_id` from `cluster_id` using `rsplit("_", 1)` (use the fork-equivalent `_parse_cluster_id` approach).
- [ ] **H3 (notify_completion atomicity)**: `notify_completion` must be atomic (idempotency check + insert into `pending_completion_requests` under the same scheduler lock).
- [ ] **P0 Issue 107 (state inconsistency)**: any execution-phase failure triggers controlled-fatal shutdown (`shutdown(force=True)`).
- [ ] **P0 Issue 81 & 124 (expansion signaling)**: ensure expansion completion signals cannot be orphaned; on any failure, fail-fast and shut down.
- [ ] **P1 Issue 64 (gap-ratio starvation)**: implement gap-ratio fairness constraints as described (work inflation prevention).
- [ ] **P1 Issue 202 & 216 (proportional rebalancing/session migration)**: implement required rebalancing logic and define migration semantics.
- [ ] **P0 Issue 69/80/93/158 (topology validation)**: implement or explicitly defer with a single decision (do not leave ambiguous "P0 but backlog" behavior).
- [ ] **P0 Issue 87 (ResourceManager init race)**: implement node discovery gating with a bounded wait budget and fail-fast error message.

### 🆕 State Model & Scheduler Logic

- [ ] 🆕 **State model implementation**: Physical GPU pool, cluster allocations, generation allocations as DP-worker bundles (Lines 1647-1654)
- [ ] 🆕 **Scheduler loop + planner**: Event-driven loop with phase ordering: completion → non-gen → gen → validate → execute → commit (Lines 1656-1669)
- [ ] 🆕 **Validation logic port**: Port from `gpu_scheduler_validation.py` to `schedrl/scheduler/validation.py` with all 11 critical conditions (Lines 1665-1667)
- [ ] 🆕 **Progress ingestion**: Implement `report_progress(...)` ingestion with explicit denominator, 2% banding (Lines 1677-1687)
- [ ] 🆕 **FIFO policy**: Ordering source is scheduler-side arrival time, not wall-clock timestamps (Lines 1642-1644)
- [ ] 🆕 **Proportional Rebalancing Algorithm (Issue 202 & 216)**: Implement "deterministic best-effort" algorithm: (1) Filter abortable mappings to old_active_dp_ranks; (2) Iteratively pick `dp_rank = argmax(len(dp_rank_to_src_ranks[dp_rank]))` to steal from most-loaded; (3) Stop if `max_load == 0`; (4) Snapshot active ranks before update (Lines 1047-1056)

### 🆕 Lifecycle & Safety (Implement in Phase 2)

- [ ] 🆕 **Two-lock usage**: `routing_lock` for metadata; `swapping_lock` for lifecycle serialization; `generate_one_request()` must not acquire `swapping_lock` (Lines 259-264)
- [ ] 🆕 **Strict sequencing**: Scheduler issues lifecycle actions sequentially; no retries/replays; single lifecycle caller (Lines 249-252)
- [ ] 🆕 **Lifecycle Sequencing Enforcement**: Verify that the Scheduler strictly serializes lifecycle batches (no new batch issued until previous ACKed) (Line 249)

### 🆕 Lifecycle Invariants (Implement in Phase 2)

- [ ] 🆕 **traj_id availability**: Move traj_id computation into `reset()`; add field to RolloutCache (Lines 1970-1974)
- [ ] 🆕 **creation_ts persistence**: Deferred; if added, record in `reset()` (Lines 1975-1979)

### 🆕 Push-Based Reporting (Implement in Phase 2)

- [ ] 🆕 **Scheduler MUST NOT poll workers for memory**: Push-based reporting only; release entity includes memory snapshot in ACK (Lines 425-427)
- [ ] 🆕 **Reference implementation**: Consult ROLL_multi_pipeline reference code for shrink-to-zero and subset sync semantics (Lines 432-438)

### 🆕 Negative Constraints (Must NOT Do)

- [ ] 🆕 **Do NOT copy fork's progress percentage math**: Treat fork progress math as potentially outdated; use shared protocol definition

### 🆕 Module-Level Utilities to Port

- [ ] 🆕 **_get_env_timeout_s()**: Port to SchedRL utils (24 lines)
- [ ] 🆕 **_get_env_timeout_optional_s()**: Port to SchedRL utils (29 lines)
- [ ] 🆕 **timeout_context()**: Port to SchedRL utils (21 lines)
- [ ] 🆕 **_get_named_actor_with_timeout()**: Port to SchedRL utils (19 lines)
- [ ] 🆕 **get_episode_scores()**: Port to Adapter utils (7 lines)
- [ ] 🆕 **get_traj_rollout_time()**: Port to Adapter utils (7 lines)
- [ ] 🆕 **get_traj_env_time()**: Port to Adapter utils (7 lines)
- [ ] 🆕 **compute_data_metrics()**: Port to Adapter utils (91 lines)

### 🆕 ConcurrentAgenticPipeline Decomposition

- [ ] 🆕 **SchedRL Core (GPU client)**: Port ~212 lines to SchedRL core
- [ ] 🆕 **Adapter (pipeline orchestration)**: Port ~1,230 lines to Adapter
- [ ] 🆕 **Observability (timeline tracing)**: Defer ~365 lines (post-ENG-123)
- [ ] 🆕 **Module-level utilities**: Port ~226 lines (Phase 2-3)

### Validation Milestone (Phase 2)

- [ ] Single-pipeline run: scheduler progresses through phases; progress ingestion updates are observed by scheduler.
- [ ] Induced failure during execution phase causes immediate job shutdown (no partial continuation).
- [ ] 🆕 **Topology Validation (Issue 69/80/93/158)**: Implement topology validation or explicitly defer
- [ ] 🆕 **Node Discovery (Issue 87)**: Implement node discovery gating with bounded wait budget and fail-fast error
- [ ] 🆕 **Gap-Ratio Fairness (Issue 64)**: Implement gap-ratio fairness constraints (work inflation prevention)
- [ ] 🆕 **Proportional Rebalancing (Issue 202 & 216)**: Implement required rebalancing logic and define migration semantics

---

## Phase 3 Checklist — Implement ROLL Adapter + Upstream ROLL Shim

**Primary goals**: create `schedrl:adapter:{pipeline_id}` actor; isolate all ROLL actors per pipeline namespace (Option A); keep SharedStorage job-global; implement safe shrink/expand/offload; port required upstream patches.

### Original Items (Preserved) — Namespace + SharedStorage

- [ ] **Namespace (Option A; required)**: set `ROLL_RAY_NAMESPACE=f"pipeline_{pipeline_id}_NS"` via Ray `runtime_env.env_vars` for all ROLL processes in this pipeline.
- [ ] **Global storage namespace (required)**: add `GLOBAL_STORAGE_NAMESPACE="global_storage_namespace"` and update all SharedStorage get/create sites to use it (never `RAY_NAMESPACE`).

### 🆕 Request ID Dual-Write Conflict (E1 - P0)

- [ ] 🆕 **E1: Remove stale build_request_id instruction (P0)**: The plan introduced `schedrl_request_id` as the canonical ID solution but left a stale `build_request_id` instruction in Phase 3. MUST explicitly remove this stale instruction from the implementation workflow to prevent accidental regressions. Use `schedrl_request_id` consistently for canonical ID. (Gap Analysis Line 508; Verified Issues Master Tracker Line 492-504)

### Original Items (Preserved) — Lifecycle Correctness + Upstream Patches

- [ ] **P0 Issue 35 (job launching gap)**: implement the job launch path separation (system orchestrator vs test runner) so ENG-123 can start a job-scoped Ray cluster + SchedRL actors and then admit pipelines reliably.
- [ ] **P0 Issue 28 & 72 (fractional GPUs)**: ensure worker actor options use the intended fractional GPU accounting consistently.
- [ ] **P0 Issue 43 & 108 (resource recovery on unregister)**: adapter/orchestrator must `ray.kill` pipeline actors, clear registries, and release PGs + SharedStorage keys on teardown.
- [ ] **P0 (LogMonitorListener lifecycle)**: disable/guard `LogMonitorListener` `atexit` hooks that call `ray.shutdown()` / `ray stop --force` in Library Mode; only SchedRL orchestrator may stop the job cluster.
- [ ] **P0 (placement group leak)**: explicitly call `destroy_placement_group()` (or the extracted equivalent) on pipeline teardown; verify no PG resources remain reserved after teardown.
- [ ] **P0 Issue 26, 208 & 241 (mandatory validation)**: reject invalid offload/sleep-level/partial-GPU configs at registration time (fail-fast).
- [ ] **P0 Issue 49 + 134 + 75/141 (ports/storage)**: implement atomic port claim + deterministic cleanup; ensure `delete_prefix(pipeline_id)` actually releases per-pipeline indices.
- [ ] **P0 (SharedStorage rendezvous key)**: change master rendezvous keying from `cluster_name` to `{pipeline_id}:{cluster_name}` to prevent cross-pipeline overwrites.
- [ ] **P0 Issue 85 (SGLang ports)**: remove hardcoded ports; use safe port allocation mechanism.
- [ ] **P0 Issue 86 (vLLM offload)**: offload must trigger regardless of colocation detection (library mode must free GPU memory).
- [ ] **P0 Issue 119 (SharedStorage detached)**: ensure storage actor lifetime is detached at first creation and shared across pipelines.
- [ ] **P1 Issue 30 & 41 (Adapter RPC API alignment)**: ensure Adapter RPC signatures match the SchedRL scheduler expectations and the upstream ROLL strategy/worker API surfaces (no mismatched names/args).
- [ ] **P0 Issue 213 + P1 Issue 212 (API mismatch / stop-before-offload)**: implement correct abort→drain→offload sequencing under new API.
- [ ] **P0-1 / P0-2 / P0-3 tasks**: `swapping_lock` + suspend re-check + prevent RolloutScheduler auto-resume.
- [ ] **P1 Issue 105 (lazy query race)**: implement fix and verify scheduler/adapter ordering constraints.
- [ ] 🆕 **P1 Issue 211 (max concurrency override)**: Ensure actor-infer concurrency settings are correct and not shared across pipelines. Required formulas:
    - Actor infer: `max(train_env_workers * envs_per_worker + 1, val_env_workers * envs_per_worker + 1)`
    - GroupQueueManager & RequestScheduler: `max(1000, env_num + 100)`
    - SGLang actors: 1000 (Lines 1105-1110)
- [ ] **P0 Issue 214 (HF cache isolation)**: enforce HF env isolation via `runtime_env` before any HF imports (do not rely on Worker `__init__`).
- [ ] **P1 (CheckpointManager cache keys)**: include `{pipeline_id}` in model download cache keys (or explicitly document + enforce cross-pipeline-safe invariants for shared cache).
- [ ] **P1 (BaseConfig global env)**: avoid mutating process-global `os.environ` for pipeline-specific settings in shared driver processes; route settings through `runtime_env.env_vars`.
- [ ] **P1 (vLLM/FlashInfer cache paths)**: ensure cache/workspace roots are pipeline-scoped (do not rely on `WORKER_NAME`, which can be overwritten).
- [ ] **P1 (state_offload_manger concurrency)**: replace `del os.environ[...]` with `os.environ.pop(..., None)` (or remove global env dependency) to avoid concurrency errors.
- [ ] **P1 (hidden retry loops)**: remove or explicitly configure retry loops (OpenAI proxy / sandbox reward / driver utils) to preserve ENG-123 fail-fast semantics.
- [ ] **P1 (mode_switch_lock)**: remove unused `mode_switch_lock` or enforce its usage; do not rely on dead serialization mechanisms.
- [ ] **P1 (rebalance termination)**: fix `_rebalance_on_expand` to have an explicit termination condition (no infinite `cycle()` behavior).

### 🆕 ROLL Adapter Implementation Details

- [ ] 🆕 **ROLL Adapter files**: Create `adapter.py` and `concurrent_pipeline.py` in `roll/schedrl_adapter/` (Lines 1710-1723)
- [ ] 🆕 **shrink_workers 6-step ordering**: Full sequence: (1) acquire swapping_lock, (2) call suspend(), (3) acquire routing_lock briefly, (4) issue abort + wait for drain WITHOUT holding routing_lock, (5) offload/stop workers, (6) call resume() and release swapping_lock (Lines 1725-1736, 1993-2003)
- [ ] 🆕 **expand_workers 4-step ordering**: Full sequence: (1) onload/start workers, (2) ModelUpdateService call, (3) open_admission, (4) verify ready (Lines 1737-1742)
- [ ] 🆕 **Expand failure policy**: Call `orchestrator.shutdown(force=True, reason="expansion_failed")` on any failure (Lines 1743-1746)
- [ ] 🆕 **Request_id plumbing**: Set `meta_info` fields (`traj_id`, `turn_id`, `attempt`, `schedrl_request_id`); `RolloutCache.attempt` persistence (Lines 1747-1766)
- [ ] 🆕 **RolloutScheduler pipeline_id prefix**: Add `pipeline_id` to `__init__` and prefix actor names (Line 1782)
- [ ] 🆕 **_validate_calculated_ranks fix**: Fix expand-mode validation bug (shrink validates active, expand validates inactive) (Line 1783)
- [ ] 🆕 **Sticky routing fix**: Ensure sticky routing never targets inactive dp ranks after shrink (Lines 1784-1785)
- [ ] 🆕 **Rollout offload validation**: Verify device-level occupied percent <= 10% after offload (Lines 1791-1796)
- [ ] 🆕 **Progress mapping**: Emit SchedRL `report_progress` from ROLL rollout bookkeeping (Lines 1797-1800)

### 🆕 P0 Tasks from Critical Issues Audit

- [ ] 🆕 **P0-1: Add swapping_lock**: Add `swapping_lock = asyncio.Lock()` to `RequestScheduler.__init__` (Lines 1786-1787)
- [ ] 🆕 **P0-2: Suspend re-check**: Add while loop in `generate_one_request()` re-checking `need_suspend` after acquiring `routing_lock` (Lines 1788-1790)
- [ ] 🆕 **P0-3: Prevent auto-resume**: Prevent RolloutScheduler from auto-resuming RequestScheduler (Lines 1791-1795)

### 🆕 Namespace & Collision Fixes (P0)

- [ ] 🆕 **RLVR GenerateScheduler actor collision (P0)**: RLVR pipeline creates `GenerateScheduler` with name derived only from `actor_infer.cluster_name` and `get_if_exists=True`. If multiple pipelines share the same cluster name, they share a scheduler actor. Fix: Ensure GenerateScheduler names are prefixed with `pipeline_id` or isolated in the per-pipeline namespace. (Extraction Plan Line 485-488)
- [ ] 🆕 **RewardScheduler collision**: Global + `get_if_exists=True` causes cross-pipeline state bleed; must use per-pipeline namespace (Lines 465-471)
- [ ] 🆕 **Global dataset actors collision**: `GlobalDataset`, `GlobalDatasetManager`, `val_dataset_manager` collide; must use per-pipeline namespace (Lines 472-479)
- [ ] 🆕 **GlobalLimiter collision**: Cross-pipeline throttling; must use per-pipeline namespace (Lines 480-484)
- [ ] 🆕 **AsyncGenerateScheduler GlobalCounter collision**: Cross-pipeline request_id/counter coupling; must use per-pipeline namespace (Lines 490-494)
- [ ] 🆕 **Megatron model_update locker collision**: Multi-pipeline serialization/deadlock; must use per-pipeline namespace (Lines 495-499)
- [ ] 🆕 **SGLang multi-node slave actor names**: Port allocation must account for multiple pipelines (Lines 500+)
- [ ] 🆕 **LogMonitorListener kills Ray cluster**: `atexit` hook calls `ray.shutdown()` + `ray stop --force`; must disable/guard (Lines 510-516)
- [ ] 🆕 **SharedStorage rendezvous key**: Key is `cluster_name` only, needs `{pipeline_id}` prefix (Lines 517-521)
- [ ] 🆕 **Port-lock key schema incompatibility**: `delete_prefix(pipeline_id)` cannot release `MASTER_ADDR_PORT:{ip}:{port}` keys; fix schema (Lines 522-526)

### 🆕 SGLang Port Allocation (Implement in Phase 3)

- [ ] 🆕 **SGLang Port Allocation**: Framework strategies must not hardcode ports. Use `get_free_port()` backed by SharedStorage atomic `claim_port` logic. Do NOT use deterministic port offsets/hashes as primary mechanism. (Line 1801)

### 🆕 Dual-Path Offload Verification (Implement in Phase 3)

- [ ] 🆕 **Training/Critic Path Validation**: Memory validation in `state_offload_manager` context manager `__exit__` block - verify device-level occupied percent <= 10% (Lines 1072-1085)
- [ ] 🆕 **Rollout Path Validation**: Memory validation after `offload_states_partial(dp_ranks)` completes - verify device-level occupied percent <= 10% (Lines 1804-1809)

**Rationale for two checks**: They cover different mechanisms:
- `state_offload_manager` validates offload/reload via context-manager path (training/critic steps)
- Rollout shrink/offload uses `offload_states_partial(dp_ranks)` and does NOT go through `state_offload_manager`

### 🆕 Negative Constraints - Do NOT Port (Phase 3 Reminders)

- [ ] 🆕 **Do NOT port fork vLLM worker_helper.py**: Reuse upstream `send_recv_utils.py` instead (Lines 1700-1708)
- [ ] 🆕 **Do NOT port fork TP-shard-aware receiver assembly**: Not required for ENG-123
- [ ] 🆕 **Do NOT port `notify_ready_to_release` inside GroupQueueManager.put**: Currently suppressed by early return in fork; keep it that way
- [ ] 🆕 **Do NOT port GenerateScheduler (v1)**: Synchronous/threaded model; use RequestScheduler (v2) only (Lines 905-909)
- [ ] 🆕 **Do NOT port agentic_rollout_pipeline**: Dead file (Lines 910-914)
- [ ] 🆕 **Do NOT port agentic_actor_pg_worker.py**: Obsolete due to unified ResourceManager (Lines 915-919)
- [ ] 🆕 **Do NOT port Legacy AgenticPipeline**: Use ConcurrentAgenticPipeline (Lines 920-924)
- [ ] 🆕 **Do NOT remove upstream AgenticPipeline class**: Keep upstream AgenticPipeline; use ConcurrentAgenticPipeline for multi-pipeline. Do NOT remove upstream class. (Lines 920-923)
- [ ] 🆕 **Do NOT port DynamicSamplingScheduler**: RLVR-only; out of scope (Lines 951-955)
- [ ] 🆕 **Do NOT port rlvr package**: Out of scope for ENG-123 (Lines 956-960)
- [ ] 🆕 **Do NOT port start_dual_pipeline_test.py**: Deprecated (Lines 971-975)
- [ ] 🆕 **Do NOT port CentralizedGPUSchedulerImpl**: Legacy code, not production scheduler (Lines 976-980)
- [ ] 🆕 **Do NOT port broken val()**: Dead/commented validation logic (Lines 1012-1015)
- [ ] 🆕 **Do NOT port `user_defined_rollout_loop`**: Dead file (Line 910)
- [ ] 🆕 **Do NOT port `agent_native_env_manager`**: Dead access pattern (Line 961)
- [ ] 🆕 **Do NOT port `tir_env_manager`**: Dead access pattern (Line 961)
- [ ] 🆕 **Do NOT port `step_concat`**: Dead access pattern (Line 961)
- [ ] 🆕 **Do NOT port `vl_traj`**: Dead access pattern (Line 961)
- [ ] 🆕 **Do NOT port `utils/local_code`**: Dead utility (Line 966)
- [ ] 🆕 **Do NOT port `impl specific tests`**: Dead utility (Line 966)
- [ ] 🆕 **Do NOT port GroupFilter**: Dead/duplicate logic (Lines 989-993)
- [ ] 🆕 **Boundary Constraint**: Do NOT move `model_update_service` to the `schedrl` package; it remains framework-specific in ROLL (Line 190)

### 🆕 Additional Infrastructure (Phase 3)

- [ ] 🆕 **P2 Issue 205: Proactive Release logic**: Port standard `release_gpus()` logic; DO NOT port proactive `notify_ready_to_release` call
- [ ] 🆕 **P2 Issue 203 & 209: Offload verification**: Memory validation (10% threshold); fix typo `state_offload_manger` → `state_offload_manager`
- [ ] 🆕 **Remove hardcoded partial_gpu_mode**: Set `partial_gpu_mode = False` - SchedRL takes full control (Lines 1020-1031)

### 🆕 Lifecycle & Safety Implementation (Phase 3)

- [ ] 🆕 **6-Step Atomic Teardown Sequence**: Mandatory ordering: (1) acquire swapping_lock, (2) call suspend(), (3) acquire routing_lock briefly to mark ranks inactive, (4) issue abort + wait for drain WITHOUT holding routing_lock, (5) offload/stop workers, (6) call resume() and release swapping_lock (Lines 1993-2003)
- [ ] 🆕 **Constants Patch (P0)**: (1) Patch `third_party/ROLL/roll/utils/constants.py` to set `RAY_NAMESPACE = os.environ.get("ROLL_RAY_NAMESPACE", "roll")`; (2) Add `GLOBAL_STORAGE_NAMESPACE = "global_storage_namespace"`; all SharedStorage sites must use this constant (Lines 174, 182)
- [ ] 🆕 **Admission State Inconsistency**: Implement check in `generate_one_request`: if worker set is empty but `need_suspend` is False, raise `RuntimeError("No active workers and not suspended")` (Line 326)
- [ ] 🆕 **Fail-Fast Lifecycle Assertions**: Transplant upstream assertions in ROLL Adapter: `load_states_partial(dp_ranks)` MUST assert `is_loaded` is False; `offload_states_partial(dp_ranks)` MUST assert `is_loaded` is True (Lines 1154-1157)

### Validation Milestone (Phase 3)

- [ ] Two pipelines in same job: no Ray named-actor collisions (verify both pipelines fully initialize).
- [ ] Shrink-to-zero + expand: no `No active DP ranks` error; no re-open of suspend by RolloutScheduler; GPU memory is actually freed.
- [ ] Pipeline teardown + re-admission in same job: ports/metadata reusable (no leaks).
- [ ] 🆕 **Phase 3 Success Criteria**: Manual verification: trigger shrink mid-rollout, confirm aborted turns retry safely, verify shrink-to-zero releases GPUs (Lines 1818-1825)
- [ ] 🆕 **Placement Group Leak (P0)**: Verify `destroy_placement_group()` is explicitly called on pipeline teardown and no PG resources remain reserved
- [ ] 🆕 **Atomic Port Claim/Cleanup (P0 Issue 49)**: Verify `delete_prefix(pipeline_id)` actually releases per-pipeline indices and ports are available for reuse
- [ ] 🆕 **Safe Port Allocation (P0 Issue 85)**: Verify framework strategies (SGLang) uses `get_free_port()` backed by atomic claim logic (no hardcoded ports)
- [ ] 🆕 **vLLM Offload Trigger (P0 Issue 86)**: Verify offload triggers regardless of colocation detection (Library Mode must free GPU memory)
- [ ] 🆕 **HF Env Isolation (P0 Issue 214)**: Verify HF env vars are enforced via `runtime_env` *before* any HF imports occur in the worker process
- [ ] 🆕 **Rebalance Termination (P1)**: Verify `_rebalance_on_expand` has an explicit termination condition to prevent infinite `cycle()` loops

---

## Phase 4 Checklist — Selective Model Update Behind Adapter

**Primary goals**: selective sync-on-resume; minimal upstream hooks; correctness under expand after weight update.

**Decision record (implementation)**:
- During ENG-123 implementation, we chose to follow the **fork pattern** from `third_party/ROLL_multi_pipeline` for selective model update behavior/control flow wherever practical.
- If this checklist is ambiguous, treat the extraction plan Phase 4 text as authoritative, and prefer fork reference behavior unless explicitly overridden by ENG-123 constraints.

### Original Items (Preserved)

- [ ] **P1 Issue 66 (selective expansion safety)**: ensure expansion is safe and does not admit stale weights; fail-fast on any mismatch.
- [ ] **P1 Issue 207 (dual-path GPU release logic)**: implement/verify the correct release path(s) under the adapter contract.
- [ ] Implement `ModelUpdateService` selective sync trigger (no promotion forwarding, no version choosing, no coalescing/validation).

### 🆕 ModelUpdateService Implementation Details

- [ ] 🆕 **Promotion is the source-of-truth (no version choosing in service)**: Coordinator must explicitly promote the active rollout checkpoint version (checkpoint_version + global_step) by calling the sender-side component that owns cache pointers (strategy method or minimal Worker RPC). Sender strategy holds the cache map + `active_cached` pointer; `ModelUpdateService` triggers sender-side selective sync and does **not** choose versions. (Phase 4; Lines 1829-1866)
- [ ] 🆕 **Selective sync contract**: `ModelUpdateService.sync_selected_workers(tgt_dp_ranks)` only. No `requested_global_step`, no monotonic/anti-stale validation, no coalescing. Sender applies its currently promoted `active_cached` version.
- [ ] 🆕 **Namespace dependency (Phase 3 prerequisite)**: Phase 4 assumes Phase 3 has already implemented `runtime_env.env_vars` injection so `ROLL_RAY_NAMESPACE` + `PIPELINE_ID` are set before worker imports; fail-fast if missing/mis-scoped
- [ ] 🆕 **Bucket caching location/order (required)**: Sender-side bucket cache is built inside `train_step` after weights update and before any offload. Do not add a new requirement that caching cadence is driven by an `ActorWorker.initialize` hook.
- [ ] 🆕 **Selective Update Flow (targeting vs membership)**: Service targets **DP engines** via `tgt_dp_ranks`; collective membership is “sender + selected receiver TP/PP workers inside those DP engines” (subset-scoped group creation) (Lines 1867-1882)
- [ ] 🆕 **Worker requirements (verify then port minimal set)**: Do not assume upstream `Worker` already has selective-update RPCs; verify what exists in `third_party/ROLL` and port only the minimal required set used by the final implementation (Lines 1883-1897)
- [ ] 🆕 **Guardrail allowlist semantics**: `set_model_update_allowed_dp_ranks(tgt_dp_ranks)` is a safety net only; excluded ranks must not be in any selective-update collective group membership
- [ ] 🆕 **CRITICAL vLLM payload-shape constraint**: vLLM `update_parameter_in_bucket` indexes `serialized_named_tensors[self.rank]`; sender must provide a **rank-indexed list-like payload** covering the engine’s TP/PP workers (no sparse dict) unless an explicit upstream receiver change is made
- [ ] 🆕 **Required upstream change**: `GroupManager.destroy_collective_group()` must call `dist.destroy_process_group()` and only destroy groups created by the current update instance (Lines 1932-1935)
- [ ] 🆕 **Integration code pattern**: `expand_workers` calls selective sync for newly expanded `tgt_dp_ranks` and then proceeds with admission/routing reopen (Lines 1909-1924)
- [ ] 🆕 **No coalescing**: No service-side coalescing logic. Service calls may arrive concurrently; end-to-end serialization is provided by the sender/trainer `cache_lock` (lock ordering is not guaranteed/FIFO is not a contract). (Line 1866)
- [ ] 🆕 **No CPU-staging flag dependency (required)**: Do not rely on `ROLL_SELECTIVE_MODEL_UPDATE_RECEIVER_DISABLE_CPU_STAGING=1`, and do not rely on fork receiver-shard APIs (`meta_infos/buffer/ranks_in_worker`). Receiver-side target remains upstream vLLM `update_parameter_in_bucket(serialized_named_tensors)` indexing `serialized_named_tensors[self.rank]`; sender must provide a rank-indexed list-like payload compatible with upstream `send_recv_utils.py` serialization format.


### Validation Milestone (Phase 4)

- [ ] Expand after a weight update: newly expanded dp ranks serve the correct weights (no stale admission).
- [ ] 🆕 **Phase 4 Success Criteria**: Manual verification: dynamic NCCL group teardown doesn't leak, colocated paths use CUDA IPC (Lines 1941-1950)
- [ ] 🆕 **Dual-Path GPU Release (P1 Issue 207)**: Verify the correct release path (immediate vs. planned/atomic) is used under the adapter contract

---

## 🆕 P2 Findings (Deferred but Documented)

These items are deferred to backlog but should be documented for future reference:

- [ ] 🆕 **P2 Issue 90: Mixed TP Key Support**: `tensor_parallel_size` vs `tensor_model_parallel_size` - Backlog
- [ ] 🆕 **P2 Issue 102 & 242: Timeout Scaling & Overrides**: For large clusters - Backlog
- [ ] 🆕 **P2 Issue 27, 61 & 210: Profiling & Timeline Tooling**: Backlog
- [ ] 🆕 **P2 Issue 24: Typed Resource Requirements dataclass**: Backlog
- [ ] 🆕 **P2 A3: notify_ready_to_release idempotency**: 107 lines with offload_lock, offload_notified flag - P2
- [ ] 🆕 **P2 A4: Abort retry backoff**: No backoff between retries after abort - P2
- [ ] 🆕 **P2 B2: offload thread cleanup**: Patch strategies (vLLM/SGLang) to support scheduler-mandated offload bypassing `is_actor_infer_colocated` check - Framework strategies must allow scheduler-mandated offload even when colocation detection returns False (Gap Analysis Line 490)
- [ ] 🆕 **P2 D2: update_parameter_in_bucket**: Use indexable wrapper (`__getitem__`) for rank-indexed list - If the payload is identical for all TP/PP ranks, use an indexable wrapper that returns the same serialized bytes for any rank to avoid building a full list and prevent OOMs when broadcasting to many TP/PP ranks (Gap Analysis Line 502; Verified Issues Master Tracker Line 455-467)
- [ ] 🆕 **P2 H4: End-of-cycle guard**: Remove misplaced break; implement proper any() check - P2
- [ ] 🆕 **P2 H5: _execute_expansions failure**: Ensure signaling in finally block for partial failures - P2
- [ ] 🆕 **P2 H6: unregister_pipeline leaks**: Full state cleanup - MUST clear `active_allocations`, `pending_requests`, AND `pending_completion_requests` to prevent orphaned state crashes on subsequent cycles (Gap Analysis Line 516; Verified Issues Master Tracker Line 631-635)
- [ ] 🆕 **P2 H7: Silent shrink skip**: Change to assert to fail fast - P2
- [ ] 🆕 **P2 H8: Phase 1 dead code**: Remove unused _fifo_sorted_pending_and_active_cluster call - P2
- [ ] 🆕 **P2 H9: Invariant dead code**: Remove dead _dedup_and_query_timestamps method - P2
- [ ] 🆕 **P3 H10: Lock precondition**: Add locking to public wrapper rebalance_on_expand - P3
- [ ] 🆕 **P3 H11: _try_activate_one side effects**: Check eligibility before committing donor shrinks - P3

---

## Non-Action Decisions (Must Remain True)

These are listed in the plan as P0/P1 but are **explicitly closed/invalid by design**; the implementation must not accidentally reintroduce them:

### Original Items (Preserved)

- [ ] **P0 Issue 132**: deadlock analysis is "invalid by design" under enforced offload config + fail-fast.
- [ ] **P0 Issue 236 & 217**: marked closed/invalid in the plan; if ENG-123 requires shrink-to-zero, treat the shrink-to-zero tasks above as the authoritative implementation path (do not rely on the closed label).
- [ ] **P1 Issue 143 & 147**: closed/invalid; do not add complex recovery/graceful swap behavior in ENG-123.
- [ ] **P1 Issue 65, 218 & 104**: invalid by design; do not add an extra notification protocol surface in ENG-123.
- [ ] **P1 Issue 151**: closed/invalid; do not build scheduler logic that depends on racy progress reporting to preserve correctness.

---

## Summary Statistics

| Phase | Original Items | New Items (🆕) | Total |
|-------|----------------|----------------|-------|
| Phase 1 | 7 | 58 | 65 |
| Phase 2 | 9 | 25 | 34 |
| Phase 3 | 26 | 65 | 91 |
| Phase 4 | 3 | 9 | 12 |
| Non-Action Decisions | 5 | 0 | 5 |
| P2 Findings (Deferred) | 0 | 16 | 16 |
| **TOTAL** | **50** | **173** | **223** |

The enhanced checklist provides **comprehensive coverage** of all requirements from both the original checklist and the gap analysis findings, organized by execution phase for practical implementation.
