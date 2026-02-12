# ENG-123: Extract ROLL_multi_pipeline time-sharing into SchedRL (Option A)

## Overview

We will re-plan and implement SchedRL as the shared **Ray-based** time-sharing library (**Library Mode only for ENG-123**) and migrate the working multi-pipeline time-sharing logic from `third_party/ROLL_multi_pipeline/` into `schedrl/`, while keeping **framework-specific mechanics** in `third_party/ROLL` behind a small Adapter/shim.

Option A boundary + framework patching policy (ENG-123):
- Ownership split:
  - `schedrl/` owns: protocol/types + client + central scheduler actor (Library Mode) + planner/executor + state.
  - `third_party/ROLL` owns: ROLL Adapter actor implementation, and ROLL-specific mechanics (DP subset lifecycle, targeted abort+ACK semantics, progress mapping) used to satisfy the shared Adapter RPC surface.
- Adapter-first integration is the default: prefer implementing framework-specific behavior in the ROLL Adapter/shim.
- Minimal, upstreamable patches to RL frameworks (e.g., ROLL internals) are allowed only when adapter-only alternatives are clearly worse (e.g., Ray actor limitations or missing required framework hook points).
- Any framework patch must be narrowly scoped, must not move scheduler policy into framework core, and must include a short rationale in the PR.

Primary goal for ENG-123:
- **ROLL-only integration**, but the extracted scheduler core must be designed so SkyRL-train can be added next with a new Adapter (no scheduler redesign).
- Support **Library Mode** from day 1 (job-owned scheduler/orchestrator, fork-parity). Service Mode is deferred to backlog.

Backlog (post-ENG-123): implement Service Mode per `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`.

Configuration decisions (ENG-123 scope):
- SchedRL configuration is expressed as typed Python dataclasses passed to Ray actors (no SchedRL YAML).
- Adapter contract behavior (abort ACK = “not in-flight”, strict shrink ordering, shrink-to-zero support) is enforced in the ROLL adapter and treated as SchedRL assumptions for ENG-123.
- Framework-specific job configuration remains framework-owned (e.g. ROLL YAML). Frameworks map their config into SchedRL’s typed registration payload when calling `register(...)`.
- If a future framework cannot satisfy an ENG-123 assumption (e.g. cannot shrink-to-zero), we will add a capabilities/constraints payload at pipeline registration/admission time so SchedRL can reject or plan with constraints.

Timeout configuration decision (ENG-123):
- Keep action timeouts controlled by environment variables as in the fork (do not centralize into SchedRL config in ENG-123).
- Backlog: centralize timeout configuration into `schedrl` typed config and propagate it to pipelines/adapters (so we can eliminate env-var coupling).

Plan-wide rule (ENG-123):
- Any `schedrl.*timeout*` typed config fields must be treated as **disabled** for ENG-123 by using an explicit invalid/sentinel value (for example `-1`). The source of truth for all action timeouts remains environment variables until the backlog item is completed.

## Current State Analysis

### What exists today

1) A complete working implementation of multi-pipeline time-sharing exists in a fork:
- `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py` (central scheduler)
- `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/generate_scheduler.py` (per-pipeline RequestScheduler with DP subset shrink/expand)
- `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/rollout_scheduler.py` (GroupQueueManager progress emission)
- `third_party/ROLL_multi_pipeline/roll/distributed/executor/model_update_service.py` (selective model update on resume)
- `third_party/ROLL_multi_pipeline/roll/pipeline/agentic/concurrent_agentic_pipeline.py` and `.../multi_pipeline_orchestrator.py` (orchestration)

2) Upstream ROLL (`third_party/ROLL/`) does **not** have the multi-pipeline scheduler/time-sharing system integrated; it has a single-pipeline Agentic pipeline and existing primitives we can adapt:
- Abort primitive + suspend/resume: `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`
- Env loop retry on abort (two-phase-commit compatible): `third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py` (referenced in the ROLL adaptation plan)

3) `schedrl/` is not implemented yet (no existing extraction patterns in code; only plans): `schedrl/` currently has no Python implementation.

### Key constraints (non-negotiables)

From the (future) dual-mode plan (Service Mode deferred; Library Mode only for ENG-123):
- SchedRL is Ray-actor based; importing `ray` in `schedrl` is acceptable.
- Library Mode discovery + naming conventions must exist from day 1 (create-if-missing).
- Safety-critical timeouts are fail-fast.

Ray lifecycle (ENG-123, Library Mode):
- Ray clusters are **job-scoped** and created by a **SchedRL-managed launcher**.
- The launcher runs on **all nodes** (MPI-style): rank 0 starts the Ray head, other ranks start Ray workers.
- Each node performs a freshness check; if Ray is already running on that node, execute `ray stop --force` before starting the new cluster.
- The SchedRL Orchestrator does **not** manage Ray start/stop; it only connects to the launcher-created cluster and creates SchedRL actors.

ENG-123 assumption (naming / isolation):
- Library Mode runs on a Ray cluster that is **independent per job** (no shared Ray cluster between separate jobs). Therefore we use a single standardized Ray namespace and actor naming scheme for the job.
- Ray cluster lifecycle is owned by the SchedRL launcher (fresh stop/start per node); the orchestrator only connects.

Backlog (post-ENG-123): define/validate isolation rules for Service Mode (detached scheduler shared across jobs).

From the shared protocol:
- Shrink ordering is strict: `close_admission` → (abort+ACK OR drain) → stop/offload (must fully release GPU memory).
- `REQUEST_RETRY` safety requires deterministic per-turn request IDs and an abort ACK that is safe under races (see “Verified issues” below).

## Verified issues to address (revise plan tasks accordingly)

This section verifies the reported issues against the codebase and records the concrete fixes + test expectations that must be reflected in the phases below.

### Critical 1: Shrink-to-zero logic and lifecycle transitions

**Verified**.

**Evidence**: `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py` → `RequestScheduler._rebalance_on_shrink(...)`:
- `keep_ranks = list(self.active_dp_ranks - set(shrink_dp_ranks))`
- `if not keep_ranks: raise ValueError("Cannot shrink to zero active ranks")` (see grep around lines ~1506–1509).

**Why it matters**: Full time-sharing requires the scheduler to be able to shrink a pipeline’s generation cluster to zero active DP ranks. This is a special lifecycle transition that requires careful ordering to avoid races.

**Required fix (Phase 3: ROLL adapter work)**:
- Relax the guard and implement a safe “shrink-to-zero” state in upstream ROLL:
  - Allow `active_dp_ranks = set()`.
  - **Ordering Requirement**: The Adapter must call `suspend()` (setting `need_suspend=True`) *before* clearing any internal routing metadata or `active_dp_ranks`. This ensures new requests block rather than hitting an empty worker set.
  - Shrink-to-zero must:
    1) set `need_suspend=True` and clear `suspend_notifier`.
    2) close admission
    3) abort/drain in-flight requests (waiting for `running_requests` to drain)
    4) wait for ACK
    5) offload/stop servers for all dp ranks
    6) clear routing metadata (`active_dp_ranks`, `src_rank2_dp_rank`)
    7) return success with a `shrunk_to_zero` signal so SchedRL can reclaim GPUs.

**Validation**:
- Verify shrink-to-zero transition results in `active_dp_ranks == set()`.
- Verify shrink-to-zero → expand cycle works without losing queued work.



### Critical 2: Abort ACK semantics must tolerate “normal completion while abort is issued”

**Verified (race exists)**.

**Evidence**:
- Upstream ROLL `RequestScheduler.rebalance_on_shrink(...)` calls `_rebalance_on_shrink(...)` with `asyncio.wait_for(..., timeout=30.0)` and uses in-memory tracking of `running_requests` as completion signal (see around lines ~1483–1490 and the `running_requests` bookkeeping).
- `VllmStrategy.generate_request(...)` normalizes `finish_reason=None` to `"abort"` for compatibility, so `finish_reason == "abort"` is not a reliable strict ACK in all versions.

**Required protocol tweak (this plan)**:
- For ROLL Adapter/shrink correctness, define ACK as: **“targeted request IDs are no longer in-flight (removed from `running_requests`)”**.
- If backend finish reasons are available, record them for observability, but do **not** require `finish_reason == "abort"` (normal completion is acceptable).

**Abort timeout policy (explicit)**:
- Keep the dual-mode plan’s fail-fast default: if not all targeted requests leave in-flight state before timeout, **crash pipeline / return failure and do not offload**.

**Validation**:
- Verify “normal-complete-while-abort” scenario proceeds safely.
- Verify fail-fast behavior on timeout.

### Critical 3: Minimize “design drift” — keep ENG-123 aligned with fork reality

Several items in the older intended design are not requirements for ENG-123 (or were inaccurate). For ENG-123 we explicitly align to fork reality:

- **Priority queue internals**: keep the existing fork implementation (per-priority FIFO lists + wakeup loop). Do not refactor to heapq in ENG-123.
- **Priority taxonomy**: keep the existing 7-tier `Priority` enum; do not compress to 4 tiers in ENG-123.
- **API naming/return values**: keep fork API naming for ENG-123 (`register_pipeline` / `admit_pipeline` / `request_gpus`). `request_gpus` is a blocking activation grant over registered GPU IDs (ROLL YAML is source-of-truth).
- **Min-DP constraints**: do not add `min_dp_workers_per_pipeline` enforcement in ENG-123; track as future work.

These are documented here so implementers do not “close a gap” by changing behavior beyond ENG-123 scope.

Decision notes (from `multi_pipeline_gaps.md`) that are binding for ENG-123:

- Keep fork queue implementation; do not pursue heapq/lock-free refactor in ENG-123.
- Keep fork FIFO semantics; do not add priority boosting in ENG-123.
- Keep fork API naming (`register_pipeline`/`admit_pipeline`/`request_gpus`) for ENG-123; interpret `request_gpus` as a blocking activation grant (no GPU selection by SchedRL).
- Treat locking vs atomicity as an implementation detail; rely on the coordinator contract + fork’s existing lock discipline.

Post-ENG-123 follow-ups (explicitly reflecting the gaps doc decisions):
- Priority boosting: defer until after ENG-123.
- Min-DP constraints: keep fork shrink planning behavior for ENG-123; add `min_dp_workers_per_pipeline` enforcement after ENG-123.
- Priority taxonomy: keep fork 7-tier taxonomy for ENG-123; revisit after ENG-123 if needed.
- Queue refactors: heapq/lock-free refactor is deferred until after ENG-123; the “What is needed to close gap” item proposing heapq in `multi_pipeline_gaps.md` is superseded by the ENG-123 Decision.
- **SharedStorage cleanup (G4)**:
  - ENG-123: implement explicit `SharedStorage.delete_prefix(pipeline_id)` cleanup on pipeline teardown so ports/metadata can be reused within the same job.
  - Post-ENG-123: add TTL / detached-actor lifecycle hardening for Service Mode.

## Concrete contracts (must match exactly)

See `## Workflow + Contracts (SPEC; ENG-123)` for the admission-gated startup flow, GPU source-of-truth (ROLL YAML decides concrete GPU IDs), and TP>1 weight transfer requirements.

### Standardized actor names (ENG-123)

This plan uses **two kinds of Ray namespaces**:

1) **SchedRL-owned actors** (job-scoped control plane) live in Ray namespace `schedrl`.
2) **ROLL-owned actors** (per-pipeline data plane + helper actors) live in a **per-pipeline Ray namespace**:
   - `pipeline_{pipeline_id}_NS` (string template; must be constructed from the canonical `pipeline_id`)

**Exception (job-global)**:
- The `SharedStorage` actor is a **job-global coordination primitive** (ports + metadata + caches) and MUST live in a dedicated, shared namespace:
  - `global_storage_namespace`
  - Rationale: port claims must be unique across all pipelines in the job; per-pipeline storage would reintroduce collisions.

**Implementation mechanism (ENG-123; Option A namespace override; required)**
- We do NOT want to touch every `.options(namespace=...)` call site. Instead, ENG-123 uses an env-var-driven namespace override:
  1. Patch upstream ROLL constant to be env-driven:
     - In `third_party/ROLL/roll/utils/constants.py`, set:
       - `RAY_NAMESPACE = os.environ.get("ROLL_RAY_NAMESPACE", "roll")`
  2. At pipeline admission, SchedRL/Adapter MUST inject per-pipeline env vars via Ray `runtime_env.env_vars` for all ROLL processes (adapter + workers + helper actors):
     - `PIPELINE_ID = {pipeline_id}`
     - `ROLL_RAY_NAMESPACE = f"pipeline_{pipeline_id}_NS"`
  3. Result: all existing ROLL actor creations/lookups that use `namespace=RAY_NAMESPACE` automatically become per-pipeline.
- **SharedStorage exception**: all SharedStorage get/create sites MUST ignore `RAY_NAMESPACE` and instead use `namespace="global_storage_namespace"` so there is exactly one storage actor shared across all pipelines.
- **SharedStorage namespace constant (required)**: Define a single hardcoded namespace constant in upstream ROLL (do not derive from `pipeline_id`):
  - Add `GLOBAL_STORAGE_NAMESPACE = "global_storage_namespace"` to `third_party/ROLL/roll/utils/constants.py`.
  - All SharedStorage get/create/lookup must use `namespace=GLOBAL_STORAGE_NAMESPACE` (never `namespace=RAY_NAMESPACE`).

- `schedrl:orchestrator`
- `schedrl:scheduler`
- `schedrl:resource_manager` (regular class converted to Ray actor via `ray.remote(ResourceManager)` at runtime)
- `schedrl:adapter:{pipeline_id}`

Explicit non-goal / boundary:
- `model_update_service` is framework-specific and lives in `third_party/ROLL` (see **Option A boundary + framework patching policy (ENG-123)**). Do not add `schedrl:model_update_service:{pipeline_id}`.

### Canonical `request_id`

All modes (Service + Library):
- `request_id = "{pipeline_id}:{traj_id}:{turn_id}:{attempt}"`
- Types: `pipeline_id: str`, `traj_id: str`, `turn_id: int >= 0`, `attempt: int >= 0`

`schedrl/protocol/request_id.py` must implement:
- `build_request_id(pipeline_id: str, traj_id: str, turn_id: int, attempt: int) -> str`
- `parse_request_id(request_id: str) -> tuple[pipeline_id, traj_id, turn_id, attempt]`
- `validate_request_id(request_id: str) -> None` (raise `ValueError` with message)

Additional required helper:
- `validate_pipeline_id(pipeline_id: str) -> None` (raise `ValueError` if invalid; must reject the request-id delimiter `:`)

+**Validation (delimiter safety)**
+- Implementation rule: the plan requires that `pipeline_id` and `traj_id` MUST NOT contain the request-id delimiter `:`. `schedrl.protocol.request_id.validate_request_id()` must raise `ValueError` if the parsed components contain `:` or invalid characters. This keeps parsing unambiguous.
+- Alternate option (if desired): switch the canonical delimiter to a safer character (for example `|`) and update the helper/parsers accordingly; if that route is chosen, update this plan text and all adapters before Phase 2 coding.

Usage requirement:
- The registration/admission path must call `validate_pipeline_id(pipeline_id)` (fail-fast) before accepting a pipeline.

### Release ACK payload (`release_reports`)

The releasing entity (adapter/worker) returns a release ACK payload. Post-release GPU memory reporting is deferred until after ENG-123; use `-1` sentinel values and add the real measurement later:
```json
{
  "aborted": 12,
  "remapped": 4,
  "release_reports": [
    {
      "dp_rank": 2,
      "gpu_map": [0, 1],
      "free_bytes_by_gpu": [-1, -1],
      "total_bytes_by_gpu": [-1, -1]
    }
  ]
}
```
Semantics:
- bytes are integers
- `gpu_map` are visible-device indices for that dp worker

### `report_progress(...)` schema + cadence

`report_progress(pipeline_id: str, queued_trajectories: int, inflight_trajectories: int, step_target_trajectories: int, metrics: dict[str, Any] | None)`

Cadence:
- emit at batch start
- emit on 2% band crossings of `percent_completed = collected_trajectories / step_target_trajectories` (do **not** clamp; allow `> 1.0`)
- always emit once when `percent_completed >= 1.0`

### High: Adapter lifecycle RPCs have no intent/version token (`activation_epoch`) — rely on strict sequencing

**Intentional for ENG-123**.

**Evidence**: No occurrences of `activation_epoch` in `third_party/ROLL` code (grep shows no matches).

**Requirement (must be validated by implementor)**:
- Scheduler issues lifecycle actions strictly sequentially (no overlapping batches).
- No retries / no replays of lifecycle RPCs after timeout/failure.
- Single lifecycle caller (only SchedRL scheduler issues lifecycle RPCs).

**Serialization (by design; no retry tolerance)**:
- Adapter actions are **not** required to be idempotent.
- Actions must be **serialized** (no interleaving/concurrency) with `swapping_lock` for full shrink/expand operations and `routing_lock` for brief routing-metadata edits.
- Assumption (ENG-123 scope): callers do **not** issue duplicate/retry RPCs after timeouts; the system assumes the action went through.

**Two-lock usage (source of truth)**:
- `routing_lock` protects routing metadata changes (`active_dp_ranks`, `src_rank2_dp_rank`, and any per-src sticky mappings); keep hold time brief.
- `swapping_lock` serializes full lifecycle operations (shrink/expand onload/offload/abort-drain windows) to prevent concurrent physical worker-state races.
- `generate_one_request()` must not acquire `swapping_lock`; request routing should only touch `routing_lock`.
- Any add/remove of active DP ranks and any clearing of sticky mappings during shrink/expand happens under `routing_lock`.
- For required ordering during shrink/expand (suspend-before-clear, brief `routing_lock` hold, no `routing_lock` held across awaits, lifecycle serialization), see **“Routing lock atomicity and suspend-before-clear ordering (required)”** in the verification clarifications.

**Documented limitation (explicit)**:
- ENG-123 intentionally does **not** implement `activation_epoch` / `action_id` / idempotency tokens or “Superseded” ACK semantics.
- Correctness depends on a strict **single-caller + no-retry** contract for lifecycle RPCs.
- If we later need retries, multi-caller safety, or overlapping lifecycle batches, we must reintroduce an intent/version token (epoch/sequence) and define supersession behavior.

### High: Deterministic request_id uniqueness across pipelines

**Verified as a consistency gap**.

**Evidence**: Upstream ROLL currently sets request IDs internally in schedulers (e.g., `request_id = f"{self.request_id}_{self.request_counter}"`), and does not incorporate `pipeline_id`.

**Required fix (this plan)**:
- Adopt canonical request id format for ROLL when used with SchedRL:
  - **All modes**: `{pipeline_id}:{traj_id}:{turn_id}:{attempt}`
- Provide a `schedrl` helper (in protocol) to build/parse/validate these IDs; ROLL adapter uses it to validate and to target aborts.

**Upstream overwrite note (required; request_id dual-write)**:
- Upstream ROLL `RequestScheduler.generate_one_request()` overwrites `data.meta_info["request_id"]` with an internal `uuid_counter`-style id.
- For ENG-123 we keep the internal `request_id` as the primary abort-tracking key, and we carry the canonical SchedRL id in a separate field:
  - Set `data.meta_info["schedrl_request_id"] = build_request_id(...)` in `traj_env_manager.py`.
  - Preserve `data.meta_info["request_id"]` as the scheduler-generated internal id.
- Any targeted abort / running_requests tracking uses the internal `request_id`; scheduler/protocol-facing logs and cross-pipeline correlation use `schedrl_request_id`.

**Validation**:
- Verify parser/validator works correctly.
- Verify targeted abort cannot affect a different pipeline (IDs are pipeline-scoped).

### High: Sticky routing must not route to removed ranks after shrink

**Partially verified**.

**Evidence**:
- Upstream ROLL `RequestScheduler` keeps `src_rank2_dp_rank` sticky mappings and maintains `active_dp_ranks`.
- There is mapping cleanup logic (e.g., `_clear_src_rank_mappings` invoked in shrink paths), but the plan must require that **all** DP-rank removal paths clear any mapping pointing to removed ranks and routing always consults `active_dp_ranks`.

**Required fix**:
- Ensure `generate_one_request` revalidates that the sticky dp rank is still active; otherwise clear and re-pick from active ranks.
- Ensure shrink/expand teardown follows the documented suspend-before-clear ordering (see “Lifecycle invariants & routing atomicity”) so `generate_one_request()` cannot observe an empty `active_dp_ranks` set while not suspended.

**Validation**:
- Verify after shrink, retries select from active ranks only.

### High: Admission Error Handling (Verified from ROLL_multi_pipeline)

**Verified behavior**:
- Requests block at `_check_suspend` while `need_suspend=True`.
- Requests raise `RuntimeError` if `active_dp_ranks` is empty while `need_suspend=False` (state inconsistency).

**Required implementation in upstream ROLL `RequestScheduler` (per-pipeline RequestScheduler actor instance)**:
- Implement a blocking gate in `generate_one_request` using `asyncio.Event` (e.g., `suspend_notifier`) and a `need_suspend` flag.
  - **Behavior during transition**:
  - `close_admission(dp_ranks)`: 
    - Locking note (avoid confusion): `routing_lock` lives on `RequestScheduler` (not on the outer `schedrl:adapter:{pipeline_id}` Ray actor). Implement `RequestScheduler.close_admission(...)` as `async with self.routing_lock:` (or assert caller holds it). This path mutates `active_dp_ranks` / sticky routing metadata and must be atomic w.r.t. `generate_one_request()`.
    - If it results in zero active workers (shrink-to-zero): set `need_suspend=True`, clear `suspend_notifier`.
      - Under `routing_lock`, perform the required mapping updates/clears (mark ranks inactive; clear `src_rank2_dp_rank` / sticky mappings) using the ordering in **Routing lock atomicity and suspend-before-clear ordering (required)**.
    - Otherwise: remove `dp_ranks` from `active_dp_ranks` under `routing_lock` (see **Two-lock usage (source of truth)**).
  - `open_admission(dp_ranks)`: 
    - Add `dp_ranks` to `active_dp_ranks` under `routing_lock` (see **Two-lock usage (source of truth)**).
    - If `need_suspend` was True: set `need_suspend=False`, set `suspend_notifier` to unblock queued requests.
**Error conditions**:
- Raise `RuntimeError("No active workers and not suspended")` if `generate_one_request` is reached with an empty `active_dp_ranks` set while `need_suspend` is False.
- This ensures that if expansion fails or state diverges, the pipeline fails loudly rather than hanging forever.

**Validation**:
- Verify `generate_one_request` blocks during shrink-to-zero; expand then verify it unblocks and completes.
- Verify `RuntimeError` when `active_dp_ranks` is empty and `need_suspend` is False.

### High: `model_update` is full-cluster only (subset sync missing)

**Verified**.

**Evidence**: `third_party/ROLL/roll/distributed/executor/model_update_group.py` calls `start_model_update` on all `src_cluster.workers` and does not accept `dp_ranks`.

**Required fix**:
- Add subset-aware sync-on-resume (either via porting `ModelUpdateService` from the fork or by extending `ModelUpdateGroup` with a `dp_ranks` argument).

**Validation**:
- Verify subset update only touches specified ranks.
- Verify shrink some workers; update remaining; no deadlock.

### High: Upstream API Change (`broadcast_bucket` removed)

**Verified**.

**Evidence**: Upstream ROLL commit `3077bef` removed `broadcast_bucket` and replaced it with `update_parameter_in_bucket`.

**Required fix**:
- Migrate `ROLL_multi_pipeline`'s `broadcast_bucket` usage to `update_parameter_in_bucket`.
- Support **Selective Model Update**: The new `update_parameter_in_bucket` must support broadcasting to a subset of workers (selective update on resume).
- Support **Bucket Caching**: Ensure `ROLL_VLLM_STAGE_MODEL_UPDATE_BUCKETS=1` (or equivalent logic) works with `update_parameter_in_bucket` to cache the serialized bucket on the trainer side for efficient broadcasting to multiple rollout workers.
- Other env-based flags from the old API can be ignored for now.

**Validation**:
- Verify model updates work with the new API.
- Verify selective updates target correct workers.
- Verify bucket caching provides performance benefit (or at least works).

### High: SchedRL Client/Scheduler API Missing (`release_and_request` & `notify_ready_to_release`)

**Verified missing in upstream**.

**Evidence**:
`release_and_request` (implemented as `release_and_request_gpus` in fork) and `notify_ready_to_release` are core time-sharing primitives but missing in upstream ROLL and SchedRL.

**Required fix**:
- Implement `release_and_request` (Atomic Release+Request) in **SchedRL Scheduler**.
- Implement `notify_ready_to_release` (Blocking Planned Release) in **SchedRL Scheduler**.
- Expose these via **`schedrl.client`** for Coordinator to call.
- These are **Coordinator → Scheduler** calls (not Adapter RPCs).

**Validation**:
- Verify atomic handoff between training and inference clusters.
- Verify gap-ratio fairness relies on `notify_ready_to_release`.

### Medium: Progress reporting (`report_progress`) missing in upstream ROLL

**Verified missing**.

**Evidence**: no `report_progress` in `third_party/ROLL/roll/**`.

**Required fix**:
- Add progress computation + emission in ROLL adapter or a small hook point:
  - trajectory units
  - emit at batch start + 2% band crossings
  - TODO: add `oldest_unfinished_creation_ts` later (requires tracking creation ts; upstream GroupQueue currently tracks `create_step`, not wall-clock).

Additional trigger (plan-only clarification):
- Also emit progress when the training-side sample buffer dequeues trajectories for training (so the scheduler sees instantaneous progress changes).

Plan-only note:
- If percent completion resets between steps (e.g. 100% → 0% for next step), treat each step independently; the scheduler should not assume monotonic progress across steps.

**Validation**:
- Verify band-crossing logic.
- Verify scheduler receives progress updates.

### Medium: Library Mode connect(get-or-create) race handling must be implemented in `schedrl/client`

**Verified missing in code** (schedrl not implemented yet).

**Required fix**:
- Implement connect semantics: get-then-create, handle “already exists” by re-get with backoff.

### Low: vLLM offload semantics need runtime verification

**Verified as an environment-dependent risk**.

**Evidence**: `VllmStrategy.offload_states` calls `await self.model.offload_states(self.sleep_level)` and flips `is_model_in_gpu=False`, but actual memory release depends on vLLM.

**Required plan note**:
- Assume offload is sufficient for ENG-123. Defer GPU memory measurement until after ENG-123 and return `-1` sentinel values in `release_reports` for `free_bytes_by_gpu` / `total_bytes_by_gpu`.
- TODO (later): add a black-box runtime measurement returned from the adapter after memory-release operations (shrink/offload) so the scheduler can verify GPUs are actually freed.

**Implementation sketch (deferred; Phase 3+ later)**:
- After completing shrink/offload/stop on targeted dp workers, measure locally post-release free memory per GPU and include it in the release ACK payload (scheduler never polls).
- Use a black-box approach that does not depend on vLLM internals (e.g. `torch.cuda.mem_get_info(device)` where available).

**Scheduler behavior (Phase 2/3)**:

**Push-based reporting (no polling)**:
- The scheduler MUST NOT poll workers for memory.
- Later (after ENG-123), the entity that completes GPU release will include post-release memory snapshot in the same ACK path.

**Scheduler behavior (Phase 2/3)**:
- Scheduler records/logs this data and can optionally validate thresholds before reassigning those GPUs.

Note (implementation reference):
- The forked, working implementation under `third_party/ROLL_multi_pipeline/` already implements full multi-pipeline time-sharing semantics, including support for shrinking to zero active DP ranks and selective model-update on resume. When addressing the gaps in upstream `third_party/ROLL/` (for example, relaxing the `Cannot shrink to zero active ranks` guard), implementers should consult the multi-pipeline reference code:
  - `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/generate_scheduler.py` (DP subset shrink/expand logic and mapping cleanup)
  - `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py` (centralized planning patterns)
  - `third_party/ROLL_multi_pipeline/roll/distributed/executor/model_update_service.py` (selective sync-on-resume helpers)

  Port these reference behaviors into the ROLL Adapter and/or `schedrl` runtime as the canonical implementation for shrink-to-zero and subset sync semantics.


### Verified Issues & Implementation Gaps

This section consolidates all issues discovered during the deep verification of the `ROLL_multi_pipeline` fork, `start_multi_pipeline_test.py`, and the legacy codebase analysis. These are mandatory constraints for the extraction and porting process.

#### 0. Additional P0/P1 Findings (2026-02-11 audit rounds; MUST be reflected in ENG-123 scope)

These findings were discovered by scanning upstream `third_party/ROLL` for multi-pipeline hazards that are **not** explicitly covered by the existing ENG-123 plan text. They are **not** “upstream bugs” in isolation; they become critical because post-extraction we run **multiple pipelines in the same Ray job/cluster** and then tear them down/reuse resources during the same job.

**Namespace decision (systematic fix; preferred)**:
- Most multi-pipeline hazards below are caused by **fixed Ray named actors** and `get_if_exists=True` being used in a shared namespace.
- ENG-123 adopts **per-pipeline ROLL namespaces** (`pipeline_{pipeline_id}_NS`) for isolation of all ROLL actors, instead of rewriting every actor name to include `{pipeline_id}`.
- This is mandatory: do not rely on “implicit default namespace” behavior. Every ROLL actor creation/lookup MUST use the per-pipeline namespace (directly or via a single `RAY_NAMESPACE` value derived from `pipeline_id` at process start).
- The only exception is the job-global `SharedStorage` actor (see `global_storage_namespace` below).

**P0 (NEW): Worker Ray actor-name collisions across pipelines (hard crash on 2nd pipeline)**
- **Context / Failure Mode**: Upstream ROLL names every Worker actor as `{cluster_name}-{rank}(-G...)` (no `pipeline_id`). If two pipelines both create a cluster with the same `cluster_name` (e.g. `actor_infer`), Ray named-actor creation collides and the second pipeline fails during initialization.
- **Evidence**: `Cluster._create_workers` builds `worker_name = f"{self.cluster_name}-{rank}..."` and passes it to `.options(name=worker_name, namespace=RAY_NAMESPACE)`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Every ROLL actor MUST live in `pipeline_{pipeline_id}_NS` (per-pipeline namespace). No ROLL actor creation/lookup may use the shared `roll` namespace in ENG-123 Library Mode.”; **Touchpoint**: `third_party/ROLL/roll/distributed/executor/cluster.py` (`.options(namespace=...)` for workers).

**P0 (NEW): RolloutScheduler named actors collide across pipelines**
- **Context / Failure Mode**: Agentic pipeline creates `RolloutScheduler-train` and `RolloutScheduler-val` with fixed names. In multi-pipeline, the second pipeline crashes on named actor creation.
- **Evidence**: `agentic_pipeline.py` uses `.options(name="RolloutScheduler-train")` / `.options(name="RolloutScheduler-val")`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “RolloutScheduler actors MUST be created/looked up in `pipeline_{pipeline_id}_NS`.”; **Touchpoint**: `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py` (RolloutScheduler `.options(namespace=...)`).

**P0 (NEW): RewardScheduler is global + `get_if_exists=True` (cross-pipeline state bleed)**
- **Context / Failure Mode**: RewardScheduler is created as a Ray named actor with `get_if_exists=True`, and env managers fetch it by name. In multi-pipeline, a pipeline can accidentally connect to another pipeline’s RewardScheduler (silent correctness failure).
- **Evidence**:
  - Create: `RewardScheduler-{reward.name}`, `get_if_exists=True` in `agentic_pipeline.py`.
  - Fetch: `ray.get_actor(name=f"RewardScheduler-{pipeline_config.reward.name}")` in `vl_traj_env_manager.py`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Any `ray.get_actor(...)` lookup MUST target `pipeline_{pipeline_id}_NS` (not the shared `roll` namespace).”; **Touchpoint**: `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py` (creation) + `third_party/ROLL/roll/pipeline/agentic/env_manager/vl_traj_env_manager.py` (lookup).

**P0 (NEW): Global dataset actors collide (GlobalDataset / GlobalDatasetManager / val_dataset_manager)**
- **Context / Failure Mode**: Several dataset actors are created as named actors with `get_if_exists=True` using only `mode` / `dataset_name`, and `val_dataset_manager` uses a fixed name. Multi-pipeline can share datasets and iterators unexpectedly (silent correctness failure).
- **Evidence**:
  - GEM Math: `GlobalDataset` name `f"{mode}_{dataset_name}"` and `GlobalDatasetManager` name `f"{mode}_dataset_manager"`.
  - DeepEyes: dataset name `f"{mode}_deepeyes"`, dataset manager name `f"{mode}_dataset_manager"`.
  - Agentic pipeline: `val_dataset_manager` fixed name.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Dataset actors MUST live in `pipeline_{pipeline_id}_NS`; no dataset actor may be job-global unless explicitly designed.”; **Touchpoint**: `third_party/ROLL/roll/pipeline/agentic/env/gem/math_env.py`, `third_party/ROLL/roll/pipeline/agentic/env/deepeyes/env.py`, `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py`.

**P0 (NEW): GlobalLimiter named actor collides (cross-pipeline throttling)**
- **Context / Failure Mode**: Env step concurrency limiter is a global named actor `GlobalLimiter_{tag}` with `get_if_exists=True`. Two pipelines using the same tag will throttle each other.
- **Evidence**: `LimiterClient._initialize_limiter()` uses `GlobalLimiter.options(name=f"GlobalLimiter_{tag}", get_if_exists=True, namespace=RAY_NAMESPACE)`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Limiter actors MUST live in `pipeline_{pipeline_id}_NS` (per-pipeline), unless explicitly configured as a shared job-global limiter.”; **Touchpoint**: `third_party/ROLL/roll/utils/env_action_limiter.py`.

**P0 (NEW): RLVR GenerateScheduler actor collides**
- **Context / Failure Mode**: RLVR pipeline creates `GenerateScheduler` with name derived only from `actor_infer.cluster_name` and `get_if_exists=True`. If multiple pipelines share the same cluster name, they share a scheduler actor.
- **Evidence**: `GenerateScheduler.options(name=f"{GENERATE_SCHEDULER_NAME}_{self.actor_infer.cluster_name}", get_if_exists=True, ...)`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “RLVR helper actors MUST live in `pipeline_{pipeline_id}_NS`.”; **Touchpoint**: `third_party/ROLL/roll/pipeline/rlvr/rlvr_math_vlm_pipeline.py`.

**P0 (NEW): AsyncGenerateScheduler GlobalCounter collides**
- **Context / Failure Mode**: AsyncGenerateScheduler uses a named `GlobalCounter` actor `DynamicSchedulerRequestCounter` with `get_if_exists=True`. Multiple pipelines can share the same counter, producing cross-pipeline request_id/counter coupling.
- **Evidence**: `GlobalCounter.options(name="DynamicSchedulerRequestCounter", get_if_exists=True, ...)`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Counters MUST live in `pipeline_{pipeline_id}_NS` (never job-global by accident).”; **Touchpoint**: `third_party/ROLL/roll/distributed/scheduler/async_generate_scheduler.py`.

**P0 (NEW): Megatron model_update locker collides**
- **Context / Failure Mode**: Megatron model update uses a global named locker actor `model_update_locker` with `get_if_exists=True`. Multi-pipeline can serialize across pipelines unexpectedly or deadlock.
- **Evidence**: `Locker.options(name="model_update_locker", get_if_exists=True, ...)`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Any synchronization primitive represented as a named actor MUST live in `pipeline_{pipeline_id}_NS`.”; **Touchpoint**: `third_party/ROLL/roll/third_party/megatron/model_update.py`.

**P0 (NEW): SGLang multi-node slave actor names collide**
- **Context / Failure Mode**: SGLang creates `sglang-slave-{i}` named actors with no `pipeline_id`. Multi-pipeline crashes on actor creation.
- **Evidence**: `.options(name=f"sglang-slave-{i}", namespace=RAY_NAMESPACE, ...)`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “All SGLang auxiliary actors MUST live in `pipeline_{pipeline_id}_NS`.”; **Touchpoint**: `third_party/ROLL/roll/distributed/strategy/sglang_strategy.py`.

**P0 (NEW): Placement groups are created but never destroyed (GPU reservation leak across pipeline teardown)**
- **Context / Failure Mode**: ResourceManager creates placement groups on init; there is a `destroy_placement_group()` method but it has no callers. Post-extraction, we frequently tear down pipelines and reuse GPUs within the same job; leaked PGs can prevent subsequent allocations and block the scheduler.
- **Evidence**: PG creation in `ResourceManager.__init__` and unused `destroy_placement_group()`.
- **Owner/Invariant/Touchpoint**: **Owner=SchedRL** (orchestrator teardown owns lifecycle); **Invariant**: “Pipeline teardown MUST release Ray placement groups associated with the pipeline’s ResourceManager/allocation before returning GPUs to `idle_gpus`.”; **Touchpoint**: `third_party/ROLL/roll/distributed/scheduler/resource_manager.py` (ensure destroy is called from orchestrator teardown path) and/or the extracted `schedrl` ResourceManager wrapper.

**P0 (NEW): LogMonitorListener/ExceptionMonitor can kill the job Ray cluster (violates “SchedRL owns lifecycle”)**
- **Context / Failure Mode**: `LogMonitorListener.start()` registers an `atexit` hook that calls `ray.shutdown()` and `ray stop --force`, and uses a fixed named actor `ExceptionMonitor` with `get_if_exists=True`. In a multi-pipeline job, this can:
  - create named actor collisions, and/or
  - tear down the entire job-scoped Ray cluster outside SchedRL’s orchestrator control.
- **Evidence**: `atexit.register(self.stop)` and `stop()` executes `ray.shutdown()` + `ray stop --force`; `ExceptionMonitor.options(name="ExceptionMonitor", get_if_exists=True, ...)`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL** (must not self-manage cluster lifecycle in Library Mode); **Invariant**: “In ENG-123 Library Mode, ROLL components MUST NOT call `ray.shutdown()` or `ray stop --force` via `atexit` or background threads; only SchedRL orchestrator may stop the cluster.”; **Touchpoint**: `third_party/ROLL/roll/distributed/scheduler/log_monitor.py` (disable/guard `atexit` + name-scope ExceptionMonitor or remove it from library path).

**P0 (NEW): SharedStorage rendezvous key is `cluster_name` only (cross-pipeline overwrite)**
- **Context / Failure Mode**: Worker rank0 writes master rendezvous info under key `self.cluster_name`. If two pipelines share the same `CLUSTER_NAME` (very likely: `actor_infer`, `actor_train`), they overwrite each other’s MASTER_ADDR/PORT metadata.
- **Evidence**: `self.shared_storage.put.remote(self.cluster_name, {"MASTER_ADDR": ..., "MASTER_PORT": ...})`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Any SharedStorage key used for rendezvous/cluster metadata MUST be pipeline-scoped (e.g. `{pipeline_id}:{cluster_name}`), never `cluster_name` alone.”; **Touchpoint**: `third_party/ROLL/roll/distributed/executor/worker.py` (SharedStorage key).

**P0 (NEW): SharedStorage `delete_prefix(pipeline_id)` cleanup is incompatible with current port-lock key schema**
- **Context / Failure Mode**: The plan mandates `SharedStorage.delete_prefix(pipeline_id)` on pipeline teardown, but Worker’s port-lock keys are `MASTER_ADDR_PORT:{ip}:{port}` (no `pipeline_id`). Cleanup cannot release these locks deterministically, causing port exhaustion/collision over repeated create/teardown within a job.
- **Evidence**: `master_addr_port_key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"` in `Worker.get_free_port()`.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “All per-pipeline ephemeral keys (ports, rendezvous, metadata) MUST include `{pipeline_id}` so teardown can delete them by prefix.”; **Touchpoint**: `third_party/ROLL/roll/distributed/executor/worker.py` (`MASTER_ADDR_PORT` key scheme + claim API).

**SharedStorage namespace rule (job-global; required for cross-pipeline coordination)**
- **Decision**: Keep existing “create-if-exists / `get_if_exists=True`” logic, but force a single namespace for the storage actor:
  - `global_storage_namespace`
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “All SharedStorage get/create/lookup MUST use `namespace='global_storage_namespace'` so the same storage actor is shared across all pipelines (even when all other ROLL actors run in `pipeline_{pipeline_id}_NS`).”; **Touchpoint**: `third_party/ROLL/roll/distributed/executor/worker.py` and `third_party/ROLL/roll/utils/checkpoint_manager.py` (SharedStorage `.options(namespace=...)`).

**P0 (NEW if shrink-to-zero is required): Upstream RequestScheduler forbids shrink-to-zero (plan contradiction risk)**
- **Context / Failure Mode**: If ENG-123 time-sharing requires full shrink-to-zero active ranks, upstream `RequestScheduler` currently throws `ValueError("Cannot shrink to zero active ranks")`.
- **Decision required**: The plan must explicitly state whether shrink-to-zero is a hard requirement in ENG-123. If yes, this guard must be removed and empty-rank routing behavior must be defined.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “If `sleep_level==2` time-sharing is in scope, RequestScheduler MUST support shrinking to zero active ranks and must not throw when active set is empty; routing must suspend before any empty state is visible to request generation.”; **Touchpoint**: `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py` (`Cannot shrink to zero` guard + empty-rank selection paths).

**P1 (NEW): `RolloutScheduler.mode_switch_lock` appears unused (outer serialization is currently dead)**
- **Context / Failure Mode**: `RolloutScheduler` defines `mode_switch_lock = asyncio.Lock()` with a comment “Prevent concurrent shrink/expand”, but shrink/expand are delegated to `RequestScheduler` and there is no evidence this lock is used to serialize lifecycle operations. If implementers assume this lock provides safety, they may accidentally introduce concurrent lifecycle operations or regress lock ordering.
- **Evidence**: `mode_switch_lock` is instantiated but not referenced in this file.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “There MUST be a single, well-defined outer lifecycle serialization lock for shrink/expand (either use `mode_switch_lock` consistently or remove it and enforce serialization at the adapter/SchedRL layer); do not rely on dead locks.”; **Touchpoint**: `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py` (`mode_switch_lock` usage/removal + shrink/expand call sites).

**P1 (NEW): Expand rebalance loop has a documented ‘may cycle indefinitely’ risk**
- **Context / Failure Mode**: The proportional rebalance on expand uses `cycle(...)` round-robin selection. The upstream code comment warns it may cycle indefinitely if all lists are exhausted before reaching target. In multi-pipeline time-sharing, this can hang lifecycle transitions, blocking the scheduler.
- **Evidence**: The comment in `_rebalance_on_expand` explicitly calls out the indefinite cycling risk.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Rebalance selection loops MUST have an explicit termination condition and MUST fail-fast with a clear error if the target cannot be met (no infinite `cycle()` loops).”; **Touchpoint**: `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py` (`_rebalance_on_expand` loop / termination condition).

**P1 (NEW): Ray retry semantics plan text is underspecified / uses wrong knob name**
- **Context / Failure Mode**: The plan says “set `max_retries=0` for all SchedRL lifecycle actors”, but Ray’s retry knobs for actors/tasks are typically `max_restarts` / `max_task_retries` / `retry_exceptions`, not `max_retries` (which is not a standard Ray actor option). This creates a false sense of “no retries/no replays”.
- **Owner/Invariant/Touchpoint**: **Owner=SchedRL**; **Invariant**: “All SchedRL lifecycle actors MUST set Ray retry knobs correctly (`max_restarts=0`, tasks `max_task_retries=0`, and avoid implicit retries) to preserve ‘no retries/no replays’ contract.”; **Touchpoint**: `schedrl` actor creation sites (Orchestrator/Scheduler `.options(...)` / `@ray.remote(...)`).

**P1 (NEW): HF isolation must be done at process start (Worker.__init__ is too late with current imports)**
- **Context / Failure Mode**: Upstream `worker.py` imports `download_model` at module import time, and `checkpoint_manager.py` imports `huggingface_hub` at import time. Any attempt to set `HF_HOME` inside `Worker.__init__` is too late to prevent cache races if HF is imported before the env var is set.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL** (for import ordering) + **Owner=SchedRL** (for runtime_env enforcement); **Invariant**: “HF isolation env vars MUST be set via Ray `runtime_env.env_vars` before the worker process imports HF; do not rely on setting them inside `Worker.__init__`.”; **Touchpoint**: `third_party/ROLL/roll/distributed/executor/cluster.py` (runtime_env env_vars injection) + any HF import sites (e.g. `third_party/ROLL/roll/utils/checkpoint_manager.py`).

**P1 (NEW): CheckpointManager model-path caching keys are not pipeline-scoped**
- **Context / Failure Mode**: `download_model` caches `{node_ip}:{model_name_or_path}` in SharedStorage. In multi-pipeline, this can return a cached path created under another pipeline’s HF_HOME layout and/or with different model download settings.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Model download cache keys MUST be pipeline-scoped (include `{pipeline_id}`) or the cache MUST be explicitly documented as cross-pipeline shared and safe.”; **Touchpoint**: `third_party/ROLL/roll/utils/checkpoint_manager.py` (SharedStorage cache key format).

**P1 (NEW): Process-global env var mutation in base_config leaks across pipelines if they share a driver process**
- **Context / Failure Mode**: BaseConfig sets `os.environ["ROLL_LOG_DIR"]`, `PROFILER_OUTPUT_DIR`, `roll_RPC_TIMEOUT`, and merges `system_envs` globally. If multiple pipelines are configured/constructed within the same Python process (common in orchestrated runs), later pipelines can overwrite earlier pipeline’s env settings.
- **Owner/Invariant/Touchpoint**: **Owner=SchedRL**; **Invariant**: “Pipeline-specific env vars MUST be injected via Ray `runtime_env.env_vars` and MUST NOT be set via process-global `os.environ` in shared drivers; any unavoidable global env must be pipeline-invariant.”; **Touchpoint**: `third_party/ROLL/roll/configs/base_config.py` (global os.environ writes) + SchedRL adapter runner process boundaries.

**P1 (NEW): vLLM/FlashInfer cache paths depend on `WORKER_NAME`, which is overwritten and not pipeline-scoped**
- **Context / Failure Mode**: vLLM sets cache/workspace roots under `.../<WORKER_NAME>/...`, but `EnvironmentWorker` overwrites `WORKER_NAME` to `EnvironmentWorker_{rank}` (not pipeline-scoped). This creates collisions in shared home/cache directories across pipelines.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Cache/workspace directories MUST be pipeline-scoped independently of `WORKER_NAME` (use `{pipeline_id}`), and `WORKER_NAME` MUST not be overwritten in a way that breaks cache isolation.”; **Touchpoint**: `third_party/ROLL/roll/third_party/vllm/__init__.py` (cache root) + `third_party/ROLL/roll/pipeline/agentic/environment_worker.py` (WORKER_NAME overwrite).

**P1 (NEW): `state_offload_manger` uses a non-concurrency-safe env var cleanup**
- **Context / Failure Mode**: `state_offload_manger` sets `os.environ["roll_EXEC_FUNC_NAME"]` and then unconditionally `del os.environ["roll_EXEC_FUNC_NAME"]`. If multiple offload managers run concurrently in the same process, this can throw `KeyError` or corrupt profiling labels.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “Profiling context MUST be concurrency-safe (use `os.environ.pop(key, None)` or avoid global env mutation entirely for per-call labels).”; **Touchpoint**: `third_party/ROLL/roll/utils/context_managers.py` (`state_offload_manger`).

**P1 (NEW): Hidden retry loops violate the plan’s default fail-fast semantics**
- **Context / Failure Mode**: Several components implement retries (OpenAI proxy retries, RLVR sandbox reward retries, driver command retries). In ENG-123 we assume fail-fast on critical lifecycle actions; silent retries can mask faults and create unexpected latency/jitter in scheduling.
- **Owner/Invariant/Touchpoint**: **Owner=ROLL**; **Invariant**: “All retry loops that can affect scheduling latency or correctness MUST be either removed in Library Mode or made explicit and configurable (default: fail-fast) per ENG-123 policy.”; **Touchpoint**: `third_party/ROLL/roll/pipeline/agentic/llm_proxy/openai_proxy.py`, `third_party/ROLL/roll/pipeline/rlvr/rewards/code_sandbox_reward_worker.py`, `third_party/ROLL/roll/distributed/scheduler/driver_utils.py`.

**P0/P1 (DOC CORRECTION): Ray namespace statement must match actual split (SchedRL vs ROLL)**
- **Context / Failure Mode**: The plan currently states “All actors are in Ray namespace `schedrl`.” Upstream ROLL uses `RAY_NAMESPACE="roll"` and creates many named actors under that namespace. Implementers will mis-scope `ray.get_actor(...)` lookups and miss collisions if the plan’s namespace statement is taken literally.
- **Owner/Invariant/Touchpoint**: **Owner=SchedRL** (plan/spec correctness); **Invariant**: “SchedRL-owned actors live in namespace `schedrl`. ROLL-owned actors MUST run in `pipeline_{pipeline_id}_NS` (per-pipeline). SharedStorage MUST run in `global_storage_namespace` (job-global). Do not run multi-pipeline ROLL actors in the shared `roll` namespace in ENG-123.”; **Touchpoint**: Plan text at `### Standardized actor names (ENG-123)` and upstream `third_party/ROLL/roll/utils/constants.py` (`RAY_NAMESPACE`).

**Re-verified (already covered elsewhere in this plan; keep as P0)**:
- **Issue 86 (vLLM offload gating)**: `VllmStrategy.offload_states` skips offload if not colocated. See **P0: Issue 86**; touchpoint `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py`.
- **Issue 119 (SharedStorage lifecycle)**: SharedStorage must be durable (detached) across rank0 death. See **P0: Issue 119**; touchpoint `third_party/ROLL/roll/distributed/executor/worker.py` + `third_party/ROLL/roll/distributed/scheduler/storage.py`.
- **Issue 134 (MASTER_PORT race)**: get→put port claim is non-atomic. See **P0: Issue 134**; touchpoint `third_party/ROLL/roll/distributed/executor/worker.py`.

#### 1. Test Parity, Orchestration & Reliability
*Issues related to ensuring SchedRL matches the rigorous test conditions of the fork.*

**P0: Issue 21 & 215: Orchestrator Ownership (Service Mode Gap)**
- **Finding**: The fork's `MultiPipelineOrchestrator` explicitly creates and owns the `ResourceManager` and `CentralizedGPUScheduler` actors.
- **Fix**: In ENG-123 (Library Mode), the job-owned Orchestrator must create and own the lifecycle of core components (ResourceManager + Scheduler) and enforce a strict singleton pattern to prevent split-brain state. Service Mode ownership semantics are deferred.
- **Reviewed**: yes

**P2: Issue 23: Strict Freshness ("Force Fresh")**
- **Finding**: The fork's `initialize.py` handles Ray cluster lifecycle with `ray stop --force` and `ray start`. Need similar fresh start for SchedRL.
- **Fix**: Two-component approach:
  1. **SchedRL Launcher Utility** (manages Ray cluster lifecycle):
     - Detect if Ray cluster exists (like fork's `is_ray_cluster_running()`)
     - If yes, execute `ray stop --force` to kill existing cluster
     - Start fresh Ray cluster: `ray start --head` (rank 0) or `ray start --address` (workers)
     - Similar to fork's `start_ray_cluster()` in `initialize.py`
  2. **Orchestrator** (connects to existing Ray cluster):
     - Assumes Ray cluster already initialized by launcher
     - Creates SchedRL actors (ResourceManager, Scheduler) in the fresh cluster
     - No Ray cluster management in orchestrator
- **Launcher Script**: Runs on multiple nodes (MPI-style) to form Ray cluster before orchestrator starts
- **Reviewed**: Yes

**P1: Issue 25 & 31: [CLOSED - By Design] Global Fail-Fast & Force Shutdown**
- **Finding**: Tests kill ALL pipelines if one fails.
- **Resolution**: This is intentional fail-fast behavior per AGENTS.md. The `shutdown(force=True)` pattern is the correct implementation.
- **Reviewed**: Yes - closed as by design



**P0: Issue 35: Job Launching Gap (Test Runner vs System Orchestrator)**
- **Finding**: The fork's `MultiPipelineOrchestrator` handles both pipeline spawning (user-land) and admission/scheduling (system-land).
- **Fix**: Split responsibilities:
    1. **Launcher/Test Runner (ROLL Side)**: Reads YAMLs, spawns pipeline actors, monitors futures.
    2. **SchedRL Orchestrator (Service Side)**: Handles `register_pipeline`, `admit`, resource state management, and cleanup.
    SchedRL itself will **not** spawn pipeline actors.
- **Reviewed**: Yes

**P2: Issue 121: [CLOSED - Invalid] Inefficient Pipeline Monitoring Polling**
- **Finding**: `MultiPipelineOrchestrator.monitor_pipeline_completion` blocks unnecessarily.
- **Fix**: Use `num_returns=1` in `ray.wait`.
- **Review Decision**: Not a failure-handling issue. We block wait on all pipelines (`num_returns=len(futures)`) and any failure from any pipeline triggers a global `shutdown(force=True)`.
- **Reviewed**: Yes - invalid

#### 2. Runtime Correctness, Concurrency & State Safety
*Issues involving race conditions, deadlocks, and atomicity.*

**P1: Issue 29 & 68: Robust Timeouts (Client & Phase-Specific)**
- **Finding**: Fork uses `signal.alarm` for blocking calls and granular timeouts for Shrink/Alloc/Expand.
- **Fix**: Implement an explicit timeout pattern in the Client and adopt phase-specific timeouts in `CentralizedScheduler`. On timeout, call `orchestrator.shutdown(force=True, reason="timeout", source="phase5.X")` for controlled termination.
- **Reviewed**: yes

**P1: Issue 65, 218 & 104: [INVALID - By Design] Notification Protocol**
- **Status**: INVALID - The "notify once" protocol assumption makes this expected behavior, not a bug.
- **Component**: `CentralizedGPUScheduler.notify_ready_to_release()`
- **Finding**: Original issue described handling duplicate notifications with idempotency. However, per protocol, each trajectory should notify exactly once.
- **Expected Behavior**: On duplicate notification, treat as protocol violation and fail-fast via `orchestrator.shutdown(force=True, reason="duplicate_notification", source=...)`.
- **Reviewed**: Yes

**P0: Issue 81 & 124: Orphaned Expansion Signaling (Partial & Total Failure)**
- **Finding**: `_execute_expansions` and its wrapper loops fail to signal waiting events if RPCs fail (partially or totally), causing deadlocks.
- **Fix**: Catch expansion exceptions, log context, then call `orchestrator.shutdown(force=True, reason="expansion_failed", source="phase5.4")` for controlled termination.
- **Reviewed**: yes


**P1: Issue 105: RequestScheduler Lazy Query Race Condition**
- **Finding**: `_get_request_scheduler` performs lazy queries that can race → duplicate Ray queries and cache writes. Race is benign but wastes resources.
- **Fix**: Add single scheduler-level lock with double-checked locking pattern:
  ```python
  # Fast path: check cache without lock
  if cluster_name in cache:
      return cache[cluster_name]
  
  # Slow path: acquire lock and double-check
  async with self._scheduler_cache_lock:
      if cluster_name in cache:  # Double-check inside lock
          return cache[cluster_name]
      request_scheduler = ray.get_actor(...)
      cache[cluster_name] = request_scheduler
      return request_scheduler
  ```
- **Naming requirement (E2)**: actor lookup must use a pipeline-scoped name (include `pipeline_id` prefix) to avoid collisions across multiple pipelines in the same Ray namespace.
- **Reviewed**: Yes

**P0: Issue 107: State Inconsistency on Execution Phase Failure**
- **Component**: `CentralizedGPUScheduler.process_cycle()` - Phase 5 execution
- **Finding**: Execution failures (Phase 5) leave stale state because Phase 6 is skipped.
- **Fix**: Treat any execution-phase failure as fatal and immediately call `orchestrator.shutdown(force=True, reason=..., source=...)` to force-stop the job-scoped Ray cluster (`ray stop --force` on head).
- **Reviewed**: yes




**P1: Issue 151: [CLOSED - Invalid] Progress Reporting Race with Scheduler State**
- **Finding**: `report_progress` is async and can result in stale state usage.
- **Fix**: Use `ray.get()` or versioned progress cache.
- **Review Decision**: **[Invalid]**. Staleness is acceptable as progress is reported frequently enough. The minor delay in progress reporting does not impact scheduling correctness.
- **Reviewed**: Yes - invalid

#### 3. Resource Management, Leaks & Cleanup
*Issues related to GPU allocation, port management, and cleanup.*

**P0: Issue 28 & 72: Ray Resource Strategy (Fractional GPUs)**
- **Finding**: The fork allocates 0.01 GPUs per worker to allow time-sharing.
- **Fix**: SchedRL must mandate this fractional resource strategy.
- **Reviewed**: Yes

**P0: Issue 43 & 108: Resource Recovery on Unregister**
- **Component**: `CentralizedGPUScheduler.unregister_pipeline()`, `MultiPipelineOrchestrator.kill_pipeline()`
- **Finding**:
  1. `unregister_pipeline` only removes registry entry, leaving `active_allocations` and `pending_requests` (GPU leak)
  2. `kill_pipeline` only kills pipeline actor via `ray.kill(pipeline)`, not cluster workers (actor_train, actor_infer, critic)
- **Fix**: Complete pipeline removal requires:
  1. **Kill all cluster workers**: Iterate pipeline's clusters (actor_train, actor_infer, critic) and `ray.kill()` each worker
  2. **Clear active_allocations**: Remove all cluster entries for this pipeline_id from `active_allocations`, return GPUs to `idle_gpus`
  3. **Clear pending_requests**: Remove all pending GPU requests for this pipeline's clusters
  4. **Remove from registry**: Delete from `pipeline_registry`
  5. **Cleanup tracking**: Remove from `pipelines`, `run_futures`, `pipeline_states`
  6. **(New) Clear Shared Storage**: Call `shared_storage.delete_prefix.remote(pipeline_id)` to release ports and metadata.
- **Reviewed**: Yes

**P1: Issue 66: Selective Expansion Safety**
- **Finding**: Selective updates must only target newly added DP ranks to avoid NCCL conflicts.
- **Fix**: SchedRL Scheduler ensures the DP rank list (delta) is correct. Adapter validates this before execution.
- **Review Decision**: **[Invalid / Backlog]**. SchedRL passes the delta. Per-adapter validation/enforcement of selective updates is deferred to future work.
- **Reviewed**: Yes

**P1: Issue 75 & 141: MASTER_PORT Leak in SharedStorage**
- **Finding**: Ports are reserved but never released, leaking over time.
- **Fix**: Explicitly release ports and connection info during pipeline cleanup.
  1. Add `delete_prefix(prefix: str)` method to `SharedStorage` to allow wiping keys by pattern.
  2. Implement cleanup call in `MultiPipelineOrchestrator.cleanup_pipeline` as the **final step** (after killing workers): `self.shared_storage.delete_prefix.remote(pipeline_id)`.
  3. This ensures that `{pipeline_id}_actor_train` keys and associated port locks are freed for reuse.
- **Reviewed**: Yes

**P0: Issue 86: VllmStrategy Skipping Offload in Multi-Pipeline [CRITICAL]**
- **Finding**: `VllmStrategy.offload_states` skips offload if `is_actor_infer_colocated` is false, causing OOMs in multi-pipeline shared GPU setups.
- **Fix**: **Explicit API Control**: Update `vllm_strategy.py` to add `force=False` to `offload_states`. The implementation must check `if force or ...` to allow the SchedRL Adapter to mandate offloading regardless of `is_actor_infer_colocated` status.
- **Reviewed**: Yes

**P0: Issue 119: Transient `SharedStorage` Lifecycle**
- **Finding**: `SharedStorage` dies if its creator worker dies (e.g. during shrink-to-zero), causing `ActorDiedError` for other workers.
- **Fix**: Update `SharedStorage` creation in `Worker` and `CheckpointManager` to use `lifetime="detached"`. This ensures the actor survives even if the creator worker dies. Explicit Orchestrator management is not required because the Orchestrator tears down the entire Ray cluster on shutdown, which naturally cleans up the detached actor.
- **Reviewed**: Yes

**P2: Issue 154: [CLOSED - Invalid] Orchestrator Shutdown Resource Leak**
- **Finding**: Shutdown can leave orphaned actors.
- **Fix**: Wait for pipeline cleanup before killing the scheduler.
- **Review Decision**: Obsolete with `ray stop --force` design. `shutdown(force=True)` kills all actors immediately - no orphans possible.
- **Reviewed**: Yes - invalid


#### 4. Protocol, API & Mismatches
*Issues with RPC signatures and behavior.*

**P1: Issue 30 & 41: Adapter RPC API Alignment**
- **Finding**: Mismatches in `expand_workers` (needs selective sync logic) and Indexing (Physical vs Logical args).
- **Fix**: Update Adapter RPC surface to use `dp_ranks` directly as the lifecycle target identifier:
  - **Issue 30 (`expand_workers`)**: Adapter selects rollout `dp_ranks` only, then calls `ModelUpdateService` for selective sync. Active rollout checkpoint version is promoted by coordinator signaling and stored on sender strategy cache state.
  - **Issue 41 (Indexing)**: For ENG-123, `dp_ranks` are the canonical identity in adapter lifecycle APIs. Do not introduce separate logical-vs-physical index translation in the adapter contract.
- **Reviewed**: Yes


#### 5. Scheduling Policy & Fairness
*Issues affecting prioritization and starvation.*

 **P2: Issue 56: [CLOSED - Invalid] Priority Timestamp (FIFO Fairness) Hook**
 - **Finding**: Original plan suggested using `min(oldest_rollout_ts, request_ts)` for fairness.
  - **Fix**: None required for ENG-123. The plan adopts scheduler-side arrival time as the canonical FIFO source. `oldest_unfinished_creation_ts` reporting is deferred to backlog (see Section 10: Backlog) as it requires persistent wall-clock tracking.
 - **Review Decision**: **[Invalid]**. Implementation of wall-clock FIFO is deferred. Arrival-time FIFO is sufficient for initial multi-pipeline arbitration.
 - **Reviewed**: Yes

**P1: Issue 64: Gap-Ratio Starvation (Work Inflation)**
- **Finding**: New pipelines are starved (0 work) because they haven't reported progress yet.
- **Fix**: Port "work inflation" heuristic from fork (already implemented at line 2805-2810 in `centralized_gpu_scheduler.py`). SchedRL Scheduler adds `rollout_batch_size` to remaining work for pipelines with pending requests.
- **Review Decision**: **[Invalid]**. Already implemented in fork; needs porting to SchedRL Scheduler.
- **Reviewed**: Yes




**P0: Issue 132: [CLOSED - Invalid] Circular High-Priority Admission Deadlock**
- **Finding**: Concurrent initialization of pipelines causes a deadlock. Pipelines compete for Logical Locks (Scheduler) and Physical Resources (Ray Actors/PGs) simultaneously. P1 holds Physical and waits for Logical; P2 holds Logical and waits for Physical.
- **Fix**: Enforce **Global Sequential Admission** in `MultiPipelineOrchestrator`. Add a `ready()` method to `ConcurrentAgenticPipeline` and block on it (`ray.get(pipeline.ready.remote())`) inside `admit_pipeline` ensuring P1 is fully initialized (and has released startup resources) before P2 starts.
- **Review Decision**: **[Invalid]**. Analysis confirms that `ConcurrentAgenticPipeline.__init__` performs a **Blocking Offload** (`offload_states(..., blocking=True)`) before returning. This guarantees that P1 physically releases GPU memory before allowing P2 (or P1's next phase) to proceed, preventing the described deadlock. Any observed deadlocks are likely due to **Misconfiguration** (e.g., `sleep_level!=2`) or CPU/Actor slot exhaustion, not scheduler logic.
- **Reviewed**: Yes - invalid

**P2: Issue 137: [CLOSED - Invalid] Unfair Lexicographical Tie-Breaking**
- **Finding**: Tie-breaking by ID hurts pipelines with "larger" names.
- **Fix**: Implement Round-Robin or Random tie-breaking.
- **Review Decision**: **[Invalid]**. Not important for ENG-123; lexicographical tie-breaking is acceptable for initial release.
- **Reviewed**: Yes - invalid

**P1: Issue 143 & 147: [CLOSED - Invalid] Atomic & Safe GPU Release/Swap (Timeout Case)**
- **Finding**: Concern about `event.wait()` hanging indefinitely.
- **Resolution**: User feedback confirms waiting for resources is valid and expected behavior. Timeouts would prematurely kill healthy but waiting pipelines. `release_gpus` is already non-blocking.
- **Reviewed**: Yes - invalid

#### 6. Validation, Configuration & Environment
*Issues with input validation and env settings.*

 **P2: Issue 24: [CLOSED - Backlog] Typed Resource Requirements**
 - **Finding**: Fork uses raw strings for mapping (`tensor_parallel_size`, `tensor_model_parallel_size`).
 - **Proposed Fix**: Define `ResourceRequirements` dataclass for type safety and validation.
 - **Decision**: Keep current dict-based approach for ENG-123. Acknowledged as good future improvement but not required for initial release.
 - **Reviewed**: Yes - backlog

**P0: Issue 26, 208 & 241: Mandatory Offload Configuration & Validation**
- **Finding**: System fails if `sleep_level != 2` (no offload) or if `partial_gpu_mode=True` (pipeline self-management conflicts with SchedRL orchestration).
- **Fix**: Orchestrator must validate at **registration time** (`register_pipeline`):
  1. `sleep_level=2` - Required for full model offload
  2. `partial_gpu_mode=False` - Required to disable pipeline self-shrink/expand; SchedRL controls all GPU allocation
  This ensures fail-fast behavior and avoids wasting Ray actor creation on invalid configs.
- **Review Decision**: Verified that `sleep_level=2` validation exists in `multi_pipeline_orchestrator.py` and `agentic_pipeline.py`. The `enforce_eager` validation is not required for SchedRL. Added `partial_gpu_mode=False` validation per Issue 238 decision. Validation should be moved from admission to registration time for earliest failure.
- **Reviewed**: Yes

**P0: Issue 49: Port Collision & Shared Storage Coordination**
- **Finding**: Namespace collisions between SchedRL and ROLL.
- **Fix**: Ensure shared services are reachable or share the `schedrl` namespace.
- **Review Decision**: Invalid - By Design. The plan already specifies standardized SchedRL actor names under the `schedrl` namespace (`schedrl:orchestrator`, `schedrl:scheduler`, `schedrl:adapter:{pipeline_id}`), and Library Mode uses independent Ray clusters per job (no cross-job collisions). Note: within a single job (multi-pipeline), upstream ROLL scheduler helper actor names must be pipeline-scoped (see E2: `RolloutScheduler(pipeline_id=...)` prefixes `RequestScheduler`/`GroupQueueManager` names).
- **Reviewed**: Yes - invalid

**P0: Issue 51: Pipeline ID Delimiter Collision**
- **Finding**: `:` delimiter collision.
- **Review Decision**: Invalid - By Design. The plan already specifies this in `schedrl/protocol/request_id.py`: `validate_request_id` must reject `:` in `pipeline_id` or `traj_id`.
- **Reviewed**: Yes - invalid

**P1: Issue 62, 204 & 206: Hydra Config Resolution & Validation**
- **Finding**: Complex **Hydra** relative path resolution (Issue 206), strict env var validation for `BUCKET=1`, and mutual exclusion between `skip_load` and `selective_update`.
- **Fix**:
    - **Issue 206**: Adapter must resolve all relative paths to absolute paths before passing to Ray actors.
    - **BUCKET=1**: Adapter/Orchestrator validation must ensure `ROLL_VLLM_STAGE_MODEL_UPDATE_BUCKETS` is true (validating the feature flag is enabled).
    - **skip_load vs selective_update**: Keep using existing runtime checking (no new validation logic required).
- **Reviewed**: Yes

**P0: Issue 69, 80, 93 & 158: Topology Validation & Segment Alignment**
- **Finding**: Scheduler accepts invalid IDs, misaligned allocations, or non-contiguous shrinking (Partial Mode).
- **Fix**: Pass topology in `register`; validate IDs and enforce segment alignment/contiguity on all alloc/shrink ops.
- **Review Decision**: **[BACKLOG]**. Deferred as a safety enhancement. The current system assumes valid inputs. Runtime validation of physical topology constraints will be added later.
  - **Registration Time**: Validate `device_mapping` corresponds to physically contiguous/peer-accessible GPUs (intra-node segments should be contiguous on their respective nodes).
  - **Request Time**: Ensure resource requests (e.g. `request_gpus`) imply a topology that matches the registered constraints.
  - **Operational Time (Alloc/Shrink)**: Add strict validation to the **Validations Phase** of the scheduling loop (Phase 4). Specifically, enforce vertical alignment (e.g. shrinking maintains contiguous TP groups) and reject any allocation that splits a TP group across non-contiguous GPUs (e.g., `{1, 2}` for TP=2).
- **Reviewed**: Yes - backlog



**P0: Issue 85: Port Collision in Multi-Pipeline SGLang**
- **Finding**: Hardcoded ports in SGLang strategy cause collisions.
- **Fix**: Use `self.worker.get_free_port()` for SGLang (same as vLLM in `vllm_strategy.py:106`). Replace `"port": 30000 + dp_rank * 500` with dynamically allocated ports.
- **Review Decision**: Verified. `sglang_strategy.py:85` uses `"port": 30000 + dp_rank * 500` which causes collisions across pipelines (P1 rank 0 = 30000, P2 rank 0 = 30000). vLLM already uses `get_free_port()` correctly.
- **Reviewed**: Yes

**P0: Issue 87: ResourceManager Initialization Race**
- **Finding**: The current implementation has a flaky "sleep loop" (6s total) that crashes if Ray nodes take longer to appear. The current logic also has a bug where it `breaks` the loop prematurely on the first failure.
- **Fix**: Extend the polling duration to **300 seconds** and keep the existing adaptive logic.
  1. Increase the wait budget to 300s (e.g., `max_retries=300` with 1s sleeps).
  2. Preserve the "Inspect Node -> Create Bundle" logic to keep the adaptive CPU sizing (`node_cpu / 2`).
  3. Fix the loop logic to ensure it continues polling until the full 300s timeout is reached or nodes appear.
- **Reviewed**: Yes

**P2: Issue 90: [BACKLOG] Mixed TP Key Support**
- **Finding**: Configs use both `tensor_parallel_size` (vLLM/SGLang) and `tensor_model_parallel_size` (Megatron).
- **Proposed Fix**: Support unified alias mapping to canonical `tensor_parallel_size` key.
- **Decision**: Keep current approach for ENG-123 - handle key mapping in adapter code (like fork does in `_get_tp_size_for_worker`). Not required for initial release.
- **Reviewed**: Yes - backlog



**P2: Issue 102 & 242: [BACKLOG] Timeout Scaling & Overrides**
- **Finding**: `ROLL_TIMEOUT_SCALE` is used globally, and 64-GPU config (Issue 242) disables timeouts. Fixed timeouts are often too short for large-scale clusters (64+ GPUs) or too long for local debugging.
- **Proposed Fix**: Adopt `SCHEDRL_TIMEOUT_SCALE` as the standard and allow specific RPCs to pass an `override_timeout` for experimental configurations.
- **Decision**: Keep current timeout approach for ENG-123. Timeout scaling improvements are acknowledged as useful but not required for initial release.
- **Reviewed**: Yes - backlog

**P0: Issue 134: Master Port Collision Race Condition**
- **Finding**: Non-atomic check-then-act pattern for master ports in `SharedStorage`. Workers can interleave between checking if a port is free and reserving it. Furthermore, relying on Ray actor single-threading for atomicity is risky/flaky as it depends on `max_concurrency=1`, which is not enforced and may change during refactoring.
- **Fix**: Implement a synchronous, atomic `claim_port` method in `SharedStorage` and update `Worker`.
  1. **SharedStorage Implementation**: Add an internal `threading.Lock` and implement `claim_port(key, expected, new)` as an atomic Compare-And-Swap (CAS).
  2. **No Async Ripple**: The method must be synchronous (`def`, not `async def`) to maintain compatibility with synchronous Strategy initialization.
  3. **ObjectRef Compatibility**: Must use `storage[key] = ray.put(new)` to ensure compatibility with existing `get()` behavior.
  4. **Worker Update**: Replace the `get()` -> `put()` sequence in `Worker.get_free_port` with a single `ray.get(storage.claim_port.remote(key, None, True))` call.
- **Reviewed**: Yes





#### 7. Observability & Diagnostics
*Issues with logging and tools.*

**P2: Issue 22 & 106: [CLOSED - Invalid] Config Flexibility & Naming (Trackers & Clusters)**
- **Finding**: Manual config hacking for trackers (Issue 22) and hardcoded `"actor_infer"` cluster name (Issue 106).
- **Analysis**: SchedRL's design already solves this:
  - Uses `schedrl` namespace with standardized actor names (`schedrl:orchestrator`, `schedrl:scheduler`, `schedrl:adapter:{pipeline_id}`)
  - Each pipeline gets unique adapter name via `pipeline_id`
  - SchedRL orchestrator manages all tracker initialization centrally at registration time
  - No global registry conflicts like fork's `tracker_registry`
- **Decision**: No fix needed. SchedRL's namespace and naming scheme provides better isolation than fork.
- **Reviewed**: Yes - invalid

**P2: Issue 27, 61 & 210: [BACKLOG] Profiling & Timeline Tooling**
- **Finding**: `merge_timelines.py` and metadata hooks (`get_worker_metadata`, `_save_timeline_snapshot`) are useful for debugging performance gaps but missing from SchedRL plans.
- **Proposed Fix**: Port `merge_timelines.py` to `schedrl/tools/` and implement metadata hooks in the Adapter/Orchestrator.
- **Decision**: Defer to testing phase. Implement when profiling SchedRL implementation to visualize GPU utilization and scheduling decisions.
- **Reviewed**: Yes - backlog

**P2: Issue 101 & 103: Debugging & Diagnostics (NCCL & ResourceManager)**
- **Finding**: Fork hardcodes `NCCL_DEBUG=INFO` in orchestrator. Need global control for debugging NCCL deadlocks.
- **Fix**: Orchestrator shall accept arbitrary environment variables and propagate them globally to all Ray actors:
  1. Orchestrator accepts `env_vars` dict (e.g., `{"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"}`)
  2. SchedRL propagates these env vars to all pipeline workers via Ray `runtime_env`
  3. Global control only - no per-pipeline env var configuration
- **Priority**: Implement in Day 1 - essential for debugging distributed issues.
- **Out of Scope**: ResourceManager under-provisioning error message enhancement (Issue 103) - not required for ENG-123.
- **Reviewed**: Yes



#### 8. Category A: Dead Code & Legacy Artifacts (Do Not Port)
*Features that are implemented but unused, deprecated, or belong to legacy single-pipeline modes.*

**P2: Issue 201, 221 & 243: Legacy Scheduler Classes**
- **Finding**: `GenerateScheduler` (v1) uses a synchronous/threaded model with `threading.Lock` and is designed for static clusters. It lacks support for dynamic topology changes (shrink/expand). `RequestScheduler` (v2) is the active, async-first Ray implementation designed for multi-pipeline time-sharing.
- **Fix**: Port only `RequestScheduler`. It is the only class that implements the atomic routing locks and subset management required for SchedRL.
- **Reviewed**: Yes (Verified that `ConcurrentAgenticPipeline` exclusively uses `RequestScheduler` for all multi-pipeline paths)

**P2: Issue 219 & 225: Dead Files (`agentic_rollout_pipeline`, `user_defined_rollout_loop`)**
- **Finding**: Unused files.
- **Fix**: Do not port.
- **Reviewed**: Yes (Verified as not imported by core multi-pipeline components)

**P2: Issue 220: Dead File `agentic_actor_pg_worker.py`**
- **Finding**: Pure PG worker is unused.
- **Fix**: Do not port.
- **Reviewed**: Yes (Verified as obsolete due to unified `ResourceManager`)

**P2: Issue 222: Legacy `AgenticPipeline` Class**
- **Finding**: Unused class.
- **Fix**: keep upstream `AgenticPipeline` class.
- **Reviewed**: Yes (Verified that `ConcurrentAgenticPipeline` is the active orchestrator)

**P2: Issue 223: Legacy Initialization & Entry Point**
- **Finding**: The source repo (`ROLL_multi_pipeline`) uses an `init()` that aggressively force-kills and restarts Ray, breaking compatibility with managed clusters. Upstream `ROLL` is safer (idempotent), but we are porting the fork's logic which introduced this regression.
- **Fix**:
  1.  **Refactor `initialize.init()`**: Adopt **"Connect-First, Fallback-Local"** pattern (like NeMo-RL). Try `ray.init(address="auto")`; only provision local Ray if connection fails. Remove aggressive `ray stop`.
  2.  **Port Entry Point**: Port `examples/multi_pipeline/start_multi_pipeline_test.py` to `examples/run_schedrl.py` as the canonical entry point.
  3.  **Framework Integration Guide**: Insert SchedRL initialization immediately after Ray init in each framework:

| Framework | Entry Script | Ray Init Location | SchedRL Hook Location | Hook Pattern |
|-----------|-------------|-------------------|----------------------|--------------|
| **ROLL (Upstream)** | `examples/start_agentic_pipeline.py` | Line 26: `init()` | **Line 27** (after `init()`) | `schedrl.init()` |
| **ROLL (Fork)** | `examples/multi_pipeline/start_multi_pipeline_test.py` | Line 240: `init()` | **Line 248** (after cluster verification) | `schedrl.init()` |
| **NeMo-RL** | `examples/run_grpo_math.py` | Line 162: `init_ray()` | **Line 163** (before tokenizer setup) | `schedrl.init()` |
| **SkyRL** | `examples/algorithms/dapo/main_dapo.py` | Line 88: `initialize_ray(cfg)` | **Line 89** (before `ray.get()`) | `schedrl.init(cfg)` |
| **Miles** | `examples/formal_math/single_round/run.py` | Line 174: `U.execute_train()` → `command_utils.py:133` | **Use `before_ray_job_submit` callback** | `U.execute_train(..., before_ray_job_submit=schedrl.init)` |

  4.  **Multi-Node Execution Behavior** (VERIFIED):
      - **ROLL**: Entry script runs on **ALL nodes** (via `torchrun`/MPI). Worker nodes (rank > 0) execute `ray start` then `sys.exit(0)` at Line 86/164. **SchedRL init must check rank** and only run on rank 0.
      - **NeMo-RL**: **VERIFIED via `ray.sub`** - Slurm script provisions Ray cluster on all nodes (Lines 274-409: `ray start --head` on node 0, `ray start --address` on workers). Entry script (`COMMAND`) runs **ONLY on head node** via `srun --overlap --nodes=1 -w $head_node` (Line 451). SchedRL init runs once.
      - **SkyRL**: Similar to NeMo-RL - uses SkyPilot/Slurm for cluster provisioning. Entry script runs on single driver node.
      - **Miles**: **Two modes** (controlled by `MILES_SCRIPT_EXTERNAL_RAY` env var):
          - **External Ray** (`MILES_SCRIPT_EXTERNAL_RAY=1`): Assumes Ray cluster pre-provisioned (like NeMo-RL). Entry script runs on single node, connects via `ray job submit`.
          - **Local Ray** (default): `execute_train()` calls `ray start --head` locally (Line 133), then submits job via `ray job submit` (Line 180). Single-node execution.
      - **Implication**: SchedRL init logic must be **rank-aware for ROLL only**. For NeMo-RL/SkyRL/Miles, simple single-execution init is sufficient (runs once on driver node).

- **Reviewed**: Yes (Verified exact integration points, line numbers, and multi-node execution patterns with source code evidence for all frameworks)

**P2: Issue 224: Disabled `DynamicSamplingScheduler`**
- **Finding**: Used only by RLVR.
- **Fix**: Do not port.
- **Reviewed**: Yes (Verified as tied to out-of-scope `rlvr` package)

**P2: Issue 226 & 239: Dead Package `roll/pipeline/rlvr`**
- **Finding**: RLVR is out of scope.
- **Fix**: Exclude.
- **Reviewed**: Yes (Verified as out of scope for ENG-123)

**P2: Issue 227, 230, 232: Dead/Empty Access Patterns**
- **Finding**: `agent_native_env_manager`, `tir_env_manager`, `step_concat`, `vl_traj` are unused.
- **Fix**: Do not port.
- **Reviewed**: Yes (Verified as abandoned environment management strategies)

**P2: Issue 231, 234: Dead Utilities**
- **Finding**: `utils/local_code`, `impl specific tests` are unused.
- **Fix**: Exclude.
- **Reviewed**: Yes (Verified as developer-specific or redundant utilities)

**P2: Issue 233 & 240: Dead Script `start_dual_pipeline_test.py`**
- **Finding**: Deprecated.
- **Fix**: Do not port.
- **Reviewed**: Yes (Verified as superseded by `start_multi_pipeline_test.py`)

**P2: Issue 235: Dead Code in `CentralizedGPUSchedulerImpl`**
- **Finding**: Unused methods (`_plan_generation_fifo`) and legacy stubs in `CentralizedGPUSchedulerImpl` (not the fork's production `CentralizedGPUScheduler`).
- **Fix**: Do not port.
- **Reviewed**: Yes (Verified as unused stubs/legacy methods)

#### 9. Category B: Broken or Disabled Features
*Features that are present but broken or commented out.*

**P2: Issue 205: Proactive Release Implementation**
- **Finding**: **ACTIVE - NOT DISABLED**. `release_gpus()` at line 592 is fully implemented and actively used by `concurrent_agentic_pipeline.py`. Triggers proactive expansion via `wakeup_event.set()` at line 667. Deadlock comments (lines 565-568) refer to fire-and-forget notification pattern safety, NOT feature disablement.
- **Fix**: Port standard `release_gpus()` logic; **DO NOT port** the proactive `notify_ready_to_release` call inside `GroupQueueManager.put` (currently suppressed by early return in fork).
- **Reviewed**: Yes

**P2: Issue 228: Dead/Duplicate `GroupFilter` Logic**
- **Finding**: "Smart" filter is unused; system uses a no-op implementation.
- **Fix**: Do not port. Discard the class entirely.
- **Reviewed**: Yes (Verified as dead code; no-op implementation is unnecessary for SchedRL)

**P2: Issue 229: Broken Example Config**
- **Finding**: Config points to missing files or uses hardcoded paths.
- **Fix**: Port the working examples:
  1. `examples/multi_pipeline/pipeline1_sokoban_grpo.yaml` (Clean reference implementation)
  2. `examples/multi_pipeline/pipeline2_sokoban_grpo.yaml` (Companion pipeline)
  3. `examples/multi_pipeline/start_multi_pipeline_test.py` (Main entry point for multi-pipeline testing)
- **Reviewed**: Yes (Verified as the canonical multi-pipeline example set)

**Note (policy update)**:
- This plan merged the boundary statement and patch policy into a single top-level section: **Option A boundary + framework patching policy (ENG-123)**.
- When interpreting “port examples” in ENG-123, apply that merged policy: keep scheduler logic in `schedrl/`, keep ROLL-specific mechanics behind the Adapter/shim, and only apply minimal upstreamable ROLL patches when adapter-only alternatives are clearly worse.

**P0: Issue 236 & 217: [CLOSED - Invalid] Robust Suspend/Resume (Shrink-to-Zero)**
- **Finding**: Multi-pipeline time-sharing requires pipelines to "shrink to 0" workers, but this logic is fragile/disabled.
- **Fix**: Re-enable `suspend()` calls and harden the 0-worker topology handling in the Adapter/Scheduler.
- **Review Decision**: Already covered in detail under "Critical 1: Shrink-to-zero logic and lifecycle transitions" section above. See lines ~95-115 for required fix and validation.
- **Reviewed**: Yes - invalid

**P2: Issue 237: Broken Validation Logic**
- **Finding**: Validation logic is dead/commented out.
- **Fix**: Do not port broken `val()`; reimplement if needed.
- **Reviewed**: Yes (Verified that validation logic is unreachable/dead)

#### 10. Category C: Configuration Risks & Hardcoded Overrides
*Configurations that are risky or hardcoded.*

**P2: Issue 238: [CLOSED - Invalid] Hardcoded Partial GPU Mode**
- **Finding**: `ConcurrentAgenticPipeline` hardcodes `partial_gpu_mode = True` for internal shrink/expand logic.
- **Decision**: **Disable `partial_gpu_mode` entirely** for ENG-123. SchedRL takes full control of GPU allocation.
- **Rationale**:
  - SchedRL acts as "Cluster OS" - pipelines should not self-manage physical resources
  - Eliminates race conditions between pipeline implicit state and SchedRL decisions
  - Reduces complexity (~100 lines of state-tracking code replaced by 2 explicit RPCs)
- **Implementation**:
  - Set `partial_gpu_mode = False` in pipeline configuration
  - SchedRL Adapter handles `expand_workers()` and `shrink_workers()` RPCs
  - **Critical**: Adapter must update `GroupQueueManager` routing before releasing GPUs (replicate old `shrink_sampler` side effect)
- **Reviewed**: Yes - invalid





#### 11. Category D: Implementation Deficiencies (Port with Care)
*Functional but fragile logic.*

**P1: Issue 202 & 216: Proportional Rebalancing & Session Migration**
- **Finding**: `RequestScheduler.rebalance_on_expand` implements a specific "proportional abort" strategy: when expanding, it calculates how many sticky sessions (src_ranks) to keep on old workers and aborts the rest, forcing them to re-route to new workers.
- **Fix**: SchedRL's `RequestScheduler` must implement this specific "Aborts-for-Rebalancing" logic to ensure new workers actually get traffic after expansion.
  - Termination safety: guard `_rebalance_on_expand` against "zombie inflation" where `src_rank2_dp_rank` contains mappings to inactive ranks (inflates `src_ranks_to_abort` beyond the selectable pool). Filter abortable mappings to `old_active_dp_ranks` before doing proportional math (or clamp planned abort to available) and `logger.warning(...)` if clamping occurs; proceed best-effort (rebalancing accuracy is not correctness-critical for ENG-123).
  - Selection policy (balance): each iteration, abort one `src_rank` from the most-loaded old worker (max remaining `src_ranks`), not `cycle(...)` round-robin. If `max_load == 0` (no `src_ranks` left to steal), stop and warn.
  - Implementation note (why this matters; avoid hangs):
    - Current upstream-style selection uses `cycle(dp_rank_to_src_ranks.keys())` and has no `await` inside the selection loop. If the abort target is inflated (due to zombie entries) and all per-worker lists drain, the loop can spin forever and block the event loop, so `asyncio.wait_for(..., timeout=30s)` may not fire.
    - Recommended best-effort algorithm (deterministic, terminates):
      1. Snapshot `old_active_dp_ranks = active_dp_ranks.copy()` before `active_dp_ranks.update(expand_dp_ranks)`.
      2. Build `dp_rank_to_src_ranks` by filtering `src_rank2_dp_rank.items()` to `dp_rank in old_active_dp_ranks`.
      3. Let `available = sum(len(v) for v in dp_rank_to_src_ranks.values())`.
      4. Compute `planned_to_abort` from `available` (or compute from `len(src_rank2_dp_rank)` then clamp: `planned_to_abort = min(planned_to_abort, available)` and `logger.warning(...)` if clamped).
      5. While `remaining_to_abort > 0`:
         - pick `dp_rank = argmax len(dp_rank_to_src_ranks[dp_rank])` over old active ranks
         - if `max_load == 0`: `logger.warning(...)` and break (no src env left to steal)
         - pop one `src_rank` from that dp rank, decrement `remaining_to_abort`
- **Reviewed**: Yes

**P2: Issue 203 & 209: Offload Verification & Gap Closure**
- **Finding**: Fork's offload sequence may not guarantee immediate GPU memory release. vLLM/SGLang internal caching can delay actual VRAM release. Also need to close feature gaps between fork and upstream.
- **Gap Analysis (Fork vs Upstream)**:
  | Feature | Upstream | Fork | Action |
  |---------|----------|------|--------|
  | Spelling | `state_offload_manger` ❌ | `state_offload_manager` ✅ | Fix typo |
  | `sleep_level` control | ❌ Missing | ✅ Supported | Port from fork |
  | Context object | ❌ `yield` | ✅ `yield context` | Port from fork |
  | Partial load/offload | ❌ Missing | ✅ `load_states_partial` / `offload_states_partial` | Port from fork |
  | Memory validation | ❌ None | ❌ None (add now) | New addition |
- **Fix - Port and Enhance `state_offload_manager`**:
  1. **Fix typo**: Rename `state_offload_manger` → `state_offload_manager`
  2. **Port `sleep_level` control**: Add `OffloadContext` with mutable `sleep_level` attribute
  3. **Port context yield**: Change `yield` → `yield context`
  4. **Add memory validation** (NEW) - runs in `__exit__` block of context manager:
     ```python
     with Timer(name=f"{metric_infix}_offload") as offload_timer:
         # ... existing offload logic ...
         
         # Memory validation after offload completes
         if context.offloaded or (context.sleep_level and context.sleep_level > 0):
             torch.cuda.synchronize()
             free, total = torch.cuda.mem_get_info()
             occupied_pct = (total - free) / total * 100
             if occupied_pct > 10:  # 10% threshold
                 raise RuntimeError(f"Offload validation failed: {occupied_pct:.1f}% occupied")
             metrics[f"{metric_infix}/occupied_pct"] = occupied_pct
     ```
  5. **Port partial APIs**: Add `load_states_partial(active_dp_ranks)` and `offload_states_partial(offload_dp_ranks)` to Worker class
- **PR Strategy**:
  1. PR 1: Fix typo + add `sleep_level` control + context object (port from fork)
  2. PR 2: Add memory validation percentage check (new feature)
  3. PR 3: Add partial load/offload APIs (port from fork)
- **Reviewed**: Yes





**P1: Issue 207: Dual-Path GPU Release Logic**
- **Finding**: Different paths for dynamic vs static clusters.
- **Fix**: Both APIs already planned in extraction:
  - **Static clusters**: Use `release_gpus()` for immediate release
  - **Dynamic clusters**: Use `notify_ready_to_release()` for planned/coordinated release
- **Review Decision**: **[Invalid]**. Both APIs documented in "SchedRL Client/Scheduler API Missing" section.
- **Reviewed**: Yes

**P1: Issue 211: Actor Infer Max Concurrency Override**
- **Finding**: Multi-pipeline requires minimum max_concurrency values to avoid deadlocks. The fork overrides max_concurrency in several components:
  1. **AgenticConfig.actor_infer**: `max(train_env_workers * envs_per_worker + 1, val_env_workers * envs_per_worker + 1)`
  2. **GroupQueueManager**: `max(1000, env_num + 100)` (was `env_num + 1`)
  3. **RequestScheduler**: `max(1000, env_num + 100)` (was `env_num + 1`)
  4. **SglangSlaveActor**: hardcoded `1000`
- **Fix**: Port the same max_concurrency overrides to SchedRL:
  - Actor infer: `max(train_env_workers * envs_per_worker + 1, val_env_workers * envs_per_worker + 1)`
  - GroupQueueManager: `max(1000, env_num + 100)`
  - RequestScheduler: `max(1000, env_num + 100)`
  - SGLang actors: `1000`
- **Reviewed**: Yes

**P1: Issue 212: Port Stop-Before-Offload Pattern with New API**
- **Finding**: v0.2.0 fork removed `start_server/stop_server` API, replaced with `load_states/offload_states`. The correct "Abort -> Wait -> Offload" pattern must be ported using upstream ROLL's implementation (not fork's flawed version).
- **API Changes**:
  - **Old**: `start_server()` / `stop_server()`
  - **New**: `load_states()` / `offload_states()`
- **Correct Pattern** (use upstream ROLL's `suspend()` logic, not fork's `rebalance_on_shrink`):
  1. `abort_requests(dp_ranks)` - send abort signals to workers
  2. `while not empty(): await empty_notifier.wait()` - wait for `running_requests` to be empty (actual completion)
  3. `clear_src_rank_mappings()` - remove sticky routing
  4. THEN `offload_states_partial(dp_ranks)` - safe to move weights to CPU
- **Why Not Fork's Pattern**: Fork's `await asyncio.gather(*abort_futures)` only waits for RPC return, not actual request completion. This creates race condition where requests still run during offload.
- **Fix for ENG-123**:
  1. Remove `start_server/stop_server` from upstream ROLL
  2. Use `load_states/offload_states` with correct abort+wait pattern
  3. Use upstream's `while not empty()` completion check (not fork's gather)
- **Reviewed**: Yes

**P0: Issue 213: API Mismatch - Port stop_server to load_states/offload_states**
- **Finding**: Fork uses legacy `stop_server`/`start_server` APIs, but upstream ROLL replaced them with `load_states`/`offload_states` in v0.2.0. Need to port extraction plan to use new APIs.
- **API Mismatch**:
  | API | Fork Multi-Pipeline | Upstream ROLL (v0.2.0+) | Status |
  |-----|---------------------|-------------------------|--------|
  | `stop_server()` | ✅ Active (lines 209+) | ❌ Removed (commented out) | **Deprecated - Port away** |
  | `start_server()` | ✅ Active (lines 176+) | ❌ Removed (commented out) | **Deprecated - Port away** |
  | `load_states()` | ❌ Not used | ✅ New API | **Use this** |
  | `offload_states()` | ✅ Partial use | ✅ New API | **Use this** |
- **Fix for ENG-123**: Port all `stop_server`/`start_server` usage to new APIs:
  1. **Replace `stop_server()` calls** with `offload_states()`:
     - Server thread stops automatically when model is offloaded
     - Already handled in `offload_states_partial()` implementation
  2. **Replace `start_server()` calls** with `load_states()`:
     - Server thread starts automatically when model is loaded
     - KV cache initialized, server ready for requests
  3. **Update Adapter RPC handlers**:
     - `shrink_workers()` → call `offload_states_partial()` (replaces stop_server)
     - `expand_workers()` → call `load_states_partial()` (replaces start_server)
  4. **Precondition behavior (B1)**:
     - Keep upstream fail-fast assertions in partial lifecycle APIs.
     - `load_states_partial(dp_ranks)`: keep `assert is_loaded is False`.
     - `offload_states_partial(dp_ranks)`: keep `assert is_loaded is True`.
  5. **Server lifecycle change**:
     - Old: Explicit `start_server()` → `stop_server()` calls
     - New: Implicit via `load_states()` → `offload_states()` (server tied to model state)
- **Implementation Note**: This is a breaking API change. Ensure all fork code using `stop_server`/`start_server` is updated during porting.
- **Reviewed**: Yes

**P0: Issue 214: HuggingFace Cache Isolation (HF_HOME / HF_HUB_CACHE)**
- **Finding**: When multiple pipelines start simultaneously, they race on HuggingFace's automap cache (`~/.cache/huggingface/modules/`), causing `JSONDecodeError` crashes.
- **Root Cause**: HuggingFace generates Python metadata files (KB) on-the-fly in `HF_HOME/modules/`. Concurrent writes corrupt these files.
- **Standard Solution** (Day 1 Fix):
  Use official HuggingFace environment variables for isolation:
  | Variable | Purpose | Setting |
  |----------|---------|---------|
  | `HF_HOME` | Metadata, tokens, temp files | **Private per pipeline** |
  | `HF_HUB_CACHE` | Model weights (GB) | **Shared across pipelines** |
  ```python
  # In Worker initialization (before model load):
  pipeline_id = os.environ.get("PIPELINE_ID", "default")
  
  # Isolate metadata (small, KB) - prevents race conditions
  os.environ["HF_HOME"] = f"/tmp/schedrl/pipelines/{pipeline_id}/hf_home"
  
  # Share weights (large, GB) - avoid duplication
  os.environ["HF_HUB_CACHE"] = "/shared/huggingface/hub"
  ```
- **Why This Works**:
  - `HF_HOME/modules/` (automap cache) is isolated per pipeline → no race
  - `HF_HUB_CACHE` (model weights) is shared → saves disk space
  - Uses standard, documented HuggingFace variables
- **Implementation**: Set in `Worker.__init__()` before any HuggingFace imports
- **Reviewed**: Yes





## Desired End State

After ENG-123:

1) `schedrl/` is a real Python package at repo root implementing:
- protocol/types/actions/validation
- client connect/get-or-create + registration + progress report helpers
- scheduler actor (Library Mode) with planning + ordered execution (serialized actions; no per-action id)

2) `third_party/ROLL` can run (single framework) under Library Mode: creates/uses a job-scoped SchedRL scheduler actor

3) A ROLL Adapter actor exists in `third_party/ROLL` that implements the minimal Adapter RPC surface and internally uses ROLL primitives.

4) `third_party/ROLL_multi_pipeline` is **frozen** (no further feature development). It remains as reference and as a migration source, but the live path becomes `schedrl` + `third_party/ROLL`.

## What We’re NOT Doing (ENG-123 scope)

- Not implementing multi-framework arbitration yet (only ROLL is wired end-to-end).
- Not migrating NeMo-RL or Miles.
- Not deleting `third_party/ROLL_multi_pipeline`.
- Not adding new third-party Python dependencies.
- Not building new tests outside existing suites unless explicitly approved later.
- Not implementing scheduler policy refactors (e.g. heapq-based queues), priority taxonomy changes, or min-dp constraints beyond what the fork already does; we keep fork behavior as-is for ENG-123.
- Not centralizing timeout configuration into SchedRL typed config in ENG-123 (keep env-var timeouts); centralization + propagation into pipelines/adapters is a post-ENG-123 follow-up.
- Not implementing post-release GPU memory measurement in ENG-123 (return `-1` sentinel fields in `release_reports`); add black-box measurement after ENG-123.
- Not implementing MULTI_LORA specifics in ENG-123; only reserve protocol/types with a TODO to cross-validate against `third_party/ROLL_multi_lora` and implement adapter hooks after ENG-123.
- Not adding `oldest_unfinished_creation_ts` in progress reporting in ENG-123; requires persistent `creation_ts` tracking and will be added after ENG-123.
- Not implementing lifecycle idempotency tokens (`activation_epoch` / `action_id`) or retry-tolerant semantics in ENG-123; add intent/version tokens after ENG-123 if retries/HA/multi-caller are needed.

## Implementation Approach

### High-level strategy

We will **port concepts** from `third_party/ROLL_multi_pipeline` into `schedrl/` as the general scheduler core, while keeping framework-specific mechanics behind a minimal Adapter.

Key design choice: keep the Adapter surface identical regardless of lifetime model. (Service Mode is deferred.)

## Workflow + Contracts (SPEC; ENG-123)

### Workflow (ENG-123, Library Mode; admission-gated spawn; source-of-truth GPUs in ROLL)

Reference workflow: `third_party/ROLL_multi_pipeline/examples/multi_pipeline/start_multi_pipeline_test.py`.

1. **ROLL runner loads YAML** and determines **concrete GPU IDs** / device mapping for each cluster (e.g. `actor_train`, `actor_infer`, `critic`) including DP/TP structure (TP>1 supported/required).
2. **ROLL runner connects to Ray** (job-scoped Ray cluster for ENG-123).
3. **ROLL → SchedRL Orchestrator**: `register_pipeline(pipeline_id, registration_payload)` where `registration_payload` includes:
   - concrete GPU IDs / device mapping (ROLL is the source-of-truth),
   - any topology metadata needed by the adapter (dp ranks, tp groups, rank2devices, etc.),
   - required config validations (e.g. `sleep_level=2`, `partial_gpu_mode=False`).
4. **ROLL → SchedRL Orchestrator**: `admit_pipeline(pipeline_id) -> AdmitResponse` (**blocking**).
   This is the admission gate: ROLL must not start GPU-heavy initialization until this returns. `AdmitResponse` includes a scheduler handle/locator for subsequent coordinator→scheduler RPCs.
5. **After admit ACK**, ROLL spawns and initializes the pipeline’s GPU-heavy Ray actors/workers and enters the training loop.
6. During runtime:
   - **Pipeline/adapter → SchedRL**: `report_progress(pipeline_id, ...)` (batch start + 2% band crossings + completion).
   - **SchedRL Scheduler → Adapter (`schedrl:adapter:{pipeline_id}`)**: lifecycle RPCs (`close/open_admission`, `shrink/expand`, `abort`/drain, offload/stop).
   - **Coordinator (pipeline) → SchedRL Scheduler**: coordinator primitives (e.g. `notify_ready_to_release`, `release_and_request`) as required by the SchedRL protocol.

### Contracts / Invariants (must hold)

Single source of truth for contracts (avoid duplication):
- Resource model + admission gate + GPU source-of-truth: see **## Concrete contracts (must match exactly)**.
- Shrink-to-zero and shrink ordering: see **Verified issues to address** → **Critical 1**.
- Abort ACK semantics + timeout policy: see **Verified issues to address** → **Critical 2**.
- Routing metadata/lifecycle locking + ordering: see **Two-lock usage (source of truth)** and **Routing lock atomicity and suspend-before-clear ordering (required)**.

### Adapter RPC Surface (minimum)

Scheduler → Adapter:
- `close_admission(dp_ranks) -> ActionResponse`
- `open_admission(dp_ranks) -> ActionResponse`
- `shrink_workers(dp_ranks) -> ActionResponse`
- `expand_workers(dp_ranks) -> ActionResponse`
  - Adapter decides `tgt_dp_ranks`; sender strategy cache state holds active rollout checkpoint version
  - Always performs selective sync-on-resume (hardcoded behavior)

Coordinator/Pipeline loop → ModelUpdateService:
- `promote_active_checkpoint(checkpoint_version: int, global_step: int) -> None`
  - Explicit coordinator signal for activation target. `ModelUpdateService` forwards to sender strategy, which stores version metadata and cache state. shrink/expand APIs do not carry version args

#### Assumptions (validate now) + Backlog (enforce later): prerequisites for Adapter RPCs without `activation_epoch` / `base_version`

This plan assumes a stronger decoupling where the scheduler only provisions GPUs/topology and the pipeline coordinator fully owns checkpoint selection (i.e., no versions in the scheduler path).

Because we remove `activation_epoch` / `base_version` now, implementors MUST validate these system-level requirements hold end-to-end. Enforcement mechanisms are backlog items and will be implemented later.

- **Single lifecycle caller**: only the central scheduler is allowed to issue Adapter lifecycle actions (`expand_workers`, `shrink_workers`, `open_admission`, `close_admission`).
  - Control flow note: upstream ROLL currently uses fixed Ray actor names for scheduler helpers (no `pipeline_id`), which can collide across multiple pipelines. ENG-123 patches upstream `RolloutScheduler(pipeline_id=...)` to prefix both `RequestScheduler` and `GroupQueueManager` `.options(name=...)` names so each pipeline instance is uniquely discoverable (e.g., `{pipeline_id}_RequestScheduler-...-{mode}`).
  - Enforcement options (pick one):
    - Do not register lifecycle actors under globally discoverable names; pass handles only to the scheduler.
    - Or require a capability token / scheduler secret on lifecycle RPCs (fail closed).

- **No retry / no replay contract for lifecycle intents**: lifecycle RPCs must be strictly fail-fast (no external retry loops) and must not be replayed after actor restart/failover.
  - If we later add retries, HA, or multiple callers, we must reintroduce an intent freshness token (epoch/sequence).

### Protocol Definition (Refined for SchedRL)
1. **Implicit sequencing**: requests define dependencies, but the scheduler executes one atomic action per pipeline at a time.
2. **Strict sequencing**: the scheduler must not issue a new lifecycle intent batch until the previous batch has ACKed, and the coordinator/adapter must execute lifecycle actions sequentially (local locks are fine, but they do not replace freshness tokens).

Orchestrator (Launcher) → Scheduler:
- `register(pipeline_id, total_gpus, ...)`
  - **Resource Registration**: Registers the pipeline's static resource topology (TP sizes, device mappings) with the scheduler before the pipeline actor is started.

Coordinator (Pipeline Actor) → Scheduler:
- `request_gpus(cluster_id, priority, ..., global_step: Optional[int] = None)`
  - **Blocking Allocation**: Blocks until the requested GPUs are allocated by the scheduler.
- `release_gpus(cluster_id, ..., global_step: Optional[int] = None)`
  - **Immediate Release**: Frees GPUs immediately. Used by stateful clusters (Training, Critic) that do not require "shrink" logic.
- `release_and_request(release_cluster_id, request_cluster_id, ..., global_step: Optional[int] = None)`
  - **Atomic Release+Request**: Atomically submits a release intent for one registered cluster and a follow-on activation request for another cluster as a single scheduler transaction. Under ENG-123 static GPU mapping, this is a logical activation-transition *request* over pre-registered GPU IDs (not physical GPU re-assignment); the actual activation grant occurs when the scheduler executes the plan.
- `notify_ready_to_release(..., global_step: Optional[int] = None)`
  - **Planned Release (Blocking)**: Signals that a generation batch is complete. Unlike `release_gpus` (immediate), this blocks until the scheduler's Phase 0-6 planning loop safely reclaims the resources. Essential for "Gap-Ratio" fairness and partial-GPU synchronization.
- `report_progress(..., fifo_timestamp: Optional[float] = None)`
  - **Fairness Hook**: Passes the original creation timestamp of the oldest waiting episode. Used by the scheduler to break ties in the GENERATION (P6) queue, enforcing global FIFO ordering across pipelines.
- `unregister_pipeline(...)`

### Critical protocol clarifications for ENG-123 (ROLL-only, Library Mode)

Single source of truth (avoid duplication):
- Shrink-to-zero support + ordering: see **Verified issues to address** → **Critical 1**.
- Abort ACK definition (ROLL): see **Verified issues to address** → **Critical 2**.
- request_id format: see **Concrete contracts (must match exactly)** → **Canonical `request_id`**.

## Implementation Execution Checklist (ENG-123; required)

This section is the **implementation breakdown** for human review. Each checklist item is intended to be a small, localized patch. Each phase ends with a validation milestone.

### Phase 1 checklist — SchedRL core package skeleton

**Primary goals**: create job-scoped Orchestrator/Scheduler singletons; define canonical protocol + fail-fast semantics; define correct Ray retry knobs; establish namespace strategy hooks (Option A) at the contract layer.

- [ ] **P0 Issue 21 & 215 (ownership)**: Orchestrator owns core components; enforce singleton actor creation.
- [ ] **P1 Issue 25 & 31 (fail-fast)**: implement global fail-fast shutdown path (already by-design; ensure it is enforced, not bypassed).
- [ ] **P1 Issue 29 & 68 (timeouts)**: define phase-specific timeout constants (register/admit/shrink/expand/abort-ACK/offload).
- [ ] **P0 Issue 51 (delimiter collision)**: implement and enforce `validate_pipeline_id` (reject `:`) and canonical `request_id` helpers.
- [ ] **P1 Issue 62, 204 & 206 (config resolution/validation)**: define validation entrypoints (even if full Hydra resolution is implemented later, the error boundaries must be clear in Phase 1).
- [ ] **P1 (namespace override plumbing)**: define `ROLL_RAY_NAMESPACE` contract + `PIPELINE_ID` propagation via Ray `runtime_env.env_vars` (implementation occurs in Phase 3, but the contract must be explicit here).
- [ ] **P1 (Ray retry semantics; correctness)**: remove any mention/usage of non-Ray knobs like `max_retries`; use Ray-supported knobs for “no retry/no replay” (`max_restarts=0`, tasks `max_task_retries=0`, avoid implicit retries).

Validation milestone (Phase 1)
- [ ] Start a fresh Ray job; create `schedrl:orchestrator` + `schedrl:scheduler`; verify they start once and are discoverable by name.
- [ ] Negative test: invalid `pipeline_id` (contains `:`) fails fast at registration/admission.

### Phase 2 checklist — Port scheduler model into `schedrl` (policy + state)

**Primary goals**: port the fork scheduling model (planning loop, fairness, progress ingestion) into SchedRL; keep ROLL as mechanics.

- [ ] **H1 (dead assertions)**: replace fork “no-op tuple assertions” with real `assert` statements or explicit `ValueError`/`RuntimeError` (fail-fast). Do not port silent tuple expressions like `(cond, "msg")`.
- [ ] **H2 (cluster_id parsing)**: implement a suffix-aware parser for cluster IDs; never parse `pipeline_id` from `cluster_id` using `rsplit("_", 1)` (use the fork-equivalent `_parse_cluster_id` approach).
- [ ] **H3 (notify_completion atomicity)**: `notify_completion` must be atomic (idempotency check + insert into `pending_completion_requests` under the same scheduler lock).
- [ ] **P0 Issue 107 (state inconsistency)**: any execution-phase failure triggers controlled-fatal shutdown (`shutdown(force=True)`).
- [ ] **P0 Issue 81 & 124 (expansion signaling)**: ensure expansion completion signals cannot be orphaned; on any failure, fail-fast and shut down.
- [ ] **P1 Issue 64 (gap-ratio starvation)**: implement gap-ratio fairness constraints as described (work inflation prevention).
- [ ] **P1 Issue 202 & 216 (proportional rebalancing/session migration)**: implement required rebalancing logic and define migration semantics.
- [ ] **P0 Issue 69/80/93/158 (topology validation)**: implement or explicitly defer with a single decision (do not leave ambiguous “P0 but backlog” behavior).
- [ ] **P0 Issue 87 (ResourceManager init race)**: implement node discovery gating with a bounded wait budget and fail-fast error message.

Validation milestone (Phase 2)
- [ ] Single-pipeline run: scheduler progresses through phases; progress ingestion updates are observed by scheduler.
- [ ] Induced failure during execution phase causes immediate job shutdown (no partial continuation).

### Phase 3 checklist — Implement ROLL Adapter + upstream ROLL shim

**Primary goals**: create `schedrl:adapter:{pipeline_id}` actor; isolate all ROLL actors per pipeline namespace (Option A); keep SharedStorage job-global; implement safe shrink/expand/offload; port required upstream patches.

Namespace + SharedStorage (systematic isolation)
- [ ] **Namespace (Option A; required)**: set `ROLL_RAY_NAMESPACE=f"pipeline_{pipeline_id}_NS"` via Ray `runtime_env.env_vars` for all ROLL processes in this pipeline.
- [ ] **Global storage namespace (required)**: add `GLOBAL_STORAGE_NAMESPACE="global_storage_namespace"` and update all SharedStorage get/create sites to use it (never `RAY_NAMESPACE`).

Lifecycle correctness + upstream patches
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
- [ ] **P1 Issue 211 (max concurrency override)**: ensure actor-infer concurrency settings are correct and not shared across pipelines accidentally.
- [ ] **P0 Issue 214 (HF cache isolation)**: enforce HF env isolation via `runtime_env` before any HF imports (do not rely on Worker `__init__`).
- [ ] **P1 (CheckpointManager cache keys)**: include `{pipeline_id}` in model download cache keys (or explicitly document + enforce cross-pipeline-safe invariants for shared cache).
- [ ] **P1 (BaseConfig global env)**: avoid mutating process-global `os.environ` for pipeline-specific settings in shared driver processes; route settings through `runtime_env.env_vars`.
- [ ] **P1 (vLLM/FlashInfer cache paths)**: ensure cache/workspace roots are pipeline-scoped (do not rely on `WORKER_NAME`, which can be overwritten).
- [ ] **P1 (state_offload_manger concurrency)**: replace `del os.environ[...]` with `os.environ.pop(..., None)` (or remove global env dependency) to avoid concurrency errors.
- [ ] **P1 (hidden retry loops)**: remove or explicitly configure retry loops (OpenAI proxy / sandbox reward / driver utils) to preserve ENG-123 fail-fast semantics.
- [ ] **P1 (mode_switch_lock)**: remove unused `mode_switch_lock` or enforce its usage; do not rely on dead serialization mechanisms.
- [ ] **P1 (rebalance termination)**: fix `_rebalance_on_expand` to have an explicit termination condition (no infinite `cycle()` behavior).

Validation milestone (Phase 3)
- [ ] Two pipelines in same job: no Ray named-actor collisions (verify both pipelines fully initialize).
- [ ] Shrink-to-zero + expand: no `No active DP ranks` error; no re-open of suspend by RolloutScheduler; GPU memory is actually freed.
- [ ] Pipeline teardown + re-admission in same job: ports/metadata reusable (no leaks).

### Phase 4 checklist — Selective model update behind Adapter

**Primary goals**: selective sync-on-resume; minimal upstream hooks; correctness under expand after weight update.

- [ ] **P1 Issue 66 (selective expansion safety)**: ensure expansion is safe and does not admit stale weights; fail-fast on any mismatch.
- [ ] **P1 Issue 207 (dual-path GPU release logic)**: implement/verify the correct release path(s) under the adapter contract.
- [ ] Implement `ModelUpdateService` port + required upstream hooks (TP>1 bucket caching policy as documented).

Validation milestone (Phase 4)
- [ ] Expand after a weight update: newly expanded dp ranks serve the correct weights (no stale admission).

### Non-action decisions (must remain true)

These are listed in the plan as P0/P1 but are **explicitly closed/invalid by design**; the implementation must not accidentally reintroduce them:
- [ ] **P0 Issue 132**: deadlock analysis is “invalid by design” under enforced offload config + fail-fast.
- [ ] **P0 Issue 236 & 217**: marked closed/invalid in the plan; if ENG-123 requires shrink-to-zero, treat the shrink-to-zero tasks above as the authoritative implementation path (do not rely on the closed label).
- [ ] **P1 Issue 143 & 147**: closed/invalid; do not add complex recovery/graceful swap behavior in ENG-123.
- [ ] **P1 Issue 65, 218 & 104**: invalid by design; do not add an extra notification protocol surface in ENG-123.
- [ ] **P1 Issue 151**: closed/invalid; do not build scheduler logic that depends on racy progress reporting to preserve correctness.

## Phase 1: Create `schedrl` core package skeleton (Library Mode only)

### Overview
Implement the minimal protocol, client, and scheduler scaffolding so ROLL can connect and register, even before full scheduling policy parity.

### Changes Required

#### 0) Ray retry semantics (required for ENG-123 no-retry contract)

**Correction (Ray default semantics)**:
- Ray actor tasks execute with **at-most-once** semantics by default (`max_task_retries=0` for actor tasks).
- However, Ray can still raise `RayActorError` for an actor task even if the task executed successfully (e.g. actor dies immediately after executing). ENG-123 treats this as a hard failure and will **not** attempt to “retry” lifecycle intents.

To preserve the ENG-123 lifecycle contract (“no retries / no replays”), enforce the following:
- For SchedRL lifecycle actors: set `max_restarts=0` at actor definition/creation (do not enable actor restarts for lifecycle actors).
- For lifecycle RPCs (actor tasks): keep `max_task_retries=0` (default) and do not enable `retry_exceptions` or any other retry mechanism.
- For any non-actor Ray tasks used in lifecycle control paths: explicitly disable retries using Ray-supported knobs (do not use non-Ray knobs like `max_retries`).
- Policy: on `RayActorError` / connectivity ambiguity, **fail fast** and shut down the job; do not attempt recovery or “best-effort” retries in ENG-123.

Apply this to all SchedRL lifecycle actors:
- `schedrl:orchestrator`
- `schedrl:scheduler`

And for framework lifecycle actors (Phase 3):
- `schedrl:adapter:{pipeline_id}`

Backlog (post-ENG-123): if we ever need retries/HA, introduce intent/version tokens (`action_id`/`activation_epoch`) before enabling retries.

#### 1) `schedrl/protocol/*`
**Files** (new):
- `schedrl/protocol/types.py`
- `schedrl/protocol/actions.py`
- `schedrl/protocol/validation.py`

**Contents**
- Dataclasses/enums for:
  - IDs: `PipelineId`, `ClusterId`, `AdapterId`
  - `ModelMode` (`FULL_FT`, `MULTI_LORA`) (TODO: cross-validate against `third_party/ROLL_multi_lora` before implementing MULTI_LORA specifics)
  - TODO (MULTI_LORA): add `adapter_id` mapping + per-adapter progress reporting hooks for fairness decisions, and validation rules for `base_version` by mode
  - `ActionResponse {success: bool, error: Optional[str]}`
- Canonical validation rules (fail-fast):
  - Abort ACK definition per-adapter (for ROLL: “no longer in-flight”; capture finish_reason if present)

**Additional required module**
- `schedrl/protocol/request_id.py`:
  - builder/parser/validator for canonical request id (all modes): `{pipeline_id}:{traj_id}:{turn_id}:{attempt}`
  - Required API:
    - `build_request_id(pipeline_id: str, traj_id: str, turn_id: int, attempt: int) -> str`
    - `parse_request_id(request_id: str) -> tuple[pipeline_id, traj_id, turn_id, attempt]`
    - `validate_request_id(request_id: str) -> None` (raise `ValueError` with message)

**Config keys (Phase 1 defaults)**



Disable central timeout config in ENG-123 (timeouts are env-var driven):
- `schedrl.abort_timeout_secs = -1`  # invalid/sentinel => disabled; use env vars instead

Backlog (when centralizing timeouts into schedrl config):
- replace `-1` with a real default and propagate into pipelines/adapters

Note: these are fields on SchedRL typed config objects (dataclasses) passed to the scheduler/client; they are not defined via a SchedRL YAML schema.

#### 2) `schedrl/client/*`
**Files** (new):
- `schedrl/protocol/adapter.py` (Adapter ABC; methods raise `NotImplementedError`)
- `schedrl/client/client.py` (connect/get-or-create; register/report helpers)

**Key behavior**
- Library Mode discovery (ENG-123):
  - Namespace: `schedrl`
  - Orchestrator actor name: `schedrl:orchestrator`
  - Scheduler actor name: `schedrl:scheduler`

Library Mode placement requirement (ENG-123): pin **ALL** SchedRL system actors to the Ray head node.
- **Strategy**: Use `NodeAffinitySchedulingStrategy` to programmatically identify and pin to the head node.
- **Rationale**: Custom resources (e.g., `{"head": 1}`) require modifying `ray start` commands, which is incompatible with externally-managed clusters (NeMo-RL, SkyRL, Miles). `NodeAffinitySchedulingStrategy` works with any Ray cluster.
- **Implementation**:
  ```python
  from ray.util.state import list_nodes
  from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
  
  # Identify head node programmatically (once during init)
  nodes = list_nodes(filters=[("is_head_node", "==", "True")])
  if not nodes:
      raise RuntimeError("Could not identify head node.")
  head_node_id = nodes[0]["node_id"]
  
  # Create actors with strict head node affinity
  orchestrator = Orchestrator.options(
      name="schedrl:orchestrator",
      namespace="schedrl",
      scheduling_strategy=NodeAffinitySchedulingStrategy(
          node_id=head_node_id,
          soft=False  # Fail if head node unreachable
      )
  ).remote()
  
  scheduler = Scheduler.options(
      name="schedrl:scheduler",
      namespace="schedrl",
      scheduling_strategy=NodeAffinitySchedulingStrategy(
          node_id=head_node_id,
          soft=False
      )
  ).remote()
  ```
- **Apply to ALL SchedRL system actors**:
  - `schedrl:orchestrator` ✓
  - `schedrl:scheduler` ✓
  - `schedrl:resource_manager` (if implemented as a Ray actor) ✓
  - Any other SchedRL infrastructure actors ✓

Client connection flow (ENG-123, Library Mode only):
- `connect(create_if_missing=True)` returns the **orchestrator** handle.
- `admit_pipeline(...)` is **blocking** and returns an `AdmitResponse` that includes a **scheduler handle (or actor locator)**. After admission, the coordinator may call the scheduler directly for high-frequency RPCs (e.g., progress reporting) while orchestration/admission remains orchestrator-owned.
- The orchestrator is responsible for validating that scheduler calls are only used after admission (fail-fast on misuse).

`connect(create_if_missing=True)` implements get-then-create and handles create races.

Backlog (post-ENG-123): Service Mode connect-only semantics and detached actor lifecycle.
- Standardized names (ENG-123): `schedrl:orchestrator`, `schedrl:scheduler`, `schedrl:adapter:{pipeline_id}`.
- Additional standardized names (ENG-123): `schedrl:resource_manager` (if implemented as a Ray actor rather than an internal module).

Orchestrator RPC surface (ENG-123; required signatures)

The orchestrator owns registration/admission and returns the scheduler handle post-admission.

- `register_pipeline(...) -> RegisterResponse`
- `admit_pipeline(...) -> AdmitResponse`  # includes scheduler handle/locator
- `get_pipeline_state(pipeline_id: str) -> PipelineState`
- `monitor_pipelines() -> dict[str, PipelineState]` (snapshot state; caller polls)
- `cleanup_pipeline(pipeline_id: str) -> None`
- `kill_pipeline(pipeline_id: str) -> None`
- `shutdown(force: bool = True, reason: str | None = None, source: str | None = None) -> None`

`shutdown(...)` semantics (ENG-123, Library Mode):
- This is the global fail-fast button that **terminates the entire Ray cluster** (including all SchedRL actors AND framework actors from NeMo-RL, SkyRL, Miles, etc.).
- On `force=True`, execute `ray stop --force` on **ALL nodes** (workers first, head node last) to immediately kill the cluster.
  - **Implementation**: `Orchestrator.shutdown()` method dispatches remote tasks to each node to stop Ray locally, then stops head node last:
    ```python
    import sys
    import subprocess
    from pathlib import Path
    import ray
    
    # Module-level helper (defined outside Orchestrator class)
    @ray.remote
    def _kill_local_ray():
        """Stop Ray on the node where this task executes."""
        python_bin_dir = Path(sys.executable).parent
        ray_executable = python_bin_dir / "ray"
        subprocess.run([str(ray_executable), "stop", "--force"])
    
    class Orchestrator:
        # ... other methods ...
        
        def shutdown(self, force: bool = True, reason: str | None = None, source: str | None = None):
            """Stop Ray on all nodes (workers first, head last)."""
            if not force:
                # Graceful shutdown: cleanup actors but don't kill cluster
                return
            
            # Force shutdown: kill entire cluster
            nodes = ray.nodes()
            head_node_id = None
            worker_tasks = []
            
            # Identify head node and dispatch tasks to workers
            for node in nodes:
                node_id = node['NodeID']
                is_head = node.get('Alive', False) and 'head' in node.get('Resources', {})
                
                if is_head:
                    head_node_id = node_id
                else:
                    # Dispatch to worker node
                    node_ip = node['NodeManagerAddress']
                    task = _kill_local_ray.options(
                        resources={f"node:{node_ip}": 0.01}
                    ).remote()
                    worker_tasks.append(task)
            
            # Wait for workers to stop (with timeout to avoid hanging)
            if worker_tasks:
                try:
                    ray.wait(worker_tasks, timeout=10, num_returns=len(worker_tasks))
                except Exception:
                    pass  # Best effort - workers may already be dead
            
            # Finally, stop head node (this will kill the orchestrator itself)
            python_bin_dir = Path(sys.executable).parent
            ray_executable = python_bin_dir / "ray"
            subprocess.run([str(ray_executable), "stop", "--force"])
    ```
  - This terminates **everything**: SchedRL actors, framework training/inference actors, and the Ray runtime itself.
  - Must be called from orchestrator actor (which is pinned to head node via `NodeAffinitySchedulingStrategy`).
  - **Zombie Process Prevention**: Set these environment variables when starting Ray workers:
    - `RAY_kill_child_processes_on_worker_exit=1` - Auto-kill child processes on worker exit
    - `RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper=1` - (Linux 3.4+) Aggressively cleanup grandchild processes
- It should be safe to call multiple times; the first caller wins and later calls return immediately (idempotent).

#### 3) `schedrl/scheduler/*` skeleton
**Files** (new):
- `schedrl/scheduler/scheduler.py` (Ray actor)
- `schedrl/scheduler/state.py`
- `schedrl/scheduler/executor.py`
- `schedrl/scheduler/run.py`

Additional scheduler component (ENG-123):
- `schedrl/scheduler/resource_manager.py` (ResourceManager: placement group allocation + topology queries)

Platform independence note:
- Do not import framework-specific platform helpers (e.g. `roll.platforms.current_platform`) in `schedrl/`. If platform details are required, inject a small `PlatformConfig` dataclass (e.g., `ray_device_key`, `device_control_env_var`) into SchedRL actors.

**Implementation Reminders (Verified Issues)**:
- **Issue 80 (Validation)**: `register_pipeline` must validate that provided GPU IDs are a subset of `total_gpus`.
- **Issue 87 (ResourceManager)**: Use `Ray.wait` or robust retries for node discovery instead of a simple sleep loop.
- **Issue 51 (Delimiter)**: `validate_request_id` must reject `:` in `pipeline_id` or `traj_id`.

**Minimum behavior**
- In-memory state per pipeline:
  - registered adapter handle
  - desired vs actual allocations for generation cluster
  - latest progress sample
- An execution path that can issue Adapter RPCs in strict order and enforce timeouts.

### Success Criteria

#### Manual Verification
- [ ] In a dev Ray session, a ROLL driver can connect in library mode (create-if-missing) and register successfully.

## Phase 2: Port the scheduling model from `ROLL_multi_pipeline` into `schedrl` (single-framework first)

### Overview
Implement the central scheduler loop and core data model in `schedrl` by porting the proven concepts:
- pending requests + active allocations + idle GPUs
- plan validation
- ordered execution (shrink → allocations → expand)

Start with **single-framework (ROLL-only)** semantics, but keep types general.

FIFO policy (plan-only clarification):
- Where the plan says “FIFO”, the ordering source is scheduler-side arrival time (the timestamp/sequence when the request reaches the central scheduler actor), not wall-clock timestamps persisted in pipeline state.

### Changes Required

#### 1) State model and allocation units
**Source reference**: `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/gpu_scheduler_types.py`

**SchedRL state** should represent:
- Physical GPU pool (ids)
- Cluster allocations keyed by `cluster_id` and `pipeline_id`
- Generation allocations tracked as DP-worker bundles (each DP rank consumes `tp_size` GPUs)

#### 2) Scheduler loop + planner
**Source reference**: `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/centralized_gpu_scheduler.py`

**Implement in `schedrl/scheduler/`**
- Event-driven loop (wake on new request/release/progress)
- Phase ordering (conceptual):
  - process completion notifications
  - plan non-generation
  - plan generation (initially simple; can start FIFO)
  - validate execution plan
    - **TASK**: Port logic from `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/gpu_scheduler_validation.py` to `schedrl/scheduler/validation.py`.
    - Must include all 11 critical conditions: operation uniqueness, DP rank boundaries, device mapping validation, and double-free detection.
  - execute shrinks/allocations/expansions
  - commit state

**Implementation Reminders (Verified Issues)**:
- **Issue 81 (Orphaned Signaling)**: `_execute_expansions` must ensure all successful expansions are signaled even if some fail (use `return_exceptions=True` in `asyncio.gather`).
- **H1 (Dead assertions)**: when porting expand-from-zero allocation construction, do not copy fork's no-op tuple “assertions” (`len(...) > 0, "msg"`). Use real `assert` / raise in SchedRL scheduler.
- **H2 (pipeline_id parsing)**: never parse `pipeline_id` from `cluster_id` using `rsplit("_", 1)`. Use a suffix-aware cluster-id parser (equivalent to fork `CentralizedGPUScheduler._parse_cluster_id`).
- **H3 (notify_completion race)**: `notify_completion` must perform the idempotency check and insertion into `pending_completion_requests` under the scheduler lock (one critical section).

#### 3) Progress ingestion
**Source reference**: `third_party/ROLL_multi_pipeline/roll/distributed/scheduler/rollout_scheduler.py` progress buckets

Implement `report_progress(...)` ingestion and store latest per pipeline.

Explicit requirements:
- Denominator is explicit per-pipeline registration/config; do not infer it.
- Do **not** copy progress percentage math from `third_party/ROLL_multi_pipeline` as-is; treat it as potentially outdated.
- Use the shared protocol’s newer definition from `design_doc/multi-pipeline-adaptation-plan.md`:
  - report `percent_completed = collected_trajectories / step_target_trajectories` (trajectory units)
  - 2% banding is applied on `percent_completed` (do **not** clamp; allow `> 1.0`), and always emit a final update when `percent_completed >= 1.0`.

### Success Criteria

#### Manual Verification
- [ ] With a single ROLL pipeline, SchedRL can allocate generation DP workers and perform at least one shrink/expand cycle without crashing (manual harness is fine).

## Phase 3: Implement the ROLL Adapter + compatibility shim in `third_party/ROLL`

### Overview
Add a small shim in upstream ROLL to integrate with `schedrl`.

### Changes Required

#### 1) Utility Porting
**Target location**: `third_party/ROLL/roll/schedrl_adapter/`
- **TASK (minimal; reuse upstream)**: Do not port fork vLLM `worker_helper.py` or fork TP-shard-aware receiver assembly logic for ENG-123.
- **TASK (required)**: Reuse upstream `roll/utils/send_recv_utils.py` (`serialize_named_weights(...)`, `named_tensors_from_bucket(...)`) and upstream vLLM worker RPC contract for model-update payloads.

Reference SHAs (verified):
- Fork spec: `ROLL_multi_pipeline@262dd2c0527695a26f389a6c44a3a85701f48cc6` (`(feat): multi-pipeline.`)
- Fork base: `ROLL_multi_pipeline@3077befc` (`(feat): publish roll v0.2.0.`)
- Upstream baseline: `ROLL@777dad6180a32e278802f4775eeb9d821511f648`

#### 2) ROLL Adapter actor
**Target location**: `third_party/ROLL/roll/schedrl_adapter/adapter.py`
- **TASK**: Implement `third_party/ROLL/roll/schedrl_adapter/concurrent_pipeline.py`. Port the logic from the fork's `ConcurrentAgenticPipeline` to wire up upstream worker types and the upstream `RequestScheduler` (patched in-place for ENG-123).

**Scope (thin wiring; do not re-port fork orchestration)**:
- `concurrent_pipeline.py` is wiring/glue only. Keep multi-pipeline orchestration and planning in `schedrl/` + the ROLL runner.
- Must wire (framework mechanics):
  - creation/ownership of the per-pipeline upstream `RequestScheduler` handle (owns `routing_lock`-protected routing metadata, `swapping_lock` serialization for shrink/expand, suspend gate, and targeted abort/drain helpers)
  - adapter RPC backing hooks used by `adapter.py` (`close/open_admission`, `shrink/expand` helpers, stop/offload/onload primitives)
  - progress reporting hooks (emit `report_progress(...)` at batch start + 2% bands + completion)
  - request_id plumbing integration (canonical request_id + attempt increment)
- Must not own (policy/orchestration):
  - no global scheduling policy, no fairness logic, no multi-pipeline monitoring loops, no Ray cluster lifecycle management

**Implements** the Adapter RPC surface by mapping to ROLL primitives:
- **Admission gating** via a per-pipeline `active_dp_ranks` set and `need_suspend` flag (see "Admission Error Handling" section above).
- `shrink_workers(P)` performs strict ordering:
  0) acquire `swapping_lock` (serialize against concurrent expand/shrink)
  1) `close_admission(P)`
  1.a) *Implementation ordering requirement*: when `shrink_workers(P)` infers shrink-to-zero (`active_dp_ranks - P == ∅`), the adapter MUST set the scheduler suspend gate (call `suspend()` / set `need_suspend=True`) before clearing `active_dp_ranks` or calling `_clear_src_rank_mappings`. This ensures `generate_one_request()` (which awaits `_check_suspend()` before selecting a dp rank) blocks rather than calling `_get_least_active_dp_rank()` and raising `RuntimeError`.
  1.b) *Routing-lock atomicity*: Acquire `routing_lock` only for a short mapping update window (mark ranks inactive and clear relevant `src_rank2_dp_rank` mappings), then release it. **Do not** hold `routing_lock` while issuing abort RPCs or awaiting their completion — wait for drains using `empty_notifier`/`empty()` after releasing the lock. The drain/wait should specifically target `shrink_dp_ranks` (sum of running_requests for those ranks) so other active ranks may continue processing.

  2) targeted abort of in-flight requests on P (no `routing_lock`)
  3) wait for abort ACK with timeout; fail-fast on timeout (no `routing_lock`)
  4) stop/offload P (full GPU memory release; still under `swapping_lock`)
  5) return release ACK payload (`release_reports`); ENG-123 uses `-1` sentinel values for post-release GPU memory fields; TODO add real measurement after ENG-123

- `expand_workers(A)`:
  0) acquire `swapping_lock` (serialize against concurrent shrink/expand)
  1) onload/start servers for A (under `swapping_lock`)
  2) Adapter computes `tgt_dp_ranks=A` and calls `ModelUpdateService.sync_selected_workers(tgt_dp_ranks)` (service reads active rollout checkpoint version from sender strategy cache state)
  3) `open_admission(A)` with brief `routing_lock` metadata update, then release `swapping_lock`

**Expand failure policy (fatal; Phase 3+4)**:
- If any step in `expand_workers(A)` fails (onload/start, sync-on-resume, or `open_admission`), treat it as a controlled-fatal error: log context and immediately call `orchestrator.shutdown(force=True, reason="expansion_failed", source="phase3+4.expand_workers")`.
- Rationale/reference: see **Verified Issues** → **Issue 81 & 124: Orphaned Expansion Signaling (Partial & Total Failure)** (this plan already mandates `shutdown(force=True)` on expansion failure).

#### 3) ROLL-side request_id plumbing + abort ACK
**Target files (upstream ROLL)**:
- `third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py` (set deterministic request_id)
- `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py` (respect request_id; targeted abort; active-rank routing)

**Implementation Details**:
- In `traj_env_manager.py` (`make_decision`):
  - `lm_input.meta_info["traj_id"] = traj_id`
  - `lm_input.meta_info["turn_id"] = self.rollout_cache.step`
  - `lm_input.meta_info["attempt"] = self.rollout_cache.attempt`
  - `lm_input.meta_info["schedrl_request_id"] = build_request_id(...)` (canonical; pipeline-scoped; stable across the system)
- Increment `rollout_cache.attempt` in `make_decision()` retry path.
+
+**RolloutCache `attempt` persistence (required)**
+- Implementation note: `RolloutCache` must include an `attempt: int` field to persist the current attempt counter for the active turn. This ensures `attempt` is stable across retries and is used in the canonical `request_id`:
+  - Add `attempt: int = 0` to `RolloutCache` in `third_party/ROLL/roll/pipeline/agentic/env_manager/base_env_manager.py`.
+  - Initialize `rollout_cache.attempt = 0` when the cache is created/reset for a new trajectory.
+  - In `make_decision()` propagate `lm_input.meta_info['attempt'] = rollout_cache.attempt` (instead of hard-coding 0).
+  - When a retry of the same turn is issued (adapter/scheduler decides to re-run the turn), increment `rollout_cache.attempt += 1` before reissuing the generation request so `request_id` reflects the new attempt.
+  - Add unit tests to cover attempt persistence and increment semantics.

#### 4) Explicit tasks to address upstream ROLL gaps (must-do in this phase)

- Implement shrink-to-zero (remove `Cannot shrink to zero active ranks` guard) and define behavior for `active_dp_ranks == set()`.
- **P0-1 (explicit task)**: Add `swapping_lock = asyncio.Lock()` to upstream `RequestScheduler.__init__()` and require **all** shrink/expand lifecycle operations to hold `swapping_lock` for the full duration (abort → drain → offload/onload → resume). This is mandatory to prevent concurrent shrink/expand races on the same worker physical state (offload vs load).
- **P0-2 (explicit task)**: Add a suspend re-check in upstream `RequestScheduler.generate_one_request()` to close the TOCTOU window:
  - After passing `_check_suspend()`, once `routing_lock` is acquired, re-check `need_suspend` (or equivalently re-enter `_check_suspend()` in a loop).
  - If `need_suspend=True`, release `routing_lock` and wait/loop until resumed, so request routing cannot observe `active_dp_ranks == ∅` and raise `RuntimeError("No active DP ranks")`.
- **P0-3 (explicit task)**: Prevent RolloutScheduler from “auto-resuming” RequestScheduler behind the Adapter/SchedRL lifecycle contract.
  - **Finding**: Upstream `RolloutScheduler.get_batch()` currently calls `await self.generate_scheduler.resume.remote()` unconditionally before it waits for environment outputs. This breaks ENG-123 ownership: during a shrink/offload transition, SchedRL/Adapter may intentionally keep the suspend gate closed, but `get_batch()` can re-open it and reintroduce routing races.
  - **Required change**: In the Adapter-controlled multi-pipeline path, `RolloutScheduler.get_batch()` MUST NOT call `resume()` (and MUST NOT call `suspend()` either). Only the Adapter/SchedRL may change suspend state.
  - **Touchpoint**: `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py` (`get_batch` resume call).
- **Decision (do NOT port fork-only helpers)**:
  - Do NOT port fork `RequestScheduler.get_suspend_state()` into upstream as an ENG-123 requirement. With P0-3, suspend ownership is single-source-of-truth, and state checks (if needed) can be done in SchedRL/Adapter logs/telemetry.
  - Do NOT port fork `check_gen_active_allocation_with_no_dp_workers()` into upstream. Cross-scheduler consistency validation is a SchedRL concern (SchedRL owns the centralized allocation view).
- Add `pipeline_id: Optional[str] = None` to upstream `RolloutScheduler.__init__()` and prefix both `GroupQueueManager` + `RequestScheduler` Ray actor names via `.options(name=...)` to avoid multi-pipeline actor-name collisions (E2). Keep this as a naming-only change (no child-actor signature changes required).
- Fix `RequestScheduler._validate_calculated_ranks(ranks, mode)` expand-mode validation bug in upstream ROLL: shrink must validate ranks are active; expand must validate ranks are inactive (mode-aware check).
- Enforce canonical request id format using `schedrl.protocol.request_id`.
- Ensure sticky routing never targets inactive dp ranks after shrink.

**Implementation Reminders (Verified Issues)**:
- **Issue 85 (Port)**: Framework strategies (SGLang) must not hardcode ports. Use `get_free_port()` (the same mechanism used by vLLM paths) backed by the ENG-123 SharedStorage atomic `claim_port` logic. Do not use deterministic port offsets/hashes as the primary mechanism.
- **Issue 86 (Offload)**: `VllmStrategy` weight offloading must be triggered whenever the scheduler issues a shrink/stop, regardless of whether internal colocation is detected, to ensure GPUs are freed for other pipelines.
  - Apply the same rule to SGLang: `SGLangStrategy.offload_states` must release memory on scheduler shrink/stop even if `is_actor_infer_colocated` is false.
  - Rollout offload validation (required; shrink/offload path): after `offload_states_partial(dp_ranks)` completes, verify device-level occupied percent via `torch.cuda.mem_get_info()` is `<= 10%` (fail-fast if higher).
    - Use the same `10%` threshold as the `state_offload_manager` memory validation section (device-level `occupied_pct` check).
    - We need both checks because they cover different mechanisms:
      - `state_offload_manager` validates offload/reload done via the context-manager path (training/critic steps).
      - rollout shrink/offload uses `offload_states_partial(dp_ranks)` and does not go through `state_offload_manager`.

#### 4) Progress mapping
Emit SchedRL `report_progress(...)` from ROLL rollout bookkeeping (trajectory units; 2% bands), mapping from groups as described in:
- `design_doc/multi-pipeline-adaptation-plan.md`
- `design_doc/adaptation_roll.md`

### Success Criteria

#### Manual Verification
- [ ] Run the MathEnv config and confirm end-to-end training still works:
  - `third_party/ROLL/examples/qwen3_agentic_gem/gem_math_dapo.yaml`
- [ ] Trigger a shrink mid-rollout and confirm:
  - no routing to removed ranks after shrink
  - aborted turns retry safely (no duplicated env steps)
  - shrink waits for abort ACK before stop/offload
  - shrink-to-zero works (generation cluster can fully release GPUs), then expand back and continue.

## Phase 4: Migrate selective model update (resume/expand weight sync) behind the Adapter

### Overview
Port `ModelUpdateService` from `ROLL_multi_pipeline` to enable selective weight synchronization for expanding DP workers.

**Upstream parity note (critical)**:
- The fork `ModelUpdateService.selective_update()` depends on fork selective-update components, including dynamic NCCL group creation + teardown (`SelectiveModelUpdateGroup` + `teardown_collective_groups(...)` on all participants).
- For ENG-123 we allow minimal upstreamable ROLL patches when adapter-only alternatives are clearly worse (see framework patching policy above).
- This phase explicitly enumerates the upstream additions required; do not claim “upstream-only worker RPCs” unless verified.

**Bucket caching policy (Option A boundary exception; required for TP>1)**:
- Selective sync performance for TP>1 requires bucket caching/staging; adapter-only solutions are typically too slow (rebuilding/staging per update) and risk timeouts.
- ENG-123 explicitly allows a **minimal, upstreamable framework hook** to wire sender-strategy-owned bucket caching (and only that).
- Keep the patch narrowly scoped (no scheduler policy, no pipeline orchestration logic in core ROLL): the hook should only expose the minimal strategy/worker surface needed to build/cache CPU buckets for selective updates.

### Changes Required

#### 1) Port `ModelUpdateService` to Adapter
**Source**: `third_party/ROLL_multi_pipeline/roll/distributed/executor/model_update_service.py`  
**Target**: `third_party/ROLL/roll/schedrl_adapter/model_update_service.py`

**What to keep**:
- Sender-side cache logic (`refresh_sender_cache`, `register_sender_cache`).
- Subset-DP targeting semantics:
  - Primary: Adapter passes `tgt_dp_ranks` into `ModelUpdateService.sync_selected_workers(tgt_dp_ranks=...)`, selecting which DP-rank engines to sync (vLLM `collective_rpc(_async)` scope is TP/PP workers inside that engine, not all DP ranks).
  - Guardrail: keep the fork's receiver-side allowlist (`set_model_update_allowed_dp_ranks(tgt_dp_ranks)` + cleanup reset) so even if a future strategy path fans out wider at the cluster level, non-target DP ranks can fail fast / early return. This mirrors the fork behavior and makes the subset contract explicit.
- **Bucket caching (required; TP>1; Megatron-only in ENG-123)**: implement bucket building + caching via an adapter-owned thin wrapper around the upstream strategy.
  - Hook point (agentic pipelines): wrap the strategy once right after `self.strategy = create_strategy(worker=self)` in `third_party/ROLL/roll/pipeline/base_worker.py:ActorWorker.initialize`.
  - Backend dispatch: the wrapper checks `self.strategy.strategy_name` (fallback: `worker_config.strategy_args.strategy_name`).
  - ENG-123 scope: only `megatron_train` is supported for bucket caching initially; other backends must fail fast with `NotImplementedError`.
  - Reference implementation: follow fork behavior (`third_party/ROLL_multi_pipeline/roll/distributed/executor/worker.py: build_model_update_bucket_cache` and fork selective update path) when implementing the Megatron caching logic.
  - Control flow note (avoid confusion; source of truth):
    - Cache build cadence: sender-side bucket cache is built after each train step (or whenever a new weight version is produced) and keyed by checkpoint version / `global_step`.
    - Promotion/activation: coordinator controls when a cached version becomes `active_rollout_checkpoint_version` via explicit promote signaling (strategy holds version metadata + cache).
    - Sync trigger: the global scheduler triggers selective sync (e.g., on expand/resume). Adapter calls `ModelUpdateService.sync_selected_workers(tgt_dp_ranks=...)`, and the service syncs the already-promoted active version from sender strategy cache state (service does not choose versions).
    - `ActorWorker.initialize` remains wiring-only: it enables the sender strategy to maintain cache state; it does not drive build cadence or promotion timing.

  - **Boundary note (intentional minimal framework hook)**: this `ActorWorker.initialize` hook is an intentionally small, upstreamable ROLL patch to enable sender-strategy-owned bucket caching needed for ENG-123 selective sync. Keep the hook narrowly scoped to bucket-cache wiring only (no scheduler policy in core ROLL), and prefer adapter-only integration patterns elsewhere.
- Timeout handling and validation.
- **Strict Versioning**: sender strategy owns `active_rollout_checkpoint_version` and cache metadata; coordinator controls promotion timing via explicit signal. `ModelUpdateService` orchestrates sync using sender strategy state only. Strategy-side checks enforce sender-cache freshness (`cached_global_step <= active_global_step`) so expanding workers never receive "future" weights if activated mid-step.
- **Serialization & Coalescing**: Never abort an active sync. If multiple sync requests arrive, skip intermediate versions and only sync the latest requested one (Highest Version Wins).
- Colocated vs. separated worker classification.

**What to change**:
- Port fork's `SelectiveModelUpdateGroup` teardown semantics into upstream and ensure dynamic group lifecycle is safe:
  - Teardown must run on every participant worker (sender + all receiver ranks that joined the group).
  - Prefer `try: ... finally: group.teardown()` in the selective update orchestration so exceptions/timeouts do not leak groups.
- **Worker RPC surface (ENG-123)**: port only the Worker RPCs that are actually required by the selective-update design; do not assume they exist upstream.
  - If the implementation needs fork-only RPCs (as the fork does), add them as minimal upstreamable Worker methods and list them explicitly in the PR.
- **Efficient Broadcast**: Ensure the service uses the cached buckets. For colocated targets, pass the cached serialized buffer directly in the `update_parameter_in_bucket.remote()` call.
  - **CRITICAL (upstream signature compatibility)**: upstream vLLM `RollWorker.update_parameter_in_bucket(serialized_named_tensors, ...)` indexes `serialized_named_tensors[self.rank]`. Therefore, pass a **list** (not a dict) where the element at index `worker.rank` contains that worker's bucket payload.
    - For selective updates to a subset of workers, use a full-length list sized to the rank-space (typically `len(infer_cluster.workers)` / `max(worker.rank)+1`), filling non-target indices with a sentinel (e.g., `None`) and ensuring each target worker receives its payload at `serialized_named_tensors[worker.rank]`.
    - If we need a more memory-efficient sparse transport, that requires either (a) a small upstreamable worker change to accept a dict, or (b) an adapter-defined rank remapping layer; do not assume dict indexing works with upstream vLLM.

#### 2) Implementation: Selective Update Flow (high-level)

The service (Ray Actor) orchestrates the sync using only upstream Worker RPCs (`setup_collective_group`, `broadcast_parameter`, `update_parameter_in_bucket`) and strategy-level bucket caching.

High-level steps:
1. Identify newly expanded DP ranks (`tgt_dp_ranks`) and read `active_rollout_checkpoint_version` from sender strategy cache state.
2. Trigger sender-side strategy bucket caching for the sender-strategy active rollout checkpoint/global step (required for TP>1). The cached CPU buckets are reused across future expansions.
3. Split targets into:
   - **colocated targets** (same node as sender): use `update_parameter_in_bucket` (CUDA IPC path) with cached buckets.
   - **separated targets** (different node):
     - create a collective group for sender + selected receivers
     - broadcast cached buckets via `broadcast_parameter`
     - **required upstream fix**: group teardown must call `dist.destroy_process_group` to avoid leaks; do not rely on dict-only cleanup. Teardown must be executed on all participant workers.
4. Receiver apply path (ENG-123): rely on `ROLL_SELECTIVE_MODEL_UPDATE_RECEIVER_DISABLE_CPU_STAGING=1` so receivers apply directly (no fork CPU staging / TP shard-aware unpack). The bucket payload uses upstream `send_recv_utils.py` format and upstream vLLM `update_parameter_in_bucket` contract.
5. Fail-fast on any timeout or partial failure.

#### 3) Worker Requirements

Reuse upstream APIs where possible:
- `worker.setup_collective_group.remote(...)` — init NCCL group participant
- `worker.broadcast_parameter.remote(group_name, ...)` — receive via NCCL
- `worker.update_parameter_in_bucket.remote(...)` — receive via CUDA IPC

**Upstream additions (required if used by the chosen implementation)**:
- Port only the fork-only Worker RPCs that the implementation actually calls (determine from the final selective-update implementation; keep this list minimal).
  - Candidate fork-only RPCs used by the reference implementation (TBD minimal set; verify against the final design):
    - **Required (fork-like, worker-driven selective update)**:
      - `start_selective_model_update(...)` — core selective update execution (sender-side orchestration calling strategy `selective_model_update_from_cache`).
      - `build_model_update_bucket_cache(...)` — required if using cached-bucket staging (default in fork); can be skipped only if we accept rebuilding/staging buckets each update (likely too slow for TP>1).

All three methods already exist in upstream `Worker` and are used by upstream `WeightUpdater` classes.

#### 4) Key Differences from Upstream

| Aspect | Upstream (Full-Cluster) | Selective (Adapter) |
|--------|------------------------|---------------------|
| **NCCL Group** | Static, created at init (`setup_model_update`) | Dynamic per update (requires real teardown via `dist.destroy_process_group`) |
| **Target Workers** | All workers in `infer_cluster.workers` | Subset specified by `tgt_dp_ranks` |
| **Group Lifecycle** | Created once, reused for all updates | Created before update, destroyed after |
| **When Used** | Standard training step model sync | Expand/resume operations only |

#### 5) Integration with Adapter RPC Surface

The `expand_workers` RPC in the ROLL Adapter calls the service:

```python
class ROLLAdapter:
    async def expand_workers(self, dp_ranks: List[int]):
        # ... validate, onload, start servers ...
        
        # Selective model update for newly expanded workers
        if self.model_update_service:
            await self.model_update_service.sync_selected_workers(
                tgt_dp_ranks=dp_ranks
            )
        
        # ... open_admission ...
```

### Success Criteria

#### Manual Verification
- [ ] Expand after a weight update results in new dp ranks serving the correct weights (no stale admission).
- [ ] Dynamic NCCL group creation/teardown does not leak resources (verify via `nvidia-smi` and process inspection).

**Required upstream change (ENG-123)**:
- Upstream `GroupManager.destroy_collective_group()` must call `dist.destroy_process_group(...)` (not just delete Python dict entries) and must have correct bookkeeping so dynamic group lifecycle does not leak NCCL resources.
- Add upstream `teardown_collective_groups(model_update_name, group_names)` surface (Strategy/Worker) and ensure selective update orchestration calls it for all participant ranks (typically from `SelectiveModelUpdateGroup.teardown()` in a `finally:`).
- [ ] Colocated path uses CUDA IPC (verify no NCCL traffic for same-node updates).
- [ ] Separated path uses NCCL broadcast (verify efficient cross-node transfer).

## Migration Notes

- `third_party/ROLL_multi_pipeline` becomes frozen reference code. No deletions in ENG-123.
- Scheduler logic is extracted into `schedrl/` and becomes the long-term home for cross-framework concurrency.
- ROLL changes remain a small shim + Adapter + minimal internal hooks (request_id, targeted abort+ACK, subset lifecycle, progress mapping).

## References

- Dual-mode plan: `thoughts/shared/plans/2026-01-28-schedrl-dual-mode-final-plan.md`
- Shared protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- ROLL adaptation notes: `design_doc/adaptation_roll.md`
- Original multi-pipeline reference design: `design_doc/archive/multi-pipeline_roll_old_design.md`
- Existing working implementation map: `thoughts/shared/research/2026-02-04-roll-multi-pipeline-timesharing-impl.md`
- Existing fork source: `third_party/ROLL_multi_pipeline/`

### Scheduler recovery on restart (fail-fast behavior)

- Assumption (ENG-123 scope): we only consider fresh pipeline starts; the plan does NOT implement recovery or rehydration of previous pipeline runtime state after a scheduler restart. Pipelines are expected to re-register from initial state.
- This project will NOT implement automatic scheduler recovery or complex re-attach reconciliation in ENG-123. The plan is explicit: on scheduler restart the system uses a fail-fast model.

- Operational policy (fresh-restart workflow):
  1. If the scheduler process stops or restarts, operators must treat the event as a full reset: stop existing pipeline jobs (or allow them to detect scheduler unavailability), then restart pipelines so they re-register against the fresh scheduler. No attempt is made to preserve or reconcile prior allocation state.
  2. Pipelines must be implemented to tolerate a scheduler restart by performing a full re-registration (register() + request_gpus()) on startup and not rely on previously-held `allocation_id` values.
  3. The scheduler will respond to any stale registration/metadata with a clear error message requesting re-registration; pipelines receiving that error must abort and restart their registration flow.
  4. For operators who need continuity across failures, run the scheduler as a long-lived service outside pipeline jobs and treat it as an HA service (outside ENG-123 scope).
- Config flag (Phase 1): `schedrl.fail_fast_on_restart = true` (default true) — ensures this behavior is applied by default.
- Test (manual / integration): simulate scheduler restart; pipelines must re-register and acquire fresh allocations. Any attempt to resume using stale allocation metadata must return an explicit, documented error instructing the pipeline to re-register.

## Lifecycle invariants & routing atomicity (MANDATORY clarifications)

These clarifications remove ambiguity around identifiers, retry semantics, FIFO stability, and safe rank teardown ordering. Add the following three items to the plan as REQUIRED implementation notes (copy-pasteable for PR descriptions).

1) `traj_id` availability in `make_decision()` (required)
- Finding: `traj_id` is currently computed after the rollout/decision loop in `run_rollout_loop()` and is not available when `make_decision()` calls the LLM proxy.
- Required change (code + plan): Move the `traj_id` computation into the `reset()` path so every `RolloutCache` instance has a stable `traj_id` available during generation.
  - Code sketch (implementer): in `third_party/ROLL/roll/pipeline/agentic/env_manager/base_env_manager.py` add a `traj_id: Optional[str] = None` field to `RolloutCache`; in `TrajEnvManager.reset()` compute and assign `rollout_cache.traj_id = f"{tag}_{group_id}_{episode_id}_{seed}_{env_id}"` (use the same canonical format used elsewhere); in `make_decision()` set `lm_input.meta_info['traj_id'] = self.rollout_cache.traj_id` before calling `llm_proxy.generate(...)`.

2) `creation_ts` persistence for FIFO stability (required)
- Finding: FIFO fairness uses `oldest_unfinished_creation_ts` but code does not persist a wall-clock creation timestamp across retries.
- TODO (deferred): we are not reporting `oldest_unfinished_creation_ts` in ENG-123.
  - If/when we add it, record `creation_ts` when the episode/trajectory is first created (in `reset()`) and propagate it in `RolloutCache` for the life of that trajectory, including across retries of the same turn/request.

3) Two-lock ordering + suspend-before-clear sequence (required)
- Finding: Clearing `src_rank2_dp_rank` mappings during shrink/expand can race with in-flight `generate_one_request()` calls which choose a dp rank using `_get_least_active_dp_rank()`; `_get_least_active_dp_rank()` raises if `active_dp_ranks` is empty.
- Required change (ordering + plan): adopt and document this safe sequence for rank teardown (shrink-to-zero and normal shrink):
  1. Acquire `swapping_lock` to serialize lifecycle operations.
  2. Set the suspend gate (call `suspend()` / set `need_suspend=True`) so `generate_one_request()` blocks at `_check_suspend()` and does not proceed to `_get_least_active_dp_rank()`.
  3. Acquire `routing_lock` and perform a *brief* mapping update: mark ranks as inactive and clear `src_rank2_dp_rank` entries for removed ranks. Release `routing_lock` immediately after updating mappings.
  4. Issue targeted aborts to affected workers and wait for `running_requests` to drain to zero. **Do not** hold `routing_lock` while awaiting abort ACKs (this avoids deadlock with other async callbacks). Use `empty_notifier`/`empty()` to await completion.
  5. Offload/stop the workers and collect post-release memory; return release ACK.
  6. If resuming, perform expand actions and call `resume()` (clear suspend gate), then release `swapping_lock`.
- Rationale: `swapping_lock` prevents concurrent offload/load races on the same workers; setting `need_suspend` first prevents TOCTOU assignment races; holding `routing_lock` only for the short mapping-edit window prevents stale mapping use while avoiding long lock holds that can deadlock callbacks.

These clarifications are mandatory: add them to the plan file and require implementers to reference them in PR descriptions. They are doc-only changes; no functional code is changed by this edit.
