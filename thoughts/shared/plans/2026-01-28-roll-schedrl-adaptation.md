# ROLL (AgenticPipeline, vLLM-only) SchedRL Adaptation Implementation Plan

## Overview

Adapt **ROLL’s Agentic pipeline** to the shared SchedRL protocol (`design_doc/multi-pipeline-adaptation-plan.md`) by adding: (1) DP-subset lifecycle control (shrink/expand), (2) deterministic per-turn request IDs + safe abort/retry migration with **targeted abort + backend-confirmed ACK from day one**, (3) selective sync-on-resume (subset-scoped weight sync), and (4) standardized `report_progress(...)` heartbeats.

Default protocol settings for ROLL Phase 1:
- `update_policy = QUIESCE-by-abort` (strict boundary; reuse ROLL’s `suspend()` + abort primitive)
- `migration_policy = REQUEST_RETRY` (abort in-flight turn, env loop retries without stepping; **targeted abort + ACK is required**)
- `expand_rebalance_policy = REBALANCE_QUEUED` (and optional in-flight rebalance later)

Main reference workload (Phase 1 validation target): `third_party/ROLL/examples/qwen3_agentic_gem/gem_math_dapo.yaml` (MathEnv).

---

## Current State Analysis

High-level gaps (summary):
- **Cluster control** is largely “all workers” today (no subset lifecycle surface).
- **Routing** uses sticky mappings that persist unless explicitly cleared.
- **Abort** must be upgraded to targeted abort + backend-confirmed ACK (finish reason `"abort"`) before stopping/offloading GPUs.
- **Weight sync** assumes full-cluster broadcast (no subset-scoped `model_update`).

### Coordinator and lifecycle today
- `AgenticPipeline.run()` sequences: offload train → suspend generation → (optionally stop server) → `model_update()` → start server → `RolloutScheduler.get_batch()` → train. (`third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:138`)
- Generation pause today is implemented as `RolloutScheduler.suspend()` which calls `RequestScheduler.suspend()` (abort + admission gate). (`third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py:357`, `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:959`)

### Abort/retry safety already present (good fit for REQUEST_RETRY)
- Env loop does **not** call `env.step(...)` unless generation completes with `FINISH`. Aborts retry the same turn. (`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:120`, `third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:222`)

### Request routing and request IDs (gaps)
- Agentic generation uses `RequestScheduler` (Ray actor) for routing and async futures. (`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:866`)
- `RequestScheduler.generate_one_request()`:
  - Sticky routing: `src_rank2_dp_rank[src_rank] -> dp_rank` where `src_rank` comes from env id. (`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:883`)
  - Request IDs are generated inside scheduler (`uuid + counter`), so the coordinator cannot provide deterministic per-turn IDs required by SchedRL. (`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py:888`)

### Subset lifecycle (missing)
- Inference server lifecycle is cluster-wide: `actor_infer.start_server(...)` and `actor_infer.stop_server()` in the pipeline loop. (`third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:154`, `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py:162`)
- There is no `expand_workers(indices)` / `shrink_workers(indices)` API that starts/stops only a subset of DP ranks.

### Selective sync (missing)
- `ModelUpdateGroup` builds a **static** broadcast comm plan and sets up collective groups once at init. (`third_party/ROLL/roll/distributed/executor/model_update_group.py:30`, `third_party/ROLL/roll/distributed/executor/model_update_group.py:111`)
- `model_update()` iterates the entire `broadcast_comm_pan` without filtering; this is incompatible with subset-scoped sync. (`third_party/ROLL/roll/distributed/executor/model_update_group.py:142`)

### Progress heartbeats (missing)
- Rollout progress is only a local `tqdm` bar in `GroupQueue`; there is no SchedRL-standard `report_progress(...)`. (`third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py:159`, `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py:138`)

---

## Desired End State

SchedRL-critical properties (summary):
- **vLLM-only** (this plan does not cover SGLang).
- **Strict ordering** for shrink: `Close Admission` → `Abort(P)` → `Wait for ACK` → `Stop/Offload(P)`.
- **Fail-fast safety**: if abort ACK does not arrive within timeout, crash the pipeline; do not proceed with stop/offload.
- **Elastic lifecycle**: `start_server_subset(indices)` / `stop_server_subset(indices)` (implemented as `expand_workers` / `shrink_workers` on the coordinator surface).
- **Robust routing**: sticky routing maps are cleared/rewritten on shrink/expand so new work never targets inactive ranks.

### SchedRL-facing contract (ROLL adapter)
Expose a coordinator-owned Adapter surface that matches the Final Plan:
- `close_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `open_admission(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `shrink_workers(worker_indices, action_id, activation_epoch) -> ActionResponse`
- `expand_workers(worker_indices, checkpoint_version, action_id, activation_epoch) -> ActionResponse`

### Multi-LoRA extension hook (ROLL tag/domain → `adapter_id`)
This plan is the baseline for **FULL_FT**. For **MULTI_LORA** (multiple LoRA adapters on a frozen base) see:
`thoughts/shared/plans/2026-02-02-schedrl-multi-lora-adapter-extension.md`.

Required ROLL-specific mapping for MULTI_LORA:
- Standardize on canonical `adapter_id` at the protocol boundary.
- Map `adapter_id := env_config["tag"]` (agentic envs) and `adapter_id := domain` (async sampling).
- Mixed-adapter batching uses a per-prompt `lora_request` list (one LoRARequest per prompt) so a single inference batch can contain multiple adapters.
- Shrink/expand semantics remain physical, and `migration_policy=REQUEST_RETRY` (abort + backend-confirmed ACK + retry) stays required.

Notes:
- `ActionResponse` is `{success: bool, error: Optional[str]}`; use `error="Superseded"` for supersession ACKs.
- Scheduler-owned timeouts are enforced at the scheduler→adapter RPC boundary; the Adapter MUST NOT require `timeout_s` parameters in the public RPC signature.
- On (re)registration, assume `S_actual={}` and release/kill any leftover inference servers from a prior scheduler session.

Additional (non-Adapter) surfaces used by the scheduler client (unchanged from shared protocol):
- `report_progress(queued_trajectories, inflight_trajectories, percent_completed, oldest_unfinished_creation_ts, metrics=...)`

### Deterministic request IDs (SchedRL requirement)
The coordinator must provide a deterministic per-turn `request_id` and pass it into the backend (vLLM `request_id`) so shrink can do **targeted abort + backend-confirmed ACK + retry** safely.
Chosen format for ROLL Phase 1:
- `trajectory_id = traj_id` already produced in env loop (`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py:129`)
- `turn_id = rollout_cache.step` (the current decision index within trajectory)
- `request_id = f\"{traj_id}:{turn_id}\"`
- Engine-error retries (not preemption retries) append a small `:err{n}` suffix; preemption retries reuse the same `request_id` (SchedRL rule: do not cap preemption retries).

### Subset-scoped sync-on-resume
Implement selective sync so that:
- For time-sharing resize with unchanged weights: resume does **not** call `model_update()`.
- For resumes across a weight change: resume does `model_update(worker_indices=active_subset)` right before admission reopens on that subset.

### Heartbeats
Implement a deterministic mapping from ROLL group/episode bookkeeping to:
- `queued_trajectories` (queued not started)
- `inflight_trajectories` (running / pending)
- `percent_completed` (collected ready-for-train / step_target_trajectories)
- `oldest_unfinished_creation_ts` (enqueue time of oldest unfinished trajectory)

---

## What We’re NOT Doing

- No Ray placement-group resizing or runtime device remapping (Phase 1 uses fixed placement).
- No cross-pipeline shared inference router (SchedRL assumes per-pipeline isolated engine groups).
- No SGLang support in this plan (vLLM-only target).
- No Phase 2 env migration work (WebShop) in this plan.
- No new test files; use existing ROLL tests and existing examples.
- No new third-party libraries or packages.

---

## Implementation Approach

Principle: make the smallest changes that satisfy the shared protocol while reusing ROLL’s existing strengths:
- Use the existing abort/retry-at-turn-boundary semantics in the env loop.
- Add optional `worker_indices` arguments and small helper APIs rather than large refactors.
- Prefer coordinator-level sequencing (close admission → abort/ACK or drain → stop/offload) exactly as required by the shared protocol.

Key safety change (targeted abort + ACK):
- Do not “fake-complete” local futures on abort. Instead, send abort to vLLM and wait for the backend to return `finish_reason == "abort"` for each targeted request id. Only after ACK is it safe to stop/offload the shrinking subset.

---

## Phase 1: Targeted Abort+ACK + Deterministic Request IDs + DP-Subset Lifecycle

### Overview
Make shrink/expand correct and safe for agentic multi-turn rollouts by:
1) letting the coordinator provide deterministic request IDs,
2) preventing routing to inactive DP ranks,
3) implementing **targeted abort + backend-confirmed ACK** (vLLM `finish_reason == "abort"`) and waiting for ACK before retry,
4) adding subset start/stop APIs (DP-granular server lifecycle),
5) enforcing strict ordering: `Close Admission` → `Send Abort` → `Wait for ACK` → `Stop/Offload`.

### Changes Required

#### 1) Deterministic per-turn `request_id`
**Files**:
- `third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py`
- `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`

**Changes**:
- In `TrajEnvManager.make_decision(...)`, set a deterministic request id on `lm_input.meta_info["request_id"]` using the same `traj_id` format used for completed rollouts and `rollout_cache.step`.
- In `RequestScheduler.generate_one_request(...)`, respect `data.meta_info["request_id"]` if present, instead of overwriting it.
  - Keep the current uuid+counter fallback only if `request_id` is missing.

#### 2) Admission gating and sticky routing cleanup on shrink/expand
**File**: `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`

**Changes**:
- Add `active_dp_ranks: set[int]` to `RequestScheduler` (initialized to all ranks).
- Change routing to pick only from `active_dp_ranks`.
- When `src_rank2_dp_rank[src_rank]` points to an inactive rank, delete the mapping and re-pick from `active_dp_ranks`.
- On shrink, clear sticky routing mappings that target removed ranks.

#### 3) Targeted abort + backend-confirmed ACK (day one)
**Files**:
- `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`
- `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py`

**Definition (SchedRL-standard)**
- ACK means: the request finishes with `finish_reason == "abort"` (from vLLM output).

**Changes**
- Modify `RequestScheduler.report_response(...)` to treat an output as “abort” when `data.meta_info["finish_reasons"]` contains `"abort"` (and resolve the future to `None` in this case).
- Implement `RequestScheduler.abort_request(worker_indices: list[int] | None = None, timeout_s: float = ...)`:
  - Snapshot targeted request ids and their futures *before any await*.
  - For each targeted dp rank, send one vLLM abort command with `meta_info["request_id"] = [rid1, rid2, ...]`.
  - Wait for ACK by awaiting the snapped futures, and require that each resolves to `None` (abort) or to a completed response (already finished before abort).
  - Timeout policy: if not all targeted requests resolve within the configured abort-ACK timeout, crash the pipeline (fail fast and loudly).
- Ensure vLLM abort supports list abort (already the case for ROLL’s vLLM wrapper): `roll/third_party/vllm/*/llm.py` defines `abort_request(self, request_id: Union[str, Iterable[str]])`.

#### 4) Subset lifecycle APIs (scheduler → coordinator surface)
**File**: `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py`

**Changes** (host these on `RolloutScheduler` for Phase 1 to keep surface small and colocated with existing rollout control):
- Add `close_admission(worker_indices: list[int], action_id: str, activation_epoch: int) -> ActionResponse`:
  - Set `active_dp_ranks := S \\ P` (so no new work routes to `P`).
- Add `open_admission(worker_indices: list[int], action_id: str, activation_epoch: int) -> ActionResponse`:
  - Set `active_dp_ranks := S ∪ A` (only after expand is complete and any required sync is done).
- Add `expand_workers(worker_indices: list[int], checkpoint_version: int, action_id: str, activation_epoch: int) -> ActionResponse`:
  - Call `infer_cluster.workers[i].start_server.remote(DataProto(meta_info={...}))` for each `i`.
  - Perform subset-scoped sync-on-resume if `checkpoint_version` changed while workers were inactive (Phase 3).
  - Do NOT reopen admission here; the scheduler calls `open_admission(...)` as a separate action.
  - Optionally clear sticky routing for queued rebalance (Phase 1 uses “rebalance queued” only).
- Add `shrink_workers(worker_indices: list[int], action_id: str, activation_epoch: int) -> ActionResponse`:
  - Close admission MUST be done first (scheduler calls `close_admission(...)` as a separate action).
  - Abort in-flight work on `P` and wait for ACK before stop/offload (uses `RequestScheduler.abort_request(worker_indices=P, ...)`).
  - Call `infer_cluster.workers[i].stop_server.remote()` for each `i` in `P`.

#### 5) Strict ordering enforcement
**Files**:
- `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py`
- `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`

**Changes**
- Enforce ordering for shrink:
  1) Set `active_dp_ranks := S \\ P` (close admission to `P`)
  2) `abort_request(worker_indices=P)` and **wait for ACK**
  3) Stop/offload `P` via `stop_server_subset(P)`
- Ensure ordering for expand:
  1) Start servers on `A`
  2) Open admission (`active_dp_ranks := S ∪ A`) only after start is complete (and after sync if a weight change is involved, Phase 3).

### Success Criteria

#### Automated Verification
- [ ] ROLL unit/integration suite passes: `cd third_party/ROLL && make test`

#### Manual Verification
- [ ] Run MathEnv config (`third_party/ROLL/examples/qwen3_agentic_gem/gem_math_dapo.yaml`) and confirm training still runs end-to-end.
- [ ] Force a mid-rollout shrink (manually call `RolloutScheduler.shrink_workers([some_dp_rank])`) and verify:
  - rollouts continue (no deadlock),
  - aborted turns retry (no env-step double-commit),
  - no routing to the removed rank occurs after shrink.
  - retries happen only after ACK (no immediate “fake” completion).

Implementation note: pause after this phase for a human confirmation that the manual shrink test is safe for the targeted env (MathEnv).

---

## Phase 2: `report_progress(...)` Heartbeats (2% Bands, Trajectory Units)

### Overview
Emit SchedRL-standard progress heartbeats so the central scheduler can make fair shrink/expand decisions and avoid thrashing.

### Changes Required

#### 1) Track creation timestamps for unfinished work
**File**: `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py`

**Changes**
- Extend `GroupData` with `creation_ts: float` (set when group is created in `advance_group(...)`).
- Use this timestamp as the trajectory enqueue time for `oldest_unfinished_creation_ts`.

#### 2) Implement progress aggregation in trajectory units
**File**: `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py`

**Mapping decisions**
- A “trajectory unit” corresponds to one `DataProto` rollout produced by an env manager (group size already counts trajectories).
- Convert group counts → trajectory counts by multiplying by `group_size`.

**Implementation sketch**
- In `GroupQueueManager`, add methods to compute:
  - `queued_groups`: groups with `running_rollouts == 0`
  - `inflight_groups`: groups with `running_rollouts > 0` and not complete
  - `ready_groups`: groups where `len(rollouts) >= group_size` and not yet consumed
  - `oldest_unfinished_creation_ts`: min `creation_ts` among queued+inflight groups
- Convert to trajectory units:
  - `queued_trajectories = queued_groups * group_size`
  - `inflight_trajectories = inflight_groups * group_size`
  - `collected_trajectories = ready_groups * group_size`
  - `percent_completed = collected_trajectories / rollout_batch_size` (ROLL mapping from shared doc)
- Emit heartbeats:
  - at batch start, and
  - whenever `percent_completed` crosses a 2% band boundary.

### Success Criteria

#### Automated Verification
- [ ] ROLL suite passes: `cd third_party/ROLL && make test`

#### Manual Verification
- [ ] Heartbeats reflect expected progress as rollouts are collected (monotonic band increases under normal operation).
- [ ] `oldest_unfinished_creation_ts` is stable across preemption retries (not reset per retry).

---

## Phase 3: Selective Sync-on-Resume (Subset-Scoped Model Update + CPU Weight Cache)

### Overview
Enable scheduler-driven shrink/expand without requiring rollout GPUs to remain active at every train step by:
- staging latest weights into a trainer-side CPU cache after training, and
- broadcasting/selectively syncing weights to only the active rollout subset when resuming admission.

### Changes Required

#### 1) Add a “latest weights CPU cache” owned by training side
**File(s)**:
- `third_party/ROLL/roll/distributed/executor/model_update_group.py`
- (or training strategy) `third_party/ROLL/roll/distributed/strategy/megatron_strategy.py`

**Decision**
Keep only the latest version in the cache for Phase 1 ROLL (keyed by `global_step`), and treat it as the source of truth for sync-on-resume.

**Implementation sketch**
- After `train_step()` completes (or at the end of the step boundary), snapshot weights to CPU buckets and retain as `latest_cpu_weights[global_step]`.
- Do not require rollout workers to be active during this snapshot.

#### 2) Make `ModelUpdateGroup` subset-scoped
**File**: `third_party/ROLL/roll/distributed/executor/model_update_group.py`

**Changes**
- Add `model_update(worker_indices: list[int], step: int, version: int)`:
  - Filter target devices to the selected DP subset only.
  - Create subset-scoped collective groups for the update, then tear them down afterward.
  - Execute broadcast using the cached CPU weights as source (load on training side only when needed).

#### 3) Move from “sync every step” to “sync-on-resume”
**File**: `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py`

**Changes**
- Replace unconditional `model_update(global_step)` with scheduler-driven sync intent:
  - coordinator calls `request_checkpoint_sync(active_checkpoint_version, worker_indices=desired_S)` when required,
  - scheduler decides timing and then calls into the coordinator surface to actually run the sync and reopen admission.

### Success Criteria

#### Automated Verification
- [ ] ROLL suite passes: `cd third_party/ROLL && make test`

#### Manual Verification
- [ ] Resume after shrink+train causes rollout workers to generate using the new weights (no stale-weight admission).
- [ ] Sync is performed only on the active subset; shrunk/offloaded workers do not participate.

---

## Testing Strategy (No New Test Files)

### Automated
- `cd third_party/ROLL && make test`

### Manual (core correctness)
1. Run MathEnv agentic training (`third_party/ROLL/examples/qwen3_agentic_gem/gem_math_dapo.yaml`).
2. Trigger shrink while rollouts are in flight (Phase 1: targeted abort+ACK on `P`, then stop/offload `P`).
3. Confirm no duplicated env steps or tool side effects occur across abort/retry.

---

## Performance Considerations

- Phase 1 global aborts are correct but may waste work; Phase 2 targeted abort reduces wasted work.
- Subset-scoped NCCL group setup/teardown adds overhead; use it only when resuming across a weight change.
- Heartbeat cadence is event-driven (2% bands) to avoid spamming the scheduler.

---

## References

- Protocol: `design_doc/multi-pipeline-adaptation-plan.md`
- ROLL adaptation notes: `design_doc/adaptation_roll.md`
- Research: `thoughts/shared/research/2026-01-28-schedrl-framework-mechanisms.md`
- Research: `thoughts/shared/research/2026-01-28-schedrl-adaptation-research.md`
