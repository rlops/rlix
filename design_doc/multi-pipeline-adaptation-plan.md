---
title: "SchedRL: Multi-Pipeline Adaptation Plan (Clean Structure)"
status: active
canonical_path: design_doc/multi-pipeline-adaptation-plan_clean.md
source_docs:
  - design_doc/multi-pipeline_roll_old_design.md
  - design_doc/archive/adaptation_review.md
  - design_doc/adaptation_nemo_rl.md
  - design_doc/adaptation_roll.md
  - design_doc/adaptation_miles.md
  - design_doc/adaptation_skyrl.md
---

# SchedRL: Multi-Pipeline Adaptation Plan (Clean Structure)

This is the main shared protocol for multi-pipeline GPU sharing across
**NeMo-RL**, **ROLL**, **Miles**, and **SkyRL-train**.

For concrete per-framework code entrypoints and limitations, see:
- `design_doc/adaptation_nemo_rl.md`
- `design_doc/adaptation_roll.md`
- `design_doc/adaptation_miles.md`
- `design_doc/adaptation_skyrl.md`

Main reference async + multi-turn examples (one per framework):
- ROLL: `third_party/ROLL/examples/qwen3_agentic_gem/gem_math_dapo.yaml` (MathEnv; already exists)
- NeMo-RL: sliding puzzle async example (planned; see `design_doc/adaptation_nemo_rl.md`)
- Miles: Retool async example (planned; see `design_doc/adaptation_miles.md`)
- SkyRL-train: GSM8K multi-turn + async trainers (supported; see `design_doc/adaptation_skyrl.md`)

Phase 2 (complex agent tasks) examples:
- ROLL: **WebShop** (planned as the Phase 2 agent task in ROLL; see `design_doc/adaptation_roll.md`)
- NeMo-RL: **Mini-SWE** (planned; reuse NeMo-Gym Mini-SWE agent server; see `design_doc/adaptation_nemo_rl.md`)
- Miles: **Mini-SWE** (planned; upgrade `third_party/miles/examples/experimental/swe-agent/` to `train_async.py`; see `design_doc/adaptation_miles.md`)
- SkyRL-train: **Mini-SWE** (planned; add async entrypoints for `third_party/SkyRL/skyrl-train/examples/mini_swe_agent/`; see `design_doc/adaptation_skyrl.md`)
  - Safety note for Phase 2 tool loops: for shrink/time-sharing, default to “stop new starts + wait for drain”. Do not assume mid-tool abort is safe.

Reference (ROLL-specific sequence/design inspiration):
- `design_doc/multi-pipeline_roll_old_design.md`

---

## 1) Terminology (Brief)

- **Pipeline**: one RL run (generator + trainer + worker clusters).
- **Pipeline coordinator**: per-pipeline driver; chooses what checkpoint is *active* for rollouts.
- **Central scheduler**: global service; allocates/reclaims GPUs and decides timing of sync/resize to satisfy the coordinator’s intent.
- **Training cluster**: non-preemptible compute once granted (train/value/logprob/reference/reward).
- **Generation cluster**: preemptible rollout compute (DP workers), dynamically resized by the scheduler.
- **DP worker / DP rank**: atomic generation allocation unit; consumes `tp_size` GPUs as a bundle.
- **Active subset** `S`: currently active DP workers for a pipeline’s generation cluster.
  - **Preempted subset** `P = old_S − new_S`, **Activated subset** `A = new_S − old_S`.
- **Request**: one generation call (prompt → response). **Trajectory**: multi-step rollout across requests.
- **Admission**: whether new work may start on a worker/subset (admission control is used for safety).

**Checkpoints**
- **Checkpoint version**: monotonic id, typically `global_step`.
- **Latest checkpoint**: newest produced by training.
- **Active checkpoint**: the **authoritative target** checkpoint version that rollout workers must **sync/activate to** before (re)opening admission. New rollout admission uses this version. It can intentionally lag latest.
- **Trainer-side checkpoint cache service (CPU bucket list)**: source-of-truth CPU-resident weights cache owned by a trainer-side service/actor. The pipeline coordinator controls it via RPC. This cache is used to sync weights to rollout workers during expand/resume, because shrink/offload can drop worker-local weights.

**Isolation assumption (important)**
- Each pipeline owns a dedicated **engine group** (cluster / worker group / set of rollout engines).
- That engine group is isolated and is **not shared** with other pipelines.
- Sharing happens only at the **GPU allocation** level (time-sharing via shrink/expand/offload), not by sending multiple pipelines’ requests into the same rollout engine/router.

---

## 2) Taxonomy (Mechanisms, Staleness, Framework Mapping)

SchedRL unifies cross-framework behavior using two orthogonal knobs:

**A) Model update / activation policy (`update_policy`)**
- `QUIESCE-by-drain`: close admission, then wait for in-flight work to finish naturally (`inflight -> 0`), then sync/activate; strict consistency. **Drain granularity is framework-defined** (e.g., drain per request/turn vs per prompt-group rollout job).
- `QUIESCE-by-abort`: close admission, abort in-flight work, wait for abort ACK, ensure `inflight == 0`, then sync/activate; strict consistency.
- `BATCH`: activation only at batch boundaries (no mid-batch cutover); typical for pipelined trainers.
- `INFLIGHT`: in-place update on active workers without requiring stop/resume; in-flight requests may finish on old weights (requires version tagging and backend support).

**QUIESCE variants (rules)**
- Pick exactly one of `QUIESCE-by-drain` or `QUIESCE-by-abort` for a given boundary (do not mix drain and abort).
- `QUIESCE-by-abort` requires `REQUEST_RETRY` safety (two-phase commit) because aborted work must be retried elsewhere/later.
- Make the drain unit explicit in adapters:
  - **Turn/request drain**: close admission for new turns/requests and wait until no in-flight requests remain. This guarantees no mid-request mixing, but a multi-turn trajectory can still span versions across turns.
  - **Trajectory-group drain**: close admission for new trajectory groups (e.g., NeMo-RL prompt-group rollout threads) and wait until no in-flight groups remain. This guarantees no interruption inside that group’s multi-turn rollout loop.


**B) Preemption migration policy during shrink (`migration_policy`)**
- `REQUEST_RETRY` (baseline for elastic time-sharing): abort/cancel in-flight work on `P` and retry it on `new_S`.
  - This is typically **request/turn-level retry** (re-run the current generation step on another worker) and does **not** imply restarting the entire multi-turn trajectory if the framework can re-issue the same turn with the same trajectory context and the environment state has not advanced.
  - `PARTIAL_ROLLOUT_MIGRATE` (future/backlog): persist step-level checkpoints and re-enqueue remaining steps as backlog/resume tokens.

**Safety invariant (required for `REQUEST_RETRY`)**
- Two-phase commit for stateful effects: the framework must structure the agent loop so that any stateful tool/env side effects (e.g., `env.step()`) happen **only after**
  a **non-abort** generation result is received, and there is a single “writer” that commits those effects.
- This must be validated per framework adapter; if it is not true, `REQUEST_RETRY` can cause duplicate side effects and is unsafe.
- Stable turn request id is required: the pipeline coordinator must generate a deterministic per-turn request id, e.g. `request_id = f"{trajectory_id}:{turn_id}:{attempt}"`,
  and pass it into the rollout engine as the backend request id (vLLM `request_id`, SGLang `rid`). This is required so shrink can do targeted abort+ACK+retry safely.
  - This requires a stable `trajectory_id` (same across retries) and a stable `turn_id` within that trajectory.
  - On retry after abort ACK, increment `attempt` so each attempt has a unique backend request id.
  - Error retry limit (safety): cap **engine error retries** per `(trajectory_id, turn_id)` (default: 3, configurable). If exceeded, drop the trajectory and report a metric.
    - Do **not** cap “preemption retries” (abort due to shrink/expand rebalance). Preemption retries are part of time-sharing.
    - Phase 1 guard: if a turn is preempted “too many” times (high threshold), crash the pipeline (fail fast and loudly). Backlog: replace crash with “wait for drain”.
- Abort confirmation is required: when a worker is preempted and in-flight work must be retried elsewhere, the adapter must not reissue the same request/turn until the
  original request has returned an explicit abort outcome (or equivalent “request is stopped” confirmation from the backend). HTTP 200 from an abort endpoint alone is not sufficient.
  - Adapters may send abort commands in batch (where supported) and then wait for abort outcomes/ACKs collectively before reissuing retries on the remaining active workers.
  - For vLLM/SGLang, ACK means: the request finishes with `finish_reason/stop_reason == "abort"`.
  - Timeout rule: if ACK does not arrive in time, **crash the pipeline** (fail fast and loudly) and do not continue shrink/expand (do not offload/stop a worker that may still be running work).

**C) Expand rebalance policy (`expand_rebalance_policy`)**
- Expand does not require migration for correctness (new workers can just take new work after syncing weights).
- For elasticity/time-sharing we use a **stronger expand rebalance by default** (faster load redistribution):
  1. Route new arrivals to the expanded workers first.
  2. Reassign queued/not-started work so it can run on the expanded workers.
  3. If load is still uneven, abort selected in-flight turns on overloaded workers and retry them on underloaded workers until the load is within tolerance.
     - This uses the same `REQUEST_RETRY` safety rules as shrink: deterministic per-turn request id + abort ACK required + two-phase commit for any stateful tool/env effects.
	     - Stop condition: stop abort+retry when the load gap across active workers is within **5%**.
	       - Definition: let `load[dp] = queued_trajectories_by_worker[dp] + inflight_trajectories_by_worker[dp]` (computed locally by the pipeline coordinator). Stop when:
	         `(max(load) - min(load)) / step_target_trajectories <= 0.05`.
	       - Note: `queued_trajectories_by_worker` and `inflight_trajectories_by_worker` are local-only numbers used for rebalance decisions. `report_progress(...)` stays pipeline-level (sum across all workers).

Important distinction:
- **Shrink** must migrate/cancel in-flight work on `P` (mandatory), because `P` may be offloaded/stopped and lose state.
- **Expand** can choose how aggressive it is. We default to the stronger behavior above, but it should be configurable (e.g., limit how many in-flight turns can be aborted per rebalance cycle).

Note:
- `SUSPEND_RESUME`-style mechanisms are **not valid for shrink**, because shrink implies the worker may be offloaded/deactivated and lose its local state (e.g., KV cache).
  Therefore: shrink must rely on a migration mechanism (`REQUEST_RETRY`). If shrink happens at a safe boundary (no in-flight work), the migration path is a no-op but still the required contract.

**Staleness rule of thumb**
- `QUIESCE-by-drain` and `QUIESCE-by-abort` target strict consistency (new admission never starts with stale weights).
- `BATCH` yields fixed, batch-scoped staleness (e.g., one-step-off).
- `INFLIGHT` yields bounded overlap; correctness requires output version tagging and a clear “which version is active” cutover rule.

---

## 2.1 Decisions (Answering open questions)

These are the chosen design decisions for the protocol:

1) Who computes per-worker load?
- Decision: the **pipeline coordinator** computes `load[dp]` because it dispatches requests and tracks which request is on which dp worker.

2) What counts as abort ACK (safe to retry elsewhere)?
- Decision: ACK means the request finishes with `finish_reason/stop_reason == "abort"`.

3) What if abort ACK does not arrive before timeout?
- Decision: **crash the pipeline** (fail fast and loudly). Do not offload/stop that worker. Treat it as unsafe to proceed.

4) What is `load[dp]` for the 5% rule?
- Decision: `load[dp] = queued_trajectories_by_worker[dp] + inflight_trajectories_by_worker[dp]` (trajectory counts).

5) What is the expand rebalance target?
- Decision: balance across all active dp workers and stop when the gap is within 5%.

6) Retry limit policy
- Decision: cap **engine error retries** per `(trajectory_id, turn_id)` (default max error retries: 3, configurable).
- Decision: do not cap preemption retries (abort due to shrink/expand rebalance), but add a high “too many preempts” threshold to fail loudly (Phase 1).

7) Preempted too many times (Phase 1 vs backlog)
- Phase 1 decision: if a turn is preempted “too many” times (high threshold, configurable), **crash the pipeline** (fail fast and loudly). This makes the bug visible.
- Backlog: add a “wait for drain” fallback:
  - Expand rebalance: stop aborting that turn; let it finish.
  - Shrink: stop sending new work to the shrinking worker(s), wait for that turn to finish, then offload/flush and release GPUs.

8) `report_progress(...)` units
- Decision: `report_progress` must use **trajectory counts**. The pipeline coordinator converts internal units (groups/prompt-groups) into trajectories before reporting.

9) “Superseded activation requests” (v2 then v3)
- Decision: coordinator chooses the version; scheduler only chooses timing.
- Decision: require an `activation_epoch` in activation/sync requests. Scheduler must skip any older request when a newer epoch exists (“newest wins”).

10) Subset lifecycle changes (Phase 1 style)
- Decision: use the smallest code change that works: add optional `worker_indices` / `indices` arguments to existing APIs.

11) Placement changes (Phase 1 style)
- Decision: fixed placement is enough for Phase 1. We do not require “resize Ray placement groups at runtime”.

12) Drain granularity selection (Turn vs Trajectory-Group)
- Decision: **Match the framework's native execution unit.** Do not force a granularity that conflicts with the framework's internal batching.
  - **Turn/request drain**: Use if the framework schedules individual requests/turns (e.g., Miles, SkyRL-train).
  - **Trajectory-group drain**: Use if the framework schedules opaque groups/threads (e.g., NeMo-RL prompt groups, ROLL rollout groups).

## 3) Protocol (Shared)

### 3.1 Responsibilities (Coordinator vs Scheduler)

- **Coordinator chooses** `active_checkpoint_version` locally (which version new admission should target).
- **Scheduler controls timing** of when to actually perform a weight sync, because it must coordinate with shrink/expand (time-sharing) and global GPU reallocation.
- The coordinator triggers a sync only when it must force weights to be present on a worker set (e.g., after expand, before opening admission under `QUIESCE-by-drain`/`QUIESCE-by-abort`, or at a batch boundary).

### 3.2 Logical RPCs

Allocation and resize:
- `request_gpus(cluster_id, role, priority, ...)` / `release_gpus(cluster_id, role, ...)`
- `request_release_gpus(cluster_id, role, ...)` (scheduler-driven generation release)
- `expand_workers(worker_indices=...)` / `shrink_workers(worker_indices=...)` (scheduler → coordinator DP-subset control; blocking RPC that returns only after completion)
- `notify_cluster_released(cluster_id, role, ...)` (adapter/proxy release ACK)

Checkpoint sync:
- `request_checkpoint_sync(pipeline_id, version, ...)` (coordinator → scheduler sync intent; scheduler may merge with shrink/expand)

Execution note:
- The actual in-place weight update and “sync-on-expand” are performed inside the pipeline coordinator (e.g., a `ParameterSyncService` / model update group). The scheduler only decides *when* and *which workers* should be updated and may run a global shrink barrier first.

Progress input:
- `report_progress(queued_trajectories, inflight_trajectories, percent_completed, oldest_unfinished_creation_ts, metrics=...)`

#### `report_progress(...)` semantics (SchedRL-standard)

SchedRL standardizes progress reporting across frameworks so the scheduler can make fair decisions.

**Units**
- All progress counters are in **trajectory units** (not turn/request units).
- Many frameworks internally use “groups” (one group contains multiple trajectories). The pipeline coordinator converts group counts into trajectory counts before calling `report_progress(...)` (or the adapter can do it if the coordinator cannot).

**Reported counts**
- `queued_trajectories`: trajectories that are queued but not started.
- `inflight_trajectories`: trajectories that are currently running (at least one turn/request is running or pending on a worker/engine).

**Cadence (2% bands)**
- We keep the 2% reporting cadence, but the denominator is the **per-step training batch size** (total samples trained per step, in trajectory units): `step_target_trajectories`.
- Mapping: ROLL `rollout_batch_size`; NeMo-RL `train_global_batch_size`; Miles `rollout_batch_size * n_samples_per_prompt`; SkyRL-train `train_batch_size * n_samples_per_prompt` (no single knob).
- `percent_completed >= 1.0` means the next train step’s batch is ready (enough trajectories are collected/qualified and available for training).
- `percent_completed` may be **greater than 1.0** if the pipeline over-collects. For event-driven 2% bands, the coordinator may cap at `min(percent_completed, 1.0)` to avoid spam.

**What fields mean**
- `percent_completed`: `collected_trajectories / step_target_trajectories`, where `collected_trajectories` is the number of trajectories that are complete and ready for training for the next step (e.g., buffered/qualified and not yet consumed by the trainer).
- `oldest_unfinished_creation_ts`: the oldest **trajectory enqueue time** among `unfinished` only (queued + in-flight).
  - For queued items, this is their enqueue time.
  - For in-flight items, use the same trajectory enqueue time (do not replace with worker start time).
- For retries of the current turn/request (`REQUEST_RETRY`), enqueue time is attached to the trajectory identity and is not reset per retry.

 


#### Scheduler policy (high-level, standardized)

SchedRL uses `report_progress(...)` primarily for **fairness, lease-expiry, and anti-thrashing**. It is not a global
priority score.

**Primary scheduling signals (global)**
- **Priority tier** (configured): higher tiers first (e.g., non-preemptible training clusters ahead of preemptible generation).
- **Update policy constraints**: do not schedule actions that would violate the pipeline’s configured `update_policy` safety rules.
- **Age/fairness**: `oldest_unfinished_creation_ts` is the FIFO tie-break (older unfinished work gets service first).
- **Throughput/rate** (optional): if available, prefer allocations that improve global utilization (without violating fairness).

**How `percent_completed` is used**
- **Anti-thrashing**: avoid repeatedly shrinking/expanding a pipeline’s generation cluster when it is actively making progress (band changes) or when it is near completion.
- **Lease-expiry guidance**: if a pipeline is not making progress (no band changes) and other higher-priority work needs GPUs, it is a better donor for shrink.

This keeps behavior stable for both finite backlogs and continuously-refilled pipelines, and avoids percent-based starvation.

**Priority tiers from multiple DAGs (standardization approach)**
- If the system defines multiple per-framework/per-pipeline dependency DAGs (e.g., different training recipes) and some nodes are shared, SchedRL should build a single
  **union DAG** by treating shared nodes as identical, and then compute priority tiers from a topological ordering of that union DAG.
- This is valid only if the union graph remains acyclic. If merging introduces a cycle (conflicting ordering constraints), the configuration must be rejected or the DAGs
  must be refactored (e.g., by splitting/renaming “shared” nodes or relaxing the shared-node mapping).

Checkpoint intent and sync timing:
- Coordinator tracks `active_checkpoint_version` locally (which version new admission should target).
- Scheduler does **not** need to be informed on every local activation change.
- `request_checkpoint_sync(pipeline_id, version, ...)`
  - Forced operation (intent): the coordinator calls this only when it must **actually synchronize weights** (load/broadcast) for a specific version before it can safely proceed (source is the trainer-side CPU bucket cache by default).
  - Scheduler chooses a safe execution moment and the concrete worker set (may merge with shrink/expand and admission gates); if it cannot do so safely, it must fail fast.

### 3.3 Checkpoint State + Cache Contract

Coordinator tracks (per pipeline / generation):
- `latest_checkpoint_version` (informational)
- `active_checkpoint_version` (local target for new admission; may intentionally lag latest)
- `update_policy`

Trainer-side checkpoint cache service tracks:
- `checkpoint_cpu_cache_versions` (CPU bucket-list versions currently retained)

Coordinator also tracks:
- `worker_active_checkpoint_version[worker]` (what version each rollout worker is known to have loaded/activated)
- `in_use_checkpoint_versions` (versions referenced by unfinished work; used for safe cache GC)

Rollout workers may have local caches, but they are not relied on for correctness.

Correctness invariant:
- The coordinator controls admission; it must never dispatch new work to a worker unless `worker_active_checkpoint_version[worker]` matches the required version for that work.
  - For `QUIESCE-by-drain`/`QUIESCE-by-abort`: required version is `active_checkpoint_version`.
  - For `BATCH`: required version is the batch-pinned version chosen by the coordinator at the boundary.
  - For `INFLIGHT`: in-flight work may finish on older versions, so the coordinator must track which versions are still in use (`in_use_checkpoint_versions`) for safe cache retention/GC.

**Race: superseded sync requests (coalescing)**
- The coordinator may request sync of `v2` and then quickly request sync of `v3` before syncing `v2` completes.
- Scheduler/proxy behavior must be “last-writer-wins”: it must not reopen admission or mark cutover complete for an older sync that is already superseded by a newer sync request.

Trainer-side CPU cache retention (minimum):
- After each train step, the trainer-side cache service materializes the trained weights into CPU bucket-list form and updates `latest_checkpoint_version`.
- Keep `{active_checkpoint_version, latest_checkpoint_version}` in the trainer-side CPU bucket cache at minimum:
  - `active_checkpoint_version` is required for correctness (expand/resume must be able to re-sync it after shrink/offload drops worker-local weights).
  - `latest_checkpoint_version` is kept because it is expected to become active later (and is already produced every step).
  - Note: `active_checkpoint_version` and `latest_checkpoint_version` may be the same.
- Always keep any version in `in_use_checkpoint_versions` (as reported/pinned by the coordinator).
  - Example: under `QUIESCE-by-drain`, in-flight work may still be running on an older weight version, so that version must remain in the CPU cache until the drain completes.

**Race: trainer-side CPU cache GC**
- Cache eviction must be safe with respect to late expand/resume and in-flight sync.
- Default GC rule (safe):
  - Keep `{active_checkpoint_version, latest_checkpoint_version}`.
  - Keep any version currently being synced (an in-flight `request_checkpoint_sync` task).
  - Keep any version in `in_use_checkpoint_versions` (unfinished work pinned to it, as tracked by the coordinator).
  - Delete older versions only once they are not `{active, latest}`, not in-flight sync, and not in-use.

### 3.4 Elastic Rollout Workers (Shrink/Expand + Sync)

This section covers:
1) shrink/expand alone (time-sharing resize, weights unchanged),
2) model sync without resize (`INFLIGHT`/`QUIESCE` without shrink/expand),
3) merged resize + model sync.

#### 3.4.1 Shrink/Expand Alone (Weights Unchanged)

**Shrink (`old_S -> new_S`, `P = old_S − new_S`)**
1. Coordinator closes admission for `P`.
2. Migrate/cancel in-flight work on `P` according to `migration_policy` (mandatory for shrink), and wait until `inflight(P) == 0` (abort ACK or drain completion).
3. Offload/remove model weights from GPU for `P` (sleep/offload/stop) to free memory.
4. Return from `shrink_workers(...)` so the scheduler can reassign GPUs.

If you want to “pause and later resume on the same worker”, model it as **pause without shrink** (no deallocation/offload), not as a shrink.

Miles/SGLang note:
- Any “retract” that only re-queues work inside the local engine process is **not shrink-safe** (work would be lost when the engine is offloaded/stopped).
  For shrink, do not rely on local retract; use `REQUEST_RETRY` (cancel + re-issue) via a global dispatcher/queue.
- If offload/sync waits on “engine queue empty” (flush/drain), admission must be closed at the router/dispatcher level first; otherwise new arrivals can keep the engine
  busy indefinitely and starve offload/weight sync.

**Expand (`old_S -> new_S`, `A = new_S − old_S`)**
1. Sync+activate `active_checkpoint_version` on `A` from the trainer-side CPU bucket cache service (default).
2. Coordinator opens admission for `A` after sync completes.
3. Optional (enabled by default): rebalance queued/not-started work onto `A` (queue/tail rebalance).
4. Return from `expand_workers(...)` so the scheduler can proceed.

Expand safety rule:
- Before dispatching an expand that will trigger a weight sync, the scheduler must re-validate the current `active_checkpoint_version` and expand using that version
  (not a stale queued intent).

#### 3.4.2 Model Sync Without Resize

This is the “model sync only” case: the scheduler does not change allocation (`old_S == desired_S == S`), but the pipeline needs to move the existing workers to a new active checkpoint version.

**Pause without shrink (no deallocation)**
- Coordinator closes admission (no new starts) but keeps the workers allocated and their local state intact.

**In-place sync (no allocation change) — INFLIGHT**
- Goal: ensure no new request/turn starts on `v_old` after switching `active_checkpoint_version := v_new`, while allowing already-running work to finish (mixing is allowed).
- Steps:
  1. Coordinator closes admission for **new starts** on `S`.
  2. Coordinator flips `active_checkpoint_version := v_new`.
  3. Coordinator (via `ParameterSyncService` / model update group) performs an in-place weight sync to `v_new` on `S` (scheduled by the scheduler from the coordinator’s `request_checkpoint_sync(...)` intent).
  4. Coordinator reopens admission on `S`.
- Optional (stronger semantics): abort the current turn and redo it on `v_new` (still INFLIGHT across turns, but avoids continuing a partially-generated turn on `v_old`).

**In-place sync (no allocation change) — QUIESCE-by-drain / QUIESCE-by-abort**
- Goal: ensure no in-flight work remains before syncing workers to `v_new` (strict boundary).
- Steps:
  1. Coordinator closes admission on `S`.
  2. Reach `inflight(S) == 0`:
     - `QUIESCE-by-drain`: wait for in-flight work to finish naturally.
     - `QUIESCE-by-abort`: abort in-flight work, wait abort ACK, and rely on `REQUEST_RETRY` to redo aborted turns later under `v_new`.
  3. Coordinator flips `active_checkpoint_version := v_new`.
  4. Coordinator (via `ParameterSyncService` / model update group) performs an in-place weight sync to `v_new` on `S` (scheduled by the scheduler from the coordinator’s `request_checkpoint_sync(...)` intent).
  5. Coordinator reopens admission on `S`.

This section is also a special case of the merged scenario below when `desired_S == old_S`.

#### 3.4.3 Merged: Resize + Model Sync

**Two-phase scheduling: shrink (all pipelines) → sync+expand (all pipelines)**
- When time-sharing resize and activation (policy update) must be coordinated across many pipelines, the scheduler should run a two-phase barrier:
  1) **Shrink phase (global)**: issue `shrink_workers(...)` for each pipeline to free GPUs first.
  2) **Sync+expand phase (global)**: after all shrinks complete, execute the per-pipeline “sync+expand” actions (below) using the freed GPUs.

This keeps the benefit you want (expand can be fused with selective model update) without introducing a new RPC: “sync+expand” is a logical unit composed from `request_checkpoint_sync(...)` + `expand_workers(...)`.

**Per-pipeline merged steps (after the shrink phase completes)**
1. Scheduler decides the target active version `v_new` and the post-decision worker set `desired_S`.
2. Let `S_remain` be the workers that remain allocated after shrink (typically `S_remain = desired_S` after phase-1 shrinks).
3. Policy update + in-place sync (option B reuse):
   - If `update_policy` is `INFLIGHT` (no QUIESCE):
     1) Coordinator closes admission for **new starts** on `S_remain`.
     2) Coordinator flips `active_checkpoint_version := v_new`.
     3) Coordinator (via `ParameterSyncService` / model update group) performs an in-place weight sync to `v_new` on `S_remain` (scheduled by the scheduler from the coordinator’s `request_checkpoint_sync(...)` intent).
     4) Coordinator reopens admission on `S_remain`.
   - If `update_policy` is `QUIESCE-by-drain` or `QUIESCE-by-abort`:
     - Scheduler should have already shrunk the pipeline to zero workers in the shrink phase (set `desired_S := ∅` for that pipeline in phase 1).
     - In that case, there is no in-place sync (no active rollout workers remain); the next step is expand-from-zero under `v_new`.
4. Expand if needed: compute `A = desired_S − S_remain`, then call `expand_workers(A)`.
   - `expand_workers(A)` must sync/load `v_new` from the trainer-side CPU cache service before admission opens on `A` (selective “sync-on-expand” handled by the coordinator’s `ParameterSyncService` / model update group).
5. Coordinator opens admission on `desired_S`.

### 3.5 Version Tagging (Backlog)

Optional (recommended for debugging/analysis):
- Tag every produced sample/trajectory with `generation_checkpoint_version`.
  - This can help when a logical “trajectory” spans multiple weight versions (e.g., `INFLIGHT`, or `QUIESCE-by-abort` where an aborted turn is retried later under a newer active version).

---

## 4) Validation / Implementation (Framework Mapping)

This section maps frameworks to protocol settings and highlights real integration points (details live in `design_doc/adaptation_*.md`).

### 4.1 Coverage Matrix (Async Modes and Use Cases)

The protocol covers a mode if it can define:
(1) an `update_policy` (how/when rollouts observe new weights),
(2) a `migration_policy` for shrink preemption (how in-flight work moves),
(3) a checkpoint alignment mechanism (pull+activate vs in-place),
and (4) a clear rule for how training consumes mixed-version data when overlap is allowed.

| Mode / Use case | What happens | `update_policy` | Allocation change required? | Preemption during rollout | Coverage notes |
|---|---|---|---|---|---|
| **On-policy (sync)** | Rollout then train; no overlap | `QUIESCE-by-drain` or `BATCH` | No | Yes (shrink/expand during rollout) | Trivial activation; resize uses `REQUEST_RETRY` (cancel+retry). If there is no in-flight work, the migration path is a no-op. |
| **Pipelined batch overlap (one/two-step-off)** | Train step N overlaps rollout for step N+1 | `BATCH` | No | Yes | Activation at batch boundary; training must define how it consumes any mixed-version data (if it happens). |
| **Buffered async generation (bounded staleness)** | Rollout continues while training consumes from buffer | `QUIESCE-by-drain` (typical) or `QUIESCE-by-abort` | No | Yes | Activation requires a QUIESCE boundary and strict “no admission with stale weights”. |
| **Always-on background rollout worker** | Background worker keeps producing rollouts; training drains | N/A | No | Hard | Not supported for SchedRL adaptation (example: Miles `third_party/miles/examples/fully_async`). Hard to time-share GPUs because there is no clean scheduler-controlled “stop point” and shrink timing becomes unpredictable. |
| **In-flight update (no stop)** | Update active rollout workers in-place | `INFLIGHT` | No | Yes (but merged with activation) | Scheduler may sync existing `S` without resize; merge with shrink decisions to avoid syncing preempted workers. |
| **Time-sharing resize (weights unchanged)** | Scheduler shrinks/expands `S` mid-rollout | N/A (activation unchanged) | Yes | Yes | Uses `migration_policy` on shrink; expand pulls `active` checkpoint if weights were offloaded. |

### 4.2 Per-framework Coverage (What Works vs What Requires Extensions)

| Framework | Covered by protocol | What is required in the codebase to fully realize it |
|---|---|---|
| **NeMo-RL** | `QUIESCE-by-drain` + `INFLIGHT` activation; resize time-sharing | Subset lifecycle + admission gating wiring (see `design_doc/adaptation_nemo_rl.md`); version tagging already exists as `(generation_weight_version, target_weight_version)` and should map to `generation_checkpoint_version` / `active_checkpoint_version`. |
| **ROLL (Agentic)** | `QUIESCE-by-abort` activation; abort+retry at turn boundary | Subset start/stop + routing remap (clear sticky mappings) so aborted turns retry on remaining/new DP ranks (see `design_doc/adaptation_roll.md`). |
| **Miles** | `BATCH` activation and step-based async (`train_async.py`) | Support `train.py` (sync) and `train_async.py` (one-step-ahead overlap) plus sync-by-interval (`update_weights_interval`). Do not use `third_party/miles/examples/fully_async`. Subset targeting (`indices=...`) for `RolloutManager.onload/offload`; implement `REQUEST_RETRY` by aborting subset engines and re-queueing work to the global data source (see `design_doc/adaptation_miles.md`). |
| **SkyRL-train** | `BATCH` (one-step-off) and `QUIESCE-by-abort` (fully-async) | Use existing SkyRL-train async entrypoints first (`third_party/SkyRL/skyrl-train/examples/async`, `third_party/SkyRL/skyrl-train/examples/fully_async`), then add subset lifecycle if/when SchedRL needs live shrink/expand (see `design_doc/adaptation_skyrl.md`). |

### 4.3 Doc vs Code Status (Reality Check)

This table is a **code reality check** (not an aspiration list). Each row is backed by concrete file pointers.

Legend: `present` / `partial` / `missing`.

| Framework | Subset lifecycle | Shrink migration (`REQUEST_RETRY`) | Expand rebalance (queued) | Admission close | Selective sync | Heartbeat (`report_progress`) | Activation epoch + cache GC | Tests |
|---|---|---|---|---|---|---|---|---|
| **ROLL** | `missing` (only cluster `start_server/stop_server`) — `third_party/ROLL/roll/pipeline/agentic/agentic_pipeline.py`, `third_party/ROLL/roll/distributed/strategy/vllm_strategy.py` | `partial` (abort+retry exists, but not subset-targeted) — abort is global `RequestScheduler.abort_request` (`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`); retry-on-abort exists in env loop (`third_party/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py`) | `missing` (no explicit “rebalance queued” mechanism; would require routing/mapping refresh) — sticky mapping is `src_rank2_dp_rank` (`third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py`) | `missing` (no `active_dp_ranks` gate in scheduler) — `third_party/ROLL/roll/distributed/scheduler/generate_scheduler.py` | `missing` (no `model_update(worker_indices=...)`) — `third_party/ROLL/roll/distributed/executor/model_update_group.py` | `missing` (only local progress bar today) — `third_party/ROLL/roll/distributed/scheduler/rollout_scheduler.py` | `missing` in code (protocol-only guidance today) — `design_doc/multi-pipeline-adaptation-plan_clean.md` | `missing` (no subset preempt/resume/selective-sync integration tests found) |
| **NeMo-RL** | `missing` (no subset worker-group helper) — `third_party/nemo-rl/nemo_rl/distributed/worker_groups.py` | `missing` (no abort API + no retry queue; failures are logged/dropped) — `_run_prompt_group_worker` catches exceptions (`third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`) | `partial` (round-robin DP leader exists; no explicit “queued rebalance” knob) — `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py` | `present` for “stop new starts” at refit boundary — `_refit_pause_cleared` (`third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`) | `partial` (in-flight weight update mode exists at refit boundary; subset-scoped selective sync is not implemented) — `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py` | `missing` (no `report_progress` in code) | `missing` in code (protocol-only guidance today) | `missing` (no subset shrink/expand integration tests found) |
| **Miles** | `missing` (cluster-wide `onload/offload`) — `third_party/miles/miles/ray/rollout.py` | `partial` (abort primitive exists; SchedRL needs subset-targeted abort + retry wiring in the main rollout loop, not only in examples) — abort helper in `third_party/miles/miles/rollout/sglang_rollout.py` | `partial` (global data buffer exists; but no explicit scheduler-driven rebalance API) — `third_party/miles/miles/rollout/data_source.py` | `missing` in router API (MilesRouter has no `/remove_worker`; but engine shutdown calls it under `use_miles_router`) — `third_party/miles/miles/router/router.py`, `third_party/miles/miles/backends/sglang_utils/sglang_engine.py` | `partial` (weight_version supported in engine update calls; no centralized CPU cache ownership/bucket staging defined in code) — `third_party/miles/miles/backends/sglang_utils/sglang_engine.py` | `missing` (no `report_progress` in code) | `missing` in code (protocol-only guidance today) | `missing` (no tests for shared-router offload / retry scenarios found) |
| **SkyRL-train** | `present` (configured by `generator.num_inference_engines`; no live subset API) — `third_party/SkyRL/skyrl-train/skyrl_train/config/ppo_base_config.yaml` | `partial` (abort/pause exists; subset-targeted shrink/expand is not designed yet) — `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` | `partial` (routing exists; scheduler-driven “rebalance queued” hook not present) — `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/utils.py` | `present` (pause blocks new submissions; used for in-flight sync) — `third_party/SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` | `present` (weight sync infra exists; backend supports nccl/cuda_ipc strategies) — `third_party/SkyRL/skyrl-train/skyrl_train/weight_sync/` | `missing` (no SchedRL `report_progress` emission) | `missing` (SchedRL activation epoch + trainer cache GC are protocol-level) | `missing` (no SchedRL integration tests) |

### 4.4 Progress Mapping (Per-framework)

This section makes the `report_progress(...)` contract implementable by mapping the three state sets to concrete places
in each framework. The scheduler consumes `queued_trajectories`, `inflight_trajectories`, `percent_completed`, and `oldest_unfinished_creation_ts`.

#### Shared definitions (trajectory units)

- **Trajectory**: one unit that the framework considers “one rollout item” for training (may internally include multiple samples per prompt).
- `enqueue_time`: timestamp for the trajectory identity; used for FIFO fairness and preserved across turn-level retries.

#### ROLL (Agentic)

- **Trajectory unit**: one env rollout trajectory (one item returned by the env loop).
- Internal “group” contains `group_size` trajectories.
- Report:
  - `queued_trajectories = queued_groups * group_size`
  - `inflight_trajectories = inflight_groups * group_size` (or count per-trajectory if available)
  - `step_target_trajectories = rollout_batch_size` (already in trajectory units in ROLL config)

#### NeMo-RL

- **Trajectory unit**: one generated sample/trajectory.
- Internal “prompt-group” contains `num_generations_per_prompt` trajectories.
- Report:
  - `queued_trajectories = queued_prompt_groups * num_generations_per_prompt`
  - `inflight_trajectories = inflight_prompt_groups * num_generations_per_prompt`
  - `step_target_trajectories = train_global_batch_size` (where `train_global_batch_size = num_prompt_groups_needed * num_generations_per_prompt`)

#### Miles

- **Trajectory unit**: one sample/trajectory.
- Internal “group” contains `n_samples_per_prompt` trajectories.
- Report:
  - `queued_trajectories = queued_groups * n_samples_per_prompt`
  - `inflight_trajectories = inflight_groups * n_samples_per_prompt`
  - `step_target_trajectories = rollout_batch_size * n_samples_per_prompt`

#### SkyRL-train (GSM8K baseline)

- **Trajectory unit**: one `TrajectoryID` (already one trajectory).
- Report:
  - `queued_trajectories`: queued `TrajectoryID`s
  - `inflight_trajectories`: running `TrajectoryID`s
  - `step_target_trajectories = train_batch_size * n_samples_per_prompt`

#### Qualification + consumption events (important)

Adapters must update the denominator window correctly:
- If a trajectory is filtered/unqualified and will be retried, keep it in `unfinished` (queued or in-flight).
- If a trajectory is filtered/unqualified and is dropped permanently, remove it from the denominator immediately.
- If a trajectory is qualified and buffered, it stays in the denominator until the trainer consumes it (or it is dropped before training).

### 4.5 Known Gaps (Explicitly Out of Scope / Backlog)

- `PARTIAL_ROLLOUT_MIGRATE` is a future feature (resume tokens / backlog items) and is not required for baseline correctness.
- Subset-level lifecycle (`indices=...`) is required for full time-sharing benefits; if missing, the protocol still works at cluster granularity but loses fine-grained sharing.
- Any mode that allows overlap (`INFLIGHT`, some `BATCH` pipelines) requires a clear “mixed-version consumption” rule in training and accurate `in_use_checkpoint_versions` tracking. Optional: `generation_checkpoint_version` tagging for debugging/analysis.
- Backlog: rLLM+VeRL agentic training integration is archived; it does not provide the needed async training modes (one-step-off, fully-async with staleness control, and elastic subset shrink/expand). If we revisit it, start from `design_doc/archive/adaptation_rllm.md`.
- Backlog: SkyAgent SWE + async training integration (reuse SkyRL-train async trainers with the SkyAgent generator). Start from `third_party/SkyRL/skyrl-agent/examples/run_skyrl/run_skyrl_swe.sh` and `design_doc/adaptation_skyrl.md`.

### 4.6 Critical Implementation Gaps (Must Fix Before Phase 0)

Framework-specific gaps are documented in each per-framework adaptation file:
- ROLL: `design_doc/adaptation_roll.md` (sticky routing, static comm plan, abort ACK, creation_ts)
- Miles: `design_doc/adaptation_miles.md` (missing `/remove_worker`, creation_ts)
- NeMo-RL: `design_doc/adaptation_nemo_rl.md` (two-phase admission, creation_ts)

**Cross-cutting gaps (apply to all frameworks)**:

| Gap | Issue | Fix Required |
|-----|-------|--------------|
| **No `in_use_checkpoint_versions`** | Cache GC safety mechanism exists only in plan | Implement version reference counting |
| **No thrashing prevention** | No `minimum_hold` or lease mechanism | Add minimum allocation hold time |


| Framework | Typical `update_policy` | Resize/migration baseline | Notes |
|----------|--------------------------|---------------------------|------|
| **NeMo-RL** | `INFLIGHT` when `async_engine && in_flight_weight_updates`, else `QUIESCE-by-drain` | `REQUEST_RETRY` (retry current request/turn for prompt group) | QUIESCE-by-drain path waits for pending **prompt-group rollout threads** at the refit boundary; INFLIGHT path updates weights while in-flight prompt-group rollouts continue (no global abort). Required: coordinator-provided vLLM `request_id` (not worker-generated UUID) + targeted abort/ACK + retry queue; reuse `prepare_for_refit` admission gating. |
| **ROLL (Agentic)** | `QUIESCE-by-abort` (with timeout) | abort + remap + retry (turn-level) | Boundary is implemented by stopping the generate server; vLLM STOP path aborts outstanding requests before model update (abort-to-zero semantics with a timeout). Abort exists (`GenerateRequestType.ABORT`); env loop retries on `ABORT` and resumes without stepping env state; subset start/stop + remap glue needed for time-sharing shrink/expand. |
| **Miles** | `BATCH` | `REQUEST_RETRY` (retry current turn; trajectory can continue if context is preserved) | Weight updates are triggered at rollout boundaries; `train_async.py` drains the “next rollout” future before `update_weights()` to avoid updating weights mid-generation (BATCH boundary). Required: coordinator-provided SGLang `rid` + targeted abort/ACK; global data buffer can re-queue work for retry; subset offload/onload + router admission-close are extensions. |
| **SkyRL-train** | `BATCH` (one-step-off) and `QUIESCE-by-abort` (fully-async with atomic pause/resume) | `REQUEST_RETRY` (mid-flight shrink required) | Fully-async uses inference-engine pause + abort-resume + sync + resume (transparently re-issuing aborted work). Strict consistency is preserved (requests must restart). For shrink, still need coordinator-provided vLLM `request_id` + targeted abort/ACK; subset shrink/expand requires active-engine-set control in the client/router. |

Validation checklist (shared):
- Activation without resize is supported (INFLIGHT case).
- Resize + activation merged plan is supported (avoid syncing workers that will be immediately preempted).
- Expand pulls checkpoints from trainer cache, not from dropped worker-local GPU weights.
- `report_progress` exists with the required cadence (batch start + progress-band).

---

## 5) Adapter Checklist (What to Implement)

Baseline requirements (all frameworks):
- Subset lifecycle: `expand_workers(worker_indices)` / `shrink_workers(worker_indices)` with fixed placement.
- Admission control: close/open admission for subsets and/or pipeline-wide pause.
- Checkpoint alignment:
  - sync from the trainer-side CPU bucket cache service (default)
  - activate to serving engine
  - in-place update support where available (INFLIGHT)
- Migration on shrink: implement at least `REQUEST_RETRY` (cancel + retry/re-issue). This typically restarts the **current request/turn** (safe boundary), not the entire multi-turn trajectory.
  - Validate the two-phase commit invariant: do not execute stateful tool/env effects unless a non-abort generation result is received (single-writer commit).
  - Required: coordinator-generated per-turn request id must be used as the rollout-engine request id (vLLM `request_id`, SGLang `rid`) so targeted abort+ACK+retry is possible.
- Progress reporting: report at **batch start** and whenever progress crosses a **2% band**, including the final **100% (ready)** signal (`percent_completed >= 1.0`).
  - Recommended: always emit an update when `percent_completed >= 1.0` for the current backlog window so the scheduler can release/reallocate promptly.
  - Required: if the denominator window changes (queued/in-flight/buffered sets change), emit immediately and rebase the 2% bands (see `report_progress(...)` semantics above).
- Release ACK: emit `notify_cluster_released` after normal release/shrink completes.

Optional/backlog features:
- `PARTIAL_ROLLOUT_MIGRATE` (resume tokens / backlog items to avoid restarting the current request/turn).
- Trainer-side cache service (always-on): materializes the latest trained checkpoint into CPU bucket-list form after each train step, so `request_checkpoint_sync(...)` can broadcast/load from CPU without disk/object-store fetch.
- Version tagging (debugging/analysis): tag samples/trajectories with `generation_checkpoint_version` (useful for `INFLIGHT` and `QUIESCE-by-abort` retry paths).
- Backlog (safety hardening): add idempotency keys for stateful env/tool commits (e.g., `trajectory_id + turn_id + action_id`) so retries after timeouts/crashes cannot double-apply side effects.

**Code-change inventory (minimum to satisfy elastic time-sharing)**
- ROLL: subset start/stop DP ranks + subset-targeted abort + clear sticky routing so aborts retry on remaining/new ranks.
- NeMo-RL: coordinator-provided vLLM `request_id` + subset wake/sleep in `RayWorkerGroup` + shrink-time targeted abort/ACK + retry queue for aborted prompt-groups.
- Miles: coordinator-provided SGLang `rid` + subset `onload/offload(indices=...)` + admission close for subset before offload/sync (router-side worker disable/remove) + targeted abort/ACK on subset engines + re-queue work for retry + **required** rollout trajectory progress accounting (queued/in-flight/buffered/trained-consumed) so `report_progress` can be computed from arrival vs consumption in async mode.

**Shared backlog (merged from `design_doc/archive/adaptation_review.md`)**
- Add a minimal scheduler client API/stub (`schedrl/`) if the scheduler is not purely external.
- Add heartbeat/progress hooks in each framework loop (batch start + 2% band).
- Add subset APIs consistently (`indices`/`worker_indices`) for lifecycle and selective sync (or accept cluster-granularity fallback).
- Add a small test/harness validating subset ops + heartbeat cadence.
- Enforce scheduler thrash controls (min lease / hysteresis) and tail-end policy (avoid expanding when remaining work is tiny).

## 6) Edge Cases, Races & Failure Handling

The protocol must handle these distributed system realisms:

### 6.1 Logical Conflict: Shrink vs Suspend
- **Issue**: `PARTIAL_ROLLOUT_SUSPEND_RESUME` relies on the worker *existing* to resume (e.g., local KV cache / in-process state).
- **Rule**: This policy is **INVALID** for Shrink/Preemption (where the worker is removed).
- **Enforcement**: Shrink MUST use `REQUEST_RETRY` (cancel + retry/re-issue). `SUSPEND_RESUME` is reserved for **Update-Without-Resize** or **Time-Sharing-Without-Release**.

### 6.2 Race: Superseded Activations
- **Scenario**: Coordinator requests `v2`, then quickly requests `v3` (before `v2` finishes syncing).
- **Rule**: Eventual Consistency. The Scheduler MAY skip `v2` if it hasn't started, or abort `v2` sync, and proceed directly to `v3`.
- **Invariant**: Workers never serve a version *newer* than what the Coordinator has requested, but skipping intermediate versions is allowed.

### 6.3 Race: Cache Garbage Collection
- **Scenario**: Coordinator moves active to `v11` and deletes `v10`. Scheduler (lagging) tries to Expand `v10`.
- **Rule**: Scheduler must check `active_checkpoint_version` at the *moment of dispatch*.
- **Cache Contract**: Trainer Cache should implement a small grace period or ref-counting ("is anyone transitioning to v10?") to prevent 404s during tight races.

### 6.4 Failure: Partial Subset Expansion
- **Scenario**: `expand_workers([1, 2])`. Worker 1 succeeds, Worker 2 fails (OOM/Network).
- **Rule**: Adapter must report `notify_allocation_failed(indices=[2], reason=...)`.
- **Handling**: Scheduler marks index 2 as bad/cooldown and tries expanding a different index (e.g., 3), or releases 1 and retries later. It does NOT treat the bundle as ready until full success.

### 6.5 SGLang/Miles `retract` Scope
- **Scenario**: Using `mode='retract'` during Shrink.
- **Risk**: If `retract` only re-queues to the *local engine memory*, and the engine is killed, requests are lost.
- **Rule**: Local-only retract is valid only for **pause without shrink** (no deallocation/offload).
- **Fix**: For Shrink, do not rely on `retract`. Use `REQUEST_RETRY` (cancel + re-issue) via a global dispatcher/queue (restart current request/turn is acceptable).
