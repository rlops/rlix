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

This is the canonical shared protocol for multi-pipeline GPU sharing across
**NeMo-RL**, **ROLL**, **Miles**, and **SkyRL-train**.

For concrete per-framework code entrypoints and limitations, see:
- `design_doc/adaptation_nemo_rl.md`
- `design_doc/adaptation_roll.md`
- `design_doc/adaptation_miles.md`
- `design_doc/adaptation_skyrl.md`

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
- **Trainer checkpoint cache**: source-of-truth cache used by rollout workers to pull checkpoints during expand/resume (because shrink/offload can drop worker-local weights).

**Isolation assumption (important)**
- Each pipeline owns a dedicated **engine group** (cluster / worker group / set of rollout engines).
- That engine group is isolated and is **not shared** with other pipelines.
- Sharing happens only at the **GPU allocation** level (time-sharing via shrink/expand/offload), not by sending multiple pipelines’ requests into the same rollout engine/router.

---

## 2) Taxonomy (Mechanisms, Staleness, Framework Mapping)

SchedRL unifies cross-framework behavior using two orthogonal knobs:

**A) Model update / activation policy (`update_policy`)**
- `QUIESCE`: stop generation work before activation (close admission + cancel + wait until in-flight work finishes + possibly shrink-to-zero); safest.
- `BATCH`: activation only at batch boundaries (no mid-batch cutover); typical for pipelined trainers.
- `INFLIGHT`: in-place update on active workers without requiring stop/resume; in-flight requests may finish on old weights (requires version tagging and backend support).

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
- Abort confirmation is required: when a worker is preempted and in-flight work must be retried elsewhere, the adapter must not reissue the same request/turn until the
  original request has returned an explicit abort outcome (or equivalent “request is stopped” confirmation from the backend). HTTP 200 from an abort endpoint alone is not sufficient.
  - Adapters may send abort commands in batch (where supported) and then wait for abort outcomes/ACKs collectively before reissuing retries on the remaining active workers.
  - For vLLM/SGLang, ACK means: the request finishes with `finish_reason/stop_reason == "abort"`.
  - Timeout rule: if ACK does not arrive in time, fail loudly and do not continue shrink/expand (do not offload/stop a worker that may still be running work).

**C) Expand rebalance policy (`expand_rebalance_policy`)**
- Expand does not require migration for correctness (new workers can just take new work after syncing weights).
- For elasticity/time-sharing we use a **stronger expand rebalance by default** (faster load redistribution):
  1. Route new arrivals to the expanded workers first.
  2. Reassign queued/not-started work so it can run on the expanded workers.
  3. If load is still uneven, abort selected in-flight turns on overloaded workers and retry them on underloaded workers until the load is within tolerance.
     - This uses the same `REQUEST_RETRY` safety rules as shrink: deterministic per-turn request id + abort ACK required + two-phase commit for any stateful tool/env effects.
     - Stop condition: stop abort+retry when the load gap across active workers is within **5%**.
       - Definition: let `load[dp] = queued_trajectories_by_worker[dp] + inflight_trajectories_by_worker[dp]` (computed locally by the pipeline coordinator). Stop when:
         `(max(load) - min(load)) / max(queued_trajectories + inflight_trajectories, 1) <= 0.05`.
       - Note: `queued_trajectories_by_worker` and `inflight_trajectories_by_worker` are local-only numbers used for rebalance decisions. `report_progress(...)` stays pipeline-level (sum across all workers).

Important distinction:
- **Shrink** must migrate/cancel in-flight work on `P` (mandatory), because `P` may be offloaded/stopped and lose state.
- **Expand** can choose how aggressive it is. We default to the stronger behavior above, but it should be configurable (e.g., limit how many in-flight turns can be aborted per rebalance cycle).

Note:
- `SUSPEND_RESUME`-style mechanisms are **not valid for shrink**, because shrink implies the worker may be offloaded/deactivated and lose its local state (e.g., KV cache).
  Therefore: shrink must rely on a migration mechanism (`REQUEST_RETRY`). If shrink happens at a safe boundary (no in-flight work), the migration path is a no-op but still the required contract.

**Staleness rule of thumb**
- `QUIESCE` targets strict consistency (new admission never starts with stale weights).
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
- Decision: **fail shrink/expand** and do not offload/stop that worker. Treat it as a worker failure case, not a normal shrink case.

4) What is `load[dp]` for the 5% rule?
- Decision: `load[dp] = queued_trajectories_by_worker[dp] + inflight_trajectories_by_worker[dp]` (trajectory counts).

5) What is the expand rebalance target?
- Decision: balance across all active dp workers and stop when the gap is within 5%.

6) Retry limit policy
- Decision: cap **engine error retries** per `(trajectory_id, turn_id)` (default max error retries: 3, configurable). Do not cap preemption retries.

Backlog (safety improvement):
- Track how many times a turn has been preempted (aborted due to shrink/expand rebalance). If a turn is preempted “too many” times, stop aborting it and wait for it to finish.
  - Expand rebalance: do not abort that turn again; let it finish.
  - Shrink: delay `offload/flush`: stop sending new work to the shrinking worker(s), wait for that turn to finish, then offload/flush and release GPUs.

## 3) Protocol (Shared)

### 3.1 Responsibilities (Coordinator vs Scheduler)

- **Coordinator chooses** `active_checkpoint_version` and **notifies the scheduler** of the new desired active version.
- **Scheduler controls timing** of when to actually sync/activate that version on rollout workers, because it must coordinate with:
  - shrink/expand (time-sharing),
  - admission control,
  - global GPU reallocation.
- The coordinator controls timing **indirectly** by selecting the active version and (optionally) requesting an urgent sync; the scheduler still decides the safe execution moment.
- **Coordinator can force** immediate syncing by requesting the scheduler to do so (may block/fail if unsafe).

### 3.2 Logical RPCs

Allocation and resize:
- `request_gpus(cluster_id, role, priority, ...)` / `release_gpus(cluster_id, role, ...)`
- `request_release_gpus(cluster_id, role, ...)` (scheduler-driven generation release)
- `expand_workers(worker_indices=...)` / `shrink_workers(worker_indices=...)` (DP-subset control)
- `notify_allocation_applied(pipeline_id, role, active_worker_indices, ...)` (scheduler → coordinator completion signal)
- `notify_cluster_released(cluster_id, role, ...)` (adapter/proxy release ACK)

Progress input:
- `report_progress(queued_trajectories, inflight_trajectories, percent_remaining, oldest_unfinished_creation_ts, metrics=...)`

#### `report_progress(...)` semantics (SchedRL-standard)

SchedRL standardizes progress reporting across frameworks so the scheduler can make fair decisions.

**Units**
- All progress counters are in **trajectory units** (not turn/request units).
- Many frameworks internally use “groups” (one group contains multiple trajectories). Adapters must convert group counts into trajectory counts before reporting.

**Reported counts**
- `queued_trajectories`: trajectories that are queued but not started.
- `inflight_trajectories`: trajectories that are currently running (at least one turn/request is running or pending on a worker/engine).

**Cadence (2% bands)**
- We keep the 2% reporting cadence, but the denominator is the **rollout target per training step** (in trajectory units).
- This is always available as a config value in the target frameworks (ROLL rollout_batch_size, NeMo num_prompt_groups_needed, Miles rollout_batch_size, SkyRL policy_mini_batch_size).
- Because the denominator is “per step”, `percent_remaining` can be **greater than 100%** when the pipeline backlog is larger than one step.
- Band thresholds are applied on the **raw percent** (no cap at 100%). For example: 146%, 144%, 142%, ..., 100%, 98%, ...

**What fields mean**
- `percent_remaining`: `(queued_trajectories + inflight_trajectories) / step_target_trajectories`.
- `oldest_unfinished_creation_ts`: the oldest **trajectory enqueue time** among `unfinished` only (queued + in-flight).
  - For queued items, this is their enqueue time.
  - For in-flight items, use the same trajectory enqueue time (do not replace with worker start time).
- For retries of the current turn/request (`REQUEST_RETRY`), enqueue time is attached to the trajectory identity and is not reset per retry.

**Optional (recommended) metrics**
- `metrics.dropped_trajectories_total`: how many trajectories were permanently dropped (unqualified and not retried). Not needed for correctness, but useful for monitoring.
- `metrics.retried_trajectories_total`: how many trajectories/turns were retried. Not needed for correctness.

#### Scheduler policy (high-level, standardized)

SchedRL uses `report_progress(...)` primarily for **fairness, lease-expiry, and anti-thrashing**. It is not a global
priority score.

**Primary scheduling signals (global)**
- **Priority tier** (configured): higher tiers first (e.g., non-preemptible training clusters ahead of preemptible generation).
- **Staleness / activation constraints**: do not schedule actions that would violate the pipeline’s `update_policy` or checkpoint activation intent.
- **Age/fairness**: `oldest_unfinished_creation_ts` is the FIFO tie-break (older unfinished work gets service first).
- **Throughput/rate** (optional): if available, prefer allocations that improve global utilization (without violating fairness).

**How `percent_remaining` is used**
- **Anti-thrashing**: avoid repeatedly shrinking/expanding a pipeline’s generation cluster when it is actively making progress (band changes) or when it is near completion.
- **Lease-expiry guidance**: if a pipeline is not making progress (no band changes) and other higher-priority work needs GPUs, it is a better donor for shrink.

This keeps behavior stable for both finite backlogs and continuously-refilled pipelines, and avoids percent-based starvation.

**Priority tiers from multiple DAGs (standardization approach)**
- If the system defines multiple per-framework/per-pipeline dependency DAGs (e.g., different training recipes) and some nodes are shared, SchedRL should build a single
  **union DAG** by treating shared nodes as identical, and then compute priority tiers from a topological ordering of that union DAG.
- This is valid only if the union graph remains acyclic. If merging introduces a cycle (conflicting ordering constraints), the configuration must be rejected or the DAGs
  must be refactored (e.g., by splitting/renaming “shared” nodes or relaxing the shared-node mapping).

Checkpoint intent and sync timing:
- `request_checkpoint_activation(pipeline_id, active_version, active_checkpoint_ref, update_policy, ...)`
  - Declares *which* checkpoint should become active. This updates the scheduler’s target `active_checkpoint_version`, but does not necessarily perform sync immediately.
- `request_checkpoint_sync(pipeline_id, version, urgency={best_effort|force}, ...)`
  - Requests the scheduler to perform syncing/activation at the chosen time; `force` means “do it now or block/fail”.
- Optional: `notify_checkpoint_published(pipeline_id, version, checkpoint_ref, ...)` for observability/prefetch; not required for correctness.

### 3.3 Checkpoint State + Cache Contract

Scheduler tracks (per pipeline / generation):
- `latest_checkpoint_version` (informational)
- `active_checkpoint_version` (authoritative sync/activation target; admission must only open when workers match it)
- `active_checkpoint_ref` (how rollout workers pull the active checkpoint; default source is the trainer checkpoint cache)
- `update_policy`

Each rollout worker tracks:
- `worker_active_checkpoint_version`
- `cached_checkpoint_versions` (optional)

Correctness invariant:
- Never open admission on a worker unless `worker_active_checkpoint_version == active_checkpoint_version`.

**Race: superseded activations (coalescing)**
- The coordinator may request activation of `v2` and then quickly request activation of `v3` before syncing `v2` completes.
- Scheduler behavior must be “last-writer-wins”: coalesce pending work and always converge to the newest requested `active_checkpoint_version`.
  - Before starting an expensive sync/activate for version `v`, re-check that `v` is still the currently requested active version; if not, skip `v`.
  - If a sync for `v` is already running and a newer activation arrives, best-effort cancel; if cancellation is not supported, allow it to finish but treat its result
    as stale (do not reopen admission / do not mark cutover complete for `v`).
- Recommendation: include an `activation_epoch`/`request_id` in `request_checkpoint_activation(...)` so the scheduler can ignore stale completions safely.

Trainer cache retention (minimum):
- Keep `{active_checkpoint_version, latest_checkpoint_version}` in the trainer CPU checkpoint cache:
  - `active_checkpoint_version` is required for correctness (rollouts must be able to pull/sync it even after shrink/offload drops worker-local weights).
  - `latest_checkpoint_version` is kept as a prefetch optimization because it is expected to become active later.
  - Note: `active_checkpoint_version` and `latest_checkpoint_version` may be the same; in that case they refer to the same cached copy.
- In practice, the trainer checkpoint cache is populated after each `trainer.step()`; CPU-cached versions older than `{active, latest}` can be GC’d once the coordinator advances `active_checkpoint_version`
  *and* it is guaranteed that no in-flight work can still depend on older versions.

**Race: trainer cache GC**
- The coordinator should not “hard delete” old versions unilaterally. Cache eviction must be safe with respect to late scheduler actions (expand/resume).
- Use one of:
  - **Pin/lease/refcount**: scheduler pins `active_checkpoint_version` (and any in-flight activation target) until it ACKs completion; trainer cache only GC’s
    unpinned versions.
  - **Grace policy**: keep `{active, previous}` (and optionally `{latest}`) with `keep_last_N>=2` and/or TTL, so fast active transitions do not break late pulls.

### 3.4 Elastic Rollout Workers (Shrink/Expand + Sync)

**Shrink (`old_S -> new_S`, `P = old_S − new_S`)**
1. Close admission for `P`.
2. Migrate/cancel in-flight work on `P` according to `migration_policy` (mandatory for shrink).
3. Offload/remove model weights from GPU for `P` (sleep/offload/stop) to free memory.
4. ACK completion (`notify_allocation_applied` or blocking return) so GPUs can be reassigned.

If you want to “pause and later resume on the same worker”, model it as **pause without shrink** (no deallocation/offload), not as a shrink.

Miles/SGLang note:
- Any “retract” that only re-queues work inside the local engine process is **not shrink-safe** (work would be lost when the engine is offloaded/stopped).
  For shrink, do not rely on local retract; use `REQUEST_RETRY` (cancel + re-issue) via a global dispatcher/queue.
- If offload/sync waits on “engine queue empty” (flush/drain), admission must be closed at the router/dispatcher level first; otherwise new arrivals can keep the engine
  busy indefinitely and starve offload/weight sync.

**Pause without shrink (no deallocation)**
- Close admission on a subset or pipeline, but keep the workers allocated and their local state intact.
- Used for: `INFLIGHT` activation barriers, batch boundaries, or “stop admitting new work” without giving GPUs back.

**Expand (`old_S -> new_S`, `A = new_S − old_S`)**
1. Pull+activate `active_checkpoint_version` on `A` from `active_checkpoint_ref` (trainer checkpoint cache by default).
2. Open admission for `A`.
3. Optional (enabled by default): rebalance queued/not-started work onto `A` (queue/tail rebalance).

Expand safety rule:
- Before dispatching an expand that will pull checkpoints, the scheduler must re-validate the current `active_checkpoint_version` (and its ref) and expand using that
  version, not a stale queued activation intent.

**In-place sync (no allocation change)**
- If `old_S == new_S` but `active_checkpoint_version` changes (common under `INFLIGHT`), the scheduler may apply checkpoint alignment to the existing subset `S`
  without any new allocations.

**Merging activation with time-sharing resize**
- If activation and resize happen concurrently, treat it as one plan:
  - compute post-decision `new_S`,
  - migrate/offload `P`,
  - pull+activate for `A`,
  - in-place update for `new_S ∩ old_S`,
  - reopen admission on `new_S`.

### 3.5 Version Tagging

Required when `INFLIGHT` is used (and recommended otherwise):
- Every produced sample/trajectory is tagged with `generation_checkpoint_version`.

---

## 4) Validation / Implementation (Framework Mapping)

This section maps frameworks to protocol settings and highlights real integration points (details live in `design_doc/adaptation_*.md`).

### 4.1 Coverage Matrix (Async Modes and Use Cases)

The protocol covers a mode if it can define:
(1) an `update_policy` (how/when rollouts observe new weights),
(2) a `migration_policy` for shrink preemption (how in-flight work moves),
(3) a checkpoint alignment mechanism (pull+activate vs in-place),
and (4) version tagging rules when overlap is allowed.

| Mode / Use case | What happens | `update_policy` | Allocation change required? | Preemption during rollout | Coverage notes |
|---|---|---|---|---|---|
| **On-policy (sync)** | Rollout then train; no overlap | `QUIESCE` or `BATCH` | No | Yes (shrink/expand during rollout) | Trivial activation; resize uses `REQUEST_RETRY` (cancel+retry). If there is no in-flight work, the migration path is a no-op. |
| **Pipelined batch overlap (one/two-step-off)** | Train step N overlaps rollout for step N+1 | `BATCH` | No | Yes | Activation at batch boundary; requires tagging if training consumes mixed versions. |
| **Buffered async generation (bounded staleness)** | Rollout continues while training consumes from buffer | `QUIESCE` (typical) | No | Yes | Activation requires drain/abort boundary; strict “no admission with stale weights”. |
| **Fully async queue (unbounded/implicit staleness)** | Rollout tasks decoupled from train loop | `QUIESCE` (default) or framework-specific | No | Yes | Requires explicit admission control + version tagging; retry granularity may be task/trajectory. |
| **In-flight update (no stop)** | Update active rollout workers in-place | `INFLIGHT` | No | Yes (but merged with activation) | Scheduler may sync existing `S` without resize; merge with shrink decisions to avoid syncing preempted workers. |
| **Time-sharing resize (weights unchanged)** | Scheduler shrinks/expands `S` mid-rollout | N/A (activation unchanged) | Yes | Yes | Uses `migration_policy` on shrink; expand pulls `active` checkpoint if weights were offloaded. |

### 4.2 Per-framework Coverage (What Works vs What Requires Extensions)

| Framework | Covered by protocol | What is required in the codebase to fully realize it |
|---|---|---|
| **NeMo-RL** | `QUIESCE` + `INFLIGHT` activation; resize time-sharing | Subset lifecycle + admission gating wiring (see `design_doc/adaptation_nemo_rl.md`); version tagging already exists as `(generation_weight_version, target_weight_version)` and should map to `generation_checkpoint_version` / `active_checkpoint_version`. |
| **ROLL (Agentic)** | `QUIESCE` activation; abort+retry at turn boundary | Subset start/stop + routing remap (clear sticky mappings) so aborted turns retry on remaining/new DP ranks (see `design_doc/adaptation_roll.md`). |
| **Miles** | `BATCH` activation | Subset targeting (`indices=...`) for `RolloutManager.onload/offload`; implement `REQUEST_RETRY` by aborting subset engines and re-queueing prompt/groups to the global data source (see `design_doc/adaptation_miles.md`). |
| **SkyRL-train** | `BATCH` (one-step-off) and fully-async with staleness control | Use existing SkyRL-train async entrypoints first (`SkyRL/skyrl-train/examples/async`, `SkyRL/skyrl-train/examples/fully_async`), then add subset lifecycle if/when SchedRL needs live shrink/expand (see `design_doc/adaptation_skyrl.md`). |

### 4.3 Doc vs Code Status (Reality Check)

This table is a **code reality check** (not an aspiration list). Each row is backed by concrete file pointers.

Legend: `present` / `partial` / `missing`.

| Framework | Subset lifecycle | Shrink migration (`REQUEST_RETRY`) | Expand rebalance (queued) | Admission close | Selective sync | Heartbeat (`report_progress`) | Activation epoch + cache GC | Tests |
|---|---|---|---|---|---|---|---|---|
| **ROLL** | `missing` (only cluster `start_server/stop_server`) — `ROLL/roll/pipeline/agentic/agentic_pipeline.py`, `ROLL/roll/distributed/strategy/vllm_strategy.py` | `partial` (abort+retry exists, but not subset-targeted) — abort is global `RequestScheduler.abort_request` (`ROLL/roll/distributed/scheduler/generate_scheduler.py`); retry-on-abort exists in env loop (`ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py`) | `missing` (no explicit “rebalance queued” mechanism; would require routing/mapping refresh) — sticky mapping is `src_rank2_dp_rank` (`ROLL/roll/distributed/scheduler/generate_scheduler.py`) | `missing` (no `active_dp_ranks` gate in scheduler) — `ROLL/roll/distributed/scheduler/generate_scheduler.py` | `missing` (no `model_update(worker_indices=...)`) — `ROLL/roll/distributed/executor/model_update_group.py` | `missing` (only local progress bar today) — `ROLL/roll/distributed/scheduler/rollout_scheduler.py` | `missing` in code (protocol-only guidance today) — `design_doc/multi-pipeline-adaptation-plan_clean.md` | `missing` (no subset preempt/resume/selective-sync integration tests found) |
| **NeMo-RL** | `missing` (no subset worker-group helper) — `nemo-rl/nemo_rl/distributed/worker_groups.py` | `missing` (no abort API + no retry queue; failures are logged/dropped) — `_run_prompt_group_worker` catches exceptions (`nemo-rl/nemo_rl/algorithms/async_utils.py`) | `partial` (round-robin DP leader exists; no explicit “queued rebalance” knob) — `nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py` | `present` for “stop new starts” at refit boundary — `_refit_pause_cleared` (`nemo-rl/nemo_rl/algorithms/async_utils.py`) | `partial` (in-flight weight update mode exists at refit boundary; subset-scoped selective sync is not implemented) — `nemo-rl/nemo_rl/algorithms/async_utils.py` | `missing` (no `report_progress` in code) | `missing` in code (protocol-only guidance today) | `missing` (no subset shrink/expand integration tests found) |
| **Miles** | `missing` (cluster-wide `onload/offload`) — `miles/miles/ray/rollout.py` | `partial` (abort primitive exists; requeue-on-abort exists in fully-async example, but not a universal policy) — abort all workers in `miles/miles/rollout/sglang_rollout.py`; requeue aborted group example `miles/examples/fully_async/fully_async_rollout.py` | `partial` (global data buffer exists; but no explicit scheduler-driven rebalance API) — `miles/miles/rollout/data_source.py` | `missing` in router API (MilesRouter has no `/remove_worker`; but engine shutdown calls it under `use_miles_router`) — `miles/miles/router/router.py`, `miles/miles/backends/sglang_utils/sglang_engine.py` | `partial` (weight_version supported in engine update calls; no centralized CPU cache ownership/bucket staging defined in code) — `miles/miles/backends/sglang_utils/sglang_engine.py` | `missing` (no `report_progress` in code) | `missing` in code (protocol-only guidance today) | `missing` (no tests for shared-router offload / retry scenarios found) |
| **SkyRL-train** | `present` (configured by `generator.num_inference_engines`; no live subset API) — `SkyRL/skyrl-train/skyrl_train/config/ppo_base_config.yaml` | `partial` (abort/pause exists; subset-targeted shrink/expand is not designed yet) — `SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` | `partial` (routing exists; scheduler-driven “rebalance queued” hook not present) — `SkyRL/skyrl-train/skyrl_train/inference_engines/utils.py` | `present` (pause blocks new submissions; used for in-flight sync) — `SkyRL/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` | `present` (weight sync infra exists; backend supports nccl/cuda_ipc strategies) — `SkyRL/skyrl-train/skyrl_train/weight_sync/` | `missing` (no SchedRL `report_progress` emission) | `missing` (SchedRL activation epoch + trainer cache GC are protocol-level) | `missing` (no SchedRL integration tests) |

### 4.4 Progress Mapping (Per-framework)

This section makes the `report_progress(...)` contract implementable by mapping the three state sets to concrete places
in each framework. The scheduler consumes `queued_trajectories`, `inflight_trajectories`, `percent_remaining`, and `oldest_unfinished_creation_ts`.

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
  - `step_target_trajectories = num_prompt_groups_needed * num_generations_per_prompt`

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
  - `step_target_trajectories = policy_mini_batch_size`

#### Qualification + consumption events (important)

Adapters must update the denominator window correctly:
- If a trajectory is filtered/unqualified and will be retried, keep it in `unfinished` (queued or in-flight).
- If a trajectory is filtered/unqualified and is dropped permanently, remove it from the denominator immediately.
- If a trajectory is qualified and buffered, it stays in the denominator until the trainer consumes it (or it is dropped before training).

### 4.5 Known Gaps (Explicitly Out of Scope / Backlog)

- `PARTIAL_ROLLOUT_MIGRATE` is a future feature (resume tokens / backlog items) and is not required for baseline correctness.
- Subset-level lifecycle (`indices=...`) is required for full time-sharing benefits; if missing, the protocol still works at cluster granularity but loses fine-grained sharing.
- Any mode that allows overlap (`INFLIGHT`, some `BATCH` pipelines) requires reliable `generation_checkpoint_version` tagging and downstream handling during training.
- Backlog: rLLM+VeRL agentic training integration is archived; it does not provide the needed async training modes (one-step-off, fully-async with staleness control, and elastic subset shrink/expand). If we revisit it, start from `design_doc/archive/adaptation_rllm.md`.
- Backlog: SkyAgent SWE + async training integration (reuse SkyRL-train async trainers with the SkyAgent generator). Start from `SkyRL/skyrl-agent/examples/run_skyrl/run_skyrl_swe.sh` and `design_doc/adaptation_skyrl.md`.

| Framework | Typical `update_policy` | Resize/migration baseline | Notes |
|----------|--------------------------|---------------------------|------|
| **NeMo-RL** | `INFLIGHT` when `async_engine && in_flight_weight_updates`, else `QUIESCE` | `REQUEST_RETRY` (retry current request/turn for prompt group) | Required: coordinator-provided vLLM `request_id` (not worker-generated UUID) + targeted abort/ACK + retry queue; reuse `prepare_for_refit` admission gating. |
| **ROLL (Agentic)** | `QUIESCE` | abort + remap + retry (turn-level) | Abort exists (`GenerateRequestType.ABORT`); env loop retries on `ABORT` and resumes without stepping env state; subset start/stop + remap glue needed for time-sharing shrink/expand. |
| **Miles** | `BATCH` | `REQUEST_RETRY` (retry current turn; trajectory can continue if context is preserved) | Required: coordinator-provided SGLang `rid` + targeted abort/ACK; global data buffer can re-queue work for retry; subset offload/onload + router admission-close are extensions. |
| **SkyRL-train** | One-step-off: `BATCH`; Fully-async: framework-managed staleness | `REQUEST_RETRY` (mid-flight shrink required) | vLLM-only for fully-async; required: coordinator-provided vLLM `request_id` + targeted abort/ACK; subset shrink/expand requires active-engine-set control in the client/router. |

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
  - pull/sync from `active_checkpoint_ref` (trainer checkpoint cache)
  - activate to serving engine
  - in-place update support where available (INFLIGHT)
- Migration on shrink: implement at least `REQUEST_RETRY` (cancel + retry/re-issue). This typically restarts the **current request/turn** (safe boundary), not the entire multi-turn trajectory.
  - Validate the two-phase commit invariant: do not execute stateful tool/env effects unless a non-abort generation result is received (single-writer commit).
  - Required: coordinator-generated per-turn request id must be used as the rollout-engine request id (vLLM `request_id`, SGLang `rid`) so targeted abort+ACK+retry is possible.
- Version tagging (simple, for debugging):
  - Record `generation_checkpoint_version` when the **first turn** of a multi-turn trajectory is submitted (trajectory start version).
  - Record `generation_checkpoint_version` again when the **last turn** finishes (trajectory end version).
  - This shows if a trajectory ran across a weight update (start version != end version).
- Progress reporting: report at **batch start** and whenever progress crosses a **2% band**, including the final **0% (completion)** signal.
  - Recommended: always emit an update when `remaining_trajectories_total == 0` for the current backlog window so the scheduler can release/reallocate promptly.
  - Required: if the denominator window changes (queued/in-flight/buffered sets change), emit immediately and rebase the 2% bands (see `report_progress(...)` semantics above).
- Release ACK: emit `notify_cluster_released` after normal release/shrink completes.

Optional/backlog features:
- `PARTIAL_ROLLOUT_MIGRATE` (resume tokens / backlog items to avoid restarting the current request/turn).
- Prefetch/warm cache driven by `notify_checkpoint_published`.
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
