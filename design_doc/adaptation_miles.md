# Miles Adaptation Plan

## 1. Overview
Miles is the Phase 4 target. Being a specialized prototype for SWE-Agent workloads, it already enforces explicit `onload`/`offload` patterns, making it the easiest integration candidate structurally, though it serves a narrower use case.
We keep the core training loop intact and introduce a **proxy layer** to intercept framework-specific operations. The pipeline coordinator calls the central scheduler **directly**; the proxy is unidirectional (wrapper only) and emits release ACKs.

## 1.2 Simple async + multi-turn example (planned): Retool

Miles has:
- multi-turn agent loops (examples), and
- an async training loop (`train_async.py`),
but they are not combined into one simple example today.

We will make one main reference “async + multi-turn” example based on Retool because it is usually the easiest multi-turn example (no extra services).

- Multi-turn Retool example: `third_party/miles/examples/retool/generate_with_retool.py`
- Async training loop to reuse: `third_party/miles/train_async.py`

Planned work (doc-level):
- Add one entrypoint/config that runs Retool multi-turn rollouts under the async loop.
- Keep it simple (no tool server). Use vLLM or SGLang as the rollout engine.

Rough estimate: 3–7 days (wire multi-turn generator into async loop + debug).

## 1.3 Phase 2 (Mini-SWE): async agent + tools

Goal: make the existing Mini-SWE tool-loop work in Miles run with the Miles async training loop.

What exists today:
- Tool-loop training example: `third_party/miles/examples/experimental/swe-agent/`
  - It uses a Mini-SWE agent server (via NeMo-Gym) and runs `train.py` today.

Phase 2 plan:
- Add an async variant of this example:
  - start from the same configs and custom rollout functions, but switch the driver to `train_async.py`.
- Keep the scheduler rules simple for safety:
  - stop new starts,
  - wait for in-flight tool loops to finish,
  - then offload/shrink engines.

Why we are conservative:
- Tool loops have real side effects (shell commands inside a sandbox).
- If we abort and retry at the wrong time, we can run the same tool action twice.

Backlog (safer later):
- Add idempotency keys for tool actions and a resume story (so retry does not repeat side effects).

## 1.1 Protocol Fit (Against `multi-pipeline-adaptation-plan_clean.md`)

This section reality-checks Miles against the shared protocol in `design_doc/multi-pipeline-adaptation-plan_clean.md`.

**Already present in the codebase**
- **Global router / dispatcher exists**:
  - Miles can launch a router (`third_party/miles/miles/router/router.py`, also launched from `third_party/miles/miles/ray/rollout.py`).
  - SGLang engines register with the router (`third_party/miles/miles/backends/sglang_utils/sglang_engine.py` uses `/add_worker`).
  - **Reality check**: MilesRouter currently exposes `/add_worker` and `/list_workers` but does **not** implement `/remove_worker`. Meanwhile `SGLangEngine.shutdown()` calls `/remove_worker` when `args.use_miles_router` (or older sglang-router versions). This mismatch must be resolved for admission-close and clean shrink/offload.
  - This matches the shared protocol’s requirement that shrink-time retries must go back to a global dispatcher, not local engine memory.
- **Engine-side memory lifecycle primitives**:
  - SGLang server supports `release_memory_occupation` / `resume_memory_occupation(tags=[...])` (`third_party/miles/miles/backends/sglang_utils/sglang_engine.py`), which maps to “offload/remove weights from GPU” vs “resume weights/kv_cache”.
- **Abort endpoint for in-flight work**:
  - Miles can abort requests across workers via router-discovered worker URLs (`third_party/miles/miles/rollout/sglang_rollout.py` `abort()` calls `/abort_request` on workers).

**Gaps / required extensions for elastic shrink/expand**
- **Subset lifecycle (indices)**:
  - `RolloutManager.onload/offload` are currently cluster-wide (`third_party/miles/miles/ray/rollout.py`); shared protocol needs subset `indices=...` to implement `expand_workers` / `shrink_workers`.
  - Engine registration exists (`/add_worker`), and `SGLangEngine` has a shutdown path that *attempts* to unregister via `/remove_worker`, but MilesRouter does not currently implement `/remove_worker`. For true subset admission-close, either add router-side disable/remove (recommended) or implement an alternative admission gating mechanism.
  - The missing part is plumbing indices through `RolloutManager` and mapping indices → engine handles, plus router-side admission gating for `P`.
- **Shrink-time migration semantics**:
  - Shrink implies the engine may be stopped/offloaded; any “retract” that only re-queues locally would lose work.
  - For SchedRL mid-flight shrink, we must implement `migration_policy=REQUEST_RETRY` using existing Miles primitives:
    - **Cancel in-flight (targeted)** on the shrinking subset using SGLang `/abort_request` by `rid` (per-engine, not global).
    - **Abort ACK (required)**: the coordinator must wait until the aborted requests return with stop_reason/finish_reason == `abort` before offloading those engines.
    - **Retry the current turn** on remaining engines by re-issuing the same turn with preserved token/history state (token-in/token-out), not “restart the whole trajectory”.
  - This is feasible because Miles already has:
    - a global data source buffer (`RolloutDataSourceWithBuffer`),
    - a per-worker abort endpoint (`/abort_request` supports `rid` in SGLang), and
    - a step-based training loop (`train.py` / `train_async.py`) that can be extended to re-enqueue aborted work back into that global data source.
  - Not supported (for SchedRL adaptation): the “persistent background worker” rollout mode in `third_party/miles/examples/fully_async`.
    - Reason: it is not driven by the training loop, and it does not provide a clean scheduler-controlled stop point for GPU time-sharing shrink.
  - **Required extension (hard requirement)**: coordinator-generated deterministic request id.
    - The coordinator must set SGLang `rid = f"{trajectory_id}:{turn_id}:{attempt}"` for every turn request.
    - This requires a stable `trajectory_id` (same across retries) and a stable `turn_id` within that trajectory.
    - Miles’ generate wrapper must pass `rid` through the router/engine so abort can target it later.
- **Admission control before offload / weight sync (starvation risk)**:
  - `RolloutManager.offload()` calls `SGLangEngine.release_memory_occupation()`, which first calls `flush_cache()` that polls `/flush_cache` until the engine reports an empty queue (`third_party/miles/miles/ray/rollout.py`, `third_party/miles/miles/backends/sglang_utils/sglang_engine.py`).
  - Today this does **not** close admission at the router/dispatcher level. If requests keep being routed to that engine while `flush_cache()` is polling, the queue may never drain (or time out), delaying offload and any update that depends on it.
  - Required: the adapter must close admission for the target subset before offload/sync (e.g., remove/disable those workers from routing, or otherwise ensure no new requests are routed to them), then abort/drain, then offload.
  - **Required code change**: add a router-side worker disable/remove API (or equivalent gating) so subset shrink can stop new admissions reliably, and/or change `SGLangEngine.shutdown()` to not depend on an unsupported router endpoint.
- **Checkpoint/weight version tagging**:
  - SGLang supports `weight_version` in weight update calls (e.g. `update_weights_from_tensor(..., weight_version=...)` in `third_party/miles/miles/backends/sglang_utils/sglang_engine.py`).
  - The adapter must propagate `active_checkpoint_version` into rollout outputs (e.g., sample metadata) as `generation_checkpoint_version`.
- **Progress reporting for the central scheduler**:
  - The shared scheduler needs heartbeats based on **trajectory counts**: `queued_trajectories`, `inflight_trajectories`, `percent_completed`, and `oldest_unfinished_creation_ts`.
  - Miles often reasons in “groups” (one group contains multiple trajectories: `n_samples_per_prompt`). The adapter must convert group counts into trajectory counts before reporting.
  - Readiness rule: when `percent_completed >= 1.0`, the next train step’s batch is ready.

**Model update boundary (make this explicit)**
- **Miles assumes weight sync happens at safe boundaries** (between rollout batches / when the rollout engines are quiesced), not as a “hot swap” in the middle of multi-turn interactions.
  - `third_party/miles/train.py` does `generate → (optional offload) → train → onload_weights → actor_model.update_weights → onload_kv` (update happens after generation finishes).
  - `third_party/miles/train_async.py` explicitly syncs generation before updating weights: “sync generate before update weights to prevent update weight in the middle of generation”.
- **What this means for multi-turn rollouts**:
  - The multi-turn reference implementations treat `finish_reason=abort` as terminal and return `Sample.Status.ABORTED` immediately (e.g. `third_party/miles/examples/geo3k_vlm_multi_turn/rollout.py`, `third_party/miles/examples/retool/generate_with_retool.py`, `third_party/miles/examples/search-r1/generate_with_search.py`, `third_party/miles/examples/tau-bench/trainable_agents.py`).
  - There is no generic “pause / update / resume the same trajectory” mechanism in these examples. If a model update happens mid-trajectory and triggers abort, the trajectory is dropped (or retried from scratch by higher-level logic).
  - `partial_rollout`-style continuation is not available when using the Miles router today (`third_party/miles/miles/router/middleware_hub/radix_tree_middleware.py` asserts `not args.partial_rollout`), so “resume after abort” is not a baseline capability.
  - For SchedRL time-sharing shrink/expand, we only require turn-level abort+retry. We do not need a full multi-turn “resume from inside a turn” feature.

**Concrete file refs & immediate actions**
- Files: `third_party/miles/miles/ray/rollout.py`, `third_party/miles/miles/backends/sglang_utils/sglang_engine.py`, `third_party/miles/miles/rollout/sglang_rollout.py`.
- Actions:
  - Update `RolloutManager.onload(offload)` to accept `worker_indices` and map indices to engine handles so the scheduler can onload/offload a DP subset.
  - Ensure `onload_weights()` propagates a `generation_checkpoint_version` into sample metadata so trainer and scheduler can reconcile staleness.
    - Simple rule (debugging-friendly):
      - record `generation_checkpoint_version` when the first turn of a trajectory is submitted, and
      - record it again when the last turn finishes.
  - Add heartbeat hooks at the start of each rollout batch (`RolloutManager.generate`) to report:
    - `queued_trajectories`, `inflight_trajectories` (trajectory units),
    - `percent_completed = collected_trajectories / (rollout_batch_size * n_samples_per_prompt)`,
    - `oldest_unfinished_creation_ts`.

**Recommended baseline mapping**
- `update_policy = BATCH` (Miles generation is batch/round scoped; activation + weight sync at batch boundaries).
- `migration_policy = REQUEST_RETRY` (required for time-sharing): abort in-flight work on shrinking subset (by `rid`), wait ACK, then retry the same turn on remaining engines.
- `expand_rebalance_policy = REBALANCE_QUEUED` (enabled by default): after expand, preferentially route new work to the newly activated workers and allow queued/not-started items to flow to them.

**Baseline validation (required)**
- Validate the `REQUEST_RETRY` safety invariant: do not execute stateful env/tool side effects unless a non-abort generation result is received (single-writer commit). In Miles, this means the multi-turn env/tool execution must only advance state after the current turn’s generation is confirmed non-abort.

**Concise actionable items (merged from `design_doc/archive/adaptation_review.md`)**
- Add `indices=...` filtering for `RolloutManager.onload/offload` (and ensure remote calls accept indices) to enable DP-granularity.
- Wire standardized progress heartbeats (`report_progress`) at batch start and on 2% bands from the rollout loop.

## 2. Existing Code Integration Points (Pre-Adaptation)

### 2.1 Training Entry Point
*   **File**: `third_party/miles/train.py`
*   **Hook**: `actor_model.async_train()` loop.

### 2.2 Generation Entry Point
*   **File**: `third_party/miles/ray/rollout.py`
*   **Method**: `RolloutManager.generate()`
*   **Hook**: `rollout_manager.onload()` / `offload()` to manage engine lifecycle.

### 2.3 Weight Sync (Pre-Adaptation)
*   **Mechanism**: `actor_model.update_weights()` + `rollout_manager.onload_weights()`
*   **Hook**: Explicit calls in the training loop; not sync-on-resume.

## 3. Architecture Mapping

### 3.1 Component Mapping
| Design Doc Concept | Miles Implementation | Key Classes |
|-------------------|----------------------|-------------|
| **Policy Training** | `RayTrainGroup` | `actor_group.py` |
| **Generation/Rollout** | `RolloutManager` | `rollout.py` |
| **Weight Sync** | Explicit Broadcast (`update_weights`) | `actor_group.py` |
| **Cluster Abstraction** | `create_placement_groups` | `placement_group.py` |
| **Worker Management** | `RayTrainGroup` / `RolloutManager` | - |

### 3.2 Lifecycle Operations Mapping (Post-Adaptation)

**Terminology Distinction:**
- **Cluster-Level Operations**: Enable or disable the entire cluster (all workers). These are the existing APIs.
- **Subset-Level Operations (Required)**: Activate or deactivate specific DP workers. These require the extensions described in Section 5.1.

| Design Doc Verb | Miles Implementation | Method / Action |
|-----------------|----------------------|-----------------|
| **expand (cluster)** | `RolloutManager.onload()` | Wakes ALL engines |
| **shrink (cluster)** | `RolloutManager.offload()` | Sleeps ALL engines |
| **expand (subset)** | `RolloutManager.onload(indices=...)` (requires extension) | Wakes specific engines |
| **shrink (subset)** | `RolloutManager.offload(indices=...)` (requires extension) | Sleeps specific engines |
| **offload (DP)** | `release_memory_occupation` | Destroys weights |
| **load/backload** | `resume_memory_occupation` + Sync | Reloads weights |
| **sync (weights)** | `onload_weights()` | Loads from training to rollout |
| **broadcast** | `RayTrainGroup.update_weights()` | Rank 0 broadcast |

### 3.3 Progress/Heartbeat Mapping
| Metric | Miles Implementation |
|--------|----------------------|
| **queued_trajectories** | `queued_groups * n_samples_per_prompt` |
| **inflight_trajectories** | `inflight_groups * n_samples_per_prompt` |
| **percent_completed** | `collected_trajectories / (rollout_batch_size * n_samples_per_prompt)` (`percent_completed >= 1.0` means next train step batch is ready) |
| **oldest_unfinished** | **Required**: timestamp of the oldest unfinished group (queued or in-flight); used for scheduler tie-breaks. |

### 3.4 Preemption & Release Protocol (Post-Adaptation)
| Protocol | Miles Implementation |
|----------|----------------------|
| **request_gpus (train)** | Coordinator calls central scheduler; `train_group.offload()` gates compute |
| **release_gpus (train)** | Coordinator calls central scheduler; `train_group.offload()` |
| **request_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `onload(indices)` via proxy |
| **release_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `offload(indices)` via proxy |
| **preempt gen** | admission-close (router remove) + abort on shrinking subset + re-queue affected work for retry on remaining workers (count/report in trajectories, not groups) |
| **resume gen** | Scheduler triggers `onload_weights()` then `onload(indices)` (sync-on-resume) |

## 4. Swe-agent Integration (`examples/experimental/swe-agent/`)

For the `swe-agent` example, integration involves modifying the `train.py` loop:

1.  **Initialization**: Initialize `MilesAdapter` with `rollout_manager` and `actor_model` (`RayTrainGroup`).
2.  **Before Rollout**:
    *   Call `adapter_gen.request_resources()` to activate generation cluster.
    *   `rollout_manager` is already "onloaded" by the adapter.
3.  **After Rollout**:
    *   Call `adapter_gen.release_resources()` to offload/sleep generation cluster.
4.  **Before Training**:
    *   Call `adapter_train.request_resources()` to activate training cluster.
5.  **After Training**:
    *   Call `adapter_train.release_resources()` to offload training cluster.

**Constraints**:
*   **External Gym**: `RolloutManager` manages inference engines, not the external Nemo-Gym container via HTTP.
*   **Async Generation**: `RolloutManager.generate.remote` is async but currently awaited immediately.

## 5. Required Extensions

### 5.1 DP-Granular Selective Execution
*   **Status**: **Structurally Ready, Needs Parameter**.
*   **Analysis**: `onload`/`offload` methods exist but target all engines.
*   **Required Action**: Update `RayTrainGroup` and `RolloutManager` methods (`onload`, `offload`, `update_weights`) to accept `indices` and filter the internal list of handles.

### 5.2 Scheduler Progress Hooks
*   **Integration Point**: `rollout_manager.py`.
*   **Trigger**: When `results` are pushed to the training queue / buffer and at batch start (hook at the top of the `for rollout_id in range(...)` loop in `third_party/miles/train.py`, before `rollout_manager.generate(...)`).
*   **Action**: Inject `scheduler.report_progress(queued_trajectories, inflight_trajectories, percent_completed, oldest_unfinished_creation_ts)`.
*   **Required TODO (Miles async semantics)**: implement **arrival vs consumption** accounting so the denominator/remaining logic is correct even when generation and training overlap.
    - Track at least these monotonic counters for the current backlog window:
        - `groups_released_for_generation` (submitted/queued into the global data source)
        - `groups_arrived_from_generation` (qualified results buffered for training)
        - `groups_consumed_by_training` (removed from the buffer by the trainer)
        - `groups_dropped` (filtered/failed permanently; removed from denominator)
    - Compute trajectory counts from the **buffer state**, not from `rollout_id` alone:
        - `queued_trajectories = queued_groups * n_samples_per_prompt`
        - `inflight_trajectories = inflight_groups * n_samples_per_prompt`
        - `percent_completed = collected_trajectories / (rollout_batch_size * n_samples_per_prompt)`
    - Emit an immediate heartbeat whenever the denominator window changes (new groups released, groups dropped), then continue 2% band reporting within that window.

### 5.3 Native Request Migration During Stop (Framework-Specific)
*   **Miles native pattern (baseline)**: stop at batch boundaries; then sync weights; then start the next batch.
*   **What exists in code**:
    1.  `RolloutManager.offload()` offloads engine memory (`third_party/miles/miles/ray/rollout.py`).
    2.  There is an explicit abort primitive for in-flight work (`third_party/miles/miles/rollout/sglang_rollout.py` `abort()` calls `/abort_request`), but the common multi-turn examples treat abort as terminal (no automatic resume).
*   **SchedRL integration**:
    - Prefer `update_policy=BATCH`: coordinate so weight sync happens only when there is no in-flight generation on the target subset.
    - For time-sharing shrink, implement `REQUEST_RETRY`: remove subset from router admission, abort those engines, and re-queue the affected prompt/groups back to the global data source so they are regenerated on surviving workers.

### 5.4 Minimal Mid-Flight Shrink/Expand Checklist (Implementation-Ready)

Goal: implement `migration_policy=REQUEST_RETRY` for mid-flight shrink by reusing Miles’ existing global data source buffer and SGLang abort endpoint, with coordinator-provided per-turn `rid`.

**Shrink (mid-flight) — required**
- Subset lifecycle: extend `RolloutManager.onload/offload` to accept `worker_indices` and map indices → engine handles (`third_party/miles/miles/ray/rollout.py`).
- Admission control: remove/disable the shrinking subset from routing so no new requests are routed to it.
  - Required extension: MilesRouter needs a worker disable/remove API (today it only supports `/add_worker` + `/list_workers`).
- Deterministic request id (required): coordinator generates `rid = f"{trajectory_id}:{turn_id}:{attempt}"` and passes it into every `/generate` request.
- Cancel in-flight on subset: call SGLang `/abort_request` on just those engines by `rid` (not `abort_all` unless you are aborting the whole engine).
- Wait abort ACK (required): only proceed to `offload(indices=P)` after those rids return `abort`.
- Retry current turn: re-queue the affected work so surviving engines can retry the same turn using preserved token/history state (not restarting the whole multi-turn trajectory).
  - Error retry limit (safety): cap **engine error retries** per `(trajectory_id, turn_id)` (default: 3, configurable). If exceeded, drop the trajectory and report a metric.
    - Do **not** cap preemption retries (abort due to shrink/expand rebalance).
    - Backlog: track how many times a turn is preempted; if it is preempted too many times, stop aborting it and wait for it to finish for shrink (close admission, wait for it to finish, then offload/flush).

**Expand (default-enabled rebalance) — optional migration**
After `onload(indices=A)` + weight sync, rebalance is enabled by default (stronger expand):
- Route new groups to the expanded engines first.
- Reassign queued/not-started groups so they can run on `A` (global data source makes this natural).
- If still unbalanced, abort selected in-flight turns on overloaded engines and retry them on underloaded engines.
  - Require abort ACK before retry.
  - Stop condition (5% rule): the pipeline coordinator computes `load[dp] = queued_by_worker[dp] + inflight_by_worker[dp]` (trajectory counts) and stops when:
    `(max(load) - min(load)) / max(queued_trajectories + inflight_trajectories, 1) <= 0.05`.

## 6. Post-Adaptation Integration Overview
This section describes how reused Miles components and required extensions implement the protocol in `design_doc/multi-pipeline_roll_old_design.md`.

### 6.0 Proxy Layer (Framework Interception)
*   **Purpose**: Wrap `RolloutManager` to emit **release ACKs** and execute scheduler-initiated subset ops.
*   **Behavior**: The proxy forwards calls by default and only injects release notifications; it does **not** mediate scheduler decisions.
*   **Minimal intrusion**: The pipeline coordinator (`train.py`) calls the central scheduler directly at phase boundaries. Progress reporting must be added (Section 1.1 / 5.2 / 5.4) so the scheduler can make better time-sharing decisions.

### 6.1 Pipeline Coordinator ↔ Central Scheduler
*   **Who is the pipeline coordinator?** The `train.py` loop.
*   **Request/Release (Training)**: The coordinator calls the **central scheduler API** directly.
*   **Training offload granularity**: Cluster-wide only (no subset offload for training).
*   **Request/Release (Generation)**: The coordinator calls the **central scheduler API** directly; the scheduler triggers `RolloutManager.onload(indices)` via the proxy.
*   **Preempt/Resume**: The scheduler initiates preempt/resume; the proxy executes offload/onload, and sync runs before restarting the subset.
*   **Control split**: The scheduler initiates expand/shrink; the coordinator only requests/releases; the proxy executes expand/shrink on rollout workers.
*   **Sync timing change**: Post-adaptation, `onload_weights()` moves to **sync-on-resume** (right before generation resumes).
*   **Async training constraint**: For async training setups, rollout **must be stopped after each training step** before `onload_weights()` can proceed. The scheduler coordinates this stop before triggering sync-on-resume.
*   **Versioning**: Keep only the **latest** CPU weight cache (by `global_step`) for sync-on-resume.

### 6.2 Cluster Controller ↔ Scheduler (DP-Granular)
*   **Expand/shrink semantics**: `expand` resumes engines via `onload(indices)`, `shrink` preempts via `offload(indices)`.
*   **Selective sync**: `onload_weights()` must be scoped to the active subset group (explicit call).

### 6.3 Rollout Progress ↔ Scheduler Heartbeats
*   **Integration point**: `RolloutManager.generate` loop or callback.
*   **Tie-break**: `oldest_unfinished` timestamp from the task list.
*   **Release ACK**: After normal generation release, the proxy notifies the central scheduler (`notify_cluster_released`) before new preemption decisions are applied.

## 7. Implementation Steps (Phase 4)
1.  **Add Indices**: Update `onload`/`offload` signatures in `third_party/miles/ray/rollout.py` and `actor_group.py`.
2.  **Proxy Layer**: Implement a lightweight proxy to emit release ACKs and execute scheduler-initiated expand/shrink.
3.  **Inject Hooks**: Add batch-start + 2% band progress reporting to `generate_rollout` and integrate with `scheduler.report_progress`.
4.  **Verify**: Validate subset selective execution with `swe-agent`.

## 8. Arbitrary Placement & Selective Sync
Placement is arbitrary at config time via manual Ray placement groups, then fixed for the run; the scheduler only controls active subsets within that fixed placement.
Selective sync runs on resume for the active subset only.

## 9. Configuration Example
**Miles** (`miles_config.yaml`):
```yaml
train_backend: megatron
rollout_backend: sglang
schedrl:
  enabled: true
  scheduler_name: "CentralizedGPUScheduler"
  pipeline_id: "miles_pipeline_0"
```

## 10. Estimated Effort
*   **Complexity**: Medium (subset lifecycle + proxy).
*   **Size**: ~150–250 LOC.
