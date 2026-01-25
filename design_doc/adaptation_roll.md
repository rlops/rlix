# ROLL Adaptation Plan

## 1. Overview
ROLL is the Phase 1 target for SchedRL adaptation. This phase is **explicitly scoped to the Agentic pipeline** (`AgenticPipeline`) and uses its mature system-level orchestration layer (`GroupQueue`, `RolloutScheduler`) as the baseline for implementing the centralized scheduler.
This adaptation assumes **no Ray-level GPU reclaim** (placement groups are fixed). GPU sharing relies on **offload/load of training and inference states**, while SchedRL gates execution and still needs **subset-level DP worker activation** to improve scheduling efficiency.
We keep the core training loop intact and introduce a **proxy layer** to intercept framework-specific operations (e.g., `RolloutScheduler`, `RequestScheduler`). The pipeline coordinator calls the central scheduler **directly**; the proxy is unidirectional (wrapper only).

## 1.1 Protocol Fit (Against `multi-pipeline-adaptation-plan_clean.md`)

This section reality-checks ROLL (Agentic pipeline) against the shared protocol in `design_doc/multi-pipeline-adaptation-plan_clean.md`.

**Already present in the codebase**
- **Abort primitive (request cancellation)**: `GenerateRequestType.ABORT` is implemented and sent to inference workers (`ROLL/roll/distributed/scheduler/generate_scheduler.py`).
- **Generation pause/resume gate**: `RequestScheduler.suspend()` aborts in-flight requests, `resume()` re-opens scheduling (`ROLL/roll/distributed/scheduler/generate_scheduler.py`).
- **Coordinator-controlled update boundary (QUIESCE-like)**: in `ROLL/roll/pipeline/agentic/agentic_pipeline.py`, the loop calls `train_rollout_scheduler.suspend()` and stops the server before `model_update()`.
- **Retry safety (two-phase commit) is plausible**: on `ABORT`, the env loop does not advance env state (`GenerateStopReason.ABORT`), so a canceled turn can be retried without duplicate env side effects (`ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py`).

**Gaps / required extensions to support elastic shrink/expand**
- **Subset lifecycle (DP-granular start/stop)**:
  - Today `cluster.start_server()` / `cluster.stop_server()` are cluster-wide (`ROLL/roll/distributed/scheduler/generate_scheduler.py` and strategy impls).
  - The shared protocol needs subset `expand_workers(indices)` / `shrink_workers(indices)`; this requires adapter-level “start/stop only these dp ranks” or per-rank server control.
- **Subset-aware abort and reroute on shrink**:
  - Today `RequestScheduler.abort_request()` aborts *all* in-flight requests across all dp ranks.
  - For time-sharing shrink, we need “abort only requests running on `P`” and clear any sticky routing so the **next retry** goes to the remaining active ranks.
  - Practically, this means `RequestScheduler` needs:
    - an active-rank set (must never assign new requests to ranks not in `S`), and
    - a way to clear/refresh `src_rank → dp_rank` mapping when a dp rank is preempted, so retries remap correctly.
  - **Request identity requirement (SchedRL hard requirement)**:
    - The pipeline coordinator must generate a deterministic per-turn request id, e.g. `request_id = f"{trajectory_id}:{turn_id}:{attempt}"`.
    - This requires a stable `trajectory_id` (same across retries) and a stable `turn_id` within that trajectory.
    - The inference backends (vLLM / SGLang) accept string request ids, so this can be used as the engine-facing `request_id` in ROLL.
    - Note: ROLL currently has a comment about “globally increasing request ids” in `GenerateScheduler`. This appears to be a local assumption; we do not rely on monotonicity, only uniqueness.
- **Selective sync / subset NCCL groups**:
  - `model_update()` currently assumes full-cluster collectives. For subset resume, we need per-subset comm groups (or an equivalent mechanism) so only `S` is updated when resuming after a weight change.
- **Progress reporting for global scheduling**:
  - The shared scheduler needs heartbeats based on **trajectory counts** (not group counts): `queued_trajectories`, `inflight_trajectories`, `percent_remaining`, and `oldest_unfinished_creation_ts`.
  - ROLL internally batches trajectories into “groups” (`group_size` trajectories per group). The adapter must convert group counts to trajectory counts when reporting.
  - Cadence: report at batch start and whenever `percent_remaining` crosses a 2% band, where the denominator is the rollout target per training step (trajectory units).

**Concrete file refs & immediate actions**
- Files: `ROLL/roll/distributed/scheduler/generate_scheduler.py`, `ROLL/roll/distributed/executor/model_update_group.py`, `ROLL/roll/pipeline/agentic/agentic_pipeline.py`.
- Actions:
  - Implement `start_server_subset(worker_indices)` / `stop_server_subset(worker_indices)` helpers in the scheduler/cluster adapter to support subset expand/shrink.
  - Extend `ModelUpdateGroup.model_update(worker_indices=...)` to build per-subset comm-plans and add a `LatestWeightsCache` snapshot after training for sync-on-resume.
  - Emit heartbeat RPCs from `GroupQueue.put` and `RolloutScheduler.get_batch()` with:
    - `queued_trajectories`, `inflight_trajectories` (trajectory units),
    - `percent_remaining = (queued_trajectories + inflight_trajectories) / rollout_batch_size`,
    - `oldest_unfinished_creation_ts`.
  - Add simple version tagging for debugging:
    - record `generation_checkpoint_version` when the first turn of a trajectory is submitted, and
    - record it again when the last turn finishes.

**Recommended baseline mapping**
- `update_policy = QUIESCE` (keep the existing “suspend + stop + model_update + start” boundary).
- `migration_policy = REQUEST_RETRY` (ROLL-native “abort + retry”) for shrink preemption (turn-level retry; does not step env on abort).
- `expand_rebalance_policy = REBALANCE_QUEUED` (enabled by default): on expand, clear sticky routing so new turns can land on newly activated DP ranks.

**Baseline validation (required)**
- Validate the `REQUEST_RETRY` safety invariant: stateful env/tool side effects must only happen after a non-abort generation result is received (single-writer commit). If this is not true for a specific agent/env integration, mid-flight shrink is unsafe.

**Concise actionable items (merged from `design_doc/archive/adaptation_review.md`)**
- Implement `start_server_subset(worker_indices)` / `stop_server_subset(worker_indices)` (or equivalent adapter-level subset control).
- Add `ModelUpdateGroup.model_update(worker_indices=...)` / comm-plan filtering so selective sync-on-resume is possible.
- Wire `report_progress(queued_trajectories, inflight_trajectories, percent_remaining, oldest_unfinished_creation_ts)` from the rollout buffer enqueue point (e.g., `GroupQueue.put`/`GroupQueueManager`).

## 2. Existing Code Integration Points (Pre-Adaptation)

### 2.1 Training Entry Point
*   **File**: `ROLL/roll/pipeline/agentic/agentic_pipeline.py`
*   **Hook**: Inside `AgenticPipeline.run()`, before/after the training phase (`actor_train.train_step()` and any critic/reference work).

### 2.2 Generation Entry Point
*   **File**: `ROLL/roll/pipeline/agentic/agentic_pipeline.py`
*   **Method**: `AgenticPipeline.run()`
*   **Hook**: `actor_infer.start_server()` / `stop_server()` are invoked in the existing pipeline loop (before/after `RolloutScheduler.get_batch(...)`).
*   **Hook (rollout loop)**: `RolloutScheduler.get_batch(...)` resumes generation via `RequestScheduler.resume()` and drains completed trajectories from `GroupQueueManager.get_batch(...)`.

### 2.3 Weight Sync (Pre-Adaptation)
*   **Mechanism**: `model_update()` via `set_model_update_pair()`
*   **Hook (pre-adaptation)**: `model_update()` is invoked in the pipeline loop before rollout, typically after `RolloutScheduler.suspend()` and before `actor_infer.start_server(...)`; it is not sync-on-resume.
*   **File**: `roll/distributed/executor/model_update_group.py`

### 2.4 Running Requests & Trajectories (Pre-Adaptation)
*   **Running request handling**
    *   **Files**: `roll/distributed/scheduler/rollout_scheduler.py`, `roll/distributed/scheduler/generate_scheduler.py`
    *   **Hook**: `RolloutScheduler.suspend()` → `RequestScheduler.suspend()` → `RequestScheduler.abort_request()`
    *   **Behavior**:
        *   Aborts all in-flight inference requests on each active DP rank (sends `GenerateRequestType.ABORT`).
        *   Prevents new requests from being enqueued until `RequestScheduler.resume()` is called.
*   **Running trajectory handling**
    *   **Files**: `roll/pipeline/agentic/env_manager/traj_env_manager.py`, `roll/distributed/scheduler/rollout_scheduler.py`
    *   **Hook**: When a request is aborted, `PolicyProxy.generate(...)` returns `None`, and the env loop maps this to `GenerateStopReason.ABORT`.
    *   **Behavior**:
        *   The env loop does not advance the environment step on abort, so the trajectory remains logically paused until generation resumes.
        *   Completed trajectories are pushed into `GroupQueueManager` via `output_queue.put(...)` and later drained by `get_batch(...)`.

## 3. Architecture Mapping

### 3.1 Component Mapping
| Design Doc Concept | ROLL Implementation |
|-------------------|---------------------|
| **Policy Training** | `Cluster` with `MegatronTrainStrategy` / `DeepSpeedTrainStrategy` |
| **Generation/Rollout** | `Cluster` with `VllmStrategy` / `SglangStrategy` |
| **Rollout Orchestration** | `RolloutScheduler` + `GroupQueue` (async trajectory collection) |
| **Weight Sync** | `model_update()` via broadcast comm plans |
| **Cluster Abstraction** | `Cluster` class (manages Ray actors) |
| **Worker Management** | `Cluster` spawns actor workers |
| **TP Size** | `strategy_config.tensor_model_parallel_size` |

### 3.2 Lifecycle Operations Mapping (Post-Adaptation)
| Design Doc Verb | ROLL Implementation | Method / Action |
|-----------------|---------------------|-----------------|
| **offload (DP)** | `stop_server()` (+ `offload_states` if needed) | Clears GPU memory, keeps actor alive |
| **load/backload** | `start_server()` | Wakes inference server on fixed workers |
| **sync (weights)** | `model_update()` before `start_server()` | Sync-on-resume broadcast |
| **broadcast** | `start_model_update()` | Collective broadcast |

### 3.3 Progress/Heartbeat Mapping


| Metric | ROLL Implementation |
|--------|---------------------|
| **queued_trajectories** | `queued_groups * group_size` from `GroupQueue` / `GroupQueueManager` |
| **inflight_trajectories** | `inflight_groups * group_size` (or per-trajectory inflight if available) |
| **percent_remaining** | `(queued_trajectories + inflight_trajectories) / rollout_batch_size` (may be > 1.0 if backlog > one step) |
| **oldest_unfinished** | New timestamp captured at **group creation** for the oldest unfinished group (aligns with `oldest_unfinished_creation_ts` in `design_doc/multi-pipeline_roll_old_design.md`; do not reuse `create_step`, which is a step id) |

### 3.4 Preemption & Release Protocol (Post-Adaptation)
**Important**: ROLL has two distinct “stop/resume” reasons that must be handled differently:
1. **Model-update-driven resume** (weights change): occurs when training advances the policy; generation must sync on resume.
2. **Scheduler-driven time-sharing** (weights unchanged): occurs when SchedRL shrinks/expands the rollout DP subset within the same weight version; this requires migrating in-flight requests/trajectories away from the preempted subset, but does not require a weight sync.

| Protocol | ROLL Implementation |
|----------|---------------------|
| **request_gpus (train)** | Coordinator calls central scheduler; compute gated by `actor_train.offload_states()` |
| **release_gpus (train)** | Coordinator calls central scheduler; `actor_train.offload_states()` |
| **request_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `start_server_subset` via proxy |
| **release_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `stop_server_subset` via proxy |
| **preempt gen (time-share)** | Scheduler triggers `stop_server_subset` via proxy + `suspend()` (abort in-flight requests; pause new ones); requests are retried on remaining active workers |
| **resume gen (time-share)** | Scheduler triggers `start_server_subset` then `resume()` to allow requests to flow again (no `model_update()` if weights unchanged) |
| **resume gen (weight update)** | Scheduler triggers `model_update()` then `start_server_subset` (sync-on-resume), then `resume()` |

## 4. Required Extensions

### 4.1 DP-Granular Selective Sync (WIP Priority)
*   **Status**: **Required** for subset resume.
*   **Analysis**:
    *   `Worker.start_model_update` accepts `broadcast_tgt_devices`, enabling precise GPU targeting.
    *   `ModelUpdateGroup.model_update` executes the *entire* static `comm_plan`.
    *   Protocol requires a **CPU-side latest-weights buffer** after each train step so sync-on-resume can broadcast without keeping rollout GPUs active.
*   **Required Action**:
    *   Maintain a **latest-weights CPU cache** after training completes, owned by the training cluster (e.g., add a `LatestWeightsCache` in `roll/roll/distributed/executor/model_update_group.py` and/or in `roll/roll/distributed/strategy/*_strategy.py` to snapshot model params to CPU buckets).
    *   After `train_step()` completes, **stage all weights to the CPU cache**. Sync is decoupled from cache/staging and happens later on resume.
    *   Create **subset-scoped collective groups** for the active DP workers, and **tear them down after update** (NCCL broadcast is group-scoped).
    *   Update `ModelUpdateGroup.model_update` to accept `worker_indices` and drive subset group setup + broadcast.
    *   Adjust strategy broadcast paths so only active subset workers participate (avoid global broadcast).

### 4.2 Scheduler Progress Hooks
*   **Integration Point**: `GroupQueue.put` (within `GroupQueueManager`).
*   **Trigger**: Report at **batch start** and whenever `percent_remaining` crosses a 2% progress band (event-driven).
*   **Action**: Inject `scheduler.report_progress(queued_trajectories, inflight_trajectories, percent_remaining, oldest_unfinished_creation_ts)` where `percent_remaining` is computed against the per-step rollout target (`rollout_batch_size`, in trajectories). Use `RolloutScheduler.get_batch()` or `advance_step` for the batch-start heartbeat.
*   **Note**: If a group is later dropped/removed, the reported `percent_remaining` may decrease accordingly.

### 4.3 Native Request Migration During Stop (Framework-Specific)
*   **ROLL Native Pattern**: Explicit `abort_request()` (cancellation).
*   **Behavior**:
    1.  `RequestScheduler.suspend()` sets `need_suspend=True` and calls `abort_request()` (lines 959-964 in `generate_scheduler.py`).
    2.  `abort_request()` sends `GenerateRequestType.ABORT` to all in-flight requests, which cancels them at the vLLM/SGLang level (lines 939-953).
    3.  At the env layer, an aborted request returns `None` to the env loop (`PolicyProxy.generate(...)`), which is mapped to `GenerateStopReason.ABORT` (`TrajEnvManager.make_decision`).
    4.  When `stop_reason=ABORT`, the env loop does **not** call `env.step(...)` and simply loops again, effectively retrying the same turn on the next scheduler resume (`TrajEnvManager.run_rollout_loop`).
*   **SchedRL Integration**: Use `abort_request()` as the cancellation primitive. ROLL already retries aborted turns in the env loop; to make shrink/expand correct, ensure routing remaps away from preempted dp ranks (active-rank gating + clear sticky mappings).

### 4.4 Time-Sharing Migration of Running Requests/Trajectories (Required for SchedRL)
Time-sharing shrink/expand differs from model-update-driven resume: when the scheduler preempts/stops a DP subset, any in-flight work on those workers must be moved to still-active rollout workers. In this case, the **policy weights do not change**; only the active worker set changes.

*   **Migration semantics (ROLL)**:
    1.  **Freeze and cancel**: call `RolloutScheduler.suspend()` → `RequestScheduler.suspend()` which aborts all in-flight requests.
    2.  **Shrink subset**: call `stop_server_subset(...)` on the preempted DP ranks.
    3.  **Remap + retry**: clear routing for any preempted dp ranks and call `resume()` to reopen admission on the remaining active ranks; env loops will retry aborted turns and will naturally be routed to the remaining active dp ranks.

### 4.5 Handling Weight Version vs Worker Migration (Required)
*   **Case A: weights change (sync-on-resume)**:
    *   Before restarting generation on any subset, call `model_update()` (using the latest CPU weights cache) so resumed workers generate with the updated policy.
    *   Any aborted turns are retried by the env loop after resume, and will run under the new weights.
*   **Case B: weights unchanged (time-sharing)**:
    *   Do not call `model_update()`; time-sharing shrink/expand uses abort + remap + retry on the remaining active ranks under the same weight version.

### 4.6 Minimal Mid-Flight Shrink/Expand Checklist (Implementation-Ready)

Goal: support elastic time-sharing with `migration_policy=REQUEST_RETRY` (abort/cancel on shrinking subset, retry on remaining/new subset) while reusing ROLL’s existing turn-level retry in the env loop.

**Shrink (mid-flight) — required**
- Subset lifecycle: implement `start_server_subset(worker_indices)` / `stop_server_subset(worker_indices)` (DP-granular) on top of `Cluster`/strategy.
- Admission control: add an `active_dp_ranks` set in `RequestScheduler` and refuse routing to inactive ranks.
  - Defensive rule: if `src_rank2_dp_rank[src_rank]` points to an inactive dp rank (not in `active_dp_ranks`), delete that mapping and re-pick from `active_dp_ranks` (do not block).
- Targeted cancel: extend `RequestScheduler.abort_request()` to `abort_request(worker_indices=..., request_ids=None)` so we can:
  - abort all in-flight requests running on `P` (the shrinking subset) by default, and
  - optionally abort a caller-provided subset of request IDs in the future.
- Routing + bookkeeping cleanup:
  - When a dp rank is shrunk, clear sticky mappings that point to it (`src_rank2_dp_rank`) so the next retry goes to a remaining active dp rank (`ROLL/roll/distributed/scheduler/generate_scheduler.py` `RequestScheduler.generate_one_request`).
  - When aborting requests on `P`, also clear any per-request routing/bookkeeping for the aborted request IDs (e.g., remove entries from `request_id_2_dp_rank`) to avoid stale mappings and make retries land cleanly on `new_S`.
- Concurrency detail (async safety): when building the abort set per dp-rank, take a snapshot of `inflight_requests[dp_rank].keys()` **before any `await`** (e.g., `list(...)`). This avoids dict-mutation hazards without requiring an `asyncio.Lock`.
- Abort/drain ordering (safety): do **not** stop/offload `P` immediately after sending abort.
  - Wait until the abort is ACKed and `P` is drained (no running requests on those dp ranks) before calling `stop_server_subset(P)`.
  - Rationale: sending an abort command does not guarantee it has executed yet; stopping/offloading while work is still running risks GPU memory access errors.
  - Phase 1 (A): drain check inside `RequestScheduler`: spin-wait until `inflight_requests[dp_rank]` is empty for all `dp_rank in P`, then sleep an extra `0.5s` as a conservative buffer before stop/offload.
  - Backlog (C): remove the magic sleep by adding worker-side engine drain/flush + “engine idle” confirmation (vLLM/SGLang), and require **A AND C** before stop/offload for maximum safety.
    - Preferred shape: add a unified `wait_for_engine_idle(worker_indices, timeout_s)` RPC at the scheduler/proxy layer, and implement it by wrapping backend-specific calls under vLLM vs SGLang.
  - Timeout policy: if drain does not complete within a configured timeout, **fail loudly** (raise/emit an error and do not proceed with stop/offload), rather than forcing offload and risking memory errors.
- Retry trigger (already exists): aborted request yields `None` from `PolicyProxy.generate`, which becomes `GenerateStopReason.ABORT`, and the env loop retries the same turn without stepping env state (`ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py`).

## 4.X Mid-Flight Shrink (DP workers) — Implementation sketch

Goal: shrink a subset `P` of DP ranks while work is in flight, without losing trajectories.

Coordinator responsibility (protocol-level):
- Decide which trajectories/turns to preempt from `P`.
- Issue deterministic `request_id = f"{trajectory_id}:{turn_id}:{attempt}"` for every turn.
- Retry aborted turns on the remaining active ranks after abort ACK.
- Error retry limit (safety): cap **engine error retries** per `(trajectory_id, turn_id)` (default: 3, configurable). If exceeded, drop the trajectory and report a metric.
  - Do **not** cap preemption retries (abort due to shrink/expand rebalance).
  - Backlog: if a turn is preempted too many times, stop aborting it and wait for it to finish for shrink (stop new work to `P`, wait for it to finish, then stop/offload).

Adapter / framework hooks (ROLL-level):
1. Admission-close for `P`: mark `P` inactive so the scheduler stops dispatching new requests to those dp ranks.
2. Abort in-flight on `P`: snapshot `inflight_requests[dp_rank]` for `dp_rank in P`, send abort in batch.
3. Wait abort ACK (required): wait until those request ids complete with `finish_reason == abort` (or `inflight_requests[dp_rank]` is empty for all `dp_rank in P`).
4. Stop/offload `P`: only after ACK, call `stop_server_subset(P)` + offload states to release GPUs.

Failure policy:
- If abort/drain does not complete within timeout, fail loudly and do not proceed with offload.

**Expand (default-enabled rebalance) — optional migration**
After `start_server_subset(A)`, rebalance is enabled by default (stronger expand):
- Route new turns to `A` first (update `active_dp_ranks` and clear `src_rank2_dp_rank` or use a TTL).
- Reassign queued/not-started work so it can run on `A` (if/when a pre-dispatch queue exists).
- If still unbalanced, abort selected in-flight turns on overloaded ranks and retry them on underloaded ranks (turn-level retry; does not step env on `ABORT`).
  - Require abort ACK before retry and before stopping/offloading any rank.
  - Stop condition (5% rule): the pipeline coordinator computes `load[dp] = queued_by_worker[dp] + inflight_by_worker[dp]` (trajectory counts) and stops when:
    `(max(load) - min(load)) / max(queued_trajectories + inflight_trajectories, 1) <= 0.05`.

## 5. Post-Adaptation Integration Overview
This section describes how the reused ROLL components (Section 3) and required extensions (Section 4) implement the distributed protocol in `design_doc/multi-pipeline_roll_old_design.md`.
Important background: Training and inference clusters remain separate, with fixed placement groups; GPU sharing relies on offload/load plus subset gating. The protocol respects `async_generation_ratio` to bound staleness while allowing subset preemption/resume.

### 5.0 Proxy Layer (Framework Interception)
*   **Purpose**: Wrap framework-specific components (e.g., `RolloutScheduler`, `RequestScheduler`) to emit **release ACKs** without changing their core behavior.
*   **Behavior**: The proxy forwards calls by default and only injects release notifications; it does **not** mediate scheduler decisions.
*   **Minimal intrusion**: The pipeline coordinator keeps its core logic and calls the central scheduler directly at phase boundaries. Progress *can* be emitted from the rollout buffer (`GroupQueue.put`), but the scheduler heartbeat wiring is a required extension (it currently only updates a local `tqdm` progress bar).

### 5.1 Pipeline Coordinator ↔ Central Scheduler
*   **Who is the pipeline coordinator?** The `AgenticPipeline.run()` loop in `ROLL/roll/pipeline/agentic/agentic_pipeline.py`.
*   **Request/Release (Training)**: The coordinator calls the **central scheduler API** directly to request/release training GPUs; once granted, it gates compute via `actor_train.offload_states()` at phase boundaries.
*   **Blocking allocation**: Training compute starts only after the allocation is granted.
*   **Training offload granularity**: Cluster-wide only in Phase 1 (no subset offload for training).
*   **Request/Release (Generation)**: The coordinator calls the **central scheduler API** directly to request/release a DP subset for generation; the **central scheduler** triggers `start_server_subset` / `stop_server_subset` through the proxy.
*   **Preempt/Resume**: The scheduler initiates preempt/resume; the proxy executes stop/start. `model_update()` is only required when resuming across a **weight change** (post-train sync-on-resume), not for pure time-sharing shrink/expand within the same weight version.
*   **Wrapper role**: The proxy only wraps rollout scheduling components to emit release ACKs and to receive start/stop commands from the central scheduler. Progress reporting must be added (recommended hook: `GroupQueue.put`) so the scheduler can make better time-sharing decisions. The coordinator still decides *when* to call the scheduler and *when* to enter each phase.
*   **Control split**: The scheduler initiates expand/shrink; the coordinator only requests/releases; the proxy executes expand/shrink on rollout workers.
*   **Coordinator boundary**: The coordinator does **not** call `start_server_subset` / `stop_server_subset` directly; those are invoked only via scheduler-initiated commands through the proxy.
*   **Sync timing change**: For weight-update-driven resume, `model_update()` moves to **sync-on-resume** (right before generation resumes). For time-sharing resume with unchanged weights, resume skips `model_update()`.
*   **Versioning**: Keep only the **latest** CPU weight cache (by `global_step`) for sync-on-resume.

### 5.2 Cluster Controller ↔ Scheduler (DP-Granular)
*   **Expand/shrink semantics + API mapping**: `expand` resumes active DP ranks via `start_server_subset`, `shrink` preempts/stops ranks via `stop_server_subset` (mirrors the mermaid workflow).
*   **Selective sync** requires **new collective groups per active subset**. NCCL broadcast is group-scoped, so the subset must form its own group before `model_update()` can broadcast only to active DP workers. The subset group should be **torn down after the update completes**.

### 5.3 Rollout Progress ↔ Scheduler Heartbeats
*   **Integration point & cadence**: See Section 4.2 (2% progress-band reporting from `GroupQueue.put`).
*   **Tie-break data** includes `oldest_unfinished` timestamp from rollout tracking (new timestamp source).
*   **Release ACK**: After **normal release**, the proxy notifies the central scheduler (`notify_cluster_released`) before new preemption decisions are applied. Preempted stops use the scheduler's preempt/ack path.

## 6. Implementation Steps (Phase 1)
1.  **Proxy layer**: Add `SchedRLProxy` wrappers for `RolloutScheduler` and `RequestScheduler` to emit release ACKs while preserving default behavior.
2.  **Minimal loop hooks**: Add small hook points in `AgenticPipeline.run()` for direct central-scheduler calls (request/release, preempt/resume); keep core training logic unchanged.
3.  **Subset lifecycle APIs**: Implement subset start/stop via helper calls to selected `Cluster.workers` without changing `Cluster` if possible.
4.  **Selective sync groups**: Create and tear down NCCL groups per active subset; extend `ModelUpdateGroup` to drive subset group setup + `model_update(worker_indices=...)`.
5.  **Coordinator signals**: Inform the scheduler to stop all rollout DP workers immediately after training finishes; resume with **sync-on-resume** then `start_server_subset` (time-sharing-only resumes skip the sync).
6.  **Inject progress hooks**: Add 2% band reporting in `GroupQueue.put` (within `GroupQueueManager`), plus `oldest_unfinished` timestamp tracking.
7.  **Verify**: Exercise subset preemption/resume and `async_generation_ratio > 0` to confirm bounded staleness and selective sync correctness.

## 7. Arbitrary Placement & Selective Sync
The pooling setting allows **arbitrary placement** of training and inference workers (no strict partition or colocation assumptions). To support this:

*   **Placement generalization**: Placement is arbitrary at config time via `device_mapping` per cluster (no strict partition/colocation assumptions), then fixed for the run.
*   **Control path**: The scheduler only controls active subsets within that fixed placement, applied via expand/shrink.
*   **Selective model update**:
    *   **CUDA IPC**: Use CUDA IPC handles for point-to-point transfer when sender/receiver share a node/GPU topology.
    *   **NCCL broadcast**: Build subset-scoped NCCL groups for cross-node broadcast; broadcast only to active DP workers.
    *   **Hybrid path**: Prefer IPC where possible, fall back to NCCL for remaining targets.
*   **Implementation sketch**:
    *   Extend `ModelUpdateGroup` to generate per-subset comm plans based on **actual device mapping** (not fixed cluster partitions).
    *   Maintain a **latest CPU weight cache** and launch subset sync on resume, using IPC/NCCL as needed.

## 8. Configuration Example
**ROLL** (`agentic_config.yaml`):
```yaml
actor_train:
  strategy_args:
    strategy_name: megatron_train
actor_infer:
  strategy_args:
    strategy_name: vllm
schedrl:
  enabled: true
  scheduler_name: "CentralizedGPUScheduler"
  pipeline_id: "roll_pipeline_0"
```

## 9. Estimated Effort
*   **Complexity**: Medium/High (subset lifecycle APIs + NCCL group management).
*   **Size**: ~400–700 LOC.
