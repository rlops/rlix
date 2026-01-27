# NeMo-RL Adaptation Plan

## 1. Overview
NeMo-RL is the Phase 2 target, serving as the "Structural Pilot" for SchedRL. It provides a clean, rigid abstraction (`RayWorkerGroup`) that must be extended to support subset execution, a foundational pattern for the scheduler.
We keep the core training loop intact and introduce a **proxy layer** to intercept framework-specific operations (e.g., rollout/generation controllers). The pipeline coordinator calls the central scheduler **directly**; the proxy is unidirectional (wrapper only) and emits release ACKs.

## 1.2 Simple async + multi-turn example (planned)

NeMo-RL supports async GRPO and it already has a multi-turn environment example. But the repo does not yet have a ready “async + multi-turn” example that we can run as-is.

- Best multi-turn task to start from: sliding puzzle
  - Script: `third_party/nemo-rl/examples/run_grpo_sliding_puzzle.py`
  - Config: `third_party/nemo-rl/examples/configs/grpo_sliding_puzzle.yaml` (already multi-turn; `max_rollout_turns` is set)
- Best async reference to reuse:
  - Guide: `third_party/nemo-rl/docs/guides/async-grpo.md`
  - Core logic: `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py` already runs multi-turn async rollout (`run_async_multi_turn_rollout`)

Planned work (doc-level):
- Add one new “async sliding puzzle” config and/or run script that turns on:
  - async GRPO mode, and
  - async vLLM engine (if needed for the run).
- Use it as the NeMo-RL main reference “async + multi-turn” example for SchedRL.

Rough estimate: 2–5 days (add config/script + debug runtime issues).

## 1.3 Phase 2 (Mini-SWE): async agent + tools

Goal: add a **Mini-SWE** training example for NeMo-RL that is:
- multi-turn,
- uses real tools (shell / git patch),
- and runs under NeMo-RL async training.

What we can reuse in this workspace:
- Mini-SWE agent server code exists in NeMo-Gym:
  - `third_party/nemo-gym/responses_api_agents/mini_swe_agent/`
- Mini-SWE resource configs exist:
  - `third_party/nemo-gym/resources_servers/mini_swe_agent/`

Planned work (doc-level first, then code):
- Define a NeMo-RL rollout “task” that uses the Mini-SWE agent server as the environment/tool runner.
- Start with one-step-off style overlap (generate batch N+1 while training batch N) using existing async GRPO plumbing.
- Add clear “safe stop” points for shrink/time-share:
  - stop new starts,
  - wait for in-flight tool loops to finish,
  - then sleep/offload a DP subset.

Safety rules (important for tools):
- Do not allow mid-tool abort as the default for Mini-SWE.
- If we need mid-flight shrink later, add idempotency keys for tool actions and teach the tool server to dedupe.

## 1.1 Protocol Fit (Against `multi-pipeline-adaptation-plan_clean.md`)

This section reality-checks NeMo-RL against the shared protocol in `design_doc/multi-pipeline-adaptation-plan_clean.md` and lists concrete gaps.

**Already present in the codebase**
- **In-flight / in-place weight update (INFLIGHT semantics)**: `third_party/nemo-rl/nemo_rl/algorithms/grpo.py` `refit_policy_generation(...)` supports colocated (IPC/ZMQ) and non-colocated (NCCL collective) weight update.
- **Admission gating around update**: `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py` `AsyncTrajectoryCollector.prepare_for_refit()` pauses launching new **prompt-group rollout threads** (one thread per prompt index; each thread runs a multi-turn rollout over `num_generations_per_prompt` trajectories) and optionally waits for those in-flight threads depending on `async_engine` + `in_flight_weight_updates`.
- **Version tagging for async GRPO**: trajectories are pushed with `(generation_weight_version, target_weight_version)` and consumed with target matching (bounded staleness) (`third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`).

**Gaps / required extensions to support elastic shrink/expand**
- **Subset lifecycle (DP-granular wake/sleep)**:
  - `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py` `prepare_for_generation()` / `finish_generation()` call wake/sleep for *all* workers via `run_all_workers_single_data`.
  - The shared protocol requires subset activation/deactivation for `expand_workers(indices)` / `shrink_workers(indices)`.
  - Good news: `third_party/nemo-rl/nemo_rl/distributed/worker_groups.py` has `run_single_worker_single_data(...)`, so we can target specific worker indices once the generation wrapper exposes an indices parameter.
- **Preemption migration on shrink**:
  - There is no obvious “abort these request_ids on this subset” API exposed at the NeMo-RL generation layer today (unlike ROLL’s `GenerateRequestType.ABORT`).
  - To support SchedRL mid-flight shrink (required for time-sharing), we should reuse existing primitives and add a small extension:
    - NeMo-RL already tracks in-flight prompt-group rollout threads in `AsyncTrajectoryCollector` (`_inflight_threads`) and already supports pausing new starts (`prepare_for_refit()`).
    - NeMo-RL vLLM async worker currently creates per-request UUIDs (`request_id = str(uuid.uuid4())` in `vllm_worker_async.py`), which prevents coordinator-owned request identity.
      - **Required extension (hard requirement)**: allow the pipeline coordinator to provide the vLLM `request_id` for each turn, e.g. `request_id = f"{trajectory_id}:{turn_id}:{attempt}"`, and pass it through to `llm.generate(..., request_id=request_id)`.
        - This requires a stable `trajectory_id` (same across retries) and a stable `turn_id` within that trajectory.
      - **Required extension**: expose a per-worker `abort_requests(worker_indices, request_ids)` hook at the generation wrapper level so the coordinator can abort only the shrinking subset `P`.
      - **Required behavior**: wait for abort ACK before shrink proceeds. ACK means the generation returns with stop_reason/finish_reason == `abort`.
      - Retry semantics: the coordinator reissues the **same turn** on a remaining active worker (this does not restart the whole trajectory if multi-turn state is preserved and the commit point was not crossed).
- **Retry safety invariant (two-phase commit) must be validated**:
  - For `migration_policy=REQUEST_RETRY` to be safe, any stateful env/tool effects must only happen after a non-abort generation result is received (single-writer commit).
  - NeMo-RL prompt-group rollouts are often stateless scoring tasks, but if a NeMo-Gym style stateful env is used, the adapter must explicitly define the commit point and ensure abort/cancel happens before commit.
- **Terminology mapping**:
  - Shared protocol uses `active_checkpoint_version` / `generation_checkpoint_version`. NeMo-RL uses `current_weight_version`, `generation_weight_version`, `target_weight_version`. We should map these 1:1 (active = target for new work).
- **Progress reporting for scheduling**:
  - The central scheduler needs heartbeats based on **trajectory counts**: `queued_trajectories`, `inflight_trajectories`, `percent_completed`, and `oldest_unfinished_creation_ts`.
  - NeMo-RL often reasons in “prompt-groups” (one prompt-group contains multiple trajectories: `num_generations_per_prompt`). The adapter must convert prompt-group counts into trajectory counts.

**Concrete file refs & immediate actions**
- Files: `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`, `third_party/nemo-rl/nemo_rl/algorithms/grpo.py`, `third_party/nemo-rl/nemo_rl/models/generation/vllm/vllm_generation.py`, `third_party/nemo-rl/nemo_rl/distributed/worker_groups.py`.
- Actions:
  - Add `wake(worker_indices)` / `sleep(worker_indices)` to `VllmGeneration` and thread through `RayWorkerGroup` helpers (use `run_single_worker_single_data` as a reference).
  - Instrument `AsyncTrajectoryCollector` to emit `scheduler.report_progress(...)` at batch start and on 2% progress bands:
    - `queued_trajectories`, `inflight_trajectories` (trajectory units),
    - `percent_completed = collected_trajectories / train_global_batch_size` (where `train_global_batch_size = num_prompt_groups_needed * num_generations_per_prompt`; `percent_completed >= 1.0` means next train step batch is ready),
    - `oldest_unfinished_creation_ts`.
  - Add simple version tagging for debugging:
    - record `generation_checkpoint_version` when the first turn of a trajectory is submitted, and
    - record it again when the last turn finishes (final output pushed into the replay buffer).
  - Add `REQUEST_RETRY` support for shrink:
    - Make `request_id` coordinator-provided and deterministic per turn (`trajectory_id:turn_id:attempt`) instead of worker-generated UUID.
    - Track `(request_id -> dp_idx)` for in-flight async vLLM requests in the coordinator/generation layer.
    - Implement `abort_requests(worker_indices, request_ids)` in the generation wrapper; require ACK (stop_reason == abort) before retry.
    - On abort, re-enqueue the affected work into a retry queue so it is retried on a non-preempted dp worker.
  - Enable `expand_rebalance_policy = REBALANCE_QUEUED` (default): after `wake(worker_indices=A)` + weight sync, preferentially schedule queued/not-started prompt groups (including retries) onto the expanded subset.

**Baseline validation (required)**
- Validate the `REQUEST_RETRY` safety invariant: do not execute stateful env/tool side effects unless a non-abort generation result is received (single-writer commit). If not true, mid-flight shrink is unsafe and must be disabled until fixed.

**Implication**
- NeMo-RL can implement `update_policy=INFLIGHT` (no allocation change) in a protocol-correct way.
- The main missing piece for multi-pipeline GPU sharing is **subset shrink/expand with mid-flight `REQUEST_RETRY` migration** (plus shared heartbeats).

**Concise actionable items (merged from `design_doc/archive/adaptation_review.md`)**
- Add DP-subset lifecycle entrypoints (indices/ranks) so the scheduler can wake/sleep only a subset of generation workers (`expand_workers`/`shrink_workers`).
- Add a shrink-time preemption hook that can abort/cancel in-flight work for a subset; if per-request abort is not feasible initially, treat subset `sleep()` as a hard stop and rely on `REQUEST_RETRY` via the retry queue (retry current request/turn on remaining workers). (Today NeMo-RL’s vLLM worker calls `llm.sleep(level=1)`; making “deeper” sleep a tunable parameter would be an extension.)
- Wire standardized progress heartbeats from `AsyncTrajectoryCollector` after enqueue/buffer updates.

**Critical Implementation Gaps (Must Fix Before Phase 0)**

| Gap | Location | Issue | Fix Required |
|-----|----------|-------|--------------|
| **Two-phase admission incomplete** | `async_utils.py` | Already-queued prompt-group threads can start after admission close via `_refit_pause_cleared` | Add queue-drain barrier; ensure no queued work starts after close |
| **No `creation_ts` tracking** | `async_utils.py` | `oldest_unfinished_creation_ts` required but not tracked; timestamp set at completion not submission | Add enqueue timestamp to prompt-group at submission time, not completion |


## 2. Existing Code Integration Points (Pre-Adaptation)

### 2.1 Training Entry Point
*   **File**: `nemo_rl/algorithms/grpo/grpo_trainer.py` (or similar)
*   **Hook**: `grpo_train()` loop, before/after `policy.train_step()`.

### 2.2 Generation Entry Point
*   **File**: `nemo_rl/generation/vllm_generation.py`
*   **Method**: `run_async_multi_turn_rollout()`
*   **Hook**: `VllmGeneration.prepare_for_generation()`
*   **Hook (async GRPO)**: `AsyncTrajectoryCollector` drives concurrent rollout threads and gates new generation starts during refit via `prepare_for_refit()` / `resume_after_refit()` (`nemo_rl/algorithms/async_utils.py`).

### 2.3 Weight Sync (Pre-Adaptation)
*   **Mechanism**: `refit_policy_generation()` selects one of two weight-sync paths:
    *   **Colocated inference (`colocated_inference=True`)**: CUDA-IPC + ZMQ streaming (`policy.stream_weights_via_ipc_zmq(...)` + `policy_generation.update_weights_via_ipc_zmq()`).
    *   **Non-colocated inference (`colocated_inference=False`)**: NCCL collective broadcast (`policy.broadcast_weights_for_collective(...)` + `policy_generation.update_weights_from_collective()`).
*   **Hook**: `refit_policy_generation()` is invoked at generation boundaries inside the training loop (e.g., `third_party/nemo-rl/nemo_rl/algorithms/grpo.py`), before `run_*_rollout`.
*   **Note (async GRPO)**: `AsyncTrajectoryCollector.prepare_for_refit()` pauses launching *new* prompt-group rollouts and may optionally wait for in-flight prompt-group threads depending on `async_engine` and `in_flight_weight_updates` (see Section 2.4).

### 2.4 Running Requests & Trajectories (Pre-Adaptation)
*   **Running request handling (async GRPO / vLLM async engine)**
    *   **File**: `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`
    *   **Hook**: `AsyncTrajectoryCollector.prepare_for_refit()`
    *   **Behavior**:
        *   Pauses **new** generation starts by clearing `_refit_pause_cleared`.
        *   If `vllm_cfg.async_engine=true` and `grpo.async_grpo.in_flight_weight_updates=true`, in-flight prompt-group rollouts are allowed to complete while weights are updated (no global abort).
        *   Otherwise, refit waits for all in-flight prompt-group threads to complete (`wait_for_pending_generations()`).
*   **Running trajectory handling (buffering + versioning)**
    *   **Files**: `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py`, `third_party/nemo-rl/nemo_rl/algorithms/grpo.py`
    *   **Behavior**:
        *   Each rollout thread produces a per-prompt trajectory group and pushes it to `ReplayBuffer.push_with_wait_signal(...)` with `(generation_weight_version, target_weight_version)`.
        *   Training consumes only trajectories whose `target_weight_version == current_weight_version` (bounded staleness controlled by `max_trajectory_age_steps`).
*   **Important distinction: async training vs time-sharing scheduling**
    *   **Async training** (NeMo async GRPO) can tolerate in-flight requests finishing on the *same* rollout workers while weights are updated (when `in_flight_weight_updates=true`).
    *   **Time-sharing scheduling** (SchedRL preemption/sleep of a DP subset) requires **migration** of in-flight requests/trajectories away from the workers being preempted (see Section 4.5).

## 3. Architecture Mapping

### 3.1 Component Mapping
| Design Doc Concept | NeMo-RL Implementation | Key Classes |
|-------------------|------------------------|-------------|
| **Policy Training** | `RayWorkerGroup` with `MegatronPolicyWorker` | `Policy`, `lm_policy.py` |
| **Generation/Rollout** | `VllmGeneration` with `VllmGenerationWorker` | `VllmGeneration`, `vllm_generation.py` |
| **Weight Sync** | `refit_policy_generation()` (CUDA-IPC/ZMQ or NCCL) | `grpo.py` |
| **Cluster Abstraction** | `RayVirtualCluster` (placement groups) | `virtual_cluster.py` |
| **Worker Management** | `RayWorkerGroup` (Ray actors for DP) | `worker_groups.py` |
| **Rollout Buffer** | `AsyncTrajectoryCollector` | `trajectory_collector.py` |
| **TP Size** | `policy.generation.vllm_cfg.tensor_parallel_size` | Config |

### 3.2 Lifecycle Operations Mapping (Post-Adaptation)

**Terminology Distinction:**
- **Cluster-Level Operations**: Enable or disable the entire cluster (all workers). These are the existing APIs.
- **Subset-Level Operations (Required)**: Activate or deactivate specific DP workers. These require the extensions described in Section 4.1.

| Design Doc Verb | NeMo-RL Implementation | Method / Action |
|-----------------|------------------------|-----------------|
| **expand (cluster)** | `prepare_for_generation()` | Wakes ALL vLLM workers, loads weights |
| **shrink (cluster)** | `finish_generation()` | Sleeps ALL workers, resets KV cache |
| **expand (subset)** | `wake(worker_indices)` (requires extension) | Wakes specific workers, loads weights |
| **shrink (subset)** | `sleep(worker_indices)` (requires extension) | Sleep specific workers (today vLLM wrapper uses `sleep(level=1)`; making level tunable is an extension) |
| **offload (DP)** | `sleep()` | Sleep vLLM engine (today uses `sleep(level=1)`; clarify desired “drop weights” level in extension) |
| **load/backload** | `wake()` | Triggers new IPC setup + weight broadcast |
| **sync (weights)** | `refit_policy_generation()` | CUDA-IPC/ZMQ streaming or NCCL collective |
| **broadcast** | `update_weights_via_ipc_zmq()` / `update_weights_from_collective()` | GPU-to-GPU transfer |

### 3.3 Progress/Heartbeat Mapping
| Metric | NeMo-RL Implementation |
|--------|------------------------|
| **queued_trajectories** | `queued_prompt_groups * num_generations_per_prompt` |
| **inflight_trajectories** | `inflight_prompt_groups * num_generations_per_prompt` |
| **percent_completed** | `collected_trajectories / train_global_batch_size` (where `train_global_batch_size = num_prompt_groups_needed * num_generations_per_prompt`; `percent_completed >= 1.0` means next train step batch is ready) |
| **oldest_unfinished** | Trajectory timestamp in collector |

### 3.4 Preemption & Release Protocol (Post-Adaptation)
**Important**: NeMo-RL has two distinct “stop/resume” reasons that must be handled differently:
1. **Model-update-driven refit** (weights change): occurs when training advances the policy and generation must sync to new weights.
2. **Scheduler-driven time-sharing** (weights unchanged): occurs when SchedRL shrinks/expands the rollout DP subset to share GPUs; this requires migrating in-flight work away from preempted workers.

| Protocol | NeMo-RL Implementation |
|----------|------------------------|
| **request_gpus (train)** | Coordinator calls central scheduler; compute gated by `policy.offload_after_refit()` |
| **release_gpus (train)** | Coordinator calls central scheduler; `policy.offload_after_refit()` |
| **request_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `wake(worker_indices)` via proxy |
| **release_gpus (gen)** | Coordinator calls central scheduler; scheduler triggers `sleep(worker_indices)` via proxy |
| **preempt gen (time-share)** | Pause new starts, migrate/cancel in-flight work on preempted subset (Section 4.5), then `sleep(worker_indices)` |
| **resume gen (time-share)** | `wake(worker_indices)`; if weights were destroyed by sleep, reload **the same** weight version before resuming new starts (Section 4.6) |
| **refit gen (weight update)** | `prepare_for_refit()` then `refit_policy_generation()` then `resume_after_refit()` (Section 4.4); migration is not required unless preemption is also requested |

## 4. Required Extensions

### 4.1 DP-Granular Selective Execution (Critical)
*   **Status**: **Rigid / Missing**.
*   **Analysis**: `RayWorkerGroup` relies on `run_all_workers_single_data`, which allows no subset filtering.
*   **Required Action**: 
    1.  Modify `run_all_workers_single_data` in `nemo_rl/distributed/worker_groups.py` to accept `worker_indices: list[int]`.
    2.  Update `Policy` methods (`stream_weights_via_ipc_zmq`, `broadcast_weights_for_collective`) to propagate these indices.
    3.  Add `wake(worker_indices)` and `sleep(worker_indices)` methods to `VllmGeneration` or `RayWorkerGroup` to support subset activation/deactivation without affecting the whole cluster.

### 4.2 Scheduler Progress Hooks
*   **Integration Point**: `AsyncTrajectoryCollector._run_prompt_group_worker` in `third_party/nemo-rl/nemo_rl/algorithms/async_utils.py` (buffer enqueue via `replay_buffer.push_with_wait_signal(...)`), plus the `grpo_train` loop in `third_party/nemo-rl/nemo_rl/algorithms/grpo.py`.
*   **Trigger**: On buffer enqueue (inside `_run_prompt_group_worker`) and at the start of the batch loop (`for batch in dataloader:` around line 1118).
*   **Action**: Inject `scheduler.report_progress(queued_trajectories, inflight_trajectories, percent_completed, oldest_unfinished_creation_ts)`.
*   **Frequency**: Report at **batch start** and whenever `percent_completed` crosses a **2% progress band** (event-driven). Denominator is the per-step rollout target in trajectories: `train_global_batch_size` (i.e., `num_prompt_groups_needed * num_generations_per_prompt`). Use `oldest_unfinished` timestamp for scheduler tie-breaking.

### 4.3 Step-Level Retry (If Applicable)
*   **Status**: Optional.
*   **Note**: SchedRL mid-flight shrink does not require step-level state migration. Baseline correctness can use `REQUEST_RETRY` (retry the current request/turn for the affected prompt group on another worker).

### 4.4 Native Request Migration During Stop (Framework-Specific)
*   **NeMo-RL Native Pattern (model-update-driven refit)**: `prepare_for_refit()` + optional `wait_for_pending_generations()`.
*   **Behavior**:
    1.  `_refit_pause_cleared.clear()` pauses **new** generation starts (lines 541-543 in `async_utils.py`).
    2.  For **in-flight weight updates** (vLLM V1 async engine with `in_flight_weight_updates=True`): pending prompt-group rollouts are **allowed to complete** with their current weights/KV caches while weights are updated (lines 558-567). This is the preferred async pattern.
    3.  For **non-async engines**: `wait_for_pending_generations()` blocks until all in-flight threads complete (lines 569-573).
*   **SchedRL Integration**: When calling `sleep(worker_indices)` for preemption, the proxy should invoke the framework-native pause mechanism. Do **not** force ROLL-style `abort_request()` semantics.

### 4.5 Time-Sharing Migration of Running Requests/Trajectories (Required for SchedRL)
Time-sharing scheduling differs from async training: when the scheduler preempts/sleeps a DP subset, any in-flight work on those workers must be moved to still-active rollout workers. In this case, the **policy weights do not change**; only the active worker set changes.

*   **Goal**: support mid-flight shrink via `migration_policy=REQUEST_RETRY` (cancel on `P`, retry on `new_S`) while preserving eventual completion.
*   **Reuse in current codebase**:
    - `prepare_for_refit()` already closes admission for new starts (`_refit_pause_cleared.clear()`).
    - Async vLLM worker already uses per-request UUIDs (`request_id = str(uuid.uuid4())` in `vllm_worker_async.py`).
    - The collector already has per-prompt-group worker threads and has the `repeated_batch` needed to restart generation.
*   **Minimal extension (Phase 2)**:
    1. **Close admission**: call `prepare_for_refit()` (or a new `prepare_for_preempt()`) to stop launching new prompt groups.
    2. **Abort/cancel subset `P`**:
       - Track `(request_id -> dp_idx)` for in-flight requests, and expose `abort_requests(worker_indices, request_ids)` in the vLLM wrapper; or
       - As a fallback, subset `sleep(worker_indices)` is treated as a hard stop for those workers (generation calls error/cancel).
    3. **Retry**: on abort/cancel/error in `_run_prompt_group_worker`, enqueue the same `repeated_batch` into a retry queue; `_process_batch` drains that retry queue and regenerates on remaining active dp workers.
*   **Expand behavior**:
    - Expand is followed by weight sync for new workers; **rebalance-on-expand is enabled by default for queued/not-started prompt groups** (future prompt groups flow to the expanded dp ranks via routing).
*   **Config/behavior control**:
    *   For time-sharing, the proxy may need to override the “allow in-flight completion” behavior even when `in_flight_weight_updates=true`, because preempted workers must vacate GPUs promptly.

### 4.7 Minimal Mid-Flight Shrink/Expand Checklist (Implementation-Ready)

Goal: implement `migration_policy=REQUEST_RETRY` for mid-flight shrink by reusing the existing “prompt-group worker thread” structure and retrying the **current turn** on a remaining worker after abort ACK.

**Shrink (mid-flight) — required**
- Subset lifecycle: add `wake(worker_indices)` / `sleep(worker_indices)` in `VllmGeneration` by reusing `RayWorkerGroup` single-worker calls.
- Admission control: reuse `AsyncTrajectoryCollector.prepare_for_refit()` to stop launching new prompt groups.
- Cancel/abort shrinking subset `P`:
  - Required: expose `abort_requests(worker_indices, request_ids)` and track `(request_id -> dp_idx)` for async vLLM requests.
  - Required: wait for abort ACK (stop_reason == `abort`) before proceeding to sleep/offload those workers.
- Retry queue (retry current request/turn):
  - On error/cancel in `_run_prompt_group_worker`, enqueue the same `repeated_batch` into a `retry_queue`.
  - In `_process_batch`, drain `retry_queue` first so retries land on remaining active dp workers.
  - Error retry limit (safety): cap **engine error retries** per `(trajectory_id, turn_id)` (default: 3, configurable). If exceeded, drop the trajectory and report a metric.
    - Do **not** cap preemption retries (abort due to shrink/expand rebalance).
    - Backlog: track how many times a turn is preempted; if it is preempted too many times, stop aborting it and wait for it to finish for shrink (stop new work to that dp subset, wait for it to finish, then sleep/offload).

**Expand (default-enabled rebalance) — optional migration**
After `wake(worker_indices=A)` + weight sync, rebalance is enabled by default (stronger expand):
- Route new prompt-groups to the expanded dp ranks first.
- Reassign queued/not-started work (including `retry_queue`) so it can run on `A`.
- If still unbalanced, abort selected in-flight turns on overloaded workers and retry them on underloaded workers.
  - Require abort ACK before retry.
  - Stop condition (5% rule): the pipeline coordinator computes `load[dp] = queued_by_worker[dp] + inflight_by_worker[dp]` (trajectory counts) and stops when:
    `(max(load) - min(load)) / train_global_batch_size <= 0.05`.

### 4.6 Handling Weight Version vs Worker Migration (Required)
Time-sharing shrink/expand and model-update-driven refit interact but are logically separate:

*   **Case A: weights change (refit)**:
    *   Any trajectories that complete during refit must be tagged with their `generation_weight_version` and only consumed for compatible `target_weight_version` (bounded staleness).
    *   If strict weight consistency is required for a run, the proxy should cancel in-flight work during refit (even if `in_flight_weight_updates=true`) and regenerate after the sync completes.
*   **Case B: weights unchanged (time-share migration)**:
    *   Migrated/retried work should keep the same `generation_weight_version`; only routing changes.
    *   If preempted workers were put to `sleep()`, their weights/prefix caches may be dropped depending on vLLM sleep level; when they are later expanded back, they must **reload/sync the active version** before accepting new work. Practically, reuse `refit_policy_generation()` but target the current (unchanged) policy weights/version, not a new training step.

## 5. Post-Adaptation Integration Overview
This section describes how reused NeMo-RL components and required extensions implement the protocol in `design_doc/multi-pipeline_roll_old_design.md`.

### 5.0 Proxy Layer (Framework Interception)
*   **Purpose**: Wrap framework-specific components to emit **release ACKs** without changing core behavior.
*   **Behavior**: The proxy forwards calls by default and only injects release notifications; it does **not** mediate scheduler decisions.
*   **Minimal intrusion**: The pipeline coordinator keeps its core logic and calls the central scheduler directly at phase boundaries. Progress reporting must be added (Section 4.2/4.7) so the scheduler can make better time-sharing decisions.

### 5.1 Pipeline Coordinator ↔ Central Scheduler
*   **Who is the pipeline coordinator?** The `grpo_train()` loop in `nemo_rl/algorithms/grpo/grpo_trainer.py`.
*   **Request/Release (Training)**: The coordinator calls the **central scheduler API** directly to request/release training GPUs; once granted, it gates compute via `policy.offload_after_refit()` at phase boundaries.
*   **Blocking allocation**: Training compute starts only after the allocation is granted.
*   **Training offload granularity**: Cluster-wide only in Phase 2 (no subset offload for training).
*   **Request/Release (Generation)**: The coordinator calls the **central scheduler API** directly; the **central scheduler** triggers `wake(worker_indices)` / `sleep(worker_indices)` via the proxy.
*   **Preempt/Resume**: The scheduler initiates preempt/resume; the proxy executes sleep/wake, and sync-on-resume runs before restarting the subset.
*   **Control split**: The scheduler initiates expand/shrink; the coordinator only requests/releases; the proxy executes expand/shrink on rollout workers.
*   **Sync timing change**: Post-adaptation, `refit_policy_generation()` moves to **sync-on-resume** (right before generation resumes).
*   **Async training constraint**: For async GRPO (`async_grpo_train`), rollout **must be stopped after each training step** before `refit_policy_generation()` can proceed. The scheduler coordinates this stop before triggering sync-on-resume.
*   **Versioning**: Keep only the **latest** weight version (by `global_step`) as the resume target; `refit_policy_generation()` always syncs the current policy weights.

### 5.2 Cluster Controller ↔ Scheduler (DP-Granular)
*   **Expand/shrink semantics + API mapping**: `expand` resumes DP ranks via `wake(worker_indices)`, `shrink` preempts/stops ranks via `sleep(worker_indices)`.
*   **Selective sync** uses CUDA-IPC/ZMQ (colocated) or NCCL collective (non-colocated) for the active subset; rebuild the corresponding groups per resume and tear down after update.

### 5.3 Rollout Progress ↔ Scheduler Heartbeats
*   **Integration point & cadence**: See Section 4.2 (batch start + 2% progress-band reporting).
*   **Tie-break data** includes `oldest_unfinished` timestamp from the trajectory buffer.
*   **Release ACK**: After normal generation release, the proxy (wrapping `VllmGeneration` / generation controller) notifies the central scheduler (`notify_cluster_released`) before new preemption decisions are applied.

## 6. Implementation Steps (Phase 2)
1.  **Proxy layer**: Add `SchedRLProxy` wrappers for generation controllers to emit release ACKs while preserving default behavior.
2.  **Minimal loop hooks**: Add small hook points in `grpo_train()` for direct central-scheduler calls (request/release, preempt/resume); keep core training logic unchanged.
3.  **Subset lifecycle APIs**: Add `worker_indices` support to `RayWorkerGroup` and propagate to `VllmGeneration.wake/sleep`.
4.  **Selective sync-on-resume**: Ensure `refit_policy_generation()` is invoked on resume for the active subset only; target only the latest weight version.
5.  **Inject progress hooks**: Add batch-start + 2% band reporting in `AsyncTrajectoryCollector`, including `oldest_unfinished_creation_ts`.
6.  **Verify**: Test with `colocated: false` config to validate independent scaling/sleeping.

## 7. Arbitrary Placement & Selective Sync
Placement is arbitrary at config time via **manual Ray placement groups / cluster mapping**, then fixed for the run; the scheduler only controls active subsets within that fixed placement.
Selective sync uses CUDA-IPC/ZMQ (colocated) or NCCL collective (non-colocated) for subset workers and rebuilds the corresponding groups on resume.

## 8. Configuration Example
**NeMo-RL** (`grpo_config.yaml`):
```yaml
policy:
  megatron_cfg:
    enabled: true
  generation:
    backend: vllm
schedrl:
  enabled: true
  scheduler_name: "CentralizedGPUScheduler"
  pipeline_id: "nemo_pipeline_0"
```

## 9. Estimated Effort
*   **Complexity**: Medium/High (subset lifecycle + IPC group management).
*   **Size**: ~200–400 LOC.
