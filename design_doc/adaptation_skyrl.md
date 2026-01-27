# SkyRL Adaptation Plan (SkyRL-train + optional SkyAgent)

## 1) Scope and intent

SkyRL-train provides two async training modes out of the box, and can be a replacement path for the
current rLLM+VeRL stack when we need:
- fully-async training with bounded staleness control, and/or
- one-step-off pipelined generation + training.

This plan focuses on integrating **SkyRL-train** into SchedRL as a supported framework.

**Explicit Scope Exclusion**:
- **SkyContent (SkyAgent)** integration is **dropped** from this plan. It is strictly on-policy/sync today and provides no BATCH/INFLIGHT async modes. We will not support `QUIESCE-by-drain` for it in this phase.
- All "async agent" examples (Mini-SWE) will be implemented purely within **SkyRL-train** (using its `MiniSweAgentGenerator`).

## 1.1 What we will support as async multi-turn examples (scoped)

To keep the first SchedRL integration simple, we only target **GSM8K multi-turn** as the “async + multi-turn” reference for SkyRL-train:
- Multi-turn task: `third_party/SkyRL/skyrl-train/examples/turn_level_rewards/` (env: `gsm8k_multi_turn`)
- Async trainers:
  - One-step-off (recommended): `third_party/SkyRL/skyrl-train/examples/async/async_trainer.py`
  - Fully-async bounded-staleness (supported): `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py`

We do **not** target Mini-SWE-Agent / Terminal-Bench as async examples yet, because those are long tool loops and need extra “resume on abort” handling to be reliable.

## 1.2 Phase 2 (Mini-SWE): async agent + tools

Goal: add a **Mini-SWE** example that runs with async training.

Where Mini-SWE already exists in this workspace:
- SkyRL-train Mini-SWE training example (sync today): `third_party/SkyRL/skyrl-train/examples/mini_swe_agent/`

Phase 2 plan (Mini-SWE in SkyRL-train):
- One-step-off async:
  - Create a new entrypoint that mirrors `third_party/SkyRL/skyrl-train/examples/async/main_async.py`,
    but uses `MiniSweAgentGenerator` from `third_party/SkyRL/skyrl-train/examples/mini_swe_agent/mini_swe_generator.py`.
- Fully-async bounded-staleness:
  - Create a new entrypoint that mirrors `third_party/SkyRL/skyrl-train/examples/fully_async/main_fully_async.py`.
  - Keep the “fully-async constraints” in mind:
    - `generator.batched=false` is required in SkyRL-train fully-async.
    - vLLM-only for abort/pause in the current SkyRL-train code.

Safety rules (important for tools):
- For Phase 2, prefer “stop new starts + wait for drain” when shrinking GPUs.
- Backlog: add idempotency keys for tool actions so retries cannot apply the same tool action twice.

## 2) What SkyRL-train already provides

### 2.1 One-step-off async (pipelined)

SkyRL-train provides a minimal “one-step-off” async trainer that overlaps:
- generation of the next batch, and
- training on the previous batch.

Reference implementation:
- Entrypoint: `third_party/SkyRL/skyrl-train/examples/async/main_async.py`
- Trainer: `third_party/SkyRL/skyrl-train/examples/async/async_trainer.py` (`AsyncRayPPOTrainer`)
- Doc: `third_party/SkyRL/skyrl-train/docs/tutorials/one_step_off_async.rst`

Important constraints:
- `trainer.placement.colocate_all=false` (asserted in the async trainer).

### 2.2 Fully-async training (streaming / bounded staleness)

SkyRL-train provides a fully-async trainer with a staleness budget:
- knob: `trainer.fully_async.max_staleness_steps` in `third_party/SkyRL/skyrl-train/skyrl_train/config/ppo_base_config.yaml`
- implementation: `third_party/SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py` (`FullyAsyncRayPPOTrainer`)
- entrypoint: `third_party/SkyRL/skyrl-train/examples/fully_async/main_fully_async.py`
- doc: `third_party/SkyRL/skyrl-train/docs/tutorials/fully_async.rst`

Important constraints:
- `trainer.placement.colocate_all=false` (asserted)
- `generator.batched=false` (asserted; pause/resume is not supported for batched generate)
- `generator.async_engine=true` (asserted)
- In-flight weight update uses `pause_generation()` + `resume_generation()` which relies on abort support.
  - vLLM path supports abort; SGLang abort is not implemented in SkyRL-train today, so treat fully-async as **vLLM-only**.

## 3) Mapping to SchedRL protocol concepts

### 3.1 Update policy

- One-step-off async maps to `update_policy=BATCH`:
  - activation occurs at the pipelined boundary where the trainer triggers weight sync.
- Fully-async maps to `update_policy=QUIESCE-by-abort`:
  - **Protocol View**: The scheduler sees `inflight` drop to zero (active on engine). The client's holding of tasks ("resume loop") is semantically equivalent to re-queueing.
  - **Mechanism**: The adapter implements the sync via `pause_generation()` (abort-to-zero) -> `weight_sync` -> `resume_generation()` (re-issue).
  - Note: Strict consistency is preserved (no requests finish on old weights).

### 3.2 Migration policy (shrink/expand)

SkyRL-train’s async trainers assume a fixed set of inference engines for a run (`generator.num_inference_engines`).

Baseline for SchedRL time-sharing:
- Use `migration_policy=REQUEST_RETRY` (abort current request/turn and retry on remaining engines).

Reality check:
- **SchedRL hard requirement**: shrink must support mid-flight abort+ACK+retry (not only safe boundaries).
- SkyRL-train already has:
  - abort primitives (vLLM path), and
  - retry-on-abort semantics for single-prompt generate (token-in/token-out).
- **Required extensions for DP/engine shrink** (to satisfy hard requirement):
  - Subset lifecycle: add “active engine set” control so the coordinator can shrink a subset `P` (remove from routing) and later expand it back.
  - Deterministic request id: coordinator must provide `request_id = f"{trajectory_id}:{turn_id}:{attempt}"` for every turn, and vLLM must use it as the engine `request_id`.
    - This requires a stable `trajectory_id` (same across retries) and a stable `turn_id` within that trajectory.
  - Targeted abort: abort only the request_ids running on the shrinking subset `P` (not abort-all).
  - Abort ACK (required): wait for stop_reason == `abort` before releasing/offloading `P`.
  - Retry: reissue the same turn on remaining engines after ACK (does not restart the whole trajectory if commit point is not crossed).
  - Error retry limit (safety): cap **engine error retries** per `(trajectory_id, turn_id)` (default: 3, configurable). If exceeded, drop the trajectory and report a metric.
    - Do **not** cap preemption retries (abort due to shrink/expand rebalance).
    - Backlog: track how many times a turn is preempted; if it is preempted too many times, stop aborting it and wait for it to finish for shrink (stop new work to those engines, wait for it to finish, then release/offload).

Expand rebalance (stronger expand, enabled by default):
- Route new items to newly expanded engines first.
- Reassign queued/not-started work so it can run on newly expanded engines.
- If still unbalanced, abort selected in-flight turns and retry them on underloaded engines (abort ACK required).
  - Stop condition (5% rule): the pipeline coordinator computes `load[dp] = queued_by_worker[dp] + inflight_by_worker[dp]` (trajectory counts) and stops when:
    `(max(load) - min(load)) / (train_batch_size * n_samples_per_prompt) <= 0.05`.

**Baseline validation (required)**
- Validate the `REQUEST_RETRY` safety invariant: do not execute stateful env/tool side effects unless a non-abort generation result is received (single-writer commit).
  - For GSM8K-style tasks this is typically trivial (no stateful env).
  - For any SkyRL-Gym / agentic task with stateful tools, define the commit point explicitly before enabling mid-flight shrink.

## 3.3 Progress reporting (2% bands, trajectory counts)

SkyRL-train reports in trajectory units already (one `TrajectoryID` is one trajectory).

- Report `queued_trajectories` and `inflight_trajectories` separately.
- Report `oldest_unfinished_creation_ts` (timestamp of the oldest unfinished TrajectoryID, queued or in-flight) for scheduler tie-breaks.
- Keep the 2% cadence, with denominator = `train_batch_size * n_samples_per_prompt` (trajectory units) for one training step.
  - `percent_completed = collected_trajectories / (train_batch_size * n_samples_per_prompt)` (`percent_completed >= 1.0` means next train step batch is ready)
  - This may be > 100% if the backlog is larger than one step.

Version tagging (simple, for debugging):
- Record `generation_checkpoint_version` when the first turn of a trajectory is submitted, and record it again when the last turn finishes.

## 4) How to run today (use existing examples)

For now, treat SkyRL-train as a supported pipeline by using its existing example entrypoints:
- One-step-off: run `third_party/SkyRL/skyrl-train/examples/async/main_async.py`
- Fully-async: run `third_party/SkyRL/skyrl-train/examples/fully_async/main_fully_async.py`

This provides the async training modes without requiring custom SchedRL adapter code immediately.

For the first **async + multi-turn** reference, use the GSM8K multi-turn task setup:
- `third_party/SkyRL/skyrl-train/examples/turn_level_rewards/` (multi-turn env + turn-level rewards)
Then wire it to the async trainers above (one-step-off first).
