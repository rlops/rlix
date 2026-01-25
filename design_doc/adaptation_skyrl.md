# SkyRL Adaptation Plan (SkyRL-train + optional SkyAgent)

## 1) Scope and intent

SkyRL-train provides two async training modes out of the box, and can be a replacement path for the
current rLLM+VeRL stack when we need:
- fully-async training with bounded staleness control, and/or
- one-step-off pipelined generation + training.

This plan focuses on integrating **SkyRL-train** into SchedRL as a supported framework.

SkyAgent integration (agentic multi-turn, e.g., SWE + OpenHands) is **currently on-policy only** in this repo.
Async SkyAgent (SWE + OpenHands) is noted as **future work**.

## 2) What SkyRL-train already provides

### 2.1 One-step-off async (pipelined)

SkyRL-train provides a minimal “one-step-off” async trainer that overlaps:
- generation of the next batch, and
- training on the previous batch.

Reference implementation:
- Entrypoint: `SkyRL/skyrl-train/examples/async/main_async.py`
- Trainer: `SkyRL/skyrl-train/examples/async/async_trainer.py` (`AsyncRayPPOTrainer`)
- Doc: `SkyRL/skyrl-train/docs/tutorials/one_step_off_async.rst`

Important constraints:
- `trainer.placement.colocate_all=false` (asserted in the async trainer).

### 2.2 Fully-async training (streaming / bounded staleness)

SkyRL-train provides a fully-async trainer with a staleness budget:
- knob: `trainer.fully_async.max_staleness_steps` in `SkyRL/skyrl-train/skyrl_train/config/ppo_base_config.yaml`
- implementation: `SkyRL/skyrl-train/skyrl_train/fully_async_trainer.py` (`FullyAsyncRayPPOTrainer`)
- entrypoint: `SkyRL/skyrl-train/examples/fully_async/main_fully_async.py`
- doc: `SkyRL/skyrl-train/docs/tutorials/fully_async.rst`

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
- Fully-async maps to `update_policy=INFLIGHT` (within SkyRL-train’s own semantics):
  - trainer pauses generation, syncs weights, then resumes.
  - staleness control is governed by `trainer.fully_async.max_staleness_steps`.

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
    `(max(load) - min(load)) / max(queued_trajectories + inflight_trajectories, 1) <= 0.05`.

**Baseline validation (required)**
- Validate the `REQUEST_RETRY` safety invariant: do not execute stateful env/tool side effects unless a non-abort generation result is received (single-writer commit).
  - For GSM8K-style tasks this is typically trivial (no stateful env).
  - For any SkyRL-Gym / agentic task with stateful tools, define the commit point explicitly before enabling mid-flight shrink.

## 3.3 Progress reporting (2% bands, trajectory counts)

SkyRL-train reports in trajectory units already (one `TrajectoryID` is one trajectory).

- Report `queued_trajectories` and `inflight_trajectories` separately.
- Keep the 2% cadence, with denominator = `policy_mini_batch_size` (trajectory units) for one training step.
  - `percent_remaining = (queued_trajectories + inflight_trajectories) / policy_mini_batch_size`
  - This may be > 100% if the backlog is larger than one step.

Version tagging (simple, for debugging):
- Record `generation_checkpoint_version` when the first turn of a trajectory is submitted, and record it again when the last turn finishes.

## 4) How to run today (use existing examples)

For now, treat SkyRL-train as a supported pipeline by using its existing example entrypoints:
- One-step-off: run `SkyRL/skyrl-train/examples/async/main_async.py`
- Fully-async: run `SkyRL/skyrl-train/examples/fully_async/main_fully_async.py`

This provides the async training modes without requiring custom SchedRL adapter code immediately.

## 5) Future work: SkyAgent SWE + async training

SkyAgent has a SkyRL-train integration entrypoint today:
- `SkyRL/skyrl-agent/skyrl_agent/integrations/skyrl_train/skyrl_train_main.py`

But it uses a sync-style trainer wrapper:
- `SkyRL/skyrl-agent/skyrl_agent/integrations/skyrl_train/trainer.py` (`SkyRLAgentPPOTrainer`)

So, for SkyAgent SWE + OpenHands tasks, the current SkyRL-train integration should be treated as:
- `update_policy=QUIESCE` / on-policy style (generate → train → sync),
- no one-step-off pipelining, and
- no fully-async staleness-controlled training.

To support async modes for SkyAgent SWE:
- Add new async entrypoints under `SkyRL/skyrl-agent/skyrl_agent/integrations/skyrl_train/` that mirror:
  - `SkyRL/skyrl-train/examples/async/main_async.py` (one-step-off), and
  - `SkyRL/skyrl-train/examples/fully_async/main_fully_async.py` (fully-async).
- Add corresponding trainer classes that reuse SkyRL-train’s async training loops but keep SkyAgent’s generator.

This is intentionally postponed until the base SkyRL-train async modes are validated in SchedRL first.
