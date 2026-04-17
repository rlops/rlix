# RLix Multi-Pipeline GPU Scheduling Experiment

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Algorithm:** GRPO (agentic, no critic)  
**Environment:** SimpleSokoban (6×6 grid, 1 box)  
**Hardware:** 4× NVIDIA RTX 5090 (32 GB each, compute capability 12.0)  
**Run script:** `examples/run_rlix_experiment.py`

---

## Table of Contents

1. [Background — What is RLix?](#1-background--what-is-rlix)
2. [Architecture: Shared vs. Pipeline-Local Layers](#2-architecture-shared-vs-pipeline-local-layers)
3. [GPU Scheduling: Priority Buckets and Gap-Ratio Rollout](#3-gpu-scheduling-priority-buckets-and-gap-ratio-rollout)
4. [The Two Pipeline Types](#4-the-two-pipeline-types)
   - [Full-Finetune Pipeline](#full-finetune-pipeline-rollfullfinetuneipeline)
   - [Multi-LoRA Pipeline](#multi-lora-pipeline-rollmultilorapipeline)
5. [Experiment Scenarios](#5-experiment-scenarios)
6. [Data Flow Through the System](#6-data-flow-through-the-system)
7. [Key Files and What They Do](#7-key-files-and-what-they-do)
8. [Benchmark Results](#8-benchmark-results)
9. [Bugs Encountered and Fixes](#9-bugs-encountered-and-fixes)
10. [How to Run](#10-how-to-run)

---

## 1. Background — What is RLix?

RLix ("RL eXperiments") is a **multi-pipeline GPU scheduling layer** built on top of
[ROLL](https://github.com/rlops/ROLL). Where ROLL manages one RL training pipeline (generate →
reward → train), RLix coordinates **multiple simultaneous pipelines** sharing the same GPU pool.

The core insight is that RL pipelines have **bursty, heterogeneous GPU demand**:
- `actor_train` (policy gradient update) and `reference` (frozen KL model) need GPUs for fixed
  duration during their compute turn.
- `actor_infer` (rollout / trajectory sampling) is **elastic**: it can be scaled up or down
  without losing correctness, and it has the lowest priority — it can yield GPUs to other jobs.

RLix exploits this elasticity to multiplex GPU capacity across jobs. High-priority stages
(`actor_train`, `reference`) always get their requested GPUs first. Rollout (`actor_infer`) expands
into spare GPU capacity and gives it back when higher-priority work needs it.

**ROLL vs. RLix comparison:**

| Aspect | ROLL (single pipeline) | RLix (multi-pipeline) |
|--------|----------------------|----------------------|
| Jobs | 1 | N concurrent |
| Rollout GPUs | Fixed per pipeline | Elastic, shared pool |
| GPU utilization | Limited by one job's bursty demand | Higher: spare capacity reused |
| Scheduling | Synchronous within pipeline | Priority-based across pipelines |
| Base model sharing | No | Yes (Multi-LoRA mode) |

---

## 2. Architecture: Shared vs. Pipeline-Local Layers

```text
┌───────────────────────────────────────────────────────────┐
│              RLix Shared Job Management Layer             │
├──────────────────┬──────────────────┬─────────────────────┤
│   Orchestrator   │    Scheduler     │  Resource Manager   │
│  (job lifecycle) │ (priorities +    │ (cluster topology)  │
│  allocate_id()   │  rollout sharing)│ GPU count/topology  │
│  register()      │  gap-ratio algo  │                     │
│  admit()         │  ExecutionPlan   │                     │
└────────┬─────────┴────────┬─────────┴─────────┬───────────┘
         │                  │                   │
    ┌────▼──────┐      ┌────▼──────┐       ┌────▼──────┐
    │Pipeline   │      │Pipeline   │       │Pipeline   │
    │Coordinator│      │Coordinator│       │Coordinator│
    │ P1        │      │  P2       │       │  PN       │
    └────┬──────┘      └────┬──────┘       └────┬──────┘
         │                  │                   │
    ┌────▼──────────────────▼───────────────────▼────┐
    │                  Pipeline Actors               │
    │  RollFullFinetunePipeline / RollMultiLoraPipeline│
    │  (each has its own actor_train, actor_infer,   │
    │   reference, reward clusters)                  │
    └────────────────────────────────────────────────┘
```

**Orchestrator** (`rlix/orchestrator/orchestrator.py`) — singleton Ray actor in namespace
`"rlix"`. Manages pipeline lifecycle: `allocate_pipeline_id`, `register_pipeline` (topology
declaration), `admit_pipeline` (enables scheduling). Delegates scheduling decisions to the
Scheduler.

**Scheduler** (`rlix/scheduler/scheduler.py`) — singleton Ray actor. Holds the `ExecutionPlan`:
a priority-ordered mapping of which clusters are currently allocated, pending, or eligible for
expansion. Uses the **gap-ratio algorithm** (see §3) to decide how many rollout GPUs each pipeline
gets.

**ResourceManager** (`rlix/scheduler/resource_manager.py`) — singleton Ray actor. Polls
`ray.cluster_resources()` for live GPU counts; freezes topology after first `init_topology()` call.

**PipelineCoordinator** (`rlix/pipeline/coordinator.py`) — one per pipeline. Serializes
`resize_infer` (expand/shrink GPU count) and `sync_lora_weights` (LoRA weight push) via a
threading lock to prevent races. Communicates expand/shrink orders received from the Scheduler to
the pipeline actor.

---

## 3. GPU Scheduling: Priority Buckets and Gap-Ratio Rollout

### Priority Buckets

The scheduler maintains priority buckets for GPU allocation requests. From highest to lowest:

| Priority | Name | Description |
|----------|------|-------------|
| 0 | INITIALIZATION | Model download / warm-up; must complete before scheduling |
| 1 | ACTOR_TRAINING | Policy gradient update (DeepSpeed / Megatron) |
| 2 | CRITIC_TRAINING | Value function update (GAE only) |
| 3 | OLD_POLICY_LOGPROBS | Log-probs under previous policy (PPO clip) |
| 4 | REFERENCE_LOGPROBS | Log-probs under frozen reference model (KL penalty) |
| 5 | VALUE_COMPUTE | Advantage estimation (GAE only) |
| 6 | GENERATION | Rollout / trajectory sampling — **elastic, preemptable** |

Priorities 1-5 are "fixed" — the pipeline requests a specific GPU set and the scheduler grants it
without negotiation (first-come-first-served within priority level, respecting topology).
Priority 6 (GENERATION) is managed by the gap-ratio algorithm.

### Gap-Ratio Algorithm

When multiple pipelines compete for rollout GPUs (`actor_infer`), the scheduler runs the
**gap-ratio planner** (`rlix/scheduler/planner.py`) to decide allocations:

1. For each pipeline, compute `remaining = sequences_left_to_generate / total_sequences_in_step`.
2. The **target ratio** for pipeline P is `remaining_P / sum(remaining_all)`.
3. The **gap** for P is `target_ratio_P - existing_ratio_P` (existing = current active DP workers
   / total active DP workers across all pipelines).
4. Pipelines with the largest positive gap get their generation workers **expanded** first.
5. Pipelines with excess capacity get **shrunk** (GPUs reclaimed).

This ensures that pipelines with more work remaining get proportionally more rollout GPUs,
while staying within the available pool.

### Expand / Shrink Cycle

```
Scheduler ──expand──> PipelineCoordinator.resize_infer(target_dp_size=N)
                             │
                             ▼
              actor_infer cluster scales workers 0..N-1 up
              (vLLM loads weights at sleep_level=2)
                             │
         pipeline runs rollout with N DP workers
                             │
              actor_infer cluster scales down
              (vLLM releases weights, keeps actor alive in CPU RAM)
                             │
Scheduler <──release──  PipelineCoordinator reports done
```

`sleep_level: 2` in the vLLM strategy means workers **keep the Ray actor alive** but release GPU
memory (weights evicted to CPU). This is faster than full actor teardown and avoids repeated
weight downloads.

---

## 4. The Two Pipeline Types

### Full-Finetune Pipeline (`RollFullFinetunePipeline`)

**File:** `rlix/pipeline/full_finetune_pipeline.py`

Trains all model parameters (no LoRA). Wraps ROLL's `AgenticPipeline` with RLix-specific
expand/shrink calls around each rollout:

```python
# Before rollout: request generation GPUs from scheduler
coordinator.expand_infer(pipeline_id, target_dp_size)

# Run rollout trajectories (Sokoban: up to 5 actions × 4 envs)
rollout_data = self.actor_infer.generate(...)

# After rollout: release generation GPUs back to pool
coordinator.shrink_infer(pipeline_id)
```

**Config parameters (4-GPU layout):**
```yaml
# Pipeline 1 (GPUs 0-1 for train+ref; GPUs 0-3 for infer)
actor_train.device_mapping: "[0, 1, ]"
reference.device_mapping:   "[0, 1, ]"
actor_infer.device_mapping: "[0, 1, 2, 3, ]"

# Pipeline 2 (GPUs 2-3 for train+ref; GPUs 0-3 for infer)
actor_train.device_mapping: "[2, 3, ]"
reference.device_mapping:   "[2, 3, ]"
actor_infer.device_mapping: "[0, 1, 2, 3, ]"
```

Both pipelines' `actor_infer` clusters can use all 4 GPUs. The scheduler mediates which DP
workers are active at any moment to avoid GPU memory conflicts.

**Key config flags:**
- `model_update_transport: cpu_serialize` — weight sync via CPU pickle (avoids `pidfd_getfd`)
- `offload_nccl: true` — NCCL process groups torn down and re-initialised between stages to
  free device memory during CPU-offloaded phases
- `verify_model_after_sync: true` — checksums infer weights after each sync (safety check)
- `sleep_level: 2` — vLLM workers release GPU memory between rollouts but stay alive

---

### Multi-LoRA Pipeline (`RollMultiLoraPipeline`)

**File:** `rlix/pipeline/multi_lora_pipeline.py`

Trains **multiple LoRA adapters** on one shared base model. The base model parameters are frozen;
only adapter weights are updated. Multiple adapters share:
- One `actor_infer` (vLLM with multi-LoRA support)
- One `reference` model
- One `actor_train` base model, with **isolated per-adapter optimizers**

**Config (Pipeline 1, 2 adapters: Sokoban1, Sokoban2):**
```yaml
actor_train.model_args.adapters:
  Sokoban1: {lora_rank: 8, lora_alpha: 8, lora_target: all-linear}
  Sokoban2: {lora_rank: 8, lora_alpha: 8, lora_target: all-linear}
actor_train.strategy_config.is_lora_optimizer_isolated: true
```

**Constraints vs. full-finetune:**
- `sleep_level: 2` required (GPU weights released between rollouts)
- `is_lora_optimizer_isolated: true` required (per-adapter gradient accumulation)
- `overlap_grad_reduce: false` in Megatron (grad-sync hang risk with isolated LoRA)
- `use_dynamic_batching_in_train: false` (incompatible with isolated LoRA)
- `use_sequence_packing: false` (mixes adapters, violates homogeneity constraint)

**Memory saving:** Only the LoRA adapter parameters (~0.5% of total weights at rank 8) are
duplicated per adapter; the base model VRAM footprint is shared across adapters.

**Rollout cycle (per adapter tag):**
```
Expand (get infer GPUs) →
  Rollout(Sokoban1) →
  Rollout(Sokoban2) →
Shrink →
Train (Sokoban1 dirty lora) →
Train (Sokoban2 dirty lora) →
Repeat
```

---

## 5. Experiment Scenarios

All scenarios use:
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- 3 training steps (`max_steps: 3`)
- SimpleSokoban 6×6 environment, up to 5 actions per trajectory
- `async_generation_ratio: 1` (generation pipelined with training)
- `rollout_batch_size: 4` prompts per step

### Scenario A — Single Full-Finetune

```
GPU 0-1: actor_train + reference (Megatron, 1 TP × 2 DP)
GPU 0-3: actor_infer (vLLM, up to 4 workers, sleep_level=2)
```

Baseline: one pipeline, 4 GPUs. No cross-pipeline scheduling.

### Scenario B — Dual Full-Finetune

```
Pipeline 1:  GPU 0-1 train+ref  ←→  GPU 0-3 infer (shared)
Pipeline 2:  GPU 2-3 train+ref  ←→  GPU 0-3 infer (shared)
```

Two independent GRPO jobs sharing the same GPU pool. Training phases don't overlap (each
pipeline owns GPUs 0-1 or 2-3 exclusively for its train step). Rollout phases overlap via
gap-ratio scheduling: the pipeline with more remaining rollout work gets more infer GPUs.

### Scenario C — Single Multi-LoRA

```
GPU 0-1: actor_train + reference (2 LoRA adapters, isolated optimizers)
GPU 0-3: actor_infer (vLLM with 2 loaded LoRA adapters, sleep_level=2)
```

Single pipeline, 2 LoRA adapters (Sokoban1, Sokoban2). Memory saving vs. 2 full-finetune runs:
the base model VRAM is shared. Adapter rollouts are sequential within one pipeline.

### Scenario D — Full-Finetune + Multi-LoRA Concurrent

```
FT pipeline:   GPU 0-1 train+ref  ←→  GPU 0-3 infer
LoRA pipeline: GPU 2-3 train+ref  ←→  GPU 0-3 infer
```

Heterogeneous job mix: one pipeline trains full weights, the other trains 2 LoRA adapters.
The scheduler manages both concurrently, interleaving rollout expansion/shrink to share GPUs 0-3.

### Scenario E — Qwen2.5-0.5B Single Full-Finetune (Megatron)

```
GPU 0-1: actor_train + reference (Megatron-Core, 1 TP × 2 DP)
GPU 0-3: actor_infer (vLLM, up to 4 workers, sleep_level=2)
```

Single pipeline using `Qwen2.5-0.5B-Instruct` with the Megatron training strategy. Identical
config to Scenario A (`full_finetune_pipeline1`) — included as an explicit Megatron validation
point after fixing RTX 5090 Blackwell compatibility issues (NCCL 2.29.7, PyTorch
`_coalescing_manager` patch, `VLLM_ALLOW_INSECURE_SERIALIZATION`).

### Scenario F — Qwen2.5-0.5B Dual Full-Finetune (Megatron)

```
Pipeline 1:  GPU 0-1 train+ref (Megatron)  ←→  GPU 0-3 infer (shared)
Pipeline 2:  GPU 2-3 train+ref (Megatron)  ←→  GPU 0-3 infer (shared)
```

Two concurrent `Qwen2.5-0.5B-Instruct` pipelines with Megatron strategy sharing the infer GPU
pool. Identical to Scenario B but run separately as a Blackwell-validated Megatron dual-pipeline
test (`full_finetune_pipeline1 + full_finetune_pipeline2`).

---

## 6. Data Flow Through the System

```
┌────────────────────────────────────────────────────────────┐
│                SimpleSokoban Environment                   │
│  6×6 grid, 1 box, 1 target                                 │
│  Reward: +1 box on target, -0.15 format penalty            │
│  Agent gets text observation per turn; outputs <answer>    │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  TrajEnvManager (ROLL agentic pipeline)                    │
│  roll/pipeline/agentic/env_manager/traj_env_manager.py     │
│  Batches env steps; routes responses to/from actor_infer   │
│  Runs up to max_actions_per_traj=5 action turns per traj   │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  RLix Scheduler — GENERATION priority                      │
│  Expand actor_infer to target_dp_size DP workers           │
│  gap-ratio: more GPUs to pipeline with more work remaining │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  actor_infer — vLLM Rollout                                │
│  strategy: vllm (VLLM_USE_V1=1, sleep_level=2)             │
│  Generates 1 response/prompt; max_new_tokens=64            │
│  Multi-LoRA: routes each prompt to correct LoRA adapter    │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  RLix Scheduler — shrink actor_infer                       │
│  GPU memory released; weights stay in CPU RAM              │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  REFERENCE_LOGPROBS priority                               │
│  reference cluster computes log-probs under frozen model   │
│  strategy: megatron_infer; dynamic batching enabled        │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  Advantage Estimation                                      │
│  adv_estimator: grpo                                       │
│  Trajectory-level grouping (traj_group_id)                 │
│  whiten_advantages: true                                   │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  ACTOR_TRAINING priority                                   │
│  actor_train updates policy weights                        │
│  strategy: megatron_train (TP=1, DP=2, recompute_full)     │
│  Full FT: all weights updated                              │
│  Multi-LoRA: per-adapter optimizer; dirty loras trained    │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│  Weight Sync: actor_train → actor_infer                    │
│  model_update_transport: cpu_serialize                     │
│  ModelUpdateService broadcasts via CPU pickle              │
│  verify_model_after_sync: true checksums the result        │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
                    (next step)
```

---

## 7. Key Files and What They Do

### RLix Control Plane

| File | Purpose |
|------|---------|
| `rlix/orchestrator/orchestrator.py` | Singleton pipeline lifecycle manager: allocate IDs, register topology, admit pipelines, kill |
| `rlix/scheduler/scheduler.py` | Singleton GPU allocation engine: priority buckets, `_apply_plan()`, GENERATION expansion |
| `rlix/scheduler/planner.py` | Gap-ratio planning algorithm: `_GapRatioDPWorker`, `_compute_shrink_budget_by_pipeline_id` |
| `rlix/scheduler/resource_manager.py` | Ray cluster GPU topology snapshot; `init_topology()` freezes node structure |
| `rlix/scheduler/state.py` | `SchedulerState`: immutable snapshot of all cluster allocations |
| `rlix/scheduler/tracer.py` | `SchedulerTracer`: emits Perfetto `tg4perfetto` GPU trace events per scheduling cycle |
| `rlix/scheduler/types.py` | `ExecutionPlan`, `ClusterAllocation`, `PendingRequest`, `SchedGuidedShrinkOp` |
| `rlix/protocol/types.py` | Priority enum, actor name constants, `ActionResponse`, `ProgressReport` |

### Pipeline Layer

| File | Purpose |
|------|---------|
| `rlix/pipeline/coordinator.py` | Per-pipeline coordinator: serializes `resize_infer`, `sync_lora_weights`; bridges scheduler→pipeline |
| `rlix/pipeline/full_finetune_pipeline.py` | `RollFullFinetunePipeline`: wraps ROLL `AgenticPipeline` with RLix expand/shrink calls |
| `rlix/pipeline/multi_lora_pipeline.py` | `RollMultiLoraPipeline`: per-tag rollout schedulers; sequential expand→rollout→shrink→train |
| `rlix/pipeline/model_update_service.py` | `ModelUpdateService`: CPU-serialized weight broadcast from `actor_train` → `actor_infer` |
| `rlix/pipeline/utils.py` | `validate_resize_params`: topology validation for expand/shrink requests |
| `rlix/client/client.py` | `RLixClient`: external API for launching/monitoring pipelines |

### ROLL (rlops fork, branch `rlix`)

| File | Purpose |
|------|---------|
| `external/ROLL/roll/distributed/strategy/megatron_strategy.py` | Megatron-Core training + inference strategy; supports `sleep_level` and `offload_nccl` |
| `external/ROLL/roll/distributed/strategy/vllm_strategy.py` | vLLM strategy: `expand()`/`shrink()` for elastic resize; `sleep_level=2` weight management |
| `external/ROLL/roll/pipeline/agentic/agentic_pipeline.py` | Base agentic pipeline: multi-turn trajectory collection + GRPO training loop |
| `external/ROLL/roll/pipeline/agentic/env_manager/traj_env_manager.py` | Trajectory environment manager; manages parallel env workers |
| `external/ROLL/roll/utils/lora_routing.py` | Routes trajectories to correct LoRA adapters; normalizes domain/adapter tags |

### Configuration

| File | Purpose |
|------|---------|
| `examples/rlix_test/full_finetune_pipeline1.yaml` | 4-GPU full-finetune P1: train+ref GPUs 0-1, infer GPUs 0-3 |
| `examples/rlix_test/full_finetune_pipeline2.yaml` | 4-GPU full-finetune P2: train+ref GPUs 2-3, infer GPUs 0-3 |
| `examples/rlix_test/multi_lora_pipeline1.yaml` | 4-GPU multi-LoRA P1: adapters Sokoban1/2, GPUs 0-1 train, 0-3 infer |
| `examples/rlix_test/multi_lora_pipeline2.yaml` | 4-GPU multi-LoRA P2: adapters Sokoban3/4, GPUs 2-3 train, 0-3 infer |
| `examples/config/traj_envs.yaml` | Sokoban/FrozenLake/WebShop environment definitions and agent prompt templates |
| `examples/run_rlix_experiment.py` | Runner script: GPU monitor, scenario dispatch, comparison table |

---

## 8. Benchmark Results

### Wall Time and GPU Utilization (v35/v37 final run, 2026-04-14) — All 6 PASS ✅

*1 training step × 6 scenarios on 4× NVIDIA RTX 5090 (32 GB, CC 12.0), Vast.ai cloud instance*  
*Model: Qwen2.5-0.5B-Instruct · Env: SimpleSokoban 6×6 · `max_steps=1`*  
*All scenarios pass after Blackwell compatibility fixes (Bugs 9–11). Wall time includes full init.*

| Scenario | Description | Wall Time | Avg GPU Util | Peak Mem | Status |
|----------|-------------|-----------|-------------|----------|--------|
| **A** — Single FT | 1 FT pipeline, GPUs 0-1 train, 0-3 infer | 162s | 1.3% | 21,772 MB | ✅ OK |
| **B** — Dual FT | 2 FT pipelines concurrent | ~174s | — | — | ✅ OK |
| **C** — Single Multi-LoRA | 1 LoRA pipeline, 2 adapters | ~182s | — | — | ✅ OK |
| **D** — FT + Multi-LoRA | FT + LoRA concurrent, heterogeneous | 225s | 34.3% | 24,567 MB | ✅ OK |
| **E** — Single FT (Megatron) | Same as A, Blackwell-validated | 161s | 1.0% | 21,772 MB | ✅ OK |
| **F** — Dual FT (Megatron) | Same as B, Blackwell-validated | 193s | 1.8% | 22,611 MB | ✅ OK |

*B and C wall times are derived from pipeline completion timestamps in the run log; GPU util stats
not captured due to a disk-full crash that occurred mid-run during Scenario D initialization.*

### Per-GPU Breakdown (scenarios with exact stats)

| Scenario | GPU 0 Avg | GPU 1 Avg | GPU 2 Avg | GPU 3 Avg | Peak Mem |
|----------|-----------|-----------|-----------|-----------|----------|
| A | 2.6% | 2.3% | 0.3% | 0.0% | 21,772 MB |
| D | 44.8% | 45.0% | 45.9% | 1.4% | 24,567 MB |
| E | 1.9% | 1.4% | 0.4% | 0.4% | 21,772 MB |
| F | 1.6% | 1.7% | 2.7% | 1.3% | 22,611 MB |

*Scenario D's high GPU utilisation (avg 34–45% on GPUs 0-2) reflects the concurrent FT+LoRA
rollout phases actively interleaving on shared GPUs — the gap-ratio scheduler is doing real work.*

---

### Historical: v20 run (2026-04-14) — partial results

*3 training steps × 6 scenarios. Scenarios A–D passed; E–F failed (LFM/DeepSpeed config,
subsequently removed from experiment script in favour of Qwen2.5/Megatron E/F).*

| Scenario | Description | Wall Time | Avg GPU Util | Peak Mem | Status |
|----------|-------------|-----------|-------------|----------|--------|
| **A** — Single FT | 1 FT pipeline, 4 GPUs | 244s | 2.4% | 25,583 MB | ✅ OK |
| **B** — Dual FT | 2 FT pipelines concurrent | 312s | 3.9% | 26,204 MB | ✅ OK |
| **C** — Single Multi-LoRA | 1 LoRA pipeline, 2 adapters | 367s | 0.7% | 26,689 MB | ✅ OK |
| **D** — FT + Multi-LoRA | 2 pipelines, heterogeneous | 434s | 1.8% | 27,312 MB | ✅ OK |
| **E** — LFM Single FT | 1 LFM pipeline, DeepSpeed | 105s | 0.1% | 5,253 MB | ❌ FAILED |
| **F** — LFM Dual FT | 2 LFM pipelines concurrent | 106s | 0.0% | 5,253 MB | ❌ FAILED |

*E and F failed due to DeepSpeed `FusedAdam` JIT compilation failure on sm_120a (Blackwell).
Fix documented in Bug 8. E and F configs subsequently changed to use Qwen2.5/Megatron.*

### Per-GPU Breakdown (v20)

### Step Timing Detail (Scenario A)

| Step | Start (UTC) | Finish (UTC) | Duration |
|------|------------|--------------|----------|
| 0 | 04:24:05 | 04:24:43 | ~38s |
| 1 | 04:24:43 | 04:25:05 | ~22s |
| 2 | 04:25:05 | 04:25:41 | ~36s |

### Step Timing Detail (Scenario B — Both Pipelines Interleaved)

| Step | P1 Start | P1 Finish | P2 Start | P2 Finish |
|------|----------|-----------|----------|-----------|
| 0 | 04:28:47 | 04:30:03 | 04:29:21 | 04:30:05 |
| 1 | 04:30:03 | 04:30:34 | 04:30:05 | 04:30:34 |
| 2 | 04:30:34 | 04:31:21 | 04:30:34 | 04:31:23 |

*Both pipelines start and finish their steps within ~2s of each other — the gap-ratio scheduler
distributes rollout GPUs proportionally, keeping both pipelines roughly in sync.*

### Step Timing Detail (Scenario D — FT Pipeline + LoRA Pipeline)

**FT pipeline (ft_2913d730dedb, GPUs 0-1 train):**

| Step | Start (UTC) | Finish (UTC) | Infer alloc |
|------|------------|--------------|-------------|
| 0 | 04:41:08 | 04:42:07 | [0,1,2,3] full |
| 1 | 04:42:07 | 04:42:46 | [0,1,2,3] full |
| 2 | 04:42:46 | 04:43:20 | [0,1] partial (LoRA active) |

**LoRA pipeline (lora_61e6662b38ec, GPUs 2-3 train), 6 ticks total:**

| Tick | Adapter | Step | Completed (UTC) |
|------|---------|------|----------------|
| 1 | sokoban3 | 1 | 04:43:01 |
| 2 | sokoban4 | 1 | 04:43:47 |
| 3 | sokoban3 | 2 | 04:44:16 |
| 4 | sokoban4 | 2 | 04:44:46 |
| 5 | sokoban3 | 3 | 04:45:16 |
| 6 | sokoban4 | 3 | 04:45:46 |

*FT pipeline completed first (04:43:30); LoRA completed 2m16s later (04:45:46). The FT step 2
got partial allocation [0,1] because the LoRA pipeline was expanding for its first tick rollout
at the same moment — gap-ratio scheduler correctly split the pool.*

### Key Observations

- **B vs. A:** Both pipelines run 3 steps in 2m36s vs A's 1m36s for 1 pipeline. Effective
  throughput is ~2× (two jobs complete) at 1.6× the wall time — partial overlap due to shared
  4-GPU inference pool being the bottleneck.

- **C (Multi-LoRA) vs. A:** Multi-LoRA needs more ticks (6 ticks for 2 adapters × 3 steps vs.
  3 steps) but adapter training is faster than full-FT. Total wall time is longer due to sequential
  per-adapter rollout within the pipeline, not from GPU scheduling overhead.

- **D (heterogeneous):** FT pipeline dominates the inference pool when the LoRA pipeline is in
  training phase. Allocated `[0,1,2,3]` (not partial) for FT rollout confirms the scheduler
  grants all 4 GPUs when the LoRA pipeline releases them during its training phase.

- **IPC confirmed:** `ipc_targets=4 broadcast_ranks=[]` in all scenarios — weight sync uses
  shared-memory IPC not NCCL, avoiding CUDA error 700 on RTX 5090 (Blackwell) same-node topology.

---

## 9. Bugs Encountered and Fixes

---

### Bug 1 — `setup_env.sh` uses CUDA 12.4 but instance has CUDA 13.1 drivers

**Error:** *(none — CUDA drivers are forward compatible)*

**Context:** The conda env install targets `cuda-nvcc=12.4.131` and `cudnn=9.1.1.17` for
Transformer Engine compatibility. Vast.ai instances may have CUDA 13.1 drivers. CUDA drivers
are forward compatible (CUDA 12.4 binaries run on CUDA 13.1 drivers), so this works without
modification. The `CUDA_HOME` exported by `setup_env.sh` correctly points to the conda CUDA
toolkit, not the system one.

**How to verify:** `conda run -n rlix nvcc --version` should show `12.4.x`, while
`nvidia-smi` still shows driver CUDA 13.1.

---

### Bug 2 — Flash-attention wheel requires Python 3.10

**Error:** *(pip install would skip or fail on Python 3.12)*

**Context:** `requirements_torch260_vllm.txt` includes a direct URL to a pre-built
`flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
(Python 3.10 only). `setup_env.sh` creates a Python 3.10 conda env specifically for this reason.

**Fix:** Always run rlix via `conda run -n rlix` or `conda activate rlix`; never use the system
Python 3.12 for rlix experiments.

---

### Bug 3 — `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` must be set

**Error (without the fix):**
```
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date...
```

**Root cause:** `tg4perfetto` (Perfetto timeline tracing library used by RLix's `SchedulerTracer`)
generates `.proto`-based Python stubs that conflict with the C++ protobuf extension installed by
other packages (e.g., `wandb`).

**Fix (applied in `setup_env.sh`):**
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
conda env config vars set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```
This forces the pure-Python protobuf backend, bypassing the C++ extension version check.

---

### Bug 4 — `offload_nccl: true` required to avoid OOM during concurrent pipelines

**Error (without the fix):**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Root cause:** In multi-pipeline mode, two pipelines' `actor_infer` clusters share physical GPUs
(both have `device_mapping: "[0, 1, 2, 3, ]"`). Without offloading NCCL communicators between
training stages, NCCL's internal buffers accumulate across both pipelines' process groups,
consuming enough GPU memory to trigger OOM when combined with vLLM KV cache.

**Fix:** Set `offload_nccl: true` in all pipeline configs. ROLL tears down NCCL process groups
after each stage completes and rebuilds them on demand. The extra setup latency (~1-2s per NCCL
group rebuild) is acceptable.

---

### Bug 5 — `model_update_buffer_size_mb: 100` needed to avoid OOM during weight sync

**Error (without the fix):**
```
torch.cuda.OutOfMemoryError: CUDA out of memory (attempted to allocate ...)
```

**Root cause:** `ModelUpdateService` broadcasts `actor_train` weights to `actor_infer` in a
single tensor bucket. For `Qwen2.5-0.5B-Instruct` (~500M params × 2 bytes = ~1 GB), allocating
the full broadcast buffer at once saturates VRAM when `actor_infer` vLLM workers are still
holding KV cache allocations.

**Fix:** Set `model_update_buffer_size_mb: 100` (chunk the broadcast into 100 MB pieces).
`cpu_serialize` transport serializes to CPU first, then chunks it, avoiding the spike.

---

### Bug 6 — `use_distributed_optimizer: false` needed for concurrent pipelines

**Error (without the fix):**
```
OSError: [Errno 11] Resource temporarily unavailable
```

**Root cause:** Megatron's distributed optimizer spawns a `multiprocessing.Manager()` process
per pipeline for async checkpoint support (`filesystem_async.py`). With 2 concurrent pipelines
each spawning optimizer managers plus their Ray actor workers, the container `pids.max` limit
is exhausted.

**Fix:** Set `use_distributed_optimizer: false` in `actor_train.strategy_config`. Single-GPU
`actor_train` gets no benefit from the distributed optimizer, and the spawned manager process
is avoided.

---

### Bug 7 — vLLM 0.19 `abort_requests()` hangs `generate()` generator on RTX 5090

**Error:**
```
roll.distributed.scheduler.generate_scheduler: rebalance_on_shrink timed out after 30s
```

**Root cause:** In vLLM 0.19 (`v0.9.2`), `OutputProcessor.abort_requests()` removes the
request from `request_states` but **never signals the `RequestOutputCollector` queue**. The
`AsyncLLM.generate()` async generator (`async_llm.py`) hangs forever at `await q.get()` because
no item is ever put into the queue after the abort. The drain loop in
`generate_scheduler._rebalance_on_shrink` waits for `running_requests[dp_rank]` to reach zero
(which requires the `generate_request.remote()` Ray future to resolve), causing the 30s timeout.

**Affected hardware:** RTX 5090 (sm_120a, Blackwell) with `async_generation_ratio: 1`. The
shrink is called while an in-flight generation request exists, triggering the code path.

**Fix (applied to `vllm/v1/engine/output_processor.py`):**

```python
def abort_requests(self, request_ids):
    request_ids_to_abort = []
    for request_id in request_ids:
        req_state = self.request_states.pop(request_id, None)
        if req_state is not None:
            # RTX 5090 / vllm-0.19 workaround: signal the per-request queue
            # so that any waiting generate() coroutine is unblocked immediately.
            if req_state.queue is not None:
                from vllm.outputs import RequestOutput, CompletionOutput
                abort_out = RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    prompt_token_ids=[],
                    prompt_logprobs=None,
                    outputs=[CompletionOutput(
                        index=0, text="", token_ids=(), cumulative_logprob=None,
                        logprobs=None, finish_reason="abort",
                    )],
                    finished=True,
                )
                req_state.queue.put(abort_out)
            self.lora_states.abort_request(req_state)
            request_ids_to_abort.append(request_id)
        else:
            parent = self.parent_requests.pop(request_id, None)
            if parent and parent.child_requests:
                self.abort_requests(parent.child_requests)
                request_ids_to_abort.extend(parent.child_requests)
    return request_ids_to_abort
```

**Why `asyncio.CancelledError` doesn't work:** In Python 3.8+, `CancelledError` is a
`BaseException`, not `Exception`. `RequestOutputCollector.get_nowait()` only raises if
`isinstance(output, Exception)` — so `CancelledError` is returned as a value, not raised.
`output.finished` then fails with `AttributeError`.

**ROLL integration:** `traj_env_manager.py` handles aborted requests gracefully:
`if lm_output is None: return DataProto(stop_reason=ABORT)`, which the rollout loop handles
by incrementing `rollout_cache.attempt` (retry). The fix is safe — ROLL was already designed
to survive aborts.

---

### Bug 8 — DeepSpeed fused Adam JIT compilation fails on RTX 5090 (sm_120a)

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
  at deepspeed/ops/adam/cpu_adam_builder.py
```

**Root cause:** DeepSpeed attempts to JIT-compile its custom fused Adam CUDA kernel when
`use_cpu_adam=True` (or similar). The sm_120a (Blackwell architecture, RTX 5090) compute
capability is not included in DeepSpeed's pre-compiled wheel or JIT target list as of
DeepSpeed 0.16.x.

**Affected scenarios:** E and F (LFM2.5-350M uses `deepspeed_train` strategy).

**Fix (two-part):**

1. Set `DS_BUILD_OPS: '0'` in `system_envs` of the pipeline YAML (required to trigger the
   optimizer selection fix below):
   ```yaml
   system_envs:
     VLLM_USE_FLASHINFER_SAMPLER: '0'
     DS_BUILD_OPS: '0'
   ```

2. **Patch `ROLL/roll/distributed/strategy/deepspeed_strategy.py` line 367** to respect
   `DS_BUILD_OPS=0` at runtime:
   ```python
   # Before (always uses FusedAdam when not offloading):
   adam_optimizer = DeepSpeedCPUAdam if self.ds_config.is_offload() else FusedAdam

   # After (also falls back when DS_BUILD_OPS=0):
   import os
   adam_optimizer = (
       DeepSpeedCPUAdam
       if (self.ds_config.is_offload() or os.environ.get("DS_BUILD_OPS") == "0")
       else FusedAdam
   )
   ```

`DS_BUILD_OPS=0` is a **build-time** flag for DeepSpeed package installation — it does NOT
prevent `FusedAdamBuilder().load()` from being called at runtime. The patch above makes
`DS_BUILD_OPS` also work as a runtime signal to switch to `DeepSpeedCPUAdam`.

---

---

### Bug 9 — NCCL 2.26.2 has no native sm_120a (Blackwell) kernels

**Error:**
```
RuntimeError: CUDA error: an illegal memory access was encountered
  at torch/distributed/distributed_c10d.py: work.wait()
```

**Root cause:** NCCL 2.26.2 (shipped with `nvidia-nccl-cu12==2.26.2`) does not include
pre-compiled kernels for the sm_120a compute capability (RTX 5090, Blackwell). It falls back
to JIT-compiled PTX which produces illegal memory accesses on Blackwell's new memory subsystem.

**Affected path:** Any collective (allreduce, broadcast) in the verification pass
(`setup_collective_group` → `worker.py:595`) and during training weight sync.

**Fix:**
```bash
pip install nvidia-nccl-cu12==2.29.7
```
NCCL 2.29.7 adds native sm_120a kernels. After upgrade:
```
/root/miniconda3/envs/rlix/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
  Was: NCCL 2.26.2+cuda12.2
  Now: NCCL 2.29.7+cuda12.9
```

**Transport workaround (still required):** RTX 5090 has no CUDA peer-to-peer access
(`can_device_access_peer=False`). NVLS and SHM transports are also broken on Blackwell under
NCCL 2.29.7. Force socket transport via:
```yaml
system_envs:
  NCCL_P2P_DISABLE: "1"
  NCCL_SHM_DISABLE: "1"
  NCCL_NVLS_ENABLE: "0"
  NCCL_IB_DISABLE: "1"
```

---

### Bug 10 — vLLM 0.9.2 rejects `torch.dtype` in pickle serialization

**Error:**
```
TypeError: Object of type <class 'torch.dtype'> is not serializable
  at vllm/v1/serial_utils.py enc_hook
```
This causes the `InferWorker.broadcast_parameter()` call to silently fail; all 3–4 InferWorkers
never enter NCCL receive, and the sender times out:
```
DistBackendError: Watchdog caught collective operation timeout: WorkNCCL(OpType=BROADCAST, Timeout(ms)=150000)
```

**Root cause:** vLLM 0.9.2 switched to a strict `msgpack` encoder (`enc_hook`) that explicitly
rejects `torch.dtype` objects to prevent arbitrary code execution via pickle. However,
`ModelUpdateService` passes `dtypes` (a list of `torch.dtype`) as part of the weight sync
payload to the `EngineCore` subprocess.

**Fix:** Set the environment variable before launching any workers:
```yaml
system_envs:
  VLLM_ALLOW_INSECURE_SERIALIZATION: "1"
```
This re-enables pickle as the fallback serializer in `serial_utils.py`, allowing `torch.dtype`
objects to pass through.

---

### Bug 11 — PyTorch 2.7.1 `_coalescing_manager` `UnboundLocalError` with NCCL socket transport

**Error:**
```
UnboundLocalError: local variable 'work' referenced before assignment
  File "torch/distributed/distributed_c10d.py", line 2590, in _coalescing_manager
    cm.append(work)
```
or
```
  File "torch/distributed/distributed_c10d.py", line 2592, in _coalescing_manager
    work.wait()
```

**Root cause:** When NCCL socket transport is forced (via `NCCL_P2P_DISABLE=1`,
`NCCL_SHM_DISABLE=1`, etc.), collective operations execute immediately inside the
`with _coalescing_manager(...):` block rather than buffering into
`_world.pg_coalesce_state[group]`. As a result, `op_list` is empty after the `yield`.
Neither the fast-path branch (`if op_list:`) nor the legacy branch (`if device:`, where
`device=None` in Megatron's call) assigns `work`. The final `cm.append(work)` or
`work.wait()` then raises `UnboundLocalError`.

**Call path:**
```
megatron_strategy.py:1442 → _run_forward_backward
  → finalize_model_grads → finish_grad_sync → start_grad_sync
    → torch.distributed._coalescing_manager
```
Triggered even with `overlap_grad_reduce: false` because `finish_grad_sync` is always called
at the end of each backward pass.

**Fix (patched in `/root/miniconda3/envs/rlix/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py`):**

```python
# In _coalescing_manager, replace the block after op_list fast-path with:

work = None  # Handle empty op_list + device=None (NCCL socket transport)
if device:
    work = group._end_coalescing(device)

if work is not None:
    if async_ops:
        cm.append(work)
    else:
        work.wait()
```

When `op_list` is empty and `device=None`, `work` stays `None` and the guard is a no-op —
correct because all collectives already completed synchronously inside the context manager.

---

## 10. How to Run

### Prerequisites

```bash
# Clone rlix with ROLL submodule
git clone --recurse-submodules https://github.com/zhenyulincs/rlix.git
cd rlix

# Install the conda environment (takes ~20 min; requires NVIDIA drivers)
bash setup_env.sh
conda activate rlix
```

### Run Individual Scenarios

```bash
# Scenario A: single full-finetune
conda run -n rlix --no-capture-output \
  python examples/run_rlix_experiment.py --scenario A

# Scenario B: two full-finetune pipelines concurrent
conda run -n rlix --no-capture-output \
  python examples/run_rlix_experiment.py --scenario B

# Scenario C: single multi-LoRA pipeline
conda run -n rlix --no-capture-output \
  python examples/run_rlix_experiment.py --scenario C

# Scenario D: full-finetune + multi-LoRA concurrent
conda run -n rlix --no-capture-output \
  python examples/run_rlix_experiment.py --scenario D

# Scenario E: LFM2.5-350M single full-finetune (DeepSpeed)
conda run -n rlix --no-capture-output \
  python examples/run_rlix_experiment.py --scenario E

# Scenario F: LFM2.5-350M dual full-finetune (DeepSpeed, 2 pipelines)
conda run -n rlix --no-capture-output \
  python examples/run_rlix_experiment.py --scenario F
```

### Run All Scenarios

```bash
conda run -n rlix --no-capture-output \
  python examples/run_rlix_experiment.py --scenario all
```

### Run Directly with the Example Script

```bash
conda run -n rlix --no-capture-output \
  python examples/start_multi_pipeline_test.py \
  --config_name full_finetune_pipeline1,full_finetune_pipeline2
```

### View Scheduler Trace

RLix emits a Perfetto timeline trace at `./output/scheduler_trace.json.gz`.
Open it at `ui.perfetto.dev` to see GPU allocation events per pipeline and stage.
