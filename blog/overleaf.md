---
title: "Time-Sharing a GPU Cluster Across RL Pipelines: Decoupling Recipes from GPU Scheduling"
goal: for blog post and research paper
---
## Terminology
todo cross check with nvidia terminology 
w
https://docs.nvidia.com/nemo/gym/latest/about/concepts/key-terminology.html


We define the following standard terminology to clarify concepts across different RL frameworks.

| Concept Definition | Recommended Term | SchedRL (Draft) | NeMo-RL | SkyRL | ROLL | Miles |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **System Scope**<br>An independent RL training run. | **RL Job** | Pipeline | Job | System | Pipeline | Run |
| **Global Driver**<br>The main loop managing phases. | **Job Coordinator** | Pipeline Coordinator | Alg Loop | Controller | Pipeline.run | Script |
| **Resource Scheduler**<br>Centralized service allocating GPUs. | **Global Scheduler** | Global Scheduler | *(N/A)* | *(N/A)* | *(N/A)* | *(N/A)* |
| **Generation Role**<br>Component running agent logic + tool use. | **Agentic Rollout Worker** | Generator | Data Gen Process | Generator | TrajEnvManager | Rollout Function |
| **Training Role**<br>Component computing gradients/updates. | **Trainer** | Trainer | Trainer | Trainer | actor_train | Trainer |
| **Inference Layer**<br>LLM serving engine (vLLM/SGLang). | **Inference Engine** | *(N/A)* | Policy Model | InferenceEngine | actor_infer | EngineAdapter |
| **Environment**<br>Simulator interface (Gym/ALE). | **Environment** | *(N/A)* | Task / Env | Environment | Environment | Environment |
| **Data Artifact**<br>A complete interaction sequence. | **Trajectory** | Rollout | Trajectory | Trajectory | Trajectory | Sample |
| **Input Unit**<br>Batch of prompts for parallel rollout. | **Prompt Batch** | Request | Prompt Group | Batch | Batch | Data Batch |
| **Logical Cluster**<br>A named group of distributed actors. | **Worker Group** | Worker Cluster | RayWorkerGroup | Actor Group | Cluster | Replicas |
| **Trajectory Buffer**<br>Stores trajectories before training. | **Trajectory Buffer** | RolloutBuffer | ReplayBuffer | Buffer | GroupQueueManager | Buffer |
| **Router**<br>Dispatches LLM requests to engines. | **Request Router** | RolloutRouter | *(Implicit)* | Client | RequestScheduler | Router |
| **Physical Infra**<br>Physical machines and devices. | **GPU Cluster** | GPU Cluster | Cluster | Cluster | Cluster | Cluster |
| **Managed Infra**<br>Subset of GPUs managed by scheduler. | **GPU Pool** | GPU Pool | *(N/A)* | *(N/A)* | *(N/A)* | *(N/A)* |
| **Resource Unit**<br>Atomic allocation unit (multi-GPU). | **DP Worker** | DP Worker | Rank | Worker | Worker | Replica |
| **Parallelism Size**<br>Number of GPUs per DP Worker. | **TP Size** | TP Size | TP Size | TP Size | TP Size | TP Size |
| **Group Manager**<br>Controller for a Worker Group. | **Group Controller** | Cluster Controller | Wrapper | Manager | WorkerGroup | *(Implicit)* |
| **User Logic**<br>Config + Code defining the task. | **Training Recipe** | Training Recipe | Script | Script | Config | Script |
| **Sync Action**<br>Update worker weights from trainer. | **Sync Weights** | Sync | Refit | Sync | Sync | Sync |
| **Swap Action**<br>Move GPU state to CPU RAM. | **Swap** | Offload | Sleep | Offload | Offload | *(N/A)* |
| **Release Action**<br>Free GPU memory fully. | **Drop** | Drop | *(N/A)* | *(N/A)* | *(N/A)* | *(N/A)* |
| **Gen Phase**<br>Phase for collecting data. | **Rollout Phase** | Generation Phase | Collection | Rollout | Rollout Loop | Rollout |
| **Train Phase**<br>Phase for updating the model. | **Training Phase** | Training Phase | Training | Optimization | Training | Training |


## Outline

- 1) Introduction
  - The problem to solve
    - Background: LLM development focus is shifting from pre-training to RL post-training, and RL training is a multi-phase loop.
    - Agentic sampling is becoming longer-horizon and long-tailed:
      - Episodes can span many turns with retries; a few stragglers dominate wall time.
      - Straggler effect: the system often waits for the slowest episode before it can advance the phase, which wastes GPUs.
    - Concurrency is the default: agentic RL is no longer one job.
      - Research workflow: parallel experiments across configs/algorithms/base models/datasets/hyperparameters.
      - Multi-tenant service: tuning APIs (e.g., Tinker, OpenAI RFT) run many user jobs at once.
    - Shared-GPU reality: the same GPU cluster is shared across these concurrent RL jobs.
    - Core requirement: strong programmability/usability for researchers and high hardware utilization.
  - Existing approaches (and why they fall short)
    - Approach 1: fixed GPUs per RL job
      - Each RL job gets a fixed GPU budget for its roles.
      - Some systems time-share within a job across phases by running async, but that often changes the algorithm and adds staleness.
      - Problem: phase bubbles and long-tail trajectories make the best GPU split change over time, so fixed budgets waste capacity.
    - Approach 2: two disaggregated fixed pools by phase (RollMux as an example)
      - Keep a rollout pool and a training pool, then time-share across RL jobs within each pool.
      - Problem: the two pools still need careful balancing as rollout dynamics change; one pool can overload while the other idles.
    - Approach 3: one shared homogeneous pool with a single global loop (strawman)
      - Put all RL jobs and all phases under one monolithic controller with a global view.
      - Pro: good utilization potential and simple for engineers to implement in one place.
      - Con: the controller couples scheduling and execution, becomes a bottleneck, does not scale well, and is hard to maintain and hard for
        researchers to program and evolve.
  - Transition: the more common reality is one shared homogeneous pool
    - Many teams do not have (or do not want) two dedicated rollout/training clusters; they have one shared pool of similar GPUs.
    - In that setting, rollout and training for many RL jobs all contend for the same GPUs.
    - The dilemma: GPUs should be shared as much as possible across stages, but each RL job's training logic is independently managed by
      researchers.
    - The punch line: you need a Global Scheduler, with decentralized per-job training logic.
  - Our solution: one homogeneous pool shared by rollout and training
    - Treat the cluster as a single homogeneous GPU pool; time-share it across phases and RL jobs.
    - Decouple per-job logic from GPU scheduling (policy vs execution).
    - Semantics note: time-sharing does not change the training algorithm/semantics; behavior is equivalent to resource-isolated training (exclusive GPUs, time-sliced).
  - Quick comparison (four common approaches)

    | Approach | Hardware utilization | Researcher usability |
    |---|---|---|
    | Fixed GPUs per RL job, per-job controller | ❌ Low: long-tailed rollout introduces an on-policy (stability) vs utilization tradeoff | ✅ High: isolated, easy to program and evolve per job |
    | Separate rollout/training pools, time-sharing within each pool | ⚠️ Medium: good within each pool, but needs rebalancing across pools | ❓ Implementation dependent |
    | Single global controller for one shared pool | ⚠️ High potential, but the controller limits scalability | ❌ Low: job logic is tightly coupled to centralized scheduling |
    | Our approach: one shared pool, Global Scheduler coordinates multiple RL jobs | ✅ High: global sharing across phases and RL jobs | ✅ High: job logic stays independent while the scheduler handles sharing |
  - What makes it hard
    - Coordination across RL jobs and the scheduler: job coordinators and the Global Scheduler are separate distributed components, so you need a clear
      protocol for GPU ownership, preemption, and release.
    - Scheduling mixed workloads: many tasks (training/value/logprob) are non-preemptible; rollout is preemptible but has dynamic demand and stragglers, and there
      are cross-stage dependencies, which makes it hard to keep the system busy.
    - Model sync and memory pressure: CPU/GPU memory is shared across RL jobs and limited, naive syncing/caching can blow up memory easily.

- 2) Architecture: decouple GPU scheduling from job coordination
  - Hierarchical control:
    - Global Scheduler: decides GPU allocation across RL jobs (global policy).
    - Job Coordinator (per RL job): manages phase progression and requests GPUs.
    - Group Controller (per worker group): turns allocations into DP Worker actions (expand/shrink/preempt/resume) and lifecycle ops (load/swap/run).
    - DP Workers: directly own and use TP-sized GPU bundles to execute compute.
  - Execution plane: DP Workers do the compute; group controllers provide the mechanics (preempt/resume/load/swap/run).
  - Design sketch (top = RL jobs, bottom = GPUs):
    ```text
    RL Jobs (N independent runs)
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │ RL Job A                      │   │ RL Job B                      │
    │  - Job Coordinator            │   │  - Job Coordinator            │
    │  - Group Controller(s)        │   │  - Group Controller(s)        │
    │  - Training groups            │   │  - Training groups            │
    │  - Worker groups              │   │  - Worker groups              │
    │  - Request Router             │   │  - Request Router             │
    │  - Trajectory Buffer          │   │  - Trajectory Buffer          │
    └───────────────┬───────────────┘   └───────────────┬───────────────┘
                    │ request/release GPUs                           │
                    └──────────────────────────┬─────────────────────┘
                                               v
                             ┌─────────────────────────────────┐
                             │       Global Scheduler     │
                             └──────────────────┬──────────────┘
                                                v
                             ┌─────────────────────────────────┐
                             │            Shared GPU pool      │
                             └─────────────────────────────────┘
    ```
  - Coordination protocol:
    - Training groups (trainer-side compute: train/value/logprobs): execution is driven by the job coordinator (request → run → `release_gpus()`),
      i.e. the scheduler allocates but does not preempt the computation once granted.
    - Worker group (generator-side compute: `actor_infer` rollouts): preemptible DP Workers and completion-driven release:
      - Scheduling is driven by the Global Scheduler: it can preempt/resume DP ranks while rollouts run.
      - Allocation is at DP/TP-bundle granularity and may be partial.
      - After each training step, the job coordinator releases (or is preempted from) agentic rollout DP Workers, so those GPUs can be reclaimed and reused.
      - Workload rebalance on preemption/resume: `shrink_workers()` / `expand_workers()` rebalance routing by aborting in-flight prompt batches and clearing sticky mappings,
        so affected work is retried and naturally re-routed to the remaining/new DP Workers (“migration by abort + remap”).
      - Protocol invariants:
        - Ownership: each GPU id is either idle or owned by exactly one worker group allocation (never both).
        - Allocation unit: `DP Worker` (one DP rank) is the atomic preemption/activation unit; each DP Worker consumes `tp_size` GPUs.
        - Per-training-step preemption: the scheduler can safely preempt generation only at training-step boundaries.
    - Model-sync coordination (selective sync-on-resume):
      - After each training step, the scheduler can suspend/preempt agentic rollout DP Workers, so they are stopped and safe to sync new weights on resume.
      - Later, when the scheduler resumes `actor_infer`, it integrates model sync into scheduling by calling
        `expand_workers(selective_update=True)`: only the re-activated DP ranks sync the latest weights during resume (no push-to-all).
      - Versioning: keep only the latest weights cache, keyed by `global_step`; resume passes `requested_global_step=latest` and only newly activated DP ranks sync/apply.
  - Control-plane component discovery:
    - Job components and the Global Scheduler are Ray actors, discovered by name within a job-specific Ray namespace
      (Global Scheduler / Request Router / Trajectory Buffer; code: scheduler/request-scheduler/group-queue-manager).
  - Workflow diagram
    ```mermaid
    sequenceDiagram
    participant GS as GlobalScheduler
    participant RCP as Job Coordinator (per job)
    participant W as Group Controllers + DP Workers (incl. Request Router/Trajectory Buffer)

      RCP->>GS: request_gpus(training group, priority)
      GS-->>RCP: allocated GPUs
      RCP->>W: run training compute (train/value/logprobs)
      RCP->>GS: release_gpus(training group)

      RCP->>GS: request_gpus(worker group (actor_infer), rollout phase)
      GS->>W: resume/preempt DP ranks via expand_workers(...) / shrink_workers(...) (preemptible)
      W->>W: (re)activate DP ranks, run rollout

      RCP->>W: release generation GPUs (blocking)
      W->>GS: notify_group_released() (blocking ACK)
      GS->>W: preempt to reclaim GPUs
    ```

- 3) Scheduling heterogeneous, dynamic workloads: priority + FIFO + progress-aware allocation
  - Priority tiers are ordered by “closeness to end of a training step” (a topological sort of the per-training-step dependency graph):
    - Higher-priority groups are closer to producing a completed training step (e.g., training/logprob/value), so scheduling them first reduces training-step tail latency.
    - The worker group/rollout is lowest priority and preemptible.
    - Within the same priority tier, we use FIFO (older request first).
    - This approximates shortest-remaining-job-first scheduling.
  - Heartbeat-driven monitoring keeps it lightweight: rollout workers send progress heartbeats (`remaining`, `percent_remaining`, `oldest_unfinished_creation_ts`), so the
    scheduler can plan using backlog signals without per-episode polling, reducing scheduler overhead while staying responsive to workload changes.
    - Metric meanings + reporting cadence:
      - `remaining`: how many prompt batches are still missing for the current batch (`remaining = total_required - collected`).
      - `percent_remaining`: remaining ratio in `[0, 1]` (`percent_remaining = remaining / max(total_required, 1)`), not a true percentage.
      - `oldest_unfinished_creation_ts`: timestamp of the oldest unfinished prompt batch; used as a FIFO tie-break within a priority tier.
      - Heartbeat cadence is event-driven: send at batch start, and whenever `percent_remaining` crosses a 2% progress band (`floor(percent_remaining * 50)` changes).
      - Under preemption: agentic rollout DP Workers may be shrunk; in-flight prompt batches on shrinking workers are aborted and remapped to active workers, so progress can pause
        until retries complete (no “fake” progress from aborted work).
  - Planning rule (high-level pseudocode):
    ```text
    # Inputs:
    #   pending_requests: (group_id, priority, timestamp)
    #   generation_progress[job]: remaining, percent_remaining, oldest_unfinished_creation_ts
    #   active_allocations for generation: active_dp_workers (DP rank bundles), inactive_dp_workers
    #
    # Step 1: Non-generation planning (non-preemptible)
    for req in sort(pending_non_generation, by=(priority asc, timestamp asc)):
        if enough idle GPUs for req: allocate full device_mapping
        else: shrink agentic rollout DP Workers (lowest priority) to free DP bundles, then allocate if possible
    #
    # Step 2: Generation planning (preemptible, gap-ratio)
    weight[p] = remaining[p] * tp_size[p]
    target_ratio[p] = weight[p] / sum(weight)
    existing_ratio[p] = active_gpu[p] / gen_budget_gpus
    gap[p] = target_ratio[p] - existing_ratio[p]
    while exists receiver with gap>0 and (idle GPUs or shrinkable donors exist):
        pick receiver with max normalized_gap
        activate one inactive DP Worker bundle for receiver
          - if bundle GPUs not idle: pick donor DP bundles from jobs with gap<0 (within shrink budget)
            prefer donors with larger percent_remaining (least progress) to minimize wasted work
          - plan: shrink donor bundles, expand receiver bundle
    ```
  - Gap-ratio allocation happens at DP Worker granularity: DP Workers are fixed `tp_size` GPU bundles with fixed placement, so the scheduler resumes/preempts whole DP
    Workers to match backlog (`remaining * tp_size`). When shrinking donors, we preferentially preempt trajectories with the least progress (highest `percent_remaining`).


- 4) Model syncing and memory management optimization: selective + sync-on-resume
  - Problem: broadcasting new weights to all rollout workers right after each training step is unsafe and wasteful. Some rollout GPUs may be busy with other work, so an
    immediate sync can easily trigger OOM; but allocating the full rollout GPUs just to do lightweight weight syncing is slow and overkill.
  - Solution:
    - Keep a single “latest weights” cache (CPU buckets) per job; only designated sender ranks build/own this cache (one cached copy per job).
    - Use bucket-based broadcast on the sender side: keep all trainer states swapped to CPU, and only stage one small weight bucket on GPU at a time during sync.
    - Sync selectively on demand during scheduling: when `actor_infer` resumes, `expand_workers(selective_update=True)` syncs lastest weights only for the re-activated DP
      ranks, avoiding unnecessary syncs and memory blow-ups on rollout workers.
    - When `actor_infer` is preempted, workers aggressively drop footprint (weights + KV cache) instead of keeping inactive copies around.

----------

- 5) Evaluation
  - Goal: beat static per-job partitioning and single-job-only runs in number of gpus per job while maintaining the comparable training speed.
  - Workload: Terminal Bench
  - Throughput metric: `training steps / hour / GPU` (steps normalized by total GPUs used by all jobs).
  - Scale settings:

    | Model | Total GPUs | RL Jobs | GPUs / job |
    | --- | ---: | ---: | ---: |
    | Qwen3 14B | 8 | 1 | 8 |
    | Qwen3 14B | 8 | 2 | 4 |
    | Qwen3 14B | 12 | 4 | 3 |
    | Qwen3 30A3 | 64 | 1 | 64 |
    | Qwen3 30A3 | 64 | 2 | 32 |
    | Qwen3 30A3 | 96 | 4 | 24 |

  - Plots: the results show that end-to-end time per training step and rollout time per training step remain comparable across different job counts for the same model, indicating ~3× throughput improvement without slowing training.


30a3 1 2 4 jobs terminal-bench

rollout time per training step:
setup: grpo 30a3 64gpu
data:
 time/rollout

![alt text](./fig/30a3-terminal-bench-rollout.png)

end to end time per training step:
data:
 time/per_step_e2e

![alt text](./fig/30a3-terminal-bench-endtoend.png)

14b 1 2 4 jobs terminal-bench

rollout time per training step:
setup: grpo 14b 8gpu
data:
 time/rollout

![alt text](./fig/14b-terminal-bench-rollout.png)

end to end time per training step:

data:
 time/per_step_e2e

![alt text](./fig/14b-terminal-bench-endtoend.png)

- 6) What works today, and what’s next
  - Current stable parts and rough edges.
    - Tested mainly with a fixed set of RL jobs (system can support dynamic jobs, but that is not fully tested yet).
    - GPU placement / device_mapping for each task group is manually constructed today (not yet auto-placed).
  - Next steps:
    - Optimize task placement policy (GPU bundles / topology-aware placement).
    - Tighter integration with hyperparameter tuning (tuning loop as a first-class workload).
    - Keep evolving the scheduling policy (better priorities, fairness, progress signals, and efficiency).
    - Production deployment as a service for dynamic, user-programmable training jobs.
    - Improve admission control for dynamic jobs(avoid overload, enforce quotas/SLOs).
    - Cross-framework compatibility (interoperability at the scheduling layer): run jobs from different RL frameworks( https://www.anyscale.com/blog/open-source-rl-libraries-for-llms) on the same cluster without rewriting them (OS-style time-sharing across processes)
    - integrate multi-lora adapters per job

## extra content for paper submission
This is extra content for paper submission beyond the content for blog post

- [ ] measure the model update overhead and compare with the baseline
- [ ] measure the memory overhead in GPU, unswapped part
- [ ] Sanity-check quality: compare reward/success curves vs resource-isolated baseline (same configs, fixed seeds if possible).
- [ ] Baselines to report:
  - Static partition (fixed GPUs per job)
  - Sequential jobs (run N jobs one-by-one)
- [ ] Minimal ablations:
  - No preemption
  - No progress-aware planning
  - Full-sync instead of selective sync


## title candidates
### paper style
Scaling Concurrent RL by Decoupling GPU Scheduling from Training Orchestration

(Scalable is not the major benefits?  but it is a important() benefits for large systems)

Centralized GPU Scheduling with Decentralized Training Orchestration for Scaling Concurrent RL
Decoupling RL Job Execution from GPU Scheduling in Scalable RL Systems

"Decoupling Hardware Allocation from Decentralized RL Job Execution"


**Option B: Principle-Focused**
"Decoupling RL Job Control from GPU Scheduling in Concurrent RL"

**Option C: With Subtitle (Connects to Blog)**
Title: "Centralized GPU Scheduling with Decentralized Training Jobs for Concurrent RL"
Subtitle: "Decoupling Hardware Allocation from Training Job"

### blog style

**Recommended:**
"How We Time-Share GPUs Across RL Jobs Without Changing Training Recipes"

**Alternatives:**
- "Running 4× More RL Experiments on the Same GPU Cluster Without Slowing Down"
- "Centralized Scheduling, Decentralized Recipes: Scaling Concurrent RL Runs"
- "How We Time-Share One GPU Cluster Across Multiple RL Jobs"


## Backlog (future writing)

-----

pass 2

- P0 Terminology + structure
  - Issue: mixed taxonomy (`GEN`/`generation`/`ROLLOUT`; recipe/training logic/job loop; group meanings; DP Worker vs DP rank).
    - Done:
      - Added `## Terminology` after Section 1 and standardized doc language to `generation/training`, `job coordinator`, and `Global Scheduler`.
  - Issue: inconsistent title/subtitle wording (“GPU scheduling” vs “cluster scheduling”).
    - Done:
      - Standardized on `GPU scheduling` in title/subtitle and updated the Section 2 header to match.

- P0 Protocol correctness (what is guaranteed)
  - Issue: trajectory preemption/release protocol lacks explicit invariants and failure handling.
    - Done:
      - Added a “Protocol invariants” block under Section 2 worker group (ownership, DP-worker atomicity, per-training-step preemption before sync/resume).
  - Issue: model-sync versioning is not stated (how resumed ranks get the latest weights).
    - Done: added a single “Versioning” bullet under Section 2 model-sync coordination; the system keeps only the latest `global_step` cache.

- P1 Scheduling policy clarity (define the policy, not just name it)
  - Issue: “SRJF-like + FIFO + gap-ratio” is asserted without a definition or rationale.
    - Done:
      - Added a “Priority tiers” explanation and simple planning pseudocode in Section 3 (priority+FIFO for non-generation, gap-ratio for generation).
  - Issue: progress signals (`remaining`, `percent_remaining`, `oldest_unfinished_creation_ts`) are not sourced.
    - Done:
      - Added “Metric meanings + reporting cadence” under Section 3, including 2% progress-band reporting and preempt abort/remap behavior.

- P0/P1 Evaluation completeness
  - Issue: workload mismatch (Terminal Bench main text vs Sokoban appendix).
    - Done:
      - Moved Sokoban results to the appendix; keep main Evaluation focused on Terminal Bench.
      - Add a one-line caption per figure: model, total GPUs, jobs, metric.
  - Issue: throughput claim lacks baselines and causality (which technique causes which gain).
    - Done:
      - Define “throughput” as `training steps / hour / GPU` (at fixed `prompt_batch_size` and comparable configs).
      - Baselines + ablations moved to `## extra content for paper submission`.
  - Issue: quality/convergence not discussed but “without slowing training” can be misread as “no quality impact”.
    - Done:
      - Added a “Semantics note” in Section 1 (under “Our solution”) clarifying the training semantics are unchanged (equivalent to resource-isolated training).
      - Added a paper-only TODO to sanity-check quality vs a resource-isolated baseline.

- P2 Polish
  - Issue: draft markers (`status: draft`, inline TODO comment) and “step” ambiguity.
    - Done:
      - Removed `status: draft` and the `<!-- rvst ... -->` comment.
      - Clarified “step” usages as `training step` in the main outline.

---
pass 1

## Appendix

experiment results for sokoban

4B 1 2 4 jobs  sokoban
![alt text](./fig/4b-sokoban-4gpu.png)
