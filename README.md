<div align="center">

<img src="assets/rlix-logo-text-horizontal.svg" width="40%" alt="RLix Logo">

<h3>A control plane for concurrent LLM RL on shared GPUs</h3>

<p>
  <a href="https://github.com/rlops/rlix/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>

  <a href="https://github.com/rlops/rlix/stargazers">
    <img src="https://img.shields.io/github/stars/rlops/rlix?style=social" alt="Repo stars">
  </a>

  <a href="https://github.com/rlops/rlix/issues">
    <img src="https://img.shields.io/github/issues/rlops/rlix" alt="GitHub issues">
  </a>

  <a href="https://deepwiki.com/rlops/rlix" target="_blank">
    <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
  </a>
</p>

</div>

In agentic RL, long-horizon rollouts are increasingly long-tailed: a small number of stragglers dominate wall-clock time while many GPUs sit underutilized.

RLix addresses this by time-sharing GPUs across concurrent RL training jobs, expanding rollout workers onto temporarily idle capacity and shrinking them when that capacity is needed elsewhere. RLix does not change pipeline-level training semantics: each recipe retains its original behavior, whether on-policy or off-policy under staleness bounds, while delivering much higher GPU utilization.

RLix builds on **Partial Overlapping** scheduling from [**Alibaba/ROLL**](https://github.com/alibaba/ROLL) and extends it with a distributed control plane for coordinating multiple independent training jobs on a shared GPU cluster.

RLix is an AI-native project, with AI deeply involved across design, planning, implementation, testing, and code review, alongside human oversight. Correctness, code quality, and maintainability remain first-class concerns.

## Features

- **Recipe-Transparent Scheduling**: GPU scheduling stays decoupled from training logic, so researchers can develop each pipeline independently while preserving its original recipe semantics.
- **Two-Level GPU Sharing**: GPUs are shared across pipelines through elastic expand/shrink and within pipelines through concurrent LoRA adapters on a shared base model.
- **Demand-Driven Rollout Scaling**: Rollout workers expand into idle GPU capacity and shrink in response to heartbeat-reported demand.
- **Minimal Rollout Memory Footprint**: Rollout workers load inference weights only when active. Weights are cached on the trainer CPU, synchronized on resume, and dropped when workers shrink.

## Installation

```bash
git clone https://github.com/rlops/rlix.git
cd rlix
pip install -e .
````

## Quick Start

The example below shows a minimal control-plane setup for registering and running a pipeline under RLix.

```python
import ray
import rlix
from rlix.pipeline import PipelineCoordinator
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX

# Pipeline-specific configuration object
my_config = ...

# 1. Initialize the RLix control plane
orchestrator = rlix.init(create_if_missing=True)

# 2. Allocate a pipeline ID
pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("ft"))

# 3. Register the pipeline's GPU topology
ray.get(
    orchestrator.register_pipeline.remote(
        pipeline_id=pipeline_id,
        ray_namespace=f"pipeline_{pipeline_id}_NS",
        cluster_tp_configs={"actor_train": 8, "actor_infer": 8},
        cluster_device_mappings={
            "actor_train": [0, 1, 2, 3, 4, 5, 6, 7],
            "actor_infer": [0, 1, 2, 3, 4, 5, 6, 7],
        },
    )
)

# 4. Admit the pipeline before GPU allocation
ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))

# 5. Create the pipeline coordinator
CoordinatorActor = ray.remote(PipelineCoordinator)
coordinator = CoordinatorActor.options(
    name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
    namespace=f"pipeline_{pipeline_id}_NS",
).remote(pipeline_id=pipeline_id, pipeline_config=my_config)

# 6. Create and run the pipeline
pipeline_actor = ray.get(
    coordinator.create_pipeline_actor.remote(pipeline_config=my_config)
)
ray.get(pipeline_actor.run.remote())
```

See [examples/](examples/) for complete multi-pipeline examples and full configuration options.

## Pipeline Types

RLix currently supports two built-in pipeline types:

### Full Finetune Pipeline (`RollFullFinetunePipeline`)

Full-parameter training with elastic GPU expand/shrink. Each job trains all model weights while yielding idle GPUs to other jobs.

### Multi-LoRA Pipeline (`RollMultiLoraPipeline`)

Concurrent training of multiple LoRA adapters on a shared base model, with an isolated optimizer for each adapter. Jobs share the base model in GPU memory while keeping adapter weights and optimizer states fully independent.

Beyond these built-in options, RLix supports custom pipelines and integrations that follow the RLix control-plane protocol.

## Architecture

RLix separates scheduling from per-pipeline training logic: a shared control plane coordinates multiple independent jobs, while rollout stages elastically consume residual GPU capacity.

```text
┌───────────────────────────────────────────────────────────┐
│                     RLix Control Plane                    │
├──────────────────┬──────────────────┬─────────────────────┤
│   Orchestrator   │    Scheduler     │  Resource Manager   │
│ (lifecycle mgmt) │ (priority +      │   (GPU topology)    │
│                  │ rollout preempt) │                     │
└────────┬─────────┴────────┬─────────┴─────────┬───────────┘
         │                  │                   │
    ┌────▼─────┐       ┌────▼─────┐        ┌────▼─────┐
    │FullFine- │       │Multi-LoRA│        │Custom /  │
    │tune Job 1│       │  Job 2   │        │External  │
    │          │       │          │        │  Job N   │
    └────┬─────┘       └────┬─────┘        └────┬─────┘
         │                  │                   │
    ┌────▼──────────────────▼───────────────────▼────┐
    │               Shared GPU Capacity              │
    │   [GPU 0] [GPU 1] [GPU 2] [GPU 3] ... [GPU N]  │
    └────────────────────────────────────────────────┘
```

## Scheduling Policy

The scheduler assigns GPUs in priority order, with lower values indicating higher priority. All stages except rollout are non-preemptable: once they acquire GPUs, they keep them until completion. Rollout (6) is the lowest-priority stage and is always preemptable, using only the capacity remaining after all higher-priority stages are satisfied. When multiple jobs roll out concurrently, the remaining GPUs are divided proportionally to each job’s outstanding rollout demand, subject to placement constraints.

* **0 Initialization**: Model loading; must complete before scheduling begins.
* **1 Actor Training**: Policy gradient update.
* **2 Critic Training**: Value function update.
* **3 Old-Policy Log Probs**: Log-probability computation under the previous policy.
* **4 Reference-Model Log Probs**: Log-probability computation under the reference model.
* **5 Value Compute**: Value estimation for advantage calculation.
* **6 Rollout**: Trajectory sampling; always preemptable.

