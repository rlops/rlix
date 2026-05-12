from __future__ import annotations

from rlix.pipeline.coordinator import COORDINATOR_MAX_CONCURRENCY, PipelineCoordinator

# ROLL-based pipelines are intentionally not eagerly imported — the NeMo RL
# port has no ROLL dependency, and the roll.* package may not be installed.
# Consumers that still need them should import via the dotted path directly:
#   from rlix.pipeline.full_finetune_pipeline import RollFullFinetunePipeline
#   from rlix.pipeline.multi_lora_pipeline import RollMultiLoraPipeline

__all__ = [
    "PipelineCoordinator",
    "COORDINATOR_MAX_CONCURRENCY",
]
