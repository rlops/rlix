from __future__ import annotations

# Lazy imports for the heavy ROLL-backed pipelines so importing a
# MILES-only path (e.g. ``from rlix.pipeline.miles_coordinator import
# MilesCoordinator``) does not eagerly pull ``full_finetune_pipeline`` →
# ``roll.pipeline.agentic`` → ``roll.models.model_providers`` →
# transformers symbols that may not exist in the ambient transformers
# version. The MILES path has its own narrow imports under
# ``rlix.pipeline.miles_*`` and does not require ROLL agentic plumbing.
from rlix.pipeline.coordinator import COORDINATOR_MAX_CONCURRENCY, PipelineCoordinator

__all__ = [
    "PipelineCoordinator",
    "COORDINATOR_MAX_CONCURRENCY",
    "RollFullFinetunePipeline",
    "RollMultiLoraPipeline",
]


def __getattr__(name: str) -> object:
    if name == "RollFullFinetunePipeline":
        from rlix.pipeline.full_finetune_pipeline import RollFullFinetunePipeline

        return RollFullFinetunePipeline
    if name == "RollMultiLoraPipeline":
        from rlix.pipeline.multi_lora_pipeline import RollMultiLoraPipeline

        return RollMultiLoraPipeline
    raise AttributeError(f"module 'rlix.pipeline' has no attribute {name!r}")
