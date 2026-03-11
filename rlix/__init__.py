"""Rlix: Ray-based multi-pipeline GPU time-sharing (ENG-123).

Phase 1 provides the core package skeleton + protocol contracts + Library Mode discovery.
"""

from __future__ import annotations

__all__ = [
    "init",
    "__version__",
    "RlixCoordinator",
    "RlixFullFinetunePipeline",
    "RlixMultiLoraPipeline",
]

__version__ = "0.0.0"

from rlix.init import init  # noqa: E402


# Lazy imports to avoid circular dependency: rlix.pipeline imports roll.pipeline
# which imports rlix.protocol, causing a circular chain if loaded eagerly here.
def __getattr__(name: str) -> object:
    if name in ("RlixCoordinator", "RlixFullFinetunePipeline", "RlixMultiLoraPipeline"):
        from rlix.pipeline import RlixCoordinator, RlixFullFinetunePipeline, RlixMultiLoraPipeline
        _lazy_exports = {
            "RlixCoordinator": RlixCoordinator,
            "RlixFullFinetunePipeline": RlixFullFinetunePipeline,
            "RlixMultiLoraPipeline": RlixMultiLoraPipeline,
        }
        return _lazy_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
