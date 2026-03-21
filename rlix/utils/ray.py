"""Ray actor and scheduling utilities for rlix."""

from __future__ import annotations

from typing import Any

import ray


def get_head_node_id() -> str:
    try:
        from ray.util.state import list_nodes
    except Exception as e:
        raise RuntimeError("ray.util.state.list_nodes is required to identify the head node") from e

    # ray.util.state list_* APIs only support "=" / "!=" predicates (not "==").
    nodes = list_nodes(filters=[("is_head_node", "=", "True")])
    if not nodes:
        raise RuntimeError("Could not identify head node via ray.util.state.list_nodes")
    node_id: Any = nodes[0].get("node_id")  # type: ignore[no-untyped-call]
    if not node_id:
        raise RuntimeError(f"Head node record missing node_id: {nodes[0]!r}")
    return str(node_id)


def head_node_affinity_strategy(*, soft: bool = False) -> Any:
    """Return a Ray scheduling strategy that pins actor placement to the head node.

    Args:
        soft: If False (default), placement on the head node is mandatory.
            If True, the head node is preferred but Ray may fall back to another node.
    """
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except Exception as e:
        raise RuntimeError("ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy is required") from e

    return NodeAffinitySchedulingStrategy(node_id=get_head_node_id(), soft=soft)


def get_actor_or_raise(name: str, namespace: str, *, error_context: str) -> Any:
    """Get an existing Ray actor by name, raising RuntimeError if not found.

    Used when the caller requires the actor to already exist (e.g., scheduler,
    coordinator) and wants a clear error message on startup ordering problems.
    """
    try:
        return ray.get_actor(name, namespace=namespace)
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve actor {name!r} in namespace {namespace!r}. {error_context}") from exc
