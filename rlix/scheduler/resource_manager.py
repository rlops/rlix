from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import ray

from rlix.protocol.types import RESOURCE_MANAGER_ACTOR_NAME, RLIX_NAMESPACE
from rlix.utils.ray import head_node_affinity_strategy

# Platform/resource assumption
# Rlix and ROLL resource keys differ: this module uses Ray's "GPU" resource
# key (as returned by `ray.cluster_resources()` / `ray.nodes()`), whereas ROLL's
# scheduler stack references `current_platform.ray_device_key`. For parity across
# deployment platforms we would need to align these abstractions. We
# document that Rlix currently targets CUDA-only setups where the Ray GPU
# resource key is present and meaningful. Prefer naming/standardizing the
# platform-level device key if broader device types are required in future.


@dataclass(slots=True)
class ResourceManager:
    """GPU topology and cluster resource state for the rlix control plane.

    Tracks the per-node GPU count (frozen after init_topology) and provides
    polling snapshots of live Ray cluster resources. Deployed as the singleton
    Ray actor ``rlix:resource_manager`` via get_or_create_resource_manager.

    Note: this is the *rlix-native* resource manager. ROLL pipelines use a
    separate ``rlix:roll_resource_manager`` actor (RollResourceManagerProxy)
    that manages per-pipeline placement groups. The two actors coexist in the
    ``rlix`` namespace with distinct names.
    """

    required_gpus_per_node: int | None = None

    def init_topology(self, *, required_gpus_per_node: int | None = None) -> int:
        """Initialize and freeze required GPU topology assumptions for this job.

        Returns the finalized required_gpus_per_node.

        Contract: fail-fast if GPU-per-node is inconsistent across GPU nodes.
        """
        if self.required_gpus_per_node is not None:
            raise RuntimeError("ResourceManager topology already initialized")

        alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
        gpu_counts = []
        for n in alive_nodes:
            res = n.get("Resources") or {}
            count = int(res.get("GPU", 0) or 0)
            if count > 0:
                gpu_counts.append(count)
        if not gpu_counts:
            raise RuntimeError("No GPU nodes found when initializing topology")

        observed = gpu_counts[0]
        if any(x != observed for x in gpu_counts):
            raise RuntimeError(f"Inconsistent GPU-per-node across alive nodes: {sorted(gpu_counts)!r}")

        if required_gpus_per_node is None:
            required = observed
        else:
            required = int(required_gpus_per_node)
            if required <= 0:
                raise ValueError(f"required_gpus_per_node must be > 0, got {required_gpus_per_node!r}")
            if required != observed:
                raise RuntimeError(
                    f"required_gpus_per_node={required} does not match observed GPUs-per-node={observed}"
                )

        self.required_gpus_per_node = required
        return int(required)

    def get_required_gpus_per_node(self) -> int:
        """Return the frozen per-node GPU count. Raises if init_topology has not been called."""
        if self.required_gpus_per_node is None:
            raise RuntimeError("ResourceManager topology not initialized; call init_topology() first")
        return int(self.required_gpus_per_node)

    def get_num_gpus(self) -> int:
        """Return current Ray cluster GPU count (no waiting / gating)."""
        cluster_resources = ray.cluster_resources()
        return int(cluster_resources.get("GPU", 0))

    def snapshot(
        self,
        *,
        wait_timeout_s: float = 10.0,
        poll_interval_s: float = 0.2,
        expected_num_gpus: int | None = None,
    ) -> dict[str, Any]:
        """Poll Ray until GPU resources are available and return a cluster snapshot.

        Args:
            wait_timeout_s: Max seconds to poll before raising RuntimeError.
            poll_interval_s: Sleep interval between polls.
            expected_num_gpus: If set, wait until at least this many GPUs are visible.
                If None, any positive GPU count satisfies the wait.

        Returns:
            Dict with keys ``num_gpus``, ``alive_nodes``, and ``cluster_resources``.
        """
        if wait_timeout_s <= 0:
            raise ValueError(f"wait_timeout_s must be > 0, got {wait_timeout_s!r}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s!r}")
        if expected_num_gpus is not None and expected_num_gpus < 0:
            raise ValueError(f"expected_num_gpus must be >= 0, got {expected_num_gpus!r}")

        deadline = time.monotonic() + float(wait_timeout_s)
        last_num_gpus = None
        last_alive_nodes = None
        last_cluster_resources = None
        while time.monotonic() < deadline:
            cluster_resources = ray.cluster_resources()
            alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
            num_gpus = int(cluster_resources.get("GPU", 0))

            last_num_gpus = num_gpus
            last_alive_nodes = alive_nodes
            last_cluster_resources = cluster_resources

            if expected_num_gpus is not None:
                if num_gpus >= expected_num_gpus:
                    break
            else:
                if num_gpus > 0:
                    break

            time.sleep(float(poll_interval_s))

        else:
            raise RuntimeError(
                "Timed out waiting for Ray GPU topology to be ready. "
                f"expected_num_gpus={expected_num_gpus!r}, last_num_gpus={last_num_gpus!r}, "
                f"alive_nodes={len(last_alive_nodes or [])}, cluster_resources={last_cluster_resources!r}"
            )

        return {
            "num_gpus": int(last_num_gpus or 0),
            "alive_nodes": [
                {"NodeID": n.get("NodeID"), "NodeManagerAddress": n.get("NodeManagerAddress")}
                for n in (last_alive_nodes or [])
            ],
            "cluster_resources": dict(last_cluster_resources or {}),
        }


def get_or_create_resource_manager(*, name: str = RESOURCE_MANAGER_ACTOR_NAME, namespace: str = RLIX_NAMESPACE) -> Any:
    """Return the singleton ``rlix:resource_manager`` actor, creating it on the head node if needed."""
    strategy = head_node_affinity_strategy(soft=False)

    @ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)
    class _ResourceManagerActor(ResourceManager):
        pass

    # get_if_exists=True: Ray returns the existing actor if already created,
    # avoiding manual race handling for concurrent creation attempts.
    return _ResourceManagerActor.options(  # type: ignore[attr-defined]
        name=name,
        namespace=namespace,
        scheduling_strategy=strategy,
        max_restarts=0,
        max_task_retries=0,
        get_if_exists=True,
    ).remote()
