"""RayVirtualCluster-compatible adapter wrapping RLix-owned placement groups.

NeMo RL's VllmGeneration / RayWorkerGroup / LmPolicy consumers expect a
`RayVirtualCluster` surface. In RLix mode the placement groups are owned by
ROLL's RollResourceManagerProxy so that NeMo RL and ROLL can share bundles
in partial-overlap topologies. This adapter duck-types the subset of the
RayVirtualCluster surface that those consumers actually touch, without
importing nemo_rl or subclassing the real class.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import ray

logger = logging.getLogger(__name__)


class RLixVirtualClusterAdapter:
    """Duck-typed stand-in for RayVirtualCluster backed by RLix-owned PGs.

    Placement-group lifecycle is owned by the RLix coordinator (via
    RollResourceManagerProxy); this adapter never creates or destroys PGs.
    """

    def __init__(
        self,
        *,
        placement_groups: List[Any],
        bundle_ct_per_node_list: List[int],
        num_gpus_per_node: int,
        use_gpus: bool = True,
        max_colocated_worker_groups: int = 1,
        name: str = "",
    ) -> None:
        self._placement_groups: List[Any] = list(placement_groups)
        self._bundle_ct_per_node_list: List[int] = list(bundle_ct_per_node_list)
        self.num_gpus_per_node: int = num_gpus_per_node
        self.use_gpus: bool = use_gpus
        self.max_colocated_worker_groups: int = max_colocated_worker_groups
        self.name: str = name

    def world_size(self) -> int:
        return sum(self._bundle_ct_per_node_list)

    def node_count(self) -> int:
        return len(self._bundle_ct_per_node_list)

    def get_placement_groups(self) -> List[Any]:
        return list(self._placement_groups)

    def _init_placement_groups(
        self,
        strategy: Optional[str] = None,
        use_unified_pg: bool = False,
    ) -> List[Any]:
        return list(self._placement_groups)

    def shutdown(self) -> bool:
        logger.debug(
            "RLixVirtualClusterAdapter.shutdown() no-op: RLix coordinator owns PG lifecycle"
        )
        return True

    def get_available_address_and_port(
        self, pg_idx: int = 0, bundle_idx: int = 0
    ) -> Tuple[str, int]:
        pg = self._placement_groups[pg_idx]

        @ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_idx,
            ),
        )
        def _find_address_and_port() -> Tuple[str, int]:
            import socket

            address = socket.gethostbyname(socket.gethostname())
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            return address, port

        return ray.get(_find_address_and_port.remote())

    def get_master_address_and_port(self) -> Tuple[str, int]:
        return self.get_available_address_and_port(pg_idx=0, bundle_idx=0)
