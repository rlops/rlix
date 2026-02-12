from __future__ import annotations

from dataclasses import dataclass

from schedrl.utils.ray_head import head_node_affinity_strategy


def _require_ray():
    try:
        import ray  # noqa: F401
    except Exception as e:
        raise RuntimeError("schedrl.scheduler.resource_manager requires ray") from e


@dataclass(slots=True)
class ResourceManager:
    def __post_init__(self):
        _require_ray()

    def snapshot(self) -> dict:
        raise NotImplementedError("Phase 1 provides resource manager skeleton only")


def get_or_create_resource_manager(*, name: str = "schedrl:resource_manager", namespace: str = "schedrl"):
    _require_ray()
    import ray

    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        pass

    strategy = head_node_affinity_strategy(soft=False)

    @ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)
    class _ResourceManagerActor(ResourceManager):
        pass

    try:
        return (
            _ResourceManagerActor.options(
                name=name,
                namespace=namespace,
                scheduling_strategy=strategy,
                max_restarts=0,
                max_task_retries=0,
            )
            .remote()
        )
    except Exception:
        return ray.get_actor(name, namespace=namespace)

