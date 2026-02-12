from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from schedrl.protocol.request_id import validate_pipeline_id
from schedrl.protocol.validation import RegisterValidationInput, validate_register_pipeline
from schedrl.utils.ray_head import get_head_node_id
from schedrl.utils.ray_head import head_node_affinity_strategy

SCHEDRL_NAMESPACE = "schedrl"
ORCHESTRATOR_ACTOR_NAME = "schedrl:orchestrator"
SCHEDULER_ACTOR_NAME = "schedrl:scheduler"


@dataclass(frozen=True, slots=True)
class RegisterResponse:
    pipeline_id: str


@dataclass(frozen=True, slots=True)
class AdmitResponse:
    pipeline_id: str
    scheduler: Any


@dataclass(frozen=True, slots=True)
class PipelineState:
    pipeline_id: str
    registered: bool
    admitted: bool


def _ray_cli_path() -> str:
    python_bin_dir = Path(sys.executable).parent
    return str(python_bin_dir / "ray")


def _require_ray():
    try:
        import ray  # noqa: F401
    except Exception as e:
        raise RuntimeError("schedrl.orchestrator requires ray") from e


def _kill_local_ray() -> None:
    ray_executable = _ray_cli_path()
    subprocess.run([ray_executable, "stop", "--force"], check=False)


def _ensure_scheduler_singleton():
    _require_ray()
    import ray

    from schedrl.scheduler.scheduler import scheduler_actor_class

    try:
        return ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=SCHEDRL_NAMESPACE)
    except ValueError:
        pass

    strategy = head_node_affinity_strategy(soft=False)
    SchedulerActor = scheduler_actor_class()
    try:
        scheduler = (
            SchedulerActor.options(
                name=SCHEDULER_ACTOR_NAME,
                namespace=SCHEDRL_NAMESPACE,
                scheduling_strategy=strategy,
                max_restarts=0,
                max_task_retries=0,
            )
            .remote()
        )
    except Exception:
        scheduler = ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=SCHEDRL_NAMESPACE)

    try:
        ray.get(scheduler.initialize.remote())
    except Exception as e:
        raise RuntimeError("Failed to initialize SchedRL scheduler actor") from e
    return scheduler


def _kill_ray_on_node(node_ip: str):
    _require_ray()
    import ray

    @ray.remote(max_retries=0)
    def _kill_local_ray_task():
        _kill_local_ray()

    return _kill_local_ray_task.options(resources={f"node:{node_ip}": 0.01}).remote()


def _force_stop_cluster_workers_first(*, timeout_s: float = 10.0) -> None:
    _require_ray()
    import ray

    head_node_id = get_head_node_id()
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get("Alive")]

    worker_tasks = []
    for node in alive_nodes:
        if node.get("NodeID") == head_node_id:
            continue
        node_ip = node.get("NodeManagerAddress")
        if not node_ip:
            continue
        worker_tasks.append(_kill_ray_on_node(node_ip))

    if worker_tasks:
        ray.wait(worker_tasks, timeout=timeout_s, num_returns=len(worker_tasks))

    time.sleep(0.2)
    _kill_local_ray()


class Orchestrator:
    def __init__(self, env_vars: Optional[Dict[str, str]] = None):
        _require_ray()
        if env_vars is not None:
            if not isinstance(env_vars, dict):
                raise ValueError(f"env_vars must be dict[str,str] | None, got {type(env_vars).__name__}")
            for k, v in env_vars.items():
                if not isinstance(k, str) or k == "":
                    raise ValueError(f"env_vars keys must be non-empty str, got {k!r}")
                if not isinstance(v, str):
                    raise ValueError(f"env_vars[{k!r}] must be str, got {type(v).__name__}")
        self._env_vars = dict(env_vars or {})
        self._scheduler = _ensure_scheduler_singleton()
        self._pipelines: Dict[str, PipelineState] = {}
        self._shutdown_started = False

    def register_pipeline(self, *, pipeline_id: str, total_gpus: int, gpu_ids: list[int]) -> RegisterResponse:
        validate_register_pipeline(
            RegisterValidationInput(pipeline_id=pipeline_id, total_gpus=total_gpus, gpu_ids=gpu_ids)
        )
        import ray
        ray.get(self._scheduler.register_pipeline.remote(pipeline_id=pipeline_id))

        state = self._pipelines.get(pipeline_id)
        if state is None:
            state = PipelineState(pipeline_id=pipeline_id, registered=True, admitted=False)
        else:
            state = PipelineState(pipeline_id=pipeline_id, registered=True, admitted=state.admitted)
        self._pipelines[pipeline_id] = state
        return RegisterResponse(pipeline_id=pipeline_id)

    def admit_pipeline(self, *, pipeline_id: str) -> AdmitResponse:
        validate_pipeline_id(pipeline_id)
        state = self._pipelines.get(pipeline_id)
        if state is None or not state.registered:
            raise RuntimeError(f"Pipeline {pipeline_id!r} must be registered before admission")
        if state.admitted:
            return AdmitResponse(pipeline_id=pipeline_id, scheduler=self._scheduler)
        import ray
        ray.get(self._scheduler.admit_pipeline.remote(pipeline_id=pipeline_id))
        self._pipelines[pipeline_id] = PipelineState(pipeline_id=pipeline_id, registered=True, admitted=True)
        return AdmitResponse(pipeline_id=pipeline_id, scheduler=self._scheduler)

    def get_pipeline_state(self, pipeline_id: str) -> PipelineState:
        validate_pipeline_id(pipeline_id)
        state = self._pipelines.get(pipeline_id)
        if state is None:
            return PipelineState(pipeline_id=pipeline_id, registered=False, admitted=False)
        return state

    def monitor_pipelines(self) -> Dict[str, PipelineState]:
        return dict(self._pipelines)

    def cleanup_pipeline(self, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        self._pipelines.pop(pipeline_id, None)

    def kill_pipeline(self, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        import ray

        # Best-effort SharedStorage cleanup (job-global).
        # This releases pipeline-owned coordination metadata (e.g., MASTER_ADDR_PORT claims) so ports can be reused
        # within the same job.
        try:
            shared_storage = ray.get_actor("SHARED_STORAGE_ACTOR", namespace="global_storage_namespace")
        except Exception:
            shared_storage = None

        try:
            ray_namespace = ray.get(self._scheduler.get_pipeline_namespace.remote(pipeline_id=pipeline_id))
        except Exception as e:
            raise RuntimeError(f"Failed to resolve ray_namespace for pipeline_id {pipeline_id!r}") from e

        # First, remove scheduler-side state under the scheduler lock so future scheduling cycles ignore this pipeline.
        # This also unblocks any callers waiting on scheduler events for this pipeline.
        ray.get(self._scheduler.unregister_pipeline.remote(pipeline_id=pipeline_id))

        try:
            from ray.util.state import list_actors
        except Exception as e:
            raise RuntimeError("ray.util.state.list_actors is required for kill_pipeline(namespace=...)") from e

        def _attr(obj: Any, key: str, default: Any = None) -> Any:
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default

        def _list_alive_actors(*, name_filter: Optional[str] = None):
            filters = [("ray_namespace", "=", ray_namespace)]
            if name_filter is not None:
                filters.append(("name", "=", name_filter))
            states = list_actors(filters=filters)
            alive = []
            for s in states:
                if _attr(s, "state") == "ALIVE":
                    alive.append(s)
            return alive

        # Kill all named actors in this pipeline namespace.
        for s in _list_alive_actors():
            name = _attr(s, "name")
            if not isinstance(name, str) or name == "":
                continue
            try:
                handle = ray.get_actor(name, namespace=ray_namespace)
            except Exception:
                continue
            try:
                ray.kill(handle, no_restart=True)
            except Exception:
                continue

        # Unnamed actors: assume temporary and wait briefly for natural teardown.
        deadline = time.time() + 10.0
        while True:
            unnamed_alive = _list_alive_actors(name_filter="")
            if not unnamed_alive:
                break
            if time.time() >= deadline:
                break
            time.sleep(0.2)

        unnamed_alive = _list_alive_actors(name_filter="")
        if unnamed_alive:
            # Nuclear option: use internal Ray APIs to kill by ActorID.
            try:
                from ray._raylet import ActorID  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Found {len(unnamed_alive)} unnamed ALIVE actors in namespace {ray_namespace!r} but cannot import ActorID"
                ) from e

            sys.stderr.write(
                f"[schedrl][ERROR] Found {len(unnamed_alive)} unnamed ALIVE actors in namespace {ray_namespace!r}; "
                "using internal core_worker.get_actor_handle(...) to force kill them. "
                "These actors should be named (or their handles retained) to avoid relying on Ray internals.\n"
            )
            for s in unnamed_alive:
                actor_id_hex = _attr(s, "actor_id")
                try:
                    actor_id_obj = ActorID.from_hex(actor_id_hex)
                    handle = ray.worker.global_worker.core_worker.get_actor_handle(actor_id_obj)
                    ray.kill(handle, no_restart=True)
                except Exception as e:
                    sys.stderr.write(f"[schedrl][ERROR] Failed to force-kill unnamed actor_id={actor_id_hex!r}: {e}\n")

        if shared_storage is not None:
            try:
                ray.get(shared_storage.delete_port_claims.remote(pipeline_id))
            except Exception as e:
                sys.stderr.write(f"[schedrl][ERROR] SharedStorage.delete_port_claims failed for pipeline_id={pipeline_id!r}: {e}\n")
            try:
                ray.get(shared_storage.delete_prefix.remote(f"{pipeline_id}:"))
            except Exception:
                pass

        self._pipelines.pop(pipeline_id, None)

    def unregister_pipeline(self, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        import ray
        ray.get(self._scheduler.unregister_pipeline.remote(pipeline_id=pipeline_id))
        self._pipelines.pop(pipeline_id, None)

    def shutdown(self, force: bool = True, reason: Optional[str] = None, source: Optional[str] = None) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        if not force:
            raise RuntimeError("shutdown(force=False) is not supported in ENG-123 Phase 1")
        _force_stop_cluster_workers_first()

    def get_env_vars(self) -> Dict[str, str]:
        return dict(self._env_vars)
