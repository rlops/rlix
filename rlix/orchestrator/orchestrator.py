from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

# Identifies whether a pipeline trains a full model or LoRA adapters.
# Used as a prefix in pipeline_id for trace readability (e.g., "ft_abc123", "lora_abc123").
PipelineType = Literal["ft", "lora"]

from rlix.protocol.request_id import validate_pipeline_id
from rlix.protocol.types import ORCHESTRATOR_ACTOR_NAME, RLIX_NAMESPACE, SCHEDULER_ACTOR_NAME
from rlix.protocol.validation import RegisterValidationInput, validate_register_pipeline
from rlix.scheduler.resource_manager import get_or_create_resource_manager
from rlix.scheduler.scheduler import scheduler_actor_class
from rlix.utils.ray_head import get_head_node_id
from rlix.utils.ray_head import head_node_affinity_strategy
import ray


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


def _kill_local_ray() -> None:
    # WARNING: Calling `ray stop --force` via subprocess from inside a Ray actor triggers
    # a deepcopy bug with Sentinel enum objects (Ray 2.47.1 + Python 3.10).
    # Use ray.shutdown() instead, which is safe to call from within Ray workers.
    sys.stderr.write("[rlix][WARN] _kill_local_ray() called from inside Ray actor - using ray.shutdown() instead of 'ray stop --force'\n")
    sys.stderr.flush()
    try:
        ray.shutdown()
    except Exception as e:
        sys.stderr.write(f"[rlix][WARN] ray.shutdown() failed: {e}\n")
        sys.stderr.flush()


def _ensure_scheduler_singleton(env_vars: Optional[Dict[str, str]] = None):
    try:
        return ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=RLIX_NAMESPACE)
    except ValueError:
        pass

    strategy = head_node_affinity_strategy(soft=False)
    SchedulerActor = scheduler_actor_class()
    # Thread-limiting env vars for the scheduler actor process.
    scheduler_runtime_env = {"env_vars": env_vars} if env_vars else {}
    try:
        scheduler = (
            SchedulerActor.options(
                name=SCHEDULER_ACTOR_NAME,
                namespace=RLIX_NAMESPACE,
                scheduling_strategy=strategy,
                max_restarts=0,
                max_task_retries=0,
                runtime_env=scheduler_runtime_env,
            )
            .remote()
        )
    except Exception:
        scheduler = ray.get_actor(SCHEDULER_ACTOR_NAME, namespace=RLIX_NAMESPACE)

    try:
        resource_manager = get_or_create_resource_manager()
        # Admission control / topology gating happens here (orchestrator-owned).
        # Scheduler only seeds its idle_gpus from the resource manager after this is ready.
        ray.get(resource_manager.snapshot.remote(wait_timeout_s=10.0, poll_interval_s=0.2))
        # Default: infer GPUs-per-node from Ray topology (portable for local smoke tests).
        # If set, this env var pins a stricter topology contract and must match observation.
        required_gpus_per_node_raw = os.environ.get("RLIX_REQUIRED_GPUS_PER_NODE")
        if required_gpus_per_node_raw is None or required_gpus_per_node_raw.strip() == "":
            required_gpus_per_node = None
        else:
            try:
                required_gpus_per_node = int(required_gpus_per_node_raw)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid RLIX_REQUIRED_GPUS_PER_NODE={required_gpus_per_node_raw!r}, expected int"
                ) from e
            if required_gpus_per_node <= 0:
                raise RuntimeError(
                    f"Invalid RLIX_REQUIRED_GPUS_PER_NODE={required_gpus_per_node_raw!r}, expected > 0"
                )
        ray.get(resource_manager.init_topology.remote(required_gpus_per_node=required_gpus_per_node))
        ray.get(scheduler.initialize.remote(resource_manager=resource_manager))
    except Exception as e:
        raise RuntimeError("Failed to initialize Rlix scheduler actor") from e
    return scheduler


def _kill_ray_on_node(node_ip: str):
    @ray.remote(max_retries=0)
    def _kill_local_ray_task():
        _kill_local_ray()

    return _kill_local_ray_task.options(resources={f"node:{node_ip}": 0.01}).remote()


def _force_stop_cluster_workers_first(*, timeout_s: float = 10.0) -> None:
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
        if env_vars is not None:
            if not isinstance(env_vars, dict):
                raise ValueError(f"env_vars must be dict[str,str] | None, got {type(env_vars).__name__}")
            for k, v in env_vars.items():
                if not isinstance(k, str) or k == "":
                    raise ValueError(f"env_vars keys must be non-empty str, got {k!r}")
                if not isinstance(v, str):
                    raise ValueError(f"env_vars[{k!r}] must be str, got {type(v).__name__}")
        self._env_vars = dict(env_vars or {})
        self._scheduler = _ensure_scheduler_singleton(env_vars=self._env_vars)
        self._pipelines: Dict[str, PipelineState] = {}
        self._shutdown_started = False

    def allocate_pipeline_id(self, pipeline_type: PipelineType) -> str:
        """Allocate a new pipeline_id prefixed with the pipeline type.

        Contract: driver scripts call this first, then create the pipeline coordinator actor using the returned id.
        The prefix ("ft_" or "lora_") makes the pipeline type visible in Perfetto trace labels.
        """
        while True:
            # 12 hex chars = 48 bits of entropy; unique enough for any realistic cluster size.
            pipeline_id = f"{pipeline_type}_{uuid.uuid4().hex[:12]}"
            validate_pipeline_id(pipeline_id)
            if pipeline_id not in self._pipelines:
                return pipeline_id

    def register_pipeline(
        self,
        *,
        pipeline_id: str,
        ray_namespace: str,
        cluster_tp_configs: Dict[str, int],
        cluster_device_mappings: Dict[str, list[int]],
    ) -> RegisterResponse:
        validate_register_pipeline(
            RegisterValidationInput(
                pipeline_id=pipeline_id,
                ray_namespace=ray_namespace,
                cluster_tp_configs=cluster_tp_configs,
                cluster_device_mappings=cluster_device_mappings,
            )
        )
        ray.get(
            self._scheduler.register_pipeline.remote(
                pipeline_id=pipeline_id,
                ray_namespace=ray_namespace,
                cluster_tp_configs=cluster_tp_configs,
                cluster_device_mappings=cluster_device_mappings,
            )
        )

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
        ray.get(self._scheduler.admit_pipeline.remote(pipeline_id=pipeline_id))
        self._pipelines[pipeline_id] = PipelineState(pipeline_id=pipeline_id, registered=True, admitted=True)
        return AdmitResponse(pipeline_id=pipeline_id, scheduler=self._scheduler)

    def kill_pipeline(self, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)

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
        kill_lookup_failures = 0
        kill_failures = 0
        for s in _list_alive_actors():
            name = _attr(s, "name")
            if not isinstance(name, str) or name == "":
                continue
            try:
                handle = ray.get_actor(name, namespace=ray_namespace)
            except Exception:
                kill_lookup_failures += 1
                continue
            try:
                ray.kill(handle, no_restart=True)
            except Exception:
                kill_failures += 1
                continue
        if kill_lookup_failures or kill_failures:
            sys.stderr.write(
                f"[rlix][WARN] kill_pipeline(namespace={ray_namespace!r}) had {kill_lookup_failures} actor lookup failures "
                f"and {kill_failures} ray.kill failures for pipeline_id={pipeline_id!r}\n"
            )

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
                f"[rlix][ERROR] Found {len(unnamed_alive)} unnamed ALIVE actors in namespace {ray_namespace!r}; "
                "using internal core_worker.get_actor_handle(...) to force kill them. "
                "These actors should be named (or their handles retained) to avoid relying on Ray internals.\n"
            )
            for s in unnamed_alive:
                actor_id_hex = _attr(s, "actor_id")
                try:
                    # FIXME: Last-resort kill path using Ray internals.
                    # This relies on `ray._raylet.ActorID` and
                    # `ray.worker.global_worker.core_worker.get_actor_handle(...)`, which are
                    # internal, brittle APIs and may break across Ray versions. Prefer naming
                    # actors or retaining actor handles so this code path is never required.
                    actor_id_obj = ActorID.from_hex(actor_id_hex)
                    handle = ray.worker.global_worker.core_worker.get_actor_handle(actor_id_obj)
                    ray.kill(handle, no_restart=True)
                except Exception as e:
                    sys.stderr.write(f"[rlix][ERROR] Failed to force-kill unnamed actor_id={actor_id_hex!r}: {e}\n")

        # Placement groups are owned by the RollResourceManager singleton actor;
        # Ray cleans them up automatically when that actor is killed.

        if shared_storage is not None:
            try:
                ray.get(shared_storage.delete_port_claims.remote(pipeline_id))
            except Exception as e:
                sys.stderr.write(f"[rlix][ERROR] SharedStorage.delete_port_claims failed for pipeline_id={pipeline_id!r}: {e}\n")
            try:
                ray.get(shared_storage.delete_prefix.remote(f"{pipeline_id}:"))
            except Exception as e:
                sys.stderr.write(f"[rlix][ERROR] SharedStorage.delete_prefix failed for prefix={pipeline_id + ':'!r}: {e}\n")

        self._pipelines.pop(pipeline_id, None)

    def unregister_pipeline(self, pipeline_id: str) -> None:
        validate_pipeline_id(pipeline_id)
        ray.get(self._scheduler.unregister_pipeline.remote(pipeline_id=pipeline_id))
        self._pipelines.pop(pipeline_id, None)

    def shutdown(self, force: bool = True, reason: Optional[str] = None, source: Optional[str] = None) -> None:
        import traceback
        sys.stderr.write(f"[rlix][DEBUG] orchestrator.shutdown called: force={force!r} reason={reason!r} source={source!r}\n")
        sys.stderr.write("".join(traceback.format_stack()))
        sys.stderr.flush()
        if self._shutdown_started:
            return
        self._shutdown_started = True
        if not force:
            raise RuntimeError("shutdown(force=False) is not supported in ENG-123 Phase 1")

        # GPU Tracing: Explicit scheduler shutdown for trace finalization (with short timeout)
        # Use ray.wait() with timeout to avoid blocking indefinitely in fail-fast scenarios
        # 0.5s is enough for flush() under normal conditions, but won't stall on dead actors
        try:
            shutdown_ref = self._scheduler.shutdown.remote()
            ray.wait([shutdown_ref], timeout=0.5)
        except Exception:
            pass  # Best-effort, don't stall shutdown

        _force_stop_cluster_workers_first()
