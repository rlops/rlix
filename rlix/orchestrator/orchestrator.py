"""RLix Orchestrator.

Pipeline lifecycle management: allocate IDs, register topology with the scheduler,
admit pipelines for scheduling, kill pipelines (teardown actors + cleanup), and
force-shutdown the entire Ray cluster.

Singleton actor name: ``rlix:orchestrator``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import ray

from rlix.protocol.types import RLIX_NAMESPACE, SCHEDULER_ACTOR_NAME
from rlix.protocol.validation import RegisterValidationInput, validate_pipeline_id, validate_register_pipeline
from rlix.scheduler.resource_manager import get_or_create_resource_manager
from rlix.scheduler.scheduler import scheduler_actor_class
from rlix.utils.env import parse_env_timeout_s
from rlix.utils.ray import get_head_node_id, head_node_affinity_strategy

# Identifies whether a pipeline trains a full model or LoRA adapters.
# Used as a prefix in pipeline_id for trace readability (e.g., "ft_abc123", "lora_abc123").
PipelineType = Literal["ft", "lora"]

# Timeouts for orchestrator operations (seconds).  None means "no timeout".
_RESOURCE_SNAPSHOT_TIMEOUT_S: Optional[float] = parse_env_timeout_s("RLIX_RESOURCE_SNAPSHOT_TIMEOUT_S", 10.0)
_RESOURCE_SNAPSHOT_POLL_S: Optional[float] = parse_env_timeout_s("RLIX_RESOURCE_SNAPSHOT_POLL_S", 0.2)
_WORKER_STOP_TIMEOUT_S: Optional[float] = parse_env_timeout_s("RLIX_WORKER_STOP_TIMEOUT_S", 10.0)
_POST_STOP_SETTLE_S: Optional[float] = parse_env_timeout_s("RLIX_POST_STOP_SETTLE_S", 0.2)
_UNNAMED_ACTOR_CLEANUP_TIMEOUT_S: Optional[float] = parse_env_timeout_s("RLIX_UNNAMED_ACTOR_CLEANUP_TIMEOUT_S", 10.0)
_SCHEDULER_FLUSH_TIMEOUT_S: Optional[float] = parse_env_timeout_s("RLIX_SCHEDULER_FLUSH_TIMEOUT_S", 0.5)
# 12 hex chars = 48 bits of entropy; unique enough for any realistic cluster size.
_PIPELINE_ID_RANDOM_HEX_LEN: int = 12

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegisterResponse:
    """Returned by ``Orchestrator.register_pipeline``."""

    pipeline_id: str


@dataclass(frozen=True, slots=True)
class AdmitResponse:
    """Returned by ``Orchestrator.admit_pipeline``; includes the scheduler handle for direct RPC."""

    pipeline_id: str
    scheduler: Any


@dataclass(frozen=True, slots=True)
class PipelineState:
    """Orchestrator-local bookkeeping for a single pipeline's lifecycle stage."""

    pipeline_id: str
    registered: bool
    admitted: bool


def _kill_local_ray() -> None:
    """Shut down the local Ray worker/head process.

    Uses ``ray.shutdown()`` rather than ``ray stop --force`` because subprocess-based
    stop triggers a deepcopy bug with Sentinel enums (Ray 2.47.1 + Python 3.10).
    """
    logger.info("_kill_local_ray() called from inside Ray actor - using ray.shutdown() instead of 'ray stop --force'")
    try:
        ray.shutdown()
    except Exception as e:
        logger.warning("ray.shutdown() failed: %s", e)


def _kill_ray_on_node(node_ip: str) -> ray.ObjectRef[None]:
    """Spawn a one-shot remote task on ``node_ip`` that calls ``_kill_local_ray()``."""

    @ray.remote(max_retries=0, max_task_retries=0)
    def _kill_local_ray_task() -> None:
        _kill_local_ray()

    return _kill_local_ray_task.options(resources={f"node:{node_ip}": 0.01}).remote()


def _force_stop_cluster_workers_first(*, timeout_s: Optional[float] = _WORKER_STOP_TIMEOUT_S) -> None:
    """Shut down every Ray worker node, then the head node.

    Worker nodes are killed first (in parallel via remote tasks) so that
    in-flight work drains before the head disappears.
    """
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

    if _POST_STOP_SETTLE_S is not None:
        time.sleep(_POST_STOP_SETTLE_S)
    _kill_local_ray()


def _ensure_scheduler_singleton(env_vars: Optional[Dict[str, str]] = None) -> Any:
    """Create (or retrieve) the singleton scheduler actor and initialize it.

    Idempotent: ``get_if_exists=True`` means concurrent callers converge on
    the same actor.  Initialization seeds the scheduler's GPU topology from
    the resource manager and starts the central scheduling loop.
    """
    strategy = head_node_affinity_strategy(soft=False)
    SchedulerActor = scheduler_actor_class()
    # Thread-limiting env vars for the scheduler actor process.
    scheduler_runtime_env = {"env_vars": env_vars} if env_vars else {}
    # get_if_exists=True: Ray returns the existing actor if already created,
    # avoiding manual race handling for concurrent creation attempts.
    scheduler = SchedulerActor.options(
        name=SCHEDULER_ACTOR_NAME,
        namespace=RLIX_NAMESPACE,
        scheduling_strategy=strategy,
        max_restarts=0,
        max_task_retries=0,
        runtime_env=scheduler_runtime_env,
        get_if_exists=True,
    ).remote()

    try:
        resource_manager = get_or_create_resource_manager()
        # Admission control / topology gating happens here (orchestrator-owned).
        # Scheduler only seeds its idle_gpus from the resource manager after this is ready.
        ray.get(
            resource_manager.snapshot.remote(
                wait_timeout_s=_RESOURCE_SNAPSHOT_TIMEOUT_S, poll_interval_s=_RESOURCE_SNAPSHOT_POLL_S
            )
        )
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
                raise RuntimeError(f"Invalid RLIX_REQUIRED_GPUS_PER_NODE={required_gpus_per_node_raw!r}, expected > 0")
        ray.get(resource_manager.init_topology.remote(required_gpus_per_node=required_gpus_per_node))
        ray.get(scheduler.initialize.remote(resource_manager=resource_manager))
    except Exception as e:
        raise RuntimeError("Failed to initialize Rlix scheduler actor") from e
    return scheduler


class Orchestrator:
    """Central control-plane actor for pipeline lifecycle management.

    Responsibilities:
      - Allocate globally unique pipeline IDs.
      - Register pipeline topology with the scheduler.
      - Admit pipelines so the scheduler begins GPU allocation.
      - Kill individual pipelines (teardown actors, release ports).
      - Force-shutdown the entire Ray cluster.
    """

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
            pipeline_id = f"{pipeline_type}_{uuid.uuid4().hex[:_PIPELINE_ID_RANDOM_HEX_LEN]}"
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
        """Register a pipeline's cluster topology with the scheduler.

        Must be called after ``allocate_pipeline_id`` and before ``admit_pipeline``.
        """
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
        """Admit a registered pipeline so the scheduler begins allocating GPUs for it."""
        validate_pipeline_id(pipeline_id)
        state = self._pipelines.get(pipeline_id)
        if state is None or not state.registered:
            logger.warning("Pipeline %r must be registered before admission", pipeline_id)
            return AdmitResponse(pipeline_id=pipeline_id, scheduler=None)
        if state.admitted:
            return AdmitResponse(pipeline_id=pipeline_id, scheduler=self._scheduler)
        ray.get(self._scheduler.admit_pipeline.remote(pipeline_id=pipeline_id))
        self._pipelines[pipeline_id] = PipelineState(pipeline_id=pipeline_id, registered=True, admitted=True)
        return AdmitResponse(pipeline_id=pipeline_id, scheduler=self._scheduler)

    def _cleanup_shared_storage(self, shared_storage: Any, pipeline_id: str) -> None:
        """Best-effort SharedStorage cleanup: release port claims and prefixed keys."""
        if shared_storage is None:
            return
        try:
            ray.get(shared_storage.delete_port_claims.remote(pipeline_id))
        except Exception as exc:
            logger.error("SharedStorage.delete_port_claims failed for pipeline_id=%r: %s", pipeline_id, exc)
        try:
            ray.get(shared_storage.delete_prefix.remote(f"{pipeline_id}:"))
        except Exception as exc:
            logger.error("SharedStorage.delete_prefix failed for prefix=%r: %s", f"{pipeline_id}:", exc)

    def kill_pipeline(self, pipeline_id: str) -> None:
        """Tear down a pipeline: unregister from scheduler, kill all actors, release ports.

        Cleanup order:
          1. Resolve SharedStorage handle and pipeline Ray namespace.
          2. Unregister from scheduler (stops future GPU allocation).
          3. Kill named actors in the pipeline's Ray namespace.
          4. Wait for unnamed actors to exit; force-kill via internal APIs as last resort.
          5. Clean up SharedStorage coordination metadata (port claims, prefixed keys).
        """
        validate_pipeline_id(pipeline_id)

        # Step 1: Resolve SharedStorage handle and pipeline Ray namespace.
        try:
            shared_storage = ray.get_actor("SHARED_STORAGE_ACTOR", namespace="global_storage_namespace")
        except Exception:
            shared_storage = None

        try:
            ray_namespace = ray.get(self._scheduler.get_pipeline_namespace.remote(pipeline_id=pipeline_id))
        except Exception as e:
            logger.warning("Failed to resolve ray_namespace for pipeline_id %r: %s", pipeline_id, e)
            self._cleanup_shared_storage(shared_storage, pipeline_id)
            self._pipelines.pop(pipeline_id, None)
            return

        # Step 2: Unregister from scheduler (stops future GPU allocation, unblocks waiters).
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

        def _list_alive_actors(*, name_filter: Optional[str] = None) -> list[Any]:
            filters = [("ray_namespace", "=", ray_namespace)]
            if name_filter is not None:
                filters.append(("name", "=", name_filter))
            states = list_actors(filters=filters)
            alive = []
            for s in states:
                if _attr(s, "state") == "ALIVE":
                    alive.append(s)
            return alive

        # Step 3: Kill all named actors in this pipeline namespace.
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
            logger.warning(
                "kill_pipeline(namespace=%r) had %d actor lookup failures and %d ray.kill failures for pipeline_id=%r",
                ray_namespace,
                kill_lookup_failures,
                kill_failures,
                pipeline_id,
            )

        # Step 4: Wait for unnamed actors to exit; force-kill via internal APIs as last resort.
        if _UNNAMED_ACTOR_CLEANUP_TIMEOUT_S is None:
            deadline = None
        else:
            deadline = time.time() + _UNNAMED_ACTOR_CLEANUP_TIMEOUT_S
        while True:
            unnamed_alive = _list_alive_actors(name_filter="")
            if not unnamed_alive:
                break
            if deadline is not None and time.time() >= deadline:
                break
            if _POST_STOP_SETTLE_S is not None:
                time.sleep(_POST_STOP_SETTLE_S)

        unnamed_alive = _list_alive_actors(name_filter="")
        if unnamed_alive:
            # Nuclear option: use internal Ray APIs to kill by ActorID.
            try:
                from ray._raylet import ActorID
            except Exception as e:
                raise RuntimeError(
                    f"Found {len(unnamed_alive)} unnamed ALIVE actors in namespace {ray_namespace!r} but cannot import ActorID"
                ) from e

            logger.error(
                "Found %d unnamed ALIVE actors in namespace %r; "
                "using internal core_worker.get_actor_handle(...) to force kill them. "
                "These actors should be named (or their handles retained) to avoid relying on Ray internals.",
                len(unnamed_alive),
                ray_namespace,
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
                    logger.error("Failed to force-kill unnamed actor_id=%r: %s", actor_id_hex, e)

        # Step 5: Clean up SharedStorage coordination metadata (port claims, prefixed keys).
        # Placement groups are owned by the RollResourceManager singleton actor;
        # Ray cleans them up automatically when that actor is killed.
        self._cleanup_shared_storage(shared_storage, pipeline_id)

        self._pipelines.pop(pipeline_id, None)

    def unregister_pipeline(self, pipeline_id: str) -> None:
        """Remove a pipeline from the scheduler without killing its actors."""
        validate_pipeline_id(pipeline_id)
        ray.get(self._scheduler.unregister_pipeline.remote(pipeline_id=pipeline_id))
        self._pipelines.pop(pipeline_id, None)

    def shutdown(self, force: bool = True, reason: Optional[str] = None, source: Optional[str] = None) -> None:
        """Force-shutdown the entire Ray cluster (workers first, then head).

        Idempotent: subsequent calls after the first are no-ops.
        """
        import traceback

        logger.info(
            "orchestrator.shutdown called: force=%r reason=%r source=%r\n%s",
            force,
            reason,
            source,
            "".join(traceback.format_stack()),
        )
        if self._shutdown_started:
            return
        self._shutdown_started = True
        if not force:
            raise RuntimeError("shutdown(force=False) is not supported")

        # GPU Tracing: Explicit scheduler shutdown for trace finalization (with short timeout)
        # Use ray.wait() with timeout to avoid blocking indefinitely in fail-fast scenarios
        # 0.5s is enough for flush() under normal conditions, but won't stall on dead actors
        try:
            shutdown_ref = self._scheduler.shutdown.remote()
            ray.wait([shutdown_ref], timeout=_SCHEDULER_FLUSH_TIMEOUT_S)
        except Exception:
            pass  # Best-effort, don't stall shutdown

        _force_stop_cluster_workers_first()
