from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import ray

from rlix.orchestrator.orchestrator import Orchestrator
from rlix.protocol.types import ORCHESTRATOR_ACTOR_NAME, RLIX_NAMESPACE
from rlix.utils.ray import head_node_affinity_strategy


@dataclass(frozen=True, slots=True)
class ConnectOptions:
    """Internal options bundle passed to ``_get_or_create_orchestrator``."""

    address: str = "auto"
    create_if_missing: bool = True
    backoff_s: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    env_vars: Optional[dict[str, str]] = None


def connect(
    *,
    create_if_missing: bool = True,
    address: str = "auto",
    env_vars: Optional[dict[str, str]] = None,
) -> Any:
    """Initialize Ray and return the Rlix orchestrator actor handle.

    Exported as ``rlix.init()``. Initializes Ray if not already connected,
    then gets or creates the singleton orchestrator actor on the head node.

    Args:
        create_if_missing: If True (default), create the orchestrator when it
            does not exist. If False, raise if the actor is not found.
        address: Ray cluster address. Defaults to ``"auto"``.
        env_vars: Environment variables forwarded to the orchestrator and
            scheduler actors via Ray ``runtime_env``.

    Returns:
        Ray actor handle for the orchestrator.
    """
    if not ray.is_initialized():
        from rlix.utils.env import thread_limit_env_vars
        ray.init(
            address=address, namespace=RLIX_NAMESPACE,
            ignore_reinit_error=True, log_to_driver=True,
            runtime_env={"env_vars": thread_limit_env_vars()},
        )

    # Forward RLix's own scheduler tuning knobs from the driver environment
    # (Ray actors do not inherit the driver shell on multi-node clusters).
    # Caller-provided env_vars win on key collision.
    from rlix.utils.env import scheduler_env_passthrough
    merged_env = {**scheduler_env_passthrough(), **(env_vars or {})}
    opts = ConnectOptions(
        address=address, create_if_missing=create_if_missing, env_vars=merged_env or None
    )
    return _get_or_create_orchestrator(opts)


def _get_or_create_orchestrator(opts: ConnectOptions) -> Any:
    try:
        return ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=RLIX_NAMESPACE)
    except ValueError:
        if not opts.create_if_missing:
            raise

    strategy = head_node_affinity_strategy(soft=False)
    runtime_env = {"env_vars": dict(opts.env_vars or {})}
    for sleep_s in (0.0,) + opts.backoff_s:
        if sleep_s:
            time.sleep(sleep_s)
        try:
            return (
                ray.remote(Orchestrator)
                .options(
                    name=ORCHESTRATOR_ACTOR_NAME,
                    namespace=RLIX_NAMESPACE,
                    scheduling_strategy=strategy,
                    # Stays at max_restarts=0 by design. Orchestrator.__init__
                    # would re-bootstrap the scheduler topology on restart
                    # (re-running _ensure_scheduler_singleton.initialize), but
                    # the ``self._pipelines`` registry of admitted pipelines is
                    # in-memory and would be wiped — every active pipeline
                    # would become unreachable for admit / unregister. Cheaper
                    # to surface the underlying Ray race (worker.py:1039
                    # 'core_worker' AttributeError) as a hard failure than to
                    # mask it with a half-recovered state.
                    max_restarts=0,
                    max_task_retries=0,
                    # max_concurrency=4 mitigates the Ray-2.55.1 worker.py:1039
                    # ``'Worker' object has no attribute 'core_worker'`` race
                    # that fires when concurrent inbound RPCs (register_pipeline
                    # / admit_pipeline / unregister + the per-rollout
                    # signal_rollout_demand → scheduler.report_progress chain)
                    # contend on the default single-threaded actor task loop.
                    # The same fix is applied to MilesCoordinator
                    # (rlix/pipeline/miles_coordinator.py:602). 4 matches the
                    # number of cross-actor surfaces that can fire in parallel
                    # under the 2-pipeline 4-GPU topology.
                    max_concurrency=4,
                    runtime_env=runtime_env,
                )
                .remote(env_vars=opts.env_vars)
            )
        except Exception:
            try:
                return ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=RLIX_NAMESPACE)
            except ValueError:
                continue
    raise RuntimeError(f"Failed to create or get orchestrator actor {ORCHESTRATOR_ACTOR_NAME!r}")
