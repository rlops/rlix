from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from schedrl.orchestrator.orchestrator import (
    ORCHESTRATOR_ACTOR_NAME,
    SCHEDRL_NAMESPACE,
    AdmitResponse,
    Orchestrator,
)
from schedrl.utils.ray_head import head_node_affinity_strategy
import ray


@dataclass(frozen=True, slots=True)
class ConnectOptions:
    address: str = "auto"
    namespace: str = SCHEDRL_NAMESPACE
    create_if_missing: bool = True
    backoff_s: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    env_vars: Optional[dict[str, str]] = None


def connect(
    *,
    create_if_missing: bool = True,
    address: str = "auto",
    namespace: str = SCHEDRL_NAMESPACE,
    env_vars: Optional[dict[str, str]] = None,
):
    if not ray.is_initialized():
        ray.init(address=address, namespace=namespace, ignore_reinit_error=True, log_to_driver=True)

    # Ray actors don't inherit the driver's environment. Snapshot the full driver env
    # so SchedRL actors (orchestrator, scheduler) see the same vars the driver sees.
    # Explicit env_vars overrides take priority over the driver snapshot.
    driver_env: dict[str, str] = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    opts = ConnectOptions(address=address, namespace=namespace, create_if_missing=create_if_missing, env_vars=driver_env)
    return _get_or_create_orchestrator(opts)


def _get_or_create_orchestrator(opts: ConnectOptions):
    try:
        return ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=opts.namespace)
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
                    namespace=opts.namespace,
                    scheduling_strategy=strategy,
                    max_restarts=0,
                    max_task_retries=0,
                    runtime_env=runtime_env,
                )
                .remote(env_vars=opts.env_vars)
            )
        except Exception:
            try:
                return ray.get_actor(ORCHESTRATOR_ACTOR_NAME, namespace=opts.namespace)
            except ValueError:
                continue
    raise RuntimeError(f"Failed to create or get orchestrator actor {ORCHESTRATOR_ACTOR_NAME!r}")


def admit_pipeline(*, orchestrator, pipeline_id: str) -> AdmitResponse:
    return ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))
