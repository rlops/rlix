from __future__ import annotations

import os
from typing import Dict, Optional


def thread_limit_env_vars() -> Dict[str, str]:
    """Thread-count limits to stay under container pids.max.

    Safe to call from any context (driver or actor). Returns defaults when
    the env vars are not set; shell exports override the defaults.
    """
    env_vars: Dict[str, str] = {}
    for var, default in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("RAY_num_server_call_thread", "4"),
    ):
        env_vars[var] = os.environ.get(var, default)
    return env_vars


def pipeline_identity_env_vars(*, pipeline_id: str, ray_namespace: str) -> Dict[str, str]:
    """Pipeline identity vars for Ray actor runtime_env.

    Reads ``RLIX_CONTROL_PLANE`` from the environment so that actors inside an
    existing pipeline preserve the inherited value; defaults to ``"rlix"`` when
    called from the driver.
    """
    return {
        "PIPELINE_ID": pipeline_id,
        "ROLL_RAY_NAMESPACE": ray_namespace,
        "RLIX_CONTROL_PLANE": os.environ.get("RLIX_CONTROL_PLANE", "rlix"),
    }


def resolve_nemo_rl_pipeline_namespace(*, default: str = "roll") -> str:
    """Ray namespace for NeMo RL child actors, read from the inherited runtime env.

    Intended for NeMo RL's Ray actors (e.g. ``AsyncTrajectoryCollector``,
    ``ReplayBuffer``, ``ModelUpdateService``) that need to be created in the
    per-pipeline namespace propagated by the rlix driver via
    :func:`pipeline_identity_env_vars`.

    Reads ``ROLL_RAY_NAMESPACE`` from the environment. When running under the
    rlix control plane (``RLIX_CONTROL_PLANE=rlix``) the env var must be
    set — otherwise the actor would leak into the default namespace and
    break cross-pipeline isolation. In standalone mode falls back to
    *default*.

    Intentionally independent of the inline read in
    ``rlix/pipeline/full_finetune_pipeline.py`` (ROLL side) so the NeMo RL
    error path stays distinct.

    Raises ValueError when ``RLIX_CONTROL_PLANE=rlix`` and
    ``ROLL_RAY_NAMESPACE`` is unset or empty.
    """
    raw = os.environ.get("ROLL_RAY_NAMESPACE")
    if os.environ.get("RLIX_CONTROL_PLANE") == "rlix" and not raw:
        raise ValueError(
            "NeMo RL child actor requires ROLL_RAY_NAMESPACE env var when "
            "RLIX_CONTROL_PLANE=rlix; the rlix driver must propagate it via "
            "runtime_env (see pipeline_identity_env_vars())."
        )
    return raw if raw else default


def parse_env_timeout_s(env_key: str, default_s: Optional[float] = None) -> Optional[float]:
    """Read a timeout in seconds from an env var; fail-fast on invalid values.

    Returns *default_s* when the env var is unset.  Returns ``None`` when the
    env var is explicitly set to a value <= 0, which callers should interpret
    as "no timeout" (i.e. wait indefinitely).

    Raises RuntimeError if the value cannot be parsed as a number.
    """
    raw = os.environ.get(env_key)
    if raw is None:
        return default_s
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{env_key} must be a number, got: {raw!r}") from exc
    return None if value <= 0 else value
