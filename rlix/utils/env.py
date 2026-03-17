from __future__ import annotations

import os
from typing import Optional


def parse_env_timeout_s(env_key: str, default_s: float) -> Optional[float]:
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
