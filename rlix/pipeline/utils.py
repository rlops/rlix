from __future__ import annotations

from typing import List


def validate_resize_params(dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> None:
    """Validate resize_infer parameters: exactly one list must be non-empty."""
    if not isinstance(dp_ranks_to_remove, list):
        raise ValueError("dp_ranks_to_remove must be list[int]")
    if not isinstance(dp_ranks_to_add, list):
        raise ValueError("dp_ranks_to_add must be list[int]")
    if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
        raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")
