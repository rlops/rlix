"""CPU-resident bucket cache for PP collective gather and weight sync.

Each ``BucketRecord`` packs multiple named parameters into a single contiguous
uint8 CPU tensor (512-byte aligned offsets).  This format is shared between
the IPC path (cpu_serialize ZMQ multipart) and the NCCL broadcast path
(packed_broadcast_producer/consumer).

Two-pointer versioning mirrors ROLL ``megatron_strategy.py:1049–1065``:
- ``build_latest(version, buckets)`` — store a new version (not yet active).
- ``promote(version)`` — atomically make it active; GC old versions.
- ``get_active_buckets()`` — read active version (caller holds ``_cache_lock``).

Thread-safety:
    All public methods acquire ``_cache_lock``.  ``selective_sync_active_cache``
    holds the lock for the entire per-bucket transport loop (prevents a
    concurrent ``promote`` / ``build_latest`` from racing the sender read).

Typical lifecycle::

    cache = VersionedBucketCache()

    # --- init (base model) ---
    cache.build_latest(-1, pack_model_weights(base_model))
    cache.promote(-1)

    # --- post train-step ---
    cache.build_latest(step, pack_model_weights(new_model))
    cache.promote(step)

    # --- sync ---
    with cache._cache_lock:
        buckets = cache.get_active_buckets()
        for b in buckets:
            transport(b)
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    _Tensor = torch.Tensor
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    import types as _types
    _torch_stub = _types.ModuleType("torch")

    class _Tensor:  # type: ignore[no-redef]
        pass

    _torch_stub.Tensor = _Tensor  # type: ignore[attr-defined]
    torch = _torch_stub  # type: ignore[assignment]
    _HAS_TORCH = False


# 512-byte alignment matches NeMo RL ``policy/utils.py:calculate_aligned_size``
_ALIGNMENT = 512


def _aligned_offset(offset: int, alignment: int = _ALIGNMENT) -> int:
    """Round *offset* up to the next multiple of *alignment*."""
    return (offset + alignment - 1) // alignment * alignment


@dataclass
class BucketRecord:
    """Single packed weight buffer containing one or more named parameters.

    All parameters are flattened, cast to uint8, and concatenated into a
    single contiguous CPU tensor with 512-byte-aligned boundaries between
    them.  This layout is directly usable as a ``cpu_serialize`` payload for
    the ZMQ IPC path and as a broadcast buffer for the NCCL path.

    Attributes:
        param_names: HF param names packed in this buffer, in order.
        shapes:      Per-param original shapes (used to split after receive).
        dtypes:      Per-param original dtypes (used to cast after receive).
        offsets:     Byte offsets into ``cpu_uint8_bucket`` for each param
                     (length == len(param_names)).
        used_bytes:  Total bytes actually written (bucket may be over-allocated).
        cpu_uint8_bucket: Contiguous uint8 CPU tensor holding all params.
    """

    param_names: List[str]
    shapes: List  # List[torch.Size]
    dtypes: List  # List[torch.dtype]
    offsets: List[int]
    used_bytes: int
    cpu_uint8_bucket: _Tensor


def _bucket_named_tensors(
    named_tensors: List[Tuple[str, _Tensor]],
) -> BucketRecord:
    """Pack a list of ``(name, tensor)`` pairs into a single ``BucketRecord``.

    Each tensor is flattened and viewed as uint8, then concatenated with
    512-byte alignment padding between params (mirrors ROLL's
    ``send_recv_utils.py:214`` ``serialize_named_weights`` and NeMo RL's
    ``calculate_aligned_size``).

    Args:
        named_tensors: Non-empty list of ``(param_name, cpu_tensor)`` pairs.
            Tensors must already be on CPU.

    Returns:
        A ``BucketRecord`` with all params packed into
        ``cpu_uint8_bucket``.

    Raises:
        ValueError: If *named_tensors* is empty.
    """
    if not named_tensors:
        raise ValueError("named_tensors must be non-empty")

    param_names: List[str] = []
    shapes = []
    dtypes = []
    uint8_views: List[_Tensor] = []
    offsets: List[int] = []
    current_offset = 0

    for name, tensor in named_tensors:
        shape = tensor.shape
        dtype = tensor.dtype
        # Flatten + view as uint8 (same as ROLL send_recv_utils.py:214)
        uint8_view = tensor.detach().cpu().contiguous().flatten().view(torch.uint8)
        nbytes = uint8_view.numel()

        offsets.append(current_offset)
        param_names.append(name)
        shapes.append(shape)
        dtypes.append(dtype)
        uint8_views.append(uint8_view)

        aligned = _aligned_offset(current_offset + nbytes)
        current_offset = aligned

    used_bytes = sum(t.numel() for t in uint8_views)
    # Total allocated size includes alignment padding
    total_bytes = current_offset

    # Allocate contiguous buffer and copy each param into its aligned slot
    bucket_buf = torch.zeros(total_bytes, dtype=torch.uint8)
    for i, uint8_view in enumerate(uint8_views):
        start = offsets[i]
        nbytes = uint8_view.numel()
        bucket_buf[start : start + nbytes].copy_(uint8_view)

    return BucketRecord(
        param_names=param_names,
        shapes=shapes,
        dtypes=dtypes,
        offsets=offsets,
        used_bytes=used_bytes,
        cpu_uint8_bucket=bucket_buf,
    )


def unpack_bucket_record(
    record: BucketRecord,
) -> List[Tuple[str, _Tensor]]:
    """Unpack a ``BucketRecord`` into a list of ``(name, tensor)`` pairs.

    Inverse of ``_bucket_named_tensors``.  Used on the receiver side
    (``update_parameter_in_bucket``) to reconstruct per-param tensors.

    Args:
        record: Packed bucket as produced by ``_bucket_named_tensors``.

    Returns:
        List of ``(param_name, tensor)`` in original order and dtype.
    """
    result: List[Tuple[str, _Tensor]] = []
    buf = record.cpu_uint8_bucket
    for name, shape, dtype, offset in zip(
        record.param_names, record.shapes, record.dtypes, record.offsets
    ):
        num_elements = 1
        for s in shape:
            num_elements *= s
        # Use torch.empty to get element size — never slice a uint8 buffer and view
        # as a wider dtype (e.g. 1 uint8 byte cannot be viewed as float32 in real torch).
        element_bytes = torch.empty(0, dtype=dtype).element_size()
        nbytes = num_elements * element_bytes
        flat = buf[offset : offset + nbytes].view(dtype)
        tensor = flat.reshape(shape)
        result.append((name, tensor))
    return result


class VersionedBucketCache:
    """Thread-safe two-pointer CPU bucket cache with version tracking.

    Mirrors ROLL ``megatron_strategy.py:1049–1065``:
    - ``_latest_cached``: version just built (may not be active yet).
    - ``_active_cached``: version safe to read for sync.

    Only the cache owner (pp_rank==0, dp_rank==0, tp_rank==0, cp_rank==0)
    ever stores buckets.  Non-owner workers hold an empty cache and return
    immediately from ``build_latest`` / ``promote``.

    GC invariant:
        After each ``promote(v)`` call, all versions except
        ``_latest_cached`` and ``_active_cached`` are deleted from
        ``_cache_map``.  This keeps peak memory bounded to ≤ 2×model.
    """

    def __init__(self) -> None:
        self._cache_map: Dict[int, List[BucketRecord]] = {}
        self._latest_cached: Optional[int] = None
        self._active_cached: Optional[int] = None
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write operations (called from training worker)
    # ------------------------------------------------------------------

    def build_latest(self, version: int, buckets: List[BucketRecord]) -> None:
        """Store *buckets* as the 'latest' version.

        Does **not** make this version active.  The pipeline calls
        ``promote(version)`` separately after confirming the training step
        has fully completed.

        Args:
            version: Checkpoint version (step number, or ``-1`` for base model).
            buckets: List of ``BucketRecord`` packed by ``_bucket_named_tensors``.
        """
        with self._cache_lock:
            self._cache_map[version] = list(buckets)
            self._latest_cached = version
            self._gc_unlocked()

    def promote(self, version: int) -> None:
        """Switch the active pointer to *version*.

        After this call, ``get_active_buckets()`` returns the buckets for
        *version*.  Old versions (except ``_latest_cached``) are GC'd.

        Args:
            version: Must match a version passed to a prior ``build_latest``
                call.  Raises ``KeyError`` if *version* was never built.
        """
        with self._cache_lock:
            if version not in self._cache_map:
                raise KeyError(
                    f"VersionedBucketCache.promote: version {version} not found "
                    f"(built versions: {sorted(self._cache_map)})"
                )
            self._active_cached = version
            self._gc_unlocked()

    def get_active_buckets(self) -> List[BucketRecord]:
        """Return the buckets for the currently active version.

        Must be called with ``_cache_lock`` held (caller is responsible).
        Raises ``RuntimeError`` if ``promote()`` has never been called.
        """
        if self._active_cached is None:
            raise RuntimeError(
                "VersionedBucketCache: promote() has never been called. "
                "Call build_latest() + promote() before reading active buckets."
            )
        return self._cache_map[self._active_cached]

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    @property
    def cache_ready_step(self) -> Optional[int]:
        """The currently active version, or ``None`` if never promoted."""
        with self._cache_lock:
            return self._active_cached

    @property
    def latest_version(self) -> Optional[int]:
        """The most recently built version, or ``None`` if never built."""
        with self._cache_lock:
            return self._latest_cached

    def is_version_built(self, version: int) -> bool:
        """Return ``True`` if *version* has been built but not necessarily promoted."""
        with self._cache_lock:
            return version in self._cache_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gc_unlocked(self) -> None:
        """Delete all versions except ``_latest_cached`` and ``_active_cached``.

        Called while holding ``_cache_lock`` — do NOT re-acquire.
        """
        keep = {v for v in (self._latest_cached, self._active_cached) if v is not None}
        stale = [v for v in self._cache_map if v not in keep]
        for v in stale:
            del self._cache_map[v]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        with self._cache_lock:
            versions = sorted(self._cache_map)
            return (
                f"VersionedBucketCache("
                f"active={self._active_cached}, "
                f"latest={self._latest_cached}, "
                f"versions={versions})"
            )
