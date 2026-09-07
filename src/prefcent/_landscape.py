"""Capacity landscape: the model's fixed capacity ``R`` and optional labels."""

from __future__ import annotations

import numpy as np

from prefcent._hashing import hash_array


def _readonly(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


def _validate_labels_dtype(dtype: np.dtype) -> None:
    if dtype.kind in ("i", "u", "S", "U"):
        return
    raise ValueError(
        f"labels dtype must be integer, fixed-width bytes, or unicode; got {dtype!r}"
    )


class Landscape:
    """Fixed capacity ``R`` and optional opaque zone labels."""

    def __init__(
        self,
        capacity: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        cap = np.asarray(capacity)
        if cap.ndim != 1:
            raise ValueError(f"capacity must be 1-D, got shape {cap.shape}")
        if cap.shape[0] == 0:
            raise ValueError("capacity must be nonempty")
        if cap.dtype != np.float64:
            raise ValueError(f"capacity must be float64, got {cap.dtype}; never upcast")
        if not np.isfinite(cap).all():
            raise ValueError("capacity must be finite")
        if not bool(np.all(cap > 0.0)):
            raise ValueError("capacity must be > 0 everywhere")

        stored = np.array(cap, dtype=np.float64, copy=True)
        self._capacity = stored
        self._n = int(stored.shape[0])
        self._total_capacity = float(stored.sum())
        self._capacity_sha256 = hash_array(stored)

        if labels is None:
            self._labels: np.ndarray | None = None
            self._labels_sha256: str | None = None
        else:
            lab = np.asarray(labels)
            if lab.ndim not in (1, 2) or lab.shape[0] != self._n:
                raise ValueError(
                    "labels must be 1-D or 2-D with shape[0] == n "
                    f"(n={self._n}, got {lab.shape}) — e.g. an (n, 2) key array"
                )
            _validate_labels_dtype(lab.dtype)
            stored_lab = np.array(lab, copy=True)
            self._labels = stored_lab
            self._labels_sha256 = hash_array(stored_lab)

    @property
    def n(self) -> int:
        return self._n

    @property
    def total_capacity(self) -> float:
        return self._total_capacity

    @property
    def capacity(self) -> np.ndarray:
        return _readonly(self._capacity)

    @property
    def labels(self) -> np.ndarray | None:
        if self._labels is None:
            return None
        return _readonly(self._labels)

    @property
    def capacity_sha256(self) -> str:
        return self._capacity_sha256

    @property
    def labels_sha256(self) -> str | None:
        return self._labels_sha256
