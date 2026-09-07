"""OperatorKernel: wrap a ``matvec`` operator without inspecting entries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from prefcent._kernels import _as_consumer_identity


class OperatorKernel:
    """Protocol adapter around an operator with ``shape``, ``dtype``, ``matvec``.

    ``identity`` has no default: pass a consumer identity, or spell
    ``identity=None`` to record ``consumer_identity: null``.
    """

    def __init__(
        self,
        op: Any,
        *,
        is_symmetric: bool,
        self_interaction: bool,
        identity: str | dict[str, Any] | None,
    ) -> None:
        if not isinstance(is_symmetric, bool):
            raise TypeError("is_symmetric is required and must be bool (no default)")
        if not isinstance(self_interaction, bool):
            raise TypeError("self_interaction must be bool")
        for attr in ("shape", "dtype", "matvec"):
            if not hasattr(op, attr):
                raise TypeError(f"operator must expose {attr}")
        if not is_symmetric and not hasattr(op, "rmatvec"):
            raise TypeError("rmatvec is required unless is_symmetric")
        shape = tuple(int(s) for s in op.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"operator must be square, got shape {shape}")
        dtype = np.dtype(op.dtype)
        if dtype != np.dtype(np.float64):
            raise ValueError(
                f"v0.1.0 accepts float64 kernels only, got {dtype}; never upcast"
            )
        self._op = op
        self.shape: tuple[int, int] = (shape[0], shape[1])
        self.dtype: np.dtype[Any] = np.dtype(np.float64)
        self.is_symmetric = is_symmetric
        self.self_interaction = self_interaction
        self.is_symmetric_source = "declared"
        self.content_sha256: str | None = None
        self.consumer_identity = _as_consumer_identity(identity)
        self.provenance: dict[str, Any] | None = None
        self.symmetry_check: dict[str, Any] | None = None

    def _as_vec(self, x: np.ndarray) -> np.ndarray:
        n = self.shape[0]
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape == (n, 1):
            arr = arr.reshape(n)
        if arr.shape != (n,):
            raise ValueError(f"vector must have shape {(n,)} or {(n, 1)}")
        if arr.dtype != np.float64:
            raise ValueError("vector must be float64; never upcast")
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _from_op(self, y: Any) -> np.ndarray:
        n = self.shape[0]
        out = np.asarray(y)
        if out.ndim == 2 and out.shape == (n, 1):
            out = out.reshape(n)
        if out.shape != (n,):
            raise ValueError(
                f"operator output must have shape {(n,)} or {(n, 1)}, got {out.shape}"
            )
        if out.dtype != np.float64:
            raise ValueError("operator output must be float64; never upcast")
        return np.ascontiguousarray(out, dtype=np.float64)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        # Exceptions from the wrapped operator propagate unchanged; the evolve
        # loop wraps them in KernelError with the iteration index attached.
        return self._from_op(self._op.matvec(self._as_vec(x)))

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        vec = self._as_vec(x)
        if self.is_symmetric:
            return self._from_op(self._op.matvec(vec))
        return self._from_op(self._op.rmatvec(vec))

    @classmethod
    def from_callables(
        cls,
        matvec: Callable[[np.ndarray], np.ndarray],
        rmatvec: Callable[[np.ndarray], np.ndarray] | None,
        n: int,
        *,
        is_symmetric: bool,
        self_interaction: bool,
        identity: str | dict[str, Any] | None,
    ) -> OperatorKernel:
        if rmatvec is None and not is_symmetric:
            raise ValueError("rmatvec is required unless is_symmetric")
        rmv = matvec if rmatvec is None else rmatvec

        class _Op:
            shape = (n, n)
            dtype = np.dtype(np.float64)

            def matvec(self, x: np.ndarray) -> np.ndarray:
                return matvec(x)

            def rmatvec(self, x: np.ndarray) -> np.ndarray:
                return rmv(x)

        return cls(
            _Op(),
            is_symmetric=is_symmetric,
            self_interaction=self_interaction,
            identity=identity,
        )
