"""FFT-based circulant kernel on a ring of evenly spaced zones."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import numpy as np

from prefcent._hashing import frame_update


class CirculantKernel:
    """Symmetric circulant operator; never holds an N×N matrix."""

    def __init__(
        self,
        row: np.ndarray,
        *,
        n: int,
        spacing: float,
        d0: float,
        beta: float,
        self_interaction: bool,
        decay: dict[str, Any],
    ) -> None:
        self._row = np.array(row, dtype=np.float64, copy=True)
        self.shape: tuple[int, int] = (n, n)
        self.dtype: np.dtype[Any] = np.dtype(np.float64)
        self.is_symmetric = True
        self.self_interaction = self_interaction
        self.is_symmetric_source = "derived"
        self.content_sha256 = _circulant_content_sha256(self._row, n)
        self.consumer_identity: dict[str, Any] | None = None
        self.provenance: dict[str, Any] = {
            "builder": "ring",
            "n": int(n),
            "spacing": float(spacing),
            "decay": decay,
        }
        self.symmetry_check: dict[str, Any] | None = None
        self._row_rfft = np.fft.rfft(self._row)

    @classmethod
    def ring(
        cls,
        n: int,
        spacing: float,
        d0: float,
        beta: float,
        *,
        self_interaction: bool,
    ) -> CirculantKernel:
        """Ring distances ``min(|i−j|, n−|i−j|)·spacing`` with the named power decay.

        A translation-invariant test bed: it exercises properties that hold at
        any ``gamma`` — equivariance under rotation, conservation of mass, the
        uniform state being a fixed point — without ever storing an N×N matrix.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            raise ValueError("n must be an integer ≥ 2")
        if not isinstance(self_interaction, bool):
            raise TypeError("self_interaction must be a bool (no coercion)")
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing must be finite and > 0")
        if not np.isfinite(d0) or d0 <= 0.0:
            raise ValueError("d0 must be finite and > 0")
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and > 0")
        idx = np.arange(n, dtype=np.float64)
        dist = np.minimum(idx, n - idx) * float(spacing)
        row = (dist + float(d0)) ** (-float(beta))
        if not self_interaction:
            row[0] = 0.0
        decay = {
            "form": "power",
            "beta": float(beta),
            "d0": float(d0),
            "cost_units": None,
        }
        return cls(
            row,
            n=n,
            spacing=float(spacing),
            d0=float(d0),
            beta=float(beta),
            self_interaction=self_interaction,
            decay=decay,
        )

    def matvec(self, x: np.ndarray) -> np.ndarray:
        n = self.shape[0]
        vec = np.asarray(x, dtype=np.float64)
        if vec.shape != (n,):
            raise ValueError(f"vector must have shape {(n,)}")
        y = np.fft.irfft(self._row_rfft * np.fft.rfft(vec, n=n), n=n)
        return cast(np.ndarray, np.asarray(y, dtype=np.float64))

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)


def _circulant_content_sha256(row: np.ndarray, n: int) -> str:
    # The hash covers the generating row and n, then a UTF-8 layout tag, so a
    # circulant never collides with a dense kernel holding the same bytes.
    h = hashlib.sha256()
    frame_update(h, row)
    frame_update(h, np.asarray([int(n)], dtype=np.int64))
    h.update(b"layout:circulant")
    return h.hexdigest()
