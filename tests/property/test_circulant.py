"""Circulant ring: matrix agreement, equivariance, uniform fixed point."""

from __future__ import annotations

import numpy as np
import pytest

import prefcent as pc


def _ring_row(n: int, spacing: float, d0: float, beta: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    dist = np.minimum(idx, n - idx) * spacing
    row = (dist + d0) ** (-beta)
    row[0] = 0.0
    return row


def _dense_from_row(row: np.ndarray) -> np.ndarray:
    n = row.shape[0]
    k = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        k[i] = np.roll(row, i)
    return k


@pytest.mark.property
def test_circulant_agrees_with_explicit_matrix_to_fft_roundoff() -> None:
    n = 32
    spacing, d0, beta = 1.0, 1.0, 2.0
    ring = pc.CirculantKernel.ring(
        n, spacing=spacing, d0=d0, beta=beta, self_interaction=False
    )
    dense = pc.DenseKernel(
        _dense_from_row(_ring_row(n, spacing, d0, beta)),
        is_symmetric=True,
        self_interaction=False,
    )
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    err = float(np.max(np.abs(ring.matvec(x) - dense.matvec(x))))
    rel = err / max(float(np.max(np.abs(dense.matvec(x)))), 1e-300)
    assert rel <= 1e-13


@pytest.mark.property
def test_rolling_capacity_rolls_mass() -> None:
    n = 24
    kernel = pc.CirculantKernel.ring(
        n, spacing=1.0, d0=1.0, beta=2.0, self_interaction=False
    )
    capacity = 1.0 + 0.2 * np.arange(n, dtype=np.float64)
    shift = 5
    r1 = pc.Model(kernel, pc.Landscape(capacity), gamma=0.5).evolve(max_iter=3)
    r2 = pc.Model(kernel, pc.Landscape(np.roll(capacity, shift)), gamma=0.5).evolve(
        max_iter=3
    )
    assert np.allclose(np.roll(r1.mass, shift), r2.mass, rtol=1e-12, atol=1e-12)


@pytest.mark.property
def test_uniform_capacity_is_an_exact_fixed_point() -> None:
    n = 16
    kernel = pc.CirculantKernel.ring(
        n, spacing=2.0, d0=1.0, beta=2.0, self_interaction=False
    )
    capacity = np.ones(n, dtype=np.float64)
    result = pc.Model(kernel, pc.Landscape(capacity), gamma=0.0).evolve(max_iter=1)
    rel = float(np.max(np.abs(result.mass - capacity)))
    assert rel <= 1e-15
