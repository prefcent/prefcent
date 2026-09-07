"""Isolation certificate and sink rejection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import prefcent as pc


@pytest.mark.property
def test_appended_zero_row_and_column_is_isolated_with_zero_mass(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    k = z["K_beta2.0"]
    r = z["R"]
    n = k.shape[0]
    k2 = np.zeros((n + 1, n + 1), dtype=np.float64)
    k2[:n, :n] = k
    extra = 42.0
    r2 = np.concatenate([r, np.array([extra], dtype=np.float64)])
    landscape = pc.Landscape(r2)
    kernel = pc.DenseKernel(k2, is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    result = model.evolve(max_iter=3)
    cert = result.certificates["isolated_zones"]
    assert cert.passed is True
    assert cert.data["indices"] == (n,)
    assert cert.data["count"] == 1
    assert cert.data["excluded_capacity"] == extra
    assert result.mass[n] == 0.0
    assert not bool(result.active[n])
    assert abs(float(result.mass.sum()) - float(r.sum())) <= 1e-9 * float(r.sum())


@pytest.mark.property
def test_zero_row_with_nonzero_column_raises_model_domain_error(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    k = np.array(z["K_beta2.0"], dtype=np.float64, copy=True)
    k[0, :] = 0.0
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(k, is_symmetric=False, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    with pytest.raises(pc.ModelDomainError):
        model.evolve(max_iter=1)
