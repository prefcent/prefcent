"""Adjointness probe metric and pass/fail cases."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from tests.support import load_fixture

import prefcent as pc


@pytest.mark.property
def test_probe_passes_on_a_grid_dense_kernel(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    cert = pc.kernels.adjointness_probe(kernel, np.random.default_rng(0))
    assert cert.passed is True
    assert float(cast(float, cert.data["max_err"])) < 1e-14


@pytest.mark.property
def test_probe_passes_on_declared_symmetric_pa_small() -> None:
    _meta, z = load_fixture("pa_small")
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    cert = pc.kernels.adjointness_probe(kernel, np.random.default_rng(1))
    assert cert.passed is True


@pytest.mark.property
def test_probe_fails_on_a_non_adjoint_pair() -> None:
    n = 8
    rng = np.random.default_rng(2)
    k = rng.random((n, n))
    k2 = rng.random((n, n))
    op = pc.OperatorKernel.from_callables(
        lambda x: k @ x,
        lambda x: k2 @ x,
        n,
        is_symmetric=False,
        self_interaction=False,
        identity=None,
    )
    cert = pc.kernels.adjointness_probe(op, np.random.default_rng(3), rtol=1e-10)
    assert cert.passed is False


@pytest.mark.property
def test_zero_operator_passes_with_zero_max_err() -> None:
    n = 5
    z = np.zeros((n, n), dtype=np.float64)
    op = pc.OperatorKernel.from_callables(
        lambda x: z @ x,
        None,
        n,
        is_symmetric=True,
        self_interaction=False,
        identity="zero",
    )
    cert = pc.kernels.adjointness_probe(op, np.random.default_rng(4), n_probes=4)
    assert cert.passed is True
    assert cert.data["max_err"] == 0.0
