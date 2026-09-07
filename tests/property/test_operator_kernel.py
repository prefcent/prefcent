"""OperatorKernel contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

import prefcent as pc


@pytest.mark.property
def test_operator_kernel_evolve_matches_dense_bit_for_bit(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    arr = z["K_beta2.0"]
    landscape = pc.Landscape(z["R"])
    dense = pc.DenseKernel(arr, is_symmetric=True, self_interaction=False)
    op = pc.OperatorKernel(
        aslinearoperator(arr),
        is_symmetric=True,
        self_interaction=False,
        identity="grid12-K",
    )
    r_dense = pc.Model(dense, landscape, gamma=1.0).evolve(max_iter=5)
    r_op = pc.Model(op, landscape, gamma=1.0).evolve(max_iter=5)
    assert np.array_equal(r_dense.mass, r_op.mass)


@pytest.mark.property
def test_float32_operator_raises() -> None:
    n = 3
    arr = np.eye(n, dtype=np.float32)

    class _Op:
        shape = (n, n)
        dtype = np.dtype(np.float32)

        def matvec(self, x: np.ndarray) -> np.ndarray:
            return np.asarray(arr @ x)

    with pytest.raises(ValueError):
        pc.OperatorKernel(
            _Op(), is_symmetric=True, self_interaction=False, identity=None
        )


@pytest.mark.property
def test_operator_kernel_identity_has_no_default() -> None:
    n = 2
    arr = np.eye(n, dtype=np.float64)

    class _Op:
        shape = (n, n)
        dtype = np.dtype(np.float64)

        def matvec(self, x: np.ndarray) -> np.ndarray:
            return np.asarray(arr @ x)

    with pytest.raises(TypeError):
        pc.OperatorKernel(_Op(), is_symmetric=True, self_interaction=False)  # type: ignore[call-arg]


@pytest.mark.property
def test_operator_negative_matvec_raises_kernel_error_with_step(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    arr = z["K_beta2.0"]

    def mv(x: np.ndarray) -> np.ndarray:
        y = arr @ x
        out = np.array(y, copy=True)
        out[0] = -1.0
        return out

    op = pc.OperatorKernel.from_callables(
        mv,
        None,
        arr.shape[0],
        is_symmetric=True,
        self_interaction=False,
        identity=None,
    )
    model = pc.Model(op, pc.Landscape(z["R"]), gamma=1.0)
    with pytest.raises(pc.KernelError) as excinfo:
        model.evolve(max_iter=1)
    assert excinfo.value.step == 0
