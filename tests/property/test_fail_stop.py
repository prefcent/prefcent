"""Fail-stop on constructed kernel, closure, and input breaches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import prefcent as pc


class _ShiftedClosure:
    identity: dict[str, Any] | None = None
    name = "shift"
    params: dict[str, float] = {}

    def __init__(self, shift: float) -> None:
        self.params = {}
        self._shift = shift

    def apply(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: pc.Landscape,
        active: np.ndarray,
    ) -> np.ndarray:
        return inflow - self._shift

    def domain_check(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: pc.Landscape,
        active: np.ndarray,
    ) -> pc.Certificate:
        return pc.Certificate(name=self.name, passed=True, data={})


class _NegativeMatvec(pc.DenseKernel):
    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = super().matvec(x)
        out = np.array(y, dtype=np.float64, copy=True)
        out[0] = -1.0
        return out


@pytest.mark.property
def test_closure_returning_a_shifted_inflow_raises_with_step_and_state(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0, closure=_ShiftedClosure(1e9))
    with pytest.raises(pc.ClosureDomainBreach) as excinfo:
        model.evolve(max_iter=1)
    err = excinfo.value
    assert err.step == 1
    assert err.state is not None


@pytest.mark.property
def test_kernel_with_a_nan_raises_at_construction() -> None:
    arr = np.ones((3, 3), dtype=np.float64)
    arr[0, 1] = np.nan
    with pytest.raises(ValueError):
        pc.DenseKernel(arr, is_symmetric=True, self_interaction=False)


@pytest.mark.property
def test_matvec_negative_entry_raises_kernel_error(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = _NegativeMatvec(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    with pytest.raises(pc.KernelError):
        model.evolve(max_iter=1)


@pytest.mark.property
def test_breach_raised_inside_a_closure_gets_the_step_attached(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    class _RaisesOnThird:
        name = "raises"
        params: dict[str, float] = {}
        identity: dict[str, Any] | None = None
        calls = 0

        def apply(
            self,
            mass: np.ndarray,
            inflow: np.ndarray,
            landscape: pc.Landscape,
            active: np.ndarray,
        ) -> np.ndarray:
            self.calls += 1
            if self.calls == 3:
                raise pc.ClosureDomainBreach("left the domain")  # no step given
            return np.asarray(inflow)

        def domain_check(
            self,
            mass: np.ndarray,
            inflow: np.ndarray,
            landscape: pc.Landscape,
            active: np.ndarray,
        ) -> pc.Certificate:
            return pc.Certificate(name=self.name, passed=True, data={})

    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    with pytest.raises(pc.ClosureDomainBreach) as ei:
        pc.Model(kernel, landscape, gamma=1.0, closure=_RaisesOnThird()).evolve(
            max_iter=10
        )
    assert ei.value.step == 3
    assert ei.value.state is not None and ei.value.state.shape == (landscape.n,)
