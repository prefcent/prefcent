"""The density penalty closure: additive negative feedback on relative density."""

from __future__ import annotations

from typing import Any

import numpy as np

from prefcent._errors import ClosureDomainBreach, NumericalBreach
from prefcent._landscape import Landscape
from prefcent._result import Certificate


class DensityPenaltyV1:
    """Additive density penalty, capacity-weighted:

    ``C = inflow − kappa · capacity ⊙ (g − ḡ_R)`` with ``g = (mass/capacity)^rho``
    and ``ḡ_R`` the capacity-weighted mean of ``g`` over the active set.

    The penalty is capacity-weighted mean-zero over the active set, so it conserves
    total mass exactly; it is also invariant under subdividing a zone into smaller
    zones of the same total capacity (a property test asserts both). ``kappa = 0``
    is exactly the identity closure's arithmetic.

    Fail-stop: if the penalty drives any active zone negative, ``apply`` raises
    ``ClosureDomainBreach`` (the evolve loop attaches the step and state) — the
    state is never clamped or floored. ``domain_check`` reports the margin at the
    final state: the minimum of the closed update and ``kappa_at_zero``, the
    interpolated ``kappa`` at which the first active zone would reach zero. At a
    state of uniform density the penalty vanishes, so ``kappa_at_zero`` is
    ``"unbounded"``. Otherwise it is a finite number. Non-finite calculations
    raise ``NumericalBreach`` with state attached. At ``kappa = 0`` the update
    bypasses penalty arithmetic; if the optional margin calculation fails
    numerically, ``kappa_at_zero`` is ``"unavailable"``. These strings keep the
    certificate and its manifest JSON-safe without confusing an unbounded margin
    with one that could not be computed.
    """

    name = "density_penalty_v1"
    params: dict[str, float]
    identity: dict[str, Any] | None = None  # built-in identity is assigned by type

    def __init__(self, kappa: float, rho: float = 2.0) -> None:
        if not np.isfinite(kappa) or kappa < 0.0:
            raise ValueError("kappa must be finite and ≥ 0")
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("rho must be finite and > 0")
        self.params = {"kappa": float(kappa), "rho": float(rho)}

    def _penalty(
        self, mass: np.ndarray, landscape: Landscape, active: np.ndarray
    ) -> np.ndarray:
        capacity = landscape.capacity
        g = np.zeros_like(mass)
        p = np.zeros_like(mass)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                g[active] = (mass[active] / capacity[active]) ** self.params["rho"]
                gbar = float(
                    (capacity[active] * g[active]).sum() / capacity[active].sum()
                )
                p[active] = capacity[active] * (g[active] - gbar)
        except FloatingPointError as exc:
            raise NumericalBreach(
                f"{self.name}: penalty calculation failed numerically", state=mass
            ) from exc
        if not np.isfinite(p).all():
            raise NumericalBreach(f"{self.name}: penalty is non-finite", state=mass)
        return p

    def _closed_update(
        self, mass: np.ndarray, inflow: np.ndarray, penalty: np.ndarray
    ) -> np.ndarray:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                closed = inflow - self.params["kappa"] * penalty
        except FloatingPointError as exc:
            raise NumericalBreach(
                f"{self.name}: closed update failed numerically", state=mass
            ) from exc
        if not np.isfinite(closed).all():
            raise NumericalBreach(
                f"{self.name}: closed update is non-finite", state=mass
            )
        return closed

    def apply(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: Landscape,
        active: np.ndarray,
    ) -> np.ndarray:
        if self.params["kappa"] == 0.0:
            return inflow
        closed = self._closed_update(
            mass, inflow, self._penalty(mass, landscape, active)
        )
        if bool(np.any(closed[active] < 0.0)):
            n_neg = int((closed[active] < 0.0).sum())
            raise ClosureDomainBreach(
                f"{self.name}: the penalty drove {n_neg} active zones negative",
                state=mass,
            )
        return closed

    def domain_check(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: Landscape,
        active: np.ndarray,
    ) -> Certificate:
        kappa = self.params["kappa"]
        if not np.isfinite(inflow).all():
            raise NumericalBreach(
                f"{self.name}: certificate inflow is non-finite", state=mass
            )
        closed = inflow
        kappa_at_zero: float | str
        try:
            p = self._penalty(mass, landscape, active)
            if kappa != 0.0:
                closed = self._closed_update(mass, inflow, p)
            pos = active & (p > 0.0)
            if bool(pos.any()):
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        kappa_at_zero = float((inflow[pos] / p[pos]).min())
                except FloatingPointError as exc:
                    raise NumericalBreach(
                        f"{self.name}: margin calculation failed numerically",
                        state=mass,
                    ) from exc
                if not np.isfinite(kappa_at_zero):
                    raise NumericalBreach(
                        f"{self.name}: margin is non-finite", state=mass
                    )
            else:
                kappa_at_zero = "unbounded"
        except NumericalBreach:
            if kappa != 0.0:
                raise
            kappa_at_zero = "unavailable"
        return Certificate(
            name=self.name,
            passed=bool(np.all(closed[active] >= 0.0)),
            data={
                "min_closed_mass": float(closed[active].min()),
                "kappa": kappa,
                "kappa_at_zero": kappa_at_zero,
            },
        )
