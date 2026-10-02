"""One iterate of the model's update map."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from prefcent._closure import Closure
from prefcent._errors import ClosureDomainBreach, KernelError, NumericalBreach
from prefcent._landscape import Landscape

Matvec = Callable[[np.ndarray, int], np.ndarray]


def _readonly(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


def _require_finite_nonneg(
    vec: np.ndarray,
    *,
    step: int,
    state: np.ndarray,
    n: int,
    what: str,
) -> np.ndarray:
    if not isinstance(vec, np.ndarray):
        raise KernelError(f"{what} did not return an ndarray", step=step, state=state)
    if vec.shape != (n,) or vec.dtype != np.float64:
        raise KernelError(
            f"{what} must return float64 shape {(n,)}; "
            f"got dtype={vec.dtype} shape={vec.shape}",
            step=step,
            state=state,
        )
    if not np.isfinite(vec).all():
        raise KernelError(f"{what} produced a non-finite entry", step=step, state=state)
    if bool(np.any(vec < 0.0)):
        raise KernelError(f"{what} produced a negative entry", step=step, state=state)
    return vec


def apply_update(
    *,
    mass: np.ndarray,
    capacity: np.ndarray,
    gamma: float,
    omega: float,
    closure: Closure,
    landscape: Landscape,
    active: np.ndarray,
    total_capacity: np.float64,
    n_active: int,
    n: int,
    step: int,
    matvec: Matvec,
    rmatvec: Matvec,
) -> np.ndarray:
    """One complete iterate. At ``omega == 1.0`` skip the convex combination."""
    attraction = gamma * mass + capacity
    attraction_potential = matvec(attraction, step)
    attraction_potential = _require_finite_nonneg(
        attraction_potential, step=step, state=mass, n=n, what="matvec"
    )

    if n_active == n:
        share_rate = mass / attraction_potential
    else:
        share_rate = np.zeros(n, dtype=np.float64)
        share_rate[active] = mass[active] / attraction_potential[active]

    raw = rmatvec(share_rate, step)
    raw = _require_finite_nonneg(raw, step=step, state=mass, n=n, what="rmatvec")
    inflow = attraction * raw
    if not np.isfinite(inflow).all():
        raise NumericalBreach("inflow is non-finite", step=step, state=mass)
    if bool(np.any(inflow < 0.0)):
        raise NumericalBreach("inflow has a negative entry", step=step, state=mass)

    mass_ro = _readonly(mass)
    inflow_ro = _readonly(inflow)
    try:
        closed = closure.apply(mass_ro, inflow_ro, landscape, _readonly(active))
    except NumericalBreach as exc:
        if exc.step is None:
            # A closure cannot know the iteration index; the loop attaches it
            # and the state the closure was called with.
            raise type(exc)(
                str(exc.args[0]) if exc.args else "closure domain breach",
                step=step,
                state=mass if exc.state is None else exc.state,
            ) from exc
        raise
    if not isinstance(closed, np.ndarray):
        raise ClosureDomainBreach(
            "closure.apply did not return an ndarray", step=step, state=mass
        )
    if closed.shape != (n,) or closed.dtype != np.float64:
        raise ClosureDomainBreach(
            "closure.apply must return float64 shape "
            f"{(n,)}; got dtype={closed.dtype} shape={closed.shape}",
            step=step,
            state=mass,
        )
    if not np.isfinite(closed).all():
        raise ClosureDomainBreach(
            "closure.apply produced a non-finite entry", step=step, state=mass
        )
    if bool(np.any(closed < 0.0)):
        raise ClosureDomainBreach(
            "closure.apply produced a negative entry", step=step, state=mass
        )

    closed = np.array(closed, dtype=np.float64, copy=True)
    closed[~active] = 0.0
    if omega == 1.0:
        damped_mass = closed
    else:
        damped_mass = (1.0 - omega) * mass + omega * closed
        damped_mass[~active] = 0.0
    if not np.isfinite(damped_mass).all() or bool(np.any(damped_mass < 0.0)):
        raise NumericalBreach(
            "damped update is non-finite or negative", step=step, state=mass
        )

    denom = damped_mass.sum() if n_active == n else damped_mass[active].sum()
    if not np.isfinite(denom) or denom <= 0.0:
        raise NumericalBreach(
            "active closed mass is not positive", step=step, state=mass
        )
    new_mass = damped_mass * (total_capacity / denom)
    new_mass[~active] = 0.0
    if not np.isfinite(new_mass).all() or bool(np.any(new_mass < 0.0)):
        raise NumericalBreach(
            "renormalised mass is non-finite or negative", step=step, state=mass
        )
    return cast(np.ndarray, new_mass)
