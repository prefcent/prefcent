"""Opt-in adjointness probe: does ``rmatvec`` really apply the transpose?"""

from __future__ import annotations

import numpy as np

from prefcent._kernels import Kernel
from prefcent._result import Certificate


def adjointness_probe(
    kernel: Kernel,
    rng: np.random.Generator,
    n_probes: int = 8,
    rtol: float = 1e-10,
) -> Certificate:
    """Randomized ``⟨y, Kx⟩`` vs ``⟨Kᵀ y, x⟩`` check.

    For each Gaussian pair ``(x, y)``,
    ``err = |⟨y, Kx⟩ − ⟨Kᵀy, x⟩| / (‖y‖‖Kx‖ + ‖Kᵀy‖‖x‖)``.
    A zero denominator is ``err = 0`` if the numerator is also 0, else ``+inf``.
    """
    if n_probes < 1:
        raise ValueError("n_probes must be ≥ 1")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be > 0")
    n = int(kernel.shape[0])
    max_err = 0.0
    for _ in range(n_probes):
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        kx = np.asarray(kernel.matvec(x), dtype=np.float64)
        kty = np.asarray(kernel.rmatvec(y), dtype=np.float64)
        if (
            kx.shape != (n,)
            or kty.shape != (n,)
            or not np.isfinite(kx).all()
            or not np.isfinite(kty).all()
        ):
            # A non-finite or mis-shaped output can never certify adjointness,
            # so return an explicitly failed certificate rather than a number.
            return Certificate(
                name="adjointness",
                passed=False,
                data={
                    "max_err": float("inf"),
                    "n_probes": int(n_probes),
                    "rtol": float(rtol),
                    "nonfinite_output": 1,
                },
            )
        num = float(np.abs(np.dot(y, kx) - np.dot(kty, x)))
        den = float(
            np.linalg.norm(y) * np.linalg.norm(kx)
            + np.linalg.norm(kty) * np.linalg.norm(x)
        )
        if den == 0.0:
            err = 0.0 if num == 0.0 else float("inf")
        else:
            err = num / den
        if err > max_err:
            max_err = err
    return Certificate(
        name="adjointness",
        passed=bool(max_err <= rtol),
        data={"max_err": max_err, "n_probes": int(n_probes), "rtol": float(rtol)},
    )
