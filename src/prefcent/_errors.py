"""Typed fail-stop errors. Input-contract violations are ``ValueError``."""

from __future__ import annotations

import numpy as np


class PrefcentError(Exception):
    """Common base for solver failures."""


class NumericalBreach(PrefcentError):
    """Non-finite or negative intermediate in the update map.

    ``step`` is attached by the evolve loop; a closure raising from inside
    ``apply`` may omit it (``None``) and the loop fills it in.
    """

    def __init__(
        self,
        message: str,
        *,
        step: int | None = None,
        state: np.ndarray | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.state = state


class ClosureDomainBreach(NumericalBreach):
    """Closed update left the closure's domain (e.g. a negative entry)."""


class KernelError(PrefcentError):
    """Kernel ``matvec`` / ``rmatvec`` failed or returned an invalid vector."""

    def __init__(
        self,
        message: str,
        *,
        step: int,
        state: np.ndarray | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.state = state


class ModelDomainError(PrefcentError):
    """The model is undefined on these inputs (sinks, all zones isolated)."""

    def __init__(
        self,
        message: str,
        *,
        indices: np.ndarray | None = None,
    ) -> None:
        super().__init__(message)
        self.indices = indices
