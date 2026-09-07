"""Closure protocol and the Identity closure."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from prefcent._landscape import Landscape
from prefcent._result import Certificate


class Closure(Protocol):
    """Post-update transform: ``apply`` every step, ``domain_check`` at the end."""

    name: str
    params: dict[str, float]
    # Identity of the *implementation*: a closure with the same name and
    # params but different code must not share a reproducible fingerprint.
    # Built-ins carry {"kind": "builtin", ...}; consumer closures carry
    # {"kind": "consumer", "value", "trusted"}, or None — which makes the run
    # non-reproducible by fingerprint.
    identity: dict[str, Any] | None

    def apply(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: Landscape,
        active: np.ndarray,
    ) -> np.ndarray: ...

    def domain_check(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: Landscape,
        active: np.ndarray,
    ) -> Certificate:
        """Margin certificate at the final state.

        Receives the raw balance update ``inflow`` evaluated *at* ``mass``: a
        post-update transform's domain margin is a function of the update it
        transforms, so a check without it could only be answered from state
        cached during ``apply`` — and closures are required to be pure.
        """
        ...


class Identity:
    """The identity closure: the closed update is the raw inflow."""

    name = "identity"
    params: dict[str, float]
    identity: dict[str, Any] | None

    def __init__(self) -> None:
        # Built-in identity is assigned by the package from the exact type;
        # this attribute is informational and is never trusted.
        self.params = {}
        self.identity = None

    def apply(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: Landscape,
        active: np.ndarray,
    ) -> np.ndarray:
        return inflow

    def domain_check(
        self,
        mass: np.ndarray,
        inflow: np.ndarray,
        landscape: Landscape,
        active: np.ndarray,
    ) -> Certificate:
        return Certificate(name=self.name, passed=True, data={})
