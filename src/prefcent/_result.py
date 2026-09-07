"""Public record types returned by ``Model.evolve``."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from prefcent._landscape import Landscape


class EvolveStatus(enum.Enum):
    CONVERGED = "CONVERGED"
    FIXED_BUDGET = "FIXED_BUDGET"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, eq=False)
class Certificate:
    name: str
    passed: bool
    data: Mapping[str, float | int | str | tuple[int, ...]]

    def __post_init__(self) -> None:
        # Deep-frozen: lists become tuples and the mapping becomes read-only,
        # so a certificate cannot be edited after the run that produced it.
        frozen = {
            k: tuple(int(i) for i in v) if isinstance(v, list) else v
            for k, v in dict(self.data).items()
        }
        object.__setattr__(self, "data", MappingProxyType(frozen))


@dataclass(frozen=True, eq=False)
class ControllerState:
    kind: str


@dataclass(frozen=True, eq=False)
class StepInfo:
    iteration: int
    step_norm: float
    mass: np.ndarray
    stats_so_far: Mapping[str, float]


@dataclass(frozen=True, eq=False)
class EvolveResult:
    mass: np.ndarray
    potential: np.ndarray
    landscape: Landscape
    active: np.ndarray
    status: EvolveStatus
    iterations: int
    final_step_norm: float
    criterion: str
    tol: float | None
    max_iter: int
    omega: float
    step_norm_history: np.ndarray
    matvec_count: int
    rmatvec_count: int
    stats: Mapping[str, float]
    certificates: Mapping[str, Certificate]
    trajectory: np.ndarray | None
    iteration_index: np.ndarray | None
    controller_state: ControllerState
    manifest: Any
    # How the starting state was obtained: "capacity", "custom", "continued"
    # (a prior result of an identical model), or "continued_foreign".
    start_convention: str
