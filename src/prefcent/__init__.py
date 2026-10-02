"""prefcent — preferential centrality solver."""

from __future__ import annotations

from prefcent import kernels
from prefcent._circulant import CirculantKernel
from prefcent._closure import Closure, Identity
from prefcent._density_penalty import DensityPenaltyV1
from prefcent._errors import (
    ClosureDomainBreach,
    KernelError,
    ModelDomainError,
    NumericalBreach,
    PrefcentError,
)
from prefcent._kernels import DenseKernel, Kernel
from prefcent._landscape import Landscape
from prefcent._manifest import RunManifest
from prefcent._model import Model
from prefcent._operator import OperatorKernel
from prefcent._result import (
    Certificate,
    EvolveResult,
    EvolveStatus,
    StepInfo,
)

__version__ = "0.1.1"

__all__ = [
    "Landscape",
    "Model",
    "Identity",
    "DensityPenaltyV1",
    "Closure",
    "Kernel",
    "DenseKernel",
    "OperatorKernel",
    "CirculantKernel",
    "kernels",
    "EvolveResult",
    "EvolveStatus",
    "StepInfo",
    "Certificate",
    "RunManifest",
    "PrefcentError",
    "NumericalBreach",
    "ClosureDomainBreach",
    "KernelError",
    "ModelDomainError",
]
