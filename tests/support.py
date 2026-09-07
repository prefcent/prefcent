"""Shared fixture loading for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"

WORST_REL: dict[str, float] = {}


def load_fixture(name: str) -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
    meta = json.loads((FIXTURES / f"{name}.json").read_text())
    data = np.load(FIXTURES / f"{name}.npz")
    return meta, data


def kernel_to_array(kernel: Any, n: int) -> np.ndarray:
    """Recover a dense matrix from ``matvec`` of the standard basis."""
    eye = np.eye(n, dtype=np.float64)
    return np.column_stack([kernel.matvec(eye[:, j]) for j in range(n)])
