"""Conservation of mass along a recorded trajectory."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import prefcent as pc


@pytest.mark.property
def test_mass_is_conserved_each_recorded_step(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    meta, z = grid12
    capacity = z["R"]
    landscape = pc.Landscape(capacity)
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    result = model.evolve(max_iter=10, record=1)
    assert result.trajectory is not None
    total = float(capacity.sum())
    for row in result.trajectory:
        assert abs(float(row.sum()) - total) <= 1e-12 * total
