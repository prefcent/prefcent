"""gamma = 0 uses the same evolve path and matches the stored case."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import prefcent as pc


@pytest.mark.property
def test_gamma0_uses_the_same_manifest_shape_and_matches_fixture(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    meta, z = grid12
    capacity = z["R"]
    landscape = pc.Landscape(capacity)
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    m0 = pc.Model(kernel, landscape, gamma=0.0)
    m1 = pc.Model(kernel, landscape, gamma=1.0)
    r0 = m0.evolve(max_iter=200)
    r1 = m1.evolve(max_iter=5)
    assert set(r0.manifest.fingerprint.keys()) == set(r1.manifest.fingerprint.keys())
    assert set(r0.manifest.observations.keys()) == set(r1.manifest.observations.keys())
    assert (
        r0.manifest.fingerprint["solver"].keys()
        == r1.manifest.fingerprint["solver"].keys()
    )
    assert r0.manifest.fingerprint["model"]["gamma"] == 0.0
    ref = z["a_gamma0_beta2"]
    rel = float(np.max(np.abs(r0.mass - ref) / np.abs(ref)))
    assert rel <= 1e-14
