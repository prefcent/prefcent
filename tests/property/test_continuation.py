"""Warm continuation: same-model bit identity and foreign classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import prefcent as pc


@pytest.mark.property
def test_continued_run_matches_single_run_bit_for_bit(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    r5 = model.evolve(max_iter=5)
    r_cont = model.evolve(start=r5, max_iter=3)
    r8 = model.evolve(max_iter=8)
    assert np.array_equal(r_cont.mass, r8.mass)
    assert r_cont.start_convention == "continued"


@pytest.mark.property
def test_changed_kernel_capacity_or_omega_is_continued_foreign(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    k2 = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(k2, landscape, gamma=1.0)
    r5 = model.evolve(max_iter=5)

    k25 = pc.DenseKernel(z["K_beta2.5"], is_symmetric=True, self_interaction=False)
    other_k = pc.Model(k25, landscape, gamma=1.0)
    r_k = other_k.evolve(start=r5, max_iter=1)
    assert r_k.start_convention == "continued_foreign"

    landscape2 = pc.Landscape(z["R"] * 2.0)
    other_r = pc.Model(k2, landscape2, gamma=1.0)
    r_r = other_r.evolve(start=r5, max_iter=1)
    assert r_r.start_convention == "continued_foreign"

    r_w = model.evolve(start=r5, max_iter=1, omega=0.5)
    assert r_w.start_convention == "continued_foreign"


@pytest.mark.property
def test_custom_start_projection_is_recorded(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    capacity = z["R"]
    landscape = pc.Landscape(capacity)
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    supplied = capacity * 2.0
    result = model.evolve(start=supplied, max_iter=1)
    assert result.start_convention == "custom"
    proj = result.manifest.fingerprint["solver"]["start_projection"]
    assert proj["isolated_mass_dropped"] == 0.0
    assert proj["scale"] == pytest.approx(0.5)
    assert result.manifest.fingerprint["solver"]["start_sha256"] is not None


@pytest.mark.property
@pytest.mark.parametrize("name", ["grid12", "grid16t", "grid20", "pa_small"])
def test_continuation_is_accepted_and_bit_exact_for_every_case_and_budget(
    name: str,
) -> None:
    """A continued start is taken verbatim; the total is within rounding of the
    capacity, and is never re-checked to the bit."""
    from tests.support import load_fixture

    meta, z = load_fixture(name)
    landscape = pc.Landscape(z["R"])
    for case in meta["cases"]:
        kernel = pc.DenseKernel(
            z[f"K_beta{case['beta']}"], is_symmetric=True, self_interaction=False
        )
        model = pc.Model(kernel, landscape, gamma=case["gamma"])
        for k in (1, 2, 5, 20):
            r_k = model.evolve(max_iter=k)
            r_cont = model.evolve(start=r_k, max_iter=3)
            r_all = model.evolve(max_iter=k + 3)
            assert r_cont.start_convention == "continued"
            assert np.array_equal(r_cont.mass, r_all.mass)
