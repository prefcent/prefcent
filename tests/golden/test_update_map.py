"""Update-map goldens against stored snapshots and fixed points."""

from __future__ import annotations

import numpy as np
import pytest
from tests.support import WORST_REL, load_fixture

import prefcent as pc

FIXTURE_NAMES = ("grid12", "grid16t", "grid20", "pa_small")
REL_TOL = 1e-14


def _relmax(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(got - ref) / np.abs(ref)))


@pytest.mark.golden
@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_update_map_matches_stored_trajectories(fixture_name: str) -> None:
    meta, z = load_fixture(fixture_name)
    capacity = z["R"]
    landscape = pc.Landscape(capacity)
    worst = 0.0
    snapshots: list[int] = list(meta["snapshots"])
    for case in meta["cases"]:
        gamma = float(case["gamma"])
        beta = case["beta"]
        kernel = pc.DenseKernel(
            z[f"K_beta{beta}"],
            is_symmetric=True,
            self_interaction=False,
        )
        model = pc.Model(kernel, landscape, gamma=gamma)
        for k in snapshots:
            result = model.evolve(max_iter=k)
            ref = z[f"a_{case['name']}_step{k}"]
            rel = _relmax(result.mass, ref)
            worst = max(worst, rel)
            assert rel <= REL_TOL, f"{fixture_name} {case['name']} step {k}: {rel}"
            assert result.status is pc.EvolveStatus.FIXED_BUDGET
            assert result.iterations == k
            total = float(capacity.sum())
            assert abs(float(result.mass.sum()) - total) <= 1e-9 * total
        result = model.evolve(max_iter=int(meta["iterations"]))
        ref = z[f"a_{case['name']}"]
        rel = _relmax(result.mass, ref)
        worst = max(worst, rel)
        assert rel <= REL_TOL, f"{fixture_name} {case['name']} fixed point: {rel}"
        assert result.status is pc.EvolveStatus.FIXED_BUDGET
        assert result.iterations == int(meta["iterations"])
        total = float(capacity.sum())
        assert abs(float(result.mass.sum()) - total) <= 1e-9 * total
    WORST_REL[fixture_name] = worst
    print(f"worst relative error {fixture_name}: {worst:.6g}")


@pytest.mark.golden
def test_gamma0_fixed_point_matches_dominant_linear_map_eigenvector() -> None:
    meta, z = load_fixture("pa_small")
    capacity = z["R"]
    kernel_arr = z["K_beta2.0"]
    landscape = pc.Landscape(capacity)
    kernel = pc.DenseKernel(kernel_arr, is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=0.0)
    result = model.evolve(max_iter=int(meta["iterations"]))

    from scipy.sparse.linalg import LinearOperator, eigs

    kr = kernel_arr @ capacity

    def mv(x: np.ndarray) -> np.ndarray:
        return np.asarray(capacity * (kernel_arr.T @ (x / kr)))

    n = capacity.shape[0]
    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    _vals, vecs = eigs(op, k=1, which="LM")
    vec = np.real(vecs[:, 0])
    if float(vec.sum()) < 0.0:
        vec = -vec
    vec = vec * float(capacity.sum()) / float(vec.sum())
    rel = _relmax(result.mass, vec)
    assert rel <= 1e-10, rel
