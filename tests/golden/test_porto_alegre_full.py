"""The Porto Alegre reference configurations at full scale, from the routing graph.

These need the full-scale inputs — the routing graph, the zone keys, and the study's
reference columns — which are too large to distribute with the repository (see
REPRODUCE.md); the tests skip wherever they are absent. Around 50 GiB of memory, and
roughly four minutes per run on 64 cores.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, eigs

import prefcent as pc

ARCH = Path(os.environ.get("PREFCENT_PA_DATA", "/workdata/pa-full"))
NEEDED = (
    "routing_edges.npz",
    "zone_graph_index.npy",
    "zone_keys.csv",
    "reference_columns.csv",
)
COST_UNITS = "metre-equivalents at 60 km/h (1000 per minute)"
TOL = 2e-14

needs_inputs = pytest.mark.skipif(
    not all((ARCH / f).exists() for f in NEEDED),
    reason="full-scale Porto Alegre inputs are not distributed with the repo",
)


def _inputs() -> tuple[
    np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, list[tuple[str, str]]
]:
    z = np.load(ARCH / "routing_edges.npz")
    edges = np.stack([z["u"], z["v"]], axis=1).astype(np.int64)
    weights = z["cost"].astype(np.float64)
    sources = np.load(ARCH / "zone_graph_index.npy")
    with open(ARCH / "zone_keys.csv") as fh:
        rows = list(csv.DictReader(fh))
    capacity = np.array([float(r["garea"]) for r in rows])
    keys = [(r["node1"], r["node2"]) for r in rows]
    return edges, weights, int(z["n_nodes"]), sources, capacity, keys


def _reference(column: str, keys: list[tuple[str, str]]) -> np.ndarray:
    ref: dict[tuple[str, str], float] = {}
    with open(ARCH / "reference_columns.csv") as fh:
        for r in csv.DictReader(fh):
            ref[(r["node1"], r["node2"])] = float(r[column])
    return np.array([ref[k] for k in keys])


def _kernel(beta: float) -> pc.DenseKernel:
    edges, weights, n_nodes, sources, _, _ = _inputs()
    return pc.kernels.from_graph(
        edges,
        weights,
        n_nodes,
        sources=sources,
        directed=False,
        beta=beta,
        d0=5000.0,
        cost_units=COST_UNITS,
        self_interaction=False,
    )


@pytest.mark.golden
@pytest.mark.slow
@needs_inputs
@pytest.mark.parametrize(
    ("beta", "column"),
    [(2.0, "ca24g10b2k0"), (1.5, "ca35g1b15k0"), (2.5, "ca44g1b25k0")],
)
def test_gamma_one_runs_match_the_reference_columns_at_full_scale(
    beta: float, column: str
) -> None:
    _, _, _, _, capacity, keys = _inputs()
    result = pc.Model(_kernel(beta), pc.Landscape(capacity), gamma=1.0).evolve(
        max_iter=200
    )
    assert result.status is pc.EvolveStatus.FIXED_BUDGET
    assert result.certificates["isolated_zones"].data["count"] == 0
    ref = _reference(column, keys)
    rel = np.abs(result.mass - ref) / ref
    assert float(rel.max()) <= TOL, float(rel.max())


@pytest.mark.golden
@pytest.mark.slow
@needs_inputs
def test_gamma_zero_run_is_the_dominant_eigenvector_at_full_scale() -> None:
    """γ = 0 is eigenvector centrality: the fixed point is the Perron vector of the
    linear map x ↦ R ⊙ Kᵀ(x / K·R), scaled to the total capacity."""
    _, _, _, _, capacity, _ = _inputs()
    kernel = _kernel(2.0)
    landscape = pc.Landscape(capacity)
    result = pc.Model(kernel, landscape, gamma=0.0).evolve(max_iter=200)
    assert (
        result.final_step_norm <= 1e-12
    )  # a linear map converges to machine precision
    potential = kernel.matvec(capacity)

    def mv(x: np.ndarray) -> np.ndarray:
        return np.asarray(
            capacity * kernel.rmatvec(np.asarray(x).reshape(-1) / potential)
        )

    op = LinearOperator((landscape.n, landscape.n), matvec=mv, dtype=np.float64)
    vals, vecs = eigs(op, k=1, which="LM", v0=capacity.copy(), tol=1e-13, maxiter=5000)
    assert abs(float(np.real(vals[0])) - 1.0) <= 1e-12
    v = np.real(vecs[:, 0])
    v = v * np.sign(v.sum()) * (landscape.total_capacity / abs(v.sum()))
    rel = np.abs(result.mass - v) / v
    assert float(rel.max()) <= TOL, float(rel.max())
