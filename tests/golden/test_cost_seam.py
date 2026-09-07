"""Goldens for the cost path: rebuilt costs, kernels from costs, and evolve."""

from __future__ import annotations

import time

import numpy as np
import pytest
from tests.support import WORST_REL, kernel_to_array, load_fixture

import prefcent as pc

FIXTURE_NAMES = ("grid12", "grid16t", "grid20", "pa_small")
D0 = 5000.0


def _relmax(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))


@pytest.mark.golden
def test_graph_costs_matches_stored_od_exactly() -> None:
    meta, z = load_fixture("pa_small")
    n_nodes = int(meta["graph_nodes"])
    sources = z["zone_graph_index"]
    t0 = time.perf_counter()
    rebuilt = pc.kernels.graph_costs(
        z["graph_edges"],
        z["graph_costs"],
        n_nodes,
        sources=sources,
        directed=False,
        n_jobs=1,
    )
    elapsed = time.perf_counter() - t0
    ms_per = elapsed / len(sources) * 1000.0
    WORST_REL["graph_costs_ms_per_source"] = ms_per
    print(f"graph_costs pa_small: {ms_per:.4g} ms/source")
    od = z["od"]
    assert np.array_equal(rebuilt, od)
    off = ~np.eye(od.shape[0], dtype=bool)
    assert int((rebuilt[off] == od[off]).sum()) == int(off.sum())


@pytest.mark.golden
@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_from_costs_matches_stored_kernels(fixture_name: str) -> None:
    meta, z = load_fixture(fixture_name)
    od = z["od"]
    worst = 0.0
    for beta in (1.5, 2.0, 2.5):
        kernel = pc.kernels.from_costs(od, beta=beta, d0=D0, self_interaction=False)
        ref = z[f"K_beta{beta}"]
        got = kernel_to_array(kernel, ref.shape[0])
        rel = _relmax(got, ref)
        worst = max(worst, rel)
        assert rel <= 1e-15, f"{fixture_name} beta={beta}: {rel}"
    WORST_REL[f"from_costs:{fixture_name}"] = worst


@pytest.mark.golden
@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_evolve_on_rebuilt_kernels_matches_fixed_points(fixture_name: str) -> None:
    meta, z = load_fixture(fixture_name)
    landscape = pc.Landscape(z["R"])
    worst = 0.0
    for case in meta["cases"]:
        kernel = pc.kernels.from_costs(
            z["od"],
            beta=float(case["beta"]),
            d0=D0,
            self_interaction=False,
        )
        model = pc.Model(kernel, landscape, gamma=float(case["gamma"]))
        result = model.evolve(max_iter=int(meta["iterations"]))
        ref = z[f"a_{case['name']}"]
        rel = _relmax(result.mass, ref)
        worst = max(worst, rel)
        assert rel <= 1e-14, f"{fixture_name} {case['name']}: {rel}"
    WORST_REL[f"rebuilt:{fixture_name}"] = worst
    print(f"worst rebuilt {fixture_name}: {worst:.6g}")
