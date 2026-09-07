"""graph_costs edge cases: zero weights, duplicates, self-loops, unreachable."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from tests.support import load_fixture

import prefcent as pc


def _line_graph() -> tuple[np.ndarray, np.ndarray, int]:
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    return edges, weights, 3


@pytest.mark.property
def test_zero_weight_edge_keeps_the_graph_connected() -> None:
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    weights = np.array([0.0, 1.0], dtype=np.float64)
    costs = pc.kernels.graph_costs(
        edges, weights, 3, sources=np.array([0, 2]), directed=False, n_jobs=1
    )
    assert np.isfinite(costs[0, 1])
    assert costs[0, 1] == 1.0


@pytest.mark.property
def test_duplicate_edges_take_the_minimum_weight() -> None:
    edges = np.array([[0, 1], [0, 1], [1, 2]], dtype=np.int64)
    weights = np.array([5.0, 2.0, 1.0], dtype=np.float64)
    costs = pc.kernels.graph_costs(
        edges, weights, 3, sources=np.array([0]), targets=np.array([1]), n_jobs=1
    )
    assert costs[0, 0] == 2.0


@pytest.mark.property
def test_undirected_reverse_pair_is_a_duplicate() -> None:
    # On an undirected graph (0, 1) and (1, 0) are the same edge, so the pair
    # below is a duplicate and the smaller weight wins.
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    weights = np.array([5.0, 2.0], dtype=np.float64)
    costs = pc.kernels.graph_costs(
        edges,
        weights,
        2,
        sources=np.array([0]),
        targets=np.array([1]),
        directed=False,
        n_jobs=1,
    )
    assert costs[0, 0] == 2.0


@pytest.mark.property
def test_graph_costs_rejects_negative_or_nan_weights() -> None:
    edges = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError):
        pc.kernels.graph_costs(
            edges,
            np.array([-1.0]),
            2,
            sources=np.array([0]),
            n_jobs=1,
        )
    with pytest.raises(ValueError):
        pc.kernels.graph_costs(
            edges,
            np.array([np.nan]),
            2,
            sources=np.array([0]),
            n_jobs=1,
        )


@pytest.mark.property
def test_self_loops_are_ignored() -> None:
    edges = np.array([[0, 0], [0, 1]], dtype=np.int64)
    weights = np.array([99.0, 3.0], dtype=np.float64)
    costs = pc.kernels.graph_costs(
        edges, weights, 2, sources=np.array([0]), targets=np.array([1]), n_jobs=1
    )
    assert costs[0, 0] == 3.0


@pytest.mark.property
def test_unreachable_target_is_inf_then_zero_kernel_weight() -> None:
    edges = np.array([[0, 1]], dtype=np.int64)
    weights = np.array([1.0], dtype=np.float64)
    costs = pc.kernels.graph_costs(
        edges,
        weights,
        3,
        sources=np.array([0]),
        targets=np.array([2]),
        directed=True,
        n_jobs=1,
    )
    assert np.isinf(costs[0, 0])
    c = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    k = pc.kernels.from_costs(c, beta=2.0, d0=1.0)
    col1 = k.matvec(np.array([0.0, 1.0], dtype=np.float64))
    col0 = k.matvec(np.array([1.0, 0.0], dtype=np.float64))
    assert col1[0] == 0.0
    assert col0[1] == 0.0


@pytest.mark.property
def test_n_jobs_one_and_two_agree_bit_for_bit() -> None:
    meta, z = load_fixture("pa_small")
    n_nodes = int(meta["graph_nodes"])
    kwargs: dict[str, Any] = dict(
        edges=z["graph_edges"],
        weights=z["graph_costs"],
        n_nodes=n_nodes,
        sources=z["zone_graph_index"],
        directed=False,
    )
    a = pc.kernels.graph_costs(**kwargs, n_jobs=1)
    b = pc.kernels.graph_costs(**kwargs, n_jobs=2)
    assert np.array_equal(a, b)


@pytest.mark.property
def test_rectangular_targets_and_from_graph_rejects_targets() -> None:
    edges, weights, n_nodes = _line_graph()
    costs = pc.kernels.graph_costs(
        edges,
        weights,
        n_nodes,
        sources=np.array([0, 2]),
        targets=np.array([1]),
        n_jobs=1,
    )
    assert costs.shape == (2, 1)
    with pytest.raises(TypeError):
        pc.kernels.from_graph(
            edges,
            weights,
            n_nodes,
            sources=np.array([0, 1, 2]),
            targets=np.array([0, 1]),
            beta=2.0,
            d0=1.0,
        )


@pytest.mark.property
def test_graph_sha256_is_invariant_to_edge_list_permutation() -> None:
    meta, z = load_fixture("pa_small")
    n_nodes = int(meta["graph_nodes"])
    src = z["zone_graph_index"]
    k1 = pc.kernels.from_graph(
        z["graph_edges"],
        z["graph_costs"],
        n_nodes,
        sources=src,
        beta=2.0,
        d0=5000.0,
        n_jobs=1,
    )
    perm = np.random.default_rng(0).permutation(z["graph_edges"].shape[0])
    k2 = pc.kernels.from_graph(
        z["graph_edges"][perm],
        z["graph_costs"][perm],
        n_nodes,
        sources=src,
        beta=2.0,
        d0=5000.0,
        n_jobs=1,
    )
    assert k1.provenance is not None and k2.provenance is not None
    assert k1.provenance["graph_sha256"] == k2.provenance["graph_sha256"]


@pytest.mark.property
def test_n_jobs_none_reads_cgroup_quota(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000\n")
    with patch("prefcent._builders.CPU_MAX_PATH", cpu_max):
        from prefcent._builders import resolve_n_jobs

        assert resolve_n_jobs(None) == 2


@pytest.mark.property
@pytest.mark.parametrize("quota", [None, "max 100000\n"])
def test_n_jobs_none_without_quota_uses_the_affinity_count(
    tmp_path: Path, quota: str | None
) -> None:
    cpu_max = tmp_path / "cpu.max"
    if quota is not None:
        cpu_max.write_text(quota)
    with (
        patch("prefcent._builders.CPU_MAX_PATH", cpu_max),
        patch(
            "prefcent._builders.os.sched_getaffinity", return_value={2, 3}, create=True
        ),
    ):
        from prefcent._builders import resolve_n_jobs

        assert resolve_n_jobs(None) == 2


@pytest.mark.property
@pytest.mark.parametrize("quota", [None, "max 100000\n"])
@pytest.mark.parametrize("cpu_count, expected", [(10, 10), (None, 1)])
def test_n_jobs_none_without_affinity_uses_cpu_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quota: str | None,
    cpu_count: int | None,
    expected: int,
) -> None:
    import os

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    cpu_max = tmp_path / "cpu.max"
    if quota is not None:
        cpu_max.write_text(quota)
    with patch("prefcent._builders.CPU_MAX_PATH", cpu_max):
        from prefcent._builders import resolve_n_jobs

        assert resolve_n_jobs(None) == expected


@pytest.mark.property
@pytest.mark.parametrize("builder", ["from_graph", "from_costs"])
def test_one_way_reachability_evolves_with_a_serializable_manifest(
    builder: str,
) -> None:
    edges = np.array([[0, 1], [1, 2], [2, 1]])
    weights = np.array([1.0, 2.0, 1.0])
    costs = np.array([[0.0, 1.0, 3.0], [np.inf, 0.0, 2.0], [np.inf, 1.0, 0.0]])
    if builder == "from_graph":
        kernel = pc.kernels.from_graph(
            edges,
            weights,
            3,
            sources=np.arange(3),
            directed=True,
            n_jobs=1,
            beta=2.0,
            d0=1.0,
        )
    else:
        kernel = pc.kernels.from_costs(costs, beta=2.0, d0=1.0)
    expected_kernel = pc.DenseKernel(
        np.array([[0.0, 1 / 4, 1 / 16], [0.0, 0.0, 1 / 9], [0.0, 1 / 4, 0.0]]),
        is_symmetric=False,
        self_interaction=False,
    )
    landscape = pc.Landscape(np.ones(3))
    result = pc.Model(kernel, landscape, gamma=1.0).evolve(max_iter=2)
    expected = pc.Model(expected_kernel, landscape, gamma=1.0).evolve(max_iter=2)
    np.testing.assert_allclose(result.mass, expected.mass, rtol=1e-15, atol=0.0)
    assert result.status is pc.EvolveStatus.FIXED_BUDGET
    assert result.active.all()
    restored = pc.RunManifest.from_json(result.manifest.to_json())
    assert restored.fingerprint_sha256 == result.manifest.fingerprint_sha256
    assert restored.fingerprint["kernel"]["symmetry_check"] == {
        "exact": False,
        "max_abs_asym": None,
        "asymmetric_reachability": True,
        "max_abs_finite_asym": 1.0,
    }


@pytest.mark.property
def test_kernels_module_exports_exactly_the_four_builders() -> None:
    assert pc.kernels.__all__ == [
        "from_costs",
        "graph_costs",
        "from_graph",
        "adjointness_probe",
    ]
    assert hasattr(pc, "OperatorKernel")
    assert hasattr(pc, "CirculantKernel")
