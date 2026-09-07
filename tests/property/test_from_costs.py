"""from_costs validation rows and provenance."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from tests.support import load_fixture

import prefcent as pc


def _square() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


@pytest.mark.property
def test_from_costs_rejects_nonsquare_nan_negative_and_mixed_paths() -> None:
    with pytest.raises(ValueError):
        pc.kernels.from_costs(np.ones((2, 3)), beta=2.0, d0=1.0)
    bad = _square()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError):
        pc.kernels.from_costs(bad, beta=2.0, d0=1.0)
    bad = _square()
    bad[0, 1] = -1.0
    with pytest.raises(ValueError):
        pc.kernels.from_costs(bad, beta=2.0, d0=1.0)
    with pytest.raises(ValueError):
        pc.kernels.from_costs(_square())
    with pytest.raises(ValueError):
        pc.kernels.from_costs(
            _square(),
            beta=2.0,
            d0=1.0,
            func=lambda c: c,
            func_name="x",
            is_symmetric=True,
        )
    with pytest.raises(ValueError):
        pc.kernels.from_costs(_square(), beta=2.0)
    with pytest.raises(ValueError):
        pc.kernels.from_costs(_square(), beta=0.0, d0=1.0)
    with pytest.raises(ValueError):
        pc.kernels.from_costs(np.ones((2, 2), dtype=np.float32), beta=2.0, d0=1.0)
    inf_c = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    kernel = pc.kernels.from_costs(inf_c, beta=2.0, d0=1.0)
    col = kernel.matvec(np.array([0.0, 1.0], dtype=np.float64))
    assert col[0] == 0.0


@pytest.mark.property
def test_callable_path_requires_func_name_and_is_symmetric() -> None:
    def fn(c: np.ndarray) -> np.ndarray:
        return np.exp(-c)

    with pytest.raises(ValueError):
        pc.kernels.from_costs(_square(), func=fn, is_symmetric=True)
    with pytest.raises(ValueError):
        pc.kernels.from_costs(_square(), func=fn, func_name="exp")
    kernel = pc.kernels.from_costs(
        _square(), func=fn, func_name="exp", is_symmetric=True
    )
    assert kernel.is_symmetric_source == "declared"
    assert kernel.provenance is not None
    assert kernel.provenance["decay"]["self_declared"] is True


@pytest.mark.property
def test_explicit_is_symmetric_is_declared_with_evidence() -> None:
    meta, z = load_fixture("pa_small")
    kernel = pc.kernels.from_costs(z["od"], beta=2.0, d0=5000.0, is_symmetric=True)
    assert kernel.is_symmetric is True
    assert kernel.is_symmetric_source == "declared"
    assert kernel.symmetry_check is not None
    assert kernel.symmetry_check["exact"] is False
    assert kernel.symmetry_check["max_abs_asym"] > 0.0


@pytest.mark.property
def test_callable_decay_is_not_reproducible_by_fingerprint() -> None:
    meta, z = load_fixture("grid12")
    landscape = pc.Landscape(z["R"])

    def fn(c: np.ndarray) -> np.ndarray:
        return (c + 5000.0) ** (-2.0)

    kernel = pc.kernels.from_costs(
        z["od"], func=fn, func_name="power_copy", is_symmetric=True
    )
    with patch(
        "prefcent._model.source_identity",
        return_value={"kind": "git", "commit": "abc123", "dirty": False},
    ):
        result = pc.Model(kernel, landscape, gamma=1.0).evolve(max_iter=1)
    assert result.manifest.fingerprint["reproducible"] is False
    assert "self_declared_decay" in result.manifest.fingerprint["reproducible_reasons"]


@pytest.mark.property
def test_from_graph_named_path_is_reproducible_on_a_clean_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    meta, z = load_fixture("pa_small")
    landscape = pc.Landscape(z["R"])
    kernel = pc.kernels.from_graph(
        z["graph_edges"],
        z["graph_costs"],
        int(meta["graph_nodes"]),
        sources=z["zone_graph_index"],
        beta=2.0,
        d0=5000.0,
        n_jobs=1,
    )
    assert kernel.is_symmetric is True
    assert kernel.is_symmetric_source == "derived"
    assert kernel.symmetry_check is not None
    assert kernel.symmetry_check["exact"] is False
    with patch(
        "prefcent._model.source_identity",
        return_value={"kind": "git", "commit": "abc123", "dirty": False},
    ):
        result = pc.Model(kernel, landscape, gamma=1.0).evolve(max_iter=1)
    assert result.manifest.fingerprint["reproducible"] is True
    assert result.manifest.fingerprint["reproducible_reasons"] == ()
    assert result.manifest.fingerprint["kernel"]["provenance"]["builder"] == (
        "from_graph"
    )


@pytest.mark.property
def test_exponential_named_path_matches_expression_and_is_recorded() -> None:
    c = np.array(
        [[0.0, 300.0, 900.0], [300.0, 0.0, 600.0], [900.0, 600.0, 0.0]],
        dtype=np.float64,
    )
    kernel = pc.kernels.from_costs(c, decay_length=250.0, cost_units="m")
    expected = np.exp(-c / 250.0)
    np.fill_diagonal(expected, 0.0)
    ones = np.ones(3, dtype=np.float64)
    assert np.array_equal(kernel.matvec(ones), expected @ ones)
    assert kernel.is_symmetric and kernel.is_symmetric_source == "derived"
    assert kernel.provenance is not None
    assert kernel.provenance["decay"] == {
        "form": "exponential",
        "decay_length": 250.0,
        "cost_units": "m",
    }
    inf_c = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    inf_k = pc.kernels.from_costs(inf_c, decay_length=250.0)
    assert np.array_equal(
        inf_k.matvec(np.ones(2, dtype=np.float64)), np.zeros(2, dtype=np.float64)
    )


@pytest.mark.property
def test_exponential_path_is_exclusive_and_validated() -> None:
    c = _square()
    with pytest.raises(ValueError):
        pc.kernels.from_costs(c, beta=2.0, d0=1.0, decay_length=250.0)
    with pytest.raises(ValueError):
        pc.kernels.from_costs(
            c, decay_length=250.0, func=lambda x: x, func_name="x", is_symmetric=True
        )
    with pytest.raises(ValueError):
        pc.kernels.from_costs(c, decay_length=0.0)
    with pytest.raises(ValueError):
        pc.kernels.from_costs(c, decay_length=np.inf)
    with pytest.raises(ValueError):
        pc.kernels.from_costs(c, decay_length=250.0, func_name="x")
    with pytest.raises(ValueError):
        pc.kernels.from_costs(c, decay_length=250.0, func_params={"a": 1.0})
