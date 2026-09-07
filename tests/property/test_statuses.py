"""Evolve statuses, budgets, counts, trajectory, and StepInfo."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import prefcent as pc


def _model(grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile]) -> pc.Model:
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    return pc.Model(kernel, landscape, gamma=1.0)


@pytest.mark.property
def test_all_four_statuses_are_reachable_with_final_step_norm(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    model = _model(grid12)
    fixed = model.evolve(max_iter=2)
    assert fixed.status is pc.EvolveStatus.FIXED_BUDGET
    assert np.isfinite(fixed.final_step_norm)

    conv = model.evolve(max_iter=5, tol=1e9)
    assert conv.status is pc.EvolveStatus.CONVERGED
    assert np.isfinite(conv.final_step_norm)

    exhausted = model.evolve(max_iter=1, tol=1e-30)
    assert exhausted.status is pc.EvolveStatus.BUDGET_EXHAUSTED
    assert np.isfinite(exhausted.final_step_norm)

    cancelled = model.evolve(max_iter=5, should_stop=lambda: True)
    assert cancelled.status is pc.EvolveStatus.CANCELLED
    assert cancelled.iterations >= 1
    assert np.isfinite(cancelled.final_step_norm)


@pytest.mark.property
def test_max_iter_zero_and_tol_zero_raise(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    model = _model(grid12)
    with pytest.raises(ValueError):
        model.evolve(max_iter=0)
    with pytest.raises(ValueError):
        model.evolve(max_iter=1, tol=0.0)


@pytest.mark.property
def test_operator_counts_match_iterations(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    result = _model(grid12).evolve(max_iter=7)
    assert result.matvec_count == result.iterations + 2
    assert result.rmatvec_count == result.iterations + 2


@pytest.mark.property
def test_trajectory_contains_start_and_final(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    model = _model(grid12)
    result = model.evolve(max_iter=5, record=2)
    assert result.trajectory is not None
    assert result.iteration_index is not None
    assert result.iteration_index[0] == 0
    assert np.array_equal(result.trajectory[0], z["R"])
    assert result.iteration_index[-1] == result.iterations
    assert np.array_equal(result.trajectory[-1], result.mass)


@pytest.mark.property
def test_stepinfo_stats_so_far_cannot_be_mutated(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    seen: list[pc.StepInfo] = []

    def cb(info: pc.StepInfo) -> None:
        seen.append(info)
        with pytest.raises((TypeError, AttributeError)):
            info.stats_so_far["trajectory_min"] = 0.0  # type: ignore[index]

    _model(grid12).evolve(max_iter=2, callback=cb)
    assert seen
    with pytest.raises((TypeError, AttributeError)):
        seen[0].stats_so_far["final_min"] = 0.0  # type: ignore[index]
