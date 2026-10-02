"""The density penalty closure: conservation, fail-stop, margin, invariances."""

from __future__ import annotations

import numpy as np
import pytest

import prefcent as pc

N = 64
GAMMA = 0.8


def _kernel_and_start() -> tuple[pc.CirculantKernel, np.ndarray]:
    k = pc.CirculantKernel.ring(N, 1000.0, 5000.0, 2.0, self_interaction=True)
    rng = np.random.default_rng(0)
    a = 1.0 + 1e-3 * rng.standard_normal(N)
    return k, a * (N / a.sum())


@pytest.mark.property
def test_kappa_zero_is_exactly_identity() -> None:
    k, a0 = _kernel_and_start()
    land = pc.Landscape(np.ones(N))
    r0 = pc.Model(k, land, gamma=GAMMA).evolve(start=a0, max_iter=20)
    r1 = pc.Model(k, land, gamma=GAMMA, closure=pc.DensityPenaltyV1(0.0)).evolve(
        start=a0, max_iter=20
    )
    assert np.array_equal(r0.mass, r1.mass)


@pytest.mark.property
def test_runs_are_reproducible_with_a_builtin_identity() -> None:
    k, a0 = _kernel_and_start()
    r = pc.Model(
        k, pc.Landscape(np.ones(N)), gamma=GAMMA, closure=pc.DensityPenaltyV1(0.02)
    ).evolve(start=a0, max_iter=10)
    ident = r.manifest.fingerprint["model"]["closure_identity"]
    assert ident["kind"] == "builtin" and ident["name"] == "density_penalty_v1"
    assert r.manifest.fingerprint["model"]["closure_params"] == {
        "kappa": 0.02,
        "rho": 2.0,
    }
    assert (
        "closure_identity_unverified"
        not in (r.manifest.fingerprint["reproducible_reasons"])
    )


@pytest.mark.property
def test_mass_is_conserved_with_the_penalty_active() -> None:
    k, a0 = _kernel_and_start()
    land = pc.Landscape(np.ones(N))
    r = pc.Model(k, land, gamma=GAMMA, closure=pc.DensityPenaltyV1(0.02)).evolve(
        start=a0, max_iter=50, record=1
    )
    assert r.trajectory is not None
    sums = r.trajectory.sum(axis=1)
    assert np.max(np.abs(sums - land.total_capacity)) <= 1e-9 * land.total_capacity


@pytest.mark.property
def test_continued_run_matches_single_run_bit_for_bit() -> None:
    k, a0 = _kernel_and_start()
    model = pc.Model(
        k,
        pc.Landscape(np.ones(N)),
        gamma=GAMMA,
        closure=pc.DensityPenaltyV1(0.02),
    )
    first = model.evolve(start=a0, max_iter=7)
    continued = model.evolve(start=first, max_iter=3)
    single = model.evolve(start=a0, max_iter=10)
    assert continued.start_convention == "continued"
    assert np.array_equal(continued.mass, single.mass)


@pytest.mark.property
def test_breach_fail_stops_with_step_and_state() -> None:
    k, a0 = _kernel_and_start()
    with pytest.raises(pc.ClosureDomainBreach) as ei:
        pc.Model(
            k, pc.Landscape(np.ones(N)), gamma=GAMMA, closure=pc.DensityPenaltyV1(50.0)
        ).evolve(start=a0, max_iter=50)
    assert ei.value.step is not None and ei.value.step >= 1
    assert ei.value.state is not None and ei.value.state.shape == (N,)


@pytest.mark.property
def test_margin_certificate_separates_passing_from_breaching_kappa() -> None:
    k, _ = _kernel_and_start()
    rng = np.random.default_rng(3)
    land = pc.Landscape(1.0 + 0.5 * rng.random(N))
    r = pc.Model(k, land, gamma=GAMMA, closure=pc.DensityPenaltyV1(0.02)).evolve(
        max_iter=5000, tol=1e-12
    )
    cert = r.certificates["closure:density_penalty_v1"]
    k_star = cert.data["kappa_at_zero"]
    assert cert.passed and isinstance(k_star, float) and 0.02 < k_star < 1e3
    pc.Model(k, land, gamma=GAMMA, closure=pc.DensityPenaltyV1(0.9 * k_star)).evolve(
        start=r.mass, max_iter=1
    )
    with pytest.raises(pc.ClosureDomainBreach) as ei:
        pc.Model(
            k, land, gamma=GAMMA, closure=pc.DensityPenaltyV1(1.1 * k_star)
        ).evolve(start=r.mass, max_iter=1)
    assert ei.value.step == 1


@pytest.mark.property
def test_dynamics_are_invariant_under_zone_subdivision() -> None:
    """Splitting every zone into two half-capacity zones at the same location leaves
    the per-unit dynamics unchanged: the penalty is capacity-weighted."""
    k, a0 = _kernel_and_start()
    dense = np.empty((N, N))
    for i in range(N):
        e = np.zeros(N)
        e[i] = 1.0
        dense[:, i] = k.matvec(e)
    kd = pc.DenseKernel(
        np.kron(dense, np.ones((2, 2))), is_symmetric=True, self_interaction=True
    )
    c = pc.DensityPenaltyV1(0.02)
    r1 = pc.Model(k, pc.Landscape(np.ones(N)), gamma=GAMMA, closure=c).evolve(
        start=a0, max_iter=20
    )
    r2 = pc.Model(kd, pc.Landscape(np.full(2 * N, 0.5)), gamma=GAMMA, closure=c).evolve(
        start=np.repeat(a0, 2) / 2, max_iter=20
    )
    rel = np.abs(np.repeat(r1.mass, 2) / 2 - r2.mass) / r2.mass
    assert float(rel.max()) <= 1e-12


@pytest.mark.property
def test_parameter_validation() -> None:
    for bad in (float("nan"), -0.1, float("inf")):
        with pytest.raises(ValueError):
            pc.DensityPenaltyV1(bad)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            pc.DensityPenaltyV1(0.1, rho=bad)


@pytest.mark.property
@pytest.mark.parametrize("isolated", [False, True])
def test_uniform_margin_survives_manifest_round_trip(isolated: bool) -> None:
    diagonal = np.array([1.0, 1.0, 0.0]) if isolated else np.ones(2)
    capacity = np.array([1.0, 1.0, 10.0]) if isolated else np.ones(2)
    kernel = pc.DenseKernel(np.diag(diagonal), is_symmetric=True, self_interaction=True)
    result = pc.Model(
        kernel, pc.Landscape(capacity), gamma=0.0, closure=pc.DensityPenaltyV1(0.01)
    ).evolve(max_iter=1)
    cert = result.certificates["closure:density_penalty_v1"]
    assert cert.passed
    assert cert.data["min_closed_mass"] == 1.0
    assert cert.data["kappa_at_zero"] == "unbounded"
    restored = pc.RunManifest.from_json(result.manifest.to_json())
    assert restored.to_dict() == result.manifest.to_dict()


@pytest.mark.property
def test_zero_penalty_survives_overflow_and_serializes_unavailable_margin() -> None:
    kernel = pc.DenseKernel(np.eye(2), is_symmetric=True, self_interaction=True)
    landscape = pc.Landscape(np.ones(2))
    start = np.array([1.5, 0.5])
    identity = pc.Model(kernel, landscape, gamma=0.0).evolve(
        start=start, max_iter=3, record=1
    )
    with np.errstate(all="raise"):
        result = pc.Model(
            kernel, landscape, gamma=0.0, closure=pc.DensityPenaltyV1(0.0, rho=2048.0)
        ).evolve(start=start, max_iter=3, record=1)
    assert result.trajectory is not None and identity.trajectory is not None
    assert np.array_equal(result.trajectory, identity.trajectory)
    cert = result.certificates["closure:density_penalty_v1"]
    assert cert.passed
    assert cert.data["min_closed_mass"] == 0.5
    assert cert.data["kappa_at_zero"] == "unavailable"
    restored = pc.RunManifest.from_json(result.manifest.to_json())
    assert restored.to_dict() == result.manifest.to_dict()


@pytest.mark.property
def test_zero_penalty_still_reports_a_computable_margin() -> None:
    mass = np.array([1.5, 0.5])
    cert = pc.DensityPenaltyV1(0.0, rho=1.0).domain_check(
        mass, mass, pc.Landscape(np.ones(2)), np.ones(2, dtype=bool)
    )
    assert cert.passed
    assert cert.data["kappa_at_zero"] == 3.0


@pytest.mark.property
def test_penalty_overflow_raises_numerical_breach_with_step_and_state() -> None:
    kernel = pc.DenseKernel(np.eye(2), is_symmetric=True, self_interaction=True)
    start = np.array([1.5, 0.5])
    with np.errstate(all="raise"), pytest.raises(pc.NumericalBreach) as exc:
        pc.Model(
            kernel,
            pc.Landscape(np.ones(2)),
            gamma=0.0,
            closure=pc.DensityPenaltyV1(0.01, rho=2048.0),
        ).evolve(start=start, max_iter=1)
    assert type(exc.value) is pc.NumericalBreach
    assert exc.value.step == 1
    assert exc.value.state is not None
    assert np.array_equal(exc.value.state, start)


@pytest.mark.property
def test_final_certificate_overflow_raises_with_final_state() -> None:
    kernel = pc.DenseKernel(
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        is_symmetric=False,
        self_interaction=True,
    )
    with np.errstate(all="raise"), pytest.raises(pc.NumericalBreach) as exc:
        pc.Model(
            kernel,
            pc.Landscape(np.ones(2)),
            gamma=0.0,
            closure=pc.DensityPenaltyV1(0.01, rho=1024.0),
        ).evolve(max_iter=1)
    assert type(exc.value) is pc.NumericalBreach
    assert exc.value.step == 0
    assert exc.value.state is not None
    assert np.array_equal(exc.value.state, np.array([2.0, 0.0]))


@pytest.mark.property
@pytest.mark.parametrize("method", ["apply", "domain_check"])
def test_closed_update_overflow_is_a_numerical_breach(method: str) -> None:
    mass = np.array([2.0, 0.0])
    closure = pc.DensityPenaltyV1(1e308)
    with np.errstate(all="raise"), pytest.raises(pc.NumericalBreach) as exc:
        getattr(closure, method)(
            mass, mass, pc.Landscape(np.ones(2)), np.ones(2, dtype=bool)
        )
    assert type(exc.value) is pc.NumericalBreach
    assert exc.value.state is not None
    assert np.array_equal(exc.value.state, mass)


@pytest.mark.property
@pytest.mark.parametrize("kappa", [0.0, 0.01])
def test_margin_overflow_is_distinct_from_an_unbounded_margin(kappa: float) -> None:
    mass = np.array([1.5, 0.5])
    inflow = np.array([1e308, 0.0])
    closure = pc.DensityPenaltyV1(kappa, rho=1.0)
    landscape = pc.Landscape(np.ones(2))
    active = np.ones(2, dtype=bool)
    with np.errstate(all="raise"):
        if kappa == 0.0:
            cert = closure.domain_check(mass, inflow, landscape, active)
            assert cert.passed
            assert cert.data["kappa_at_zero"] == "unavailable"
        else:
            with pytest.raises(pc.NumericalBreach):
                closure.domain_check(mass, inflow, landscape, active)


@pytest.mark.property
def test_finite_infeasibility_returns_a_failed_certificate() -> None:
    mass = np.array([1.5, 0.5])
    cert = pc.DensityPenaltyV1(4.0, rho=1.0).domain_check(
        mass, mass, pc.Landscape(np.ones(2)), np.ones(2, dtype=bool)
    )
    assert not cert.passed
    assert cert.data["min_closed_mass"] == -0.5
    assert cert.data["kappa_at_zero"] == 3.0
