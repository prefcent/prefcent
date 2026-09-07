"""Identity, immutability, and no-coercion contracts of the public API."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

import prefcent as pc


def _tiny() -> tuple[pc.DenseKernel, pc.Landscape]:
    k = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.0]])
    return (
        pc.DenseKernel(k, is_symmetric=True, self_interaction=False),
        pc.Landscape(np.array([1.0, 2.0, 3.0])),
    )


class _NamedClosure:
    """A consumer closure; two different implementations share name and params."""

    name = "scaled"
    params: dict[str, float] = {"factor": 1.0}
    identity: dict[str, Any] | None = None

    def __init__(self, shift: float, identity: dict[str, Any] | None = None) -> None:
        self._shift = shift
        self.identity = identity

    def apply(
        self, mass: np.ndarray, inflow: np.ndarray, landscape: Any, active: np.ndarray
    ) -> np.ndarray:
        return np.asarray(inflow + self._shift)

    def domain_check(
        self, mass: np.ndarray, inflow: np.ndarray, landscape: Any, active: np.ndarray
    ) -> Any:
        return pc.Certificate(name=self.name, passed=True, data={})


@pytest.mark.property
def test_closures_with_same_name_but_no_identity_are_not_reproducible() -> None:
    k, land = _tiny()
    r1 = pc.Model(k, land, gamma=1.0, closure=_NamedClosure(0.0)).evolve(max_iter=2)
    r2 = pc.Model(k, land, gamma=1.0, closure=_NamedClosure(0.1)).evolve(max_iter=2)
    assert r1.manifest.fingerprint["reproducible"] is False
    assert (
        "closure_identity_unverified" in r1.manifest.fingerprint["reproducible_reasons"]
    )
    assert not np.array_equal(r1.mass, r2.mass)


@pytest.mark.property
def test_distinct_consumer_closure_identities_give_distinct_fingerprints() -> None:
    k, land = _tiny()
    a = _NamedClosure(0.0, {"kind": "consumer", "value": "impl-A", "trusted": True})
    b = _NamedClosure(0.1, {"kind": "consumer", "value": "impl-B", "trusted": True})
    ra = pc.Model(k, land, gamma=1.0, closure=a).evolve(max_iter=2)
    rb = pc.Model(k, land, gamma=1.0, closure=b).evolve(max_iter=2)
    assert ra.manifest.fingerprint_sha256 != rb.manifest.fingerprint_sha256
    assert ra.manifest.fingerprint["model"]["closure_identity"]["value"] == "impl-A"


@pytest.mark.property
def test_builtin_identity_closure_is_recorded_as_builtin() -> None:
    k, land = _tiny()
    r = pc.Model(k, land, gamma=1.0).evolve(max_iter=1)
    assert r.manifest.fingerprint["model"]["closure_identity"]["kind"] == "builtin"


@pytest.mark.property
def test_closure_cannot_mutate_the_active_mask() -> None:
    class _Mutating(_NamedClosure):
        def apply(
            self,
            mass: np.ndarray,
            inflow: np.ndarray,
            landscape: Any,
            active: np.ndarray,
        ) -> np.ndarray:
            active[0] = False
            return np.asarray(inflow)

    k, land = _tiny()
    with pytest.raises(ValueError, match="read-only"):
        pc.Model(k, land, gamma=1.0, closure=_Mutating(0.0)).evolve(max_iter=1)


@pytest.mark.property
def test_manifest_and_certificates_are_deep_frozen() -> None:
    k, land = _tiny()
    r = pc.Model(k, land, gamma=1.0).evolve(max_iter=1)
    with pytest.raises(TypeError):
        r.manifest.fingerprint["model"]["gamma"] = 99
    with pytest.raises(TypeError):
        r.certificates["isolated_zones"].data["count"] = 999  # type: ignore[index]


@pytest.mark.property
def test_from_json_verifies_the_fingerprint_hash() -> None:
    k, land = _tiny()
    r = pc.Model(k, land, gamma=1.0).evolve(max_iter=1)
    text = r.manifest.to_json()
    assert (
        pc.RunManifest.from_json(text).fingerprint_sha256
        == r.manifest.fingerprint_sha256
    )
    tampered = text.replace('"gamma":1.0', '"gamma":2.0')
    assert tampered != text
    with pytest.raises(ValueError, match="fingerprint_sha256"):
        pc.RunManifest.from_json(tampered)


@pytest.mark.property
def test_from_graph_callable_decay_requires_explicit_symmetry() -> None:
    edges = np.array([[0, 1], [1, 2]])
    w = np.array([1.0, 2.0])
    src = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="is_symmetric"):
        pc.kernels.from_graph(
            edges,
            w,
            3,
            sources=src,
            func=lambda c: c + 1.0,
            func_name="shift",
            n_jobs=1,
        )
    with pytest.raises(TypeError, match="coercion"):
        pc.kernels.from_graph(
            edges,
            w,
            3,
            sources=src,
            beta=2.0,
            d0=1.0,
            is_symmetric="false",
            n_jobs=1,
        )


@pytest.mark.property
def test_adjointness_probe_fails_on_nonfinite_output() -> None:
    k = pc.OperatorKernel.from_callables(
        lambda x: np.full_like(x, np.nan),
        None,
        3,
        is_symmetric=True,
        self_interaction=False,
        identity=None,
    )
    cert = pc.kernels.adjointness_probe(k, np.random.default_rng(0))
    assert cert.passed is False
    assert cert.data["nonfinite_output"] == 1


@pytest.mark.property
def test_two_column_key_labels_are_accepted_and_hashed() -> None:
    keys = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    land = pc.Landscape(np.array([1.0, 2.0, 3.0]), labels=keys)
    assert land.labels is not None and land.labels.shape == (3, 2)
    assert land.labels_sha256 is not None


@pytest.mark.property
def test_streamed_hash_equals_framed_hash() -> None:
    from prefcent._hashing import frame_array, hash_array

    a = np.random.default_rng(1).normal(size=(37, 53))
    assert hash_array(a) == hashlib.sha256(frame_array(a)).hexdigest()
    assert hash_array(np.asfortranarray(a)) == hash_array(a)


# ---- identities cannot be forged; declarations are never coerced ----


@pytest.mark.property
def test_identity_subclass_or_claimed_builtin_is_not_a_builtin() -> None:
    class _Sub(pc.Identity):
        def apply(
            self,
            mass: np.ndarray,
            inflow: np.ndarray,
            landscape: Any,
            active: np.ndarray,
        ) -> np.ndarray:
            return np.asarray(inflow * 1.0001)

    k, land = _tiny()
    r = pc.Model(k, land, gamma=1.0, closure=_Sub()).evolve(max_iter=1)
    assert r.manifest.fingerprint["model"]["closure_identity"] is None
    assert r.manifest.fingerprint["reproducible"] is False

    claimed = _NamedClosure(0.0, {"kind": "builtin", "name": "identity"})
    with pytest.raises(ValueError, match="reserved"):
        pc.Model(k, land, gamma=1.0, closure=claimed).evolve(max_iter=1)


@pytest.mark.property
def test_trusted_must_be_a_real_bool() -> None:
    k, land = _tiny()
    bad = _NamedClosure(0.0, {"kind": "consumer", "value": "x", "trusted": "false"})
    with pytest.raises(TypeError, match="trusted"):
        pc.Model(k, land, gamma=1.0, closure=bad).evolve(max_iter=1)
    with pytest.raises(TypeError, match="trusted"):
        pc.DenseKernel(
            np.zeros((2, 2)),
            is_symmetric=True,
            self_interaction=False,
            identity={"value": "x", "trusted": "false"},
        )


@pytest.mark.property
def test_semantic_booleans_are_never_coerced() -> None:
    edges = np.array([[0, 1]])
    w = np.array([1.0])
    with pytest.raises(TypeError, match="directed"):
        pc.kernels.graph_costs(edges, w, 2, sources=np.array([0, 1]), directed="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="self_interaction"):
        pc.CirculantKernel.ring(8, 1.0, 1.0, 2.0, self_interaction="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="self_interaction"):
        pc.kernels.from_costs(
            np.ones((2, 2)),
            beta=2.0,
            d0=1.0,
            self_interaction="false",  # type: ignore[arg-type]
        )


@pytest.mark.property
def test_schema_version_must_be_an_integer() -> None:
    k, land = _tiny()
    text = pc.Model(k, land, gamma=1.0).evolve(max_iter=1).manifest.to_json()
    for bad in (
        '"schema_version":"1"',
        '"schema_version":1.5',
        '"schema_version":true',
    ):
        with pytest.raises(ValueError, match="schema_version"):
            pc.RunManifest.from_json(text.replace('"schema_version":1', bad))


@pytest.mark.property
def test_thread_variables_are_parsed_before_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prefcent._manifest import blas_threads, effective_blas_threads

    monkeypatch.setenv("OMP_NUM_THREADS", "4,3,2")
    assert blas_threads()["OMP_NUM_THREADS"] == 4  # OpenMP nesting list: first level
    monkeypatch.setenv("OMP_NUM_THREADS", "four")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        blas_threads()
    # an unrelated variable is not evidence for the detected backend
    assert effective_blas_threads(
        {"OMP_NUM_THREADS": None, "OPENBLAS_NUM_THREADS": None, "MKL_NUM_THREADS": 1},
        "openblas",
    ) == (None, None)
    assert effective_blas_threads(
        {"OMP_NUM_THREADS": 16, "OPENBLAS_NUM_THREADS": None, "MKL_NUM_THREADS": 1},
        "openblas",
    ) == (16, "OMP_NUM_THREADS")


@pytest.mark.property
def test_malformed_thread_variable_fails_before_the_first_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    k, land = _tiny()
    calls: list[int] = []
    monkeypatch.setenv("OMP_NUM_THREADS", "sixteen")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        pc.Model(k, land, gamma=1.0).evolve(
            max_iter=3, callback=lambda i: calls.append(1)
        )
    assert calls == []
