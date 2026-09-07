"""Array hashing and RunManifest contracts."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import prefcent as pc


@pytest.mark.property
def test_c_order_and_f_order_copies_hash_identically(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    k = z["K_beta2.0"]
    c = np.ascontiguousarray(k)
    f = np.asfortranarray(k)
    kc = pc.DenseKernel(c, is_symmetric=True, self_interaction=False)
    kf = pc.DenseKernel(f, is_symmetric=True, self_interaction=False)
    assert kc.content_sha256 == kf.content_sha256


@pytest.mark.property
def test_object_dtype_labels_raise(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    n = z["R"].shape[0]
    labels = np.empty(n, dtype=object)
    labels[:] = [str(i) for i in range(n)]
    with pytest.raises(ValueError):
        pc.Landscape(z["R"], labels=labels)


@pytest.mark.property
def test_identical_runs_share_fingerprint_and_result_hashes(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    a = model.evolve(max_iter=4)

    def _delay(_info: pc.StepInfo) -> None:
        time.sleep(0.002)

    b = model.evolve(max_iter=4, callback=_delay)
    assert a.manifest.fingerprint_sha256 == b.manifest.fingerprint_sha256
    assert (
        a.manifest.observations["result_sha256"]
        == b.manifest.observations["result_sha256"]
    )
    assert (
        a.manifest.observations["wall_time_s"] != b.manifest.observations["wall_time_s"]
    )


@pytest.mark.property
def test_manifest_json_round_trip_and_sorted_keys(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    result = pc.Model(kernel, landscape, gamma=1.0).evolve(max_iter=2)
    text = result.manifest.to_json()
    restored = pc.RunManifest.from_json(text)
    assert restored.to_json() == text
    parsed = json.loads(text)
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert list(parsed["fingerprint"].keys()) == sorted(parsed["fingerprint"].keys())
    assert " " not in text


@pytest.mark.property
def test_reproducible_is_false_when_git_is_dirty(
    grid12: tuple[dict[str, Any], np.lib.npyio.NpzFile],
) -> None:
    # There is no public hook for the source identity, so the test patches
    # the internal helper.
    _meta, z = grid12
    landscape = pc.Landscape(z["R"])
    kernel = pc.DenseKernel(z["K_beta2.0"], is_symmetric=True, self_interaction=False)
    model = pc.Model(kernel, landscape, gamma=1.0)
    with patch(
        "prefcent._model.source_identity",
        return_value={"kind": "git", "commit": "abc123", "dirty": True},
    ):
        result = model.evolve(max_iter=1)
    assert result.manifest.fingerprint["source"]["dirty"] is True
    assert result.manifest.fingerprint["reproducible"] is False
    assert "git_dirty" in result.manifest.fingerprint["reproducible_reasons"]
