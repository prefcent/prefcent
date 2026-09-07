"""Run the pa_small tutorial and check its final mass."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import prefcent as pc
from tests.support import load_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_tutorial_mass_matches_stored_fixed_point() -> None:
    path = ROOT / "examples" / "pa_small_tutorial.py"
    spec = importlib.util.spec_from_file_location("pa_small_tutorial", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.main()
    _meta, z = load_fixture("pa_small")
    ref = z["a_gamma1_beta2"]
    rel = float(np.max(np.abs(result.mass - ref) / np.abs(ref)))
    assert rel <= 1e-14, rel
    assert result.status is pc.EvolveStatus.FIXED_BUDGET
    assert result.landscape.labels is not None
    kernel_block = result.manifest.fingerprint["kernel"]
    assert kernel_block["is_symmetric"] is True
    assert kernel_block["is_symmetric_source"] == "derived"
    assert kernel_block["provenance"]["builder"] == "from_graph"
