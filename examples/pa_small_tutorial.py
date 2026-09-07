"""pa_small end to end: deposited network → costs → kernel → evolve.

This reproduces the published preferential-centrality *model*
(Hellervik, Nilsson & Andersson 2019; citation in the package README) on
real Porto Alegre structure. It is not a worked example from that paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import prefcent as pc

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "pa_small"
COST_UNITS = "metre-equivalents at 60 km/h (1000 per minute)"


def main() -> pc.EvolveResult:
    meta = json.loads(FIXTURE.with_suffix(".json").read_text())
    z = np.load(FIXTURE.with_suffix(".npz"))
    n_nodes = int(meta["graph_nodes"])
    # The one-call route: shortest paths on the full routing graph,
    # then the named power decay. Symmetry is derived from directed=False —
    # the cost matrix itself is numerically asymmetric at ~1e-12 (path-sum
    # accumulation), which the manifest records as evidence.
    kernel = pc.kernels.from_graph(
        z["graph_edges"],
        z["graph_costs"],
        n_nodes,
        sources=z["zone_graph_index"],
        directed=False,
        n_jobs=1,
        beta=2.0,
        d0=5000.0,
        cost_units=COST_UNITS,
        self_interaction=False,
    )
    landscape = pc.Landscape(z["R"], labels=z["zone_ids"])
    model = pc.Model(kernel, landscape, gamma=1.0)
    result = model.evolve(max_iter=200)
    density = result.mass / result.landscape.capacity
    print(f"max density: {density.max():.6g}")
    print(f"kernel is_symmetric: {kernel.is_symmetric} ({kernel.is_symmetric_source})")
    print(f"status: {result.status.name}")
    print(f"final_step_norm: {result.final_step_norm}")
    print(f"certificates: {list(result.certificates)}")
    iso = result.certificates["isolated_zones"]
    print(f"isolated_zones: count={iso.data['count']} passed={iso.passed}")
    clos = result.certificates["closure:identity"]
    print(f"closure:identity: passed={clos.passed}")
    print(f"fingerprint_sha256: {result.manifest.fingerprint_sha256}")
    print(f"reproducible: {result.manifest.fingerprint['reproducible']}")
    return result


if __name__ == "__main__":
    main()
