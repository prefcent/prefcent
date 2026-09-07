# REPRODUCE — what prefcent 0.1.0 reproduces, and to what tolerance

This file is the cross-implementation protocol for the release: what an independent
implementation, or a reader with the deposited data, should be able to match, and
how closely. Tolerances below are **measured**, not chosen.

## The reference run

The preferential-centrality model (Hellervik, Nilsson & Andersson 2019 —
see `CITATION.cff`) as run for the Porto Alegre land-value study, a preprint under review
([https://doi.org/10.21203/rs.3.rs-6264491/v1](https://doi.org/10.21203/rs.3.rs-6264491/v1)): γ ∈ {1, 0}, β = 2,
`d0 = 5000` (a five-minute start/end penalty in metre-equivalents at 60 km/h, 1000 per
minute), `f(c) = (c + d0)^(−β)`, self-interaction off, capacity `R` = buildable area
in a 30 m buffer per segment, a₀ = R, direct substitution (ω = 1), exactly **200
iterations with no tolerance**, conservation renormalisation each step. In prefcent
that is the package's **defaults** plus `evolve(max_iter=200, tol=None)`:

```python
kernel = pc.kernels.from_graph(
    edges,
    cost,
    n_nodes,
    sources=zone_nodes,
    directed=False,
    beta=2.0,
    d0=5000.0,
    cost_units="metre-equivalents at 60 km/h (1000 per minute)",
    self_interaction=False,
)
landscape = pc.Landscape(R, labels=keys)
result = pc.Model(kernel, landscape, gamma=1.0).evolve(max_iter=200)
density = result.mass / result.landscape.capacity  # the preprint's `cd` columns
```

The reference network is the preprint's own segment file: **65,357 segments**, of
which **56,162 carry capacity and are zones**; the remaining 9,195 carry routes but no
mass and are exported with `density = 0`. Outputs are scattered onto segments by the
`(node1, node2)` key. The routing graph is recovered from the segment file by a
documented recipe (shipped with the study's data deposit, not with this package):
every segment is split at its midpoint, the two halves carry `cost/2` each, and each
zone sits at the midpoint of its own segment with a zero-cost connector — three
properties measured on a small extract where both the routing graph and the segment
network survive. Self-loop segments produce duplicate half-edges; `graph_costs`
resolves duplicates by minimum, which this case requires.

## Hardware and time (measured)

Peak memory is the cost matrix plus the kernel, **2 × 24.74 GiB ≈ 50 GiB**, plus the
Dijkstra worker pool's transient (≈0.5 GB per worker). Choose `n_jobs` to fit. Wall
time on 64 cores: shortest paths **49 s** (0.84 ms per source), kernel build ≈2 min
including the symmetry evidence pass and the content hash, each 200-iteration run
**≈2 min** (202 `matvec` + 201 `rmatvec` applications). A 16 GB machine cannot run the
full case; it can run the `pa_small` tutorial and the property battery.

## What was matched

**The preprint's columns themselves** — `cd1g1b2k0` / `ca1g1b2k0` (γ = 1) and
`cd4g0b2k0` / `ca4g0b2k0` (γ = 0) of the preprint's segment file — reproduced from that
file's own network and capacities and nothing else (the result columns are read only
to score against), on all 65,357 rows:


| run          | within 0.1 %   | median relative | p99     | max relative |
| ------------ | -------------- | --------------- | ------- | ------------ |
| γ = 1, β = 2 | **100.0000 %** | 2.2e-14         | 6.1e-13 | **1.6e-12**  |
| γ = 0, β = 2 | **100.0000 %** | 8.9e-15         | 2.2e-13 | 4.3e-13      |


**Convergence of the reference run itself — and why the budget matters.** With no
tolerance, the status is `FIXED_BUDGET`; the run reports its final step norm
(`max|Δmass| / mean capacity`): **2.2e-5** for γ = 1 on the reference network and **≈1e-15** (machine precision — a linear map) for
γ = 0. The γ = 1 run is therefore *near* but not *at* its fixed point after 200
iterations, and it matches the preprint's column to 1e-14 precisely because the
preprint's run also stopped at 200. **The 200-iteration budget is a reproduction
requirement, not a convenience: run to convergence, the numbers would not match.**
This is why the budget-only stop exists as a first-class status. The state minimum
over the whole trajectory was 2.24 for γ = 1 and 4.84 for γ = 0 — comfortably
positive, so the domain guards were never near firing.

## Provenance

Every run emits a `RunManifest`: kernel provenance (`graph_sha256`, source index hash,
the decay form and `cost_units`), `is_symmetric` **derived** from `directed=False`
with the cost matrix's measured accumulation asymmetry (1.7e-10 absolute, ≈1e-14
relative) recorded as evidence, the capacity hash, solver settings, BLAS thread
counts, certificates (`isolated_zones`: none in this network), status, step norm,
operator counts, and the SHA-256 of the result. *Identical fingerprint ⇒ identical
result hash* on one platform, for runs whose fingerprint says `reproducible: true`
(a clean commit or an exact release version, a content-identified kernel, a
built-in or trusted closure, known BLAS thread counts) **and whose status is not
`CANCELLED`** — a cancelled run's state depends on when it was stopped. The
deposited manifests are the reference a reader compares against.

This guarantee requires the kernel's contents to remain unchanged after its
identity is recorded. `DenseKernel` wraps the caller's array without copying and
hashes it once at construction; the caller must not modify that buffer, including
through another view. `evolve()` does not re-hash it. Call
`kernel.verify_identity()` to check that the contents still match the recorded
hash; it raises if they have changed. Construct a new `DenseKernel` when changing
weights so the new contents are validated and receive their own identity.

## Export (consumer side)

prefcent returns zone-indexed arrays and never writes files. The Porto Alegre export
contract — GeoPackage preferred, `(node1, node2)` carried through unchanged,
`ca…`/`cd…` column naming, `density = 0` on zero-capacity segments — is the study pipeline's; it is reproduced in `docs/` when the pipeline's contract is released.

## Fixture-scale checks anyone can run

`pytest -m "golden or property"`: the update map against recorded reference
trajectories on three synthetic grids and the 76-zone real-structure `pa_small` at
≤ 3.1e-15; `pa_small`'s cost matrix rebuilt from its routing graph **exactly**; the
tutorial end to end.
