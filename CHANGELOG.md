# Changelog

## 0.1.1 — 2026-10-02

### Added

- `DensityPenaltyV1(kappa, rho=2.0)` — the additive, capacity-weighted density
  penalty closure: `C = inflow − kappa · capacity ⊙ (g − ḡ_R)` with
  `g = (mass/capacity)^rho`. Built-in identity; fail-stop on domain breach; margin
  certificate with `kappa_at_zero`. `kappa = 0` reproduces `Identity` exactly.

### Fixed — release review

- Density-penalty margins serialize as finite numbers or `"unbounded"` when the
  penalty vanishes. At zero strength, the update bypasses penalty arithmetic and
  reports `"unavailable"` if the optional margin cannot be computed numerically.
- Non-finite penalty, closed-update, and margin calculations at nonzero strength
  raise `NumericalBreach` with state and step attached, including final diagnostics.

## 0.1.0 — 2026-09-07

### Fixed — release review

- Default graph worker counts fall back to CPU affinity or CPU count when the
  cgroup quota file is absent.
- Directed networks with one-way reachability emit JSON-safe symmetry evidence.
- Citation metadata includes the software author; dense-kernel documentation
  states the caller's buffer immutability responsibility and identity check.
- Property tests set up their own thread environment and cover portable worker
  selection and one-way reachability through manifest serialization.

### Added — the cost path and the remaining kernels

- `OperatorKernel` and `OperatorKernel.from_callables`; `CirculantKernel.ring`.
- `prefcent.kernels`: `from_costs`, `graph_costs`, `from_graph`,
  `adjointness_probe`.
- `pa_small` tutorial (`examples/pa_small_tutorial.py`, `docs/tutorial.md`)
  and deposit schema by example (`docs/deposit-schema.md`).

### Added — the core package

- Public names: `Landscape`, `Model`, `Identity`, `Closure`, `Kernel`,
  `DenseKernel`, `EvolveResult`, `EvolveStatus`, `StepInfo`, `Certificate`,
  `RunManifest`, `PrefcentError`, `NumericalBreach`, `ClosureDomainBreach`,
  `KernelError`, `ModelDomainError`.
- `prefcent.DenseKernel` wrapping a float64 dense array (declared
  symmetry, content hash, no copy).
- `Model.evolve()` — damped iteration of the base map with the `Identity`
  closure, isolation decided once before step 1, conservation renormalisation
  last, four-value `EvolveStatus` (`CONVERGED`, `FIXED_BUDGET`,
  `BUDGET_EXHAUSTED`, `CANCELLED`).
- `EvolveResult` with `mass`, `potential`, `landscape`, `active`, named
  statistics (`trajectory_min`, `final_min`, `final_max_density`), certificates
  (`isolated_zones`, `closure:identity`), and a `RunManifest` split into
  fingerprint and observations.
- Starts: `"capacity"`, `"custom"` (projected and recorded), `"continued"` /
  `"continued_foreign"` from a prior result.
- Public fixtures: synthetic grids and one 76-zone real-structure case.
