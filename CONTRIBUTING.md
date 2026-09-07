# Contributing to prefcent

## Scope

This repository is the canonical public implementation of the published
preferential-centrality model (Hellervik, Nilsson & Andersson 2019 — linear
attraction, gravity flows, a mass-conserving balanced fixed point).

In scope: the base model, the generic machinery around it (kernel builders and
operators, content hashing, run manifests, certificates), and the evidence that it
reproduces the reference results (`REPRODUCE.md`). An extension to the model itself is a specification
change, not an implementation detail — please open an issue before writing one.

Docstrings, test names, commit messages and changelog entries are all published
along with the code. Write them for a reader who has nothing but this repository.

## Design rules

Each of these rules exists because breaking it has produced a wrong number in the
project's history. They apply to the whole API, including parts not yet released:

1. Numerical guards must not silently change the model. A domain violation raises an
   error with the offending state attached; there are no fallback floors or clamps.
2. Convergence and feasibility certificates are returned in the result object. A
   certificate that only appears in log output does not count.
3. `evolve()` runs the model's own adjustment dynamics. `solve()` finds fixed points
   by any method and must certify stability before claiming physical relevance. The
   two stay separate.
4. The name of a statistic states which estimator produced it (for example, pooled
   maximum versus per-configuration median).
5. A parameter sweep that loses a solution branch reports where the branch ended and
   why the sweep stopped there. It may not claim the branch turned around (a fold: a
   parameter value beyond which that equilibrium no longer exists) unless a test
   located the turning point. A sweep alone cannot tell a fold from a solver failure.
6. Every run can emit a `RunManifest` with enough information to reproduce it
   bit-for-bit.

## Dependencies

Hard runtime dependencies are **numpy and scipy only**. Everything else belongs in the
`[dev]` extra. Adding a runtime dependency is a specification change.

## Reproducing reference runs

The v0.1.0 reproduction (see `REPRODUCE.md`) needed no legacy switches: every run's
manifest carries a `legacy_policies` block, and in this release it is always empty.
The principle behind that block binds future work: if a historical behavior ever has
to be selectable, it gets a flag named after the *choice* — a normalization or an
isolated-zone policy by its own name — never a blanket `legacy=True`, and every
active policy is recorded in the manifest. `REPRODUCE.md` (per release) states what
an independent re-implementation should match, and to what tolerance.

## Versioning

Semver. `0.x` may break API with a changelog notice; `1.0` freezes the core protocols.
License: MIT. `CITATION.cff` points at the 2019 paper; further citations are added as
they are published.
