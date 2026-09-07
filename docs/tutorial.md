# Tutorial: `pa_small` end to end

This is the reader's route in miniature: a deposited network, shortest-path
costs, the named power kernel, `evolve()`, and density. It reproduces the
published preferential-centrality **model** (Hellervik, Nilsson & Andersson
2019 — citation in `README.md`) on real Porto Alegre structure. The 2019
paper contains no worked example this tutorial could regenerate.

Run:

```bash
python examples/pa_small_tutorial.py
```

The script, in order:

1. Load `tests/fixtures/pa_small.npz` (schema in `docs/deposit-schema.md`).
2. `kernels.from_graph(graph_edges, graph_costs, n_nodes, sources=zone_graph_index,
   directed=False, beta=2, d0=5000, cost_units="metre-equivalents at 60 km/h (1000 per minute)")`
   — shortest paths on the full routing graph, then the named power decay, in one
   call. Symmetry is *derived* from `directed=False`; the cost matrix is numerically
   asymmetric at ≈1e-12 from path-sum accumulation, and the manifest records that as
   `symmetry_check` evidence.
3. If you hold a cost matrix rather than a graph, use `kernels.from_costs(costs, ...)`
   and **declare** `is_symmetric=True` for an undirected network: the array path
   derives symmetry from an *exact* check, which accumulation noise fails.
4. `Landscape(R, labels=zone_ids)`.
5. `Model(gamma=1)` and `evolve(max_iter=200)`.
6. `density = mass / capacity`.
7. Print status, final step norm, the `isolated_zones` and `closure:identity`
   certificates, `fingerprint_sha256`, and `reproducible`.

There is no parameter sweep.
