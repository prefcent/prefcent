# prefcent

**prefcent** is the canonical open-source solver for preferential centrality
(Hellervik, Nilsson & Andersson 2019): linear attraction, gravity flows, and a
mass-conserving balanced fixed point. It takes an operator (`matvec` /
`rmatvec`) plus a capacity landscape and runs the model's own adjustment
dynamics via `evolve()`.

## The model

With interaction kernel $K_{ij} = f(c_{ij})$ built from travel costs and attraction
$W_j = \gamma a_j + R_j$, the interaction sent from zone $i$ to zone $j$ is

$$S_{ij} = a_i \frac{W_j K_{ij}}{\sum_k W_k K_{ik}},$$

and preferential centrality is the balanced state in which every zone's activity
equals its inflow, under conservation of the total:

$$a_j = \sum_i S_{ij}, \qquad \sum_j a_j = \sum_j R_j.$$

In matrix notation, with the potential $A = KW$ and elementwise product $\odot$
and division, the same fixed point reads

$$a = W \odot K^{\mathsf{T}}\left(\frac{a}{A}\right).$$

`evolve()` iterates exactly this map — damped by $\omega$, renormalised to the total
capacity each step. At $\gamma = 0$ the map is linear in $a$ and the fixed point is an
eigenvector centrality; $\gamma > 0$ adds the preferential feedback of activity
attracting activity.

## Symbol table

The mathematics keeps its single letters; the public API uses words.

| symbol | API name | meaning |
|---|---|---|
| `a` | `mass` | the state; conserved, `Σ mass = Σ capacity`; its fixed point is the preferential centrality |
| `R` | `capacity` | the landscape's fixed capacity |
| `K` | `kernel` | `K_ij` weights interaction from zone `i` to `j`; built by `kernels.from_costs` / `from_graph` |
| `W = γa + R` | `attraction` | computed inside `evolve()`; not exposed in v0.1 |
| `A = K·W` | `potential` | the potential of attraction at each origin |
| `I = W ⊙ Kᵀ(a/A)` | `inflow` | the raw balance update; `⊙` is the elementwise product |
| `s = a/A` | `share_rate` | computed inside `evolve()`; to be exposed in a later release |
| `a/R` | `density` | computed by the caller from `mass` and `capacity` |

Parameters keep the paper's symbols (`gamma`, `beta`, `d0`, `omega`).

## Usage

```python
import prefcent as pc

# One call from a deposited network to a kernel. Symmetry is derived from
# directed=False (the array-path builder, from_costs, needs it declared for
# an undirected network — Dijkstra costs are asymmetric at ~1e-14 relative).
kernel = pc.kernels.from_graph(
    edges,
    weights,
    n_nodes,
    sources=zones,
    directed=False,
    beta=2.0,
    d0=5.0,  # a five-minute start/end penalty, in the same unit as the edge weights
    cost_units="minutes",
    self_interaction=False,
)
result = pc.Model(kernel, pc.Landscape(capacity, labels=zone_ids), gamma=1.0).evolve(
    max_iter=200
)
density = result.mass / result.landscape.capacity
```

See `docs/tutorial.md` for the `pa_small` walkthrough.

