# prefcent

**prefcent** is the canonical open-source solver for preferential centrality
(Hellervik, Nilsson & Andersson 2019): linear attraction, gravity flows, and a
mass-conserving balanced fixed point. It takes an operator (`matvec` /
`rmatvec`) plus a capacity landscape and runs the model's own adjustment
dynamics via `evolve()`.

Version 0.1.1 adds an optional density penalty: additive negative feedback on
relative density.

## The base model

With interaction kernel $K_{ij} = f(c_{ij})$ built from travel costs and attraction
$W_j = \gamma a_j + R_j$, the interaction sent from zone $i$ to zone $j$ is

$$S_{ij} = a_i \frac{W_j K_{ij}}{\sum_k W_k K_{ik}},$$

and preferential centrality is the balanced state in which every zone's activity
equals its inflow, under conservation of the total:

$$a_j = \sum_i S_{ij}, \qquad \sum_j a_j = \sum_j R_j.$$

In matrix notation, with the potential $A = KW$ and elementwise product $\odot$
and division, the same fixed point reads

$$a = W \odot K^{\mathsf{T}}\left(\frac{a}{A}\right).$$

With the default `Identity` closure, `evolve()` iterates this map — damped by
$\omega$, renormalised to the total capacity each step. At $\gamma = 0$ the base
map is linear in $a$ and the fixed point is an eigenvector centrality;
$\gamma > 0$ adds the preferential feedback of activity attracting activity.

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
import numpy as np
import prefcent as pc

# A three-zone undirected network, with edge weights in minutes.
edges = np.array([[0, 1], [1, 2]])
weights = np.array([2.0, 3.0])
n_nodes = 3
capacity = np.array([1.0, 2.0, 1.0])

kernel = pc.kernels.from_graph(
    edges,
    weights,
    n_nodes,
    sources=np.arange(n_nodes),
    directed=False,
    n_jobs=1,
    beta=2.0,
    d0=5.0,  # a five-minute start/end penalty, in the same unit as the edge weights
    cost_units="minutes",
    self_interaction=False,
)
result = pc.Model(kernel, pc.Landscape(capacity), gamma=1.0).evolve(max_iter=200)
density = result.mass / result.landscape.capacity
```

See `docs/tutorial.md` for the `pa_small` walkthrough.

## Density penalty

`DensityPenaltyV1(kappa, rho=2.0)` applies negative feedback on relative density
before damping and renormalisation. The raw inflow into zone $j$ is
$I_j = \sum_i S_{ij}$: the total activity sent to that zone from all origins,
using the interaction flows defined above. The penalty adjusts this inflow to
give the proposed next activity $C_j$:

$$C_j = I_j - \kappa R_j (g_j - \bar g_R), \qquad
g_j = \left(\frac{a_j}{R_j}\right)^\rho, \qquad
\bar g_R = \frac{\sum_{i \in V} R_i g_i}{\sum_{i \in V} R_i},$$

where $V$ is the active set of non-isolated zones. The correction sums to zero
over that set, conserving total mass in exact arithmetic. A zone with
$g_j > \bar g_R$ has activity subtracted from its raw inflow; a zone below that
mean receives an addition.

At each step, the current activity $a$ determines both the inflow $I$ and the
penalty. Their combination $C$ is the proposed next state. With `omega=1`, the
next iterate is $C$; with damping, it is $(1-\omega)a + \omega C$. The solver
then renormalises to the total capacity to correct floating-point drift.

At a fixed point, the proposed state equals the current state:
$a_j = C_j = I_j - \kappa R_j(g_j - \bar g_R)$. Thus a zone with a positive
penalty must receive enough inflow to cover both its activity and that penalty.
The balance is now between activity and the adjusted inflow.

`kappa` controls the penalty strength and must be finite and nonnegative;
`rho` controls the density exponent and must be finite and positive.
`kappa=0` recovers the base-model update and fixed-point equation $a_j = I_j$.

Here is a small two-zone example:

```python
import numpy as np
import prefcent as pc

kernel = pc.DenseKernel(
    np.array([[1.0, 0.5], [0.5, 1.0]]),
    is_symmetric=True,
    self_interaction=True,
)
result = pc.Model(
    kernel,
    pc.Landscape(np.array([1.0, 2.0])),
    gamma=1.0,
    closure=pc.DensityPenaltyV1(kappa=0.01, rho=2.0),
).evolve(max_iter=200)

density = result.mass / result.landscape.capacity
```

## Citing

Cite the 2019 paper together with the software version used. When using the density
penalty, also cite *Preferential centrality with a density penalty: formulation and
an open-source implementation* (Hellervik, 2026). `CITATION.cff` contains both
references. The
[Zenodo software record](https://doi.org/10.5281/zenodo.22643613) covers all
versions; the existing [0.1.0 archive](https://doi.org/10.5281/zenodo.22643614)
has its own version DOI.
