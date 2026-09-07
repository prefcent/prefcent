# Deposit schema (by example)

prefcent defines no on-disk format. The `pa_small` fixture is the worked
example of a deposited network a reader can copy.

Files: `tests/fixtures/pa_small.npz` plus `pa_small.json`.

## Arrays (`pa_small.npz`)

| key | shape / dtype | meaning |
|---|---|---|
| `R` | `(n,)` float64 | zone capacity |
| `od` | `(n, n)` float64 | shortest-path costs between zones |
| `K_beta1.5`, `K_beta2.0`, `K_beta2.5` | `(n, n)` float64 | stored kernels, zero diagonal |
| `graph_edges` | `(m, 2)` int64 | routing-graph endpoints |
| `graph_costs` | `(m,)` float64 | per-edge costs (`dist / (speed/60)`) |
| `zone_graph_index` | `(n,)` int64 | zone → graph-node index, `od` row order |
| `zone_ids` | `(n,)` unicode | opaque labels |
| `segment_node1`, `segment_node2` | unicode | consumer-side segment keys |

`n = 76` zones on a `graph_nodes = 227` routing graph (`pa_small.json`).

**Zero-weight edges are edges.** 76 of 248 edges have cost 0 — one per zone,
attaching that zone to the road network. Dropping them disconnects every zone.

## JSON sidecar

`n_zones`, `graph_nodes`, `d0 = 5000` (five minutes at 1000 per minute),
`self_interaction`, `iterations`, and the named `(gamma, beta)` cases.

Cost unit: metre-equivalents at 60 km/h; `1000` is one minute.
