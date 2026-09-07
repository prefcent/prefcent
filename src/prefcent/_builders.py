"""Cost-matrix builders: ``graph_costs``, ``from_costs``, ``from_graph``."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from prefcent._hashing import hash_array, hash_graph
from prefcent._kernels import DenseKernel

CPU_MAX_PATH = Path("/sys/fs/cgroup/cpu.max")
MAX_BLOCK_BYTES = 256 * 1024 * 1024


def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def cgroup_n_jobs(path: Path | None = None) -> int:
    """Use the cgroup quota when available; otherwise affinity or CPU count."""
    target = path if path is not None else CPU_MAX_PATH
    try:
        text = target.read_text().strip()
    except FileNotFoundError:
        return _available_cpu_count()
    parts = text.split()
    if len(parts) != 2:
        raise ValueError(f"unrecognised cgroup cpu.max: {text!r}")
    quota_s, period_s = parts
    # Quota ``max`` means no cap: with no quota to oversubscribe against, the
    # affinity count is the right size.
    if quota_s == "max":
        return _available_cpu_count()
    quota = int(quota_s)
    period = int(period_s)
    if period <= 0:
        raise ValueError(f"invalid cgroup period: {text!r}")
    return max(1, quota // period)


def resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return cgroup_n_jobs()
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be an integer ≥ 1 or None")
    return n_jobs


def source_block_size(n_nodes: int, max_bytes: int = MAX_BLOCK_BYTES) -> int:
    n = max(int(n_nodes), 1)
    return max(1, int(max_bytes // (8 * n)))


def _as_index_1d(name: str, values: np.ndarray, n_nodes: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty 1-D index array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    out = np.asarray(arr, dtype=np.int64)
    if int(np.unique(out).shape[0]) != int(out.shape[0]):
        raise ValueError(f"{name} must be unique")
    if bool(np.any(out < 0) or np.any(out >= n_nodes)):
        raise ValueError(f"{name} entries must lie in 0..n_nodes-1")
    return out


def _dedup_edges(
    edges: np.ndarray,
    weights: np.ndarray,
    *,
    directed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(edges, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f"edges must have shape (m, 2), got {e.shape}")
    if w.shape != (e.shape[0],):
        raise ValueError("weights must have shape (m,) matching edges")
    keep = e[:, 0] != e[:, 1]
    e = e[keep]
    w = w[keep]
    if e.shape[0] == 0:
        return e, w
    if not directed:
        lo = np.minimum(e[:, 0], e[:, 1])
        hi = np.maximum(e[:, 0], e[:, 1])
        e = np.stack([lo, hi], axis=1)
    order = np.lexsort((w, e[:, 1], e[:, 0]))
    e = e[order]
    w = w[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    first = np.ones(e.shape[0], dtype=bool)
    first[1:] = ~same
    return e[first], w[first]


def _build_csr(
    edges: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
    *,
    directed: bool,
) -> Any:
    e, w = _dedup_edges(edges, weights, directed=directed)
    if bool(np.any(e < 0) or np.any(e >= n_nodes)):
        raise ValueError("edge endpoints must lie in 0..n_nodes-1")
    csr = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    # Zero-weight edges are edges: a zero-cost link still connects its
    # endpoints, so they must survive the conversion to CSR.
    if int(csr.nnz) != int(e.shape[0]):
        raise ValueError(
            "CSR dropped stored edges (nnz "
            f"{csr.nnz} != {e.shape[0]}); zero-weight edges must survive"
        )
    return csr


def _dijkstra_slice(
    csr: Any,
    directed: bool,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    dist = dijkstra(csr, directed=directed, indices=source_indices)
    dist_arr = np.asarray(dist, dtype=np.float64)
    if dist_arr.ndim == 1:
        dist_arr = dist_arr.reshape(1, -1)
    return dist_arr[:, target_indices]


def graph_costs(
    edges: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
    *,
    sources: np.ndarray,
    targets: np.ndarray | None = None,
    directed: bool = False,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Shortest-path costs from sources to targets over an edge-weighted graph."""
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes < 1:
        raise ValueError("n_nodes must be an integer ≥ 1")
    if not isinstance(directed, bool):
        raise TypeError("directed must be a bool (no coercion)")
    w = np.asarray(weights)
    if w.ndim != 1:
        raise ValueError("weights must be 1-D")
    if not np.isfinite(w).all() or bool(np.any(w < 0.0)):
        raise ValueError("weights must be finite and ≥ 0")
    e = np.asarray(edges)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f"edges must have shape (m, 2), got {e.shape}")
    if e.shape[0] != w.shape[0]:
        raise ValueError("edges and weights must have the same length")
    if not np.issubdtype(e.dtype, np.integer):
        raise ValueError("edges must have an integer dtype")
    if bool(np.any(e < 0) or np.any(e >= n_nodes)):
        raise ValueError("edge endpoints must lie in 0..n_nodes-1")
    src = _as_index_1d("sources", sources, n_nodes)
    tgt = src if targets is None else _as_index_1d("targets", targets, n_nodes)
    csr = _build_csr(e, w, n_nodes, directed=directed)
    workers = resolve_n_jobs(n_jobs)
    block = source_block_size(n_nodes)
    n_s = int(src.shape[0])
    if workers > 1:
        block = min(block, max(1, (n_s + workers - 1) // workers))
    n_t = int(tgt.shape[0])
    out = np.empty((n_s, n_t), dtype=np.float64)
    ranges: list[tuple[int, np.ndarray]] = []
    for start in range(0, n_s, block):
        ranges.append((start, src[start : start + block]))
    if workers == 1 or len(ranges) == 1:
        for start, chunk in ranges:
            out[start : start + chunk.shape[0]] = _dijkstra_slice(
                csr, directed, chunk, tgt
            )
    else:
        # Bounded submission: at most 2·workers blocks in flight, consumed in
        # order, so completed results never pile up in memory.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            window = 2 * workers
            pending: list[tuple[int, int, Any]] = []
            next_i = 0
            while next_i < len(ranges) or pending:
                while next_i < len(ranges) and len(pending) < window:
                    start, chunk = ranges[next_i]
                    fut = pool.submit(_dijkstra_slice, csr, directed, chunk, tgt)
                    pending.append((start, int(chunk.shape[0]), fut))
                    next_i += 1
                start, size, fut = pending.pop(0)
                out[start : start + size] = fut.result()
    return out


def _symmetry_check(c: np.ndarray, chunk: int = 4096) -> dict[str, Any]:
    n = int(c.shape[0])
    exact = True
    max_abs = 0.0
    asymmetric_reachability = False
    for i in range(0, n, chunk):
        i2 = min(i + chunk, n)
        row = c[i:i2, :]
        col = c[:, i:i2].T
        if not np.array_equal(row, col):
            exact = False
        with np.errstate(invalid="ignore"):
            diff = np.abs(row - col)
        # +inf vs +inf is NaN; costs are ≥ 0, so treat that as zero asymmetry.
        diff = np.where(np.isnan(diff), 0.0, diff)
        finite = np.isfinite(diff)
        if bool(finite.any()):
            max_abs = max(max_abs, float(diff[finite].max()))
        if bool(np.isinf(diff).any()):
            asymmetric_reachability = True
    if asymmetric_reachability:
        # A one-way-reachable pair has unbounded cost asymmetry. Keep the
        # finite statistic separate and use null for the unbounded maximum,
        # so the evidence remains valid canonical JSON.
        return {
            "exact": False,
            "max_abs_asym": None,
            "asymmetric_reachability": True,
            "max_abs_finite_asym": max_abs,
        }
    return {"exact": bool(exact), "max_abs_asym": max_abs}


def _power_weights(c: np.ndarray, beta: float, d0: float) -> np.ndarray:
    # (c + d0) ** (-beta) computed in place on one fresh array — bit-identical
    # to the expression form (verified on this repository's fixtures) and one
    # 8·n² buffer instead of three. +inf cost → 0.
    w = c + d0
    np.power(w, -beta, out=w)
    return cast(np.ndarray, w)


def _exp_weights(c: np.ndarray, decay_length: float) -> np.ndarray:
    # exp(-c / decay_length) in place on one fresh array; +inf cost → 0.
    w = c / (-decay_length)
    np.exp(w, out=w)
    return cast(np.ndarray, w)


def _decay_block_power(
    beta: float, d0: float, cost_units: str | None
) -> dict[str, Any]:
    return {
        "form": "power",
        "beta": float(beta),
        "d0": float(d0),
        "cost_units": cost_units,
    }


def _decay_block_exponential(
    decay_length: float, cost_units: str | None
) -> dict[str, Any]:
    return {
        "form": "exponential",
        "decay_length": float(decay_length),
        "cost_units": cost_units,
    }


def _decay_block_callable(
    func_name: str,
    func_params: dict[str, float] | None,
    cost_units: str | None,
) -> dict[str, Any]:
    return {
        "form": func_name,
        "params": dict(func_params) if func_params is not None else {},
        "cost_units": cost_units,
        "self_declared": True,
    }


def from_costs(
    c: np.ndarray,
    *,
    beta: float | None = None,
    d0: float | None = None,
    decay_length: float | None = None,
    func: Callable[[np.ndarray], np.ndarray] | None = None,
    func_name: str | None = None,
    func_params: dict[str, float] | None = None,
    cost_units: str | None = None,
    is_symmetric: bool | None = None,
    self_interaction: bool = False,
) -> DenseKernel:
    """Dense kernel from a square cost matrix.

    Decay is one of two named forms or a caller-supplied function:

    - power: ``beta`` and ``d0`` give ``(c + d0) ** (-beta)``.
    - exponential: ``decay_length`` gives ``exp(-c / decay_length)``. There is
      no offset parameter here: adding a constant to every cost only scales all
      weights by ``exp(-offset / decay_length)``, which cancels in the model's
      normalized interaction shares.
    - callable: ``func`` (with ``func_name`` and an explicit ``is_symmetric``);
      recorded as self-declared, so runs using it are not fingerprint-reproducible.

    If reachability differs by direction, ``symmetry_check`` records
    ``asymmetric_reachability=True``, ``max_abs_asym=None`` (unbounded), and
    ``max_abs_finite_asym`` for pairs reachable in both directions.
    """
    if not isinstance(self_interaction, bool):
        raise TypeError("self_interaction must be a bool (no coercion)")
    costs = np.asarray(c)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError(f"costs must be square 2-D, got {costs.shape}")
    if costs.dtype != np.float64:
        raise ValueError(f"costs must be float64, got {costs.dtype}; never upcast")
    if bool(np.isnan(costs).any()):
        raise ValueError("costs must not contain NaN")
    if bool(np.any(costs < 0.0)):
        raise ValueError("costs must be ≥ 0")

    power = beta is not None or d0 is not None
    exponential = decay_length is not None
    callable_path = func is not None
    if power + exponential + callable_path != 1:
        raise ValueError(
            "pass exactly one decay: beta and d0 (power), "
            "decay_length (exponential), or func with func_name"
        )

    if power:
        if beta is None or d0 is None:
            raise ValueError("the power decay requires both beta and d0")
        if not np.isfinite(beta) or not np.isfinite(d0) or beta <= 0.0 or d0 <= 0.0:
            raise ValueError("beta and d0 must be finite and > 0")
        if func_name is not None or func_params is not None:
            raise ValueError("func_name/func_params belong to the callable path")
        weights = _power_weights(costs, float(beta), float(d0))
        decay = _decay_block_power(float(beta), float(d0), cost_units)
    elif exponential:
        if decay_length is None or not np.isfinite(decay_length) or decay_length <= 0.0:
            raise ValueError("decay_length must be finite and > 0")
        if func_name is not None or func_params is not None:
            raise ValueError("func_name/func_params belong to the callable path")
        weights = _exp_weights(costs, float(decay_length))
        decay = _decay_block_exponential(float(decay_length), cost_units)
    else:
        if func is None:
            raise ValueError("callable path requires func")
        if func_name is None:
            raise ValueError("callable path requires func_name")
        if is_symmetric is None:
            raise ValueError("callable path requires an explicit is_symmetric")
        weights = func(costs)
        weights = np.asarray(weights)
        if weights.shape != costs.shape:
            raise ValueError("func must return an array of the same shape as c")
        if weights.dtype != np.float64:
            raise ValueError("func must return float64")
        if not np.isfinite(weights).all():
            raise ValueError("func must return finite weights")
        if bool(np.any(weights < 0.0)):
            raise ValueError("func must return weights ≥ 0")
        decay = _decay_block_callable(func_name, func_params, cost_units)

    if not self_interaction:
        # Named path returns a fresh array; a callable may hand back a view of
        # c, in which case (and only then) copy before zeroing the diagonal.
        if np.shares_memory(weights, costs) or not weights.flags.writeable:
            weights = np.array(weights, dtype=np.float64, copy=True)
        np.fill_diagonal(weights, 0.0)

    check = _symmetry_check(costs)
    if is_symmetric is None:
        # Named array path: derive from exact c == c.T.
        is_sym = bool(check["exact"])
        source = "derived"
    else:
        if not isinstance(is_symmetric, bool):
            raise TypeError("is_symmetric must be bool or None")
        is_sym = is_symmetric
        source = "declared"

    provenance = {
        "builder": "from_costs",
        "costs_sha256": hash_array(costs),
        "decay": decay,
    }
    return DenseKernel(
        weights,
        is_symmetric=is_sym,
        self_interaction=self_interaction,
        is_symmetric_source=source,
        provenance=provenance,
        symmetry_check=check,
    )


def from_graph(
    edges: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
    *,
    sources: np.ndarray,
    directed: bool = False,
    n_jobs: int | None = None,
    **decay: Any,
) -> DenseKernel:
    """``graph_costs`` composed with ``from_costs``; a kernel is always square."""
    if "targets" in decay:
        raise TypeError("from_graph does not accept targets")
    if not isinstance(directed, bool):
        raise TypeError("directed must be a bool (no coercion)")
    explicit_sym = decay.pop("is_symmetric", None)
    if explicit_sym is not None and not isinstance(explicit_sym, bool):
        raise TypeError("is_symmetric must be a bool or None (no coercion)")
    costs = graph_costs(
        edges,
        weights,
        n_nodes,
        sources=sources,
        targets=None,
        directed=directed,
        n_jobs=n_jobs,
    )
    if decay.get("func") is not None:
        # A callable decay can break the symmetry of an undirected cost matrix,
        # so symmetry may not be inferred from ``directed`` here: an explicit
        # declaration is required, and from_costs enforces that.
        kernel = from_costs(costs, is_symmetric=explicit_sym, **decay)
    elif explicit_sym is None:
        kernel = from_costs(costs, is_symmetric=(not directed), **decay)
        # With the named decay, symmetry follows from ``directed``: record it
        # as derived rather than declared, since no caller asserted it.
        kernel.is_symmetric_source = "derived"
    else:
        kernel = from_costs(costs, is_symmetric=explicit_sym, **decay)
    src = _as_index_1d("sources", sources, n_nodes)
    kernel.provenance = {
        "builder": "from_graph",
        "graph_sha256": hash_graph(edges, weights, n_nodes, directed),
        "sources_sha256": hash_array(src),
        "targets_sha256": hash_array(src),
        "directed": bool(directed),
        "decay": (kernel.provenance or {}).get("decay"),
    }
    return kernel
