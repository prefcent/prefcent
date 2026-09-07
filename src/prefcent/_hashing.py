"""Canonical array framing and SHA-256 content hashes."""

from __future__ import annotations

import hashlib
import struct
import sys

import numpy as np

# Every length in the framing below is a little-endian int64, so the same
# array hashes to the same digest on any platform.


def _dtype_tag(dtype: np.dtype) -> str:
    """Numpy dtype string in little-endian form (``<f8``, ``|S12``, ``<U8``)."""
    dt = dtype
    if dt.byteorder == ">":
        dt = dt.newbyteorder("<")
    elif dt.byteorder == "=" and sys.byteorder == "big":
        dt = dt.newbyteorder("<")
    return dt.str


def _canonical_c_le(array: np.ndarray) -> np.ndarray:
    """C-order view/copy whose in-memory bytes are little-endian."""
    a = np.ascontiguousarray(array)
    dt = a.dtype
    if dt.hasobject:
        raise ValueError("object-dtype arrays cannot be hashed")
    if dt.byteorder == ">":
        a = a.astype(dt.newbyteorder("<"), copy=True)
    elif dt.byteorder == "=" and sys.byteorder == "big":
        a = a.astype(dt.newbyteorder("<"), copy=True)
    return np.ascontiguousarray(a)


_CHUNK = 64 * 1024 * 1024


def frame_header(array: np.ndarray, canonical: np.ndarray) -> bytes:
    """``dtype_tag ‖ ndim ‖ dims`` — the framing that precedes the bytes."""
    tag = _dtype_tag(array.dtype).encode("utf-8")
    parts = [struct.pack("<q", len(tag)), tag, struct.pack("<q", canonical.ndim)]
    for dim in canonical.shape:
        parts.append(struct.pack("<q", int(dim)))
    return b"".join(parts)


def frame_update(h: hashlib._Hash, array: np.ndarray) -> None:
    """Feed ``frame_array(array)`` into ``h`` without materialising it.

    A C-contiguous little-endian input is streamed through bounded memoryviews —
    no copy at all — so hashing a multi-gigabyte kernel costs no extra memory.
    Only a non-canonical layout is copied once.
    """
    canonical = _canonical_c_le(array)
    h.update(frame_header(array, canonical))
    view = memoryview(canonical).cast("B")
    for i in range(0, len(view), _CHUNK):
        h.update(view[i : i + _CHUNK])


def frame_array(array: np.ndarray) -> bytes:
    """``dtype_tag ‖ ndim ‖ dims ‖ C-order little-endian bytes`` (small arrays)."""
    canonical = _canonical_c_le(array)
    return frame_header(array, canonical) + canonical.tobytes(order="C")


def hash_array(array: np.ndarray) -> str:
    """Hex SHA-256 of the framed array (lowercase), streamed."""
    h = hashlib.sha256()
    frame_update(h, array)
    return h.hexdigest()


def hash_graph(
    edges: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
    directed: bool,
) -> str:
    """Content hash of an edge-weighted graph, independent of edge order.

    After undirected min/max-normalisation, rows are sorted lexicographically on
    ``(u, v, weight)``, so permuting the edge list leaves the hash unchanged even
    when the same pair of endpoints appears more than once.
    """
    e = np.array(np.asarray(edges, dtype=np.int64), copy=True)
    w = np.array(np.asarray(weights, dtype=np.float64), copy=True)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f"edges must have shape (m, 2), got {e.shape}")
    if w.shape != (e.shape[0],):
        raise ValueError("weights must be 1-D aligned with edges")
    if not directed and e.shape[0] > 0:
        lo = np.minimum(e[:, 0], e[:, 1])
        hi = np.maximum(e[:, 0], e[:, 1])
        e = np.stack([lo, hi], axis=1)
    if e.shape[0] > 0:
        order = np.lexsort((w, e[:, 1], e[:, 0]))
        e = e[order]
        w = w[order]
    h = hashlib.sha256()
    frame_update(h, np.asarray([int(n_nodes)], dtype=np.int64))
    frame_update(h, e)
    frame_update(h, w)
    frame_update(h, np.asarray([1 if directed else 0], dtype=np.int8))
    return h.hexdigest()
