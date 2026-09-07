"""Kernel protocol and the dense in-RAM wrapper."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np

from prefcent._hashing import hash_array


class Kernel(Protocol):
    """Finite, nonnegative square operator: ``matvec`` and ``rmatvec``.

    Protocol requirement: entries are finite and ``≥ 0``, so the
    isolation test ``K·1 == 0`` ⇔ a zero row is valid. ``DenseKernel``
    validates that in the same pass as the content hash. Operator kernels
    declare it by construction; the evolve loop fail-stops on a negative
    or non-finite output.
    """

    shape: tuple[int, int]
    dtype: np.dtype[Any]
    is_symmetric: bool
    self_interaction: bool

    def matvec(self, x: np.ndarray) -> np.ndarray: ...

    def rmatvec(self, x: np.ndarray) -> np.ndarray: ...


def _as_consumer_identity(
    identity: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    # A consumer-supplied identity is normalised to {"value", "trusted"}: a
    # bare string is an unverified claim, so it is recorded as trusted=False.
    if identity is None:
        return None
    if isinstance(identity, str):
        return {"value": identity, "trusted": False}
    if isinstance(identity, dict) and "value" in identity:
        trusted = identity.get("trusted", False)
        if not isinstance(trusted, bool):
            raise TypeError("identity 'trusted' must be a bool (no coercion)")
        return {"value": str(identity["value"]), "trusted": trusted}
    raise ValueError(
        "identity must be None, a string, or a mapping with 'value' and 'trusted'"
    )


class DenseKernel:
    """In-RAM dense kernel. Wraps ``array`` without copying.

    The caller must keep the buffer unchanged for this kernel's lifetime,
    including through any other views of the array. Validation and content
    hashing happen once at construction; ``evolve()`` does not re-hash it.
    Call ``verify_identity()`` to check that the buffer still matches its
    recorded hash. To change weights, construct a new ``DenseKernel`` so its
    validation and manifest identity describe the new contents.
    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        is_symmetric: bool,
        self_interaction: bool,
        identity: str | dict[str, Any] | None = None,
        is_symmetric_source: str = "declared",
        provenance: dict[str, Any] | None = None,
        symmetry_check: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(is_symmetric, bool):
            raise TypeError("is_symmetric is required and must be bool (no default)")
        if not isinstance(self_interaction, bool):
            raise TypeError("self_interaction must be bool")
        if is_symmetric_source not in ("declared", "derived"):
            raise ValueError("is_symmetric_source must be 'declared' or 'derived'")
        arr = array
        if not isinstance(arr, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"kernel must be square 2-D, got shape {arr.shape}")
        if arr.dtype != np.float64:
            raise ValueError(
                f"v0.1.0 accepts float64 kernels only, got {arr.dtype}; never upcast"
            )
        if not np.isfinite(arr).all():
            raise ValueError("kernel entries must be finite")
        if bool(np.any(arr < 0.0)):
            raise ValueError("kernel entries must be ≥ 0")

        self._array = arr
        self.shape: tuple[int, int] = (int(arr.shape[0]), int(arr.shape[1]))
        self.dtype: np.dtype[Any] = np.dtype(np.float64)
        self.is_symmetric = is_symmetric
        self.self_interaction = self_interaction
        self.is_symmetric_source = is_symmetric_source
        self.content_sha256 = hash_array(arr)
        self.consumer_identity = _as_consumer_identity(identity)
        self.provenance = provenance
        self.symmetry_check = symmetry_check

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, self._array @ x)

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        # A kernel declared symmetric applies the same operator for the
        # transpose; the declaration, not a numerical check, selects this —
        # and the declaration is recorded in the manifest.
        if self.is_symmetric:
            return self.matvec(x)
        return cast(np.ndarray, self._array.T @ x)

    def verify_identity(self) -> str:
        """Re-hash the wrapped array; raise if it no longer matches."""
        current = hash_array(self._array)
        if current != self.content_sha256:
            raise ValueError(
                "kernel buffer changed after construction "
                f"(stored {self.content_sha256}, now {current})"
            )
        return current
