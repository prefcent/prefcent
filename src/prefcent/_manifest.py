"""RunManifest: fingerprint + observations, serialised as canonical JSON."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = 1

_BLAS_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def git_state(repo: Path) -> tuple[str | None, bool]:
    """``(commit, dirty)``. Missing repo → ``(None, True)``."""
    if not (repo / ".git").exists():
        return None, True
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return commit, bool(porcelain.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, True


def package_git_state() -> tuple[str | None, bool]:
    """Git identity of the installed package's repository, if present."""
    # src/prefcent/_manifest.py → repo root is parents[2]
    root = Path(__file__).resolve().parents[2]
    return git_state(root)


def source_identity() -> dict[str, Any]:
    """What code ran: a git checkout, or an installed release.

    A wheel has no ``.git``, so its identity is the released version — exact only
    when the installed files still match their own recorded hashes.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / ".git").exists():
        commit, dirty = git_state(root)
        return {"kind": "git", "commit": commit, "dirty": dirty}
    ver = package_version()
    artifact = installed_artifact_identity()
    return {
        "kind": "release",
        "version": ver,
        "artifact_sha256": artifact["artifact_sha256"],
        "verified": artifact["verified"],
        # Exact only when the installed files match their own RECORD hashes:
        # a version string is not evidence of what code ran.
        "exact": bool(artifact["verified"]),
    }


def installed_artifact_identity() -> dict[str, Any]:
    """Identity of the installed artifact from its ``RECORD``.

    ``artifact_sha256`` is the hash of the RECORD text (per-file hashes of the
    installed wheel); ``verified`` is whether every recorded file on disk still
    matches its recorded hash — a modified or foreign install fails it.
    """
    try:
        dist = distribution("prefcent")
        record = dist.read_text("RECORD")
    except PackageNotFoundError:
        record = None
    if not record:
        return {"artifact_sha256": None, "verified": False}
    artifact_sha256 = hashlib.sha256(record.encode("utf-8")).hexdigest()
    verified = True
    for line in record.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or not parts[1]:
            continue  # RECORD itself and unhashed entries
        algo, _, digest = parts[1].partition("=")
        if algo != "sha256":
            verified = False
            break
        try:
            path = dist.locate_file(parts[0])
            data = Path(str(path)).read_bytes()
        except OSError:
            verified = False
            break
        import base64

        actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        if actual.decode("ascii") != digest:
            verified = False
            break
    return {"artifact_sha256": artifact_sha256, "verified": verified}


def package_version() -> str:
    try:
        return version("prefcent")
    except PackageNotFoundError:
        from prefcent import __version__

        return __version__


def _parse_thread_var(name: str, raw: str) -> int:
    """OpenMP permits a comma-separated nesting list; the first entry is the
    initial-level count (OpenMP 5.1, section 6.2). Anything else is an error,
    raised before iteration starts."""
    first = raw.split(",")[0].strip()
    if not first.isdigit() or int(first) < 1:
        raise ValueError(f"{name}={raw!r} is not a positive integer thread count")
    return int(first)


def blas_threads() -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for name in _BLAS_VARS:
        raw = os.environ.get(name)
        out[name] = (
            None if raw is None or raw.strip() == "" else _parse_thread_var(name, raw)
        )
    return out


def blas_backend() -> str:
    """``"openblas"`` | ``"mkl"`` | ``"unknown"`` from numpy's build config."""
    try:
        import numpy as np

        cfg = np.show_config(mode="dicts")
        name = str(cfg["Build Dependencies"]["blas"]["name"]).lower()
    except Exception:
        return "unknown"
    if "openblas" in name:
        return "openblas"
    if "mkl" in name:
        return "mkl"
    return "unknown"


def effective_blas_threads(
    threads: dict[str, int | None], backend: str
) -> tuple[int | None, str | None]:
    """The variable that actually governs the detected backend.

    An unrelated variable (``MKL_NUM_THREADS`` under OpenBLAS) is not evidence.
    """
    order = {
        "openblas": ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"),
        "mkl": ("MKL_NUM_THREADS", "OMP_NUM_THREADS"),
        "unknown": ("OMP_NUM_THREADS",),
    }[backend]
    for name in order:
        if threads.get(name) is not None:
            return threads[name], name
    return None, None


def environment_block() -> dict[str, Any]:
    import numpy as np

    threads = blas_threads()
    backend = blas_backend()
    effective, source = effective_blas_threads(threads, backend)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "blas_backend": backend,
        "blas_threads": threads,
        "blas_threads_effective": effective,
        "blas_threads_source": source,
        "platform": platform.platform(),
    }


def _freeze(obj: Any) -> Any:
    """Deep-freeze: mappings → read-only proxies, sequences → tuples."""
    if isinstance(obj, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _normalize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NaN/Inf are forbidden in canonical JSON")
        return float(obj)
    if isinstance(obj, str):
        return obj
    raise TypeError(f"cannot put {type(obj)!r} in canonical JSON")


def canonical_dumps(obj: Any) -> str:
    """UTF-8 JSON, sorted keys, no whitespace, shortest-repr floats."""
    return json.dumps(
        _normalize(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint_sha256(fingerprint: Mapping[str, Any]) -> str:
    payload = canonical_dumps(fingerprint).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_reproducible(
    *,
    content_sha256: str | None,
    consumer_identity: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
    closure_identity: Mapping[str, Any] | None,
    blas_threads_known: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    trusted = bool(consumer_identity is not None and consumer_identity.get("trusted"))
    if not content_sha256 and not trusted:
        reasons.append("kernel_identity_unverified")
    if source.get("kind") == "git":
        if source.get("dirty"):
            reasons.append("git_dirty")
        if source.get("commit") is None:
            reasons.append("git_commit_null")
    elif not source.get("exact"):
        reasons.append("release_artifact_unverified")
    if provenance is not None:
        decay = provenance.get("decay")
        if isinstance(decay, Mapping) and decay.get("self_declared"):
            reasons.append("self_declared_decay")
    if closure_identity is None:
        reasons.append("closure_identity_unverified")
    elif closure_identity.get("kind") == "consumer" and not closure_identity.get(
        "trusted"
    ):
        reasons.append("closure_identity_unverified")
    if not blas_threads_known:
        reasons.append("blas_threads_unknown")
    return len(reasons) == 0, reasons


@dataclass(frozen=True, eq=False)
class RunManifest:
    schema_version: int
    fingerprint: Mapping[str, Any]
    observations: Mapping[str, Any]
    fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": _normalize(self.fingerprint),
            "observations": _normalize(self.observations),
            "fingerprint_sha256": self.fingerprint_sha256,
        }

    def to_json(self) -> str:
        return canonical_dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        """Parse, validate the schema, and **verify** the fingerprint hash."""
        data = json.loads(text)
        for key in (
            "schema_version",
            "fingerprint",
            "observations",
            "fingerprint_sha256",
        ):
            if key not in data:
                raise ValueError(f"manifest is missing {key!r}")
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported manifest schema_version {data['schema_version']!r}"
            )
        expected = fingerprint_sha256(data["fingerprint"])
        if str(data["fingerprint_sha256"]) != expected:
            raise ValueError(
                "fingerprint_sha256 does not match the fingerprint "
                f"(stored {data['fingerprint_sha256']!r}, computed {expected!r})"
            )
        return cls(
            schema_version=int(data["schema_version"]),
            fingerprint=_freeze(data["fingerprint"]),
            observations=_freeze(data["observations"]),
            fingerprint_sha256=expected,
        )


def build_manifest(
    *,
    fingerprint: dict[str, Any],
    observations: dict[str, Any],
) -> RunManifest:
    fp_hash = fingerprint_sha256(fingerprint)
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        fingerprint=_freeze(fingerprint),
        observations=_freeze(observations),
        fingerprint_sha256=fp_hash,
    )
