"""Model construction, isolation, start conventions, and ``evolve``."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from prefcent._closure import Closure, Identity
from prefcent._density_penalty import DensityPenaltyV1
from prefcent._errors import KernelError, ModelDomainError, NumericalBreach
from prefcent._evolve import apply_update
from prefcent._hashing import hash_array
from prefcent._kernels import Kernel
from prefcent._landscape import Landscape
from prefcent._manifest import (
    build_manifest,
    compute_reproducible,
    environment_block,
    package_version,
    source_identity,
)
from prefcent._result import (
    Certificate,
    ControllerState,
    EvolveResult,
    EvolveStatus,
    StepInfo,
)

CRITERION = "max_step_rel_mean"
CONTROLLER_KIND = "fixed_omega"


def _readonly(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


def _as_plain_list_int(indices: np.ndarray) -> list[int]:
    return [int(i) for i in indices]


def _kernel_identity(kernel: Kernel) -> tuple[str | None, dict[str, Any] | None]:
    content = getattr(kernel, "content_sha256", None)
    consumer = getattr(kernel, "consumer_identity", None)
    if content is not None:
        content = str(content)
    if consumer is not None and not isinstance(consumer, dict):
        consumer = None
    return content, consumer


def _kernel_manifest_block(kernel: Kernel) -> dict[str, Any]:
    content, consumer = _kernel_identity(kernel)
    dtype = kernel.dtype
    dtype_str = dtype.name if hasattr(dtype, "name") else str(dtype)
    block: dict[str, Any] = {
        "content_sha256": content,
        "consumer_identity": consumer,
        "dtype": dtype_str,
        "shape": [int(kernel.shape[0]), int(kernel.shape[1])],
        "self_interaction": bool(kernel.self_interaction),
        "is_symmetric": bool(kernel.is_symmetric),
        "is_symmetric_source": getattr(kernel, "is_symmetric_source", "declared"),
        "provenance": getattr(kernel, "provenance", None),
    }
    check = getattr(kernel, "symmetry_check", None)
    if check is not None:
        block["symmetry_check"] = check
    return block


_BUILTIN_CLOSURES: dict[type, str] = {
    Identity: "identity",
    DensityPenaltyV1: "density_penalty_v1",
}


def _closure_identity(closure: Closure) -> dict[str, Any] | None:
    """Identity of the closure implementation.

    Built-in identity comes from an **exact-type registry** inside the package —
    ``type(closure) is Identity`` — never from a closure-supplied attribute, so a
    consumer closure (or an ``Identity`` subclass overriding ``apply``) cannot
    impersonate a built-in. Everything else is a consumer identity or ``None``.
    """
    from prefcent import __version__

    builtin_name = _BUILTIN_CLOSURES.get(type(closure))
    if builtin_name is not None:
        return {
            "kind": "builtin",
            "name": builtin_name,
            "package_version": __version__,
        }
    ident = getattr(closure, "identity", None)
    if ident is None:
        return None
    if not isinstance(ident, Mapping) or ident.get("kind") != "consumer":
        raise ValueError(
            "closure.identity must be None or "
            "{'kind': 'consumer', 'value', 'trusted'}; "
            "'builtin' is reserved for the package's own closures"
        )
    if "value" not in ident:
        raise ValueError("consumer closure identity needs 'value'")
    trusted = ident.get("trusted", False)
    if not isinstance(trusted, bool):
        raise TypeError("closure identity 'trusted' must be a bool (no coercion)")
    return {"kind": "consumer", "value": str(ident["value"]), "trusted": trusted}


def _kernel_match_id(fingerprint: Mapping[str, Any]) -> tuple[str, Any] | None:
    kernel = fingerprint.get("kernel")
    if not isinstance(kernel, Mapping):
        return None
    content = kernel.get("content_sha256")
    if content is not None:
        return ("content", content)
    consumer = kernel.get("consumer_identity")
    if isinstance(consumer, Mapping) and consumer.get("value") is not None:
        return ("consumer", consumer.get("value"))
    return None


def _compatible(prior: EvolveResult, fingerprint: Mapping[str, Any]) -> bool:
    prev = prior.manifest.fingerprint
    id_prev = _kernel_match_id(prev)
    id_now = _kernel_match_id(fingerprint)
    if id_prev is None or id_now is None or id_prev != id_now:
        return False
    pairs = (
        ("landscape", "capacity_sha256"),
        ("landscape", "labels_sha256"),
        ("model", "gamma"),
        ("model", "closure_name"),
        ("model", "closure_params"),
        ("model", "closure_identity"),
        ("solver", "omega"),
        ("solver", "controller"),
    )
    for block, field in pairs:
        if prev.get(block, {}).get(field) != fingerprint.get(block, {}).get(field):
            return False
    return True


def _validate_start_array(mass: np.ndarray, n: int, active: np.ndarray) -> np.ndarray:
    arr = np.asarray(mass)
    if arr.shape != (n,):
        raise ValueError(f"start must have shape {(n,)}, got {arr.shape}")
    if arr.dtype != np.float64:
        raise ValueError(f"start must be float64, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError("start must be finite")
    if bool(np.any(arr < 0.0)):
        raise ValueError("start must be ≥ 0")
    if float(arr[active].sum()) <= 0.0:
        raise ValueError("start must have a positive total on the active set")
    return arr


def _project(
    mass: np.ndarray,
    isolated: np.ndarray,
    total_capacity: np.float64,
) -> tuple[np.ndarray, float, float]:
    projected_mass = np.array(mass, dtype=np.float64, copy=True)
    dropped = (
        projected_mass[isolated].sum() if bool(isolated.any()) else np.float64(0.0)
    )
    if bool(isolated.any()):
        projected_mass[isolated] = 0.0
    active_total = projected_mass[~isolated].sum()
    scale = total_capacity / active_total
    if scale != np.float64(1.0):
        projected_mass *= scale
    if bool(isolated.any()):
        projected_mass[isolated] = 0.0
    return projected_mass, float(dropped), float(scale)


class Model:
    """Preferential-centrality model: one kernel, one landscape, one closure."""

    def __init__(
        self,
        kernel: Kernel | None = None,
        landscape: Landscape | None = None,
        *,
        kernels: Sequence[Kernel] | None = None,
        gamma: float,
        closure: Closure | None = None,
    ) -> None:
        if landscape is None:
            raise TypeError("landscape is required")
        if kernel is not None and kernels is not None:
            raise ValueError("pass kernel or kernels, not both")
        if kernels is not None:
            if len(kernels) != 1:
                raise ValueError(
                    f"v0.1 accepts exactly one kernel (got len={len(kernels)})"
                )
            kernel = kernels[0]
        if kernel is None:
            raise TypeError("kernel is required")

        n = landscape.n
        if kernel.shape != (n, n):
            raise ValueError(f"kernel shape must be {(n, n)}, got {kernel.shape}")
        if kernel.dtype != np.dtype(np.float64):
            raise ValueError(f"v0.1.0 accepts float64 kernels only, got {kernel.dtype}")
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and ≥ 0")

        self._kernel = kernel
        self._landscape = landscape
        self.gamma = float(gamma)
        self._closure: Closure = Identity() if closure is None else closure

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @property
    def landscape(self) -> Landscape:
        return self._landscape

    @property
    def closure(self) -> Closure:
        return self._closure

    def _probe_isolation(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, Certificate, int, int]:
        kernel = self._kernel
        ones = np.ones(n, dtype=np.float64)
        # The probe runs before iteration 1, so a failure inside it is
        # reported with step=0 rather than a numbered iterate.
        step = 0

        def _call(fn: Callable[[np.ndarray], np.ndarray], name: str) -> np.ndarray:
            try:
                out = fn(ones)
            except KernelError:
                raise
            except Exception as exc:
                raise KernelError(
                    f"{name} failed during isolation probe: {exc}",
                    step=step,
                ) from exc
            if not isinstance(out, np.ndarray) or out.shape != (n,):
                raise KernelError(
                    f"{name} during isolation must return shape {(n,)}",
                    step=step,
                )
            if out.dtype != np.float64:
                raise KernelError(
                    f"{name} during isolation must return float64",
                    step=step,
                )
            if not np.isfinite(out).all() or bool(np.any(out < 0.0)):
                raise KernelError(
                    f"{name} during isolation produced a non-finite or negative entry",
                    step=step,
                )
            return out

        row_sums = _call(kernel.matvec, "matvec")
        col_sums = _call(kernel.rmatvec, "rmatvec")
        row_zero = row_sums == 0.0
        col_zero = col_sums == 0.0
        isolated = row_zero & col_zero
        sink = row_zero & ~col_zero
        if bool(sink.any()):
            idx = np.flatnonzero(sink)
            raise ModelDomainError(
                "kernel has sink zones (zero row, nonzero column) at "
                f"indices {_as_plain_list_int(idx)}",
                indices=idx,
            )
        if bool(isolated.all()):
            raise ModelDomainError("all zones are isolated; there is no model")

        active = ~isolated
        excluded = float(self._landscape.capacity[isolated].sum())
        cert = Certificate(
            name="isolated_zones",
            passed=True,
            data={
                "indices": tuple(_as_plain_list_int(np.flatnonzero(isolated))),
                "count": int(isolated.sum()),
                "excluded_capacity": excluded,
            },
        )
        return isolated, active, cert, 1, 1

    def evolve(
        self,
        start: np.ndarray | EvolveResult | None = None,
        *,
        max_iter: int,
        tol: float | None = None,
        omega: float = 1.0,
        record: int | None = None,
        callback: Callable[[StepInfo], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> EvolveResult:
        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise ValueError("max_iter must be an integer ≥ 1")
        if max_iter < 1:
            raise ValueError("max_iter must be ≥ 1")
        if tol is not None and (not np.isfinite(tol) or float(tol) <= 0.0):
            raise ValueError("tol must be > 0 or None")
        if not np.isfinite(omega) or not (0.0 < float(omega) <= 1.0):
            raise ValueError("omega must be in (0, 1]")
        if record is not None and (not isinstance(record, int) or record < 1):
            raise ValueError("record must be ≥ 1 or None")

        omega_f = float(omega)
        landscape = self._landscape
        n = landscape.n
        capacity = np.array(landscape.capacity, dtype=np.float64, copy=True)
        kernel = self._kernel
        closure = self._closure
        # Environment is parsed and validated BEFORE the first step: a
        # malformed thread variable must not fail after an expensive run.
        env = environment_block()
        # Snapshot the closure's declared parameters and identity before the
        # first step: a closure that mutates its params mid-run cannot rewrite
        # its own fingerprint.
        closure_params = dict(closure.params)
        closure_identity = _closure_identity(closure)

        wall_start = time.perf_counter()
        isolated, active, iso_cert, probe_matvecs, probe_rmatvecs = (
            self._probe_isolation(n)
        )
        n_active = int(active.sum())
        if n_active == n:
            total_capacity = np.float64(capacity.sum())
        else:
            total_capacity = np.float64(capacity[active].sum())
        matvec_count = probe_matvecs
        rmatvec_count = probe_rmatvecs

        def matvec(x: np.ndarray, step: int) -> np.ndarray:
            nonlocal matvec_count
            try:
                y = kernel.matvec(x)
            except KernelError:
                raise
            except Exception as exc:
                raise KernelError(f"matvec failed: {exc}", step=step, state=x) from exc
            matvec_count += 1
            return y

        def rmatvec(x: np.ndarray, step: int) -> np.ndarray:
            nonlocal rmatvec_count
            try:
                y = kernel.rmatvec(x)
            except KernelError:
                raise
            except Exception as exc:
                raise KernelError(f"rmatvec failed: {exc}", step=step, state=x) from exc
            rmatvec_count += 1
            return y

        # Every start convention records the hash of what was supplied, plus
        # the projection onto the active set wherever one was applied.
        start_projection: dict[str, float] | None
        if start is None:
            start_convention = "capacity"
            supplied = capacity
            start_sha256 = hash_array(supplied)
            mass, dropped, scale = _project(supplied, isolated, total_capacity)
            start_projection = {
                "isolated_mass_dropped": dropped,
                "scale": scale,
            }
        elif isinstance(start, EvolveResult):
            kernel_block = _kernel_manifest_block(kernel)
            trial_fp = {
                "kernel": kernel_block,
                "landscape": {
                    "capacity_sha256": landscape.capacity_sha256,
                    "labels_sha256": landscape.labels_sha256,
                },
                "model": {
                    "gamma": self.gamma,
                    "closure_name": closure.name,
                    "closure_params": closure_params,
                    "closure_identity": closure_identity,
                },
                "solver": {
                    "omega": omega_f,
                    "controller": CONTROLLER_KIND,
                },
            }
            if _compatible(start, trial_fp):
                # A start continued from an identical model is taken verbatim;
                # admissibility is structural only. Every step renormalises, so
                # the total is already within rounding of the capacity and is
                # not re-checked to the bit.
                start_convention = "continued"
                supplied = np.array(start.mass, dtype=np.float64, copy=True)
                start_sha256 = hash_array(supplied)
                if not np.isfinite(supplied).all() or bool(np.any(supplied < 0.0)):
                    raise ValueError("continued start is not finite and ≥ 0")
                if bool(np.any(supplied[isolated] != 0.0)):
                    raise ValueError("continued start carries mass on isolated zones")
                mass = supplied
                start_projection = None
            else:
                start_convention = "continued_foreign"
                supplied = _validate_start_array(start.mass, n, active)
                start_sha256 = hash_array(np.asarray(start.mass))
                mass, dropped, scale = _project(supplied, isolated, total_capacity)
                start_projection = {
                    "isolated_mass_dropped": dropped,
                    "scale": scale,
                }
        else:
            start_convention = "custom"
            supplied = _validate_start_array(start, n, active)
            start_sha256 = hash_array(np.asarray(start))
            mass, dropped, scale = _project(supplied, isolated, total_capacity)
            start_projection = {
                "isolated_mass_dropped": dropped,
                "scale": scale,
            }

        mean_capacity = total_capacity / n_active
        trajectory_min = float(mass[active].min())
        frames: list[np.ndarray] = []
        frame_idx: list[int] = []
        if record is not None:
            frames.append(mass.copy())
            frame_idx.append(0)

        history: list[float] = []
        status = EvolveStatus.FIXED_BUDGET
        iterations = 0
        step_norm = 0.0

        for step in range(1, max_iter + 1):
            previous_mass = mass
            mass = apply_update(
                mass=mass,
                capacity=capacity,
                gamma=self.gamma,
                omega=omega_f,
                closure=closure,
                landscape=landscape,
                active=active,
                total_capacity=total_capacity,
                n_active=n_active,
                n=n,
                step=step,
                matvec=matvec,
                rmatvec=rmatvec,
            )
            delta = np.abs(mass - previous_mass)
            step_norm = float(delta[active].max() / mean_capacity)
            history.append(step_norm)
            iterations = step
            trajectory_min = min(trajectory_min, float(mass[active].min()))
            stats_now = {
                "trajectory_min": trajectory_min,
                "final_min": float(mass[active].min()),
                "final_max_density": float((mass[active] / capacity[active]).max()),
            }
            if record is not None and step % record == 0:
                frames.append(mass.copy())
                frame_idx.append(step)
            if callback is not None:
                info = StepInfo(
                    iteration=step,
                    step_norm=step_norm,
                    mass=_readonly(mass),
                    stats_so_far=MappingProxyType(dict(stats_now)),
                )
                callback(info)
            if tol is not None and step_norm <= float(tol):
                status = EvolveStatus.CONVERGED
                break
            if should_stop is not None and should_stop():
                status = EvolveStatus.CANCELLED
                break
        else:
            status = (
                EvolveStatus.FIXED_BUDGET
                if tol is None
                else EvolveStatus.BUDGET_EXHAUSTED
            )

        attraction_final = self.gamma * mass + capacity
        # The final potential is computed after the loop, so a failure here is
        # reported with step=0 rather than a numbered iterate.
        try:
            potential = kernel.matvec(attraction_final)
        except KernelError:
            raise
        except Exception as exc:
            raise KernelError(
                f"matvec failed on final potential: {exc}",
                step=0,
                state=mass,
            ) from exc
        matvec_count += 1
        if (
            not isinstance(potential, np.ndarray)
            or potential.shape != (n,)
            or potential.dtype != np.float64
            or not np.isfinite(potential).all()
            or bool(np.any(potential < 0.0))
        ):
            raise KernelError(
                "final potential is not a finite nonnegative float64 vector",
                step=0,
                state=mass,
            )

        # The raw balance update AT the final state, for the closure's margin
        # certificate: one rmatvec, counted like any other.
        share_final = np.zeros(n, dtype=np.float64)
        share_final[active] = mass[active] / potential[active]
        try:
            raw_final = kernel.rmatvec(share_final)
        except KernelError:
            raise
        except Exception as exc:
            raise KernelError(
                f"rmatvec failed on final inflow: {exc}", step=0, state=mass
            ) from exc
        rmatvec_count += 1
        inflow_final = attraction_final * np.asarray(raw_final, dtype=np.float64)
        inflow_final[~active] = 0.0
        try:
            domain_cert = closure.domain_check(
                _readonly(mass), _readonly(inflow_final), landscape, _readonly(active)
            )
        except NumericalBreach as exc:
            if exc.step is None:
                # Final diagnostics use step=0, like the final potential check.
                raise type(exc)(
                    str(exc.args[0]) if exc.args else "closure certificate breach",
                    step=0,
                    state=mass if exc.state is None else exc.state,
                ) from exc
            raise
        closure_key = f"closure:{closure.name}"
        certificates: dict[str, Certificate] = {
            "isolated_zones": iso_cert,
            closure_key: domain_cert,
        }

        if record is not None:
            if frame_idx[-1] != iterations:
                frames.append(mass.copy())
                frame_idx.append(iterations)
            trajectory = np.stack(frames, axis=0)
            iteration_index = np.asarray(frame_idx, dtype=np.int64)
            trajectory.flags.writeable = False
            iteration_index.flags.writeable = False
        else:
            trajectory = None
            iteration_index = None

        stats = {
            "trajectory_min": trajectory_min,
            "final_min": float(mass[active].min()),
            "final_max_density": float((mass[active] / capacity[active]).max()),
        }
        step_norm_history = np.asarray(history, dtype=np.float64)
        step_norm_history.flags.writeable = False
        mass_out = mass
        mass_out.flags.writeable = False
        potential.flags.writeable = False
        active_out = np.array(active, copy=True)
        active_out.flags.writeable = False

        source = source_identity()
        kernel_block = _kernel_manifest_block(kernel)
        reproducible, reasons = compute_reproducible(
            content_sha256=kernel_block["content_sha256"],
            consumer_identity=kernel_block["consumer_identity"],
            source=source,
            provenance=kernel_block["provenance"],
            closure_identity=closure_identity,
            blas_threads_known=env["blas_threads_effective"] is not None,
        )
        fingerprint: dict[str, Any] = {
            "package_version": package_version(),
            "source": source,
            "reproducible": reproducible,
            "reproducible_reasons": reasons,
            "kernel": kernel_block,
            "landscape": {
                "n": n,
                "total_capacity": landscape.total_capacity,
                "capacity_sha256": landscape.capacity_sha256,
                "labels_sha256": landscape.labels_sha256,
            },
            "model": {
                "gamma": self.gamma,
                "closure_name": closure.name,
                "closure_params": closure_params,
                "closure_identity": closure_identity,
            },
            "solver": {
                "verb": "evolve",
                "omega": omega_f,
                "start_convention": start_convention,
                "start_sha256": start_sha256,
                "start_projection": start_projection,
                "criterion": CRITERION,
                "tol": None if tol is None else float(tol),
                "max_iter": max_iter,
                "controller": CONTROLLER_KIND,
            },
            "legacy_policies": {},
            "environment": env,
            "seeds": None,
        }
        wall = time.perf_counter() - wall_start
        cert_obs = {
            name: {"passed": cert.passed, "data": dict(cert.data)}
            for name, cert in certificates.items()
        }
        observations: dict[str, Any] = {
            "status": status.value,
            "iterations": iterations,
            "final_step_norm": step_norm,
            "matvec_count": matvec_count,
            "rmatvec_count": rmatvec_count,
            "certificates": cert_obs,
            "stats": dict(stats),
            "wall_time_s": wall,
            "result_sha256": hash_array(mass_out),
        }
        manifest = build_manifest(fingerprint=fingerprint, observations=observations)

        return EvolveResult(
            mass=mass_out,
            potential=potential,
            landscape=landscape,
            active=active_out,
            status=status,
            iterations=iterations,
            final_step_norm=step_norm,
            criterion=CRITERION,
            tol=None if tol is None else float(tol),
            max_iter=max_iter,
            omega=omega_f,
            step_norm_history=step_norm_history,
            matvec_count=matvec_count,
            rmatvec_count=rmatvec_count,
            stats=MappingProxyType(stats),
            certificates=MappingProxyType(certificates),
            trajectory=trajectory,
            iteration_index=iteration_index,
            controller_state=ControllerState(kind=CONTROLLER_KIND),
            manifest=manifest,
            start_convention=start_convention,
        )
