"""Characteristic-based utilities for random-batch measure transport."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"{name} requires positive values")
    ratio = numerator / denominator
    rounded = int(round(ratio))
    if rounded <= 0 or not np.isclose(ratio, rounded, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name} must be an integer")
    return rounded


def _validate_masses(masses, n_points: int) -> np.ndarray:
    weights = np.asarray(masses, dtype=float)
    if (
        weights.shape != (n_points,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("masses must be finite, non-negative, and sum to one")
    return weights


def coupling_squared(first, second, masses) -> float:
    """Weighted squared synchronous-coupling displacement."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("point arrays must share shape [n_points, d]")
    weights = _validate_masses(masses, len(first))
    return float(np.sum(weights * np.sum((first - second) ** 2, axis=1)))


def coupling_rms(first, second, masses) -> float:
    r"""Weighted RMS displacement, hence an upper bound on discrete ``W_2``."""
    return float(np.sqrt(coupling_squared(first, second, masses)))


def _validate_schedule(schedule: Sequence | None, n_intervals: int):
    if schedule is None:
        return None
    if len(schedule) != n_intervals:
        raise ValueError(
            f"schedule must contain {n_intervals} batches, got {len(schedule)}"
        )
    return schedule


def _rhs(model, t, x, batch, inclusion_probs, log_sign: float):
    if batch is None:
        velocity = model(t, x)
        divergence = model.divergence(t, x)
    else:
        velocity = model.forward_batch(t, x, batch, inclusion_probs)
        divergence = model.divergence_batch(t, x, batch, inclusion_probs)
    return velocity, log_sign * divergence


@torch.no_grad()
def integrate_characteristics(
    model,
    points: Tensor,
    T: float,
    dt: float,
    h: float,
    schedule: Sequence | None = None,
    *,
    inclusion_probs=None,
    direction: str = "forward",
    track_log_density: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Integrate characteristics without storing their full time history.

    ``direction='forward'`` maps time 0 to time ``T``.  If log density is
    tracked, the second return value is ``-integral div F`` along each path.

    ``direction='backward'`` starts from terminal Eulerian points and maps them
    to time 0.  The same second return value is again ``-integral_0^T div F``
    along the recovered forward characteristic.  For a random flow, the exact
    same schedule must therefore be supplied in forward order; this routine
    reverses the switching intervals internally.
    """
    if not isinstance(points, Tensor) or points.ndim != 2:
        raise TypeError("points must be a two-dimensional torch.Tensor")
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'")
    if track_log_density and (
        not hasattr(model, "divergence") or not hasattr(model, "divergence_batch")
    ):
        raise TypeError("model must implement full and batch divergences")

    n_intervals = _integer_ratio(float(T), float(h), "T / h")
    steps_per_interval = _integer_ratio(float(h), float(dt), "h / dt")
    n_steps = n_intervals * steps_per_interval
    schedule = _validate_schedule(schedule, n_intervals)
    if schedule is not None and inclusion_probs is None:
        raise ValueError("inclusion_probs are required for a random schedule")

    state = points.clone()
    log_increment = state.new_zeros(state.shape[0]) if track_log_density else None
    step_size = float(dt) if direction == "forward" else -float(dt)
    # q'=-div forward; q'=+div when integrating backward in physical time.
    log_sign = -1.0 if direction == "forward" else 1.0

    for step in range(n_steps):
        if direction == "forward":
            t = step * float(dt)
            interval = step // steps_per_interval
        else:
            t = float(T) - step * float(dt)
            interval = n_intervals - 1 - step // steps_per_interval
        batch = None if schedule is None else schedule[interval]
        half = step_size / 2.0

        if track_log_density:
            k1x, k1q = _rhs(model, t, state, batch, inclusion_probs, log_sign)
            k2x, k2q = _rhs(
                model,
                t + half,
                state + half * k1x,
                batch,
                inclusion_probs,
                log_sign,
            )
            k3x, k3q = _rhs(
                model,
                t + half,
                state + half * k2x,
                batch,
                inclusion_probs,
                log_sign,
            )
            k4x, k4q = _rhs(
                model,
                t + step_size,
                state + step_size * k3x,
                batch,
                inclusion_probs,
                log_sign,
            )
            state = state + (step_size / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            log_increment = log_increment + (step_size / 6.0) * (
                k1q + 2 * k2q + 2 * k3q + k4q
            )
        else:
            if batch is None:
                field = model
            else:
                field = lambda time, x, active=batch: model.forward_batch(
                    time, x, active, inclusion_probs
                )
            k1 = field(t, state)
            k2 = field(t + half, state + half * k1)
            k3 = field(t + half, state + half * k2)
            k4 = field(t + step_size, state + step_size * k3)
            state = state + (step_size / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return state, log_increment


def terminal_log_density_from_forward(initial_density, initial_points, log_increment) -> np.ndarray:
    """Evaluate terminal log-density on points transported from time zero."""
    rho0 = np.asarray(initial_density(np.asarray(initial_points, dtype=float)), dtype=float)
    increment = np.asarray(log_increment, dtype=float)
    if rho0.shape != increment.shape:
        raise ValueError("initial density and log increment have incompatible shapes")
    result = np.full_like(rho0, -np.inf, dtype=float)
    positive = rho0 > 0
    result[positive] = np.log(rho0[positive]) + increment[positive]
    return result


def terminal_log_density_from_backward(initial_density, preimages, log_increment) -> np.ndarray:
    """Evaluate terminal log-density after backward characteristic integration."""
    return terminal_log_density_from_forward(initial_density, preimages, log_increment)


def l1_from_full_change_of_variables(
    full_log_density: np.ndarray,
    random_log_density: np.ndarray,
    masses,
) -> float:
    r"""Estimate ``||rho-rho_hat||_1`` on full-flow quadrature points.

    Exact full and random flows preserve the same unit mass, hence
    ``int (rho-rho_hat) = 0`` and ``||rho-rho_hat||_1 = 2 int (rho-rho_hat)_+``.
    Substituting ``x=Phi_T(y)`` on ``{rho>0}`` gives
    ``2 E_{rho_0}[(1-rho_hat(Phi_T(y))/rho(Phi_T(y)))_+]``.
    This also accounts for random mass outside the full support. Equality of
    exact masses is an assumption of this estimator, not a numerical mass
    check: validate density mass and quadrature separately. No renormalization
    is performed here.
    """
    full_log = np.asarray(full_log_density, dtype=float)
    random_log = np.asarray(random_log_density, dtype=float)
    if full_log.shape != random_log.shape or full_log.ndim != 1:
        raise ValueError("log densities must be one-dimensional and have equal shape")
    weights = _validate_masses(masses, len(full_log))
    if not np.all(np.isfinite(full_log)):
        raise ValueError("full density must be positive on quadrature points")

    if np.any(np.isnan(random_log) | np.isposinf(random_log)):
        raise ValueError("random log density must be finite or -inf (zero density)")
    deficit = np.zeros_like(full_log)
    smaller = random_log < full_log
    # Only nonpositive exponents are evaluated. expm1 preserves accuracy near
    # equal densities; -inf gives deficit=1. Extreme negative differences may
    # underflow to -inf during subtraction, which has the same limiting value.
    with np.errstate(over="ignore", under="ignore"):
        deficit[smaller] = -np.expm1(random_log[smaller] - full_log[smaller])
    return float(2.0 * np.sum(weights * deficit))
