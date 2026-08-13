"""Differentiable fixed-grid integrators aligned with RBM switching times."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def _integer_ratio(numerator: float, denominator: float, name: str, tol: float) -> int:
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"{name} requires positive values")
    ratio = numerator / denominator
    rounded = int(round(ratio))
    if rounded <= 0 or not np.isclose(ratio, rounded, rtol=tol, atol=tol):
        raise ValueError(f"{name} must be an integer within tolerance {tol}")
    return rounded


def integrate_fixed_grid(
    model,
    x0: Tensor,
    T: float,
    dt: float,
    h: float,
    schedule: Sequence | None = None,
    *,
    inclusion_probs=None,
    method: str = "rk4",
    t0: float = 0.0,
    tol: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """Integrate a full or random-batch field on an aligned fixed grid.

    Args:
        model: Object implementing ``forward(t, x)`` and, for an RBM run,
            ``forward_batch(t, x, indices, inclusion_probs)``.
        x0: Initial state, normally shaped ``[n_data, d]``.
        T: Integration duration.
        dt: Numerical step size.
        h: Switching interval.
        schedule: Exactly one neuron-index batch per switching interval.
            ``None`` selects the full field.
        inclusion_probs: Complete neuron-wise inclusion-probability vector for
            an RBM run.
        method: ``"euler"`` or ``"rk4"``.

    Returns:
        ``(times, trajectory)``, including both initial and terminal states.

    The active batch is captured once per numerical step and reused by every
    Runge--Kutta stage, including ``k4`` at a switching boundary.
    """
    if not isinstance(x0, Tensor):
        raise TypeError("x0 must be a torch.Tensor")
    duration = float(T)
    step_size = float(dt)
    switch_size = float(h)
    n_intervals = _integer_ratio(duration, switch_size, "T / h", tol)
    steps_per_interval = _integer_ratio(switch_size, step_size, "h / dt", tol)
    n_steps = n_intervals * steps_per_interval

    method = method.lower()
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")

    if schedule is not None:
        if len(schedule) != n_intervals:
            raise ValueError(
                f"schedule must contain {n_intervals} batches, got {len(schedule)}"
            )
        if inclusion_probs is None:
            inclusion_probs = getattr(model, "inclusion_probs", None)
        if inclusion_probs is None:
            raise ValueError("inclusion_probs are required for a batch schedule")
        if not hasattr(model, "forward_batch"):
            raise TypeError("model must implement forward_batch for an RBM run")

    times = torch.linspace(
        float(t0),
        float(t0) + duration,
        n_steps + 1,
        dtype=x0.dtype,
        device=x0.device,
    )
    states = [x0]
    state = x0

    for step in range(n_steps):
        time = times[step]
        numerical_dt = times[step + 1] - time
        if schedule is None:
            field = model
        else:
            # One lookup per step.  Every stage below closes over this batch.
            batch = schedule[step // steps_per_interval]

            def field(stage_time, stage_state, active_batch=batch):
                return model.forward_batch(
                    stage_time,
                    stage_state,
                    active_batch,
                    inclusion_probs,
                )

        if method == "euler":
            state = state + numerical_dt * field(time, state)
        else:
            half_dt = numerical_dt / 2
            k1 = field(time, state)
            k2 = field(time + half_dt, state + half_dt * k1)
            k3 = field(time + half_dt, state + half_dt * k2)
            # Crucially, ``field`` still closes over this step's active batch.
            k4 = field(time + numerical_dt, state + numerical_dt * k3)
            state = state + (numerical_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(state)

    return times, torch.stack(states, dim=0)


def integrate_masked_ensemble(
    model,
    x0: Tensor,
    T: float,
    dt: float,
    h: float,
    masks,
    inclusion_probs,
    *,
    method: str = "rk4",
    t0: float = 0.0,
    tol: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """Vectorised aligned integration for many independent schedules.

    ``masks`` has shape ``[ensemble, T/h, p]``.  The returned trajectory has
    shape ``[time, ensemble, n_data, d]``.  This is mathematically equivalent
    to repeated :func:`integrate_fixed_grid` calls but reuses each evaluation
    of the shared time-dependent control.
    """
    if not isinstance(x0, Tensor) or x0.ndim not in {2, 3}:
        raise ValueError("x0 must have shape [n_data, d] or [ensemble, n_data, d]")
    duration = float(T)
    n_intervals = _integer_ratio(duration, float(h), "T / h", tol)
    steps_per_interval = _integer_ratio(float(h), float(dt), "h / dt", tol)
    n_steps = n_intervals * steps_per_interval
    method = method.lower()
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")

    masks = torch.as_tensor(masks, dtype=x0.dtype, device=x0.device)
    if masks.ndim != 3 or masks.shape[1] != n_intervals:
        raise ValueError("masks must have shape [ensemble, T/h, p]")
    ensemble_size, _, p = masks.shape
    if isinstance(inclusion_probs, Tensor):
        pi = inclusion_probs.to(dtype=x0.dtype, device=x0.device)
    else:
        pi = torch.tensor(
            np.asarray(inclusion_probs).copy(), dtype=x0.dtype, device=x0.device
        )
    if pi.shape != (p,) or not bool(torch.all((pi > 0) & (pi <= 1))):
        raise ValueError("inclusion_probs must have shape [p] with values in (0, 1]")
    if x0.ndim == 2:
        state = x0.unsqueeze(0).expand(ensemble_size, -1, -1).clone()
    elif x0.shape[0] == ensemble_size:
        state = x0
    else:
        raise ValueError("x0 ensemble dimension does not match masks")

    times = torch.linspace(
        float(t0),
        float(t0) + duration,
        n_steps + 1,
        dtype=x0.dtype,
        device=x0.device,
    )
    states = [state]

    def masked_field(stage_time, stage_state, active_masks):
        n_data, dimension = stage_state.shape[1:]
        contributions = model.neuron_contributions(
            stage_time, stage_state.reshape(ensemble_size * n_data, dimension)
        ).reshape(ensemble_size, n_data, p, dimension)
        scale = active_masks / pi[None, :]
        return (contributions * scale[:, None, :, None]).sum(dim=2)

    for step in range(n_steps):
        time = times[step]
        numerical_dt = times[step + 1] - time
        active_masks = masks[:, step // steps_per_interval]
        if method == "euler":
            state = state + numerical_dt * masked_field(time, state, active_masks)
        else:
            half_dt = numerical_dt / 2
            k1 = masked_field(time, state, active_masks)
            k2 = masked_field(time + half_dt, state + half_dt * k1, active_masks)
            k3 = masked_field(time + half_dt, state + half_dt * k2, active_masks)
            k4 = masked_field(
                time + numerical_dt, state + numerical_dt * k3, active_masks
            )
            state = state + (numerical_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(state)
    return times, torch.stack(states, dim=0)
