"""Training routines for one shared control under fresh random schedules."""

from __future__ import annotations

import time

import numpy as np
import torch

from .batches import BatchScheme, sample_batch_sequence
from .integrators import integrate_fixed_grid
from .objectives import paper_objective


def _integer_ratio(numerator: float, denominator: float) -> int:
    ratio = numerator / denominator
    rounded = int(round(ratio))
    if rounded <= 0 or not np.isclose(ratio, rounded):
        raise ValueError("integration ratios must be positive integers")
    return rounded


def _schedule_evaluations(schedule, steps_per_interval: int, stages: int) -> int:
    return stages * steps_per_interval * sum(len(batch) for batch in schedule)


def train_expected_objective(
    model,
    features,
    targets,
    *,
    scheme: BatchScheme,
    h: float,
    dt: float,
    T: float,
    epochs: int,
    m_train: int,
    learning_rate: float,
    alpha: float,
    beta: float,
    seed: int,
    method: str = "rk4",
    control_dt: float | None = None,
    progress_callback=None,
    progress_every: int | None = None,
) -> dict:
    """Minimize a Monte Carlo estimate of the expected RBM objective.

    One model is retained throughout. Every optimizer step samples fresh
    schedules, forms all losses with the shared parameters, averages them,
    calls ``backward`` once, updates once, and projects once.
    """
    if epochs <= 0 or m_train <= 0:
        raise ValueError("epochs and m_train must be positive")
    if scheme.p != model.hidden_dim:
        raise ValueError("scheme and model hidden dimensions differ")
    intervals = _integer_ratio(T, h)
    steps_per_interval = _integer_ratio(h, dt)
    stages = 4 if method == "rk4" else 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    control_steps = _integer_ratio(T, dt if control_dt is None else control_dt)
    control_times = torch.linspace(
        0.0,
        T,
        control_steps + 1,
        dtype=features.dtype,
        device=features.device,
    )
    rng = np.random.default_rng(seed)
    history, all_schedule_seeds = [], []
    neuron_evaluations = 0
    started = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_seeds = rng.integers(0, 2**32 - 1, m_train, dtype=np.uint32)
        losses = []
        for schedule_seed in epoch_seeds:
            schedule = sample_batch_sequence(
                scheme, intervals, np.random.default_rng(int(schedule_seed))
            )
            times, trajectory = integrate_fixed_grid(
                model,
                features,
                T,
                dt,
                h,
                schedule,
                inclusion_probs=scheme.inclusion_probs,
                method=method,
            )
            terms = paper_objective(
                trajectory,
                times,
                targets,
                model,
                alpha=alpha,
                beta=beta,
                control_times=control_times,
            )
            losses.append(terms.total / features.shape[0])
            neuron_evaluations += _schedule_evaluations(
                schedule, steps_per_interval, stages
            )
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()
        model.project_parameters_()
        all_schedule_seeds.append(epoch_seeds.tolist())
        history.append(
            {
                "epoch": epoch + 1,
                "expected_objective_estimate": float(mean_loss.detach().cpu()),
                "member_min": float(torch.stack(losses).min().detach().cpu()),
                "member_max": float(torch.stack(losses).max().detach().cpu()),
            }
        )
        if progress_callback is not None:
            every = progress_every or max(1, epochs // 20)
            if (epoch + 1) % every == 0 or epoch + 1 == epochs:
                progress_callback(epoch + 1, epochs, history[-1])

    wall_seconds = time.perf_counter() - started
    return {
        "model": model,
        "history": history,
        "schedule_seeds": all_schedule_seeds,
        "wall_seconds": wall_seconds,
        "optimizer_steps": epochs,
        "backward_calls": epochs,
        "neuron_evaluations": neuron_evaluations,
        "neuron_data_evaluations": neuron_evaluations * features.shape[0],
    }


def train_full_objective(
    model,
    features,
    targets,
    *,
    dt: float,
    T: float,
    epochs: int,
    learning_rate: float,
    alpha: float,
    beta: float,
    method: str = "rk4",
    control_dt: float | None = None,
    progress_callback=None,
    progress_every: int | None = None,
) -> dict:
    """Train the deterministic full objective with one update per epoch."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    stages = 4 if method == "rk4" else 1
    n_steps = _integer_ratio(T, dt)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    control_steps = _integer_ratio(T, dt if control_dt is None else control_dt)
    control_times = torch.linspace(
        0.0,
        T,
        control_steps + 1,
        dtype=features.dtype,
        device=features.device,
    )
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        times, trajectory = integrate_fixed_grid(
            model, features, T, dt, T, method=method
        )
        terms = paper_objective(
            trajectory,
            times,
            targets,
            model,
            alpha=alpha,
            beta=beta,
            control_times=control_times,
        )
        loss = terms.total / features.shape[0]
        loss.backward()
        optimizer.step()
        model.project_parameters_()
        history.append(
            {
                "epoch": epoch + 1,
                "expected_objective_estimate": float(loss.detach().cpu()),
            }
        )
        if progress_callback is not None:
            every = progress_every or max(1, epochs // 20)
            if (epoch + 1) % every == 0 or epoch + 1 == epochs:
                progress_callback(epoch + 1, epochs, history[-1])
    neuron_evaluations = epochs * n_steps * stages * model.hidden_dim
    return {
        "model": model,
        "history": history,
        "schedule_seeds": [],
        "wall_seconds": time.perf_counter() - started,
        "optimizer_steps": epochs,
        "backward_calls": epochs,
        "neuron_evaluations": neuron_evaluations,
        "neuron_data_evaluations": neuron_evaluations * features.shape[0],
    }
