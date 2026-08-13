"""Objective and target-based metrics from the current paper."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .data import labels_to_targets


class ObjectiveTerms(NamedTuple):
    """Unweighted paper terms plus their weighted total."""

    terminal: Tensor
    running: Tensor
    control: Tensor
    total: Tensor


def trapezoidal_integral(values: Tensor, times: Tensor) -> Tensor:
    """Integrate values along their leading time axis."""
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if values.shape[0] != times.numel():
        raise ValueError("values and times must have the same leading length")
    if times.numel() < 2:
        raise ValueError("trapezoidal integration requires at least two times")
    if not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("times must be strictly increasing")
    return torch.trapezoid(values, times, dim=0)


def control_regularization(model, times: Tensor) -> Tensor:
    r"""Compute ``integral (|A(t)|^2 + |b(t)|^2 + |W(t)|^2) dt``.

    This evaluates the generated control itself, never the parameters of its
    hyper-network.
    """
    try:
        control = model.control_parameters
    except AttributeError as error:
        raise TypeError(
            "model must expose control_parameters(t) returning (A, b, W)"
        ) from error
    energy = []
    for time in times:
        tensors = control(time)
        if not isinstance(tensors, (tuple, list)) or len(tensors) != 3:
            raise TypeError("control_parameters(t) must return exactly (A, b, W)")
        energy.append(sum(component.square().sum() for component in tensors))
    return trapezoidal_integral(torch.stack(energy), times)


def paper_objective(
    trajectory: Tensor,
    times: Tensor,
    targets: Tensor,
    model=None,
    *,
    alpha: float = 0.01,
    beta: float = 0.5,
    control_times: Tensor | None = None,
) -> ObjectiveTerms:
    r"""Evaluate the paper objective using trapezoidal time integration.

    ``terminal``, ``running`` and ``control`` in the result are the raw terms;
    ``total = terminal + beta * running + alpha/2 * control``.
    A random-batch run passes its random trajectory but the same underlying
    control model, so only the trajectory-dependent terms differ.
    """
    if trajectory.ndim != 3:
        raise ValueError("trajectory must have shape [n_times, n_data, d]")
    if times.ndim != 1 or trajectory.shape[0] != times.numel():
        raise ValueError("times must match the trajectory's leading dimension")
    if targets.shape != trajectory.shape[1:]:
        raise ValueError(
            f"targets must have shape {tuple(trajectory.shape[1:])}, "
            f"got {tuple(targets.shape)}"
        )
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    targets = targets.to(dtype=trajectory.dtype, device=trajectory.device)
    squared_error = (trajectory - targets.unsqueeze(0)).square()
    terminal = squared_error[-1].sum()
    running_integrand = squared_error.sum(dim=(1, 2))
    running = trapezoidal_integral(running_integrand, times)

    if model is None:
        if alpha != 0:
            raise ValueError("model is required when alpha is non-zero")
        control_term = trajectory.new_zeros(())
    else:
        quadrature_times = times if control_times is None else control_times
        control_term = control_regularization(model, quadrature_times)
    total = terminal + beta * running + (alpha / 2.0) * control_term
    return ObjectiveTerms(terminal, running, control_term, total)


def nearest_target_predictions(predictions: Tensor) -> Tensor:
    """Return class labels by nearest paper target, not a coordinate sign."""
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions must have shape [n_data, 2]")
    prototypes = predictions.new_tensor([[-1.0, 0.0], [0.0, 1.0]])
    class_index = torch.cdist(predictions, prototypes).argmin(dim=1)
    return class_index * 2 - 1


def nearest_target_accuracy(predictions: Tensor, labels_or_targets: Tensor) -> float:
    """Classification accuracy using the closest of ``(-1,0)`` and ``(0,1)``."""
    predicted_labels = nearest_target_predictions(predictions)
    if labels_or_targets.ndim == 2 and labels_or_targets.shape[1] == 2:
        true_labels = nearest_target_predictions(labels_or_targets)
    else:
        true_labels = labels_or_targets.reshape(-1).to(
            dtype=predicted_labels.dtype, device=predicted_labels.device
        )
        # Reuse the target validation for clear errors on unexpected classes.
        labels_to_targets(true_labels)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("labels/targets and predictions have different sample counts")
    return (predicted_labels == true_labels).float().mean().item()
