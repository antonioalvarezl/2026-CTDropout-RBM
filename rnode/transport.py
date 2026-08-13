"""Deterministic particle and density utilities for measure transport."""

from __future__ import annotations

import numpy as np
import torch

from .integrators import integrate_fixed_grid


def cell_area(x_grid, y_grid) -> float:
    """Cell area of a uniform Cartesian grid."""
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 2:
        raise ValueError("x_grid and y_grid must contain at least two coordinates")
    dx, dy = np.diff(x), np.diff(y)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise ValueError("density grids must be uniform")
    return float(dx[0] * dy[0])


def normalize_density(density, x_grid, y_grid) -> np.ndarray:
    """Numerically normalize a non-negative density on a fixed grid."""
    values = np.asarray(density, dtype=float)
    expected_shape = (len(y_grid), len(x_grid))
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(f"density must be finite with shape {expected_shape}")
    if np.any(values < 0):
        raise ValueError("density must be non-negative")
    mass = values.sum() * cell_area(x_grid, y_grid)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("density has no positive finite mass")
    return values / mass


def weighted_kde(
    points,
    masses,
    x_grid,
    y_grid,
    bandwidth: float,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Normalized isotropic Gaussian KDE for fixed Dirac masses."""
    density, _ = weighted_kde_with_diagnostics(
        points,
        masses,
        x_grid,
        y_grid,
        bandwidth,
        chunk_size=chunk_size,
    )
    return density


def weighted_kde_with_diagnostics(
    points,
    masses,
    x_grid,
    y_grid,
    bandwidth: float,
    *,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a normalized KDE and report mass lost outside the grid domain.

    The pre-normalization mass is the grid quadrature of the Gaussian mixture
    before its values are rescaled.  Unlike the mass of :func:`weighted_kde`,
    it is therefore an independent diagnostic of domain truncation.
    """
    locations = np.asarray(points, dtype=float)
    weights = np.asarray(masses, dtype=float)
    if locations.ndim != 2 or locations.shape[1] != 2:
        raise ValueError("points must have shape [n_particles, 2]")
    if weights.shape != (len(locations),) or np.any(weights < 0):
        raise ValueError("masses must be non-negative with one value per particle")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Dirac masses must sum to one")
    if bandwidth <= 0 or chunk_size <= 0:
        raise ValueError("bandwidth and chunk_size must be positive")
    xx, yy = np.meshgrid(np.asarray(x_grid), np.asarray(y_grid))
    evaluation = np.stack([xx.ravel(), yy.ravel()], axis=1)
    values = np.zeros(len(evaluation), dtype=float)
    scale = 1.0 / (2.0 * np.pi * bandwidth**2)
    for start in range(0, len(locations), chunk_size):
        stop = min(start + chunk_size, len(locations))
        difference = evaluation[:, None, :] - locations[None, start:stop, :]
        kernel = np.exp(-np.sum(difference * difference, axis=2) / (2 * bandwidth**2))
        values += scale * kernel.dot(weights[start:stop])
    raw_density = values.reshape(xx.shape)
    mass_before = float(raw_density.sum() * cell_area(x_grid, y_grid))
    density = normalize_density(raw_density, x_grid, y_grid)
    diagnostics = {
        "mass_before_normalization": mass_before,
        "renormalization_factor": float(1.0 / mass_before),
        "estimated_truncation_loss": float(max(0.0, 1.0 - mass_before)),
    }
    return density, diagnostics


def average_densities(densities, x_grid, y_grid) -> np.ndarray:
    """Average already-built densities, then remove numerical mass drift."""
    values = np.asarray(densities, dtype=float)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("densities must have shape [realization, y, x]")
    return normalize_density(values.mean(axis=0), x_grid, y_grid)


def l1_density_error(first, second, x_grid, y_grid) -> float:
    """Grid quadrature of the terminal density difference."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("densities must have the same shape")
    return float(np.abs(first - second).sum() * cell_area(x_grid, y_grid))


def particle_coupling_error(first, second, masses) -> float:
    r"""Weighted RMS displacement, an upper bound on ``W_2``."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    weights = np.asarray(masses, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 2:
        raise ValueError("particle arrays must share shape [n_particles, 2]")
    if (
        weights.shape != (len(first),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("masses must be finite, non-negative, and sum to one")
    return float(np.sqrt(np.sum(weights * np.sum((first - second) ** 2, axis=1))))


def transport_particles(
    model,
    points,
    masses,
    T,
    dt,
    h,
    schedule=None,
    *,
    inclusion_probs=None,
    method="rk4",
):
    """Transport positions while returning an unchanged copy of Dirac masses."""
    weights = torch.as_tensor(masses).clone()
    if (
        weights.ndim != 1
        or not bool(torch.all(torch.isfinite(weights)))
        or bool(torch.any(weights < 0))
        or not torch.isclose(weights.sum(), weights.new_tensor(1.0))
    ):
        raise ValueError("masses must be a finite, non-negative normalized vector")
    times, trajectory = integrate_fixed_grid(
        model,
        points,
        T,
        dt,
        h,
        schedule,
        inclusion_probs=inclusion_probs,
        method=method,
    )
    return times, trajectory, weights
