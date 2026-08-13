"""Datasets and probability measures for the paper experiments."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


INITIAL_CENTER = np.array([-1.0, -1.0])
TARGET_MIXTURE_WEIGHTS = np.full(3, 1.0 / 3.0)
TARGET_MIXTURE_CENTERS = np.array([[6.0, 0.0], [4.5, 3.0], [6.0, 2.0]], dtype=float)
TARGET_MIXTURE_COVARIANCES = np.array(
    [
        [[0.2, 0.05], [0.05, 0.2]],
        [[0.2, 0.05], [0.05, 0.2]],
        [[0.05, 0.0], [0.0, 0.05]],
    ],
    dtype=float,
)


def _rng(
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.random.Generator:
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    # ``rng`` takes precedence so old ``seed=...`` call sites and new explicit
    # generator call sites can coexist without touching global state.
    return rng if rng is not None else np.random.default_rng(seed)


def _make_circles_np(
    n_samples: int,
    factor: float,
    noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy-only two-circles generator."""
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not 0 < factor < 1:
        raise ValueError("factor must satisfy 0 < factor < 1")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    outer_angles = np.linspace(0.0, 2.0 * np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0.0, 2.0 * np.pi, n_inner, endpoint=False)
    outer = np.stack([np.cos(outer_angles), np.sin(outer_angles)], axis=1)
    inner = factor * np.stack([np.cos(inner_angles), np.sin(inner_angles)], axis=1)

    features = np.concatenate([outer, inner], axis=0)
    zero_one_labels = np.concatenate(
        [np.zeros(n_outer, dtype=np.int64), np.ones(n_inner, dtype=np.int64)]
    )
    permutation = rng.permutation(n_samples)
    features = features[permutation]
    zero_one_labels = zero_one_labels[permutation]
    if noise > 0:
        features = features + rng.normal(0.0, noise, size=features.shape)
    return features, zero_one_labels


def labels_to_targets(labels: Tensor) -> Tensor:
    """Map class ``-1`` to ``(-1, 0)`` and class ``+1`` to ``(0, 1)``."""
    flat_labels = labels.reshape(-1)
    if not bool(torch.all((flat_labels == -1) | (flat_labels == 1))):
        raise ValueError("labels must only contain -1 and +1")
    targets = torch.zeros(
        (flat_labels.numel(), 2),
        dtype=flat_labels.dtype,
        device=flat_labels.device,
    )
    targets[flat_labels == -1, 0] = -1.0
    targets[flat_labels == 1, 1] = 1.0
    return targets


def make_circles_with_targets(
    n_samples: int = 100,
    noise: float = 0.05,
    factor: float = 0.5,
    seed: int | None = 2,
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return features, ``{-1,+1}`` labels, and paper targets in two dimensions."""
    features, labels01 = _make_circles_np(n_samples, factor, noise, _rng(rng, seed))
    X = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels01, dtype=torch.float32).unsqueeze(1) * 2 - 1
    return X, labels, labels_to_targets(labels)


def initial_density(points: np.ndarray) -> np.ndarray:
    r"""Normalised compact density ``2/pi * (1 - |x-c|^2)_+``."""
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 2:
        raise ValueError("points must have final dimension two")
    radius_squared = np.sum((points - INITIAL_CENTER) ** 2, axis=-1)
    return (2.0 / np.pi) * np.maximum(1.0 - radius_squared, 0.0)


def sample_initial_compact(
    n_samples: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample the compact initial density exactly by inverse radial CDF."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    local_rng = _rng(rng, seed)
    uniform_radius = local_rng.random(n_samples)
    radius = np.sqrt(1.0 - np.sqrt(1.0 - uniform_radius))
    angle = local_rng.uniform(0.0, 2.0 * np.pi, n_samples)
    offsets = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return INITIAL_CENTER + offsets


def sample_target(
    n_samples: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample the paper's three-Gaussian mixture with weights exactly ``1/3``."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    local_rng = _rng(rng, seed)
    components = local_rng.choice(3, size=n_samples, p=TARGET_MIXTURE_WEIGHTS)
    samples = np.empty((n_samples, 2), dtype=float)
    for component in range(3):
        mask = components == component
        samples[mask] = local_rng.multivariate_normal(
            TARGET_MIXTURE_CENTERS[component],
            TARGET_MIXTURE_COVARIANCES[component],
            int(mask.sum()),
        )
    return samples


def target_density(points: np.ndarray) -> np.ndarray:
    """Evaluate the balanced target Gaussian-mixture density."""
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 2:
        raise ValueError("points must have final dimension two")
    density = np.zeros(points.shape[:-1], dtype=float)
    for weight, center, covariance in zip(
        TARGET_MIXTURE_WEIGHTS,
        TARGET_MIXTURE_CENTERS,
        TARGET_MIXTURE_COVARIANCES,
    ):
        difference = points - center
        exponent = np.einsum(
            "...i,ij,...j->...", difference, np.linalg.inv(covariance), difference
        )
        normalizer = 2.0 * np.pi * np.sqrt(np.linalg.det(covariance))
        density += weight * np.exp(-0.5 * exponent) / normalizer
    return density


def initial_density_quadrature(n_points: int = 200) -> tuple[Tensor, Tensor]:
    """Return a Cartesian quadrature cloud and normalised probability masses."""
    if n_points < 2:
        raise ValueError("n_points must be at least two")
    coordinates = np.linspace(-2.0, 0.0, n_points)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    values = initial_density(points)
    mask = values > 0.0
    points = points[mask]
    # Equal cell areas cancel when normalising the discrete probability masses.
    masses = values[mask]
    total_mass = masses.sum()
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("quadrature grid contains no positive mass")
    masses = masses / total_mass
    return (
        torch.tensor(points, dtype=torch.float32),
        torch.tensor(masses, dtype=torch.float64),
    )
