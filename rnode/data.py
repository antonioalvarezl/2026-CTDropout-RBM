"""
Dataset generation for rNODE experiments.

- Two-circles classification data (§5.1, §5.3)
- Initial / target densities for flow matching (§5.2)
- Mesh particles for density transport
- Evaluation grid for decision boundaries
"""

import numpy as np
import torch


def _make_circles_np(n_samples: int, factor: float, noise: float):
    """NumPy-only two-circles generator (scikit-learn compatible shape)."""
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
    inner = np.stack([np.cos(inner_angles), np.sin(inner_angles)], axis=1) * factor

    X = np.concatenate([outer, inner], axis=0)
    y = np.concatenate([
        np.zeros(n_outer, dtype=np.int64),
        np.ones(n_inner, dtype=np.int64),
    ])

    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    if noise > 0:
        X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)
    return X, y


# ── Classification data ────────────────────────────────────────────────────

def make_circles_data(n_samples: int = 100, noise: float = 0.05,
                      factor: float = 0.5, seed: int = 2):
    """Two concentric circles with labels in {-1, +1}.

    Returns:
        X: Tensor (n_samples, 2)
        y: Tensor (n_samples, 1) with values in {-1, +1}
    """
    np.random.seed(seed)
    X, y = _make_circles_np(n_samples=n_samples, factor=factor, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) * 2 - 1
    return X, y


def make_grid(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), n_points: int = 100):
    """Evaluation grid for decision-boundary visualisation.

    Returns:
        xx, yy: meshgrid arrays
        grid: Tensor (n_points^2, 2)
    """
    x = np.linspace(*xlim, n_points)
    y = np.linspace(*ylim, n_points)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return xx, yy, torch.tensor(grid, dtype=torch.float32)


# ── Flow matching data ─────────────────────────────────────────────────────

def sample_initial(n_samples: int, seed: int = None) -> np.ndarray:
    """Gaussian at (-1, -1) with covariance 0.2 I."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.multivariate_normal([-1, -1], [[0.2, 0], [0, 0.2]],
                                         n_samples)


def sample_target(n_samples: int, seed: int = None) -> np.ndarray:
    """Mixture of three Gaussians (balanced thirds)."""
    if seed is not None:
        np.random.seed(seed)
    cov1 = [[0.2, 0.05], [0.05, 0.2]]
    cov2 = [[0.05, 0], [0, 0.05]]
    n1 = n_samples // 3
    n2 = n_samples // 3
    n3 = n_samples - n1 - n2
    X1 = np.random.multivariate_normal([6, 0], cov1, n1)
    X2 = np.random.multivariate_normal([4.5, 3], cov1, n2)
    X3 = np.random.multivariate_normal([6, 2], cov2, n3)
    return np.concatenate([X1, X2, X3])


def sample_mesh_particles(n_points: int = 200):
    """Mesh particles inside unit disk centred at (-1, -1).

    Returns:
        particles: Tensor (N, 2)
        weights:   Tensor (N,) summing to 1
    """
    x = np.linspace(-2, 0, n_points)
    y = np.linspace(-2, 0, n_points)
    xx, yy = np.meshgrid(x, y)
    dist = (xx.ravel() + 1) ** 2 + (yy.ravel() + 1) ** 2
    mask = dist <= 1.0
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    dA = dx * dy
    particles = np.stack([xx.ravel()[mask], yy.ravel()[mask]], axis=1)
    weights = (1.0 - dist[mask]) * dA
    weights = weights / np.sum(weights)
    return (torch.tensor(particles, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32))
