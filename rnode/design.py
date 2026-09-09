"""Sampling-variance identities used by the paper experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .batches import BatchScheme


def trapezoidal_weights(times: Tensor) -> Tensor:
    """Return weights for trapezoidal integration on a one-dimensional grid."""
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be one-dimensional with at least two entries")
    differences = times[1:] - times[:-1]
    if not bool(torch.all(differences > 0)):
        raise ValueError("times must be strictly increasing")

    weights = torch.zeros_like(times)
    weights[0] = differences[0] / 2
    weights[-1] = differences[-1] / 2
    if times.numel() > 2:
        weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights


def neuron_contributions_along_trajectory(
    model,
    times: Tensor,
    trajectory: Tensor,
) -> Tensor:
    """Evaluate the neuron fields along a full trajectory.

    Returns a tensor with shape ``[time, data, neuron, state_dimension]``.
    """
    if trajectory.ndim != 3 or trajectory.shape[0] != times.numel():
        raise ValueError("trajectory must have shape [len(times), n_data, d]")

    contributions = [
        model.neuron_contributions(time_value, state)
        for time_value, state in zip(times, trajectory)
    ]
    return torch.stack(contributions, dim=0)


def _validate_contributions(contributions: Tensor, times: Tensor) -> tuple[int, int]:
    if contributions.ndim != 4:
        raise ValueError("contributions must have shape [time, data, neuron, d]")
    if contributions.shape[0] != times.numel():
        raise ValueError("times and contributions have incompatible lengths")
    if contributions.shape[1] == 0 or contributions.shape[2] == 0:
        raise ValueError("data and neuron dimensions must be non-empty")
    return contributions.shape[1], contributions.shape[2]


def _integrated_data_average(values: Tensor, times: Tensor) -> Tensor:
    """Average a ``[time, data]`` tensor over data and integrate over time."""
    return torch.dot(trapezoidal_weights(times), values.mean(dim=1))


def lambda_monte_carlo(
    contributions: Tensor,
    times: Tensor,
    scheme: BatchScheme,
    n_draws: int,
    rng: np.random.Generator,
    *,
    chunk_size: int = 64,
    return_se: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Estimate the integrated field variance directly by Monte Carlo.

    The returned quantity is the data average of

    ``integral E_omega |F(t)-F_hat^omega(t)|^2 dt``

    evaluated along the supplied full trajectories. ``return_se`` also returns
    the standard error across independent batch draws (not across data/time).
    """
    _, p = _validate_contributions(contributions, times)
    if p != scheme.p:
        raise ValueError("scheme.p does not match the contribution dimension")
    if n_draws <= 0 or chunk_size <= 0:
        raise ValueError("n_draws and chunk_size must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    pi = torch.as_tensor(
        scheme.inclusion_probs.copy(),
        dtype=contributions.dtype,
        device=contributions.device,
    )
    weights = trapezoidal_weights(times)
    gram = torch.einsum("tmpd,tmqd,t->pq", contributions, contributions, weights)
    gram /= contributions.shape[1]
    samples = []

    completed = 0
    while completed < n_draws:
        current = min(chunk_size, n_draws - completed)
        coefficients = -torch.ones(
            (current, p),
            dtype=contributions.dtype,
            device=contributions.device,
        )
        for draw in range(current):
            batch = scheme.sample(rng)
            if len(batch):
                index = torch.tensor(
                    np.asarray(batch, dtype=np.int64).copy(),
                    dtype=torch.long,
                    device=contributions.device,
                )
                coefficients[draw, index] += pi[index].reciprocal()

        # Integrate first: one p-by-p Gram matrix replaces the large
        # [draw,time,data,dimension] intermediate. Same draws, same statistic.
        samples.append(((coefficients @ gram) * coefficients).sum(1).clamp_min(0))
        completed += current

    values = torch.cat(samples)
    mean = values.mean()
    if return_se:
        se = values.std(unbiased=True) / np.sqrt(n_draws) if n_draws > 1 else mean.new_tensor(float("nan"))
        return mean, se
    return mean


def lambda_uniform_fixed_size(
    contributions: Tensor,
    times: Tensor,
    r: int,
) -> Tensor:
    r"""Analytic integrated variance for uniform fixed-size sampling.

    For ``p`` neurons and batches of size ``r`` this implements

    ``Lambda_t = p^2 (p-r) / ((p-1) r) * sigma_t^2``.
    """
    _, p = _validate_contributions(contributions, times)
    if not 1 <= r <= p:
        raise ValueError("r must lie in [1, p]")
    if r == p:
        return contributions.new_zeros(())

    centered = contributions - contributions.mean(dim=2, keepdim=True)
    pointwise = (
        p * (p - r) / (r * (p - 1))
    ) * centered.square().sum(dim=(2, 3))
    return _integrated_data_average(pointwise, times)


def lambda_bernoulli(
    contributions: Tensor,
    times: Tensor,
    q: float,
) -> Tensor:
    r"""Analytic integrated variance for independent Bernoulli sampling."""
    _validate_contributions(contributions, times)
    if not 0 < q <= 1:
        raise ValueError("q must lie in (0, 1]")

    pointwise = ((1.0 - q) / q) * contributions.square().sum(dim=(2, 3))
    return _integrated_data_average(pointwise, times)


def validate_balanced_partition(
    partition: Sequence[Sequence[int]],
    *,
    p: int | None = None,
    r: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Validate and canonicalize an equal-size partition of ``{0,...,p-1}``."""
    blocks = tuple(
        tuple(sorted(int(index) for index in block))
        for block in partition
    )
    if not blocks or any(len(block) == 0 for block in blocks):
        raise ValueError("partition blocks must be non-empty")

    inferred_r = len(blocks[0])
    if any(len(block) != inferred_r for block in blocks):
        raise ValueError("partition must have equal-size blocks")
    if r is not None and inferred_r != r:
        raise ValueError(f"partition blocks must have size {r}")

    flattened = [index for block in blocks for index in block]
    inferred_p = len(flattened) if p is None else int(p)
    if sorted(flattened) != list(range(inferred_p)):
        raise ValueError("partition must contain every neuron exactly once")

    return tuple(sorted(blocks))


def lambda_fixed_disjoint(
    contributions: Tensor,
    times: Tensor,
    partition: Sequence[Sequence[int]],
) -> Tensor:
    r"""Analytic integrated variance for a balanced fixed partition.

    One block is selected uniformly at each switching interval.  If there are
    ``n_b = p/r`` blocks, this evaluates the identity

    ``Lambda_t = n_b * sum_B |sum_{i in B}(f_i-f_bar)|^2``.
    """
    _, p = _validate_contributions(contributions, times)
    blocks = validate_balanced_partition(partition, p=p)
    n_blocks = len(blocks)

    centered = contributions - contributions.mean(dim=2, keepdim=True)
    pointwise = torch.zeros(
        contributions.shape[:2],
        dtype=contributions.dtype,
        device=contributions.device,
    )
    for block in blocks:
        index = torch.tensor(
            np.asarray(block, dtype=np.int64).copy(),
            dtype=torch.long,
            device=contributions.device,
        )
        block_sum = centered[:, :, index].sum(dim=2)
        pointwise += block_sum.square().sum(dim=-1)

    pointwise *= n_blocks
    return _integrated_data_average(pointwise, times)


def random_balanced_partition(
    p: int,
    r: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    """Generate a random balanced partition by shuffling neuron indices once."""
    if p <= 0 or r <= 0 or p % r:
        raise ValueError("p must be positive and divisible by r")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    indices = rng.permutation(p)
    return validate_balanced_partition(
        [indices[start : start + r] for start in range(0, p, r)],
        p=p,
        r=r,
    )
