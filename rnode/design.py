"""Variance identities and balanced-partition design for random-batch ODEs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import time
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from torch import Tensor

from .batches import BatchScheme


def trapezoidal_weights(times: Tensor) -> Tensor:
    """Return weights whose dot product implements trapezoidal quadrature."""
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
    model, times: Tensor, trajectory: Tensor
) -> Tensor:
    """Evaluate ``f_i(t, x_m(t))`` with shape ``[time, data, neuron, d]``."""
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
    """Average over data, then integrate a ``[time, data]`` tensor."""
    return torch.dot(trapezoidal_weights(times), values.mean(dim=1))


def lambda_monte_carlo(
    contributions: Tensor,
    times: Tensor,
    scheme: BatchScheme,
    n_draws: int,
    rng: np.random.Generator,
    *,
    chunk_size: int = 64,
) -> Tensor:
    r"""Direct Monte Carlo estimate of integrated ``E|F-F^omega|^2``."""
    _, p = _validate_contributions(contributions, times)
    if p != scheme.p:
        raise ValueError("scheme.p does not match the contribution dimension")
    if n_draws <= 0 or chunk_size <= 0:
        raise ValueError("n_draws and chunk_size must be positive")
    pi = torch.tensor(
        scheme.inclusion_probs.copy(),
        dtype=contributions.dtype,
        device=contributions.device,
    )
    accumulated = torch.zeros(
        contributions.shape[:2],
        dtype=contributions.dtype,
        device=contributions.device,
    )

    completed = 0
    while completed < n_draws:
        current = min(chunk_size, n_draws - completed)
        coefficients = -torch.ones(
            (current, p), dtype=contributions.dtype, device=contributions.device
        )
        for draw in range(current):
            batch = scheme.sample(rng)
            if len(batch):
                index = torch.tensor(
                    batch, dtype=torch.long, device=contributions.device
                )
                coefficients[draw, index] += pi[index].reciprocal()
        difference = torch.einsum("sp,tmpd->stmd", coefficients, contributions)
        accumulated += difference.square().sum(dim=-1).sum(dim=0)
        completed += current

    return _integrated_data_average(accumulated / n_draws, times)


def lambda_finite_exact(
    contributions: Tensor,
    times: Tensor,
    scheme: BatchScheme,
) -> Tensor:
    """Exact integrated variance for an explicitly represented batch family."""
    _, p = _validate_contributions(contributions, times)
    if scheme.p != p or scheme.batches is None or scheme.batch_probs is None:
        raise ValueError("an explicit finite scheme with matching p is required")
    full_field = contributions.sum(dim=2)
    expected_squared_error = torch.zeros_like(full_field[..., 0])
    pi = torch.tensor(
        scheme.inclusion_probs.copy(),
        dtype=contributions.dtype,
        device=contributions.device,
    )
    for probability, batch in zip(scheme.batch_probs, scheme.batches):
        index = torch.tensor(batch, dtype=torch.long, device=contributions.device)
        batch_field = (contributions[:, :, index] / pi[index, None]).sum(dim=2)
        expected_squared_error += float(probability) * (
            batch_field - full_field
        ).square().sum(dim=-1)
    return _integrated_data_average(expected_squared_error, times)


def lambda_uniform_fixed_size(
    contributions: Tensor,
    times: Tensor,
    r: int,
) -> Tensor:
    """Analytic variance for uniform sampling of ``r`` out of ``p`` neurons."""
    _, p = _validate_contributions(contributions, times)
    if not 1 <= r <= p:
        raise ValueError("r must lie in [1, p]")
    if r == p:
        return contributions.new_zeros(())
    centered = contributions - contributions.mean(dim=2, keepdim=True)
    pointwise = (p * (p - r) / (r * (p - 1))) * centered.square().sum(dim=(2, 3))
    return _integrated_data_average(pointwise, times)


def lambda_bernoulli(
    contributions: Tensor,
    times: Tensor,
    q: float,
) -> Tensor:
    """Analytic variance for independent Bernoulli sampling."""
    _validate_contributions(contributions, times)
    if not 0 < q <= 1:
        raise ValueError("q must lie in (0, 1]")
    pointwise = ((1.0 - q) / q) * contributions.square().sum(dim=(2, 3))
    return _integrated_data_average(pointwise, times)


def lambda_fixed_disjoint(
    contributions: Tensor,
    times: Tensor,
    partition: Sequence[Sequence[int]],
) -> Tensor:
    """Analytic variance for uniform sampling from an equal-size partition."""
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
        index = torch.tensor(block, dtype=torch.long, device=contributions.device)
        pointwise += centered[:, :, index].sum(dim=2).square().sum(dim=-1)
    pointwise *= n_blocks
    return _integrated_data_average(pointwise, times)


def weighted_gram_matrix(contributions: Tensor, times: Tensor) -> Tensor:
    r"""Build ``G_ij = sum_l w_l <g_i^l, g_j^l>`` on a full trajectory."""
    n_data, _ = _validate_contributions(contributions, times)
    centered = contributions - contributions.mean(dim=2, keepdim=True)
    weights = trapezoidal_weights(times)[:, None].expand(-1, n_data) / n_data
    return torch.einsum("tmid,tmjd,tm->ij", centered, centered, weights)


def validate_balanced_partition(
    partition: Sequence[Sequence[int]],
    *,
    p: int | None = None,
    r: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Validate and canonicalise a balanced partition."""
    blocks = tuple(tuple(sorted(int(index) for index in block)) for block in partition)
    if not blocks or any(len(block) == 0 for block in blocks):
        raise ValueError("partition blocks must be non-empty")
    inferred_r = len(blocks[0])
    if any(len(block) != inferred_r for block in blocks):
        raise ValueError("partition must have equal-size blocks")
    if r is not None and inferred_r != r:
        raise ValueError(f"partition blocks must have size {r}")
    flattened = [index for block in blocks for index in block]
    inferred_p = len(flattened) if p is None else p
    if sorted(flattened) != list(range(inferred_p)):
        raise ValueError("partition must contain every neuron exactly once")
    return tuple(sorted(blocks))


def partition_to_one_based(
    partition: Sequence[Sequence[int]], *, p: int | None = None
) -> tuple[tuple[int, ...], ...]:
    """Return a validated partition with indices converted for paper display."""
    blocks = validate_balanced_partition(partition, p=p)
    return tuple(tuple(index + 1 for index in block) for block in blocks)


def partition_objective(
    G: Tensor | np.ndarray, partition: Sequence[Sequence[int]]
) -> float:
    r"""Evaluate ``sum_B sum_{i,j in B} G_ij``."""
    matrix = np.asarray(G.detach().cpu() if isinstance(G, Tensor) else G, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("G must be square")
    blocks = validate_balanced_partition(partition, p=matrix.shape[0])
    return float(sum(matrix[np.ix_(block, block)].sum() for block in blocks))


def random_balanced_partition(
    p: int,
    r: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], ...]:
    """Generate a uniformly shuffled balanced partition."""
    if p <= 0 or r <= 0 or p % r:
        raise ValueError("p must be positive and divisible by r")
    indices = rng.permutation(p)
    return validate_balanced_partition(
        [indices[start : start + r] for start in range(0, p, r)], p=p, r=r
    )


def enumerate_balanced_partitions(
    p: int, r: int
) -> Iterable[tuple[tuple[int, ...], ...]]:
    """Enumerate unlabeled balanced partitions; intended only for small ``p``."""
    if p <= 0 or r <= 0 or p % r:
        raise ValueError("p must be positive and divisible by r")

    def recurse(remaining: tuple[int, ...]):
        if not remaining:
            yield ()
            return
        first = remaining[0]
        tail = remaining[1:]
        for partners in combinations(tail, r - 1):
            block = (first, *partners)
            selected = set(block)
            rest = tuple(index for index in remaining if index not in selected)
            for suffix in recurse(rest):
                yield (tuple(sorted(block)), *suffix)

    yield from recurse(tuple(range(p)))


def exhaustive_partition_extrema(
    G: Tensor | np.ndarray,
    r: int,
) -> tuple[tuple[tuple[int, ...], ...], float, tuple[tuple[int, ...], ...], float]:
    """Return exact minimum and maximum over small balanced partitions."""
    p = np.asarray(G.detach().cpu() if isinstance(G, Tensor) else G).shape[0]
    minimum_partition = maximum_partition = None
    minimum_value = np.inf
    maximum_value = -np.inf
    for partition in enumerate_balanced_partitions(p, r):
        value = partition_objective(G, partition)
        if value < minimum_value:
            minimum_partition, minimum_value = partition, value
        if value > maximum_value:
            maximum_partition, maximum_value = partition, value
    assert minimum_partition is not None and maximum_partition is not None
    return minimum_partition, minimum_value, maximum_partition, maximum_value


@dataclass(frozen=True)
class PartitionDesignResult:
    partition: tuple[tuple[int, ...], ...] | None
    status_code: int
    status: str
    message: str
    elapsed_seconds: float
    incumbent: float | None
    mip_gap: float | None
    mip_node_count: int | None
    certified_optimal: bool
    sense: str

    @property
    def label(self) -> str:
        if self.sense == "minimize":
            return "optimal" if self.certified_optimal else "optimized"
        return (
            "adversarial-optimal" if self.certified_optimal else "adversarial-optimized"
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.partition is not None:
            result["partition"] = [list(block) for block in self.partition]
        result["label"] = self.label
        return result


_STATUS_NAMES = {
    0: "optimal",
    1: "limit_reached",
    2: "infeasible",
    3: "unbounded",
    4: "solver_error",
}


def optimize_balanced_partition(
    G: Tensor | np.ndarray,
    r: int,
    *,
    maximize: bool = False,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
) -> PartitionDesignResult:
    """Solve balanced partition design using a binary linearised MILP."""
    matrix = np.asarray(G.detach().cpu() if isinstance(G, Tensor) else G, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("G must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("G must contain finite values")
    matrix = (matrix + matrix.T) / 2
    p = matrix.shape[0]
    if r <= 0 or p % r:
        raise ValueError("r must be positive and divide p")
    n_blocks = p // r
    pairs = list(combinations(range(p), 2))
    n_x = p * n_blocks
    n_variables = n_x + len(pairs) * n_blocks

    def x_index(neuron: int, block: int) -> int:
        return block * p + neuron

    def y_index(pair_index: int, block: int) -> int:
        return n_x + block * len(pairs) + pair_index

    objective = np.zeros(n_variables)
    for block in range(n_blocks):
        for neuron in range(p):
            objective[x_index(neuron, block)] = matrix[neuron, neuron]
        for pair_index, (left, right) in enumerate(pairs):
            objective[y_index(pair_index, block)] = 2.0 * matrix[left, right]
    solver_objective = -objective if maximize else objective

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_row(entries: Sequence[tuple[int, float]], lower: float, upper: float):
        row = len(lower_bounds)
        for column, value in entries:
            row_indices.append(row)
            column_indices.append(column)
            values.append(value)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Every neuron belongs to exactly one labeled block.
    for neuron in range(p):
        add_row([(x_index(neuron, block), 1.0) for block in range(n_blocks)], 1.0, 1.0)
    # Every block contains exactly r neurons.
    for block in range(n_blocks):
        add_row([(x_index(neuron, block), 1.0) for neuron in range(p)], r, r)
    # McCormick constraints for binary y_ijb = x_ib * x_jb.
    for block in range(n_blocks):
        for pair_index, (left, right) in enumerate(pairs):
            y = y_index(pair_index, block)
            xi = x_index(left, block)
            xj = x_index(right, block)
            add_row([(y, 1.0), (xi, -1.0)], -np.inf, 0.0)
            add_row([(y, 1.0), (xj, -1.0)], -np.inf, 0.0)
            add_row([(y, 1.0), (xi, -1.0), (xj, -1.0)], -1.0, np.inf)
    # Safe symmetry breaking: label the block containing neuron zero as block 0.
    add_row([(x_index(0, 0), 1.0)], 1.0, 1.0)
    # Order the remaining interchangeable blocks by the sum of their indices.
    for block in range(1, n_blocks - 1):
        entries = [(x_index(neuron, block), float(neuron)) for neuron in range(p)] + [
            (x_index(neuron, block + 1), -float(neuron)) for neuron in range(p)
        ]
        add_row(entries, -np.inf, 0.0)

    constraints = LinearConstraint(
        coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(lower_bounds), n_variables),
        ).tocsr(),
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
    )
    options: dict[str, float | bool] = {"presolve": True, "mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    started = time.perf_counter()
    result = milp(
        solver_objective,
        integrality=np.ones(n_variables, dtype=np.int8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=constraints,
        options=options,
    )
    elapsed = time.perf_counter() - started

    partition = None
    incumbent = None
    if result.x is not None:
        x_values = result.x[:n_x].reshape(n_blocks, p)
        decoded = [
            tuple(np.flatnonzero(x_values[block] > 0.5)) for block in range(n_blocks)
        ]
        try:
            partition = validate_balanced_partition(decoded, p=p, r=r)
            incumbent = partition_objective(matrix, partition)
        except ValueError:
            partition = None

    status_code = int(result.status)
    return PartitionDesignResult(
        partition=partition,
        status_code=status_code,
        status=_STATUS_NAMES.get(status_code, "unknown"),
        message=str(result.message),
        elapsed_seconds=elapsed,
        incumbent=incumbent,
        mip_gap=(
            None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)
        ),
        mip_node_count=(
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
        certified_optimal=status_code == 0 and partition is not None,
        sense="maximize" if maximize else "minimize",
    )
