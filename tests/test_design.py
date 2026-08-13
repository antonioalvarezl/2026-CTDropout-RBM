from itertools import combinations

import numpy as np
import pytest
import torch

from rnode.batches import make_finite_batch_scheme, make_fixed_disjoint_partition
from rnode.design import (
    exhaustive_partition_extrema,
    lambda_bernoulli,
    lambda_finite_exact,
    lambda_fixed_disjoint,
    lambda_uniform_fixed_size,
    optimize_balanced_partition,
    partition_to_one_based,
    partition_objective,
    weighted_gram_matrix,
)


def _example_contributions():
    generator = torch.Generator().manual_seed(17)
    contributions = torch.randn(5, 3, 4, 2, generator=generator, dtype=torch.float64)
    times = torch.tensor([0.0, 0.1, 0.4, 0.75, 1.0], dtype=torch.float64)
    return contributions, times


def test_uniform_fixed_size_identity_matches_finite_enumeration():
    contributions, times = _example_contributions()
    batches = list(combinations(range(4), 2))
    scheme = make_finite_batch_scheme(
        4, batches, np.full(len(batches), 1 / len(batches))
    )
    exact = lambda_finite_exact(contributions, times, scheme)
    analytic = lambda_uniform_fixed_size(contributions, times, r=2)
    torch.testing.assert_close(analytic, exact, rtol=1e-12, atol=1e-12)


def test_bernoulli_identity_includes_empty_batch():
    contributions, times = _example_contributions()
    q = 0.3
    batches = []
    probabilities = []
    for mask in range(2**4):
        batch = [index for index in range(4) if mask & (1 << index)]
        batches.append(batch)
        probabilities.append(q ** len(batch) * (1 - q) ** (4 - len(batch)))
    scheme = make_finite_batch_scheme(4, batches, probabilities)
    assert len(scheme.batches[0]) == 0
    exact = lambda_finite_exact(contributions, times, scheme)
    analytic = lambda_bernoulli(contributions, times, q=q)
    torch.testing.assert_close(analytic, exact, rtol=1e-12, atol=1e-12)


def test_fixed_disjoint_identity_and_gram_objective():
    contributions, times = _example_contributions()
    partition = ((0, 2), (1, 3))
    scheme = make_fixed_disjoint_partition(4, partition)
    exact = lambda_finite_exact(contributions, times, scheme)
    analytic = lambda_fixed_disjoint(contributions, times, partition)
    gram = weighted_gram_matrix(contributions, times)
    objective = partition_objective(gram, partition)
    torch.testing.assert_close(analytic, exact, rtol=1e-12, atol=1e-12)
    assert analytic.item() == pytest.approx(len(partition) * objective, rel=1e-12)


def test_milp_matches_exhaustive_enumeration_for_p8_r2():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(12, 8, 3))
    centered = features - features.mean(axis=1, keepdims=True)
    gram = np.einsum("lid,ljd->ij", centered, centered)
    exact_min_partition, exact_min, exact_max_partition, exact_max = (
        exhaustive_partition_extrema(gram, r=2)
    )

    minimum = optimize_balanced_partition(gram, r=2, time_limit=30.0)
    maximum = optimize_balanced_partition(gram, r=2, maximize=True, time_limit=30.0)

    assert minimum.certified_optimal, minimum.message
    assert maximum.certified_optimal, maximum.message
    assert minimum.incumbent == pytest.approx(exact_min, abs=1e-8)
    assert maximum.incumbent == pytest.approx(exact_max, abs=1e-8)
    assert partition_objective(gram, exact_min_partition) == pytest.approx(exact_min)
    assert partition_objective(gram, exact_max_partition) == pytest.approx(exact_max)


def test_partition_indices_are_explicitly_converted_to_one_based_for_display():
    partition = ((0, 2), (1, 3))
    assert partition_to_one_based(partition, p=4) == ((1, 3), (2, 4))
