from itertools import combinations

import numpy as np
import pytest
import torch

from rnode.batches import (
    make_bernoulli,
    make_finite_batch_scheme,
    make_fixed_disjoint_partition,
    make_full_batch,
    make_uniform_fixed_size,
    sample_batch_sequence,
)
from rnode.models import TimeDepODE


def test_finite_family_computes_analytic_inclusion_probabilities():
    scheme = make_finite_batch_scheme(
        3, [[0], [0, 1], [2]], [0.2, 0.3, 0.5], name="non-uniform"
    )
    np.testing.assert_allclose(scheme.inclusion_probs, [0.5, 0.3, 0.5])
    assert scheme.expected_batch_size == pytest.approx(1.3)


def test_finite_family_rejects_zero_inclusion_probability():
    with pytest.raises(ValueError, match="positive inclusion"):
        make_finite_batch_scheme(3, [[0], [1]], [0.5, 0.5])


def test_unequal_disjoint_partition_has_equal_neuron_inclusion():
    scheme = make_fixed_disjoint_partition(5, [[0, 1, 2], [3, 4]])
    np.testing.assert_allclose(scheme.inclusion_probs, np.full(5, 0.5))
    assert scheme.expected_batch_size == pytest.approx(2.5)


@pytest.mark.parametrize(
    "scheme, expected",
    [
        (make_full_batch(4), np.ones(4)),
        (make_uniform_fixed_size(4, 2), np.full(4, 0.5)),
        (make_bernoulli(4, 0.3), np.full(4, 0.3)),
    ],
)
def test_inclusion_probabilities_are_analytic(scheme, expected):
    np.testing.assert_allclose(scheme.inclusion_probs, expected)


def test_uniform_fixed_size_samples_without_enumerating_combinations():
    scheme = make_uniform_fixed_size(100, 50)
    assert scheme.batches is None
    batch = scheme.sample(np.random.default_rng(4))
    assert len(batch) == len(np.unique(batch)) == 50


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_full_batch(4),
        lambda: make_uniform_fixed_size(4, 2),
        lambda: make_fixed_disjoint_partition(4, [[0], [1, 2, 3]]),
        lambda: make_bernoulli(4, 0.4),
    ],
)
def test_empirical_horvitz_thompson_unbiasedness(factory):
    scheme = factory()
    contributions = np.array([[0.2, -0.5], [1.0, 0.25], [-0.3, 0.7], [0.4, 0.1]])
    expected = contributions.sum(axis=0)
    rng = np.random.default_rng(12345)
    estimate = np.zeros(2)
    for _ in range(40_000):
        batch = scheme.sample(rng)
        estimate += (contributions[batch] / scheme.inclusion_probs[batch, None]).sum(0)
    np.testing.assert_allclose(estimate / 40_000, expected, atol=2.5e-2)


def test_bernoulli_keeps_empty_batches():
    scheme = make_bernoulli(3, q=0.1)
    batches = sample_batch_sequence(scheme, 10_000, np.random.default_rng(9))
    assert any(len(batch) == 0 for batch in batches)
    empirical = [np.mean([index in batch for batch in batches]) for index in range(3)]
    np.testing.assert_allclose(empirical, 0.1, atol=0.012)


def test_full_batch_matches_complete_model_exactly():
    torch.manual_seed(3)
    model = TimeDepODE(hidden_dim=5)
    x = torch.randn(7, 2)
    scheme = make_full_batch(5)
    observed = model.forward_batch(0.3, x, scheme.batches[0], scheme.inclusion_probs)
    torch.testing.assert_close(observed, model(0.3, x), rtol=0, atol=0)


def test_nonuniform_ht_correction_is_exact_in_expectation():
    torch.manual_seed(4)
    model = TimeDepODE(hidden_dim=3)
    x = torch.randn(4, 2)
    scheme = make_finite_batch_scheme(3, [[0], [0, 1], [2]], [0.2, 0.3, 0.5])
    expected = sum(
        probability * model.forward_batch(0.0, x, batch, scheme.inclusion_probs)
        for probability, batch in zip(scheme.batch_probs, scheme.batches)
    )
    torch.testing.assert_close(expected, model(0.0, x), rtol=1e-6, atol=1e-6)


def test_exact_ht_unbiasedness_for_each_small_batch_family():
    torch.manual_seed(14)
    model = TimeDepODE(hidden_dim=4).double()
    x = torch.randn(3, 2, dtype=torch.float64)
    q = 0.3
    uniform = make_uniform_fixed_size(4, 2)
    disjoint = make_fixed_disjoint_partition(4, [[0, 2], [1, 3]])
    bernoulli = make_bernoulli(4, q)
    full = make_full_batch(4)
    cases = [
        (
            uniform,
            [np.asarray(batch) for batch in combinations(range(4), 2)],
            np.full(6, 1.0 / 6.0),
        ),
        (disjoint, list(disjoint.batches), disjoint.batch_probs),
        (
            bernoulli,
            [
                np.asarray([index for index in range(4) if mask & (1 << index)])
                for mask in range(2**4)
            ],
            np.asarray(
                [
                    q ** int(mask.bit_count()) * (1 - q) ** (4 - int(mask.bit_count()))
                    for mask in range(2**4)
                ]
            ),
        ),
        (full, list(full.batches), full.batch_probs),
    ]
    for scheme, batches, probabilities in cases:
        expected = sum(
            float(probability)
            * model.forward_batch(0.37, x, batch, scheme.inclusion_probs)
            for probability, batch in zip(probabilities, batches)
        )
        torch.testing.assert_close(expected, model(0.37, x), rtol=1e-12, atol=1e-12)


def test_p24_monte_carlo_ht_unbiasedness_and_unique_indices():
    p = 24
    contribution_rng = np.random.default_rng(71)
    contributions = contribution_rng.normal(scale=0.2, size=(p, 2))
    target = contributions.sum(axis=0)
    schemes = [
        make_uniform_fixed_size(p, 8),
        make_fixed_disjoint_partition(p, np.arange(p).reshape(3, 8)),
        make_bernoulli(p, 1 / 3),
        make_full_batch(p),
    ]
    for scheme_index, scheme in enumerate(schemes):
        rng = np.random.default_rng(1000 + scheme_index)
        samples = np.empty((20_000, 2))
        for draw in range(len(samples)):
            batch = scheme.sample(rng)
            assert len(batch) == len(np.unique(batch))
            samples[draw] = (
                contributions[batch] / scheme.inclusion_probs[batch, None]
            ).sum(axis=0)
        standard_error = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
        np.testing.assert_array_less(
            np.abs(samples.mean(axis=0) - target), 6 * standard_error + 1e-12
        )
