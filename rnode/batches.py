"""Batch families with neuron-wise inclusion probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
Sampler = Callable[[np.random.Generator], Array]


def _validate_p(p: int) -> int:
    if isinstance(p, bool) or int(p) != p or p <= 0:
        raise ValueError("p must be a positive integer")
    return int(p)


@dataclass(frozen=True)
class BatchScheme:
    p: int
    name: str
    inclusion_probs: Array
    expected_batch_size: float
    batches: tuple[Array, ...] | None = None
    batch_probs: Array | None = None
    _sampler: Sampler = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        p = _validate_p(self.p)
        pi = np.asarray(self.inclusion_probs, dtype=float)
        if pi.shape != (p,) or np.any(pi <= 0) or np.any(pi > 1):
            raise ValueError("inclusion_probs must have shape [p] and values in (0, 1]")
        if not np.isclose(pi.sum(), self.expected_batch_size):
            raise ValueError("expected_batch_size must equal sum(inclusion_probs)")
        if self._sampler is None:
            raise ValueError("a sampler is required")
        pi = pi.copy()
        pi.setflags(write=False)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "inclusion_probs", pi)

    def sample(self, rng: np.random.Generator) -> Array:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return np.asarray(self._sampler(rng), dtype=np.int64).copy()


def make_finite_batch_scheme(
    p: int,
    batches: Sequence[Sequence[int] | Array],
    probs: Sequence[float] | Array,
    name: str = "Finite batch family",
) -> BatchScheme:
    p = _validate_p(p)
    if not batches:
        raise ValueError("at least one candidate batch is required")
    candidates = []
    for raw_batch in batches:
        batch = np.asarray(raw_batch, dtype=np.int64)
        if batch.ndim != 1 or len(np.unique(batch)) != len(batch):
            raise ValueError("candidate batches must be one-dimensional and unique")
        if batch.size and (batch.min() < 0 or batch.max() >= p):
            raise ValueError("candidate index out of range")
        batch = batch.copy()
        batch.setflags(write=False)
        candidates.append(batch)
    probabilities = np.asarray(probs, dtype=float)
    if probabilities.shape != (len(candidates),) or np.any(probabilities < 0):
        raise ValueError("one non-negative probability is required per batch")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("candidate probabilities must sum to one")
    pi = np.zeros(p)
    for batch, probability in zip(candidates, probabilities):
        pi[batch] += probability
    if np.any(pi <= 0):
        raise ValueError("every neuron must have positive inclusion probability")
    probabilities = probabilities.copy()
    probabilities.setflags(write=False)
    candidate_tuple = tuple(candidates)

    def sampler(rng):
        return candidate_tuple[int(rng.choice(len(candidate_tuple), p=probabilities))]

    return BatchScheme(
        p, name, pi, float(pi.sum()), candidate_tuple, probabilities, sampler
    )


def make_full_batch(p: int) -> BatchScheme:
    return make_finite_batch_scheme(p, [np.arange(p)], [1.0], "Full batch")


def make_uniform_fixed_size(p: int, r: int) -> BatchScheme:
    p = _validate_p(p)
    if not 1 <= r <= p:
        raise ValueError("r must lie in [1, p]")

    def sampler(rng):
        return rng.choice(p, size=r, replace=False)

    return BatchScheme(
        p,
        f"Uniform fixed-size (r={r})",
        np.full(p, r / p),
        float(r),
        _sampler=sampler,
    )


def make_fixed_disjoint_partition(
    p: int,
    batches: Sequence[Sequence[int] | Array] | None = None,
    *,
    n_batches: int | None = None,
    rng: np.random.Generator | None = None,
    name: str | None = None,
) -> BatchScheme:
    p = _validate_p(p)
    if batches is None:
        if n_batches is None or not 1 <= n_batches <= p:
            raise ValueError("n_batches must lie in [1, p]")
        if rng is None:
            raise ValueError("an explicit rng is required to construct a partition")
        batches = np.array_split(rng.permutation(p), n_batches)
    elif n_batches is not None:
        raise ValueError("do not pass n_batches with explicit batches")
    partition = [np.asarray(batch, dtype=np.int64) for batch in batches]
    if not partition or any(len(batch) == 0 for batch in partition):
        raise ValueError("partition blocks must be non-empty")
    if not np.array_equal(np.sort(np.concatenate(partition)), np.arange(p)):
        raise ValueError("partition must contain each neuron exactly once")
    return make_finite_batch_scheme(
        p,
        partition,
        np.full(len(partition), 1 / len(partition)),
        name or "Fixed disjoint partition",
    )


def make_bernoulli(p: int, q: float = 0.5) -> BatchScheme:
    p = _validate_p(p)
    if not 0 < q <= 1:
        raise ValueError("q must lie in (0, 1]")

    def sampler(rng):
        return np.flatnonzero(rng.random(p) < q)

    return BatchScheme(p, f"Bernoulli (q={q})", np.full(p, q), p * q, _sampler=sampler)


def sample_batch_sequence(
    scheme: BatchScheme, n_intervals: int, rng: np.random.Generator
) -> list[Array]:
    if n_intervals < 0:
        raise ValueError("n_intervals must be non-negative")
    return [scheme.sample(rng) for _ in range(n_intervals)]
