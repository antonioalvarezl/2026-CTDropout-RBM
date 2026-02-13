"""
Batch sampling schemes for random batch neural ODEs.

Implements the seven canonical schemes from Section 4.1 of the paper:
    (1) Single-batch          — all neurons active
    (2) Drop-one              — remove one neuron per interval
    (3) Pick-one              — keep one neuron per interval
    (4) Balanced subsets       — sample uniformly among all r-subsets
    (5) Balanced disjoint      — partition into p/r disjoint batches
    (6) All subsets            — uniform over 2^p subsets  (impractical; for reference)
    (7) Bernoulli dropout      — i.i.d. Bernoulli(q_B) mask per neuron

Each ``make_*`` function returns a ``BatchScheme`` namedtuple containing:
    batches   — list of index arrays (the possible batches B_j)
    probs     — sampling probabilities q_j
    pi_min    — minimum inclusion probability
    name      — human-readable label

The function ``sample_batch_sequence`` draws a random schedule from a scheme,
and ``sample_batch_sequence_from_h`` converts a switching interval h to the
appropriate number of repetitions on a given time grid.
"""

from __future__ import annotations

import math
import random
from collections import namedtuple
from itertools import combinations

import numpy as np
import torch

BatchScheme = namedtuple("BatchScheme", ["batches", "probs", "pi_min", "name"])


# ---------------------------------------------------------------------------
# Scheme constructors
# ---------------------------------------------------------------------------

def make_single_batch(p: int) -> BatchScheme:
    """(1) Single-batch: all neurons always active."""
    batches = [np.arange(p)]
    return BatchScheme(batches=batches, probs=[1.0], pi_min=1.0,
                       name="Single-batch")


def make_drop_one(p: int) -> BatchScheme:
    """(2) Drop-one: n_b = p batches, each of size p-1."""
    batches = [np.setdiff1d(np.arange(p), [j]) for j in range(p)]
    probs = [1.0 / p] * p
    pi_min = (p - 1) / p
    return BatchScheme(batches=batches, probs=probs, pi_min=pi_min,
                       name="Drop-one")


def make_pick_one(p: int) -> BatchScheme:
    """(3) Pick-one: n_b = p batches, each of size 1."""
    batches = [np.array([j]) for j in range(p)]
    probs = [1.0 / p] * p
    pi_min = 1.0 / p
    return BatchScheme(batches=batches, probs=probs, pi_min=pi_min,
                       name="Pick-one")


def make_balanced_subsets(p: int, r: int) -> BatchScheme:
    """(4) Balanced subsets: all C(p, r) subsets of size r, uniform sampling.

    Warning: the number of batches grows combinatorially.  Use only for
    small p and moderate r.
    """
    if r <= 0 or r > p:
        raise ValueError(f"r must be in [1, p], got r={r}, p={p}")
    batches = [np.array(list(c)) for c in combinations(range(p), r)]
    n_b = len(batches)
    probs = [1.0 / n_b] * n_b
    pi_min = r / p
    return BatchScheme(batches=batches, probs=probs, pi_min=pi_min,
                       name=f"Balanced-subsets (r={r})")


def make_balanced_disjoint(p: int, r: int, seed: int | None = None) -> BatchScheme:
    """(5) Balanced disjoint batches of size r (r must divide p).

    Args:
        p: Total number of neurons.
        r: Batch size; must divide p.
        seed: Optional random seed for the partition.
    """
    if p % r != 0:
        raise ValueError(f"r={r} must divide p={p}")
    indices = np.arange(p)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_b = p // r
    batches = [indices[i * r:(i + 1) * r] for i in range(n_b)]
    probs = [1.0 / n_b] * n_b
    pi_min = r / p
    return BatchScheme(batches=batches, probs=probs, pi_min=pi_min,
                       name=f"Balanced-disjoint (r={r})")


def make_bernoulli(p: int, q_B: float = 0.5) -> BatchScheme:
    """(7) Bernoulli dropout with keep probability q_B.

    Instead of enumerating all 2^p subsets, batches are sampled on-the-fly
    via ``sample_bernoulli_mask``.  We store a sentinel so that
    ``sample_batch_sequence`` knows to use the Bernoulli sampler.

    Returns a BatchScheme where ``batches`` is set to ``None`` (sentinel)
    and ``probs`` encodes q_B for downstream use.
    """
    if not 0 < q_B <= 1:
        raise ValueError(f"q_B must be in (0, 1], got {q_B}")
    pi_min = q_B
    return BatchScheme(batches=None, probs=q_B, pi_min=pi_min,
                       name=f"Bernoulli (q_B={q_B})")


def sample_bernoulli_mask(p: int, q_B: float) -> np.ndarray:
    """Draw one Bernoulli batch: each neuron included independently with prob q_B."""
    mask = np.random.rand(p) < q_B
    if not mask.any():
        # Ensure at least one neuron is active
        mask[np.random.randint(p)] = True
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Schedule sampling
# ---------------------------------------------------------------------------

def sample_batch_sequence(scheme: BatchScheme, n_steps: int, p: int = None,
                          rep: int = 1) -> list:
    """Draw a random batch schedule for *n_steps* integration steps.

    Each batch is held constant for *rep* consecutive steps.

    Args:
        scheme: A ``BatchScheme`` returned by any ``make_*`` function.
        n_steps: Total number of ODE solver steps.
        p: Number of neurons (required only for Bernoulli schemes).
        rep: Number of consecutive steps to hold each batch.

    Returns:
        List of length *n_steps* with one index-array per step.
    """
    is_bernoulli = scheme.batches is None

    schedule: list = []
    n_switches = math.ceil(n_steps / rep)

    for _ in range(n_switches):
        if is_bernoulli:
            if p is None:
                raise ValueError("p is required for Bernoulli schemes")
            batch = sample_bernoulli_mask(p, scheme.probs)
        else:
            idx = _weighted_choice(scheme.probs)
            batch = scheme.batches[idx]
        schedule.extend([batch] * rep)

    return schedule[:n_steps]


def sample_batch_sequence_from_h(scheme: BatchScheme, t_span: torch.Tensor,
                                 h: float, p: int = None) -> list:
    """Generate batch schedule from switching interval *h*.

    Converts h to the number of ODE steps per switch (rep) on the given
    time grid, then delegates to ``sample_batch_sequence``.

    Args:
        scheme: A ``BatchScheme``.
        t_span: 1-D tensor of solver time points.
        h: Duration to hold each batch (seconds).
        p: Number of neurons (required only for Bernoulli).

    Returns:
        List of length ``len(t_span)`` with one index-array per step.
    """
    n_steps = len(t_span)
    T = float(t_span[-1] - t_span[0])
    rep = max(1, int(round(h * n_steps / T)))
    return sample_batch_sequence(scheme, n_steps, p=p, rep=rep)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_choice(probs) -> int:
    """Fast weighted random choice (uniform shortcut when possible)."""
    if isinstance(probs, (list, np.ndarray)):
        n = len(probs)
        # Check if uniform (common case)
        if all(abs(p - probs[0]) < 1e-12 for p in probs):
            return random.randrange(n)
        return random.choices(range(n), weights=probs, k=1)[0]
    raise TypeError(f"Unsupported probs type: {type(probs)}")


def expected_batch_size(scheme: BatchScheme, p: int = None) -> float:
    """Compute the expected number of active neurons per step.

    Useful for the cost formula  C_RM = (T / Δt) × E[|B|].
    """
    if scheme.batches is None:  # Bernoulli
        q_B = scheme.probs
        return q_B * (p if p is not None else 1)
    sizes = np.array([len(b) for b in scheme.batches])
    probs = np.array(scheme.probs)
    return float(np.dot(sizes, probs))
