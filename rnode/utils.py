"""
Shared utilities: metrics, timing, evaluation helpers, batch generation.
"""

import time, random
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint


# ── Batch generation ───────────────────────────────────────────────────────

def split_neurons(n_neurons: int, n_batches: int, seed: int = None):
    """Split neurons into disjoint batches via interleaved assignment.

    Returns:
        List of lists, each containing neuron indices.
    """
    indices = np.arange(n_neurons)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    return [indices[i::n_batches].tolist() for i in range(n_batches)]


def generate_batch_sequence(predefined_batches, n_steps, rep=1):
    """Generate a random batch schedule for the ODE time span.

    Each batch is randomly chosen and held constant for *rep* steps.

    Returns:
        List of length *n_steps* with one batch per step.
    """
    batches = []
    n_batches = len(predefined_batches)
    for i in range(0, n_steps, rep):
        chosen = predefined_batches[random.randint(0, n_batches - 1)]
        batches.extend([chosen] * min(rep, n_steps - i))
    return batches[:n_steps]


# ── Metrics ────────────────────────────────────────────────────────────────


def compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Classification accuracy using sign of first component.

    Args:
        y_pred: (n, 2) or (n, d) predictions.
        y_true: (n, 1) labels in {-1, +1}.
    """
    pred = torch.sign(y_pred[:, 0])
    pred[pred == 0] = 1
    return (pred == y_true.squeeze()).float().mean().item()


def compute_loss(y_pred, y_target, dt, model, alpha=0.01, beta=0.5):
    """Training loss from §5.3:

        J = MSE(x_T, y) + beta * dt * mean(||x_t - y||^2) + (alpha/2) * ||θ||^2

    Args:
        y_pred: Trajectory tensor (T, N, d).
        y_target: Target tensor (N, d) or (N, 1).
        dt: Time step.
        model: nn.Module (for weight regularisation).
        alpha, beta: Regularisation coefficients.
    """
    terminal = torch.mean((y_pred[-1] - y_target) ** 2)
    y_exp = y_target.unsqueeze(0).expand_as(y_pred)
    traj = dt * torch.mean(torch.sum((y_pred - y_exp) ** 2, dim=2))
    l2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
    return terminal + beta * traj + (alpha / 2) * l2


def timeit(fn, n_repeats: int = 10):
    """Warm-up + timed repeats.  Returns (mean_s, std_s)."""
    fn()
    ts = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.mean(ts), np.std(ts)


def l1_histogram_error(p1, w1, p2, w2, x_edges, y_edges):
    """L1 error between two weighted particle clouds on a histogram grid."""
    ca = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])

    def _h(p, w):
        pn = p.detach().cpu().numpy()
        wn = w.detach().cpu().numpy()
        h, _, _ = np.histogram2d(pn[:, 0], pn[:, 1],
                                 bins=[x_edges, y_edges], weights=wn)
        return h

    return float(np.abs(_h(p1, w1) - _h(p2, w2)).sum() * ca)


def count_trainable_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    """Total number of parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())
