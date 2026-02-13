"""
Flow matching models for the continuity-equation experiment (§5.2).

- Flow: velocity field v(t, x) for density transport
- FlowRBM: random-batch variant with neuron subsampling
- Midpoint integration with log-determinant tracking
- Kernel density estimation on a grid
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from bisect import bisect_right
from scipy.stats import gaussian_kde


# ── Flow network ───────────────────────────────────────────────────────────

class Flow(nn.Module):
    """Neural velocity field  v(t, x)  for flow matching."""

    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return self.net(torch.cat((t, x), -1))

    def divergence(self, t: Tensor, x: Tensor) -> Tensor:
        """Hutchinson trace estimator of div v."""
        with torch.enable_grad():
            x.requires_grad_(True)
            f = self(t, x)
            e = torch.randn_like(f)
            e_dfdx = torch.autograd.grad(f, x, e, create_graph=False)[0]
            return torch.sum(e_dfdx * e, dim=1)

    def step(self, x: Tensor, log_det: Tensor,
             t_start: Tensor, t_end: Tensor):
        """Midpoint RK2 step with log-determinant tracking."""
        batch = x.shape[0]
        dt = t_end - t_start

        t_exp = t_start.view(1, 1).expand(batch, 1)
        k1 = self(t_exp, x)
        x_mid = x + (dt / 2) * k1

        t_mid = (t_start + dt / 2).view(1, 1).expand(batch, 1)
        k2 = self(t_mid, x_mid)
        x_next = x + dt * k2

        div = self.divergence(t_mid, x_mid)
        return x_next, log_det - div * dt


# ── FlowRBM ───────────────────────────────────────────────────────────────

class FlowRBM(nn.Module):
    """Random-batch variant of Flow.

    At each time step the hidden layer is restricted to the neurons in the
    active batch, and the output is rescaled by 1/pi.

    Args:
        trained_flow: A trained ``Flow`` instance.
        t_span: 1-D tensor of solver time points.
        batch_schedule: List of neuron-index arrays, one per time step.
        pi: Minimum inclusion probability for Horvitz–Thompson scaling.
    """

    def __init__(self, trained_flow: Flow, t_span: Tensor,
                 batch_schedule: list, pi: float):
        super().__init__()
        self.layer1 = trained_flow.net[0]
        self.layer2 = trained_flow.net[2]
        self.activation = nn.ELU()
        self.pi = pi
        self.t_vals = sorted(t_span.cpu().numpy().tolist())
        self.batch_schedule = batch_schedule

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        i = self._interval_index(t)
        batch = self.batch_schedule[i]

        batch_size = y.shape[0]
        t_rep = t.view(1, 1).expand(batch_size, 1)
        inp = torch.cat((t_rep, y), dim=-1)

        W1 = self.layer1.weight[batch, :]
        b1 = self.layer1.bias[batch]
        z = inp.matmul(W1.t()) + b1
        a = self.activation(z)

        W2 = self.layer2.weight[:, batch]
        return a.matmul(W2.t()) / self.pi

    def divergence(self, t: Tensor, x: Tensor) -> Tensor:
        with torch.enable_grad():
            x.requires_grad_(True)
            f = self(t, x)
            e = torch.randn_like(f)
            e_dfdx = torch.autograd.grad(f, x, e, create_graph=False)[0]
            return torch.sum(e_dfdx * e, dim=1)

    def step(self, x: Tensor, log_det: Tensor,
             t_start: Tensor, t_end: Tensor):
        batch = x.shape[0]
        dt = t_end - t_start
        t_exp = t_start.view(1, 1).expand(batch, 1)
        k1 = self(t_exp, x)
        x_mid = x + (dt / 2) * k1
        t_mid = (t_start + dt / 2).view(1, 1).expand(batch, 1)
        k2 = self(t_mid, x_mid)
        x_next = x + dt * k2
        div = self.divergence(t_mid, x_mid)
        return x_next, log_det - div * dt

    def _interval_index(self, t):
        t_val = float(t.item())
        i = bisect_right(self.t_vals, t_val) - 1
        return max(0, min(i, len(self.t_vals) - 1))


# ── Integration ────────────────────────────────────────────────────────────

def integrate_flow(model, particles, weights, n_steps, dt, T,
                   snapshot_times=None):
    """Step-by-step midpoint integration.

    Args:
        model: Flow or FlowRBM with a ``.step`` method.
        particles, weights: Initial particle cloud.
        n_steps, dt, T: Integration parameters.
        snapshot_times: Optional list of times at which to save state.

    Returns:
        (final_x, final_w, snapshots_dict)
    """
    if snapshot_times is None:
        snapshot_times = []

    x = particles.clone()
    log_det = torch.zeros(len(particles))
    snaps = {}

    with torch.no_grad():
        for step in range(n_steps):
            ts = torch.tensor(step * dt)
            te = torch.tensor((step + 1) * dt)
            x, log_det = model.step(x, log_det, ts, te)
            cur_t = (step + 1) * dt
            for st in snapshot_times:
                if abs(cur_t - st) < dt / 2:
                    snaps[st] = (x.clone(),
                                 weights * torch.exp(log_det.clone()))

    return x, weights * torch.exp(log_det), snaps


def compute_trajectory(model, seed_pt, dt, T):
    """Integrate a single particle and return its path as ndarray."""
    x = seed_pt.clone()
    log_det = torch.zeros(1)
    traj = [x.detach().cpu().numpy().flatten()]
    t = 0.0
    with torch.no_grad():
        while t < T - dt / 2:
            ts = torch.tensor(t)
            te = torch.tensor(min(t + dt, T))
            x, log_det = model.step(x, log_det, ts, te)
            traj.append(x.detach().cpu().numpy().flatten())
            t += dt
    return np.array(traj)


# ── Density estimation ─────────────────────────────────────────────────────

def compute_kde(particles, weights, x_grid, y_grid, bw_method=None):
    """Weighted KDE on a 2-D grid.

    Returns:
        Density array of shape (len(y_grid), len(x_grid)).
    """
    pts = particles.detach().cpu().numpy().T
    wts = weights.detach().cpu().numpy()
    kde = gaussian_kde(pts, weights=wts, bw_method=bw_method)
    xx, yy = np.meshgrid(x_grid, y_grid)
    return kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
