"""Flow-matching velocity field used by the transport experiment."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .models import _batch_tensors


class Flow(nn.Module):
    r"""Exact paper field ``W0 tanh(A0 x + b0 + b1 t)``."""

    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, dim, bias=False)
        )

    def control_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        A0 = self.net[0].weight[:, 1:]
        b0 = self.net[0].bias
        b1 = self.net[0].weight[:, 0]
        W0 = self.net[2].weight
        return A0, b0, b1, W0

    def _time_column(self, t: Tensor | float, x: Tensor) -> Tensor:
        value = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if value.numel() == 1:
            return value.reshape(1, 1).expand(x.shape[0], 1)
        if value.shape == (x.shape[0],):
            return value[:, None]
        if value.shape == (x.shape[0], 1):
            return value
        raise ValueError("t must be scalar or have one entry per point")

    def neuron_contributions(self, t: Tensor | float, x: Tensor) -> Tensor:
        A0, b0, b1, W0 = self.control_parameters()
        activation = torch.tanh(x.matmul(A0.t()) + b0 + self._time_column(t, x) * b1)
        return activation.unsqueeze(-1) * W0.t().unsqueeze(0)

    def forward(self, t: Tensor | float, x: Tensor) -> Tensor:
        return self.neuron_contributions(t, x).sum(dim=1)

    def forward_batch(self, t, x, indices, inclusion_probs) -> Tensor:
        index, pi = _batch_tensors(indices, inclusion_probs, self.hidden_dim, x)
        A0, b0, b1, W0 = self.control_parameters()
        activation = torch.tanh(
            x.matmul(A0[index].t()) + b0[index] + self._time_column(t, x) * b1[index]
        )
        scaled_weights = W0[:, index].t() / pi[index, None]
        return (activation.unsqueeze(-1) * scaled_weights.unsqueeze(0)).sum(dim=1)

    def divergence(self, t: Tensor | float, x: Tensor) -> Tensor:
        """Analytic trace of the spatial Jacobian; no stochastic estimator."""
        A0, b0, b1, W0 = self.control_parameters()
        activation = torch.tanh(x.matmul(A0.t()) + b0 + self._time_column(t, x) * b1)
        derivative = 1.0 - activation.square()
        diagonal_coefficients = (W0.t() * A0).sum(dim=1)
        return derivative.matmul(diagonal_coefficients)
