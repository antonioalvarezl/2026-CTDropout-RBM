"""Current time-dependent neural vector field."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _scalar_time(t: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(t, dtype=reference.dtype, device=reference.device)
    if value.numel() != 1:
        raise ValueError("time-dependent controls require a scalar time")
    return value.reshape(1, 1)


def _batch_tensors(
    indices: Sequence[int] | np.ndarray | Tensor,
    inclusion_probs: Sequence[float] | np.ndarray | Tensor,
    p: int,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    raw_indices = (
        indices.detach().cpu().numpy() if isinstance(indices, Tensor) else indices
    )
    index_array = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
    if len(np.unique(index_array)) != len(index_array):
        raise ValueError("indices cannot contain duplicates")
    if index_array.size and (index_array.min() < 0 or index_array.max() >= p):
        raise ValueError(f"indices must lie in [0, {p})")
    index = torch.tensor(index_array, dtype=torch.long, device=reference.device)
    if isinstance(inclusion_probs, Tensor):
        pi = inclusion_probs.to(dtype=reference.dtype, device=reference.device)
    else:
        pi = torch.tensor(
            np.asarray(inclusion_probs).copy(),
            dtype=reference.dtype,
            device=reference.device,
        )
    if pi.shape != (p,) or not bool(torch.all((pi > 0) & (pi <= 1))):
        raise ValueError("inclusion_probs must have shape [p] and values in (0, 1]")
    return index, pi


def _linear_without_blas(x: Tensor, layer: nn.Linear) -> Tensor:
    """Evaluate a small linear layer without macOS Accelerate GEMM calls."""
    result = (x.unsqueeze(-2) * layer.weight).sum(dim=-1)
    return result if layer.bias is None else result + layer.bias


def _rowwise_projection(x: Tensor, rows: Tensor) -> Tensor:
    """Return ``x @ rows.T`` using stable elementwise operations."""
    return (x.unsqueeze(-2) * rows).sum(dim=-1)


@torch.no_grad()
def project_trainable_parameters_(
    model: nn.Module, lower: float, upper: float
) -> nn.Module:
    """Project every trainable parameter onto a compact scalar box."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("projection bounds must be finite and ordered")
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.clamp_(lower, upper)
    return model


class TimeDepWeights(nn.Module):
    """Hyper-network producing a matrix and, optionally, a bias."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int, *, bias: bool):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, input_dim * output_dim)
        )
        self.bias_net = (
            nn.Sequential(
                nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, output_dim)
            )
            if bias
            else None
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, t: Tensor | float) -> tuple[Tensor, Tensor | None]:
        reference = next(self.weight_net.parameters())
        t_input = _scalar_time(t, reference)
        weight_hidden = torch.tanh(
            _linear_without_blas(t_input, self.weight_net[0])
        )
        weight = _linear_without_blas(
            weight_hidden, self.weight_net[2]
        ).reshape(self.output_dim, self.input_dim)
        bias = (
            None
            if self.bias_net is None
            else _linear_without_blas(
                torch.tanh(_linear_without_blas(t_input, self.bias_net[0])),
                self.bias_net[2],
            ).reshape(self.output_dim)
        )
        return weight, bias


class TimeDepODE(nn.Module):
    r"""Paper field ``sum_i w_i(t) GeLU(a_i(t)^T x + b_i(t))``."""

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int = 2,
        net_hidden: int = 20,
        parameter_box: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.layer1 = TimeDepWeights(input_dim, hidden_dim, net_hidden, bias=True)
        self.layer2 = TimeDepWeights(hidden_dim, input_dim, net_hidden, bias=False)
        self.activation = nn.GELU()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.parameter_box = parameter_box

    def control_parameters(self, t: Tensor | float) -> tuple[Tensor, Tensor, Tensor]:
        A, b = self.layer1(t)
        W, unused_bias = self.layer2(t)
        assert b is not None and unused_bias is None
        return A, b, W

    def neuron_contributions(self, t: Tensor | float, x: Tensor) -> Tensor:
        A, b, W = self.control_parameters(t)
        activation = self.activation(_rowwise_projection(x, A) + b)
        return activation.unsqueeze(-1) * W.t().unsqueeze(0)

    def forward(self, t: Tensor | float, x: Tensor) -> Tensor:
        return self.neuron_contributions(t, x).sum(dim=1)

    def forward_batch(
        self,
        t: Tensor | float,
        x: Tensor,
        indices,
        inclusion_probs,
    ) -> Tensor:
        index, pi = _batch_tensors(indices, inclusion_probs, self.hidden_dim, x)
        A, b, W = self.control_parameters(t)
        activation = self.activation(_rowwise_projection(x, A[index]) + b[index])
        scaled_weights = W[:, index].t() / pi[index, None]
        return (activation.unsqueeze(-1) * scaled_weights.unsqueeze(0)).sum(dim=1)

    def project_parameters_(
        self, lower: float | None = None, upper: float | None = None
    ) -> "TimeDepODE":
        if lower is None or upper is None:
            if self.parameter_box is None:
                raise ValueError("set parameter_box or pass projection bounds")
            lower = self.parameter_box[0] if lower is None else lower
            upper = self.parameter_box[1] if upper is None else upper
        project_trainable_parameters_(self, float(lower), float(upper))
        return self
