"""
Neural ODE and Random Batch Neural ODE (rNODE) models.

This module contains the core model architectures used in the experiments:
- Constant-parameter Neural ODEs
- Time-dependent Neural ODEs  
- Random Batch Methods (RBM) for efficient inference
- Trainable RBM architectures
"""

import torch
import torch.nn as nn
from bisect import bisect_right


# =============================================================================
# Constant-Parameter Neural ODE
# =============================================================================

class ConstantODE(nn.Module):
    """Neural ODE with constant (time-independent) parameters."""
    
    def __init__(self, hidden_dim: int, input_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.activation = nn.ReLU()

    def forward(self, t, y):
        z = self.activation(self.fc1(y))
        return self.fc2(z)


# =============================================================================
# Time-Dependent Neural ODE
# =============================================================================

class TimeDepWeights(nn.Module):
    """Network that outputs weights and biases as functions of time t."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 20):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, input_dim * output_dim)
        )
        self.bias_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim)
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, t):
        t_in = t.view(1, 1)
        W = self.weight_net(t_in).view(self.output_dim, self.input_dim)
        b = self.bias_net(t_in).view(self.output_dim)
        return W, b


class TimeDepODE(nn.Module):
    """Neural ODE with time-dependent parameters W(t), b(t)."""
    
    def __init__(self, hidden_dim: int, input_dim: int = 2, net_hidden: int = 20):
        super().__init__()
        torch.manual_seed(42)
        self.layer1 = TimeDepWeights(input_dim, hidden_dim, net_hidden)
        torch.manual_seed(42)
        self.layer2 = TimeDepWeights(hidden_dim, input_dim, net_hidden)
        self.activation = nn.ReLU()

    def forward(self, t, y):
        W1, b1 = self.layer1(t)
        W2, _ = self.layer2(t)
        z = self.activation(y.mm(W1.t()) + b1)
        return z.mm(W2.t())


# =============================================================================
# Random Batch Method (RBM) - Inference from Trained Models
# =============================================================================

class RBMConstant(nn.Module):
    """
    RBM inference module for constant-parameter NODEs.
    
    Uses neuron batch selection for efficient forward pass while
    maintaining convergence to the full model in expectation.
    """
    
    def __init__(self, trained_func: ConstantODE, t_span: torch.Tensor,
                 list_batches: list, pi: float):
        super().__init__()
        self.W1 = trained_func.fc1.weight
        self.b1 = trained_func.fc1.bias
        self.W2 = trained_func.fc2.weight
        self.pi = pi
        self.t_vals = sorted(t_span.detach().cpu().tolist())
        self.list_batches = list_batches
        self.activation = trained_func.activation

    def forward(self, t, y):
        i = self._get_batch_idx(t)
        batch = self.list_batches[i]
        
        W1_b = self.W1[batch, :]
        b1_b = self.b1[batch]
        W2_b = self.W2[:, batch]
        
        z = y.mm(W1_b.t()) + b1_b
        a = self.activation(z)
        return a.mm(W2_b.t()) / self.pi

    def _get_batch_idx(self, t):
        t_val = float(t.item())
        i = bisect_right(self.t_vals, t_val) - 1
        return max(0, min(i, len(self.t_vals) - 1))


class RBMTimeDep(nn.Module):
    """
    RBM inference module for time-dependent NODEs.
    
    Applies random batch selection to time-varying weight networks.
    """
    
    def __init__(self, trained_func: TimeDepODE, t_span: torch.Tensor,
                 list_batches: list, pi: float):
        super().__init__()
        self.trained_func = trained_func
        self.pi = pi
        self.t_vals = sorted(t_span.detach().cpu().tolist())
        self.list_batches = list_batches
        self.activation = trained_func.activation

    def forward(self, t, y):
        i = self._get_batch_idx(t)
        batch = self.list_batches[i]
        
        W1, b1 = self.trained_func.layer1(t)
        W2, _ = self.trained_func.layer2(t)
        
        W1_b = W1[batch, :]
        b1_b = b1[batch]
        W2_b = W2[:, batch]
        
        z = y.mm(W1_b.t()) + b1_b
        a = self.activation(z)
        return a.mm(W2_b.t()) / self.pi

    def _get_batch_idx(self, t):
        t_val = float(t.item())
        i = bisect_right(self.t_vals, t_val) - 1
        return max(0, min(i, len(self.t_vals) - 1))


# =============================================================================
# Trainable Architectures — Experiment 3a (all parameters trainable)
# =============================================================================

class TimeDepODE_ELU(nn.Module):
    """Time-dependent Neural ODE with ELU activation (§5.3a)."""

    def __init__(self, hidden_dim: int, input_dim: int = 2, net_hidden: int = 10):
        super().__init__()
        self.layer1 = TimeDepWeights(input_dim, hidden_dim, net_hidden)
        self.layer2 = TimeDepWeights(hidden_dim, input_dim, net_hidden)
        self.activation = nn.ELU()
        self.hidden_dim = hidden_dim

    def forward(self, t, y):
        W1, b1 = self.layer1(t)
        W2, _ = self.layer2(t)
        z = self.activation(y.mm(W1.t()) + b1)
        return z.mm(W2.t())


class RBMTrainableODE(nn.Module):
    """Trainable RBM Neural ODE with a FIXED dropout schedule (§5.3a).

    Both layers are time-dependent and trainable. At each solver step the
    hidden layer is restricted to the neurons in the active batch and the
    output is rescaled by 1/pi (Horvitz–Thompson correction).

    The batch schedule is fixed at construction and reused across all
    training epochs, making the architecture equivalent to structured
    time-dependent pruning.
    """

    def __init__(self, hidden_dim: int, pi: float, list_batches: list,
                 t_span: torch.Tensor, input_dim: int = 2,
                 net_hidden: int = 10):
        super().__init__()
        self.layer1 = TimeDepWeights(input_dim, hidden_dim, net_hidden)
        self.layer2 = TimeDepWeights(hidden_dim, input_dim, net_hidden)
        self.activation = nn.ELU()
        self.hidden_dim = hidden_dim
        self.pi = pi
        self.list_batches = list_batches
        self.t_vals = t_span.detach().cpu().tolist()

    def forward(self, t, y):
        idx = self._get_batch_idx(t)
        batch = self.list_batches[idx]

        W1, b1 = self.layer1(t)
        W2, _ = self.layer2(t)

        W1_b = W1[batch, :]
        b1_b = b1[batch]
        W2_b = W2[:, batch]

        z = self.activation(y.mm(W1_b.t()) + b1_b)
        return z.mm(W2_b.t()) / self.pi

    def _get_batch_idx(self, t):
        t_val = float(t.item()) if t.dim() == 0 else float(t)
        i = bisect_right(self.t_vals, t_val) - 1
        return max(0, min(i, len(self.list_batches) - 1))


# =============================================================================
# Trainable Architectures — Experiment 3b (fixed inner weights)
# =============================================================================

class TimeDependentLinear(nn.Module):
    """Time-dependent linear layer: t -> W(t) (no bias)."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, input_dim * output_dim),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, t):
        t_in = t.view(1, 1)
        return self.net(t_in).view(self.output_dim, self.input_dim)


class FixedInnerODE(nn.Module):
    """Neural ODE with FIXED inner weights (§5.3b).

    dx/dt = W_2(t) · sigma(W_1 x + b_1)

    W_1, b_1 are randomly initialised and frozen (not trainable).
    Only the time-dependent outer layer W_2(t) is trained.
    """

    def __init__(self, hidden_dim: int, input_dim: int = 2, seed: int = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.5,
                               requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(hidden_dim) * 0.5,
                               requires_grad=False)
        self.layer2 = TimeDependentLinear(hidden_dim, input_dim)
        self.activation = nn.ELU()
        self.hidden_dim = hidden_dim

    def forward(self, t, x):
        z = self.activation(x.mm(self.W1.t()) + self.b1)
        W2 = self.layer2(t)
        return z.mm(W2.t())


class FixedInnerRBM(nn.Module):
    """RBM Neural ODE with FIXED inner weights (§5.3b).

    Same architecture as ``FixedInnerODE`` but with neuron dropout
    controlled by a fixed batch schedule.
    """

    def __init__(self, hidden_dim: int, pi: float, list_batches: list,
                 t_span: torch.Tensor, input_dim: int = 2,
                 seed: int = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.5,
                               requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(hidden_dim) * 0.5,
                               requires_grad=False)
        self.layer2 = TimeDependentLinear(hidden_dim, input_dim)
        self.activation = nn.ELU()
        self.hidden_dim = hidden_dim
        self.pi = pi
        self.list_batches = list_batches
        self.t_vals = t_span.detach().cpu().tolist()

    def forward(self, t, x):
        idx = self._get_batch_idx(t)
        batch = self.list_batches[idx]

        W1_b = self.W1[batch, :]
        b1_b = self.b1[batch]
        z = self.activation(x.mm(W1_b.t()) + b1_b)

        W2 = self.layer2(t)
        W2_b = W2[:, batch]
        return z.mm(W2_b.t()) / self.pi

    def _get_batch_idx(self, t):
        t_val = float(t.item()) if t.dim() == 0 else float(t)
        i = bisect_right(self.t_vals, t_val) - 1
        return max(0, min(i, len(self.list_batches) - 1))
