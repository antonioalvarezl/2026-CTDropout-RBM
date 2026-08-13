import numpy as np
import pytest
import torch

from rnode.flow import Flow
from rnode.models import TimeDepODE, project_trainable_parameters_


@pytest.mark.parametrize("model", [TimeDepODE(4), Flow(hidden=4)])
def test_neuron_contributions_sum_to_full_field(model):
    x = torch.tensor([[0.2, -0.1], [0.5, 0.7]])
    contributions = model.neuron_contributions(torch.tensor(0.4), x)
    assert contributions.shape == (2, 4, 2)
    torch.testing.assert_close(contributions.sum(dim=1), model(0.4, x))


@pytest.mark.parametrize("model", [TimeDepODE(3), Flow(hidden=3)])
def test_empty_batch_is_a_differentiable_zero(model):
    x = torch.randn(5, 2, requires_grad=True)
    result = model.forward_batch(0.2, x, [], np.ones(3))
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
    result.sum().backward()
    assert x.grad is not None


def test_time_dependent_paper_architecture_and_control_shapes():
    model = TimeDepODE(hidden_dim=7, input_dim=2)
    assert isinstance(model.activation, torch.nn.GELU)
    assert model.layer2.bias_net is None
    A, b, W = model.control_parameters(torch.tensor(0.5))
    assert A.shape == (7, 2)
    assert b.shape == (7,)
    assert W.shape == (2, 7)


def test_blas_free_time_dependent_evaluation_matches_linear_layers():
    torch.manual_seed(11)
    model = TimeDepODE(hidden_dim=5, input_dim=2, net_hidden=3).double()
    t = torch.tensor([[0.37]], dtype=torch.float64)
    expected_A = model.layer1.weight_net(t).reshape(5, 2)
    expected_b = model.layer1.bias_net(t).reshape(5)
    expected_W = model.layer2.weight_net(t).reshape(2, 5)
    A, b, W = model.control_parameters(t)
    torch.testing.assert_close(A, expected_A)
    torch.testing.assert_close(b, expected_b)
    torch.testing.assert_close(W, expected_W)


def test_flow_matches_exact_paper_formula():
    model = Flow(dim=2, hidden=2)
    A0 = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    b0 = torch.tensor([0.2, -0.1])
    b1 = torch.tensor([0.4, -0.3])
    W0 = torch.tensor([[1.0, -2.0], [0.5, 0.7]])
    with torch.no_grad():
        model.net[0].weight[:, 0] = b1
        model.net[0].weight[:, 1:] = A0
        model.net[0].bias.copy_(b0)
        model.net[2].weight.copy_(W0)
    x = torch.tensor([[0.3, -0.2], [0.0, 0.8]])
    expected = torch.tanh(x @ A0.T + b0 + 0.6 * b1) @ W0.T
    torch.testing.assert_close(model(torch.tensor(0.6), x), expected)


def test_flow_analytic_divergence_matches_autograd_trace():
    torch.manual_seed(2)
    model = Flow(hidden=5).double()
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    velocity = model(0.3, x)
    trace = sum(
        torch.autograd.grad(velocity[:, j].sum(), x, retain_graph=True)[0][:, j]
        for j in range(2)
    )
    torch.testing.assert_close(model.divergence(0.3, x), trace)


def test_parameter_projection_clamps_trainable_parameters():
    model = TimeDepODE(5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(10.0)
    project_trainable_parameters_(model, -0.25, 0.25)
    assert all(bool(torch.all(parameter <= 0.25)) for parameter in model.parameters())
