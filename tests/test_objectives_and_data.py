import numpy as np
import pytest
import torch

from rnode.data import (
    TARGET_MIXTURE_CENTERS,
    TARGET_MIXTURE_COVARIANCES,
    TARGET_MIXTURE_WEIGHTS,
    initial_density_quadrature,
    make_circles_with_targets,
    sample_initial_compact,
    sample_target,
)
from rnode.models import TimeDepODE
from rnode.objectives import (
    control_regularization,
    nearest_target_accuracy,
    paper_objective,
    trapezoidal_integral,
)


class ConstantControl:
    def control_parameters(self, t):
        return torch.ones(2, 2), torch.ones(2), torch.ones(2, 2)


def test_constant_integrands_give_expected_paper_objective():
    times = torch.linspace(0.0, 2.0, 5)
    trajectory = torch.ones(5, 3, 2)
    targets = torch.zeros(3, 2)
    terms = paper_objective(
        trajectory,
        times,
        targets,
        ConstantControl(),
        alpha=0.2,
        beta=0.5,
    )
    assert terms.terminal == pytest.approx(6.0)
    assert terms.running == pytest.approx(12.0)
    assert terms.control == pytest.approx(20.0)
    assert terms.total == pytest.approx(14.0)


def test_control_regularization_is_grid_invariant_for_constant_control():
    coarse = control_regularization(ConstantControl(), torch.linspace(0.0, 1.0, 3))
    fine = control_regularization(ConstantControl(), torch.linspace(0.0, 1.0, 101))
    assert coarse == pytest.approx(10.0)
    torch.testing.assert_close(coarse, fine)


def test_objective_regularizes_actual_control_not_hypernetwork_parameters():
    torch.manual_seed(7)
    model = TimeDepODE(4)
    times = torch.linspace(0.0, 1.0, 9)
    energies = []
    for time in times:
        A, b, W = model.control_parameters(time)
        energies.append(A.square().sum() + b.square().sum() + W.square().sum())
    expected = trapezoidal_integral(torch.stack(energies), times)
    torch.testing.assert_close(control_regularization(model, times), expected)


def test_nearest_target_accuracy_does_not_use_first_coordinate_sign():
    predictions = torch.tensor([[0.2, 0.9], [-0.6, 0.1]])
    labels = torch.tensor([[1.0], [-1.0]])
    assert nearest_target_accuracy(predictions, labels) == 1.0


def test_classification_data_returns_paper_targets():
    _, labels, targets = make_circles_with_targets(20, seed=5)
    torch.testing.assert_close(
        targets[labels.squeeze() == -1],
        torch.tensor([[-1.0, 0.0]]).expand(10, 2),
    )
    torch.testing.assert_close(
        targets[labels.squeeze() == 1],
        torch.tensor([[0.0, 1.0]]).expand(10, 2),
    )


def test_compact_initial_sampler_has_correct_support_and_radial_moment():
    samples = sample_initial_compact(50_000, rng=np.random.default_rng(12))
    radius_squared = np.sum((samples + 1.0) ** 2, axis=1)
    assert np.all(radius_squared < 1.0)
    # For density proportional to 1-r^2 in two dimensions, E[r^2] = 1/3.
    assert radius_squared.mean() == pytest.approx(1.0 / 3.0, abs=0.006)


def test_initial_quadrature_masses_are_normalized():
    points, masses = initial_density_quadrature(101)
    assert points.shape[1] == 2
    assert bool(torch.all(masses > 0))
    assert masses.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_target_mixture_parameters_and_sampling_are_normalized():
    np.testing.assert_allclose(TARGET_MIXTURE_WEIGHTS, np.full(3, 1.0 / 3.0))
    assert TARGET_MIXTURE_CENTERS.shape == (3, 2)
    assert TARGET_MIXTURE_COVARIANCES.shape == (3, 2, 2)
    samples = sample_target(30, rng=np.random.default_rng(1))
    assert samples.shape == (30, 2)
