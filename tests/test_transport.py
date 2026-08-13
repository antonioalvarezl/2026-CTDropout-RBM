import numpy as np
import pytest
import torch

from rnode.data import initial_density_quadrature
from rnode.flow import Flow
from rnode.transport import (
    average_densities,
    l1_density_error,
    transport_particles,
    weighted_kde,
    weighted_kde_with_diagnostics,
)
from experiments.exp4_measure_transport import assess_flow_quality


def test_transport_preserves_every_dirac_mass():
    points, masses = initial_density_quadrature(15)
    model = Flow(hidden=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    _, trajectory, transported_masses = transport_particles(
        model, points, masses, 1.0, 0.25, 1.0
    )
    torch.testing.assert_close(transported_masses, masses)
    torch.testing.assert_close(trajectory[-1], points)
    assert transported_masses.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_average_of_densities_is_not_kde_of_average_particles():
    x = y = np.linspace(-3.0, 3.0, 121)
    masses = np.array([1.0])
    left = weighted_kde([[-1.0, 0.0]], masses, x, y, 0.2)
    right = weighted_kde([[1.0, 0.0]], masses, x, y, 0.2)
    correct = average_densities([left, right], x, y)
    incorrect = weighted_kde([[0.0, 0.0]], masses, x, y, 0.2)
    assert l1_density_error(correct, incorrect, x, y) > 1.5


def test_l1_error_is_zero_for_identical_densities():
    grid = np.linspace(-2.0, 2.0, 41)
    density = weighted_kde([[0.0, 0.0]], [1.0], grid, grid, 0.3)
    assert l1_density_error(density, density.copy(), grid, grid) == 0.0


def test_kde_is_stable_under_grid_refinement():
    points, masses = initial_density_quadrature(25)
    coarse = np.linspace(-2.5, 0.5, 81)
    fine = np.linspace(-2.5, 0.5, 161)
    coarse_density = weighted_kde(points, masses, coarse, coarse, 0.2)
    fine_density = weighted_kde(points, masses, fine, fine, 0.2)
    sampled_fine = fine_density[::2, ::2]
    assert l1_density_error(coarse_density, sampled_fine, coarse, coarse) < 0.01


def test_kde_diagnostics_measure_mass_before_normalization():
    narrow = np.linspace(-0.1, 0.1, 41)
    density, diagnostics = weighted_kde_with_diagnostics(
        [[0.0, 0.0]], [1.0], narrow, narrow, 0.5
    )
    normalized_mass = density.sum() * (narrow[1] - narrow[0]) ** 2
    assert normalized_mass == pytest.approx(1.0)
    assert diagnostics["mass_before_normalization"] < 0.05
    assert diagnostics["renormalization_factor"] == pytest.approx(
        1.0 / diagnostics["mass_before_normalization"]
    )
    assert diagnostics["estimated_truncation_loss"] > 0.95


def test_transport_rejects_negative_masses_even_when_they_sum_to_one():
    points = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="non-negative"):
        transport_particles(
            Flow(hidden=2), points, torch.tensor([1.2, -0.2]), 1, 0.5, 1
        )


def test_flow_quality_is_scaled_by_kde_resolution_benchmark():
    passing = assess_flow_quality(0.4, [0.2, 0.22], max_ratio=2.0)
    failing = assess_flow_quality(0.5, [0.2, 0.22], max_ratio=2.0)
    assert passing["passed"]
    assert not failing["passed"]
    assert passing["target_kde_benchmark_l1_mean"] == pytest.approx(0.21)


def test_flow_quality_rejects_invalid_benchmark():
    with pytest.raises(ValueError):
        assess_flow_quality(0.4, [0.0], max_ratio=2.0)
