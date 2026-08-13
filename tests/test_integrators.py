import numpy as np
import pytest
import torch

from rnode.integrators import integrate_fixed_grid, integrate_masked_ensemble
from rnode.models import TimeDepODE


class TrackingField:
    def __init__(self):
        self.calls = []

    def __call__(self, t, x):
        self.calls.append(None)
        return x * 0.0

    def forward_batch(self, t, x, indices, inclusion_probs):
        self.calls.append(tuple(np.asarray(indices).tolist()))
        return x * 0.0


@pytest.mark.parametrize("method, stages", [("euler", 1), ("rk4", 4)])
def test_integrator_keeps_batch_fixed_inside_each_step(method, stages):
    model = TrackingField()
    schedule = [np.array([0]), np.array([1])]
    times, trajectory = integrate_fixed_grid(
        model,
        torch.ones(2, 2),
        T=1.0,
        dt=0.25,
        h=0.5,
        schedule=schedule,
        inclusion_probs=np.ones(2),
        method=method,
    )
    assert model.calls == [(0,)] * (2 * stages) + [(1,)] * (2 * stages)
    assert times.shape == (5,)
    assert trajectory.shape == (5, 2, 2)


def test_integrator_rejects_unaligned_switching_and_numerical_grids():
    model = TimeDepODE(2)
    x0 = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="T / h"):
        integrate_fixed_grid(model, x0, T=1.0, dt=0.1, h=0.3)
    with pytest.raises(ValueError, match="h / dt"):
        integrate_fixed_grid(model, x0, T=1.0, dt=0.07, h=0.25)


def test_full_field_uses_same_interface_and_remains_differentiable():
    model = TimeDepODE(3)
    x0 = torch.randn(4, 2)
    _, trajectory = integrate_fixed_grid(model, x0, T=0.2, dt=0.1, h=0.2, method="rk4")
    trajectory[-1].square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_masked_ensemble_matches_individual_aligned_integrations():
    torch.manual_seed(8)
    model = TimeDepODE(3).double()
    x0 = torch.randn(2, 2, dtype=torch.float64)
    schedules = [
        [np.array([0, 1]), np.array([2])],
        [np.array([1]), np.array([0, 2])],
    ]
    masks = np.zeros((2, 2, 3))
    for ensemble, schedule in enumerate(schedules):
        for interval, batch in enumerate(schedule):
            masks[ensemble, interval, batch] = 1.0
    pi = np.array([0.5, 0.75, 0.5])
    times, ensemble_trajectory = integrate_masked_ensemble(
        model, x0, 1.0, 0.25, 0.5, masks, pi, method="rk4"
    )
    for index, schedule in enumerate(schedules):
        individual_times, individual_trajectory = integrate_fixed_grid(
            model,
            x0,
            1.0,
            0.25,
            0.5,
            schedule,
            inclusion_probs=pi,
            method="rk4",
        )
        torch.testing.assert_close(times, individual_times)
        torch.testing.assert_close(ensemble_trajectory[:, index], individual_trajectory)


def test_rk4_k4_at_switching_boundary_uses_previous_batch():
    class TimeTrackingField:
        def __init__(self):
            self.calls = []

        def forward_batch(self, t, x, indices, inclusion_probs):
            self.calls.append((float(t), int(np.asarray(indices)[0])))
            return torch.zeros_like(x)

    model = TimeTrackingField()
    integrate_fixed_grid(
        model,
        torch.ones(1, 1, dtype=torch.float64),
        T=1.0,
        dt=0.5,
        h=0.5,
        schedule=[np.array([0]), np.array([1])],
        inclusion_probs=np.ones(2),
        method="rk4",
    )
    assert model.calls[:4] == [(0.0, 0), (0.25, 0), (0.25, 0), (0.5, 0)]
    assert model.calls[4:] == [(0.5, 1), (0.75, 1), (0.75, 1), (1.0, 1)]


def test_full_rk4_error_decreases_at_fourth_order():
    class ExponentialField:
        def __call__(self, t, x):
            return x

    errors = []
    for dt in (0.2, 0.1, 0.05):
        _, trajectory = integrate_fixed_grid(
            ExponentialField(),
            torch.ones(1, 1, dtype=torch.float64),
            T=1.0,
            dt=dt,
            h=1.0,
            method="rk4",
        )
        errors.append(abs(float(trajectory[-1, 0, 0]) - np.e))
    assert errors[0] / errors[1] > 12
    assert errors[1] / errors[2] > 12
