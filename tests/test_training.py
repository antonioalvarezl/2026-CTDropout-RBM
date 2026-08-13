import copy

import numpy as np
import torch

from rnode.batches import make_uniform_fixed_size, sample_batch_sequence
from rnode.integrators import integrate_fixed_grid
from rnode.models import TimeDepODE
from rnode.objectives import paper_objective
from rnode.training import train_expected_objective


def test_expected_training_uses_one_model_and_fresh_schedules_per_step():
    torch.manual_seed(3)
    model = TimeDepODE(2, net_hidden=3, parameter_box=(-0.2, 0.2)).double()
    identity = id(model)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    features = torch.randn(4, 2, dtype=torch.float64)
    targets = torch.randn(4, 2, dtype=torch.float64)
    result = train_expected_objective(
        model,
        features,
        targets,
        scheme=make_uniform_fixed_size(2, 1),
        h=0.25,
        dt=0.25,
        T=0.25,
        epochs=2,
        m_train=2,
        learning_rate=1e-2,
        alpha=1e-3,
        beta=0.1,
        seed=19,
        control_dt=0.25,
    )
    assert id(result["model"]) == identity
    assert result["optimizer_steps"] == result["backward_calls"] == 2
    assert result["schedule_seeds"][0] != result["schedule_seeds"][1]
    assert result["neuron_evaluations"] == 16
    assert any(
        not torch.equal(initial[key], value)
        for key, value in model.state_dict().items()
    )
    assert all(
        bool(torch.all(parameter.abs() <= 0.2)) for parameter in model.parameters()
    )


def test_schedule_seed_stream_is_reproducible_without_global_numpy_state():
    def train_once():
        with torch.random.fork_rng():
            torch.manual_seed(8)
            model = TimeDepODE(2, net_hidden=3, parameter_box=(-1.0, 1.0)).double()
        return train_expected_objective(
            model,
            torch.zeros(2, 2, dtype=torch.float64),
            torch.ones(2, 2, dtype=torch.float64),
            scheme=make_uniform_fixed_size(2, 1),
            h=0.25,
            dt=0.25,
            T=0.25,
            epochs=2,
            m_train=2,
            learning_rate=1e-3,
            alpha=0.0,
            beta=0.0,
            seed=27,
        )["schedule_seeds"]

    np.random.seed(999)
    assert train_once() == train_once()


def test_training_history_uses_paper_objective_divided_globally_by_n_data():
    torch.manual_seed(21)
    model = TimeDepODE(2, net_hidden=3, parameter_box=(-10.0, 10.0)).double()
    reference_model = copy.deepcopy(model)
    features = torch.randn(3, 2, dtype=torch.float64)
    targets = torch.randn(3, 2, dtype=torch.float64)
    scheme = make_uniform_fixed_size(2, 1)
    seed = 93
    epoch_seed = int(
        np.random.default_rng(seed).integers(0, 2**32 - 1, 1, dtype=np.uint32)[0]
    )
    schedule = sample_batch_sequence(
        scheme, 2, np.random.default_rng(epoch_seed)
    )
    times, trajectory = integrate_fixed_grid(
        reference_model,
        features,
        T=0.5,
        dt=0.25,
        h=0.25,
        schedule=schedule,
        inclusion_probs=scheme.inclusion_probs,
        method="rk4",
    )
    terms = paper_objective(
        trajectory,
        times,
        targets,
        reference_model,
        alpha=0.2,
        beta=0.3,
        control_times=times,
    )
    expected = float((terms.total / len(features)).detach())
    result = train_expected_objective(
        model,
        features,
        targets,
        scheme=scheme,
        h=0.25,
        dt=0.25,
        T=0.5,
        epochs=1,
        m_train=1,
        learning_rate=0.0,
        alpha=0.2,
        beta=0.3,
        seed=seed,
        control_dt=0.25,
    )
    assert result["history"][0]["expected_objective_estimate"] == expected
