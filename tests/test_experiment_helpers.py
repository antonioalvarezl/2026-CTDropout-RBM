import numpy as np

from experiments._paper_common import ordered_trajectory_statistic
from experiments.exp5_training_consistency import _slope_row


def test_trajectory_statistic_averages_schedules_before_time_maximum():
    # The two schedules peak at different times.  The theorem-ordered value is
    # max_t mean_k = 5, whereas the prohibited mean_k max_t would equal 10.
    squared_errors = np.array(
        [
            [[10.0], [0.0]],
            [[0.0], [10.0]],
        ]
    )
    assert ordered_trajectory_statistic(squared_errors) == 5.0


def test_weak_slope_is_not_fit_with_fewer_than_three_significant_points():
    row = _slope_row(
        "weak",
        np.array([0.5, 0.25, 0.125]),
        np.array([0.2, 0.1, 0.05]),
        np.ones((5, 3)),
        np.array([True, False, True]),
        "bias not distinguishable",
    )
    assert row["fit_performed"] is False
    assert np.isnan(row["slope"])
