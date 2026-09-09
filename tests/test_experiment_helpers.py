import numpy as np

from experiments._paper_common import ordered_trajectory_statistic


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
