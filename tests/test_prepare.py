"""Tests for the cleaning and summary helpers in running_coach.data.prepare.

They use tiny hand-made tables, so they don't need the downloaded dataset.
"""

import pandas as pd
import pytest

from running_coach.data.prepare import (
    athletes_with_enough_runs,
    clean_runs,
    summarize_athletes,
)


def make_days(rows: list[tuple[str, float, float]], athlete: int = 1) -> pd.DataFrame:
    """Build a raw-style table from (date, distance_km, duration_min) tuples."""
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime([date for date, _, _ in rows]),
            "athlete": athlete,
            "distance": [distance for _, distance, _ in rows],
            "duration": [duration for _, _, duration in rows],
            "gender": "F",
        }
    )


def test_good_run_is_kept_with_correct_pace() -> None:
    days = make_days([("2019-01-07", 10.0, 55.0)])
    runs = clean_runs(days)
    assert len(runs) == 1
    assert runs.loc[0, "pace"] == pytest.approx(5.5)


@pytest.mark.parametrize(
    "distance, duration, reason",
    [
        (0.0, 0.0, "rest day"),
        (0.3, 2.0, "shorter than 0.5 km"),
        (120.0, 700.0, "more than 100 km"),
        (10.0, 20.0, "pace 2.0 min/km, faster than world record"),
        (5.0, 100.0, "pace 20 min/km, walking"),
    ],
)
def test_impossible_day_is_removed(distance: float, duration: float, reason: str) -> None:
    days = make_days([("2019-01-07", distance, duration)])
    assert clean_runs(days).empty, reason


def test_values_exactly_on_the_limits_are_kept() -> None:
    days = make_days(
        [
            ("2019-01-07", 0.5, 5.0),  # minimum distance
            ("2019-01-08", 100.0, 600.0),  # maximum distance
            ("2019-01-09", 10.0, 25.0),  # pace 2.5, fastest allowed
            ("2019-01-10", 10.0, 150.0),  # pace 15, slowest allowed
        ]
    )
    assert len(clean_runs(days)) == 4


def test_athletes_with_enough_runs() -> None:
    few = make_days([("2019-01-07", 5.0, 30.0)] * 3, athlete=1)
    many = make_days([("2019-01-07", 5.0, 30.0)] * 10, athlete=2)
    runs = clean_runs(pd.concat([few, many]))
    assert list(athletes_with_enough_runs(runs, min_runs=10)) == [2]


def test_summary_counts_rest_weeks_between_runs_as_zero() -> None:
    # Week 1: 2 runs, 20 km in 120 min. Weeks 2 and 3: rest. Week 4: 1 run, 20 km in 80 min.
    days = make_days(
        [
            ("2019-01-07", 10.0, 60.0),
            ("2019-01-09", 10.0, 60.0),
            ("2019-01-28", 20.0, 80.0),
        ]
    )
    summary = summarize_athletes(clean_runs(days)).loc[1]
    assert summary["weekly_km"] == pytest.approx(10.0)  # 40 km / 4 weeks
    assert summary["runs_per_week"] == pytest.approx(0.75)  # 3 runs / 4 weeks
    assert summary["pace"] == pytest.approx(5.0)  # 200 min / 40 km
    assert summary["level"] == "10-20 km"  # 10.0 belongs to the higher band
