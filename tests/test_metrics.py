"""Tests for the analysis functions in running_coach.analysis.metrics.

Each test uses a tiny hand-made table, so the expected answer can be checked by hand.
Dates: 2019-01-01 is a Tuesday. With end date 2019-01-14, the last 7-day week is
01-08..01-14 and the week before is 01-01..01-07.
"""

import json

import pandas as pd
import pytest

from running_coach.analysis.metrics import (
    compare_to_peers,
    consistency,
    format_pace,
    load_ramp,
    longest_run,
    pace_trend,
    weekly_volume,
)


def make_runs(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Build a runs table from (date, distance_km, duration_min) tuples."""
    runs = pd.DataFrame(rows, columns=["date", "distance_km", "duration_min"])
    runs["date"] = pd.to_datetime(runs["date"])
    runs["pace_min_km"] = runs["duration_min"] / runs["distance_km"]
    return runs


NO_RUNS = make_runs([])


# ---------- format_pace ----------


@pytest.mark.parametrize(
    "pace, text",
    [(5.5, "5:30"), (6.0, "6:00"), (4.25, "4:15"), (4.999, "5:00")],
)
def test_format_pace(pace: float, text: str) -> None:
    assert format_pace(pace) == text


# ---------- weekly_volume ----------


def test_weekly_volume_counts_km_and_change() -> None:
    # Week before: 2 x 5 km = 10 km. Last week: 3 x 5 km = 15 km. Change = +50%.
    runs = make_runs(
        [
            ("2019-01-02", 5, 30),
            ("2019-01-04", 5, 30),
            ("2019-01-09", 5, 30),
            ("2019-01-11", 5, 30),
            ("2019-01-14", 5, 30),
        ]
    )
    result = weekly_volume(runs)
    assert result["status"] == "ok"
    assert [week["km"] for week in result["weekly_km"]] == [10.0, 15.0]
    assert result["total_km"] == 25.0
    assert result["last_week_km"] == 15.0
    assert result["previous_week_km"] == 10.0
    assert result["change_pct"] == 50.0
    assert result["period"] == {"start": "2019-01-01", "end": "2019-01-14", "weeks_analysed": 2}


def test_weekly_volume_rest_week_counts_as_zero() -> None:
    # 3 weeks: 5 km, a rest week, 5 km. The previous week is 0, so no % change.
    runs = make_runs([("2019-01-01", 5, 30), ("2019-01-15", 5, 30)])
    result = weekly_volume(runs)
    assert [week["km"] for week in result["weekly_km"]] == [5.0, 0.0, 5.0]
    assert result["change_pct"] is None


def test_weekly_volume_keeps_only_last_n_weeks() -> None:
    runs = make_runs([("2019-01-01", 5, 30), ("2019-01-08", 6, 36), ("2019-01-15", 7, 42)])
    result = weekly_volume(runs, weeks=2)
    assert result["period"]["weeks_analysed"] == 2
    assert [week["km"] for week in result["weekly_km"]] == [6.0, 7.0]
    assert result["total_km"] == 13.0  # the 5 km of the older week is not counted


def test_weekly_volume_single_week_is_not_enough() -> None:
    runs = make_runs([("2019-01-01", 5, 30), ("2019-01-03", 5, 30)])
    assert weekly_volume(runs)["status"] == "not_enough_data"


def test_weekly_volume_no_runs_is_not_enough() -> None:
    assert weekly_volume(NO_RUNS)["status"] == "not_enough_data"


# ---------- pace_trend ----------


@pytest.mark.parametrize(
    "durations, trend, slope",
    [
        ([60, 59, 58], "improving", -6.0),  # 6:00 -> 5:54 -> 5:48 per km: 6 s faster each week
        ([60, 60, 60], "stable", 0.0),
        ([60, 61, 62], "getting worse", 6.0),
    ],
)
def test_pace_trend_direction(durations: list[float], trend: str, slope: float) -> None:
    # One 10 km run per week, so pace = duration / 10.
    dates = ["2019-01-01", "2019-01-08", "2019-01-15"]
    runs = make_runs([(d, 10, minutes) for d, minutes in zip(dates, durations)])
    result = pace_trend(runs)
    assert result["trend"] == trend
    assert result["slope_sec_per_km_per_week"] == pytest.approx(slope)


def test_pace_trend_keeps_gaps_between_weeks() -> None:
    # Runs in weeks 0, 2 and 4 (rest weeks between). Pace 6.0 -> 5.9 -> 5.8 over 4 weeks
    # = -0.05 min/km per week = -3 s/km per week. Ignoring the gaps would wrongly give -6.
    runs = make_runs([("2019-01-01", 10, 60), ("2019-01-15", 10, 59), ("2019-01-29", 10, 58)])
    result = pace_trend(runs)
    assert result["slope_sec_per_km_per_week"] == pytest.approx(-3.0)
    assert result["change_over_period_sec_per_km"] == pytest.approx(-12.0)
    assert result["weeks_with_runs"] == 3


def test_pace_trend_two_weeks_is_not_enough() -> None:
    runs = make_runs([("2019-01-01", 10, 60), ("2019-01-08", 10, 55)])
    assert pace_trend(runs)["status"] == "not_enough_data"


# ---------- consistency ----------


def test_consistency_counts_runs_and_zero_weeks() -> None:
    # Week 1: 2 runs, week 2: 0 runs, week 3: 3 runs. Average = 5 / 3 = 1.7.
    runs = make_runs(
        [
            ("2019-01-01", 5, 30),
            ("2019-01-03", 5, 30),
            ("2019-01-16", 5, 30),
            ("2019-01-18", 5, 30),
            ("2019-01-21", 5, 30),
        ]
    )
    result = consistency(runs)
    assert result["runs_per_week"] == [2, 0, 3]
    assert result["average_runs_per_week"] == 1.7
    assert result["weeks_with_zero_runs"] == 1


def test_consistency_single_run_works() -> None:
    result = consistency(make_runs([("2019-01-01", 5, 30)]))
    assert result["status"] == "ok"
    assert result["runs_per_week"] == [1]


def test_consistency_end_date_before_first_run_is_not_enough() -> None:
    runs = make_runs([("2019-01-10", 5, 30)])
    assert consistency(runs, end_date="2019-01-01")["status"] == "not_enough_data"


# ---------- load_ramp ----------

# One run per 7-day week, ending on 2019-01-28: exactly 28 days of history.
FOUR_WEEK_DATES = ["2019-01-01", "2019-01-08", "2019-01-15", "2019-01-28"]


@pytest.mark.parametrize(
    "weekly_km, ratio, label",
    [
        ([10, 10, 10, 10], 1.0, "normal"),  # 10 / 10
        ([10, 10, 10, 30], 2.0, "high risk"),  # 30 / ((10+10+10+30) / 4 = 15)
        ([20, 20, 20, 4], 0.25, "low"),  # 4 / 16
        ([9, 9, 9, 13], 1.3, "normal"),  # exactly on the upper limit: still normal
        ([12, 10, 10, 8], 0.8, "normal"),  # exactly on the lower limit: still normal
    ],
)
def test_load_ramp_ratio_and_label(weekly_km: list[float], ratio: float, label: str) -> None:
    runs = make_runs([(d, km, km * 6) for d, km in zip(FOUR_WEEK_DATES, weekly_km)])
    result = load_ramp(runs)
    assert result["ratio"] == ratio
    assert result["label"] == label


def test_load_ramp_less_than_28_days_is_not_enough() -> None:
    # 2019-01-02 to 2019-01-28 is only 27 days.
    runs = make_runs([("2019-01-02", 10, 60), ("2019-01-28", 10, 60)])
    assert load_ramp(runs)["status"] == "not_enough_data"


def test_load_ramp_no_runs_in_last_28_days_is_not_enough() -> None:
    runs = make_runs([("2019-01-01", 10, 60)])
    assert load_ramp(runs, end_date="2019-06-01")["status"] == "not_enough_data"


# ---------- longest_run ----------


def test_longest_run_and_share_of_week() -> None:
    # Week: 5 + 10 + 5 = 20 km. Longest = 10 km = 50% of the week.
    runs = make_runs([("2019-01-01", 5, 30), ("2019-01-03", 10, 55), ("2019-01-05", 5, 30)])
    result = longest_run(runs)
    assert result["date"] == "2019-01-03"
    assert result["distance_km"] == 10.0
    assert result["week_km"] == 20.0
    assert result["share_of_week_pct"] == 50.0
    assert result["pace"] == "5:30"


def test_longest_run_ignores_runs_before_the_window() -> None:
    # The 30 km run is 10 weeks before the end, outside the last 2 weeks.
    runs = make_runs([("2018-11-01", 30, 180), ("2019-01-03", 8, 48), ("2019-01-10", 6, 36)])
    result = longest_run(runs, weeks=2)
    assert result["distance_km"] == 8.0


def test_longest_run_no_runs_in_window_is_not_enough() -> None:
    runs = make_runs([("2019-01-01", 10, 60)])
    assert longest_run(runs, end_date="2019-06-01")["status"] == "not_enough_data"


# ---------- compare_to_peers ----------

REFERENCE = pd.DataFrame(
    {
        "level": ["10-20 km"],
        "n_athletes": [100],
        "weekly_km_p25": [12.0],
        "weekly_km_p50": [15.0],
        "weekly_km_p75": [18.0],
        "pace_p25": [5.0],
        "pace_p50": [5.5],
        "pace_p75": [5.9],
        "runs_per_week_p25": [1.0],
        "runs_per_week_p50": [1.5],
        "runs_per_week_p75": [2.0],
    }
)


def two_weeks_of_runs(minutes_per_5_km: float) -> pd.DataFrame:
    """2 Mon-Sun weeks with 2 runs of 5 km each: 10 km and 2 runs per week."""
    dates = ["2019-01-07", "2019-01-10", "2019-01-14", "2019-01-20"]
    return make_runs([(d, 5, minutes_per_5_km) for d in dates])


def test_compare_to_peers_level_and_positions() -> None:
    # 10 km/week -> level "10-20 km" (10 is included in that band). Pace 6.0 min/km.
    result = compare_to_peers(two_weeks_of_runs(30), REFERENCE)
    assert result["level"] == "10-20 km"
    comparison = result["comparison"]
    assert comparison["weekly_km"]["you"] == 10.0
    assert comparison["weekly_km"]["position"] == "below p25"
    assert comparison["runs_per_week"]["position"] == "between p25 and p75"  # 2.0 = p75
    assert comparison["pace"]["position"] == "above p75"
    assert comparison["pace"]["meaning"] == "slower than most runners at your level"
    assert comparison["pace"]["you_text"] == "6:00"


def test_compare_to_peers_lower_pace_means_faster() -> None:
    # 20 min per 5 km = 4.0 min/km, below p25 (5.0): FASTER than most.
    result = compare_to_peers(two_weeks_of_runs(20), REFERENCE)
    assert result["comparison"]["pace"]["position"] == "below p25"
    assert result["comparison"]["pace"]["meaning"] == "faster than most runners at your level"


def test_compare_to_peers_one_run_is_not_enough() -> None:
    runs = make_runs([("2019-01-07", 5, 30)])
    assert compare_to_peers(runs, REFERENCE)["status"] == "not_enough_data"


# ---------- every output works with json.dumps ----------


@pytest.mark.parametrize(
    "function",
    [weekly_volume, pace_trend, consistency, load_ramp, longest_run],
)
def test_outputs_are_json_serialisable(function) -> None:
    runs = make_runs([(d, km, km * 6) for d, km in zip(FOUR_WEEK_DATES, [10, 10, 10, 10])])
    json.dumps(function(runs))
    json.dumps(function(NO_RUNS))


def test_compare_to_peers_is_json_serialisable() -> None:
    json.dumps(compare_to_peers(two_weeks_of_runs(30), REFERENCE))
