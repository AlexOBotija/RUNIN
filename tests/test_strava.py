"""Tests for load_strava_csv(). Every CSV is a tiny fake made here, never real data."""

import io

import pytest

from running_coach.data.loaders import RUN_COLUMNS
from running_coach.data.strava import StravaFormatError, load_strava_csv

# Like a real export: "Distance" twice (first in km, second in metres) and times in seconds.
REAL_LAYOUT = """\
Activity ID,Activity Date,Activity Name,Activity Type,Elapsed Time,Distance,Moving Time,Distance
1,"Feb 17, 2022, 7:00:00 AM",Morning Run,Run,3100,10.0,3000,10000.0
2,"Feb 18, 2022, 6:00:00 PM",Commute,Ride,1800,12.5,1700,12500.0
3,"Feb 19, 2022, 8:00:00 AM",Long Run,Run,6300,20.0,6000,20000.0
4,"Feb 20, 2022, 9:00:00 AM",Walk with dog,Walk,2400,3.0,2300,3000.0
5,"Feb 21, 2022, 7:30:00 AM",Hills,Trail Run,2500,5.0,2400,5000.0
"""


def read(text: str):
    """Load a CSV given as text, as if it were an uploaded file."""
    return load_strava_csv(io.StringIO(text))


def test_returns_our_run_columns_with_correct_values():
    runs, notes = read(REAL_LAYOUT)
    assert list(runs.columns) == RUN_COLUMNS
    first = runs.iloc[0]
    assert str(first["date"].date()) == "2022-02-17"
    assert first["distance_km"] == 10.0
    assert first["duration_min"] == 50.0  # moving time: 3000 s, not the 3100 s elapsed
    assert first["pace_min_km"] == 5.0
    assert any("metres" in note for note in notes)


def test_distance_in_metres_and_in_km_gives_the_same_distance_km():
    in_metres = """\
Activity Date,Activity Type,Moving Time,Distance
"Feb 17, 2022, 7:00:00 AM",Run,3000,10000.0
"Feb 19, 2022, 8:00:00 AM",Run,6000,20000.0
"""
    in_km = """\
Activity Date,Activity Type,Moving Time,Distance
"Feb 17, 2022, 7:00:00 AM",Run,3000,10.0
"Feb 19, 2022, 8:00:00 AM",Run,6000,20.0
"""
    runs_m, _ = read(in_metres)
    runs_km, _ = read(in_km)
    assert list(runs_m["distance_km"]) == [10.0, 20.0]
    assert list(runs_km["distance_km"]) == [10.0, 20.0]
    # The second "Distance" column (metres) of a real export gives the same values too.
    runs_real, _ = read(REAL_LAYOUT)
    assert list(runs_real["distance_km"]) == [10.0, 20.0, 5.0]


def test_non_run_activities_are_removed():
    runs, notes = read(REAL_LAYOUT)
    assert len(runs) == 3  # Run, Run, Trail Run. The Ride and the Walk are gone.
    assert "Removed 2 activities that are not runs (Ride: 1, Walk: 1)." in notes


def test_wrong_columns_give_a_clear_error():
    wrong = "date,km,minutes\n2022-02-17,10,50\n"
    with pytest.raises(StravaFormatError, match="missing column.*Activity Date"):
        read(wrong)


def test_a_file_that_is_not_a_csv_gives_a_clear_error():
    with pytest.raises(StravaFormatError, match="could not be read as a CSV"):
        load_strava_csv(io.BytesIO(b""))


def test_a_file_with_only_rides_gives_a_clear_error():
    rides = 'Activity Date,Activity Type,Moving Time,Distance\n"Feb 17, 2022, 7:00:00 AM",Ride,3000,30.0\n'
    with pytest.raises(StravaFormatError, match="No runs found"):
        read(rides)


def test_runs_on_the_same_day_are_added_together():
    two_runs = """\
Activity Date,Activity Type,Moving Time,Distance
"Feb 17, 2022, 7:00:00 AM",Run,1500,5.0
"Feb 17, 2022, 6:00:00 PM",Run,1500,5.0
"""
    runs, notes = read(two_runs)
    assert len(runs) == 1
    assert runs.iloc[0]["distance_km"] == 10.0
    assert runs.iloc[0]["duration_min"] == 50.0
    assert any("added together" in note for note in notes)


def test_elapsed_time_is_used_when_moving_time_is_missing():
    no_moving = """\
Activity Date,Activity Type,Elapsed Time,Moving Time,Distance
"Feb 17, 2022, 7:00:00 AM",Run,3000,,10.0
"Feb 19, 2022, 8:00:00 AM",Run,6600,6000,20.0
"""
    runs, notes = read(no_moving)
    assert list(runs["duration_min"]) == [50.0, 100.0]
    assert "Used 'Elapsed Time' for 1 run(s) without a moving time." in notes


def test_both_regional_date_formats_are_read():
    dates = """\
Activity Date,Activity Type,Moving Time,Distance
"Feb 17, 2022, 7:00:00 AM",Run,3000,10.0
"19 Feb 2022, 08:00:00",Run,3000,10.0
"""
    runs, _ = read(dates)
    assert [str(day.date()) for day in runs["date"]] == ["2022-02-17", "2022-02-19"]


def test_days_outside_the_cleaning_rules_are_removed():
    with_errors = """\
Activity Date,Activity Type,Moving Time,Distance
"Feb 17, 2022, 7:00:00 AM",Run,3000,10.0
"Feb 18, 2022, 7:00:00 AM",Run,60,0.2
"Feb 19, 2022, 7:00:00 AM",Run,600,10.0
"""
    runs, notes = read(with_errors)  # 0.2 km is too short; 1:00 min/km is too fast
    assert len(runs) == 1
    assert any("Removed 2 day(s) outside the cleaning rules" in note for note in notes)
