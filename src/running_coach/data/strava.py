"""Read a Strava export (activities.csv) and turn it into our run columns.

How to get the file: Strava -> Settings -> My Account -> Download or Delete Your Account
-> Request your archive. The zip file has activities.csv in it.

What we know about activities.csv (checked in other projects that read it):
- The export must be in English: we look for "Activity Date", "Activity Type",
  "Distance" and "Moving Time" / "Elapsed Time".
- "Distance" appears TWICE. pandas renames the second one to "Distance.1", and it is in
  METRES. The first one is in the units of the runner's settings, so we prefer the second.
- "Elapsed Time" also appears twice. All times are in seconds.
- The date text depends on the region, e.g. "Feb 17, 2022, 12:18:26 PM" or
  "19 Feb 2022, 10:14:12".
- "Activity Date" is in UTC, not the runner's local time. Checked with a real export:
  the file said 7:04 PM, and Strava showed 8:04 PM to a runner in UTC+1. We convert it
  to the runner's time zone, so a run near midnight counts on the right day.

The result has the same columns as get_athlete_runs() (date, distance_km, duration_min,
pace_min_km), so the charts, tools and agents work with it without any change.
"""

from typing import IO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from running_coach.analysis.metrics import format_pace
from running_coach.data.loaders import RUN_COLUMNS
from running_coach.data.prepare import (
    MAX_DISTANCE_KM,
    MAX_PACE,
    MIN_DISTANCE_KM,
    MIN_PACE,
    clean_runs,
)

DATE_COLUMN = "Activity Date"
TYPE_COLUMN = "Activity Type"
# Most reliable first: the second "Distance" column is always in metres.
DISTANCE_COLUMNS = ["Distance.1", "Distance"]
# Moving time leaves out stops (traffic lights), like the pace Strava shows you.
MOVING_TIME_COLUMN = "Moving Time"
ELAPSED_TIME_COLUMNS = ["Elapsed Time.1", "Elapsed Time"]

# Activity types that count as runs (compared in lower case, without spaces).
RUN_TYPES = {"run", "trailrun", "virtualrun"}

# With only one "Distance" column we can't be sure of the unit. No real run is longer
# than MAX_DISTANCE_KM (100 km), so a typical (median) value above it must be metres.
METRES_IF_MEDIAN_ABOVE = MAX_DISTANCE_KM


class StravaFormatError(ValueError):
    """The file is not a Strava activities.csv we can read. The message says why."""


def _first_present(columns: list[str], available: pd.Index) -> str | None:
    """Return the first name in `columns` that is in the file, or None."""
    return next((name for name in columns if name in available), None)


def _read_csv(file: str | IO) -> pd.DataFrame:
    """Read the CSV, turning any reading problem into a clear StravaFormatError."""
    try:
        return pd.read_csv(file)
    except ValueError as error:  # bad CSV, empty file, not text (all are ValueErrors)
        raise StravaFormatError(
            "This file could not be read as a CSV. Please upload activities.csv "
            "from your Strava archive."
        ) from error


def _check_columns(activities: pd.DataFrame) -> str:
    """Make sure the needed columns exist. Return the name of the distance column to use."""
    distance_column = _first_present(DISTANCE_COLUMNS, activities.columns)
    has_time = MOVING_TIME_COLUMN in activities.columns or _first_present(
        ELAPSED_TIME_COLUMNS, activities.columns
    )
    missing = [name for name in (DATE_COLUMN, TYPE_COLUMN) if name not in activities.columns]
    if distance_column is None:
        missing.append("Distance")
    if not has_time:
        missing.append("Moving Time or Elapsed Time")
    if missing:
        raise StravaFormatError(
            f"This doesn't look like a Strava activities.csv: missing column(s) "
            f"{', '.join(missing)}. Please upload activities.csv from your Strava archive, "
            "exported in English."
        )
    return distance_column


def _keep_runs(activities: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Keep only running activities and note what was removed."""
    types = activities[TYPE_COLUMN].astype(str).str.strip()
    is_run = types.str.lower().str.replace(" ", "").isin(RUN_TYPES)
    removed = types[~is_run].value_counts()
    if not removed.empty:
        details = ", ".join(f"{name}: {count}" for name, count in removed.items())
        notes.append(f"Removed {int(removed.sum())} activities that are not runs ({details}).")
    return activities[is_run]


def _local_dates(date_text: pd.Series, timezone: str, notes: list[str]) -> pd.Series:
    """Read the "Activity Date" text (in UTC) and convert it to the runner's local time."""
    # format="mixed": each region writes the date differently (see the top of the file).
    dates = pd.to_datetime(date_text, format="mixed", errors="coerce")
    try:
        ZoneInfo(timezone)  # only to check that the name is a real time zone
    except (ZoneInfoNotFoundError, ValueError):
        notes.append(f"Unknown time zone {timezone!r}, so dates are kept in UTC.")
        timezone = "UTC"
    if timezone == "UTC":
        notes.append(
            "Date: Strava writes 'Activity Date' in UTC, and we used it as it is. A run near "
            "midnight can count on the day before or after your local date."
        )
        return dates
    notes.append(
        f"Date: Strava writes 'Activity Date' in UTC; we converted it to your time zone "
        f"({timezone}), so each run counts on your local date."
    )
    # tz_localize("UTC"): "these times are UTC". tz_convert: change to local time.
    # tz_localize(None): drop the time zone again, like every other date in the project.
    return dates.dt.tz_localize("UTC").dt.tz_convert(timezone).dt.tz_localize(None)


def _distance_km(activities: pd.DataFrame, column: str, notes: list[str]) -> pd.Series:
    """Return the distance in km, and note which unit we assumed."""
    distance = pd.to_numeric(activities[column], errors="coerce")
    if column == "Distance.1":
        notes.append("Distance: from the second 'Distance' column, which Strava writes in metres.")
        return distance / 1000
    if distance.median() > METRES_IF_MEDIAN_ABOVE:
        notes.append(
            "Distance: only one 'Distance' column, and its values are large, "
            "so we read them as metres."
        )
        return distance / 1000
    notes.append(
        "Distance: only one 'Distance' column, and its values are small, so we read them "
        "as km. (Miles are not supported.)"
    )
    return distance


def _duration_min(activities: pd.DataFrame, notes: list[str]) -> pd.Series:
    """Return the duration in minutes: moving time, or elapsed time when it is missing."""
    elapsed_column = _first_present(ELAPSED_TIME_COLUMNS, activities.columns)
    elapsed = (
        pd.to_numeric(activities[elapsed_column], errors="coerce")
        if elapsed_column
        else pd.Series(float("nan"), index=activities.index)
    )
    if MOVING_TIME_COLUMN not in activities.columns:
        notes.append("Duration: 'Elapsed Time' (there is no 'Moving Time' column), so stops count.")
        return elapsed / 60

    moving = pd.to_numeric(activities[MOVING_TIME_COLUMN], errors="coerce")
    moving = moving.where(moving > 0)  # 0 seconds means "missing", not a real time
    seconds = moving.fillna(elapsed)
    notes.append("Duration: 'Moving Time' in seconds, so stops are not counted (like Strava).")
    used_elapsed = int((moving.isna() & elapsed.notna()).sum())
    if used_elapsed:
        notes.append(f"Used 'Elapsed Time' for {used_elapsed} run(s) without a moving time.")
    return seconds / 60


def load_strava_csv(file: str | IO, timezone: str = "UTC") -> tuple[pd.DataFrame, list[str]]:
    """Read a Strava activities.csv. Return (runs, notes).

    timezone: the runner's time zone, for example "Europe/London" (the app uses the
    browser's). The dates in the file are UTC and are converted to it.

    runs: one row per day with running, with the columns in RUN_COLUMNS, sorted by date.
    notes: short sentences that explain every assumption, for the app to show.
    Raises StravaFormatError (a ValueError) with a clear message if the file can't be used.
    """
    activities = _read_csv(file)
    distance_column = _check_columns(activities)
    notes: list[str] = []

    runs = _keep_runs(activities, notes)
    if runs.empty:
        raise StravaFormatError("No runs found in this file (only other activity types).")

    table = pd.DataFrame(
        {
            "datetime": _local_dates(runs[DATE_COLUMN], timezone, notes),
            "distance": _distance_km(runs, distance_column, notes),
            "duration": _duration_min(runs, notes),
        }
    )
    unreadable = int(table.isna().any(axis=1).sum())
    if unreadable:
        notes.append(f"Skipped {unreadable} run(s) with an unreadable date, distance or time.")
    table = table.dropna()

    # Same rule as the public dataset: one row per day, same-day runs added together.
    table["datetime"] = table["datetime"].dt.normalize()
    runs_per_day = table.groupby("datetime").size()
    days = table.groupby("datetime", as_index=False)[["distance", "duration"]].sum()
    if (runs_per_day > 1).any():
        notes.append(
            f"{int((runs_per_day > 1).sum())} day(s) had 2 or more runs: they were added "
            "together, like in the public dataset."
        )

    # The same cleaning rules as the public dataset (Step 1).
    cleaned = clean_runs(days)
    if len(cleaned) < len(days):
        notes.append(
            f"Removed {len(days) - len(cleaned)} day(s) outside the cleaning rules "
            f"({MIN_DISTANCE_KM:g}–{MAX_DISTANCE_KM:g} km, pace {format_pace(MIN_PACE)}–"
            f"{format_pace(MAX_PACE)} min/km)."
        )
    if cleaned.empty:
        raise StravaFormatError("No usable runs left after cleaning (check the distance units).")

    result = cleaned.rename(
        columns={
            "datetime": "date",
            "distance": "distance_km",
            "duration": "duration_min",
            "pace": "pace_min_km",
        }
    )
    first, last = result["date"].min(), result["date"].max()
    notes.insert(0, f"Found {len(result)} days with running, from {first:%d %b %Y} to {last:%d %b %Y}.")
    return result[RUN_COLUMNS].sort_values("date").reset_index(drop=True), notes
