"""Analysis functions: turn one runner's runs into small, AI-readable summaries.

Every function receives a DataFrame with the columns date, distance_km, duration_min
and pace_min_km (see running_coach.data.loaders) and returns a small dictionary:
- "status" is "ok" or "not_enough_data";
- numbers are rounded plain Python numbers and dates are "YYYY-MM-DD" strings,
  so the dictionary always works with json.dumps();
- "note" is a short sentence that helps the reader understand the numbers.

What "the last N weeks" means: N blocks of 7 days that end on end_date. By default
end_date is the runner's LAST run date (the dataset is from 2019, so ending today would
give empty weeks). Every week is complete, so two weeks can be compared fairly.
Weeks before the runner's first run are not counted.
"""

import math

import numpy as np
import pandas as pd

from running_coach.data.prepare import summarize_athletes

# pace_trend: a weekly pace change smaller than this (seconds per km, per week) is "stable".
STABLE_PACE_SEC_PER_WEEK = 2.0
MIN_WEEKS_FOR_TREND = 3

# load_ramp: acute:chronic ratio limits (a common guideline from sports science).
LOW_RATIO = 0.8
HIGH_RATIO = 1.3
DAYS_FOR_CHRONIC = 28

# compare_to_peers: the reference table uses average weekly values, so we need some history.
MIN_DAYS_FOR_PEERS = 14


# ---------- Small helpers ----------


def format_pace(pace_min_km: float) -> str:
    """Turn a pace in decimal minutes into "m:ss" text, e.g. 5.5 -> "5:30"."""
    total_seconds = round(pace_min_km * 60)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def format_duration(minutes: float) -> str:
    """Turn a duration in decimal minutes into text: 55.4 -> "55:24", 125.5 -> "2:05:30".

    The LLM gets this text instead of 55.4, which it once read as "55 minutes 4 seconds".
    """
    hours, seconds = divmod(round(minutes * 60), 3600)
    minutes_part, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}:{minutes_part:02d}:{seconds:02d}"
    return f"{minutes_part}:{seconds:02d}"


def _not_enough_data(note: str) -> dict:
    """The answer every function gives when it can't calculate its result."""
    return {"status": "not_enough_data", "note": note}


def _to_text(date: pd.Timestamp) -> str:
    """Format a date as "YYYY-MM-DD"."""
    return date.strftime("%Y-%m-%d")


def _resolve_end_date(runs: pd.DataFrame, end_date: str | pd.Timestamp | None) -> pd.Timestamp:
    """Return end_date as a day (no time), or the runner's last run date if it is None."""
    if end_date is None:
        return runs["date"].max().normalize()
    return pd.Timestamp(end_date).normalize()


def _weekly_table(runs: pd.DataFrame, weeks: int, end_date: pd.Timestamp) -> pd.DataFrame:
    """Return one row per 7-day week, oldest first, ending on end_date.

    Columns: week_start, week_end, distance_km, duration_min, runs.
    Weeks with no runs have 0 in every column. Weeks before the first run are left out,
    so the table can have fewer than `weeks` rows (or none at all).
    """
    first_run = runs["date"].min().normalize()
    days_of_history = (end_date - first_run).days + 1
    n_weeks = min(weeks, math.ceil(days_of_history / 7)) if days_of_history > 0 else 0

    # Week number 0 = the last 7 days, 1 = the 7 days before, and so on.
    days_before_end = (end_date - runs["date"].dt.normalize()).dt.days
    week_number = days_before_end // 7
    in_window = (days_before_end >= 0) & (week_number < n_weeks)
    window = runs[in_window].assign(week_number=week_number[in_window])

    table = window.groupby("week_number").agg(
        distance_km=("distance_km", "sum"),
        duration_min=("duration_min", "sum"),
        runs=("distance_km", "size"),
    )
    table = table.reindex(range(n_weeks), fill_value=0).sort_index(ascending=False)
    table["week_start"] = [end_date - pd.Timedelta(days=7 * k + 6) for k in table.index]
    table["week_end"] = [end_date - pd.Timedelta(days=7 * k) for k in table.index]
    return table.reset_index(drop=True)


def _period(table: pd.DataFrame) -> dict:
    """Describe the weeks that were analysed."""
    return {
        "start": _to_text(table["week_start"].iloc[0]),
        "end": _to_text(table["week_end"].iloc[-1]),
        "weeks_analysed": len(table),
    }


# ---------- The six analysis functions ----------


def weekly_volume(
    runs: pd.DataFrame, weeks: int = 8, end_date: str | pd.Timestamp | None = None
) -> dict:
    """Km per week for the last `weeks` weeks, their total, and the % change between the last two."""
    if runs.empty:
        return _not_enough_data("No runs found.")
    table = _weekly_table(runs, weeks, _resolve_end_date(runs, end_date))
    if len(table) < 2:
        return _not_enough_data("Need at least 2 weeks of history to compare weeks.")

    last_km = float(table["distance_km"].iloc[-1])
    previous_km = float(table["distance_km"].iloc[-2])
    first_run = runs["date"].min().normalize()
    if table["week_start"].iloc[-2] < first_run:
        # A new runner: the week before started before their first run, so it covers only
        # a few days of running. Comparing it with a full week would give a misleading %.
        change_pct = None
        note = (
            "Weeks are 7-day blocks, oldest first. The week before only partly counts: the "
            f"first run was on {_to_text(first_run)}, so there is no % change."
        )
    elif previous_km > 0:
        change_pct = round((last_km - previous_km) / previous_km * 100, 1)
        note = "Weeks are 7-day blocks, oldest first."
    else:
        change_pct = None
        note = "Weeks are 7-day blocks, oldest first. The previous week had 0 km, so there is no % change."

    return {
        "status": "ok",
        "period": _period(table),
        "weekly_km": [
            {"week_start": _to_text(row.week_start), "km": round(float(row.distance_km), 1)}
            for row in table.itertuples()
        ],
        "total_km": round(float(table["distance_km"].sum()), 1),
        "average_km_per_week": round(float(table["distance_km"].mean()), 1),
        "last_week_km": round(last_km, 1),
        "previous_week_km": round(previous_km, 1),
        "change_pct": change_pct,
        "note": note,
    }


def pace_trend(
    runs: pd.DataFrame, weeks: int = 8, end_date: str | pd.Timestamp | None = None
) -> dict:
    """Average pace per week and whether it is improving, stable or getting worse.

    Method: weekly pace = total minutes / total km (so long runs count more). Then we fit
    a straight line (linear trend) through the weekly paces. The slope is the change in
    seconds per km each week. A LOWER pace is FASTER, so a negative slope means improving.
    """
    if runs.empty:
        return _not_enough_data("No runs found.")
    table = _weekly_table(runs, weeks, _resolve_end_date(runs, end_date))
    # Keep the week position (0 = oldest) so empty weeks still leave a gap in time.
    table["position"] = range(len(table))
    with_runs = table[table["distance_km"] > 0].copy()
    if len(with_runs) < MIN_WEEKS_FOR_TREND:
        return _not_enough_data(
            f"Need at least {MIN_WEEKS_FOR_TREND} weeks with runs to see a pace trend."
        )

    with_runs["pace"] = with_runs["duration_min"] / with_runs["distance_km"]
    slope_min, _ = np.polyfit(with_runs["position"], with_runs["pace"], deg=1)
    slope_sec = float(slope_min) * 60
    weeks_between = int(with_runs["position"].iloc[-1] - with_runs["position"].iloc[0])

    if slope_sec < -STABLE_PACE_SEC_PER_WEEK:
        trend = "improving"
    elif slope_sec > STABLE_PACE_SEC_PER_WEEK:
        trend = "getting worse"
    else:
        trend = "stable"

    return {
        "status": "ok",
        "period": _period(table),
        "weekly_pace": [
            {
                "week_start": _to_text(row.week_start),
                "pace_min_km": round(float(row.pace), 2),
                "pace": format_pace(row.pace),
            }
            for row in with_runs.itertuples()
        ],
        "weeks_with_runs": len(with_runs),
        "slope_sec_per_km_per_week": round(slope_sec, 1),
        "change_over_period_sec_per_km": round(slope_sec * weeks_between, 1),
        "trend": trend,
        "note": (
            "Linear trend of weekly pace. Lower pace = faster. 'stable' means the change is "
            f"under {STABLE_PACE_SEC_PER_WEEK:g} s/km per week. Pace also depends on the type "
            "of runs (easy or hard), so this is a hint, not proof."
        ),
    }


def consistency(
    runs: pd.DataFrame, weeks: int = 8, end_date: str | pd.Timestamp | None = None
) -> dict:
    """Runs per week and the number of weeks with zero runs."""
    if runs.empty:
        return _not_enough_data("No runs found.")
    table = _weekly_table(runs, weeks, _resolve_end_date(runs, end_date))
    if table.empty:
        return _not_enough_data("No history before the end date.")

    return {
        "status": "ok",
        "period": _period(table),
        "runs_per_week": [int(n) for n in table["runs"]],
        "average_runs_per_week": round(float(table["runs"].mean()), 1),
        "weeks_with_zero_runs": int((table["runs"] == 0).sum()),
        "note": "Weeks are 7-day blocks, oldest first.",
    }


def load_ramp(runs: pd.DataFrame, end_date: str | pd.Timestamp | None = None) -> dict:
    """Compare the last 7 days (acute load) with the usual week (chronic load).

    acute = km in the last 7 days; chronic = average km per week over the last 28 days.
    The ratio acute / chronic shows if the runner suddenly does much more than their body
    is used to. It is a guideline from sports science, not a medical rule.
    """
    if runs.empty:
        return _not_enough_data("No runs found.")
    end = _resolve_end_date(runs, end_date)
    days_of_history = (end - runs["date"].min().normalize()).days + 1
    if days_of_history < DAYS_FOR_CHRONIC:
        return _not_enough_data(
            f"Need at least {DAYS_FOR_CHRONIC} days of history to know the usual weekly load."
        )

    table = _weekly_table(runs, DAYS_FOR_CHRONIC // 7, end)
    acute_km = float(table["distance_km"].iloc[-1])
    chronic_km = float(table["distance_km"].mean())
    if chronic_km == 0:
        return _not_enough_data(f"No runs in the last {DAYS_FOR_CHRONIC} days.")

    # Label the rounded ratio, so the label always matches the number the reader sees.
    ratio = round(acute_km / chronic_km, 2)
    if ratio < LOW_RATIO:
        label = "low"
    elif ratio <= HIGH_RATIO:
        label = "normal"
    else:
        label = "high risk"

    return {
        "status": "ok",
        "period": _period(table),
        "acute_km_last_7_days": round(acute_km, 1),
        "chronic_km_per_week_last_28_days": round(chronic_km, 1),
        "ratio": ratio,
        "label": label,
        "note": (
            f"Ratio below {LOW_RATIO} is low, {LOW_RATIO}-{HIGH_RATIO} normal, above "
            f"{HIGH_RATIO} high risk. A guideline from sports science, not a medical rule."
        ),
    }


def longest_run(
    runs: pd.DataFrame, weeks: int = 8, end_date: str | pd.Timestamp | None = None
) -> dict:
    """The longest run in the last `weeks` weeks and its share of that week's distance.

    In the public dataset one row is one day (runs on the same day were added together).
    """
    if runs.empty:
        return _not_enough_data("No runs found.")
    end = _resolve_end_date(runs, end_date)
    table = _weekly_table(runs, weeks, end)
    if table.empty:
        return _not_enough_data("No history before the end date.")

    start = table["week_start"].iloc[0]
    window = runs[(runs["date"] >= start) & (runs["date"].dt.normalize() <= end)]
    if window.empty:
        return _not_enough_data(f"No runs in the last {len(table)} weeks.")

    longest = window.loc[window["distance_km"].idxmax()]
    run_day = longest["date"].normalize()
    week = table[(table["week_start"] <= run_day) & (table["week_end"] >= run_day)].iloc[0]
    share_pct = longest["distance_km"] / week["distance_km"] * 100

    return {
        "status": "ok",
        "period": _period(table),
        "date": _to_text(run_day),
        "distance_km": round(float(longest["distance_km"]), 1),
        "duration": format_duration(longest["duration_min"]),
        "pace": format_pace(longest["pace_min_km"]),
        "week_km": round(float(week["distance_km"]), 1),
        "share_of_week_pct": round(float(share_pct), 1),
        "note": "The week is the 7-day block that contains the longest run.",
    }


def _position(value: float, p25: float, p75: float) -> str:
    """Say where a value sits compared with the 25th and 75th percentiles."""
    if value < p25:
        return "below p25"
    if value > p75:
        return "above p75"
    return "between p25 and p75"


def compare_to_peers(runs: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Find the runner's level and compare them with runners at the same level.

    Uses ALL the runner's runs and summarize_athletes() from Step 1, the same method that
    built the reference table, so we compare like with like. For pace a LOWER number is
    FASTER, so "below p25" means faster than most runners at that level.
    """
    if runs.empty:
        return _not_enough_data("No runs found.")
    days_of_history = (runs["date"].max() - runs["date"].min()).days + 1
    if days_of_history < MIN_DAYS_FOR_PEERS:
        return _not_enough_data(
            f"Need at least {MIN_DAYS_FOR_PEERS} days between the first and last run."
        )

    # summarize_athletes expects the column names of the processed sample.
    as_sample = runs.rename(
        columns={"date": "datetime", "distance_km": "distance", "duration_min": "duration"}
    ).assign(athlete=0)
    me = summarize_athletes(as_sample).iloc[0]
    level = str(me["level"])
    peers = reference[reference["level"] == level].iloc[0]

    meanings = {
        "weekly_km": {"below p25": "less than most", "above p75": "more than most"},
        "pace": {"below p25": "faster than most", "above p75": "slower than most"},
        "runs_per_week": {"below p25": "fewer than most", "above p75": "more than most"},
    }
    comparison = {}
    for metric, words in meanings.items():
        value = float(me[metric])
        position = _position(value, peers[f"{metric}_p25"], peers[f"{metric}_p75"])
        comparison[metric] = {
            "you": round(value, 2),
            "p25": float(peers[f"{metric}_p25"]),
            "p50": float(peers[f"{metric}_p50"]),
            "p75": float(peers[f"{metric}_p75"]),
            "position": position,
            "meaning": words.get(position, "typical for") + " runners at your level",
        }
    comparison["pace"]["you_text"] = format_pace(me["pace"])

    return {
        "status": "ok",
        "level": level,
        "peers_in_level": int(peers["n_athletes"]),
        "comparison": comparison,
        "note": (
            "Level is based on average weekly km over all your runs. Peers are all 2019 "
            "runners at that level. Pace is min/km: lower = faster."
        ),
    }
