"""Turn the raw 2019 daily file into the two small files the app uses.

- sample.parquet: every clean run of SAMPLE_SIZE random athletes.
- reference.parquet: percentiles of weekly distance, pace and runs per week for
  each runner level, calculated from ALL athletes of 2019.

Run with: python -m running_coach.data.prepare
"""

import math

import pandas as pd

from running_coach.config import PROCESSED_DIR, RAW_DIR, get_sample_size

RAW_FILE = RAW_DIR / "run_ww_2019_d.parquet"
SAMPLE_FILE = PROCESSED_DIR / "sample.parquet"
REFERENCE_FILE = PROCESSED_DIR / "reference.parquet"

# The only columns we use. Your Strava export will be converted to these too.
COLUMNS = ["datetime", "athlete", "distance", "duration", "gender"]

# Cleaning rules for one day of running.
MIN_DISTANCE_KM = 0.5  # shorter than this is usually a watch started by accident
MAX_DISTANCE_KM = 100.0  # more than this in one day is almost always a GPS or typing error
MIN_PACE = 2.5  # min/km; faster than the 10K world record (about 2:37 min/km)
MAX_PACE = 15.0  # min/km; slower than this is walking or a watch left running

# Athletes with fewer runs than this in 2019 are skipped: too little data to analyse.
MIN_RUNS = 10

# Runner levels by average weekly distance (km). Each band includes its lower edge:
# 10.0 km/week is "10-20 km", not "under 10 km".
LEVEL_EDGES = [0, 10, 20, 40, 60, math.inf]
LEVEL_LABELS = ["under 10 km", "10-20 km", "20-40 km", "40-60 km", "60+ km"]

PERCENTILES = [0.25, 0.50, 0.75]


def load_raw() -> pd.DataFrame:
    """Read the raw 2019 daily file, keeping only the columns we use."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_FILE} not found. Run: python -m running_coach.data.download"
        )
    return pd.read_parquet(RAW_FILE, columns=COLUMNS)


def clean_runs(days: pd.DataFrame) -> pd.DataFrame:
    """Keep only days with possible running values and add a pace column (min/km).

    Used by both build_sample and build_reference_table, so they share the same rules.
    """
    runs = days[(days["distance"] > 0) & (days["duration"] > 0)].copy()
    runs["pace"] = runs["duration"] / runs["distance"]
    good_distance = runs["distance"].between(MIN_DISTANCE_KM, MAX_DISTANCE_KM)
    good_pace = runs["pace"].between(MIN_PACE, MAX_PACE)
    return runs[good_distance & good_pace].reset_index(drop=True)


def athletes_with_enough_runs(runs: pd.DataFrame, min_runs: int = MIN_RUNS) -> pd.Index:
    """Return the IDs of athletes with at least min_runs clean runs, sorted."""
    runs_per_athlete = runs.groupby("athlete").size()
    return runs_per_athlete[runs_per_athlete >= min_runs].index.sort_values()


def summarize_athletes(runs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per athlete: weekly_km, pace, runs_per_week and level.

    Weeks go from Monday to Sunday. We count every week from the athlete's first run
    to their last run, so rest weeks in the middle count as 0 km.
    """
    weekly = (
        runs.set_index("datetime")
        .groupby("athlete")["distance"]
        .resample("W-SUN")
        .agg(["sum", "count"])
    )
    per_week = weekly.groupby("athlete").mean()

    totals = runs.groupby("athlete")[["distance", "duration"]].sum()
    summary = pd.DataFrame(
        {
            "weekly_km": per_week["sum"],
            # Total minutes / total km, so long runs count more than short jogs.
            "pace": totals["duration"] / totals["distance"],
            "runs_per_week": per_week["count"],
        }
    )
    summary["level"] = pd.cut(
        summary["weekly_km"], bins=LEVEL_EDGES, labels=LEVEL_LABELS, right=False
    )
    return summary


def build_sample(sample_size: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Pick sample_size random athletes, save their clean runs and return them.

    Only athletes with at least MIN_RUNS clean runs can be picked. The fixed seed
    makes the choice the same every time.
    """
    runs = clean_runs(load_raw())
    eligible = athletes_with_enough_runs(runs)
    if sample_size > len(eligible):
        raise ValueError(
            f"sample_size={sample_size} but only {len(eligible)} athletes are eligible"
        )

    chosen = pd.Series(eligible).sample(n=sample_size, random_state=seed)
    sample = runs[runs["athlete"].isin(chosen)]
    sample = sample.sort_values(["athlete", "datetime"]).reset_index(drop=True)

    # The file is committed to Git so the cloud app has it, so we keep it small (under
    # 1 MB): pace is not saved (load_sample() calculates it again with the same formula)
    # and zstd compresses better than the default (snappy).
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sample.drop(columns="pace").to_parquet(SAMPLE_FILE, index=False, compression="zstd")
    return sample


def build_reference_table() -> pd.DataFrame:
    """Build, save and return the percentile table by runner level from ALL 2019 athletes.

    One row per level. For pace, a lower number means faster, so pace_p25 is the
    pace of the faster runners in that level.
    """
    runs = clean_runs(load_raw())
    runs = runs[runs["athlete"].isin(athletes_with_enough_runs(runs))]
    athletes = summarize_athletes(runs)

    by_level = athletes.groupby("level", observed=True)
    table = pd.DataFrame({"n_athletes": by_level.size()})
    for metric in ["weekly_km", "pace", "runs_per_week"]:
        for p in PERCENTILES:
            table[f"{metric}_p{int(p * 100)}"] = by_level[metric].quantile(p)
    table = table.round(2).reset_index()
    table["level"] = table["level"].astype(str)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(REFERENCE_FILE, index=False)
    return table


if __name__ == "__main__":
    sample_size = get_sample_size()
    print(f"Building sample with {sample_size} athletes ...")
    sample = build_sample(sample_size)
    print(f"Saved {SAMPLE_FILE} ({len(sample):,} runs)")

    print("Building reference table from all 2019 athletes ...")
    reference = build_reference_table()
    print(f"Saved {REFERENCE_FILE}")
    print(reference.to_string(index=False))
