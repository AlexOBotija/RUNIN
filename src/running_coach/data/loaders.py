"""Read the small processed files that the app and the agents use.

The files are built once by: python -m running_coach.data.prepare
"""

from pathlib import Path

import pandas as pd

from running_coach.data.prepare import REFERENCE_FILE, SAMPLE_FILE

# Column names used by every analysis function (units in the name, so nobody has to guess).
RUN_COLUMNS = ["date", "distance_km", "duration_min", "pace_min_km"]


def _read_processed(path: Path) -> pd.DataFrame:
    """Read a processed parquet file, with a clear message if it doesn't exist yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m running_coach.data.prepare"
        )
    return pd.read_parquet(path)


def load_sample() -> pd.DataFrame:
    """Return every clean run of the sample athletes (one row per athlete per day)."""
    return _read_processed(SAMPLE_FILE)


def load_reference() -> pd.DataFrame:
    """Return the percentile table: one row per runner level."""
    return _read_processed(REFERENCE_FILE)


def get_athlete_runs(athlete_id: int) -> pd.DataFrame:
    """Return one athlete's runs, sorted by date, with the columns in RUN_COLUMNS."""
    sample = load_sample()
    runs = sample[sample["athlete"] == athlete_id]
    if runs.empty:
        raise ValueError(f"Athlete {athlete_id} is not in the sample.")

    runs = runs.rename(
        columns={
            "datetime": "date",
            "distance": "distance_km",
            "duration": "duration_min",
            "pace": "pace_min_km",
        }
    )
    return runs[RUN_COLUMNS].sort_values("date").reset_index(drop=True)
