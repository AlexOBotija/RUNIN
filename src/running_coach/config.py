"""Project paths and settings shared by all modules."""

import os
from pathlib import Path

from dotenv import load_dotenv

# config.py lives in src/running_coach/, so the project root is two folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_SAMPLE_SIZE = 1000


def get_sample_size() -> int:
    """Return SAMPLE_SIZE from .env, or DEFAULT_SAMPLE_SIZE if it is not set."""
    load_dotenv()
    value = os.getenv("SAMPLE_SIZE", "").strip()
    if not value:
        return DEFAULT_SAMPLE_SIZE
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(f"SAMPLE_SIZE must be a whole number, got {value!r}") from None
