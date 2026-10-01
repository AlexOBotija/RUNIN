"""Download the raw running dataset into data/raw/.

Dataset: "A public dataset on long-distance running training in 2019 and 2020"
(Afonseca, Watanabe & Duarte, BMClab/UFABC, PeerJ 2022), licence CC BY 4.0.
Hosted on Figshare: https://doi.org/10.6084/m9.figshare.16620238.v5

Run with: python -m running_coach.data.download
"""

import urllib.request
from pathlib import Path

from running_coach.config import RAW_DIR

# File name -> Figshare download link.
# We only need the 2019 daily file: weeks can be built from days, not the other way round.
FILES = {
    "run_ww_2019_d.parquet": "https://ndownloader.figshare.com/files/33801866",
}


def download_file(url: str, target: Path) -> None:
    """Download url to target, writing to a temporary file first.

    If the download stops halfway, only the .part file is left behind, so the next
    run does not mistake a broken file for a finished one.
    """
    temp_path = target.with_name(target.name + ".part")
    urllib.request.urlretrieve(url, temp_path)
    temp_path.replace(target)


def download_all() -> None:
    """Download every file in FILES into RAW_DIR, skipping files that already exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = RAW_DIR / name
        if target.exists():
            print(f"Skipping {name}: already in {RAW_DIR}")
            continue
        print(f"Downloading {name} ...")
        download_file(url, target)
        size_mb = target.stat().st_size / 1_000_000
        print(f"Saved {target} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_all()
