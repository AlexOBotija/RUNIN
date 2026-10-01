# Progress

## Step 0 — Project setup ✅

### What was done
- Created the folder structure: `data/raw/` and `data/processed/` (git-ignored, kept with `.gitkeep`), `notebooks/`, `src/running_coach/` with `data/`, `analysis/` and `agents/` sub-packages, `app/`, `tests/` and `docs/`.
- Added `pyproject.toml` (src layout, pytest `testpaths = tests`), `.gitignore`, `.env.example` and `CLAUDE.md`.
- Created a Python 3.12 virtual environment in `.venv` and installed `requirements.txt`.
- Installed the project in editable mode (`pip install -e .`), so `import running_coach` works everywhere.
- Pinned every library in `requirements.txt` to the installed version (`==`).
- Wrote `src/running_coach/llm.py`: `get_chat_model()` loads `.env`, checks that `GOOGLE_API_KEY` and `GEMINI_MODEL` are set (clear error if not) and returns a `ChatGoogleGenerativeAI` model. `python -m running_coach.llm` sends a test message and prints Gemini's reply.
- Added a placeholder `README.md`.
- Initialised Git and pushed the first commit to GitHub: https://github.com/AlexOBotija/RUNIN

### Decisions taken
- **Python 3.12** in a virtual environment (`py -3.12 -m venv .venv`). It is installed next to Python 3.14; 3.12 is a stable version that all our libraries support.
- **src layout + editable install.** The package code lives in `src/running_coach/`. `pip install -e .` lets notebooks, tests and the app import it without path tricks, and code changes work without reinstalling.
- **Pinned versions.** Only the 9 top-level libraries are pinned (pandas 3.0.6, pyarrow 25.0.1, langgraph 1.2.12, langchain 1.4.3, langchain-google-genai 4.4.0, streamlit 1.64.0, plotly 7.1.0, python-dotenv 1.2.4, pytest 9.1.1). Streamlit Cloud (Step 6) will install exactly the versions we tested.
- **Model: `gemini-3.5-flash-lite`** (free tier: 15 RPM, 250K TPM, 500 RPD).
  - The first choice was `gemini-3.8-flash`, but its free tier is only 5 RPM and 20 RPD.
  - One question uses about 4–6 LLM calls (supervisor, data agent, analyst, coach), so 20 RPD means only 3–5 questions per day. That is too few for development.
  - 500 RPD gives about 80–120 questions per day. The Python tools do the calculations, so a "Lite" model is good enough for choosing tools and writing the answers.
  - The model name is read from `.env`, so changing it later needs no code change.
- **A dedicated Google AI Studio project** for this app. Free-tier limits are counted per project, not per key, so other apps no longer use this app's budget.
- **Local Git identity** (this repository only, not global): Alexandre Andrade, alexandrade4444@gmail.com.
- **Secrets stay out of Git.** `.env` is git-ignored; only `.env.example` (with empty values) is committed.

### Things to remember
- Activate the venv in every new PowerShell window: `.\.venv\Scripts\Activate.ps1` (you should see `(.venv)`). Without it, `pip` installs into the global Python.
- Each run of `python -m running_coach.llm` uses 1 of the 500 daily requests.

## Step 1 — Data ✅

### What was done
- `src/running_coach/config.py`: shared project paths (`RAW_DIR`, `PROCESSED_DIR`) and `get_sample_size()`, which reads `SAMPLE_SIZE` from `.env` (default 1000). `SAMPLE_SIZE=1000` added to `.env.example`.
- `src/running_coach/data/download.py` (`python -m running_coach.data.download`): downloads `run_ww_2019_d.parquet` from Figshare into `data/raw/`. It skips files that already exist and writes to a `.part` file first, so a broken download is never mistaken for a finished one.
- `src/running_coach/data/prepare.py` (`python -m running_coach.data.prepare`, about 3 minutes):
  - `clean_runs()`: the one shared cleaning helper (keeps running days, adds `pace`, applies the rules below).
  - `build_sample(sample_size=1000, seed=42)`: saves `data/processed/sample.parquet`.
  - `build_reference_table()`: saves `data/processed/reference.parquet` (percentiles by level from all 2019 athletes).
  - `summarize_athletes()`: one row per athlete with `weekly_km`, `pace`, `runs_per_week` and `level`.
- `tests/test_prepare.py`: 9 tests on tiny hand-made tables (each cleaning rule, values on the limits, the 10-runs filter, rest weeks counted as 0).
- `notebooks/01_explore.ipynb`: sizes, distance and pace distributions, runs per week, athletes per level (sample vs all 2019) and the reference table. Saved with outputs so the charts show on GitHub.
- `requirements-dev.txt`: `jupyter==1.1.1` and `matplotlib==3.11.2` (plus everything in `requirements.txt`).

### The raw data
- Source: Figshare, DOI 10.6084/m9.figshare.16620238.v5, licence CC BY 4.0.
- We use only `run_ww_2019_d.parquet` (154.7 MB). Weeks can be built from days, but not the other way round. The 2020 files (COVID year), the COVID CSVs and the Power BI dashboard are not needed.
- 13,290,380 rows: one row per athlete per day for all 365 days; 65% of rows are rest days (distance 0). 36,412 athletes. No missing values, no duplicate days.
- Runs on the same day are added together (two 10 km runs = one 20 km day).
- The full table uses about **1,274 MB** of RAM in pandas.

### Cleaning rules (in `clean_runs`, one place for both files)
| Rule | Why |
|---|---|
| distance > 0 and duration > 0 | keep only days with running (removes rest days, avoids dividing by 0) |
| distance ≥ 0.5 km | shorter is usually a watch started by accident |
| distance ≤ 100 km | more in one day is almost always a GPS or typing error (max in raw data: 390 km) |
| pace between 2.5 and 15 min/km | faster than 2.5 is faster than the 10K world record (~2:37/km): GPS error or a bike ride; slower than 15 is walking or a watch left running |
| athlete has ≥ 10 clean runs in 2019 | fewer runs is too little data to analyse (used for both the sample and the reference table) |

The rules remove only about 0.5% of running days. The 100 km and 15 min/km limits already cap duration at 25 h, so no separate duration rule is needed.

### Weekly averages and levels
- Weeks go from Monday to Sunday (`resample("W-SUN")`), like Strava.
- Each athlete's weeks go from their **first to their last run** of 2019. Rest weeks in the middle count as 0 km; weeks before they started or after they stopped do not count.
- Average pace = total minutes / total km, so long runs count more than short jogs.
- Levels by average weekly distance (each band includes its lower edge): **under 10, 10-20, 20-40, 40-60, 60+ km**. "40+" was split because the dataset has many marathon runners; 40-60 and 60+ are clearly different (median pace 5.25 vs 4.93 min/km). The smallest level still has 3,778 athletes.

### Results
| File | Size | Content |
|---|---|---|
| `sample.parquet` | 1.74 MB (6.5 MB in RAM) | 133,562 runs, 1,000 athletes; columns `datetime, athlete, distance, duration, gender, pace` |
| `reference.parquet` | 7.5 KB | 5 rows (one per level): `n_athletes` + p25/p50/p75 of `weekly_km`, `pace`, `runs_per_week` |

- Same seed → identical sample (checked by running `build_sample` twice).
- The sample represents the population: 133.6 runs per athlete in both, same median distance (10.0 km) and pace (5.29 min/km), level shares within 1 percentage point.
- Changing the eligible list (e.g. a new cleaning rule) changes which athletes the seed picks. That is expected: a seed makes results repeatable only for the same input.

### Why we process the full data once, and the app loads only the small files
The full 2019 file needs about 1.3 GB of RAM and about 3 minutes to process. Free Streamlit hosting gives about 1 GB of RAM, and the app would repeat that work on every start. So the heavy work runs **once, on the laptop**. The app only loads the 1.7 MB sample and the 7.5 KB reference table, which is fast and fits easily in memory. The agents also get small summaries, never raw data.

### Things to remember
- New machine or new clone: `pip install -r requirements-dev.txt`, then `python -m running_coach.data.download`, then `python -m running_coach.data.prepare`.
- The data files are git-ignored; anyone who clones the repo rebuilds them with the two commands above.
- For Step 2: in `pace`, a lower number means faster. Reuse `summarize_athletes()`, `LEVEL_EDGES` and `LEVEL_LABELS` instead of writing the weekly logic again.

## Step 2 — Analysis functions ✅

### What was done
- `src/running_coach/data/loaders.py`: `load_sample()`, `load_reference()` and `get_athlete_runs(athlete_id)`. A missing file gives a clear error ("Run: python -m running_coach.data.prepare"); an unknown athlete gives a `ValueError`. Runs come back sorted by date with the columns `date, distance_km, duration_min, pace_min_km` (units in the names). The Strava data in Step 6 will be converted to the same 4 columns.
- `src/running_coach/analysis/metrics.py`: 6 functions (`weekly_volume`, `pace_trend`, `consistency`, `load_ramp`, `longest_run`, `compare_to_peers`) plus `format_pace()` (5.5 → "5:30").
- `tests/test_metrics.py`: 36 tests on tiny hand-made tables (45 tests in total with Step 1), with at least one edge case per function, the exact limits of the load-ratio labels, and a `json.dumps()` check for every function.
- Checked on 3 sample athletes (13, 30974 with 365 runs, 22771 with only 54 days of history) and on a hand-made 1-run table: nothing crashes, every output passes `json.dumps()` and is under 1,000 characters.

### Output format (same for every function)
- `"status"`: `"ok"` or `"not_enough_data"`. When there isn't enough data, the dictionary has only `status` and a short `note`; it never raises an error.
- Plain Python numbers, rounded (km 1 decimal, pace 2 decimals), dates as `"YYYY-MM-DD"`, pace also as `"m:ss"` text.
- A short `"note"` that explains the numbers to the reader (the AI in Step 3).
- No DataFrames and no raw runs: the agents get small summaries, never raw data.

### Decisions taken
- **"The last N weeks" = N blocks of 7 days ending on `end_date`.** By default `end_date` is the runner's **last run date**: the dataset is from 2019, so ending today would give empty weeks for everyone. Every week has 7 full days, so "last week vs the week before" is a fair comparison (Mon–Sun weeks would often make the last week look short). Every week-based function takes an optional `end_date` (e.g. today, for Strava data). Weeks before the first run are not counted; the output gives `weeks_analysed`.
- **`weekly_volume`**: km per week (rest weeks = 0) and % change between the last two weeks. `change_pct` is `null` when the previous week had 0 km. Needs 2 weeks of history.
- **`pace_trend`**: weekly pace = total minutes / total km. A straight line (linear trend, `numpy.polyfit`) through the weekly paces; weeks without runs are skipped but keep their position in time. Slope in seconds per km per week: **below −2 = improving, above +2 = getting worse, otherwise stable** (lower pace = faster). Needs 3 weeks with runs.
- **`consistency`**: runs per week, average and weeks with zero runs. A window with zero runs is a valid answer ("ok" with zeros), not "not enough data".
- **`load_ramp`**: acute = km in the last 7 days; chronic = km in the last 28 days / 4. **Ratio < 0.8 low, 0.8–1.3 normal, > 1.3 high risk** (limits included in "normal"; the rounded ratio is labelled, so label and number always match). A guideline from sports science (Gabbett 2016), not a medical rule. Needs 28 days of history.
- **`longest_run`**: longest run in the window and its % of the 7-day week that contains it. In the public dataset a "run" is one day.
- **`compare_to_peers`**: uses **all** the runner's runs and `summarize_athletes()` from Step 1 — the same method that built the reference table — so we compare like with like. Position: below p25 / between p25 and p75 (inclusive) / above p75, plus plain words; for pace "below p25" = **faster** than most. Needs 14 days between first and last run.

### Things to remember
- Weekly pace is noisy (athlete 30974 goes from 4:35 to 5:36 between weeks), so the trend label is a hint. The note says so, and the Coach should not overstate it.
- The ratio label "high risk" is a guideline. The Coach must phrase it as "you increased quickly", never as a medical warning.
- `get_athlete_runs` reads the sample file on every call. That's fine now; in Step 5 Streamlit will cache it.

## Next: Step 3 — First single agent
