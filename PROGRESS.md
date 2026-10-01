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

## Next: Step 1 — Data
