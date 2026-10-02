"""Running Coach Crew: the Streamlit app.

Run it from the project root with:
    streamlit run app/streamlit_app.py

Remember: Streamlit runs this whole script again, from top to bottom, every time the
visitor clicks or types something. Slow work is cached (st.cache_data) and values that
must survive a rerun are kept in st.session_state.
"""

import io
import os
import random
import time

import pandas as pd
import streamlit as st
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from running_coach.agents import crew
from running_coach.analysis.metrics import (
    compare_to_peers,
    consistency,
    pace_trend,
    weekly_volume,
)
from running_coach.charts import pace_chart, weekly_distance_chart
from running_coach.data.loaders import get_athlete_runs, load_reference, load_sample
from running_coach.data.strava import StravaFormatError, load_strava_csv
from running_coach.limits import DailyLimit
from running_coach.llm import ModelBusy, RateLimitReached

SAMPLE_SOURCE = "Sample runner"
STRAVA_SOURCE = "Upload Strava export"

DEFAULT_ATHLETE = 30974  # a sample runner with a full year of runs: a good first view
DEFAULT_WEEKS = 12
CHART_CONFIG = {"displayModeBar": False}  # hide Plotly's toolbar: cleaner for visitors

EXAMPLE_QUESTIONS = [
    "Am I improving?",
    "Am I increasing my distance too fast?",
    "How do I compare with runners at my level?",
]

# Protect the free Gemini quota (500 requests per day; one question uses 2 to 4).
MAX_QUESTIONS_PER_SESSION = 5  # each visitor (a page refresh starts a new session)
MAX_QUESTIONS_PER_DAY = 60  # all visitors together: at most about 240 requests per day

# Settings that Streamlit Cloud keeps in its Secrets box (secrets.toml locally, if any).
SECRET_NAMES = ["GOOGLE_API_KEY", "GEMINI_MODEL"]


def use_cloud_secrets() -> None:
    """Copy the API settings from st.secrets into the environment, where llm.py reads them.

    On Streamlit Cloud the settings are in st.secrets. On the laptop there is no
    secrets.toml, so nothing is copied and llm.py reads .env as before. (load_dotenv()
    never overwrites a variable that is already set, so the cloud values win.)
    """
    if not st.secrets.load_if_toml_exists():  # no secrets file: we are on the laptop
        return
    for name in SECRET_NAMES:
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])


# ---------- Cached data (loaded once, then reused on every rerun) ----------


@st.cache_data
def cached_reference() -> pd.DataFrame:
    """The percentile table by runner level (7.5 KB)."""
    return load_reference()


@st.cache_data
def athlete_ids() -> list[int]:
    """Every athlete id in the sample, sorted, for the dropdown."""
    return sorted(int(athlete) for athlete in load_sample()["athlete"].unique())


@st.cache_data
def cached_athlete_runs(athlete_id: int) -> pd.DataFrame:
    """One sample runner's runs. Each runner is read from the file only once."""
    return get_athlete_runs(athlete_id)


@st.cache_data
def cached_strava_runs(file_bytes: bytes, timezone: str) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded activities.csv. The same file (and time zone) is only read once."""
    return load_strava_csv(io.BytesIO(file_bytes), timezone)


@st.cache_resource
def cached_crew() -> CompiledStateGraph:
    """The agent crew, built once and shared by every visitor (cache_resource, not a copy).

    It is built on the first question, not when the page opens, so the charts still work
    if the API key is missing.
    """
    return crew.build_crew()


@st.cache_resource
def cached_daily_limit() -> DailyLimit:
    """ONE question counter shared by every visitor (cache_resource gives the same object)."""
    return DailyLimit(MAX_QUESTIONS_PER_DAY)


# ---------- Sidebar ----------


def pick_random_athlete(ids: list[int]) -> None:
    """Button callback: choose a random runner. It runs BEFORE the next rerun, so the
    dropdown already shows the new runner when it is drawn."""
    st.session_state.athlete_id = random.choice(ids)


def sample_runner_sidebar() -> tuple[pd.DataFrame, str]:
    """Draw the sample-runner controls. Return the chosen runner's runs and a label."""
    ids = athlete_ids()
    if "athlete_id" not in st.session_state:
        st.session_state.athlete_id = DEFAULT_ATHLETE if DEFAULT_ATHLETE in ids else ids[0]

    # key="athlete_id" connects the dropdown to st.session_state.athlete_id.
    athlete_id = st.sidebar.selectbox(
        "Runner (athlete ID)", ids, key="athlete_id", help="1,000 runners from the 2019 dataset"
    )
    st.sidebar.button(
        "🎲 Random runner", on_click=pick_random_athlete, args=(ids,), width="stretch"
    )
    return cached_athlete_runs(athlete_id), f"Sample runner {athlete_id}"


def strava_sidebar() -> tuple[pd.DataFrame, str]:
    """Draw the upload control. Return the uploaded runs and a label.

    If there is no file yet, or the file can't be used, show a friendly message and stop
    the script here (st.stop), so the rest of the page is not drawn.
    """
    uploaded = st.sidebar.file_uploader(
        "Your Strava activities.csv",
        type="csv",
        help="Strava → Settings → My Account → Download or Delete Your Account → "
        "Request your archive. activities.csv is inside the zip file.",
    )
    if uploaded is None:
        st.info(
            "Upload **activities.csv** from your Strava archive in the sidebar. "
            "Get it on Strava: Settings → My Account → Download or Delete Your Account → "
            "Request your archive. Your file is only used in this session."
        )
        st.stop()
    try:
        # Strava's dates are UTC: convert them to the visitor's browser time zone.
        timezone = st.context.timezone or "UTC"
        runs, notes = cached_strava_runs(uploaded.getvalue(), timezone)
    except StravaFormatError as error:
        st.warning(f"**We couldn't use this file.** {error}")
        st.stop()

    with st.sidebar.expander("What we assumed about your file"):
        st.markdown("\n".join(f"- {note}" for note in notes))
    return runs, f"Your Strava runs ({uploaded.name})"


# ---------- Main page ----------


def show_key_numbers(volume: dict, runs_per_week: dict, peers: dict) -> None:
    """Four numbers at the top of the page. Each one comes from an analysis function."""
    km_col, runs_col, pace_col, level_col = st.columns(4)

    if volume["status"] == "ok":
        change = volume["change_pct"]
        km_col.metric(
            "Km this week",
            f"{volume['last_week_km']} km",
            delta=None if change is None else f"{change:+.1f}% vs week before",
            delta_color="off",  # more km is not always good, so no green/red
        )
    else:
        km_col.metric("Km this week", "–")

    if runs_per_week["status"] == "ok":
        runs_col.metric("Runs this week", runs_per_week["runs_per_week"][-1])
    else:
        runs_col.metric("Runs this week", "–")

    if peers["status"] == "ok":
        pace_col.metric(
            "Average pace",
            f"{peers['comparison']['pace']['you_text']} /km",
            help="Total minutes / total km over all runs (lower = faster)",
        )
        level_col.metric(
            "Level",
            peers["level"],
            help="Based on average km per week over all runs",
        )
    else:
        pace_col.metric("Average pace", "–")
        level_col.metric("Level", "–", help=peers["note"])


def show_charts(volume: dict, trend: dict, peers: dict) -> None:
    """The two charts side by side, or the reason when there isn't enough data."""
    distance_col, pace_col = st.columns(2)
    with distance_col:
        if volume["status"] == "ok":
            st.plotly_chart(weekly_distance_chart(volume, peers), theme=None, config=CHART_CONFIG)
        else:
            st.info(f"Distance chart: not enough data. {volume['note']}")
    with pace_col:
        if trend["status"] == "ok":
            st.plotly_chart(pace_chart(trend), theme=None, config=CHART_CONFIG)
        else:
            st.info(f"Pace chart: not enough data. {trend['note']}")


# ---------- Chat with the crew ----------


def error_message(text: str) -> dict:
    """A chat message that explains a problem in friendly words."""
    return {"role": "assistant", "content": text, "is_error": True}


def format_answer(answer: str) -> str:
    """Make the coach's "Next week:" line stand out with a bold label."""
    lines = []
    for line in answer.splitlines():
        plain = line.strip().replace("**", "")  # the LLM sometimes adds its own bold
        if plain.startswith("Next week:"):
            # The empty line first: in Markdown, a single line break doesn't start a new line.
            line = "\n👟 **Next week:**" + plain.removeprefix("Next week:")
        lines.append(line)
    return "\n".join(lines)


def ask_crew(question: str, runs: pd.DataFrame) -> dict:
    """Ask the crew and return a chat message: the answer, or a friendly error."""
    start = time.perf_counter()
    try:
        result = crew.ask(question, runs, crew=cached_crew())
    except RateLimitReached:
        return error_message(
            "The free AI quota is used up for the moment. Please wait a minute and ask "
            "again. If it still doesn't work, today's limit is reached: try again tomorrow."
        )
    except ModelBusy:
        return error_message(
            "The AI service is busy right now (too many people are using it). "
            "Please try again in a minute."
        )
    except GraphRecursionError:
        return error_message("The agents took too many steps. Please ask again.")
    except RuntimeError as error:  # for example GOOGLE_API_KEY is not set
        return error_message(f"The AI is not set up yet: {error}")
    except Exception as error:  # anything else: never show a red traceback to a visitor
        print(f"Crew error: {error!r}")  # shown in the terminal, for the developer
        return error_message("Something went wrong while the agents were working. Please try again.")

    if not result["final_answer"]:
        return error_message("The agents stopped before writing an answer. Please ask again.")
    return {
        "role": "assistant",
        "content": format_answer(result["final_answer"]),
        "steps": result["steps"],
        "llm_calls": result["llm_calls"],
        "seconds": time.perf_counter() - start,
        "tool_results": result["tool_results"],
        "analyst_notes": result["analyst_notes"],
    }


def show_agent_steps(message: dict) -> None:
    """The "How the agents worked" section under an answer."""
    with st.expander("How the agents worked"):
        # Escape "_" so tool names like weekly_volume are not shown in italics.
        steps = [step.replace("_", "\\_") for step in message["steps"]]
        st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(steps, 1)))
        st.caption(f"{message['llm_calls']} AI calls · {message['seconds']:.1f} s")
        if message["tool_results"]:
            st.markdown("**Numbers from the analysis tools**")
            st.json(message["tool_results"], expanded=False)
        if message["analyst_notes"]:
            st.markdown("**Analyst's notes for the coach**")
            st.markdown(message["analyst_notes"])


def show_message(message: dict) -> None:
    """Draw one chat message (question, answer or error)."""
    with st.chat_message(message["role"]):
        if message.get("is_error"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])
        if "steps" in message:
            show_agent_steps(message)


def questions_left() -> int:
    """How many questions this visitor can still ask in this session."""
    return MAX_QUESTIONS_PER_SESSION - st.session_state.get("questions_asked", 0)


def check_limits() -> dict | None:
    """Return a friendly error message if the question is over a limit, otherwise None.

    A question that passes is counted (for this visitor and for the whole app), and only
    then goes to the crew. A question over the limit never calls the crew.
    """
    if questions_left() <= 0:
        return error_message(
            f"You have used your {MAX_QUESTIONS_PER_SESSION} questions for this visit. "
            "This demo runs on a free AI quota, so each visitor gets a few questions. "
            "Thanks for trying the coach team! You can still explore the charts and other runners."
        )
    if not cached_daily_limit().try_use():
        return error_message(
            "The coach team has answered all its questions for today (the app runs on a free "
            "AI quota). Please come back tomorrow. The charts still work."
        )
    st.session_state.questions_asked = st.session_state.get("questions_asked", 0) + 1
    return None


def chat_section(runs: pd.DataFrame, runner_label: str) -> None:
    """Example buttons, the chat history, and a new question if there is one."""
    st.divider()
    st.subheader("Ask the coach team")
    # An empty placeholder: we fill it at the end, when we know if a question was just
    # counted, so "questions left" is never one behind.
    limit_note = st.empty()

    # The history is kept in session_state so it survives reruns. Old answers were about
    # another runner, so a new runner starts a new chat.
    if st.session_state.get("chat_runner") != runner_label:
        st.session_state.messages = []
        st.session_state.chat_runner = runner_label

    # A button returns True only on the rerun right after it was clicked.
    # A horizontal container: each button is as wide as its text, and the buttons wrap
    # to a new line on narrow screens (equal columns cut long questions off).
    question = None
    with st.container(horizontal=True):
        for example in EXAMPLE_QUESTIONS:
            if st.button(example):
                question = example
    typed = st.chat_input("Ask about this runner's training…")
    question = typed or question

    for message in st.session_state.messages:
        show_message(message)

    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        show_message(user_message)
        answer = check_limits()
        if answer is None:  # within the limits: ask the crew
            with st.spinner("The agents are working on it… (usually 3–5 s)"):
                answer = ask_crew(question, runs)
        st.session_state.messages.append(answer)
        show_message(answer)

    limit_note.caption(
        "Each question is answered on its own: the agents don't remember earlier questions. "
        f"One question uses 2 to 4 AI calls. Questions left in this visit: "
        f"{max(questions_left(), 0)} of {MAX_QUESTIONS_PER_SESSION}."
    )


def main() -> None:
    """Draw the whole page: sidebar, key numbers, charts and the chat."""
    st.set_page_config(page_title="Running Coach Crew", page_icon="🏃", layout="wide")
    st.title("🏃 Running Coach Crew")
    st.caption(
        "Ask questions about a runner's training. A team of AI agents answers, "
        "using numbers calculated from the real runs."
    )
    use_cloud_secrets()

    st.sidebar.header("Choose a runner")
    source = st.sidebar.radio("Data source", [SAMPLE_SOURCE, STRAVA_SOURCE])
    try:
        if source == SAMPLE_SOURCE:
            runs, runner_label = sample_runner_sidebar()
        else:
            runs, runner_label = strava_sidebar()
        reference = cached_reference()
    except FileNotFoundError:
        st.error(
            "The data files are missing. Build them with "
            "`python -m running_coach.data.download`, then "
            "`python -m running_coach.data.prepare`"
        )
        st.stop()
    weeks = st.sidebar.slider("Weeks in the charts", 4, 52, DEFAULT_WEEKS)

    # Every number and chart on the page comes from these analysis results (no LLM).
    volume = weekly_volume(runs, weeks=weeks)
    runs_per_week = consistency(runs, weeks=weeks)
    trend = pace_trend(runs, weeks=weeks)
    peers = compare_to_peers(runs, reference)

    st.subheader(runner_label)
    last_run = runs["date"].max().strftime("%d %b %Y")
    caption = (
        f"{len(runs)} runs. The data ends on {last_run}, so \"this week\" means the "
        "7 days up to that date."
    )
    if peers["status"] == "ok":
        caption += (
            " The band shows the middle half (25th–75th percentile) of 2019 runners "
            "at the same level."
        )
    st.caption(caption)
    show_key_numbers(volume, runs_per_week, peers)
    show_charts(volume, trend, peers)
    chat_section(runs, runner_label)


main()
