"""Tests for the Streamlit app with streamlit.testing.v1.AppTest. No real LLM, no API calls.

AppTest runs app/streamlit_app.py like a browser would (but without one), and lets us
click buttons and read what is on the page. The crew is replaced by a fake, so the tests
are fast, free and always give the same result. The processed data files are needed,
so the tests are skipped if they don't exist yet.
"""

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from running_coach.agents import crew
from running_coach.analysis.metrics import compare_to_peers, weekly_volume
from running_coach.data.loaders import get_athlete_runs, load_reference
from running_coach.data.prepare import REFERENCE_FILE, SAMPLE_FILE
from running_coach.limits import DailyLimit
from running_coach.llm import ModelBusy, RateLimitReached

pytestmark = pytest.mark.skipif(
    not (SAMPLE_FILE.exists() and REFERENCE_FILE.exists()),
    reason="Needs the processed data. Run: python -m running_coach.data.prepare",
)

APP_FILE = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
FIRST_RUNNER = 30974  # the runner the app shows first (DEFAULT_ATHLETE)


def start_app() -> AppTest:
    """Run the app once, as when a visitor opens the page."""
    return AppTest.from_file(str(APP_FILE), default_timeout=30).run()


def click(app: AppTest, label: str) -> AppTest:
    """Click the button with this label and rerun the app."""
    button = next(button for button in app.button if button.label == label)
    return button.click().run()


@pytest.fixture
def fake_crew(monkeypatch) -> list[str]:
    """Replace the real crew with a fake one. Returns the list of questions it was asked."""
    questions: list[str] = []

    def fake_ask(question, runs, crew=None):
        questions.append(question)
        return {
            "final_answer": "Your pace is stable.\nNext week: add one easy run.",
            "steps": [
                "Supervisor: plan = data -> coach",
                "Data agent: called pace_trend()",
                "Coach: wrote the final answer",
            ],
            "llm_calls": 3,
            "tool_results": {"pace_trend()": {"status": "ok"}},
            "analyst_notes": "",
        }

    st.cache_resource.clear()  # forget a crew cached by an earlier test
    monkeypatch.setattr(crew, "build_crew", lambda model=None: "fake crew")
    monkeypatch.setattr(crew, "ask", fake_ask)
    return questions


def test_app_loads_with_a_sample_runner_and_shows_the_key_numbers(monkeypatch):
    # Opening the page must not build the crew (that would need the API key).
    def must_not_be_called(model=None):
        raise AssertionError("The crew was built when the page opened")

    st.cache_resource.clear()
    monkeypatch.setattr(crew, "build_crew", must_not_be_called)
    app = start_app()

    assert not app.exception
    assert app.subheader[0].value == f"Sample runner {FIRST_RUNNER}"
    numbers = {metric.label: metric.value for metric in app.metric}
    assert list(numbers) == ["Km this week", "Runs this week", "Average pace", "Level"]

    # The numbers on the page are exactly what the analysis functions calculate.
    runs = get_athlete_runs(FIRST_RUNNER)
    peers = compare_to_peers(runs, load_reference())
    assert numbers["Km this week"] == f"{weekly_volume(runs)['last_week_km']} km"
    assert numbers["Average pace"] == f"{peers['comparison']['pace']['you_text']} /km"
    assert numbers["Level"] == peers["level"]
    assert len(app.get("plotly_chart")) == 2


def test_example_button_asks_the_crew_and_shows_how_the_agents_worked(fake_crew):
    app = click(start_app(), "Am I improving?")

    assert not app.exception
    assert fake_crew == ["Am I improving?"]
    question, answer = app.chat_message
    assert question.markdown[0].value == "Am I improving?"
    assert "👟 **Next week:** add one easy run." in answer.markdown[0].value
    assert app.expander[0].label == "How the agents worked"
    assert "3 AI calls" in app.expander[0].caption[0].value


def test_a_new_runner_starts_a_new_chat(fake_crew):
    app = click(start_app(), "Am I improving?")
    assert len(app.session_state.messages) == 2

    app.selectbox[0].set_value(13).run()
    assert app.session_state.messages == []
    assert len(app.chat_message) == 0


def test_rate_limit_shows_a_friendly_message(monkeypatch, fake_crew):
    def rate_limited(question, runs, crew=None):
        raise RateLimitReached("429 (fake)")

    monkeypatch.setattr(crew, "ask", rate_limited)
    app = click(start_app(), "Am I improving?")

    assert not app.exception
    assert "free AI quota is used up" in app.warning[0].value


def test_a_busy_ai_service_shows_a_friendly_message(monkeypatch, fake_crew):
    def busy(question, runs, crew=None):
        raise ModelBusy("503 (fake)")

    monkeypatch.setattr(crew, "ask", busy)
    app = click(start_app(), "Am I improving?")

    assert not app.exception
    assert "AI service is busy" in app.warning[0].value


def test_the_sixth_question_shows_the_limit_message_and_does_not_call_the_crew(fake_crew):
    app = start_app()
    for _ in range(5):
        app = click(app, "Am I improving?")
    assert len(fake_crew) == 5
    assert any("Questions left in this visit: 0 of 5" in c.value for c in app.caption)

    app = click(app, "Am I improving?")

    assert not app.exception
    assert len(fake_crew) == 5  # the 6th question never reached the crew
    assert "You have used your 5 questions" in app.warning[-1].value


def test_the_daily_limit_for_the_whole_app_stops_the_question(monkeypatch, fake_crew):
    monkeypatch.setattr(DailyLimit, "try_use", lambda self: False)  # today's limit reached
    app = click(start_app(), "Am I improving?")

    assert not app.exception
    assert fake_crew == []
    assert "answered all its questions for today" in app.warning[0].value


def test_cloud_secrets_are_copied_into_the_environment(monkeypatch):
    # setenv first, so pytest puts the old value back after the test.
    monkeypatch.setenv("GEMINI_MODEL", "model-from-dotenv")
    app = AppTest.from_file(str(APP_FILE), default_timeout=30)
    app.secrets["GEMINI_MODEL"] = "model-from-cloud-secrets"
    app.run()

    assert not app.exception
    assert os.environ["GEMINI_MODEL"] == "model-from-cloud-secrets"


def test_a_wrong_upload_shows_a_friendly_message():
    app = start_app()
    app.radio[0].set_value("Upload Strava export").run()
    app.file_uploader[0].upload("photo.csv", b"name,age\nAna,30\n", "text/csv").run()

    assert not app.exception
    assert "We couldn't use this file" in app.warning[0].value
    assert len(app.metric) == 0  # the page stops: no numbers for a file we can't read
