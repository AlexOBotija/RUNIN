"""Tests for the agent graph with a FAKE model: no API calls.

The fake model gives answers we wrote in advance, so we can check the loop itself:
agent -> tools -> agent -> END, with the tool reading the runs from the state.
"""

import json
from collections.abc import Iterator

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from running_coach.agents.single_agent import ask, build_agent
from running_coach.analysis import metrics


class FakeModel(GenericFakeChatModel):
    """A fake chat model that accepts tools (the original one doesn't)."""

    def bind_tools(self, tools, **kwargs):
        return self


def fake_agent(replies: Iterator[AIMessage]):
    return build_agent(model=FakeModel(messages=replies))


def make_runs(days: int) -> pd.DataFrame:
    """One 5 km run per day, at 6:00 min/km."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2019-03-01", periods=days, freq="D"),
            "distance_km": 5.0,
            "duration_min": 30.0,
            "pace_min_km": 6.0,
        }
    )


def test_agent_calls_a_tool_then_answers():
    runs = make_runs(days=30)
    replies = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "weekly_volume", "args": {"weeks": 4}, "id": "call_1"}],
            ),
            AIMessage(content="You ran 35 km per week."),
        ]
    )

    messages = ask("How much did I run?", runs, agent=fake_agent(replies))

    # question, tool request, tool result, final answer
    assert [type(m).__name__ for m in messages] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    tool_message = messages[2]
    assert isinstance(tool_message, ToolMessage)
    assert json.loads(tool_message.content) == metrics.weekly_volume(runs, weeks=4)
    assert messages[-1].text == "You ran 35 km per week."


def test_question_without_tools_ends_after_one_llm_call():
    replies = iter([AIMessage(content="I can't answer that from your running data.")])
    messages = ask("What shoes should I buy?", make_runs(days=30), agent=fake_agent(replies))
    assert len(messages) == 2  # question + answer


def test_step_limit_stops_a_model_that_never_stops_calling_tools():
    looping_call = {"name": "load_ramp", "args": {}, "id": "call_x"}
    endless = (AIMessage(content="", tool_calls=[looping_call]) for _ in range(100))
    with pytest.raises(GraphRecursionError):
        ask("Am I doing too much?", make_runs(days=30), agent=fake_agent(endless))
