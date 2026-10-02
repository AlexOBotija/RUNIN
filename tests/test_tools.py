"""Tests for the agent tools. No LLM and no API calls.

We play the LLM's part by hand: we write the tool call message ourselves, and a real
LangGraph ToolNode runs it, injecting the runs from the state. These tests need the
processed data files, so they are skipped if the files don't exist yet.
"""

import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from running_coach.agents.tools import ALL_TOOLS
from running_coach.analysis import metrics
from running_coach.data.loaders import get_athlete_runs, load_reference
from running_coach.data.prepare import REFERENCE_FILE, SAMPLE_FILE

pytestmark = pytest.mark.skipif(
    not (SAMPLE_FILE.exists() and REFERENCE_FILE.exists()),
    reason="Needs the processed data. Run: python -m running_coach.data.prepare",
)

ATHLETE_ID = 30974  # 365 runs, so every function has enough data


class ToolTestState(MessagesState):
    runs: pd.DataFrame


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    return get_athlete_runs(ATHLETE_ID)


@pytest.fixture(scope="module")
def tools_graph():
    """A tiny graph that only runs tools: START -> tools -> END."""
    graph = StateGraph(ToolTestState)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def run_tool(tools_graph, runs: pd.DataFrame, name: str, args: dict) -> ToolMessage:
    """Send one fake tool call (as if the LLM asked for it) and return the tool's reply."""
    call = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])
    result = tools_graph.invoke({"messages": [call], "runs": runs})
    return result["messages"][-1]


@pytest.mark.parametrize(
    "name, args, direct_call",
    [
        ("weekly_volume", {}, lambda r: metrics.weekly_volume(r)),
        ("weekly_volume", {"weeks": 4}, lambda r: metrics.weekly_volume(r, weeks=4)),
        ("pace_trend", {"weeks": 12}, lambda r: metrics.pace_trend(r, weeks=12)),
        ("consistency", {"weeks": 6}, lambda r: metrics.consistency(r, weeks=6)),
        ("load_ramp", {}, lambda r: metrics.load_ramp(r)),
        ("longest_run", {"weeks": 8}, lambda r: metrics.longest_run(r, weeks=8)),
        ("compare_to_peers", {}, lambda r: metrics.compare_to_peers(r, load_reference())),
    ],
)
def test_tool_matches_direct_function_call(tools_graph, runs, name, args, direct_call):
    message = run_tool(tools_graph, runs, name, args)
    assert message.status == "success"
    assert json.loads(message.content) == direct_call(runs)


def test_every_tool_is_tested():
    assert {t.name for t in ALL_TOOLS} == {
        "weekly_volume", "pace_trend", "consistency",
        "load_ramp", "longest_run", "compare_to_peers",
    }


@pytest.mark.parametrize("agent_tool", ALL_TOOLS, ids=lambda t: t.name)
def test_llm_cannot_see_or_set_the_runs(agent_tool):
    # tool_call_schema is exactly what the LLM receives.
    llm_arguments = agent_tool.tool_call_schema.model_json_schema().get("properties", {})
    assert "runs" not in llm_arguments
    assert set(llm_arguments) <= {"weeks"}


@pytest.mark.parametrize("bad_weeks", [0, 53])
def test_bad_weeks_value_goes_back_to_the_llm_as_an_error(tools_graph, runs, bad_weeks):
    # The graph must not crash: the LLM gets an error message and can try again.
    message = run_tool(tools_graph, runs, "weekly_volume", {"weeks": bad_weeks})
    assert message.status == "error"
    assert "weeks" in message.content
