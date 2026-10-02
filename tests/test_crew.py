"""Tests for the crew. None of them call the real API.

- Routing tests call the plain-Python functions directly (no LLM, no graph).
- Graph tests use a FAKE model that gives answers we wrote in advance.
"""

import json

import pandas as pd
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END
from pydantic import Field

from running_coach.agents.crew import (
    MAX_STEPS,
    build_crew,
    choose_next,
    initial_state,
    make_plan,
)
from running_coach.agents.tools import ALL_TOOLS
from running_coach.analysis import metrics


class FakeModel(GenericFakeChatModel):
    """A fake chat model that accepts tools and remembers who called it with which tools.

    bind_tools() returns a copy that knows its tool names. The copies share the same
    replies and the same `calls` list, so the test sees every LLM call in order.
    """

    tool_names: list[str] = Field(default_factory=list)
    calls: list[tuple[str, list[str]]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        names = [getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        first_prompt_line = messages[0].content.splitlines()[0]
        self.calls.append((first_prompt_line, self.tool_names))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def plan_reply(needs_data: bool, needs_analysis: bool) -> AIMessage:
    """What the supervisor's structured output looks like to the fake model."""
    args = {"needs_data": needs_data, "needs_analysis": needs_analysis, "reason": "test"}
    return AIMessage(content="", tool_calls=[{"name": "SupervisorPlan", "args": args, "id": "p"}])


def tool_reply(name: str, args: dict) -> AIMessage:
    """The data agent asking for one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "t"}])


def run_crew(replies: list[AIMessage], runs: pd.DataFrame) -> tuple[dict, FakeModel]:
    model = FakeModel(messages=iter(replies))
    crew = build_crew(model=model)
    result = crew.invoke(initial_state("A question", runs), config={"recursion_limit": 20})
    return result, model


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


def state_with(**changes) -> dict:
    """A crew state with some fields changed."""
    return {**initial_state("A question", make_runs(days=30)), **changes}


OK_RESULT = {"weekly_volume(weeks=4)": {"status": "ok", "total_km": 140.0}}
NO_DATA_RESULT = {"load_ramp()": {"status": "not_enough_data", "note": "Need 28 days."}}


# --- make_plan: the supervisor's decision -> ordered list of agents ---


def test_plan_for_a_simple_question_skips_the_analyst():
    assert make_plan(needs_data=True, needs_analysis=False) == ["data", "coach"]


def test_plan_for_a_complex_question_uses_every_agent():
    assert make_plan(needs_data=True, needs_analysis=True) == ["data", "analyst", "coach"]


def test_plan_without_data_is_coach_only_even_if_analysis_was_asked():
    assert make_plan(needs_data=False, needs_analysis=False) == ["coach"]
    assert make_plan(needs_data=False, needs_analysis=True) == ["coach"]


# --- choose_next: the plain-Python router ---


def test_router_follows_the_plan_in_order():
    plan = ["data", "analyst", "coach"]
    assert choose_next(state_with(plan=plan)) == "data"
    after_data = state_with(plan=plan, agents_done=["data"], tool_results=OK_RESULT)
    assert choose_next(after_data) == "analyst"
    after_analyst = {**after_data, "agents_done": ["data", "analyst"]}
    assert choose_next(after_analyst) == "coach"


def test_router_ends_when_every_agent_in_the_plan_is_done():
    done = state_with(plan=["data", "coach"], agents_done=["data", "coach"])
    assert choose_next(done) == END


def test_simple_question_route_goes_from_data_straight_to_coach():
    state = state_with(plan=["data", "coach"], agents_done=["data"], tool_results=OK_RESULT)
    assert choose_next(state) == "coach"


def test_router_skips_the_analyst_when_there_are_no_usable_numbers():
    plan = ["data", "analyst", "coach"]
    no_numbers = state_with(plan=plan, agents_done=["data"], tool_results=NO_DATA_RESULT)
    assert choose_next(no_numbers) == "coach"
    no_tools_called = state_with(plan=plan, agents_done=["data"], tool_results={})
    assert choose_next(no_tools_called) == "coach"


def test_router_goes_to_the_end_when_the_step_count_reaches_the_maximum():
    # The coach has not run yet, but the safety limit wins.
    state = state_with(plan=["data", "coach"], step_count=MAX_STEPS)
    assert choose_next(state) == END
    # One step before the limit, it still routes normally.
    assert choose_next({**state, "step_count": MAX_STEPS - 1}) == "data"


# --- The whole graph with a fake model ---


def test_simple_question_skips_the_analyst_and_uses_3_llm_calls():
    runs = make_runs(days=30)
    replies = [
        plan_reply(needs_data=True, needs_analysis=False),
        tool_reply("weekly_volume", {"weeks": 4}),
        AIMessage(content="You ran 140 km.\nNext week: run 4 easy runs."),
    ]
    result, _ = run_crew(replies, runs)

    assert result["plan"] == ["data", "coach"]
    assert result["agents_done"] == ["data", "coach"]
    assert result["analyst_notes"] == ""
    assert result["llm_calls"] == 3
    assert result["final_answer"].endswith("Next week: run 4 easy runs.")
    # The tool ran on the runs from the state.
    assert result["tool_results"] == {
        "weekly_volume(weeks=4)": metrics.weekly_volume(runs, weeks=4)
    }
    assert result["steps"][1] == "Data agent: called weekly_volume(weeks=4)"


def test_complex_question_uses_every_agent_and_4_llm_calls():
    replies = [
        plan_reply(needs_data=True, needs_analysis=True),
        tool_reply("load_ramp", {}),
        AIMessage(content="- Ratio is normal."),
        AIMessage(content="Your load is normal.\nNext week: keep the same distance."),
    ]
    result, _ = run_crew(replies, make_runs(days=30))

    assert result["agents_done"] == ["data", "analyst", "coach"]
    assert result["analyst_notes"] == "- Ratio is normal."
    assert result["llm_calls"] == 4


def test_question_without_data_goes_straight_to_the_coach():
    replies = [
        plan_reply(needs_data=False, needs_analysis=False),
        AIMessage(content="I can't answer that from your data.\nNext week: ..."),
    ]
    result, _ = run_crew(replies, make_runs(days=30))

    assert result["agents_done"] == ["coach"]
    assert result["tool_results"] == {}
    assert result["llm_calls"] == 2


def test_only_the_data_agent_has_the_analysis_tools():
    replies = [
        plan_reply(needs_data=True, needs_analysis=True),
        tool_reply("load_ramp", {}),
        AIMessage(content="- notes"),
        AIMessage(content="answer"),
    ]
    _, model = run_crew(replies, make_runs(days=30))

    tools_by_agent = {prompt_line: names for prompt_line, names in model.calls}
    analysis_tools = [tool.name for tool in ALL_TOOLS]
    assert len(tools_by_agent) == 4  # supervisor, data, analyst, coach
    for prompt_line, names in tools_by_agent.items():
        if prompt_line.startswith("You are the data agent"):
            assert names == analysis_tools
        else:
            # The supervisor's only "tool" is its plan format; analyst and coach have none.
            assert not set(names) & set(analysis_tools), prompt_line


def test_bad_tool_arguments_become_an_error_result_not_a_crash():
    replies = [
        plan_reply(needs_data=True, needs_analysis=True),
        tool_reply("weekly_volume", {"weeks": 0}),
        AIMessage(content="I could not get your numbers.\nNext week: ..."),
    ]
    result, _ = run_crew(replies, make_runs(days=30))

    error = result["tool_results"]["weekly_volume(weeks=0)"]
    assert error["status"] == "error"
    json.dumps(result["tool_results"])  # still plain data the prompts can use
    # No usable numbers, so the router skipped the analyst.
    assert result["agents_done"] == ["data", "coach"]
    assert "Supervisor: skipped the analyst (no usable numbers)" in result["steps"]
