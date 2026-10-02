"""LangChain tools that let an agent use the Step 2 analysis functions.

The LLM reads each tool's name, docstring and arguments to decide which tool to call,
so the docstrings below are written for the LLM.

The runner's data is NOT an argument the LLM can fill in. LangGraph injects it from the
agent state (state["runs"]), so the LLM only chooses WHICH tool to call and simple
settings like the number of weeks. It can never invent or edit the data.
"""

from typing import Annotated

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from running_coach.analysis import metrics
from running_coach.data.loaders import load_reference

# Filled in by LangGraph from state["runs"]. Hidden from the LLM.
Runs = Annotated[pd.DataFrame, InjectedState("runs")]

# The only setting the LLM chooses. The limits stop values like 0 or 1000 weeks.
Weeks = Annotated[
    int, Field(ge=1, le=52, description="How many 7-day weeks to look back (1 to 52).")
]


@tool
def weekly_volume(runs: Runs, weeks: Weeks = 8) -> dict:
    """Kilometres run per week, the total, the weekly average, and the % change between the last two weeks.

    Use it for questions about how much the runner runs: total or weekly distance, and
    whether their volume went up or down.
    """
    return metrics.weekly_volume(runs, weeks=weeks)


@tool
def pace_trend(runs: Runs, weeks: Weeks = 8) -> dict:
    """Average pace per week and whether it is improving, stable or getting worse.

    Use it for questions about speed, pace, or getting faster. Pace is minutes per km,
    so a lower pace is faster. Weekly pace is noisy, so use at least 8 weeks (the default)
    for a trend, unless the runner asks about a specific period.
    """
    return metrics.pace_trend(runs, weeks=weeks)


@tool
def consistency(runs: Runs, weeks: Weeks = 8) -> dict:
    """Number of runs per week, the weekly average, and weeks with zero runs.

    Use it for questions about how often or how regularly the runner runs.
    """
    return metrics.consistency(runs, weeks=weeks)


@tool
def load_ramp(runs: Runs) -> dict:
    """Compare the km of the last 7 days with the runner's usual week (last 28 days).

    Returns the ratio (last 7 days / usual week) and a label: low, normal or high risk.
    Use it for questions about increasing distance too fast or doing too much lately.
    """
    return metrics.load_ramp(runs)


@tool
def longest_run(runs: Runs, weeks: Weeks = 8) -> dict:
    """The longest run in the last weeks, and its share of that week's distance.

    Use it for questions about long runs.
    """
    return metrics.longest_run(runs, weeks=weeks)


@tool
def compare_to_peers(runs: Runs) -> dict:
    """Find the runner's level and compare them with other runners at the same level.

    Compares weekly km, pace and runs per week with the 25th, 50th and 75th percentiles
    of all 2019 runners at that level. Use it for questions about other runners.
    """
    return metrics.compare_to_peers(runs, load_reference())


ALL_TOOLS = [weekly_volume, pace_trend, consistency, load_ramp, longest_run, compare_to_peers]
