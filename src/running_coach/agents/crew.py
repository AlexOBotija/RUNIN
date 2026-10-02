"""A crew of agents that answers running questions: supervisor, data, analyst and coach.

    START -> supervisor --(router)--> data agent ----> back to supervisor
                                 +--> analyst -------> back to supervisor
                                 +--> coach ---------> back to supervisor
                                 +--> END (plan finished, or max steps reached)

- supervisor: on its first visit, ONE LLM call makes a plan (does the question need numbers?
  does it need interpretation?). After that, routing is plain Python: no more LLM calls.
- data agent: the ONLY agent with the tools. It collects the numbers the question needs.
- analyst: no tools. It reads the numbers and writes short notes for the coach.
- coach: no tools. It writes the final answer for the runner.

The agents never talk to each other directly: they all read and write the shared state.
"""

import json
import operator
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, ValidationError

from running_coach.agents.tools import ALL_TOOLS
from running_coach.llm import call_with_retry, get_chat_model

# The supervisor can send work to an agent at most this many times. The longest normal
# path needs 3 (data, analyst, coach), so 6 is enough with room to spare. It protects
# the free-tier quota if the routing ever loops (for example after a future change).
MAX_STEPS = 6

SUPERVISOR_PROMPT = """\
You are the supervisor of a team that answers a recreational runner's questions about \
their own training data. You don't answer the question. You decide which team members \
are needed:
- Data agent: calls analysis tools on the runner's data: weekly distance, pace trend, \
consistency (runs per week), recent load (increasing distance too fast), long runs, and \
comparison with other runners at the same level.
- Analyst: interprets the numbers: trends, warning signs, comparison with other runners.
- Coach: always writes the final answer.

Decide:
- needs_data: true if the answer needs the runner's own numbers. False if the data can't \
answer it (for example shoes, food or gear) or it is a general question.
- needs_analysis: true if the numbers need interpretation: trends ("am I improving?"), \
judgements ("too fast?", "too much?", "should I run more?"), comparison with other \
runners, or pain or injury together with training. False for simple facts: "how much", \
"how often", "what was my longest run".
"""

DATA_PROMPT = """\
You are the data agent in a team that helps recreational runners. Your only job is to \
call the analysis tools that give the numbers needed to answer the runner's question.
- Call every tool you need in this one reply (you can call several at once). You won't \
get a second turn.
- Call only the tools the question needs.
- Don't write an answer: other team members interpret the numbers.
"""

ANALYST_PROMPT = """\
You are the analyst in a team that helps recreational runners. You get the runner's \
question and the results of analysis tools (JSON). Write short notes for the coach, not \
for the runner: at most 5 bullet points, about 80 words.
Cover what is relevant to the question:
- the key numbers that answer it, and the dates they cover
- patterns: going up, going down or stable
- warning signs in the training, for example a fast increase in distance or many weeks \
without runs. These are training observations, never medical ones.
- the comparison with other runners at the same level, if there is one
- the limits of the numbers: if a tool note says a result is only a hint (for example the \
pace trend), say so

Rules:
- Use only numbers and labels that appear in the tool results. Never calculate new \
numbers, and never mention a label (for example "high risk") or a tool that is not in \
the results.
- If a result is "not_enough_data", say so and give its note.
- Pace is in minutes per km, and a lower pace is faster. Use the "m:ss" text.
- A "high risk" load label means the runner increased their distance quickly. It is not \
a medical warning.
"""

COACH_PROMPT = """\
You are a friendly running coach for recreational runners. Write the final answer to the \
runner's question. You get the results of analysis tools and, sometimes, notes from an \
analyst.

Rules:
1. Every number in your answer must come from the tool results or the analyst notes. \
Never guess, estimate or calculate new numbers.
2. If a tool result is "not_enough_data", say that the data is not enough to answer and why.
3. If there are no tool results because the question can't be answered from running data \
(for example shoes, food or gear), say so clearly. Then say what you can answer: weekly \
distance, pace, consistency, recent load, long runs and comparison with other runners. \
Your "Next week:" line must be one general tip about the topic of their question (for \
example, how to choose shoes), with no numbers and no brand names.
4. Never give medical advice and never diagnose pain or injuries. If the runner mentions \
pain, an injury or feeling unwell, recommend that they see a doctor or a physiotherapist. \
In that case, your suggestion must never be to run more or harder. A "high risk" load \
label means the runner increased their distance quickly: say it that way, not as a \
medical warning. Don't use the word "risk" for a normal or low load.
5. The data ends on the runner's last run, not today. Say which dates your numbers cover.
6. Pace is in minutes per km, and a lower pace is faster. Use the "m:ss" text.
7. Write simple, friendly English. Keep the whole answer under 130 words. Use only the \
few numbers that matter most for the question. Explain technical values (like a slope) in \
plain words instead of quoting them.
8. If a tool note or the analyst says a result is only a hint, add one short sentence \
that says so and why (for example: "Pace also depends on the type of run, so this is a \
hint, not proof.").
9. End with exactly one line that starts with "Next week:" and gives ONE concrete \
suggestion for next week, based on the runner's numbers when there are any. Say it in \
words, compared with their current training (for example "keep the same weekly distance" \
or "add one rest day"), not as a new number target. Don't give any other suggestions in \
the rest of the answer.
"""


class SupervisorPlan(BaseModel):
    """The supervisor's decision for one question (the LLM fills in these fields)."""

    needs_data: bool = Field(description="True if the answer needs the runner's own numbers.")
    needs_analysis: bool = Field(
        description="True if the numbers need interpretation (trends, judgements, peers, pain)."
    )
    reason: str = Field(description="One short sentence that explains the decision.")


class CrewState(TypedDict):
    """The data shared by all the agents.

    Fields with operator.add are ADDED to (a node returns only its new items);
    the other fields are replaced by the node that writes them.
    """

    question: str  # the runner's question
    runs: pd.DataFrame  # the runner's runs. Only the tools read it; no LLM ever sees it.
    plan: list[str]  # agents to run, in order, e.g. ["data", "coach"]. Set by the supervisor.
    tool_results: dict[str, dict]  # "weekly_volume(weeks=4)" -> tool result. Data agent.
    analyst_notes: str  # short notes for the coach. Analyst.
    final_answer: str  # the answer for the runner. Coach.
    next_agent: str  # where the router sends the graph next. Supervisor.
    steps: Annotated[list[str], operator.add]  # what each agent did, to show the reasoning
    agents_done: Annotated[list[str], operator.add]  # agents that already ran
    step_count: int  # how many times the supervisor has routed. Compared with MAX_STEPS.
    llm_calls: Annotated[int, operator.add]  # each LLM call adds 1


def initial_state(question: str, runs: pd.DataFrame) -> CrewState:
    """The state before any agent runs: the question, the runs and empty results."""
    return {
        "question": question,
        "runs": runs,
        "plan": [],
        "tool_results": {},
        "analyst_notes": "",
        "final_answer": "",
        "next_agent": "",
        "steps": [],
        "agents_done": [],
        "step_count": 0,
        "llm_calls": 0,
    }


def make_plan(needs_data: bool, needs_analysis: bool) -> list[str]:
    """Turn the supervisor's decision into an ordered list of agents.

    Python fixes the order, so the analyst can never run before the data agent, and the
    coach always runs last. Analysis without data is impossible, so it becomes coach only.
    """
    if not needs_data:
        return ["coach"]
    if needs_analysis:
        return ["data", "analyst", "coach"]
    return ["data", "coach"]


def has_usable_numbers(tool_results: dict[str, dict]) -> bool:
    """True if at least one tool returned real numbers (status "ok")."""
    return any(result.get("status") == "ok" for result in tool_results.values())


def choose_next(state: CrewState) -> str:
    """Plain-Python router: return the next agent to run, or END. No LLM call.

    - Max steps reached -> END (a safety stop: it can never loop forever).
    - Otherwise, the first agent in the plan that has not run yet.
    - Skip the analyst when the data agent found no usable numbers: nothing to analyse.
    """
    if state["step_count"] >= MAX_STEPS:
        return END
    for agent in state["plan"]:
        if agent in state["agents_done"]:
            continue
        if agent == "analyst" and not has_usable_numbers(state["tool_results"]):
            continue
        return agent
    return END


# --- The data agent runs the tools itself, with the runs from the state ---

TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}


def describe_call(tool_call: dict) -> str:
    """A short text for one tool call, e.g. "weekly_volume(weeks=4)"."""
    arguments = ", ".join(f"{name}={value}" for name, value in tool_call["args"].items())
    return f"{tool_call['name']}({arguments})"


def run_tool(tool_call: dict, runs: pd.DataFrame) -> dict:
    """Run one tool the LLM asked for. The runs come from the state, never from the LLM.

    A bad request (unknown tool, weeks=0) gives an "error" result instead of a crash.
    """
    tool = TOOLS_BY_NAME.get(tool_call["name"])
    if tool is None:
        return {"status": "error", "note": f"Unknown tool: {tool_call['name']}"}
    try:
        # The LLM's arguments first, then "runs", so the LLM can never replace the data.
        return tool.invoke({**tool_call["args"], "runs": runs})
    except ValidationError as error:
        return {"status": "error", "note": f"Invalid arguments: {error}"}


def to_json(value: object) -> str:
    """Format tool results for a prompt."""
    return json.dumps(value, indent=1)


# --- The graph ---

AGENTS = ["data", "analyst", "coach"]


def build_crew(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Build and compile the crew graph. Pass `model` to use a fake model in tests."""
    model = model or get_chat_model()
    # The supervisor answers with a SupervisorPlan object instead of free text.
    planner = model.with_structured_output(SupervisorPlan)
    # Only the data agent gets the tools. The analyst and the coach use the plain model.
    data_model = model.bind_tools(ALL_TOOLS)

    def supervisor_node(state: CrewState) -> dict:
        """First visit: make the plan (1 LLM call). Every visit: choose the next agent (Python)."""
        update: dict = {"steps": []}
        if not state["plan"]:
            messages = [SystemMessage(SUPERVISOR_PROMPT), HumanMessage(state["question"])]
            decision = call_with_retry(lambda: planner.invoke(messages))
            if decision is None:
                # The LLM gave no usable plan: use every agent (safe, just slower).
                plan = AGENTS
                update["steps"].append("Supervisor: no clear plan from the LLM, using every agent")
            else:
                plan = make_plan(decision.needs_data, decision.needs_analysis)
                update["steps"].append(
                    f"Supervisor: plan = {' -> '.join(plan)}. Reason: {decision.reason}"
                )
            update["plan"] = plan
            update["llm_calls"] = 1
            state = {**state, "plan": plan}  # so choose_next() sees the new plan

        next_agent = choose_next(state)
        if next_agent == END and state["step_count"] >= MAX_STEPS:
            update["steps"].append(f"Supervisor: stopped after {MAX_STEPS} steps (the maximum)")
        elif next_agent == "coach" and "analyst" in state["plan"]:
            if "analyst" not in state["agents_done"]:
                update["steps"].append("Supervisor: skipped the analyst (no usable numbers)")
        update["next_agent"] = next_agent
        update["step_count"] = state["step_count"] + 1
        return update

    def data_node(state: CrewState) -> dict:
        """Ask the LLM which tools to call (1 LLM call), then run them in Python."""
        messages = [SystemMessage(DATA_PROMPT), HumanMessage(state["question"])]
        reply = call_with_retry(lambda: data_model.invoke(messages))
        tool_results = {
            describe_call(call): run_tool(call, state["runs"]) for call in reply.tool_calls
        }
        called = ", ".join(tool_results) or "no tools"
        return {
            "tool_results": tool_results,
            "agents_done": ["data"],
            "llm_calls": 1,
            "steps": [f"Data agent: called {called}"],
        }

    def analyst_node(state: CrewState) -> dict:
        """Read the numbers and write short notes for the coach (1 LLM call, no tools)."""
        content = (
            f"Runner's question: {state['question']}\n\n"
            f"Tool results:\n{to_json(state['tool_results'])}"
        )
        messages = [SystemMessage(ANALYST_PROMPT), HumanMessage(content)]
        reply = call_with_retry(lambda: model.invoke(messages))
        return {
            "analyst_notes": reply.text,
            "agents_done": ["analyst"],
            "llm_calls": 1,
            "steps": ["Analyst: wrote notes for the coach"],
        }

    def coach_node(state: CrewState) -> dict:
        """Write the final answer for the runner (1 LLM call, no tools)."""
        tool_results = to_json(state["tool_results"]) if state["tool_results"] else "none"
        content = (
            f"Runner's question: {state['question']}\n\n"
            f"Tool results:\n{tool_results}\n\n"
            f"Analyst notes:\n{state['analyst_notes'] or 'none'}"
        )
        messages = [SystemMessage(COACH_PROMPT), HumanMessage(content)]
        reply = call_with_retry(lambda: model.invoke(messages))
        return {
            "final_answer": reply.text,
            "agents_done": ["coach"],
            "llm_calls": 1,
            "steps": ["Coach: wrote the final answer"],
        }

    def route(state: CrewState) -> str:
        """The conditional edge: go where the supervisor decided."""
        return state["next_agent"]

    graph = StateGraph(CrewState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data", data_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("coach", coach_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, [*AGENTS, END])
    for agent in AGENTS:
        graph.add_edge(agent, "supervisor")  # every agent reports back to the supervisor
    return graph.compile()


# Each routing is 2 node runs (supervisor + agent), plus the last supervisor visit.
# This LangGraph limit is a second safety net: our MAX_STEPS check always stops first.
RECURSION_LIMIT = 2 * MAX_STEPS + 2


def ask(question: str, runs: pd.DataFrame, crew: CompiledStateGraph | None = None) -> CrewState:
    """Ask the crew one question about these runs. Returns the final state.

    The answer is in result["final_answer"]; result["steps"] shows what each agent did.
    """
    crew = crew or build_crew()
    return crew.invoke(initial_state(question, runs), config={"recursion_limit": RECURSION_LIMIT})
