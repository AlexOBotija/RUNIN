"""One agent that answers running questions by choosing which analysis tools to call.

The graph is the agent loop:

    START -> agent --(did the LLM ask for tools?)-- yes -> tools -> back to agent
                                               +-- no  -> END

- agent node: sends the system prompt + conversation to Gemini, gets one reply.
- tools node: runs the tools Gemini asked for (with the runs from the state).
"""

from typing import Literal

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from running_coach.agents.tools import ALL_TOOLS
from running_coach.llm import call_with_retry, get_chat_model

SYSTEM_PROMPT = """\
You are a running assistant for recreational runners. You answer questions about the \
runner's own training data, using the tools.

Rules:
1. Every number in your answer must come from a tool result. Never guess, estimate or \
calculate new numbers yourself. Call the tools you need first (you can call several at once).
2. If a tool returns "not_enough_data", say that the data is not enough to answer and why \
(use its note). Do not fill the gap with guesses.
3. If the question cannot be answered from running data (for example shoes, food or gear), \
say clearly that you can't answer it from their data. Then give one short general tip with \
no numbers and no brand names, and say what you can answer: weekly distance, pace, \
consistency, recent load, long runs and comparison with other runners.
4. Never give medical advice and never diagnose pain or injuries. If the runner mentions \
pain or an injury, suggest seeing a doctor or a physiotherapist. A "high risk" load label \
means the runner increased their distance quickly: say it that way, not as a medical warning. \
Don't use the word "risk" for a normal or low load.
5. The data ends on the runner's last run, not today. "The last 4 weeks" means the 4 weeks \
ending on that date. Say which dates your numbers cover.
6. Pace is in minutes per km, and a lower pace is faster. When a tool gives a pace as \
"m:ss" text, use that text.
7. Answer in simple, friendly English, in about 120 words at most.
"""

# Each node run is one step, so a normal question takes 3 steps (agent, tools, agent).
# 10 steps allow at most 5 LLM calls, then LangGraph stops the loop with an error.
MAX_STEPS = 10


class AgentState(MessagesState):
    """The data shared by the nodes.

    messages: the conversation so far (from MessagesState). New messages are added, not replaced.
    runs: the runner's runs. Only the tools read it; the LLM never sees it.
    """

    runs: pd.DataFrame


def build_agent(model: BaseChatModel | None = None) -> CompiledStateGraph:
    """Build and compile the agent graph. Pass `model` to use a fake model in tests."""
    model = model or get_chat_model()
    # bind_tools sends the tool descriptions to the LLM with every request.
    model_with_tools = model.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """Ask the LLM what to do next: call tools, or write the final answer."""
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        reply = call_with_retry(lambda: model_with_tools.invoke(messages))
        return {"messages": [reply]}

    def after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        """Go to the tools if the LLM asked for any; otherwise the answer is ready."""
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


def ask(
    question: str, runs: pd.DataFrame, agent: CompiledStateGraph | None = None
) -> list[BaseMessage]:
    """Ask one question about these runs. Returns every message of the conversation.

    The last message is the final answer. The others show the tools that were called.
    """
    agent = agent or build_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(question)], "runs": runs},
        config={"recursion_limit": MAX_STEPS},
    )
    return result["messages"]


def count_llm_calls(messages: list[BaseMessage]) -> int:
    """Each LLM reply is one AIMessage, so counting them counts the LLM calls."""
    return sum(isinstance(message, AIMessage) for message in messages)
