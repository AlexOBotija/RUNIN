"""Ask the running coach one question from the terminal.

By default the crew answers (supervisor, data agent, analyst, coach).
Use --single to ask the Step 3 single agent instead, to compare the two.

Examples:
    python scripts/ask.py --athlete 30974 "Is my pace improving?"
    python scripts/ask.py --athlete 30974 --single "Is my pace improving?"
    python scripts/ask.py "How much did I run in the last 4 weeks?"   (random athlete)
    python scripts/ask.py --athlete 30974 --verbose "Am I increasing my distance too fast?"

Each question uses Gemini API calls (crew: 2 to 4, single agent: 1 or 2). The script
prints how many, and how long the answer took.
"""

import argparse
import json
import random
import sys
import time

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from running_coach.agents import crew, single_agent
from running_coach.agents.crew import CrewState
from running_coach.data.loaders import get_athlete_runs, load_sample


def pick_random_athlete() -> int:
    """Return the id of a random athlete from the sample."""
    athletes = load_sample()["athlete"].unique()
    return int(random.choice(athletes))


def print_crew_report(result: CrewState, seconds: float, verbose: bool) -> None:
    """Print the steps taken, the LLM call count, the time and the answer."""
    print("\nSteps taken:")
    for number, step in enumerate(result["steps"], start=1):
        print(f"  {number}. {step}")
    if verbose:
        print("\nTool results:")
        print(json.dumps(result["tool_results"], indent=1) if result["tool_results"] else "  (none)")
        print(f"\nAnalyst notes:\n{result['analyst_notes'] or '  (none)'}")

    print(f"\nLLM calls: {result['llm_calls']}")
    print(f"Time: {seconds:.1f} s")
    answer = result["final_answer"] or "(no answer: the crew stopped before the coach)"
    print(f"\nAnswer:\n{answer}")


def print_single_report(messages: list[BaseMessage], seconds: float, verbose: bool) -> None:
    """Print the tools that were called (with arguments), the LLM call count, the time and the answer."""
    print("\nTools called:")
    tool_count = 0
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                tool_count += 1
                print(f"  - {crew.describe_call(call)}")
        elif isinstance(message, ToolMessage) and verbose:
            print(f"      result: {message.content}")
    if tool_count == 0:
        print("  (none)")

    print(f"\nLLM calls: {single_agent.count_llm_calls(messages)}")
    print(f"Time: {seconds:.1f} s")
    print(f"\nAnswer:\n{messages[-1].text}")


def main() -> None:
    """Read the command-line options, ask the question and print the report."""
    parser = argparse.ArgumentParser(description="Ask the running coach a question.")
    parser.add_argument("question", help='For example: "Is my pace improving?"')
    parser.add_argument("--athlete", type=int, help="Athlete id from the sample (default: random)")
    parser.add_argument("--single", action="store_true", help="Use the Step 3 single agent")
    parser.add_argument("--verbose", action="store_true", help="Also print the tool results")
    args = parser.parse_args()

    athlete_id = args.athlete if args.athlete is not None else pick_random_athlete()
    print(f"Athlete: {athlete_id}")
    print(f"Question: {args.question}")
    print(f"Version: {'single agent' if args.single else 'crew'}")

    try:
        runs = get_athlete_runs(athlete_id)
        start = time.perf_counter()
        if args.single:
            messages = single_agent.ask(args.question, runs)
        else:
            result = crew.ask(args.question, runs)
        seconds = time.perf_counter() - start
    except (ValueError, RuntimeError, GraphRecursionError) as error:
        # ValueError: unknown athlete. RuntimeError: GOOGLE_API_KEY or GEMINI_MODEL not set,
        # or Gemini still rate-limited or busy after all retries (RateLimitReached, ModelBusy).
        # GraphRecursionError: the graph hit LangGraph's step limit.
        sys.exit(f"Error: {error}")

    if args.single:
        print_single_report(messages, seconds, args.verbose)
    else:
        print_crew_report(result, seconds, args.verbose)


if __name__ == "__main__":
    main()
