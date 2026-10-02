"""Ask the running agent one question from the terminal.

Examples:
    python scripts/ask.py --athlete 30974 "Is my pace improving?"
    python scripts/ask.py "How much did I run in the last 4 weeks?"   (random athlete)
    python scripts/ask.py --athlete 30974 --verbose "Am I increasing my distance too fast?"

Each question uses Gemini API calls (usually 2). The script prints how many.
"""

import argparse
import random
import sys

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from running_coach.agents.single_agent import ask
from running_coach.data.loaders import get_athlete_runs, load_sample


def pick_random_athlete() -> int:
    """Return the id of a random athlete from the sample."""
    athletes = load_sample()["athlete"].unique()
    return int(random.choice(athletes))


def print_report(messages: list[BaseMessage], verbose: bool) -> None:
    """Print the tools that were called (with arguments), the LLM call count and the answer."""
    print("\nTools called:")
    tool_count = 0
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                tool_count += 1
                arguments = ", ".join(f"{name}={value}" for name, value in call["args"].items())
                print(f"  - {call['name']}({arguments})")
        elif isinstance(message, ToolMessage) and verbose:
            print(f"      result: {message.content}")
    if tool_count == 0:
        print("  (none)")

    llm_calls = sum(isinstance(message, AIMessage) for message in messages)
    print(f"\nLLM calls: {llm_calls}")
    print(f"\nAnswer:\n{messages[-1].text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the running agent a question.")
    parser.add_argument("question", help='For example: "Is my pace improving?"')
    parser.add_argument("--athlete", type=int, help="Athlete id from the sample (default: random)")
    parser.add_argument("--verbose", action="store_true", help="Also print each tool result")
    args = parser.parse_args()

    athlete_id = args.athlete if args.athlete is not None else pick_random_athlete()
    print(f"Athlete: {athlete_id}")
    print(f"Question: {args.question}")

    try:
        runs = get_athlete_runs(athlete_id)
        messages = ask(args.question, runs)
    except (ValueError, RuntimeError, GraphRecursionError) as error:
        # ValueError: unknown athlete. RuntimeError: rate limit after all retries.
        # GraphRecursionError: the agent hit the step limit.
        sys.exit(f"Error: {error}")

    print_report(messages, args.verbose)


if __name__ == "__main__":
    main()
