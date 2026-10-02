"""Ask the same questions to the single agent (Step 3) and to the crew (Step 4), and compare.

    python scripts/compare.py                  (athlete 30974, the 5 Step 3 questions)
    python scripts/compare.py --athlete 6601

For each question it prints both answers, the LLM calls and the time taken, then a
summary table. It uses about 30 Gemini API calls, and waits between questions so it
stays under the free-tier limit of 15 requests per minute.
"""

import argparse
import time

from running_coach.agents import crew, single_agent
from running_coach.data.loaders import get_athlete_runs

QUESTIONS = [
    "How much did I run in the last 4 weeks?",
    "Is my pace improving?",
    "Am I running more than other people at my level?",
    "Am I increasing my distance too fast?",
    "What shoes should I buy?",
]

# 1 question = about 6 LLM calls (single 2 + crew 4). 20 s between questions keeps us
# under 15 requests per minute. The pause is not counted in the times.
PAUSE_SECONDS = 20


def count_words(text: str) -> int:
    return len(text.split())


def count_next_week_lines(text: str) -> int:
    """How many lines start with "Next week:" (the coach must write exactly one)."""
    return sum(line.strip().startswith("Next week:") for line in text.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the single agent and the crew.")
    parser.add_argument("--athlete", type=int, default=30974, help="Athlete id (default 30974)")
    args = parser.parse_args()
    runs = get_athlete_runs(args.athlete)

    rows = []
    for number, question in enumerate(QUESTIONS, start=1):
        if number > 1:
            time.sleep(PAUSE_SECONDS)
        print(f"\n{'=' * 70}\nQ{number}. {question}\n{'=' * 70}")

        start = time.perf_counter()
        messages = single_agent.ask(question, runs)
        single_seconds = time.perf_counter() - start
        single_calls = single_agent.count_llm_calls(messages)
        single_answer = messages[-1].text
        print(f"\n--- Single agent ({single_calls} LLM calls, {single_seconds:.1f} s)")
        print(single_answer)

        start = time.perf_counter()
        result = crew.ask(question, runs)
        crew_seconds = time.perf_counter() - start
        crew_answer = result["final_answer"]
        print(f"\n--- Crew ({result['llm_calls']} LLM calls, {crew_seconds:.1f} s)")
        for step in result["steps"]:
            print(f"  * {step}")
        print(f"\nAnalyst notes:\n{result['analyst_notes'] or '(none)'}")
        print(f"\nAnswer:\n{crew_answer}")

        rows.append(
            f"| Q{number} | {single_calls} | {single_seconds:.1f} | {count_words(single_answer)} "
            f"| {result['llm_calls']} | {crew_seconds:.1f} | {count_words(crew_answer)} "
            f"| {count_next_week_lines(crew_answer)} |"
        )

    print("\n| Q | Single calls | Single s | Single words | Crew calls | Crew s | Crew words "
          "| 'Next week:' lines |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
