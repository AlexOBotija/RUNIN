# Project: Running Coach Crew

## About me
I'm a recent Computer Engineering graduate (Python intermediate, pandas and notebooks used before) building my first GitHub portfolio project to find a junior job. I have NEVER built AI agents before. I'm also practising English, so explain things in simple, clear English.

## What we are building
A Streamlit app where a runner asks questions in plain English ("Am I improving?", "Am I increasing my distance too fast?"). A multi-agent system built with LangGraph answers:
- Supervisor (orchestrator): reads the question and decides which agents to call and in what order.
- Data agent: calls Python analysis functions (tools) that return small summaries, never raw data.
- Analyst agent: interprets the numbers, finds patterns and warning signs.
- Coach agent: writes the final answer with a practical suggestion for next week. It never gives medical advice.

## Data
- Public dataset: "A public dataset on long-distance running training in 2019 and 2020" (Afonseca, Watanabe & Duarte, BMClab/UFABC, PeerJ 2022). Columns: datetime, athlete, distance (km), duration (min), gender. Parquet files.
- Working sample: SAMPLE_SIZE athletes from 2019 (default 1000, configurable).
- Reference table: statistics by runner level calculated ONCE from the full 2019 data, so users are compared with runners at their own level.
- Demo: my own Strava export (activities.csv), converted to the same columns.

## Stack
Python 3.11+, pandas, pyarrow, LangGraph + LangChain, Google Gemini API (free tier, model name set in .env), Streamlit, Plotly, pytest. Windows + PowerShell. Git + GitHub.

## How to work with me (always follow)
1. Work ONLY on the step I give you. Do not start future steps.
2. Before writing code, explain the plan for the step: what we will build, why, and the new concepts involved. Then WAIT for my OK.
3. Implement in small pieces. After each piece, explain what the code does in simple words and how I can run or test it.
4. Keep the code simple and readable. Type hints and short docstrings. No clever tricks I can't explain in an interview.
5. Never commit secrets (.env) or data files. Never hard-code API keys.
6. Use clear commit messages (e.g. "Add weekly volume analysis function").
7. At the end of each step: update PROGRESS.md (what was done, decisions taken, what's next), suggest the commit, and give me 3 to 5 questions an interviewer could ask about this step, with short answers.
8. If something I ask is unclear, ask me instead of guessing.
