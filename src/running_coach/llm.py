"""Create the Gemini chat model used by all agents.

Settings come from the .env file in the project root:
- GOOGLE_API_KEY: your Gemini API key.
- GEMINI_MODEL: the model name, for example "gemini-3.5-flash-lite".

Free-tier limits (see PROGRESS.md): 15 requests per minute and 500 per day. When we go
over a limit, Gemini answers with a rate-limit error (HTTP 429). call_with_retry()
waits and tries again.
"""

import os
import time
from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv
from langchain_core.exceptions import ModelRateLimitError
from langchain_google_genai import ChatGoogleGenerativeAI

MAX_TRIES = 3
FIRST_WAIT_SECONDS = 10  # then 20 s: the wait doubles after each failed try

T = TypeVar("T")


class RateLimitReached(RuntimeError):
    """Gemini still says "rate limit" after every retry. The app shows a friendly message.

    It is a RuntimeError, so code that catches RuntimeError (like scripts/ask.py) still works.
    """


def _get_setting(name: str) -> str:
    """Return the value of an environment variable, or raise a clear error if it is missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env in the project root "
            f"and fill in {name}=..."
        )
    return value


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Load settings from .env and return a ready-to-use Gemini chat model."""
    load_dotenv()
    api_key = _get_setting("GOOGLE_API_KEY")
    model_name = _get_setting("GEMINI_MODEL")
    # max_retries=1 turns off the library's own hidden retries (by default it tries
    # 6 times). call_with_retry() is our only retry, so every extra request is visible.
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, max_retries=1)


def call_with_retry(
    func: Callable[[], T],
    max_tries: int = MAX_TRIES,
    first_wait_seconds: float = FIRST_WAIT_SECONDS,
) -> T:
    """Call func(). If Gemini says "rate limit", wait and try again (10 s, then 20 s).

    Only rate-limit errors are retried. Other errors (a wrong API key, a bug in our
    code) would fail again in the same way, so they are raised at once.
    """
    for attempt in range(1, max_tries + 1):
        try:
            return func()
        except ModelRateLimitError as error:
            if attempt == max_tries:
                raise RateLimitReached(
                    f"Gemini rate limit: still blocked after {max_tries} tries. "
                    "Per-minute limit: wait a minute and ask again. "
                    "Daily limit: try again tomorrow."
                ) from error
            wait = first_wait_seconds * 2 ** (attempt - 1)
            print(f"Rate limit hit (try {attempt} of {max_tries}). Waiting {wait:g} s...")
            time.sleep(wait)
    raise ValueError("max_tries must be at least 1")


if __name__ == "__main__":
    # Quick manual check: python -m running_coach.llm
    llm = get_chat_model()
    reply = llm.invoke("Say hello in 5 words")
    print(f"Model: {llm.model}")
    print(f"Reply: {reply.text}")
