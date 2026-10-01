"""Create the Gemini chat model used by all agents.

Settings come from the .env file in the project root:
- GOOGLE_API_KEY: your Gemini API key.
- GEMINI_MODEL: the model name, for example "gemini-3.5-flash-lite".
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


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
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)


if __name__ == "__main__":
    # Quick manual check: python -m running_coach.llm
    llm = get_chat_model()
    reply = llm.invoke("Say hello in 5 words")
    print(f"Model: {llm.model}")
    print(f"Reply: {reply.text}")
