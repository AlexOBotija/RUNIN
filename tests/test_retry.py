"""Tests for call_with_retry(). No network: fake functions and a fake sleep."""

import pytest
from langchain_core.exceptions import ModelRateLimitError

from running_coach import llm


class FakeGemini:
    """Raises a rate-limit error for the first `failures` calls, then answers."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise ModelRateLimitError("429 RESOURCE_EXHAUSTED (fake)")
        return "answer"


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Replace time.sleep so the tests don't really wait; record each wait instead."""
    recorded: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", recorded.append)
    return recorded


def test_no_error_means_one_call_and_no_wait(waits):
    fake = FakeGemini(failures=0)
    assert llm.call_with_retry(fake) == "answer"
    assert fake.calls == 1
    assert waits == []


def test_succeeds_on_the_third_try_after_waiting_longer_each_time(waits):
    fake = FakeGemini(failures=2)
    assert llm.call_with_retry(fake, max_tries=3, first_wait_seconds=10) == "answer"
    assert fake.calls == 3
    assert waits == [10, 20]


def test_gives_up_after_the_maximum_tries(waits):
    fake = FakeGemini(failures=99)
    with pytest.raises(RuntimeError, match="still blocked after 3 tries"):
        llm.call_with_retry(fake, max_tries=3)
    assert fake.calls == 3
    assert len(waits) == 2  # no wait after the last try


def test_other_errors_are_not_retried(waits):
    calls = []

    def broken() -> str:
        calls.append(1)
        raise ValueError("a bug in our code")

    with pytest.raises(ValueError, match="a bug in our code"):
        llm.call_with_retry(broken)
    assert len(calls) == 1
    assert waits == []


def test_builtin_library_retries_are_turned_off(monkeypatch):
    # Fake settings: building the model object doesn't call the API.
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    assert llm.get_chat_model().max_retries == 1
