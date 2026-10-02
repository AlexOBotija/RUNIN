"""A daily question limit for the whole app, to protect the free Gemini quota.

The free tier allows 500 LLM requests per day, and one question uses 2 to 4. Each visitor
also has their own limit (in the app's session_state), but a visitor can reset that by
refreshing the page. This counter is shared by EVERY visitor, so it is the real protection.

It lives in memory: when the cloud app restarts (for example after it sleeps), it starts
again from 0. That's fine for a demo app; a real product would store it in a database.
"""

import threading
from collections.abc import Callable
from datetime import date, datetime, timezone


def today_utc() -> date:
    """Today's date in UTC, so the day changes at the same moment on every server."""
    return datetime.now(timezone.utc).date()


class DailyLimit:
    """Count questions for the whole app. The count starts again from 0 every day."""

    def __init__(self, max_per_day: int, today: Callable[[], date] = today_utc):
        self.max_per_day = max_per_day
        self._today = today  # tests pass a fake "today" to check the reset
        self._day = today()
        self._count = 0
        # Several visitors can ask at the same moment (Streamlit runs each visitor in
        # its own thread). The lock lets only one thread change the count at a time.
        self._lock = threading.Lock()

    def try_use(self) -> bool:
        """Count one question and return True, or return False if today's limit is reached."""
        with self._lock:
            if self._today() != self._day:  # a new day: start again from 0
                self._day = self._today()
                self._count = 0
            if self._count >= self.max_per_day:
                return False
            self._count += 1
            return True
