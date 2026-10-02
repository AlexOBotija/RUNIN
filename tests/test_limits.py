"""Tests for DailyLimit, the question counter shared by every visitor. A fake "today"."""

from datetime import date

from running_coach.limits import DailyLimit


class FakeToday:
    """A clock we can move forward by hand."""

    def __init__(self, day: date):
        self.day = day

    def __call__(self) -> date:
        return self.day


def test_allows_the_maximum_then_refuses():
    limit = DailyLimit(max_per_day=3, today=FakeToday(date(2026, 10, 2)))

    assert [limit.try_use() for _ in range(3)] == [True, True, True]
    assert limit.try_use() is False
    assert limit.try_use() is False  # still refused, and not counted again


def test_a_new_day_starts_again_from_zero():
    today = FakeToday(date(2026, 10, 2))
    limit = DailyLimit(max_per_day=2, today=today)
    limit.try_use()
    limit.try_use()
    assert limit.try_use() is False

    today.day = date(2026, 10, 3)
    assert limit.try_use() is True
