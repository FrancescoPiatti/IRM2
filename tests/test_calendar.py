"""
Tests for the MarketCalendar canonical date helpers.
"""
import pandas as pd
import pytest

from src.dataloaders import MarketCalendar


@pytest.fixture
def calendar():
    dates = pd.to_datetime([
        "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08",
        "2020-01-09", "2020-01-10", "2020-01-13", "2020-01-14",
    ])
    return MarketCalendar(dates=dates)


def test_endpoints(calendar):
    assert calendar.start_date == pd.Timestamp("2020-01-02")
    assert calendar.end_date == pd.Timestamp("2020-01-14")


def test_between_inclusive(calendar):
    out = calendar.between("2020-01-03", "2020-01-08")
    assert list(out) == pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]).to_list()


def test_last_available_on_or_before(calendar):
    # On a calendar date -> same date
    assert calendar.last_available_on_or_before("2020-01-07") == pd.Timestamp("2020-01-07")
    # Off-calendar -> previous calendar date
    assert calendar.last_available_on_or_before("2020-01-05") == pd.Timestamp("2020-01-03")


def test_last_available_on_or_before_before_start(calendar):
    with pytest.raises(ValueError):
        calendar.last_available_on_or_before("2019-12-31")


def test_next_available_after(calendar):
    assert calendar.next_available_after("2020-01-07") == pd.Timestamp("2020-01-08")
    assert calendar.next_available_after("2020-01-04") == pd.Timestamp("2020-01-06")


def test_next_available_after_past_end(calendar):
    with pytest.raises(ValueError):
        calendar.next_available_after("2020-01-14")


def test_training_windows_basic(calendar):
    windows = list(calendar.training_windows(window_size=3, step=1))
    # 9 dates, window=3, step=1 -> 3 windows of size 3
    assert len(windows) == 3
    assert len(windows[0]) == 3
    assert windows[0][0] == pd.Timestamp("2020-01-02")


def test_training_windows_with_step(calendar):
    # step=2 -> every other date; ceil(9 / 2) = 5 dates -> 2 windows of size 3, last size 2
    windows = list(calendar.training_windows(window_size=3, step=2))
    assert len(windows) == 2
    assert len(windows[0]) == 3
    assert len(windows[1]) == 2


def test_training_windows_rejects_bad_args(calendar):
    with pytest.raises(ValueError):
        list(calendar.training_windows(window_size=0))
    with pytest.raises(ValueError):
        list(calendar.training_windows(window_size=3, step=0))
