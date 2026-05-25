# src/dataloaders/calendar.py
from dataclasses import dataclass
import pandas as pd

from typing import Iterable
from typing import List
from typing import Optional

from ..types import Date


@dataclass(frozen=True)
class MarketCalendar:
    """
    Canonical date spine for the project.

    We choose yield curve dates as canonical because:
    - The encoder is fed yield curve history.
    - Yield curves are the most regular daily object you have.
    - Everything else can be missing per-date and is optional.

    The calendar provides:
    - available dates list
    - date slicing
    - nearest <= date lookup
    - next > date lookup
    """

    dates: pd.DatetimeIndex

    def __post_init__(self):
        if not isinstance(self.dates, pd.DatetimeIndex):
            raise TypeError("MarketCalendar.dates must be a pd.DatetimeIndex.")
        if not self.dates.is_monotonic_increasing:
            object.__setattr__(self, "dates", self.dates.sort_values())
        if not self.dates.is_unique:
            object.__setattr__(self, "dates", self.dates[~self.dates.duplicated(keep="last")])


    @property
    def start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates.min())


    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates.max())


    def between(self, start: Optional[Date] = None, end: Optional[Date] = None) -> pd.DatetimeIndex:
        """
        Get all available dates between start and end (inclusive).
        If start or end is None, use calendar start/end.

        Returns
        -------
        pd.DatetimeIndex
            A slice of the canonical date spine (still sorted and unique).
        """
        s = pd.Timestamp(start) if start is not None else self.start_date
        e = pd.Timestamp(end) if end is not None else self.end_date
        sel = self.dates[(self.dates >= s) & (self.dates <= e)]
        return sel


    def last_available_on_or_before(self, date: Date) -> pd.Timestamp:
        """
        Get the last available date on or before the given date.
        """
        ts = pd.Timestamp(date)
        pos = self.dates.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"No available date <= {ts} (earliest={self.start_date})")
        return pd.Timestamp(self.dates[pos])


    def next_available_after(self, date: Date) -> pd.Timestamp:
        """
        Get the next available date after the given date.
        """
        ts = pd.Timestamp(date)
        pos = self.dates.searchsorted(ts, side="right")
        if pos >= len(self.dates):
            raise ValueError(f"No available date > {ts} (latest={self.end_date})")
        return pd.Timestamp(self.dates[pos])


    def training_windows(
        self,
        window_size: int,
        start: Optional[Date] = None,
        end: Optional[Date] = None,
        step: int = 1,
    ) -> Iterable[List[pd.Timestamp]]:
        """
        Yield consecutive (non-overlapping) windows of dates in [start, end] (inclusive).

        Parameters
        ----------
        window_size : int
            Number of dates per window (length of each yielded list).
        start : Optional[Date]
            Start date (inclusive). If None, uses calendar start.
        end : Optional[Date]
            End date (inclusive). If None, uses calendar end.
        step : int
            Subsampling factor applied within the selected date range.
            Example: window_size=20, step=2 -> each window contains 20 dates,
            but spaced by 2 in the underlying calendar (i.e., spans ~40 business days).

        Yields
        ------
        List[pd.Timestamp]
            A list of length `window_size` (last window may be shorter).
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if step <= 0:
            raise ValueError("step must be a positive integer.")

        sel = self.between(start, end)
        sel = sel[::step]  # subsample, but keep window_size as requested

        for i in range(0, len(sel), window_size):
            yield sel[i : i + window_size].to_list()