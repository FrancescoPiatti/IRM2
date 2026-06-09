# src/dataloaders/yield_store.py
from dataclasses import dataclass
from pathlib import Path

from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..utils.checks import _check_positive_integer_value
from ..types import Date


@dataclass
class YieldCurveStore:
    """
    Store for daily yield curves.

    Dataframe index = dates
    columns = pillars (SVENY1..SVENY30 or similar)
    values = yields
    """
    data: pd.DataFrame
    max_maturity: int
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        *,
        max_maturity: int = 30,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "YieldCurveStore":

        dev = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32

        # Load CSV
        df = pd.read_csv(
            csv_path,
            parse_dates=["Date"],
            index_col="Date",
        )
        df.index = pd.to_datetime(df.index, errors="raise").normalize()
        df = df.filter(like="SVENY").iloc[:, :max_maturity]

        # CSV yields are in PERCENT (e.g. SVENY01 = 6.10 means a 6.10% yield).
        # The Pricer assumes rates / yields are in DECIMAL, so we convert
        # once at the loader boundary and keep everything else in decimal
        # (math_review.md §1).
        df = df / 100.0

        # (There shouldn't be any na, but still)
        df.dropna(how="all", inplace=True)

        # Ensure sorted unique index
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]

        # Apply date range filter
        if start_date is not None:
            df = df.loc[pd.Timestamp(start_date):]
        if end_date is not None:
            df = df.loc[:pd.Timestamp(end_date)]

        if df.empty:
            raise ValueError("YieldCurveStore: no data in the requested range.")

        return cls(data=df, max_maturity=max_maturity, device=dev, dtype=dtype)


    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.data.index


    def get_curve_on_or_before(
            self, 
            date: Date, 
            *, 
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
            ) -> Tensor:
        """
        Return the yield curve at the most recent available date <= `date`.
        """
        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype

        ts = pd.Timestamp(date)

        pos = self.data.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"No yield curve <= {ts}")
        row = self.data.iloc[pos]
        return torch.as_tensor(row.to_numpy(copy=True), dtype=dtype, device=dev)
    

    def get_curve(
            self, 
            date: Date, 
            *, 
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
            ) -> Tensor:
        """
        Return the yield curve exactly at `date`.

        Raises if `date` is not in the dataset index.
        """
        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype

        ts = pd.Timestamp(date)

        if ts not in self.data.index:
            raise KeyError(f"YieldCurveStore: no curve available exactly on {ts.date()}.")

        row = self.data.loc[ts]
        return torch.as_tensor(row.to_numpy(copy=True), dtype=dtype, device=dev)


    def get_next_curve(
            self, 
            date: Date, 
            *, 
            device: Optional[torch.device] = None, 
            dtype: Optional[torch.dtype] = None
            ) -> Tensor:
        """
        Return the yield curve at the next available date > `date`.
        """
        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype
        ts = pd.Timestamp(date)

        pos = self.data.index.searchsorted(ts, side="right")
        if pos >= len(self.data):
            raise ValueError(f"No yield curve > {ts}")
        row = self.data.iloc[pos]
        return torch.as_tensor(row.to_numpy(copy=True), dtype=dtype, device=dev)


    def get_maturities(
            self, 
            *, 
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
            ) -> Tensor:
        """
        Return the maturities corresponding to the columns, as a 1D tensor of shape (M,).
        """
        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype
        return torch.arange(1, self.max_maturity + 1, device=dev, dtype=dtype)


    def get_encoder_history(
        self,
        date: Date,
        *,
        lookback_days: int,
        frequency: int = 1,
        return_dates: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Union[Tensor, Tuple[List[pd.Timestamp], Tensor]]:
        """
        Return `lookback_days` rows sampled by index stride `frequency`,
        ending at anchor <= date.

        Output shape: (T, M)
        """
        _check_positive_integer_value(lookback_days, "lookback_days")
        _check_positive_integer_value(frequency, "frequency")

        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype
        ts = pd.Timestamp(date)

        idx = self.data.index
        anchor_pos = idx.searchsorted(ts, side="right") - 1
        if anchor_pos < 0:
            raise ValueError(f"No yield curve <= {ts}")

        if frequency == 1:
            start_pos = max(0, anchor_pos - lookback_days + 1)
            window_df = self.data.iloc[start_pos:anchor_pos + 1]
        else:
            start_pos = max(0, anchor_pos - (lookback_days - 1) * frequency)
            positions = np.arange(start_pos, anchor_pos + 1, frequency, dtype=int)
            window_df = self.data.iloc[positions]

        out = torch.as_tensor(window_df.to_numpy(copy=True), dtype=dtype, device=dev)

        if return_dates:
            return window_df.index.to_list(), out
        return out