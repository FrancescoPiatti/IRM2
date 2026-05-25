# src/dataloaders/short_rate_store.py
from dataclasses import dataclass
from pathlib import Path

from typing import Optional 
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..types import Date

from ..utils.checks import _check_positive_integer_value


@dataclass
class ShortRateStore:
    """
    Store for short rate proxy (e.g., Fed Funds).
    Index = dates, column = 'DFF' or similar.
    """
    data: pd.DataFrame
    # col: str = "DFF"
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        *,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ShortRateStore":

        dev = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32

        # Load CSV
        df = pd.read_csv(
            csv_path,
            parse_dates=["Date"],
            index_col="Date",
        )
        df.index = pd.to_datetime(df.index, errors="raise").normalize()

        
        # (There shouldn't be any na, but still)
        df.dropna(how="all", inplace=True)

        # Ensure sorted unique index
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]

        # Apply date filtering
        if start_date is not None:
            df = df.loc[pd.Timestamp(start_date):]
        if end_date is not None:
            df = df.loc[:pd.Timestamp(end_date)]

        if df.empty:
            raise ValueError("ShortRateStore: no data in the requested range.")

        return cls(data=df, device=dev, dtype=dtype)


    def get_rate(
            self, 
            date: Date, 
            *, 
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
            ) -> Tensor:
        """
        Return the short rate exactly at `date`.

        Raises if `date` is not in the dataset index.
        Output shape: () tensor.
        """
        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype
        ts = pd.Timestamp(date)

        if ts not in self.data.index:
            raise KeyError(f"ShortRateStore: no rate available exactly on {ts.date()}.")

        val = float(self.data.loc[ts].iloc[0])
        return torch.as_tensor(val, dtype=dtype, device=dev)


    def get_rate_on_or_before(
            self, 
            date: Date, 
            *, 
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
            ) -> Tensor:
        """
        Rate at the most recent available date <= date.
        Output shape: () tensor.
        """
        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype
        ts = pd.Timestamp(date)

        idx = self.data.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"No short rate <= {ts}")

        val = float(self.data.iloc[pos].iloc[0])
        return torch.as_tensor(val, dtype=dtype, device=dev)
    

    def get_encoder_history(
        self,
        date: Date,
        *,
        lookback_days: int,
        frequency: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        """
        TODO

        Return `lookback_days` rows sampled by index stride `frequency`,
        ending at anchor <= date.

        Output shape: (T, 1)
        """
        _check_positive_integer_value(lookback_days, "lookback_days")
        _check_positive_integer_value(frequency, "frequency")

        dev = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype
        ts = pd.Timestamp(date)

        idx = self.data.index
        anchor_pos = idx.searchsorted(ts, side="right") - 1
        if anchor_pos < 0:
            raise ValueError(f"No short rate <= {ts}")

        if frequency == 1:
            start_pos = max(0, anchor_pos - lookback_days + 1)
            window_df = self.data.iloc[start_pos : anchor_pos + 1]
        else:
            start_pos = max(0, anchor_pos - (lookback_days - 1) * frequency)
            positions = np.arange(start_pos, anchor_pos + 1, frequency, dtype=int)
            window_df = self.data.iloc[positions]

        out = torch.as_tensor(window_df.values, dtype=dtype, device=dev)  # (T, 1)

        return out