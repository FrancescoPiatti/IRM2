# src/dataloaders/futures_store.py
from dataclasses import dataclass
from pathlib import Path

from typing import Optional
from typing import Union
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..types import Date
from ..types.data_types import SingleFutureTarget
from ..types.data_types import BatchedFuturesTarget
from ..types.types_utils import merge_single_future_targets


@dataclass
class FuturesStore:
    """
    Store for futures quotes, expirations and delivery baskets.

    Expected CSV files
    ------------------
    futures.csv:
        Date,Root,Ticker,Price
    futures_expirations.csv:
        Ticker,DLV_Date,Year
    futures_dlv.csv:
        Ticker,Bond_ID,CF,Year

    Attributes
    ----------
    quotes_wide : pd.DataFrame
        Date-indexed wide table of prices (columns = Ticker).
    expirations : pd.DataFrame
        Ticker-indexed table with a `DLV_Date` column.
    deliverables : pd.DataFrame
        Long table with columns Ticker, Bond_ID, CF.
    device : torch.device
        Default device for tensors returned by this store.
    dtype : torch.dtype
        Default dtype for floating-point tensors.
    """
    quotes_wide: pd.DataFrame
    expirations: pd.DataFrame
    deliverables: pd.DataFrame
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_csv(
        cls,
        quotes_path: Union[str, Path],
        expirations_path: Union[str, Path],
        deliverables_path: Union[str, Path],
        *,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FuturesStore":
        """
        Load the three CSVs and return an initialised FuturesStore.

        Parameters
        ----------
        quotes_path, expirations_path, deliverables_path : Union[str, Path]
            Paths to the three CSV files.
        start_date, end_date : Optional[Date]
            Inclusive bounds applied to:
            - the quotes dataframe (by quote Date), and
            - expirations / deliverables (by Year, conservatively).
        device : Optional[torch.device]
            Default device. Defaults to CPU.
        dtype : Optional[torch.dtype]
            Default dtype. Defaults to float32.
        """
        dev = device if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32

        # ------------------------------------------------------------
        # Quotes
        # ------------------------------------------------------------
        q = pd.read_csv(quotes_path)
        q["Date"] = pd.to_datetime(q["Date"], errors="raise").dt.normalize()

        if start_date is not None:
            q = q[q["Date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            q = q[q["Date"] <= pd.Timestamp(end_date)]

        q = q.sort_values(["Date", "Ticker"])
        # Defensive: a duplicate (Date, Ticker) row would crash ``pivot``
        # with an unhelpful message. Keep the last occurrence per pair
        # (math_review.md §10).
        q = q.drop_duplicates(subset=["Date", "Ticker"], keep="last")
        quotes_wide = q.pivot(index="Date", columns="Ticker", values="Price").sort_index()

        # ------------------------------------------------------------
        # Expirations
        # ------------------------------------------------------------
        e = pd.read_csv(expirations_path)
        e["DLV_Date"] = pd.to_datetime(e["DLV_Date"], errors="raise").dt.normalize()

        if start_date is not None:
            start_year = pd.Timestamp(start_date).year
            e = e[e["Year"] >= start_year]
        if end_date is not None:
            end_year = pd.Timestamp(end_date).year
            e = e[e["Year"] <= end_year]

        e = e.drop(columns=["Year"]).set_index("Ticker").sort_index()
        if not e.index.is_unique:
            e = e[~e.index.duplicated(keep="last")]

        # ------------------------------------------------------------
        # Deliverables
        # ------------------------------------------------------------
        d = pd.read_csv(deliverables_path)

        if start_date is not None:
            start_year = pd.Timestamp(start_date).year
            d = d[d["Year"] >= start_year]
        if end_date is not None:
            end_year = pd.Timestamp(end_date).year
            d = d[d["Year"] <= end_year]

        d = d.drop(columns=["Year"])
        d["CF"] = pd.to_numeric(d["CF"], errors="raise")

        return cls(
            quotes_wide=quotes_wide,
            expirations=e,
            deliverables=d,
            device=dev,
            dtype=dt,
        )

    # ------------------------------------------------------------------
    # Calendar / lookup helpers
    # ------------------------------------------------------------------

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Quote dates available in the store."""
        return self.quotes_wide.index

    @property
    def tickers(self) -> List[str]:
        """Tickers known to the store (union of quotes and expirations)."""
        return list(self.quotes_wide.columns)

    def get_date_on_or_before(self, date: Date) -> pd.Timestamp:
        """
        Return the most recent date <= `date` for which any futures quote is available.
        """
        ts = pd.Timestamp(date).normalize()
        idx = self.quotes_wide.index
        pos = idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise ValueError(f"No futures quote date <= {ts}")
        return idx[pos]

    def get_price_on_or_before(
            self,
            date: Date,
            *,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            ) -> Tensor:
        """
        Return the row of futures prices at the most recent date <= `date`.

        Returns a 1D tensor with `len(tickers)` entries (NaN for missing).
        """
        dev = device if device is not None else self.device
        out_dtype = dtype if dtype is not None else self.dtype

        ts = self.get_date_on_or_before(date)
        row = self.quotes_wide.loc[ts]
        return torch.as_tensor(row.values, dtype=out_dtype, device=dev)

    def get_prices(
            self,
            date: Date,
            *,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            ) -> Tensor:
        """
        Return all futures prices observed exactly on `date`.

        Raises if `date` is not present in the quote index.
        """
        dev = device if device is not None else self.device
        out_dtype = dtype if dtype is not None else self.dtype

        ts = pd.Timestamp(date).normalize()
        if ts not in self.quotes_wide.index:
            raise KeyError(f"FuturesStore: no futures quotes available exactly on {ts.date()}.")

        row = self.quotes_wide.loc[ts]
        return torch.as_tensor(row.values, dtype=out_dtype, device=dev)

    # ------------------------------------------------------------------
    # Active-ticker selection
    # ------------------------------------------------------------------

    def get_active_tickers(
        self,
        date: Date,
        *,
        max_delivery_years: Optional[float] = None,
        business_days_per_year: float = 252.0,    # accepted for back-compat
    ) -> List[str]:
        """
        Return the tickers considered active on `date`.

        A ticker is active iff:
          - it has a non-NaN quote on `date`, and
          - its `DLV_Date` is strictly after `date`, and
          - if `max_delivery_years` is provided: `DLV_Date - date <= max_delivery_years`
            (delivery within the simulation horizon), measured in
            **calendar years** to match the yield-curve convention
            (math_review.md §2).

        ``business_days_per_year`` is kept on the signature for back-compat
        but is not used in the horizon filter.

        Parameters
        ----------
        date : Date
            As-of date.
        max_delivery_years : Optional[float]
            If provided, drops contracts whose delivery date lies beyond this
            horizon (calendar years).
        business_days_per_year : float
            Ignored. Kept on the signature so old callers don't break.

        Returns
        -------
        List[str]
            Tickers active at `date`, in the column order of `quotes_wide`.
        """
        ts = pd.Timestamp(date).normalize()
        if ts not in self.quotes_wide.index:
            return []

        row = self.quotes_wide.loc[ts]
        priced = row.dropna().index.tolist()

        # Filter by expiration / horizon
        active: List[str] = []
        for t in priced:
            if t not in self.expirations.index:
                continue
            dlv = pd.Timestamp(self.expirations.at[t, "DLV_Date"]).normalize()
            if dlv <= ts:
                continue
            if max_delivery_years is not None:
                # Calendar-day delta -> calendar-year fraction (365.25 d/yr).
                days = (dlv - ts).days
                years = days / 365.25
                if years > float(max_delivery_years):
                    continue
            active.append(t)
        return active

    # ------------------------------------------------------------------
    # Target builders
    # ------------------------------------------------------------------

    def _get_single_future_target(
        self,
        date: Date,
        ticker: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> SingleFutureTarget:
        """
        Build a SingleFutureTarget for `(date, ticker)`.

        Parameters
        ----------
        date : Date
            As-of date.
        ticker : str
            Futures ticker.
        device : Optional[torch.device]
            Device for output tensors. Defaults to `self.device`.
        dtype : Optional[torch.dtype]
            Dtype for floating tensors. Defaults to `self.dtype`.

        Returns
        -------
        SingleFutureTarget
        """
        dev = device if device is not None else self.device
        out_dtype = dtype if dtype is not None else self.dtype

        ts = pd.Timestamp(date).normalize()

        if ticker not in self.quotes_wide.columns:
            raise KeyError(f"FuturesStore: ticker '{ticker}' not in quotes.")
        if ts not in self.quotes_wide.index:
            raise KeyError(f"FuturesStore: no quote date {ts.date()}.")
        if ticker not in self.expirations.index:
            raise KeyError(f"FuturesStore: ticker '{ticker}' missing from expirations.")

        price_val = self.quotes_wide.loc[ts, ticker]
        if pd.isna(price_val):
            raise KeyError(f"FuturesStore: NaN price for {ticker} on {ts.date()}.")

        price = torch.as_tensor(float(price_val), dtype=out_dtype, device=dev)
        delivery_date = pd.Timestamp(self.expirations.at[ticker, "DLV_Date"]).normalize()

        sub = self.deliverables[self.deliverables["Ticker"] == ticker]
        if sub.empty:
            raise KeyError(f"FuturesStore: no deliverables for ticker '{ticker}'.")

        deliverable_ids = sub["Bond_ID"].astype(str).tolist()
        cf_vals = torch.as_tensor(sub["CF"].to_numpy(dtype=np.float32), dtype=out_dtype, device=dev)

        return SingleFutureTarget(
            id=ticker,
            price=price,
            date=ts,
            delivery_date=delivery_date,
            deliverable_ids=deliverable_ids,
            conversion_factors=cf_vals,
            metadata={},
        )

    def get_futures_target(
        self,
        date: Date,
        tickers: Union[str, List[str]],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[SingleFutureTarget, BatchedFuturesTarget]:
        """
        Build a `SingleFutureTarget` for a string ticker, or a
        `BatchedFuturesTarget` for a list of tickers.

        Parameters
        ----------
        date : Date
            As-of date.
        tickers : Union[str, List[str]]
            Ticker or list of tickers to retrieve.
        device : Optional[torch.device]
            Output device.
        dtype : Optional[torch.dtype]
            Output dtype.
        """
        dev = device if device is not None else self.device
        out_dtype = dtype if dtype is not None else self.dtype

        if isinstance(tickers, str):
            return self._get_single_future_target(date, tickers, device=dev, dtype=out_dtype)

        if len(tickers) == 0:
            raise ValueError("get_futures_target: empty ticker list.")

        singles = [
            self._get_single_future_target(date, t, device=dev, dtype=out_dtype)
            for t in tickers
        ]
        return merge_single_future_targets(singles, device=dev)

    def get_batched_futures_target(
        self,
        date: Date,
        *,
        max_delivery_years: Optional[float] = None,
        business_days_per_year: float = 252.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[BatchedFuturesTarget]:
        """
        Return a `BatchedFuturesTarget` covering every active ticker at `date`.

        Active = priced at `date`, not yet delivered, and (optionally) delivery
        within `max_delivery_years` (year-fraction via `business_days_per_year`).

        Returns `None` if no contracts are active at `date`.
        """
        dev = device if device is not None else self.device
        out_dtype = dtype if dtype is not None else self.dtype

        tickers = self.get_active_tickers(
            date,
            max_delivery_years=max_delivery_years,
            business_days_per_year=business_days_per_year,
        )
        if not tickers:
            return None

        result = self.get_futures_target(date, tickers, device=dev, dtype=out_dtype)
        if isinstance(result, SingleFutureTarget):
            # get_futures_target returns single when given a string;
            # we always pass a list here so this branch never triggers, but
            # be defensive.
            result = merge_single_future_targets([result], device=dev)
        return result
