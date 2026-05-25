# src/dataloaders/market_loader.py
from dataclasses import dataclass

from typing import Iterable
from typing import List
from typing import Optional

import pandas as pd
from datetime import timedelta

import torch
from torch import Tensor

from .calendar import MarketCalendar
from .yield_store import YieldCurveStore
from .short_rate_store import ShortRateStore
from .futures_store import FuturesStore
from .bond_metadata_store import BondMetadataStore

from ..configs.config_loader import DataLoaderCfg

from ..types.data_types import EncoderInputs
from ..types.data_types import MarketSnapshot
from ..types.data_types import YieldCurveTarget
from ..types.data_types import ShortRateTarget
from ..types import Date

from ..utils.checks import _check_positive_integer_value


@dataclass
class MarketDataLoader:
    """
    Top-level data loader used by the Trainer.

    Wraps individual stores (yield curve, short rate, futures, bond metadata)
    behind a single object that exposes:
    - the canonical calendar,
    - encoder history builders,
    - per-date `MarketSnapshot` builders,
    - market conventions (year-fraction, dtype, device).

    Attributes
    ----------
    cfg : DataLoaderCfg
        Loader configuration. See `DataLoaderCfg` for field semantics.
    """

    cfg: DataLoaderCfg

    def __post_init__(self):

        # Resolve device and dtype from config
        if isinstance(self.cfg.device, torch.device):
            self.device = self.cfg.device
        else:
            self.device = torch.device(str(self.cfg.device))

        if isinstance(self.cfg.dtype, torch.dtype):
            self.dtype = self.cfg.dtype
        else:
            self.dtype = getattr(torch, str(self.cfg.dtype))

        # Expose enable flags on the loader
        self.enable_yield = bool(self.cfg.enable_yield)
        self.enable_short_rate = bool(self.cfg.enable_short_rate)
        self.enable_bonds = bool(self.cfg.enable_bonds)
        self.enable_futures = bool(self.cfg.enable_futures)
        self.enable_options = bool(self.cfg.enable_options)

        # Market conventions
        self.business_days_per_year = float(self.cfg.business_days_per_year)

        # ------------------------------------------------------------------
        # Yield curve store (always built — defines the canonical calendar)
        # ------------------------------------------------------------------
        try:
            yield_path = self.cfg.data_path + '/yield_curves.csv'
            self.yield_store = YieldCurveStore.from_csv(
                csv_path=yield_path,
                max_maturity=int(self.cfg.max_maturity),
                start_date=self.cfg.start_date,
                end_date=self.cfg.end_date,
                device=self.device,
                dtype=self.dtype,
            )
        except Exception as e:
            raise ImportError('Failed to build Yield Curve Store') from e

        # ------------------------------------------------------------------
        # Short rate store (always built — used as r0 anchor by Trainer)
        # ------------------------------------------------------------------
        try:
            short_rate_path = self.cfg.data_path + '/short_rate.csv'
            self.short_rate_store = ShortRateStore.from_csv(
                csv_path=short_rate_path,
                start_date=self.cfg.start_date,
                end_date=self.cfg.end_date,
                device=self.device,
                dtype=self.dtype,
            )
        except Exception as e:
            raise ImportError('Failed to build Short Rate Store') from e

        # ------------------------------------------------------------------
        # Futures + bond metadata stores (only when futures/bonds are enabled)
        # ------------------------------------------------------------------
        self.futures_store: Optional[FuturesStore] = None
        self.bond_metadata_store: Optional[BondMetadataStore] = None

        if self.enable_futures:
            try:
                self.futures_store = FuturesStore.from_csv(
                    quotes_path=self.cfg.data_path + '/futures.csv',
                    expirations_path=self.cfg.data_path + '/futures_expirations.csv',
                    deliverables_path=self.cfg.data_path + '/futures_dlv.csv',
                    start_date=self.cfg.start_date,
                    end_date=self.cfg.end_date,
                    device=self.device,
                    dtype=self.dtype,
                )
            except Exception as e:
                raise ImportError('Failed to build Futures Store') from e

        if self.enable_bonds or self.enable_futures:
            try:
                self.bond_metadata_store = BondMetadataStore.from_csv(
                    csv_path=self.cfg.data_path + '/bond_meta.csv',
                    business_days_per_year=self.business_days_per_year,
                    device=self.device,
                    dtype=self.dtype,
                )
            except Exception as e:
                raise ImportError('Failed to build Bond Metadata Store') from e

        # Options store: not implemented yet
        # if self.enable_options:
        #     ...

        # Canonical calendar from yield curve store
        self.calendar = MarketCalendar(self.yield_store.dates)

        # Expose convenience values
        self.max_maturity = int(self.yield_store.max_maturity)

        # Precompute a single (N_dates, M+1) history tensor with yields and the
        # short-rate column already concatenated (optimisation_plan §2.2). This
        # makes ``get_history(date)`` an O(lookback) slice instead of a fresh
        # pandas → torch conversion + concat per call.
        yc_df = self.yield_store.data
        sr_df = self.short_rate_store.data

        # Align short-rate rows to the yield-curve calendar. ``reindex`` with
        # ffill matches the "exact-date" behaviour of ShortRateStore.get_rate
        # for dates that lie on the canonical calendar; if a date is missing
        # from the short-rate CSV, the most recent value is used.
        sr_aligned = sr_df.reindex(yc_df.index, method="ffill")
        # Replace residual NaNs (only at the very start, before any sr quote)
        # with zero so the encoder still sees finite numbers; these rows
        # should never actually be sampled because the calendar starts no
        # earlier than the short-rate series.
        sr_aligned = sr_aligned.fillna(0.0)

        combined = pd.concat([yc_df, sr_aligned], axis=1)
        self._full_history = torch.as_tensor(
            combined.values, dtype=self.dtype, device=self.device
        )                                                          # (N_dates, M + 1)
        self._history_dates = yc_df.index

    # ------------------------------------------------------------------
    # Calendar helpers (used in Trainer)
    # ------------------------------------------------------------------

    def get_dates_between(
        self,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
    ) -> List[pd.Timestamp]:
        """
        Return available canonical dates in [start_date, end_date] (inclusive).
        """
        return list(self.calendar.between(start_date, end_date))

    def get_last_available_date(self) -> pd.Timestamp:
        """
        Return the last available canonical date.
        """
        return self.calendar.end_date

    def get_next_available_yield_curve_date(self, date: Date) -> pd.Timestamp:
        """
        Next available canonical date strictly after `date`.
        """
        return self.calendar.next_available_after(date)

    def get_available_yield_curve_date(self, date: Date) -> pd.Timestamp:
        """
        Last available canonical date on or before `date`.
        """
        return self.calendar.last_available_on_or_before(date)

    def _check_valid_start_date(self, date: Optional[Date], lookback: int) -> pd.Timestamp:
        """
        Ensure the requested start date is valid given the encoder lookback.

        The earliest valid start date is the calendar date `lookback` positions
        after `calendar.start_date` — i.e. we step ``lookback`` business days
        forward along the canonical calendar (not ``lookback`` calendar days).
        If `date` is None, the minimum-valid date is returned.
        """
        _check_positive_integer_value(lookback, 'lookback')

        dates = self.calendar.dates
        lb = min(int(lookback), len(dates) - 1)
        min_valid = pd.Timestamp(dates[lb])

        if date is None:
            return min_valid

        requested = pd.Timestamp(date)
        return pd.Timestamp(max(requested, min_valid))

    def get_batch_windows(
        self,
        window_days: int,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        step: int = 1,
    ) -> Iterable[List[pd.Timestamp]]:
        """
        Yield consecutive windows from the canonical calendar.

        Parameters
        ----------
        window_days : int
            Number of dates per window.
        start_date : Optional[Date]
            Window start (inclusive). Defaults to calendar start.
        end_date : Optional[Date]
            Window end (inclusive). Defaults to calendar end.
        step : int
            Subsampling factor inside the calendar (window length stays window_days).

        Yields
        ------
        List[pd.Timestamp]
            Each element is a list of timestamps (last window may be shorter).
        """
        return self.calendar.training_windows(
            window_size=window_days,
            start=start_date,
            end=end_date,
            step=step,
        )

    # ------------------------------------------------------------------
    # History builder
    # ------------------------------------------------------------------

    def get_histories(
        self,
        dates: List[Date],
        *,
        lookback_days: int,
        frequency: int = 1,
        return_short_rate: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Stack encoder histories for several dates into a single batched tensor.

        Each row of the output corresponds to one entry in ``dates`` and has
        shape ``(lookback_days, M[+1])``. The full output is therefore
        ``(B, lookback_days, M[+1])`` and can be fed directly to a batched
        encoder forward (optimisation_plan §6.3).

        Parameters
        ----------
        dates : List[Date]
            Anchor dates. Must all be on the canonical calendar with
            ``lookback_days * frequency`` rows of history available.
        lookback_days, frequency, return_short_rate, device
            Same semantics as ``get_history``.

        Returns
        -------
        Tensor
            Shape ``(B, lookback_days, M[+1])``.
        """
        _check_positive_integer_value(lookback_days, "lookback_days")
        _check_positive_integer_value(frequency, "frequency")

        idx = self._history_dates
        full = self._full_history
        if not return_short_rate:
            full = full[:, : self.max_maturity]

        windows = []
        for d in dates:
            ts = pd.Timestamp(d)
            anchor_pos = idx.searchsorted(ts, side="right") - 1
            if anchor_pos < 0:
                raise ValueError(f"No yield curve <= {ts}")
            if frequency == 1:
                start_pos = max(0, anchor_pos - lookback_days + 1)
                w = full[start_pos:anchor_pos + 1]
            else:
                start_pos = max(0, anchor_pos - (lookback_days - 1) * frequency)
                w = full[start_pos:anchor_pos + 1:frequency]
            if w.shape[0] != lookback_days:
                raise ValueError(
                    f"get_histories: date {ts.date()} produced a {w.shape[0]}-row "
                    f"history; expected {lookback_days}. Consider advancing start_date."
                )
            windows.append(w)

        out = torch.stack(windows, dim=0)
        if device is not None and device != out.device:
            out = out.to(device=device)
        return out


    def get_history(
        self,
        date: Date,
        *,
        lookback_days: int,
        frequency: int = 1,
        return_dates: bool = False,
        return_short_rate: bool = True,
        device: Optional[torch.device] = None,
    ) -> EncoderInputs:
        """
        Build encoder history ending at the anchor ``<= date``.

        Implemented as a slice into a precomputed ``(N_dates, M + 1)``
        history tensor, so the cost is O(lookback_days) memory and no
        per-call pandas conversion (optimisation_plan §2.1).

        Parameters
        ----------
        date : Date
            Anchor date.
        lookback_days : int
            Number of rows in the returned history.
        frequency : int
            Subsampling factor on the calendar index.
        return_dates : bool
            If True, also include the selected dates in the output.
        return_short_rate : bool
            If True, attach the short-rate history (already pre-stacked, so
            the encoder can just consume `curve_history` and ignore
            `short_rate`). When False, the short-rate column is dropped.
        device : Optional[torch.device]
            Optional device override for the output tensor.

        Returns
        -------
        EncoderInputs
            Wrapper around the history tensor of shape ``(T, M[+1])``.
        """
        _check_positive_integer_value(lookback_days, "lookback_days")
        _check_positive_integer_value(frequency, "frequency")

        idx = self._history_dates
        ts = pd.Timestamp(date)
        anchor_pos = idx.searchsorted(ts, side="right") - 1
        if anchor_pos < 0:
            raise ValueError(f"No yield curve <= {ts}")

        if frequency == 1:
            start_pos = max(0, anchor_pos - lookback_days + 1)
            window = self._full_history[start_pos:anchor_pos + 1]
        else:
            start_pos = max(0, anchor_pos - (lookback_days - 1) * frequency)
            window = self._full_history[start_pos:anchor_pos + 1:frequency]

        if device is not None and device != window.device:
            window = window.to(device=device)

        if return_short_rate:
            # Short-rate column is the last column of the precomputed tensor
            # but we pre-stack everything inside curve_history so the model
            # can read it directly (optimisation_plan §2.2). Setting
            # short_rate=None tells `_preprocess_encoder_input` not to
            # concatenate again.
            hist = window
        else:
            hist = window[:, : self.max_maturity]

        dates = list(idx[start_pos:anchor_pos + 1:frequency]) if return_dates else None
        return EncoderInputs(curve_history=hist, dates=dates, short_rate=None)

    # ------------------------------------------------------------------
    # MarketSnapshot builder
    # ------------------------------------------------------------------

    def get_snapshot(
        self,
        date: Date,
        *,
        device: Optional[torch.device] = None,
    ) -> MarketSnapshot:
        """
        Build a MarketSnapshot for a single date.

        Returns
        -------
        MarketSnapshot
            All enabled targets at `date`. Disabled (or absent) targets are None.
        """
        dev = device if device is not None else self.device
        ts = pd.Timestamp(date)

        if ts not in self.calendar.dates:
            raise KeyError(f"MarketDataLoader: {ts.date()} is not in yield curve dates.")

        # 1. Yield curve target
        yield_target = None
        if self.enable_yield:
            maturities = self.yield_store.get_maturities(device=dev)
            yc = self.yield_store.get_curve(date, device=dev)
            yield_target = YieldCurveTarget(date=ts, maturities=maturities, yields=yc)

        # 2. Short rate target (optional)
        short_rate_target = None
        if self.enable_short_rate:
            r = self.short_rate_store.get_rate(date, device=dev)
            short_rate_target = ShortRateTarget(date=ts, rate=r)

        # 3. Bond price targets (not implemented yet)
        bonds_target = None

        # 4. Futures target + bond metadata
        futures_target = None
        bonds_metadata = None
        if self.enable_futures and self.futures_store is not None:
            futures_target = self.futures_store.get_batched_futures_target(
                date,
                max_delivery_years=float(self.max_maturity),
                business_days_per_year=self.business_days_per_year,
                device=dev,
                dtype=self.dtype,
            )

            # Precompute delivery year-fractions once per snapshot and stash
            # them in metadata so the pricer doesn't re-convert (optimisation
            # plan §3.1). The pricer falls back to ``to_year_fraction`` if
            # this key is missing.
            if futures_target is not None and futures_target.n_futures > 0:
                from ..finance.pricer_v2 import to_year_fraction
                dlv_years = to_year_fraction(
                    futures_target.delivery_dates,
                    futures_target.asof_date,
                    business_days_per_year=self.business_days_per_year,
                ).to(device=dev, dtype=self.dtype)
                # MarketSnapshot fields are frozen; mutate the dict in-place.
                futures_target.metadata["delivery_years"] = dlv_years

        if (
            (self.enable_bonds or (self.enable_futures and futures_target is not None))
            and self.bond_metadata_store is not None
        ):
            slot_ids: List[str] = []
            if futures_target is not None:
                slot_ids.extend(futures_target.deliverable_ids_flat)

            if slot_ids:
                # Deduplicate deliverable bonds across overlapping futures
                # baskets (optimisation_plan §3.2). We compute one feature
                # row per *unique* bond and store an int64 ``slot_to_unique``
                # gather index on ``futures.metadata`` so the pricer can
                # expand back to slot positions just before BondNet.
                unique_ids: List[str] = []
                id_to_uniq: dict = {}
                slot_to_unique: List[int] = []
                for bid in slot_ids:
                    u = id_to_uniq.get(bid)
                    if u is None:
                        u = len(unique_ids)
                        id_to_uniq[bid] = u
                        unique_ids.append(bid)
                    slot_to_unique.append(u)

                bonds_metadata = self.bond_metadata_store.get_bond_features(
                    asof_date=date,
                    bond_ids=unique_ids,
                    to_torch=True,
                    device=dev,
                    dtype=self.dtype,
                )

                if futures_target is not None:
                    futures_target.metadata["slot_to_unique"] = torch.as_tensor(
                        slot_to_unique, dtype=torch.long, device=dev
                    )

        # 5. Options target (not implemented yet)

        return MarketSnapshot(
            date=ts,
            yield_curve=yield_target,
            short_rate=short_rate_target,
            bonds=bonds_target,
            bonds_metadata=bonds_metadata,
            futures=futures_target,
            meta={},
        )
