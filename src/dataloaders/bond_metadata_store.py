from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import torch

from ..types.types_utils import Date
from ..types.data_types import BondFeatures


# Day-count basis used to convert (maturity_date - asof_date).days into a
# year fraction. The whole stack uses 252 days/year so the BondNet
# features, pricer year-fractions, and `Trainer.dt = 1/252` agree
# (user spec — single year convention everywhere). A 1-year bond
# (252 trading-day deltas) therefore yields years_to_maturity = 1.0
# in the BondNet features. Note that we *count* calendar days between
# the two dates and divide by 252 — i.e. 252 acts as the year-fraction
# normaliser, not as a working-day filter.
_DAYS_PER_YEAR = 252.0


@dataclass
class BondMetadataStore:
    """
    Store for static bond metadata and very fast approximate bond feature computation.

    Expected raw CSV columns
    ------------------------
    Bond_ID
    Maturity Date
    Coupon Rate
    Coupon Frequency
    Coupon Type
    First Coupon Date
    Last Coupon Date
    Issue Date

    Notes
    -----
    This implementation intentionally avoids month arithmetic.
    Coupon timing features are approximated from:
        - years to maturity
        - coupon frequency

    Approximation
    -------------
    Let:
        ytm = years to maturity
        f   = coupon frequency
        u   = ytm * f

    Then:
        remaining_coupon_count ~ ceil(u)
        frac = u - floor(u)
        years_to_next_coupon ~ frac / f   (or 1/f when frac == 0)
        years_from_last_coupon = 1/f - years_to_next_coupon
        accrued_fraction = years_from_last_coupon / (1/f)
        accrued_interest_per_100 = accrued_fraction * (coupon_rate / f)

    Assumptions
    -----------
    - No zero-coupon bonds
    - No NaNs in required fields
    - Standard fixed-rate coupon bonds
    """

    data: pd.DataFrame
    business_days_per_year: float = 252.0
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        *,
        business_days_per_year: float = 252.0,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "BondMetadataStore":
    
        dev = device if device is not None else torch.device("cpu")
        dt = dtype if dtype is not None else torch.float32

        df = pd.read_csv(csv_path)
        df = cls._preprocess(df)

        return cls(
            data=df,
            business_days_per_year=float(business_days_per_year),
            device=dev,
            dtype=dt,
        )

    @staticmethod
    def _to_ord(series: pd.Series) -> np.ndarray:
        return (series.astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.int64)

    @classmethod
    def _preprocess(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["Maturity Date"] = pd.to_datetime(df["Maturity Date"], errors="raise").dt.normalize()
        df["First Coupon Date"] = pd.to_datetime(df["First Coupon Date"], errors="coerce").dt.normalize()
        df["Last Coupon Date"] = pd.to_datetime(df["Last Coupon Date"], errors="coerce").dt.normalize()
        df["Issue Date"] = pd.to_datetime(df["Issue Date"], errors="coerce").dt.normalize()

        df = df.rename(
            columns={
                "Maturity Date": "maturity_date",
                "Coupon Rate": "coupon_rate",
                "Coupon Frequency": "coupon_frequency",
                "Coupon Type": "coupon_type",
                "First Coupon Date": "first_coupon_date",
                "Last Coupon Date": "last_coupon_date",
                "Issue Date": "issue_date",
            }
        )

        # CSV coupon_rate is in PERCENT (e.g. 2.25 means 2.25%/yr). Convert
        # to DECIMAL so it matches the rate / yield convention used by the
        # rest of the stack after math_review.md §1.
        df["coupon_rate"] = (
            pd.to_numeric(df["coupon_rate"], errors="raise").astype(np.float32) / 100.0
        )
        df["coupon_frequency"] = pd.to_numeric(df["coupon_frequency"], errors="raise").astype(np.int64)

        # Enforce current assumptions
        # TODO change here
        assert df["maturity_date"].notna().all(), "Missing maturity_date is not implemented."
        assert (df["coupon_frequency"] > 0).all(), "Zero-coupon or non-positive coupon_frequency is not implemented."
        assert df["coupon_rate"].notna().all(), "Missing coupon_rate is not implemented."

        df["maturity_ord"] = cls._to_ord(df["maturity_date"])

        df = df.set_index("Bond_ID").sort_index()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]

        return df

    @property
    def ids(self) -> List[str]:
        return self.data.index.tolist()

    def get_metadata(self, bond_ids: Optional[List[str]] = None) -> pd.DataFrame:
        if bond_ids is None:
            return self.data.copy()
        return self.data.loc[bond_ids].copy()

    def get_bond_features(
        self,
        asof_date: Date,
        bond_ids: List[str],
        *,
        to_torch: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> BondFeatures:
        """
        Compute fast approximate fixed-size bond features at reference date `asof_date`.

        Feature order
        -------------
        0. years_to_maturity
        1. years_to_next_coupon
        2. years_from_last_coupon
        3. coupon_rate
        4. coupon_frequency
        5. remaining_coupon_count
        6. accrued_fraction
        7. accrued_interest_per_100
        """
        ts = pd.Timestamp(asof_date).normalize()
        asof_ord = ts.value // 86_400_000_000_000

        dev = device if device is not None else self.device
        out_dtype = dtype if dtype is not None else self.dtype

        df = self.data.loc[bond_ids]

        maturity_ord = df["maturity_ord"].to_numpy(dtype=np.int64)
        coupon = df["coupon_rate"].to_numpy(dtype=np.float32)
        freq = df["coupon_frequency"].to_numpy(dtype=np.int64).astype(np.float32)

        # Compute years_to_maturity on the BUSINESS-day / 252 convention
        # shared by the whole stack (pricer.to_year_fraction does the
        # same): count weekdays between asof and maturity, divide by 252.
        # A ~1-calendar-year bond gives ~1.04 (261 weekdays / 252); a
        # 3-month gap gives ~0.25 — short horizons are exact, long ones
        # stretch ~3.6% (accepted slack of the convention).
        asof_d64 = np.datetime64(ts, "D")
        mat_d64 = df["maturity_date"].to_numpy().astype("datetime64[D]")
        busdays = np.busday_count(asof_d64, mat_d64)
        years_to_maturity = np.maximum(busdays, 0).astype(np.float32) / _DAYS_PER_YEAR

        # Approximate coupon-cycle features from ytm and frequency only
        u = years_to_maturity * freq                           # ~ remaining coupon periods
        remaining_coupon_count = np.ceil(u).astype(np.float32)

        frac = u - np.floor(u)
        period_years = 1.0 / freq

        years_to_next_coupon = np.where(frac > 1e-8, frac / freq, period_years).astype(np.float32)
        years_from_last_coupon = (period_years - years_to_next_coupon).astype(np.float32)

        accrued_fraction = (years_from_last_coupon / period_years).astype(np.float32)
        accrued_fraction = np.clip(accrued_fraction, 0.0, 1.0)

        # Accrued interest as a fraction of face value, in DECIMAL (matches
        # coupon_rate's units). The legacy name was "accrued_interest_per_100"
        # but with coupon now in decimal the value is no longer per-100.
        accrued_interest = (accrued_fraction * (coupon / freq)).astype(np.float32)

        feats = np.column_stack(
            [
                years_to_maturity.astype(np.float32),
                years_to_next_coupon,
                years_from_last_coupon,
                coupon.astype(np.float32),
                freq.astype(np.float32),
                remaining_coupon_count,
                accrued_fraction,
                accrued_interest,
            ]
        )

        feature_names = [
            "years_to_maturity",
            "years_to_next_coupon",
            "years_from_last_coupon",
            "coupon_rate",
            "coupon_frequency",
            "remaining_coupon_count",
            "accrued_fraction",
            "accrued_interest",
        ]

        out = torch.as_tensor(feats, dtype=out_dtype, device=dev) if to_torch else feats

        return BondFeatures(
            ids=list(bond_ids),
            features=out,
            feature_names=feature_names,
            asof_date=ts,
            metadata={},
        )