# src/types/data_types.py
"""
Frozen dataclasses used as the contract between the data layer, the model,
and the pricer.

All dataclasses below are intentionally frozen and carry only tensors / metadata
— treat them as immutable bundles. They are NOT meant to be compared by value
(tensor ``__eq__`` is elementwise and would either misfire or be expensive).
"""
from dataclasses import dataclass
from dataclasses import field

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
import pandas as pd

from .types_utils import Date

# ---------------------------------------------------------------------
# Encoder input contract
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderInputs:
    """
    Encoder input bundle.

    Attributes
    ----------
    curve_history : Union[Tensor, Tuple[Tensor, Tensor]]
        Yield-curve history. For 'simple' encoders: a Tensor of shape (T, M).
        For 'hierarchical' encoders: a tuple (fast_history, slow_history).
    short_rate : Optional[Union[Tensor, Tuple[Tensor, Tensor]]]
        Optional short-rate history aligned with `curve_history`.
        Same shape convention as `curve_history`. May be None.
    dates : Optional[list]
        Optional list of timestamps aligned with `curve_history` rows
        (debug / plotting only — never used by the encoder forward).
    """
    curve_history: Union[Tensor, Tuple[Tensor, Tensor]]
    short_rate: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None
    dates: Optional[list] = None


# ---------------------------------------------------------------------
# Bond features (single source of truth)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BondFeatures:
    """
    Fixed-size bond features evaluated at a given reference date.

    Attributes
    ----------
    ids : List[str]
        Bond identifiers in the same order as rows of `features`.
    features : Tensor
        Feature matrix of shape (N_bonds, d_features).
    feature_names : List[str]
        Column names for `features`, length d_features.
    asof_date : pd.Timestamp
        Reference date at which the features were computed.
    metadata : Dict[str, Any]
        Free-form metadata bag (provenance, warnings, etc.).
    """
    ids: List[str]
    features: Tensor
    feature_names: List[str]
    asof_date: pd.Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Target contracts (what the pricer/loss wants)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class YieldCurveTarget:
    """
    Observed yield curve at a date.

    Attributes
    ----------
    date : pd.Timestamp
        As-of date.
    maturities : Tensor
        Maturities in years, shape (M,).
    yields : Tensor
        Observed yields aligned with `maturities`, shape (M,).
    """
    date: pd.Timestamp
    maturities: Tensor
    yields: Tensor


@dataclass(frozen=True)
class ShortRateTarget:
    """
    Observed short rate at a date.

    Attributes
    ----------
    date : pd.Timestamp
        As-of date.
    rate : Tensor
        Scalar tensor (shape `()` or `(1,)`).
    """
    date: pd.Timestamp
    rate: Tensor


@dataclass(frozen=True)
class BondTarget:
    """
    Bond quotes observed at a date.

    Attributes
    ----------
    ids : List[str]
        Bond identifiers.
    prices : Tensor
        Bond prices, shape (N_bonds,).
    metadata : Dict[str, Any]
        Free-form metadata (coupon schedule, daycount, accrual, settlement, ...).
    """
    ids: List[str]
    prices: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SingleFutureTarget:
    """
    Observation of one futures contract at one date.

    Attributes
    ----------
    id : str
        Contract ticker (e.g. 'TYH2016').
    date : pd.Timestamp
        As-of date of the observation.
    price : Tensor
        Quoted price as a scalar tensor.
    delivery_date : pd.Timestamp
        Contract delivery date.
    deliverable_ids : List[str]
        Bond identifiers in the delivery basket.
    conversion_factors : Tensor
        Per-bond conversion factors, shape (basket_size,).
    metadata : Dict[str, Any]
        Free-form metadata.
    """
    id: str
    date: pd.Timestamp
    price: Tensor
    delivery_date: pd.Timestamp
    deliverable_ids: List[str]
    conversion_factors: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def basket_size(self) -> int:
        return len(self.deliverable_ids)


@dataclass(frozen=True)
class BatchedFuturesTarget:
    """
    Batch of futures observed at the same as-of date, flattened-ragged.

    Attributes
    ----------
    ids : List[str]
        Tickers, length N_futures.
    prices : Tensor
        Quoted prices, shape (N_futures,).
    asof_date : pd.Timestamp
        Shared as-of date for the batch.
    delivery_dates : List[pd.Timestamp]
        Per-future delivery dates, length N_futures.
    basket_lengths : Tensor
        Per-future basket size, shape (N_futures,), dtype long.
    conversion_factors_flat : Tensor
        Conversion factors concatenated across baskets,
        shape (sum_i basket_lengths[i],).
    deliverable_ids_flat : List[str]
        Bond identifiers concatenated across baskets,
        length sum_i basket_lengths[i].
    metadata : Dict[str, Any]
        Free-form metadata.
    """
    ids: List[str]
    prices: Tensor
    asof_date: pd.Timestamp
    delivery_dates: List[pd.Timestamp]
    basket_lengths: Tensor
    conversion_factors_flat: Tensor
    deliverable_ids_flat: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_futures(self) -> int:
        return len(self.ids)

    @property
    def total_deliverables(self) -> int:
        return len(self.deliverable_ids_flat)

    def split_conversion_factors(self) -> List[Tensor]:
        """
        Return the per-future conversion factor tensors as a list.
        """
        lengths = [int(x) for x in self.basket_lengths.tolist()]
        return list(torch.split(self.conversion_factors_flat, lengths, dim=0))

    def split_deliverable_ids(self) -> List[List[str]]:
        """
        Return the per-future deliverable id lists.
        """
        out: List[List[str]] = []
        start = 0
        for n in self.basket_lengths.tolist():
            n = int(n)
            out.append(self.deliverable_ids_flat[start:start + n])
            start += n
        return out


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Pricing template for a single date.

    Attributes
    ----------
    date : Date
        As-of date.
    yield_curve : Optional[YieldCurveTarget]
        Yield curve target. Usually present on every canonical date.
    short_rate : Optional[ShortRateTarget]
        Optional scalar short-rate target.
    bonds : Optional[BondTarget]
        Optional bond price observations.
    bonds_metadata : Optional[BondFeatures]
        Pre-computed bond features for any bonds referenced by `bonds`
        and/or by `futures.deliverable_ids_flat`. None if no bonds/futures
        are active.
    futures : Optional[BatchedFuturesTarget]
        Optional batch of futures targets.
    meta : Dict[str, Any]
        Free-form metadata.

    Notes
    -----
    Missing instruments are represented as None. Conventions (e.g. yield
    maturities) are read from the corresponding target object.
    """
    date: Date

    yield_curve: Optional[YieldCurveTarget] = None
    short_rate: Optional[ShortRateTarget] = None

    bonds: Optional[BondTarget] = None
    bonds_metadata: Optional[BondFeatures] = None
    futures: Optional[BatchedFuturesTarget] = None

    meta: Dict[str, Any] = field(default_factory=dict)
