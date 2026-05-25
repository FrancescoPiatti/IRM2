"""
Tests for the type-layer dataclasses and the merge helper.
"""
import pandas as pd
import pytest
import torch

from src.types.data_types import (
    SingleFutureTarget,
    BatchedFuturesTarget,
    MarketSnapshot,
    BondFeatures,
    YieldCurveTarget,
)
from src.types.types_utils import merge_single_future_targets, normalize_date


def _make_single(ticker, asof, dlv, basket_ids, cfs):
    return SingleFutureTarget(
        id=ticker,
        date=pd.Timestamp(asof),
        price=torch.tensor(100.0),
        delivery_date=pd.Timestamp(dlv),
        deliverable_ids=list(basket_ids),
        conversion_factors=torch.as_tensor(cfs, dtype=torch.float32),
    )


def test_single_future_target_basket_size():
    s = _make_single("F1", "2021-01-04", "2021-04-01", ["A", "B", "C"], [0.9, 0.95, 1.0])
    assert s.basket_size == 3


def test_merge_targets_combines_baskets():
    a = _make_single("F1", "2021-01-04", "2021-04-01", ["A", "B"], [0.9, 0.95])
    b = _make_single("F2", "2021-01-04", "2021-07-01", ["C", "D", "E"], [1.0, 1.05, 1.1])
    batch = merge_single_future_targets([a, b])

    assert isinstance(batch, BatchedFuturesTarget)
    assert batch.n_futures == 2
    assert batch.total_deliverables == 5
    assert batch.basket_lengths.tolist() == [2, 3]
    assert batch.deliverable_ids_flat == ["A", "B", "C", "D", "E"]
    assert batch.conversion_factors_flat.tolist() == pytest.approx([0.9, 0.95, 1.0, 1.05, 1.1])


def test_merge_targets_rejects_different_asof():
    a = _make_single("F1", "2021-01-04", "2021-04-01", ["A"], [1.0])
    b = _make_single("F2", "2021-01-05", "2021-04-01", ["B"], [1.0])
    with pytest.raises(ValueError, match="as-of date"):
        merge_single_future_targets([a, b])


def test_merge_targets_rejects_basket_mismatch():
    bad = SingleFutureTarget(
        id="F1",
        date=pd.Timestamp("2021-01-04"),
        price=torch.tensor(100.0),
        delivery_date=pd.Timestamp("2021-04-01"),
        deliverable_ids=["A", "B"],
        conversion_factors=torch.tensor([1.0]),  # wrong length
    )
    with pytest.raises(ValueError, match="Mismatch"):
        merge_single_future_targets([bad])


def test_batched_split_helpers():
    a = _make_single("F1", "2021-01-04", "2021-04-01", ["A", "B"], [0.9, 0.95])
    b = _make_single("F2", "2021-01-04", "2021-07-01", ["C", "D", "E"], [1.0, 1.05, 1.1])
    batch = merge_single_future_targets([a, b])
    cfs = batch.split_conversion_factors()
    ids = batch.split_deliverable_ids()
    assert len(cfs) == 2
    assert cfs[0].tolist() == pytest.approx([0.9, 0.95])
    assert ids[0] == ["A", "B"]
    assert ids[1] == ["C", "D", "E"]


def test_market_snapshot_defaults_to_none():
    snap = MarketSnapshot(date=pd.Timestamp("2021-01-04"))
    assert snap.yield_curve is None
    assert snap.short_rate is None
    assert snap.bonds is None
    assert snap.bonds_metadata is None
    assert snap.futures is None


def test_normalize_date():
    assert normalize_date("2021-01-04T12:00") == pd.Timestamp("2021-01-04")


def test_bond_features_dataclass():
    bf = BondFeatures(
        ids=["A", "B"],
        features=torch.zeros(2, 4),
        feature_names=["a", "b", "c", "d"],
        asof_date=pd.Timestamp("2021-01-04"),
    )
    assert bf.features.shape == (2, 4)
    assert bf.metadata == {}


def test_yield_curve_target_requires_date():
    # date is a required field — building without it should raise TypeError
    with pytest.raises(TypeError):
        YieldCurveTarget(maturities=torch.tensor([1.0]), yields=torch.tensor([0.05]))  # type: ignore[call-arg]
