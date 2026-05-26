"""
Tests for the individual data stores: YieldCurveStore, ShortRateStore,
FuturesStore, BondMetadataStore. These exercise the on-disk CSVs in `data2/`.
"""
import pandas as pd
import pytest
import torch

from src.dataloaders import (
    YieldCurveStore,
    ShortRateStore,
    FuturesStore,
    BondMetadataStore,
)


# ---------------------------------------------------------------------------
# YieldCurveStore
# ---------------------------------------------------------------------------


def test_yield_store_loads(data_path, small_date_range):
    s, e = small_date_range
    store = YieldCurveStore.from_csv(
        csv_path=f"{data_path}/yield_curves.csv",
        max_maturity=10,
        start_date=s,
        end_date=e,
    )
    assert len(store.dates) > 100
    curve = store.get_curve(store.dates[0])
    assert curve.shape == (10,)
    assert curve.dtype == torch.float32
    # Values are returned in DECIMAL (post math_review.md §1). A US
    # Treasury yield in the modern era sits comfortably in (-0.01, 0.20).
    assert torch.all((curve > -0.01) & (curve < 0.20))


def test_yield_store_get_curve_off_calendar_raises(data_path, small_date_range):
    s, e = small_date_range
    store = YieldCurveStore.from_csv(csv_path=f"{data_path}/yield_curves.csv",
                                     max_maturity=5, start_date=s, end_date=e)
    with pytest.raises(KeyError):
        store.get_curve(pd.Timestamp("2020-12-25"))


def test_yield_store_maturities(data_path, small_date_range):
    s, e = small_date_range
    store = YieldCurveStore.from_csv(csv_path=f"{data_path}/yield_curves.csv",
                                     max_maturity=7, start_date=s, end_date=e)
    m = store.get_maturities()
    assert m.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_yield_store_encoder_history_shape(data_path, small_date_range):
    s, e = small_date_range
    store = YieldCurveStore.from_csv(csv_path=f"{data_path}/yield_curves.csv",
                                     max_maturity=5, start_date=s, end_date=e)
    d = store.dates[50]
    hist = store.get_encoder_history(d, lookback_days=10, frequency=1)
    assert hist.shape == (10, 5)


def test_yield_store_encoder_history_with_freq(data_path, small_date_range):
    s, e = small_date_range
    store = YieldCurveStore.from_csv(csv_path=f"{data_path}/yield_curves.csv",
                                     max_maturity=5, start_date=s, end_date=e)
    d = store.dates[50]
    hist = store.get_encoder_history(d, lookback_days=5, frequency=3)
    assert hist.shape == (5, 5)


# ---------------------------------------------------------------------------
# ShortRateStore
# ---------------------------------------------------------------------------


def test_short_rate_store_loads(data_path, small_date_range):
    s, e = small_date_range
    store = ShortRateStore.from_csv(csv_path=f"{data_path}/short_rate.csv",
                                    start_date=s, end_date=e)
    d = store.data.index[0]
    r = store.get_rate(d)
    assert r.dim() == 0
    assert r.dtype == torch.float32


def test_short_rate_store_on_or_before(data_path, small_date_range):
    s, e = small_date_range
    store = ShortRateStore.from_csv(csv_path=f"{data_path}/short_rate.csv",
                                    start_date=s, end_date=e)
    # An off-calendar date still returns the most recent <= rate
    r = store.get_rate_on_or_before(pd.Timestamp("2021-12-25"))
    assert r.dim() == 0


# ---------------------------------------------------------------------------
# FuturesStore
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def futures_store(data_path):
    return FuturesStore.from_csv(
        quotes_path=f"{data_path}/futures.csv",
        expirations_path=f"{data_path}/futures_expirations.csv",
        deliverables_path=f"{data_path}/futures_dlv.csv",
        start_date="2021-01-01",
        end_date="2022-12-31",
    )


def test_futures_store_loads(futures_store):
    assert len(futures_store.dates) > 100
    assert len(futures_store.tickers) > 0


def test_futures_store_active_tickers(futures_store):
    d = futures_store.dates[100]
    tickers = futures_store.get_active_tickers(d)
    assert isinstance(tickers, list)
    assert len(tickers) > 0
    # All active tickers have a delivery date strictly after d
    for t in tickers:
        dlv = pd.Timestamp(futures_store.expirations.at[t, "DLV_Date"])
        assert dlv > d


def test_futures_store_active_tickers_with_horizon(futures_store):
    d = futures_store.dates[100]
    long_horizon = futures_store.get_active_tickers(d, max_delivery_years=10.0)
    short_horizon = futures_store.get_active_tickers(d, max_delivery_years=0.1)
    assert len(short_horizon) <= len(long_horizon)


def test_futures_store_batched_target(futures_store):
    d = futures_store.dates[200]
    target = futures_store.get_batched_futures_target(d, max_delivery_years=15.0)
    if target is None:
        pytest.skip("no active futures on this date")
    assert target.n_futures > 0
    assert target.prices.shape == (target.n_futures,)
    assert target.basket_lengths.sum().item() == target.total_deliverables
    assert target.conversion_factors_flat.shape == (target.total_deliverables,)
    assert len(target.deliverable_ids_flat) == target.total_deliverables


def test_futures_store_split_helpers(futures_store):
    d = futures_store.dates[200]
    target = futures_store.get_batched_futures_target(d, max_delivery_years=15.0)
    if target is None:
        pytest.skip("no active futures on this date")
    cf_split = target.split_conversion_factors()
    id_split = target.split_deliverable_ids()
    assert len(cf_split) == target.n_futures
    assert len(id_split) == target.n_futures
    for cf, ids in zip(cf_split, id_split):
        assert cf.shape[0] == len(ids)


# ---------------------------------------------------------------------------
# BondMetadataStore
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bond_meta_store(data_path):
    return BondMetadataStore.from_csv(csv_path=f"{data_path}/bond_meta.csv")


def test_bond_meta_store_loads(bond_meta_store):
    assert len(bond_meta_store.ids) > 0


def test_bond_meta_store_features(bond_meta_store):
    ids = bond_meta_store.ids[:5]
    feats = bond_meta_store.get_bond_features(asof_date="2021-06-01", bond_ids=ids)
    assert feats.features.shape == (5, 8)
    assert feats.features.dtype == torch.float32
    assert feats.ids == list(ids)
    # years_to_maturity should be non-negative
    assert (feats.features[:, 0] >= 0).all()
    # accrued_fraction in [0, 1]
    af = feats.features[:, 6]
    assert (af >= 0).all() and (af <= 1).all()
