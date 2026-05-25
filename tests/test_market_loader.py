"""
Tests for the top-level MarketDataLoader: snapshot construction and history
building, including conditional inclusion of futures and bond metadata.
"""
import pandas as pd
import pytest

from src.configs import DataLoaderCfg
from src.dataloaders import MarketDataLoader


@pytest.fixture(scope="module")
def yield_only_loader(data_path, small_date_range):
    s, e = small_date_range
    cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=s, end_date=e,
        max_maturity=10,
        enable_yield=True,
        enable_short_rate=False,
        enable_bonds=False,
        enable_futures=False,
    )
    return MarketDataLoader(cfg)


@pytest.fixture(scope="module")
def full_loader(data_path, small_date_range):
    s, e = small_date_range
    cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=s, end_date=e,
        max_maturity=15,
        enable_yield=True,
        enable_short_rate=True,
        enable_bonds=False,
        enable_futures=True,
    )
    return MarketDataLoader(cfg)


def test_yield_only_loader_has_no_futures(yield_only_loader):
    assert yield_only_loader.futures_store is None
    assert yield_only_loader.bond_metadata_store is None


def test_full_loader_has_futures_and_bond_meta(full_loader):
    assert full_loader.futures_store is not None
    assert full_loader.bond_metadata_store is not None


def test_calendar_is_yield_calendar(full_loader):
    assert full_loader.calendar.start_date == full_loader.yield_store.dates[0]
    assert full_loader.calendar.end_date == full_loader.yield_store.dates[-1]


def test_yield_only_snapshot(yield_only_loader):
    d = yield_only_loader.calendar.dates[100]
    snap = yield_only_loader.get_snapshot(d)
    assert snap.yield_curve is not None
    assert snap.yield_curve.yields.shape == (10,)
    assert snap.short_rate is None
    assert snap.futures is None
    assert snap.bonds_metadata is None


def test_full_snapshot(full_loader):
    d = full_loader.calendar.dates[200]
    snap = full_loader.get_snapshot(d)
    assert snap.yield_curve is not None
    assert snap.short_rate is not None
    if snap.futures is not None:
        assert snap.bonds_metadata is not None
        # bonds_metadata rows align with futures.deliverable_ids_flat
        assert snap.bonds_metadata.features.shape[0] == len(snap.futures.deliverable_ids_flat)
        assert snap.bonds_metadata.ids == list(snap.futures.deliverable_ids_flat)


def test_snapshot_off_calendar_raises(full_loader):
    with pytest.raises(KeyError):
        full_loader.get_snapshot(pd.Timestamp("2020-12-26"))


def test_get_history_simple(full_loader):
    d = full_loader.calendar.dates[100]
    out = full_loader.get_history(d, lookback_days=20, frequency=1)
    # Loader pre-stacks yields + short_rate as the last column.
    assert out.curve_history.shape == (20, full_loader.max_maturity + 1)
    assert out.short_rate is None


def test_get_history_no_short_rate(full_loader):
    d = full_loader.calendar.dates[100]
    out = full_loader.get_history(d, lookback_days=20, frequency=1, return_short_rate=False)
    # When short rate is opted out, the stacked column is dropped.
    assert out.curve_history.shape == (20, full_loader.max_maturity)
    assert out.short_rate is None


def test_get_batch_windows_lengths(full_loader):
    windows = list(full_loader.get_batch_windows(window_days=10, step=1))
    assert len(windows) > 0
    assert all(len(w) <= 10 for w in windows)
    assert sum(len(w) for w in windows) == len(full_loader.calendar.dates)


def test_loader_propagates_business_days(full_loader):
    assert full_loader.business_days_per_year == 252.0
