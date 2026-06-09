"""
Tests for the Pricer (yield curve, short rate, and futures pricing).

These exercise the pricer with synthetic short-rate / latent paths so the tests
are deterministic and do not depend on a trained model.
"""
import math

import pandas as pd
import pytest
import torch

from src.configs import SimpleBondNetCfg
from src.finance.pricer_v2 import Pricer, to_year_fraction
from src.models.bond_net import get_bond_net
from src.types.data_types import (
    BatchedFuturesTarget,
    MarketSnapshot,
    SingleFutureTarget,
    YieldCurveTarget,
    ShortRateTarget,
    BondFeatures,
)


# ---------------------------------------------------------------------------
# to_year_fraction
# ---------------------------------------------------------------------------


def test_to_year_fraction_scalar():
    # Single 252-day year convention used everywhere in the stack
    # (user spec) — 366 calendar days between 2020-01-01 (leap year)
    # and 2021-01-01 -> 366/252 ≈ 1.452.
    out = to_year_fraction(pd.Timestamp("2021-01-01"), pd.Timestamp("2020-01-01"))
    assert out.shape == (1,)
    assert out.item() == pytest.approx(366 / 252.0, rel=1e-5)


def test_to_year_fraction_list():
    out = to_year_fraction(
        [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-07-01")],
        pd.Timestamp("2020-01-01"),
    )
    assert out.shape == (2,)
    assert out[0].item() < out[1].item()


def test_to_year_fraction_zero():
    out = to_year_fraction([pd.Timestamp("2020-01-01")], pd.Timestamp("2020-01-01"))
    assert out.item() == 0.0


# ---------------------------------------------------------------------------
# Yield-curve pricing (analytic: constant short rate -> P(0,T) = exp(-r T))
# ---------------------------------------------------------------------------


def test_price_zcb_constant_rate_recovers_exp():
    """
    With a constant short rate r and a uniform grid, the Monte Carlo discount
    factor should exactly recover exp(-r * T) up to rounding.
    """
    pricer = Pricer(steps_per_year=252)
    r = 0.03
    n_paths, T_years = 16, 5
    steps = T_years * 252
    paths = torch.full((n_paths, steps), r)
    maturities = torch.tensor([1.0, 2.0, 5.0])
    P = pricer.price_zcb(paths, maturities)
    expected = torch.exp(-r * maturities)
    assert torch.allclose(P, expected, atol=1e-5)


def test_price_yield_curve_constant_rate():
    """Yields are now returned in DECIMAL — math_review.md §1."""
    pricer = Pricer(steps_per_year=252)
    r = 0.04
    n_paths, T_years = 8, 3
    steps = T_years * 252
    paths = torch.full((n_paths, steps), r)
    maturities = torch.tensor([1.0, 2.0, 3.0])
    y = pricer.price_yield_curve(paths, maturities)
    # y in decimal; expected = r for every maturity (flat short-rate path).
    assert torch.allclose(y, torch.full_like(y, r), atol=1e-5)


def test_price_short_rate_is_first_step_mean():
    pricer = Pricer(steps_per_year=252)
    paths = torch.zeros(4, 10)
    paths[:, 0] = torch.tensor([0.01, 0.02, 0.03, 0.04])
    r = pricer.price_short_rate(paths)
    assert r.item() == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# Index extraction (no look-ahead)
# ---------------------------------------------------------------------------


def test_extract_latent_idx_no_lookahead():
    ts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    targets = torch.tensor([0.0, 0.3, 0.5, 0.99, 2.0])
    idx = Pricer._extract_latent_idx_at_delivery(ts, targets)
    # 0.0 -> 0, 0.3 -> 1 (right=True), 0.5 -> 2, 0.99 -> 3, 2.0 -> 4
    assert idx.tolist() == [0, 1, 2, 3, 4]


def test_extract_latent_idx_clamped_to_zero():
    ts = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([0.5])
    idx = Pricer._extract_latent_idx_at_delivery(ts, targets)
    assert idx.tolist() == [0]


# ---------------------------------------------------------------------------
# Segmented min (CTD reduction)
# ---------------------------------------------------------------------------


def test_segmented_min_basic():
    values = torch.tensor([[1., 2., 3., 4., 5., 6., 7.],
                           [7., 6., 5., 4., 3., 2., 1.]])
    basket_lengths = torch.tensor([3, 2, 2], dtype=torch.long)
    out = Pricer._segmented_min(values, basket_lengths)
    assert out.shape == (2, 3)
    # Path 0: min(1,2,3)=1, min(4,5)=4, min(6,7)=6
    assert out[0].tolist() == [1.0, 4.0, 6.0]
    # Path 1: min(7,6,5)=5, min(4,3)=3, min(2,1)=1
    assert out[1].tolist() == [5.0, 3.0, 1.0]


# ---------------------------------------------------------------------------
# End-to-end futures pricing
# ---------------------------------------------------------------------------


def _build_synthetic_batch(asof, n_futures=2, baskets=(3, 2), dtype=torch.float32):
    """Manually build a tiny BatchedFuturesTarget for unit tests."""
    asof_ts = pd.Timestamp(asof)
    delivery_dates = [asof_ts + pd.Timedelta(days=180), asof_ts + pd.Timedelta(days=270)]
    total = sum(baskets)
    return BatchedFuturesTarget(
        ids=[f"F{i}" for i in range(n_futures)],
        prices=torch.full((n_futures,), 100.0, dtype=dtype),
        asof_date=asof_ts,
        delivery_dates=delivery_dates,
        basket_lengths=torch.tensor(baskets, dtype=torch.long),
        conversion_factors_flat=torch.linspace(0.8, 1.0, total, dtype=dtype),
        deliverable_ids_flat=[f"B{i}" for i in range(total)],
    )


def test_price_futures_smoke_shapes():
    torch.manual_seed(0)
    bondnet_cfg = SimpleBondNetCfg(
        latent_dim=4, bond_feat_dim=3,
        latent_n_layers=1, latent_n_units=8,
        bond_n_layers=1, bond_n_units=8,
        fusion_n_layers=1, fusion_n_units=8,
        output_positive=True,
    )
    bondnet = get_bond_net(bondnet_cfg)
    target = _build_synthetic_batch("2021-01-04", n_futures=2, baskets=(3, 2))

    n_paths, n_steps = 5, 200
    latent_paths = torch.randn(n_paths, n_steps, 4)
    dt = 1.0 / 64.0
    ts = torch.arange(0.0, n_steps * dt, dt)[:n_steps]
    bond_features = torch.randn(5, 3)

    pricer = Pricer(steps_per_year=64, business_days_per_year=252.0)
    prices = pricer.price_futures(
        bondnet=bondnet,
        bond_features=bond_features,
        latent_paths=latent_paths,
        simulated_times=ts,
        target=target,
    )
    assert prices.shape == (2,)
    assert prices.dtype == torch.float32
    assert torch.isfinite(prices).all()


def test_price_futures_single_to_batched():
    """A SingleFutureTarget should be promoted to a one-element batch."""
    torch.manual_seed(1)
    bondnet_cfg = SimpleBondNetCfg(
        latent_dim=3, bond_feat_dim=2,
        latent_n_layers=1, latent_n_units=4,
        bond_n_layers=1, bond_n_units=4,
        fusion_n_layers=1, fusion_n_units=4,
        output_positive=True,
    )
    bondnet = get_bond_net(bondnet_cfg)
    target = SingleFutureTarget(
        id="F0",
        date=pd.Timestamp("2021-01-04"),
        price=torch.tensor(100.0),
        delivery_date=pd.Timestamp("2021-07-01"),
        deliverable_ids=["B0", "B1", "B2"],
        conversion_factors=torch.tensor([0.9, 0.95, 1.0]),
    )

    n_paths, n_steps = 4, 100
    latent_paths = torch.randn(n_paths, n_steps, 3)
    dt = 1.0 / 64.0
    ts = torch.arange(0.0, n_steps * dt, dt)[:n_steps]
    bond_features = torch.randn(3, 2)

    pricer = Pricer(steps_per_year=64)
    prices = pricer.price_futures(
        bondnet=bondnet,
        bond_features=bond_features,
        latent_paths=latent_paths,
        simulated_times=ts,
        target=target,
    )
    assert prices.shape == (1,)


def test_price_futures_matches_manual_ctd():
    """
    Use a BondNet that just returns the latent_dim=1 path value (a deterministic
    mapping), so we can compute the expected CTD by hand.
    """
    # Build a tiny pricer + a hand-crafted "bondnet" that returns the path value.
    class IdentityBondNet:
        def __call__(self, z, bond_features):
            # z: (n_paths, n_dlv_flat, 1) ; bond_features ignored
            return z.squeeze(-1) + bond_features.sum(-1) * 0.0  # = z[..., 0]

    target = _build_synthetic_batch("2021-01-04", n_futures=2, baskets=(2, 3))

    # 4 paths, simple grid. Force delivery indices to known points.
    n_paths = 4
    ts = torch.linspace(0.0, 5.0, 200)
    latent_paths = torch.zeros(n_paths, ts.shape[0], 1)
    # Set delivery-step values manually
    idx_at_delivery = torch.searchsorted(ts, to_year_fraction(target.delivery_dates, target.asof_date), right=True) - 1
    for j, i in enumerate(idx_at_delivery.tolist()):
        latent_paths[:, i, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0]) + j  # different per path

    bond_features = torch.zeros(target.total_deliverables, 1)
    pricer = Pricer(steps_per_year=int(round(1 / (ts[1] - ts[0]).item())))

    prices = pricer.price_futures(
        bondnet=IdentityBondNet(),
        bond_features=bond_features,
        latent_paths=latent_paths,
        simulated_times=ts,
        target=target,
    )

    # By construction every slot in basket 0 has the same positive value per
    # path (call it z), so the per-slot adjusted prices are z / cf_i. The min
    # over the basket is therefore z / max(cf_i) (largest divisor -> smallest
    # quotient). Then we average over paths.
    cf_split = target.split_conversion_factors()
    expected_0 = (torch.tensor([1., 2., 3., 4.]) / cf_split[0].max()).mean()
    expected_1 = ((torch.tensor([1., 2., 3., 4.]) + 1) / cf_split[1].max()).mean()
    assert prices[0].item() == pytest.approx(expected_0.item(), rel=1e-5)
    assert prices[1].item() == pytest.approx(expected_1.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# price_snapshot smoke
# ---------------------------------------------------------------------------


def test_price_snapshot_yield_only():
    """Yields in DECIMAL post math_review.md §1."""
    pricer = Pricer(steps_per_year=252)
    n_paths, steps = 4, 252 * 2
    r = 0.05
    realisations = torch.full((n_paths, steps), r)
    snap = MarketSnapshot(
        date=pd.Timestamp("2021-01-04"),
        yield_curve=YieldCurveTarget(
            date=pd.Timestamp("2021-01-04"),
            maturities=torch.tensor([1.0, 2.0]),
            yields=torch.tensor([r, r]),
        ),
    )
    out = pricer.price_snapshot(realisations=realisations, snapshot=snap)
    assert out.yield_curve is not None
    assert torch.allclose(out.yield_curve.yields, torch.tensor([r, r]), atol=1e-5)


def test_price_futures_records_diagnostics():
    """`price_futures` should populate `last_bond_stats` and `last_ctd_freq`."""
    torch.manual_seed(2)
    bondnet_cfg = SimpleBondNetCfg(
        latent_dim=3, bond_feat_dim=2,
        latent_n_layers=1, latent_n_units=4,
        bond_n_layers=1, bond_n_units=4,
        fusion_n_layers=1, fusion_n_units=4,
        output_positive=True,
    )
    bondnet = get_bond_net(bondnet_cfg)
    target = _build_synthetic_batch("2021-01-04", n_futures=2, baskets=(2, 3))

    n_paths, n_steps = 6, 80
    latent_paths = torch.randn(n_paths, n_steps, 3)
    dt = 1.0 / 64.0
    ts = torch.arange(0.0, n_steps * dt, dt)[:n_steps]
    bf = torch.randn(5, 2)

    p = Pricer(steps_per_year=64)
    assert p.last_bond_stats is None
    p.price_futures(
        bondnet=bondnet, bond_features=bf,
        latent_paths=latent_paths, simulated_times=ts, target=target,
    )
    assert p.last_bond_stats is not None
    assert set(p.last_bond_stats) == {"mean", "std", "min", "max"}
    assert p.last_ctd_freq is not None
    assert p.last_ctd_freq.shape == (target.total_deliverables,)
    # Each basket's frequencies sum to 1
    splits = torch.split(p.last_ctd_freq, target.basket_lengths.tolist())
    for s in splits:
        assert s.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_price_snapshot_futures_requires_kwargs():
    pricer = Pricer()
    target = _build_synthetic_batch("2021-01-04")
    snap = MarketSnapshot(
        date=pd.Timestamp("2021-01-04"),
        futures=target,
        bonds_metadata=BondFeatures(
            ids=target.deliverable_ids_flat,
            features=torch.zeros(target.total_deliverables, 2),
            feature_names=["a", "b"],
            asof_date=target.asof_date,
        ),
    )
    realisations = torch.zeros(2, 100)
    with pytest.raises(ValueError, match="latent_paths"):
        pricer.price_snapshot(realisations=realisations, snapshot=snap)
