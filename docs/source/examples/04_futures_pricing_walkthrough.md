# Example 4 — Futures pricing, end-to-end

This example walks through `Pricer.price_futures` step by step. The goal is to
demystify the cheapest-to-deliver Monte Carlo: every tensor shape is shown.

We will:

1. fetch a real snapshot from `data/`,
2. simulate latent paths from a freshly-initialised model,
3. call `Pricer.price_futures` and verify shapes,
4. cross-check the CTD reduction with a manual `min`.

## 4.1 Build the pieces

```python
from datetime import datetime
import torch

from src.configs import (
    DataLoaderCfg, EncoderCfg, NSDECfg, SimpleBondNetCfg
)
from src.dataloaders import MarketDataLoader
from src.models.short_rate_model import ShortRateModel
from src.finance.pricer_v2 import Pricer, to_year_fraction

torch.manual_seed(0)

dl = MarketDataLoader(DataLoaderCfg(
    data_path="data",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 12, 31),
    max_maturity=15,
    enable_yield=True,
    enable_short_rate=True,
    enable_futures=True,
))

date = dl.calendar.dates[200]
snap = dl.get_snapshot(date)

print("date           :", snap.date.date())
print("active futures :", snap.futures.ids)
print("baskets        :", snap.futures.basket_lengths.tolist())
print("dlv flat       :", len(snap.futures.deliverable_ids_flat))
print("bond features  :", tuple(snap.bonds_metadata.features.shape))
```

Typical output:

```
date           : 2021-10-19
active futures : ['FVH2022', 'FVM2022', 'FVZ2021', 'TUH2022', 'TUM2022', 'TUZ2021', 'TYH2022', 'TYM2022', 'TYZ2021']
baskets        : [10, 11, 11, 12, 11, 12, 20, 20, 19]
dlv flat       : 126
bond features  : (126, 8)
```

## 4.2 Wire up the model

```python
bondnet_cfg = SimpleBondNetCfg(
    latent_dim=8, bond_feat_dim=8,
    latent_n_layers=1, latent_n_units=16,
    bond_n_layers=1, bond_n_units=16,
    fusion_n_layers=1, fusion_n_units=16,
    output_positive=True,
)
model = ShortRateModel(
    name="walkthrough",
    encoder=EncoderCfg(mode="simple"),
    nsde=NSDECfg(type="simple"),
    bondnet=bondnet_cfg,
    latent_dim=8,
)
```

## 4.3 Simulate latent paths

```python
n_paths = 32
horizon = 15.0
dt = 1.0 / 64.0
ts = torch.arange(0.0, horizon + dt, dt)        # (n_steps,)
n_steps = ts.shape[0]

# Fake an encoded latent state for this walkthrough
z0 = torch.randn(model.latent_dim)
latent_paths = model.simulate(
    z0, n_paths=n_paths, horizon=horizon, dt=dt, decode=False
)
print("ts          :", ts.shape, ts[0].item(), "...", ts[-1].item())
print("latent_paths:", latent_paths.shape)       # (n_paths, n_steps, d_z)
```

## 4.4 Call the pricer

```python
pricer = Pricer(
    steps_per_year=int(round(1 / dt)),
    business_days_per_year=dl.business_days_per_year,
)

prices = pricer.price_futures(
    bondnet=model.bondnet,
    bond_features=snap.bonds_metadata.features,
    latent_paths=latent_paths,
    simulated_times=ts,
    target=snap.futures,
)
print("prices:", prices.shape, prices.tolist())  # (n_futures,)
```

## 4.5 What just happened — shape-by-shape

```
ts.shape                = (n_steps,)               # n_steps  ≈ 961
latent_paths.shape      = (n_paths, n_steps, d_z)  # (32, 961, 8)

# year-fractions for each delivery date:
dlv_years.shape         = (n_futures,)             # (9,)

# searchsorted -> grid index per future:
idx.shape               = (n_futures,)             # (9,)

# Gather latent state at delivery:
z_at_dlv.shape          = (n_paths, n_futures, d_z)
                        = (32, 9, 8)

# Broadcast across deliverable bonds (repeat_interleave by basket length):
z_per_dlv.shape         = (n_paths, N_dlv_flat, d_z)
                        = (32, 126, 8)

# Bond features broadcast across paths:
bf_expanded.shape       = (n_paths, N_dlv_flat, d_b)
                        = (32, 126, 8)

# BondNet output:
bond_values.shape       = (n_paths, N_dlv_flat)
                        = (32, 126)

# Divide by CF, then segmented min over basket → per-future CTD per path:
ctd.shape               = (n_paths, n_futures)
                        = (32, 9)

# Mean over paths → final futures prices:
prices.shape            = (n_futures,)
                        = (9,)
```

## 4.6 Sanity check the CTD reduction by hand

```python
# Recompute the intermediate tensor and check segmented_min against torch.min.
dlv_years = to_year_fraction(
    snap.futures.delivery_dates, snap.futures.asof_date,
    business_days_per_year=dl.business_days_per_year,
)
idx = Pricer._extract_latent_idx_at_delivery(ts, dlv_years)
z_at_dlv = latent_paths.index_select(1, idx)
z_per_dlv = z_at_dlv.repeat_interleave(snap.futures.basket_lengths, dim=1)
bf = snap.bonds_metadata.features.unsqueeze(0).expand(n_paths, -1, -1)
bond_values = model.bondnet(z_per_dlv, bf)
cf_adj = bond_values / snap.futures.conversion_factors_flat

manual = []
start = 0
for n in snap.futures.basket_lengths.tolist():
    manual.append(cf_adj[:, start:start + n].min(dim=1).values)
    start += n
manual = torch.stack(manual, dim=1).mean(dim=0)

print(torch.allclose(prices, manual))  # True
```

The same trick is exercised in `tests/test_pricer.py::test_segmented_min_basic`
and `test_price_futures_matches_manual_ctd`.

## What to remember

- All Monte Carlo paths share one simulation grid (`ts`); each future's
  delivery date is independently mapped onto that grid with
  `searchsorted(..., right=True) - 1`.
- The per-future CTD is a *vectorised* segmented `min` over the flattened
  deliverable axis — no Python loop, no per-future BondNet call.
- The conversion-factor division and the mean-over-paths are the only
  pure-finance operations; everything before is differentiable network work.
