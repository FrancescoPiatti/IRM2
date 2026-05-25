# Data flow

This page describes how data moves through the codebase at a single training
date `t`.

## Step 1 — store construction (once)

`MarketDataLoader(cfg)` reads the CSVs in `cfg.data_path`. The yield curve
calendar is the canonical date spine: every other store is assumed to share
those dates.

- `YieldCurveStore` (`yield_curves.csv`) — always built.
- `ShortRateStore` (`short_rate.csv`) — always built.
- `FuturesStore` (`futures.csv`, `futures_expirations.csv`, `futures_dlv.csv`)
  — built only when `enable_futures=True`.
- `BondMetadataStore` (`bond_meta.csv`) — built when `enable_bonds=True` or
  `enable_futures=True`.

## Step 2 — snapshot per date

`MarketDataLoader.get_snapshot(date)` builds a `MarketSnapshot`:

```
MarketSnapshot
├── date                : pd.Timestamp
├── yield_curve         : YieldCurveTarget(date, maturities, yields)
├── short_rate          : ShortRateTarget(date, rate)            or None
├── futures             : BatchedFuturesTarget(...)              or None
└── bonds_metadata      : BondFeatures(...)                      or None
```

### Active-futures filter

A ticker is considered active at `date` iff:

1. It has a non-NaN quote on `date`.
2. Its delivery date is strictly after `date`.
3. Its delivery date is within `max_maturity` years of `date` (year-fraction
   uses `DataLoaderCfg.business_days_per_year`).

If no contracts pass the filter, `futures` is None.

### Bond metadata

When `futures` is non-None, the loader computes one row of bond features per
slot in `futures.deliverable_ids_flat` (no deduplication at this point — see
the optimisation plan). `bonds_metadata.ids` and `bonds_metadata.features`
align with `futures.deliverable_ids_flat`.

## Step 3 — encoder input

`MarketDataLoader.get_history(date, lookback_days, frequency)` builds an
`EncoderInputs` with `curve_history: (T, M)` and `short_rate: (T, 1)`. The
trainer asks for one history per date and one snapshot per date.

## Step 4 — model forward

```
EncoderInputs            ─►  encoder    ─►  z ∈ R^{d_z}
z (initial state)        ─►  NSDE       ─►  z_paths ∈ R^{n_paths × n_steps × d_z}
z_paths                  ─►  decoder    ─►  r_paths ∈ R^{n_paths × n_steps}
z_paths, bond features   ─►  BondNet    ─►  bond values
```

## Step 5 — pricing

The simulation grid spans `[0, max_maturity]` in steps of `Trainer.dt`. The
pricer maps:

- yield maturities → grid indices (exact integer arithmetic).
- futures delivery dates → year-fractions (via
  `business_days_per_year`) → grid indices via
  `searchsorted(..., right=True) - 1` (no look-ahead).

## Step 6 — loss

`Pricer.price_snapshot` produces a model-implied `MarketSnapshot`. The trainer
compares each non-None target against its observed counterpart and sums them
into a single scalar loss.
