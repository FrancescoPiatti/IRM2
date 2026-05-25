# Overview

IRM2 is built around four layers, each with a single responsibility. New
colleagues should be able to follow this diagram top-to-bottom and understand
exactly where each piece of code lives.

## High-level pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  data2/                                                              │
│   ├── yield_curves.csv         ◄── canonical calendar (yield dates)  │
│   ├── short_rate.csv                                                 │
│   ├── futures.csv                                                    │
│   ├── futures_expirations.csv                                        │
│   ├── futures_dlv.csv          ◄── delivery basket + CF              │
│   └── bond_meta.csv            ◄── coupon, maturity, etc.            │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                ┌─────────────────▼─────────────────┐
                │  src.dataloaders                  │
                │  YieldCurveStore / ShortRateStore │
                │  FuturesStore / BondMetadataStore │
                │  MarketDataLoader  ──►  MarketSnapshot
                └─────────────────┬─────────────────┘
                                  │
                ┌─────────────────▼─────────────────┐
                │  src.models                       │
                │  Encoder → NSDE → Decoder         │
                │              ↑                    │
                │        BondNet                    │
                └─────────────────┬─────────────────┘
                                  │
                ┌─────────────────▼─────────────────┐
                │  src.finance.Pricer               │
                │  price_yield_curve()              │
                │  price_short_rate()               │
                │  price_futures()  (CTD MC)        │
                └─────────────────┬─────────────────┘
                                  │
                ┌─────────────────▼─────────────────┐
                │  src.training.Trainer             │
                │  Sequential-window training       │
                │  λ_y·L_yield + λ_f·L_fut          │
                └───────────────────────────────────┘
```

## Layer responsibilities

| Layer | Module | Owns |
|-------|--------|------|
| Configs | `src.configs.*` | Dataclasses for every component, with `validate()`. |
| Data | `src.dataloaders.*` | CSV loading, calendar, snapshot construction. |
| Types | `src.types.*` | Frozen dataclasses that flow between data → model → pricer. |
| Models | `src.models.*` | Encoder, Neural SDE, Decoder, BondNet. |
| Pricer | `src.finance.pricer_v2.Pricer` | Convert simulated paths into model-implied yields / futures. |
| Training | `src.training.*` | Windowed loop, AMP, checkpointing, evaluation, optuna. |
| Utils | `src.utils.*` | Logger, artifact manager, validators. |
| Analysis | `src.analysis.*` | Post-run plotting. |

## Where to put things

- A new instrument? Add a `Target` to `src.types.data_types`, a store to
  `src.dataloaders`, plumb it through `MarketDataLoader.get_snapshot`, and
  add a branch to `Pricer.price_snapshot`.
- A new neural backbone? Add it to `src.nn`, register it in
  `src.nn.generator.SUPPORTED_NETWORK_TYPES`, and (optionally) ship a
  `DEFAULT_CONFIG_<NAME>` in `src.configs.config_nn`.
- A new Neural SDE variant? Add the config in `src.configs.config_nsde`, the
  module in `src.models.nsde` inheriting from `BaseNSDE`, and a branch in
  `create_nsde_from_config`.
