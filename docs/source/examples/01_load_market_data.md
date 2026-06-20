# Example 1 — Loading and inspecting market data

This example walks through the data layer in isolation. The goal is to get a
feel for what the snapshots look like before any model is involved.

## 1.1 The simplest loader (yields only)

```python
from datetime import datetime
from src.configs import DataLoaderCfg
from src.dataloaders import MarketDataLoader

cfg = DataLoaderCfg(
    data_path="data",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 12, 31),
    max_maturity=10,
    enable_yield=True,
)
dl = MarketDataLoader(cfg)

print(f"calendar: {dl.calendar.start_date.date()} ... {dl.calendar.end_date.date()}")
print(f"          {len(dl.calendar.dates)} dates")
```

The canonical calendar is taken from the yield curve CSV. Yields are the only
target that must be present on every date.

## 1.2 Inspect a single snapshot

```python
date = dl.calendar.dates[100]
snap = dl.get_snapshot(date)

print(snap.yield_curve.date.date(), snap.yield_curve.maturities.tolist())
print(snap.yield_curve.yields.tolist())
```

Output (illustrative):

```
2021-05-27 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
[0.13, 0.30, 0.56, 0.82, 1.05, 1.25, 1.42, 1.58, 1.71, 1.83]
```

`short_rate`, `futures`, and `bonds_metadata` are all `None` because we didn't
enable them.

## 1.3 Enable futures and bond metadata

```python
cfg = DataLoaderCfg(
    data_path="data",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 12, 31),
    max_maturity=15,
    enable_yield=True,
    enable_short_rate=True,
    enable_futures=True,
)
dl = MarketDataLoader(cfg)

snap = dl.get_snapshot(dl.calendar.dates[200])
print(snap.futures.ids)
print(snap.futures.delivery_dates)
print(snap.futures.basket_lengths.tolist())
print(snap.bonds_metadata.features.shape)
```

Output:

```
['FVH2022', 'FVM2022', 'FVZ2021', 'TUH2022', 'TUM2022', 'TUZ2021', 'TYH2022', 'TYM2022', 'TYZ2021']
[Timestamp('2022-04-05 00:00:00'), Timestamp('2022-07-06 00:00:00'), ...]
[10, 11, 11, 12, 11, 12, 20, 20, 19]
torch.Size([126, 8])
```

Each row of `bonds_metadata.features` corresponds to one *slot* in
`futures.deliverable_ids_flat`. The same bond can appear in multiple slots
(across futures with overlapping baskets) — this is the "per-slot" design
decision documented in `future_pricing_plan.md`.

## 1.4 Walking the calendar

`MarketCalendar.training_windows` is what the trainer iterates over.

```python
for w in dl.get_batch_windows(window_days=5, step=1):
    print([t.date() for t in w])
    break
```

Output:

```
[datetime.date(2021, 1, 4), datetime.date(2021, 1, 5),
 datetime.date(2021, 1, 6), datetime.date(2021, 1, 7),
 datetime.date(2021, 1, 8)]
```

## 1.5 Encoder histories

```python
hist = dl.get_history(
    dl.calendar.dates[100],
    lookback_days=20, frequency=1,
)
print(hist.curve_history.shape)   # (20, max_maturity)
print(hist.short_rate.shape)      # (20, 1)
```

Pass `return_short_rate=False` to drop the short-rate branch (e.g. for an
encoder that does not consume it — currently the simple encoder always does).

## What to remember

- The yield-curve calendar is canonical.
- A "snapshot" is whatever set of targets you enabled on a given date.
- `bonds_metadata.features` is aligned slot-by-slot with
  `futures.deliverable_ids_flat`. The pricer relies on this alignment.
