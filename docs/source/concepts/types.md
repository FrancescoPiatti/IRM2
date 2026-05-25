# Type contracts

All inter-module communication uses frozen dataclasses from
`src.types.data_types`. This page is the canonical reference; the
`@dataclass(frozen=True)` source is the implementation.

## EncoderInputs

```python
@dataclass(frozen=True)
class EncoderInputs:
    curve_history: Union[Tensor, Tuple[Tensor, Tensor]]
    short_rate:    Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None
    dates:         Optional[list] = None
```

- `simple` encoder: `curve_history: (T, M)`, `short_rate: (T, 1)`.
- `hierarchical` encoder: each field is a `(fast, slow)` tuple.

## YieldCurveTarget / ShortRateTarget

```python
YieldCurveTarget(date, maturities, yields)   # yields shape (M,)
ShortRateTarget(date, rate)                  # rate shape ()
```

## SingleFutureTarget

```python
SingleFutureTarget(
    id, date, price,
    delivery_date,
    deliverable_ids, conversion_factors,
    metadata,
)
```

Used for one futures contract observed at one date. The price tensor is
scalar; `conversion_factors` is 1D of length `len(deliverable_ids)`.

## BatchedFuturesTarget

```python
BatchedFuturesTarget(
    ids, prices, asof_date,
    delivery_dates,                     # list, length N_futures
    basket_lengths,                     # tensor, shape (N_futures,)
    conversion_factors_flat,            # tensor, shape (sum_i basket_lengths[i],)
    deliverable_ids_flat,               # list, same length as conversion_factors_flat
    metadata,
)
```

A ragged-flattened representation of multiple futures observed on the same
day. Helpers:

- `n_futures`, `total_deliverables` — properties.
- `split_conversion_factors()` — list of per-future CF tensors.
- `split_deliverable_ids()` — list of per-future bond id lists.

Build one from a list of `SingleFutureTarget` via
`src.types.types_utils.merge_single_future_targets(...)`.

## BondFeatures

```python
BondFeatures(ids, features, feature_names, asof_date, metadata)
```

Row `i` of `features` describes bond `ids[i]`. Currently 8 features:

| # | Name | Notes |
|---|------|-------|
| 0 | `years_to_maturity` | from `asof_date` |
| 1 | `years_to_next_coupon` | approximate (no exact coupon schedule) |
| 2 | `years_from_last_coupon` | approximate |
| 3 | `coupon_rate` | annualised |
| 4 | `coupon_frequency` | per year |
| 5 | `remaining_coupon_count` | ceil(ytm * frequency) |
| 6 | `accrued_fraction` | in [0, 1] |
| 7 | `accrued_interest_per_100` | per 100 face |

## MarketSnapshot

```python
MarketSnapshot(date, yield_curve, short_rate, bonds, bonds_metadata, futures, meta)
```

Holes are represented as `None`. The trainer compares each non-None target
against its model-implied counterpart and accumulates the loss.
