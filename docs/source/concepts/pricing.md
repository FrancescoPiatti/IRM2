# Pricing

The pricer (`src.finance.pricer_v2.Pricer`) is stateless except for the
year-fraction convention. It converts simulated paths into model-implied
observables.

## Yield-curve pricing

Given decoded short-rate paths `r ∈ R^{n_paths × n_steps}` on a uniform grid
`dt = 1 / steps_per_year`, the model-implied zero-coupon price is

```
P(0, T) ≈ (1 / n_paths) · sum_i  exp( - sum_{j < idx(T)} r_{i, j} · dt )
```

where `idx(T) = T * steps_per_year` is read off the grid. The implementation
is a `torch.cumsum` followed by `index_select` and `exp(-·).mean()`. The
continuously compounded yield is then `-log(P) / T`.

## Short-rate observable

Defined as the path-wise mean of the first decoded step: a sanity-check signal
rather than a calibration target on its own.

## Futures pricing — cheapest-to-deliver MC

For each active futures contract with delivery date `T_f` and basket `k = 1,
…, K`:

1. Convert `T_f` to a year-fraction via `to_year_fraction`.
2. Map year-fractions onto the simulation grid using
   `searchsorted(..., right=True) - 1` (no look-ahead).
3. Gather latent state at the resulting grid index: `z_at_dlv` of shape
   `(n_paths, n_futures, d_z)`.
4. Broadcast across deliverables via `repeat_interleave(basket_lengths)` to
   shape `(n_paths, N_dlv_flat, d_z)`.
5. Evaluate BondNet on `(z, bond_features)` to get `(n_paths, N_dlv_flat)`
   bond values.
6. Divide by conversion factors (broadcast on the path axis).
7. Take the segmented min over the deliverable axis — one `min` per future —
   using `scatter_reduce_(..., reduce="amin")` to stay vectorised.
8. Average over paths to recover one futures price per contract.

The mathematical statement of step 7-8 is

```
F̂_t  =  E_t^Q  [ min_k  B_k(T_f) / cf_k ]
     ≈  (1 / n_paths) · sum_i min_k  B̂_k(T_f)^{(i)} / cf_k
```

## No-look-ahead alignment

For a delivery date `T_f` that falls between grid points `t_j` and `t_{j+1}`,
the pricer reads the latent state at index `j` — never `j+1`. This is
implemented as a single `searchsorted(..., right=True) - 1`, clamped to ≥ 0.

## Year-fraction convention

`DataLoaderCfg.business_days_per_year` (default `252.0`) propagates to:

- `BondMetadataStore` (feature computation),
- `FuturesStore.get_active_tickers` (horizon filter),
- `Pricer.business_days_per_year` (delivery-date conversion).

Override anywhere by changing the cfg field — the rest of the stack picks it
up automatically.
