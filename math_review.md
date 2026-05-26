# Math + finance + bug review

Focused review of correctness — what the code *says* vs. what the project
mathematics *requires*. Findings are tagged:

- **CRITICAL** — provably wrong; the loss / pricer disagrees with the
  market by an order-of-magnitude factor.
- **HIGH** — silently biases the model in a measurable way.
- **MEDIUM** — incorrect under edge cases or specific configurations.
- **LOW** — robustness / hygiene.

For each item: where it lives, what the symptom is, a *minimal* fix
sketch, and (for the numerical ones) a reproducer.

---

## 1. CRITICAL — percent vs. decimal scale mismatch in the rate pipeline

### Symptom

The CSVs store rates and yields in **percent** (e.g. `DFF=0.06` on
2021-03-31 means a 0.06% Fed-funds rate, and `SVENY01=6.10` on
2000-01-03 means a 6.10% one-year yield). The pricer's discount /
yield formulas, however, assume rates are in **decimal**:

```python
# src/finance/pricer_v2.py
cum_int = torch.cumsum(realisations, dim=1) * dt
integral = cum_int.index_select(1, idx - 1)
P  = exp(-integral)                              # expects integral in decimal years
y_percent = -100.0 * torch.log(P) / maturities   # rescales decimal -> percent
```

Concrete check (`r_t = r0 = 0.06`, flat path, T=1Y, real data 2021-03-31):

```
Market y(0,1Y)             = 0.0747     (percent, i.e. ~0.07% yield)
Model y(0,1Y) (current)    = 6.0000     (off by ~80×)
Model y(0,1Y) (with r/100) = 0.0600     (matches market scale)
```

Equivalently, in 2000 with `r=5.43`:

```
Market y(0,1Y)             ~ 5.43
Model y(0,1Y) (current)    ~ 543        (off by 100×)
```

### Why this matters

The MSE-on-yields loss is dominated by this scale gap, **and** the
`r0` anchor in `decode()` forces every path to start at the market
percent value. The model is squeezed into producing very negative
`ρ(z_t)` for `t>0` to cancel out the percent-scale anchor — that is a
mathematical contortion the architecture isn't designed for.

The futures loss is less affected (BondNet outputs are calibrated to
their own scale by training), but the **discount factor inside
`price_futures` is also broken** by the same factor of 100 because it
goes through the same `cumsum(r) * dt`. *Actually* — looking at
`price_futures` more carefully, the discount factor is not used:
futures pricing reads the latent `z` at delivery and runs BondNet on
it. The discount is implicit via the change of numéraire. So the
**futures path is unaffected** by the percent/decimal bug; only the
**yield-curve loss** is broken.

### Where

- `src/dataloaders/yield_store.py:get_curve` returns the CSV value
  directly.
- `src/dataloaders/short_rate_store.py:get_rate` returns the CSV
  value directly.
- `src/dataloaders/market_loader.py:_full_history` stores the percent
  values.
- `src/finance/pricer_v2.py:price_zcb` + `price_yield_curve` assume
  decimal.
- `src/training/trainer.py:_get_r0` returns the percent value.

### Minimal fix

Divide every rate / yield by 100 at the loader boundary so the rest of
the stack stays in decimal:

1. `YieldCurveStore.from_csv` divides the loaded `df` by 100 after the
   `filter(like="SVENY")` slice.
2. `ShortRateStore.from_csv` divides by 100 after loading.
3. `MarketDataLoader._full_history` is built from the (already
   converted) stores, so no further change needed.
4. Drop the `* 100` in `price_yield_curve`:
   ```python
   y = -torch.log(P) / maturities.float()    # now in decimal
   ```
5. Update the loss to expect decimal yields (no code change needed —
   the targets come from `YieldCurveStore.get_curve`, which is now also
   decimal).
6. Update any test reference values that pinned percent magnitudes
   (currently `tests/test_pricer.py::test_price_yield_curve_constant_rate`
   asserts `100 * r` — that becomes `r`).

This is a one-loader-line + one-pricer-line + one-test-line change.

### Reproducer

The block in the conversation that prints `model y 1Y : 6.0000  vs
market y 1Y : 0.0747` on 2021-03-31 reproduces this in ~15 lines.

---

## 2. HIGH — `business_days_per_year = 252` is inconsistent with the
        yield-curve maturity convention

### Symptom

The simulation grid is built as
``torch.arange(0, max_maturity + dt, dt)``. The yield-curve maturities
are `[1.0, 2.0, …, M]` — these are **calendar years** (SVENY01 is a
1-calendar-year zero, ACT/365 convention).

But `BondMetadataStore.get_bond_features` and
`Pricer.to_year_fraction` both convert calendar-day distances to
"years" using `business_days_per_year = 252.0`:

```python
years_to_maturity = (maturity_ord - asof_ord) / 252.0
```

A calendar-1-year bond is therefore reported as
`years_to_maturity = 365.25 / 252 ≈ 1.45`. The factor is ~1.45×.

### Why this matters

- **BondNet** sees feature `years_to_maturity` that is 1.45× larger
  than the same maturity expressed against the yield curve. The
  network will compensate by learning a 1/1.45 scaling internally —
  but downstream interpretation (e.g. "how does BondNet respond to
  years-to-maturity?") becomes wrong by that factor.
- **Futures delivery indexing**: `to_year_fraction(delivery_dates,
  asof_date)` returns 0.5 *business years* for a 6-calendar-month
  contract. We then read the latent at simulation step
  `round(0.5 × steps_per_year) = round(0.5 × 64) = 32`, which is the
  latent at simulation time `32/64 = 0.5` — but in the yield-curve
  grid this is **half a calendar year**, while the contract delivers
  at **0.5 calendar years from now**. So both sides happen to land at
  the same numerical year-fraction, but **the units the model
  internally associates with "0.5" are inconsistent**: the yield grid
  thinks 0.5 = 0.5 calendar years (180 days), the bond features think
  0.5 = 0.5 business years (~126 calendar days = 4.1 months).

Concretely, the bug is dormant *if and only if* you never compare a
calendar-year-indexed yield maturity to a business-year-indexed bond
feature in the same model. We do this implicitly through the joint
loss.

### Where

- `src/configs/config_loader.py` — `business_days_per_year: float = 252.0`.
- Propagated to: `BondMetadataStore`, `FuturesStore.get_active_tickers`,
  `Pricer.business_days_per_year`, `Pricer.to_year_fraction`.

### Minimal fix

Either:

**(a) Adopt calendar years throughout.** Change the default to `365.25`
and update the docs. Bond features and futures delivery fractions both
become calendar years, matching the yield curve. Recommended.

**(b) Adopt business years throughout.** Convert SVENY maturities
from calendar to business years (`m_cal × 252/365.25`) when computing
`idx` in `price_zcb`. Cleaner inside the simulator but harder to
explain to anyone who knows yield-curve conventions.

(a) is one-character cleaner. The cost is that the user's existing
hyperparameters tuned against the 252-convention will need re-fitting.

---

## 3. HIGH — OU `κ` may be negative in the default config

### Symptom

`OU_NeuralSDE.f` computes the drift as `κ(t, z) · (θ(t, z) − z)`. For
mean reversion we need `κ ≥ 0`. The default mean-reversion network is
created from `_default_mlp()`, which is just `{"type": "mlp"}` and
inherits the MLP's default `out_activation = "Identity"`. So the
network's output is unconstrained — and on initialisation it produces
roughly mean-zero values, half of which are negative.

### Why this matters

If `κ < 0`, the OU drift is **explosive**, not mean-reverting. The
latent state runs away, the decoded short rate blows up, and you get
NaNs or huge losses in the first few epochs.

### Where

- `src/configs/config_nsde.py:_default_mlp` — no `out_activation`
  specification.
- `src/models/nsde.py:OU_NeuralSDE.__init__` — accepts the default.
- The Vasicek / OU experiment scripts I refreshed explicitly set
  `out_activation: "softplus"` on `mean_reversion`. The gridsearch I
  just rewrote also uses `softplus`. The library default is still
  unsafe.

### Minimal fix

In `NSDECfg.validate()` for `type == "ou"`, set:

```python
if self.mean_reversion is None:
    self.mean_reversion = freeze_dict({
        "type": "mlp",
        "out_activation": "softplus",       # enforce kappa >= 0
    })
```

(Same idea for `diffusion`'s default — already covered there because
the diffusion default is softplus in spirit but the `_default_mlp()`
doesn't say so explicitly. The diffusion default *should* also be
softplus.)

---

## 4. HIGH — the `r0` anchor in `decode` couples paths in a way that
        biases the discount factor

### Symptom

```python
# src/models/short_rate_model.py:decode
return out + (r0_t - out[:, 0, :]).unsqueeze(1)
```

This forces `r_path[:, 0, :] = r0` exactly. Mathematically the model is
no longer "neural short rate ρ(z_s) under Q" — it's
``r_s = ρ(z_s) + (r0 − ρ(z_0))``, which adds a **path-dependent**
*constant* (in time) shift. The shift is identical across paths
*because* all paths share `z_0` after `_expand_z0`, so the constant
shift commutes with the expectation:

```
E[exp(-∫r ds)] = exp(-shift · T) · E[exp(-∫ρ(z_s) ds)]
```

### Why this matters

The discount factor is biased by a deterministic factor `exp(-shift · T)`.
The model can learn to absorb this — but the bias interacts badly with
the percent/decimal bug above: when the shift is `0.06` (in percent,
i.e. an actual 0.06% rate), `exp(-0.06 · T)` is essentially 1, and
everything is fine. When the shift is `5.43` (a 5.43% rate read as if
it were 5.43 decimal), the discount factor drops to ~10⁻⁵ regardless
of what the simulated paths do.

### Where

- `src/models/short_rate_model.py:decode`.

### Minimal fix

Two options:

**(a)** Project the SHIFT onto the simulated paths via the latent
state at initialisation (a "controlled NSDE" where `z_0` is chosen so
that `ρ(z_0) ≈ r0`). This is the principled neural-short-rate
formulation but requires re-architecting the encoder/decoder.

**(b)** Keep the post-hoc shift but warn that it's a calibration
hack, and ensure rates are in decimal (fix #1). Then the shift is
small (~0.05) and the bias is negligible.

I recommend **(b)** in the short term — it's a 0-LOC change in this
file and the issue is largely an artefact of #1.

---

## 5. MEDIUM — `_check_valid_start_date` ignores `lookback_freq`

### Symptom

```python
# src/dataloaders/market_loader.py
def _check_valid_start_date(self, date, lookback):
    ...
    lb = min(int(lookback), len(dates) - 1)
    min_valid = pd.Timestamp(dates[lb])
```

For `lookback_freq > 1` the encoder needs ``lookback × lookback_freq``
historical rows, but the check only ensures `lookback`.

### Why this matters

If a user sets `lookback=64, lookback_freq=5` on a calendar that has
only 200 history rows before their requested `start_date`, the check
passes (200 > 64) but `get_history` then fails with a confusing
"history shape mismatch" error somewhere inside `get_histories`.

### Where

- `src/dataloaders/market_loader.py:_check_valid_start_date`.

### Minimal fix

Pass the effective lookback (`lookback * frequency`) into the check
from the trainer, OR expose `lookback_freq` on the loader API and
multiply inside.

---

## 6. MEDIUM — `OptunaGridSearch` writes per-trial warnings when
        switching `nsde.type` mid-grid

### Symptom

When the grid includes `nsde.type: ["simple", "ou"]` and the base
NSDE config has BOTH `drift` and `long_term_mean/mean_reversion`
specified (as in my joint-gridsearch base — necessary so the OU
branch has its own networks), each `validate()` call emits a
`UserWarning` about the unused fields. Cosmetic; floods the log.

### Where

- `src/configs/config_nsde.py:validate`.

### Minimal fix

In `validate()` only warn when the user-provided value is "non-default"
(e.g. not the same object as `_default_mlp()`'s output). Or simply
demote the warning to `logger.debug` since we already document the
behaviour.

---

## 7. MEDIUM — `price_zcb` silently clips maturities past the simulation horizon

### Symptom

```python
idx = idx.clamp(min=1, max=n_steps)
integral = cum_int.index_select(1, idx - 1)
```

If a user requests `maturities = [1, 2, …, 12]` but the simulation
grid only goes out to `max_maturity = 10`, `idx` is clipped to
`n_steps`, the model silently reports `P(0, 12) = P(0, 10)`, and the
yield comparison runs against the wrong target.

### Where

- `src/finance/pricer_v2.py:price_zcb`.

### Minimal fix

Validate at call-time:

```python
if maturities.max().item() * self.steps_per_year > n_steps:
    raise ValueError(
        f"Requested maturity {maturities.max().item():.2f} exceeds the simulation "
        f"horizon {n_steps / self.steps_per_year:.2f} (n_steps={n_steps})."
    )
```

---

## 8. MEDIUM — the gridsearch passes BondNet config separately via an adapter

### Symptom

`OptunaGridSearch` hard-codes the kwargs forwarded to `model_cls`:
``name, encoder, nsde, latent_dim, noise_dim``. To use a BondNet, the
joint-gridsearch script defines a thin wrapper class that injects a
fresh `bondnet_cfg` per trial.

This works but means **you can't grid-search BondNet hyperparameters**
through the normal `bondnet.*` path the way you can for `encoder.*` or
`nsde.*`.

### Where

- `src/training/gridsearch.py:_apply_choice` — only knows about
  `encoder.`, `nsde.`, `trainer.`, `model.` roots.
- `src/training/gridsearch.py` — the `objective` function's
  `model = self.model_cls(...)` call.

### Minimal fix

Extend `OptunaGridSearch.__init__` to accept an optional
`base_bondnet_cfg`, copy it per trial like `enc_cfg`/`nsde_cfg`, route
`bondnet.*` paths to it via `_apply_choice`, and forward it to
`model_cls(..., bondnet=bondnet_cfg)`. ~30 LOC change in one file.

---

## 9. LOW — `Pricer.price_zcb` issues a warning-via-clamp when `idx == 0`

### Symptom

For `maturity = 0` (zero-coupon at the as-of date), `idx = round(0) = 0`,
clamped to 1, `idx - 1 = 0`, `integral[0] = r_0 · dt`. So
`P(0, 0) = exp(-r_0 · dt)` instead of the mathematically correct `1`.
For typical `dt = 1/64` and `r ≈ 0.05`, that's `1 - 8e-4`. Tiny but
"wrong by design".

### Where

- `src/finance/pricer_v2.py:price_zcb`.

### Minimal fix

Add a fast path:

```python
if maturities.min().item() <= 0:
    raise ValueError("maturities must be positive")
```

---

## 10. LOW — the futures-CSV `pivot` will crash on duplicate (date, ticker) rows

### Symptom

```python
# src/dataloaders/futures_store.py
quotes_wide = q.pivot(index="Date", columns="Ticker", values="Price")
```

`pivot` (without `pivot_table`) errors with a not-very-friendly
"Index contains duplicate entries, cannot reshape" if any (Date,
Ticker) pair appears twice.

### Where

- `src/dataloaders/futures_store.py:from_csv`.

### Minimal fix

Drop duplicates (last wins) before the pivot:

```python
q = q.drop_duplicates(subset=["Date", "Ticker"], keep="last")
quotes_wide = q.pivot(index="Date", columns="Ticker", values="Price")
```

---

## 11. LOW — bond `coupon_rate` units are not converted

### Symptom

`BondMetadataStore` reads `coupon_rate` directly from the CSV and
feeds it to BondNet as a feature. The CSV stores coupon rate as
e.g. `2.25` (meaning 2.25% / year). If the rest of the pipeline is
later converted to decimal (per fix #1), this column is still in
"percent units" inside the feature vector. BondNet would learn a
local scaling, but the feature is no longer comparable to e.g. an
analytical bond price function.

### Where

- `src/dataloaders/bond_metadata_store.py:_preprocess` —
  `df["coupon_rate"] = pd.to_numeric(df["coupon_rate"], …)`

### Minimal fix

If you adopt fix #1, also divide `coupon_rate` by 100 here:

```python
df["coupon_rate"] = pd.to_numeric(df["coupon_rate"], errors="raise") / 100.0
```

And update `accrued_interest_per_100` accordingly (currently
`accrued_fraction * (coupon / freq)` — if `coupon` is decimal, this is
fraction-times-decimal so the "_per_100" naming becomes
misleading; rename to `accrued_fraction_x_coupon` or scale back to
percent in the feature output).

---

## 12. LOW — `nsde.solver` defaults to `"torchsde"` rather than
        `"custom_euler"`

### Symptom

Users opting into the modern pipeline get the slow path by default.

### Where

- `src/configs/config_nsde.py` — `solver: SolverType = "torchsde"`.

### Minimal fix

Flip the default to `"custom_euler"`. Already verified equivalent on
diagonal + Euler in `tests/test_extras.py`.

---

## 13. LOW — `Trainer.compute_prices` and `price_snapshot` build a new
        `MarketSnapshot` with `meta={"source": "model_implied"}`, then drop
        the original snapshot's `meta`

### Symptom

User-supplied metadata on a snapshot is silently lost when the pricer
returns its model-implied counterpart.

### Where

- `src/finance/pricer_v2.py:price_snapshot` — `meta={"source":
  "model_implied"}`.

### Minimal fix

Merge: `meta={**snapshot.meta, "source": "model_implied"}`.

---

## 14. LOW — autocast warning under CUDA + custom_euler

The custom Euler loop is float32 throughout. If the user enables
`use_amp = True` and `compile_nsde = False` on CUDA, the autocast
context inside `_forward_one_date` will downcast the latent paths to
bf16/fp16 inside the SDE inner loop. The Brownian increments
(`torch.randn_like(z)`) follow `z`'s dtype, but `_pack_tz` writes the
time column into the cached `t_col` buffer at the buffer's dtype.
Result: under AMP, the first call has dtype=fp32 (cache init), the
second call still uses the cached fp32 buffer but `z` is bf16, so
`torch.cat([z, t_col])` upcasts both to fp32, breaking the autocast
guarantee.

Low severity because AMP defaults to False, but worth a guard:
recreate `_t_col` whenever `z.dtype` changes.

### Where

- `src/models/nsde.py:_pack_tz`.

---

## 15. Sanity checks that all currently pass

These are NOT bugs — included so the next reviewer doesn't waste time
on them:

- **No-look-ahead grid alignment**: `searchsorted(..., right=True) - 1`
  is correctly implemented in `Pricer._extract_latent_idx_at_delivery`.
- **Log-sum-exp in `price_zcb`**: the stability rewrite is correct.
- **CTD min**: `_segmented_min` with `scatter_reduce_(amin)` matches
  a manual `min` per future (verified in
  `tests/test_pricer.py::test_price_futures_matches_manual_ctd`).
- **autograd through `custom_euler`**: gradients flow correctly
  through the time-column cache because `torch.cat` materialises a
  fresh output (verified in `test_custom_euler_returns_correct_shape_and_backprops`).
- **`searchsorted` for delivery indices**: works on torch ≥ 1.6.

---

## Recommended fix order

If you have a day: fix **#1** (percent/decimal). One loader change,
one pricer line, one test value. Everything that depends on rates
becomes correct. This is non-negotiable before you trust any joint
training run.

If you have another day: fix **#2** (year-fraction convention) and
**#3** (OU κ default). These are the next two that change what the
model fits, not just how fast.

After those, the rest are quality-of-life improvements.
