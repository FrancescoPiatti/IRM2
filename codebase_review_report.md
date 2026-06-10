# IRM2 — full codebase & math review

**Date:** 2026-06-10
**Scope:** models, training loop, data pipeline, pricer, evaluation.
**Grounding:** the `GridSearch_YCFut_tier1_2` results (4 trials, all eval
losses ~5e-4 to 1.9e-3, stable training, OU trial starting at 0.39 and
collapsing to 2e-3).
**Mandate:** challenge everything. The list below does.

---

## 0. The headline

Your eval losses look tiny (~5e-4) but the fit is poor. That is not a
paradox — it's the tell. **The aggregate loss is dominated by terms that
are trivially satisfied or decoupled from the curve.** Concretely, the
loss is `λ_y·MSE_yield + λ_sr·MSE_sr + λ_f·MSE_fut` where:

- `MSE_sr ≈ 0` **by construction** (the short rate is anchored — §B3),
- `MSE_yield` is small because the front is anchored and the curve is
  near-flat (the model isn't really fitting *shape*),
- `MSE_fut` is whatever the free BondNet head produces — and it is
  **not tied to the yield curve at all** (§B2).

So a small number here tells you almost nothing about whether the model
learned the term structure. **The objective you're minimising and
selecting on is not measuring what you care about.** That is the root of
"tiny loss, poor results", and several independent bugs make it worse.

Severity legend: 🔴 high · 🟠 medium · 🟡 low.

---

## A. Mathematical foundations — are they correctly implemented?

**A1. Risk-neutral ZCB / yield. ✅ mostly correct.**
`price_zcb` computes `P(T) = E[exp(−∫₀ᵀ r ds)]` with a numerically stable
`logsumexp` and a left-Riemann integral; `y(T) = −log P / T`. The
log-sum-exp and the `clamp(1e-12, 1)` are sound. Minor: the left-Riemann
sum and `round()`-to-grid introduce O(dt) bias — fine at dt=1/32.

**A2. The r0 anchor — questionable construction. 🟠**
`decode()` sets `r_s = decoder(z_s) + r0 − decoder(z_0)`. Because all
paths share `z_0`, this is a per-path *parallel shift* pinning `r_0` to
the observed short rate. Two problems:
- The decoder is a **single `nn.Linear`**, so `r_s = r0 + w·(z_s − z_0)`
  — a **scalar projection** of the latent. Despite `latent_dim=64`, the
  observable short rate is effectively **one factor**. Curvature/slope
  flexibility is far lower than the latent dimension suggests.
- You anchor the *instantaneous* rate to the overnight rate (Fed Funds),
  but the shortest yield you fit is the **1-year** (SVENY01). Forcing
  `r_0 = `overnight pins the curve to a point the market curve doesn't
  contain, and systematically biases the front.

**A3. Futures CTD. ✅ correct *form*, ❌ decoupled (see B2).**
`futures = E_paths[min_i BondNet(z_dlv)/CF_i]`, no discounting — that is
the right futures formula (CTD option, daily-settled). But the bond
prices come from a free net, not the model's own rates.

---

## B. Bugs & math/code inconsistencies

**B1. 🔴 Time-unit inconsistency between yields, the grid, and futures.**
- Yield maturities are integer **calendar years** `[1..10]`
  (`yield_store.get_maturities → arange(1, 11)`; SVENY01 = 1 calendar-year
  zero).
- The simulation grid `ts = arange(0, max_maturity, dt)` is therefore in
  **calendar years** (T=10 maps to the last grid point).
- But futures delivery and bond maturity are converted with **`days/252`**
  (`to_year_fraction`, `bond_metadata_store`), which is **calendar-years
  × 365.25/252 = ×1.449**.

So a 3-month future (91 days) is mapped to `0.361` and the pricer samples
the latent at **~4.3 months instead of 3** (`searchsorted` into a
calendar-year grid); a 7-year deliverable bond gets a
`years_to_maturity` feature of **10.15**. The "use 252 everywhere"
convention was applied to *date differences* but **not** to the
yield/grid axis (which is fixed in calendar years), so the two no longer
agree. This corrupts the futures channel.
**Fix:** convert dates with **365.25** (calendar) so date-derived times
match the calendar-year grid and SVENY maturities. (The "252" request is
only self-consistent if the grid and maturities are *also* re-expressed
in 252-units — but they can't be, because SVENY maturities are calendar
years. So 365.25 is the correct divisor here.)

**B2. 🔴 Futures and yields are priced by decoupled mechanisms (no-arbitrage broken).**
The yield curve comes from the SDE short-rate integral. The deliverable
**bond prices come from a free `BondNet(z, features)`** that never
touches the simulated rates. Nothing constrains BondNet's bond prices to
be consistent with the model's own ZCB curve. Consequences:
- The model is **internally arbitrage-inconsistent**: its 7-year yield
  and its 7-year deliverable-bond price are unrelated quantities.
- "Joint YC + futures calibration" is in reality **two heads on a shared
  encoder** (multi-task learning), not one consistent term-structure
  model. Fitting futures gives the yield curve **no information** and
  vice versa — they only share `z`. This is probably the single biggest
  reason the joint setup doesn't help the curve.
**Fix (proper):** price deliverable bonds from the *same* short-rate
paths — `B_i(T_dlv) = E[exp(−∫ r) · cashflows | z_{T_dlv}]` — i.e. keep
simulating past delivery and discount the bond cashflows. **Fix (cheap):**
keep BondNet but add a consistency penalty tying its implied discount
factors to the SDE's `price_zcb` on the same dates.

**B3. 🟠 The short-rate target is trivially zero — a dead loss term.**
`price_short_rate` returns `E[r_0]`, and `r_0` is *anchored* to the
observed short rate, so `MSE_sr ≈ 1e-19` always. With `λ_sr = 1` you are
spending a target on something that is satisfied by construction and
contributes no gradient. Either drop it, or make it a *non-anchored*
prediction target.

**B4. 🔴 Encoder input is never normalised.**
`_preprocess_encoder_input` raises `NotImplementedError` for any
`preprocess_mode` other than `None`, so the LSTM is fed **raw decimal
yields** (~0.0–0.06) with day-to-day differences of ~5 bp = **5e-4**.
That is a minuscule input signal; the encoder struggles to distinguish
curves, which directly weakens `z0` — the *only* day-specific quantity in
the whole model. Output `LayerNorm`/`RMSNorm` does not fix this (it
normalises the embedding, not the input). **Fix:** implement
`preprocess_mode` (per-feature z-score or ÷ a fixed scale like 0.01) on
the encoder input. This is cheap and likely high-impact.

**B5. 🟠 Model selection is on the wrong quantity.**
The grid objective is the mean eval `total_loss` = the same weighted sum
above. Dominated by the futures term and the trivial sr term, it ranks
configs by futures fit, not curve fit, and is uninformative once the
front is anchored. **Fix:** select on a yield-curve RMSE (bp) computed on
a held-out fold, reported per maturity.

**B6. Already-known, already-addressed in later edits (document for completeness):**
- 🔴 Loss scale imbalance (futures absolute MSE ≫ yield MSE) →
  `futures_relative_loss` (implemented).
- 🔴 Diffusion too large → convexity (`diffusion_scale`, implemented).
- 🔴 OU mean reversion erasing `z0` → constant curve
  (`mean_reversion_max`, implemented).
- 🟠 Artificial tanh bounds fighting learning → replaced by scales.

---

## C. Flawed or unjustified assumptions

**C1. 🔴 All day-specific information enters only through `z0`.** The SDE
drift/diffusion/decoder are global; only the initial condition is
day-specific. That is a severe information bottleneck and the reason OU's
mean reversion is so damaging (it erases the one day-specific input).
**Challenge:** feed the encoder embedding into the drift/decoder too
(make the dynamics conditioned on the day), not just the initial state.

**C2. 🟠 Linear decoder ⇒ a 1-factor short rate.** See A2. A 2-layer MLP
decoder would let the 64-dim latent actually express multi-factor curve
shapes. Config-only change.

**C3. 🟠 BondNet as a stand-alone bond pricer.** See B2. The assumption
that a net can replace consistent discounting is what decouples the two
objectives.

**C4. 🟠 `window_step = 4` is a *subsample*, not a slide.** It silently
discards 75% of trading days (same subsample every epoch) and yields only
17 optimiser updates/epoch. Defensible for speed, but you are training on
a quarter of your data. Use `1`–`2` for real runs.

**C5. 🟠 Early stopping on the *training*-loss EMA, no validation set.**
Can stop early / be fooled (we saw it "improve" while every step was
skipped). No held-out model selection at all.

**C6. 🟡 Calendar-year vs business-day mixing.** `nsde.dt = 1/252`,
`trainer.dt = 1/32`, maturities in calendar years, dates in `/252`. The
multiple time conventions are a recurring source of bugs (B1). Pick one
unit (calendar years) and use it everywhere.

---

## D. Why the results are poor — synthesis

1. **The objective doesn't measure curve fit** (B5, B3, B2): tiny loss,
   unconstrained shape.
2. **The encoder can barely see the curve** (B4): raw decimal inputs →
   weak `z0` → the model can't tell days apart → near-constant curves.
3. **Futures don't help the curve and are themselves corrupted** (B2 +
   B1): decoupled head, wrong delivery horizon, inflated maturity
   features.
4. **The dynamics are over-constrained for shape and mis-scaled**
   (A2 linear decoder, C1 z0-only, plus the diffusion/OU issues in B6).
5. **Only a quarter of the data, 17 updates/epoch, no validation** (C4,
   C5).

None of these is "the bug". The model is under-identified for the yield
curve from several directions at once.

---

## E. Prioritised fixes

| # | Fix | Sev | Effort | Why |
|---|---|---|---|---|
| 1 | **Normalise encoder input** (implement `preprocess_mode`, z-score or ÷0.01) | 🔴 | ~15 LOC | `z0` is the only day signal and the encoder can't see it |
| 2 | **Date→year with 365.25**, consistent with the calendar-year grid/maturities (revert the `/252` on `to_year_fraction` & bond maturities) | 🔴 | ~5 LOC | fixes the 1.449× futures horizon + bond-maturity bug |
| 3 | **Tie bonds to the SDE** (discount cashflows on the simulated rates) **or** add a BondNet↔`price_zcb` consistency penalty | 🔴 | medium / large | makes it a real joint no-arbitrage calibration instead of 2 decoupled heads |
| 4 | **Select on a held-out per-maturity yield RMSE**, add a validation fold | 🔴 | ~30 LOC | you are currently ranking configs on the wrong metric |
| 5 | Drop or de-anchor the short-rate target | 🟠 | ~2 LOC | it contributes zero signal |
| 6 | **2-layer MLP decoder** | 🟠 | config | escape the 1-factor straitjacket |
| 7 | Condition drift/decoder on the encoder embedding (esp. for OU) | 🟠 | ~20 LOC | break the z0-only bottleneck (C1), fix OU properly |
| 8 | `window_step = 1`–`2`, validation-based early stopping | 🟠 | config | 4× data, honest stopping |
| 9 | Keep the already-made fixes: `futures_relative_loss`, `diffusion_scale=0.02`, `mean_reversion_max` | — | done | scale + balance |

**If you do only three things:** #1 (normalise the encoder input), #2
(fix the 365.25 vs 252 time bug), and #4 (select/measure on a real
per-maturity yield RMSE). Until #4 you can't even tell whether the other
changes help, because the loss you're watching doesn't track curve
quality.

---

## F. A blunt recommendation on scope

The cleanest path to a model that *demonstrably* fits the yield curve is
to **stop training jointly until the yield-only model works.** Strip to:
encoder (normalised input) → SDE → **MLP decoder** → ZCB/yield loss,
selected on per-maturity bp RMSE on a validation fold. Get that to a few
bp. *Then* add futures — and add them **consistently** (price bonds off
the same rates, B2), not as a second free head. The current joint setup
is optimising a number that is small for the wrong reasons.
