# GridSearch_YCFut_tier1_3 — postmortem

**Date:** 2026-06-12
**Evidence:** the four trial folders (summaries, model_info, checkpoints)
plus the `analysis_v2` diagnostics you generated (per-maturity yields,
gradients, futures tables), plus direct inspection of the trained
weights (manual forward through the checkpointed diffusion/decoder nets).

---

## 0. Verdict

**Root cause found, with quantitative evidence: the LSMC consistency
term I added last round hijacked the objective.** It made up **~91% of
the loss**, its gradient flowed *into the SDE* (bidirectional by my
design choice), and — with BondNet pinned near par by its init and the
market-futures pull — it **bent the yield curve to meet BondNet** instead
of the other way around. That is the −250…−300 bp long-end droop you
see in every trial.

This was my mistake, in two layers: (1) letting the regression target's
gradient flow back through the pathwise discounting into the SDE, and
(2) weighting the term at 1.0 when its irreducible pathwise-variance
floor alone is ~20× the yield MSE. Both are now fixed (§5).

The genuinely good news: **everything else now works.** This is visible
in the same tables — see §3.

---

## 1. What the data says

### 1.1 The yield curves — identical failure shape in all four trials

`yields_2024-07-01.csv`, model vs market (market ≈ flat 4.4–5.0%):

| maturity | t000 err (bp) | t001 | t002 | t003 |
|--:|--:|--:|--:|--:|
| 1y | +14 | +13 | +31 | +13 |
| 2y | +25 | +24 | +45 | +26 |
| 4y | −4 | −3 | +6 | 0 |
| 6y | −73 | −72 | −78 | −78 |
| 8y | −162 | −173 | −176 | −180 |
| 10y | **−256** | **−297** | **−273** | **−288** |

Four different architectures (simple/OU × linear/MLP decoder × small/big
diffusion), one identical signature: **front fits, long end droops.**
When every architecture fails the same way, the cause is a term in the
objective, not the architecture.

### 1.2 The loss decomposition — the objective wasn't about yields

At eval (trial_000, total = 3.20e-3):

| component | value | share |
|---|--:|--:|
| yield MSE | 1.57e-4 | **4.9%** |
| futures (relative) | ~1.3e-4 | ~4% |
| **bond_consistency** | **~2.9e-3** | **~91%** |

The optimizer was, to first order, minimising the consistency term. The
yield term — the thing you care about — was a 5% afterthought.

### 1.3 The droop is drift, not convexity (measured from the weights)

I rebuilt the trained diffusion and decoder from the checkpoints and
forward-passed them manually:

| | σ_z /dim | decoder gain | **σ_r** | implied 10y convexity |
|---|--:|--:|--:|--:|
| trial_000 (MLP dec) | 0.0101 | 0.083 | **0.08 %/yr** | ~0 bp |
| trial_001 (lin dec) | 0.0103 | 0.372 | **0.38 %/yr** | ~2 bp |

The `diffusion_scale=0.02` fix did its job — rate vol is sane and the
convexity term is negligible. The droop (model yield slope settling at
≈ −44 bp/yr beyond 5y) is therefore a **learned negative risk-neutral
drift**: the model expects the short rate to decline ~0.9%/yr for a
decade. The market curve says no such thing.

### 1.4 BondNet never really trained — and that's the key

The fusion-head output bias after 70–100 epochs: **100.01–100.03**
(init: 100.0). BondNet stayed ≈ a constant par-pricer; its futures fit
(0.4–2.4% errors) is still mostly the `output_init_level=100` ÷
conversion-factor accident, lightly tuned. Why it matters: the
consistency term compares BondNet (≈ par, ~100) against the model's
pathwise-discounted bond PVs. With deliverable coupons around 2–4%, a
bond prices at par only if **rates ≈ coupon**. So a term worth 91% of
the objective was demanding: *make the model's long-horizon rates
≈ 2–3%*. The yield loss (5%) demanded 4.4%. The equilibrium of that
20:1 tug-of-war is exactly the curve you got: pinned at the front (r0
anchor + encoder), sagging to ~1.5–1.9% at 10y.

### 1.5 Mechanism summary

```
market futures ──(rel. loss)──> BondNet ≈ par
                                   ▲
                                   │  consistency (weight 1.0,
                                   │  gradient INTO the SDE — bug)
                                   ▼
              model pathwise PVs  ←──  SDE long rates dragged to ~coupon
                                              │
                                              ▼
                              5–10y yields droop −250…−300 bp
```

---

## 2. Why eval/early-stopping never caught it

The consistency value also carries an **irreducible pathwise-variance
floor** (a single-path discount factor is a noisy estimate of the
conditional bond price; even a perfect BondNet leaves `Var(PV|z_T)` in
the MSE). That floor sat in both the training total and the eval
objective, so: (a) the EMA early-stopper watched a number dominated by
a near-constant ~2–3e-3 and stopped on its plateau, blind to
yield-scale (1e-4) progress; (b) the grid ranked trials mostly on the
floor, not on curve quality.

---

## 3. What is now verified WORKING (do not re-fix)

- **Encoder + `scale100`:** the 1–4y pillars fit at ±25 bp — the model
  *does* distinguish days now. (And I verified `_encode_window` routes
  through the preprocessing — no train/eval input mismatch.)
- **Diffusion scale:** σ_r ≈ 0.1–0.4%/yr, convexity gone (was the 269%
  disaster two rounds ago).
- **Stability:** 70–100 epochs, no skipped steps, all 59 gradient
  tensors finite. OU no longer collapses to a constant (trial_002's
  curve has the same shape as the others — the `mean_reversion_max` cap
  works; its start at 3.2 was just a rough init epoch).
- **Futures channel:** model futures within 0.4–2.4% of market.
- **Time conventions, CF cleanup, relative futures loss:** all behaving
  as designed in the artifacts.

---

## 4. Fixes implemented (this round)

1. **`pv = pv.detach()` in `_lsmc_consistency_loss`** — the regression
   target no longer back-propagates through the discounting into the
   SDE. Gradient reaches **BondNet only**: BondNet learns the
   model-consistent conditional bond price; any genuine futures/curve
   basis becomes a *reported residual*, never a corrupted curve.
2. **Consistency excluded from the eval objective** (`_get_loss` adds it
   only when `model.training`; always logged as a component). Eval
   totals and grid selection are now yield + futures only.
3. **Weight 1.0 → 0.25** in the experiment, so the variance floor in
   the *training* total no longer drowns yield-scale progress for the
   EMA early-stopper.

With these, the effective curve-shaping gradient becomes ~50/50
yield : futures-relative (1.6e-4 vs 1.3e-4) — balanced, as intended.

---

## 5. What to expect on the rerun (falsifiable)

1. **The droop disappears or shrinks drastically.** Nothing now rewards
   low long-end rates; the yield gradient (−256 bp at 10y is a *huge*
   signal at 50% weight) pushes the drift up. If a droop persists at
   >50 bp, the remaining suspect is drift capacity (`drift_scale=0.5`)
   or the single-`z0` bottleneck — escalate in that order.
2. **Eval totals drop ~10×** (the 2.9e-3 floor leaves the objective);
   expect eval ≈ 2–4e-4 immediately, then lower as the long end fixes.
3. **BondNet finally moves** (fusion bias drifts away from 100.0x as the
   detached regression trains it toward model PVs ≈ 88–95 for low-coupon
   deliverables). Watch the `bond_consistency` component fall.
4. **Trials should separate** — with the dominating common term gone,
   the architecture axes (simple/OU, decoder) can finally show
   differences.

## Addendum (same day): the fan charts and the vol-identification hole

Inspecting `short_rate_fan_2024-07-01.png` (trials 000/001) shows two
pathologies, one already covered above and one **deeper**:

1. The median path plunges from 5.3% to −2.3% / −5.0% by year 10 — the
   consistency-induced drift droop seen from the rate side (the path
   average reproduces the 10y yield exactly: trial_000 avg ≈ 1.9% =
   its 10y yield). Fix already in (detach).
2. **The 5–95% band is invisible** — all 512 paths collapsed onto one
   line (σ_r trained to 0.1–0.4 %/yr). Beyond the (now removed)
   consistency variance penalty, the deeper truth is that **nothing in
   the calibration data identifies the volatility**: yields are
   vol-insensitive up to bp-level convexity and futures only weakly
   vol-sensitive via CTD optionality. Left free, σ is whatever the
   optimizer drifts to, and every vol-dependent output (fan charts, MC
   dispersion, option-style quantities) is meaningless.

**Fix (implemented): a short-rate vol anchor.** By Girsanov, the
diffusion coefficient is the *same* under P and Q — only the drift
changes measure. So pinning the model's 1y cross-path std to the
historically measured short-rate vol (~1 %/yr for USD) is principled,
not a fudge: `TrainerCfg.rate_vol_target = 0.01`,
`rate_vol_weight = 10.0` (train-only in the objective, logged as the
`rate_vol` component at eval). Additionally, the grid's diffusion axis
was changed from "small vs big MLP" (meaningless — data can't rank vol
capacity when it can't even pin vol magnitude) to **constant vol vs
state-dependent MLP vol**, the question that actually matters
(Vasicek/Hull-White vs local-vol style).

On the noise-type question: `noise_type='diagonal'` is the right
choice and is NOT the problem — `'general'` would add a 64×64
unidentified diffusion matrix on top of an already unidentified scalar
magnitude. The legitimate open hyperparameter is `latent_dim=64`
(unjustified; 16–32 would concentrate signal), left unchanged this
round to keep the rerun attributable.

## 6. Standing recommendations (unchanged, now unblocked)

- Selection on **per-maturity yield RMSE (bp) on a held-out fold** is
  still the right end-state for the grid objective.
- If after this rerun the long end is good on train but unstable across
  eval days, the next structural lever is conditioning the drift on the
  encoder embedding (the single-`z0` bottleneck, review C1).
- The futures/curve **basis** (consistency residual) is now an
  *observable* — worth a plot in the paper: it measures how far market
  futures sit from the model's own no-arbitrage forward bond prices.
