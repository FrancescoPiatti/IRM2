# Why the model doesn't learn the market dynamics — forensics

**Date:** 2026-06-10
**Evidence base:** `GridSearch_YCFut_tier1_2` (4 trials, summaries +
`model_info.json` + `training.log`), plus a line-by-line re-read of the
training pipeline. This report goes *beyond* `codebase_review_report.md`:
it reconstructs what that run actually optimised, and lists **new**
failure mechanisms found on this pass, several of them engineering-level
as you suspected.

---

## 1. Config archaeology — what that run actually was

From `trial_002/model_info.json`, the run used:

```
drift_bound = 5.0, diffusion_bound = 2.0      (the OLD tanh clamps)
init_output_scale = 0.1
λ = (1, 1, 1e-4), futures loss = ABSOLUTE MSE
encoder input = raw decimals (no preprocessing)
dropout = 0.1 inside encoder AND all SDE coefficient nets
window_step = 4 (17 optimizer updates/epoch), lr = 2e-4
trainer.dt not recorded (gap now fixed; was 1/32)
```

**None of the recent fixes (`diffusion_scale=0.02`, relative futures
loss, `mean_reversion_max`, input scaling) were active.** So these
results don't yet test our diagnosis — but they *do* let us check the
old diagnosis against data, and it holds up (next section).

---

## 2. What the numbers say

| trial | nsde | diffusion | epochs | train first→last | eval mean |
|---|---|---|---|---|---|
| 000 | simple | big | 44 | 0.0031 → 0.0018 | 6.6e-4 |
| 001 | **ou** | big | 65 | **0.386** → 0.0024 | 1.9e-3 |
| 002 | ou | small | 53 | 0.0031 → 0.0019 | 4.8e-4 |
| 003 | simple | small | 47 | 0.0031 → 0.0018 | 6.8e-4 |

Reading this with the loss decomposition
`L = MSE_yield + MSE_sr(≈0) + 1e-4·MSE_fut`:

1. **The futures term was ≈ solved at initialisation.** BondNet's
   `output_init_level=100` divided by CF≈0.84 gives futures ≈ 119 vs
   market ≈ 120 — so `1e-4·MSE_fut ≈ 1e-4·(1–3)² ≈ 1e-4`. An accident of
   the init, but it means **the gradient was mostly the yield term** in
   this run.
2. **Therefore the plateau at train ≈ 0.0018 IS the yield MSE**, i.e. a
   **yield RMSE of ≈ √0.0018 ≈ 4.2% across the training period** — the
   model's curves are *grossly* wrong (market yields are 0.5–5%). On the
   calm eval quarter it's √4.8e-4 ≈ **2.2% RMSE** — still ~50× worse
   than a useless flat-at-anchor baseline should even be. The model did
   not learn the curve. Your perception of "poor results" is exactly
   right, and the tiny absolute numbers were masking it.
3. **All four trials plateau at nearly the same value after epoch ~10**
   (the 0.0031→0.0018 path is almost identical for 000/002/003). When
   four different architectures converge to the same mediocre loss, the
   bottleneck is **not** the architecture axis you searched — it's
   something common to all of them (input scaling, diffusion regime,
   decoder, optimisation budget).
4. trial_001 (OU + big diffusion) starting at **0.386** = 100× the
   others: consistent with a big random initial drift through the OU
   `κ(θ−z)` channel before `init_output_scale` tamed... except it
   *didn't* tame it — see finding F1.

---

## 3. NEW findings from this pass

### F1 🔴 `init_output_scale` does nothing for softplus heads — the "calm init" was an illusion
`_shrink_output_layer` scales the last Linear's weights ×0.1 and zeros
its bias → the **pre-activation** starts at ≈0. For the drift (identity
output) that means drift ≈ 0 ✓. But the diffusion and OU mean-reversion
heads end in **softplus**, and `softplus(0) = 0.693`. So the diffusion
started at σ_z ≈ 0.69·tanh-clamped ≈ 0.67 — i.e. **the implied rate vol
was ~39%/yr in this very run**, the exact convexity disaster regime from
`model_diagnosis_report.md` §2. The yield gradient then mostly says
"kill the diffusion", and what's left fits a near-deterministic flat-ish
curve — matching the 4.2% RMSE plateau.
**Status:** fixed as a side effect of `diffusion_scale = 0.02`
(0.693·0.02 ≈ 0.014), but worth knowing the init never did what its
name claims for softplus heads.

### F2 🔴 Dropout *inside the SDE coefficient networks*
All drift/diffusion/θ/κ MLPs had `dropout=0.1`. In an Euler loop this
draws a **fresh mask at every step**, so during training the model
integrates a *different, noisier process* than the deterministic one
used at eval. Three effects: (i) train/eval measure mismatch (the thing
you calibrate is not the thing you evaluate — visible as eval < train in
every trial); (ii) the masks act as extra unmodelled noise on top of the
Brownian term, inflating the effective vol the optimiser sees; (iii) it
adds gradient variance to an already 17-updates/epoch budget.
**Status: fixed** — dropout removed from all SDE nets and the encoder in
both experiment files.

### F3 🔴 Encoder input scale (B4) — now actually fixed
`preprocess_mode` raised `NotImplementedError`; the LSTM saw raw
decimals where the *entire* 2016–2024 yield range spans 0.005–0.05 and
day-to-day moves are ~5e-4. After LayerNorm'd gates this is near-floor
signal; `z0` (the only day-specific input) barely varied across days —
consistent with all trials plateauing at the same shape-blind solution.
**Status: implemented** — `preprocess_mode='scale100'` (percent units;
preserves level/slope/curvature exactly, just rescales). `norm_z` and
`norm_max` also implemented but `norm_z` removes level — documented.

### F4 🟠 `nsde.dt` is silently ignored by `custom_euler` when the trainer supplies `ts`
The trainer always passes its own grid (`trainer.dt`), so the
`base_nsde.dt = 1/252` line in the experiments is dead config — the
solver actually stepped at 1/32. Not a numerical bug (1/32 is fine), but
a documentation trap that cost us archaeology time, compounded by
`trainer.dt` not being recorded in `training_info`.
**Status:** `dt`, `loss_weights`, `futures_relative_loss`, and
`bondnet_consistency_weight` are now recorded in `training_info`.

### F5 🟡 The time feature fed to drift/diffusion is unnormalised
`_pack_tz` appends raw `t ∈ [0, 10]` to an O(1) latent. The first
linear layer can compensate, but it skews initial gradients toward the
t-direction. Low priority; consider `t/horizon`.

### F6 🟡 `busday`/252 convention — implemented per your spec, one caveat
Date deltas are now counted in **business days** (`np.busday_count`)
÷ 252: 91 calendar days → 63 busdays → **0.25 ✓** (your 3-month check).
Caveat: weekday-counting without a holiday calendar gives ~261/yr, so a
1-calendar-year delta maps to 1.036 and 10y → ~10.36 — a ~3.6% stretch
vs the integer SVENY maturities at long horizons. Short-horizon
quantities (futures deliveries, coupon gaps) — the ones that matter for
the futures channel — are now correct. If the 3.6% ever bothers you,
the divisor for *long-dated* quantities can be set to 261, but mixing
divisors is worse than one consistent 252.

### F7 🔴 (Restated from review, now mitigated) Futures were a free head
The decoupling (B2) plus the init accident (futures ≈ solved at epoch 0)
meant the futures channel contributed *nothing* to the curve in this
run — the entire model effectively trained on the broken yield gradient
alone. **Status: mitigated** by the new **LSMC consistency loss**
(`bondnet_consistency_weight`): BondNet is regressed onto the model's
**own pathwise-discounted cashflows** (computed on the same simulated
paths — no nested simulation; index math validated against the
constant-rate closed form). Gradients flow into *both* BondNet and the
SDE, so market futures now genuinely inform the term structure:
market futures → BondNet ← consistency → SDE rates.

---

## 4. Cumulative state — every active fix going into the next run

| Area | Fix | Mechanism |
|---|---|---|
| Encoder | `preprocess_mode='scale100'` | day-signal 100× stronger (F3/B4) |
| Encoder/SDE | dropout removed | deterministic dynamics, train = eval (F2) |
| SDE | `diffusion_scale=0.02` | implied rate vol ~1%/yr; convexity 269%→0.1% |
| SDE | `drift_scale=0.5`, hard bounds OFF | magnitude set without gradient saturation |
| OU | `mean_reversion_max=0.5` | z0 survives the 1–10y horizon (no flat collapse) |
| Loss | `futures_relative_loss=True`, λ=(1,0,1) | yields and futures actually balanced; dead sr term off |
| Coupling | `bondnet_consistency_weight=1.0` (LSMC) | futures channel tied to the SDE curve (F7/B2) |
| Time | busday/252 everywhere | 3-month delivery = 0.25 (B1, your convention) |
| Capacity | `model.decoder` grid axis: linear vs 2-layer MLP | escapes the 1-factor projection |
| Budget | `window_step=2` | 2× data, ~34 updates/epoch |
| Bookkeeping | `dt` + loss config recorded in `training_info` | no more archaeology |

## 5. What to look for in the next run (falsifiable signatures)

1. **Epoch-0 yield component** (`loss_components["yield"]`) should start
   near (2–3%)² ≈ 4–9e-4 and fall to **< 1e-6 (≡ <10 bp RMSE)** within
   ~20 epochs. If it stalls above ~1e-5 (≈30 bp), the bottleneck is
   capacity → check whether the MLP-decoder trials beat the linear ones.
2. **`bond_consistency` component** should fall steadily; if it stays
   O(1e-2), BondNet and the SDE disagree structurally → raise the weight.
3. The four/eight trials should now **separate** (different plateaus).
   If they still cluster, the remaining common bottleneck is the
   optimisation budget (raise epochs / updates) or the single-`z0`
   information bottleneck (next item).
4. If yields fit on train but eval curves look frozen/similar across
   days, the encoder is still under-informative → that is the cue to
   implement the day-conditioned dynamics (encoder embedding into the
   drift), the one structural change deliberately *not* made yet.

## 6. Remaining suspects, ranked (if the next run still fails)

1. **Single-`z0` bottleneck** — all day information through one initial
   condition (review C1). Fix: condition drift/decoder on the embedding.
2. **Selection metric** — still the aggregate loss; add a validation
   fold + per-maturity bp RMSE before drawing conclusions from grids.
3. **Optimisation budget** — 34 updates/epoch is still small; consider
   window_step=1 or accumulate_windows=1 for the final model.
4. **Maturity-grid resolution** — dt=1/32 puts only 32 points/yr; fine
   for ZCB integrals, but check `round()` snapping isn't visible in the
   1y pillar.

## 7. Recommended protocol

**Stage A (cheap, decisive):** run the **YC-only** grid first
(`grid_YC_tier1.py`, now 8 trials with the decoder axis). Success
criterion: best trial < 10 bp RMSE on train, < 30 bp on eval. This
isolates encoder+SDE+decoder from everything futures-related.
**Stage B:** run the joint grid (`grid_YCFut_tier1.py`) and require the
yield component not to degrade vs Stage A while `bond_consistency`
falls. If Stage A fails, no amount of futures machinery matters; fix the
curve first.
