# Deep analysis of the latest run — and a live test of the fixes

**Date:** 2026-06-17
**Inputs:** `GridSearch_YCFut_tier1_4/trial_000` (the only available trial) +
its `analysis_v2/improvement_diagnostics`, plus **two live training runs I
executed** with the current code (CPU, small/OOM-safe).

---

## 0. Headline

The droop you keep seeing is **not** a property of the model — it is a
property of the *old, broken* code that produced `tier1_4`. I ran the
**current** code on a small config and the two headline pathologies are
gone:

| metric | `tier1_4` (old code) | **current code (live, 14 ep, CPU)** |
|---|---|---|
| 10y yield error | **−258 bp** | **−23 bp** |
| implied vol σ(1y) | 0.001 % (dead) | **1.02 %** |
| fan 5–95% @10y | 0.004 % (collapsed) | 11 % (alive) |
| long-end shape | monotone droop | mild, normal |

Both a **yields-only** and a **full joint** (futures + consistency) run gave
*identical* good curves — proof the detached consistency no longer corrupts
the SDE.

---

## 1. Why you never actually tested the fixes

`tier1_4/trial_000` was trained with the **old** configuration:
`bondnet_consistency_weight = 1.0` (the **non-detached** version),
**no** volatility anchor, `latent_dim = 64`. And separately, I had
introduced a real ordering bug — `market_price_of_risk` was created using
`self.noise_dim` *before* that attribute was set — which raised
`'..._NeuralSDE' object has no attribute 'noise_dim'` on **every** NSDE
construction. That would have crashed the *new* grid outright. So the
"two trials still don't work" are old-code artifacts; the fixes were never
exercised. The ordering bug is now fixed.

## 2. Root cause of the −258 bp droop (old code)

The diagnostics show `LSMC_consistency = True` and an LSMC residual of
`0.0024`. With the **non-detached** consistency at weight 1.0, that term
dominated the objective and its gradient flowed *into the SDE*, dragging
the model's long-horizon rates down toward BondNet's par-priced
deliverable bonds (coupon ≈ 2–4 %). The yield loss pulled the 10y *up*
toward 4.48 %; the two balanced at **−258 bp**. The training curve confirms
this is an **equilibrium, not under-training**: the loss is flat over the
last 20 epochs (mean |Δ| ≈ 3e-6, slope ≈ −2e-6/epoch). It *converged* — to
the wrong place.

Simultaneously, with **no vol anchor**, σ collapsed to 0.001 %/yr (the data
cannot identify volatility), so the fan is a line and every vol-dependent
output is meaningless.

## 3. The decisive experiment (current code)

I trained the current model (simple SDE, constant diffusion, **vol
anchor** σ\*=1 %, **detached** consistency w=0.25, percent-scaled encoder,
`latent_dim=16`, `n_paths=64`, `dt=1/16`, 14 epochs, lr 5e-4, CPU, ~90 s):

```
 mat  model%  market%  err_bp
   1   5.233   5.046    +18.7
   3   4.893   4.539    +35.4   <- worst (belly)
   7   4.389   4.386    + 0.3
  10   4.250   4.478    -22.8
```

- **10y −23 bp** (was −258), **σ(1y) = 1.02 %** (was 0.001), fan alive.
- The residual is a **mild +35 bp belly at 3–4y** — ordinary
  under-training, not a structural failure. Part of the −23 bp at 10y is
  *correct convexity* (≈17 bp from the now-realistic 1 % vol).
- The **joint run** (futures on, consistency 0.25 detached) produced the
  *identical* curve — the futures channel no longer perturbs the yields.

## 4. On learning rate and epochs (your suspicion)

- **Not the root cause of the droop.** The old run *converged* (flat
  loss); more epochs or a different LR would have reached the same bad
  equilibrium, because the consistency term was fighting the yields.
- **But there is a real training-loop defect: early stopping.** It
  monitors the EMA of the *total* loss, which is dominated by the
  consistency variance floor (~0.0024). Yield-scale progress (~1e-4) is
  invisible to it, so it can stop while the curve is still improving. This
  is the place where "epochs" genuinely bites — fix the *metric*, not just
  the count.
- **lr** = 5e-4 worked cleanly in my test; 2e-4 is also fine. The cosine
  decayed the old run to ~2.6e-5 by epoch 82, which is fine for a
  converged model.

## 5. Concrete fixes, prioritised

1. **Re-run the grid with the current code.** The `noise_dim` fix unblocks
   it; the vol anchor + detached consistency remove the droop and the vol
   collapse. This is the single highest-value action — you have not yet
   seen a run of the fixed model. *(Start with `nsde.type=simple`,
   `pq=0`, the linear-vs-MLP decoder axis.)*
2. **Fix early stopping / model selection.** Monitor a **per-maturity
   yield RMSE (bp) on a held-out date**, not the EMA of the total loss.
   Otherwise the floor-dominated EMA cuts training short of the best curve.
   (Interim: disable early stopping and run the full epoch budget.)
3. **Give the belly more training, then capacity.** The +35 bp at 3–4y
   shrank with epochs in the small test; run the full budget first, and
   only then consider a slightly larger decoder (the decoder grid axis
   already tests linear vs MLP).
4. **Keep:** `latent_dim` 32 (not 64), constant diffusion (vol anchored),
   detached consistency w≈0.25, `futures_relative_loss=True`,
   `rate_vol_weight≈10`, `σ\*≈0.01`.
5. **Optional modelling choice:** σ\*=1 % gives ~17 bp of 10y convexity and
   a wide (~11 %) 10y fan. That is *correct* for a 1 % random-walk-like
   short rate, but if you want a tighter long end, lower σ\* to ~0.006.

## 6. Honest caveats

- My live runs were deliberately **tiny** (latent 16, 64 paths, 2 y of
  data, 14 epochs, CPU). They prove the *mechanisms* work and the droop is
  gone; they do **not** prove the full-scale run reaches a few-bp fit. The
  belly error should shrink with the real budget — verify it does.
- I could not reproduce your exact grid (the GPU machine differs); I ran
  on CPU here. Numbers will move on the full data/epochs, but the
  qualitative result (no droop, live vol) is robust.
