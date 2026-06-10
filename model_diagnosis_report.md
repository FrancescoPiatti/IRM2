# Why the model can't learn the yield curve — the math

**Date:** 2026-06-09
**TL;DR:** The latent diffusion is ~100× too large. The convexity term it
injects into every yield is so big the model is forced to either blow up
(hit the discount-factor clamp) or freeze its diffusion to ~0 — and once
frozen, the futures loss (which dominates the gradient) prevents it from
fitting the curve shape. Two changes fix it: shrink `diffusion_bound`
from `2` to `~0.02`, and put the yield and futures losses on a comparable
scale.

---

## 1. What curve the model actually produces (derivation)

Trace the code:

* `decode()` (`short_rate_model.py:394`) anchors every path:
  `r_{p,s} = decoder(z_{p,s}) + r0 − decoder(z_{p,0})`.
  **All paths share the same initial latent `z0`** (the encoder output,
  broadcast to `n_paths`), so `decoder(z_{p,0}) = D0` is a constant.
  Write `D_{p,s} = decoder(z_{p,s})`. Then

  ```
  r_{p,s} = D_{p,s} + (r0 − D0).
  ```

* `price_zcb()` (`pricer_v2.py:140`) computes `P(T) = E_paths[exp(−∫₀ᵀ r ds)]`,
  then `price_yield_curve` returns `y(T) = −log P(T) / T`.

Substitute and pull the deterministic part out of the expectation:

```
∫₀ᵀ r_{p,s} ds = J_p + T·(r0 − D0),     J_p := ∫₀ᵀ D_{p,s} ds
P(T)           = exp(−T(r0−D0)) · E[exp(−J)]
```

so

```
            ┌ level (parallel) ┐   ┌──── expectations ────┐   ┌──── convexity ────┐
  y(T)  =   (r0 − D0)           +   (1/T) ∫₀ᵀ E[D_s] ds      −   Var(J) / (2T)        + …
```

(the last two come from the cumulant expansion
`−log E[e^{−J}] = E[J] − ½Var(J) + …`).

**This is a perfectly good short-rate model.** The middle term — driven by
the SDE drift and `z0` — can produce level, slope and curvature. The
encoder even sees the *current* curve in its input history, so fitting it
should be easy. The problem is the **third term**.

---

## 2. The convexity term is ~100× too big

`Var(J) = Var(∫₀ᵀ D_s ds)` where `D_s = decoder(z_s)`. The default decoder
is `nn.Linear(64, 1)` with `‖w‖ ≈ 0.58`, and `z_s` diffuses with per-dim
volatility `σ_z` (the diffusion coefficient). So the **implied short-rate
volatility** is

```
σ_r  =  ‖w‖ · σ_z.
```

Now plug in the current config (`grid_YCFut_tier1.py`):
`diffusion_bound = 2`, and the diffusion net starts at `softplus(0) ≈ 0.69`,
so `σ_z ≈ 0.69` (and may grow toward 2):

```
σ_r ≈ 0.58 × 0.69 ≈ 0.40   →   40 % per year.
```

Real short-rate vol is **~1 %/yr**. The model's is **~40× too high.** The
convexity at the 10-year point:

```
Var(J) ≈ σ_r² · T³/3 = 0.40² · 1000/3 ≈ 53
convexity = Var(J)/(2T) = 53/20 ≈ 2.65   →   −265 % off the 10y yield.
```

A −265 % adjustment makes the raw yield hugely negative, so `P` hits its
`clamp(1e-12, 1.0)` ceiling and `y → 0`. **At this diffusion the long-end
yields are pure clamp garbage.**

The only way the optimizer can get sane yields is to drive `σ_z → 0`. And
that is almost certainly what your trained trial did — which is why the
*reported* yield loss is tiny (~1e-5): the model fled to the near-flat,
frozen-diffusion corner. But there it hits the second problem.

---

## 3. With diffusion frozen, the futures loss owns the gradient

Set `σ_z ≈ 0`. The convexity vanishes and

```
y(T) ≈ (r0 − D0) + (1/T) ∫₀ᵀ E[D_s] ds        (the expectations curve)
```

which the drift *could* shape to fit the market. But the joint loss is

```
L = λ_y·MSE_yield + λ_sr·MSE_sr + λ_f·MSE_fut
```

with `MSE_yield ~ 1e-5` (decimal²) and `MSE_fut ~ 10²–10³` (price²). Even
with `λ_f = 1e-4`, the futures term is **10³–10⁵× larger in the gradient**.
So the drift/encoder are shaped by the futures fit, and the curve is just
whatever falls out — roughly flat, mis-sloped. The yield curve is never
actually optimized.

**Net:** the diffusion scale forces the model out of the regime where it
*could* fit the curve, and the loss imbalance ensures that even in the
safe regime it *doesn't*.

---

## 4. What to change (in priority order)

### (1) Shrink `diffusion_bound`: 2 → ~0.02  ← the key fix
Target a realistic `σ_r ≈ 1–1.5 %`. With `‖w‖ ≈ 0.58` that means
`σ_z ≈ 0.02`. Re-check the convexity:

```
σ_r ≈ 0.58 × 0.02 ≈ 1.2 %
Var(J) ≈ 0.012² · 1000/3 ≈ 0.045
convexity ≈ 0.045/20 ≈ 0.0023  =  23 bp   ← correct order for a 10y point
```

Now convexity is a ~20 bp correction, not −265 %, and the model lives in
the regime where the curve = expectations curve. **This is the change that
lets the curve be learnable at all.** (Caveat: the decoder can still grow
`‖w‖` during training and raise `σ_r`; if needed, also lower the decoder
LR or bound the short-rate vol directly — but start here.)

### (2) Tighten `drift_bound`: 5 → 0.5
The drift sets the expectations curve. `5`/yr in the latent is ~290 %/yr
in the rate — absurd, and it lets the futures gradient drag the drift to
nonsense. `0.5` is still far more than any real curve needs.

### (3) Put the two losses on the same scale
So `λ` actually balances them. Cleanest: **relative** futures error,
`MSE((model−market)/market)`, which is O(1) like the yield error. Then
`λ_y = λ_f = 1` and the curve gets real gradient. (~10 LOC in
`Trainer._get_loss`; I can add it behind a flag.) As an interim, you can
instead **warm up yields-only** (`λ_f = 0`) for ~20 epochs, then enable
futures.

### (4) `window_step`: 4 → 2
You asked for this — it doubles the data seen (every 2nd day instead of
every 4th) and the optimizer updates per epoch (17 → ~34). More signal for
the curve.

---

## 5. Sanity checks after the change

Re-run `result_analyzer_v2` (now that the encoder loads correctly) and look at:

* `yields_<date>.csv` — `error_bp` should be small and **flat across
  maturities**, not growing with T. Growing-with-T error = still
  convexity- or shape-starved.
* `short_rate_fan_<date>.png` — the fan should be a **narrow** band
  (~1 %/yr widening), not a 100 %-wide explosion. A wide fan means
  diffusion is still too big.
* `gradient_flow_<date>.png` — encoder/SDE grad norms should be within an
  order of magnitude of BondNet's. If BondNet dwarfs everything, the loss
  is still futures-dominated (do step 3).

The headline: **`diffusion_bound = 2` was a mistake on my part** — I picked
it as a generic "stop blow-ups" cap without calibrating it to the implied
rate volatility. For a short-rate model the diffusion bound *is* the rate
vol, and it has to be ~0.02, not 2.

---

## 6. The OU configuration gives a *constant* curve — a separate, deeper bug

Empirically the **OU** NSDE produced yields that are constant across
maturities (and largely across dates). This is not the diffusion issue
above — it is specific to mean reversion, and it is the more serious flaw.

### 6.1 The math

Linearise the OU drift `f = κ·(θ − z)`. The expected short rate is

```
E[r_s] = r0 + w·(θ − z0)·(1 − e^{−κs}),      A := w·(θ − z0)
y(T)  ≈ r0 + A·g(κT),     g(x) = 1 − (1−e^{−x})/x   (0 → 1, increasing)
```

The curve's entire shape is `g(κT)`. And `g(κT) → 1` for **every** `T>0`
as `κ` grows, so the maturities collapse onto one value:

| κ | y(1y)/A | y(10y)/A | 1y–10y spread |
|--:|--:|--:|--:|
| 0.25 | 0.115 | 0.633 | **0.518** (healthy slope) |
| 5 | 0.80 | 0.98 | 0.179 |
| 20 | 0.95 | 0.995 | **0.045** (flat — the bug) |

`κ = softplus(MLP)` is **unbounded above**, and nothing in the old setup
kept it small, so it drifts into the flat regime.

### 6.2 Why it's structural, not incidental

The day's information enters the model **only through `z0`** (the encoder
output = the SDE initial condition). Mean reversion's whole job is to pull
`z` away from `z0` toward `θ`, **erasing `z0`** over timescale `1/κ`. The
yield integrates `r` over `[0,T]`; once `T ≫ 1/κ` the curve is dominated
by the post-erasure regime (`z ≈ θ`, day-independent), so:

> **OU mean reversion deletes exactly the signal the encoder provides.**
> Long maturities — and, for large `κ`, the whole curve — become
> day-independent and flat.

The Simple NSDE has no mean-reversion term, so `z0` persists and the curve
stays day-specific and sloped. That is precisely why Simple worked and OU
collapsed. OU also has a *manifold* of flat solutions (`κ→∞`, `κ→0`, or
`θ→z`) vs Simple's single `drift→0` point, so it collapses far more
readily when the yield gradient is weak.

### 6.3 The fix (implemented)

`NSDECfg.mean_reversion_max` caps `κ` via `max·tanh(κ/max)`. The grid sets
`mean_reversion_max = 0.5` (with `drift_scale = 0.5` ⇒ effective `κ ≤ 0.25`,
timescale ≥ ~4y), which keeps `z0` alive across the 1–10y curve (spread
0.518 above). This is OU-only; the Simple trials ignore it.

### 6.4 The deeper fix (recommended, not yet implemented)

Capping `κ` is a patch. The clean solution is to **stop routing the day's
information solely through the initial condition**: make `θ` (the
long-term mean) a function of the **encoder embedding**, not just of `z`.
Then mean reversion pulls each day toward its *own* day-specific level
instead of a global attractor, and the model can use mean reversion
*and* fit per-day curves. That's a small architectural change to
`OU_NeuralSDE` (concatenate the encoder embedding into the `long_term_mean`
input) — worth doing if you want OU to be genuinely competitive with
Simple. Until then, expect Simple to win the grid.
