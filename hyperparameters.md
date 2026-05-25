# Hyperparameter reference

Comprehensive reference for **every configurable field** in the project, with:

- a one-line description,
- the **type** and **default**,
- guidance on **how / whether to tune it**,
- whether it makes sense to put it in a **grid search**.

Tuning advice is summarised by tier:

| Tier | Meaning |
|---|---|
| **T1** | First-order driver of accuracy or stability — search early. |
| **T2** | Second-order. Add to the grid once T1 has settled. |
| **T3** | Set-and-forget. Usually one value is enough; don't grid unless you suspect it. |
| **Fixed** | Effectively a project-level choice. Don't grid. |
| **Derived** | Computed from other fields. Don't tune directly. |

Cross-reference: ``gridsearch_hparams.md`` gives a copy-paste-ready
Tier-1+2 grid; this document explains **each individual knob**.

---

## 1. `DataLoaderCfg` — data layer

These describe **what gets loaded**, not how the model is trained. They
are normally fixed per project; never grid them.

| Field | Type | Default | Tier | What it does / how to set it |
|---|---|---|:--:|---|
| `data_path` | str | — | Fixed | Folder containing the CSVs. |
| `start_date` | Optional[Date] | None | Fixed | Inclusive lower bound on the calendar. None ⇒ earliest available. |
| `end_date` | Optional[Date] | None | Fixed | Inclusive upper bound. |
| `max_maturity` | int | 30 | Fixed | Defines the simulation horizon AND the yield-curve column slice (SVENY01 .. SVENYmax). Has to cover all maturities you want to fit + all futures delivery dates. |
| `business_days_per_year` | float | 252.0 | Fixed | Year-fraction convention propagated to bond meta + pricer + futures-horizon filter. Use 365.25 if you switch to calendar-day accounting (don't mix). |
| `enable_yield` | bool | True | Fixed | Yields are the canonical target. Almost always True. |
| `enable_short_rate` | bool | False | Fixed | Whether to add a short-rate target to the loss. Cheap; usually True if you have the data. |
| `enable_bonds` | bool | False | Fixed | Spot bond targets — not currently implemented in the loss path. |
| `enable_futures` | bool | False | Fixed | Turns on the futures pricer + auto-loads bond metadata. Set True for joint YC+futures training. |
| `enable_options` | bool | False | Fixed | Not implemented yet. |
| `device` | str/torch.device | cpu | Fixed | Device for returned tensors. |
| `dtype` | str/torch.dtype | float32 | Fixed | Project policy is float32 (see `project_description.md` §15). |

---

## 2. `EncoderCfg` — encoder topology

The encoder summarises a window of past yield curves into a latent state
``z_t``. Two structural choices and a couple of fine-grained ones.

| Field | Type | Default | Tier | Notes / guidance |
|---|---|---|:--:|---|
| `mode` | "simple" \| "hierarchical" | "simple" | **T2** | "simple": one RNN sees the whole lookback. "hierarchical": fast and slow streams at different sub-sampling frequencies. Hierarchical is more expressive but costs more. |
| `net` | mapping | `{"type": "lstm"}` | **T2** | Only used in simple mode. Replace the whole mapping; deep edits are forbidden by the gridsearch. |
| `fast_net` | mapping | `{"type": "lstm"}` | **T2** | Only used in hierarchical mode. Same rule. |
| `slow_net` | mapping | `{"type": "lstm"}` | **T2** | Only used in hierarchical mode. |
| `combine` | "concat"/"add"/"project" | "concat" | T3 | How to merge fast and slow embeddings. "concat" doubles d, "project" preserves d via a linear layer, "add" needs both streams at full d. |
| `preprocess_mode` | "none"/"norm_z"/"norm_max" | None | T3 | Pre-normalisation. None works fine; the encoder learns to scale. |
| `out_norm` | "layernorm"/"rmsnorm"/None | "layernorm" | T3 | Output norm. LayerNorm is the sane default. |

### 2.1 Backbone-net keys (inside `net` / `fast_net` / `slow_net`)

The mapping passed in must contain at least a ``"type"`` discriminator.
Supported types: ``"rnn"``, ``"lstm"``, ``"gru"``, ``"mamba"`` (MLP is
*not* allowed for the encoder — the encoder must be sequence-aware).

| Sub-field | Type | Typical | Tier | Notes |
|---|---|---|:--:|---|
| `type` | str | "lstm" | **T2** | LSTM and GRU are the workhorses. Mamba is faster on long sequences but newer in this repo. |
| `n_layers` | int | 2 | **T2** | 2–3 is usually enough. |
| `n_units` | int | 64–128 | **T2** | Hidden width. Scale with `latent_dim`. |
| `dropout` | float | 0.0–0.1 | T3 | Between-layer dropout. 0.0 if the run is short. |
| `bidirectional` | bool | True | T3 | True for the encoder — the lookback is fully observed (no causality constraint). |

---

## 3. `NSDECfg` — Neural SDE for the latent state

| Field | Type | Default | Tier | Notes |
|---|---|---|:--:|---|
| `type` | "simple" \| "ou" | "simple" | **T1** | Structural prior. OU adds a mean-reversion term `κ(θ-z)`, biasing the drift to revert to a learned mean. Often easier to train under Q. |
| `noise_type` | "diagonal" \| "general" | "diagonal" | **T2** | Diagonal noise has Brownian dim = state dim. General noise allows a learned `(latent_dim, noise_dim)` diffusion. |
| `solver` | "torchsde" \| "custom_euler" | "torchsde" | Fixed | **Always use `"custom_euler"`** when `method="euler"` — 7–15× faster than torchsde on CPU with identical numerics. Switch back to `"torchsde"` if and only if you change `method` to a higher-order solver. |
| `method` | str | "euler" | Fixed | Only consumed by `solver="torchsde"`. Stick with `"euler"`. |
| `adjoint` | bool | False | Fixed | Adjoint differentiation re-simulates on backward (memory ↓, time ↑). Keep False unless OOM-bound. |
| `rtol` | float | 1e-3 | Fixed | Adaptive-solver tolerance. Ignored by Euler / custom_euler. |
| `atol` | float | 1e-6 | Fixed | Adaptive-solver tolerance. Ignored by Euler / custom_euler. |
| `dt` | float | 1/252 | T3 | Solver step size in years. Smaller = finer discretisation = more compute. Keep ≤ `Trainer.dt`. |
| `drift` | mapping | `{"type": "mlp"}` | **T1** | Drift network spec (`type="simple"` only). |
| `diffusion` | mapping | `{"type": "mlp"}` | **T1** | Diffusion network. **Always use `out_activation: "softplus"`** so σ stays positive. |
| `long_term_mean` | mapping | `{"type": "mlp"}` | **T1** | OU `θ(t, z)` network (`type="ou"` only). Default `out_activation: identity`. |
| `mean_reversion` | mapping | `{"type": "mlp"}` | **T1** | OU `κ(t, z)` network. Use `out_activation: "softplus"` to keep `κ ≥ 0`. |

### 3.1 Network keys (`drift`, `diffusion`, `long_term_mean`, `mean_reversion`)

Each is a mapping with a `"type"` discriminator. Supported: ``"mlp"``,
``"affine"``, ``"constant"`` (NB: ``"rnn"``/``"lstm"``/``"gru"``/``"mamba"``
are **not** valid for NSDE networks — they receive 2D `(B, D)` input).

| Sub-field | Type | Typical | Tier | Notes |
|---|---|---|:--:|---|
| `type` | str | "mlp" | **T2** | "constant" gives a Vasicek-style state-independent diffusion. "affine" is a learned linear map. |
| `n_layers` | int | 2–3 | **T2** | Drift can be deeper than diffusion. |
| `n_units` | int or list[int] | 64–128 | **T2** | List gives per-layer widths. |
| `dropout` | float | 0.0–0.1 | T3 | Mild regularisation. |
| `activation` | str | "gelu" / "silu" | T3 | GELU/SiLU are the safe defaults. |
| `out_activation` | str | varies | **T1** for diffusion | Crucial for the **diffusion**: use `"softplus"` to keep σ ≥ 0. Drift typically uses `"identity"`. `κ` for OU usually uses `"softplus"`. |

---

## 4. `BondNetCfg` — bond pricer at delivery (futures only)

Two variants: `SimpleBondNetCfg` (late-fusion) and `FiLMBondNetCfg`
(feature-modulated trunk). Shared base fields first.

### 4.1 Shared (`BaseBondNetCfg`)

| Field | Type | Default | Tier | Notes |
|---|---|---|:--:|---|
| `latent_dim` | int | — (required) | Derived | Must equal the model's `latent_dim`. |
| `bond_feat_dim` | int | — (required) | Fixed | Must equal `BondMetadataStore` feature width (currently 8). |
| `activation` | str/Module | "SiLU" | T3 | SiLU works well; identity-only blocks underfit. |
| `out_activation` | str/Module | Identity | T3 | Leave as Identity — the `Softplus` for positivity is applied separately via `output_positive`. |
| `dropout` | float | 0.0 | T3 | Bondnet is small; usually 0.0. |
| `output_positive` | bool | False | Fixed | **Always set True.** Dirty bond prices must be positive; otherwise the CTD min in the pricer is meaningless. |

### 4.2 `SimpleBondNetCfg` extras (late-fusion)

| Field | Type | Default | Tier | Notes |
|---|---|---|:--:|---|
| `latent_n_layers` | int | 2 | T3 | Latent branch depth. |
| `latent_n_units` | int or tuple | 128 | **T2** | Scale with `latent_dim`. |
| `bond_n_layers` | int | 2 | T3 | Bond-feature branch depth. |
| `bond_n_units` | int or tuple | 64 | T3 | The bond branch sees only 8 features; doesn't need to be wide. |
| `fusion_n_layers` | int | 2 | T3 | Fusion head depth. |
| `fusion_n_units` | int or tuple | 128 | **T2** | Sets BondNet capacity. |

### 4.3 `FiLMBondNetCfg` extras

| Field | Type | Default | Tier | Notes |
|---|---|---|:--:|---|
| `trunk_n_layers` | int | 2 | T3 | Latent trunk depth (before FiLM modulation). |
| `trunk_n_units` | int or tuple | 128 | **T2** | — |
| `film_n_layers` | int | 2 | T3 | FiLM net producing (γ, β). |
| `film_n_units` | int or tuple | 64 | T3 | — |
| `head_n_layers` | int | 2 | T3 | Pricing head depth. |
| `head_n_units` | int or tuple | 128 | T3 | — |
| `hidden_dim` | int | 128 | **T2** | Width of the modulated representation. |

---

## 5. `TrainerCfg` — training loop

| Field | Type | Default | Tier | Notes |
|---|---|---|:--:|---|
| `n_paths` | int | 500 | T3 | Monte Carlo paths. More = lower variance, more compute. Diminishing returns past ~500. |
| `batch_window` | int | 30 | T3 | Days per window. Bigger = smoother grads, fewer optimiser steps per epoch. |
| `window_step` | int | 2 | Fixed | Sub-sampling stride of consecutive windows. |
| `dt` | Optional[float] | None ⇒ inherits nsde.dt | **T2** | Simulation grid spacing in years. Must be ≥ `nsde.dt`. 1/64 is a common choice. |
| `lookback` | int | 252 | **T2** | Encoder lookback (simple mode). Long lookback captures regimes but slows the window. |
| `lookback_freq` | int | 1 | T3 | Sub-sampling stride inside the lookback. |
| `lookback_fast` | int | 63 | **T2** | Hierarchical encoder — fast stream length. |
| `lookback_fast_freq` | int | 1 | T3 | — |
| `lookback_slow` | int | 252 | **T2** | Hierarchical encoder — slow stream length. |
| `lookback_slow_freq` | int | 5 | T3 | Stride for the slow stream (so it sees a longer effective horizon). |
| `optimizer.name` | str | "adamw" | Fixed | AdamW is the project default. SGD only for ablations. |
| `optimizer.params` | dict | `{"lr": 1e-3, "weight_decay": 1e-4}` | **T1** for `lr`, T3 for the rest | The learning rate is the single biggest knob after capacity. |
| `scheduler.name` | Optional[str] | None | **T2** | "plateau" is the safest. "warmup_cosine" works well for long runs. |
| `scheduler.params` | dict | {} | T3 | Most schedulers run fine with defaults. |
| `loss.name` | str | "mse" | Fixed | MSE for both yields and futures prices. |
| `loss.params` | dict | {} | Fixed | — |
| `use_amp` | bool | False | Fixed | CUDA-only. Forwards run in mixed precision, financial numerics stay in fp32 (see code). |
| `compile_nsde` | bool | False | Fixed | Wraps NSDE sub-nets with `torch.compile` on CUDA. Off by default — only flip when you have a known-good torch/Triton stack. |
| `grad_clip_norm` | Optional[float] | 1.0 | **T2** | Stability knob. 0.5–1.0 is the working range. |
| `accumulate_windows` | int | 1 | T3 | Effective batch size. Combine with `batch_window`. |
| `seed` | Optional[int] | 0 | Fixed | Reproducibility. |
| `deterministic` | bool | False | Fixed | Also pins cuDNN. |
| `results_root` | str | "results" | Fixed | Output root. |
| `run_name` | Optional[str] | None | Fixed | Subfolder name for this run. |
| `debug` | bool | False | Fixed | Verbose logging. |
| `checkpoint.mode` | "min"/"max" | "min" | Fixed | Monitored direction. |
| `checkpoint.save_best_only` | bool | True | Fixed | Keeps the best checkpoint by metric. |
| `checkpoint.every_n_epochs` | int | 30 | T3 | Periodic checkpointing cadence. |
| `checkpoint.max_to_keep` | int | 3 | T3 | Cap on periodic checkpoints. |
| `early_stopping.enabled` | bool | True | Fixed | Stop when training-loss EMA plateaus. |
| `early_stopping.patience` | int | 20 | T3 | Epochs with no improvement before stopping. |
| `early_stopping.min_delta` | float | 1e-4 | T3 | What counts as "improvement". |
| `early_stopping.use_ema` | bool | True | Fixed | Smooth the loss before checking plateau. |
| `early_stopping.ema_alpha` | float | 0.2 | T3 | Higher = more reactive. |
| `skip_nan_loss` | bool | True | Fixed | Skip days that produce NaN/Inf losses. |
| `log_every_n_windows` | int | 1 | Fixed | Logging cadence. |

---

## 6. Model constructor kwargs

These are top-level kwargs for `ShortRateModel(...)`. They appear under
``model.*`` in the gridsearch.

| Kwarg | Type | Default | Tier | Notes |
|---|---|---|:--:|---|
| `name` | str | — | Fixed | Output folder tag. |
| `latent_dim` | int | — (required) | **T1** | Capacity of the latent state. Most important T1 knob. |
| `noise_dim` | Optional[int] | None | **T2** | Only consumed when `nsde.noise_type = "general"`. Brownian dim. |
| `input_dim` | Optional[int] | None | Derived | Inferred from the dataloader. Don't set manually. |

---

## 7. Quick "what to actually grid" cheat sheet

(Same content as ``gridsearch_hparams.md``; quoted here for one-stop
reference.)

### Tier 1 — must search

| Path | Suggested values |
|---|---|
| `model.latent_dim` | 8, 16, 32 |
| `nsde.type` | "simple", "ou" |
| `nsde.diffusion` | two MLP variants with `softplus` |
| `trainer.optimizer.params.lr` | 1e-3, 2e-3, 5e-4 |
| `nsde.solver` | "custom_euler" (single value — locked) |

### Tier 2 — add once Tier 1 is settled

| Path | Suggested values |
|---|---|
| `encoder.net` | LSTM vs GRU, 2 widths |
| `trainer.lookback` | 32, 64, 128 |
| `trainer.dt` | 1/64, 1/128 |
| `trainer.scheduler.name` | "plateau", "warmup_cosine" |
| `trainer.grad_clip_norm` | 0.5, 1.0 |
| `nsde.noise_type` | "diagonal", "general" |
| `model.noise_dim` (general only) | 8, 16 |

### Never grid

`nsde.method`, `nsde.rtol/atol`, `nsde.adjoint`, `bondnet.output_positive`,
the dataloader horizons (`max_maturity`, `business_days_per_year`),
`trainer.use_amp` (CUDA-only switch), `trainer.compile_nsde`.

### Currently exposed but **arguably worth grid-searching once added**

`λ_y`, `λ_f` (per-target loss weights) — not yet exposed in `TrainerCfg`.
See `optimization_plan.md` §6.1.
