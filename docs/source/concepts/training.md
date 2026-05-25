# Training loop

`Trainer.train(num_epochs, start_date, end_date)` is a sequential-window
loop. The trainer iterates over windows of consecutive canonical dates and
applies one backward step per (group of) window(s).

## Window structure

- `TrainerCfg.batch_window`: number of dates per window.
- `TrainerCfg.window_step`: subsampling factor inside the calendar.
- `TrainerCfg.accumulate_windows`: how many windows are accumulated before an
  optimizer step.

For each date in a window, `_forward_one_date` runs the encoder, the NSDE,
the decoder, and the pricer; the resulting day-loss is added to the window
total. NaN/Inf day-losses are skipped per `TrainerCfg.skip_nan_loss`.

### Per-window batching

Two of the per-date forward steps are batched across the whole window:

- **Encoder.** ``_encode_window(batch_dates)`` stacks every date's history
  into a single ``(B, T, M+1)`` tensor and runs the encoder once. The
  hierarchical encoder falls back to per-date encoding.
- **NSDE simulate.** ``_simulate_window(window_latents)`` stacks the
  ``(B, d_z)`` initial latents into ``(B·N, d_z)`` and invokes the NSDE
  forward in a **single** solver call, returning ``(B, N, T, d_z)``. The
  per-date pricer / decoder loop then reads one slice per date.

The simulate batching is solver-agnostic — see the next section.

### NSDE solver backend

`NSDECfg.solver` selects how the latent SDE is integrated:

- ``"torchsde"`` (default) — wraps ``torchsde.sdeint`` (or
  ``sdeint_adjoint`` when ``adjoint=True``). Honours ``method``,
  ``rtol``, ``atol`` and ``dt``. Required if you ever flip ``method`` to
  ``"milstein"`` or any adaptive solver.
- ``"custom_euler"`` — in-house fixed-step Euler-Maruyama loop on
  ``self.f`` and ``self.g``. Same numerical scheme as
  ``method="euler"`` in torchsde, but the Brownian increments come
  directly from ``torch.randn`` so the interval-tree machinery (Lévy
  area, trampoline-based recursion, per-interval reseeding) is skipped.
  Measured **7–15× faster** than torchsde for Euler on CPU.

Switch via the config:

```python
nsde_cfg = NSDECfg(type="simple")
nsde_cfg.solver = "custom_euler"
```

Behaviour is identical at the model API level; only the backend differs.

## Loss composition

`_get_loss` walks the snapshot:

- if `snapshot.yield_curve is not None` → add `MSE(model_yields,
  observed_yields)`;
- if `snapshot.short_rate is not None` → add `MSE(model_r, observed_r)`;
- if `snapshot.futures is not None` → add `MSE(model_futures_prices,
  observed_futures_prices)`.

The pricer also returns a `MarketSnapshot`, so the model-implied prices are
available as `model_snapshot.futures.prices`. Loss weighting (`λ_y`, `λ_f`)
is on the optimisation backlog.

## AMP and precision policy

- Training runs in float32 by default.
- When `TrainerCfg.use_amp=True` and the device is CUDA, the trainer wraps
  forwards in `torch.amp.autocast`. Per the project description, financial
  numerics (CTD min, CF division, discount factors, MC averages) should
  remain in float32 — see :doc:`/index` and the optimisation plan for the
  follow-up.

## Reproducibility

`TrainerCfg.seed` is applied to `torch.manual_seed` and (when available)
`torch.cuda.manual_seed_all`. `TrainerCfg.deterministic=True` additionally
sets `torch.backends.cudnn.deterministic`.

## Checkpoints

`src.utils.artifacts.ArtifactManager` saves:

- best checkpoint (whenever the monitored metric improves) under
  `<output_dir>/checkpoints/checkpoint_best.pt`;
- periodic checkpoints every `every_n_epochs`;
- the final state dict to `<output_dir>/model_params.pt`;
- evaluation CSVs to `<output_dir>/eval/`.

Optuna trials disable IO entirely via `_NullArtifactManager`.

## Evaluation

`Trainer.evaluate(date=..., start_date=..., end_date=..., step=...)` reuses
the same forward path as training, but under `@torch.no_grad()` and
`model.eval()`. By default it evaluates the first date strictly after the
training window.
