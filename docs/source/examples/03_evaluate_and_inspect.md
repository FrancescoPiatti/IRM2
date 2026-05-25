# Example 3 — Evaluation and inspection

Once a model has been trained, three things matter:

1. evaluating it on held-out dates,
2. plotting the loss / metric trajectories,
3. extracting model-implied observables for a deeper look.

## 3.1 Evaluation

```python
from datetime import timedelta

# Single date — the first canonical date strictly after training
single = trainer.evaluate()
print(single.total_loss, single.components)

# A range
range_results = trainer.evaluate(
    start_date=end + timedelta(days=1),
    end_date=end + timedelta(days=30),
    step=1,
)
for r in range_results[:3]:
    print(r.date.date(), r.total_loss, r.components)
```

`EvalResults.components` is a dict like `{'yield': 0.012, 'short_rate': 1e-6,
'futures': 1.2}` — one entry per active loss target. They are saved as a CSV
under `<output_dir>/eval/`.

## 3.2 Model-implied observables (no loss)

`Trainer.compute_prices(date)` returns a `MarketSnapshot` containing
*model-implied* yields and short rate at the given date. Useful for plotting
the model vs. market curve.

```python
import matplotlib.pyplot as plt

date = end + timedelta(days=5)
obs   = dl.get_snapshot(date)
model = trainer.compute_prices(date)

mats = obs.yield_curve.maturities.cpu().numpy()
plt.plot(mats, obs.yield_curve.yields.cpu().numpy(),   "o-", label="market")
plt.plot(mats, model.yield_curve.yields.cpu().numpy(), "x--", label="model")
plt.xlabel("Maturity (years)")
plt.ylabel("Yield (%)")
plt.legend(); plt.show()
```

## 3.3 Loss trajectories

The `ResultAnalyzer` provides built-in plotting:

```python
from src.analysis.result_analyzer import ResultAnalyzer

ra = ResultAnalyzer(run_dir=trainer.output_dir)
print(ra.summary())
ra.plot_epoch_loss()
ra.plot_eval_total_loss()
ra.plot_eval_components()
```

> **Note.** As of the current version, `ArtifactManager.save_losses` writes a
> pickle (`epoch_losses.pkl`) while `ResultAnalyzer.load_epoch_losses` looks
> for `losses.csv` — so the `plot_epoch_loss` step may be a no-op until that
> mismatch is fixed (see `review_notes.md`).

## 3.4 Loading a saved model back

```python
import torch
from src.utils.artifacts import load_model_from_dir

# Recreate the same architecture (must match the trained one!)
model_reloaded = ShortRateModel(
    name="example_full",
    encoder=encoder_cfg,
    nsde=nsde_cfg,
    bondnet=bondnet_cfg,
    latent_dim=16,
)
model_reloaded, epoch, path = load_model_from_dir(
    model_reloaded,
    output_dir=trainer.output_dir,
    device=torch.device("cpu"),
)
print(f"Loaded from {path} (epoch={epoch})")
```

`load_model_from_dir` prefers `checkpoint_best.pt` and falls back to
`model_params.pt`.

## What to remember

- `evaluate()` with no args = single-day evaluation right after training ends.
- `compute_prices()` is the "no loss, just inference" entrypoint.
- All evaluation CSVs land under `<run_dir>/eval/`.
