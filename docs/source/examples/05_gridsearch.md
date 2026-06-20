# Example 5 — Optuna grid search

`OptunaGridSearch` runs a fixed grid of configurations against the same
dataloader, sharing one results folder. This page covers two things:

1. A minimal, runnable example.
2. **Which hyperparameters to actually search**, in the order the project's
   sensitivity to them justifies.

## 5.1 A minimal grid

```python
from datetime import datetime
from src import MarketDataLoader, ShortRateModel, Trainer, OptunaGridSearch
from src.configs import DataLoaderCfg, EncoderCfg, NSDECfg, TrainerCfg

dl = MarketDataLoader(DataLoaderCfg(
    data_path="data",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 12, 31),
    max_maturity=10,
    enable_yield=True, enable_short_rate=True, enable_futures=True,
))

trainer_cfg = TrainerCfg()
trainer_cfg.results_root = "results"
trainer_cfg.n_paths       = 256
trainer_cfg.batch_window  = 16
trainer_cfg.window_step   = 1
trainer_cfg.lookback      = 64
trainer_cfg.lookback_freq = 1
trainer_cfg.dt            = 1 / 64
trainer_cfg.early_stopping.enabled = True
trainer_cfg.optimizer.name = "adamw"

search = OptunaGridSearch(
    param_grid={
        # Tier 1 — first-order knobs
        "model.latent_dim": [8, 16, 32],
        "nsde.type": ["simple", "ou"],
        "nsde.diffusion": [
            {"type": "mlp", "n_layers": 2, "n_units": [64, 64], "out_activation": "softplus"},
            {"type": "mlp", "n_layers": 2, "n_units": [32, 32], "out_activation": "softplus"},
        ],
        "trainer.optimizer.params.lr": [2e-3, 5e-4],
        # Lock the fast backend
        "nsde.solver": ["custom_euler"],
    },
    dataloader=dl,
    base_encoder_cfg=EncoderCfg(mode="simple"),
    base_nsde_cfg=NSDECfg(type="simple"),
    base_trainer_cfg=trainer_cfg,
    model_cls=ShortRateModel,
    trainer_cls=Trainer,
    direction="minimize",
    study_name="tier1",
)

results = search.run(
    num_epochs=30,
    train_start_date=datetime(2022, 1, 1),
    train_end_date=datetime(2022, 9, 30),
    eval_start_date=datetime(2022, 10, 3),
    eval_end_date=datetime(2022, 12, 30),
    eval_step=5,
)

print("Best:", results.best_params, "value=", results.best_value)
for trial in results.trials:
    print(trial.number, trial.params, trial.value)
```

Outputs land under
``results/GridSearch_tier1/``:

- ``grid_results.json`` — full results blob.
- ``epochs.csv`` — per-trial / per-epoch training losses.
- ``eval_losses.csv`` — per-trial / per-date evaluation losses.
- ``training.log`` — shared log across every trial.

## 5.2 What to search

The full recipe lives in
[gridsearch_hparams.md](../../../gridsearch_hparams.md) at the repo
root. Headline guidance:

### Tier 1 — must search

| Path | Why | Suggested |
|---|---|---|
| `model.latent_dim` | Capacity of the latent state. | 8, 16, 32 |
| `nsde.type` | Structural prior on the drift. | "simple", "ou" |
| `nsde.diffusion` | Vol parameterisation. Use `softplus` to keep σ > 0. | two MLP variants |
| `trainer.optimizer.params.lr` | Biggest single accuracy lever after capacity. | 1e-3, 2e-3, 5e-4 |
| `nsde.solver` | Lock to ``"custom_euler"`` for ~10× speed. | (single value) |

### Tier 2 — add once Tier 1 settles

| Path | Why | Suggested |
|---|---|---|
| `encoder.net` | LSTM vs GRU at a couple of widths. | two recurrent mappings |
| `trainer.lookback` | History the encoder sees. | 32, 64, 128 |
| `trainer.dt` | Simulation grid spacing. | 1/64, 1/128 |
| `trainer.scheduler.name` | Convergence shape. | "plateau", "warmup_cosine" |
| `trainer.grad_clip_norm` | Stability for OU + general noise. | 0.5, 1.0 |
| `nsde.noise_type` | Diffusion expressiveness. | "diagonal", "general" |
| `model.noise_dim` | Only with `nsde.noise_type="general"`. | 8, 16 |

### Tier 3 — set-and-forget

`encoder.out_norm`, `trainer.optimizer.params.weight_decay`,
`trainer.batch_window`, `trainer.accumulate_windows`,
`trainer.lookback_freq`, `trainer.early_stopping.patience`,
`trainer.n_paths`. Usually one or two values is enough.

### Do **not** grid

- ``nsde.method`` — only consumed by ``solver="torchsde"``.
- ``nsde.rtol``, ``nsde.atol`` — adaptive-solver knobs we never use.
- ``nsde.adjoint`` — re-simulates on backward; only useful when OOM-bound.
- ``bondnet.output_positive`` — must be True for the CTD min to be sane.
- Dataloader horizons (``max_maturity``, futures delivery cap) — these
  are data choices, not hyperparameters.

## 5.3 Anatomy of a `param_grid` key

| Root | Path format | Behaviour |
|---|---|---|
| `model.` | flat: ``model.latent_dim``, ``model.noise_dim`` | Forwarded to ``ShortRateModel(...)`` kwargs. |
| `encoder.` | flat: ``encoder.mode``, ``encoder.net`` | Deep mapping edits **not** allowed — replace the whole mapping. |
| `nsde.` | flat: ``nsde.type``, ``nsde.solver``, ``nsde.diffusion`` | Same rule: replace whole mappings. |
| `bondnet.` | flat: ``bondnet.fusion_n_units``, ``bondnet.hidden_dim`` | Requires `base_bondnet_cfg`. `bondnet.latent_dim` is **set automatically** to match the trial's `model.latent_dim`. |
| `trainer.` | dot-path: ``trainer.optimizer.params.lr``, ``trainer.scheduler.params.step_size`` | Deep mapping edits **are** allowed. |

## 5.4 Joint YC + futures grids

For the joint setting, pass a ``SimpleBondNetCfg`` (or ``FiLMBondNetCfg``)
as ``base_bondnet_cfg``. The gridsearch overwrites
``bondnet.latent_dim`` per trial so it always matches the trial's
``model.latent_dim`` — you do **not** need to grid them together.

```python
from src.configs import SimpleBondNetCfg

base_bondnet = SimpleBondNetCfg(
    latent_dim=16,                       # overwritten per trial
    bond_feat_dim=8,
    latent_n_layers=2, latent_n_units=128,
    bond_n_layers=2,   bond_n_units=64,
    fusion_n_layers=2, fusion_n_units=128,
    activation="silu",
    output_positive=True,
)

search = OptunaGridSearch(
    param_grid={
        "model.latent_dim":     [16, 32],
        "nsde.type":            ["simple", "ou"],
        "bondnet.fusion_n_units": [64, 128],
    },
    dataloader=dl,
    base_encoder_cfg=base_enc,
    base_nsde_cfg=base_nsde,
    base_trainer_cfg=base_tr,
    base_bondnet_cfg=base_bondnet,
    model_cls=ShortRateModel,
    trainer_cls=Trainer,
    direction="minimize",
    study_name="YCFut_tier1",
)
```

A ready-to-run version (10-year window, warmup-cosine, custom_euler)
lives at
``experiments_fra/gridsearch_experiments_on_YC_and_futures/grid_YCFut_tier1.py``.
