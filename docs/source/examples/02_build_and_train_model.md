# Example 2 — Building and training a model

End-to-end: from a `DataLoaderCfg` to a trained model checkpoint. This
example mirrors the structure of `experiments_fra/one_model_experiments_on_YC`
but with a smaller configuration that fits comfortably on CPU.

## 2.1 Imports and config

```python
import os
from datetime import datetime, timedelta
import torch

from src.configs import (
    DataLoaderCfg, EncoderCfg, NSDECfg, SimpleBondNetCfg, TrainerCfg
)
from src.dataloaders import MarketDataLoader
from src.models.short_rate_model import ShortRateModel
from src.training.trainer import Trainer
```

## 2.2 Data + encoder + NSDE

```python
# Data
start = datetime(2021, 10, 1)
end   = datetime(2022, 9, 30)
max_maturity = 10

dl_cfg = DataLoaderCfg(
    data_path="data2",
    start_date=start - timedelta(days=60),     # include encoder lookback
    end_date=end   + timedelta(days=30),
    max_maturity=max_maturity,
    enable_yield=True,
    enable_short_rate=True,
    enable_futures=True,
)
dl = MarketDataLoader(dl_cfg)

# Encoder — simple, LSTM backbone
encoder_cfg = EncoderCfg(mode="simple")
encoder_cfg.net = {
    "type": "lstm",
    "n_layers": 2,
    "n_units": 64,
    "dropout": 0.1,
    "bidirectional": True,
}
encoder_cfg.out_norm = "layernorm"

# NSDE — Simple Neural SDE under risk-neutral Q
nsde_cfg = NSDECfg(type="simple", noise_type="diagonal")
nsde_cfg.solver = "custom_euler"           # 7–15× faster than torchsde on CPU
nsde_cfg.dt = 1 / 128
nsde_cfg.drift = {
    "type": "mlp",
    "n_layers": 2,
    "n_units": [64, 64],
    "activation": "gelu",
}
nsde_cfg.diffusion = {
    "type": "mlp",
    "n_layers": 2,
    "n_units": [32, 32],
    "activation": "gelu",
    "out_activation": "softplus",          # ensure positive volatility
}
```

> **Tip.** ``nsde.solver = "custom_euler"`` is the recommended default. It
> implements the same Euler-Maruyama discretisation that
> ``torchsde.sdeint(method="euler")`` would, but skips the interval-tree
> Brownian-bridge machinery that's only needed by Lévy-area solvers
> (Milstein, SRK). Use ``"torchsde"`` if you ever switch ``nsde.method``
> away from ``"euler"``.

## 2.3 BondNet

The `bond_feat_dim` must match the data layer's feature width (8 in this
project).

```python
bondnet_cfg = SimpleBondNetCfg(
    latent_dim=16,
    bond_feat_dim=8,
    latent_n_layers=2, latent_n_units=64,
    bond_n_layers=2, bond_n_units=32,
    fusion_n_layers=2, fusion_n_units=64,
    activation="silu",
    output_positive=True,                  # bond prices >= 0
)
```

## 2.4 Model

```python
model = ShortRateModel(
    name="example_full",
    encoder=encoder_cfg,
    nsde=nsde_cfg,
    bondnet=bondnet_cfg,
    latent_dim=16,
)
```

## 2.5 Trainer

```python
trainer_cfg = TrainerCfg()
trainer_cfg.run_name = "example_full"
trainer_cfg.n_paths = 64
trainer_cfg.batch_window = 8
trainer_cfg.window_step = 2
trainer_cfg.accumulate_windows = 2
trainer_cfg.dt = 1 / 64                    # simulation grid step (years)
trainer_cfg.lookback = 32
trainer_cfg.lookback_freq = 2
trainer_cfg.optimizer.name = "adamw"
trainer_cfg.optimizer.params = {"lr": 2e-3, "weight_decay": 2e-4}
trainer_cfg.scheduler.name = "plateau"
trainer_cfg.use_amp = False
trainer_cfg.grad_clip_norm = 1.0
trainer_cfg.early_stopping.enabled = True
trainer_cfg.early_stopping.patience = 10
trainer_cfg.checkpoint.save_best_only = True
trainer_cfg.log_every_n_windows = 5

trainer = Trainer(model=model, dataloader=dl, config=trainer_cfg, device="cpu")

trainer.train(num_epochs=5, start_date=start, end_date=end)
```

## 2.6 Outputs

After `train()` returns, the run directory holds:

```
results/<TIMESTAMP>_example_full/
├── model_info.json               # human-readable run metadata
├── model_params.pt               # final model state_dict
├── epoch_losses.pkl              # per-epoch averages
├── training.log                  # logger output
└── checkpoints/
    ├── checkpoint_best.pt
    └── checkpoint_index.json
```

## What to remember

- `bondnet_cfg.bond_feat_dim` must equal the feature width emitted by
  `BondMetadataStore` (currently 8).
- `Trainer.dt` is the *simulation grid step* in years (e.g. 1/64). It does
  not have to equal `nsde.dt` (the solver step). Use `nsde.dt` smaller than or
  equal to `trainer.dt`.
- `output_positive=True` on BondNet is essential; otherwise BondNet can return
  negative bond prices and the CTD min becomes meaningless.
- Use `accumulate_windows > 1` if you want larger effective batch sizes on
  GPU.
