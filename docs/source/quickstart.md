# Quickstart

The smallest working snippet from raw CSVs to a single training step.

```python
import torch
from datetime import datetime

from src.configs import (
    DataLoaderCfg, EncoderCfg, NSDECfg, SimpleBondNetCfg, TrainerCfg
)
from src.dataloaders import MarketDataLoader
from src.models.short_rate_model import ShortRateModel
from src.training.trainer import Trainer


# 1. Data
data_cfg = DataLoaderCfg(
    data_path="data2",
    start_date=datetime(2021, 1, 1),
    end_date=datetime(2022, 12, 31),
    max_maturity=5,
    enable_yield=True,
    enable_short_rate=True,
    enable_futures=True,
)
dl = MarketDataLoader(data_cfg)


# 2. Inspect what the snapshot looks like
date = dl.calendar.dates[200]
snap = dl.get_snapshot(date)
print(snap.yield_curve.yields.shape)            # (max_maturity,)
print(snap.futures.ids)                          # active futures tickers
print(snap.bonds_metadata.features.shape)        # (N_dlv_flat, 8)


# 3. Build the model
bondnet_cfg = SimpleBondNetCfg(
    latent_dim=8,
    bond_feat_dim=snap.bonds_metadata.features.shape[1],
    latent_n_layers=1, latent_n_units=16,
    bond_n_layers=1, bond_n_units=16,
    fusion_n_layers=1, fusion_n_units=16,
    output_positive=True,
)

model = ShortRateModel(
    name="quickstart",
    encoder=EncoderCfg(mode="simple"),
    nsde=NSDECfg(type="simple"),
    bondnet=bondnet_cfg,
    latent_dim=8,
)


# 4. Trainer
tc = TrainerCfg()
tc.n_paths = 32
tc.batch_window = 4
tc.window_step = 1
tc.lookback = 16
tc.lookback_freq = 1
tc.dt = 1 / 64.0
tc.use_amp = False
tc.early_stopping.enabled = False
tc.run_name = "quickstart"

trainer = Trainer(model=model, dataloader=dl, config=tc, device="cpu")


# 5. One forward + backward pass
loss = trainer._forward_one_date(date)
loss.backward()
print(f"loss = {loss.item():.4f}, grad_fn = {loss.grad_fn.__class__.__name__}")
```

That's the entire pipeline in 50 lines. From here:

- For a full training run, replace step 5 with `trainer.train(num_epochs=10,
  start_date=..., end_date=...)`.
- For evaluation, see :doc:`examples/03_evaluate_and_inspect`.
- For a deep dive into how futures are priced, see
  :doc:`examples/04_futures_pricing_walkthrough`.
