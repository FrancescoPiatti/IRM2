"""
Tier-1 grid search for the **joint YC + futures** calibration.

Sized to fit a realistic compute budget on CPU: 18 trials total.

Grid axes (in order of importance):

1. ``model.latent_dim``               — capacity of the latent state.
2. ``nsde.type``                       — drift family (simple vs OU).
3. ``trainer.optimizer.params.lr``     — base learning rate.

The BondNet width is held fixed at a sensible default; tune it in a
tier-2 sweep once you've identified a baseline latent_dim / NSDE combo.

Run from the repo root:

    python -m experiments_fra.gridsearch_experiments_on_YC_and_futures.grid_YCFut_tier1
"""
import os
from datetime import datetime, timedelta

import torch

from src import MarketDataLoader, ShortRateModel, Trainer, OptunaGridSearch
from src.configs import (
    DataLoaderCfg, EncoderCfg, NSDECfg, TrainerCfg, SimpleBondNetCfg,
)


BOND_FEAT_DIM = 8


class _ShortRateModelWithBondNet:
    """
    Adapter so the gridsearch builds a ShortRateModel with a BondNet attached.

    OptunaGridSearch only forwards `encoder=`, `nsde=`, `latent_dim=`,
    `noise_dim=`, and `name=`. We need `bondnet=` as well, so we wrap the
    model class in a callable that injects a fresh `bondnet_cfg` per trial
    (its `latent_dim` follows the trial's choice).
    """

    def __init__(self, base_bondnet_kwargs: dict):
        self._bondnet_kwargs = dict(base_bondnet_kwargs)

    def __call__(self, *, name, encoder, nsde, latent_dim, noise_dim):
        bondnet_cfg = SimpleBondNetCfg(
            latent_dim=int(latent_dim),
            **self._bondnet_kwargs,
        )
        return ShortRateModel(
            name=name, encoder=encoder, nsde=nsde,
            bondnet=bondnet_cfg, latent_dim=int(latent_dim),
            noise_dim=int(noise_dim),
        )


def main() -> None:
    data_path = "data2"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    start, end = datetime(2021, 10, 1), datetime(2024, 6, 30)
    eval_start = datetime(2024, 7, 1)
    eval_end   = datetime(2024, 9, 30)

    device = torch.device("cpu")

    data_cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=start - timedelta(days=60),
        end_date=eval_end + timedelta(days=30),
        max_maturity=10,
        enable_yield=True,
        enable_short_rate=True,
        enable_futures=True,
        device=device,
    )
    dl = MarketDataLoader(data_cfg)

    # -------------------------------------------------------------------
    # Base configs
    # -------------------------------------------------------------------
    base_enc = EncoderCfg(mode="simple")
    base_enc.out_norm = "layernorm"
    base_enc.net = {
        "type": "lstm",
        "n_layers": 2,
        "n_units": 96,
        "dropout": 0.1,
        "bidirectional": True,
    }

    base_nsde = NSDECfg(type="simple", noise_type="diagonal")
    base_nsde.solver = "custom_euler"
    base_nsde.dt = 1 / 128
    base_nsde.diffusion = {
        "type": "mlp",
        "n_layers": 2, "n_units": [64, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "softplus",
    }
    base_nsde.drift = {
        "type": "mlp",
        "n_layers": 2, "n_units": [128, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "identity",
    }
    base_nsde.long_term_mean = {
        "type": "mlp",
        "n_layers": 2, "n_units": [128, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "identity",
    }
    base_nsde.mean_reversion = {
        "type": "mlp",
        "n_layers": 2, "n_units": [128, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "softplus",
    }

    base_tr = TrainerCfg()
    base_tr.results_root = "results"
    base_tr.run_name = "YCFut_grid_tier1"
    base_tr.n_paths = 128
    base_tr.batch_window = 12
    base_tr.window_step = 2
    base_tr.accumulate_windows = 2
    base_tr.dt = 1 / 64
    base_tr.lookback = 32
    base_tr.lookback_freq = 2
    base_tr.optimizer.name = "adamw"
    base_tr.scheduler.name = "plateau"
    base_tr.grad_clip_norm = 1.0
    base_tr.early_stopping.enabled = True
    base_tr.early_stopping.patience = 10

    # -------------------------------------------------------------------
    # Bondnet kwargs (latent_dim is filled per trial by the adapter)
    # -------------------------------------------------------------------
    base_bondnet_kwargs = dict(
        bond_feat_dim=BOND_FEAT_DIM,
        latent_n_layers=2, latent_n_units=128,
        bond_n_layers=2,   bond_n_units=64,
        fusion_n_layers=2, fusion_n_units=128,
        activation="silu",
        output_positive=True,
    )

    # -------------------------------------------------------------------
    # Grid: 3 (latent) × 2 (nsde.type) × 3 (lr) = 18 trials.
    # -------------------------------------------------------------------
    param_grid = {
        "model.latent_dim": [8, 16, 32],
        "nsde.type": ["simple", "ou"],
        "trainer.optimizer.params.lr": [2e-3, 5e-4, 1e-4],
    }

    search = OptunaGridSearch(
        param_grid=param_grid,
        dataloader=dl,
        base_encoder_cfg=base_enc,
        base_nsde_cfg=base_nsde,
        base_trainer_cfg=base_tr,
        model_cls=_ShortRateModelWithBondNet(base_bondnet_kwargs),
        trainer_cls=Trainer,
        direction="minimize",
        seed=0,
        study_name="YCFut_tier1",
    )

    results = search.run(
        num_epochs=20,
        train_start_date=start,
        train_end_date=end,
        eval_start_date=eval_start,
        eval_end_date=eval_end,
        eval_step=5,
        save_eval=True,
    )

    print("Best:", results.best_params, "value=", results.best_value)
    for t in results.trials:
        print(f"  trial {t.number}  {t.params}  value={t.value:.4f}")


if __name__ == "__main__":
    main()
