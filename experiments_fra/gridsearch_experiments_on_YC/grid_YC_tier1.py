"""
Tier-1 grid search for the **yield-curve-only** calibration.

Searches the first-order knobs (capacity, NSDE family, learning rate)
recommended in ``gridsearch_hparams.md``. Sized for ~12 trials so the run
finishes in human-scale time on CPU.

Run from the repo root:

    python -m experiments_fra.gridsearch_experiments_on_YC.grid_YC_tier1
"""
import os
from datetime import datetime, timedelta

import torch

from src import MarketDataLoader, ShortRateModel, Trainer, OptunaGridSearch
from src.configs import DataLoaderCfg, EncoderCfg, NSDECfg, TrainerCfg


def main() -> None:
    # -------------------------------------------------------------------
    # Shared dataloader
    # -------------------------------------------------------------------
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
        device=device,
    )
    dl = MarketDataLoader(data_cfg)

    # -------------------------------------------------------------------
    # Base configs (per-trial copies)
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

    base_tr = TrainerCfg()
    base_tr.results_root = "results"
    base_tr.run_name = "YC_grid_tier1"
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
    # Tier-1 grid — first-order knobs.  Total = 2 * 2 * 3 = 12 trials.
    # -------------------------------------------------------------------
    diff_small = {
        "type": "mlp", "n_layers": 2, "n_units": [32, 32],
        "dropout": 0.1, "activation": "gelu", "out_activation": "softplus",
    }
    diff_big = {
        "type": "mlp", "n_layers": 2, "n_units": [64, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "softplus",
    }

    param_grid = {
        "model.latent_dim": [16, 32],
        "nsde.diffusion":  [diff_small, diff_big],
        "trainer.optimizer.params.lr": [2e-3, 5e-4, 1e-4],
    }

    search = OptunaGridSearch(
        param_grid=param_grid,
        dataloader=dl,
        base_encoder_cfg=base_enc,
        base_nsde_cfg=base_nsde,
        base_trainer_cfg=base_tr,
        model_cls=ShortRateModel,
        trainer_cls=Trainer,
        direction="minimize",
        seed=0,
        study_name="YC_tier1",
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
