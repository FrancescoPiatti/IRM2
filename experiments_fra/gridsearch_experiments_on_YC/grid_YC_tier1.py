"""
Tier-1 grid search for the **yield-curve-only** calibration.

Training window: 2016-01-01 → 2024-06-30 (~8.5 years), evaluation rolls
forward one quarter past the training end.

Design choices
--------------
* **Grid axes (4 trials total)** — kept small per project guidance.
  Only the architectural decisions vary:

    1. ``nsde.type``       — drift family {simple, OU}
    2. ``nsde.diffusion``  — diffusion network width {small, big}

  Learning rate, latent dim, and encoder layout are fixed at the values
  that survived the §2 / §3 / §5 hot-path work in
  ``optimization_plan.md``.

* **Fixed hparams (GPU-tuned)**:

    - ``device = cuda`` if available (falls back to CPU as a safety net).
    - ``n_paths = 256``                — modest MC budget for the
                                         yields-only objective.
    - ``batch_window = 16``            — fits on a 10 GB GPU with AMP.
    - ``window_step = 2``              — denser optimiser steps when
                                         the dataset is single-target.
    - ``trainer.dt = 1/128``           — simulation grid spacing.
    - ``nsde.dt = 1/252``              — solver step (≤ trainer.dt).
    - ``lookback = 64`` / ``freq = 2`` — ~one quarter of YC history.
    - ``epochs = 50``                  — warmup_cosine(warmup=10, max=50)
                                         + patience=10 early stopping.
    - ``use_amp = True`` on CUDA.
    - ``checkpoint_chunk_size = 8``    — gradient checkpointing on the
                                         SDE Euler loop.
    - ``grad_clip_norm = 1.0``         — keeps NaN losses away
                                         (``code_review_report.md`` §2).

* **Loss weighting.** Yields-only, so we set
  ``loss_weights.futures = 0.0`` and ``loss_weights.short_rate = 0.0``.
  The trainer's per-target short-circuit (``Trainer._get_loss``) then
  skips those branches entirely, saving the redundant pricer calls.

* **Encoder**: bi-LSTM 2×96 with RMSNorm output normalisation.

* **Solver**: ``custom_euler`` (in-house Euler-Maruyama, ~10-15× faster
  than torchsde at this step size).

Run from the repo root:

    python -m experiments_fra.gridsearch_experiments_on_YC.grid_YC_tier1
"""
import os
from datetime import datetime, timedelta

import torch

from src import MarketDataLoader, ShortRateModel, Trainer, OptunaGridSearch
from src.configs import DataLoaderCfg, EncoderCfg, NSDECfg, TrainerCfg


LATENT_DIM = 32


def main() -> None:
    # -------------------------------------------------------------------
    # Dates — 8.5-year training window starting 2016-01-01 (user spec)
    # -------------------------------------------------------------------
    train_start = datetime(2016, 1, 1)
    train_end   = datetime(2024, 6, 30)
    eval_start  = datetime(2024, 7, 1)
    eval_end    = datetime(2024, 9, 30)

    # -------------------------------------------------------------------
    # Device — expected GPU; CPU only as a safety net
    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print(
            "WARNING: GPU not detected. This grid is sized for GPU; "
            "consider lowering batch_window or n_paths on CPU."
        )

    # -------------------------------------------------------------------
    # Shared dataloader
    # -------------------------------------------------------------------
    data_path = "data2"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    data_cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=train_start - timedelta(days=200),
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
    base_enc.out_norm = "rmsnorm"
    base_enc.net = {
        "type": "lstm",
        "n_layers": 2,
        "n_units": 96,
        "dropout": 0.1,
        "bidirectional": True,
    }

    base_nsde = NSDECfg(type="simple", noise_type="diagonal")
    base_nsde.solver = "custom_euler"
    base_nsde.dt = 1 / 252                      # solver step on the 252-day year
    base_nsde.checkpoint_chunk_size = 8         # SDE gradient checkpointing

    base_tr = TrainerCfg()
    base_tr.results_root = "results"
    base_tr.run_name = "YC_grid_tier1"
    base_tr.n_paths = 256
    base_tr.batch_window = 16
    base_tr.window_step = 2
    base_tr.accumulate_windows = 2
    base_tr.dt = 1 / 128
    base_tr.lookback = 64
    base_tr.lookback_freq = 2

    # Yields-only — zero out the other branches so they short-circuit.
    base_tr.loss_weights.yield_curve = 1.0
    base_tr.loss_weights.short_rate  = 0.0
    base_tr.loss_weights.futures     = 0.0

    base_tr.optimizer.name = "adamw"
    base_tr.optimizer.params = {"lr": 1e-3, "weight_decay": 1e-4}

    base_tr.scheduler.name = "warmup_cosine"
    base_tr.scheduler.params = {
        "warmup_epochs": 10,
        "max_epochs": 50,
        "eta_min": 1e-5,
    }

    base_tr.use_amp = device.type == "cuda"
    base_tr.compile_nsde = False
    base_tr.grad_clip_norm = 1.0
    base_tr.log_every_n_windows = 20

    base_tr.early_stopping.enabled = True
    base_tr.early_stopping.patience = 10
    base_tr.early_stopping.min_delta = 1e-4

    base_tr.checkpoint.mode = "min"
    base_tr.checkpoint.save_best_only = True
    base_tr.checkpoint.every_n_epochs = 10
    base_tr.checkpoint.max_to_keep = 3

    # -------------------------------------------------------------------
    # Tier-1 grid — 2 × 2 = 4 trials
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
        "nsde.type":      ["simple", "ou"],
        "nsde.diffusion": [diff_small, diff_big],
    }

    search = OptunaGridSearch(
        param_grid=param_grid,
        dataloader=dl,
        base_encoder_cfg=base_enc,
        base_nsde_cfg=base_nsde,
        base_trainer_cfg=base_tr,
        model_cls=ShortRateModel,
        trainer_cls=Trainer,
        latent_dim=LATENT_DIM,            # fixed across trials
        direction="minimize",
        seed=0,
        study_name="YC_tier1",
    )

    results = search.run(
        num_epochs=50,
        train_start_date=train_start,
        train_end_date=train_end,
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
