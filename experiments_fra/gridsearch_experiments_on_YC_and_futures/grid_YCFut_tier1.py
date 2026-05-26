"""
Tier-1 grid search for the **joint YC + Treasury futures** calibration.

Training window: 2015-01-01 → 2024-09-30 (~10 years), evaluation rolls
forward one quarter past the training end.

Design choices
--------------
* **Grid axes (8 trials total)** — only the architectural / structural
  decisions the project_description leaves underdetermined:

    1. ``model.latent_dim``      — latent capacity {16, 32}
    2. ``nsde.type``              — drift family {simple, OU}
    3. ``nsde.diffusion``         — diffusion network size {small, big}

  Learning rate is **fixed**, not gridded — once `warmup_cosine` is in
  play the LR sensitivity is much smaller than the structural choices
  above. The grid is reduced to the things that change the *model*,
  not the *optimiser*.

* **Fixed knobs** were chosen for a 10-year joint run with a few
  hundred Monte Carlo paths:

    - ``n_paths = 512``           — MC budget recommended by the project.
    - ``batch_window = 16``       — windows × n_paths set the simulate
                                    memory peak; 16×512×641×32×4 ≈ 670 MB.
    - ``window_step = 4``         — ~150 windows / epoch on 10 yrs (~one
                                    optimiser step per business week).
    - ``trainer.dt = 1/64``       — simulation grid spacing.
    - ``nsde.dt = 1/128``         — solver step (≤ trainer.dt).
    - ``lookback = 64``           — ~one quarter of yield-curve history.
    - ``epochs = 50``             — with warmup_cosine(warmup=5, max=50)
                                    and patience=12 early stopping.

* **Scheduler**: ``warmup_cosine`` (5 warmup epochs, cosine decay to
  ``eta_min`` thereafter) — preferred by the project_description for
  joint calibration runs.

* **Solver**: ``custom_euler`` — ~10× faster than torchsde for the
  Euler scheme used throughout.

Note on memory
--------------
On a 16 GB CPU machine this configuration sits at ~3–4 GB peak. If you
need more headroom, drop ``batch_window`` to 8 first (halves the
simulate tensor), then ``n_paths`` to 384. Avoid dropping ``dt`` —
the discretisation bias is hard to recover.

Note on the joint-loss imbalance
--------------------------------
At init the futures loss (~thousands) dominates the yield loss (~tenths
of a percent²). Until `λ_y, λ_f` are exposed on TrainerCfg (see
``optimization_plan.md`` §10.1 P1), this run effectively fits futures.
Use the YC-only baseline if you need a clean yield-curve fit first.

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


def main() -> None:
    # -------------------------------------------------------------------
    # Dates — 10-year training window, 1-quarter held-out evaluation
    # -------------------------------------------------------------------
    train_start = datetime(2015, 1, 1)
    train_end   = datetime(2024, 6, 30)
    eval_start  = datetime(2024, 7, 1)
    eval_end    = datetime(2024, 9, 30)

    # -------------------------------------------------------------------
    # Device — auto-detect CUDA, fall back to CPU
    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------
    data_path = "data2"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    data_cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=train_start - timedelta(days=150),    # extra room for the lookback
        end_date=eval_end + timedelta(days=30),
        max_maturity=10,
        enable_yield=True,
        enable_short_rate=True,
        enable_futures=True,
        device=device,
    )
    dl = MarketDataLoader(data_cfg)

    # -------------------------------------------------------------------
    # Base encoder — simple, bi-LSTM. 64 hidden units, 2 layers.
    # -------------------------------------------------------------------
    base_enc = EncoderCfg(mode="simple")
    base_enc.out_norm = "layernorm"
    base_enc.net = {
        "type": "lstm",
        "n_layers": 2,
        "n_units": 128,
        "dropout": 0.1,
        "bidirectional": True,
        "out_activation": "identity",
    }

    # -------------------------------------------------------------------
    # Base NSDE config — overrides per trial set `type` and `diffusion`.
    # -------------------------------------------------------------------
    base_nsde = NSDECfg(type="simple", noise_type="diagonal")
    base_nsde.solver = "custom_euler"            # ~10x faster than torchsde
    base_nsde.dt = 1 / 252                       # solver step  (≤ trainer.dt)

    common_drift_net = {
        "type": "mlp",
        "n_layers": 3, "n_units": [128, 128, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "identity",
    }
    base_nsde.drift = common_drift_net           # used by type="simple"
    base_nsde.long_term_mean = common_drift_net  # used by type="ou"
    base_nsde.mean_reversion = {                 # used by type="ou"
        **common_drift_net,
        "out_activation": "softplus",            # kappa >= 0 → genuine MR
    }

    # -------------------------------------------------------------------
    # Trainer — fixed knobs for the joint run
    # -------------------------------------------------------------------
    base_tr = TrainerCfg()
    base_tr.results_root = "results"
    base_tr.run_name = "YCFut_grid_tier1"
    base_tr.log_every_n_windows = 20

    # MC + window sizing — see "Note on memory" in the module docstring
    base_tr.n_paths = 512
    base_tr.batch_window = 32
    base_tr.window_step = 4
    base_tr.accumulate_windows = 2
    base_tr.dt = 1 / 128

    # Encoder lookback ~ one quarter of business days
    base_tr.lookback = 64
    base_tr.lookback_freq = 2

    # Optimiser — adamw with a single learning rate (no grid axis)
    base_tr.optimizer.name = "adamw"
    base_tr.optimizer.params = {"lr": 1e-3, "weight_decay": 1e-4}

    # Scheduler — warmup_cosine with 5 warmup epochs and cosine decay to
    # eta_min over the remaining (max_epochs - warmup_epochs) epochs.
    base_tr.scheduler.name = "warmup_cosine"
    base_tr.scheduler.params = {
        "warmup_epochs": 20,
        "max_epochs": 100,
        "eta_min": 1e-5,
    }

    # AMP only on CUDA. compile_nsde kept off — the in-house Euler is
    # already fast and torch.compile interacts poorly with autograd-heavy
    # SDE loops.
    base_tr.use_amp = device.type == "cuda"
    base_tr.compile_nsde = False
    base_tr.grad_clip_norm = 1.0

    # Early stopping — 12 epochs of patience on the EMA-smoothed train
    # loss. Warmup epochs (5) won't trigger early-stop on their own.
    base_tr.early_stopping.enabled = True
    base_tr.early_stopping.patience = 20
    base_tr.early_stopping.min_delta = 1e-4

    # Checkpoints — keep just the best one + periodic for resume.
    base_tr.checkpoint.mode = "min"
    base_tr.checkpoint.save_best_only = True
    base_tr.checkpoint.every_n_epochs = 10
    base_tr.checkpoint.max_to_keep = 3

    # -------------------------------------------------------------------
    # BondNet — passed via `base_bondnet_cfg` (math_review.md §8). The
    # gridsearch keeps `bondnet.latent_dim` in sync with the per-trial
    # `model.latent_dim` choice automatically.
    # -------------------------------------------------------------------
    base_bondnet = SimpleBondNetCfg(
        latent_dim=64,                       # overwritten per trial
        bond_feat_dim=BOND_FEAT_DIM,
        latent_n_layers=2, 
        latent_n_units=128,
        bond_n_layers=2,   
        bond_n_units=64,
        fusion_n_layers=2, 
        fusion_n_units=128,
        activation="silu",
        output_positive=True,
    )

    # -------------------------------------------------------------------
    # Grid — 2 × 2 × 2 = 8 trials (LR deliberately NOT in the grid)
    # -------------------------------------------------------------------
    diffusion_small = {
        "type": "mlp",
        "n_layers": 2, "n_units": [64, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "softplus",
    }
    diffusion_big = {
        "type": "mlp",
        "n_layers": 3, "n_units": [128, 128, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "softplus",
    }

    param_grid = {
        "model.latent_dim":  [16, 32],
        "nsde.type":         ["simple", "ou"],
        "nsde.diffusion":    [diffusion_small, diffusion_big],
    }

    search = OptunaGridSearch(
        param_grid=param_grid,
        dataloader=dl,
        base_encoder_cfg=base_enc,
        base_nsde_cfg=base_nsde,
        base_trainer_cfg=base_tr,
        base_bondnet_cfg=base_bondnet,
        model_cls=ShortRateModel,
        trainer_cls=Trainer,
        direction="minimize",
        seed=0,
        study_name="YCFut_tier1",
    )

    results = search.run(
        num_epochs=100,
        train_start_date=train_start,
        train_end_date=train_end,
        eval_start_date=eval_start,
        eval_end_date=eval_end,
        eval_step=5,
        save_eval=True,
    )

    print(f"\nBest: {results.best_params}  value={results.best_value:.4f}")
    for t in results.trials:
        print(f"  trial {t.number}  {t.params}  value={t.value:.4f}")


if __name__ == "__main__":
    main()
