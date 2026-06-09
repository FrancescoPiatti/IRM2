"""
Tier-1 grid search for the **joint YC + Treasury futures** calibration.

Training window: 2016-01-01 → 2024-06-30 (~8.5 years post-GFC tightening
cycle, futures basket data well-populated), evaluation rolls forward
one quarter past the training end.

Design choices
--------------
* **Grid axes (4 trials total)** — kept deliberately small per project
  guidance: the grid only varies the architectural decisions that the
  project_description leaves underdetermined, not optimiser knobs.

    1. ``nsde.type``       — drift family {simple, OU}
    2. ``nsde.diffusion``  — diffusion network width {small, big}

  Learning rate is **fixed** at 1e-3 (the moderate-config value verified
  stable across 24 windows in ``code_review_report.md`` §2.1). Latent
  dim is fixed at 32 — bigger dims gave diminishing returns on the
  yield-curve fit in the §2 profile run.

* **Fixed hparams (GPU-tuned)**:

    - ``device = cuda`` if available — auto-fallback to CPU only as a
      safety net. The grid expects to run on GPU.
    - ``n_paths = 512``                — MC budget recommended by the
                                         project_description.
    - ``batch_window = 16``            — fits ~10 GB VRAM with AMP +
                                         gradient checkpointing.
    - ``window_step = 4``              — ~one optimiser step per
                                         business week of training data.
    - ``trainer.dt = 1/32``            — simulation grid spacing →
                                         10yr × 32 = 320 Euler steps.
                                         Short unroll = stable backprop
                                         (was 1/128 → 1280 steps, which
                                         exploded the gradient).
    - ``nsde.drift_bound = 5``,
      ``nsde.diffusion_bound = 2``     — smooth tanh bounds on the SDE
                                         coefficients so the latent state
                                         can't run away over the unroll.
    - ``lookback = 64`` / ``freq = 2`` — ~one quarter of YC history.
    - ``epochs = 100``                 — warmup_cosine(warmup=20, max=100)
                                         + patience=20 early stopping.
    - ``use_amp = False``              — AMP is deliberately OFF here.
                                         With ``latent_dim=64``, the
                                         1280-step Euler loop running
                                         under autocast pushed gradients
                                         to NaN within a handful of
                                         windows. Once stability is
                                         confirmed on a config, you can
                                         flip this back on for the
                                         throughput win.
    - ``checkpoint_chunk_size = 8``    — Gradient checkpointing on the
                                         SDE Euler loop is back ON here
                                         (it was the AMP+checkpointing
                                         *combination* that caused NaN,
                                         not checkpointing itself —
                                         ``test_custom_euler_checkpointed_matches_uncheckpointed``
                                         confirms fp32 checkpointing is
                                         bitwise-equivalent). Without it
                                         the full fp32 autograd graph
                                         doesn't fit on a single GPU at
                                         ``latent_dim=64`` /
                                         ``batch_window=16`` /
                                         ``n_paths=512``.
    - ``grad_clip_norm = 0.5``         — Tighter than the 1.0 default
                                         because we observed parameter
                                         drift even with clip=1.0 under
                                         the aggressive joint loss.

* **Loss weighting (CRITICAL).** ``loss_weights.futures = 1e-4`` so
  the futures MSE (raw scale ~10^3 on $-prices) doesn't drown out the
  yield-curve MSE (raw scale ~10^-4 on decimals). 1e-2 was still ~10^5
  larger than the yield contribution at init and contributed to the
  gradient blow-up observed in the first GPU run. The components are
  still logged unweighted under ``loss_components`` so you can read
  the underlying fit quality on its own scale. If you want a clean
  yields-only warm-up run, set this to ``0.0`` — the trainer
  short-circuits the futures branch entirely when the weight is zero.

* **Scheduler**: ``warmup_cosine`` (20 warmup epochs, cosine decay to
  ``eta_min = 1e-5``) — preferred for joint calibration runs.

* **Solver**: ``custom_euler`` — ~10–15× faster than torchsde for the
  Euler scheme used throughout (``optimization_report.md``).

* **Encoder**: bi-LSTM 2×128 with ``rmsnorm`` output normalisation.
  RMSNorm drops the mean-subtraction step of LayerNorm and tends to
  be slightly cheaper while behaving comparably (Zhang & Sennrich,
  2019).

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

# Fixed structural hparams (not gridded). Defining them up here keeps the
# search width explicit ("only 4 trials") and the magic numbers in one
# place for the docstring above.
LATENT_DIM = 64


def main() -> None:
    # -------------------------------------------------------------------
    # Dates — 8.5-year training window starting 2016-01-01 (user spec)
    # -------------------------------------------------------------------
    train_start = datetime(2016, 1, 1)
    train_end   = datetime(2024, 6, 30)
    eval_start  = datetime(2024, 7, 1)
    eval_end    = datetime(2024, 9, 30)

    # -------------------------------------------------------------------
    # Device — expected GPU. Falls back to CPU only as a safety net so
    # the script doesn't crash on machines without CUDA, but the hparams
    # below assume CUDA is available (AMP, gradient checkpointing chunks).
    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print(
            "WARNING: GPU not detected. This grid is sized for GPU; "
            "consider lowering batch_window or n_paths if you really "
            "intend to run it on CPU."
        )

    # -------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------
    data_path = "data2"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    data_cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=train_start - timedelta(days=200),    # extra room for the lookback
        end_date=eval_end + timedelta(days=30),
        max_maturity=10,
        enable_yield=True,
        enable_short_rate=True,
        enable_futures=True,
        device=device,
    )
    dl = MarketDataLoader(data_cfg)

    # -------------------------------------------------------------------
    # Base encoder — simple, bi-LSTM. 2 layers, 128 hidden units.
    # `out_norm="rmsnorm"` exercises the new pre-norm option added to
    # MambaBlock / the encoder out_norm registry. Switch to "layernorm"
    # if you want the previous behaviour.
    # -------------------------------------------------------------------
    base_enc = EncoderCfg(mode="simple")
    base_enc.out_norm = "rmsnorm"
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
    base_nsde.solver = "custom_euler"           # ~10-15x faster than torchsde
    base_nsde.dt = 1 / 252                      # solver step on the 252-day year
    # Gradient checkpointing on the SDE Euler loop. The NaN issue was the
    # AMP + checkpointing *combination*; checkpointing on its own is
    # bitwise-equivalent in fp32 (covered by the
    # ``test_custom_euler_checkpointed_matches_uncheckpointed`` test).
    # With the shorter unroll below (trainer.dt = 1/32 -> 320 steps) the
    # autograd graph is already 4x smaller, so this is mostly insurance.
    base_nsde.checkpoint_chunk_size = 8

    # SMOOTH COEFFICIENT BOUNDS — the key stability fix. Backprop through
    # the multi-hundred-step Euler unroll multiplies one Jacobian per
    # step; once the drift/diffusion grow, that product explodes to inf
    # and the optimizer-step guard rejects every update (the "non-finite
    # gradient" wall seen around epoch 8). Squashing the drift/diffusion
    # with ``bound * tanh(raw / bound)`` keeps the latent state — and
    # hence the Jacobians — in a sane range. The bounds are generous, so
    # they're near-identity for normal dynamics and only bite on blow-ups.
    base_nsde.drift_bound = 5.0
    base_nsde.diffusion_bound = 2.0

    # NEAR-IDENTITY INIT — shrink the output layer of every drift/diffusion
    # network by 0.1 (and zero its bias) so the SDE starts almost
    # coefficient-free and the first few epochs are calm. This is what
    # removes the residual ~3% optimizer-step skips on the less-stable grid
    # configs (e.g. OU drift / big diffusion), which otherwise begin with
    # large random coefficients that spike the gradient through the unroll.
    base_nsde.init_output_scale = 0.1

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

    # MC + window sizing — GPU memory budget
    base_tr.n_paths = 512
    base_tr.batch_window = 16
    base_tr.window_step = 4
    base_tr.accumulate_windows = 2
    # Simulation grid spacing. 1/32 -> 10yr * 32 = 320 Euler steps (was
    # 1/128 -> 1280). The backprop chain is the #1 driver of exploding
    # gradients: cutting it 4x makes the same LR far more stable, uses 4x
    # less autograd memory, and runs ~4x faster per window. 320 steps is
    # still plenty fine for the discount-factor integral over 1..10y.
    base_tr.dt = 1 / 32

    # Encoder lookback ~ one quarter of business days
    base_tr.lookback = 64
    base_tr.lookback_freq = 2

    # Per-target loss weights — the futures branch is on a much larger
    # scale than yields/short-rate, so we down-weight it to keep all
    # three components contributing to the gradient. Raw components are
    # still logged unweighted under `loss_components` for monitoring.
    base_tr.loss_weights.yield_curve = 1.0
    base_tr.loss_weights.short_rate  = 1.0
    base_tr.loss_weights.futures     = 1e-4

    # Optimiser — adamw with a single learning rate (no grid axis).
    # Peak LR = 2e-4. In the previous run the instability kicked in
    # around epoch 8 — exactly while the 20-epoch warmup was still
    # *ramping LR up* past ~2e-4. We both lower the peak and shorten the
    # warmup (below) so the optimiser isn't being pushed harder into the
    # unstable region as it goes.
    base_tr.optimizer.name = "adamw"
    base_tr.optimizer.params = {"lr": 2e-4, "weight_decay": 1e-4}

    # Scheduler — warmup_cosine with a SHORT 5-epoch warmup then cosine
    # decay to eta_min. Short warmup means LR peaks early and only ever
    # decays afterwards, instead of climbing into the instability.
    base_tr.scheduler.name = "warmup_cosine"
    base_tr.scheduler.params = {
        "warmup_epochs": 5,
        "max_epochs": 100,
        "eta_min": 1e-5,
    }

    # AMP is deliberately OFF for the stability run — see the
    # docstring at the top of the file. Re-enable with
    # ``base_tr.use_amp = device.type == "cuda"`` once you've
    # established a configuration that converges cleanly without it.
    base_tr.use_amp = False
    base_tr.compile_nsde = False
    base_tr.grad_clip_norm = 0.5

    # Early stopping — 20 epochs of patience on the EMA-smoothed train
    # loss. Warmup epochs (20) won't trigger early-stop on their own.
    base_tr.early_stopping.enabled = True
    base_tr.early_stopping.patience = 20
    base_tr.early_stopping.min_delta = 1e-4

    # Checkpoints — keep just the best one + periodic for resume.
    base_tr.checkpoint.mode = "min"
    base_tr.checkpoint.save_best_only = True
    base_tr.checkpoint.every_n_epochs = 10
    base_tr.checkpoint.max_to_keep = 3

    # -------------------------------------------------------------------
    # BondNet — passed via `base_bondnet_cfg`. The gridsearch keeps
    # `bondnet.latent_dim` in sync with the per-trial `model.latent_dim`
    # choice automatically.
    # -------------------------------------------------------------------
    base_bondnet = SimpleBondNetCfg(
        latent_dim=LATENT_DIM,
        bond_feat_dim=BOND_FEAT_DIM,
        latent_n_layers=2,
        latent_n_units=128,
        bond_n_layers=2,
        bond_n_units=64,
        fusion_n_layers=2,
        fusion_n_units=128,
        activation="silu",
        output_positive=True,
        # NEAR-TARGET INIT — the key stability fix for the joint run.
        # Deliverable bonds sit near par (~100); a zero-init Softplus head
        # starts at ~0.69 and has to grow its weights ~150x to reach the
        # bond-price level. Those huge weights amplify the gradient flowing
        # back into the SDE latent path and blow training up right after
        # warmup (the 82% → 100% skip wall at epoch 7-8). Starting the head
        # at ~100 keeps its weights — and that gradient — small.
        output_init_level=100.0,
    )

    # -------------------------------------------------------------------
    # Grid — 2 × 2 = 4 trials (kept small per project guidance)
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
        "nsde.type":      ["simple", "ou"],
        "nsde.diffusion": [diffusion_small, diffusion_big],
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
        latent_dim=LATENT_DIM,            # fixed across trials
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
