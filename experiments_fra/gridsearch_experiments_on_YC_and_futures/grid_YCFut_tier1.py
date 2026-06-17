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
    - ``nsde.diffusion_scale = 0.02`` — sets the IMPLIED short-rate vol to
                                         ~1.2 %/yr (sigma_r ~ 0.58*scale).
                                         The key yield-curve fix: too large
                                         a diffusion makes the convexity
                                         term swamp every yield. A *scale*
                                         (not a tanh bound) keeps gradients
                                         flowing. See model_diagnosis_report.md.
    - ``nsde.drift_scale = 0.5``      — natural magnitude of the drift
                                         (the expectations curve).
    - ``futures_relative_loss=True``  — futures loss is a dimensionless
                                         relative error, so λ_y = λ_f = 1
                                         balances yields against futures.
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

* **Loss weighting (CRITICAL).** With ``futures_relative_loss=True`` the
  futures term is a dimensionless relative error, so it lives on the same
  O(1e-4) scale as the yield MSE and ``λ_y = λ_f = 1`` genuinely balances
  the two. (Previously, absolute futures MSE on ~$120 prices was 10³-10⁵×
  the yield contribution, so no small ``λ_f`` could balance them and the
  model effectively fit futures only.) Components are still logged
  unweighted under ``loss_components``. For a clean yields-only warm-up,
  set ``loss_weights.futures = 0.0`` — the trainer short-circuits the
  futures branch entirely when the weight is zero.

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
# latent_dim reduced 64 -> 32 (hparam audit): the short rate is a scalar
# projection of the latent, so 64 dims dilute the signal without adding
# identifiable structure; 32 keeps ample capacity for the curve factors.
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
    # INPUT PREPROCESSING — feed the encoder PERCENT units (x100). Raw
    # decimal yields differ day-to-day by only ~5e-4, far too weak a signal
    # for the LSTM to distinguish curves; this was crippling z0, the only
    # day-specific quantity in the whole model (codebase_review B4).
    base_enc.preprocess_mode = "scale100"
    base_enc.net = {
        "type": "lstm",
        "n_layers": 2,
        "n_units": 128,
        # Dropout OFF: dropout in the encoder (and especially inside SDE
        # coefficient nets, below) makes the train-time dynamics stochastic
        # and different from eval-time dynamics — a measure mismatch a
        # calibration model should not have.
        "dropout": 0.0,
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

    # COEFFICIENT SCALES (not hard clamps) — set the *natural magnitude*
    # of the SDE coefficients by construction, while leaving the network
    # free to adapt them (no tanh saturation killing gradients). For this
    # model the diffusion scale IS the implied rate vol:
    # sigma_r ≈ ||decoder_w|| * diffusion_scale ≈ 0.58 * diffusion_scale.
    # A realistic ~1.2 %/yr vol means diffusion_scale ≈ 0.02. With the old
    # implicit scale (~0.69 from the softplus floor) the implied rate vol
    # was ~40 %/yr, whose convexity term (Var(∫r)/2T ~ hundreds of %)
    # swamped every yield and forced the model into the degenerate
    # near-flat corner — see model_diagnosis_report.md.
    base_nsde.diffusion_scale = 0.02
    base_nsde.drift_scale = 0.5
    # OU-only: cap the mean-reversion rate so the encoder's z0 isn't erased
    # over the 1-10y curve (uncapped kappa -> instant convergence to the
    # long-term mean -> flat, day-independent yields; the "OU gives constant
    # yields" failure). Effective rate = drift_scale * min(kappa, this) <=
    # 0.25, i.e. a mean-reversion timescale >= ~4y. Ignored by the simple
    # trials. See model_diagnosis_report.md.
    base_nsde.mean_reversion_max = 0.5
    # Hard bounds left OFF (None): with a tiny diffusion, near-identity
    # init, grad clipping and the NaN-guard, the SDE is already calm — we
    # don't need a saturating clamp fighting the drift it needs to learn.
    base_nsde.drift_bound = None
    base_nsde.diffusion_bound = None

    # NEAR-IDENTITY INIT — shrink the output layer of every drift/diffusion
    # network by 0.1 (and zero its bias) so the SDE starts almost
    # coefficient-free and the first few epochs are calm. This is what
    # removes the residual ~3% optimizer-step skips on the less-stable grid
    # configs (e.g. OU drift / big diffusion), which otherwise begin with
    # large random coefficients that spike the gradient through the unroll.
    base_nsde.init_output_scale = 0.1

    # NO dropout inside SDE coefficient nets: a fresh dropout mask at every
    # Euler step makes the drift/diffusion stochastic during training but
    # deterministic at eval — the trained dynamics and the evaluated
    # dynamics are then *different processes* (and the masks act as extra
    # unmodelled noise on top of the Brownian term).
    common_drift_net = {
        "type": "mlp",
        "n_layers": 3, "n_units": [128, 128, 64],
        "dropout": None, "activation": "gelu", "out_activation": "identity",
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
    base_tr.window_step = 2          # every 2nd day (was 4): 2x data + updates/epoch
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
    # Relative futures loss makes λ_f interpretable and comparable to λ_y
    # (absolute MSE on $120 prices vs decimal² yields was a 10³-10⁵×
    # mismatch that no small λ_f could fix). With it on, λ_y = λ_f = 1
    # genuinely balances the curve against the futures.
    base_tr.futures_relative_loss = True
    base_tr.loss_weights.yield_curve = 1.0
    # Short-rate target weight = 0: r0 is ANCHORED to the observed short
    # rate in decode(), so MSE_sr ≈ 1e-19 by construction — a dead term
    # that contributes no gradient. NOTE: enable_short_rate stays True in
    # the dataloader (the short-rate history feeds the encoder and the r0
    # anchor); only the loss weight is zeroed.
    base_tr.loss_weights.short_rate  = 0.0
    base_tr.loss_weights.futures     = 1.0

    # BondNet -> model-PV consistency (LSMC): regress BondNet's
    # deliverable-bond prices onto the model's OWN pathwise-discounted
    # cashflows (same simulated paths — no nested simulation; target
    # DETACHED, gradient reaches BondNet only). Weight reduced from 1.0:
    # in tier1_3 the term carried an irreducible pathwise-variance floor
    # that made up ~91% of the total, hiding yield-scale progress from the
    # EMA early-stopper and (in its non-detached form) bending the curve
    # itself. At 0.25 the BondNet regression still gets plenty of signal
    # while the training total stays legible.
    base_tr.bondnet_consistency_weight = 0.25

    # SHORT-RATE VOL ANCHOR — pins the one quantity the calibration data
    # cannot identify. Yields are vol-insensitive up to bp-level convexity,
    # so left free the diffusion drifts to degenerate values (tier1_3
    # trained to sigma_r ~ 0.1-0.4 %/yr; the MC fan collapsed to a single
    # line). By Girsanov the diffusion is identical under P and Q, so
    # anchoring the model's 1y cross-path std to the historically measured
    # short-rate vol (~1 %/yr for USD) is principled. Scale: at the
    # anchored optimum the term ~ 0; a 0.9% mismatch contributes
    # 10 * (0.009)^2 ~ 8e-4 — strong enough to dominate a stale yield
    # plateau, gentle enough not to swamp live yield gradients.
    base_tr.rate_vol_target = 0.01
    base_tr.rate_vol_weight = 10.0

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

    # CONSTANT diffusion fixed (not gridded): a learnable per-dim constant
    # (the Vasicek/Hull-White choice). With the rate-vol anchor pinning its
    # magnitude, state-dependent vol is a second-order question; we spend
    # the grid budget on the measure axis instead.
    base_nsde.diffusion = {
        "type": "constant",
        "init": "zeros",                      # softplus(0)=0.693 * scale = sane start
        "out_activation": "softplus",
    }

    decoder_mlp = {
        "type": "mlp",
        "n_layers": 2, "n_units": [64, 32],
        "dropout": None, "activation": "gelu", "out_activation": "identity",
    }

    # -------------------------------------------------------------------
    # Grid — 2 × 2 × 2 = 8 trials. The headline axis is the MEASURE:
    # * nsde.type: simple vs OU drift family.
    # * model.decoder: linear (one-factor short rate, since r = w·z + b is a
    #   scalar projection) vs 2-layer MLP (multi-factor curve shapes).
    # * trainer.pq_consistency_weight: 0.0 = pure risk-neutral (Q-only)
    #   calibration; 0.5 = JOINT P/Q calibration — the physical-measure
    #   forecast of the short rate is matched to realised data, training the
    #   market price of risk lambda (term premium). This is the modelling
    #   question of interest: does pinning the P-dynamics improve / stabilise
    #   the Q curve, and what term premium does the data imply.
    # -------------------------------------------------------------------
    param_grid = {
        "nsde.type":                      ["simple", "ou"],
        "model.decoder":                  [None, decoder_mlp],   # None = default nn.Linear
        "trainer.pq_consistency_weight":  [0.0, 0.5],            # Q-only vs P/Q
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
