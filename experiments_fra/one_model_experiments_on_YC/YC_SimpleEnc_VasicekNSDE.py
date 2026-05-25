"""
Yield-curve-only training: **simple encoder + Vasicek-style NSDE** (diagonal noise).

"Vasicek-style" = OU drift + **constant** diffusion network: σ does not depend
on (t, z). This is the closest neural analogue of the classical Vasicek
short-rate model:

    dz = kappa(t, z) * (theta(t, z) - z) dt + sigma * dW   with sigma constant

Run from the repo root:

    python -m experiments_fra.one_model_experiments_on_YC.YC_SimpleEnc_VasicekNSDE
"""
import os
from datetime import datetime, timedelta

import torch

from src import MarketDataLoader, ShortRateModel, Trainer
from src.configs import EncoderCfg, NSDECfg, TrainerCfg, DataLoaderCfg
from src.analysis.result_analyzer import ResultAnalyzer


def main() -> None:
    start, end = datetime(2021, 10, 1), datetime(2024, 10, 10)
    max_maturity = 15
    device = torch.device("cpu")

    latent_dim = 64
    epochs = 70

    data_path = "data2"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    data_cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=start - timedelta(days=60),
        end_date=end + timedelta(days=30),
        max_maturity=max_maturity,
        enable_yield=True,
        device=device,
    )
    dl = MarketDataLoader(data_cfg)

    encoder_cfg = EncoderCfg(mode="simple")
    encoder_cfg.out_norm = "layernorm"
    encoder_cfg.net = {
        "type": "lstm",
        "n_layers": 3,
        "n_units": 128,
        "dropout": 0.1,
        "bidirectional": True,
        "out_activation": "identity",
    }

    nsde_cfg = NSDECfg(type="ou", noise_type="diagonal")
    nsde_cfg.solver = "custom_euler"
    nsde_cfg.dt = 1 / 128
    nsde_cfg.long_term_mean = {
        "type": "mlp",
        "n_layers": 3, "n_units": [128, 128, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "identity",
    }
    nsde_cfg.mean_reversion = {
        "type": "mlp",
        "n_layers": 3, "n_units": [128, 128, 64],
        "dropout": 0.1, "activation": "gelu", "out_activation": "softplus",
    }
    # Vasicek-style: constant (state-independent) volatility.
    nsde_cfg.diffusion = {
        "type": "constant",
        "out_activation": "softplus",
        "init": "normal",
        "init_std": 0.05,
    }

    model = ShortRateModel(
        name="YC_SimpleEnc_VasicekNSDE",
        encoder=encoder_cfg,
        nsde=nsde_cfg,
        latent_dim=latent_dim,
    )

    trainer_cfg = TrainerCfg()
    trainer_cfg.run_name = "YC_SVas_v1"
    trainer_cfg.log_every_n_windows = 5
    trainer_cfg.n_paths = 256
    trainer_cfg.batch_window = 16
    trainer_cfg.accumulate_windows = 2
    trainer_cfg.window_step = 2
    trainer_cfg.dt = 1 / 64
    trainer_cfg.lookback = 32
    trainer_cfg.lookback_freq = 2
    trainer_cfg.optimizer.name = "adamw"
    trainer_cfg.optimizer.params = {"lr": 2e-3, "weight_decay": 2e-4}
    trainer_cfg.scheduler.name = "plateau"
    trainer_cfg.use_amp = device.type == "cuda"
    trainer_cfg.grad_clip_norm = 1.0
    trainer_cfg.early_stopping.enabled = True
    trainer_cfg.checkpoint.save_best_only = True

    trainer = Trainer(model=model, dataloader=dl, config=trainer_cfg)
    trainer.train(epochs, start_date=start, end_date=end)

    eval_d1 = dl.get_next_available_yield_curve_date(end)
    eval_d2 = eval_d1 + timedelta(days=10)
    _ = trainer.evaluate(date=eval_d1)
    _ = trainer.evaluate(start_date=eval_d1, end_date=eval_d2)
    _ = trainer.evaluate_training_set()

    ra = ResultAnalyzer(run_dir=trainer.output_dir)
    print(ra.summary())
    ra.plot_epoch_loss()
    ra.plot_eval_total_loss()
    ra.plot_eval_components()
    print("Done.")


if __name__ == "__main__":
    main()
