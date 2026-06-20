"""
End-to-end smoke run: hierarchical encoder + OU NSDE with general noise.

Expensive — do NOT include in the standard pytest sweep. Run manually:

    cd <repo-root>
    python -m tests.full_test_run.fullrun_HierOU_gen
"""
import os
from datetime import datetime
from datetime import timedelta

from src import MarketDataLoader
from src import ShortRateModel
from src import Trainer

from src.configs import EncoderCfg
from src.configs import NSDECfg
from src.configs import TrainerCfg
from src.configs import DataLoaderCfg

from src.analysis.result_analyzer import ResultAnalyzer


def main() -> None:
    start = datetime(2020, 1, 1)
    end = datetime(2020, 2, 28)

    data_path = "data"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Yield curve csv not found: {data_path}")

    # Hierarchical encoder + general-noise OU NSDE.
    data_cfg = DataLoaderCfg(
        data_path=data_path,
        start_date=start,
        end_date=end,
        max_maturity=5,
    )
    dl = MarketDataLoader(data_cfg)

    encoder_cfg = EncoderCfg(mode="hierarchical", combine="concat")

    nsde_cfg = NSDECfg(type="ou")
    nsde_cfg.noise_type = "general"
    nsde_cfg.diffusion = {
        "type": "mlp",
        "n_layers": 2,
        "n_units": 32,
        "activation": "relu",
        "out_activation": "tanh",
    }

    model = ShortRateModel(
        name="HierOU_general",
        encoder=encoder_cfg,
        nsde=nsde_cfg,
        noise_dim=4,
        latent_dim=16,
    )

    trainer_cfg = TrainerCfg()
    trainer_cfg.run_name = "HierOU_gen"
    trainer_cfg.early_stopping.enabled = False
    trainer_cfg.use_amp = False
    trainer_cfg.accumulate_windows = 1

    trainer_cfg.n_paths = 32
    trainer_cfg.batch_window = 8
    trainer_cfg.window_step = 1

    trainer_cfg.lookback_fast = 16
    trainer_cfg.lookback_fast_freq = 1
    trainer_cfg.lookback_slow = 16
    trainer_cfg.lookback_slow_freq = 2

    trainer_cfg.debug = True

    trainer = Trainer(model=model, dataloader=dl, config=trainer_cfg)

    trainer.train(5, start_date=start, end_date=end)

    evaluation_date1 = dl.get_next_available_yield_curve_date(start)
    evaluation_date2 = evaluation_date1 + timedelta(days=5)
    _ = trainer.evaluate(date=evaluation_date1)
    _ = trainer.evaluate(start_date=evaluation_date1, end_date=evaluation_date2)
    _ = trainer.evaluate_training_set()

    ra = ResultAnalyzer(run_dir=trainer.output_dir)
    print(ra.summary())
    ra.plot_epoch_loss()
    ra.plot_eval_total_loss()
    ra.plot_eval_components()
    print("Done.")


if __name__ == "__main__":
    main()
