"""
Optuna grid-search smoke run over simple encoder + simple NSDE.

Expensive — do NOT include in the standard pytest sweep. Run manually:

    cd <repo-root>
    python -m tests.full_test_run.grid_SimpleSimple_v1
"""
import os
from datetime import datetime, timedelta

from src import MarketDataLoader, ShortRateModel, Trainer
from src.configs import DataLoaderCfg, EncoderCfg, NSDECfg, TrainerCfg
from src import OptunaGridSearch


def main():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 2, 28)

    data_path = "data"
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    # -------------------------------------------------------------------
    # Dataloader is EXTERNAL (created once)
    dl_cfg = DataLoaderCfg(data_path=data_path, start_date=start, end_date=end, max_maturity=4)
    dl = MarketDataLoader(dl_cfg)

    # Base configs (copied per trial)
    enc_cfg = EncoderCfg(mode="simple")
    nsde_cfg = NSDECfg(type="simple")

    tr_cfg = TrainerCfg()

    tr_cfg.run_name = "Grid"
    tr_cfg.early_stopping.enabled = False
    tr_cfg.use_amp = False
    tr_cfg.accumulate_windows = 1

    tr_cfg.batch_window = 8
    tr_cfg.window_step = 1
    tr_cfg.lookback = 8
    tr_cfg.lookback_freq = 1
    tr_cfg.n_paths = 16

    diffusion_small = {
        "type": "mlp",
        "n_layers": 2,
        "n_units": 16,
        "activation": "softmax",
        "out_activation": "identity",
    }

    diffusion_big = {
        "type": "mlp",
        "n_layers": 2,
        "n_units": 32,
        "activation": "softmax",
        "out_activation": "identity",
    }

    # IMPORTANT:
    # - model.* controls model init dims
    # - trainer.* supports deep dot-setting (including dict traversal under trainer.*)
    # - encoder/nsde nets must be replaced as a whole mapping (no deep edits)
    param_grid = {
        "model.latent_dim": [16],
        "model.noise_dim": [4, 8],

        "nsde.noise_type": ["diagonal", "general"],
        "nsde.diffusion": [diffusion_small, diffusion_big],
        "trainer.scheduler.name": ['step', 'plateau'],
        # dot-path into trainer optimizer dict (supported by gridsearch)
        "trainer.optimizer.params.lr": [1e-3],
    }

    # Evaluation window example
    eval_date1 = dl.get_next_available_yield_curve_date(start)
    eval_date2 = eval_date1 + timedelta(days=5)

    search = OptunaGridSearch(
        param_grid=param_grid,
        dataloader=dl,                    # <-- external dataloader
        base_encoder_cfg=enc_cfg,
        base_nsde_cfg=nsde_cfg,
        base_trainer_cfg=tr_cfg,
        model_cls=ShortRateModel,
        trainer_cls=Trainer,
        direction="minimize",
        seed=0,
        study_name="irm_grid",
    )

    search.run(
        num_epochs=3,
        train_start_date=start,
        train_end_date=end,
        eval_start_date=eval_date1,
        eval_end_date=eval_date2,
        eval_step=1,
        save_eval=True,
    )


if __name__ == "__main__":
    main()