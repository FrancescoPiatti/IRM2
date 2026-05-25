"""
End-to-end smoke run: simple encoder + simple Neural SDE with general noise.

Expensive — do NOT include in the standard pytest sweep. Run manually:

    cd <repo-root>
    python -m tests.full_test_run.fullrun_SimpleSimple_gen
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

if __name__ == "__main__":

    start = datetime(2020,1,1)
    end = datetime(2020,2,28)

    data_path = "data2"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Yield curve csv not found: {data_path}")
    
    # -------------------------------------------------------------------

    # Initailize Dataloader (only YC enabled in loss function)
    data_cfg = DataLoaderCfg(data_path=data_path,
                             start_date=start,
                             end_date=end,
                             max_maturity=5)
    
    dl = MarketDataLoader(data_cfg)
    
    
    # Initialize Encoder and NSDE
    encoder_cfg = EncoderCfg(mode='simple')  # Default

    nsde_cfg = NSDECfg(type='simple')       
    nsde_cfg.noise_type = 'general'
    nsde_cfg.diffusion = {
        "type": "mlp",
        "n_layers": 2,
        "n_units": [64, 64],
        "dropout": 0.0,
        "activation": "relu",
        "out_activation": "tanh",
    }


    # Initailize ShortRateModel
    model = ShortRateModel(name='TestV1',
                           encoder=encoder_cfg,
                           nsde=nsde_cfg,
                           noise_dim=8, 
                           latent_dim=32)

    # Initialize Trainer
    trainer_cfg = TrainerCfg()
    trainer_cfg.early_stopping.enabled = False
    trainer_cfg.use_amp = False
    trainer_cfg.accumulate_windows = 1

    # Just to test
    trainer_cfg.n_paths = 64
    trainer_cfg.batch_window = 8
    trainer_cfg.window_step = 1

    trainer_cfg.lookback = 16
    trainer_cfg.lookback_freq = 1

    trainer_cfg.run_name = 'Test_V1'
    trainer_cfg.debug = True

    trainer_cfg.scheduler.name = 'step'
    trainer_cfg.scheduler.params = {'step_size':1}

    trainer = Trainer(model=model, dataloader=dl, config=trainer_cfg, resume_from=None)

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

