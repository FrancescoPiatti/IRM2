# src/configs/config_trainer.py
from dataclasses import dataclass
from dataclasses import field

from typing import Any
from typing import Dict
from typing import Optional
from typing import Literal

from ..utils.checks import _check_positive_integer_value
from ..utils.checks import _check_positive_value


@dataclass
class OptimizerCfg:
    """
    Optimizer configuration.

    Attributes
    ----------
    name : str
        Optimizer name: 'adam', 'adamw', 'sgd', ...
    params : Dict[str, Any]
        Keyword arguments passed to the optimizer constructor.
    """
    name: str = "adamw"
    params: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3, "weight_decay": 1e-4})


@dataclass
class SchedulerCfg:
    """
    Scheduler configuration.

    Attributes
    ----------
    name : Optional[str]
        Scheduler name: 'cosine', 'warmup_cosine', 'step', 'plateau', None.
    params : Dict[str, Any]
        Keyword arguments passed to the scheduler constructor.
    """
    name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossCfg:
    """
    Loss function configuration.

    Attributes
    ----------
    name : str
        Loss name: 'mse', 'l1', 'huber', ...
    params : Dict[str, Any]
        Keyword arguments passed to the loss constructor.
    """
    name: str = "mse"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointCfg:
    """
    Checkpoint configuration (passed to ArtifactManager).

    Attributes
    ----------
    mode : str
        'min' or 'max' depending on monitored metric.
    save_best_only : bool
        Save only best checkpoint.
    every_n_epochs : int
        Periodic checkpoint saving frequency.
    max_to_keep : int
        Keep at most this many periodic checkpoints.
    """
    mode: Literal["min", "max"] = "min"
    save_best_only: bool = True
    every_n_epochs: int = 30
    max_to_keep: int = 3


@dataclass
class LossWeightsCfg:
    """
    Per-target weights applied inside ``Trainer._get_loss``.

    The joint training objective is

    ::

        L = λ_y · L_yield + λ_sr · L_short_rate + λ_f · L_fut

    By default every weight is 1.0, so existing runs keep the unweighted
    behaviour. For joint YC + futures training the futures loss runs
    several orders of magnitude larger than the yield loss in the early
    epochs (market futures sit near $120, BondNet at init outputs ~1);
    dropping ``λ_f`` (e.g. 0.01) lets the yield curve fit first instead
    of the gradient being dominated by the BondNet calibration error.

    Attributes
    ----------
    yield_curve : float
        Multiplier on the yield-curve MSE component.
    short_rate : float
        Multiplier on the short-rate MSE component.
    futures : float
        Multiplier on the futures MSE component.
    """
    yield_curve: float = 1.0
    short_rate: float = 1.0
    futures: float = 1.0


@dataclass
class EarlyStoppingCfg:
    """
    Early stopping configuration.

    Notes
    -----
    This implements early stopping WITHOUT a validation set by monitoring
    the training loss (optionally smoothed via EMA).
    
    Attributes
    ----------
    enabled : bool
        Whether early stopping is active.
    patience : int
        Number of epochs with no improvement after which training stops.
    min_delta : float
        Minimum decrease in the monitored metric to qualify as an improvement.
    use_ema : bool
        If True, monitor an EMA-smoothed version of the training loss.
    ema_alpha : float
        EMA smoothing coefficient in [0, 1]. Higher = more reactive.
    """
    enabled: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    use_ema: bool = True
    ema_alpha: float = 0.2


@dataclass
class TrainerCfg:
    """
    Training configuration for sequential-window training.

    Core parameters
    ---------------
    n_paths : int
        Number of Monte Carlo paths used for pricing simulation.
    batch_window : int
        Window size in days. Backprop happens once per window.

    Encoder lookback configuration
    ------------------------------
    - simple encoder uses (lookback, lookback_freq)
    - hierarchical encoder uses (lookback_fast, lookback_fast_freq, lookback_slow, lookback_slow_freq)

    Optimization
    ------------
    - optimizer / scheduler / loss configs

    Stability + performance
    -----------------------
    use_amp : bool
        Mixed precision (CUDA only).
    grad_clip_norm : Optional[float]
        Clip gradient norm if not None.
    accumulate_windows : int
        Gradient accumulation over multiple windows before stepping.
    """

    # Sequential training structure
    n_paths: int = 500
    batch_window: int = 30
    window_step: int = 2

    # Trainer's dt is the timestep at which the latent repr are output
    # Not to be confused with nsde's dt which is the solver dt
    dt: Optional[float] = None

    # Simple encoder lookback
    lookback: int = 252
    lookback_freq: int = 1

    # Hierarchical encoder lookback
    lookback_fast: int = 63
    lookback_fast_freq: int = 1
    lookback_slow: int = 252
    lookback_slow_freq: int = 5

    # Optimization configs
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    loss_weights: LossWeightsCfg = field(default_factory=LossWeightsCfg)

    # Runtime options
    use_amp: bool = False
    grad_clip_norm: Optional[float] = 1.0
    accumulate_windows: int = 1
    # If True (and the device is CUDA), the trainer wraps drift/diffusion
    # sub-networks with `torch.compile(mode='reduce-overhead')`. Off by default
    # because some torch/Triton/Inductor combos blow up at first forward.
    compile_nsde: bool = False

    # Reproducibility
    seed: Optional[int] = 0
    deterministic: bool = False

    # Logging + artifacts
    results_root: str = "results"
    run_name: Optional[str] = None
    debug: bool = False

    # Checkpoints and EarlyStopping
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    early_stopping: EarlyStoppingCfg = field(default_factory=EarlyStoppingCfg)

    # Loss shaping
    # ------------
    # When True, the futures term is computed as a RELATIVE error,
    # ``mean(((model - market) / market)**2)``, instead of an absolute MSE
    # on prices. Futures sit near $120, so an absolute MSE lives on a
    # price² scale (~10²-10³) while the yield MSE is on a decimal² scale
    # (~1e-5); even a tiny ``loss_weights.futures`` can't put them on a
    # comparable footing. The relative error is dimensionless and O(1e-4)
    # for a 1 % pricing error, so ``loss_weights`` become interpretable
    # and ``λ_y = λ_f = 1`` actually balances yields against futures.
    # Yields/short-rate stay ABSOLUTE (they're already on a natural [0,~0.06]
    # decimal scale, and market yields can be ~0 → relative would blow up).
    futures_relative_loss: bool = False

    # Weight of the BondNet <-> SDE consistency loss (LSMC). When > 0, the
    # pricer regresses BondNet's deliverable-bond prices onto the model's
    # OWN pathwise-discounted cashflows (computed on the same simulated
    # short-rate paths — no nested simulation). This is what ties the
    # futures channel to the yield-curve dynamics: without it, BondNet is a
    # free head and "joint calibration" degenerates into two unrelated
    # tasks sharing an encoder. The loss is normalised by 100² so it lives
    # on the same dimensionless scale as the relative futures loss; 1.0 is
    # a sensible starting weight. Gradients flow into BOTH BondNet and the
    # SDE (bidirectional coupling) by design.
    bondnet_consistency_weight: float = 0.0

    # Safety options
    skip_nan_loss: bool = True
    log_every_n_windows: int = 1


    # -------------------------
    # Validation
    # -------------------------

    def validate(self) -> None:
        """
        Validate trainer configuration values.

        Call after all fields are set. Checks that integer fields are positive,
        float fields are positive where required, and early stopping params are valid.
        """
        _check_positive_integer_value(self.n_paths, 'n_paths')
        _check_positive_integer_value(self.batch_window, 'batch_window')
        _check_positive_integer_value(self.window_step, 'window_step')
        _check_positive_integer_value(self.accumulate_windows, 'accumulate_windows')
        _check_positive_integer_value(self.log_every_n_windows, 'log_every_n_windows')

        if self.dt is not None:
            _check_positive_value(self.dt, 'dt')
        if self.grad_clip_norm is not None:
            _check_positive_value(self.grad_clip_norm, 'grad_clip_norm')

        # Lookback checks (all are validated; Trainer selects which to use)
        _check_positive_integer_value(self.lookback, 'lookback')
        _check_positive_integer_value(self.lookback_freq, 'lookback_freq')
        _check_positive_integer_value(self.lookback_fast, 'lookback_fast')
        _check_positive_integer_value(self.lookback_slow, 'lookback_slow')
        _check_positive_integer_value(self.lookback_fast_freq, 'lookback_fast_freq')
        _check_positive_integer_value(self.lookback_slow_freq, 'lookback_slow_freq')

        # Loss-weights sub-config — each weight must be non-negative
        # (zero disables the corresponding target, which is the
        # idiomatic way to "warm up on yields only").
        for name in ("yield_curve", "short_rate", "futures"):
            w = float(getattr(self.loss_weights, name))
            if w < 0.0:
                raise ValueError(f"loss_weights.{name} must be >= 0; got {w}.")

        if float(self.bondnet_consistency_weight) < 0.0:
            raise ValueError(
                f"bondnet_consistency_weight must be >= 0; got {self.bondnet_consistency_weight}."
            )

        # Early stopping sub-config
        if self.early_stopping.enabled:
            _check_positive_integer_value(self.early_stopping.patience, 'early_stopping.patience')
            _check_positive_value(self.early_stopping.min_delta, 'early_stopping.min_delta')
            if not (0.0 <= self.early_stopping.ema_alpha <= 1.0):
                raise ValueError("early_stopping.ema_alpha must be in [0, 1].")