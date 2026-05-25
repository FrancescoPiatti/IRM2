# src/training/train_utils.py
import math
import warnings
from typing import List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


from ..configs.config_trainer import OptimizerCfg
from ..configs.config_trainer import SchedulerCfg
from ..configs.config_trainer import LossCfg


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    During the first ``warmup_epochs`` epochs the learning rate increases
    linearly from ``eta_min`` to the optimizer's base LR. After warmup it
    follows a cosine decay back to ``eta_min`` over the remaining
    ``max_epochs - warmup_epochs`` epochs.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of warmup epochs (must be < `max_epochs`).
    max_epochs : int
        Total number of training epochs.
    eta_min : float
        Minimum learning rate (used at the end of cosine decay and as the
        start of warmup).
    last_epoch : int
        Index of the last epoch (used to resume).

    Notes
    -----
    This scheduler implements the modern PyTorch contract:

    - ``get_lr`` is the *computed* schedule and is intentionally side-effect
      free.
    - Calling ``get_lr`` outside ``step()`` emits the standard PyTorch
      warning ("To get the last learning rate computed by the scheduler,
      please use `get_last_lr()`.").
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if warmup_epochs >= max_epochs:
            raise ValueError("warmup_epochs must be strictly less than max_epochs.")

        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Emit the standard PyTorch warning if the user calls get_lr() outside
        # the scheduler step.
        if not getattr(self, "_get_lr_called_within_step", False):
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )

        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: eta_min -> base_lr.
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * alpha
                for base_lr in self.base_lrs
            ]
        # Cosine decay: base_lr -> eta_min.
        t = self.last_epoch - self.warmup_epochs
        T = max(1, self.max_epochs - self.warmup_epochs)
        return [
            self.eta_min + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))
            for base_lr in self.base_lrs
        ]


def build_optimizer(model: nn.Module, cfg: OptimizerCfg) -> Optimizer:
    """
    Build optimizer from config.

    Supported:
    - adam
    - adamw
    - sgd
    """
    name = str(cfg.name).lower()
    params = dict(cfg.params)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **params)

    raise ValueError(f"Unsupported optimizer '{cfg.name}'. Try: adam|adamw|sgd")


def build_loss(cfg: LossCfg) -> nn.Module:
    """
    Build loss function from config.

    Supported:
    - mse
    - l1
    - huber
    """
    name = str(cfg.name).lower()
    params = dict(cfg.params)

    if name == "mse":
        return nn.MSELoss(**params)
    if name in ("l1", "mae"):
        return nn.L1Loss(**params)
    if name == "huber":
        return nn.HuberLoss(**params)

    raise ValueError(f"Unsupported loss '{cfg.name}'. Try: mse|l1|huber")


def build_scheduler(optimizer: Optimizer, cfg: SchedulerCfg) -> Optional[_LRScheduler]:
    """
    Build scheduler from config.

    Supported:
    - None
    - step
    - cosine
    - warmup_cosine
    - plateau (ReduceLROnPlateau)
    """
    if cfg.name is None:
        return None

    name = str(cfg.name).lower()
    params = dict(cfg.params)

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)

    if name == "warmup_cosine":
        return WarmupCosineScheduler(optimizer, **params)

    if name == "plateau":
        # note: plateau scheduler needs metric passed to .step(metric)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)

    raise ValueError(f"Unsupported scheduler '{cfg.name}'. Try: step|cosine|warmup_cosine|plateau|None")


