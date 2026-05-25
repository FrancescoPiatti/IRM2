# src/utils/artifacts.py
import os
import math
import json
import pickle
import tempfile
from pathlib import Path
import pandas as pd

from types import SimpleNamespace
from types import MappingProxyType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple
from typing import Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from dataclasses import is_dataclass
from dataclasses import fields

# ============================ Atomic save helpers ============================

def atomic_save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Safely save a Python object with pickle (atomic write).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        pickle.dump(obj, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    os.replace(tmp_path, str(path))


def atomic_save_torch(obj: Any, path: Union[str, Path]) -> None:
    """
    Safely save a PyTorch object with torch.save (atomic write).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        torch.save(obj, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    os.replace(tmp_path, str(path))


def atomic_save_json(obj: Any, path: Union[str, Path]) -> None:
    """
    Safely save a JSON file (atomic write).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        json.dump(obj, tmp, indent=2, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    os.replace(tmp_path, str(path))


def _to_jsonable(x):
    """
    Convert configs/artifacts into objects that are JSON-safe AND pickle-safe.
    (mappingproxy -> dict, dataclasses -> dict, tensors -> python types, etc.)
    """
    # MappingProxyType (freeze_dict output)
    if isinstance(x, MappingProxyType):
        return {k: _to_jsonable(v) for k, v in x.items()}

    # Generic Mapping (dict, etc.)
    if isinstance(x, Mapping):
        return {k: _to_jsonable(v) for k, v in x.items()}

    # Dataclass (EncoderCfg / NSDECfg etc.)
    if is_dataclass(x):
        return {f.name: _to_jsonable(getattr(x, f.name)) for f in fields(x)}

    # Torch tensors
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()

    # Lists / tuples
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # Numpy (if ever sneaks in)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
    except Exception:
        pass

    # datetime
    if hasattr(x, "isoformat"):
        return x.isoformat()

    return x


# ============================ Artifact Manager ============================

class ArtifactManager:
    """
    Handles checkpoints, results, and model artifacts saving/loading.

    Notes
    -----
    - Checkpoints store model + optimizer + scheduler + RNG states.
    - Final model stores ONLY model.state_dict() (recommended).
    - Everything is written atomically to reduce risk of corrupted artifacts.

    Directory structure
    -------------------
    output_dir/
      epoch_losses.pkl
      model_params.pt
      evaluation_results_train.pkl
      evaluation_results_test.pkl
      checkpoints/
          checkpoint_best.pt
          checkpoint_epoch5.pt
          checkpoint_index.json   (manifest)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        logger: Optional[Any] = None,
        ckpt_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.ckpt_cfg = ckpt_cfg or {}

        # Create directory for checkpoints
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint configuration parameters
        self.ckpt_mode = self.ckpt_cfg.get("mode", "min")  # 'min' or 'max'
        self.ckpt_best_only = self.ckpt_cfg.get("save_best_only", True)
        self.ckpt_every_n = self.ckpt_cfg.get("every_n_epochs", 0)
        self.ckpt_max_to_keep = self.ckpt_cfg.get("max_to_keep", 3)

        # Tracking
        self.best_metric = math.inf if self.ckpt_mode == "min" else -math.inf
        self._saved_epochs: List[int] = []

        # Manifest index
        self._index_path = self.ckpt_dir / "checkpoint_index.json"
        self._last_best_tag: Optional[str] = None


    # -------------------- Path helpers --------------------

    def paths(self) -> SimpleNamespace:
        """
        Common artifact paths.
        """
        return SimpleNamespace(
            losses=str(self.output_dir / "epoch_losses.pkl"),
            model_params=str(self.output_dir / "model_params.pt"),
            ckpt_best=str(self.ckpt_dir / "checkpoint_best.pt"),
        )


    # -------------------- Checkpoint helpers --------------------

    def _is_better(self, current: float) -> bool:
        """
        Return True if current metric improves the best according to mode.
        """
        if self.ckpt_mode == "min":
            return current < self.best_metric
        return current > self.best_metric


    def _payload(self, epoch: int, metric: float) -> Dict[str, Any]:
        """
        Compose checkpoint payload with states and RNG.

        Safe even when optimizer/scheduler are None.
        """
        payload = {
            "epoch": int(epoch),
            "metric": float(metric),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            # Periodic-checkpoint bookkeeping (so resume preserves prune state)
            "saved_epochs": list(self._saved_epochs),
            "best_metric": float(self.best_metric),
        }
        return payload


    def _write_index(self) -> None:
        """
        Save a small manifest for browsing checkpoints without torch.load.
        """
        index = {
            "mode": self.ckpt_mode,
            "save_best_only": bool(self.ckpt_best_only),
            "every_n_epochs": int(self.ckpt_every_n),
            "max_to_keep": int(self.ckpt_max_to_keep),
            "best_metric": float(self.best_metric) if self.best_metric not in (math.inf, -math.inf) else None,
            "best_checkpoint": str(self.ckpt_dir / "checkpoint_best.pt") if self._last_best_tag else None,
            "saved_epochs": list(self._saved_epochs),
        }
        atomic_save_json(index, self._index_path)


    # -------------------- On epoch end --------------------


    def on_epoch_end(self, epoch: int, epoch_metric: float) -> None:
        """
        Callback to be called by Trainer at each epoch end.
        """
        # Best checkpoint
        if self.ckpt_best_only and self._is_better(epoch_metric):
            self.best_metric = float(epoch_metric)
            self.save_checkpoint(epoch, epoch_metric, tag="best")
            self._last_best_tag = "best"

        # Periodic checkpoint
        if self.ckpt_every_n > 0 and (epoch % self.ckpt_every_n == 0):
            tag = f"epoch{epoch}"
            self.save_checkpoint(epoch, epoch_metric, tag=tag)
            self._saved_epochs.append(epoch)

            # Remove old periodic checkpoints
            if len(self._saved_epochs) > self.ckpt_max_to_keep:
                old_epoch = self._saved_epochs.pop(0)
                old_path = self.ckpt_dir / f"checkpoint_epoch{old_epoch}.pt"
                if old_path.exists():
                    try:
                        old_path.unlink()
                        if self.logger:
                            self.logger.info(f"Removed old checkpoint: {old_path}")
                    except OSError:
                        pass

        # Write checkpoint index
        self._write_index()


    def save_checkpoint(self, epoch: int, metric: float, tag: str = "epoch") -> None:
        """
        Save a checkpoint atomically.
        """
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        atomic_save_torch(self._payload(epoch, metric), path)

        if self.logger:
            self.logger.info(f"Saved checkpoint: {path}")


    # -------------------- Save results helpers --------------------

    def save_losses(self, losses: List[float]) -> None:
        """
        Save epoch losses to both ``epoch_losses.pkl`` (atomic pickle) and
        ``losses.csv`` (analyser-friendly CSV with columns ``epoch, loss``).
        """
        pkl_path = self.paths().losses
        atomic_save_pickle(losses, pkl_path)

        csv_path = self.output_dir / "losses.csv"
        df = pd.DataFrame(
            {"epoch": list(range(1, len(losses) + 1)), "loss": [float(x) for x in losses]}
        )
        df.to_csv(csv_path, index=False)

        if self.logger:
            self.logger.info(f"Saved losses at {pkl_path} and {csv_path}")


    def save_evaluation(self, results: Dict[str, Any], suffix: str = "") -> None:
        """
        Save evaluation payload atomically.

        suffix examples:
        - '_train'
        - '_test'
        """
        filename = f"evaluation_results{suffix}.pkl"
        path = self.output_dir / filename
        atomic_save_pickle(results, path)

        if self.logger:
            self.logger.info(f"Saved evaluation results at {path}")


    def save_final_model(self) -> None:
        """
        Save plain model state_dict to model_params.pt (atomic).
        """
        atomic_save_torch(self.model.state_dict(), self.paths().model_params)

        if self.logger:
            self.logger.info(f"Model parameters saved at {self.paths().model_params}")


    def save_eval_csv(self, df: pd.DataFrame, *, filename: str) -> str:
        """
        Save evaluation DataFrame as CSV in ``<output_dir>/eval/``.
        """
        if not filename.endswith(".csv"):
            filename = filename + ".csv"

        eval_dir = self.output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        path = eval_dir / filename
        df.to_csv(path, index=False)

        if self.logger is not None:
            self.logger.info(f"Saved eval CSV to: {path}")

        return str(path)


    # -------------------- Loaders --------------------

    def load_checkpoint(self, path: Union[str, Path], device: torch.device) -> int:
        """
        Load a full checkpoint (model/optimizer/scheduler/RNG). Return epoch.

        Notes
        -----
        Uses ``weights_only=False`` because the checkpoint contains optimizer
        / scheduler / RNG state in addition to tensors. Only load files you
        trust.
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state"])

        if self.optimizer is not None and ckpt.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if self.scheduler is not None and ckpt.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

        if ckpt.get("rng_state") is not None:
            torch.set_rng_state(ckpt["rng_state"])

        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])

        # Restore best-metric and periodic-checkpoint bookkeeping if present.
        if "best_metric" in ckpt:
            self.best_metric = float(ckpt["best_metric"])
        else:
            self.best_metric = ckpt.get("metric", self.best_metric)

        if "saved_epochs" in ckpt:
            self._saved_epochs = list(ckpt["saved_epochs"])

        self._write_index()

        return int(ckpt.get("epoch", 0))


# ============================ Convenience loader ============================

def load_model_from_dir(
    model: nn.Module,
    output_dir: Union[str, Path],
    device: torch.device,
    *,
    prefer_best: bool = True,
    fallback_to_params: bool = True,
    strict: bool = True,
) -> Tuple[nn.Module, Optional[int], Optional[str]]:
    """
    Load model from artifacts in `output_dir`.

    Priority:
    - checkpoints/checkpoint_best.pt (if prefer_best)
    - model_params.pt (if fallback_to_params)

    Returns:
        (model, epoch_or_None, used_path_or_None)
    """
    output_dir = Path(output_dir)
    ckpt_best = output_dir / "checkpoints" / "checkpoint_best.pt"
    params = output_dir / "model_params.pt"

    if prefer_best and ckpt_best.exists():
        # Best checkpoint contains optimizer/scheduler/RNG -> weights_only=False
        blob = torch.load(ckpt_best, map_location=device, weights_only=False)
        if isinstance(blob, dict) and "model_state" in blob:
            model.load_state_dict(blob["model_state"], strict=strict)
            return model, int(blob.get("epoch", 0)), str(ckpt_best)

    if fallback_to_params and params.exists():
        # model_params.pt is just a state_dict -> weights_only=True is safe
        state = torch.load(params, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=strict)
        return model, None, str(params)

    raise FileNotFoundError(f"No suitable artifact in: {output_dir}")



class _NullArtifactManager:
    """
    No-op replacement for ArtifactManager when IO is disabled (e.g. Optuna runs).
    Keeps Trainer code unchanged by exposing the same methods.
    """
    def __init__(self, *args, **kwargs): 
        pass

    def on_epoch_end(self, *args, **kwargs):
        return

    def save_losses(self, *args, **kwargs):
        return

    def save_final_model(self, *args, **kwargs):
        return

    def save_eval_csv(self, *args, **kwargs):
        return

    def load_checkpoint(self, *args, **kwargs):
        raise RuntimeError("Checkpoint loading is disabled in optuna_mode.")