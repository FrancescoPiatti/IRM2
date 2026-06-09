# src/training/trainer.py
import os
import logging
from datetime import datetime
import pandas as pd

from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import List
from typing import Tuple

import torch
from torch import Tensor
from torch.amp import autocast
from torch.amp import GradScaler

from .train_utils import build_optimizer
from .train_utils import build_loss
from .train_utils import build_scheduler

from ..dataloaders.market_loader import MarketDataLoader

from ..finance.pricer_v2 import Pricer
from ..models.short_rate_model import ShortRateModel

from ..utils.logger import SimpleLogger
from ..utils.artifacts import ArtifactManager
from ..utils.artifacts import _NullArtifactManager
from ..utils.checks import _check_positive_integer_value

from ..configs.config_trainer import TrainerCfg

from ..types.types_utils import Date
from ..types.data_types import MarketSnapshot
from ..types.data_types import EncoderInputs
from ..types.eval_results_types import EvalResults
from ..types.eval_results_types import eval_results_to_frame


# Optional Optuna support (no hard dependency)
try:
    import optuna  # type: ignore
    _OPTUNA_AVAILABLE = True
except Exception:
    optuna = None  # type: ignore
    _OPTUNA_AVAILABLE = False


class Trainer:
    """
    Sequential-window trainer for the short-rate model.

    Backpropagation is performed once per window (e.g. every 3 months).
    The encoder receives a lookback of data and the model is trained to
    price the last curve fed into the encoder.

    Parameters
    ----------
    model : ShortRateModel
        The model to be trained.
    dataloader : MarketDataLoader
        Wrapper dataloader (canonical calendar = yield curve dates).
    config : Optional[TrainerCfg]
        Trainer configuration.
    device : Optional[str or torch.device]
        Device to run on. If None: uses 'cuda' if available else 'cpu'.
    resume_from : Optional[str]
        Path to a checkpoint to resume training.
    optuna_trial : Optional[Any]
        Optuna trial object. Deactivates IO and enables pruning.
    """


    def __init__(
        self,
        model: ShortRateModel,
        dataloader: MarketDataLoader,
        config: Optional[TrainerCfg] = None,
        device: Optional[Union[str, torch.device]] = None,
        resume_from: Optional[str] = None,
        optuna_trial: Optional[Any] = None
    ):
        
        self.cfg = config or TrainerCfg()
        self.cfg.validate()

        # ------------------------------------------
        # Optuna (are we in gridsearch setting?)
        # ------------------------------------------
        # Optuna mode is ON when trial is provided (and optuna is installed)
        self._optuna = bool(optuna_trial is not None)

        if self._optuna and not _OPTUNA_AVAILABLE:
            raise ImportError("optuna_trial was provided but optuna is not installed.")
        
        self._optuna_trial = optuna_trial

        # ------------------------------------------
        # Device setup
        # ------------------------------------------
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(str(device))

        # ------------------------------------------
        # Reproducibility
        # ------------------------------------------
        if self.cfg.seed is not None:
            torch.manual_seed(int(self.cfg.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.cfg.seed))

        if self.cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        # ------------------------------------------
        # Model and Dataloader
        # ------------------------------------------
        self.model = model.to(self.device)
        self.dataloader = dataloader


        # ------------------------------------------
        # Optimizer, Loss, Scheduler
        # ------------------------------------------
        # Loss will potentially change here
        self.loss_fn = build_loss(self.cfg.loss)
        self.optimizer = build_optimizer(self.model, self.cfg.optimizer)
        self.scheduler = build_scheduler(self.optimizer, self.cfg.scheduler)


        # ------------------------------------------
        # AMP setup
        # ------------------------------------------
        self.use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        # device_type must be explicit with torch.amp
        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = GradScaler(device=scaler_device, enabled=self.use_amp)


        # ------------------------------------------
        # Training parameters
        # ------------------------------------------
        self.n_paths = int(self.cfg.n_paths)
        self.batch_window = int(self.cfg.batch_window)
        self.window_step = int(self.cfg.window_step)
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.accumulate_windows = max(1, int(self.cfg.accumulate_windows))

        if self.cfg.dt is None:
            self.dt = self.model.nsde.dt
        else:
            self.dt = float(self.cfg.dt)

        # ------------------------------------------
        # Pricing Engine
        # ------------------------------------------
        # Match steps_per_year to the simulation grid spacing and propagate the
        # year-fraction convention from the dataloader. The pricer autocasts
        # only the BondNet forward when AMP is enabled.
        self.pricer = Pricer(
            device=self.device,
            steps_per_year=int(round(1.0 / float(self.dt))),
            business_days_per_year=float(getattr(self.dataloader, "business_days_per_year", 252.0)),
            use_amp=self.use_amp,
        )


        # ------------------------------------------
        # Lookback settings
        # ------------------------------------------
        # The dataloader is encoder-agnostic.
        # The trainer controls how many histories to request based on encoder_type.
        if self.model.encoder_type == "simple":
            self.lookback = int(self.cfg.lookback)
            self.lookback_freq = int(self.cfg.lookback_freq)

            self.lookback_fast = None
            self.lookback_slow = None
            self.lookback_fast_freq = None
            self.lookback_slow_freq = None

        elif self.model.encoder_type == "hierarchical":
            self.lookback = None
            self.lookback_freq = None

            self.lookback_fast = int(self.cfg.lookback_fast)
            self.lookback_slow = int(self.cfg.lookback_slow)
            self.lookback_fast_freq = int(self.cfg.lookback_fast_freq)
            self.lookback_slow_freq = int(self.cfg.lookback_slow_freq)

        else:
            raise ValueError(f"Unknown encoder_type={self.model.encoder_type}")
        

        # ------------------------------------------
        # Output directory
        # ------------------------------------------
        run_tag = self.cfg.run_name or self.model.name

        if self._optuna:
            # shared folder created by gridsearch (no timestamp here)
            self.output_dir = os.path.join(self.cfg.results_root, run_tag)
            # do NOT mkdir here; gridsearch owns folder creation & collision policy
        else:
            timestamp = datetime.now().strftime("%d%b_%H%M")
            self.output_dir = os.path.join(self.cfg.results_root, f"{timestamp}_{run_tag}")
            os.makedirs(self.output_dir, exist_ok=True)


        # ------------------------------------------
        # Logger setup
        # ------------------------------------------
        lev = logging.DEBUG if self.cfg.debug else logging.INFO
        
        self.logger = SimpleLogger(
            name=run_tag,
            log_dir=self.output_dir,
            level=lev,
            unique=not self._optuna,   # gridsearch: same logger name every trial
            add_console=self._optuna,  # optional: see logs while grid runs
        ).get_logger()

        self.logger.info(f"Trainer initialized on device={self.device} use_amp={self.use_amp}")
        self.log_every_n_windows = int(self.cfg.log_every_n_windows)

        # ------------------------------------------
        # Artifact manager
        # ------------------------------------------
        if self._optuna:
            self.artifacts = _NullArtifactManager()    
        else:            
            self.artifacts = ArtifactManager(
                output_dir=self.output_dir,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                logger=self.logger,
                ckpt_cfg={
                    "mode": self.cfg.checkpoint.mode,
                    "save_best_only": self.cfg.checkpoint.save_best_only,
                    "every_n_epochs": self.cfg.checkpoint.every_n_epochs,
                    "max_to_keep": self.cfg.checkpoint.max_to_keep,
                },
            )

        # ------------------------------------------
        # Early stopping 
        # ------------------------------------------
        # We stop if the TRAINING loss plateaus for many epochs.
        self.early_stopping_enabled = bool(self.cfg.early_stopping.enabled)
        self.es_patience = int(self.cfg.early_stopping.patience)
        self.es_min_delta = float(self.cfg.early_stopping.min_delta)
        self.es_use_ema = bool(self.cfg.early_stopping.use_ema)
        self.es_ema_alpha = float(self.cfg.early_stopping.ema_alpha)


        # ------------------------------------------
        # Dataloader enable_* check
        # ------------------------------------------
        dl = self.dataloader
        if not any([
            dl.enable_yield, 
            dl.enable_short_rate,
            dl.enable_bonds,
            dl.enable_futures,
            dl.enable_options
            ]):
            raise ValueError(
                "Trainer requires at least one target to be enabled in the dataloader "
                "(enable_yield, enable_short_rate, enable_bonds, enable_futures, enable_options)."
            )


        # ------------------------------------------
        # torch.compile drift/diffusion networks (CUDA only, opt-in)
        # ------------------------------------------
        if bool(getattr(self.cfg, "compile_nsde", False)) and self.device.type == "cuda":
            self._maybe_compile_nsde()


        # ------------------------------------------
        # Resume from checkpoint
        # ------------------------------------------
        self.start_epoch = 1
        if resume_from is not None:
            if self._optuna:
                raise RuntimeError("resume_from is not allowed in optuna_mode (IO disabled).")
            epoch = self.artifacts.load_checkpoint(resume_from, device=self.device)
            self.start_epoch = int(epoch) + 1
            self.logger.info(f"Resumed from checkpoint={resume_from} start_epoch={self.start_epoch}")


    # ------------------------------------------------------------------
    # Optional torch.compile
    # ------------------------------------------------------------------

    def _maybe_compile_nsde(self) -> None:
        """
        Wrap NSDE drift/diffusion sub-networks with ``torch.compile``.

        Each compile call is guarded by try/except so a failure in one
        sub-network does not poison the whole training run — we log and
        proceed with the un-compiled module.
        """
        nsde = self.model.nsde
        for attr in ("drift", "diffusion", "long_term_mean", "mean_reversion"):
            mod = getattr(nsde, attr, None)
            if mod is None:
                continue
            try:
                setattr(nsde, attr, torch.compile(mod, mode="reduce-overhead"))
                self.logger.info(f"torch.compile applied to nsde.{attr}")
            except Exception as exc:
                self.logger.warning(
                    f"torch.compile failed for nsde.{attr}: {exc!r} — using uncompiled module."
                )


    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_history(self, date: Date) -> Union[EncoderInputs, Tuple[EncoderInputs, EncoderInputs]]:
        """
        Fetch encoder history based on the model encoder topology.

        Returns
        -------
        EncoderInputs or Tuple[EncoderInputs, EncoderInputs]

        Notes
        -----
        The tensors inside the same EncoderInputs instance have the same shape 
        but `lookback` may be smaller for initial windows.
        """
        if self.model.encoder_type == 'simple':

            # Fetch past data for the encoder
            # This will return an instance of EncoderInputs: 
            # tensors inside past_data are of shape (lookback, n_features)
            past_data = self.dataloader.get_history(
                date,
                lookback_days=self.lookback,
                frequency=self.lookback_freq,
                return_dates=False,
                device=self.device
            )
            return past_data
        
        elif self.model.encoder_type == 'hierarchical':

            # Fetch past data for the fast encoder
            # This will return a Tuple of EncoderInputs: (fast_data, slow_data)
            # where fast_data contains tensors of shape (lookback_fast, n_features_fast)
            # and slow_data contains tensors of shape (lookback_slow, n_features_slow)
            past_data_fast = self.dataloader.get_history(
                date,
                lookback_days=self.lookback_fast,
                frequency=self.lookback_fast_freq,
                return_dates=False,
                device=self.device
                )
            past_data_slow = self.dataloader.get_history(
                date, 
                lookback_days=self.lookback_slow,
                frequency=self.lookback_slow_freq,
                return_dates=False,
                device=self.device
                )

            return (past_data_fast, past_data_slow)

        raise ValueError(f"Unknown encoder_type={self.model.encoder_type}")
    

    def _get_r0(self, date: Date) -> Tensor: 
        """
        Short-rate anchor used as the initial condition for simulation.
        """
        return self.dataloader.short_rate_store.get_rate(date=date, device=self.device)
    

    def _get_snapshot(self, date: Date) -> MarketSnapshot:
        """
        Fetch the market snapshot at the given date.
        """
        return self.dataloader.get_snapshot(date, device=self.device)


    def _make_ts(self, snapshot: MarketSnapshot) -> Tensor:
        """
        Return the simulation time grid (year-fractions) for the current snapshot.

        The grid spans ``[0, max_maturity]`` in steps of ``self.dt`` and only
        depends on the trainer's `dt` and the dataloader's `max_maturity`, so
        it is constructed once and cached on the Trainer instance.
        """
        ts = getattr(self, "_ts_cache", None)
        if ts is None or ts.device != self.device:
            hor = float(self.dataloader.max_maturity)
            ts = torch.arange(0.0, hor + self.dt, self.dt, device=self.device)
            self._ts_cache = ts
        return ts
    

    # ------------------------------------------------------------------
    # Shortcut Helpers
    # ------------------------------------------------------------------

    def get_latent_representation_from_date(
            self, 
            date: Date, 
            n_paths: Optional[int] = None, 
            ts: Optional[Tensor] = None
        ) -> Tensor:
        """
        Encode history at `date` and simulate latent paths.

        Parameters
        ----------
        date : Date
            Anchor date used for history and market snapshot conventions.
        n_paths : Optional[int]
            Override the trainer-level Monte Carlo count.
        """
        _n_paths = n_paths if n_paths is not None else self.n_paths

        # Fetch encoder data for the current date
        past_data = self._get_history(date)

        # Encoder pass
        latent_repr = self.model.encode(past_data)

        # Resolve time grid
        # Either use input ts or just sample at NSDE dt)
        if ts is None:
            horizon = self.dataloader.max_maturity
        else:
            horizon = None

        # Simulate short-rate paths (decoded)
        realisations = self.model.simulate(
            latent_repr, 
            n_paths=_n_paths, 
            horizon=horizon, 
            ts=ts,
            decode=False,
            )
        
        return realisations
    

    def _simulate_window(
        self,
        window_latents: Tensor,
        n_paths: int,
        ts: Tensor,
    ) -> Tensor:
        """
        Run the NSDE once for every (date, path) pair in the window.

        Stacks B initial latents into a single ``(B*N, d_z)`` tensor and
        invokes the NSDE forward in a single solver call (optimisation §2.2).
        Independent of which solver backend is in use — torchsde and the
        in-house Euler both handle a wider initial-state batch identically.

        Parameters
        ----------
        window_latents : Tensor
            Per-date initial latent states, shape ``(B, d_z)``.
        n_paths : int
            Monte Carlo paths per date.
        ts : Tensor
            Time grid, shape ``(T,)``.

        Returns
        -------
        Tensor
            Latent paths, shape ``(B, n_paths, T, d_z)``.
        """
        B = window_latents.size(0)
        z0_stack = window_latents.repeat_interleave(n_paths, dim=0)         # (B*N, d_z)
        all_paths = self.model.nsde(ts, z0_stack, n_paths=B * n_paths)      # (B*N, T, d_z)
        return all_paths.view(B, n_paths, all_paths.size(1), -1)


    def _encode_window(self, batch_dates: List[Date]) -> Optional[Tensor]:
        """
        Encode every date in ``batch_dates`` in a single batched forward.

        Only implemented for the ``simple`` encoder topology — for the
        hierarchical topology we fall back to the per-date encode path used
        by `get_latent_representation_from_date`.

        Returns
        -------
        Optional[Tensor]
            Latent states of shape ``(len(batch_dates), latent_dim)`` for the
            simple encoder, or ``None`` to signal "use the per-date path".
        """
        if self.model.encoder_type != "simple":
            return None

        # Pre-stacked history tensor (B, T, M+1)
        stacked = self.dataloader.get_histories(
            batch_dates,
            lookback_days=self.lookback,
            frequency=self.lookback_freq,
            return_short_rate=True,
            device=self.device,
        )
        # _preprocess_encoder_input sees short_rate=None and forwards the
        # pre-stacked tensor as-is.
        inputs = EncoderInputs(curve_history=stacked, short_rate=None, dates=None)
        return self.model.encode(inputs)                          # (B, latent_dim)


    def _decode(self, latent_repr: Tensor, r0: Tensor) -> Tensor:
        """
        Decode latent representation into short-rate paths.

        This is a simple wrapper around model.decode, which may be useful if we want to 
        add extra logic here later (e.g. for bondnet integration).
        """
        return self.model.decode(latent_repr, r0=r0)
    
    
    def get_realisations_from_date(
            self, 
            date: Date, 
            n_paths: Optional[int] = None, 
            ts: Optional[Tensor] = None
        ) -> Tensor:
        """
        Encode history at `date` and simulate decoded short-rate paths.

        Parameters
        ----------
        date : Date
            Anchor date used for history and market snapshot conventions.
        n_paths : Optional[int]
            Override the trainer-level Monte Carlo count.
        """
        latent_repr = self.get_latent_representation_from_date(
            date, 
            n_paths=n_paths, 
            ts=ts
            )
        
        # Fetch short-rate anchor (initial condition for simulation)
        r0 = self._get_r0(date)

        # Decode
        realisations = self._decode(latent_repr, r0)

        return realisations


    # ------------------------------------------------------------------
    # One-step train logic
    # ------------------------------------------------------------------ 
        
    def _get_loss(
            self,
            realisation: Tensor,
            snapshot: MarketSnapshot,
            latent_repr: Tensor,
            ts: Tensor,
            ) -> Tuple[Tensor, Dict]:
        """
        Compose the per-date training objective from enabled market targets.

        Parameters
        ----------
        realisation : Tensor
            Decoded short-rate paths, shape (n_paths, n_steps) or (n_paths, n_steps, 1).
        snapshot : MarketSnapshot
            Observed snapshot used as the loss template.
        latent_repr : Tensor
            Latent paths, shape (n_paths, n_steps, d_z).
        ts : Tensor
            Simulation time grid (year-fractions), shape (n_steps,).
        """
        day_loss = torch.zeros((), device=self.device, dtype=realisation.dtype)
        loss_components: Dict[str, float] = {}

        _bondnet = self.model.bondnet if self.model.bondnet is not None else None

        model_snapshot = self.pricer.price_snapshot(
            realisations=realisation,
            snapshot=snapshot,
            latent_paths=latent_repr,
            simulated_times=ts,
            bondnet=_bondnet,
        )

        # Per-target weights (math_review.md §1 + optimization_plan.md §10.1 P1).
        # We log both the RAW component (for monitoring the underlying fit
        # quality on its own scale) AND the WEIGHTED contribution to the
        # actual training loss.
        lw_y  = float(self.cfg.loss_weights.yield_curve)
        lw_sr = float(self.cfg.loss_weights.short_rate)
        lw_f  = float(self.cfg.loss_weights.futures)

        # -------------------------------------------------------
        # Yield curve target (canonical)
        if snapshot.yield_curve is not None and lw_y > 0.0:
            if model_snapshot.yield_curve is None:
                raise RuntimeError("price_snapshot returned yield_curve=None but snapshot.yield_curve is not None.")

            observed_yields = snapshot.yield_curve.yields
            model_yields = model_snapshot.yield_curve.yields

            yield_loss_raw = self.loss_fn(model_yields, observed_yields)
            loss_components["yield"] = float(yield_loss_raw.detach().cpu().item())
            day_loss = day_loss + lw_y * yield_loss_raw

        # -------------------------------------------------------
        # Short rate target (optional)
        if snapshot.short_rate is not None and lw_sr > 0.0:
            if model_snapshot.short_rate is None:
                raise RuntimeError("price_snapshot returned short_rate=None but snapshot.short_rate is not None.")

            observed_r = snapshot.short_rate.rate
            model_r = model_snapshot.short_rate.rate

            sr_loss_raw = self.loss_fn(model_r, observed_r)
            loss_components["short_rate"] = float(sr_loss_raw.detach().cpu().item())
            day_loss = day_loss + lw_sr * sr_loss_raw

        # -------------------------------------------------------
        # Futures target (optional)
        if snapshot.futures is not None and lw_f > 0.0:
            if model_snapshot.futures is None:
                raise RuntimeError("price_snapshot returned futures=None but snapshot.futures is not None.")

            observed_futures = snapshot.futures.prices
            model_futures = model_snapshot.futures.prices

            fut_loss_raw = self.loss_fn(model_futures, observed_futures)
            loss_components["futures"] = float(fut_loss_raw.detach().cpu().item())
            day_loss = day_loss + lw_f * fut_loss_raw

        # -------------------------------------------------------
        # Bonds / Options (later)
        # if snapshot.bonds is not None: ...
        # if snapshot.options is not None: ...

        return day_loss, loss_components


    def _forward_one_date(self,
                          date: Date,
                          return_components: bool = False,
                          n_paths: Optional[int] = None,
                          *,
                          encoded_state: Optional[Tensor] = None,
                          precomputed_latent_repr: Optional[Tensor] = None,
                          ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        """
        Forward pass at `date`: simulate -> price -> loss.

        Parameters
        ----------
        date : Date
            Anchor date.
        return_components : bool
            Return the per-target loss breakdown alongside the scalar.
        n_paths : Optional[int]
            Override Monte Carlo path count.
        encoded_state : Optional[Tensor]
            If provided, use this latent state ``(latent_dim,)`` as the
            initial condition for the NSDE — skips the encoder forward.
            Used by `_train_one_window` to amortise the encoder across the
            window (optimisation_plan §6.3).
        precomputed_latent_repr : Optional[Tensor]
            If provided, use this latent path ``(n_paths, n_steps, d_z)``
            directly — skips both the encoder and the NSDE simulate.
            Used by `_train_one_window` to amortise the simulate call
            across the window (optimisation_plan §2.2).
        """
        _n_paths = n_paths if n_paths is not None else self.n_paths

        # Get snapshot at current date
        current_snapshot = self._get_snapshot(date)

        # Extract timegrid
        ts = self._make_ts(snapshot=current_snapshot)

        # NN forwards run under autocast when AMP is enabled; pricing arithmetic
        # stays in float32 (project_description §15).
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            if precomputed_latent_repr is not None:
                latent_repr = precomputed_latent_repr
            elif encoded_state is None:
                latent_repr = self.get_latent_representation_from_date(
                    date, n_paths=_n_paths, ts=ts,
                )
            else:
                latent_repr = self.model.simulate(
                    encoded_state,
                    n_paths=_n_paths,
                    horizon=None,
                    ts=ts,
                    decode=False,
                )
            r0 = self._get_r0(date)
            realisations = self._decode(latent_repr, r0=r0)

        # Promote autocast outputs back to float32 before the pricer / loss
        # so the discount factor, CTD min, and final MSE all run in fp32
        # — what project_description §15 calls for, and what `GradScaler`
        # needs to see (otherwise ``scaler.scale(half_loss).backward()``
        # raises "Found type Float but expected Half").
        if self.use_amp:
            realisations = realisations.float()
            latent_repr  = latent_repr.float()

        # Compute loss via pricer + trainer loss composition (float32).
        day_loss, components = self._get_loss(
            realisation=realisations,
            snapshot=current_snapshot,
            latent_repr=latent_repr,
            ts=ts,
        )

        return (day_loss, components) if return_components else day_loss


    def _train_one_window(self, batch_dates: List[Date]) -> Tuple[float, Tensor, int]:
        """
        Aggregate losses across a window of dates (no optimizer step here).

        Returns
        -------
        window_loss_float : float
            Detached scalar used for logging.
        window_loss_tensor : Tensor
            Scalar tensor used for backprop (may be non-differentiable if all days are skipped).
        non_nan_losses : int
            Number of dates that contributed to the window loss.
        """
        window_loss = torch.zeros((), device=self.device)
        non_nan_losses = 0

        # Batched encoder: encode every date in the window in one shot, so the
        # encoder cost is paid once per window instead of once per date
        # (optimisation_plan §6.3). For hierarchical encoders this returns
        # None and we fall back to per-date encoding.
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            window_latents = self._encode_window(list(batch_dates))

        # Batched NSDE simulate across the window (optimisation_plan §2.2).
        # We amortise one solver call across the B dates rather than running
        # B sequential calls. Requires that all dates share the same time
        # grid — which they do, since `_make_ts` returns a cached grid that
        # depends only on `max_maturity` and `dt`.
        window_paths: Optional[Tensor] = None
        if window_latents is not None:
            ts = self._make_ts(snapshot=None)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                window_paths = self._simulate_window(
                    window_latents=window_latents,
                    n_paths=self.n_paths,
                    ts=ts,
                )                                          # (B, N, T, d_z)

        # Forward pass over dates in this window.
        # Autocast is applied *inside* _forward_one_date around the NN forwards
        # only — the pricing arithmetic (CTD min, CF division, MC averages,
        # losses) stays in float32 per project_description §15.
        for i, date in enumerate(batch_dates):
            if window_paths is not None:
                day_loss, components = self._forward_one_date(
                    date,
                    precomputed_latent_repr=window_paths[i],
                    return_components=True,
                )
            else:
                day_loss, components = self._forward_one_date(
                    date,
                    return_components=True,
                )

            # Skip day if NaN/Inf loss. We include the per-target
            # components in the log line so the user can tell which
            # branch produced the NaN (yield / short_rate / futures)
            # without having to re-run with extra instrumentation.
            if self.cfg.skip_nan_loss and (torch.isnan(day_loss) or torch.isinf(day_loss)):
                self.logger.warning(
                    f"Skipping NaN/Inf loss at date={date} components={components}"
                )
                continue

            window_loss = window_loss + day_loss
            non_nan_losses += 1

        if non_nan_losses == 0:
            # Well-defined tensor return, but unusable for backprop.
            window_loss_float = float("nan")
            return window_loss_float, window_loss, non_nan_losses

        # Normalize by number of dates for stability 
        window_loss = window_loss / float(non_nan_losses)

        window_loss_float = float(window_loss.detach().cpu().item())
        return window_loss_float, window_loss, non_nan_losses
    

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    
    def _optimizer_step(self) -> bool:
        """
        Apply one optimizer update (with optional AMP + grad clipping).

        Includes a defensive NaN/Inf guard: if any gradient is non-finite
        after the (optional) AMP unscale, the step is *skipped* — the
        gradients are zeroed and the AMP scaler is told the step failed
        (so it can shrink the loss scale next round). This prevents a
        single bad backward from polluting the Adam moments and turning
        every subsequent window into NaN.

        Returns
        -------
        bool
            ``True`` if a real optimizer step was applied, ``False`` if it
            was skipped because of a non-finite gradient. Callers use this
            to report a per-epoch ``applied/total`` ratio.
        """
        # (AMP) unscale grads so clipping operates in true scale.
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        # Defensive NaN/Inf guard. Under AMP, ``scaler.step`` already
        # checks for non-finite grads internally and is a no-op when it
        # finds them — but we still want the same behaviour in the
        # no-AMP path (and an explicit log line either way so users can
        # see when it fires). Scanning is O(params) and the cost is
        # negligible compared with a single backward pass.
        bad = False
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad = True
                break

        if bad:
            # Drop the polluted gradients and tell the AMP scaler the
            # step failed (so its loss scale halves) — but do NOT touch
            # the Adam moments, which are still clean from before. Logged
            # at DEBUG to avoid spam; the per-epoch summary in ``train``
            # reports the applied/total ratio at INFO/WARNING instead.
            self.logger.debug(
                "Skipping optimizer step: non-finite gradient detected."
            )
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                # ``scaler.update`` shrinks the loss scale when no
                # successful step was reported this round.
                self.scaler.update()
            return False

        # (Optional) clip global grad norm for stability.
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))

        # Optimizer step (AMP-safe if enabled).
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Zero gradients (set_to_none=True for performance).
        self.optimizer.zero_grad(set_to_none=True)
        return True


    def _early_stopping_update(
        self,
        epoch_avg: float,
        best_score: float,
        bad_epochs: int,
        ema_loss: Optional[float],
    ) -> Tuple[bool, float, int, Optional[float]]:
        """
        Update early-stopping state using training loss (optionally EMA-smoothed).

        Returns
        -------
        stop : bool
            True if patience is exhausted.
        best_score : float
            Updated best score.
        bad_epochs : int
            Updated bad epoch counter.
        ema_loss : Optional[float]
            Updated EMA (if enabled).
        """
        if not self.early_stopping_enabled:
            return False, best_score, bad_epochs, ema_loss

        if self.es_use_ema:
            if ema_loss is None:
                ema_loss = epoch_avg
            else:
                ema_loss = self.es_ema_alpha * epoch_avg + (1.0 - self.es_ema_alpha) * ema_loss
            score = ema_loss
            score_name = "ema_train_loss"
        else:
            score = epoch_avg
            score_name = "train_loss"

        improved = (best_score - score) > self.es_min_delta

        if improved:
            best_score = score
            bad_epochs = 0
            self.logger.info(f"EarlyStop: improved {score_name} -> {best_score:.6f}")
            return False, best_score, bad_epochs, ema_loss

        bad_epochs += 1
        self.logger.info(
            f"EarlyStop: no improvement ({score_name}={score:.6f}, best={best_score:.6f}) "
            f"bad_epochs={bad_epochs}/{self.es_patience}"
        )

        stop = bad_epochs >= self.es_patience
        if stop:
            self.logger.info("EarlyStop: patience reached -> stopping training.")
        return stop, best_score, bad_epochs, ema_loss
    

    def _scheduler_step(self, epoch_avg: float):
        """
        Scheduler step to be called inside training loop
        """
        if self.scheduler is not None:
            if self.cfg.scheduler.name is not None and str(self.cfg.scheduler.name).lower() == "plateau":
                self.scheduler.step(epoch_avg)
            else:
                self.scheduler.step()


    def _optuna_prune_check(self, epoch: int, metric: float) -> None:
        """
        Report intermediate metric to Optuna and prune if requested.

        We report once per epoch using `epoch_avg` (training loss).
        """
        if not self._optuna:
            return
        if self._optuna_trial is None:
            return

        # Defensive: metric must be finite or Optuna may behave oddly
        if not (pd.notna(metric) and float(metric) == float(metric) and abs(float(metric)) != float("inf")):
            return

        # report + prune
        self._optuna_trial.report(float(metric), step=int(epoch))
        if self._optuna_trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch={epoch} metric={metric}")  # type: ignore[attr-defined]
    

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int, start_date: Optional[Date], end_date: Optional[Date]) -> Optional[List[float]]:
        """
        Sequential-window training loop with optional gradient accumulation.

        Parameters
        ----------
        num_epochs : int
            Number of passes over the windowed training calendar.
        start_date : Optional[Date]
            Training range start (inclusive). If None, inferred by dataloader.
        end_date : Optional[Date]
            Training range end (inclusive). If None, uses last available date.
        """
        _check_positive_integer_value(num_epochs, 'num_epochs')
        self.model.train()

        # Adjust start date based on lookback (and its stride — the encoder
        # needs ``lookback × lookback_freq`` historical rows for the first
        # window). For hierarchical encoders, slow lookback dominates.
        if self.model.encoder_type == "simple":
            start_date = self.dataloader._check_valid_start_date(
                start_date, self.lookback, frequency=self.lookback_freq,
            )
        else:
            start_date = self.dataloader._check_valid_start_date(
                start_date, self.lookback_slow, frequency=self.lookback_slow_freq,
            )

        # Check valid date range
        if start_date is not None and end_date is not None:
            if pd.Timestamp(start_date) > pd.Timestamp(end_date):
                raise ValueError("train: start_date must be <= end_date")

        # Cache training metadata + write model_info.json
        self.record_training_info(start_date_train=start_date, end_date_train=end_date)
        self.logger.info(f"Training from {start_date} to {end_date} for {num_epochs} epochs")

        # -----------------------------------------------------------------------------------

        # Early stopping state (train-loss plateau)
        best_score = float("inf")
        bad_epochs = 0
        ema_loss = None

        # Main training loop
        epoch_losses: List[float] = []
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):

            self.logger.info(f"Epoch {epoch}/{self.start_epoch + num_epochs - 1}")

            # Build windows
            window_batches = self.dataloader.get_batch_windows(
                window_days=self.batch_window,
                start_date=start_date,
                end_date=end_date,
                step=self.window_step
            )

            # Epoch accounting
            total_epoch_loss = 0.0
            windows_counted = 0
            opt_steps_total = 0      # optimizer-step attempts this epoch
            opt_steps_skipped = 0    # of which were skipped (non-finite grad)

            self.optimizer.zero_grad(set_to_none=True)

            # Loop for window batches
            for w, batch_dates in enumerate(window_batches, start=1):

                window_loss_float, window_loss_tensor, non_nan_losses = self._train_one_window(batch_dates)

                # If every day in the window was skipped, there is nothing to backprop.
                if non_nan_losses == 0:
                    self.logger.warning(
                        f"(Epoch {epoch}) Window {w}: all days were skipped (NaN/Inf). "
                        "No backward will be performed for this window."
                    )
                    continue

                # Guard: loss tensor exists but may be non-differentiable in rare cases.
                if (not window_loss_tensor.requires_grad) or (window_loss_tensor.grad_fn is None):
                    self.logger.warning(
                        f"(Epoch {epoch}) Window {w}: loss is non-differentiable "
                        "(requires_grad=False or missing grad_fn). Skipping backward."
                    )
                    continue

                # Backward (AMP-safe)
                if self.use_amp:
                    self.scaler.scale(window_loss_tensor).backward()
                else:
                    window_loss_tensor.backward()

                # Gradient accumulation: step every N usable windows
                if (w % self.accumulate_windows) == 0:
                    stepped = self._optimizer_step()
                    opt_steps_total += 1
                    opt_steps_skipped += int(not stepped)

                # Logging
                if (w % max(1, self.log_every_n_windows)) == 0:
                    self.logger.info(f"(Epoch {epoch}) Window {w} loss={window_loss_float:.6f}")

                # Update loss and accouting
                total_epoch_loss += window_loss_float
                windows_counted += 1

            if windows_counted == 0:
                self.logger.warning(
                    "No usable windows were produced for this epoch (all skipped / non-differentiable). "
                    "Skipping optimizer step and epoch accounting."
                )
                continue

            # Flush accumulation at epoch end
            if (windows_counted % self.accumulate_windows) != 0:
                stepped = self._optimizer_step()
                opt_steps_total += 1
                opt_steps_skipped += int(not stepped)

            # -------------------------------------------------------
            
            # Epoch avg
            epoch_avg = total_epoch_loss / max(1, windows_counted)
            epoch_losses.append(epoch_avg)
            self.logger.info(f"Epoch {epoch} avg_loss={epoch_avg:.6f}")

            # Optimizer-step health: how many steps were actually applied
            # vs skipped because of a non-finite gradient this epoch. A
            # small skip count is harmless; a large/growing one signals the
            # SDE is still unstable (raise init_output_scale↓, drift_bound↓,
            # lr↓, or trainer.dt↓).
            opt_applied = opt_steps_total - opt_steps_skipped
            level = self.logger.info if opt_steps_skipped == 0 else self.logger.warning
            level(
                f"Epoch {epoch} optimizer steps: {opt_applied}/{opt_steps_total} applied "
                f"({opt_steps_skipped} skipped, "
                f"{(100.0 * opt_steps_skipped / max(1, opt_steps_total)):.1f}%)"
            )

            # Store partial history so pruned trials still have epoch_avgs available
            if self._optuna and (self._optuna_trial is not None):
                self._optuna_trial.set_user_attr("epoch_avgs", list(epoch_losses))

            # Maybe prune if gridsearching with Optuna
            self._optuna_prune_check(epoch=epoch, metric=epoch_avg)

            # Scheduler step (if not None)
            self._scheduler_step(epoch_avg)

            # Artifacts hook (checkpoints)
            self.artifacts.on_epoch_end(epoch, epoch_avg)
          
            # Early stopping (training-loss plateau)
            stop, best_score, bad_epochs, ema_loss = self._early_stopping_update(
                epoch_avg,
                best_score=best_score,
                bad_epochs=bad_epochs,
                ema_loss=ema_loss,
            )
            if stop:
                break

        self.logger.info("Training complete.")
        self.save_training_artifacts(epoch_losses)

        if self._optuna:
            return epoch_losses
        return None


    # ------------------------------------------------------------------
    # Evaluation date helpers (compressed date management)
    # ------------------------------------------------------------------

    def _fetch_training_window(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Return (start_train, end_train) from model.training_info, with validation.
        """
        if self.model.training_info is None:
            raise RuntimeError("No training_info found in model. Cannot infer evaluation defaults.")

        s = self.model.training_info.get("start_training_date", None)
        e = self.model.training_info.get("end_training_date", None)
        if s is None or e is None:
            raise RuntimeError("Training window not found in model.training_info.")

        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if s_ts > e_ts:
            raise ValueError(f"Invalid training window: {s_ts.date()} > {e_ts.date()}")

        return s_ts, e_ts
    

    def _resolve_eval_dates_request(
        self,
        date: Optional[Date],
        start_date: Optional[Date],
        end_date: Optional[Date],
        step: int,
    ) -> Tuple[str, Union[pd.Timestamp, Tuple[pd.Timestamp, pd.Timestamp]], int]:
        """
        Normalize evaluation inputs into either:
        - ("single", eval_date, step), or
        - ("range", (start_ts, end_ts), step)
        """
        _check_positive_integer_value(step, "step")

        train_s, train_e = self._fetch_training_window()

        # Single date explicit
        if date is not None:
            d_ts = pd.Timestamp(date)
            if train_s <= d_ts <= train_e:
                self.logger.info("Evaluation date is within training range.")
            return "single", d_ts, step

        # Default single day after training
        if start_date is None and end_date is None:
            eval_date = self.dataloader.get_next_available_yield_curve_date(train_e)
            return "single", pd.Timestamp(eval_date), step

        # Range: fill missing edges
        # The rules are:
        # - if start and end are both None: evaluate the day after training end
        # - if start is None: start from day after training end
        # - if end is None: end at last available calendar date
        # - otherwise: evaluate within [start, end]

        if start_date is None:
            start_date = self.dataloader.get_next_available_yield_curve_date(train_e)
            self.logger.info(f"Start_date not provided -> using day after training: {pd.Timestamp(start_date).date()}")

        if end_date is None:
            end_date = self.dataloader.get_last_available_date()
            self.logger.info(f"End_date not provided -> using last available date: {pd.Timestamp(end_date).date()}")

        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if start_ts > end_ts:
            raise ValueError(f"Invalid eval range: start_date={start_ts.date()} is after end_date={end_ts.date()}")

        # Overlap log
        if not (end_ts < train_s or start_ts > train_e):
            self.logger.info("Evaluation range overlaps with training window.")

        return "range", (start_ts, end_ts), step


    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_one_date(self, date: Date, n_paths: Optional[int] = None) -> EvalResults:
        """
        Evaluate the model at `date` and return loss + component breakdown.
        """
        self.model.eval()
        _n_paths = n_paths if n_paths is not None else self.n_paths

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            loss_tensor, components = self._forward_one_date(date, return_components=True, n_paths=_n_paths)

        # Defensive checks (eval should never explode silently)
        if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
            self.logger.warning(f"Eval loss is NaN/Inf at date={date}")
            total_loss = float("inf")
        else:
            total_loss = float(loss_tensor.detach().cpu().item())

        return EvalResults(
            n_paths=_n_paths,
            date=pd.Timestamp(date),
            total_loss=total_loss,
            components=components,
            meta={}
            )


    def evaluate(
        self,
        date: Optional[Date] = None,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        n_paths: Optional[int] = None,
        step: int = 1,
        save: bool = True,
    ) -> Union[EvalResults, List[EvalResults]]:
        """
        Evaluate either a single date or over a requested window.

        If all inputs are None, defaults to evaluating the first date
        strictly after the end of the training window.

        Parameters
        ----------
        date : Optional[Date]
            If provided, runs evaluation only on this date.
        start_date : Optional[Date]
            Start date of the evaluation range (inclusive).
        end_date : Optional[Date]
            End date of the evaluation range (inclusive).
        step : int
            Subsample the evaluation calendar by taking every `step` dates.
            Example: step=5 evaluates roughly weekly if calendar is daily.
        save : bool

        Returns
        -------
        EvalResults or List[EvalResults]
            - Single day: EvalResults
            - Range: list[EvalResults]
        """
        if n_paths is not None:
            _check_positive_integer_value(n_paths, "n_paths")
        
        # Resolve input date(s)
        kind, payload, step = self._resolve_eval_dates_request(
            date=date, start_date=start_date, end_date=end_date, step=step
        )

        # -------------------------------------------------------
        # Single day evaluation
        if kind == "single":
            
            eval_ts = payload  #pd.Timestamp
            self.logger.info(f"Evaluating single date: {eval_ts.date()}")
            res = self._eval_one_date(eval_ts, n_paths=n_paths)

            if save:
                df = eval_results_to_frame(res)
                fname = f"eval_{eval_ts.date()}.csv"
                self.artifacts.save_eval_csv(df, filename=fname)
            
            return res

        # -------------------------------------------------------
        # kind == "range"

        start_ts, end_ts = payload  # Tuple[pd.Timestamp, pd.Timestamp]
        self.logger.info(f"Evaluating range: {start_ts.date()} -> {end_ts.date()}  (step={step})")

        dates = self.dataloader.get_dates_between(start_ts, end_ts)
        if step > 1:
            dates = dates[::step]

        # Run evaluation
        results: List[EvalResults] = [self._eval_one_date(d, n_paths=n_paths) for d in dates]

        if save:
            df = eval_results_to_frame(results)
            fname = f"eval_{start_ts.date()}_{end_ts.date()}_step{step}.csv"
            self.artifacts.save_eval_csv(df, filename=fname)

        return results


    def evaluate_training_set(
        self, 
        n_paths: Optional[int] = None, 
        step: int = 1, 
        save: bool = True
    ) -> List[EvalResults]:
        """
        Evaluate the model on the full training window.

        Notes
        -----
        - No window batching is used for evaluation.
        - `step` can be used to subsample the calendar for quicker diagnostics.

        Parameters
        ----------
        step : int
            Evaluate every `step` training dates.
        save : bool
            Placeholder: keep interface stable for later.

        Returns
        -------
        List[EvalResults]
            One result per evaluated day.
        """
        # Checks
        _check_positive_integer_value(step, "step")
        if n_paths is not None:
            _check_positive_integer_value(n_paths, "n_paths")

        # Fetch training dates
        train_s, train_e = self._fetch_training_window()
        self.logger.info(f"Evaluating training set: {train_s.date()} -> {train_e.date()}  (step={step})")

        dates = self.dataloader.get_dates_between(train_s, train_e)
        if step > 1:
            dates = dates[::step]

        if len(dates) == 0:
            self.logger.warning("No dates found in training window for evaluation.")
            return []

        # Run evaluation
        results: List[EvalResults] = [self._eval_one_date(d, n_paths=n_paths) for d in dates]

        if save:
            df = eval_results_to_frame(results)
            fname = f"eval_train_{train_s.date()}_{train_e.date()}_step{step}.csv"
            self.artifacts.save_eval_csv(df, filename=fname)

        return results


    @torch.no_grad()
    def compute_prices(self, date: Date, n_paths: Optional[int] = None) -> MarketSnapshot:
        """
        Compute model-implied observables at a given date.

        Notes
        -----
        - This does NOT compute the loss.
        - This does NOT require market targets (except maturities).
        - Output shape and structure mirrors MarketSnapshot (future proof for bonds/futures).

        Parameters
        ----------
        date : Date
            Anchor date at which to compute model-implied observables.

        Returns
        -------
        MarketSnapshot
            A snapshot-like object containing model-implied targets at `date`.
        """
        self.model.eval()

        # Build model-implied targets
        obs_snapshot = self._get_snapshot(date)
        ts = self._make_ts(snapshot=obs_snapshot)

        # NN forwards under autocast; pricing arithmetic in float32.
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            latent_repr = self.get_latent_representation_from_date(date, n_paths=n_paths, ts=ts)
            r0 = self._get_r0(date)
            realisations = self._decode(latent_repr, r0=r0)

        # Delegate to the pricer for consistent handling of every target,
        # including the futures branch.
        model_snapshot = self.pricer.price_snapshot(
            realisations=realisations,
            snapshot=obs_snapshot,
            latent_paths=latent_repr,
            simulated_times=ts,
            bondnet=self.model.bondnet,
        )
        return model_snapshot
    
    
    # # ------------------------------------------------------------------
    # # Fine-tuning (decoder-only) hooks
    # # ------------------------------------------------------------------

    # def _freeze_all_but_decoder(self) -> None:
    #     """
    #     Freeze all model parameters except the decoder.
    #     """
    #     for p in self.model.parameters():
    #         p.requires_grad = False

    #     if not hasattr(self.model, "decoder"):
    #         raise AttributeError("Model has no attribute 'decoder'.")

    #     for p in self.model.decoder.parameters():
    #         p.requires_grad = True


    # def _build_finetune_optimizer(self, lr: Optional[float] = None) -> torch.optim.Optimizer:
    #     """
    #     Create a fresh optimizer for finetuning decoder only.
    #     """
    #     params = list(self.model.decoder.parameters())
    #     if not params:
    #         raise RuntimeError("Decoder has no parameters.")

    #     # Copy optimizer cfg
    #     opt_cfg = dict(self.cfg.optimizer.params)
    #     if lr is not None:
    #         opt_cfg["lr"] = lr

    #     # Build by name
    #     name = str(self.cfg.optimizer.name).lower()
    #     if name == "adam":
    #         return torch.optim.Adam(params, **opt_cfg)
    #     if name == "adamw":
    #         return torch.optim.AdamW(params, **opt_cfg)
    #     if name == "sgd":
    #         return torch.optim.SGD(params, **opt_cfg)

    #     raise ValueError(f"Unsupported optimizer for finetune: {self.cfg.optimizer.name}")


    # def fine_tune(self, *args, **kwargs):
    #     """
    #     Not implemented yet.
    #     """
    #     raise NotImplementedError("fine_tune is not implemented yet.")
    

    # ------------------------------------------------------------------
    # Saving logic
    # ------------------------------------------------------------------

    def record_training_info(self, *, start_date_train: Optional[Date], end_date_train: Optional[Date]) -> None:
        """
        Cache training metadata into the model and write model_info.json.
        Call once at the start of training.
        """

        # Normalise dates to ISO strings so model_info.json is portable across
        # platforms (default=str fallback in atomic_save_json is locale-sensitive).
        def _iso(d: Optional[Date]) -> Optional[str]:
            if d is None:
                return None
            return pd.Timestamp(d).normalize().date().isoformat()

        info: Dict[str, Any] = {
            "start_training_date": _iso(start_date_train),
            "end_training_date": _iso(end_date_train),
            "device": str(self.device),
            "n_paths": self.n_paths,
            "batch_window": self.batch_window,
            "window_step": self.window_step,
            "use_amp": self.use_amp,
            "grad_clip_norm": self.grad_clip_norm,
            "accumulate_windows": self.accumulate_windows,
            "optimizer": self.cfg.optimizer.name,
            "optimizer_params": self.cfg.optimizer.params,
            "loss": self.cfg.loss.name,
            "loss_params": self.cfg.loss.params,
            "scheduler": self.cfg.scheduler.name,
            "scheduler_params": self.cfg.scheduler.params,
            "early_stopping": self.cfg.early_stopping.enabled,  
            "early_stopping_params": {
                "patience": self.cfg.early_stopping.patience,
                "min_delta": self.cfg.early_stopping.min_delta,
                "use_ema": self.cfg.early_stopping.use_ema,
                "ema_alpha": self.cfg.early_stopping.ema_alpha,
            },
        }

        if self.model.encoder_type == "hierarchical":
            info.update({
                "lookback_fast": self.lookback_fast,
                "lookback_slow": self.lookback_slow,
                "lookback_fast_freq": self.lookback_fast_freq,
                "lookback_slow_freq": self.lookback_slow_freq,
            })
        else:
            info.update({
                "lookback": self.lookback,
                "lookback_freq": self.lookback_freq,
            })

        self.model.cache_training_info(info)

        if self._optuna:
            return

        self.model.save_model_info(self.output_dir)
        self.logger.info("Recorded training info and wrote model_info.json")


    def record_finetune_info(self, *, finetune_date: datetime) -> None:
        """
        Cache a finetune stamp into the model and update model_info.json.
        """
        # self.model.cache_finetune_info(finetune_date)
        self.model.save_model_info(self.output_dir)
        self.logger.info("Recorded finetune info and updated model_info.json")


    def save_training_artifacts(self, epoch_losses: List[float]) -> None:
        """
        Save epoch losses, final weights, and refresh model_info.json.
        Call once at the end of training.
        """
        if self._optuna:
            return

        self.artifacts.save_losses(epoch_losses)
        self.artifacts.save_final_model()
        self.model.save_model_info(self.output_dir)
        self.logger.info(f"Saved training artifacts to {self.output_dir}")