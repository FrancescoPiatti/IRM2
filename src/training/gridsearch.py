# src/training/gridsearch.py
import os
import inspect
import warnings

from dataclasses import is_dataclass
from dataclasses import replace

from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence

import optuna
import pandas as pd

from ..types.types_utils import Date
from ..types.gridsearch_types import TrialResult
from ..types.gridsearch_types import GridSearchResults

from ..utils.checks import _check_positive_integer_value


# -----------------------------------------------------------------------------
# Grid Search
# -----------------------------------------------------------------------------

class OptunaGridSearch:
    """
    Optuna grid-search over your config objects with minimal ceremony.

    Design
    ------
    - `dataloader` is external and passed in (not constructed per trial).
    - You provide *base* EncoderCfg / NSDECfg / TrainerCfg.
    - `param_grid` is a dict: { "path": [choices...] }.
    - Paths are dot-based and start with one of:
        - "encoder."
        - "nsde."
        - "trainer."
        - "model."

    Rules
    -----
    - Deep edits into encoder/nsde network dicts are NOT supported:
        - allowed:   nsde.diffusion = {...}         (replace whole mapping)
        - forbidden: nsde.diffusion.n_layers = 3    (raises)
    - Deep edits into trainer mappings ARE supported (user-friendly):
        - example: trainer.optimizer.params.lr = 1e-3   (dict traversal allowed)
    - `model.latent_dim` and `model.noise_dim` are supported and routed to model init.

    GridSearch outputs folder
    -------------------------
    - A single folder is created once per run:
        <results_root>/GridSearch_<study_name>
      If the folder already exists, an error is raised (no timestamps).
    - All trials share the same logger inside that folder (Trainer optuna-mode).
    - This class also writes:
        - grid_results.json
        - epochs.csv        (rows=trial, cols=epoch_1..epoch_N; padded with NaN if pruned)
        - eval_losses.csv   (rows=trial, cols=eval dates; padded with NaN if missing)
    """

    def __init__(
        self,
        *,
        param_grid: Mapping[str, Sequence[Any]],
        dataloader: Any,  # MarketDataLoader instance
        base_encoder_cfg: Any,
        base_nsde_cfg: Any,
        base_trainer_cfg: Any,
        model_cls: Any,   # ShortRateModel class
        trainer_cls: Any, # Trainer class
        direction: str = "minimize",
        seed: Optional[int] = 0,
        study_name: str = "optuna_grid",
        latent_dim: Optional[int] = None,
        noise_dim: Optional[int] = None,
    ):
        if not isinstance(param_grid, Mapping) or len(param_grid) == 0:
            raise ValueError("param_grid must be a non-empty mapping {path: [choices...]}.")

        self.dataloader = dataloader

        # Configs
        self.base_encoder_cfg = base_encoder_cfg
        self.base_nsde_cfg = base_nsde_cfg
        self.base_trainer_cfg = base_trainer_cfg

        self.model_cls = model_cls
        self.trainer_cls = trainer_cls

        self.direction = str(direction).lower()
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'.")

        self.seed = seed
        self.study_name = str(study_name)

        # Resolve latent_dim and noise_dim
        if latent_dim is not None:
            _check_positive_integer_value(int(latent_dim), "latent_dim")
        if noise_dim is not None:
            _check_positive_integer_value(int(noise_dim), "noise_dim")
        self._default_latent_dim = int(latent_dim) if latent_dim is not None else None
        self._default_noise_dim = int(noise_dim) if noise_dim is not None else None

        # We must feed GridSampler only primitives; keep a decoding table per key.
        # - Optuna will see tokens (strings) for non-primitive values.
        # - We decode them back inside objective and when producing results.
        self._value_registry: Dict[str, Dict[str, Any]] = {}
        self.param_grid: Dict[str, List[Any]] = self._encode_param_grid(param_grid)

        # Keep a stable key order for reproducibility
        self._grid_keys = sorted(self.param_grid.keys())


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        num_epochs: int,
        train_start_date: Optional[Date],
        train_end_date: Optional[Date],
        eval_start_date: Optional[Date] = None,
        eval_end_date: Optional[Date] = None,
        eval_step: int = 1,
        save_eval: bool = True,
        n_trials: Optional[int] = None,
    ) -> GridSearchResults:
        """
        Execute the grid-search.

        Notes
        -----
        - Creates a single output folder:
            <results_root>/GridSearch_<study_name>
          and raises if it already exists.
        - All trials write logs into that same folder (via Trainer optuna-mode).
        - Additionally saves:
            - grid_results.json
            - epochs.csv
            - eval_losses.csv

        Parameters
        ----------
        num_epochs : int
            Number of training epochs per trial.
        train_start_date, train_end_date :
            Forwarded into trainer.train(...).
        eval_start_date, eval_end_date :
            Forwarded into trainer.evaluate(...). If not provided, defaults to the
            first available yield curve date strictly after train_end_date and a
            single-day evaluation.
        eval_step : int
            Subsample evaluation calendar by taking every `eval_step` dates.
        save_eval : bool
            Forwarded into trainer.evaluate(..., save=save_eval). In optuna-mode
            the trainer may ignore IO anyway; we still compute eval losses and
            store them in eval_losses.csv from returned objects.
        n_trials : Optional[int]
            If provided, cap the number of trials (otherwise runs full grid).

        Returns
        -------
        GridSearchResults
            Best trial summary + all completed trial results (decoded).
        """

        _check_positive_integer_value(num_epochs, "num_epochs")
        _check_positive_integer_value(eval_step, "eval_step")

        if eval_start_date is None and train_end_date is not None:
            eval_start_date = self.dataloader.get_next_available_yield_curve_date(train_end_date)

        if eval_end_date is None and eval_start_date is not None:
            eval_end_date = eval_start_date

        # ------------------------------------------------------------------
        # Create one shared GridSearch output folder (no timestamps, error if exists)
        # ------------------------------------------------------------------
        base_tr_cfg = self._clone_cfg(self.base_trainer_cfg)
        results_root = getattr(base_tr_cfg, "results_root", None)
        if results_root is None:
            raise AttributeError("base_trainer_cfg must expose 'results_root' to build the GridSearch folder.")

        grid_run_name = f"GridSearch_{self.study_name}"
        grid_dir = os.path.join(str(results_root), grid_run_name)

        # Raise if already exists
        os.makedirs(grid_dir, exist_ok=False)

        # ------------------------------------------------------------------
        # Optuna study
        # ------------------------------------------------------------------
        sampler = optuna.samplers.GridSampler(self.param_grid, seed=self.seed)

        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=self.study_name,
        )

        # In-memory tracking for CSV exports
        epoch_series: Dict[int, List[float]] = {}         # trial -> [epoch_avg...]
        eval_loss_series: Dict[int, Dict[str, float]] = {}  # trial -> {date_str: loss}

        def objective(trial: optuna.Trial) -> float:
            # Build fresh configs per trial (NO deepcopy: frozen mappings may be unpicklable)
            enc_cfg = self._clone_cfg(self.base_encoder_cfg)
            nsde_cfg = self._clone_cfg(self.base_nsde_cfg)
            tr_cfg = self._clone_cfg(self.base_trainer_cfg)

            # Model init kwargs routed from "model.*"
            model_kwargs: Dict[str, Any] = {}

            # Choose one value per grid key
            for key in self._grid_keys:
                choices = self.param_grid[key]
                raw_val = trial.suggest_categorical(key, choices)
                val = self._decode_value(key, raw_val)
                self._apply_choice(key, val, enc_cfg, nsde_cfg, tr_cfg, model_kwargs)

            # Force all trials into the same shared grid folder naming convention
            tr_cfg.run_name = grid_run_name

            # -------------------------------------------------------
            # Model dims: prefer grid -> constructor defaults -> user-provided init defaults
            # -------------------------------------------------------
            latent_dim = self._resolve_model_dim(
                dim_name="latent_dim",
                model_kwargs=model_kwargs,
                fallback=self._default_latent_dim,
            )

            # If diagonal noise: noise_dim is irrelevant; make it consistent with torchsde convention
            # (diagonal noise => Brownian dim = state dim)
            noise_type = str(getattr(nsde_cfg, "noise_type", "")).lower()
            if noise_type == "diagonal":
                noise_dim = int(latent_dim)
            else:
                noise_dim = self._resolve_model_dim(
                    dim_name="noise_dim",
                    model_kwargs=model_kwargs,
                    fallback=self._default_noise_dim,
    )

            # Build model + trainer
            model = self.model_cls(
                name=str(getattr(tr_cfg, "run_name", grid_run_name)),
                encoder=enc_cfg,
                nsde=nsde_cfg,
                latent_dim=int(latent_dim),
                noise_dim=int(noise_dim),
            )

            # IMPORTANT: pass optuna_trial so Trainer can enable pruning + disable IO
            trainer = self.trainer_cls(
                model=model,
                dataloader=self.dataloader,
                config=tr_cfg,
                resume_from=None,
                optuna_trial=trial,
            )

            # ----------------
            # Train
            # ----------------

            train_out = trainer.train(num_epochs=num_epochs, start_date=train_start_date, end_date=train_end_date)

            # We want per-epoch averages. Trainer is expected to return a list[float].
            # If it returns None, keep empty and warn once per trial.
            if train_out is None:
                epoch_series[int(trial.number)] = []
                trial.set_user_attr("epoch_avgs", [])
            else:
                try:
                    ep = list(train_out)
                    epoch_series[int(trial.number)] = [float(x) for x in ep]
                    trial.set_user_attr("epoch_avgs", epoch_series[int(trial.number)])
                except Exception:
                    epoch_series[int(trial.number)] = []
                    trial.set_user_attr("epoch_avgs", [])

            # ----------------
            # Evaluate (range or single day)
            # ----------------
            res = trainer.evaluate(
                start_date=eval_start_date,
                end_date=eval_end_date,
                step=eval_step,
                save=save_eval,
            )

            # Convert eval output to scalar objective (mean total_loss)
            if isinstance(res, list):
                vals = [float(r.total_loss) for r in res]
                value = float(sum(vals) / max(1, len(vals)))

                # store per-date eval loss for CSV
                per_date: Dict[str, float] = {str(pd.Timestamp(r.date).date()): float(r.total_loss) for r in res}
                eval_loss_series[int(trial.number)] = per_date
                trial.set_user_attr("eval_losses", per_date)
            else:
                value = float(res.total_loss)
                d = str(pd.Timestamp(res.date).date())
                per_date = {d: float(res.total_loss)}
                eval_loss_series[int(trial.number)] = per_date
                trial.set_user_attr("eval_losses", per_date)

            return value

        if n_trials is None:
            study.optimize(objective)
        else:
            study.optimize(objective, n_trials=int(n_trials))

        # ------------------------------------------------------------------
        # Collect results (DECODE params back to original objects)
        # ------------------------------------------------------------------
        completed: List[TrialResult] = []
        for t in study.trials:
            if t.value is None:
                continue
            decoded_params = {k: self._decode_value(k, v) for k, v in dict(t.params).items()}
            completed.append(
                TrialResult(
                    number=int(t.number),
                    value=float(t.value),
                    params=decoded_params,
                )
            )

        best = study.best_trial
        best_params_decoded = {k: self._decode_value(k, v) for k, v in dict(best.params).items()}

        out = GridSearchResults(
            best_value=float(best.value),
            best_params=best_params_decoded,
            trials=completed,
        )

        # ------------------------------------------------------------------
        # Write artifacts into the shared GridSearch folder
        # ------------------------------------------------------------------
        # grid_results.json
        with open(os.path.join(grid_dir, "grid_results.json"), "w") as f:
            f.write(out.to_json())

        # epochs.csv (pad to num_epochs) — include pruned trials via user_attrs
        ep_cols = [f"epoch_{i}" for i in range(1, int(num_epochs) + 1)]
        ep_rows: Dict[int, List[float]] = {}

        for t in study.trials:
            losses = t.user_attrs.get("epoch_avgs", [])
            row = [float(x) for x in list(losses)[: int(num_epochs)]]
            if len(row) < int(num_epochs):
                row += [float("nan")] * (int(num_epochs) - len(row))
            ep_rows[int(t.number)] = row

        df_epochs = pd.DataFrame.from_dict(ep_rows, orient="index", columns=ep_cols)
        df_epochs.index.name = "trial"
        df_epochs.to_csv(os.path.join(grid_dir, "epochs.csv"))

        # eval_losses.csv (union of all eval dates as columns) — include all trials
        trial_eval_maps: Dict[int, Dict[str, float]] = {}
        for t in study.trials:
            m = t.user_attrs.get("eval_losses", {})
            trial_eval_maps[int(t.number)] = {str(k): float(v) for k, v in dict(m).items()}

        all_dates = sorted({d for m in trial_eval_maps.values() for d in m.keys()})
        rows: Dict[int, List[float]] = {
            trn: [float(trial_eval_maps[trn].get(d, float("nan"))) for d in all_dates]
            for trn in trial_eval_maps.keys()
        }

        df_eval = pd.DataFrame.from_dict(rows, orient="index", columns=all_dates)
        df_eval.index.name = "trial"
        df_eval.to_csv(os.path.join(grid_dir, "eval_losses.csv"))

        return out


    # ------------------------------------------------------------------
    # Grid encoding/decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _is_primitive(v: Any) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def _encode_param_grid(self, param_grid: Mapping[str, Sequence[Any]]) -> Dict[str, List[Any]]:
        """
        Encode any non-primitive choices into string tokens so Optuna GridSampler
        can store them safely. Keep a per-key registry to decode them later.

        This allows users to keep run scripts like:
            "nsde.diffusion": [ {..}, {..} ]
        without changing anything externally.
        """
        encoded: Dict[str, List[Any]] = {}

        for k, vals in param_grid.items():
            key = str(k)
            choices = list(vals)

            enc_choices: List[Any] = []
            registry: Dict[str, Any] = {}

            for i, v in enumerate(choices):
                if self._is_primitive(v):
                    enc_choices.append(v)
                else:
                    token = f"__obj_{i}__"
                    # ensure uniqueness per key even if the user repeats values
                    while token in registry:
                        token = token.replace("__", "_", 1)  # trivial bump
                    registry[token] = v
                    enc_choices.append(token)

            if registry:
                self._value_registry[key] = registry

            encoded[key] = enc_choices

        return encoded

    def _decode_value(self, key: str, value: Any) -> Any:
        reg = self._value_registry.get(str(key))
        if reg is None:
            return value
        if isinstance(value, str) and value in reg:
            return reg[value]
        return value


    # ------------------------------------------------------------------
    # Model dim resolution
    # ------------------------------------------------------------------

    def _infer_model_default(self, name: str) -> Optional[int]:
        """
        Try to infer default from model_cls.__init__ signature.
        Returns None if not found.
        """
        try:
            sig = inspect.signature(self.model_cls.__init__)
        except Exception:
            return None

        p = sig.parameters.get(name)
        if p is None or p.default is inspect._empty:
            return None
        try:
            return int(p.default)
        except Exception:
            return None

    def _resolve_model_dim(self, *, dim_name: str, model_kwargs: Dict[str, Any], fallback: Optional[int]) -> int:
        """
        Resolution order:
        1) model_kwargs (from grid: model.latent_dim / model.noise_dim)
        2) fallback passed at OptunaGridSearch init (latent_dim/noise_dim)
        3) default from model_cls.__init__ (if any)
        4) warn + hard fallback 16
        """
        if dim_name in model_kwargs:
            v = int(model_kwargs[dim_name])
            _check_positive_integer_value(v, dim_name)
            return v

        if fallback is not None:
            v = int(fallback)
            _check_positive_integer_value(v, dim_name)
            return v

        inferred = self._infer_model_default(dim_name)
        if inferred is not None:
            _check_positive_integer_value(int(inferred), dim_name)
            return int(inferred)

        warnings.warn(
            f"OptunaGridSearch: '{dim_name}' not provided in param_grid and no default was found on "
            f"{getattr(self.model_cls, '__name__', 'model_cls')}. Using hard fallback {16}.",
            category=UserWarning,
            stacklevel=2,
        )
        return 16


    # ------------------------------------------------------------------
    # Param application
    # ------------------------------------------------------------------

    def _apply_choice(
        self,
        path: str,
        value: Any,
        encoder_cfg: Any,
        nsde_cfg: Any,
        trainer_cfg: Any,
        model_kwargs: Dict[str, Any],
    ) -> None:
        parts = str(path).split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid param path '{path}'. Expected 'root.field[.field...]'.")

        root = parts[0].lower()
        rest = parts[1:]

        if root == "encoder":
            self._set_attr_path(
                obj=encoder_cfg,
                parts=rest,
                value=value,
                allow_mapping_traversal=False,
                context="encoder",
            )
            return

        if root == "nsde":
            self._set_attr_path(
                obj=nsde_cfg,
                parts=rest,
                value=value,
                allow_mapping_traversal=False,
                context="nsde",
            )
            return

        if root == "trainer":
            self._set_attr_path(
                obj=trainer_cfg,
                parts=rest,
                value=value,
                allow_mapping_traversal=True,
                context="trainer",
            )
            return

        if root == "model":
            if len(rest) != 1:
                raise ValueError(f"Invalid model path '{path}'. Use model.latent_dim / model.noise_dim only.")
            key = rest[0]
            if key not in ("latent_dim", "noise_dim"):
                raise ValueError(f"Unsupported model hyperparameter '{path}'.")
            model_kwargs[key] = value
            return

        raise ValueError(f"Unknown param root '{root}' in path '{path}'.")


    def _set_attr_path(
        self,
        *,
        obj: Any,
        parts: List[str],
        value: Any,
        allow_mapping_traversal: bool,
        context: str,
    ) -> None:
        cur = obj

        for i, name in enumerate(parts):
            is_last = (i == len(parts) - 1)

            if isinstance(cur, Mapping):
                if not allow_mapping_traversal:
                    raise ValueError(
                        f"Deep mapping edits are not supported for {context}: "
                        f"attempted to set '{context}." + ".".join(parts) + "'. "
                        f"Replace the whole mapping instead (e.g. nsde.diffusion={{...}})."
                    )

                if is_last:
                    cur[name] = value  # type: ignore[index]
                    return

                if name not in cur:
                    cur[name] = {}  # type: ignore[index]
                cur = cur[name]  # type: ignore[index]
                continue

            if not hasattr(cur, name):
                raise AttributeError(
                    f"{context}: '{type(cur).__name__}' has no attribute '{name}' "
                    f"(path='{context}." + ".".join(parts) + "')"
                )

            if is_last:
                setattr(cur, name, value)
                return

            cur = getattr(cur, name)


    # ------------------------------------------------------------------
    # Cloning (no deepcopy)
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_cfg(cfg: Any) -> Any:
        """
        Clone a config object without deepcopy.

        Rationale
        ---------
        Your configs contain frozen mappings (e.g. mappingproxy) which can be
        unpicklable and break deepcopy. We only need a *new dataclass instance*
        whose fields reference the same immutable mappings.
        """
        if is_dataclass(cfg):
            return replace(cfg)

        # Fallback: best-effort shallow clone
        if hasattr(cfg, "__dict__"):
            try:
                return type(cfg)(**dict(cfg.__dict__))
            except Exception:
                pass

        return cfg


