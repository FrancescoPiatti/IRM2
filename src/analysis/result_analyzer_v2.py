# src/analysis/result_analyzer_v2.py
"""
TrialAnalyzer — deep analysis of a single grid-search trial folder.

A grid search writes one subfolder per configuration::

    results/GridSearch_<study>_<n>/
        grid_results.json
        epochs.csv
        eval_losses.csv
        trial_000/
            model_state.pt      # torch state_dict of the trained model
            summary.json        # params, eval value, per-epoch + per-date losses
            model_info.json      # model manifest (encoder/nsde/decoder/bondnet cfgs)

``TrialAnalyzer`` points at one ``trial_XXX/`` folder and gives you:

Static (artifacts only, no data needed)
---------------------------------------
* ``summary()``                      — text overview (params, value, losses)
* ``architecture_table()``           — per-module parameter counts
* ``parameter_stats()``              — per-tensor weight norm/mean/std/finite
* ``plot_training_curve()``          — train loss vs epoch
* ``plot_eval_losses()``             — held-out loss vs date
* ``plot_weight_distributions()``    — weight histograms by sub-network
* ``plot_weight_matrix_3d(name)``    — 3-D surface of any weight matrix

Dynamic (needs a ``MarketDataLoader`` on ``data2``)
---------------------------------------------------
* ``simulate(date)``                 — latent + short-rate Monte-Carlo paths
* ``plot_short_rate_fan(date)``      — short-rate fan chart (percentiles)
* ``plot_latent_surface_3d(date)``   — mean latent state over (time × dim)
* ``yield_curve_table(date)``        — model vs market yields + error
* ``plot_yield_surface_3d(dates)``   — model/market/error yield surfaces
* ``gradient_report(date)``          — per-layer gradient-norm table + bar chart
* ``futures_report(date)``           — CTD selection frequency + BondNet stats

``run_all(...)`` produces everything into ``trial_XXX/analysis_v2/``.

The model is reconstructed from ``model_info.json`` (preferred) or, for older
artifacts that predate the BondNet manifest entry, the BondNet architecture is
inferred from the ``state_dict`` shapes. Dynamic methods reuse the project's
``Trainer`` so the forward/pricing path (r0 anchoring, time grid, CTD pricing)
is identical to training — no re-implementation drift.
"""
from __future__ import annotations

import os
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe; we save PNGs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import torch

from ..configs import (
    EncoderCfg,
    NSDECfg,
    TrainerCfg,
    SimpleBondNetCfg,
    FiLMBondNetCfg,
)
from ..models.short_rate_model import ShortRateModel


Date = Union[str, pd.Timestamp]


# =====================================================================
# Config reconstruction helpers
# =====================================================================

def _filter_to_fields(cls, d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only keys that are real constructor fields of dataclass ``cls``."""
    if not d:
        return {}
    names = {f.name for f in dataclass_fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _branch_linear_shapes(state: Dict[str, torch.Tensor], prefix: str) -> List[Tuple[int, int]]:
    """
    Return ``(out_features, in_features)`` of every ``nn.Linear`` weight under
    ``prefix`` in ``state``, ordered by the numeric layer index. Used to
    reverse-engineer an MLP branch from a state_dict.
    """
    found: List[Tuple[int, Tuple[int, int]]] = []
    for k, v in state.items():
        if k.startswith(prefix) and k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 2:
            tail = k[len(prefix):]                 # e.g. ".3.weight"
            parts = [p for p in tail.split(".") if p != ""]
            try:
                idx = int(parts[0])
            except (ValueError, IndexError):
                idx = len(found)
            found.append((idx, (int(v.shape[0]), int(v.shape[1]))))
    found.sort(key=lambda t: t[0])
    return [shp for _, shp in found]


def _infer_encoder_input_dim(state: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Recover the encoder's input feature dimension from a saved state_dict.

    The recurrent encoders (LSTM/GRU/RNN) are built *lazily* — the underlying
    ``nn.LSTM`` only materialises on the first forward, when the input width is
    known. A model reconstructed for offline analysis never runs that forward,
    so unless we pass ``input_dim`` up front the encoder stays empty and its
    trained weights load as "unexpected keys" (i.e. silently dropped). We read
    the width back from the checkpoint instead:

    * recurrent: ``*.weight_ih_l0`` has shape ``(gates*hidden, input)``.
    * mamba:     ``*.input_proj.weight`` has shape ``(n_units, input)``.
    """
    for k, v in state.items():
        if k.startswith("encoder.") and k.endswith("weight_ih_l0") and getattr(v, "ndim", 0) == 2:
            return int(v.shape[1])
    for k, v in state.items():
        if k.startswith("encoder.") and k.endswith("input_proj.weight") and getattr(v, "ndim", 0) == 2:
            return int(v.shape[1])
    return None


def _infer_bondnet_cfg(
    state: Dict[str, torch.Tensor],
    *,
    activation: str = "silu",
    output_positive: bool = True,
) -> Optional[Union[SimpleBondNetCfg, FiLMBondNetCfg]]:
    """
    Best-effort reconstruction of a BondNet config from ``state_dict`` shapes,
    for artifacts saved before ``bondnet_config`` was added to the manifest.

    Handles ``SimpleBondNet`` (latent_branch/bond_branch/fusion_head) and
    ``FiLMBondNet`` (latent_trunk/film_net/head). Activation and
    ``output_positive`` carry no parameters, so they cannot be inferred — they
    default to the project's BondNet defaults and can be overridden.
    """
    bn_keys = [k for k in state if k.startswith("bondnet.")]
    if not bn_keys:
        return None

    def _branch(prefix: str) -> List[Tuple[int, int]]:
        return _branch_linear_shapes(state, f"bondnet.{prefix}")

    # ----- SimpleBondNet -----
    if any(k.startswith("bondnet.latent_branch") for k in bn_keys):
        lat = _branch("latent_branch")
        bond = _branch("bond_branch")
        fus = _branch("fusion_head")
        if not (lat and bond and fus):
            return None
        return SimpleBondNetCfg(
            latent_dim=lat[0][1],
            bond_feat_dim=bond[0][1],
            latent_n_layers=max(1, len(lat) - 1),
            latent_n_units=tuple(s[0] for s in lat[:-1]) or (lat[-1][0],),
            bond_n_layers=max(1, len(bond) - 1),
            bond_n_units=tuple(s[0] for s in bond[:-1]) or (bond[-1][0],),
            fusion_n_layers=max(1, len(fus) - 1),
            fusion_n_units=tuple(s[0] for s in fus[:-1]) or (fus[-1][0],),
            activation=activation,
            output_positive=output_positive,
        )

    # ----- FiLMBondNet -----
    if any(k.startswith("bondnet.latent_trunk") for k in bn_keys):
        trunk = _branch("latent_trunk")
        film = _branch("film_net")
        head = _branch("head")
        if not (trunk and film and head):
            return None
        hidden_dim = trunk[-1][0]
        return FiLMBondNetCfg(
            latent_dim=trunk[0][1],
            bond_feat_dim=film[0][1],
            trunk_n_layers=max(1, len(trunk) - 1),
            trunk_n_units=tuple(s[0] for s in trunk[:-1]) or (hidden_dim,),
            film_n_layers=max(1, len(film) - 1),
            film_n_units=tuple(s[0] for s in film[:-1]) or (film[-1][0],),
            head_n_layers=max(1, len(head) - 1),
            head_n_units=tuple(s[0] for s in head[:-1]) or (head[-1][0],),
            hidden_dim=hidden_dim,
            activation=activation,
            output_positive=output_positive,
        )

    return None


# =====================================================================
# Analyzer
# =====================================================================

class TrialAnalyzer:
    """
    Analyze one grid-search trial folder.

    Parameters
    ----------
    trial_dir : str | Path
        Path to a ``trial_XXX/`` folder containing ``model_state.pt`` and
        ``model_info.json``.
    dataloader : MarketDataLoader, optional
        Needed only for the dynamic (data-driven) analyses. Build it on the
        same ``data2`` and date range used for training, ideally on CPU.
    device : str
        Device for the reloaded model. Defaults to ``"cpu"``.
    bondnet_activation, bondnet_output_positive :
        Only used when the BondNet config has to be inferred from the
        state_dict (old artifacts). Match your training config.
    loss_weights : tuple(float, float, float), optional
        (yield, short_rate, futures) weights used by ``gradient_report`` /
        ``futures_report``. Defaults to (1, 1, 1e-4) to mirror the joint grid.
    """

    def __init__(
        self,
        trial_dir: Union[str, Path],
        *,
        dataloader: Any = None,
        device: str = "cpu",
        bondnet_activation: str = "silu",
        bondnet_output_positive: bool = True,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1e-4),
    ):
        self.trial_dir = Path(trial_dir)
        if not self.trial_dir.exists():
            raise FileNotFoundError(f"trial_dir does not exist: {self.trial_dir}")

        self.dataloader = dataloader
        self.device = torch.device(device)
        self.bondnet_activation = bondnet_activation
        self.bondnet_output_positive = bool(bondnet_output_positive)
        self.loss_weights = tuple(float(x) for x in loss_weights)

        self.analysis_dir = self.trial_dir / "analysis_v2"

        self._model: Optional[ShortRateModel] = None
        self._state: Optional[Dict[str, torch.Tensor]] = None
        self._info: Optional[Dict[str, Any]] = None
        self._summary: Optional[Dict[str, Any]] = None
        self._trainer = None  # lazy

    # -----------------------------------------------------------------
    # IO
    # -----------------------------------------------------------------

    def _ensure_out(self) -> Path:
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        return self.analysis_dir

    @property
    def summary_json(self) -> Dict[str, Any]:
        if self._summary is None:
            p = self.trial_dir / "summary.json"
            self._summary = json.loads(p.read_text()) if p.exists() else {}
        return self._summary

    @property
    def model_info(self) -> Dict[str, Any]:
        if self._info is None:
            p = self.trial_dir / "model_info.json"
            self._info = json.loads(p.read_text()) if p.exists() else {}
        return self._info

    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        if self._state is None:
            p = self.trial_dir / "model_state.pt"
            if not p.exists():
                raise FileNotFoundError(f"model_state.pt not found in {self.trial_dir}")
            self._state = torch.load(p, map_location=self.device)
        return self._state

    # -----------------------------------------------------------------
    # Model reconstruction
    # -----------------------------------------------------------------

    @property
    def model(self) -> ShortRateModel:
        """Reconstruct the model architecture and load the trained weights."""
        if self._model is not None:
            return self._model

        info = self.model_info
        state = self.state_dict

        latent_dim = info.get("latent_dim")
        noise_dim = info.get("noise_dim")

        enc_raw = info.get("encoder_config") or info.get("encoder_cfg") or {}
        nsde_raw = info.get("nsde_config") or info.get("nsde_cfg") or {}
        dec_raw = info.get("decoder_config")

        enc_cfg = EncoderCfg(**_filter_to_fields(EncoderCfg, enc_raw))
        nsde_cfg = NSDECfg(**_filter_to_fields(NSDECfg, nsde_raw))
        decoder = dec_raw if (isinstance(dec_raw, dict) and dec_raw.get("type")) else None

        # BondNet: prefer saved config, else infer from state_dict.
        bondnet_cfg = None
        bn_raw = info.get("bondnet_config")
        if isinstance(bn_raw, dict):
            cls = FiLMBondNetCfg if info.get("bondnet_class") == "FiLMBondNet" else SimpleBondNetCfg
            try:
                bondnet_cfg = cls(**_filter_to_fields(cls, bn_raw))
            except Exception:
                bondnet_cfg = None
        if bondnet_cfg is None:
            bondnet_cfg = _infer_bondnet_cfg(
                state,
                activation=self.bondnet_activation,
                output_positive=self.bondnet_output_positive,
            )

        # CRITICAL: pass the encoder input width so the lazily-built
        # recurrent encoder materialises BEFORE we load weights. Without this
        # the encoder's trained parameters load as "unexpected keys" and the
        # analysis silently runs on a randomly-initialised encoder.
        input_dim = _infer_encoder_input_dim(state)

        model = ShortRateModel.from_dicts(
            name=str(info.get("name", "reloaded")),
            encoder=enc_cfg,
            nsde=nsde_cfg,
            decoder=decoder,
            bondnet=bondnet_cfg,
            latent_dim=latent_dim,
            input_dim=input_dim,
            noise_dim=noise_dim,
        )

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[TrialAnalyzer] WARNING: {len(missing)} missing keys on load "
                  f"(first few: {list(missing)[:4]})")
        if unexpected:
            print(f"[TrialAnalyzer] WARNING: {len(unexpected)} unexpected keys on load "
                  f"(first few: {list(unexpected)[:4]})")
        if not missing and not unexpected:
            print(f"[TrialAnalyzer] OK: all {len(state)} parameter tensors loaded "
                  f"(encoder input_dim={input_dim}).")

        model.to(self.device).eval()
        self._model = model
        return model

    def _get_trainer(self):
        """Lazily build a read-only Trainer to drive the correct forward path."""
        if self._trainer is not None:
            return self._trainer
        if self.dataloader is None:
            raise RuntimeError(
                "This analysis needs market data. Pass a MarketDataLoader as "
                "`dataloader=` when constructing TrialAnalyzer."
            )
        from ..training.trainer import Trainer

        cfg = TrainerCfg()
        ti = self.model_info.get("training_info", {}) or {}
        for k in ("n_paths", "batch_window", "window_step", "lookback", "lookback_freq"):
            if ti.get(k) is not None:
                setattr(cfg, k, ti[k])
        if ti.get("dt") is not None:
            cfg.dt = ti["dt"]
        cfg.loss_weights.yield_curve = self.loss_weights[0]
        cfg.loss_weights.short_rate = self.loss_weights[1]
        cfg.loss_weights.futures = self.loss_weights[2]
        cfg.results_root = str(self._ensure_out())
        cfg.run_name = "trainer_readonly"
        cfg.use_amp = False
        cfg.early_stopping.enabled = False

        self._trainer = Trainer(model=self.model, dataloader=self.dataloader, config=cfg)
        return self._trainer

    # -----------------------------------------------------------------
    # Static: text + tables
    # -----------------------------------------------------------------

    def summary(self) -> str:
        info = self.model_info
        s = self.summary_json
        lines: List[str] = ["=== Trial Summary ===", f"trial_dir: {self.trial_dir}"]

        lines.append("> Config")
        lines.append(f"  - name:        {info.get('name')}")
        lines.append(f"  - encoder:     {info.get('encoder_type')}")
        lines.append(f"  - nsde:        {info.get('nsde_type')}")
        lines.append(f"  - latent_dim:  {info.get('latent_dim')}")
        if s.get("params"):
            lines.append(f"  - grid params: {s['params']}")

        if s.get("value") is not None:
            lines.append(f"> Eval value (objective): {float(s['value']):.6f}")

        ep = s.get("epoch_avgs") or []
        if ep:
            lines.append("> Training loss")
            lines.append(f"  - epochs: {len(ep)}  first={ep[0]:.6f}  last={ep[-1]:.6f}  min={min(ep):.6f}")

        ev = s.get("eval_losses") or {}
        if ev:
            vals = list(ev.values())
            lines.append("> Eval losses")
            lines.append(f"  - dates: {len(ev)}  mean={np.mean(vals):.6f}  min={np.min(vals):.6f}")

        # Param totals
        try:
            tot = sum(int(np.prod(v.shape)) for v in self.state_dict.values())
            lines.append(f"> Parameters: {tot:,} total tensors={len(self.state_dict)}")
        except Exception:
            pass

        return "\n".join(lines)

    def architecture_table(self) -> pd.DataFrame:
        """Per top-level sub-network: tensor count + parameter count."""
        groups: Dict[str, Dict[str, int]] = {}
        for k, v in self.state_dict.items():
            top = k.split(".")[0]
            g = groups.setdefault(top, {"tensors": 0, "params": 0})
            g["tensors"] += 1
            g["params"] += int(np.prod(v.shape))
        rows = [{"module": m, **d} for m, d in groups.items()]
        df = pd.DataFrame(rows).sort_values("params", ascending=False).reset_index(drop=True)
        total = df["params"].sum()
        df["param_share_%"] = (100.0 * df["params"] / max(1, total)).round(2)
        return df

    def parameter_stats(self) -> pd.DataFrame:
        """Per-tensor weight statistics — the static health check of the net."""
        rows = []
        for k, v in self.state_dict.items():
            t = v.detach().float()
            rows.append({
                "param": k,
                "shape": tuple(v.shape),
                "count": int(np.prod(v.shape)),
                "norm": float(t.norm().item()),
                "mean": float(t.mean().item()),
                "std": float(t.std().item()) if t.numel() > 1 else 0.0,
                "min": float(t.min().item()),
                "max": float(t.max().item()),
                "finite": bool(torch.isfinite(t).all().item()),
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Static: plots
    # -----------------------------------------------------------------

    def plot_training_curve(self, *, show: bool = False) -> Optional[str]:
        ep = self.summary_json.get("epoch_avgs") or []
        if not ep:
            return None
        fig, ax = plt.subplots()
        ax.plot(range(1, len(ep) + 1), ep, marker="o", ms=3)
        ax.set(title="Training loss per epoch", xlabel="epoch", ylabel="avg loss")
        ax.grid(True, alpha=0.3)
        return self._finish(fig, "training_curve.png", show)

    def plot_eval_losses(self, *, show: bool = False) -> Optional[str]:
        ev = self.summary_json.get("eval_losses") or {}
        if not ev:
            return None
        dates = [pd.Timestamp(d) for d in ev.keys()]
        vals = list(ev.values())
        order = np.argsort(dates)
        fig, ax = plt.subplots()
        ax.plot([dates[i] for i in order], [vals[i] for i in order], marker="o", ms=3)
        ax.set(title="Held-out eval loss", xlabel="date", ylabel="total_loss")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        return self._finish(fig, "eval_losses.png", show)

    def plot_weight_distributions(self, *, show: bool = False) -> Optional[str]:
        """Histogram of weights for each top-level sub-network."""
        groups: Dict[str, List[np.ndarray]] = {}
        for k, v in self.state_dict.items():
            if k.endswith(".weight"):
                groups.setdefault(k.split(".")[0], []).append(v.detach().float().cpu().numpy().ravel())
        if not groups:
            return None
        n = len(groups)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
        for ax, (name, arrs) in zip(axes.ravel(), groups.items()):
            w = np.concatenate(arrs)
            ax.hist(w, bins=80, color="steelblue", alpha=0.8)
            ax.set(title=f"{name}  (μ={w.mean():.2e}, σ={w.std():.2e})", yscale="log")
            ax.grid(True, alpha=0.3)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Weight distributions by sub-network")
        return self._finish(fig, "weight_distributions.png", show)

    def plot_weight_matrix_3d(self, param_name: str, *, show: bool = False) -> Optional[str]:
        """3-D surface of a single 2-D weight matrix (|value| over (out, in))."""
        if param_name not in self.state_dict:
            cands = [k for k in self.state_dict if param_name in k and k.endswith(".weight")]
            if not cands:
                raise KeyError(f"No weight matching '{param_name}'.")
            param_name = cands[0]
        w = self.state_dict[param_name].detach().float().cpu().numpy()
        if w.ndim != 2:
            raise ValueError(f"'{param_name}' is not a 2-D matrix (shape {w.shape}).")
        yy, xx = np.meshgrid(np.arange(w.shape[1]), np.arange(w.shape[0]))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(xx, yy, w, cmap="viridis", linewidth=0, antialiased=True)
        ax.set(title=f"Weight surface: {param_name}", xlabel="out", ylabel="in", zlabel="value")
        safe = param_name.replace(".", "_")
        return self._finish(fig, f"weight3d_{safe}.png", show)

    # -----------------------------------------------------------------
    # Dynamic: simulation
    # -----------------------------------------------------------------

    def simulate(self, date: Date, *, n_paths: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Run the model forward at ``date`` and return Monte-Carlo paths.

        Returns
        -------
        dict with keys: ``ts`` (T,), ``short_rate`` (n_paths, T),
        ``latent`` (n_paths, T, d_z).
        """
        tr = self._get_trainer()
        date = pd.Timestamp(date)
        with torch.no_grad():
            snap = tr._get_snapshot(date)
            ts = tr._make_ts(snap)
            latent = tr.get_latent_representation_from_date(date, n_paths=n_paths, ts=ts)
            r0 = tr._get_r0(date)
            short_rate = tr._decode(latent, r0=r0)
        sr = short_rate.squeeze(-1) if short_rate.dim() == 3 else short_rate
        return {
            "ts": ts.detach().cpu().numpy(),
            "short_rate": sr.detach().cpu().numpy(),
            "latent": latent.detach().cpu().numpy(),
        }

    def plot_short_rate_fan(self, date: Date, *, n_paths: Optional[int] = None,
                            show: bool = False) -> Optional[str]:
        """Short-rate fan chart: median + percentile bands across MC paths."""
        sim = self.simulate(date, n_paths=n_paths)
        ts, sr = sim["ts"], sim["short_rate"]          # sr: (paths, T)
        pct = {p: np.percentile(sr, p, axis=0) for p in (5, 25, 50, 75, 95)}
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(ts, pct[5] * 100, pct[95] * 100, alpha=0.2, color="navy", label="5–95%")
        ax.fill_between(ts, pct[25] * 100, pct[75] * 100, alpha=0.35, color="navy", label="25–75%")
        ax.plot(ts, pct[50] * 100, color="black", lw=1.5, label="median")
        ax.set(title=f"Short-rate fan chart @ {pd.Timestamp(date).date()}",
               xlabel="years ahead", ylabel="short rate (%)")
        ax.legend(); ax.grid(True, alpha=0.3)
        return self._finish(fig, f"short_rate_fan_{pd.Timestamp(date).date()}.png", show)

    def plot_latent_surface_3d(self, date: Date, *, n_paths: Optional[int] = None,
                               show: bool = False) -> Optional[str]:
        """3-D surface of the mean latent state over (time × latent dimension)."""
        sim = self.simulate(date, n_paths=n_paths)
        ts, z = sim["ts"], sim["latent"]                # z: (paths, T, d_z)
        zbar = z.mean(axis=0)                           # (T, d_z)
        dd, tt = np.meshgrid(np.arange(zbar.shape[1]), ts)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(tt, dd, zbar, cmap="coolwarm", linewidth=0, antialiased=True)
        ax.set(title=f"Mean latent state @ {pd.Timestamp(date).date()}",
               xlabel="years ahead", ylabel="latent dim", zlabel="E[z]")
        return self._finish(fig, f"latent_surface_{pd.Timestamp(date).date()}.png", show)

    # -----------------------------------------------------------------
    # Dynamic: yields
    # -----------------------------------------------------------------

    def yield_curve_table(self, date: Date, *, n_paths: Optional[int] = None) -> pd.DataFrame:
        """Model vs market yields (in %) at ``date`` with the per-pillar error."""
        tr = self._get_trainer()
        date = pd.Timestamp(date)
        with torch.no_grad():
            snap = tr._get_snapshot(date)
            ts = tr._make_ts(snap)
            latent = tr.get_latent_representation_from_date(date, n_paths=n_paths, ts=ts)
            realis = tr._decode(latent, r0=tr._get_r0(date))
            mats = snap.yield_curve.maturities
            model_y = tr.pricer.price_yield_curve(realisations=realis, maturities=mats)
            mkt_y = snap.yield_curve.yields
        mats = mats.detach().cpu().numpy()
        model_y = model_y.detach().cpu().numpy() * 100.0
        mkt_y = mkt_y.detach().cpu().numpy() * 100.0
        return pd.DataFrame({
            "maturity_y": mats,
            "model_%": model_y,
            "market_%": mkt_y,
            "error_bp": (model_y - mkt_y) * 100.0,
        })

    def plot_yield_surface_3d(self, dates: Sequence[Date], *, n_paths: Optional[int] = None,
                              show: bool = False) -> Optional[str]:
        """
        Three side-by-side 3-D surfaces over (maturity × date): model yields,
        market yields, and the model−market error (bp).
        """
        tr = self._get_trainer()
        dts = [pd.Timestamp(d) for d in dates]
        model_rows, mkt_rows, mats_ref = [], [], None
        with torch.no_grad():
            for d in dts:
                snap = tr._get_snapshot(d)
                ts = tr._make_ts(snap)
                latent = tr.get_latent_representation_from_date(d, n_paths=n_paths, ts=ts)
                realis = tr._decode(latent, r0=tr._get_r0(d))
                mats = snap.yield_curve.maturities
                my = tr.pricer.price_yield_curve(realisations=realis, maturities=mats)
                model_rows.append(my.detach().cpu().numpy() * 100.0)
                mkt_rows.append(snap.yield_curve.yields.detach().cpu().numpy() * 100.0)
                mats_ref = mats.detach().cpu().numpy()
        M = np.array(model_rows)                         # (D, P)
        K = np.array(mkt_rows)
        E = (M - K) * 100.0                              # bp
        di = np.arange(len(dts))
        xx, yy = np.meshgrid(mats_ref, di)               # (D, P)

        fig = plt.figure(figsize=(16, 5))
        for i, (Z, title, cmap) in enumerate(
            [(M, "Model yields (%)", "viridis"),
             (K, "Market yields (%)", "plasma"),
             (E, "Error (bp)", "coolwarm")], start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            ax.plot_surface(xx, yy, Z, cmap=cmap, linewidth=0, antialiased=True)
            ax.set(title=title, xlabel="maturity (y)", ylabel="date idx", zlabel="")
        fig.suptitle("Yield surface: model vs market")
        return self._finish(fig, "yield_surface_3d.png", show)

    # -----------------------------------------------------------------
    # Dynamic: gradients
    # -----------------------------------------------------------------

    def gradient_report(self, date: Date, *, n_paths: Optional[int] = None,
                        make_plot: bool = True, show: bool = False
                        ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Forward + backward one date and report per-parameter gradient norms.

        This is the "are gradients flowing / vanishing / exploding" check:
        a healthy net shows comparable grad norms across layers; vanishing
        shows orders-of-magnitude smaller grads in the encoder/early layers;
        exploding shows huge norms (or non-finite) in the SDE / BondNet.
        """
        tr = self._get_trainer()
        date = pd.Timestamp(date)
        model = self.model

        model.train()
        model.zero_grad(set_to_none=True)
        loss, components = tr._forward_one_date(date, n_paths=n_paths, return_components=True)
        loss.backward()

        rows = []
        for name, p in model.named_parameters():
            g = p.grad
            rows.append({
                "param": name,
                "module": name.split(".")[0],
                "count": int(p.numel()),
                "grad_norm": float(g.norm().item()) if g is not None else np.nan,
                "grad_absmax": float(g.abs().max().item()) if g is not None else np.nan,
                "grad_finite": bool(torch.isfinite(g).all().item()) if g is not None else False,
                "has_grad": g is not None,
            })
        df = pd.DataFrame(rows)
        model.zero_grad(set_to_none=True)
        model.eval()

        df.attrs["loss"] = float(loss.detach().cpu().item())
        df.attrs["components"] = {k: float(v) for k, v in components.items()}

        out = None
        if make_plot and not df.empty:
            agg = (df.dropna(subset=["grad_norm"])
                     .groupby("module")["grad_norm"].agg(["sum", "max", "mean"])
                     .sort_values("sum", ascending=False))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(agg.index, agg["sum"], color="indianred", alpha=0.85)
            ax.set(title=f"Gradient norm by sub-network @ {date.date()} "
                         f"(loss={df.attrs['loss']:.4g})",
                   ylabel="Σ grad norm", yscale="log")
            ax.grid(True, axis="y", alpha=0.3)
            out = self._finish(fig, f"gradient_flow_{date.date()}.png", show)
        return df, out

    # -----------------------------------------------------------------
    # Dynamic: futures / CTD
    # -----------------------------------------------------------------

    def futures_report(self, date: Date, *, n_paths: Optional[int] = None) -> Dict[str, Any]:
        """
        Price the futures at ``date`` and report BondNet output statistics and
        the cheapest-to-deliver selection frequency per basket slot. Returns
        ``{}`` if the snapshot has no futures.
        """
        tr = self._get_trainer()
        date = pd.Timestamp(date)
        with torch.no_grad():
            snap = tr._get_snapshot(date)
            if snap.futures is None:
                return {}
            ts = tr._make_ts(snap)
            latent = tr.get_latent_representation_from_date(date, n_paths=n_paths, ts=ts)
            realis = tr._decode(latent, r0=tr._get_r0(date))
            model_snap = tr.pricer.price_snapshot(
                realisations=realis, snapshot=snap,
                latent_paths=latent, simulated_times=ts, bondnet=tr.model.bondnet,
            )
        model_f = model_snap.futures.prices.detach().cpu().numpy()
        mkt_f = snap.futures.prices.detach().cpu().numpy()
        ctd = tr.pricer.last_ctd_freq
        return {
            "tickers": list(snap.futures.ids),
            "model_price": model_f,
            "market_price": mkt_f,
            "error": model_f - mkt_f,
            "bond_value_stats": tr.pricer.last_bond_stats,
            "ctd_freq": None if ctd is None else ctd.detach().cpu().numpy(),
        }

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------

    def run_all(
        self,
        *,
        sample_date: Optional[Date] = None,
        surface_dates: Optional[Sequence[Date]] = None,
        n_paths: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Produce the full analysis bundle into ``trial_XXX/analysis_v2/``.

        Static plots/tables always run. Dynamic ones run only if a dataloader
        was supplied; ``sample_date`` (and ``surface_dates`` for the yield
        surface) pick which date(s) to analyze — defaults to the eval dates
        recorded in ``summary.json``.
        """
        out = self._ensure_out()
        produced: Dict[str, Any] = {"dir": str(out), "files": [], "tables": []}

        # ---- text + tables ----
        (out / "summary.txt").write_text(self.summary())
        self.architecture_table().to_csv(out / "architecture.csv", index=False)
        pstats = self.parameter_stats()
        pstats.to_csv(out / "parameter_stats.csv", index=False)
        produced["tables"] += ["architecture.csv", "parameter_stats.csv"]
        non_finite = pstats.loc[~pstats["finite"], "param"].tolist()
        if non_finite:
            print(f"[TrialAnalyzer] WARNING: non-finite weights in: {non_finite}")

        # ---- static plots ----
        for fn in (self.plot_training_curve, self.plot_eval_losses, self.plot_weight_distributions):
            try:
                p = fn()
                if p:
                    produced["files"].append(p)
            except Exception as e:
                print(f"[TrialAnalyzer] {fn.__name__} failed: {e}")

        # a representative 3-D weight surface (largest 2-D matrix)
        try:
            mats = [(k, v) for k, v in self.state_dict.items()
                    if k.endswith(".weight") and v.ndim == 2]
            if mats:
                biggest = max(mats, key=lambda kv: int(np.prod(kv[1].shape)))[0]
                produced["files"].append(self.plot_weight_matrix_3d(biggest))
        except Exception as e:
            print(f"[TrialAnalyzer] weight_matrix_3d failed: {e}")

        # ---- dynamic ----
        if self.dataloader is not None:
            ev_dates = list((self.summary_json.get("eval_losses") or {}).keys())
            sd = sample_date or (ev_dates[0] if ev_dates else None)
            if sd is not None:
                for fn, args in [
                    (self.plot_short_rate_fan, (sd,)),
                    (self.plot_latent_surface_3d, (sd,)),
                ]:
                    try:
                        produced["files"].append(fn(*args, n_paths=n_paths))
                    except Exception as e:
                        print(f"[TrialAnalyzer] {fn.__name__} failed: {e}")
                try:
                    self.yield_curve_table(sd, n_paths=n_paths).to_csv(
                        out / f"yields_{pd.Timestamp(sd).date()}.csv", index=False)
                    produced["tables"].append(f"yields_{pd.Timestamp(sd).date()}.csv")
                except Exception as e:
                    print(f"[TrialAnalyzer] yield_curve_table failed: {e}")
                try:
                    gdf, gpng = self.gradient_report(sd, n_paths=n_paths)
                    gdf.to_csv(out / f"gradients_{pd.Timestamp(sd).date()}.csv", index=False)
                    produced["tables"].append(f"gradients_{pd.Timestamp(sd).date()}.csv")
                    if gpng:
                        produced["files"].append(gpng)
                except Exception as e:
                    print(f"[TrialAnalyzer] gradient_report failed: {e}")
                try:
                    fr = self.futures_report(sd, n_paths=n_paths)
                    if fr:
                        pd.DataFrame({
                            "ticker": fr["tickers"],
                            "model": fr["model_price"],
                            "market": fr["market_price"],
                            "error": fr["error"],
                        }).to_csv(out / f"futures_{pd.Timestamp(sd).date()}.csv", index=False)
                        produced["tables"].append(f"futures_{pd.Timestamp(sd).date()}.csv")
                except Exception as e:
                    print(f"[TrialAnalyzer] futures_report failed: {e}")

            sds = surface_dates or ev_dates
            if sds and len(sds) >= 2:
                try:
                    produced["files"].append(self.plot_yield_surface_3d(sds, n_paths=n_paths))
                except Exception as e:
                    print(f"[TrialAnalyzer] yield_surface_3d failed: {e}")

        print(f"[TrialAnalyzer] wrote {len(produced['files'])} plots, "
              f"{len(produced['tables'])} tables to {out}")
        return produced

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------

    def _finish(self, fig, fname: str, show: bool) -> str:
        path = str(self._ensure_out() / fname)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path


# =====================================================================
# CLI
# =====================================================================

def _build_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Analyze a grid-search trial folder.")
    ap.add_argument("trial_dir", help="Path to a trial_XXX/ folder.")
    ap.add_argument("--data-path", default=None,
                    help="data2 path; enables the dynamic (data-driven) analyses.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-paths", type=int, default=None)
    ap.add_argument("--sample-date", default=None)
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_cli().parse_args(argv)

    dataloader = None
    if args.data_path:
        # Build a minimal loader spanning the trial's training window.
        from ..dataloaders.market_loader import MarketDataLoader
        from ..configs import DataLoaderCfg
        info = json.loads((Path(args.trial_dir) / "model_info.json").read_text())
        ti = info.get("training_info", {}) or {}
        start = ti.get("start_training_date")
        end = ti.get("end_training_date")
        dl_cfg = DataLoaderCfg(
            data_path=args.data_path,
            start_date=(pd.Timestamp(start) - pd.Timedelta(days=400)) if start else None,
            end_date=(pd.Timestamp(end) + pd.Timedelta(days=120)) if end else None,
            max_maturity=10,
            enable_yield=True, enable_short_rate=True, enable_futures=True,
            device=args.device,
        )
        dataloader = MarketDataLoader(dl_cfg)

    az = TrialAnalyzer(args.trial_dir, dataloader=dataloader, device=args.device)
    print(az.summary())
    az.run_all(sample_date=args.sample_date, n_paths=args.n_paths)


if __name__ == "__main__":
    main()
