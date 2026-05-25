# src/analysis/result_analyzer
import json
from dataclasses import dataclass
from pathlib import Path

from typing import Any
from typing import Dict
from typing import Optional
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ResultAnalyzer:
    """
    Load run artifacts and produce quick summaries and plots.

    Expected run structure:
    -----------------------
    run_dir/
        model_info.json          (or similar)
        losses.csv               (epoch losses)
        eval/
            eval_YYYY-MM-DD.csv
            eval_YYYY-MM-DD_YYYY-MM-DD_stepK.csv
        plots/
            *.png
    """

    run_dir: Union[str, Path]

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {self.run_dir}")

        # Path references only; the directories are created on demand when
        # something is actually saved. A read-only "analyzer" should not have
        # filesystem side effects in its constructor.
        self.eval_dir = self.run_dir / "eval"
        self.plot_dir = self.run_dir / "plots"

    def _ensure_plot_dir(self) -> Path:
        """Create ``plots/`` if not present and return the path."""
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        return self.plot_dir

    # ---------------------------------------------------------------------
    # IO helpers
    # ---------------------------------------------------------------------

    def load_model_info(self) -> Dict[str, Any]:
        """
        Load model metadata (JSON).
        """
        # if you always save exactly "model_info.json", keep it strict
        direct = self.run_dir / "model_info.json"
        if direct.exists():
            return json.loads(direct.read_text())

        # fallback: find any json containing model_info
        cand = list(self.run_dir.glob("*model_info*.json"))
        if len(cand) > 0:
            return json.loads(cand[0].read_text())

        return {}

    def load_epoch_losses(self) -> pd.DataFrame:
        """
        Load training epoch losses from losses.csv.

        Expected format: at least a column called "loss"
        Common formats:
          - epoch, loss
          - loss (single column)
        """
        p = self.run_dir / "losses.csv"
        if not p.exists():
            return pd.DataFrame()

        df = pd.read_csv(p)

        # normalize column names
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "loss" not in df.columns:
            # try single column fallback
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "loss"})
            else:
                raise ValueError(f"losses.csv does not contain a 'loss' column: columns={df.columns.tolist()}")

        if "epoch" not in df.columns:
            df.insert(0, "epoch", np.arange(1, len(df) + 1))

        return df

    def list_eval_csvs(self) -> List[Path]:
        """
        Return all eval CSV files sorted by name.
        """
        if not self.eval_dir.exists():
            return []
        return sorted(self.eval_dir.glob("*.csv"))


    def load_eval_csv(self, filename: str | Path) -> pd.DataFrame:
        """
        Load an eval CSV from run_dir/eval/.
        Accepts either:
        - "eval_2020-01-02.csv"
        - "results/.../eval/eval_2020-01-02.csv"
        """
        p = Path(filename)

        # If already a full path (absolute OR already includes run_dir), use it directly
        if p.is_absolute() or str(p).startswith(str(self.run_dir)):
            csv_path = p
        else:
            # Otherwise interpret it as a filename inside eval_dir
            csv_path = self.eval_dir / p

        if not csv_path.exists():
            raise FileNotFoundError(f"Eval CSV not found: {csv_path}")

        return pd.read_csv(csv_path, parse_dates=["date"])

    def load_all_eval(self) -> pd.DataFrame:
        """
        Load and merge all evaluation CSVs inside eval/.

        - concatenates
        - sorts by date
        - removes duplicates (keeps last)
        """
        files = self.list_eval_csvs()
        if len(files) == 0:
            return pd.DataFrame()

        dfs = [self.load_eval_csv(f) for f in files]
        out = pd.concat(dfs, axis=0, ignore_index=True)

        # If multiple rows same date, keep the last (latest file wins)
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        return out

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------

    def summary(self) -> str:
        info = self.load_model_info()
        losses = self.load_epoch_losses()
        eval_df = self.load_all_eval()

        lines: List[str] = []
        lines.append("=== Run Summary ===")
        lines.append(f"run_dir: {self.run_dir}")

        # Model info
        if info:
            lines.append("> Model")
            lines.append(f"  - name:         {info.get('name', None)}")
            lines.append(f"  - encoder_type: {info.get('encoder_type', None)}")
            lines.append(f"  - latent_dim:   {info.get('latent_dim', None)}")

            ti = info.get("training_info", None)
            if isinstance(ti, dict):
                lines.append("> Training")
                lines.append(f"  - start: {ti.get('start_training_date', None)}")
                lines.append(f"  - end:   {ti.get('end_training_date', None)}")
                lines.append(f"  - n_paths: {ti.get('n_paths', None)}")
                lines.append(f"  - optimizer: {ti.get('optimizer', None)}")
                lines.append(f"  - scheduler: {ti.get('scheduler', None)}")
                lines.append(f"  - early_stopping: {ti.get('early_stopping', None)}")
        else:
            lines.append("No model_info.json found.")

        # Losses
        if len(losses) > 0:
            lines.append("> Training losses")
            lines.append(f"  - epochs: {len(losses)}")
            lines.append(f"  - first:  {float(losses['loss'].iloc[0]):.6f}")
            lines.append(f"  - last:   {float(losses['loss'].iloc[-1]):.6f}")
            lines.append(f"  - min:    {float(losses['loss'].min()):.6f}")
        else:
            lines.append("No losses.csv found.")

        # Eval
        if len(eval_df) > 0:
            lines.append("> Eval results")
            lines.append(f"  - rows: {len(eval_df)}")
            lines.append(f"  - start: {eval_df['date'].min().date()}")
            lines.append(f"  - end:   {eval_df['date'].max().date()}")
            if "total_loss" in eval_df.columns:
                lines.append(f"  - total_loss min: {float(eval_df['total_loss'].min()):.6f}")
                lines.append(f"  - total_loss last:{float(eval_df['total_loss'].iloc[-1]):.6f}")
        else:
            lines.append("No eval/*.csv found.")

        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Plotting (matplotlib only)
    # ---------------------------------------------------------------------

    def plot_epoch_loss(self, *, out: Optional[str] = None, show: bool = False) -> Optional[str]:
        df = self.load_epoch_losses()
        if df.empty:
            return None

        plt.figure()
        plt.plot(df["epoch"].values, df["loss"].values)
        plt.title("Training epoch loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)

        if out is None:
            out = str(self._ensure_plot_dir() / "epoch_loss.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)

        if show:
            plt.show()
        plt.close()
        return out

    def plot_eval_total_loss(self, *, out: Optional[str] = None, show: bool = False) -> Optional[str]:
        df = self.load_all_eval()
        if df.empty or "total_loss" not in df.columns:
            return None

        plt.figure()
        plt.plot(df["date"].values, df["total_loss"].values)
        plt.title("Evaluation total loss")
        plt.xlabel("date")
        plt.ylabel("total_loss")
        plt.grid(True, alpha=0.3)

        if out is None:
            out = str(self._ensure_plot_dir() / "eval_total_loss.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)

        if show:
            plt.show()
        plt.close()
        return out

    def plot_eval_components(self, *, out: Optional[str] = None, show: bool = False) -> Optional[str]:
        """
        Plot all component columns if present.

        Convention:
        - columns other than ['date', 'total_loss'] are treated as components,
          e.g. yield, short_rate, etc.
        """
        df = self.load_all_eval()
        if df.empty:
            return None

        base_cols = {"date", "total_loss"}
        comp_cols = [c for c in df.columns if c not in base_cols]

        # nothing to plot
        if len(comp_cols) == 0:
            return None

        plt.figure()
        for c in comp_cols:
            # skip non-numeric columns
            if not np.issubdtype(df[c].dtype, np.number):
                continue
            plt.plot(df["date"].values, df[c].values, label=c)

        plt.title("Evaluation components")
        plt.xlabel("date")
        plt.ylabel("loss component")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)

        if out is None:
            out = str(self._ensure_plot_dir() / "eval_components.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)

        if show:
            plt.show()
        plt.close()
        return out