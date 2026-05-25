# src/types/eval_results_types.py
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

import pandas as pd
from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class EvalResults:
    """
    Evaluation output for one date.

    Notes
    -----
    - total_loss is a python float (safe to log/serialize)
    - components is a dict of per-instrument loss scalars
    - meta can store anything extra (e.g. n_paths used, warnings, etc.)
    """
    date: pd.Timestamp
    n_paths: int
    total_loss: float
    components: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)



def eval_results_to_frame(results: Union[EvalResults, Sequence[EvalResults]]) -> pd.DataFrame:
    """
    Convert EvalResults or Sequence[EvalResults] into a flat DataFrame.

    Columns
    -------
    - date
    - n_paths
    - total_loss
    - component/<name>   (flattened from EvalResults.components)
    """
    if isinstance(results, EvalResults):
        results = [results]

    rows: List[Dict[str, Any]] = []
    for r in results:
        row = {
            "date": pd.Timestamp(r.date),
            "n_paths": int(r.n_paths),
            "total_loss": float(r.total_loss),
        }

        comps = getattr(r, "components", None) or {}
        for k, v in comps.items():
            row[f"component/{k}"] = float(v)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df