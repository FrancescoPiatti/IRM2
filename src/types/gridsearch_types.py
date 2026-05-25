# src/types/gridsearch_types.py
from typing import Any
from typing import Dict
from typing import List

from dataclasses import dataclass
from dataclasses import asdict

import json


@dataclass(frozen=True)
class TrialResult:
    """
    Result summary for one Optuna trial.

    Attributes
    ----------
    number : int
        Trial number (Optuna assigned).
    value : float
        Objective value returned by the evaluation routine.
    params : Dict[str, Any]
        Parameter choices for this trial (flat dict using your dot-path keys).

        Note: values are DECODED back to the original python objects where applicable
        (e.g. dict configs for nsde.diffusion).
    """
    number: int
    value: float
    params: Dict[str, Any]


@dataclass(frozen=True)
class GridSearchResults:
    """
    Full grid-search output.

    Attributes
    ----------
    best_value : float
        Best objective value found.
    best_params : Dict[str, Any]
        Parameters for the best trial (decoded).
    trials : List[TrialResult]
        One entry per completed trial (decoded).
    """
    best_value: float
    best_params: Dict[str, Any]
    trials: List[TrialResult]

    def to_json(self) -> str:
        """Serialize results to JSON (safe to write to disk)."""
        payload = {
            "best_value": float(self.best_value),
            "best_params": dict(self.best_params),
            "trials": [asdict(t) for t in self.trials],
        }
        return json.dumps(payload, indent=2, default=str)