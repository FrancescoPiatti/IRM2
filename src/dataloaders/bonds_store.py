# src/dataloaders/bonds_store.py
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd
import torch

from ..types import Date

@dataclass
class BondStore:
    """
    Placeholder bond store.

    The design is SOTA-ready:
    - prices_df: indexed by date, columns are bond IDs (or ISINs)
    - meta_df: indexed by bond ID (coupon, maturity, daycount, etc.)
    """
    prices_df: Optional[pd.DataFrame] = None
    meta_df: Optional[pd.DataFrame] = None
    device: torch.device = torch.device("cpu")

    def available_on(self, date: Date) -> bool:
        if self.prices_df is None:
            return False
        ts = pd.Timestamp(date)
        return ts in self.prices_df.index
    

    def get_bond_targets(self, date: Date, *, device: Optional[torch.device] = None) -> Optional[Dict[str, Any]]:
        """
        Return bond targets and metadata. If not available, return None.

        Output dict structure is intentionally flexible:
        {
            "ids": list[str],
            "prices": Tensor[N],
            "metadata": {...}
        }
        """
        if self.prices_df is None:
            return None

        ts = pd.Timestamp(date)
        if ts not in self.prices_df.index:
            return None

        dev = device if device is not None else self.device

        row = self.prices_df.loc[ts].dropna()
        if row.empty:
            return None

        ids = list(row.index)
        prices = torch.as_tensor(row.values, dtype=torch.float32, device=dev)

        metadata: Dict[str, Any] = {}
        if self.meta_df is not None:
            # return only metadata for bonds in ids
            meta = self.meta_df.loc[self.meta_df.index.intersection(ids)]
            metadata["meta_df"] = meta.to_dict(orient="index")

        return {"ids": ids, "prices": prices, "metadata": metadata}