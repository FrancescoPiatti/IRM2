# src/types/types_utils.py
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Union

from datetime import datetime
import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.types.data_types import SingleFutureTarget
    from src.types.data_types import BatchedFuturesTarget


Date = Union[str, datetime, pd.Timestamp]


def normalize_date(d: Date) -> pd.Timestamp:
    """
    Convert any accepted date type to a normalised pd.Timestamp.

    Parameters
    ----------
    d : Date
        Date in any accepted form.

    Returns
    -------
    pd.Timestamp
        Timestamp normalised to midnight.
    """
    return pd.Timestamp(d).normalize()


def merge_single_future_targets(
    targets: List["SingleFutureTarget"],
    *,
    device: Optional[torch.device] = None,
) -> "BatchedFuturesTarget":
    """
    Merge a list of `SingleFutureTarget` (all at the same as-of date)
    into a single `BatchedFuturesTarget` using a flattened-ragged layout.

    Parameters
    ----------
    targets : List[SingleFutureTarget]
        Non-empty list of single-future targets sharing the same `date`.
    device : Optional[torch.device]
        Device for the output tensors. Defaults to the device of
        `targets[0].price`.

    Returns
    -------
    BatchedFuturesTarget
        Batched representation suitable for vectorised pricing.

    Raises
    ------
    ValueError
        If `targets` is empty, or any target has a different as-of date,
        or basket / conversion-factor shapes are inconsistent.
    """
    from src.types.data_types import BatchedFuturesTarget  # noqa: deferred to break circular import

    if len(targets) == 0:
        raise ValueError("merge_single_future_targets: empty target list.")

    asof_date = pd.Timestamp(targets[0].date)
    for t in targets:
        if pd.Timestamp(t.date) != asof_date:
            raise ValueError("All SingleFutureTarget objects must share the same as-of date.")

    dev = device if device is not None else targets[0].price.device

    ids: List[str] = []
    delivery_dates: List[pd.Timestamp] = []
    prices_list: List[Tensor] = []

    basket_lengths_list: List[int] = []
    cf_list: List[Tensor] = []
    deliverable_ids_flat: List[str] = []

    for t in targets:
        ids.append(t.id)
        delivery_dates.append(pd.Timestamp(t.delivery_date))

        price = t.price.to(dev)
        if price.ndim == 0:
            price = price.unsqueeze(0)
        elif price.numel() != 1:
            raise ValueError(f"SingleFutureTarget.price for {t.id} must be scalar-like.")
        prices_list.append(price.reshape(1))

        cf = t.conversion_factors.to(dev)
        if cf.ndim != 1:
            raise ValueError(f"SingleFutureTarget.conversion_factors for {t.id} must be 1D.")

        if len(t.deliverable_ids) != cf.shape[0]:
            raise ValueError(
                f"Mismatch for {t.id}: len(deliverable_ids)={len(t.deliverable_ids)} "
                f"but conversion_factors.shape[0]={cf.shape[0]}"
            )

        basket_lengths_list.append(cf.shape[0])
        cf_list.append(cf)
        deliverable_ids_flat.extend(t.deliverable_ids)

    prices = torch.cat(prices_list, dim=0)
    basket_lengths = torch.as_tensor(basket_lengths_list, dtype=torch.long, device=dev)
    conversion_factors_flat = torch.cat(cf_list, dim=0)

    return BatchedFuturesTarget(
        ids=ids,
        prices=prices,
        asof_date=asof_date,
        delivery_dates=delivery_dates,
        basket_lengths=basket_lengths,
        conversion_factors_flat=conversion_factors_flat,
        deliverable_ids_flat=deliverable_ids_flat,
        metadata={},
    )
