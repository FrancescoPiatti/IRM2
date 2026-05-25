# src/configs/config_loader.py
from dataclasses import dataclass

from typing import Optional
from typing import Union

import torch

from ..types import Date
from ..utils.checks import _check_positive_integer_value
from ..utils.checks import _check_positive_value


@dataclass
class DataLoaderCfg:
    """
    Data loader configuration.

    Attributes
    ----------
    data_path : str
        Root path to the dataset (folder containing the CSVs).
    start_date : Optional[Date]
        Inclusive start date of the usable calendar. If None, uses the earliest
        available date in the data source.
    end_date : Optional[Date]
        Inclusive end date of the usable calendar. If None, uses the latest
        available date in the data source.
    max_maturity : int
        Maximum maturity (in years) used to define the simulation horizon and
        the curve grid (e.g., 30 means up to 30Y).
    business_days_per_year : float
        Year-fraction convention used across the loader and pricer (252.0 by
        default — matches BondMetadataStore and the typical Treasury convention).
        Use 365.25 for a calendar-day convention.
    enable_yield : bool
        If True, yield curve targets are loaded / exposed in snapshots.
    enable_short_rate : bool
        If True, short-rate targets are loaded / exposed in snapshots.
    enable_bonds : bool
        If True, bond instruments (and their targets) are loaded / exposed.
    enable_futures : bool
        If True, futures instruments (and their targets) are loaded / exposed.
        Implies that bond metadata is loaded (needed for deliverable features).
    enable_options : bool
        If True, option instruments (and their targets) are loaded / exposed.
    device : Union[str, torch.device]
        Device used by the dataloader to place returned tensors.
    dtype : Union[str, torch.dtype]
        Default dtype for returned tensors. Pass either a `torch.dtype` or the
        attribute name as a string (e.g., 'float32').
    """

    # Data sources
    data_path: str

    # Date range / curve shape
    start_date: Optional[Date] = None
    end_date: Optional[Date] = None
    max_maturity: int = 30

    # Market conventions
    business_days_per_year: float = 252.0

    # Enabled targets
    enable_yield: bool = True
    enable_short_rate: bool = False
    enable_bonds: bool = False
    enable_futures: bool = False
    enable_options: bool = False

    # Device / dtype
    device: Union[str, torch.device] = torch.device("cpu")
    dtype: Union[str, torch.dtype] = torch.float32


    # -------------------------
    # Validation
    # -------------------------

    def validate(self) -> None:
        """
        Validate data loader configuration values.
        """
        _check_positive_integer_value(self.max_maturity, 'max_maturity')
        _check_positive_value(self.business_days_per_year, 'business_days_per_year')

        if not any([
            self.enable_yield,
            self.enable_short_rate,
            self.enable_bonds,
            self.enable_futures,
            self.enable_options
            ]):

            raise ValueError(
                "DataLoaderCfg: no targets enabled (enable_yield, enable_short_rate, etc. are all False)."
                )
