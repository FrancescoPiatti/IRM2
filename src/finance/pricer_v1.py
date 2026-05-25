"""
Legacy yield-only pricer kept for reference.

The canonical pricer used by `Trainer` is `src.finance.pricer_v2.Pricer`. This
module remains as documentation of the original API and is not exercised by the
test suite.
"""
import torch
from torch import Tensor
from typing import Optional
from typing import List


class Pricer:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cpu')

    @staticmethod
    def compute_yield_from_realisations(realisations: Tensor, maturities: List[int]) -> Tensor:

        # I think it should take also as input the dt. For now hard code it to be 1/252. I'll adjust it later in the whole library

        # Realisation is a tensor of shape (n_paths, longest_matuity_years * 252)
        # maturity_indexes are the indices of the maturities we want to price
        # You can come up with another logic if you find a better one

        """
        Get the prices of zero-coupon yields from the simulated paths.

        Args:
            realisations (Tensor): The simulated paths of the short rate - shape (n_paths, timesteps).
            maturities (List[int]): The indexes of the maturities of the bonds.

        Returns:
            Tensor: The prices of the zero-coupon bonds.
        """
        steps_per_year = 252
        dt = 1.0 / steps_per_year

        # map years → step indices 
        idx = (torch.as_tensor(maturities, device=realisations.device, dtype=torch.float32)
               * steps_per_year).long()

        # cumulative integrals
        cum_int = torch.cumsum(realisations, dim=1) * dt               # (paths, steps)

        # extract integrals at maturities
        integral = cum_int.index_select(1, idx)                        # (paths, n_mats)

        # average discounts across paths
        P = torch.exp(-integral).mean(dim=0).clamp(1e-12, 1.0)         # (n_mats,)

        # yields in percent
        y_percent = -100.0 * torch.log(P) / maturities

        return y_percent
    
    

    @staticmethod
    def compute_swap_prices(realisations : Tensor, maturities: List[int]) -> Tensor:
        """
        Get the prices of interest rate swaps from the simulated paths.

        Args:
            realisations (Tensor): The simulated paths of the short rate - shape (n_paths, timesteps).
            maturities (List[int]): The indexes of the maturities of the swaps.

        Returns:
            Tensor: The prices of the interest rate swaps.
        """
        # Implement the pricing logic here
        pass

    
    def get_prices(self, realisations: Tensor, maturities: List[int], instrument_type: str) -> Tensor:
        """
        Get the prices of financial instruments based on the type.

        Args:
            realisations (Tensor): The simulated paths of the short rate - shape (n_paths, timesteps).
            maturities (List[int]): The indexes of the maturities of the instruments.
            instrument_type (str): The type of financial instrument ('bond', 'swap', etc.).

        Returns:
            Tensor: The prices of the specified financial instrument.
        """
        if instrument_type == 'yield':
            return self.compute_yield_from_realisations(realisations, maturities)
        elif instrument_type == 'swap':
            return self.compute_swap_prices(realisations, maturities)
        else:
            raise ValueError(f"Unsupported instrument type: {instrument_type}")
