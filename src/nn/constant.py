# src/nn/constant.py
import torch
from torch import nn
from torch import Tensor
from torch.nn import Identity
from torch.nn import Module

from typing import Union

from .activations import _get_activation


class ConstantNet(Module):
    """
    Constant network: learns a single output vector and repeats it across the batch.

    Parameters
    ----------
    out_features : int
        Output dimensionality.
    out_activation : str or nn.Module, default Identity
        Activation applied to the constant output.
    init : str, default "zeros"
        Initialization for the constant parameter. Supported: {"zeros", "normal"}.
    init_std : float, default 0.01
        Std used when init="normal".
    """

    def __init__(
        self,
        out_features: int,
        out_activation: Union[str, Module] = Identity,
        init: str = "zeros",
        init_std: float = 0.01,
    ):
        super().__init__()

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError("out_features must be a positive integer.")

        self.out_features = int(out_features)
        self.out_activation = _get_activation(out_activation)

        self.value = nn.Parameter(torch.empty(self.out_features))

        init = str(init).lower()
        if init == "zeros":
            nn.init.zeros_(self.value)
        elif init == "normal":
            nn.init.normal_(self.value, mean=0.0, std=float(init_std))
        else:
            raise ValueError("init must be one of {'zeros', 'normal'}.")


    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor. The values are ignored, but the leading batch dimension
            (if present) is used to determine how many copies to return.

        Returns
        -------
        Tensor
            Tensor of shape (B, out_features) where B is inferred from x.
            If x is 1D, returns shape (1, out_features).
        """
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor.")

        batch = 1 if x.dim() == 1 else int(x.shape[0]) # x.shape[0] if x.dim() > 1 else 1
        out = self.value.unsqueeze(0).expand(batch, -1)
        return self.out_activation(out)