# src/nn/affine.py
from torch import nn
from torch import Tensor
from torch.nn import Identity
from torch.nn import Module

from typing import Optional
from typing import Union

from .activations import _get_activation


class AffineNet(Module):
    """
    Affine network: a single linear layer (optionally lazy) + output activation.

    Parameters
    ----------
    in_features : int, optional
        Input dimensionality. If None, uses LazyLinear to infer on first forward.
    out_features : int
        Output dimensionality.
    bias : bool, default True
        Whether to include an additive bias term.
    out_activation : str or nn.Module, default Identity
        Activation applied to the affine output.
    """

    def __init__(
        self,
        in_features: Optional[int],
        out_features: int,
        bias: bool = True,
        out_activation: Union[str, Module] = Identity,
    ):
        super().__init__()

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError("out_features must be a positive integer.")

        self.out_features = int(out_features)
        self.out_activation = _get_activation(out_activation)

        if in_features is None:
            # LazyLinear supports bias arg in recent PyTorch
            self.linear = nn.LazyLinear(self.out_features, bias=bool(bias))
        else:
            if not isinstance(in_features, int) or in_features <= 0:
                raise ValueError("in_features must be a positive integer or None.")
            self.linear = nn.Linear(int(in_features), self.out_features, bias=bool(bias))


    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, in_features) (or compatible).

        Returns
        -------
        Tensor
            Tensor of shape (B, out_features).
        """
        out = self.linear(x)
        return self.out_activation(out)