# src/nn/mlp.py
import torch
from torch import nn
from torch.nn import Identity
from torch.nn import Module
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import ReLU

from typing import List
from typing import Optional
from typing import Union
from typing import Sequence

from .activations import _get_activation

class MLP(Sequential):
    """
    Configurable multi-layer perceptron (MLP).

    Parameters
    ----------
    in_features : int, optional
        Input dimensionality. If ``None`` a :class:`torch.nn.LazyLinear` layer is used to infer
        the size on the first forward pass.
    out_features : int, default 1
    n_layers : int, default 3
        Number of hidden layers (excluding the final output layer).
    n_units : int or Sequence[int], default 64
        Hidden layer widths. If a single int is provided, it is repeated ``n_layers`` times.
    dropout : float, optional
        Dropout probability applied after each hidden activation (if provided).
    activation : str or nn.Module, default ``ReLU``
        Activation applied after each hidden linear layer.
    out_activation : str or nn.Module, default ``Identity``
        Activation applied to the output layer.
    """

    def __init__(self, 
                 in_features : Optional[int] = None,
                 out_features : int = 1,
                 n_layers : int = 3,
                 n_units : Union[int, Sequence[int]] = 64,
                 dropout : Optional[float] = None,
                 activation : Union[str, Module] = ReLU, 
                 out_activation : Union[str, Module] = Identity):

        # Normalize n_units: allow a single int or a sequence with one width per layer
        n_units = (n_units,) * n_layers if isinstance(n_units, int) else n_units
        assert len(n_units) == n_layers, "n_units must match n_layers"
        
        activation = _get_activation(activation)
        out_activation = _get_activation(out_activation)
        dropout = float(dropout) if dropout is not None else 0.0

        layers : List[Module] = []

        # Build hidden layers
        for i in range(n_layers):

            # First hidden layer: from input dimension to first hidden width
            if i == 0:
                if in_features is None:
                    layers.append(LazyLinear(n_units[0]))
                else:
                    layers.append(Linear(in_features, n_units[0]))
            else:
                # Subsequent hidden layer
                layers.append(Linear(n_units[i-1], n_units[i]))
            
            # Hidden activation
            layers.append(activation)

            # Dropout if specified
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(Linear(n_units[-1], out_features))
        
        # Output activation
        layers.append(out_activation)

        # Initialize the parent Sequential with the collected layers
        super().__init__(*layers)


