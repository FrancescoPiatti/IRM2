# src/nn/gru.py
from torch import nn
from torch import Tensor

from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Sequential

from typing import Optional
from typing import Union
from typing import Tuple

from .activations import _get_activation


class GRU(nn.Module):
    """
    Stateful GRU wrapper with a linear readout.

    Mirrors the interface of :class:`RNN` and :class:`LSTM` in this package.

    Attributes
    ----------
    in_features : Optional[int]
        Input feature dimensionality. If None, inferred on the first forward.
    out_features : int
        Output dimensionality.
    n_layers : int
        Number of stacked GRU layers.
    n_units : int
        Hidden width per layer.
    bidirectional : bool
        If True, the underlying GRU is bidirectional.
    out_activation : nn.Module
        Activation applied after the readout layer.
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 32,
        n_layers: int = 2,
        n_units: int = 64,
        dropout: Optional[float] = None,
        out_activation: Union[str, nn.Module] = Identity,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.in_features    = in_features
        self.out_features   = out_features
        self.n_layers       = n_layers
        self.n_units        = n_units
        self.bidirectional  = bidirectional
        self._dropout       = float(dropout) if dropout is not None else 0.0
        self.out_activation = _get_activation(out_activation)

        self._gru     = None
        self.readout  = None

        if self.in_features is not None:
            self._build(self.in_features)

    def _build(self, input_size: int, device=None, dtype=None) -> None:
        """
        Instantiate the underlying nn.GRU and the readout. When called from
        forward, ``device``/``dtype`` are taken from the input tensor so the
        lazy submodules land where the user expects.
        """
        gru = nn.GRU(
            input_size   = input_size,
            hidden_size  = self.n_units,
            num_layers   = self.n_layers,
            dropout      = self._dropout,
            bidirectional= self.bidirectional,
            batch_first  = True,
        )
        rnn_output_dim = self.n_units * (2 if self.bidirectional else 1)
        readout = Sequential(
            Linear(rnn_output_dim, self.out_features),
            self.out_activation,
        )
        if device is not None or dtype is not None:
            gru = gru.to(device=device, dtype=dtype)
            readout = readout.to(device=device, dtype=dtype)
        self._gru = gru
        self.readout = readout

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        return_state: bool = False,
        return_sequence: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, T, D) or (T, D); 2D inputs are reshaped to (1, T, D).
        state : Optional[Tensor]
            Initial hidden state of shape (n_layers * num_directions, B, hidden_size).
        return_state : bool
            If True, also return the final hidden state.
        return_sequence : bool
            If True, return outputs at every timestep; otherwise return only the
            last step.

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            Network output (with optional final state).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self._gru is None:
            self._build(x.size(-1), device=x.device, dtype=x.dtype)

        output, new_state = self._gru(x, state)

        if return_sequence:
            out = self.readout(output)
        else:
            last = output[:, -1, :]
            out  = self.readout(last)

        if return_state:
            return out, new_state
        return out
