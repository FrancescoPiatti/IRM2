# src/nn/lstm.py
from torch import nn
from torch import Tensor
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Sequential

from typing import Optional
from typing import Union
from typing import Tuple

from .activations import _get_activation


class LSTM(nn.Module):
    """
    Stateful LSTM wrapper with a linear readout.

    The class mirrors the interface of :class:`RNN` and :class:`GRU` in this
    package: input is `(B, T, D)` (or `(T, D)`), the underlying `nn.LSTM` is
    lazily built when the input feature dimension is None, and the readout is
    a single linear layer followed by an output activation.

    Attributes
    ----------
    in_features : Optional[int]
        Input feature dimensionality. If None, inferred on the first forward.
    out_features : int
        Output dimensionality (number of targets).
    n_layers : int
        Number of stacked LSTM layers.
    n_units : int
        Hidden width (same for every layer).
    bidirectional : bool
        If True, the underlying LSTM is bidirectional.
    out_activation : nn.Module
        Activation applied after the linear readout.
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

        # Placeholders; built on first forward when in_features is None.
        self._lstm   = None
        self.readout = None

        if self.in_features is not None:
            self._build(self.in_features)

    def _build(self, input_size: int, device=None, dtype=None) -> None:
        """
        Instantiate the underlying nn.LSTM and the linear readout.

        Parameters
        ----------
        input_size : int
            Input feature dimensionality observed on the first forward.
        device, dtype : optional
            If provided, the newly built submodules are immediately moved to
            the given device/dtype — this avoids a CPU/GPU mismatch when the
            outer module is `.to(device)`-ed before the first forward call.
        """
        lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = self.n_units,
            num_layers   = self.n_layers,
            dropout      = self._dropout,
            bidirectional= self.bidirectional,
            batch_first  = True,
        )
        rnn_out_dim = self.n_units * (2 if self.bidirectional else 1)
        readout = Sequential(
            Linear(rnn_out_dim, self.out_features),
            self.out_activation,
        )
        if device is not None or dtype is not None:
            lstm = lstm.to(device=device, dtype=dtype)
            readout = readout.to(device=device, dtype=dtype)
        self._lstm = lstm
        self.readout = readout

    def forward(
        self,
        x: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
        return_state: bool = False,
        return_sequence: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, T, D) or (T, D). 2D input is reshaped to (1, T, D).
        state : Optional[Tuple[Tensor, Tensor]]
            Optional (h_0, c_0) initial states.
        return_state : bool
            If True, also return the final (h_n, c_n).
        return_sequence : bool
            If True, return outputs at every timestep (B, T, out_features);
            otherwise return only the last step (B, out_features).

        Returns
        -------
        Tensor or Tuple[Tensor, Tuple[Tensor, Tensor]]
            Network output, optionally accompanied by the final hidden state.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self._lstm is None:
            self._build(x.size(-1), device=x.device, dtype=x.dtype)

        output, (h_n, c_n) = self._lstm(x, state)

        if return_sequence:
            out = self.readout(output)
        else:
            last = output[:, -1, :]
            out  = self.readout(last)

        if return_state:
            return out, (h_n, c_n)
        return out
