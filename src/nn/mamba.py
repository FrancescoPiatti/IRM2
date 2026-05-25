# src/nn/mamba.py
import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Sequential

from typing import Optional
from typing import Union
from typing import Tuple

from .activations import _get_activation


class MambaBlock(nn.Module):
    """
    Single Mamba block: selective state space model (S6) with gating.

    Implements the Mamba architecture (Gu & Dao, 2023) in pure PyTorch
    without requiring the mamba_ssm CUDA package.

    Parameters
    ----------
    d_model : int
        Model dimension (input/output width of this block).
    d_state : int
        SSM state expansion factor (N in the paper).
    d_conv : int
        Local convolution width.
    expand : int
        Expansion factor for the inner dimension.

    Attributes
    ----------
    d_inner : int
        Inner projection dimension (``d_model * expand``).
    in_proj : nn.Linear
        Projects input to gating and SSM branches.
    conv1d : nn.Conv1d
        Causal depthwise 1-D convolution.
    x_proj : nn.Linear
        Projects convolution output to SSM parameters (dt, B, C).
    dt_proj : nn.Linear
        Rank-1 projection for the discretisation step dt.
    A_log : nn.Parameter
        Log-space diagonal state matrix, shape ``(d_inner, d_state)``.
    D : nn.Parameter
        Skip-connection weights, shape ``(d_inner,)``.
    out_proj : nn.Linear
        Projects d_inner back to d_model.
    norm : nn.LayerNorm
        Pre-norm applied to the input.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projection: x -> (z, x_proj) where both are d_inner
        self.in_proj = Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D depthwise convolution (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters: input-dependent B, C, dt
        # x -> (dt, B, C) projections
        self.x_proj = Linear(self.d_inner, self.d_state + self.d_state + 1, bias=False)

        # dt projection (rank-1 + bias)
        self.dt_proj = Linear(1, self.d_inner, bias=True)

        # A parameter (diagonal, log-space for stability)
        # Initialized as -log(1, 2, ..., N) per channel
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)  # (d_inner, N)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        return_state: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of a single Mamba block.

        Args:
            x: (B, T, d_model)
            state: Optional SSM hidden state of shape (B, d_inner, d_state).
            return_state: If True, also return final SSM state.

        Returns:
            Output of shape (B, T, d_model), and optionally the final state.
        """
        residual = x
        x = self.norm(x)

        B, T, D = x.shape

        # Input projection -> z (gate) and x_proj
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal 1D convolution
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # causal: trim to T
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = torch.nn.functional.silu(x_conv)

        # Compute input-dependent SSM parameters
        ssm_params = self.x_proj(x_conv)  # (B, T, d_state + d_state + 1)
        dt_raw = ssm_params[..., :1]  # (B, T, 1)
        B_input = ssm_params[..., 1:1 + self.d_state]  # (B, T, N)
        C_input = ssm_params[..., 1 + self.d_state:]  # (B, T, N)

        # dt: softplus(dt_proj(dt_raw))
        dt = torch.nn.functional.softplus(self.dt_proj(dt_raw))  # (B, T, d_inner)

        # A from log-space
        A = -torch.exp(self.A_log)  # (d_inner, N)

        # Selective scan (sequential, pure PyTorch)
        y, final_state = self._selective_scan(
            x_conv, dt, A, B_input, C_input, state
        )

        # Apply D skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv  # (B, T, d_inner)

        # Gate with z
        y = y * torch.nn.functional.silu(z)

        # Output projection + residual
        out = self.out_proj(y) + residual  # (B, T, d_model)

        if return_state:
            return out, final_state
        return out

    def _selective_scan(
        self,
        x: Tensor,       # (B, T, d_inner)
        dt: Tensor,       # (B, T, d_inner)
        A: Tensor,        # (d_inner, N)
        B: Tensor,        # (B, T, N)
        C: Tensor,        # (B, T, N)
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Selective scan (S6) — sequential implementation.

        Returns:
            y: (B, T, d_inner)
            final_state: (B, d_inner, N)
        """
        batch, T, d_inner = x.shape
        N = A.shape[1]

        # Initialize state
        if state is None:
            h = torch.zeros(batch, d_inner, N, device=x.device, dtype=x.dtype)
        else:
            h = state

        outputs = []
        for t in range(T):
            # Discretize: A_bar = exp(dt * A), B_bar = dt * B
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, N)
            B_bar = dt_t * B[:, t, :].unsqueeze(1)  # (B, d_inner, N) via broadcast (B,1,N) * (B,d_inner,1)
            # Corrected: B[:, t, :] is (B, N), need (B, 1, N) to broadcast with dt_t (B, d_inner, 1)
            # dt_t * B gives (B, d_inner, N) which is what we want

            # State update: h = A_bar * h + B_bar * x_t
            x_t = x[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            h = A_bar * h + B_bar * x_t  # (B, d_inner, N)

            # Output: y_t = C_t @ h
            C_t = C[:, t, :].unsqueeze(1)  # (B, 1, N)
            y_t = (h * C_t).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        return y, h


class MambaEncoder(nn.Module):
    """
    Multi-block Mamba encoder following the same interface as RNN/LSTM/GRU.

    Parameters
    ----------
    in_features : int, optional
        Input dimensionality (D). If None, inferred on first forward.
    out_features : int
        Output dimensionality.
    n_layers : int
        Number of stacked Mamba blocks.
    n_units : int
        Model dimension (d_model) for internal Mamba blocks.
    d_state : int
        SSM state expansion factor.
    d_conv : int
        Convolution kernel size.
    expand : int
        Inner dimension expansion factor.
    dropout : float, optional
        Dropout probability between blocks.
    out_activation : str or nn.Module
        Activation after the readout layer.

    Attributes
    ----------
    input_proj : nn.Linear or None
        Lazy-built projection from in_features to n_units.
    blocks : nn.ModuleList
        Stacked MambaBlock layers.
    drop : nn.Dropout or None
        Dropout layer between blocks (None if dropout is 0).
    readout : nn.Sequential
        Linear + activation mapping n_units to out_features.
    _built : bool
        Whether the lazy input projection has been constructed.
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 32,
        n_layers: int = 2,
        n_units: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: Optional[float] = None,
        out_activation: Union[str, nn.Module] = Identity,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.n_units = n_units
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self._dropout = float(dropout) if dropout is not None else 0.0
        self.out_activation = _get_activation(out_activation)

        # Placeholders — will be set in _build
        self.input_proj = None
        self.blocks = None
        self.readout = None

        # Eager build if in_features known
        if self.in_features is not None:
            self._build(self.in_features)

    def _build(self, input_size: int, device=None, dtype=None):
        """
        Build the input projection, Mamba blocks, and readout layer. The
        optional ``device`` / ``dtype`` keep newly built parameters consistent
        with the input tensor when build happens lazily.
        """
        input_proj = Linear(input_size, self.n_units)

        blocks_list = []
        for _ in range(self.n_layers):
            blocks_list.append(
                MambaBlock(
                    d_model=self.n_units,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                )
            )
            if self._dropout > 0:
                blocks_list.append(nn.Dropout(self._dropout))
        blocks = nn.ModuleList(blocks_list)

        readout = Sequential(
            Linear(self.n_units, self.out_features),
            self.out_activation,
        )

        if device is not None or dtype is not None:
            input_proj = input_proj.to(device=device, dtype=dtype)
            blocks = blocks.to(device=device, dtype=dtype)
            readout = readout.to(device=device, dtype=dtype)

        self.input_proj = input_proj
        self.blocks = blocks
        self.readout = readout

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
        return_state: bool = False,
        return_sequence: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the Mamba encoder.

        Args:
            x: (B, T, D) or (T, D); 2D input is reshaped to (1, T, D).
            state: Optional hidden state. Shape: (n_layers, B, d_inner, d_state)
                   where d_inner = n_units * expand.
            return_state: If True, returns (out, final_state).
            return_sequence: If True, returns all timesteps (B, T, out_features).
                             Else, returns only last timestep (B, out_features).

        Returns:
            out or (out, final_state).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Lazy build (preserve device/dtype of the input tensor)
        if self.input_proj is None:
            self._build(x.size(-1), device=x.device, dtype=x.dtype)

        # Project input to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Track per-block states
        block_idx = 0
        new_states = []
        for module in self.blocks:
            if isinstance(module, MambaBlock):
                s = state[block_idx] if state is not None else None
                if return_state:
                    x, new_s = module(x, state=s, return_state=True)
                    new_states.append(new_s)
                else:
                    x = module(x, state=s, return_state=False)
                block_idx += 1
            else:
                # Dropout or other non-MambaBlock modules
                x = module(x)

        # x: (B, T, d_model)
        if return_sequence:
            out = self.readout(x)  # (B, T, out_features)
        else:
            last = x[:, -1, :]  # (B, d_model)
            out = self.readout(last)  # (B, out_features)

        if return_state:
            final_state = torch.stack(new_states, dim=0)  # (n_layers, B, d_inner, d_state)
            return out, final_state
        return out
