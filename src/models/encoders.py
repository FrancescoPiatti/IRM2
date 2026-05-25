# src/models/encoder.py
import torch
from torch import Tensor, nn

from typing import Optional
from typing import Union
from typing import Tuple
from typing import Any
from typing import Mapping

from ..nn.generator import create_network_from_config

from ..configs.config_encoder import EncoderCfg

from ..utils.checks import _check_positive_integer_value


class Encoder(nn.Module):
    """
    Encoder Module supporting 'simple' and 'hierarchical' modes.

    Modes
    -----
    - simple:
        z = net(x)  -> last dim is `output_dim`

    - hierarchical:
        fast_out = fast_net(fast_x)
        slow_out = slow_net(slow_x)

        combine options:
        - "project": concat then linear projection back to `output_dim`
            fast_out[..., D] , slow_out[..., D] -> cat[..., 2D] -> Linear(2D->D)

        - "concat": split dims across branches so concat is already `output_dim`
            d_fast = D//2, d_slow = D-d_fast
            fast_out[..., d_fast], slow_out[..., d_slow] -> cat[..., D]

        - "add": elementwise sum
            fast_out[..., D] + slow_out[..., D] -> [..., D]

    Output dimension policy
    -----------------------
    The encoder output last dimension is ALWAYS `output_dim` in both modes.

    Attributes
    ----------
    input_dim : int or None
        Input feature dimension.
    output_dim : int
        Output (latent) dimension.
    cfg : EncoderCfg
        Validated encoder configuration.
    network : nn.Module or None
        Backbone network (simple mode only).
    fast_encoder : nn.Module or None
        Fast-stream network (hierarchical mode only).
    slow_encoder : nn.Module or None
        Slow-stream network (hierarchical mode only).
    combine_proj : nn.Linear or None
        Projection layer for hierarchical "project" combine mode.
    out_norm : nn.Module or None
        Output normalisation layer (LayerNorm, RMSNorm, or None).
    """
    def __init__(
        self,
        output_dim: int,
        input_dim: Optional[int] = None,
        config: Optional[EncoderCfg] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cfg = config or EncoderCfg()

        # Validate config
        self.cfg.validate()
        self._validate_cfg()

        # -------------------------
        # Build networks
        # -------------------------
        self.network = None
        self.fast_encoder = None
        self.slow_encoder = None
        self.combine_proj = None

        if self.cfg.mode == "simple":
            self.network = create_network_from_config(
                config=dict(self.cfg.net),
                input_dim=self.input_dim,
                output_dim=self.output_dim
            )

        else:  # hierarchical
            if self.cfg.combine == "concat":
                self.fast_dim = self.output_dim // 2
                self.slow_dim = self.output_dim - self.fast_dim
            else:
                # project or add: both branches output full D
                self.fast_dim = self.output_dim
                self.slow_dim = self.output_dim

            self.fast_encoder = create_network_from_config(
                config=dict(self.cfg.fast_net),
                input_dim=self.input_dim,
                output_dim=self.fast_dim
            )
            self.slow_encoder = create_network_from_config(
                config=dict(self.cfg.slow_net),
                input_dim=self.input_dim,
                output_dim=self.slow_dim
            )

            if self.cfg.combine == "project":
                # concat dims = 2D -> project back to D
                self.combine_proj = nn.Linear(2 * self.output_dim, self.output_dim)

        # -------------------------
        # Output normalization (optional)
        # -------------------------

        if self.cfg.out_norm in (None, "none", "None"):
            self.out_norm = None
        elif self.cfg.out_norm == "layernorm":
            self.out_norm = nn.LayerNorm(self.output_dim)
        elif self.cfg.out_norm == "rmsnorm":
            self.out_norm = nn.RMSNorm(self.output_dim)


    # -------------------------
    # Validation
    # -------------------------
    def _validate_cfg(self) -> None:
        """
        Validate encoder-specific constraints that depend on __init__ args.
        Config-level validation is handled by cfg.validate().
        """
        _check_positive_integer_value(self.output_dim, "Encoder output_dim")

        if self.cfg.mode == 'hierarchical':
            if str(self.cfg.combine).lower() == "concat" and self.output_dim < 2:
                raise ValueError("Hierarchical concat requires output_dim >= 2.")


    def _validate_input(self, x: Any) -> None:
        """
        Validate input `x` based on the encoder mode.
        """
        if self.cfg.mode == "simple":
            assert isinstance(x, Tensor), "Simple mode expects x to be a Tensor."
        else:  # hierarchical
            assert isinstance(x, tuple) and len(x) == 2, \
                "Hierarchical mode expects x=(fast_x, slow_x)."
            fast_x, slow_x = x
            assert isinstance(fast_x, Tensor) and isinstance(slow_x, Tensor), \
                "Both elements of input x must be Tensors in 'hierarchical' mode."


    def _validate_state(self, state: Any) -> None:
        if self.cfg.mode == "simple":
            assert state is None or isinstance(state, Tensor), \
                "Simple mode state must be a Tensor or None."
            return

        # hierarchical
        if state is not None:
            assert isinstance(state, tuple) and len(state) == 2, \
                "Hierarchical mode expects state=(fast_state, slow_state)."
            fast_state, slow_state = state
            assert fast_state is None or isinstance(fast_state, Tensor), \
                "fast_state must be a Tensor or None."
            assert slow_state is None or isinstance(slow_state, Tensor), \
                "slow_state must be a Tensor or None."


    # -------------------------
    # Forward
    # -------------------------

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        return_state: bool = False,
        return_sequence: bool = False,
    ):
        """
        Forward pass through the encoder.

        Args:
            x : Union[Tuple, Tensor]
                Input data. Can be a Tensor for simple mode or a tuple of Tensors for hierarchical mode.
            state : Optional[Tensor]
                Initial state for the encoder. If hierarchical, should be a tuple of states for fast and slow encoders.
            return_state : bool
                If True, returns the final state of the encoder.
            return_sequence : bool
                If True, returns the full sequence of outputs instead of just the last output.

        Returns:
            torch.Tensor
                Encoded representation. If return_state is True, returns also a tuple of (encoded_representation, final_state).
        """

        self._validate_input(x)
        self._validate_state(state)

        # Simple encoder
        if self.cfg.mode == "simple":

            out = self.network(x, state=state, return_state=return_state, return_sequence=return_sequence)
            return self._apply_norm_out(out, return_state)

        # Hierarchical encoder
        fast_x , slow_x = x

        if state is None:
            fast_state, slow_state = None, None
        else:
            fast_state, slow_state = state

        if return_state:
            fast_out, fast_state = self.fast_encoder(
                fast_x, state=fast_state, return_state=True, return_sequence=return_sequence
            )
            slow_out, slow_state = self.slow_encoder(
                slow_x, state=slow_state, return_state=True, return_sequence=return_sequence
            )
            z = self._combine(fast_out, slow_out)
            z = self._apply_norm_tensor(z)
            return z, (fast_state, slow_state)

        else:
            fast_out = self.fast_encoder(fast_x, state=fast_state, return_state=False, return_sequence=return_sequence)
            slow_out = self.slow_encoder(slow_x, state=slow_state, return_state=False, return_sequence=return_sequence)
            z = self._combine(fast_out, slow_out)
            z = self._apply_norm_tensor(z)
            return z


    # -------------------------
    # Combine + Norm helpers
    # ------------------------

    def _combine(self, fast_out: Tensor, slow_out: Tensor) -> Tensor:
        """
        Merge fast and slow branch outputs according to ``cfg.combine``.
        """

        if self.cfg.combine == "add":
            return fast_out + slow_out
        out = torch.cat([fast_out, slow_out], dim=-1)
        if self.cfg.combine == "project":
            return self.combine_proj(out)
        else:  # concat
            return out


    def _apply_norm_out(self, out, return_state: bool):
        """
        Apply output normalisation, handling the (tensor, state) tuple case.
        """
        if not return_state:
            return self._apply_norm_tensor(out)
        z, st = out
        z = self._apply_norm_tensor(z)
        return z, st


    def _apply_norm_tensor(self, z: Tensor) -> Tensor:
        """
        Apply output normalisation to a single tensor.
        """
        if self.out_norm is None:
            return z
        return self.out_norm(z)

    # -------------------------
    # Properties
    # -------------------------

    @property
    def latent_dim(self) -> int:
        """
        Returns the output dimension of the encoder.
        """
        return self.output_dim
