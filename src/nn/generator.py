# src/nn/generator.py
import torch
import torch.nn as nn

from .mlp import MLP
from .rnn import RNN
from .lstm import LSTM
from .gru import GRU
from .mamba import MambaEncoder
from .constant import ConstantNet
from .affine import AffineNet

from typing import Optional
from typing import Mapping
from typing import Any


from ..configs.config_nn import (
    DEFAULT_CONFIG_MLP,
    DEFAULT_CONFIG_RNN,
    DEFAULT_CONFIG_LSTM,
    DEFAULT_CONFIG_GRU,
    DEFAULT_CONFIG_MAMBA,
    DEFAULT_CONFIG_CONSTANT,
    DEFAULT_CONFIG_AFFINE,
)

SUPPORTED_NETWORK_TYPES = {
    "mlp": (MLP, DEFAULT_CONFIG_MLP),
    "rnn": (RNN, DEFAULT_CONFIG_RNN),
    "lstm": (LSTM, DEFAULT_CONFIG_LSTM),
    "gru": (GRU, DEFAULT_CONFIG_GRU),
    "mamba": (MambaEncoder, DEFAULT_CONFIG_MAMBA),
    "constant": (ConstantNet, DEFAULT_CONFIG_CONSTANT),
    "affine": (AffineNet, DEFAULT_CONFIG_AFFINE),
}

def create_network_from_config(
        config: Mapping[str, Any],
        input_dim: Optional[int],
        output_dim: int
    ) -> nn.Module:
    """
    Create a neural network from a config mapping.

    This function merges a default config with user overrides.

    Args:
        config: dict containing at least a 'type' key (one of 'mlp','rnn','lstm','gru', 'constant', 'affine').
                Other keys override defaults for that network.
        input_dim: Optional[int] specifying input feature dimension (D). 
                   If None, some networks (with lazy init) may infer it.
        output_dim: int specifying output feature dimension.

    Returns:
        Initialized nn.Module corresponding to the requested network.

    Raises:
        ValueError: if an unsupported network type is specified.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Config must be a Mapping[str, Any].")

    # Determine network type 
    net_type = config.get('type', None)
    if net_type is None:
        raise ValueError(
            f"Network config must contain key 'type'. Supported: {list(SUPPORTED_NETWORK_TYPES.keys())}"
        )
    
    net_type = str(net_type).lower()
    if net_type not in SUPPORTED_NETWORK_TYPES:
        raise ValueError(
            f"Unsupported network type: '{net_type}'. Supported: {list(SUPPORTED_NETWORK_TYPES.keys())}"
        )


    net_cls, default_cfg = SUPPORTED_NETWORK_TYPES[net_type]

    # Merge default config with user overrides
    _cfg = dict(default_cfg)
    _cfg.update(dict(config))

    # ------------------------------------------------------------
    # Constant network
    # ------------------------------------------------------------
    if net_type == "constant":
        return net_cls(
            out_features=_cfg.get("out_features", output_dim),
            out_activation=_cfg.get("out_activation"),
            init=_cfg.get("init"),
            init_std=_cfg.get("init_std"),
        )

    # ------------------------------------------------------------
    # Affine network
    # ------------------------------------------------------------
    if net_type == "affine":
        return net_cls(
            in_features=input_dim,
            out_features=_cfg.get("out_features", output_dim),
            bias=_cfg.get("bias"),
            out_activation=_cfg.get("out_activation"),
        )

    # ------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------
    if net_type == 'mlp':
        return net_cls(
            in_features    = input_dim,
            out_features   = output_dim,
            n_layers       = _cfg.get('n_layers'),
            n_units        = _cfg.get('n_units'),
            dropout        = _cfg.get('dropout'),
            activation     = _cfg.get('activation'),
            out_activation = _cfg.get('out_activation')
        )

    # ------------------------------------------------------------
    # Mamba
    # ------------------------------------------------------------
    if net_type == "mamba":
        _n_units = _cfg.get('n_units')
        if _n_units is not None and not isinstance(_n_units, int):
            raise TypeError(f"n_units must be an integer for network {net_type}.")

        return net_cls(
            in_features    = input_dim,
            out_features   = output_dim,
            n_layers       = _cfg.get('n_layers'),
            n_units        = _cfg.get('n_units'),
            d_state        = _cfg.get('d_state', 16),
            d_conv         = _cfg.get('d_conv', 4),
            expand         = _cfg.get('expand', 2),
            dropout        = _cfg.get('dropout'),
            out_activation = _cfg.get('out_activation'),
            norm_type      = _cfg.get('norm_type', 'layernorm'),
        )

    # ------------------------------------------------------------
    # Recurrent (rnn/lstm/gru)
    # ------------------------------------------------------------
    _n_units = _cfg.get('n_units')
    if _n_units is not None and not isinstance(_n_units, int):
        raise TypeError(f"n_units must be an integer for network {net_type}.")

    # rnn/lstm/gru share signature
    return net_cls(
        in_features     = input_dim,
        out_features    = output_dim,
        n_layers        = _cfg.get('n_layers'),
        n_units         = _cfg.get('n_units'),
        dropout         = _cfg.get('dropout'),
        out_activation  = _cfg.get('out_activation'),
        bidirectional   = _cfg.get('bidirectional')
    )
        