# src/configs/config_bondnet.py
from dataclasses import dataclass

from typing import Union
from typing import Optional

from torch.nn import Module
from torch.nn import Identity


@dataclass
class BaseBondNetCfg:  # noqa: D101 — see subclasses
    """
    Base config shared by all BondNet variants.

    Attributes
    ----------
    latent_dim : int
        Dimension of the latent state z_{T_f}.
    bond_feat_dim : int
        Dimension of the bond feature vector.
    activation : str | Module
        Hidden-layer activation.
    out_activation : str | Module
        Output activation of the final head.
    dropout : float | None
        Dropout probability used in MLP blocks.
    output_positive : bool
        If True, apply Softplus to the final output.
    """
    latent_dim: int
    bond_feat_dim: int

    activation: Union[str, Module] = "SiLU"
    out_activation: Union[str, Module] = Identity
    dropout: Optional[float] = 0.0
    output_positive: bool = False


@dataclass
class SimpleBondNetCfg(BaseBondNetCfg):
    """
    Config for the two-branch late-fusion BondNet.

    See :class:`BaseBondNetCfg` for the shared attributes
    (``latent_dim``, ``bond_feat_dim``, ``activation``, ``out_activation``,
    ``dropout``, ``output_positive``).

    Attributes
    ----------
    latent_n_layers : int
        Number of hidden layers in the latent branch.
    latent_n_units : int | tuple[int, ...]
        Hidden widths in the latent branch.
    bond_n_layers : int
        Number of hidden layers in the bond-feature branch.
    bond_n_units : int | tuple[int, ...]
        Hidden widths in the bond-feature branch.
    fusion_n_layers : int
        Number of hidden layers in the fusion head.
    fusion_n_units : int | tuple[int, ...]
        Hidden widths in the fusion head.
    """
    # Latent branch
    latent_n_layers: int = 2
    latent_n_units: int | tuple[int, ...] = 128

    # Bond feature branch
    bond_n_layers: int = 2
    bond_n_units: int | tuple[int, ...] = 64

    # Fusion head
    fusion_n_layers: int = 2
    fusion_n_units: int | tuple[int, ...] = 128


@dataclass
class FiLMBondNetCfg(BaseBondNetCfg):
    """
    Config for the FiLM/gated BondNet.

    See :class:`BaseBondNetCfg` for the shared attributes.

    Attributes
    ----------
    trunk_n_layers : int
        Number of hidden layers in the latent trunk.
    trunk_n_units : int | tuple[int, ...]
        Hidden widths in the latent trunk.
    film_n_layers : int
        Number of hidden layers in the FiLM network.
    film_n_units : int | tuple[int, ...]
        Hidden widths in the FiLM network.
    head_n_layers : int
        Number of hidden layers in the pricing head.
    head_n_units : int | tuple[int, ...]
        Hidden widths in the pricing head.
    hidden_dim : int
        Hidden size of the latent trunk representation to be modulated.
    """
    # Latent trunk
    trunk_n_layers: int = 2
    trunk_n_units: int | tuple[int, ...] = 128

    # FiLM network
    film_n_layers: int = 2
    film_n_units: int | tuple[int, ...] = 64

    # Pricing head
    head_n_layers: int = 2
    head_n_units: int | tuple[int, ...] = 128

    hidden_dim: int = 128


# Convenience union for type hints and isinstance() checks downstream.
BondNetCfg = BaseBondNetCfg
