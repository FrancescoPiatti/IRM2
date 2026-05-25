# src/models/bond_net.py
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Identity
from torch.nn import Softplus

from ..nn.mlp import MLP

from ..configs.config_bondnet import BaseBondNetCfg
from ..configs.config_bondnet import SimpleBondNetCfg
from ..configs.config_bondnet import FiLMBondNetCfg


class SimpleBondNet(Module):
    """
    Two-branch MLP BondNet with late fusion.

    The latent state and bond features are processed by separate branches and
    then concatenated for a fusion head producing the dirty bond price.

    Attributes
    ----------
    cfg : SimpleBondNetCfg
        Configuration used to build the network.
    latent_branch : MLP
        MLP mapping the latent state to a hidden vector.
    bond_branch : MLP
        MLP mapping bond features to a hidden vector.
    fusion_head : MLP
        MLP mapping the concatenated hidden vectors to a scalar.
    positive_head : nn.Module
        Optional Softplus applied to the output (if `cfg.output_positive`).
    """

    def __init__(self, cfg: SimpleBondNetCfg):
        super().__init__()
        self.cfg = cfg

        latent_out_dim = cfg.latent_n_units if isinstance(cfg.latent_n_units, int) else cfg.latent_n_units[-1]
        bond_out_dim = cfg.bond_n_units if isinstance(cfg.bond_n_units, int) else cfg.bond_n_units[-1]

        self.latent_branch = MLP(
            in_features=cfg.latent_dim,
            out_features=latent_out_dim,
            n_layers=cfg.latent_n_layers,
            n_units=cfg.latent_n_units,
            dropout=cfg.dropout,
            activation=cfg.activation,
            out_activation=Identity,
        )

        self.bond_branch = MLP(
            in_features=cfg.bond_feat_dim,
            out_features=bond_out_dim,
            n_layers=cfg.bond_n_layers,
            n_units=cfg.bond_n_units,
            dropout=cfg.dropout,
            activation=cfg.activation,
            out_activation=Identity,
        )

        self.fusion_head = MLP(
            in_features=latent_out_dim + bond_out_dim,
            out_features=1,
            n_layers=cfg.fusion_n_layers,
            n_units=cfg.fusion_n_units,
            dropout=cfg.dropout,
            activation=cfg.activation,
            out_activation=cfg.out_activation,
        )

        self.positive_head = Softplus() if cfg.output_positive else Identity()

    def forward(self, z: Tensor, bond_features: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor
            Latent state, shape (..., latent_dim).
        bond_features : Tensor
            Bond feature vector, shape (..., bond_feat_dim). Leading dims must
            match `z`.

        Returns
        -------
        Tensor
            Dirty bond price, shape (...,).
        """
        assert z.shape[:-1] == bond_features.shape[:-1]
        assert z.shape[-1] == self.cfg.latent_dim
        assert bond_features.shape[-1] == self.cfg.bond_feat_dim

        hz = self.latent_branch(z)
        hb = self.bond_branch(bond_features)
        out = self.fusion_head(torch.cat([hz, hb], dim=-1)).squeeze(-1)
        return self.positive_head(out)


class FiLMBondNet(Module):
    """
    BondNet with FiLM modulation.

    The latent state is processed through a trunk; bond features generate
    `(gamma, beta)` parameters that modulate the trunk hidden state as
    `h = (1 + gamma) * h + beta`, before a head MLP produces the bond price.

    Attributes
    ----------
    cfg : FiLMBondNetCfg
        Configuration used to build the network.
    latent_trunk : MLP
        Trunk applied to the latent state.
    film_net : MLP
        FiLM parameter generator producing (gamma, beta).
    head : MLP
        Pricing head producing the scalar bond value.
    positive_head : nn.Module
        Optional Softplus applied to the output.
    """

    def __init__(self, cfg: FiLMBondNetCfg):
        super().__init__()
        self.cfg = cfg

        self.latent_trunk = MLP(
            in_features=cfg.latent_dim,
            out_features=cfg.hidden_dim,
            n_layers=cfg.trunk_n_layers,
            n_units=cfg.trunk_n_units,
            dropout=cfg.dropout,
            activation=cfg.activation,
            out_activation=Identity,
        )

        self.film_net = MLP(
            in_features=cfg.bond_feat_dim,
            out_features=2 * cfg.hidden_dim,
            n_layers=cfg.film_n_layers,
            n_units=cfg.film_n_units,
            dropout=cfg.dropout,
            activation=cfg.activation,
            out_activation=Identity,
        )

        self.head = MLP(
            in_features=cfg.hidden_dim,
            out_features=1,
            n_layers=cfg.head_n_layers,
            n_units=cfg.head_n_units,
            dropout=cfg.dropout,
            activation=cfg.activation,
            out_activation=cfg.out_activation,
        )

        self.positive_head = Softplus() if cfg.output_positive else Identity()

    def forward(self, z: Tensor, bond_features: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor
            Latent state, shape (..., latent_dim).
        bond_features : Tensor
            Bond feature vector, shape (..., bond_feat_dim).

        Returns
        -------
        Tensor
            Dirty bond price, shape (...,).
        """
        assert z.shape[:-1] == bond_features.shape[:-1]
        assert z.shape[-1] == self.cfg.latent_dim
        assert bond_features.shape[-1] == self.cfg.bond_feat_dim

        h = self.latent_trunk(z)
        gamma_beta = self.film_net(bond_features)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)

        h = (1.0 + gamma) * h + beta
        out = self.head(h).squeeze(-1)
        return self.positive_head(out)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def get_bond_net(cfg: BaseBondNetCfg) -> Module:
    """
    Build the BondNet module matching the config variant.

    Parameters
    ----------
    cfg : BaseBondNetCfg
        Either a `SimpleBondNetCfg` or a `FiLMBondNetCfg` instance.

    Returns
    -------
    Module
        Initialised BondNet module.

    Raises
    ------
    TypeError
        If `cfg` is not a supported BondNet config subclass.
    """
    if isinstance(cfg, SimpleBondNetCfg):
        return SimpleBondNet(cfg)
    if isinstance(cfg, FiLMBondNetCfg):
        return FiLMBondNet(cfg)
    raise TypeError(f"Unsupported BondNet config type: {type(cfg).__name__}")


# Backwards-compatible alias used by ShortRateModel: a `BondNet` "type" that
# really resolves to the right subclass at construction time.
BondNet = SimpleBondNet
