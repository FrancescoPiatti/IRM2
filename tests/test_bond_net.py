"""
Tests for BondNet variants and the factory.
"""
import pytest
import torch

from src.configs import SimpleBondNetCfg, FiLMBondNetCfg
from src.models.bond_net import SimpleBondNet, FiLMBondNet, get_bond_net


def _make_inputs(batch=(7, 5), latent_dim=4, bond_feat_dim=3):
    z = torch.randn(*batch, latent_dim)
    bf = torch.randn(*batch, bond_feat_dim)
    return z, bf


def test_simple_bondnet_forward_shapes():
    cfg = SimpleBondNetCfg(latent_dim=4, bond_feat_dim=3,
                           latent_n_layers=1, latent_n_units=8,
                           bond_n_layers=1, bond_n_units=8,
                           fusion_n_layers=1, fusion_n_units=8)
    net = SimpleBondNet(cfg)
    z, bf = _make_inputs((7, 5), 4, 3)
    out = net(z, bf)
    assert out.shape == (7, 5)


def test_film_bondnet_forward_shapes():
    cfg = FiLMBondNetCfg(latent_dim=4, bond_feat_dim=3,
                         trunk_n_layers=1, trunk_n_units=8,
                         film_n_layers=1, film_n_units=8,
                         head_n_layers=1, head_n_units=8,
                         hidden_dim=8)
    net = FiLMBondNet(cfg)
    z, bf = _make_inputs((6,), 4, 3)
    out = net(z, bf)
    assert out.shape == (6,)


def test_get_bond_net_factory():
    s_cfg = SimpleBondNetCfg(latent_dim=4, bond_feat_dim=3,
                             latent_n_layers=1, latent_n_units=8,
                             bond_n_layers=1, bond_n_units=8,
                             fusion_n_layers=1, fusion_n_units=8)
    f_cfg = FiLMBondNetCfg(latent_dim=4, bond_feat_dim=3,
                           trunk_n_layers=1, trunk_n_units=8,
                           film_n_layers=1, film_n_units=8,
                           head_n_layers=1, head_n_units=8,
                           hidden_dim=8)
    assert isinstance(get_bond_net(s_cfg), SimpleBondNet)
    assert isinstance(get_bond_net(f_cfg), FiLMBondNet)


def test_get_bond_net_rejects_other_cfg():
    class Stub:
        pass
    with pytest.raises(TypeError):
        get_bond_net(Stub())


def test_bondnet_output_positive():
    cfg = SimpleBondNetCfg(latent_dim=4, bond_feat_dim=3,
                           latent_n_layers=1, latent_n_units=4,
                           bond_n_layers=1, bond_n_units=4,
                           fusion_n_layers=1, fusion_n_units=4,
                           output_positive=True)
    net = SimpleBondNet(cfg)
    z, bf = _make_inputs((10,), 4, 3)
    out = net(z, bf)
    assert (out > 0).all()


def test_bondnet_gradients():
    cfg = SimpleBondNetCfg(latent_dim=4, bond_feat_dim=3,
                           latent_n_layers=1, latent_n_units=4,
                           bond_n_layers=1, bond_n_units=4,
                           fusion_n_layers=1, fusion_n_units=4)
    net = SimpleBondNet(cfg)
    z, bf = _make_inputs((3,), 4, 3)
    out = net(z, bf).sum()
    out.backward()
    for p in net.parameters():
        assert p.grad is not None
