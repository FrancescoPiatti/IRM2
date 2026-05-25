"""
Tests for the top-level ShortRateModel composition (encoder + NSDE + decoder + bondnet).
"""
import pytest
import torch

from src.configs import EncoderCfg, NSDECfg, SimpleBondNetCfg
from src.models.short_rate_model import ShortRateModel
from src.models.nsde import Simple_NeuralSDE, OU_NeuralSDE
from src.models.encoders import Encoder
from src.models.bond_net import SimpleBondNet, FiLMBondNet
from src.types.data_types import EncoderInputs


def _make_model_simple(latent_dim=8, with_bondnet=False):
    bondnet = None
    if with_bondnet:
        bondnet = SimpleBondNetCfg(
            latent_dim=latent_dim, bond_feat_dim=4,
            latent_n_layers=1, latent_n_units=4,
            bond_n_layers=1, bond_n_units=4,
            fusion_n_layers=1, fusion_n_units=4,
        )
    return ShortRateModel(
        name="t",
        encoder=EncoderCfg(mode="simple"),
        nsde=NSDECfg(type="simple"),
        bondnet=bondnet,
        latent_dim=latent_dim,
    )


def test_model_builds_with_simple_encoder_nsde():
    m = _make_model_simple()
    assert isinstance(m.encoder, Encoder)
    assert isinstance(m.nsde, Simple_NeuralSDE)
    assert m.latent_dim == 8
    assert m.bondnet is None


def test_model_builds_with_ou_nsde():
    m = ShortRateModel(
        name="t",
        encoder=EncoderCfg(mode="simple"),
        nsde=NSDECfg(type="ou"),
        latent_dim=8,
    )
    assert isinstance(m.nsde, OU_NeuralSDE)


def test_model_builds_with_bondnet_from_cfg():
    m = _make_model_simple(with_bondnet=True)
    assert isinstance(m.bondnet, SimpleBondNet)


def test_model_rejects_invalid_bondnet_type():
    with pytest.raises(TypeError):
        ShortRateModel(
            name="t",
            encoder=EncoderCfg(mode="simple"),
            nsde=NSDECfg(type="simple"),
            bondnet="not a config",  # type: ignore[arg-type]
            latent_dim=8,
        )


def test_model_latent_dim_one_rejected():
    with pytest.raises(ValueError, match="latent_dim"):
        ShortRateModel(
            name="t",
            encoder=EncoderCfg(mode="simple"),
            nsde=NSDECfg(type="simple"),
            latent_dim=1,
        )


def test_model_encode_simple_shape():
    m = _make_model_simple(latent_dim=6)
    # Curve history: (T, M); short rate: (T, 1)
    T, M = 10, 5
    inputs = EncoderInputs(
        curve_history=torch.randn(T, M),
        short_rate=torch.randn(T, 1),
    )
    z = m.encode(inputs)
    # Encoder returns (B, latent_dim); single batch
    assert z.shape[-1] == 6


def test_model_simulate_smoke():
    torch.manual_seed(0)
    m = _make_model_simple(latent_dim=4)
    T, M = 5, 3
    inputs = EncoderInputs(
        curve_history=torch.randn(T, M),
        short_rate=torch.randn(T, 1),
    )
    z = m.encode(inputs)
    paths = m.simulate(z, n_paths=3, horizon=0.5, dt=1.0 / 32, decode=False)
    assert paths.dim() == 3
    assert paths.shape[0] == 3
    assert paths.shape[-1] == 4
