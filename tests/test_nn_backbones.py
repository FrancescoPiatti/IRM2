"""
Tests for the building-block neural networks under src/nn/.
"""
import pytest
import torch

from src.nn import MLP, RNN, LSTM, GRU
from src.nn.affine import AffineNet
from src.nn.constant import ConstantNet
from src.nn.activations import _get_activation
from src.nn.generator import create_network_from_config


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def test_activation_str_lookup():
    assert isinstance(_get_activation("relu"), torch.nn.ReLU)
    assert isinstance(_get_activation("SILU"), torch.nn.SiLU)
    assert isinstance(_get_activation("identity"), torch.nn.Identity)


def test_activation_module_passthrough():
    mod = torch.nn.Tanh()
    assert _get_activation(mod) is mod


def test_activation_class_invocation():
    out = _get_activation(torch.nn.ELU)
    assert isinstance(out, torch.nn.ELU)


def test_activation_rejects_unknown():
    with pytest.raises(ValueError):
        _get_activation("not_a_real_activation")


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def test_mlp_shapes_eager():
    net = MLP(in_features=4, out_features=2, n_layers=2, n_units=8)
    out = net(torch.randn(3, 4))
    assert out.shape == (3, 2)


def test_mlp_shapes_lazy():
    net = MLP(in_features=None, out_features=3, n_layers=2, n_units=8)
    out = net(torch.randn(5, 7))
    assert out.shape == (5, 3)


def test_mlp_n_units_sequence():
    net = MLP(in_features=4, out_features=2, n_layers=3, n_units=(8, 4, 4))
    out = net(torch.randn(2, 4))
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Recurrent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Cls", [RNN, LSTM, GRU])
def test_recurrent_eager(Cls):
    net = Cls(in_features=4, out_features=3, n_layers=2, n_units=8)
    out = net(torch.randn(2, 5, 4))  # (B, T, D)
    assert out.shape == (2, 3)


@pytest.mark.parametrize("Cls", [RNN, LSTM, GRU])
def test_recurrent_lazy(Cls):
    net = Cls(in_features=None, out_features=3, n_layers=2, n_units=8)
    out = net(torch.randn(2, 5, 4))
    assert out.shape == (2, 3)


@pytest.mark.parametrize("Cls", [RNN, LSTM, GRU])
def test_recurrent_return_sequence(Cls):
    net = Cls(in_features=4, out_features=3, n_layers=1, n_units=8)
    out = net(torch.randn(2, 5, 4), return_sequence=True)
    assert out.shape == (2, 5, 3)


def test_rnn_unsqueeze_2d_input():
    net = RNN(in_features=4, out_features=3, n_layers=1, n_units=8)
    out = net(torch.randn(5, 4))  # (T, D)
    assert out.shape == (1, 3)


# ---------------------------------------------------------------------------
# AffineNet / ConstantNet
# ---------------------------------------------------------------------------


def test_affine_shape():
    net = AffineNet(in_features=4, out_features=3)
    out = net(torch.randn(2, 4))
    assert out.shape == (2, 3)


def test_affine_lazy():
    net = AffineNet(in_features=None, out_features=3)
    out = net(torch.randn(2, 7))
    assert out.shape == (2, 3)


def test_constant_returns_same_per_row():
    net = ConstantNet(out_features=4, init="normal", init_std=0.5)
    out = net(torch.randn(3, 9))
    # Every row equal
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[0], out[2])


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def test_generator_builds_mlp():
    net = create_network_from_config({"type": "mlp", "n_layers": 2, "n_units": 8},
                                     input_dim=4, output_dim=2)
    out = net(torch.randn(3, 4))
    assert out.shape == (3, 2)


def test_generator_rejects_unknown_type():
    with pytest.raises(ValueError):
        create_network_from_config({"type": "transformer"}, input_dim=4, output_dim=2)


def test_generator_rejects_missing_type():
    with pytest.raises(ValueError):
        create_network_from_config({}, input_dim=4, output_dim=2)
