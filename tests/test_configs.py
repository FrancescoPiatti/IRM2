"""
Tests for the dataclass configs.

Covers validate() behaviour, default propagation, and the type-driven
nullification rules in EncoderCfg / NSDECfg.
"""
import pytest
import torch

from src.configs import DataLoaderCfg, EncoderCfg, NSDECfg, TrainerCfg
from src.configs import SimpleBondNetCfg, FiLMBondNetCfg, BondNetCfg, BaseBondNetCfg


# ---------------------------------------------------------------------------
# DataLoaderCfg
# ---------------------------------------------------------------------------


def test_data_loader_cfg_defaults_validate(tmp_path):
    cfg = DataLoaderCfg(data_path=str(tmp_path))
    cfg.validate()
    assert cfg.business_days_per_year == 252.0
    assert isinstance(cfg.dtype, torch.dtype)
    assert cfg.max_maturity > 0


def test_data_loader_cfg_rejects_no_targets(tmp_path):
    cfg = DataLoaderCfg(
        data_path=str(tmp_path),
        enable_yield=False,
        enable_short_rate=False,
        enable_bonds=False,
        enable_futures=False,
        enable_options=False,
    )
    with pytest.raises(ValueError, match="no targets enabled"):
        cfg.validate()


def test_data_loader_cfg_rejects_non_positive_business_days(tmp_path):
    cfg = DataLoaderCfg(data_path=str(tmp_path), business_days_per_year=0.0)
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------
# EncoderCfg
# ---------------------------------------------------------------------------


def test_encoder_cfg_simple_default_validates():
    cfg = EncoderCfg(mode="simple")
    cfg.validate()
    assert cfg.net is not None
    assert cfg.net["type"] in ("rnn", "gru", "lstm", "mamba")
    assert cfg.fast_net is None and cfg.slow_net is None


def test_encoder_cfg_hierarchical_default_validates():
    cfg = EncoderCfg(mode="hierarchical")
    cfg.validate()
    assert cfg.fast_net is not None
    assert cfg.slow_net is not None
    assert cfg.net is None


def test_encoder_cfg_rejects_unknown_mode():
    cfg = EncoderCfg(mode="quantum")
    with pytest.raises(ValueError):
        cfg.validate()


def test_encoder_cfg_rejects_mlp_backbone():
    cfg = EncoderCfg(mode="simple", net={"type": "mlp"})
    with pytest.raises(ValueError, match="Unsupported network type"):
        cfg.validate()


# ---------------------------------------------------------------------------
# NSDECfg
# ---------------------------------------------------------------------------


def test_nsde_cfg_simple_defaults():
    cfg = NSDECfg(type="simple")
    cfg.validate()
    assert cfg.drift is not None
    assert cfg.diffusion is not None
    assert cfg.long_term_mean is None
    assert cfg.mean_reversion is None


def test_nsde_cfg_ou_defaults():
    cfg = NSDECfg(type="ou")
    cfg.validate()
    assert cfg.drift is None
    assert cfg.long_term_mean is not None
    assert cfg.mean_reversion is not None
    assert cfg.diffusion is not None


def test_nsde_cfg_rejects_bad_noise_type():
    cfg = NSDECfg(type="simple", noise_type="full_rank")
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------
# TrainerCfg
# ---------------------------------------------------------------------------


def test_trainer_cfg_defaults_validate():
    cfg = TrainerCfg()
    cfg.validate()
    assert cfg.n_paths > 0
    assert cfg.batch_window > 0


def test_trainer_cfg_rejects_bad_ema_alpha():
    cfg = TrainerCfg()
    cfg.early_stopping.enabled = True
    cfg.early_stopping.ema_alpha = 1.5
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------
# BondNet configs
# ---------------------------------------------------------------------------


def test_bondnet_cfg_alias():
    assert BondNetCfg is BaseBondNetCfg


def test_simple_bondnet_cfg_fields():
    cfg = SimpleBondNetCfg(latent_dim=4, bond_feat_dim=8)
    assert cfg.latent_n_layers >= 1
    assert cfg.fusion_n_layers >= 1


def test_film_bondnet_cfg_fields():
    cfg = FiLMBondNetCfg(latent_dim=4, bond_feat_dim=8)
    assert cfg.hidden_dim > 0
