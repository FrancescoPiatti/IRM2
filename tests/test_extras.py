"""
Additional unit tests inherited (and modernised) from ``tests_old/``.

Picks up coverage we didn't yet have in the main test files:

- generator-level parametrised shape tests (including Mamba),
- Encoder ``simple`` and ``hierarchical`` modes across `combine` variants,
- Encoder ``out_norm`` parametrisation,
- NSDE general-noise factory + z0 expansion + wrong-batch rejection,
- ``WarmupCosineScheduler`` endpoint / phase behaviour and `build_scheduler`,
- Config "conflicting fields" warnings + immutability of default network specs.
"""
from types import MappingProxyType

import pytest
import torch

from src.configs.config_encoder import EncoderCfg
from src.configs.config_nsde import (
    NSDECfg,
    DEFAULT_NSDECfg_Simple,
    DEFAULT_NSDECfg_OU,
)
from src.configs.config_nn import (
    DEFAULT_CONFIG_MLP,
    DEFAULT_CONFIG_RNN,
    DEFAULT_CONFIG_LSTM,
    DEFAULT_CONFIG_GRU,
    DEFAULT_CONFIG_MAMBA,
    DEFAULT_CONFIG_CONSTANT,
    DEFAULT_CONFIG_AFFINE,
)
from src.configs.config_trainer import SchedulerCfg

from src.models.encoders import Encoder
from src.models.nsde import (
    Simple_NeuralSDE,
    OU_NeuralSDE,
    create_nsde_from_config,
)
from src.nn.generator import create_network_from_config
from src.training.train_utils import WarmupCosineScheduler, build_scheduler


# ---------------------------------------------------------------------------
# Generator shape tests (extra: parametrised across every type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "net_type, default_cfg",
    [
        ("rnn", DEFAULT_CONFIG_RNN),
        ("lstm", DEFAULT_CONFIG_LSTM),
        ("gru", DEFAULT_CONFIG_GRU),
        ("mamba", DEFAULT_CONFIG_MAMBA),
    ],
)
def test_recurrent_generator_shapes(net_type, default_cfg):
    net = create_network_from_config(default_cfg, input_dim=10, output_dim=7)

    x3 = torch.randn(5, 13, 10)
    y3 = net(x3)
    y3_seq = net(x3, return_sequence=True)
    assert y3.shape == (5, 7)
    assert y3_seq.shape == (5, 13, 7)


def test_constant_generator_ignores_input_values():
    const = create_network_from_config(DEFAULT_CONFIG_CONSTANT, input_dim=10, output_dim=7)
    x = torch.randn(4, 10)
    y = const(x)
    # Every row of the output equals every other row (constant per-batch)
    assert torch.allclose(y[0], y[1])
    assert torch.allclose(y[0], y[3])


def test_affine_generator_lazy_input_dim():
    cfg = {"type": "affine", "bias": True, "out_activation": "identity"}
    aff = create_network_from_config(cfg, input_dim=None, output_dim=7)
    y = aff(torch.randn(5, 10))
    assert y.shape == (5, 7)


# ---------------------------------------------------------------------------
# Encoder — hierarchical combine variants + out_norm parametrisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combine", ["concat", "add", "project"])
def test_encoder_hierarchical_modes(combine):
    cfg = EncoderCfg(mode="hierarchical", combine=combine)
    enc = Encoder(config=cfg, output_dim=7, input_dim=10)

    fast_x = torch.randn(5, 13, 10)
    slow_x = torch.randn(5, 13, 10)
    y = enc((fast_x, slow_x))
    y_seq = enc((fast_x, slow_x), return_sequence=True)
    assert y.shape == (5, 7)
    assert y_seq.shape == (5, 13, 7)


@pytest.mark.parametrize("out_norm", ["layernorm", "rmsnorm", "none", None])
def test_encoder_out_norm_variants(out_norm):
    cfg = EncoderCfg(mode="simple", out_norm=out_norm)
    enc = Encoder(config=cfg, output_dim=7, input_dim=10)
    y = enc(torch.randn(5, 13, 10))
    assert y.shape == (5, 7)


# ---------------------------------------------------------------------------
# NSDE — factory routing and z0 acceptance
# ---------------------------------------------------------------------------


def test_simple_nsde_general_noise():
    cfg = {
        "type": "simple",
        "noise_type": "general",
        "diffusion": {"type": "mlp", "activation": "gelu"},
    }
    nsde = create_nsde_from_config(cfg, latent_dim=16, noise_dim=5)
    ts = torch.linspace(0.0, 0.5, 32)
    z0 = torch.randn(16)
    out = nsde(ts, z0, n_paths=12)
    assert out.shape == (12, 32, 16)


def test_factory_returns_correct_types():
    nsde_simple = create_nsde_from_config({"type": "simple"}, latent_dim=16, noise_dim=5)
    nsde_ou     = create_nsde_from_config({"type": "ou"},     latent_dim=16, noise_dim=5)
    assert isinstance(nsde_simple, Simple_NeuralSDE)
    assert isinstance(nsde_ou,     OU_NeuralSDE)


def test_nsde_accepts_per_path_z0_and_rejects_wrong_batch():
    nsde = Simple_NeuralSDE(latent_dim=16, noise_dim=5)
    ts = torch.linspace(0.0, 0.5, 16)

    z0_batched = torch.randn(10, 16)
    out = nsde(ts, z0_batched, n_paths=10)
    assert out.shape == (10, 16, 16)

    z0_wrong = torch.randn(9, 16)
    with pytest.raises((ValueError, AssertionError)):
        _ = nsde(ts, z0_wrong, n_paths=10)


def test_custom_euler_checkpointed_matches_uncheckpointed():
    """
    Gradient checkpointing on the Euler loop must produce numerically
    identical forward output AND identical parameter gradients as the
    plain solver, modulo float-rounding noise from RNG-state preservation.
    """
    torch.manual_seed(0)
    cfg_plain = NSDECfg(type="simple", noise_type="diagonal")
    cfg_plain.solver = "custom_euler"
    cfg_plain.validate()
    nsde_plain = Simple_NeuralSDE(latent_dim=4, config=cfg_plain)
    nsde_plain.train()

    torch.manual_seed(0)
    cfg_ckpt = NSDECfg(type="simple", noise_type="diagonal")
    cfg_ckpt.solver = "custom_euler"
    cfg_ckpt.checkpoint_chunk_size = 8
    cfg_ckpt.validate()
    nsde_ckpt = Simple_NeuralSDE(latent_dim=4, config=cfg_ckpt)
    nsde_ckpt.train()
    nsde_ckpt.load_state_dict(nsde_plain.state_dict())

    ts = torch.linspace(0.0, 1.0, 65)
    z0 = torch.randn(6, 4, requires_grad=True)
    z0_c = z0.detach().clone().requires_grad_(True)

    torch.manual_seed(42)
    out_plain = nsde_plain(ts, z0, n_paths=6)
    torch.manual_seed(42)
    out_ckpt  = nsde_ckpt(ts, z0_c, n_paths=6)

    assert torch.allclose(out_plain, out_ckpt, atol=1e-5, rtol=1e-5)

    out_plain.sum().backward()
    out_ckpt.sum().backward()

    # Parameter gradients should agree (chunk-by-chunk recomputation
    # reuses the same RNG state thanks to preserve_rng_state=True).
    for (n1, p1), (n2, p2) in zip(nsde_plain.named_parameters(), nsde_ckpt.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None, f"missing grad for {n1}"
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4, rtol=1e-4), (
            f"grad mismatch for {n1}: max |diff| = {(p1.grad - p2.grad).abs().max().item():.3e}"
        )


def test_pack_tz_resets_cache_on_dtype_change():
    """Regression test for math_review.md §14 — the time-column cache must
    be rebuilt when `z`'s dtype changes (e.g. when entering / leaving
    ``torch.amp.autocast``). Otherwise we'd silently mix fp32 and bf16."""
    nsde = Simple_NeuralSDE(latent_dim=4)

    z_fp32 = torch.randn(8, 4, dtype=torch.float32)
    out_a = nsde._pack_tz(0.5, z_fp32)
    assert out_a.dtype == torch.float32

    z_bf16 = torch.randn(8, 4, dtype=torch.bfloat16)
    out_b = nsde._pack_tz(0.5, z_bf16)
    assert out_b.dtype == torch.bfloat16
    assert nsde._t_col.dtype == torch.bfloat16

    out_c = nsde._pack_tz(0.5, z_fp32)
    assert out_c.dtype == torch.float32
    assert nsde._t_col.dtype == torch.float32


# ---------------------------------------------------------------------------
# Configs — conflicting-fields warnings + default-network immutability
# ---------------------------------------------------------------------------


def _assert_mapping_is_immutable(m):
    """Setting an item should fail for frozen mappings (MappingProxyType)."""
    with pytest.raises(Exception):
        m["__test__"] = 123  # type: ignore[index]


def test_encoder_cfg_conflicting_fields_warn():
    cfg = EncoderCfg(
        mode="simple",
        net={"type": "lstm"},
        fast_net={"type": "gru"},
        slow_net={"type": "lstm"},
        combine="concat",
    )
    with pytest.warns(UserWarning):
        cfg.validate()
    # Hierarchical fields are nulled out in simple mode
    assert cfg.fast_net is None and cfg.slow_net is None and cfg.combine is None


def test_nsde_cfg_drops_fields_for_wrong_type_silently():
    # Used to warn; now silent (math_review.md §6) — the gridsearch
    # routinely flips nsde.type and a populated base would spam warnings.
    cfg = NSDECfg(
        type="simple",
        drift={"type": "mlp"},
        diffusion={"type": "mlp"},
        long_term_mean={"type": "mlp"},
        mean_reversion={"type": "mlp"},
    )
    cfg.validate()
    # OU fields are nulled out
    assert cfg.long_term_mean is None and cfg.mean_reversion is None


def test_default_nsde_cfg_net_mappings_are_immutable():
    """`freeze_dict` should make the default network specs frozen."""
    cfg_s = DEFAULT_NSDECfg_Simple
    _assert_mapping_is_immutable(cfg_s.drift)
    _assert_mapping_is_immutable(cfg_s.diffusion)

    cfg_ou = DEFAULT_NSDECfg_OU
    _assert_mapping_is_immutable(cfg_ou.long_term_mean)
    _assert_mapping_is_immutable(cfg_ou.mean_reversion)
    _assert_mapping_is_immutable(cfg_ou.diffusion)


def test_default_encoder_cfg_net_mapping_is_immutable():
    cfg = EncoderCfg(mode="simple")
    cfg.validate()
    assert cfg.net is not None
    _assert_mapping_is_immutable(cfg.net)


# ---------------------------------------------------------------------------
# WarmupCosineScheduler — phase behaviour and endpoints
# ---------------------------------------------------------------------------


def _dummy_optimizer(lr: float = 0.1):
    """Tiny SGD optimizer used by the scheduler tests below."""
    model = torch.nn.Linear(2, 2)
    return torch.optim.SGD(model.parameters(), lr=lr)


# The scheduler tests deliberately call ``sched.step()`` without doing any
# training in between (we're just inspecting the LR-schedule shape, not
# running optimisation). PyTorch's `_LRScheduler` therefore warns on every
# such call. Scope the suppression to these tests so unrelated UserWarnings
# still surface.
_WARMUP_FILTER = pytest.mark.filterwarnings(
    "ignore::UserWarning:torch.optim.lr_scheduler",
)


@_WARMUP_FILTER
def test_warmup_cosine_warmup_phase_is_monotone_increasing():
    opt = _dummy_optimizer(lr=0.1)
    sched = WarmupCosineScheduler(opt, warmup_epochs=5, max_epochs=20, eta_min=0.0)
    lrs = []
    for _ in range(5):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert all(lrs[i] > lrs[i - 1] for i in range(1, len(lrs)))


@_WARMUP_FILTER
def test_warmup_cosine_cosine_phase_is_non_increasing():
    opt = _dummy_optimizer(lr=0.1)
    sched = WarmupCosineScheduler(opt, warmup_epochs=2, max_epochs=20, eta_min=0.0)
    for _ in range(2):
        sched.step()
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert all(lrs[i] <= lrs[i - 1] + 1e-9 for i in range(1, len(lrs)))


@_WARMUP_FILTER
def test_warmup_cosine_endpoints():
    base_lr, eta_min = 0.1, 0.001
    opt = _dummy_optimizer(lr=base_lr)
    sched = WarmupCosineScheduler(opt, warmup_epochs=10, max_epochs=100, eta_min=eta_min)
    assert abs(opt.param_groups[0]["lr"] - eta_min) < 1e-7
    for _ in range(10):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - base_lr) < 1e-7


def test_warmup_cosine_rejects_bad_params():
    opt = _dummy_optimizer()
    with pytest.raises(ValueError):
        WarmupCosineScheduler(opt, warmup_epochs=-1, max_epochs=10)
    with pytest.raises(ValueError):
        WarmupCosineScheduler(opt, warmup_epochs=10, max_epochs=10)


@_WARMUP_FILTER
def test_build_scheduler_warmup_cosine_routes():
    opt = _dummy_optimizer(lr=0.1)
    cfg = SchedulerCfg(name="warmup_cosine", params={"warmup_epochs": 5, "max_epochs": 50})
    sched = build_scheduler(opt, cfg)
    assert isinstance(sched, WarmupCosineScheduler)
    sched.step()


# ---------------------------------------------------------------------------
# custom_euler solver + batched window simulation
# ---------------------------------------------------------------------------


def test_custom_euler_returns_correct_shape_and_backprops():
    """custom_euler should produce (n_paths, T, latent_dim) and be differentiable."""
    cfg = NSDECfg(type="simple"); cfg.solver = "custom_euler"
    cfg.validate()
    nsde = Simple_NeuralSDE(latent_dim=8, config=cfg)

    ts = torch.linspace(0.0, 1.0, 64)
    z0 = torch.randn(8)
    out = nsde(ts, z0, n_paths=16)
    assert out.shape == (16, 64, 8)

    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in nsde.parameters())


def test_custom_euler_general_noise():
    cfg = NSDECfg(type="simple", noise_type="general"); cfg.solver = "custom_euler"
    cfg.validate()
    nsde = Simple_NeuralSDE(latent_dim=6, noise_dim=3, config=cfg)
    ts = torch.linspace(0.0, 1.0, 32)
    z0 = torch.randn(6)
    out = nsde(ts, z0, n_paths=8)
    assert out.shape == (8, 32, 6)


def test_custom_euler_distribution_matches_torchsde():
    """
    With matched weights and many paths, custom_euler and torchsde must
    agree in distribution. We check mean / std of the final-step state.
    """
    torch.manual_seed(7)
    cfg_t = NSDECfg(type="simple"); cfg_t.validate()
    cfg_c = NSDECfg(type="simple"); cfg_c.solver = "custom_euler"; cfg_c.validate()

    nsde_t = Simple_NeuralSDE(latent_dim=8, config=cfg_t)
    nsde_c = Simple_NeuralSDE(latent_dim=8, config=cfg_c)
    nsde_c.load_state_dict(nsde_t.state_dict())

    ts = torch.arange(0.0, 1.0 + 1/64, 1/64)
    z0 = torch.randn(8)
    with torch.no_grad():
        a = nsde_t(ts, z0, n_paths=256)[..., -1, :]
        b = nsde_c(ts, z0, n_paths=256)[..., -1, :]
    # Allow a generous tolerance — these are MC estimates, not exact paths
    assert torch.allclose(a.mean(0), b.mean(0), atol=0.5)
    assert torch.allclose(a.std(0),  b.std(0),  atol=0.5)


def test_nsde_cfg_rejects_bad_solver():
    cfg = NSDECfg(type="simple"); cfg.solver = "magic"
    with pytest.raises(ValueError, match="solver"):
        cfg.validate()


# ---------------------------------------------------------------------------
# Loss weights (math_review.md §1 / optimization_plan.md §10.1 P1)
# ---------------------------------------------------------------------------


def _build_joint_trainer(tmp_path, *, lw_yield=1.0, lw_sr=1.0, lw_fut=1.0, seed=0):
    """Build a small joint YC+futures trainer + first-window batch."""
    import torch
    from datetime import datetime
    from src.configs import (
        DataLoaderCfg, EncoderCfg, NSDECfg, SimpleBondNetCfg,
        TrainerCfg, LossWeightsCfg,
    )
    from src.dataloaders import MarketDataLoader
    from src.models.short_rate_model import ShortRateModel
    from src.training.trainer import Trainer

    torch.manual_seed(seed)

    dl = MarketDataLoader(DataLoaderCfg(
        data_path="data2",
        start_date=datetime(2021, 1, 1), end_date=datetime(2021, 6, 30),
        max_maturity=3,
        enable_yield=True, enable_short_rate=True, enable_futures=True,
    ))
    enc = EncoderCfg(mode="simple")
    nsde = NSDECfg(type="simple", noise_type="diagonal"); nsde.solver = "custom_euler"
    bondnet = SimpleBondNetCfg(
        latent_dim=4, bond_feat_dim=8,
        latent_n_layers=1, latent_n_units=4,
        bond_n_layers=1, bond_n_units=4,
        fusion_n_layers=1, fusion_n_units=4,
        output_positive=True,
    )
    model = ShortRateModel(
        name="lw", encoder=enc, nsde=nsde, bondnet=bondnet, latent_dim=4,
    )

    tcfg = TrainerCfg()
    tcfg.results_root = str(tmp_path)
    tcfg.n_paths = 4; tcfg.batch_window = 2; tcfg.window_step = 1
    tcfg.lookback = 5; tcfg.lookback_freq = 1
    tcfg.dt = 1 / 32; tcfg.early_stopping.enabled = False
    tcfg.loss_weights = LossWeightsCfg(
        yield_curve=lw_yield, short_rate=lw_sr, futures=lw_fut,
    )
    tr = Trainer(model=model, dataloader=dl, config=tcfg, device="cpu")

    batch = list(dl.calendar.dates[60:62])
    return tr, batch


def test_loss_weights_record_raw_components_and_scale_grad(tmp_path):
    """
    The per-target component logged in ``loss_components`` should be the
    RAW (unweighted) loss — that's the signal users monitor. The
    backward gradients should still be scaled by λ.
    """
    import torch
    # Run once with all weights = 1
    tr1, batch = _build_joint_trainer(tmp_path / "w1", seed=0)
    d_loss_1, comps_1 = tr1._forward_one_date(batch[0], return_components=True)
    # Same setup, futures weight = 0.01
    tr2, _ = _build_joint_trainer(tmp_path / "w2", seed=0,
                                  lw_yield=1.0, lw_sr=1.0, lw_fut=0.01)
    d_loss_2, comps_2 = tr2._forward_one_date(batch[0], return_components=True)

    # Raw components should match (same weights × same network init).
    for key in ("yield", "short_rate", "futures"):
        assert comps_1[key] == pytest.approx(comps_2[key], rel=1e-5), (
            f"raw component '{key}' should be unweighted: "
            f"{comps_1[key]} vs {comps_2[key]}"
        )

    # The actual scalar loss should obey:
    # loss_2 = comps['yield'] + comps['short_rate'] + 0.01 * comps['futures']
    expected_2 = (
        comps_2["yield"] + comps_2["short_rate"] + 0.01 * comps_2["futures"]
    )
    assert d_loss_2.item() == pytest.approx(expected_2, rel=1e-5)

    # And loss_1 = sum of raw components.
    expected_1 = comps_1["yield"] + comps_1["short_rate"] + comps_1["futures"]
    assert d_loss_1.item() == pytest.approx(expected_1, rel=1e-5)


def test_loss_weights_zero_futures_skips_that_branch(tmp_path):
    """``loss_weights.futures = 0`` should disable the futures branch
    entirely — no futures component is logged, no grad reaches BondNet."""
    import torch
    tr, batch = _build_joint_trainer(tmp_path, lw_fut=0.0, seed=1)
    d_loss, comps = tr._forward_one_date(batch[0], return_components=True)
    assert "futures" not in comps, comps
    assert d_loss.item() == pytest.approx(
        comps.get("yield", 0.0) + comps.get("short_rate", 0.0), rel=1e-5,
    )

    # No futures grad should flow to BondNet parameters.
    d_loss.backward()
    bondnet_grads = [
        p.grad.abs().sum().item()
        for p in tr.model.bondnet.parameters() if p.grad is not None
    ]
    assert all(g == 0.0 for g in bondnet_grads), bondnet_grads


def test_trainer_cfg_rejects_negative_loss_weight():
    from src.configs import TrainerCfg
    cfg = TrainerCfg()
    cfg.loss_weights.futures = -1.0
    with pytest.raises(ValueError, match="loss_weights"):
        cfg.validate()


def test_optuna_gridsearch_runs_and_returns_results(tmp_path):
    """
    Tiny 2-trial grid that exercises:
    - the new `nsde.solver` field flowing through `_set_attr_path`,
    - that `OptunaGridSearch.run()` actually returns a `GridSearchResults`
      (a regression test for the bug fixed alongside the solver work).
    """
    import logging
    import optuna
    from datetime import datetime
    from src import MarketDataLoader, ShortRateModel, Trainer, OptunaGridSearch
    from src.configs import DataLoaderCfg
    from src.types.gridsearch_types import GridSearchResults

    logging.getLogger().setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dl = MarketDataLoader(DataLoaderCfg(
        data_path="data2",
        start_date=datetime(2021, 1, 1), end_date=datetime(2021, 6, 30),
        max_maturity=3,
    ))

    from src.configs import TrainerCfg
    tr_cfg = TrainerCfg()
    tr_cfg.results_root = str(tmp_path)
    tr_cfg.n_paths = 8; tr_cfg.batch_window = 3; tr_cfg.window_step = 1
    tr_cfg.lookback = 5; tr_cfg.lookback_freq = 1; tr_cfg.dt = 1 / 32
    tr_cfg.early_stopping.enabled = False

    search = OptunaGridSearch(
        param_grid={
            "model.latent_dim": [4],
            "nsde.solver": ["torchsde", "custom_euler"],
        },
        dataloader=dl,
        base_encoder_cfg=EncoderCfg(mode="simple"),
        base_nsde_cfg=NSDECfg(type="simple"),
        base_trainer_cfg=tr_cfg,
        model_cls=ShortRateModel,
        trainer_cls=Trainer,
        direction="minimize",
        study_name="extras",
    )
    results = search.run(
        num_epochs=1,
        train_start_date=datetime(2021, 1, 15),
        train_end_date=datetime(2021, 2, 15),
        eval_start_date=datetime(2021, 3, 1),
        eval_end_date=datetime(2021, 3, 1),
        eval_step=1,
        save_eval=False,
    )
    # The bug we're guarding against returned None here.
    assert isinstance(results, GridSearchResults)
    assert len(results.trials) == 2
    solvers_tried = {tr.params["nsde.solver"] for tr in results.trials}
    assert solvers_tried == {"torchsde", "custom_euler"}


@pytest.mark.parametrize("solver", ["torchsde", "custom_euler"])
def test_trainer_window_forward_works_with_both_solvers(solver, tmp_path):
    """End-to-end window forward+backward; gradients reach encoder + bondnet."""
    from datetime import datetime
    from src.configs import DataLoaderCfg, SimpleBondNetCfg, TrainerCfg
    from src.dataloaders import MarketDataLoader
    from src.models.short_rate_model import ShortRateModel
    from src.training.trainer import Trainer

    torch.manual_seed(0)
    dl = MarketDataLoader(DataLoaderCfg(
        data_path="data2",
        start_date=datetime(2021, 1, 1), end_date=datetime(2021, 6, 30),
        max_maturity=5, enable_yield=True, enable_short_rate=True, enable_futures=True,
    ))
    snap = dl.get_snapshot(dl.calendar.dates[60])
    bondnet_cfg = SimpleBondNetCfg(
        latent_dim=8, bond_feat_dim=snap.bonds_metadata.features.shape[1],
        latent_n_layers=1, latent_n_units=8,
        bond_n_layers=1, bond_n_units=8,
        fusion_n_layers=1, fusion_n_units=8,
        output_positive=True,
    )
    nsde_cfg = NSDECfg(type="simple"); nsde_cfg.solver = solver
    model = ShortRateModel(
        name=f"t_{solver}",
        encoder=EncoderCfg(mode="simple"),
        nsde=nsde_cfg, bondnet=bondnet_cfg, latent_dim=8,
    )
    tc = TrainerCfg()
    tc.n_paths = 8; tc.batch_window = 3; tc.window_step = 1
    tc.lookback = 6; tc.lookback_freq = 1; tc.dt = 1 / 64
    tc.early_stopping.enabled = False
    tc.results_root = str(tmp_path); tc.run_name = f"t_{solver}"
    tr = Trainer(model=model, dataloader=dl, config=tc, device="cpu")

    batch = [dl.calendar.dates[60 + i] for i in range(3)]
    fl, lt, n = tr._train_one_window(batch)
    assert n == 3
    assert lt.requires_grad
    lt.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.bondnet.parameters())
