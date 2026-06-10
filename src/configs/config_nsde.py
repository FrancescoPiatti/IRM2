# src/configs/config_nsde.py
from dataclasses import dataclass

from typing import Any
from typing import Mapping
from typing import Literal
from typing import Optional

import warnings

from ..utils.misc import freeze_dict
from ..utils.checks import _check_positive_value


NSDEType = Literal['simple', 'ou']
NoiseType = Literal['diagonal', 'general']
SolverType = Literal['torchsde', 'custom_euler']

def _default_mlp() -> Mapping[str, Any]:
    """Default drift / long-term mean network — unconstrained MLP."""
    return freeze_dict({"type": "mlp"})


def _default_softplus_mlp() -> Mapping[str, Any]:
    """
    Default MLP for fields that must stay non-negative — OU's
    ``mean_reversion`` (κ ≥ 0 for genuine mean reversion) and the
    diffusion network (σ ≥ 0 for a well-posed SDE).
    """
    return freeze_dict({"type": "mlp", "out_activation": "softplus"})


@dataclass
class NSDECfg:
    """
    NSDE configuration.

    The config is type-driven:
    - type="simple" uses `drift` and `diffusion`
    - type="ou" uses `long_term_mean`, `mean_reversion`, and `diffusion`

    Attributes
    ----------
    type : Literal["simple","ou"]
        Selects the NSDE structure.
    noise_type : NoiseType
        Diffusion parameterization type.
    solver : Literal["torchsde", "custom_euler"]
        Which simulation backend to use. ``"torchsde"`` (default) routes
        through ``torchsde.sdeint`` and respects ``method``, ``adjoint``,
        ``rtol`` and ``atol``. ``"custom_euler"`` runs an in-house
        fixed-step Euler-Maruyama loop that skips torchsde's interval-tree
        Brownian-bridge machinery — same numerical scheme as
        ``method="euler"``, but ~7–15× faster on CPU because the
        Lévy-area calculation is unused at the Euler order.
    method : str
        SDE solver method identifier. Only consumed when ``solver="torchsde"``.
    adjoint : bool
        Use adjoint differentiation. Only consumed when ``solver="torchsde"``.
    rtol : float
        Relative tolerance for adaptive solvers. Ignored by ``custom_euler``.
    atol : float
        Absolute tolerance for adaptive solvers. Ignored by ``custom_euler``.
    dt : float
        Default time step used by the solver. Applied by ``torchsde`` as the
        Euler step and read by ``custom_euler`` only as a default `ts` grid.

    drift : Optional[Mapping[str, Any]]
        Drift network config for the baseline NSDE (type="simple").
    diffusion : Optional[Mapping[str, Any]]
        Diffusion network config (used by both types).
    long_term_mean : Optional[Mapping[str, Any]]
        OU long-term mean network config (type="ou").
    mean_reversion : Optional[Mapping[str, Any]]
        OU mean-reversion network config (type="ou").
    """

    # Type
    type: NSDEType = "simple"

    # Solver / SDE settings
    noise_type: NoiseType = "diagonal"
    solver: SolverType = "torchsde"
    method: str = "euler"
    adjoint: bool = False
    rtol: float = 1e-3
    atol: float = 1e-6
    dt: float = 1 / 252

    # Gradient checkpointing on the ``custom_euler`` solver loop. If
    # positive, the Euler iteration is run in chunks of
    # ``checkpoint_chunk_size`` steps; only the chunk INPUTS are saved
    # for backward, and each chunk is re-simulated on the backward pass
    # to reconstruct activations. Trades ~2x compute for an
    # (n_steps / chunk_size)× reduction of the SDE autograd graph
    # footprint. Ignored by the ``torchsde`` backend.
    checkpoint_chunk_size: Optional[int] = None

    # Multiplicative coefficient SCALES — the principled way to set the
    # natural magnitude of the SDE coefficients (preferred over the hard
    # ``*_bound`` clamps below). The drift network output is multiplied by
    # ``drift_scale`` and the diffusion network output by
    # ``diffusion_scale``. Unlike a tanh bound these do NOT saturate: the
    # coefficient starts at the chosen scale but the network can still
    # grow or shrink it freely (gradients flow everywhere). For a
    # short-rate model the diffusion scale fixes the implied rate vol:
    # ``sigma_r ≈ ||decoder_w|| * diffusion_scale``, so a realistic
    # ~1 %/yr vol means ``diffusion_scale ≈ 0.02`` (see
    # model_diagnosis_report.md). Default 1.0 = no rescaling.
    drift_scale: float = 1.0
    diffusion_scale: float = 1.0

    # OU-only: hard cap on the mean-reversion rate kappa, applied as
    # ``kappa = mean_reversion_max * tanh(kappa / mean_reversion_max)``
    # (kappa is >= 0 from softplus, so this maps [0, inf) -> [0, max)).
    # WHY THIS MATTERS: the day-specific signal enters the model only
    # through z0 (the encoder output = the SDE initial condition). Mean
    # reversion pulls z away from z0 toward theta with timescale 1/kappa,
    # *erasing* z0. The yield y(T) integrates r over [0, T]; once
    # T >> 1/kappa the curve no longer depends on the day's input, so it
    # collapses to a (near-)constant, day-independent curve — the "OU
    # gives constant yields" failure. Capping kappa (e.g. 0.3 -> timescale
    # >= ~3y) keeps z0 alive across the 1-10y curve. ``None`` = uncapped.
    mean_reversion_max: Optional[float] = None

    # Optional HARD bounds on the SDE coefficients (applied AFTER the
    # scales above). When set to a positive float, the drift is squashed
    # via ``drift_bound * tanh(raw / drift_bound)`` and likewise for the
    # diffusion. This is a blunt safety clamp against runaway dynamics —
    # it caps the coefficient but SATURATES (kills gradients at the
    # boundary), so it can fight learning if the model legitimately needs
    # a larger coefficient. Prefer ``*_scale`` for setting magnitude and
    # leave these ``None`` unless you actually see a blow-up.
    drift_bound: Optional[float] = None
    diffusion_bound: Optional[float] = None

    # Near-identity initialisation. When set to a small positive float
    # (e.g. 0.1), the OUTPUT linear layer of every drift/diffusion network
    # is rescaled by this factor and its bias zeroed, so the SDE starts
    # almost coefficient-free (drift ~ 0, diffusion ~ constant). The
    # network then *learns* structure from a calm starting point instead
    # of beginning with large random coefficients that make the long
    # Euler unroll — and its backward pass — blow up in the first epochs.
    # ``None`` (default) keeps PyTorch's standard init.
    init_output_scale: Optional[float] = None

    # Simple-only networks
    drift: Optional[Mapping[str, Any]] = None

    # Shared
    diffusion: Optional[Mapping[str, Any]] = None

    # OU-only networks
    long_term_mean: Optional[Mapping[str, Any]] = None
    mean_reversion: Optional[Mapping[str, Any]] = None


    # -------------------------
    # Validation
    # -------------------------

    def validate(self) -> None:
        """
        Validate and normalise the NSDE config.

        Call after all fields are set. This method:
        - Lowercases type and noise_type
        - Fills defaults for missing network configs
        - Nullifies fields that don't apply to the selected type (with warnings)
        - Validates solver parameters (dt, rtol, atol > 0)
        """
        self.type = str(self.type).lower()
        self.noise_type = str(self.noise_type).lower()
        self.solver = str(self.solver).lower()

        if self.noise_type not in ("diagonal", "general"):
            raise ValueError(
                f"Unknown NSDECfg.noise_type='{self.noise_type}'. Expected 'diagonal' or 'general'."
            )

        if self.solver not in ("torchsde", "custom_euler"):
            raise ValueError(
                f"Unknown NSDECfg.solver='{self.solver}'. Expected 'torchsde' or 'custom_euler'."
            )

        # Solver parameter checks
        _check_positive_value(self.dt, 'cfg.dt')
        _check_positive_value(self.rtol, 'cfg.rtol')
        _check_positive_value(self.atol, 'cfg.atol')

        # Coefficient scales — strictly positive.
        _check_positive_value(self.drift_scale, 'cfg.drift_scale')
        _check_positive_value(self.diffusion_scale, 'cfg.diffusion_scale')
        if self.mean_reversion_max is not None:
            _check_positive_value(self.mean_reversion_max, 'cfg.mean_reversion_max')

        # Optional coefficient bounds — positive when provided.
        if self.drift_bound is not None:
            _check_positive_value(self.drift_bound, 'cfg.drift_bound')
        if self.diffusion_bound is not None:
            _check_positive_value(self.diffusion_bound, 'cfg.diffusion_bound')
        if self.init_output_scale is not None:
            _check_positive_value(self.init_output_scale, 'cfg.init_output_scale')

        if self.type == "simple":
            # Fill defaults — diffusion must stay non-negative.
            if self.drift is None:
                self.drift = _default_mlp()
            if self.diffusion is None:
                self.diffusion = _default_softplus_mlp()

            # Conflicts: OU fields provided -> ignore (no warning; the
            # gridsearch routinely flips type=simple<->ou and a populated
            # base would otherwise spam UserWarnings — math_review.md §6).
            self.long_term_mean = None
            self.mean_reversion = None

        elif self.type == "ou":
            # Fill defaults — mean reversion AND diffusion must stay
            # non-negative (κ ≥ 0 for genuine mean reversion;
            # σ ≥ 0 for a well-posed SDE) — math_review.md §3.
            if self.long_term_mean is None:
                self.long_term_mean = _default_mlp()
            if self.mean_reversion is None:
                self.mean_reversion = _default_softplus_mlp()
            if self.diffusion is None:
                self.diffusion = _default_softplus_mlp()

            # Conflicts: simple-only fields provided -> ignored silently
            # (same rationale as above).
            self.drift = None

        else:
            raise ValueError(f"Unknown NSDECfg.type='{self.type}'. Expected 'simple' or 'ou'.")


# Convenient defaults 
# Users should call .validate() after modifying.
DEFAULT_NSDECfg_Simple: NSDECfg = NSDECfg(type="simple")
DEFAULT_NSDECfg_Simple.validate()

DEFAULT_NSDECfg_OU: NSDECfg = NSDECfg(type="ou")
DEFAULT_NSDECfg_OU.validate()
