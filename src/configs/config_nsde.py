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

# Change here to a better one
def _default_mlp() -> Mapping[str, Any]:
    return freeze_dict({"type": "mlp"})


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

        if self.type == "simple":
            # Fill defaults
            if self.drift is None:
                self.drift = _default_mlp()
            if self.diffusion is None:
                self.diffusion = _default_mlp()

            # Conflicts: OU fields provided -> ignore with warning
            if self.long_term_mean is not None or self.mean_reversion is not None:
                warnings.warn(
                    "NSDECfg(type='simple'): long_term_mean/mean_reversion were provided but will be ignored.",
                    category=UserWarning,
                    stacklevel=2,
                )

            # Remove OU-only fields
            self.long_term_mean = None
            self.mean_reversion = None

        elif self.type == "ou":
            # Fill defaults
            if self.long_term_mean is None:
                self.long_term_mean = _default_mlp()
            if self.mean_reversion is None:
                self.mean_reversion = _default_mlp()
            if self.diffusion is None:
                self.diffusion = _default_mlp()

            # Conflicts: simple-only fields provided -> ignore with warning
            if self.drift is not None:
                warnings.warn(
                    "NSDECfg(type='ou'): drift was provided but will be ignored.",
                    category=UserWarning,
                    stacklevel=2,
                )

            # Remove simple-only fields
            self.drift = None

        else:
            raise ValueError(f"Unknown NSDECfg.type='{self.type}'. Expected 'simple' or 'ou'.")


# Convenient defaults 
# Users should call .validate() after modifying.
DEFAULT_NSDECfg_Simple: NSDECfg = NSDECfg(type="simple")
DEFAULT_NSDECfg_Simple.validate()

DEFAULT_NSDECfg_OU: NSDECfg = NSDECfg(type="ou")
DEFAULT_NSDECfg_OU.validate()
