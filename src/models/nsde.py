# src/models/nsde.py
import torch
from torch import nn
from torch import Tensor
import torchsde

from abc import abstractmethod

from dataclasses import replace

from typing import Any
from typing import Mapping
from typing import Union
from typing import Optional

from ..utils.checks import _check_positive_integer_value
from ..utils.checks import _check_positive_value

from ..nn.generator import create_network_from_config

from ..configs.config_nsde import NSDECfg
from ..configs.config_nsde import DEFAULT_NSDECfg_Simple
from ..configs.config_nsde import DEFAULT_NSDECfg_OU


# ===========================================================================
# When implementing a new NeuralSDE:
# 1. add a new config dataclass in src/configs/config_nsde.py with a 'type' discriminator
# 2. implement a new module inheriting BaseNSDE and define f and g
# 3. update create_nsde_from_config to route to your new class
# ===========================================================================


class BaseNSDE(nn.Module):
    """
    Base class for latent Neural SDE models (torchsde-compatible).

    This module wraps solver-level configuration (time step, tolerances, method)
    and exposes a single forward interface that simulates paths via torchsde.

    Subclasses must implement f() and g().

    Attributes
    ----------
    cfg : NSDECfg
        NSDE configuration (solver settings + network configs).
    latent_dim : int
        Latent state dimension.
    noise_dim : int
        Brownian motion dimension.
    diffusion_out : int
        Output dimension of the diffusion network (depends on noise_type).
    noise_type : str
        torchsde noise type ("diagonal" or "general"), taken from cfg.noise_type.
    sde_type : str
        torchsde SDE type. This code assumes Itô SDEs ("ito").
    dt : float
        Default solver time step.
    method : str
        torchsde solver method name.
    adjoint : bool
        If True, uses torchsde.sdeint_adjoint for memory-efficient gradients.
    rtol : float
        Relative tolerance for adaptive solvers.
    atol : float
        Absolute tolerance for adaptive solvers.
    """

    def __init__(self,
                 config: NSDECfg,
                 latent_dim: int,
                 noise_dim: Optional[int] = None):

        super().__init__()

        # Validate config (safety net)
        config.validate()

        self.cfg = config

        # torchsde API expects these attributes
        self.noise_type = self.cfg.noise_type
        self.sde_type = 'ito'

        # Time step
        self.dt = float(config.dt)

        # Settings
        self.adjoint = config.adjoint
        self.method = config.method
        self.rtol = float(config.rtol)
        self.atol = float(config.atol)
        self.solver = str(getattr(config, "solver", "torchsde")).lower()

        # Optional smooth coefficient bounds (None => unbounded).
        db = getattr(config, "drift_bound", None)
        gb = getattr(config, "diffusion_bound", None)
        self.drift_bound = float(db) if db else None
        self.diffusion_bound = float(gb) if gb else None

        # Optional near-identity init scale (None => standard init).
        ios = getattr(config, "init_output_scale", None)
        self.init_output_scale = float(ios) if ios else None

        self.sdeint_fn = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

        # ------------------------------------------
        # Shared dimension logic (deduped from subclasses)
        # ------------------------------------------
        _check_positive_integer_value(latent_dim, 'latent_dim')
        self.latent_dim = int(latent_dim)

        # Noise dim is enabled only when noise_type='general'
        if self.noise_type == 'general':
            if noise_dim is not None:
                _check_positive_integer_value(noise_dim, 'noise_dim')
                self.noise_dim = int(noise_dim)
            else:
                raise ValueError("Noise dim has to be assigned if noise_type = 'general'")
        else:
            self.noise_dim = self.latent_dim

        # Diffusion output dim depends on noise structure
        if self.noise_type == 'diagonal':
            self.diffusion_out = self.latent_dim
        elif self.noise_type == 'general':
            self.diffusion_out = self.latent_dim * self.noise_dim

        # Reusable time-column buffer for the ``[z | t]`` packing inside f/g.
        # torchsde calls f and g hundreds-to-thousands of times per
        # simulation, so we cache the (n_paths, 1) time tensor across steps
        # and only fill_ the scalar value. The cache is invalidated when
        # n_paths / device / dtype change between simulations.
        #
        # Aliasing is safe for the time column because:
        #   * it carries no gradient (we never differentiate wrt t),
        #   * `torch.cat([z, t_col], dim=1)` materialises a *fresh* output
        #     tensor per call, so the next step's fill_ cannot corrupt the
        #     already-recorded forward graph for the previous step.
        self._t_col: Optional[Tensor] = None


    def _pack_tz(self, t: float, z: Tensor) -> Tensor:
        """
        Return ``[z | t]`` of shape (n_paths, latent_dim + 1) as the input to
        drift / diffusion networks. Allocates the time column once per
        simulation and reuses it for every solver step.
        """
        n_paths = z.size(0)
        t_col = self._t_col
        if (
            t_col is None
            or t_col.shape[0] != n_paths
            or t_col.device != z.device
            or t_col.dtype != z.dtype
        ):
            t_col = torch.empty((n_paths, 1), device=z.device, dtype=z.dtype)
            self._t_col = t_col

        t_col.fill_(float(t))
        return torch.cat([z, t_col], dim=1)


    # -------------------------
    # Abstract Methods
    # -------------------------

    @abstractmethod
    def f(self, t, z):
        """Drift term f(t, z)."""
        raise NotImplementedError


    @abstractmethod
    def g(self, t, z):
        """Diffusion term g(t, z)."""
        raise NotImplementedError


    @staticmethod
    def _soft_bound(x: Tensor, bound: float) -> Tensor:
        """
        Smoothly squash ``x`` into ``(-bound, bound)`` via
        ``bound * tanh(x / bound)``. Near-identity for ``|x| << bound``,
        saturates for ``|x| >> bound``. Used to keep the SDE coefficients
        (and hence the latent state) from running away over a long Euler
        unroll, which is the main source of non-finite gradients through
        the backward chain.
        """
        return bound * torch.tanh(x / bound)


    @staticmethod
    def _shrink_output_layer(net: nn.Module, scale: float) -> None:
        """
        Near-identity init: rescale the *last* ``nn.Linear`` in ``net`` by
        ``scale`` and zero its bias, in-place. This makes the network
        output start near zero (for drift) or near a constant (for a
        softplus diffusion), so the SDE begins almost coefficient-free and
        the long Euler unroll is calm during the fragile first epochs.
        No-op if ``net`` contains no ``nn.Linear``.
        """
        last_linear = None
        for m in net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            return
        with torch.no_grad():
            last_linear.weight.mul_(float(scale))
            if last_linear.bias is not None:
                last_linear.bias.zero_()


    # -------------------------
    # Validation and Helpers
    # -------------------------

    def _expand_z0(self, z0: Tensor, n_paths: int) -> Tensor:
        """
        Normalize z0 to shape (n_paths, latent_dim).

        Accepts:
        - z0 shape (latent_dim,)
        - z0 shape (1, latent_dim)
        - z0 shape (n_paths, latent_dim) already
        """
        assert isinstance(z0, Tensor), "z0 must be a torch.Tensor."

        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)  # (1, latent_dim)

        assert z0.dim() == 2, "z0 must be 1D or 2D (latent_dim,) or (B, latent_dim)."

        if z0.size(0) == 1:
            return z0.expand(n_paths, -1)

        assert z0.size(0) == n_paths, (
            f"z0 batch dimension is {z0.size(0)} but n_paths is {n_paths}. "
            "Provide z0 with shape (latent_dim,) or (1, latent_dim) to broadcast."
        )

        return z0


    def _validate_ts(self, ts: Tensor) -> None:
        """
        Validate a 1D time grid.
        """
        assert isinstance(ts, Tensor), "ts must be a torch.Tensor."
        assert ts.dim() == 1, "ts must be a 1D tensor of time points."
        assert ts.numel() >= 2, "ts must contain at least two time points."


    # -------------------------
    # Forward
    # -------------------------

    def forward(self, ts, z0, n_paths: int = 1000):
        """
        Simulate latent paths.

        Dispatches between two backends based on ``self.solver``:
        - ``"torchsde"`` — wraps ``torchsde.sdeint`` (or ``sdeint_adjoint``).
          Honours ``method``, ``rtol``, ``atol`` and ``dt`` from the config.
        - ``"custom_euler"`` — in-house fixed-step Euler-Maruyama loop.
          Skips torchsde's interval-tree Brownian bridge (much faster on
          CPU). Numerically equivalent to ``method="euler"``.

        Parameters
        ----------
        ts : Tensor
            Time grid of shape (T,). Must be increasing and contain at least
            two points.
        z0 : Tensor
            Initial latent state, shape ``(latent_dim,)``, ``(1, latent_dim)``
            or ``(n_paths, latent_dim)``.
        n_paths : int
            Number of Monte Carlo paths to simulate. Must be > 1.

        Returns
        -------
        Tensor
            Simulated latent paths, shape ``(n_paths, T, latent_dim)``.
        """
        assert isinstance(n_paths, int) and n_paths > 1, "n_paths must be an int > 1"

        self._validate_ts(ts)
        z0 = self._expand_z0(z0, n_paths)  # (n_paths, latent_dim)

        if self.solver == "custom_euler":
            return self._simulate_custom_euler(ts, z0)

        zs = self.sdeint_fn(
            sde=self,
            y0=z0,
            ts=ts,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            dt=self.dt,
            logqp=False,
        )
        # torchsde returns (T, batch, latent_dim)
        return zs.swapaxes(0, 1)


    # -------------------------
    # In-house Euler-Maruyama
    # -------------------------

    def _simulate_custom_euler(self, ts: Tensor, z0: Tensor) -> Tensor:
        """
        Fixed-step Euler-Maruyama using ``self.f`` and ``self.g``.

        Same numerical scheme as ``torchsde.sdeint(..., method="euler")``,
        but without the interval-tree Brownian bridge (no Lévy area, no
        trampolined recursion, no per-interval reseeding). Brownian
        increments are drawn directly via ``torch.randn``.

        Parameters
        ----------
        ts : Tensor
            Increasing 1D time grid (year-fractions).
        z0 : Tensor
            Already path-expanded initial state, shape (n_paths, latent_dim).

        Returns
        -------
        Tensor
            Simulated paths, shape (n_paths, T, latent_dim).

        Notes
        -----
        If ``self.cfg.checkpoint_chunk_size`` is positive, the Euler
        iteration is run in chunks under
        ``torch.utils.checkpoint.checkpoint`` so that the autograd graph
        only holds chunk-boundary states. Brownian increments still
        replay identically on backward (default
        ``preserve_rng_state=True``).
        """
        n_paths = z0.size(0)
        n_steps = ts.numel()
        device = z0.device
        dtype = z0.dtype

        # Step sizes and sqrt(dt) — read once on CPU to avoid per-step .item().
        dts = (ts[1:] - ts[:-1]).tolist()           # length n_steps - 1
        sqrt_dts = [d ** 0.5 for d in dts]
        t_vals = ts[:-1].tolist()                   # left-endpoint of each step

        diagonal = self.noise_type == "diagonal"
        noise_dim = self.noise_dim

        # ------------------------------------------------------------------
        # Single Euler step — used by both the no-checkpoint and the
        # checkpointed branches below.
        # ------------------------------------------------------------------
        def _step(t_i: float, dt_i: float, sqrt_dt: float, z: Tensor) -> Tensor:
            drift = self.f(t_i, z)
            diff  = self.g(t_i, z)
            if diagonal:
                dW = torch.randn_like(z) * sqrt_dt
                return z + drift * dt_i + diff * dW
            dW = torch.randn(n_paths, noise_dim, device=device, dtype=dtype) * sqrt_dt
            return z + drift * dt_i + torch.bmm(diff, dW.unsqueeze(-1)).squeeze(-1)

        chunk = getattr(self.cfg, "checkpoint_chunk_size", None)
        use_checkpointing = bool(chunk) and chunk > 0 and self.training and z0.requires_grad

        # ------------------------------------------------------------------
        # Plain (no-checkpoint) path
        # ------------------------------------------------------------------
        if not use_checkpointing:
            z = z0
            out = [z]
            for i in range(n_steps - 1):
                z = _step(t_vals[i], dts[i], sqrt_dts[i], z)
                out.append(z)
            return torch.stack(out, dim=1)          # (n_paths, T, latent_dim)

        # ------------------------------------------------------------------
        # Checkpointed path: run K-step chunks under
        # ``torch.utils.checkpoint`` so only the chunk INPUT is saved.
        # ------------------------------------------------------------------
        from torch.utils.checkpoint import checkpoint

        n_chunks = (n_steps - 1 + chunk - 1) // chunk
        # We capture the per-chunk schedule into closures so backward
        # recomputation uses the same step sizes / left-times.
        out = [z0]
        z = z0
        for c in range(n_chunks):
            start = c * chunk
            end = min(start + chunk, n_steps - 1)        # exclusive
            chunk_t  = t_vals[start:end]
            chunk_dt = dts[start:end]
            chunk_sd = sqrt_dts[start:end]

            def _run_chunk(z_in, ct=chunk_t, cdt=chunk_dt, csd=chunk_sd):
                zs = []
                cur = z_in
                for j in range(len(ct)):
                    cur = _step(ct[j], cdt[j], csd[j], cur)
                    zs.append(cur)
                # Stack so that the entire chunk's outputs are materialised in
                # the FORWARD (the autograd graph between chunks only needs
                # the chunk-input z; intermediate activations are dropped).
                return torch.stack(zs, dim=0)            # (K_c, n_paths, d_z)

            chunk_out = checkpoint(_run_chunk, z, use_reentrant=False)
            # chunk_out: (K_c, n_paths, d_z) — split back into per-step
            # entries so the final ``torch.stack`` below produces the
            # canonical (n_paths, T, d_z) layout.
            for k in range(chunk_out.shape[0]):
                out.append(chunk_out[k])
            z = chunk_out[-1]

        return torch.stack(out, dim=1)              # (n_paths, T, latent_dim)



class Simple_NeuralSDE(BaseNSDE):
    """
    Simple NeuralSDE with a single drift and diffusion network.
    """

    def __init__(self,
                 latent_dim: int = 64,
                 noise_dim: Optional[int] = None,
                 config: NSDECfg = DEFAULT_NSDECfg_Simple):

        if config.type.lower() != 'simple':
            raise ValueError(f"Simple_NeuralSDE expects cfg.type='simple', got '{config.type}'")

        super().__init__(config=config, latent_dim=latent_dim, noise_dim=noise_dim)

        # Drift f(z, t)
        self.drift = create_network_from_config(
            config=dict(self.cfg.drift),
            input_dim=self.latent_dim+1,
            output_dim=self.latent_dim
        )

        # Diffusion g(z, t)
        self.diffusion = create_network_from_config(
            config=dict(self.cfg.diffusion),
            input_dim=self.latent_dim+1,
            output_dim=self.diffusion_out
        )

        # To be changed
        if self.cfg.drift['type'] not in ('mlp', 'affine', 'constant'):
            raise NotImplementedError
        if self.cfg.diffusion['type'] not in ('mlp', 'affine', 'constant'):
            raise NotImplementedError

        # Near-identity init (optional): start with tiny drift/diffusion
        # output so the first-epoch unroll is calm.
        if self.init_output_scale is not None:
            self._shrink_output_layer(self.drift, self.init_output_scale)
            self._shrink_output_layer(self.diffusion, self.init_output_scale)



    def f(self, t, z) -> Tensor:
        """
        Drift f(t, z). Expects z of shape (n_paths, latent_dim).
        """
        drift = self.drift(self._pack_tz(t, z))            # (n_paths, latent_dim)
        if self.drift_bound is not None:
            drift = self._soft_bound(drift, self.drift_bound)
        return drift


    def g(self, t, z) -> Tensor:
        """
        Diffusion g(t, z). Expects z of shape (n_paths, latent_dim).
        """
        out = self.diffusion(self._pack_tz(t, z))
        if self.diffusion_bound is not None:
            out = self._soft_bound(out, self.diffusion_bound)
        if self.noise_type == 'diagonal':
            return out
        return out.view(-1, self.latent_dim, self.noise_dim)   # (n_paths, latent_dim, noise_dim)



class OU_NeuralSDE(BaseNSDE):
    """
    Ornstein Uhlenbeck-structured NeuralSDE.
    """

    def __init__(self,
                 latent_dim: int = 64,
                 noise_dim: Optional[int] = None,
                 config: NSDECfg = DEFAULT_NSDECfg_OU):

        if config.type.lower() != 'ou':
            raise ValueError(f"OU_NeuralSDE expects cfg.type='ou', got '{config.type}'")

        super().__init__(config=config, latent_dim=latent_dim, noise_dim=noise_dim)

        # Drift networks
        self.long_term_mean = create_network_from_config(
            config=dict(self.cfg.long_term_mean),
            input_dim=self.latent_dim+1,
            output_dim=self.latent_dim,
        )
        self.mean_reversion = create_network_from_config(
            config=dict(self.cfg.mean_reversion),
            input_dim=self.latent_dim+1,
            output_dim=self.latent_dim,
        )

        # Diffusion g(z, t)
        self.diffusion = create_network_from_config(
            config=dict(self.cfg.diffusion),
            input_dim=self.latent_dim+1,
            output_dim=self.diffusion_out,
        )

        # To be changed
        if self.cfg.long_term_mean['type'] not in ('mlp', 'affine', 'constant'):
            raise NotImplementedError
        if self.cfg.mean_reversion['type'] not in ('mlp', 'affine', 'constant'):
            raise NotImplementedError
        if self.cfg.diffusion['type'] not in ('mlp', 'affine', 'constant'):
            raise NotImplementedError

        # Near-identity init (optional): shrink theta / kappa / diffusion
        # output layers so the OU dynamics start gentle.
        if self.init_output_scale is not None:
            self._shrink_output_layer(self.long_term_mean, self.init_output_scale)
            self._shrink_output_layer(self.mean_reversion, self.init_output_scale)
            self._shrink_output_layer(self.diffusion, self.init_output_scale)


    def f(self, t, z) -> Tensor:
        """
        OU drift: kappa(t, z) * (theta(t, z) - z).
        """
        x = self._pack_tz(t, z)
        theta = self.long_term_mean(x)                 # (n_paths, latent_dim)
        kappa = self.mean_reversion(x)                 # (n_paths, latent_dim)
        drift = kappa * (theta - z)
        if self.drift_bound is not None:
            drift = self._soft_bound(drift, self.drift_bound)
        return drift


    def g(self, t, z) -> Tensor:
        """
        Diffusion g(t, z). Expects z of shape (n_paths, latent_dim).
        """
        out = self.diffusion(self._pack_tz(t, z))
        if self.diffusion_bound is not None:
            out = self._soft_bound(out, self.diffusion_bound)
        if self.noise_type == 'diagonal':
            return out
        return out.view(-1, self.latent_dim, self.noise_dim)



# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def create_nsde_from_config(
    config: Union[NSDECfg, Mapping[str, Any]],
    *,
    latent_dim: int,
    noise_dim: Optional[int] = None,
) -> BaseNSDE:
    """
    Factory for NSDE modules.

    Parameters
    ----------
    config : Union[NSDECfg, Mapping[str, Any]]
        Either:
        - an NSDECfg dataclass instance, or
        - a mapping containing at least (optionally) the discriminator key 'type'.
        If 'type' is missing, defaults to "simple".
    latent_dim : int
        Latent state dimension.
    noise_dim : int
        Brownian dimension used when cfg.noise_type="general".

    Returns
    -------
    BaseNSDE
        Instantiated NSDE module matching cfg.type.

    Raises
    ------
    ValueError
        If cfg.type is not supported.
    """
    _check_positive_integer_value(latent_dim, "latent_dim")
    if noise_dim is not None:
        _check_positive_integer_value(noise_dim, "noise_dim")

    # -------------------------------------------------------
    # Resolve cfg (always end up with an NSDECfg instance)
    # -------------------------------------------------------
    if isinstance(config, NSDECfg):
        cfg = config
    else:
        if not isinstance(config, Mapping):
            raise TypeError("config must be an NSDECfg or a Mapping[str, Any].")

        cfg_type = str(config.get("type", "simple")).lower()

        if cfg_type == "simple":
            base = DEFAULT_NSDECfg_Simple
        elif cfg_type == "ou":
            base = DEFAULT_NSDECfg_OU
        else:
            raise ValueError(f"Unsupported NSDE type '{cfg_type}'")

        # Override only known fields of the chosen base dataclass.
        # NSDECfg.validate() will warn+null-out incompatible fields.
        overrides = {k: v for k, v in dict(config).items() if hasattr(base, k)}
        cfg = replace(base, **overrides)

    cfg_type = str(cfg.type).lower()

    # -------------------------------------------------------
    # Route to module
    # -------------------------------------------------------
    if cfg_type == "simple":
        return Simple_NeuralSDE(config=cfg, latent_dim=latent_dim, noise_dim=noise_dim)
    if cfg_type == "ou":
        return OU_NeuralSDE(config=cfg, latent_dim=latent_dim, noise_dim=noise_dim)

    raise ValueError(f"Unsupported NSDE type '{cfg_type}'")
