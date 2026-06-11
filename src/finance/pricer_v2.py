# src/finance/pricer_v2.py
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.nn import Module

from ..types.data_types import MarketSnapshot
from ..types.data_types import YieldCurveTarget
from ..types.data_types import ShortRateTarget
from ..types.data_types import SingleFutureTarget
from ..types.data_types import BatchedFuturesTarget
from ..types.types_utils import Date

from ..utils.checks import _check_positive_integer_value
from ..utils.checks import _check_positive_value


# ---------------------------------------------------------------------
# Date / year-fraction helper
# ---------------------------------------------------------------------

# Single year-fraction convention used across the whole stack: a "year"
# is 252 BUSINESS days. Date deltas are counted in business days
# (``np.busday_count``, Mon-Fri) and divided by 252, so e.g. 91 calendar
# days -> ~63 business days -> ~0.25y (3 months), matching the intuition
# that the SVENY maturity grid [1..10] is "years of 252 trading days".
# Caveat: a calendar year holds ~261 weekdays (no holiday calendar), so
# long horizons stretch by ~3.6% (10 calendar years -> ~10.36). This is
# the accepted slack of the convention — short-horizon quantities
# (futures deliveries, coupon gaps) are the ones that must be accurate.
DAYS_PER_YEAR = 252.0


def to_year_fraction(
    target_dates: Union[Date, List[Date]],
    asof_date: Date,
    *,
    business_days_per_year: float = 252.0,    # kept for back-compat only
) -> Tensor:
    """
    Convert dates to a 1D tensor of year-fractions relative to `asof_date`.

    Year-fractions use the **business-day / 252 convention**: the number
    of weekdays between the two dates (``np.busday_count``) divided by
    252. The ``business_days_per_year`` argument is accepted for
    back-compat but is **not used** — the divisor is hard-wired to
    ``DAYS_PER_YEAR = 252`` so the convention is identical everywhere.

    Parameters
    ----------
    target_dates : Union[Date, List[Date]]
        Target dates. A scalar input is wrapped into a single-element list.
    asof_date : Date
        Reference date.
    business_days_per_year : float
        Ignored. Kept on the signature so old callers don't break.

    Returns
    -------
    Tensor
        1D float32 tensor of year-fractions (business-day basis) with the
        same length as `target_dates`. Negative if a target precedes
        `asof_date`; exactly 0.0 for the same date.
    """
    if not isinstance(target_dates, (list, tuple)):
        target_dates = [target_dates]

    asof = np.datetime64(pd.Timestamp(asof_date).normalize(), "D")
    targets = pd.to_datetime(list(target_dates)).normalize().values.astype("datetime64[D]")
    busdays = np.busday_count(asof, targets).astype(np.float32)
    return torch.from_numpy(busdays / float(DAYS_PER_YEAR))


# ---------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------

class Pricer:
    """
    Pricing utilities for the short-rate model.

    The Pricer takes simulated short-rate / latent paths and converts them
    into model-implied observables (yields, futures, ...).

    Attributes
    ----------
    device : torch.device
        Default device for ad-hoc tensor allocations.
    steps_per_year : int
        Number of solver steps per year used by the simulation grid
        (must match `Trainer.dt`).
    business_days_per_year : float
        Year-fraction convention. Used to convert delivery dates to year
        fractions and to interpret integer day counts.
    use_amp : bool
        If True (and the device is CUDA), wrap the BondNet forward in
        ``torch.amp.autocast`` so it benefits from mixed precision while the
        surrounding pricing arithmetic stays in float32 (per
        ``project_description.md`` §15).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        steps_per_year: int = 252,
        business_days_per_year: float = 252.0,
        use_amp: bool = False,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.steps_per_year = int(steps_per_year)
        self.business_days_per_year = float(business_days_per_year)
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        _check_positive_integer_value(self.steps_per_year, 'steps_per_year')
        _check_positive_value(self.business_days_per_year, 'business_days_per_year')

        # Per-snapshot diagnostics, overwritten on every `price_futures` call.
        # These are read by Trainer / users for monitoring. Default to None so
        # consumers can detect "no futures priced yet" (optimisation_plan §8).
        self.last_bond_stats: Optional[Dict[str, float]] = None
        self.last_ctd_freq: Optional[Tensor] = None

        # BondNet <-> SDE consistency (LSMC). When ``consistency_enabled``
        # is True and ``price_futures`` receives the short-rate
        # ``realisations``, it also computes a Longstaff-Schwartz-style
        # regression loss tying BondNet's bond prices to the *model's own*
        # pathwise-discounted cashflows. Stored here (WITH grad) so the
        # Trainer can add it to the objective. This is what couples the
        # futures channel to the yield-curve dynamics — without it BondNet
        # is a free head and the joint calibration is two unrelated tasks.
        self.consistency_enabled: bool = False
        self.last_consistency_loss: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Yield-curve pricing
    # ------------------------------------------------------------------

    def _to_2d_paths(self, realisations: Tensor) -> Tensor:
        """
        Ensure paths have shape (n_paths, steps).
        Accepts (n_paths, steps) or (n_paths, steps, 1).
        """
        if realisations.dim() == 3 and realisations.size(-1) == 1:
            realisations = realisations.squeeze(-1)
        assert realisations.dim() == 2, (
            f"Expected realisations with shape (n_paths, steps) or (n_paths, steps, 1), "
            f"got {tuple(realisations.shape)}"
        )
        return realisations

    def price_zcb(self, realisations: Tensor, maturities: Tensor) -> Tensor:
        """
        Price zero-coupon bonds P(0, T) from simulated short-rate paths.

        Uses a left Riemann sum for the path-wise integral and a numerically
        stable ``logsumexp`` for the Monte Carlo average of the discount
        factor (project_description §5 + optimisation_plan §5.2).

        Parameters
        ----------
        realisations : Tensor
            Short-rate paths, shape (n_paths, steps) or (n_paths, steps, 1).
        maturities : Tensor
            Maturities in years, shape (M,). Must be positive.

        Returns
        -------
        Tensor
            ZCB prices, shape (M,), in (0, 1].
        """
        realisations = self._to_2d_paths(realisations)

        dt = 1.0 / float(self.steps_per_year)
        n_steps = realisations.size(1)

        # Guard maturities: must be strictly positive (P(0,0)=1 by definition,
        # so a zero maturity is ill-posed for the loss) AND fit inside the
        # simulation horizon (math_review.md §7 + §9).
        if maturities.numel() == 0:
            return torch.empty(0, device=realisations.device, dtype=realisations.dtype)
        if float(maturities.min().item()) <= 0.0:
            raise ValueError(
                f"price_zcb: maturities must be strictly positive; got min={maturities.min().item():.4f}."
            )
        max_grid_years = float(n_steps) * dt
        if float(maturities.max().item()) > max_grid_years + 1e-9:
            raise ValueError(
                f"price_zcb: requested maturity {maturities.max().item():.4f} exceeds the "
                f"simulation horizon {max_grid_years:.4f} (n_steps={n_steps}, "
                f"steps_per_year={self.steps_per_year}). Increase max_maturity or trainer.dt."
            )

        # Round, don't truncate, when mapping maturities onto the grid
        # (project_description §9, optimisation_plan §5.1). Clamp to >= 1 so
        # idx - 1 >= 0 in the index_select below (project_description §5.3).
        idx = torch.round(maturities * float(self.steps_per_year)).long()
        idx = idx.clamp(min=1, max=n_steps)

        cum_int = torch.cumsum(realisations, dim=1) * dt              # (paths, steps)
        integral = cum_int.index_select(1, idx - 1)                   # (paths, M)

        # Stable estimate of P = E[exp(-integral)] via log-sum-exp:
        #     log E[exp(-integral)] = logsumexp(-integral, axis=paths) - log(N)
        # Then exponentiate. Equivalent to ``exp(-integral).mean(0)`` in
        # exact arithmetic but resists underflow when individual ``exp(-·)``
        # would round to zero.
        n_paths = realisations.size(0)
        log_P = torch.logsumexp(-integral, dim=0) - math.log(float(n_paths))
        P = torch.exp(log_P).clamp(1e-12, 1.0)                        # (M,)
        return P

    def price_yield_curve(self, realisations: Tensor, maturities: Tensor) -> Tensor:
        """
        Compute model-implied continuously compounded yields, in DECIMAL units.

        Convention: ``y = -log(P) / T``. To get a percentage figure for
        display, multiply by 100 at the call-site. The whole stack
        (loader, pricer, loss) is now consistently decimal — see
        ``math_review.md`` §1.

        Parameters
        ----------
        realisations : Tensor
            Short-rate paths (decimal), shape (n_paths, steps) or (n_paths, steps, 1).
        maturities : Tensor
            Maturities in years, shape (M,).

        Returns
        -------
        Tensor
            Yields in decimal, shape (M,).
        """
        P = self.price_zcb(realisations=realisations, maturities=maturities.float())
        y = -torch.log(P) / maturities.float()
        return y

    def price_short_rate(self, realisations: Tensor) -> Tensor:
        """
        Sensible short-rate observable: E[r_0] across paths.

        Returns
        -------
        Tensor
            Scalar tensor.
        """
        realisations = self._to_2d_paths(realisations)
        return realisations[:, 0].mean()

    # ------------------------------------------------------------------
    # Futures pricing
    # ------------------------------------------------------------------

    def price_futures(
        self,
        bondnet: Module,
        bond_features: Tensor,
        latent_paths: Tensor,
        simulated_times: Tensor,
        target: Union[SingleFutureTarget, BatchedFuturesTarget],
        realisations: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute model-implied futures prices via cheapest-to-deliver Monte Carlo.

        Parameters
        ----------
        bondnet : Module
            BondNet module. Called as `bondnet(z, bond_features)` and expected
            to return a tensor of shape (...,).
        bond_features : Tensor
            Bond feature matrix aligned with `target.deliverable_ids_flat`,
            shape (N_dlv_flat, bond_feat_dim). The row order must match the
            flattened deliverable list inside `target`.
        latent_paths : Tensor
            Simulated latent paths of shape (n_paths, n_steps, d_z).
        simulated_times : Tensor
            1D tensor of year-fractions at which `latent_paths` was sampled,
            shape (n_steps,).
        target : Union[SingleFutureTarget, BatchedFuturesTarget]
            Futures target(s) to price. A `SingleFutureTarget` is promoted to
            a one-element batch.

        Returns
        -------
        Tensor
            Model-implied futures prices, shape (n_futures,) — one per contract.
        """
        if isinstance(target, SingleFutureTarget):
            # Promote to a batched representation for uniform handling.
            from ..types.types_utils import merge_single_future_targets
            target = merge_single_future_targets([target], device=latent_paths.device)

        if not isinstance(target, BatchedFuturesTarget):
            raise TypeError(
                f"price_futures expects SingleFutureTarget or BatchedFuturesTarget; "
                f"got {type(target).__name__}"
            )

        # Year-fraction from as-of to each delivery date. The loader caches
        # this on `target.metadata["delivery_years"]` so we don't re-convert
        # per snapshot; fall back to recomputing if it's absent.
        cached = target.metadata.get("delivery_years") if hasattr(target, "metadata") else None
        if isinstance(cached, Tensor):
            dlv_years = cached.to(device=latent_paths.device, dtype=latent_paths.dtype)
        else:
            dlv_years = to_year_fraction(
                target.delivery_dates,
                target.asof_date,
                business_days_per_year=self.business_days_per_year,
            ).to(device=latent_paths.device, dtype=latent_paths.dtype)

        # Map delivery year-fractions onto the simulation grid (no look-ahead).
        idx = self._extract_latent_idx_at_delivery(
            simulated_times=simulated_times,
            target_times=dlv_years,
        )                                       # (n_futures,)

        # Broadcast latent state across basket slots in one gather, avoiding the
        # intermediate (n_paths, n_futures, d_z) tensor produced by index_select +
        # repeat_interleave (optimisation_plan §3.3).
        basket_lengths = target.basket_lengths.to(device=latent_paths.device, dtype=torch.long)
        per_slot_idx = idx.repeat_interleave(basket_lengths)            # (N_dlv_flat,)
        z_per_dlv = latent_paths.index_select(dim=1, index=per_slot_idx)  # (n_paths, N_dlv_flat, d_z)

        # Bond features. When ``slot_to_unique`` is present, the loader has
        # passed only the *unique* deliverable feature rows; we gather them up
        # to the per-slot order here (optimisation_plan §3.2). Otherwise
        # bond_features is already aligned with the slot order.
        bf = bond_features.to(device=latent_paths.device, dtype=latent_paths.dtype)
        slot_to_unique = target.metadata.get("slot_to_unique") if hasattr(target, "metadata") else None
        if isinstance(slot_to_unique, Tensor) and bf.shape[0] != per_slot_idx.numel():
            bf = bf.index_select(0, slot_to_unique.to(device=bf.device, dtype=torch.long))
        bf_expanded = bf.unsqueeze(0).expand(z_per_dlv.size(0), -1, -1)  # (n_paths, N_dlv_flat, d_b)

        # BondNet forward: (n_paths, N_dlv_flat). The neural net is the only
        # piece that benefits from autocast — keep the rest of the pricing
        # arithmetic in float32 (project_description §15).
        if self.use_amp:
            with torch.amp.autocast(device_type=latent_paths.device.type, enabled=True):
                bond_values = bondnet(z_per_dlv, bf_expanded)
            bond_values = bond_values.float()
        else:
            bond_values = bondnet(z_per_dlv, bf_expanded)

        # Divide by conversion factors (broadcast along path dim)
        cf = target.conversion_factors_flat.to(device=latent_paths.device, dtype=bond_values.dtype)
        cf_adj = bond_values / cf

        # Segmented min over basket: (n_paths, n_futures)
        ctd = self._segmented_min(cf_adj, basket_lengths)

        # BondNet <-> SDE consistency (LSMC regression). See __init__ note.
        self.last_consistency_loss = None
        if self.consistency_enabled and realisations is not None:
            self.last_consistency_loss = self._lsmc_consistency_loss(
                bond_values=bond_values,
                bond_features=bf,
                per_slot_delivery_idx=per_slot_idx,
                realisations=realisations,
                simulated_times=simulated_times,
            )

        # Per-snapshot diagnostics — cheap and detached so they don't perturb
        # autograd (optimisation_plan §8.1 + §8.2). Recording these lets the
        # trainer / users watch for BondNet collapse and single-bond CTD
        # dominance during early training.
        with torch.no_grad():
            self.last_bond_stats = {
                "mean": float(bond_values.mean().item()),
                "std": float(bond_values.std(unbiased=False).item()),
                "min": float(bond_values.min().item()),
                "max": float(bond_values.max().item()),
            }
            self.last_ctd_freq = self._ctd_selection_freq(cf_adj, basket_lengths)

        # Mean over paths -> (n_futures,)
        return ctd.mean(dim=0)

    # ------------------------------------------------------------------
    # Snapshot pricing
    # ------------------------------------------------------------------

    def price_snapshot(
        self,
        realisations: Tensor,
        snapshot: MarketSnapshot,
        *,
        latent_paths: Optional[Tensor] = None,
        simulated_times: Optional[Tensor] = None,
        bondnet: Optional[Module] = None,
        bond_features: Optional[Tensor] = None,
    ) -> MarketSnapshot:
        """
        Convert simulated paths into model-implied observables matching `snapshot`.

        Parameters
        ----------
        realisations : Tensor
            Decoded short-rate paths, shape (n_paths, steps) or (n_paths, steps, 1).
        snapshot : MarketSnapshot
            Observed snapshot used as a template (defines which targets to compute
            and supplies maturities / delivery dates / baskets).
        latent_paths : Optional[Tensor]
            Latent paths of shape (n_paths, n_steps, d_z). Required iff
            `snapshot.futures` is not None.
        simulated_times : Optional[Tensor]
            Simulation time grid (year-fractions), shape (n_steps,). Required
            iff `snapshot.futures` is not None.
        bondnet : Optional[Module]
            BondNet module. Required iff `snapshot.futures` is not None.
        bond_features : Optional[Tensor]
            Per-slot bond feature matrix aligned with
            `snapshot.futures.deliverable_ids_flat`, shape (N_dlv_flat, d_b).
            If None, the matrix is read from `snapshot.bonds_metadata.features`.

        Returns
        -------
        MarketSnapshot
            Model-implied snapshot aligned with `snapshot`.
        """
        # ----- Yield curve -----
        model_yield_curve = None
        if snapshot.yield_curve is not None:
            maturities = snapshot.yield_curve.maturities
            model_yields = self.price_yield_curve(realisations=realisations, maturities=maturities)
            model_yield_curve = YieldCurveTarget(
                date=snapshot.date,
                maturities=maturities,
                yields=model_yields,
            )

        # ----- Short rate -----
        model_short_rate = None
        if snapshot.short_rate is not None:
            model_r = self.price_short_rate(realisations=realisations)
            model_short_rate = ShortRateTarget(date=snapshot.date, rate=model_r)

        # ----- Futures -----
        model_futures = None
        if snapshot.futures is not None:
            if latent_paths is None or simulated_times is None or bondnet is None:
                raise ValueError(
                    "price_snapshot: snapshot.futures is set but "
                    "latent_paths / simulated_times / bondnet are missing."
                )
            if bond_features is None:
                if snapshot.bonds_metadata is None:
                    raise ValueError(
                        "price_snapshot: snapshot.futures is set but bond_features "
                        "is None and snapshot.bonds_metadata is also None."
                    )
                bond_features = snapshot.bonds_metadata.features

            prices = self.price_futures(
                bondnet=bondnet,
                bond_features=bond_features,
                latent_paths=latent_paths,
                simulated_times=simulated_times,
                target=snapshot.futures,
                realisations=realisations,
            )
            # Preserve any per-instrument metadata the user attached upstream
            # (math_review.md §13). The "source" tag wins on conflict so
            # downstream code can tell the model-implied target apart.
            merged_fut_meta = {**dict(snapshot.futures.metadata), "source": "model_implied"}
            model_futures = BatchedFuturesTarget(
                ids=list(snapshot.futures.ids),
                prices=prices,
                asof_date=snapshot.futures.asof_date,
                delivery_dates=list(snapshot.futures.delivery_dates),
                basket_lengths=snapshot.futures.basket_lengths,
                conversion_factors_flat=snapshot.futures.conversion_factors_flat,
                deliverable_ids_flat=list(snapshot.futures.deliverable_ids_flat),
                metadata=merged_fut_meta,
            )

        merged_meta = {**dict(getattr(snapshot, "meta", {}) or {}), "source": "model_implied"}
        return MarketSnapshot(
            date=snapshot.date,
            yield_curve=model_yield_curve,
            short_rate=model_short_rate,
            bonds=None,
            bonds_metadata=None,
            futures=model_futures,
            meta=merged_meta,
        )

    # ------------------------------------------------------------------
    # Helpers for futures pricing
    # ------------------------------------------------------------------

    def _lsmc_consistency_loss(
        self,
        *,
        bond_values: Tensor,            # (n_paths, N_slots) — BondNet output (per 100 face)
        bond_features: Tensor,          # (N_slots, 8) — BondMetadataStore feature order
        per_slot_delivery_idx: Tensor,  # (N_slots,) long — grid index at delivery
        realisations: Tensor,           # (n_paths, steps[, 1]) — short-rate paths
        simulated_times: Tensor,        # (steps,) — year-fraction grid
    ) -> Optional[Tensor]:
        """
        Longstaff-Schwartz consistency loss between BondNet and the SDE.

        For each deliverable slot, reconstruct an approximate cashflow
        schedule from the bond features (coupon every ``1/freq`` years
        starting at ``years_to_next_coupon``, principal at
        ``years_to_maturity``) and compute the **pathwise** present value at
        delivery using the model's own simulated short rate:

            PV_p = sum_j c_j * exp(-(I_p(t_j) - I_p(T_dlv))),   I = cumsum(r)*dt

        ``E[PV | z_T]`` is the model-consistent bond price, so regressing
        BondNet(z_T, b) onto PV (MSE over paths and slots) drives BondNet
        toward the conditional expectation of the model's own discounting —
        the classical LSMC trick, with NO nested simulation. Prices are
        normalised by 100 so the loss is on the same dimensionless O(1e-4)
        scale as the relative futures loss and the yield MSE.

        Slots whose maturity falls beyond the simulation horizon are
        excluded (their cashflows cannot be discounted on this grid).
        Returns ``None`` when no slot is usable.
        """
        realisations = self._to_2d_paths(realisations)
        n_steps = realisations.size(1)
        dt = 1.0 / float(self.steps_per_year)

        # Cumulative integral of r on the same left-Riemann convention as
        # price_zcb: I[:, k] = sum_{s<=k} r_s * dt, value *at* grid idx k+1.
        cum_int = torch.cumsum(realisations, dim=1) * dt          # (P, S)

        # Feature columns (BondMetadataStore order).
        ytm  = bond_features[:, 0]          # years to maturity
        ytnc = bond_features[:, 1]          # years to next coupon
        cpn  = bond_features[:, 3]          # coupon rate (decimal)
        freq = bond_features[:, 4].clamp_min(1.0)
        ncp  = bond_features[:, 5]          # remaining coupon count

        horizon = float(simulated_times[-1].item())
        slot_ok = ytm <= horizon + 1e-9                            # (N,)
        if not bool(slot_ok.any()):
            return None

        # Coupon time grid per slot: t_k = ytnc + k/freq, k = 0..K-1.
        K = int(min(max(float(ncp.max().item()), 1.0), 80.0))
        ks = torch.arange(K, device=bond_features.device, dtype=bond_features.dtype)
        t_cpn = ytnc.unsqueeze(1) + ks.unsqueeze(0) / freq.unsqueeze(1)   # (N, K)

        # Delivery time per slot (from the grid, so it matches the latent
        # state BondNet was evaluated at).
        t_dlv = simulated_times.index_select(0, per_slot_delivery_idx)    # (N,)

        valid = (ks.unsqueeze(0) < ncp.unsqueeze(1))                       # within coupon count
        valid &= t_cpn > t_dlv.unsqueeze(1) + 1e-9                         # strictly after delivery
        valid &= t_cpn <= horizon + 1e-9                                   # on the grid
        valid &= slot_ok.unsqueeze(1)

        # Map times -> grid indices (same round-and-clamp as price_zcb).
        def _idx(t: Tensor) -> Tensor:
            return torch.round(t * float(self.steps_per_year)).long().clamp(1, n_steps)

        idx_cpn = _idx(t_cpn)                                              # (N, K)
        idx_mat = _idx(ytm)                                                # (N,)
        idx_dlv = per_slot_delivery_idx.clamp(min=1)                       # (N,)

        # Pathwise integrals at the relevant indices: I(t) = cum_int[:, idx-1].
        N, Kk = idx_cpn.shape
        I_cpn = cum_int.index_select(1, (idx_cpn - 1).reshape(-1)).reshape(-1, N, Kk)  # (P, N, K)
        I_mat = cum_int.index_select(1, idx_mat - 1)                       # (P, N)
        I_dlv = cum_int.index_select(1, idx_dlv - 1)                       # (P, N)

        # Pathwise discount factors from delivery to each cashflow.
        D_cpn = torch.exp(-(I_cpn - I_dlv.unsqueeze(-1)))                  # (P, N, K)
        D_mat = torch.exp(-(I_mat - I_dlv))                                # (P, N)

        cpn_amt = (100.0 * cpn / freq).unsqueeze(0).unsqueeze(-1)          # (1, N, 1)
        pv = (valid.unsqueeze(0) * cpn_amt * D_cpn).sum(dim=-1)            # (P, N)
        pv = pv + 100.0 * D_mat * slot_ok.unsqueeze(0)                     # principal

        diff = (bond_values - pv) / 100.0                                  # dimensionless
        mask = slot_ok.unsqueeze(0).expand_as(diff)
        return diff.pow(2)[mask].mean()


    @staticmethod
    def _extract_latent_idx_at_delivery(
        simulated_times: Tensor,
        target_times: Tensor,
    ) -> Tensor:
        """
        Map each target year-fraction to a simulation grid index, with no
        look-ahead (`searchsorted(..., right=True) - 1`, clamped to >= 0).
        """
        assert simulated_times.dim() == 1
        assert target_times.dim() == 1

        idx = torch.searchsorted(simulated_times, target_times, right=True) - 1
        idx = torch.clamp(idx, min=0)
        return idx

    @staticmethod
    def _segmented_min(values: Tensor, basket_lengths: Tensor) -> Tensor:
        """
        Path-wise segmented min over the deliverable dim.

        Parameters
        ----------
        values : Tensor
            Shape (n_paths, N_dlv_flat).
        basket_lengths : Tensor
            Per-future basket sizes, shape (n_futures,), dtype long.
            Must satisfy `basket_lengths.sum() == N_dlv_flat`.

        Returns
        -------
        Tensor
            Per-future minima, shape (n_paths, n_futures).
        """
        assert values.dim() == 2
        assert basket_lengths.dim() == 1
        n_futures = int(basket_lengths.numel())

        # Build segment ids: for basket_lengths = [3, 2, 4] we get
        # [0,0,0,1,1,2,2,2,2].
        seg_ids = torch.repeat_interleave(
            torch.arange(n_futures, device=values.device, dtype=torch.long),
            basket_lengths.to(device=values.device, dtype=torch.long),
        )

        n_paths = values.size(0)
        # Initialise the scatter target with +inf so min() correctly reduces.
        out = torch.full(
            (n_paths, n_futures),
            float("inf"),
            device=values.device,
            dtype=values.dtype,
        )
        seg_expanded = seg_ids.unsqueeze(0).expand(n_paths, -1)
        out.scatter_reduce_(dim=1, index=seg_expanded, src=values, reduce="amin", include_self=False)
        return out

    @staticmethod
    def _ctd_selection_freq(values: Tensor, basket_lengths: Tensor) -> Tensor:
        """
        Per-basket-slot frequency of being selected as the CTD across paths.

        Parameters
        ----------
        values : Tensor
            Per-slot adjusted prices ``B_k / cf_k``, shape (n_paths, N_dlv_flat).
        basket_lengths : Tensor
            Per-future basket sizes, shape (n_futures,).

        Returns
        -------
        Tensor
            Float tensor of shape (N_dlv_flat,) whose entries sum to
            ``n_futures`` (each basket contributes 1 unit of probability mass
            distributed across its slots).
        """
        n_paths = values.size(0)
        n_total = values.size(1)
        bl = basket_lengths.to(device=values.device, dtype=torch.long)
        n_futures = int(bl.numel())

        # Slot offsets into the flat array per future.
        offsets = torch.zeros(n_futures + 1, device=values.device, dtype=torch.long)
        offsets[1:] = torch.cumsum(bl, dim=0)

        freq = torch.zeros(n_total, device=values.device, dtype=values.dtype)
        for j in range(n_futures):
            s, e = int(offsets[j].item()), int(offsets[j + 1].item())
            sub = values[:, s:e]                          # (n_paths, basket_j)
            winners = sub.argmin(dim=1)                   # (n_paths,)
            counts = torch.bincount(winners, minlength=(e - s)).to(values.dtype)
            freq[s:e] = counts / float(max(n_paths, 1))
        return freq
