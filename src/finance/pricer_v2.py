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

# Single year-fraction convention used across the whole stack
# (loader / pricer / trainer.dt = 1/252). User spec: 252 days/year is the
# divisor everywhere; we count calendar days between dates and divide by
# 252, so 252 acts purely as the normaliser, not as a working-day filter.
DAYS_PER_YEAR = 252.0


def to_year_fraction(
    target_dates: Union[Date, List[Date]],
    asof_date: Date,
    *,
    business_days_per_year: float = 252.0,    # kept for back-compat only
) -> Tensor:
    """
    Convert dates to a 1D tensor of year-fractions relative to `asof_date`.

    Year-fractions use the **single 252 days/year convention** shared by
    the loader, pricer, and trainer (``trainer.dt = 1/252``). The
    ``business_days_per_year`` argument is accepted for back-compat but
    is **not used** — the divisor is hard-wired to ``DAYS_PER_YEAR =
    252`` so the convention is identical everywhere.

    Vectorised via numpy ``datetime64`` arithmetic.

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
        1D float32 tensor of year-fractions (252-day basis) with the same
        length as `target_dates`.
    """
    if not isinstance(target_dates, (list, tuple)):
        target_dates = [target_dates]

    asof = np.datetime64(pd.Timestamp(asof_date).normalize(), "D")
    targets = pd.to_datetime(list(target_dates)).normalize().values.astype("datetime64[D]")
    day_deltas = (targets - asof).astype("int64").astype(np.float32)
    return torch.from_numpy(day_deltas / float(DAYS_PER_YEAR))


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
