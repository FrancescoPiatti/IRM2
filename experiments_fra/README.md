# experiments_fra

End-to-end training scripts. Each one is **self-contained** — no
hidden state, no shared CLI — and runnable directly from the repo root
via ``python -m experiments_fra.<subfolder>.<script_name>``.

Every script uses ``nsde.solver = "custom_euler"`` by default
(7–15× faster than `torchsde` for the Euler scheme used throughout the
project — see `optimization_report.md`).

## Layout

| Subfolder | What lives here |
|---|---|
| ``one_model_experiments_on_YC/``               | YC-only baselines — 6 (encoder, NSDE) combinations. |
| ``one_model_experiments_on_YC_and_futures/``   | Joint YC + Treasury futures + BondNet. 3 configurations. |
| ``gridsearch_experiments_on_YC/``              | Tier-1 Optuna sweep on YC-only (~12 trials). |
| ``gridsearch_experiments_on_YC_and_futures/``  | Tier-1 Optuna sweep on the joint setup (~18 trials). |

## Naming conventions

- ``YC_*`` — yield-curve only
- ``YCFut_*`` — yield curves + Treasury futures (BondNet attached)
- ``SimpleEnc`` / ``HierEnc`` — encoder topology
- ``SimpleNSDE`` / ``OUNSDE`` / ``VasicekNSDE`` — drift / volatility family
  - ``SimpleNSDE``: neural drift, neural diffusion.
  - ``OUNSDE``: mean-reverting OU drift, neural diffusion.
  - ``VasicekNSDE``: OU drift, **constant** diffusion (neural Vasicek).

## How to choose configurations

For the meaning of every hyperparameter that appears in these scripts —
and which ones are worth gridding vs. setting and forgetting — see
``hyperparameters.md`` at the repo root.
