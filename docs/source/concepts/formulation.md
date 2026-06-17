# Mathematical formulation

This page is the precise specification of the model the code implements:
the state-space construction, the risk-neutral pricing maps, the training
objective, and the $\mathbb{P}/\mathbb{Q}$ extension.

## 1. State space

Let $\{\mathbf{c}_t\}$ be the observed daily yield curve history
($\mathbf{c}_t \in \mathbb{R}^{M}$, the $M$ SVENY pillars) and $\{x_t\}$
the observed short rate. An encoder $E_\phi$ (a recurrent network) maps a
lookback window to a latent initial state

$$
z_0 \;=\; E_\phi\!\big(\mathbf{c}_{t-L:t},\, x_{t-L:t}\big) \in \mathbb{R}^{d}.
$$

The latent state evolves under a **Neural SDE** with learnable drift
$\mu_\theta$ and diffusion $\sigma_\theta$,

$$
\mathrm{d}z_s \;=\; \mu_\theta(s, z_s)\,\mathrm{d}s
            \;+\; \sigma_\theta(s, z_s)\,\mathrm{d}W_s^{\mathbb{Q}},
\qquad s\in[0,T_{\max}],
$$

simulated by Euler–Maruyama on a grid of step $\Delta = 1/\texttt{steps\_per\_year}$.
Two drift families are supported:

$$
\textbf{(simple)}\;\; \mu_\theta = f_\theta(s,z), \qquad
\textbf{(OU)}\;\; \mu_\theta = \kappa_\theta(s,z)\,\big(\vartheta_\theta(s,z)-z\big),
$$

with $\kappa_\theta\ge 0$ (softplus) capped at $\kappa_{\max}$.

A decoder $D_\psi:\mathbb{R}^d\to\mathbb{R}$ produces the short rate, with
an **additive anchor** that pins the front of the curve to the observed
short rate $x_t$:

$$
r_s \;=\; D_\psi(z_s) \;+\; \big(x_t - D_\psi(z_0)\big),
\qquad\text{so}\quad r_0 = x_t .
$$

## 2. Risk-neutral pricing

### Zero-coupon bonds and yields

$$
P(0,\tau) \;=\; \mathbb{E}^{\mathbb{Q}}\!\Big[\exp\big(-\!\textstyle\int_0^\tau r_s\,\mathrm{d}s\big)\Big],
\qquad
y(\tau) \;=\; -\frac{\log P(0,\tau)}{\tau}.
$$

The expectation is a Monte-Carlo average over $N$ paths; the code uses a
numerically stable log-sum-exp,

$$
\log P(0,\tau)\;=\;\operatorname*{logsumexp}_{n}\big(-I^{(n)}_\tau\big)-\log N,
\qquad I^{(n)}_\tau=\Delta\!\!\sum_{s\le\tau} r^{(n)}_s .
$$

**Convexity.** With $r_s = r_0 + (D_\psi(z_s)-D_\psi(z_0))$ and $z_0$ shared
across paths, a cumulant expansion gives

$$
y(\tau)\;\approx\;
\underbrace{(x_t-D_0)}_{\text{level}}
+\underbrace{\frac{1}{\tau}\!\int_0^\tau\!\mathbb{E}[D_\psi(z_s)]\,\mathrm{d}s}_{\text{expectations}}
-\underbrace{\frac{\operatorname{Var}\!\big(\int_0^\tau D_\psi(z_s)\,\mathrm{d}s\big)}{2\tau}}_{\text{convexity}} .
$$

The convexity term scales with the *implied short-rate volatility*
$\sigma_r \approx \lVert\nabla D_\psi\rVert\,\sigma_z$; keeping it at the
~bp level requires $\sigma_r\sim 1\%/\text{yr}$, which is why the diffusion
magnitude must be controlled (§4).

### Treasury futures (cheapest-to-deliver)

For a contract with delivery $T_f$ and deliverable basket
$\mathcal{B}$ with conversion factors $\{\mathrm{CF}_i\}$,

$$
F \;=\; \mathbb{E}^{\mathbb{Q}}\!\Big[\min_{i\in\mathcal{B}}\frac{B_i(z_{T_f})}{\mathrm{CF}_i}\Big],
$$

where $B_i$ is the (forward) deliverable bond price. The code learns $B_i$
with a **BondNet** $B_\chi(z, b_i)$ taking the latent state and bond
features $b_i$ — avoiding a nested simulation.

## 3. Training objective

$$
\mathcal{L}
=\lambda_y \mathcal{L}_{\text{yield}}
+\lambda_f \mathcal{L}_{\text{fut}}
+\lambda_c \mathcal{L}_{\text{cons}}
+\lambda_\sigma \mathcal{L}_{\text{vol}}
+\lambda_{\mathbb{P}} \mathcal{L}_{\mathbb{P}/\mathbb{Q}} .
$$

- **Yield** — absolute MSE on decimals,
  $\mathcal{L}_{\text{yield}}=\frac1M\sum_\tau (y(\tau)-y^{\star}(\tau))^2$.
- **Futures** — *relative* error (dimensionless, so $\lambda$'s are
  comparable), $\mathcal{L}_{\text{fut}}=\frac1{|\mathcal{F}|}\sum_k\big((F_k-F^\star_k)/F^\star_k\big)^2$.
- **BondNet ↔ SDE consistency (LSMC).** Regress BondNet onto the model's
  own pathwise-discounted cashflows, computed on the *same* paths (no
  nested simulation). With $\hat B_i^{(n)} = \sum_j c_{ij}\,e^{-(I^{(n)}_{t_{ij}}-I^{(n)}_{T_f})}$,

  $$
  \mathcal{L}_{\text{cons}}=\frac1{100^2}\,\mathbb{E}_n\!\Big[\big(B_\chi(z^{(n)}_{T_f},b_i)-\operatorname{sg}[\hat B_i^{(n)}]\big)^2\Big],
  $$

  where $\operatorname{sg}[\cdot]$ is stop-gradient: the gradient reaches
  BondNet only, never the SDE (a non-detached version bends the curve).
- **Volatility anchor.** The data barely identifies $\sigma$, so pin the
  $1$-year cross-path std to the historically measured rate vol
  $\sigma^\star$ ($\approx 1\%$):
  $\mathcal{L}_{\text{vol}}=(\operatorname{std}_n[r^{(n)}_{1\mathrm{y}}]-\sigma^\star)^2$.
  Legitimate because, by Girsanov, $\sigma$ is **measure-invariant**.

## 4. The $\mathbb{P}/\mathbb{Q}$ extension

The drift $\mu_\theta$ learned above is the **risk-neutral** drift
$\mu^{\mathbb{Q}}$. By Girsanov, with market price of risk
$\lambda\in\mathbb{R}^{d}$ and $\mathrm{d}W^{\mathbb{Q}}=\mathrm{d}W^{\mathbb{P}}+\lambda\,\mathrm{d}s$,
the **physical** drift is

$$
\mu^{\mathbb{P}}(s,z)\;=\;\mu^{\mathbb{Q}}(s,z)\;+\;\sigma_\theta(s,z)\,\lambda .
$$

The volatility is the same under both measures. The $\mathbb{P}/\mathbb{Q}$
consistency term matches a one-step physical forecast of the short rate to
the realised future rate $h$ ahead,

$$
\widehat r^{\,\mathbb{P}}_{t+h}
= x_t + D_\psi\!\big(z_0+\mu^{\mathbb{P}}(0,z_0)\,h\big)-D_\psi(z_0),
\qquad
\mathcal{L}_{\mathbb{P}/\mathbb{Q}}=\big(\widehat r^{\,\mathbb{P}}_{t+h}-x_{t+h}\big)^2 .
$$

This is the only place a physical-measure quantity enters; it identifies
$\lambda$ (the **term premium**: forward rate $=$ expected future rate
$+\,\sigma\lambda$). With $\lambda_{\mathbb{P}}=0$ the model is a pure
$\mathbb{Q}$-calibration and $\lambda\equiv 0$.

## 5. Known identifiability issues (and the corresponding terms)

| Quantity | Identified by the data? | Mechanism in the code |
|---|---|---|
| curve level / slope | yes (yields) | $\mathcal{L}_{\text{yield}}$ |
| short-rate volatility | **no** (yields are vol-insensitive) | $\mathcal{L}_{\text{vol}}$ anchor |
| deliverable bond prices | only via CTD min | $\mathcal{L}_{\text{cons}}$ (LSMC) |
| term premium / $\lambda$ | **no** under $\mathbb{Q}$ alone | $\mathcal{L}_{\mathbb{P}/\mathbb{Q}}$ |

These are *structural* — the calibration data does not pin volatility or
the market price of risk, so the corresponding terms are anchors/priors,
not fits. See the ``report/`` folder for the empirical diagnostics.
