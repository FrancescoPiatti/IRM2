# Mathematical formulation

The precise specification of the model: the construction of each block
(encoder, SDE, decoder, BondNet), the risk-neutral pricing maps, the
$\mathbb{Q}$-objective, a catalogue of the problems found during
development, and the optional $\mathbb{P}/\mathbb{Q}$ extension. The
companion `report/` folder (PDF/HTML) carries the same content with the
full diagnostic narrative.

## 1. Model construction

The model is a composition
$\text{history}\xrightarrow{E_\phi}z_0\xrightarrow{\text{SDE}_\theta}(z_s)\xrightarrow{D_\psi}(r_s)\to\text{prices}$,
plus an auxiliary network $B_\chi$ for deliverable bonds.

### 1.1 Encoder

With lookback $L$ and stride $\nu$, the input is the window of $M$ SVENY
pillars stacked with the overnight rate $x$,
$X_t=[\mathbf c_{t-\nu(L-1)},\dots,\mathbf c_t\,;\,x_{t-\nu(L-1)},\dots,x_t]\in\mathbb{R}^{L\times(M+1)}$.
A fixed preprocessing rescales to percent units, $\pi(X)=100X$ (raw decimal
yields move by $\sim5\times10^{-4}$/day — too weak a signal; problem **E3**).
The latent initial state is

$$
z_0=\mathcal N\!\Big(W\,\mathrm{LSTM}_\phi^{\leftrightarrow}\big(\pi(X_t)\big)_L+b\Big)\in\mathbb{R}^d,
$$

a bidirectional LSTM (last step) + affine readout + output normalisation
$\mathcal N\in\{\text{LayerNorm},\text{RMSNorm}\}$, which fixes $\lVert z_0\rVert$
so the *direction* of $z_0$ carries the day. $z_0$ is the **only**
day-specific quantity in the model (problem **A1**).

### 1.2 Latent dynamics (Neural SDE)

$$
\mathrm{d}z_s=\mu_\theta(s,z_s)\,\mathrm{d}s+\sigma_\theta(s,z_s)\,\mathrm{d}W_s^{\mathbb{Q}},
\qquad\text{diagonal noise,}
$$

time-inhomogeneous (time is a network input). Two drift families:

$$
\textbf{(simple)}\ \mu_\theta=f_\theta(s,z),\qquad
\textbf{(OU)}\ \mu_\theta=\kappa_\theta\odot(\vartheta_\theta-z),\ \ 0\le\kappa_\theta\le\kappa_{\max}.
$$

The diffusion factors a magnitude scale from a shape network,
$\sigma_\theta=\eta\,\zeta_\theta$, $\zeta_\theta\ge0$; $\eta$ sets the
implied vol *by construction* (problem **M2**). Integration is
Euler–Maruyama, step $\Delta=1/\texttt{spy}$, $N$ paths sharing $z_0$.

### 1.3 Decoder and short-rate anchor

$$
r_s=D_\psi(z_s)+\big(x_t-D_\psi(z_0)\big)
\;\Longrightarrow\;
r_s=x_t+\big(D_\psi(z_s)-D_\psi(z_0)\big),\quad r_0=x_t.
$$

A linear $D_\psi=w^\top z+b$ makes $r$ a scalar projection of $z$ — one factor
regardless of $d$ (problem **A2**); a multilayer decoder restores curvature.

### 1.4 BondNet (deliverable bonds)

Futures pricing needs each deliverable's forward price as a function of
$z_{T_f}$ — a conditional expectation $\mathbb{E}^{\mathbb{Q}}[\cdot\mid z_{T_f}]$.
A network learns it from static features $b_i\in\mathbb{R}^8$ (maturity,
coupon timing, coupon, frequency, accruals):

$$
B_\chi(z,b_i)=\rho\big(g_{\text{fus}}[\,g_z(z)\,;\,g_b(b_i)\,]\big),\quad \rho=\text{softplus},
$$

a two-branch late fusion returning a dirty price per $100$ face (output bias
initialised near par; problem **E1**). $B_\chi$ is a priori *unconstrained* —
not tied to the simulated discount factors (defect **A3**).

## 2. Risk-neutral pricing

**Yields.** With $I^{(n)}_\tau=\Delta\sum_{s_k\le\tau}r^{(n)}_{s_k}$,

$$
P(0,\tau)=\mathbb{E}^{\mathbb{Q}}[e^{-I_\tau}]
=\exp\!\big(\operatorname*{logsumexp}_n(-I^{(n)}_\tau)-\log N\big),
\qquad y(\tau)=-\tfrac1\tau\log P(0,\tau).
$$

**Convexity.** $y(\tau)\approx(x_t-D_0)+\tfrac1\tau\int_0^\tau\mathbb{E}[D_\psi(z_s)]\mathrm{d}s-\tfrac1{2\tau}\mathrm{Var}(\int_0^\tau D_\psi(z_s)\mathrm{d}s)$;
the last term scales with $\sigma_r^2$, $\sigma_r\approx\lVert\nabla D_\psi\rVert\sigma_z$
(problems **M1, M2**).

**Futures (CTD).** With delivery index $\iota_f=\max\{k:s_k\le T_f\}$ and
conversion factors $\{\mathrm{CF}_i\}$,

$$
F=\mathbb{E}^{\mathbb{Q}}\!\Big[\min_{i\in\mathcal B}\tfrac{B_\chi(z_{\iota_f},b_i)}{\mathrm{CF}_i}\Big],
$$

the inner $\min$ being the cheapest-to-deliver option (a segmented reduction
across active baskets).

## 3. Risk-neutral objective

$$
\mathcal L_{\mathbb{Q}}=\lambda_y\mathcal L_{\text{yield}}+\lambda_f\mathcal L_{\text{fut}}
+\lambda_c\mathcal L_{\text{cons}}+\lambda_\sigma\mathcal L_{\text{vol}}.
$$

Yield = absolute MSE; futures = relative error (problem **E5**); the
**LSMC consistency** term regresses BondNet onto the model's own
pathwise-discounted cashflows with a stop-gradient (gradient to BondNet only;
**A3**); the **vol anchor** pins the $1$y cross-path std to $\sigma^\star\approx1\%$
(**M1**).

## 4. Problems (catalogue)

| Type | Problem | Resolution |
|---|---|---|
| **M1** math | volatility unidentified (yields vol-insensitive) | vol anchor (Girsanov: $\sigma$ measure-invariant) |
| **M2** math | convexity blow-up at large $\sigma_r$ | magnitude scale $\eta$ |
| **M3** math | OU erases $z_0$: $y\propto g(\kappa\tau)\to$ flat | cap $\kappa\le\kappa_{\max}$ |
| **A1** arch | single-$z_0$ bottleneck (global dynamics) | *open* — day-condition the drift |
| **A2** arch | linear decoder ⇒ one-factor | multilayer decoder (grid axis) |
| **A3** arch | BondNet decoupled from rates | LSMC consistency (detached) |
| **E1** eng | gradient explosion / NaN over long unroll | short $\Delta$, init, scales, clip, guard |
| **E2** eng | dropout in SDE ⇒ train/eval mismatch | removed |
| **E3** eng | encoder input too small | percent preprocessing |
| **E4** eng | dead short-rate target | weight 0 |
| **E5** eng | absolute futures MSE dominates | relative error |
| **E6** eng | selection on aggregate loss | per-maturity bp RMSE on a fold |
| **E7** eng | percent/decimal, 365-vs-252, NaN CF | loader fixes |

## 5. The $\mathbb{P}/\mathbb{Q}$ extension (optional)

Off by default; logically separate from the $\mathbb{Q}$-calibration. The
learned drift is $\mu^{\mathbb{Q}}$; with market price of risk $\lambda$ and
$\mathrm{d}W^{\mathbb{Q}}=\mathrm{d}W^{\mathbb{P}}+\lambda\,\mathrm{d}s$,
Girsanov gives $\mu^{\mathbb{P}}=\mu^{\mathbb{Q}}+\sigma_\theta\lambda$. A
one-step physical forecast is matched to the realised future rate $h$ ahead,
$\mathcal L_{\mathbb{P}/\mathbb{Q}}=(\hat r^{\mathbb{P}}_{t+h}-x_{t+h})^2$ with
$\hat r^{\mathbb{P}}_{t+h}=x_t+D_\psi(z_0+\mu^{\mathbb{P}}(0,z_0)h)-D_\psi(z_0)$.
This identifies the term premium $\lambda$; with $\lambda_{\mathbb{P}}=0$ the
model is a pure $\mathbb{Q}$-calibration. The grid runs both.
