# Neural Short-Rate Model for Treasury Curves and Futures  
## Mathematical outline and implementation algorithm

## 1. Goal of the project

The goal is to build a **risk-neutral neural short-rate model** for the Treasury market that:

1. uses **historical yield curves** as the canonical source of information,
2. encodes them into a latent state,
3. evolves this latent state under a **Neural SDE**,
4. decodes the latent state into a short rate,
5. uses a learned **BondNet** to approximate coupon-bond values at futures delivery,
6. prices Treasury futures through the **cheapest-to-deliver (CTD)** mechanism,
7. is trained by combining:
   - a **yield-curve fitting loss**,
   - a **futures pricing loss**.

At the current stage of the project:

- the model is under the **risk-neutral measure** $\mathbb{Q}$,
- we are **not** modelling under the historical measure $\mathbb{P}$,
- there is **no repo curve / implied repo** in the current formulation,
- there is **no options component** in this document.

So the project should be interpreted primarily as a **pricing / calibration** framework, not as a real-world forecasting model.

---

## 2. Risk-neutral short-rate framework

Let $r_t$ denote the instantaneous short rate. The bank account numeraire satisfies

$$
dB_t = r_t B_t \, dt,
\qquad
B_t = B_0 \exp\left(\int_0^t r_s\,ds\right).
$$

Under the risk-neutral measure $\mathbb{Q}$, the discount factor from $t$ to $T$ is

$$
D(t,T) = \exp\left(-\int_t^T r_s\,ds\right).
$$

The zero-coupon bond price is therefore

$$
P(t,T)
=
\mathbb{E}^{\mathbb{Q}}_t\!\left[D(t,T)\right]
=
\mathbb{E}^{\mathbb{Q}}_t
\left[
\exp\left(-\int_t^T r_s\,ds\right)
\right].
$$

If yields $y(t,T)$ are continuously compounded, then

$$
P(t,T) = e^{-y(t,T)(T-t)}.
$$

So if a market zero-coupon yield curve is observed at time $t$, it provides the corresponding cross-section of zero-coupon prices.

---

## 3. Observed market data

The project currently uses the following market objects.

### 3.1 Yield curves

Yield curves are the canonical object of the project.

For each business date $t$, we observe a vector of yields at fixed maturities:

$$
\mathbf{y}_t =
\big(
y(t,t+\tau_1),\dots,y(t,t+\tau_M)
\big)\in\mathbb{R}^M.
$$

These are the inputs to the encoder and define the canonical calendar of the project.

### 3.2 Futures prices

For a set of Treasury futures contracts, we observe market prices

$$
F_t^{\mathrm{mkt}}.
$$

Each futures contract comes with:

- a **delivery date** $T_f$,
- a **delivery basket** of coupon bonds,
- a **conversion factor** $cf_k$ for each bond $k$ in the basket.

### 3.3 Bond metadata

For each deliverable bond we have static metadata such as:

- maturity date,
- coupon rate,
- coupon frequency,
- issue date / coupon dates if available.

At runtime, this metadata is transformed into a fixed-size feature vector to be passed to BondNet.

---

## 4. Latent-state Neural SDE model

### 4.1 Encoder

The model first maps a history of yield curves into a latent state:

$$
z_t = \Psi\big(\mathbf{y}_{t-\ell:t}\big),
\qquad
z_t \in \mathbb{R}^{d_z}.
$$

Here:

- $\Psi$ is a learned encoder,
- $\ell$ is the lookback window,
- $d_z$ is the latent dimension.

The purpose of the encoder is to summarize recent curve history into a latent state that can be evolved forward under $\mathbb{Q}$.

### 4.2 Latent Neural SDE

The latent state evolves under a Neural SDE of the form

$$
dz_s = \mu(s,z_s)\,ds + \sigma(s,z_s)\,dW_s^{\mathbb{Q}},
\qquad s \ge t.
$$

Here:

- $\mu$ is a drift network,
- $\sigma$ is a diffusion network,
- $W^{\mathbb{Q}}$ is Brownian motion under $\mathbb{Q}$.

### 4.3 Short-rate decoder

The latent state is mapped into the short rate through a decoder

$$
r_s = \rho(z_s).
$$

A simple and natural choice is a linear decoder

$$
r_s = \beta^\top z_s,
$$

but more general decoders are possible.

---

## 5. Yield-curve pricing from the latent short rate

Given a path of the short rate, we define the pathwise discount factor

$$
D(t,T) = \exp\left(-\int_t^T r_u\,du\right).
$$

The model-implied zero-coupon price is

$$
\widehat{P}(t,T)
=
\mathbb{E}^{\mathbb{Q}}_t[D(t,T)].
$$

From this, the model-implied yield is

$$
\widehat{y}(t,T)
=
-\frac{1}{T-t}\log \widehat{P}(t,T).
$$

Collecting this at the observed maturities $(\tau_1,\dots,\tau_M)$, we obtain the model-implied yield vector

$$
\widehat{\mathbf{y}}_t =
\big(
\widehat{y}(t,t+\tau_1),\dots,\widehat{y}(t,t+\tau_M)
\big).
$$

The basic yield loss is then

$$
\mathcal{L}_{\mathrm{yield}}
=
\|\widehat{\mathbf{y}}_t - \mathbf{y}_t\|^2,
$$

or an equivalent weighted MSE.

---

## 6. Coupon bonds and the need for BondNet

For a coupon bond $k$, suppose the remaining cashflows are $(\tau_j^{(k)}, CF_j^{(k)})$. Then the dirty price at time $s$ is

$$
B_k^{\mathrm{dirty}}(s)
=
\mathbb{E}^{\mathbb{Q}}_s
\left[
\sum_j CF_j^{(k)} D(s,\tau_j^{(k)})
\right].
$$

In the futures setting, the relevant object is the bond value at the **delivery date** $T_f$:

$$
B_k(T_f)
=
\mathbb{E}^{\mathbb{Q}}_{T_f}
\left[
\sum_j CF_j^{(k)} D(T_f,\tau_j^{(k)})
\right].
$$

From time $t<T_f$, this is a conditional expectation at a future time $T_f$.  
If one tried to compute it exactly by simulation, this would naturally suggest **nested Monte Carlo**.

To avoid nested Monte Carlo, we introduce a learned approximation.

---

## 7. BondNet

### 7.1 Purpose

BondNet approximates the coupon-bond pricing functional at the future delivery date:

$$
\widehat{B}_k(T_f)
=
\Pi\big(z_{T_f}, \phi_k(T_f)\big).
$$

Here:

- $z_{T_f}$ is the latent state at delivery,
- $\phi_k(T_f)$ is a fixed-size feature vector describing bond $k$ at delivery,
- $\Pi$ is the BondNet.

So BondNet is a learned approximation to the map

$$
(z_{T_f}, \text{bond metadata at }T_f)
\mapsto
\text{dirty bond value at }T_f.
$$

### 7.2 Bond features

The project currently uses an approximate fixed-size bond feature vector, computed from maturity and coupon information.

A typical feature vector is:

1. years to maturity,
2. years to next coupon,
3. years from last coupon,
4. coupon rate,
5. coupon frequency,
6. remaining coupon count,
7. accrued fraction,
8. accrued interest per 100.

The current implementation intentionally uses a **fast approximation** and does **not** reconstruct the exact coupon schedule by month arithmetic.

### 7.3 BondNet architectures currently considered

Two main candidate designs:

#### (a) SimpleBondNet
A two-branch MLP with late fusion:

- one branch processes the latent state $z_{T_f}$,
- one branch processes bond features $\phi_k(T_f)$,
- the two hidden representations are concatenated,
- a fusion MLP outputs the bond price.

#### (b) FiLMBondNet
A feature-conditioned latent trunk:

- $z_{T_f}$ is processed through a latent trunk,
- bond features produce FiLM modulation parameters $(\gamma,\beta)$,
- the hidden state is modulated as

$$
h = (1+\gamma)\odot h + \beta,
$$

- a head MLP outputs the bond price.

The simplest viable implementation is usually the **two-branch late-fusion MLP**.

---

## 8. Treasury futures pricing under the current no-repo setup

For a futures contract with delivery basket $k=1,\dots,K$ and conversion factors $cf_k$, define the delivery-adjusted value of bond $k$ at delivery as

$$
V_k(T_f) = \frac{B_k(T_f)}{cf_k}.
$$

The cheapest-to-deliver (CTD) value is

$$
V_{\mathrm{CTD}}(T_f)
=
\min_{k=1,\dots,K} \frac{B_k(T_f)}{cf_k}.
$$

In the current simplified formulation (no repo), the model-implied futures price is taken as

$$
\widehat{F}_t
=
\mathbb{E}^{\mathbb{Q}}_t\!\left[V_{\mathrm{CTD}}(T_f)\right].
$$

Operationally, this is estimated by Monte Carlo:

1. simulate paths to $T_f$,
2. compute $z_{T_f}^{(i)}$ for each path $i$,
3. evaluate BondNet for each path and each deliverable bond,
4. divide by conversion factors,
5. take the pathwise minimum across the basket,
6. average across paths.

So:

$$
\widehat{F}_t
\approx
\frac{1}{N_{\text{paths}}}
\sum_{i=1}^{N_{\text{paths}}}
\min_k
\frac{\widehat{B}_k^{(i)}(T_f)}{cf_k}.
$$

The futures loss is then

$$
\mathcal{L}_{\mathrm{fut}}
=
\|\widehat{F}_t - F_t^{\mathrm{mkt}}\|^2.
$$

---

## 9. Simulation time grid and delivery-date extraction

The latent SDE is simulated on a discrete time grid

$$
t_0=t,\ t_1=t+\Delta t,\ \dots,\ t_N=t+N\Delta t.
$$

A futures delivery date $T_f$ usually does not lie exactly on this grid.  
Therefore, the implementation maps each delivery date to the simulation grid using a **no-look-ahead rule**:

$$
\text{idx}(T_f)
=
\max\{i : t_i \le T_f\}.
$$

In code this corresponds to `searchsorted(..., right=True) - 1`.

This ensures that:
- if $T_f$ lies exactly on the grid, the exact grid point is used,
- otherwise the previous grid point is used,
- no look-ahead bias is introduced.

---

## 10. Current training objective

At the current stage, the model is trained in **one stage only**, using:

$$
\mathcal{L}
=
\lambda_y \mathcal{L}_{\mathrm{yield}}
+
\lambda_f \mathcal{L}_{\mathrm{fut}}.
$$

where:

- $\mathcal{L}_{\mathrm{yield}}$ fits the current yield curve,
- $\mathcal{L}_{\mathrm{fut}}$ fits observed futures prices.

There is currently:
- no options loss,
- no bond spot-price supervision,
- no repo term.

The recommendation is to start with a relatively small $\lambda_f$ so the futures component does not destabilize the curve fit.

---

# Part II — Algorithmic design

## 11. Main data objects

### 11.1 YieldCurveStore
Purpose:
- load the historical yield curves,
- define the canonical calendar,
- provide encoder histories,
- provide current curve targets.

### 11.2 BondMetadataStore
Purpose:
- load static bond metadata,
- preprocess it once,
- compute approximate bond features at a requested date,
- return these features vectorized for a delivery basket.

### 11.3 FuturesStore
Purpose:
- load futures prices,
- load delivery dates,
- load delivery baskets and conversion factors,
- provide:
  - `SingleFutureTarget`,
  - `BatchedFuturesTarget`.

### 11.4 MarketDataLoader
Purpose:
- combine yield curves, optional short-rate proxies, futures, bond metadata,
- provide training snapshots at each date.

---

## 12. Target objects

### 12.1 SingleFutureTarget
Represents one futures contract at one observation date.

Contains:
- futures id / ticker,
- observed price,
- as-of date,
- delivery date,
- deliverable bond ids,
- conversion factors.

### 12.2 BatchedFuturesTarget
Represents multiple futures at the same as-of date.

Uses a flattened ragged representation:
- prices of shape `(N_futures,)`,
- basket lengths `(N_futures,)`,
- flattened deliverable ids,
- flattened conversion factors.

This avoids padding while still allowing splitting by future.

---

## 13. Core forward-pass algorithm at one training date

Fix a training date $t$.

### Step 1 — build encoder input
Using the yield store, build the historical window:

$$
\mathbf{y}_{t-\ell:t}.
$$

### Step 2 — encode
Compute the latent state:

$$
z_t = \Psi(\mathbf{y}_{t-\ell:t}).
$$

### Step 3 — determine required horizon
Find the maximum horizon required among:

- the maturities needed for yield pricing,
- the delivery dates of the futures used at date $t$.

Simulate the Neural SDE once up to this maximum horizon.

### Step 4 — simulate latent paths
Simulate:

$$
z_{t_0}, z_{t_1}, \dots, z_{t_N}
$$

for all Monte Carlo paths.

### Step 5 — compute yield loss
Using the simulated short-rate paths:
- compute model-implied discount factors,
- compute model-implied yields,
- compare to the observed curve.

This gives $\mathcal{L}_{\mathrm{yield}}$.

### Step 6 — price futures
For each futures contract active at date $t$:

1. read the delivery date $T_f$,
2. map $T_f$ to the simulation grid using `searchsorted(..., right=True) - 1`,
3. extract the simulated latent states $z_{T_f}^{(i)}$,
4. get the deliverable basket and conversion factors,
5. compute bond features at $T_f$,
6. run BondNet to obtain pathwise bond values,
7. divide by conversion factors,
8. take the pathwise minimum across deliverables,
9. average across paths.

This gives $\widehat{F}_t$ and the corresponding futures loss contribution.

### Step 7 — total loss
Combine:

$$
\mathcal{L}
=
\lambda_y \mathcal{L}_{\mathrm{yield}}
+
\lambda_f \mathcal{L}_{\mathrm{fut}}.
$$

### Step 8 — backpropagation
Run backpropagation and update parameters.

---

## 14. Vectorization strategy

The recommended first implementation is:

- loop over **futures**,
- vectorize **within each future** over:
  - Monte Carlo paths,
  - deliverable bonds.

For one future:

- latent states at delivery:
  $$
  z_{T_f} \in \mathbb{R}^{N_{\text{paths}}\times d_z}
  $$

- bond features:
  $$
  \phi \in \mathbb{R}^{N_{\text{bonds}}\times d_b}
  $$

Broadcast to:

$$
(N_{\text{paths}}, N_{\text{bonds}}, d_z)
\quad\text{and}\quad
(N_{\text{paths}}, N_{\text{bonds}}, d_b),
$$

then BondNet outputs:

$$
(N_{\text{paths}}, N_{\text{bonds}}).
$$

This is the recommended first design because it keeps the code simple and robust.

---

## 15. Precision and autocast policy

At the current stage:
- use **float32** throughout,
- do **not** use float64.

If mixed precision is introduced later, the recommended policy is:

### Safe to autocast
- encoder forward,
- drift network forward,
- diffusion network forward,
- BondNet forward.

### Keep in float32
- solver state updates,
- path accumulation,
- discount factors,
- CTD logic,
- conversion-factor division,
- Monte Carlo averages,
- losses.

So the financial numerics remain in float32 even if neural-network components use autocast.

---

## 16. What is intentionally approximate at this stage

The current version deliberately simplifies several aspects.

### Included approximations
- no repo,
- no implied repo,
- no options,
- approximate bond feature construction,
- no exact coupon-schedule arithmetic,
- no bond spot-price supervision,
- one-stage training only.

These approximations are acceptable because the main goal is to get a coherent end-to-end pipeline working first.

---

## 17. Recommended implementation priorities

If Claude is meant to finish the project, the order of work should be:

1. ensure the data layer is consistent:
   - yield curves,
   - futures targets,
   - bond metadata,
2. finalize the latent Neural SDE forward pass,
3. finalize BondNet,
4. finalize the futures pricer,
5. finalize the single-stage training loop,
6. add tests on shapes, data flow, and simple synthetic cases,
7. only later consider:
   - bond supervision,
   - repo,
   - options,
   - staged training,
   - more precise coupon logic.

---

## 18. Minimal end-to-end specification

A minimal working version should be able to do the following:

1. read a date $t$,
2. build a yield-history encoder input,
3. encode to $z_t$,
4. simulate latent paths forward,
5. fit the current yield curve,
6. read one or more futures at date $t$,
7. map delivery dates to the simulation grid,
8. compute delivery-date bond features,
9. evaluate BondNet,
10. compute pathwise CTD values,
11. compute model-implied futures prices,
12. compute yield + futures loss,
13. backpropagate.

If all of these are working coherently, the core of the project is in place.

---

## 19. Final summary

This project is a **risk-neutral neural short-rate calibration framework** for Treasury curves and futures.

Mathematically:
- the short rate is decoded from a latent Neural SDE,
- yields are priced through discount factors,
- delivery-basket bond prices at futures delivery are approximated by BondNet,
- futures are priced through pathwise CTD logic.

Algorithmically:
- yield curves define the canonical calendar and encoder history,
- a latent state is inferred and simulated,
- futures delivery dates are aligned to the simulation grid without look-ahead,
- BondNet produces delivery-date bond values,
- futures losses are computed through CTD-adjusted Monte Carlo pricing.

This document should be used as the reference mathematical and algorithmic blueprint for completing the implementation.
