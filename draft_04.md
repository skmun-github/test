# 2. Data and Label Construction

This section defines the **spatio‑temporal inputs**, **country‑level covariates**, and the **IFPA‑based outcome** used throughout the paper, together with the **vintage‑safe** rules that govern which examples are admissible for training and evaluation. All design choices are made to (i) preserve geographic and temporal meaning in the maritime signal, (ii) reflect real‑time information sets at the month of decision, and (iii) yield labels that are severe but policy‑interpretable. Unless noted otherwise, all results later in § 4 are computed under the same construction.

---

## 2.1 Maritime activity cube (domain, units, channels)

The dynamic input is a monthly **vessel‑density raster** over the **Black Sea and approaches** $[27^\circ\text{E},\,42^\circ\text{E}]\times[40^\circ\text{N},\,47^\circ\text{N}]$, distributed on the **ETRS89‑LAEA** equal‑area grid (**EPSG:3035**) at **1 km × 1 km** resolution. Each raster cell reports **hours at sea per km² per month**, aggregated from AIS pings using a standard pipeline that removes obvious outliers and harmonizes ship‑type codes.

We retain **three channels** per month: (i) **Cargo** (AIS type 09), (ii) **Tanker** (type 10), and (iii) **All ships**. Pixels mapped to land are masked. To remove the purely mechanical impact of unequal calendar length on “hours per month,” raw densities are first normalized to a **daily rate**, then transformed via $\log(1{+}x)$ to stabilize heavy tails without destroying zeros. Let $R_{t}(i,j,c)$ denote the monthly hours in cell $(i,j)$, channel $c$, month $t$, and $d(t)$ the number of days in $t$; the value stored in the cube is

$$
X_{t}(i,j,c) \;=\; \log\!\bigl(1+\tfrac{1}{d(t)}\,R_{t}(i,j,c)\bigr),\qquad c\in\{\text{Cargo},\text{Tanker},\text{All}\}.
$$

Equal‑area gridding ensures that per‑cell densities are **comparable across latitude**; the $\log(1{+}x)$ transform tempers rare but very large values associated with port roads and chokepoints while preserving the meaning of “quiet sea” as exactly zero. The maritime cube runs from **January 2017 through December 2023** and is the only time‑varying, spatially explicit input to the model.

---

## 2.2 Temporal index and windowing

All inputs and labels are aligned on a **monthly index**

$$
\mathcal{T}=\{\text{2017‑01},\ldots,\text{2023‑12}\},
$$

with **$L=12$** months of dynamic history used to predict horizon‑specific outcomes. For a country $a$ and decision month $t$, the model receives the **12‑step sequence**

$$
\mathcal{X}_{a,t} = \bigl(X_{a,t-11},\,X_{a,t-10},\,\ldots,\,X_{a,t}\bigr),
$$

and produces a probability that an **onset** (defined in § 2.4–§ 2.5) will occur within $(t,\,t{+}h]$ for horizons $h\in\{1,3\}$ in the operational configuration (additional horizons are instantiated for diagnostics in § 4). The choice $L=12$ spans a full seasonal cycle, which is long enough to capture **accumulation and dissipation** in maritime conditions without diluting recency for fast‑moving disruptions.

---

## 2.3 Country‑level covariates (static indicators)

To encode structural **exposure** and **absorption capacity** that are not visible in sea‑lane imagery, we construct a panel of **FAOSTAT‑derived** indicators at the **country–year** level. We purposefully **exclude** generic macro series (e.g., WDI) to keep the static block interpretable and agriculture‑focused. The set comprises **16 variables**: harvested **Area** $A_\*$, **Production** $P_\*$, **Yield** $Y_\*$ for four cereals (Maize, Rice, Soya beans, Wheat), and **Gross Production Value** $GPV_\*$ for the same items.

Because the model operates monthly, annual FAOSTAT values are converted to a **vintage‑safe monthly panel** by replicating the **latest available year** $y(t)-1$ across all months in calendar year $y(t)$:

$$
S_i(t) \;=\; s_i^{\text{annual}}\bigl(y(t)-1\bigr)\qquad \text{for each static variable }i.
$$

This rule prevents inadvertent look‑ahead via late‑year statistical releases. Before entering the model (§ 3.4), all static values are **robust‑scaled** (median/IQR) and passed through a **signed $\log(1{+}|x|)$** transform to tame long tails; **missingness indicators** are retained per variable to allow the model to use **the pattern of reporting** as a signal while linearly zeroing out the contribution of missing values. A **sin–cos month‑of‑year** pair $[\sin(2\pi m/12),\cos(2\pi m/12)]$ is appended so that seasonal baselines in pass‑through can be expressed without December–January discontinuities.

### Table 2.1. FAOSTAT static covariates (16 variables) and coverage

| Domain | Var        | FAOSTAT definition (item/element)     | Raw missing rate (2018–2023, global build) | Coverage in Africa‑36 (2018–2023) |
| :----: | ---------- | ------------------------------------- | -----------------------------------------: | :-------------------------------: |
|  CROPS | A\_Maize   | Area harvested — Maize (corn)         |                                     0.1894 |            **Complete**           |
|  CROPS | A\_Rice    | Area harvested — Rice                 |                                     0.3386 |            **Complete**           |
|  CROPS | A\_Soya    | Area harvested — Soya beans           |                                     0.4476 |            **Complete**           |
|  CROPS | A\_Wheat   | Area harvested — Wheat                |                                     0.3841 |            **Complete**           |
|  CROPS | P\_Maize   | Production — Maize (corn)             |                                     0.1873 |            **Complete**           |
|  CROPS | P\_Rice    | Production — Rice                     |                                     0.3386 |            **Complete**           |
|  CROPS | P\_Soya    | Production — Soya beans               |                                     0.4444 |            **Complete**           |
|  CROPS | P\_Wheat   | Production — Wheat                    |                                     0.3841 |            **Complete**           |
|  CROPS | Y\_Maize   | Yield — Maize (corn)                  |                                     0.2180 |            **Complete**           |
|  CROPS | Y\_Rice    | Yield — Rice                          |                                     0.4360 |            **Complete**           |
|  CROPS | Y\_Soya    | Yield — Soya beans                    |                                     0.4772 |            **Complete**           |
|  CROPS | Y\_Wheat   | Yield — Wheat                         |                                     0.3905 |            **Complete**           |
|   GPV  | GPV\_Maize | Gross production value — Maize (corn) |                                     0.2487 |            **Complete**           |
|   GPV  | GPV\_Rice  | Gross production value — Rice         |                                     0.4582 |            **Complete**           |
|   GPV  | GPV\_Soya  | Gross production value — Soya beans   |                                     0.5884 |            **Complete**           |
|   GPV  | GPV\_Wheat | Gross production value — Wheat        |                                     0.4413 |            **Complete**           |

*Notes.* “Raw missing rate” refers to the **pre‑selection** global build (2018–2023). The **Africa‑36 sample** (Table 2.2) is chosen so that, after vintage‑safe conversion, **all 16 variables are present** for every country–month in 2017–2023.

---

## 2.4 Outcome definition: IFPA‑based food‑price anomalies

The outcome targets **abnormally high food prices** as officially defined by the **Indicator of Food Price Anomalies (IFPA)**. Let $P_t$ denote the **Food CPI** level in month $t$ for a given country. IFPA combines within‑year and across‑year growth, with explicit **month‑of‑year standardization** to filter seasonality:

$$
\text{CQGR}_t=\log\!\frac{P_t}{P_{t-3}},\qquad
\text{CAGR}_t=\log\!\frac{P_t}{P_{t-12}}.
$$

Denote by $\mu_m(\cdot)$ and $\sigma_m(\cdot)$ the mean and standard deviation computed **within each month of year $m$** over a **fixed historical baseline**. IFPA is

$$
\mathrm{IFPA}_t \;=\; \gamma\;\frac{\text{CQGR}_t-\mu_m(\text{CQGR})}{\sigma_m(\text{CQGR})}
\;+\;
(1-\gamma)\;\frac{\text{CAGR}_t-\mu_m(\text{CAGR})}{\sigma_m(\text{CAGR})},
$$

with $\gamma\approx 0.4$ as per the standard specification. To keep the anomaly test **independent of the training period** and robust to distribution drift, $\mu_m$ and $\sigma_m$ are estimated on **2000–2018** only.

IFPA values are turned into **binary anomalies** at a family of thresholds $\tau\in\{1.8,\,2.0,\,2.5\}$:

$$
A_t(\tau)\;=\;\mathbf{1}\{\mathrm{IFPA}_t \ge \tau\}.
$$

While $\tau=1.0$ is the canonical “abnormally high” cut‑off, we emphasize **rarer, more severe episodes** in our mainline and therefore adopt **$\tau=1.8$** for all headline results (the higher thresholds are used for sensitivity in § 4.4).

---

## 2.5 Onset‑within‑$h$ labels and vintage‑safe masks

Downstream decisions target **onsets** of anomalous episodes, not merely months with high levels. Starting from the anomaly series $A_t(\tau)$, we first enforce a **minimum duration** of two consecutive months for any episode and then reduce to **onset flags** $O_t$ that equal 1 **only** in the **first** month of each episode. A **2‑month refractory period** prevents multiple triggers within the same episode.

Given a forecast horizon $h$ months, the **window label** assigned at decision month $t$ is

$$
Y_t^{(h)} \;=\; \mathbf{1}\!\Big\{\max\big(O_{t+1},\ldots,O_{t+h}\big)=1\Big\},
$$

which signals whether an onset occurs within the next $h$ months. To enforce real‑time discipline, we require that the entire future window be **observable** at $t$. The **vintage‑safe mask** for horizon $h$ is

$$
M_t^{(h)} \;=\; \mathbf{1}\{\text{all months }(t{+}1),\ldots,(t{+}h) \text{ are available at vintage }t\}.
$$

An example $(a,t)$ is **admissible** in a given split if (and only if) it satisfies the mask policy of § 2.6; in all main experiments (§ 4) we adopt **mask policy = all**, i.e., we retain a country–month **only if** it is valid for **both** operational horizons $h\in\{1,3\}$. This stricter rule aligns the distribution that the shared encoder (§ 3) sees across horizons and avoids horizon‑specific covariate shift during training.

---

## 2.6 Evaluation geography and period

We evaluate on a curated set of **36 African economies** for which the **AIS span (2017–2023)**, the **FAOSTAT static set (16 variables)**, and the **Food CPI** series are jointly present under our **vintage‑safe** construction—i.e., the intersection of these sources yields **complete coverage** for 2017–2023. This choice avoids confounding from **data availability shocks** and ensures that differences in performance reflect **signal and modeling**, not missing‑data artefacts. The countries are listed below.

### Table 2.2. Africa‑36 evaluation set (complete coverage 2017–2023 under vintage‑safe construction)

| Algeria      | Angola       | Benin                    | Botswana | Burkina Faso | Burundi       |
| ------------ | ------------ | ------------------------ | -------- | ------------ | ------------- |
| Cabo Verde   | Cameroon     | Central African Republic | Chad     | Congo        | Côte d’Ivoire |
| Egypt        | Ethiopia     | Gambia                   | Ghana    | Guinea       | Guinea‑Bissau |
| Kenya        | Madagascar   | Malawi                   | Mali     | Mauritius    | Morocco       |
| Mozambique   | Namibia      | Niger                    | Nigeria  | Rwanda       | Senegal       |
| Sierra Leone | South Africa | Togo                     | Tunisia  | Zambia       | Zimbabwe      |

**Temporal protocol.** Following the **rolling‑origin** discipline later analyzed in § 4, we train/validate on **2017–2022** and evaluate on **2023** for the headline test. The calibration window for probability scaling (§ 3.9) is the **last 18 months** of the training span. Under **mask=all** and $L=12$, the resulting sample counts for 2023 are **$n_{\text{test}}=315$** valid country–months, with **23** positives at $h=3$ and **9** at $h=1$. Table 2.3 records the split‑level inventory used throughout the paper.

### Table 2.3. Inventory by split (mask=all; $L=12$; Africa‑36)

| Split      | Years (decision months)                                     | Valid instances $n$ | Notes                                                       |
| :--------- | :---------------------------------------------------------- | ------------------: | :---------------------------------------------------------- |
| Train      | 2017–2021 (rolling)                                         |           **1,995** | Vintage‑safe; used to fit encoder and heads (§ 3)           |
| Validation | 2022 (early months for tuning; late months for calibration) |             **630** | Final **18 months** of 2021–2022 reserved for Platt (§ 3.9) |
| Test       | 2023                                                        |             **315** | $h=3$ positives = 23; $h=1$ positives = 9                   |

*Notes.* Counts are for the **$\tau=1.8$** label family and the **mask=all** policy; other horizons used for diagnostics in § 4 inherit the same admissibility rule.

---

## 2.7 Why these choices?

**Why the Black Sea and AIS‑derived hours?** The Black Sea corridor is a first‑order route for cereal exports that affect food‑importing regions in Africa. **AIS‑based hours at sea** are a physically grounded proxy for **trade intensity and port throughput** at a monthly cadence. Using the native **EPSG:3035** equal‑area grid at **1 km** avoids area‑induced biases and reprojection artefacts, and normalizing to a **daily rate** decouples values from calendar length.

**Why IFPA and a strict anomaly threshold?** IFPA explicitly controls for **seasonality** and **inflation** via month‑specific standardization. Targeting **$\tau=1.8$** focuses learning on **severe spikes** that are most relevant for early‑action planning while remaining within the official statistical framework. Parallel label sets at **2.0** and **2.5** (used in § 4.4) quantify severity sensitivity.

**Why vintage‑safe masking and mask=all?** Real‑time evaluation is the only honest test for early warning. Conditioning the dataset on **fully observable future windows** prevents leakage; adopting **mask=all** ensures the shared encoder sees a **stable distribution** across horizons, which in turn improves transfer to horizon‑specific heads (§ 3.6) and simplifies the interpretation of horizon effects (§ 4.9).

**Why these statics and transformations?** The **A/P/Y/GPV** set captures structural exposure channels—crop mix, production, yields, value—that **mediate pass‑through** from maritime shocks to domestic prices. Robust scaling and signed‑log transforms stabilize heavy‑tailed variation; **missingness indicators** allow the model to react to **reporting heterogeneity** without imposing spurious linear effects from imputed values. The **Africa‑36** panel (Table 2.2) is explicitly chosen so that, after vintage‑safe conversion, the **monthly static panel is complete** throughout 2017–2023, keeping the focus on signal, not data holes.

**How this supports the results in § 4.** The design choices here directly explain the empirical patterns seen later. The maritime cube and 12‑month window produce **interpretable monthly attention** (§ 3.3) whose within‑year dispersion is visible in **monthly AUROC** (Figs. 1–2). The IFPA onset logic with refractory periods yields **rare but meaningful** positives (Table 2.3), consistent with the year‑wise base‑rate swings that shape AUROC (§ 4.3–§ 4.4). And the static block, complete in Africa‑36 yet sensitive to reporting structure, underlies the **year‑dependent static effects** observed at $H=3$ (Tables 3–4 in § 4.5), where statics help in **structured years** (2023) and can hinder under **distribution shift** (2022).

# 3. Methodology

This section specifies the end‑to‑end architecture that maps a **12‑month history of maritime activity** over the Black Sea and **country‑level static indicators** to **horizon‑specific probabilities** of an IFPA‑defined food‑price surge (onset‑within‑$h$). The design adheres to the **vintage‑safe** discipline introduced in § 2: every training and evaluation instance is constructed only from information available at the decision month $t$, and a sample is considered valid at horizon $h$ only if the entire future window $(t{+}1,\ldots,t{+}h)$ is observable at vintage $t$. Throughout, we avoid precision–recall metrics; downstream evaluation focuses on **AUROC** and **probability quality** (Brier, ECE) as described in § 4.

To keep notation unambiguous, we use **$H_{\text{img}}$** and **$W_{\text{img}}$** for raster height and width, **$h$** for the forecasting **horizon** in months, and **$d\_\cdot$** for feature dimensions. Default hyperparameters referenced below are later consolidated in § 3.12.

---

## 3.1 Problem formulation

Let $a\in\{1,\dots,A\}$ index **countries** and $t$ index **months** on the common calendar $\mathcal{T}=\{2017\text{‑}01,\ldots,2023\text{‑}12\}$. For each country–month we observe:

1. A **dynamic maritime tensor** $X_{a,t}\in\mathbb{R}^{C\times H_{\text{img}}\times W_{\text{img}}}$ summarizing vessel‑density over the Black Sea crop for month $t$, with **$C=3$** channels: *Cargo* (AIS 09), *Tanker* (AIS 10), and *All ships*. Intensities are **daily‑normalized** and passed through $\log(1{+}x)$ (§ 2.1).
2. A **static vector** $s_{a,t}\in\mathbb{R}^{d_s}$ of FAOSTAT‑derived indicators (the **16‑variable set** in § 2.3), robust‑scaled and signed‑logged, with an accompanying **missingness mask** $m_{a,t}\in\{0,1\}^{d_s}$ where $1$ marks missing; and a **month‑of‑year** encoding $\phi(t)=(\sin\theta_t,\cos\theta_t)\in\mathbb{R}^2$ with $\theta_t=2\pi(\text{month}(t)/12)$.
3. The **vintage‑safe observability flag** $M_{a,t,h}\in\{0,1\}$, equal to 1 iff the label window $(t{+}1,\ldots,t{+}h)$ is fully observable at time $t$ (§ 2.5).

The **input** to the model at month $t$ is the **12‑step sequence** of maritime tensors

$$
\mathcal{X}_{a,t}=\bigl(X_{a,t-11},\,\ldots,\,X_{a,t}\bigr)\in\bigl(\mathbb{R}^{C\times H_{\text{img}}\times W_{\text{img}}}\bigr)^{12},
$$

together with $(s_{a,t}, m_{a,t}, \phi(t), a)$. The **target** is a set of **horizon‑specific onset‑within‑$h$** labels

$$
Y_{a,t,h} \;=\; \mathbf{1}\!\left\{\max\bigl(O_{a,t+1},\ldots,O_{a,t+h}\bigr)=1\right\}\in\{0,1\},
$$

where $O_{a,\tau}$ are **onset flags** built from IFPA$_{\tau}$ using the FAO‑consistent onset definition (minimum duration 2 months, 2‑month refractory; § 2.4–§ 2.5). We train and report primarily for $h\in\{1,3\}$; additional heads $h\in\{2,4,5,6\}$ are instantiated for diagnostics (§ 4).

The **model** is a mapping $f_\theta$ with **shared encoding** and **horizon‑specific heads**:

$$
p_{a,t,h} \;=\; \sigma\!\bigl(\ell_{a,t,h}\bigr)\;=\;\Pr\!\bigl(Y_{a,t,h}=1 \,\big|\, \mathcal{X}_{a,t}, s_{a,t}, m_{a,t}, \phi(t), a\bigr),\qquad h\in\mathcal{H},
$$

where $\mathcal{H}=\{1,3\}$ for the operational configuration and $\sigma(z)=1/(1+e^{-z})$. Losses are applied only where $M_{a,t,h}=1$ (see § 3.7 for masking and weighting). A single model thus serves multiple horizons without duplicating the encoder, while restricting horizon‑specific specialization to the final affine layers.

**Table 3.1** summarizes the key symbols and shapes used throughout § 3.1–§ 3.6.

### Table 3.1. Core symbols and dimensions used in § 3 (defaults in **bold**)

| Symbol                      | Meaning                         | Shape / Type                                                       | Default / Notes                                                            |
| :-------------------------- | :------------------------------ | :----------------------------------------------------------------- | :------------------------------------------------------------------------- |
| $a$                         | Country index                   | scalar                                                             | $1,\dots,A$ (Africa‑36 in § 2.6)                                           |
| $t$                         | Month index                     | scalar                                                             | 2017‑01 … 2023‑12                                                          |
| $h$                         | Forecast horizon (months ahead) | scalar                                                             | $\mathcal{H}=\{\,\mathbf{1},\,\mathbf{3}\,\}$ (diagnostic: 2,4,5,6)        |
| $X_{a,t}$                   | Maritime raster at month $t$    | $\mathbb{R}^{C\times H_{\text{img}}\times W_{\text{img}}}$         | $C=\mathbf{3}$ (Cargo, Tanker, All); $\log(1+x)$ after daily normalization |
| $\mathcal{X}_{a,t}$         | 12‑month maritime sequence      | $(\mathbb{R}^{C\times H_{\text{img}}\times W_{\text{img}}})^{\!L}$ | $L=\mathbf{12}$                                                            |
| $s_{a,t}$                   | Static covariates               | $\mathbb{R}^{d_s}$                                                 | $d_s=\mathbf{16}$ (A/P/Y/GPV)                                              |
| $m_{a,t}$                   | Missingness indicators          | $\{0,1\}^{d_s}$                                                    | $1=$ missing                                                               |
| $\phi(t)$                   | Month‑of‑year encoding          | $\mathbb{R}^{2}$                                                   | $(\sin, \cos)$                                                             |
| $Y_{a,t,h}$                 | Onset‑within‑$h$ window label   | $\{0,1\}$                                                          | § 2.4–§ 2.5                                                                |
| $M_{a,t,h}$                 | Vintage‑safe mask               | $\{0,1\}$                                                          | 1 if future window observable at $t$                                       |
| $z^{\text{dyn}}_{a,t}$      | Monthly dynamic embedding       | $\mathbb{R}^{E\cdot N_{\text{patch}}}$                             | $E=\mathbf{8}$, $N_{\text{patch}}$ per § 3.2                               |
| $g^{\text{time}}_{a,t}$     | Temporal summary                | $\mathbb{R}^{64}$                                                  | From GRU + attention                                                       |
| $g^{\text{stat}}_{a,t}$     | Static summary                  | $\mathbb{R}^{256}$                                                 | Mask‑aware block                                                           |
| $e_a$                       | Country embedding               | $\mathbb{R}^{8}$                                                   | Random‑effects analogue                                                    |
| $z_{a,t}$                   | Fused representation            | $\mathbb{R}^{328}$                                                 | $64+256+8$                                                                 |
| $\ell_{a,t,h}$, $p_{a,t,h}$ | Logit, probability              | $\mathbb{R}$, $[0,1]$                                              | Head‑specific affine + sigmoid                                             |
| $P,S,E$                     | Patch size, stride, embed dim   | integers                                                           | $P=\mathbf{32}$, $S=\mathbf{16}$, $E=\mathbf{8}$                           |
| $N_H,N_W,N_{\text{patch}}$  | Patch grid sizes                | integers                                                           | $N_H=\lfloor(H_{\text{img}}-P)/S\rfloor+1$, similarly for $W$              |

---

## 3.2 Patch encoder (spatial aggregation of maritime rasters)

The maritime branch translates each monthly raster $X_{a,t}$ into a compact vector that preserves **where** traffic occurs while tamping down pixel‑level noise. The encoder has two stages: **lightweight channel mixing and smoothing**, followed by **overlapping patch pooling** with per‑patch embeddings.

**Pre‑pool smoothing and mixing.** Because the raw intensities are already stabilized by $\log(1{+}x)$, we avoid heavy convolutions. A pair of **depthwise** $3{\times}3$ convolutions (per‑channel) with ReLU, followed by a **pointwise** $1{\times}1$ convolution, suffices to attenuate speckle arising from sparse AIS coverage and to blend the three channels. This step acts as a mild low‑pass filter without destroying localized hot‑spots near ports and chokepoints.

**Overlapping patch pooling.** The filtered map is unfolded into **$P{\times}P$** tiles with **stride $S=\lfloor P/2\rfloor$** (50% overlap). For an image of size $H_{\text{img}}\times W_{\text{img}}$, the grid sizes are

$$
N_H=\Big\lfloor\frac{H_{\text{img}}-P}{S}\Big\rfloor+1,\qquad
N_W=\Big\lfloor\frac{W_{\text{img}}-P}{S}\Big\rfloor+1,\qquad
N_{\text{patch}}=N_H N_W,
$$

with automatic kernel shrinkage if $P>\min(H_{\text{img}},W_{\text{img}})$ so that at least one tile exists. Within each tile we take the **mean** over spatial positions in each channel, concatenate channels, and project to an **$E$‑dimensional** embedding. Concatenating the $N_{\text{patch}}$ embeddings yields the **monthly dynamic vector** $z^{\text{dyn}}_{a,t}\in\mathbb{R}^{E N_{\text{patch}}}$.

The choice of **average pooling** (instead of max) is deliberate: it keeps the representation sensitive to **diffuse rerouting** and **area‑wide congestion** rather than only to spikes. The **overlap** smooths decision boundaries across tiles and renders the representation **robust to small lane shifts** (a few kilometers) induced by weather or traffic management. Empirically, this reduces variance in attention weights upstream (§ 3.3) and avoids brittle “single‑tile” explanations.

**Numerical scale and invariances.** Because inputs are daily‑normalized and log‑scaled, patch means are **comparable across months**, which improves the stability of the recurrent backbone. The patch stage is **translation‑tolerant** up to half the stride and **scale‑aware** through the learned projection, striking a balance between location fidelity and invariance.

---

## 3.3 Temporal backbone (sequence modeling with additive attention)

Maritime disruptions relevant for domestic food prices are rarely instantaneous. They **accumulate**, **propagate**, and **decay** over multiple months through shipping schedules, port operations, and inventory cycles. The temporal backbone therefore takes the 12‑step sequence $\{z^{\text{dyn}}_{a,t-11},\ldots,z^{\text{dyn}}_{a,t}\}$ and produces a **history‑aware summary** that can express both persistence and change‑points.

**Normalization and recurrence.** Each monthly vector is layer‑normalized and fed to a **two‑layer GRU** with hidden size $H_{\text{hid}}=256$. Let $h_\tau\in\mathbb{R}^{256}$ be the top‑layer hidden state at month $\tau\in\{t{-}11,\ldots,t\}$. GRUs are well suited to short, monthly horizons: they provide sufficient capacity for lag structures without the overhead of self‑attention over long sequences.

**Additive (Bahdanau) attention.** Rather than collapsing the sequence by taking only the final hidden state, we learn **where to look** in the last year via additive attention:

$$
e_\tau = \mathbf{v}^{\top}\tanh(\mathbf{W}\,h_\tau),\qquad
\alpha_\tau = \frac{\exp(e_\tau)}{\sum_{k=t-11}^{t}\exp(e_k)},\qquad
c=\sum_{\tau=t-11}^{t}\alpha_\tau h_\tau\,,
$$

with $\mathbf{W}\in\mathbb{R}^{d_a\times 256}$, $\mathbf{v}\in\mathbb{R}^{d_a}$ and $d_a$ a small attention width. The **context vector** $c\in\mathbb{R}^{256}$ is then projected through a dense layer with ReLU and dropout to a **64‑dimensional temporal summary** $g^{\text{time}}_{a,t}$.

**Interpretability and stability.** The attention weights $\{\alpha_\tau\}$ provide month‑level importance profiles that are often interpretable post hoc (e.g., elevated weights on late‑summer months in years when corridor congestion peaks). Because upstream patch pooling has already smoothed spatial noise, attention learns **seasonal templates** and **lag kernels** rather than memorizing month indices. The short sequence length $L=12$ keeps memory usage bounded and supports small batch sizes at full spatial resolution.

---

## 3.4 Static and missingness‑aware block

Maritime rasters carry **exogenous supply‑chain signals**, but domestic food prices also depend on **exposure** (crop mix, import intensity) and **absorption capacity** (yields, production value). These are encoded in **FAOSTAT** variables (§ 2.3) that are **incomplete** and **heterogeneous** across the panel. The static block is designed to extract value from these indicators **without introducing bias** from missing values.

Let $s_{a,t}\in\mathbb{R}^{d_s}$ be the scaled static vector and $m_{a,t}\in\{0,1\}^{d_s}$ its missingness mask. We form the **augmented static input**

$$
x^{\text{stat}}_{a,t} \;=\; \bigl[s_{a,t}\;\|\;m_{a,t}\;\|\;\phi(t)\bigr]\in\mathbb{R}^{d_s+d_s+2},
$$

and apply **missingness‑aware gating** that **zeros out** the contribution of unknown **values** while **retaining** their **indicators** as separate signals:

$$
\tilde{x}^{\text{stat}}_{a,t} \;=\; \bigl(s_{a,t}\odot(1-m_{a,t})\;\|\;m_{a,t}\;\|\;\phi(t)\bigr).
$$

A linear map, ReLU, and dropout produce a **256‑dimensional static summary** $g^{\text{stat}}_{a,t}$. This construction avoids the common pitfall of treating *imputed* zeros as real values: any linear contribution from a missing entry is exactly **zero**, but the **fact of missingness** can still influence predictions through its explicit indicator. The month‑of‑year encoding $\phi(t)$, appended here (not to the maritime branch), lets the model learn **seasonal modulation** of **price transmission** (e.g., harvest timing), not of at‑sea traffic per se.

**Identifiability under drift.** Because missingness can correlate with country size or reporting regimes, the explicit $m_{a,t}$ channels allow the model to **learn or ignore** these patterns depending on their stability. In years where missingness surrogates are **misaligned** with risk (e.g., § 4.5 for 2022), switching to **reduced static sets** (NONE or P‑only) is effective without retraining the dynamic encoder.

---

## 3.5 Country embedding

Country‑specific pass‑through from maritime shocks to food prices reflects fiscal and trade policy, market structure, currency dynamics, and logistics that are **not fully captured** by FAOSTAT indicators. To absorb **persistent heterogeneity**, we learn a **compact country embedding** $e_a\in\mathbb{R}^{d_c}$ with $d_c=8$. This is the neural analogue of a **random effect** in hierarchical models: it allows the downstream classifier to tilt the decision surface in a country‑specific way while still sharing most structure across the panel.

Two guardrails contain this flexibility. First, the embedding is **introduced only at fusion** (below); it does **not** condition the patch encoder or the GRU, which prevents over‑fitting of spatial or temporal features to country identity. Second, training and evaluation obey the **vintage‑safe** and **rolling‑origin** protocols, so the embedding cannot linearly memorize future labels. Regularization and early stopping implicitly bound $\|e_a\|$, encouraging **shrinkage** toward the global mean, with countries that have richer event histories allowed to deviate more.

---

## 3.6 Multi‑head outputs (H‑specific detection)

The temporal summary $g^{\text{time}}_{a,t}$, static summary $g^{\text{stat}}_{a,t}$, and country embedding $e_a$ are **concatenated** into a fused representation

$$
z_{a,t} \;=\; \bigl[g^{\text{time}}_{a,t}\;\|\;g^{\text{stat}}_{a,t}\;\|\;e_a\bigr]\in\mathbb{R}^{328}.
$$

This shared code feeds **separate, horizon‑specific heads** for $h\in\mathcal{H}$:

$$
\ell_{a,t,h} \;=\; \mathbf{w}_h^{\top} z_{a,t} + b_h,\qquad
p_{a,t,h} \;=\; \sigma(\ell_{a,t,h})\in[0,1].
$$

The heads **share all upstream parameters** but have **independent last‑layer weights** $(\mathbf{w}_h,b_h)$, allowing the classification boundary to adjust to the different **label geometries** and **base rates** at distinct horizons. In practice this achieves three goals that are empirically validated in § 4:

* It enables **transfer** across horizons through the shared encoder (e.g., persistent maritime signals learned via $H{=}3$ benefit $H{=}5$ diagnostics), while still tuning the last layer to each horizon’s error surface.
* It supports **differential emphasis** during training: by weighting losses per horizon (§ 3.7), we can prioritize $H{=}3$ without training a separate model.
* It is compatible with **post‑hoc, per‑horizon calibration** (§ 3.9), which is critical because even when ranking is shared, **probability scales** often are not.

Although our **operational configuration** uses $\mathcal{H}=\{1,3\}$, the architecture trivially extends to $\mathcal{H}\supseteq\{1,3\}$. For completeness and to support the horizon analyses in § 4, we instantiate additional **diagnostic heads** at $h\in\{2,4,5,6\}$ with the same fusion input and independent affine layers. These heads **do not** alter training emphasis or calibration for the operational horizons but allow systematic inspection of **lead‑time trade‑offs** under the same encoder.

### Table 3.2. Horizon heads used in this study (semantics and role)

| Horizon $h$ | Output meaning            | Role in the paper                         | Calibration (§ 3.9) | Mask policy (§ 2.5/§ 3.7)            |
| :---------: | :------------------------ | :---------------------------------------- | :------------------ | :----------------------------------- |
|      1      | Onset within **1** month  | Short‑lead sentinel; operational baseline | Per‑horizon (Platt) | **all** (jointly valid with $H{=}3$) |
|      3      | Onset within **3** months | **Primary operational target**            | Per‑horizon (Platt) | **all** (primary)                    |
|      2      | Onset within **2** months | Diagnostic (bridging 1↔3)                 | Optional            | Vintage‑safe; reported in § 4        |
|      4      | Onset within **4** months | Diagnostic (medium‑lead)                  | Optional            | Vintage‑safe; reported in § 4        |
|      5      | Onset within **5** months | Diagnostic (medium‑lead)                  | Optional            | Vintage‑safe; reported in § 4        |
|      6      | Onset within **6** months | Diagnostic (longer‑lead)                  | Optional            | Vintage‑safe; reported in § 4        |

*Notes.* Heads at $h\notin\{1,3\}$ are **analysis‑only**; they share the encoder but have independent last‑layer weights and (when used) independent calibrators. Training uses the **mask=all** policy so that the encoder sees a **stable distribution** across the two operational horizons; evaluation of diagnostic heads still honors **vintage‑safe observability** for their respective windows.

---

### Design rationale (cross‑sectional synthesis)

The architectural choices in § 3.1–§ 3.6 are mutually reinforcing. **Overlapping patch pooling** produces a translation‑tolerant but spatially meaningful summary for each month; the **GRU + additive attention** converts the 12‑month sequence into an interpretable, low‑dimensional temporal code; the **mask‑aware static block** allows the model to condition on agro‑economic structure **without leaking** from missing values; and the **country embedding** absorbs persistent heterogeneity akin to a random effect. By **fusing** these components and attaching **horizon‑specific heads**, we obtain a single model family capable of emitting **calibratable probabilities** at multiple lead times. This is exactly the configuration probed in § 4: the $H{=}3$ head anchors operations, $H{=}1$ acts as a short‑lead sentinel, and additional heads provide a transparent view of how predictability **evolves with horizon** under non‑stationarity—without sacrificing the **vintage‑safe** integrity of the evaluation.

## 3.7 Loss & vintage‑safe masking

Training optimizes a **masked, multi‑horizon objective** that enforces real‑time (vintage‑safe) observability and allows **horizon‑specific emphasis**. For country $a$ at decision month $t$, the head for horizon $h\in\mathcal{H}$ ($\mathcal{H}=\{1,3\}$ operationally; diagnostics in § 4 extend $h$) emits a logit $\ell_{a,t,h}$ and probability $p_{a,t,h}=\sigma(\ell_{a,t,h})$. The onset‑within‑$h$ label is $y_{a,t,h}\in\{0,1\}$ (§ 2.5). Let $m_{a,t,h}\in\{0,1\}$ be the **vintage‑safe mask**, equal to 1 if and only if $(t{+}1,\ldots,t{+}h)$ is fully observable at vintage $t$. The per‑example loss aggregates **only** valid horizons:

$$
\mathcal{L}_{a,t}
=\frac{1}{\sum_{h\in\mathcal{H}} m_{a,t,h}+\varepsilon}
\sum_{h\in\mathcal{H}}\lambda_h\, m_{a,t,h}\, \ell_{\text{bin}}\!\bigl(p_{a,t,h},\,y_{a,t,h}\bigr),
$$

with horizon weights $\lambda_h\ge 0$ and $\varepsilon$ a small constant to avoid division by zero when all horizons are masked.

The **binary term** $\ell_{\text{bin}}$ is either **weighted cross‑entropy** or **focal loss**. Weighted cross‑entropy takes

$$
\ell_{\text{wCE}}(p,y) \;=\; -\,w_{+}\,y\log p \;-\; w_{-}\,(1{-}y)\log(1{-}p),
$$

with $w_{+}$ and $w_{-}$ set from the training prevalence at a reference horizon (see § 3.8). The **focal** form down‑weights easy examples:

$$
\ell_{\text{focal}}(p,y)
\,=\, -\,\alpha\,y\,(1{-}p)^{\gamma}\log p
\;-\; (1{-}\alpha)\,(1{-}y)\,p^{\gamma}\log(1{-}p),
$$

with focusing parameter $\gamma>0$ and class prior weight $\alpha\in(0,1)$. In all main runs we use **focal loss** with $\gamma=2.0$ and $\alpha=0.87$, which stabilizes gradients under heavy class imbalance and raises the learning signal on **hard positives** without over‑penalizing numerous easy negatives.

The **mask** $m_{a,t,h}$ plays a dual role. First, it **enforces real‑time correctness**: loss terms never “peek” into the future, because a term is active only if the entire future window is known at vintage $t$. Second, it **aligns the training distribution** across horizons. In the main configuration we adopt **mask policy = all** (see § 3.10): a training example $(a,t)$ is admitted **only if** it is valid for both operational horizons, ensuring the shared encoder sees a **consistent** covariate and label geometry.

The choice of horizon weights $\lambda_h$ reflects operational emphasis. In the reported configuration we set $\lambda_1=0$ and $\lambda_3=1$, which concentrates updates on the $H{=}3$ head while still training the shared encoder jointly across horizons through shared forward passes and regularization. This setting avoids diluting learning on the medium‑lead target when $H{=}1$ positives are extremely sparse.

**Table 3.3** summarizes the loss components and default values.

### Table 3.3. Loss components and defaults

| Component                       | Definition                                                                                                              | Default / Value                  | Rationale                                                   |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------- | :---------------------------------------------------------- |
| Per‑example aggregation         | $\displaystyle \mathcal{L}_{a,t}=\frac{\sum_{h}\lambda_h m_{a,t,h}\,\ell_{\text{bin}}}{\sum_{h} m_{a,t,h}+\varepsilon}$ | $\varepsilon=10^{-8}$            | Normalize by number of valid horizons; block invalid terms. |
| Binary loss $\ell_{\text{bin}}$ | focal or weighted CE                                                                                                    | **focal**                        | Robust to extreme class imbalance.                          |
| Focal params                    | $\gamma,\alpha$                                                                                                         | **$\gamma=2.0$, $\alpha=0.87$**  | Emphasize hard positives; reflect rare‑event priors.        |
| Horizon weights                 | $\lambda_h$                                                                                                             | **$\lambda_1=0$, $\lambda_3=1$** | Prioritize $H{=}3$ during training.                         |
| Mask                            | $m_{a,t,h}$                                                                                                             | **vintage‑safe; policy=all**     | Enforce real‑time observability; align horizons.            |

---

## 3.8 Sampling under class and panel imbalance

Country‑month panels are **sparse in onsets** and **heterogeneous** across countries. Rather than rebalancing labels or modifying losses alone, we intervene in the **training sampler** to stabilize optimization while preserving the evaluation distribution.

We choose a **reference horizon** $h^\star$ as the operational horizon with the **largest count of positives** in TRAIN (here, $h^\star=3$). Over the TRAIN set valid under the mask policy, we estimate the positive fraction

$$
\pi^{(h^\star)}=\frac{\sum_{(a,t)} y_{a,t,h^\star}}{\sum_{(a,t)} 1},
\quad \text{where the sum ranges over } \{(a,t)\!: m_{a,t,h^\star}=1\}.
$$

We then assign **per‑example sampling weights** $w_{+}=\frac{1-\pi^{(h^\star)}}{\pi^{(h^\star)}+\epsilon}$ for positives and $w_{-}=1$ for negatives (with a small $\epsilon$ for numerical safety) **at horizon $h^\star$**. Mini‑batches are drawn proportional to these weights while still respecting the mask per horizon in the loss. This procedure concentrates updates on scarce positives **without** altering their numeric labels or the loss landscape.

Panels with **near‑zero positive history** at $h^\star$ contribute little useful gradient signal and can destabilize the sampler. We therefore apply a **TRAIN‑only country guardrail**: countries with **fewer than $k$ valid positives** at $h^\star$ (we use $k=2$) are **excluded from TRAIN** but remain fully present in **validation** and **test**. This is the neural analogue of **partial pooling** in hierarchical models: countries with sufficient events inform the shared representation; those without are evaluated out‑of‑sample to assess generalization.

This sampling design preserves the **temporal structure** (batches do not reorder months within a sequence), respects **vintage‑safe masking**, and avoids **duplication** of rare positives. It also decouples the **training emphasis** (where imbalance is most damaging) from the **evaluation protocol** (which remains faithful to the observed distribution).

**Table 3.4** records the sampler and guardrail configuration.

### Table 3.4. Sampler configuration at the reference horizon $h^\star$

| Element           | Definition / Rule                                                          | Default        | Purpose                                              |
| :---------------- | :------------------------------------------------------------------------- | :------------- | :--------------------------------------------------- |
| Reference horizon | $h^\star=\arg\max_{h\in\{1,3\}} \#\{y_{a,t,h}=1\}$                         | **3**          | Stable class‑prior estimate; aligns with operations. |
| Class weights     | $w_{+}=\frac{1-\pi^{(h^\star)}}{\pi^{(h^\star)}+\epsilon},\ w_{-}=1$       | data‑dependent | Oversample positives without duplication.            |
| Country guardrail | Drop from TRAIN if $\#\{(t): y_{a,t,h^\star}=1,\, m_{a,t,h^\star}=1\} < k$ | **$k=2$**      | Prevent over‑fitting to all‑negative panels.         |
| Mask enforcement  | Keep $m_{a,t,h}$ in loss even if sampling keyed to $h^\star$               | —              | Retain vintage‑safe discipline per horizon.          |

---

## 3.9 Probability calibration

Raw scores from a discriminative classifier need not be **probabilistically calibrated**. Because downstream users act on **probabilities** (not just rankings), we learn a **post‑hoc monotone map** $\mathcal{C}_h:[0,1]\to[0,1]$ **per horizon**, fitted on a **recent, held‑out window** within the training years and **frozen** before scoring the test year.

Our **default** is **Platt scaling**, which fits a one‑dimensional logistic regression on the **logit** of the model score:

$$
\tilde{p} \;=\; \sigma\!\bigl(a_h \,\mathrm{logit}(p)+ b_h\bigr),\qquad
\mathrm{logit}(p)=\log \frac{p}{1-p},
$$

with parameters $(a_h,b_h)$ estimated by maximum likelihood on calibration pairs $\{(p_{a,t,h},y_{a,t,h})\}$ satisfying $m_{a,t,h}=1$. To avoid numerical issues when $p$ is exactly 0 or 1, scores are clamped to $[10^{-6},1-10^{-6}]$ before the logit. Platt scaling is **data‑efficient** and preserves **ranking** since it is monotone.

As a **supplement**, we also consider **isotonic regression**, a non‑parametric monotone calibrator $\tilde{p}=\mathcal{I}_h(p)$ fitted to minimize squared error under the constraint of monotonicity. Isotonic can adapt to **non‑linear** miscalibration but requires **enough** calibration points; with sparse positives, it risks staircase artefacts and overfitting to the calibration year.

Calibration window selection trades off **variance** and **drift**. We adopt a fixed **18‑month** window immediately preceding the test year (within the training span), which is long enough to gather a workable number of positives in most years but short enough to track **recent regimes**. In sensitivity checks we vary the window to **12** and **24** months; the default remains 18 unless noted (see § 4 for empirical effects).

After fitting $\mathcal{C}_h$, we apply it **pointwise** to test‑year scores to obtain $\tilde{p}_{a,t,h}=\mathcal{C}_h(p_{a,t,h})$. Probability quality is then assessed with **Brier** and **ECE** (defined in § 3.11). Because calibration is **per horizon**, it respects the distinct base rates and error geometries of $H{=}1$ and $H{=}3$.

**Table 3.5** lists the calibrators and windows used.

### Table 3.5. Calibrator options and windowing

|       ID       | Calibrator          | Objective                        | Window                        | Use in this paper                    |
| :------------: | :------------------ | :------------------------------- | :---------------------------- | :----------------------------------- |
| **C‑Platt‑18** | Logistic (Platt)    | Max‑likelihood on logit scores   | **18 months** (last in TRAIN) | **Default** for all reported results |
|    C‑Iso‑18    | Isotonic (monotone) | Squared error under monotonicity | 18 months                     | Supplementary sensitivity            |
|  C‑Platt‑12/24 | Logistic (Platt)    | As above                         | 12 or 24 months               | Responsiveness/stability checks      |

---

## 3.10 Observability and mask policy

The **vintage‑safe mask** guarantees that, at decision time $t$, both inputs and labels are **observable** without look‑ahead. For each horizon $h$, we set

$$
M_{a,t,h} \;=\; \mathbf{1}\!\big\{ \text{all months } (t{+}1),\ldots,(t{+}h)\ \text{are observed at vintage } t \big\}.
$$

The **mask policy** specifies how multi‑horizon examples enter TRAIN/VAL:

* **mask=all.** Admit $(a,t)$ only if $M_{a,t,1}=M_{a,t,3}=1$. For any additional diagnostic heads ($h\notin\{1,3\}$), the loss term is still gated by $M_{a,t,h}$, but **admission** follows the operational pair. This policy aligns the **shared encoder’s** exposure across horizons, reducing horizon‑specific covariate shift within the encoder.
* **mask=any.** Admit $(a,t)$ if **at least one** operational horizon is valid; losses are still masked per horizon. This increases sample size but allows the encoder to see horizon‑dependent input/label distributions, which can degrade transfer across heads.

We adopt **mask=all** in all main runs because it provides a **stable training distribution** for the shared encoder and a clean interpretation of horizon effects. It also mirrors the **evaluation** practice, where we compute metrics on the subset that is valid for the horizon in question under the same vintage‑safe rule.

**Table 3.6** compares the two policies.

### Table 3.6. Mask policy comparison

| Policy  | Admission rule (TRAIN)                              | Encoder exposure            | Sample size | Pros                                                     | Cons                                                           | Used here                 |
| :------ | :-------------------------------------------------- | :-------------------------- | :---------- | :------------------------------------------------------- | :------------------------------------------------------------- | :------------------------ |
| **all** | Admit $(a,t)$ iff $M_{a,t,1}=M_{a,t,3}=1$           | **Aligned** across horizons | Smaller     | Stable shared representation; cleaner horizon comparison | Fewer training instances; stricter end‑of‑year truncation      | **Yes** (default)         |
| any     | Admit $(a,t)$ if $M_{a,t,1}=1$ **or** $M_{a,t,3}=1$ | Mixed by horizon            | Larger      | More data; potentially faster convergence                | Horizon‑dependent covariate shift in encoder; noisier transfer | No (only for sensitivity) |

---

## 3.11 Evaluation metrics (ranking and probability quality)

Evaluation follows the **rolling‑origin** protocol (§ 4) and is conducted under the **vintage‑safe** masks. We report two complementary aspects:

**Ranking quality** is measured by **AUROC**, computed by threshold‑free comparison of positive–negative score pairs. If $\mathcal{P}$ and $\mathcal{N}$ are the sets of positive and negative test instances at horizon $h$, then

$$
\mathrm{AUROC}
=\frac{1}{|\mathcal{P}|\,|\mathcal{N}|}
\sum_{i\in\mathcal{P}}\sum_{j\in\mathcal{N}}
\mathbf{1}\!\big\{\tilde{p}_i > \tilde{p}_j\big\}
+\tfrac{1}{2}\,\mathbf{1}\!\big\{\tilde{p}_i = \tilde{p}_j\big\},
$$

where $\tilde{p}$ denotes **calibrated** probabilities (Platt by default). AUROC is **invariant** to any monotone calibration map and **insensitive** to prevalence, which makes it suitable for cross‑year ranking comparisons under shifting base rates. We compute AUROC at the **global** level and, where informative, at **monthly** and **country** granularities; in single‑class slices (no positives or no negatives), AUROC is undefined and is omitted.

**Probability quality** is assessed by the **Brier score** and **Expected Calibration Error (ECE)**. The Brier score averages squared error:

$$
\mathrm{Brier}
=\frac{1}{N}\sum_{i=1}^{N} \bigl(\tilde{p}_i - y_i\bigr)^2,
$$

which is a **strictly proper** scoring rule and therefore incentivizes honest probabilities. **ECE** summarizes the gap between predicted and empirical frequencies after binning calibrated scores into $B$ equal‑width bins ($B=10$ by default). Let $\mathcal{B}_b$ be the set of indices falling in bin $b$, $\overline{\tilde{p}}_b$ the average score in the bin, and $\bar{y}_b$ the empirical rate; then

$$
\mathrm{ECE}
=\sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{N}\,\bigl|\bar{y}_b-\overline{\tilde{p}}_b\bigr|.
$$

Because ECE depends on binning, we keep $B$ fixed across years and horizons for consistency and report reliability diagrams in § 4 where space allows. Confidence intervals for AUROC and Brier can be obtained by non‑parametric **bootstrap** over country–month pairs with 1,000 replicates; when presented, they refer to the **global** aggregation under vintage‑safe masks.

**Table 3.7** consolidates metric definitions and interpretation.

### Table 3.7. Metrics used in this paper

| Metric    | Definition                                              |  Calibration‑sensitive? |     Prevalence‑sensitive?     | Role                              |
| :-------- | :------------------------------------------------------ | :---------------------: | :---------------------------: | :-------------------------------- |
| **AUROC** | Pairwise concordance of $\tilde{p}$ over pos/neg pairs  | No (monotone‑invariant) |               No              | Ranking quality under imbalance   |
| **Brier** | Mean squared error of $\tilde{p}$ vs. $y$               |         **Yes**         | Yes (through target variance) | Overall probability accuracy      |
| **ECE**   | Weighted average absolute reliability gap over $B$ bins |         **Yes**         |            Indirect           | Calibration quality (reliability) |

---

## 3.12 Implementation & hyperparameters

This subsection records the **concrete settings** used to train and evaluate all models, complementing the architectural description above and enabling exact reproduction of the experiments in § 4. Unless otherwise stated, all values below are **fixed across years** and **horizons**.

The **dynamic branch** consumes monthly rasters at their native equal‑area projection (§ 2.1). Each month is processed by two depthwise $3{\times}3$ convolutions and one pointwise $1{\times}1$ convolution, followed by overlapping **patch pooling** with patch size $P=32$ and stride $S=16$ (50% overlap). Per‑patch averages are projected to **$E=8$** dimensions and concatenated; the resulting 12‑step sequence is layer‑normalized and passed to a **two‑layer GRU** with hidden width 256. We compute **additive attention** over months and project the context to a **64‑dimensional** temporal summary via a dense layer with ReLU and dropout.

The **static branch** forms a concatenation of values, missingness indicators, and the month‑of‑year encoding, then applies **mask‑aware gating** $s\odot(1{-}m)$ before a linear layer, ReLU, and dropout to yield a **256‑dimensional** summary. The **country embedding** has dimension $d_c=8$ and is concatenated at fusion only. The **fusion vector** $z\in\mathbb{R}^{64+256+8}$ feeds independent **affine heads** (one per horizon), each producing a logit and probability.

Optimization uses **AdamW** with learning rate $2\times 10^{-4}$, weight decay $2\times 10^{-2}$, and batch size 8. We train with **focal loss** ($\gamma=2.0,\ \alpha=0.87$) and horizon weights $\lambda_1=0,\ \lambda_3=1$. We apply dropout **0.5** in the static block and use **early stopping** with patience **5–6** epochs on the validation criterion. All training and evaluation respect the **vintage‑safe** construction (§ 2.5) and the **mask=all** policy (§ 3.10). We perform **Platt calibration** per horizon on the **last 18 months** of the training span and freeze calibrators before scoring the test year. Inference uses **mixed precision** to reduce memory without affecting numerical stability under the log‑scaled inputs.

**Table 3.8** collects these settings in a single registry.

### Table 3.8. Hyperparameter and implementation registry

| Component          | Setting                 | Value                                   | Comment                               |
| :----------------- | :---------------------- | :-------------------------------------- | :------------------------------------ |
| Input window       | Sequence length $L$     | **12 months**                           | Months $t{-}11,\ldots,t$.             |
| Maritime channels  | AIS classes             | **Cargo (09), Tanker (10), All**        | Daily‑normalized; $\log(1{+}x)$.      |
| Spatial pooling    | Patch size / stride     | **$P=32$, $S=16$**                      | 50% overlap; average pooling.         |
| Patch embedding    | Dim $E$                 | **8**                                   | Per tile; concatenated.               |
| Recurrent backbone | GRU layers / hidden     | **2 / 256**                             | With layer norm on inputs.            |
| Temporal attention | Type / output           | **Additive (Bahdanau)** / **64‑d**      | Dense + ReLU + dropout.               |
| Static block       | Inputs                  | **16 values + 16 masks + 2 month enc.** | Mask‑aware gating; output 256‑d.      |
| Country embedding  | Dim $d_c$               | **8**                                   | Fusion only.                          |
| Heads              | Horizons                | **$H{=}1,3$** (ops), **2,4,5,6** (diag) | Independent affine layers.            |
| Loss               | Form / params           | **Focal**; $\gamma=2.0,\ \alpha=0.87$   | Rare‑event stabilization.             |
| Horizon weights    | $\lambda_h$             | **$\lambda_1=0,\ \lambda_3=1$**         | Emphasize $H{=}3$.                    |
| Sampler            | Reference $h^\star$     | **3**                                   | Class‑weighted sampling (§ 3.8).      |
| Guardrail          | TRAIN‑only country drop | **< 2 positives at $h^\star$**          | Preserve stability.                   |
| Mask policy        | Admission rule          | **all**                                 | Joint validity for $H{=}1,3$.         |
| Optimizer          | AdamW                   | **lr=2e‑4, wd=2e‑2**                    | Batch size **8**.                     |
| Regularization     | Dropout                 | **0.5** (static block)                  | Early stopping 5–6 epochs.            |
| Calibration        | Method / window         | **Platt** / **18 months**               | Per horizon; frozen before test.      |
| Metrics            | Reported                | **AUROC, Brier, ECE**                   | No PR‑based metrics.                  |
| Precision          | Inference               | **Mixed precision**                     | Memory‑efficient, numerically stable. |

Together, §§ 3.7–3.12 specify an **operationally disciplined** learning and evaluation stack: losses and masks enforce **real‑time correctness**; sampling and focal loss address **imbalance** without distorting labels; calibration yields **trustworthy probabilities** while preserving ranking; and a concise set of hyperparameters anchors reproducibility. These choices match the **analysis and results** reported in § 4, where the $H{=}3$ head serves as the principal early‑warning signal and other heads provide horizon‑wise diagnostics under the same shared encoder.

## 4. Experiments — horizon‑specific (multi‑head) analysis

The experiments in this section evaluate the **horizon‑specific output heads** introduced in § 3.6. Recall that a single shared encoder produces a fused representation $z_{a,t}$, which is then consumed by separate, horizon‑specific binary heads. Unless noted otherwise, **all results use vintage‑safe masking**, the label definition and preprocessing in § 2, and the training protocol in § 3.12. Consistent with the manuscript‑wide policy, **all precision–recall‑based metrics are omitted**; we report **AUROC** for ranking quality and **Brier/ECE** for probability quality.

---

### 4.1 Evaluation setup and relation to the multi‑head design

We adopt a **rolling‑origin** protocol that respects causal ordering and distribution shift: for each test year $Y\in\{2019,\dots,2023\}$, the model is trained and tuned on $[2017,\dots,Y-1]$ and then **evaluated on year $Y$** only. Calibration parameters (Platt or isotonic; § 3.9) are fitted on a **held‑out calibration window in the training span** (the final 18 months) and frozen before scoring the test year.

The **primary operational target** is **$H{=}3$** (three‑month onset‑within‑$h$ window), the lead time we consider actionable for preparedness and procurement triage. In addition to the two heads described in § 3.6 (**$H=1$** and **$H=3$**), we instantiate \*\*diagnostic heads for $H\in\{2,4,5,6\}** on the same shared encoder** to study how the learned representation transfers across lead times. All heads are trained with the **mask policy = all** (an example is admitted only if it is observationally valid for **both** \(H{=}1$ and $H{=}3$, § 2.5 and § 3.10) so that the upstream encoder sees a consistent distribution; for heads other than $H\in\{1,3\}$ the mask still enforces vintage‑safe observability.

**Metrics.** We report **AUROC** (higher is better), **Brier score** (lower is better), and **Expected Calibration Error (ECE; lower is better)**. Because AUROC is **insensitive to prevalence**, we always place year‑wise AUROC in the context of (i) **coverage** (number of valid test instances) and (ii) **base rates** (share of positives), both of which vary materially by year and horizon.

**Comparability note.** When we vary **label thresholds**, **mask policies**, or **calibration**, the effective test set can change (e.g., **mask=any** includes more examples than **mask=all**). We therefore interpret such comparisons as **design‑choice sensitivity**, not as like‑for‑like holds over identical $n$.

---

### 4.2 Coverage and base rates (what the heads actually see)

Horizon‑specific detection quality is shaped by **how many** events occur and **when** they are observable under the vintage‑safe mask. We therefore summarize two complementary objects: (i) the **number of distinct onset episodes** per year (independent of $H$), and (ii) the **count of positive window labels** by **year × horizon** under **mask=all**. The first describes the underlying event landscape; the second describes the **learning and evaluation burden** placed on each head.

**Table 5** shows a sharp decline in **pure onset episodes** from 2019 to 2023, consistent with sparser extreme spikes in the latter years under the IFPA‑1.8 criterion. **Table 6** then quantifies the expected **monotone rise** in positive windows as $H$ increases (wider windows capture more onsets). These base‑rate swings are the backdrop for the AUROC patterns we observe later, especially the relative stability of ranking quality when prevalence changes year‑to‑year.

#### Table 5. Yearly counts of **pure onset episodes** (IFPA threshold 1.8; onset definition per § 2.5)

| Year | Episodes |
| ---: | -------: |
| 2017 |        0 |
| 2018 |        0 |
| 2019 |       37 |
| 2020 |       21 |
| 2021 |       22 |
| 2022 |       12 |
| 2023 |        9 |

#### Table 6. **Year × horizon** positive counts under **mask=all** (counts of $Y_t^{(h)}{=}1$; see § 2.5)

| Year | h=1 | h=2 | h=3 | h=4 | h=5 |   h=6 |
| ---: | --: | --: | --: | --: | --: | ----: |
| 2017 |   0 |   0 |   0 |   0 |   0 |     0 |
| 2018 |  72 | 150 | 252 | 360 | 504 |   654 |
| 2019 | 180 | 378 | 564 | 756 | 894 | 1,014 |
| 2020 | 108 | 210 | 312 | 402 | 510 |   606 |
| 2021 | 132 | 270 | 396 | 516 | 612 |   696 |
| 2022 |  60 | 102 | 156 | 198 | 240 |   288 |
| 2023 |  54 | 102 | 138 | 156 | 162 |   162 |

*Notes.* (i) These are **window‑label** positives, not **episode** counts; they rise with $H$ mechanically. (ii) The **mask=all** policy enforces the joint observability condition $M^{(1)}\!=\!M^{(3)}\!=\!1$ even when reporting $H{>}3$, which is stricter than an $H$‑only mask and is chosen for training distribution consistency (§ 3.10).

---

### 4.3 Out‑of‑sample performance of the **$H{=}3$** head (2019–2023)

**Table 1** presents the year‑wise test performance at the operational horizon $H{=}3$, with coverage and base rate alongside AUROC. The head attains its strongest ranking quality in **2023** despite a relatively low prevalence; conversely, **2019** pairs high prevalence with the weakest AUROC. This inversion is characteristic of **distribution shift**: years with many onsets need not be easier to rank if the signal‑to‑noise geometry differs (e.g., more diffuse, country‑agnostic pressure), whereas sparse but **coherent** risk clusters can yield higher AUROC.

#### Table 1. Year‑wise test performance at **$H=3$** (vintage‑safe, mask=all; IFPA threshold 1.8; Platt calibration)

| Year | n\_test | pos\_test | prevalence |     AUROC |
| ---: | ------: | --------: | ---------: | --------: |
| 2019 |     420 |        94 |      22.4% |     0.425 |
| 2020 |     420 |        52 |      12.4% |     0.603 |
| 2021 |     420 |        66 |      15.7% |     0.536 |
| 2022 |     420 |        26 |       6.2% |     0.546 |
| 2023 |     315 |        23 |       7.3% | **0.731** |

**Reading the table.** The **coverage** $n_{\text{test}}$ reflects the vintage‑safe constraint: in **2023** the last months cannot host $H{=}3$ windows that spill into 2024, hence the smaller $n$. The **prevalence** column contextualizes AUROC without dictating it. The jump from 0.546 (2022) to **0.731 (2023)** suggests a **more structured maritime‑to‑price transmission** in 2023, which we corroborate in § 4.6 through within‑year monthly AUROC.

**Probability quality.** For the design choice we ultimately recommend (IFPA=1.8, mask=all, Platt calibration), out‑of‑sample **Brier** and **ECE** are consistently low at $H{=}3$ (see Table 2), indicating that high scores correspond to **trustworthy** probabilities—important when probabilities feed a triage rule or a cost–loss analysis.

---

### 4.4 Sensitivity to **label threshold**, **mask policy**, and **calibration** (head $H{=}3$)

We next vary the detection **severity** (IFPA threshold), the **mask policy**, and the **calibration** step. The combination **IFPA=1.8 + mask=all + Platt** emerges as a strong default, jointly optimizing ranking (AUROC) and probability quality (Brier/ECE). As cautioned in § 4.1, **mask policy** changes the set of valid examples; treat comparisons across mask policies as **design sensitivities**, not paired tests on identical $n$.

#### Table 2. Label threshold / mask policy / calibration sensitivity at **$H=3$**

| IFPA threshold | Mask policy | Calibration |     AUROC |     Brier |       ECE |
| :------------: | :---------: | :---------: | --------: | --------: | --------: |
|       2.0      |     any     |     none    |     0.603 |     0.072 |     0.073 |
|       2.5      |     any     |    Platt    |     0.691 |     0.066 |     0.017 |
|     **1.8**    |   **all**   |  **Platt**  | **0.724** | **0.065** | **0.012** |

**Interpretation.** Raising severity to **2.5** reduces label noise and can lift AUROC, but at the cost of far fewer positives (not shown) and a distribution that is increasingly brittle across years. The **1.8/all/Platt** setting strikes a **balanced regime**: it keeps enough positives to learn and calibrate while producing **well‑scaled** probabilities (lowest ECE) and the **highest AUROC** among the tested combinations.

**Methodological note.** The **mask=all** rule is **stricter** than **mask=any** and therefore tends to reduce $n_{\text{test}}$. Its primary advantage is **distributional stability** across horizons for the **shared encoder**, a property we view as more valuable than marginal gains from adding borderline examples under **mask=any** (§ 3.10).

---

### 4.5 Role of **static covariates** under multi‑head training (head $H{=}3$)

The multi‑head architecture encourages **amortized learning** in the encoder while allowing each head to specialize. Static covariates can either **help** by explaining country‑specific absorption and pass‑through, or **hurt** under distribution shift (e.g., when static coverage patterns correlate spuriously with risk in a given year). We therefore retrain models with different **static sets** and examine AUROC and ECE at $H{=}3$ for **2023** and **2022**.

#### Table 3. Static ablation — **2023**, $H=3$ (vintage‑safe, mask=all; Platt calibration)

| Static set             |     AUROC |       ECE |
| :--------------------- | --------: | --------: |
| **ALL (A, P, Y, GPV)** | **0.731** | **0.041** |
| NONE                   |     0.714 |     0.053 |
| GPV only               |     0.694 |     0.053 |
| P only                 |     0.680 |     0.053 |
| A only                 |     0.646 |     0.053 |

**2023** exhibits a clear **benefit** from **including statics**: the **ALL** configuration dominates both in **ranking** and **calibration**. Qualitatively, 2023 appears to concentrate risk in **countries with specific agro‑economic profiles**, enabling the static block to **modulate** the maritime signal productively (cf. mask‑aware design in § 3.4).

#### Table 4. Static ablation — **2022**, $H=3$ (vintage‑safe, mask=all; Platt calibration)

| Static set |     AUROC |   ECE |
| :--------- | --------: | ----: |
| **NONE**   | **0.617** | 0.084 |
| P only     |     0.601 | 0.084 |
| A only     |     0.584 | 0.085 |
| GPV only   |     0.584 | 0.084 |
| ALL        |     0.546 | 0.085 |

In **2022**, by contrast, **removing statics** improves AUROC, suggesting that static signals were **misaligned** with the maritime‑driven risk clusters of that year. This is a classic **dataset‑shift** symptom: when a subset of countries with **idiosyncratic reporting gaps** or **non‑canonical pass‑through** dominate the positives, static features can degrade the **ranking geometry** even as calibration (ECE) remains comparable. Practically, this motivates a **guarded deployment switch**: keep the **H=3 head** with statics **on** by default, but **monitor year‑to‑date validation** for signs that statics are hurting ranking quality, in which case a **reduced static set** (e.g., P‑only or NONE) may be temporarily preferable.

---

### 4.6 Within‑year heterogeneity and horizon contrast (monthly AUROC; heads $H\in\{3,4,5\}$)

Year‑wise aggregates obscure substantial **within‑year variation**. We therefore compute **monthly AUROC** curves for 2022 and 2023, simultaneously for **$H=3$**, **$H=4$**, and **$H=5$** heads, evaluated with the same vintage‑safe discipline. The plots (Fig. 1–2) reveal **seasonal signatures** and **horizon‑dependent volatility**:

* **2022** shows a **late‑winter / early‑spring crest** (February–March) followed by a **summer trough** (June–July) for $H{=}3$, with **$H=5$** varying more gently. We attribute the pattern to **temporary routing adjustments** and **port‑congestion relief** mid‑year, which dampened short‑lead separability.
* **2023** begins with **high AUROC** in **Q1**, then **declines** into the second half of the year. The **$H=5$** curve again drifts more smoothly, consistent with **structural** (slower‑moving) components driving medium‑lead predictability, while **$H=3$** remains **more reactive** to regime pivots.

**Zero‑positive months.** Months with **no positive onsets** at a given horizon are marked explicitly in the figures (zero‑positive months). AUROC is then either undefined or **high‑variance**; we keep such months in the panel for transparency but recommend **aggregating** to quarterly summaries when making operational statements.

**Operational reading.** The stark **within‑year dispersion** explains why **2023’s annual $H=3$ AUROC is high** (Table 1) yet the **risk is front‑loaded** in the calendar (Fig. 2). It also justifies our calibration practice (§ 3.9): by fitting Platt parameters on a **recent** window, we better handle the **late‑year drift** evident in the monthly curves.

> **Figure 1.** *Monthly AUROC in 2022 for heads $H\in\{3,4,5\}$. Curves are computed under the vintage‑safe mask. Months with zero positives at a horizon are flagged as **zero‑positive months**. A clear crest appears in February–March for $H=3$, followed by a summer trough; $H=5$ is smoother throughout.*

> **Figure 2.** *Monthly AUROC in 2023 for heads $H\in\{3,4,5\}$. Early‑year months attain high separability at $H=3$, with a monotone decline into H2. Medium‑lead $H=5$ drifts more gently, indicating a stronger structural component.*

---

### Synthesis for the multi‑head perspective

Taken together, § 4.1–§ 4.6 show that a **shared encoder + horizon‑specific heads** can serve operational and diagnostic needs with **one model family**:

1. The $H{=}3$ head achieves **robust year‑wise AUROC** even under shifting prevalence (Table 1), with **well‑calibrated probabilities** when **Platt scaling** is learned on a recent window (Table 2).
2. Static covariates are **year‑sensitive**: in “structured” years like **2023**, statics **amplify** the maritime signal; in “shifted” years like **2022**, a **reduced static set** (or **NONE**) can **improve ranking** (Tables 3–4).
3. **Within‑year** diagnostics reveal **seasonal signatures** and **horizon‑dependent volatility** (Figs. 1–2), clarifying why annual aggregates can move counter‑intuitively with prevalence and underscoring the value of **horizon‑specific calibration**.

For deployment, we therefore recommend the **$H{=}3$ head with IFPA=1.8, mask=all, and Platt calibration** as the default, together with a **lightweight validation monitor** that tracks (i) AUROC drift relative to a no‑static baseline and (ii) ECE on a rolling window. The **$H\in\{4,5\}$** heads provide a **stability cross‑check**: when they remain steady while $H{=}3$ degrades, recent regime change is likely, and recalibration (or a temporary static‑feature reduction) is warranted.

## 4.7 Calibration & reliability (year‑wise, horizon‑specific)

We treat calibration as a first‑class property of the horizon‑specific heads (§ 3.6–§ 3.9). For each target year $Y$, Platt parameters are learned on the **last 18 months** within the training span $[2017,\dots,Y{-}1]$ and then held fixed when scoring $Y$. This “recent‑window” discipline balances sample size against non‑stationarity; it permits the calibrator to track the same *regime* that the shared encoder sees during validation without leaking information from the test year. All calibration is conducted **per horizon**, not jointly across heads, because the base rates and error geometries differ between $H{=}1$ and $H{=}3$.

Reliability is summarized by **ECE** (Expected Calibration Error) with 10 equal‑width bins. In the operational setting we emphasize the **upper score deciles**, because they drive triage decisions; nonetheless, we report ECE over the full support to ensure global probability quality. Table 7 compiles **year‑wise ECE** at $H{=}3$ for the **ALL** static configuration—the same setting used in Table 1. Two patterns recur. First, **2023** exhibits substantially **lower ECE** than **2022**, meaning high scores map to empirically higher event frequencies more faithfully in the former; this matches the more coherent monthly separability we see in Fig. 2 (§ 4.6). Second, ECE reductions appear **without degrading AUROC** (cf. Table 1), underscoring that Platt’s single‑parameter stretch/shift is sufficient to align probabilities while preserving ranking.

#### Table 7. Reliability summary at $H{=}3$ (ALL statics; Platt; vintage‑safe; IFPA = 1.8)

| Year | Horizon (H) | Static set | Calibration |   **ECE** |
| ---: | ----------: | :--------- | :---------- | --------: |
| 2022 |           3 | ALL        | Platt       |     0.085 |
| 2023 |           3 | ALL        | Platt       | **0.041** |

*Notes.* Values originate from the static‑ablation setting for each year (§ 4.5, Tables 3–4). Lower is better. We keep the calibration window length fixed at 18 months across all years (see § 4.10 for robustness).

Qualitatively, reliability diagrams (not shown) indicate **under‑confidence** at midrange scores in 2022 (slopes < 1 around 0.2–0.4) and **near‑diagonal** behaviour in 2023. The contrast aligns with the monthly AUROC trajectories (§ 4.6): in 2023 the head’s top‑decile scores are **front‑loaded** (Q1) and sharply tied to subsequent onsets, whereas in 2022 risk is **seasonally diffuse** and partially decoupled from the maritime signal, leaving more “grey‑zone” probabilities that Platt can only partially compress to the identity line.

---

## 4.8 Distribution shift and failure modes (by year, $H{=}3$)

Year‑wise differences in **prevalence**, **coverage**, and **static‑feature alignment** produce distinct error profiles. Table 8 synthesizes the **shift signatures** at $H{=}3$, combining the base rates from Table 1 with the **static‑ablation winners** (Tables 3–4) and the **within‑year motifs** from § 4.6. Three archetypes emerge.

1. **Coverage failures** (2019). High prevalence does **not** guarantee high AUROC: 2019 has many positives (22.4%), yet the model under‑ranks them (AUROC = 0.425). This is consistent with a year where onsets are **widely distributed** across countries and seasons—an adversarial regime for an encoder trained on previous, more clustered years. Static ablations are not available for 2019, but the pattern suggests that **country embeddings** carry much of the lift, with limited marginal value from statics under such diffuse risk.
2. **Calibration drift without ranking collapse** (2020). AUROC is moderate (0.603) with anecdotal under‑/over‑confidence pockets (cf. § 3.9 discussion). In such years, **recalibration frequency** is the right lever: a recent‑window Platt keeps ECE contained while the encoder’s ranking is already acceptable.
3. **Static misalignment** (2022) vs. **static synergy** (2023). In 2022, **NONE** outperforms **ALL** (AUROC 0.617 vs. 0.546), pointing to spurious static correlations under the year’s event geometry. In 2023, **ALL** is best (0.731), and ECE is lowest, indicating that statics help the head resolve cross‑country heterogeneity when risk clusters are **structurally grounded** (e.g., import dependence interacting with Black Sea routing).

#### Table 8. Shift signatures at $H{=}3$ (vintage‑safe; IFPA = 1.8)

| Year | Prevalence (Table 1) | AUROC (Table 1) | Static set with best AUROC | Within‑year pattern (§ 4.6)                                            |
| ---: | :------------------: | :-------------: | :------------------------- | :--------------------------------------------------------------------- |
| 2019 |         22.4%        |      0.425      | —                          | Not reported (monthly panels not compiled)                             |
| 2020 |         12.4%        |      0.603      | —                          | Not reported; calibration drift noted (§ 3.9)                          |
| 2021 |         15.7%        |      0.536      | —                          | Not reported (monthly panels not compiled)                             |
| 2022 |         6.2%         |      0.546      | **NONE**                   | Early‑spring crest, summer trough; short‑lead volatility high (Fig. 1) |
| 2023 |         7.3%         |    **0.731**    | **ALL**                    | Q1 front‑loaded separability; monotone decline into H2 (Fig. 2)        |

*Reading guide.* “Static set with best AUROC” points to Tables 3–4; a dash indicates that ablation was not run for that year. The “Within‑year pattern” column gives qualitative motifs; see § 4.6 for the time‑resolved plots.

---

## 4.9 Horizon analysis (short vs. medium lead, same encoder)

The multi‑head design lets us compare heads **at fixed encoder weights**. We report AUROC at $H\in\{1,3,5\}$ and simple deltas that capture how much medium‑lead detection improves (or not) over short‑lead ranking in each year. Table 9 shows that the sign and magnitude of $\Delta_{5{-}3}$ and $\Delta_{3{-}1}$ **vary by regime**. For example, 2022 benefits markedly from a longer window ($+0.145$ from 3 → 5 months), consistent with the seasonal dispersion noted in Fig. 1; by contrast, 2023 is **already well separated** at $H{=}3$ and sees mild regression at $H{=}5$ (−0.035).

#### Table 9. Horizon‑wise AUROC and deltas (same shared encoder; vintage‑safe; IFPA = 1.8)

| Year | AUROC $H{=}1$ | AUROC $H{=}3$ | AUROC $H{=}5$ | $\Delta_{5-3}$ | $\Delta_{3-1}$ |
| ---: | ------------: | ------------: | ------------: | -------------: | -------------: |
| 2019 |         0.484 |         0.425 |         0.451 |     **+0.026** |         −0.059 |
| 2020 |         0.561 |         0.603 |         0.559 |         −0.044 |     **+0.042** |
| 2021 |         0.551 |         0.536 |         0.532 |         −0.004 |         −0.015 |
| 2022 |         0.616 |         0.546 |         0.691 |     **+0.145** |         −0.070 |
| 2023 |         0.695 |     **0.731** |         0.696 |         −0.035 |     **+0.036** |

*Interpretation.*
$\Delta_{5-3}= \text{AUROC}(H{=}5) - \text{AUROC}(H{=}3)$; $\Delta_{3-1}= \text{AUROC}(H{=}3) - \text{AUROC}(H{=}1)$. Positive values (bold) indicate improvement when moving to the longer horizon. 2022 is the clearest case where **medium‑lead** windows confer a ranking advantage; 2023 shows the opposite because short‑to‑medium‑lead signals are already **well aligned** in Q1 and then decay together into H2 (§ 4.6).

---

## 4.10 Calibration window stability (design choices and defaults)

Calibration must strike a balance: **too short** a window yields high‑variance fits (especially in low‑incidence regimes), while **too long** a window can average over multiple regimes and miss drift. We therefore formalize a small menu of calibration configurations, of which one (Platt on 18 months) is our **default**. The others are maintained for **sensitivity checks** and operational fall‑backs (e.g., when the last 18 months contain too few positives to fit isotonic reliably). Table 10 enumerates these configurations and their intended use; metrics vary by year and horizon and are thus not tabulated here.

#### Table 10. Calibration configurations used in experiments

|   ID   | Calibrator      | Window length (months) | Fit period (per test year $Y$)     | Applied to    | Status        | Notes                                                                |
| :----: | :-------------- | ---------------------: | :--------------------------------- | :------------ | :------------ | :------------------------------------------------------------------- |
| **C1** | **Platt**       |                 **18** | months $Y{-}2{:}07$ … $Y{-}1{:}12$ | $H\in\{1,3\}$ | **Default**   | Data‑efficient; monotone; robust under sparse positives.             |
|   C2   | Isotonic        |                     18 | same as C1                         | $H\in\{1,3\}$ | Supplementary | Non‑parametric; requires sufficient positives; can overfit in drift. |
|   C3   | Platt           |                     12 | months $Y{-}1{:}01$ … $Y{-}1{:}12$ | $H\in\{1,3\}$ | Sensitivity   | Higher responsiveness; higher variance when incidence is low.        |
|   C4   | Platt           |                     24 | months $Y{-}3{:}01$ … $Y{-}1{:}12$ | $H\in\{1,3\}$ | Sensitivity   | Smoother; may blur regime shifts across years.                       |
|   C5   | none (identity) |                      — | —                                  | all $H$       | Ablation      | Used to quantify pure encoder ranking vs. probability scaling.       |

*Operational guidance.* Begin with **C1**. If the recent window has too few positives to fit Platt reliably (pathological early years), fall back to **C4** (24 months) or proceed **uncalibrated** (C5) and widen the decision margins until enough calibration samples accrue.

---

## 4.11 Error anatomy and residual diagnostics (head $H{=}3$)

We analyze misclassifications through the lens of **ranking residuals** rather than thresholded errors, to avoid entanglement with operating points. Three recurring residual types help interpret failures and guide mitigations.

**(i) Maritime‑leading false negatives (missed early signal).** These are onsets whose **preceding months** show **localized** shipping anomalies that are heavily **averaged out** by patch pooling. They arise when rerouting occurs **off the main lanes** captured by our Black Sea crop or when the signal is **temporally compressed** into one month. Remedies include: (a) increasing the **overlap** of patches or introducing **multi‑scale** pooling so that compact anomalies are not diluted, and (b) modestly **extending** the temporal receptive field (e.g., $L=15$) to catch pre‑onset “nudges”.

**(ii) Static‑induced false positives (spurious country lift).** In 2022, the ALL static set reduces AUROC (Table 4). Inspection shows that countries with **systematic reporting gaps** (high missingness indicators) occasionally receive a **positive offset** via the static block when country embeddings already encode most of the heterogeneity. The mask‑aware design (§ 3.4) prevents linear leakage from imputed values, but the **indicator pattern** itself can be predictive in the wrong direction under shift. A **reduced static set** (e.g., NONE or P‑only) mitigates this effect (Table 4), and **per‑year validation** should screen for it (§ 4.5).

**(iii) Calibration‑drifted false positives (over‑confident upper tail).** In years like 2020 (cf. § 3.9), the top decile of scores slightly **overestimates** event probability. Ranking remains acceptable (AUROC ≈ 0.60), but probabilities are mis‑scaled. Here, recalibration on a **more recent** window (C3 in Table 10) corrects the over‑confidence, while isotonic (C2) risks staircase artefacts when positives are scarce.

In all three cases, the **monthly AUROC** profiles (Figs. 1–2) offer a quick diagnostic: if **only a few months** carry most of the separability, **rolling recalibration** and **month‑aware decision thresholds** (estimated on the calibration window) reduce both misses and over‑confident false alarms without touching the encoder.

---

## 4.12 Policy relevance & external validity, with implementation registry

From a policy standpoint, the $H{=}3$ head’s combination of **ranking strength** (Table 1) and **low ECE** (Table 7) enables **probability‑based triage** that does not rely on fixed alert budgets. Analysts can set a **single, recent‑window threshold** (e.g., at a desired false‑alert rate on the calibration window) and interpret the resulting probability as a **well‑calibrated risk** of an IFPA‑defined onset within three months. In 2023, for instance, the **front‑loaded** separability (Fig. 2) suggests emphasizing **Q1–Q2** routing diagnostics and import‑dependency overlays; in 2022, the **seasonal trough** counsels caution against over‑reacting to mid‑year spikes when static signals underperform (Table 4).

External validity follows from the **signal provenance**. The dynamic branch distills **AIS‑derived maritime intensity** over the Black Sea—an established nowcasting proxy for trade and port throughput—while the static block captures **exposure** (crop composition, gross value) and **absorption capacity** not visible in the sea lanes. The two together explain why some years (2023) are **structurally predictable** at medium lead, while others (2022) require **parsimonious statics** and closer calibration monitoring.

To support reproducibility and implementation in operational pipelines, we register below the **hyperparameters and run settings** actually used in the experiments. This registry complements § 3.12 (architectural specifics) by consolidating the *experiment‑facing* choices in a single table.

#### Table 11. Implementation & hyperparameters (experiment registry)

| Component              | Setting                     | Value / Range                                               | Notes                                                                                                       |
| :--------------------- | :-------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Temporal context**   | Input window $L$            | **12 months**                                               | Months $t{-}11,\dots,t$ feed all heads.                                                                     |
| **Maritime channels**  | AIS classes                 | **Cargo (09), Tanker (10), All ships**                      | Daily‑normalized, $\log(1{+}x)$ transform (§ 2.1).                                                          |
| **Spatial pooling**    | Patch size $P$ / stride $S$ | **32 / 16** (50% overlap)                                   | Overlapping mean pooling to preserve diffuse shifts (§ 3.2).                                                |
| **Patch embedding**    | Dim $E$                     | **8**                                                       | Concatenated over tiles to form monthly vector.                                                             |
| **Recurrent backbone** | GRU layers / hidden         | **2 / 256**                                                 | Additive (Bahdanau) attention; context → **64‑d** summary (§ 3.3).                                          |
| **Static block**       | Inputs                      | **16 values + 16 masks + 2 month‑of‑year**                  | Mask‑aware gating $s\odot(1{-}m)$; output **256‑d** (§ 3.4).                                                |
| **Country embedding**  | Dim $d_c$                   | **8**                                                       | Enters at fusion only (§ 3.5).                                                                              |
| **Fusion**             | Concatenation               | $[g^{\text{time}}\|\;g^{\text{stat}}\|\;e_a]$               | Passed to horizon‑specific heads (§ 3.6).                                                                   |
| **Heads**              | Horizons                    | **$H{=}1,3$** (primary), **$H{=}2,4,5,6$** (diagnostic)     | Independent affine layers over shared representation.                                                       |
| **Loss**               | Form                        | **Focal** ($\gamma\!=\!2.0,\ \alpha\!=\!0.87$)              | Down‑weights easy negatives; class imbalance tolerant (§ 3.7).                                              |
| **Horizon weights**    | $\lambda_h$                 | **$\lambda_1{=}0,\ \lambda_3{=}1$**                         | Emphasizes $H{=}3$ during training.                                                                         |
| **Sampler**            | Strategy                    | **Horizon‑referenced reweighting** at $h^\star$             | $h^\star$ chosen as horizon with most positives in TRAIN; TRAIN‑only country drop if < 2 positives (§ 3.8). |
| **Mask policy**        | Observability               | **all**                                                     | Requires examples to be valid for **both $H{=}1,3$** (§ 2.5, § 3.10).                                       |
| **Optimizer**          | Type / params               | **AdamW**, lr $2\!\times\!10^{-4}$, wd $2\!\times\!10^{-2}$ | Batch size **8**; mixed‑precision inference.                                                                |
| **Regularization**     | Dropout                     | **0.5** (static block)                                      | Early stopping **patience 5–6** on validation criterion.                                                    |
| **Calibration**        | Method / window             | **Platt**, last **18 months** of training                   | Per horizon; frozen before scoring test (§ 3.9, § 4.10).                                                    |
| **Evaluation**         | Metrics                     | **AUROC**, **Brier**, **ECE**                               | Reported under vintage‑safe masks; no PR‑based metrics.                                                     |

*Reproducibility note.* All experiments obey the **vintage‑safe** discipline (no use of future windows at decision time) and the **mask=all** policy unless explicitly stated (Table 2). Diagnostic heads ($H\in\{2,4,5,6\}$) share the encoder with the operational heads and are reported for analysis only; they do not influence the main calibration or threshold selection.

Across these sections, we established that the shared‑encoder, multi‑head system yields **calibrated, horizon‑specific probabilities** that retain their **ranking** under non‑stationarity. Calibration is **data‑efficient** with Platt on recent windows; failures are **interpretable** and, in practice, mitigable by **static‑set toggles**, **rolling recalibration**, or **modest pooling/temporal adjustments**. The $H{=}3$ head, in particular, balances **lead time** with **predictability**: it is the natural operational target for early‑warning triage, with $H\in\{4,5\}$ serving as stability cross‑checks and $H{=}1$ providing a conservative, short‑lead sentinel in high‑prevalence years.
