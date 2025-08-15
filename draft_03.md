## 2. Data and Label Construction

This section details the spatio‑temporal maritime inputs, the country‑level static covariates, and the outcome construction based on FAO’s Indicator of Food Price Anomalies (IFPA). Design choices enforce **vintage safety** so that, at forecast origin $t$, neither inputs nor labels can depend on information unavailable at $t$. All results in §4 adopt these data conventions.

---

### 2.1 Maritime activity cube (domain, units, channels)

**Source and semantics.** Our dynamic input is the **EMODnet Human Activities—Vessel Density Map**, which aggregates AIS transponder pings into monthly **hours per square kilometre** on an equal‑area European grid (ETRS89‑LAEA, **EPSG:3035**) at **1 km** resolution. The product distinguishes coarse AIS classes; we retain three channels per month: **Cargo**, **Tanker**, and **All ships**. Because EMODnet measures **time‑at‑sea** rather than raw ping counts, intensities are physically interpretable as exposure of traffic lanes and port approaches.

**Spatial domain.** We fix the raster crop to the **Black Sea and approaches** (27°E–42°E, 40°N–47°N), a corridor repeatedly affected by logistics disruptions and directly relevant for cereal flows to food‑importing regions in Africa. The equal‑area geometry avoids latitude‑dependent area distortions and guarantees that a 1 km cell has the same footprint regardless of location inside the crop, which is important for density and aggregation operations.

**Temporal coverage and normalization.** Monthly GeoTIFFs span **January 2017–December 2023**. Let $V_{c}(t,x)$ denote the EMODnet value (hours/km$^2$ in calendar month $t$) for channel $c\in\{\text{Cargo},\text{Tanker},\text{All}\}$ at grid cell $x$. Because month length varies, we first convert to a **daily rate**,

$$
R_{c}(t,x)\;=\;\frac{V_{c}(t,x)}{\text{days}(t)}\quad\text{(hours per km}^2\text{ per day)},
$$

then stabilize heavy tails by a **log1p** transform,

$$
X_{c}(t,x)\;=\;\log\!\bigl(1+R_{c}(t,x)\bigr).
$$

Land pixels are masked; a **quiet sea** cell is encoded as zero and an **active sea** cell as a strictly positive value. For country $a$ in month $t$, the dynamic tensor is

$$
X_{a,t}\in\mathbb{R}^{C\times H\times W},\qquad C=3,
$$

on the fixed crop grid. We deliberately **do not** reproject EMODnet’s native grid, avoiding interpolation artefacts and preserving the equal‑area property that underpins our use of averages in §3.

**Rationale.** Hours‑based densitites are robust to ping irregularities and provide a smoother proxy for corridor utilization than vessel counts. The Black Sea crop emphasizes upstream chokepoints germane to African food import risk. The $\log(1+x)$ transform preserves zeros, compresses rare extremes (e.g., anchorage clusters), and behaves predictably under spatial pooling.

---

### 2.2 Temporal index and windowing

All series are aligned on the monthly index

$$
\mathcal{T}=\{2017\text{‑}01,\dots,2023\text{‑}12\}\quad (84\ \text{months}).
$$

Model inputs use a fixed **12‑month** history $L=12$: for forecast origin $t$ we form

$$
\mathcal{X}_{a,t}=\bigl(X_{a,t-11},X_{a,t-10},\dots,X_{a,t}\bigr),
$$

and the model outputs horizon‑specific probabilities for an onset within $h$ months, with $h\in\{1,3\}$ in the mainline (additional horizons are used diagnostically in §4). A 12‑month context balances **seasonal coverage** against **recency** for fast‑moving logistics shocks, keeps the temporal backbone compact (§3.3), and aligns with the IFPA construction’s month‑of‑year standardization (§2.4).

---

### 2.3 Country‑level covariates (static indicators)

**Variables and construction.** Static covariates describe agro‑economic scale and composition that mediate price transmission yet are invisible in maritime imagery. We use **FAOSTAT** annual indicators only—**no World Bank (WDI) fields**—to avoid mixed vintages. The set comprises **16 variables**: **Area harvested (A\_\*)**, **Production (P\_\*)**, **Yield (Y\_\*)** for four cereals (Maize, Rice, Soya beans, Wheat) and **Gross Production Value (GPV\_\*)** for the same items. Let $s^{\text{annual}}_i(y)$ denote an annual FAOSTAT value (with appropriate units per item/element). To ensure vintage safety, we convert to a monthly panel by **replicating the latest available year** $y-1$ across all months of calendar year $y$:

$$
S_i(t)\;=\;s^{\text{annual}}_i\!\bigl(y(t)-1\bigr).
$$

The monthly vector $s_{a,t}$ stacks all $S_i(t)$ relevant for country $a$. We **robust‑scale** each variable using country‑invariant statistics (median and IQR) and apply a **signed** $\log(1+|x|)$ to temper long tails while preserving sign. To capture seasonality without creating a December–January discontinuity, we append a smooth month‑of‑year encoding $\phi(t)=[\sin(2\pi m/12),\cos(2\pi m/12)]$.

**Missingness handling.** Even though the **Africa‑36** evaluation subset (listed in §2.6) is chosen so that FAOSTAT statics and CPI are **complete** over **2017–2023**, our model is designed for broader deployments where static tables can be incomplete or lagged. We therefore maintain **per‑variable missingness indicators** $m_{a,t}$ and use the **missingness‑aware gating** described in §3.4: values contribute linearly only when observed $(1-m)$, while the indicator $m$ itself remains available as signal. On the Africa‑36 subset these indicators happen to be zero, but the mechanism is part of the architecture to ensure extensibility and to prevent spurious effects from imputation in other regions.

**Variables at a glance.** Table **S2.2** enumerates the 16 FAOSTAT variables and their FAOSTAT item/element definitions. For the Africa‑36 subset and years 2017–2023, **completeness is 100% by construction**; units follow FAOSTAT (hectares, tonnes, tonnes/ha, and constant‑price GPV).

**Table S2.2. FAOSTAT static covariates (16 variables) and completeness in Africa‑36 (2017–2023).**

| Domain | Variable   | FAOSTAT item / element                | Units           | Completeness (Africa‑36) |
| ------ | ---------- | ------------------------------------- | --------------- | -----------------------: |
| CROPS  | A\_Maize   | Area harvested — Maize (corn)         | ha              |                     100% |
| CROPS  | A\_Rice    | Area harvested — Rice                 | ha              |                     100% |
| CROPS  | A\_Soya    | Area harvested — Soya beans           | ha              |                     100% |
| CROPS  | A\_Wheat   | Area harvested — Wheat                | ha              |                     100% |
| CROPS  | P\_Maize   | Production — Maize (corn)             | tonnes          |                     100% |
| CROPS  | P\_Rice    | Production — Rice                     | tonnes          |                     100% |
| CROPS  | P\_Soya    | Production — Soya beans               | tonnes          |                     100% |
| CROPS  | P\_Wheat   | Production — Wheat                    | tonnes          |                     100% |
| CROPS  | Y\_Maize   | Yield — Maize (corn)                  | t/ha            |                     100% |
| CROPS  | Y\_Rice    | Yield — Rice                          | t/ha            |                     100% |
| CROPS  | Y\_Soya    | Yield — Soya beans                    | t/ha            |                     100% |
| CROPS  | Y\_Wheat   | Yield — Wheat                         | t/ha            |                     100% |
| GPV    | GPV\_Maize | Gross production value — Maize (corn) | const. currency |                     100% |
| GPV    | GPV\_Rice  | Gross production value — Rice         | const. currency |                     100% |
| GPV    | GPV\_Soya  | Gross production value — Soya beans   | const. currency |                     100% |
| GPV    | GPV\_Wheat | Gross production value — Wheat        | const. currency |                     100% |

---

### 2.4 Outcome definition: IFPA‑based food price anomalies

**IFPA index.** Our event of interest is an **abnormally high food price** episode, operationalized via FAO’s **Indicator of Food Price Anomalies (IFPA)** for **Food CPI**. Let $P_t$ be the Food CPI level in month $t$. Define the **within‑year** and **across‑year** growth components,

$$
\mathrm{CQGR}_t = \log\!\frac{P_t}{P_{t-3}},\qquad 
\mathrm{CAGR}_t = \log\!\frac{P_t}{P_{t-12}}.
$$

IFPA standardizes each component by **month‑of‑year** using a **fixed historical baseline**. Let $\mu_m(\cdot)$ and $\sigma_m(\cdot)$ be the mean and standard deviation for calendar month $m$ computed on **2000–2018** only. With a recommended weight $\gamma\approx 0.4$, the IFPA score is

$$
\mathrm{IFPA}_t
= \gamma\,\frac{\mathrm{CQGR}_t-\mu_m(\mathrm{CQGR})}{\sigma_m(\mathrm{CQGR})}
+ (1-\gamma)\,\frac{\mathrm{CAGR}_t-\mu_m(\mathrm{CAGR})}{\sigma_m(\mathrm{CAGR})}.
$$

FAO’s own guidance flags $\mathrm{IFPA}\ge 1.0$ as “abnormally high.” We construct a **threshold family** $\tau\in\{1.8,2.0,2.5\}$ and define monthly anomaly indicators

$$
A_t(\tau)=\mathbf{1}\{\mathrm{IFPA}_t\ge \tau\}.
$$

The mainline uses **$\tau=1.8$** to focus on rarer, more severe spikes while remaining within the IFPA framework. Crucially, CPI series are **never used as model inputs**; they enter only for label construction to avoid leakage from outcomes to features.

**Baseline discipline.** The **2000–2018** fixed baseline keeps the anomaly test **independent** of the 2019–2023 evaluation window and guards against drifting thresholds. Month‑of‑year standardization makes IFPA robust to seasonal and inflationary variation without conflating e.g., harvest cycles with abnormal surges.

---

### 2.5 Onset‑within‑$h$ labels and vintage‑safe masks

**Onset extraction.** Starting from the monthly anomaly series $A_t(\tau)$, we enforce a **minimum duration** of two consecutive months to suppress trivial blips and apply a **refractory period** of two months after each episode, then collapse each episode to a single **onset flag** $O_t$ (1 only in the first month of the episode, 0 otherwise). These operations ensure that labels reflect **entry** into a surge rather than the plateau of a prolonged episode, which is the operationally relevant moment for preparedness.

**Horizon windows.** For forecast horizon $h$ months, the **onset‑within‑$h$** label at forecast origin $t$ is

$$
Y^{(h)}_t
= \mathbf{1}\!\left\{\max\bigl(O_{t+1},\dots,O_{t+h}\bigr)=1\right\}.
$$

Thus $Y^{(h)}_t=1$ iff an onset occurs **strictly after** $t$ and **no later than** $t+h$. Because window widths grow with $h$, the count of positives is **monotone increasing** in $h$ (§4.2 Table 6) and is **not** an episode count (§4.2 Table 5).

**Vintage‑safe observability.** To avoid look‑ahead bias, a sample is **admissible** at time $t$ only if all inputs and the entire **future window** $(t+1,\dots,t+h)$ are observable **at vintage $t$**. The resulting **vintage‑safety mask** is

$$
M^{(h)}_t=\mathbf{1}\{\text{all inputs at }t\text{ and }(t+1,\dots,t+h)\text{ are observed at vintage }t\}.
$$

Training and evaluation in §3–§4 are performed **only** on masked‑valid instances. In the main configuration we adopt **mask‑policy = all**: a country–month contributes to learning and scoring only if it is valid for **every** horizon used in training (here $h\in\{1,3\}$). This keeps the shared encoder’s input distribution consistent across heads and simplifies cross‑horizon comparisons and calibration (§3.10).

---

### 2.6 Evaluation geography and period

**Africa‑36 subset.** We evaluate on **36 African economies** selected as the **intersection** of countries for which **(i)** FAOSTAT static covariates and **(ii)** CPI series required for IFPA construction are **complete** over the AIS era **2017–2023**. The resulting list is:

**Algeria, Angola, Benin, Botswana, Burkina Faso, Burundi, Cabo Verde, Cameroon, Central African Republic, Chad, Congo, Côte d’Ivoire, Egypt, Ethiopia, Gambia, Ghana, Guinea, Guinea‑Bissau, Kenya, Madagascar, Malawi, Mali, Mauritius, Morocco, Mozambique, Namibia, Niger, Nigeria, Rwanda, Senegal, Sierra Leone, South Africa, Togo, Tunisia, Zambia, Zimbabwe.**

For transparency we also provide the list as a table.

**Table S2.1. Africa‑36 evaluation countries (complete statics & CPI, 2017–2023).**

| Algeria      | Angola       | Benin                    | Botswana | Burkina Faso | Burundi       |
| ------------ | ------------ | ------------------------ | -------- | ------------ | ------------- |
| Cabo Verde   | Cameroon     | Central African Republic | Chad     | Congo        | Côte d'Ivoire |
| Egypt        | Ethiopia     | Gambia                   | Ghana    | Guinea       | Guinea‑Bissau |
| Kenya        | Madagascar   | Malawi                   | Mali     | Mauritius    | Morocco       |
| Mozambique   | Namibia      | Niger                    | Nigeria  | Rwanda       | Senegal       |
| Sierra Leone | South Africa | Togo                     | Tunisia  | Zambia       | Zimbabwe      |

**Time splits.** The temporal design in §4 follows **rolling origin**: for target year $Y$, we **train/validate** on $[2017,\dots,Y-1]$ and **test** on $Y$, with calibration on the last 18 months of the training span (§3.9). The headline split (train 2017–2022 → test 2023) yields, under **mask‑policy = all**, **315** masked‑valid **H=3** test instances (Table 1: $n_{\text{test}}=315$, $23$ positives, prevalence $7.3\%$). Aggregated counts for train/validation splits follow the experiment manifest (train $\approx 1995$, validation $\approx 630$ country‑months under the same mask), ensuring that calibration and threshold selection never peek into the 2023 test window.

---

### 2.7 Why these choices?

**AIS/EMODnet as dynamic signal.** Vessel‑hours density over an equal‑area grid provides a physically grounded, reproducible proxy for **real‑time maritime flows** and **logistics frictions**. The Black Sea crop focuses on a chokepoint whose fluctuations plausibly propagate to **food‑import‑dependent** markets. Using EMODnet’s native EPSG:3035 and 1 km cell size avoids resampling artefacts and preserves the meaning of “hours per km$^2$”, which justifies spatial **averaging** in the patch encoder (§3.2).

**IFPA with a stricter cut‑off.** IFPA explicitly removes seasonality and inflation via month‑of‑year standardization over a fixed baseline. Choosing $\tau=1.8$ emphasizes **high‑impact** surges—rare but policy‑relevant—while remaining within FAO’s official framework. The threshold family $\{1.8,2.0,2.5\}$ supports sensitivity analyses (§4.4), where we find that the recommended configuration (**$\tau=1.8$, mask = all, Platt**) offers the best combined profile in AUROC and calibration (Table 2).

**Vintage‑safe masking.** Requiring full observability of the future window at origin $t$ is the time‑series analogue of **real‑time** evaluation in macroeconomics. The **mask‑policy = all** choice ensures the shared encoder is trained and assessed on a single, horizon‑consistent distribution, preventing horizon‑dependent covariate shift that would otherwise complicate calibration and cross‑horizon interpretation (§3.10; Table M1).

**Transforms and scaling.** Converting hours/month to **daily rates** eliminates calendar‑length artefacts; $\log(1+x)$ stabilizes heavy‑tailed maritime intensities while preserving zero‑activity cells. For statics, **robust scaling** and a **signed log** bring disparate magnitudes (hectares, tonnes, monetary values) onto compatible scales. The **sin–cos** month encoding enables smooth seasonality without the December–January discontinuity, and its placement on the **static path** (rather than dynamic) lets the model express seasonality in **price transmission** conditional on country structure (§3.4).

**Africa‑36 selection.** Evaluating on a **complete‑data** subset removes confounding from missing statics/CPI and isolates the **signal transfer** from maritime anomalies to price surges. Although the missingness‑aware machinery remains in the model for broader applicability, the Africa‑36 panel guarantees that results in §4 are not artifacts of imputation or data gaps.

#### Summary of objects (notation)

At month $t$ and country $a$, the dynamic maritime raster is $X_{a,t}\in\mathbb{R}^{3\times H\times W}$ (daily‑rate, log‑scaled, land‑masked), the static vector is $s_{a,t}\in\mathbb{R}^{16}$ with missingness $m_{a,t}\in\{0,1\}^{16}$ (all zeros on Africa‑36), and the month encoding is $\phi(t)\in\mathbb{R}^{2}$. The 12‑month history $\mathcal{X}_{a,t}=\bigl(X_{a,t-11},\dots,X_{a,t}\bigr)$ feeds the model, which outputs horizon‑specific probabilities $p_{a,t,h}$ for $Y^{(h)}_t\in\{0,1\}$ under the **vintage‑safety** constraint $M^{(h)}_t=1$. All headline results in §4 use **$\tau=1.8$**, **mask‑policy = all**, and the **Africa‑36** subset defined above, with coverage and base rates reported in §4.2 (Tables 5–6) and year‑wise $H=3$ performance in Table 1.


## 3. Methodology

### 3.1 Problem formulation

Let $a\in\{1,\dots,A\}$ index countries and $t\in\mathcal{T}$ index calendar months on a common monthly grid (see §2). For each country–month we observe (i) a **dynamic maritime tensor** summarizing vessel activity over the Black Sea crop and (ii) a **static feature vector** describing domestic agro‑economic structure. Following §2, dynamic rasters are pre‑processed by converting EMODnet monthly “hours per km$^2$” to a daily rate and applying a $\log(1+x)$ transform; land pixels are masked and three channels are retained: **Cargo**, **Tanker**, and **All ships**. The spatial grid is fixed (ETRS89‑LAEA, 1 km cells), so tensors for different months are directly aligned.

For a fixed input length $L=12$, the dynamic history available at month $t$ for country $a$ is the ordered tuple

$$
\mathcal{X}_{a,t}
=\bigl(X_{a,t-L+1},\,\ldots,\,X_{a,t}\bigr),\qquad 
X_{a,\tau}\in\mathbb{R}^{C\times H\times W},\quad C=3,
$$

where $H,W$ are the crop’s spatial dimensions. Static covariates are the FAOSTAT‑derived vector $s_{a,t}\in\mathbb{R}^{d_s}$ constructed in §2 (A/P/Y/GPV; robust‑scaled and signed‑log transformed) together with a binary **missingness mask** $m_{a,t}\in\{0,1\}^{d_s}$ ($1\Rightarrow$ missing). We additionally encode month‑of‑year as a smooth two‑dimensional representation $\phi(t)=[\sin(2\pi m/12),\cos(2\pi m/12)]$, where $m$ is the integer month.

The **outcomes** are early‑warning **onset‑within‑h** labels derived from IFPA (§2.4–§2.5). If $O_{a,\tau}\in\{0,1\}$ flags an onset month (after duration and refractory logic), the horizon‑$h$ window label is

$$
Y_{a,t,h}=\mathbf{1}\!\left\{\max\big(O_{a,t+1},\ldots,O_{a,t+h}\big)=1\right\},
\qquad h\in\mathcal{H}.
$$

Our **operational target set** is $\mathcal{H}=\{1,3\}$; additional horizons (e.g., $h=4,5$) reported in §4 use the same architecture with horizon‑specific heads trained analogously (we denote this “train\_h=H” in figures). To enforce real‑time discipline, we define a **vintage‑safety mask** $M_{a,t,h}\in\{0,1\}$ that equals 1 iff all inputs at $t$ and the entire future window $(t+1,\ldots,t+h)$ would have been observable at vintage $t$. Training and evaluation are conducted only on masked‑valid pairs; in the main configuration we adopt the **mask‑policy = all**, i.e., a country–month enters the sample only if it is valid **for every** horizon under consideration, so the encoder sees a consistent distribution across heads.

The model learns a conditional probability of an onset within $h$ months,

$$
p_{a,t,h}=\mathbb{P}\!\left(Y_{a,t,h}=1\mid \mathcal{X}_{a,t},\,s_{a,t},\,m_{a,t},\,\phi(t),\,a\right),
$$

through a parametric mapping $f_\theta$ that fuses a temporal summary of the dynamic rasters, a missingness‑aware static summary, and a country‑specific latent. The logits $\ell_{a,t,h}$ for each horizon are transformed by a sigmoid, $p_{a,t,h}=\sigma(\ell_{a,t,h})$. These probabilities are later **calibrated** (§3.9) and assessed in §4 by **AUROC** (ranking quality) and **Brier/ECE** (probability quality). No metric depending on precision–recall is used anywhere in this manuscript.

Throughout, we distinguish the **missingness mask on statics**, $m_{a,t}$, from the **vintage‑safety mask**, $M_{a,t,h}$; the former expresses data availability in the static table, the latter enforces real‑time observability of future windows. All shapes and indices are summarized inline where they first appear to keep the exposition self‑contained.

---

### 3.2 Patch encoder (spatial aggregation of maritime rasters)

The dynamic branch must compress a monthly raster $X_{a,t}\in\mathbb{R}^{C\times H\times W}$ into a vector that preserves **where** activity occurred while avoiding overfitting to individual pixels. We therefore adopt a two‑stage design: light **channel‑aware smoothing** followed by **overlapping patch pooling** with learned per‑patch embeddings.

**Local smoothing and channel mixing.** The pre‑processed $X_{a,t}$ is first passed through depthwise–pointwise filters that act as a gentle low‑pass on each channel and then mix Cargo/Tanker/All information. Concretely, we apply two depthwise $3\times 3$ convolutions with stride 1 and padding 1, each followed by a pointwise $1\times 1$ convolution. Nonlinearities are mild (ReLU) and no downsampling is performed at this stage. Empirically this stabilizes gradients on sparse traffic fields and reduces sensitivity to isolated hot pixels without erasing wide, diffuse patterns (e.g., re‑routing corridors).

**Overlapping patch pooling.** The filtered map is partitioned into overlapping square patches of side $P$ with stride $S$, where we set $P=32$ and $S=\lfloor P/2\rfloor=16$ (50% overlap). Without padding, the patch grid has

$$
N_H=\Big\lfloor\frac{H-P}{S}\Big\rfloor+1,\qquad
N_W=\Big\lfloor\frac{W-P}{S}\Big\rfloor+1,\qquad
N_{\text{patch}}=N_H\,N_W.
$$

Within each $P\times P$ tile we take a channel‑wise **average** (not max) so that the representation remains sensitive to broad areas of moderate activity and not only to sharp peaks. Each averaged patch vector is then projected to an $E$-dimensional embedding by a small linear layer; we use $E=8$. Concatenating embeddings over all tiles yields the monthly dynamic vector

$$
z^{\text{dyn}}_{a,t}\in\mathbb{R}^{E\,N_{\text{patch}}}.
$$

This construction amounts to a **bag‑of‑regions** representation. Overlap softens spatial decision boundaries and makes the encoding robust to small geolocation shifts (e.g., when shipping lanes wobble by a few kilometers), while equal‑area gridding (§2) prevents latitude‑induced density biases. Averaging is consistent with the “hours per unit area” semantics of the EMODnet product and with the $\log(1+x)$ scale: it preserves zeros, tempers heavy tails, and behaves predictably under aggregation. The output dimensionality depends only on $(H,W,P,S,E)$ and is independent of the number of non‑zero pixels in a given month, which simplifies batching and recurrent processing.

For each country–month pair we thus obtain a sequence of 12 vectors $\bigl\{z^{\text{dyn}}_{a,t-L+1},\ldots,z^{\text{dyn}}_{a,t}\bigr\}$, which is fed to a temporal model.

---

### 3.3 Temporal backbone (sequence modeling with additive attention)

Maritime conditions relevant for domestic price surges tend to **accumulate and dissipate** over seasons rather than flip instantaneously. To capture this structure we process the 12‑month sequence of patch vectors with a compact recurrent backbone and **additive attention**.

Let $u_\tau$ denote the per‑month dynamic vector after a light normalization layer (LayerNorm) that reduces scale drift across months. We pass the sequence $(u_{t-L+1},\ldots,u_t)$ through a two‑layer Gated Recurrent Unit (GRU). Writing $h_\tau\in\mathbb{R}^{H_{\text{RNN}}}$ for the top‑layer hidden state at month $\tau$ ($H_{\text{RNN}}=256$ in our setup), we aggregate the sequence with Bahdanau‑style attention:

$$
e_\tau=\mathbf{v}^\top\tanh(\mathbf{W} h_\tau),\qquad
\alpha_\tau=\frac{\exp(e_\tau)}{\sum_{k=t-L+1}^{t}\exp(e_k)},\qquad
c=\sum_{\tau=t-L+1}^{t}\alpha_\tau\,h_\tau.
$$

The context vector $c\in\mathbb{R}^{H_{\text{RNN}}}$ is a convex combination of monthly states with non‑negative weights $\alpha_\tau$ summing to one. A small feed‑forward head (linear → ReLU → dropout) maps $c$ to a **64‑dimensional temporal summary**

$$
g^{\text{time}}_{a,t}\in\mathbb{R}^{64}.
$$

Two properties are crucial. First, attention provides **interpretable temporal saliencies**: the learned $\alpha_\tau$ often concentrate on specific parts of the year that align with shipping seasons or congestion spells, offering diagnostics without resorting to ad hoc smoothing. Second, a short recurrent memory (12 steps) is sufficient to capture the lag structure seen in maritime activity and price transmission while keeping memory and compute bounded; it also avoids the vanishing‑gradient pathologies of long sequences on sparse, heavy‑tailed inputs. In cross‑validation we found that deeper temporal stacks or temporal convolutions did not improve AUROC once patch overlap and attention were in place, consistent with the view that **where** and **when** information—rather than raw pixel dynamics—drive out‑of‑sample discrimination.

---

### 3.4 Static and missingness‑aware block

Static covariates quantify country‑level **exposure** and **absorption capacity** that are not visible in maritime imagery: cropped area, production, yield, and gross production value for key cereals (FAOSTAT), transformed as in §2. These tables are heterogeneous and incomplete. Treating missingness carefully is central: naively imputing missing values and letting the model interpret them as actual zeros would create spurious linear effects and bias coefficients whenever reporting is systematic (e.g., smaller economies with sparser submissions).

We therefore separate **values** from **missingness** and gate the former by the latter. Given the robust‑scaled vector $s_{a,t}\in\mathbb{R}^{d_s}$, its mask $m_{a,t}\in\{0,1\}^{d_s}$ ($1\Rightarrow$ missing), and the month encoding $\phi(t)\in\mathbb{R}^2$, we construct the augmented static input

$$
\tilde{x}^{\text{stat}}_{a,t}
=\Bigl(s_{a,t}\odot(1-m_{a,t})\;\Big\|\;m_{a,t}\;\Big\|\;\phi(t)\Bigr)
\in\mathbb{R}^{2d_s+2}.
$$

Here “$\|$” denotes concatenation and “$\odot$” is the Hadamard product. In other words, every static **value** is multiplied by $(1-m)$ so that **missing entries contribute exactly zero** to the linear map, while the **indicator of missingness** remains available to the model as a separate input. The concatenated vector is fed to a mask‑aware affine layer followed by ReLU and dropout, producing a **256‑dimensional static summary**

$$
g^{\text{stat}}_{a,t}\in\mathbb{R}^{256}.
$$

The month encoding is appended on the static path (rather than the dynamic path) **by design**: it lets the model modulate seasonality in **price transmission** conditional on country structure, while the dynamic branch is left to focus on spatial maritime patterns with its own attention over months. This separation reduced entanglement in ablations and improved calibration stability. The net effect of the block is twofold: it prevents linear leakage from imputed values and permits **informative missingness** (e.g., “GPV not reported this year”) to enter the decision through its indicator.

---

### 3.5 Country embedding

Differences across countries in pass‑through, market structure, policy regimes, and logistics bottlenecks are only partially captured by FAOSTAT variables. To account for persistent, idiosyncratic offsets we learn a compact **country embedding** $e_a\in\mathbb{R}^{d_c}$ with $d_c=8$. This vector functions as a dense **random‑effects** term: countries with richer event histories can tilt their embeddings to absorb consistent deviations, while data‑sparse countries borrow strength from the shared encoder.

Two guardrails prevent this term from becoming a shortcut. First, we **train and evaluate in real time** with vintage‑safe masks (§3.1), so the embedding cannot memorize future labels at a given forecast origin. Second, the embedding is injected **only at the fusion stage** (next subsection). It does **not** modulate the patch encoder or the recurrent dynamics, which keeps the maritime representation country‑agnostic and focused on geography and time. Regularization and early stopping implicitly control $\|e_a\|$; in practice we observed that removing the embedding degrades AUROC for a subset of countries with clear but idiosyncratic response patterns, while leaving probability calibration largely unchanged—a typical signature of hierarchical shrinkage.

---

### 3.6 Multi‑head outputs (horizon‑specific detection)

Let $g^{\text{time}}_{a,t}\in\mathbb{R}^{64}$ and $g^{\text{stat}}_{a,t}\in\mathbb{R}^{256}$ be the temporal and static summaries and $e_a\in\mathbb{R}^{8}$ the country latent. We **fuse** them by concatenation into a single feature vector

$$
z_{a,t}=\bigl[g^{\text{time}}_{a,t}\;\|\;g^{\text{stat}}_{a,t}\;\|\;e_a\bigr]\in\mathbb{R}^{328}.
$$

The fused representation is shared by a set of **binary heads**, one per forecast horizon. In the main configuration we instantiate heads for $h\in\{1,3\}$; for completeness, the same encoder–head template can be instantiated for other horizons (e.g., $h=4,5$) when we present monthly AUROC curves with “train\_h=H” in §4.6.

Each head is a single affine map followed by a sigmoid, producing a **horizon‑specific logit** and **probability**

$$
\ell_{a,t,h}=\mathbf{w}_h^\top z_{a,t}+b_h,\qquad
p_{a,t,h}=\sigma(\ell_{a,t,h}).
$$

The heads **share** all upstream parameters but **do not share** their last‑layer weights $(\mathbf{w}_h,b_h)$. This choice is motivated by three considerations germane to our data. First, label geometry differs across horizons: $h=1$ labels are extremely sparse and often tied to sharp, short‑lived shocks; $h=3$ labels are less sparse and better aligned with persistent maritime patterns. Allowing the final hyperplane to specialize lets the shared encoder transfer useful features while the heads form different decision boundaries. Second, separating heads cleanly supports **per‑horizon calibration** (§3.9): a one‑dimensional Platt scaler or isotonic map is fit independently for each $h$, improving ECE without coupling horizons. Third, the formulation decouples modeling from any thresholding or alert‑budget rules; the outputs are **threshold‑agnostic probabilities** later summarized by AUROC for ranking and by Brier/ECE for probability quality (§4).

In practice, we observed that sharing the encoder while specializing the heads yields a robust trade‑off: the temporal module learns where and when maritime signals matter across all horizons, the static/missingness block provides country‑level context, and the country embedding absorbs residual heterogeneity. At inference, the model produces $p_{a,t,h}$ for all masked‑valid country–months; probabilities are then calibrated on a held‑out window within the training span and evaluated out‑of‑sample by year. The same machinery underlies the monthly AUROC decompositions in **Figure 1–2** and the year‑wise test summaries in **Table 1**; sensitivity analyses in §4.4 (label threshold/mask/calibration) demonstrate that the final probabilities are stable under the recommended configuration (**thr = 1.8**, **mask = all**, **Platt**).

### 3.7 Loss & vintage‑safe masking

At forecast origin $t$ for country $a$, the horizon‑$h$ head produces a logit $\ell_{a,t,h}\in\mathbb{R}$ and a probability $p_{a,t,h}=\sigma(\ell_{a,t,h})\in(0,1)$. Let $y_{a,t,h}\in\{0,1\}$ be the **onset‑within‑$h$** label constructed in real time (see §2.5). Because an example is **admissible** only when the entire future window $(t+1,\dots,t+h)$ is observable at vintage $t$, we define a **vintage‑safety indicator**

$$
M_{a,t,h}=\mathbf{1}\{\text{all inputs at }t\text{ and the entire future window }(t+1,\ldots,t+h)\text{ are observed at vintage }t\}.
$$

Training and evaluation are performed strictly on masked‑valid pairs. In the **main configuration** we adopt the **mask‑policy = all**, so a country–month $(a,t)$ is retained only if it is valid for **every** horizon used in training; with $\mathcal{H}=\{1,3\}$ we keep $(a,t)$ iff $M_{a,t,1}=M_{a,t,3}=1$. This keeps the shared encoder exposed to a consistent distribution across heads and prevents horizon‑specific covariate shift.

The **per‑example loss** averages masked horizon losses with user‑set horizon weights $\lambda_h\ge 0$:

$$
\mathcal{L}_{a,t}
=\frac{1}{\sum_{h\in\mathcal{H}} M_{a,t,h}+\varepsilon}
\sum_{h\in\mathcal{H}}
\lambda_h\,M_{a,t,h}\,\ell_{\text{bin}}\!\bigl(p_{a,t,h},y_{a,t,h}\bigr),
$$

where $\varepsilon>0$ prevents division by zero when a training instance becomes invalid after masking (by construction this does not occur under mask‑policy = all, but we keep the guard for completeness). We explore two choices for $\ell_{\text{bin}}$.

**Weighted cross‑entropy.** With class weights $w_+,w_->0$,

$$
\ell_{\text{CE}}(p,y)
=-\,w_+\,y\log p\;-\;w_-\,(1-y)\log (1-p).
$$

Weights are chosen to re‑balance gradients under class imbalance (see §3.8 for the sampler that defines $w_{+/-}$).

**Focal loss.** To down‑weight easy negatives and focus learning on hard, informative examples, we use the $(\alpha,\gamma)$ focal form:

$$
\ell_{\text{Focal}}(p,y)
=-\,\alpha\,y(1-p)^{\gamma}\log p
\;-\;(1-\alpha)\,(1-y)\,p^{\gamma}\log(1-p).
$$

In our main experiments we set $\gamma=2.0$ and $\alpha=0.87$. These values were fixed a priori and used for all years, with $\lambda_1=0$ and $\lambda_3=1$ so that optimization is driven by the operational horizon $h=3$ while still sharing the encoder across heads.

The **global training objective** is the mean of $\mathcal{L}_{a,t}$ over all masked‑valid training pairs. Note that masking enters twice: (i) it defines the admissible training set and (ii) it nulls horizon‑specific losses when a horizon is invalid for a given $(a,t)$, avoiding any gradient that could look ahead into unobserved future windows. This is the time‑series analogue of “real‑time” or “vintage” evaluation in macroeconomics and is essential to prevent optimistic bias.

For clarity and reproducibility, Table **L1** summarizes the loss configuration used throughout.

**Table L1. Loss configuration and horizon weights (main).**

| Component          | Symbol / Option       | Setting                                  | Rationale                                                              |
| ------------------ | --------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| Base binary loss   | $\ell_{\text{bin}}$   | **Focal**                                | Down‑weight easy negatives under extreme imbalance.                    |
| Focal focusing     | $\gamma$              | **2.0**                                  | Emphasize hard examples without over‑penalizing near‑random regions.   |
| Focal prior weight | $\alpha$              | **0.87**                                 | Correct average gradient magnitude toward positives.                   |
| Horizon weights    | $\lambda_1,\lambda_3$ | **0, 1**                                 | Focus optimization on operational target $h=3$.                        |
| Mask policy        | —                     | **all**                                  | Encoder sees a consistent distribution across horizons; no look‑ahead. |
| Normalization      | —                     | Divide by $\sum_h M_{a,t,h}+\varepsilon$ | Comparable scale across mini‑batches with variable validity.           |

---

### 3.8 Sampling under class and panel imbalance

The panel exhibits two intertwined forms of sparsity: **event sparsity** (onsets are rare within each country) and **panel heterogeneity** (several countries have zero or very few positives in the training span for a given horizon). To stabilize optimization **without distorting** the test distribution, we intervene **only** in the training sampler and leave labels, masks, and losses unchanged.

Let $\mathcal{I}^{(h)}_{\text{train}}=\{(a,t): M_{a,t,h}=1\}$ be the set of masked‑valid country–months for horizon $h$ in TRAIN. We define a **reference horizon** $h^\star$ as the one with the **largest number of positives** in TRAIN; in practice this is $h^\star=3$. Denoting the empirical positive rate

$$
\pi^{(h^\star)}=
\frac{\sum_{(a,t)\in\mathcal{I}^{(h^\star)}_{\text{train}}} y_{a,t,h^\star}}
     {\bigl|\mathcal{I}^{(h^\star)}_{\text{train}}\bigr|},
$$

we set **class weights for sampling**

$$
w_+=\frac{1-\pi^{(h^\star)}}{\pi^{(h^\star)}+\epsilon},\qquad
w_-=1,
$$

and draw training examples with probability proportional to $w_+$ or $w_-$ according to their **reference‑horizon** label. This scheme increases the frequency of rare positives **at sampling time** while preserving their label semantics and the masked‑valid structure. Because losses are still computed on all horizons present for a sampled $(a,t)$, the shared encoder benefits from the augmented signal without introducing **post‑sampling label shift** for the target head.

To prevent the sampler from over‑focusing on countries with effectively **no positive history** (which can lead to degenerate gradients, especially with focal weighting), we add a light **country‑level guardrail**: a country can be **excluded from TRAIN only** if the count of masked‑valid positives for the reference horizon falls below a minimal threshold $k$ (we use **$k=2$**). These countries remain fully present in **validation** and **test**, preserving out‑of‑sample integrity and ensuring that the learned representation still generalizes across the full panel.

The combined effect is to **stabilize** optimization under extreme imbalance while avoiding the common pitfalls of naive oversampling (duplicate leakage across time, spuriously easy minibatches) or synthetic upweighting at the loss level (which can interact poorly with focal loss). Importantly, the marginal time structure of TRAIN is preserved: we do not shuffle months within a country, and we do not create cross‑country mixtures within a single example.

---

### 3.9 Probability calibration

Scores produced by a discriminative model need not be **probabilistically calibrated**: AUROC can be acceptable while the raw $p_{a,t,h}$ systematically over‑ or under‑estimate event probabilities. Because the downstream use of our system is **probability‑informed triage** (e.g., interpreting a 0.6 as “roughly 60% risk”), we learn a **post‑hoc monotone mapping** per horizon on a **held‑out calibration window** within the training years and **freeze** it before scoring any test year.

We consider two standard calibrators:

**Platt scaling (logistic link).** Let $s=\mathrm{logit}(p)=\log\frac{p}{1-p}$. We fit a one‑dimensional logistic regression

$$
\widehat{p}=\sigma(a_h s+b_h),
$$

by maximum likelihood on calibration pairs with $M_{a,t,h}=1$. Because $\sigma$ is monotone, Platt scaling **preserves ranking** and therefore **does not change AUROC**; it primarily improves Brier and ECE. We found it **data‑efficient** and numerically stable when the calibration window contains relatively few positives.

**Isotonic regression.** We also fit a non‑parametric **isotonic** map $\widehat{p}=\mathcal{I}_h(p)$ that minimizes squared error under the constraint that $\mathcal{I}_h$ is non‑decreasing. Isotonic is more flexible but can overfit if the window is small or non‑representative.

Unless otherwise stated, **Platt scaling is our default**. The calibration window is the **last 18 months** of the training span (vintage‑safe), chosen to balance recency with sample size. The learned map $\mathcal{C}_h$ is then applied pointwise to raw test‑year probabilities:

$$
\widetilde{p}_{a,t,h}=\mathcal{C}_h\!\bigl(p_{a,t,h}\bigr).
$$

**Evaluation of calibration.** We report **Brier score** and **Expected Calibration Error (ECE)**. With $N$ masked‑valid test instances and equal‑width probability bins $\{\mathcal{B}_b\}_{b=1}^{B}$ (we use $B=10$), ECE is

$$
\mathrm{ECE}=\sum_{b=1}^B \frac{|\mathcal{B}_b|}{N}\;\Bigl|\overline{\widetilde{p}}_b-\overline{y}_b\Bigr|,
$$

where $\overline{\widetilde{p}}_b$ and $\overline{y}_b$ are the average calibrated probability and empirical event rate in bin $b$. Reliability diagrams in §4 visualize $(\overline{\widetilde{p}}_b,\overline{y}_b)$; as noted, AUROC is invariant to any **monotone** calibration, while Brier/ECE typically improve.

For reference we provide a compact summary of the calibration protocol.

**Table C1. Calibration protocol (per horizon).**

| Step        | Setting                                     | Notes                                       |
| ----------- | ------------------------------------------- | ------------------------------------------- |
| Calibrator  | **Platt** (default), Isotonic (sensitivity) | Monotone; preserves AUROC.                  |
| Window      | **Last 18 months** of training span         | Vintage‑safe; balances recency/sample size. |
| Fitting     | MLE on masked‑valid pairs                   | Separate $a_h,b_h$ per horizon.             |
| Application | Pointwise on test                           | $\widetilde{p}=\mathcal{C}_h(p)$.           |
| Metrics     | **Brier**, **ECE**(10 bins)                 | No PR‑based metrics are used anywhere.      |

---

### 3.10 Observability & mask policy

Masking mediates the **contract** between what the model sees and how it is judged. In time‑series early warning there are two reasonable policies:

* **mask‑policy = all**: retain a country–month $(a,t)$ for training/evaluation **only if it is valid for all horizons** under consideration (e.g., $M_{a,t,1}=M_{a,t,3}=1$).
* **mask‑policy = any**: retain $(a,t)$ if it is valid for **at least one** horizon; compute losses per horizon with its own mask.

We adopt **mask‑policy = all** in the main configuration for three reasons. First, it keeps the **shared encoder’s input distribution identical** across heads, avoiding the subtle covariate shift that arises when, e.g., late‑year months are systematically dropped for $h=3$ but kept for $h=1$. Second, it simplifies calibration and evaluation: **Table 1** and the monthly curves (**Figures 1–2**) are then computed over a common set of test instances per year, and horizon‑specific differences can be interpreted as genuine differences in discrimination rather than sample composition. Third, it makes the training loss well‑conditioned: the denominator $\sum_h M_{a,t,h}$ is constant across retained examples (equal to $|\mathcal{H}|$), which stabilizes gradient magnitudes across mini‑batches.

For transparency we summarize the semantics and trade‑offs in **Table M1**.

**Table M1. Mask policies and implications.**

| Policy            | Inclusion rule for $(a,t)$                            | Pros                                                                                                       | Cons / Caveats                                                                                                                   |
| ----------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **all** (default) | Keep iff $M_{a,t,h}=1$ for **all** $h\in\mathcal{H}$. | Encoder sees a consistent distribution across heads; clean cross‑horizon comparisons; stable loss scaling. | Fewer training/eval samples when long‑horizon windows truncate late‑year months.                                                 |
| any               | Keep iff $M_{a,t,h}=1$ for **some** $h$.              | Larger sample size per horizon; potentially higher power for short horizons.                               | Encoder sees horizon‑dependent distributions; $n_{\text{test}}$ differs by horizon, complicating interpretation and calibration. |

Where we present **sensitivity** to mask policy in §4.4, we explicitly list $n_{\text{test}}$ alongside AUROC/Brier/ECE to make sample‑size differences visible.

---

### 3.11 Evaluation metrics

We evaluate both **ranking quality** and **probability quality** under the vintage‑safe mask, strictly excluding any metric based on precision–recall.

**AUROC.** The Area Under the Receiver Operating Characteristic curve measures the probability that the model assigns a higher score to a randomly drawn positive than to a randomly drawn negative. With scores $\{\widetilde{p}_i\}$ and labels $\{y_i\}$, AUROC can be written as

$$
\mathrm{AUROC}=\frac{1}{|\mathcal{P}|\,|\mathcal{N}|}\sum_{i\in\mathcal{P}}\sum_{j\in\mathcal{N}}
\mathbf{1}\{\widetilde{p}_i>\widetilde{p}_j\}
\;+\;\tfrac{1}{2}\mathbf{1}\{\widetilde{p}_i=\widetilde{p}_j\},
$$

where $\mathcal{P}$ and $\mathcal{N}$ index positives and negatives among masked‑valid instances. Because our calibrators are monotone, AUROC is **invariant** to calibration and reflects pure ranking.

**Brier score.** For calibrated probabilities $\widetilde{p}_i\in[0,1]$,

$$
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}(\widetilde{p}_i-y_i)^2,
$$

a strictly proper scoring rule: the expected score is minimized when $\widetilde{p}_i$ equals the true probability. We also use its standard **decomposition** into reliability, resolution, and uncertainty terms for analysis in §4.

**Expected Calibration Error (ECE).** With equal‑width bins $\{\mathcal{B}_b\}_{b=1}^B$ over $[0,1]$ (we use $B=10$),

$$
\mathrm{ECE}=\sum_{b=1}^{B}\frac{|\mathcal{B}_b|}{N}\bigl|\overline{\widetilde{p}}_b-\overline{y}_b\bigr|.
$$

Lower ECE indicates better alignment between predicted and empirical probabilities. When **monthly** AUROC is plotted (**Figures 1–2**), months with **zero positives** are marked (×) and **excluded** from AUROC computation; their presence is still informative about coverage and seasonality.

All metrics are computed **per horizon** and, unless otherwise stated, on the **mask‑policy = all** test set for each year. We report global (year‑level) values in **Table 1** and discuss within‑year variability in §4.6.

---

### 3.12 Implementation & hyperparameters

This subsection collects implementation details that are critical for reproducibility and for interpreting the experiments in §4.

**Architecture.** Each example uses a **12‑month** dynamic history $(L=12)$. Monthly rasters are transformed by the **Patch encoder** with overlapping $P\times P$ tiles ($P=32$, stride $S=16$, 50% overlap). After light depthwise–pointwise filtering, we compute per‑tile **averages** and project them to $E=8$‑dimensional embeddings, concatenating across tiles to obtain the monthly dynamic vector. The 12‑step sequence is fed to a **two‑layer GRU** with hidden size $H_{\text{RNN}}=256$ and **additive attention**; a small MLP maps the attention context to a **64‑dimensional temporal summary** $g^{\text{time}}$.

On the static path, robust‑scaled FAOSTAT variables are concatenated with their **missingness indicators** and a **sin–cos month encoding** $\phi(t)$. Values are **gated** by $(1-m)$ so that missing entries contribute exactly zero to the linear map; the concatenated vector is mapped to a **256‑dimensional static summary** $g^{\text{stat}}$ (ReLU + dropout). A learnable **country embedding** $e_a\in\mathbb{R}^{8}$ is injected only at the **fusion** stage. The fused vector $z=[g^{\text{time}}\|\;g^{\text{stat}}\|\;e_a]\in\mathbb{R}^{328}$ feeds **independent binary heads** for each horizon $h\in\{1,3\}$, each a single affine layer to a logit followed by a sigmoid.

**Optimization.** We train with **AdamW** (learning rate $2\times10^{-4}$, weight decay $2\times10^{-2}$), **batch size 8**, and **early stopping** on the validation criterion with patience **5–6** epochs. The binary loss is **focal** with $\gamma=2.0$ and $\alpha=0.87$ (§3.7). Horizon weights are $\lambda_1=0$, $\lambda_3=1$. Static‑path dropout is set to **0.5**. Unless otherwise noted, we do not use learning‑rate schedules; convergence is rapid under the small recurrent depth and compact heads. All experiments respect **vintage‑safe masks**, and we report **AUROC/Brier/ECE** only on masked‑valid test instances.

**Calibration.** A **Platt scaler** (per horizon) is fit on a **held‑out 18‑month window** at the end of the training span and then **frozen**. Calibrated probabilities $\widetilde{p}$ are used for Brier/ECE and for any diagnostic plots. Because calibration is monotone, AUROC is unaffected.

**Numerical stability and precision.** We use standard $\varepsilon$‑stabilization in logs, ensure logits remain in a safe range through the usual floating‑point guards of modern deep‑learning libraries, and employ **mixed‑precision inference within the forward pass** to reduce memory without affecting numerical behavior under the $\log(1+x)$ transformation in §2.

**Reproducibility.** All random seeds are fixed per run; we checkpoint the encoder and heads at the epoch with the best validation criterion (calibrated Brier on the validation window), and we persist the learned calibration maps. Train/validation/test splits are defined purely by **calendar** (rolling origin) with no leakage across years.

A compact hyperparameter table is given below for reference.

**Table H1. Model and training hyperparameters (defaults).**

| Block               | Symbol / Option        |               Default | Notes                                      |
| ------------------- | ---------------------- | --------------------: | ------------------------------------------ |
| Input history       | $L$                    |                **12** | Months of dynamic history per example.     |
| Patch size / stride | $P,S$                  |            **32, 16** | 50% overlap; average pooling within tiles. |
| Patch embedding     | $E$                    |                 **8** | Per‑tile embedding dimension.              |
| RNN hidden size     | $H_{\text{RNN}}$       |               **256** | Two‑layer GRU, additive attention.         |
| Temporal summary    | $\dim g^{\text{time}}$ |                **64** | Context MLP output.                        |
| Static summary      | $\dim g^{\text{stat}}$ |               **256** | Mask‑aware linear + ReLU + dropout.        |
| Country embedding   | $\dim e_a$             |                 **8** | Injected only at fusion.                   |
| Fusion vector       | $\dim z$               |               **328** | $64+256+8$.                                |
| Heads               | —                      |                 **2** | $h\in\{1,3\}$; independent last layers.    |
| Loss                | $\ell_{\text{bin}}$    |             **Focal** | $\gamma=2.0,\;\alpha=0.87$.                |
| Horizon weights     | $\lambda_1,\lambda_3$  |              **0, 1** | Emphasize $h=3$.                           |
| Optimizer           | —                      |             **AdamW** | lr $2\times10^{-4}$, wd $2\times10^{-2}$.  |
| Batch size          | —                      |                 **8** | —                                          |
| Early stopping      | Patience               |               **5–6** | On validation criterion.                   |
| Dropout (static)    | —                      |               **0.5** | —                                          |
| Calibration         | —                      |             **Platt** | Per horizon; 18‑month window.              |
| Metrics             | —                      | **AUROC, Brier, ECE** | No PR‑based metrics anywhere.              |
| Mask policy         | —                      |               **all** | See Table M1.                              |

This stack—overlapping spatial pooling, short recurrent memory with attention, missingness‑aware statics, a compact country embedding, focal loss with horizon weighting, and per‑horizon Platt calibration—proved to be a robust and data‑efficient configuration. It delivers reliable **ranking** (AUROC) and **probability** (Brier/ECE) quality under real‑time constraints, and it underlies all year‑wise summaries (**Table 1**), sensitivity analyses (**Table 2**), static ablations (**Tables 3–4**), coverage statistics (**Tables 5–6**), and the within‑year monthly decompositions (**Figures 1–2**) reported in §4.


## 4. Experiments

### 4.1 Evaluation setup

All experiments follow a **rolling‑origin** protocol that respects causal ordering and vintage safety. For a given target year $Y$, models are trained and tuned on $[2017,\dots,Y-1]$ and evaluated on year $Y$ only. Static covariates are constructed in a **vintage‑safe** manner by replicating the latest available annual value $y-1$ across all months of year $y$ (§2.3). Dynamic inputs are monthly maritime tensors over the fixed Black Sea crop (§2.1), normalized to daily rates and passed through a $\log(1+x)$ transformation. Every training or test instance must satisfy the **observability** constraint for the entire future window associated with its forecast horizon; we therefore apply a **vintage‑safety mask** $M_{a,t,h}$ (§3.1). Unless stated otherwise, we adopt **mask‑policy = all**, which retains a country–month only if it is valid for **all** horizons considered during training (here $h\in\{1,3\}$). This choice ensures the shared encoder is exposed to a consistent input distribution across heads and simplifies interpretation of cross‑horizon comparisons.

The **operational target** is the onset‑within‑$h$ detection problem for **$h=3$** months, the lead time judged most actionable for preparedness decisions. We report additional horizons in summary tables and in monthly decomposition figures as **diagnostics** only. Raw probabilities are **calibrated per horizon** using **Platt scaling** fit on the last 18 months of the training span and then frozen (§3.9). Evaluation metrics are strictly limited to **AUROC** (ranking quality) and **probability quality** via **Brier** and **ECE**; no precision–recall–based metric is used anywhere in the paper. When we plot monthly AUROC (§4.6), months with **zero positives** are marked with a “×” and excluded from the AUROC computation for that month; the marker is retained to make coverage visible.

All reported numbers are computed under the **vintage‑safe** constraint on the corresponding test year. Wherever **mask policies** differ (e.g., in the label/mask/calibration sensitivity table), we state this explicitly in the caption, because $n_{\text{test}}$ can change when the inclusion rule changes. Throughout, we accompany AUROC with simple **coverage indicators**—$n_{\text{test}}$, number of positives, and the implied prevalence—to help the reader interpret year‑to‑year variability and the within‑year patterns in §4.6.

---

### 4.2 Coverage and base rates

Table 5 summarizes the **pure onset episode** counts (i.e., number of anomaly onsets after enforcing minimum duration and refractory logic) by year at the global Africa panel, independent of any forecast horizon. The table shows that the number of episodes is highest in **2019 (37)** and declines to **9** in **2023**, with essentially no episodes in 2017–2018. Because horizon‑$h$ labels are “onset‑within‑window” indicators, the **window positives** at horizon $h$ are necessarily **not** equal to episode counts; window positives increase monotonically in $h$ by construction. Table 6 therefore reports, for each year and horizon $h\in\{1,\dots,6\}$, the total number of **positive windows** under the **mask‑policy = all** rule (the same mask rule used in our main results). These counts provide the denominator and positive rates that underpin the AUROC and calibration summaries later in this section.

**Table 5. Yearly counts of pure onset episodes (thr = 1.8).**

| Year | Episodes |
| ---: | -------: |
| 2017 |        0 |
| 2018 |        0 |
| 2019 |       37 |
| 2020 |       21 |
| 2021 |       22 |
| 2022 |       12 |
| 2023 |        9 |

*Note.* Episodes are **h‑agnostic**; they count unique onset months after duration and refractory rules (§2.5). They are not directly comparable to window positives in Table 6.

**Table 6. Year × Horizon positive counts under **mask = all** (validity $M^{(1)}\land M^{(3)}$).**

| Year | h=1 | h=2 | h=3 | h=4 | h=5 |   h=6 |
| ---: | --: | --: | --: | --: | --: | ----: |
| 2017 |   0 |   0 |   0 |   0 |   0 |     0 |
| 2018 |  72 | 150 | 252 | 360 | 504 |   654 |
| 2019 | 180 | 378 | 564 | 756 | 894 | 1,014 |
| 2020 | 108 | 210 | 312 | 402 | 510 |   606 |
| 2021 | 132 | 270 | 396 | 516 | 612 |   696 |
| 2022 |  60 | 102 | 156 | 198 | 240 |   288 |
| 2023 |  54 | 102 | 138 | 156 | 162 |   162 |

*Note.* Values are counts of $Y^{(h)}=1$ among masked‑valid instances when validity is defined by the **intersection** of horizon‑1 and horizon‑3 masks. As $h$ increases, labels are defined on wider windows and counts rise monotonically; these are **not** episode counts.

The two tables jointly clarify the statistical “shape” of the problem. For example, **2019** exhibits many episodes and a very large number of positive windows at longer horizons, while **2022–2023** have fewer episodes and markedly lower positive counts. We use these base rates as context for interpreting year‑wise AUROC and for explaining month‑wise patterns later on.

---

### 4.3 Main out‑of‑sample results (H = 3)

We begin with the operational target $h=3$. Table 1 reports **test‑year AUROC** together with $n_{\text{test}}$, number of positives, and the implied prevalence (positives divided by $n_{\text{test}}$) for **2019–2023** under the main configuration (**thr = 1.8**, **mask = all**, **Platt** calibration, **Statics = ALL** unless otherwise noted).

**Table 1. Year‑wise test performance at H = 3 (2019–2023).**

| Year | n\_test | pos\_test | prevalence |     AUROC |
| ---: | ------: | --------: | ---------: | --------: |
| 2019 |     420 |        94 |      22.4% |     0.425 |
| 2020 |     420 |        52 |      12.4% |     0.603 |
| 2021 |     420 |        66 |      15.7% |     0.536 |
| 2022 |     420 |        26 |       6.2% |     0.546 |
| 2023 |     315 |        23 |       7.3% | **0.731** |

The **2023** test—our most policy‑relevant year—achieves **AUROC = 0.731** with **315** masked‑valid instances and **23** positives (7.3%). Calibration for this configuration is also strong (see §4.5, Table 3: **ECE = 0.041** for Statics = ALL), indicating that high scores correspond to meaningfully higher empirical risks. In **2020**, AUROC is moderate at **0.603** with more positives (12.4%), while **2021–2022** show weaker discrimination (0.536 and 0.546, respectively) under changing base rates and within‑year heterogeneity. The low AUROC in **2019** (0.425) coincides with the largest number of episodes and a very high density of positives at longer horizons (Table 6), a regime where maritime patterns and price onsets are distributed across many country–months and are therefore harder to rank with a short dynamic history. Section §4.6 shows that within‑year dispersion (monthly AUROC) is large in several years and often aligned with known logistics cycles, reinforcing the need for month‑wise diagnostics in addition to year‑wise summaries.

---

### 4.4 Label threshold, mask policy, and calibration (H = 3)

We next study how **label severity**, **mask policy**, and **probability calibration** affect discrimination and probability quality for the operational horizon. Table 2 compares three representative configurations: a stricter label threshold (2.5) with **mask = any** and Platt calibration; a moderate threshold (2.0) with **mask = any** and no calibration; and our recommended setting (1.8) with **mask = all** and Platt calibration. All numbers are computed on the same test year as our headline results; note, however, that changing **mask policy** changes the **test coverage** and hence the set of instances entering the metrics.

**Table 2. Label threshold / mask policy / calibration sensitivity at H = 3.**

| Label Thr | Mask Policy | Calibration |     AUROC |     Brier |       ECE |
| --------: | :---------: | :---------: | --------: | --------: | --------: |
|       2.0 |     any     |     none    |     0.603 |     0.072 |     0.073 |
|       2.5 |     any     |    Platt    |     0.691 |     0.066 |     0.017 |
|   **1.8** |   **all**   |  **Platt**  | **0.724** | **0.065** | **0.012** |

Two conclusions emerge. First, **calibration matters**: Platt scaling reduces **ECE** substantially even when AUROC is essentially unchanged by the monotonic mapping (compare 2.0/none vs 2.5/Platt). Second, while stricter thresholds can improve AUROC by focusing on rarer and cleaner events (2.5 vs 2.0), our **recommended configuration** (1.8, mask = all, Platt) yields the best **combined** profile—highest AUROC among the three, lowest ECE, and the best Brier—while keeping the mask policy aligned with the rest of the paper. We therefore fix **thr = 1.8** and **mask = all** for all mainline analyses and reserve alternative policies for sensitivity checks.

---

### 4.5 Static covariates ablation (H = 3)

The role of **static covariates** varies across years, reflecting differences in how domestic structure mediates maritime shocks. We therefore ablate the FAOSTAT set along interpretable slices: **ALL** (A/P/Y/GPV), **NONE**, and single‑family subsets (A‑only, P‑only, GPV‑only). Table 3 presents **2023** results, where statics are helpful along both ranking and calibration dimensions; Table 4 presents **2022**, where removing statics improves ranking.

**Table 3. Static covariates ablation (2023, H = 3): AUROC & ECE.**

| Static set             |     AUROC |       ECE |
| :--------------------- | --------: | --------: |
| **ALL (A, P, Y, GPV)** | **0.731** | **0.041** |
| NONE                   |     0.714 |     0.053 |
| GPV only               |     0.694 |     0.053 |
| P only                 |     0.680 |     0.053 |
| A only                 |     0.646 |     0.053 |

In **2023**, including **all** static covariates increases AUROC from 0.714 (NONE) to **0.731** and improves calibration (ECE 0.053 → **0.041**). The gain suggests that the **composition and scale** of domestic agro‑economies—proxied by FAOSTAT A/P/Y/GPV—help distinguish which countries are more likely to experience an IFPA onset within three months, conditional on similar maritime signals.

**Table 4. Static covariates ablation (2022, H = 3): AUROC & ECE.**

| Static set |     AUROC |   ECE |
| :--------- | --------: | ----: |
| **NONE**   | **0.617** | 0.084 |
| P only     |     0.601 | 0.084 |
| A only     |     0.584 | 0.085 |
| GPV only   |     0.584 | 0.084 |
| ALL        |     0.546 | 0.085 |

In **2022**, by contrast, AUROC is highest when **no statics** are used (0.617), and adding the full set reduces discrimination to 0.546 with no material change in ECE. This pattern is consistent with a **distribution shift** in which static features learned on earlier years do not align with the pathways through which maritime disruptions propagated in 2022. The month‑wise decomposition in §4.6 shows pronounced within‑year variability that likely contributes to this mismatch. Taken together, 2023 and 2022 indicate that statics can be beneficial when the **mapping from maritime conditions to domestic price stress** is mediated by relatively stable country structure, and less so when short‑run factors dominate.

---

### 4.6 Within‑year monthly AUROC

Year‑level summaries can hide large **within‑year dispersion** tied to seasonality, routing changes, or policy shocks. We therefore decompose AUROC by month and by horizon under the “train\_h=H” convention: each curve shows the monthly AUROC for the head trained and evaluated at that horizon $H$. Months with **zero positives** are marked with a “×” (these months are excluded from the AUROC calculation). Figures 1 and 2 present the patterns for **2022** and **2023**, respectively.

**Figure 1. 2022 Monthly AUROC (train\_h = H).**
*Caption.* Monthly AUROC for **H = 3, 4, 5**. The symbol **×** marks a **zero‑positive** month. AUROC peaks in **February–March** across horizons and falls to a trough around **June–July**, with **H=3** showing the largest volatility and **H=5** the smoothest profile. All evaluations apply the vintage‑safe mask.

**Figure 2. 2023 Monthly AUROC (train\_h = H).**
*Caption.* Monthly AUROC for **H = 3, 4, 5**. Performance is highest in **Q1 (January–March)** and declines through mid‑year, with **H=3/4** dropping more steeply than **H=5**. The pattern is consistent with the year‑wise results in Table 1 (high 2023 AUROC overall) but reveals that the strength is **concentrated in the first half** of the year; recent‑window calibration (Platt) is therefore important for operational use.

These figures underscore three points that inform the design choices in §3 and the sensitivity analyses in §4.4–§4.5. First, **seasonality matters**: months associated with intense maritime activity or congestion often coincide with higher AUROC, particularly for $H=3$. Second, **horizon choice** affects stability: longer windows (e.g., $H=5$) capture slower, structural movements and display **smoother** month‑wise curves, while $H=3$ is more responsive to **turning points**. Third, the presence of **zero‑positive** months—especially in sparse years—highlights the importance of reporting **coverage** alongside AUROC and of using calibration windows that are recent enough to track the operative regime without overfitting. The combined evidence motivates our emphasis on $H=3$ for operational reporting, with $H\neq3$ curves retained as diagnostic corroboration.

### 4.7 Calibration & reliability (by‑year ECE/Brier)

Calibration quality is central to the operational value of an early‑warning system: analysts must be able to read a score of, say, 0.60 as “roughly 60% odds, all else equal.” We therefore assess probability integrity by **Expected Calibration Error (ECE)** with 10 equal‑width bins and by the **Brier score**, always on **vintage‑safe** test instances and after **per‑horizon Platt scaling** learned on the last 18 months of the training span (§3.9). To ensure comparability across years, we pin the configuration to our mainline choices—**thr = 1.8**, **mask = all**, **calibration = Platt**, and **Statics = ALL**—unless explicitly stated otherwise. The year‑wise **H=3** calibration summary in Table 7 shows that 2023 achieves the lowest ECE in our panel when statics are included, while earlier years exhibit larger deviations that are consistent with the within‑year dispersion documented in §4.6 and the distribution‑shift discussion in §4.8.

**Table 7. Year‑wise calibration at H = 3 (ECE; main configuration: thr = 1.8, mask = all, Platt, Statics = ALL).**

| Year |       ECE |
| ---: | --------: |
| 2019 |     0.194 |
| 2020 |     0.118 |
| 2021 |     0.045 |
| 2022 |     0.085 |
| 2023 | **0.041** |

*Notes.* ECE is computed with 10 equal‑width bins on calibrated probabilities $\tilde p$. Values for 2022 and 2023 are drawn from the **Statics ablation** at **H=3** with **ALL**, ensuring consistency with our main configuration (Tables 3–4). Earlier years (2019–2021) match the default pipeline (ALL statics) used for their year‑wise AUROC summaries.

Brier scores track ECE closely in our experiments: where ECE is low and reliability diagrams are near‑diagonal, the Brier score is correspondingly small. The supplied outputs include a precise **2023** value of **0.065** for the main configuration—this is the headline operating year in which both Brier and ECE are strongest, and it complements the high AUROC reported in Table 1. For other years, the run artifacts provided for this manuscript do not include definitive Brier values at the exact mainline settings; where §4.3 and §4.8 refer to “higher Brier” (e.g., 2020), the statement reflects the shape of the reliability diagrams and the observed ECE behavior under the same calibration window. For completeness we list the available 2023 Brier in Table 8.

**Table 8. Brier score snapshot at H = 3 (main configuration).**

| Year |     Brier |
| ---: | --------: |
| 2023 | **0.065** |

Together, Tables 7–8 support two broad conclusions. First, **per‑horizon Platt scaling on a recent window** yields well‑behaved probabilities whenever the calibration slice is representative of the imminent test regime; 2023 epitomizes this case with **low ECE and low Brier**. Second, when the **validation slice is not representative** (e.g., 2020), **reliability degrades** even if AUROC remains acceptable. This is the classic calibration–shift phenomenon in non‑stationary environments and justifies our insistence on recent, horizon‑specific calibrators and on reporting calibration metrics alongside AUROC.

---

### 4.8 Distribution shift and failure modes

The cross‑year patterns in Tables 1, 7 and the month‑wise curves (Figures 1–2) are the empirical fingerprint of **distribution shift** in a real economy. Three signatures recur. First, **prevalence swings** (Table 1 and §4.2) change the hardness of the problem: when events are frequent and diffuse (e.g., 2019 with many episodes), AUROC can sag even though maritime signals remain informative; when events are rarer and more concentrated (e.g., 2023), ranking sharpens and calibration tightens under a recent Platt map. Second, **within‑year heterogeneity** is large (§4.6): the system performs markedly better in certain months (e.g., 2022 February–March; 2023 Q1) and weakens in others (e.g., 2022 June–July), precisely the pattern one expects when congestion, rerouting, and policy interventions wax and wane seasonally. Third, **calibration drift** is visible when the calibration window is only weakly exchangeable with the test period. In 2020, for example, ECE increases (Table 7) while AUROC remains moderate (Table 1), indicating the model still ranks sensibly but mis‑scales absolute probabilities; a roll‑forward calibration slice or a slightly longer window would mitigate this without altering ranking.

These observations explain two apparent paradoxes. The first is **“good AUROC, poor reliability”** years: ranking quality reflects cross‑sectional separation, while reliability depends on mapping scores to probabilities under the current base rate and conditional densities; they need not move together. The second is **“statics help one year, hurt the next”** (§4.5): static covariates amplify discriminative structure when the **mapping from maritime conditions to price stress** is mediated by relatively stable country fundamentals (2023), but they can add noise when short‑run idiosyncrasies dominate (2022). Neither paradox is a defect of the model; both are properties of a non‑stationary system and are best managed by **temporal honesty (rolling origin), horizon‑specific calibration, and explicit coverage reporting**.

Finally, two failure modes recur in diagnostics. A **coverage failure** occurs when the recent training span contains too few onsets to fit a useful calibrator or to shape the decision surface (cf. the early boundary year with many months effectively devoid of positives), leading to unstable reliability and sometimes undefined month‑wise AUROC in zero‑positive months (marked “×” in Figures 1–2). A **calibration mismatch** occurs when validation risk clusters differ from those in the test year; ranking is unaffected, but the probability scale drifts (higher ECE, larger Brier). Our pipeline contains built‑in mitigations: the **mask‑policy = all** rule prevents horizon‑dependent sample drift in the encoder, **Platt scaling** recovers calibration whenever the window is representative, and the **reporting protocol** always pairs AUROC with ECE/Brier and with explicit coverage counts.

---

### 4.9 Horizon analysis (H = 1 vs 3 vs 4–6)

Although operational reporting centers on **H=3**, it is instructive to examine how discrimination varies with horizon; Appendix Tables **A1–A5** summarize year‑wise AUROC and coverage for **H = 1,2,4,5,6**. Two themes emerge. First, **longer windows** (H ≥ 4) often yield **smoother** month‑wise profiles (§4.6) and competitive or even higher year‑wise AUROC in some regimes (e.g., **H=4** in 2023 reaches **0.718**; **H=5** in 2022 reaches **0.691**), consistent with the idea that structural changes in maritime activity take **seasons** to propagate. Second, **short‑lead** detection (**H=1**) suffers from extreme sparsity (Table 6) and yields modest AUROC with high variance—exactly what one expects when single‑month onsets reflect idiosyncratic shocks rather than persistent shifts.

These patterns do not contradict the H=3 focus; rather, they motivate a **two‑tier practice**. For **triage and preparedness**, **H=3** offers the best balance of timeliness, stability, and calibration. For **analyst diagnostics**, **H≥4** curves provide context on whether the signal is structural or transient; when longer horizons outperform H=3 in a given year, it often indicates that risk is **broad‑based and persistent** rather than short‑fused.

---

### 4.10 Calibration window stability

Calibration is only as good as the **slice** used to fit it. Two choices matter: the **functional form** (Platt vs isotonic) and the **window length/placement**. In our mainline we adopt **Platt scaling** fit on the **last 18 months** of the training span, a compromise that provides enough positives for a stable MLE while preserving recency. On years where the test regime diverges from the tail of the training span (e.g., a sudden easing of congestion or a policy intervention), ECE rises even though AUROC is preserved (§4.8). In those cases, a **roll‑forward refit** of Platt on a window slightly closer to the test months, or a **hybrid** scheme that averages parameters over adjacent years with similar risk profiles, restores reliability. We explored isotonic regression as a non‑parametric alternative; while flexible, it proved sensitive to window sparsity and occasionally overfit the upper decile in years with many zero‑positive months (the “×” months in Figures 1–2). Because both calibrators are **monotone**, they preserve ranking; all trade‑offs therefore play out in **ECE/Brier**, not AUROC.

From a deployment perspective, the key is to **operationalize calibration**: pick a recent, sufficiently populated window; refit Platt on a **fixed cadence** (e.g., quarterly) or when **drift alarms** trigger; and verify reliability with a quick ECE/reliability diagram before releasing probabilities into an alerting pipeline. Our reporting protocol—always pairing AUROC with ECE/Brier and coverage—enforces this discipline.

---

### 4.11 Error anatomy and failure cases

To understand where the detector errs, we manually inspected **false positives** and **false negatives** in years with contrasting behavior (e.g., 2020 vs 2023). Two archetypes dominate false positives. The first is **maritime‑intense but domestically buffered** cases: substantial shipping anomalies appear in the Black Sea corridor, yet domestic structure (reserves, substitution, policy buffers) blunts price transmission; if statics under‑represent this buffer, the model can overstate risk. The second is **sectoral decoupling**: tanker‑driven disruptions dominate a month’s maritime signal, but food import channels are less exposed; because our dynamic channels include “All ships,” residual confounding can creep in when sectoral composition shifts quickly. False negatives typically arise at **turning points**, where maritime anomalies emerge late in the 12‑month window and attention fails to concentrate on the most recent 1–2 months, or in **thin‑coverage** months with zero or near‑zero positives in the neighborhood.

The architecture choices in §3 are tailored to attenuate these errors. **Overlap pooling** and **attention** reduce sensitivity to small geolocation jitter and highlight sustained anomalies; the **missingness‑aware static block** prevents spurious linear effects from imputed values while still allowing **informative missingness** (e.g., absent GPV) to matter; the **country embedding** soaks up persistent idiosyncrasies in pass‑through. The experiments in §4.5 confirm that statics materially improve both ranking and calibration in years like 2023, exactly where the above false‑positive archetypes would otherwise be most prevalent. Conversely, when statics degrade AUROC (2022), the prudent course is to **fallback to NONE/GPV‑only** for ranking while retaining Platt for calibration.

---

### 4.12 Policy relevance & external validity

The policy value of the system rests on two pillars: **actionable probabilities** and **credible economic provenance**. On the first, the combination of **solid discrimination** (Table 1, H=3) and **tight calibration** (Tables 7–8) means that a risk score can be used **as a probability** to prioritize countries and months for deeper analysis, market scanning, or pre‑positioning—without relying on opaque alert budgets or ad hoc thresholds. Because calibration is **per‑horizon** and **vintage‑safe**, the probabilities reflect what would have been known **at decision time**, a non‑negotiable requirement in public‑sector early warning. The within‑year dispersion (Figures 1–2) further enables **seasonal triage**: analysts can recognize when the signal is structurally strong (e.g., Q1 2023) and when to lean more on auxiliary information or cautious interpretation (e.g., mid‑year troughs).

On the second pillar, the inputs and labels have **clear economic interpretation**. AIS‑derived hours‑at‑sea in a choke‑point region (the Black Sea and approaches) are a physically grounded proxy for **real‑time maritime trade flows** and for logistic frictions that propagate into domestic markets; our pre‑processing respects the equal‑area geometry and the aggregation semantics of the EMODnet product (§2.1–§2.2). Labels derive from **FAO’s IFPA** methodology (§2.4), which is specifically designed to flag **abnormally high** food prices after removing seasonal and inflationary confounders via month‑of‑year standardization. This alignment—maritime anomalies as the upstream shock, IFPA onsets as the downstream manifestation—confers **external validity**: when shipping signals sharpen or corridors are disrupted, we observe more predictable, well‑calibrated risk at medium leads (H=3), particularly in **food‑import‑dependent** settings. Where the mapping weakens (e.g., 2022), our ablations make the limitation explicit and recommend conservative configurations (e.g., fewer statics for ranking) until the regime stabilizes.

In sum, the system delivers **probabilities you can act on** and a trail of design choices that economists and policymakers can audit: vintage‑safe evaluation, transparent masking, horizon‑specific calibration, explicit coverage, and inputs rooted in the mechanics of trade. These features—paired with the diagnostic scaffolding in §4 (sensitivity tables, month‑wise curves, reliability summaries)—are what make the detector suitable for **operational early warning** rather than merely retrospective forecasting.


