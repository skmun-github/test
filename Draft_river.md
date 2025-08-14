## 1) Goal

**Problem.** We want to forecast how **hydro-climate conditions along the Mississippi River** cascade into (i) **river logistics/throughput** and, in turn, (ii) **international trade flows associated with the Port of New Orleans (Port NOLA)**—by partner **country** and, where possible, by **foreign port**.

**Why this matters.** The Mississippi River System (MRS) underpins U.S. bulk exports—especially grains. When stages/flows are abnormal (floods, droughts, saltwater intrusion), tows are shortened, drafts are reduced, and transits slow, which elevates freight costs and crimps export volumes; we saw this during 2022–2024 low-water episodes on the Mississippi and Ohio Rivers. Federal reporting documents these effects and use Lock 27 near Granite City, IL as a key indicator for downbound grain volumes and rate spikes, e.g., 2022’s record barge rates and depressed 2023 volumes. [source](https://www.ams.usda.gov/services/transportation-analysis/gtr) Low water and saltwater wedge management (e.g., underwater sills near Myrtle Grove, LA) have recently required extraordinary measures and directly affect navigation and water supply—conditions that are measurable and thus forecastable. [source](https://www.mvn.usace.army.mil/Missions/Engineering/Saltwater-Wedge/)

**Two-stage forecasting architecture (deep learning).**

- **Stage A (Hydro-logistics):** Map **hydrometeorological forcings** (rainfall, temperature, soil moisture proxies, modeled flows) and **river stages** (e.g., the Carrollton gage at RM 102.8 AHP in New Orleans) to **logistics indicators** on the river system (e.g., barge throughput, queueing/lock delays upstream, navigation bulletins and Coast Guard advisories, air gaps/clearances). The Carrollton gage is the standard stage reference for the Lower Mississippi reach around New Orleans. [source](https://water.weather.gov/ahps2/hydrograph.php?wfo=lix&gage=norl1)

- **Stage B (Logistics-to-Trade):** Map those **intermediate logistics indicators** (plus lagged hydrology) to **Port NOLA-linked monthly trade flows** by **country** (public data) and, where licensed, by **foreign port** from **bill-of-lading (BOL)** microdata. Public U.S. trade systems provide **port (Schedule D)** and **country** dimensions; **foreign port of unlading** is collected in export filings (Schedule **K**), but not disseminated in public aggregates—so BOL data (PIERS, Datamyne) is the defensible route to port-to-port flows. [source](https://usatrade.census.gov/) [source](https://www.census.gov/foreign-trade/reference/codes/index.html)

**Targets & horizons.**

- **Stage A outputs (weekly / monthly):** Downbound barge tonnage proxies (e.g., at Lock 27), barge rate indices (USDA-AMS), incident/advisory indicators (USACE NTNI, USCG MSIB), bridge clearance/air gap risk metrics at critical spans. [source](https://www.ams.usda.gov/services/transportation-analysis/gtr)
- **Stage B outputs (monthly):**  
  - **Primary:** Value/quantity of exports (and imports) **from Port NOLA** by **partner country** (public). [source](https://usatrade.census.gov/)  
  - **Extended (requires license):** Port-NOLA **→ foreign port** flows using BOL (PIERS or Datamyne). [source](https://www.spglobal.com/marketintelligence/en/solutions/piers) [source](https://www.datamyne.com/)
- **Forecast horizons:** 1–12 weeks for Stage A; 1–6 months for Stage B (with lags reflecting transit/booking cycles).

**Geospatial frame.** River miles measured **Above Head of Passes (AHP)** is the standard for the Lower Mississippi; we will anchor sensors and events to AHP and to USACE/NOAA station IDs (e.g., the Carrollton gage at 102.8 AHP). [source](https://rivergages.mvr.usace.army.mil/WaterControl/shefgraph-miss.cfm?sid=NORL1&detailed=true)

**Scientific value.** This design leverages best-available hydro models (NOAA **National Water Model** retrospective v2.1; forthcoming v3.0) and daily 1-km climate grids (**Daymet**) to create physically meaningful features, then couples them to operational logistics and official trade statistics—enabling interpretable deep-learning forecasts and counterfactuals under shocks (e.g., low-water or saltwater wedge episodes). [source](https://registry.opendata.aws/nwm-archive/) [source](https://daymet.ornl.gov/)

## 2) Databases & Pre-processing

Below are the **minimum viable datasets**, **why** each is needed, and **how** we will clean/align them for deep learning. I distinguish **open** data (download-ready) from **licensed** data needed to resolve micro-level relationships (e.g., port-to-port).

### 2.1 Hydro-meteorology & hydrology (inputs)

1) **USGS real-time and historical gage data (NWIS/WDFN).**  
   *What:* River **stage** and **discharge** at key Lower Mississippi points (e.g., **New Orleans—Carrollton gage**, RM 102.8 AHP). Near-real-time readings are generally taken every 15 minutes and disseminated hourly via the **Instantaneous Values (iv)** service, with daily aggregates via the **Daily Values (dv)** service. [source](https://waterdata.usgs.gov/nwis)  
   *Why:* Stages drive draft/clearance constraints, tow sizes, and safety measures.  
   *Access & API:* USGS Water Services (REST), National Water Dashboard for discovery. [source](https://dashboard.waterdata.usgs.gov/)  
   *Pre-processing:*  
   - Convert gage heights to common vertical reference where needed (e.g., **NAVD88** adjustments published for Carrollton). [source](https://rivergages.mvr.usace.army.mil/)  
   - Harmonize timestamps to UTC and compute rolling statistics (7/14/30-day means/anomalies); interpolate gaps conservatively.

2) **NOAA National Water Model (NWM) – retrospective + real-time.**  
   *What:* Gridded, modeled **streamflow**, **soil moisture**, and other hydrologic fields for CONUS; **retrospective v2.1** spans **1979–2020**, with **v3.0 retro** extending to 2023 (being rolled out via NOAA dissemination channels). [source](https://registry.opendata.aws/nwm-archive/)  
   *Why:* Basin-wide hydrologic context (beyond point gages), enabling robust features and backtesting over decades.  
   *Access:* NOAA Big Data Program (AWS S3; Zarr/NetCDF), OWP documentation. [source](https://water.noaa.gov/about/nwm)  
   *Pre-processing:*  
   - Extract reaches feeding Lower Mississippi segments; aggregate to subbasins feeding Baton Rouge → New Orleans corridor; compute flow/runoff anomalies and persistence indices.

3) **High-resolution daily climate grids.**  
   - **Daymet v4** (daily, **1-km**; Tmin/Tmax, precip, vapor pressure, shortwave, SWE): 1980→present year (CONUS); authoritative ORNL/NASA product. [source](https://daymet.ornl.gov/)  
   - **PRISM** (daily, **4-km** CONUS; 30-yr **normals at 800 m**): used here for climatologies and cross-validation. [source](https://prism.oregonstate.edu/)  
   *Why:* Weather forcings underpin river stages and navigation risk.  
   *Pre-processing:* Spatially average grids over MRS subbasins; compute standardized anomalies vs. PRISM normals; create lagged features (1–8 weeks).

4) **Operational river/coastal observing networks.**  
   - **NOAA PORTS®—Lower Mississippi** (real-time water levels, currents, met; includes bridge **air gap** sensors where deployed). [source](https://ports.noaa.gov/ports/)  
   - **NWS Lower Mississippi River Forecast Center (LMRFC)** (official hydrologic forecasts for this basin). [source](https://www.weather.gov/lmrfc/)  
   *Why:* Real-time constraints on navigation (clearance, currents, winds).  
   *Pre-processing:* Pull hourly; align to gage–bridge metadata; derive minimum clearances by span from air-gap relationships (e.g., published adjustments tying **Carrollton** stage to Baton Rouge I-10 bridge clearance).

5) **Saltwater intrusion / navigation constraints.**  
   *What:* USACE New Orleans **saltwater wedge** overview and **sill construction** bulletins (2022–2024). [source](https://www.mvn.usace.army.mil/Missions/Engineering/Saltwater-Wedge/)  
   *Why:* Episodes change allowable drafts and potentially halt some operations; we will encode **event flags** and **severity** (distance of wedge toe, sill status).  
   *Pre-processing:* Time-stamp USACE notices; join to river mile AHP; propagate lags.

---

### 2.2 River logistics, navigation & barging (Stage-A labels and key covariates)

1) **USACE LPMS / Corps Locks (public “Corps Locks” portal).**  
   *What:* **Near real-time (≈15–30 min)** snapshots from the **Lock Performance Monitoring System**: queue/lockage status, tonnage roll-ups, outages. Public site provides FOIA-cleared aggregates; APIs expose queue/tonnage/traffic. [source](https://corpslocks.usace.army.mil/)  
   *Why:* Although the Lower Mississippi (below Baton Rouge) is lock-free, **upstream locks** (e.g., **Lock & Dam 27**) are the controlling choke for **downbound** barge flows that feed Port NOLA. BTS relies on Lock 27 volumes in its Port Performance report to illustrate low-water effects; we will do the same.  
   *Pre-processing:* Pull monthly tonnage and delay metrics by lock; create a **downbound flow index** (Lock 27) and regional indices for tributaries (Ohio/Illinois).

2) **USDA-AMS Grain Transportation** (weekly).  
   *What:* **Downbound grain barge movements** and **barge freight rates** (St. Louis, Cairo-Memphis, etc.)—used by BTS to track low-water disruptions. [source](https://www.ams.usda.gov/services/transportation-analysis/gtr)  
   *Why:* Price/quantity signals complement physical throughput; strong proxies for export pipeline capacity.  
   *Pre-processing:* Weekly → monthly aggregation (end-of-month and average); compute spreads vs. historical medians.

3) **USACE Notices to Navigation Interests (NTNI) & USCG MSIB (Sector New Orleans).**  
   *What:* Official navigational **restrictions/outages** (dredging, channel width/depth changes, high-water measures, fog/visibility advisories, anchorage changes, safety zones). NTNI is USACE’s portal; USCG MSIB lists Sector New Orleans bulletins. [source](https://www.mvn.usace.army.mil/Missions/Navigation/Notices-to-Navigation-Interests/) [source](https://homeport.uscg.mil/)  
   *Why:* These are immediate operational frictions; we encode **binary flags** and **intensity scores** (e.g., one-way traffic, tow size limits).  
   *Pre-processing:* Parse text, assign to river-mile segments (AHP), add duration and lead/lag features.

4) **Bridge clearance / air-gap constraints.**  
   *What:* Clearance formulas tied to river stage (e.g., **Baton Rouge I-10 bridge** “153 ft minus Carrollton gage,” as used by local industry), and **Carrollton gage** stage used to set restrictions at key bridges.  
   *Why:* Clearance affects tow configurations and shipment timing.  
   *Pre-processing:* Compute dynamic clearances; flag threshold crossings.

5) **AIS vessel movement data (context & model features).**  
   - **Open:** NOAA/BOEM **MarineCadastre** distributes historic AIS (primarily coastal/near-coastal; monthly zips). [source](https://marinecadastre.gov/ais/)  
   - **Licensed, higher fidelity:** **Lloyd’s List Intelligence Seasearcher** (global vessel movement database) or **Spire Maritime** (satellite AIS + terrestrial feeds).  
   *Pre-processing:* Derive transit times into/out of Port NOLA anchorage, queue indicators, and river speed/turning proxies; aggregate by week/month.

---

### 2.3 Trade (Stage-B labels and join keys)

1) **USA Trade Online (U.S. Census Bureau).**  
   *What:* Official monthly **merchandise trade**; includes **Port (Schedule D)** dimension (6-digit HS at the port level), and **country** of partner. (Free account required.) [source](https://usatrade.census.gov/)  
   *Why:* This is the authoritative, public source for **Port NOLA ↔ Country** flows (value/quantity).  
   *Limits:* Public outputs do **not** include the **foreign port of unlading** dimension even though it is reported in export filings. That foreign-port code is **Schedule K** in the Electronic Export Information (EEI). [source](https://www.census.gov/foreign-trade/reference/codes/index.html)

2) **USITC DataWeb.**  
   *What:* Parallel access to official Census trade statistics and U.S. tariffs; useful for country-level checks and alternative API. [source](https://dataweb.usitc.gov/)

3) **Schedule codebooks (for joins and QC).**  
   - **Schedule D** (U.S. **port** codes): current CBP/Census lists. [source](https://www.census.gov/foreign-trade/reference/codes/schedule-d.html)  
   - **Schedule K** (foreign **port** codes): maintained by USACE WCSC and referenced by CBP. (Used in filings; needed for BOL alignment.) [source](https://www.census.gov/foreign-trade/reference/codes/schedule-k.html)

4) **Licensed microdata to recover port-to-port.**  
   - **S&P Global PIERS** (U.S. **bill-of-lading** microdata with foreign **port of unlading/loading**). [source](https://www.spglobal.com/marketintelligence/en/solutions/piers)  
   - **Descartes Datamyne** (BOL microdata; daily updates for U.S. maritime imports; export coverage and global linkages). [source](https://www.datamyne.com/)  
   - **Optional AIS intelligence layers:** **Lloyd’s List Intelligence (Seasearcher)**, **Kpler/MarineTraffic**, or **Spire Maritime** to confirm port calls and voyage routing.

5) **USACE Waterborne Commerce Statistics Center (WCSC).**  
   *What:* National waterborne tonnage statistics, port and waterway characteristics, and port statistical area boundaries. Useful for **port boundary** validation and benchmarking against public indicators; BTS relies on WCSC for its port rankings. [source](https://usace.contentdm.oclc.org/digital/collection/p16021coll2/id/1264)

---

### 2.4 Pre-processing plan (end-to-end)

**A. Ingestion & versioning.**  
- Build reproducible pullers for **USGS iv/dv APIs** (JSON/WaterML), **NWM retro** (S3/Zarr/NetCDF), **Daymet** (tiles/NetCDF/CSV), **PORTS** feeds, **LPMS/Corps Locks** APIs, **NTNI/MSIB** bulletins, **USA Trade Online** exports, and (if licensed) **PIERS/Datamyne**. Store raw → bronze → silver layers with metadata (source, retrieval time, license). [source](https://waterdata.usgs.gov/nwis) [source](https://registry.opendata.aws/nwm-archive/) [source](https://daymet.ornl.gov/) [source](https://ports.noaa.gov/ports/) [source](https://corpslocks.usace.army.mil/) [source](https://www.mvn.usace.army.mil/Missions/Navigation/Notices-to-Navigation-Interests/) [source](https://homeport.uscg.mil/) [source](https://usatrade.census.gov/)

**B. Temporal alignment.**  
- Normalize to **UTC**; retain local time for interpretability flags (e.g., LIX WFO for New Orleans). Aggregate to **weekly** features for Stage A and **monthly** labels for Stage B.  
- For long-context hydrology, use **NWM v2.1 (1979–2020)** for backtesting; append **operational NWM** for 2021→present while noting configuration differences; transition to **v3.0 retro** when fully published. [source](https://registry.opendata.aws/nwm-archive/)

**C. Spatial alignment.**  
- Snap in-river observations to **AHP mileposts** and critical assets (bridges, anchorages). Use USACE/NOAA documentation on **AHP** conventions and station locations (e.g., **Carrollton gage at 102.8 AHP**).  
- Build **river-segment features** (e.g., New Orleans–Baton Rouge corridor) and associate **NTNI/MSIB** notices to nearest segments.  
- For ports, use **USACE Port/Port Statistical Area** layers to check that “Port of New Orleans” boundaries align with your joins.

**D. Feature engineering (examples).**  
- **Hydro-climate:** rolling 7/14/30-day sums of Daymet precip; Tmin/Tmax anomalies vs. PRISM normals; NWM reach-scale flow percentiles and persistence.  
- **Hydraulic constraints:** dynamic **bridge air-gap** estimates from stage (e.g., Baton Rouge I-10 formula), min clearance flags, flow velocity proxies (USACE tables tied to Carrollton stage).  
- **Operations/disruptions:** one-hot and duration features for **NTNI** dredging/closures and **MSIB** safety measures (e.g., one-way traffic windows, tow-size restrictions near **81-Mile Point**).  
- **Barging/throughput:** Lock 27 downbound tonnage index; weekly barge rate indices from USDA-AMS; AIS-derived anchorage/berth waiting times if licensed.  
- **Seasonality & calendar:** harvest timing for corn/soy (month dummies), holiday effects, fiscal quarters.

**E. Label construction.**  
- **Stage A labels:** monthly logistics indicators (e.g., Lock 27 downbound tonnage; barge rate levels/spreads; counts of active restrictions).  
- **Stage B labels (public):** monthly **Port NOLA ↔ Country** values/quantities from **USA Trade Online** (Schedule **D** port + partner country).  
- **Stage B labels (licensed, optional):** **Port-to-foreign-port** flows by merging **BOL** (PIERS/Datamyne) with **Schedule K** ports; reconcile commodity/value using USA Trade as the controlling aggregate (to avoid double counting).

**F. Codebooks & validation.**  
- Maintain **Schedule D** (U.S. ports) and **Schedule K** (foreign ports) reference tables; audit against CBP/Census update notices.  
- Confirm the **Carrollton gage** datum adjustments and milepost for all clearance calculations.  
- Cross-check port-level totals against **USACE WCSC** port statistics and **BTS** rankings for sanity and drift.

**G. Data quality & governance.**  
- **Missingness:** flag sensors with downtime; prefer medians & robust smoothing for features; never fill labels.  
- **Licensing:** ensure BOL and AIS licensed data are stored in segregated layers; outputs for publication should **not** expose firm-level information (consistent with DataWeb/USA Trade aggregation rules).  
- **Reproducibility:** log exact query strings/date stamps (e.g., USGS iv/dv URLs), S3 object versions (NWM/Daymet), and bulletin IDs (NTNI/MSIB).

---

### 2.5 Paid databases you can license (with links)

- **S&P Global PIERS (BOL microdata; U.S. maritime imports/exports; foreign port fields).** Best-in-class coverage for **port-to-port** inference. [source](https://www.spglobal.com/marketintelligence/en/solutions/piers)  
- **Descartes Datamyne (BOL microdata; daily updates).** Competitive alternative with standardized parties/TEU fields. [source](https://www.datamyne.com/)  
- **Lloyd’s List Intelligence—Seasearcher (AIS + vessel registry intelligence).** Voyage and call confirmation for routing validation. [source](https://lloydslist.maritimeintelligence.informa.com/seasearcher)  
- **Spire Maritime (S-AIS + T-AIS APIs).** High refresh rates in blue-water; helpful for precise transit timing and congestion features. [source](https://spire.com/maritime/)  
- **Kpler / MarineTraffic data services.** Supplemental AIS-based port call and congestion analytics. [source](https://www.kpler.com/) [source](https://www.marinetraffic.com/)

*(Open alternatives for historical AIS include NOAA/BOEM MarineCadastre, but with lower temporal fidelity than commercial feeds.)* [source](https://marinecadastre.gov/ais/)

---

### 2.6 Important definitional/measurement notes we will adhere to

- **AHP (Above Head of Passes)** is the official river-mile datum for the Lower Mississippi; **Carrollton gage** is at **RM 102.8 AHP**. We will reference all river features in AHP.  
- **USA Trade Online** disseminates **port (Schedule D)** and **country** dimensions publicly; **foreign port of unlading** is a **Schedule K** code required in export filings but is not provided in public aggregates—hence the need for **PIERS/Datamyne** to recover **port-to-port** flows.  
- **BTS Port Performance** uses **WCSC** as the underlying port statistics authority and highlights Lock 27/low-water disruptions—useful as validation targets for Stage-A outputs.

---

### 2.7 What this setup enables (for the later sections)

With these data and transformations, Stage-A models learn **how much hydrology “translates” into river capacity frictions** (stages/clearances, restrictions, queueing), and Stage-B models then learn **how those frictions translate into monthly trade by partner** (and **by foreign port** when BOL is licensed). This architecture is aligned with how U.S. agencies produce and use these data—and is fully auditable against their definitions and reporting standards.

## 3) Methodology — Targets, Features, and Models

### Stage A — Hydrometeorology ⇒ Navigational/operational constraints

**Targets (daily, multi-horizon).** We forecast the operational signals that mariners and forecasters use around New Orleans: (i) **river stage at New Orleans (Carrollton gage; station NORL1, river mile 102.8 AHP)** in feet relative to LWRP/NAVD88, and (ii) **bridge air-gap at the Huey P. Long Bridge** (NOAA PORTS station 8762002), optionally extended to other spans when instrumented or when a documented clearance-vs-stage relationship exists. These are the canonical references for navigation on the Lower Mississippi and are also the basis for NWS/LMRFC hydrographs. [source](https://water.weather.gov/ahps2/hydrograph.php?wfo=lix&gage=norl1) [source](https://tidesandcurrents.noaa.gov/ports/index.html?port=lm)

**Features (daily).** We use three categories, aligned with multi-horizon forecasting practice:

1) **Observed-past exogenous drivers** \(x^{\text{obs}}\):  
   • **Daymet v4** gridded daily precipitation, temperature, radiation, vapor pressure (1-km), aggregated to major Mississippi sub-basins with lag embeddings; period 1980–most recent full year. [source](https://daymet.ornl.gov/)  
   • **National Water Model (NWM) retrospective v2.1** streamflow at selected reaches (hourly re-sampled to daily) for Ohio/Missouri/Upper Mississippi tributaries; 1979–2020. [source](https://registry.opendata.aws/nwm-archive/)  
   • Prior day stage at NORL1. [source](https://water.weather.gov/ahps2/hydrograph.php?wfo=lix&gage=norl1)

2) **Known-in-advance drivers** \(x^{\text{known}}\):  
   • Calendar, holiday, and “day-of-year” terms.  
   • **Official LMRFC 28-day guidance flags/values** where available (operational with short QPF horizon; longer-lead visuals currently experimental). [source](https://www.weather.gov/lmrfc/)

3) **Static metadata** \(s\): river-mile positions, datum offsets (NAVD88↔LWRP), and sensor/bridge identifiers. NORL1 station metadata provides river-mile and datum adjustments used to reconcile LWRP and NAVD88. [source](https://rivergages.mvr.usace.army.mil/)

**Shapes.** Let the encoder window be \(L_A\) days (e.g., 56) and the horizon be \(H_A\) days (e.g., 1–28). Using a batch of \(B\) training instances and \(Q_A\) targets (here \(Q_A\in\{1,2\}\) for stage and air-gap):

- Observed-past inputs: \(X^{\text{obs}}_A \in \mathbb{R}^{B\times L_A\times F^{\text{obs}}_A}\).  
- Known-future inputs: \(X^{\text{known}}_A \in \mathbb{R}^{B\times (L_A+H_A)\times F^{\text{known}}_A}\).  
- Static inputs: \(S_A \in \mathbb{R}^{B\times d_s}\).  
- Targets: \(Y_A \in \mathbb{R}^{B\times H_A\times Q_A}\).

**Model.** We use a **Temporal Fusion Transformer (TFT)** to learn \(f_A:\{X^{\text{obs}}_A,X^{\text{known}}_A,S_A\}\mapsto \widehat{Y}_A\) because it natively separates static covariates, observed-past covariates, and known-future covariates while providing attention-based variable attribution. We pair it with a global univariate baseline (**N-BEATS**) per target. [source](https://arxiv.org/abs/1912.09363) [source](https://arxiv.org/abs/1905.10437)

**Losses and uncertainty.** We train TFT/N-BEATS with **pinball loss** for quantiles \(\tau\in\{0.1,0.5,0.9\}\):  
\[
\mathcal{L}_{\tau}(y,\hat y)=\max\{\tau(y-\hat y),(\tau-1)(y-\hat y)\}.
\]  
We post-hoc **conformalize** the quantile forecasts on a rolling calibration set (size \(C\)) using **Conformalized Quantile Regression (CQR)**. If \(r_i = y_i - \hat q_{\tau}(x_i)\) on calibration samples \(i=1,\dots,C\), then the adjusted quantile at level \(\tau\) is
\[
\tilde q_{\tau}(x) \;=\; \hat q_{\tau}(x) \;+\; \mathrm{Quantile}_{1-\alpha}\{r_1,\dots,r_C\},
\]
yielding finite-sample marginal coverage \(1-\alpha\) under exchangeability.

---

### Stage B — Navigational constraints ⇒ Mid-river logistics (weekly)

**Targets (weekly, multi-task).** We predict the **downbound grain barge tonnage at Lock & Dam 27** (Upper Mississippi choke-point feeding the Lower Mississippi) and **grain barge unload counts in the New Orleans region**. Both series are curated openly by USDA AMS in the **Grain Transportation Report (GTR)** and on the **AgTransport** portal; **barge rate indices** (percent of tariff) at St. Louis and other corridors are used as exogenous context and, in some experiments, as auxiliary targets. [source](https://www.ams.usda.gov/services/transportation-analysis/gtr)

**Features (weekly).** We weekly-aggregate Stage-A outputs (e.g., median stage, 7-day changes, frequency of low-clearance hours) and join them with (i) barge **rate** indices (GTR Table 9), (ii) **lock/queue snapshots** from USACE **LPMS** via **Corps Locks**, and (iii) event flags derived from **USACE Navigation Bulletins (NTNI)** and **USCG MSIBs** in the New Orleans sector. [source](https://corpslocks.usace.army.mil/) [source](https://www.mvn.usace.army.mil/Missions/Navigation/Notices-to-Navigation-Interests/) [source](https://homeport.uscg.mil/)

**Shapes.** With weekly window \(L_B\) (e.g., 104 weeks) and horizon \(H_B\) (e.g., 1–8 weeks) over \(N_B\) river nodes/series (e.g., \(\{ \text{L27 tonnage}, \text{NOLA unloads}\}\Rightarrow N_B=2\)):

- Node-wise inputs at week \(t\): \(X_{B,t}\in\mathbb{R}^{N_B\times F_B}\); over a window: \(X^{(L)}_B\in\mathbb{R}^{L_B\times N_B\times F_B}\).  
- Targets: \(Y_B\in\mathbb{R}^{H_B\times N_B\times Q_B}\) (typically \(Q_B=1\) per node).  
- If we include additional upstream control points as nodes, \(N_B\) expands naturally; the adjacency is defined by the river network.

**Model.** We model river-propagated impacts using a **Diffusion Convolutional Recurrent Neural Network (DCRNN)** with a directed adjacency \(A\in\{0,1\}^{N_B\times N_B}\) connecting upstream to downstream nodes. The diffusion convolution at order \(K\) uses bidirectional random walks:
\[
\textstyle \Gamma(X) \;=\; \sum_{k=0}^{K-1}\!\Big((D_O^{-1}A)^k X \Theta_k^{(f)} \;+\; (D_I^{-1}A^\top)^k X \Theta_k^{(b)}\Big),
\]
where \(D_O\) and \(D_I\) are out- and in-degree matrices, \(\Theta^{(f)},\Theta^{(b)}\) are learnable weights, and the recurrent cell handles temporal evolution over \(L_B\) steps. We ensemble DCRNN with a global **TFT** at weekly cadence to capture non-graph covariates (rates, seasonality) and retain interpretability on variable importance.

**Losses and uncertainty.** As in Stage A, we train with horizon-wise pinball loss and overlay **CQR** for calibrated weekly intervals. For rate indices treated as exogenous only, we monitor Granger-style lag responses but do not train on them as targets unless explicitly included in multi-task learning.

---

### Stage C — Mid-river logistics ⇒ International trade through Port of New Orleans (monthly)

**Targets (monthly, multi-output).** We forecast customs-port (**Schedule D**) **monthly export/import value and weight** for the **Port of New Orleans** and, in the extended setup, **foreign port of unlading** (**Schedule K**) and **country** splits by commodity. The Census Bureau documents that **port data are available monthly** via USA Trade Online (2003-present for ports; API coverage since 2013 for many endpoints). Schedule D defines U.S. customs districts/ports; **foreign port of unlading must be reported in Schedule K codes**, maintained by CBP/USACE. [source](https://usatrade.census.gov/) [source](https://www.census.gov/foreign-trade/reference/codes/index.html)

**Features (monthly).** We aggregate Stage-B outputs to calendar months (sums/means), include barge rate levels, and optionally add **AIS vessel-call features** near Southwest Pass/Lower Mississippi anchorages derived from **MarineCadastre** (USCG NAIS-sourced coastal AIS, 2009+; land-based receivers). [source](https://marinecadastre.gov/ais/)

**Shapes and hierarchy.** Let \(T_C\) months in sample and \(M\) bottom-level series (e.g., commodity × foreign port pairs). Stack them column-wise:
\[
Y_C \in \mathbb{R}^{T_C \times M},\quad X_C \in \mathbb{R}^{T_C \times F_C}.
\]
Let \(S\in\{0,1\}^{m\times M}\) be the **summing matrix** mapping bottom-level series to all nodes in the hierarchy (e.g., bottom → country → total). Base forecasts \(\widehat y\in\mathbb{R}^{M}\) for a given month are reconciled using **MinT**:
\[
\tilde y \;=\; S\big(S^\top W^{-1}S\big)^{-1}S^\top W^{-1}\widehat y,
\]
where \(W\) is a shrinkage estimate of base-forecast error covariance. This enforces **coherence** across port/foreign-port/country totals and typically improves accuracy.

**Model.** We use a **multi-output TFT** at monthly cadence to produce \(\widehat Y_C\in\mathbb{R}^{H_C\times M}\) for horizon \(H_C\) (e.g., 1–6 months). For sparse series, we add per-series **N-BEATS** univariate baselines and combine via linear stacking before **MinT** reconciliation. The training loss is a weighted sum of (i) sum-normalized pinball losses across series and horizons and (ii) a **coherency penalty** \(\lambda\lVert S\widehat y - \widehat g\rVert_2^2\) on pre-reconciliation predictions, where \(\widehat g\) are model outputs at aggregated nodes, to bias the network toward coherent structure before post-hoc MinT.

**Definitions guarded in labeling.** We keep **Schedule D (customs port)** separate from **Schedule K (foreign port)** in the schema and only reconcile within consistent hierarchies (e.g., K→country totals or D→district totals) to avoid spurious cross-geography learning; the Census glossary specifies that “foreign port of unlading” must be reported **in Schedule K terms**. [source](https://www.census.gov/foreign-trade/reference/codes/index.html)

---

## Implementation-ready tensor summaries (concise)

**Stage A (daily TFT/N-BEATS):**  
Batch \(B\), window \(L_A\), horizon \(H_A\), targets \(Q_A\).  
\(\;X^{\text{obs}}_A\in\mathbb{R}^{B\times L_A\times F^{\text{obs}}_A}, \;X^{\text{known}}_A\in\mathbb{R}^{B\times(L_A+H_A)\times F^{\text{known}}_A}, \;S_A\in\mathbb{R}^{B\times d_s}, \;Y_A\in\mathbb{R}^{B\times H_A\times Q_A}.\)

**Stage B (weekly DCRNN+TFT):**  
\(A\in\{0,1\}^{N_B\times N_B}\) (river network); \(X^{(L)}_B\in\mathbb{R}^{L_B\times N_B\times F_B}; \;Y_B\in\mathbb{R}^{H_B\times N_B\times Q_B}.\)

**Stage C (monthly TFT + MinT):**  
\(X_C\in\mathbb{R}^{T_C\times F_C};\;Y_C\in\mathbb{R}^{T_C\times M};\) bottom-level base forecasts \(\widehat y\in\mathbb{R}^{M}\) reconciled via MinT with \(S\in\{0,1\}^{m\times M}\).






( version 2)

## 3) Methodology

This project is designed as a three-stage, end-to-end forecasting pipeline that links hydrometeorology on the Mississippi River to near-river logistics and finally to international trade flows through the Port of New Orleans. Each stage produces supervised targets that become structured inputs to the next stage. The model stack emphasizes (i) spatiotemporal learning with physically meaningful features (river miles, gage/bridge geometry, lock positions), (ii) multi-horizon forecasts with calibrated uncertainty, and (iii) reconciliation between daily/weekly logistics signals and monthly customs statistics to ensure operational usefulness and statistical coherence.

---

### Stage A — Hydrometeorology → Navigational/operational constraints

We first translate climate and hydrologic conditions into **navigational constraints** that shippers and pilots actually act on: stage at key river miles, forecast hydrographs, channel clearance under the two New Orleans bridges, salinity intrusion risk, and inferred draft restrictions.

**Targets.**  
Daily nowcasts/forecasts (1–28 day horizons) for: (a) Mississippi River stage at New Orleans (Carrollton gage, RM 102.8 AHP), (b) bridge air gap (Huey P. Long and Crescent City Connection), and (c) probabilistic *low-flow risk indices* for saltwater wedge intrusion and associated water-quality constraints. These targets are derived from authoritative series and guidance: USACE/NOAA stage at the Carrollton gage (NORL1) and LWRP conversion, NOAA PORTS air-gap sensors, and LMRFC official hydrographs for the Lower Mississippi. We also incorporate retrospective and near-real-time **National Water Model (NWM)** guidance as exogenous covariates to improve lead times beyond the official horizon.

**Features.**  
(i) **Hydromet drivers**: daily precipitation/temperature and derived indicators from Daymet v4 and/or PRISM (1 km–800 m), aggregated to the Ohio/Missouri/Upper Mississippi sub-basins with flow-timelagged embeddings; (ii) **NWM v2.1 retrospective** (1979–2020) and available v3.0 guidance streams to supply gridded streamflow/soil-moisture covariates; (iii) **Geometrics**: gage-to-bridge clearance mapping formulas (e.g., “air gap = reference minus gage stage” for bridges where that relationship is documented) and LWRP/NAVD88 datum adjustments.

**Models.**  
For A→constraints we employ a **Temporal Fusion Transformer (TFT)** for multi-horizon forecasting, because it natively handles mixed covariates (static geography, known-in-advance calendars, observed exogenous series) and provides attention-based interpretability on driver importance. We run TFT alongside a strong univariate global baseline (**N-BEATS**) and a graph-aware model for spatial hydrology (**DCRNN**) using the river network as a directed graph; the ensemble reduces variance and improves robustness under regime shifts (e.g., drought years).

**Uncertainty.**  
We wrap point forecasts with **conformalized quantile regression** to produce distribution-free, finite-sample coverage intervals at each horizon; in backtests we monitor conditional coverage during drought/low-flow regimes separately from normal regimes.

**Why this is reliable.**  
LMRFC hydrographs are the operational benchmark for the Lower Mississippi; NWM adds gridded physics-based guidance at millions of reaches; PORTS air-gap sensors reflect real clearance conditions used by pilots. Conditioning on these authoritative signals lets the learning system translate weather into **operational constraints** that logistics decisions actually depend on.

---

### Stage B — Navigational/operational constraints → Mid-river logistics & port-approach activity

Stage B predicts **through-river logistics** that respond quickly to hydrology: southbound grain barge tonnage at Lock & Dam 27 (proxy for Lower Mississippi throughput), empty barge returns, barge unloads in the New Orleans region, lock delay/queue metrics, and dynamic bridge-clearance windows.

**Targets.**  
Weekly totals for: downbound grain tonnage through Lock 27; number of barges unloaded in the New Orleans region; upbound empty barges; and lock/queue delay indicators from USACE LPMS/Corps Locks. These measures are widely used as timely indicators of Mississippi logistics and are curated weekly by USDA AMS in the Grain Transportation Report (with public spreadsheets and dashboards).

**Features.**  
(i) Stage-A outputs (clearance/draft risk indices; bridge air gaps; forecast stages at New Orleans and upstream control points), (ii) **river-mile graph features** encoding distances, lock positions, tributary confluences; (iii) **calendar/seasonal** drivers (harvest windows); (iv) market context (diesel price, indicative barge rates published by AMS since Oct-2024).

**Models.**  
We frame B as a **multi-task spatiotemporal forecasting** problem. A **graph recurrent model (DCRNN)** captures propagation from upstream constraints to downstream tonnage/queues over the river network; a **TFT** captures nonlinear interactions and known-in-advance seasonality. We also train direct global baselines (N-BEATS) for each target as variance-reducing components of an ensemble.

**Interpretation & validation.**  
We use attention/feature attributions (TFT) and impulse-response probes (graph model) to ensure that predicted barge flows respond plausibly to stage/clearance changes. External case studies (e.g., 2022–2024 low-water spikes with elevated barge rates and reduced drafts) are used as out-of-sample stress tests.

---

### Stage C — Mid-river logistics → International trade flows at New Orleans (port-level and country/port pairs)

Stage C maps weekly logistics proxies into **monthly trade outcomes** at the Port of New Orleans—both at the **Schedule D customs-port level** (e.g., New Orleans, code 2002) and at **foreign Schedule K ports/countries** (e.g., Santos, 35141; Rotterdam, 42182), focusing on values and tonnages for river-relevant commodities (grains, fertilizers, petroleum products, steel). We explicitly account for the difference between **customs geography** (Schedule D) and **foreign port geography** (Schedule K) in the Census methodology to avoid mis-assignment of flows.

**Targets & labels.**  
Monthly export/import values and weights by commodity, foreign port (K-code), and country, sourced from USA Trade Online / Census APIs (with appropriate licensing) and reconciled against Waterborne Commerce Statistics Center (WCSC) port-level statistics to benchmark physical throughput for Lower Mississippi ports.

**Features.**  
(i) Stage-B outputs lagged/aggregated to calendar months; (ii) river-operational features (bridge clearance availability windows, lock delays); (iii) external demand controls (e.g., seasonality dummies; optional commodity price indicators if included as exogenous features).

**Models.**  
We cast C as **multi-horizon, multi-output forecasting** with **hierarchical reconciliation**: the model predicts at granular (foreign port × commodity) and aggregates to country and total; we then apply **MinT reconciliation** to enforce additivity and improve statistical efficiency across the hierarchy. The base learner is TFT (for covariate richness) with N-BEATS as a fast univariate complement per series; we train a shared representation across commodities to share statistical strength for sparse series.

**Handling definitional pitfalls.**  
We encode **port-definition constraints** directly: “Port of unlading/export” in Census data reflects customs processing geography and shipping mode rules; “foreign port of unlading” must be reported in **Schedule K** terms; these are not the same as USACE physical terminal definitions. During labeling we keep fields separate (D vs K) and add mapping tables so the model never learns spurious equivalences.

---

### Cross-stage integration and training protocol

We train the pipeline **sequentially with feedback**. Stage-A and Stage-B are trained first (daily→weekly). Stage-C consumes rolling monthly aggregates of Stage-B predictions and their calibrated uncertainty. For end-to-end optimization we fine-tune with a **multi-task loss** that includes (i) weekly logistics accuracy and (ii) monthly trade accuracy, with penalties for violating hierarchical additivity.

**Temporal cross-validation.**  
We use **rolling-origin (prequential) evaluation** with expanding windows. For A and B we roll at weekly cadence; for C at monthly cadence. We report scale-free errors such as **MASE/RMSSE** and horizon-wise quantile loss; this is the recommended practice for non-stationary, multi-series forecasting. We stratify folds to ensure that known stress periods (e.g., 2022–2024 low-water) appear in test sets.

**Uncertainty & risk communication.**  
At every stage we output central forecasts and **conformal** 50/80/90% bands. For C, we propagate upstream uncertainty by Monte-Carlo draws from Stage-B conformal bands into monthly aggregates before applying conformalization at the trade layer, preserving conservative coverage at the business decision point.

**Evaluation under events.**  
We run **event-study diagnostics** around documented saltwater wedge episodes and bridge-clearance updates (2023–2024) to check that predicted logistics and trade react in the correct direction with realistic lags. We identify event windows from USACE saltwater-wedge bulletins and NOAA PORTS/Coast Pilot update notices.

---

### Feature engineering details (selected)

We standardize **river-mile geometry** and datum conversions so that physical quantities enter the model consistently. The Carrollton gage (RM 102.8 AHP) is treated as the reference for New Orleans; when PORTS air-gap is available we use it directly, and when not, we compute clearance via documented gage formulas or bridge reference points, keeping both **observed** and **derived** features as separate channels. We keep **AHP/NAVD88/LWRP** adjustments explicit as static metadata.

Hydromet grids (Daymet/PRISM) are **basin-weighted and time-lagged** using simple hydrologic travel-time kernels to reflect that rainfall in the Ohio/Missouri basins moves downstream into the Lower Mississippi with delay. We include **NWM** streamflow/soil-moisture at upstream reaches as exogenous features to extend lead time (v2.1 retrospective for backfilling, v3.0 guidance where available).

For **weekly logistics**, we transform the AMS GTR spreadsheets/dashboards into tidy series (Lock 27 downbound tonnage; New Orleans region barge unloads; upbound empties) and align their week-ending dates to the daily Stage-A signals before aggregation. We add **rate** signals (e.g., per-ton barge rates now published by AMS) as exogenous covariates to help the model disambiguate supply vs. constraint-driven slowdowns.

For **monthly trade**, we preserve **Schedule D** (U.S. customs port) and **Schedule K** (foreign port) as separate dimensions; we rely on the Census technical documentation to interpret “port of unlading/export” and on WCSC for physical port-throughput benchmarking.

---

### Robustness, bias control, and ablations

We anticipate **distribution shifts** from infrastructure and policy (e.g., updated bridge clearance references; channel-deepening’s effect on salinity intrusion risk; trade policy shocks). We therefore:  
• Perform **time-localized recalibration** of conformal intervals during drought/low-flow regimes;  
• Run **ablation studies** removing each major exogenous block (hydromet, NWM, stage/air-gap, rates) to quantify marginal value;  
• Stress-test on documented low-water episodes with known operational impacts (reduced barge drafts, spikes in rates, queueing), verifying that Stage-B/C predictions reproduce the direction/magnitude of effects observed in AMS and academic case studies.

---

### Data, access, and licensing (Stage-C enhancement with commercial sources)

Where finer-than-Census granularity is required (e.g., **bill-of-lading–level flows by vessel/foreign terminal**), we will optionally incorporate **paid** datasets as *auxiliary features* or labels for specific experiments:

* **S&P Global PIERS** (U.S. waterborne bills of lading; company names normalized; near-real-time ingestion) — used to build vessel-level panels and foreign **Schedule K** port mappings and to distinguish river-borne exports via New Orleans from rail/truck transloads.  
* **Descartes Datamyne** (global BOL research platform) — used to cross-validate PIERS entity resolution and to augment shipper/consignee networks for commodities of interest.  
* **Lloyd’s List Intelligence Seasearcher** or **Spire Maritime Historical AIS** — used to generate independent vessel traffic features (port calls, dwell, AIS-based congestion) near the Lower Mississippi anchorages and Southwest Pass. These improve near-term trade nowcasts and help explain divergences between logistics and customs flows.

---

### Implementation notes

**Granularity & reconciliation.** We train A at **daily** resolution, B at **weekly**, and C at **monthly**. We aggregate upstream predictions to the next stage and apply **MinT** reconciliation at the trade hierarchy (foreign-port → country → total) to ensure coherence across levels and across commodity families.

**Losses & metrics.** Each stage optimizes horizon-aware pinball loss (for quantiles) plus MASE/RMSSE-aligned objectives for point accuracy; model selection relies on rolling-origin validation to avoid look-ahead bias.

**Interpretability.** We will log TFT variable importances and attention maps, keep SHAP-style local explanations for operational variables (e.g., which combination of low stage and air-gap windows suppressed barge unloads), and record counterfactual probes (e.g., “+1 ft at Carrollton for 7 days”) to demonstrate physical plausibility against USACE/NOAA constraints.

**Event-linked validation.** For saltwater wedge episodes, we pair Stage-A salinity-risk indices with USACE timeline bulletins; for bridge-clearance updates, we reference NOAA PORTS/Coast Pilot change notices to ensure predicted clearance windows and vessel passage patterns adjust accordingly.

---

### Why this three-stage, DL-first approach is the most defensible

1) It **grounds** learning in the same operational signals the river community uses (LMRFC hydrographs, PORTS air-gaps, USACE gages) rather than in generic weather features, reducing spurious correlations and improving face validity.  
2) It **matches decision cadence**: shippers act weekly on draft/clearance constraints (Stage-B), while trade statistics are released monthly (Stage-C). Reconciliation ensures forecasts add up across levels and horizons.  
3) It **handles uncertainty rigorously** via conformal prediction, giving calibrated intervals at every horizon — crucial for operational planning under drought/low-water regimes.  
4) It is **data-expandable**: where public sources are too coarse, BOL-level and AIS feeds can be added cleanly as auxiliary features without changing the core physical-to-logistics structure.

This methodology yields a transparent chain from rain and river physics to barge flows and, ultimately, to export volumes and values — exactly the causal pathway your research question aims to quantify.

