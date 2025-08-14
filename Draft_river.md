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
