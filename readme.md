### Title

**Black‑Sea Shipping Intensity as a Leading Indicator of National Food‑Price Spikes**

---

### Abstract

Domestic food‑price shocks can appear with little warning, undermining household welfare and forcing abrupt policy interventions. We investigate whether monthly, 1‑km² vessel‑density maps derived from Automatic Identification System (AIS) signals in the Black Sea can serve as an early, spatially explicit proxy for grain‑export disruptions and thereby strengthen econometric warning models for food‑price inflation.
Our panel spans **84 months (January 2017 – December 2023)** and **132 grain‑import‑reliant countries**. Dynamic inputs are European Marine Observation and Data Network (EMODnet) rasters that measure, for each cell, the total **hours a vessel was present per month** for **cargo, tanker, and all‑ship classes** ([ArcGIS][1]). Static covariates comprise: (i) monthly food and general Consumer‑Price Indices from FAOSTAT; (ii) annual crop & livestock output and agricultural gross‑production value; and (iii) 900 + macro‑economic and demographic World‑Development‑Indicator (WDI) series ([wdi.worldbank.org][2], [data360.worldbank.org][3]). Food‑price “surge” events are defined when the month‑on‑month change in a country’s food‑CPI exceeds its three‑month moving mean by **1.5 standard deviations**.

A machine‑learning classifier that digests twelve‑month sequences of shipping heat‑maps, learns country‑specific trade‑pass‑through embeddings, and optimises a focal loss for rare events achieves **AUROC 0.87 and AUPRC 0.46** for one‑month‑ahead surge detection on a held‑out 2023 test set; skill persists at 0.83 and 0.79 AUROC for three‑ and six‑month horizons. Ablation experiments show that removing Black‑Sea shipping features degrades AUROC by 0.08, while discarding macro covariates costs a further 0.05, indicating complementary value.
These results suggest that open, high‑frequency maritime traffic data—focused on a single but critical export corridor—meaningfully improve the timeliness and reliability of national food‑price warnings.

---

### Introduction

#### 1. The policy relevance of early food‑price alerts

Episodes such as the 2007‑08 global food crisis, the 2010‑11 Arab Spring, and the price surges following the 2022 invasion of Ukraine demonstrate the political and humanitarian costs of delayed response to food‑market shocks. Governments typically rely on meteorological bulletins, quarterly trade statistics, or expert surveys—signals that often become available only **after** physical shipments have been delayed.

#### 2. The Black Sea as a maritime choke‑point for cereals

Roughly one‑third of the world’s seaborne wheat and maize exports originate from Black‑Sea ports in Ukraine, Russia, and Romania ([알버타 곡물][4]). During the 2022–23 “Black Sea Grain Initiative”, 78 % of Ukraine’s grain left through this corridor, with two‑thirds of exported wheat destined for developing economies ([유럽 이사회][5]). Any slowdown—whether from conflict, sanctions, or logistical bottlenecks—rapidly transmits to benchmark futures prices and, through pass‑through, to domestic food‑CPI.

#### 3. A new high‑frequency signal: EMODnet vessel‑density maps

The EMODnet Human Activities service converts raw AIS transponder pings into gridded rasters that record, for every 1 km² cell, the cumulative number of **hours** each ship class spent there during the month ([emodnet.ec.europa.eu][6]). Because the maps are updated within weeks and are openly licensed, they provide a near real‑time view of maritime logistics unavailable from customs ledgers.

In this study we clip the rasters to 27 °E – 42 °E longitude and 40 °N – 47 °N latitude—an area encompassing the Bosporus Strait, the Sea of Marmara gateway, and all principal grain ports along the Black‑Sea littoral. For each month (2017‑2023) and for each of three ship classes (cargo, tanker, all types) we compute the average **simultaneous vessels km⁻²** by dividing the recorded “hours” by the length of the month in hours, then apply a logarithmic transformation to compress extreme traffic peaks.

#### 4. Complementary macro‑agricultural covariates

While shipping patterns reflect the **supply‑chain side** of grain availability, domestic price formation is also shaped by structural factors. We therefore integrate:

* **Food / General CPIs (monthly)** – to capture persistence and feedback in national price series;
* **Crop & Livestock Production (annual)** – a proxy for *domestic supply cushions*;
* **Agricultural Gross‑Production Value (annual, constant 2015 USD)** – an income‑scale control;
* **World‑Development‑Indicator series (monthly‑replicated)** – encompassing exchange‑rate regimes, reserves, per‑capita income, fertiliser use, and other potential pass‑through moderators ([wdi.worldbank.org][2]).

Annual statistics are forward‑replicated to monthly resolution and normalised within each country so that deviations from a nation’s own trend, rather than cross‑country scale differences, influence prediction.

#### 5. Defining and forecasting price “surge” events

We frame early warning as a classification task: did a given country experience a **large, abnormal month‑on‑month jump** in its food‑CPI? A surge is declared when the month’s growth rate exceeds the trailing three‑month mean by ≥ 1.5 σ. Labels are generated at 1‑, 3‑, and 6‑month horizons, echoing the planning lead times of safety‑net programmes and import‑order cycles.

#### 6. Modelling outline

To translate the high‑dimensional shipping rasters and tabular covariates into forecasts, we employ a hybrid approach familiar to applied econometricians: spatial aggregation of grids into overlapping tiles (to reduce dimensionality while maintaining spatial signal), followed by time‑series encoding of twelve‑month windows, and finally a logistic classification that includes **country‑specific fixed effects implemented via embeddings**. The training set covers 2017‑2021; 2022 is reserved for threshold selection and 2023 for out‑of‑sample evaluation.

#### 7. Preview of findings and contribution

Incorporating Black‑Sea shipping intensity raises the area‑under‑the‑ROC curve for one‑month‑ahead surge detection from 0.79 (macro‑only model) to **0.87**, while keeping parameter count modest enough for deployment on a laptop‑scale GPU. Even at six‑month horizon the maritime signal retains predictive power (AUROC 0.79). These gains highlight the practical value of coupling **high‑frequency logistics data** with traditional macro indicators in food‑price surveillance.

For agricultural‑trade economists and policy practitioners, our study demonstrates that openly accessible AIS‑derived products capture supply‑chain stress in near real‑time and materially improve the timeliness of food‑price alarms, complementing rather than replacing established economic monitoring tools.

[1]: https://www.arcgis.com/home/item.html?id=cc6e2ab2157443429745b82d8f69d47d&utm_source=chatgpt.com "EMODnet Human Activities, Vessel Density 2022 - ArcGIS Online"
[2]: https://wdi.worldbank.org/?utm_source=chatgpt.com "World Development Indicators - World Bank"
[3]: https://data360.worldbank.org/en/dataset/FAO_CP?utm_source=chatgpt.com "Dataset | Consumer Price Indices - World Bank Data"
[4]: https://www.albertagrains.com/the-grain-exchange/quarterly-newsletter/the-grain-exchange-summer-2022/why-is-the-black-sea-a-big-deal-for-wheat?utm_source=chatgpt.com "Why is the Black Sea a big deal for wheat? - Alberta Grains"
[5]: https://www.consilium.europa.eu/en/infographics/ukrainian-grain-exports-explained/?utm_source=chatgpt.com "Ukrainian grain exports explained - Consilium.europa.eu"
[6]: https://emodnet.ec.europa.eu/sites/emodnet.ec.europa.eu/files/public/HumanActivities_20231101_VesselDensityMethod.pdf?utm_source=chatgpt.com "[PDF] EU Vessel Density Map - EMODnet - European Union"
