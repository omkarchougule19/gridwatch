# GridWatch: project status and questions for the TA

## Where the project stands
- **Goal:** GridWatch gives grid operators and energy traders a day's warning of power-grid stress. Every afternoon at 2 PM it predicts, for each hour of the next day, the probability of a real-time price spike in **ERCOT** (Texas) and **MISO** (Midwest).
- **Data:** all collected by the team from public sources that were never packaged as ML datasets.
  - EIA-930 hourly load and the operator's own day-ahead load forecast.
  - ERCOT and MISO hub prices.
  - Archived NOAA GFS weather forecasts *as they were issued*, so no future weather leaks into training.
  - In total that is about 49k hours for ERCOT (2021–2026) and 34k for MISO (2023–2026).
- **Labels:** the primary label is a price spike, meaning a price at or above the **99th percentile of training-period prices** for that region (ERCOT $329/MWh, MISO $111/MWh). A flat $1,000 gave MISO zero training events. Operator alerts are the secondary label (hand-curated).
- **Models (following the course):** a rule baseline, kNN, logistic regression, Naive Bayes, random forest, XGBoost, an MLP, and Chronos-2 (zero-shot and fine-tuned).
- **Evaluation:**
  - Time-based train (2021–Jun 2024), val (Jul 2024–Jun 2025) and test (Jul 2025–Oct 2026).
  - Metrics: PR-AUC, precision/recall at a threshold tuned on val, event-level recall ("did we warn in the 24 hours before?"), Brier score.
  - Significance: a paired day-block bootstrap.
  - Ablations by dropping feature groups.
- **Finding so far:** the two regions shift in **opposite directions**.
  - MISO's test period has about 3.5× more spike hours than training.
  - ERCOT spikes nearly vanished after 2023 (132 spike hours in 2023 vs 8 in 2025), leaving only about 11 val and about 11 test events at p99.

## First results (validation year, Jul 2024–Jun 2025)
| PR-AUC (higher is better) | ERCOT (spikes in 0.2% of hours) | MISO (spikes in 2.0% of hours) |
|---|---|---|
| Rule baseline | 0.009 | 0.059 |
| Logistic regression, **without** tomorrow's DA price | 0.012 | 0.119 |
| **Logistic regression, with tomorrow's DA price** | **0.077** | **0.322** |
| XGBoost, with DA price | 0.062 | 0.050 |
| MLP, with DA price | 0.059 | 0.135 |

- **The biggest gain came from tomorrow's day-ahead price**, which is published before our 2 PM issue time: about 3–6× higher PR-AUC.
- **Logistic regression beats the tree models.** In MISO, the day-ahead price grew more informative and moved beyond the training range over time:
  - the share of spikes the DA market already priced in went from 6% in train to 45% in test;
  - 177 test hours had DA prices above anything seen in training.
  
  Trees can't extrapolate beyond their training range; a linear model can.
- **Precision is still low:** about 0.12 at 70% recall for MISO, against a target of 40–50%.

## Abbreviations and terms
| Term | Meaning |
|---|---|
| **ERCOT** | Electric Reliability Council of Texas. It runs most of the Texas power grid and its wholesale electricity market. |
| **MISO** | Midcontinent Independent System Operator. It runs the grid and market across 15 central US states (including Illinois) and Manitoba. |
| **ISO / RTO** | Independent System Operator / Regional Transmission Organization. These are the grid operators, such as ERCOT and MISO, that balance supply and demand and run electricity markets. |
| **BA** | Balancing Authority. The entity that keeps supply and demand balanced in an area and reports load to EIA. |
| **EIA / EIA-930** | U.S. Energy Information Administration. Form EIA-930 is its hourly report of each BA's demand and day-ahead demand forecast. |
| **NOAA / GFS** | National Oceanic and Atmospheric Administration / Global Forecast System, NOAA's main weather forecast model. |
| **Load** | Electricity demand, in megawatts (MW). |
| **MWh, $/MWh** | Megawatt-hour, the unit of energy. Wholesale prices are quoted in dollars per MWh, usually about $20–50. |
| **Real-time (RT) price** | The price set close to the moment power is delivered (5–15 min intervals, averaged to hourly). Spikes show up here. |
| **Day-ahead (DA) price / DAM** | The price set in the Day-Ahead Market the afternoon before delivery. ERCOT posts it about 13:30 Central; MISO about 12:30 Central (to be verified). |
| **Hub (HB_HUBAVG, ILLINOIS.HUB)** | A trading point whose price summarizes a region. We use ERCOT's hub average and MISO's Illinois hub. |
| **LMP / SPP** | Locational Marginal Price (MISO) / Settlement Point Price (ERCOT). Both mean the electricity price at a location. |
| **EEA** | Energy Emergency Alert. ERCOT's emergency levels when supply runs short. |
| **Issue time** | When GridWatch makes its daily forecast: 14:00 local time on the day before (D-1), covering all 24 hours of day D. |
| **p99 threshold** | The 99th percentile of training-period prices. Hours at or above it count as a spike. |
| **PR-AUC** | Area under the precision-recall curve (average precision). Our main metric, because spikes are rare (0.2–2% of hours). |
| **Event-level recall** | The share of spike events that got at least one warning before they started. |
| **Brier score** | Mean squared error of the predicted probabilities, which measures calibration. |
| **Block bootstrap** | Resampling whole days to get confidence intervals and p-values that respect hour-to-hour correlation. |
| **Chronos-2** | Amazon's pretrained time-series foundation model. *Zero-shot* means used as-is; *fine-tuned* means further trained on our data. |
| **Winter Storm Uri** | The February 2021 storm that caused massive Texas blackouts and $9,000/MWh prices. |

## Questions for the TA
1. **Thresholds:** your feedback said to use the price threshold as the primary label. Is a **per-region percentile computed on the training years** (p99) acceptable instead of a fixed $/MWh value?
2. **Too few ERCOT events:** ERCOT has only about 11 test events at p99. Which would you prefer?
   - (a) lower ERCOT to p98 (about 28 val / 34 test events);
   - (b) add regions such as SPP or PJM;
   - (c) pool the regions for event-level metrics;
   - (d) keep p99 and report the regime change as a finding.
3. **Distribution shift:** the test period differs a lot from training in both regions, in opposite directions. Should we adjust the splits (for example, a rolling or expanding-window evaluation), or is it better to keep a single fixed test period and analyze the shift?
4. **Training window:** archived day-ahead forecasts only start in April 2021, so training excludes Winter Storm Uri (Feb 2021). Is that acceptable, or is it worth extra effort to recover older NOAA forecast archives?
5. **Test set requirement:** the test labels come from public price records over a continuous period, plus hand-picked extreme-weather windows. Does that meet the "evaluation set designed by the team" rule?
6. **Significance testing:** is a paired day-block bootstrap of PR-AUC enough, or should event-level recall also get confidence intervals or tests?
7. **Scope:** is comparing seven classic models plus Chronos the right breadth, or would you rather see fewer models with deeper ablations (features, label definition, forecast lead time)?
8. **Val reuse:** we use the validation year both to calibrate probabilities and to choose the alert threshold, and test is never touched. Is that acceptable, or should calibration use a separate slice?
9. **Chronos:** is univariate price forecasting enough for the foundation-model comparison, or should we fine-tune with covariates (load forecast, temperature), which Chronos-2 supports?
10. **Day-ahead price as a feature:** tomorrow's DA price is public before our issue time, and it's what traders and operators really see. Is it fair to include? Since it dominates the results, should we frame the task as "predict real-time spikes beyond what the DA market already shows"?
11. **Precision target:** our proposal targeted 40–50% precision at 70–80% recall. Current results are far below that. Is it acceptable to report this honestly, with an analysis of why (drift, rare events), rather than chase the target?
