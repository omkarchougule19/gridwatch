# GridWatch: Day-Ahead Early Warning for Power-Price Spikes in ERCOT and MISO

*CS441 Applied Machine Learning, Fall 2026, final project report (DRAFT)*
*Team: [names]. Code: https://github.com/omkarchougule19/gridwatch*

> **Draft notes:** numbers come from `reports/metrics_y_price_final.csv` and `reports/significance_y_price_final.csv` unless stated otherwise. `[TODO]` marks things the team must fill in or check. Figures are in `docs/visuals/` and `reports/figures/`.

---

## Abstract
We build GridWatch, a system that issues a warning every day at 2 PM Central, giving for each hour of the next day the probability of a real-time electricity price spike in two US grids:
- **ERCOT** (Texas);
- **MISO** (the central US).

A spike is an hour at or above the region's 99th percentile of training-period prices ($329/MWh in ERCOT, $111/MWh in MISO).

**Data:** we assembled about 49k (ERCOT) and 34k (MISO) hours from public sources: grid load and load forecasts, hub prices, and archived weather forecasts exactly as they were issued, so that no future information leaks into the inputs.

**Models:** we compared eight course methods, from a rule baseline to the Chronos-2 time-series foundation model, under one protocol:
- tuning on 2021–mid-2024;
- calibration on the following year;
- a single evaluation on a held-out 16-month test period and on team-designed extreme-weather windows.

**Results:**
- In MISO, logistic regression is best (test PR-AUC 0.34 [0.25, 0.45], about 8.5× the 4% base rate). It is significantly better than the rule baseline, the tree models, Naive Bayes and Chronos-2 (paired day-block bootstrap, p ≤ 0.002), but not significantly better than an MLP.
- Ablations show that most of the signal comes from tomorrow's day-ahead market price.
- The model catches nearly all spike hours of headline events such as Winter Storm Fern, but only 62% of "surprise" spikes that the day-ahead market did not anticipate.
- ERCOT's spikes almost vanished after 2023, consistent with its battery-storage build-out, leaving too few test events (11) to separate models.

---

## 1. Problem (rubric: 15)
**Motivation.** Wholesale electricity prices are usually $20–50/MWh but can jump to thousands during heat waves, cold snaps or supply shortfalls. Advance warning lets:
- grid operators line up reserves and issue conservation appeals;
- large consumers shift their load;
- traders hedge.

**Task definition.**
- **Issue time:** 14:00 local (America/Chicago) on day D-1.
- **Output:** for each of the 24 hours of day D, P(spike), so the lead time is 10–33 hours.
- **Label (primary):** spike = real-time hub price ≥ the region's 99th percentile of **training** prices.
  - We use a per-region percentile rather than a fixed $1,000/MWh because MISO had zero training hours above $1,000.
  - The percentile is computed on train only, so the label never depends on future data.
- **Decision:** an alert is raised when P(spike) ≥ an alert threshold tuned on the validation year to reach at least 70% recall.

**Why this is hard:**
- Spikes are rare (0.2% of ERCOT test hours, 4% of MISO test hours).
- Spikes cluster into multi-hour events.
- The market itself changes over time (section 4.5).

**Proposal targets:** 70–80% recall at 40–50% precision. We report honestly where we land (section 4.2).

## 2. Data (rubric: 40)
The full pipeline is in `src/gridwatch/data/ingest.py`. None of these sources is packaged as an ML dataset.

| Source | What | Why |
|---|---|---|
| EIA-930 (U.S. Energy Information Administration) [1] | Hourly balancing-authority load and the operator's **day-ahead load forecast** | Demand and the operator's own expectation |
| ERCOT (hub average HB_HUBAVG) and MISO (ILLINOIS.HUB) market data, via `gridstatus` [2] | Hourly real-time and day-ahead prices | Label and price features |
| NOAA GFS through the Open-Meteo *Previous Runs* archive [3] | Temperature forecasts **as issued the day before** (`temperature_2m_previous_day1`) for 4 ERCOT and 5 MISO cities | Weather as the operator saw it at 2 PM, not hindsight weather |
| `holidays` package | US federal holidays | Calendar effects |

**Coverage:**
- ERCOT: Apr 2021–Oct 2026, 48,960 hours.
- MISO: Jan 2023–Oct 2026, 33,600 hours. MISO's public daily price files are unavailable before 2023.
- Archived day-ahead weather forecasts start in about April 2021, so training excludes Winter Storm Uri (Feb 2021).

**Leakage rule (Figure 6, `06_what_is_known_when.png`).** Every input must be known at 2 PM on D-1:
- price and load history up to the issue time;
- lags of at least 48 hours relative to the target hour;
- weather and load *forecasts* published before 2 PM;
- the **day-ahead market price for the target hour**, which ERCOT posts at about 13:30 CT and MISO at about 12:30 CT [TODO verify the MISO posting time].

Unit tests (`tests/test_core.py`) check that every load feature only uses values available at the issue time, that weather forecasts never come from a later model run, and that the scale-free price features use only past prices.

**Features (about 50):**
- calendar: hour, day of week, month, cyclic encodings, holiday;
- lagged and rolling price and load;
- last values at the issue time;
- forecast temperature (mean, max and min over cities) and degree-days;
- the load forecast, and the load forecast relative to last week's maximum;
- the target hour's day-ahead price;
- optional scale-free price ratios (`rel_*`), used in an ablation.

**Splits (time-based, no shuffling):**
- train: Apr 2021 (MISO: Jan 2023) to Jun 2024;
- validation: Jul 2024–Jun 2025;
- test: Jul 2025–Oct 2026.

| | Train events / spike hours | Val events | Test events / spike hours |
|---|---|---|---|
| ERCOT (≥ $329) | 105 | 11 | 11 / 26 |
| MISO (≥ $111) | 101 | 89 | 184 / 473 |

(Events are spike hours merged when less than 6 hours apart.) [TODO: confirm the train spike-hour counts from `scripts/count_events.py`.]

**Team-designed evaluation set.** Besides the continuous test period we built **curated windows** (`scripts/curate_windows.py`, `data/external/test_events.csv`) with a rule fixed before looking at model output:
- the 10 hottest and 10 coldest forecast days of the test period per region, grouped into windows;
- a matched normal-weather control window for each, of the same length, within 4 weeks;
- windows overlapping documented grid emergencies (Winter Storm Fern, Jan 2026 [4], [5]; the MISO heat dome, Sept 2026 [6]) are tagged with their source.

Windows are selected on **weather only**, never on prices, so hot or cold windows without spikes stay in as hard negatives. Each row is checked by two team members [TODO: fill in `checked_by`].

**What the data looks like:**
- Figures 1–5 show the price series and threshold, spike counts per month, when spikes happen (summer late afternoons, and MISO winter mornings and evenings), and day-ahead vs real-time price.
- Temperature vs load shows that spikes sit at the high-load extremes, but many extreme hours are calm.

## 3. ML approach (rubric: 40)
**Models (all course topics [7]; implemented with scikit-learn [8] unless noted):**

| Model | Course topic | Notes |
|---|---|---|
| Rule baseline | baseline | Alert if the forecast max is ≥ 100 °F or the forecast min is ≤ 15 °F, **and** the load forecast is ≥ the train 95th percentile |
| k-nearest neighbours | instance-based | k ∈ {15, 50}, distance-weighted |
| Logistic regression | linear models | L2, C ∈ {0.01, 0.1, 1}, class-balanced |
| Gaussian Naive Bayes | probabilistic | |
| Random forest | trees and ensembles | 300 trees, depth ∈ {8, 16, none}, balanced subsampling |
| XGBoost [9] | boosting | depth ∈ {3, 6}, learning rate ∈ {0.03, 0.1}, `scale_pos_weight` |
| MLP | neural networks | (64, 32), early stopping, α ∈ {1e-4, 1e-2}, balanced sample weights |
| Chronos-2 (zero-shot and fine-tuned) [10], [11] | foundation models, 1-D signals | Forecasts price quantiles 34 h ahead from 512 h of history; converted to P(price ≥ threshold) |

**Training protocol (identical for every model):**
1. **Pipeline:** median imputation → standardization → estimator (`src/gridwatch/models/__init__.py`, config-driven).
2. **Hyperparameter search on train:** `GridSearchCV` with `TimeSeriesSplit(4)`, scored by average precision. Early folds can have no spikes, so the refit ignores empty folds.
3. **Class imbalance:** handled with class weights, `scale_pos_weight` (XGBoost) or sample weights (MLP). Without them the MLP collapsed on MISO.
4. **Calibration:** isotonic regression on the validation year, with the model frozen.
5. **Alert threshold:** the highest threshold reaching ≥ 70% recall on validation. The rule uses 0.5.
6. **Chronos-2:**
   - zero-shot on CPU;
   - fine-tuned on a Colab T4 (full fine-tuning, 1,000 steps, learning rate 1e-5) on the **train-period price series only**, using validation for early monitoring;
   - then the same calibration and threshold steps.

**Design decisions varied (the "key design decisions" requirement):**
- including tomorrow's day-ahead price or not;
- scale-free price features or not;
- removing each feature group (ablation);
- zero-shot vs fine-tuned Chronos;
- the label threshold, explored with event counts (p99 vs p98 vs $1,000).

**Model selection:** done per region on **validation**, before the test set was touched.
- MISO: logistic regression without the `rel_*` features (val PR-AUC 0.322).
- ERCOT: MLP (val 0.107).

**Metrics:**
- **PR-AUC** (average precision; the main metric because of class imbalance);
- precision and recall at the alert threshold;
- **event-level recall:** an event counts as caught if an alert was issued in the 34 hours before it started;
- false-alarm episodes per week;
- Brier score.

**Significance:** a **paired day-block bootstrap** (1,000 resamples). Whole days are resampled, because hours within a day are correlated, and every model is scored on the same resample. We report 95% CIs and two-sided p-values for the difference against a reference model.

## 4. Experiments and results (rubric: 40)

### 4.1 Main comparison (test year, Jul 2025–Oct 2026), Figure 8 (`08_model_comparison_test.png`)
| Test PR-AUC [95% CI] | MISO (184 events, base rate 4.0%) | p vs logreg | ERCOT (11 events, base rate 0.2%) |
|---|---|---|---|
| Rule baseline | 0.070 [0.05, 0.10] | < 0.001 | 0.003 |
| kNN | 0.190 [0.12, 0.30] | < 0.001 | 0.229 * |
| Naive Bayes | 0.157 [0.12, 0.21] | < 0.001 | 0.090 |
| **Logistic regression** | **0.343 [0.25, 0.45]** | — | **0.238** [0.00, 0.45] |
| Random forest | 0.076 [0.06, 0.09] | < 0.001 | 0.182 |
| XGBoost | 0.052 [0.04, 0.07] | < 0.001 | 0.080 |
| MLP | 0.305 [0.23, 0.40] | 0.21 | 0.114 |
| Chronos-2 zero-shot | 0.199 [0.13, 0.28] | < 0.001 | 0.016 * |
| Chronos-2 fine-tuned | 0.214 [0.15, 0.29] | 0.002 | 0.021 * |

\* The val-tuned threshold degenerates to alerting every hour (alert rate 1.0), so these models have no usable operating point even where their PR-AUC is valid.

- **MISO:** the validation pick (logistic regression) is also the best on test. It is significantly better than every model except the MLP.
- **ERCOT:** every learned model beats the rule, but the confidence intervals are so wide that **no pairwise difference among the learned models is significant**. The val pick (MLP) is not the test best; we did not re-select on test.

### 4.2 Operating point vs the proposal's target (Figure `reports/figures/pr_calib_MISO_test_y_price_final.png`)
At the validation-tuned threshold, MISO logistic regression reaches:
- **77% recall at 17% precision**;
- event-level recall 78%;
- about 5.8 false-alarm episodes per week.

Moving along the PR curve gives about 40% precision at 40% recall, or about 60% at 20%. The proposal's precision target is reachable, but not together with 70–80% recall. The reliability diagram shows the calibrated probabilities are overconfident at the top end, because the test year is much spikier than the calibration year.

### 4.3 Ablations: what carries the signal (Figure 9, `09_feature_ablation_miso.png`)
Each run retrains MISO logistic regression with one feature group removed.

| Removed | Test PR-AUC | p vs full model |
|---|---|---|
| nothing (full model) | 0.343 | — |
| tomorrow's day-ahead price | 0.151 | < 0.001 |
| all price features | 0.135 | < 0.001 |
| weather forecast | 0.343 | 0.98 |
| load forecast | 0.345 | 0.30 |
| calendar | 0.340 | 0.71 |

- The day-ahead market price carries most of the signal. Weather and load add nothing *on top of it*, plausibly because the day-ahead market already prices the forecast weather and load.
- Without any price information the model still roughly doubles the rule baseline.
- On validation, adding the target-hour day-ahead price raised PR-AUC 3–6× (MISO 0.119 → 0.322; ERCOT 0.012 → 0.077).
- ERCOT shows the same direction (0.238 → 0.039 without the day-ahead price), but with 11 events we report it descriptively only.

### 4.4 Why trees fail in MISO, and the scale-free feature experiment
- **The puzzle:** tree models were competitive in cross-validation but collapse on validation and test in MISO.
- **The cause is drift in the strongest feature:**
  - the share of MISO spike hours whose day-ahead price was *already* above the threshold went from 6% (train) to 30% (validation) to 45% (test);
  - 177 test hours had day-ahead prices above anything seen in training (Figure 4).
- **Trees cannot extrapolate beyond the training range; a linear model can.**
- **What we tried:** scale-free price features (day-ahead price relative to last week's median, log price, within-day rank).
  - They left the trees mixed (MISO RF 0.073 → 0.076, XGBoost 0.062 → 0.052).
  - They *hurt* logistic regression (0.343 → 0.186).
- So we kept them out of the selected model. Calibration was ruled out as the cause.

### 4.5 Chronos-2: does a foundation model help?
- Chronos-2 sees only the real-time price history.
- **Zero-shot:** in MISO it beats the rule (0.199 vs 0.070).
- **Fine-tuning:** on our training prices it gave a small, **non-significant** gain (+0.015, p = 0.18).
- **Against logistic regression** it remains well behind (p ≤ 0.002), consistent with the ablation: the price history alone misses the day-ahead market's information.
- **A natural extension:** fine-tuning Chronos-2 with the day-ahead price as a known-future input, which the model supports [TODO: keep as future work, or mention the TA's view].

### 4.6 Team-designed curated windows
| Curated PR-AUC | MISO (28 events, base rate 13%) | ERCOT (4 events) |
|---|---|---|
| Rule | 0.178 | 0.015 |
| **Logistic regression** | **0.568** [0.31, 0.77] | 0.349 |
| MLP | 0.473 (p = 0.32) | 0.196 |
| Chronos-2 fine-tuned | 0.325 (p = 0.06) | 0.049 |
| Random forest / XGBoost | 0.220 / 0.156 (p < 0.001) | 0.262 / 0.163 |

Precision is inflated in this view because the windows are enriched with spikes, so we compare models *within* it only.

Breakdown by window type (MISO, logistic regression):
- documented events: **69 of 70** spike hours caught;
- other extreme heat: 15 of 19;
- **matched normal-weather days: 33 of 51**. Spikes also happen in mild weather, and those are harder.

ERCOT's hottest test week (Aug 17–24, 2026; forecast up to 109 °F, load about 90 GW) had only **one** spike hour, and the model correctly stayed mostly quiet.

### 4.7 Where the model succeeds and fails
We split the MISO test spike hours by whether the day-ahead market had already priced them above the threshold:
- **DA price already high:** 211 hours, **97% caught**.
- **"Surprise" spikes** (DA price below the threshold): 262 hours, **62% caught**.
- ERCOT: 100% vs 20%.

Named events show the same pattern:
- Winter Storm Fern (MISO, 69 of 69);
- the Sept 2026 heat dome (15 of 15);
- the June 2025 heat wave (val, 26 of 28).

The honest limit of a day-ahead system is the surprise spike: sudden outages, wind or solar falling short of forecast, or unanticipated evening ramps.

### 4.8 Trends and the news behind them (Figures 10–11)
- **ERCOT: spikes collapsed while demand grew.**
  - Average load rose about 17% from 2023 to 2026, yet spike hours fell from 132 (2023) to 8 (2025) [12].
  - ERCOT's grid battery fleet nearly doubled in 2025 alone (7.8 → 13.9 GW) [13], [14], and the market monitor credits storage with fewer price spikes [15], [16].
  - The 2022–2023 training years contain the heat-driven scarcity era [17] (conservation appeals in July 2022 [18]; the Sept 6, 2023 emergency, EEA2 [19], [20]). That is concept drift with a physical cause, and it explains the scarce ERCOT test events.
- **MISO: spikes grew every year** (83 → 102 → 233 → 365 hours).
  - Shrinking reserve margins (summer surplus 6.5 → 2.6 GW as coal and gas units retired) [21]–[23].
  - Record capacity prices ($666.50/MW-day for summer 2025/26, vs $30) [21], [22].
  - Higher gas prices [24], [25].
  - The largest episodes match documented emergencies: Winter Storm Fern (Max Gen emergency, Jan 2026) [5], [26], the June 2025 heat wave [27], and the Sept 2026 heat dome (EEA2, record September peak) [6].
  - The day-ahead market increasingly anticipates scarcity (hours with the DA price above threshold: 16 → 404), which is exactly why the day-ahead price grew into the dominant feature.

Sources: [12]–[27]. These explanations are correlational; they are consistent with our data, but we did not test them causally.

## 5. Discussion, limitations and future work
- **Drift is the central difficulty.** The two regions drifted in opposite directions. A single fixed split exposes this honestly; a rolling-origin evaluation is a natural next step.
- **ERCOT is underpowered** at p99 (11 test events). A lower threshold (p98: about 34 test events) or pooling regions would add statistical power. We kept the pre-registered p99 definition [TODO: per TA guidance].
- **Calibration:** isotonic calibration on a calmer year leaves overconfident top probabilities in a spikier year.
- **Feature scope:** no generator outages, wind/solar forecasts or gas prices. These would target exactly the "surprise" spikes we miss.
- **Secondary label (operator alerts):** planned but not completed; we discuss documented operator emergencies qualitatively instead (section 4.8).
- **Ethics and deployment:** a false-negative warning can mean an unprepared grid, while frequent false alarms erode trust. The alert threshold is a policy choice and should be set with the operator's costs in mind.

## 6. Conclusion
Using only information public at 2 PM the day before, a calibrated logistic regression warns of MISO price spikes with test PR-AUC 0.34 (8.5× chance). It catches almost every hour of headline weather emergencies, and it significantly outperforms tree ensembles and a time-series foundation model.

The day-ahead market price is the key input, and the hard residual is the surprise spike the market itself did not foresee. ERCOT shows how quickly a market can change: its battery build-out made spikes rare enough that we cannot separate models there.

## Group contributions
All four members contributed equally and worked together from start to end. Background research, data collection and cleaning, modelling, evaluation, the report and the video were shared across the team; each member also led the parts listed below.

| Member | Contributions |
|---|---|
| Omkar | Shared background research and data cleaning (ERCOT and MISO price, load and weather data). Led the code repository and modelling pipeline, the ERCOT threshold decision (p99 vs p98), the video intro and results segment (Figures 8–9), and final report assembly and submission. |
| Ayush | Shared background research and data cleaning. First reviewer of the curated test windows; verified when MISO posts day-ahead prices (leakage check); video segment on the data and what is known at 2 PM (Figures 1 and 6). |
| Avni | Shared background research and data cleaning. Independent second reviewer of the curated test windows; checked every news and market-monitor source and formatted the references in IEEE style; video segment on trends and news (Figures 10–11). |
| Trupti | Shared background research and data cleaning. Wrote the video script, made the slides by hand and edited the final video; proofread the full report. |

## Code and reproducibility
Code: https://github.com/omkarchougule19/gridwatch. To reproduce:
```
pip install -e ".[dev]"
python -m gridwatch.data.ingest ...
python scripts/train.py --model <name>
python scripts/evaluate.py --tag final --models logreg-no_rel rule knn naive_bayes random_forest xgboost mlp chronos-zeroshot chronos-finetuned
python scripts/make_visuals.py
```
Chronos: `notebooks/chronos_colab.ipynb`. The decision log is in `DECISIONS.md`.

## References
All online sources accessed Nov. 26, 2026.

[1] U.S. Energy Information Administration, "Hourly electric grid monitor (Form EIA-930)," API v2, dataset electricity/rto/region-data. [Online]. Available: https://www.eia.gov/electricity/gridmonitor/

[2] Grid Status, *gridstatus: Python library for electricity market data*, ver. 0.37.0. [Software]. Available: https://github.com/gridstatus/gridstatus

[3] Open-Meteo, "Previous Runs API" (NOAA GFS, model `gfs_seamless`). [Online]. Available: https://open-meteo.com/en/docs/previous-runs-api

[4] Electric Reliability Council of Texas (ERCOT), "Winter Storm Fern – January 2026: Post-event report," Jan. 28, 2026. [Online]. Available: https://www.ercot.com/files/docs/2026/01/28/ERCOT-Post-Event-Report-Winter-Storm-Fern.pdf

[5] Midcontinent Independent System Operator (MISO), "Overview of Winter Storm Fern," presented to the Reliability Subcommittee, Feb. 17, 2026. [Online]. Available: https://cdn.misoenergy.org/20260217%20RSC%20Item%2005%20Winter%20Storm%20Fern%20Report741721.pdf

[6] A. Durish Cook, "MISO enters emergency in early Sept. heat, breaks demand record," *RTO Insider*, Sep. 3, 2026. [Online]. Available: https://www.rtoinsider.com/142336-miso-enters-emergency-in-early-sept-heat/

[7] D. Forsyth, *Applied Machine Learning*. Cham, Switzerland: Springer, 2019.

[8] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, San Francisco, CA, USA, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[10] A. F. Ansari *et al.*, "Chronos: Learning the language of time series," *Trans. Mach. Learn. Res.*, 2024. [Online]. Available: https://arxiv.org/abs/2403.07815

[11] A. F. Ansari *et al.*, "Chronos-2: From univariate to universal forecasting," 2025, *arXiv:2510.15821*.

[12] L. Hotz, "Why were ERCOT battery revenues so low in 2025?," Modo Energy, Feb. 3, 2026. [Online]. Available: https://modoenergy.com/research/why-were-ercot-battery-revenues-so-low-in-2025-weather-energy-arbitrage-builodout

[13] Modo Energy, "ERCOT annual buildout report: Battery capacity reaches 14 GW entering 2026," Feb. 2, 2026. [Online]. Available: https://modoenergy.com/research/en/ercot-battery-buildout-2025-annual-report

[14] C. Rose, "ERCOT's battery storage boom: A strategic opportunity for utilities, IPPs, and developers," Yes Energy, Jun. 20, 2025. [Online]. Available: https://www.yesenergy.com/blog/ercots-battery-storage-boom

[15] J. McDonald, "Independent Market Monitor (IMM) report," Potomac Economics, presented at the ERCOT Board of Directors Meeting, Sep. 22–23, 2025. [Online]. Available: https://www.ercot.com/files/docs/2025/09/15/8.1-Independent-Market-Monitor-IMM-Report.pdf

[16] D. Wamsted and S. Feaster, "Despite record summer demand, solar, storage kept ERCOT prices low," Institute for Energy Economics and Financial Analysis (IEEFA), 2026. [Online]. Available: https://ieefa.org/resources/despite-record-summer-demand-solar-storage-kept-ercot-prices-low

[17] Ascend Analytics, "Can ERCOT's energy supply keep pace with soaring demand appetite?," Jul. 31, 2026. [Online]. Available: https://www.ascendanalytics.com/blog/ercot-market-outlook-scarcity-pricing-load-growth

[18] ERCOT, "ERCOT issues conservation appeal to Texans and businesses," press release, Jul. 13, 2022. [Online]. Available: https://www.ercot.com/news/release/2022-07-13-ercot-issues-conservation

[19] ERCOT, "ERCOT has initiated Energy Emergency Alert Level 2 (EEA 2), conservation is critical," press release, Sep. 6, 2023. [Online]. Available: https://www.ercot.com/news/release/2023-09-06-ercot-has-initiated

[20] A. Bennitt, "ERCOT south wind contributed to EEA2 event," Yes Energy, Nov. 15, 2023. [Online]. Available: https://www.yesenergy.com/blog/ercot-south-wind-contributed-to-eea-2-event

[21] Voltus, "MISO's 2025 PRA: Why high prices are likely here to stay & how you can benefit," May 8, 2025. [Online]. Available: https://www.voltus.co/blog/miso-2025-pra

[22] Enel North America, "MISO's 2025 capacity auction results: Summer prices soar," May 14, 2025. [Online]. Available: https://www.enelnorthamerica.com/insights/blogs/miso-2025-capacity-auction-results

[23] Enverus, "PJM and MISO winter outlook: A mild season with thin margins," Nov. 9, 2026. [Online]. Available: https://www.enverus.com/blog/pjm-and-miso-2026-winter-outlook-thin-margin/

[24] C. Milton, "IMM quarterly report: Spring 2025," Potomac Economics, presented to the MISO Market Subcommittee, Jul. 10, 2025. [Online]. Available: https://www.potomaceconomics.com/wp-content/uploads/2025/06/IMM-Quarterly-Report_Spring-2025-MSC.pdf

[25] C. Milton, "IMM quarterly report: Winter 2026," Potomac Economics, presented to the MISO Markets Committee of the Board of Directors, Mar. 24, 2026. [Online]. Available: https://cdn.misoenergy.org/20260324%20Markets%20Committee%20of%20the%20BOD%20Item%2009%20IMM%20Quarterly%20Report746764.pdf

[26] A. Durish Cook, "MISO enters Max Gen emergency in arctic blast," *RTO Insider*, Jan. 24, 2026. [Online]. Available: https://www.rtoinsider.com/124096-miso-enters-max-gen-emergency-in-arctic-blast/

[27] D. B. Patton, "IMM quarterly report: Summer 2025," Potomac Economics, presented to the MISO Market Subcommittee, Sep. 30, 2025. [Online]. Available: https://www.potomaceconomics.com/wp-content/uploads/2025/11/IMM-Quarterly-Report_Summer-2025-MSC.pdf
