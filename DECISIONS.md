# Decisions and findings log

A running record of what was decided and why, and what was found. Newest session first.
Status and next steps are in [PROGRESS.md](PROGRESS.md).

---

## Session 1: 2026-11-17

### 1. Inputs reviewed
- **The proposal and TA feedback** (course feedback page, not stored in the repo).
  - **Task:** for each hour of the next 24, give P(the ERCOT/MISO grid is stressed).
  - **Models:** a rule baseline, XGBoost, and Chronos (zero-shot and fine-tuned).
  - **Metrics:** PR-AUC (main metric), precision and recall at a threshold, event-level recall, Brier score.
  - **Targets:** 70–80% recall at 40–50% or better precision.
- **The syllabus** (cs441_aml_syllabus.pdf, Hoiem). It covers policy only, with no list of topics. The course site pages (`/CS441/fa2026/`) returned 404, probably because they need a login. What matters for the project:
  - The final project (200 pts) has to define a custom problem, collect data for training and evaluation, compare models, and report results. Graded checkpoints come along the way.
  - **Academic integrity:** no code from the Internet unless the instructor approves it. AI use is allowed. We therefore call library APIs and don't copy snippets.
  - Compute: Colab, or Illinois Computes (A100) for deep models such as Chronos fine-tuning.
  - Textbook: Forsyth, *Applied Machine Learning* (2019).

### 2. TA feedback → how it's handled
| TA point | Decision | Where |
|---|---|---|
| Use weather forecasts as issued, not observed weather | Use only forecasts issued the day before (GFS `previous_day1`). Never use ERA5 or ISD as features | `data/ingest.py: weather_forecasts`, `data/load.py: asof_forecasts` |
| Price threshold as the primary label, alerts as the secondary | `label.primary: price`, `label.secondary: alert`. Pick one with `--label` | `labels.py`, `configs/default.yaml` |
| Balanced curated test set inflates precision | Report both the continuous test period and the curated windows | `scripts/evaluate.py` |
| Count events before fixing the test window; add regions if scarce | `scripts/count_events.py` | |
| Chronos fine-tuning needs a GPU | Fine-tune on Colab or Illinois Computes. Zero-shot runs locally | `models/chronos_model.py` |

### 3. Design decisions (scaffold)
- **Issue time:** a forecast is made once a day at **14:00 local (America/Chicago) on day D-1**, covering all 24 hours of day D. Lead times run from 10 to 33 hours.
- **Leak-safe features:**
  - Lags of at least 48h (more than the longest lead).
  - Rolling stats shifted by 48h.
  - Values at the issue time (the last observation at or before 14:00 D-1).
  - Calendar features of the target hour.
  - Weather forecasts only as issued.
  - These are enforced by `tests/test_core.py::test_features_have_no_future_leakage`.
- **Price threshold:**
  - An absolute $1,000/MWh by default.
  - If `price_percentile` is set, the threshold is computed on the **train split only**.
  - MISO prices rarely reach $1,000, so MISO will probably need a percentile.
- **Events:** stress hours less than 6h apart are merged into one event (`event_merge_gap_hours`).
- **Event-level recall:** an event counts as caught if some alert with p ≥ threshold was **issued in [start-24h, start)** and targets an hour inside the event. False alarms are also reported as alert episodes with no stress hour, per week.
- **Alert threshold:** tuned on **val**. It is the highest threshold that still gives recall ≥ 0.7 (`evaluation.min_recall`).
- **Splits:** time-based, inclusive local dates. Train ends 2024-06-30, val runs 2024-07-01 → 2025-06-30, test runs 2025-07-01 → 2026-10-31. Disjointness is asserted.
- **Rule baseline:** alert when the forecast temperature is ≥100°F or ≤15°F AND load is at or above its train 95th percentile. It currently uses `load_mw_lag48` as a stand-in for the load forecast. The plan is to switch to EIA's `load_fcst_mw`.
- **XGBoost:** `scale_pos_weight = neg/pos` on train, `eval_metric=aucpr`, early stopping on val. Calibration (isotonic) is a TODO before reporting the Brier score.
- **Chronos:** forecast price quantiles, then P(price ≥ threshold) by interpolating the quantile CDF (0 above the top quantile, 1 below the bottom). The tails are coarse, so consider adding 0.95 and 0.99 quantiles.
- **Environment:** `.venv` with Python 3.13, set up with `pip install -e ".[dev]"`. Chronos and torch are an optional extra: `[chronos]`.

### 4. Bugs found and fixed
- `features.at_issue_features`: `reindex(..., method="ffill")` on tz-aware duplicate timestamps raised a TypeError. It was replaced with a `searchsorted` lookup.
- Deprecated pandas accessors `dayofweek`/`dayofyear` were changed to `day_of_week`/`day_of_year` (Pandas4Warning).

### 5. Smoke test (synthetic data, kept in the scratchpad, not in the project)
- The pipeline ran end to end: count_events → train (rule, xgboost) → evaluate.
- With a base rate of about 0.4%, XGBoost scored val PR-AUC of about 0.45 and event recall of about 0.73–0.80, and test PR-AUC of about 0.34–0.37. These numbers only show that the plumbing works. They say nothing about real performance.
- The curated view was skipped because `test_events.csv` is empty. Expected.

### 6. Data sources (preferred, citable)
| Data | Source | Access |
|---|---|---|
| Hourly load and BA day-ahead load forecast | U.S. EIA, Hourly Electric Grid Monitor (Form EIA-930), API v2 `electricity/rto/region-data` (ERCO, MISO) | `gridstatus.EIA`. Needs a free `EIA_API_KEY` |
| ERCOT prices | ERCOT MIS: Historical RTM SPP (15-min, hourly mean) and DAM SPP, hub **HB_HUBAVG** | `gridstatus.Ercot.get_rtm_spp/get_dam_spp(year)` |
| MISO prices | MISO Market Reports: RT final ex-post hourly LMP and DA ex-post hourly LMP, **ILLINOIS.HUB** | `gridstatus.MISO.get_lmp_real_time_hourly_final/get_lmp_day_ahead_hourly` |
| Weather forecasts as issued | NOAA GFS (`gfs_seamless`), the run issued the day before, from the archive at the Open-Meteo Previous Runs API | `requests`. Averaged, plus max and min, over load-center cities |
| Operator alerts | Hand-curated `data/external/alerts.csv` from ERCOT ops messages/notices (EEA, Watch, Advisory, conservation appeals) and MISO RT notifications (Hot/Cold Weather Alerts, Max Gen, Conservative Ops) | manual |
| Event context (for the report) | FERC–NERC Feb 2021 cold weather report (Uri); FERC–NERC Winter Storm Elliott inquiry (2023); NERC seasonal reliability assessments | cite |
| Methods | Chen & Guestrin 2016 (XGBoost); Ansari et al. 2024 TMLR (Chronos); Weron 2014 IJF; Lago et al. 2021 Applied Energy; Saito & Rehmsmeier 2015 PLoS ONE (PR vs ROC); Brier 1950 | cite |
| Optional backup for weather | NOAA NBM/NDFD archives (NCEI, AWS), via `Herbie`. This is the rigorous route if earlier years are needed | not used yet |

Citations were written from memory. **Check the exact titles and report IDs before citing them.**

### 7. Data coverage findings (probed 2026-11-17)
- **Open-Meteo Previous Runs, `temperature_2m_previous_day1`:**
  - null for 2019-07-01, 2021-01-01, 2021-03-01, 2021-03-20.
  - values present from **2021-04-01** onward (2022, 2023, 2025, 2026 all checked).
  - **a gap on 2024-01-01** (null for both the default model and `gfs_seamless`). Expect scattered gaps.
- **`apparent_temperature` / `wind_speed_10m` `_previous_day1`:** null until at least 2023-06 and present from 2024-08. **Decision: drop heat index and wind** and use only temperature (mean, max and min across cities).
- **MISO daily price files** (`docs.misoenergy.org/marketreports/YYYYMMDD_rt_lmp_final.csv`): 404 for 2021-08 to 2022-11 and 200 from **2023-01-01** onward. The annual zip names tried also returned 404. **MISO training data starts in 2023.**
- **ERCOT RTM/DAM SPP:** annual files from 2011, so there's no constraint.
- **ERCOT operations messages (gridstatus):** historical pulls come from **Wayback Machine snapshots** (only 2 snapshots for Jul–Sep 2023), so they're incomplete. **Decision: alerts come from the curated CSV** rather than this endpoint.
- **gridstatus 0.37.0:** the class is `Ercot`, not `ERCOT`. MISO hubs available include ARKANSAS, ILLINOIS, INDIANA, LOUISIANA, MICHIGAN, MINN, MS, TEXAS.

### 8. Resulting data decisions
- **Training start moves from 2019 to 2021-04-01** because of the forecast coverage. This loses Winter Storm Uri (Feb 2021) as a training event; it can still be used as context in the write-up. *(Not yet changed in the config. See PROGRESS step 1.)*
- MISO trains on 2023-01 → 2024-06 only (one summer). It will probably need a percentile threshold, or more regions/hubs.
- Each raw pull is saved in `data/interim/` so it isn't downloaded twice.

### 9. Working preferences (from the user)
- A **"lazy" coding style**: reuse libraries such as sklearn, gridstatus and pandas, keep the code modular, and **write new code only when nothing existing can be reused**.
- Align the models and the train/val/test methods with what CS441 teaches.
- **Planned comparison set:**
  - rule baseline
  - logistic regression
  - random forest
  - XGBoost
  - MLP
  - Chronos (zero-shot and fine-tuned)

  These will be built through one config-driven sklearn pipeline factory, with hyperparameters chosen by TimeSeriesSplit CV (`average_precision`) and optional calibration.

### 10. Open questions
- Can we get the actual CS441 lecture schedule (it needs a login) to confirm which models and methods to emphasize?
- Should we add regions (e.g. SPP, PJM via gridstatus) if the 2025–26 event counts are small?
- Should we use NOAA NBM/NDFD archives to recover 2019–2021 training data (and Uri)?
- Should we put the folder under git?

---

## Session 1, continued: 2026-11-17 (afternoon)

### 11. Course schedule (the site loaded later: courses.grainger.illinois.edu/cs441/fa2026/)
- **Topics:**
  - K-NN and generalization
  - clustering, PCA
  - linear regression, logistic regression, SVM
  - Naive Bayes
  - EM, density estimation (MoG, KDE)
  - outliers
  - decision trees, ensembles and random forests
  - SGD, MLPs
  - CNNs, attention and transformers
  - foundation models (CLIP, GPT)
  - ethics and fairness, deploying ML
  - audio and 1-D signals
  - RL
- **Final project due Sun Dec 13, 11:59pm. It cannot be late.** Exam 3 is Dec 3–8.
- **Project doc requirements (Google Doc linked from the site):**
  - Build a data pipeline, train models **while varying key design decisions**, and **evaluate and plot results, including which differences are statistically significant**.
  - Write a report from the template and record a **5–6 min video**. The video and slides may not be AI-generated.
  - The final evaluation set must be **designed by the team**, not taken from an ML benchmark (the penalty is -20).
  - Code link required (-30 if missing). List group contributions (-10 if missing).
  - Rubric: problem 15, ML approach 40, data 40, experiments 40, video 30.

### 12. Model and training decisions (implemented)
- **One config-driven sklearn factory** (`models/__init__.py`): `Pipeline(SimpleImputer(median) → StandardScaler → estimator)`. It replaced the hand-written XGBoost wrapper (`models/xgb.py` was deleted). Every model is a config entry.
- **The comparison set follows the course:** rule, kNN, logistic regression, Naive Bayes, random forest, XGBoost, MLP, and Chronos (notebook).
- **Imbalance handling:**
  - logreg and RF: `class_weight` balanced.
  - XGBoost: `scale_pos_weight = neg/pos`.
  - **MLP: `sample_weight` balanced.** Without it, the MLP collapsed on MISO (val PR-AUC 0.005 → 0.47 on synthetic data).
- **Hyperparameters:** `GridSearchCV` with `TimeSeriesSplit(4)` on train, scored by average precision.
  - **Fix:** early time folds can have no stress hours, so their AP is NaN. Before the fix, MISO's grid search picked hyperparameters arbitrarily. The best candidate is now chosen by the mean over the folds that have positives (`refit=best_ignoring_empty_folds`).
- **Calibration:** isotonic `CalibratedClassifierCV(FrozenEstimator(model))` fit on val. The alert threshold is then tuned on val as well. Val is used for both; this is acceptable here, and test is never touched.
- **Rows without a price are masked out of every split**, so a missing price never counts as "no spike". This matters for MISO before 2023.
- **Train start changed to 2021-04-01** (the forecast archive coverage, §7).

### 13. Feature decisions
- Weather features are `temp_f_fcst` and its max and min across cities, plus CDD and HDD. Heat index and wind were dropped (§7).
- **Added `load_fcst_mw`** (the BA day-ahead load forecast from EIA-930) and `load_fcst_vs_week_max`.
  - *Assumption to verify:* the BA's DF value for day D is published before 14:00 on D-1.
- **Removed `temp_dev_from_normal`:** its normals were computed over all years, test included, which is a leak.
- **Added `is_holiday`** (US federal holidays from the `holidays` package).
- The rule baseline now uses `load_fcst_mw`, plus `temp_f_fcst_max` for heat and `temp_f_fcst_min` for cold, falling back to `load_mw_lag48`.

### 14. Experiments and significance (implemented)
- **`train.py --drop <prefixes>`:** feature-group ablations, saved under a tagged name (e.g. `xgboost-no_temp-no_cdd-no_hdd`). This covers "varying key design decisions". The label choice (`--label`) is another axis.
- **`evaluate.py --models ...`:** produces a metrics table, a **paired day-block bootstrap** of PR-AUC vs the first model, and PR and reliability plots for each region and view.
  - Whole days are resampled because hours are autocorrelated, so an i.i.d. bootstrap would be overconfident.
  - This is the only hand-written statistics code; nothing in sklearn or scipy does a paired block bootstrap of AP directly.
- **Synthetic smoke test** (this only shows the plumbing works; the numbers mean nothing about real performance): all 8 models and the ablation trained and evaluated, and the figures render.

### 15. Still open
- Run on real data (needs an EIA key). Then check event counts and choose MISO's threshold.
- Chronos notebook on Colab (zero-shot first).
- Build the curated `alerts.csv` and `test_events.csv` (two reviewers per row).
- Possible further ablations: label (price vs alert), issue time, price threshold, and the price-at-issue features (`--drop rt_price da_price`).

### 16. Small decisions (2026-11-17, evening)
- **Azure removed:** the proposal mentions an Azure pipeline, but the ingest pulls directly from the public sources and saves them locally. The Azure settings were dropped from `.env.example`, and `.env` holds only `EIA_API_KEY`.
- **MISO ingest starts 2023-01-01:** MISO price files only start then, and hours without a price are masked anyway, so earlier load and weather for MISO would go unused.
- Added the `data/external/alerts.csv` template (format and allowed `kind` values).

### 17. Chronos and data download (2026-11-17, ~13:00)
- **Chronos-2 (`amazon/chronos-2`) instead of Chronos-Bolt:** in chronos-forecasting 2.3.2, only `Chronos2Pipeline` has a built-in `.fit()` for fine-tuning (full or LoRA). That means no custom training loop, and zero-shot vs fine-tuned compares the same model.
- **Quantile levels changed to [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]**, because spike detection depends on the upper tail.
- `predict_proba_from_history` now runs in batches (64 forecasts per call), and the code handles both output formats: Bolt returns a tensor (batch, h, q), Chronos-2 a list of (variates, h, q) tensors. Tested locally on CPU with `chronos-bolt-tiny`: all target hours matched to their issue-time forecast, with no gaps.
- **New `predict_for_targets`:** maps each target hour to the forecast issued at 14:00 the day before.
- **Colab notebook** (`notebooks/chronos_colab.ipynb`):
  - Zero-shot and fine-tuned runs, each calibrated (isotonic) and with its threshold tuned on val.
  - Predictions are saved as `models/chronos-*.pkl` (`{"preds": ...}`), and `evaluate.py` reads stored predictions, so Chronos joins the same significance test and plots.
  - Fine-tuning uses the **train-split prices only**, with val prices for validation loss.
- **The network on this machine drops DNS lookups intermittently** (ercot.com, pythonhosted, Open-Meteo, EIA all failed at least once). So downloads now run in **parallel chunks with retries** (one ERCOT year or one MISO month per chunk, 8 at a time). Each chunk is saved on its own, so a failure only re-runs that chunk.
- **MISO load/weather coverage (2023-01 → 2026-10):** load 0.1% missing, BA load forecast 0.4%, GFS day-ahead temperature 1.5%.
- **ERCOT weather:** 48,960 hours, 1.0% missing.

### 18. Real data, first results (2026-11-17, ~13:20)
- **MISO built** (`data/processed/MISO.parquet`): 33,600 hours, 2023-01-01 → 2026-10-31.
  - Missing: prices 0%, load 0.1%, BA load forecast 0.4%, forecast temperatures 1.5%.
- **MISO ILLINOIS.HUB real-time price:**
  - median $25.5, p99 $177, p99.9 $710, max $7,485/MWh.
  - The minimum is -$140; negative prices are real.
- **MISO threshold decision: $1,000 fixed gives 0 train events**, so nothing can be learned. **MISO now uses the train-split 99th percentile = $111.3/MWh** (config `label.regions.MISO`). The per-region override takes 2 lines in `_common.prepare`.

  | threshold | train hrs / events | val hrs / events | test hrs / events |
  |---|---|---|---|
  | $1000 | 0 / 0 | 1 / 1 | 20 / 13 |
  | $500 | 5 / 5 | 11 / 9 | 41 / 26 |
  | p99.5 ($154) | 66 / 59 | 95 / 57 | 261 / 120 |
  | **p99 ($111)** | **132 / 101** | **178 / 89** | **473 / 184** |
  | p98 ($83) | 263 / 173 | 338 / 150 | 716 / 240 |
- **Distribution shift:** the MISO test period has about 3.5× the spike hours of train (473 vs 132 at p99). 2025–26 was more volatile than 2023–H1 2024. Mention this in the report; it will lower test precision and make calibration harder.
- **Download issue:** running all 6 ERCOT yearly archive downloads in parallel failed every time (OSError and BadZipFile). The same file downloads fine on its own (DAM 2024 in 9 s), so the parallel requests to ERCOT collide or get throttled. **ERCOT years now download one at a time.** MISO daily files work fine in parallel (14 workers).
- **ERCOT load:** 48,937 hours, nothing missing; BA forecast 0.3% missing; 29–91 GW. The EIA request treats the end date as exclusive, so Oct 31 2026 is missing (a negligible loss).

### 19. ERCOT built, threshold (2026-11-17, ~13:27)
- **ERCOT built** (`data/processed/ERCOT.parquet`): 48,960 hours, 2021-04-01 → 2026-10-31.
  - Missing: prices 0%, load 0.1%, BA forecast 0.3%, forecast temperatures 1.0%.
  - Prices: median $25, p99 $235, p99.9 $2,015, max $5,087.
  - Downloaded one year at a time, 64–91 s per year.
- **The opposite shift from MISO:** ERCOT spikes were common in 2021–23 but nearly vanished in 2025–26. One likely cause is the large build-out of batteries and solar; that needs a citation if it goes in the report.

  | threshold | train hrs / events | val hrs / events | test hrs / events |
  |---|---|---|---|
  | $1000 | 105 / 44 | 8 / 6 | **1 / 1** |
  | $500 | 209 / 77 | 9 / 7 | 13 / 6 |
  | p99.5 ($745) | 143 / 51 | 9 / 7 | 4 / 3 |
  | **p99 ($329)** | **285 / 105** | **17 / 11** | **26 / 11** |
- **Decision: ERCOT also uses the train-split 99th percentile ($329/MWh)**, the same rule as MISO, applied consistently across regions.
- **Open (team decision): ERCOT test has only about 11 events and val about 11.** The TA said to add regions if events are scarce. Options:
  - (a) add SPP, PJM or more hubs via gridstatus;
  - (b) lower ERCOT to p98;
  - (c) pool regions for the event-level metrics;
  - (d) report ERCOT as a "regime change" finding.

### 20. First real training runs (2026-11-17, ~23:00)
- **Run 1 (no day-ahead price for the target hour):** weak. Val PR-AUC was ERCOT 0.009–0.019 (base rate 0.2%) and MISO 0.036–0.119 (base rate 2.0%). The best model was logreg on MISO.
- **Bug fixed, event-level recall:** the window was "issued ≤24 h before the event start", but a 14:00 D-1 run reaches 10–33 h ahead. Events starting after 14:00 on day D could never count as caught. **The window is now 34 h**, the full forecast horizon.
- **Bug fixed, rule baseline:** its output is 0/1, so threshold tuning fell back to "alert every hour". **It now uses a fixed threshold of 0.5.**
- **New feature `da_price_target`, the day-ahead (DA) price of the target hour** (config `features.target_da_price: true`).
  - It is known at the 14:00 D-1 issue time: ERCOT posts DAM results around 13:30 CT. A FERC order (2020) directs MISO to post them at 13:30 ET = 12:30 CT. **Confirm MISO's current posting time.**
  - This is what a trader or operator actually has when deciding.
- **Run 2, with DA price:** the biggest gain so far.

  | val PR-AUC | ERCOT, without → with | MISO, without → with |
  |---|---|---|
  | logreg | 0.012 → **0.077** | 0.119 → **0.322** |
  | xgboost | 0.017 → 0.062 | 0.044 → 0.050 |
  | random_forest | — → 0.053 | — → 0.063 |
  | mlp | — → 0.059 | — → 0.135 |
  | naive_bayes | — → 0.029 | — → 0.136 |
  | knn | — → 0.059 | — → 0.070 |
  | rule | 0.009 | 0.059 |

  Logreg is best in both regions. MISO's 0.322 is about 16× the base rate, but precision at 70% recall is still only 0.12, far below the 40–50% target.
- **Calibration is not why the trees lag:** without isotonic calibration, xgboost/RF/logreg val PR-AUC barely changes (MISO xgb 0.048, RF 0.060, logreg 0.334).
- **Why the trees lag: the DA–spike relationship drifts and the range grows.**
  - Share of MISO spike hours where DA was already ≥ threshold: train 6% → val 30% → test 45%. Early MISO spikes were real-time surprises; later ones were increasingly anticipated by the DA market.
  - MISO hours with DA above the training max: val 10, **test 177**. Trees cannot extrapolate past their training range; logreg (linear) can.
  - ERCOT has no out-of-range DA hours, but its spikes are rare.
  - **Next experiment:** scale-free features (DA price relative to the trailing week's median or max, log price), and/or monotone constraints.
- **New metric `alert_rate`** (share of hours flagged). kNN reported recall 1.0 with "0 false alarms/week" because no threshold reached 70% recall, so it alerted on every hour, which merges into one giant alarm episode that overlaps real events. `alert_rate` makes this visible.

### 21. Scale-free price features (2026-11-17, ~23:30)
- **Why:** a web literature check suggested this.
  - DART-spread spike papers (Hubert, Lolas & Sircar 2026; Wang et al. 2024) confirm the DA price is the key signal.
  - Tree models can't extrapolate; the standard fix is a transform to relative or scale-free values (Snowflake transforms blog; LightGBM `linear_tree` is "not a silver bullet").
  - I found no spike-specific study of rolling normalization under drift.
- **What:** `relative_features` (config `features.relative_prices: true`, ablate with `--drop rel_`):
  - DA price of the target hour ÷ the trailing-week median RT price at the issue time;
  - signed-log DA price;
  - DA price rank within the target day;
  - RT price at the issue time ÷ the trailing-week median.
  
  Only data known at 14:00 D-1 is used (new test: a future spike does not change earlier features). The lookup was refactored into a shared helper, `value_at`.
- **Fits the assignment:** it's one more design decision with its own ablation, it doesn't touch the test set, and it's one function behind a config flag.
- **Result: mixed. The extrapolation hypothesis is NOT confirmed for MISO.**

  | val PR-AUC | ERCOT, no rel → rel | MISO, no rel → rel |
  |---|---|---|
  | logreg | 0.077 → 0.083 | **0.322 → 0.281** |
  | random_forest | 0.053 → **0.093** | 0.063 → 0.061 |
  | xgboost | 0.062 → 0.064 | 0.050 → 0.046 |
  | mlp (vs the run-2 baseline) | 0.059 → 0.107 | 0.135 → 0.263 |
  | naive_bayes (vs run 2) | 0.029 → 0.056 | 0.136 → 0.148 |
  | knn | 0.108 (alert_rate 1.0, degenerate) | 0.097 (alert_rate 1.0) |
  - **ERCOT:** helps most models, but only 11 val events, so the differences are probably within noise. Check with the bootstrap on test.
  - **MISO:** the trees stay at about 0.05 even with in-range features. So their weakness is more likely **concept drift** (the DA–spike relationship changed: the share of spikes already priced in by DA went from 6% in train to 30% in val) than range extrapolation. Logreg is best **without** the rel features.
- **Decision:** keep the flag on by default, report the ablation, and **choose each region's proposed model on val**. MISO: logreg-no_rel (0.322). ERCOT: mlp (0.107) or random_forest (0.093). kNN is excluded because its threshold degenerates to "alert always".
- The `alert_rate` metric worked as intended: it shows kNN's threshold falling back to alerting every hour.

### 22. Test-year results (2026-11-17, ~23:40). Test set used once; models chosen on val beforehand (§21)
- **Test period:** Jul 2025 – Oct 2026 (11,707 h per region).
  - MISO: 473 spike hours / 184 events (base rate 4.0%).
  - ERCOT: 26 spike hours / 11 events (base rate 0.2%).
- **Files:**
  - Full table: `reports/metrics_y_price.csv`.
  - Bootstrap vs rule: `reports/significance_y_price_vs_rule.csv`.
  - Bootstrap vs the best MISO model: `reports/significance_y_price_vs_logreg-no_rel.csv`.
  - Figures: `reports/figures/pr_calib_{ERCOT,MISO}_test_y_price.png`.

| test PR-AUC [95% CI] | ERCOT | MISO |
|---|---|---|
| rule (baseline) | 0.003 | 0.070 |
| **logreg-no_rel** (MISO pick) | 0.238 | **0.343 [0.245, 0.445]** |
| mlp (ERCOT pick) | 0.114 | 0.305 [0.234, 0.398] |
| logreg (rel features) | 0.251 | 0.186 |
| random_forest | 0.182 | 0.076 |
| xgboost | 0.080 | 0.052 |
| naive_bayes | 0.090 | 0.157 |
| knn (alert_rate 1.0, degenerate) | 0.229 | 0.190 |

- **MISO:**
  - **logreg-no_rel, chosen on val, is also best on test**, at about 8.5× the base rate.
  - Significantly better than the rule (p < 0.001), and than logreg with rel features, naive_bayes and random_forest (all p < 0.001).
  - **Not** significantly better than the MLP (diff -0.036, CI [-0.101, 0.016], p = 0.21).
  - The tree models are **not** significantly better than the rule (RF p = 0.69, XGB p = 0.20). This confirms the drift finding (§20–21).
- **MISO operating point** (val-tuned threshold, target recall 70%):
  - precision 0.17, recall 0.77, event recall 0.78, about 5.8 false-alarm episodes per week.
  - From the PR curve, about **0.60 precision at 0.20 recall** and **about 0.40 precision at 0.40 recall**. The proposal's 40–50% precision is reachable only at lower recall than the 70–80% target. **Report this trade-off.**
- **MISO calibration:** overconfident in the top bin (predicted about 0.40, observed about 0.27). Brier 0.035 vs 0.042 for the rule.
- **ERCOT:**
  - Every model is significantly better than the rule (which nearly never fires correctly), but the CIs are enormous: logreg [0.005, 0.43]. With 11 events, **no pairwise model difference is significant** (vs logreg-no_rel: MLP p = 0.19, RF p = 0.72, NB p = 0.24).
  - The val-chosen ERCOT model (MLP, 0.114) is not the test-best (logreg 0.251), but that gap is not significant either.
  - **Do not re-select on test.** Report ERCOT as underpowered: this supports the TA's "add regions" advice or a lower ERCOT threshold.
- **Rel-feature ablation on test:** ERCOT RF 0.087 → 0.182 and logreg 0.238 → 0.251 (both within noise). MISO logreg 0.343 → 0.186 (rel features hurt, consistent with val).
- **Not done yet:** the curated-window view (needs `data/external/test_events.csv`), the alert label (needs `alerts.csv`), and Chronos (Colab).
