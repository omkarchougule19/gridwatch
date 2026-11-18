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
