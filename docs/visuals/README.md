# GridWatch visuals

Regenerate everything with `python scripts/make_visuals.py`. The figures use the processed data, the saved models and the CSVs in `reports/`.

## What GridWatch predicts (and what it doesn't)
GridWatch is a **classifier, not a price forecaster**.
- **What it outputs:** each day at 2 PM it gives one number per hour of tomorrow, **the probability that the hour is a spike**.
- **What a spike is:** a real-time price at or above the region's 99th percentile of training prices ($329/MWh in ERCOT, $111/MWh in MISO).
- **There are two thresholds:**
  - The **spike threshold** (p99) defines the labels the model learns. It is a config setting (`label.regions.<REGION>.price_percentile`); p98, or a fixed $/MWh value, means changing that setting and retraining.
  - The **alert threshold** turns the probability into a yes/no warning. It is tuned on the validation year to catch at least 70% of spike hours.
- **Chronos-2 is the one exception:** it does forecast the price itself, as a range of quantiles for the next 34 hours. That forecast is then converted into the same P(price ≥ spike threshold), so all models are compared on the same task.

## The figures
| # | File | What it shows | Why it matters for the project |
|---|---|---|---|
| 1 | `01_prices_and_threshold.png` | Daily max real-time price, 2021–2026, with the spike threshold and the train/val/test periods shaded | What the target is: rare excursions above a line. Prices are normally $20–50/MWh |
| 2 | `02_spike_hours_per_month.png` | Spike hours per month in each region | The core difficulty: ERCOT spikes nearly vanish after 2023, while MISO's grow in the test period (opposite drift) |
| 3 | `03_when_spikes_happen.png` | Spike rate by hour of day × month | Why calendar features exist: spikes concentrate in summer late afternoons, and in MISO also on winter mornings and evenings |
| 4 | `04_dayahead_vs_realtime.png` | Tomorrow's day-ahead price vs the real-time price, train (gray) vs test (blue) | The strongest signal, plus the drift that breaks tree models: in MISO the test points go beyond the training range |
| 5 | `05_temperature_load_spikes.png` | Forecast temperature vs load, with spike hours in orange | The intuitive weather story (heat and cold drive load and stress), and why it isn't enough: many high-load hours are calm |
| 6 | `06_what_is_known_when.png` | Timeline of what information is available at the 2 PM issue time | The no-leakage rule behind every feature |
| 7 | `07_example_test_week.png` | The MISO test week with the most spikes: actual price (top), the model's day-ahead probability (bottom) | What a forecast actually looks like in use |
| 8 | `08_model_comparison_test.png` | Test PR-AUC for all models, with 95% bootstrap CIs and the random-guess line | Headline result: logistic regression is best in MISO; ERCOT CIs are too wide to pick a winner |
| 9 | `09_feature_ablation_miso.png` | MISO test PR-AUC when each feature group is removed | Removing the day-ahead price halves the score; weather and load add nothing on top of it |

**Also see `reports/figures/`:** precision-recall curves and reliability (calibration) plots per region, for example `pr_calib_MISO_test_y_price.png`.
