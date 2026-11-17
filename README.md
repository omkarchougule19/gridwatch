# GridWatch

A 24-hour early warning for power grid stress and price spikes on ERCOT and MISO. This is the CS 441 (Fall 2026) final project.

GridWatch runs once a day at 14:00 local time and predicts, for each hour of the next day, the probability that the grid will be stressed.
- **Primary label:** the real-time price goes above a spike threshold.
- **Secondary label:** an operator emergency alert or notice is in effect.

Decisions and data findings are in [DECISIONS.md](DECISIONS.md). Status and next steps are in [PROGRESS.md](PROGRESS.md).

## Layout

```
configs/default.yaml        regions, data sources, label threshold, splits, the MODEL COMPARISON SET, evaluation
data/interim/               cached raw pulls (git-ignored)
data/processed/             one hourly table per region (git-ignored)
data/external/              hand-curated: alerts.csv (secondary label), test_events.csv (curated test windows)
src/gridwatch/
  data/ingest.py            EIA-930, ERCOT/MISO prices (gridstatus), GFS day-ahead forecasts (Open-Meteo)
  data/load.py              loads the hourly table, issue times, as-of join for forecasts
  labels.py                 price/alert labels, groups stress hours into events
  features.py               leak-safe calendar, holiday, lag, value-at-issue and forecast features
  splits.py                 time-based train/val/test, curated-window mask
  models/__init__.py        config-driven sklearn Pipeline factory (impute → scale → estimator)
  models/rule_baseline.py   rule baseline (extreme temperature AND forecast load ≥ p95)
  models/chronos_model.py   Chronos foundation model → P(price ≥ threshold)
  evaluate.py               PR-AUC, P/R at threshold, event-level recall, Brier, paired block bootstrap
scripts/                    count_events.py, train.py, evaluate.py
tests/                      pytest
```

## Models (course topic → model)
| CS441 topic | Model (config key) |
|---|---|
| baseline | `rule` |
| K-NN (wk 1–2) | `knn` |
| linear classifiers (wk 4) | `logreg` |
| Naive Bayes (wk 4) | `naive_bayes` |
| trees / ensembles (wk 8) | `random_forest`, `xgboost` |
| MLPs (wk 9) | `mlp` |
| foundation models, 1-D signals (wk 12, 15) | `chronos` (zero-shot / fine-tuned on Colab) |

To add a model, add a config entry with `estimator`, `params`, and optionally `search` and `balance`. No new code is needed.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"            # add ".[chronos]" for torch + Chronos
copy .env.example .env             # set EIA_API_KEY (free: https://www.eia.gov/opendata/)
```

## Workflow
```bash
python -m gridwatch.data.ingest --region ERCOT --start 2021-04-01 --end 2026-10-31   # same for MISO
python scripts/count_events.py                       # check event counts before fixing the test window
python scripts/train.py --model xgboost              # CV search on train, calibrate + set the threshold on val
python scripts/train.py --model xgboost --drop temp_ cdd hdd    # ablation (design-decision comparison)
python scripts/evaluate.py --models rule logreg random_forest xgboost mlp knn naive_bayes
#   -> reports/metrics_*.csv, significance_*.csv (paired bootstrap vs the first model), figures/ (PR + reliability)
```

## Guardrails (from the proposal review)
- Use only weather forecasts **as issued** (the GFS run from the day before), never observed weather.
- The price threshold is the primary label and operator alerts are the secondary label.
- Report both the continuous test period and the curated windows, because balanced windows inflate precision.
- Count events before fixing the test window. Add regions if events are scarce.
