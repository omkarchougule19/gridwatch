# GridWatch

A 24-hour early warning for power grid stress and price spikes on ERCOT and MISO. This is the CS 441 (Fall 2026) project.

For each hour of the next day, GridWatch predicts the probability that the grid will be stressed. It is run once each afternoon, at the "issue time".

- **Primary label:** the real-time price goes above a spike threshold (see `configs/default.yaml`).
- **Secondary label:** the operator issues an emergency alert, notice or conservation appeal.

## Layout

```
configs/default.yaml        regions, label thresholds, split dates, model params
data/raw/                   untouched pulls (EIA load, prices, weather forecasts, alerts)
data/interim/               cleaned per-source tables
data/processed/             one hourly modeling table per region (parquet)
data/external/              hand-curated files, e.g. test_events.csv
notebooks/                  EDA and one-off analysis
reports/figures/            plots for the write-up
models/                     saved model artifacts
scripts/                    command-line entry points (count_events, train, evaluate)
src/gridwatch/
  config.py                 YAML config loader
  data/load.py              reads the hourly table, as-of join for issued forecasts
  data/ingest.py            source pulls (wire up to the existing Azure pipeline)
  labels.py                 price and alert labels, event extraction
  features.py               leak-safe calendar, lag and weather features
  splits.py                 time-based train/val/test split, curated test windows
  models/                   rule baseline, XGBoost, Chronos
  evaluate.py               PR-AUC, P/R at threshold, event-level recall, Brier
tests/                      pytest unit tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"         # core + tests
pip install -e ".[chronos]"     # torch + chronos (use a GPU, e.g. Colab, to fine-tune)
cp .env.example .env            # add your EIA API key, etc.
```

## Workflow

1. Ingest the raw data, then build `data/processed/{region}.parquet` (schema below).
2. `python scripts/count_events.py`: count the stress events in each split. If the 2025–26 window has only a handful, add regions (TA feedback).
3. `python scripts/train.py --model xgboost` (or `rule`, `chronos`)
4. `python scripts/evaluate.py --model xgboost`: reports results on the **continuous** test period and on the **curated** extreme-weather windows.

### Processed table schema (one row per hour, UTC `ts` index)

| column | meaning |
|---|---|
| `load_mw` | actual hourly load (EIA) |
| `rt_price` | real-time price at the hub or system level, $/MWh |
| `da_price` | day-ahead price, $/MWh |
| `alert` | 1 if an operator emergency alert, notice or conservation appeal is in effect |
| `temp_f_fcst`, `heat_index_fcst`, `wind_fcst` | weather forecast **as issued** before the issue time |

## Guardrails from the proposal review

- Use weather forecasts **as they were issued**, never the observed weather (`data/load.py: asof_forecasts`).
- Make the price threshold the primary label and operator alerts the secondary label.
- Report results on a continuous period as well as the curated (balanced) windows, because balancing inflates precision.
- Count the events before fixing the test window. If they are scarce, add regions.
