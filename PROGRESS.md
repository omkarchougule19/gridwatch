# Progress (paused 2026-11-17)

The full log of decisions and findings is in [DECISIONS.md](DECISIONS.md). Append to it each session.

## Done
- The project scaffold is in place: `src/gridwatch` (labels, features, splits, evaluate, models), the scripts, and 9 passing tests (`.venv\Scripts\python -m pytest`).
- A smoke test on synthetic data ran end to end: count_events → train (rule, xgboost) → evaluate.
- `src/gridwatch/data/ingest.py` has been rewritten to pull from the preferred sources through libraries (not run on real data yet):
  - EIA-930 load and the BA's day-ahead load forecast via `gridstatus.EIA`. This needs `EIA_API_KEY` in `.env` (free from eia.gov/opendata).
  - ERCOT RTM/DAM SPP (HB_HUBAVG) via `gridstatus.Ercot`.
  - MISO RT-final / DA hourly LMP (ILLINOIS.HUB) via `gridstatus.MISO`.
  - The GFS day-ahead temperature forecast, as issued, from the Open-Meteo Previous Runs API, averaged over load-center cities (in the config).
  - Alerts from a hand-curated `data/external/alerts.csv` (columns: region,start_utc,end_utc,kind,source).

## Findings on data coverage (these affect the split dates)
- Day-ahead forecast temperature is available from about 2021-04, with some gaps (e.g. 2024-01-01). Heat index and wind are only available from about 2024, so they are dropped.
- MISO daily price files start around 2023-01, so MISO has a shorter training window.
- ERCOT operations messages in gridstatus come from Wayback snapshots, which have gaps. That's why alerts are a curated CSV.
- The full syllabus (cs441_aml_syllabus.pdf) has policy only and no list of topics. The course website schedule returned 404 (it may need a login).
  - Project requirement: define the problem, collect data, compare models, report results.
  - Academic integrity: no code copied from the Internet. Library APIs are fine.

## Next steps (planned, not started)
1. Change the train split start in `configs/default.yaml` from 2019 to 2021-04-01, to match the forecast coverage.
2. Replace `models/xgb.py` with a config-driven sklearn factory (Pipeline of imputer + scaler + estimator). This adds course-style comparisons almost for free: logistic regression, random forest, XGBoost and MLP, alongside the rule baseline and Chronos.
3. `train.py`: optional GridSearchCV with TimeSeriesSplit (scoring=average_precision), plus optional calibration. Remove the XGBoost early-stopping branch.
4. `features.py`: use the `temp_f_fcst*` columns, add `load_fcst_mw`, add is_holiday (the `holidays` package). The rule baseline should use `load_fcst_mw`.
5. Update the README and the tests. Add docs/sources.md with the citations.
6. Get an EIA key, run ingest for ERCOT and MISO, then run `scripts/count_events.py`. Decide on a percentile threshold for MISO, or add regions.
