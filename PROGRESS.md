# Progress (updated 2026-11-17)

The full log of decisions and findings is in [DECISIONS.md](DECISIONS.md). Append to it each session.
**Final project due Sun Dec 13, 11:59pm. It cannot be late.**

## Done
- **Package:** labels, leak-safe features (including the holiday flag and the BA load forecast), time splits, metrics, and a paired block bootstrap.
- **Data ingest** from EIA-930, ERCOT/MISO prices (gridstatus) and GFS day-ahead forecasts (Open-Meteo). Written but **not yet run on real data**.
- **Comparison set from the config:** rule, kNN, logreg, Naive Bayes, random forest, XGBoost, MLP. Plus Chronos code (zero-shot).
- **train.py:** TimeSeriesSplit grid search (skipping folds with no stress hours), class balancing, isotonic calibration on val, threshold tuned on val, and `--drop` ablations.
- **evaluate.py:** compares several models, reports significance vs a reference model, and saves PR and reliability figures.
- **Checks:** 11 passing tests. A synthetic end-to-end smoke test passed for every model.

## Next steps
1. **Get an EIA API key** into `.env`, then run ingest for ERCOT and MISO. Check the missing-data shares that get printed.
2. `python scripts/count_events.py`. Then choose MISO's threshold (probably `price_percentile`), or add hubs or regions.
3. Curate `data/external/alerts.csv` (the secondary label) and `test_events.csv` (the curated windows). Two reviewers per row.
4. Train every model and run `evaluate.py`, plus the ablations: weather, price-at-issue, load forecast, and the label.
5. Chronos notebook on Colab: zero-shot first, then fine-tuned.
6. Report from the template and the 5–6 min video (no AI-generated video or slides). Include group contributions and the code link.
