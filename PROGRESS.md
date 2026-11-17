# Progress (updated 2026-11-17, 13:20)

The full log of decisions and findings is in [DECISIONS.md](DECISIONS.md). Append to it each session.
**Final project due Sun Dec 13, 11:59pm. It cannot be late.**

## Done
- **Package:** labels, leak-safe features, time splits, metrics, paired block bootstrap. 11 tests passing.
- **Comparison set (config-driven):** rule, kNN, logreg, Naive Bayes, random forest, XGBoost, MLP.
  - `train.py`: CV search, balancing, calibration, threshold, `--drop` ablations.
  - `evaluate.py`: significance and figures.
- **Chronos-2 Colab notebook** (`notebooks/chronos_colab.ipynb`): zero-shot and fine-tuned. Not yet run on a GPU.
- **Real data:**
  - **MISO is complete** (`data/processed/MISO.parquet`, 2023-01 → 2026-10). Threshold set to the train 99th percentile, $111/MWh.
  - **ERCOT** load and weather are cached. **Prices were downloading one year at a time** (parallel downloads failed).

## Resume here
1. **Check the ERCOT price chunks:** `data/interim/chunks/ERCOT_*.parquet` should have 6 files (2021–2026). If any are missing, rerun the ones missing, one at a time:
   `.venv\Scripts\python -I <scratchpad>\fetch_chunk.py ERCOT 2022-01-01 2022-12-31 data/interim/chunks`
   Helpers are in `scripts/download/` (chunk list: `chunks.txt`; live view: `progress.py data/interim/chunks scripts/download/chunks.txt`).
2. `python scripts/download/assemble.py` to assemble and build `data/processed/ERCOT.parquet`, then run `python scripts/count_events.py`. Decide whether ERCOT stays at $1,000 or moves to a percentile.
3. Train all models (`scripts/train.py --model <name>` for rule, logreg, random_forest, xgboost, mlp, knn, naive_bayes) and run `scripts/evaluate.py --models rule ...`.
4. Ablations: `--drop temp_ cdd hdd`, `--drop rt_price da_price`, `--drop load_fcst`, and `--label secondary` once alerts.csv exists.
5. Chronos on Colab (zero-shot, then fine-tuned). Copy `models/chronos-*.pkl` back and include them in `evaluate.py`.
6. Curate `data/external/alerts.csv` and `test_events.csv` (two reviewers per row).
7. Report and 5–6 min video (no AI-generated video or slides). Include group contributions and the code link.
