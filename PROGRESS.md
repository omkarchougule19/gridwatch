# Progress (updated 2026-11-18)

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
  - **ERCOT is complete** (`data/processed/ERCOT.parquet`, 2021-04 → 2026-10). Threshold set to the train 99th percentile, $329/MWh.
  - Event counts at p99, train / val / test: MISO 101 / 89 / 184, ERCOT 105 / 11 / 11 (DECISIONS.md §18–19).
- **First real training runs, all 7 models** (DECISIONS.md §20). Adding tomorrow's DA price raised val PR-AUC 3–6×. Best so far is logreg: ERCOT 0.077, MISO 0.322. Trees lag because of DA drift and out-of-range values.

## Resume here
1. **Team decision:** ERCOT has only about 11 val/test events. Add regions (SPP/PJM via gridstatus), lower ERCOT to p98, or pool regions? (DECISIONS.md §19)
2. ~~Scale-free features + test evaluation~~ done (DECISIONS.md §21–22). MISO best is logreg-no_rel, test PR-AUC 0.343. ERCOT is underpowered (11 events).
3. ~~Feature-group ablations~~ done (DECISIONS.md §23): price/DA carries the skill, while weather/load/calendar add nothing beyond it. Still to run: `--label secondary` once alerts.csv exists.
4. ~~Chronos zero-shot and fine-tuned~~ done (DECISIONS.md §24). MISO test 0.199 / 0.214 vs logreg 0.343. Fine-tuning gave no significant gain. Optional: Chronos-2 with covariates (the DA price).
5. Curate `data/external/alerts.csv` and `test_events.csv` (two reviewers per row).
6. Report and 5–6 min video (no AI-generated video or slides). Include group contributions and the code link.

TA meeting notes and questions: [docs/TA_meeting.md](docs/TA_meeting.md).

Re-downloading data on a new machine: `scripts/download/` (chunked, with retries; ERCOT years must run one at a time). Then `python scripts/download/assemble.py`.
