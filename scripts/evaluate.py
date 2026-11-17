"""Compare saved models on the test set, on the CONTINUOUS period and on the CURATED windows. Outputs:
  - a metrics table per model (PR-AUC, P/R at the threshold, event recall, Brier)
  - significance: a paired day-block bootstrap of PR-AUC vs the first model listed (the reference)
  - figures: PR curves and reliability diagrams per region/view (reports/figures)

Usage: python scripts/evaluate.py --models rule logreg random_forest xgboost mlp [--label primary|secondary]
Ablations trained with `train.py --drop ...` are listed by their tagged name, e.g. xgboost-no_temp.
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay

from _common import label_col, prepare
from gridwatch.config import load_config
from gridwatch.evaluate import evaluate, paired_bootstrap
from gridwatch.models import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="the first one is the significance reference")
    ap.add_argument("--label", default="primary", choices=["primary", "secondary"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ycol, gap = label_col(cfg, args.label), cfg["label"]["event_merge_gap_hours"]
    reports = cfg["paths"]["reports"]
    (reports / "figures").mkdir(parents=True, exist_ok=True)

    metrics, sig = [], []
    for region in cfg["regions"]:
        d = prepare(region, cfg)
        saved = {}
        for name in args.models:
            with open(cfg["paths"]["models"] / f"{name}_{region}_{ycol}.pkl", "rb") as f:
                saved[name] = pickle.load(f)
        for view in ("test", "curated"):  # continuous period vs curated windows (balanced, so precision is inflated)
            m = d["masks"][view]
            if not m.any():
                print(f"[{region}] no rows for {view}; skipping")
                continue
            y, it = d["y"].loc[m, ycol], d["issue_time"][m]
            preds = {}
            for name, s in saved.items():
                X = d["X"][m].drop(columns=s.get("dropped", []))
                preds[name] = pd.Series(score(s["model"], X), index=y.index)
                metrics.append({"region": region, "view": view, "model": name,
                                **evaluate(y, preds[name], it, s["threshold"], gap)})
            if y.sum() and len(preds) > 1:
                b = paired_bootstrap(y, preds, ref=args.models[0], n=args.n_boot)
                sig.append(b.assign(region=region, view=view).rename_axis("model").reset_index())
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
            for name, p in preds.items():
                PrecisionRecallDisplay.from_predictions(y, p, name=name, ax=a1)
                CalibrationDisplay.from_predictions(y, p, n_bins=10, strategy="quantile", name=name, ax=a2)
            a1.set_title(f"{region} {view}: precision-recall (base rate {y.mean():.3%})")
            a2.set_title("reliability")
            for a in (a1, a2):
                a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
            fig.tight_layout()
            fig.savefig(reports / "figures" / f"pr_calib_{region}_{view}_{ycol}.png", dpi=120)
            plt.close(fig)

    tag = f"{ycol}"
    table = pd.DataFrame(metrics)
    table.to_csv(reports / f"metrics_{tag}.csv", index=False)
    cols = ["region", "view", "model", "pr_auc", "precision", "recall", "event_recall", "false_alarms_per_week", "brier"]
    print(table[cols].round(3).to_string(index=False))
    if sig:
        sig = pd.concat(sig)
        sig.to_csv(reports / f"significance_{tag}.csv", index=False)
        print(f"\nPaired day-block bootstrap of PR-AUC vs {args.models[0]} (p < 0.05 means a significant difference):")
        print(sig.round(3).to_string(index=False))
    print(f"\nsaved {reports}/metrics_{tag}.csv, significance_{tag}.csv, figures/")


if __name__ == "__main__":
    main()
