"""Evaluate a saved model on the test set: on the CONTINUOUS test period and on the CURATED windows.

Usage: python scripts/evaluate.py --model xgboost [--label primary|secondary]
"""
from __future__ import annotations

import argparse
import pickle

import pandas as pd

from _common import label_col, prepare, to_frame
from gridwatch.config import load_config
from gridwatch.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="primary", choices=["primary", "secondary"])
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ycol = label_col(cfg, args.label)
    results = {}
    for region in cfg["regions"]:
        with open(cfg["paths"]["models"] / f"{args.model}_{region}_{ycol}.pkl", "rb") as f:
            saved = pickle.load(f)
        d = prepare(region, cfg)
        for view in ("test", "curated"):  # continuous period vs curated (balanced, precision is inflated)
            m = d["masks"][view]
            if not m.any():
                print(f"[{region}] no rows for {view}; skipping")
                continue
            p = pd.Series(saved["model"].predict_proba(d["X"][m]), index=d["X"].index[m])
            results[f"{region}/{view}"] = evaluate(d["y"].loc[m, ycol], p, d["issue_time"][m],
                                                   saved["threshold"], cfg["label"]["event_merge_gap_hours"])
    table = to_frame(results)
    print(table.to_string())
    out = cfg["paths"]["reports"] / f"metrics_{args.model}_{ycol}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
