"""Train a model per region on the train split, tune the alert threshold on val, and save it.

Usage: python scripts/train.py --model {rule,xgboost} [--label primary|secondary]
(For Chronos, use a notebook on Colab; see src/gridwatch/models/chronos_model.py.)
"""
from __future__ import annotations

import argparse
import pickle

import pandas as pd

from _common import label_col, prepare
from gridwatch.config import load_config
from gridwatch.evaluate import evaluate, pick_threshold
from gridwatch.models import get_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rule", "xgboost"])
    ap.add_argument("--label", default="primary", choices=["primary", "secondary"])
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ycol = label_col(cfg, args.label)
    out_dir = cfg["paths"]["models"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for region in cfg["regions"]:
        d = prepare(region, cfg)
        X, y, m = d["X"], d["y"][ycol], d["masks"]
        model = get_model(args.model, cfg)
        if args.model == "xgboost":
            model.fit(X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]])
        else:
            model.fit(X[m["train"]], y[m["train"]])

        p_val = model.predict_proba(X[m["val"]])
        thr_cfg = cfg["evaluation"]["alert_threshold"]
        thr = pick_threshold(y[m["val"]].values, p_val, cfg["evaluation"]["min_recall"]) if thr_cfg == "tune_on_val" else float(thr_cfg)
        res = evaluate(y[m["val"]], pd.Series(p_val, index=y[m["val"]].index),
                       d["issue_time"][m["val"]], thr, cfg["label"]["event_merge_gap_hours"])
        print(f"[{region}] val: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))

        path = out_dir / f"{args.model}_{region}_{ycol}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "threshold": thr, "price_threshold": d["price_threshold"], "label": ycol}, f)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
