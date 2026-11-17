"""Train one model per region: hyperparameter search on train (TimeSeriesSplit CV), calibrate + pick the alert
threshold on val, save. Chronos runs in a notebook (Colab GPU); see src/gridwatch/models/chronos_model.py.

Usage: python scripts/train.py --model <name in configs/default.yaml models> [--label primary|secondary]
       python scripts/train.py --model xgboost --drop temp_ cdd hdd    # ablation, saved as xgboost-no_temp-no_cdd-no_hdd
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from _common import label_col, prepare
from gridwatch.config import load_config
from gridwatch.evaluate import evaluate, pick_threshold
from gridwatch.models import get_model, score


def best_ignoring_empty_folds(r: dict) -> int:
    """Early time folds can have no stress hours (AP = NaN), so average over the folds that do."""
    folds = np.array([v for k, v in r.items() if k.startswith("split") and k.endswith("_test_score")])
    return int(np.nanargmax(np.nanmean(folds, axis=0)))


def fit(model, spec: dict, X, y, ev: dict):
    fit_params = {}
    if spec.get("balance") == "scale_pos_weight":
        pos = max(int(y.sum()), 1)
        model.set_params(model__scale_pos_weight=(len(y) - pos) / pos)
    elif spec.get("balance") == "sample_weight":  # for models without class_weight (MLP)
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y)
    grid = spec.get("search")
    if not grid:
        return model.fit(X, y, **fit_params)
    gs = GridSearchCV(model, {f"model__{k}": v for k, v in grid.items()}, cv=TimeSeriesSplit(ev["cv_folds"]),
                      scoring="average_precision", refit=best_ignoring_empty_folds, n_jobs=-1)
    gs.fit(X, y, **fit_params)
    folds = np.array([gs.cv_results_[f"split{i}_test_score"][gs.best_index_] for i in range(ev["cv_folds"])])
    print(f"  CV best AP={np.nanmean(folds):.3f} ({np.isnan(folds).sum()} empty folds) {gs.best_params_}")
    return gs.best_estimator_


def calibrate(model, X, y, method):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="primary", choices=["primary", "secondary"])
    ap.add_argument("--drop", nargs="*", default=[],
                    help="ablation: drop feature columns starting with these prefixes, e.g. temp_ rt_price load_fcst")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tag = args.model + "".join(f"-no_{p.rstrip('_')}" for p in args.drop)
    ev, spec, ycol = cfg["evaluation"], cfg["models"][args.model], label_col(cfg, args.label)
    cfg["paths"]["models"].mkdir(parents=True, exist_ok=True)

    for region in cfg["regions"]:
        print(f"[{region}] {tag}")
        d = prepare(region, cfg)
        dropped = [c for c in d["X"].columns if c.startswith(tuple(args.drop))] if args.drop else []
        X, y, m = d["X"].drop(columns=dropped), d["y"][ycol], d["masks"]
        tr, va = m["train"], m["val"]
        model = fit(get_model(args.model, cfg), spec, X[tr], y[tr], ev) if args.model != "rule" \
            else get_model("rule", cfg).fit(X[tr], y[tr])
        if ev.get("calibrate") and args.model != "rule":
            model = calibrate(model, X[va], y[va], ev["calibrate"])

        p_val = pd.Series(score(model, X[va]), index=X.index[va])
        thr = pick_threshold(y[va].values, p_val.values, ev["min_recall"]) \
            if ev["alert_threshold"] == "tune_on_val" else float(ev["alert_threshold"])
        res = evaluate(y[va], p_val, d["issue_time"][va], thr, cfg["label"]["event_merge_gap_hours"])
        print("  val: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))

        path = cfg["paths"]["models"] / f"{tag}_{region}_{ycol}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "threshold": thr, "price_threshold": d["price_threshold"], "label": ycol,
                         "dropped": dropped}, f)


if __name__ == "__main__":
    main()
