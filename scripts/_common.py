"""Shared helper for the scripts: builds X, labels, issue times and split masks for one region."""
from __future__ import annotations

import pandas as pd

from gridwatch.data.load import issue_times, load_region
from gridwatch.features import build_features
from gridwatch.labels import make_labels, price_threshold
from gridwatch.splits import curated_mask, time_split


def prepare(region: str, cfg: dict) -> dict:
    tz = cfg["issue"]["timezone"][region]
    df = load_region(region, cfg["paths"]["processed"])
    it = issue_times(df.index, tz, cfg["issue"]["hour_local"])
    masks = time_split(df.index, cfg["splits"], tz)
    lab = cfg["label"]
    thr = price_threshold(df.loc[masks["train"], "rt_price"], lab["price_threshold_usd"], lab["price_percentile"])
    y = make_labels(df, thr)
    X = build_features(df, it, cfg, tz)
    masks["curated"] = curated_mask(df.index, region, cfg["paths"]["external"] / "test_events.csv", tz) & masks["test"]
    return {"df": df, "X": X, "y": y, "issue_time": it, "masks": masks, "price_threshold": thr, "tz": tz}


def label_col(cfg: dict, which: str = "primary") -> str:
    return "y_" + cfg["label"][which]


def to_frame(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T
