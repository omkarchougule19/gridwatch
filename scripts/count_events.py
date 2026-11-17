"""Count stress events per region and split. Run this BEFORE fixing the test window (TA feedback).

Usage: python scripts/count_events.py [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse

import pandas as pd

from _common import prepare
from gridwatch.config import load_config
from gridwatch.labels import find_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gap = cfg["label"]["event_merge_gap_hours"]
    rows = []
    for region in cfg["regions"]:
        d = prepare(region, cfg)
        for split in ("train", "val", "test"):
            m = d["masks"][split]
            for col in ("y_price", "y_alert"):
                y = d["y"].loc[m, col]
                rows.append({"region": region, "split": split, "label": col, "pos_hours": int(y.sum()),
                             "base_rate": round(float(y.mean()), 5), "events": len(find_events(y, gap)),
                             "price_threshold": d["price_threshold"]})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
