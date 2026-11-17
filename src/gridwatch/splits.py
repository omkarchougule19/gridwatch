"""Time-based splits. The model never trains on data after its validation or test period."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _mask(index: pd.DatetimeIndex, tz: str, start: str, end: str) -> pd.Series:
    local_date = index.tz_convert(tz).normalize().tz_localize(None)
    return pd.Series((local_date >= pd.Timestamp(start)) & (local_date <= pd.Timestamp(end)), index=index)


def time_split(index: pd.DatetimeIndex, splits: dict, tz: str) -> dict[str, pd.Series]:
    """Boolean masks for train/val/test (the test mask is the CONTINUOUS period)."""
    masks = {name: _mask(index, tz, *splits[name]) for name in ("train", "val", "test")}
    assert not (masks["train"] & masks["val"]).any() and not (masks["val"] & masks["test"]).any()
    return masks


def curated_mask(index: pd.DatetimeIndex, region: str, windows_csv: Path, tz: str) -> pd.Series:
    """Mask for the hand-curated extreme-weather windows (plus their matched normal days) in test_events.csv."""
    w = pd.read_csv(windows_csv, comment="#")
    w = w[w["region"] == region]
    mask = pd.Series(False, index=index)
    local = index.tz_convert(tz).tz_localize(None)
    for _, row in w.iterrows():
        mask |= (local >= pd.Timestamp(row["start_local"])) & (local <= pd.Timestamp(row["end_local"]))
    return mask
