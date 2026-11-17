"""Load the processed hourly table and attach issue times / as-issued forecasts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["load_mw", "rt_price", "da_price", "alert"]


def load_region(region: str, processed_dir: Path) -> pd.DataFrame:
    """Read data/processed/{region}.parquet; returns an hourly frame with a tz-aware UTC index."""
    df = pd.read_parquet(Path(processed_dir) / f"{region}.parquet")
    if "ts" in df.columns:
        df = df.set_index("ts")
    df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{region}: missing columns {sorted(missing)}")
    return df.sort_index()


def issue_times(index: pd.DatetimeIndex, tz: str, hour_local: int) -> pd.Series:
    """For each target hour, the time the forecast for it was issued: hour_local on the previous local day."""
    local = index.tz_convert(tz)
    issue_local = local.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=hour_local)
    return pd.Series(issue_local.tz_convert("UTC"), index=index, name="issue_time")


def asof_forecasts(forecasts: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Pick, for each (valid_time, issue_time) target, the latest forecast issued at or before issue_time.

    forecasts: columns issue_time, valid_time, <forecast vars>
    targets:   columns valid_time, issue_time
    This makes sure the model only sees weather that was known when the warning would have been issued.
    """
    f = forecasts.sort_values("issue_time")
    t = targets.reset_index(drop=True).sort_values("issue_time")
    out = pd.merge_asof(t, f, on="issue_time", by="valid_time", direction="backward")
    return out.sort_values("valid_time").reset_index(drop=True)
