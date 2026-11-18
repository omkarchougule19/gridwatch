"""Feature engineering. Every feature must be known at the forecast's issue time.

Allowed: calendar features of the target hour; weather forecasts as issued (already as-of joined);
observations at or before issue_time; lags of at least the maximum lead time (about 34h, so we use 48h or more).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

OBS_COLS = ["load_mw", "rt_price", "da_price"]
# Known at the issue time: GFS temperature from the day before (mean/max/min over cities) and the BA's day-ahead load forecast
FCST_COLS = ["temp_f_fcst", "temp_f_fcst_max", "temp_f_fcst_min", "load_fcst_mw"]


def calendar_features(index: pd.DatetimeIndex, tz: str) -> pd.DataFrame:
    local = index.tz_convert(tz)
    out = pd.DataFrame(index=index)
    out["hour"] = local.hour
    out["dow"] = local.day_of_week
    out["month"] = local.month
    out["is_weekend"] = (local.day_of_week >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * local.day_of_year / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * local.day_of_year / 365.25)
    us = holidays.country_holidays("US")
    out["is_holiday"] = [d in us for d in local.date]
    return out


def lag_features(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    min_lag = min(lags)
    for col in OBS_COLS:
        for lag in lags:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in windows:
            out[f"{col}_roll{w}_mean"] = df[col].shift(min_lag).rolling(w, min_periods=1).mean()
        out[f"{col}_roll168_max"] = df[col].shift(min_lag).rolling(168, min_periods=1).max()
    out["load_growth_7d"] = df["load_mw"].shift(min_lag) / df["load_mw"].shift(min_lag + 168) - 1
    return out


def at_issue_features(df: pd.DataFrame, issue_time: pd.Series) -> pd.DataFrame:
    """The last observed values at the issue time (the freshest information that is still leak-safe)."""
    obs = df[OBS_COLS].sort_index()
    pos = obs.index.searchsorted(pd.DatetimeIndex(issue_time), side="right") - 1  # last row at or before issue
    vals = pd.DataFrame(obs.to_numpy()[np.clip(pos, 0, None)], index=issue_time.index, columns=OBS_COLS)
    vals[pos < 0] = np.nan
    return vals.add_suffix("_at_issue")


def weather_features(df: pd.DataFrame, tz: str, include_target_da: bool = False) -> pd.DataFrame:
    # No "deviation from normal" feature: normals computed over all years would leak test-period weather.
    # The trees/MLP can learn the month x temperature interaction themselves.
    out = df[[c for c in FCST_COLS if c in df.columns]].copy()
    if "temp_f_fcst" in out:
        out["cdd"] = (out["temp_f_fcst"] - 65).clip(lower=0)
        out["hdd"] = (65 - out["temp_f_fcst"]).clip(lower=0)
    # Day-ahead market price for the TARGET hour: posted ~13:30 CT on D-1 (ERCOT DAM; MISO ~12:30 CT),
    # i.e. before the 14:00 issue time, so it is known when the warning is made (DECISIONS.md §20).
    if include_target_da and "da_price" in df:
        out["da_price_target"] = df["da_price"]
    if "load_fcst_mw" in out:
        out["load_fcst_vs_week_max"] = out["load_fcst_mw"] / df["load_mw"].shift(48).rolling(168, min_periods=1).max()
    return out


def build_features(df: pd.DataFrame, issue_time: pd.Series, cfg: dict, tz: str) -> pd.DataFrame:
    f = cfg["features"]
    return pd.concat(
        [
            calendar_features(df.index, tz),
            lag_features(df, f["lags_hours"], f["rolling_windows_hours"]),
            at_issue_features(df, issue_time),
            weather_features(df, tz, f.get("target_da_price", False)),
        ],
        axis=1,
    )
