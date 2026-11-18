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
    return pd.concat({f"{c}_at_issue": value_at(df[c], issue_time) for c in OBS_COLS}, axis=1)


def value_at(s: pd.Series, issue_time: pd.Series) -> pd.Series:
    """For each target hour, the last value of `s` at or before that hour's issue time (leak-safe lookup)."""
    s = s.sort_index()
    pos = s.index.searchsorted(pd.DatetimeIndex(issue_time), side="right") - 1
    vals = s.to_numpy()[np.clip(pos, 0, None)].astype(float)
    vals[pos < 0] = np.nan
    return pd.Series(vals, index=issue_time.index)


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


def relative_features(df: pd.DataFrame, issue_time: pd.Series, tz: str) -> pd.DataFrame:
    """Scale-free price features (prefix rel_), so tree models still work when price levels drift past the training
    range (DECISIONS.md §21). Uses only data known at the issue time: the trailing-week median real-time price up to
    the issue hour, and the target day's DA prices (posted before the issue time)."""
    level = value_at(df["rt_price"].rolling(168, min_periods=24).median(), issue_time).clip(lower=1.0)
    slog = lambda s: np.sign(s) * np.log1p(np.abs(s))  # log that also handles negative prices
    out = pd.DataFrame(index=df.index)
    out["rel_da_target_vs_week"] = df["da_price"] / level
    out["rel_log_da_target"] = slog(df["da_price"])
    out["rel_da_target_rank_in_day"] = df["da_price"].groupby(df.index.tz_convert(tz).date).rank(pct=True)
    out["rel_rt_at_issue_vs_week"] = value_at(df["rt_price"], issue_time) / level
    return out


def build_features(df: pd.DataFrame, issue_time: pd.Series, cfg: dict, tz: str) -> pd.DataFrame:
    f = cfg["features"]
    parts = [
        calendar_features(df.index, tz),
        lag_features(df, f["lags_hours"], f["rolling_windows_hours"]),
        at_issue_features(df, issue_time),
        weather_features(df, tz, f.get("target_da_price", False)),
    ]
    if f.get("relative_prices", False):
        parts.append(relative_features(df, issue_time, tz))
    return pd.concat(parts, axis=1)
