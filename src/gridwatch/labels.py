"""Stress labels and event extraction."""
from __future__ import annotations

import pandas as pd


def price_threshold(train_prices: pd.Series, threshold_usd: float | None, percentile: float | None) -> float:
    """The spike threshold. When a percentile is given it is computed on TRAIN prices only, which avoids leakage."""
    if percentile is not None:
        return float(train_prices.quantile(percentile))
    if threshold_usd is None:
        raise ValueError("set price_threshold_usd or price_percentile")
    return float(threshold_usd)


def price_label(rt_price: pd.Series, threshold: float) -> pd.Series:
    return (rt_price >= threshold).astype(int).rename("y_price")


def alert_label(alert: pd.Series) -> pd.Series:
    return alert.fillna(0).astype(int).clip(0, 1).rename("y_alert")


def make_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.concat([price_label(df["rt_price"], threshold), alert_label(df["alert"])], axis=1)


def find_events(y: pd.Series, merge_gap_hours: int = 6) -> pd.DataFrame:
    """Group positive hours into events. Stress hours less than merge_gap_hours apart belong to the same event.

    Returns columns: start, end, hours (the number of positive hours).
    """
    pos = y.index[y.astype(bool)]
    if len(pos) == 0:
        return pd.DataFrame(columns=["start", "end", "hours"])
    gaps = pd.Series(pos).diff() > pd.Timedelta(hours=merge_gap_hours)
    event_id = gaps.cumsum().to_numpy()
    s = pd.Series(pos, index=event_id)
    g = s.groupby(level=0)
    return pd.DataFrame({"start": g.min(), "end": g.max(), "hours": g.size()}).reset_index(drop=True)
