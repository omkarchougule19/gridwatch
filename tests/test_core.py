import numpy as np
import pandas as pd
import pytest

from gridwatch.data.load import asof_forecasts, issue_times
from gridwatch.evaluate import event_recall, hourly_metrics, pick_threshold
from gridwatch.features import build_features
from gridwatch.labels import find_events, price_label, price_threshold
from gridwatch.models.chronos_model import exceedance_prob
from gridwatch.splits import time_split


def hourly(n=24 * 30, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def test_price_threshold_percentile_uses_given_series():
    s = pd.Series(np.arange(100.0))
    assert price_threshold(s, 1000, 0.99) == pytest.approx(98.01)
    assert price_threshold(s, 1000, None) == 1000


def test_find_events_merges_close_hours():
    idx = hourly(48)
    y = pd.Series(0, index=idx)
    y.iloc[[5, 6, 9, 30]] = 1  # 5,6,9 merge (gap 3 <= 6); 30 separate
    ev = find_events(y, merge_gap_hours=6)
    assert len(ev) == 2
    assert ev.loc[0, "hours"] == 3


def test_issue_times_previous_day_afternoon():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-07-02 10:00", tz="America/Chicago")]).tz_convert("UTC")
    it = issue_times(idx, "America/Chicago", 14)
    assert it.iloc[0] == pd.Timestamp("2024-07-01 14:00", tz="America/Chicago")


def test_asof_forecasts_never_uses_later_issue():
    v = pd.Timestamp("2024-07-02 15:00", tz="UTC")
    fc = pd.DataFrame({
        "issue_time": pd.to_datetime(["2024-07-01 00:00", "2024-07-01 12:00", "2024-07-02 00:00"], utc=True),
        "valid_time": [v, v, v],
        "temp_f_fcst": [90.0, 95.0, 105.0],
    })
    tg = pd.DataFrame({"valid_time": [v], "issue_time": [pd.Timestamp("2024-07-01 19:00", tz="UTC")]})
    assert asof_forecasts(fc, tg)["temp_f_fcst"].iloc[0] == 95.0


def test_features_have_no_future_leakage():
    idx = hourly()
    df = pd.DataFrame({"load_mw": np.arange(len(idx), dtype=float), "rt_price": 30.0, "da_price": 30.0,
                       "alert": 0, "temp_f_fcst": 80.0}, index=idx)
    it = issue_times(idx, "America/Chicago", 14)
    cfg = {"features": {"lags_hours": [48, 72], "rolling_windows_hours": [24]}}
    X = build_features(df, it, cfg, "America/Chicago")
    # load_mw equals the hour number, so every load feature must be < the value at issue time + 1
    i = 500
    hours_at_issue = int((it.iloc[i] - idx[0]) / pd.Timedelta(hours=1))
    assert X["load_mw_at_issue"].iloc[i] == hours_at_issue
    assert X["load_mw_lag48"].iloc[i] <= hours_at_issue
    assert X["load_mw_roll24_mean"].iloc[i] <= hours_at_issue


def test_time_split_disjoint():
    idx = hourly(24 * 400, "2024-01-01")
    m = time_split(idx, {"train": ["2024-01-01", "2024-06-30"], "val": ["2024-07-01", "2024-09-30"],
                         "test": ["2024-10-01", "2025-01-31"]}, "America/Chicago")
    assert not (m["train"] & m["test"]).any()
    assert m["train"].sum() > 0 and m["test"].sum() > 0


def test_hourly_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.6, 0.9])
    thr = pick_threshold(y, p, min_recall=1.0)
    m = hourly_metrics(y, p, thr)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0


def test_event_recall_requires_warning_before_event():
    idx = hourly(72)
    y = pd.Series(0, index=idx); y.iloc[40:43] = 1
    p = pd.Series(0.0, index=idx); p.iloc[41] = 0.9
    early = pd.Series(idx - pd.Timedelta(hours=12), index=idx)  # issued 12h before target
    late = pd.Series(idx, index=idx)                             # issued at the target hour (too late)
    assert event_recall(y, p, early, 0.5)["event_recall"] == 1.0
    assert event_recall(y, p, late, 0.5)["event_recall"] == 0.0


def test_exceedance_prob():
    levels = np.array([0.1, 0.5, 0.9])
    q = np.array([[10.0, 50.0, 90.0]])
    assert exceedance_prob(q, levels, 50.0)[0] == pytest.approx(0.5)
    assert exceedance_prob(q, levels, 1000.0)[0] == 0.0
