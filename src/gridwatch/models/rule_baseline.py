"""Rule baseline: alert when the forecast temperature is extreme AND forecast demand is above its 95th percentile.

There is no demand forecast in the feature set yet, so the load at issue time (or its lag) stands in for it.
TODO: swap in the operator's day-ahead load forecast if you get one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RuleBaseline:
    def __init__(self, temp_hot_f: float = 100, temp_cold_f: float = 15, load_quantile: float = 0.95,
                 load_col: str = "load_mw_lag48", temp_col: str = "temp_f_fcst"):
        self.temp_hot_f = temp_hot_f
        self.temp_cold_f = temp_cold_f
        self.load_quantile = load_quantile
        self.load_col = load_col
        self.temp_col = temp_col
        self.load_cut_: float | None = None

    def fit(self, X: pd.DataFrame, y=None, **_):
        self.load_cut_ = float(X[self.load_col].quantile(self.load_quantile))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        t = X[self.temp_col]
        extreme = (t >= self.temp_hot_f) | (t <= self.temp_cold_f)
        high_load = X[self.load_col] >= self.load_cut_
        return (extreme & high_load).astype(float).to_numpy()
