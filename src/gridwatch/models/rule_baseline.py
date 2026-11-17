"""Rule baseline (proposal 4c): alert when forecast temperature is extreme AND forecast demand is above its
train 95th percentile. Uses the BA's day-ahead load forecast from EIA-930 (falls back to 48h-lagged load)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class RuleBaseline:
    def __init__(self, temp_hot_f: float = 100, temp_cold_f: float = 15, load_quantile: float = 0.95):
        self.temp_hot_f = temp_hot_f
        self.temp_cold_f = temp_cold_f
        self.load_quantile = load_quantile

    @staticmethod
    def _col(X, *names):
        return next(n for n in names if n in X.columns)

    def fit(self, X: pd.DataFrame, y=None):
        self.load_col_ = self._col(X, "load_fcst_mw", "load_mw_lag48")
        self.load_cut_ = float(X[self.load_col_].quantile(self.load_quantile))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        hot = X[self._col(X, "temp_f_fcst_max", "temp_f_fcst")] >= self.temp_hot_f
        cold = X[self._col(X, "temp_f_fcst_min", "temp_f_fcst")] <= self.temp_cold_f
        p = ((hot | cold) & (X[self.load_col_] >= self.load_cut_)).astype(float).to_numpy()
        return np.column_stack([1 - p, p])
