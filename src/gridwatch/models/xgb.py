"""XGBoost classifier with class weighting for the rare stress events."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class XGBStress:
    def __init__(self, scale_pos_weight="auto", early_stopping_rounds: int | None = 50, **xgb_params):
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params
        self.model: XGBClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        spw = self.scale_pos_weight
        if spw == "auto":
            pos = max(int(y.sum()), 1)
            spw = (len(y) - pos) / pos
        use_es = X_val is not None and self.early_stopping_rounds
        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=spw,
            early_stopping_rounds=self.early_stopping_rounds if use_es else None,
            tree_method="hist",
            **self.xgb_params,
        )
        eval_set = [(X_val, y_val)] if use_es else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Class weighting distorts the probabilities. TODO: calibrate on val (isotonic) before reporting the Brier score.
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_names_in_).sort_values(ascending=False)
