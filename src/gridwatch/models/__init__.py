"""Model factory. sklearn-style models are built from the config (`estimator` class path + `params`), so adding
a model to the comparison is a config change, not new code."""
from __future__ import annotations

import importlib

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridwatch.models.rule_baseline import RuleBaseline


def get_model(name: str, cfg: dict):
    spec = cfg["models"][name]
    if name == "rule":
        return RuleBaseline(**spec)
    if name == "chronos":
        from gridwatch.models.chronos_model import ChronosStress  # heavy import (torch)

        return ChronosStress(**spec)
    module, cls = spec["estimator"].rsplit(".", 1)
    est = getattr(importlib.import_module(module), cls)(**spec.get("params", {}))
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", est),
    ])


def score(model, X) -> np.ndarray:
    """P(stress) as a 1-D array for any model in the comparison."""
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p
