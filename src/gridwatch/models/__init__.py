from gridwatch.models.rule_baseline import RuleBaseline
from gridwatch.models.xgb import XGBStress


def get_model(name: str, cfg: dict):
    params = cfg["models"][name]
    if name == "rule":
        return RuleBaseline(**params)
    if name == "xgboost":
        return XGBStress(**params)
    if name == "chronos":
        from gridwatch.models.chronos_model import ChronosStress  # heavy import (torch)

        return ChronosStress(**params)
    raise ValueError(f"unknown model {name!r}")
