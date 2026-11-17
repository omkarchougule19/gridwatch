"""Metrics: PR-AUC (primary), precision/recall at the alert threshold, event-level recall, Brier score."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from gridwatch.labels import find_events


def pick_threshold(y: np.ndarray, p: np.ndarray, min_recall: float) -> float:
    """The highest threshold whose recall is at least min_recall (that gives the best precision at that recall). Tune it on VAL."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = rec[:-1] >= min_recall
    return float(thr[ok].max()) if ok.any() else float(thr.min())


def hourly_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y, p)) if y.any() else float("nan"),
        "base_rate": float(y.mean()),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "threshold": threshold,
        "n_hours": int(len(y)),
        "n_pos_hours": int(y.sum()),
    }


def event_recall(
    y: pd.Series, p: pd.Series, issue_time: pd.Series, threshold: float, merge_gap_hours: int = 6, lead_hours: int = 24
) -> dict:
    """Did we warn at least once in the 24 hours before each event?

    An event is caught if some prediction with p >= threshold was issued in [start - lead_hours, start)
    and targets an hour inside the event.
    """
    events = find_events(y, merge_gap_hours)
    alerts = p >= threshold
    caught = 0
    for ev in events.itertuples():
        in_event = (p.index >= ev.start) & (p.index <= ev.end)
        issued_ok = (issue_time >= ev.start - pd.Timedelta(hours=lead_hours)) & (issue_time < ev.start)
        caught += bool((alerts & in_event & issued_ok).any())
    n = len(events)
    # False alarm "episodes": alert runs that don't overlap any true stress hour
    alarm_events = find_events(alerts.astype(int), merge_gap_hours)
    false_alarms = sum(not y.loc[a.start : a.end].any() for a in alarm_events.itertuples())
    weeks = (p.index.max() - p.index.min()) / pd.Timedelta(weeks=1) if len(p) else np.nan
    return {
        "n_events": n,
        "events_caught": caught,
        "event_recall": caught / n if n else float("nan"),
        "false_alarm_episodes": int(false_alarms),
        "false_alarms_per_week": false_alarms / weeks if weeks else float("nan"),
    }


def paired_bootstrap(y: pd.Series, preds: dict[str, pd.Series], ref: str, n: int = 1000, block: str = "D",
                     seed: int = 0) -> pd.DataFrame:
    """Is model X's PR-AUC really different from `ref`'s? Paired block bootstrap over days.

    Days are resampled with replacement and every model is scored on the same resample. Whole days are
    resampled because hours inside a day are correlated, so an i.i.d. bootstrap would be overconfident.
    Returns a 95% CI per model, plus the difference vs ref with its CI and a two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    days = y.index.floor(block)
    uniq = days.unique()
    rows = {d: np.flatnonzero(days == d) for d in uniq}
    names = list(preds)
    stats = []
    for _ in range(n):
        idx = np.concatenate([rows[d] for d in rng.choice(uniq, len(uniq))])
        yy = y.values[idx]
        if yy.sum() == 0:
            continue
        stats.append([average_precision_score(yy, preds[m].values[idx]) for m in names])
    s = pd.DataFrame(stats, columns=names)
    out = pd.DataFrame({"pr_auc_lo": s.quantile(0.025), "pr_auc_hi": s.quantile(0.975)})
    diff = s.sub(s[ref], axis=0)
    out[f"diff_vs_{ref}"] = diff.mean()
    out["diff_lo"], out["diff_hi"] = diff.quantile(0.025), diff.quantile(0.975)
    out["p_value"] = (2 * np.minimum((diff <= 0).mean(), (diff >= 0).mean())).clip(upper=1)
    out.loc[ref, ["diff_lo", "diff_hi", "p_value"]] = np.nan
    return out


def evaluate(y: pd.Series, p: pd.Series, issue_time: pd.Series, threshold: float, merge_gap_hours: int = 6) -> dict:
    return {**hourly_metrics(y.values, p.values, threshold), **event_recall(y, p, issue_time, threshold, merge_gap_hours)}
