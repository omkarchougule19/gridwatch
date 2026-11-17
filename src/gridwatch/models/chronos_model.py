"""Chronos: forecast the next 24h of price with a pretrained time-series foundation model, then turn the
quantile forecast into P(price >= spike threshold).

Zero-shot: use the pretrained weights as they are. Fine-tuned: see the TODO in `fit` (needs a GPU, e.g. Colab).
Requires: pip install -e ".[chronos]"
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def exceedance_prob(quantiles: np.ndarray, levels: np.ndarray, threshold: float) -> np.ndarray:
    """P(X >= threshold) from the predicted quantiles (... x n_levels), by linear interpolation of the CDF.

    Outside the predicted range the result is 0 (threshold above the top quantile) or 1 (below the bottom one).
    The tails are coarse, so consider adding more extreme quantile levels (e.g. 0.95, 0.99) if the model supports them.
    """
    q = np.sort(quantiles, axis=-1)
    flat_q = q.reshape(-1, q.shape[-1])
    cdf = np.array([np.interp(threshold, row, levels, left=0.0, right=1.0) for row in flat_q])
    return (1.0 - cdf).reshape(q.shape[:-1])


class ChronosStress:
    def __init__(self, model_id: str = "amazon/chronos-bolt-small", context_hours: int = 512,
                 target: str = "rt_price", quantile_levels=None, threshold: float | None = None, device: str | None = None):
        self.model_id = model_id
        self.context_hours = context_hours
        self.target = target
        self.levels = np.asarray(quantile_levels or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.threshold = threshold
        self.device = device
        self.pipeline = None

    def _load(self):
        import torch
        from chronos import BaseChronosPipeline

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map=device, torch_dtype=torch.float32)

    def fit(self, series: pd.Series | None = None, **_):
        if self.pipeline is None:
            self._load()
        # TODO fine-tuning: train on the train-split price/load series (see the chronos-forecasting
        # training scripts or AutoGluon's Chronos fine-tune), then reload from the checkpoint.
        return self

    def predict_proba_from_history(self, history: pd.Series, issue_times, horizon: int = 34,
                                   batch_size: int = 64) -> pd.DataFrame:
        """For each issue time, forecast `horizon` hours ahead using only the history up to that time.

        horizon=34 covers 14:00 D-1 through 23:00 D. Returns long rows: issue_time, target_time, p.
        Missing prices in the context stay NaN; Chronos treats them as missing values.
        """
        import torch

        if self.pipeline is None:
            self._load()
        if self.threshold is None:
            raise ValueError("set threshold (the same price threshold used for the labels)")
        history = history.sort_index()
        rows = []
        issue_times = list(issue_times)
        for i in range(0, len(issue_times), batch_size):
            batch = issue_times[i:i + batch_size]
            ctxs = [history.loc[:t0].iloc[-self.context_hours:] for t0 in batch]
            quantiles, _ = self.pipeline.predict_quantiles(
                [torch.tensor(c.to_numpy(), dtype=torch.float32) for c in ctxs],
                prediction_length=horizon, quantile_levels=self.levels.tolist())
            # Bolt returns a (batch, horizon, q) tensor; Chronos-2 returns a list of (n_variates, horizon, q) tensors.
            q = [t.float().cpu().numpy().reshape(-1, horizon, len(self.levels))[0] for t in quantiles]
            for t0, c, qi in zip(batch, ctxs, q):
                rows.append(pd.DataFrame({
                    "issue_time": t0,
                    "target_time": pd.date_range(c.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"),
                    "p": exceedance_prob(qi, self.levels, self.threshold)}))
        return pd.concat(rows, ignore_index=True)

    def predict_for_targets(self, history: pd.Series, issue_time: pd.Series, **kw) -> pd.Series:
        """P(stress) for each target hour in `issue_time`'s index, from the forecast issued at that hour's issue time."""
        long = self.predict_proba_from_history(history, issue_time.unique(), **kw)
        want = pd.DataFrame({"issue_time": issue_time.reset_index(drop=True), "target_time": issue_time.index})
        got = want.merge(long, on=["issue_time", "target_time"], how="left")
        return pd.Series(got["p"].to_numpy(), index=issue_time.index, name="p")
