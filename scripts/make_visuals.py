"""Explainer figures: what the data looks like, how it relates to the task, and the headline results.

Writes PNGs to docs/visuals/ (see docs/visuals/README.md). Reuses prepare() and the saved models and report CSVs.
Usage: python scripts/make_visuals.py
"""
from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import prepare
from gridwatch.config import PROJECT_ROOT, load_config
from gridwatch.models import score

OUT = PROJECT_ROOT / "docs" / "visuals"
# Validated categorical palette (dataviz reference, light mode), always used in this order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED, GRID = "#b9b8b2", "#0b0b0b", "#52514e", "#e6e5e0"
SPLIT_SHADE = {"train": "#f3f2ee", "val": "#e8f0fb", "test": "#fdeee6"}
NAMES = {"rule": "Rule baseline", "knn": "kNN", "naive_bayes": "Naive Bayes", "logreg-no_rel": "Logistic regression",
         "random_forest": "Random forest", "xgboost": "XGBoost", "mlp": "MLP",
         "chronos-zeroshot": "Chronos-2 zero-shot", "chronos-finetuned": "Chronos-2 fine-tuned"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.4, "axes.axisbelow": True,
})


def shade_splits(ax, cfg, label=True):
    for name in ("train", "val", "test"):
        a, b = cfg["splits"][name]
        ax.axvspan(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), color=SPLIT_SHADE[name], zorder=0, lw=0)
        if label:
            ax.text(pd.Timestamp(a, tz="UTC"), 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top",
                    fontsize=8, color=MUTED)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def fig_prices(data, cfg):
    """01: daily max real-time price over time, the spike threshold, and the time splits."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for ax, (r, d) in zip(axes, data.items()):
        p = d["df"]["rt_price"].resample("D").max().clip(lower=5)  # log scale; the rare negative daily max shows as 5
        shade_splits(ax, cfg)
        ax.plot(p.index, p, color=BLUE, lw=0.8, label="Daily max real-time price")
        ax.axhline(d["price_threshold"], color=ORANGE, lw=1.6, ls="--",
                   label=f"Spike threshold: ${d['price_threshold']:.0f}/MWh (train p99)")
        ax.set_yscale("log")
        ax.set_ylim(5, 10000)
        ax.set_ylabel("$/MWh (log scale)")
        ax.set_title(f"{r}: real-time hub price")
        ax.legend(loc="lower right", fontsize=8, ncols=2)
        if p.index.min() > pd.Timestamp("2022-01-01", tz="UTC"):
            ax.text(pd.Timestamp("2021-06-01", tz="UTC"), 30, "(no MISO price files before 2023)", fontsize=8,
                    color=MUTED)
    fig.suptitle("Prices are usually $20–50/MWh; GridWatch looks for the rare hours above the orange line",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "01_prices_and_threshold.png")


def fig_spikes_per_month(data, cfg):
    """02: how many spike hours each month; shows the opposite drift of the two regions."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    for ax, (r, d) in zip(axes, data.items()):
        y = d["y"]["y_price"].dropna().resample("MS").sum()
        shade_splits(ax, cfg)
        ax.bar(y.index, y.values, width=24, color=BLUE, align="edge")
        ax.set_ylabel("spike hours / month")
        ax.set_title(f"{r}: spike hours per month")
    axes[0].annotate("ERCOT spikes almost stop after 2023", xy=(pd.Timestamp("2025-03-01", tz="UTC"), 5),
                     xytext=(pd.Timestamp("2024-06-01", tz="UTC"), 40), fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.suptitle("The two regions drift in opposite directions: fewer spikes in ERCOT, more in MISO",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "02_spike_hours_per_month.png")


def fig_when(data, cfg):
    """03: spike rate by hour of day and month (when do spikes happen?)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (r, d) in zip(axes, data.items()):
        y = d["y"]["y_price"].dropna()
        local = y.index.tz_convert("America/Chicago")
        tab = y.groupby([local.hour, local.month]).mean().unstack() * 100
        im = ax.imshow(tab.values, aspect="auto", cmap="Blues", origin="lower")
        ax.set_xticks(range(12), list("JFMAMJJASOND"))
        ax.set_yticks(range(0, 24, 3))
        ax.set_xlabel("month")
        ax.set_ylabel("hour of day (Central)")
        ax.set_title(f"{r}: % of hours that are spikes")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="% spike hours", shrink=0.85)
    fig.suptitle("Spikes cluster in summer late afternoons; MISO also spikes on winter mornings and evenings",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "03_when_spikes_happen.png")


def fig_da_vs_rt(data, cfg):
    """04: tomorrow's day-ahead price vs the real-time price: the strongest signal, and its drift."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (r, d) in zip(axes, data.items()):
        df = d["df"]
        for split, color in (("train", GRAY), ("test", BLUE)):
            m = d["masks"][split]
            ax.scatter(df.loc[m, "da_price"], df.loc[m, "rt_price"], s=4, alpha=0.35, color=color, lw=0,
                       label=split, rasterized=True)
        t = d["price_threshold"]
        ax.axhline(t, color=ORANGE, ls="--", lw=1.4, label=f"spike threshold ${t:.0f}")
        ax.set_xscale("symlog", linthresh=20)
        ax.set_yscale("symlog", linthresh=20)
        ax.set_xlim(-30, None)
        ax.set_ylim(-30, None)
        ax.set_xlabel("day-ahead price for that hour ($/MWh)")
        ax.set_ylabel("real-time price ($/MWh)")
        ax.set_title(r)
        ax.legend(fontsize=8, loc="upper left", markerscale=3)
    fig.suptitle("A high day-ahead price usually means a real-time spike, and in MISO test (blue) it goes beyond "
                 "the training range (gray)", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "04_dayahead_vs_realtime.png")


def fig_weather_load(data, cfg):
    """05: forecast temperature vs load, spikes highlighted: the classic 'hot or cold means stress' story."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (r, d) in zip(axes, data.items()):
        df, y = d["df"], d["y"]["y_price"]
        ok = y.notna()
        ax.scatter(df.loc[ok & (y == 0), "temp_f_fcst"], df.loc[ok & (y == 0), "load_mw"] / 1000, s=3, alpha=0.15,
                   color=GRAY, lw=0, label="normal hour", rasterized=True)
        ax.scatter(df.loc[y == 1, "temp_f_fcst"], df.loc[y == 1, "load_mw"] / 1000, s=10, color=ORANGE, lw=0,
                   label="spike hour")
        ax.set_xlabel("forecast temperature, average of cities (°F)")
        ax.set_ylabel("load (GW)")
        ax.set_title(r)
        ax.legend(fontsize=8, loc="upper left", markerscale=2)
    fig.suptitle("Load rises with heat and cold; spikes sit at the high-load ends, but many high-load hours are calm",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "05_temperature_load_spikes.png")


def fig_timeline():
    """06: what is known when the forecast is issued (the leakage rule), as a timeline over days D-2, D-1, D."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    rows = [  # (label, start h, end h, color, hatch)  hours from 00:00 on D-2
        ("Observed prices & load\n(everything up to 2 PM D-1)", 0, 38, BLUE, None),
        ("Never used: real-time data\nafter the issue time", 38, 72, GRAY, "//"),
        ("Published before 2 PM, about day D:\nweather + load forecasts, DA prices", 48, 72, AQUA, None),
        ("PREDICTED: P(spike) for\neach hour of day D", 48, 72, ORANGE, None),
    ]
    for i, (lab, a, b, c, h) in enumerate(rows):
        ax.barh(len(rows) - 1 - i, b - a, left=a, height=0.55, color=c, hatch=h, edgecolor="white" if h else c,
                alpha=0.9 if not h else 0.5)
    ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows], fontsize=9)
    ax.axvline(38, color=ORANGE, lw=2)
    ax.text(38.5, 1.5, "issue time:\n2 PM on D-1", ha="left", va="center", color=ORANGE, fontweight="bold", fontsize=9)
    for x in (24, 48):
        ax.axvline(x, color=GRAY, lw=0.8, ls=":")
    ax.set_xticks([12, 36, 60], ["day D-2", "day D-1", "day D (target)"])
    ax.set_xlim(0, 72)
    ax.grid(False)
    ax.set_title("No peeking: every input is available at 2 PM the day before\n(lagged features use data at least "
                 "48 h before the target hour)", loc="left", fontsize=10, pad=10)
    save(fig, "06_what_is_known_when.png")


def fig_example_week(cfg, region="MISO"):
    """07: one test week: actual price (top) and the model's spike probability (bottom). Two panels, no dual axis."""
    d = prepare(region, cfg)
    s = pickle.load(open(cfg["paths"]["models"] / f"logreg-no_rel_{region}_y_price.pkl", "rb"))
    m = d["masks"]["test"]
    p = pd.Series(score(s["model"], d["X"][m].drop(columns=s["dropped"])), index=d["X"].index[m])
    y = d["y"]["y_price"][m]
    best = y.rolling(168).sum().idxmax()  # the test week with the most spike hours
    w = slice(best - pd.Timedelta(hours=167), best)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    rt = d["df"]["rt_price"][w]
    a1.plot(rt.index, rt, color=BLUE, label="real-time price")
    a1.axhline(d["price_threshold"], color=ORANGE, ls="--", label=f"spike threshold ${d['price_threshold']:.0f}")
    a1.set_ylabel("$/MWh")
    a1.set_title(f"{region} test week with the most spikes: what happened")
    a1.legend(fontsize=8, loc="upper left")
    a2.plot(p[w].index, p[w], color=BLUE, label="predicted P(spike), issued 2 PM the day before")
    a2.fill_between(y[w].index, 0, y[w].fillna(0) * p[w].max() * 1.05, step="mid", color=ORANGE, alpha=0.18,
                    lw=0, label="actual spike hours")
    a2.axhline(s["threshold"], color=MUTED, ls=":", label=f"alert threshold {s['threshold']:.2f} (tuned on val)")
    a2.set_ylabel("probability")
    a2.set_title("what the logistic regression predicted a day ahead")
    a2.legend(fontsize=8, loc="upper left")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz="America/Chicago"))
    save(fig, "07_example_test_week.png")


def fig_models(cfg):
    """08: test PR-AUC per model with 95% bootstrap CIs, and the base rate (= a random guesser's PR-AUC)."""
    rep = cfg["paths"]["reports"]
    met = pd.concat([pd.read_csv(rep / "metrics_y_price.csv"), pd.read_csv(rep / "metrics_y_price_chronos.csv")])
    sig = pd.concat([pd.read_csv(rep / "significance_y_price_vs_rule.csv"),
                     pd.read_csv(rep / "significance_y_price_chronos.csv")])
    met = met[met.view == "test"].drop_duplicates(["region", "model"]).set_index(["region", "model"])
    sig = sig[sig.view == "test"].drop_duplicates(["region", "model"]).set_index(["region", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    order = list(NAMES)
    for ax, r in zip(axes, ("MISO", "ERCOT")):
        v = met.loc[r].reindex(order)["pr_auc"]
        lo, hi = sig.loc[r].reindex(order)["pr_auc_lo"], sig.loc[r].reindex(order)["pr_auc_hi"]
        colors = [ORANGE if n == "logreg-no_rel" else BLUE for n in order]
        ypos = np.arange(len(order))[::-1]
        ax.barh(ypos, v, color=colors, height=0.62)
        ax.errorbar(v, ypos, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        base = met.loc[r]["base_rate"].iloc[0]
        ax.axvline(base, color=MUTED, ls=":", lw=1.2)
        ax.text(base, -0.9, f" random guess = {base:.1%}", fontsize=8, color=MUTED)
        for yp, val in zip(ypos, v):
            ax.text(val + 0.005, yp + 0.18, f"{val:.2f}", fontsize=8, color=INK)
        ax.set_yticks(np.arange(len(order))[::-1], [NAMES[n] for n in order])
        ev = int(met.loc[(r, "rule"), "n_events"])
        ax.set_title(f"{r} test year ({ev} spike events)")
        ax.set_xlabel("PR-AUC (higher is better), with 95% CI")
        ax.set_ylim(-1.2, len(order) - 0.4)
    fig.suptitle("Logistic regression wins in MISO; ERCOT has too few events for clear winners (wide CIs)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "08_model_comparison_test.png")


def fig_ablation(cfg):
    """09: MISO test PR-AUC when one feature group is removed."""
    rep = cfg["paths"]["reports"]
    met = pd.read_csv(rep / "metrics_y_price_ablation.csv").query("view == 'test' and region == 'MISO'")
    sig = pd.read_csv(rep / "significance_y_price_ablation.csv").query("view == 'test' and region == 'MISO'")
    lab = {"logreg-no_rel": "all features", "logreg-no_rel-no_da_price_target": "− tomorrow's day-ahead price",
           "logreg-no_rel-no_temp-no_cdd-no_hdd": "− weather forecast", "logreg-no_rel-no_load_fcst": "− load forecast",
           "logreg-no_rel-no_rt_price-no_da_price": "− all price features",
           "logreg-no_rel-no_hour-no_dow-no_month-no_is-no_doy": "− calendar"}
    df = met.set_index("model").join(sig.set_index("model")[["pr_auc_lo", "pr_auc_hi", "p_value"]]).loc[list(lab)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ypos = np.arange(len(df))[::-1]
    colors = [ORANGE if (p < 0.05) else BLUE for p in df["p_value"].fillna(1)]
    colors[0] = GRAY
    ax.barh(ypos, df["pr_auc"], color=colors, height=0.6)
    ax.errorbar(df["pr_auc"], ypos, xerr=[df["pr_auc"] - df["pr_auc_lo"], df["pr_auc_hi"] - df["pr_auc"]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    for yp, (v, p) in zip(ypos, zip(df["pr_auc"], df["p_value"])):
        pt = "" if np.isnan(p) else ("  (p<0.001)" if p < 0.001 else f"  (p={p:.2f})")
        ax.text(v + 0.006, yp + 0.15, f"{v:.2f}{pt}", fontsize=8, color=INK)
    ax.set_yticks(ypos, [lab[m] for m in df.index])
    ax.set_xlabel("MISO test PR-AUC, logistic regression (95% CI)")
    ax.set_title("Removing tomorrow's day-ahead price halves the score; weather and load add nothing on top\n"
                 "(orange = significant drop vs all features)", loc="left", fontsize=10)
    save(fig, "09_feature_ablation_miso.png")


EVENTS = {  # (date, label) of the news behind the biggest episodes; sources in docs/trends_and_news.md
    "ERCOT": [("2022-07-13", "Jul 2022 heat:\nconservation appeals"), ("2023-08-20", "Aug–Sep 2023 heat,\nSep 6 EEA2"),
              ("2025-01-01", "battery fleet ~doubles\nin 2025 (7.8→13.9 GW)"), ("2026-01-25", "Winter Storm\nFern")],
    "MISO": [("2025-01-21", "Jan 2025\narctic blast"), ("2025-06-23", "Jun 2025 heat:\nMax Gen event"),
             ("2026-01-24", "Winter Storm Fern:\nMax Gen emergency"), ("2026-09-02", "Sep 2026 heat dome:\nEEA2, record peak")],
}


def fig_news_timeline(data, cfg):
    """10: monthly spike hours with the real-world events behind the biggest episodes."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, (r, d) in zip(axes, data.items()):
        y = d["y"]["y_price"].dropna().resample("MS").sum()
        shade_splits(ax, cfg)
        ax.bar(y.index, y.values, width=24, color=BLUE, align="edge")
        top = y.max()
        for i, (day, lab) in enumerate(EVENTS[r]):
            t = pd.Timestamp(day, tz="UTC")
            ax.axvline(t, color=ORANGE, lw=1, ls="--")
            ax.text(t, top * (0.95 - 0.28 * (i % 2)), " " + lab, fontsize=7.5, color=INK, va="top")
        ax.set_ylim(0, top * 1.05)
        ax.set_ylabel("spike hours / month")
        ax.set_title(f"{r}: spike hours per month and the news behind them")
    fig.suptitle("Spike clusters line up with named heat waves and winter storms; ERCOT's quiet 2024–25 follows its "
                 "battery boom", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "10_spikes_and_news.png")


def fig_load_vs_spikes(data):
    """11: yearly average load vs yearly spike hours: in ERCOT load keeps rising while spikes collapse."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), sharex="col")
    for j, (r, d) in enumerate(data.items()):
        df, y = d["df"], d["y"]["y_price"]
        load = df["load_mw"].groupby(df.index.year).mean() / 1000
        spikes = y.groupby(y.index.year).sum()
        for i, (s, lab, c) in enumerate(((load, "average load (GW)", GRAY), (spikes, "spike hours", BLUE))):
            ax = axes[i, j]
            ax.bar(s.index, s.values, color=c, width=0.65)
            for x, v in s.items():
                ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_ylabel(lab)
            ax.set_ylim(0, s.max() * 1.18)
            if i == 0:
                ax.set_title(r)
        axes[1, j].set_xticks(load.index)
    axes[1, 0].text(2025.5, 120, "2021 = Apr–Dec, 2026 = Jan–Oct", fontsize=7.5, color=MUTED, ha="right")
    fig.suptitle("ERCOT: load keeps rising (+17% since 2023) yet spike hours collapse. MISO: load +7%, spike hours up 3–4×",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "11_load_vs_spike_hours_by_year.png")


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {r: prepare(r, cfg) for r in cfg["regions"]}
    fig_prices(data, cfg)
    fig_spikes_per_month(data, cfg)
    fig_when(data, cfg)
    fig_da_vs_rt(data, cfg)
    fig_weather_load(data, cfg)
    fig_timeline()
    fig_example_week(cfg)
    fig_models(cfg)
    fig_ablation(cfg)
    fig_news_timeline(data, cfg)
    fig_load_vs_spikes(data)


if __name__ == "__main__":
    main()
