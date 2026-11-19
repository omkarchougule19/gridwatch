"""Build data/external/test_events.csv: curated test windows chosen on forecast WEATHER only (never on prices).

Rule (decided before looking at model output):
  - per region, the 10 hottest (max forecast temp) and 10 coldest (min forecast temp) days of the test period;
  - days within 3 days of each other merge into one window;
  - a window overlapping a documented grid event (NAMED below) is tagged *_named, with the source;
  - each window gets a matched_normal control: same length, within 4 weeks, forecast temp closest to that month's median.
Usage: python scripts/curate_windows.py   (team then reviews each row: checked_by)
"""
from __future__ import annotations

import pandas as pd

from _common import prepare
from gridwatch.config import load_config

TZ = "America/Chicago"
NAMED = {  # region: [(first day, last day, description + source)]
    "ERCOT": [("2026-01-24", "2026-01-27", "Winter Storm Fern; https://www.ercot.com/files/docs/2026/01/28/ERCOT-Post-Event-Report-Winter-Storm-Fern.pdf")],
    "MISO": [("2026-01-23", "2026-01-27", "Winter Storm Fern, Max Gen emergency; https://cdn.misoenergy.org/20260217%20RSC%20Item%2005%20Winter%20Storm%20Fern%20Report741721.pdf"),
             ("2026-09-01", "2026-09-03", "Sept 2026 heat dome, EEA2, record Sept peak; https://www.rtoinsider.com/142336-miso-enters-emergency-in-early-sept-heat/")],
}
HEADER = "# " + __doc__.strip().replace("\n", "\n# ") + "\n"


def windows(days: pd.DatetimeIndex, gap: int = 3) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    days = days.sort_values()
    out, start, prev = [], days[0], days[0]
    for d in days[1:]:
        if (d - prev).days > gap:
            out.append((start, prev))
            start = d
        prev = d
    return out + [(start, prev)]


def main():
    cfg = load_config()
    rows = []
    for region in cfg["regions"]:
        d = prepare(region, cfg)
        df = d["df"][d["masks"]["test"]]
        day = df.index.tz_convert(TZ).normalize().tz_localize(None)
        tmax, tmin = df["temp_f_fcst_max"].groupby(day).max(), df["temp_f_fcst_min"].groupby(day).min()
        tmean = df["temp_f_fcst"].groupby(day).mean()
        picks = []
        for kind, s in (("heat", tmax.nlargest(10)), ("cold", tmin.nsmallest(10))):
            picks += [(kind, a, b) for a, b in windows(s.index)]
        taken = {x for _, a, b in picks for x in pd.date_range(a, b)}
        for kind, a, b in sorted(picks, key=lambda t: t[1]):
            src = "GFS day-ahead forecast: among the 10 hottest/coldest test days"
            for na, nb, desc in NAMED.get(region, []):
                if a <= pd.Timestamp(nb) and b >= pd.Timestamp(na):
                    kind, src = kind + "_named", desc
            rows.append((region, f"{a.date()} 00:00", f"{b.date()} 23:00", kind, src))
            n = (b - a).days + 1
            med = tmean[tmean.index.month == a.month].median()
            best = None
            for off in [o for o in range(-28, 29) if o <= -n or o >= n]:
                blk = pd.date_range(a + pd.Timedelta(days=off), periods=n)
                if taken & set(blk) or not blk.isin(tmean.index).all():
                    continue
                err = (tmean.loc[blk] - med).abs().mean()
                if best is None or err < best[0]:
                    best = (err, blk)
            taken |= set(best[1])
            rows.append((region, f"{best[1][0].date()} 00:00", f"{best[1][-1].date()} 23:00", "matched_normal",
                         f"control for {a.date()}: forecast temp within {best[0]:.1f} F of the month's median"))
    out = pd.DataFrame(rows, columns=["region", "start_local", "end_local", "kind", "source"])
    out["checked_by"] = "TODO: two reviewers"
    path = cfg["paths"]["external"] / "test_events.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER + out.to_csv(index=False))
    print(out[["region", "start_local", "end_local", "kind"]].to_string(index=False))


if __name__ == "__main__":
    main()
