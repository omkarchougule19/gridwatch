"""Pull raw data from the preferred (citable) sources and build data/processed/{region}.parquet.

Sources (see docs/sources.md):
  load + BA day-ahead load forecast  EIA-930 Hourly Electric Grid Monitor        via gridstatus.EIA (needs EIA_API_KEY)
  ERCOT prices                       ERCOT MIS: RTM / DAM settlement point prices via gridstatus.Ercot (2011+)
  MISO prices                        MISO Market Reports: RT final / DA ex-post   via gridstatus.MISO  (daily files ~2023+)
  weather forecasts AS ISSUED        NOAA GFS day-ahead run (archive: Open-Meteo Previous Runs API, ~2021-04+)
  operator alerts                    data/external/alerts.csv, hand-curated from ERCOT/MISO notices
Each pull is cached in data/interim/, so re-running only downloads what's missing.

Usage: python -m gridwatch.data.ingest --region ERCOT --start 2021-04-01 --end 2026-10-31
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

from gridwatch.config import load_config

EIA_RESPONDENT = {"ERCOT": "ERCO", "MISO": "MISO"}
OPEN_METEO = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _cached(path: Path, fn):
    if path.exists():
        return pd.read_parquet(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _hourly(df: pd.DataFrame, value_col: str, name: str) -> pd.Series:
    s = df.set_index(pd.to_datetime(df["Interval Start"], utc=True))[value_col]
    return s.resample("h").mean().rename(name)


def eia_load(region, start, end):
    import gridstatus

    df = gridstatus.EIA(api_key=os.environ["EIA_API_KEY"]).get_dataset(
        "electricity/rto/region-data", start, end, facets={"respondent": [EIA_RESPONDENT[region]]})
    return pd.concat([_hourly(df, "Load", "load_mw"), _hourly(df, "Load Forecast", "load_fcst_mw")], axis=1)


def prices(region, start, end, hub):
    import gridstatus

    if region == "ERCOT":
        iso, years = gridstatus.Ercot(), range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
        rt = pd.concat([iso.get_rtm_spp(y) for y in years])
        da = pd.concat([iso.get_dam_spp(y) for y in years])
        rt, da, col = rt[rt.Location == hub], da[da.Location == hub], "SPP"
    else:
        iso = gridstatus.MISO()
        rt = iso.get_lmp_real_time_hourly_final(start, end=end)
        da = iso.get_lmp_day_ahead_hourly(start, end=end)
        rt, da, col = rt[rt.Location == hub], da[da.Location == hub], "LMP"
    if rt.empty:
        raise ValueError(f"no rows for hub {hub!r}")
    return pd.concat([_hourly(rt, col, "rt_price"), _hourly(da, col, "da_price")], axis=1)


def weather_forecasts(cities: dict, start, end, model):
    """The GFS forecast issued the day before each hour (`*_previous_day1`), averaged over the region's load centers.

    Because these are forecasts, not observations, no weather leaks from the future (TA feedback).
    """
    lats, lons = zip(*cities.values())
    r = requests.get(OPEN_METEO, params={
        "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
        "start_date": start, "end_date": end, "models": model, "temperature_unit": "fahrenheit",
        "hourly": "temperature_2m_previous_day1", "timezone": "UTC"}, timeout=120)
    r.raise_for_status()
    out = r.json()
    out = out if isinstance(out, list) else [out]
    temps = pd.concat([pd.Series(o["hourly"]["temperature_2m_previous_day1"],
                                 index=pd.to_datetime(o["hourly"]["time"], utc=True)) for o in out], axis=1)
    return pd.DataFrame({"temp_f_fcst": temps.mean(axis=1), "temp_f_fcst_max": temps.max(axis=1),
                         "temp_f_fcst_min": temps.min(axis=1)})


def alerts(region, index, path):
    """Hourly 0/1 from the curated alert windows (columns: region,start_utc,end_utc,kind,source)."""
    y = pd.Series(0, index=index, name="alert")
    if path.exists():
        a = pd.read_csv(path, comment="#", parse_dates=["start_utc", "end_utc"])
        for r in a[a.region == region].itertuples():
            y[(index >= r.start_utc.tz_localize("UTC")) & (index <= r.end_utc.tz_localize("UTC"))] = 1
    return y


def build_processed(region: str, start: str, end: str, cfg: dict) -> pd.DataFrame:
    src, interim = cfg["sources"][region], cfg["paths"]["interim"]
    tag = f"{region}_{start}_{end}"
    df = pd.concat([
        _cached(interim / f"load_{tag}.parquet", lambda: eia_load(region, start, end)),
        _cached(interim / f"prices_{tag}.parquet", lambda: prices(region, start, end, src["hub"])),
        _cached(interim / f"weather_{tag}.parquet",
                lambda: weather_forecasts(src["cities"], start, end, cfg["sources"]["weather_model"])),
    ], axis=1).loc[start:end]
    df["alert"] = alerts(region, df.index, cfg["paths"]["external"] / "alerts.csv")
    df.index.name = "ts"
    out = cfg["paths"]["processed"] / f"{region}.parquet"
    df.to_parquet(out)
    print(f"{out}: {len(df)} rows, missing share:\n{df.isna().mean().round(3).to_string()}")
    return df


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--config")
    a = ap.parse_args()
    build_processed(a.region, a.start, a.end, load_config(a.config))
