"""Join the downloaded price chunks into the cache files ingest.build_processed expects, then build data/processed.
Usage: python scripts/download/assemble.py   (after all chunks exist)"""
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from gridwatch.config import load_config
from gridwatch.data.ingest import build_processed

load_dotenv()
cfg = load_config()
interim = cfg["paths"]["interim"]
END = "2026-10-31"
for region, start in [("ERCOT", "2021-04-01"), ("MISO", "2023-01-01")]:
    files = sorted(Path(interim / "chunks").glob(f"{region}_*.parquet"))
    p = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    p = p[~p.index.duplicated()].loc[start:f"{END} 23:59"]
    p.to_parquet(interim / f"prices_{region}_{start}_{END}.parquet")
    print(f"{region}: {len(files)} chunks -> {len(p)} hours, {p.index.min()} .. {p.index.max()}")
    build_processed(region, start, END, cfg)
