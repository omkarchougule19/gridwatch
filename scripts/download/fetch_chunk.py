"""One price chunk -> parquet, with retries (DNS on this machine fails intermittently). Skips chunks already saved.
Usage: python fetch_chunk.py REGION START END OUTDIR"""
import sys
import time
from pathlib import Path

from gridwatch.config import load_config
from gridwatch.data.ingest import prices

region, start, end, outdir = sys.argv[1:5]
out = Path(outdir) / f"{region}_{start}_{end}.parquet"
if out.exists():
    sys.exit(0)
hub = load_config()["sources"][region]["hub"]
# Per-file progress log for the live viewer (gridstatus logs one "Downloading ..." line per daily file)
import logging

(Path(outdir) / "_logs").mkdir(parents=True, exist_ok=True)
_h = logging.FileHandler(Path(outdir) / "_logs" / f"{region}_{start}_{end}.log")
_h.setLevel(logging.INFO)
logging.getLogger().addHandler(_h)
logging.getLogger().setLevel(logging.INFO)
for attempt in range(6):
    try:
        df = prices(region, start, end, hub)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        print(f"OK {out.name} {len(df)} rows", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"retry {attempt + 1} {out.name}: {type(e).__name__}", flush=True)
        time.sleep(5 * (attempt + 1))
print(f"FAILED {out.name}", flush=True)
sys.exit(1)
