"""Live download progress. Usage: python progress.py CHUNK_DIR CHUNK_LIST"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

os.system("")  # enable ANSI escape codes on Windows
C, LIST = Path(sys.argv[1]), sys.argv[2]
chunks = [l.split() for l in open(LIST) if l.strip()]
t0, rows_cache, first_done = time.time(), {}, None


def bar(n, total, w=36):
    f = int(w * n / total) if total else 0
    return f"[{'#' * f}{'.' * (w - f)}] {n:>5}/{total:<5} {100 * n / total if total else 0:5.1f}%"


while True:
    stats = {r: {"chunks": 0, "n": 0, "files": 0, "files_done": 0, "rows": 0} for r in ("ERCOT", "MISO")}
    active, last = [], None
    for reg, s, e in chunks:
        key = f"{reg}_{s}_{e}"
        out, log = C / f"{key}.parquet", C / "_logs" / f"{key}.log"
        days = (pd.Timestamp(e) - pd.Timestamp(s)).days
        st = stats[reg]
        st["chunks"] += 1
        st["files"] += 2 * days if reg == "MISO" else 2  # MISO: RT + DA file per day; ERCOT: RTM + DAM yearly archive
        if out.exists():
            st["n"] += 1
            st["files_done"] += 2 * days if reg == "MISO" else 2
            if key not in rows_cache:
                rows_cache[key] = pq.ParquetFile(out).metadata.num_rows
            st["rows"] += rows_cache[key]
            if last is None or out.stat().st_mtime > last[1]:
                last = (key, out.stat().st_mtime)
        elif log.exists():
            n = sum(1 for line in open(log, errors="ignore") if "Downloading" in line)
            st["files_done"] += min(n, 2 * days) if reg == "MISO" else 0
            st["rows"] += (min(n, 2 * days) // 2) * 24 if reg == "MISO" else 0  # ~24 hourly hub records per day
            active.append(f"{key}  ({n} files)" if reg == "MISO" else f"{key}  (parsing yearly archive)")
    el = time.time() - t0
    m, er = stats["MISO"], stats["ERCOT"]
    done_chunks = m["n"] + er["n"]
    if first_done is None:
        first_done = m["files_done"]  # measure speed only from when the viewer opened
    rate = (m["files_done"] - first_done) / el if el > 5 else 0
    eta = (m["files"] - m["files_done"]) / rate if rate else float("nan")
    lines = [
        f"GridWatch price download  {time.strftime('%H:%M:%S')}   (refreshes every second, close this window anytime)",
        "",
        f"ERCOT chunks (years)   {bar(er['n'], er['chunks'])}",
        f"MISO  chunks (months)  {bar(m['n'], m['chunks'])}",
        f"MISO  daily files      {bar(m['files_done'], m['files'])}",
        f"ALL   chunks           {bar(done_chunks, er['chunks'] + m['chunks'])}",
        "",
        f"hourly price records saved/in flight:  ERCOT {er['rows']:,}   MISO {m['rows']:,}   total {er['rows'] + m['rows']:,}",
        f"MISO speed: {rate:.1f} files/s   ETA (MISO): {'--' if eta != eta else f'{eta / 60:.1f} min'}",
        f"last finished: {last[0] if last else '-'}",
        "",
        f"active ({len(active)}):  * chunks started before the live log was added show 0 files until they finish",
        *[f"   {a}" for a in active[:16]],
    ]
    print("\x1b[H\x1b[J" + "\n".join(lines), flush=True)
    if done_chunks == len(chunks):
        print("\nAll chunks downloaded.")
        break
    time.sleep(1)
