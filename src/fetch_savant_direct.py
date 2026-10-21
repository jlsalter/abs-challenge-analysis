"""
Pull regular-season Statcast pitches straight from Baseball Savant's CSV endpoint,
one day per request (Savant caps a single search at ~25k rows).
Cached per day in data/raw/, so reruns resume.

    python src/fetch_savant_direct.py
"""

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import urllib.request

RAW = Path("data/raw_daily")
RAW.mkdir(parents=True, exist_ok=True)

SEASONS = {2025: (date(2025, 3, 18), date(2025, 9, 28)),
           2026: (date(2026, 3, 25), date(2026, 9, 27))}

URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
       "&player_type=pitcher&hfGT=R%7C&game_date_gt={d}&game_date_lt={d}")


def fetch_day(d: date) -> str:
    out = RAW / f"{d}.pkl.gz"
    if out.exists():
        return "cached"
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(d=d), headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=180).read().decode("utf-8-sig")
            if not raw.strip():
                pd.DataFrame().to_pickle(out)   # off day
                return "empty"
            df = pd.read_csv(io.StringIO(raw), low_memory=False)
            df.to_pickle(out)
            return f"{len(df)} rows"
        except Exception as e:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
            err = e
    return f"FAILED: {err}"


if __name__ == "__main__":
    days = []
    for start, end in SEASONS.values():
        d = start
        while d <= end:
            days.append(d)
            d += timedelta(days=1)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for d, status in zip(days, ex.map(fetch_day, days)):
            print(d, status, flush=True)

    frames = [pd.read_pickle(p) for p in sorted(RAW.glob("*.pkl.gz"))]
    full = pd.concat([f for f in frames if len(f)], ignore_index=True)
    full = full[full["game_type"] == "R"]
    full.to_pickle("data/statcast_2025_2026.pkl.gz")
    print("TOTAL", full.groupby("game_year").size().to_dict())
    sys.exit(0)
