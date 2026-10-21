"""
Pitch-level ABS challenges for every 2026 regular-season game, from Savant's
per-game feed (baseballsavant.mlb.com/gf?game_pk=...). Each pitch in the feed
carries an `abs_challenge` record when it was challenged.

Run after fetch_savant_direct.py (it reads the 2026 game_pks from the daily cache):
    python src/fetch_challenges.py
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

RAW_DAILY = Path("data/raw_daily")
OUT_DIR = Path("data/raw_challenges")
OUT_DIR.mkdir(parents=True, exist_ok=True)
URL = "https://baseballsavant.mlb.com/gf?game_pk={}"

KEEP = ["game_pk", "play_id", "inning", "half_inning", "ab_number", "pitch_number",
        "batter", "pitcher", "catcher", "team_batting", "team_fielding",
        "pre_balls", "pre_strikes", "outs", "call", "plate_x", "plate_z", "sz_top", "sz_bot"]


def game_pks_2026() -> list[int]:
    pks = set()
    for p in sorted(RAW_DAILY.glob("2026-*.pkl.gz")):
        d = pd.read_pickle(p)
        if len(d):
            pks.update(d.loc[d["game_type"] == "R", "game_pk"].astype(int).unique())
    return sorted(pks)


def fetch_game(pk: int) -> str:
    out = OUT_DIR / f"{pk}.json"
    if out.exists():
        return "cached"
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(pk), headers={"User-Agent": "Mozilla/5.0"})
            g = json.loads(urllib.request.urlopen(req, timeout=120).read())
            rows = []
            for side in ("team_home", "team_away"):
                for p in g.get(side, []) or []:
                    ch = p.get("abs_challenge")
                    if not ch:
                        continue
                    r = {k: p.get(k) for k in KEEP}
                    r.update({f"chal_{k}": v for k, v in ch.items()})
                    rows.append(r)
            out.write_text(json.dumps({"hasAbs": g.get("hasAbs"), "challenges": rows}))
            return f"{len(rows)}"
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (attempt + 1))
    return f"FAILED {err}"


if __name__ == "__main__":
    pks = game_pks_2026()
    print(f"{len(pks)} games", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for pk, s in zip(pks, ex.map(fetch_game, pks)):
            if s.startswith("FAILED"):
                print(pk, s, flush=True)

    rows = []
    for p in OUT_DIR.glob("*.json"):
        rows += json.loads(p.read_text())["challenges"]
    ch = pd.DataFrame(rows).drop_duplicates("play_id")
    ch["game_pk"] = ch["game_pk"].astype(int)
    ch.to_pickle("data/challenges_2026.pkl.gz")
    print("challenges:", len(ch))
    print(ch["chal_challenging_player_type"].value_counts().to_dict())
    print("overturn rate:", round(ch["chal_is_overturned"].mean(), 3))
