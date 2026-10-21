"""
Build the analysis dataset:
  1. Put 2025 and 2026 pitch locations on the same measurement (middle of the plate).
  2. Give every batter one fixed ABS zone (from 2026) and apply it to both seasons.
  3. Recover the umpire's ORIGINAL call in 2026 (Statcast records the post-challenge call).
  4. Add game state, run expectancy, runs at stake, and each team's challenges remaining.

    python src/build_dataset.py   ->  data/pitches.pkl.gz, data/called.pkl.gz
"""

import numpy as np
import pandas as pd

Y_FRONT = 17 / 12          # front of plate (pre-2026 Statcast reference), ft
Y_MID = 17 / 24            # middle of plate (ABS / 2026 reference), ft
HALF_PLATE = 17 / 24       # 8.5 in, ft
BALL_R = 1.45 / 12         # baseball radius, ft
CALLED = {"called_strike", "ball", "blocked_ball"}


# ---------------------------------------------------------------------------
# 1. Location at the middle of the plate
# ---------------------------------------------------------------------------
def _time_at_y(d, y):
    """Time (s) after the y=50 ft reference until the ball reaches distance y."""
    a, b, c = 0.5 * d["ay"], d["vy0"], 50.0 - y
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


def shift_location(d, y_from, y_to):
    """Move (plate_x, plate_z) from depth y_from to depth y_to along the fitted trajectory."""
    t1, t2 = _time_at_y(d, y_from), _time_at_y(d, y_to)
    dx = d["vx0"] * (t2 - t1) + 0.5 * d["ax"] * (t2 ** 2 - t1 ** 2)
    dz = d["vz0"] * (t2 - t1) + 0.5 * d["az"] * (t2 ** 2 - t1 ** 2)
    return d["plate_x"] + dx, d["plate_z"] + dz


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def walk_bases(state):
    on1, on2, on3 = state & 1, (state >> 1) & 1, (state >> 2) & 1
    runs = 0
    if on1:
        if on2:
            if on3:
                runs = 1
            on3 = 1
        on2 = 1
    return 1 + 2 * on2 + 4 * on3, runs


def main():
    df = pd.read_pickle("data/statcast_2025_2026.pkl.gz")
    df = df[df["game_type"] == "R"].copy()
    df["game_year"] = df["game_year"].astype(int)
    df = df.dropna(subset=["plate_x", "plate_z", "vy0", "ay"])

    # --- 1. common location -------------------------------------------------
    is25 = df["game_year"] == 2025
    x_mid, z_mid = shift_location(df[is25], Y_FRONT, Y_MID)
    df["px"] = df["plate_x"]
    df["pz"] = df["plate_z"]
    df.loc[is25, "px"] = x_mid
    df.loc[is25, "pz"] = z_mid

    # --- 2. one ABS zone per batter -----------------------------------------
    abs_zone = (df[~is25].groupby("batter")[["sz_top", "sz_bot"]].median()
                .rename(columns={"sz_top": "abs_top", "sz_bot": "abs_bot"}))
    df = df.merge(abs_zone, left_on="batter", right_index=True, how="left")

    dx = df["px"].abs() - (HALF_PLATE + BALL_R)
    dz = np.maximum((df["abs_bot"] - BALL_R) - df["pz"], df["pz"] - (df["abs_top"] + BALL_R))
    outside = np.sqrt(np.clip(dx, 0, None) ** 2 + np.clip(dz, 0, None) ** 2)
    df["edge_dist_in"] = 12 * np.where((dx > 0) | (dz > 0), outside, np.maximum(dx, dz))
    df["abs_strike"] = (df["edge_dist_in"] <= 0).astype(int)
    df["z_norm"] = (df["pz"] - df["abs_bot"]) / (df["abs_top"] - df["abs_bot"])

    # --- 3. original umpire call --------------------------------------------
    ch = pd.read_pickle("data/challenges_2026.pkl.gz")
    ch = ch.rename(columns={"ab_number": "at_bat_number"})
    ch = ch[["game_pk", "at_bat_number", "pitch_number", "chal_is_overturned",
             "chal_challenging_player_type", "chal_challenge_team_id", "chal_edge_distance",
             "team_batting", "team_fielding"]]
    ch["challenged"] = 1
    df = df.merge(ch, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
    df["challenged"] = df["challenged"].fillna(0).astype(int)
    df["chal_is_overturned"] = df["chal_is_overturned"].fillna(False).astype(bool)
    df["chal_by_batter"] = (df["chal_challenging_player_type"] == "batter").astype(int)

    final_strike = (df["description"] == "called_strike").astype(int)
    df["is_called"] = df["description"].isin(CALLED)
    df["ump_strike"] = np.where(df["chal_is_overturned"], 1 - final_strike, final_strike)
    df["final_strike"] = final_strike

    # --- 4. game state ------------------------------------------------------
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    df["base_state"] = (df["on_1b"].notna().astype(int) + 2 * df["on_2b"].notna().astype(int)
                        + 4 * df["on_3b"].notna().astype(int))
    df["outs"] = df["outs_when_up"].astype(int)
    df["bat_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])
    df["fld_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    half = df["game_pk"].astype(str) + "_" + df["inning"].astype(str) + df["inning_topbot"]
    df["runs_rest"] = df.groupby(half)["post_bat_score"].transform("max") - df["bat_score"]

    # run expectancy by (outs, bases, balls, strikes), 2026, innings 1-8
    re_src = df[(df["game_year"] == 2026) & (df["inning"] <= 8) & (df["balls"] <= 3) & (df["strikes"] <= 2)]
    re = re_src.groupby(["outs", "base_state", "balls", "strikes"])["runs_rest"].mean()
    re.to_frame("re").reset_index().to_csv("data/run_expectancy_2026.csv", index=False)
    re_ob = re_src.groupby(["outs", "base_state"])["runs_rest"].mean().to_dict()
    re = re.to_dict()

    def lookup(o, b, bl, st):
        return re.get((o, b, bl, st), re_ob.get((o, b), 0.0))   # fallback only for unseen states

    def value(o, b, bl, st):
        if st >= 3:
            o += 1
            return 0.0 if o >= 3 else lookup(o, b, 0, 0)
        if bl >= 4:
            nb, r = walk_bases(b)
            return r + lookup(o, nb, 0, 0)
        return lookup(o, b, bl, st)

    called = df[df["is_called"] & (df["balls"] <= 3) & (df["strikes"] <= 2)].copy()
    keys = zip(called["outs"], called["base_state"], called["balls"], called["strikes"])
    called["runs_at_stake"] = [value(o, b, bl + 1, st) - value(o, b, bl, st + 1) for o, b, bl, st in keys]

    # --- challenges remaining before each pitch (2026, innings 1-9) ----------
    # 2 per team, kept on success, lost on failure. Extra innings handled separately.
    c26 = df[df["game_year"] == 2026]
    left = {}
    bat_left, fld_left = [], []
    for g, b, f, ch_flag, over, by_bat in zip(c26["game_pk"], c26["bat_team"], c26["fld_team"],
                                              c26["challenged"], c26["chal_is_overturned"],
                                              c26["chal_by_batter"]):
        bat_left.append(left.get((g, b), 2))
        fld_left.append(left.get((g, f), 2))
        if ch_flag and not over:
            team = b if by_bat else f
            left[(g, team)] = left.get((g, team), 2) - 1
    df.loc[c26.index, "bat_chal_left"] = bat_left
    df.loc[c26.index, "fld_chal_left"] = fld_left
    called = called.join(df[["bat_chal_left", "fld_chal_left"]])

    df.to_pickle("data/pitches.pkl.gz")
    called.to_pickle("data/called.pkl.gz")
    print("pitches:", df.groupby("game_year").size().to_dict())
    print("called:", called.groupby("game_year").size().to_dict())
    print("2026 challenges matched:", int(df["challenged"].sum()))


if __name__ == "__main__":
    main()
