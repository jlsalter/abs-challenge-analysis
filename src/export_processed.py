"""Write the analysis-ready datasets in data/processed/ (see data/README.md)."""
from pathlib import Path

import pandas as pd

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

called = pd.read_pickle("data/called.pkl.gz")
cols = ["game_pk", "game_date", "game_year", "inning", "inning_topbot", "at_bat_number", "pitch_number",
        "batter", "pitcher", "fielder_2", "stand", "p_throws", "pitch_type", "release_speed",
        "balls", "strikes", "outs", "base_state", "bat_score", "fld_score", "bat_team", "fld_team",
        "px", "pz", "abs_top", "abs_bot", "edge_dist_in", "abs_strike", "ump_strike", "final_strike",
        "challenged", "chal_is_overturned", "chal_by_batter", "runs_at_stake", "bat_chal_left", "fld_chal_left"]
c = called[[x for x in cols if x in called.columns]].rename(columns={
    "fielder_2": "catcher", "px": "plate_x_mid_ft", "pz": "plate_z_mid_ft", "abs_top": "abs_zone_top_ft",
    "abs_bot": "abs_zone_bot_ft", "edge_dist_in": "edge_distance_in", "chal_is_overturned": "overturned",
    "chal_by_batter": "challenged_by_batter"})
for f in ["plate_x_mid_ft", "plate_z_mid_ft", "abs_zone_top_ft", "abs_zone_bot_ft"]:
    c[f] = c[f].round(4)
c["edge_distance_in"] = c["edge_distance_in"].round(3)
c["runs_at_stake"] = c["runs_at_stake"].round(4)
c.to_csv(OUT / "called_pitches_2025_2026.csv.gz", index=False)
c[c.challenged == 1].drop(columns=["challenged"]).to_csv(OUT / "challenges_2026.csv.gz", index=False)

opp = pd.read_pickle("data/opportunities_graded.pkl.gz")
for k in opp.select_dtypes("float").columns:
    opp[k] = opp[k].round(5)
opp.rename(columns={
    "d_fav": "distance_in_challengers_favor_in", "ov_w": "option_value_optimal",
    "ov_act": "option_value_actual_policy", "p_star": "breakeven_confidence", "tau_act": "threshold_actual_in",
    "tau_opt": "threshold_optimal_in", "q_act": "p_challenge_avg_player", "q_opt": "p_challenge_optimal",
    "chal_is_overturned": "overturned", "realized": "value_realized_wins",
    "expected_avg": "value_expected_avg_player_wins"}).to_csv(OUT / "challenge_opportunities_2026.csv.gz", index=False)

pd.read_csv("data/run_expectancy_2026.csv").round(4).to_csv(OUT / "run_expectancy_2026.csv", index=False)
pd.read_csv("data/player_names.csv")[["id", "name", "role"]].to_csv(OUT / "player_names.csv", index=False)
print("wrote", sorted(p.name for p in OUT.iterdir()))
