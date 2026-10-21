"""
Win-based team ranking for Part 2 (replaces the run-based team scorecard in analysis.py).

For every challenge opportunity (from part2_extended.py):
    realized      = value of what the player actually did (win stake if a won challenge,
                    minus the value of the challenge lost if a lost one, 0 if no challenge)
    expected_avg  = what an average player of the same role would have earned on that pitch
Team value above average = sum(realized - expected_avg) over the team's batters and catchers.
Room to improve = simulated wins under the best policy minus the actual policy (team_sim_gap.csv).

    python src/team_ranking.py  ->  results/part2/team_ranking.csv, figures/fig10_team_ranking.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT, FIG = Path("results/part2"), Path("figures")
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})

opp = pd.read_pickle("data/opportunities_graded.pkl.gz")
teams = pd.read_pickle("data/called.pkl.gz")[["game_pk", "at_bat_number", "pitch_number", "bat_team", "fld_team"]]
opp = opp.merge(teams, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
opp["team"] = np.where(opp.role == "batter", opp.bat_team, opp.fld_team)
opp["above_avg"] = opp["realized"] - opp["expected_avg"]

rng = np.random.default_rng(3)
rows = []
for team, g in opp.groupby("team"):
    v = g["above_avg"].values
    # bootstrap by game: a team's opportunities within a game share a pitcher, an umpire, a score
    by_game = g.groupby("game_pk")["above_avg"].sum().values
    boots = by_game[rng.integers(0, len(by_game), size=(2000, len(by_game)))].sum(1)
    chal = g[g.challenged == 1]
    rows.append({"team": team, "wins_above_avg": v.sum(), "ci_low": np.percentile(boots, 5),
                 "ci_high": np.percentile(boots, 95), "challenges": len(chal),
                 "win_rate": chal.chal_is_overturned.mean(),
                 "wins_from_batters": g.loc[g.role == "batter", "above_avg"].sum(),
                 "wins_from_catchers": g.loc[g.role == "catcher", "above_avg"].sum()})
rank = pd.DataFrame(rows).set_index("team")
# center on the league: players beat the role model slightly overall, so "above average" = above
# the average TEAM (subtracting a constant leaves the order and interval widths unchanged)
shift = rank["wins_above_avg"].mean()
for c in ("wins_above_avg", "ci_low", "ci_high"):
    rank[c] -= shift
print("league mean shift:", round(shift, 3))
gap = pd.read_csv(OUT / "team_sim_gap.csv", index_col="team")
rank = rank.join(gap[["wins_gap"]].rename(columns={"wins_gap": "room_to_improve_wins"}))
rank = rank.sort_values("wins_above_avg", ascending=False).round(3)
rank.to_csv(OUT / "team_ranking.csv")
print(rank.to_string())
print("corr(value, win rate):", round(rank.wins_above_avg.corr(rank.win_rate), 3))
print("corr(value, challenges):", round(rank.wins_above_avg.corr(rank.challenges), 3))
print("corr(value, room to improve):", round(rank.wins_above_avg.corr(rank.room_to_improve_wins), 3))
print("teams separable at 90%:", int(((rank.ci_low > 0) | (rank.ci_high < 0)).sum()), "of", len(rank))

# Fig 10 - team ranking with 90% intervals
t = rank.sort_values("wins_above_avg")
fig, ax = plt.subplots(figsize=(7.5, 7.8))
y = np.arange(len(t))
cols = [C1 if v >= 0 else C2 for v in t.wins_above_avg]
ax.hlines(y, t.ci_low, t.ci_high, color=cols, lw=1.6, alpha=0.45)
ax.scatter(t.wins_above_avg, y, color=cols, s=40, zorder=3)
for i, v in enumerate(t.wins_above_avg):
    ax.text(t.ci_high.iloc[i] + 0.05, i, f"{round(v, 2) + 0.0:+.2f}", va="center", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y, t.index)
ax.set_xlim(t.ci_low.min() - 0.1, t.ci_high.max() + 0.35)
ax.set_xlabel("Wins from challenge decisions above an average team on the same pitches, 2026\n(dots = estimate; bars = 90% interval)")
ax.set_title(f"{t.index[-1]} made the most of its challenges; {t.index[0]} the least")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "07_team_ranking.png", dpi=200, bbox_inches="tight")
