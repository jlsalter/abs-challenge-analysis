"""
All results and figures for the write-up.

    python src/analysis.py   ->  results/results.json, results/*.csv, figures/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
(FIG / "extra").mkdir(parents=True, exist_ok=True)
RES = Path("results"); RES.mkdir(exist_ok=True)
R = {}

# palette (validated reference instance): slot 1 blue, slot 2 orange; diverging blue<->red, gray mid
C26, C25 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGE = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19c9b", "#b42c2c"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


CALLED_COLS = ["game_pk", "batter", "game_year", "at_bat_number", "pitch_number", "inning", "stand", "fielder_2",
               "px", "pz", "abs_top", "abs_bot", "edge_dist_in", "abs_strike", "z_norm",
               "ump_strike", "final_strike", "challenged", "chal_is_overturned", "chal_by_batter",
               "balls", "strikes", "outs", "base_state", "bat_team", "fld_team",
               "runs_at_stake", "bat_chal_left", "fld_chal_left"]
called = pd.read_pickle("data/called.pkl.gz")[CALLED_COLS]
pitches = pd.read_pickle("data/pitches.pkl.gz")[["game_pk", "game_year", "home_team", "away_team", "events"]]
import gc; gc.collect()
called = called[called["abs_top"].notna() & called["edge_dist_in"].notna()].copy()
called["x_away"] = np.where(called["stand"] == "R", called["px"], -called["px"]) * 12   # inches, + = away from batter
R["n_called"] = called.groupby("game_year").size().to_dict()
R["n_batters_with_abs_zone"] = int(called["batter"].nunique())

# ===========================================================================
# PART 1 - Did the called zone change?
# ===========================================================================
called["correct"] = (called["ump_strike"] == called["abs_strike"]).astype(int)
acc = called.groupby("game_year")["correct"].mean()
R["ump_accuracy"] = acc.round(4).to_dict()
near = called[called["edge_dist_in"].abs() <= 2]
R["ump_accuracy_within_2in"] = near.groupby("game_year")["correct"].mean().round(4).to_dict()
R["strike_outside_rate"] = called[called.abs_strike == 0].groupby("game_year")["ump_strike"].mean().round(4).to_dict()
R["ball_inside_rate"] = (1 - called[called.abs_strike == 1].groupby("game_year")["ump_strike"].mean()).round(4).to_dict()
R["final_accuracy_2026_after_challenges"] = round(
    float((called.loc[called.game_year == 2026, "final_strike"] == called.loc[called.game_year == 2026, "abs_strike"]).mean()), 4)

# Fig 1 - accuracy by distance from the zone edge
bins = np.arange(-6, 6.5, 0.5)
called["dist_bin"] = pd.cut(called["edge_dist_in"], bins)
by = called.groupby(["game_year", "dist_bin"], observed=True)["correct"].agg(["mean", "size"]).reset_index()
by["mid"] = by["dist_bin"].apply(lambda b: b.mid)
by.to_csv(RES / "accuracy_by_distance.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4))
for yr, col in [(2025, C25), (2026, C26)]:
    s = by[by.game_year == yr]
    ax.plot(s["mid"], 100 * s["mean"], color=col, lw=2, marker="o", ms=4, label=str(yr))
ax.axvline(0, color=INK2, lw=1, ls="--")
ax.text(0.15, 33, "zone edge", color=INK2, fontsize=9)
ax.text(-5.8, 33, "inside the zone", color=INK2, fontsize=9)
ax.text(3.6, 33, "outside the zone", color=INK2, fontsize=9)
ax.set_xlabel("Distance from the ABS zone edge (inches)")
ax.set_ylabel("Calls agreeing with the ABS zone (%)")
ax.set_ylim(30, 101)
ax.set_title("2026 umpires called fewer edge strikes, inside and outside the zone")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.45))
save(fig, "extra/accuracy_by_distance.png")

# Called-strike probability surfaces (location only), one per year
FEAT = ["x_away", "z_norm"]
H_IN = 12 * float((called["abs_top"] - called["abs_bot"]).median())   # median zone height, inches
R["median_abs_zone_height_in"] = round(H_IN, 2)


def fit_surface(d):
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, random_state=0)
    m.fit(d[FEAT], d["ump_strike"])
    return m


xs = np.linspace(-15, 15, 121)
zs = np.linspace(-0.4, 1.4, 121)
gx, gz = np.meshgrid(xs, zs)
grid = pd.DataFrame({"x_away": gx.ravel(), "z_norm": gz.ravel()})
cell_area = (xs[1] - xs[0]) * (zs[1] - zs[0]) * H_IN       # square inches
win = called[(called.x_away.abs() <= 18) & called.z_norm.between(-0.7, 1.7)]

surf = {}
for yr in (2025, 2026):
    surf[yr] = fit_surface(win[win.game_year == yr]).predict_proba(grid[FEAT])[:, 1]
area = {yr: float((surf[yr] >= 0.5).sum() * cell_area) for yr in surf}
R["called_zone_area_sq_in"] = {k: round(v, 1) for k, v in area.items()}
R["abs_zone_area_sq_in"] = round(float((17 + 2 * 1.45) * (H_IN + 2 * 1.45)), 1)

# bootstrap by game for the area change (subsampled for speed)
rng = np.random.default_rng(7)
games = {yr: win.loc[win.game_year == yr, "game_pk"].unique() for yr in (2025, 2026)}
by_game = {yr: win[win.game_year == yr].groupby("game_pk").indices for yr in (2025, 2026)}
CI_CACHE = RES / "area_bootstrap.json"
deltas = json.load(open(CI_CACHE)) if CI_CACHE.exists() else []
for _ in range(0 if deltas else 25):
    a = {}
    for yr in (2025, 2026):
        g = rng.choice(games[yr], size=len(games[yr]), replace=True)
        idx = np.concatenate([by_game[yr][k] for k in g])
        d = win[win.game_year == yr].iloc[idx]
        d = d.sample(n=min(len(d), 250_000), random_state=int(rng.integers(1e9)))
        a[yr] = (fit_surface(d).predict_proba(grid[FEAT])[:, 1] >= 0.5).sum() * cell_area
    deltas.append(a[2026] - a[2025])
json.dump([float(x) for x in deltas], open(CI_CACHE, "w"))
R["area_change_sq_in_ci95"] = [round(float(v), 1) for v in np.percentile(deltas, [2.5, 50, 97.5])]

# Fig 2 - difference surface
from scipy.ndimage import gaussian_filter
diff = gaussian_filter((surf[2026] - surf[2025]).reshape(gz.shape), 1.5)   # display smoothing only
fig, ax = plt.subplots(figsize=(6, 5.6))
im = ax.pcolormesh(xs, zs * H_IN, diff * 100, cmap=DIVERGE, norm=TwoSlopeNorm(0, -30, 30), shading="auto")
ex = 8.5 + 1.45
ax.plot([-ex, ex, ex, -ex, -ex], [-1.45, -1.45, H_IN + 1.45, H_IN + 1.45, -1.45], color=INK, lw=1.2)
for yr, ls in [(2025, ":"), (2026, "-")]:
    ax.contour(xs, zs * H_IN, surf[yr].reshape(gz.shape), levels=[0.5], colors=[INK2], linewidths=1.2, linestyles=ls)
ax.set_xlabel("Horizontal (in): inside  <-  0  ->  away from batter")
ax.set_ylabel("Height above bottom of ABS zone (in)")
ax.set_title("Where the called zone changed\n(2026 minus 2025 called-strike probability)")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Change in strike probability (pct. points)", color=INK2)
fig.text(0.02, -0.01, "Black box = ABS zone  ·  dotted = 2025 50% line  ·  solid = 2026 50% line",
         fontsize=8, color=INK2)
save(fig, "extra/zone_change_heatmap.png")   # superseded by fig_zone.py

# Framing: spread of catcher strikes-above-expected, umpire call vs final call
base = fit_surface(win)
win = win.assign(p_hat=base.predict_proba(win[FEAT])[:, 1])
RUN_PER_STRIKE = float(called["runs_at_stake"].mean())
fr = []
for yr in (2025, 2026):
    d = win[win.game_year == yr]
    for col in (["ump_strike", "final_strike"] if yr == 2026 else ["ump_strike"]):
        g = d.assign(sae=d[col] - d["p_hat"]).groupby("fielder_2").agg(n=("sae", "size"), sae=("sae", "sum"))
        g = g[g.n >= 2500]
        g["runs"] = g["sae"] * RUN_PER_STRIKE
        g["runs"] -= g["runs"].mean()
        fr.append({"year": yr, "call": col, "n_catchers": len(g),
                   "sd_runs": round(float(g["runs"].std()), 2),
                   "range_runs": [round(float(g["runs"].min()), 1), round(float(g["runs"].max()), 1)]})
R["framing_spread"] = fr

# context: league outcomes
pa = pitches[pitches["events"].notna()]
R["k_rate"] = pa.groupby("game_year")["events"].apply(lambda e: e.isin(["strikeout", "strikeout_double_play"]).mean()).round(4).to_dict()
R["bb_rate"] = pa.groupby("game_year")["events"].apply(lambda e: e.isin(["walk"]).mean()).round(4).to_dict()

# ===========================================================================
# PART 2 - When should you challenge?
# ===========================================================================
c26 = called[called.game_year == 2026].copy()
c26["stake"] = c26["runs_at_stake"].abs()
ch = c26[c26.challenged == 1].copy()
R["n_challenges"] = int(len(ch))
R["overturn_rate"] = round(float(ch["chal_is_overturned"].mean()), 4)
R["overturn_rate_by_role"] = ch.groupby("chal_by_batter")["chal_is_overturned"].mean().round(4).rename({1: "batter", 0: "fielder"}).to_dict()
R["challenges_by_role"] = ch["chal_by_batter"].map({1: "batter", 0: "fielder"}).value_counts().to_dict()

# sanity check: our zone geometry vs the official result of each challenge
ch["we_say_wrong"] = (ch["ump_strike"] != ch["abs_strike"]).astype(int)
R["geometry_agrees_with_official_result"] = round(float((ch["we_say_wrong"] == ch["chal_is_overturned"].astype(int)).mean()), 4)

# Runs at stake by count (Fig 3)
stake_ct = c26.groupby(["balls", "strikes"])["stake"].mean().unstack()
stake_ct.to_csv(RES / "runs_at_stake_by_count.csv")
R["stake_by_count"] = {f"{b}-{s}": round(float(stake_ct.loc[b, s]), 3) for b in stake_ct.index for s in stake_ct.columns}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
order = [(b, s) for s in range(3) for b in range(4)]
order = sorted(order, key=lambda t: stake_ct.loc[t])
vals = [stake_ct.loc[t] for t in order]
ax.barh([f"{b}-{s}" for b, s in order], vals, color=C26, height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_xlabel("Average runs swung by a ball/strike call")
ax.set_ylabel("Count")
ax.set_title("A full-count call is worth several times a first-pitch call")
ax.grid(axis="y", visible=False)
save(fig, "03_runs_at_stake_by_count.png")

# --- Option value of a challenge ---------------------------------------------
ch["chal_team"] = np.where(ch["chal_by_batter"] == 1, ch["bat_team"], ch["fld_team"])
ch["gain"] = np.where(ch["chal_is_overturned"], ch["stake"], 0.0)
ch = ch.sort_values(["game_pk", "at_bat_number", "pitch_number"])
teams_games = pitches[pitches.game_year == 2026][["game_pk", "home_team", "away_team"]].drop_duplicates("game_pk")
rows = []
chg = {k: v for k, v in ch.groupby(["game_pk", "chal_team"])}
for g, h, a in teams_games.itertuples(index=False):
    for t in (h, a):
        sub = chg.get((g, t))
        for inn in range(1, 10):
            if sub is None:
                rows.append((inn, 2, 0.0)); continue
            before = sub[sub.inning < inn]
            k = 2 - int((~before["chal_is_overturned"]).sum())
            fut = float(sub.loc[sub.inning >= inn, "gain"].sum()) if inn <= 9 else 0.0
            fut = float(sub.loc[(sub.inning >= inn) & (sub.inning <= 9), "gain"].sum())
            rows.append((inn, max(k, 0), fut))
ov = pd.DataFrame(rows, columns=["inning", "k", "future_gain"])
V = ov.groupby(["inning", "k"])["future_gain"].agg(["mean", "size"]).reset_index()
V.to_csv(RES / "challenge_value_by_inning.csv", index=False)
# V(k, t): expected future runs from challenges with k left at the start of inning t.
# Use cells with >= 100 team-games; nobody starts the 1st with one challenge, so
# missing early cells borrow the next inning's value. V(0, t) = 0 by definition.
Vm = V[V["size"] >= 100].pivot(index="inning", columns="k", values="mean").reindex(range(1, 10))
Vm = Vm.reindex(columns=[0, 1, 2]).bfill()
Vm[0] = 0.0
OV = pd.DataFrame({"lose_2nd_to_last": Vm[2] - Vm[1], "lose_last": Vm[1] - Vm[0]})
# smooth: option value can only shrink as the game goes on -> enforce monotone non-increasing
for c in OV:
    OV[c] = np.minimum.accumulate(OV[c].clip(lower=0).values)
OV.to_csv(RES / "option_value.csv")
R["option_value_runs"] = OV.round(3).to_dict()

# --- Break-even confidence: challenge if P(overturn) > OV / (stake + OV) ------
counts = ["0-0", "1-0", "0-1", "1-1", "2-1", "1-2", "2-2", "3-1", "3-2", "0-2", "3-0", "2-0"]
be = []
for inn in (1, 4, 7, 9):
    for k, col in ((2, "lose_2nd_to_last"), (1, "lose_last")):
        o = OV.loc[inn, col]
        for c in counts:
            b, s = map(int, c.split("-"))
            st = stake_ct.loc[b, s]
            be.append({"inning": inn, "challenges_left": k, "count": c, "stake": round(float(st), 3),
                       "option_value": round(float(o), 3), "breakeven_p": round(float(o / (st + o)), 3)})
be = pd.DataFrame(be)
be.to_csv(RES / "breakeven_confidence.csv", index=False)

# Fig 4 - break-even overturn probability, 2 challenges left
fig, ax = plt.subplots(figsize=(7.5, 4))
sel = ["0-0", "1-1", "0-2", "2-2", "3-1", "3-2"]
for i, (inn, ls) in enumerate([(1, "-"), (7, "--"), (9, ":")]):
    s = be[(be.inning == inn) & (be.challenges_left == 1)].set_index("count").loc[sel]
    ax.plot(sel, 100 * s["breakeven_p"], ls=ls, marker="o", ms=5, lw=2, color=[C26, C25, "#1baf7a"][i],
            label=f"Inning {inn}")
ax.set_ylabel("Confidence needed to challenge (%)")
ax.set_xlabel("Count when the call is made")
ax.set_ylim(0, 100)
ax.set_title("With one challenge left, how sure should you be?")
ax.legend(frameon=False)
save(fig, "extra/breakeven_runs_empirical.png")   # superseded by part2_extended.py

# --- Challenge behavior: do players weigh the stakes? --------------------------
# eligible = called against the side, side has a challenge left, innings 1-9
c26["against_batter"] = c26["ump_strike"] == 1
c26["side_left"] = np.where(c26["against_batter"], c26["bat_chal_left"], c26["fld_chal_left"])
c26["wrong"] = (c26["ump_strike"] != c26["abs_strike"]).astype(int)
el = c26[(c26.side_left > 0) & (c26.inning <= 9)].copy()
el["stake_bin"] = pd.qcut(el["stake"], 3, labels=["low", "medium", "high"])
clear = el[el.wrong == 1]
R["challenge_rate_on_wrong_calls_by_stake"] = clear.groupby("stake_bin", observed=True)["challenged"].mean().round(4).to_dict()
R["challenge_rate_on_correct_calls_by_stake"] = el[el.wrong == 0].groupby("stake_bin", observed=True)["challenged"].mean().round(4).to_dict()
R["share_of_wrong_calls_challenged"] = round(float(clear["challenged"].mean()), 4)

# --- Team scorecard ------------------------------------------------------------
OVl = OV.copy()
def cost(r):
    if r.chal_is_overturned:
        return 0.0
    k = r.bat_chal_left if r.chal_by_batter == 1 else r.fld_chal_left
    inn = int(min(r.inning, 9))
    return float(OVl.loc[inn, "lose_last" if k <= 1 else "lose_2nd_to_last"])
ch["cost"] = ch.apply(cost, axis=1)
ch["net"] = ch["gain"] - ch["cost"]
miss = el[(el.wrong == 1) & (el.challenged == 0) & (el.edge_dist_in.abs() >= 1.5)].copy()
miss["team"] = np.where(miss["against_batter"], miss["bat_team"], miss["fld_team"])
team = ch.groupby("chal_team").agg(challenges=("net", "size"), overturn_rate=("chal_is_overturned", "mean"),
                                   avg_stake=("stake", "mean"), runs_gained=("gain", "sum"),
                                   cost_of_lost=("cost", "sum"), net_runs=("net", "sum"))
team["missed_clear_calls"] = miss.groupby("team").size()
team["missed_runs"] = miss.groupby("team")["stake"].sum()
team["net_vs_avg"] = team["net_runs"] - team["net_runs"].mean()
team = team.sort_values("net_runs", ascending=False).round(3)
team.to_csv(RES / "team_scorecard.csv")
R["team_top3"] = team.head(3)["net_vs_avg"].to_dict()
R["league_avg_net_runs"] = round(float(team["net_runs"].mean()), 2)
R["avg_missed_runs_per_team"] = round(float(team["missed_runs"].mean()), 2)
R["avg_missed_clear_calls_per_team"] = round(float(team["missed_clear_calls"].mean()), 1)
R["team_bottom3"] = team.tail(3)["net_vs_avg"].to_dict()
R["corr_net_vs_avg_stake"] = round(float(team["net_runs"].corr(team["avg_stake"])), 3)
R["corr_net_vs_overturn_rate"] = round(float(team["net_runs"].corr(team["overturn_rate"])), 3)

# Fig 5 - team scorecard
t = team.sort_values("net_vs_avg")
fig, ax = plt.subplots(figsize=(7.5, 7.5))
cols = [C26 if v >= 0 else C25 for v in t["net_vs_avg"]]
ax.barh(t.index, t["net_vs_avg"], color=cols, height=0.7)
ax.axvline(0, color=INK2, lw=1)
for i, v in enumerate(t["net_vs_avg"]):
    ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=8, color=INK2)
ax.set_xlabel("Net challenge runs vs. league average\n(runs won minus the value of challenges lost)")
ax.set_xlim(t["net_vs_avg"].min() - 3, t["net_vs_avg"].max() + 3)
ax.set_title("2026 challenge scorecard by team")
ax.grid(axis="y", visible=False)
save(fig, "extra/team_scorecard_runs.png")   # superseded by team_ranking.py

json.dump(R, open(RES / "results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
