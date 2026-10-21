"""
Part 2, extended: when is a challenge worth making?

  A. Win-probability stakes  - value each call in wins, not runs, so game situation counts.
  B. Perception model         - how precisely batters and catchers judge pitch location,
                                estimated from when they choose to challenge.
  C. Threshold check          - the distance at which each role actually challenges vs. the best
                                threshold for a player with the same eyesight; wins lost to the gap.
  D. Batter vs. catcher, and a player leaderboard with bootstrap intervals.

    python src/part2_extended.py   ->  results/part2/*, figures/fig6-9*.png
Requires data/pitches.pkl.gz, data/called.pkl.gz (build_dataset.py), data/player_names.csv.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

OUT = Path("results/part2"); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path("figures")
(FIG / "extra").mkdir(parents=True, exist_ok=True)
R = {}

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b8b6b0"
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


class LapseLogit:
    """P(challenge) = lam * logistic(b0 + b.x). lam < 1 is the 'lapse' ceiling: the share of
    opportunities in which a player acts on what he sees at all (attention, willingness, or
    players who never challenge). Without it, the plateau on obvious misses flattens the slope
    and overstates judgment error. Fit by maximum likelihood."""

    def fit(self, X, y):
        from scipy.optimize import minimize
        X = np.asarray(X, float); y = np.asarray(y, float)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        Xs = (X - mu) / sd
        init = LogisticRegression(C=1e6, max_iter=2000).fit(Xs, y)
        th0 = np.concatenate([[2.0, init.intercept_[0]], init.coef_[0]])

        def nll(th):
            lam = 1 / (1 + np.exp(-th[0]))
            z = th[1] + Xs @ th[2:]
            p = np.clip(lam / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

        res = minimize(nll, th0, method="L-BFGS-B")
        th = res.x
        self.lam = float(1 / (1 + np.exp(-th[0])))
        self.coef_ = np.array([th[2:] / sd])
        self.intercept_ = np.array([th[1] - (th[2:] * mu / sd).sum()])
        self.converged = bool(res.success)
        return self

    def predict_proba(self, X):
        z = self.intercept_[0] + np.asarray(X, float) @ self.coef_[0]
        p = self.lam / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
P_COLS = ["game_pk", "game_year", "at_bat_number", "pitch_number", "inning", "inning_topbot",
          "outs", "base_state", "bat_score", "fld_score", "post_home_score", "post_away_score",
          "bat_win_exp"]
pitches = pd.read_pickle("data/pitches.pkl.gz")[P_COLS]
C_COLS = ["game_pk", "game_year", "at_bat_number", "pitch_number", "inning", "inning_topbot",
          "batter", "fielder_2", "edge_dist_in", "abs_strike", "ump_strike", "challenged",
          "chal_is_overturned", "chal_by_batter", "balls", "strikes", "outs", "base_state",
          "bat_score", "fld_score", "bat_team", "fld_team", "runs_at_stake",
          "bat_chal_left", "fld_chal_left"]
called = pd.read_pickle("data/called.pkl.gz")[C_COLS]
called = called[(called.game_year == 2026) & called.edge_dist_in.notna()].copy()

re_tab = pd.read_csv("data/run_expectancy_2026.csv")
RE = {(r.outs, r.base_state, r.balls, r.strikes): r.re for r in re_tab.itertuples()}
RE_OB = re_tab[(re_tab.balls == 0) & (re_tab.strikes == 0)].set_index(["outs", "base_state"])["re"].to_dict()

# ===========================================================================
# A. Win probability model
# ===========================================================================
# WP for the batting team from (inning, half, effective lead), where
#   effective lead L = score margin + runs expected for the rest of this half-inning.
# One logistic curve per inning (1..9, extras pooled with 9) x half. Fit on the first pitch
# of every plate appearance, 2025-26.
finals = pitches.groupby("game_pk")[["post_home_score", "post_away_score"]].max()
home_won = (finals.post_home_score > finals.post_away_score).rename("home_won")
pa = pitches[pitches.pitch_number == 1].join(home_won, on="game_pk")
pa["inn"] = pa["inning"].clip(upper=9)
pa["half"] = np.where(pa["inning_topbot"] == "Top", "Top", "Bot")
pa["bat_won"] = np.where(pa["half"] == "Bot", pa["home_won"], ~pa["home_won"]).astype(int)
pa["L"] = (pa["bat_score"] - pa["fld_score"]) + [RE_OB.get((o, b), 0.0) for o, b in zip(pa.outs, pa.base_state)]
pa["L"] = pa["L"].clip(-10, 10)

WP = {}
for (inn, half), g in pa.groupby(["inn", "half"]):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(g[["L"]], g["bat_won"])
    WP[(inn, half)] = (float(m.intercept_[0]), float(m.coef_[0, 0]))
pd.DataFrame([{"inning": k[0], "half": k[1], "intercept": v[0], "slope_per_run": v[1]} for k, v in WP.items()]) \
    .to_csv(OUT / "wp_model.csv", index=False)


def wp_of(inn, half, L):
    a, b = WP[(min(int(inn), 9), half)]
    return 1 / (1 + np.exp(-(a + b * L)))


# check: Brier score, and agreement with Savant's own win expectancy
pa["wp_hat"] = [wp_of(i, h, l) for i, h, l in zip(pa.inn, pa.half, pa.L)]
R["wp_brier"] = round(float(((pa.wp_hat - pa.bat_won) ** 2).mean()), 4)
ok = pa.bat_win_exp.notna()
R["wp_corr_with_savant"] = round(float(np.corrcoef(pa.loc[ok, "wp_hat"], pa.loc[ok, "bat_win_exp"])[0, 1]), 4)
cal = pa.assign(b=pd.cut(pa.wp_hat, np.linspace(0, 1, 11))).groupby("b", observed=True) \
    .agg(pred=("wp_hat", "mean"), actual=("bat_won", "mean"), n=("bat_won", "size"))
cal.to_csv(OUT / "wp_calibration.csv")
R["wp_max_calibration_gap"] = round(float((cal.pred - cal.actual).abs().max()), 4)


# Value of each called pitch in wins: WP if the call is a ball minus WP if it is a strike.
def after(o, b, bl, st):
    """(runs scored now, runs expected after) once the pitch is ruled; outs after."""
    if st >= 3:
        o += 1
        return 0, (0.0 if o >= 3 else RE.get((o, b, 0, 0), RE_OB.get((o, b), 0))), o
    if bl >= 4:
        on1, on2, on3 = b & 1, (b >> 1) & 1, (b >> 2) & 1
        runs = int(on1 and on2 and on3)
        nb = 1 + 2 * (on2 or on1) + 4 * (on3 or (on1 and on2))
        return runs, RE.get((o, nb, 0, 0), RE_OB.get((o, nb), 0)), o
    return 0, RE.get((o, b, bl, st), RE_OB.get((o, b), 0)), o


def wp_after(inn, half, margin, runs, re_after, outs_after):
    m = margin + runs
    if half == "Bot" and inn >= 9 and m > 0:
        return 1.0                       # walk-off
    if outs_after >= 3 and half == "Bot" and inn >= 9 and m < 0:
        return 0.0                       # game over
    return wp_of(inn, half, np.clip(m + re_after, -10, 10))


called["half"] = np.where(called["inning_topbot"] == "Top", "Top", "Bot")
called["margin"] = called["bat_score"] - called["fld_score"]
wb, ws = [], []
for inn, half, mg, o, b, bl, st in zip(called.inning, called.half, called.margin, called.outs,
                                       called.base_state, called.balls, called.strikes):
    wb.append(wp_after(inn, half, mg, *after(o, b, bl + 1, st)))
    ws.append(wp_after(inn, half, mg, *after(o, b, bl, st + 1)))
called["win_stake"] = np.abs(np.array(wb) - np.array(ws))
called["run_stake"] = called["runs_at_stake"].abs()

R["win_stake_mean_pct"] = round(100 * float(called.win_stake.mean()), 3)
srt = np.sort(called.win_stake.values)[::-1]
R["share_of_win_stakes_in_top_10pct_calls"] = round(float(srt[: len(srt) // 10].sum() / srt.sum()), 3)
R["share_of_run_stakes_in_top_10pct_calls"] = round(float(np.sort(called.run_stake.values)[::-1][: len(srt) // 10].sum() / called.run_stake.sum()), 3)

# situations a manager thinks in
def situation(r):
    late = r.inning >= 7
    close = abs(r.margin) <= 1
    if close and late:
        return "Close & late (7th+, within 1)"
    if close:
        return "Close, innings 1-6"
    if abs(r.margin) >= 4:
        return "Blowout (4+ runs)"
    return "Other"
called["situation"] = [situation(r) for r in called[["inning", "margin"]].itertuples()]
sit = called.groupby("situation").agg(calls=("win_stake", "size"), win_stake_pct=("win_stake", "mean"),
                                      run_stake=("run_stake", "mean"))
sit["win_stake_pct"] *= 100
sit.round(3).to_csv(OUT / "stakes_by_situation.csv")
R["stakes_by_situation"] = sit.round(3).to_dict(orient="index")

# ===========================================================================
# Challenge state and option value, in wins
# ===========================================================================
called = called.sort_values(["game_pk", "at_bat_number", "pitch_number"])
called["against_batter"] = called["ump_strike"] == 1
called["side_left"] = np.where(called["against_batter"], called["bat_chal_left"], called["fld_chal_left"])
called["role"] = np.where(called["against_batter"], "batter", "catcher")
called["d_fav"] = np.where(called["against_batter"], called["edge_dist_in"], -called["edge_dist_in"])  # >0 = call wrong
called["wrong"] = (called["d_fav"] > 0).astype(int)

ch = called[called.challenged == 1].copy()
ch["chal_team"] = np.where(ch["chal_by_batter"] == 1, ch["bat_team"], ch["fld_team"])
ch["gain_w"] = np.where(ch["chal_is_overturned"], ch["win_stake"], 0.0)
ch["gain_r"] = np.where(ch["chal_is_overturned"], ch["run_stake"], 0.0)
games = called[["game_pk", "bat_team", "fld_team"]].drop_duplicates("game_pk")
chg = {k: v for k, v in ch.groupby(["game_pk", "chal_team"])}
rows = []
for g, t1, t2 in games.itertuples(index=False):
    for t in (t1, t2):
        sub = chg.get((g, t))
        for inn in range(1, 10):
            if sub is None:
                rows.append((inn, 2, 0.0, 0.0)); continue
            k = 2 - int((~sub.loc[sub.inning < inn, "chal_is_overturned"]).sum())
            fut = sub[(sub.inning >= inn) & (sub.inning <= 9)]
            rows.append((inn, max(k, 0), float(fut.gain_w.sum()), float(fut.gain_r.sum())))
ov = pd.DataFrame(rows, columns=["inning", "k", "fw", "fr"])


def option_values(col):
    V = ov.groupby(["inning", "k"])[col].agg(["mean", "size"]).reset_index()
    Vm = V[V["size"] >= 100].pivot(index="inning", columns="k", values="mean").reindex(range(1, 10))
    Vm = Vm.reindex(columns=[0, 1, 2]).bfill()
    Vm[0] = 0.0
    out = pd.DataFrame({"last": Vm[1] - Vm[0], "second": Vm[2] - Vm[1]})
    for c in out:
        out[c] = np.minimum.accumulate(out[c].clip(lower=0).values)
    # a second challenge can't be worth more than the last one (teams down to one challenge
    # are, by selection, weaker challengers, which biases V(1, t) down)
    out["second"] = np.minimum(out["second"], out["last"])
    return out


OVW, OVR = option_values("fw"), option_values("fr")
pd.concat({"wins": OVW, "runs": OVR}, axis=1).to_csv(OUT / "option_value_wins_runs.csv")
R["option_value_wins_pct"] = (100 * OVW).round(3).to_dict()

called["ov_w"] = [OVW.loc[min(i, 9), "last" if k <= 1 else "second"] if k > 0 else np.nan
                  for i, k in zip(called.inning, called.side_left.fillna(0))]
called["ov_r"] = [OVR.loc[min(i, 9), "last" if k <= 1 else "second"] if k > 0 else np.nan
                  for i, k in zip(called.inning, called.side_left.fillna(0))]



# ===========================================================================
# B. Perception model
# ===========================================================================
# Opportunity = a called pitch that went against a side still holding a challenge (innings 1-9).
# Assume a player challenges when his perceived distance  d + e  (e ~ noise) passes a threshold
# that depends on the stakes. Then P(challenge | d) is an S-curve in d whose steepness measures
# how precisely the player judges location. Fit by role with logistic regression.
el = called[(called.side_left > 0) & (called.inning <= 9)].copy()
el["log_run_stake"] = np.log(el["run_stake"].clip(lower=0.01))
el["log_leverage"] = np.log((el["win_stake"] / el["run_stake"].clip(lower=0.01)).clip(lower=1e-4))
el["late"] = (el["inning"] >= 7).astype(int)
el["last_one"] = (el["side_left"] == 1).astype(int)
FEATS = ["d_fav", "log_run_stake", "log_leverage", "late", "last_one"]

perc = {}
for role, g in el.groupby("role"):
    fit_rows = g[g.d_fav.abs() <= 6]
    m = LapseLogit().fit(fit_rows[FEATS], fit_rows["challenged"])
    coef = dict(zip(FEATS, m.coef_[0]))
    s_logit = 1 / coef["d_fav"]
    sigma = s_logit * np.pi / np.sqrt(3)                 # SD of the implied judgment error, inches
    # threshold at median stakes, not late, two left: distance where P(challenge) = 50%
    x0 = fit_rows[FEATS].median()
    x0["late"], x0["last_one"] = 0, 0
    lin_wo_d = m.intercept_[0] + sum(coef[f] * x0[f] for f in FEATS if f != "d_fav")
    thresh = -lin_wo_d / coef["d_fav"]
    perc[role] = {"model": m, "sigma_in": sigma, "threshold_in": thresh, "lapse_ceiling": m.lam,
                  "coef": {k: round(float(v), 4) for k, v in coef.items()}}
    # observed vs fitted challenge rate by distance
    bins = np.arange(-6, 6.01, 0.5)
    obs = fit_rows.assign(b=pd.cut(fit_rows.d_fav, bins)).groupby("b", observed=True) \
        .agg(rate=("challenged", "mean"), n=("challenged", "size"))
    obs["mid"] = [b.mid for b in obs.index]
    obs["fit"] = [m.predict_proba(fit_rows[fit_rows.d_fav.between(b.left, b.right)][FEATS])[:, 1].mean()
                  for b in obs.index]
    obs.to_csv(OUT / f"perception_fit_{role}.csv")
    perc[role]["obs"] = obs

R["perception"] = {r: {"sigma_in": round(v["sigma_in"], 2), "threshold_in_at_median_stakes": round(v["threshold_in"], 2),
                       "ceiling": round(v["lapse_ceiling"], 3), "converged": v["model"].converged,
                       "coef": v["coef"]} for r, v in perc.items()}

# Fig 7 - challenge rate vs distance, by role
fig, ax = plt.subplots(figsize=(7.5, 4))
for role, col in [("catcher", C1), ("batter", C2)]:
    o = perc[role]["obs"]
    ax.scatter(o["mid"], 100 * o["rate"], color=col, s=22, zorder=3)
    ax.plot(o["mid"], 100 * o["fit"], color=col, lw=2,
            label=f"{role.title()}s: judgment error ≈ {perc[role]['sigma_in']:.1f} in, ceiling {100 * perc[role]['lapse_ceiling']:.0f}%")
ax.axvline(0, color=INK2, lw=1, ls="--")
ax.text(0.15, 2, "call is wrong →", color=INK2, fontsize=9)
ax.text(-0.15, 2, "← call is right", color=INK2, fontsize=9, ha="right")
ax.set_xlabel("How far the pitch was on the challenger's side of the zone edge (inches)")
ax.set_ylabel("Share of calls challenged (%)")
ax.set_title("Catchers read the zone more sharply than batters")
ax.legend(frameon=False, loc="upper left")
save(fig, "05_perception_by_role.png")


# ===========================================================================
# C. Is each role's challenge threshold in the right place?
# ===========================================================================
# Under the model a player challenges when his perceived distance X = d + e passes a threshold.
# The best threshold for someone with the same eyesight sets
#     P(call wrong | X = tau*) = OV / (stake + OV)
# so it moves with the stakes and with the inning. Compare it with the threshold players use.
def posterior_curve(d_values, sigma):
    """p(x) = P(true distance > 0 | perceived = x); prior = the role's distances."""
    grid = np.arange(-12, 12.001, 0.05)
    prior, _ = np.histogram(d_values.clip(-12, 12), bins=np.append(grid - 0.025, grid[-1] + 0.025))
    prior = prior / prior.sum()
    like = norm.pdf((grid[:, None] - grid[None, :]) / sigma)
    post = like * prior[None, :]
    return grid, post[:, grid > 0].sum(1) / post.sum(1)


curves = {r: posterior_curve(el.loc[el.role == r, "d_fav"].values, perc[r]["sigma_in"]) for r in perc}


def tau_star(role, p_star):
    grid, p = curves[role]
    return np.interp(p_star, p, grid)        # p is increasing in x


# ---------------------------------------------------------------------------
# Simulated option value (fixed point).
# The empirical option value was measured under teams' actual, cautious behavior. A policy that
# challenges more spends challenges faster and should value a saved one more. So: replay every
# 2026 game (innings 1-9) for each team, with perception noise drawn per pitch, under a policy;
# re-estimate V(k, t) from those games; recompute the policy; repeat until it settles.
# ---------------------------------------------------------------------------
S = 30
sim = called[called.inning <= 9].copy()
sim["team"] = np.where(sim["against_batter"], sim["bat_team"], sim["fld_team"])
sim["key"] = sim["game_pk"].astype(str) + "_" + sim["team"]
keys, key_idx = np.unique(sim["key"].values, return_inverse=True)
sim_key, sim_inn = key_idx, sim["inning"].values.astype(int)
sim_role = (sim["role"].values == "catcher").astype(int)
sim_d, sim_ws, sim_wrong = sim["d_fav"].values, sim["win_stake"].values, sim["wrong"].values
SIG = np.array([perc["batter"]["sigma_in"], perc["catcher"]["sigma_in"]])
Z = np.random.default_rng(5).standard_normal((len(sim), S)).astype(np.float32)   # common random numbers
U = np.random.default_rng(6).random((len(sim), S)).astype(np.float32)
LAM = np.array([perc["batter"]["lapse_ceiling"], perc["catcher"]["lapse_ceiling"]])


def lin_wo_d_all(role):
    mdl, coef = perc[role]["model"], perc[role]["coef"]
    x = pd.DataFrame({"log_run_stake": np.log(sim["run_stake"].clip(lower=0.01)),
                      "log_leverage": np.log((sim["win_stake"] / sim["run_stake"].clip(lower=0.01)).clip(lower=1e-4)),
                      "late": (sim["inning"] >= 7).astype(int)})
    base = mdl.intercept_[0] + sum(coef[f] * x[f].values for f in x)
    return base, coef["last_one"], coef["d_fav"]


ACT = {r: lin_wo_d_all(r) for r in ("batter", "catcher")}


def simulate(policy, OV):
    # policies: 'optimal' = best thresholds, never lapses; 'optimal_lapse' = best thresholds
    # with the role's actual lapse rate; 'actual' = fitted thresholds and lapses
    """policy: 'optimal' (uses OV) or 'actual' (fitted thresholds). Returns V, totals."""
    nk = len(keys)
    k = np.full((nk, S), 2, dtype=np.int8)
    kstart = np.full((nk, 10, S), -1, dtype=np.int8)       # k at the start of innings 1..9
    filled = np.zeros(nk, dtype=np.int8)                   # last inning whose start is filled
    gains = np.zeros((nk, 10, S), dtype=np.float32)
    n_chal = np.zeros(2); n_fail = np.zeros(2); g_role = np.zeros(2)
    ov_last = OV["last"].values; ov_sec = OV["second"].values
    for i in range(len(sim)):
        kk, t, r = sim_key[i], sim_inn[i], sim_role[i]
        if filled[kk] < t:
            kstart[kk, filled[kk] + 1:t + 1] = k[kk]
            filled[kk] = t
        kv = k[kk]
        avail = kv > 0
        if not avail.any():
            continue
        perceived = sim_d[i] + SIG[r] * Z[i]
        if policy in ("optimal", "optimal_lapse"):
            ov = np.where(kv >= 2, ov_sec[t - 1], ov_last[t - 1])
            p_star = ov / (sim_ws[i] + ov)
            g, pc = curves["catcher" if r else "batter"]
            tau = np.interp(p_star, pc, g)
        else:
            base, b_last, b_d = ACT["catcher" if r else "batter"]
            tau = -(base[i] + b_last * (kv == 1)) / b_d
        go = avail & (perceived > tau)
        if policy in ("actual", "optimal_lapse"):
            go &= U[i] < LAM[r]                            # lapses: sees it, doesn't act
        if not go.any():
            continue
        n_chal[r] += go.sum()
        if sim_wrong[i]:
            gains[kk, t, go] += sim_ws[i]
            g_role[r] += go.sum() * sim_ws[i]
        else:
            k[kk, go] -= 1
            n_fail[r] += go.sum()
    for kk in range(nk):
        if filled[kk] < 9:
            kstart[kk, filled[kk] + 1:10] = k[kk]
    fut = gains[:, ::-1, :].cumsum(1)[:, ::-1, :]           # gains from inning t onward
    V = {}
    for t in range(1, 10):
        for kv in (0, 1, 2):
            m = kstart[:, t, :] == kv
            V[(t, kv)] = (float(fut[:, t, :][m].mean()) if m.sum() >= 200 else np.nan, int(m.sum()))
    total = gains.sum(1)                                    # per key x sim
    return V, {"wins_per_team_game": float(total.mean()), "total": total,
               "chal_per_team_game": n_chal.sum() / (nk * S), "fail_per_team_game": n_fail.sum() / (nk * S),
               "chal_by_role": n_chal / S, "fail_by_role": n_fail / S, "wins_by_role": g_role / S}


def ov_from_V(V):
    out = pd.DataFrame(index=range(1, 10), columns=["last", "second"], dtype=float)
    for t in range(1, 10):
        v0, v1, v2 = 0.0, V[(t, 1)][0], V[(t, 2)][0]
        out.loc[t, "last"], out.loc[t, "second"] = v1 - v0, v2 - v1
    out = out.bfill().ffill()
    for c in out:
        out[c] = np.minimum.accumulate(out[c].clip(lower=0).values)
    out["second"] = np.minimum(out["second"], out["last"])
    return out


OV_opt = OVW.copy()
history = []
for it in range(8):
    V, info = simulate("optimal", OV_opt)
    new = ov_from_V(V)
    change = float((new - OV_opt).abs().max().max())
    history.append({"iter": it, "max_change_pct": 100 * change, "wins_per_team_game": info["wins_per_team_game"]})
    OV_opt = new
    print("fixed point iter", it, "max change (win %)", round(100 * change, 4), flush=True)
    if change < 0.0005:
        break
V_opt, info_opt = simulate("optimal", OV_opt)
V_act, info_act = simulate("actual", OV_opt)
V_ol, info_ol = simulate("optimal_lapse", OV_opt)
OV_opt.to_csv(OUT / "option_value_simulated.csv")
pd.DataFrame(history).to_csv(OUT / "fixed_point_history.csv", index=False)
n_team_games = len(keys)
R["option_value_simulated_wins_pct"] = (100 * OV_opt).round(3).to_dict()
R["fixed_point_iterations"] = len(history)
for name, inf in (("optimal", info_opt), ("optimal_thresholds_actual_lapses", info_ol), ("actual", info_act)):
    R[f"sim_{name}"] = {
        "wins_per_team_season": round(inf["wins_per_team_game"] * 162, 2),
        "challenges_per_team_game": round(inf["chal_per_team_game"], 3),
        "failed_per_team_game": round(inf["fail_per_team_game"], 3),
        "overturn_rate": round(1 - inf["fail_per_team_game"] / inf["chal_per_team_game"], 3),
        "wins_by_role_league": {"batter": round(float(inf["wins_by_role"][0]), 2), "catcher": round(float(inf["wins_by_role"][1]), 2)},
        "challenges_by_role_league": {"batter": round(float(inf["chal_by_role"][0])), "catcher": round(float(inf["chal_by_role"][1]))},
    }
# check: the 'actual' simulation should reproduce what really happened
real = el[el.challenged == 1]
R["real_wins_per_team_season"] = round(float((real.chal_is_overturned * real.win_stake).sum() / 30), 2)
R["real_challenges_per_team_game"] = round(float(len(real) / n_team_games), 3)
team_of_key = np.array([k.split("_", 1)[1] for k in keys])
team_gap = pd.DataFrame({"team": team_of_key,
                         "opt": info_opt["total"].mean(1), "opt_lapse": info_ol["total"].mean(1),
                         "act": info_act["total"].mean(1)}).groupby("team").sum()
team_gap["wins_gap"] = team_gap["opt"] - team_gap["act"]
team_gap["gap_thresholds"] = team_gap["opt_lapse"] - team_gap["act"]
team_gap["gap_lapses"] = team_gap["opt"] - team_gap["opt_lapse"]
team_gap.sort_values("wins_gap").round(3).to_csv(OUT / "team_sim_gap.csv")
R["team_sim_gap_mean_wins"] = round(float(team_gap.wins_gap.mean()), 3)
R["team_sim_gap_range"] = [round(float(team_gap.wins_gap.min()), 3), round(float(team_gap.wins_gap.max()), 3)]
el["ov_w"] = [OV_opt.loc[min(i, 9), "last" if k <= 1 else "second"] for i, k in zip(el.inning, el.side_left)]

# Fig 6 - break-even by situation (one challenge left)
counts = ["0-0", "1-1", "0-2", "2-2", "3-1", "3-2"]
cases = [("Tied, 3rd inning", 3, "Top", 0), ("Up 5, 7th inning", 7, "Top", 5), ("Tied, 9th inning", 9, "Top", 0)]
be_rows = []
for label, inn, half, mg in cases:
    for c in counts:
        bl, st = map(int, c.split("-"))
        o, b = 1, 0                                    # one out, bases empty
        s = abs(wp_after(inn, half, mg, *after(o, b, bl + 1, st)) - wp_after(inn, half, mg, *after(o, b, bl, st + 1)))
        ovw = OV_opt.loc[inn, "last"]
        be_rows.append({"case": label, "count": c, "win_stake_pct": 100 * s, "ov_pct": 100 * ovw,
                        "breakeven": ovw / (s + ovw)})
be = pd.DataFrame(be_rows)
be.round(4).to_csv(OUT / "breakeven_by_situation.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4))
for (label, *_), col, ls in zip(cases, [MUTED, C2, C1], ["-", "--", "-"]):
    s = be[be.case == label]
    ax.plot(s["count"], 100 * s["breakeven"], color=col, ls=ls, lw=2.2, marker="o", ms=5, label=label)
ax.set_ylim(0, 100)
ax.set_ylabel("Confidence needed to challenge (%)")
ax.set_xlabel("Count when the call is made (one out, bases empty, last challenge)")
ax.set_title("In a blowout, almost no call is worth your last challenge")
ax.legend(frameon=False, loc="upper right")
save(fig, "04_breakeven_by_situation.png")

el["p_star"] = el["ov_w"] / (el["win_stake"] + el["ov_w"])
el["q_act"] = np.nan; el["q_opt"] = np.nan; el["tau_act"] = np.nan; el["tau_opt"] = np.nan
for role in ("batter", "catcher"):
    m = el.role == role
    mdl, coef = perc[role]["model"], perc[role]["coef"]
    el.loc[m, "q_act"] = mdl.predict_proba(el.loc[m, FEATS])[:, 1]
    lin_wo_d = mdl.intercept_[0] + sum(coef[f] * el.loc[m, f] for f in FEATS if f != "d_fav")
    el.loc[m, "tau_act"] = -lin_wo_d / coef["d_fav"]
    el.loc[m, "tau_opt"] = tau_star(role, el.loc[m, "p_star"].values)
    el.loc[m, "q_opt"] = 1 - norm.cdf((el.loc[m, "tau_opt"] - el.loc[m, "d_fav"]) / perc[role]["sigma_in"])

R["opportunities"] = int(len(el))

# value of a lost challenge for a team that behaves as teams actually do (from the actual-policy
# simulation): the cost side of a player's realized challenge value
OV_act = ov_from_V(V_act)
OV_act.to_csv(OUT / "option_value_actual_policy.csv")
el["ov_act"] = [OV_act.loc[min(i, 9), "last" if k <= 1 else "second"] for i, k in zip(el.inning, el.side_left)]
el["value_if_challenged"] = np.where(el["wrong"] == 1, el["win_stake"], -el["ov_act"])
el["realized"] = el["challenged"] * el["value_if_challenged"]
el["expected_avg"] = el["q_act"] * el["value_if_challenged"]      # an average player of the role

# ===========================================================================
# D. Batter vs catcher (actual behavior + the simulation, which respects the two-challenge budget)
# ===========================================================================
roles = pd.DataFrame(index=["batter", "catcher"])
for role in roles.index:
    g = el[el.role == role]
    w = g.wrong == 1
    i = 0 if role == "batter" else 1
    roles.loc[role, "opportunities"] = len(g)
    roles.loc[role, "challenges"] = int(g.challenged.sum())
    roles.loc[role, "overturn_rate_pct"] = 100 * g.loc[g.challenged == 1, "chal_is_overturned"].mean()
    roles.loc[role, "pct_wrong_calls_challenged"] = 100 * g.loc[w, "challenged"].mean()
    roles.loc[role, "pct_right_calls_challenged"] = 100 * g.loc[~w, "challenged"].mean()
    roles.loc[role, "judgment_error_in"] = perc[role]["sigma_in"]
    roles.loc[role, "acts_on_what_he_sees_pct"] = 100 * perc[role]["lapse_ceiling"]
    roles.loc[role, "sim_wins_actual"] = info_act["wins_by_role"][i]
    roles.loc[role, "sim_wins_best_thresholds"] = info_ol["wins_by_role"][i]
    roles.loc[role, "sim_wins_best_thresholds_no_lapses"] = info_opt["wins_by_role"][i]
    roles.loc[role, "sim_challenges_actual"] = info_act["chal_by_role"][i]
    roles.loc[role, "sim_challenges_best"] = info_opt["chal_by_role"][i]
roles.round(3).to_csv(OUT / "role_comparison.csv")
R["roles"] = roles.round(3).to_dict(orient="index")

# Fig 8 - actual vs optimal threshold by count (5th inning, tied, one out, bases empty, last challenge)
counts = ["0-0", "1-1", "0-2", "2-2", "3-1", "3-2"]
rows = []
for c in counts:
    bl, st = map(int, c.split("-"))
    rs = abs((lambda a, b: (a[0] + a[1]) - (b[0] + b[1]))(after(1, 0, bl + 1, st), after(1, 0, bl, st + 1)))
    ws_ = abs(wp_after(5, "Top", 0, *after(1, 0, bl + 1, st)) - wp_after(5, "Top", 0, *after(1, 0, bl, st + 1)))
    ovw = OV_opt.loc[5, "last"]
    for role in ("batter", "catcher"):
        mdl, coef = perc[role]["model"], perc[role]["coef"]
        x = {"log_run_stake": np.log(max(rs, 0.01)), "log_leverage": np.log(max(ws_ / max(rs, 0.01), 1e-4)),
             "late": 0, "last_one": 1}
        ta = -(mdl.intercept_[0] + sum(coef[f] * x[f] for f in x)) / coef["d_fav"]
        to = float(tau_star(role, ovw / (ws_ + ovw)))
        rows.append({"count": c, "role": role, "tau_actual": ta, "tau_optimal": to})
th = pd.DataFrame(rows)
th.round(3).to_csv(OUT / "thresholds_by_count.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for role, col in [("batter", C2), ("catcher", C1)]:
    s = th[th.role == role]
    ax.plot(s["count"], s["tau_actual"], color=col, lw=2.2, marker="o", ms=5, label=f"{role.title()}s: actual")
    ax.plot(s["count"], s["tau_optimal"], color=col, lw=2, ls="--", marker="o", ms=4, mfc=SURF,
            label=f"{role.title()}s: optimal")
ax.axhline(0, color=INK2, lw=1)
ax.text(5.35, 0.08, "zone edge", color=INK2, fontsize=8.5, ha="right")
ax.set_ylabel("Challenge when the pitch looks this far\non your side of the edge (inches)")
ax.set_xlabel("Count (5th inning, tied, one out, bases empty, last challenge)")
ax.set_title("The best threshold falls steeply with the count; players' barely moves")
ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="upper right")
save(fig, "extra/thresholds_actual_vs_optimal.png")

# ===========================================================================
# Player leaderboard: challenge value above an average player of the same role, same pitches
# ===========================================================================
names = pd.read_csv("data/player_names.csv").drop_duplicates("id").set_index("id")["name"]
el["player"] = np.where(el.role == "batter", el["batter"], el["fielder_2"])
el["above_avg"] = el["realized"] - el["expected_avg"]
rng = np.random.default_rng(11)
MIN_OPP = {"batter": 150, "catcher": 600}
lb = []
for (role, pid), g in el.groupby(["role", "player"]):
    if len(g) < MIN_OPP[role]:
        continue
    v = g["above_avg"].values * 100                  # win %
    boots = v[rng.integers(0, len(v), size=(1000, len(v)))].sum(1)
    lb.append({"role": role, "player_id": int(pid), "name": names.get(int(pid), str(int(pid))),
               "opportunities": len(g), "challenges": int(g.challenged.sum()),
               "challenges_expected_avg": g["q_act"].sum(),
               "overturn_rate": g.loc[g.challenged == 1, "chal_is_overturned"].mean(),
               "win_pct_above_avg": v.sum(), "ci_low": np.percentile(boots, 5), "ci_high": np.percentile(boots, 95)})
lb = pd.DataFrame(lb)
lb["separable"] = (lb.ci_low > 0) | (lb.ci_high < 0)
lb = lb.sort_values(["role", "win_pct_above_avg"], ascending=[True, False])
lb.round(4).to_csv(OUT / "player_leaderboard.csv", index=False)
R["leaderboard_n"] = lb.role.value_counts().to_dict()
R["leaderboard_share_separable_90pct"] = lb.groupby("role")["separable"].mean().round(3).to_dict()
for role in ("batter", "catcher"):
    s_ = lb[lb.role == role][["name", "win_pct_above_avg", "ci_low", "ci_high", "challenges", "challenges_expected_avg", "overturn_rate"]].round(3)
    R[f"best_{role}s"] = s_.head(5).to_dict(orient="records")
    R[f"worst_{role}s"] = s_.tail(5).to_dict(orient="records")
# year-to-year stability can't be checked with one season; split-half reliability instead
half = el.assign(h=(el.game_pk % 2)).groupby(["role", "player", "h"])["above_avg"].sum().unstack()
half = half.join(el.groupby(["role", "player"]).size().rename("n"))
for role, mn in MIN_OPP.items():
    hh = half.loc[role].dropna()
    hh = hh[hh.n >= mn]
    R[f"split_half_corr_{role}"] = round(float(np.corrcoef(hh[0], hh[1])[0, 1]), 3)

# Fig 9 - catcher leaderboard: top 10 and bottom 10, win % above average with 90% intervals
c = lb[lb.role == "catcher"]
show = pd.concat([c.head(10), c.tail(10)]).sort_values("win_pct_above_avg").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
y = np.arange(len(show))
cols = [C1 if v > 0 else MUTED for v in show.win_pct_above_avg]
ax.hlines(y, show.ci_low / 100, show.ci_high / 100, color=cols, lw=2)
ax.scatter(show.win_pct_above_avg / 100, y, color=cols, s=36, zorder=3)
ax.axvline(0, color=INK2, lw=1)
ax.axhline(9.5, color=GRID, lw=1)
ax.set_yticks(y, show["name"])
ax.set_xlabel("Wins from challenges above an average catcher on the same pitches, 2026\n(dots = estimate; bars = 90% interval)")
ax.set_title("Catcher challenge value: top 10 and bottom 10")
ax.grid(axis="y", visible=False)
save(fig, "06_catcher_leaderboard.png")

el[["game_pk", "at_bat_number", "pitch_number", "role", "player", "inning", "balls", "strikes",
    "d_fav", "run_stake", "win_stake", "ov_w", "ov_act", "p_star", "tau_act", "tau_opt", "q_act", "q_opt",
    "challenged", "chal_is_overturned", "realized", "expected_avg"]].to_pickle("data/opportunities_graded.pkl.gz")

json.dump(R, open(OUT / "results_part2.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
