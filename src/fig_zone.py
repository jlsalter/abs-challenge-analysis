"""
Strike-zone figure (Part 1): how the called zone changed from 2025 to 2026.

    python src/fig_zone.py   ->  figures/fig2_zone_*.png
Fits the two called-strike surfaces once and caches them in results/zone_surfaces.npz.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier

FIG = Path("figures")
(FIG / "extra").mkdir(parents=True, exist_ok=True)
CACHE = Path("results/zone_surfaces.npz")
C26, C25 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left",
})

# ---------------------------------------------------------------------------
# Surfaces: P(called strike) by location, per season (same model as analysis.py)
# ---------------------------------------------------------------------------
if CACHE.exists():
    z = np.load(CACHE)
    xs, zs, H_IN, s25, s26 = z["xs"], z["zs"], float(z["H_IN"]), z["s25"], z["s26"]
else:
    cols = ["game_year", "stand", "px", "abs_top", "abs_bot", "edge_dist_in", "z_norm", "ump_strike"]
    called = pd.read_pickle("data/called.pkl.gz")[cols]
    called = called[called.abs_top.notna() & called.edge_dist_in.notna()]
    called["x_away"] = np.where(called.stand == "R", called.px, -called.px) * 12
    H_IN = 12 * float((called.abs_top - called.abs_bot).median())
    win = called[(called.x_away.abs() <= 18) & called.z_norm.between(-0.7, 1.7)]
    xs = np.linspace(-15, 15, 241)
    zs = np.linspace(-0.4, 1.4, 241)
    gx, gz = np.meshgrid(xs, zs)
    grid = pd.DataFrame({"x_away": gx.ravel(), "z_norm": gz.ravel()})
    out = {}
    for yr in (2025, 2026):
        d = win[win.game_year == yr]
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, random_state=0)
        m.fit(d[["x_away", "z_norm"]], d["ump_strike"])
        out[yr] = m.predict_proba(grid)[:, 1].reshape(gz.shape)
    s25, s26 = out[2025], out[2026]
    np.savez(CACHE, xs=xs, zs=zs, H_IN=H_IN, s25=s25, s26=s26)

Z_IN = zs * H_IN                                   # height above the bottom of the ABS zone, inches
# light smoothing removes the tree model's stair-steps; areas below use the unsmoothed surfaces
sm25, sm26 = gaussian_filter(s25, 2.0), gaussian_filter(s26, 2.0)
cell = (xs[1] - xs[0]) * (Z_IN[1] - Z_IN[0])
# areas on the analysis grid (every other point = the 121x121 grid analysis.py uses), so numbers match
cell2 = (xs[2] - xs[0]) * (Z_IN[2] - Z_IN[0])
A25, A26 = (s25[::2, ::2] >= 0.5).sum() * cell2, (s26[::2, ::2] >= 0.5).sum() * cell2
EX = 8.5 + 1.45                                   # ABS zone half-width incl. ball radius
BOX = (-EX, -1.45, 2 * EX, H_IN + 2 * 1.45)        # x, y, w, h
ABS_AREA = BOX[2] * BOX[3]


def plate(ax, y):
    """Home plate seen from the catcher, under the zone, for orientation."""
    w = 8.5
    ax.add_patch(Polygon([(-w, y), (w, y), (w, y - 1.6), (0, y - 3.4), (-w, y - 1.6)], closed=True,
                         facecolor="#ffffff", edgecolor=INK2, lw=1))


def frame(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-8, 27)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ===========================================================================
# Option A - the shrink, drawn as shapes
# ===========================================================================
fig, ax = plt.subplots(figsize=(6.4, 7))
lost = (sm25 >= 0.5) & (sm26 < 0.5)
gained = (sm26 >= 0.5) & (sm25 < 0.5)
ax.contourf(xs, Z_IN, lost.astype(float), levels=[0.5, 1.5], colors=[C25], alpha=0.85)
ax.contourf(xs, Z_IN, gained.astype(float), levels=[0.5, 1.5], colors=[C26], alpha=0.85)
ax.add_patch(Rectangle(BOX[:2], BOX[2], BOX[3], fill=False, edgecolor=INK, lw=1.6))
ax.contour(xs, Z_IN, sm25, levels=[0.5], colors=[INK2], linewidths=1.2, linestyles="--")
ax.contour(xs, Z_IN, sm26, levels=[0.5], colors=[INK], linewidths=1.6)
plate(ax, -3.4)
frame(ax)
ax.set_title(f"Umpires gave up {A25 - A26:.0f} sq in of strike zone in 2026")
ax.text(-15, 25.6, f"Called zone: {A25:.0f} sq in in 2025  →  {A26:.0f} in 2026.  ABS zone: {ABS_AREA:.0f}.",
        fontsize=9.5, color=INK2)
# legend by direct labels
ax.annotate("Strikes in 2025,\nballs in 2026", xy=(6.5, -2.4), xytext=(9.6, -6.6), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("ABS zone", xy=(EX, 14), xytext=(11.2, 14), fontsize=9, color=INK, va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("2025 called zone", xy=(-6.5, 21.55), xytext=(-14.8, 23.3), fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("2026 called zone", xy=(4.5, 20.95), xytext=(5.2, 23.3), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
fig.text(0.04, 0.035, "Catcher's view; left-handed batters mirrored so the batter is always on the left.",
         fontsize=8, color=INK2)
ax.text(-13.6, 10, "Inside", fontsize=8.5, color=INK2, rotation=90, va="center")
ax.text(13.0, 10, "Away", fontsize=8.5, color=INK2, rotation=90, va="center")
for name in ("01_zone_change.png",):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# Option B - improved difference heatmap
# ===========================================================================
DIV = LinearSegmentedColormap.from_list("div", ["#b42c2c", "#f19c9b", "#f0efec", "#86b6ef", "#1c5cab"])
diff = 100 * (sm26 - sm25)
fig, ax = plt.subplots(figsize=(6.6, 7))
im = ax.pcolormesh(xs, Z_IN, diff, cmap=DIV, norm=TwoSlopeNorm(0, -40, 40), shading="gouraud", rasterized=True)
ax.add_patch(Rectangle(BOX[:2], BOX[2], BOX[3], fill=False, edgecolor=INK, lw=1.6))
ax.contour(xs, Z_IN, sm25, levels=[0.5], colors=[INK2], linewidths=1.1, linestyles="--")
ax.contour(xs, Z_IN, sm26, levels=[0.5], colors=[INK], linewidths=1.4)
plate(ax, -3.4)
frame(ax)
ax.set_title("Strikes disappeared just outside the zone in 2026")
cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, ticks=[-40, -20, 0, 20, 40])
cb.ax.set_yticklabels(["−40", "−20", "0", "+20", "+40"])
cb.set_label("Change in chance of a called strike (pct. points)", color=INK2)
cb.outline.set_visible(False)
ax.text(-15, 25.6, "Red = fewer strikes in 2026.  Dashed = 2025 called zone, solid = 2026, box = ABS zone.",
        fontsize=8.8, color=INK2)
fig.text(0.04, 0.035, "Catcher's view; left-handed batters mirrored.", fontsize=8, color=INK2)
fig.savefig(FIG / "extra" / "zone_change_heatmap_alt.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ===========================================================================
# Option C - small multiples: 2025, 2026 and the change
# ===========================================================================
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))
fig.subplots_adjust(left=0.02, right=0.9, wspace=0.05)
for ax, surf, title in [(axes[0], sm25, "2025: no challenges"), (axes[1], sm26, "2026: ABS challenges")]:
    ax.pcolormesh(xs, Z_IN, 100 * surf, cmap=SEQ, vmin=0, vmax=100, shading="gouraud", rasterized=True)
    ax.contour(xs, Z_IN, surf, levels=[0.5], colors=[INK], linewidths=1.3)
    ax.add_patch(Rectangle(BOX[:2], BOX[2], BOX[3], fill=False, edgecolor=INK2, lw=1.2, ls="--"))
    plate(ax, -3.4); frame(ax)
    a = A25 if surf is sm25 else A26
    ax.set_title(title, fontsize=11)
    ax.text(0, 25.6, f"called zone {a:.0f} sq in", ha="center", fontsize=9, color=INK2)
# the scale's midpoint is the page colour, so "no change" reads as background, not a gray box
DIV_PAGE = LinearSegmentedColormap.from_list("divp", ["#b42c2c", "#f19c9b", SURF, "#86b6ef", "#1c5cab"])
im = axes[2].pcolormesh(xs, Z_IN, diff, cmap=DIV_PAGE, norm=TwoSlopeNorm(0, -40, 40),
                        shading="gouraud", rasterized=True)
axes[2].add_patch(Rectangle(BOX[:2], BOX[2], BOX[3], fill=False, edgecolor=INK, lw=1.4))
plate(axes[2], -3.4); frame(axes[2])
axes[2].set_title("Change (red = fewer strikes)", fontsize=11)
axes[2].text(0, 25.6, f"{A26 - A25:+.0f} sq in", ha="center", fontsize=9, color=INK2)
cax = fig.add_axes([0.915, 0.3, 0.008, 0.4])
cb = fig.colorbar(im, cax=cax, ticks=[-40, -20, 0, 20, 40])
cb.ax.set_yticklabels(["−40", "−20", "0", "+20", "+40"], fontsize=8)
cb.set_label("Change in chance of\na called strike (pts)", color=INK2, fontsize=8.5)
cb.outline.set_visible(False)
fig.suptitle("The called zone shrank to fit the ABS zone", x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.02, 0.08, f"Blue shading = chance of a called strike. Black line = 50%. Dashed box = ABS zone ({ABS_AREA:.0f} sq in). "
         "Catcher's view; left-handed batters mirrored.", fontsize=8.5, color=INK2)
for name in ("02_zone_panels.png",):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
plt.close(fig)
print("areas", round(A25, 1), round(A26, 1), "ABS", round(ABS_AREA, 1))
