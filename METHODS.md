# Methods and decisions

This document records each modeling decision: the choice, why it was made, and the alternatives considered. Sections 1–9 cover the data construction and the first-pass models. Sections 10–16 cover the final challenge-decision models used in the README.

## 1. Putting 2025 and 2026 on the same scale
**Problem.** In 2026 Savant changed where `plate_z` is measured, from the front of the plate (y = 17 in) to the middle (y = 8.5 in), because the ABS zone is judged at mid-plate. A sinking pitch is lower at mid-plate than at the front. Comparing raw `plate_z` across years would therefore make 2025 pitches look about an inch higher than they were.

**Choice.** Statcast stores each pitch's trajectory as a constant-acceleration fit (`vx0, vy0, vz0, ax, ay, az`, referenced at y = 50 ft). For each 2025 pitch I solve for the time the ball reaches y = 17/12 ft and y = 17/24 ft, then add the movement between those times to `plate_x` and `plate_z`. I confirmed the 2025 reference point by checking a 2025 game feed, where `plateYPoly = 1.4167` ft.

**Why not normalize each year to its own zone?** The zone definition changed too (see 2). Normalizing would mix the two changes together.

## 2. One ABS zone per batter, applied to both years
2026 `sz_top` / `sz_bot` are each batter's fixed ABS zone, at 53.5% and 27% of his height. 2025 values were per-pitch estimates based on stance. I take each batter's median 2026 zone and apply it in both seasons, so the target is identical across years. The cost is that 2025 batters who didn't play in 2026 drop out of Part 1.

**Caveat to state.** 2025 umpires weren't calling the ABS box, so "agreement" in 2025 means agreement with a standard they weren't using. The zone-size result doesn't depend on that framing.

## 3. Recovering the umpire's original call
Statcast's `description` in 2026 is the call after any challenge (an overturned called strike shows as `ball`). For the zone model the question is what the umpire called, so overturned pitches are flipped back. Challenges were joined on `game_pk` + at-bat number + pitch number (10,550 of 10,556 matched).

## 4. Zone geometry
A pitch is a strike if any part of the ball touches the zone at mid-plate: |x| ≤ 8.5 in + 1.45 in (ball radius), and z within [bottom − 1.45 in, top + 1.45 in]. Distance from the edge is signed (negative = inside). Check: this agrees with the official result on 99.7% of challenges, and the distances match Savant's `edge_distance` field.

## 5. Called-zone model and its uncertainty
- **Model:** gradient-boosted classifier of P(called strike) on (horizontal location toward/away from the batter, height as a fraction of the batter's zone), one model per season. It is flexible enough to follow the zone's rounded corners without hand-made spline choices. A GAM would be a reasonable, more interpretable alternative.
- **Summary number:** area where P(strike) ≥ 0.5, in square inches (height scaled by the median zone height, 19.1 in).
- **Uncertainty:** bootstrap that resamples *games*, not pitches, because pitches in one game share an umpire and aren't independent. 25 replicates, each subsampled to 250k pitches for speed. Resampling by umpire would be stricter; the interval is narrow enough that the conclusion wouldn't change.
- The difference map is smoothed (Gaussian, σ = 1.5 grid cells) for display only. The area numbers use the unsmoothed surfaces.

## 6. Framing
Strikes above expected = actual strike minus the pooled location model's prediction, summed per catcher (≥ 2,500 called pitches) and converted to runs at the average value of a ball/strike call. It is centered within each year, because 2026 had fewer strikes overall and only the spread across catchers matters. Location-only expected values are a simplification; public framing models also adjust for pitcher, batter, umpire and count.

## 7. Runs at stake
Run expectancy for each of 288 states (outs × runners × balls × strikes) is the mean runs scored for the rest of the inning, from 2026, innings 1–8. Innings 9+ are excluded because walk-offs cut innings short. A call's stake = RE(state after a ball) − RE(state after a strike). Strike three adds an out (a third out means RE = 0). Ball four forces runners and can score a run.

## 8. Option value of a challenge (first pass)
*Superseded by the simulated option value in section 13; kept because it is the starting point of that fixed-point iteration.*

For every team-game and every inning t = 1…9, record k = challenges left at the start of the inning and the runs the team went on to win from successful challenges in innings ≥ t. V(k, t) = the average of those future gains. Then:
- cost of losing your last challenge = V(1, t) − V(0, t), with V(0, t) = 0
- cost of losing one of two = V(2, t) − V(1, t)

Cells with fewer than 100 team-games are dropped. Nobody starts the 1st inning with one challenge, so that cell borrows the 2nd inning's value. Values are forced non-increasing over innings, since a challenge can't be worth more later.

**Known bias.** Teams with k = 1 got there by failing once, so they may be weaker challengers, which pushes V(1, t) down. A dynamic program built on the distribution of future challengeable pitches would avoid this and is the natural next step.

## 9. Decision rule and team scoring (first pass)
*The decision rule carries over; the run-based team scoring is superseded by sections 14–15.*

Challenge when p · stake > (1 − p) · OV, i.e. p > OV / (stake + OV). Team net runs = Σ stake over won challenges − Σ OV over lost challenges (at that inning and number of challenges left), reported relative to the league average.

"Missed" calls = calls that were wrong by ABS, at least 1.5 in off the edge (about one ball width), made while the team held a challenge, innings 1–9, and not challenged. This counts opportunities in hindsight; players see pitches with error, so not every missed call was visible to them.

---

# Part 2, extended (`src/part2_extended.py`)

## 10. Win-probability stakes
**Model.** One logistic curve per inning (extras pooled with the 9th) × half-inning. P(batting team wins) depends on the effective lead L = score margin + runs expected for the rest of the half-inning (outs × runners). It is fit on the first pitch of every 2025–26 plate appearance.

**Checks.** Brier 0.157; calibration within 1.5 points in every decile; correlation 0.996 with Savant's `bat_win_exp`.

**Why not use Savant's `bat_win_exp` directly?** It gives win probability before the pitch, but a challenge needs win probability in two counterfactual states: after a ball and after a strike. A simple, smooth model I control can be evaluated in both.

**Call value.** WP(state after a ball) − WP(state after a strike), with edge cases handled: a strikeout that ends the inning, a walk that forces a run, a walk-off, and a final out in the bottom of the 9th.

## 11. Perception model (with lapses)
**Setup.** Opportunity = a called pitch against a side that still holds a challenge (innings 1–9). d = signed distance from the zone edge in the challenger's favor (d > 0 means the call was wrong).

**Model.** P(challenge) = λ · logistic(b₀ + b_d·d + b₁·log run stake + b₂·log leverage + b₃·late + b₄·last challenge), fit by maximum likelihood separately for batters and catchers (|d| ≤ 6 in).

**Interpretation.** Under a threshold model (challenge when d + ε passes a threshold), the slope b_d is 1/scale of the perception error. Judgment error SD = (π/√3) / b_d. λ is the ceiling: the share of opportunities in which the player acts on what he sees.

**Why the ceiling.** Challenge rates level off at ~40% (batters) and ~70% (catchers) even on pitches 4–6 inches wrong. A plain logistic regression flattens its slope to fit that plateau, inflating judgment error (2.9 / 1.8 in without λ vs. 2.4 / 1.5 with it).

**Caveat.** Variation in thresholds *between* players also flattens the pooled curve, so σ is an "effective" error for a typical player, not a pure eyesight measure.

## 12. Optimal threshold
For a player with error σ, P(call wrong | perceived x) comes from Bayes' rule, using the role's distribution of true distances as the prior. The best threshold τ* solves P(wrong | τ*) = OV / (stake + OV). The actual threshold is where the fitted logistic part equals 0.5.

## 13. Simulated option value (fixed point)
**Why simulate.** Option value estimated from actual results reflects teams' actual, cautious behavior. A policy that challenges more runs out sooner and should value a saved challenge more. Pricing the optimal policy with the actual policy's option value is inconsistent.

**Method.** Replay every 2026 team-game (innings 1–9), 30 draws of perception noise per pitch, with common random numbers across policies:
1. Start with the empirical option value.
2. Simulate the optimal policy.
3. Re-estimate V(k, t), the expected future win probability gained when holding k challenges at the start of inning t.
4. Set OV = V(k, t) − V(k−1, t).
5. Repeat from step 2 until it moves less than 0.05 points (four rounds).

**Constraints.** OV is forced non-increasing over innings, and a second challenge can't be worth more than the last.

**Validation.** The actual-policy replay gives 2.14 challenges per team-game (real: 2.13) and 3.55 wins per team-season (real: 3.84).

**Policies compared.**
- Actual thresholds with actual lapses.
- Best thresholds with actual lapses. The difference from the actual policy isolates threshold placement.
- Best thresholds, always acting. This is an upper bound: some lapses are pitches the player genuinely couldn't judge.

## 14. Player value
For each opportunity: value if challenged = win stake when the call was wrong, −OV when it was right. OV here comes from the actual-policy simulation, because that is the value a challenge has for a team that behaves as teams do.
- Player value = Σ (actually challenged × value) − Σ (average player's challenge probability × value), on the same pitches.
- Average player's challenge probability = the fitted model for the player's role.
- 90% intervals bootstrap over a player's opportunities.
- Minimum opportunities: 150 for batters, 600 for catchers.

**Reliability.** Split-half (odd vs. even game_pk) correlation is 0.16 for batters and 0.14 for catchers. Treat individual rankings as noisy.

**Not modeled.** Extra innings (where teams with no challenges left get one per inning); pitcher-initiated challenges, which are attributed to the catcher (179 of 10,546); player-specific perception error.

## 15. Team ranking (`src/team_ranking.py`)
A team's value is the player measure from section 14, summed over its batters and catchers. Then the league average is subtracted. Players beat the role model by about 0.3 wins per team overall, and "above average" should mean above the average team. 90% intervals come from resampling games rather than pitches, since pitches in one game share a pitcher, an umpire and a score. "Room to improve" is the simulated wins under the best policy minus the actual policy, from section 13.

This replaces the run-based team scorecard from `analysis.py` (now `figures/extra/team_scorecard_runs.png`). That scorecard credited every won challenge with its full stake and valued lost challenges with the empirical option value, so it didn't compare each team with what an average team would have done on the same pitches.

## 16. Strike-zone figures (`src/fig_zone.py`)
The same called-strike models as section 5, evaluated on a 241 × 241 grid and smoothed lightly (Gaussian, σ = 2 cells) so the 50% lines are drawn without stair-steps. The areas in the title come from the unsmoothed surfaces on every other grid point, which is exactly the 121 × 121 grid behind the bootstrap interval, so the numbers match section 5.
