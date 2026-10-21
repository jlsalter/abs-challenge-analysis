# Data

All data come from [Baseball Savant](https://baseballsavant.mlb.com) (MLB Advanced Media): Statcast pitch data for the 2025 and 2026 regular seasons, and pitch-level ABS challenge records from Savant's per-game feeds. This project is not affiliated with MLB.

`processed/` holds the analysis-ready datasets used by every model and figure. The raw pulls (~500 MB) aren't committed. `src/fetch_savant_direct.py` and `src/fetch_challenges.py` rebuild them into `data/raw_daily/` and `data/raw_challenges/`, and `src/build_dataset.py` turns the raw pulls into the processed data.

## `processed/called_pitches_2025_2026.csv.gz`
740,228 called balls and strikes from 2025 and 2026, one row per pitch.

| Column | Meaning |
|---|---|
| `game_pk`, `at_bat_number`, `pitch_number` | Unique pitch key (Statcast) |
| `game_date`, `game_year`, `inning`, `inning_topbot` | When it happened |
| `batter`, `pitcher`, `catcher` | MLBAM player ids (`player_names.csv` maps the batters and catchers in the challenge leaderboards) |
| `stand`, `p_throws`, `pitch_type`, `release_speed` | Batter side, pitcher hand, pitch type, velocity (mph) |
| `balls`, `strikes`, `outs`, `base_state` | Count and game state before the pitch. `base_state` = 1·(runner on 1st) + 2·(2nd) + 4·(3rd) |
| `bat_score`, `fld_score`, `bat_team`, `fld_team` | Score and teams, batting-team perspective |
| `plate_x_mid_ft`, `plate_z_mid_ft` | Location at the **middle** of the plate (ft), the ABS reference point. 2025 pitches were moved from Statcast's front-of-plate measurement along their tracked trajectory |
| `abs_zone_top_ft`, `abs_zone_bot_ft` | The batter's ABS zone (his median 2026 zone), applied to both seasons. Empty for batters who didn't play in 2026 |
| `edge_distance_in` | Signed distance from the ABS zone edge, counting the ball's radius (inches; negative = inside) |
| `abs_strike` | 1 if the pitch is a strike under ABS |
| `ump_strike` | The umpire's **original** call (1 = strike). 2026 overturns are flipped back |
| `final_strike` | The call after any challenge |
| `challenged`, `overturned`, `challenged_by_batter` | Challenge flags (2026) |
| `runs_at_stake` | Run expectancy after a ball minus after a strike (batting team's view) |
| `bat_chal_left`, `fld_chal_left` | Challenges each side held before the pitch (2026) |

## `processed/challenges_2026.csv.gz`
The 10,546 challenges matched to a called pitch. Same columns as above.

## `processed/challenge_opportunities_2026.csv.gz`
341,624 challenge opportunities: called pitches in innings 1–9 that went against a side still holding a challenge. Each row carries the decision model's estimates.

| Column | Meaning |
|---|---|
| `role`, `player` | Who could challenge (`batter` or `catcher`) and his MLBAM id |
| `distance_in_challengers_favor_in` | Distance from the zone edge in the challenger's favor (> 0 means the call was wrong) |
| `run_stake`, `win_stake` | Value of reversing the call, in runs and in win probability |
| `option_value_optimal`, `option_value_actual_policy` | Value of the challenge that would be spent, from the simulation under the optimal policy and under teams' actual policy |
| `breakeven_confidence` | Minimum P(call wrong) at which a challenge pays: OV / (stake + OV) |
| `threshold_actual_in`, `threshold_optimal_in` | Perceived distance at which the role actually challenges, and at which it should |
| `p_challenge_avg_player`, `p_challenge_optimal` | Probability that an average player in the role challenges, and that an optimal decision-maker with the same eyesight would |
| `challenged`, `overturned` | What actually happened |
| `value_realized_wins`, `value_expected_avg_player_wins` | Win value of the actual decision, and the expected value for an average player |

## Other files
- `processed/run_expectancy_2026.csv`: runs expected for the rest of the inning from each of 288 states (outs × runners × count), 2026, innings 1–8.
- `processed/player_names.csv`: MLBAM id → name for batters and catchers on Savant's ABS leaderboards.
