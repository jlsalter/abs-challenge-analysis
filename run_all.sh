#!/usr/bin/env bash
# Rebuild everything from scratch: raw pulls -> processed data -> models -> figures.
set -euo pipefail
cd "$(dirname "$0")"
python src/fetch_savant_direct.py     # 2025-26 regular-season pitches from Baseball Savant (~30 min, cached)
python src/fetch_challenges.py        # every 2026 ABS challenge from Savant's game feeds (~20 min, cached)
python src/build_dataset.py           # mid-plate locations, ABS zones, umpire calls, run expectancy
python src/analysis.py                # Part 1: called-zone model, bootstrap, framing; runs at stake
python src/part2_extended.py          # Part 2: win probability, perception model, simulation, players
python src/team_ranking.py            # team challenge value
python src/fig_zone.py                # strike-zone figures
python src/export_processed.py        # data/processed/*.csv.gz
