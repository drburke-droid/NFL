# Archetype / game-script expansion to 2012–2025

Extended the full archetype/game-script ecosystem from 3 seasons to the full window so
it can feed the expanded explosion model.

## Tables rebuilt (full window)
| Table | Before (rows) | After | Span |
|---|---|---|---|
| quarter_scores | 3,466 | 15,544 | 2012–2025 |
| game_scripts | 855 | 3,829 | 2012–2025 |
| team_archetypes | 192 | 896 | 2012–2025 |
| player_archetypes | 1,453 | 6,991 | 2011–2025 |
| game_script_tendencies | 16 | rebuilt | — |
| script_prediction_factors | 80 | 50 | 2012–2025 |

**How:** widened `SEASONS` in `fetch_quarter_scores`, `cluster_game_scripts`,
`cluster_team_archetypes` (all nflverse-sourced); repointed `cluster_archetypes` to
`nflv_weekly` (re-ingested with 5 extra columns); rebuilt `script_prediction_factors`
from the extended `game_scripts` + `nflv_game_lines` (no Odds/event_id dependency).

## Do archetype/game-script features help the explosion model?
Added team off/def archetype tiers + predicted game-script probabilities on top of the
full-window rolling+xFP+game-environment model (55,911 player-weeks, base rate 7.8%):

| Feature set | ROC AUC | Avg Precision | Lift @ top-10% |
|---|---|---|---|
| NOARCH (rolling + xFP + game env) | 0.7129 | 0.1707 | 2.68× |
| **ARCH (+ archetype tiers + game-script probs)** | **0.7183** | **0.1727** | 2.67× |

**Verdict: a small but real AUC bump (+0.005), negligible on precision@k.** The archetype/
game-script features rank low in importance (14th–19th of 20); the one that earns its keep
is the **game-script × production interaction** (`ix_gshigh_x_ppr`, top-10). The dominant
explosion signals remain `roll_luck`, `roll_xfp_std`, `roll_xfp`, and game environment.

## Takeaways
- The expansion is **done and usable** — archetypes and game scripts now exist for all of
  2012–2025 and are wired into the explosion model.
- **Data volume >> archetype sophistication**: the 3-season production model *with* the full
  archetype machinery scored AUC 0.686; the full-window model *without* archetypes scored
  0.713; adding archetypes on the full window nudges it to **0.718**. Most of the gain came
  from more data, not richer features.
- Best explosion model to date: **full-window + archetypes, AUC 0.718, 2.67× top-decile lift.**

> The production `predict_explosions.py` now benefits from the improved full-window
> `script_prediction_factors` and `team_archetypes`, but its *prediction* window is still
> 3 seasons (bound to `player_stats`/`game_odds`). The full-window `model_explosion_arch.py`
> is the stronger artifact; fully migrating production to the wide window remains optional.

*Scripts: `fetch_quarter_scores.py`, `cluster_game_scripts.py`, `cluster_team_archetypes.py`,
`cluster_archetypes.py` (widened), `scripts/build_script_factors_full.py`,
`scripts/model_explosion_arch.py`. Tables/metrics in `outputs/models/`.*
