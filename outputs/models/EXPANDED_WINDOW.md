# Expanded-window results (2012–2025) — historical game lines

Added free nflverse closing lines (`nflv_game_lines`, 1999–2025, directional implied
team totals) and rebuilt the weekly + explosion models on `nflv_weekly` + game lines +
xFP over **2012–2025** — ~5× the training data of the 3-season Odds-API window.

## Weekly mean PPR — still saturated, but more data helped the floor
62,346 out-of-sample player-weeks (was ~14K).

| | MAE | Corr |
|---|---|---|
| BASE (rolling) | 4.489 | 0.660 |
| FULL (+xFP/luck/game env) | 4.491 | 0.660 |
| BASE, 2023–25 subset | 4.279 | — |
| FULL, 2023–25 subset | 4.274 | — |

- **FULL still gives ~0 lift over BASE on mean points** — the saturation finding holds, now on 5× the data (robust, not a small-sample artifact).
- But the **recent-years MAE improved from 4.37 → 4.28** vs the 3-season run: same model, more history = ~2% better. More data helps the level even when the new *features* don't.
- The new features ARE used heavily — `implied_team_total` is the #3 feature, with `roll_xfp`, `ix_itt_x_offpct`, `roll_luck` all prominent — they just don't reduce mean error (collinear with opportunity).

## Explosion / tail — the big win from more data
55,911 player-weeks, base rate 7.8%.

| Model | ROC AUC | Avg Precision | Precision @ top-10% | Lift |
|---|---|---|---|---|
| 3-season BASE (ref) | 0.655 | 0.132 | 0.159 | 2.05× |
| 3-season FULL (ref) | 0.662 | 0.139 | 0.172 | 2.22× |
| **Full-window BASE** | **0.714** | 0.166 | 0.203 | 2.63× |
| **Full-window FULL** | 0.713 | **0.171** | **0.207** | **2.68×** |

- **The expanded window is a large jump**: AUC 0.662 → 0.713, precision@top-10% 0.172 → 0.207 (now **2.68× the base rate**). More tail examples ⇒ a much better ceiling model.
- FULL still edges BASE on the **DFS-relevant tail metrics** (AP 0.171 vs 0.166, precision@k 0.207 vs 0.203); AUC is tied. On the 2023–25 subset FULL's AP is clearly higher (0.177 vs 0.168).
- Top features are the new ones: **`roll_luck`, `roll_xfp_std`, `roll_xfp`**, plus `ix_itt_x_opp` (game-environment interaction).

## Verdict on "pull odds back to 2011?"
**Helpful, not redundant — confirmed empirically.** The historical *game lines* (free via
nflverse) unlocked a 5× training window that materially improved the **tail/explosion**
model (AUC +0.05, precision@k 2.2× → 2.7×) and modestly lowered weekly mean error. The new
*features* remain ceiling-only (no mean lift), as the EDA predicted. Historical *player
props* would still be redundant (collinear with opportunity) and aren't freely available.

> Note: the **production** `predict_explosions.py` still runs on the 3-season window (it's
> tied to the archetype/game-script tables, which only exist for 2023–25, AUC 0.686). It
> could be migrated to the full window, but that's a larger refactor.

*Tables: `nflv_game_lines`, `weekly_v2_predictions`, `explosion_v2_predictions`.
Scripts: `fetch_schedule_lines.py`, `model_weekly_v2.py`, `model_explosion_v2.py`.*
