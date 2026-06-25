# Injury / games-aware prior — fixing players coming off a shortened season

Problem (Burrow case): a top-3-when-healthy QB hurt in 2025 (8 games) was ranked
**QB29** — the projection over-anchored on the injury-shortened season for BOTH its
per-game rate and its games projection. Two validated fixes (both improve overall
accuracy, so they SHIP — not display tags).

## Fix 1 — injury/games-aware PPG features (`test_injury_prior.py`)
New leakage-free features on top of BASE: `prior2_games`, `gw_prior` (games-weighted
2-yr PPG), `healthy_prior` (max of last 2 yrs), `short_season`, `bounce`, `games_trend`.

| Central MAE | BASE | +INJ |
|---|---|---|
| QB | 4.001 | **3.922** |
| WR | 2.561 | 2.537 |
| RB / TE | 2.937 / 1.923 | 2.940 / 1.925 (flat) |
| ALL | 2.694 | **2.676** |

Injury-return subgroup (n=122): bias +0.95 → +0.69, MAE 3.48 → 3.43. Boom/bust AUC
+0.003/+0.001. Face validity — biggest corrections: Burrow '24, Dak '25, Rodgers '18,
Thielen '20, JuJu '22. Folded into `projection_overhaul` (production FEATS = BASE+INJ,
and BASE+INJ+FFA once 2026 consensus loads).

## Fix 2 — two-year games projection (`model_season.project_games`)
Games-played has low year-over-year persistence (r = 0.42), so the old 0.55×prior +
0.45×mean let one injury year tank the season total. New: **0.35×prior + 0.30×prior2 +
0.35×mean** when a second year is available.

| Next-games MAE | old (1-yr) | new (2-yr) |
|---|---|---|
| All | 3.646 | **3.585** |
| Injury-return | 4.447 | **4.267** |

Better on both. Wired through `build_2026_targets` (board) and `projection_overhaul`.

## Effect on Burrow
| | before | after |
|---|---|---|
| proj PPG | 16.4 | 17.1 |
| proj games | 8.6 | 11.1 |
| season total | 188 | 245 |
| **position rank** | **QB29** | **QB17** |
| VORP | −77 | −20.5 |

From undraftable to a draftable QB. The remaining gap to "top-3" is by design: a
prior-anchored model can't fully rate a player whose only recent full-season sample is
injury-wrecked — the **FFA consensus** (auto-loaded when the 2026 file lands) supplies
that market correction and will push him toward elite. The 📈 Late Risers tab also flags
his 9.9→25.6 second-half surge.
