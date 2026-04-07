# NFL DFS Prediction Pipeline — Project Summary

## Overview

This project builds a complete NFL Daily Fantasy Sports (DFS) prediction system from scratch, combining historical odds data, player statistics, archetype clustering, game script analysis, and machine learning models to predict player fantasy point output and identify "explosion" games for tournament play.

**Data coverage**: 3 NFL seasons (2023, 2024, 2025) — 855 regular season + playoff games, 56,979 player-game stat rows, ~24,000 player prop lines.

**Tech stack**: Python, SQLite, scikit-learn, LightGBM, nflverse (nflreadpy/nfl_data_py), The Odds API.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION                        │
│  fetch_odds.py → game odds, spreads, totals, props       │
│  fetch_player_stats.py → 57K player-game stat rows       │
│  fetch_quarter_scores.py → quarter-by-quarter scoring    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   CLUSTERING / ARCHETYPES                 │
│  cluster_archetypes.py → player archetypes (primary+style)│
│  cluster_team_archetypes.py → team OFF/DEF archetypes     │
│  cluster_game_scripts.py → 8 game script types            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    BACKTESTING                            │
│  backtest_archetypes.py → matchup edges with bootstrap CI │
│  predict_game_scripts.py → pre-game script prediction     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    PREDICTION MODELS                      │
│  predict_dfs.py → walk-forward PPR regression (LightGBM)  │
│  predict_explosions.py → boom game classifier (LightGBM)  │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Ingestion

### Player Stats (`fetch_player_stats.py`)
- Source: nflverse via `nflreadpy` (free, no API key, no rate limits)
- **56,979 player-game rows** across 2023–2025 (2,836 unique players)
- 53 stat columns per player per game: passing, rushing, receiving, EPA, fantasy points (standard + PPR)
- 100% match rate linking player stats to existing odds database via schedule-based game mapping
- Also backfilled final scores and week numbers into 855 games

### Quarter Scores (`fetch_quarter_scores.py`)
- Derived from nflverse play-by-play data (~49K plays per season)
- **3,466 quarter-score rows** (Q1–Q4 + OT for 46 overtime games)
- Per-quarter and cumulative scoring for both teams
- Scoring pattern: Q2 (14.0 avg) and Q4 (12.9 avg) are highest-scoring quarters; home advantage strongest in Q2

### Existing Data (from prior work)
- `game_odds`: spreads, totals, moneylines from DraftKings and other books
- `player_props`: 16 prop markets (pass yards, rush yards, receptions, TDs, etc.)
- Coverage: ~24,000 prop entries across 3 seasons

---

## Step 2: Player Archetype Clustering (`cluster_archetypes.py`)

### Method
- K-Means clustering on per-game stat averages per player-season
- Minimum 4 games to qualify
- **Two-part label**: primary archetype (K-Means cluster by volume/role) + style sub-category (percentile-based trait within position group)

### Results: 1,453 player-season archetypes

**QB (3 primary × 6 styles)**:
| Primary | Avg PPR/gm | Example Styles |
|---|---|---|
| Dual-Threat | 17.0 | Scrambler, Efficient, Deep Ball |
| High-Volume Passer | 13.3 | Gunslinger, Deep Ball, Short/Timing |
| Game Manager | 3.3 | Short/Timing, Balanced |

**RB (4 primary × 5 styles)**:
| Primary | Avg PPR/gm | Example Styles |
|---|---|---|
| Workhorse | 13.7 | Pass-Catching, Goal-Line, Power, Explosive |
| Starter | 9.5 | Goal-Line, Pass-Catching, Power |
| Rotational | 3.8 | Explosive, Balanced |
| Depth | 0.6 | Balanced |

**WR (4 primary × 6 styles)**:
| Primary | Avg PPR/gm | Example Styles |
|---|---|---|
| Alpha WR1 | 14.5 | YAC Monster, Red Zone, Deep Threat, Gadget |
| WR2 | 8.5 | Gadget, Deep Threat, YAC Monster |
| WR3/Flex | 6.9 | Red Zone, Deep Threat, Field Stretcher |
| Depth | 2.0 | Balanced, Gadget, Field Stretcher |

**TE (3 primary × 5 styles)**:
| Primary | Avg PPR/gm | Example Styles |
|---|---|---|
| Elite Receiving | 9.7 | High-Volume, YAC, Seam Threat, Red Zone |
| Secondary Receiver | 6.4 | YAC, Red Zone, Seam Threat |
| Blocking/Depth | 2.4 | Balanced, Red Zone |

### Validation Examples
- McCaffrey: `Workhorse / Pass-Catching` (2025), `Workhorse / Goal-Line` (2023)
- Ja'Marr Chase: `Alpha WR1 / YAC Monster` → `Alpha WR1 / Red Zone` (tracks his TD spike year)
- Josh Allen: `Dual-Threat / Scrambler` — consistent across 3 seasons
- Deebo Samuel: `WR2 / Gadget` — captures his rushing usage

---

## Step 3: Team Archetype Clustering (`cluster_team_archetypes.py`)

### Method
- Team offense: aggregated from nflverse team-level game stats (passing, rushing, EPA, tempo)
- Team defense: derived from opponent offensive production + own defensive stats (sacks, INTs, TFL, pressure rate)
- K-Means with 4 clusters per unit, style assigned via percentile thresholds

### Results: 192 team-season archetypes (32 teams × 3 seasons × 2 units)

**Offense archetypes**: Elite, Pass-First, Run-Heavy, Bottom-Tier
- Styles: Explosive, Power Run, Uptempo, Air Raid, West Coast, Ball Control, Zone Run, Turnover-Prone

**Defense archetypes**: Elite, Above-Average, Middle-of-Pack, Bottom-Tier
- Styles: Pass Rush Dom., Blitz Heavy, Lockdown Secondary, Ball Hawk, Run Stuffing, Bend Don't Break, Takeaway Machine

### Validation Examples
- BAL: `Elite / Power Run` offense + `Elite / Pass Rush Dom.` defense — Lamar's identity
- DET: `Elite / Uptempo` → `Elite / Explosive` → `Elite / RPO/Balanced` — offensive evolution under Dan Campbell
- MIA: `Elite / Explosive` (2023 Tyreek peak) declining to `Run-Heavy / Turnover-Prone` (2025)
- DAL defense: `Elite / Lockdown Secondary` (2023) → `Bottom-Tier / Blitz Heavy` (2025)

---

## Step 4: Game Script Classification (`cluster_game_scripts.py`)

### Method
- K-Means on quarter-by-quarter scoring trajectories (14 features: total points, margin, halftime deficit, lead changes, winner/loser 1H/2H scoring, etc.)
- Validated against 105,278 actual plays from play-by-play data

### Results: 8 Game Script Archetypes

| Script | Games | Avg Pts | Margin | Key Characteristic |
|---|---|---|---|---|
| Shootout | 59 | 60.8 | 4.0 | High-scoring, close, lead changes |
| High-Scoring Pulled Away | 86 | 66.3 | 11.4 | Big scoring, one team separates |
| Wire-to-Wire Blowout | 98 | 47.0 | 24.6 | Dominant start to finish |
| 2nd Half Blowout | 124 | 44.3 | 21.4 | Close at half, then dominance |
| Comeback/Competitive | 154 | 45.9 | 4.9 | Trailing team rallies |
| Steady Build | 134 | 42.9 | 6.8 | Gradual separation |
| Low-Scoring Seesaw | 95 | 36.6 | 5.2 | Tight, low-scoring, lead swaps |
| Defensive Slugfest | 105 | 26.5 | 9.4 | Very low scoring |

### Play-Calling Tendency Validation (2nd Half Shifts)

| Script | Winner Δ Pass Rate | Loser Δ Pass Rate | Loser Δ Hurry-Up |
|---|---|---|---|
| Wire-to-Wire Blowout | **-14.8%** | +6.0% | **+12.1%** |
| 2nd Half Blowout | **-13.1%** | +8.1% | **+11.5%** |
| Shootout | -3.4% (keeps passing) | +6.4% | +3.2% |
| Defensive Slugfest | -11.5% | +7.9% | +8.7% |

**DFS implications confirmed**: Blowout winners go run-heavy (fade their WRs late), losers spike pass rate and hurry-up (boost their pass-catchers). Shootouts keep both offenses passing.

---

## Step 5: Archetype Matchup Backtesting (`backtest_archetypes.py`)

### Method
- 17,928 player-game matchups tested across 3 analysis layers
- 5,000-iteration bootstrap confidence intervals at 90% level
- Three layers: individual archetype vs defense, granular style vs style, and same-team combo stacks

### Layer 1: Individual Archetype vs Defense (statistically significant edges)

**Smash spots**:
| Matchup | Sample | PPR Delta | % vs Baseline |
|---|---|---|---|
| Workhorse RB vs Bottom-Tier D | 110 | +2.95 | **+18.0%** |
| Dual-Threat QB vs Bottom-Tier D | 80 | +2.44 | +13.5% |
| High-Volume Passer vs Bottom-Tier D | 92 | +2.23 | +15.7% |

**Fade spots**:
| Matchup | Sample | PPR Delta | % vs Baseline |
|---|---|---|---|
| Game Manager QB vs Elite D | 62 | -1.24 | **-34.6%** |
| WR2 vs Elite D | 121 | -1.59 | -18.7% |
| High-Volume Passer vs Elite D | 255 | -1.54 | -10.9% |

### Layer 2: Granular Style Matchups (89 significant out of 519 tested)

Top exploits:
- **Workhorse/Pass-Catching RB vs Pass Rush Dom. D**: +6.50 PPR — defenses selling out for sacks get burned by checkdowns
- **Dual-Threat/Efficient QB vs Blitz Heavy D**: +4.89 PPR
- **Dual-Threat/Scrambler QB vs Run Stuffing D**: +4.32 PPR — scramblers feast when box is stacked
- **Alpha WR1/Deep Threat vs Blitz Heavy D**: +3.08 PPR — blitzing leaves 1-on-1 deep

### Layer 3: Combo Stacks (102 significant out of 272 tested)

Top stacks vs Bottom-Tier defense:
- QB (High-Vol Passer) + RB (Workhorse): **+6.32 combined PPR**
- QB (Dual-Threat) + WR (Alpha WR1): **+3.67 combined PPR**
- QB (Dual-Threat) + TE (Elite Receiving): **+3.56 combined PPR**

---

## Step 6: Game Script Prediction (`predict_game_scripts.py`)

### Method
- Random Forest + Gradient Boosting classification
- Features: spread, over/under, implied win probability, team archetypes, previous game scripts

### Key Findings

**Over/Under is the strongest predictor**:
| O/U Range | P(Shootout) | P(High-Scoring) | P(Def Slugfest) |
|---|---|---|---|
| High (50+) | 11.9% | 25.4% | 8.5% |
| Low (≤38) | 5.6% | 3.7% | 27.8% |

**Spread × O/U combined ("money buckets")**:
| Scenario | Dominant Script | Probability |
|---|---|---|
| Close + High Total | Shootout or High-Scoring | 34.5% combined |
| Big Fav + Med Total | Any blowout | 44.3% combined |
| Close + Low Total | Defensive grind | ~30% |

**Archetype matchups that skew scripts**:
- Elite OFF vs Bottom-Tier DEF: 30% chance of 2nd Half Blowout, 0% Defensive Slugfest
- Bottom-Tier OFF vs Above-Avg DEF: 29% Defensive Slugfest

---

## Step 7: Walk-Forward DFS Point Prediction (`predict_dfs.py`)

### Method
- LightGBM regression, walk-forward (train on all weeks before current, predict current)
- 75 features including:
  - Exponentially weighted rolling player stats (α=0.65, 6-game window)
  - Rolling team offense/defense stats
  - Dynamic archetypes (re-assigned each week from rolling stats via nearest cluster center)
  - Betting lines (spread, O/U) and player prop lines
  - Opponent defense quality (rolling)
  - Interaction features: player stats × defense weakness, props × defense, O/U × player tier, archetype matchup encodings
  - Predicted game script probabilities

### Results: 15,887 walk-forward predictions

| Metric | Model | Baseline (Rolling Avg) | Improvement |
|---|---|---|---|
| **MAE** | **4.55 PPR** | 4.94 PPR | **+8.0%** |
| **Correlation** | **0.6404** | — | — |
| **RMSE** | **6.20** | — | — |

**By position**:
| Position | MAE | Correlation | vs Baseline |
|---|---|---|---|
| QB | 6.19 | 0.514 | +9.7% |
| RB | 4.47 | 0.666 | +7.1% |
| WR | 4.59 | 0.586 | +7.6% |
| TE | 3.70 | 0.563 | +8.7% |

**By player tier (where it matters most for DFS)**:
| Tier | Model MAE | Baseline MAE | Improvement |
|---|---|---|---|
| Low (0-5 PPR) | 2.96 | 2.87 | -3.2% |
| Mid (5-10) | 4.90 | 5.30 | +7.6% |
| Great (15-20) | 6.56 | 7.42 | **+11.5%** |
| **Elite (20+)** | **6.90** | **9.61** | **+28.2%** |

The model adds the most value for high-ceiling players — exactly where DFS edges matter.

**Improvement over time** (more training data = better):
- 2023: +5.3%
- 2024: +8.0%
- 2025: +9.9%

### Top Features (with interaction features)

| Rank | Feature | Type | Importance |
|---|---|---|---|
| 1 | prop_rec_yds | Prop line | 273 |
| 2 | roll_ppr_min | Rolling stat | 231 |
| 3 | roll_ppr_max | Rolling stat | 217 |
| 6 | **ix_roll_ppr_x_defweak** | **Interaction** | **199** |
| 8 | **ix_prop_rec_yds_x_defweak** | **Interaction** | **194** |
| 10 | **ix_rush_x_opp_rush** | **Interaction** | **156** |
| 13 | **ix_ou_x_tier** | **Interaction** | **151** |

Interaction features account for **20.8% of total model importance** — a massive lift from raw archetype encodings which were negligible. The model learned that crossing player profiles with matchup context (e.g., "prop line × defense weakness") captures the same edges the backtesting identified, but in continuous form.

---

## Step 8: Explosion Prediction (`predict_explosions.py`)

### Definition
An "explosion" = player scores ≥ 2 standard deviations above their rolling average. Base rate: ~8.3% of player-games among DFS-relevant players (rolling PPR ≥ 3).

### Method
- LightGBM binary classifier, walk-forward, class-balanced
- 78 features including all DFS model features plus:
  - Coefficient of variation (boom/bust profile)
  - Previous explosion frequency and recency
  - Recent trend (last 2 games vs last 6)
  - Predicted game script probabilities × player profile interactions

### Results: 11,134 walk-forward predictions

| Metric | Value |
|---|---|
| **ROC AUC** | **0.6793** |
| **Average Precision** | 0.1515 (1.83x lift over base rate) |

**Threshold analysis for DFS strategies**:
| Threshold | Precision | Recall | Flagged | Hit Rate vs Base |
|---|---|---|---|---|
| 10% | 12.7% | 72.6% | 5,275 | 1.5x |
| 25% | 14.8% | 44.5% | 2,777 | 1.8x |
| 30% | 16.1% | 39.0% | 2,238 | 1.9x |
| **50%+** | **19.4%** | — | 895 | **2.3x** |

**Calibration**:
| Model Says | Actually Explodes |
|---|---|
| 0-5% | 3.7% |
| 15-20% | 12.0% |
| 30-50% | 13.8% |
| **50%+** | **19.4%** |

**Who explodes (by archetype)?**
| Archetype | Explosion Rate | Why |
|---|---|---|
| Game Manager QB | 15.6% | Low baseline + high variance |
| Secondary Receiver TE | 11.1% | Inconsistent targets = boom/bust |
| Workhorse RB | 3.3% | Stable floor = rarely "booms" relatively |
| Alpha WR1 | 4.6% | High baseline makes 2σ jumps harder |

**Top features for explosion prediction**:
1. `roll_ppr_std` (#1 by far) — volatile players explode
2. `roll_opp_pass_epa` — opponent pass defense quality
3. `roll_ppr_trend` — recent momentum
4. `ix_highscoring_x_ppr` (interaction) — P(high-scoring game) × rolling PPR
5. `ix_gs_pass_delta_x_targets` (interaction) — expected 2H pass rate shift × target share

---

## Database Schema

The SQLite database (`nfl_odds.db`) contains these tables:

| Table | Rows | Description |
|---|---|---|
| games | 1,080 | Game metadata, scores, event IDs |
| game_odds | ~50K | Spreads, totals, moneylines by bookmaker |
| player_props | ~24K | Player prop lines (16 markets) |
| player_stats | 56,979 | Individual player game stats |
| quarter_scores | 3,466 | Per-quarter scoring per game |
| player_archetypes | 1,453 | Player primary archetype + style per season |
| team_archetypes | 192 | Team OFF/DEF archetype per season |
| game_scripts | 855 | Game script classification per game |
| game_script_tendencies | 16 | 2H play-calling shifts per script × role |
| script_prediction_factors | 80 | Script probability lookup by O/U and spread |
| backtest_individual | ~50 | Archetype vs defense edge results |
| backtest_granular | ~500 | Style vs style matchup edges |
| backtest_combos | ~270 | Same-team combo stack edges |
| dfs_predictions | 15,887 | Walk-forward PPR predictions |
| dfs_feature_importance | 75 | Feature rankings for PPR model |
| explosion_predictions | 11,134 | Walk-forward explosion probabilities |
| explosion_feature_importance | 78 | Feature rankings for explosion model |

---

## Key Limitations & Areas for Critique

1. **Archetype leakage**: Season-level archetypes use full-season data. The dynamic archetypes (rolling K-Means assignment) mitigate this but use cluster centers fit on all data. A stricter approach would refit centers only on prior data.

2. **Prop line dominance**: DraftKings prop lines are the #1 feature — the model may be partly learning to trust the market rather than finding independent edges. Prop lines embed professional projections.

3. **Explosion calibration gap**: At high confidence (30-50%), the model predicts ~39% but actual rate is ~14%. The model is overconfident in the upper range, though directionally correct.

4. **Sample size**: 3 seasons (855 games) is modest. Some granular matchup edges (style × style) have n=10-15, making them fragile despite bootstrap CIs.

5. **No injury/weather/rest features**: The model doesn't account for injuries, weather, short weeks, or roster changes — all of which impact DFS outcomes.

6. **Walk-forward start**: Predictions begin at week 5 each season. Early-season predictions would need priors or preseason data.

7. **Low-usage player explosions**: The explosion model struggles with depth players who randomly go off (e.g., Jeremy McNichols 20 PPR from a 0.1 baseline). These are inherently unpredictable.

8. **No lineup optimization**: The pipeline predicts player-level outcomes but doesn't yet optimize full DFS lineups under salary constraints.

---

## Files

| File | Purpose |
|---|---|
| `fetch_odds.py` | Pull game odds from The Odds API (existing) |
| `fetch_player_stats.py` | Pull player stats from nflverse |
| `fetch_quarter_scores.py` | Derive quarter scoring from PBP |
| `cluster_archetypes.py` | Player archetype clustering |
| `cluster_team_archetypes.py` | Team OFF/DEF archetype clustering |
| `cluster_game_scripts.py` | Game script classification + PBP validation |
| `backtest_archetypes.py` | Matchup edge backtesting |
| `predict_game_scripts.py` | Game script prediction from pre-game features |
| `predict_dfs.py` | Walk-forward PPR prediction model |
| `predict_explosions.py` | Walk-forward explosion classifier |
| `nfl_odds.db` | SQLite database with all tables |
