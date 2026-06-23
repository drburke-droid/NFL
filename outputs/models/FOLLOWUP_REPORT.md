# Follow-up Report — 2026 projection · explosion/tail model · ADP benchmark

Three follow-ups to the deep dive. Scripts: `project_2026.py`, `model_explosion_v2.py`,
`fetch_adp.py`, `benchmark_adp.py`.

---

## 1. Forward 2026 season projection ✅
Built 2026 feature rows from each player's 2025 production aged forward, trained the
season model on all 2012–2025 transitions, projected PPG → finish → VBD for 532 players.
Output: `season_proj_2026` table + `outputs/models/season_proj_2026.csv`.

**Top projected (by VBD):** Ja'Marr Chase, Jaxon Smith-Njigba, Puka Nacua, Amon-Ra
St. Brown (WR); Bijan Robinson, CMC, Jahmyr Gibbs, Jonathan Taylor (RB); Trey McBride
(TE1); Josh Allen (QB1). These pass the smell test.

**Caveat:** finish/VBD include a **projected-games (durability) term**, so a healthy
mid-PPG player can out-*finish* a higher-PPG player who missed time (e.g. Burrow's
injury-shortened 2025 lowers his projected games and thus his finish, despite a high
per-game projection). It also can't see 2026 offseason moves (team set to 2025 team,
`team_change=0`). **This is a prior-year-only projection; see the ADP benchmark for why
that matters.**

---

## 2. Weekly explosion / tail model ✅ — new signals DO help here
Same walk-forward window (2023–25), but the target is the **GPP ceiling**: explosion =
PPR ≥ rolling-mean + 2·rolling-std (DFS-relevant players). This is where the mean-point
model (v2) showed no lift — so it's the real test of the new data.

| Feature set | ROC AUC | Avg Precision | Precision @ top-10% | Lift vs base |
|---|---|---|---|---|
| BASE (rolling only) | 0.655 | 0.132 | 0.159 | 2.05× |
| **FULL (+ game env + xFP var)** | **0.662** | **0.139** | **0.172** | **2.22×** |

**The new features add tail signal** (+8% relative precision among the top decile of
flagged players) even though they added nothing to mean-point MAE. The top explosion
features are exactly the new ones: **`roll_xfp_std` (expected-points volatility),
`roll_luck`, `roll_xfp`**, then rolling std/CV, **team spread, game total**.

→ Confirms the deep-dive thesis: **the new data's weekly payoff is in the ceiling, not
the mean.** Folding `roll_xfp_std`, `roll_luck`, and game-environment features into the
existing `predict_explosions.py` (78-feature model, AUC 0.679) should push it further.
Output: `explosion_v2_predictions` table + importances.

---

## 3. Historical ADP benchmark ✅ — the market is the bar, and it's high
Sourced **preseason consensus ADP/ECR for 2021–2025** (last-August FantasyPros redraft
rankings via `load_ff_rankings('all')`, 93.8% mapped to gsis_id → `nflv_adp`). Compared
on 1,737 veteran player-seasons that have **both** an ADP and a model projection.

**Ranking players vs their actual season finish (mean within-position Spearman):**

| Ranker | vs actual finish | vs actual PPG |
|---|---|---|
| **Market (ADP/ECR)** | **0.729** | **0.755** |
| Season model | 0.677 | 0.711 |
| Model + Market (naive ensemble) | 0.727 | — |

**The market beats the model at every position** (QB 0.73 vs 0.65, RB 0.74 vs 0.67,
WR 0.74 vs 0.70, TE 0.71 vs 0.69), and a naive rank-average ensemble matches but does
not beat ADP.

**Why this is the expected — and useful — result:** ADP prices in everything the
prior-year model is blind to: offseason signings/trades, training-camp news, injuries,
coaching/scheme changes, and rookies. A model built only on last year's box score
*should not* beat it.

**So what is the model for?** Not as a standalone market-beater, but as: (a) a
**complementary signal** that encodes specific, explainable edges (TD regression, role
stickiness, aging) the market may under/over-weight; (b) the engine behind the
**mechanistic findings**; (c) a base for a smarter blend — using **ADP as the anchor and
the model's regression/role deltas as adjustments** (the naive ensemble was too blunt;
a learned ADP-anchored model is the next step). The largest model-vs-market disagreements
were mostly low-end noise, not a clean exploitable edge.

---

## 4. ADP-anchored season model ✅ — beats the market (modestly)
The benchmark said ADP beats the prior-year model. So the real product uses **ADP as
the anchor and the model's prior-year regression/role signals as deltas on top.** Tested
three models on the same 1,737 veteran player-seasons, walk-forward over ADP years
(test 2023–25):

| Model | MAE (PPG) | ρ(PPG) | rank-vs-finish |
|---|---|---|---|
| Market (ECR + pos rank) | 2.693 | 0.762 | 0.706 |
| Prior-year only | 2.829 | 0.730 | 0.645 |
| **ADP-anchored (market + deltas)** | **2.644** | **0.778** | **0.717** |

**The anchored model beats the market on every metric** — MAE −1.8%, ρ +0.016, finish
rank +0.011. The edge is **concentrated where ADP is noisiest: TE (−8.5% MAE) and WR
(−1.7%)**; QB/RB are essentially flat (the market is already highly efficient there).
Top-N hit rate is mixed (QB better, RB/WR slightly worse) — the gain is in overall
calibration, not in nailing the exact chalk tier.

**What the model leans on (anchored feature importance):** `prior_cv`,
`prior_receiving_epa`, `prior_ppr_std`, **`ecr`**, `prior_ppg`, `prior_off_pct`,
weight/athleticism, `prior2_ppg`, **`pos_rank`**, `draft_pick`. It genuinely blends the
market signal with regression/role deltas.

**Honest read:** the win is **small** (~1.8% MAE over a very strong market baseline) and
rests on 3 test years — markets are efficient and a hair is the realistic ceiling. But it
is a *consistent* hair in the right direction, and it's the first config to beat ADP. The
biggest practical use: **TE and WR**, where the model meaningfully sharpens noisy ADP.
Output: `season_adp_predictions` table + `season_adp_importance.csv`.

> Note: a true 2026 ADP-anchored projection needs **2026 preseason ADP**, which isn't
> published yet (the stored `nflv_ff_rankings_current` is a Dec-2025 snapshot, not draft
> ADP). Rerun once August 2026 ADP is available.

## Bottom line across all follow-ups
- **2026 projections** are delivered and reasonable (prior-year-only).
- **Markets are sharp** both weekly (mean points) and seasonally (ADP) — beating them on
  central tendency is hard.
- **The edge lives in the tail and in specific regression signals**, not in mean
  accuracy: the new xFP-variance / luck / game-environment features measurably improve
  **explosion** detection, and the season model's value is as an **ADP-anchored
  adjustment layer**, not a replacement for the market.

**Update:** the ADP-anchored model (§4) now exists and **beats the market by ~1.8% MAE**,
concentrated at TE/WR.

**Recommended next:** (1) productionize the anchored model and rerun once **2026 preseason
ADP** publishes (also unlocks a market-grade 2026 projection); (2) port the winning tail
features (`roll_xfp_std`, `roll_luck`, game environment) into `predict_explosions.py`;
(3) extend the anchored approach to rookies (no prior season → draft capital + combine +
landing spot as the delta layer over rookie ADP).
