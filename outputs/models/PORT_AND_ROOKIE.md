# Production tail-feature port · Rookie model

## 1. Ported tail features into production `predict_explosions.py` ✅
Added the EDA/v2-winning signals to the full 78-feature production explosion model and
fixed its DB path (it pointed at a non-existent root `nfl_odds.db`; now `db/nfl_odds.db`).

**New features:** `roll_xfp` (rolling expected PPR), **`roll_xfp_std`** (expected-points
volatility), **`roll_luck`** (rolling actual − expected), `roll_xfp_gap` (recent ceiling −
expectation), and directional **`implied_team_total`** / `team_spread`.

**Result (walk-forward, 11,134 predictions):**

| Metric | Before | After |
|---|---|---|
| ROC AUC | 0.679 | **0.6855** |
| Average Precision | 0.152 | **0.1537** |
| Precision @ 0.40 threshold | — | 19.0% (2.3× base) |

The lift is modest but real, and importantly the **new features rank among the most
important in the model** — `roll_luck`, `roll_xfp_std`, `roll_xfp_gap`, `roll_xfp` all
land in the top tier. This confirms the deep-dive thesis in the *production* model: the
expected-points signals matter for the **ceiling/tail**, where the mean-point model saw
nothing. Predictions written back to `explosion_predictions`.

---

## 2. Rookie season model ✅ — anchored beats rookie ADP by ~7%
Rookies have no prior NFL season, so inputs are pre-NFL only: **draft capital, combine
athleticism, age, and landing spot** (prior-year team offense + position-group
environment). 1,433 rookie skill-seasons, 2011–2025.

**Full model vs baselines (walk-forward, test 2015–25):**

| Model | MAE (PPG) | ρ |
|---|---|---|
| Position mean | 3.503 | 0.158 |
| Draft-pick only | 2.726 | 0.530 |
| **Full (draft + combine + landing)** | **2.699** | **0.536** |

**Honest read:** **draft capital is the overwhelming rookie signal** — pick alone gets you
most of the way (MAE 2.73). Combine + landing spot add only a hair (2.73 → 2.70). The most
important features are landing-spot/volume (`team_rush_att_prior`, `team_pos_ppg_prior`,
`team_pass_att_prior`), weight, then `draft_pick`, `forty`.

**ADP-anchored test (rookie ADP exists 2021–25; 324 rookies, test 2023–25):**

| Model | MAE (PPG) | ρ |
|---|---|---|
| Market (rookie ADP) | 3.196 | 0.551 |
| Model (no ADP) | 3.179 | 0.501 |
| **ADP-anchored** | **2.981** | **0.606** |

**The anchored model beats rookie ADP by 6.7% MAE (+0.055 ρ)** — a *wider* margin than the
veteran anchored model (1.8%). Rookie ADP is noisier than veteran ADP, so the structured
signals (draft slot, athleticism, landing spot) add more on top. Output: `rookie_predictions`.

---

## Bottom line
- The expected-points signals are now **in production** for explosions, where they
  measurably help (AUC 0.679 → 0.686, top-tier importance).
- Rookies: **draft capital is king**, but **ADP-anchored beats the rookie market by ~7%** —
  the anchored approach generalizes, and pays off *more* for rookies than veterans.
- The unified lesson across the whole project holds: **markets are sharp; the durable edge
  is anchoring to the market and adding structured deltas, plus modeling the tail.**

*Scripts: `predict_explosions.py` (updated), `scripts/model_rookie.py`.*
