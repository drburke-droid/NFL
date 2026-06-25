# Implosion / Fade-risk — high picks who get drafted much later next year

The mirror of value-leap. Among players drafted HIGH (startable studs last year),
predict who collapses below startable this season — the ones whose ADP craters the
following year.

`scripts/test_implosion.py` (validation), `scripts/build_implosion.py` (2026 scores).

## Definition
- Pool: prior_ppg ≥ {QB 16, RB 12, WR 12, TE 8} (clearly startable / drafted high).
- Collapse: next_ppg < {QB 14, RB 9, WR 9, TE 7} (below startable) AND a drop ≥ 4 PPG.
- All four positions (a high QB collapse is also costly; no backup noise here since the pool is high-PPG only).

## Validated (walk-forward 2017-2025)
- Base collapse rate **13.3%** (~1 in 7 startable players craters).
- **AUC 0.668**, AP 0.277.
- **precision@top10% = 32.1% → 2.24× base** (top-20% 1.95×, top-30% 1.72×) — ~1 in 3 of the top decile implode.
- **Beats naive baselines decisively**: model 0.668 vs age 0.548, prior_cv 0.566, prior_total_tds 0.407 — it is NOT just "fade the old/volatile."
- Face validity: Mike Williams '24 (16.7→4.1), Kenny Golladay '21, Le'Veon Bell '20, Jamaal Williams '23, Devonta Freeman '20, O.J. Howard '19.
- **Top drivers**: `prior2_ppg` (one-year-wonder — low two years ago, spiked, unsustainable), fading late-season usage (`ht_d_tch`/`ht_d_snap`), volatility (`prior_ppr_std`/`prior_cv`), draft capital, incoming competition.

## Productized
Calibrated (Platt) `nflv_implosion` scores 98 high-drafted 2026 players;
`build_draft_tool` carries `fade_prob` into data.js; web app has a **💣 Fade Risk**
tab ranking them with a "why" (age-fragile archetype / faded late / competition).
`refresh_2026` runs value-leap → implosion before the board.

2026 top fade risk: **Tyreek Hill 47%** (aging deep-threat — consistent with the
archetype age-risk finding), then RBs facing competition / one-year profiles
(Etienne, Bucky Irving, Gainwell) and thin-profile TEs. Complements the calibrated
bust% by adding the late-season-fade + incoming-competition signals bust% lacks.
Use it to avoid overpaying, not as a hard "don't draft."
