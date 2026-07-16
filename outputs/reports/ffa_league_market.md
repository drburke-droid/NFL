# League-scored FFA (PPR + 6-pt pass TD) — market benchmark & model anchor

The league-scored FFA exports use our league's actual scoring: full PPR, 6-pt pass TD,
−1 INT. Vs the standard-scored files: QBs +76 pts avg, WRs +69, TEs +59, RBs +33.

## How it's integrated

`scripts/fetch_ffa_league.py` → **`nflv_ffa_league`**: full history 2014–2026
(2014–15 sparse stubs, as in the standard archive), 95% gsis-matched.

Originally ingested as 2026-only (mixing one league-scaled season into a
standard-scaled history would have saturated the trees), but once the user
regenerated the FULL history in league scoring, the head-to-head backtest showed the
league-scored anchor is simply better (MAE 2.848 vs 2.876; QB rank-finish 0.431→0.471
— see `FFA_VALUE.md` addendum). **`attach_ffa` now anchors the production model on
`nflv_ffa_league`.** The model still predicts in PPR/4-pt space and the board layer
converts QBs to 6-pt league scoring (`build_draft_tool.py` / `report_2026_targets.py`:
`+2*pass_td + 1*int`), so no double-counting.

So `draft_board_2026.proj_pts` and `nflv_ffa_league.ffa_points` are directly comparable
— both on true league scale.

## Board vs league-scored market (220 matched players)

Rank agreement (correlation of projected points):

| Pos | corr | mean offset (ours − FFA) |
|---|---|---|
| RB | 0.94 | −36 |
| TE | 0.90 | −34 |
| WR | 0.89 | −45 |
| QB | **0.60** | **−102** |

- The **systematic negative offset is mostly games**: we project 13–15 games
  (durability-shrunk), FFA assumes ~17. Rankings are unaffected.
- **QB is the real disagreement zone** (corr 0.60). Two drivers:
  1. FFA carries 2026 **offseason depth-chart signal** our prior-year features can't
     see: Jaxson Dart 376 (we say 230), Tyler Shough 340 (we say 173), **Malik Willis
     on MIA at 301** (we say 91 — and we're *higher* than market on Tua, 199 vs 173,
     which the market clearly reads as a lost job).
  2. Historically the expert consensus is **hardest to beat at QB** (FFA_VALUE.md:
     our blend's smallest edge, +3.4%). Take the market seriously here.
- Biggest non-QB gap: **Cam Skattebo** — market 246 (RB ~top-15), us 107.
- We're notably above market on: Tank Bigsby (+35), Troy Franklin (+31), Tua (+26),
  Zach Ertz (+25), Keenan Allen (+18).

## Useful next steps

- Surface `ffa_points` / `ffa_pos_rank` / `ffa_aav` from `nflv_ffa_league` as market
  columns on the draft board (true league-scale AAV is auction gold).
- Treat QB rows where market >> model as depth-chart-change candidates, not model wins.

*Scripts: `scripts/fetch_ffa_league.py`. Source: `data/ffanalytics/FFAn/projections_2026_wk0_league.csv`.*
