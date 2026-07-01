# WR team changes: how usage redistributes (2018–2024)

Reproducible: `scripts/team_change_usage.py` (nfl_data_py weekly, PPR, target share = player targets
/ team targets). 115 WR movers (target share ≥8%, ≥6 games, changed primary team Y→Y+1) vs 371
stayer controls.

## Findings
| Question | Result |
|---|---|
| **Q1 mover — flat?** | No. Target share **15.1% → 12.4% (−2.7pp)**, PPG **9.8 → 8.3 (−1.5)**. Stayers decline only −1.5pp / −0.8, so the **net moving penalty ≈ −1.2pp / −0.7 PPG**. |
| **Q2 new teammates** | Incumbent WRs on the new team dip **−2.1pp** combined (newcomer takes targets). |
| **Q3 replacement** | The single biggest gainer on the old team absorbs **~114% of the vacated share**; **85%** of moves have one player taking ≥60% of the role. The job concentrates on one guy. |
| **Q4 old teammates** | Holdover WRs on the old team gain **+1.8pp** combined (plus the concentrated replacement). |

## Interpretation
Target share is roughly **conserved and redistributes predictably**. A departing WR's ~15% mostly
flows to **one replacement** (not spread), with a small bump to holdovers; meanwhile on his new team
he *loses* share (incumbents entrenched) and underperforms his prior line. **The vacated role is the
opportunity; the mover himself tends to fade.**

## Caveats
- Q1 has **selection bias**: WRs who change teams are often the ones being squeezed out/declining, so
  part of the −1.2pp penalty is who-moves, not moving itself. Directionally it's a real, modest fade.
- Descriptive (usage), not a projection claim — whether the *replacement* beats projection needs its
  own walk-forward backtest before acting on it.
