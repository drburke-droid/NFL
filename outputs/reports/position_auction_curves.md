# Per-position auction curves (league drafts 2023–25)

From `scripts/position_auction_curves.py` on `outputs/espn_drafts.csv`. Open-auction (non-keeper)
prices for top-$/ramp shape; replacement rank counts keepers too (a stud kept cheap still holds a
"worth-money" roster spot). 2023 had 0 keepers = a pure full auction.

## Pooled ($ by positional rank, recency-weighted 2023×1/24×2/25×3)
| Pos | r1 | r3 | r5 | r8 | r12 | r16 | r20 | r24 | r30 | **$1 replacement rank** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| **WR** | 64 | 51 | 38 | 30 | 25 | 18 | 14 | 11 | 7 | **~WR42** (money runs deep) |
| **RB** | 59 | 51 | 44 | 28 | 20 | 14 | 11 | 9 | 4 | **~RB43** |
| **TE** | 28 | 17 | 10 | 7 | 2 | 1 | 1 | – | – | **~TE13** (very shallow) |
| **QB** | 34 | 23 | 20 | 8 | 4 | 2 | 1 | – | – | **~QB16** (shallow) |

## Takeaways
- **Top $:** WR/RB top ~$60–68; QB ~$34–44; TE ~$24–28 (2023 Kelce $51 was an outlier). WR and RB
  are where the big money goes.
- **Ramp shape:** WR is the gentlest/deepest — 40+ WRs get real money, decaying slowly ($25 at
  WR12, $11 at WR24, still ~$7 at WR30). RB is steeper up top (r5 $44) then similar tail. **TE and QB
  are top-heavy and short:** they fall to a few dollars by rank ~8 and to $1 by ~12–16.
- **Replacement rank (where $1 hits) is very position-specific:** WR ~42, RB ~43, **QB ~16, TE ~13**.
  RB in 2024 looks shallow (RB27) only because it was the 11-team season; 2023/25 are ~36–46.

## Model implication
The value model's bench-floor uses a **uniform 1.6× the startable cutoff** as the "last rosterable"
depth (WR47 / RB47 / QB19 / TE21). Against the data that is:
- **WR ~1.45× (≈WR42)** and **RB ~1.5× (≈RB43)** — close, 1.6× is slightly deep.
- **QB ~1.4× (≈QB16)** — a touch deep.
- **TE ~1.05× (≈TE13)** — **much too deep** (1.6× = TE21). The model gives fringe TEs (14–21) a
  bench value when the league pays them $1.

**Suggested fix:** make the bench-floor depth position-specific — WR 1.5 / RB 1.5 / QB 1.4 / TE 1.1
— so fringe TE/QB collapse to $1 like the market does, while WR/RB keep their genuine deep value.
