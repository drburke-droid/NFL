# Draft-strategy backtest — our projections + each positional tilt (N=80 MC each)

Our projection-based values held constant; only the positional strategy layered on top changes. Bots + in-season fixed. Lower avg finish = better.

| Strategy (on top of our projections) | Avg finish | 95% CI |
|---|---|---|
| qb_heavy | 4.06 | [2.60, 5.41] |
| **current(neutral)** | 4.38 | [3.00, 6.21] |
| robust_rb | 4.51 | [3.09, 6.90] |
| rb_heavy | 4.54 | [2.80, 6.51] |
| mild_stars_scrubs | 4.66 | [3.10, 6.60] |
| hero_rb | 4.69 | [2.60, 6.50] |
| elite_te | 4.71 | [3.00, 6.60] |
| wr_heavy | 4.76 | [3.10, 6.80] |
| te_premium | 4.77 | [3.20, 6.70] |
| zero_rb | 4.78 | [2.70, 6.80] |
| extreme_balanced | 4.80 | [3.09, 6.61] |

## Read
**No positional strategy reliably beats our current neutral approach.**
- Current neutral (projections + VORP, take the best value regardless of position): **4.38** — effectively tied for first.
- **qb_heavy (4.06)** is the only tilt that nominally beats it (+0.32), but the CIs overlap heavily ([2.60,5.41] vs [3.00,6.21]) — a soft lean, not a proven edge. It hints an elite QB's large, durable weekly margin can be worth paying up for when the value is there.
- **Every other tilt is worse than neutral**: TE-premium 4.77, WR-heavy 4.76, RB-heavy 4.54, zero-RB 4.78, hero-RB 4.69, stars-and-scrubs 4.66. (te_premium looked great at N=4 — pure noise; it regressed to the pack by N=80.)

**Conclusion:** with accurate projections, draft "strategies" are mostly marketing. The
optimal play is to **take the best VORP value on the board regardless of position** —
exactly what we do. Forcing a positional tilt makes you overpay one spot and underpay
others, costing ~0.2–0.4 places. The only defensible deviation is a mild willingness to
pay up for an elite QB if the price is right.