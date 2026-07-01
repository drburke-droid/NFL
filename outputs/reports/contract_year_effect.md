# Do NFL skill players overperform in contract years?

**Short answer: yes — about +1.0 PPR PPG (~+10%) above their own age-adjusted career average in
the walk year, strongest at RB — but selection/regression-to-mean means treat ~+1 PPG as an
upper-ish bound.** Reproducible: `scripts/contract_year_analysis.py` (nfl_data_py, 2011–2024).

## Method (fighting the selection trap)
Good players *earn* second contracts, so "walk-year players are good" is pure selection. Controls:
1. **Within-player** — each player is his own control (deviation from his own career mean PPG).
2. **Age-adjusted** — subtract the league aging curve (walk years cluster at peak age).
3. **Event study + post-signing** — a bump that *fades once paid* is the fingerprint of a real
   effect, not just a good player.

A *walk year* = the season **before** a player signs a new multi-year veteran deal (years ≥ 2,
APY ≥ $3M, not his rookie contract). Panel: 2,638 skill player-seasons, 428 players, 392 walk
years, games ≥ 6, PPR PPG (mean 9.3).

## Result — age-adjusted deviation from own career average
| Season | n | Age-adj PPG | % | p |
|---|--:|--:|--:|--:|
| **Contract (walk) year** | 392 | **+0.97** | **+10.5%** | <0.001 |
| Year after signing | 396 | +0.41 | +4.4% | 0.005 |
| All other seasons | 1874 | −0.30 | −3.2% | <0.001 |

**By position (walk-year):** RB **+1.53** (p<0.001) · QB +1.05 (p=0.014) · WR +0.91 (p<0.001) ·
TE +0.54 (p=0.058, marginal). The effect is real everywhere but biggest for running backs.

## Event study (0 = first year of the new deal)
```
-3  -0.36     -2  -0.14     -1 +1.04  ← walk year     0 +0.47     +1 +0.24     +2 +0.18
```
Performance sits **below** career average in the years before, **spikes** in the walk year, then
**fades** progressively once the player is paid (~0.56 PPG evaporates from walk year to the first
paid year). The ramp-up before and fade after are what distinguish a behavioral effect from a lone
random high season.

## Caveats
- The walk year is **by construction** the season that earned the contract, so selection /
  regression-to-mean inflates it. Within-player + age adjustment remove player quality and the
  aging curve, **not** the walk-year-is-selected-high problem. Hence ~+1 PPG is an **upper-ish bound**.
- The post-signing fade is the best hint of a genuine effect, but is also partly what
  regression-to-mean alone would produce.
- Single-season, ≥6-game samples; simple 1-sample t-tests (not player-clustered).

## Practical read for drafting/keepers
- A **modest, real edge (~1 PPG / ~10%)** — worth a small bump as a tiebreaker, most for **RBs**
  entering a walk year.
- Mind the flip side: players who **just signed** give back roughly half the bump (post-contract
  fade) — a mild fade signal, again largest at RB.
