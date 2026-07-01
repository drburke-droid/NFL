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

## Does it improve projection MAE? (walk-forward, no leakage — `scripts/contract_year_mae.py`)
Tested by adding a walk-year feature to an OLS projection (next-season PPG ~ prior-2-season PPG +
age + age² + position), trained only on prior seasons, evaluated 2016–2024:

| | MAE baseline | MAE +walk-year | Δ |
|---|--:|--:|--:|
| Full panel | 2.650 | 2.603 | **−0.046** (−1.7%) |
| Walk-year players only | 2.647 | 2.604 | −0.043 |

- Out-of-sample walk-year coefficient **+1.41 PPG, stable every year** (+1.29…+1.55) — a learnable
  signal, not noise. (Larger than the +0.97 within-player figure because here it's conditional on
  prior PPG + age.)
- **Full-panel MAE improved in all 9 test years** — the feature never hurt.
- On walk-year players specifically it helps on average but is noisy year-to-year (small n).

**Verdict: it genuinely improves projections, but modestly** (~1.7% MAE; only ~15% of players
affected). Worth a small bump, not a needle-mover.

### Is there a post-contract "fade"? No — it fails validation
The event study shows performance dropping after signing (walk +1.04 → first paid year +0.47), which
looks like a "cash-in and coast" fade. But once a projection regresses the inflated walk-year lag,
that drop is already accounted for. Adding a post-signing feature to the walk-forward test gives a
**+0.90 PPG coefficient (positive, stable every year)** — post-signing players sit slightly *above*
their lag/age baseline, not below (talent selection). Full-panel MAE barely moves (2.603 → 2.592),
and on post-signing players themselves MAE gets **worse** (2.650 → 2.761). So the "fade" is
regression-to-the-mean, not a behavioral letdown: **no fade value adjustment is applied** — "just
signed" stays an informational tag only.

## Practical read for drafting/keepers
- A **modest, real edge (~1 PPG / ~10%)** — worth a small bump as a tiebreaker, most for **RBs**
  entering a walk year.
- Mind the flip side: players who **just signed** give back roughly half the bump (post-contract
  fade) — a mild fade signal, again largest at RB.
