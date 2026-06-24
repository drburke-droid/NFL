# Career-Trajectory Comp Finder

For any player, find the historical players whose **career arc** is most similar, then
see how those comps' careers *continued* — a data-driven analog projection.

## Data & method
- **`nflv_traj`**: 2006–2025 seasonal production + role + context, **11,143 player-seasons,
  2,855 players** (2,369 with fully-observed arcs entering 2006+). Modern era so air-yards /
  target-share / EPA are charted.
- Every season aligned to a **career year** (years_exp + 1), so players are compared at the
  same stage regardless of when they entered the league.
- Metrics **era-normalized** (z-scored within position × season) → comps match the *relative
  shape* of the arc, not absolute era scoring levels.
- **Distance** = mean per-career-year Euclidean over a position-specific feature set
  (PPG, opportunity, efficiency, role share, EPA…), with a penalty for missing/injured years.
- For a player at career year N, candidates are same-position players observed ≥ year N;
  the **continuation** = comps' median PPG at years N+1…N+3.

## Sample results (current ascending players, 2025)

| Player | Pos · Yr | 2025 PPG | Top trajectory comps | Comp-based next-yr |
|---|---|---|---|---|
| **Puka Nacua** | WR · 3 | 23.4 | Amon-Ra St. Brown, Justin Jefferson, Michael Thomas, Tee Higgins | 18.6 |
| **Bijan Robinson** | RB · 3 | 21.8 | Jahmyr Gibbs, Arian Foster, CMC, Le'Veon Bell | 16.5 |
| **JSN** | WR · 3 | 21.2 | DeAndre Hopkins, CeeDee Lamb, Tyreek Hill | 15.7 |
| **Drake Maye** | QB · 2 | 20.7 | Carson Wentz, Trevor Lawrence, Caleb Williams | 16.4 |
| **Jalen Hurts** | QB · 6 | 18.8 | Kyler Murray, Russell Wilson, Cam Newton, **Josh Allen, Lamar Jackson** | 17.6 |
| **Trevor Lawrence** | QB · 5 | 19.9 | Tannehill, Flacco, Dalton, Goff | 15.2 |
| **Trey McBride** | TE · 4 | 18.6 | Hockenson, Mark Andrews, Ertz | 12.2 |
| **Rashee Rice** | WR · 3 | 18.8 | Keenan Allen, Cooper Kupp, Percy Harvin | 14.0 |
| **Ja'Marr Chase** | WR · 5 | 19.6 | A.J. Brown, Amon-Ra, CeeDee Lamb, Justin Jefferson | 15.5 |

The archetype matching is the tell: **Hurts** lands on the dual-threat QB cluster (Murray/
Wilson/Newton/Allen/Jackson); **Lawrence** lands on the competent-but-capped pocket passers
(Tannehill/Flacco/Dalton/Goff) — a meaningfully different (and more sobering) trajectory than
his draft pedigree implies.

## How to read it
- The comps tell you **what kind of career a player is tracking** — elite-WR1 arc vs.
  solid-starter arc vs. boom/bust.
- The continuation projection is the **comps' median outcome**, i.e. "players who looked like
  this at the same stage averaged X PPG next year." It naturally captures regression (e.g.
  Nacua 23.4 → 18.6) and breakout continuation.

## Backtest (leakage-free): the projection is a *blend*
For 2,341 historical (player, career-year) cases, projected year N+1 PPG using only comps
whose own continuation was knowable at the time, vs a repeat-last-year baseline:

| Projection | MAE | Spearman |
|---|---|---|
| Comp median alone | 3.53 | 0.584 |
| Repeat last year | 3.30 | 0.688 |
| **50/50 blend (comp + last year)** | **3.03** | **0.693** |

**The comp projection alone is *worse* than repeating last year** (it over-regresses everyone
to the analog median), **but it carries independent signal** — blending it 50/50 with the
player's own prior year beats both by **~8% MAE**. So the app's **"Proj '26" uses the blend**,
and the **named comps are the real qualitative payoff** (what *kind* of career arc a player is
tracking), not the point estimate.

## The real power: the outcome *distribution*, not the point estimate
The far-future *average* of a comp cohort decays toward zero as careers end — a survivorship
artifact, not signal: mean forward PPG fades **9.7 → 7.7 → 6.0 → 4.4 → 3.3** over the next 5
years (share still playing drops **83% → 30%**). The fix is to summarize the cohort's **outcome
distribution** instead:

- **Ceiling** = comps' prime PPG (best of next 3 years, attrition counted), 75th percentile
- **Bust%** = share of comps whose prime stayed under 8 PPG
- **Elite%** = share that reached 18+ PPG

**Leakage-free backtest (n=1,825):** these rates carry strong, novel signal that a single
projection cannot:

| Signal | Result |
|---|---|
| Cohort **bust rate → actual bust** | **AUC 0.758** (base 30%) |
| Cohort **elite rate → actual elite** | **AUC 0.800** (base 14%) |
| Actual-elite rate by cohort-elite quintile | 5% · 4% · 6% · 17% → **36%** |

A player whose comps frequently hit elite reaches elite **~36%** of the time vs **~5%** for those
whose comps rarely did — a **7× spread**. Meanwhile the comp *point* projection (prime or
next-year) still does **not** beat repeat-last-year on MAE. **So comps are powerful as a
risk/ceiling profile, not as a better number** — and the app now shows Ceiling / Bust% / Elite%
alongside each player. Example: **Hurts elite 62% vs Lawrence elite 12%** (same-ish projection,
very different ceiling); **Trevor Lawrence's archetype rarely sustains elite** despite his pedigree.

## Caveats
- Rare archetypes get thin comps; era normalization helps but doesn't fully erase scheme/era.
- 2006-start cutoff means very early-2000s stars aren't in the pool.

*Tables: `nflv_traj`, `comp_results`. Scripts: `fetch_trajectories.py`, `comp_finder.py`,
`backtest_comps.py`. Web: `docs/comps.js` + Career Comps tab.*
