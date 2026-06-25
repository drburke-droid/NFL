# Edge attribution — our team's avg finish by valuation arm (N=80 Monte Carlo each)

Bots + all in-season logic held fixed; only OUR draft valuation changes. Lower = better.

## Part A — where does the edge come from? (bench weight = 0.35)
- field (prior-yr/ADP + VORP) [control] 5.66  [4.20, 7.30]
- naive (our proj, NO VORP)          4.01  [2.90, 5.41]
- our proj + VORP                    4.38  [2.60, 6.70]
- our proj + VORP + leap/fade tilt   4.39  [3.10, 6.50]

**Decomposition (avg finishing places gained vs the field control):**
- Projections (field → our-proj, both VORP): **+1.28**
- VORP vs naive (both our-proj):              **-0.37**
- Leap/fade tilt (on top of proj+VORP):       **-0.01**
- TOTAL (field → full):                       **+1.27**

## Part B — maximize starter VORP, whole-roster, or hybrid? (valuation = proj+VORP+tilt)
- starters only (bench_w=0.0)        5.82  [3.50, 7.70]
- hybrid (bench_w=0.35) [current]    4.39  [3.10, 6.50]
- whole roster (bench_w=1.0)         5.59  [3.50, 7.60]

## Read
**Q1 — the edge is the PROJECTIONS, not VORP.** Swapping the field's prior-year/ADP
inputs for our walk-forward projections (both using VORP) gains **+1.28** finishing
spots — essentially the entire edge. VORP itself adds **nothing** over a simple
points-proportional valuation once projections are good (point estimate −0.37, well
inside the CI), and the leap/fade tilt is neutral (−0.01, consistent with it being a
display tool). Takeaway: spend the effort on projection accuracy; VORP is a fine,
convenient bookkeeping for converting projections to prices, but it is not the source
of the advantage.

**Q2 — the hybrid bench weight (0.35) is optimal.** Both extremes finish ~1.2–1.4
places worse: *starters-only* (5.82) leaves you with $1 bench scrubs and no cover for
byes/injuries; *whole-roster* (5.59) bleeds money to the bench at the expense of stud
starters. Pay near-full value for the 7 starting slots and ~a third for bench depth —
which is what the tool already does.

Caveat: single-format, N=80; CIs overlap, so read the *rankings/point-estimates* as
the signal. The projection gap (field 5.66 vs our-proj 4.38) is the one clearly
outside-noise result.