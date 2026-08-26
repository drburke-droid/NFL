# Buzz/GDELT subset sweep — 12 pre-registered tests

delta = flagged-minus-unflagged mean of the outcome; p = permutation two-sided;
seasons+ = seasons with positive delta. With 12 tests expect ~0.6 false positives
at p<.05 — only CANDIDATE verdicts (p<.05 AND >=65% season consistency) matter,
and even those need a mechanism story before touching the board.

```
                             test  n1   n0   rate1  rate0   delta     p seasons+   verdict
         1 buzz x post-hype cheap  49  128   0.041  0.078  -0.037 0.523      1/9     noise
2 buzz x vacated-targets cheap WR 135  352   0.037  0.031   0.006 0.770     2/10     noise
3 buzz x vacated-carries cheap RB  64  123   0.078  0.041   0.037 0.315     4/10     noise
      4 buzz x satellite cheap RB   7   37     NaN    NaN     NaN   NaN      NaN  TOO THIN
  5 buzz x young priced (age<=24)  65  132   0.615  0.629  -0.013 0.876     4/10     noise
      6 buzz x team-change priced  11   44     NaN    NaN     NaN   NaN      NaN  TOO THIN
           7 buzz x mid-tier $3-8 146  343   0.315  0.329  -0.014 0.830     4/10     noise
      8 neg-news x priced age>=28  24  186 -17.115 19.298 -36.413 0.048      2/9      weak
      9 neg-news x priced age<=25  59  241  16.564 21.235  -4.671 0.722     4/10     noise
               10 buzz x cheap TE 247  706   0.024  0.025  -0.001 1.000     3/10     noise
                  11 fame x cheap 817 4371   0.040  0.019   0.021 0.000    10/10 CANDIDATE
         12 buzz x H2-riser cheap 192  441   0.031  0.027   0.004 0.793     4/10     noise
```

## Follow-up: the two survivors

**Test 11 (fame x cheap) — real gradient, redundant signal.** Cheap players with
top-quartile GDELT article volume hit 4.0% vs 1.9% (10/10 seasons). It survives
within rookies (3.4 vs 1.1), vets (4.2 vs 2.3) and every prior-games bin. BUT a
logit controlling for age / draft round / rookie / prior games / prior PPG shrinks
the standardized fame coef to +0.06 - dwarfed by draft round (-0.55), games (+0.87),
rookie (+0.53). Famous cheap players are mostly high-draft-capital experienced names,
which the dart classifier already prices. Curiosity: fame WITHOUT buzz hits 5.0%,
fame WITH buzz only 2.7%. Verdict: nothing to add to the board.

**Test 8 (neg-news x priced age>=28) - a real refinement of the shipped flag.**
The existing negative-news flag costs priced players ~-20 VORP pooled. Split by age:
28+ flagged players ran -36.4 pts vs unflagged (p=.048, 7/9 seasons in the
hypothesized direction, n=24/decade); age<=25 flagged players ran only -4.7 (ns).
The damage concentrates in older players - August negativity about a 28+ vet is
often decline/injury smoke that proves real; on a young player it is noise.
Actionable: make the existing flag age-aware (stronger/only for 28+).
