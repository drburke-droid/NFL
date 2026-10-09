# DK props pattern hunt

frame: 35,136 graded DK closing props 2023-25; overs hit 0.484
```
baseline (every prop, one side):
scope  side   d_n  d_hit  d_roi   v_n  v_hit  v_roi
  all  over 23276  0.491 -0.078 11860  0.470 -0.106
  all under 23276  0.509 -0.047 11860  0.530 -0.007
   qb  over  5440  0.494 -0.069  3179  0.484 -0.077
   qb under  5440  0.506 -0.056  3179  0.516 -0.038
 rush  over  4440  0.472 -0.119  2321  0.473 -0.104
 rush under  4440  0.528 -0.008  2321  0.527 -0.009
  rec  over 13396  0.495 -0.069  6360  0.463 -0.121
  rec under 13396  0.505 -0.056  6360  0.537  0.009

rules tested: 436  (cell x side x scope, n >= 150 discovery / 60 validation)
passed discovery (2023-24, ROI > 0, p < .05): 4   expected by luck: 21.8
survived validation too (2025, ROI > 0, p < .10): 0   expected by luck: 2.2

discovery winners and what 2025 did with them:
scope             rule  side  d_n  d_hit  d_roi   d_p  v_n  v_hit  v_roi   v_p
   qb   last_ppg_z>1.0  over  922  0.567  0.066 0.017  440  0.500 -0.054 0.882
   qb   last_ppg_z>1.5  over  664  0.569  0.069 0.030  237  0.540  0.019 0.381
 rush   line_vs_l3>1.0 under  169  0.609  0.117 0.046  109  0.523 -0.025 0.611
   qb last_stat_z<-1.0 under  610  0.570  0.062 0.049  383  0.522 -0.022 0.677

SURVIVORS:
  none
```

## Follow-up: the user's two ideas, and wind (2026-10-09)

Fade after a big game (UNDER when last game > 1 SD above his season average), 2023-24 -> 2025 ROI:
ppg z>1 -7.8% -> -1.2%; z>2 -6.0% -> -1.8%. GDELT volume spike z>2 -7.4% -> +1.7%; top-10% article count
-5.7% -> -6.5%; negative tone -6.2% -> +4.8% (the rule that already failed true OOS on 2023). DK prices
all of it.

Only structural lead: passing/receiving UNDERS (interceptions excluded) by FORECAST kickoff wind
(nflv_game_wind_fc, covers 2024-25 + 3% of 2023):
  0-4kn -4.3% (n 2648) | 4-8 -3.0% | 8-11 -0.1% | 11-13 +0.8% | 13-15 +12.0% (n 484) | 15+ +5.6% (n 296)
  >= 10kn: n 2512, +5.6%, z 3.0 (2024 +0.1%, 2025 +11.1%); >= 13kn: n 780, +9.6% (2024 -0.8%, 2025 +14.0%)
Observed wind >= 15mph, pass+rec: +0.9% / +2.1% / +10.3% (2023/24/25), pooled +5.1% z 2.3.
Dose-response, three seasons the same sign on observed wind, consistent with the validated wind-unders
playbook (game totals + QB pass yds). 2024 is flat on forecasts, so: WATCH (paper-trade), not proven.
