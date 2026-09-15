# Practice reports vs availability and performance (2017-2025)

**Verdict: the first non-null feature since the DraftKings blend.** The weekly injury report (designation + last
practice status, nflverse `injuries`, on hand Friday before every send) predicts who sits far better than the FFA
tag the model uses today, and a listed player who plays produces ~90% of a healthy player's output:

- **Availability.** Out/Doubtful is near-deterministic (0-3% play). Questionable splits by practice: full 70%,
  limited 62%, DNP 44%; by position Q+LP QBs play only 45% and Q+DNP TEs 32%. Walk-forward P(plays) log-loss
  improves 0.409 -> 0.387 (-5.5%), AUC 0.785 -> 0.827, over the FFA tag alone.
- **The FFA tag is noisy at scrape time.** Week 1 2026: FFA tagged 57 players Q; the final report listed 8 as
  Questionable and 44 not at all. The DOUBT path's P(plays) = 0.2 for FFA Out/Doubtful is far too high in the
  history (0.2% played), and the Questionable population it treats as one group spans 32-81%.
- **Performance when they play.** Relative to unlisted players at the same projection: FP 0.98, LP 0.92, DNP
  0.90 (WR 0.85, RB 0.92); "rest day" DNPs with no designation 0.88; illness 0.84, groin 0.88, knee/hip 0.91.
  Early exits are only mildly elevated (10% vs 6%). FFA already discounts LP by ~0.4 pts vs trailing form,
  about a third of the observed shortfall.
- **A walk-forward haircut on played rows** (shrunk per designation x practice cell) cuts MAE on listed players
  5.108 -> 5.085 (-0.45%, 8/8 seasons) and on LP/DNP 5.177 -> 5.143 (-0.65%, 8/8); all rows -0.05%.

Proposed wiring (not yet built): (1) replace the FFA tag in the doubt path with the official designation +
practice status (Sleeper's players feed carries `practice_participation` at T-90; nflverse for history) and the
position-specific play-rate table below as P(plays); (2) multiply the playing projection by 0.92 (LP) / 0.90
(DNP) for Questionable players, WR 0.85 on DNP. Script: scripts/practice_report_study.py.

---

# Detail

Projected player-weeks (QB/RB/WR/TE with an FFA line): 65,692; 66.3% played. Injury report = the nflverse weekly final report (one row per player-week: designation + last practice status). Report timestamps fall on {'Friday': 0.79, 'Wednesday': 0.08, 'Saturday': 0.08, 'Thursday': 0.05}, i.e. it is on hand before the T-90 send. Listed players: 17.5% of rows.

## 1. Who sits: play rate by designation x practice status

| designation | practice | rows | play rate |
|---|---|---|---|
| not listed | not listed | 54,223 | 65.7% |
| no designation | FP | 5,653 | 84.8% |
| no designation | LP | 923 | 89.6% |
| no designation | DNP | 481 | 77.5% |
| Questionable | FP | 608 | 70.2% |
| Questionable | LP | 2,060 | 62.2% |
| Questionable | DNP | 453 | 43.7% |
| Questionable | listed, no practice status | 44 | 70.5% |
| Doubtful | LP | 59 | 3.4% |
| Doubtful | DNP | 173 | 1.2% |
| Out | LP | 96 | 0.0% |
| Out | DNP | 886 | 0.0% |

What the model sees today is the FFA tag alone. Play rate by FFA tag, then split by the practice status the model does NOT see:

| FFA tag | practice | rows | play rate |
|---|---|---|---|
| **none** | all | 61,523 | 69.0% |
| none | not listed | 53,353 | 66.7% |
| none | FP | 5,883 | 84.8% |
| none | LP | 1,635 | 85.7% |
| none | DNP | 629 | 73.1% |
| **Q** | all | 2,038 | 54.5% |
| Q | not listed | 202 | 18.8% |
| Q | FP | 341 | 68.9% |
| Q | LP | 1,181 | 60.0% |
| Q | DNP | 289 | 38.8% |
| **O/D** | all | 2,131 | 0.2% |
| O/D | not listed | 668 | 0.3% |
| O/D | FP | 56 | 1.8% |
| O/D | LP | 322 | 0.3% |
| O/D | DNP | 1,075 | 0.1% |

Questionable by practice status and position:

| position | Q + FP | Q + LP | Q + DNP |
|---|---|---|---|
| QB | 81% (n=47) | 45% (n=155) | 42% (n=24) |
| RB | 63% (n=169) | 65% (n=512) | 41% (n=107) |
| WR | 75% (n=245) | 65% (n=1008) | 49% (n=244) |
| TE | 68% (n=147) | 57% (n=385) | 32% (n=78) |

Walk-forward P(plays) (logistic, train = earlier seasons, test 2018-2025) on rows with any injury information (FFA tag or report listing):

| model | log-loss | Brier | AUC |
|---|---|---|---|
| FFA tag only | 0.4092 | 0.1298 | 0.785 |
| + designation | 0.3923 | 0.1227 | 0.804 |
| + designation + practice | 0.3919 | 0.1225 | 0.810 |
| + ... + position | 0.3866 | 0.1209 | 0.827 |

Rows in that population: 12,339 (19% of all); base play rate 64.7%.

DOUBT path today assumes P(plays) = 0.2 for FFA Out/Doubtful. Observed: 0.2% over 2,131 rows; of those the official designation was Out 43% (played 0.0%), Doubtful 10% (played 0.5%), Questionable 15% (played 0.0%), nothing/none 33% (played 0.6%).

## 2. When a listed player plays: performance vs the model

Median actual/projection relative to unlisted players in the same projection band (1.00 = performs like a healthy player at that projection), with mean residual, snap share vs own norm, early exits (<50% of usual snaps) and bust rate (residual <= -10):

| designation | practice | played rows | relative output | mean residual | median snap ratio | early exit | bust rate |
|---|---|---|---|---|---|---|---|
| not listed | not listed | 35,622 | 1.00 | +0.69 | 1.02 | 6% | 2.5% |
| no designation | FP | 4,796 | 0.98 | +0.28 | 1.02 | 5% | 4.3% |
| no designation | LP | 827 | 0.98 | +0.07 | 1.01 | 5% | 5.4% |
| no designation | DNP | 373 | 0.88 | -0.49 | 1.00 | 5% | 5.1% |
| Questionable | FP | 427 | 0.98 | +0.50 | 1.00 | 8% | 3.3% |
| Questionable | LP | 1,282 | 0.92 | -0.10 | 0.99 | 10% | 4.1% |
| Questionable | DNP | 198 | 0.90 | -1.00 | 0.97 | 7% | 6.1% |

By position, Questionable players who played:

| position | Q + FP output | Q + LP output | Q + DNP output |
|---|---|---|---|
| QB | 0.96 (n=38) | 0.99 (n=70) | n/a |
| RB | 0.99 (n=106) | 0.96 (n=333) | 0.92 (n=44) |
| WR | 0.97 (n=183) | 0.91 (n=658) | 0.85 (n=119) |
| TE | 1.00 (n=100) | 0.93 (n=221) | 1.14 (n=25) |

By primary injury (listed players who played, LP or DNP, n >= 60):

| injury | played rows | relative output | early exit | play rate when listed |
|---|---|---|---|---|
| Ankle | 234 | 0.95 | 9% | 42% |
| Groin | 79 | 0.88 | 4% | 45% |
| Hamstring | 185 | 0.97 | 16% | 34% |
| Hip | 72 | 0.91 | 8% | 56% |
| Illness | 74 | 0.84 | 7% | 56% |
| Knee | 249 | 0.91 | 9% | 39% |
| Shoulder | 114 | 0.99 | 10% | 45% |

Does FFA already discount listed players? FFA line minus the player's trailing-3 average (played rows):

| practice | not listed | FP | LP | DNP |
|---|---|---|---|---|
| FFA - trailing 3 (mean) | -0.05 | -0.02 | -0.42 | -0.26 |

## 3. Would a practice-status haircut improve the projection for players who play?

- all played rows (n = 38,713): incumbent MAE 4.603; additive haircut 4.601 (7/8 seasons better); multiplicative 4.603 (3/8 seasons better).
- listed players only (n = 7,157): incumbent MAE 5.108; additive haircut 5.085 (8/8 seasons better); multiplicative 5.090 (8/8 seasons better).
- LP or DNP only (n = 2,444): incumbent MAE 5.177; additive haircut 5.143 (8/8 seasons better); multiplicative 5.143 (8/8 seasons better).
