# Stars who came cheap — do below-average buys ever finish top-10%?

`scripts/cheap_stars_build.py` (frame) → `scripts/cheap_stars_study.py` (this report).

**The question.** Not "does a $2 dart clear replacement" (that is
`LATE_BREAKOUTS.md`), but: is there a *class* of player who goes for **less than the
average price at his position** and finishes **top-3 QB / top-6 RB / top-9 WR /
top-3 TE** — the top ~10% of startable players at the spot?

## Setup

- **Universe** — 2016-2025, the league-scored FFA preseason (wk0) file, skill
  positions, AAV ≥ $1, inside the top 168 by AAV (the ~14 skill players per team a
  12-team/16-slot room actually buys). 10 seasons, 1,678 buys, 204 elite finishes.
- **Price** — FFA league AAV as a **ratio to that season's positional mean** inside the
  pool, so 2016 dollars and 2025 dollars are comparable. Cheap = ratio < 1.
  (FFA AAV vs this league's real clearing bids: Spearman 0.90, `LATE_BREAKOUTS.md`;
  §6 re-runs everything on the real bids.)
- **Star** — season total in league scoring (full PPR, 6-pt pass TD, −1 INT), ranked
  QB≤3, RB≤6, WR≤9, TE≤3. Season totals, not per-game: missed games count against you.
- **Value** — VORP against QB13/RB27/WR35/TE13. `E[val if kept]` is VORP floored at 0,
  because a busted $8 buy gets cut in week 4 for a waiver body: you lose the dollars
  and the slot, not 60 points of VORP.

## 1. Yes — and at QB and TE it is most of the market

Below-average buys star far less often than expensive ones, but they are **not** a
rounding error: they supplied a quarter of all elite seasons, and at QB nearly half.

| pos | star = top | pos. avg $ | cheap buys | cheap stars | P(star) if cheap | P(star) if at/above | % of elite seasons bought cheap |
|---|---|---|---|---|---|---|---|
| QB | 3 | 8.5 | 212 | 12 | 5.7 | 20.2 | 40 |
| RB | 6 | 17.2 | 367 | 11 | 3.0 | 26.1 | 19 |
| WR | 9 | 14.1 | 415 | 22 | 5.3 | 31.7 | 25 |
| TE | 3 | 8.3 | 148 | 8 | 5.4 | 34.5 | 30 |

Pooled: **53 of 204 elite seasons (26%) were bought below the positional average**, at 4.6% per buy.

The asymmetry is structural. QB and TE are one-starter positions where the whole
market is compressed into single digits, so "below average" means $4-6 and the gap
between the average QB buy and an elite one is a few dollars. RB is the opposite: the
elite tier is bought, and 81% of top-6 RB seasons came from at-or-above-average prices.

## 2. The bargain bin is not where cheap stars come from

Sorting the whole pool by price relative to the positional mean:

| price vs pos. avg | n | avg $ | star % | stars | $ per star | P(above repl.) | E[val if kept] | share of stars % |
|---|---|---|---|---|---|---|---|---|
| under 25% | 422 | 2.4 | 2.1 | 9 | 110 | 20.6 | 8 | 4 |
| 25-50% | 413 | 4.5 | 3.1 | 13 | 142 | 26.9 | 13 | 6 |
| 50-75% | 175 | 7.8 | 9.7 | 17 | 81 | 48.6 | 21 | 8 |
| 75-100% | 132 | 11.6 | 10.6 | 14 | 109 | 51.5 | 27 | 7 |
| 100-150% | 149 | 16.5 | 17.4 | 26 | 94 | 59.7 | 36 | 13 |
| 150-250% | 180 | 28.1 | 21.7 | 39 | 130 | 69.4 | 53 | 19 |
| 250%+ | 207 | 44.7 | 41.5 | 86 | 108 | 81.2 | 84 | 42 |

Two things fall out of this table:

1. **P(star) is monotone in price.** The market is not fooled: there is no price band
   where stardom is more *likely* than the band above it.
2. **`$ per star` is not monotone, and it is flattest where you would not expect.**
   Across the whole board a top-10% season costs $80-140 in expectation no matter
   where you shop — the cheapest cell is 50-75% of the positional average ($81),
   the most expensive is 25-50% ($142). Dollars buy stardom at roughly one rate;
   what changes down the price curve is the **slot cost and the floor**: 76% of
   sub-50% buys finish below replacement, against 49% at 50-75% and 19% at the top.

So "cheap star" does not mean the $1-3 flier. Below 50% of the positional average
the star rate is 2.6% and three-quarters of the buys are dead roster spots by
October. The cheap stars live in the **discount rack right below the average price** —
the 50-100% band, roughly $6-16 depending on position.

## 3. Every draft-day feature, run against the price

Every draft-day feature we have, tested inside the cheap pool against controls for
log(price ratio) and position — so this is *residual* signal, over and above what the
price already says:

| draft-day feature | n with it | raw P(star) | odds ratio | p (vs price + position) |
|---|---|---|---|---|
| projected inside starter tier | 256 | 12.5 | 3.94 | 0.0002 |
| price ranks worse than projection (gap>=5) | 267 | 6.4 | 2.98 | 0.0019 |
| changed teams | 249 | 2.4 | 0.5 | 0.0873 |
| age <= 25 and NFL round 1-2 | 307 | 6.5 | 1.43 | 0.2389 |
| prior year: missed 4+ games | 330 | 3.3 | 0.69 | 0.2728 |
| age >= 30 | 224 | 4.0 | 0.71 | 0.3686 |
| new team vacated 100+ targets | 887 | 4.4 | 0.76 | 0.4175 |
| years 1-3 of career | 411 | 5.1 | 1.25 | 0.4564 |
| prior year: target share >= 18% | 249 | 7.2 | 1.23 | 0.6009 |
| FFA ceiling premium (continuous) | 1142 |  | 1.86 | 0.6150 |
| new team vacated 120+ carries | 567 | 5.1 | 1.15 | 0.6239 |
| prior year: starter-tier PPG | 255 | 7.1 | 1.09 | 0.7872 |
| NFL round 1-2 (any age) | 615 | 5.5 | 1.06 | 0.8515 |
| prior year: snap share >= 60% | 581 | 5.9 | 1.07 | 0.8518 |
| FFA uncertainty below median | 580 | 5.7 | 1.05 | 0.8662 |
| rookie with round 1-2 capital | 122 | 4.9 | 0.97 | 0.9496 |

Two rows clear the bar (they are the same finding, and §4 takes them apart);
with 16 tests and 53 events nothing else survives a multiplicity correction.
Note what died: **age + draft capital, second/third-year breakout window,
vacated targets/carries, prior snap and target share, ceiling premium**. Those are all
real breakout signals — `LATE_BREAKOUTS.md` finds them at the $1 end — but at this
price level the market has *already* paid for them. Changing teams is, if anything, a
negative, which matches the dart study.

## 4. The one thing that survives: the projection already knows

The class is defined by two draft-day facts, no model required:

> **Priced at 50-100% of the positional average, and projected inside the positional
> starter tier (QB1-12 / RB1-24 / WR1-30 / TE1-12 by consensus points).**

| band | projection | n | stars | star % | avg $ | $ per star |
|---|---|---|---|---|---|---|
| under 50% of avg | outside starter tier | 762 | 16 | 2.1 | 3.3 | 159 |
| under 50% of avg | starter-tier projection | 73 | 6 | 8.2 | 3.9 | 48 |
| 50-100% of avg | outside starter tier | 124 | 5 | 4.0 | 8.8 | 217 |
| 50-100% of avg | starter-tier projection | 183 | 26 | 14.2 | 9.9 | 70 |
| at/above avg | outside starter tier | 12 | 0 | 0.0 | 16.1 | — |
| at/above avg | starter-tier projection | 524 | 151 | 28.8 | 31.7 | 110 |

**26/183 = 14.2%** (95% CI 9-20%) versus
**4.0%** for buys at the *same price* whose projection is weaker
(Fisher p = 0.0035). At **$9.9 a head and ~18 of them a season**,
this is the cheapest stardom on the board: **$70 per elite
season** against $113 at the top of the market.

The mechanism is boring and that is the point. These are not sleepers — they are
*consensus starters the room does not want to pay for*: the quarterback who finished
QB15 on a bad team, the tight end coming off an injury year, the WR3 on a good
offense. The projections have them in the starter tier; the auction prices them as
depth. **No breakout is required for the class to pay** — one healthy season at the
projected role is often enough, because the elite tier at QB/TE/WR is thin.

One caution, stated plainly: an earlier cut of this analysis used *price rank minus
projection rank ≥ 5* ("the market disagrees with the projection") and got 37% on
n=27. Put both variables in one model and the disagreement term goes to zero
(p = 0.77) while the projection rank stays (p = 0.005). The disagreement version was
the same signal with a smaller sample; the projection-tier version above is the one
to trade.

### Per season and per position

| season | n | stars |
|---|---|---|
| 2016 | 18 | 1 |
| 2017 | 21 | 5 |
| 2018 | 18 | 2 |
| 2019 | 14 | 3 |
| 2020 | 16 | 2 |
| 2021 | 12 | 1 |
| 2022 | 21 | 4 |
| 2023 | 23 | 5 |
| 2024 | 18 | 1 |
| 2025 | 22 | 2 |

| position | n | stars | avg $ | star % |
|---|---|---|---|---|
| QB | 33 | 6 | 5.8 | 18.2 |
| RB | 47 | 5 | 14.0 | 10.6 |
| TE | 32 | 3 | 6.2 | 9.4 |
| WR | 71 | 12 | 10.8 | 16.9 |

Stars in **all 10 seasons**, 1-5 a year. WR is the deepest source (12 of 26), which
is simply where the pool is biggest; TE and QB have the best rate per dollar.

## 5. Does it hold out?

| era | cell | n | stars | star % |
|---|---|---|---|---|
| 2016-2021 | the class | 99 | 14 | 14.1 |
| 2016-2021 | same price, weaker projection | 54 | 3 | 5.6 |
| 2016-2021 | under 50% of avg + starter-tier proj. | 61 | 6 | 9.8 |
| 2016-2021 | under 50% of avg (all) | 539 | 14 | 2.6 |
| 2016-2021 | at/above avg (all) | 314 | 90 | 28.7 |
| 2022-2025 | the class | 84 | 12 | 14.3 |
| 2022-2025 | same price, weaker projection | 70 | 2 | 2.9 |
| 2022-2025 | under 50% of avg + starter-tier proj. | 12 | 0 | 0.0 |
| 2022-2025 | under 50% of avg (all) | 296 | 8 | 2.7 |
| 2022-2025 | at/above avg (all) | 222 | 61 | 27.5 |

The class rate is flat across eras (14.1% → 14.3%) while the *cheaper* cell that
looked good early — a starter-tier projection under 50% of the positional average —
went 6/61 in 2016-21 and 0/12 in 2022-25. Same conclusion as the dart study: the
market closed the deep end, not the discount rack.

### Sensitivity to the cutoffs

The two cutoffs — a 50% price floor and the starter-tier projection — were chosen
after looking at the data, so here is the same class under every reasonable
alternative definition:

| variant | class n | class stars | class star % | same price, weaker proj. % | p |
|---|---|---|---|---|---|
| as reported: top 168 pool, mean, top 3/6/9/3 | 183 | 26 | 14.2 | 4.0 | 0.0035 |
| pool = top 192 (every drafted body) | 165 | 21 | 12.7 | 3.8 | 0.0044 |
| pool = top 140 (tighter) | 207 | 32 | 15.5 | 4.1 | 0.0038 |
| centre = positional MEDIAN, not mean | 56 | 3 | 5.4 | 2.7 | 0.3931 |
| star = top 2/4/6/2 (tighter) | 183 | 17 | 9.3 | 2.4 | 0.0179 |
| star = top 4/8/12/4 (looser) | 183 | 32 | 17.5 | 4.0 | 0.0003 |
| star = top 20% (6/12/18/6) | 183 | 51 | 27.9 | 8.9 | <0.0001 |

It survives every pool size and every star definition. It does **not** survive
swapping the positional mean for the median, which is informative rather than fatal:
the median sits at $4-8, so "50-100% of median" lands squarely in the bargain bin,
where §2 already showed there is nothing to find. *Below average* is the frame that
works, which is also the frame the question was asked in.

## 6. Replication on this league's own auction (2023-25)

Everything above rides on FFA AAV. Re-run it on the actual winning bids in this
room (425 non-keeper buys, `outputs/espn_drafts.csv`), with the positional
average computed from what the room actually paid:

| band | n | stars | avg bid | star % |
|---|---|---|---|---|
| under 50% of avg | 198 | 5 | 2.3 | 2.5 |
| 50-100% of avg | 77 | 8 | 9.7 | 10.4 |
| at/above avg | 150 | 31 | 29.2 | 20.7 |

And the class itself: **7/51 = 14%** at an average winning bid of **$9.9** — the study said 14.2% at $9.9. Independent data,
same number. The hits: Rachaad White ($15, RB4), Breece Hall ($12, RB2), Mike Evans ($8, WR7), Dak Prescott ($8, QB2), Terry McLaurin ($10, WR7), Travis Kelce ($7, TE3), George Pickens ($9, WR5).

## 7. What it is worth at the table

| cell | per season | avg $ | star % | $ per star | P(above repl.) | E[val if kept] | value per $ |
|---|---|---|---|---|---|---|---|
| the class (50-100% avg + starter-tier proj.) | 18.3 | 9.9 | 14.2 | 70 | 56.3 | 30 | 3.0 |
| bargain bin (under 50% of avg) | 83.5 | 3.4 | 2.6 | 129 | 23.7 | 10 | 3.0 |
| mid market (100-200% of avg) | 24.4 | 19.6 | 18.4 | 106 | 63.9 | 41 | 2.1 |
| studs (200%+ of avg) | 29.2 | 41.1 | 36.3 | 113 | 77.4 | 77 | 1.9 |

Note what the `value per $` column does *not* say: the class and the bargain bin
return the same **3.0× per dollar**. Dollars are not the scarce thing at this end of the
auction — **roster spots are**, and that is the whole difference. A class buy clears
replacement **56%** of the time, so the slot is live even when the elite season does
not come; a bargain-bin buy is a dead slot **76%** of the time. Stacking them:

- **1 class buy (~$10)** → 14% chance of at least one elite finisher
- **2 class buys (~$20)** → 26% chance of at least one elite finisher
- **3 class buys (~$30)** → 37% chance of at least one elite finisher
- **4 class buys (~$40)** → 46% chance of at least one elite finisher
- **one stud (~$41)** → 36% chance, and he is a top-6 asset when he misses

Four class buys cost the same as one stud and land an elite season more often — but
they burn four roster spots and their misses are droppable, not tradeable. This is
not an argument for punting the top of the draft; it is an argument for where the
**middle** of your budget goes. The room's own history says the same thing: ~18 of these
exist per auction, so budgeting **$30-40 for three or four of them** is the whole play.

## 8. The 2026 slate

Same rule applied to `projections_2026_wk0.csv` (league-scored FFA). Positional
averages inside the 2026 pool: QB $9, RB $20, TE $10, WR $17.

| player | pos | team | FFA $ | pos. avg $ | proj pts | proj pos. rank | ADP |
|---|---|---|---|---|---|---|---|
| Caleb Williams | QB | CHI | 9.0 | 9.1 | 377.0 | 9.0 | 72.0 |
| Bo Nix | QB | DEN | 7.1 | 9.1 | 373.0 | 12.0 | 96.7 |
| Cam Skattebo | RB | NYG | 19.1 | 20.4 | 233.0 | 19.0 | 40.7 |
| Quinshon Judkins | RB | CLE | 17.5 | 20.4 | 222.0 | 21.0 | 50.6 |
| Bucky Irving | RB | TB | 15.5 | 20.4 | 230.0 | 20.0 | 50.0 |
| Bhayshul Tuten | RB | JAC | 14.1 | 20.4 | 208.0 | 23.0 | 60.4 |
| TreVeyon Henderson | RB | NE | 12.4 | 20.4 | 205.0 | 24.0 | 64.6 |
| Tucker Kraft | TE | GB | 9.2 | 10.0 | 177.0 | 8.0 | 73.8 |
| Sam LaPorta | TE | DET | 9.0 | 10.0 | 169.0 | 12.0 | 76.7 |
| George Kittle | TE | SF | 8.7 | 10.0 | 172.0 | 9.0 | 98.8 |
| Kyle Pitts | TE | ATL | 8.5 | 10.0 | 190.0 | 6.0 | 77.1 |
| Harold Fannin | TE | CLE | 6.8 | 10.0 | 195.0 | 5.0 | 77.7 |
| Travis Kelce | TE | KC | 5.9 | 10.0 | 180.0 | 7.0 | 102.0 |
| Mark Andrews | TE | BAL | 5.1 | 10.0 | 171.0 | 10.0 | 118.0 |
| Dallas Goedert | TE | PHI | 5.1 | 10.0 | 171.0 | 11.0 | 105.0 |
| Mike Evans | WR | SF | 13.0 | 16.9 | 206.0 | 29.0 | 75.6 |
| Luther Burden | WR | CHI | 12.6 | 16.9 | 211.0 | 25.0 | 66.4 |
| Jameson Williams | WR | DET | 11.4 | 16.9 | 222.0 | 21.0 | 63.8 |
| Rome Odunze | WR | CHI | 10.6 | 16.9 | 209.0 | 27.0 | 66.6 |
| DK Metcalf | WR | PIT | 8.8 | 16.9 | 197.0 | 30.0 | 89.2 |
| Courtland Sutton | WR | DEN | 8.5 | 16.9 | 206.0 | 28.0 | 90.1 |

21 names, ~$218 to buy the lot — obviously you take three or four,
not twenty. Two health warnings on this list:

- **FFA AAV is the national market, not this room.** Keeper inflation runs this
  league's prices above national AAV. Apply the rule to the draft tool's live Exp $:
  the test is *cheaper than the positional average of what this room is paying*, and
  it has to be re-checked live, not from this table.
- **The TE list is long this year** because 2026 TE pricing is unusually flat ($10
  average with eight bodies inside the starter tier). That is the market saying the
  position is a coin flip, and the class rule cannot break the tie.

## 9. Limits

- 26 events over 10 seasons. The 95% CI on the class rate is 9-20%; this is a real
  effect but not a precise one.
- The 50% price floor and the starter-tier cutoff were chosen after looking at the
  data. The sensitivity table above says they are not knife-edge, and the era split
  holds — but the honest forward test is the next two drafts, not this report.
- Everything here is priced pre-season. It says nothing about in-season acquisition,
  and it deliberately ignores keeper value, which is the *other* reason these buys are
  good (`LATE_BREAKOUTS.md` §5).

