# Late Breakouts — finding the $1/undrafted players who finish VORP-positive

**The question.** Every season a handful of players go undrafted (or for $1-2) and
finish the year above replacement — then command a real auction bid the next
spring. Puka Nacua 2023: not even in the FFA top-200, finished **+108 VORP**, FFA
AAV the next draft **$40**. If one of your end-of-auction bench darts hits, you get
the season value *plus* a nearly-free keeper. This study quantifies how often that
happens, what predicts it, and what a dart is actually worth.

Scripts: `late_breakout_study.py` (dataset + base rates), `late_breakout_variants*.py`
(model design search), `late_breakout_final.py` (validation + 2026 scores).
Outputs: `late_breakout_2026.csv`, DB table `nflv_late_breakout`.

## Definitions

- **Universe** — every rostered QB/RB/WR/TE 2016-2025 who was in the FFA file,
  played the prior year, or was a rookie (6,783 player-seasons).
- **Price** — FFA preseason ADP + AAV (2016-2025). Validated against this league's
  actual clearing bids (2023-25, n=404 matched): **Spearman 0.90 (AAV), −0.85 (ADP)**.
  The proxy is sound, and players in the "cheap pool" below really did cost ~$1 here
  (median matched bid $1).
- **Cheap pool** — ADP > 120 (or absent from FFA entirely) AND AAV ≤ $2.
  ~520/season; this is the end-of-auction dart zone.
- **Hit** — season total league points (PPR, 6-pt pass TD, −1 INT) above the
  replacement starter: QB13 / RB27 / WR35 / TE13 for the 12-team roster. VORP > 0.

## 1. Base rates — P(VORP-positive | draft-day price)

| FFA ADP bucket | n | P(hit) | mean VORP |
|---|---|---|---|
| 1-36 | 335 | **76.7%** | +58 |
| 37-72 | 341 | 57.2% | +9 |
| 73-120 | 499 | 36.3% | −28 |
| 121-168 | 666 | **20.7%** | −59 |
| 169+ | 162 | 13.6% | −84 |
| not in FFA top-200 | 4,780 | **0.9%** | −156 |

By position, late-round value survives best at **TE (24% at ADP 169+)** and WR
(19%); it dies at QB (3% past 169 — a cheap QB hit is just a streaming job win).
Roughly **10-12 players per season** league-wide go from ~$1/undrafted to
VORP-positive. The cliff from 0.9% (unlisted) to 16.6% (listed but ADP 121+) is
the single most important number in this study: *being anywhere in the consensus
top-200 is already a 7x lift.*

## 2. The re-pricing mechanism is real (and big)

Cheap-pool players, next draft (2016-2024):

| | n | in FFA next year | median next AAV | p75 next AAV |
|---|---|---|---|---|
| missed (VORP ≤ 0) | 4,565 | 9% | $2.0 | $3.2 |
| **hit (VORP > 0)** | 109 | **93%** | **$5.8** | **$11.9** |

A hit almost always re-enters the drafted pool, and the right tail is where the
keeper value lives (Nacua $40, Kyren Williams $41, James Robinson $22 next-year AAV).

## 3. What separates hits inside the cheap pool (univariate lifts)

Cheap-pool base rate 2.3%. Signals ranked by lift:

| signal | n | P(hit) | lift |
|---|---|---|---|
| still listed in FFA (ADP 121+) | 469 | 16.6% | **7.4x** |
| prior-year target share ≥ 12% | 270 | 15.2% | **6.7x** |
| rookie with R1-2 draft capital | 98 | 11.2% | **5.0x** |
| year 2-3, drafted R1-2 | 128 | 8.6% | 3.8x |
| prior snap% ≥ 40 | 1,177 | 5.4% | 2.4x |
| prior PPG 5-9 (fringe role, not zero) | 817 | 5.3% | 2.3x |
| vacated backfield (≥100 carries) | 794 | 4.5% | 2.0x |
| H2 snap% climb ≥ +10 (late-season surge) | 972 | 4.1% | 1.8x |
| vacated targets ≥ 80 | 1,235 | 3.9% | 1.7x |
| — team change | 1,406 | 1.7% | 0.76x |
| — rookie day-3/UDFA | 1,253 | 0.5% | **0.21x** |
| — no prior production (PPG < 3) | 3,193 | 1.0% | 0.46x |

Read this as a two-stage probability: **P(breakout) ≈ P(role opens) × P(talent
seizes it)**. The positive signals are all one or the other — an opening (vacated
touches, H2 snap climb, fringe role already established) times a talent prior
(draft capital, target share earned per snap). The anti-signals matter as much:
*day-3 rookies and guys with zero prior involvement are dead darts* (that's where
the 0.9% base rate hides), and a team change alone is a *negative*.

## 4. Model — what a walk-forward learner adds (and honestly doesn't)

LightGBM P(hit), trained on all players, scored on the skill-position cheap pool,
walk-forward 2019→2025 (never sees the future). AUC **0.83**, well calibrated:

| blend score bucket | n | predicted | observed |
|---|---|---|---|
| bottom 50% | 1,627 | 0.4% | 0.5% |
| 90-97th pct | 226 | 5.9% | 8.8% |
| top 3% | 98 | **25.7%** | **25.5%** |

Precision in the top 10 darts per season:

| ranking of the cheap pool | p@10 | notes |
|---|---|---|
| random dart | 2.6% | mean VORP −130, next AAV $0.43 |
| model prob alone | ~11% | over-trusts rookie capital |
| **model + projection-points blend** | **27.1%** | best VORP@10 (−48), youngest slate |
| FFA projected points alone | 28.6% | statistically indistinguishable from blend |

The honest read: **identification is worth ~10x over a random dart, but at top-10
depth a good projection consensus already captures nearly all of it.** The blend's
real advantages: it surfaces a *younger* top-10 (47-50% ≤2 yrs exp vs 34% for
FFA-order → more keeper-eligible), higher next-year AAV per pick, and it works in
years/players where FFA isn't published yet (i.e., right now for 2026).

Top features: draft_pick, FFA ceiling−points spread, projected points, age,
prior PPG, prior CV, H2 snap climb, targets/game, snap%, uncertainty.

**The market got sharper.** 2019-21 top-10 darts included D.K. Metcalf, A.J. Brown,
Mark Andrews, Darren Waller (7 hits in 2019 alone). 2022-25 the same process found
1-2/year — players like that no longer fall past ADP 120. What still falls through:
the **camp risers** (Nacua-type) whose signal existed only in August beat-writer
reports, preseason snaps with the 1s, and depth-chart moves — none of which are in
this dataset. That's the identified data gap: an August depth-chart/camp-news
ingest would attack exactly the residual the historical features can't reach.

## 5. Keeper economics — why the dart is still +EV

League empirics (`espn_drafts.csv`, keepers matched to prior-year bids, n=52):
keeper cost escalation was **median +$4 (2024) and +$0 (2025)** — a $1 dart that
hits is kept the next year for ~$1-6 while carrying a market price of $6-40.

Per-pick expected keeper surplus, E[max(0, next-year AAV − keep cost)]:

| strategy | cost | P(hit) | surplus/pick | surplus per $ |
|---|---|---|---|---|
| random $1-2 dart | $1.5 | 2.6% | $0.15 | 0.1x |
| mid-tier bench vet ($3-8, ADP 121-168) | $5.5 | 26.6% | $2.05 | 0.37x |
| **model top-10 dart** | $1 | ~25% | **~$1.50** | **1.5x** |

Portfolio math for the bench: with per-slot P(hit) ≈ 0.25 on guided darts,
**P(≥1 hit across 4 darts) = 1 − 0.75⁴ ≈ 68%** (vs 10% for 4 random darts). The
optimal bench under these numbers is a *mix*: 2-3 mid-tier vets (better in-season
VORP, decent keeper odds at moderate cost) + 2-3 guided $1 darts (worse expected
season value, but the best surplus-per-dollar and the only route to a Nacua-class
keeper). Never spend a bench slot on a day-3 rookie or a zero-involvement player —
that's paying $1 for a 0.5% ticket when 11-25% tickets cost the same.

## 6. 2026 provisional darts (re-rank when August ADP publishes)

Pool: expected league price ≤ $2 (price-anchor model) and outside the top-120 by
projected points; 426 players scored. `exp_price = 0` means no league bid
precedent — verify the room actually lets them go cheap. Top young keeper darts:

| player | pos | team | exp | why | P(hit) |
|---|---|---|---|---|---|
| Omar Cooper Jr. | WR | NYJ | R | R1 capital, thin WR room | .37 |
| Germie Bernard | WR | PIT | R | R2, vacated targets | .36 |
| Matthew Golden | WR | GB | 1 | R1, 153 vacated targets | .34 |
| Xavier Legette | WR | CAR | 2 | R1, 68% snaps, 236 vacated carries | .34 |
| Bhayshul Tuten | RB | JAX | 1 | 260 vacated carries | .33 |
| Adonai Mitchell | WR | NYJ | 2 | R2, 114 vacated targets | .33 |
| Tre Harris | WR | LAC | 1 | R2, 152 vacated targets | .31 |
| Elijah Arroyo | TE | SEA | 1 | R2 TE, 229 vacated carries context | .30 |
| Travis Hunter | WR | JAX | 1 | R1 capital, role expansion | .27 |

Veteran cheap flags (not keeper plays, but VORP darts): Jerry Jeudy (85% snaps,
20% target share, H2 surge), Chig Okonkwo, Rashid Shaheed, Pat Freiermuth.

*P(hit) here is calibrated on the historical blend; treat >0.30 as "top-3%-of-pool
grade" (~25% historical), not a literal probability, until August prices confirm
pool membership.*

## 7. News/attention buzz — the camp-riser signal (added after the base study)

The base study's blind spot was August: Puka-class breakouts whose only pre-draft
signal was camp buzz. We ingested two attention sources:

- **Wikipedia pageviews** (`fetch_wiki_buzz.py` → `nflv_wiki_buzz`): monthly views
  per player back to 2015-07; 2,049 of 2,313 study players resolved (85% of
  player-seasons covered).
- **GDELT news volume + tone** (`fetch_gdelt_news.py` → `nflv_gdelt_news`): weekly
  article counts + avg tone, 2017+; heavily rate-limited, fetched overnight for
  the 570 highest-value players (darts, hits, control misses).

Features per player-season: `aug_views` (August pageviews — camp + preseason,
pre-fantasy-draft), `buzz_spike` = log((aug+10)/(median Jan-May +10)), and their
within-season percentiles.

**Validation (walk-forward, same protocol as §4):**
- August attention **level** is monotonic at the top of the cheap pool: deciles
  8-10 hit at 1.8-2.1x base; the bottom four deciles are near-dead (0.1-0.4x).
- Adding buzz to the classifier: **AUC 0.851 → 0.872, clf p@10 0.100 → 0.157**.
  Blend p@5 **0.257 → 0.286** (p@10 unchanged — the FFA component already covers
  the listed players; buzz's edge is concentrated in the unlisted region).
- The smoking gun: **Puka Nacua 2023 ranked #15 of 446** in the cheap pool by the
  buzz-augmented classifier (Aug-2023 spike = 98th percentile) — invisible to
  every prior feature set. Kyren '23 (76th pctl spike) and ARSB '21 (68th) also lit up.

**Workflow:** buzz features are now in the production model. August 2026 doesn't
exist yet, so current 2026 scores use buzz=NaN; **in draft week run
`python scripts/fetch_wiki_buzz.py refresh` then `late_breakout_final.py` and
`build_draft_tool.py`** — partial-August dailies are rate-scaled to a full month
and the dart board picks up camp risers automatically.

## 8. Does buzz generalize to PRICED players? Tested — no. (`buzz_priced_players.py`)

Same buzz features, applied to the drafted pool (FFA ADP ≤ 120; 1,077 player-seasons,
92% buzz coverage), outcomes measured as VORP vs same-season/bucket/position peers and
actual − projected points, attention normalized within season × ADP bucket:

- **Spike vs own baseline** (the dart signal): dead. Spearman **−0.01** vs peer-relative
  VORP (p=0.67), −0.04 vs projection residual. No quintile pattern in any bucket.
- **Attention level**: +0.05 vs VORP residual (p=0.10, n.s.); +0.089 vs projection
  residual (p=0.003) — statistically real but tiny, and it reverses sign in the
  ADP 73-120 bucket. Not actionable.

**Why the asymmetry:** for an unlisted $1 player, a big August has essentially one
cause — he's seizing a job in camp (Puka). For a priced player, August attention is
ambiguous: the top-decile spikes are a mix of genuine risers (Chase Brown '24 +150
VORP-Δ) and holdouts, injury sagas and hype (Jonathan Taylor '23 trade demand −67,
Dalvin Cook '23 release −149, Jonathon Brooks '24 injury −98). Hype and havoc look
identical in pageviews, so the signal's meaning is conditional on price tier.

**Conclusion:** buzz stays in the late-breakout dart model only; it must NOT be wired
into the main board values or Exp $. Possible future disambiguator: GDELT article
*tone* alongside volume (positive camp reports vs negative injury/contract news) —
current GDELT coverage (570 selected players) is too thin for the priced pool.

## Caveats

- FFA ADP/AAV missing before 2016; 2015 partial. Study window 2016-2025.
- 2025 cheap hits (e.g. Michael Wilson, Alec Pierce) have no 2026 FFA AAV yet, so
  re-pricing stats end at 2024 cohorts.
- VORP uses full-season totals — a mid-season waiver pickup counts as a "hit" even
  if a drafter might have dropped him in September. This inflates base rates
  slightly relative to "hit while on YOUR roster"; stash discipline matters.
- The 2026 pool uses our own price anchor + projections in place of unpublished
  market data; re-run `late_breakout_final.py` after the August FFA/ADP refresh.
