# GM runbook: the league simulator and trade finder

Everything in `gm/` answers one question — **does this trade make me more likely to win my
league?** — by simulating the rest of the season rather than consulting a value chart. This is how
to run it on another machine and what the numbers mean.

The SaberSim send pipeline and this are unrelated. `gm/` imports nothing from it and nothing in
the send path imports `gm/`. Breaking one cannot break the other.

## Setup (~5 min, once)

```
git clone https://github.com/drburke-droid/NFL.git
cd NFL
pip install -r requirements-sabersim.txt
```

`gm/` needs only pandas, numpy and the standard library — no Model_Burke package, no Odds API key,
none of the send pipeline's inputs. If you can run `python scripts/gm_report.py --check`, you are
set up. The first run downloads the 2012-2024 nflverse weekly files (~25 MB) into
`data/nflverse_cache/`, which is gitignored; the current season and last are fetched fresh every run.

ESPN cookies are needed only to **refresh** league data, not to run the simulator on what is
already committed. See `ESPN_SETUP.md`; locally they go in `data/espn_cookies.json` (gitignored).

## Daily use

```
python scripts/gm_report.py                    # title odds for every team
python scripts/gm_report.py --trades           # + trades worth proposing
python scripts/gm_report.py --trades --keeper-discount 0     # this season only (default weights next season fully)
python scripts/gm_report.py --check            # what is stale and how to refresh it
python scripts/gm_report.py --roster           # my depth chart: pts/wk, bye, value over waiver, weeks started
```

Each proposal prints the players moving (pts/wk, P(plays), bye, value over the waiver wire) and
**your expected lineup delta for every remaining week**, so a trade that is +3 most weeks and -9
in week 11 shows exactly that. Byes and waiver pickups are in the simulation (below), so a
"lateral" trade between similar receivers reads as lateral: the deltas are small every week.

`--sims` raises the simulation count (default 20,000) if the numbers feel jumpy; `--seed` makes a
run reproducible. `--team <id>` views the league as another manager, which is how you check what a
counterparty should think of your offer.

## Refreshing after games or a roster move

Order matters — each step feeds the next.

| step | command | writes |
|---|---|---|
| 0. FFA weekly file | the Wednesday FFA upload on the SaberSim page | `data/ffanalytics/FFAn_weekly/raw_stats_<S>_wk<W>.csv` |
| 1. pull ESPN | Actions tab -> **Pull ESPN league** -> Run workflow | `outputs/espn_league.json`, `outputs/espn_league_state*.json`, drafts, standings |
| 2. rebuild the config | `python scripts/build_gm_league.py` | `gm/leagues/kuhn_2026.json` |
| 3. rebuild keeper values | `python scripts/predict_keepers.py` | `outputs/draft_tool/keepers_2026.js` |

Step 1 runs in GitHub Actions because ESPN is unreachable from some networks and the cookies are
already repo Secrets. It commits its output, so `git pull` afterwards.

**Step 0 is what the player model reads for the coming week.** Until next week's FFA file is
there, the current week's projection stands in and the report header says `ffa stale`. The
model was fitted with the same rule, so a stale read is a slightly weaker estimate, not a wrong one.

**The keeper board goes stale quietly.** It is built from a roster snapshot, so after any trade or
waiver claim it values the wrong team — it once had Josh Allen on a roster he had left. There is a
test for this (`test_the_board_matches_the_current_rosters`); run step 3 whenever rosters move.

## What the numbers mean

**P(title), P(playoffs)** come from simulating every remaining week: player scores drawn from
their own distributions, lineups set on projection and filtered by availability, the real schedule
played out, then the bracket. Not a rating.

**The win curve is the point.** A point a week is worth 0.018 title probability to the league
leader and 0.002 to the bottom team — nine times apart. That is why the same trade is genuinely
good for one side and bad for the other, and why trade output is always reported per team.

**Trade deltas are exact, not noisy.** Each player's random draws are keyed to the player, so
everyone who does not move draws the identical season either side of the trade. An empty trade
produces a delta of exactly 0.0000000000; anything non-zero is signal.

**`--keeper-discount` is a preference, not a fact.** 0 judges a trade on this season alone; 1
(the default since 2026-09-22, because that is how this owner actually reasons about a bench)
treats next season's keeper surplus as fully comparable. Dollars convert to title probability at
*each team's own* win-curve slope, measured by nudging that roster and re-simulating — a league
average would misprice exactly the teams that most need to sell. Keeper deltas are reported at any
setting; the knob only decides whether they enter the ranking.

**"Nothing clears the bar" is an answer.** At discount 1 no trade currently available to Glass Joe
helps both sides, because each costs about $7 of keeper surplus — roughly what its title gain is
worth. A win-now-only calculator cannot produce that finding.

## Calibration, and where the constants come from

Nothing here is a round number someone liked. Every constant is measured, and the tests refuse to
let them drift:

| constant | value | measured on |
|---|---|---|
| weekly team sd | 20.5 | 490 regular-season team-weeks, 2023-25 |
| true between-team sd | 8.2 | the same, with sampling noise removed |
| shrinkage prior | ~7 games | variance components (6.25) and direct regression (7.7) |
| player weekly sd | 0.383 x ppg + 2.23 | 1,154 player-seasons in *this league's* scoring |
| availability | 0.75 + 0.030 x ppg, cap 0.97 | availability rises steeply with projection |
| injury caps | Out .485 / Doubtful .543 / Questionable .644; IR .10 assumed | share of the remaining season played, 5,573 listings 2016-25 |
| year-over-year drift, sd | young +0.3..+0.7, old -0.5..-0.9; sd RB/WR 4, TE 3, QB 5 | players with 6+ games both seasons, 2012-25 |
| player mean | per-position ridge, `gm/ros_model.json` | 26,690 player-weeks 2013-25, walk-forward MAE 3.11 vs 3.31 for the old blend |
| mean shrinkage | 1.00 (ridge); 0.80 for the legacy blend | the ridge is already regressed: its 6.7 team spread + 5.1 uncertainty = 8.4 vs the 8.7 target |
| uncertainty about a team | 5.1 pts/wk | disagreement between two independent roster views |

The simulated league averages 124.2 points a week against an actual 126.4, between-team sd 8.72
against 8.68, within-team 22.7 against 21.7.

## What is NOT trustworthy yet

- **The probabilities have never been checked against outcomes.** The moments are calibrated; the
  probabilities are not. Three seasons of history is three champions from one league, which cannot
  calibrate a title probability. A chart of it would look like evidence without being any.
- **Roster churn is only half modelled.** Waiver pickups that fill a hole are in (the virtual
  waiver player); injuries that remove a starter for weeks, speculative adds and other teams'
  trades are not, so every team's odds are biased toward its current roster being permanent.
  Least harmful for a trade comparison, where both sides share the assumption.
- **The mean is fitted; availability is not.** `scripts/ros_player_study.py` scored every
  rest-of-season estimator walk-forward on 2019-25 (34,875 player-weeks since 2012, decision weeks
  2-12). The old games/(games+4) blend ran 0.55 high overall and 1.42 high on the top quarter of
  each position: it believed hot starts. A per-position ridge on this season, the prior two, the
  career and FFA's next-week projection is 6% better on MAE and unbiased; age and archetype added
  nothing (age +0.01 MAE, archetype +0.02), and FFA alone was worse than history alone. That model
  is what `gm/players.py` now uses. P(plays) is still the straight line above, fitted to nothing
  but the pooled rate.
- **K and DST are not differentiated.** Every kicker is 8.85 a week and every defence 5.59. They
  add the right noise without pretending to rank, which is roughly why nobody trades kickers.
- **Acceptance is "should", not "will".** The search finds trades a rational manager ought to take.

## Where the pieces live

```
gm/config.py     league schema + validation that rejects configs which would make a simulator
                 lie rather than crash
gm/simulate.py   the season simulator, trade evaluation, roster-limit enforcement
gm/players.py    per-player rest-of-season estimates in league scoring
gm/ros_model.json  the fitted ridge (coefficients, medians, gap flags) the estimates read
scripts/ros_player_study.py  the study that fits it: --rebuild --fit-production after each season
gm/keepers.py    next-season value: the dollar curve, the year-over-year spread, next_season_board
gm/trades.py     the two-stage league-wide search
scripts/gm_report.py      the CLI
scripts/build_gm_league.py  ESPN pull -> league config
tests/test_gm_*.py        118 tests; run `python -m pytest tests/ -k gm -q` (~4 min, the
                 trade search dominates)
```

**Next season is priced from this one, not from last year's board.** `keepers_2026.js` was the
board for keeping INTO 2026 and that decision is made. `gm/keepers.py:next_season_board` builds
next year's from the current rosters: cost = this year's auction price ($1 for any waiver pickup,
per league rule) + a $5 bump (the owner's real bump is not known until the standings settle);
value = the league's auction dollars at the player's *healthy* level (the fitted mean, which does
not know he is hurt), averaged over a measured year of drift and spread (young players +0.5 ppg,
everyone else -0.5 to -0.9, sd 4 at RB/WR). The dollar curve is fitted from the 2026 board against
the draft tool's projections and is convex -- RB 8 ppg is $1, 11 is $6, 14 is $29 -- which is
exactly why a cheap young player is worth more than the dollars at his expected level. The
`--roster` table prints healthy level, cost, value, surplus and a "why held" tag per player, so a
$1 stash is judged on the reason he is held rather than on this week's zero. A forced cut spares
players with keeper surplus (their `hold`, in weekly points at the chosen discount).

**Injury designations cap availability.** The ESPN status on each roster entry (carried into the
config by build_gm_league.py) caps P(plays) at the measured share of the remaining season played
by players listed with it: Out 0.485, Doubtful 0.543, Questionable 0.644 (5,573 listings 2016-25).
INJURY_RESERVE is an ESPN roster state with no NFL-report equivalent, so its 0.10 is an assumption.

**Byes and the waiver wire are modelled.** Each score column is a league week (`week_columns`),
so a player sits out his NFL bye in that column and the next man starts. Every team also carries
one virtual waiver player per position, priced at the average of the best five unrostered players
under the fitted model (K/DST at the flat values), who fills any slot the roster cannot -- the only
tight end's bye costs the gap to a streamer, not the whole slot. The same rule means a bench player
below the waiver level is worth nothing: the manager would pick up the free agent instead. Stage
one of the trade search ranks on the average of the per-week lineups with both effects in, so a
roster that loses five backs in week 11 is charged for it before anything is simulated.

**A forced cut never empties a dedicated slot.** Rosters are full, so an uneven trade drops
somebody, and the drop is the lowest expected contribution *that leaves every dedicated slot
filled*: a team taking on a receiver cuts a bench back, not its only defence. The flat K/DST
means (8.85 / 5.59) are usually the lowest on a roster, and the old rule cut them, which made
receiving a star for free score as a loss once the fitted means priced bench backs above the
defence.

Three league rules are encoded because they are not guessable: keeper cost is last year's basis
plus the **current owner's** inflation bump (so a player's price changes when traded), ESPN's
`acquisitionType` decides that basis (`TRADE` carries it, `ADD` resets to $1), and only the best
three keepers count. Rosters are full at 16, so any uneven trade forces a cut, and the cut is a
real cost the receiving side pays.

## Next piece of work

Done 2026-09-22: the player mean is the fitted ridge above. Next is availability — P(plays) is
the one player quantity still set by a rule rather than fitted, and the same frame (a player's
injury-report history is in `nflv_injuries`) could fit it walk-forward the same way.
