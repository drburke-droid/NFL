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
set up.

ESPN cookies are needed only to **refresh** league data, not to run the simulator on what is
already committed. See `ESPN_SETUP.md`; locally they go in `data/espn_cookies.json` (gitignored).

## Daily use

```
python scripts/gm_report.py                    # title odds for every team
python scripts/gm_report.py --trades           # + trades worth proposing
python scripts/gm_report.py --trades --keeper-discount 1     # judge trades on next season too
python scripts/gm_report.py --check            # what is stale and how to refresh it
```

`--sims` raises the simulation count (default 20,000) if the numbers feel jumpy; `--seed` makes a
run reproducible. `--team <id>` views the league as another manager, which is how you check what a
counterparty should think of your offer.

## Refreshing after games or a roster move

Order matters — each step feeds the next.

| step | command | writes |
|---|---|---|
| 1. pull ESPN | Actions tab -> **Pull ESPN league** -> Run workflow | `outputs/espn_league.json`, `outputs/espn_league_state*.json`, drafts, standings |
| 2. rebuild the config | `python scripts/build_gm_league.py` | `gm/leagues/kuhn_2026.json` |
| 3. rebuild keeper values | `python scripts/predict_keepers.py` | `outputs/draft_tool/keepers_2026.js` |

Step 1 runs in GitHub Actions because ESPN is unreachable from some networks and the cookies are
already repo Secrets. It commits its output, so `git pull` afterwards.

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
| mean shrinkage | 0.80 | re-solved to match the two moments above |
| uncertainty about a team | 5.1 pts/wk | disagreement between two independent roster views |

The simulated league averages 124.2 points a week against an actual 126.4, between-team sd 8.72
against 8.68, within-team 22.7 against 21.7.

## What is NOT trustworthy yet

- **The probabilities have never been checked against outcomes.** The moments are calibrated; the
  probabilities are not. Three seasons of history is three champions from one league, which cannot
  calibrate a title probability. A chart of it would look like evidence without being any.
- **Roster churn is not modelled.** Injuries, waiver adds and other teams' trades over thirteen
  weeks are all absent, so every team's odds are biased toward its current roster being permanent.
  Least harmful for a trade comparison, where both sides share the assumption.
- **Estimates lean on two games.** The rest-of-season analysis found a single next-week projection
  beats to-date scoring outright (MAE 2.645 vs 2.746 on 2025), and `gm/players.py` does not use it
  — it is the fallback that analysis named, not the better thing it found. Wiring the walk-forward
  pipeline in per player is the most likely fix for the one roster that reads 2.1 sd above the
  field.
- **K and DST are not differentiated.** Every kicker is 8.85 a week and every defence 5.59. They
  add the right noise without pretending to rank, which is roughly why nobody trades kickers.
- **Acceptance is "should", not "will".** The search finds trades a rational manager ought to take.

## Where the pieces live

```
gm/config.py     league schema + validation that rejects configs which would make a simulator
                 lie rather than crash
gm/simulate.py   the season simulator, trade evaluation, roster-limit enforcement
gm/players.py    per-player rest-of-season estimates in league scoring
gm/keepers.py    next-season value, reading the board predict_keepers.py builds
gm/trades.py     the two-stage league-wide search
scripts/gm_report.py      the CLI
scripts/build_gm_league.py  ESPN pull -> league config
tests/test_gm_*.py        118 tests; run `python -m pytest tests/ -k gm -q` (~4 min, the
                 trade search dominates)
```

Three league rules are encoded because they are not guessable: keeper cost is last year's basis
plus the **current owner's** inflation bump (so a player's price changes when traded), ESPN's
`acquisitionType` decides that basis (`TRADE` carries it, `ADD` resets to $1), and only the best
three keepers count. Rosters are full at 16, so any uneven trade forces a cut, and the cut is a
real cost the receiving side pays.

## Next piece of work

Replace the to-date player means with the walk-forward projection, which measured better and is
the most likely fix for the roster that currently takes 46% of titles.
