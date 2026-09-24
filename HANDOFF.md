# Handoff — state as of 2026-09-24 (week 3 Thursday)

## What changed 2026-09-24: Proj is the playing median; the bias correction reads played rows

Branch `median-proj`, awaiting merge. Read the 09-21/22 section after this one.

1. **Proj = the median of the player's distribution given he plays** (was the mean). The
   scoreboards we are ranked on, the sites' published accuracy and our own Accuracy page, are MAE
   over players who appeared, and MAE is minimised by the median. Misses are right-skewed, so the
   mean sits 1.1-1.3 PPR above the median. On 2025 played rows (n 5,887, walk-forward history in
   the run parquet) the mean point scored MAE 4.269, WORSE than FFA 4.190; the median point 4.123,
   better than FFA. Every send in weeks 1-3 carried the mean. Two reasons for the switch, both in
   the comment block above `p["mean_ev"]` in `sabersim_weekly.py`: the scoreboard, and a reader
   building a projection off ours wants the typical outcome for a player who suits up with the
   risk shown separately. New **Mean** column (after Median) = P(plays) x playing mean, the
   optimiser quantity. Floor/Median/Ceiling keep the inactive mixture; Note carries P(plays).
   `HEALTH["proj_kind"] = "playing_median"`. Stat lines still reconcile to Proj.
2. **The bias correction was measuring inactives.** It read the all-rows mean error, inactives
   scored 0 included: wk1 all-rows -0.55, played-only +0.11, the whole gap from 116 inactives we
   had left at 2-3 points. It helped MAE only by accident (a downward shift moved the mean point
   toward the median). It now reads `med_err_played` (median of actual - Median column over
   played rows, weighted by `n_played`), which the grader emits from this commit on together with
   `bias_played`. Grades made earlier lack the key and are skipped, so the first sends after the
   merge carry **no correction** until the next grade lands (health record says so). Same caps.
3. `tests/test_bias_correction.py` covers the new keys and the skip of old grades (12 pass).
   Dry run 2026-09-24 12:42 ET, ATL @ GB, 26 rows: Proj == Median on every non-OUT skill row,
   Mean > Proj on 21 of 22, Jayden Reed OUT at 0 everywhere.
4. Study landed alongside: `scripts/game_script_oracle_study.py` ->
   `outputs/reports/game_script_oracle_study.md`. Game scripts are not discrete, a perfect
   pregame call of a 5-way script is worth 2.2% MAE (QB 10%, TE nothing), a line-based pick
   LOSES 1.3-1.9%, reader break-even 54-64% accuracy vs ~30% achievable. Do not wire a script
   tweak into projections.

# Handoff — state as of 2026-09-22 (week 2 closed, week 3 opens today)

For picking the project up from another machine or the cloud console. Everything below is in this
repo; the only things that are NOT are the Model_Burke package (private repo `model-burke-private`,
or `pkg/` from the backup zip — see HOME_PC_RUNBOOK.md), the Odds API key file
(`data/odds_api_key.txt`, gitignored), and Claude's memory notes (local `.claude` folder; the home
PC gets them from the backup zip). This file carries the context those notes would otherwise hold.

## What changed 2026-09-21 / 22

Read this section first; it is the current state. Everything below it is older history kept for
context.

1. **The send is bias-corrected (#130, merged).** `scripts/bias_correction.py` reads the rolling
   error out of `docs/sabersim_accuracy.json` and `sabersim_weekly.py` adds it to every skill-position
   projection after `eval_dnp()`. It currently resolves to **-0.528 pts/player from 841 graded rows
   over weeks 1-2** — the model runs high, so the constant is negative. Out of sample (fit wk1, apply
   wk2) it moved MAE 3.609 -> 3.469 and bias -0.674 -> -0.304. It reorders nobody; it is a level
   shift. `--no-bias-cal` disables it and `HEALTH["bias_cal"]` reports what was applied.
   **Watch it settle**: the magnitude swung ~70% between the two weeks it is fit on, so treat the
   current value as provisional until weeks 3-4 are in.

2. **Week 2 is closed at 16/16 games.** Final: n 464, MAE 3.600, bias -0.611, Spearman 0.771,
   80% coverage 0.800. Season to date n 934, MAE 3.739, bias -0.494, Spearman 0.774, cov 0.813.
   On the same rows we beat FFA in both weeks (wk2 3.764 vs 3.929) and lose to DK in both
   (wk2 4.208 vs 4.081). The DK gap is calibration, not ranking — the orderings agree.

3. **The T-90 inactives question is answered, and the answer is not the one the probe was chasing.**
   Monday night (NYG @ LA, one game, a clean read) the official inactive list did not reach ANY
   feed until ~T-70: espn_league LAR 2->7 and NYG 1->8 between the 23:00 and 23:10 ticks, sleeper
   likewise. The T-80 send is therefore structurally too early to ever see it, matching the ~T-73
   seen on 2026-09-13. `espn_fantasy` does NOT lead — it was flat across the exact tick where the
   other two registered the drop. It is healthy otherwise (`field: true, via: kona_season,
   tried: []`), so the view question from 09-20 is settled; the latency premise is what failed.
   **What actually caught Puka Nacua that night was the nfl.com injury report plus sleeper's status
   field**, which had him OUT hours before the inactive list existed. Keep those two paths healthy;
   chasing a faster inactives feed is the lower-value thread.

4. **The OUT redistribution has a shape problem, not a size problem.** Nacua was correctly sent at
   0.00. His vacated share was spread across four Rams receivers by projection weight, and the game
   concentrated nearly all of it in the WR1 and the tight end:

   | player | sent | actual | err |
   |---|---|---|---|
   | Davante Adams | 17.41 | 39.50 | +22.09 |
   | Terrance Ferguson | 6.81 | 17.40 | +10.59 |
   | Konata Mumpfield | 4.22 | 2.20 | -2.02 |
   | Tutu Atwell | 3.33 | 2.90 | -0.43 |
   | Xavier Smith | 3.20 | 0.00 | -3.20 |

   Adams got +1.7 and needed something closer to +22, while the three small bumps all landed on
   players who came in UNDER their boosted numbers. That points at `SHARES` in `sabersim_weekly.py`:
   `s_top` too low and `s_rest` too high when the absent player is a genuine alpha. One game is not
   a fit — but it is a specific, testable hypothesis, and the backtest to settle it (re-score past
   OUT events under alternative share splits) is the highest-value open modelling item.

5. **The grader zero-fills, and you should know it when reading MAE.** Once BOTH teams of a game
   are in nflverse, a projected player with no box-score row is scored as an actual 0 — deliberate,
   and correct (no row = no snaps). But it means an inactive we sent at 0.00 grades as a free
   perfect row. Season-wide that is **40 of 934 rows (4.3%)** sitting at 0-vs-0. MNF reads MAE 3.791
   with them and 4.052 without. Do not confuse this with the ad-hoc analysis frames, which drop
   unmatched rows instead — conflating the two is easy and I did it once in conversation.

6. **The trade calculator's player means are now a fitted model (later on 09-22).** The
   `gm/players.py` blend (this season and last at games/(games+4)) was scored walk-forward on
   2019-25 against everything the 2012-26 weekly history can offer (`scripts/ros_player_study.py`,
   report `outputs/reports/ros_player_study.md`): it ran +0.55 high overall and +1.42 high on the
   top quarter of each position — it took two hot games at face value, which is exactly why it kept
   proposing Mahomes for Burrow (25.5 vs 21.0 a week). A per-position ridge on this season, the
   prior two, the career and FFA's next-week projection is 6% better on MAE (3.11 vs 3.32), lifts
   rank correlation 0.55 -> 0.57 and is unbiased; **age and archetype add nothing** (fifth null
   study for them), and FFA alone is worse than history alone. It is stored in `gm/ros_model.json`
   and the runtime needs only numpy. Under it Mahomes is 22.8 and Burrow 22.0 — under a point a
   week apart, a lateral move. Also fixed on the way: lost fumbles were charged twice (nflverse
   carries components and a total), and the mean-shrink constant is 1.0 for the ridge (the 0.80
   was solved for the blend; the ridge's 6.7 team spread plus the 5.1 uncertainty lands the 8.7
   target). Refit after each season with `--rebuild --fit-production`. The change also exposed a
   trade-search rule worth knowing: a forced cut took the lowest mean on the roster, which is
   usually the flat DST (5.59), so receiving a star for free could score as a loss; cuts now skip
   the last holder of any dedicated slot. With honest means fewer trades help both sides (the
   search cleared 111 stage-one candidates against 386 before), and "nothing" is a real answer.
   Later the same day: **NFL byes and the waiver wire are in the simulation.** Each score column
   is a league week, players sit out their bye, and a virtual waiver player per position (best five
   unrostered under the fitted model: RB ~7, WR ~10, TE ~9, QB ~20) fills any slot the roster
   cannot -- so a one-TE roster is charged the gap to a streamer for its bye, and bench players
   below waiver level are worth nothing. `gm_report.py --roster` prints the depth chart (pts/wk,
   bye, value over waiver, weeks started) and every proposal prints the per-week lineup delta.
   Glass Joe's bench receivers are at or below waiver; Irving and Henderson are the only backs
   above it, and week 11 takes five backs at once.
   Then, because "my backs are terrible" ignores WHY the bench is held: next-season keeper value
   is now priced from this season (cost = this year's price, $1 for a waiver pickup, + $5 bump;
   value = auction $ at the healthy level after a measured year of drift on a convex dollar curve),
   ESPN injury designations cap availability at measured rates (Out .49 / Doubtful .54 / Q .64;
   IR .10 assumed), forced cuts spare players with keeper surplus, and `--keeper-discount` defaults
   to 1. The `--roster` table prints healthy level, next-year cost, value, surplus and a "why held"
   tag. Its honest reading of the two stashes: Charbonnet (OUT, healthy ~8 ppg) is worth ~$6 next
   year against a ~$5 cost (the bump is now expected over the playoff race: $5 in, $3 out; Glass
   Joe at 38% expects $3.77) -- roughly break-even, +$1; Sampson (IR, ~$9 cost) is ~$5 under water
   even with young-dart option value. Irving (+$28) and Burrow (+$19) are the keepers.

7. **New: the `gm/` package** — a rest-of-season league simulator and trade finder for the ESPN
   keeper league, with `scripts/gm_report.py` as its CLI. It shares NO import path with the send
   pipeline in either direction. **See `GM_RUNBOOK.md`**, which carries its own setup, refresh order,
   measured constants and limitations. Next piece of work is named at the bottom of that file.

## What changed 2026-09-18

1. **Grader credited a backup with the starter's line (fixed).** `sabersim_grade.py` fell through to
   a last-name match for any row that missed on both player id and full name, so Kyle Allen (rostered,
   no snaps, no box-score row) was scored with Josh Allen's 40.8 in week 2. The fallback is now only
   for FFA-only rows with no nflverse id; a row that carries one and has no box-score row is a 0.
2. **K/DST rows reached the CSV with a blank Opp (fixed, both ends).** `context()` reads the opponent
   from the game-lines export, which carried 28 of 32 teams for week 2 — no BUF/DET, no CIN/HOU. The
   skill frame has a slate-based fill after the pipeline; K and DST did not, so both kickers in the
   Thursday send had no Opp. The grader's "has nflverse ingested this game" check requires Team and
   Opp, so it dropped the kickers and flagged Detroit @ Buffalo as awaiting box scores with the rest
   of the game already graded. `sabersim_weekly.py` now fills K/DST opp from the slate, and the
   grader recovers a blank Opp from the game's own skill rows so existing sends grade correctly.
   Week 2 went from MAE 6.762 / bias +3.55 / 24 rows to MAE 4.944 / bias +1.509 / 26 rows; week 1 is
   unchanged. Without the generator fix the Sunday CIN @ HOU send would have hit the same thing.
3. **Generator no longer needs legacy tzdata.** `sabersim_weekly.py` asked for `US/Eastern`, a legacy
   alias that full tzdata ships but slim images (including the Claude Code cloud console) do not, so
   the console recipe below died on `ZoneInfoNotFoundError` before reading a single file. All five
   uses now say `America/New_York` — the same zone, and what `sabersim_grade.py` already used.
   Verified on a container without the alias: output identical to the run before the change, bar the
   Generated stamp.

## SaberSim page showed the wrong "latest" file (fixed)

`docs/sabersim.html` picked the newest preview and the newest send with `b.name.localeCompare(a.name)`.
Published names are `<...>_<season>_<slate tag>_<MMDD>_<HHMM>.csv` with the slate tag BEFORE the date,
so a string sort orders by weekday name — `wk1` > `wed` > `thu` > `sun` > `mon` — not by time. The
Preview panel kept showing `thu08pm_0917_1918` after `sun01pm_0919_1721` was published, and "Download
the latest send" had been handing back `wk1_wed08pm_0909_1622`, the week-1 Wednesday file, all season.
Both sorts (and the sends/previews list) now use a `stamp()` parsed from the name; Jan-Jul is treated
as the season's second calendar year so the rollover sorts right.

## Game lines: why the table was missing games, and what changed

The week-2 blank-Opp bug traced back to `data/sabersim/game_lines_*.parquet` holding 78 of 272 games
for 2026 — no DET @ BUF and no CIN @ HOU in week 2, and almost nothing from week 8 on. Not a TNF/MNF
pattern: the missing games cover every kickoff slot.

`scripts/fetch_schedule_lines.py` pulls nflverse schedules, drops games with no `spread_line` /
`total_line`, and wrote the table with `if_exists="replace"`. For settled seasons that is exact
(1999-2025 is 100% covered). For the CURRENT season it is not: nflverse carries a line only while a
book has the game on the board, so a pull covers the next few weeks and nothing past them, and a game
taken off the board vanishes. Each replace therefore froze one snapshot and discarded everything else.
The committed parquet was exported in the preseason off look-ahead lines that have since come down —
today's live nflverse has only 48 priced 2026 games (weeks 1-3), FEWER than the parquet's 78, so
re-running the old script would have made the hole bigger, not smaller.

- The ingestion now UPSERTS on (season, week, game_type, team): the pull refreshes what it carries and
  leaves every other row alone. `--replace` restores the old rebuild if you ever want a clean purge.
- `--export-parquet` refreshes the two committed exports the model code reads, and refuses to write a
  file smaller than the one on disk unless `--force` — the whole bug class was silent shrinkage.
- `tests/test_game_lines_upsert.py` covers it: look-aheads survive a shrunken pull, new weeks get
  added, fresh lines win on conflict, settled history is untouched, and the shrink guard holds.
- Re-running it needs `db/nfl_odds.db`, so it is a home-PC job: `python scripts/fetch_schedule_lines.py
  --export-parquet`, then commit the parquets.

Impact was limited to `Opp`, because `sabersim_weekly.py` refreshes spread / total / implied totals
from live DK lines at send time. Offline runs (`--no-market`, no Odds API key — the cloud-console
recipe below) have no such backstop: teams absent from the parquet go into the model with NaN game
environment. Refreshing the parquets closes that. The DK refresh now also covers the K frame; nothing
reads K's context columns today, but it was the same omission that left K/DST without an Opp.

## What changed 2026-09-16 / 17

1. **Generator week bug (critical, fixed).** `sabersim_weekly.py` anchored weeks on the earliest
   event the Odds API still listed, and the API only lists unplayed games, so every run was "week 1".
   Week-1 sends were right by luck; the first week-2 run loaded the week-1 FFA file, K/DST override
   and opponents. The week now comes from `data/schedule_2026.csv` (week opens Tuesday, current until
   its last game is 4 h old). Verified: week-2 runs load the week-2 FFA file, week-2 K/DST, correct Opp.
2. **OUT path hardened.** Source order at send time: NFL.com injury report (Out/Doubtful → OUT,
   Questionable → Q; nflverse parquet fallback) → ESPN league injuries (no User-Agent; a browser UA
   gets 403 on Actions runners) → Sleeper players dump → FFA's own O/IR/SUS/PUP tag (new last resort)
   → manual `data/inactives/inactives_<S>_wk<W>.txt` (wins; `name, TEAM, active` overrides). The
   lineups log line and `data/sabersim/sent_log.json["lineups"]` record counts per source. Runner
   proof: preview runs #1650 / #1918 pulled ESPN 138–161 and Sleeper ~100 statuses.
   Remaining gap: official inactives post at T-90, feeds carry them ~T-73, the last send inside
   SaberSim's T-75 cutoff starts at T-80 — so sends carry the injury report, not surprise scratches.
3. **Q handling (unchanged, documented).** Generation only happens at T-80, so the report read is the
   final one. Q + DK props posted = plays, full projection, minus the DNP haircut if the last practice
   was DNP (WR .85 / QB .88 / RB .94). Q + no DK line while teammates are priced = DOUBT: P(plays)
   DNP .18 / LP .30 / FP .45, remainder redistributed.
4. **SaberSim page: Preview (no email) + data-pull health check.** Send card button and a per-slate
   button dispatch `force + dry_run`. The wrapper publishes the CSV to the private repo as
   `preview_*` (the graders glob `Burke_Model_Burke_*`, so previews are never graded; sent log
   untouched) with `<csv>.health.json` from the generator: FFA file rows/age + slate counts, DK game
   lines n/n slate games, DK props players priced + per-stat counts + no-line list, injury report
   source, lineup feeds by source, K/DST override hits, wrapper warnings. Real sends get the same file
   and a compact copy in the sent log. Page shows ✓/✗ lines and a download button.
   Local equivalent: `PUBLISH_DIR=<dir> MODEL_BURKE_PKG=<pkg> python scripts/sabersim_auto.py --force --dry-run [-- --no-market]`.
5. **Subvertadown parser** reads each table's week header (2..17 from week 2 on) instead of assuming
   1..17; the page recognises full-site pastes (and the site's glued `4.5Player/Team` line breaks).
   Week-2 paste parsed: 3,456 long rows, K/DST override 32+32.
6. **My Team tab** zeroes FFA O/IR/SUS/PUP and ESPN INJURY_RESERVE/OUT/SUSPENSION players, drops
   them from the wire and from the "my weakest" comparison. Rebake: `python scripts/my_team_tab.py --pull --board`.
7. **Fan Picks (new, shadow mode).** `docs/fan.html` (🙋 tab): every skill player's projected stat
   line for the week with ▲/▼ arrows, 10% of the baseline per press, unlimited presses (ten ▼ = zero).
   Submit → a `FAN1.<season>.<week>.<base64>` code; "Email it" opens the fan's mail app to the
   address in `FAN_TO` (top of the page's script). Nothing reads the arrows except the grader.
   - Bake weekly: `python scripts/sabersim_weekly.py <pkg> --all-games --no-market --out outputs/fan/proj_<S>_wk<W>.csv`
     then `python scripts/fan_proj_bake.py outputs/fan/proj_<S>_wk<W>.csv <S> <W>` → `docs/fan/proj_latest.json`
     + a stamped `proj_<S>_wk<W>_<bake_id>.json`; the code carries the bake id so a re-bake never
     changes what an earlier submission meant. Fan Picks numbers are the model WITHOUT the DK market
     blend (sends and previews have it on); the frozen file records what the fan saw.
   - Record: drop codes in `data/fan_adjustments/inbox/*.txt`, run `python scripts/fan_adjust_record.py --inbox`
     → `data/fan_adjustments/fan_adjustments_long.csv` (idempotent per fan + timestamp).
   - Grade: `python scripts/fan_grade.py` → `docs/fan/grade.json`, rendered at the bottom of fan.html
     (direction hit rate, error removed in DK points, by fan / stat / arrow count / boost-fade, graded
     rows, pending games). Runs in the daily grade of `sabersim_send.yml` and in `sabersim_grade.yml`.
     `--selftest` checks the math. No submissions recorded yet.

## Weekly routine (all from the SaberSim page unless noted)

- Wed: upload the FFA raw-stats file (DraftKings profile, with `rec`) and paste the Subvertadown
  tables. **That is the whole routine for the tabs**: each upload lands on main as a commit and the
  `Refresh tabs` workflow (`.github/workflows/refresh_tabs.yml`, driver `scripts/refresh_tabs.py`)
  re-pulls ESPN, parses the pastes, and rebakes the waiver board, My Team, Fan Picks, Props Watch
  and the gm league config for the week of the newest FFA file, then commits them with
  `docs/refresh_status.json` saying what ran. A Wednesday 18:00 UTC cron is the fallback. Locally
  the same thing is `python scripts/refresh_tabs.py --pkg <model_burke dir>`; `git pull` after an
  Actions run. Then press Preview on the next slate and read the health lines.
  Runner facts learned on 2026-09-22: the job runs Python 3.12 (3.11 rejects nested f-string
  quotes the tab scripts use); the ODDS_API_KEY secret is comma-separated and must be split into
  one key per line (the workflow does; props_watch.py accepts either); two runs minutes apart
  rebase onto each other and the newer bake wins. Actions logs can be read without gh: the git
  credential store holds a github.com token, and `scripts/secret_keys_check.py` shows the pattern
  (it verified the secret holds all six keys).
- Game day: sends fire at T-80 automatically (external cron → workflow_dispatch every 5 min). If no
  receipt by T-72, press Run on that slate. `data/sabersim/sent_log.json` shows what each send pulled.
- Morning after: the 8 am MDT tick grades sends (`docs/sabersim_accuracy.json`) and fan picks
  (`docs/fan/grade.json`); Grade now on the page does it sooner.

## Open items

- ~~T-90 inactives gap~~ **CLOSED 2026-09-22, negative result.** The probe did its job: the
  official list lands ~T-70 on every clean read so far (09-13 ~T-73, 09-21 between the 23:00 and
  23:10 ticks), which is AFTER the last send that clears SaberSim's T-75 cutoff. No free feed
  fixes that, because the league does not publish earlier. `espn_fantasy` was the last candidate
  and does not lead. `scripts/inactives_probe.py` can stay (it costs one step per tick and writes
  nothing), but treat this thread as finished unless a paid feed with a documented pre-T-90 flag
  is on the table. The working substitute is already in place and was proven on 09-21: the
  nfl.com injury report + sleeper status + the DOUBT mixture had Nacua at P(plays) 0.18 a full
  day early, so the send absorbed ~82% of the news before the inactive list existed.
- **`SHARES` redistribution shape (new, highest-value modelling item).** See item 4 of the
  2026-09-22 section. Re-score historical OUT events under alternative `s_top`/`s_rest` splits
  and see whether a more top-heavy split lowers MAE on the teammates of absent alphas.
- **Bias constant provisional.** -0.528 now; refit weekly and watch whether it stabilises across
  weeks 3-4 before trusting the magnitude.
- **Fan Picks is a game now (2026-09-22).** `docs/fan.html`: username kept on the device, a
  per-browser id inside every code, "Lock them in" posts username + code to a Google Form in the
  background (config `docs/fan/dropbox.json`; empty = the old email/copy flow), and a leaderboard
  (season, this week, who's in -- the last read live from the form's sheet). Score = error removed
  against the shipped projection, the same number as before. `scripts/fan_inbox_pull.py` reads the
  sheet into `data/fan_adjustments/inbox/dropbox.txt`, records (`fan_adjust_record.py`, now with a
  `fan_id` column) and grades (`fan_grade.py`, now with `by_fan_week`, `rank`, `best_call`); the
  refresh workflow (`fanpicks` step) and the daily grade both run it. The drop box exists
  (created 2026-09-22 through the owner's Google account, links in `docs/fan/dropbox.json`): the
  form "Fan Picks drop box" with Username + Code, published to anyone with the link, its response
  sheet shared read-only by link and read as gviz CSV. The sheet's CSV carries no CORS header, so
  the page's live "who's in" falls back to the baked `submissions.json`; empty submissions (0
  arrows, e.g. the setup test row) are skipped by the puller.
  **Identity is username + PIN** (added the same day): the page hashes them (SHA-256, first 20 hex)
  into `fan_id`, so the same person reproduces it on any device and a name-alike without the PIN
  becomes a separate player (shown with the id tail, e.g. "Clay (a1f3)"). The PIN never leaves the
  browser. Rows from before PINs (no fan_id) attach to the first PIN identity that claims the same
  name. "Forgot my PIN" = a new identity; merge it by hand in `data/fan_adjustments/aliases.json`
  ({old_fan_id: new_fan_id}). `device_id` (random per browser) is also recorded, for the mining.
  The page opens with a purpose blurb (people vs. the model; the FFA consensus it beats is CBS,
  ESPN, FantasyPros, FantasySharks, FFToday, numberFire, FleaFlicker, NFL.com) and shows the live
  scorecard from `docs/sabersim_accuracy.json` -- note it does NOT claim to beat the DraftKings
  lines, because the season grade has DK ahead on MAE (4.10 vs 4.23 on the rows they price). An
  optional email goes to a SECOND, private form ("Fan Picks contact list", not linked to a sheet)
  with username + fan_id, sent once per identity+address after a lock-in; owner reads it in the
  form's Responses tab. Links in `docs/fan/dropbox.json`.
  **Zero rows flatter the headline MAE (found 2026-09-23).** 36% of graded skill rows scored 0 --
  deep-bench players the DFS send carries at 0-3 pts, three quarters of whom never touched the
  ball -- and the whole edge over the eight-site blend on identical rows was those rows (model
  2.28 vs blend 2.65 on players who scored 0; on players who scored, blend 4.73 vs model 4.76).
  The grader now emits `mae_played` / `n_played` (players with a box-score row that week; a
  player who dressed and scored 0 still counts, an inactive we zeroed does not) at every level,
  and `--rows-out` dumps the graded rows for audits. The fan page ranks the model by
  `mae_played` in one table with the sites' published weekly MAE (`docs/fan/sites_accuracy.json`,
  pasted by hand as each week closes): wk1 4.47 (3rd of 8), wk2 4.09 (3rd, tied). The blurb says
  "ranks with the best", not "more accurate than". The SaberSim accuracy page still headlines the
  all-rows MAE; showing both there is the obvious follow-up.
  **Standings are a newspaper (2026-09-23)**, after the W.A.R. Street Journal in the user's
  WAR-street repo (newsprint #ddd5c3 + gradient texture, double rules, Playfair masthead "THE
  SUNDAY ORACLE", dateline, small-caps column heads). Rows = every model: the unedited model
  plus "<fan> + model" (the model with that fan's arrows applied), all scored on the same played
  player-games: variant error for a week = (model mae_played x n_played - fan's removed_pts) /
  n_played, season = pooled; W-L = weeks the arrows helped vs hurt (untouched weeks are ties).
  Columns: season standings, this week, who's in + best calls. Computed in the page from
  grade.json (by_fan_week) + sabersim_accuracy.json (weeks[].mae_played/n_played). Fans also pick
  the team they know best (fan_team column); its game leads the list, their players first.
  **The page is an arcade cabinet (2026-09-23).** `docs/fan/cabinet.jpg` (the user's "Football
  Expert" render, 1149x1369) is the stage; the CRT glass is at left 21.6% / top 23.5% / 54.4% x
  34.4% of the image, the control panel 64-73%. Attract mode on the glass (Press Start 2P; HIGH SCORES is the FULL standings crawling
  upward -- every model, Burke_v1 included -- with a two-line gap before the loop restarts so it
  reads as a list; PLAY NOW blinking); click -> `#stage.on`: `fit()` scales
  `#cab` so y 20.5%-75.5% / x 13%-87% fill the viewport and counter-zooms the glass content
  (`zoom = 1/scale`) so text draws at natural size; `⏏ cabinet` returns. A folded Sunday Oracle
  (`#fold`, top-4 lines) sits on the control panel; tap (or the 📰 bar button) opens the full
  paper in `#paperwrap`. Phones (`body.plain`, min side < 600px or portrait < 760) skip the
  cabinet entirely. Player cards use content-visibility:auto because repainting ~200 cards
  under the transform was the slow part. og:image is the cabinet, for sharing.
  **Phones keep the cabinet (2026-09-23 pm, portrait face 2026-09-23 eve).** `body.mobile` (width
  < 760 or short side < 600): landing = the wide cabinet.jpg fitted to the HEIGHT (rails cropped).
  Tapping the glass sets `#stage.on` and the face becomes a second render, the user's portrait
  cabinet `docs/fan/cabinet_phone.jpg` (1142x1377), scaled to COVER the viewport (`#mface`;
  marquee top, control panel bottom, rails just off the sides on a 430-wide phone). `fit()` writes
  the rendered image box and the glass rect as CSS vars on `#screen` (--fx/--fy/--fw/--fh and
  --gx/--gy/--gw/--gh; the glass is x 23.2-76.9%, y 19.5-82.8% of the picture, measured with PIL);
  `#mglass` (a flex column around `#crt` + the bar, display:contents on desktop) sits on that
  rect with the scanline overlay, `#eject` on the bezel above it, `#mpaper` (the folded Oracle
  that opens the standings) on the control panel at y 86.5%. `docs/fan/phone_harness.html` frames
  the page at 430x860, cache-busted, for checking in a desktop browser (the Chrome extension
  cannot resize a maximised window and refuses file:// URLs; scale the iframe with a CSS
  transform to see the whole phone). The old `body.plain` CSS remains but nothing sets it.
  *CRT effects on the phone glass (2026-09-24):* `#mglass` and `#scan` take `filter:url(#barrel)`, an
  inline SVG `feDisplacementMap` over a 128x128 map baked as a data-URI PNG (R = x shift, G = y shift,
  128 = none, growing with x*r^2; scale 16 = ~8px at the corners, 0 at the centre;
  `color-interpolation-filters="sRGB"` matters or the map is read in linear light). Plus a power-on
  animation (`crton`, scaleY from a line), a steps() flicker (`flick`), red/cyan text fringing on
  `#crt main`, glare and a 9s rolling band on `#scan::before/::after`, and `navigator.vibrate` ticks on
  the buttons. Desktop glass untouched. Not measured on a real phone: an SVG filter over a scrolling
  container re-rasterises on scroll; if it stutters, drop the filter from `#mglass` first.
- **Everything fans see is DraftKings points now, and arrows move in units (2026-09-24).**
  *Why DraftKings:* the sites' weekly accuracy we rank against (`docs/fan/sites_accuracy.json`, from
  Fantasy Football Analytics' DFS accuracy page) is graded in DraftKings points on QB/RB/WR/TE. Our
  `mae_played` is nflverse PPR (INT and fumble lost -2, no bonuses) AND includes kickers, so the
  Oracle was ranking two different measurements. The FFA page is blocked from the cloud container;
  the DraftKings basis rests on the search index's reading of it and on the owner's own account --
  **confirm the scoring and which players it counts on the page itself**; a different player pool
  (e.g. a projection floor) would still make the numbers incomparable.
  *What changed:* `scripts/dk_scoring.py` rescores the box score (actual) and the projected stat line
  (projection) in DraftKings points -- the stat line is scoring-neutral, so no retraining. The +3
  yardage bonuses are projected as expectations, 3 x P(yards >= line), from gammas fitted on
  2018-25 player-games (shapes 10.9 pass / 1.83 rush / 1.81 rec) and checked on our own 2026 wk1-2
  projections: 18.2 / 24.8 / 33.0 expected bonuses vs 18 / 20 / 33 observed. `sabersim_grade.py`
  now also emits `mae_vs_sites` / `n_vs_sites` / `bias_vs_sites` (DraftKings, skill only, played
  only) at every level; `mae_played` is untouched because the SaberSim page and the bias correction
  read it. The Oracle ranks Burke_v1 on `mae_vs_sites`.
  **Result: like for like, Burke_v1 is 5th of 8 in both weeks** (wk1 4.77 behind numberFire 4.41,
  FleaFlicker 4.44, ESPN 4.58, FantasySharks 4.64; wk2 4.30 behind FantasySharks 4.05, ESPN 4.08,
  numberFire 4.09, FleaFlicker 4.10), not the 3rd the PPR-with-kickers number showed. The blurb's
  "ranks with the best" was written on the old basis -- owner's call whether it stands.
  *Cards:* the card shows the stat line in DraftKings points (`dkPoints` in `docs/fan.html`, same
  constants as dk_scoring.py; `tests/test_dk_scoring.py` reads them out of the page and fails on
  drift). The bake now carries `fl` (projected fumbles lost); week 3's live bake was backfilled from
  `outputs/fan/proj_2026_wk3.csv`, frozen per-bake copies left alone. Each tap reprices the card,
  including the bonus odds (+20 rec yds on an 82-yard receiver is +2.0 linear and +0.6 of bonus).
  *Arrow rule u1:* a press = half a TD or INT, one catch, 10 rush/rec yards, 25 pass yards; ▼ stops
  at zero. Two presses is one more touchdown. Codes carry `"r": "u1"`; `scripts/fan_rules.py` is the
  one place a press becomes a number, used by both `fan_adjust_record.py` (new `rule`, `delta`
  columns; `pct` keeps meaning 10 x presses on old codes) and `fan_grade.py`. A code with no rule is
  scored under the old 10% rule forever, so Rob_Burke's week-3 submission means what it meant. A
  draft saved on a device under the old rule converts to units on load, by where the arrow LANDED
  (ten ▼ still means zero). Verified in Chromium: the page's gamma matches scipy to 1e-6, and every
  card, tap, floor, conversion and Oracle number matched an independent Python computation.
- **Fans can come back during the week, on any device (2026-09-24).**
  *Grading fix first, because it was losing picks:* the grader kept only a fan's latest set for the
  week, THEN dropped arrows placed after their game kicked off -- so locking in Thursday afternoon and
  again on Saturday threw away every Thursday-night arrow. `fan_rules.live_arrows` now decides per
  game: each game counts the last set locked in BEFORE it kicked off (a set that leaves a game out
  still removes its arrows). `fan_grade.py` uses it; tests cover the Thursday-then-Saturday case.
  *Same device:* drafts are keyed per bake, and a mid-week re-bake renumbers players, so every draft
  used to vanish on re-bake (there were six re-bakes on 09-22). The page now carries a draft forward
  by player id (`carryDraft`/`remap` in `docs/fan.html`). After a lock-in it keeps the locked code
  (`fanpicks_<S>_wk<W>_locked`) and the status line reads "locked in Thu 7:14 PM (12 arrows) — 3
  changes not locked in yet". Arrows on a game lock at kickoff: the bake now carries `kick_utc` per
  game (backfilled for week 3's live bake), buttons disable, and a timer re-renders at each kickoff.
  *Another device:* "Copy my link" (or the button after lock-in) gives `fan.html#p=FAN1...` -- the
  username, time, bake and arrows, NOT the PIN or the fingerprint made from it. Opening it on any
  device loads the picks (remapped if the page has been re-baked since), fills the username, and
  asks before replacing picks already there; a paste box takes a link or a FAN1 code. The PIN is
  still needed to lock in, so a forwarded link cannot enter picks as its owner.
  *Not built -- owner's decision:* signing in on a new device with username + PIN and finding your
  picks there with no link. That needs the page to read submissions back, and it can't read the
  Google sheet (no CORS); the repo only gets them at the daily pull, and more frequent commits to
  main re-trigger the send workflow. Options are in the 2026-09-24 conversation.
- **Quick picks = swipe mode (2026-09-24 pm, default for everyone).** `docs/fan.html` opens on a mode
  line under the identity card: **Quick picks** (one player at a time, headshot, projected DraftKings
  points, swipe right = boost / left = fade, buttons and arrow keys too, "skip" and "undo") or, one
  button away, **Full control** (the old per-stat ▲▼ list). `fanpicks_mode` in localStorage remembers
  it. A swipe is a pick keyed `"<i>:-1"` (stat index -1 = `fan_rules.SWIPE_STAT`) with +1/-1, and moves
  the player's whole line `SWIPE_PCT` = 10%; it rides in the code's `a` list like any arrow, so the
  Apps Script counts it and old codes are unaffected. `fan_adjust_record.py` expands it into one row
  per stat under rule **s1**; the grader and the fan model then see ordinary rows (`rule == "s1"` marks
  a swipe for the data mining). A player carries a swipe OR stat arrows, never both: swiping deletes his
  arrows, tapping a stat deletes his swipe (the Full-control card shows a "swiped" tag with undo).
  Deck order: the fan's team's players, then the opponent's, then everyone else by projected points;
  players under 3 DK points and games already kicked off are left out; it resumes at the first
  unjudged player. The identity card folds to one line once username + PIN are in, and the page
  scrolls to the deck. `fan_proj_bake.py` now writes `img` (nflverse roster `headshot_url`, cached a
  day in data/nflverse_cache) and `espn` (ESPN id, the card's fallback picture) per player; week 3's
  live bake was backfilled in place (356 of 366 have a picture). Verified in Chrome on the desktop
  cabinet: a drag boosts, the deck advances, counts update, headshots load; the phone layout and the
  Full-control tag after a swipe were checked by code only (the extension wedged) -- eyeball them.
  Skip exists on purpose (a forced guess on an unknown player is noise, not signal); remove `#dkskip`
  and the ArrowDown key to make every card a call.
  *Card v2 (same evening):* the photo sits behind the whole card at its own 4:3 proportions (118% wide,
  shoulders off the sides) under one full-card gradient, the name / matchup / projected points /
  Burke_v1 line / last-two-weeks chips in front. The bake writes `lw` = DraftKings points per earlier
  week (nflverse stats_player_week via dk_scoring.actual_frame; null = no box score). Trap fixed on
  the way: `#cab img { height:100vh }` was the cabinet picture's rule and matched every img in the
  cabinet, so headshots were blown up on desktop and display:none on phones -- it is `#cabimg` now.
  The deck re-measures the glass (`crt.clientHeight - 150`) when the screen lights and on resize.
- **Fan Picks API: DEPLOYED and live 2026-09-24 09:04 MT.** Bound script "Fan Picks API" on the response
  sheet, deployment v1 (execute as owner, access anyone); the /exec URL is `api` in
  `docs/fan/dropbox.json`. Verified from the live page: `?a=who` and `?a=mine` both 200 with
  `Access-Control-Allow-Origin: *`, no console errors, "who's in" showing the live row. Code changes
  go out as a NEW VERSION of the same deployment (Deploy > Manage deployments) so the URL holds.
  Built as: `apps_script/fanpicks_api.gs` is
  a read-only Apps Script web app on the drop-box response sheet: `?a=mine&f=<fingerprint>&s=&w=`
  (a fan's latest locked set for the week) and `?a=who&s=&w=` (names + arrow counts). The page asks it
  once username + PIN settle: a device with no picks loads the last lock-in ("Welcome back"), a device
  with different picks gets a "Load it" button, and "who's in" goes live (the sheet read never worked
  from a browser -- no CORS). Everything is gated on `"api"` in `docs/fan/dropbox.json`; empty = the
  page behaves as before. Setup: `apps_script/README.md`. Tested: the script under Node with stubbed
  Google services on real codes (8 checks), the page against a mocked API in Chromium (12 checks).
  *Fingerprints are no longer published:* `grade.json` used to list every fan's fan_id beside the
  name, which with this API would have let anyone read a leader's picks before kickoff. It now
  carries `pid` = `fan_rules.public_id(fan_id)` (sha256 of "fanpicks-public|" + fan_id, 16 hex); the
  page computes the same to find "you" (`publicId()`; a test pins the two). Residual: PINs are 4+
  digits, so a determined person can still brute-force one from a name; the response sheet is
  still readable by link (next step: Actions pull through the script with a token, then unshare).
- **The Oracle ranks Burke_v1 on ALL projected players, inactives included (owner's call, 2026-09-24).**
  `ORACLE_BASIS = "all"` in `docs/fan.html` reads `mae_vs_sites_all` (every skill player projected,
  an inactive scored 0) instead of the played-only `mae_vs_sites`. On it Burke_v1 is **1st of 8**
  (wk1 4.05, wk2 3.73; played-only was 5th, 4.77 / 4.30). Know what it rests on: an inactive we zeroed
  at T-80 grades as a perfect row, and against the FFA consensus those rows are MORE than the whole
  edge (18 inactives FFA's Saturday file still had at 3+ pts are worth -0.18 of a -0.13 gap; played
  only, the two are level at +0.02). It is also not necessarily the pool FFA grades the sites on --
  the consensus itself ranks 5th on our played pool, which suggests theirs is easier. The page copy
  was reworded to say what is counted (no "like for like", no "only players who took the field").
  Set `ORACLE_BASIS = "played"` and restore that copy to go back. The owner is weighing a real
  eight-site ranking from per-source projections (ffanalytics with sources kept separate), which
  would remove the need for either choice.
- **Kickers: Subvertadown publishes "Standard" (3/3/3/4/5 by distance) and "Decimal" (0.1/yd).** Our
  grading is Standard's brackets + 1 per PAT, no miss penalty. The pastes carry no label; their mean
  (8.46 over 96 kicker-weeks) matches our Standard (8.37 league mean 2023-25) but cannot rule out
  Decimal-with-miss-penalties (8.34). Copy from the Standard tab. Kickers are outside the sites
  comparison entirely (it is QB/RB/WR/TE). The grader's rule text calls K scoring "DK kicker
  scoring"; DraftKings classic has no kicker, so read that as Standard.
- **gm/ tests were red on main 2026-09-24: the FFA scrape workflow clobbered the week-3 upload.**
  The owner's raw-stats upload (commit 5852955, 09-22) was overwritten on Wed 09-23 10:00 UTC by
  `.github/workflows/ffa_weekly.yml`, whose R template writes ffanalytics' points-summary table
  (`first_name`/`last_name`/`points`, no stat columns) under the same filename. The 09-23 refresh
  then failed its board, fan and props steps ("not a stat-level projection file"). Fixed 09-24: the
  upload restored from 5852955 (200 tests green), and the scrape now exits if the week's file
  already exists and refuses to commit a file without `player`/`pass_yds` columns. The R script
  still needs the real stat-level recipe from the home PC before it is useful.
- Fan Picks are graded against THE SEND, not the frozen bake (changed 2026-09-20). The page shows a
  file baked days earlier, so scoring against it credits a fan with every point of error the news
  removed between bake and kickoff — reading the injury report scores as forecasting skill. Week 2
  made it concrete: Clay faded Nico Collins to zero for +16.99 of a +17.23 total, and the send
  already had Collins at 0.00 because he was ruled OUT while the bake still said 17.18. On the send
  basis those three arrows are correctly neutral, and his 24 graded arrows go from 50% / +9.52 to
  33% / -9.09. The mean baseline error also falls from 3.19 to 2.36 points, which is the same fact
  from the other side: the send is a much harder baseline to beat than the bake. fan_grade takes
  --sends (both workflows pass pkg/sends outputs/sabersim); --vs-bake restores the old basis for
  comparison. The decision rule is unchanged, same as Subvertadown's: several hundred graded
  player-stats with a consistent direction edge before any weight is considered.
- `data/news/wiki_2026-09.jsonl` (139 MB) is not in the repo: over GitHub's 100 MB limit and LFS is
  exhausted. Regenerable by the news collector, or move it as a Release asset like the odds DB.
- The Odds API key file holds 6 keys (~2.4k credits on 2026-09-16); the health line shows the
  balance of the single key a call used, not the total.

## Generating a projection CSV from the cloud console (no key file, no PC)

The generator runs OFFLINE when `data/odds_api_key.txt` is absent: the slate comes from
`data/schedule_2026.csv`, the DK lines/props blend is skipped, and everything else (FFA file, NFL.com
injury report, ESPN/Sleeper lineups, K/DST override, the model, the quantile layer) runs as usual.
Verified 2026-09-17: with the key hidden, the Thursday slate produced a 28-row CSV in ~2 min.

What the session needs: this repo, `pip install -r requirements-sabersim.txt`, and the Model_Burke
package (`git clone https://<PAT>@github.com/drburke-droid/model-burke-private.git`, then point the
positional argument at that clone, or set `MODEL_BURKE_PKG`). Then:

    python scripts/sabersim_weekly.py <pkg dir> --out outputs/sabersim/manual.csv        # next slate
    python scripts/sabersim_weekly.py <pkg dir> --kickoff 2026-09-20T17:00:00Z --out ...  # a specific slate
    python scripts/sabersim_weekly.py <pkg dir> --all-games --out ...                     # the whole week

Email the CSV to SaberSim yourself (address and timing in `SABERSIM_AUTOMATION.md`). If the page's
Run / Preview buttons work, prefer them: they include the DK blend and need nothing installed.
