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

- Wed: upload the FFA raw-stats file (DraftKings profile, with `rec`), paste the Subvertadown tables,
  press Preview on the next slate and read the health lines. Rebake My Team and Fan Picks (commands
  above) and push. Re-uploading FFA mid-week: press Preview again and rebake the two tabs.
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
