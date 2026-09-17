# Handoff — state as of 2026-09-17 (week 2)

For picking the project up from another machine or the cloud console. Everything below is in this
repo; the only things that are NOT are the Model_Burke package (private repo `model-burke-private`,
or `pkg/` from the backup zip — see HOME_PC_RUNBOOK.md), the Odds API key file
(`data/odds_api_key.txt`, gitignored), and Claude's memory notes (local `.claude` folder; the home
PC gets them from the backup zip). This file carries the context those notes would otherwise hold.

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

- T-90 inactives gap (item 2). `scripts/inactives_probe.py` samples feeds each tick T-100..T-60 into
  the Actions log; no faster free feed found (ESPN has no pregame inactives endpoint; SportsDataIO
  documents one but is paid).
- Fan Picks grading has no data yet; the decision rule is the same as Subvertadown's: several hundred
  graded player-stats with a consistent direction edge before any weight is considered.
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
