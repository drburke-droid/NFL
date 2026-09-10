# Unattended SaberSim sends — setup

What runs: `.github/workflows/sabersim_send.yml` fires every 15 minutes during game windows.
Each tick costs ~20 s and nothing else unless the next slate's earliest kickoff is 62-95 minutes
away and unsent. Then it fetches the private Model_Burke package, runs `scripts/sabersim_weekly.py`
(live ESPN/Sleeper lineups, DK lines, inactives at T-90 are already posted), and emails the CSV
from your Gmail to analysts+robert@sabersim.com with you in CC. `scripts/sabersim_auto.py` is the
wrapper; it works the same from Task Scheduler on a PC if you ever prefer that.

Why not GitHub Pages: Pages serves static files only. It cannot run Python or send mail. Actions
can, and on a public repo the minutes are free and unlimited.

## One-time setup (about 20 minutes)

1. **Private repo for the model code.** Create `model-burke-private` (private). Unzip
   `Downloads/model_burke_pkg.zip` and push so the repo root contains `model_burke/` (the
   `pkg/` folder's contents). Only the code is needed, not `data/sample_input.csv.gz`.
   This keeps the model out of the public NFL repo; the workflow checks it out at run time.
2. **Fine-grained personal access token** (GitHub → Settings → Developer settings → Fine-grained
   tokens): repository access = `model-burke-private` only, permission Contents: Read.
3. **Gmail App Password**: Google Account → Security → 2-Step Verification (must be on) → App
   passwords → create one for "Mail". 16 characters.
4. **Secrets** in the NFL repo (Settings → Secrets and variables → Actions → New repository secret):

   | secret | value |
   |---|---|
   | `ODDS_API_KEY` | one key from `data/odds_api_key.txt` (events endpoint is free; DK props ~6 credits/game/send) |
   | `SMTP_USER` | your Gmail address |
   | `SMTP_PASS` | the App Password |
   | `MAIL_CC` | your address, so every send lands in your inbox too |
   | `MODEL_BURKE_REPO` | `drburke-droid/model-burke-private` |
   | `MODEL_BURKE_TOKEN` | the fine-grained token |

5. **Test without sending**: Actions tab → "SaberSim send" → Run workflow → `dry_run=true`,
   `force=true`. The log shows a JSON line with the row count. Then once with `dry_run=false`
   to confirm the email arrives (send it to yourself first by temporarily setting a repo
   variable `MAIL_TO` if you want; the wrapper reads `MAIL_TO` from env).
6. **Commit and push** the workflow, wrapper and this file. Scheduled runs start on the next tick.

## Weekly inputs that still need a human

| input | where | if missing |
|---|---|---|
| FFA weekly raw stats `raw_stats_2026_wkN.csv` | `data/ffanalytics/FFAn_weekly/` (your R scrape, Wednesday). `ffa_weekly.yml` + `scripts/ffa_weekly_scrape.R` are an untested template for running it in Actions. | the send FAILS loudly and GitHub emails you; nothing is sent |
| K/DST paste `data/kdst/paste_2026_wkN.txt` → `parse_kdst_paste.py` | optional | generator falls back to the FFA-scored K/DST line (already supported) |
| Manual inactives `data/inactives/inactives_2026_wkN.txt` | optional override | live ESPN + Sleeper statuses are used |

Push the FFA file by Wednesday night and the rest is hands-off.

## Timing and failure modes

- **Cron drift.** GitHub may delay scheduled runs by 5-30 min under load. The 62-95 minute window
  is 33 minutes wide and ticks come every 15 minutes, so one tick lands in the window even with
  a 15-minute delay. Worst case the send goes at T-62 instead of T-75. If you ever want exact
  timing, a free cron-job.org job hitting the workflow_dispatch API at T-80 is the upgrade.
- **Slate grouping** is the generator's: games within 90 minutes of the earliest upcoming kickoff.
  Sunday 1 pm and 4:05/4:25 are separate sends; SNF and MNF are their own.
- **Double sends are prevented** by `data/sabersim/sent_log.json` (committed after each send).
  Deleting a key there re-arms that slate.
- **Manual fallback from your phone**: GitHub mobile app → Actions → Run workflow → force=true.
- **Credits**: each real send pulls DK props for the slate's games (~6 credits/game). A full
  week is ~100-150 credits, the same as the manual routine; the 15-minute ticks cost none.
- **Public logs**: the workflow prints only the JSON summary lines. Generator output goes to a
  local log that is not uploaded, and no CSV is committed or attached as an artifact.
- **Nothing is auto-merged or retrained.** The model is exactly the one you have been sending;
  the workflow only runs it on time.

## External trigger (required — GitHub's cron never fired for this repo)

GitHub Actions never produced a scheduled run for this workflow, so the ticks come from cron-job.org
(free) calling the workflow_dispatch endpoint. The window check inside the workflow still decides
whether anything is sent, so extra ticks are harmless.

1. cronjob.org → Sign up (free) → Cronjobs → Create cronjob.
2. Title: `SaberSim tick`. URL: `https://api.github.com/repos/drburke-droid/NFL/actions/workflows/sabersim_send.yml/dispatches`
3. Execution schedule: Every 5 minutes.
4. Advanced tab:
   - Request method: POST
   - Headers: `Authorization: Bearer <the sabersim-page token>` · `Accept: application/vnd.github+json` · `Content-Type: application/json`
   - Request body: `{"ref":"main"}`
   - Treat 204 as success (it is the normal response).
5. Save, then press "Test run": the Actions tab should show a new `workflow_dispatch` run within seconds.

Timing: a tick at T-90..T-72 starts the run; the CSV lands ~2.5 min later. cron-job.org fires within seconds
of the minute, so the T-88 tick is the usual one and the email arrives around T-85.
