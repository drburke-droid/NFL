# Home-PC runbook: regenerate and send the SaberSim CSV by hand

Use this when the automated send did not arrive (no receipt in Gmail by T-72) and the Pages
button also failed. Everything the generator needs is in this repo except three things that
live in the backup zip: the Model_Burke package, the Odds API key file, and Claude's memory notes.

## Fallback order (try in this order)

1. **Pages button** (no PC needed): open the 🏈 SaberSim tab on the draft-tool site, paste the
   `sabersim-page` token, press **Run now & email**. It dispatches the GitHub workflow, which
   emails SaberSim and CCs you. Takes ~3 min. If the run fails, the Actions log says why.
2. **Re-push trigger**: any push to `main` re-runs the window check (empty commit is enough).
3. **This runbook**: generate locally and email the CSV yourself.

## One-time setup on the home PC (~15 min)

1. Install Git and Python 3.11 or newer (3.11-3.14 all tested).
2. Clone to the SAME path as the work PC so Claude's memory notes attach to the project:
   ```
   git clone https://github.com/drburke-droid/NFL.git C:\Users\<you>\Documents\GitHub\NFL
   cd C:\Users\<you>\Documents\GitHub\NFL
   pip install -r requirements-sabersim.txt
   ```
3. Unzip `sabersim_home_backup_<date>.zip` (from the work PC's Downloads, or a private cloud
   copy) and place its contents:

   | zip folder | goes to | why |
   |---|---|---|
   | `pkg/` | `C:\Users\<you>\Documents\GitHub\NFL\pkg\` | the Model_Burke package (`pkg/model_burke/`) |
   | `data/odds_api_key.txt` | `NFL\data\odds_api_key.txt` | Odds API keys, one per line (gitignored, never commit) |
   | `data/props_frames/lines_cache.json` | `NFL\data\props_frames\` | offline fallback for the events list only |
   | `claude_memory/` | `C:\Users\<you>\.claude\projects\C--Users-<you>-Documents-GitHub-NFL\memory\` | Claude's project notes (the folder name encodes the repo path) |

   Alternative to the zip for the package: `git clone https://<PAT>@github.com/drburke-droid/model-burke-private.git`
   and point `--pkg`/the positional argument at that clone (it contains `model_burke/`).
4. Install Claude Code, open a terminal in the repo folder and start `claude`. The memory
   index loads automatically if step 3 put the folder at the matching path.

Check: `python scripts/sabersim_weekly.py pkg --no-market --no-lineups --out outputs\sabersim\smoke.csv`
should print the history summary and a 2023-25 scorecard, then write a CSV. Delete `smoke.csv`.

## Weekly inputs (same as the automated path)

- FFA weekly file: `data/ffanalytics/FFAn_weekly/raw_stats_<season>_wk<week>.csv`, exported
  from FFA under the **DraftKings** scoring profile (so it has the `rec` column). Uploading via
  the Pages tab commits it to the repo; `git pull` brings it down.
- Subvertadown paste: parsed by `scripts/parse_subvertadown.py` into `data/subvertadown/` and
  `data/kdst/kdst_<season>_wk<week>.csv` (the K/DST override). Also arrives via `git pull`.
- Manual inactives (optional): `data/inactives/inactives_<season>_wk<week>.txt`, one player per
  line; `active: <name>` overrides a wrongly-listed OUT.

## Generate the CSV (window: T-90 to T-75 before the slate's first kickoff)

```
git pull
python scripts/sabersim_weekly.py pkg --out outputs\sabersim\Burke_Model_Burke_<season>_<slate>.csv
```

Defaults do the right thing: next slate only (games within 90 min of the earliest upcoming
kickoff), live DK lines and props blended per stat, live ESPN/Sleeper lineups with OUT
redistribution, DOUBT players at P(plays) 0.2, K/DST override from the Subvertadown paste at
weight 0.65. Useful flags:

| flag | when |
|---|---|
| `--all-remaining` | every game not yet kicked off (a whole Sunday in one file) |
| `--hours 6` | only games within 6 hours |
| `--no-market` | Odds API keys exhausted (FFA-only baseline; say so in the email) |
| `--no-lineups` | ESPN/Sleeper down (then use the inactives file by hand) |
| `--week N` | force the week if the FFA file name and the calendar disagree |

The run takes ~2.5 min (package walk-forward on 2023-25 history). The log's scorecard block is
the sanity check: 2025 MAE ~4.12 vs FFA ~4.19 and coverage ~0.80. If the scorecard is far off
or the row count is wrong (expect ~30 skill rows per game plus 2 K and 2 DST), do not send.

Or use the wrapper, which names the file, logs the send and can email directly:
```
set SMTP_USER=<gmail address>
set SMTP_PASS=<gmail app password (2-step verification -> App Passwords)>
set MAIL_TO=analysts+robert@sabersim.com
set MAIL_CC=<gmail address>
set MODEL_BURKE_PKG=pkg
python scripts/sabersim_auto.py --force
```
(`--dry-run` generates without emailing; `--force` ignores the T-90..T-72 window and the sent log.
The wrapper uses `data/odds_api_key.txt` directly when `ODDS_API_KEY` is unset.)

## Email it by hand

To `analysts+robert@sabersim.com`, subject `Robert Burke - Model_Burke projections - <slate label>`,
the CSV attached, one line of body: which games, generated time (ET), and any caveat (`--no-market`
used, a QB ruled out after generation). No model mechanics in the CSV or the body.
The CSV must land before T-75. If you miss T-75 for a slate, send the next slate instead.

## What the automated path did that you are replacing

cron-job.org hits the workflow every 5 min; inside T-90..T-72 the wrapper generates, emails
SaberSim (MAIL_TO) with the CC receipt to Gmail, publishes CSV + run parquet to the private repo
(`sends/`), and records `data/sabersim/sent_log.json`. A manual send is not in that log; add a
line to the receipt email to yourself so grading can find the file (the grader reads the run
parquet in `outputs/sabersim/` on this machine if you run `scripts/sabersim_grade.py` locally).
