# Connect your ESPN league to the Draft Room

Your league (`leagueId 1211359110`, `teamId 5`, season 2026) is **private**, so the pull
needs your two ESPN login cookies. Two ways to run it.

## Get your two cookies (one-time)
1. Log in at https://fantasy.espn.com
2. DevTools (F12) → **Application → Cookies → https://fantasy.espn.com**
3. Copy the **values** of:
   - `espn_s2` — a long URL-encoded string
   - `SWID` — looks like `{XXXXXXXX-XXXX-...}` (keep the curly braces)

---

## Option A — GitHub (recommended; remote, no local files)
1. In the repo: **Settings → Secrets and variables → Actions → New repository secret**
   - `ESPN_S2` = your espn_s2 value
   - `ESPN_SWID` = your SWID value (with braces)
2. **Actions** tab → **Pull ESPN league** → **Run workflow** (the league/season/team are
   pre-filled). It runs on GitHub's servers and commits **`outputs/espn_league.json`** to the repo.
3. Open the **🏟️ Draft Room** → **📥 Import ESPN league** → select that JSON
   (download it from the repo, or pull the repo if you have it locally).

Secrets are encrypted and never printed. The repo is private, so the committed league JSON
stays private.

## Option B — Local (if you can run Python on the machine)
1. Copy `data/espn_cookies.example.json` → `data/espn_cookies.json` (gitignored) and paste
   your two values.  *(Or set env vars `ESPN_S2` / `ESPN_SWID`.)*
2. `python scripts/fetch_espn.py`
3. Import the printed `outputs/espn_league.json` in the Draft Room.

---

## What gets imported
League size, roster slots, auction budget, the team names, **your** team, and everyone's
**keepers** (matched to our players by name + position, with their keeper $). The live Draft
Room engine (scarcity/VONA + risk-adjusted bids) then runs against your real league.

Pre-draft, ESPN exposes settings + teams + keepers (the draft itself hasn't happened — which
is what the Draft Room is for). Scoring is captured raw in the JSON; share the printed summary
if you want exact ESPN scoring mapped into the projections.
