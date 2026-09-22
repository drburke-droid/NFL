"""Refresh every weekly tab to the week of the newest FFA file, in one command.

The SaberSim page uploads raw_stats_<S>_wk<W>.csv and the Subvertadown paste; each tab then needs
its own bake (My Team, waiver board, Fan Picks, Props Watch, the gm league config), each with its
own script, its own inputs and its own idea of "the week". This runs them all, in the order their
inputs demand, against ONE week -- the week of the newest FFA file -- and reports what refreshed
and what did not. The Actions workflow refresh_tabs.yml runs it on every FFA upload; locally:

    python scripts/refresh_tabs.py [--pkg <model_burke dir>] [--week N] [--skip props,fan,...]

Steps (each one is optional and failure-isolated; the summary says which ran):
    espn      fetch_espn.py + league state + standings   (needs ESPN_S2/ESPN_SWID or the cookie file)
    pastes    parse_subvertadown.py on every raw paste   (idempotent)
    board     waiver_board.py --week W                   -> docs/waiver.json
    myteam    my_team_tab.py                             -> docs/myteam_2026.js
    gm        build_gm_league.py                         -> gm/leagues/kuhn_2026.json
    fan       sabersim_weekly.py --all-games --no-market + fan_proj_bake.py -> docs/fan/proj_latest.json
    props     props_watch.py                             -> docs/props_watch_2026.js (needs the key file)
Writes docs/refresh_status.json so a page can show when each tab was last baked, and for which week.
"""
import argparse, glob, json, os, re, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFA_DIR = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
STATUS = os.path.join(ROOT, "docs", "refresh_status.json")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def ffa_week(season):
    """The newest FFA weekly file on disk: the week the user just uploaded."""
    wks = [int(m.group(1)) for f in glob.glob(os.path.join(FFA_DIR, f"raw_stats_{season}_wk*.csv"))
           for m in [re.search(r"_wk(\d+)\.csv$", f)] if m]
    return max(wks) if wks else None


def schedule_week(season):
    """The week the schedule says is current: first week with a game not yet four hours old."""
    p = os.path.join(ROOT, "data", f"schedule_{season}.csv")
    if not os.path.exists(p):
        return None
    import pandas as pd
    s = pd.read_csv(p); s = s[s.game_type == "REG"]
    kick = pd.to_datetime(s.gameday + " " + s.gametime.fillna("13:00"), errors="coerce").dt.tz_localize("America/New_York")
    live = s[kick + pd.Timedelta(hours=4) > pd.Timestamp.now(tz="America/New_York")]
    return int(live.week.min()) if len(live) else int(s.week.max())


ERRORS = {}


def run(name, cmd, env=None, cwd=ROOT):
    """Run a step, echo its output, and keep the tail of a failure for the status file -- the
    Actions logs need a token to read, the committed status file does not."""
    t = time.time()
    print(f"\n== {name}: {' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"},
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    print(out, end="" if out.endswith("\n") else "\n", flush=True)
    ok = r.returncode == 0
    if not ok:
        ERRORS[name] = [ln for ln in out.strip().splitlines() if ln.strip()][-25:]
    print(f"   {'ok' if ok else 'FAILED (rc %d)' % r.returncode} in {time.time() - t:.0f}s", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None, help="default: the newest FFA file's week")
    ap.add_argument("--pkg", default=os.environ.get("MODEL_BURKE_PKG", ""),
                    help="Model_Burke package dir (folder containing model_burke/); fan + props need it")
    ap.add_argument("--skip", default="", help="comma list of steps to skip: espn,pastes,board,myteam,gm,fan,props")
    a = ap.parse_args()
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    py = sys.executable
    S = a.season
    W = a.week or ffa_week(S)
    sched = schedule_week(S)
    if W is None:
        sys.exit(f"no FFA weekly file for {S} in {FFA_DIR}")
    print(f"refreshing tabs for {S} week {W} (newest FFA file)"
          + (f"; the schedule says week {sched}" + (" -- MISMATCH, the upload is for another week" if sched != W else "")
             if sched else ""))
    pkg = a.pkg if a.pkg and os.path.isdir(os.path.join(a.pkg, "model_burke")) else ""
    if not pkg:
        print("no Model_Burke package: fan and props steps will be skipped (pass --pkg or set MODEL_BURKE_PKG)")
        skip |= {"fan", "props"}
    have_cookies = bool(os.environ.get("ESPN_S2") and os.environ.get("ESPN_SWID")) or \
        os.path.exists(os.path.join(ROOT, "data", "espn_cookies.json"))
    if not have_cookies:
        print("no ESPN cookies: espn step skipped (My Team uses the committed roster pull)")
        skip.add("espn")
    if not os.path.exists(os.path.join(ROOT, "data", "odds_api_key.txt")):
        print("no data/odds_api_key.txt: props step skipped")
        skip.add("props")

    league = "1211359110"
    status = {}
    def step(name, fn):
        if name in skip:
            status[name] = "skipped"; print(f"\n== {name}: skipped"); return
        try:
            status[name] = "ok" if fn() else "failed"
        except Exception as ex:
            status[name] = f"failed: {ex}"[:120]; print(f"   FAILED: {ex}")

    step("espn", lambda: all([
        run("espn rosters", [py, "scripts/fetch_espn.py", league, str(S), "5"]),
        run("espn league state", [py, "scripts/fetch_espn_league_state.py", league, "2023", "2024", "2025", str(S)]),
        run("espn standings", [py, "scripts/fetch_espn_standings.py", league, "2023", "2024", "2025"])]))

    def pastes():
        ok = True
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "subvertadown", "raw", "paste_*_wk*.txt"))
                        + glob.glob(os.path.join(ROOT, "data", "kdst", "paste_*_wk*.txt"))):
            m = re.search(r"paste_(\d{4})_wk(\d+)\.txt$", f)
            if m:
                ok &= run(f"paste {os.path.basename(f)}", [py, "scripts/parse_subvertadown.py", m.group(1), m.group(2), f])
        return ok
    step("pastes", pastes)
    step("board", lambda: run("waiver board", [py, "scripts/waiver_board.py", "--season", str(S), "--week", str(W)]))
    step("myteam", lambda: run("my team tab", [py, "scripts/my_team_tab.py", "--season", str(S)]))
    step("gm", lambda: run("gm league config", [py, "scripts/build_gm_league.py"]))

    def fan():
        out = os.path.join(ROOT, "outputs", "fan", f"proj_{S}_wk{W}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        return (run("fan projections", [py, "scripts/sabersim_weekly.py", pkg, "--all-games", "--no-market", "--out", out])
                and run("fan bake", [py, "scripts/fan_proj_bake.py", out, str(S), str(W)]))
    step("fan", fan)
    step("props", lambda: run("props watch", [py, "scripts/props_watch.py", pkg]))

    status_doc = {"season": S, "week": W, "schedule_week": sched,
                  "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "steps": status,
                  "python": sys.version.split()[0], "platform": sys.platform, "errors": ERRORS,
                  # how many Odds API keys the props step could rotate through (never the keys)
                  "odds_keys": len([k for line in open(os.path.join(ROOT, "data", "odds_api_key.txt"))
                                    if line.strip() and not line.startswith("#")
                                    for k in re.split(r"[,;\s]+", line.strip()) if k])
                  if os.path.exists(os.path.join(ROOT, "data", "odds_api_key.txt")) else 0}
    with open(STATUS, "w", encoding="utf-8") as fh:
        json.dump(status_doc, fh, indent=1)
    print("\nsummary:", json.dumps(status))
    print(f"wrote {os.path.relpath(STATUS, ROOT)}")
    bad = [k for k, v in status.items() if str(v).startswith("failed")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
