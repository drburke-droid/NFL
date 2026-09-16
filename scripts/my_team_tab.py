"""🏠 My Team tab — bake docs/myteam_2026.js for the draft tool's My Team page.

One JS global, MY_TEAM, holding:
  roster    my ESPN roster (outputs/espn_league.json, team myTeamId) with slot, keeper, ESPN injury tag,
            season / last-week ESPN points, this week's opponent or BYE, and the league-scored FFA
            projection for the board week (docs/waiver.json, scripts/waiver_board.py)
  board     per position: the top free agents (nobody in the league rosters them) with projection,
            delta vs my worst projected player at that slot, injury tag, opponent; plus my own list
  meta      league, team, board week vs current week (stale flag), generated time, refresh notes

Refresh order (each step optional if its input is already fresh):
  python scripts/fetch_espn.py                     # rosters (needs the two ESPN cookies; or the
                                                   #   "Pull ESPN league" Actions workflow)
  python scripts/waiver_board.py                   # projections + free agents from the newest FFA weekly file
  python scripts/my_team_tab.py [--pull] [--board] # --pull / --board run the two steps above first
Then commit + push docs/myteam_2026.js (and docs/waiver.json, outputs/espn_league.json).
"""
import argparse, csv, json, os, re, subprocess, sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--pull", action="store_true", help="run fetch_espn.py first")
ap.add_argument("--board", action="store_true", help="run waiver_board.py first")
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "myteam_2026.js"))
A = ap.parse_args()
if A.pull: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fetch_espn.py")], check=True)
if A.board: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "waiver_board.py"), "--season", str(A.season)], check=True)

LG = json.load(open(os.path.join(ROOT, "outputs", "espn_league.json"), encoding="utf-8"))
W = json.load(open(os.path.join(ROOT, "docs", "waiver.json"), encoding="utf-8"))
me = next(t for t in LG["teams"] if t["id"] == LG["myTeamId"])


def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)


# ---- this week's opponent / bye per NFL team (schedule uses nflverse codes; FFA/ESPN use LVR/JAC/LAR) ----
FIX = {"LVR": "LV", "JAC": "JAX", "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}
week = int(W.get("current_week") or W.get("week") or 1)
opp = {}
with open(os.path.join(ROOT, "data", f"schedule_{A.season}.csv"), encoding="utf-8") as fh:
    for g in csv.DictReader(fh):
        if int(g["week"]) == week and g["game_type"] == "REG":
            opp[g["home_team"]] = "vs " + g["away_team"]; opp[g["away_team"]] = "@ " + g["home_team"]
def opp_of(team): return opp.get(FIX.get(team, team), "BYE") if team else ""

# ---- projections by normalised name (board week) ----
proj = {}
for pos, b in W["board"].items():
    for p in b.get("mine", []) + b.get("free", []): proj[(p["n"], pos)] = p
mine_n = {}
for pos, b in W["board"].items():
    for p in b.get("mine", []): mine_n[p["n"]] = p

SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "OP", "K", "DST", "BENCH", "IR"]
roster = []
for r in me["roster"]:
    n = norm(r["name"]); p = proj.get((n, r["pos"])) or mine_n.get(n) or {}
    espn_tag = (r.get("inj") or "").upper()
    is_out = bool(p.get("out")) or espn_tag in ("INJURY_RESERVE", "OUT", "SUSPENSION", "SUSPENDED")
    roster.append({"name": r["name"], "pos": r["pos"], "team": r.get("team", ""), "slot": r.get("slot", ""), "opp": opp_of(r.get("team", "")),
                   "keeper": bool(r.get("keeper")), "keeper_price": r.get("keeper_price") or 0, "acq": r.get("acq"),
                   "inj_espn": r.get("inj") or "", "inj_ffa": (p.get("inj") or "") if p.get("inj") not in (None, "NA") else "",
                   "season_pts": r.get("season_pts"), "last_pts": r.get("last_pts"),
                   "proj": (0.0 if is_out else p.get("pts")), "out": is_out, "n": n})
roster.sort(key=lambda r: (SLOT_ORDER.index(r["slot"]) if r["slot"] in SLOT_ORDER else 99, -(r["proj"] or 0)))

board = {}
for pos, b in W["board"].items():
    worst = b.get("worst_of_mine", 0.0)
    free = [{"name": p["name"], "team": p["team"], "opp": opp_of(p["team"]), "pts": p["pts"], "delta": round(p["pts"] - worst, 2),
             "inj": (p.get("inj") or "") if p.get("inj") not in (None, "NA") else ""} for p in b.get("free", [])]
    mine = sorted([{"name": p["name"], "team": p["team"], "opp": opp_of(p["team"]), "pts": p["pts"],
                    "inj": (p.get("inj") or "") if p.get("inj") not in (None, "NA") else ""} for p in b.get("mine", [])], key=lambda x: -x["pts"])
    board[pos] = {"free": free, "mine": mine, "worst_of_mine": worst}

# ---- the RB usage screen, when this week's file exists (scripts/waiver_rb_screen.py, Tuesdays) ----
screen = []
for fn in (f"waiver_rb_screen_{A.season}_wk{week - 1}.csv", f"waiver_rb_screen_{A.season}_wk{week}.csv"):
    fp = os.path.join(ROOT, "outputs", fn)
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as fh:
            for r in csv.DictReader(fh): screen.append({k: r.get(k) for k in ("full_name", "team", "why", "own_status") if k in r})
        break

espn_mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(ROOT, "outputs", "espn_league.json")), timezone.utc)
payload = {"meta": {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "season": A.season,
                    "league": LG.get("name"), "team": me["name"], "owner": me.get("owner"), "team_id": me["id"],
                    "board_week": W.get("week"), "current_week": week, "stale": bool(W.get("stale")),
                    "roster_pulled_utc": espn_mtime.strftime("%Y-%m-%dT%H:%MZ"),
                    "starting_slots": {k: v for k, v in (LG.get("roster_slots") or {}).items() if k not in ("BENCH", "IR")},
                    "unmapped": W.get("unmapped", [])},
           "roster": roster, "board": board, "rb_screen": screen}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
open(A.out, "w", encoding="utf-8").write("// baked by scripts/my_team_tab.py — do not edit\nconst MY_TEAM = " + json.dumps(payload, separators=(",", ":")) + ";\n")
print(f"{me['name']} ({me.get('owner')}): {len(roster)} rostered, {sum(1 for r in roster if r['proj'] is not None)} with a week-{W.get('week')} projection; "
      f"board {', '.join(f'{p} {len(b['free'])}' for p, b in board.items())}; stale={payload['meta']['stale']} -> {os.path.relpath(A.out, ROOT)}")
