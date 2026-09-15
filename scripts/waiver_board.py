"""Cross-reference the ESPN league's rosters with our projections to find waiver adds.

Availability in this league is not a guess: outputs/espn_league.json carries all 12 rosters, so
anyone projected and not on one of them is a free agent. (waiver_rb_screen.py approximates the
same thing with "outside the preseason FFA top-36", which both misses rostered scrubs and hides
undrafted players other teams already grabbed.)

Scoring is read from the league's own settings rather than hardcoded, because this league is not
DK: 6-point passing TDs, full PPR, and no yardage bonuses. Reusing the SaberSim Proj column would
rank QBs wrong by roughly two points per passing TD, so every player is rescored from the FFA
component projections.

Exactness, stated up front because a waiver claim is a real decision:
  * QB/RB/WR/TE are exact — every non-zero rule that touches them maps to an FFA column.
  * K is close: the league pays 5 for a 50-59 yarder and 6 for 60+, and FFA publishes one fg_50
    bucket, so 60+ kicks are paid at 5.
  * DST is NOT trusted and is excluded unless --dst is passed; see the note by UNMAPPED below.
Anything in the league's scoring this script cannot map is printed, never dropped in silence.

Usage: python scripts/waiver_board.py [--season 2026] [--week N] [--top 12] [--dst]
"""
import argparse, csv, json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAGUE = os.path.join(ROOT, "outputs", "espn_league.json")
FFA_DIR = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
SKILL = ("QB", "RB", "WR", "TE")

# statId -> the FFA column it scores. Decoded from the standard ESPN fantasy stat table; the ones
# this league leaves at zero never reach here. FOLD means the rule is real but FFA reports it
# combined with another rule (two-point plays, and the several return/defensive TD flavours), so it
# is counted once via the lead statId and only if the folded rules pay the same rate.
STAT_MAP = {
    3: "pass_yds", 4: "pass_tds", 20: "pass_int",
    24: "rush_yds", 25: "rush_tds",
    42: "rec_yds", 43: "rec_tds", 53: "rec",
    72: "fumbles_lost",
    19: "two_pts",                       # 2pt pass, leads 26 (rush) and 44 (rec)
    101: "return_tds",                   # kick return TD, leads 102/103/104
    93: "dst_td",
    77: "fg_4049", 80: "fg_0039", 86: "xp", 198: "fg_50",
}
FOLD = {26: 19, 44: 19, 102: 101, 103: 101, 104: 101}

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--week", type=int, default=None, help="default: the newest FFA weekly file present")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--dst", action="store_true", help="include DST despite the unmapped scoring rules")
ap.add_argument("--json-out", default=os.path.join(ROOT, "docs", "waiver.json"))
A = ap.parse_args()


def norm(s):
    s = str(s).lower().strip().replace("’", "'")
    s = re.sub(r"\s+d/st$|\s+dst$", "", s)
    s = re.sub(r"[.'\-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def f(row, col):
    try:
        return float(row.get(col) or 0.0)
    except ValueError:
        return 0.0


# ---------- league ----------
if not os.path.exists(LEAGUE):
    sys.exit(f"no {LEAGUE} — run the 'Pull ESPN league' workflow first")
lg = json.load(open(LEAGUE, encoding="utf-8"))
mine_id = lg["myTeamId"]
rostered, owner_of, by_surname = set(), {}, defaultdict(list)
my_players = []
for t in lg["teams"]:
    for p in t["roster"]:
        n = norm(p["name"])
        rostered.add(n)
        owner_of[n] = t["name"]
        by_surname[(n.split()[-1] if n.split() else n, p["pos"], p.get("team", ""))].append(n)
        if t["id"] == mine_id:
            my_players.append({"name": p["name"], "pos": p["pos"], "slot": p["slot"], "n": n})


def is_rostered(n, pos, team):
    """Exact name, then surname + NFL team + position when that is unambiguous.

    ESPN and FFA disagree on given names — ESPN's "Kenny Gainwell" is FFA's "Kenneth Gainwell" —
    and on an exact-name-only join he came out as the top available RB while another team already
    owned him. Recommending a rostered player is the failure that makes the board useless.

    Team is what makes the fallback safe. Surname+position alone matched 24 players here and 23 of
    them were different people (Chris Brooks onto Jonathon Brooks, Brian Robinson onto Bijan
    Robinson, three separate Johnsons onto Emmett Johnson) — which hides real free agents instead
    of revealing them. Adding the team leaves only genuine aliases, and a surname still shared
    inside one team's depth chart is left alone rather than guessed.
    """
    if n in rostered:
        return True, owner_of[n], "name"
    cand = by_surname.get((n.split()[-1] if n.split() else n, pos, team), [])
    if len(cand) == 1:
        a, b = n.split()[0], cand[0].split()[0]
        if compatible_first(a, b):
            return True, owner_of[cand[0]], "surname+team"
        REJECTED.append((n, cand[0], team, pos))
    return False, "", ""


REJECTED = []


def compatible_first(a, b):
    """Are these two given names plausibly the same person?

    Surname + team + position is still not enough on its own: Bijan Robinson and Brian Robinson are
    both 2026 Atlanta running backs, and only Bijan is rostered, so the looser rule hid a real free
    agent. What the fallback is actually for is a nickname — ESPN's "Kenny" for FFA's "Kenneth" —
    so require one name to be a prefix of the other, or a shared prefix of four characters.
    kenny/kenneth share "kenn" and pass; brian/bijan share "b" and do not. The rule errs toward
    rejecting, which shows a player as available rather than hiding him, and every rejection is
    printed so a genuine alias surfaces instead of going quiet.
    """
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i >= 4
print(f"league: {lg.get('name')} | {lg['size']} teams | {len(rostered)} rostered players")
print(f"my team: {next(t['name'] for t in lg['teams'] if t['id'] == mine_id)} ({len(my_players)} players)")

# ---------- scoring ----------
rules = {int(s["statId"]): float(s["points"]) for s in lg["scoring_raw"] if s["points"]}
scoring, unmapped, folded = {}, [], []
for sid, pts in sorted(rules.items()):
    if sid in FOLD:
        lead = FOLD[sid]
        if rules.get(lead) == pts:
            folded.append(sid); continue
        unmapped.append((sid, pts, f"pays differently from statId {lead}, which FFA reports combined"))
    elif sid in STAT_MAP:
        scoring[STAT_MAP[sid]] = pts
    else:
        unmapped.append((sid, pts, "no FFA column"))
print(f"scoring: {len(scoring)} rules mapped, {len(folded)} folded into a lead rule")
print(f"  PPR {scoring.get('rec', 0)} | pass TD {scoring.get('pass_tds', 0)} | "
      f"pass yd {scoring.get('pass_yds', 0)} | rush/rec yd {scoring.get('rush_yds', 0)}/{scoring.get('rec_yds', 0)}")
if unmapped:
    print("  UNMAPPED (not counted — every one is a DST or a rare-TD rule, which is why DST is "
          "off by default):")
    for sid, pts, why in unmapped:
        print(f"    statId {sid} = {pts}  ({why})")

# ---------- projections ----------
week = A.week
if week is None:
    got = sorted(int(m.group(1)) for fn in os.listdir(FFA_DIR)
                 for m in [re.match(rf"raw_stats_{A.season}_wk(\d+)\.csv$", fn)] if m)
    if not got:
        sys.exit(f"no FFA weekly file for {A.season} in {FFA_DIR}")
    week = got[-1]
path = os.path.join(FFA_DIR, f"raw_stats_{A.season}_wk{week}.csv")
if not os.path.exists(path):
    sys.exit(f"no {path}")
rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["position"] in SKILL + ("K", "DST")]
print(f"projections: {A.season} week {week} ({len(rows)} players at scoring positions)")
# The FFA weekly scrape is a Wednesday job whose own workflow header calls itself untested, so the
# newest file on disk can easily be last week's. A waiver claim made on the previous week's
# projections is worse than no board at all, so say it loudly rather than in a status line.
sched = os.path.join(ROOT, "data", f"schedule_{A.season}.csv")
cur = None
if os.path.exists(sched):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    wks = defaultdict(list)
    for r in csv.DictReader(open(sched, encoding="utf-8")):
        for key in ("gameday", "game_date", "date", "kickoff", "start_time"):
            if r.get(key):
                try:
                    wks[int(float(r.get("week")))].append(datetime.fromisoformat(
                        str(r[key]).replace("Z", "+00:00")).replace(tzinfo=timezone.utc))
                except Exception:
                    pass
                break
    upcoming = sorted(w for w, ds in wks.items() if max(ds) > now)
    cur = upcoming[0] if upcoming else None
if cur and cur != week:
    print(f"\n  *** WARNING: these are WEEK {week} projections, but week {cur} is the one being "
          f"played.\n      The week {cur} FFA file is not in the repo yet (it is a Wednesday "
          f"scrape).\n      Ranks below reflect last week's expectations — do not claim on them "
          f"blind. ***")

proj = []
for r in rows:
    r["fg_0039"] = f(r, "fg_0019") + f(r, "fg_2029") + f(r, "fg_3039")
    pts = sum(f(r, col) * mult for col, mult in scoring.items())
    n = norm(r["player"])
    held, who, how = is_rostered(n, r["position"], r["team"])
    proj.append({"name": r["player"], "pos": r["position"], "team": r["team"], "n": n,
                 "pts": round(pts, 2), "inj": (r.get("injury_status") or "").strip(),
                 "rostered": held, "owner": who, "match": how})

# a projected player nobody rosters is, by definition, a free agent in a 12-team league
hit = {p["owner"] + "|" + p["n"] for p in proj if p["rostered"]}
alias = [p for p in proj if p["match"] == "surname+team"]
print(f"matched {len(hit)} of {len(rostered)} rostered players to a projection "
      f"({len(rostered) - len(hit)} unmatched — not projected this week, IR, or spelling)")
if alias:
    print("  matched by surname+team (nickname): "
          + ", ".join(f"{p['name']} [{p['team']}]" for p in alias))
if REJECTED:
    print("  same surname/team/position but different given names — treated as AVAILABLE:")
    for fn, rn, tm, ps in REJECTED:
        print(f"    {fn} vs rostered {rn} ({ps} {tm})")

positions = SKILL + ("K",) + (("DST",) if A.dst else ())
board = {}
for pos in positions:
    pool = sorted([p for p in proj if p["pos"] == pos], key=lambda x: -x["pts"])
    free = [p for p in pool if not p["rostered"]]
    ours = [p for p in pool if p["n"] in {m["n"] for m in my_players}]
    worst = min((p["pts"] for p in ours), default=0.0)
    board[pos] = {"mine": ours, "free": free[:A.top], "worst_of_mine": worst}

    print(f"\n=== {pos} ===")
    if ours:
        print("  mine: " + ", ".join(f"{p['name']} {p['pts']}" for p in ours))
    for p in free[:A.top]:
        d = p["pts"] - worst
        flag = "  <-- upgrade" if ours and d > 0 else ""
        inj = f"  [{p['inj']}]" if p["inj"] and p["inj"].lower() not in ("healthy", "probable") else ""
        print(f"  {p['pts']:>6.2f}  {p['name'][:24]:<26} {p['team']:<4} "
              f"{d:+6.2f} vs my worst{inj}{flag}")

os.makedirs(os.path.dirname(A.json_out), exist_ok=True)
json.dump({"season": A.season, "week": week, "league": lg.get("name"),
           "scoring": scoring, "unmapped": [list(u) for u in unmapped],
           "board": board}, open(A.json_out, "w"), indent=1)
print(f"\nwrote {os.path.relpath(A.json_out, ROOT)}")
