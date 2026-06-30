"""
Hook into your real ESPN fantasy league (private leagues need YOUR login cookies).

ESPN's fantasy API is unofficial and (a) requires the espn_s2 + SWID cookies for a private
league, (b) blocks cross-origin browser calls. So this runs LOCALLY: it pulls your league
settings, teams, owners, rosters and keepers, and writes a normalized JSON the Draft Room
can import (outputs/espn_league.json) — same idea as the FFA CSV upload.

--- Get your two cookies (one-time) ---
  1. Log in at fantasy.espn.com in your browser.
  2. DevTools (F12) -> Application -> Cookies -> https://fantasy.espn.com
  3. Copy the VALUES of:  espn_s2   (long URL-encoded string)   and   SWID  (looks like {XXXX-...})
Provide them either way:
  - env:   set ESPN_S2=... and ESPN_SWID={...}
  - file:  data/espn_cookies.json  ->  {"espn_s2":"...","SWID":"{...}"}   (gitignored)

Usage:  python scripts/fetch_espn.py            (uses defaults below)
        python scripts/fetch_espn.py <leagueId> <season> <teamId>
"""
import os, sys, json
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LEAGUE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1211359110
SEASON = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
TEAM_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 5
OUT = os.path.join(ROOT, "outputs", "espn_league.json")
RAW = os.path.join(ROOT, "data", "espn"); os.makedirs(RAW, exist_ok=True)

SLOT = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DST", 17: "K", 20: "BENCH", 21: "IR", 23: "FLEX", 7: "OP"}
POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}


def cookies():
    s2, swid = os.environ.get("ESPN_S2"), os.environ.get("ESPN_SWID")
    f = os.path.join(ROOT, "data", "espn_cookies.json")
    if (not s2 or not swid) and os.path.exists(f):
        d = json.load(open(f)); s2 = s2 or d.get("espn_s2"); swid = swid or d.get("SWID") or d.get("swid")
    return s2, swid


def main():
    s2, swid = cookies()
    if not s2 or not swid:
        print("No ESPN cookies found. Set env ESPN_S2 / ESPN_SWID, or create data/espn_cookies.json")
        print('  {"espn_s2":"<value>","SWID":"{<value>}"}')
        print("(See the header of this file for how to copy them from your browser.)")
        sys.exit(1)
    url = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}"
           f"/segments/0/leagues/{LEAGUE_ID}")
    params = [("view", v) for v in ("mSettings", "mTeam", "mRoster", "mNav")]
    r = requests.get(url, params=params, cookies={"espn_s2": s2, "SWID": swid},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:200]}")
        print("401 -> cookies missing/expired or wrong league; 404 -> bad leagueId/season.")
        sys.exit(1)
    data = r.json()
    json.dump(data, open(os.path.join(RAW, f"league_{SEASON}.json"), "w"), indent=1)

    st = data.get("settings", {})
    roster = {SLOT.get(int(k), str(k)): v for k, v in st.get("rosterSettings", {}).get("lineupSlotCounts", {}).items() if v}
    members = {m.get("id"): (m.get("displayName") or m.get("firstName", "")) for m in data.get("members", [])}
    teams = []
    for t in data.get("teams", []):
        owner = next((members.get(o) for o in (t.get("owners") or []) if members.get(o)), None)
        rost = []
        for e in (t.get("roster", {}) or {}).get("entries", []):
            ppe = e.get("playerPoolEntry") or {}; p = ppe.get("player", {})
            kval = ppe.get("keeperValue") or ppe.get("keeperValueFuture") or 0
            rost.append({"name": p.get("fullName"), "pos": POS.get(p.get("defaultPositionId"), "?"),
                         "slot": SLOT.get(e.get("lineupSlotId"), str(e.get("lineupSlotId"))),
                         "keeper": bool(kval or e.get("keeper")), "keeper_price": kval})
        teams.append({"id": t.get("id"), "name": (t.get("name") or f"{t.get('location','')} {t.get('nickname','')}").strip(),
                      "abbrev": t.get("abbrev"), "owner": owner, "roster": rost})
    out = {"leagueId": LEAGUE_ID, "season": SEASON, "myTeamId": TEAM_ID,
           "name": st.get("name"), "size": st.get("size"), "roster_slots": roster,
           "scoringType": st.get("scoringSettings", {}).get("scoringType"),
           "scoring_raw": [{"statId": si.get("statId"), "points": si.get("points")}
                           for si in st.get("scoringSettings", {}).get("scoringItems", [])],
           "draft": st.get("draftSettings", {}), "teams": teams}
    json.dump(out, open(OUT, "w"), indent=1)

    print(f"League: {out['name']!r}  ({out['size']} teams, {SEASON})  scoring={out['scoringType']}")
    print("Starting slots:", {k: v for k, v in roster.items() if k not in ("BENCH", "IR")},
          "| bench", roster.get("BENCH", 0))
    dr = out["draft"]; print("Draft:", dr.get("type"), "auctionBudget", dr.get("auctionBudget"), "date set:", bool(dr.get("date")))
    print(f"\nTeams ({len(teams)}):")
    for t in sorted(teams, key=lambda x: x["id"]):
        kp = [p["name"] for p in t["roster"] if p["keeper"]]
        me = "  <-- YOU" if t["id"] == TEAM_ID else ""
        print(f"  [{t['id']:>2}] {str(t['name'])[:24]:24s} {str(t['owner'] or '')[:16]:16s} keepers: {', '.join(kp) if kp else '—'}{me}")
    print(f"\nSaved {os.path.relpath(OUT)} (+ raw in data/espn/). Import it into the Draft Room.")


if __name__ == "__main__":
    main()
