"""
Pull your ESPN league's FINAL STANDINGS for past seasons (default 2023-2025)
-> outputs/espn_standings.csv (+ .json).

Why: keeper inflation is finish-based (champ +$5 / top-6 +$3 / 7-11 +$2 / last +$0 per
keeper), so to compute each year's keeper costs we need each team's final rank. The 2025
keepers' inflation is set by the 2024 finish; the 2026 keepers by the 2025 finish.

Same private-league auth as fetch_espn.py (espn_s2 + SWID via env or data/espn_cookies.json).
The mTeam view returns rankCalculatedFinal (1 = champion), playoffSeed, and the W-L record.

Usage:  python scripts/fetch_espn_standings.py                 (2023 2024 2025)
        python scripts/fetch_espn_standings.py 2024 2025
        python scripts/fetch_espn_standings.py <leagueId> 2023 2024 2025
"""
import os, sys, json, csv
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LEAGUE_ID = 1211359110
args = sys.argv[1:]
if args and len(args[0]) >= 7 and args[0].isdigit(): LEAGUE_ID = int(args.pop(0))
YEARS = [int(a) for a in args] or [2023, 2024, 2025]
OUT_CSV = os.path.join(ROOT, "outputs", "espn_standings.csv"); OUT_JSON = os.path.join(ROOT, "outputs", "espn_standings.json")
RAW = os.path.join(ROOT, "data", "espn"); os.makedirs(RAW, exist_ok=True)


def cookies():
    s2, swid = os.environ.get("ESPN_S2"), os.environ.get("ESPN_SWID")
    f = os.path.join(ROOT, "data", "espn_cookies.json")
    if (not s2 or not swid) and os.path.exists(f):
        d = json.load(open(f)); s2 = s2 or d.get("espn_s2"); swid = swid or d.get("SWID") or d.get("swid")
    return s2, swid


def bump(rank, size):
    """Finish-based per-keeper inflation: champ +5, top-6 +3, mid +2, last +0."""
    if not rank: return None
    if rank == 1: return 5
    if rank <= 6: return 3
    if rank >= size: return 0
    return 2


def main():
    s2, swid = cookies()
    if not s2 or not swid:
        print("No ESPN cookies. Set ESPN_S2/ESPN_SWID env or data/espn_cookies.json (see ESPN_SETUP.md).")
        sys.exit(1)
    ck = {"espn_s2": s2, "SWID": swid}
    rows = []
    for season in YEARS:
        url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{LEAGUE_ID}"
        r = requests.get(url, params=[("view", "mTeam"), ("view", "mSettings")], cookies=ck,
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 200:
            print(f"  {season}: HTTP {r.status_code} ({r.text[:80]}) — skipping"); continue
        d = r.json(); json.dump(d, open(os.path.join(RAW, f"standings_{season}.json"), "w"))
        members = {m.get("id"): (m.get("displayName") or m.get("firstName", "")) for m in d.get("members", [])}
        teams = d.get("teams", [])
        size = len(teams)
        srows = []
        for t in teams:
            rec = (t.get("record") or {}).get("overall") or {}
            rank = t.get("rankCalculatedFinal") or t.get("playoffSeed") or t.get("rankFinal")
            srows.append({
                "season": season, "team_id": t.get("id"),
                "team": (t.get("name") or f"{t.get('location','')} {t.get('nickname','')}").strip(),
                "owner": next((members.get(o) for o in (t.get("owners") or []) if members.get(o)), None),
                "rank_final": rank, "playoff_seed": t.get("playoffSeed"),
                "wins": rec.get("wins"), "losses": rec.get("losses"), "ties": rec.get("ties"),
                "points_for": round(rec.get("pointsFor") or 0, 1), "points_against": round(rec.get("pointsAgainst") or 0, 1),
                "keeper_bump": bump(rank, size),
            })
        srows.sort(key=lambda x: (x["rank_final"] or 99))
        rows.extend(srows)
        print(f"  {season}: {size} teams, final standings:")
        for s in srows:
            print(f"    #{str(s['rank_final'] or '?'):<2} {str(s['team'])[:24]:24s} {str(s['owner'] or '')[:14]:14s} "
                  f"({s['wins']}-{s['losses']})  PF {s['points_for']}  keeper-bump +${s['keeper_bump']}")
    if not rows:
        print("No standings pulled."); sys.exit(1)
    cols = ["season", "team_id", "team", "owner", "rank_final", "playoff_seed", "wins", "losses", "ties",
            "points_for", "points_against", "keeper_bump"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(OUT_JSON, "w"), indent=1)
    print(f"\nSaved {os.path.relpath(OUT_CSV)} ({len(rows)} team-seasons)")


if __name__ == "__main__":
    main()
