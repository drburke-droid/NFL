"""
Pull your ESPN league's historical DRAFT results (auction $ or snake picks) for past
seasons (default 2023-2025) -> outputs/espn_drafts.csv (+ .json).

Same private-league auth as fetch_espn.py (espn_s2 + SWID via env or data/espn_cookies.json).
For each pick: season, overall pick, round, team + owner, player + position, bid amount
(auction) or pick number, and keeper flag. This is the real clearing-price history for the
league — useful for calibrating auction values to how your league actually drafts.

Usage:  python scripts/fetch_espn_drafts.py                 (2023 2024 2025)
        python scripts/fetch_espn_drafts.py 2022 2023 2024
        python scripts/fetch_espn_drafts.py <leagueId> 2023 2024 2025
"""
import os, sys, json
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LEAGUE_ID = 1211359110
args = sys.argv[1:]
if args and len(args[0]) >= 7 and args[0].isdigit(): LEAGUE_ID = int(args.pop(0))
YEARS = [int(a) for a in args] or [2023, 2024, 2025]
OUT_CSV = os.path.join(ROOT, "outputs", "espn_drafts.csv"); OUT_JSON = os.path.join(ROOT, "outputs", "espn_drafts.json")
RAW = os.path.join(ROOT, "data", "espn"); os.makedirs(RAW, exist_ok=True)
POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}


def cookies():
    s2, swid = os.environ.get("ESPN_S2"), os.environ.get("ESPN_SWID")
    f = os.path.join(ROOT, "data", "espn_cookies.json")
    if (not s2 or not swid) and os.path.exists(f):
        d = json.load(open(f)); s2 = s2 or d.get("espn_s2"); swid = swid or d.get("SWID") or d.get("swid")
    return s2, swid


def player_map(season, ck):
    """ESPN playerId -> (name, pos) for the season."""
    url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/players?view=players_wl"
    try:
        r = requests.get(url, cookies=ck, timeout=40, headers={
            "User-Agent": "Mozilla/5.0", "x-fantasy-filter": json.dumps({"players": {"limit": 3000}})})
        if r.status_code != 200: return {}
        return {p["id"]: (p.get("fullName"), POS.get(p.get("defaultPositionId"), "?")) for p in r.json()}
    except Exception:
        return {}


def main():
    s2, swid = cookies()
    if not s2 or not swid:
        print("No ESPN cookies. Set ESPN_S2/ESPN_SWID env or data/espn_cookies.json (see ESPN_SETUP.md).")
        sys.exit(1)
    ck = {"espn_s2": s2, "SWID": swid}
    rows = []
    for season in YEARS:
        url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{LEAGUE_ID}"
        r = requests.get(url, params=[("view", "mDraftDetail"), ("view", "mTeam")], cookies=ck,
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 200:
            print(f"  {season}: HTTP {r.status_code} ({r.text[:80]}) — skipping"); continue
        d = r.json(); json.dump(d, open(os.path.join(RAW, f"draft_{season}.json"), "w"))
        members = {m.get("id"): (m.get("displayName") or m.get("firstName", "")) for m in d.get("members", [])}
        team = {t.get("id"): {"name": (t.get("name") or f"{t.get('location','')} {t.get('nickname','')}").strip(),
                              "owner": next((members.get(o) for o in (t.get("owners") or []) if members.get(o)), None)}
                for t in d.get("teams", [])}
        picks = (d.get("draftDetail") or {}).get("picks", [])
        if not picks:
            print(f"  {season}: no draft picks found (draft not run / not in history)"); continue
        pm = player_map(season, ck)
        auction = any(p.get("bidAmount") for p in picks)
        for p in picks:
            nm, pos = pm.get(p.get("playerId"), (str(p.get("playerId")), "?"))
            tm = team.get(p.get("teamId"), {})
            rows.append({"season": season, "overall_pick": p.get("overallPickNumber"), "round": p.get("roundId"),
                         "round_pick": p.get("roundPickNumber"), "team": tm.get("name"), "owner": tm.get("owner"),
                         "player": nm, "pos": pos, "bid": p.get("bidAmount") or 0, "keeper": bool(p.get("keeper"))})
        print(f"  {season}: {len(picks)} picks ({'AUCTION' if auction else 'snake'})"
              + (f", names matched {sum(1 for p in picks if pm.get(p.get('playerId')))}/{len(picks)}" if pm else " (no name map)"))
    if not rows:
        print("No draft data pulled."); sys.exit(1)
    df = pd.DataFrame(rows).sort_values(["season", "overall_pick"])
    df.to_csv(OUT_CSV, index=False); df.to_json(OUT_JSON, orient="records")
    print(f"\nSaved {os.path.relpath(OUT_CSV)} ({len(df)} picks, {df.season.nunique()} seasons)")
    for season in sorted(df.season.unique()):
        s = df[df.season == season]
        if s.bid.max() > 0:
            print(f"\n  {season} top auction prices:")
            for _, r in s.sort_values("bid", ascending=False).head(8).iterrows():
                print(f"    ${int(r.bid):>3}  {r.pos:<3} {str(r.player)[:22]:22s} -> {str(r.team)[:18]}")


if __name__ == "__main__":
    main()
