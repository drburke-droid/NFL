"""Bake the week's projections for the Fan Picks page (docs/fan.html).

The page shows every skill player's projected raw stats for the week and lets a fan boost or fade
each stat by 1-3 arrows. Those arrows are RECORDED, not used (scripts/fan_adjust_record.py) — the
same treatment as the Subvertadown matchup bonuses: collect now, grade later, wire in only if it
proves out.

Input: a generator CSV for the whole week, i.e.
    python scripts/sabersim_weekly.py <pkg> --all-games --no-market --out outputs/fan/proj_<S>_wk<W>.csv
(--no-market keeps it off the Odds API credits; the fan baseline does not need the DK blend, the
grade compares fan arrows against THIS baseline and against actuals.)

Outputs:
    docs/fan/proj_latest.json         what the page loads
    docs/fan/proj_<S>_wk<W>.json      frozen copy for the week (the record script decodes codes against it)

Usage: python scripts/fan_proj_bake.py <csv> <season> <week>
"""
import os, sys, json, csv
from datetime import datetime, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src, S, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
STATS = {"QB": ["pass_yds", "pass_tds", "pass_int", "rush_yds", "rush_tds"],
         "RB": ["rush_yds", "rush_tds", "rec", "rec_yds", "rec_tds"],
         "WR": ["rec", "rec_yds", "rec_tds", "rush_yds"],
         "TE": ["rec", "rec_yds", "rec_tds"]}
MIN_PROJ = 1.0
NAME2ABBR = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
rows = list(csv.DictReader(open(src, encoding="utf-8")))
gen = rows[0].get("Generated", "") if rows else ""
players, games = [], {}
def f(v):
    try: return round(float(v), 2)
    except (TypeError, ValueError): return None
for r in rows:
    pos = r["Pos"]
    if pos not in STATS: continue
    proj = f(r["Proj"]) or 0.0
    if proj < MIN_PROJ or (r.get("Status") or "") == "OUT": continue
    game = r["Game"]; away, home = [NAME2ABBR.get(g.strip(), g.strip()) for g in game.split("@")]
    games.setdefault(game, {"game": game, "kickoff": r["Kickoff"], "away": away, "home": home, "teams": set()})["teams"].add(r["Team"])
    opp = r.get("Opp") or (home if r["Team"] == away else away if r["Team"] == home else "")
    stats = {k: f(r.get(k)) for k in STATS[pos]}
    if pos == "WR" and (stats.get("rush_yds") or 0) < 3: stats.pop("rush_yds", None)   # only show a WR's rushing when it is a real part of his line
    players.append({"i": len(players), "id": r.get("ID") or "", "name": r["Player"], "team": r["Team"], "opp": opp, "pos": pos,
                    "game": game, "proj": proj, "status": r.get("Status") or "", "inj": r.get("Injury") or "", "note": (r.get("Note") or "")[:80],
                    "stats": stats})
for g in games.values(): g["teams"] = sorted(g["teams"])
order = sorted(games.values(), key=lambda g: (datetime.strptime(f"{S} " + g["kickoff"][4:], "%Y %m/%d %I:%M %p ET"), g["game"]))
out = {"season": S, "week": W, "generated": gen, "baked": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
       "stat_keys": ["pass_yds", "pass_tds", "pass_int", "rush_yds", "rush_tds", "rec", "rec_yds", "rec_tds"],
       "games": order, "players": players}
d = os.path.join(ROOT, "docs", "fan"); os.makedirs(d, exist_ok=True)
for name in ("proj_latest.json", f"proj_{S}_wk{W}.json"):
    with open(os.path.join(d, name), "w", encoding="utf-8") as fh: json.dump(out, fh, separators=(",", ":"))
print(f"baked {len(players)} players in {len(order)} games ({S} wk{W}, generated {gen}) -> docs/fan/proj_latest.json + proj_{S}_wk{W}.json")
