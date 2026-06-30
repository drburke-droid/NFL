"""
Predict each team's likely 2026 keepers in the ESPN league.

For every player on a team's carried-over 2026 roster (outputs/espn_league.json):
  cost_2026  = TRUE keeper cost entering 2026 + that team's 2026 inflation bump
  value_2026 = our calibrated Draft-Room auction value (risk-adjusted VONA)
  surplus    = value - cost
Keep the 3 best-surplus players per team (the rational keep).

ICO keeper rule (real, from league PDF — see fetch_espn_standings.py):
  inflation bump = +$5 default, -$2 if the owner MISSED the top-6 playoffs, -$3 more if a
  non-playoff owner DRAFTED THE EVENTUAL CHAMPION (so playoff +$5 / non-playoff +$3 / picked
  champ +$0). Undrafted/waiver keepers floor at $1 (and a first-time waiver keep is $1, no
  inflation). The 2025 draft FORGOT to apply inflation, so a 2025 keeper's TRUE cost = its
  recorded base + the keeping owner's 2024 bump. The 2026 cost then adds the current owner's
  2025 bump on top of that true basis. 3-yr cap doesn't bind for 2026 (keepers started 2024).
"""
import os, json, csv, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower())).replace("  ", " ").strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

# ---- our calibrated auction value per player ----
P = json.loads(open(os.path.join(ROOT, "docs", "data.js"), encoding="utf-8").read().split("const PLAYERS = ")[1].rsplit(";", 1)[0])
isK = lambda p: p["position"] in ("K", "DST"); INJ = {"QB": .26, "RB": .40, "WR": .33, "TE": .39}
def risk(p):
    if isK(p): return 0
    ppg = p.get("proj_ppg") or 0; fl = p.get("floor") if p.get("floor") is not None else ppg; bu = p.get("bust") if p.get("bust") is not None else .3
    inj = max(0, ((INJ.get(p["position"], .33) - .26) / .14)); down = max(0, (ppg - fl) / ppg) if ppg > 0 else 0
    return min(.9, .45 * inj + .40 * bu + .30 * down)
ra = lambda p: (p.get("proj_pts") or 0) * (1 - .7 * risk(p))
open_ = {"QB": 12, "RB": 24, "WR": 24, "TE": 12, "FLEX": 12}
fa = lambda pos: open_["FLEX"] * ({"RB": .45, "WR": .45, "TE": .10}.get(pos, 0))
repl = {}; within = set()
for pos in ("QB", "RB", "WR", "TE"):
    n = round(open_[pos] + fa(pos)); arr = sorted([p for p in P if p["position"] == pos], key=ra, reverse=True)
    repl[pos] = ra(arr[min(max(n - 1, 0), len(arr) - 1)])
    for p in arr[:n]: within.add(p["name"])
sumE = sum(max(ra(p) - repl[p["position"]], 0) for p in P if p["name"] in within)
per = (12 * 200 - 12 * 16) / sumE
comp = lambda v: v if v <= 25 else 25 + (v - 25) * 0.75
VAL = {}
for p in P:
    if isK(p): continue
    e = max(ra(p) - repl[p["position"]], 0); raw = (1 + e * per) if p["name"] in within else 1
    VAL[(norm(p["name"]), p["position"])] = round(comp(raw))

# ---- standings -> per-owner inflation bump (made playoffs +5 / non-playoff +3 / picked-champ +0) ----
stand = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_standings.csv"), encoding="utf-8")))
bump = {}                                              # (season, owner) -> $ inflation
for r in stand:
    if r["keeper_bump"] not in ("", "None"):
        bump[(r["season"], r["owner"])] = int(r["keeper_bump"])

# ---- 2025 draft -> TRUE keeper cost basis entering 2026 ----
drafts = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
def fbid(r):
    try: return float(r["bid"])
    except: return 0.0
basis25 = {}; keptyrs = {}                             # norm(name) -> true 2025 cost ; keeper-year count
for r in drafts:
    k = norm(r["player"]); kept = str(r["keeper"]).lower() in ("true", "1")
    if kept: keptyrs[k] = keptyrs.get(k, 0) + 1
    if r["season"] == "2025":
        base = max(fbid(r), 1)                         # undrafted/$0 floors at $1 (league rule)
        # a 2025 keeper was entered at base but OWED its keeper-owner's 2024 bump -> add it back
        basis25[k] = base + (bump.get(("2024", r["owner"]), 0) if kept else 0)

# ---- per-team keeper prediction ----
L = json.load(open(os.path.join(ROOT, "outputs", "espn_league.json")))
print(f"Predicted 2026 keepers — {L.get('name')} ({L.get('size')} teams). "
      f"cost = true 2025 keeper cost + this team's 2026 bump (made playoffs +$5 / non-playoff +$3 / "
      f"picked-champ +$0); waiver pickups = $1, no inflation. value = calibrated board $; keep top-3 surplus.\n")
for t in sorted(L.get("teams", []), key=lambda x: x["id"]):
    owner = t.get("owner"); b26 = bump.get(("2025", owner), 0)
    cand = []
    for p in t.get("roster", []):
        if p.get("pos") in ("K", "DST", "?"): continue
        k = norm(p.get("name") or ""); val = VAL.get((k, p.get("pos")))
        if val is None: continue
        basis = basis25.get(k); waiver = basis is None    # not in 2025 draft -> waiver pickup
        cost = 1 if waiver else round(basis + b26)
        cand.append((val - cost, p["name"], p["pos"], val, cost, keptyrs.get(k, 0), waiver, basis or 0, b26))
    cand.sort(key=lambda x: -x[0])
    keep = [c for c in cand if c[0] > 0][:3]
    me = "  <-- YOU" if t["id"] == L.get("myTeamId") else ""
    print(f"[{t['id']:>2}] {str(t['name'])[:26]:26s}{me}  (2026 bump +${b26})")
    for surplus, nm, pos, val, cost, ky, wv, basis, bp in keep:
        tag = f" (kept {ky}yr)" if ky else ""
        brk = " [$1 waiver keeper]" if wv else f" (=${int(round(basis))} +${bp})"
        print(f"     KEEP  {pos:<3} {nm[:22]:22s} value ${val:>2}  cost ${cost:>2}{brk}  surplus +${surplus:>2}{tag}")
    nxt = next((c for c in cand if c not in keep), None)
    if nxt: print(f"     next: {nxt[1]} (surplus {nxt[0]:+d})")
    print()

open(os.path.join(ROOT, "outputs", "espn_keepers_2026.md"), "w", encoding="utf-8").write(
    "# Predicted 2026 keepers — " + str(L.get("name")) + "\n\n```\n" + "\n".join(_out) + "\n```\n")
