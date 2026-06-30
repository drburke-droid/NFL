"""
Predict each team's likely 2026 keepers in the ESPN league.

For every player on a team's carried-over 2026 roster (outputs/espn_league.json):
  keeper_cost_2026 = 2025 acquisition cost (outputs/espn_drafts.csv) + finish inflation
  value_2026       = our calibrated Draft-Room auction value (risk-adjusted VONA)
  surplus          = value - cost
Keep the 3 best-surplus players per team (the rational keep). 3-yr cap doesn't bind for 2026
(keepers started 2024). Inflation is estimated (needs 2025 final standings for exact $).
"""
import os, json, csv, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFLATION = 3          # flat estimate of the finish-based keeper bump (champ +5 / top6 +3 / 7-11 +2 / last +0)
WAIVER_COST = 1        # league rule: waiver pickups are kept for $1, no inflation
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

# ---- 2025 acquisition cost + keeper history ----
drafts = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
def fbid(r):
    try: return float(r["bid"])
    except: return 0.0
cost25 = {}; keptyrs = {}
for r in drafts:
    k = norm(r["player"])
    if r["season"] == "2025": cost25[k] = fbid(r) or 0
    if str(r["keeper"]).lower() in ("true", "1"): keptyrs[k] = keptyrs.get(k, 0) + 1

# ---- per-team keeper prediction ----
L = json.load(open(os.path.join(ROOT, "outputs", "espn_league.json")))
print(f"Predicted 2026 keepers — {L.get('name')} ({L.get('size')} teams). "
      f"cost = 2025 price + ~${INFLATION} inflation (waiver pickups = $1, no inflation); "
      f"value = calibrated board $; keep top-3 surplus.\n")
for t in sorted(L.get("teams", []), key=lambda x: x["id"]):
    cand = []
    for p in t.get("roster", []):
        if p.get("pos") in ("K", "DST", "?"): continue
        k = norm(p.get("name") or ""); val = VAL.get((k, p.get("pos")))
        if val is None: continue
        base = cost25.get(k); waiver = base is None       # not in 2025 draft -> waiver pickup
        cost = WAIVER_COST if waiver else round(max(base, 1) + INFLATION)
        cand.append((val - cost, p["name"], p["pos"], val, cost, keptyrs.get(k, 0), waiver))
    cand.sort(key=lambda x: -x[0])
    keep = [c for c in cand if c[0] > 0][:3]
    me = "  <-- YOU" if t["id"] == L.get("myTeamId") else ""
    print(f"[{t['id']:>2}] {str(t['name'])[:26]:26s}{me}")
    for surplus, nm, pos, val, cost, ky, wv in keep:
        tag = f" (kept {ky}yr)" if ky else ""; w = " [$1 waiver keeper]" if wv else ""
        print(f"     KEEP  {pos:<3} {nm[:22]:22s} value ${val:>2}  cost ${cost:>2}  surplus +${surplus:>2}{tag}{w}")
    nxt = next((c for c in cand if c not in keep), None)
    if nxt: print(f"     next: {nxt[1]} (surplus {nxt[0]:+d})")
    print()

open(os.path.join(ROOT, "outputs", "espn_keepers_2026.md"), "w", encoding="utf-8").write(
    "# Predicted 2026 keepers — " + str(L.get("name")) + "\n\n```\n" + "\n".join(_out) + "\n```\n")
