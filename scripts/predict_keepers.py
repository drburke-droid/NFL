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
import os, json, csv, re, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower())).replace("  ", " ").strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

# ---- our calibrated auction value per player ----
P = json.loads(open(os.path.join(ROOT, "docs", "data.js"), encoding="utf-8").read().split("const PLAYERS = ")[1].rsplit(";", 1)[0])
try:  # blended healthy PPG for injury-return players (scripts/healthy_ppg.py); keyed by exact name
    HEALTHY = json.loads(open(os.path.join(ROOT, "docs", "healthy_ppg.js"), encoding="utf-8").read().split("= ", 1)[1].rstrip(";\n"))
except Exception:
    HEALTHY = {}
try:  # 2026 walk-year (contract-year) value bump (scripts/contract_status.py); keyed by exact name
    CONTRACT = json.loads(open(os.path.join(ROOT, "docs", "contract_status.js"), encoding="utf-8").read().split("= ", 1)[1].rstrip(";\n"))
except Exception:
    CONTRACT = {}
try:  # 2026 clean-vacancy WR inheritor bump (scripts/vacated_role_2026.py)
    VACATED = json.loads(open(os.path.join(ROOT, "docs", "vacated_role.js"), encoding="utf-8").read().split("= ", 1)[1].rstrip(";\n"))
except Exception:
    VACATED = {}
isK = lambda p: p["position"] in ("K", "DST"); INJ = {"QB": .26, "RB": .40, "WR": .33, "TE": .39}
def risk(p):
    if isK(p): return 0
    ppg = p.get("proj_ppg") or 0; fl = p.get("floor") if p.get("floor") is not None else ppg; bu = p.get("bust") if p.get("bust") is not None else .3
    inj = max(0, ((INJ.get(p["position"], .33) - .26) / .14)); down = max(0, (ppg - fl) / ppg) if ppg > 0 else 0
    return min(.9, .45 * inj + .40 * bu + .30 * down)
# FULL-SEASON VALUE: value established players on ppg x a normal-season games baseline, so a
# resolved past injury (low projected games) doesn't bury a healthy bounce-back. Only ever LIFTS
# (max with proj_pts), so it targets injury-returns and never inflates the healthy board. Baseline
# = median games among each position's projected starters (rookies excluded from the treatment).
_DEMAND = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
NORMG = {}
for _pos, _n in _DEMAND.items():
    _arr = sorted([p for p in P if p["position"] == _pos], key=lambda p: -(p.get("proj_pts") or 0))[:_n]
    _gs = [p.get("proj_games") for p in _arr if p.get("proj_games")]
    NORMG[_pos] = statistics.median(_gs) if _gs else 15
def eff_pts(p):
    pts = p.get("proj_pts") or 0
    if p.get("is_rookie") or p["position"] not in NORMG: return pts
    ppg = HEALTHY.get(p["name"]) or p.get("proj_ppg"); ng = NORMG.get(p["position"])  # healthy-rate override
    base = max(pts, ppg * ng) if (ppg and ng) else pts
    cb = (CONTRACT.get(p["name"]) or {}).get("bump", 0)                                # walk-year bump
    vb = (VACATED.get(p["name"]) or {}).get("bump", 0)                                 # vacated-role bump
    return base + (cb + vb) * (ng or 0)
ra = lambda p: eff_pts(p) * (1 - .7 * risk(p))
open_ = {"QB": 12, "RB": 24, "WR": 24, "TE": 12, "FLEX": 12}
fa = lambda pos: open_["FLEX"] * ({"RB": .45, "WR": .45, "TE": .10}.get(pos, 0))
BENCH_CAP = 6                                          # bubble/bench players ramp $BENCH_CAP -> $1 (no cliff)
DEEP_MULT = {"QB": 1.4, "RB": 1.5, "WR": 1.5, "TE": 1.1}   # per-pos "last rosterable" depth (auction data)
repl = {}; rdeep = {}; emarg = {}; within = set()
for pos in ("QB", "RB", "WR", "TE"):
    n = round(open_[pos] + fa(pos)); arr = sorted([p for p in P if p["position"] == pos], key=ra, reverse=True)
    repl[pos] = ra(arr[min(max(n - 1, 0), len(arr) - 1)])
    for p in arr[:n]: within.add(p["name"])
    rdeep[pos] = ra(arr[min(len(arr) - 1, round(n * DEEP_MULT[pos]))])   # deep "last rosterable" replacement
    emarg[pos] = max(repl[pos] - rdeep[pos], 1e-6)
sumE = sum(max(ra(p) - repl[p["position"]], 0) for p in P if p["name"] in within)
per = (12 * 200 - 12 * 16) / sumE
comp = lambda v: v if v <= 25 else 25 + (v - 25) * 0.75
VAL = {}; PMAP = {}
for p in P:
    if isK(p): continue
    pos = p["position"]
    e = max(ra(p) - repl[pos], 0); raw = (1 + e * per) if p["name"] in within else 1
    floor = min(BENCH_CAP, BENCH_CAP * max(ra(p) - rdeep[pos], 0) / emarg[pos])   # smooth bench floor
    VAL[(norm(p["name"]), pos)] = round(max(comp(raw), floor, 1)); PMAP[(norm(p["name"]), pos)] = p
pidof = lambda p: p["name"] + "|" + p["position"] + "|" + (p.get("team") or "")  # matches BYID in index.html

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

# ---- league bid curves: typical non-keeper auction $ by rank (recency-weighted) ----
# How the league ACTUALLY bids — used to simulate expected purchase prices. We build a curve PER
# POSITION (QB/RB/WR/TE), because positions have very different ceilings (WR/RB top ~$64, QB ~$34,
# TE ~$28 — see outputs/reports/position_auction_curves.md). Pricing the r-th most valuable QB off
# the QB curve (not a pooled curve) stops elite QBs being priced like elite WR/RB.
SKILL = ("QB", "RB", "WR", "TE")
BW = {"2023": 1, "2024": 2, "2025": 3}                  # recent seasons weighted more
def build_curve(season_lists):                          # {season: [bids desc]} -> smoothed $/rank
    ml = max((len(v) for v in season_lists.values()), default=0)
    raw = []
    for rk in range(ml):
        num = den = 0.0
        for s, arr in season_lists.items():
            if rk < len(arr): w = BW.get(s, 1); num += arr[rk] * w; den += w
        raw.append(num / den if den else 0)
    sm = raw[:]                                         # light 3-pt smoothing, keep the top 2 sharp
    for i in range(2, len(raw) - 1): sm[i] = (raw[i - 1] + raw[i] + raw[i + 1]) / 3
    return [max(1, round(x)) for x in sm]

def season_lists(pos=None):
    out = {}
    for r in drafts:
        if str(r["keeper"]).lower() in ("true", "1"): continue
        if r["pos"] not in SKILL or (pos and r["pos"] != pos): continue
        b = fbid(r)
        if b >= 1: out.setdefault(r["season"], []).append(b)
    for s in out: out[s].sort(reverse=True)
    return out

bid_curve = build_curve(season_lists())                          # pooled (fallback)
bid_curve_pos = {pos: build_curve(season_lists(pos)) for pos in SKILL}   # per-position

# ---- per-owner positional spend tendencies (non-keeper auction picks, recency-weighted) ----
YW = {"2023": 1, "2024": 2, "2025": 3}                 # recent behavior weighted more
osp = {}; omax = {}                                    # owner -> {pos: weighted$} ; owner -> {pos: max single bid}
for r in drafts:
    if str(r["keeper"]).lower() in ("true", "1"): continue
    pos = r["pos"]
    if pos not in ("QB", "RB", "WR", "TE"): continue
    o = r["owner"]; b = fbid(r); w = YW.get(r["season"], 1)
    sp = osp.setdefault(o, {}); sp[pos] = sp.get(pos, 0) + b * w
    mx = omax.setdefault(o, {}); mx[pos] = max(mx.get(pos, 0), b)
shares = {}
for o, sp in osp.items():
    tot = sum(sp.values()) or 1
    shares[o] = {p: sp.get(p, 0) / tot for p in ("QB", "RB", "WR", "TE")}
avg = {p: (sum(shares[o][p] for o in shares) / len(shares) if shares else .25) for p in ("QB", "RB", "WR", "TE")}
def tend_for(o):
    sh = shares.get(o, {p: .25 for p in avg}); mx = omax.get(o, {})
    clip = lambda v: round(max(0.4, min(2.2, v)), 2)
    td = {p: clip(sh[p] / (avg[p] or .25)) for p in ("QB", "RB", "WR", "TE")}
    td["payTE"] = mx.get("TE", 0) >= 15            # has paid up for a TE -> may chase a 2nd
    td["eliteQB"] = mx.get("QB", 0) >= 18          # buys a real QB rather than streaming
    td["top"] = max(list(mx.values()) or [0])      # their biggest single buy (stars-and-scrubs)
    return td

# ---- expected auction (redraft) price per player, from the position bid curve ----
# The r-th most valuable player at a position is expected to redraft at ~the $ the league pays for
# its r-th most expensive buy at that position. This is the key to the KEEP decision: you only keep
# a player if keeping is CHEAPER than redrafting him (exp > cost) AND he's good value (value > cost).
exp_price = {}
for _pos in SKILL:
    _arr = sorted([p for p in P if p["position"] == _pos and not isK(p)],
                  key=lambda p: -(VAL.get((norm(p["name"]), _pos)) or 0))
    _C = bid_curve_pos.get(_pos, [])
    for _r, p in enumerate(_arr):
        _v = VAL.get((norm(p["name"]), _pos)) or 0
        exp_price[pidof(p)] = (_C[_r] if _r < len(_C) else 1) if _v > 1 else 1

# ---- per-team keeper prediction ----
L = json.load(open(os.path.join(ROOT, "outputs", "espn_league.json")))
print(f"Predicted 2026 keepers — {L.get('name')} ({L.get('size')} teams). "
      f"cost = true 2025 keeper cost + this team's 2026 bump. KEEP only if keeping beats redrafting "
      f"(exp > cost) AND good value (value > cost); rank by $ saved. 'redraft' = good value but cheaper to re-buy.\n")
teams_out = []
for t in sorted(L.get("teams", []), key=lambda x: x["id"]):
    owner = t.get("owner"); b26 = bump.get(("2025", owner), 0)
    cand = []
    for p in t.get("roster", []):
        if p.get("pos") in ("K", "DST", "?"): continue
        k = norm(p.get("name") or ""); val = VAL.get((k, p.get("pos")))
        if val is None: continue
        pl = PMAP[(k, p.get("pos"))]; cid = pidof(pl)      # canonical board player
        basis = basis25.get(k); waiver = basis is None     # not in 2025 draft -> waiver pickup
        cost = 1 if waiver else round(basis + b26)
        exp = exp_price.get(cid, 1); savings = exp - cost; good = val > cost
        decision = "keep" if (good and savings > 0) else ("redraft" if good else "pass")
        cand.append({"pid": cid, "name": pl["name"], "pos": pl["position"], "team": pl.get("team") or "",
                     "value": val, "cost": cost, "exp": exp, "savings": savings, "surplus": val - cost,
                     "decision": decision, "basis": round(basis or 0), "bump": b26,
                     "waiver": waiver, "keptyrs": keptyrs.get(k, 0)})
    keep = sorted([c for c in cand if c["decision"] == "keep"], key=lambda c: -c["savings"])[:3]
    keepids = {c["pid"] for c in keep}
    for c in cand: c["predicted"] = c["pid"] in keepids    # the model's recommended keeps
    cand.sort(key=lambda c: (c["decision"] != "keep", c["decision"] != "redraft", -c["savings"], -c["surplus"]))
    teams_out.append({"id": t["id"], "name": t.get("name"), "owner": owner, "bump2026": b26,
                      "tend": tend_for(owner), "candidates": cand})
    me = "  <-- YOU" if t["id"] == L.get("myTeamId") else ""
    print(f"[{t['id']:>2}] {str(t['name'])[:26]:26s}{me}  (2026 bump +${b26})")
    for c in keep:
        tag = f" (kept {c['keptyrs']}yr)" if c["keptyrs"] else ""
        print(f"     KEEP    {c['pos']:<3} {c['name'][:22]:22s} value ${c['value']:>2}  cost ${c['cost']:>2}  "
              f"exp ${c['exp']:>2}  saves +${c['savings']:>2}{tag}")
    for c in [c for c in cand if c["decision"] == "redraft"][:3]:
        print(f"     redraft {c['pos']:<3} {c['name'][:22]:22s} value ${c['value']:>2}  cost ${c['cost']:>2}  "
              f"exp ${c['exp']:>2}  (re-buy ${c['exp']} < keep ${c['cost']})")
    print()

open(os.path.join(ROOT, "outputs", "espn_keepers_2026.md"), "w", encoding="utf-8").write(
    "# Predicted 2026 keepers — " + str(L.get("name")) + "\n\n```\n" + "\n".join(_out) + "\n```\n")

# ---- structured data for the Mock Keepers screen (docs/keepers_2026.js) ----
slots = L.get("roster_slots") or {}
data = {"league": {"name": L.get("name"), "size": L.get("size"), "myTeamId": L.get("myTeamId"),
                   "budget": ((L.get("draft") or {}).get("auctionBudget")) or 200,
                   "roster": sum(v for k, v in slots.items() if k != "IR") or 16},
        "bidcurve": bid_curve, "bidcurve_pos": bid_curve_pos, "teams": teams_out}
js = "// AUTO-GENERATED by scripts/predict_keepers.py — per-team 2026 keeper candidates (value-ranked).\n" \
     "const KEEPERS_2026 = " + json.dumps(data, separators=(",", ":")) + ";\n"
for d in (os.path.join(ROOT, "docs"), os.path.join(ROOT, "outputs", "draft_tool")):
    try: open(os.path.join(d, "keepers_2026.js"), "w", encoding="utf-8").write(js)
    except OSError: pass
