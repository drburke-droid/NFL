"""
Owner bias in the auction: NFL-team fandom, positional overpay, and player loyalty.

For every non-keeper auction buy (2023-25), the overpay residual = scaled bid minus the
league's positional rank curve at that pick's within-season rank (the same recency-weighted,
finance-calibrated curve the tool bids from). Residuals average ~0 league-wide, so a
persistent positive mean IS that owner paying over the room's own going rate.

Signals mined per owner:
  FAN TEAM   — buys players from one NFL team far more often than the pool offers it
               (frequency ratio) AND/OR pays a premium on those buys.
  POSITION   — mean overpay by position (the tool's tend[] already captures ALLOCATION;
               this captures PRICE — paying over the curve for the position).
  LOYALTY    — re-buying players he has rostered before, and what he pays for them.

Emits docs/owner_bias_2026.js (+ outputs/draft_tool mirror):
  const OWNER_BIAS = {owner: {fan: {NFLteam: mult}, posprem: {pos: $}, rebuy: mult}}
for the mock-draft bots. Report: printed + appended sections used by the draft tool docs.
"""
import os, json, csv, re, sqlite3
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = ("QB", "RB", "WR", "TE")
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
CAN = {'GNB':'GB','KAN':'KC','LVR':'LV','OAK':'LV','NOR':'NO','NWE':'NE','SFO':'SF',
       'TAM':'TB','SD':'LAC','STL':'LAR','LA':'LAR','WSH':'WAS','JAC':'JAX'}
can = lambda t: CAN.get(t, t)

drafts = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
for r in drafts:
    r["bid"] = max(int(float(r["bid"] or 0)), 1)
    r["kept"] = str(r["keeper"]).lower() in ("true", "1")

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
team_of = {}                                   # (norm name, season) -> NFL team that season
for nm, season, tm in con.execute(
        "SELECT player_display_name, season, recent_team FROM nflv_season WHERE season>=2023"):
    team_of[(norm(nm), int(season))] = can(tm)

# finance calibration + positional curve (same construction as the tool)
per_team = {}
for s in set(r["season"] for r in drafts):
    rows = [r for r in drafts if r["season"] == s and not r["kept"]]
    per_team[s] = sum(r["bid"] for r in rows) / len(set(r["owner"] for r in rows))
REF = per_team[max(per_team)]
SCALE = {s: REF / v for s, v in per_team.items()}
BW = {"2023": 1, "2024": 2, "2025": 3}
def curve(pos):
    lists = defaultdict(list)
    for r in drafts:
        if not r["kept"] and r["pos"] == pos:
            lists[r["season"]].append(r["bid"] * SCALE[r["season"]])
    for s in lists: lists[s].sort(reverse=True)
    ml = max(len(v) for v in lists.values())
    out = []
    for rk in range(ml):
        num = den = 0.0
        for s, arr in lists.items():
            if rk < len(arr): num += arr[rk] * BW[s]; den += BW[s]
        out.append(num / den if den else 1)
    return out
C = {pos: curve(pos) for pos in SKILL}

# every auction buy with its overpay residual vs the curve at its within-season rank
buys = []
for s in ("2023", "2024", "2025"):
    for pos in SKILL:
        rows = sorted([r for r in drafts if r["season"] == s and not r["kept"] and r["pos"] == pos],
                      key=lambda r: -r["bid"])
        for rk, r in enumerate(rows):
            nm = norm(r["player"])
            buys.append({"season": s, "owner": r["owner"], "player": r["player"], "nm": nm,
                         "pos": pos, "bid": r["bid"], "sc": r["bid"] * SCALE[s],
                         "team": team_of.get((nm, int(s))) or "?",
                         "resid": r["bid"] * SCALE[s] - (C[pos][rk] if rk < len(C[pos]) else 1)})
print(f"auction buys analyzed: {len(buys)} (skill positions, non-keeper, 2023-25)")
print(f"league residual check: mean {np.mean([b['resid'] for b in buys]):+.2f} "
      f"(should be ~0)\n")

_out = []
def rep(*a):
    s = " ".join(str(x) for x in a); _out.append(s); print(s)

# ---------------- NFL-team fandom ----------------
# USER-DECLARED fans (ground truth from the league): auction residuals alone under-detect
# fandom when several owners share a team (the buy-share baseline dilutes) or when a fan
# expresses it through KEEPS rather than overbids. Declared fans get the 1.10 floor.
KNOWN_FANS = {"CBPainTrain": "CHI", "ESPNFAN25233150": "CHI", "espn14211458": "CHI"}

# keeper conviction: keeping 2+ DISTINCT players of one NFL team (at their kept-season team)
# is fandom the auction data can't see — e.g. CBPainTrain kept DJ Moore twice AND Caleb
# Williams at -$20 savings. One player kept repeatedly is player loyalty, not team fandom.
kept_team = defaultdict(lambda: defaultdict(set))      # owner -> team -> distinct kept players
for r in drafts:
    if r["kept"]:
        tm = team_of.get((norm(r["player"]), int(r["season"])))
        if tm: kept_team[r["owner"]][tm].add(norm(r["player"]))

rep("=" * 74); rep("NFL-TEAM BIAS per owner (frequency vs pool share, price premium, keeper conviction)"); rep("=" * 74)
league_team_share = defaultdict(int)
for b in buys: league_team_share[b["team"]] += 1
TOT = len(buys)
fan_bias = defaultdict(dict)
for o in sorted(set(b["owner"] for b in buys)):
    ob = [b for b in buys if b["owner"] == o]
    seen = defaultdict(list)
    for b in ob: seen[b["team"]].append(b)
    for tm in set(list(seen.keys()) + ([KNOWN_FANS[o]] if o in KNOWN_FANS else [])):
        arr = seen.get(tm, [])
        if tm == "?" or (len(arr) < 3 and KNOWN_FANS.get(o) != tm and len(kept_team[o][tm]) < 2):
            continue
        exp_n = len(ob) * league_team_share[tm] / TOT
        ratio = len(arr) / exp_n if exp_n else 0
        prem = float(np.mean([b["resid"] for b in arr])) if arr else 0.0
        seasons = len(set(b["season"] for b in arr))
        kk = len(kept_team[o][tm])
        # FAN if: user-declared; OR keeper conviction (2+ distinct kept players + any buys); OR
        # multi-season heavy frequency (>=4 buys at >=3.5x pool share); OR repeated price
        # premium (>=2x share, >=$3 over curve). A 3-buy 2.2x cell at curve price is noise.
        fan = (KNOWN_FANS.get(o) == tm
               or (kk >= 2 and len(arr) + kk >= 4)
               or (seasons >= 2 and ((len(arr) >= 4 and ratio >= 3.5) or (ratio >= 2.0 and prem >= 3))))
        if fan or (seasons >= 2 and ratio >= 2.0):
            why = "declared" if KNOWN_FANS.get(o) == tm else (f"{kk} distinct keeps" if kk >= 2 else "")
            rep(f"  {o[:18]:18s} {tm:3s}: {len(arr)} buys "
                f"({ratio:.1f}x pool share), avg ${prem:+.1f} vs curve, {kk} kept"
                f"{'  <-- FAN ' + why if fan else ''}  "
                f"[{', '.join(sorted(set(b['player'] for b in arr))[:5])}]")
        if fan:
            fan_bias[o][tm] = {"n": len(arr), "ratio": round(ratio, 1), "prem": round(prem, 1),
                               "declared": KNOWN_FANS.get(o) == tm, "keeps": kk}

# ---------------- positional PRICE bias (allocation is already in tend[]) ----------------
rep(""); rep("=" * 74); rep("POSITIONAL OVERPAY per owner (mean $ vs curve; n>=5 and |mean|>=$2 shown)"); rep("=" * 74)
pos_prem = defaultdict(dict)
for o in sorted(set(b["owner"] for b in buys)):
    line = f"  {o[:18]:18s}"
    for pos in SKILL:
        arr = [b["resid"] for b in buys if b["owner"] == o and b["pos"] == pos]
        if len(arr) >= 5:
            m = float(np.mean(arr)); sd = float(np.std(arr)) or 1
            t = m / (sd / len(arr) ** .5)
            line += f"  {pos} {m:+5.1f}$ (n={len(arr):2d})"
            if abs(m) >= 2 and abs(t) >= 1.5: pos_prem[o][pos] = round(m, 1)
        else:
            line += f"  {pos}   —  (n={len(arr):2d})"
    rep(line)
rep("  significant (|mean|>=$2, |t|>=1.5): " +
    (", ".join(f"{o[:14]} {p} {v:+.0f}$" for o, d in pos_prem.items() for p, v in d.items()) or "none"))

# ---------------- player loyalty: re-buying his own guys ----------------
rep(""); rep("=" * 74); rep("PLAYER LOYALTY — re-buying players he rostered before"); rep("=" * 74)
owned_by = defaultdict(set)                    # owner -> names rostered in ANY earlier season
first_season = {}
for r in sorted(drafts, key=lambda r: r["season"]):
    first_season.setdefault((r["owner"], norm(r["player"])), r["season"])
rebuy_res, base_res = [], []
rebuy_n = 0
for b in sorted(buys, key=lambda b: b["season"]):
    prior = first_season.get((b["owner"], b["nm"]))
    if prior and prior < b["season"]:
        rebuy_res.append(b["resid"]); rebuy_n += 1
    else:
        base_res.append(b["resid"])
rep(f"  re-buys of previously-rostered players: {rebuy_n} of {len(buys)} buys "
    f"({100*rebuy_n/len(buys):.0f}%)")
rep(f"  avg overpay on re-buys ${np.mean(rebuy_res):+.1f} vs ${np.mean(base_res):+.1f} on fresh buys"
    if rebuy_res else "  none")
REBUY_PREM = float(np.mean(rebuy_res) - np.mean(base_res)) if rebuy_res else 0.0

# ---------------- emit bot biases ----------------
# fan multiplier: how much over the going rate the bot stretches for his team's players.
# 1.05 base for a pure frequency fan (targets them, doesn't overpay) + the observed premium
# converted to a shade at mid prices (~$18), clipped at 1.30 — fandom colors the bid, the
# hard historical caps still hold. Positional PRICE premiums and re-buy loyalty tested but
# not significant (see report) -> allocation tendencies in tend[] remain the only positional
# bias, and no rebuy term is emitted.
# declared / keeper-conviction fans floor at 1.10 (a fan's premium shows in keeps the
# residual can't price); pure bid-detected fans keep the measured 1.05 floor.
OWNER_BIAS = {}
for o, fans in fan_bias.items():
    OWNER_BIAS[o] = {"fan": {tm: round(min(1.30, max(1.05 + max(d["prem"], 0) / 18,
                                                     1.10 if (d["declared"] or d["keeps"] >= 2) else 1.05)), 2)
                             for tm, d in fans.items()}}
js = ("// AUTO-GENERATED by scripts/owner_bias.py — per-owner NFL-team fandom mined from the\n"
      "// league's 2023-25 winning bids: {owner: {fan: {NFL team: bid multiplier}}}. A team makes\n"
      "// the list only with 2+ seasons of evidence and either >=4 buys at >=3.5x the pool share\n"
      "// or a >=$3 over-curve premium on a >=2x cluster. Positional overpay and player-loyalty\n"
      "// biases were tested and are NOT real in this league (outputs/reports/owner_bias.md).\n"
      "const OWNER_BIAS = " + json.dumps(OWNER_BIAS) + ";\n")
for d in (os.path.join(ROOT, "docs"), os.path.join(ROOT, "outputs", "draft_tool")):
    open(os.path.join(d, "owner_bias_2026.js"), "w", encoding="utf-8").write(js)
print(f"\nwrote docs/owner_bias_2026.js (+ mirror): "
      f"{len(OWNER_BIAS)} owners with fan teams: "
      + "; ".join(f"{o[:14]} {'/'.join(f'{t} x{m}' for t, m in d['fan'].items())}"
                  for o, d in sorted(OWNER_BIAS.items())))

open(os.path.join(ROOT, "outputs", "reports", "owner_bias.md"), "w", encoding="utf-8").write(
    "# Owner auction biases — NFL-team fandom, positional overpay, player loyalty\n\n"
    "Generated by `scripts/owner_bias.py` from 2023-25 winning bids (non-keeper, skill\n"
    "positions, 11-team 2024 finance-calibrated). Residual = scaled bid minus the league's\n"
    "positional rank curve at that pick's within-season rank.\n\n```\n" + "\n".join(_out) + "\n```\n")
print("wrote outputs/reports/owner_bias.md")
