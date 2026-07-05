"""Owner auction-spending profiles for the ESPN league (2023-25 drafts, keepers included at cost).
1) Avg $ per roster SLOT per owner: highest-$ QB -> QB, top-2 RB -> RB1/RB2, top-2 WR -> WR1/WR2,
   best TE -> TE, best remaining RB/WR/TE -> FLEX, K, DST, rest -> bench B1..B7 (ranked by $).
2) Winning-bid lists per owner x position (sorted, season-tagged).
Writes outputs/reports/owner_spending.md."""
import os, csv, json, sys
from collections import defaultdict
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dr = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
L = json.load(open(os.path.join(ROOT, "outputs", "espn_league.json"), encoding="utf-8"))
NAME = {t.get("owner"): t.get("name") for t in L.get("teams", [])}
def bid(r):
    try: return max(float(r["bid"]), 1.0)
    except Exception: return 1.0
SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "TE", "FLEX", "K", "DST"] + [f"B{i}" for i in range(1, 8)]
def assign(picks):
    """greedy slot assignment by $ within position"""
    by = defaultdict(list)
    for p in picks: by[p["pos"]].append(p)
    for k in by: by[k].sort(key=lambda x: -x["b"])
    out = {}
    def take(pos, n):
        got = by.get(pos, [])[:n]; by[pos] = by.get(pos, [])[n:]; return got
    q = take("QB", 1);  out["QB"] = q[0]["b"] if q else None
    r = take("RB", 2);  out["RB1"] = r[0]["b"] if r else None; out["RB2"] = r[1]["b"] if len(r) > 1 else None
    w = take("WR", 2);  out["WR1"] = w[0]["b"] if w else None; out["WR2"] = w[1]["b"] if len(w) > 1 else None
    t = take("TE", 1);  out["TE"] = t[0]["b"] if t else None
    flexpool = sorted(by.get("RB", []) + by.get("WR", []) + by.get("TE", []), key=lambda x: -x["b"])
    if flexpool:
        fx = flexpool[0]; out["FLEX"] = fx["b"]; by[fx["pos"]].remove(fx)
    else: out["FLEX"] = None
    k = take("K", 1);   out["K"] = k[0]["b"] if k else None
    d = take("DST", 1); out["DST"] = d[0]["b"] if d else None
    bench = sorted([p for ps in by.values() for p in ps], key=lambda x: -x["b"])
    for i in range(7): out[f"B{i+1}"] = bench[i]["b"] if i < len(bench) else None
    return out
rows = defaultdict(lambda: defaultdict(list))          # owner -> slot -> [values]
bids = defaultdict(lambda: defaultdict(list))          # owner -> pos -> [(bid, season, player, keeper)]
seasons = sorted({r["season"] for r in dr})
for (own, season), picks in [((o, s), [r for r in dr if r["owner"] == o and r["season"] == s])
                             for o in {r["owner"] for r in dr} for s in seasons]:
    if not picks: continue
    pl = [{"pos": r["pos"], "b": bid(r)} for r in picks if r["pos"] not in ("?",)]
    sl = assign(pl)
    for s2, v in sl.items():
        if v is not None: rows[own][s2].append(v)
    for r in picks:
        if r["pos"] in ("QB", "RB", "WR", "TE", "K", "DST"):
            bids[own][r["pos"]].append((bid(r), r["season"], r["player"],
                                        str(r["keeper"]).lower() in ("true", "1")))
own_order = sorted(rows, key=lambda o: -(sum(sum(v) for v in rows[o].values())))
lines = []
lines.append("# Owner auction spending — Kuhn and Friends (2023–25, keepers at cost)\n")
lines.append("## Average $ per roster slot (across seasons)\n")
hdr = f"{'owner/team':26s}" + "".join(f"{s:>6s}" for s in SLOTS) + f"{'BENCHtot':>9s}"
lines.append("```\n" + hdr)
for o in own_order:
    nm = (NAME.get(o) or o)[:25]
    vals = []
    for s in SLOTS:
        v = rows[o].get(s)
        vals.append(f"{sum(v)/len(v):>6.0f}" if v else f"{'-':>6s}")
    btot = sum(sum(rows[o].get(f'B{i}', [0])) / max(len(rows[o].get(f'B{i}', [1])), 1) for i in range(1, 8))
    lines.append(f"{nm:26s}" + "".join(vals) + f"{btot:>9.0f}")
lines.append("```\n")
lines.append("## Winning bids by position (sorted; k = keeper cost, year)\n")
for o in own_order:
    nm = NAME.get(o) or o
    lines.append(f"### {nm}")
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        bl = sorted(bids[o].get(pos, []), reverse=True)
        if not bl: continue
        s = ", ".join(f"${b:.0f}{'k' if kp else ''}('{sea[2:]}{'' if not p else ' ' + p.split()[-1]})" for b, sea, p, kp in bl)
        lines.append(f"- **{pos}**: {s}")
    lines.append("")
txt = "\n".join(lines)
open(os.path.join(ROOT, "outputs", "reports", "owner_spending.md"), "w", encoding="utf-8").write(txt)
print(txt[:6000])
print("...\nWrote outputs/reports/owner_spending.md")
