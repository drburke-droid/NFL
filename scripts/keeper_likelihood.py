"""
How does THIS league actually keep players — and who will each owner keep in 2026?

Part 1 (investigate): reconstruct every owner's historical keep/pass decision from the
2023-25 drafts: eligible pool entering year Y = players the owner drafted (or kept) in
Y-1; kept = flagged keeper in Y's draft. Waiver/trade keeps (10 of 31 in 2024, 5 of 30
in 2025) have no observable "pass" pool, so they inform the patterns but not the fit.

Part 2 (model): logistic regression on the drafted-pool decisions — savings (expected
redraft $ minus keeper cost), keeper cost, prior-season PPG, kept-before flag.
Validated season-out (train 2024 -> test 2025 and reverse) on AUC + top-3 hit rate
against the rational rule predict_keepers.py uses (keep the 3 best positive-savings).

Part 3 (score): apply the fit to every skill player on every 2026 roster, using the
SAME cost / Exp $ the draft tool displays (docs/keepers_2026.js candidates, which
predict_keepers.py builds with the anchored expected prices). Emits
outputs/reports/keeper_likelihood_2026.md; does NOT touch the predicted-keeper files.
"""
import os, json, csv, re, sqlite3
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = ("QB", "RB", "WR", "TE")
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

# ---------------- data ----------------
drafts = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
for r in drafts:
    r["bid"] = max(int(float(r["bid"] or 0)), 1)
    r["kept"] = str(r["keeper"]).lower() in ("true", "1")

stand = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_standings.csv"), encoding="utf-8")))
bump = {(r["season"], r["owner"]): int(r["keeper_bump"]) for r in stand
        if r["keeper_bump"] not in ("", "None")}

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
ppg, games = {}, {}
for nm, season, g, ppr in con.execute(
        "SELECT player_display_name, season, games, fantasy_points_ppr FROM nflv_season WHERE season>=2022"):
    if g and ppr is not None:
        ppg[(norm(nm), int(season))] = ppr / g
        games[(norm(nm), int(season))] = g
id2nm = dict(con.execute("SELECT DISTINCT player_id, player_display_name FROM nflv_season"))
age_by_name = {}
for pid_, season, a in con.execute(
        "SELECT player_id, season, age FROM season_dataset WHERE age IS NOT NULL"):
    if pid_ in id2nm: age_by_name[(norm(id2nm[pid_]), int(season))] = a

# positional $/rank curve from ALL seasons (recency-weighted, finance-calibrated) — the
# market-level expected-redraft price for the r-th best returning player at a position
per_team = {}
for s in set(r["season"] for r in drafts):
    rows = [r for r in drafts if r["season"] == s and not r["kept"]]
    per_team[s] = sum(r["bid"] for r in rows) / len(set(r["owner"] for r in rows))
REF = per_team[max(per_team)]
SCALE = {s: REF / v for s, v in per_team.items()}
BW = {"2023": 1, "2024": 2, "2025": 3}
def curve(pos, skip_season=None):
    lists = {}
    for r in drafts:
        if r["kept"] or r["pos"] != pos or r["season"] == skip_season: continue
        lists.setdefault(r["season"], []).append(r["bid"] * SCALE[r["season"]])
    for s in lists: lists[s].sort(reverse=True)
    ml = max((len(v) for v in lists.values()), default=0)
    out = []
    for rk in range(ml):
        num = den = 0.0
        for s, arr in lists.items():
            if rk < len(arr): num += arr[rk] * BW.get(s, 1); den += BW.get(s, 1)
        out.append(max(1, num / den if den else 1))
    return out

# ---------------- decision rows: (year, owner, player) -> kept? ----------------
def build_rows(Y, skip_curve_season=None):
    """Eligible pool entering Y = players each Y-active owner drafted/kept in Y-1."""
    prev = str(Y - 1)
    owners_Y = set(r["owner"] for r in drafts if r["season"] == str(Y))
    kept_Y = {}                                     # norm name -> keeping owner
    for r in drafts:
        if r["season"] == str(Y) and r["kept"]: kept_Y[norm(r["player"])] = r["owner"]
    kept_prev = set(norm(r["player"]) for r in drafts if r["season"] == prev and r["kept"])
    C = {pos: curve(pos, skip_curve_season) for pos in SKILL}
    # value-rank proxy: prior-season PPG rank within position across the whole eligible pool
    pool = [r for r in drafts if r["season"] == prev and r["owner"] in owners_Y and r["pos"] in SKILL]
    rank = {}
    for pos in SKILL:
        arr = sorted([r for r in pool if r["pos"] == pos],
                     key=lambda r: -(ppg.get((norm(r["player"]), Y - 1), 0)))
        for i, r in enumerate(arr): rank[norm(r["player"])] = i
    rows, seen = [], set()
    for r in pool:
        nm = norm(r["player"])
        if (r["owner"], nm) in seen: continue
        seen.add((r["owner"], nm))
        if nm in kept_Y and kept_Y[nm] != r["owner"]: continue      # traded away, not a pass
        cost = r["bid"] + bump.get((prev, r["owner"]), 0)
        exp = C[r["pos"]][rank[nm]] if rank.get(nm, 999) < len(C[r["pos"]]) else 1
        rows.append({"year": Y, "owner": r["owner"], "name": r["player"], "pos": r["pos"],
                     "cost": cost, "exp": exp, "savings": exp - cost,
                     "ppg": ppg.get((nm, Y - 1), 0.0), "gms": games.get((nm, Y - 1), 0),
                     "age": age_by_name.get((nm, Y - 1), 26.0),
                     "keptprev": 1.0 if nm in kept_prev else 0.0,
                     "kept": 1.0 if kept_Y.get(nm) == r["owner"] else 0.0})
    return rows

hist = build_rows(2024) + build_rows(2025)

# ---------------- Part 1: how the league actually keeps ----------------
print("=" * 74)
print("PART 1 — how this league has actually kept (2024 & 2025 keeper drafts)")
print("=" * 74)
for Y in (2024, 2025):
    ks = [r for r in drafts if r["season"] == str(Y) and r["kept"]]
    by_owner = {}
    for r in ks: by_owner.setdefault(r["owner"], []).append(r)
    n_owners = len(set(r["owner"] for r in drafts if r["season"] == str(Y)))
    print(f"\nentering {Y}: {len(ks)} keeps by {len(by_owner)}/{n_owners} owners "
          f"(max 3/team -> league used {len(ks)}/{3*n_owners} slots)")
    full3 = sum(1 for v in by_owner.values() if len(v) >= 3)
    print(f"  kept 3: {full3} owners | kept 2: {sum(1 for v in by_owner.values() if len(v)==2)}"
          f" | kept 1: {sum(1 for v in by_owner.values() if len(v)==1)}"
          f" | kept 0: {n_owners-len(by_owner)}")
    print(f"  keeper cost: min ${min(r['bid'] for r in ks)}  median "
          f"${int(np.median([r['bid'] for r in ks]))}  max ${max(r['bid'] for r in ks)}"
          f"  | at $1-6: {sum(1 for r in ks if r['bid']<=6)} of {len(ks)}")
    posmix = {p: sum(1 for r in ks if r["pos"] == p) for p in SKILL}
    print(f"  positions kept: {posmix}")

rows = hist
print(f"\ndrafted-pool decisions (kept vs passed on own draftees): {len(rows)} rows, "
      f"{int(sum(r['kept'] for r in rows))} keeps")
print("\nkeep rate by SAVINGS (expected redraft $ - keeper cost):")
for lo, hi in ((-99, 0), (0, 5), (5, 10), (10, 20), (20, 99)):
    b = [r for r in rows if lo <= r["savings"] < hi]
    if b: print(f"  ${lo:>3} to ${hi:>2}: {100*sum(r['kept'] for r in b)/len(b):5.1f}%  ({int(sum(r['kept'] for r in b))}/{len(b)})")
print("keep rate by KEEPER COST:")
for lo, hi in ((1, 5), (5, 15), (15, 30), (30, 99)):
    b = [r for r in rows if lo <= r["cost"] < hi]
    if b: print(f"  ${lo:>2} to ${hi:>2}: {100*sum(r['kept'] for r in b)/len(b):5.1f}%  ({int(sum(r['kept'] for r in b))}/{len(b)})")
print("keep rate by prior-season PPG:")
for lo, hi in ((0, 8), (8, 12), (12, 16), (16, 99)):
    b = [r for r in rows if lo <= r["ppg"] < hi]
    if b: print(f"  {lo:>2}-{hi:>2} ppg: {100*sum(r['kept'] for r in b)/len(b):5.1f}%  ({int(sum(r['kept'] for r in b))}/{len(b)})")
kp = [r for r in rows if r["keptprev"]]
if kp: print(f"re-keep rate (was already a keeper): {100*sum(r['kept'] for r in kp)/len(kp):.0f}% "
             f"({int(sum(r['kept'] for r in kp))}/{len(kp)})")
neg = [r for r in rows if r["kept"] and r["savings"] <= 0]
print(f"keeps with NEGATIVE savings (loyalty/conviction keeps): {len(neg)} of {int(sum(r['kept'] for r in rows))}")
for r in neg: print(f"   {r['year']} {r['owner'][:18]:18s} {r['pos']} {r['name'][:20]:20s} cost ${r['cost']} exp ${r['exp']:.0f}")

print("\nper-owner style (drafted-pool keeps; slots = 3/yr max):")
print(f"  {'owner':20s} {'keeps':>5} {'avg save':>9} {'avg cost':>9} {'avg ppg':>8}")
for o in sorted(set(r["owner"] for r in rows)):
    ok = [r for r in rows if r["owner"] == o and r["kept"]]
    if ok:
        print(f"  {o[:20]:20s} {len(ok):>5} {np.mean([r['savings'] for r in ok]):>8.1f}$ "
              f"{np.mean([r['cost'] for r in ok]):>8.1f}$ {np.mean([r['ppg'] for r in ok]):>8.1f}")
    else:
        print(f"  {o[:20]:20s} {0:>5}")

# ---------------- Part 2: likelihood model ----------------
FEATS = ["savings", "cost", "ppg", "keptprev", "gms", "age"]
def design(rs): return np.array([[r[f] for f in FEATS] for r in rs], float)
def fit(rs, lam=2.0):
    X, y = design(rs), np.array([r["kept"] for r in rs])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Xs.shape[1]); I = np.eye(len(w)); I[0, 0] = 0
    for _ in range(100):
        p = 1 / (1 + np.exp(-Xs @ w)); W = np.clip(p * (1 - p), 1e-6, None)
        g = Xs.T @ (y - p) - lam * (I @ w)
        H = Xs.T @ (Xs * W[:, None]) + lam * I
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-9: break
    return lambda rs2: 1 / (1 + np.exp(-(np.hstack([np.ones((len(rs2), 1)), (design(rs2) - mu) / sd]) @ w))), w

def auc(y, p):
    pos = [pp for pp, yy in zip(p, y) if yy]; negv = [pp for pp, yy in zip(p, y) if not yy]
    return np.mean([[0.0, 0.5, 1.0][int(np.sign(a - b)) + 1] for a in pos for b in negv])

print("\n" + "=" * 74)
print("PART 2 — likelihood model (logistic on savings, cost, prior PPG, kept-before)")
print("=" * 74)
for tr_y, te_y in ((2024, 2025), (2025, 2024)):
    tr = [r for r in hist if r["year"] == tr_y]; te = [r for r in hist if r["year"] == te_y]
    model, _ = fit(tr)
    p = model(te); y = [r["kept"] for r in te]
    hits = base_hits = tot = 0
    for o in set(r["owner"] for r in te):
        orow = [(r, pp) for r, pp in zip(te, p) if r["owner"] == o]
        actual = set(r["name"] for r, _ in orow if r["kept"]); tot += len(actual)
        top3 = set(r["name"] for r, _ in sorted(orow, key=lambda t: -t[1])[:3])
        rat = set(r["name"] for r, _ in sorted([t for t in orow if t[0]["savings"] > 0],
                  key=lambda t: -t[0]["savings"])[:3])
        hits += len(actual & top3); base_hits += len(actual & rat)
    print(f"  train {tr_y} -> test {te_y}: AUC {auc(y, p):.3f} | top-3 hit rate "
          f"{hits}/{tot} vs rational-rule {base_hits}/{tot}")
model, w = fit(hist)
print("\nfinal fit on both seasons — standardized coefficients:")
for nm, c in zip(["intercept"] + FEATS, w): print(f"  {nm:10s} {c:+6.2f}")

# SLAM-DUNK RULE: a young, healthy, elite-production player at big savings is a lock.
# Audited on history — every pass that LOOKS like a counterexample was an injury-stale
# PPG (Godwin 7 gms, Rice 4, M.Williams 3), an age-30+ vet the room wouldn't re-price
# at his PPG rank (Keenan 32, Kamara 30, Evans 31), or slot competition (Nico Collins
# '24 was the 4th-best keep on a stacked team). The rule requires all four guards plus
# a top-3 slot on his own team, and only ever RAISES a probability (floor 0.97).
DUNK = lambda r: r["savings"] >= 15 and r["ppg"] >= 17 and r["gms"] >= 10 and r["age"] <= 28
d = [r for r in hist if DUNK(r)]
print(f"\nslam-dunk audit (savings>=15, ppg>=17, gms>=10, age<=28): "
      f"kept {int(sum(r['kept'] for r in d))}/{len(d)}")
for r in d:
    print(f"   {'KEPT' if r['kept'] else 'PASS'} {r['year']} {r['owner'][:16]:16s} {r['pos']:3s} "
          f"{r['name'][:22]:22s} save +{r['savings']:>3.0f} ppg {r['ppg']:4.1f} age {r['age']:.0f}")

# ---------------- Part 3: score every player on every 2026 roster ----------------
K = json.loads(open(os.path.join(ROOT, "docs", "keepers_2026.js"), encoding="utf-8")
               .read().split("const KEEPERS_2026 = ")[1].rsplit(";", 1)[0])
print("\n" + "=" * 74)
print("PART 3 — 2026 keep likelihood, every rostered skill player (cost/Exp $ = the")
print("tool's own numbers from keepers_2026.js; * = predicted keep of the rational rule)")
print("=" * 74)
teams_scored = []
for t in K["teams"]:
    cand = []
    for c in t["candidates"]:
        nm = norm(c["name"])
        r = {"savings": c["exp"] - c["cost"], "cost": c["cost"],
             "ppg": ppg.get((nm, 2025), 0.0), "gms": games.get((nm, 2025), 0),
             "age": age_by_name.get((nm, 2025), 26.0),
             "keptprev": 1.0 if c.get("keptyrs") else 0.0}
        cand.append((c, r))
    if cand: teams_scored.append((t, cand, model([r for _, r in cand])))
# BASE-RATE CALIBRATION: the fit only saw drafted-pool decisions, but ~20% of real keeps
# come off waivers/trades (invisible historically, visible on the 2026 rosters we score).
# Scale so the league-wide expected keep count matches the observed 2.65 keeps/team.
all_keeps = sum(1 for r in drafts if r["season"] in ("2024", "2025") and r["kept"])
team_seasons = len(set((r["season"], r["owner"]) for r in drafts if r["season"] in ("2024", "2025")))
target = all_keeps / team_seasons                   # 61/23 = 2.65 keeps/team, waiver keeps included
raw_total = sum(p.sum() for _, _, p in teams_scored)
CAL = (target * len(teams_scored)) / raw_total if raw_total > 0 else 1.0
print(f"(base-rate calibration x{CAL:.2f}: raw expected {raw_total:.1f} keeps -> "
      f"{target*len(teams_scored):.1f} = the league's real {target:.2f} keeps/team incl. waiver keeps)")
league_keeps = []
for t, cand, p in teams_scored:
    padj = np.minimum(p * CAL, 0.99)
    # a team may keep at most 3: rescale within team so probabilities sum to <= 3
    s = padj.sum()
    if s > 3.0: padj = padj * (3.0 / s)
    order = sorted(zip(cand, padj), key=lambda t: -t[1])
    # slam-dunk floor (audited above): only for players holding one of the team's top-3 slots
    order = [((c, r), max(pp, 0.97) if i < 3 and DUNK(r) else pp)
             for i, ((c, r), pp) in enumerate(order)]
    exp_keeps = sum(pp for _, pp in order)
    league_keeps.append(exp_keeps)
    print(f"\n[{t['id']:>2}] {str(t['name'])[:30]:30s} (owner {t['owner'][:16]}, expected keeps {exp_keeps:.1f})")
    for (c, r), pp in order:
        if pp < 0.02 and not c.get("predicted"): continue
        star = "*" if c.get("predicted") else " "
        lock = "LOCK" if pp >= 0.97 and DUNK(r) else "    "
        tag = " wvr" if c.get("waiver") else ""
        print(f"   {100*pp:5.1f}% {star} {c['pos']:<3} {c['name'][:22]:22s} "
              f"cost ${c['cost']:>2}  exp ${c['exp']:>2}  save {c['exp']-c['cost']:+3d}  "
              f"ppg {r['ppg']:4.1f} {lock}{tag}")
print(f"\nleague-wide expected keeper count: {sum(league_keeps):.1f} "
      f"(actual history: 31 in 2024, 30 in 2025)")

open(os.path.join(ROOT, "outputs", "reports", "keeper_likelihood_2026.md"), "w", encoding="utf-8").write(
    "# Keeper likelihood 2026 — who will each owner actually keep?\n\n"
    "Generated by `scripts/keeper_likelihood.py`. Historical keep/pass decisions are\n"
    "reconstructed from `espn_drafts.csv` (eligible = own prior-year draftees); the\n"
    "logistic model is validated season-out and applied to the 2026 rosters using the\n"
    "draft tool's own keeper costs and anchored Exp $.\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nwrote outputs/reports/keeper_likelihood_2026.md")
