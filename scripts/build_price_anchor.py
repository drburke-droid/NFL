"""
Per-player auction price anchor for the draft tool's Exp $.

Fits what THIS league pays returning (non-rookie) players from its own 2023-25 winning
bids: last year's price (the league anchors ~half of it), prior-season PPG and games,
career-best PPG, age (with a 28+ decline term), and position. 2024 was an 11-team
league, so every season's bids are first rescaled to the most-recent season's
per-team auction spend (finance calibration).

Validation is season-out (train one keeper-era season, test the other) against the
positional bid-curve baseline the tool already uses; the blend weight that wins
out-of-sample is emitted alongside the anchors.

Writes docs/price_anchor_2026.js:  const PRICE_ANCHOR = {pid: $}; const PRICE_ANCHOR_W = w;
(and mirrors to outputs/draft_tool/). Rookies are excluded - the rank curve prices them.
"""
import os, json, csv, re, sqlite3
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = ("QB", "RB", "WR", "TE")

def norm(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

# ---------------- data ----------------
drafts = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "espn_drafts.csv"), encoding="utf-8")))
for r in drafts:
    r["bid"] = max(int(float(r["bid"] or 0)), 1)
    r["kept"] = str(r["keeper"]).lower() in ("true", "1")

# finance calibration: dollars-per-team actually at auction, rescaled to the latest season
per_team = {}
for s in set(r["season"] for r in drafts):
    rows = [r for r in drafts if r["season"] == s and not r["kept"]]
    per_team[s] = sum(r["bid"] for r in rows) / len(set(r["owner"] for r in rows))
REF = per_team[max(per_team)]
SCALE = {s: REF / v for s, v in per_team.items()}
print("per-team auction spend:", {s: round(v) for s, v in sorted(per_team.items())}, "-> scale", {s: round(v, 3) for s, v in sorted(SCALE.items())})

auction = {}   # (nm, season) -> scaled bid
keptcost = {}  # (nm, season) -> scaled keeper cost
pos_of = {}
for r in drafts:
    key = (norm(r["player"]), r["season"])
    v = r["bid"] * SCALE[r["season"]]
    (keptcost if r["kept"] else auction)[key] = v
    pos_of[key] = r["pos"]

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
ppg, games = {}, {}
for nm, season, g, ppr in con.execute(
        "SELECT player_display_name, season, games, fantasy_points_ppr FROM nflv_season WHERE season>=2020"):
    if g and ppr is not None:
        ppg[(norm(nm), int(season))] = ppr / g
        games[(norm(nm), int(season))] = g
age_of = {}
for pid_, season, a in con.execute(
        "SELECT s.player_id, s.season, s.age FROM season_dataset s WHERE s.age IS NOT NULL"):
    age_of[(pid_, int(season))] = a
id2nm = dict(con.execute("SELECT DISTINCT player_id, player_display_name FROM nflv_season"))
age_by_name = {}
for (pid_, season), a in age_of.items():
    if pid_ in id2nm:
        age_by_name[(norm(id2nm[pid_]), season)] = a

def career_best(nm, y):
    return max([ppg.get((nm, s), 0) for s in range(y - 3, y) if games.get((nm, s), 0) >= 4] or [0])

def feats(nm, pos, y, age):
    """Feature vector for a player entering draft year y (uses only pre-draft info)."""
    lb = auction.get((nm, str(y - 1)), keptcost.get((nm, str(y - 1))))
    p1 = ppg.get((nm, y - 1)) if games.get((nm, y - 1), 0) >= 4 else None
    if lb is None and p1 is None:
        return None                                # no league or NFL track record: not anchorable
    return [
        lb or 0.0, 1.0 if lb is not None else 0.0,
        1.0 if (nm, str(y - 1)) in keptcost else 0.0,
        p1 or 0.0, 1.0 if p1 is not None else 0.0,
        games.get((nm, y - 1), 0),
        career_best(nm, y),
        age or 26.0, max(0.0, (age or 26.0) - 28.0),
        1.0 if pos == "QB" else 0.0, 1.0 if pos == "TE" else 0.0, 1.0 if pos == "RB" else 0.0,
    ]
FNAMES = ["lastBid", "hasLastBid", "keptPrev", "priorPPG", "hasPPG", "priorGames",
          "career3Best", "age", "age28plus", "QB", "TE", "RB"]

def ridge(X, y, lam=3.0):
    X = np.asarray(X); y = np.asarray(y)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1]); A[0, 0] -= lam
    w = np.linalg.solve(A, Xs.T @ y)
    return lambda Z: np.hstack([np.ones((len(Z), 1)), (np.asarray(Z) - mu) / sd]) @ w, w, mu, sd

def curve_from(seasons, pos):
    """positional $/rank curve from given seasons (scaled bids), like predict_keepers."""
    BW = {"2023": 1, "2024": 2, "2025": 3}
    lists = {}
    for (nm, s), v in auction.items():
        if s in seasons and pos_of[(nm, s)] == pos:
            lists.setdefault(s, []).append(v)
    ml = max((len(v) for v in lists.values()), default=0)
    out = []
    for rk in range(ml):
        num = den = 0.0
        for s, arr in lists.items():
            arr.sort(reverse=True)
            if rk < len(arr): num += arr[rk] * BW.get(s, 1); den += BW.get(s, 1)
        out.append(max(1, num / den if den else 1))
    return out

# ---------------- training rows (2024 & 2025 targets, non-rookies) ----------------
rows = []   # (season, features, target, nm, pos)
for (nm, s), bid in auction.items():
    y = int(s)
    if y < 2024 or pos_of[(nm, s)] not in SKILL:
        continue
    if games.get((nm, y - 1), 0) < 8:
        continue    # anchor domain = players whose prior season was REPRESENTATIVE; injury
                    # returns / breakouts are priced forward by the room -> curve handles them
    f = feats(nm, pos_of[(nm, s)], y, age_by_name.get((nm, y)))
    if f is not None:
        rows.append((s, f, bid, nm, pos_of[(nm, s)]))
print(f"training rows (returning players, keeper era): {len(rows)} "
      f"(2024: {sum(1 for r in rows if r[0]=='2024')}, 2025: {sum(1 for r in rows if r[0]=='2025')})")

# ---------------- season-out validation vs curve baseline ----------------
def evaluate(train_s, test_s):
    tr = [r for r in rows if r[0] in train_s]; te = [r for r in rows if r[0] == test_s]
    model, w, mu, sd = ridge([r[1] for r in tr], [r[2] for r in tr])
    pred_m = np.clip(model([r[1] for r in te]), 1, 80)
    # baseline: leave-test-out curve at prior-PPG rank within position
    base = np.zeros(len(te))
    for pos in SKILL:
        idx = [i for i, r in enumerate(te) if r[4] == pos]
        C = curve_from([s for s in ("2023", "2024", "2025") if s != test_s], pos)
        order = sorted(idx, key=lambda i: -(te[i][1][3]))          # rank by prior PPG
        for rank, i in enumerate(order):
            base[i] = C[rank] if rank < len(C) else 1
    yv = np.array([r[2] for r in te])
    out = {}
    for W in (0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        p = W * pred_m + (1 - W) * base
        out[W] = float(np.abs(p - yv).mean())
    return out, yv, pred_m, base

res24, *_ = evaluate(["2025"], "2024")
res25, *_ = evaluate(["2024"], "2025")
print("\nseason-out MAE by blend weight (0=curve only, 1=model only):")
print("  W     :", "  ".join(f"{W:>5}" for W in res24))
print("  ->2024:", "  ".join(f"{res24[W]:5.2f}" for W in res24))
print("  ->2025:", "  ".join(f"{res25[W]:5.2f}" for W in res25))
avg = {W: (res24[W] + res25[W]) / 2 for W in res24}
best = min(avg.values())
# most parsimonious anchor weight within 1% of the best MAE - the tool's LIVE curve is
# demand-aware and projection-ranked (stronger than this offline baseline), so lean low
BEST_W = min(W for W, m in avg.items() if m <= best * 1.01)
print(f"  avg   : best W = {BEST_W} (MAE {avg[BEST_W]:.2f} vs curve-only {avg[0]:.2f}, min {best:.2f})")

# ---------------- final fit on all keeper-era rows + 2026 emission ----------------
model, w, mu, sd = ridge([r[1] for r in rows], [r[2] for r in rows])
print("\ncoefficients (standardized):")
for name, c in zip(["intercept"] + FNAMES, w):
    print(f"  {name:12s} {c:+7.2f}")

P = json.loads(open(os.path.join(ROOT, "docs", "data.js"), encoding="utf-8").read()
               .split("const PLAYERS = ")[1].rsplit(";", 1)[0])
anchors, skipped = {}, 0
for p in P:
    if p["position"] not in SKILL or p.get("is_rookie") == 1:
        continue
    nm = norm(p["name"])
    # anchor domain only: 2025 must be representative (>=8 games) and the 2026 projection must
    # not scream "he's back / breaking out" (room prices those FORWARD; the curve handles them)
    if games.get((nm, 2025), 0) < 8:
        skipped += 1; continue
    if (p.get("proj_ppg") or 0) - (p.get("prior_ppg") or 0) > 3:
        skipped += 1; continue
    f = feats(nm, p["position"], 2026, p.get("age") or age_by_name.get((nm, 2025)))
    if f is None:
        skipped += 1; continue
    val = float(np.clip(model([f])[0], 1, 80))
    anchors[f'{p["name"]}|{p["position"]}|{p.get("team") or ""}'] = round(val, 1)
print(f"\n2026 anchors: {len(anchors)} players ({skipped} outside the anchor domain -> curve prices them)")
top = sorted(anchors.items(), key=lambda kv: -kv[1])[:12]
for k, v in top: print(f"  ${v:5.1f}  {k}")

js = ("// AUTO-GENERATED by scripts/build_price_anchor.py - per-player expected-price anchors\n"
      "// fitted on this league's own 2023-25 winning bids (finance-calibrated for the 11-team\n"
      "// 2024 season). Blend weight validated season-out vs the positional bid curve.\n"
      f"const PRICE_ANCHOR = {json.dumps(anchors)};\n"
      f"const PRICE_ANCHOR_W = {BEST_W};\n")
for d in (os.path.join(ROOT, "docs"), os.path.join(ROOT, "outputs", "draft_tool")):
    open(os.path.join(d, "price_anchor_2026.js"), "w", encoding="utf-8").write(js)
print("wrote docs/price_anchor_2026.js (+ mirror)")
