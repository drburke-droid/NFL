"""
Pure DRAFT-STRATEGY tournament (2012-2025), no keepers, no waivers.

12 owners draft head-to-head in a $200 auction. EVERY owner values players from the
SAME projections (a leave-one-season-out season model — realistic error, identical for
all), so the ONLY thing that differs is each owner's positional/archetype STRATEGY.
Rosters are 14 skill players (K/DST streamed = ignored). Scored BEST-BALL: each week
the optimal lineup (1QB/2RB/2WR/1TE/1FLEX) by ACTUAL points — so start/sit and waivers
are a wash and we measure pure roster construction. Repeated N times/year with draft
bid noise; teams ranked by season points. Aggregated 2012-2025 x N.

Strategies include the two combos asked about: elite-QB + pass-catching RB, and
cheap-QB + two workhorse RBs.

Usage: python scripts/draft_tournament.py [N]   (default N=120)
"""
import os, sqlite3, sys
import numpy as np
import lightgbm as lgb
import importlib.util
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(os.path.dirname(__file__), n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V2 = _load("league_sim_v2"); L1 = V2.L1; MS = V2.MS
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "DRAFT_TOURNAMENT.md")
YEARS = list(range(2012, 2026))
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}; FLEX = {"RB", "WR", "TE"}; ROSTER = 14; BENCH = 7
REPL = {"QB": 12, "RB": 30, "WR": 36, "TE": 14}
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

STRATS = ["neutral", "elite_qb", "cheap_qb", "zero_rb", "robust_rb(2 workhorse)", "hero_rb",
          "elite_te", "wr_heavy", "stars_scrubs", "eliteQB+pcRB", "cheapQB+2workRB", "balanced_depth"]


def smult(s, pos, arch, orank, isr):
    if s == "neutral": return 1.0
    if s == "elite_qb": return 1.5 if pos == "QB" else 0.90
    if s == "cheap_qb": return 0.30 if pos == "QB" else 1.08
    if s == "zero_rb": return (0.40 if orank <= 24 else 0.70) if pos == "RB" else (1.30 if pos == "WR" else (1.12 if pos == "TE" else 1.0))
    if s == "robust_rb(2 workhorse)": return 1.35 if pos == "RB" else 0.88
    if s == "hero_rb": return (1.55 if orank <= 12 else 0.55) if pos == "RB" else (1.15 if pos == "WR" else 1.0)
    if s == "elite_te": return (1.6 if orank <= 45 else 0.85) if pos == "TE" else 0.96
    if s == "wr_heavy": return 1.30 if pos == "WR" else 0.90
    if s == "stars_scrubs": return 1.6 if orank <= 18 else (1.05 if orank <= 36 else 0.40)
    if s == "eliteQB+pcRB": return 1.45 if pos == "QB" else (1.35 if arch == "RB_rec" else (0.80 if pos == "RB" else 0.95))
    if s == "cheapQB+2workRB": return 0.30 if pos == "QB" else (1.45 if arch == "RB_work" else (0.90 if pos == "RB" else 1.02))
    if s == "balanced_depth": return 0.6 if orank <= 12 else (1.2 if orank <= 48 else (1.1 if orank <= 110 else 0.85))
    return 1.0


def vorp_dollars(pts_pos):
    import pandas as pd
    rows = pd.DataFrame([(p, v[0], v[1]) for p, v in pts_pos.items()], columns=["p", "pts", "pos"])
    repl = {}
    for pos, g in rows.groupby("pos"):
        s = g.pts.sort_values(ascending=False).values; r = REPL.get(pos, 12)
        repl[pos] = s[r] if r < len(s) else (s[-1] if len(s) else 0)
    rows["vorp"] = (rows.pts - rows.pos.map(repl)).clip(lower=0)
    tot = rows.vorp.sum(); money = 12 * 200 - 12 * ROSTER
    rows["val"] = 1 + rows.vorp / tot * money if tot > 0 else 1.0
    return dict(zip(rows.p, rows.val))


def precompute(D):
    import pandas as pd
    ds, sk, draft = D["ds"], D["sk"], D["draft"]
    con = sqlite3.connect(V2.DB)
    us = pd.read_sql("SELECT player_id, season, carries, receptions, targets, games FROM nflv_season WHERE position='RB'", con); con.close()
    U = {(r.season, r.player_id): (r.carries or 0, r.receptions or 0, r.targets or 0, max(r.games, 1)) for r in us.itertuples()}
    yd = {}
    for Y in YEARS:
        skY = sk[sk.season == Y]; pos_of = dict(zip(skY.player_id, skY.position))
        # common projection: leave-one-season-out model (same realistic proj for all owners)
        tr = ds[ds.season != Y]; te = ds[ds.season == Y]
        ppg = {}
        if len(te) > 0:
            m = lgb.LGBMRegressor(objective="regression_l1", **GB).fit(tr[MS.FEATURES].astype(float).fillna(-1), tr.next_ppg)
            ppg = dict(zip(te.player_id, np.clip(m.predict(te[MS.FEATURES].astype(float).fillna(-1)), 0, None)))
        prevpts = dict(zip(sk[sk.season == Y - 1].player_id, sk[sk.season == Y - 1].pts))
        # archetype from prior-year usage (RBs only)
        arch = {}
        for p in pos_of:
            if pos_of[p] == "RB" and (Y - 1, p) in U:
                c, rec, tg, g = U[(Y - 1, p)]; cpg = c / g; rpg = rec / g; tpg = tg / g; rs = rpg / max(cpg + rpg, 1e-6)
                arch[p] = "RB_rec" if (tpg >= 3 or rs >= 0.28) else ("RB_work" if cpg >= 11 else "RB_rot")
        pts = {p: (ppg[p] * 16 if p in ppg else prevpts.get(p, 0.0)) for p in pos_of}
        base = vorp_dollars({p: (pts[p], pos_of[p]) for p in pos_of})
        order = sorted(base, key=lambda p: -base[p]); orank = {p: i + 1 for i, p in enumerate(order)}
        info = {p: (pos_of[p], arch.get(p, pos_of[p]), orank[p], False) for p in pos_of}
        yd[Y] = dict(pos_of=pos_of, base=base, info=info)
    return yd


def need(t, pos):
    if pos in STARTERS and t["filled"][pos] < STARTERS[pos]: return "start"
    if pos in FLEX and t["flex"] < 1: return "flex"
    if t["bench"] < BENCH: return "bench"
    return None


def add(t, p, pos, price):
    t["roster"].append((p, pos)); t["budget"] -= price; t["prices"][p] = price
    sl = need(t, pos)
    if sl == "start": t["filled"][pos] += 1
    elif sl == "flex": t["flex"] += 1
    else: t["bench"] += 1


def draft(strats, y, rng):
    base, info, pos_of = y["base"], y["info"], y["pos_of"]
    teams = [{"strat": s, "budget": 200, "roster": [], "prices": {}, "filled": {p: 0 for p in STARTERS}, "flex": 0, "bench": 0} for s in strats]
    order = sorted(base, key=lambda p: -base[p])
    for p in order:
        pos, arch, orank, isr = info[p]; bids = []
        for t in teams:
            if len(t["roster"]) >= ROSTER: continue
            sl = need(t, pos)
            if sl is None: continue
            v = base[p] * smult(t["strat"], pos, arch, orank, isr) * (0.35 if sl == "bench" else 1.0)
            v *= float(np.exp(rng.normal(0, 0.15)))
            mb = max(1, t["budget"] - (ROSTER - len(t["roster"]) - 1))
            bid = min(v, mb)
            if bid >= 1: bids.append((bid, t))
        if not bids: continue
        bids.sort(key=lambda x: -x[0]); win = bids[0][1]; second = bids[1][0] if len(bids) > 1 else 1
        mb = max(1, win["budget"] - (ROSTER - len(win["roster"]) - 1))
        price = int(max(1, min(round(bids[0][0]), round(second) + 1, mb)))
        add(win, p, pos, price)
    # cleanup: fill rosters AND deploy leftover budget
    taken = {p for t in teams for p, _ in t["roster"]}
    left = [p for p in order if p not in taken]
    for t in teams:
        for p in list(left):
            if len(t["roster"]) >= ROSTER: break
            if need(t, pos_of[p]):
                op = ROSTER - len(t["roster"]); price = max(1, min(t["budget"] - (op - 1), int(round(t["budget"] / op))))
                add(t, p, pos_of[p], price); left.remove(p)
    return teams


def bestball(roster, WACT, Y):
    wks = set()
    for p, _ in roster: wks |= set(WACT.get((Y, p), {}).keys())
    total = 0.0
    for w in wks:
        byp = {}
        for p, pos in roster: byp.setdefault(pos, []).append(WACT.get((Y, p), {}).get(w, 0.0))
        for pos in byp: byp[pos].sort(reverse=True)
        used = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}; s = 0.0
        for pos, n in [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)]:
            s += sum(byp.get(pos, [])[:n]); used[pos] = min(n, len(byp.get(pos, [])))
        flexc = []
        for pos in FLEX: flexc += byp.get(pos, [])[used[pos]:]
        if flexc: s += max(flexc)
        total += s
    return total


def main():
    import pandas as pd
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    con = sqlite3.connect(V2.DB); D = V2.load(con); con.close()
    WACT = D["WACT"]
    print("Precomputing leave-one-out projections per year..."); yd = precompute(D)
    fin = {s: [] for s in STRATS}; wins = {s: 0 for s in STRATS}
    print(f"Running {N} drafts/year x {len(YEARS)} years...")
    for run in range(N):
        rng = np.random.default_rng(7000 + run)
        for Y in YEARS:
            teams = draft(STRATS, yd[Y], rng)
            scored = sorted(((bestball(t["roster"], WACT, Y), t["strat"]) for t in teams), key=lambda x: -x[0])
            for rank, (pts, s) in enumerate(scored, 1):
                fin[s].append(rank)
                if rank == 1: wins[s] += 1
        if (run + 1) % 30 == 0: print(f"  {run+1}/{N}")
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    board = sorted(STRATS, key=lambda s: np.mean(fin[s]))
    games = N * len(YEARS)
    L = [f"# Pure draft-strategy tournament (2012-2025, no keepers/waivers, best-ball) — {N} drafts/yr\n",
         "All 12 owners value players from the SAME leave-one-out projections; only the positional/archetype STRATEGY differs. 14 skill, K/DST streamed. Lower avg finish = better.\n",
         "| Rank | Strategy | Avg finish | Win % | 95% CI |", "|---|---|---|---|---|"]
    for i, s in enumerate(board, 1):
        a = np.array(fin[s]); lo, hi = ci([np.mean(a[k::len(YEARS)]) for k in range(len(YEARS))] if False else a)
        L.append(f"| {i} | {s} | **{a.mean():.2f}** | {100*wins[s]/games:.0f}% | [{lo:.0f}, {hi:.0f}] |")
    open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("\n" + "\n".join(L)); print("Saved", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
