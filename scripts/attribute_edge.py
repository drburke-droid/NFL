"""
Attribute our league edge: is it the PROJECTIONS, the VORP valuation, or the
leap/fade TILT? And should we maximize STARTER VORP, WHOLE-ROSTER VORP, or a hybrid?

Holds the bots and all in-season logic fixed; only swaps OUR team's draft valuation.
Each arm is a small Monte Carlo (randomized bids + schedule). Precompute (model
training) is shared across arms.

Part A — value source (bench weight fixed at the current 0.35):
  field        our team drafts like a bot: prior-year/ADP + VORP   (control)
  naive        our projections, $ ~ projected points, NO replacement/VORP
  vorp         our projections + VORP, no tilt
  vorp+tilt    our projections + VORP + value-leap/fade tilt        (current)
Part B — bench weight (valuation = vorp+tilt):
  starters(0.0)   hybrid(0.35, current)   whole-roster(1.0)

Usage: python scripts/attribute_edge.py [N]   (default N=60)
"""
import os, sqlite3, sys
import numpy as np
import lightgbm as lgb
import importlib.util
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(os.path.dirname(__file__), n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V2 = _load("league_sim_v2"); L1 = V2.L1; MS = V2.MS
ROSTER = V2.ROSTER; OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "EDGE_ATTRIBUTION.md")


def precompute(D):
    ds, FEATS, sk, adp, draft = D["ds"], D["FEATS"], D["sk"], D["adp"], D["draft"]
    yd = {}
    for Y in V2.YEARS:
        skY = sk[sk.season == Y]; pos_of = dict(zip(skY.player_id, skY.position))
        rookies = set(draft[draft.season == Y].player_id.dropna())
        prevY = dict(zip(sk[sk.season == Y - 1].player_id, sk[sk.season == Y - 1].pts))
        base_pts = {p: prevY.get(p, 0.0) for p in pos_of}
        for r in draft[draft.season == Y].dropna(subset=["pick"]).itertuples():
            if r.player_id in pos_of: base_pts[r.player_id] = max(base_pts.get(r.player_id, 0), max(0, 230 - r.pick) * 0.6)
        bot_val = L1.vorp_dollars({p: (base_pts.get(p, 0.0), pos_of[p]) for p in pos_of})
        if Y >= 2021:
            ecr = dict(zip(adp[adp.season == Y].player_id, adp[adp.season == Y].ecr))
            ranked = sorted(pos_of, key=lambda p: ecr.get(p, 9999)); mags = sorted([bot_val[p] for p in pos_of], reverse=True)
            for p, m in zip(ranked, mags): bot_val[p] = m
        tr = ds[ds.season < Y]; te = ds[ds.season == Y].copy(); our_ppg, leap, fade = {}, {}, {}
        if len(tr) > 200 and len(te) > 0:
            pm = lgb.LGBMRegressor(objective="regression_l1", **V2.GB).fit(tr[MS.FEATURES].astype(float).fillna(-1), tr.next_ppg)
            te["our_ppg"] = pm.predict(te[MS.FEATURES].astype(float).fillna(-1)).clip(min=0)
            for tgt, mask in [("leap_y", tr.prior_ppg < tr.position.map(V2.LOW)), ("fade_y", tr.prior_ppg >= tr.position.map(V2.HIGH))]:
                trp = tr[mask]
                te[tgt + "_p"] = (lgb.LGBMClassifier(objective="binary", **V2.GB).fit(trp[FEATS].astype(float).fillna(-1), trp[tgt]).predict_proba(te[FEATS].astype(float).fillna(-1))[:, 1]) if trp[tgt].sum() >= 12 else 0.0
            our_ppg = dict(zip(te.player_id, te.our_ppg)); leap = dict(zip(te.player_id, te.leap_y_p)); fade = dict(zip(te.player_id, te.fade_y_p))
        yd[Y] = dict(pos_of=pos_of, rookies=rookies, bot_val=bot_val, our_ppg=our_ppg, leap=leap, fade=fade, base_pts=base_pts)
    return yd


def our_val_for(arm, y):
    pos_of = y["pos_of"]
    if arm == "field": return dict(y["bot_val"])
    pts = {p: (y["our_ppg"][p] * 16 if p in y["our_ppg"] else y["base_pts"].get(p, 0.0)) for p in pos_of}
    if arm == "naive":
        tot = sum(max(0, v) for v in pts.values()) or 1; money = 12 * 200 - 12 * ROSTER
        return {p: 1 + max(0, pts[p]) / tot * money for p in pos_of}
    ov = L1.vorp_dollars({p: (pts[p], pos_of[p]) for p in pos_of})
    if arm == "vorp": return ov
    return {p: ov[p] * (1 + 0.25 * y["leap"].get(p, 0.0)) * (1 - 0.30 * y["fade"].get(p, 0.0)) for p in pos_of}  # vorp+tilt


def run_arm(D, wproj, yd, arm, bench_w, N):
    run_avg = []
    for run in range(N):
        rng = np.random.default_rng(3000 + run)
        teams = [V2.Team(i, V2.STYLES[i]) for i in range(12)]
        us = next(i for i in range(12) if teams[i].style == "ours"); teams[us].bench_w = bench_w
        fs = []
        for Y in V2.YEARS:
            y = yd[Y]; pv = (y["pos_of"], y["rookies"], y["bot_val"], our_val_for(arm, y))
            teams[us].bench_w = bench_w
            rank, _ = V2.simulate_year(Y, teams, D, pv, wproj, rng=rng, bid_sigma=0.15)
            fs.append(rank[us])
        run_avg.append(np.mean(fs))
    return np.array(run_avg)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    con = sqlite3.connect(V2.DB); D = V2.load(con); con.close()
    wproj = V2.make_wproj(D)
    print("Precomputing per-year models/values once...")
    yd = precompute(D)
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))

    print(f"Running arms (N={N} each)...")
    A = {}
    for arm in ["field", "naive", "vorp", "vorp+tilt"]:
        A[arm] = run_arm(D, wproj, yd, arm if arm != "vorp+tilt" else "tilt", 0.35, N); print(f"  A:{arm} done")
    B = {}
    for bw in [0.0, 0.35, 1.0]:
        B[bw] = run_arm(D, wproj, yd, "tilt", bw, N); print(f"  B:bench_w={bw} done")

    def line(name, a):
        lo, hi = ci(a); return f"- {name:34s} {a.mean():.2f}  [{lo:.2f}, {hi:.2f}]"
    L = [f"# Edge attribution — our team's avg finish by valuation arm (N={N} Monte Carlo each)\n",
         "Bots + all in-season logic held fixed; only OUR draft valuation changes. Lower = better.\n",
         "## Part A — where does the edge come from? (bench weight = 0.35)",
         line("field (prior-yr/ADP + VORP) [control]", A["field"]),
         line("naive (our proj, NO VORP)", A["naive"]),
         line("our proj + VORP", A["vorp"]),
         line("our proj + VORP + leap/fade tilt", A["vorp+tilt"]),
         "\n**Decomposition (avg finishing places gained vs the field control):**",
         f"- Projections (field → our-proj, both VORP): **{A['field'].mean()-A['vorp'].mean():+.2f}**",
         f"- VORP vs naive (both our-proj):              **{A['naive'].mean()-A['vorp'].mean():+.2f}**",
         f"- Leap/fade tilt (on top of proj+VORP):       **{A['vorp'].mean()-A['vorp+tilt'].mean():+.2f}**",
         f"- TOTAL (field → full):                       **{A['field'].mean()-A['vorp+tilt'].mean():+.2f}**",
         "\n## Part B — maximize starter VORP, whole-roster, or hybrid? (valuation = proj+VORP+tilt)",
         line("starters only (bench_w=0.0)", B[0.0]),
         line("hybrid (bench_w=0.35) [current]", B[0.35]),
         line("whole roster (bench_w=1.0)", B[1.0])]
    open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("\n" + "\n".join(L)); print("Saved", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
