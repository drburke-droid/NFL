"""
Walk-forward backtest of the BENCH DARTS composite, 2016-2025.

For each season T, build the list exactly as we would have at draft time — using
ONLY information available before week 1 of T:

  pool     RB/WR/TE veterans (played season T-1), market-cheap at T:
           FFA AAV_T <= $8, or absent from the FFA file (undrafted tier)
  signals  HEIR      (T-1: backup RB out-ran his own lead on ypc AND EPA/carry, 40+ car)
           VACATED   (nflv_opportunity season T — offseason roster state)
           ALPHA-HI  (T-1 teammate-adjusted alpha skill >= 80th pctile of position)
           H2 TREND  (T-1 second-half surge: +1.5 ppg and +5% snaps)
           FLASH     (T-1 best-4-week PPR avg >= 12 — the ceiling proxy)
  weights  mirror bench_darts_2026.py where reproducible; the 2026 model layers
           (dart_prob / leap_prob / lottery) are separately walk-forward validated
           (LATE_BREAKOUTS.md, lottery screen 2016-25) and NOT re-tested here.
  skew     x keeper-runway age curve (<=23 1.25 ... 30+ 0.6), x RB 1.15

Outcomes in season T:
  RELIABLE = >= 6 weeks scored as a weekly starter (RB/WR top-24 of week, TE top-12)
  STAR     = season top-12 positional PPG (min 10 games, PPR)

Baseline = the whole cheap pool's hit rates. Report: outputs/reports/bench_darts_backtest.md
"""
import os, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
pd.set_option("display.width", 240)
TOPN = 10
START_RANK = {"RB": 24, "WR": 24, "TE": 12}

con = sqlite3.connect(DB)
S = pd.read_sql("""SELECT player_id, player_display_name name, position, season, recent_team team,
                          games, carries, rushing_yards, rushing_epa, fantasy_points_ppr
                   FROM nflv_season WHERE position IN ('RB','WR','TE')""", con)
W = pd.read_sql("""SELECT player_id, position, season, week, fantasy_points_ppr pts
                   FROM nflv_weekly WHERE season_type='REG' AND position IN ('RB','WR','TE')""", con)
A = pd.read_sql("SELECT player_id, position, season, alpha_skill FROM player_skill_alpha", con)
OPP = pd.read_sql("SELECT player_id, season, vacated_role FROM nflv_opportunity", con)
HT = pd.read_sql("SELECT player_id, season, ht_d_ppg, ht_d_snap FROM nflv_half_trend", con)
FFA = pd.read_sql("""SELECT season, player_id, ffa_aav FROM nflv_ffa_league
                     WHERE player_id IS NOT NULL""", con).drop_duplicates(["season", "player_id"])
AGE = pd.read_sql("SELECT player_id, season, age FROM season_dataset", con).drop_duplicates(["player_id", "season"])
con.close()

A["apct"] = A.groupby(["position", "season"])["alpha_skill"].rank(pct=True)
W["wk_rank"] = W.groupby(["season", "week", "position"])["pts"].rank(ascending=False)
W["starter_wk"] = W.apply(lambda r: r.wk_rank <= START_RANK[r.position], axis=1)

# outcomes per player-season
out = W.groupby(["player_id", "position", "season"]).agg(
    starter_wks=("starter_wk", "sum"), g=("week", "count")).reset_index()
ppg = S.assign(ppg=S.fantasy_points_ppr / S.games.clip(lower=1))
ppg["pos_rank"] = ppg[ppg.games >= 10].groupby(["season", "position"])["ppg"].rank(ascending=False)
out = out.merge(ppg[["player_id", "season", "pos_rank"]], on=["player_id", "season"], how="left")
out["reliable"] = out.starter_wks >= 6
out["star"] = out.pos_rank <= 12

# HEIR per season T (from T-1): backup with 40+ carries out-running his team's lead back
def heirs(t1):
    rb = S[(S.position == "RB") & (S.season == t1) & (S.carries >= 40)].copy()
    rb["ypc"] = rb.rushing_yards / rb.carries
    rb["epc"] = rb.rushing_epa / rb.carries
    ids = set()
    for tm, g in rb.groupby("team"):
        if len(g) < 2: continue
        g = g.sort_values("carries", ascending=False)
        lead = g.iloc[0]
        for _, b in g.iloc[1:].iterrows():
            if b.ypc > lead.ypc and b.epc > lead.epc: ids.add(b.player_id)
    return ids

runway = lambda a: 1.25 if a <= 23 else 1.15 if a <= 25 else 1.0 if a <= 27 else 0.8 if a <= 29 else 0.6
POS_X = {"RB": 1.15, "WR": 1.0, "TE": 0.85}

rows, picks_log = [], []
for T in range(2016, 2026):
    prev = S[(S.season == T - 1) & (S.games >= 4)][["player_id", "name", "position", "games"]]
    aav = FFA[FFA.season == T].set_index("player_id")["ffa_aav"]
    pool = prev.copy()
    pool["aav"] = pool.player_id.map(aav)
    pool = pool[(pool.aav.fillna(0) <= 8)]
    # flash: T-1 best-4-week avg
    w1 = W[W.season == T - 1].sort_values("pts", ascending=False).groupby("player_id").head(4) \
          .groupby("player_id")["pts"].mean()
    pool["flash"] = (pool.player_id.map(w1).fillna(0) >= 12).astype(int)
    hs = heirs(T - 1)
    pool["heir"] = pool.player_id.isin(hs).astype(int)
    pool["vac"] = pool.player_id.map(OPP[OPP.season == T].set_index("player_id")["vacated_role"]).fillna(0)
    ap = A[A.season == T - 1].set_index("player_id")["apct"]
    pool["alpha_hi"] = (pool.player_id.map(ap).fillna(0) >= 0.80).astype(int)
    ht = HT[HT.season == T - 1].set_index("player_id")
    pool["h2"] = ((pool.player_id.map(ht["ht_d_ppg"]).fillna(0) > 1.5) &
                  (pool.player_id.map(ht["ht_d_snap"]).fillna(0) > 5)).astype(int)
    pool["age"] = pool.player_id.map(AGE[AGE.season == T - 1].set_index("player_id")["age"]).fillna(26) + 1
    # weights REFIT from the per-signal breakdown (signal_breakdown run, same pools):
    # ALPHA 3.9x reliable / 6.2x star dominates; VAC 2.4x/2.4x; HEIR 2.6x reliable but
    # 0.7x star; FLASH 2.3x/2.4x; H2 was dead weight (1.2x/0.6x) -> dropped.
    pool["score"] = ((1.00 * pool.alpha_hi + 0.45 * pool.vac + 0.40 * pool.heir
                      + 0.35 * pool.flash)
                     * pool.age.map(runway) * pool.position.map(POS_X))
    pool = pool[pool.score > 0].sort_values("score", ascending=False)

    oT = out[out.season == T].set_index("player_id")
    pool["reliable"] = pool.player_id.map(oT["reliable"]).fillna(False)
    pool["star"] = pool.player_id.map(oT["star"]).fillna(False)
    top = pool.head(TOPN)
    rows.append(dict(season=T, pool=len(pool),
                     top_rel=int(top.reliable.sum()), top_star=int(top.star.sum()),
                     pool_rel_rate=pool.reliable.mean(), pool_star_rate=pool.star.mean()))
    for _, r in top.iterrows():
        why = "+".join(k for k, v in [("HEIR", r.heir), ("VAC", r.vac), ("ALPHA", r.alpha_hi),
                                      ("H2", r.h2), ("FLASH", r.flash)] if v)
        picks_log.append(f"  {T} {r.position} {r['name']:<22} [{why}] -> "
                         f"{'⭐STAR' if r.star else ('✔ reliable' if r.reliable else '✗ miss')}")

R = pd.DataFrame(rows)
L = ["# Bench-darts composite — walk-forward backtest 2016-2025\n",
     f"Top-{TOPN} per season, ex-ante signals only (see script docstring). RELIABLE = 6+",
     "weekly-starter weeks (RB/WR top-24 of week, TE top-12). STAR = season top-12 PPG.\n", "```"]
L.append(R.assign(pool_rel_rate=(100 * R.pool_rel_rate).round(0), pool_star_rate=(100 * R.pool_star_rate).round(1)).to_string(index=False))
L.append("```\n")
tr, ts = R.top_rel.sum(), R.top_star.sum()
n = TOPN * len(R)
pr, ps = (R.pool_rel_rate * 1).mean(), (R.pool_star_rate * 1).mean()
L.append(f"**Top-{TOPN} picks: {tr}/{n} reliable ({100 * tr / n:.0f}%), {ts}/{n} star ({100 * ts / n:.0f}%).**")
L.append(f"Cheap-pool baseline: {100 * pr:.0f}% reliable, {100 * ps:.1f}% star.")
L.append(f"**Lift: {tr / n / pr:.1f}x reliable, {ts / n / ps:.1f}x star.**\n")
L.append("## Every pick\n```")
L += picks_log
L.append("```")
open(os.path.join(ROOT, "outputs", "reports", "bench_darts_backtest.md"), "w", encoding="utf-8").write("\n".join(L))
print(R.to_string(index=False))
print(f"\nTop-{TOPN}: {tr}/{n} reliable ({100*tr/n:.0f}%), {ts}/{n} star ({100*ts/n:.0f}%)")
print(f"pool baseline: {100*pr:.0f}% reliable, {100*ps:.1f}% star -> lift {tr/n/pr:.1f}x / {ts/n/ps:.1f}x")
print("\nWrote outputs/reports/bench_darts_backtest.md")
