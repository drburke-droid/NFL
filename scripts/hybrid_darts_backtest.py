"""
HYBRID DARTS BACKTEST — the tiered 10-pick strategy, walk-forward 2016-2025.

Selection each draft year T (ex-ante, T-1 signals only, same cheap pool as
bench_darts_backtest.py / dart_synergy_study.py — FFA AAV<=$8 or unlisted):

  tier 1  ALL HEIR+BUZZ (backup who out-ran his lead RB + top-quartile August
          wiki spike) — the one pair with positive synergy (1.48x, dart_synergy.md)
  tier 2  up to 3 young buried ALPHAs (skill >=80th pctl, age<=27) by skill pctl
  tier 3  fill to 10 with YOUNG FLASH (best-4-wk >=12, age<=26) by flash value —
          the age gate is what rescues recent years (vet cheap-FLASH stopped
          hitting as that market got efficient; Gronk/Edelman/Kupp-type picks)

Age gates are strict: a missing age FAILS the gate (no fillna leakage).
Outcomes: RELIABLE = 6+ weekly-starter weeks; STAR = top-12 positional PPG.

Output: outputs/reports/hybrid_darts_backtest.md
"""
import os, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
S = pd.read_sql("""SELECT player_id, player_display_name name, position, season, recent_team team,
                          games, carries, rushing_yards, rushing_epa, fantasy_points_ppr
                   FROM nflv_season WHERE position IN ('RB','WR','TE')""", con)
W = pd.read_sql("""SELECT player_id, position, season, week, fantasy_points_ppr pts
                   FROM nflv_weekly WHERE season_type='REG' AND position IN ('RB','WR','TE')""", con)
A = pd.read_sql("SELECT player_id, position, season, alpha_skill FROM player_skill_alpha", con)
FFA = pd.read_sql("""SELECT season, player_id, ffa_aav FROM nflv_ffa_league
                     WHERE player_id IS NOT NULL""", con).drop_duplicates(["season", "player_id"])
WB = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
AGE = pd.read_sql("SELECT player_id, season, age FROM season_dataset", con).drop_duplicates(["player_id", "season"])
con.close()

A["apct"] = A.groupby(["position", "season"])["alpha_skill"].rank(pct=True)
START_RANK = {"RB": 24, "WR": 24, "TE": 12}
W["wk_rank"] = W.groupby(["season", "week", "position"])["pts"].rank(ascending=False)
W["starter_wk"] = W.apply(lambda r: r.wk_rank <= START_RANK[r.position], axis=1)
out = W.groupby(["player_id", "position", "season"]).agg(starter_wks=("starter_wk", "sum")).reset_index()
ppg = S.assign(ppg=S.fantasy_points_ppr / S.games.clip(lower=1))
ppg["pos_rank"] = ppg[ppg.games >= 10].groupby(["season", "position"])["ppg"].rank(ascending=False)
out = out.merge(ppg[["player_id", "season", "pos_rank"]], on=["player_id", "season"], how="left")
out["reliable"] = out.starter_wks >= 6
out["star"] = out.pos_rank <= 12

WB["y"] = WB.ym.str[:4].astype(int); WB["m"] = WB.ym.str[-2:].astype(int)
aug = WB[WB.m == 8].rename(columns={"views": "augv"})[["player_id", "y", "augv"]]
bs = WB[WB.m.between(1, 5)].groupby(["player_id", "y"]).views.median().rename("basev").reset_index()
BZ = aug.merge(bs, on=["player_id", "y"], how="left")
BZ["spike"] = np.log((BZ.augv + 100) / (BZ.basev.fillna(0) + 100))


def heirs(t1):
    rb = S[(S.position == "RB") & (S.season == t1) & (S.carries >= 40)].copy()
    rb["ypc"] = rb.rushing_yards / rb.carries; rb["epc"] = rb.rushing_epa / rb.carries
    ids = set()
    for tm, g in rb.groupby("team"):
        if len(g) < 2: continue
        g = g.sort_values("carries", ascending=False); lead = g.iloc[0]
        for _, b in g.iloc[1:].iterrows():
            if b.ypc > lead.ypc and b.epc > lead.epc: ids.add(b.player_id)
    return ids


_out = []
def pr(*a): s = " ".join(str(x) for x in a); _out.append(s); print(s)


pr("year  hit star  picks  (*=star, +=reliable; tiers: H+B | ALPHA | young FLASH)")
tot_h = tot_s = 0; base_r = []; per_year = []
for T in range(2016, 2026):
    prev = S[(S.season == T - 1) & (S.games >= 4)][["player_id", "name", "position"]]
    aav = FFA[FFA.season == T].set_index("player_id")["ffa_aav"]
    pool = prev.copy(); pool["aav"] = pool.player_id.map(aav)
    pool = pool[(pool.aav.fillna(0) <= 8)].copy()
    w4 = W[W.season == T - 1].sort_values("pts", ascending=False).groupby("player_id").head(4) \
          .groupby("player_id")["pts"].mean()
    pool["w4"] = pool.player_id.map(w4).fillna(0)
    pool["heir"] = pool.player_id.isin(heirs(T - 1)).astype(int)
    pool["ap"] = pool.player_id.map(A[A.season == T - 1].set_index("player_id")["apct"]).fillna(0)
    bz = BZ[BZ.y == T].set_index("player_id")["spike"]
    pool["spike"] = pool.player_id.map(bz)
    q = pool.spike.quantile(0.75)
    pool["buzz"] = ((pool.spike >= q) & pool.spike.notna()).astype(int)
    # age at pick time = T-1 age + 1; STRICT gate — missing age fails
    pool["age"] = pool.player_id.map(AGE[AGE.season == T - 1].set_index("player_id")["age"]) + 1
    oT = out[out.season == T].set_index("player_id")
    pool["reliable"] = pool.player_id.map(oT["reliable"]).fillna(False).astype(bool)
    pool["star"] = pool.player_id.map(oT["star"]).fillna(False).astype(bool)
    base_r.append(pool.reliable.mean())

    t1 = pool[(pool.heir == 1) & (pool.buzz == 1)].sort_values("spike", ascending=False).head(4)
    rest = pool.drop(t1.index)
    t2 = rest[(rest.ap >= 0.80) & (rest.age <= 27)].sort_values("ap", ascending=False).head(3)
    rest = rest.drop(t2.index)
    t3 = rest[(rest.w4 >= 12) & (rest.age <= 26)].sort_values("w4", ascending=False).head(10 - len(t1) - len(t2))
    sel = pd.concat([t1, t2, t3]).head(10)
    h = int(sel.reliable.sum()); s = int(sel.star.sum()); tot_h += h; tot_s += s
    names = ", ".join((("*" if r.star else "+" if r.reliable else "") + r["name"].split()[-1])
                      for _, r in sel.iterrows())
    per_year.append(h)
    pr(f"{T:4d} {h:4d} {s:4d}   {names}")

n = 10 * 10
pr(f"\nTOTAL: {tot_h}/{n} reliable ({100*tot_h/n:.0f}%), {tot_s} star")
pr(f"cheap-pool base ~{100*np.mean(base_r):.1f}% reliable -> lift {tot_h/n/np.mean(base_r):.1f}x")
pr(f"worst year: {min(per_year)} hits; 2022-25: {sum(per_year[-4:])}/40 vs the composite's 5/40")
pr("reference: composite top-10 backtest 23/100 [10 star]. TRADE-OFF: the young gates")
pr("buy recent-year consistency (the market stopped letting old cheap FLASH hit) at")
pr("the cost of star upside — most of the composite's stars were 2016-21 vet ALPHAs.")

open(os.path.join(ROOT, "outputs", "reports", "hybrid_darts_backtest.md"), "w", encoding="utf-8").write(
    "# Hybrid darts — tiered 10-pick strategy, walk-forward 2016-2025\n\n"
    "Generated by `scripts/hybrid_darts_backtest.py`. Tier 1: all HEIR+BUZZ (the one\n"
    "synergy pair, `dart_synergy.md`); tier 2: up to 3 young buried ALPHAs (age<=27);\n"
    "tier 3: young FLASH fill (age<=26). Strict age gates (missing age fails).\n\n```\n"
    + "\n".join(_out) + "\n```\n")
print("\nwrote outputs/reports/hybrid_darts_backtest.md")
