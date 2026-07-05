"""Game-to-game player prop model + EV backtest vs Vegas closing lines.
Markets: reception yds / receptions / rush yds / pass yds / rush att / completions.
Features: rolling player stats+advanced (xFP, shares, EPA), prior-season archetype, team & opponent
context (implied total/spread), defense-allowed-to-position, defense-vs-archetype, player-vs-def-style.
Walk-forward (train strictly earlier seasons), P(over) from train residuals, edge vs de-vigged
consensus closing prob, ROI at best closing price. Honest question: are we +EV vs the closing line?"""
import os, sqlite3, warnings, sys, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import norm
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
MK = {"player_reception_yds": "receiving_yards", "player_receptions": "receptions",
      "player_rush_yds": "rushing_yards", "player_pass_yds": "passing_yards",
      "player_rush_attempts": "carries", "player_pass_completions": "completions"}
DISC = {"player_receptions": 1, "player_rush_attempts": 1, "player_pass_completions": 1}
con = sqlite3.connect(DB)

# ---------- 1) CLOSING LINES ----------
print("extracting closing lines...")
mlist = ",".join("'" + m + "'" for m in MK)
cl = pd.read_sql(f"""
WITH r AS (SELECT event_id,bookmaker,market,player_name,outcome_type,price,point,
   ROW_NUMBER() OVER (PARTITION BY event_id,bookmaker,market,player_name,outcome_type
                      ORDER BY snapshot_time DESC) rn
   FROM player_props WHERE market IN ({mlist}))
SELECT event_id,bookmaker,market,player_name,outcome_type,price,point FROM r WHERE rn=1""", con)
cl["dec"] = np.where(cl.price > 0, 1 + cl.price / 100.0, 1 + 100.0 / cl.price.abs())
med = cl.groupby(["event_id", "market", "player_name"]).point.median().rename("mline").reset_index()
cl = cl.merge(med, on=["event_id", "market", "player_name"])
cl = cl[cl.point == cl.mline]
piv = cl.pivot_table(index=["event_id", "market", "player_name", "bookmaker", "mline"],
                     columns="outcome_type", values="dec", aggfunc="first").reset_index()
piv = piv.dropna(subset=["Over", "Under"])
piv["p_o"] = 1 / piv.Over; piv["p_u"] = 1 / piv.Under
piv["novig"] = piv.p_o / (piv.p_o + piv.p_u)
mk = piv.groupby(["event_id", "market", "player_name", "mline"]).agg(
    novig=("novig", "median"), best_over=("Over", "max"), best_under=("Under", "max"),
    books=("novig", "size")).reset_index()
mk = mk[mk.books >= 3]
print(f"  {len(mk):,} closing markets (>=3 books) across {mk.event_id.nunique()} events")

# ---------- 2) PANEL ----------
ps = pd.read_sql("""SELECT event_id,player_id,player_display_name nm,position,team,opponent,
   season,week,completions,attempts,passing_yards,passing_epa,carries,rushing_yards,rushing_epa,
   receptions,targets,receiving_yards,receiving_epa,receiving_air_yards,target_share,air_yards_share,
   wopr,fantasy_points_ppr FROM player_stats""", con)
xf = pd.read_sql("""SELECT player_id,season,week,total_fantasy_points_exp xfp,rec_fantasy_points_exp xrec,
   rush_fantasy_points_exp xrush,pass_fantasy_points_exp xpass FROM nflv_ff_opp WHERE season>=2022""", con)
xf["season"] = xf.season.astype(int); xf["week"] = xf.week.astype(int)
gl = pd.read_sql("SELECT season,week,team,implied_team_total itt,team_spread spread,game_total gt FROM nflv_game_lines WHERE season>=2023", con)
pa = pd.read_sql("SELECT player_id,season,archetype arch,style pstyle FROM player_archetypes", con).drop_duplicates(["player_id", "season"])
ta = pd.read_sql("SELECT team,season,unit,style FROM team_archetypes", con)
con.close()
ps = ps.merge(xf, on=["player_id", "season", "week"], how="left").merge(gl, on=["season", "week", "team"], how="left")
pa["season"] += 1                                             # prior-season archetype (leakage-free)
ps = ps.merge(pa, on=["player_id", "season"], how="left")
td = ta[ta.unit == "defense"][["team", "season", "style"]].rename(columns={"team": "opponent", "style": "dstyle"})
td["season"] += 1
ps = ps.merge(td, on=["opponent", "season"], how="left")
ps = ps.sort_values(["season", "week"]).reset_index(drop=True)
ps["ord"] = ps.season * 100 + ps.week
g = ps.groupby("player_id", group_keys=False)
for c in ["receiving_yards", "receptions", "rushing_yards", "passing_yards", "carries", "completions",
          "targets", "attempts", "target_share", "air_yards_share", "wopr", "xfp", "xrec", "xrush",
          "xpass", "receiving_epa", "rushing_epa", "passing_epa", "receiving_air_yards"]:
    ps["l3_" + c] = g[c].apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ps["l6_" + c] = g[c].apply(lambda x: x.shift(1).rolling(6, min_periods=2).mean())
ps["gp"] = g.cumcount()
dal = ps.groupby(["opponent", "position", "season", "week"]).agg(
    a_ry=("receiving_yards", "sum"), a_rc=("receptions", "sum"),
    a_ru=("rushing_yards", "sum"), a_py=("passing_yards", "sum")).reset_index().sort_values(["season", "week"])
gd = dal.groupby(["opponent", "position"], group_keys=False)
for c in ["a_ry", "a_rc", "a_ru", "a_py"]:
    dal["d_" + c] = gd[c].apply(lambda x: x.shift(1).rolling(6, min_periods=2).mean())
ps = ps.merge(dal[["opponent", "position", "season", "week", "d_a_ry", "d_a_rc", "d_a_ru", "d_a_py"]],
              on=["opponent", "position", "season", "week"], how="left")
da = ps.dropna(subset=["arch"]).groupby(["opponent", "arch", "season", "week"]).fantasy_points_ppr.sum() \
       .rename("aa").reset_index().sort_values(["season", "week"])
da["d_arch"] = da.groupby(["opponent", "arch"], group_keys=False).aa.apply(lambda x: x.shift(1).rolling(6, min_periods=2).mean())
ps = ps.merge(da[["opponent", "arch", "season", "week", "d_arch"]], on=["opponent", "arch", "season", "week"], how="left")
ps["p_vs_style"] = ps.groupby(["player_id", "dstyle"], group_keys=False).fantasy_points_ppr \
                     .apply(lambda x: x.shift(1).expanding(1).mean())
ps["arch_id"] = ps.arch.astype("category").cat.codes
ps["dstyle_id"] = ps.dstyle.astype("category").cat.codes
ps["pos_id"] = ps.position.map({p: i for i, p in enumerate(["QB", "RB", "WR", "TE"])})
FE = [c for c in ps.columns if c.startswith(("l3_", "l6_", "d_"))] + \
     ["gp", "itt", "spread", "gt", "arch_id", "dstyle_id", "pos_id", "p_vs_style"]
print(f"panel {len(ps):,} player-games, {len(FE)} features")

# ---------- 3) match props to panel ----------
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
ps["k"] = ps.nm.map(nrm); mk["k"] = mk.player_name.map(nrm)
mm = mk.merge(ps, on=["event_id", "k"], how="inner")
print(f"prop-panel match: {len(mm):,} of {len(mk):,} markets ({len(mm)/len(mk):.0%})")

# ---------- 4) walk-forward per market + EV ----------
P = dict(objective="regression_l1", n_estimators=400, learning_rate=0.04, num_leaves=24,
         min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
rows = []
for market, stat in MK.items():
    d = mm[mm.market == market].copy(); d["y"] = d[stat]
    if len(d) < 400:
        print(f"{market}: too few ({len(d)})"); continue
    outs = []
    for T in (2024, 2025):
        tr = ps[(ps.ord < T * 100) & ps[stat].notna() & (ps["l3_" + stat] > 0)]
        te = d[d.season == T]
        if not len(te) or len(tr) < 800: continue
        m = lgb.LGBMRegressor(**P).fit(tr[FE].astype(float).fillna(-1), tr[stat])
        trp = m.predict(tr[FE].astype(float).fillna(-1)); res = tr[stat] - trp
        qb = pd.qcut(trp, 3, labels=False, duplicates="drop")
        sig = {int(k): max(res[qb == k].std(), 1e-3) for k in pd.unique(qb[~pd.isna(qb)])}
        cuts = np.quantile(trp, [1/3, 2/3])
        t2 = te.copy(); t2["mu"] = m.predict(te[FE].astype(float).fillna(-1))
        t2["sig"] = [sig.get(int(np.digitize(v, cuts)), res.std()) for v in t2.mu]
        outs.append(t2)
    if not outs: continue
    t = pd.concat(outs); adj = 0.5 * DISC.get(market, 0)
    t["p_over"] = 1 - norm.cdf((t.mline + adj - t.mu) / t.sig)
    t["edge"] = t.p_over - t.novig
    t["won_over"] = (t.y > t.mline).astype(int); t = t[t.y != t.mline]
    mae = (t.y - t.mu).abs().mean(); base = (t.y - t["l6_" + stat]).abs().mean()
    lmae = (t.y - t.mline).abs().mean()
    print(f"\n{market} (n={len(t)}): model MAE {mae:.2f} | naive-l6 {base:.2f} | LINE {lmae:.2f}")
    for th in (0.03, 0.06, 0.10):
        b = t[t.edge.abs() >= th].copy()
        if len(b) < 25:
            print(f"  edge>={th:.0%}: n={len(b)} (too few)"); continue
        b["side"] = np.where(b.edge > 0, 1, 0)
        b["win"] = (b.side == b.won_over).astype(int)
        b["dp"] = np.where(b.side == 1, b.best_over, b.best_under)
        b["pnl"] = np.where(b.win == 1, b.dp - 1, -1.0)
        print(f"  edge>={th:.0%}: n={len(b):4d}  hit {b.win.mean():.1%}  ROI {b.pnl.mean():+.1%}")
    rows.append(t.assign(market=market))
if rows:
    A = pd.concat(rows)
    A.to_pickle(os.path.join(ROOT, "outputs", "prop_ev_backtest.pkl"))
    b = A[A.edge.abs() >= 0.06].copy(); b["side"] = np.where(b.edge > 0, 1, 0)
    b["win"] = (b.side == b.won_over).astype(int)
    b["dp"] = np.where(b.side == 1, b.best_over, b.best_under)
    b["pnl"] = np.where(b.win == 1, b.dp - 1, -1.0)
    print(f"\nALL MARKETS edge>=6%: n={len(b)}, hit {b.win.mean():.1%}, ROI {b.pnl.mean():+.1%}")
    print("calibration (our p_over decile -> realized over rate):")
    A["decl"] = pd.qcut(A.p_over, 10, labels=False, duplicates="drop")
    print(A.groupby("decl").agg(p=("p_over", "mean"), real=("won_over", "mean"), n=("won_over", "size")).round(3).to_string())
