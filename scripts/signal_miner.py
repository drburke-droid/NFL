"""Systematic signal miner over props: is there ANY combination of stats / advanced stats /
game context / TEAMMATE PROP LINES that predicts what the closing market missed?
Discipline (anti-data-mining):
  - target = residual (won_over - novig), never raw ROI
  - Stage 1: global LGBM (tests millions of interactions implicitly), season-blocked walk-forward
  - Stage 2: interpretable rule miner (feature x threshold x side), n>=200, same-sign in 2024 AND
    2025, Benjamini-Hochberg FDR q<0.10 across ALL rules tested
NEW feature family: teammate prop lines (market-implied usage structure) - e.g. own line share of
team's total lines, QB pass-yds line for a receiver's team, teammates' max line."""
import os, sqlite3, warnings, sys, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import brier_score_loss
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
MK = {"player_reception_yds": "receiving_yards", "player_receptions": "receptions",
      "player_rush_yds": "rushing_yards", "player_pass_yds": "passing_yards",
      "player_rush_attempts": "carries", "player_pass_completions": "completions",
      "player_pass_attempts": "attempts", "player_pass_tds": "passing_tds"}
con = sqlite3.connect(DB)
mlist = ",".join("'" + m + "'" for m in MK)
cl = pd.read_sql(f"""WITH r AS (SELECT event_id,bookmaker,market,player_name,outcome_type,price,point,
   ROW_NUMBER() OVER (PARTITION BY event_id,bookmaker,market,player_name,outcome_type ORDER BY snapshot_time DESC) rn
   FROM player_props WHERE market IN ({mlist}))
SELECT event_id,bookmaker,market,player_name,outcome_type,price,point FROM r WHERE rn=1""", con)
cl["dec"] = np.where(cl.price > 0, 1 + cl.price / 100.0, 1 + 100.0 / cl.price.abs())
med = cl.groupby(["event_id", "market", "player_name"]).point.median().rename("mline").reset_index()
cl = cl.merge(med, on=["event_id", "market", "player_name"]); cl = cl[cl.point == cl.mline]
piv = cl.pivot_table(index=["event_id", "market", "player_name", "mline"], columns="outcome_type",
                     values="dec", aggfunc="median").reset_index().dropna(subset=["Over", "Under"])
piv["novig"] = (1 / piv.Over) / (1 / piv.Over + 1 / piv.Under)
nb = cl.groupby(["event_id", "market", "player_name", "mline"]).bookmaker.nunique().rename("books").reset_index()
piv = piv.merge(nb, on=["event_id", "market", "player_name", "mline"]); piv = piv[piv.books >= 3]
ps = pd.read_sql("""SELECT event_id,player_id,player_display_name nm,position,team,opponent,season,week,
   completions,attempts,passing_yards,passing_tds,passing_epa,carries,rushing_yards,rushing_epa,
   receptions,targets,receiving_yards,receiving_epa,receiving_air_yards,target_share,air_yards_share,
   wopr,fantasy_points_ppr FROM player_stats""", con)
xfw = pd.read_sql("""SELECT player_id,season,week,total_fantasy_points_exp xfp,
   total_fantasy_points_diff xgap FROM nflv_ff_opp WHERE season>=2022""", con)
xfw["season"] = xfw.season.astype(int); xfw["week"] = xfw.week.astype(int)
gl = pd.read_sql("SELECT season,week,team,implied_team_total itt,team_spread spread,game_total gt FROM nflv_game_lines WHERE season>=2023", con)
con.close()
ps = ps.merge(xfw, on=["player_id", "season", "week"], how="left").merge(gl, on=["season", "week", "team"], how="left")
ps = ps.sort_values(["season", "week"]).reset_index(drop=True)
g = ps.groupby("player_id", group_keys=False)
BASE = ["receiving_yards", "receptions", "rushing_yards", "passing_yards", "carries", "completions",
        "targets", "attempts", "target_share", "air_yards_share", "wopr", "xfp", "xgap",
        "receiving_epa", "rushing_epa", "passing_epa", "receiving_air_yards", "fantasy_points_ppr"]
for c in BASE:
    ps["l3_" + c] = g[c].apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ps["l6_" + c] = g[c].apply(lambda x: x.shift(1).rolling(6, min_periods=2).mean())
ps["vol3_ppr"] = g.fantasy_points_ppr.apply(lambda x: x.shift(1).rolling(3, min_periods=2).std())
ps["gp"] = g.cumcount()
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
ps["k"] = ps.nm.map(nrm); piv["k"] = piv.player_name.map(nrm)
M = piv.merge(ps, on=["event_id", "k"], how="inner")
M["y"] = [r[MK[m]] for m, r in zip(M.market, M.to_dict("records"))]
M = M.dropna(subset=["y", "novig"]); M = M[M.y != M.mline]
M["won_over"] = (M.y > M.mline).astype(int)
M["resid"] = M.won_over - M.novig
# ---------- TEAMMATE-PROP features (market-implied usage structure) ----------
tm = M[["event_id", "team", "market", "k", "mline"]].drop_duplicates()
agg = tm.groupby(["event_id", "team", "market"]).mline.agg(tm_sum="sum", tm_max="max", tm_n="size").reset_index()
M = M.merge(agg, on=["event_id", "team", "market"], how="left")
M["line_share"] = M.mline / M.tm_sum.replace(0, np.nan)              # own share of team's total lines
M["mates_sum"] = M.tm_sum - M.mline                                   # teammates' combined line
M["is_tm_max"] = (M.mline >= M.tm_max).astype(int)                    # market's designated alpha
qb = M[M.market == "player_pass_yds"].groupby(["event_id", "team"]).mline.max().rename("qb_line").reset_index()
M = M.merge(qb, on=["event_id", "team"], how="left")                  # team QB pass-yds line (market's team volume)
rb = M[M.market == "player_rush_attempts"].groupby(["event_id", "team"]).mline.max().rename("rb_att_line").reset_index()
M = M.merge(rb, on=["event_id", "team"], how="left")
M["line_vs_l6"] = [r["mline"] - (r.get("l6_" + MK[r["market"]]) or np.nan) for r in M.to_dict("records")]
FEAT = [c for c in M.columns if c.startswith(("l3_", "l6_"))] + \
       ["vol3_ppr", "gp", "itt", "spread", "gt", "line_share", "mates_sum", "is_tm_max",
        "qb_line", "rb_att_line", "line_vs_l6", "mline", "books"]
M["mkt_id"] = M.market.astype("category").cat.codes
print(f"mart: {len(M):,} settled consensus props 2023-25, {len(FEAT)} features (incl. teammate-line family)")
# ---------- STAGE 1: global LGBM, season-blocked ----------
print("\n=== STAGE 1: can ANY feature combination beat market prob? (Brier, walk-forward) ===")
GB = dict(objective="binary", n_estimators=400, learning_rate=0.03, num_leaves=31,
          min_child_samples=60, subsample=0.8, colsample_bytree=0.7, random_state=0, verbosity=-1)
for T in (2024, 2025):
    tr, te = M[M.season < T], M[M.season == T]
    m1 = lgb.LGBMClassifier(**GB).fit(tr[["novig", "mkt_id"] + FEAT].astype(float).fillna(-1), tr.won_over)
    p = np.clip(m1.predict_proba(te[["novig", "mkt_id"] + FEAT].astype(float).fillna(-1))[:, 1], .01, .99)
    shift = tr.won_over.mean() - tr.novig.mean()
    b_mkt = brier_score_loss(te.won_over, np.clip(te.novig, .01, .99))
    b_shr = brier_score_loss(te.won_over, np.clip(te.novig + shift, .01, .99))
    b_ml = brier_score_loss(te.won_over, p)
    print(f"  test {T}: market {b_mkt:.5f} | market+shrink {b_shr:.5f} | LGBM(all features) {b_ml:.5f} "
          f"{'<-- BEATS' if b_ml < b_shr else '(no gain)'}")
# ---------- STAGE 2: FDR-controlled rule miner ----------
print("\n=== STAGE 2: interpretable rules, n>=200, same sign 2024 & 2025, BH-FDR q<0.10 ===")
rules = []
Mq = M[M.season >= 2024]
for f in FEAT:
    v = Mq[f]
    if v.notna().mean() < 0.5: continue
    for q in (0.2, 0.8):
        thr = v.quantile(q)
        for dirn, mask in (("<=", v <= thr), (">=", v >= thr)):
            if q == 0.2 and dirn == ">=": continue
            if q == 0.8 and dirn == "<=": continue
            for scope in ("ALL",) + tuple(MK):
                mm2 = mask & ((Mq.market == scope) if scope != "ALL" else True)
                d = Mq[mm2]
                if len(d) < 200: continue
                e = d.resid.mean(); se = d.resid.std() / np.sqrt(len(d))
                if se == 0: continue
                s24, s25 = d[d.season == 2024].resid.mean(), d[d.season == 2025].resid.mean()
                if np.isnan(s24) or np.isnan(s25) or np.sign(s24) != np.sign(s25): continue
                from scipy.stats import norm as _n
                pval = 2 * (1 - _n.cdf(abs(e / se)))
                rules.append({"rule": f"{f} {dirn} p{int(q*100)}", "scope": scope, "n": len(d),
                              "resid": e, "z": e / se, "p": pval, "e24": s24, "e25": s25})
R = pd.DataFrame(rules)
print(f"  rules evaluated: {len(R):,} (both-year same-sign filter already applied)")
if len(R):
    R = R.sort_values("p").reset_index(drop=True)
    R["bh"] = R.p * len(R) / (R.index + 1)                    # Benjamini-Hochberg
    R["q"] = R.bh[::-1].cummin()[::-1]
    surv = R[R.q < 0.10]
    print(f"  SURVIVORS at FDR q<0.10: {len(surv)}")
    print(R.head(12)[["rule", "scope", "n", "resid", "z", "p", "q", "e24", "e25"]].round(4).to_string(index=False))

# ---------- STAGE 3: 2023 HOLDOUT validation of the two mined segments + vig test ----------
print("\n=== STAGE 3: 2023 holdout (data the miner never saw) + under-ROI at MEDIAN price ===")
segs={"big underdogs (spread>=p80)":M.spread>=Mq.spread.quantile(.8),
      "star receivers (tgt share>=p80)":M.l3_target_share>=Mq.l3_target_share.quantile(.8),
      "GLOBAL (all props)":M.novig.notna()}
for nm,mask in segs.items():
    for yr in (2023,2024,2025):
        d=M[mask&(M.season==yr)]
        if len(d)<150: continue
        uw=(d.won_over==0).mean(); pnl=np.where(d.won_over==0,d.Under-1,-1.0)
        print(f"  {nm:34s} {yr}: n={len(d):5d}  over-resid {d.resid.mean():+.4f}  underROI@median {pnl.mean():+.1%}")
    print()
