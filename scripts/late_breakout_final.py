"""
Final pass for the late-breakout study:
  1. no-market model design (works for 2026 where FFA isn't published yet),
     walk-forward validated on the same skill cheap pool
  2. paired bootstrap: blend vs ffa_points significance
  3. keeper-escalation distribution + mid-tier ($3-8) vs $1-dart economics
  4. calibrated P(hit) for the blend
  5. 2026 scoring: universe = nflv_rosters_2026, cheap = PRICE_ANCHOR <= $2 or
     absent, projection-rank component = board_2026 pred points
Writes outputs/models/late_breakout_2026.csv + nflv_late_breakout table.
"""
import os, sys, re, json, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "models")
DB = os.path.join(ROOT, "db", "nfl_odds.db")

df = pd.read_pickle(os.path.join(OUT, "late_breakout_frame.pkl"))
df = pd.concat([df, pd.get_dummies(df.position, prefix="pos")], axis=1)


def buzz_features(con_):
    """August Wikipedia-pageview buzz per player-season (validated: AUC +0.02,
    clf p@10 0.10->0.16; catches Puka-class camp risers). For the target season,
    August-to-date daily views are rate-scaled to a full month; NaN before August."""
    wb = pd.read_sql("SELECT * FROM nflv_wiki_buzz", con_)
    wb["year"] = wb.ym.str[:4].astype(int); wb["month"] = wb.ym.str[4:6].astype(int)
    rows = []
    for (pid, y), g in wb[wb.year >= 2016].groupby(["player_id", "year"]):
        m = dict(zip(g.month, g.views))
        aug = m.get(8, np.nan)
        base = np.nanmedian([m.get(k_, np.nan) for k_ in range(1, 6)])
        rows.append({"player_id": pid, "season": y, "aug_views": aug, "base_views": base})
    b = pd.DataFrame(rows)
    try:  # pre-draft refresh: partial-August dailies for the current season
        d = pd.read_sql("SELECT * FROM nflv_wiki_buzz_daily", con_)
        if len(d):
            d["year"] = d.date.str[:4].astype(int)
            aug = d[d.date.str[4:6] == "08"]
            if len(aug):
                g = aug.groupby(["player_id", "year"]).agg(v=("views", "sum"), n=("views", "size")).reset_index()
                g["aug_scaled"] = g.v * 31.0 / g.n
                b = b.merge(g.rename(columns={"year": "season"})[["player_id", "season", "aug_scaled"]],
                            on=["player_id", "season"], how="outer")
                b["aug_views"] = b.aug_views.fillna(b.aug_scaled)
                b = b.drop(columns=["aug_scaled"])
    except Exception:
        pass
    b["buzz_spike"] = np.log((b.aug_views + 10) / (b.base_views + 10))
    b["aug_pctl"] = b.groupby("season").aug_views.rank(pct=True)
    b["spike_pctl"] = b.groupby("season").buzz_spike.rank(pct=True)
    return b[["player_id", "season", "aug_views", "buzz_spike", "aug_pctl", "spike_pctl"]]


_conb = sqlite3.connect(DB)
df = df.merge(buzz_features(_conb), on=["player_id", "season"], how="left")
_conb.close()

MARKET = ["adp_f", "aav_f", "in_ffa", "ffa_points", "ffa_upside", "ffa_uncertainty"]
CORE = ["age", "years_exp", "is_rookie", "draft_round", "draft_pick",
        "prior_games", "prior_ppg", "prior2_ppg", "prior_cv", "prior_snap_pct",
        "prior_target_share", "prior_wopr", "prior_receiving_epa", "prior_rushing_epa",
        "prior_carries_pg", "prior_targets_pg", "prior_tds", "team_change",
        "ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_d_tch", "ht_slope",
        "vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets",
        "pos_RB", "pos_TE", "pos_WR",
        "aug_views", "buzz_spike", "aug_pctl", "spike_pctl"]
for c in MARKET + CORE:
    if c not in df.columns: df[c] = np.nan

P = dict(n_estimators=350, learning_rate=0.03, num_leaves=31, min_child_samples=40,
         subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
YEARS = range(2019, 2026)
allp = df[(df.season <= 2025) & (df.position != "QB")].copy()
cheap = allp[allp.cheap == 1]


def wf_clf(feats):
    preds = []
    for Y in YEARS:
        tr = allp[allp.season < Y]; te = cheap[cheap.season == Y]
        m = lgb.LGBMClassifier(**P).fit(tr[feats], tr.hit)
        t = te.copy(); t["prob"] = m.predict_proba(te[feats])[:, 1]; preds.append(t)
    return pd.concat(preds)


def pk(bt, score, k=10):
    return np.array([g.sort_values(score, ascending=False).head(k).hit.mean()
                     for _, g in bt.groupby("season")])


# 1. designs -----------------------------------------------------------------
bt_full = wf_clf(MARKET + CORE)
bt_nm = wf_clf(CORE)
for bt in (bt_full, bt_nm):
    bt["r_model"] = bt.groupby("season").prob.rank(pct=True)
    bt["r_ffa"] = bt.groupby("season").ffa_points.rank(pct=True).fillna(0)
    bt["blend"] = bt.r_model + bt.r_ffa

print("p@10 by design (mean over 2019-25):")
print(f"  full blend    {pk(bt_full,'blend').mean():.3f}")
print(f"  no-mkt blend  {pk(bt_nm,'blend').mean():.3f}")
print(f"  no-mkt clf    {pk(bt_nm,'prob').mean():.3f}")
print(f"  ffa_points    {pk(bt_full,'ffa_points').mean():.3f}  (NaN->last)")
print(f"  p@5: full blend {pk(bt_full,'blend',5).mean():.3f} | no-mkt blend {pk(bt_nm,'blend',5).mean():.3f} | ffa {pk(bt_full,'ffa_points',5).mean():.3f}")

# 2. paired bootstrap blend vs ffa (by season) --------------------------------
d = pk(bt_full, "blend") - pk(bt_full, "ffa_points")
boot = [np.random.choice(d, len(d)).mean() for _ in range(10000)]
print(f"\nblend - ffa p@10: mean {d.mean():+.3f}, P(diff>0) = {np.mean(np.array(boot) > 0):.2f}")

# 3. economics ----------------------------------------------------------------
ed = pd.read_csv(os.path.join(ROOT, "outputs", "espn_drafts.csv"))
ed["key"] = ed.player.str.lower().str.replace(r"[^a-z ]", "", regex=True)
ks = []
for y in [2024, 2025]:
    k = ed[(ed.season == y) & ed.keeper].merge(
        ed[ed.season == y - 1][["key", "bid"]].rename(columns={"bid": "prior_bid"}),
        on="key", how="left")
    ks.append(k)
k = pd.concat(ks); k["delta"] = k.bid - k.prior_bid
print("\nkeeper escalation deltas:", k.delta.describe().round(1).to_dict())
print("delta counts:", k.delta.value_counts().sort_index().to_dict())
print("by season:\n", k.groupby("season").delta.agg(["count", "median", "mean"]).round(2).to_string())

# mid-tier ($3-8 AAV, adp 121-168) vs $1 dart, per pick incl. keeper surplus
mid = allp[(allp.ffa_aav.between(3, 8)) & (allp.adp_f > 120) & (allp.season <= 2024)]
drt = cheap[cheap.season <= 2024]
for nm, g, cost in [("mid-tier $3-8", mid, 5.5), ("$1-2 dart (all)", drt, 1.5)]:
    surp = np.maximum(0, g.next_aav.fillna(0) - (cost + 4))
    print(f"{nm}: n={len(g)}, hit={g.hit.mean():.3f}, mean vorp={g.vorp.mean():.0f}, "
          f"mean nextAAV={g.next_aav.fillna(0).mean():.2f}, keeper surplus/pick=${surp.mean():.2f}")

# 4. calibration of the full blend --------------------------------------------
from sklearn.linear_model import LogisticRegression
cal = LogisticRegression().fit(bt_full[["blend"]], bt_full.hit)
bins = pd.qcut(bt_full.blend, [0, .5, .75, .9, .97, 1.0])
print("\nblend calibration:\n",
      bt_full.groupby(bins).agg(n=("hit", "size"), obs=("hit", "mean"),
                                pred=("blend", lambda s: cal.predict_proba(s.values.reshape(-1, 1))[:, 1].mean())).round(3).to_string())

# 5. 2026 scoring ---------------------------------------------------------------
con = sqlite3.connect(DB)
r26 = pd.read_sql("SELECT * FROM nflv_rosters_2026", con)
board = pd.read_sql("SELECT player_id, player_display_name AS name, position, team, age, "
                    "pred_ppg, proj_games, breakout_prob, is_rookie FROM board_2026", con)
# expected league price
txt = open(os.path.join(ROOT, "outputs", "draft_tool", "price_anchor_2026.js"), encoding="utf-8").read()
m = re.search(r"const PRICE_ANCHOR = (\{.*?\});", txt, re.S)
anchor = json.loads(m.group(1))
board["exp_price"] = [anchor.get(f"{r['name']}|{r.position}|{r.team}", 0.0) for _, r in board.iterrows()]

# features for 2026: prior season = 2025
f26 = df[df.season == 2026]
if len(f26) == 0:
    # rebuild minimal 2026 feature rows from the study's frame pieces
    import importlib.util
    spec = importlib.util.spec_from_file_location("lbs", os.path.join(ROOT, "scripts", "late_breakout_study.py"))
    lbs = importlib.util.module_from_spec(spec); spec.loader.exec_module(lbs)
    seas, ros, ffa, draft, cv, snaps, ht, opp = lbs.load_frames(con)
    u = board.merge(r26[["player_id"]], on="player_id", how="left")
    pr = seas[seas.season == 2025][["player_id", "recent_team", "games", "ppg", "target_share", "wopr",
                                    "receiving_epa", "rushing_epa", "carries", "targets",
                                    "passing_tds", "rushing_tds", "receiving_tds"]].copy()
    pr["prior_tds"] = pr[["passing_tds", "rushing_tds", "receiving_tds"]].sum(axis=1)
    pr["carries_pg"] = pr.carries / pr.games.replace(0, np.nan)
    pr["targets_pg"] = pr.targets / pr.games.replace(0, np.nan)
    pr = pr[["player_id", "recent_team", "games", "ppg", "target_share", "wopr", "receiving_epa",
             "rushing_epa", "carries_pg", "targets_pg", "prior_tds"]]
    pr.columns = ["player_id"] + ["prior_" + c for c in pr.columns[1:]]
    u = u.merge(pr, on="player_id", how="left")
    u = u.merge(seas[seas.season == 2024][["player_id", "ppg"]].rename(columns={"ppg": "prior2_ppg"}),
                on="player_id", how="left")
    u = u.merge(cv[cv.season == 2025][["player_id", "cv"]].rename(columns={"cv": "prior_cv"}), on="player_id", how="left")
    pmap = ros[ros.season == 2025][["gsis_id", "pfr_id", "years_exp"]].rename(columns={"gsis_id": "player_id"})
    pmap["years_exp"] += 1
    u = u.merge(pmap, on="player_id", how="left")
    u = u.merge(snaps[snaps.season == 2025][["pfr_player_id", "snap_pct"]]
                .rename(columns={"pfr_player_id": "pfr_id", "snap_pct": "prior_snap_pct"}), on="pfr_id", how="left")
    u = u.merge(ht[ht.season == 2025].drop(columns=["season"]), on="player_id", how="left")
    u = u.merge(opp[opp.season == 2026].drop(columns=["season", "team", "position"]), on="player_id", how="left")
    u = u.merge(draft, on="player_id", how="left")
    u["team_change"] = ((u.prior_recent_team.notna()) & (u.team != u.prior_recent_team)).astype(int)
    u["years_exp"] = u.years_exp.fillna(0)
    u.loc[u.is_rookie == 1, "years_exp"] = 0
    u["draft_round"] = u.draft_round.fillna(8); u["draft_pick"] = u.draft_pick.fillna(270)
    for p_ in ["RB", "TE", "WR"]:
        u[f"pos_{p_}"] = (u.position == p_).astype(int)
    f26 = u

f26 = f26[f26.position.isin(["RB", "WR", "TE"])].copy()
if "aug_pctl" not in f26.columns:      # from-scratch path: attach 2026 buzz (NaN pre-August)
    b26 = buzz_features(con)
    b26 = b26[b26.season == 2026].drop(columns=["season"])
    f26 = f26.merge(b26, on="player_id", how="left")
for c in CORE:
    if c not in f26.columns: f26[c] = np.nan

m_final = lgb.LGBMClassifier(**P).fit(allp[CORE], allp.hit)
f26["prob_raw"] = m_final.predict_proba(f26[CORE])[:, 1]

# blend with board projection points (proxy for ffa_points rank)
f26["proj_pts"] = f26.pred_ppg * f26.proj_games
# cheap = expected price <= $2 AND outside the drafted-starter zone (mirror of the
# historical ADP>120 condition; anchor misses default to $0 so price alone leaks stars)
board["proj_pts_all"] = board.pred_ppg * board.proj_games
board["proj_rank_all"] = board.proj_pts_all.rank(ascending=False)
f26 = f26.merge(board[["player_id", "proj_rank_all"]], on="player_id", how="left")
f26["cheap26"] = ((f26.exp_price.fillna(0) <= 2.0) & (f26.proj_rank_all > 120)).astype(int)
ch26 = f26[f26.cheap26 == 1].copy()
ch26["r_model"] = ch26.prob_raw.rank(pct=True)
ch26["r_proj"] = ch26.proj_pts.rank(pct=True).fillna(0)
ch26["blend"] = ch26.r_model + ch26.r_proj
ch26["p_hit"] = cal.predict_proba(ch26[["blend"]].values)[:, 1]
ch26["young"] = (ch26.years_exp <= 2).astype(int)

cols = ["name", "position", "team", "age", "years_exp", "is_rookie", "draft_round",
        "exp_price", "prior_ppg", "prior_snap_pct", "prior_target_share",
        "vac_pc_targets", "vac_rb_carries", "ht_d_ppg", "proj_pts", "prob_raw", "p_hit", "young"]
top = ch26.sort_values("p_hit", ascending=False)
print(f"\n2026 cheap pool (exp price <= $2): {len(ch26)} players")
print("\nTOP 25 overall:\n", top[cols].head(25).round(3).to_string(index=False))
print("\nTOP 15 YOUNG (<=2 yrs exp) keeper darts:\n",
      top[top.young == 1][cols].head(15).round(3).to_string(index=False))

top[cols + ["player_id"]].to_csv(os.path.join(OUT, "late_breakout_2026.csv"), index=False)
top[["player_id", "name", "position", "team", "exp_price", "p_hit", "young"]].to_sql(
    "nflv_late_breakout", con, if_exists="replace", index=False)
print("\nwrote late_breakout_2026.csv + nflv_late_breakout")
