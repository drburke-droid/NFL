"""Game scripts: are they real, can they be called before kickoff, and what would knowing one be worth?

Three questions, asked of every player-week 2016-25 that has an FFA weekly line and a box score
(the league frame: league/data/visible_frame.parquet + the 2025 holdout), joined to the per-game
script labels and final scores in db/nfl_odds.db (game_scripts, built from quarter-by-quarter
trajectories by cluster_game_scripts.py) and the closing spread/total.

 a) Is a game's fantasy outcome "a script"? Cluster the per-game RESIDUAL shape (how each
    position on each side landed vs its FFA line) and the raw shape; report silhouette. Then
    the honest version of the question: how much of a player's miss is shared with the other
    players in the same game (the ceiling for ANY game-level information), and how much of
    that a 5-way label captures.
 b) Can the script be called pregame? Walk-forward multinomial logit on spread / total / FFA
    team projections; accuracy and log-loss against the base rate.
 c) If an oracle named the script before kickoff and we applied a fitted per-(script, side,
    position) multiplier to the FFA line, how far does MAE fall? Then the reader version: a
    picker who is right a% of the time and otherwise picks like the pregame model -- where is
    break-even?

Three taxonomies are tested so the answer does not hinge on one set of names:
    traj   the DB's 5 quarter-trajectory clusters (Shootout, Wire-to-Wire Blowout, 2nd Half
           Blowout, Steady Build, Tight Throughout), side = winner/loser
    abs    fan-facing 5 classes on the final score: Fav blowout (fav by 14+), Upset (dog wins,
           spread >= 2.5), Shootout (total >= 52), Slog (total <= 37), Standard; side = fav/dog
    rel    same 5 but Shootout/Slog relative to the closing total (+/-7), the "surprise" form

Usage:
    python scripts/game_script_oracle_study.py [--out outputs/reports/game_script_oracle_study.md]
"""
import os, sys, argparse, sqlite3, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, log_loss
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "db", "nfl_odds.db")
LABELS = os.environ.get("LEAGUE_PRIVATE_LABELS",
                        "C:/Users/drbur/Downloads/model_burke_private/league_holdout/holdout_labels.parquet")
POS = ["QB", "RB", "WR", "TE"]
TEAMFIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}
L = []


def w(s=""):
    L.append(s); print(s)


def md(df, index_name=""):
    head = [index_name] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [f"{v}" for v in row.values]) + " |")
    return "\n".join(out)


def gid_of(df):
    lo = df[["team", "opponent_team"]].min(axis=1); hi = df[["team", "opponent_team"]].max(axis=1)
    return df.season.astype(str) + "_" + df.week.astype(str) + "_" + lo + "_" + hi


# ----------------------------------------------------------------------------- data
def load_players():
    cols = ["player_id", "position", "team", "opponent_team", "season", "week", "baseline_proj",
            "ctx_spread", "ctx_total", "ctx_implied", "ctx_home", "ffa_pass_yds", "ffa_rush_yds",
            "ffa_rec_yds", "ffa_pass_tds", "ffa_rush_tds", "ffa_rec_tds"]
    v = pd.read_parquet(os.path.join(ROOT, "league", "data", "visible_frame.parquet"),
                        columns=cols + ["actual_ppr", "actual_active"])
    v["_rid"] = v.index
    h = pd.read_parquet(os.path.join(ROOT, "league", "data", "holdout_features.parquet"), columns=cols)
    if os.path.exists(LABELS):
        lab = pd.read_parquet(LABELS)[["player_id", "season", "week", "actual_ppr", "actual_active"]]
        h = h.merge(lab, on=["player_id", "season", "week"], how="inner")
        h["_rid"] = -1
        v = pd.concat([v, h], ignore_index=True)
    else:
        print("holdout labels not found; 2025 excluded", file=sys.stderr)
    v = v[v.position.isin(POS)].copy()
    v["team"] = v.team.replace(TEAMFIX); v["opponent_team"] = v.opponent_team.replace(TEAMFIX)
    return v.dropna(subset=["team", "opponent_team", "baseline_proj", "actual_ppr"])


def load_games():
    c = sqlite3.connect(DB)
    g = pd.read_sql("select season, week, home_team, away_team, game_script, total_pts, home_total, away_total "
                    "from game_scripts", c)
    for k in ("home_team", "away_team"):
        g[k] = g[k].replace(TEAMFIX)
    return g.drop_duplicates(["season", "week", "home_team", "away_team"])


def team_games(p, g):
    """One row per (season, week, team) with the game's outcome, market and FFA team projection."""
    t = p.groupby(["season", "week", "team", "opponent_team"]).agg(
        spread=("ctx_spread", "first"), total_line=("ctx_total", "first"), home=("ctx_home", "first"),
        proj_tot=("baseline_proj", "sum"), proj_pass=("ffa_pass_yds", "sum"), proj_rush=("ffa_rush_yds", "sum"),
        proj_rec=("ffa_rec_yds", "sum"), act_tot=("actual_ppr", "sum")).reset_index()
    for pos in POS:
        s = p[p.position == pos].groupby(["season", "week", "team"]).agg(
            **{f"proj_{pos}": ("baseline_proj", "sum"), f"act_{pos}": ("actual_ppr", "sum")}).reset_index()
        t = t.merge(s, on=["season", "week", "team"], how="left")
    fill = {f"proj_{pos}": 0 for pos in POS}; fill.update({f"act_{pos}": 0 for pos in POS})
    t = t.fillna(fill)
    gh = g.rename(columns={"home_team": "team", "away_team": "opponent_team", "home_total": "pts_for", "away_total": "pts_against"})
    ga = g.rename(columns={"away_team": "team", "home_team": "opponent_team", "away_total": "pts_for", "home_total": "pts_against"})
    gg = pd.concat([gh, ga], ignore_index=True)[["season", "week", "team", "opponent_team", "game_script", "total_pts", "pts_for", "pts_against"]]
    t = t.merge(gg, on=["season", "week", "team", "opponent_team"], how="inner")
    t = t.dropna(subset=["spread", "total_line"])
    t["margin"] = t.pts_for - t.pts_against
    t["side_wl"] = np.where(t.margin > 0, "winner", "loser")
    t["side_fd"] = np.where(t.spread < 0, "fav", np.where(t.spread > 0, "dog", "pick"))
    fav_margin = np.where(t.spread < 0, t.margin, -t.margin)
    fav_margin = np.where(t.spread == 0, np.where(t.home == 1, t.margin, -t.margin), fav_margin)
    t["fav_margin"] = fav_margin
    t["abs_spread"] = t.spread.abs()
    t["total_surprise"] = t.total_pts - t.total_line
    t["gid"] = gid_of(t)
    return t


def label_abs(t):
    s = np.full(len(t), "Standard", dtype=object)
    s[(t.total_pts <= 37).values] = "Slog"
    s[(t.total_pts >= 52).values] = "Shootout"
    s[((t.fav_margin < 0) & (t.abs_spread >= 2.5)).values] = "Upset"
    s[(t.fav_margin >= 14).values] = "Fav blowout"
    return pd.Series(s, index=t.index)


def label_rel(t):
    s = np.full(len(t), "Standard", dtype=object)
    s[(t.total_surprise <= -7).values] = "Slog"
    s[(t.total_surprise >= 7).values] = "Shootout"
    s[((t.fav_margin < 0) & (t.abs_spread >= 2.5)).values] = "Upset"
    s[(t.fav_margin >= 14).values] = "Fav blowout"
    return pd.Series(s, index=t.index)


# ----------------------------------------------------------------------------- a) discreteness
def game_matrix(t):
    """Per game (one row): both sides' residual by position + totals, favourite first."""
    t = t[t.side_fd != "pick"].copy()
    for pos in POS:
        t[f"res_{pos}"] = t[f"act_{pos}"] - t[f"proj_{pos}"]
    t["res_tot"] = t.act_tot - t.proj_tot
    f = t[t.side_fd == "fav"].set_index("gid")
    d = t[t.side_fd == "dog"].set_index("gid")
    both = f.join(d, rsuffix="_dog", how="inner")
    res = [f"res_{pos}" for pos in POS] + ["res_tot"]
    raw = [f"act_{pos}" for pos in POS] + ["act_tot"]
    R = both[res + [c + "_dog" for c in res]]
    A = both[raw + [c + "_dog" for c in raw]]
    return both, R, A


def silhouettes(X, ks=range(2, 9)):
    Z = StandardScaler().fit_transform(X.values)
    out = []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=0).fit(Z)
        gm = GaussianMixture(k, n_init=3, random_state=0, covariance_type="full").fit(Z)
        out.append(dict(k=k, silhouette=round(silhouette_score(Z, km.labels_, sample_size=3000, random_state=0), 3),
                        gmm_bic=int(gm.bic(Z)), sizes=", ".join(str(x) for x in sorted(np.bincount(km.labels_), reverse=True))))
    return pd.DataFrame(out).set_index("k")


def variance_shares(p, t, taxa):
    """Of the player-level residual variance (played rows), how much sits at the team-game and
    game level, and how much of THAT each label set explains."""
    d = p.merge(t[["season", "week", "team", "gid", "side_wl", "side_fd", "margin", "total_pts", "total_surprise",
                   "fav_margin", "game_script", "lab_abs", "lab_rel", "abs_spread", "total_line", "act_tot", "proj_tot"]],
                on=["season", "week", "team"], how="inner")
    d = d[d.actual_active == 1].copy()
    d["res"] = d.actual_ppr - d.baseline_proj
    tot = d.res.var()
    rows = {}
    tg = d.groupby(["season", "week", "team"]).res.transform("mean")
    rows["team-game mean residual (ceiling for any game+side info)"] = 1 - (d.res - tg).var() / tot
    gm = d.groupby("gid").res.transform("mean")
    rows["game mean residual (both sides pooled)"] = 1 - (d.res - gm).var() / tot
    tgp = d.groupby(["season", "week", "team", "position"]).res.transform("mean")
    rows["team-game x position mean residual"] = 1 - (d.res - tgp).var() / tot
    for name, lab, side in taxa:
        cell = d.groupby([lab, side, "position"]).res.transform("mean")
        rows[f"{name}: (script, side, position) cell mean"] = 1 - (d.res - cell).var() / tot
    X = pd.get_dummies(d[["position"]], drop_first=False).astype(float)
    for c in ("margin", "total_pts", "total_surprise", "fav_margin"):
        for pos in POS:
            X[f"{c}_{pos}"] = d[c].values * (d.position == pos).values
            X[f"{c}_{pos}_x_proj"] = d[c].values * (d.position == pos).values * d.baseline_proj.values
    r = Ridge(1.0).fit(X.values, d.res.values)
    rows["ridge on realized margin/total x position x proj (continuous, in-sample)"] = r.score(X.values, d.res.values)
    return pd.Series(rows).round(4).to_frame("share of residual variance"), d


# ----------------------------------------------------------------------------- b) predictability
def pregame_model(both, lab, seasons):
    """Walk-forward multinomial logit + GBM on pregame features. Returns per-game predicted probs (index = gid)."""
    feats_v = ["abs_spread", "total_line", "home"]
    feats_f = feats_v + ["proj_tot", "proj_tot_dog", "proj_pass", "proj_pass_dog", "proj_rush", "proj_rush_dog",
                         "proj_QB", "proj_QB_dog", "proj_RB", "proj_RB_dog", "proj_WR", "proj_WR_dog"]
    feats_h = feats_f + ["hist_total", "hist_total_dog", "hist_margin", "hist_margin_dog", "hist_res", "hist_res_dog",
                         "hist_absmargin", "hist_absmargin_dog"]
    b = both.dropna(subset=feats_h + [lab]).copy()
    classes = sorted(b[lab].unique())
    res = []
    probs = pd.DataFrame(index=b.index, columns=classes, dtype=float)
    pick = pd.Series(index=b.index, dtype=object)
    for s in seasons:
        tr, te = b[b.season < s], b[b.season == s]
        if len(tr) < 300 or len(te) == 0:
            continue
        base = tr[lab].value_counts(normalize=True).reindex(classes).fillna(1e-6).values
        Pb = np.tile(base, (len(te), 1))
        row = dict(season=s, n=len(te), base_acc=round(float((te[lab] == classes[int(base.argmax())]).mean()), 3),
                   base_ll=round(log_loss(te[lab], Pb, labels=classes), 4))
        for name, feats in (("vegas", feats_v), ("vegas+ffa", feats_f), ("+history", feats_h)):
            sc = StandardScaler().fit(tr[feats].values)
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr[feats].values), tr[lab])
            P = lr.predict_proba(sc.transform(te[feats].values))
            P = pd.DataFrame(P, columns=lr.classes_, index=te.index).reindex(columns=classes).fillna(1e-6).values
            row[f"{name}_acc"] = round(float((np.array(classes)[P.argmax(1)] == te[lab].values).mean()), 3)
            row[f"{name}_ll"] = round(log_loss(te[lab], P, labels=classes), 4)
            if name == "+history":
                probs.loc[te.index, :] = P
                pick.loc[te.index] = np.array(classes)[P.argmax(1)]
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0).fit(tr[feats_h].values, tr[lab])
        P = pd.DataFrame(gb.predict_proba(te[feats_h].values), columns=gb.classes_, index=te.index).reindex(columns=classes).fillna(1e-6).values
        row["gbm_acc"] = round(float((np.array(classes)[P.argmax(1)] == te[lab].values).mean()), 3)
        row["gbm_ll"] = round(log_loss(te[lab], P, labels=classes), 4)
        res.append(row)
    return pd.DataFrame(res).set_index("season"), probs.dropna(), pick.dropna()


# ----------------------------------------------------------------------------- c) oracle
def fit_factors(tr, lab, side, min_n=40, shrink=50.0):
    """Per (script, side, position) multiplier on the FFA line, sum(actual)/sum(proj), shrunk to 1
    with pseudo-count `shrink` rows. Fitted on played rows with proj > 0.5 (a multiplier on a
    near-zero line is meaningless; those rows are left at the baseline)."""
    d = tr[(tr.baseline_proj > 0.5)]
    g = d.groupby([lab, side, "position"]).agg(a=("actual_ppr", "sum"), p=("baseline_proj", "sum"), n=("actual_ppr", "size"))
    pseudo = shrink * g.p / g.n.clip(lower=1)
    g["f"] = (g.a + pseudo) / (g.p + pseudo)
    g.loc[g.n < min_n, "f"] = 1.0
    return g["f"]


def apply_factors(te, fac, side, labcol):
    key = pd.MultiIndex.from_arrays([te[labcol].values, te[side].values, te.position.values])
    f = fac.reindex(key).fillna(1.0).values
    f = np.where(te.baseline_proj.values > 0.5, f, 1.0)
    return te.baseline_proj.values * f


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def oracle_run(d, lab, side, seasons, pick):
    """Walk-forward: factors from seasons < s, applied to season s with the TRUE label (oracle) and
    with the pregame model's pick. Returns per-season MAE table and row-level oracle predictions."""
    rows, preds = [], {}
    for s in seasons:
        tr, te = d[d.season < s], d[d.season == s]
        if len(tr) < 2000 or len(te) == 0:
            continue
        fac = fit_factors(tr, lab, side)
        po = apply_factors(te, fac, side, lab)
        te2 = te.copy(); te2["_pick"] = pick.reindex(te2.gid).values
        te2["_pick"] = te2["_pick"].fillna(te2[lab])
        pp = apply_factors(te2, fac, side, "_pick")
        rows.append(dict(season=s, n=len(te), base=mae(te.actual_ppr, te.baseline_proj), oracle=mae(te.actual_ppr, po),
                         pregame_pick=mae(te.actual_ppr, pp)))
        preds[s] = pd.DataFrame({"oracle": po, "pregame_pick": pp}, index=te.index)
    out = pd.DataFrame(rows).set_index("season")
    out.loc["ALL"] = out.mean(numeric_only=True); out.loc["ALL", "n"] = out.n.iloc[:-1].sum()
    return out, pd.concat(preds.values())


def accuracy_curve(d, lab, side, seasons, probs, base_rates):
    """Expected MAE of a picker right with probability a on every game, wrong picks drawn from the
    pregame model's probabilities over the OTHER classes (renormalised). Linear in a by row."""
    rows = []
    classes = list(probs.columns)
    for s in seasons:
        tr, te = d[d.season < s], d[d.season == s]
        if len(tr) < 2000 or len(te) == 0:
            continue
        fac = fit_factors(tr, lab, side)
        e_true = np.abs(te.actual_ppr.values - apply_factors(te, fac, side, lab))
        P = probs.reindex(te.gid).values.astype(float)
        fallback = np.tile(base_rates.reindex(classes).fillna(0).values, (len(te), 1))
        P = np.where(np.isnan(P), fallback, P)
        true_idx = np.array([classes.index(x) for x in te[lab].values])
        P[np.arange(len(te)), true_idx] = 0.0
        P = P / P.sum(1, keepdims=True)
        e_wrong = np.zeros(len(te))
        for j, c in enumerate(classes):
            te2 = te.copy(); te2["_c"] = c
            e_wrong += P[:, j] * np.abs(te.actual_ppr.values - apply_factors(te2, fac, side, "_c"))
        rows.append(pd.DataFrame({"e_true": e_true, "e_wrong": e_wrong,
                                  "e_base": np.abs(te.actual_ppr.values - te.baseline_proj.values)}))
    E = pd.concat(rows)
    out = {}
    for a in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out[f"{a:.0%}"] = a * E.e_true.mean() + (1 - a) * E.e_wrong.mean()
    base = E.e_base.mean()
    be = np.nan
    if E.e_wrong.mean() > base > E.e_true.mean():
        be = (E.e_wrong.mean() - base) / (E.e_wrong.mean() - E.e_true.mean())
    return pd.Series(out), base, be


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "reports", "game_script_oracle_study.md"))
    args = ap.parse_args()

    p = load_players(); g = load_games()
    t = team_games(p, g)
    t["lab_abs"] = label_abs(t); t["lab_rel"] = label_rel(t)
    # each team's recent tendencies, known before kickoff: trailing-8 means (prior games only)
    t = t.sort_values(["team", "season", "week"])
    t["res_tot"] = t.act_tot - t.proj_tot
    for c, h in (("total_pts", "hist_total"), ("margin", "hist_margin"), ("res_tot", "hist_res")):
        t[h] = t.groupby("team")[c].transform(lambda x: x.shift(1).rolling(8, min_periods=3).mean())
    t["hist_absmargin"] = t.groupby("team")["margin"].transform(lambda x: x.abs().shift(1).rolling(8, min_periods=3).mean())
    for h in ("hist_total", "hist_margin", "hist_res", "hist_absmargin"):
        t[h] = t[h].fillna(t[h].median())
    seasons_all = sorted(t.season.unique())
    taxa = [("traj", "game_script", "side_wl"), ("abs", "lab_abs", "side_fd"), ("rel", "lab_rel", "side_fd")]

    w(f"# Game scripts as a fantasy signal -- {t.gid.nunique()} games, {len(t)} team-games, seasons {min(seasons_all)}-{max(seasons_all)}")
    w()
    w("Player rows: every QB/RB/WR/TE with an FFA weekly line (league frame 2016-24 + 2025 holdout), PPR. Baseline = the FFA "
      "consensus line scored PPR (the league's reference; Model_Burke sits ~1% under it). MAE is on played rows "
      "(any touch/attempt), the judge's convention. Scores and the 5 quarter-trajectory script labels come from "
      "db/nfl_odds.db game_scripts; spread/total are the closing lines in the frame.")
    w()
    w("## The three taxonomies and how often each script happens")
    w()
    for name, lab, side in taxa:
        one = t.drop_duplicates("gid")
        vc = one[lab].value_counts(normalize=True).round(3)
        desc = one.groupby(lab).agg(n=("gid", "size"), total=("total_pts", "mean"), abs_margin=("margin", lambda x: x.abs().mean()),
                                    fav_margin=("fav_margin", "mean"), total_line=("total_line", "mean"), abs_spread=("abs_spread", "mean")).round(1)
        desc["share"] = vc
        w(f"**{name}** (side = {side.replace('side_', '')})")
        w(); w(md(desc.sort_values("n", ascending=False), "script")); w()

    # ---------------- a
    w("## a) Are there discrete scripts in the fantasy data?")
    w()
    both, R, A = game_matrix(t)
    w("Per game, a 10-vector: each side's actual-minus-FFA residual by position and in total (favourite, then dog). "
      "If scripts were real categories these would clump.")
    w(); w("Residual shape (what an oracle would need to exploit):"); w()
    w(md(silhouettes(R), "k")); w()
    w("Raw shape (actual PPR by position and side):"); w()
    w(md(silhouettes(A), "k")); w()
    Z = StandardScaler().fit_transform(R.values)
    km5 = KMeans(5, n_init=20, random_state=0).fit(Z).labels_
    ami = {name: round(adjusted_mutual_info_score(both[lab].values, km5), 3) for name, lab, _ in taxa}
    w(f"Adjusted mutual information between a k=5 clustering of the residual shape and each taxonomy: {ami} "
      "(0 = independent, 1 = identical).")
    w()
    w("### Where the miss lives: variance decomposition of the player residual (played rows)")
    w()
    vs, d = variance_shares(p, t, taxa)
    w(md(vs, "component")); w()
    w("Reading: the first row is the ceiling for any information that is constant across a team's players in a game -- "
      "an oracle who knew the team's exact fantasy total residual. Everything below it is what a label or the final "
      "score recovers of that ceiling. The (script, side, position) cell means are in-sample (~40 cells) so they are "
      "slightly generous.")
    w()
    w("### Effect sizes: mean residual by script, side, position (all seasons, played rows, FFA line > 0.5)")
    w()
    dd = d[(d.baseline_proj > 0.5) & (d.side_fd != "pick")]
    for name, lab, side in taxa:
        e = dd.groupby([lab, side, "position"]).agg(n=("actual_ppr", "size"), a=("actual_ppr", "sum"), p=("baseline_proj", "sum"))
        e["ratio"] = (e.a / e.p).round(2); e["mean_res"] = ((e.a - e.p) / e.n).round(2)
        piv_r = e["ratio"].unstack("position").reindex(columns=POS)
        piv_m = e["mean_res"].unstack("position").reindex(columns=POS)
        w(f"**{name}** -- actual/FFA ratio (left) and mean residual in PPR points (right)"); w()
        w(md(pd.concat([piv_r, piv_m.add_suffix(" pts")], axis=1), "script, side")); w()

    # ---------------- b
    w("## b) Can the script be called before kickoff?")
    w()
    w("Walk-forward by season (train on all earlier seasons, test on the season shown; first test season needs 2 "
      "years of history). Features: |spread|, total line, home-favourite flag (vegas); plus each side's FFA team "
      "projections by position and pass/rush yards (vegas+ffa); plus each team's trailing-8-game mean total, margin, "
      "|margin| and fantasy residual (+history; the GBM uses this full set). base = always pick the most common script. "
      "The pregame pick used in (c) is the +history logit's most likely class.")
    w()
    seasons_wf = [s for s in seasons_all if s >= min(seasons_all) + 2]
    picks, probs_all, base_rates = {}, {}, {}
    for name, lab, side in taxa:
        tab, probs, pick = pregame_model(both, lab, seasons_wf)
        picks[name], probs_all[name] = pick, probs
        base_rates[name] = both[lab].value_counts(normalize=True)
        tab.loc["ALL"] = tab.mean(numeric_only=True).round(3); tab.loc["ALL", "n"] = tab.n.iloc[:-1].sum()
        w(f"**{name}**"); w(); w(md(tab, "season")); w()
        mx = probs.max(axis=1)
        w(f"Top predicted probability (+history logit): median {mx.median():.2f}, above 0.5 in {(mx > 0.5).mean():.0%} of games, above 0.6 in {(mx > 0.6).mean():.0%}.")
        w()
    one = both.copy()
    one["spread_bin"] = pd.cut(one.abs_spread, [-0.1, 2.5, 6.5, 10, 30], labels=["0-2.5", "3-6.5", "7-10", "10.5+"])
    one["total_bin"] = pd.cut(one.total_line, [0, 41.5, 45.5, 49.5, 80], labels=["<=41.5", "42-45.5", "46-49.5", "50+"])
    w("What the closing line already implies (abs taxonomy, % of games):"); w()
    w(md((pd.crosstab(one.spread_bin, one.lab_abs, normalize="index") * 100).round(0).astype(int), "spread")); w()
    w(md((pd.crosstab(one.total_bin, one.lab_abs, normalize="index") * 100).round(0).astype(int), "total line")); w()

    # ---------------- c
    w("## c) The oracle: MAE if the script were known before kickoff")
    w()
    w("Per (script, side, position) multiplier on the FFA line = sum(actual)/sum(FFA) in the training seasons, shrunk "
      "toward 1 with a 50-row pseudo-count, cells under 40 rows left at 1. Fitted on seasons before the test season, "
      "applied with the TRUE script (oracle) and with the pregame model's pick (what a line-reading fan would do). "
      "Rows with an FFA line of 0.5 or less are never touched.")
    w()
    d = d[d.side_fd != "pick"].copy()
    summary = []
    for name, lab, side in taxa:
        tab, preds = oracle_run(d, lab, side, seasons_wf, picks[name])
        tab = tab.round(4)
        tab["oracle_gain_%"] = ((tab.base - tab.oracle) / tab.base * 100).round(2)
        tab["pick_gain_%"] = ((tab.base - tab.pregame_pick) / tab.base * 100).round(2)
        w(f"**{name}**"); w(); w(md(tab, "season")); w()
        summary.append(dict(taxonomy=name, base=tab.loc["ALL", "base"], oracle=tab.loc["ALL", "oracle"], oracle_gain_pct=tab.loc["ALL", "oracle_gain_%"],
                            pregame_pick=tab.loc["ALL", "pregame_pick"], pick_gain_pct=tab.loc["ALL", "pick_gain_%"]))
        dt = d.loc[preds.index]
        bp = pd.DataFrame({"pos": dt.position.values, "base": np.abs(dt.actual_ppr.values - dt.baseline_proj.values),
                           "oracle": np.abs(dt.actual_ppr.values - preds.oracle.values)}).groupby("pos").mean().reindex(POS).round(3)
        bp["gain_%"] = ((bp.base - bp.oracle) / bp.base * 100).round(2)
        w("By position (pooled test seasons, oracle):"); w(); w(md(bp, "position")); w()
        tier = pd.cut(dt.baseline_proj, [-100, 5, 10, 15, 100], labels=["0-5", "5-10", "10-15", "15+"])
        bt = pd.DataFrame({"tier": tier.values, "base": np.abs(dt.actual_ppr.values - dt.baseline_proj.values),
                           "oracle": np.abs(dt.actual_ppr.values - preds.oracle.values)}).groupby("tier").mean().round(3)
        bt["gain_%"] = ((bt.base - bt.oracle) / bt.base * 100).round(2)
        w("By FFA line tier:"); w(); w(md(bt, "FFA line")); w()
        cur, base, be = accuracy_curve(d, lab, side, seasons_wf, probs_all[name], base_rates[name])
        w(f"Reader accuracy curve: expected MAE when the reader names the right script a% of the time and otherwise "
          f"picks in proportion to the pregame model's probabilities over the other scripts. Baseline {base:.4f}; "
          f"break-even accuracy {'n/a' if np.isnan(be) else f'{be:.0%}'}; base rate of the most common script "
          f"{base_rates[name].max():.0%}; pregame model accuracy is in (b). The baseline here is row-pooled across test seasons, "
          f"the per-season table above averages season MAEs, hence the small difference.")
        w(); w(md(pd.DataFrame(cur.round(4)).T.rename(index={0: "MAE"}), "reader accuracy")); w()

    w("### Upper bounds: exact final score, and the team's exact fantasy total")
    w()
    rows = []

    def X_of(q):
        X = pd.DataFrame(index=q.index)
        for pos in POS:
            m = (q.position == pos).values.astype(float)
            X[f"p_{pos}"] = m * q.baseline_proj.values
            for c in ("margin", "total_pts", "total_surprise"):
                X[f"{c}_{pos}"] = m * q[c].values
                X[f"{c}_{pos}_xp"] = m * q[c].values * q.baseline_proj.values
        return X.values

    for s in seasons_wf:
        tr, te = d[d.season < s], d[d.season == s]
        if len(tr) < 2000 or len(te) == 0:
            continue
        r = Ridge(10.0).fit(X_of(tr), (tr.actual_ppr - tr.baseline_proj).values)
        ps = te.baseline_proj.values + r.predict(X_of(te))
        ratio = (te.act_tot / te.proj_tot.clip(lower=1)).clip(0.2, 5).values
        pt = np.where(te.baseline_proj > 0.5, te.baseline_proj * ratio, te.baseline_proj)
        rows.append(dict(season=s, n=len(te), base=mae(te.actual_ppr, te.baseline_proj), exact_score=mae(te.actual_ppr, ps),
                         exact_team_fantasy_total=mae(te.actual_ppr, pt)))
    ub = pd.DataFrame(rows).set_index("season"); ub.loc["ALL"] = ub.mean(numeric_only=True); ub.loc["ALL", "n"] = ub.n.iloc[:-1].sum()
    ub = ub.round(4)
    ub["score_gain_%"] = ((ub.base - ub.exact_score) / ub.base * 100).round(2)
    ub["team_total_gain_%"] = ((ub.base - ub.exact_team_fantasy_total) / ub.base * 100).round(2)
    w(md(ub, "season")); w()
    w("exact_score = ridge on the realized margin, total and total-surprise (x position, x FFA line): the most a "
      "perfectly-called final score could give. exact_team_fantasy_total = every player on a side scaled by that side's "
      "realized fantasy total / projected total: the ceiling for ANY team-level information, not attainable.")
    w()

    mb_path = os.path.join(ROOT, "league", "registry", "submissions", "model_burke_prod_visible.parquet")
    if os.path.exists(mb_path):
        mb = pd.read_parquet(mb_path)
        dm = d[d._rid.isin(mb.index)].copy()
        dm["mb"] = mb.pred.reindex(dm._rid).values
        rows = []
        for name, lab, side in taxa:
            fac = fit_factors(d[d.season < 2023], lab, side)
            key = pd.MultiIndex.from_arrays([dm[lab].values, dm[side].values, dm.position.values])
            f = np.where(dm.baseline_proj.values > 0.5, fac.reindex(key).fillna(1.0).values, 1.0)
            rows.append(dict(taxonomy=name, ffa=mae(dm.actual_ppr, dm.baseline_proj), ffa_oracle=mae(dm.actual_ppr, dm.baseline_proj * f),
                             model_burke=mae(dm.actual_ppr, dm.mb), model_burke_oracle=mae(dm.actual_ppr, dm.mb * f)))
        mbt = pd.DataFrame(rows).set_index("taxonomy").round(4)
        w(f"### Does the oracle gain survive on Model_Burke? (league eval rows 2023-24, n={len(dm)}, factors fitted 2016-22)"); w()
        w(md(mbt, "taxonomy")); w()

    w("## Summary")
    w(); w(md(pd.DataFrame(summary).set_index("taxonomy"), "taxonomy")); w()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
