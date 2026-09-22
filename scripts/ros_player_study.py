"""Rest-of-season player model study: what predicts a player's remaining per-game scoring?

The trade calculator (gm/players.py) estimates every rostered player's rest-of-season mean as a
games/(games+4) blend of this season's to-date average and last season's. That is one prior
season and nothing else: no career history, no next-week projection, no age, no archetype. This
study asks whether any of those, over the 2012-2026 weekly history, predicts the rest of the
season better -- and by how much.

Frame: every (season, decision week w) for w = 2..12, every skill player in a roster-depth pool.
Features are known at the end of week w; the target is the player's mean league-scored points in
the games he plays from week w+1 through week 17. Walk-forward by season: fit on seasons before
S, score S, for S = 2019..2025.

    python scripts/ros_player_study.py                # full study, writes outputs/reports/ros_player_study.md
    python scripts/ros_player_study.py --rebuild      # rebuild the cached frame first
"""
import argparse, glob, json, os, re, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gm.config import load as load_league
from gm.players import (norm, SKILL, PRIOR_GAMES, weekly_in_league_scoring, ffa_file_scored,
                        history_features, ridge_design, MODEL_COLS, MODEL as MODEL_PATH)

CACHE = os.path.join(ROOT, "data", "nflverse_cache")
DB = os.path.join(ROOT, "db", "nfl_odds.db")
FFA_DIR = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
FRAME = os.path.join(ROOT, "outputs", "models", "ros_player_frame.parquet")
REPORT = os.path.join(ROOT, "outputs", "reports", "ros_player_study.md")
LEAGUE = os.path.join(ROOT, "gm", "leagues", "kuhn_2026.json")

SEASONS = list(range(2012, 2027))
DECISION_WEEKS = list(range(2, 13))
HORIZON = 17                                  # league plays through week 17
POOL = {"QB": 30, "RB": 75, "WR": 90, "TE": 30}   # deeper than a 12-team roster, not the whole league
MIN_ROS_GAMES = 4
TEST_SEASONS = list(range(2019, 2026))

L = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t, flush=True); L.append(t)


# ----------------------------------------------------------------------------- data
def birth_dates():
    c = sqlite3.connect(DB)
    r = pd.read_sql("select gsis_id, birth_date from nflv_rosters where birth_date is not null", c)
    r["birth_date"] = pd.to_datetime(r.birth_date, errors="coerce")
    return r.dropna().drop_duplicates("gsis_id").set_index("gsis_id").birth_date


def archetypes():
    c = sqlite3.connect(DB)
    a = pd.read_sql("select player_id, season, archetype as arch_role, style as arch_style from player_archetypes", c)
    t = pd.read_sql("select player_id, season, archetype as arch_traj, skill_composite, trend2, ppg_z from player_traj_arch", c)
    return a.drop_duplicates(["player_id", "season"]), t.drop_duplicates(["player_id", "season"])


def ffa_history(cfg):
    """Every FFA weekly file, scored by the league: {(season, week): frame(key, position, ffa_next, ffa_sd)}."""
    out = {}
    for f in glob.glob(os.path.join(FFA_DIR, "raw_stats_*_wk*.csv")):
        s, w = map(int, re.search(r"raw_stats_(\d{4})_wk(\d+)", f).groups())
        out[(s, w)] = ffa_file_scored(cfg, f)
    return out


def ffa_for(ffa, season, week):
    """The projection for week+1, or the current week's as a stand-in -- the runtime rule."""
    if (season, week + 1) in ffa:
        return ffa[(season, week + 1)], False
    if (season, week) in ffa:
        return ffa[(season, week)], True
    return None, True


def build_frame(cfg):
    wk = weekly_in_league_scoring(cfg, SEASONS)
    say(f"weekly rows {len(wk):,} seasons {wk.season.min()}-{wk.season.max()}")
    bd = birth_dates()
    arch, traj = archetypes()
    ffa = ffa_history(cfg)
    say(f"FFA weekly files {len(ffa)}")
    out = []
    for s in SEASONS:
        ws = wk[wk.season == s]
        if ws.empty:
            continue
        for w in DECISION_WEEKS:
            if w > ws.week.max():
                continue
            fa, stale = ffa_for(ffa, s, w)
            f = history_features(wk, s, w, fa)
            ros = ws[(ws.week > w) & (ws.week <= HORIZON)]
            f = f.join(ros.groupby("player_id").pts.agg(ros_mean="mean", ros_games="size"))
            f["season"], f["ffa_stale"] = s, float(stale)
            f["remaining"] = HORIZON - w
            out.append(f.reset_index())
    F = pd.concat(out, ignore_index=True)
    F["position"] = F.position.astype(str)
    F["week"] = F.week.astype(int)
    # age at 1 Sep of the season; FFA's own birthdate covers players the roster table lacks
    ffa_bd = {}
    for f in sorted(glob.glob(os.path.join(FFA_DIR, "raw_stats_*_wk*.csv")))[-40:]:
        d = pd.read_csv(f, low_memory=False, usecols=["player", "birthdate"]).dropna()
        ffa_bd.update(dict(zip(d.player.map(norm), pd.to_datetime(d.birthdate, errors="coerce"))))
    b = F.player_id.map(bd)
    b = b.fillna(F.key.map(ffa_bd))
    F["age"] = (pd.to_datetime(F.season.astype(str) + "-09-01") - b).dt.days / 365.25
    # prior-season archetypes (what was known before the season started)
    F = F.merge(arch.assign(season=arch.season + 1), on=["player_id", "season"], how="left")
    F = F.merge(traj.assign(season=traj.season + 1), on=["player_id", "season"], how="left")
    # pool: top-N at position by a naive blend, so the frame is roster-relevant not league-wide
    F["naive"] = F.apply(lambda r: _gm_blend(r, {"QB": 10, "RB": 6, "WR": 6, "TE": 4}), axis=1)
    F["pool_rank"] = F.groupby(["season", "week", "position"]).naive.rank(ascending=False, method="first")
    F = F[F.pool_rank <= F.position.map(POOL)].copy()
    os.makedirs(os.path.dirname(FRAME), exist_ok=True)
    F.to_parquet(FRAME)
    say(f"frame {len(F):,} rows, {F.season.nunique()} seasons, target rows {F.ros_games.ge(MIN_ROS_GAMES).sum():,}")
    return pd.read_parquet(FRAME)      # the same dtypes whether built now or cached


def _gm_blend(r, repl):
    """The estimate gm/players.py makes today: games/(games+4) blend of this season and last."""
    n = r.get("ytd_games", np.nan); c = r.get("ytd_mean", np.nan); p = r.get("p1_mean", np.nan)
    if pd.notna(c) and pd.notna(p):
        return (c * n + p * PRIOR_GAMES) / (n + PRIOR_GAMES)
    if pd.notna(c):
        return (c * n + repl.get(r["position"], 5) * PRIOR_GAMES) / (n + PRIOR_GAMES)
    if pd.notna(p):
        return p
    return repl.get(r["position"], 5)


# ----------------------------------------------------------------------------- models
NUM_HIST = ["ytd_mean", "ytd_games", "last4_mean", "ytd_max", "p1_mean", "p1_games", "p2_mean", "p2_games",
            "car_mean", "car_games", "car_seasons", "week"]
# the ridge cannot bend ytd weight with games played, so give it the bent versions
NUM_SHRINK = ["ytd_w", "ytd_wt", "p1_w", "p1_wt"]
NUM_PROJ = ["ffa_next", "ffa_sd"]
assert NUM_HIST + NUM_SHRINK + NUM_PROJ == MODEL_COLS
NUM_AGE = ["age"]
CAT_ARCH = ["arch_role", "arch_style", "arch_traj"]
NUM_TRAJ = ["skill_composite", "trend2", "ppg_z"]


def fit_predict_gbm(tr, te, num, cats, seed=0):
    import lightgbm as lgb
    X = tr[num + cats].copy(); Xt = te[num + cats].copy()
    for c in cats:
        allc = pd.Categorical(pd.concat([X[c], Xt[c]]).astype(str))
        X[c] = pd.Categorical(X[c].astype(str), categories=allc.categories)
        Xt[c] = pd.Categorical(Xt[c].astype(str), categories=allc.categories)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                          objective="l1", verbose=-1, random_state=seed)
    m.fit(X, tr.ros_mean, sample_weight=tr.ros_games)
    return m.predict(Xt), m


def fit_predict_ridge(tr, te, cols):
    from sklearn.linear_model import Ridge
    out = np.full(len(te), np.nan)
    for p in SKILL:
        a, b = tr[tr.position == p], te[te.position == p]
        if a.empty or b.empty:
            continue
        spec = ridge_spec(a, cols)
        m = Ridge(alpha=1.0).fit(ridge_design(a, spec), a.ros_mean, sample_weight=a.ros_games)
        out[te.index.get_indexer(b.index)] = m.predict(ridge_design(b, spec))
    return out


def ridge_spec(a, cols):
    """Medians and missingness flags from the training rows: the part of the model that is data."""
    return {"cols": list(cols), "medians": {c: float(a[c].median()) for c in cols},
            "na_cols": [c for c in cols if a[c].isna().any()]}


def fit_production(F, summary):
    """Fit the chosen spec on every season with a target and store it for gm/players.py."""
    from sklearn.linear_model import Ridge
    T = F[(F.ros_games >= MIN_ROS_GAMES) & (F.season <= 2025)]
    model = {"fitted_on": f"{int(T.season.min())}-{int(T.season.max())}", "rows": int(len(T)),
             "features": MODEL_COLS, "walk_forward": summary, "positions": {}}
    for p in SKILL:
        a = T[T.position == p]
        spec = ridge_spec(a, MODEL_COLS)
        m = Ridge(alpha=1.0).fit(ridge_design(a, spec), a.ros_mean, sample_weight=a.ros_games)
        spec.update(coef=[float(x) for x in m.coef_], intercept=float(m.intercept_), rows=int(len(a)))
        model["positions"][p] = spec
        names = list(ridge_design(a.head(1), spec).columns)
        say(f"  {p}: n {len(a):,}  " + "  ".join(f"{n} {c:+.3f}" for n, c in zip(names, m.coef_)
                                               if abs(c) >= 0.05) + f"  intercept {m.intercept_:+.2f}")
    with open(MODEL_PATH, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)
    say(f"wrote {MODEL_PATH}")


def walk_forward(F, specs):
    """{name: predictions aligned to F.index} over the test seasons."""
    preds = {k: pd.Series(np.nan, index=F.index) for k in specs}
    T = F[F.ros_games >= MIN_ROS_GAMES]
    for s in TEST_SEASONS:
        te = T[T.season == s]
        for name, spec in specs.items():
            tr = T[(T.season < s) & (T.season >= spec.get("from", 2013))]
            if spec["kind"] == "gm":
                repl = {p: replacement_level_like(tr, p) for p in SKILL}
                preds[name].loc[te.index] = te.apply(lambda r: _gm_blend(r, repl), axis=1).values
            elif spec["kind"] == "ridge":
                preds[name].loc[te.index] = fit_predict_ridge(tr, te, spec["cols"])
            elif spec["kind"] == "gbm":
                preds[name].loc[te.index], _ = fit_predict_gbm(tr, te, spec["num"], spec.get("cats", []))
    return preds


def replacement_level_like(tr, p):
    """Roughly the gm module's replacement level: the last starter's average at position."""
    depth = {"QB": 12, "RB": 36, "WR": 36, "TE": 24}[p]
    last = tr[(tr.position == p) & (tr.week == 12)]
    if last.empty:
        return 5.0
    def pick(g):
        v = g.p1_mean.dropna().sort_values(ascending=False)
        return v.iloc[min(depth - 1, len(v) - 1)] if len(v) else np.nan
    v = last.groupby("season").apply(pick)
    return float(v.mean())


def score(F, preds, mask=None):
    T = F[F.ros_games >= MIN_ROS_GAMES]
    T = T[T.season.isin(TEST_SEASONS)]
    if mask is not None:
        T = T[mask.loc[T.index]]
    rows = []
    for name, p in preds.items():
        e = T.assign(pred=p.loc[T.index]).dropna(subset=["pred"])
        err = e.pred - e.ros_mean
        rho = e.groupby(["season", "week", "position"]).apply(
            lambda g: spearmanr(g.pred, g.ros_mean)[0] if len(g) > 5 else np.nan).mean()
        rows.append({"model": name, "n": len(e), "MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
                     "bias": err.mean(), "rho": rho})
    return pd.DataFrame(rows).set_index("model")


def score_by(F, preds, col, bins=None):
    T = F[(F.ros_games >= MIN_ROS_GAMES) & F.season.isin(TEST_SEASONS)]
    key = pd.cut(T[col], bins) if bins else T[col]
    out = {}
    for k in sorted(key.dropna().unique(), key=str):
        m = pd.Series(False, index=F.index); m.loc[T.index[key == k]] = True
        out[str(k)] = score(F, preds, m).MAE
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fit-production", action="store_true", help="fit C3 on 2013-25 and write gm/ros_model.json")
    args = ap.parse_args()
    cfg = load_league(LEAGUE)
    F = build_frame(cfg) if (args.rebuild or not os.path.exists(FRAME)) else pd.read_parquet(FRAME)
    say(f"frame {len(F):,} rows; FFA next-week coverage {F.ffa_next.notna().mean():.1%} overall, "
        f"{F[F.season >= 2016].ffa_next.notna().mean():.1%} from 2016")

    specs = {
        "A gm blend (today)":          {"kind": "gm"},
        "B ridge history":             {"kind": "ridge", "cols": NUM_HIST},
        "C ridge history+FFA":         {"kind": "ridge", "cols": NUM_HIST + NUM_PROJ, "from": 2016},
        "C2 ridge shrink+FFA":         {"kind": "ridge", "cols": NUM_HIST + NUM_SHRINK + NUM_PROJ, "from": 2016},
        "C3 ridge shrink+FFA (2013+)": {"kind": "ridge", "cols": NUM_HIST + NUM_SHRINK + NUM_PROJ},
        "D ridge history+FFA+age":     {"kind": "ridge", "cols": NUM_HIST + NUM_PROJ + NUM_AGE, "from": 2016},
        "E ridge FFA only":            {"kind": "ridge", "cols": NUM_PROJ + ["week"], "from": 2016},
        "F gbm history":               {"kind": "gbm", "num": NUM_HIST, "cats": ["position"]},
        "G gbm history+FFA":           {"kind": "gbm", "num": NUM_HIST + NUM_PROJ, "cats": ["position"], "from": 2016},
        "H gbm +FFA+age":              {"kind": "gbm", "num": NUM_HIST + NUM_PROJ + NUM_AGE, "cats": ["position"], "from": 2016},
        "I gbm +FFA+age+archetype":    {"kind": "gbm", "num": NUM_HIST + NUM_PROJ + NUM_AGE + NUM_TRAJ,
                                        "cats": ["position"] + CAT_ARCH, "from": 2016},
        "J gbm history+age+arch (no FFA)": {"kind": "gbm", "num": NUM_HIST + NUM_AGE + NUM_TRAJ,
                                        "cats": ["position"] + CAT_ARCH},
    }
    preds = walk_forward(F, specs)

    say("\n## Overall, walk-forward test seasons 2019-2025, decision weeks 2-12, ROS games >= 4\n")
    overall = score(F, preds)
    say(overall.round(3).to_markdown())
    say("\n## MAE by position\n")
    say(score_by(F, preds, "position").round(3).to_markdown())
    say("\n## MAE by decision week\n")
    say(score_by(F, preds, "week", bins=[1, 4, 8, 12]).round(3).to_markdown())
    say("\n## MAE by test season\n")
    say(score_by(F, preds, "season").round(3).to_markdown())
    top = F.naive.ge(F.groupby(["season", "week", "position"]).naive.transform(lambda s: s.quantile(0.75)))
    say("\n## MAE, top quartile of the pool at each position (the players who get traded)\n")
    say(score(F, preds, top).round(3).to_markdown())

    # the pairs the user is asking about
    say("\n## Burrow vs Mahomes under each model, every decision week 2021-2026\n")
    names = ["Joe Burrow", "Patrick Mahomes"]
    T = F[F.player_display_name.isin(names) & (F.season >= 2021)]
    # refit on everything available for 2026
    full_preds = {}
    tr = F[(F.ros_games >= MIN_ROS_GAMES) & (F.season <= 2025)]
    te26 = F[F.season == 2026]
    for name in ["A gm blend (today)", "C2 ridge shrink+FFA", "G gbm history+FFA", "I gbm +FFA+age+archetype"]:
        spec = specs[name]
        trs = tr[tr.season >= spec.get("from", 2013)]
        if spec["kind"] == "gm":
            repl = {p: replacement_level_like(trs, p) for p in SKILL}
            full_preds[name] = te26.apply(lambda r: _gm_blend(r, repl), axis=1)
        elif spec["kind"] == "ridge":
            full_preds[name] = pd.Series(fit_predict_ridge(trs, te26, spec["cols"]), index=te26.index)
        else:
            full_preds[name] = pd.Series(fit_predict_gbm(trs, te26, spec["num"], spec.get("cats", []))[0], index=te26.index)
    rows = []
    for _, r in T.sort_values(["season", "week"]).iterrows():
        d = {"season": r.season, "wk": r.week, "player": r.player_display_name.split()[-1],
             "ytd": r.ytd_mean, "n": r.ytd_games, "p1": r.p1_mean, "car": r.car_mean, "ffa+1": r.ffa_next,
             "age": r.age, "actual ROS": r.ros_mean, "ros_g": r.ros_games}
        for name in ["A gm blend (today)", "C2 ridge shrink+FFA", "G gbm history+FFA", "I gbm +FFA+age+archetype"]:
            d[name.split()[0]] = full_preds[name].get(r.name, preds[name].get(r.name, np.nan)) if r.season == 2026 else preds[name].get(r.name, np.nan)
        rows.append(d)
    say(pd.DataFrame(rows).round(1).to_markdown(index=False))

    say("\n## 2026 week-2 QB board under the candidate models (top 16 by model I)\n")
    q = te26[te26.position == "QB"].copy()
    for name, p in full_preds.items():
        q[name.split()[0]] = p.loc[q.index]
    q = q.sort_values("C2", ascending=False).head(16)
    say(q[["player_display_name", "age", "ytd_mean", "ytd_games", "p1_mean", "car_mean", "ffa_next", "arch_traj", "A", "C2", "G", "I"]]
        .round(1).to_markdown(index=False))

    # feature importance of the fullest model
    _, m = fit_predict_gbm(tr[tr.season >= 2016], te26, specs["I gbm +FFA+age+archetype"]["num"], specs["I gbm +FFA+age+archetype"]["cats"])
    say("")
    say("## 2026 FFA feature: rows using the current-week file as stand-in")
    say("")
    say(f"stale share, 2026 rows: {te26.ffa_stale.mean():.2f}; test rows: {F[F.season.isin(TEST_SEASONS)].ffa_stale.mean():.3f}")
    imp = pd.Series(m.booster_.feature_importance("gain"), index=m.feature_name_).sort_values(ascending=False)
    say("\n## Gain importance, model I\n")
    say((imp / imp.sum()).round(3).to_frame("share").to_markdown())

    if args.fit_production:
        say("\n## Production fit: C3 ridge shrink+FFA on every season with a target\n")
        fit_production(F, {k: round(float(v), 4) for k, v in overall.loc["C3 ridge shrink+FFA (2013+)"].items()})

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# Rest-of-season player model study\n\n" + "\n".join(L) + "\n")
    say(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
