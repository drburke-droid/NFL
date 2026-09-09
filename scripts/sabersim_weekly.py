"""SaberSim trial — full-slate weekly projections CSV, 75 minutes before each game.

Baseline = FFAnalytics weekly consensus stat lines (data/ffanalytics/FFAn_weekly/
raw_stats_{S}_wk{W}.csv) scored as PPR (4-pt pass TD, -2 INT, 1/rec, -2 fumble),
the same scoring as the training target (nflverse fantasy_points_ppr).
Model_Burke (walk-forward residual correction + conformal quantiles, the package
in Downloads/model_burke_pkg.zip) corrects QB/RB/WR/TE on 2023-25 history built
from the same FFA weekly baseline; K and DST pass through the scored FFA line.

Usage:
  python scripts/sabersim_weekly.py <model_burke pkg dir> [--week N] [--all-games]
                                     [--hours H] [--out path.csv]
Default: current week (first week whose kickoffs are not all in the past), only
games that have not kicked off yet (what a 75-min-before send should contain).
--hours H restricts to games kicking off within H hours (e.g. one slate).
Writes outputs/sabersim/projections_{S}_wk{W}_{stamp}.csv  (+ a full-run parquet).

Running on another PC (fresh checkout of this repo):
  1. unzip model_burke_pkg.zip somewhere; argv[1] = its pkg/ folder
  2. copy data/odds_api_key.txt (gitignored, one key per line) — only the free /events
     endpoint is called, for kickoff times
  3. pip install pandas numpy scikit-learn scipy pyarrow
  4. per week: drop raw_stats_2026_wkN.csv in data/ffanalytics/FFAn_weekly/ and the K/DST
     paste in data/kdst/paste_2026_wkN.txt, run parse_kdst_paste.py, then this script.
  Box scores / game lines come from data/sabersim/*.parquet (exported from the odds DB;
  the DB itself is not needed). 2026 box scores are pulled from nflverse when published.
"""
import argparse, json, os, re, sqlite3, sys, time, urllib.error, urllib.request, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("pkg", nargs="?", default=os.environ.get("MODEL_BURKE_PKG", ""))
ap.add_argument("--week", type=int, default=None)
ap.add_argument("--all-games", action="store_true", help="include games already kicked off")
ap.add_argument("--hours", type=float, default=None, help="only games kicking off within H hours")
ap.add_argument("--out", default=None)
ap.add_argument("--kdst-weight", type=float, default=0.65,
                help="weight on the pasted K/DST projection; remainder on the FFA-scored value")
A = ap.parse_args()
if not A.pkg or not os.path.isdir(os.path.join(A.pkg, "model_burke")):
    raise SystemExit("pass the Model_Burke package dir as argv[1] (the pkg/ folder from model_burke_pkg.zip, "
                     "i.e. the folder CONTAINING model_burke/), or set MODEL_BURKE_PKG")
sys.path.insert(0, A.pkg)
from model_burke import pipeline
from model_burke.features import build_lagged_features

FDIR = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
OUTD = os.path.join(ROOT, "outputs", "sabersim"); os.makedirs(OUTD, exist_ok=True)
API = "https://api.the-odds-api.com/v4"
SEASON = 2026
TEAM_FIX = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}
NAME2ABBR = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
ABBR2NAME = {v: k for k, v in NAME2ABBR.items()}
DOME = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LAC", "LA", "MIN", "NO"}

def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)

# ---------- Odds API key rotation (same contract as props_watch) ----------
_kp = os.path.join(ROOT, "data", "odds_api_key.txt")
if not os.path.exists(_kp):
    raise SystemExit("data/odds_api_key.txt is missing (gitignored) — copy it from the other PC, one key per line")
KEYS = [k.strip() for k in open(_kp) if k.strip() and not k.startswith("#")]
_ki = [0]
def get(url):
    while True:
        key = KEYS[_ki[0]]; sep = "&" if "?" in url else "?"
        try:
            with urllib.request.urlopen(f"{url}{sep}apiKey={key}", timeout=30) as r:
                rem = r.headers.get("x-requests-remaining"); body = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403, 429) and _ki[0] + 1 < len(KEYS):
                print(f"  key #{_ki[0]+1} rejected ({e.code}) — rotating"); _ki[0] += 1; continue
            raise
        if rem is not None and float(rem) <= 0 and _ki[0] + 1 < len(KEYS):
            _ki[0] += 1
        return body, rem

# ---------- 1. FFA weekly stat lines -> PPR baseline ----------
def score_ppr(d):
    z = lambda c: d[c].fillna(0) if c in d.columns else 0
    return (z("pass_yds") * .04 + z("pass_tds") * 4 - z("pass_int") * 2
            + z("rush_yds") * .1 + z("rush_tds") * 6 + z("rec") * 1 + z("rec_yds") * .1
            + z("rec_tds") * 6 - z("fumbles_lost") * 2 + z("two_pts") * 2 + z("return_tds") * 6)
def score_k(d):
    z = lambda c: d[c].fillna(0) if c in d.columns else 0
    return z("fg_0019") * 3 + z("fg_2029") * 3 + z("fg_3039") * 3 + z("fg_4049") * 4 + z("fg_50") * 5 + z("xp")
def dst_pa_points(opp_total):
    """expected DK points-allowed bucket score, opponent total ~ N(implied, 10)."""
    if pd.isna(opp_total): return 2.0
    from scipy.stats import norm as N
    edges = [(-1, 0, 10), (0, 6, 7), (6, 13, 4), (13, 20, 1), (20, 27, 0), (27, 34, -1), (34, 200, -4)]
    return sum((N.cdf(hi + .5, opp_total, 10) - N.cdf(lo + .5, opp_total, 10)) * p for lo, hi, p in edges)
def score_dst(d, opp_total):
    z = lambda c: d[c].fillna(0) if c in d.columns else 0
    base = z("dst_sacks") * 1 + z("dst_int") * 2 + z("dst_safety") * 2 + z("dst_td") * 6 + z("dst_blk") * 2 + 1.4
    return base + opp_total.map(dst_pa_points)

def load_ffa(s, w):
    p = os.path.join(FDIR, f"raw_stats_{s}_wk{w}.csv")
    if not os.path.exists(p): return None
    d = pd.read_csv(p, na_values=["NA"])
    d = d[d.avg_type == "weighted"] if "avg_type" in d.columns else d
    d["team"] = d.team.map(lambda t: TEAM_FIX.get(t, t))
    d["nname"] = d.player.map(norm); d["season"], d["week"] = s, w
    return d.drop_duplicates(["nname", "position"])

files = sorted(re.findall(r"raw_stats_(\d{4})_wk(\d+)\.csv", " ".join(os.listdir(FDIR))))
weeks_avail = sorted({(int(s), int(w)) for s, w in files})
print(f"FFA weekly files: {len(weeks_avail)} ({weeks_avail[0]} .. {weeks_avail[-1]})")

# ---------- 2. box scores, ids, lags ----------
# box scores + game lines: repo parquets (exported from db/nfl_odds.db so the script runs on
# any checkout); the DB is only consulted when the parquets are missing
PQ = os.path.join(ROOT, "data", "sabersim")
if os.path.exists(os.path.join(PQ, "weekly_skill_2022_2025.parquet")):
    wk = pd.read_parquet(os.path.join(PQ, "weekly_skill_2022_2025.parquet"))
    gl = pd.read_parquet(os.path.join(PQ, "game_lines_2023_2026.parquet"))
else:
    con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
    wk = pd.read_sql("""SELECT player_id, player_display_name, position, season, week, team,
                        opponent_team, targets, carries, receptions, receiving_yards, rushing_yards,
                        passing_yards, attempts, target_share, wopr, air_yards_share,
                        receiving_air_yards, fantasy_points_ppr, passing_epa, rushing_epa, receiving_epa
                        FROM nflv_weekly WHERE season BETWEEN 2022 AND 2025
                        AND position IN ('QB','RB','WR','TE')""", con)
    gl = pd.read_sql("SELECT season, week, team, opp, is_home, game_total, team_spread, implied_team_total "
                     "FROM nflv_game_lines WHERE season BETWEEN 2023 AND 2026", con)
    con.close()
for yr in (SEASON,):   # in-season box scores from nflverse
    for url in (f"player_stats/player_stats_{yr}.parquet", f"player_stats/stats_player_week_{yr}.parquet"):
        try:
            w26 = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/" + url)
            w26 = w26[w26.position.isin(["QB", "RB", "WR", "TE"])]
            wk = pd.concat([wk, w26[[c for c in wk.columns if c in w26.columns]]], ignore_index=True)
            print(f"  {yr} box scores: {len(w26):,} rows"); break
        except Exception as e: err = str(e)[:50]
    else: print(f"  {yr} box scores not published yet ({err})")
wk = wk.drop_duplicates(["player_id", "season", "week"])
wk["nname"] = wk.player_display_name.map(norm)
lag = build_lagged_features(wk)
ids = (wk.sort_values(["season", "week"]).drop_duplicates(["nname", "position"], keep="last")
       [["nname", "position", "player_id"]])
# yards-per-reception prior for FFA files that lack the rec column
last = wk[wk.season == wk.season.max()]
ypr_p = (last.groupby("player_id")[["receptions", "receiving_yards"]].sum().query("receptions >= 8")
         .eval("receiving_yards / receptions").rename("ypr"))
ypr_pos = last.groupby("position")[["receptions", "receiving_yards"]].sum().eval("receiving_yards / receptions")

# ---------- 3. current week + kickoffs ----------
try:
    events, rem = get(f"{API}/sports/americanfootball_nfl/events")
    print(f"  Odds API events: {len(events)} (credits left {rem})")
except Exception as e:
    print("  events fetch failed:", str(e)[:60]); events = json.load(open(
        os.path.join(ROOT, "data", "props_frames", "lines_cache.json"))).get("events", [])
ev = pd.DataFrame(events)
ev["kick"] = pd.to_datetime(ev.commence_time, utc=True)
ev["home"] = ev.home_team.map(NAME2ABBR); ev["away"] = ev.away_team.map(NAME2ABBR)
now = pd.Timestamp.now(tz="UTC")
# week boundaries: NFL weeks roll over Tuesday; label by nflv_game_lines when available
gl26 = gl[gl.season == SEASON]
if A.week: cur_week = A.week
else:
    # first week with any game not yet completed (kick + 4h > now)
    ev["wk_guess"] = ((ev.kick - ev.kick.min()).dt.days + 2) // 7 + 1
    live = ev[ev.kick + pd.Timedelta(hours=4) > now]
    cur_week = int(live.wk_guess.min()) if len(live) else int(ev.wk_guess.max())
    first_kick = ev.kick.min()
    if len(gl26):   # trust the lines table's week labels when it has them
        pass
ev["week"] = ((ev.kick - ev.kick.min()).dt.days + 2) // 7 + 1
sl = ev[ev.week == cur_week].copy()
if not A.all_games: sl = sl[sl.kick > now]
if A.hours: sl = sl[sl.kick <= now + pd.Timedelta(hours=A.hours)]
print(f"week {cur_week}: {len(sl)} games in this send"
      + ("" if not len(sl) else f" (kickoffs {sl.kick.min():%a %H:%M}Z .. {sl.kick.max():%a %H:%M}Z)"))
if not len(sl): raise SystemExit("nothing to send")
games = pd.concat([sl.assign(team=sl.home, opp=sl.away, is_home=1), sl.assign(team=sl.away, opp=sl.home, is_home=0)])
games["game"] = games.away_team + " @ " + games.home_team
games["kickoff_et"] = games.kick.dt.tz_convert("US/Eastern").dt.strftime("%a %m/%d %I:%M %p ET")
games = games[["team", "opp", "is_home", "home", "game", "kickoff_et", "kick", "id"]]

# ---------- 4. history frame (FFA baseline x actuals x lags x context) ----------
def context(s, w):
    g = gl[(gl.season == s) & (gl.week == w)].copy()
    g["team"] = g.team.map(lambda t: TEAM_FIX.get(t, t)); g["opp"] = g.opp.map(lambda t: TEAM_FIX.get(t, t))
    g["home"] = np.where(g.is_home == 1, g.team, g.opp)
    g["is_outdoor"] = (~g.home.isin(DOME)).astype(int)
    g["opp_implied"] = g.game_total - g.implied_team_total
    return g.rename(columns={"team_spread": "spread"})[["team", "opp", "spread", "game_total",
                                                      "implied_team_total", "is_outdoor", "opp_implied"]].drop_duplicates("team")

def frame_for(s, w, live=False):
    d = load_ffa(s, w)
    if d is None: return None
    if "rec" not in d.columns:      # estimate receptions from yards with a YPR prior
        d = d.merge(ids, on=["nname", "position"], how="left")
        d["ypr"] = d.player_id.map(ypr_p).fillna(d.position.map(ypr_pos)).fillna(10.0)
        d["rec"] = (d.rec_yds / d.ypr).where(d.rec_yds.notna())
        d = d.drop(columns=["player_id", "ypr"])
    ctx = context(s, w)
    d = d.merge(ctx, on="team", how="left")
    sk = d[d.position.isin(["QB", "RB", "WR", "TE"])].copy()
    sk["baseline_proj"] = score_ppr(sk)
    sk = sk.merge(ids, on=["nname", "position"], how="left")
    sk["player_id"] = sk.player_id.fillna("ffa_" + sk.id.astype(str))
    sk = sk.drop_duplicates(["player_id"])
    sk = sk.merge(lag, on=["player_id", "season", "week"], how="left")
    act = wk[(wk.season == s) & (wk.week == w)][["player_id", "fantasy_points_ppr"]]
    sk = sk.merge(act.rename(columns={"fantasy_points_ppr": "actual_ppr"}), on="player_id", how="left")
    sk["wind_kn"] = 0.0; sk["opponent_team"] = sk.opp
    sk["market_proj"] = np.nan
    if not live: sk = sk[sk.actual_ppr.notna()]
    k = d[d.position == "K"].copy(); k["proj"] = score_k(k)
    dst = d[d.position == "DST"].copy(); dst["proj"] = score_dst(dst, dst.opp_implied)
    return sk, k, dst

hist = []
for s, w in weeks_avail:
    if s >= SEASON: continue
    r = frame_for(s, w)
    if r is not None: hist.append(r[0])
hist = pd.concat(hist, ignore_index=True)
print(f"history: {len(hist):,} player-weeks with FFA baseline + actual, "
      f"{hist.groupby('season').week.nunique().to_dict()}")
print(f"  baseline MAE vs actual {np.abs(hist.actual_ppr - hist.baseline_proj).mean():.3f}")

cur = frame_for(SEASON, cur_week, live=True)
if cur is None: raise SystemExit(f"no FFA file for {SEASON} wk{cur_week} — drop raw_stats_{SEASON}_wk{cur_week}.csv in {FDIR}")
sk, kk, dst = cur
sk = sk[sk.team.isin(games.team)]; kk = kk[kk.team.isin(games.team)]; dst = dst[dst.team.isin(games.team)]
# K / D-ST override (Subvertadown-style paste parsed by scripts/parse_kdst_paste.py):
# blended with the FFA-scored value (--kdst-weight on the paste); FFA value kept in Baseline_FFA
ovr_p = os.path.join(ROOT, "data", "kdst", f"kdst_{SEASON}_wk{cur_week}.csv")
if os.path.exists(ovr_p):
    ovr = pd.read_csv(ovr_p)
    for pos, tbl in (("K", kk), ("DST", dst)):
        o = ovr[ovr.position == pos].set_index("team")
        tbl["baseline_ffa"] = tbl.proj
        hit = tbl.team.isin(o.index)
        w = A.kdst_weight
        tbl.loc[hit, "proj"] = w * tbl.loc[hit, "team"].map(o.proj) + (1 - w) * tbl.loc[hit, "proj"]
        if pos == "K":   # the override names the kicker actually expected to kick
            tbl.loc[hit, "player"] = tbl.loc[hit, "team"].map(o.player)
        print(f"  {pos} override: {int(hit.sum())}/{len(tbl)} teams from {os.path.basename(ovr_p)} (weight {w:.2f})"
              + ("" if hit.all() else f"; no override for {sorted(tbl.team[~hit])}"))
    kk = kk.drop_duplicates("team"); dst = dst.drop_duplicates("team")
else:
    print(f"  no K/DST override ({os.path.relpath(ovr_p, ROOT)}) — using FFA-scored K and DST")
print(f"{SEASON} wk{cur_week}: {len(sk)} skill rows on the slate, {len(kk)} K, {len(dst)} DST; "
      f"{sk.player_id.str.startswith('ffa_').sum()} without an NFL game log")

# ---------- 5. Model_Burke ----------
cols = [c for c in hist.columns if c in sk.columns]
full = pd.concat([hist[cols], sk[cols]], ignore_index=True)
ev_out, _ = pipeline.run(full, verbose=False)
# scorecard on the walk-forward history (what a reviewer asks for first)
for yr in sorted(ev_out.season.unique()):
    h = ev_out[(ev_out.season == yr) & ev_out.actual_ppr.notna() & ev_out.Model_Burke.notna()]
    if not len(h): continue
    mae = lambda c: np.abs(h.actual_ppr - h[c]).mean()
    wk_win = (h.assign(e_m=np.abs(h.actual_ppr - h.Model_Burke), e_b=np.abs(h.actual_ppr - h.baseline_proj))
              .groupby("week")[["e_m", "e_b"]].mean().eval("e_m < e_b").mean())
    cov = ((h.actual_ppr >= h.mb_p10) & (h.actual_ppr <= h.mb_p90)).mean()
    print(f"  {yr}: n={len(h):,}  MAE Model_Burke {mae('Model_Burke'):.3f} | baseline(FFA) {mae('baseline_proj'):.3f}"
          f" | control_k0 {mae('control_k0'):.3f}  · beats FFA in {wk_win:.0%} of weeks · 80% coverage {cov:.3f}")
p = ev_out[(ev_out.season == SEASON) & (ev_out.week == cur_week)].copy()
p = p.merge(sk[["player_id", "injury_status", "injury_details", "team", "opp"]], on="player_id",
            how="left", suffixes=("", "_sk"))
for c in ("team", "opp"):
    if c + "_sk" in p.columns: p[c] = p[c].fillna(p[c + "_sk"])
p["proj"] = p.Model_Burke_mean

# ---- spread conditioned on projection size ----
# The package's spread is position-wide, so a 1-point bench player inherits a starter's
# ±4 and gets negative floors. Replace it: for each player take the ~300 historical rows
# of the same position with the closest baseline, use their empirical residual
# (actual - baseline) quantiles, re-centre on the model mean, clip at the scoring floor.
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
_h = ev_out[ev_out.actual_ppr.notna()].copy()
_h["res"] = _h.actual_ppr - _h.baseline_proj
def local_quantiles(pos, base, mean, hist=_h, k=300):
    hp = hist[hist.position == pos]
    idx = np.argsort(np.abs(hp.baseline_proj.values - base))[:k]
    r = hp.res.values[idx]
    q = np.quantile(r, QS) - r.mean() + mean
    floor = -2.0 if (pos == "QB" and mean >= 5) else 0.0   # only a starting QB can go negative
    return np.maximum(q, floor)
def eval_spread(test_season):
    hist = _h[_h.season < test_season]; te = _h[(_h.season == test_season) & _h.Model_Burke_mean.notna()]
    qs = np.array([local_quantiles(r.position, r.baseline_proj, r.Model_Burke_mean, hist) for r in te.itertuples()])
    cov = ((te.actual_ppr.values >= qs[:, 0]) & (te.actual_ppr.values <= qs[:, 4])).mean()
    cov_pkg = ((te.actual_ppr >= te.mb_p10) & (te.actual_ppr <= te.mb_p90)).mean()
    pin = np.mean([np.mean(np.maximum(q * (te.actual_ppr.values - qs[:, i]), (q - 1) * (te.actual_ppr.values - qs[:, i]))) for i, q in enumerate(QS)])
    pin_pkg = np.mean([np.mean(np.maximum(q * (te.actual_ppr - te[f"mb_p{int(q*100)}"]), (q - 1) * (te.actual_ppr - te[f"mb_p{int(q*100)}"]))) for q in QS])
    print(f"  spread check {test_season} (n={len(te):,}): 80% coverage local {cov:.3f} vs package {cov_pkg:.3f}"
          f" · pinball local {pin:.3f} vs package {pin_pkg:.3f}")
eval_spread(2025)
qs = np.array([local_quantiles(r.position, r.baseline_proj, r.Model_Burke_mean) for r in p.itertuples()])
for i, q in enumerate(QS): p[f"mb_p{int(q*100)}"] = qs[:, i]
p["Model_Burke"] = p.mb_p50
p["sd"] = (p.mb_p75 - p.mb_p25) / 1.35

# ---------- 6. assemble CSV ----------
def rows(df, kind):
    o = df.merge(games[["team", "game", "kickoff_et", "kick"]], on="team", how="left")
    pid = (o.player_id.where(~o.player_id.astype(str).str.startswith("ffa_"), "") if "player_id" in o else "")
    out = pd.DataFrame({"Player": o.player, "ID": pid, "Pos": o.position, "Team": o.team, "Opp": o.opp,
                        "Game": o.game, "Kickoff": o.kickoff_et,
                        "Proj": o.proj.round(2),
                        "Median": (o.Model_Burke if kind == "skill" else o.proj).round(2),
                        "Floor_p10": (o.mb_p10 if kind == "skill" else (o.proj * 0.35).clip(lower=0)).round(2),
                        "p25": (o.mb_p25 if kind == "skill" else o.proj * 0.65).round(2),
                        "p75": (o.mb_p75 if kind == "skill" else o.proj * 1.35).round(2),
                        "Ceiling_p90": (o.mb_p90 if kind == "skill" else o.proj * 1.7).round(2),
                        "StdDev": (o.sd if kind == "skill" else o.proj * 0.5).round(2),
                        "Baseline_FFA": (o.baseline_proj if kind == "skill"
                                         else (o.baseline_ffa if "baseline_ffa" in o else o.proj)).round(2),
                        "Injury": o.injury_status.fillna("") if "injury_status" in o else "",
                        "_kick": o.kick})
    return out
out = pd.concat([rows(p, "skill"), rows(kk, "k"), rows(dst, "dst")], ignore_index=True)
out = out[out.Proj.notna() & (out.Proj > 0.3)].sort_values(["_kick", "Proj"], ascending=[True, False]).drop(columns="_kick")
stamp = pd.Timestamp.now().strftime("%m%d_%H%M")
path = A.out or os.path.join(OUTD, f"projections_{SEASON}_wk{cur_week}_{stamp}.csv")
out.to_csv(path, index=False)
ev_out.to_parquet(os.path.join(OUTD, f"run_{SEASON}_wk{cur_week}_{stamp}.parquet"))
print(f"\nwrote {os.path.relpath(path, ROOT)}: {len(out)} rows "
      f"({out.Pos.value_counts().to_dict()})")
print(out.head(12).to_string(index=False))
