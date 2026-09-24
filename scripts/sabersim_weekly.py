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
Default: current week, NEXT SLATE ONLY — the games within 90 min of the earliest
upcoming kickoff (one file per slate: Wed night, Thu night, Sun 1pm, Sun 4:25, SNF, MNF).
--kickoff ISO pins the slate explicitly, which is what the wrapper passes: a split late window
(week 2 has 16:05 and 16:25 ET) needs one file per kickoff, not one file for both.
--all-remaining = every game not yet kicked off; --hours H = games within H hours.
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
ap.add_argument("--kickoff", default=None,
                help="ISO kickoff of the slate to build; only games within --slate-tol of it. The wrapper "
                     "passes this so the generator cannot independently pick a different slate — on a split "
                     "late window (week 2: 16:05 and 16:25 ET) it would otherwise always build the earlier "
                     "one, because the earlier games have not kicked off yet.")
ap.add_argument("--slate-tol", type=float, default=15.0,
                help="minutes either side of --kickoff that count as the same slate")
ap.add_argument("--all-remaining", action="store_true",
                help="every game not yet kicked off (default: only the NEXT slate = games within 90 min of the earliest upcoming kickoff)")
ap.add_argument("--out", default=None)
ap.add_argument("--analyst", default="Robert Burke")
ap.add_argument("--model-name", default="Model_Burke v1")
ap.add_argument("--no-market", action="store_true", help="skip the live DK lines/props pull")
ap.add_argument("--spread", choices=["local", "gbm"], default="local",
                help="distribution layer: local = ~300 nearest-projection residual quantiles (per position); gbm = boosted "
                     "quantile regression on projection, volatility, role, context and injury (model league exp_001/exp_005: "
                     "tails scaled per position to 0.80 on the last history season). Tested 2026-09-15: on the generator's own "
                     "2022+ history the two TIE on 2025 (pinball 1.444 vs 1.443), so local stays the default; gbm is kept as an option")
ap.add_argument("--market-weight", type=float, default=0.6,
                help="weight on the DK-implied stat where a line exists; remainder on FFA")
ap.add_argument("--p-play-doubt", type=float, default=0.2,
                help="P(plays) for a Questionable player DK has not posted props for while teammates are priced "
                     "(2025 measured: 0.20 overall, 0.11 for FFA proj >= 8, n=54; Q WITH a DK line played 100%%)")
ap.add_argument("--no-bias-cal", action="store_true",
                help="skip the rolling bias correction read from docs/sabersim_accuracy.json "
                     "(see scripts/bias_correction.py)")
ap.add_argument("--bias-cal-weeks", type=int, default=4,
                help="completed weeks of graded history the bias correction averages over")
ap.add_argument("--no-lineups", action="store_true", help="skip the live ESPN/Sleeper status pull")
ap.add_argument("--no-report", action="store_true", help="skip the official NFL injury report (practice status)")
ap.add_argument("--no-dnp-haircut", action="store_true",
                help="skip the DNP haircut: a player whose last practice was DNP but who plays produces ~0.90 of a healthy line "
                     "(2017-25, 8/8 seasons; WR 0.85, RB 0.92, TE no shortfall; 0.78-0.86 even when DK has priced him)")
ap.add_argument("--history-start", type=int, default=2023,
                help="first FFA season used for Model_Burke training history (2016-22 files exist since 2026-09-10; "
                     "ffa_history_length_study: longer history changes 2025 MAE by <0.005, so the trial model stays on 2023+)")
ap.add_argument("--kdst-weight", type=float, default=0.65,
                help="weight on the pasted K/DST projection; remainder on the FFA-scored value")
A = ap.parse_args()
if not A.pkg or not os.path.isdir(os.path.join(A.pkg, "model_burke")):
    raise SystemExit("pass the Model_Burke package dir as argv[1] (the pkg/ folder from model_burke_pkg.zip, "
                     "i.e. the folder CONTAINING model_burke/), or set MODEL_BURKE_PKG")
sys.path.insert(0, A.pkg); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import name_match, bias_correction
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
KEYS = [k.strip() for k in open(_kp) if k.strip() and not k.startswith("#")] if os.path.exists(_kp) else []
OFFLINE = not KEYS
if OFFLINE:
    # No Odds API key (a cloud console or a fresh clone): the slate comes from data/schedule_<S>.csv
    # and the DK blend is skipped. Everything else (FFA, injury report, ESPN/Sleeper lineups, K/DST
    # override, the model) runs as usual, so a CSV can still be produced anywhere the package is.
    print("  no data/odds_api_key.txt — OFFLINE: slate from the schedule file, DK lines/props skipped (--no-market)")
    A.no_market = True
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
weeks_avail = sorted({(int(s), int(w)) for s, w in files if int(s) >= A.history_start or int(s) == SEASON})
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
    for url in (f"stats_player/stats_player_week_{yr}.parquet", f"player_stats/player_stats_{yr}.parquet", f"player_stats/stats_player_week_{yr}.parquet"):
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
ABBR2NAME = {v: k for k, v in NAME2ABBR.items()}
def schedule_events():
    """the season schedule as Odds-API-shaped events (id, commence_time, home_team, away_team), games not yet 4 h old"""
    sch = pd.read_csv(os.path.join(ROOT, "data", f"schedule_{SEASON}.csv"))
    sch = sch[(sch.season == SEASON) & (sch.game_type == "REG")]
    t = pd.to_datetime(sch.gameday.astype(str) + " " + sch.gametime.fillna("13:00").astype(str), errors="coerce")
    t = t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    keep = t.notna() & (t + pd.Timedelta(hours=4) > pd.Timestamp.now(tz="UTC"))
    return [{"id": r.game_id, "commence_time": k.strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": ABBR2NAME.get(r.home_team, r.home_team),
             "away_team": ABBR2NAME.get(r.away_team, r.away_team)} for r, k in zip(sch[keep].itertuples(), t[keep])]
events = None
if not OFFLINE:
    try:
        events, rem = get(f"{API}/sports/americanfootball_nfl/events")
        print(f"  Odds API events: {len(events)} (credits left {rem})")
    except Exception as e: print("  events fetch failed:", str(e)[:60])
if not events:
    events = schedule_events(); print(f"  slate from data/schedule_{SEASON}.csv: {len(events)} games still to play")
ev = pd.DataFrame(events)
ev["kick"] = pd.to_datetime(ev.commence_time, utc=True)
ev["home"] = ev.home_team.map(NAME2ABBR); ev["away"] = ev.away_team.map(NAME2ABBR)
now = pd.Timestamp.now(tz="UTC")
# Week labels come from the season schedule (data/schedule_{SEASON}.csv, nflverse). The Odds API only
# lists games that have not kicked off, so anchoring weeks on its earliest event made every run
# "week 1": correct through the opening week, then wrong forever after (caught 2026-09-16 on the
# first week-2 run — it loaded the week-1 FFA file and K/DST override). A week starts two days
# before its first kickoff (Tuesday) and is current until its last game is four hours old.
def schedule_weeks():
    fp = os.path.join(ROOT, "data", f"schedule_{SEASON}.csv")
    if not os.path.exists(fp): return {}
    sch = pd.read_csv(fp); sch = sch[(sch.season == SEASON) & (sch.game_type == "REG")]
    t = pd.to_datetime(sch.gameday.astype(str) + " " + sch.gametime.fillna("13:00").astype(str), errors="coerce")
    t = t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    g = pd.DataFrame({"week": sch.week.astype(int), "kick": t}).dropna().groupby("week").kick.agg(["min", "max"])
    return {int(w): (r["min"], r["max"]) for w, r in g.iterrows()}
SCHED = schedule_weeks()
def week_of(kick):
    """schedule week containing a kickoff (weeks open two days before their first game)"""
    ws = [w for w, (lo, hi) in SCHED.items() if kick >= lo - pd.Timedelta(days=2)]
    return max(ws) if ws else None
if SCHED:
    ev["week"] = ev.kick.map(week_of)
    if A.week: cur_week = A.week
    else:
        open_weeks = [w for w, (lo, hi) in SCHED.items() if hi + pd.Timedelta(hours=4) > now]
        cur_week = min(open_weeks) if open_weeks else max(SCHED)
    print(f"  schedule: {len(SCHED)} weeks; week {cur_week} runs {SCHED[cur_week][0]:%a %m/%d} .. {SCHED[cur_week][1]:%a %m/%d}")
else:
    print("  no data/schedule file: labelling weeks from the earliest listed event (only right in week 1)")
    ev["week"] = ((ev.kick - ev.kick.min()).dt.days + 2) // 7 + 1
    if A.week: cur_week = A.week
    else:
        live = ev[ev.kick + pd.Timedelta(hours=4) > now]
        cur_week = int(live.week.min()) if len(live) else int(ev.week.max())
sl = ev[ev.week == cur_week].copy()
if not A.all_games: sl = sl[sl.kick > now]
if A.kickoff:
    _k = pd.Timestamp(A.kickoff)
    if _k.tzinfo is None: _k = _k.tz_localize("UTC")
    _t = pd.Timedelta(minutes=A.slate_tol)
    sl = sl[(sl.kick >= _k - _t) & (sl.kick <= _k + _t)]
elif A.hours: sl = sl[sl.kick <= now + pd.Timedelta(hours=A.hours)]
elif not (A.all_games or A.all_remaining) and len(sl):
    sl = sl[sl.kick <= sl.kick.min() + pd.Timedelta(minutes=90)]     # next slate only
slate_tag = sl.kick.min().tz_convert("America/New_York").strftime("%a%I%p").lower() if len(sl) else "none"
print(f"week {cur_week}: {len(sl)} games in this send"
      + ("" if not len(sl) else f" (kickoffs {sl.kick.min():%a %H:%M}Z .. {sl.kick.max():%a %H:%M}Z)"))
if not len(sl): raise SystemExit("nothing to send")
games = pd.concat([sl.assign(team=sl.home, opp=sl.away, is_home=1), sl.assign(team=sl.away, opp=sl.home, is_home=0)])
games["game"] = games.away_team + " @ " + games.home_team
games["kickoff_et"] = games.kick.dt.tz_convert("America/New_York").dt.strftime("%a %m/%d %I:%M %p ET")
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
sk = sk[sk.team.isin(games.team)].copy(); kk = kk[kk.team.isin(games.team)].copy(); dst = dst[dst.team.isin(games.team)].copy()
# context() takes the opponent from the game-lines table, which can be missing a game the slate has
# (2026 wk2 carried 28 of 32 teams: no BUF/DET, no CIN/HOU). The skill frame gets a slate-based fill
# after the pipeline (section 5); K and DST go straight to the CSV, so fill them here — a blank Opp
# is what dropped both kickers from the wk2 grade and flagged the game as awaiting box scores.
_slate_opp = games.drop_duplicates("team").set_index("team").opp
for _t in (kk, dst):
    _t["opp"] = _t.opp.fillna(_t.team.map(_slate_opp)) if "opp" in _t.columns else _t.team.map(_slate_opp)
# HEALTH: what each data pull actually delivered for THIS slate, written beside the CSV as
# <csv>.health.json so the SaberSim page can show it. A feed that fails quietly must not look like a
# feed that had nothing to say.
_ffa_fp = os.path.join(FDIR, f"raw_stats_{SEASON}_wk{cur_week}.csv")
HEALTH = {"season": SEASON, "week": cur_week, "slate_games": int(games.id.nunique()),
          "games": sorted(set(games.game)), "market": "skipped (--no-market)" if A.no_market else "live",
          "ffa": {"file": os.path.basename(_ffa_fp), "rows": sum(1 for _ in open(_ffa_fp, encoding="utf-8")) - 1,
                  "age_hours": round((time.time() - os.path.getmtime(_ffa_fp)) / 3600, 1),
                  "slate_skill": int(len(sk)), "slate_k": int(len(kk)), "slate_dst": int(len(dst))}}

# ---------- 4b. live Vegas: DK spreads/totals (all games) + DK player props (slate) ----------
# The FFA file is days old; the market reprices injuries and news within minutes. Stats
# with a DK line are blended (--market-weight on the market) BEFORE scoring, so the
# baseline the model corrects already reflects the news. Cache per event, 2h TTL.
MKT_CACHE = os.path.join(ROOT, "data", "props_frames", f"mkt_cache_{SEASON}_wk{cur_week}.json")
PROP_MKTS = "player_pass_yds,player_pass_tds,player_rush_yds,player_reception_yds,player_receptions,player_anytime_td"
# Kicking markets go in their OWN request. The Odds API rejects a whole call when one market
# key is unsupported, and losing a slate's skill props to a kicker experiment is not a trade
# worth making. Availability varies by book and week, so a failure here is logged and ignored.
KICK_MKTS = "player_field_goals,player_kicking_points,player_pats"
def amer_prob(price):
    price = float(price); return 100 / (price + 100) if price > 0 else -price / (-price + 100)
def pull_market(sl_games):
    cache = json.load(open(MKT_CACHE)) if os.path.exists(MKT_CACHE) else {}
    now_ts = time.time(); rem = "?"
    if now_ts - cache.get("_lines_ts", 0) > 2 * 3600:
        try:
            j, rem = get(f"{API}/sports/americanfootball_nfl/odds?regions=us&bookmakers=draftkings"
                         f"&markets=spreads,totals&oddsFormat=american")
            lines = {}
            for e in j:
                d = {}
                for bk in e.get("bookmakers", []):
                    for m in bk.get("markets", []):
                        for o in m.get("outcomes", []):
                            if m["key"] == "totals" and o["name"] == "Over": d["total"] = o.get("point")
                            if m["key"] == "spreads" and o["name"] == e["home_team"]: d["home_spread"] = o.get("point")
                lines[e["id"]] = d
            cache["_lines"], cache["_lines_ts"] = lines, now_ts
            try:
                from odds_snapshots import record
                record([{"event": k, "market": m, "point": v.get("total" if m == "totals" else "home_spread")} for k, v in lines.items() for m in ("totals", "spreads")], "sabersim_send", SEASON, cur_week)
            except Exception as ex: print("  snapshot record failed:", str(ex)[:80])
            print(f"  DK game lines: {len(lines)} games (credits left {rem})")
        except Exception as ex: print("  DK game lines failed:", str(ex)[:60])
    n_new = 0
    kick_by = dict(zip(sl_games.id, sl_games.kick))
    for eid in sl_games.id.unique():
        c = cache.get(eid, {})
        hrs_to_kick = (kick_by[eid] - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600
        if c and (now_ts - c.get("_ts", 0) < 2 * 3600 or hrs_to_kick > 30): continue
        try:
            j, rem = get(f"{API}/sports/americanfootball_nfl/events/{eid}/odds?regions=us"
                         f"&bookmakers=draftkings&markets={PROP_MKTS}&oddsFormat=american")
            rows = []
            for bk in j.get("bookmakers", []):
                for m in bk.get("markets", []):
                    for o in m.get("outcomes", []):
                        rows.append({"market": m["key"], "player": o.get("description"), "side": o["name"],
                                     "point": o.get("point"), "price": o["price"]})
            krows = []
            try:
                jk, rem = get(f"{API}/sports/americanfootball_nfl/events/{eid}/odds?regions=us"
                              f"&bookmakers=draftkings&markets={KICK_MKTS}&oddsFormat=american")
                for bk in jk.get("bookmakers", []):
                    for m in bk.get("markets", []):
                        for o in m.get("outcomes", []):
                            krows.append({"market": m["key"], "player": o.get("description"), "side": o["name"],
                                          "point": o.get("point"), "price": o["price"]})
            except Exception as ex: print(f"  kicking props unavailable for {eid[:8]}:", str(ex)[:60])
            cache[eid] = {"_ts": now_ts, "rows": rows, "krows": krows}; n_new += 1; time.sleep(0.2)
            try:                           # append-only history of every line we ever pulled (odds_snapshots.py)
                from odds_snapshots import record
                record([dict(r, event=eid, commence=kick_by[eid].strftime("%Y-%m-%dT%H:%M:%SZ")) for r in rows + krows], "sabersim_send", SEASON, cur_week)
            except Exception as ex: print("  snapshot record failed:", str(ex)[:80])
        except Exception as ex: print(f"  props fetch failed for {eid[:8]}:", str(ex)[:60])
    if n_new: print(f"  DK props: {n_new} events pulled (credits left {rem})")
    if rem != "?": cache["_credits_left"] = rem
    json.dump(cache, open(MKT_CACHE, "w"))
    return cache

def kicker_signal(cache, sl_games, k_frame):
    """Which kickers DK has actually priced, per team.

    A book prices the kicker a team is expected to use and nobody else, which is the only free
    signal we have on a committee or a late change — the FFA file lists every kicker on the
    roster and the Subvertadown override sometimes names two (week 2 carried "Grupe / Sanders"
    for the Jets, which reached the CSV as a player SaberSim cannot match).

    Recorded, never blended. A book's "kicking points" is flat 3-per-field-goal scoring while
    ours is distance-weighted (3/4/5 by range), so the two numbers are not comparable and
    blending them would import a systematic error. Treat this as the Subvertadown and Fan Picks
    signals are treated: collect first, weigh only once there is evidence.
    """
    ids_ = set(sl_games.id)
    ev_teams = {}
    for g in sl_games.itertuples(): ev_teams.setdefault(g.id, set()).add(g.team)
    out = {}
    for eid, c in cache.items():
        if eid.startswith("_") or eid not in ids_: continue
        priced = {norm(r["player"]) for r in c.get("krows", []) if r.get("player")}
        if not priced: continue
        for team in ev_teams.get(eid, ()):
            hit = sorted({r.player for r in k_frame[k_frame.team == team].itertuples()
                          if norm(r.player) in priced})
            if hit: out[team] = hit
    return out


def market_stats(cache, sl_games, players):
    """per player: DK-implied pass_yds, pass_tds, rush_yds, rec_yds, rec, exp_td.

    Returns (frame, alias, ambiguous). DK spells some players differently from the FFA file
    (Kenny/Kenneth Gainwell, Joshua/Josh Palmer), so unmatched DK names are reconciled onto
    slate players first — see scripts/name_match.py for why that needs the game and the
    first name, not just the surname.
    """
    ids_ = set(sl_games.id)
    rows = [dict(r, event=eid) for eid, c in cache.items()
            if not eid.startswith("_") and eid in ids_ for r in c.get("rows", [])]
    if not rows: return pd.DataFrame(columns=["nname"]), {}, []
    r = pd.DataFrame(rows).dropna(subset=["player"]); r["nname"] = r.player.map(norm)
    ev_teams = {}
    for g in sl_games.itertuples(): ev_teams.setdefault(g.id, set()).add(g.team)
    alias, ambiguous = name_match.reconcile(
        list(r[["event", "nname"]].drop_duplicates().itertuples(index=False, name=None)),
        ev_teams, list(zip(players.nname, players.team)))
    # per row, not a global rename: the alias is keyed by (event, name) so one game's mapping
    # cannot reach another game's identically-spelled player (see scripts/name_match.py)
    if alias: r["nname"] = [alias.get((e, n), n) for e, n in zip(r.event, r.nname)]
    out = {}
    for mk, col in (("player_pass_yds", "mkt_pass_yds"), ("player_pass_tds", "mkt_pass_tds"),
                    ("player_rush_yds", "mkt_rush_yds"), ("player_reception_yds", "mkt_rec_yds"),
                    ("player_receptions", "mkt_rec")):
        x = r[(r.market == mk) & r.point.notna()]
        if not len(x): continue
        ov = x[x.side == "Over"].groupby("nname").agg(pt=("point", "first"), po=("price", "first"))
        un = x[x.side == "Under"].groupby("nname").price.first().rename("pu")
        x = ov.join(un, how="left")
        skew = x.apply(lambda q: (amer_prob(q.po) - amer_prob(q.pu)) if pd.notna(q.pu) else 0.0, axis=1)
        out[col] = x.pt + x.pt.abs() * 0.15 * skew   # -130/+100 skew (~.07) moves a 60-yd line ~0.6
    td = r[(r.market == "player_anytime_td") & (r.side == "Yes")].groupby("nname").price.first()
    if len(td): out["mkt_exp_td"] = td.map(lambda pr: -np.log(1 - min(amer_prob(pr), 0.95)))
    return pd.DataFrame(out).reset_index().rename(columns={"index": "nname"}), alias, ambiguous

if not A.no_market:
    cache = pull_market(games)
    lines = cache.get("_lines", {})
    _ids = list(games.id.unique())
    HEALTH["dk_lines"] = {"slate_games": len(_ids),
                          "with_lines": sum(1 for i in _ids if lines.get(i, {}).get("total") is not None and lines.get(i, {}).get("home_spread") is not None),
                          "fetched_at": (pd.Timestamp(cache["_lines_ts"], unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%MZ") if cache.get("_lines_ts") else None),
                          "credits_left": cache.get("_credits_left")}
    HEALTH["dk_props"] = {"slate_events": len(_ids), "events_with_props": sum(1 for i in _ids if cache.get(i, {}).get("rows")),
                          "with_props": 0, "by_stat": {}, "no_line": []}
    if lines:
        ctx = []
        for g in games.itertuples():
            L = lines.get(g.id, {})
            if L.get("total") is None or L.get("home_spread") is None: continue
            sp = L["home_spread"] if g.is_home else -L["home_spread"]
            itt = L["total"] / 2 - sp / 2
            ctx.append({"team": g.team, "spread": sp, "game_total": L["total"], "implied_team_total": itt,
                        "opp_implied": L["total"] - itt})
        ctx = pd.DataFrame(ctx)
        if len(ctx):
            # kk is in here for consistency, not for its own sake: score_k is pure FFA kicking
            # stats and the K CSV rows read no context column, so leaving it out changed nothing
            # today — but it is the same omission that left K/DST without an Opp.
            for df_ in (sk, kk, dst):
                m = df_[["team"]].merge(ctx, on="team", how="left")
                for c in ("spread", "game_total", "implied_team_total", "opp_implied"):
                    df_[c] = np.where(m[c].notna(), m[c], df_[c].values)
            dst["proj"] = score_dst(dst, dst.opp_implied)
            print(f"  game context refreshed for {len(ctx)} team rows from live DK lines (skill, K, DST)")
    ms, name_alias, name_amb = market_stats(cache, games, sk)
    if len(ms):
        w = A.market_weight
        # per-stat weights from outputs/reports/market_blend_weight.md (2023-25 closing lines vs FFA):
        # yardage lines beat FFA outright (best w 0.9-1.0); receptions FFA is better (best w 0.3);
        # TDs untested -> the generic --market-weight
        W_STAT = {"pass_yds": 0.9, "rush_yds": 0.9, "rec_yds": 0.9, "rec": 0.3, "pass_tds": w}
        for c in ("mkt_pass_yds", "mkt_pass_tds", "mkt_rush_yds", "mkt_rec_yds", "mkt_rec", "mkt_exp_td"):
            if c not in ms.columns: ms[c] = np.nan
        sk = sk.merge(ms, on="nname", how="left")
        blend = lambda ffa, mkt, ww: np.where(mkt.notna(), ww * mkt + (1 - ww) * ffa.fillna(0), ffa)
        for stat in ("pass_yds", "pass_tds", "rush_yds", "rec_yds", "rec"):
            sk[stat] = blend(sk[stat], sk[f"mkt_{stat}"], W_STAT[stat])
        ffa_td = sk.rush_tds.fillna(0) + sk.rec_tds.fillna(0)
        share_rush = (sk.rush_tds.fillna(0) / ffa_td.replace(0, np.nan)).fillna(
            pd.Series(np.where(sk.position == "RB", 0.8, 0.1), index=sk.index))
        new_td = np.where(sk.mkt_exp_td.notna(), w * sk.mkt_exp_td + (1 - w) * ffa_td, ffa_td)
        sk["rush_tds"], sk["rec_tds"] = new_td * share_rush, new_td * (1 - share_rush)
        sk["ffa_ppr"] = sk.baseline_proj
        sk["baseline_proj"] = score_ppr(sk)
        has = sk[["mkt_pass_yds", "mkt_pass_tds", "mkt_rush_yds", "mkt_rec_yds", "mkt_rec", "mkt_exp_td"]].notna().any(axis=1)
        mkt_only = sk.copy()
        for stat in ("pass_yds", "pass_tds", "rush_yds", "rec_yds", "rec"):
            mkt_only[stat] = mkt_only[f"mkt_{stat}"].fillna(mkt_only[stat])
        sk["market_ppr"] = np.where(has, score_ppr(mkt_only), np.nan)
        d_ = (sk.baseline_proj - sk.ffa_ppr)
        top = sk.loc[d_.abs().sort_values(ascending=False).index[:5]]
        print(f"  DK props blended (w={w}) for {int(has.sum())} slate players; mean shift {d_[has].mean():+.2f}; biggest: "
              + ", ".join(f"{r.player} {r.ffa_ppr:.1f}->{r.baseline_proj:.1f}" for r in top.itertuples()))
        sk["no_line"] = (~has) & sk.team.isin(sk.team[has].unique()) & (sk.ffa_ppr >= 8)
        if sk.no_line.any():
            print("  no DK props posted (news?):", ", ".join(sk.player[sk.no_line]))
        alias_show = sorted({f"{dk} -> {ffa}" for (_ev, dk), ffa in name_alias.items()})
        if alias_show:
            print("  DK name aliases: " + ", ".join(alias_show))
        if name_amb:
            print("  DK names left unmatched (more than one candidate):", ", ".join(sorted(name_amb)))
        # no_line keeps its ffa_ppr >= 8 gate on purpose — it also drives the DOUBT haircut, and
        # widening it there would put every unpriced bench player at risk of one. The count below
        # is the visibility fix: a sub-8 player losing the blend used to leave no trace at all.
        HEALTH["dk_props"].update({"with_props": int(has.sum()), "eligible": int((sk.ffa_ppr >= 8).sum()),
                                   "name_aliases": alias_show,
                                   "ambiguous_names": sorted(name_amb),
                                   "no_props_n": int(((~has) & sk.team.isin(sk.team[has].unique())).sum()),
                                   "by_stat": {c.replace("mkt_", ""): int(sk[c].notna().sum()) for c in ("mkt_pass_yds", "mkt_pass_tds", "mkt_rush_yds", "mkt_rec_yds", "mkt_rec", "mkt_exp_td")},
                                   "no_line": sorted(sk.player[sk.no_line].tolist())})
# K / D-ST override (Subvertadown-style paste parsed by scripts/parse_kdst_paste.py):
# blended with the FFA-scored value (--kdst-weight on the paste); FFA value kept in Baseline_FFA
# computed before the override collapses a team to one kicker row, so the alternatives are
# still visible; compared against what we actually send just below
DK_KICK = kicker_signal(cache, games, kk) if not A.no_market else {}
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
        HEALTH.setdefault("kdst", {"file": os.path.basename(ovr_p)})[pos] = {"hit": int(hit.sum()), "of": int(len(tbl))}
    kk = kk.drop_duplicates("team"); dst = dst.drop_duplicates("team")
else:
    print(f"  no K/DST override ({os.path.relpath(ovr_p, ROOT)}) — using FFA-scored K and DST")
    HEALTH["kdst"] = None
if DK_KICK:
    sending = dict(zip(kk.team, kk.player))
    off = {t: v for t, v in sorted(DK_KICK.items()) if sending.get(t) and sending[t] not in v}
    print(f"  DK priced a kicker for {len(DK_KICK)}/{kk.team.nunique()} slate teams"
          + ("" if not off else "; NOT the one we are sending: "
             + ", ".join(f"{t} sending {sending[t]}, DK prices {' / '.join(v)}" for t, v in off.items())))
    HEALTH["dk_kickers"] = {"priced_teams": len(DK_KICK),
                            "by_team": {t: v for t, v in sorted(DK_KICK.items())},
                            "not_the_one_we_send": {t: {"sending": sending[t], "dk_prices": v}
                                                    for t, v in off.items()}}
else:
    HEALTH["dk_kickers"] = None
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
STATS = ["pass_yds", "pass_tds", "pass_int", "rush_yds", "rush_tds", "rec", "rec_yds", "rec_tds", "fumbles_lost"]
extra = [c for c in ("injury_status", "injury_details", "team", "opp", "ffa_ppr", "market_ppr", "no_line") + tuple(STATS) if c in sk.columns]
p = p.merge(sk[["player_id"] + extra], on="player_id", how="left", suffixes=("", "_sk"))
p["note0"] = ""
for c in ("team", "opp"):
    if c + "_sk" in p.columns: p[c] = p[c].fillna(p[c + "_sk"])
if "opp" in p.columns:   # the lines table can lack a game (week not exported yet); the slate always knows the opponent
    p["opp"] = p.opp.fillna(p.team.map(games.drop_duplicates("team").set_index("team").opp))
p["proj"] = p.Model_Burke_mean

# ---- live lineup status: ESPN injuries + Sleeper, plus a manual inactives file ----
# FFA files are days old; official inactives post 90 min before kickoff (we send at 75).
# OUT players go to zero and their projected points are redistributed with the shares
# measured in scripts/context_effects_study.py (2012-25 starter absences, net of drift):
#   RB1 -> next RB 60%, other RBs 17%      (11% lost; pass rate does NOT change)
#   WR1 -> next WR 20%, other WRs 24%      (35% lost — WR1 production mostly evaporates)
#   TE1 -> next TE 36%, other TEs 4%, WRs 25% spread by projection (26% lost)
#   QB1 -> backup inherits 55% of the starter's number
SHARES = {"RB": (0.60, 0.17, 0.0), "WR": (0.20, 0.24, 0.0), "TE": (0.36, 0.04, 0.25), "QB": (0.55, 0.0, 0.0)}
# QB1 out -> his pass-catchers lose too (2011-25, position-group PPR in backup starts vs the same
# team-season's QB1 starts): WR 0.89-0.91, TE 0.91-0.95, RB 0.94-0.98 (carries unchanged, receiving down)
QB_OUT_HAIRCUT = {"WR": 0.10, "TE": 0.07, "RB": 0.04}
OUT_WORDS = {"out", "injured reserve", "ir", "suspension", "sus", "pup", "doubtful", "dnr", "nfi", "inactive"}
DEPTH = {}   # (team, pos) -> [(depth_chart_order, full_name, gsis_id, injury_status)] from Sleeper
def http_json(u, hdr=None, timeout=30):
    # Default to a browser UA (some feeds refuse the urllib default), but let callers override it.
    # ESPN wants the opposite: see espn_json.
    h = {"User-Agent": "Mozilla/5.0"} if hdr is None else hdr
    with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=timeout) as r:
        return json.loads(r.read().decode())
def espn_json(path):
    """ESPN 403s a browser User-Agent from a datacenter IP and serves the honest urllib one.

    Measured on a runner 2026-09-14 (scripts/espn_diag.py): site.api.espn.com returned 403 for
    "Mozilla/5.0", for a full Chrome UA, and for Chrome plus Accept/Referer, on both /injuries and
    /scoreboard — and 200 with 32 teams when no UA was set at all. So the blanket "Mozilla/5.0"
    above is what has been failing every ESPN call, not the runner's IP; no proxy is needed. The
    site.web.api host does not carry the rule and answers either way, so it is the fallback.
    """
    for host in ("site.api.espn.com", "site.web.api.espn.com"):
        try:
            return http_json(f"https://{host}{path}", hdr={})
        except Exception as e:
            last = e
    raise last
NICK2ABBR = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN",
             "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB", "Texans": "HOU", "Colts": "IND",
             "Jaguars": "JAX", "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA", "Vikings": "MIN",
             "Patriots": "NE", "Saints": "NO", "Giants": "NYG", "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
             "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS"}
PRAC_MAP = {"Did Not Participate In Practice": "DNP", "Limited Participation in Practice": "LP", "Full Participation in Practice": "FP"}
# P(plays) for a Questionable player the books have NOT priced, by last practice status (2023-25, teammates priced:
# DNP 18%, LP 30%; FP too few -> 0.45); --p-play-doubt when the practice status is unknown
PP_BY_PRAC = {"DNP": 0.18, "LP": 0.30, "FP": 0.45}
DNP_FAC = {"WR": 0.85, "QB": 0.88, "RB": 0.94, "TE": 1.0}     # playing-projection multiplier after a DNP (practice_report_study + the 2024-25 history check)
def official_report(season, week):
    """(nname, position) -> dict(team, desig, prac, injury, src) from the NFL's own injury report for this week.
    Live page first (validated to be THIS week's report), then the nflverse daily parquet, else empty."""
    out = {}
    try:
        html = urllib.request.urlopen(urllib.request.Request(f"https://www.nfl.com/injuries/league/{season}/REG{week}",
                                                             headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode("utf8", "replace")
        m = re.search(r"Injuries - WEEK (\d+)", html); sel = re.findall(r"<option[^>]*selected[^>]*>\s*WEEK (\d+)\s*</option>", html)
        if m and int(m.group(1)) == int(week) and (not sel or int(sel[0]) == int(week)):   # the page serves the latest week under any URL
            for chunk in html.split('d3-o-section-sub-title"><span>')[1:]:
                nick = chunk[:chunk.find("<")].strip(); team = NICK2ABBR.get(nick, nick)
                for row in re.findall(r"<tr>(.*?)</tr>", chunk.split("</table>")[0], re.S):
                    cells = [re.sub(r"<[^>]+>", " ", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
                    cells = [re.sub(r"\s+", " ", c).strip() for c in cells]
                    if len(cells) < 5 or cells[1] not in ("QB", "RB", "WR", "TE", "K"): continue
                    out[(norm(cells[0]), cells[1])] = {"team": team, "injury": cells[2], "prac": PRAC_MAP.get(cells[3], ""), "desig": cells[4], "src": "nfl.com"}
            if out: return out
        else: print(f"  NFL.com injury page is not week {week} yet (shows {m.group(1) if m else '?'})")
    except Exception as e: print("  NFL.com injury page unavailable:", str(e)[:60])
    try:
        cache = os.path.join(ROOT, "data", "nflverse_cache", f"injuries_{season}.parquet"); os.makedirs(os.path.dirname(cache), exist_ok=True)
        if not os.path.exists(cache) or time.time() - os.path.getmtime(cache) > 6 * 3600:
            urllib.request.urlretrieve(f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet", cache)
        d = pd.read_parquet(cache); d = d[(d.week == week) & d.position.isin(["QB", "RB", "WR", "TE", "K"])]
        for r in d.itertuples():
            out[(norm(r.full_name), r.position)] = {"team": TEAM_FIX.get(r.team, r.team), "injury": r.report_primary_injury or "", "prac": PRAC_MAP.get(r.practice_status, ""),
                                                    "desig": r.report_status or "", "src": "nflverse"}
    except Exception as e: print("  nflverse injuries unavailable:", str(e)[:60])
    return out
REPORT = {} if (A.no_report or A.no_lineups) else official_report(SEASON, cur_week)
if REPORT:
    _rc = pd.Series([v["prac"] for v in REPORT.values()]).value_counts().to_dict(); _rd = pd.Series([v["desig"] for v in REPORT.values() if v["desig"]]).value_counts().to_dict()
    print(f"  injury report ({next(iter(REPORT.values()))['src']}): {len(REPORT)} skill/K players listed; practice {_rc}; designations {_rd}")
    HEALTH["injury_report"] = {"src": next(iter(REPORT.values()))["src"], "listed": len(REPORT), "designations": _rd}
else:
    HEALTH["injury_report"] = None if not (A.no_report or A.no_lineups) else "skipped"
def sleeper_players():
    """Sleeper's player dump, asking the CDN for the origin copy rather than whatever it has cached.

    Measured on a runner 2026-09-16: the edge served this file with Age up to 555s — a copy nine
    minutes old — on a feed we are reading precisely because it carries lineup status. A ?t= query
    param reaches the origin (1530ms against 156ms cached, so it is a real fetch); a Cache-Control:
    no-cache header is ignored outright.

    Whether the origin actually carries statuses the cache lacks is still unmeasured — that needs a
    game day, and inactives_probe.py samples both on Sunday. Until then this is free upside, so it
    is taken, but NOT at the send's expense: a bypass hits the origin instead of the edge, and on an
    NFL Sunday that is the slower and less reliable path. Any failure falls straight back to the
    cached URL, which is exactly what the generator used before.
    """
    # Short timeout on the bypass on purpose. The whole run has to finish inside ~150s to clear the
    # T-75 cutoff, so a stalled origin must fail fast and hand back to the edge rather than burn a
    # fifth of the budget on the default 30s. 12s is ~8x the 1530ms measured on a quiet runner.
    try:
        return http_json(f"https://api.sleeper.app/v1/players/nfl?t={int(time.time())}", timeout=12)
    except Exception as e:
        print("  Sleeper origin fetch failed, falling back to the cached copy:", str(e)[:60])
        return http_json("https://api.sleeper.app/v1/players/nfl")
def live_status():
    st = {}   # (nname, team) -> (status, source)
    for (nm, pos), v in REPORT.items():     # the official designation outranks the aggregators: Out/Doubtful -> OUT, Questionable -> Q
        if v["desig"] in ("Out", "Doubtful"): st[(nm, v["team"])] = ("OUT", "nfl-report")
        elif v["desig"] == "Questionable": st[(nm, v["team"])] = ("Q", "nfl-report")
    try:
        for t in espn_json("/apis/site/v2/sports/football/nfl/injuries").get("injuries", []):
            tm = NAME2ABBR.get(t.get("displayName"), "")
            for i in t.get("injuries", []):
                nm = norm(i.get("athlete", {}).get("displayName", "")); sts = str(i.get("status", "")).lower()
                if sts in OUT_WORDS: st[(nm, tm)] = ("OUT", "espn")
                elif sts == "questionable" and (nm, tm) not in st: st[(nm, tm)] = ("Q", "espn")
    except Exception as e: print("  ESPN injuries unavailable:", str(e)[:50])
    try:
        for v in sleeper_players().values():
            if v.get("position") not in ("QB", "RB", "WR", "TE", "K"): continue
            nm, tm = norm(v.get("full_name", "")), TEAM_FIX.get(v.get("team") or "", v.get("team") or "")
            inj = str(v.get("injury_status") or "").lower()
            if v.get("depth_chart_order") and tm and v.get("status") == "Active":   # depth chart for synthesized backups
                DEPTH.setdefault((tm, v["position"]), []).append((int(v["depth_chart_order"]), v.get("full_name", ""), v.get("gsis_id") or "", inj))
            if inj in OUT_WORDS: st[(nm, tm)] = ("OUT", st.get((nm, tm), ("", ""))[1] + "+sleeper")
            elif inj == "questionable" and (nm, tm) not in st: st[(nm, tm)] = ("Q", "sleeper")
    except Exception as e: print("  Sleeper unavailable:", str(e)[:50])
    man = os.path.join(ROOT, "data", "inactives", f"inactives_{SEASON}_wk{cur_week}.txt")
    if os.path.exists(man):
        for line in open(man, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = [x.strip() for x in line.split(",")]
            nm, tm = parts[0], TEAM_FIX.get(parts[1], parts[1]) if len(parts) > 1 else ""
            flag = parts[2].lower() if len(parts) > 2 else "out"
            st[(norm(nm), tm)] = ("ACTIVE", "manual") if flag.startswith("act") else ("OUT", "manual")
    return st

p["nname"] = p.player.map(norm)
p["status"], p["note"] = "", p.note0
if not A.no_lineups:
    st = live_status()
    key = list(zip(p.nname, p.team))
    p["status"] = [st.get(k, ("", ""))[0] for k in key]
    p["src"] = [st.get(k, ("", ""))[1] for k in key]
    # a name-only match catches team-code mismatches, but only for OUT
    byname = {k[0]: v for k, v in st.items() if v[0] == "OUT"}
    miss = (p.status == "") & p.nname.isin(byname)
    p.loc[miss, "status"] = "OUT"; p.loc[miss, "src"] = p.loc[miss, "nname"].map(lambda n: byname[n][1] + "?team")
    # the FFA scrape's own tag (O / IR / SUS / PUP) is the last-resort OUT source: a player every
    # live feed missed (ESPN refusing the runner, Sleeper's cache) but whom the Wednesday scrape
    # already knew was gone must not keep a projection
    if "injury_status" in p.columns:
        ffa_out = (p.status == "") & p.injury_status.astype(str).str.strip().str.upper().isin(["O", "OUT", "IR", "SUS", "PUP", "NFI"])
        p.loc[ffa_out, "status"] = "OUT"; p.loc[ffa_out, "src"] = "ffa-tag"
    p["prac"] = [REPORT.get((n, ps), {}).get("prac", "") for n, ps in zip(p.nname, p.position)]
    p["inj_report"] = [REPORT.get((n, ps), {}).get("injury", "") for n, ps in zip(p.nname, p.position)]
    # a player whose last practice was DNP but who is expected to play produces ~0.88 of his line (WR 0.85) --
    # scripts/practice_report_study.py, 2017-25, 8/8 seasons, and still 0.78-0.86 when DK has priced him
    dnp = (p.prac == "DNP") & (p.status != "OUT") & (p.proj > 0.5) & p.position.map(DNP_FAC).fillna(1.0).lt(1.0) & (not A.no_dnp_haircut)
    if dnp.any():
        fac = p.loc[dnp, "position"].map(DNP_FAC).fillna(1.0).values
        p.loc[dnp, "proj"] = p.loc[dnp, "proj"] * fac
        p.loc[dnp, "note"] = [(n + "; " if n else "") + f"DNP {i or 'practice'}: x{f_:.2f}" for n, i, f_ in zip(p.loc[dnp, "note"], p.loc[dnp, "inj_report"], fac)]
        print(f"  DNP haircut on {int(dnp.sum())}: " + ", ".join(f"{r.player} ({r.position} x{DNP_FAC[r.position]:.2f}, {r.inj_report or '?'})" for r in p[dnp].sort_values("proj", ascending=False).head(8).itertuples()))
    def synth_backup(r, V, frac):
        """Starter r is out/doubtful and the FFA file has no other active QB for the team: add the
        Sleeper depth-chart backup as a new row cloned from the starter at frac x the starter's number
        (the same 0.8 share the existing rule gives a backup who IS in the file). Returns the new index."""
        have = set(p[(p.team == r.team) & (p.position == r.position)].nname)
        for order, full, gsis, inj in sorted(DEPTH.get((r.team, r.position), [])):
            nm = norm(full)
            if nm in have or nm == r.nname or inj in OUT_WORDS or st.get((nm, r.team), ("", ""))[0] == "OUT": continue
            row = p.loc[r.Index].copy()
            row["player"], row["nname"], row["player_id"] = full, nm, (gsis if gsis else "")
            # measured 2011-25 (starts vs the team QB1's per-game average): QB2 0.87 mean / 0.82 median,
            # QB3+ 0.79 / 0.74 -> a nominal QB2 gets 0.85 of the starter, deeper backups 0.78
            depth_frac = 0.85 if order <= 2 else 0.78
            frac = frac / 0.8 * depth_frac
            row["proj"] = row["baseline_proj"] = frac * V
            for c in STATS:
                if c in row.index and pd.notna(row[c]): row[c] = float(row[c]) * frac
            for c, val in (("status", ""), ("src", "sleeper-depth"), ("injury_status", np.nan), ("no_line", True), ("market_ppr", np.nan),
                           ("p_play", 1.0), ("play_mean", frac * V)):
                if c in row.index: row[c] = val
            row["note"] = f"depth-chart backup for {r.player} ({frac:.0%} of starter, depth {order})"
            new_i = p.index.max() + 1; p.loc[new_i] = row
            print(f"    + synthesized {full} ({r.team} {r.position} depth {order}) at {frac*V:.1f}")
            return new_i
        return None
    qb_top = p[p.position == "QB"].groupby("team").proj.max().to_dict()   # the team's projected starter, pre-zeroing
    outs = p[(p.status == "OUT") & (p.proj > 0.5)].sort_values("proj", ascending=False)
    _by_src = pd.Series([v[1] for v in st.values()]).str.replace("?team", "", regex=False).value_counts().to_dict()
    _out_src = outs.src.fillna("").str.replace("?team", "", regex=False).value_counts().to_dict()
    print(f"  lineups: {len(st)} statuses pulled {_by_src}; {int((p.status == 'Q').sum())} Q, {len(outs)} OUT with a projection {_out_src}")
    HEALTH["lineups"] = {"statuses": len(st), "by_src": _by_src, "q": int((p.status == "Q").sum()), "out": len(outs), "out_by_src": _out_src}
    for r in outs.itertuples():
        mates = p[(p.team == r.team) & (p.position == r.position) & (p.status != "OUT") & (p.index != r.Index)]
        mates = mates.sort_values("proj", ascending=False)
        V = float(r.proj); gain = {}
        s_top, s_rest, s_wr = SHARES[r.position]
        if len(mates):
            top = mates.index[0]; gain[top] = s_top * V
            rest = mates.iloc[1:]
            if len(rest) and rest.proj.sum() > 0:
                for i, w in (rest.proj / rest.proj.sum()).items(): gain[i] = s_rest * V * w
            if r.position == "QB":   # a backup QB inherits the role, not a share of it
                gain = {top: max(0.0, s_top * V + 0.25 * V - float(p.loc[top, "proj"]))}
        elif r.position == "QB":     # no other QB in the file: bring in the depth-chart backup
            synth_backup(r, V, s_top + 0.25)
        if r.position == "QB" and V >= 0.5 * qb_top.get(r.team, V):   # the STARTER is out: his pass-catchers lose a measured slice
            for i, m in p[(p.team == r.team) & p.position.isin(QB_OUT_HAIRCUT) & (p.status != "OUT")].iterrows():
                gain[i] = gain.get(i, 0.0) - QB_OUT_HAIRCUT[m.position] * float(m.proj)
        if s_wr > 0:   # TE1 out: a quarter of his points go to the WR room
            wrs = p[(p.team == r.team) & (p.position == "WR") & (p.status != "OUT")]
            if len(wrs) and wrs.proj.sum() > 0:
                for i, w in (wrs.proj / wrs.proj.sum()).items(): gain[i] = gain.get(i, 0.0) + s_wr * V * w
        for i, g in list(gain.items()):
            if "market_ppr" in p.columns and pd.notna(p.loc[i, "market_ppr"]) and not A.no_market:
                g *= (1 - A.market_weight)   # DK line already reflects the absence
                gain[i] = g
            old = float(p.loc[i, "proj"]); new = old + g
            p.loc[i, "proj"] = new
            p.loc[i, "note"] = (p.loc[i, "note"] + "; " if p.loc[i, "note"] else "") + f"{g:+.1f} with {r.player} out"
        p.loc[r.Index, "proj"] = 0.0
        p.loc[r.Index, "note"] = "OUT"
        print(f"    OUT {r.player:<22} {r.position} {r.team}  {V:5.1f} -> "
              + ", ".join(f"{p.loc[i,'player']} +{g:.1f}" for i, g in gain.items()))
    # ---- probable non-players: Questionable (any source) + DK posted no props for them
    # while pricing their teammates. FFA still carries a near-full number for these, and
    # a zero-line inactive is the single biggest avoidable miss. Treat as a mixture:
    # P(plays) = --p-play-doubt (0.2, measured 2025): Proj = p * playing mean, quantiles from the
    # mixture (median 0), and (1-p) of the points redistributed like an OUT.
    p["p_play"], p["play_mean"] = 1.0, p.proj
    q_any = (p.status == "Q") | p.injury_status.isin(["Q", "D", "Questionable", "Doubtful"])
    forced = p.status == "ACTIVE"
    doubt = q_any & p.no_line.fillna(False) & (p.status != "OUT") & ~forced & (p.proj > 0.5) if "no_line" in p.columns else pd.Series(False, index=p.index)
    for r in p[doubt].sort_values("proj", ascending=False).itertuples():
        pp = PP_BY_PRAC.get(getattr(r, "prac", ""), A.p_play_doubt) if REPORT else A.p_play_doubt
        V = float(r.proj)
        mates = p[(p.team == r.team) & (p.position == r.position) & (p.status != "OUT") & ~doubt & (p.index != r.Index)]
        mates = mates.sort_values("proj", ascending=False)
        gain = {}
        s_top, s_rest, s_wr = SHARES[r.position]
        if len(mates):
            top = mates.index[0]; gain[top] = s_top * V * (1 - pp)
            rest = mates.iloc[1:]
            if len(rest) and rest.proj.sum() > 0:
                for i, w_ in (rest.proj / rest.proj.sum()).items(): gain[i] = s_rest * V * (1 - pp) * w_
        elif r.position == "QB":     # doubtful starter, no other QB in the file: backup at (1-p) x 0.8
            synth_backup(r, V, (s_top + 0.25) * (1 - pp))
        if r.position == "QB" and V >= 0.5 * qb_top.get(r.team, V):
            for i, m in p[(p.team == r.team) & p.position.isin(QB_OUT_HAIRCUT) & (p.status != "OUT") & ~doubt].iterrows():
                gain[i] = gain.get(i, 0.0) - QB_OUT_HAIRCUT[m.position] * float(m.proj) * (1 - pp)
        if s_wr > 0:
            wrs = p[(p.team == r.team) & (p.position == "WR") & (p.status != "OUT")]
            if len(wrs) and wrs.proj.sum() > 0:
                for i, w_ in (wrs.proj / wrs.proj.sum()).items(): gain[i] = gain.get(i, 0.0) + s_wr * V * (1 - pp) * w_
        for i, g in list(gain.items()):
            if "market_ppr" in p.columns and pd.notna(p.loc[i, "market_ppr"]) and not A.no_market:
                g *= (1 - A.market_weight); gain[i] = g
            p.loc[i, "proj"] = float(p.loc[i, "proj"]) + g
            p.loc[i, "note"] = (p.loc[i, "note"] + "; " if p.loc[i, "note"] else "") + f"{g:+.1f} if {r.player} sits"
        p.loc[r.Index, ["status", "p_play", "play_mean"]] = ["DOUBT", pp, V]
        p.loc[r.Index, "proj"] = pp * V
        p.loc[r.Index, "note"] = f"Questionable{(' (' + r.prac + ')') if getattr(r, 'prac', '') else ''}, likely inactive: P(plays) {pp:.2f} ({V:.1f} if active)"
        print(f"    DOUBT {r.player:<20} {r.position} {r.team}  {V:5.1f} x{pp} -> {pp*V:4.1f}; "
              + ", ".join(f"{p.loc[i,'player']} +{g:.1f}" for i, g in gain.items()))
    p.loc[forced, "note"] = np.where(p.loc[forced, "note"] == "", "confirmed active (manual)", p.loc[forced, "note"])
    p["Model_Burke_mean"] = p.proj

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
# ---- boosted quantile layer (model league exp_001 / exp_005): the residual's SHAPE depends on projection
# level, recent volatility, role and context (stars and deep-ball WRs: lower median, longer right tail),
# which one projection-only neighbourhood cannot carry. Five LightGBM quantile regressors on the played
# history, target = actual - model mean; tail distances scaled per position so the last history season
# (held out from a first fit) covers 0.80; refit on everything; re-centred on the live model mean.
GBM_COLS = ["baseline_proj", "week", "spread", "game_total", "implied_team_total", "is_outdoor",
            "fantasy_points_ppr_l1", "fantasy_points_ppr_r3", "fantasy_points_ppr_r6", "fantasy_points_ppr_std3",
            "targets_r3", "targets_std3", "carries_r3", "carries_std3", "target_share_r3", "wopr_r3", "games_played", "fp_trend",
            "pass_yds", "rush_yds", "rec", "rec_yds", "pass_tds", "rush_tds", "rec_tds", "pass_yds_sd", "rush_yds_sd", "rec_yds_sd", "rec_sd"]
GBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
def _gbm_design(d, med):
    X = pd.DataFrame({c: (pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan) for c in GBM_COLS}, index=d.index).fillna(med).fillna(0.0)
    X["injury_q"] = d.injury_status.astype(str).str.upper().isin(["Q", "QUESTIONABLE"]).astype(float).values if "injury_status" in d.columns else 0.0
    for pos in ("QB", "RB", "WR", "TE"): X[f"pos_{pos}"] = (d.position == pos).astype(float).values
    return X
def gbm_quantiles(hist, rows, centre, seed=17):
    """rows x 5 quantiles of actual for `rows`, centred on `centre` (their model mean)."""
    import lightgbm as lgb
    h = hist[hist.Model_Burke_mean.notna() & hist.actual_ppr.notna()]
    med = _gbm_design(h, pd.Series(dtype=float)).median()
    y = (h.actual_ppr - h.Model_Burke_mean).values
    fit = lambda d, yy: [lgb.LGBMRegressor(objective="quantile", alpha=float(a), random_state=seed + i, **GBM_PARAMS).fit(_gbm_design(d, med), yy) for i, a in enumerate(QS)]
    predict = lambda ms, d: np.sort(np.column_stack([m.predict(_gbm_design(d, med)) for m in ms]), axis=1)
    scale = {pos: 1.0 for pos in ("QB", "RB", "WR", "TE")}
    last = int(h.season.max()); cal, h0 = h[h.season == last], h[h.season < last]
    if len(h0) >= 1000 and len(cal) >= 300:
        q0 = predict(fit(h0, (h0.actual_ppr - h0.Model_Burke_mean).values), cal); r0 = (cal.actual_ppr - cal.Model_Burke_mean).values
        for pos in scale:
            sel = (cal.position == pos).values
            if sel.sum() < 100: continue
            c50, lo, hi = q0[sel, 2], q0[sel, 0], q0[sel, 4]
            scale[pos] = float(min(np.arange(0.8, 1.6001, 0.05), key=lambda s: abs(np.mean((r0[sel] >= c50 + s * (lo - c50)) & (r0[sel] <= c50 + s * (hi - c50))) - 0.80)))
    q = predict(fit(h, y), rows)
    c50 = q[:, 2:3]; q = c50 + rows.position.map(scale).fillna(1.0).values[:, None] * (q - c50)
    q = np.sort(q, axis=1) + np.asarray(centre, dtype=float)[:, None]
    floor = np.where((rows.position == "QB").values & (np.asarray(centre) >= 5), -2.0, 0.0)[:, None]
    return np.maximum(q, floor), scale
def eval_spread(test_season):
    hist = _h[_h.season < test_season]; te = _h[(_h.season == test_season) & _h.Model_Burke_mean.notna()]
    if not len(hist) or not len(te): return
    pin = lambda qs: np.mean([np.mean(np.maximum(q * (te.actual_ppr.values - qs[:, i]), (q - 1) * (te.actual_ppr.values - qs[:, i]))) for i, q in enumerate(QS)])
    cov = lambda qs: ((te.actual_ppr.values >= qs[:, 0]) & (te.actual_ppr.values <= qs[:, 4])).mean()
    q_loc = np.array([local_quantiles(r.position, r.baseline_proj, r.Model_Burke_mean, hist) for r in te.itertuples()])
    q_pkg = te[[f"mb_p{int(q*100)}" for q in QS]].values
    line = f"  spread check {test_season} (n={len(te):,}): 80% coverage local {cov(q_loc):.3f} vs package {cov(q_pkg):.3f} · pinball local {pin(q_loc):.3f} vs package {pin(q_pkg):.3f}"
    try:
        q_gbm, sc = gbm_quantiles(hist, te, te.Model_Burke_mean.values)
        line += f" · GBM coverage {cov(q_gbm):.3f} pinball {pin(q_gbm):.3f} (tail scale {sc}) -> using {A.spread.upper()}"
    except Exception as ex: line += f" · GBM spread unavailable ({str(ex)[:60]})"
    print(line)
eval_spread(2025)
def eval_dnp(seasons=(2024, 2025)):
    """MAE on played DNP rows in the history, incumbent vs the haircut, from the nflverse report (the same source as live)."""
    try:
        rows = []
        for s in seasons:
            c = os.path.join(ROOT, "data", "nflverse_cache", f"injuries_{s}.parquet")
            if not os.path.exists(c): urllib.request.urlretrieve(f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{s}.parquet", c)
            rows.append(pd.read_parquet(c))
        inj = pd.concat(rows); inj = inj[inj.game_type == "REG"].drop_duplicates(["season", "week", "gsis_id"], keep="last")
        h = _h[_h.season.isin(seasons) & _h.Model_Burke.notna()].merge(inj[["season", "week", "gsis_id", "practice_status"]].rename(columns={"gsis_id": "player_id"}), on=["season", "week", "player_id"], how="inner")
        d = h[h.practice_status == "Did Not Participate In Practice"]
        if len(d) < 30: return
        fac = d.position.map(DNP_FAC).fillna(1.0).values
        print(f"  DNP check {seasons[0]}-{seasons[-1]} (n={len(d)} played DNP rows): MAE as-is {np.abs(d.actual_ppr - d.Model_Burke).mean():.3f} vs haircut {np.abs(d.actual_ppr - d.Model_Burke * fac).mean():.3f}"
              f" · median actual/model {(d.actual_ppr / d.Model_Burke.clip(lower=1)).median():.2f} · by pos " + ", ".join(f"{pos} {np.abs(g.actual_ppr - g.Model_Burke).mean():.2f}->{np.abs(g.actual_ppr - g.Model_Burke * DNP_FAC.get(pos, 1.0)).mean():.2f} (n={len(g)})" for pos, g in d.groupby("position")))
    except Exception as e: print("  DNP check skipped:", str(e)[:60])
eval_dnp()
# ---- rolling bias correction ----
# One constant for every skill row, read from the grade: the median error of the send's Median
# column over players who PLAYED in the last few weeks (scripts/bias_correction.py). It is a
# safety net for 2026-specific drift, not the main event. The first version (2026-09-17) read
# the all-rows mean error, which was -0.5 in wk1 while the played-only mean was +0.1: it was
# measuring the inactives we had left at 2-3 points, and it helped MAE only by accident, because
# a downward shift moved the mean point toward the median. Proj is now the median itself (below).
# The constant is applied to the playing mean BEFORE the quantiles are cut, so it moves Proj,
# Median, Mean and the band together and reorders nobody.
_bias, _bmeta = 0.0, {"applied": 0.0, "reason": "disabled"}
if not A.no_bias_cal:
    try:
        with open(os.path.join(ROOT, "docs", "sabersim_accuracy.json")) as _f: _acc = json.load(_f)
        _bias, _bmeta = bias_correction.recent_bias(_acc, season=SEASON, weeks=A.bias_cal_weeks)
    except Exception as _e:
        _bmeta = {"applied": 0.0, "reason": f"accuracy file unreadable: {str(_e)[:60]}"}
if _bias:
    # Only rows that still carry a projection. recent_bias can come back POSITIVE if the model
    # ever runs low, and adding that to a player deliberately zeroed for being OUT would put a
    # confirmed inactive back into the send with points against his name.
    _live = p.proj > 0
    p.loc[_live, "proj"] = (p.loc[_live, "proj"] + _bias).clip(lower=0)
    if "play_mean" in p.columns:
        p.loc[_live, "play_mean"] = (p.loc[_live, "play_mean"] + _bias).clip(lower=0)
    p["Model_Burke_mean"] = p.proj
HEALTH["bias_cal"] = _bmeta
print(f"  bias-cal: {_bias:+.3f} pts/player"
      + (f" from {_bmeta.get('rows', 0):,} graded rows over wk " + ",".join(str(w['week']) for w in _bmeta.get("weeks", [])) if _bias
         else f" (no correction: {_bmeta.get('reason', '')})"))

_pm = p.play_mean if "play_mean" in p.columns else p.Model_Burke_mean
_pp = p.p_play if "p_play" in p.columns else pd.Series(1.0, index=p.index)
qs = np.array([local_quantiles(r.position, r.baseline_proj, m) for r, m in zip(p.itertuples(), _pm)])
if A.spread == "gbm" and len(p):
    try: qs, _sc = gbm_quantiles(_h, p, np.asarray(_pm, dtype=float)); print(f"  spread: boosted quantiles, tail scale {_sc}")
    except Exception as ex: print(f"  spread: GBM failed ({str(ex)[:80]}) -> local quantiles")
def mix_quantiles(qrow, pplay):
    """quantiles of  (1-pplay)*delta(0) + pplay*Playing  from the playing quantiles."""
    if pplay >= 1: return qrow
    out = []
    for tau in QS:
        if tau <= 1 - pplay: out.append(0.0)
        else: out.append(float(np.interp((tau - (1 - pplay)) / pplay, QS, qrow)))
    return np.array(out)
qs_play = qs.copy()                    # quantiles GIVEN the player plays
qs = np.array([mix_quantiles(q, pp_) for q, pp_ in zip(qs, _pp)])
for i, q in enumerate(QS): p[f"mb_p{int(q*100)}"] = qs[:, i]
p["Model_Burke"] = p.mb_p50
p["sd"] = (p.mb_p75 - p.mb_p25) / 1.35
# ---- which point goes out as Proj (decided 2026-09-24) ----
# Proj = the median of the player's distribution given that he plays. Two reasons:
#  * The scoreboards we are ranked on (the sites' published accuracy, our own Accuracy page)
#    are MAE over players who appeared in the box score, and MAE is minimised by the median.
#    Misses are right-skewed, so the mean sits ~1.1-1.3 PPR above the median; on 2025 played
#    rows the mean point scored MAE 4.269 (WORSE than FFA 4.190) while the median point
#    scored 4.123 (better than FFA). Every 2026 send to date carried the mean.
#  * A reader building a projection off ours wants the typical outcome for a player who
#    suits up, plus the risk shown separately -- not the two multiplied together.
# The expected value (P(plays) x playing mean, the optimiser quantity) stays in the Mean
# column; Floor/Median/Ceiling keep the inactive mixture so the risk is visible; the Note
# carries P(plays). OUT rows stay at 0 everywhere.
p["mean_ev"] = p.proj
_out = p.status.eq("OUT") if "status" in p.columns else pd.Series(False, index=p.index)
p["proj"] = np.where(_out, 0.0, np.maximum(qs_play[:, 2], 0.0))
HEALTH["proj_kind"] = "playing_median"

# ---------- 6. assemble CSV ----------
def rows(df, kind):
    o = df.merge(games[["team", "game", "kickoff_et", "kick"]], on="team", how="left")
    pid = (o.player_id.where(~o.player_id.astype(str).str.startswith("ffa_"), "") if "player_id" in o else "")
    out = pd.DataFrame({"Player": o.player, "ID": pid, "Pos": o.position, "Team": o.team, "Opp": o.opp,
                        "Game": o.game, "Kickoff": o.kickoff_et,
                        "Proj": o.proj.round(2),
                        "Median": (o.Model_Burke if kind == "skill" else o.proj).round(2),
                        "Mean": (o.mean_ev if kind == "skill" else o.proj).round(2),
                        "Floor_p10": (o.mb_p10 if kind == "skill" else (o.proj * 0.35).clip(lower=0)).round(2),
                        "p25": (o.mb_p25 if kind == "skill" else o.proj * 0.65).round(2),
                        "p75": (o.mb_p75 if kind == "skill" else o.proj * 1.35).round(2),
                        "Ceiling_p90": (o.mb_p90 if kind == "skill" else o.proj * 1.7).round(2),
                        "StdDev": (o.sd if kind == "skill" else o.proj * 0.5).round(2),
                        "Injury": o.injury_status.fillna("") if "injury_status" in o else "",
                        "Status": o.status if "status" in o else "",
                        "Note": o.note if "note" in o else "",
                        "_kick": o.kick})
    # stat line, scaled so its PPR reconciles to Proj (the model's correction, lineup and
    # doubt adjustments all applied pro rata across the player's stats)
    if kind == "skill":
        base = o.baseline_proj.replace(0, np.nan)
        ratio = (o.proj / base).clip(0, 3).fillna(0)
        for c in STATS:
            out[c] = (o[c].fillna(0) * ratio).round(2) if c in o else 0.0
    elif kind == "k":
        r_ = (o.proj / o.baseline_ffa.replace(0, np.nan)).fillna(1) if "baseline_ffa" in o else 1.0
        out["fg_made"] = ((o.fg_0019.fillna(0) + o.fg_2029.fillna(0) + o.fg_3039.fillna(0) + o.fg_4049.fillna(0) + o.fg_50.fillna(0)) * r_).round(2)
        out["fg_40plus"] = ((o.fg_4049.fillna(0) + o.fg_50.fillna(0)) * r_).round(2)
        out["xp_made"] = (o.xp.fillna(0) * r_).round(2)
    elif kind == "dst":
        r_ = (o.proj / o.baseline_ffa.replace(0, np.nan)).fillna(1) if "baseline_ffa" in o else 1.0
        out["sacks"] = (o.dst_sacks.fillna(0) * r_).round(2); out["ints"] = (o.dst_int.fillna(0) * r_).round(2)
        out["def_td"] = (o.dst_td.fillna(0) * r_).round(2); out["pts_allowed_exp"] = o.opp_implied.round(1) if "opp_implied" in o else np.nan
    return out
out = pd.concat([rows(p, "skill"), rows(kk, "k"), rows(dst, "dst")], ignore_index=True)
STATCOLS = ["pass_yds", "pass_tds", "pass_int", "rush_yds", "rush_tds", "rec", "rec_yds", "rec_tds", "fumbles_lost",
            "fg_made", "fg_40plus", "xp_made", "sacks", "ints", "def_td", "pts_allowed_exp"]
for c in STATCOLS:
    if c not in out.columns: out[c] = np.nan
out = out[[c for c in out.columns if c not in STATCOLS + ["Generated"]] + STATCOLS + (["Generated"] if "Generated" in out.columns else [])]
for c in ("Median", "Mean", "Floor_p10", "p25", "p75", "Ceiling_p90", "StdDev"):
    out.loc[out.Status == "OUT", c] = 0.0
out = out[out.Proj.notna() & ((out.Proj > 0.3) | (out.Status == "OUT"))].sort_values(["_kick", "Proj"], ascending=[True, False]).drop(columns="_kick")
# provenance columns (a comment line would break strict CSV readers)
out.insert(0, "Analyst", A.analyst)
out.insert(1, "Model", A.model_name)
out["Generated"] = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M ET")
stamp = pd.Timestamp.now().strftime("%m%d_%H%M")
tag = A.analyst.split()[-1] + "_" + A.model_name.split()[0]
path = A.out or os.path.join(OUTD, f"{tag}_{SEASON}_wk{cur_week}_{slate_tag}_{stamp}.csv")
out.to_csv(path, index=False)
HEALTH.update({"csv": os.path.basename(path), "rows": int(len(out)), "generated": out["Generated"].iloc[0] if len(out) else None,
               "kickoffs": sorted(set(games.kickoff_et))})
try: json.dump(HEALTH, open(path.replace(".csv", ".health.json"), "w"), indent=1, default=str)
except Exception as _e: print("  health record not written:", str(_e)[:80])
# ---- DK columns for the graders, attached after the model has already run ----
# market_proj has been NaN in every send this season: it is initialised to NaN where sk is built
# and the real values are computed into market_ppr, a different name, which was never persisted.
# So the Accuracy page's DK column has always been blank, and no_line — the "books pulled his
# prop" flag that drives the DOUBT path — could not be audited after the fact either.
#
# This runs AFTER pipeline.run() has returned, so the model's feature surface is untouched. That
# matters: residual.auto_features() lists market_proj as a feature whenever the column exists, so
# populating it before training would silently change predictions. Here it cannot.
#
# Keyed on season and week as well as player_id — ev_out carries the 2023-25 walk-forward history,
# and a player-only merge would paste this week's line onto his historical rows.
try:
    _dk = [c for c in ("market_ppr", "no_line") if c in sk.columns]
    _key = [c for c in ("player_id", "season", "week") if c in sk.columns and c in ev_out.columns]
    if _dk and "player_id" in _key:
        ev_out = ev_out.merge(sk[_key + _dk].drop_duplicates(_key), on=_key, how="left", suffixes=("", "_sk"))
        if "market_ppr" in _dk:
            ev_out["market_proj"] = ev_out["market_proj"].fillna(ev_out["market_ppr"])
        print(f"  run frame: DK benchmark on {int(ev_out.market_proj.notna().sum())} rows"
              + (f", no_line set for {int(ev_out.no_line.fillna(False).sum())}" if "no_line" in ev_out else ""))
except Exception as _e:                      # never let a benchmark column cost the send
    print("  DK columns not attached:", str(_e)[:120])
ev_out.to_parquet(os.path.join(OUTD, f"run_{SEASON}_wk{cur_week}_{stamp}.parquet"))
print(f"\nwrote {os.path.relpath(path, ROOT)}: {len(out)} rows "
      f"({out.Pos.value_counts().to_dict()})")
print(out.head(12).to_string(index=False))
