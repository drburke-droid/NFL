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
ap.add_argument("--all-remaining", action="store_true",
                help="every game not yet kicked off (default: only the NEXT slate = games within 90 min of the earliest upcoming kickoff)")
ap.add_argument("--out", default=None)
ap.add_argument("--analyst", default="Robert Burke")
ap.add_argument("--model-name", default="Model_Burke v1")
ap.add_argument("--no-market", action="store_true", help="skip the live DK lines/props pull")
ap.add_argument("--market-weight", type=float, default=0.6,
                help="weight on the DK-implied stat where a line exists; remainder on FFA")
ap.add_argument("--p-play-doubt", type=float, default=0.2,
                help="P(plays) for a Questionable player DK has not posted props for while teammates are priced "
                     "(2025 measured: 0.20 overall, 0.11 for FFA proj >= 8, n=54; Q WITH a DK line played 100%%)")
ap.add_argument("--no-lineups", action="store_true", help="skip the live ESPN/Sleeper status pull")
ap.add_argument("--history-start", type=int, default=2023,
                help="first FFA season used for Model_Burke training history (2016-22 files exist since 2026-09-10; "
                     "ffa_history_length_study: longer history changes 2025 MAE by <0.005, so the trial model stays on 2023+)")
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
elif not (A.all_games or A.all_remaining) and len(sl):
    sl = sl[sl.kick <= sl.kick.min() + pd.Timedelta(minutes=90)]     # next slate only
slate_tag = sl.kick.min().tz_convert("US/Eastern").strftime("%a%I%p").lower() if len(sl) else "none"
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
sk = sk[sk.team.isin(games.team)].copy(); kk = kk[kk.team.isin(games.team)].copy(); dst = dst[dst.team.isin(games.team)].copy()

# ---------- 4b. live Vegas: DK spreads/totals (all games) + DK player props (slate) ----------
# The FFA file is days old; the market reprices injuries and news within minutes. Stats
# with a DK line are blended (--market-weight on the market) BEFORE scoring, so the
# baseline the model corrects already reflects the news. Cache per event, 2h TTL.
MKT_CACHE = os.path.join(ROOT, "data", "props_frames", f"mkt_cache_{SEASON}_wk{cur_week}.json")
PROP_MKTS = "player_pass_yds,player_pass_tds,player_rush_yds,player_reception_yds,player_receptions,player_anytime_td"
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
            cache[eid] = {"_ts": now_ts, "rows": rows}; n_new += 1; time.sleep(0.2)
        except Exception as ex: print(f"  props fetch failed for {eid[:8]}:", str(ex)[:60])
    if n_new: print(f"  DK props: {n_new} events pulled (credits left {rem})")
    json.dump(cache, open(MKT_CACHE, "w"))
    return cache

def market_stats(cache, sl_games):
    """per player: DK-implied pass_yds, pass_tds, rush_yds, rec_yds, rec, exp_td."""
    ids_ = set(sl_games.id)
    rows = [dict(r, event=eid) for eid, c in cache.items()
            if not eid.startswith("_") and eid in ids_ for r in c.get("rows", [])]
    if not rows: return pd.DataFrame(columns=["nname"])
    r = pd.DataFrame(rows).dropna(subset=["player"]); r["nname"] = r.player.map(norm)
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
    return pd.DataFrame(out).reset_index().rename(columns={"index": "nname"})

if not A.no_market:
    cache = pull_market(games)
    lines = cache.get("_lines", {})
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
            for df_ in (sk, dst):
                m = df_[["team"]].merge(ctx, on="team", how="left")
                for c in ("spread", "game_total", "implied_team_total", "opp_implied"):
                    df_[c] = np.where(m[c].notna(), m[c], df_[c].values)
            dst["proj"] = score_dst(dst, dst.opp_implied)
            print(f"  game context refreshed for {len(ctx)} team rows from live DK lines")
    ms = market_stats(cache, games)
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
STATS = ["pass_yds", "pass_tds", "pass_int", "rush_yds", "rush_tds", "rec", "rec_yds", "rec_tds", "fumbles_lost"]
extra = [c for c in ("injury_status", "injury_details", "team", "opp", "ffa_ppr", "market_ppr", "no_line") + tuple(STATS) if c in sk.columns]
p = p.merge(sk[["player_id"] + extra], on="player_id", how="left", suffixes=("", "_sk"))
p["note0"] = ""
for c in ("team", "opp"):
    if c + "_sk" in p.columns: p[c] = p[c].fillna(p[c + "_sk"])
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
OUT_WORDS = {"out", "injured reserve", "ir", "suspension", "sus", "pup", "doubtful", "dnr", "nfi", "inactive"}
DEPTH = {}   # (team, pos) -> [(depth_chart_order, full_name, gsis_id, injury_status)] from Sleeper
def http_json(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
        return json.loads(r.read().decode())
def live_status():
    st = {}   # (nname, team) -> (status, source)
    try:
        for t in http_json("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries").get("injuries", []):
            tm = NAME2ABBR.get(t.get("displayName"), "")
            for i in t.get("injuries", []):
                nm = norm(i.get("athlete", {}).get("displayName", "")); sts = str(i.get("status", "")).lower()
                if sts in OUT_WORDS: st[(nm, tm)] = ("OUT", "espn")
                elif sts == "questionable" and (nm, tm) not in st: st[(nm, tm)] = ("Q", "espn")
    except Exception as e: print("  ESPN injuries unavailable:", str(e)[:50])
    try:
        for v in http_json("https://api.sleeper.app/v1/players/nfl").values():
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
            row["proj"] = row["baseline_proj"] = frac * V
            for c in STATS:
                if c in row.index and pd.notna(row[c]): row[c] = float(row[c]) * frac
            for c, val in (("status", ""), ("src", "sleeper-depth"), ("injury_status", np.nan), ("no_line", True), ("market_ppr", np.nan),
                           ("p_play", 1.0), ("play_mean", frac * V)):
                if c in row.index: row[c] = val
            row["note"] = f"depth-chart backup for {r.player} ({frac:.0%} of starter)"
            new_i = p.index.max() + 1; p.loc[new_i] = row
            print(f"    + synthesized {full} ({r.team} {r.position} depth {order}) at {frac*V:.1f}")
            return new_i
        return None
    outs = p[(p.status == "OUT") & (p.proj > 0.5)].sort_values("proj", ascending=False)
    print(f"  lineups: {len(st)} statuses pulled; {int((p.status == 'Q').sum())} Q, {len(outs)} OUT with a projection")
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
            p.loc[i, "note"] = (p.loc[i, "note"] + "; " if p.loc[i, "note"] else "") + f"+{g:.1f} with {r.player} out"
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
    pp = A.p_play_doubt
    for r in p[doubt].sort_values("proj", ascending=False).itertuples():
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
        if s_wr > 0:
            wrs = p[(p.team == r.team) & (p.position == "WR") & (p.status != "OUT")]
            if len(wrs) and wrs.proj.sum() > 0:
                for i, w_ in (wrs.proj / wrs.proj.sum()).items(): gain[i] = gain.get(i, 0.0) + s_wr * V * (1 - pp) * w_
        for i, g in list(gain.items()):
            if "market_ppr" in p.columns and pd.notna(p.loc[i, "market_ppr"]) and not A.no_market:
                g *= (1 - A.market_weight); gain[i] = g
            p.loc[i, "proj"] = float(p.loc[i, "proj"]) + g
            p.loc[i, "note"] = (p.loc[i, "note"] + "; " if p.loc[i, "note"] else "") + f"+{g:.1f} if {r.player} sits"
        p.loc[r.Index, ["status", "p_play", "play_mean"]] = ["DOUBT", pp, V]
        p.loc[r.Index, "proj"] = pp * V
        p.loc[r.Index, "note"] = f"Questionable, likely inactive ({V:.1f} if active)"
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
_pm = p.play_mean if "play_mean" in p.columns else p.Model_Burke_mean
_pp = p.p_play if "p_play" in p.columns else pd.Series(1.0, index=p.index)
qs = np.array([local_quantiles(r.position, r.baseline_proj, m) for r, m in zip(p.itertuples(), _pm)])
def mix_quantiles(qrow, pplay):
    """quantiles of  (1-pplay)*delta(0) + pplay*Playing  from the playing quantiles."""
    if pplay >= 1: return qrow
    out = []
    for tau in QS:
        if tau <= 1 - pplay: out.append(0.0)
        else: out.append(float(np.interp((tau - (1 - pplay)) / pplay, QS, qrow)))
    return np.array(out)
qs = np.array([mix_quantiles(q, pp_) for q, pp_ in zip(qs, _pp)])
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
for c in ("Median", "Floor_p10", "p25", "p75", "Ceiling_p90", "StdDev"):
    out.loc[out.Status == "OUT", c] = 0.0
out = out[out.Proj.notna() & ((out.Proj > 0.3) | (out.Status == "OUT"))].sort_values(["_kick", "Proj"], ascending=[True, False]).drop(columns="_kick")
# provenance columns (a comment line would break strict CSV readers)
out.insert(0, "Analyst", A.analyst)
out.insert(1, "Model", A.model_name)
out["Generated"] = pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d %H:%M ET")
stamp = pd.Timestamp.now().strftime("%m%d_%H%M")
tag = A.analyst.split()[-1] + "_" + A.model_name.split()[0]
path = A.out or os.path.join(OUTD, f"{tag}_{SEASON}_wk{cur_week}_{slate_tag}_{stamp}.csv")
out.to_csv(path, index=False)
ev_out.to_parquet(os.path.join(OUTD, f"run_{SEASON}_wk{cur_week}_{stamp}.parquet"))
print(f"\nwrote {os.path.relpath(path, ROOT)}: {len(out)} rows "
      f"({out.Pos.value_counts().to_dict()})")
print(out.head(12).to_string(index=False))
