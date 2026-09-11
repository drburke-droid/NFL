"""📡 PROPS WATCH — weekly live pull + Model_Burke evaluation of DK reception-yds lines.

The validated pocket (outputs/reports/model_burke_props.md): reception-yards OVERS
at >=5% calibrated edge (+14% ROI @8% in the 2025 backtest; early lines better than
late). This script runs the whole loop for the current week:

  1. downloads 2026 weekly box scores from nflverse (rolling-usage features)
  2. pulls DraftKings player_reception_yds + alternates for the week's games
     (~35 API credits of the free key's 500/month; key in data/odds_api_key.txt)
  3. rebuilds the model on 2023-25 closing-line history (data/props_frames/) plus
     graded 2026 weeks, predicts the current week walk-forward-style
  4. calibrated P(over) per line; flags 🚨 WEAK LINES (over edge >= 8%; 5-8% = watch_over, no bet)
  5. writes docs/props_watch_2026.js (+ outputs/draft_tool copy) for the 📡 Props
     tab, appends flagged bets to the paper-trade ledger, grades past weeks

Usage:  python scripts/props_watch.py <path-to-model_burke-pkg>
        (or set MODEL_BURKE_PKG; the pkg dir is the folder CONTAINING model_burke/)

Run it when lines post (Wed/Thu) — earlier is better per the v3 early-line result.
Week 1 note: no 2026 lags exist yet, so the correction leans on the 2023-25 bias
layers and will flag conservatively. Rookies with no NFL game log are skipped
until they appear in nflverse weekly data.
"""
import json, os, re, sys, sqlite3, time, urllib.error, urllib.request, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_BURKE_PKG", "")
if not PKG or not os.path.isdir(PKG):
    raise SystemExit("pass the model_burke package dir as argv[1] (folder containing model_burke/)")
sys.path.insert(0, PKG)
from model_burke import pipeline
from model_burke.features import build_lagged_features

# one key per line (gitignored); rotate to the next when one is out of credits
KEYS = [k.strip() for k in open(os.path.join(ROOT, "data", "odds_api_key.txt"))
        if k.strip() and not k.startswith("#")]
_ki = [0]
FRAMES = os.path.join(ROOT, "data", "props_frames")
LEDGER = os.path.join(FRAMES, "ledger_2026.csv")
WATCHROWS = os.path.join(FRAMES, "watch_2026_rows.parquet")
API = "https://api.the-odds-api.com/v4"
EDGE_FLAG = 0.08     # 🚨 + ledger: reception-yds OVERS at >= 8% calibrated edge (the only pocket that held up in 2025)
WATCH_FLAG = 0.05    # shown in the table as "watch_over", no alert, not bet
ALT_EV_FLAG = 0.08

def get(url):
    """GET with the current key; on quota/auth failure (401/402/403/429) or a
    key reporting 0 credits remaining, rotate to the next key and retry."""
    while True:
        key = KEYS[_ki[0]]
        sep = "&" if "?" in url else "?"
        try:
            with urllib.request.urlopen(f"{url}{sep}apiKey={key}", timeout=30) as r:
                rem = r.headers.get("x-requests-remaining")
                body = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403, 429) and _ki[0] + 1 < len(KEYS):
                print(f"  key #{_ki[0] + 1} rejected ({e.code}) — rotating to key #{_ki[0] + 2}")
                _ki[0] += 1; continue
            raise
        if rem is not None and float(rem) <= 0 and _ki[0] + 1 < len(KEYS):
            print(f"  key #{_ki[0] + 1} burned (0 credits left) — next call uses key #{_ki[0] + 2}")
            _ki[0] += 1
        return body, f"{rem} (key #{_ki[0] + 1} of {len(KEYS)})"

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)

def amer_imp(o):
    o = float(o); return 100 / (o + 100) if o > 0 else -o / (-o + 100)
def amer_profit(o):
    o = float(o); return o / 100 if o > 0 else 100 / -o

# ---------- 1. weekly box scores (2022-2026) ----------
print("— weekly stats —")
# 2022-2025 come from the local DB (nflv_weekly, refreshed by the normal data cascade);
# only the in-progress 2026 season is pulled from nflverse each week.
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
wk_hist = [pd.read_sql("""SELECT player_id, player_display_name, position, season, week,
                          team, opponent_team, targets, carries, receptions,
                          receiving_yards, rushing_yards, passing_yards, attempts,
                          target_share, wopr, air_yards_share, receiving_air_yards,
                          fantasy_points_ppr
                          FROM nflv_weekly WHERE season BETWEEN 2022 AND 2025
                          AND position IN ('QB','RB','WR','TE')""", con)]
con.close()
print(f"  2022-2025 from nfl_odds.db: {len(wk_hist[0]):,} rows")
for yr in (2026,):
    for url in (f"player_stats/player_stats_{yr}.parquet",
                f"player_stats/stats_player_week_{yr}.parquet"):
        try:
            w = pd.read_parquet(
                "https://github.com/nflverse/nflverse-data/releases/download/" + url)
            w = w[w.position.isin(["QB", "RB", "WR", "TE"])]
            keep = [c for c in wk_hist[0].columns if c in w.columns]
            wk_hist.append(w[keep])
            print(f"  {yr}: {len(w):,} rows (nflverse)")
            break
        except Exception as e:
            err = str(e)[:60]
    else:
        print(f"  {yr}: unavailable yet ({err}) — fine before week 1 completes")
wk = pd.concat(wk_hist, ignore_index=True)
if "season_type" in wk.columns:
    wk = wk[wk.season_type.isin(["REG", "REG "])] if wk.season_type.notna().any() else wk
wk = wk.drop_duplicates(["player_id", "season", "week"])
wk["nname"] = wk.player_display_name.map(norm)
lag = build_lagged_features(wk)
cur_week_2026 = int(wk[wk.season == 2026].week.max()) + 1 if (wk.season == 2026).any() else 1
print(f"  current 2026 week to predict: {cur_week_2026}")

# ---------- 2. DK lines for upcoming games ----------
print("— DraftKings lines —")
LINECACHE = os.path.join(FRAMES, "lines_cache.json")
cache_ok = (os.path.exists(LINECACHE)
            and time.time() - os.path.getmtime(LINECACHE) < 6 * 3600)
if cache_ok:
    cj = json.load(open(LINECACHE))
    rows, rem = cj["rows"], cj.get("rem", "?")
    print(f"  using cached lines from {pd.Timestamp(os.path.getmtime(LINECACHE), unit='s')}"
          f" (delete {os.path.relpath(LINECACHE, ROOT)} to force a fresh pull)")
events = []
if not cache_ok:
    events, rem = get(f"{API}/sports/americanfootball_nfl/events")
if not cache_ok:
    now = pd.Timestamp.utcnow().tz_localize(None)
    soon = [e for e in events
            if pd.Timestamp(e["commence_time"]).tz_localize(None) < now + pd.Timedelta(days=8)]
    print(f"  {len(soon)} games in the next 8 days")
    rows = []
    for e in soon:
        try:
            j, rem = get(f"{API}/sports/americanfootball_nfl/events/{e['id']}/odds"
                         f"?bookmakers=draftkings"
                         f"&markets=player_reception_yds,player_reception_yds_alternate"
                         f"&oddsFormat=american")
        except Exception as ex:
            print("  fetch fail", e["home_team"], str(ex)[:60]); continue
        for bk in j.get("bookmakers", []):
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({"event": f"{e['away_team']} @ {e['home_team']}",
                                 "commence": e["commence_time"],
                                 "player_name": o.get("description"),
                                 "side": o["name"], "point": o.get("point"),
                                 "price": o["price"], "market": m["key"]})
        time.sleep(0.2)
    json.dump({"rows": rows, "rem": rem}, open(LINECACHE, "w"))
    print(f"  API credits remaining: {rem}")
raw = pd.DataFrame(rows).dropna(subset=["player_name", "point"])
std = raw[raw.market == "player_reception_yds"]
alt = raw[raw.market == "player_reception_yds_alternate"]
line = std.pivot_table(index=["event", "commence", "player_name"], columns="side",
                       values=["point", "price"], aggfunc="first")
line.columns = [f"{a}_{b}".lower() for a, b in line.columns]
line = line.reset_index().rename(columns={"point_over": "line", "price_over": "over_price",
                                          "price_under": "under_price"})
line = line.dropna(subset=["line", "over_price", "under_price"])
print(f"  {len(line)} standard lines, {alt.player_name.nunique()} players with alt ladders")

# ---------- 3. model frame: history + current week ----------
hist = pd.read_parquet(os.path.join(FRAMES, "props_player_reception_yds.parquet"))
histcols = ["player_id", "player", "position", "season", "week",
            "actual_ppr", "baseline_proj", "over_price", "under_price"]
hist = hist[[c for c in hist.columns if c in histcols or c.endswith(("_l1", "_r3", "_r6", "_std3"))
             or c in ("games_played", "fp_trend")]]
old_watch = pd.read_parquet(WATCHROWS) if os.path.exists(WATCHROWS) else pd.DataFrame()

# weekly FFA (news-bearing consensus): data/ffanalytics/FFAn_weekly/projections_{S}_wk{W}.csv
# DISPLAY ONLY by decision: the 2023-25 weekly backfill (raw_stats_S_wkW.csv) was
# retro-tested in scripts/ffa_weekly_props_study.py — FFA-vs-line adds no calibrated
# edge on top of (z, line) (walk-forward logloss/Brier identical, pocket ROI slightly
# worse). See outputs/reports/ffa_weekly_props_study.md before re-litigating.
ffa_wk = None
fdir = os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly")
raw_p = os.path.join(fdir, f"raw_stats_2026_wk{cur_week_2026}.csv")
pts_p = os.path.join(fdir, f"projections_2026_wk{cur_week_2026}.csv")
if os.path.exists(raw_p):
    # stat-level consensus: rec_yds is a DIRECT second projection of the priced stat,
    # rec_yds_sd is analyst disagreement, injury_status is the news channel
    fw = pd.read_csv(raw_p, na_values=["NA"])
    fw = fw[fw.avg_type == "weighted"] if "avg_type" in fw.columns else fw
    fw["nname"] = fw.player.map(norm)
    fw = fw.drop_duplicates("nname")             # source carries dup rows per player
    ffa_wk = fw[["nname", "rec_yds", "rec_yds_sd", "injury_status"]].rename(
        columns={"rec_yds": "ffa_wk_yds", "rec_yds_sd": "ffa_wk_yds_sd",
                 "injury_status": "ffa_wk_inj"})
    print(f"  weekly FFA raw stats loaded: {len(ffa_wk)} players (wk{cur_week_2026})")
if os.path.exists(pts_p):
    fp = pd.read_csv(pts_p, na_values=["NA"])
    fp["nname"] = fp.player.map(norm)
    fp = fp[["nname", "points"]].rename(columns={"points": "ffa_wk_pts"}).drop_duplicates("nname")
    ffa_wk = fp if ffa_wk is None else ffa_wk.merge(fp, on="nname", how="outer")
if ffa_wk is None:
    print(f"  no weekly FFA files in {os.path.relpath(fdir, ROOT)} — drop them there when you have them")

cur = line.copy()
cur["nname"] = cur.player_name.map(norm)
ids = wk[["nname", "player_id", "player_display_name", "position"]].drop_duplicates("nname")
cur = cur.merge(ids, on="nname", how="left")
unmatched = cur[cur.player_id.isna()].player_name.tolist()
if unmatched:
    print("  no game log yet (skipped):", ", ".join(unmatched[:8]),
          "…" if len(unmatched) > 8 else "")
cur = cur.dropna(subset=["player_id"])
cur["season"], cur["week"] = 2026, cur_week_2026
cur = cur.rename(columns={"line": "baseline_proj", "player_display_name": "player"})
cur["actual_ppr"] = np.nan
cur = cur.merge(lag, on=["player_id", "season", "week"], how="left")

# grade + fold in earlier 2026 weeks
if len(old_watch):
    act = wk[wk.season == 2026][["player_id", "week", "receiving_yards"]]
    old_watch = old_watch.drop(columns=["actual_ppr"]).merge(
        act.rename(columns={"receiving_yards": "actual_ppr"}),
        on=["player_id", "week"], how="left")
if len(old_watch):   # the fresh pull supersedes any previously-saved rows for this week
    old_watch = old_watch[old_watch.week != cur_week_2026]
train = pd.concat([hist, old_watch], ignore_index=True) if len(old_watch) else hist
full = pd.concat([train, cur[[c for c in cur.columns if c in train.columns
                              or c.endswith(("_l1", "_r3", "_r6", "_std3"))]]],
                 ignore_index=True)
full["market_proj"] = np.nan
import model_burke.schema as _sch
ev, _ = pipeline.run(full[full.actual_ppr.notna() | (full.season == 2026)], verbose=False)

# calibrated P(over) — logistic on all graded rows, applied to the current week
ev["sd"] = ((ev.mb_p75 - ev.mb_p25) / 1.35).clip(lower=1e-3)
ev["z"] = ((ev.Model_Burke_mean - ev.baseline_proj) / ev.sd).clip(-4, 4)
g = ev.dropna(subset=["actual_ppr", "Model_Burke_mean", "z", "baseline_proj"])
g = g[g.actual_ppr != g.baseline_proj]
lr = LogisticRegression(C=1.0).fit(g[["z", "baseline_proj"]].values,
                                   (g.actual_ppr > g.baseline_proj).astype(int).values)
pred = ev[(ev.season == 2026) & (ev.week == cur_week_2026)].dropna(
    subset=["Model_Burke_mean", "z", "baseline_proj", "over_price", "under_price"]).copy()
pred = pred.merge(cur[["player_id", "event"]].drop_duplicates("player_id"),
                  on="player_id", how="left")
pred["event"] = pred.event.fillna("")
pred["nname"] = pred.player.map(norm)
if ffa_wk is not None:
    pred = pred.merge(ffa_wk, on="nname", how="left")
for c in ("ffa_wk_yds", "ffa_wk_yds_sd", "ffa_wk_inj", "ffa_wk_pts"):
    if c not in pred.columns:
        pred[c] = np.nan
pred["p_over"] = lr.predict_proba(pred[["z", "baseline_proj"]].values)[:, 1]

# empirical residuals for alternate rungs, CONDITIONED on line size: the outcome
# spread depends heavily on the player's volume tier (pooled absolute residuals gave
# Derrick Henry EV on 39.5+ rec yds; pooled relative residuals gave Chase 25% at
# 139.5+). Use the ~800 historical rows with the closest lines, their absolute
# residuals, re-centred so P(X>line) matches the calibrated p_over.
_g_lines = g.baseline_proj.values
_g_res = (g.actual_ppr - g.baseline_proj).values

def p_exceed(centre_line, p_over_std, alt_point):
    if not (0 < p_over_std < 1):
        return np.nan
    idx = np.argsort(np.abs(_g_lines - centre_line))[:800]
    res_loc = _g_res[idx]
    shift = np.quantile(res_loc, 1 - p_over_std)
    return float(np.mean(res_loc - shift > (alt_point - centre_line)))

# ---------- 4. score + flag ----------
out_rows, alerts = [], []
alt["nname"] = alt.player_name.map(norm)
for _, r in pred.iterrows():
    io, iu = amer_imp(r.over_price), amer_imp(r.under_price)
    fair = io / (io + iu)
    edge = r.p_over - fair
    flag = ("WEAK_OVER" if edge >= EDGE_FLAG else "watch_over" if edge >= WATCH_FLAG
            else "weak_under" if -edge >= EDGE_FLAG else "")   # unders are informational only (overs-only strategy)
    arows = []
    for _, a in alt[(alt.nname == norm(r.player)) & (alt.side == "Over")].iterrows():
        pex = p_exceed(r.baseline_proj, r.p_over, a.point)
        evu = pex * amer_profit(a.price) - (1 - pex)
        arows.append({"point": float(a.point), "price": int(a.price),
                      "p": round(pex, 3), "ev": round(evu, 3)})
        # flag only rungs the empirical tail can support: real probability mass,
        # not a deep-tail artifact, and within shouting distance of the line
        if (evu >= ALT_EV_FLAG and a.point > r.baseline_proj
                and pex >= 0.10 and a.point <= r.baseline_proj * 1.6 + 10):
            alerts.append(f"🚨 ALT  {r.player:<22} {a.point}+ @ {a.price:+d}  "
                          f"P={pex:.2f} EV={evu:+.2f}")
    if flag == "WEAK_OVER":
        alerts.append(f"🚨 LINE {r.player:<22} O{r.baseline_proj} @ {int(r.over_price):+d}  "
                      f"model P(over)={r.p_over:.2f} vs fair {fair:.2f} (edge {edge:+.2f})")
    out_rows.append({"player": r.player, "event": r.event, "line": float(r.baseline_proj),
                     "over": int(r.over_price), "under": int(r.under_price),
                     "mb": round(float(r.Model_Burke_mean), 1),
                     "p_over": round(float(r.p_over), 3), "fair": round(fair, 3),
                     "edge": round(float(edge), 3), "flag": flag,
                     "ffa_yds": (round(float(r.ffa_wk_yds), 1)
                                 if pd.notna(r.get("ffa_wk_yds", np.nan)) else None),
                     "ffa_sd": (round(float(r.ffa_wk_yds_sd), 1)
                                if pd.notna(r.get("ffa_wk_yds_sd", np.nan)) else None),
                     "inj": (str(r.ffa_wk_inj)
                             if pd.notna(r.get("ffa_wk_inj", np.nan)) else None),
                     "alts": sorted(arows, key=lambda x: x["point"])})

# ---------- 5. ledger ----------
led = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(
    columns=["placed", "week", "player", "bet", "line", "price", "p_model", "result", "profit"])
act26 = wk[wk.season == 2026][["nname", "week", "receiving_yards"]]
for i, r in led.iterrows():
    if pd.isna(r.result) or r.result == "":
        a = act26[(act26.nname == norm(r.player)) & (act26.week == r.week)]
        if len(a):
            y = a.receiving_yards.iloc[0]
            win = y > r.line
            led.loc[i, "result"] = "win" if win else ("push" if y == r.line else "loss")
            led.loc[i, "profit"] = 0.0 if y == r.line else (amer_profit(r.price) if win else -1.0)
new_bets = [o for o in out_rows if o["flag"] == "WEAK_OVER"]
for o in new_bets:
    dup = (led.week == cur_week_2026) & (led.player == o["player"]) & (led.line == o["line"])
    if not dup.any():
        led.loc[len(led)] = [pd.Timestamp.now().isoformat(timespec="minutes"), cur_week_2026,
                             o["player"], "Over", o["line"], o["over"], o["p_over"], "", np.nan]
led.to_csv(LEDGER, index=False)
graded = led[led.result.isin(["win", "loss"])]
lsum = {"bets": int(len(led)), "graded": int(len(graded)),
        "wins": int((graded.result == "win").sum()),
        "units": round(float(graded.profit.sum()), 2)}

# persist current-week rows for next run's grading/training (+ weekly FFA columns
# for the page; they are not model inputs — see ffa_weekly_props_study.md)
if ffa_wk is not None:
    cur = cur.merge(ffa_wk, on="nname", how="left")
keepcols = [c for c in cur.columns if c in hist.columns
            or c.startswith("ffa_wk_")]
allwatch = pd.concat([old_watch[keepcols] if len(old_watch) else pd.DataFrame(columns=keepcols),
                      cur[keepcols]], ignore_index=True).drop_duplicates(
    ["player_id", "season", "week"], keep="last")
allwatch.to_parquet(WATCHROWS)

# ---------- 6. bake the page data ----------
payload = {"generated": pd.Timestamp.now().isoformat(timespec="minutes"),
           "week": int(cur_week_2026), "credits_left": rem,
           "rows": sorted(out_rows, key=lambda x: -x["edge"]),
           "ledger": led.tail(60).fillna("").to_dict("records"),
           "ledger_summary": lsum}
js = ("// AUTO-GENERATED by scripts/props_watch.py — 📡 Props Watch data\n"
      "const PROPS_WATCH = " + json.dumps(payload, separators=(",", ":")) + ";\n")
for dd in ("docs", os.path.join("outputs", "draft_tool")):
    open(os.path.join(ROOT, dd, "props_watch_2026.js"), "w", encoding="utf-8").write(js)

print(f"\n{len(out_rows)} lines evaluated · {len(new_bets)} weak · ledger {lsum}")
print("\n".join(alerts) if alerts else "no weak lines at the current thresholds")
print("\nwrote docs/props_watch_2026.js — commit+push to update the 📡 Props tab")
