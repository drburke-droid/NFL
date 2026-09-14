"""Grading scenarios: what the scorecard becomes under a different send per slate.

sabersim_grade.py answers one question — how did the sends SaberSim would have accepted
(generated >= 75 min before kickoff) actually score. This script answers the neighbouring
one: for each slate there may be several sends, and only one of them is counted. Which one
is counted changes the grade, sometimes a lot, because a corrected send that lands after the
cutoff carries the official inactives and the eligible one does not.

It emits docs/sabersim_scenarios.json: every slate, every send published for it, and each
send's projections next to the actual result. docs/sabersim_scoring.html lets you pick one
send per slate and recomputes MAE / RMSE / Spearman / coverage in the browser.

Selection, matching, scoring and the box-score gate mirror sabersim_grade.py exactly. As a
guard against the two drifting apart, the strict scenario computed here is compared against
docs/sabersim_accuracy.json and the result is written to strict_matches_published, which the
page shows. If that ever reads false, this script and the grader have diverged — trust the
grader and fix this one.

Usage:
  python scripts/sabersim_scenarios.py --sends pkg/sends outputs/sabersim
"""
import os, re, sys, glob, json, argparse
from datetime import datetime, date
from zoneinfo import ZoneInfo
import urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET = ZoneInfo("America/New_York")
ap = argparse.ArgumentParser()
ap.add_argument("--sends", nargs="+", default=[os.path.join(ROOT, "outputs", "sabersim")])
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--week1-tuesday", default="2026-09-08")
ap.add_argument("--min-lead", type=float, default=75.0, help="SaberSim's cutoff, in minutes before kickoff")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "sabersim_scenarios.json"))
ap.add_argument("--backup", choices=["sleeper", "none"], default="sleeper",
                help="second actuals source for games nflverse has not published yet")
A = ap.parse_args()
W1 = date.fromisoformat(A.week1_tuesday)

def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)

# ---------- 1. sends ----------
frames = []
for d in A.sends:
    for f in sorted(glob.glob(os.path.join(d, "Burke_Model_Burke_*.csv"))):
        try: x = pd.read_csv(f)
        except Exception: continue
        if "Generated" not in x.columns or "Kickoff" not in x.columns: continue
        x["send_file"] = os.path.basename(f); frames.append(x)
if not frames: raise SystemExit("no sends found")
s = pd.concat(frames, ignore_index=True).drop_duplicates(["send_file", "Game", "Player", "Pos"])
s["gen"] = pd.to_datetime(s.Generated.str.replace(" ET", "", regex=False), format="%Y-%m-%d %H:%M", errors="coerce").dt.tz_localize(ET)
def kick(row):
    m = re.match(r"^\w{3} (\d{2})/(\d{2}) (\d{2}):(\d{2}) (AM|PM) ET$", str(row.Kickoff))
    if not m or pd.isna(row.gen): return pd.NaT
    mo, dd, hh, mi, ap_ = int(m[1]), int(m[2]), int(m[3]) % 12, int(m[4]), m[5]
    yr = row.gen.year + (1 if (mo < row.gen.month - 6) else 0)
    return pd.Timestamp(yr, mo, dd, hh + (12 if ap_ == "PM" else 0), mi, tz=ET)
s["kick"] = s.apply(kick, axis=1)
s = s.dropna(subset=["gen", "kick"])
s["lead_min"] = (s.kick - s.gen).dt.total_seconds() / 60
s["week"] = ((s.kick.dt.tz_convert(ET).dt.date - W1).map(lambda t: t.days) // 7 + 1).astype(int)
# a slate is one first-kickoff time; every send for it covers the same games
s["slate"] = s.groupby("Game").kick.transform("min")
s["slate_key"] = s.kick.dt.strftime("%Y-%m-%dT%H:%M")

# ---------- 2. actuals (identical source and scoring to sabersim_grade.py) ----------
url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{A.season}.parquet"
a = pd.read_parquet(url); a = a[a.season_type == "REG"].copy()
a["nname"] = a.player_display_name.map(norm)
z = lambda c: a[c].fillna(0) if c in a.columns else 0
a["k_pts"] = 3*(z("fg_made_0_19")+z("fg_made_20_29")+z("fg_made_30_39")) + 4*z("fg_made_40_49") \
             + 5*(z("fg_made_50_59")+z("fg_made_60_")) + z("pat_made")
a["actual"] = np.where(a.position == "K", a.k_pts, a.fantasy_points_ppr)
act = a[["player_id", "nname", "position", "week", "actual"]].assign(
    team=a.team.values, last=a.player_display_name.map(lambda n: norm(n).split()[-1]))
by_id = {(r.player_id, int(r.week)): r.actual for r in act.itertuples() if isinstance(r.player_id, str)}
by_name = act.groupby(["nname", "position", "week"]).actual.sum().to_dict()
lt = act.groupby(["last", "team", "position", "week"]).actual.agg(["sum", "size"])
by_last = {k: v for k, v in lt["sum"].items() if lt.loc[k, "size"] == 1}
s["nname"] = s.Player.map(norm)
def lookup(r):
    w = int(r.week)
    if isinstance(r.ID, str) and (r.ID, w) in by_id: return by_id[(r.ID, w)]
    if (r.nname, r.Pos, w) in by_name: return by_name[(r.nname, r.Pos, w)]
    return by_last.get((r.nname.split()[-1], r.Team, r.Pos, w), np.nan)
s["actual"] = [lookup(r) for r in s.itertuples()]
s = s[s.Pos.isin(["QB", "RB", "WR", "TE", "K"])].copy()
# a game counts only once nflverse has BOTH teams for that week; a missing row is then a real 0
have = a.groupby("week").team.apply(set).to_dict()
s["box_ok"] = s.apply(lambda r: r.Team in have.get(int(r.week), set()) and r.Opp in have.get(int(r.week), set()), axis=1)
s.loc[s.box_ok, "actual"] = s.loc[s.box_ok, "actual"].fillna(0.0)

# ---------- 2b. backup actuals: Sleeper ----------
# nflverse publishes every asset in one batch (all 2026 files carried the same Last-Modified on
# 2026-09-13), so when it is behind there is no faster nflverse file to fall back to — only a
# different publisher. Sleeper is already a dependency of the generator, so no new vendor.
#
# Its raw counting stats are used, never its pts_ppr: the grade must stay on the target the model
# was trained on (4-pt pass TD, -2 INT, PPR, -2 fumble lost; DK scoring for K). Every Sleeper row
# that overlaps an nflverse row is compared, and the agreement is published so the formula can be
# audited rather than trusted.
def jget(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "model-burke-scenarios"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
def gv(d, *names):
    for n in names:
        v = d.get(n)
        if v is not None:
            try: return float(v)
            except (TypeError, ValueError): pass
    return 0.0
def sleeper_week(season, week):
    """{gsis_id: (actual_skill, actual_k, team)} for one week, scored the grader's way."""
    st = jget(f"https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}")
    pl = sleeper_week._players
    out = {}
    for sid, d in (st or {}).items():
        if not isinstance(d, dict): continue
        meta = pl.get(sid) or {}
        gsis = meta.get("gsis_id")
        if not gsis: continue
        skill = (0.04*gv(d, "pass_yd") + 4*gv(d, "pass_td") - 2*gv(d, "pass_int")
                 + 0.1*gv(d, "rush_yd") + 6*gv(d, "rush_td")
                 + 0.1*gv(d, "rec_yd") + 6*gv(d, "rec_td") + gv(d, "rec")
                 - 2*gv(d, "fum_lost")
                 + 2*(gv(d, "pass_2pt") + gv(d, "rush_2pt") + gv(d, "rec_2pt")))
        kpts = (3*(gv(d, "fgm_0_19") + gv(d, "fgm_20_29") + gv(d, "fgm_30_39"))
                + 4*gv(d, "fgm_40_49") + 5*gv(d, "fgm_50p", "fgm_50_59")
                + 5*gv(d, "fgm_60p") + gv(d, "xpm"))
        out[gsis] = (skill, kpts, meta.get("team"))
    return out

backup = {"used": False, "source": "sleeper", "weeks": [], "agreement": None, "error": None}
if A.backup == "sleeper" and (~s.box_ok).any():
    try:
        sleeper_week._players = jget("https://api.sleeper.app/v1/players/nfl", timeout=180)
        print(f"sleeper: {len(sleeper_week._players):,} players in the id map")
        sl, sl_team = {}, {}
        for wk in sorted(s.loc[~s.box_ok, "week"].unique()):
            try: w = sleeper_week(A.season, int(wk))
            except Exception as e:
                print(f"  week {wk}: stats fetch failed ({str(e)[:60]})"); continue
            teams = {t for (_, _, t) in w.values() if t}
            sl[int(wk)] = w; sl_team[int(wk)] = teams
            backup["weeks"].append({"week": int(wk), "players": len(w), "teams": len(teams)})
            print(f"  week {wk}: {len(w):,} players, {len(teams)} teams")

        def sl_actual(r):
            w = sl.get(int(r.week))
            if not w or not isinstance(r.ID, str): return np.nan
            hit = w.get(r.ID)
            if hit is None: return np.nan
            return hit[1] if r.Pos == "K" else hit[0]
        s["sl"] = [sl_actual(r) for r in s.itertuples()]

        # parity against nflverse wherever both published — evidence, not assumption
        both = s[s.box_ok & s.sl.notna() & s.actual.notna()]
        if len(both) >= 20:
            diff = (both.sl - both.actual).abs()
            backup["agreement"] = {"n": int(len(both)), "mean_abs_diff": round(float(diff.mean()), 4),
                                   "max_abs_diff": round(float(diff.max()), 3),
                                   "within_0_1": round(float((diff <= 0.1).mean()), 4)}
            print(f"  parity vs nflverse on {len(both)} shared rows: mean |diff| "
                  f"{diff.mean():.4f}, max {diff.max():.3f}, within 0.1 = {(diff <= 0.1).mean():.1%}")

        # fill only what nflverse is missing, and only when Sleeper has BOTH teams of the game
        fillable = s.apply(lambda r: (not r.box_ok)
                           and r.Team in sl_team.get(int(r.week), set())
                           and r.Opp in sl_team.get(int(r.week), set()), axis=1)
        if fillable.any():
            s.loc[fillable, "actual"] = s.loc[fillable, "sl"].fillna(0.0)
            s.loc[fillable, "box_ok"] = True
            s.loc[fillable, "provisional"] = True
            backup["used"] = True
            print(f"  filled {int(fillable.sum())} rows across "
                  f"{s.loc[fillable, 'Game'].nunique()} games nflverse has not published yet")
        else:
            print("  nothing to fill: Sleeper does not have both teams of any missing game either")
    except Exception as e:
        backup["error"] = str(e)[:200]; print("sleeper backup unavailable:", backup["error"])
if "provisional" not in s: s["provisional"] = False
s["provisional"] = s.provisional.fillna(False)

# ---------- 3. FFA / DK benchmarks ----------
bench = []
for d in A.sends:
    for f in glob.glob(os.path.join(d, "Burke_Model_Burke_*.parquet")):
        try: r = pd.read_parquet(f, columns=["player_id", "season", "week", "baseline_proj", "market_proj"])
        except Exception: continue
        r = r[r.season == A.season]; r["send_file"] = os.path.basename(f).replace(".parquet", ".csv"); bench.append(r)
if bench:
    b = pd.concat(bench).drop_duplicates(["send_file", "player_id"]).rename(
        columns={"player_id": "ID", "baseline_proj": "ffa", "market_proj": "dk"})
    s = s.merge(b[["send_file", "ID", "ffa", "dk"]], on=["send_file", "ID"], how="left")
else:
    s["ffa"] = np.nan; s["dk"] = np.nan

# ---------- 4. metrics (same definitions as sabersim_grade.py) ----------
from scipy.stats import spearmanr
def block(d):
    d = d.dropna(subset=["actual"])
    if d.empty: return None
    err = d.actual - d.Proj
    o = {"n": int(len(d)), "games": int(d.Game.nunique()),
         "mae": round(float(err.abs().mean()), 3), "rmse": round(float(np.sqrt((err**2).mean())), 3),
         "bias": round(float(err.mean()), 3),
         "spearman": round(float(spearmanr(d.Proj, d.actual)[0]), 3) if len(d) >= 8 else None,
         "cov80": round(float(((d.actual >= d.Floor_p10) & (d.actual <= d.Ceiling_p90)).mean()), 3),
         "median_mae": round(float((d.actual - d.Median).abs().mean()), 3)}
    dd = d.dropna(subset=["ffa"])
    if len(dd) >= 8:
        o["ffa_n"] = int(len(dd)); o["ffa_mae"] = round(float((dd.actual - dd.ffa).abs().mean()), 3)
        o["model_mae_on_ffa_rows"] = round(float((dd.actual - dd.Proj).abs().mean()), 3)
    return o

def strict_of(d):
    return block(d[d.lead_min >= A.min_lead].sort_values("gen")
                 .drop_duplicates(["Game", "ID", "Player", "Pos"], keep="last")) if len(d) else None
g = s[s.box_ok]
strict = strict_of(g)                               # what the page baselines against
# The drift guard has to compare like with like: sabersim_grade.py only ever sees nflverse, so a
# provisional row would make the badge go red for the wrong reason. Check on nflverse rows only.
strict_official = strict_of(g[~g.provisional]) if (~g.provisional).any() else None

published, matches = None, None
accp = os.path.join(ROOT, "docs", "sabersim_accuracy.json")
if strict_official and os.path.exists(accp):
    try:
        pub = json.load(open(accp)).get("overall") or {}
        keys = ["n", "mae", "rmse", "bias", "spearman", "cov80"]
        published = {k: pub.get(k) for k in keys}
        matches = all(pub.get(k) == strict_official.get(k) for k in keys if pub.get(k) is not None)
    except Exception as e:
        print("could not read the published grade:", str(e)[:80])

# ---------- 5. compact payload ----------
players = sorted(s.Player.unique()); pidx = {p: i for i, p in enumerate(players)}
POS = ["QB", "RB", "WR", "TE", "K"]
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)

slates = []
for key, d in s.groupby("slate_key"):
    games = sorted(d.Game.unique()); gidx = {gm: i for i, gm in enumerate(games)}
    # union of players across this slate's sends, ordered for stable indexing
    uni = d.drop_duplicates(["Game", "Player", "Pos"])[["Game", "Player", "Pos", "actual", "box_ok"]]
    uni = uni.sort_values(["Game", "Pos", "Player"]).reset_index(drop=True)
    keyof = lambda r: (r.Game, r.Player, r.Pos)
    order = {keyof(r): i for i, r in enumerate(uni.itertuples())}
    pl = [[pidx[r.Player], gidx[r.Game], POS.index(r.Pos) if r.Pos in POS else -1,
           (r3(r.actual) if r.box_ok else None)] for r in uni.itertuples()]
    sends = []
    for f, dd in sorted(d.groupby("send_file"), key=lambda t: t[1].gen.iloc[0]):
        v = [None] * len(uni)
        for r in dd.itertuples():
            i = order.get(keyof(r))
            if i is not None:
                v[i] = [r3(r.Proj), r3(r.Median), r3(r.Floor_p10), r3(r.Ceiling_p90), r3(r.ffa)]
        sends.append({"file": f, "short": f.replace("Burke_Model_Burke_", "").replace(".csv", ""),
                      "gen": dd.gen.iloc[0].strftime("%m/%d %H:%M ET"),
                      "lead": round(float(dd.lead_min.iloc[0])),
                      "out": int((dd.Status == "OUT").sum()) if "Status" in dd else 0,
                      "rows": int(len(dd)), "eligible": bool(dd.lead_min.iloc[0] >= A.min_lead), "v": v})
    slates.append({"key": key, "label": d.kick.iloc[0].strftime("%a %m/%d %-I:%M %p ET"),
                   "week": int(d.week.iloc[0]), "games": games,
                   "graded": bool(d.box_ok.any()), "provisional": bool(d.provisional.any()),
                   "source": ("sleeper" if d.provisional.any() else "nflverse") if d.box_ok.any() else None,
                   "pl": pl, "sends": sends})
slates.sort(key=lambda x: x["key"])

out = {"generated_at": datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%MZ"),
       "season": A.season, "min_lead_min": A.min_lead, "positions": POS, "players": players,
       "slates": slates, "strict": strict, "strict_official": strict_official,
       "published": published, "strict_matches_published": matches, "backup": backup}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(out, open(A.out, "w"), separators=(",", ":"))
kb = os.path.getsize(A.out) / 1024
gr = [x for x in slates if x["graded"]]
print(f"wrote {A.out} ({kb:.0f} KB): {len(slates)} slates ({len(gr)} graded), "
      f"{sum(len(x['sends']) for x in slates)} sends, {len(players)} players")
if strict: print(f"  strict: n={strict['n']} games={strict['games']} MAE={strict['mae']} "
                 f"rmse={strict['rmse']} rho={strict['spearman']}")
print(f"  matches the published grade: {matches} (checked on nflverse rows only)")
if backup["used"]:
    prov = [x["label"] for x in slates if x.get("provisional")]
    print(f"  PROVISIONAL from Sleeper: {', '.join(prov)} — nflverse will overwrite on its next batch")
if matches is False: print("  WARNING: diverged from sabersim_grade.py — trust the grader, fix this script")
