"""Grade Fan Picks against the box scores — shadow mode: this reads the recorded arrows and the
actuals and reports whether the fans were right; nothing here feeds the model.

Every recorded row (data/fan_adjustments/fan_adjustments_long.csv, from fan_adjust_record.py) is one
fan's arrow on one player-stat: the baseline the fan saw, the adjusted value the page showed, the
arrows themselves and the rule they were pressed under (scripts/fan_rules.py: "u1" = a set amount in
the stat's units per press; the original "pct10" = 10% of the baseline per press). The fan's number
is rebuilt from what we SENT with the fan's own presses under the fan's own rule, so a code keeps
meaning what it meant when it was locked in. Once nflverse has the game's box score:

  direction hit   the actual moved from the baseline in the arrow's direction (actual == baseline is
                  neutral and excluded from the rate)
  error removed   |actual - baseline| - |actual - adjusted|, in the stat's own units and in DK
                  fantasy points (yards x0.1 / x0.04 passing, TD x6 / x4 passing, rec x1, INT x1);
                  positive = the fan's number was closer than ours
  pending         the game has not been ingested by nflverse yet (both teams must have rows)

A player with no box-score row after his game is ingested scored 0 in every stat (inactive).

Outputs docs/fan/grade.json (what docs/fan.html renders: overall, leaderboard by fan, by stat, by
arrow size, the graded rows) and prints a summary. Refreshed by the daily grade in the send
workflow and by the grade workflow; run by hand any time.

Usage: python scripts/fan_grade.py [--season 2026] [--selftest]
"""
import os, re, sys, csv, json, glob, argparse
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fan_rules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET = ZoneInfo("America/New_York")
ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--sends", nargs="+", default=[os.path.join(ROOT, "outputs", "sabersim")],
                help="folders holding Burke_Model_Burke_*.csv — the numbers we actually sent")
ap.add_argument("--week1-tuesday", default="2026-09-08", help="Tuesday that starts week 1 (weeks roll on Tuesdays)")
ap.add_argument("--min-lead", type=float, default=75.0, help="minutes before kickoff a send must be generated to count")
ap.add_argument("--vs-bake", action="store_true",
                help="score against the frozen bake instead of the send (the old basis; see the note in sent_line)")
ap.add_argument("--selftest", action="store_true", help="grade a synthetic submission against synthetic actuals and print the checks")
A = ap.parse_args()
W1 = date.fromisoformat(A.week1_tuesday)
LONG = os.path.join(ROOT, "data", "fan_adjustments", "fan_adjustments_long.csv")
OUT = os.path.join(ROOT, "docs", "fan", "grade.json")
COL = {"pass_yds": "passing_yards", "pass_tds": "passing_tds", "pass_int": "passing_interceptions", "rush_yds": "rushing_yards",
       "rush_tds": "rushing_tds", "rec": "receptions", "rec_yds": "receiving_yards", "rec_tds": "receiving_tds"}
PTS = {"pass_yds": 0.04, "pass_tds": 4.0, "pass_int": 1.0, "rush_yds": 0.1, "rush_tds": 6.0, "rec": 1.0, "rec_yds": 0.1, "rec_tds": 6.0}
def norm(s): return re.sub(r"[^a-z]", "", str(s).lower().replace(" jr", "").replace(" sr", "").replace(" iii", "").replace(" ii", ""))

def sent_line(sends, min_lead):
    """(week, normalized name, position) -> the stat line we actually SENT, per stat key.

    Why not the bake. The fan page shows a file frozen days earlier, so grading a fan against it
    credits him with every point of error the news removed between bake and kickoff — reading the
    injury report scores as forecasting skill. Week 2 made that concrete: Clay faded Nico Collins to
    zero for +16.99 of his +17.23, and the send already had Collins at 0.00 because he was ruled OUT.
    The bake still said 17.18.

    Scoring against the send asks the question that actually decides anything: if we applied this
    fan's percentage nudge to the number we shipped, would it have helped? Same eligibility rule the
    model's own grader uses — the latest send generated at least min_lead minutes before kickoff.
    """
    frames = []
    for d in sends:
        for f in sorted(glob.glob(os.path.join(d, "Burke_Model_Burke_*.csv"))):
            try: x = pd.read_csv(f)
            except Exception: continue
            if "Generated" in x.columns and "Kickoff" in x.columns: frames.append(x)
    if not frames: return {}
    s = pd.concat(frames, ignore_index=True)
    s["gen"] = pd.to_datetime(s.Generated.str.replace(" ET", "", regex=False), format="%Y-%m-%d %H:%M",
                              errors="coerce").dt.tz_localize(ET)
    def kick(row):
        m = re.match(r"^\w{3} (\d{2})/(\d{2}) (\d{2}):(\d{2}) (AM|PM) ET$", str(row.Kickoff))
        if not m or pd.isna(row.gen): return pd.NaT
        mo, dd, hh, mi, ap_ = int(m[1]), int(m[2]), int(m[3]) % 12, int(m[4]), m[5]
        yr = row.gen.year + (1 if (mo < row.gen.month - 6) else 0)
        return pd.Timestamp(yr, mo, dd, hh + (12 if ap_ == "PM" else 0), mi, tz=ET)
    s["kick"] = s.apply(kick, axis=1)
    s = s.dropna(subset=["gen", "kick"])
    s = s[(s.kick - s.gen).dt.total_seconds() / 60 >= min_lead]
    if s.empty: return {}
    s["week"] = ((s.kick.dt.tz_convert(ET).dt.date - W1).map(lambda t: t.days) // 7 + 1).astype(int)
    s = s.sort_values("gen").drop_duplicates(["week", "Player", "Pos"], keep="last")
    out = {}
    for r in s.itertuples():
        key = (int(r.week), norm(r.Player), r.Pos)
        out[key] = {k: (float(getattr(r, k)) if pd.notna(getattr(r, k, np.nan)) else np.nan)
                    for k in COL if hasattr(r, k)}
    return out


def grade(rows, act):
    """rows: DataFrame of recorded arrows; act: nflverse weekly frame (player_id, player_display_name, position, team, week + stat cols).
    Returns the graded frame (one row per arrow) with actual / pending / err columns."""
    act = act.copy(); act["nname"] = act.player_display_name.map(norm)
    have = act.groupby("week").team.apply(set).to_dict()
    by_id = {(r.player_id, int(r.week)): r for r in act.itertuples()}
    by_nm = {(r.nname, r.position, int(r.week)): r for r in act.itertuples()}
    out = []
    for r in rows.itertuples():
        wk = int(r.week); stat = r.stat
        ingested = r.team in have.get(wk, set()) and r.opp in have.get(wk, set())
        a = by_id.get((r.player_id, wk)) if isinstance(r.player_id, str) and r.player_id else None
        if a is None: a = by_nm.get((norm(r.player), r.pos, wk))
        actual = float(getattr(a, COL[stat]) or 0.0) if a is not None else (0.0 if ingested else np.nan)
        n = int(r.arrows)
        if A.vs_bake:
            base, basis = float(r.baseline), "bake"
        else:                                  # the fan's percentage, applied to what we shipped
            sl = SENT.get((wk, norm(r.player), r.pos))
            sb = sl.get(stat, np.nan) if sl else np.nan
            base, basis = (float(sb), "send") if pd.notna(sb) else (np.nan, "no_send")
        rule = fan_rules.rule_of(getattr(r, "rule", None))       # rows recorded before rules existed = pct10
        adj = fan_rules.adjusted(rule, stat, base, n) if pd.notna(base) else np.nan
        d = dict(r._asdict()); d.pop("Index", None)
        d.update({"basis": basis, "base": None if pd.isna(base) else round(base, 3),
                  "adj": None if pd.isna(adj) else round(adj, 3), "no_send": basis == "no_send"})
        # every column exists on every row, graded or not: with all rows pending (a submission
        # in before its games kick off) these were absent entirely and summarise() died on
        # gd.direction, so a fan who submitted early broke the whole grade until his games ran
        d.update({"pending": not ingested, "actual": None if not ingested else round(actual, 2),
                  "direction": None, "err_base": np.nan, "err_adj": np.nan,
                  "removed": np.nan, "removed_pts": np.nan})
        if ingested and basis != "no_send":
            move = actual - base
            d["direction"] = "neutral" if abs(move) < 1e-9 else ("hit" if np.sign(move) == np.sign(n) else "miss")
            d["err_base"] = round(abs(actual - base), 3); d["err_adj"] = round(abs(actual - adj), 3)
            d["removed"] = round(d["err_base"] - d["err_adj"], 3)
            d["removed_pts"] = round(d["removed"] * PTS[stat], 3)
        out.append(d)
    return pd.DataFrame(out)

def summarise(g):
    def block(d):
        gd = d[~d.pending & ~d.no_send]
        dec = gd[gd.direction != "neutral"]
        return {"n": int(len(d)), "graded": int(len(gd)), "pending": int((d.pending & ~d.no_send).sum()),
                "no_send": int(d.no_send.sum()),
                "hit_rate": round(float((dec.direction == "hit").mean()), 3) if len(dec) else None, "hits": int((dec.direction == "hit").sum()), "misses": int((dec.direction == "miss").sum()),
                "closer": int((gd.removed > 0).sum()) if len(gd) else 0, "farther": int((gd.removed < 0).sum()) if len(gd) else 0,
                "removed_pts": round(float(gd.removed_pts.sum()), 2) if len(gd) else None,
                "mae_base_pts": round(float((gd.err_base * gd.stat.map(PTS)).mean()), 3) if len(gd) else None,
                "mae_adj_pts": round(float((gd.err_adj * gd.stat.map(PTS)).mean()), 3) if len(gd) else None}
    s = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "overall": block(g)}
    s["by_fan"] = sorted([dict(fan=f, fan_id=(str(d.fan_id.iloc[0]) if "fan_id" in d.columns else ""),
                               weeks=sorted(int(w) for w in d.week.unique()), **block(d)) for f, d in g.groupby("fan")],
                         key=lambda x: (-(x["removed_pts"] or 0), -x["graded"], x["fan"]))
    for rank, f in enumerate(s["by_fan"], 1):          # the leaderboard: position and best call
        f["rank"] = rank
        d = g[(g.fan == f["fan"]) & ~g.pending & ~g.no_send]
        if len(d):
            b = d.loc[d.removed_pts.idxmax()]
            f["best_call"] = {"week": int(b.week), "player": b.player, "stat": b.stat, "arrows": int(b.arrows),
                              "removed_pts": round(float(b.removed_pts), 2)}
    # weekly standings, so a fan who joins in week 6 has a race to win that week
    s["by_fan_week"] = sorted([dict(fan=f, fan_id=(str(d.fan_id.iloc[0]) if "fan_id" in d.columns else ""), week=int(w), **block(d))
                               for (f, w), d in g.groupby(["fan", "week"])],
                              key=lambda x: (-x["week"], -(x["removed_pts"] or 0), -x["graded"], x["fan"]))
    s["by_stat"] = {st: block(d) for st, d in g.groupby("stat")}
    g["size"] = g.arrows.abs().clip(upper=4).map(lambda k: {1: "1", 2: "2", 3: "3", 4: "4+"}[int(k)])
    s["by_size"] = {k: block(d) for k, d in g.groupby("size")}
    s["by_direction"] = {k: block(d) for k, d in g.groupby(g.arrows.map(lambda n: "boost" if n > 0 else "fade"))}
    s["by_week"] = {int(w): block(d) for w, d in g.groupby("week")}
    gd = g[~g.pending & ~g.no_send].sort_values(["week", "fan", "removed_pts"], ascending=[False, True, False])
    s["rows"] = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items() if k in ("week", "fan", "player", "team", "pos", "stat", "arrows", "baseline", "base", "adj", "basis", "actual", "direction", "removed_pts")}
                 for r in gd.head(400).to_dict("records")]
    pend = g[g.pending & ~g.no_send]
    s["pending_games"] = sorted({f"wk{int(r.week)} {r.team} v {r.opp}" for r in pend.itertuples()})
    return s

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    return x

if A.selftest:
    rows = pd.DataFrame([
        dict(season=2026, week=1, fan="A", submitted_at="t", bake_id="b", comment="", player_index=0, player_id="00-1", player="X One", team="BUF", pos="WR", opp="DET", stat="rec_yds", arrows=2, baseline=50.0, adjusted=60.0, proj_pts=10),
        dict(season=2026, week=1, fan="A", submitted_at="t", bake_id="b", comment="", player_index=1, player_id="00-2", player="Y Two", team="BUF", pos="RB", opp="DET", stat="rush_tds", arrows=-1, baseline=0.8, adjusted=0.72, proj_pts=10),
        dict(season=2026, week=1, fan="B", submitted_at="t", bake_id="b", comment="", player_index=2, player_id="", player="Z Three", team="DET", pos="TE", opp="BUF", stat="rec", arrows=3, baseline=4.0, adjusted=5.2, proj_pts=8),
        dict(season=2026, week=2, fan="B", submitted_at="t", bake_id="b", comment="", player_index=3, player_id="00-4", player="W Four", team="KC", pos="QB", opp="LAC", stat="pass_yds", arrows=1, baseline=250.0, adjusted=275.0, proj_pts=18)])
    act = pd.DataFrame([dict(player_id="00-1", player_display_name="X One", position="WR", team="BUF", week=1, receiving_yards=80.0, rushing_tds=0, receptions=0, passing_yards=0),
                        dict(player_id="00-2", player_display_name="Y Two", position="RB", team="BUF", week=1, receiving_yards=0, rushing_tds=1.0, receptions=0, passing_yards=0),
                        dict(player_id="00-9", player_display_name="Someone", position="WR", team="DET", week=1, receiving_yards=0, rushing_tds=0, receptions=0, passing_yards=0)])
    for c in COL.values():
        if c not in act.columns: act[c] = 0.0
    A.vs_bake = True                 # the selftest has no send CSVs; it checks the grading maths
    g = grade(rows, act); s = summarise(g)
    assert list(g.direction.fillna("")) == ["hit", "miss", "miss", ""], list(g.direction)
    assert abs(g.removed_pts.iloc[0] - 1.0) < 1e-6, g.removed_pts.iloc[0]        # 30 -> 20 yds error = 1.0 pt closer
    assert abs(g.removed_pts.iloc[1] - (-0.48)) < 1e-6, g.removed_pts.iloc[1]     # faded a TD that happened: 0.08 TD farther = -0.48 pt
    assert g.pending.tolist() == [False, False, False, True] and g.actual.iloc[2] == 0.0   # Z Three: game ingested, no row -> 0
    assert s["by_fan"][0]["fan"] == "A" and s["overall"]["pending"] == 1
    print("selftest ok:", json.dumps(clean(s["overall"]))); sys.exit(0)

if not os.path.exists(LONG):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "overall": {"n": 0, "graded": 0, "pending": 0}, "by_fan": [], "rows": [], "note": "no submissions recorded yet"}, open(OUT, "w"), indent=1)
    print("no submissions recorded yet -> wrote an empty docs/fan/grade.json"); sys.exit(0)
rows = pd.read_csv(LONG, dtype={"player_id": str, "fan_id": str}); rows = rows[rows.season == A.season].copy()
# WHO a row belongs to. Since 2026-09-22 the page hashes username + PIN into fan_id, so the same
# person reproduces it on any device and a name-alike without the PIN does not. Rows from before
# that (no fan_id) attach to the first PIN identity that later claims the same name -- the two
# early fans keep their history when they pick a PIN. aliases.json merges identities by hand
# ("forgot my PIN"): {old_fan_id: new_fan_id}. The display name is the one used most recently
# under that identity; two identities that display the same name get the id's tail appended.
ALIASES = os.path.join(ROOT, "data", "fan_adjustments", "aliases.json")
if len(rows):
    alias = json.load(open(ALIASES, encoding="utf-8")) if os.path.exists(ALIASES) else {}
    fid = rows.fan_id.fillna("").astype(str).str.strip()
    fid = fid.map(lambda x: alias.get(x, x))
    named = rows.assign(fidkey=fid)[fid != ""].sort_values("submitted_at")
    first_id_for_name = {}
    for r in named.itertuples():
        first_id_for_name.setdefault(norm(r.fan), r.fidkey)
    key = [f if f else ("name:" + norm(n) if norm(n) not in first_id_for_name else first_id_for_name[norm(n)])
           for f, n in zip(fid, rows.fan)]
    rows["fan_id"] = key
    latest_name = rows.sort_values("submitted_at").groupby("fan_id").fan.last()
    shown = {}
    for k, nm in latest_name.items():
        shown.setdefault(nm, []).append(k)
    display = {k: (nm if len(ks) == 1 else f"{nm} ({k[-4:]})") for nm, ks in shown.items() for k in ks}
    rows["fan"] = rows.fan_id.map(display)
# A fan who resubmits after tweaking his arrows produces a second code with a later submitted_at,
# and the long table is append-only by design, so both submissions sit in it. Only his last word
# counts: grading both would weigh every arrow he did not change twice over. The superseded rows
# stay in the CSV as history — this is the one place that decides which of them is live.
if len(rows):
    latest = rows.groupby(["season", "week", "fan"]).submitted_at.transform("max")
    superseded = rows.submitted_at != latest
    if superseded.any():
        for (wk, fan), d in rows[superseded].groupby(["week", "fan"]):
            kept = rows[(rows.week == wk) & (rows.fan == fan) & ~superseded].submitted_at.iloc[0]
            print(f"  {fan} wk{int(wk)}: {len(d)} arrow(s) superseded by a later submission ({kept})")
        rows = rows[~superseded].copy()
# An arrow is a forecast or it is nothing. The page bakes the whole week, so a fan scrolling it on
# Sunday can still put arrows on Thursday's game, and nflverse has that box score — the grader would
# score those instantly against a result the fan could already have read. Clay's first submission had
# 21 of 52 arrows on Detroit @ Buffalo, three days after it was played. Kickoff comes from the
# schedule rather than the bake, so this holds for rows recorded before bake ids existed.
_sch = os.path.join(ROOT, "data", f"schedule_{A.season}.csv")
if len(rows) and os.path.exists(_sch):
    _s = pd.read_csv(_sch)
    _s = _s[(_s.season == A.season) & (_s.game_type == "REG")]
    _k = pd.to_datetime(_s.gameday.astype(str) + " " + _s.gametime.fillna("13:00").astype(str),
                        errors="coerce").dt.tz_localize("America/New_York", ambiguous="NaT",
                                                        nonexistent="shift_forward").dt.tz_convert("UTC")
    kick = {}
    for r, k in zip(_s.itertuples(), _k):
        if pd.notna(k):
            kick[(int(r.week), r.home_team)] = k
            kick[(int(r.week), r.away_team)] = k
    rows["_kick"] = [kick.get((int(w), t)) for w, t in zip(rows.week, rows.team)]
    _sub = pd.to_datetime(rows.submitted_at, errors="coerce", utc=True)
    late = rows._kick.notna() & _sub.notna() & (_sub > rows._kick)
    if late.any():
        for (fan, wk), d in rows[late].groupby(["fan", "week"]):
            games = ", ".join(sorted({f"{r.team} v {r.opp}" for r in d.itertuples()}))
            print(f"  {fan} wk{int(wk)}: {len(d)} arrow(s) dropped — submitted after kickoff ({games})")
        rows = rows[~late].copy()
    rows = rows.drop(columns=["_kick"])
url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{A.season}.parquet"
act = pd.read_parquet(url); act = act[act.season_type == "REG"]
for c in COL.values():
    if c not in act.columns: act[c] = 0.0
SENT = {} if A.vs_bake else sent_line(A.sends, A.min_lead)
if not A.vs_bake:
    print(f"  sent lines for {len(SENT)} player-weeks from {', '.join(A.sends)}")
g = grade(rows, act); s = summarise(g); s["season"] = A.season
s["basis"] = "bake" if A.vs_bake else "send"
if int(g.no_send.sum()):
    for (fan, wk), d in g[g.no_send].groupby(["fan", "week"]):
        who = ", ".join(sorted({r.player for r in d.itertuples()}))
        print(f"  {fan} wk{int(wk)}: {len(d)} arrow(s) not scorable — no eligible send line ({who})")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(clean(s), open(OUT, "w"), indent=1)
o = s["overall"]
print(f"fan picks: {o['n']} arrows from {len(s['by_fan'])} fan(s); {o['graded']} graded, {o['pending']} pending"
      + (f"; direction {o['hit_rate']:.0%} ({o['hits']}-{o['misses']}), closer {o['closer']} / farther {o['farther']}, net {o['removed_pts']:+.2f} pts of error removed" if o["graded"] else ""))
for f in s["by_fan"][:10]:
    print(f"  {f['fan']:<24} graded {f['graded']:>3}  pending {f['pending']:>3}  hit {f['hit_rate'] if f['hit_rate'] is not None else '-'}  removed {f['removed_pts'] if f['removed_pts'] is not None else '-'} pts")
print(f"wrote docs/fan/grade.json")
