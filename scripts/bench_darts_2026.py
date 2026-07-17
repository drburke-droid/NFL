"""
BENCH DARTS 2026 — the ranked list of cheap players to fill the bench with.

Fuses every validated signal in the project into one score, for players the
market prices <= ~$8 (the fill-engine bin that regenerates starters):

  breakout engine   dart_prob (price-anchored late-breakout) / leap_prob
                    (usage-trend value-leap) / lottery p (validated screen:
                    top-10/season hit 8% star + 20% startable)
  role mechanisms   HEIR (backup out-ran his starter: 35% take the job),
                    TAKEOVER (skill-gap: 58%), VACATED ROLE (+1.7 PPG),
                    WON JOB (+0.7 PPG), open WR2 room (crowding)
  talent vs price   alpha_skill percentile (buried alpha), H2-2025 trend
  upside            ceiling PPG, boom%
  keeper runway     age curve x1.25 (<=23) ... x0.6 (30+) — next-year keeper
  position tilt     RB x1.15 (fill score: RB is the un-fillable position),
                    WR x1.0, TE x0.85, QB excluded (streamed in this league)

Output: outputs/reports/bench_darts_2026.md + console top-25.
"""
import json, os, re, sqlite3
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
pd.set_option("display.width", 240)

# HEIR_2026 (index.html, backup_rb_signal_study.py): out-ran own starter on ypc AND EPA/c.
# BLOCKER multiplier (blocker_entrenchment_study.py, 2016-25 + real OTC contracts):
#   BIG-PAID non-expiring blocker (>=3% cap, signed <=2yrs): 3.3% heir-star -> x0.4
#   cheap-or-EXPIRING deal AND age 29+: 18.8% heir-star (Ingram/Stewart/Mostert) -> x1.3
#   young (<27) blocker: 5.0% star -> x0.6
# 2026: Saquon is 29 ON AN EXPIRING DEAL (2yr 2025 ext ends after 2026) -> Bigsby is in
# the 18.8% cell, NOT capped. Kamara 31 / CMC 30 / Henry 32 (old deals) -> x1.3.
# Bucky Irving 24 / Kyren 26 / Vidal 24 / Woody Marks 24 / Chase Brown 26 -> x0.6.
# No 2026 heir sits behind a big-paid non-expiring blocker (the x0.4 cell).
HEIR = {"Rachaad White": 0.6, "Tank Bigsby": 1.3, "Blake Corum": 0.6, "Omarion Hampton": 0.6,
        "Devin Neal": 1.3, "Brian Robinson": 1.3, "Nick Chubb": 0.6, "Samaje Perine": 0.6,
        "Keaton Mitchell": 1.3}


def load_js_const(fname, varname):
    """Evaluate the JS data file in Node (handles unquoted keys) and return the const."""
    import subprocess
    js = (f"const s=require('fs').readFileSync({json.dumps(os.path.join(DOCS, fname))},'utf8');"
          f"const v=new Function(s+';return {varname};')();"
          f"process.stdout.write(JSON.stringify(v));")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"{fname}/{varname}: {out.stderr[:300]}")
    return json.loads(out.stdout)


players = pd.DataFrame(load_js_const("data.js", "PLAYERS"))
lottery = load_js_const("lottery_2026.js", "LOTTERY_2026")
crowding = load_js_const("crowding_2026.js", "CROWDING_2026") if os.path.exists(os.path.join(DOCS, "crowding_2026.js")) else {}
takeover = load_js_const("takeover_2026.js", "TAKEOVER_2026") if os.path.exists(os.path.join(DOCS, "takeover_2026.js")) else {}

# alpha skill percentile (2025) by name+pos
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
# day-2 small-school flag (buzz_leadlag_study.py part B: rd3-4 non-P5 RB/WR hit 22%
# vs 9% for P5 same rounds, 2016-25 — the Kupp/Diontae pattern). 2025-26 draftees.
P5 = {"Alabama","Georgia","LSU","Florida","Tennessee","Auburn","Texas A&M","Ole Miss","Mississippi State",
      "Arkansas","Kentucky","South Carolina","Missouri","Vanderbilt","Ohio State","Michigan","Penn State",
      "Michigan State","Wisconsin","Iowa","Minnesota","Illinois","Indiana","Purdue","Northwestern","Nebraska",
      "Maryland","Rutgers","Texas","Oklahoma","Oklahoma State","Baylor","TCU","Texas Tech","Kansas State",
      "Kansas","Iowa State","West Virginia","Clemson","Florida State","Miami","North Carolina","NC State",
      "Duke","Wake Forest","Virginia","Virginia Tech","Pittsburgh","Louisville","Syracuse","Boston College",
      "Georgia Tech","USC","UCLA","Oregon","Washington","Stanford","California","Oregon State",
      "Washington State","Arizona","Arizona State","Utah","Colorado","Notre Dame"}
_dr = pd.read_sql("""SELECT pfr_player_name nm, college, round FROM nflv_draft
                     WHERE season>=2025 AND position IN ('RB','WR') AND round IN (3,4)""", con)
_FIX = {"Mississippi": "Ole Miss", "Miami (FL)": "Miami", "Southern California": "USC", "Pitt": "Pittsburgh"}
_dr["college_n"] = _dr.college.str.replace(" St.", " State", regex=False).replace(_FIX)
SMALL_SCHOOL = set(_dr[~_dr.college_n.isin(P5)].nm)
# college dominator (college_dominator_study.py, classes 2018-24): WR share adds nothing
# (draft capital prices it); RB-ONLY weak-but-directional — day-3 RBs with high final-
# season scrimmage share star 11% vs 3%, low-share committee backs hit half as often.
# Small weights, RB rookies only.
_suf = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_nrm = lambda s: re.sub(r"\s+", " ", _suf.sub("", str(s).lower().replace(".", "").replace("'", "")
                        .replace("-", " ").replace(",", ""))).strip()
_cp = pd.read_sql("SELECT player, dom_scrim FROM nflv_college_prod WHERE season=2025", con)
_cp["nm"] = _cp.player.map(_nrm)
DOM_SCRIM = _cp.drop_duplicates("nm").set_index("nm")["dom_scrim"]
alpha = pd.read_sql("""SELECT s.player_display_name name, s.position, a.alpha_skill
                       FROM player_skill_alpha a JOIN nflv_season s
                         ON s.player_id=a.player_id AND s.season=a.season
                       WHERE a.season=2025""", con).drop_duplicates(["name", "position"])
board = pd.read_sql("SELECT player_display_name name, position, auction FROM draft_board_2026", con) \
        .drop_duplicates(["name", "position"])
con.close()
alpha["apct"] = alpha.groupby("position")["alpha_skill"].rank(pct=True)

d = players[players.position.isin(["RB", "WR", "TE"])].copy()
d = d.merge(alpha[["name", "position", "apct"]], on=["name", "position"], how="left") \
     .merge(board, on=["name", "position"], how="left")

# ---- cost gate: market must price him cheap (<= $8 across available sources) ----
d["mkt_cost"] = d[["ffa_aav", "espn_av"]].max(axis=1)
d["mkt_cost"] = d[["mkt_cost", "auction"]].max(axis=1)
d = d[(d.mkt_cost.fillna(1) <= 8)]

# ---- signals ----
d["lot_p"] = d.name.map(lambda n: lottery.get(n, {}).get("p", 0) if isinstance(lottery.get(n), dict) else 0)
d["dart"] = d[["dart_prob", "leap_prob", "lot_p"]].fillna(0).max(axis=1)
d["upside_p"] = np.where(d.is_rookie == 1, d.hit_prob.fillna(0), d.breakout_prob.fillna(0))
d["heir"] = d.name.map(HEIR).fillna(0)   # blocker-adjusted heir weight (0.6 / 1.3)
d["tko"] = d.name.map(lambda n: 1 if n in takeover else 0)
d["crowd_open"] = d.name.map(lambda n: 1 if (isinstance(crowding.get(n), dict) and "open" in str(crowding.get(n)).lower())
                             or crowding.get(n) == "open" else 0)
d["alpha_hi"] = (d.apct.fillna(0) >= 0.80).astype(int)
d["h2"] = ((d.trend_dppg.fillna(0) > 1.5) & (d.trend_dsnap.fillna(0) > 5)).astype(int)
d["ceil_n"] = (d.ceiling.fillna(0) / 20).clip(0, 1)
d["smallschool"] = d.name.isin(SMALL_SCHOOL).astype(int)   # day-2 small-school (24% vs 10% hit)
_dom = d.name.map(lambda n: DOM_SCRIM.get(_nrm(n), np.nan))
d["col_dom"] = np.where((d.position == "RB") & (d.is_rookie == 1) & (_dom >= 0.28), 1,
                np.where((d.position == "RB") & (d.is_rookie == 1) & (_dom < 0.15), -1, 0))

runway = lambda a: 1.25 if a <= 23 else 1.15 if a <= 25 else 1.0 if a <= 27 else 0.8 if a <= 29 else 0.6
d["run_x"] = d.age.fillna(26).map(runway)
POS_X = {"RB": 1.15, "WR": 1.0, "TE": 0.85}
d["pos_x"] = d.position.map(POS_X)

# weights refit by the 2016-25 walk-forward backtest (bench_darts_backtest.py +
# per-signal breakdown): buried ALPHA is the star-finder (26% reliable / 16% star,
# 6.2x baseline) -> heaviest non-model weight; H2 trend was dead weight -> dropped;
# HEIR makes reliable players, rarely stars -> kept moderate. Model layers (dart /
# breakout / rookie-hit) keep their separately-validated weights.
d["score"] = ((2.5 * d.dart + 1.5 * d.upside_p
               + 1.00 * d.alpha_hi
               + 0.45 * (d.vacated_role.fillna(0)) + 0.40 * d.heir + 0.35 * d.tko
               + 0.15 * (d.won_job.fillna(0)) + 0.15 * d.crowd_open
               + 0.25 * d.smallschool + 0.15 * d.col_dom
               + 0.35 * d.ceil_n)
              * d.run_x * d.pos_x)

d = d.sort_values("score", ascending=False)


def why(r):
    w = []
    if r.dart >= 0.10: w.append(f"dart {r.dart:.0%}")
    if r.upside_p >= 0.30: w.append(("hit" if r.is_rookie == 1 else "breakout") + f" {r.upside_p:.0%}")
    if r.heir: w.append("HEIR" + ("+" if r.heir > 1 else "-"))   # +: vulnerable blocker, -: young/entrenched
    if r.tko: w.append("TAKEOVER")
    if r.vacated_role == 1: w.append("VACATED")
    if r.won_job == 1: w.append("won-job")
    if r.crowd_open: w.append("WR2-open")
    if r.alpha_hi: w.append(f"alpha {r.apct:.0%}")
    if r.h2: w.append(f"H2 +{r.trend_dppg:.1f}ppg")
    if r.smallschool: w.append("small-school d2")
    if r.col_dom == 1: w.append("college workhorse")
    elif r.col_dom == -1: w.append("college committee")
    if (r.ceiling or 0) >= 13: w.append(f"ceil {r.ceiling:.0f}")
    return " · ".join(w)


L = ["# 🎯 Bench darts 2026 — ranked (cheap now, startable/keeper later)\n",
     "Cost gate: market <= $8 (max of FFA AAV / ESPN $ / board $). Score = validated breakout",
     "signals (dart/leap/lottery x2.5, breakout-hit x1.5) + role mechanisms (HEIR, TAKEOVER,",
     "vacated, won-job, open WR2) + buried alpha + H2 trend + ceiling, x keeper-runway age",
     "curve, x position fill-tilt (RB 1.15 — the wire can't rescue RB).\n",
     "| # | Pos | Player | Tm | Age | $ | Score | Why |", "|---|---|---|---|---|---|---|---|"]
for i, (_, r) in enumerate(d.head(30).iterrows(), 1):
    L.append(f"| {i} | {r.position} | {r['name']} | {r.team} | {r.age:.0f} | ${r.mkt_cost:.0f} | "
             f"{r.score:.2f} | {why(r)} |")
out = os.path.join(ROOT, "outputs", "reports", "bench_darts_2026.md")
open(out, "w", encoding="utf-8").write("\n".join(L))

print(d.head(30)[["position", "name", "team", "age", "mkt_cost", "score"]]
      .assign(why=d.head(30).apply(why, axis=1)).to_string(index=False))
print(f"\nWrote {out}")
