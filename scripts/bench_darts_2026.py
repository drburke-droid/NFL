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

# HEIR_2026 (index.html, backup_rb_signal_study.py): out-ran own starter on ypc AND EPA/c
HEIR = ["Rachaad White", "Tank Bigsby", "Blake Corum", "Omarion Hampton", "Devin Neal",
        "Brian Robinson", "Nick Chubb", "Samaje Perine", "Keaton Mitchell"]


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
d["heir"] = d.name.isin(HEIR).astype(int)
d["tko"] = d.name.map(lambda n: 1 if n in takeover else 0)
d["crowd_open"] = d.name.map(lambda n: 1 if (isinstance(crowding.get(n), dict) and "open" in str(crowding.get(n)).lower())
                             or crowding.get(n) == "open" else 0)
d["alpha_hi"] = (d.apct.fillna(0) >= 0.80).astype(int)
d["h2"] = ((d.trend_dppg.fillna(0) > 1.5) & (d.trend_dsnap.fillna(0) > 5)).astype(int)
d["ceil_n"] = (d.ceiling.fillna(0) / 20).clip(0, 1)

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
               + 0.35 * d.ceil_n)
              * d.run_x * d.pos_x)

d = d.sort_values("score", ascending=False)


def why(r):
    w = []
    if r.dart >= 0.10: w.append(f"dart {r.dart:.0%}")
    if r.upside_p >= 0.30: w.append(("hit" if r.is_rookie == 1 else "breakout") + f" {r.upside_p:.0%}")
    if r.heir: w.append("HEIR")
    if r.tko: w.append("TAKEOVER")
    if r.vacated_role == 1: w.append("VACATED")
    if r.won_job == 1: w.append("won-job")
    if r.crowd_open: w.append("WR2-open")
    if r.alpha_hi: w.append(f"alpha {r.apct:.0%}")
    if r.h2: w.append(f"H2 +{r.trend_dppg:.1f}ppg")
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
