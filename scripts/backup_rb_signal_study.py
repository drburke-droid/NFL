"""
Starter-vs-backup RB signal study: same line, different backs.

Every team-season 2011-2025 gets a starter (carry leader, >=100 carries) and a
primary backup (2nd in carries, >=50). Both run behind the same O-line, so the
gap between them is the cleanest scalpel we have for separating line from back.

  Part 1 — line vs skill: do same-line teammates co-vary more than a player
           persists year to year? (if yes, the line owns the number)
  Part 2 — does the starter's edge-over-backup predict his future better than
           his raw efficiency does?
  Part 3 — backups who out-run their starter: are they more valuable next year
           (ppg, touches, promotion to lead back)?
  Part 4 — inheritance events: starter gone next year, backup takes over — did
           the "beat the starter" flag predict who delivers?
  Part 5 — 2025 backups who beat their starter (the 2026 watch list)

Report: outputs/reports/backup_rb_signal.md
"""
import os, sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))

wk = pd.read_sql("""SELECT player_id, player_display_name nm, season, team, carries,
                    rushing_yards, rushing_epa, receptions, targets, fantasy_points_ppr
                    FROM nflv_weekly WHERE position='RB' AND season_type='REG'""", con)
tr = pd.read_sql("""SELECT player_id, season, age, draft_number FROM nflv_traj
                    WHERE position='RB'""", con)
tr["age"] = pd.to_numeric(tr.age, errors="coerce")
tr["draft_number"] = pd.to_numeric(tr.draft_number, errors="coerce")

# player-team-season lines (a trade mid-season = two stints, thresholds per stint)
pts = (wk.groupby(["player_id", "nm", "season", "team"])
         .agg(g=("carries", "size"), car=("carries", "sum"), ry=("rushing_yards", "sum"),
              repa=("rushing_epa", "sum"), rec=("receptions", "sum"), tgt=("targets", "sum"),
              ppr=("fantasy_points_ppr", "sum")).reset_index())
pts["ypc"] = pts.ry / pts.car.replace(0, np.nan)
pts["epa_c"] = pts.repa / pts.car.replace(0, np.nan)
pts["ppg"] = pts.ppr / pts.g
pts["tch_pg"] = (pts.car + pts.rec) / pts.g

# player-season totals across teams (for "next year" outcomes)
ps = (wk.groupby(["player_id", "season"])
        .agg(g=("carries", "size"), car=("carries", "sum"), rec=("receptions", "sum"),
             ppr=("fantasy_points_ppr", "sum")).reset_index())
ps["ppg"] = ps.ppr / ps.g
ps["tch_pg"] = (ps.car + ps.rec) / ps.g

# each team-season's carry leader (for "became the lead back" checks)
lead = pts.loc[pts.groupby(["season", "team"]).car.idxmax()]
lead_ids = set(zip(lead[lead.car >= 100].player_id, lead[lead.car >= 100].season))

def nxt(pid, season, col):
    r = ps[(ps.player_id == pid) & (ps.season == season + 1)]
    return r.iloc[0][col] if len(r) else np.nan

# build starter/backup pairs
rows = []
for (season, team), g in pts.groupby(["season", "team"]):
    g = g.sort_values("car", ascending=False)
    if len(g) < 2 or g.iloc[0].car < 100 or g.iloc[1].car < 50:
        continue
    s, b = g.iloc[0], g.iloc[1]
    rows.append(dict(season=season, team=team,
                     s_id=s.player_id, s_nm=s.nm, s_car=s.car, s_ypc=s.ypc,
                     s_epa=s.epa_c, s_ppg=s.ppg,
                     b_id=b.player_id, b_nm=b.nm, b_car=b.car, b_ypc=b.ypc,
                     b_epa=b.epa_c, b_ppg=b.ppg, b_tch=b.tch_pg))
pairs = pd.DataFrame(rows)
pairs["d_ypc"] = pairs.s_ypc - pairs.b_ypc
pairs["d_epa"] = pairs.s_epa - pairs.b_epa
for side in ("s", "b"):
    pairs = pairs.merge(tr.rename(columns={"player_id": f"{side}_id", "age": f"{side}_age",
                                           "draft_number": f"{side}_dr"}),
                        on=[f"{side}_id", "season"], how="left")

def corr(x, y):
    m = x.notna() & y.notna()
    return (np.corrcoef(x[m], y[m])[0, 1], m.sum()) if m.sum() > 2 else (np.nan, 0)

print("=" * 96)
print("PART 1 — LINE vs SKILL: same-line teammates vs a player's own year-to-year self")
print("=" * 96)
print(f"pairs: {len(pairs)} team-seasons, 2011-2025 (starter >=100 car, backup >=50 car)\n")
r1, n1 = corr(pairs.s_ypc, pairs.b_ypc)
r2, n2 = corr(pairs.s_epa, pairs.b_epa)
print(f"same season, same line, different backs:   ypc r={r1:+.2f}  EPA/c r={r2:+.2f}  (n={n1})")

# a starter's own persistence: year N vs N+1, split by same team (line) or new team
st = pts[(pts.car >= 100)].copy().sort_values(["player_id", "season"])
st["n_ypc"] = st.groupby("player_id").ypc.shift(-1)
st["n_epa"] = st.groupby("player_id").epa_c.shift(-1)
st["n_team"] = st.groupby("player_id").team.shift(-1)
st["n_season"] = st.groupby("player_id").season.shift(-1)
st = st[st.n_season == st.season + 1]
same = st[st.n_team == st.team]; moved = st[st.n_team != st.team]
r3, n3 = corr(same.ypc, same.n_ypc); r4, n4 = corr(same.epa_c, same.n_epa)
r5, n5 = corr(moved.ypc, moved.n_ypc); r6, n6 = corr(moved.epa_c, moved.n_epa)
print(f"same back, next year, SAME team/line:      ypc r={r3:+.2f}  EPA/c r={r4:+.2f}  (n={n3})")
print(f"same back, next year, NEW team/line:       ypc r={r5:+.2f}  EPA/c r={r6:+.2f}  (n={n5})")
print("\nreading: teammate corr ~= how much the line owns the stat; a back keeping his")
print("number on a NEW line ~= how much he owns it himself.")

print("\n" + "=" * 96)
print("PART 2 — DOES EDGE-OVER-BACKUP BEAT RAW EFFICIENCY AS A PREDICTOR?")
print("=" * 96)
p2 = pairs.copy()
p2["s_n_ppg"] = [nxt(r.s_id, r.season, "ppg") for r in p2.itertuples()]
st_next = st.set_index(["player_id", "season"])
p2 = p2.join(st_next[["n_ypc", "n_epa"]], on=["s_id", "season"])
p2 = p2[p2.season < 2025]
for tgt_col, lab in (("n_ypc", "next-yr ypc"), ("n_epa", "next-yr EPA/c"), ("s_n_ppg", "next-yr ppg")):
    ra, na = corr(p2.s_ypc, p2[tgt_col]); rb_, _ = corr(p2.d_ypc, p2[tgt_col])
    rc, _ = corr(p2.s_epa, p2[tgt_col]); rd, _ = corr(p2.d_epa, p2[tgt_col])
    print(f"predicting starter's {lab:<14} raw ypc r={ra:+.2f}  vs edge-ypc r={rb_:+.2f}   "
          f"raw EPA/c r={rc:+.2f}  vs edge-EPA r={rd:+.2f}   (n={na})")

print("\n" + "=" * 96)
print("PART 3 — BACKUPS WHO OUT-RAN THEIR STARTER: what are they worth next year?")
print("=" * 96)
p3 = pairs[pairs.season < 2025].copy()
p3["beat"] = (p3.b_ypc > p3.s_ypc) & (p3.b_epa > p3.s_epa)
p3["n_ppg"] = [nxt(r.b_id, r.season, "ppg") for r in p3.itertuples()]
p3["n_tch"] = [nxt(r.b_id, r.season, "tch_pg") for r in p3.itertuples()]
p3["n_lead"] = [(r.b_id, r.season + 1) in lead_ids for r in p3.itertuples()]
print(f"{'group':<34}{'n':>5}{'age':>6}{'drafted':>9}{'ppg now':>9}{'ppg next':>10}"
      f"{'tch/g next':>12}{'lead back%':>12}{'gone%':>7}")
for lab, s in (("beat starter (ypc AND EPA/c)", p3[p3.beat]),
               ("did not beat starter", p3[~p3.beat])):
    n = s.dropna(subset=["n_ppg"])
    print(f"{lab:<34}{len(s):>5}{s.b_age.mean():>6.1f}{s.b_dr.mean():>9.0f}"
          f"{s.b_ppg.mean():>9.1f}{n.n_ppg.mean():>10.1f}{n.n_tch.mean():>12.1f}"
          f"{s.n_lead.mean():>12.1%}{1 - len(n)/max(len(s),1):>7.1%}")
# same cut but demand a real margin and real volume
big = p3[(p3.b_ypc - p3.s_ypc >= 0.5) & (p3.b_epa > p3.s_epa) & (p3.b_car >= 80)]
n = big.dropna(subset=["n_ppg"])
print(f"{'  beat by >=0.5 ypc, >=80 car':<34}{len(big):>5}{big.b_age.mean():>6.1f}"
      f"{big.b_dr.mean():>9.0f}{big.b_ppg.mean():>9.1f}{n.n_ppg.mean():>10.1f}"
      f"{n.n_tch.mean():>12.1f}{big.n_lead.mean():>12.1%}{1 - len(n)/max(len(big),1):>7.1%}")

print("\n" + "=" * 96)
print("PART 4 — INHERITANCE: starter gone next year, backup still there — who delivers?")
print("=" * 96)
p4 = p3.copy()
p4["s_stays"] = [(r.s_id, r.season + 1) in lead_ids for r in p4.itertuples()]
inh = p4[~p4.s_stays & p4.n_ppg.notna()]
print("backup's next season when the starter does NOT repeat as a lead back anywhere:")
print(f"{'group':<34}{'n':>5}{'ppg next':>10}{'tch/g next':>12}{'lead back%':>12}{'ppg>=12%':>10}")
for lab, s in (("backup had beaten starter", inh[inh.beat]),
               ("backup had not", inh[~inh.beat])):
    print(f"{lab:<34}{len(s):>5}{s.n_ppg.mean():>10.1f}{s.n_tch.mean():>12.1f}"
          f"{s.n_lead.mean():>12.1%}{(s.n_ppg >= 12).mean():>10.1%}")
top = inh[inh.beat].nlargest(8, "n_ppg")
print("\nbest inheritances after beating the starter:")
for r in top.itertuples():
    print(f"  {r.b_nm:<24} {int(r.season)} {r.team}: {r.b_ypc:.2f} vs {r.s_ypc:.2f} ypc "
          f"behind {r.s_nm} -> {r.n_ppg:.1f} ppg next yr")

print("\n" + "=" * 96)
print("PART 5 — 2025 BACKUPS WHO BEAT THEIR STARTER (2026 watch list)")
print("=" * 96)
w = pairs[(pairs.season == 2025) & (pairs.b_ypc > pairs.s_ypc) & (pairs.b_epa > pairs.s_epa)]
print(f"{'backup':<24}{'tm':>4}{'car':>5}{'ypc':>6}{'EPA/c':>8}{'age':>5}   starter (ypc, EPA/c)")
for r in w.sort_values("d_epa").itertuples():
    age = "-" if pd.isna(r.b_age) else f"{r.b_age:.0f}"
    print(f"{r.b_nm:<24}{r.team:>4}{r.b_car:>5.0f}{r.b_ypc:>6.2f}{r.b_epa:>+8.3f}{age:>5}"
          f"   {r.s_nm} ({r.s_ypc:.2f}, {r.s_epa:+.3f})")

open(os.path.join(ROOT, "outputs", "reports", "backup_rb_signal.md"), "w", encoding="utf-8").write(
    "# Backup-vs-starter RB signal — same line, different backs\n\n"
    "Generated by `scripts/backup_rb_signal_study.py` from nflv_weekly/nflv_traj, 2011-2025.\n\n"
    "```\n" + "\n".join(_out) + "\n```\n")
_print("\nwrote outputs/reports/backup_rb_signal.md")
