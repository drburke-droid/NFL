"""
Christian McCaffrey career arc: is decline showing in the leading indicators?

Per-game production + the advanced stats that historically lead RB decline:
  - rushing EPA/carry and YPC (burst/efficiency fades before volume does)
  - receiving efficiency (EPA/target, yds/target, RACR) and target share (teams cut
    passing-down work first when they see decline)
  - snap share (the team's own real-time read on the player)
  - workload odometer (career touches) + age vs the historical RB age curve
  - within-season fades (first half vs second half efficiency, by season)
Context: age-29+ high-workload RB seasons since 2012 — base rates for the next year.

Report: outputs/reports/cmc_decline.md
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
PID = "00-0033280"

wk = pd.read_sql(f"""SELECT season, week, season_type, team, opponent_team, carries,
                     rushing_yards, rushing_tds, rushing_epa, targets, receptions,
                     receiving_yards, receiving_tds, receiving_epa, receiving_air_yards,
                     target_share, wopr, fantasy_points_ppr
                     FROM nflv_weekly WHERE player_id='{PID}' ORDER BY season, week""", con)
sn = pd.read_sql("""SELECT season, week, offense_pct FROM nflv_snaps
                    WHERE player='Christian McCaffrey'""", con)
wk = wk.merge(sn, on=["season", "week"], how="left")
wk["touches"] = wk.carries + wk.receptions
reg = wk[wk.season_type == "REG"].copy()

print("=" * 96)
print("PART 1 — SEASON-BY-SEASON PER-GAME LINE (regular season)")
print("=" * 96)
print(f"{'yr':<5}{'age':>4}{'g':>3}{'snap%':>7}{'tch/g':>7}{'ypc':>6}{'EPA/c':>7}"
      f"{'tgt/g':>7}{'y/tgt':>7}{'EPA/t':>7}{'tgtSh':>7}{'wopr':>6}{'ppr/g':>7}{'odo':>7}")
odo = 0
for season, g in reg.groupby("season"):
    age = season - 1996  # born June 1996
    n = len(g)
    ypc = g.rushing_yards.sum() / max(g.carries.sum(), 1)
    epa_c = g.rushing_epa.sum() / max(g.carries.sum(), 1)
    ytgt = g.receiving_yards.sum() / max(g.targets.sum(), 1)
    epa_t = g.receiving_epa.sum() / max(g.targets.sum(), 1)
    odo += g.touches.sum()
    print(f"{season:<5}{age:>4}{n:>3}{g.offense_pct.mean():>7.0%}{g.touches.mean():>7.1f}"
          f"{ypc:>6.2f}{epa_c:>+7.3f}{g.targets.mean():>7.1f}{ytgt:>7.2f}{epa_t:>+7.3f}"
          f"{g.target_share.mean():>7.1%}{g.wopr.mean():>6.2f}{g.fantasy_points_ppr.mean():>7.1f}{odo:>7.0f}")
print("\n(odo = cumulative career REG-season touches; 2020 ankle/shoulder, 2021 hamstring,")
print(" 2024 achilles tendinitis + knee = the injury seasons)")

print("\n" + "=" * 96)
print("PART 2 — WITHIN-SEASON FADE (first half vs rest, full seasons only)")
print("=" * 96)
print(f"{'yr':<6}{'half':<8}{'g':>3}{'ypc':>6}{'EPA/c':>8}{'y/tgt':>7}{'ppr/g':>7}{'snap%':>7}")
for season in (2017, 2018, 2019, 2022, 2023, 2025):
    g = reg[reg.season == season].sort_values("week")
    mid = len(g) // 2
    for lab, h in (("wk1-mid", g.iloc[:mid]), ("mid-end", g.iloc[mid:])):
        ypc = h.rushing_yards.sum() / max(h.carries.sum(), 1)
        epa = h.rushing_epa.sum() / max(h.carries.sum(), 1)
        ytg = h.receiving_yards.sum() / max(h.targets.sum(), 1)
        print(f"{season:<6}{lab:<8}{len(h):>3}{ypc:>6.2f}{epa:>+8.3f}{ytg:>7.2f}"
              f"{h.fantasy_points_ppr.mean():>7.1f}{h.offense_pct.mean():>7.0%}")

# playoffs 2025 as the freshest tape
po = wk[(wk.season == 2025) & (wk.season_type != "REG")]
if len(po):
    ypc = po.rushing_yards.sum() / max(po.carries.sum(), 1)
    print(f"\n2025 playoffs ({len(po)} g): {po.touches.mean():.1f} tch/g, {ypc:.2f} ypc, "
          f"EPA/c {po.rushing_epa.sum()/max(po.carries.sum(),1):+.3f}, "
          f"{po.fantasy_points_ppr.mean():.1f} ppr/g")

print("\n" + "=" * 96)
print("PART 3 — AGE CURVE CONTEXT: high-workload RB seasons at 28+ (2012-2025)")
print("=" * 96)
tr = pd.read_sql("""SELECT player_display_name nm, season, age, games, ppg, carries_pg,
                    targets_pg, rushing_yards_pg, receiving_yards_pg, rushing_epa
                    FROM nflv_traj WHERE position='RB' AND season>=2012""", con)
tr["ypc"] = tr.rushing_yards_pg / tr.carries_pg.replace(0, np.nan)
tr = tr[(tr.games >= 8) & (tr.carries_pg >= 10)]
tr = tr.sort_values(["nm", "season"])
tr["next_ppg"] = tr.groupby("nm").ppg.shift(-1)
tr["next_games"] = tr.groupby("nm").games.shift(-1)
tr["next_season"] = tr.groupby("nm").season.shift(-1)
tr.loc[tr.next_season != tr.season + 1, ["next_ppg", "next_games"]] = np.nan
print("workhorse RB seasons (>=8 g, >=10 carries/g): what happened the NEXT year?")
print(f"   {'age':<10}{'n':>4}{'ppg':>7}{'next ppg':>10}{'Δppg':>7}{'next<12g%':>11}{'gone%':>7}")
for lo, hi in ((23, 25), (26, 27), (28, 28), (29, 29), (30, 33)):
    s = tr[tr.age.between(lo, hi)]
    nxt = s.dropna(subset=["next_ppg"])
    gone = 1 - len(nxt) / max(len(s), 1)
    print(f"   {f'{lo}-{hi}':<10}{len(s):>4}{s.ppg.mean():>7.1f}{nxt.next_ppg.mean():>10.1f}"
          f"{(nxt.next_ppg - nxt.ppg).mean():>+7.1f}{(nxt.next_games < 12).mean():>11.1%}{gone:>7.1%}")
elite = tr[(tr.age >= 29) & (tr.ppg >= 18)]
nxt = elite.dropna(subset=["next_ppg"])
print(f"\nELITE age-29+ seasons (ppg>=18, n={len(elite)}): "
      f"next year {nxt.next_ppg.mean():.1f} ppg ({(nxt.next_ppg-nxt.ppg).mean():+.1f}), "
      f"{(nxt.next_games<12).mean():.0%} played <12 games, "
      f"{1-len(nxt)/max(len(elite),1):.0%} no next season")
print("who:", ", ".join(f"{r.nm} {int(r.season)} ({r.ppg:.0f}→"
      f"{'-' if pd.isna(r.next_ppg) else f'{r.next_ppg:.0f}'})" for r in elite.itertuples()))

print("\n" + "=" * 96)
print("PART 3.5 — ENVIRONMENT CONTROL: him or the blocking?")
print("=" * 96)
for yr in (2023, 2025):
    print(f"  {yr} SF RBs (>=20 carries) vs league RB avg:")
    for r in con.execute(f"""SELECT player_display_name, SUM(carries),
            ROUND(SUM(rushing_yards)*1.0/SUM(carries),2), ROUND(SUM(rushing_epa)/SUM(carries),3)
            FROM nflv_weekly WHERE season={yr} AND season_type='REG' AND team='SF'
            AND position='RB' GROUP BY 1 HAVING SUM(carries)>=20 ORDER BY 2 DESC"""):
        print(f"    {r[0]:<24} {r[1]:>4} car  {r[2]:>5} ypc  {r[3]:+.3f} EPA/c")
    lg = con.execute(f"""SELECT ROUND(SUM(rushing_yards)*1.0/SUM(carries),2),
            ROUND(SUM(rushing_epa)/SUM(carries),3) FROM nflv_weekly
            WHERE season={yr} AND season_type='REG' AND position='RB'""").fetchone()
    print(f"    {'LEAGUE RB AVG':<24}       {lg[0]:>5} ypc  {lg[1]:+.3f} EPA/c")

print("\n" + "=" * 96)
print("PART 4 — MODEL & MARKET VIEW")
print("=" * 96)
imp = con.execute("""SELECT prior_ppg, fade_prob FROM nflv_implosion
                     WHERE player_display_name='Christian McCaffrey'""").fetchone()
if imp:
    print(f"implosion model (2026): prior_ppg {imp[0]:.1f}, fade probability {imp[1]:.1%}")
try:
    b = con.execute("""SELECT * FROM board_2026 WHERE player LIKE '%McCaffrey%'""").fetchone()
    cols = [d[0] for d in con.execute("SELECT * FROM board_2026 LIMIT 1").description]
    if b:
        print("board_2026:", {k: v for k, v in zip(cols, b)
                              if k in ("player", "pos", "rank", "adp", "proj", "price", "tier", "value")})
except Exception:
    pass

open(os.path.join(ROOT, "outputs", "reports", "cmc_decline.md"), "w", encoding="utf-8").write(
    "# Christian McCaffrey — decline check (career per-game arc + leading indicators)\n\n"
    "Generated by `scripts/cmc_decline_study.py` from nflv_weekly/nflv_snaps/nflv_traj.\n\n"
    "```\n" + "\n".join(_out) + "\n```\n")
_print("\nwrote outputs/reports/cmc_decline.md")
