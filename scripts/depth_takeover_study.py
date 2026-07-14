"""Depth-chart-relative skill: does a backup out-skilling a fading starter
predict the backup taking over the workload next season?

Setup per team-position-season (2011-2024, outcomes through 2025):
  starter = most role-plays at the position on that team (QB dropbacks,
  RB designed carries, WR/TE targets); backup = second-most, if he has
  enough plays for a skill estimate.
  skill_gap   = starter_true_skill - backup_true_skill   (causal, no lookahead)
  trends      = causal trend of each (this season vs last)
  TAKEOVER    = next season the backup logs MORE role-plays than the starter
                (anywhere - benched, cut, traded and retired starters count;
                rows where the backup has no next season are excluded).

Prints takeover base rates, rates by skill-gap tercile and by flag
(fading starter + stable/rising backup), then - if the signal is positive -
the 2025 blocked-riser screen: backups with stable/improving skill stuck
behind a fading-skill workload leader.
"""
import os, sqlite3, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
POS = ["QB", "RB", "WR", "TE"]

con = sqlite3.connect(DB)
sk = pd.read_sql("""SELECT s.player_id, s.position, s.season, s.plays,
                           t.skill_true_causal AS skill, t.skill_obs
                    FROM player_skill_seasons s
                    JOIN player_skill_true t USING(player_id, season)""", con)
team = pd.read_sql("""SELECT player_id, season, recent_team AS team
                      FROM nflv_season""", con).drop_duplicates(["player_id", "season"])
names = pd.read_sql("SELECT DISTINCT player_id, player_display_name AS name FROM nflv_traj", con)
ages = pd.read_sql("""SELECT gsis_id AS player_id, season,
                             season - CAST(SUBSTR(MAX(birth_date),1,4) AS INT) AS age
                      FROM nflv_rosters
                      WHERE gsis_id IS NOT NULL AND birth_date IS NOT NULL
                      GROUP BY gsis_id, season""", con)

sk = sk.sort_values(["player_id", "season"])
g = sk.groupby("player_id")
sk["trend"] = np.where((sk["season"] - g["season"].shift(1)) == 1,
                       sk["skill"] - g["skill"].shift(1), np.nan)
sk = sk.merge(team, on=["player_id", "season"], how="left").dropna(subset=["team"])
sk = sk.merge(ages, on=["player_id", "season"], how="left")

# next-season role-plays anywhere (0 if absent = out of the league)
nx = sk[["player_id", "season", "plays"]].copy()
nx.columns = ["player_id", "season", "plays_next"]
nx["season"] -= 1
sk = sk.merge(nx, on=["player_id", "season"], how="left")

# rank within team-position-season by plays
sk["rk"] = sk.groupby(["team", "position", "season"])["plays"].rank(ascending=False, method="first")
st = sk[sk["rk"] == 1].copy()
bk = sk[sk["rk"] == 2].copy()
pair = st.merge(bk, on=["team", "position", "season"], suffixes=("_s", "_b"))
pair = pair[pair["season"] <= 2024]
# backup must appear next season (can't "take over" from outside the league);
# starter absent next season counts as a takeover (retired/cut = inherited job)
pair = pair[pair["plays_next_b"].notna()]
pair["takeover"] = (pair["plays_next_b"] > pair["plays_next_s"].fillna(0)).astype(int)
pair["gap"] = pair["skill_s"] - pair["skill_b"]
pair["flag"] = ((pair["trend_s"] <= -0.03) & (pair["trend_b"] >= -0.02)
                & (pair["gap"] <= pair.groupby("position")["gap"].transform("median"))).astype(int)

print("=== Does backup-vs-starter skill predict next-season workload takeover? ===")
print(f"pairs: {len(pair):,} team-position-seasons with a skill-qualified backup\n")
print(f"{'pos':4s} {'n':>4s} {'base':>6s} | takeover by skill-gap tercile (low=backup close/better) | flag(fading S + rising B) vs rest")
for pos in POS:
    d = pair[pair["position"] == pos].copy()
    if len(d) < 60:
        print(f"{pos:4s} {len(d):4d}  -- too few qualified backups --"); continue
    d["ter"] = pd.qcut(d["gap"].rank(method="first"), 3, labels=[0, 1, 2])
    base = d["takeover"].mean()
    t = [d[d["ter"] == i]["takeover"].mean() for i in range(3)]
    f1 = d[d["flag"] == 1]["takeover"].mean(); n1 = int(d["flag"].sum())
    f0 = d[d["flag"] == 0]["takeover"].mean()
    print(f"{pos:4s} {len(d):4d} {base:6.0%} | {t[0]:.0%} / {t[1]:.0%} / {t[2]:.0%}"
          f"{'':30s}| {f1:.0%} (n={n1}) vs {f0:.0%}")

# starter-age control: is the gap signal just "old starter"?
print("\nWithin starter-age bands (all positions pooled), takeover rate by gap tercile:")
pair["ageband"] = pd.cut(pair["age_s"], [0, 26, 29, 50], labels=["<=26", "27-29", "30+"])
for band, d in pair.groupby("ageband"):
    if len(d) < 80: continue
    d = d.copy(); d["ter"] = pd.qcut(d["gap"].rank(method="first"), 3, labels=[0, 1, 2])
    t = [d[d["ter"] == i]["takeover"].mean() for i in range(3)]
    print(f"  starter {band}: n={len(d):4d}  {t[0]:.0%} / {t[1]:.0%} / {t[2]:.0%}")

# ---------------- injury-free decomposition ----------------
print("\n=== Removing starter injuries: who loses the job while HEALTHY? ===")
inj = pd.read_sql("""SELECT gsis_id AS player_id, season, week, report_status
                     FROM nflv_injuries""", con)
maxwk = np.where(inj["season"] >= 2021, 18, 17)
inj = inj[inj["week"] <= maxwk]
outwk = (inj[inj["report_status"] == "Out"]
         .groupby(["player_id", "season"])["week"].nunique()
         .rename("out_weeks").reset_index())
outwk["season"] -= 1  # key next-season injuries to current pair row
pair = pair.merge(outwk.rename(columns={"player_id": "player_id_s",
                                        "out_weeks": "out_next_s"}),
                  on=["player_id_s", "season"], how="left")
pair["out_next_s"] = pair["out_next_s"].fillna(0)

pair["category"] = np.select(
    [pair["takeover"] == 0,
     pair["plays_next_s"].fillna(0) == 0,
     pair["out_next_s"] >= 2],
    ["held job", "starter gone (cut/retired/left)", "injury-driven takeover"],
    default="HEALTHY takeover (benched/phased out)")

print("\nAll pairs, outcome mix by whether the backup OUT-SKILLED the starter (gap <= 0):")
for lab, m in [("backup out-skilled starter (gap<=0)", pair["gap"] <= 0),
               ("starter clearly better (gap>0)", pair["gap"] > 0)]:
    d = pair[m]
    mix = d["category"].value_counts(normalize=True)
    cnt = d["category"].value_counts()
    print(f"  {lab} (n={len(d)}):")
    for c in ["held job", "HEALTHY takeover (benched/phased out)",
              "injury-driven takeover", "starter gone (cut/retired/left)"]:
        print(f"    {c:40s} {cnt.get(c,0):4d}  ({mix.get(c,0):.0%})")

print("\nHEALTHY-ONLY test: drop next seasons where the starter missed 2+ weeks or left.")
hp = pair[(pair["out_next_s"] <= 1) & (pair["plays_next_s"].fillna(0) > 0)].copy()
print(f"remaining pairs: {len(hp):,}  (healthy takeover = benching/role erosion only)\n")
print(f"{'pos':4s} {'n':>4s} {'base':>6s} | healthy takeover by skill-gap tercile | fading-S + rising-B flag vs rest")
for pos in POS:
    d = hp[hp["position"] == pos].copy()
    if len(d) < 50:
        print(f"{pos:4s} {len(d):4d}  -- too few --"); continue
    d["ter"] = pd.qcut(d["gap"].rank(method="first"), 3, labels=[0, 1, 2])
    t = [d[d["ter"] == i]["takeover"].mean() for i in range(3)]
    f1 = d[d["flag"] == 1]["takeover"].mean(); n1 = int(d["flag"].sum())
    f0 = d[d["flag"] == 0]["takeover"].mean()
    print(f"{pos:4s} {len(d):4d} {d['takeover'].mean():6.0%} | {t[0]:.0%} / {t[1]:.0%} / {t[2]:.0%}"
          f"{'':16s}| {f1:.0%} (n={n1}) vs {f0:.0%}")

# ---------------- 2025 blocked-riser screen ----------------
print("\n=== 2025 screen: stable/improving backups blocked by a fading-skill starter ===")
cur = sk[sk["season"] == 2025]
s25 = cur[cur["rk"] == 1]; b25 = cur[cur["rk"] == 2]
p25 = s25.merge(b25, on=["team", "position"], suffixes=("_s", "_b"))
p25["gap"] = p25["skill_s"] - p25["skill_b"]
scr = p25[(p25["trend_s"] <= -0.03) & (p25["trend_b"] >= -0.02) & (p25["gap"] <= 0.25)]
scr = scr.merge(names.rename(columns={"player_id": "player_id_s", "name": "starter"}), on="player_id_s", how="left")
scr = scr.merge(names.rename(columns={"player_id": "player_id_b", "name": "backup"}), on="player_id_b", how="left")
scr = scr.sort_values(["position", "gap"])
cols = ["position", "team", "starter", "age_s", "skill_s", "trend_s",
        "backup", "age_b", "skill_b", "trend_b", "gap", "plays_s", "plays_b"]
with pd.option_context("display.width", 200):
    print(scr[cols].round(2).to_string(index=False))
print(f"\n{len(scr)} blocked risers. Criteria: starter causal trend <= -0.03, "
      "backup trend >= -0.02, skill gap <= +0.25z (backup within striking distance or better).")
con.close()
