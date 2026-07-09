"""
GEG (GDELT Global Entity Graph = Google NLP annotations) as a PRECISION upgrade for the
August camp-riser buzz signal: count only articles where the player is a CENTRAL entity
(salience), instead of any raw name mention. No sentiment in GEG — this is about volume
quality. Coverage: mid-2019+ -> Augusts 2019-2025, ~120 GB scan, into db nflv_geg_aug.

Test: cheap-pool (dart) hit-rate lift by top-decile August buzz, same players and seasons
(2020-2025), three ways: (a) GEG high-salience doc count, (b) raw GKG persons volume spike,
(c) Wikipedia pageview spike (the production signal). Appends to LATE_BREAKOUTS.md.
"""
import os, re, json, sqlite3
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
if not con.execute("SELECT name FROM sqlite_master WHERE name='nflv_geg_aug'").fetchone():
    from google.cloud import bigquery
    import pydata_google_auth
    creds = pydata_google_auth.get_user_credentials(["https://www.googleapis.com/auth/bigquery"])
    bq = bigquery.Client(project=json.load(open(os.path.join(ROOT, "config", "gdelt_bq.json")))["project"],
                         credentials=creds)
    rows = con.execute("SELECT DISTINCT player_display_name FROM nflv_season WHERE season>=2012").fetchall()
    names = sorted(set(norm(r[0]) for r in rows if r[0] and len(norm(r[0]).split()) >= 2))
    con.execute("""CREATE TABLE nflv_geg_aug (name TEXT, y INTEGER, docs INTEGER,
                   hi_docs INTEGER, mentions INTEGER, avg_sal REAL, PRIMARY KEY(name,y))""")
    for yr in range(2019, 2026):
        sql = f"""
          SELECT LOWER(ent.nameNorm) person, COUNT(*) docs,
                 SUM(IF(ent.avgSalience >= 0.1, 1, 0)) hi_docs,
                 SUM(ent.numMentions) mentions, AVG(ent.avgSalience) avg_sal
          FROM `gdelt-bq.gdeltv2.geg_g1`, UNNEST(entities) ent
          WHERE date >= TIMESTAMP('{yr}-08-01') AND date < TIMESTAMP('{yr}-09-01')
            AND ent.type='PROPER' AND LOWER(ent.nameNorm) IN UNNEST(@names)
          GROUP BY person"""
        job = bq.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("names", "STRING", names)]))
        res = [(r["person"], yr, r["docs"], r["hi_docs"], r["mentions"], r["avg_sal"]) for r in job.result()]
        con.executemany("INSERT OR REPLACE INTO nflv_geg_aug VALUES (?,?,?,?,?,?)", res)
        con.commit()
        print(f"Aug {yr}: {len(res)} players (billed {job.total_bytes_billed/1e9:.1f} GB)")
geg = pd.read_sql("SELECT name nm, y, docs, hi_docs, mentions, avg_sal FROM nflv_geg_aug", con)
print(f"GEG August table: {len(geg)} player-years, {geg.y.min()}-{geg.y.max()}")

# ---- dart-pool comparison ----
df = pd.read_pickle(os.path.join(ROOT, "outputs", "models", "late_breakout_frame.pkl"))
df = df[(df.season >= 2020) & (df.cheap == 1) & df.position.isin(["QB", "RB", "WR", "TE"])].copy()
df["nm"] = df.name.map(norm)
df = df.merge(geg.rename(columns={"y": "season"}), on=["nm", "season"], how="left")
for c in ("docs", "hi_docs", "mentions"): df[c] = df[c].fillna(0)

# (b) raw GKG persons August volume spike (from nflv_gdelt_bq)
gd = pd.read_sql("SELECT name nm, ym, articles FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["y"] = gd.ym.str[:4].astype(int); gd["m"] = gd.ym.str[4:].astype(int)
aug = gd[gd.m == 8].rename(columns={"articles": "gkg_aug", "y": "season"})[["nm", "season", "gkg_aug"]]
base = gd[gd.m.between(1, 5)].groupby(["nm", "y"]).articles.median().rename("gkg_base").reset_index().rename(columns={"y": "season"})
df = df.merge(aug, on=["nm", "season"], how="left").merge(base, on=["nm", "season"], how="left")
df["gkg_spike"] = np.log((df.gkg_aug.fillna(0) + 10) / (df.gkg_base.fillna(0) + 10))

# (c) wiki pageview spike (nflv_wiki_buzz is player_id x ym monthly views)
wb = pd.read_sql("SELECT player_id, ym, views FROM nflv_wiki_buzz", con)
wb["y"] = wb.ym.str[:4].astype(int); wb["m"] = wb.ym.str[4:].astype(int)
wa = wb[wb.m == 8].groupby(["player_id", "y"]).views.sum().rename("wiki_aug").reset_index().rename(columns={"y": "season"})
wbase = wb[wb.m.between(1, 5)].groupby(["player_id", "y"]).views.median().rename("wiki_base").reset_index().rename(columns={"y": "season"})
df = df.merge(wa, on=["player_id", "season"], how="left").merge(wbase, on=["player_id", "season"], how="left")
df["wiki_spike"] = np.log((df.wiki_aug.fillna(0) + 10) / (df.wiki_base.fillna(0) + 10))

print(f"\ndart pool 2020-2025: n={len(df)}, base hit rate {df.hit.mean():.1%}")
print(f"{'signal':<26}{'top decile hit':>15}{'lift':>7}{'top 3% hit':>12}{'lift':>7}")
cov = {"hi_docs": df.docs > 0, "docs": df.docs > 0,
       "gkg_spike": df.gkg_aug.notna(), "wiki_spike": df.wiki_aug.notna()}
for col, lab in (("hi_docs", "GEG high-salience docs"), ("docs", "GEG any-mention docs"),
                 ("gkg_spike", "GKG volume spike"), ("wiki_spike", "wiki pageview spike")):
    sub = df[cov[col]].copy()                                # rank only among covered players
    sub["pct"] = sub.groupby("season")[col].rank(pct=True)
    d10 = sub[sub.pct >= 0.9]; d3 = sub[sub.pct >= 0.97]; df_ = sub
    print(f"{lab:<26}{d10.hit.mean():>15.1%}{d10.hit.mean()/max(df_.hit.mean(),1e-9):>6.1f}x"
          f"{d3.hit.mean():>12.1%}{d3.hit.mean()/max(df_.hit.mean(),1e-9):>6.1f}x  (pool n={len(df_)}, base {df_.hit.mean():.1%})")
# combo: GEG hi-salience AND wiki spike both top-decile
df["p1"] = df.groupby("season").hi_docs.rank(pct=True)
df["p2"] = df.groupby("season").wiki_spike.rank(pct=True)
both = df[(df.p1 >= 0.9) & (df.p2 >= 0.9)]
print(f"{'BOTH GEG-sal + wiki top-10%':<26}{both.hit.mean():>15.1%}{both.hit.mean()/df.hit.mean():>6.1f}x   (n={len(both)})")
puka = df[(df.nm == "puka nacua") & (df.season == 2023)]
if len(puka):
    r = puka.iloc[0]
    print(f"\nPuka 2023 check: GEG hi_docs={r.hi_docs:.0f} (pct {r.p1:.2f}) wiki spike pct {r.p2:.2f}")

with open(os.path.join(ROOT, "outputs", "models", "LATE_BREAKOUTS.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## 9. GEG salience-filtered buzz (`geg_buzz_study.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/models/LATE_BREAKOUTS.md")
