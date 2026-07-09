"""
TV news buzz (GDELT iatvgkg: Internet Archive TV captions — what actually airs) as the
PUBLIC-visibility proxy. Two questions:
  A. props: does TV fame predict over-shading (the original public-bias hypothesis, with a
     better proxy for what casual bettors see)? resid = won_over − novig by TV-mention decile.
  B. auction: do this league's owners overpay TV-famous players beyond the price curve?
     (overpay residual vs prior-August TV mentions, 2023-25 winning bids)
Fetches player×month TV mentions + tone into nflv_tv_buzz (one ~small scan), then analyzes.
"""
import os, re, sqlite3, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "",
                       re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
have = con.execute("SELECT name FROM sqlite_master WHERE name='nflv_tv_buzz'").fetchone()
if not have:
    from google.cloud import bigquery
    import pydata_google_auth
    creds = pydata_google_auth.get_user_credentials(["https://www.googleapis.com/auth/bigquery"])
    bq = bigquery.Client(project=json.load(open(os.path.join(ROOT, "config", "gdelt_bq.json")))["project"],
                         credentials=creds)
    rows = con.execute("SELECT DISTINCT player_display_name FROM nflv_season WHERE season>=2012").fetchall()
    names = sorted(set(nrm(r[0]) for r in rows if r[0] and len(nrm(r[0]).split()) >= 2))
    sql = """
      SELECT person, CAST(CAST(DATE/100000000 AS INT64) AS STRING) AS ym, COUNT(*) n,
             AVG(CAST(SPLIT(V2Tone,',')[SAFE_OFFSET(0)] AS FLOAT64)) avg_tone
      FROM `gdelt-bq.gdeltv2.iatvgkg`, UNNEST(SPLIT(LOWER(Persons),';')) person
      WHERE person IN UNNEST(@names)
      GROUP BY person, ym"""
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("names", "STRING", names)]))
    res = [(r["person"], r["ym"], r["n"], r["avg_tone"]) for r in job.result()]
    con.execute("CREATE TABLE nflv_tv_buzz (name TEXT, ym TEXT, mentions INTEGER, avg_tone REAL, PRIMARY KEY(name,ym))")
    con.executemany("INSERT OR REPLACE INTO nflv_tv_buzz VALUES (?,?,?,?)", res)
    con.commit()
    print(f"fetched TV buzz: {len(res)} player-months (billed {job.total_bytes_billed/1e9:.1f} GB)")
tv = pd.read_sql("SELECT name nm, ym, mentions, avg_tone FROM nflv_tv_buzz", con)
tv["y"] = tv.ym.str[:4].astype(int); tv["m"] = tv.ym.str[4:].astype(int)
print(f"TV buzz table: {len(tv)} rows, {tv.nm.nunique()} players, {tv.ym.min()}-{tv.ym.max()}")

# ---- A. props: TV fame vs over-shading ----
d = pd.read_pickle(os.path.join(ROOT, "outputs", "prop_ev_backtest.pkl"))
d = d.dropna(subset=["won_over", "novig"]).copy()
d["nm"] = d.player_name.map(nrm)
d["month"] = np.select([d.week <= 4, d.week <= 8, d.week <= 13, d.week <= 17], [9, 10, 11, 12], 1)
d["sy"] = d.season; d.loc[d.month == 1, "sy"] = d.season + 1
pr = tv.copy(); pr["month"] = pr.m + 1
pr.loc[pr.m == 12, "month"] = 1
pr["sy"] = pr.y; pr.loc[pr.m == 12, "sy"] += 1
d = d.merge(pr[["nm", "sy", "month", "mentions", "avg_tone"]]
            .rename(columns={"mentions": "tv_n", "avg_tone": "tv_tone"}), on=["nm", "sy", "month"], how="left")
d["tv_n"] = d.tv_n.fillna(0)
d["resid"] = d.won_over - d.novig
print(f"\nA. props (n={len(d)}): resid (over result − market prob) by prior-month TV mentions:")
d["bucket"] = pd.cut(d.tv_n, [-1, 0, 5, 20, 100, 1e9], labels=["0", "1-5", "6-20", "21-100", "100+"])
for b, sub in d.groupby("bucket", observed=True):
    se = sub.resid.std() / np.sqrt(len(sub))
    print(f"   TV {b:<8} n={len(sub):>6}  resid {sub.resid.mean():+.3f} (±{2*se:.3f})")
hi = d[d.tv_n > 0]
r, p = stats.spearmanr(hi.tv_n, hi.resid)
print(f"   Spearman TV mentions vs resid (mentions>0): r={r:+.3f} (p={p:.4f})")
u = d[d.tv_n >= 21]
ret = np.where(u.won_over == 0, u.best_under - 1, -1.0)
retb = np.where(d.won_over == 0, d.best_under - 1, -1.0)
print(f"   UNDER on TV21+ players: n={len(u)} ROI {ret.mean():+.1%}  (all-unders {retb.mean():+.1%})")

# ---- B. auction: TV-fame overpay in THIS league ----
drafts = pd.read_csv(os.path.join(ROOT, "outputs", "espn_drafts.csv"))
drafts = drafts[~drafts.keeper.astype(str).str.lower().isin(["true", "1"])]
drafts["bid"] = drafts.bid.fillna(0).clip(lower=1)
drafts = drafts[drafts.pos.isin(["QB", "RB", "WR", "TE"])]
per_team = drafts.groupby("season").apply(lambda x: x.bid.sum() / x.owner.nunique(), include_groups=False)
scale = per_team.max() / per_team
buys = []
for (s, pos), sub in drafts.groupby(["season", "pos"]):
    sub = sub.sort_values("bid", ascending=False).reset_index(drop=True)
    arr = (sub.bid * scale[s]).values
    for rk, (_, r) in enumerate(sub.iterrows()):
        buys.append({"season": s, "pos": pos, "nm": nrm(r.player), "sc": r.bid * scale[s], "rk": rk})
bdf = pd.DataFrame(buys)
curve = bdf.groupby(["pos", "rk"]).sc.mean().rename("curve").reset_index()
bdf = bdf.merge(curve, on=["pos", "rk"])
bdf["overpay"] = bdf.sc - bdf.curve
aug = tv[tv.m == 8].rename(columns={"y": "season"})[["nm", "season", "mentions"]]
bdf = bdf.merge(aug, on=["nm", "season"], how="left")
bdf["mentions"] = bdf.mentions.fillna(0)
print(f"\nB. auction overpay vs prior-August TV mentions (n={len(bdf)} buys 2023-25):")
bdf["tvb"] = pd.cut(bdf.mentions, [-1, 0, 5, 20, 1e9], labels=["0", "1-5", "6-20", "21+"])
for b, sub in bdf.groupby("tvb", observed=True):
    print(f"   TV {b:<6} n={len(sub):>4}  overpay vs rank curve {sub.overpay.mean():+.2f}$")
hi = bdf[bdf.mentions > 0]
r, p = stats.spearmanr(hi.mentions, hi.overpay)
print(f"   Spearman TV mentions vs overpay (mentions>0): r={r:+.3f} (p={p:.4f})")

with open(os.path.join(ROOT, "outputs", "reports", "tone_signal.md"), "a", encoding="utf-8") as fp:
    fp.write("\n\n## TV buzz (`tv_buzz_study.py`)\n\n```\n" + "\n".join(_out) + "\n```\n")
_print("\nappended to outputs/reports/tone_signal.md")
