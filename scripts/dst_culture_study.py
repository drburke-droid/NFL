"""
DST study 2016-2025: is there draft-night signal in team defense?

Q1  PERSISTENCE — does last season's DST rank predict this season's? (y/y Spearman,
    top-5 repeat rate). The classic result elsewhere is "barely"; verify in OUR scoring.
Q2  CULTURE — do franchises (BAL, PIT, CHI...) sit near the top more than chance?
    Franchise mean rank over 10 seasons + permutation test of the spread.
Q3  CYCLE — do BOTTOM-quartile DSTs rebound, and does defensive draft capital
    (top-50 picks spent on defense next offseason) predict the rebound?
Q4  PRACTICAL — what actually predicts a top-5 DST season we could buy for $1?

Data: nflv_team_def (custom_pts = this league's DST scoring), nflv_draft
(defensive picks), team_draft_panel (wins). Scoring sanity-checked below.
"""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
rng = np.random.default_rng(17)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "reports", "dst_culture.md")
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))

td = pd.read_sql("SELECT * FROM nflv_team_def WHERE season BETWEEN 2016 AND 2025", con)
chk = (0.5 * td.sacks + td.ints + td.fum_rec + 6 * td.def_tds + 2 * (td.safeties + td.fg_blk + td.pat_blk))
print(f"scoring check: custom_pts vs rebuilt formula, max abs diff = {(td.custom_pts - chk).abs().max():.2f}")
td["rank"] = td.groupby("season").custom_pts.rank(ascending=False)
td = td.sort_values(["team", "season"])
td["prev_rank"] = td.groupby("team")["rank"].shift(1)
td["next_rank"] = td.groupby("team")["rank"].shift(-1)

R = ["# DST culture / cycle study (2016-2025, league scoring)", ""]

# Q1 persistence
both = td.dropna(subset=["prev_rank"])
from scipy.stats import spearmanr
rho, p = spearmanr(both.prev_rank, both["rank"])
top5 = both[both.prev_rank <= 5]
rep5 = (top5["rank"] <= 5).mean(); rep10 = (top5["rank"] <= 10).mean()
bot5 = both[both.prev_rank >= 28]
print(f"\nQ1 PERSISTENCE: y/y Spearman rho={rho:.3f} (p={p:.4f}, n={len(both)})")
print(f"  last-year top-5:  {rep5:.0%} repeat top-5, {rep10:.0%} land top-10 (n={len(top5)})")
print(f"  last-year bottom-5: {(bot5['rank']<=10).mean():.0%} jump to top-10, mean rank {bot5['rank'].mean():.1f} (n={len(bot5)})")
R += [f"## Q1 persistence: rho={rho:.2f} — top-5 repeat top-5 {rep5:.0%}, top-10 {rep10:.0%}", ""]

# Q2 franchise culture
fr = td.groupby("team").agg(n=("rank", "size"), mean_rank=("rank", "mean"),
                            top5=("rank", lambda s: (s <= 5).sum()),
                            top10=("rank", lambda s: (s <= 10).sum())).sort_values("mean_rank")
print("\nQ2 FRANCHISE TABLE (mean DST rank 2016-25, best first):")
print(fr.head(10).round(1).to_string())
print("...worst:"); print(fr.tail(5).round(1).to_string())
# permutation: is the spread of franchise means bigger than chance?
obs_sd = fr.mean_rank.std()
null = []
for _ in range(4000):
    shuf = td.copy(); shuf["team"] = rng.permutation(shuf.team.values)
    null.append(shuf.groupby("team")["rank"].mean().std())
p_cult = float((np.array(null) >= obs_sd).mean())
print(f"  spread of franchise mean-ranks: sd={obs_sd:.2f} vs chance {np.mean(null):.2f} -> p={p_cult:.4f}")
for t in ["BAL", "PIT", "CHI"]:
    r_ = fr.loc[t] if t in fr.index else None
    if r_ is not None: print(f"  {t}: mean rank {r_.mean_rank:.1f}, top-5 {int(r_.top5)}x, top-10 {int(r_.top10)}x")
R += ["## Q2 franchise means", "```", fr.round(1).to_string(), f"spread sd {obs_sd:.2f} vs chance {np.mean(null):.2f}, p={p_cult:.4f}", "```", ""]

# Q3 cycle + draft capital
dr = pd.read_sql("SELECT season, round, pick, team, position FROM nflv_draft WHERE season BETWEEN 2016 AND 2026", con)
DEFPOS = {"CB", "S", "SS", "FS", "DB", "LB", "ILB", "OLB", "EDGE", "DE", "DT", "NT", "DL"}
dr["isdef"] = dr.position.str.upper().isin(DEFPOS)
cap = dr[dr.isdef & (dr["pick"] <= 50)].groupby(["team", "season"]).size().rename("def_top50").reset_index()
# team code harmonization (nflv uses LA/LV/etc. same as team_def? check overlap)
bad = td[td["rank"] >= 24].copy()                     # bottom quartile in year t
bad["draft_season"] = bad.season + 1                  # capital spent the following offseason
bad = bad.merge(cap, left_on=["team", "draft_season"], right_on=["team", "season"],
                how="left", suffixes=("", "_d"))
bad["def_top50"] = bad.def_top50.fillna(0)
bad2 = bad.dropna(subset=["next_rank"])
hi = bad2[bad2.def_top50 >= 1]; lo = bad2[bad2.def_top50 == 0]
print(f"\nQ3 CYCLE: bottom-quartile DSTs (n={len(bad2)}) next-year mean rank {bad2.next_rank.mean():.1f}")
print(f"  with >=1 top-50 defensive pick next draft (n={len(hi)}): next rank {hi.next_rank.mean():.1f}, top-10 rate {(hi.next_rank<=10).mean():.0%}")
print(f"  with none (n={len(lo)}): next rank {lo.next_rank.mean():.1f}, top-10 rate {(lo.next_rank<=10).mean():.0%}")
# two-year horizon
bad["rank_t2"] = bad.groupby("team")["rank"].shift(-2)
R += [f"## Q3 cycle: bottom-q next-year rank {bad2.next_rank.mean():.1f}; with top-50 def pick {hi.next_rank.mean():.1f} vs without {lo.next_rank.mean():.1f}", ""]

# Q4 practical: what predicts a top-5 DST? prior rank vs prior wins
panel = pd.read_sql("SELECT team, season, wins FROM team_draft_panel", con)
td2 = td.merge(panel, on=["team", "season"], how="left").sort_values(["team", "season"])
td2["prev_wins"] = td2.groupby("team").wins.shift(1)
m = td2.dropna(subset=["prev_rank", "prev_wins"])
print(f"\nQ4: corr(prev DST rank, this rank)={m.prev_rank.corr(m['rank']):.2f} | corr(prev WINS, this DST rank)={m.prev_wins.corr(m['rank']):.2f}")
byw = m.assign(wq=pd.qcut(m.prev_wins, 3, labels=["low", "mid", "high"])).groupby("wq")["rank"]
print("  this-year DST rank by PREV-season team wins tercile:", {k: round(v, 1) for k, v in byw.mean().items()})

with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(R) + "\n")
print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
