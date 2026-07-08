"""
Does news sentiment give a week-to-week edge on PLAYER PROPS?

User hypothesis: positive sentiment -> public piles on the OVER -> books shade the line
up -> the UNDER becomes +EV (and symmetrically for negative sentiment).

Data: outputs/prop_ev_backtest.pkl (17,178 props 2024-25 with consensus line `mline`,
no-vig over prob `novig`, best decimal prices, actual outcome `won_over`, and L6 trailing
stats) x nflv_gdelt_bq (monthly GDELT tone/volume per player). Sentiment = PRIOR calendar
month (weeks 1-4 <- Aug, 5-8 <- Sep, 9-13 <- Oct, 14-17 <- Nov, 18 <- Dec).

Three-link test of the causal chain:
  L1  tone -> line inflation?      (mline vs the player's own L6 trailing stat)
  L2  tone -> over under-delivers? (won_over - novig, i.e. residual AFTER the market prob)
  L3  bankroll: ROI of betting UNDER at best price when tone is high (+ symmetric overs)
Report: outputs/reports/sentiment_props.md
"""
import os, re, sqlite3
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_out = []; _print = print
def print(*a, **k):
    s = " ".join(str(x) for x in a); _out.append(s); _print(s, **k)

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"(jr|sr|ii|iii|iv|v)", "",
                        re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
d = pd.read_pickle(os.path.join(ROOT, "outputs", "prop_ev_backtest.pkl"))
d = d.dropna(subset=["won_over", "novig", "mline"]).copy()
d["nm"] = d.player_name.map(norm)                                     # frame nm is raw display names

# prior-month sentiment
con = sqlite3.connect(os.path.join(ROOT, "db", "nfl_odds.db"))
gd = pd.read_sql("SELECT name nm, ym, articles, avg_tone FROM nflv_gdelt_bq WHERE source='gkg'", con)
gd["y"] = gd.ym.str[:4].astype(int); gd["m"] = gd.ym.str[4:].astype(int)
base = gd[gd.m.between(1, 5)].groupby(["nm", "y"]).avg_tone.mean().rename("base_tone").reset_index()
d["month"] = np.select([d.week <= 4, d.week <= 8, d.week <= 13, d.week <= 17], [9, 10, 11, 12], 1)
d["sy"] = d.season; d.loc[d.month == 1, "sy"] = d.season + 1          # week 18 in January
prior = gd.rename(columns={"y": "sy"})[["nm", "sy", "m", "articles", "avg_tone"]].copy()
prior["month"] = prior.m + 1
prior.loc[prior.m == 12, "month"] = 1
prior.loc[prior.m == 12, "sy"] += 1
d = d.merge(prior[["nm", "sy", "month", "articles", "avg_tone"]]
            .rename(columns={"articles": "prev_n", "avg_tone": "prev_tone"}), on=["nm", "sy", "month"], how="left")
d = d.merge(base.rename(columns={"y": "season"}), on=["nm", "season"], how="left")
d["tone_shift"] = d.prev_tone - d.base_tone
d["resid"] = d.won_over - d.novig                                     # over result minus market prob

ok = d.dropna(subset=["prev_tone"]).copy()
print(f"props with prior-month sentiment: {len(ok)} of {len(d)} ({ok.season.min()}-{ok.season.max()})")
print(f"baseline: overs hit {ok.won_over.mean():.1%} vs market novig {ok.novig.mean():.1%} "
      f"(resid {ok.resid.mean():+.3f})\n")

# ---- L1: does tone inflate the line vs the player's own trailing production? ----
L6 = {"player_receptions": "l6_receptions", "player_reception_yds": "l6_receiving_yards",
      "player_rush_yds": "l6_rushing_yards", "player_rush_attempts": "l6_carries",
      "player_pass_completions": "l6_completions", "player_pass_yds": "l6_passing_yards"}
ok["l6stat"] = [r[L6[m]] if m in L6 else np.nan for m, r in zip(ok.market, ok.to_dict("records"))]
li = ok.dropna(subset=["l6stat"]); li = li[li.l6stat > 0].copy()
li["line_rel"] = (li.mline - li.l6stat) / li.l6stat
r, p = stats.spearmanr(li.prev_tone, li.line_rel)
r2, p2 = stats.spearmanr(li.tone_shift.fillna(0), li.line_rel)
print("L1 — tone -> line inflation (line vs own L6 trailing stat):")
print(f"   Spearman prev_tone vs line_rel: r={r:+.3f} (p={p:.4f}) | tone_shift: r={r2:+.3f} (p={p2:.4f})")
for lab, sub in (("tone < -2", li[li.prev_tone < -2]), ("-2..0", li[li.prev_tone.between(-2, 0)]),
                 ("0..+2", li[li.prev_tone.between(0, 2)]), ("tone > +2", li[li.prev_tone > 2])):
    if len(sub) > 30: print(f"   {lab:<10} n={len(sub):>5}  line vs L6: {sub.line_rel.mean():+.1%}")

# ---- L2: does tone predict the over beating/missing the market's own probability? ----
print("\nL2 — tone -> over result MINUS market prob (resid = won_over - novig):")
for lab, sub in (("tone < -2", ok[ok.prev_tone < -2]), ("-2..0", ok[ok.prev_tone.between(-2, 0)]),
                 ("0..+2", ok[ok.prev_tone.between(0, 2)]), ("tone > +2", ok[ok.prev_tone > 2])):
    if len(sub) > 30:
        se = sub.resid.std() / np.sqrt(len(sub))
        print(f"   {lab:<10} n={len(sub):>5}  resid {sub.resid.mean():+.3f} (±{2*se:.3f})")
r, p = stats.spearmanr(ok.prev_tone, ok.resid)
rs, ps = stats.spearmanr(ok.dropna(subset=['tone_shift']).tone_shift, ok.dropna(subset=['tone_shift']).resid)
print(f"   Spearman prev_tone vs resid: r={r:+.3f} (p={p:.4f}) | tone_shift: r={rs:+.3f} (p={ps:.4f})")
hv = ok[ok.prev_n >= 100]
r, p = stats.spearmanr(hv.prev_tone, hv.resid)
print(f"   high-volume only (prev_n>=100, n={len(hv)}): r={r:+.3f} (p={p:.4f})")

# ---- L3: bankroll simulation at best available prices ----
print("\nL3 — ROI simulations (flat 1u, best price across books):")
def roi(sub, side):
    if not len(sub): return 0, 0
    if side == "under":
        ret = np.where(sub.won_over == 0, sub.best_under - 1, -1.0)
    else:
        ret = np.where(sub.won_over == 1, sub.best_over - 1, -1.0)
    return ret.mean(), len(sub)
for lab, sub, side in (
    ("ALL unders (juice baseline)", ok, "under"),
    ("ALL overs (juice baseline)", ok, "over"),
    ("UNDER when tone > 0", ok[ok.prev_tone > 0], "under"),
    ("UNDER when tone > +1 & vol>=100", ok[(ok.prev_tone > 1) & (ok.prev_n >= 100)], "under"),
    ("UNDER when tone_shift > +1", ok[ok.tone_shift > 1], "under"),
    ("OVER  when tone < -2", ok[ok.prev_tone < -2], "over"),
    ("OVER  when tone < -2 & vol>=100", ok[(ok.prev_tone < -2) & (ok.prev_n >= 100)], "over"),
    ("UNDER when tone < -2", ok[ok.prev_tone < -2], "under"),
    ("UNDER when tone < -2 & vol>=100", ok[(ok.prev_tone < -2) & (ok.prev_n >= 100)], "under"),
    ("UNDER when tone < -3 & vol>=100", ok[(ok.prev_tone < -3) & (ok.prev_n >= 100)], "under"),
):
    m, n = roi(sub, side)
    print(f"   {lab:<34} n={n:>5}  ROI {m:+.1%}")

# ---- robustness: the winning cell (UNDER, tone<-2, vol>=100) by season and market ----
print("robustness of UNDER tone<-2 & vol>=100:")
cell = ok[(ok.prev_tone < -2) & (ok.prev_n >= 100)]
for ssn in sorted(ok.season.unique()):
    for lab, sub in (("cell", cell[cell.season == ssn]), ("all-unders", ok[ok.season == ssn])):
        m, n = roi(sub, "under")
        print(f"   {ssn} UNDER {lab:<11} n={n:>5}  ROI {m:+.1%}")
for mk, sub in cell.groupby("market"):
    m, n = roi(sub, "under")
    print(f"   cell {mk:<28} n={n:>4}  ROI {m:+.1%}")

open(os.path.join(ROOT, "outputs", "reports", "sentiment_props.md"), "w", encoding="utf-8").write(
    "# Sentiment vs player props — is there a weekly betting edge?\n\n"
    "Generated by `scripts/sentiment_props_study.py`: prop_ev_backtest.pkl (17k props\n"
    "2024-25, consensus lines + no-vig probs + results) x nflv_gdelt_bq monthly tone.\n\n"
    "```\n" + "\n".join(_out) + "\n```\n")
_print("\nwrote outputs/reports/sentiment_props.md")
