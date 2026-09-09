"""What weight should the DK line get vs the FFA stat line?  Uses the 2023-25 closing-line
frames (data/props_frames) for reception yds, receptions, rush yds: MAE of the stat vs actual
for FFA alone, the line alone, and blends. Output: outputs/reports/market_blend_weight.md"""
import os, re, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def norm(s):
    s = str(s).lower().strip(); s = re.sub(r"[.'’-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s); return re.sub(r"\s+", " ", s)
rows = []
for f in glob.glob(os.path.join(ROOT, "data", "ffanalytics", "FFAn_weekly", "raw_stats_*_wk*.csv")):
    s_, w_ = map(int, re.search(r"raw_stats_(\d{4})_wk(\d+)", f).groups())
    if s_ >= 2026: continue
    d = pd.read_csv(f, na_values=["NA"]); d = d[d.position.isin(["QB", "RB", "WR", "TE"])]
    d["nname"] = d.player.map(norm); d["season"], d["week"] = s_, w_
    rows.append(d[["nname", "season", "week", "rec_yds", "rec", "rush_yds"]].drop_duplicates("nname"))
ffa = pd.concat(rows)
L = ["# DK line vs FFA stat: blend weight (2023-25 closing lines)\n", "| market | n | FFA MAE | line MAE | best w (on line) | blend MAE | w=0.6 MAE |", "|---|---|---|---|---|---|---|"]
for mk, col in (("player_reception_yds", "rec_yds"), ("player_receptions", "rec"), ("player_rush_yds", "rush_yds")):
    h = pd.read_parquet(os.path.join(ROOT, "data", "props_frames", f"props_{mk}.parquet"))
    h["nname"] = h.player.map(norm)
    m = h.merge(ffa[["nname", "season", "week", col]], on=["nname", "season", "week"]).dropna(subset=[col, "actual_ppr", "baseline_proj"])
    ws = np.arange(0, 1.01, 0.1)
    maes = [np.abs(m.actual_ppr - (w * m.baseline_proj + (1 - w) * m[col])).mean() for w in ws]
    best = ws[int(np.argmin(maes))]
    L.append(f"| {col} | {len(m):,} | {maes[0]:.3f} | {maes[-1]:.3f} | {best:.1f} | {min(maes):.3f} | {maes[6]:.3f} |")
    # by season
    for s in (2023, 2024, 2025):
        ms = m[m.season == s]
        if len(ms) < 200: continue
        mm = [np.abs(ms.actual_ppr - (w * ms.baseline_proj + (1 - w) * ms[col])).mean() for w in ws]
        L.append(f"|   {s} | {len(ms):,} | {mm[0]:.3f} | {mm[-1]:.3f} | {ws[int(np.argmin(mm))]:.1f} | {min(mm):.3f} | {mm[6]:.3f} |")
open(os.path.join(ROOT, "outputs", "reports", "market_blend_weight.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
