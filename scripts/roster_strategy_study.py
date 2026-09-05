"""Roster-construction backtest 2016-2025: is filling the WHOLE roster with
'top-projected players priced below their position's average auction rate'
(value-fill) optimal, vs stars-and-scrubs / optimizer / anchor hybrids?

League: 12-team, $200, starters QB/2RB/2WR/TE/FLEX (K/DST ignored, $2 reserved).
Model roster = 12 skill players, budget $190, min $1 each.
Prices = FFA AAV (market). Projections = league-scored FFA (ex-ante only).
Realized = actual season points (PPR + 6pt pass TD proxy), ex-post optimal
lineup from the 12-man roster — bench upside counts via replacement.
Selection = greedy fill + 2-opt swap improvement on PROJECTED lineup points.
"""
import sqlite3, os
import numpy as np, pandas as pd

root = r"C:\Users\drbur\Documents\GitHub\NFL"
con = sqlite3.connect(os.path.join(root, "db", "nfl_odds.db"))

proj = pd.read_sql("""SELECT season, player_id, player, position, ffa_points, ffa_aav
                      FROM nflv_ffa_league WHERE season BETWEEN 2016 AND 2025
                      AND position IN ('QB','RB','WR','TE')""", con)
real = pd.read_sql("""SELECT season, player_id, fantasy_points_ppr + 2.0*passing_tds AS pts
                      FROM nflv_season WHERE season BETWEEN 2016 AND 2025""", con)
real = real.groupby(["season", "player_id"], as_index=False).pts.sum()
d = proj.merge(real, on=["season", "player_id"], how="left").fillna({"pts": 0.0})
d["price"] = d.ffa_aav.fillna(1).clip(lower=1).round().astype(int)
d = d.dropna(subset=["ffa_points"])

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}          # + 1 FLEX (RB/WR/TE)
FLEXP = {"RB", "WR", "TE"}
NROSTER, BUDGET = 12, 190

def lineup_points(players, col):
    """Best QB/2RB/2WR/TE/FLEX total from a roster list of (pos, value) rows."""
    df = pd.DataFrame(players, columns=["pos", "v"]).sort_values("v", ascending=False)
    used, total = [], 0.0
    need = dict(SLOTS)
    for i, r in df.iterrows():
        if need.get(r.pos, 0) > 0:
            need[r.pos] -= 1; total += r.v; used.append(i)
    flex = df[~df.index.isin(used) & df.pos.isin(FLEXP)]
    if len(flex): total += flex.v.iloc[0]
    return total

def eval_roster(rows, col):
    return lineup_points(list(zip(rows.position, rows[col])), col)

def build(pool, price_ok, seed_idx=None):
    """Greedy: fill starter needs by projection among price-eligible, then bench;
    then 2-opt swaps to improve projected lineup within budget. Optional seed_idx
    are bought first and protected from swaps; any unfilled slots at the end are
    filled with $1 bodies (real drafts never leave slots empty)."""
    pool = pool.copy()
    elig = pool[pool.apply(price_ok, axis=1)].sort_values("ffa_points", ascending=False)
    roster_idx, spent = [], 0
    need = dict(SLOTS); need["FLEX"] = 1
    seed_idx = list(seed_idx or [])
    for i in seed_idx:
        r = pool.loc[i]
        if need.get(r.position, 0) > 0: need[r.position] -= 1
        elif need["FLEX"] > 0 and r.position in FLEXP: need["FLEX"] -= 1
        roster_idx.append(i); spent += r.price
    def rem_slots(): return NROSTER - len(roster_idx)
    # starters
    for _, r in elig.iterrows():
        if len(roster_idx) >= 7: break
        want = need.get(r.position, 0) > 0 or (need["FLEX"] > 0 and r.position in FLEXP)
        if not want: continue
        if spent + r.price > BUDGET - (rem_slots() - 1): continue
        if need.get(r.position, 0) > 0: need[r.position] -= 1
        else: need["FLEX"] -= 1
        roster_idx.append(r.name); spent += r.price
    # bench: best projection that fits
    for _, r in elig.iterrows():
        if len(roster_idx) >= NROSTER: break
        if r.name in roster_idx: continue
        if spent + r.price > BUDGET - (rem_slots() - 1): continue
        roster_idx.append(r.name); spent += r.price
    # backfill any open slots with $1 bodies by projection
    if len(roster_idx) < NROSTER:
        ones = pool[(pool.price == 1) & ~pool.index.isin(roster_idx)].sort_values(
            "ffa_points", ascending=False)
        for _, r in ones.iterrows():
            if len(roster_idx) >= NROSTER: break
            if spent + 1 > BUDGET: break
            roster_idx.append(r.name); spent += 1
    # 2-opt improvement on projected lineup
    improved = True
    while improved:
        improved = False
        cur = pool.loc[roster_idx]
        base = eval_roster(cur, "ffa_points")
        for out_i in list(roster_idx):
            if out_i in seed_idx: continue
            for _, cand in elig.head(150).iterrows():
                if cand.name in roster_idx: continue
                new_spent = spent - pool.loc[out_i, "price"] + cand.price
                if new_spent > BUDGET: continue
                trial = [i for i in roster_idx if i != out_i] + [cand.name]
                val = eval_roster(pool.loc[trial], "ffa_points")
                if val > base + 1e-9:
                    roster_idx, spent, base = trial, new_spent, val
                    improved = True
                    break
            if improved: break
    return pool.loc[roster_idx], spent

STARTER_TIER = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}   # cheap_stars.md class

results = []
detail = {}
for yr, g in d.groupby("season"):
    g = g.reset_index(drop=True)
    # cheap_stars.md conventions: pool = top 168 by AAV, positional MEAN inside it
    pool168 = g.nlargest(168, "ffa_aav")
    pos_avg = pool168.groupby("position").price.mean().to_dict()
    g["pos_rank"] = g.groupby("position").ffa_points.rank(ascending=False, method="first")
    g["in_class"] = g.apply(
        lambda r: 0.5 * pos_avg.get(r.position, 10) <= r.price < pos_avg.get(r.position, 10)
        and r.pos_rank <= STARTER_TIER[r.position], axis=1)

    strategies = {
        # the user's strategy: NEVER pay >= position-average price
        "value_fill": lambda r, pa=pos_avg: r.price <= pa.get(r.position, 10),
        # optimizer: no price constraint (max projected lineup)
        "optimizer": lambda r: True,
        # stars & scrubs: stars >=$40 or scrubs <=$5, nothing mid
        "stars_scrubs": lambda r: r.price >= 40 or r.price <= 5,
        # anchor + value: one lane >= $50 allowed, else below pos-average
        # (implemented as: price<=posavg OR price>=50)
        "anchor_value": lambda r, pa=pos_avg: r.price <= pa.get(r.position, 10) or r.price >= 50,
        # mid-band only ($10-25): the price-band study's sweet spot scaled up
        "mid_band": lambda r: 6 <= r.price <= 25,
        # softer cap: allowed up to 1.25x position average
        "value_125": lambda r, pa=pos_avg: r.price <= 1.25 * pa.get(r.position, 10),
        # cheap_stars.md class ONLY (50-100% of pos avg + starter-tier proj);
        # unfilled slots become $1 bodies
        "class_fill": lambda r: bool(r.in_class),
        # class + 1-2 anchors >=$50
        "class_anchor": lambda r: bool(r.in_class) or r.price >= 50,
    }
    seeds = {}
    # the report's actual advice: force 4 class buys (~$40), rest unconstrained.
    # Seed by VORP (proj minus positional starter-tier replacement), max 2 per
    # position — raw projection would seed 4 QBs.
    repl = {p: g[g.position == p].ffa_points.nlargest(STARTER_TIER[p]).min()
            for p in STARTER_TIER}
    cls = g[g.in_class].copy()
    cls["vorp"] = cls.ffa_points - cls.position.map(repl)
    picked, per_pos = [], {}
    for i, r in cls.sort_values("vorp", ascending=False).iterrows():
        if per_pos.get(r.position, 0) >= 2: continue
        picked.append(i); per_pos[r.position] = per_pos.get(r.position, 0) + 1
        if len(picked) == 4: break
    seeds["class_x4_free"] = picked
    strategies["class_x4_free"] = lambda r: True
    # same but seeded by raw projection — lands on 3-4 cheap class QBs, i.e. the
    # league's validated cheap-QB stack (6-pt pass TDs compress QB pricing)
    seeds["class_x4_qb"] = list(g[g.in_class].sort_values(
        "ffa_points", ascending=False).head(4).index)
    strategies["class_x4_qb"] = lambda r: True
    if yr == 2025:
        print("2025 position-average prices:", {k: round(v, 1) for k, v in pos_avg.items()})
    row = {"season": yr}
    for nm, ok in strategies.items():
        ros, spent = build(g, ok, seed_idx=seeds.get(nm))
        row[nm] = eval_roster(ros, "pts")
        row[nm + "_proj"] = eval_roster(ros, "ffa_points")
        if yr in (2024, 2025):
            detail.setdefault(nm, {})[yr] = ros[["player", "position", "price", "ffa_points", "pts"]]
        row[nm + "_spent"] = spent
        row[nm + "_size"] = len(ros)
    results.append(row)

res = pd.DataFrame(results).set_index("season")
strat_cols = ["value_fill", "value_125", "optimizer", "stars_scrubs", "anchor_value",
              "mid_band", "class_fill", "class_anchor", "class_x4_free", "class_x4_qb"]
print("REALIZED lineup points by strategy (ex-post optimal QB/2RB/2WR/TE/FLEX from 12-man roster):")
print(res[strat_cols].round(0).to_string())
print("\nmean:"); print(res[strat_cols].mean().round(1).to_string())
print("\nmean spent:"); print(res[[c + "_spent" for c in strat_cols]].mean().round(0).to_string())
print("\nmean roster size:"); print(res[[c + "_size" for c in strat_cols]].mean().round(1).to_string())
print("\nwin count (best strategy per season):")
print(res[strat_cols].idxmax(axis=1).value_counts().to_string())

# paired tests vs value_fill
import scipy.stats as st
print("\npaired vs value_fill (10 seasons):")
for c in strat_cols[1:]:
    diff = res[c] - res.value_fill
    t = st.wilcoxon(diff) if diff.abs().sum() > 0 else None
    print(f"  {c:>13}: mean diff {diff.mean():+.1f}, wilcoxon p={t.pvalue:.3f}")

print("\n2025 example rosters:")
for nm in strat_cols:
    if nm in detail and 2025 in detail[nm]:
        r = detail[nm][2025].sort_values("price", ascending=False)
        names = ", ".join(f"{x.player}(${x.price})" for x in r.itertuples())
        print(f"  {nm}: {names}")
con.close()
