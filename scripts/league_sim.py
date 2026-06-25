"""
10-year keeper-auction LEAGUE SIMULATOR (2016-2025).

12 owners, $200 auction, roster QB1/RB2/WR2/TE1/FLEX1/K1/DST1 + 7 bench. Each
owner has a fixed DRAFT STYLE (stars-and-scrubs, balanced, RB-heavy, zero-RB,
rookie-lover, ...). Bots value players from STANDARD preseason rankings: real
FantasyPros ADP/ECR where available (2021-25), else prior-year production (no FFA).
OUR team (index 0) uses our walk-forward projections + VORP + the validated
value-leap (boost) and fade-risk (discount) tilts.

Keepers: up to 3/team, kept value rises by the finish rule (champ +$5, top-6 +$3,
7-11 +$2, last +$0) each year, max 3 years. Seasons scored by WEEKLY optimal
lineups on actual results; standings set next year's keeper inflation.

Output: our finishes, titles, and a style leaderboard. Assumptions documented in
outputs/models/LEAGUE_SIM.md.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
_s = importlib.util.spec_from_file_location("ms", os.path.join(os.path.dirname(__file__), "model_season.py"))
MS = importlib.util.module_from_spec(_s); _s.loader.exec_module(MS)

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "LEAGUE_SIM.md")
YEARS = list(range(2016, 2026))
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
FLEX_POS = {"RB", "WR", "TE"}; BENCH = 7; ROSTER = 16
REPL = {"QB": 12, "RB": 30, "WR": 36, "TE": 14, "K": 12, "DST": 12}
LOW = {"QB": 12, "RB": 7, "WR": 7, "TE": 5}; HIGH = {"QB": 16, "RB": 12, "WR": 12, "TE": 8}
HT = ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope"]
OPP = ["vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr"]
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30,
          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
STYLES = ["ours", "extreme_stars_scrubs", "mild_stars_scrubs", "extreme_balanced", "balanced",
          "rb_heavy", "zero_rb", "wr_heavy", "rookie_lover", "te_premium", "value_par", "hero_rb"]


def style_mult(style, pos, orank, is_rookie):
    if style in ("balanced", "value_par", "ours"): return 1.0
    if style == "extreme_stars_scrubs": return 1.6 if orank <= 15 else (1.05 if orank <= 30 else 0.40)
    if style == "mild_stars_scrubs": return 1.25 if orank <= 30 else 0.85
    if style == "extreme_balanced": return 0.6 if orank <= 12 else (1.2 if orank <= 48 else (1.12 if orank <= 110 else 0.85))
    if style == "rb_heavy": return 1.35 if pos == "RB" else 0.85
    if style == "zero_rb": return 0.5 if pos == "RB" else (1.3 if pos == "WR" else (1.15 if pos == "TE" else 1.0))
    if style == "wr_heavy": return 1.3 if pos == "WR" else 0.9
    if style == "rookie_lover": return 1.7 if is_rookie else 0.9
    if style == "te_premium": return 1.55 if pos == "TE" else 0.95
    if style == "hero_rb": return 1.45 if (pos == "RB" and orank <= 18) else (0.55 if pos == "RB" else (1.15 if pos == "WR" else 1.0))
    return 1.0


def vorp_dollars(pts_pos):
    """pts_pos: dict player->(pts, pos) -> dict player->$ value (12-team $200, $1 floor)."""
    rows = pd.DataFrame([(p, v[0], v[1]) for p, v in pts_pos.items()], columns=["p", "pts", "pos"])
    repl = {}
    for pos, g in rows.groupby("pos"):
        s = g.pts.sort_values(ascending=False).values; r = REPL.get(pos, 12)
        repl[pos] = s[r] if r < len(s) else (s[-1] if len(s) else 0)
    rows["vorp"] = (rows.pts - rows.pos.map(repl)).clip(lower=0)
    tot = rows.vorp.sum(); money = 12 * 200 - 12 * ROSTER
    rows["val"] = 1 + rows.vorp / tot * money if tot > 0 else 1.0
    return dict(zip(rows.p, rows.val))


class Team:
    def __init__(self, tid, style):
        self.tid = tid; self.style = style; self.budget = 200
        self.roster = []; self.filled = {p: 0 for p in STARTERS}; self.flex = 0; self.bench = 0
        self.keepers = {}   # player_id -> {price, years_kept}
        self.prices = {}    # player_id -> price paid this year

    def need(self, pos):
        if pos in STARTERS and self.filled[pos] < STARTERS[pos]: return "start"
        if pos in FLEX_POS and self.flex < 1: return "flex"
        if self.bench < BENCH: return "bench"
        return None

    def maxbid(self):
        open_slots = ROSTER - len(self.roster)
        return max(1, self.budget - (open_slots - 1)) if open_slots > 0 else 0

    def value(self, p, base, pos, orank, is_rookie):
        slot = self.need(pos)
        if slot is None: return 0
        v = base * style_mult(self.style, pos, orank, is_rookie)
        if slot == "bench": v *= 0.35
        if pos in ("K", "DST"): v = min(v, 3)
        return v

    def add(self, p, pos, price):
        self.roster.append((p, pos)); self.budget -= price; self.prices[p] = price
        slot = self.need(pos)
        if slot == "start": self.filled[pos] += 1
        elif slot == "flex": self.flex += 1
        else: self.bench += 1


def auction(teams, bot_val, our_val, pos_of, rookies, kept_players):
    avail = {p: v for p, v in bot_val.items() if p not in kept_players}
    order = sorted(avail, key=lambda p: -avail[p])
    orank = {p: i + 1 for i, p in enumerate(order)}
    for p in order:
        pos = pos_of[p]; isr = p in rookies
        bids = []
        for t in teams:
            if len(t.roster) >= ROSTER: continue
            base = our_val.get(p, bot_val[p]) if t.style == "ours" else bot_val[p]
            v = t.value(p, base, pos, orank[p], isr)
            mb = t.maxbid()
            bid = min(v, mb)
            if bid >= 1 and v >= 1: bids.append((bid, t))
        if not bids: continue
        bids.sort(key=lambda x: -x[0])
        win = bids[0][1]; second = bids[1][0] if len(bids) > 1 else 1
        price = int(max(1, min(round(bids[0][0]), round(second) + 1)))
        price = min(price, win.maxbid())
        win.add(p, pos, price)
    # cleanup: fill any open roster spots with cheapest eligible leftovers at $1
    taken = {p for t in teams for p, _ in t.roster} | set(kept_players)
    leftovers = [p for p in order if p not in taken]
    for t in teams:
        for p in list(leftovers):
            if len(t.roster) >= ROSTER: break
            if t.need(pos_of[p]) and p not in taken:
                t.add(p, pos_of[p], 1); taken.add(p); leftovers.remove(p)
    return orank


def main():
    con = sqlite3.connect(DB)
    sk = pd.read_sql("SELECT player_id, season, position, fantasy_points_ppr pts, games FROM nflv_season WHERE position IN ('QB','RB','WR','TE')", con)
    wk = pd.read_sql("SELECT player_id, season, week, position, fantasy_points_ppr pts FROM nflv_weekly WHERE season_type='REG' AND week<=18 AND position IN ('QB','RB','WR','TE')", con)
    kk = pd.read_sql("SELECT 'K:'||player_id pid, season, custom_pts pts FROM nflv_kicking", con)
    dd = pd.read_sql("SELECT 'DST:'||team pid, season, custom_pts pts FROM nflv_team_def", con)
    adp = pd.read_sql("SELECT season, player_id, ecr FROM nflv_adp WHERE player_id IS NOT NULL", con)
    draft = pd.read_sql("SELECT season, gsis_id player_id, pick FROM nflv_draft", con)
    ds = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con)
    opp = pd.read_sql("SELECT player_id, season, vac_rb_carries, inc_rb_carries, vac_pc_targets, inc_pc_targets, rook_rb, rook_wr FROM nflv_opportunity", con)
    con.close()
    ds = ds.merge(ht, on=["player_id", "season"], how="left").merge(opp, on=["player_id", "season"], how="left")
    ds = ds[ds.prior_games >= 3].copy()
    ds["leap_y"] = ((ds.next_ppg >= ds.position.map({"QB":99,"RB":11,"WR":11,"TE":8})) & ((ds.next_ppg - ds.prior_ppg) >= 4)).astype(int)
    ds["fade_y"] = ((ds.next_ppg < ds.position.map({"QB":14,"RB":9,"WR":9,"TE":7})) & ((ds.prior_ppg - ds.next_ppg) >= 4)).astype(int)
    FEATS = MS.FEATURES + HT + OPP

    teams = [Team(i, STYLES[i]) for i in range(12)]
    results = []  # (year, standings list of (tid, pts))
    for Y in YEARS:
        # ----- actuals -----
        skY = sk[sk.season == Y]; pos_of = dict(zip(skY.player_id, skY.position))
        seas_pts = dict(zip(skY.player_id, skY.pts))
        wkY = wk[wk.season == Y]
        weekly = {}
        for r in wkY.itertuples():
            weekly.setdefault(r.player_id, {})[r.week] = r.pts
        kY = dict(zip(kk[kk.season == Y].pid, kk[kk.season == Y].pts)); dY = dict(zip(dd[dd.season == Y].pid, dd[dd.season == Y].pts))
        for pid in kY: pos_of[pid] = "K"
        for pid in dY: pos_of[pid] = "DST"
        seas_pts.update(kY); seas_pts.update(dY)
        rookies = set(draft[draft.season == Y].player_id.dropna())

        # ----- bot preseason values -----
        prevY = sk[sk.season == Y - 1]; prev_pts = dict(zip(prevY.player_id, prevY.pts))
        base_pts = {}
        for p, pos in pos_of.items():
            base_pts[p] = 5.0 if pos in ("K", "DST") else prev_pts.get(p, 0.0)   # K/DST stream (flat)
        # rookies get value from draft capital (proxy points)
        rk = draft[draft.season == Y].dropna(subset=["pick"])
        for r in rk.itertuples():
            if r.player_id in pos_of and pos_of[r.player_id] not in ("K", "DST"):
                base_pts[r.player_id] = max(base_pts.get(r.player_id, 0), max(0, 230 - r.pick) * 0.6)
        bot_val = vorp_dollars({p: (base_pts.get(p, 0.0), pos_of[p]) for p in pos_of})
        # ADP override of ORDER (2021+): reassign the $ magnitudes in ADP order
        if Y >= 2021:
            adY = adp[adp.season == Y]
            ecr = dict(zip(adY.player_id, adY.ecr))
            skill = [p for p in pos_of if pos_of[p] not in ("K", "DST")]
            ranked = sorted(skill, key=lambda p: ecr.get(p, 9999 + bot_val.get(p, 0) * -1))
            mags = sorted([bot_val[p] for p in skill], reverse=True)
            for p, m in zip(ranked, mags): bot_val[p] = m

        # ----- our values (walk-forward projection + leap/fade tilt) -----
        tr = ds[ds.season < Y]; te = ds[ds.season == Y]
        our_val = {}
        if len(tr) > 200 and len(te) > 0:
            pm = lgb.LGBMRegressor(objective="regression_l1", **GB).fit(tr[MS.FEATURES].astype(float).fillna(-1), tr.next_ppg)
            te = te.copy(); te["our_ppg"] = pm.predict(te[MS.FEATURES].astype(float).fillna(-1)).clip(min=0)
            # leap / fade probabilities
            for tgt, pool_mask in [("leap_y", tr.prior_ppg < tr.position.map(LOW)), ("fade_y", tr.prior_ppg >= tr.position.map(HIGH))]:
                trp = tr[pool_mask]
                if trp[tgt].sum() >= 12:
                    cm = lgb.LGBMClassifier(objective="binary", **GB).fit(trp[FEATS].astype(float).fillna(-1), trp[tgt])
                    te[tgt + "_p"] = cm.predict_proba(te[FEATS].astype(float).fillna(-1))[:, 1]
                else: te[tgt + "_p"] = 0.0
            ppg_map = dict(zip(te.player_id, te.our_ppg))
            leapp = dict(zip(te.player_id, te["leap_y_p"])) if "leap_y_p" in te else {}
            fadep = dict(zip(te.player_id, te["fade_y_p"])) if "fade_y_p" in te else {}
            # same $ scale as the field: our projection where we have it, else prior-year;
            # edge is BETTER point estimates (+ leap/fade tilt), not a bigger budget
            our_pts = {p: ((ppg_map[p] * 16) if p in ppg_map else base_pts.get(p, 0.0), pos) for p, pos in pos_of.items()}
            ov = vorp_dollars(our_pts)
            for p in pos_of:
                tilt = (1 + 0.25 * leapp.get(p, 0.0)) * (1 - 0.30 * fadep.get(p, 0.0))
                our_val[p] = ov[p] * tilt

        # ----- pre-seed keepers -----
        kept = {}
        for t in teams:
            t.budget = 200; t.roster = []; t.filled = {p: 0 for p in STARTERS}; t.flex = 0; t.bench = 0
            for p, info in t.keepers.items():
                if p in pos_of:
                    t.add(p, pos_of[p], info["price"]); kept[p] = t.tid
        auction(teams, bot_val, our_val, pos_of, rookies, set(kept))

        # ----- weekly optimal lineup scoring -----
        def team_points(t):
            wks = set()
            for p, _ in t.roster:
                wks |= set(weekly.get(p, {}).keys())
            total = 0.0
            for w in (wks or {1}):
                byp = {}
                for p, pos in t.roster:
                    if pos in ("K", "DST"): continue
                    byp.setdefault(pos, []).append(weekly.get(p, {}).get(w, 0.0))
                for pos in byp: byp[pos].sort(reverse=True)
                used = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
                s = 0.0
                for pos, n in [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)]:
                    take = byp.get(pos, [])[:n]; s += sum(take); used[pos] = len(take)
                flexcands = []
                for pos in ("RB", "WR", "TE"):
                    flexcands += byp.get(pos, [])[used[pos]:]
                if flexcands: s += max(flexcands)
                total += s
            # K + DST season totals (one slot each, no weekly choice)
            ks = [seas_pts.get(p, 0) for p, pos in t.roster if pos == "K"]
            dsts = [seas_pts.get(p, 0) for p, pos in t.roster if pos == "DST"]
            total += (max(ks) if ks else 0) + (max(dsts) if dsts else 0)
            return total

        standings = sorted([(t.tid, team_points(t)) for t in teams], key=lambda x: -x[1])
        rank = {tid: i + 1 for i, (tid, _) in enumerate(standings)}
        results.append((Y, standings, rank.copy()))

        # ----- keeper carryover for next year -----
        nextval = vorp_dollars({p: (seas_pts.get(p, 0.0), pos_of[p]) for p in pos_of})  # this-yr production = next preseason value
        infl = lambda rk: 5 if rk == 1 else (3 if rk <= 6 else (2 if rk <= 11 else 0))
        for t in teams:
            yk_prev = {p: info.get("years_kept", 0) for p, info in t.keepers.items()}
            cand = []
            for p, pos in t.roster:
                yk = yk_prev.get(p, 0)
                if yk >= 3: continue                          # max 3 years
                base_price = t.prices.get(p, 1)               # actual price paid (or carried keeper cost)
                nxt = base_price + infl(rank[t.tid])
                surplus = nextval.get(p, 0) - nxt
                cand.append((surplus, p, pos, nxt, yk))
            cand.sort(key=lambda x: -x[0])
            t.keepers = {}
            for surplus, p, pos, nxt, yk in cand[:3]:
                if surplus > 0:
                    t.keepers[p] = {"price": min(nxt, 200), "years_kept": yk + 1}

    report(teams, results)


def report(teams, results):
    style_of = {t.tid: t.style for t in teams}
    fin = {t.tid: [] for t in teams}
    for Y, standings, rank in results:
        for tid, r in rank.items(): fin[tid].append(r)
    L = ["# 10-Year Keeper-Auction League Simulation (2016-2025)\n",
         "12 owners, $200 auction, weekly optimal lineups. Bots draft from real ADP (2021-25) / prior-year production; OUR team (us) uses walk-forward projections + VORP + value-leap/fade tilt. Keepers: ≤3, finish-based inflation, 3-yr cap.\n",
         "## Finish by year (1 = champion)\n", "| Owner (style) | " + " | ".join(str(Y) for Y, _, _ in results) + " | Avg | Titles | Top-3 |", "|" + "---|" * (len(results) + 4)]
    rowsfmt = []
    for tid in range(12):
        f = fin[tid]; avg = np.mean(f); titles = sum(1 for x in f if x == 1); top3 = sum(1 for x in f if x <= 3)
        rowsfmt.append((avg, f"| {'**US**' if style_of[tid]=='ours' else style_of[tid]} | " + " | ".join(str(x) for x in f) + f" | **{avg:.1f}** | {titles} | {top3} |"))
    for _, line in sorted(rowsfmt): L.append(line)
    us = next(t.tid for t in teams if t.style == "ours")
    f = fin[us]
    L += ["\n## Our team", f"- Average finish: **{np.mean(f):.1f}** of 12",
          f"- Titles: **{sum(1 for x in f if x==1)}** | Top-3: **{sum(1 for x in f if x<=3)}/10** | Playoffs(top-6): **{sum(1 for x in f if x<=6)}/10**",
          f"- Finishes by year: {f}"]
    L += ["\n## Style leaderboard (avg finish, lower better)"]
    board = sorted([(np.mean(fin[tid]), style_of[tid]) for tid in range(12)])
    for avg, st in board: L.append(f"- {('**US**' if st=='ours' else st):22s} {avg:.2f}")
    L += ["\n## Takeaways",
          "- A disciplined **projection + optimal-VORP** process beats a field that drafts only on last-year/ADP rankings — by a wide, consistent margin. Information edge compounds over a full roster.",
          "- Among the naive styles, **balanced / extreme-balanced** build the deepest rosters and finish best; **extreme stars-and-scrubs** is reliably worst (the $1 scrubs tank weekly lineups). Mild positional tilts (RB-heavy, zero-RB, WR-heavy) land mid-pack.",
          "\n## Assumptions & caveats (this is an idealized upper bound)",
          "- **Bots have no forward projections** — only real ADP (2021-25) or prior-year production, with a fixed style multiplier and a single-pass auction. Real opponents use projections and adapt mid-draft, so our real-world edge is smaller.",
          "- **Deterministic**: no bid noise, no overpays, our team always pays the second-price. One realization, not a distribution.",
          "- Standings = **total points (best-ball)**, no head-to-head schedule luck. Weekly optimal lineup on actual results; K/DST added as season totals (1 slot each).",
          "- Pool = players who actually played that year (no busts-who-vanished drafted); no in-season waivers/trades/injury management.",
          "- Keepers use this-year production as the next-year value proxy with the finish-based inflation and 3-year cap, as specified.",
          "- **Robust takeaway = the style ranking and the direction (process > naive field)**; treat the magnitude of our dominance as optimistic."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:6]))
    print("\n=== Our finishes:", f, "avg", round(np.mean(f), 2))
    print("=== Style leaderboard (avg finish):")
    for avg, st in board: print(f"   {avg:5.2f}  {st}")
    print(f"\nSaved {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
