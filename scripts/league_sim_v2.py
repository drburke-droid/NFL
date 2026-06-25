"""
REALISTIC 10-year keeper league sim (v2): head-to-head weekly matchups, FAAB
waivers, K/DST streaming, and lineups set by a weekly PROJECTION (real player
props 2023-25, else a leakage-free form x matchup model) then scored by ACTUAL.

Isolates the DRAFT-DAY edge: every team uses the IDENTICAL weekly projection and
FAAB logic (equal in-season info); only draft valuation differs (bots: real ADP
2021-25 / prior-year x style; us: walk-forward projections + VORP + leap/fade tilt).

Refactored into load() / build_year_values() / simulate_year() so the Monte Carlo
(league_sim_v3.py) can reuse the precomputed per-year values and only re-randomize
the draft bids and schedule. Report: outputs/models/LEAGUE_SIM_V2.md.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(os.path.dirname(__file__), n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
MS = _load("model_season"); L1 = _load("league_sim")

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "nfl_odds.db")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", "LEAGUE_SIM_V2.md")
YEARS = list(range(2016, 2026)); REG_WEEKS = 14
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}; FLEX = {"RB", "WR", "TE"}; BENCH = 7; ROSTER = 14
BASE_PPG = {"QB": 14, "RB": 8, "WR": 8, "TE": 6}; KDST_FLAT = 16.0
LOW = {"QB": 12, "RB": 7, "WR": 7, "TE": 5}; HIGH = {"QB": 16, "RB": 12, "WR": 12, "TE": 8}
HT = ["ht_d_ppg", "ht_h2_ppg", "ht_d_snap", "ht_h2_snap", "ht_d_tch", "ht_d_tgtsh", "ht_slope"]
OPP = ["vac_rb_carries", "inc_rb_carries", "vac_pc_targets", "inc_pc_targets", "rook_rb", "rook_wr"]
GB = dict(n_estimators=300, learning_rate=0.04, num_leaves=20, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
STYLES = L1.STYLES


def round_robin(teams, weeks):
    arr = list(teams); n = len(arr); sched = []
    for r in range(weeks):
        sched.append([(arr[i], arr[n - 1 - i]) for i in range(n // 2)])
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    return sched


class Team:
    def __init__(self, tid, style):
        self.tid, self.style = tid, style; self.faab = 100; self.keepers = {}; self.bench_w = 0.35; self.reset()
    def reset(self):
        self.budget = 200; self.roster = []; self.prices = {}
        self.filled = {p: 0 for p in STARTERS}; self.flex = 0; self.bench = 0
    def need(self, pos):
        if pos in STARTERS and self.filled[pos] < STARTERS[pos]: return "start"
        if pos in FLEX and self.flex < 1: return "flex"
        if self.bench < BENCH: return "bench"
        return None
    def maxbid(self):
        o = ROSTER - len(self.roster); return max(1, self.budget - (o - 1)) if o > 0 else 0
    def value(self, base, pos, orank, isr):
        slot = self.need(pos)
        if slot is None: return 0
        return base * L1.style_mult(self.style, pos, orank, isr) * (self.bench_w if slot == "bench" else 1.0)
    def add(self, p, pos, price):
        self.roster.append([p, pos]); self.budget -= price; self.prices[p] = price
        slot = self.need(pos)
        if slot == "start": self.filled[pos] += 1
        elif slot == "flex": self.flex += 1
        else: self.bench += 1
    def drop(self, p):
        self.roster = [r for r in self.roster if r[0] != p]; self.prices.pop(p, None)


def auction(teams, bot_val, our_val, pos_of, rookies, kept, rng=None, bid_sigma=0.0):
    avail = {p: v for p, v in bot_val.items() if p not in kept}
    order = sorted(avail, key=lambda p: -avail[p]); orank = {p: i + 1 for i, p in enumerate(order)}
    budget_left = sum(t.budget for t in teams); slots_left = sum(ROSTER - len(t.roster) for t in teams)
    rv = sum(avail.values()); rc = len(order)
    for p in order:
        # mild end-game INFLATION (excess money bids up remaining players); leftover is
        # then deployed in cleanup so budgets clear without scrambling the competitive auction
        disc_b = budget_left - slots_left; disc_v = rv - rc
        infl = min(1.6, max(1.0, disc_b / disc_v)) if disc_v > 0 else 1.0
        rv -= bot_val[p]; rc -= 1
        pos = pos_of[p]; isr = p in rookies; bids = []
        for t in teams:
            if len(t.roster) >= ROSTER: continue
            base = (our_val.get(p, bot_val[p]) if t.style == "ours" else bot_val[p]) * infl
            if rng is not None and bid_sigma > 0: base *= float(np.exp(rng.normal(0, bid_sigma)))
            bid = min(t.value(base, pos, orank[p], isr), t.maxbid())
            if bid >= 1: bids.append((bid, t.tid, t))
        if not bids: continue
        bids.sort(key=lambda x: -x[0]); win = bids[0][2]; second = bids[1][0] if len(bids) > 1 else 1
        price = int(max(1, min(round(bids[0][0]), round(second) + 1))); price = min(price, win.maxbid())
        win.add(p, pos, price); budget_left -= price; slots_left -= 1
    taken = {p for t in teams for p, _ in t.roster} | set(kept)
    left = [p for p in order if p not in taken]
    for t in teams:                                          # fill remaining slots AND deploy leftover budget
        for p in list(left):
            if len(t.roster) >= ROSTER: break
            if t.need(pos_of[p]):
                open_slots = ROSTER - len(t.roster)
                price = max(1, min(t.maxbid(), int(round(t.budget / open_slots))))
                t.add(p, pos_of[p], price); left.remove(p)


def waivers(teams, Y, w, pos_of, D):
    WACT, EXP, GAMES_WK, PRIMARY = D["WACT"], D["EXP"], D["GAMES_WK"], D["PTEAM_PRIMARY"]
    if w <= 1: return
    rostered = {p for t in teams for p, _ in t.roster}
    def form(p):
        wa = WACT.get((Y, p), {}); prev = [wa[x] for x in range(max(1, w - 3), w) if x in wa]
        return np.mean(prev) if prev else 0.0
    last = lambda p: WACT.get((Y, p), {}).get(w - 1, 0.0)
    attract = lambda p: max(form(p), last(p) * 0.7)        # a breakout last week makes a FA a target
    fa = sorted([p for p in pos_of if p not in rostered], key=lambda p: -attract(p))[:25]
    bids = []
    for t in teams:
        # protect YOUNG & CHEAP (keeper-controllable) stashes — don't cut them for a streamer
        prot = lambda p: (EXP.get((Y, p), 9) <= 2 and t.prices.get(p, 99) <= 5)
        droppable = [(form(p), p) for p, _ in t.roster if not prot(p)] or [(form(p), p) for p, _ in t.roster]
        if not droppable: continue
        worst_f, worst_p = min(droppable)
        for p in fa[:12]:
            up = attract(p) - worst_f
            if up > 0.2 and t.faab > 0:
                # realistic FAAB: routine adds $1-8; a player off a HUGE last week spikes to ~$30; ~never >$50
                base = min(8, max(1, round(up * 0.8)))
                bonus = max(0.0, last(p) - 15.0) * 1.4 if last(p) >= 18 else 0.0
                bid = int(min(t.faab, min(45, max(1, round(base + bonus)))))
                bids.append((bid, -t.tid, t, p, worst_p)); break
    bids.sort(key=lambda x: (-x[0], x[1])); taken = set(); gone = set()
    for bid, _, t, p, worst_p in bids:
        if p in taken or t.faab < bid or t.tid in gone or worst_p not in {x[0] for x in t.roster}: continue
        t.drop(worst_p); t.add(p, pos_of[p], 0); t.faab -= bid; taken.add(p); gone.add(t.tid)
    # cheap bye/depth streaming ($1-4): fill a starting hole created by byes — the routine sub-$10 churn
    gw = GAMES_WK.get((Y, w))
    if gw:
        rostered = {p for tm in teams for p, _ in tm.roster}
        avail = [p for p in pos_of if p not in rostered and PRIMARY.get((Y, p)) in gw]
        for t in teams:
            if t.tid in gone or t.faab < 1: continue
            need_pos = None
            for pos, n in STARTERS.items():
                live = sum(1 for p, pp in t.roster if pp == pos and PRIMARY.get((Y, p)) in gw)
                if live < n: need_pos = pos; break
            if not need_pos: continue
            cands = sorted([p for p in avail if pos_of[p] == need_pos], key=lambda p: -last(p))
            if cands:
                add = cands[0]; bid = int(min(t.faab, max(1, round(form(add) * 0.4)), 4))
                prot = lambda p: (EXP.get((Y, p), 9) <= 2 and t.prices.get(p, 99) <= 5)
                dr = [(form(p), p) for p, _ in t.roster if not prot(p)] or [(form(p), p) for p, _ in t.roster]
                worst = min(dr)[1]; t.drop(worst); t.add(add, pos_of[add], 0); t.faab -= bid; avail.remove(add)


def load(con):
    sk = pd.read_sql("SELECT player_id, season, position, fantasy_points_ppr pts, games FROM nflv_season WHERE position IN ('QB','RB','WR','TE')", con)
    wkdf = pd.read_sql("SELECT player_id, season, week, position, team, fantasy_points_ppr pts FROM nflv_weekly WHERE season_type='REG' AND week<=18 AND position IN ('QB','RB','WR','TE')", con)
    lines = pd.read_sql("SELECT season, week, team, implied_team_total itt FROM nflv_game_lines WHERE game_type='REG'", con)
    props = pd.read_sql("SELECT season, week, player_id, proj_pts FROM weekly_proj_props", con)
    adp = pd.read_sql("SELECT season, player_id, ecr FROM nflv_adp WHERE player_id IS NOT NULL", con)
    draft = pd.read_sql("SELECT season, gsis_id player_id, pick FROM nflv_draft", con)
    ds = pd.read_sql("SELECT * FROM season_dataset", con)
    ht = pd.read_sql("SELECT * FROM nflv_half_trend", con)
    opp = pd.read_sql("SELECT player_id, season, vac_rb_carries, inc_rb_carries, vac_pc_targets, inc_pc_targets, rook_rb, rook_wr FROM nflv_opportunity", con)
    ds = ds.merge(ht, on=["player_id", "season"], how="left").merge(opp, on=["player_id", "season"], how="left")
    ds = ds[ds.prior_games >= 3].copy()
    ds["leap_y"] = ((ds.next_ppg >= ds.position.map({"QB": 99, "RB": 11, "WR": 11, "TE": 8})) & ((ds.next_ppg - ds.prior_ppg) >= 4)).astype(int)
    ds["fade_y"] = ((ds.next_ppg < ds.position.map({"QB": 14, "RB": 9, "WR": 9, "TE": 7})) & ((ds.prior_ppg - ds.next_ppg) >= 4)).astype(int)
    rost = pd.read_sql("SELECT season, gsis_id player_id, years_exp FROM nflv_rosters WHERE gsis_id IS NOT NULL", con)
    WACT, PTEAM, tcount = {}, {}, {}
    for r in wkdf.itertuples():
        WACT.setdefault((r.season, r.player_id), {})[r.week] = r.pts
        PTEAM.setdefault((r.season, r.player_id), {})[r.week] = r.team
        tcount.setdefault((r.season, r.player_id), {}); tcount[(r.season, r.player_id)][r.team] = tcount[(r.season, r.player_id)].get(r.team, 0) + 1
    GAMES_WK = {}
    for r in lines.itertuples(): GAMES_WK.setdefault((r.season, r.week), set()).add(r.team)
    PTEAM_PRIMARY = {k: max(v, key=v.get) for k, v in tcount.items()}     # most-played team that season
    return dict(sk=sk, ds=ds, adp=adp, draft=draft, WACT=WACT, PTEAM=PTEAM, GAMES_WK=GAMES_WK, PTEAM_PRIMARY=PTEAM_PRIMARY,
                EXP={(r.season, r.player_id): r.years_exp for r in rost.itertuples()},
                LINE={(r.season, r.team, r.week): r.itt for r in lines.itertuples()},
                PROPS={(r.season, r.week, r.player_id): r.proj_pts for r in props.itertuples()},
                POS={(r.season, r.player_id): r.position for r in sk.itertuples()},
                PRIORPPG={(r.season, r.player_id): (r.pts / max(r.games, 1)) for r in sk.itertuples()},
                FEATS=MS.FEATURES + HT + OPP)


def make_wproj(D):
    WACT, PROPS, LINE, POS, PRIORPPG, PTEAM = D["WACT"], D["PROPS"], D["LINE"], D["POS"], D["PRIORPPG"], D["PTEAM"]
    GAMES_WK, PRIMARY = D["GAMES_WK"], D["PTEAM_PRIMARY"]
    def wproj(season, pid, week):
        primary = PRIMARY.get((season, pid)); gw = GAMES_WK.get((season, week))
        if primary and gw and primary not in gw: return 0.0          # team on BYE -> bench, don't start
        if (season, week, pid) in PROPS: return PROPS[(season, week, pid)]
        wa = WACT.get((season, pid), {}); prev = [wa[w] for w in range(max(1, week - 4), week) if w in wa]
        form = np.mean(prev) if prev else PRIORPPG.get((season - 1, pid), BASE_PPG.get(POS.get((season, pid), "WR"), 8))
        team = PTEAM.get((season, pid), {}).get(week, primary)
        env = np.clip(LINE.get((season, team, week), 22.5) / 22.5, 0.8, 1.25) if team else 1.0
        return max(form * env, 0.0)
    return wproj


def build_year_values(Y, D):
    sk, ds, adp, draft, FEATS = D["sk"], D["ds"], D["adp"], D["draft"], D["FEATS"]
    skY = sk[sk.season == Y]; pos_of = dict(zip(skY.player_id, skY.position))
    rookies = set(draft[draft.season == Y].player_id.dropna())
    prevY = dict(zip(sk[sk.season == Y - 1].player_id, sk[sk.season == Y - 1].pts))
    base_pts = {p: prevY.get(p, 0.0) for p in pos_of}
    for r in draft[draft.season == Y].dropna(subset=["pick"]).itertuples():
        if r.player_id in pos_of: base_pts[r.player_id] = max(base_pts.get(r.player_id, 0), max(0, 230 - r.pick) * 0.6)
    bot_val = L1.vorp_dollars({p: (base_pts.get(p, 0.0), pos_of[p]) for p in pos_of})
    if Y >= 2021:
        ecr = dict(zip(adp[adp.season == Y].player_id, adp[adp.season == Y].ecr))
        ranked = sorted(pos_of, key=lambda p: ecr.get(p, 9999)); mags = sorted([bot_val[p] for p in pos_of], reverse=True)
        for p, m in zip(ranked, mags): bot_val[p] = m
    tr = ds[ds.season < Y]; te = ds[ds.season == Y].copy(); our_val = {}
    if len(tr) > 200 and len(te) > 0:
        pm = lgb.LGBMRegressor(objective="regression_l1", **GB).fit(tr[MS.FEATURES].astype(float).fillna(-1), tr.next_ppg)
        te["our_ppg"] = pm.predict(te[MS.FEATURES].astype(float).fillna(-1)).clip(min=0)
        for tgt, mask in [("leap_y", tr.prior_ppg < tr.position.map(LOW)), ("fade_y", tr.prior_ppg >= tr.position.map(HIGH))]:
            trp = tr[mask]
            te[tgt + "_p"] = (lgb.LGBMClassifier(objective="binary", **GB).fit(trp[FEATS].astype(float).fillna(-1), trp[tgt]).predict_proba(te[FEATS].astype(float).fillna(-1))[:, 1]) if trp[tgt].sum() >= 12 else 0.0
        ppg_map = dict(zip(te.player_id, te.our_ppg)); leapp = dict(zip(te.player_id, te.leap_y_p)); fadep = dict(zip(te.player_id, te.fade_y_p))
        ov = L1.vorp_dollars({p: ((ppg_map[p] * 16) if p in ppg_map else base_pts.get(p, 0.0), pos) for p, pos in pos_of.items()})
        for p in pos_of: our_val[p] = ov[p] * (1 + 0.25 * leapp.get(p, 0.0)) * (1 - 0.30 * fadep.get(p, 0.0))
    return pos_of, rookies, bot_val, our_val


def simulate_year(Y, teams, D, pv, wproj, rng=None, bid_sigma=0.0):
    pos_of, rookies, bot_val, our_val = pv
    sk, WACT = D["sk"], D["WACT"]
    skY = sk[sk.season == Y]; seas_tot = dict(zip(skY.player_id, skY.pts))
    kept = {}
    for t in teams:
        t.reset()
        for p, info in t.keepers.items():
            if p in pos_of: t.add(p, pos_of[p], info["price"]); kept[p] = t.tid
    auction(teams, bot_val, our_val, pos_of, rookies, set(kept), rng=rng, bid_sigma=bid_sigma)
    for t in teams: t.faab = 100

    GAMES_WK, PRIMARY = D["GAMES_WK"], D["PTEAM_PRIMARY"]
    def lineup_actual(t, w):
        gw = GAMES_WK.get((Y, w)); cand = {}
        for p, pos in t.roster:
            prim = PRIMARY.get((Y, p))
            if prim and gw and prim not in gw: continue          # on BYE -> never start
            cand.setdefault(pos, []).append((wproj(Y, p, w), WACT.get((Y, p), {}).get(w, 0.0)))
        for pos in cand: cand[pos].sort(key=lambda x: -x[0])
        used = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}; s = 0.0
        for pos, n in [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)]:
            for _, act in cand.get(pos, [])[:n]: s += act
            used[pos] = min(n, len(cand.get(pos, [])))
        flexc = []
        for pos in FLEX: flexc += cand.get(pos, [])[used[pos]:]
        if flexc: s += max(flexc, key=lambda x: x[0])[1]
        return s + KDST_FLAT

    order = list(range(12))
    if rng is not None: rng.shuffle(order)
    sched = round_robin(order, REG_WEEKS)
    wins = {t.tid: 0 for t in teams}; pf = {t.tid: 0.0 for t in teams}
    for w, pairs in enumerate(sched, start=1):
        waivers(teams, Y, w, pos_of, D)
        for a, b in pairs:
            sa, sb = lineup_actual(teams[a], w), lineup_actual(teams[b], w)
            pf[a] += sa; pf[b] += sb
            if sa >= sb: wins[a] += 1
            else: wins[b] += 1
    seed = sorted(range(12), key=lambda i: (-wins[i], -pf[i])); po = seed[:6]
    game = lambda i, j, w: i if lineup_actual(teams[i], w) >= lineup_actual(teams[j], w) else j
    w1 = game(po[2], po[5], 15); w2 = game(po[3], po[4], 15)
    better_w, worse_w = sorted([w1, w2], key=lambda x: seed.index(x))
    semiA = game(po[0], worse_w, 16); semiB = game(po[1], better_w, 16)
    champ = game(semiA, semiB, 17); runner = semiB if champ == semiA else semiA
    semi_losers = sorted([(worse_w if semiA == po[0] else po[0]), (better_w if semiB == po[1] else po[1])], key=lambda x: seed.index(x))
    wc_losers = sorted([x for x in [po[2], po[5], po[3], po[4]] if x not in (w1, w2)], key=lambda x: seed.index(x))
    place = [champ, runner] + semi_losers + wc_losers + seed[6:]
    rank = {tid: i + 1 for i, tid in enumerate(place)}

    nextval = L1.vorp_dollars({p: (seas_tot.get(p, 0.0), pos_of[p]) for p in pos_of})
    infl = lambda rk: 5 if rk == 1 else (3 if rk <= 6 else (2 if rk <= 11 else 0))
    for t in teams:
        ykp = {p: i.get("years_kept", 0) for p, i in t.keepers.items()}; cand = []
        for p, pos in t.roster:
            yk = ykp.get(p, 0)
            if yk >= 3 or p not in pos_of: continue
            nxt = t.prices.get(p, 1) + infl(rank[t.tid]); cand.append((nextval.get(p, 0) - nxt, p, pos, nxt, yk))
        cand.sort(key=lambda x: -x[0]); t.keepers = {}
        for surplus, p, pos, nxt, yk in cand[:3]:
            if surplus > 0: t.keepers[p] = {"price": min(nxt, 200), "years_kept": yk + 1}
    return rank, wins


def main():
    con = sqlite3.connect(DB); D = load(con); con.close()
    wproj = make_wproj(D); teams = [Team(i, STYLES[i]) for i in range(12)]
    yr_finish = {t.tid: [] for t in teams}; yr_rec = {t.tid: [] for t in teams}
    for Y in YEARS:
        pv = build_year_values(Y, D)
        rank, wins = simulate_year(Y, teams, D, pv, wproj)
        for t in teams: yr_finish[t.tid].append(rank[t.tid]); yr_rec[t.tid].append(wins[t.tid])
    report(teams, yr_finish, yr_rec)


def report(teams, yr_finish, yr_rec):
    style_of = {t.tid: t.style for t in teams}
    L = ["# Realistic Keeper League Simulation v2 (2016-2025) — H2H, waivers, props-driven lineups\n",
         "Every team uses the SAME weekly projection (real player props 2023-25, else form×matchup) for lineups & FAAB waivers — equal in-season info. Only the DRAFT differs: bots use ADP/prior-year × style; we use projections + VORP + leap/fade. 14-wk H2H, top-6 playoffs, FAAB $100, K/DST streamed (flat-equal).\n",
         "## Finish by year (1 = champion)\n", "| Owner (style) | " + " | ".join(str(Y) for Y in YEARS) + " | Avg | Titles | Top-3 | Playoffs |", "|" + "---|" * (len(YEARS) + 5)]
    rows = []
    for tid in range(12):
        f = yr_finish[tid]; avg = np.mean(f)
        rows.append((avg, f"| {'**US**' if style_of[tid]=='ours' else style_of[tid]} | " + " | ".join(str(x) for x in f) + f" | **{avg:.1f}** | {sum(x==1 for x in f)} | {sum(x<=3 for x in f)} | {sum(x<=6 for x in f)} |"))
    for _, line in sorted(rows): L.append(line)
    us = next(t.tid for t in teams if t.style == "ours"); f = yr_finish[us]
    L += ["\n## Our team", f"- Avg finish **{np.mean(f):.1f}** | Titles **{sum(x==1 for x in f)}** | Top-3 **{sum(x<=3 for x in f)}/10** | Playoffs **{sum(x<=6 for x in f)}/10**",
          f"- Avg regular-season wins: **{np.mean(yr_rec[us]):.1f}** of {REG_WEEKS} | finishes {f}",
          "\n## Style leaderboard (avg finish, lower=better)"]
    board = sorted([(np.mean(yr_finish[tid]), style_of[tid]) for tid in range(12)])
    for avg, st in board: L.append(f"- {('**US**' if st=='ours' else st):22s} {avg:.2f}")
    L += ["\n## Notes",
          "- In-season skill is equalized (identical projection + waiver logic for all 12), so finishing differences trace to **draft-day roster construction** — our team's only edge.",
          "- Weekly props (2023-25) give true market projections; pre-2023 uses leakage-free recent-form × game-environment. Lineups set by projection, scored by actual (realistic start/sit error).",
          "- K/DST streamed as an equal flat contribution (near-random, everyone streams). H2H wins, top-6 playoffs, FAAB $100 blind weekly bids.",
          "- Single deterministic realization; league_sim_v3.py Monte-Carlos the draft bids + schedule for finish distributions / title odds."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:6])); print("\nOur finishes:", f, "avg", round(np.mean(f), 2), "| avg wins", round(np.mean(yr_rec[us]), 1))
    print("Style leaderboard:"); [print(f"   {a:5.2f}  {s}") for a, s in board]
    print("Saved", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
