"""
Detailed week-by-week TRACE of one season (default 2025) for our team + one bot,
to audit realism: draft (who/cost/keeper), opening bench, and every weekly add/drop.

Runs the full league through the prior years (so keepers entering the traced year
are realistic), then instruments the traced year. Flags two realism risks:
  - starting a player who is on BYE / not playing that week (would score 0),
  - dropping a YOUNG (<=2 yr exp, keeper-controllable) player for a weekly streamer.
"""
import os, sqlite3, warnings, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import importlib.util
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(os.path.dirname(__file__), n + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V2 = _load("league_sim_v2")
DB = V2.DB
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
TRACE_BOT_STYLE = "balanced"


def main():
    con = sqlite3.connect(DB); D = V2.load(con)
    nm = pd.read_sql("SELECT DISTINCT player_id, player_display_name nm FROM nflv_season", con)
    age = pd.read_sql(f"SELECT player_id, age, years_exp FROM season_dataset WHERE season={YEAR}", con)
    con.close()
    NAME = dict(zip(nm.player_id, nm.nm)); AGE = dict(zip(age.player_id, age.age)); EXP = dict(zip(age.player_id, age.years_exp))
    name = lambda p: NAME.get(p, str(p))
    young = lambda p: (EXP.get(p) is not None and EXP.get(p) <= 2)
    wproj = V2.make_wproj(D)
    teams = [V2.Team(i, V2.STYLES[i]) for i in range(12)]
    yv = {Y: V2.build_year_values(Y, D) for Y in V2.YEARS}
    for Y in V2.YEARS:
        if Y == YEAR: break
        V2.simulate_year(Y, teams, D, yv[Y], wproj)              # warm up keepers

    us = next(i for i in range(12) if teams[i].style == "ours")
    bot = next(i for i in range(12) if teams[i].style == TRACE_BOT_STYLE)
    pos_of, rookies, bot_val, our_val = yv[YEAR]
    WACT = D["WACT"]
    games_wk = {}
    for (s, tm, w) in D["LINE"]:
        if s == YEAR: games_wk.setdefault(w, set()).add(tm)
    pteam = {}
    for (s, pid), d in D["PTEAM"].items():
        if s == YEAR and d: pteam[pid] = pd.Series(d).value_counts().index[0]
    keepers_in = {i: dict(teams[i].keepers) for i in range(12)}

    # ---- DRAFT ----
    kept = {}
    for t in teams:
        t.reset()
        for p, info in t.keepers.items():
            if p in pos_of: t.add(p, pos_of[p], info["price"]); kept[p] = t.tid
    V2.auction(teams, bot_val, our_val, pos_of, rookies, set(kept))
    for t in teams: t.faab = 100

    L = [f"# {YEAR} Season Trace — our team vs a '{TRACE_BOT_STYLE}' bot\n"]
    for tid, who in [(us, "OUR TEAM (projections + VORP)"), (bot, f"BOT — {TRACE_BOT_STYLE}")]:
        t = teams[tid]; kin = keepers_in[tid]
        rows = sorted(t.roster, key=lambda r: -t.prices.get(r[0], 0))
        L += [f"## {who} — draft", "| Player | Pos | Cost | Status |", "|---|---|---|---|"]
        for p, pos in rows:
            kk = kin.get(p); st = f"keeper (yr {kk['years_kept']})" if kk else ("rookie" if p in rookies else "drafted")
            L.append(f"| {name(p)} | {pos} | ${t.prices.get(p,0)} | {st} |")
        L.append(f"\n*Spent ${sum(t.prices.values())}/$200 on {len(t.roster)}. Cheapest bench depth: "
                 + ", ".join(f"{name(p)} (${t.prices.get(p,0)})" for p, _ in rows[-4:]) + ".*\n")

    # ---- WEEKLY ----
    trans = {us: [], bot: []}; started_bye = {us: [], bot: []}
    def lineup(t, w, log_tid=None):
        cand = {}
        for p, pos in t.roster:
            onbye = (pteam.get(p) is not None and w in games_wk and pteam[p] not in games_wk[w])
            if onbye: continue                                   # never start a bye player
            cand.setdefault(pos, []).append((wproj(YEAR, p, w), WACT.get((YEAR, p), {}).get(w, 0.0), p, onbye))
        for pos in cand: cand[pos].sort(key=lambda x: -x[0])
        used = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}; s = 0.0; starters = []
        for pos, n in [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)]:
            for proj, act, p, onbye in cand.get(pos, [])[:n]:
                s += act; starters.append((p, onbye))
            used[pos] = min(n, len(cand.get(pos, [])))
        flexc = []
        for pos in V2.FLEX: flexc += cand.get(pos, [])[used[pos]:]
        if flexc:
            best = max(flexc, key=lambda x: x[0]); s += best[1]; starters.append((best[2], best[3]))
        if log_tid is not None:
            for p, onbye in starters:
                if onbye: started_bye[log_tid].append((w, p))
        return s + V2.KDST_FLAT

    for w, pairs in enumerate(V2.round_robin(list(range(12)), V2.REG_WEEKS), start=1):
        before = {tid: {p for p, _ in teams[tid].roster} for tid in (us, bot)}
        fb = {tid: teams[tid].faab for tid in (us, bot)}
        V2.waivers(teams, YEAR, w, pos_of, D)
        for tid in (us, bot):
            after = {p for p, _ in teams[tid].roster}
            add, drop = after - before[tid], before[tid] - after
            if add or drop: trans[tid].append((w, list(add), list(drop), fb[tid] - teams[tid].faab))
        for a, b in pairs:
            lineup(teams[a], w, a if a in (us, bot) else None)
            lineup(teams[b], w, b if b in (us, bot) else None)

    for tid, who in [(us, "OUR TEAM"), (bot, f"BOT ({TRACE_BOT_STYLE})")]:
        L.append(f"## {who} — in-season waiver moves")
        if not trans[tid]: L.append("- No moves all season.")
        for w, add, drop, faab in trans[tid]:
            adds = ", ".join(f"{name(p)} [{pos_of.get(p,'?')}]" for p in add) or "—"
            drps = ", ".join(f"{name(p)} [{pos_of.get(p,'?')}{(' ⚠️YOUNG' if young(p) else '')}]" for p in drop) or "—"
            L.append(f"- **Wk {w}**: add {adds} · drop {drps} · FAAB −${faab}")
        bb = started_bye[tid]
        L.append(f"\n*Players started while on bye/not playing: {len(bb)}"
                 + (f" → {[(w, name(p)) for w, p in bb[:6]]}" if bb else "") + "*\n")

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models", f"TRACE_{YEAR}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L)); print("Saved", os.path.relpath(out))


if __name__ == "__main__":
    main()
