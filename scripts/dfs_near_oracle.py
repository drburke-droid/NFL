"""
Every legal DK Classic roster scoring >= 95% of the slate's oracle, 2016-25 main slates, and what they
have in common that was knowable PREGAME.

Enumeration (exact, no sampling):
  1. prune: a player stays only if the best roster that contains him (integer program with him forced in)
     reaches the threshold;
  2. exhaustively combine the survivors: for each roster shape (RB/WR/TE = 3/3/1, 2/4/1, 2/3/2) build all
     RB, WR and TE combinations, join them with pruning on remaining-best bounds, then add QB and DST.

Per roster we record pregame traits only (projection = FFA DK points as of kickoff, salary, Vegas):
  proj_frac   roster projection / max-projection roster's projection
  salary      cap used
  stack       own WR/TE/RB with the QB;  bringback  opponents of the QB's team
  max_game    most players from one game;  games  distinct games
  dst_vs      players facing own DST
  punts       non-DST players under $4,000
  flex        position in the FLEX slot (the 3rd RB / 4th WR / 2nd TE)
  qb_imp      QB's team implied total;  top_game  players from the slate's highest-total game
  proj_rank   mean projection rank of the 8 non-DST players within their position on the slate
  value_rank  mean rank of projected points per $1k within position
  chalk_n     players who are in the max-projection roster
Baselines per slate: 20,000 random legal rosters (salary >= 45k), and the 150 jittered-projection rosters
our MME method builds (dfs_mme_oracle jitter15). Slates are weighted equally.

    python scripts/dfs_near_oracle.py --procs 3     # checkpoints per slate in outputs/dfs/near_oracle/
"""
import argparse, itertools, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfs_optimizer import Slate, CAP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "dfs")
CK = os.path.join(OUT, "near_oracle")
THR = 0.95
MAX_KEEP = 200000
SHAPES = [(3, 3, 1, "RB"), (2, 4, 1, "WR"), (2, 3, 2, "TE")]


def combos(ix, k, pts, sal):
    if len(ix) < k: return np.zeros((0, k), int), np.zeros(0), np.zeros(0)
    c = np.array(list(itertools.combinations(ix, k)), int)
    return c, pts[c].sum(1), sal[c].sum(1)


def enumerate_near(sl, thr):
    a, sal, pos = sl.actual.values, sl.salary.values.astype(float), sl.pos.to_numpy(dtype=object)
    S = Slate(sl)
    keep = []
    for i in range(len(sl)):
        if a[i] <= 0 and pos[i] != "DST": continue
        sc = a.copy(); sc[i] += 1e4
        idx = S.solve(sc)
        if idx is not None and i in idx and a[idx].sum() >= thr: keep.append(i)
    keep = np.array(keep)
    by = {p: keep[pos[keep] == p] for p in ("QB", "RB", "WR", "TE", "DST")}
    best = {p: np.sort(a[by[p]])[::-1] for p in by}
    out = []
    for nr, nw, nt, flex in SHAPES:
        R = combos(by["RB"], nr, a, sal); W = combos(by["WR"], nw, a, sal); T = combos(by["TE"], nt, a, sal)
        if min(len(R[0]), len(W[0]), len(T[0])) == 0 or len(best["QB"]) == 0 or len(best["DST"]) == 0: continue
        rest = best["QB"][0] + best["DST"][0]
        # RB x WR with bound
        ub_t = T[1].max()
        ri, wi = np.meshgrid(np.arange(len(R[0])), np.arange(len(W[0])), indexing="ij")
        ri, wi = ri.ravel(), wi.ravel()
        p_rw = R[1][ri] + W[1][wi]; s_rw = R[2][ri] + W[2][wi]
        ok = (p_rw + ub_t + rest >= thr) & (s_rw <= CAP)
        ri, wi, p_rw, s_rw = ri[ok], wi[ok], p_rw[ok], s_rw[ok]
        for ti in range(len(T[0])):
            p3 = p_rw + T[1][ti]; s3 = s_rw + T[2][ti]
            ok = (p3 + rest >= thr) & (s3 <= CAP)
            if not ok.any(): continue
            for q in by["QB"]:
                for dd in by["DST"]:
                    p = p3[ok] + a[q] + a[dd]; s = s3[ok] + sal[q] + sal[dd]
                    good = (p >= thr) & (s <= CAP)
                    if not good.any(): continue
                    rr, ww = ri[ok][good], wi[ok][good]
                    rows = np.c_[np.full(len(rr), q), R[0][rr], W[0][ww], np.tile(T[0][ti], (len(rr), 1)),
                                 np.full(len(rr), dd)]
                    out.append(rows)
                    if sum(len(x) for x in out) > MAX_KEEP: return np.vstack(out), True
    return (np.vstack(out) if out else np.zeros((0, 9), int)), False


def traits(sl, L, maxproj_idx):
    """Pregame traits for an (m x 9) array of roster indices."""
    obj = lambda c: sl[c].to_numpy(dtype=object)                     # arrow strings can't take 2-D indexing
    pos, team, opp, gid = obj("pos"), obj("team"), obj("opp"), obj("game_id")
    proj, sal, imp = sl.proj.values, sl.salary.values, sl.implied.fillna(22).values
    mp = proj[maxproj_idx].sum()
    gtot = sl.groupby("game_id").implied.sum(); top_g = gtot.idxmax() if gtot.notna().any() else None
    pr = sl.groupby("pos").proj.rank(ascending=False).values
    vr = (sl.proj / sl.salary * 1000).groupby(sl.pos).rank(ascending=False).values
    chalk = np.zeros(len(sl), bool); chalk[maxproj_idx] = True
    is_qb = pos[L] == "QB"; is_dst = pos[L] == "DST"
    qb = L[np.arange(len(L)), is_qb.argmax(1)]; dst = L[np.arange(len(L)), is_dst.argmax(1)]
    skill = ~is_qb & ~is_dst
    tL, oL, gL = team[L], opp[L], gid[L]
    own = (tL == team[qb][:, None]) & skill
    bring = (tL == opp[qb][:, None]) & skill
    dst_vs = ((tL == opp[dst][:, None]) & ~is_dst).sum(1)
    games = np.array([len(set(r)) for r in gL])
    max_game = np.array([np.unique(r, return_counts=True)[1].max() for r in gL])
    npos = lambda p: (pos[L] == p).sum(1)
    flex = np.where(npos("RB") == 3, "RB", np.where(npos("WR") == 4, "WR", "TE"))
    nd = np.where(skill | is_qb, 1, 0)
    return pd.DataFrame(dict(
        proj_frac=proj[L].sum(1) / mp, salary=sal[L].sum(1), stack=own.sum(1), bringback=bring.sum(1),
        max_game=max_game, games=games, dst_vs=dst_vs,
        punts=((sal[L] < 4000) & ~is_dst).sum(1), flex=flex, qb_imp=imp[qb],
        top_game=(gL == top_g).sum(1) if top_g is not None else 0,
        proj_rank=(pr[L] * nd).sum(1) / 8, value_rank=(vr[L] * nd).sum(1) / 8, chalk_n=chalk[L].sum(1)))


def random_rosters(sl, n, rng):
    pos, sal = sl.pos.to_numpy(dtype=object), sl.salary.values
    ix = {p: np.where(pos == p)[0] for p in ("QB", "RB", "WR", "TE", "DST")}
    out = []
    while sum(len(x) for x in out) < n:
        m = 4 * n
        shape = rng.integers(0, 3, m)
        cols = [rng.choice(ix["QB"], m), rng.choice(ix["DST"], m)]
        R = np.stack([rng.choice(ix["RB"], m) for _ in range(3)], 1)
        W = np.stack([rng.choice(ix["WR"], m) for _ in range(4)], 1)
        T = np.stack([rng.choice(ix["TE"], m) for _ in range(2)], 1)
        flexp = np.where(shape == 0, R[:, 2], np.where(shape == 1, W[:, 3], T[:, 1]))
        L = np.c_[cols[0], R[:, :2], W[:, :3], T[:, :1], flexp, cols[1]]
        s = sal[L].sum(1)
        ok = (s <= CAP) & (s >= 45000) & np.array([len(set(r)) == 9 for r in L])
        out.append(L[ok])
    return np.vstack(out)[:n]


def run(args):
    sl, path, seed = args
    try:
        rng = np.random.default_rng(seed)
        sl = sl.reset_index(drop=True)
        if not (sl.pos == "DST").any(): return None
        a = sl.actual.values
        S = Slate(sl); oracle = a[S.solve(a)].sum()
        L, capped = enumerate_near(sl, THR * oracle)
        hasp = sl[(sl.proj > 0.5) | (sl.pos == "DST")]
        mp_idx = hasp.index.values[Slate(hasp.reset_index(drop=True)).solve(hasp.proj.values)]
        frames = []
        if len(L):
            t = traits(sl, L, mp_idx); t["score"] = a[L].sum(1) / oracle; t["set"] = "near"; frames.append(t)
        Rr = random_rosters(sl, 20000, rng)
        t = traits(sl, Rr, mp_idx); t["score"] = a[Rr].sum(1) / oracle; t["set"] = "random"; frames.append(t)
        sub = hasp.reset_index(drop=True); Sb = Slate(sub); pj = sub.proj.values
        J = np.array([hasp.index.values[Sb.solve(pj * np.clip(1 + rng.normal(0, .15, len(pj)), 0, None))]
                      for _ in range(150)])
        t = traits(sl, J, mp_idx); t["score"] = a[J].sum(1) / oracle; t["set"] = "ours"; frames.append(t)
        f = pd.concat(frames, ignore_index=True)
        f["season"], f["week"], f["oracle"], f["n_near"], f["capped"] = sl.season.iat[0], sl.week.iat[0], oracle, len(L), capped
        # player-level: how often each player appears in near rosters
        if len(L):
            cnt = np.bincount(L.ravel(), minlength=len(sl)) / len(L)
            pl = sl[["season", "week", "player", "pos", "team", "salary", "proj", "actual", "implied"]].copy()
            pl["near_share"] = cnt; pl["oracle_pick"] = np.isin(np.arange(len(sl)), S.solve(a))
            pl.to_parquet(path.replace(".parquet", "_players.parquet"), index=False)
        # keep the file small: near rosters sampled to 5k for traits
        keep = pd.concat([f[f.set != "near"], f[f.set == "near"].sample(min(5000, (f.set == "near").sum()), random_state=0)])
        keep.to_parquet(path, index=False)
        return sl.season.iat[0], sl.week.iat[0], len(L), capped
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}", flush=True); return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seasons", default="2016-2025")
    A = ap.parse_args()
    a, b = (A.seasons.split("-") + [A.seasons])[:2]
    os.makedirs(CK, exist_ok=True)
    d = pd.read_parquet(os.path.join(OUT, "slates.parquet"))
    jobs = []
    for (y, w), sl in d[(d.season >= int(a)) & (d.season <= int(b))].groupby(["season", "week"]):
        p = os.path.join(CK, f"{y}_{w:02d}.parquet")
        if not os.path.exists(p): jobs.append((sl, p, y * 100 + w))
    print(f"{len(jobs)} slates to run", flush=True); t0 = time.time()
    with Pool(A.procs) as p:
        for i, r in enumerate(p.imap_unordered(run, jobs)):
            if r: print(f"  {r[0]} wk{r[1]}: {r[2]:,} near-oracle rosters{' (capped)' if r[3] else ''}  [{i + 1}/{len(jobs)} {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
