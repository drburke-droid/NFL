"""
DraftKings NFL Classic lineup optimizer (integer program on scipy's HiGHS milp, no extra install).

Roster: QB, RB, RB, WR, WR, WR, TE, FLEX (RB/WR/TE), DST; salary <= 50,000.
    -> QB = 1, DST = 1, RB 2-3, WR 3-4, TE 1-2, RB+WR+TE = 7.

Options (all optional, used by the backtest's strategies):
    stack=n      the QB's team supplies >= n WR/TE
    bringback=n  the QB's opponent supplies >= n RB/WR/TE
    no_dst_vs    no offensive player facing the chosen DST
    exclude      list of previous lineups (index arrays); each new lineup must differ by >= `min_diff` players
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

CAP = 50000


class Slate:
    """Static constraint rows for one slate; solve() swaps only the objective."""

    def __init__(self, df, stack=0, bringback=0, no_dst_vs=False):
        self.df = df.reset_index(drop=True)
        d = self.df
        n = len(d)
        pos = d.pos.values
        rows, lo, hi = [], [], []

        def add(coef, l, h):
            rows.append(coef); lo.append(l); hi.append(h)

        is_ = lambda *p: np.isin(pos, p).astype(float)
        add(d.salary.values.astype(float), 0, CAP)
        add(is_("QB"), 1, 1); add(is_("DST"), 1, 1)
        add(is_("RB"), 2, 3); add(is_("WR"), 3, 4); add(is_("TE"), 1, 2)
        add(is_("RB", "WR", "TE"), 7, 7)
        teams = d.team.values; opps = d.opp.values
        if stack or bringback:
            for i in np.where(pos == "QB")[0]:
                t, o = teams[i], opps[i]
                if stack:
                    c = ((teams == t) & np.isin(pos, ["WR", "TE"])).astype(float); c[i] = -stack
                    add(c, 0, np.inf)
                if bringback:
                    c = ((teams == o) & np.isin(pos, ["RB", "WR", "TE"])).astype(float); c[i] = -bringback
                    add(c, 0, np.inf)
        if no_dst_vs:
            for i in np.where(pos == "DST")[0]:
                c = ((teams == opps[i]) & (pos != "DST")).astype(float); c[i] = 8.0
                add(c, 0, 8)
        self.A = np.array(rows); self.lo = np.array(lo); self.hi = np.array(hi)
        self.n = n

    def solve(self, score, exclude=(), min_diff=1):
        A, lo, hi = self.A, self.lo, self.hi
        if exclude:
            ex = np.zeros((len(exclude), self.n))
            for k, idx in enumerate(exclude): ex[k, idx] = 1
            A = np.vstack([A, ex]); lo = np.r_[lo, np.full(len(exclude), -np.inf)]
            hi = np.r_[hi, np.full(len(exclude), 9 - min_diff)]
        r = milp(c=-np.asarray(score, float), constraints=LinearConstraint(A, lo, hi),
                 integrality=np.ones(self.n), bounds=Bounds(0, 1), options={"presolve": True})
        if r.x is None: return None
        return np.where(r.x > 0.5)[0]
