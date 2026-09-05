"""Global tests: (1) chi-square dispersion of franchise totals vs multinomial null;
(2) same but on a null that PRESERVES within-team-season stacking (permute team labels
of team-season count vectors within season) — the honest null, since one elite QB
creates 3-4 startable slots on the same team."""
import numpy as np, pandas as pd, scipy.stats as st, os

SCRATCH = os.path.dirname(os.path.abspath(__file__))
pv = pd.read_csv(os.path.join(SCRATCH, "franchise_panel.csv"), index_col=0)

k = pv.sum()
yrs = pv.notna().sum()
exp = np.array([pv.loc[y].notna().sum() for y in pv.index])
slot = pv.sum(axis=1)
per_season_exp = slot / pv.notna().sum(axis=1)  # slots / live teams, per season
expv = (pv.notna().mul(per_season_exp, axis=0)).sum()
chi2 = (((k - expv) ** 2) / expv).sum()
df = len(k) - 1
print(f"[G1] chi2 dispersion vs multinomial null: chi2={chi2:.1f}, df={df}, "
      f"p={st.chi2.sf(chi2, df):.4f}")

# stacking-preserving permutation: shuffle which franchise got each team-season count
rng = np.random.default_rng(11)
NS = 10000
mat = pv.values
chis = np.zeros(NS)
for s in range(NS):
    perm = mat.copy()
    for i in range(mat.shape[0]):
        row = perm[i]
        live = ~np.isnan(row)
        vals = row[live]
        rng.shuffle(vals)
        row[live] = vals
    tot = np.nansum(perm, axis=0)
    chis[s] = (((tot - expv.values) ** 2) / expv.values).sum()
print(f"[G2] stacking-preserving null: observed chi2={chi2:.1f}, "
      f"null mean={chis.mean():.1f}, 95th={np.quantile(chis, 0.95):.1f}, "
      f"p={ (chis >= chi2).mean():.4f}")

# same for persistence: is spearman(count_t, count_t+1) explained by shuffling?
obs_pairs = []
for i in range(mat.shape[0] - 1):
    a, b = mat[i], mat[i + 1]
    m = ~np.isnan(a) & ~np.isnan(b)
    obs_pairs.append(np.column_stack([a[m], b[m]]))
op = np.vstack(obs_pairs)
obs_rho = st.spearmanr(op[:, 0], op[:, 1]).statistic
rhos = np.zeros(2000)
for s in range(2000):
    perm = mat.copy()
    for i in range(mat.shape[0]):
        row = perm[i]; live = ~np.isnan(row); vals = row[live]
        rng.shuffle(vals); row[live] = vals
    pp = []
    for i in range(mat.shape[0] - 1):
        a, b = perm[i], perm[i + 1]
        m = ~np.isnan(a) & ~np.isnan(b)
        pp.append(np.column_stack([a[m], b[m]]))
    ppv = np.vstack(pp)
    rhos[s] = st.spearmanr(ppv[:, 0], ppv[:, 1]).statistic
print(f"[G3] persistence rho={obs_rho:.3f} vs shuffled-label null "
      f"(mean {rhos.mean():.3f}, 95th {np.quantile(rhos, 0.95):.3f}), "
      f"p={(rhos >= obs_rho).mean():.4f}")
