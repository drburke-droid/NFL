# What changes if we ignore undraftable players? (12 teams x 16 = 192 spots)

`scripts/test_pool_trim.py`. Two separate questions.

## (1) Value side — nothing changes (provably)
Of 680 board players, the bottom 488 are "undraftable":
- All 488 have **negative VORP** (−240 .. −28): every one is below replacement.
- Only **108 players have positive VORP**; **zero** of the tail do. The auction engine
  only allocates money across positive-VORP players.
- Replacement levels are anchored to a **fixed rank** (starter slots), independent of
  pool size, so dropping the tail leaves replacement points, VORP, tiers, and auction $
  **identical**.

Conclusion: trimming the tail is purely cosmetic (declutters the display). It confirms
the value system is correctly rank-anchored, not pool-dependent.

## (2) Model side — retraining on draftable-caliber only is slightly WORSE
If we also retrained projections on relevant players only (prior PPG floor QB12/RB8/
WR8/TE6), walk-forward, evaluated on the draftable players we care about:

| Metric | Train ALL | Train relevant-only |
|---|---|---|
| Central MAE (all) | 3.144 | 3.135 (≈tie) |
| Bust AUC | 0.795 | **0.784 (worse)** |
| Boom AUC | 0.793 | 0.789 (worse) |

Central accuracy is a wash (small QB gain vs WR loss); bust/boom calibration DEGRADES.
The low-projected players are informative training examples — the model's anchor for
what a decline looks like. Removing them hurts bust detection among draftable players.

**Decision:** keep training on the full pool; trimming is a display-only convenience.
