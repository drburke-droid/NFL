# Teammate-opportunity signals: which "niche situations" are real?

The season model uses a player's OWN prior stats, so it can miss structural changes
to his situation — touches vacated by a departure, or stolen by an arrival. Tested a
family of pre-specified, mechanism-driven situations against the walk-forward
projection RESIDUAL (next_ppg − model P50), requiring replication in two disjoint
eras. `scripts/test_opportunity_signals.py`. Leakage-free (prior-year volume +
known roster membership).

## Results

| Situation | n | beats proj by | verdict |
|---|---|---|---|
| **RB: lead back departed (≥150 carries vacated, no real replacement)** | 68 | **+1.71** [+0.6,+2.8] | ✅ **real & large**, replicates (+2.3 / +1.2) |
| WR/TE: targets vacated (≥100) | 936 | +0.20 [+0.0,+0.4] | ⚠️ real but tiny |
| WR: new featured back arrived (≥150 inc carries) | 265 | +0.15 [−0.2,+0.5] | ❌ null |
| RB: backfield crowded (≥150 inc or rookie RB) | 175 | −0.12 [−0.7,+0.4] | ❌ n.s. |
| WR/TE: new pass-catcher added (≥100 inc targets) | 551 | −0.02 | ❌ null |
| RB: top-50 rookie RB drafted ahead | 41 | −0.98 [−2.0,+0.1] | ❌ suggestive only |

**Only the RB-inherits-vacated-role signal survives** — and it's the largest niche
edge found to date (~+1.7 PPG, ~+27 pts/season). Face validity is textbook: James
Conner '18 (4.4 → 21.5 when Bell sat), Kyren Williams '23, Rachaad White '23,
CMC '18, Marlon Mack '18, Travis Etienne '25. The model rated each as a backup off
his own prior stats and missed the role opening up.

Notably, the user's other guesses (new RB1 suppresses the WR1; WR2 inheriting WR1's
targets) do **not** hold up: targets are fungible and spread across more players, and
offenses adapt to a new back. Honest nulls — worth knowing so we don't chase them.

## Productization (display tag, no projection override)
`build_opportunity.py` → `nflv_opportunity` (2013-2026). For 2026 it pulls the
CURRENT post-FA/draft rosters (`load_rosters(2026)` → `nflv_rosters_2026`; 47/49 lead
RBs matched) to detect this year's vacancies. `vacated_role=1` for a returning RB
with ≥150 vacated carries, <100 incoming, no top-50 rookie RB; the board then keeps
only the **top returning RB per team** (pred_ppg ≥6) so the tag points at the actual
beneficiary, not every backup.

**2026: Chuba Hubbard (CAR)** — Rico Dowdle's 236 carries left for PIT with no real
replacement. Shown as `⬆ VACATED ROLE` on the board.

Effect is large enough (+1.7, CI excludes 0, replicates) that it could justify an
actual model feature, not just a tag — a reasonable next step if desired.
