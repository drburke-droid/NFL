# $1–3 Lottery Tickets — what actually predicts a cheap player becoming a star (2026-07-01)

Reproducible: `scripts/lottery_tickets.py`. Cohort = preseason auction value ≤ $3 (FFA AAV ≤3 or
unranked), 2014–2025, n=4,006. **Star** = next-season positional finish in the upper half of startable
VORP ranks (QB≤6, RB≤12, WR≤12, TE≤6). **Startable** = QB12/RB24/WR24/TE12.

## Base rates — the honest odds
Star **2.6%**, startable 6.4%. By position: **QB 3.6% / TE 3.0% / RB 2.9% / WR 1.9%** — WR is the
WORST lottery position (deepest pool, hardest to leap the incumbents); QB/TE are the best.

## Signal lift table (P(star | signal) vs 2.6% base)
| signal | n | star | lift | startable |
|---|--:|--:|--:|--:|
| **prior target share ≥ 12%** | 565 | **9.2%** | **3.5x** | 20.9% |
| **H2 PPG ≥ 8 (finished the year producing)** | 986 | 6.6% | 2.5x | 16.3% |
| **prior snap% ≥ 50** | 1273 | 6.4% | 2.5x | 14.3% |
| prior PPG ≥ 7 (was semi-relevant) | 1329 | 6.5% | 2.5x | 15.1% |
| **draft capital ≤ pick 75** | 987 | 5.9% | 2.3x | 13.8% |
| capital ≤75 AND yrs-exp ≤2 (post-hype) | 268 | 6.0% | 2.3x | **17.2%** |
| xFP over-producer (+1.5/wk efficiency) | 307 | 4.6% | 1.8x | 11.4% |
| — MYTH: "unlucky" (xFP gap ≤ −1.5/wk) | 586 | 1.9% | **0.7x** | 5.8% |
| — MYTH: late H2 snap SURGE (+10pp delta) | 751 | 1.2% | **0.5x** | 4.3% |
| — POISON: cheap veteran changing teams | 1290 | **0.9%** | **0.4x** | 2.9% |

**The pattern: LEVEL beats DELTA.** A cheap player who *ended* last season with a real role (targets,
snaps, H2 production) hits; a player whose role merely *grew* (surge without level) or who "deserved
better" by expected fantasy points does NOT. Efficiency carries forward (xFP over-producers 1.8x);
bad luck does not mean-revert into stardom. And a cheap vet on a new team is the classic trap (0.4x)
— consistent with the aging/mover bias correction on the main model.

## Walk-forward model (2016–2025, n=3,204)
AUC **0.789**; top-10/season picks hit **8.0% star (6.7x base)** and **20% startable** — i.e., a
bench of ~10 screened tickets returns ~1 star + 2 startables/yr vs ~0.1 star unscreened. Top features:
xgap_pg, weight, prior_ppg, H2 deltas, prior2_ppg, cv, forty. Historical top-10 hits: Ertz'17 (#3),
**Dak'19 (#2)**, Hooper'19, Hockenson'20, Engram'22, Higbee'22, **Ferguson'25 (#5)**, M.Wilson'25 —
the model's archetype is the entrenched-role TE/QB the market ignores.

## 2026 top tickets (exp ≤ $3 on our board, model-ranked)
QB: Dak 10.9%, Rodgers 7.8%, Wentz 6.6%, Lawrence 5.4% (H2 28 PPG!), Purdy 4.0%, Love, Stroud, Daniels
TE: **Ferguson 5.9%, Andrews 5.7%, Otton 4.2%, Strange 3.4%, Goedert 2.9%, Mason Taylor 2.7%, Kraft 2.7%**
WR: Deebo 7.6% (unsigned — verify landing), Doubs 5.8% (moved: discount it), Kupp 3.7%, Shakir 2.6%, Watson 2.0%
RB: thin — Jerome Ford 2.5% (moved), Kimani Vidal 2.0%

## Bench strategy (my roster: TE-punt path, Allen kept → skip QB tickets)
Fill 4–5 of 7 bench slots with tickets **weighted TE > WR > RB**: the TE punt makes Ferguson/Andrews/
Otton/Kraft $1 buys doubly valuable (startable TE odds ~15–20% each + keeper-gold if they hit at $1).
Prefer level-holders (target share, snaps, H2 production, draft pedigree); avoid cheap movers and
"unlucky" narratives. Remaining bench = RB depth from the value tier / handcuff to my RB1.
