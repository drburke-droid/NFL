# Systematic signal mining over props — methodology + results (2026-07-05)

Reproducible: `scripts/signal_miner.py`. Question: is there ANY combination of stats, advanced stats,
game context, or TEAMMATE PROP LINES that predicts what the closing market missed? 29,410 settled
consensus props 2023–25, 49 features including the new teammate-line family (own line share of team
total, teammates' combined line, market-designated alpha flag, team QB pass-yds line, line-vs-form).

## The method (how to mine millions of combinations without fooling yourself)
1. **Target = residual (outcome − de-vigged market prob), never raw ROI.** The market embeds most
   information; signal must predict what it MISSED.
2. **Stage 1 — boosting as the combination tester.** One LGBM over all features tests millions of
   interactions implicitly with regularization, season-blocked walk-forward. If it can't beat the
   market prob, no explicit rule enumeration will either (rules are a subset of tree paths).
3. **Stage 2 — interpretable rule miner with guardrails**: n≥200 per rule, effect same-sign in both
   2024 AND 2025, Benjamini–Hochberg FDR q<0.10 across ALL 450 rules evaluated. (Millions of raw
   combos are pointless — enumerate only interpretable templates; boosting covers the rest.)
4. **Stage 3 — true holdout**: 2023 was excluded from mining entirely; survivors must replicate there.
5. (Stage 4, future): survivors paper-trade forward on live 2026 data before real money.

## Results
- **Stage 1: NO.** LGBM(all features incl. teammate lines) loses to raw market prob in both test
  years (2024: .26592 vs .24712; 2025: .25189 vs .24784). The interaction space is empty at close.
  The teammate-prop family adds nothing the line hasn't priced.
- **Stage 2: 187 FDR survivors — all ONE latent factor.** Every surviving rule re-expresses the
  over-shading bias, which is ~2x the global size for **big underdogs** (spread≥p80: −4.8pp) and
  **star receivers** (target-share≥p80: −4.6pp).
- **Stage 3: the holdout kills the "law".** 2023 over-residual is **+0.6pp** (no bias; blind unders
  −7.7% at median price). The bias is a **regime that started in 2024** and doubled in 2025:
  | segment | 2023 | 2024 | 2025 | 2025 under-ROI @ median |
  |---|--:|--:|--:|--:|
  | global | +0.6pp | −1.8 | −2.6 | −1.5% |
  | underdogs (spread≥p80) | −0.3 | −3.5 | −6.0 | **+4.8%** |
  | star receivers (tgt≥p80) | −0.1 | −3.1 | −6.3 | **+5.2%** |

## The lesson (answer to "how do I test millions of combinations")
A rule that was 6-sigma significant, FDR-controlled, and consistent across two seasons **still failed
the true holdout** — because the underlying phenomenon is a market regime, not a law. This is the
strongest possible demonstration of the discipline: without Stage 3 we'd have "discovered" 187 edges.
What actually exists: ONE drifting factor (books shading overs harder each year since 2024, most in
underdog/star-receiver props). Betting 2026 unders in those segments is a REGIME-PERSISTENCE bet
(+5% at median price if the 2025 regime holds; −6% if it reverts to 2023). The Tuesday-snapshot
capture remains the prerequisite for any of this to become investable — early lines + regime
monitoring (track the monthly over-residual; bet only while it stays < −3pp).
