# Does first-week-of-September buzz predict dart hits? (No.)

**Question** (2026-09-01): the validated August wiki-spike works on the cheap pool —
does any *additional* signal develop in Sep 1–7 (post-cutdown, pre-Week-1)?

**Data**: daily Wikipedia pageviews Aug 25 – Sep 7 backfilled for 4,416 cheap-pool
($1–2 dart) player-seasons 2016–2025 (`nflv_wiki_buzz_daily_sep`, kept separate from
the production `nflv_wiki_buzz_daily`). Clean window = Sep 1 → min(Sep 7, opener−1),
so zero in-season contamination. Base hit rate 2.4%.

## Pre-registered tests

| # | Test | Result |
|---|------|--------|
| 1 | Hit rate by Sep-spike (vs Jan–May base) quantile | **Flat**: 2.5 / 2.4 / 2.3 / 2.0 / 2.5% — no gradient |
| 2a | Logit hit ~ aug_spike_pctl + sep_vs_aug_pctl | sep coef +0.49, bootstrap P(≤0)=0.056 — marginal, contradicted by every cell view; multiplicity-tier noise |
| 2b | Actionable cell: Aug-quiet × Sep-surge (≥90th) | **2.4% (n=331) = exactly base**, p=0.86; ≤1 hit/season 2016–22 |
| 3 | Per-season top-10 by Sep-vs-Aug surge | **1.0% pooled — *below* the 2.4% base**; top-10 by sep_spike 4%, by aug_spike 3% (both ≈ noise as raw sorts) |
| 4 | Autocorrelation | corr(Aug pctl, Sep pctl) = 0.69 — Sep spike is mostly the Aug spike persisting, no new info |
| bonus | Cutdown week (Aug 25–31) surge, same cell | 2.6% (n=229) = base |

## Why September buzz fails where August buzz works

The all-time top Aug-quiet Sep-surgers tell the story: Ricky Pearsall 2024 (shot on
Aug 31), Sam Bradford 2016 (traded Sep 3), Josh Rosen 2020 (released), Sammie Coates,
Amara Darboh (cut). **August camp buzz on a cheap player = job-seizing. September
first-week buzz = transaction/injury churn** — cuts, trades, and bad news — which is
direction-ambiguous at best. 14 of the top 15 surgers busted (lone hit: Jermaine
Kearse 2017, traded into a starting job — the one *good*-news transaction).

## Verdict

**Dead. Don't extend the buzz window past August.** The Aug 1–31 spike (rate-scaled,
fed to the dart classifier) remains the only live wiki signal, and today's final pull
already captures all of it. A Sep-surge screen would actively *hurt* (top-10 hit 1%
vs 2.4% base) because it selects for calamity, not opportunity.

Study: `scripts/sep_week1_buzz_study.py` (backfill: `scripts/backfill_sep_wiki_daily.py`);
table `nflv_wiki_buzz_daily_sep` retained in the DB for future re-checks.
