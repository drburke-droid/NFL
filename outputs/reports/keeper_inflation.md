# Keeper inflation in this league — measured, not assumed

Source: `outputs/espn_drafts.csv` (560 picks, 2023-25) priced against same-season `nflv_ffa_league.ffa_aav`. **2023 had zero keepers** and is the control.

## Money mechanics

| season | teams | budget | keeper $ | open $ available | open $ spent | unspent |
|---|---|---|---|---|---|---|
| 2023 | 12 | $2400 | $0 | $2400 | $2355 | $45 (1.9%) |
| 2024 | 11 | $2200 | $574 | $1626 | $1538 | $88 (5.4%) |
| 2025 | 12 | $2400 | $478 | $1922 | $1802 | $120 (6.2%) |

Keeper drafts leave real money unspent — 2023: 1.9%, 2024: 5.4%, 2025: 6.2%. That is why realised inflation lands below what a fully-cleared auction implies.

## Headline

- Room baseline (2023, keeper-free): **0.904x** national AAV — this league underpays.
- **Keeper inflation: 1.153x**, bootstrap 95% CI **[1.08, 1.24]** (n=161 control picks, 243 keeper-era).
- OLS `log(paid/AAV) ~ keeper_era * log(AAV)` (n=404): keeper coefficient p=0.0095 -> **1.173x** at mean price.

## Rejected hypotheses

### Position-specific inflation — NOT supported

| pos | 2023 base | keeper-era | inflation | 95% CI | vs global |
|---|---|---|---|---|---|
| QB | 0.860 | 0.838 | 0.975 | [0.80, 1.27] | overlaps |
| RB | 0.902 | 1.040 | 1.152 | [1.01, 1.30] | overlaps |
| WR | 0.942 | 1.109 | 1.177 | [1.08, 1.29] | overlaps |
| TE | 0.782 | 1.081 | 1.382 | [1.08, 1.91] | overlaps |

Point estimates tempt you (TE looks hottest), but **every CI overlaps the global estimate** on 14-52 picks per position-season. Do not ship per-position multipliers.

### Price-tier gradient — NOT supported

The interaction term is insignificant (p=0.4984) and flips sign. A binned view suggests cheap players inflate ~1.5x, but that is an artifact of bids being floored at $1 against sub-$1 AAVs.

### Keeper composition — NO detectable effect

Does keeping most of a position inflate the survivors (scarcity) or deflate them (nobody still needs one)? **Neither, measurably**: corr(share kept, inflation) = +0.130, p=0.759 (n=8 position-seasons). Underpowered, but no signal.

## Why keepers inflate at all

- **2024**: 30 keepers cost $570 but carry $891 of AAV value — $321 of value leaves the pool free (1.56x). Surviving money chases a thinner pool.
- **2025**: 30 keepers cost $478 but carry $912 of AAV value — $434 of value leaves the pool free (1.91x). Surviving money chases a thinner pool.

## Applied

`docs/index.html` `dynamicMarket()`: ceiling **1.6 -> 1.25** (top of the measured CI) and a **0.94** unspent-money factor. The old 1.6 let Exp $ run ~22% past anything this league has ever paid. Note `PRICE_ANCHOR` is already fitted on 2023-25 bids (two keeper years), so it embeds the keeper effect — `infl` must only carry this room's money surplus, never the keeper effect twice.
