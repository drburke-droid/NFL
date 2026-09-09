# Early-season waiver-wire RB study (2026-09-09)

Universe = RBs outside the preseason FFA top-36 (undrafted in a 12-team league) who played through week k. HIT = finishes top-24 RB by rest-of-season PPR/game (>=4 games); STAR = top-12. 2013-25 REG.


## After week 1: 611 available RBs, 38 hits (6.2%), 11 stars (1.8%)

| signal | AUC (hit) | top-decile hit rate | top-decile star rate | mean ROS ppg top decile |
|---|---|---|---|---|
| touches_g | 0.750 | 20% (n=75) | 7% | 9.0 |
| ppg | 0.749 | 21% (n=63) | 6% | 9.0 |
| carries_g | 0.724 | 15% (n=71) | 4% | 8.6 |
| snap_pct | 0.724 | 19% (n=52) | 12% | 9.4 |
| last_snap | 0.724 | 19% (n=52) | 12% | 9.4 |
| pre_rank | 0.722 | 21% (n=63) | 5% | 9.3 |
| car_share | 0.715 | 13% (n=62) | 5% | 8.4 |
| tshare | 0.712 | 16% (n=62) | 6% | 8.9 |
| targets_g | 0.706 | 14% (n=63) | 6% | 8.9 |
| draft_round | 0.598 | 9% (n=103) | 3% | 7.6 |
| tds | 0.591 | 14% (n=77) | 5% | 7.6 |
| ypc | 0.586 | 11% (n=46) | 2% | 6.6 |
| epa | 0.537 | 17% (n=46) | 4% | 7.5 |
| years_exp | 0.527 | 4% (n=68) | 1% | 5.9 |
| age | 0.510 | 5% (n=61) | 2% | 6.1 |

Simple screens after week 1 (available RBs):

| rule | n | RB2+ (top-24) | RB1 (top-12) | flex (top-36) | ROS ppg |
|---|---|---|---|---|---|
| snap_pct >= 50% | 52 | 19% | 12% | 33% | 9.4 |
| snap_pct >= 60% | 25 | 20% | 12% | 28% | 9.8 |
| car_share >= 50% | 31 | 13% | 3% | 29% | 8.4 |
| touches/g >= 12 | 75 | 20% | 7% | 33% | 9.0 |
| touches/g >= 15 | 39 | 21% | 8% | 33% | 9.3 |
| targets/g >= 3 | 131 | 18% | 5% | 33% | 8.5 |
| RB1 absent & car_share >= 40% | 6 | 0% | 0% | 0% | 6.1 |
| snap >= 50% & draft rd <= 3 | 12 | 33% | 17% | 50% | 10.8 |
| snap >= 50% & age <= 25 | 29 | 24% | 17% | 34% | 10.5 |
| touches >= 12 & targets >= 2 | 50 | 24% | 8% | 40% | 9.7 |
| snap >= 55% & touches >= 12 & age <= 26 | 17 | 29% | 18% | 35% | 11.2 |

Leave-one-season-out logistic on (snap_pct, car_share, touches_g, targets_g, draft_round, age, rb1_absent, ppg): top-5 per season hit rate **11%** (base rate 6%), star rate 7%, mean ROS ppg 9.0.
Coefficients (standardised): snap_pct +0.13, car_share -0.02, touches_g +0.21, targets_g +0.02, draft_round -0.34, age -0.25, rb1_absent -0.05, ppg +0.28

Top-5 picks per season after week 1 (what the rule would have told you):

| season | player | team | snap% | touches/g | RB1 absent | ROS ppg | hit |
|---|---|---|---|---|---|---|---|
| 2025 | Dylan Sampson | CLE | 43% | 20.0 | 0 | 4.9 |  |
| 2025 | Trey Benson | ARI | 33% | 9.0 | 0 | 0.0 |  |
| 2025 | Braelon Allen | NYJ | 31% | 6.0 | 0 | 0.0 |  |
| 2025 | Nick Chubb | HOU | 51% | 13.0 | 0 | 5.9 |  |
| 2025 | DJ Giddens | IND | 25% | 12.0 | 0 | 0.0 |  |
| 2024 | J.K. Dobbins | LAC | 58% | 13.0 | 0 | 14.4 | HIT |
| 2024 | Jerome Ford | CLE | 75% | 18.0 | 0 | 8.9 |  |
| 2024 | Jordan Mason | SF | 81% | 29.0 | 0 | 8.4 |  |
| 2024 | Zach Charbonnet | SEA | 34% | 10.0 | 0 | 11.0 |  |
| 2024 | Alexander Mattison | LV | 60% | 9.0 | 0 | 9.5 |  |
| 2023 | Tyler Allgeier | ATL | 56% | 18.0 | 0 | 7.1 |  |
| 2023 | Roschon Johnson | CHI | 39% | 11.0 | 0 | 6.4 |  |
| 2023 | Kyren Williams | LA | 65% | 15.0 | 0 | 21.6 | HIT |
| 2023 | Joshua Kelley | LAC | 48% | 16.0 | 0 | 3.1 |  |
| 2023 | Kenneth Gainwell | PHI | 62% | 18.0 | 0 | 5.7 |  |
| 2022 | Michael Carter | NYJ | 60% | 17.0 | 0 | 7.6 |  |
| 2022 | Darrell Henderson | LA | 82% | 18.0 | 0 | 6.8 |  |
| 2022 | Zack Moss | BUF | 37% | 12.0 | 0 | 3.3 |  |
| 2022 | Melvin Gordon | DEN | 41% | 14.0 | 0 | 8.9 |  |
| 2022 | Nyheim Hines | IND | 28% | 9.0 | 0 | 4.1 |  |
| 2021 | Jamaal Williams | DET | 35% | 17.0 | 0 | 8.1 |  |
| 2021 | Nyheim Hines | IND | 45% | 15.0 | 0 | 6.6 |  |
| 2021 | Kenyan Drake | LV | 48% | 11.0 | 0 | 8.2 |  |
| 2021 | Kenneth Gainwell | PHI | 35% | 11.0 | 0 | 6.6 |  |
| 2021 | Mark Ingram | HOU | 46% | 26.0 | 0 | 7.2 |  |
| 2020 | Nyheim Hines | IND | 53% | 15.0 | 0 | 11.1 |  |
| 2020 | J.K. Dobbins | BAL | 39% | 7.0 | 0 | 11.0 |  |
| 2020 | Ronald Jones | TB | 47% | 19.0 | 0 | 12.6 |  |
| 2020 | Benny Snell | PIT | 45% | 19.0 | 0 | 4.5 |  |
| 2020 | Malcolm Brown | LA | 60% | 21.0 | 0 | 5.5 |  |
| 2019 | Ronald Jones | TB | 32% | 14.0 | 0 | 10.4 |  |
| 2019 | Royce Freeman | DEN | 47% | 11.0 | 0 | 9.0 |  |
| 2019 | Mike Davis | CHI | 56% | 11.0 | 0 | 0.5 |  |
| 2019 | Giovani Bernard | CIN | 63% | 9.0 | 0 | 4.0 |  |
| 2019 | Chris Thompson | WAS | 64% | 10.0 | 0 | 7.9 |  |
| 2018 | James Conner | PIT | 92% | 36.0 | 0 | 20.5 | HIT |
| 2018 | Adrian Peterson | WAS | 53% | 28.0 | 0 | 11.1 |  |
| 2018 | T.J. Yeldon | JAX | 62% | 17.0 | 0 | 12.1 |  |
| 2018 | Rashaad Penny | SEA | 44% | 11.0 | 0 | 5.2 |  |
| 2018 | Chris Thompson | WAS | 42% | 11.0 | 0 | 7.4 |  |
| 2017 | Tarik Cohen | CHI | 42% | 13.0 | 0 | 8.5 |  |
| 2017 | Javorius Allen | BAL | 50% | 21.0 | 0 | 10.6 |  |
| 2017 | Marlon Mack | IND | 34% | 11.0 | 0 | 7.1 |  |
| 2017 | Alvin Kamara | NO | 50% | 11.0 | 0 | 20.8 | HIT |
| 2017 | Giovani Bernard | CIN | 48% | 8.0 | 0 | 9.5 |  |
| 2016 | DeAngelo Williams | PIT | 82% | 32.0 | 0 | 9.3 |  |
| 2016 | Tevin Coleman | ATL | 49% | 13.0 | 0 | 14.5 | HIT |
| 2016 | Derrick Henry | TEN | 31% | 7.0 | 0 | 7.6 |  |
| 2016 | Chris Thompson | WAS | 67% | 6.0 | 0 | 9.0 |  |
| 2016 | Christine Michael | SEA | 63% | 17.0 | 0 | 9.1 |  |
| 2013 | Knowshon Moreno | DEN | 52% | 12.0 | 0 | 19.1 | HIT |
| 2013 | Bilal Powell | NYJ | 63% | 16.0 | 0 | 8.6 |  |
| 2013 | Ben Tate | HOU | 26% | 11.0 | 0 | 10.5 |  |
| 2013 | Daniel Thomas | MIA | 47% | 9.0 | 0 | 6.4 |  |
| 2013 | Vick Ballard | IND | 71% | 14.0 | 0 | 0.0 |  |

## After week 2: 745 available RBs, 48 hits (6.4%), 15 stars (2.0%)

| signal | AUC (hit) | top-decile hit rate | top-decile star rate | mean ROS ppg top decile |
|---|---|---|---|---|
| ppg | 0.777 | 24% (n=75) | 8% | 9.2 |
| touches_g | 0.773 | 21% (n=80) | 8% | 9.2 |
| targets_g | 0.757 | 22% (n=89) | 8% | 9.2 |
| tshare | 0.757 | 21% (n=75) | 8% | 9.2 |
| car_share | 0.750 | 21% (n=75) | 8% | 9.2 |
| carries_g | 0.740 | 19% (n=79) | 6% | 8.7 |
| pre_rank | 0.730 | 21% (n=76) | 9% | 9.1 |
| last_snap | 0.727 | 19% (n=62) | 16% | 9.4 |
| snap_pct | 0.715 | 21% (n=62) | 15% | 9.3 |
| tds | 0.620 | 13% (n=143) | 4% | 7.6 |
| ypc | 0.603 | 8% (n=59) | 2% | 6.4 |
| draft_round | 0.583 | 10% (n=123) | 3% | 7.6 |
| age | 0.559 | 10% (n=77) | 4% | 6.5 |
| epa | 0.533 | 12% (n=59) | 5% | 7.5 |
| snap_trend | 0.518 | 8% (n=62) | 3% | 7.3 |
| years_exp | 0.475 | 3% (n=79) | 0% | 5.7 |

Simple screens after week 2 (available RBs):

| rule | n | RB2+ (top-24) | RB1 (top-12) | flex (top-36) | ROS ppg |
|---|---|---|---|---|---|
| snap_pct >= 50% | 52 | 21% | 17% | 35% | 9.6 |
| snap_pct >= 60% | 19 | 32% | 26% | 42% | 10.7 |
| car_share >= 50% | 25 | 24% | 12% | 28% | 8.9 |
| touches/g >= 12 | 74 | 23% | 8% | 35% | 9.4 |
| touches/g >= 15 | 31 | 32% | 10% | 42% | 9.8 |
| targets/g >= 3 | 138 | 18% | 6% | 35% | 8.5 |
| RB1 absent & car_share >= 40% | 13 | 15% | 15% | 31% | 10.0 |
| snap >= 50% & draft rd <= 3 | 13 | 31% | 15% | 54% | 9.5 |
| snap >= 50% & age <= 25 | 26 | 27% | 27% | 35% | 11.0 |
| touches >= 12 & targets >= 2 | 46 | 28% | 9% | 48% | 10.4 |
| snap >= 55% & touches >= 12 & age <= 26 | 21 | 29% | 24% | 43% | 11.2 |

Leave-one-season-out logistic on (snap_pct, car_share, touches_g, targets_g, draft_round, age, rb1_absent, ppg): top-5 per season hit rate **18%** (base rate 6%), star rate 11%, mean ROS ppg 9.5.
Coefficients (standardised): snap_pct +0.08, car_share +0.03, touches_g +0.01, targets_g +0.17, draft_round -0.15, age -0.45, rb1_absent -0.13, ppg +0.56

## After week 3: 829 available RBs, 53 hits (6.4%), 16 stars (1.9%)

| signal | AUC (hit) | top-decile hit rate | top-decile star rate | mean ROS ppg top decile |
|---|---|---|---|---|
| ppg | 0.838 | 31% (n=83) | 11% | 10.0 |
| touches_g | 0.811 | 22% (n=85) | 9% | 9.4 |
| car_share | 0.787 | 24% (n=83) | 10% | 9.5 |
| targets_g | 0.781 | 25% (n=85) | 8% | 9.4 |
| tshare | 0.779 | 25% (n=83) | 8% | 9.8 |
| carries_g | 0.775 | 22% (n=83) | 8% | 9.4 |
| snap_pct | 0.756 | 20% (n=65) | 12% | 9.2 |
| pre_rank | 0.751 | 18% (n=89) | 7% | 9.0 |
| last_snap | 0.751 | 21% (n=70) | 11% | 9.1 |
| tds | 0.708 | 15% (n=207) | 6% | 7.5 |
| ypc | 0.643 | 10% (n=69) | 1% | 6.4 |
| snap_trend | 0.588 | 10% (n=72) | 6% | 7.1 |
| draft_round | 0.579 | 9% (n=140) | 2% | 7.4 |
| epa | 0.577 | 19% (n=68) | 4% | 8.0 |
| age | 0.549 | 9% (n=89) | 3% | 6.3 |
| years_exp | 0.486 | 4% (n=82) | 0% | 5.9 |

Simple screens after week 3 (available RBs):

| rule | n | RB2+ (top-24) | RB1 (top-12) | flex (top-36) | ROS ppg |
|---|---|---|---|---|---|
| snap_pct >= 50% | 46 | 22% | 15% | 33% | 9.9 |
| snap_pct >= 60% | 18 | 33% | 22% | 44% | 11.4 |
| car_share >= 50% | 29 | 24% | 14% | 34% | 9.8 |
| touches/g >= 12 | 81 | 23% | 10% | 35% | 9.5 |
| touches/g >= 15 | 35 | 31% | 14% | 49% | 10.1 |
| targets/g >= 3 | 129 | 22% | 7% | 33% | 8.9 |
| RB1 absent & car_share >= 40% | 28 | 14% | 7% | 21% | 8.2 |
| snap >= 50% & draft rd <= 3 | 13 | 31% | 8% | 38% | 9.8 |
| snap >= 50% & age <= 25 | 27 | 26% | 19% | 30% | 10.4 |
| touches >= 12 & targets >= 2 | 56 | 29% | 11% | 43% | 10.0 |
| snap >= 55% & touches >= 12 & age <= 26 | 20 | 35% | 25% | 45% | 11.6 |

Leave-one-season-out logistic on (snap_pct, car_share, touches_g, targets_g, draft_round, age, rb1_absent, ppg): top-5 per season hit rate **27%** (base rate 6%), star rate 16%, mean ROS ppg 10.0.
Coefficients (standardised): snap_pct +0.05, car_share +0.06, touches_g -0.22, targets_g +0.20, draft_round -0.04, age -0.36, rb1_absent -0.10, ppg +0.90

## Timing: act after week 1 or wait?

Available RBs with snap% >= 50% after week 1: 52, hit rate 19%. After week 3: 46, hit rate 22%. The week-3 screen is more precise but the week-1 pool contains players who are gone by week 3.
