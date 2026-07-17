# Bench-darts composite — walk-forward backtest 2016-2025

Top-10 per season, ex-ante signals only (see script docstring). RELIABLE = 6+
weekly-starter weeks (RB/WR top-24 of week, TE top-12). STAR = season top-12 PPG.

```
 season  pool  top_rel  top_star  pool_rel_rate  pool_star_rate
   2016   149        3         0           17.0             8.7
   2017   147        3         2           15.0             7.5
   2018   141        3         1           18.0             7.8
   2019   151        3         1           13.0             3.3
   2020   132        4         1           11.0             5.3
   2021   145        2         3           15.0             6.2
   2022   119        0         0           10.0             5.0
   2023   126        2         0           17.0             5.6
   2024    92        1         1           16.0             5.4
   2025   126        2         1           17.0             6.3
```

**Top-10 picks: 23/100 reliable (23%), 10/100 star (10%).**
Cheap-pool baseline: 15% reliable, 6.1% star.
**Lift: 1.5x reliable, 1.6x star.**

## Every pick
```
  2016 WR Willie Snead           [ALPHA+FLASH] -> ✔ reliable
  2016 WR Michael Crabtree       [ALPHA+FLASH] -> ✔ reliable
  2016 WR Emmanuel Sanders       [ALPHA+FLASH] -> ✗ miss
  2016 RB Giovani Bernard        [HEIR+FLASH] -> ✗ miss
  2016 RB Jerick McKinnon        [HEIR+FLASH] -> ✔ reliable
  2016 RB Marshawn Lynch         [ALPHA+FLASH] -> ✗ miss
  2016 RB Justin Forsett         [ALPHA+H2+FLASH] -> ✗ miss
  2016 RB Arian Foster           [ALPHA+FLASH] -> ✗ miss
  2016 RB Chris Thompson         [VAC+FLASH] -> ✗ miss
  2016 RB Jeremy Langford        [VAC+FLASH] -> ✗ miss
  2017 RB Jeremy Hill            [ALPHA+FLASH] -> ✗ miss
  2017 WR Stefon Diggs           [ALPHA+FLASH] -> ⭐STAR
  2017 RB DeAndre Washington     [HEIR+VAC+FLASH] -> ✗ miss
  2017 RB Jalen Richard          [HEIR+VAC+FLASH] -> ✗ miss
  2017 TE Jordan Matthews        [ALPHA+FLASH] -> ✗ miss
  2017 TE Zach Ertz              [ALPHA+FLASH] -> ⭐STAR
  2017 RB Paul Perkins           [HEIR+VAC] -> ✗ miss
  2017 RB LeGarrette Blount      [ALPHA+FLASH] -> ✗ miss
  2017 RB Jonathan Stewart       [ALPHA+FLASH] -> ✗ miss
  2017 RB Matt Forte             [ALPHA+FLASH] -> ✔ reliable
  2018 RB Peyton Barber          [HEIR+VAC+FLASH] -> ✗ miss
  2018 RB Le'Veon Bell           [ALPHA+FLASH] -> ✗ miss
  2018 RB Marshawn Lynch         [ALPHA+FLASH] -> ✗ miss
  2018 RB Corey Clement          [HEIR+VAC+FLASH] -> ✗ miss
  2018 TE Hunter Henry           [ALPHA+FLASH] -> ✗ miss
  2018 RB Dion Lewis             [ALPHA+FLASH] -> ✗ miss
  2018 TE Evan Engram            [ALPHA+FLASH] -> ⭐STAR
  2018 WR Alshon Jeffery         [ALPHA+FLASH] -> ✔ reliable
  2018 RB Duke Johnson           [HEIR+FLASH] -> ✗ miss
  2018 RB Marlon Mack            [VAC+FLASH] -> ✔ reliable
  2019 RB Kareem Hunt            [ALPHA+FLASH] -> ✗ miss
  2019 RB Jordan Howard          [ALPHA+FLASH] -> ✗ miss
  2019 WR Jarvis Landry          [ALPHA+FLASH] -> ✔ reliable
  2019 TE Eric Ebron             [ALPHA+FLASH] -> ✗ miss
  2019 WR Marvin Jones           [ALPHA+FLASH] -> ✗ miss
  2019 WR Alshon Jeffery         [ALPHA+FLASH] -> ✗ miss
  2019 RB Tarik Cohen            [VAC+FLASH] -> ✔ reliable
  2019 RB Austin Ekeler          [HEIR+H2+FLASH] -> ⭐STAR
  2019 RB Marshawn Lynch         [ALPHA+FLASH] -> ✗ miss
  2019 RB Jalen Richard          [VAC+FLASH] -> ✗ miss
  2020 RB Marlon Mack            [ALPHA+FLASH] -> ✗ miss
  2020 TE Hunter Henry           [ALPHA+FLASH] -> ⭐STAR
  2020 TE Austin Hooper          [ALPHA+FLASH] -> ✔ reliable
  2020 TE Evan Engram            [ALPHA+FLASH] -> ✔ reliable
  2020 WR Jarvis Landry          [ALPHA+FLASH] -> ✗ miss
  2020 RB Mike Boone             [HEIR+FLASH] -> ✗ miss
  2020 RB Rashaad Penny          [HEIR+FLASH] -> ✗ miss
  2020 RB Chase Edmonds          [HEIR+FLASH] -> ✔ reliable
  2020 RB Tony Pollard           [HEIR+FLASH] -> ✗ miss
  2020 RB Derrius Guice          [HEIR+FLASH] -> ✗ miss
  2021 RB J.K. Dobbins           [HEIR+ALPHA+FLASH] -> ✗ miss
  2021 RB Ty Johnson             [HEIR+VAC+FLASH] -> ✗ miss
  2021 WR JuJu Smith-Schuster    [ALPHA+FLASH] -> ✗ miss
  2021 RB Kenyan Drake           [ALPHA+H2+FLASH] -> ✗ miss
  2021 RB Gus Edwards            [ALPHA+H2+FLASH] -> ✗ miss
  2021 WR Tee Higgins            [ALPHA+FLASH] -> ⭐STAR
  2021 TE Austin Hooper          [ALPHA+FLASH] -> ✗ miss
  2021 TE Hunter Henry           [ALPHA+FLASH] -> ⭐STAR
  2021 TE Rob Gronkowski         [ALPHA+FLASH] -> ⭐STAR
  2021 WR Brandin Cooks          [ALPHA+FLASH] -> ✔ reliable
  2022 RB Darrell Henderson      [HEIR+VAC+FLASH] -> ✗ miss
  2022 RB Dontrell Hilliard      [HEIR+VAC+FLASH] -> ✗ miss
  2022 RB Chris Carson           [ALPHA+FLASH] -> ✗ miss
  2022 WR DeVante Parker         [ALPHA+FLASH] -> ✗ miss
  2022 WR Calvin Ridley          [ALPHA+FLASH] -> ✗ miss
  2022 RB Myles Gaskin           [VAC+FLASH] -> ✗ miss
  2022 TE Hunter Henry           [ALPHA+FLASH] -> ✗ miss
  2022 RB Damien Williams        [HEIR+FLASH] -> ✗ miss
  2022 RB Justin Jackson         [HEIR+FLASH] -> ✗ miss
  2022 RB Khalil Herbert         [HEIR+FLASH] -> ✗ miss
  2023 RB AJ Dillon              [ALPHA+H2+FLASH] -> ✗ miss
  2023 RB Damien Harris          [ALPHA+FLASH] -> ✗ miss
  2023 RB Tyler Allgeier         [ALPHA+FLASH] -> ✗ miss
  2023 WR Michael Pittman        [ALPHA+FLASH] -> ✔ reliable
  2023 TE Pat Freiermuth         [ALPHA+FLASH] -> ✗ miss
  2023 RB Ezekiel Elliott        [ALPHA+FLASH] -> ✔ reliable
  2023 RB Eno Benjamin           [HEIR+FLASH] -> ✗ miss
  2023 RB James Robinson         [HEIR+FLASH] -> ✗ miss
  2023 RB Bam Knight             [HEIR+FLASH] -> ✗ miss
  2023 RB Rashaad Penny          [HEIR+H2+FLASH] -> ✗ miss
  2024 TE Jake Ferguson          [ALPHA+FLASH] -> ✗ miss
  2024 TE T.J. Hockenson         [ALPHA+FLASH] -> ✗ miss
  2024 TE Dallas Goedert         [ALPHA+FLASH] -> ⭐STAR
  2024 RB Rico Dowdle            [HEIR+FLASH] -> ✔ reliable
  2024 RB Jaylen Warren          [HEIR+FLASH] -> ✗ miss
  2024 RB Ty Chandler            [HEIR+FLASH] -> ✗ miss
  2024 RB Keaton Mitchell        [HEIR+FLASH] -> ✗ miss
  2024 RB Jaleel McLaughlin      [HEIR+FLASH] -> ✗ miss
  2024 RB Justice Hill           [HEIR+FLASH] -> ✗ miss
  2024 RB Tyjae Spears           [HEIR+FLASH] -> ✗ miss
  2025 RB J.K. Dobbins           [ALPHA+FLASH] -> ✔ reliable
  2025 WR Chris Godwin Jr.       [ALPHA+FLASH] -> ✗ miss
  2025 WR Jakobi Meyers          [ALPHA+FLASH] -> ✗ miss
  2025 RB Sean Tucker            [HEIR+FLASH] -> ✗ miss
  2025 RB Travis Etienne         [VAC+FLASH] -> ⭐STAR
  2025 RB Emanuel Wilson         [HEIR+FLASH] -> ✗ miss
  2025 RB Austin Ekeler          [HEIR+VAC+FLASH] -> ✗ miss
  2025 WR Keenan Allen           [ALPHA+FLASH] -> ✗ miss
  2025 WR Cooper Kupp            [ALPHA+FLASH] -> ✗ miss
  2025 RB Nick Chubb             [ALPHA] -> ✗ miss
```