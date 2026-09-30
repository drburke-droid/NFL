# Measuring a WR1's matchup without cornerback tracking

```
16,449 WR games, 2016-2025, league scoring. 'gap' = actual points minus the receiver's own year-to-date rate. Every defensive figure uses PRIOR weeks only.
Columns are quintiles of the defence measure; 'spread' = softest minus toughest.

A0. POOLED yield to all receivers (the failing metric):
b     toughest  hard   avg  soft  softest  spread     n
role                                                   
WR1      -0.49 -0.92 -0.34 -0.24    -0.57   -0.09  4073
WR2      -0.94 -0.28 -0.23  0.48    -0.06    0.89  3973
WR3       0.13  0.38  0.48  0.40     0.70    0.56  3745
WR4+     -0.31  0.16  0.13  0.14     0.06    0.38  4658

A1. ROLE-SPECIFIC yield: what this defence allowed to the role he plays:
b     toughest  hard   avg  soft  softest  spread     n
role                                                   
WR1      -0.86 -0.64 -0.70 -0.17    -0.19    0.67  4072
WR2      -0.08 -0.35 -0.34 -0.06    -0.21   -0.13  3972
WR3       0.29  0.73  0.23  0.35     0.49    0.20  3727
WR4+     -0.04  0.12 -0.14  0.06     0.18    0.23  4656

A2. ROLE-SPECIFIC, ORACLE (whole season: the ceiling):
b     toughest  hard   avg  soft  softest  spread     n
role                                                   
WR1      -3.33 -1.74 -0.50  0.76     2.72    6.05  4073
WR2      -2.08 -1.43 -0.01  0.41     2.38    4.46  3973
WR3      -1.60 -0.27  0.43  1.23     2.59    4.19  3745
WR4+     -0.60 -0.38  0.23  0.56     0.44    1.03  4658

B1. SCHEME: man-coverage rate (soft = most man):
b     toughest  hard   avg  soft  softest  spread     n
role                                                   
WR1      -0.87 -0.51 -0.54 -0.26    -0.56    0.31  3257
WR2      -0.52 -0.70  0.27 -0.15     0.06    0.59  3166
WR3       0.04  0.48  0.26  0.78     0.50    0.46  2978
WR4+     -0.01 -0.14 -0.03  0.35    -0.05   -0.04  3723

B2. SCHEME: two-high-safety rate:
b     toughest  hard   avg  soft  softest  spread     n
role                                                   
WR1      -0.80 -0.84 -0.24 -0.46    -0.42    0.38  3257
WR2      -0.63  0.17 -0.34  0.06    -0.30    0.33  3166
WR3       0.50  0.75  0.29  0.54    -0.02   -0.51  2978
WR4+     -0.01  0.19  0.11 -0.22     0.03    0.04  3723

B3. SCHEME: pressure rate:
b     toughest  hard   avg  soft  softest  spread     n
role                                                   
WR1      -0.49 -0.80 -0.33 -0.46    -0.68   -0.19  3257
WR2      -0.01 -0.29  0.02 -0.60    -0.16   -0.14  3166
WR3       0.40  0.27  0.42  0.22     0.75    0.35  2978
WR4+     -0.05 -0.16  0.06  0.06     0.21    0.26  3723

B4. SCHEME: time to throw allowed:
b     toughest  hard   avg  soft  softest  spread     n
role                                                   
WR1      -0.93 -0.49 -0.18 -0.73    -0.42    0.51  3257
WR2      -0.57  0.69 -0.42 -0.36    -0.38    0.19  3166
WR3       0.46  0.64  0.55  0.10     0.32   -0.13  2978
WR4+      0.07  0.05  0.02 -0.08     0.04   -0.03  3723


correlation between a defence's POOLED receiver yield and its WR1-specific yield: r=+0.559 (n=4,073) - if this were near 1 the pooling would not matter
```
