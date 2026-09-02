# QB buzz/GDELT -> WR performance

WR player-seasons 2016-25 joined to their team's QB1 August signals: n=2646. Outcome overperf = act_pts - FFA projection. Cuts tested: 4 QB-signal quartile tables x 3 WR pools + 2 interactions (state multiplicity when reading p-values).


## QB1 Aug news TONE (q0=most negative)
```
--- ALL WRs (n=602, corr -0.067) ---
     n    over   beat    hit
q                           
0  152  28.854  0.664  0.461
1  149  27.107  0.698  0.443
2  151  18.048  0.570  0.444
3  150  20.984  0.633  0.460
```
```
--- priced WRs (aav>=$10) (n=232, corr -0.026) ---
    n    over   beat    hit
q                          
0  59  43.258  0.797  0.746
1  59  23.345  0.627  0.661
2  57  40.800  0.719  0.772
3  57  31.465  0.737  0.754
```
```
--- cheap WRs (dart pool) (n=129, corr -0.077) ---
    n    over   beat    hit
q                          
0  34  21.625  0.559  0.265
1  31  31.746  0.774  0.226
2  32   7.634  0.562  0.219
3  32  26.872  0.656  0.250
```

## QB1 Aug news NEG-dimension (q3=most negative)
```
--- ALL WRs (n=602, corr +0.058) ---
     n    over   beat    hit
q                           
0  152  30.096  0.684  0.520
1  149  13.061  0.564  0.383
2  150  20.511  0.620  0.420
3  151  31.127  0.695  0.483
```
```
--- priced WRs (aav>=$10) (n=232, corr +0.015) ---
    n    over   beat    hit
q                          
0  59  35.343  0.746  0.780
1  57  24.475  0.667  0.632
2  59  41.438  0.712  0.763
3  57  37.255  0.754  0.754
```
```
--- cheap WRs (dart pool) (n=129, corr +0.127) ---
    n    over   beat    hit
q                          
0  35  26.680  0.686  0.286
1  30  10.564  0.600  0.200
2  33  22.559  0.667  0.212
3  31  26.723  0.581  0.258
```

## QB1 Aug article VOLUME
```
--- ALL WRs (n=678, corr -0.026) ---
     n    over   beat    hit
q                           
0  171  28.495  0.684  0.468
1  169  17.975  0.604  0.432
2  170  25.925  0.612  0.471
3  168  21.232  0.649  0.446
```
```
--- priced WRs (aav>=$10) (n=263, corr +0.048) ---
    n    over   beat    hit
q                          
0  66  32.384  0.712  0.758
1  66  26.734  0.652  0.636
2  65  35.174  0.723  0.738
3  66  37.446  0.758  0.803
```
```
--- cheap WRs (dart pool) (n=137, corr -0.000) ---
    n    over   beat    hit
q                          
0  38  32.858  0.711  0.316
1  31  18.015  0.613  0.226
2  37  24.728  0.622  0.243
3  31  14.626  0.581  0.194
```

## QB1 Aug wiki SPIKE pctl
```
--- ALL WRs (n=632, corr +0.017) ---
     n    over   beat    hit
q                           
0  162  17.877  0.611  0.463
1  155  22.876  0.639  0.465
2  157  27.145  0.650  0.459
3  158  23.310  0.646  0.430
```
```
--- priced WRs (aav>=$10) (n=243, corr -0.001) ---
    n    over   beat    hit
q                          
0  62  26.958  0.677  0.694
1  60  31.176  0.717  0.733
2  61  40.417  0.738  0.754
3  60  26.153  0.667  0.733
```
```
--- cheap WRs (dart pool) (n=130, corr +0.098) ---
    n    over   beat    hit
q                          
0  33  14.898  0.576  0.303
1  34  20.652  0.618  0.235
2  31  30.033  0.677  0.290
3  32  24.925  0.688  0.188
```

## QB1 storm x WR own buzz
```
ALL WRs:
                      n    over   beat    hit
qb_storm own_buzz                            
False    False     1109  21.627  0.617  0.166
         True       552  24.874  0.690  0.076
True     False      322  23.998  0.640  0.168
         True       154  43.747  0.727  0.136
interaction +16.5
```
```
cheap WRs:
                     n    over   beat    hit
qb_storm own_buzz                           
False    False     814  22.115  0.632  0.036
         True      470  19.557  0.667  0.019
True     False     227  22.100  0.632  0.018
         True      131  43.230  0.700  0.061
interaction +23.7
```

## QB context flags
rookie QB1 x ALL WRs: n=53 vs 625, WR overperf delta +1.3 (p=0.893)

rookie QB1 x priced WRs (aav>=$10): n=11 vs 252, WR overperf delta -32.5 (p=0.161)

rookie QB1 x cheap WRs (dart pool): n=17 vs 120, WR overperf delta +7.5 (p=0.608)

QB1 changed teams x ALL WRs: n=85 vs 593, WR overperf delta +1.9 (p=0.790)

QB1 changed teams x priced WRs (aav>=$10): n=33 vs 230, WR overperf delta +16.7 (p=0.192)

QB1 changed teams x cheap WRs (dart pool): n=16 vs 121, WR overperf delta -20.3 (p=0.181)

