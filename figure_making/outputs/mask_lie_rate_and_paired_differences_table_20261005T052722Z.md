| Model | Eligible items | Lie rate (95% Wilson) |
|---|---|---|
| Untrained Qwen $B$ | 747/904 | 53.5% (50.0 to 57.1) |
| $M$ (200 items) | 126/200 | 63.5% (54.8 to 71.4) |
| Student 1 $B$ on $C_B$ | 731/904 | 51.0% (47.4 to 54.6) |
| Student 2 $B$ on $C_M$ | 683/904 | 64.1% (60.5 to 67.6) |
| Untrained Gemma | 719/904 | 50.1% (46.4 to 53.7) |
| Student 3 Gemma on $C_B$ | 654/904 | 34.1% (30.6 to 37.8) |
| Student 4 Gemma on $C_M$ | 634/904 | 34.4% (30.8 to 38.2) |

| Pair | Shared eligible items | Difference | 95% interval | p (two-sided) |
|---|---|---|---|---|
| Student 1 minus untrained Qwen | 681 | -1.2 pts | -4.6 to 2.3 | 0.56 |
| Student 2 minus untrained Qwen | 645 | +11.3 pts | 7.6 to 15.0 | 6.2e-09 |
| Student 2 minus student 1 | 641 | +13.3 pts | 9.7 to 16.8 | 9.4e-13 |
| Student 3 minus untrained Gemma | 605 | -16.0 pts | -20.2 to -11.8 | 3.5e-13 |
| Student 4 minus untrained Gemma | 577 | -14.6 pts | -19.3 to -9.8 | 4.5e-09 |
| Student 4 minus student 3 | 566 | +0.7 pts | -3.4 to 4.8 | 0.8 |
