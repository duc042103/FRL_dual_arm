### Table 1 - Training benchmark (mean ± std over seeds)

| Method | Seeds | Collision episodes during training | Env steps to 80% success | Final success | Filter interventions, whole training | Intervention rate (training, final) | Intervention rate (noise-free policy, final) | Collision rate if filter removed (final) |
|---|---|---|---|---|---|---|---|---|
| Soft penalty (no filter) | 10 | 281 ± 285 | 2.99 ± 0.22 M (2/10) | 19.7 ± 39.4% | - | - | - | 0.7 ± 1.3% |
| Hard filter only | 10 | 0 ± 0 | 1.53 ± 0.52 M (7/10) | 70.0 ± 45.8% | 33.1 ± 13.2% | 48.2 ± 16.5% | 56.1 ± 20.7% | 52.0 ± 37.8% |
| Hard filter + reward penalty | 10 | 0 ± 0 | 1.90 ± 0.87 M (7/10) | 69.9 ± 45.8% | 3.8 ± 2.4% | 2.5 ± 2.8% | 0.8 ± 1.4% | 0.5 ± 1.0% |
| FRL, full (L_feas + c_t reward penalty) | 10 | 0 ± 0 | 1.97 ± 0.70 M (4/10) | 39.8 ± 48.8% | 1.1 ± 0.7% | 1.1 ± 1.2% | 0.3 ± 0.3% | 0.0 ± 0.0% |
| FRL, L_feas only (w2 = 0) | 10 | 0 ± 0 | 1.77 ± 0.87 M (7/10) | 75.9 ± 39.5% | 2.1 ± 0.8% | 1.7 ± 1.2% | 0.5 ± 1.1% | 0.5 ± 1.6% |

### Table 2 - Deployment robustness (noise-free policy, 1000 episodes per seed and condition)

| Method | filter_on: success / collision | reduced_rate: success / collision | approx_model: success / collision | noisy_filter_on: success / collision | filter_off: success / collision |
|---|---|---|---|---|---|
| Soft penalty (no filter) | 15.7 ± 32.3% / 0.0 ± 0.0% | 18.3 ± 36.7% / 0.0 ± 0.0% | 19.7 ± 39.5% / 0.5 ± 0.9% | 15.4 ± 31.4% / 0.0 ± 0.0% | 19.7 ± 39.5% / 0.7 ± 1.3% |
| Hard filter only | 70.0 ± 45.8% / 0.0 ± 0.0% | 69.8 ± 45.7% / 0.9 ± 1.8% | 68.8 ± 45.1% / 44.8 ± 33.9% | 68.6 ± 45.0% / 0.0 ± 0.0% | 68.7 ± 45.0% / 51.6 ± 37.1% |
| Hard filter + reward penalty | 69.8 ± 45.7% / 0.0 ± 0.0% | 69.6 ± 45.5% / 0.0 ± 0.0% | 69.0 ± 45.2% / 0.6 ± 1.3% | 69.4 ± 45.4% / 0.0 ± 0.0% | 69.0 ± 45.2% / 0.9 ± 2.0% |
| FRL, full (L_feas + c_t reward penalty) | 39.8 ± 48.8% / 0.0 ± 0.0% | 39.8 ± 48.8% / 0.0 ± 0.0% | 39.8 ± 48.8% / 0.0 ± 0.0% | 38.6 ± 47.3% / 0.0 ± 0.0% | 39.8 ± 48.8% / 0.0 ± 0.0% |
| FRL, L_feas only (w2 = 0) | 76.2 ± 39.6% / 0.0 ± 0.0% | 76.2 ± 39.6% / 0.0 ± 0.0% | 76.2 ± 39.6% / 0.2 ± 0.6% | 76.7 ± 37.3% / 0.0 ± 0.0% | 76.2 ± 39.6% / 0.5 ± 1.4% |

### Table 2b - Filter dependence among the seeds that solve the task (success >= 80% in the training configuration)

| Method | Solved seeds | Runtime intervention, nominal filter | Filter removed: success / collision episodes | Filter removed: steps inside the 4 cm margin | Approximate filter: success / collision episodes | Filter every 3rd cycle: collision episodes |
|---|---|---|---|---|---|---|
| Soft penalty (no filter) | 2/10 | 32.8 ± 4.9% | 98.7 ± 1.3% / 0 of 2000 | 47.0 ± 5.2% | 98.7 ± 1.3% / 0 of 2000 | 0 of 2000 |
| Hard filter only | 7/10 | 50.6 ± 20.5% | 98.2 ± 2.5% / 2845 of 7000 | 50.2 ± 20.3% | 98.2 ± 2.6% / 2496 of 7000 | 38 of 7000 |
| Hard filter + reward penalty | 7/10 | 0.6 ± 0.8% | 98.6 ± 2.5% / 86 of 7000 | 2.5 ± 3.2% | 98.6 ± 2.5% / 59 of 7000 | 0 of 7000 |
| FRL, full (L_feas + c_t reward penalty) | 4/10 | 0.1 ± 0.1% | 99.6 ± 0.4% / 1 of 4000 | 0.3 ± 0.3% | 99.6 ± 0.4% / 0 of 4000 | 0 of 4000 |
| FRL, L_feas only (w2 = 0) | 7/10 | 0.1 ± 0.1% | 99.8 ± 0.2% / 0 of 7000 | 0.2 ± 0.1% | 99.8 ± 0.2% / 0 of 7000 | 0 of 7000 |

### Table 3 - Eq. (5) target form in the dual-arm task

| Target | Seeds | Filter interventions, whole training | Final success | Runtime intervention of the frozen policy | Collision rate, filter removed | Wall-clock [s / 1M steps] |
|---|---|---|---|---|---|---|
| FRL, full (L_feas + c_t reward penalty) | 10 | 1.1 ± 0.7% | 39.8 ± 48.8% | 0.2 ± 0.3% | 0.0 ± 0.0% | 259 ± 14 |
| FRL full, direct target a_feas | 3 | 6.2 ± 2.4% | 99.7 ± 0.4% | 0.1 ± 0.1% | 0.0 ± 0.0% | 261 ± 11 |
| FRL full, projected-mean target | 3 | 2.1 ± 1.1% | 55.7 ± 41.5% | 0.6 ± 0.6% | 0.0 ± 0.0% | 315 ± 3 |

### Statistical tests - FRL variants vs. each baseline (two-sided)

| Comparison | Seeds solving the task (FRL / baseline) | Fisher exact p | Permutation p, interventions over training | Permutation p, collisions with filter removed |
|---|---|---|---|---|
| FRL, L_feas only (w2 = 0) vs. Soft penalty (no filter) | 7/10 / 2/10 | 0.070 | - | - |
| FRL, L_feas only (w2 = 0) vs. Hard filter only | 7/10 / 7/10 | 1.000 | 0.0001 | 0.0008 |
| FRL, L_feas only (w2 = 0) vs. Hard filter + reward penalty | 7/10 / 7/10 | 1.000 | 0.0663 | 0.4975 |
| FRL, full (L_feas + c_t reward penalty) vs. Soft penalty (no filter) | 4/10 / 2/10 | 0.628 | - | - |
| FRL, full (L_feas + c_t reward penalty) vs. Hard filter only | 4/10 / 7/10 | 0.370 | 0.0001 | 0.0008 |
| FRL, full (L_feas + c_t reward penalty) vs. Hard filter + reward penalty | 4/10 / 7/10 | 0.370 | 0.0044 | 0.0339 |

### Table 4 - 2x2 factorial: what each ingredient contributes

| Method | Reward penalty on c_t | Feasibility loss (Eq. 5) | Final success | Env steps to 80% success | Filter interventions, whole training | Collision rate, filter removed |
|---|---|---|---|---|---|---|
| Hard filter only | no | no | 70.0 ± 45.8% | 1.53 ± 0.52 M (7/10) | 33.1 ± 13.2% | 51.6 ± 37.1% |
| Hard filter + reward penalty | yes | no | 69.9 ± 45.8% | 1.90 ± 0.87 M (7/10) | 3.8 ± 2.4% | 0.9 ± 2.0% |
| FRL, L_feas only (w2 = 0) | no | yes | 75.9 ± 39.5% | 1.77 ± 0.87 M (7/10) | 2.1 ± 0.8% | 0.5 ± 1.4% |
| FRL, full (L_feas + c_t reward penalty) | yes | yes | 39.8 ± 48.8% | 1.97 ± 0.70 M (4/10) | 1.1 ± 0.7% | 0.0 ± 0.0% |
