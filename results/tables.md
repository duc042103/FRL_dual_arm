### Table 1 - Training benchmark (mean ± std over seeds)

| Method | Seeds | Collision episodes during training | Env steps to 80% success | Final success | Filter interventions, whole training | Intervention rate (training, final) | Intervention rate (noise-free policy, final) | Collision rate if filter removed (final) |
|---|---|---|---|---|---|---|---|---|
| Soft penalty (no filter) | 5 | 447 ± 262 | 2.99 ± 0.22 M (2/5) | 39.4 ± 48.2% | - | - | - | 0.9 ± 1.7% |
| Hard filter only | 5 | 0 ± 0 | 1.82 ± 0.09 M (2/5) | 40.0 ± 49.0% | 35.9 ± 15.3% | 59.0 ± 6.9% | 68.4 ± 4.3% | 82.7 ± 14.7% |
| Hard filter + reward penalty | 10 | 0 ± 0 | 1.90 ± 0.87 M (7/10) | 69.9 ± 45.8% | 3.8 ± 2.4% | 2.5 ± 2.8% | 0.8 ± 1.4% | 0.5 ± 1.0% |
| FRL (proposed) | 10 | 0 ± 0 | 1.97 ± 0.70 M (4/10) | 39.8 ± 48.8% | 1.1 ± 0.7% | 1.1 ± 1.2% | 0.3 ± 0.3% | 0.0 ± 0.0% |

### Table 2 - Deployment robustness (noise-free policy, 1000 episodes per seed and condition)

| Method | filter_on: success / collision | reduced_rate: success / collision | approx_model: success / collision | noisy_filter_on: success / collision | filter_off: success / collision |
|---|---|---|---|---|---|
| Soft penalty (no filter) | 31.5 ± 39.9% / 0.0 ± 0.0% | 36.5 ± 45.0% / 0.0 ± 0.0% | 39.5 ± 48.4% / 0.5 ± 1.0% | 30.8 ± 38.7% / 0.0 ± 0.0% | 39.5 ± 48.4% / 0.6 ± 1.3% |
| Hard filter only | 40.0 ± 49.0% / 0.0 ± 0.0% | 39.8 ± 48.8% / 1.8 ± 2.2% | 39.3 ± 48.1% / 70.1 ± 18.5% | 39.4 ± 48.3% / 0.0 ± 0.0% | 39.2 ± 48.0% / 81.5 ± 13.5% |
| Hard filter + reward penalty | 69.8 ± 45.7% / 0.0 ± 0.0% | 69.6 ± 45.5% / 0.0 ± 0.0% | 69.0 ± 45.2% / 0.6 ± 1.3% | 69.4 ± 45.4% / 0.0 ± 0.0% | 69.0 ± 45.2% / 0.9 ± 2.0% |
| FRL (proposed) | 39.8 ± 48.8% / 0.0 ± 0.0% | 39.8 ± 48.8% / 0.0 ± 0.0% | 39.8 ± 48.8% / 0.0 ± 0.0% | 38.6 ± 47.3% / 0.0 ± 0.0% | 39.8 ± 48.8% / 0.0 ± 0.0% |

### Table 2b - Filter dependence among the seeds that solve the task (success >= 80% in the training configuration)

| Method | Solved seeds | Runtime intervention, nominal filter | Filter removed: success / collision episodes | Approximate filter: success / collision episodes | Filter every 3rd cycle: collision episodes |
|---|---|---|---|---|---|
| Soft penalty (no filter) | 2/5 | 32.8 ± 4.9% | 98.7 ± 1.3% / 0 of 2000 | 98.7 ± 1.3% / 0 of 2000 | 0 of 2000 |
| Hard filter only | 2/5 | 64.6 ± 3.0% | 98.0 ± 0.5% / 1753 of 2000 | 98.2 ± 0.7% / 1520 of 2000 | 38 of 2000 |
| Hard filter + reward penalty | 7/10 | 0.6 ± 0.8% | 98.6 ± 2.5% / 86 of 7000 | 98.6 ± 2.5% / 59 of 7000 | 0 of 7000 |
| FRL (proposed) | 4/10 | 0.1 ± 0.1% | 99.6 ± 0.4% / 1 of 4000 | 99.6 ± 0.4% / 0 of 4000 | 0 of 4000 |

### Table 3 - Eq. (5) target form in the dual-arm task

| Target | Seeds | Filter interventions, whole training | Final success | Runtime intervention of the frozen policy | Collision rate, filter removed | Wall-clock [s / 1M steps] |
|---|---|---|---|---|---|---|
| FRL (proposed) | 10 | 1.1 ± 0.7% | 39.8 ± 48.8% | 0.2 ± 0.3% | 0.0 ± 0.0% | 259 ± 14 |
| FRL, direct target a_feas | 3 | 6.2 ± 2.4% | 99.7 ± 0.4% | 0.1 ± 0.1% | 0.0 ± 0.0% | 261 ± 11 |
| FRL, projected-mean target | 3 | 2.1 ± 1.1% | 55.7 ± 41.5% | 0.6 ± 0.6% | 0.0 ± 0.0% | 315 ± 3 |

### Statistical tests - FRL vs. each baseline

| Baseline | Seeds solving the task (FRL / baseline) | Fisher exact p | Permutation p, interventions over training | Permutation p, collisions with filter removed |
|---|---|---|---|---|
| Soft penalty (no filter) | 4/10 / 2/5 | 1.000 | - | - |
| Hard filter only | 4/10 / 2/5 | 1.000 | 0.0002 | 0.0002 |
| Hard filter + reward penalty | 4/10 / 7/10 | 0.370 | 0.0044 | 0.0339 |
| FRL w/o reward penalty | 4/10 / 5/5 | 0.044 | 0.0055 | 1.0000 |

### Table 4 - 2x2 factorial: what each ingredient contributes

| Method | Reward penalty on c_t | Feasibility loss (Eq. 5) | Final success | Env steps to 80% success | Filter interventions, whole training | Collision rate, filter removed |
|---|---|---|---|---|---|---|
| Hard filter only | no | no | 40.0 ± 49.0% | 1.82 ± 0.09 M (2/5) | 35.9 ± 15.3% | 81.5 ± 13.5% |
| Hard filter + reward penalty | yes | no | 69.9 ± 45.8% | 1.90 ± 0.87 M (7/10) | 3.8 ± 2.4% | 0.9 ± 2.0% |
| FRL w/o reward penalty | no | yes | 99.5 ± 0.6% | 1.75 ± 0.88 M (5/5) | 2.3 ± 0.5% | 0.0 ± 0.0% |
| FRL (proposed) | yes | yes | 39.8 ± 48.8% | 1.97 ± 0.70 M (4/10) | 1.1 ± 0.7% | 0.0 ± 0.0% |
