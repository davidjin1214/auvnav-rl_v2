# AUV RL Experiment Report

Date: 2026-04-14

This report consolidates two experiment rounds under the same fixed benchmark protocol:

- `experiments/objective_ablation_v1`
- `experiments/efficiency_gain_sweep_v1`

The goal is to answer two questions:

1. Should the project optimize `arrival_v1` or `efficiency_v1`?
2. If `efficiency_v1` is too aggressive, what gain setting should replace it as the main efficiency objective?

## 1. Shared Experimental Setup

### 1.1 Benchmark

- Benchmark key: `single_u15_upstream_tgt15`
- Flow field: `wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy`
- Task geometry: `upstream`
- Target AUV max speed: `1.5 m/s`
- Evaluation manifest: `benchmarks/single_u15_upstream_tgt15.json`

### 1.2 Method

- Method: `sac_stack4`
- Definition: SAC with 4-step observation history
- Seeds: `42, 43, 44`

### 1.3 Training Budget

- Total environment steps: `200000`
- Random steps: `5000`
- Update after: `5000`
- Periodic evaluation every: `10000` steps
- Periodic evaluation episodes: `30`
- Checkpoint interval: `10000` steps
- Device: `cpu`

### 1.4 Interpretation Note

Unless otherwise stated, the tables below report the final checkpoint result stored in each run's `final_eval.json`.

This report also includes "best periodic eval" data from `eval_log.csv`, because several runs reached their highest success rate before the final checkpoint.

## 2. Round 1: Objective Ablation

Source directory: `experiments/objective_ablation_v1`

### 2.1 Goal

Compare:

- `arrival_v1`: time/progress/task-completion objective
- `efficiency_v1`: `arrival_v1` + `energy_cost_gain=5e-4` + `safety_cost_gain=2.0`

### 2.2 Aggregate Results

| Objective | Success Mean +/- Std | Time Mean +/- Std (s) | Energy Mean +/- Std | Safety Mean +/- Std | Progress Mean +/- Std | Path Efficiency Mean +/- Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arrival_v1` | `0.0556 +/- 0.0786` | `97.94 +/- 64.53` | `72948.45 +/- 61040.91` | `16.04 +/- 3.13` | `-0.3743 +/- 0.4556` | `-0.4298 +/- 0.4387` |
| `efficiency_v1` | `0.0111 +/- 0.0157` | `150.75 +/- 69.78` | `92867.37 +/- 63784.73` | `22.62 +/- 6.88` | `-0.3498 +/- 0.4922` | `-0.2904 +/- 0.4380` |

### 2.3 Per-Seed Final Results

| Objective | Seed | Success | Time (s) | Energy | Safety | Progress Ratio | Path Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arrival_v1` | `42` | `0.0000` | `52.39` | `28380.73` | `14.52` | `-0.6906` | `-0.7372` |
| `arrival_v1` | `43` | `0.0000` | `52.24` | `31206.71` | `14.24` | `-0.7023` | `-0.7428` |
| `arrival_v1` | `44` | `0.1667` | `189.19` | `159257.92` | `19.36` | `0.2699` | `0.1905` |
| `efficiency_v1` | `42` | `0.0333` | `180.73` | `123063.67` | `26.37` | `-0.0611` | `-0.0225` |
| `efficiency_v1` | `43` | `0.0000` | `54.34` | `4156.22` | `12.96` | `-1.0426` | `-0.9080` |
| `efficiency_v1` | `44` | `0.0000` | `217.18` | `151382.23` | `28.53` | `0.0545` | `0.0593` |

### 2.4 Termination Summary

| Objective | Goals | Out Of Bounds | Timeout | Mean Episode Safety | Mean Episode Energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `arrival_v1` | `5` | `71` | `14` | `16.04` | `72948.45` |
| `efficiency_v1` | `1` | `52` | `37` | `22.62` | `92867.37` |

### 2.5 Success-Conditioned Metrics

| Objective | Success Episodes | Success Time Mean (s) | Success Energy Mean | Success Path Length Mean (m) | Success Safety Mean | Success Path Efficiency Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arrival_v1` | `5` | `110.02` | `96664.54` | `66.16` | `2.96` | `0.7710` |
| `efficiency_v1` | `1` | `209.50` | `147462.42` | `149.17` | `0.00` | `0.2680` |

### 2.6 Best Periodic Eval by Run

| Objective | Seed | Best Step | Best Eval Success | Eval Return | Eval Time (s) | Eval Energy | Eval Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arrival_v1` | `42` | `30000` | `0.0000` | `-141.73` | `50.08` | `36391.05` | `12.33` |
| `arrival_v1` | `43` | `90000` | `0.0000` | `-141.06` | `48.90` | `34742.88` | `9.63` |
| `arrival_v1` | `44` | `180000` | `0.0667` | `-402.00` | `207.38` | `169259.75` | `57.38` |
| `efficiency_v1` | `42` | `150000` | `0.0000` | `-203.05` | `53.62` | `33531.21` | `14.07` |
| `efficiency_v1` | `43` | `30000` | `0.0000` | `-201.23` | `47.49` | `25358.54` | `10.76` |
| `efficiency_v1` | `44` | `60000` | `0.0000` | `-215.60` | `49.94` | `11304.99` | `12.26` |

### 2.7 Analysis

The conclusion from Round 1 is negative for `efficiency_v1`.

- It did not improve safety in aggregate; mean safety cost increased from `16.04` to `22.62`.
- It did not improve energy in aggregate; mean energy increased from `72.95k` to `92.87k`.
- It lowered final success from `5.56%` to `1.11%`.
- Its only successful final run was a single success episode in `seed_42`, and even that success was slow and expensive.
- Periodic evaluation is even less favorable to `efficiency_v1`: none of its periodic eval points achieved non-zero success.

Interpretation:

- In this critical matched-speed upstream benchmark, `efficiency_v1` is too strong.
- It suppresses the exploratory and corrective maneuvers needed to first learn goal-reaching behavior.
- The result is not "safer arrival"; it is mostly "less decisive failure," often shifting failures from direct boundary exits toward delayed failure.

Round 1 therefore does not support using `efficiency_v1` as the main objective.

## 3. Round 2: Efficiency Gain Sweep

Source directory: `experiments/efficiency_gain_sweep_v1`

### 3.1 Goal

Keep the benchmark and method fixed, keep the objective family fixed as `efficiency_v1`, and only vary:

- `energy_cost_gain`
- `safety_cost_gain`

Scanned gain pairs:

- `e0_s0` -> `(0.0, 0.0)`
- `e0_s0p25` -> `(0.0, 0.25)`
- `e0_s0p5` -> `(0.0, 0.5)`
- `e0_s1` -> `(0.0, 1.0)`
- `e0p0001_s0p5` -> `(1e-4, 0.5)`
- `e0p0002_s0p5` -> `(2e-4, 0.5)`
- `e0p0005_s2` -> `(5e-4, 2.0)` which is the original `efficiency_v1`

### 3.2 Aggregate Results

| Gain | `(Energy, Safety)` | Success Mean +/- Std | Time Mean +/- Std (s) | Energy Mean +/- Std | Safety Mean +/- Std | Progress Mean +/- Std | Path Efficiency Mean +/- Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e0_s0` | `(0.0, 0.0)` | `0.0000 +/- 0.0000` | `63.37 +/- 19.40` | `44365.94 +/- 12740.50` | `20.33 +/- 6.78` | `-0.2418 +/- 0.1643` | `-0.2487 +/- 0.2489` |
| `e0_s0p25` | `(0.0, 0.25)` | `0.0667 +/- 0.0943` | `92.31 +/- 36.15` | `53077.17 +/- 51594.68` | `18.73 +/- 10.72` | `-0.4811 +/- 0.7688` | `-0.3574 +/- 0.5863` |
| `e0_s0p5` | `(0.0, 0.5)` | `0.0222 +/- 0.0314` | `79.07 +/- 37.76` | `55070.20 +/- 41205.76` | `22.83 +/- 16.09` | `-0.4061 +/- 0.5027` | `-0.3723 +/- 0.4443` |
| `e0_s1` | `(0.0, 1.0)` | `0.0000 +/- 0.0000` | `91.18 +/- 24.44` | `57083.94 +/- 19983.77` | `17.01 +/- 2.56` | `-0.5918 +/- 0.1765` | `-0.5003 +/- 0.1547` |
| `e0p0001_s0p5` | `(1e-4, 0.5)` | `0.0000 +/- 0.0000` | `62.36 +/- 10.18` | `30132.70 +/- 21238.92` | `15.09 +/- 2.22` | `-0.8068 +/- 0.4026` | `-0.6832 +/- 0.3070` |
| `e0p0002_s0p5` | `(2e-4, 0.5)` | `0.0000 +/- 0.0000` | `70.66 +/- 8.26` | `32526.30 +/- 10044.38` | `13.39 +/- 2.01` | `-1.0473 +/- 0.0254` | `-0.8082 +/- 0.0607` |
| `e0p0005_s2` | `(5e-4, 2.0)` | `0.0000 +/- 0.0000` | `64.22 +/- 13.59` | `27396.89 +/- 7053.16` | `10.16 +/- 1.00` | `-0.9454 +/- 0.2096` | `-0.7855 +/- 0.0844` |

### 3.3 Per-Seed Final Results

| Gain | Seed | Success | Time (s) | Energy | Safety | Progress Ratio | Path Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e0_s0` | `42` | `0.0000` | `45.14` | `29334.66` | `13.17` | `-0.4477` | `-0.5686` |
| `e0_s0` | `43` | `0.0000` | `54.74` | `43277.74` | `18.34` | `-0.0457` | `0.0385` |
| `e0_s0` | `44` | `0.0000` | `90.23` | `60485.41` | `29.49` | `-0.2320` | `-0.2159` |
| `e0_s0p25` | `42` | `0.0000` | `71.49` | `18990.90` | `8.87` | `-1.0970` | `-0.7659` |
| `e0_s0p25` | `43` | `0.0000` | `62.29` | `14248.93` | `13.88` | `-0.9490` | `-0.7781` |
| `e0_s0p25` | `44` | `0.2000` | `143.16` | `125991.69` | `33.44` | `0.6027` | `0.4718` |
| `e0_s0p5` | `42` | `0.0000` | `49.98` | `35283.69` | `12.46` | `-0.4415` | `-0.4595` |
| `e0_s0p5` | `43` | `0.0000` | `54.84` | `17495.12` | `10.60` | `-1.0034` | `-0.8675` |
| `e0_s0p5` | `44` | `0.0667` | `132.40` | `112431.78` | `45.43` | `0.2265` | `0.2102` |
| `e0_s1` | `42` | `0.0000` | `125.74` | `84533.47` | `17.81` | `-0.5374` | `-0.3789` |
| `e0_s1` | `43` | `0.0000` | `74.68` | `49183.17` | `13.57` | `-0.4080` | `-0.4034` |
| `e0_s1` | `44` | `0.0000` | `73.13` | `37535.18` | `19.64` | `-0.8299` | `-0.7186` |
| `e0p0001_s0p5` | `42` | `0.0000` | `76.18` | `55796.12` | `14.63` | `-0.2375` | `-0.2502` |
| `e0p0001_s0p5` | `43` | `0.0000` | `51.94` | `3785.10` | `12.62` | `-1.0960` | `-0.9268` |
| `e0p0001_s0p5` | `44` | `0.0000` | `58.97` | `30816.88` | `18.01` | `-1.0869` | `-0.8726` |
| `e0p0002_s0p5` | `42` | `0.0000` | `61.97` | `21629.13` | `13.46` | `-1.0497` | `-0.8264` |
| `e0p0002_s0p5` | `43` | `0.0000` | `81.76` | `45866.21` | `10.91` | `-1.0150` | `-0.7265` |
| `e0p0002_s0p5` | `44` | `0.0000` | `68.25` | `30083.55` | `15.81` | `-1.0770` | `-0.8717` |
| `e0p0005_s2` | `42` | `0.0000` | `64.88` | `22337.47` | `11.50` | `-1.1004` | `-0.8822` |
| `e0p0005_s2` | `43` | `0.0000` | `47.25` | `22481.98` | `9.18` | `-0.6491` | `-0.6766` |
| `e0p0005_s2` | `44` | `0.0000` | `80.51` | `37371.22` | `9.81` | `-1.0868` | `-0.7977` |

### 3.4 Termination Summary

| Gain | Goals | Out Of Bounds | Timeout | Depth Hold Failure | Mean Episode Safety | Mean Episode Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e0_s0` | `0` | `90` | `0` | `0` | `20.33` | `44365.94` |
| `e0_s0p25` | `6` | `84` | `0` | `0` | `18.73` | `53077.17` |
| `e0_s0p5` | `2` | `87` | `0` | `1` | `22.83` | `55070.20` |
| `e0_s1` | `0` | `83` | `7` | `0` | `17.01` | `57083.94` |
| `e0p0001_s0p5` | `0` | `89` | `1` | `0` | `15.09` | `30132.70` |
| `e0p0002_s0p5` | `0` | `89` | `1` | `0` | `13.39` | `32526.30` |
| `e0p0005_s2` | `0` | `89` | `1` | `0` | `10.16` | `27396.89` |

### 3.5 Success-Conditioned Metrics for the Only Viable Gains

| Gain | Success Episodes | Success Time Mean (s) | Success Energy Mean | Success Path Length Mean (m) | Success Safety Mean | Success Path Efficiency Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e0_s0p25` | `6` | `119.12` | `105219.94` | `77.99` | `3.64` | `0.8194` |
| `e0_s0p5` | `2` | `149.70` | `117154.80` | `109.59` | `0.00` | `0.4597` |

### 3.6 Best Periodic Eval by Gain

| Gain | Best Run | Best Step | Best Eval Success | Eval Return | Eval Time (s) | Eval Energy | Eval Safety |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e0_s0` | `seed_43` | `120000` | `0.0333` | `-195.54` | `107.25` | `95148.99` | `24.73` |
| `e0_s0p25` | `seed_44` | `180000` | `0.3333` | `-244.58` | `147.36` | `123356.82` | `20.64` |
| `e0_s0p5` | `seed_42` | `180000` | `0.0000` | `-137.16` | `40.42` | `29209.06` | `10.44` |
| `e0_s1` | `seed_43` | `30000` | `0.0000` | `-134.42` | `35.22` | `26305.95` | `9.43` |
| `e0p0001_s0p5` | `seed_42` | `90000` | `0.0333` | `-230.18` | `87.35` | `62333.24` | `24.22` |
| `e0p0002_s0p5` | `seed_43` | `60000` | `0.0333` | `-313.07` | `138.63` | `109599.16` | `17.10` |
| `e0p0005_s2` | `seed_43` | `60000` | `0.0333` | `-212.75` | `67.38` | `52644.94` | `14.63` |

### 3.7 Analysis

The gain sweep supports four clear conclusions.

1. Strong shaping is not the answer.
   The original `efficiency_v1` setting `e0p0005_s2` still ends with zero final success and strongly negative progress.

2. The energy term is the first thing to remove.
   Every nonzero energy-shaping configuration ends with zero final success.
   Even when a few periodic evaluations show `3.33%` success, they do not stabilize into a usable final policy.

3. Weak safety shaping is the only setting that produced a meaningful final improvement.
   `e0_s0p25` is the only gain with a nontrivial final success rate (`6.67%` mean, `6/90` successful episodes overall).
   It is also the only gain whose success-conditioned path efficiency is strongly positive (`0.8194`).

4. `e0_s0p25` is better than `e0_s0p5`, not just different.
   `e0_s0p25` achieved:
   - more successful episodes (`6` vs `2`)
   - shorter successful paths (`77.99 m` vs `109.59 m`)
   - lower successful energy (`105219.94` vs `117154.80`)
   - much higher successful path efficiency (`0.8194` vs `0.4597`)

The most important run is:

- `single_u15_upstream_tgt15 / efficiency_v1 / e0_s0p25 / sac_stack4 / seed_44`

Its final checkpoint reached:

- success rate `0.20`
- success-conditioned time `119.12 s`
- success-conditioned energy `105219.94`
- success-conditioned safety cost `3.64`
- success-conditioned path efficiency `0.8194`

Its best periodic evaluation reached:

- `33.33%` success at `180000` steps

This is the strongest evidence in the current repository that weak safety shaping can improve behavior without collapsing arrival learning.

## 4. Final Recommendation

### 4.1 Recommended Objective Preset

The current recommended replacement for `efficiency_v1` is:

- `efficiency_v2`
- `energy_cost_gain = 0.0`
- `safety_cost_gain = 0.25`

### 4.2 Why `efficiency_v2`

- It is the only configuration that produced a meaningful number of successful final evaluation episodes.
- It preserves some arrival learning signal on the hardest benchmark tested here.
- It does not require the agent to solve arrival and energy minimization simultaneously in the early learning phase.

### 4.3 Protocol Recommendation

The next formal comparison should not rely only on final checkpoints.

Recommended reporting protocol:

1. Report final checkpoint metrics.
2. Report best periodic evaluation checkpoint metrics.
3. Use best-checkpoint evaluation as a secondary view, especially for hard benchmarks with high variance.

### 4.4 Immediate Next Step

Run a confirmation study with:

- `arrival_v1`
- `efficiency_v2`

under the same benchmark and method, but with:

- `5` seeds instead of `3`
- `300k` to `400k` training steps

That is the minimum follow-up needed before treating `efficiency_v2` as a stable main objective in the paper.
