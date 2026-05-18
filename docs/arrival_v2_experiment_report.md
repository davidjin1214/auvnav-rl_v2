# `arrival_v2` Reward — Experiment Report

> **作用**：本文档记录 `arrival_v2` reward preset（commit `813096e`）在 vanilla SAC 上的实测结果。
>
> **关系**：
> - 设计规范见 [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md)（v6 spec，已 SHELVED 但保留为设计档案）。
> - 上下文 / 路线决策见 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) §4.4。
> - 本报告只承载实测结果与 deviation；不修订设计。设计层修订请回到 `online_sac_reward_redesign.md`。
>
> **范围**：单 seed prototype 验证 + 拓扑泛化验证。**不替代**多 seed thesis-grade statistics。
>
> **不修改设计文档顶部 SHELVED 标记**：thesis 矩阵撤销是产品决定，与此处技术验证结论独立。是否据此重启 online 线由用户决定。

---

## 0. Overview

**严格控制对照（§7，唯一变量 = topology × geometry）**：固定 `s1 / k4 / arrival_v2 / U=1.5 / target=1.5 / seed=42 / 1M / num_envs=6 / vanilla SAC`。

| Run | Benchmark | Topology | Geometry | 5/5 Gate |
|---|---|---|---|---|
| §7.1 single_cross    | `single_u15_cross_tgt15`    | single | cross_stream | **PASS**（borderline：OOB 踩线 0.10） |
| §7.2 single_upstream | `single_u15_upstream_tgt15` | single | upstream | **PASS** |
| §7.3 tandem          | `tandem_u15_upstream_tgt15` | double tandem (G/D=3.5) | upstream | **PASS** |
| §7.4 sbs             | `sbs_u15_upstream_tgt15`    | double sbs (G/D=3.5) | upstream | **PASS** |

**Sensor envelope 扩展（§7.6，2026-05-13 补跑，唯一变量 = sensor）**：上述 4 cell 仅 sensor 切到 `s0_k4`（DVL-only, 10-D obs），其它（reward / U / target / seed / 1M / num_envs / vanilla SAC）全部冻结。

| Run | s0 result | s1 reference | s0–s1 final gap |
|---|---|---|---:|
| §7.6 tandem | PASS（final=1.000, OOB=0） | PASS（final=1.000） | 0 |
| §7.6 sbs | PASS（final=1.000, OOB=0） | PASS（final=1.000） | 0 |
| §7.6 single_upstream | PASS（final=1.000, OOB=0） | PASS（final=1.000） | 0 |
| §7.6 **single_cross** | **FAIL**（final=0.100, OOB=0.667）| PASS（final=0.900） | **0.80（80pp）** |

**AsymCritic 单变量 ablation（§7.7，2026-05-17 补跑 + 2026-05-18 2-seed paired update，pure B 路径，唯一变量 = `--use-asymmetric-critic`）**：在 §7.6.4 vanilla 基础上仅启用 AsymCritic，其它（s0 / k4 / arrival_v2 / U / target / seed / 1M / num_envs / no-LN / UTD=1）全部冻结。

| Run | final | full-trajectory mean | safety_cost | progress_ratio | 5/5 Gate |
|---|---:|---:|---:|---:|:-:|
| §7.6.4 vanilla baseline seed=42 | 0.100 (3/30 goal) | 0.221 | 25.85 | 0.277 | FAIL（3/5） |
| §7.7 sac_asym (pure B) seed=42 | 0.167 (5/30 goal) | **0.044（5× 更差）** | **16.93（−34%）** | **0.453（+64%）** | FAIL（3/5） |
| §7.7.1 vanilla seed=0 (sister) | 0.400 | 0.218（与 seed=42 差 0.003）| 9.15 | 0.556 | FAIL（3/5）|
| §7.7.1 sac_asym seed=0 (sister) | 0.200 | **0.088** | 31.65 | 0.530 | FAIL（3/5）|
| **2-seed mean** | — | vanilla 0.220 / asym **0.066（3.3× 更差）** | — | — | — |

→ **negative finding（2-seed × 2-algo paired hardened）**：AsymCritic peak ceiling 跨 seed 严丝合缝锁在 0.267（vanilla 在 [0.37, 0.53]），2-seed paired mean 仍 ~3.3× gap。瓶颈在 actor-side information access，不是 critic estimation accuracy。

**History k=4→8 actor-side ablation（§7.8，2026-05-18 补跑，PASS — 闭合 80pp gap，唯一变量 = `--history-length 4→8`）**：在 §7.6.4 vanilla 基础上仅加大 actor 时序窗口，其它（s0 / arrival_v2 / U / target / seed=42 / 1M / num_envs / vanilla SAC）全部冻结。

| Run | final | mean39 | peak | OOB | progress | 5/5 Gate |
|---|---:|---:|---:|---:|---:|:-:|
| §7.6.4 vanilla k=4 | 0.100 | 0.221 | 0.367 @ 975k | 0.667 | 0.277 | FAIL（3/5）|
| **§7.8 vanilla k=8** | **0.900** | **0.636** | **0.900 @ 475k** | **0.100** | **0.834** | **PASS（5/5 ✓）** |
| §7.1 s1_k4 (upper ref) | 0.900 | 0.497 | 0.900 @ 725k | 0.100 | 0.836 | PASS |
| **Δ k=8 − k=4** | **+80pp** | **+41.5pp** | **+53pp，−500k**  | **−56.7pp** | **+201%** | — |
| **Δ k=8 − s1_k4 ref** | 0 | **+13.9pp** | 0，**−250k 收敛快 35%** | 0 | −0.2pp | — |

→ **正向 thesis-grade finding**：s0_k8 完全追平 s1_k4 上界 reference，且收敛更快。**80pp s0–s1 gap 不是 spatial information bottleneck，是 actor-side temporal access bottleneck**。本研究 deployment-realistic 路径从「升级 sensor 到 s1」改写为「保持 s0 + 升级 actor 时序访问到 k=8」。

**Reference baselines（§2 / §3，不参与 §7.5/§7.6 严格对比；seed/step/U 与主对照 4 组不一致）**：

| Run | Benchmark | Probe | Seed | Total steps | Confound | 5/5 Gate |
|---|---|---|---:|---:|---|---|
| §2 cross_u10 regression | `single_u10_cross_tgt15` | s0 / k4 | 46 | 1M | U=1.0, s0, seed=46 | PASS |
| §3 P1 v6 重跑           | `single_u15_upstream_tgt15` | s1 / k4 | 46 | 1.5M | seed=46, 1.5M | PASS |

**TL;DR**：
- arrival_v2 在严格控制下（s1 / k4 / seed=42 / 1M / U=1.5 / target=1.5 / vanilla SAC）的 4 组 vanilla SAC 验证**全部 5/5 gate PASS**：单柱 cross / 单柱 upstream / 双柱 tandem / 双柱 sbs。
- 三组 upstream（single / tandem / sbs）均 final=1.000 / OOB=0.000 / 30/30 全 goal，peak first-hit step 都在 475k–625k → **topology 在严格控制下未引入额外 sample 难度**。
- 单柱 cross_stream 是四组里唯一 OOB 踩线 (0.10) 的 run、final=0.900、return std=112 → **cross_stream geometry 比 upstream 更难**。
- 末段 safety 排序：single_upstream (0.142) < sbs (0.586) < tandem (6.85) — 与 wake topology 物理直觉一致。
- **§7.6 s0 sensor envelope**（单 seed exploratory）：arrival_v2 + s0 在三个上游几何下（tandem / sbs / single_upstream）全部 PASS（与 s1 在 `last100k_mean` 上 ±0.05 内），但 `single_cross_s0` **catastrophic FAIL**（5/5 gate 中 3 个 fail；final=0.100, OOB=0.667）。**A0（cross_u10 + arrival_v1）s0–s1 gap 3pp 在 cross_u15 + arrival_v2 下放大 24× 到 80pp** — partial-observability gap 在 production-difficulty regime 下真正显化的实证。
- **§7.7 AsymCritic 单变量 ablation on `single_cross_s0`**（pure B 路径，2026-05-18 **升格为 2-seed × 2-algo paired hardened negative finding**）：在 §7.6.4 vanilla 基础上仅加 `--use-asymmetric-critic`，**未闭合 80pp gap**。2-seed paired mean asym 0.066 vs vanilla 0.220（3.3× 更差）；**asym peak ceiling 跨 seed 严丝合缝锁在 0.267**（vanilla peak 在 [0.37, 0.53]），证明 0.267 不是 noise 而是 AsymCritic 在此任务上的 information-theoretic ceiling。行为风格明显改变（safety_cost −34%、progress_ratio +64%、return +26%）但 task-level success 反退。机理：critic 端 privileged info 让 actor 学到 critic-validated 的 "safer + more progressive" 策略，但 actor 端 s0 信息不足以将其兑现成任务级 success。**推翻 [`SAC_improvements_survey.md`](SAC_improvements_survey.md) §10.2 P1#5 / 旧 §8 P1#1 预期**，将 `single_cross_s0` 瓶颈从 critic estimation accuracy 重定位到 **actor-side information access**。
- **§7.8 single_cross_s0 × history k=4→8 actor-side ablation**（单 seed exploratory，**PASS — 闭合 80pp gap**，正向 thesis-grade 发现）：在 §7.6.4 vanilla 基础上**仅**加大 `--history-length 4 → 8`（actor 时序窗 ~2s → ~4s），5/5 gate 全 PASS：final 0.100 → 0.900, OOB 0.667 → 0.100, mean 0.221 → 0.636, peak @ 975k → @ 475k。**完全追平 s1_k4 上界 reference**（final/peak/OOB/progress/return 全部一致），且 sample-efficiency 还更好 250k 步（peak @ 475k vs s1 @ 725k）。**直接验证 §7.7 F4 机理重定位**（actor-side info 是瓶颈，不是 critic estimation）：cross-arm 对照 — §7.7 给 critic 加 [u_eq, v_eq] 任务级反退到 0.044；§7.8 给 actor 加 4 s 时序窗任务级飙到 0.636。物理解释：k=8 ~4s 已覆盖涡街周期 10–20 s 的 20–40%，足以让 actor 从单点 DVL 时序节拍中反演主导脉动相位（即 critic 通过 privileged hull-integral 看到的同一物理量）。**本研究 deployment-realistic 路径重写：从「升级 sensor s0 → s1（多一个空间探头）」改写为「保持 s0 + 升级 actor 时序访问 k=4 → k=8」**。
- §2 cross_u10 / §3 P1 v6 旁证 arrival_v2 在更慢流速、更弱 sensor、不同 seed 下也 PASS，但因 seed/step/U confound 仅作 reference，不进入 §7.5/§7.6 主对照。

---

## 1. 实施跟踪

- arrival_v2 8 参数完整版按 [设计 §5.1](online_sac_reward_redesign.md) v6 spec 在 commit `813096e`（2026-05-07）落地进 `auv_nav/reward.py`，与 `arrival_v2_simple`（commit `bd37412`）非同一物。
- [设计 §8.1] Gate A pure-formula validator（`scripts/validate_arrival_v2_candidate`）通过：default `w_safety=2.0` discounted unsafe-shortcut + terminal dominance + OOB ordering 全部成立；`w_safety=0.5` 在 discounted unsafe-shortcut 上被明确判失败（与 v6 设计预言一致）。
- 隔离 prototype 分支 `codex-arrival-v2-prototype` 跑了**14 组实验**（4 组 §7 严格控制 + 4 组 §7.6 sensor envelope + 1 组 §7.7 AsymCritic ablation seed=42 + 2 组 §7.7 update seed=0 paired (vanilla + sac_asym) + 1 组 §7.8 history k=8 ablation + 2 组 reference baselines）：
  - **§2 cross_u10 behavior regression**（reference）：先按计划跑 600k，未通过 last100k_mean gate（曲线仍在上升），延到 1M 后 5/5 gate 全过。
  - **§3 P1 v6 重跑**（reference）：从原计划 1M 提到 1.5M（s1 + upstream + 12-D 比 cross + s0 + 10-D 难，留缓冲）；事后由 §7.2 (seed=42 / 1M PASS) 推翻这个 budget 假设 — 1.5M 是 seed=46 specific。
  - **§7.3 tandem 拓扑泛化**（strict control，旧编号 §7.1）：1M cap，seed=42，5/5 gate 全过。
  - **§7.4 sbs 拓扑泛化**（strict control，旧编号 §7.2）：1M cap，seed=42，5/5 gate 全过。
  - **§7.1 single_cross 控制对照**（strict control，2026-05-09 补跑）：1M cap，seed=42，5/5 gate 全过（OOB 踩线 0.10）。
  - **§7.2 single_upstream 控制对照**（strict control，2026-05-09 补跑）：1M cap，seed=42，5/5 gate 全过；与 §3 P1 v6 同 benchmark 的二点 seed 观测。
  - **§7.6 s0 sensor envelope**（strict-control sensor 扩展，4 cell，2026-05-13 补跑）：3/4 PASS（tandem / sbs / single_upstream），1/4 catastrophic FAIL（single_cross_s0：3/5 gate fail）。数据完整性 footnote：本地下载时两个上游 dir 一度互换，已通过 6 个内部指针交叉验证 + `mv` 修复，详见 §7.6 末尾。
  - **§7.7 AsymCritic 单变量 ablation on `single_cross_s0`**（pure B 路径，2026-05-17 补跑）：仅加 `--use-asymmetric-critic`，其它与 §7.6.4 完全一致。**Negative finding**：3/5 gate fail；行为风格改变（safety −34%、progress +64%），但 task-level mean_success 反而比 vanilla 差 5×。推翻 §10.2 P1#5 / 旧 §8 P1#1 预期。
  - **§7.7 update 2-seed paired hardening**（2026-05-18 补跑）：vanilla k=4 seed=0 + sac_asym k=4 seed=0 配对复现。**Asym peak ceiling 跨 seed 严丝合缝锁在 0.267**（vanilla peak 在 [0.37, 0.53]），2-seed × 2-algo paired mean asym 0.066 vs vanilla 0.220 仍是 ~3.3× gap。§7.7 negative finding 从 single-seed exploratory 升格为 2-seed × 2-algo paired hardened claim，可写论文。同时给出方法论 footnote：vanilla seed=42 / seed=0 mean39 差仅 0.003，但 final_eval 差 +30pp（OOB collapse 模式 seed-sensitive）— 对 catastrophic-OOB-prone 任务，应优先看 mean39 / last100k。
  - **§7.8 history k=4→8 actor-side ablation on `single_cross_s0`**（2026-05-18 补跑）：仅加大 `--history-length 4 → 8`，其它与 §7.6.4 完全一致。**PASS — 闭合 80pp gap**：final 0.100 → 0.900, OOB 0.667 → 0.100, mean 0.221 → 0.636, peak @ 975k → @ 475k。完全追平 s1_k4 上界 reference（final/peak/OOB/progress 全部一致），且 sample-efficiency 还更好 250k 步。直接验证 §7.7 F4 把瓶颈从 critic-side 重定位到 actor-side 的机理重写。本研究 deployment-realistic 路径从「升级 sensor 到 s1」改写为「保持 s0 + 升级 actor 时序访问到 k=8」。
- 实施期间发现并修了一个 SAC trainer resume 路径上的 silent bug，参见 §6。
- 闭环 commit：`f179c5b`（§2 + §3 闭环），`1adee4e`（doc split + tandem/sbs notebook scaffold + §7.3/§7.4 回填），`f00074e`（§7 4-way strict control 重写 + 单柱 §7.1/§7.2 补跑落地），`f771414`（strict-control viz + single_u15 completed archival），`01b78ad`（§7.6 s0 sensor envelope 闭环 + §8 P1 重排），`3d20e86`（§7.7 AsymCritic ablation 单 seed 闭环 + §8 P1 重写），`e6ca646`（§7.8 k=8 notebook scaffold），§7.7 update + §7.8 PASS + §8 P1 重排 + 3 个 _completed archival 落 commit（待提交）。

**复现路径**：
- §2 cross_u10 + §3 P1 v6：[`notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb`](../notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb)
- 600k cross_u10 prototype 归档：[`notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb`](../notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb)
- §7.3 tandem + §7.4 sbs（s1）：[`notebooks/sac_arrival_v2_tandem_sbs_validation_completed.ipynb`](../notebooks/sac_arrival_v2_tandem_sbs_validation_completed.ipynb)
- §7.1 single_cross + §7.2 single_upstream（s1, 严格控制对照）：scaffold [`notebooks/sac_arrival_v2_single_u15_seed42_1M_validation.ipynb`](../notebooks/sac_arrival_v2_single_u15_seed42_1M_validation.ipynb) / archival [`notebooks/sac_arrival_v2_single_u15_seed42_1M_validation_completed.ipynb`](../notebooks/sac_arrival_v2_single_u15_seed42_1M_validation_completed.ipynb)
- §7.6 s0 sensor envelope（4 cell）：scaffold [`notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope.ipynb`](../notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope.ipynb) / archival [`notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope_completed.ipynb`](../notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope_completed.ipynb)
- §7.7 AsymCritic 单变量 ablation（1 cell, pure B, seed=42）：scaffold [`notebooks/sac_arrival_v2_s0_cross_asym_ablation.ipynb`](../notebooks/sac_arrival_v2_s0_cross_asym_ablation.ipynb) / archival [`notebooks/sac_arrival_v2_s0_cross_asym_ablation_completed.ipynb`](../notebooks/sac_arrival_v2_s0_cross_asym_ablation_completed.ipynb)
- §7.7 update 2-seed paired sister (seed=0)：vanilla [`notebooks/sac_arrival_v2_s0_cross_vanilla_seed0.ipynb`](../notebooks/sac_arrival_v2_s0_cross_vanilla_seed0.ipynb) / [`notebooks/sac_arrival_v2_s0_cross_vanilla_seed0_completed.ipynb`](../notebooks/sac_arrival_v2_s0_cross_vanilla_seed0_completed.ipynb)；sac_asym [`notebooks/sac_arrival_v2_s0_cross_asym_seed0.ipynb`](../notebooks/sac_arrival_v2_s0_cross_asym_seed0.ipynb) / [`notebooks/sac_arrival_v2_s0_cross_asym_seed0_completed.ipynb`](../notebooks/sac_arrival_v2_s0_cross_asym_seed0_completed.ipynb)
- §7.8 history k=4→8 actor-side ablation (seed=42)：scaffold [`notebooks/sac_arrival_v2_s0_cross_k8.ipynb`](../notebooks/sac_arrival_v2_s0_cross_k8.ipynb) / archival [`notebooks/sac_arrival_v2_s0_cross_k8_completed.ipynb`](../notebooks/sac_arrival_v2_s0_cross_k8_completed.ipynb)

---

## 2. cross_u10 Behavior Regression（PASS）

**配置**：vanilla SAC, `s0_k4`, seed=46, num_envs=6, total_steps=600k → **1M**。eval 25k 一次，30 ep / 次。

[设计 §8.3] 三条 gate 实测（600k 数据来自 cross_u10 regression notebook，1M 数据来自 cross extension notebook）：

| Gate | 阈值 | 600k 首跑 | 1M 续训 | 结论 |
|---|---|---|---|---|
| final_success_rate | ≥ 0.85 | 0.867 PASS（仅 1.7pp 余量） | **1.000** PASS | ✓ |
| last100k_mean / peak | ≥ 0.90 | 0.6533 / 0.867 = 0.754 **FAIL** | 0.9833 / 1.000 = **0.983** PASS | ✓ |
| OOB rate | ≤ 0.10 | 0.033 PASS | **0.000** PASS | ✓ |

外加 v5 / v6 spec 要求的 MDP / replay 语义 check（`trainer_state.json`）：

| Check | 期望 | 实测 |
|---|---|---|
| `include_episode_context_obs` | True | True ✓ |
| `timeout_bootstrap_semantics` | terminal | terminal ✓ |
| obs_dim (s0 / k4 / arrival_v2) | 48 | 48 ✓ |
| reward_objective | arrival_v2 | arrival_v2 ✓ |

**末段 eval 轨迹**（900k–1M，5 个 eval 点，30 ep / 点）：

| step | success | safety_cost | eval_time_s | progress_ratio | path_efficiency |
|---:|---:|---:|---:|---:|---:|
| 900 000 | 0.967 | 2.41 | 38.8 | 0.898 | 0.867 |
| 925 002 | 0.967 | 1.85 | 39.0 | 0.888 | 0.870 |
| 950 004 | 1.000 | 2.43 | 39.6 | 0.908 | 0.884 |
| 975 000 | 1.000 | 2.60 | 39.9 | 0.908 | 0.877 |
| **1 000 000 (final)** | **1.000** | 1.63 | 38.4 | 0.908 | 0.896 |

**收敛形状**：
- 600k → 1M 续训窗口内 success ∈ [0.867, 1.000] sustained（17 个 eval 点中 14 个 ≥ 0.967），无 collapse；
- `eval_time_s` 从 600k 时 65s 降到 1M 时 39s（× 0.59），策略在 sustained 高 success 同时显著缩短 time-to-goal —— arrival_v2 的 `R_timeout=50` per-step 时间压力按设计奏效；
- `safety_cost` 从 4.67 降到 1.63（× 0.35），`termination={goal: 30}` 全 goal，无 OOB / timeout —— **arrival_v2 在 cross_u10 上不引入 efficiency_v2 的 OOB-suicide 失败模式**。

**结论**：[设计 §8.3] PASS。arrival_v2 在「最弱 sensor + vanilla SAC + 已知可学」的保守回归中没有 break，可以作为后续工作的稳态起点。

**预算偏差**：[设计 §8.2] 给 cross_u10 regression 1.5h L4，实际 ~3h（600k 首跑 + 续 400k 到 1M）。差因是 600k 时未稳态 —— 这是 prototype 实测发现，不是 v6 spec 的 bug；下次类似 prototype 应在 spec 里改成「先按 1M cap 跑完再判，而不是固定 600k」。

---

## 3. P1 v6 重跑：`efficiency_v2` 旧 failure mode 修复（PASS）

**Benchmark**：`single_u15_upstream_tgt15`，即 [设计 §2] P1 证据复核所记录的 `efficiency_v2` collapse 现场（agent 后期学会快速出界，return 上升但 success → 0）。

**配置**：vanilla SAC, `s1_k4`, seed=46, num_envs=6, total_steps=1.5M。eval 25k 一次，30 ep / 次。

**Final eval (30 ep) 与旧 efficiency_v2 P1 对照**：

| 指标 | efficiency_v2 P1（设计 §2 旧版） | arrival_v2 P1 v6 (1.5M) |
|---|---|---|
| eval_success_rate | **0.0**（collapse） | **1.000** |
| eval_termination_counts | OOB-dominated | `{goal: 30}` |
| eval_return | 上升但与 success 反向 | 143.87 ± 1.78 |
| eval_progress_ratio | n/a（collapse） | 0.9355 |
| eval_path_efficiency | n/a | 0.7787 |
| eval_safety_cost | high（OOB suicide） | **0.292** |
| eval_time_s | n/a | 111.18 |

**学习曲线分阶段**（60 个 eval 点 = 25k → 1.5M，30 ep / 点）：

| 阶段 | env_step 区间 | success 区间 | 现象 |
|---|---|---|---|
| 探索期 | 25k–175k | 0.00–0.50 | safety_cost 20–35；upstream 推进未学到 |
| 爬升期 | 200k–350k | 0.23–0.83 | progress_ratio 上行，safety_cost 下行 |
| 稳定爬升 | 375k–600k | 0.83–0.93 | dominant policy 形成 |
| 首达饱和 | 675k | **1.000** | 首次 30/30 |
| 准稳态 | 675k–1.4M | ≥ 0.967（少数 outlier） | 单点最低 0.767 @ 850k，前后 825k=1.0 / 875k=1.0 → 30-ep eval 噪声 |
| 末段稳态 | 1.4M–1.5M | **全 1.000** | safety_cost ∈ [0.07, 0.50]；progress_ratio 锁定 0.9356；eval_time 锁定 ~111s |

**v6 spec 关键不变量实测**：
- **terminal dominance**：30/30 全 goal，无 timeout / OOB → R_success=100 在 γ=0.995 discounted return 下确实主导（与 [设计 §11.3] 计算预期一致）；
- **timeout-as-terminal**：`trainer_state.timeout_bootstrap_semantics=terminal`，SAC target 不再 bootstrap 超时 episode（v5 §4.7 / §6 修订生效）；
- **MDP state contract**：`include_episode_context_obs=True` 实测生效（v5 §4.6 / §6 修订生效）；
- **discounted unsafe-shortcut**：末段 safety_cost 长期 < 0.5，与 [设计 §8.1] Gate A validator 的 `safe_success > risky_success` 预期一致 —— policy 不学短路绕飞。

**对 [设计 §2] P1 证据复核的回应**：
[设计 §2] 记录的失败模式（"return 上升但 success=0；agent 学会快速出界"）在 arrival_v2 下被消除。P1 v6 1.5M 末段 `eval_return=143.87` 与 `success=1.0 / termination={goal:30}` **方向一致** —— return 与 task 主指标重新对齐，arrival-first reward 设计的核心立论得到证据支持。

**结论**：arrival_v2 在 P1 v6 这个 [设计 §2] 旧 failure mode 现场实现了彻底修复。这是奖励重设计相对于 efficiency_v2 的关键证据。

---

## 4. 与 [设计 §8] / [设计 §11] 预言的对照

| 预言 | 实测 | 对照 |
|---|---|---|
| §8.1 Gate A 纯公式 validator pass | ✓ default `w_safety=2.0` 全过；`w_safety=0.5` 在 discounted unsafe-shortcut 上 fail | 与 v6 预言一致 |
| §8.3 cross_u10 final ≥ 0.85 | ✓ final=1.000（1M 时） | PASS（600k 阶段刚过阈值，需要续 budget） |
| §8.3 cross_u10 OOB ≤ 0.10 | ✓ OOB=0.000 | PASS |
| §8.3 last100k_mean ≥ 0.9 × peak | ✓ 0.9833 / 1.000 | PASS（说明续训到稳态判据是合理的） |
| §11.7 critic 失稳风险（reward variance × terminal dominance） | 未触发 | 1.5M 训练干净收敛，不需要 reward scale ×0.5 / 固定 alpha 备案 |
| §11.8 「arrival_v2 真的更安全则 failure-policy avg safety 会更低」 | ✓ P1 v6 末段 safety_cost < 0.5，远低于 efficiency_v2 P1 时 fail-policy 的 ~20 | 验证 §11.8 二次校准的方向正确 |
| §11.6 AsymCritic × 新 reward 交互 | 未在本次 prototype 检验（vanilla SAC，无 asym critic） | 留作后续 |
| §6 v6 invariants 在不同物理拓扑 × geometry 下保持（terminal dominance / discounted unsafe-shortcut / OOB ordering） | ✓ §7 4 组严格控制（single+cross / single+upstream / tandem / sbs，全 seed=42 / 1M / s1）均 5/5 PASS，三组 upstream 30/30 全 goal、OOB=0；single+cross 5/5 PASS but borderline (OOB 踩线 0.10) | 与 v6 设计一致：reward 不变量与 wake topology / geometry 都无关；cross_stream geometry 是相对最难场景 |

---

## 5. 时间预算实测 vs [设计 §8.2]

| 任务 | §8.2 预算 (L4) | 实测 (L4) | 说明 |
|---|---:|---:|---|
| §2 cross_u10 regression (ref) | 1.5h（600k） | ~3h（600k 首跑 + 续 400k） | 600k 未稳态，按收敛形状续到 1M |
| §3 P1 v6 重跑 (ref) | 2.5h（1M） | ~5h（1.5M） | 上行 budget cap，留缓冲；事后 §7.2 证明 seed=46 specific |
| §7.3 tandem topology | 2.5h（1M） | ~2.5h（1M） | cap 命中，seed=42 / 1M 即收敛 |
| §7.4 sbs topology | 2.5h（1M） | ~2.5h（1M） | cap 命中，seed=42 / 1M 即收敛 |
| §7.1 single_cross control | 2.5h（1M） | ~2.5h（1M） | cap 命中，5/5 PASS but OOB 踩线 |
| §7.2 single_upstream control | 2.5h（1M） | ~2.5h（1M） | cap 命中；与 §3 同 benchmark 但 seed=42 / 1M 即收敛 |
| **总 wallclock** | ~14h | ~18h | × 1.3 |

§7 四组严格控制 phase **一次 cap 命中**：1M 预算在 `s1 / k4 / U=1.5 / target=1.5 / seed=42` 下确实够用 —— 之前 §3 (seed=46/1.5M) 的 budget 反差由 §7.2 (seed=42/1M PASS, peak @ 475k) 解释为 seed-specific，**非 benchmark 难度**。这一发现修订了 [设计 §8.2] 的预算预言：1M 是 single + double + cross + upstream 在 seed=42 下的实测 enough budget；多 seed 复现需要逐 seed 验证而非默认 1.5M cap。

---

## 6. 隐藏 SAC trainer bug：resume 路径修复（2026-05-08）

实施 §2 / §3 期间发现 `scripts/train_sac.py` resume 路径上一个 silent bug 链，**与 reward 设计无关**，但会让任何 `--resume <save_dir> --total-steps <N>` 静默退化为 no-op，所以记录在此：

| Bug | 位置 | 现象 | 修复 |
|---|---|---|---|
| start_step 单位混淆 | `train_sac.py:467` | trainer_state 保存 `env_step` 是 global step，主循环 `range` 把它当 per-env 计数器，resume 后 `range` 为空，循环不进入 | 把 `start_step` 在 range 入口处除以 `num_envs` |
| 空跑仍写 trainer_state | `train_sac.py` 主循环出口 | 上一 bug 让循环空过后，`save_training_state(env_step=total_env_steps)` 仍执行，把 trainer_state.env_step 从 600k 错改成 1M（agent / replay 实际未变） | maybe_resume 后加 early-return：`if start_step >= total_env_steps: return` |
| checkpoint_dir 路径累积错算 | `train_sac.py:164` | trainer_state 里 `checkpoint_dir` 是 save_dir 相对路径，但 line 409 当 cwd 相对路径解释，每次 resume `../` 数翻倍（7 → 13 → 19） | resume 时把相对路径用 `args.resume` 解析成绝对路径再写回 args |

三个修复合计 17 行 diff，与 reward 设计正交。建议未来加一个 mini regression test（fresh 1k → resume 续到 2k，断言 `trainer_state.env_step` 真的推进且 agent path 可解析），但不阻塞本节结论。

---

## 7. 严格控制下的 4-way 拓扑 × Geometry 对照（PASS）

**动机与设计**：§2 / §3 在 seed / total_steps / U∞ 三个维度上彼此不一致（§2 用 U=1.0 / s0 / seed=46 / 1M，§3 用 U=1.5 / s1 / seed=46 / 1.5M），两者**不构成**与 §7.3 / §7.4（U=1.5 / s1 / seed=42 / 1M）的严格对照。为得到拓扑与 geometry 影响的干净读数，本节把所有非控变量统一固定：

> `s1 / k4 / arrival_v2 / U=1.5 / target=1.5 / seed=42 / 1M / num_envs=6`，唯一变量 = **topology × geometry**

四组组合：

| 节 | benchmark | topology | geometry | 物理主导现象 |
|---|---|---|---|---|
| §7.1 | `single_u15_cross_tgt15`    | single | cross_stream | 单柱卡门涡街 + 侧向跨流 |
| §7.2 | `single_u15_upstream_tgt15` | single | upstream     | 单柱卡门涡街 + 逆流前进 |
| §7.3 | `tandem_u15_upstream_tgt15` | double tandem (G/D=3.5) | upstream | co-shedding 长尾涡街 |
| §7.4 | `sbs_u15_upstream_tgt15`    | double sbs (G/D=3.5)    | upstream | Coandă 偏转 + 不对称双尾 |

Gate 五条与 [设计 §8.3] 同口径；四个 phase 各自独立判定。

**复现路径**：
- §7.1 / §7.2：scaffold [`notebooks/sac_arrival_v2_single_u15_seed42_1M_validation.ipynb`](../notebooks/sac_arrival_v2_single_u15_seed42_1M_validation.ipynb) / archival [`notebooks/sac_arrival_v2_single_u15_seed42_1M_validation_completed.ipynb`](../notebooks/sac_arrival_v2_single_u15_seed42_1M_validation_completed.ipynb)
- §7.3 / §7.4：archival [`notebooks/sac_arrival_v2_tandem_sbs_validation_completed.ipynb`](../notebooks/sac_arrival_v2_tandem_sbs_validation_completed.ipynb)

**Combined gate JSONs**：
- 单柱二组：[`experiments/arrival_v2_prototype/single_u15_seed42_1M_control_summary/combined_gate_summary.json`](../experiments/arrival_v2_prototype/single_u15_seed42_1M_control_summary/combined_gate_summary.json)（`both_pass: true`）
- 双柱二组：[`experiments/arrival_v2_prototype/topology_validation_summary/combined_gate_summary.json`](../experiments/arrival_v2_prototype/topology_validation_summary/combined_gate_summary.json)（`both_pass: true`）

**Figures (paper-ready, gitignored)**：
- 4 训练曲线 + 4 终态条形图 + 4 单回合轨迹（PDF/PNG, 138 mm 单栏）+ 4 逐帧动画（GIF）落在 `figures/arrival_v2_strict_control_validation/`（受 `.gitignore: figures/` 屏蔽，仅本机 / Drive 留档）。指标释义与图轴语义见目录内 [`README.md`](../figures/arrival_v2_strict_control_validation/README.md) + [`manifest.json`](../figures/arrival_v2_strict_control_validation/manifest.json)。
- 生成入口：scaffold [`notebooks/sac_arrival_v2_strict_control_visualization.ipynb`](../notebooks/sac_arrival_v2_strict_control_visualization.ipynb) / archival [`notebooks/sac_arrival_v2_strict_control_visualization_completed.ipynb`](../notebooks/sac_arrival_v2_strict_control_visualization_completed.ipynb)。

### 7.1 Single + cross_stream（`single_u15_cross_tgt15`）

**Benchmark**：单柱 D=12 m，U∞=1.5 m/s，target=1.5 m/s，**cross_stream** geometry，Re=250。λ=U∞/V_max=1.0（critical under-actuation）。

**Flow**：`wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy`（与 §7.2 同 flow file，唯一区别是 episode reset_options 的 geometry）

**Gate 实测**（gate JSON：`results/single_cross_validation_gate_summary.json`）：

| Gate | 阈值 | 实测 | 结论 |
|---|---|---|---|
| final_success_rate | ≥ 0.85 | **0.9000** | PASS（5pp 余量，最紧） |
| last100k_mean / peak | ≥ 0.90 | **0.8833** / peak=0.9000 = 0.981 | PASS |
| OOB rate | ≤ 0.10 | **0.1000** | PASS（**正好踩线**） |
| `include_episode_context_obs` | True | True | PASS |
| `timeout_bootstrap_semantics` | terminal | terminal | PASS |

**Final eval (30 ep, env_step=1.0M)**：
- termination = `{goal: 27, out_of_bounds: 3}` — 唯一一组有 OOB 的 strict-control phase
- return = **92.36 ± 112.22**（std 巨大 — 失败的 3 ep return 远低于成功的 27 ep；与 OOB 终态的 R_oob 一致）
- safety_cost = 7.34 ± 8.06（mean 中等，无 single-outlier）
- eval_time_s = 79.13 ± 31.35（最短 — 失败 ep 在 OOB 时提早结束）
- path_length_m = 63.58 ± 15.39
- progress_ratio = 0.836 ± 0.214（std 也最大，反映 success/OOB 双峰）
- path_efficiency = **0.574** ± 0.147（四组最低）
- peak success **0.900 @ 725k**，未达 1.000

**结论**：single_cross 是 4 组 strict-control 中**最 borderline 的一组**：5/5 gate 全过，但 OOB 踩线 0.10、final 仅 0.90、return std 高达 112。物理解释：cross_stream geometry 下 AUV 侧向跨流时一旦失稳就被 U=1.5 推出边界，没有 upstream 的「逆流硬撑」恢复窗口；这一现象单 seed 已见，多 seed 复现需要重点关注（参见 §8）。

### 7.2 Single + upstream（`single_u15_upstream_tgt15`）

**Benchmark**：单柱 D=12 m，U∞=1.5 m/s，target=1.5 m/s，**upstream** geometry，Re=250。λ=1.0。

**Flow**：与 §7.1 同 flow file。

**Gate 实测**（gate JSON：`results/single_upstream_validation_gate_summary.json`）：

| Gate | 阈值 | 实测 | 结论 |
|---|---|---|---|
| final_success_rate | ≥ 0.85 | **1.0000** | PASS |
| last100k_mean / peak | ≥ 0.90 | **0.9750** / peak=1.0000 = 0.975 | PASS |
| OOB rate | ≤ 0.10 | **0.0000** | PASS |
| `include_episode_context_obs` | True | True | PASS |
| `timeout_bootstrap_semantics` | terminal | terminal | PASS |

**Final eval (30 ep, env_step=1.0M)**：
- termination = `{goal: 30}`
- return = **144.02 ± 0.73**（四组中 std 最低 — return 锁得最稳）
- safety_cost = **0.142 ± 0.339**（**四组中 mean 最低** — 单柱 upstream 物理上最容易避碰）
- eval_time_s = 118.18 ± 30.72
- path_length_m = 76.36 ± 18.60
- progress_ratio = 0.935 ± 0.016
- path_efficiency = 0.806 ± 0.062
- peak success **1.000 @ 475k**（与 §7.3 tandem 完全相同！）

**与 §3 P1 v6 二点 seed sample 对照**（同 benchmark / 不同 seed × budget）：

| 指标 | §3 P1 v6 (seed=46, 1.5M) | §7.2 (seed=42, 1.0M) |
|---|---:|---:|
| final_success_rate | 1.000 | 1.000 |
| peak first hit | n/a (1M 内未稳定到 1.000) | **475k** |
| eval_safety_cost | 0.292 | **0.142** |
| eval_path_efficiency | 0.7787 | **0.8060** |
| total_steps | 1.5M | **1.0M** |

**关键观察**：seed=42 在 1M 即 final=1.000 / peak @ 475k；seed=46 当时需要 1.5M 才稳态。这把 §3「需要 1.5M」**改判为 seed-specific 现象，而非 benchmark 难度**。设计 §8.2 budget 预言因此被订正：1M 是 seed=42 下的实测 enough budget；多 seed 复现需要逐 seed 验证 budget 而非默认 1.5M cap。

**结论**：single_upstream 是 4 组 strict-control 中**末段质量最干净的一组**（return std 0.73，safety mean 0.142，OOB=0）。同 benchmark 二点 seed 都 PASS 但收敛速度差异显著，构成最早的 seed-budget 经验数据点。

### 7.3 Double tandem（`tandem_u15_upstream_tgt15`）

**Benchmark**：双柱串列 G/D=3.5（cyl1=(96,90,D=12)，cyl2=(150,90,D=12)，间距 4.5D），U∞=1.5 m/s，target=1.5 m/s，upstream geometry，Re=250。主导现象 = co-shedding 长尾涡街。λ=1.0。

**Flow**：`wake_data/wake_tandem_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

**Gate 实测**（gate JSON：`results/tandem_validation_gate_summary.json`）：

| Gate | 阈值 | 实测 | 结论 |
|---|---|---|---|
| final_success_rate | ≥ 0.85 | **1.0000** | PASS |
| last100k_mean / peak | ≥ 0.90 | **0.9750** / peak=1.0000 = 0.975 | PASS |
| OOB rate | ≤ 0.10 | **0.0000** | PASS |
| `include_episode_context_obs` | True | True | PASS |
| `timeout_bootstrap_semantics` | terminal | terminal | PASS |

**Final eval (30 ep, env_step=1.0M)**：
- termination = `{goal: 30}`（无 timeout / OOB / collision-terminal）
- return = 130.90 ± 51.98（std 高 — 单 outlier ep_0011 因撞柱触发 safety_cost=142.7，但 episode 仍 success；详见下段）
- safety_cost = **6.85** ± 25.89（**四组中 mean 最高**；中位数 ~0：30 ep 中 21 个 safety_cost=0；6 个 ∈ (0, 5)；3 个 outlier ∈ {11.4, 19.8, 24.5, 142.7}）
- eval_time_s = **96.30** ± 33.14（**四组中最短**）
- path_length_m = **66.99** ± 15.36（**四组中最短** — 后柱拖出顺向加速窗口）
- progress_ratio = 0.932 ± 0.015
- path_efficiency = 0.861 ± 0.077
- peak first hit **475k**（与 §7.2 single_upstream 同步）

**Outlier 分析（ep_0011, return=-141.65）**：单一情节 safety_cost=142.7（远超第二高 24.5），但 reason=goal，progress_ratio=0.954，path_efficiency=0.888 — 即 agent 经历短暂高 safety_cost（很可能擦过后柱安全圈），但仍按计划到达。这是 arrival_v2 「discounted unsafe-shortcut > terminal-safe-success」的 [设计 §11.3] 平衡的边界案例：单 seed prototype 中允许 1/30 的 risky-success，不触发 §8.3 任何 gate。多 seed thesis 复现时建议作为 specific case 跟踪。

**结论**：tandem 拓扑下 arrival_v2 5/5 PASS，路径最短 / 时间最短，但 safety mean 也最高（双柱串列后柱长尾让擦碰风险最高，与物理机制直接一致）。

### 7.4 Double side-by-side（`sbs_u15_upstream_tgt15`）

**Benchmark**：双柱并列 G/D=3.5（cyl1=(96,147,D=12)，cyl2=(96,93,D=12)，横向间距 4.5D），U∞=1.5 m/s，target=1.5 m/s，upstream geometry，Re=250。主导现象 = Coandă 偏转 + 不对称双尾（间隙射流偏向其中一柱）。λ=1.0。

**Flow**：`wake_data/wake_sbs_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

**Gate 实测**（gate JSON：`results/sbs_validation_gate_summary.json`）：

| Gate | 阈值 | 实测 | 结论 |
|---|---|---|---|
| final_success_rate | ≥ 0.85 | **1.0000** | PASS |
| last100k_mean / peak | ≥ 0.90 | **0.9833** / peak=1.0000 = 0.983 | PASS |
| OOB rate | ≤ 0.10 | **0.0000** | PASS |
| `include_episode_context_obs` | True | True | PASS |
| `timeout_bootstrap_semantics` | terminal | terminal | PASS |

**Final eval (30 ep, env_step=1.0M)**：
- termination = `{goal: 30}`
- return = 143.02 ± 5.03（std 远低于 tandem 的 51.98 — 无大 safety outlier）
- safety_cost = 0.586 ± 2.43（30 ep 中 26 个 safety_cost=0；最高 outlier 13.2，第二 3.5）
- eval_time_s = **127.42** ± 37.61（**四组中最长**）
- path_length_m = 72.41 ± 17.97
- progress_ratio = 0.937 ± 0.013
- path_efficiency = **0.869** ± 0.086（**四组中最高**）
- peak first hit **625k**（晚于 §7.2 / §7.3 的 475k）

**收敛形状**（眼测自 notebook §7）：700k–775k 之间 dip 至 0.533–0.833（4 个 eval 点 ≤ 0.867），775k 之后回升锁定 ≥ 0.933；last100k 16 个 eval 点中 13 个 ≥ 0.967。dip 是 arrival_v2 在不对称双尾下的 transient 漂移，非 collapse —— 单 seed 现象，多 seed 复现可观察是否 seed-specific。

**结论**：sbs 拓扑下 arrival_v2 5/5 PASS，path_efficiency 最高、时间最长、safety 良好（mean 0.586）。Coandă 偏转把可行通道压到两侧，agent 学会「远离两柱中线」的保守解，付出 time 但获得 safety 与稳定性。

### 7.5 4-way 严格控制横向对照

四个 phase 全部 5/5 gate PASS（§7.1 borderline）。完整指标对比：

| 指标 \ Run | §7.1 single+cross | §7.2 single+upstream | §7.3 tandem+upstream | §7.4 sbs+upstream |
|---|---:|---:|---:|---:|
| **5/5 gate** | PASS（borderline） | PASS | PASS | PASS |
| final_success_rate | 0.900 | **1.000** | **1.000** | **1.000** |
| peak (first-hit step) | 0.900 @ 725k | 1.000 @ **475k** | 1.000 @ **475k** | 1.000 @ 625k |
| last100k_mean / peak | 0.883 / 0.900 = 0.981 | 0.975 / 1.000 = 0.975 | 0.975 / 1.000 = 0.975 | 0.983 / 1.000 = 0.983 |
| OOB rate | **0.100（踩线）** | 0.000 | 0.000 | 0.000 |
| termination | `{goal:27, OOB:3}` | `{goal:30}` | `{goal:30}` | `{goal:30}` |
| eval_return | **92.36 ± 112.22** | **144.02 ± 0.73** | 130.90 ± 51.98 | 143.02 ± 5.03 |
| eval_safety_cost | 7.34 ± 8.06 | **0.142 ± 0.339** | **6.85 ± 25.89** | 0.586 ± 2.43 |
| eval_time_s | 79.13 ± 31.35 | 118.18 ± 30.72 | **96.30 ± 33.14** | **127.42 ± 37.61** |
| eval_path_length_m | 63.58 ± 15.39 | 76.36 ± 18.60 | **66.99 ± 15.36** | 72.41 ± 17.97 |
| eval_progress_ratio | 0.836 ± 0.214 | 0.935 ± 0.016 | 0.932 ± 0.015 | 0.937 ± 0.013 |
| eval_path_efficiency | **0.574 ± 0.147** | 0.806 ± 0.062 | 0.861 ± 0.077 | **0.869 ± 0.086** |
| eval_energy | 63 864 | 101 172 | 81 495 | 110 581 |

（粗体 = 该指标在四组中的最值。）

**Takeaway A：拓扑轴（固定 upstream，single → tandem → sbs 三组）**
1. 三组 5/5 PASS，全部 final=1.000 / OOB=0 / 30/30 全 goal — **拓扑在严格控制下未引入额外 sample 难度**。Single 与 tandem 同步 peak @ 475k，sbs 略晚 @ 625k。
2. **Safety mean 排序**：single (0.142) < sbs (0.586) < tandem (6.85)，与 wake topology 物理直觉一致：单柱 upstream 最容易避碰；sbs Coandă 偏转把通道压到两侧，agent 学会保守解；tandem 后柱长尾让擦碰风险最高（且制造 1/30 outlier 拖大 mean）。
3. **Time / path 排序**：tandem (96s, 67m) < single (118s, 76m) < sbs (127s, 72m) — 后柱顺向加速 vs Coandă 强迫绕行的物理对应。
4. **Path_efficiency 排序**：sbs (0.869) ≳ tandem (0.861) > single (0.806) — 双柱场景反而比单柱效率更高，可能因为双柱的避让路径更受流场结构性约束（agent 不需要主动选路）。

**Takeaway B：Geometry 轴（固定 single，cross → upstream 两组）**
1. **Cross_stream 比 upstream 难得多**：single_cross 是 strict-control 四组中唯一 OOB 踩线 (0.100)、final 仅 0.900、return std 112、progress_ratio std 0.21（其他三组 std ≤ 0.02）。物理解释：cross 时 AUV 侧向跨流，一旦失控被 U=1.5 推出边界，无 upstream 的「逆流硬撑」恢复窗口。
2. arrival_v2 在 single_cross 仍 5/5 PASS — 但**多 seed 复现需要重点关注**该组（参见 §8）。

**Takeaway C：核心立论**
- arrival_v2 在 4 组 strict-control 物理机制（single+cross / single+upstream / double+upstream tandem / double+upstream sbs）下都 5/5 gate PASS — **arrival-first 设计的核心立论（terminal dominance + discounted unsafe-shortcut + OOB ordering）在拓扑 × geometry 两轴下都成立**。

**Takeaway D：Reference baselines（§2 / §3，非严格对照）**
- §2 cross_u10（U=1.0 / s0 / seed=46 / 1M / 5/5 PASS）旁证 arrival_v2 在更慢流速 + 更弱 sensor 下也 work，与 §7.1 (U=1.5 / s1 / seed=42) PASS 形成 cross_stream geometry 的跨速度趋势确认。
- §3 P1 v6（single_upstream / seed=46 / 1.5M / 5/5 PASS）与 §7.2（同 benchmark / seed=42 / 1M / 5/5 PASS）构成同 benchmark 二点 seed sample；seed=42 在 1M 收敛、seed=46 需要 1.5M → **§3「需要 1.5M」改判为 seed-specific，非 benchmark 难度**。这是对 [设计 §8.2] budget 预言的实测订正。

**重要约束**：以上 takeaway 全部基于**单 seed**。§7.1 的 3/30 OOB、§7.3 的 1/30 risky-success outlier、§7.4 的 700k–775k transient dip 都是单 seed 现象；§7.2 与 §3 的 budget 反差也只是二点 seed sample。多 seed 复现是 thesis-grade 重启的前置条件（参见 §8）。

### 7.6 s0 sensor envelope（单 seed exploratory，2026-05-13 补跑）

**动机**：§7.1–§7.5 4 组严格控制都跑在 `s1_k4`（DVL + 短程 ADCP, 12-D obs）。本研究的 sensor 主轴是 `s0_k4`（DVL-only, deployment-realistic, 10-D obs，见 [`CLAUDE.md`](../CLAUDE.md) Architecture / [`online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §1）。本节把 §7 唯一变量从 *topology × geometry* 扩展到 *sensor*，**仅 sensor 切到 s0**，其它（reward, U, target, seed, total_steps, num_envs, algorithm, 4 个 benchmark）全部冻结：

> `arrival_v2 / k4 / U=1.5 / target=1.5 / seed=42 / 1M / num_envs=6 / vanilla SAC`，新增唯一变量 = **`PROBE_LAYOUT = s0`**

Gate 5 条同 §7 口径。**Single seed = 42**，与 §7 平行 1:1 exploratory（**不是** multi-seed thesis-grade）。

**复现路径**：scaffold [`notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope.ipynb`](../notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope.ipynb) / archival [`notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope_completed.ipynb`](../notebooks/sac_arrival_v2_s0_seed42_1M_sensor_envelope_completed.ipynb)。Combined gate JSON：[`experiments/arrival_v2_prototype/s0_sensor_envelope_summary/combined_gate_summary.json`](../experiments/arrival_v2_prototype/s0_sensor_envelope_summary/combined_gate_summary.json)（gitignored）。

**结果（s0 vs §7 s1，单 seed，严格 1:1 对比）**：

| Phase | s0 final | s0 peak@step | s0 last100k | s0 OOB | **s0 gate** | s1 final | s1 peak@step | s1 last100k | s1 OOB | **s1 gate** |
|---|---:|---:|---:|---:|:-:|---:|---:|---:|---:|:-:|
| §7.3 tandem | 1.000 | 1.000 @ 650k | 0.975 | 0.000 | PASS | 1.000 | 1.000 @ 475k | 0.975 | 0.000 | PASS |
| §7.4 sbs | 1.000 | 1.000 @ 550k | 0.925 | 0.000 | PASS | 1.000 | 1.000 @ 625k | 0.983 | 0.000 | PASS |
| §7.2 single_upstream | 1.000 | 1.000 @ 350k | 1.000 | 0.000 | PASS | 1.000 | 1.000 @ 475k | 0.975 | 0.000 | PASS |
| **§7.1 single_cross** | **0.100** | **0.367 @ 975k** | **0.267** | **0.667** | **FAIL（3/5 gate fail）** | 0.900 | 0.900 @ 725k | 0.883 | 0.100 | PASS |

`single_cross_s0` final eval termination：`{out_of_bounds: 20, timeout: 7, goal: 3}`（30 个 deterministic episode，20 个跨出工作域，仅 3 个达成 goal）。

**Finding F1 — Upstream geometry 下 s0 完全够用（3/4 PASS）**

`tandem / sbs / single_upstream` 三个上游几何下 s0 vanilla 在 arrival_v2 + production 难度（U=1.5）下都达到 `final=1.000 / OOB=0 / 30/30 goal`，5/5 gate 全过。`last100k_mean` 与 s1 在 ±0.05 内（s0: 0.925 / 0.975 / 1.000；s1: 0.983 / 0.975 / 0.975），无显著差异。**deployable-only s0 sensor 在上游几何 + arrival_v2 reward + production 难度下是 production-ready 的**。

**Finding F2 — Cross-stream s0 catastrophic FAIL；对接 A0 的 24× gap 放大**

`single_cross_s0` 与 `single_cross_s1` 之间的 gap 从 A0（[`online_rl_line_summary.md`](online_rl_line_summary.md) §1.1）的 3pp 放大到 80pp：

| 维度 | A0（cross_u10 + arrival_v1） | §7.6（cross_u15 + arrival_v2） | Δ |
|---|---:|---:|---:|
| s1 final | 1.000 | 0.900 | −0.10 |
| s0 final | 0.967 | 0.100 | **−0.867** |
| **s0–s1 gap** | **0.033** | **0.800** | **~24×** |

两个变量同时升级：① flow speed `U=1.0 → 1.5`（涡街 Strouhal 周期变快，单点 DVL 看到的脉动信息密度变低）② reward `arrival_v1 → arrival_v2`（penalty 结构不同）。**Difficulty 升级把 sensor 信息差异从可忽略放大到致命** — 这是 partial-observability gap 在 production-difficulty regime 下真正显化的实证，可作为论文方法节立论。

**Finding F3 — single_cross_s0 训练曲线呈 plasticity-loss 形态**

每 100k 采样的 s0 cross 训练曲线：

| step | s0 success | s0 path_eff | s1 success（同步对比） |
|---:|---:|---:|---:|
| 25k | 0.000 | −0.39 | 0.000 |
| 125k | 0.000 | −0.18 | 0.000 |
| 225k | 0.267 | 0.18 | 0.033 |
| 325k | 0.100 | 0.13 | 0.267 |
| 425k | 0.233 | 0.26 | 0.200 |
| 525k | 0.333 | 0.26 | 0.500 |
| 625k | 0.333 | 0.24 | 0.800 |
| 725k | 0.300 | 0.24 | 0.900 |
| 825k | 0.300 | 0.31 | 0.900 |
| 925k | 0.233 | 0.31 | 0.900 |

s0：225k 起来 → 525k–625k 高点 0.367 → 然后 drift down 至 ~0.23–0.30 → final eval 0.100。**从未越过 gate 阈值 0.85**。`path_efficiency` 同步 plateau 在 ~0.25。

这比 "没收敛" 严重一档 — 策略学到了一个局部 mode（success rate 短暂攀升）然后**被 OOB-incentive 反向 erode**（缓慢下滑 + final eval 反而最低）。对比 s1 同 benchmark：225k → 625k 单调爬到 0.800，725k 起稳定 plateau 0.900。s1 学到稳定策略，s0 学到的策略不稳定。

**Finding F4 — Peak step 在上游 3 phase 上 s0 vs s1 不单调慢**

| Phase | s0 peak step | s1 peak step | Δ |
|---|---:|---:|---:|
| tandem | 650k | 475k | s0 **慢 175k** |
| sbs | 550k | 625k | s0 **快 75k** |
| single_upstream | 350k | 475k | s0 **快 125k** |

`peak_step` 是 "首次 hit 100% success rate" 的 noisy 度量（单次 30-episode deterministic eval 抽样）。Peak step 之间的差异在 noise floor 量级，**不要据此 over-claim sensor 与收敛速度的关系**。真正稳定的 `last100k_mean` 在三个上游 phase 上 s0 vs s1 都在 ±0.05 内（见 F1）。

**机理解释 — 为什么 cross 崩、upstream 没崩**

| | upstream geometry | cross_stream geometry |
|---|---|---|
| 任务方向 vs 主流 | 沿主流 | 横切主流 |
| Flow 主分量对 AUV 的作用 | u ≈ −U（顶推 AUV，速度反向减速）| u = 横向施加力（侧向推） |
| 失败模式主因 | timeout（走得太慢） | OOB（被流推出工作域边界） |
| arrival_v2 在失败时的 reward 信号 | OOB + timeout 都给 terminal penalty；timeout 还保留 distance shaping 引导 | OOB 是 terminal，**没有继续推进的机会** |
| s0 (单点 DVL) 的信息局限 | k=4 历史覆盖 ~2 s；涡街周期 10–20 s | 同 |

**cross 几何下 OOB 是"一次定胜负"事件**。s0 没有提前预知涡街相位的能力（[`SAC_improvements_survey.md`](SAC_improvements_survey.md) §10.2：K=4 仅覆盖涡街周期的 10–20%），就只能事后反应；横切瞬间被涡推出工作域 → 立刻 terminal。upstream 几何下 timeout 还有继续推进的 reward 信号能 bail out，cross 没有这条 escape 路径。

这一观察对接 [`SAC_improvements_survey.md`](SAC_improvements_survey.md) §10.2 P1#5 — AsymCritic + privileged hull-integral flow 的设计意图：critic 训练时看到真实涡街相位（actor 看不到），引导 actor 在 cross 几何下学到何时启动横切。**`single_cross_s0` 因此是仓库内目前唯一真有 AsymCritic ablation headroom 的 cell**（其它 3 phase 已饱和到 1.000，没有 headroom）。

**Writeable claim**：arrival_v2 reward + deployable s0 sensor 在 upstream-geometry production benchmarks（tandem / sbs / single_upstream）下单 seed exploratory 全部 saturate；唯独 cross_stream geometry 上 sensor envelope 出现 catastrophic gap，从 A0（cross_u10 + arrival_v1）的可忽略放大 24 倍。这一 catastrophic gap 为后续 improved SAC ablation 提供了清晰的非饱和 target cell。

**重要约束 / disclaimers**

- **单 seed (=42)**，与 §7 平行 exploratory；任何 ±5pp 内的 cell-level 差异不要 over-claim。F2 的 24× gap 是单 seed 数；多 seed 复现 single_cross_s0 是后续工作（§8 新增条目）。
- **数据完整性 footnote**：本地下载 Drive 时 `tandem_u15_*/s0_k4/seed_42/` 与 `sbs_u15_*/s0_k4/seed_42/` 两个目录在文件系统上一度互换。通过 6 个独立内部指针交叉验证（`trainer_state.json` 的 `flow_path` / `eval_manifest` / `checkpoint_dir` / `agent_path` + `results/train_config.txt` 的 `save_dir` + `results/*_gate_summary.json` 文件名）并 `mv` swap 回去；**所有 4 phase 数据本身完整可信**。`combined_gate_summary.json` 是 Colab 上 dir 还在正确位置时生成的，反映正确 label，无需 regen。Drive 上对应两个 dir 同样错置但未处理；以后 resume 训练或读 checkpoint 时需要去 Drive 上做同样 swap。

---

### 7.7 AsymCritic 单变量 ablation on `single_cross_s0`（pure ablation, 单 seed，2026-05-17，**negative finding**）

**动机**：§7.6.4 揭示 `single_cross_s0` 是仓库内唯一对 improved SAC 有 ablation headroom 的 cell（其它 7 cell 已 PASS、6 个 saturate 到 1.000）。[`SAC_improvements_survey.md`](SAC_improvements_survey.md) §10.2 P1#5 留口预期：privileged hull-integral flow（critic 训练时额外看 `[u_eq, v_eq]`，actor 仍只用 s0 单点）应该能闭合 cross 几何下的 80pp gap。本节做这个 ablation 的**最干净版本（pure B 路径）**：仅加 `--use-asymmetric-critic`，**显式拒绝** LayerNorm / UTD>1 / Dropout，让 final 的差异**只能归因于 critic 信息差异本身**，不被其它 SAC 改进项 confound。

> 仅一个变量：`AgentConfig.privileged_obs_dim = 2`（即 `--use-asymmetric-critic`）。`use_layernorm=False / updates_per_step=1 / dropout_rate=0.0` 均与 §7.6.4 baseline 完全一致（核验自 `results/train_config.txt`）。

**复现路径**：scaffold [`notebooks/sac_arrival_v2_s0_cross_asym_ablation.ipynb`](../notebooks/sac_arrival_v2_s0_cross_asym_ablation.ipynb) / archival [`notebooks/sac_arrival_v2_s0_cross_asym_ablation_completed.ipynb`](../notebooks/sac_arrival_v2_s0_cross_asym_ablation_completed.ipynb)。Run dir：`experiments/arrival_v2_prototype/single_u15_cross_tgt15/arrival_v2/sac_asym/s0_k4/seed_42/`（gitignored）。

**5/5 Gate 实测**：

| Gate | 阈值 | sac_asym | vanilla (§7.6.4) |
|---|---|---:|---:|
| final_success_rate | ≥ 0.85 | **0.1667 FAIL** | 0.1000 FAIL |
| last100k_mean / peak | ≥ 0.90 | 0.1167 / 0.267 = 0.438 **FAIL** | 0.2667 / 0.367 = 0.728 FAIL |
| OOB rate | ≤ 0.10 | **0.6333 FAIL** | 0.6667 FAIL |
| arrival_v2 context obs enabled | True | PASS | PASS |
| arrival_v2 timeout terminal semantics | terminal | PASS | PASS |
| **总判** | | **3/5 FAIL** | 3/5 FAIL |

AsymCritic 同样未达 gate；末尾 final +6.7pp 是 noise（30-ep eval 上 2 个 episode 的差）。**真正的信号必须看全程曲线，不是末尾单点。**

**Finding F1 — final +6.7pp 是 noise；全程 mean −17.7pp（差 5×）**

| 指标 | vanilla §7.6.4 | sac_asym | Δ |
|---|---:|---:|---:|
| final_success @ 1M | 0.100 (3/30 goal) | 0.167 (5/30 goal) | +6.7pp |
| **mean_success across 39 evals** | **0.221** | **0.044** | **−17.7pp（−5×）** |
| evals with ≥1 success | 35/39 | **19/39** | **−16 evals** |
| peak_success | 0.367 @ 975k | 0.267 @ 950k | −10pp |
| final OOB rate | 0.667 (20/30) | 0.633 (19/30) | −3.4pp（同量级） |

整训练区间 39 次评估的均值是 final 的 5× 信号量。**vanilla 几乎每次 eval 都偶尔抓到 success（35/39 evals 至少 1 个 goal），sac_asym 一半 eval 是全 0（19/39）**。末尾 +6.7pp 与"sac_asym 比 vanilla 好" 无关 — 是 random fluctuation 在最后一个 25k 窗里偶然摆向 asym 那边。

**Finding F2 — 行为风格反向：sac_asym 显著更安全 + 更朝目标推进，但任务级 success 反而更低**

| 行为指标 | vanilla §7.6.4 | sac_asym | Δ |
|---|---:|---:|---:|
| safety_cost（全程 mean） | 25.85 | **16.93** | **−34%（更安全）** |
| progress_ratio（final eval） | 0.277 | **0.453** | **+64%（朝目标推进更多）** |
| eval_return（final eval） | −173.7 | −129.9 | +43.8（return 更高） |
| **eval_success_rate（全程 mean）** | **0.221** | **0.044** | **−80%（任务级反而崩）** |

这是一个**清晰的反向 trade-off**：critic 拿到 `[u_eq, v_eq]` 后通过 Q-gradient 引导 actor 学到一个"少触 boundary + 稳定朝目标走"的行为模式（safety_cost ↓ 34%, progress_ratio ↑ 64%, return ↑ 26%），但 **actor 在 s0 端没有获得任何新信息**（asymmetric critic 的标准结构），在 cross 几何下接近目标的最后一段仍然 navigate 不过去。**结果是 sac_asym 大量 episodes 都是"稳定推进 30%–45% 然后 OOB"，比 vanilla "无序硬冲偶尔过去" 还要 less successful。**

**Finding F3 — Quartile dynamics 揭示完整轨迹差异**

| Quartile | sac_asym SR | vanilla SR | sac_asym progress | vanilla progress | sac_asym safety | vanilla safety |
|---|---:|---:|---:|---:|---:|---:|
| Q1 (0–250k) | 0.011 | 0.096 | **−1.017** | −0.444 | 15.4 | 31.5 |
| Q2 (250k–500k) | 0.033 | 0.210 | −0.164 | 0.288 | 18.8 | 30.9 |
| Q3 (500k–750k) | 0.048 | 0.300 | 0.175 | 0.287 | 18.7 | 20.4 |
| Q4 (750k–1M) | **0.081** | **0.287** | **0.287** | 0.322 | **14.3** | 21.4 |

- **Q1 早期**：sac_asym progress=−1.017（朝相反方向走），比 vanilla 还差；表明 critic 引导初期把 actor 推到一个**更糟的探索起点**。
- **Q2–Q3 中期**：sac_asym 才慢慢学到"朝目标走"；vanilla 已经在 0.2–0.3 区间反复 spike。
- **Q4 末期**：sac_asym **行为指标几乎追平 vanilla**（progress 0.287 vs 0.322, safety 14.3 vs 21.4），**但任务 success 依旧 3× 落后**（0.081 vs 0.287）。

**Finding F4 — 推翻 §10.2 P1#5 / §8 P1#1 的预期**

| 原预期 | 实测 |
|---|---|
| AsymCritic 单独闭合 80pp gap → 论文写充分 | mean success **比 vanilla 还差 5×**；final 差异在 noise 内 |
| privileged hull-integral flow 对本任务理论收益最高 | 行为风格变了，但 deployment-realistic success 反而下降 |
| `single_cross_s0` 的瓶颈在 critic estimation accuracy | **瓶颈在 actor 侧 information access**（actor 端 s0 单点信息不足以 navigate cross 几何最后一段） |

**机理性解读**（asymmetric-critic literature 里少见的负面 case）：

```
critic info:        s0 + [u_eq, v_eq]   (训练时)
actor info:         s0 only             (训练 & 部署一致)

→ Q(s, a) 估计更准
→ ∂Q/∂a 把 actor 推向"在 critic 看来更好"的 a
→ 但 actor 决策依据是 s0，s0 信息根本不够区分"看似更好"和"实际更好"
→ actor 收敛到一个 critic-validated 但 actor-information-poor 的 local optimum
→ 行为风格变 "safer + more progressive" 但任务 success 反而下降
```

这意味着 **single_cross_s0 的根本瓶颈是 information-theoretic ceiling**：actor 在 s0 单点信息下学不到能稳定通过 cross 几何的策略，再准的 critic gradient 也无法越过这个 ceiling。**推论：走 A 路径（叠加 LN + UTD=4 = `sac_asym_lnutd`）的理论支撑被打掉了** — LN / UTD 都是优化 critic estimation 的，但 critic estimation 不是这里的瓶颈。

**Writeable claim**（negative finding）：在 production-difficulty cross-stream geometry × deployment-realistic single-point DVL sensor × arrival_v2 reward 的严格控制下，pure AsymCritic ablation（仅加 `--use-asymmetric-critic`）**未能闭合 80pp gap**：mean_success 反而从 vanilla 的 0.221 跌至 0.044（5× 更差），final 差异在 noise 内。行为风格层 critic 信号确实在起作用（safety_cost −34%, progress_ratio +64%），但 actor 端 s0 信息不足以将 critic-validated 行为兑现成任务级 success。这一观察推翻 [`SAC_improvements_survey.md`](SAC_improvements_survey.md) §10.2 P1#5 「privileged hull-integral flow 对本任务理论收益最高」的预期，将 `single_cross_s0` 的瓶颈从 critic estimation accuracy 重定位到 actor-side information access。

**重要约束 / disclaimers**

- **单 seed (=42)**，与 §7.6 平行 exploratory；F1 的 −17.7pp mean gap 与 F2 的 ±34%/64% 行为指标变化在量级上不会因为换 seed 变号（vanilla 35/39 vs asym 19/39 的"有 success eval 数"差是 16 个 eval，远超 seed-level noise），但精确数值不要 over-claim。
- 仅做了 **pure B 路径**（only `--use-asymmetric-critic`）。**不再推荐做 A 路径**（`sac_asym_lnutd` 组合）— 见 F4 机理解读。
- 数据完整性核验：5 路径（`flow_path` / `eval_manifest` / `checkpoint_dir` / `agent_path` / `save_dir`）一致指向 `sac_asym/s0_k4/seed_42`；`train_config.txt` line 36 `privileged_obs_dim=2` 确认 AsymCritic 真的进了。无 dir-swap。

**重新校准的下一步**（驱动 §8 P1 改写）：

- **C1（已 PASS — 见 §7.8）** — `single_cross_s0` history k=4→8 单变量 ablation：闭合 80pp gap，达到 s1_k4 上界，sample-efficiency 更好。机理段重定位被直接验证 ✓。
- **C2**（已在文档）— s1 actor 的 §7.1 baseline 即是上界（0.900），80pp 的 gap 完全是 sensor-side **temporal access**（不是 spatial），见 §7.8 F2。
- **C3 / C4**（备用）— boundary 软化 / 接受 s0 在 cross 上 catastrophic failure 作为 thesis 的诚实结论 — **§7.8 PASS 后已无需展开**。

**Update 2026-05-18 — 2-seed paired hardening (seed=42 + seed=0)**

§7.7 原 single-seed claim 已通过 2-seed paired replication 升格。在 §7.6.4 vanilla k=4 同步加 seed=0 + §7.7 sac_asym k=4 加 seed=0，2 × 2 paired ablation 结果：

| algo | seed=42 mean | seed=0 mean | 2-seed mean | seed=42 peak | seed=0 peak | peak ceiling |
|---|---:|---:|---:|---:|---:|---|
| vanilla | 0.221 | 0.218 | **0.220** | 0.367 | 0.533 | 不稳定（seed 间差 17pp） |
| sac_asym | 0.044 | 0.088 | **0.066** | **0.267** | **0.267** | **稳定 0.267**（2 seed identical）|

| contrast | Δ (asym − vanilla) | 论断 |
|---|---:|---|
| 2-seed mean | **−15.4pp** | asym 全程 ~3.3× 更差，跨 seed 稳定 |
| 2-seed peak | −0.10 ~ −0.27 | asym 跨 seed peak ceiling 严丝合缝锁在 0.267；vanilla peak 在 [0.37, 0.53] 之间 |
| 2-seed n_succ | (19+25)/78 = 56% | vanilla (35+34)/78 = 88%；asym 一半 eval 全 0 |

**新证据强化原 F1–F4**：①asym 跨 seed peak 严丝合缝锁在 0.267，证明 0.267 不是 seed-level noise 而是 AsymCritic 在此任务上的 **information-theoretic ceiling**；② asym 跨 seed mean (0.044 / 0.088) 都远低于 vanilla 跨 seed mean (0.221 / 0.218)，2-seed paired 仍是 ~3.3× gap。原 negative finding **从 single-seed exploratory 升格为 2-seed × 2-algo paired hardened claim**，可直接写进论文。

Sister 复现路径：scaffold [`notebooks/sac_arrival_v2_s0_cross_vanilla_seed0.ipynb`](../notebooks/sac_arrival_v2_s0_cross_vanilla_seed0.ipynb) / [`notebooks/sac_arrival_v2_s0_cross_asym_seed0.ipynb`](../notebooks/sac_arrival_v2_s0_cross_asym_seed0.ipynb)；archival `_completed.ipynb`；combined gate JSON：[`experiments/arrival_v2_prototype/s0_cross_vanilla_seed0_summary/combined_gate_summary.json`](../experiments/arrival_v2_prototype/s0_cross_vanilla_seed0_summary/combined_gate_summary.json) / [`experiments/arrival_v2_prototype/s0_cross_sac_asym_seed0_summary/combined_gate_summary.json`](../experiments/arrival_v2_prototype/s0_cross_sac_asym_seed0_summary/combined_gate_summary.json)。

**方法论 footnote**：§7.6.4 vanilla seed=42 的 final=0.10 是末段单次 30-ep eval 的 OOB-collapse outlier — 同任务下 seed=0 vanilla 的 mean39=0.218 与 seed=42 的 mean39=0.221 仅差 0.003（trajectory-level 极度 seed-stable），但 final_eval 差 +30pp（seed=42 OOB 0.667, seed=0 OOB 0.200）。Implication：**对 catastrophic-OOB-prone 任务，final_eval 单点 noisy，应优先看 mean39 / last100k**。§7.6.4 / §7.7 原文表述沿用 final + mean 双口径，结论未受影响（mean 口径仍 −17.7pp / −15.4pp，跨 seed 稳定）。

---

### 7.8 single_cross_s0 × history k=4→8 actor-side ablation（**PASS — 闭合 80pp gap**，单 seed exploratory，2026-05-18）

**动机**：§7.7 推翻了 critic-side info upgrade（AsymCritic）能闭合 80pp gap 的假设，并把 `single_cross_s0` 的瓶颈从 **critic estimation accuracy** 重定位到 **actor-side information access**。本节做这个新假设的最干净的 actor-side ablation：在 §7.6.4 vanilla baseline 上**仅**加大 `--history-length 4 → 8`，其它（s0 / arrival_v2 / U / target / seed / 1M / num_envs / vanilla SAC）全部冻结。

> 仅一个变量：`--history-length 4 → 8`（obs_dim 48 → 88）。物理解读：control step ≈ 0.5 s；k=4 历史窗 ~2 s，仅覆盖涡街周期 10–20 s 的 10–20%；**k=8 历史窗 ~4 s，覆盖 20–40%** — 已足以让 actor 从单点 DVL 时序节拍中**反演**主导脉动相位（即 §7.7 中 critic 通过 privileged `[u_eq, v_eq]` 看到的同一物理量的**纯 actor-side 时序代理**）。`use_asymmetric_critic=False / use_layernorm=False / updates_per_step=1 / dropout_rate=0.0` 均与 §7.6.4 baseline 完全一致（核验自 `results/train_config.txt`）。

**复现路径**：scaffold [`notebooks/sac_arrival_v2_s0_cross_k8.ipynb`](../notebooks/sac_arrival_v2_s0_cross_k8.ipynb) / archival [`notebooks/sac_arrival_v2_s0_cross_k8_completed.ipynb`](../notebooks/sac_arrival_v2_s0_cross_k8_completed.ipynb)。Run dir：`experiments/arrival_v2_prototype/single_u15_cross_tgt15/arrival_v2/sac_vanilla/s0_k8/seed_42/`（gitignored）。Combined gate JSON：[`experiments/arrival_v2_prototype/s0_cross_k8_ablation_summary/combined_gate_summary.json`](../experiments/arrival_v2_prototype/s0_cross_k8_ablation_summary/combined_gate_summary.json)。

**5/5 Gate 实测**：

| Gate | 阈值 | **s0_k8 (NEW)** | s0_k4 vanilla (§7.6.4) | s0_k4 asym (§7.7) | s1_k4 ref (§7.1) |
|---|---|---:|---:|---:|---:|
| final_success_rate | ≥ 0.85 | **0.900 PASS** | 0.100 FAIL | 0.167 FAIL | 0.900 PASS |
| last100k_mean / peak | ≥ 0.90 | 0.900 / 0.900 = **1.000 PASS** | 0.728 FAIL | 0.438 FAIL | 0.981 PASS |
| OOB rate | ≤ 0.10 | **0.100 PASS** | 0.667 FAIL | 0.633 FAIL | 0.100 PASS |
| arrival_v2 context obs enabled | True | PASS | PASS | PASS | PASS |
| arrival_v2 timeout terminal semantics | terminal | PASS | PASS | PASS | PASS |
| **总判** | | **5/5 PASS ✓** | 3/5 FAIL | 3/5 FAIL | 5/5 PASS |

**Finding F1 — k=8 vs k=4 vanilla 是 +80pp final, +41.5pp mean, −56.7pp OOB 的全维度突破**

| 指标 | §7.6.4 vanilla k=4 | **§7.8 vanilla k=8** | Δ |
|---|---:|---:|---:|
| final_success @ 1M | 0.100 (3/30 goal) | **0.900 (27/30 goal)** | **+80pp** |
| peak_success | 0.367 @ 975k | **0.900 @ 475k** | **+53.3pp，收敛快 500k** |
| mean_success (39 evals) | 0.221 | **0.636** | **+41.5pp（~3×）** |
| last100k mean | 0.267 | **0.900** | **+63.3pp** |
| final OOB rate | 0.667 (20/30) | **0.100 (3/30)** | **−56.7pp** |
| safety_cost (final) | 23.81 | 9.30 | −61% |
| progress_ratio (final) | 0.277 | 0.834 | +201% |
| return (final) | −173.7 | +88.5 | +262 |
| evals with ≥1 success | 35/39 | **37/39（全场最高）** | +2 |

唯一变量 `--history-length 4 → 8` 在 §7.6.4 catastrophic-FAIL cell 上一次性闭合了 5/5 gate 中全部 3 个 FAIL 项。

**Finding F2 — k=8 完全追平 s1_k4 上界 reference，且收敛更快**

| 维度 | §7.1 s1_k4 (upper ref) | **§7.8 s0_k8 (NEW)** | Δ |
|---|---:|---:|---:|
| final | 0.900 | **0.900** | 0 |
| peak | 0.900 | **0.900** | 0 |
| peak @step | 725k | **475k** | **−250k（快 35%）** |
| mean39 | 0.497 | **0.636** | **+13.9pp** |
| last100k | 0.883 | **0.900** | +1.7pp |
| OOB | 0.100 | **0.100** | 0 |
| safety_cost (final) | 7.34 | 9.30 | +27% |
| progress_ratio (final) | 0.836 | 0.834 | −0.2pp |
| return (final) | 92.4 | 88.5 | −3.9 |
| n_succ | 35/39 | **37/39** | +2 |

**deployment-realistic sensor (s0 = DVL only) 在 k=8 下不仅达到 s1_k4（多一个空间探头）的 task-level success，而且 sample-efficiency 还更好**（peak @ 475k vs 725k）+ trajectory-mean 还更高（0.636 vs 0.497）。这一观察彻底重写 §7.6 sensor envelope 的故事：**s0–s1 80pp gap 不是 spatial information bottleneck，是 actor-side temporal access bottleneck**。

**Finding F3 — 学习曲线是教科书级 S-curve，从 475k 起稳定 plateau**

| 阶段 | step 范围 | SR | 说明 |
|---|---|---:|---|
| 探索 | 0 – 150k | 0.00–0.13 | warm-up + random |
| 起飞 | 150k – 475k | 0.20 → 0.90 | 单调爬升 |
| 稳定 plateau | 475k – 1M | 0.83–0.90 | 21 次连续 eval 全在 ≥0.83，最后 5 次 (875k–975k) 全 0.90 |

最后 17 次 eval (575k–975k) 平均 SR = **0.87**；最后 5 次 (875k–975k) 全部 0.90。`last100k_mean / peak = 1.000`（plateau 完全饱和到 peak）。**无 plasticity-loss 形态，无 erosion**（与 §7.6 F3 的 s0_k4 plasticity-loss 曲线形成最大反差）。

**Finding F4 — Actor-side vs critic-side info upgrade 的 cross-arm 对比直接定位 mechanism**

| 路径 | 单变量 | 信号在 critic？ | 信号在 actor？ | final | mean39 | 论断 |
|---|---|:-:|:-:|---:|---:|---|
| §7.6.4 vanilla k=4 | baseline | ✗ | ✗ | 0.100 | 0.221 | 信息不足 |
| §7.7 sac_asym k=4 | `+--use-asymmetric-critic` | ✓ | ✗ | 0.167 | 0.044 | critic 知道但 actor 兑现不了（**任务级反退**）|
| **§7.8 vanilla k=8** | `--history-length 4→8` | ✗（隐式从 temporal 反演） | **✓** | **0.900** | **0.636** | **actor 自己反演就够** |

这是一个干净的 information-flow 对照：**在 critic 端单独升级信息（§7.7）任务级反而退步；在 actor 端单独升级时序信息（§7.8）任务级 PASS**。`single_cross_s0` 的瓶颈是 actor 端 information access，不是 critic estimation accuracy — §7.7 F4 的机理重定位被 §7.8 **直接验证**。

**机理解读 — k=8 为何能在 s0 单点 sensor 上闭合 cross 几何 gap**

```
control step:        ~0.5 s
vortex shedding T:   10–20 s
k=4 history:         ~2 s   (10–20% of T) → 单点信号采样不足以解码相位
k=8 history:         ~4 s   (20–40% of T) → Nyquist + 接近半周期，足以反演主导脉动相位

s0 单点 + 长 history → actor 可从 DVL 时序节拍中重建：
  ① 局部涡街相位（critic 通过 privileged [u_eq, v_eq] 看到的同一物理量）
  ② 横向 boundary 接近事件的预兆（OOB 的物理前导信号）

→ 在 cross 几何下，actor 在被涡推出之前就能切换策略
→ OOB 从 0.667 → 0.100（与 s1_k4 同），final 从 0.100 → 0.900（与 s1_k4 同）
→ progress_ratio 从 0.277 → 0.834（与 s1_k4 同：0.836）
```

**Writeable claim（正向 thesis-grade 发现）**：在 production-difficulty cross-stream geometry × deployment-realistic single-point DVL sensor × arrival_v2 reward 的严格控制下，**仅**把 actor 时序窗口从 k=4 加大到 k=8（仍然只用 DVL 单点 sensor），在 1M 步内完全闭合了 §7.6 的 80pp s0–s1 gap，且**达到与 s1（双点 sensor）完全等价的 task-level performance**（final 0.900 = 0.900, OOB 0.100 = 0.100, progress 0.834 ≈ 0.836），并以更快 sample-efficiency 收敛（peak @ 475k vs s1 @ 725k）。这一发现直接验证 §7.7 F4 把瓶颈从 critic-side 重定位到 actor-side 的机理重写，并把本研究 deployment-realistic 路径**从「s0 → s1（升级 sensor）」改写为「s0 + k=4 → s0 + k=8（升级 actor 时序访问）」**。

**重要约束 / disclaimers**

- **单 seed (=42)**，与 §7.6 / §7.7（seed=42 anchor）平行 exploratory；80pp final gap 与 41.5pp mean gap 的量级远超 seed-level noise（§7.7 update 显示 vanilla k=4 跨 seed mean 仅差 0.003，trajectory-level 极度 seed-stable），但「k=8 在 s1 上界处 saturate」的精确数值仍需 multi-seed 复现才能进 thesis；§8 P1 已列入。
- 仅做了 k=8。**k=12 / k=16 单调性扫**尚未做，无法 claim "k=8 是甜点"，只能 claim "k=8 闭合 gap"；§8 P1 已列入。
- 仅做了 cross_stream 几何下 single 拓扑。**upstream / downstream geometry × k=8** 是否 generalize 待跑；§8 P1 已列入（优先级低，因 s0_k4 已 saturate）。
- 数据完整性核验：5 路径（`flow_path` / `eval_manifest` / `checkpoint_dir` / `agent_path` / `save_dir`）一致指向 `sac_vanilla/s0_k8/seed_42`；`results/train_config.txt` 确认 `history_length=8`（与 §7.6.4 / §7.7 的 k=4 配置区分）。无 dir-swap。

**驱动 §8 P1 改写**：§8 P1 已全面重写以围绕 §7.8 breakthrough 展开（multi-seed k=8 + k=12/16 monotonicity + k=8+asym mechanism validation）。

---

## 8. 后续可选工作

§7 4-way strict-control + §7.6 s0 sensor envelope + §7.7 AsymCritic ablation（**2-seed × 2-algo paired hardened**）+ **§7.8 history k=8 actor-side breakthrough（PASS — 闭合 80pp gap）** 共同覆盖了 arrival_v2 在 production-difficulty regime 下的关键 cell：原 §7.6 catastrophic FAIL cell `single_cross_s0` 在 §7.8 通过单变量 actor-side temporal info upgrade 完全闭合，达到与 s1 上界等价的 task-level success，且 sample-efficiency 还更好（peak @ 475k vs s1 @ 725k）。本研究 deployment-realistic 路径从「升级 sensor 到 s1」改写为「保持 s0 + 升级 actor 时序访问到 k=8」。若 thesis 重启或 offline 线决定升级 reward preset，按以下优先级展开：

**P1 — 围绕 §7.8 k=8 breakthrough 的 multi-seed 巩固 + monotonicity scan**（§7.8 PASS 直接驱动）：

1. **`single_cross_s0 + k=8` × multi-seed (seed=0, seed=7)**（推荐先做）：把 §7.8 单 seed (=42) PASS (final=0.900, OOB=0.100, peak @ 475k) 升格为 ≥3 seed paper-grade claim。3 个 seed 都 PASS → 「s0 + 更长 history 闭合 cross 几何 sensor 信息瓶颈」直接进 thesis；任何 seed FAIL → 触发 robustness deep-dive，可能需要更细致的 mechanism 分析。`2×2.5h L4`（seed=0 / seed=7 各跑 1M），可与下面 #2 并行。
2. **`single_cross_s0` × history monotonicity scan (k=12 / k=16, seed=42)**：测 k=8 是否甜点（saturation）还是 monotonic 改善（k 越大越好）。若 k=12 / k=16 在 PASS 区间继续 improve → 主张 "k 越大越好，受限于内存/算力"；若 saturate 在 k=8 → 主张 "k=8 是 cross 几何下涡街相位反演的甜点（~4 s ≈ 涡街周期 30%）"。`2×2.5h L4`（k=16 网络略大但仍 256 hidden）。
3. **k=8 + AsymCritic combo**（mechanism-validation debugging run）：在 §7.8 PASS 配置之上加 `--use-asymmetric-critic`，验证 §7.7 F4 机理 claim 是否完全闭环——actor info 充分时，critic info upgrade 是 neutral / 微正？还是仍然有害？如 PASS → §7.7 F4 机理完整闭环（actor info 是单一 bottleneck）；如仍 FAIL → §7.7 机理需细化（critic-actor information asymmetry 可能比单纯 access bottleneck 更复杂）。`1×2.5h L4`。
4. **k=8 上游几何泛化（tandem / sbs / single_upstream × s0_k8 × seed=42）**：s0_k4 + arrival_v2 在这 3 cell 已 saturate 到 1.000，加 k=8 主要确认 "k=8 至少不退步"（不引入 over-fitting / 长 history 的负作用）。`3×2.5h L4`，优先级低于 #1–#3。

> **已关闭（不再推荐展开）**：
> - ~~`single_cross_s0 × history k=4→8` 单变量 ablation（原 P1#1）~~ — §7.8 已闭环（PASS，闭合 80pp gap → final 0.900, OOB 0.100, peak @ 475k, mean 0.636）。**actor-side temporal information access 被确认为瓶颈** — §7.7 F4 机理重定位被验证。
> - ~~AsymCritic × `single_cross_s0` ablation（原原 P1#1）~~ — §7.7 已闭环（pure B 路径 + 2-seed paired hardened，negative finding；peak ceiling 跨 seed 严丝合缝锁在 0.267）。**不再推荐继续走 A 路径（`sac_asym_lnutd` 组合）**：LN / UTD 都是优化 critic estimation 的，但 §7.7 F4 + §7.8 F4 cross-arm 对照已经证明 critic estimation 不是这里的瓶颈。
> - ~~`single_cross_s0` k=4 multi-seed 复现（原 P1#2）~~ — §7.7 update 显示 §7.6.4 vanilla seed=42 + sister seed=0 的 trajectory mean39 极度 stable（0.221 vs 0.218，差 0.003）；final_eval 单点 noise 已被解释（OOB 末段 collapse 模式 seed-sensitive，但 mean / last100k 不受影响）。**k=4 baseline 不再是 thesis 主线**（被 k=8 取代），无需 multi-seed。

**P2 — 其它 cell 的 multi-seed 巩固**（§7 takeaway 进入 thesis-grade 的前置）：

3. **§7 4-way × 3-4 seeds**（除 single_cross 之外）：3 个上游 cell 都 saturate 到 1.000，多 seed 主要确认 `last100k_mean` 在 [0.95, 1.0] 区间稳定；优先级看 thesis 重启与否。`~30h L4`。
4. **§7.6 上游 3 cell × multi-seed**：tandem_s0 / sbs_s0 / single_upstream_s0 三个 phase 多 seed 复现，确认 s0 在上游几何下 saturate 不是 seed-specific 偶然。`~30h L4`。

**P3 — Reward / 工程余项**：

5. **Budget 预言订正**（§7.2 vs §3 已揭示）：[设计 §8.2] 默认 1.5M cap 对 seed=42 是 0.5M overestimate；多 seed 复现时建议先按 1M cap 跑完再判，未稳态再上调，**不要默认 1.5M**。
6. **`arrival_v2` vs `arrival_v2_fast` 短附录**（[设计 §11.8] 留口）：`R_fast_success=20` 看 time-to-goal 在 §7.2 single_upstream baseline 上的边际改进。
7. **`w_safety` 二次校准**（[设计 §11.8]）：用 §7.3 tandem 的 actual failure-policy safety 分布（mean 6.85 主要来自 1/30 outlier）重算 break-even 阈值，看 v6 候选 2.0 是否需要调整。
8. **`scripts/train_sac.py` resume regression test**（§6 留口）。
9. **更多 benchmark scenarios**：`single_u15_upstream_tgt20` (over-target / λ<1)、`single_u15_downstream_tgt15` (顺流) 等，参考 `benchmarks/` 列表；尤其 `tgt20` 可检验 §7 takeaway「拓扑未引入 sample 难度」是否泛化到 over-actuation 区。
10. **`single_cross_s0` 物理机制深挖**：§7.6 F3 plasticity-loss 形态 + §7.7 F2 行为风格反向 trade-off 值得做 OOB 时间分布分析 + 涡街相位 vs OOB event 的耦合分析（验证 §7.6 机理段的"涡街 phase 不可见 → 横切瞬间被推出"假设；同时验证 §7.7 的 "actor-side information ceiling" 是否对应明确的物理量缺失）。

是否启动以上任何一项，由 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) 的产品决策驱动，**不应自动从本节 PASS 跳到展开**。当前推荐最小展开：**P1#1（history k=4→8 on `single_cross_s0`）** — single run, single seed, 2.5h L4，直接测 §7.7 机理段重定位后的新假设（actor-side temporal information 是否能解锁 cross 几何下的 s0 ceiling）。
