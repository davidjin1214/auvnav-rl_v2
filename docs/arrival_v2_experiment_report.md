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

| Run | Branch | Benchmark | Probe | Seeds | Total steps | Status | 5/5 Gate |
|---|---|---|---|---|---:|---|---|
| §2 cross_u10 regression | `codex-arrival-v2-prototype` | `single_u10_cross_tgt15` | s0 / k4 | 46 | 600k → **1M** | DONE | **PASS** |
| §3 P1 v6 重跑 | `codex-arrival-v2-prototype` | `single_u15_upstream_tgt15` | s1 / k4 | 46 | **1.5M** | DONE | **PASS** |
| §7.1 tandem topology | `codex-arrival-v2-prototype` | `tandem_u15_upstream_tgt15` | s1 / k4 | 42 | 1M (initial) | IN FLIGHT | TBD |
| §7.2 sbs topology | `codex-arrival-v2-prototype` | `sbs_u15_upstream_tgt15` | s1 / k4 | 42 | 1M (initial) | IN FLIGHT | TBD |

**TL;DR (已完成的两组)**：
- arrival_v2 在「最弱 sensor + vanilla SAC + 已知可学」的 cross_u10 保守回归中没有 break，1M 时 final=1.000 / OOB=0.000。
- arrival_v2 在 §2 P1 证据复核所记录的 `efficiency_v2` collapse 现场（`single_u15_upstream_tgt15`）实现彻底修复：旧版 success → 0，新版 success = 1.000，30/30 全 goal。

---

## 1. 实施跟踪

- arrival_v2 8 参数完整版按 [设计 §5.1](online_sac_reward_redesign.md) v6 spec 在 commit `813096e`（2026-05-07）落地进 `auv_nav/reward.py`，与 `arrival_v2_simple`（commit `bd37412`）非同一物。
- [设计 §8.1] Gate A pure-formula validator（`scripts/validate_arrival_v2_candidate`）通过：default `w_safety=2.0` discounted unsafe-shortcut + terminal dominance + OOB ordering 全部成立；`w_safety=0.5` 在 discounted unsafe-shortcut 上被明确判失败（与 v6 设计预言一致）。
- 隔离 prototype 分支 `codex-arrival-v2-prototype` 跑了两组实验：
  - **§2 cross_u10 behavior regression**：先按计划跑 600k，未通过 last100k_mean gate（曲线仍在上升），延到 1M 后 5/5 gate 全过。
  - **§3 P1 v6 重跑**：从原计划 1M 提到 1.5M（s1 + upstream + 12-D 比 cross + s0 + 10-D 难，留缓冲）。
- 实施期间发现并修了一个 SAC trainer resume 路径上的 silent bug，参见 §6。
- 闭环 commit：`f179c5b`（含两个 notebook 的归档 banner、设计 §12 的实测填充、CLAUDE.md / docs 的索引更新）。

**复现路径**：
- cross_u10 + P1 v6：[`notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb`](../notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb)
- 600k cross_u10 prototype 归档：[`notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb`](../notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb)
- tandem + sbs（in flight）：[`notebooks/sac_arrival_v2_tandem_sbs_validation.ipynb`](../notebooks/sac_arrival_v2_tandem_sbs_validation.ipynb)

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

---

## 5. 时间预算实测 vs [设计 §8.2]

| 任务 | §8.2 预算 (L4) | 实测 (L4) | 说明 |
|---|---:|---:|---|
| §2 cross_u10 regression | 1.5h（600k） | ~3h（600k 首跑 + 续 400k） | 600k 未稳态，按收敛形状续到 1M |
| §3 P1 v6 重跑 | 2.5h（1M） | ~5h（1.5M） | 上行 budget cap，留缓冲 |
| **总 wallclock** | ~4h | ~8h | × 2 |

差因都是「按收敛形状现场判断 budget」而非「先固定 cap」。这两次差距没有触发 [设计 §6] / [设计 §11] 的任何风险，但记录在此供 [设计 §8.2] 下次 review 参考。

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

## 7. 拓扑泛化验证（IN FLIGHT，待回填）

**动机**：§2 / §3 已经在两个 single-cylinder benchmark 上 PASS，覆盖 cross / upstream geometry × U=1.0 / U=1.5。但**双柱拓扑**（co-shedding tandem、Coandă-deflected sbs）的物理机制与单柱不同，是 arrival_v2 是否泛化到更复杂尾流的关键测试。

**实验设计**：vanilla SAC, `s1_k4`, seed=42, num_envs=6, total_steps=1M（initial budget；不收敛可重跑续训）。Gate 五条与 [设计 §8.3] 同口径。**两个 phase 互不门控**（不同物理机制，独立判定）。

**复现路径**：[`notebooks/sac_arrival_v2_tandem_sbs_validation.ipynb`](../notebooks/sac_arrival_v2_tandem_sbs_validation.ipynb)

### 7.1 Tandem 双柱串列（`tandem_u15_upstream_tgt15`）

**Benchmark**：双柱串列 G/D=3.5，U∞=1.5 m/s，target=1.5 m/s，upstream geometry，Re=250。主导现象 = co-shedding 长尾涡街。λ=U∞/V_max=1.0（critical under-actuation）。

**Flow**：`wake_data/wake_tandem_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

**结果**：_TBD（结果回填后填表）_

| Gate | 阈值 | 实测 | 结论 |
|---|---|---|---|
| final_success_rate | ≥ 0.85 | TBD | TBD |
| last100k_mean / peak | ≥ 0.90 | TBD | TBD |
| OOB rate | ≤ 0.10 | TBD | TBD |
| `include_episode_context_obs` | True | TBD | TBD |
| `timeout_bootstrap_semantics` | terminal | TBD | TBD |

### 7.2 SBS 双柱并列（`sbs_u15_upstream_tgt15`）

**Benchmark**：双柱并列 G/D=3.5，U∞=1.5 m/s，target=1.5 m/s，upstream geometry，Re=250。主导现象 = Coandă 偏转 + 不对称双尾。λ=1.0。

**Flow**：`wake_data/wake_sbs_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

**结果**：_TBD（结果回填后填表）_

| Gate | 阈值 | 实测 | 结论 |
|---|---|---|---|
| final_success_rate | ≥ 0.85 | TBD | TBD |
| last100k_mean / peak | ≥ 0.90 | TBD | TBD |
| OOB rate | ≤ 0.10 | TBD | TBD |
| `include_episode_context_obs` | True | TBD | TBD |
| `timeout_bootstrap_semantics` | terminal | TBD | TBD |

### 7.3 Topology 横向对照（待回填）

完成后补一张「single (P1 v6) / tandem / sbs」三栏对照表（同 U=1.5 / target=1.5 / upstream / s1，唯一变量 = topology），看 success / safety_cost / eval_time / path_efficiency 的拓扑敏感性。

---

## 8. 后续可选工作

prototype + topology 验证仅证明 arrival_v2「在 online 单 seed 上 work」。若 thesis 重启或 offline 线决定升级 reward preset，可基于本报告做下一步：

1. **多 seed 复现**（thesis statistics）：cross 1M × 4 seeds ≈ 4h L4；P1 v6 1.5M × 4 seeds ≈ 24h L4；tandem / sbs × 4 seeds 类似预算。
2. **`arrival_v2` vs `arrival_v2_fast` 短附录**（[设计 §11.8] 留口）：`R_fast_success=20` 看 time-to-goal 在 P1 v6 上的边际改进。
3. **`w_safety` 二次校准**（[设计 §11.8]）：用 P1 v6 的 actual failure-policy safety 分布重算 break-even 阈值，看 v6 候选 2.0 是否需要降到 1.5 或保持。
4. **AsymCritic × arrival_v2 联动**（[设计 §11.6] 留口）：privileged hull-integral flow 与新 reward 的交互在本 prototype 未测；若 online 线重启，可在 §3 P1 v6 baseline 上加一次 `--use-asymmetric-critic` ablation。
5. **`scripts/train_sac.py` resume regression test**（§6 留口）。
6. **更多 benchmark scenarios**：`single_u15_upstream_tgt20` (over-target / λ<1)、`single_u15_cross_tgt15` (mid-cross)、`single_u15_downstream_tgt15` 等，参考 `benchmarks/` 列表。

是否启动以上任何一项，由 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) 的产品决策驱动，**不应自动从本节 PASS 跳到展开**。
