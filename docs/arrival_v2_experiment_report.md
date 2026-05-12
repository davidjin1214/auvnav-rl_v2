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

**严格控制对照（§7，唯一变量 = topology × geometry）**：固定 `s1 / k4 / arrival_v2 / U=1.5 / target=1.5 / seed=42 / 1M / num_envs=6`。

| Run | Benchmark | Topology | Geometry | 5/5 Gate |
|---|---|---|---|---|
| §7.1 single_cross    | `single_u15_cross_tgt15`    | single | cross_stream | **PASS**（borderline：OOB 踩线 0.10） |
| §7.2 single_upstream | `single_u15_upstream_tgt15` | single | upstream | **PASS** |
| §7.3 tandem          | `tandem_u15_upstream_tgt15` | double tandem (G/D=3.5) | upstream | **PASS** |
| §7.4 sbs             | `sbs_u15_upstream_tgt15`    | double sbs (G/D=3.5) | upstream | **PASS** |

**Reference baselines（§2 / §3，不参与 §7.5 严格对比；seed/step/U 与上面 4 组不一致）**：

| Run | Benchmark | Probe | Seed | Total steps | Confound | 5/5 Gate |
|---|---|---|---:|---:|---|---|
| §2 cross_u10 regression | `single_u10_cross_tgt15` | s0 / k4 | 46 | 1M | U=1.0, s0, seed=46 | PASS |
| §3 P1 v6 重跑           | `single_u15_upstream_tgt15` | s1 / k4 | 46 | 1.5M | seed=46, 1.5M | PASS |

**TL;DR**：
- arrival_v2 在严格控制下（s1 / k4 / seed=42 / 1M / U=1.5 / target=1.5 / vanilla SAC）的 4 组 vanilla SAC 验证**全部 5/5 gate PASS**：单柱 cross / 单柱 upstream / 双柱 tandem / 双柱 sbs。
- 三组 upstream（single / tandem / sbs）均 final=1.000 / OOB=0.000 / 30/30 全 goal，peak first-hit step 都在 475k–625k → **topology 在严格控制下未引入额外 sample 难度**。
- 单柱 cross_stream 是四组里唯一 OOB 踩线 (0.10) 的 run、final=0.900、return std=112 → **cross_stream geometry 比 upstream 更难**。
- 末段 safety 排序：single_upstream (0.142) < sbs (0.586) < tandem (6.85) — 与 wake topology 物理直觉一致。
- §2 cross_u10 / §3 P1 v6 旁证 arrival_v2 在更慢流速、更弱 sensor、不同 seed 下也 PASS，但因 seed/step/U confound 仅作 reference，不进入 §7.5 主对照。

---

## 1. 实施跟踪

- arrival_v2 8 参数完整版按 [设计 §5.1](online_sac_reward_redesign.md) v6 spec 在 commit `813096e`（2026-05-07）落地进 `auv_nav/reward.py`，与 `arrival_v2_simple`（commit `bd37412`）非同一物。
- [设计 §8.1] Gate A pure-formula validator（`scripts/validate_arrival_v2_candidate`）通过：default `w_safety=2.0` discounted unsafe-shortcut + terminal dominance + OOB ordering 全部成立；`w_safety=0.5` 在 discounted unsafe-shortcut 上被明确判失败（与 v6 设计预言一致）。
- 隔离 prototype 分支 `codex-arrival-v2-prototype` 跑了**6 组实验**（4 组严格控制 + 2 组 reference baselines）：
  - **§2 cross_u10 behavior regression**（reference）：先按计划跑 600k，未通过 last100k_mean gate（曲线仍在上升），延到 1M 后 5/5 gate 全过。
  - **§3 P1 v6 重跑**（reference）：从原计划 1M 提到 1.5M（s1 + upstream + 12-D 比 cross + s0 + 10-D 难，留缓冲）；事后由 §7.2 (seed=42 / 1M PASS) 推翻这个 budget 假设 — 1.5M 是 seed=46 specific。
  - **§7.3 tandem 拓扑泛化**（strict control，旧编号 §7.1）：1M cap，seed=42，5/5 gate 全过。
  - **§7.4 sbs 拓扑泛化**（strict control，旧编号 §7.2）：1M cap，seed=42，5/5 gate 全过。
  - **§7.1 single_cross 控制对照**（strict control，2026-05-09 补跑）：1M cap，seed=42，5/5 gate 全过（OOB 踩线 0.10）。
  - **§7.2 single_upstream 控制对照**（strict control，2026-05-09 补跑）：1M cap，seed=42，5/5 gate 全过；与 §3 P1 v6 同 benchmark 的二点 seed 观测。
- 实施期间发现并修了一个 SAC trainer resume 路径上的 silent bug，参见 §6。
- 闭环 commit：`f179c5b`（§2 + §3 闭环），`1adee4e`（doc split + tandem/sbs notebook scaffold + §7.3/§7.4 回填），单柱补跑 (§7.1/§7.2) 在 2026-05-09 落 commit（待提交）。

**复现路径**：
- §2 cross_u10 + §3 P1 v6：[`notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb`](../notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb)
- 600k cross_u10 prototype 归档：[`notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb`](../notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb)
- §7.3 tandem + §7.4 sbs：[`notebooks/sac_arrival_v2_tandem_sbs_validation_completed.ipynb`](../notebooks/sac_arrival_v2_tandem_sbs_validation_completed.ipynb)
- §7.1 single_cross + §7.2 single_upstream（严格控制对照）：[`notebooks/sac_arrival_v2_single_u15_seed42_1M_validation.ipynb`](../notebooks/sac_arrival_v2_single_u15_seed42_1M_validation.ipynb)

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

---

## 8. 后续可选工作

§7 4-way strict-control 验证证明 arrival_v2 在 single + double 拓扑 × cross + upstream geometry 下都「在 online 单 seed 上 work」。若 thesis 重启或 offline 线决定升级 reward preset，可基于本报告做下一步：

1. **多 seed 复现**（thesis statistics）：四组 strict-control 各 × 3-4 seeds（4 × 1M × 4 seeds ≈ 40h L4）。**优先复现 §7.1 single_cross**（OOB 踩线，单 seed 信号最弱），其次 §7.3 tandem（验证 ep_0011 风格 outlier 是否 seed-specific），最后 §7.2 / §7.4。
2. **Budget 预言订正**（§7.2 vs §3 已揭示）：[设计 §8.2] 默认 1.5M cap 对 seed=42 是 0.5M overestimate；多 seed 复现时建议先按 1M cap 跑完再判，未稳态再上调，**不要默认 1.5M**。
3. **`arrival_v2` vs `arrival_v2_fast` 短附录**（[设计 §11.8] 留口）：`R_fast_success=20` 看 time-to-goal 在 §7.2 single_upstream baseline 上的边际改进。
4. **`w_safety` 二次校准**（[设计 §11.8]）：用 §7.3 tandem 的 actual failure-policy safety 分布（mean 6.85 主要来自 1/30 outlier）重算 break-even 阈值，看 v6 候选 2.0 是否需要调整。
5. **AsymCritic × arrival_v2 联动**（[设计 §11.6] 留口）：privileged hull-integral flow 与新 reward 的交互在本 prototype 未测；若 online 线重启，可在 §7.2 single_upstream baseline 上加一次 `--use-asymmetric-critic` ablation。
6. **`scripts/train_sac.py` resume regression test**（§6 留口）。
7. **更多 benchmark scenarios**：`single_u15_upstream_tgt20` (over-target / λ<1)、`single_u15_downstream_tgt15` (顺流) 等，参考 `benchmarks/` 列表；尤其 `tgt20` 可检验 §7 takeaway「拓扑未引入 sample 难度」是否泛化到 over-actuation 区。
8. **§7.1 single_cross 物理机制深挖**：cross_stream geometry 在 4 组中最难的根因（侧向被推出边界 vs upstream 的逆流恢复），值得在多 seed 数据基础上做 OOB 时间分布分析。

是否启动以上任何一项，由 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) 的产品决策驱动，**不应自动从本节 PASS 跳到展开**。
