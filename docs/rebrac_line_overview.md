# ReBRAC 线总览（贯通 TD3+BC 前置 ↔ ReBRAC 主线 ↔ FQL succession）

> 文档版本：2026-05-24 rev.2
> 作用：以 **ReBRAC 为中心**，把三段彼此独立成文的工作串成一条线——(0) 前置 baseline **TD3+BC**、(1) 主线 **ReBRAC**（paper 1）、(2) head-to-head 支线 **FQL succession**（paper 2，ReBRAC 在其中作对照对象）。
> 与姊妹文档的分工：
> - [`offline_rl_line_summary.md`](offline_rl_line_summary.md)：整条 offline 线（含 AUVHamNODE 等）的全景入口，**纯链接、不持数字**。
> - 本文：只聚焦 ReBRAC 这一条算法脉络，**自包含关键 headline 数字**，每个数字标注 ground-truth 出处（`文件 §节`）。任何冲突以源文档为准。
> - 数字 ground truth：[`rebrac_experiment_report.md`](rebrac_experiment_report.md)（paper 1）、[`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9 + [`fql_succession_p2_results.md`](fql_succession_p2_results.md)（paper 2）、[`td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md)（baseline）。
> - rev.2（2026-05-24）：新增 **§5 数据集与状态空间速查（SAC collector 设计参考基线）**；其后各节顺延（β1 跨线 reconciliation → §6，文件索引 → §7，一句话总结 → §8）。state-space / dataset 真值出处：[`auv_nav/env.py`](../auv_nav/env.py) `PlanarRemusEnv._build_observation` + dataset `metadata.json` / `transitions.npz`。

---

## 0. 一句话与时间轴

> **TD3+BC** 证明了 offline RL 在本任务上有价值、并拆清了两个瓶颈（数据支持集结构 + deployable critic 信息瓶颈）；**ReBRAC** 在这两个瓶颈上做到 paper-ready 的四条 finding，核心是「deployable-only 离线策略追平 privileged-critic、优于 TD3+BC 23–32pp」；**FQL succession** 用一个诚实负面回头确证 ReBRAC——正确调参（β1=1.0）的 ReBRAC 在干净+噪声双轴全面压制更复杂的 FQL，机制归因到 **BC-anchor 目标质量**。

| 段 | 时间窗口 | 算法 | 状态 | 头部入口 |
|---|---|---|---|---|
| **0 · 前置 baseline** | 2026-Q1 → ~2026-04 | TD3+BC（D4RL 经典 single-BC） | ✅ 已归档（不再扩展） | [`td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) |
| **1 · 主线（paper 1）** | ~2026-04 → 2026-05-07 | ReBRAC（dual-BC + critic-side penalty） | ✅ paper-ready 4/4 closed (rev.8) | [`rebrac_mainline_review.md`](rebrac_mainline_review.md) |
| **1.5 · 泛化广验** | 2026-05-04 → 2026-05-19 | ReBRAC 跨轴 generality | v1 ⚠ SUPERSEDED 2026-05-18；v2 ✅ PASS 2026-05-19 | [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) |
| **2 · 支线（paper 2）** | 2026-05-19 → 2026-05-24 | FQL vs ReBRAC（ReBRAC 作对照对象） | ✅ NEGATIVE 闭环 2026-05-23；P4 写作索引已建 | [`fql_succession_p2_results.md`](fql_succession_p2_results.md) |

**贯穿全线的 sim2real 命题**：在部署受限的单点传感器（`s0` = DVL water-track）下，offline RL 能否不依赖任何 privileged 仿真信息、把性能逼近 online teacher？

---

## 1. 前置工作：TD3+BC baseline（段 0，已归档）

**角色**：ReBRAC 的对照 baseline，不是论文核心算法。核心贡献是把「数据越多越差」从协议伪象**纠正为真实现象**，并拆清两个瓶颈，为 ReBRAC 创造明确问题设定。

**核心实验**（Phase 0b → 0c，漏斗式 Stage A/B/C）：修正训练预算定义（固定 steps → epoch-aligned）后，确认 **1000 episodes 最优、而非 2000**。

| crosscomp 数据规模 | TD3+BC success | 纯 BC success |
|---|---:|---:|
| 500 | 0.592 | 0.524 |
| **1000** | **0.672** | 0.588 |
| 2000 | 0.596 | 0.534 |

> 数字出处：[`td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md)；crosscomp-1000/2000 与 paper 1 主对照表一致（[`rebrac_mainline_review.md`](rebrac_mainline_review.md) §1.5：0.672 ± 0.045 / 0.596 ± 0.036）。

**两个被拆开的瓶颈**：
1. **数据支持集结构问题**：「2000 < 1000」退化在纯 BC 下同样成立 → 不是 TD3+BC 特有失败，而是大 deterministic 数据的支持集结构所致（noisy-support 诊断里加入支持后 2000 BC 可回升）。
2. **deployable critic 信息瓶颈**：worldcomp teacher-gap 上 deployable 最优退回 α≈0（纯 BC，0.858），privileged-critic 能关掉约 48.5% gap（0.922；teacher 上界 0.990）。

**为何转向 ReBRAC**：TD3+BC 的 Q 修正在大数据 / 弱观测下已无稳定收益，最优点全面回落到低 α 或纯 BC。需要更强的 in-sample 正则 + critic 耦合 → **dual-BC penalty 的 ReBRAC**。

---

## 2. 主线：ReBRAC（段 1，paper 1，✅ paper-ready 4/4 closed rev.8）

**实现口径**：**Q-normalized dual-penalty TD3+BC 变体（ReBRAC-Q）**——actor loss 把 deterministic policy gradient 除以 `|Q|.detach()`，与原始 ReBRAC 形式不同，论文 method section 已显式声明（`β1 ≈ 1/α_TD3BC`，β1=4.0 ↔ α≈0.125）。详见 [`rebrac_method_section_draft.md`](rebrac_method_section_draft.md)。

**Canonical 协议**：benchmark `single_u10_cross_tgt15`、task `cross_stream`、target 1.5、objective `efficiency_v2`、probe `s0`、history `h4`、flow `wake_v8 Re150 Ti5pct`；seeds 42–46；约 65 个新训练 run（Stage A→B0→B→C→D→E→F）。

### 2.1 四条论文级 finding（数字出处见各行）

| # | finding | 关键数字 | 出处 |
|---|---|---|---|
| **(i)** | crosscomp 上 ReBRAC 显著优于 TD3+BC，并翻转「2000<1000」退化 | crosscomp-1000 **0.902 ± 0.021**（+23.0pp）；crosscomp-2000 **0.918 ± 0.030**（+32.2pp），std 不增反降 | report §7；review §1.5 |
| **(ii)** | deployable obs 追平 privileged-critic（**最强 sim2real narrative**） | worldcomp deployable **0.928 ± 0.077** vs TD3+BC privileged 0.922（Welch p=0.92 持平）；ReBRAC privileged 5-seed 0.9340 ± 0.0261（Δ=+0.6pp，不抬 mean，仅 seed 44 救回 +12pp） | report §7.10/§7.12/§7.16 |
| **(iii)** | dual penalty 必需 | β2=0 时 mean 仅掉 2–5pp，但 `mean_target_q` 跨 dataset 漂 **+46% / +98%**（Q 高估），seed 44 崩 −17pp | report §7.13/§7.14 |
| **(iv)** | critic LayerNorm 与 dual penalty 是两个独立必要组件 | LN-off 让 mean 退化 **−16.2pp**（远超 β2=0 的 −2.4pp）且退化方向不同 | report §7.15 |

**finalist 配置**：`(β1=4.0, β2=2.0)`，两个 dataset 共用（ep2000 backup = `(β1=4.0, β2=1.0)`）。

### 2.2 「β1=4.0 winner」的真实选择逻辑（理解段 6 必读）

report §6.4 明确：在 crosscomp-1000 上，winner `(β1=4.0, β2=2.0)` 的 **peak 并不是全表最高**——`(β1=1.0, β2=2.0)` seed 42、`(β1=2.0, β2=2.0)` seed 43 都曾到 0.950。**winner 胜出完全来自把 seed 44 从 0.475~0.500 救回 0.850**，是 "robustness winner" 而非 "peak winner"。对照：`(β1=1.0, β2=2.0)` = **0.742 ± 0.198**（高 std，seed 44 崩盘）。

> 这一点是段 6（β1 跨线 reconciliation）的关键前提：原线选 β1=4.0 是 **std-driven（seed-44-robustness）** 决策，不是 mean-driven。

---

## 3. 泛化广验（段 1.5）

- **v1（`efficiency_v2`，三轴 8 spoke × 5-seed parity，~30h L4）**：8 spoke 中只有 B1（s1 传感器）clean positive，其余 7 spoke 欠功效（A1 p≈0.11 / A2 mode-collapse 仅假设 / C1 task-fundamental floor 钉在 0.195–0.225）。**2026-05-18 整套 SUPERSEDED**——`efficiency_v2` 被 [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) 证有 OOB-suicide failure mode（reward 失配），且 online §7.6 已产出更强 sensor envelope finding。v1 archive 保留、不重跑、不进 paper。
- **v2（`arrival_v2`，cross-only spotlight，✅ PASS；首轮 2-seed 4-run 2026-05-19，2026-07-12 补齐第三种子）**：
  - **N0**（crosscomp / sub-critical Re150）：**0.878 ± 0.051 → HOLDS**（3 seed {42, 0, 43}，vs anchor 仅 −2.4pp）。⚠ 原写 2-seed 的 `0.850 ± 0.024 / −5.2pp` 并读作「同向退化」，该读法已随第三种子（0.933，反向 +3.1pp）撤销，见 v2 report §2.4
  - **N2'**（privileged / critical Re250）：**0.000 ± 0.000 → STRONG_NEGATIVE**（3 seed 合计 **0/90**，rule-of-three 95% 上界 ≈ 0.033），甚至跌破 online catastrophic floor 10pp
  - **N2' asym-critic 消融**（唯一变量 `--use-asymmetric-critic`，v2 report §4.5，2026-05-27）：把完美 hull-integral flow 全程喂给 critic，s0 actor success 仍 **0.000**（三种子同样 0/90）→ verdict `ACTOR_FUNDAMENTAL_CONFIRMED`，**天花板不在 critic 价值估计**。这是下一行 paper headline 里"actor-fundamental"三个字的实证来源，2026-08-17 补入本节
  - **paper headline**：即便 oracle teacher（privileged 70% 直接成功）提供示范，critical regime 下 `s0`-conditioned BC 也传不了 hull-integral flow 知识 → **deployable s0 传感器下的 actor-fundamental partial-observability ceiling**（作 paper §experiments 的 deployability boundary map）。

> 数字出处：[`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)（v2 ground truth）；[`rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md)（v1 archive）。

---

## 4. ReBRAC 在 FQL succession 支线里的角色（段 2，paper 2，✅ NEGATIVE 闭环）

**FQL succession** 是新开的 head-to-head 研究线（FQL = Flow Q-Learning，flow-matching teacher + 1-step distill），问：**表达力更强的 flow-matching 先验是否在 sub-optimal AND multi-modal 数据上系统超过 ReBRAC（dual-BC）？** 此处 **ReBRAC 是被对比的 baseline / succession 前序对象**。

**Gate B（P0+P1 出口，2026-05-20）**：3/4 PASS + c4 marginal-FAIL（c4 失败 seed-driven、非 FQL 特异）→ 带 caveat 进 P2。

**P2 主结论：核心假设证伪 → 诚实负面 + 机制发现（B+A locked）**：

- **2×2 矩阵（modality × noise）**：discriminator 是**噪声**而非 modality——clean-multi cell（E-multi）NULL（δ=−0.020），多模态被排除。
- **机制三连**（数字出处 [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9）：

  | 步骤 | 内容 | 结果 |
  |---|---|---:|
  | Q1 | ReBRAC critic-side penalty 是否约束？（β2=2.0→0） | 仅 +0.010 → **非约束** |
  | **Q1b** | actor-side β1 4.0→1.0（noisy M-uni-noise，σ=0.5） | ReBRAC **0.705 → 0.940**，**+23.5pp 反超 FQL（0.910）**，t=4.98 SIG |
  | Q1c | β1=1.0 在 clean 是否有代价？ | clean β1=1.0 = **0.910 > FQL 0.858** → 无 trade-off |

- **C-1 RESCUE-FAIL**：给 FQL 自己的锚强度旋钮 `distill_alpha_bc` 做 log-spaced 公平复赛扫描，clean 最高仍 **0.858**（差杆 −5.2pp）→ 拯救失败，关闭「对称调参」caveat。
- **整体结论**：**单一固定 ReBRAC β1=1.0 在 clean+noisy 双轴 dominate FQL，worst-case-over-noise 0.910 > FQL 0.858**。原先看到的「FQL 赢 +20.5pp」其实是 **ReBRAC β1=4.0 mis-tuning 的 artifact**。
- **可发表的 mechanism finding**：offline RL 对 action noise 的鲁棒性由 **BC anchor 目标质量**决定，与算法族无关；最优锚强度随目标噪声翻转。
- **跨-benchmark 探测**（更难的 `single_u15_cross`，U=1.5/Re250）：实测 **FLOOR**（s0 observability floor，clean privileged 0.719 / noisy 0.098）→ 比较在该 regime 未定义，作 scope caveat（results §6.5），不弱化 u10_cross 上的主结果。

---

## 5. 数据集与状态空间速查（SAC collector 设计参考基线）

> 本节是给后续 **SAC collector**（用 arrival_v2 SAC checkpoint 收 D4RL 风格数据）的 schema 基线：任何新 collector 产出的 dataset 必须与下表 obs / action / privileged schema 对齐，才能 drop-in 进现有 offline pipeline（ReBRAC / FQL / AsymCritic）。SAC checkpoint 盘点与实施要点见 [`arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md)。
> 真值出处：dataset `metadata.json` + `transitions.npz`（`offline_data/` gitignored，完整集在 Drive）；状态空间 = [`auv_nav/env.py`](../auv_nav/env.py) `PlanarRemusEnv._build_observation` + [`environment_design.md`](environment_design.md)。

### 5.1 数据集命名约定与当前主用集

命名约定：`<collector>_<sensor>_<history>_<reward>_<flow_regime>_<geometry>_fixdone_ep<N>`。collector = [`auv_nav/baselines.py`](../auv_nav/baselines.py) 的 4 类非学习 baseline：**goalseek / crosscomp（CrossCurrentCompensation）/ worldcomp（WorldFrameCurrentCompensation）/ privileged（PrivilegedCorridor）**。

⚠ **reward 已从 `efficiency_v2` 迁到 `arrival_v2`** —— 当前主用 = arrival_v2 那批；efficiency_v2 属已收口的 paper 1 / TD3+BC 历史（两者 obs 维度也不同，见 §5.2）。

| 用途 / 阶段 | dataset | collector | reward | flow regime | obs_dim | 关键统计 |
|---|---|---|---|---|---:|---|
| paper 1 / TD3+BC（历史） | `crosscomp_s0_h4_efficiency_v2_re150_u10cross_ep1000`/`_ep2000`；`worldcomp-1000` | crosscomp / worldcomp | efficiency_v2 | Re150 / u10 cross | **40** | finalist 数据源 |
| 广验 v2 · N0（sub-critical） | `crosscomp_s0_h4_arrival_v2_re150_u10cross_ep1000` | crosscomp | arrival_v2 | Re150 / u10 | **48** | HOLDS 0.878 ± 0.051（3 seed）|
| 广验 v2 · N2'（critical） | `privileged_s0_h4_arrival_v2_re250_u15cross_ep1000` | privileged | arrival_v2 | Re250 / u15 | **48** | STRONG_NEGATIVE |
| FQL · E-uni（clean anchor） | `privileged_s0_h4_arrival_v2_re150_u10cross_ep1000` | privileged | arrival_v2 | Re150 / u10 | 48 | σ=0；succ 0.985；86,685 trans |
| FQL · M-uni-noise | `fql_succession/m_uni_noise_eps0p5_1000` | privileged | arrival_v2 | Re150 / u10 | 48 | **σ=0.5**；succ 0.632（unimodal+噪声）|
| FQL · E-multi | `fql_succession/e_multi_50priv_50goal_clean_1000` | 50% priv + 50% goalseek | arrival_v2 | Re150 / u10 | 48 | clean 多模态 |
| FQL · M-multi-mix | `fql_succession/m_multi_mix_50priv_50goal_1000` | mix + 噪声 | arrival_v2 | Re150 / u10 | 48 | 噪声多模态 |

共性轴（全部一致）：`s0`（部署主轴，DVL water-track 单点）、`h4`、`cross_stream`、`target_speed=1.5`、`ep=1000`。差异只在 collector（数据质量）、action noise σ、flow regime（Re150/u10 sub-critical ↔ Re250/u15 critical）。

### 5.2 状态空间定义

**actor 观测（单步）—— `env.py` `_build_observation`，归一化后每通道 O(1)**：

| 切片 | 通道 | 内容 |
|---|---|---|
| own `[0:5]` | [0] u/speed_scale；[1] v/speed_scale；[2] r/yaw_rate_scale；[3] cos ψ；[4] sin ψ | 本体速度 + 艏向 |
| goal `[5:8]` | [5] goal_body_x/goal_scale；[6] goal_body_y/goal_scale；[7] distance/dist_scale | 目标 body-frame 位置 + 距离 |
| probe `[8:8+2n]` | 每 probe `(u,v)` body-frame（velocity 模式） | s0=1 / s1=2 / s2=4 probe |
| context `[..+2]` | elapsed_frac（已耗时间占比）+ 归一化初始距离 | **仅 arrival_v2**（`env.py:377`：reward ∈ ARRIVAL_V2_OBJECTIVES 时自动开启）|

- 8 base（5 own + 3 goal）+ probe：s0=**10-D** / s1=12-D / s2=16-D（不含 context）。
- **arrival_v2 多 2 个 episode-context 通道** → s0 单步 = 12-D。
- 历史堆叠 `h4`（`ObservationHistoryWrapper`）= 单步 × 4：
  - efficiency_v2 s0/h4 = (8+2)×4 = **40-D**
  - arrival_v2 s0/h4 = (8+2+2)×4 = **48-D**（已实测确认）
- **action**：2-D 连续（heading command, speed command）。
- **privileged_obs**（dim=2，**不堆叠**）：body-frame `[u_eq, v_eq]`——`EquivalentCurrentModel` 的 hull-integral 等效流（真正驱动动力学的有效流，区别于单点 probe 采样）。只供 `AsymmetricQNetwork` critic / ReBRAC privileged 轨道；npz 以 `privileged_obs / next_privileged_obs` 存。

**npz 存储列**：`obs, next_obs`（堆叠 actor obs）、`actions, next_actions`（next_action 供 TD3-style target）、`rewards, costs`、`dones / terminateds / truncateds`、`terminal_reason_codes, behavior_policy_codes`、`privileged_obs, next_privileged_obs`。

### 5.3 对 SAC collector 设计的含义

要让 arrival_v2 SAC checkpoint 收的数据 drop-in 兼容现有 ReBRAC / FQL / AsymCritic pipeline：

1. **obs schema 必须 = arrival_v2 / s0 / h4 → 48-D**（含 2 个 episode-context 通道 + h4 堆叠），否则 obs_normalizer 与网络输入维度不匹配；
2. **必须同时记录 `privileged_obs`（2-D，从 env `info` 取）**，否则 AsymCritic / privileged 轨道无法复用；
3. 应覆盖与现有 baseline collector 同样的 **(数据质量 × noise σ × flow regime)** 轴，才能与 crosscomp / privileged 数据做同口径对照。SAC 提供的是 **D4RL 通行做法里的 RL-trained behavior policy** 第四类数据源（区别于 4 类 rule-based baseline）；
4. checkpoint 盘点、推荐子集、trigger 条件见 [`arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md)（当前定位为 revision / future-work tier 的 pre-implementation spec，trigger 满足前不写 adapter 代码）。

---

## 6. ⭐ 跨线关键问题：为什么最优 β1 从 4.0 变成 1.0？原 ReBRAC 实验错了吗？

**结论先行：没有矛盾，原实验没有错。** 原线选 β1=4.0 是其自身协议下合理的 **std-driven 局部最优**；FQL 线在**四个轴同时不同**、且**排除了驱动 β1=4.0 选择的那颗 seed（44）**的条件下得到 β1=1.0 更优。两边都对，paper 1 finding (i)–(iv) 全部不受影响。

### 5.1 两边数据点对照

| 实验 | dataset | reward | noise | seed pool | β1=4.0/β2=2.0 | β1=1.0/β2=2.0 |
|---|---|---|---|---|---:|---:|
| paper 1 Stage B (n=3, test=40) | crosscomp-1000 | efficiency_v2 | deterministic + 小噪 | 42/43/**44** | **0.883 ± 0.031** ← winner | 0.742 ± 0.198 |
| paper 1 Stage C (n=5, test=100) | crosscomp-1000 | efficiency_v2 | deterministic + 小噪 | 42–46 | **0.902 ± 0.021** (finalist) | — (未升级) |
| FQL P2 E-uni clean (test=100) | privileged-1000 | arrival_v2 | σ=0 | 42/**0** | 0.885 | **0.910** |
| FQL P2 M-uni-noise (test=100) | privileged-1000 | arrival_v2 | σ=0.5 clip ±0.5 | 42/**0** | 0.705 | **0.940** |

> 出处：paper 1 行 = [`rebrac_experiment_report.md`](rebrac_experiment_report.md) §6.3/§6.4 + [`rebrac_mainline_review.md`](rebrac_mainline_review.md) §2.2.6 表；FQL 行 = [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9.4。

### 5.2 四个轴同时变（任何一个都可能让最优 β1 移动）

1. **reward**：`efficiency_v2 → arrival_v2`（前者 online §7.6 已证 reward hacking / OOB-suicide）；
2. **collector**：`crosscomp`（hand-designed cross-stream-compensation）→ `privileged`（oracle 读 `[u_eq, v_eq]` hull-integral flow）；
3. **条件动作方差 `E_s[Var(a|s)]`**：**低 → 高**（M-uni-noise 是 deliberate σ=0.5 noise injection）；
4. **seed pool**：**含 seed 44 → 不含 seed 44**。这是关键——paper 1 的 β1=4.0 **完全是为了救 seed 44**（report §6.4：β1≤2.0 时 seed 44 崩盘把 std 拉到 0.198）；FQL 线 seed pool {42, 0} 里根本没有 seed 44，于是「为 robustness 选高 β1」的动机消失，干净 seed 上「低 β1 peak 更高」的倾向占了上风。

### 5.3 统一机制（两边都能解释）

BC-anchor 强度在 (collector × reward × noise) 空间里**没有单一全局最优**：
- **ReBRAC** 锚定到 *raw* dataset action（含噪）→ noisy data 上必须把 β1 调低，否则强行模仿带噪动作；
- **FQL** 锚定到 *flow-denoised* teacher 重建（target 已 clean）→ 不需要随噪声调。

**同一族旋钮、相反的最优点，完全由 anchor target 是否带噪决定。** 这正好解释了 β1 在 paper 1 与 FQL succession 上「最优值」不同。

### 5.4 paper 1 是否需要补实验？

不强制。paper 1 finding (i)–(iv) 不依赖「β1=4.0 全局最优」，只依赖「β1=4.0 在 paper 1 cells 上 ≥ TD3+BC + dual penalty 必要 + critic LN 独立必要」，FQL succession 没推翻任何一条。Revision 期防御选项（递增）：
1. **默认**：method section 加一句 implementation note（β1=4.0 是 seed-44-sensitive std 准则下的 finalist；noisy raw-action 下最优下移，cross-ref FQL succession P2 §3）；
2. **轻触发**：limitations/scope 段引用 FQL P2 §3–§6；
3. **重触发**：补 `crosscomp-1000 (β1=1.0, β2=2.0) × 5 seed × test=100`（~½ 天 L4）作 fallback——预期 mean 略升但 std 仍 > β1=4.0（seed 44 仍崩），paper 1 决策维持。

> 完整论证：[`rebrac_mainline_review.md`](rebrac_mainline_review.md) §2.2.6（paper 1 侧）+ [`fql_succession_p2_results.md`](fql_succession_p2_results.md) §8.1（paper 2 侧）。

---

## 7. 文件索引（想找 X 看 Y）

| 想找... | 看这里 |
|---|---|
| 整条 offline 线全景（含其他算法线） | [`offline_rl_line_summary.md`](offline_rl_line_summary.md) |
| 当前主用数据集 + 状态空间 schema（SAC collector 基线） | 本文 §5 + [`environment_design.md`](environment_design.md) + [`auv_nav/env.py`](../auv_nav/env.py) |
| SAC collector 设计 / checkpoint 盘点 / trigger | [`arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md) |
| baseline collector 策略类 / offline 数据收集脚本 | [`auv_nav/baselines.py`](../auv_nav/baselines.py) + [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) |
| ReBRAC One Page + 4 finding spine | [`rebrac_mainline_review.md`](rebrac_mainline_review.md) §0 + §1.5 |
| ReBRAC 任何具体数字 / per-seed | [`rebrac_experiment_report.md`](rebrac_experiment_report.md)（按 stage 索引） |
| β1=4.0 winner 的选择逻辑（seed 44 robustness） | [`rebrac_experiment_report.md`](rebrac_experiment_report.md) §6.3 + §6.4 |
| β1 跨线 reconciliation（本文段 6 的源） | [`rebrac_mainline_review.md`](rebrac_mainline_review.md) §2.2.6 + [`fql_succession_p2_results.md`](fql_succession_p2_results.md) §8.1 |
| TD3+BC baseline 详细收口 | [`td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) |
| 广验 v2（s0 partial-obs ceiling） | [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) |
| FQL succession 机制 ground truth（Q1/Q1b/Q1c/C-1） | [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9 |
| FQL succession 论文写作索引 | [`fql_succession_paper_writing_index.md`](fql_succession_paper_writing_index.md) |
| ReBRAC agent 代码 | [`auv_nav/rebrac.py`](../auv_nav/rebrac.py)；训练 entry [`scripts/train_offline.py`](../scripts/train_offline.py)；sweep launcher [`scripts/run_offline_rebrac_screen.sh`](../scripts/run_offline_rebrac_screen.sh) / [`run_offline_rebrac_broad.sh`](../scripts/run_offline_rebrac_broad.sh) |

---

## 8. 一句话总结

ReBRAC 线由 **TD3+BC 拆瓶颈 → ReBRAC 在两瓶颈上拿四条 finding（核心：deployable 追平 privileged、优于 TD3+BC 23–32pp，finalist β1=4.0）→ 广验 v2 给出 s0 partial-obs ceiling 边界 → FQL succession 诚实负面回头确证 ReBRAC（β1=1.0 双轴 dominate FQL）** 四段构成；FQL 线与原线最优 β1 不同源于 **reward / collector / 动作噪声 / seed pool 四轴差异 + 原线 β1=4.0 本是 seed-44-robustness 的 std-driven 选择**，两边不矛盾，paper 1 finding 全部站立。
