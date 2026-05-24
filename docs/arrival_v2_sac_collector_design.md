# `arrival_v2` SAC Collector — Design & Checkpoint Inventory

> **文档版本**：2026-05-24（rev.2，主线协议对齐）
> **状态**：**active — 用户已声明启动**（2026-05-24）
> **作用**：盘点 `codex-arrival-v2-prototype` 分支产出的 SAC checkpoint，对比 SAC collector vs 现有 rule-based baseline collector，给出 D4RL 风格数据收集的推荐子集与实施要点。
> **协议约束（rev.2 锁定）**：与 offline 主线（broad val v2 + FQL P2）**严格对齐 = `s0` probe + `history-length 4` + `arrival_v2` reward**。现有 offline datasets 全部 `*_s0_h4_arrival_v2_*` 命名，本文档推荐的 SAC collector 数据集复用相同 schema。
> **上下文**：[`docs/online_rl_line_summary.md`](online_rl_line_summary.md) §4.3；[`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) §4.3。
> **实验依据**：[`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7 / §7.6 / §7.7 / §7.8 / §7.9 / §7.9.7。
> **驱动动机（rev.2 锁定）**：(D) D4RL 范式对齐（paper revision 必备弹药）+ (E) FQL 在 s0_k4 SAC-trained behavior policy 数据上再验证（FQL P2 没回答过的开放问题）。详 §6.0。

---

## 目录

1. [SAC collector vs rule-based baseline collector — 本质差异](#1-sac-collector-vs-rule-based-baseline-collector--本质差异)
2. [arrival_v2 SAC ckpt 完整清单 + s0_k4 双约束子集](#2-arrival_v2-sac-ckpt-完整清单)
3. [按 D4RL quality tier 的重新视角（s0_k4 双约束 8 ckpt）](#3-按-d4rl-quality-tier-的重新视角)
4. [推荐数据收集双路径（路径 1 cross_u15 + 路径 2 cross_u10）](#4-推荐用于数据收集的子集按使用情景)
5. [工程实施要点](#5-工程实施要点)
6. [触发条件 + open questions](#6-open-questions--等待-offline-线确认的事项)

---

## 1. SAC collector vs rule-based baseline collector — 本质差异

仓库现有 [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) 注册了 4 个 rule-based baseline 策略（`POLICY_MAP`）：`goalseek` / `crosscomp` / `worldcomp` / `privileged`。SAC collector 是用训练完成的 `SquashedGaussianActor` ckpt 当 behavior policy。两者在 **policy 性质 / data 分布 / 评估范式 / 工程成本** 四个维度有本质差异。

### 1.1 Policy 性质对照

| | 仓库现有 baselines | SAC checkpoint collector |
|---|---|---|
| 类别 | 4 个手写规则策略（[`auv_nav/baselines.py`](../auv_nav/baselines.py)） | 训练完成的神经网络策略（`SquashedGaussianActor`） |
| 决策依据 | 启发式公式（goal bearing、lateral v_flow、世界系流速反解、特权 corridor 评分） | 学到的 NN 映射 `obs_t → π_θ(a\|s)` |
| 是否使用 privileged info | `privileged`：是（直接读 `env.wake_field`）；`worldcomp`：是（用 `env.last_equivalent_current_world`）；`goalseek` / `crosscomp`：仅用 obs | actor 端只看 obs（s0/s1/s2），不看 privileged_obs — **deployment-realistic** |
| 输出分布 | 全部 deterministic（一次性 closed-form） | 默认 Gaussian (`mean ± σ`)，eval 时可切 `deterministic=True`（仅取 mean） |
| 适用 sensor | 多数（`worldcomp` / `privileged`）**与 sensor 无关**——直接读 env 状态，绕过 probe | 强 sensor-bound——只看训练时配置的 `probe_layout` |

→ 最关键的差别：**`worldcomp` / `privileged` 数据集和 sensor 配置解耦**（action 与 `s0/s1/s2` 无关），同一份数据集可以拿去训不同 sensor 的 offline policy；**SAC collector 不行**——sensor 一改，behavior policy 就废。

### 1.2 State-action 分布差异

| | baseline 数据 | SAC 数据 |
|---|---|---|
| 成功率分布 | 各 baseline 差异大：`goalseek` 在 cross/upstream 几乎全失败；`worldcomp` 中等；`privileged` 接近 upper bound | A0 vanilla SAC + s0 在 cross_u10 ~79%，arrival_v2 prototype 跨 ckpt 在 0.1–1.0 全谱分布 |
| 失败模式 | timeout / out-of-bounds 模式带"规则痕迹"——轨迹相似度高 | 失败模式更"NN 风格"——震荡、突变、相位错位；与 evaluation 时实际部署的失败模式更接近 |
| 状态覆盖 | 轨迹族系窄（episode 之间高度相似） | 轨迹族系更宽，stochastic actor 有 σ；replay 在 600k–1M 训练过程中的 policy 演化也带来 distribution shift |
| Behavior policy entropy | 0（确定性） | ~0.2–0.5，受 SAC `alpha` 控制 |
| `next_actions`（已记录） | 完全可预测（确定性策略） | 真实 stochastic — 更适合训 BC penalty 类算法（ReBRAC、FQL 等） |

→ 这两类数据**喂给 offline RL 算法的训练信号本质不同**：BC penalty 在确定性 baseline 数据上等价于拟合 deterministic mapping；在 SAC 数据上等价于拟合 Gaussian — **后者才是 D4RL 标准范式的近似**。

### 1.3 D4RL 范式对齐

D4RL benchmark 的 `medium` / `medium-expert` / `expert` 数据集**几乎全部**用 SAC checkpoint 收集，原因：

- **算法可比性**：所有 offline RL 论文（CQL / IQL / TD3+BC / ReBRAC / FQL）都在"由 RL agent 收的数据"上测过；用 rule-based baseline 收的数据**不是 community 默认评测条件**
- **真实 deployment proxy**：SAC checkpoint 模拟"先 online 训一段时间，再 freeze 用作 behavior policy"的真实场景
- **Quality tier 可调**：通过取不同训练阶段的 ckpt（200k / 400k / 600k）可以做 `random` / `medium` / `expert` 三档对照

→ 仓库现有 baselines 更接近 "expert demonstrations from a handwritten controller"（imitation learning 范式），**不是 D4RL 范式**。若 paper 想说 "我们的 offline RL 算法在 D4RL-style 数据上有效"，**必须用 SAC collector**；用现有 baselines 收的数据只能说 "在 handcrafted-policy demonstrations 上有效"，审稿人会质疑泛化性。

### 1.4 训练 / 部署一致性

- **现有 baselines**：reward 由 env 计算，behavior policy 与 reward shaping **无任何耦合**——同一份 `worldcomp` 数据可以喂给训练 `efficiency_v2` 或 `arrival_v2` 的 offline policy
- **SAC collector**：behavior policy 是**在某一组 reward + sensor + benchmark 下训出来的**——数据分布隐式编码了该 reward 的偏好

→ 实操推荐：**用 ckpt 时 reward 与 ckpt 训练 reward 对齐**（arrival_v2 ckpt → 用 arrival_v2 reward 收数据 → offline 训 arrival_v2 reward）。换 reward 会引入 data/reward mismatch。

### 1.5 工程成本对照

| 维度 | baseline 数据 | SAC 数据（从 arrival_v2 ckpt） |
|---|---|---|
| 训练成本 | 0 | **已沉没**（19 个 ckpt 已存在，无需重训） |
| 收集成本 | 每个 dataset ≈ 0.5–1h（CPU pool） | 每个 dataset ≈ 0.5h + 需要写 SAC-as-policy adapter（约 1–2h 编码） |
| Adapter 需求 | 已有 `POLICY_MAP`（5 行注册即可） | 需在 [`collect_offline_data.py`](../scripts/collect_offline_data.py) 加 `SACCheckpointPolicy` wrapper：`load_state → SquashedGaussianActor → act(obs, deterministic) → action` |
| 复用性 | 多 sensor / 多 reward 一份数据通用 | **每对 (sensor, reward) 都需要单独训一次 SAC 才能收**，但 arrival_v2 prototype 19 个 ckpt 已覆盖 4 个 sensor × 5 个 benchmark |

### 1.6 一句话本质区别

> 现有 baselines 数据是"**rule-based + deterministic + sensor-agnostic 的 expert demonstrations**"；SAC collector 数据是"**learned + stochastic + sensor-bound 的 D4RL-style behavior policy**"。前者是 imitation learning 的奶妈数据，后者才是 offline RL community 默认基准。**两者不能互替**，只能在 paper 里作为"两类来源"互证。

---

## 2. `arrival_v2` SAC ckpt 完整清单

来源：`codex-arrival-v2-prototype` 分支，全部由 [`scripts/train_sac.py`](../scripts/train_sac.py) + `arrival_v2` reward + 1M step（除 p1_v6 1.5M）训练。

**物理位置**：`checkpoints/arrival_v2_prototype/<benchmark>/arrival_v2/<algo>/<sensor>/<seed_X>/` 在 Drive 上；本地 `experiments/arrival_v2_prototype/...` 树下保留 `trainer_state.json` / `train_config.txt` / `replay_latest.pkl` / `rng_state.pkl` 索引。

**总数**：19 个 ckpt。

### 2.0 按 offline 主线 s0_k4 双约束筛选 — 8 ckpt（rev.2 主推清单）

用户决定（2026-05-24）保持 SAC collector 与 offline 主线协议严格对齐 → 过滤掉 `s0_k8` / `s0_k12` / `s1_k4` ckpt，仅保留 **`s0_k4 + arrival_v2`** 双约束 ckpt：

| # | ckpt 路径 | benchmark | algo | seed | final | peak | D4RL tier |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `cross_u10_regression/sac_vanilla/s0_k4/seed_46` | u10_cross | vanilla | 46 | **1.000** | 1.000 | **EXPERT** |
| 2 | `single_u15_upstream/sac_vanilla/s0_k4/seed_42` | u15_upstream | vanilla | 42 | **1.000** | 1.000 | **EXPERT** |
| 3 | `tandem_u15_upstream/sac_vanilla/s0_k4/seed_42` | tandem | vanilla | 42 | **1.000** | 1.000 | **EXPERT** |
| 4 | `sbs_u15_upstream/sac_vanilla/s0_k4/seed_42` | sbs | vanilla | 42 | **1.000** | 1.000 | **EXPERT** |
| 5 | `single_u15_cross/sac_vanilla/s0_k4/seed_0` | u15_cross | vanilla | 0 | 0.400 | 0.533 | **MEDIUM** |
| 6 | `single_u15_cross/sac_vanilla/s0_k4/seed_42` | u15_cross | vanilla | 42 | 0.100 | 0.367 | **FAIL / medium-replay** |
| 7 | `single_u15_cross/sac_asym/s0_k4/seed_42` | u15_cross | sac_asym | 42 | 0.167 | 0.267 | **FAIL** |
| 8 | `single_u15_cross/sac_asym/s0_k4/seed_0` | u15_cross | sac_asym | 0 | 0.200 | 0.267 | **FAIL** |

**关键观察**：8 个 ckpt 自然分两组：
- **4 个 expert tier**（u10_cross + 3 个 upstream cell）：D4RL `expert` 范式数据源
- **4 个 medium/failure tier on cross_u15**：D4RL `random` / `medium-replay` 范式数据源 — **不存在 expert tier**

→ **cross_u15 + s0_k4 + arrival_v2 上不存在 expert SAC ckpt 是 sensor floor 的实证**（详 §4.5），不是 ckpt 缺失。

---

> **以下 §2.1–§2.5 是全 19 ckpt 的 cell-by-cell 完整索引（reference）**，包含 `s0_k8` / `s0_k12` / `s1_k4` 等 rev.2 协议下不采用的 ckpt，仅作 future-work 参考。本节 §2.0 已是 rev.2 主推清单。

### 2.1 `single_u15_cross_tgt15`（cross_stream，主对照 cell，11 个 ckpt — 最丰富）

| ckpt 路径 | sensor / k | algo | seed | final | peak | OOB | best_step | §-ref / 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `sac_vanilla/s1_k4/seed_42` | s1_k4 | vanilla | 42 | **0.900** | 0.900 | 0.10 | 1.0M | §7.1 upper bound reference |
| `sac_vanilla/s0_k4/seed_42` | s0_k4 | vanilla | 42 | 0.100 | 0.367 | 0.667 | 975k | §7.6.4 catastrophic FAIL — 80pp gap floor |
| `sac_vanilla/s0_k4/seed_0` | s0_k4 | vanilla | 0 | 0.400 | 0.533 | 0.20 | 625k | §7.7.1 sister |
| `sac_asym/s0_k4/seed_42` | s0_k4 | **AsymCritic** | 42 | 0.167 | 0.267 | 0.633 | 950k | §7.7 pure-B FAIL |
| `sac_asym/s0_k4/seed_0` | s0_k4 | **AsymCritic** | 0 | 0.200 | 0.267 | 0.567 | 775k | §7.7.1 sister FAIL |
| `sac_vanilla/s0_k8/seed_42` | s0_k8 | vanilla | 42 | **0.900** | 0.900 | 0.10 | 925k | §7.8 PASS — 闭合 80pp gap |
| `sac_vanilla/s0_k8/seed_0` | s0_k8 | vanilla | 0 | 0.500 | 0.500 | 0.13 | 1.0M | §7.9.1 PARTIAL |
| `sac_vanilla/s0_k8/seed_7` | s0_k8 | vanilla | 7 | 0.867 | 0.900 | 0.13 | 825k | §7.9.1'' BORDERLINE |
| `sac_vanilla/s0_k12/seed_42` | s0_k12 | vanilla | 42 | **0.900** | 0.900 | 0.10 | 1.0M | §7.9.2 PASS-PLATEAU |
| `sac_vanilla/s0_k12/seed_0` | s0_k12 | vanilla | 0 | **0.900** | 0.900 | 0.10 | 1.0M | §7.9.2' CROSS-SEED-RESCUE ⭐ |
| `sac_vanilla/s0_k12/seed_7` | s0_k12 | vanilla | 7 | 0.833 | 0.900 | 0.167 | 575k | §7.9.2'' NEAR-PASS-FLOOR-PINNED ⭐⭐ |

### 2.2 `single_u15_upstream_tgt15`（2 个 ckpt）

| ckpt 路径 | sensor / k | algo | seed | final | peak | best_step |
|---|---|---|---:|---:|---:|---:|
| `sac_vanilla/s1_k4/seed_42` | s1_k4 | vanilla | 42 | **1.000** | 1.000 | 850k |
| `sac_vanilla/s0_k4/seed_42` | s0_k4 | vanilla | 42 | **1.000** | 1.000 | 950k |

### 2.3 `tandem_u15_upstream_tgt15`（2 个 ckpt，双柱 tandem G/D=3.5）

| ckpt 路径 | sensor / k | algo | seed | final | peak | best_step |
|---|---|---|---:|---:|---:|---:|
| `sac_vanilla/s1_k4/seed_42` | s1_k4 | vanilla | 42 | **1.000** | 1.000 | 475k（最快） |
| `sac_vanilla/s0_k4/seed_42` | s0_k4 | vanilla | 42 | **1.000** | 1.000 | 750k |

### 2.4 `sbs_u15_upstream_tgt15`（2 个 ckpt，双柱 sbs G/D=3.5）

| ckpt 路径 | sensor / k | algo | seed | final | peak | best_step |
|---|---|---|---:|---:|---:|---:|
| `sac_vanilla/s1_k4/seed_42` | s1_k4 | vanilla | 42 | **1.000** | 1.000 | 825k |
| `sac_vanilla/s0_k4/seed_42` | s0_k4 | vanilla | 42 | **1.000** | 1.000 | 900k |

### 2.5 Reference / regression（2 个 ckpt，confound 不进 §7 主对照）

| ckpt 路径 | benchmark | sensor | seed | final | peak | total_steps | 用途 |
|---|---|---|---:|---:|---:|---:|---|
| `cross_u10_regression/arrival_v2/sac_vanilla/s0_k4/seed_46` | `single_u10_cross_tgt15` | s0_k4 | 46 | **1.000** | 1.000 | 1.0M | §2 cross_u10 旁证（U=1.0, 简单 wake） |
| `p1_v6/arrival_v2/sac_vanilla/s1_k4/seed_46` | `single_u15_upstream_tgt15` | s1_k4 | 46 | — | — | 1.5M | §3 P1 v6 旁证（1.5M 长 budget, best_step=1.4M） |

---

## 3. 按 D4RL quality tier 的重新视角

按 rev.2 主推 8 ckpt（s0_k4 + arrival_v2 双约束）分桶。19-ckpt 全集分桶见 §3.A reference。

### 3.1 主推 8 ckpt（rev.2）的 D4RL tier 分布

| Tier | ckpt 数 | 覆盖 cell | 说明 |
|---|---:|---|---|
| **Expert（final ≥ 0.9）** | 4 | u10_cross / u15_upstream / tandem / sbs | 4 个 cell 各 1 个 seed，**全单 seed** |
| **Medium（final 0.4–0.5）** | 1 | u15_cross | sac_vanilla seed=0 |
| **Failure / random（final ≤ 0.2）** | 3 | u15_cross | sac_vanilla seed=42 + sac_asym {42, 0} |

→ **天然形成两层结构**：(a) **4 cell 单 seed expert tier**（topology / regime diversity），(b) **u15_cross cell 4-ckpt medium/failure tier**（同 cell 多 ckpt，可做 D4RL `random` / `medium-replay` mixture）。

### 3.2 关键缺口：cross_u15 上没有 expert tier ckpt

**这不是 ckpt 缺失，是 sensor floor 实证**（详 §4.5）：

- §7.6 catastrophic FAIL（vanilla seed=42 best peak 0.367）+ §7.7 AsymCritic 不能闭合 80pp gap
- 同物理对应 broad val v2 N2' STRONG_NEGATIVE（offline ReBRAC 0.0/30）+ FQL P2 §6.5 FLOOR（offline 0.14 / collector 0.719）

→ 任何坚持 `s0_k4` 协议的 algorithm（online SAC、offline ReBRAC、offline FQL）在 cross_u15 都 FLOOR。SAC collector 的 ckpt-quality ceiling 与 offline RL 的 outcome ceiling 在此 cell 上**同被 sensor floor 锁定**。

### 3.A 全 19 ckpt 集的 D4RL tier 分布（reference，rev.2 协议下不采用）

- **Expert（final ≥ 0.9）** — 13 ckpt：4 主推 cell × s0_k4 + 8 个 s0_k8/k12/s1_k4 cell（含 single_cross s0_k12 seed=42/0 + s0_k8 seed=42 + s1_k4 seed=42）
- **Medium-expert（final 0.83–0.87）** — 2 ckpt：single_cross s0_k8/seed=7、s0_k12/seed=7
- **Medium（final 0.4–0.5）** — 2 ckpt：single_cross s0_k4/seed=0、s0_k8/seed=0
- **Failure / random（final ≤ 0.2）** — 3 ckpt：single_cross s0_k4/seed=42、s0_asym/{42, 0}

→ 19-ckpt 全集天然覆盖 D4RL `random / medium / medium-expert / expert` 四 tier，但 rev.2 协议下只采用 8 个 s0_k4 子集。

---

## 4. 推荐数据收集双路径（rev.2 主推）

按用户两个驱动动机精确化：(D) D4RL 范式对齐 + (E) FQL 在 SAC-trained behavior policy 数据上再验证。两条路径并行：

### 4.1 路径 1（PRIMARY） — cross_u15 cell × 4 ckpt：对话 broad val v2 N2' / FQL §6.5 FLOOR

**目标**：给 offline 线补一个全新数据 regime — "在 sensor floor 下挣扎的 RL agent"。当前 offline 线只有 rule-based（crosscomp/privileged）和 oracle teacher（v2 N2'）两类数据；SAC-trained-but-floored behavior policy 是第三类。

**收集清单**（4 ckpt × 1000 ep）：

| ckpt | best peak | final | 收集形式 | D4RL 类比 |
|---|---:|---:|---|---|
| `single_u15_cross/sac_vanilla/s0_k4/seed_0` | 0.533 | 0.400 | `agent_best.pt` → 1000 ep stochastic | `medium` |
| `single_u15_cross/sac_vanilla/s0_k4/seed_42` | 0.367 | 0.100 | `agent_best.pt` → 1000 ep stochastic | `medium-replay`（best ≠ final） |
| `single_u15_cross/sac_asym/s0_k4/seed_42` | 0.267 | 0.167 | `agent_best.pt` → 1000 ep stochastic | `random` |
| `single_u15_cross/sac_asym/s0_k4/seed_0` | 0.267 | 0.200 | `agent_best.pt` → 1000 ep stochastic | `random` |

**输出 dataset 命名约定**（与现有 `*_s0_h4_arrival_v2_*` 对齐）：

```
offline_data/sac_vanilla_s0_h4_arrival_v2_re250_u15cross_seed{0,42}_ep1000/
offline_data/sac_asym_s0_h4_arrival_v2_re250_u15cross_seed{0,42}_ep1000/
```

**研究问题**：当 behavior policy 自己也在 s0 floor regime 下学崩了，offline ReBRAC β1=1.0 / FQL 在这种 RL-trained-failed-policy 数据上的表现，与在 privileged-oracle-but-s0-uncovered 数据（v2 N2'）上的表现，是否有本质区别？FQL P2 §8.1 的 "BC-anchor 最优强度随目标噪声翻转" 机制预测 — SAC stochastic policy 的 `E_s[Var(a|s)]` 介于 deterministic 与 σ=0.5 injected 之间，应能给出与 P2 不同的最优 β1。

**Algorithm head-to-head**（建议同步跑）：FQL 与 ReBRAC β1∈{1.0, 4.0} 各 2 seed × 4 dataset = 16 run × ~30min L4 ≈ 8h。

---

### 4.2 路径 2（PRIMARY） — cross_u10 cell × 1 ckpt + 补 2 seed：D4RL `expert` tier

**目标**：D4RL 范式对齐需要 expert tier。cross_u10 是 ReBRAC paper 1 / FQL P2 主对照 cell。

**现有 ckpt**：`cross_u10_regression/sac_vanilla/s0_k4/seed_46` (final=1.0) — **单 seed**。

**补 seed 需求**：建议补 `seed=47, 50`（与 A0 seed 池对齐）各 1 个 600k–1M arrival_v2 SAC 训练。

| ckpt | 状态 | 工作量 |
|---|---|---|
| `cross_u10_regression/.../seed_46` | ✅ 已有 | 0 |
| `cross_u10/arrival_v2/sac_vanilla/s0_k4/seed_47` | ❌ 待训 | ~1.5h L4 (600k) |
| `cross_u10/arrival_v2/sac_vanilla/s0_k4/seed_50` | ❌ 待训 | ~1.5h L4 (600k) |

**收集清单**（3 ckpt × 1000 ep）：

```
offline_data/sac_vanilla_s0_h4_arrival_v2_re150_u10cross_seed{46,47,50}_ep1000/
```

**研究问题**：D4RL `expert` tier SAC 数据集上，FQL 与 ReBRAC β1∈{1.0, 4.0} 的 head-to-head 是否复现 FQL P2 主对照结论（FQL ≈ ReBRAC β1=1.0 on clean expert）？同时是 paper 1 / FQL P2 的 SAC-collector 第四轴补全。

**与 broad val v2 N0 的关系**：N0 dataset（`crosscomp_s0_h4_arrival_v2_re150_u10cross_ep1000`）已存在。SAC collector dataset 与 N0 同 cell × 同 reward × 同协议，**形成 N0 vs SAC head-to-head 平行对照** — broad val v2 §6.3 follow-up 列出的 "SAC collector 平行第四轴" 直接落地。

---

### 4.3 路径 3（SECONDARY，可延后） — 拓扑泛化 × 3 个 upstream cell

| ckpt | benchmark | D4RL tier |
|---|---|---|
| `single_u15_upstream/sac_vanilla/s0_k4/seed_42` | u15_upstream | EXPERT |
| `tandem_u15_upstream/sac_vanilla/s0_k4/seed_42` | tandem | EXPERT |
| `sbs_u15_upstream/sac_vanilla/s0_k4/seed_42` | sbs | EXPERT |

**用途**：paper appendix 的 topology diversity 旁证（"D4RL expert dataset 在 3 种 wake topology 下的 offline RL 表现"）。**单 seed**，不进 primary critical path。

---

### 4.4 算法对照 sub-axis（vanilla vs AsymCritic 行为差异）

路径 1 的 4 个 ckpt 自带 vanilla / sac_asym 各 2 seed → 同 seed × same cell 下 critic-side privileged info 改变 dataset 行为风格的实证：

| 指标（30-ep eval，§7.7 实测） | vanilla seed=42 | AsymCritic seed=42 | 方向 |
|---|---:|---:|---|
| safety_cost | 25.85 | 16.93 | asym 更安全（−34%） |
| progress_ratio | 0.277 | 0.453 | asym 更接近目标（+64%） |
| return | — | +26% | asym return 更高 |
| **task success** | 0.100 | 0.167 | asym 也 FAIL，但更不烂 |

→ Dataset 性质本身有研究价值：critic-side priv info 改变 behavior style 但不闭合 task-level gap。**与路径 1 同一批数据，零额外收集成本**。

---

### 4.5 Sensor floor 在 cross_u15 上的方法学含义

`s0_k4 + arrival_v2 + cross_u15` 上**所有 algorithm 同被 sensor floor 锁定**：

| 数据源 | online SAC | offline ReBRAC β1=4 | offline FQL |
|---|---:|---:|---:|
| 自身上限（best success） | 0.367 (§7.6) | 0.000 (v2 N2') | 0.14 (FQL §6.5) |

**方法学含义**：
1. SAC collector ckpt-quality ceiling = offline algorithm outcome ceiling = sensor floor — 在此 cell 上 **D4RL 数据 tier 天然只能是 random/medium-replay**，"expert tier 不存在"本身是可写入 paper 的 finding
2. 与 §7.9.7 manifest universal floor 同源（vanilla SAC s0 在此 manifest 的 inherent ceiling 由 sensor 信息量决定）
3. 路径 1 收集的 4 个 ckpt 数据集，paper 中应明确标注 "behavior policy quality bounded by s0_k4 sensor floor"，避免读者误以为是 SAC 训练不充分
4. Offline RL 算法在路径 1 数据上是否能 **超过 behavior policy peak**（如 best peak 0.533 → offline 训出 ≥ 0.6），是 D4RL 范式下"是否超过 behavior policy"的标准问题 — 真发生的话**反向证伪 s0_k4 sensor floor 在 offline 下也成立**，是高价值 finding

---

### 4.6 Best vs Final ckpt 的选择规则

路径 1 全部用 **`agent_best.pt`**：4 个 ckpt 都是 final < peak（训练 collapse），用 final 会拿到 collapse 后策略。

路径 2 / 路径 3：expert tier ckpt，best ≈ final，选哪个都可以；推荐 `agent_best.pt` 保险。

**对照表**：

| ckpt | peak | final | gap | 选 |
|---|---:|---:|---:|---|
| 路径 1 / vanilla seed=42 | 0.367 | 0.100 | −0.267 | **best** |
| 路径 1 / vanilla seed=0 | 0.533 | 0.400 | −0.133 | **best** |
| 路径 1 / asym seed=42 | 0.267 | 0.167 | −0.100 | **best** |
| 路径 1 / asym seed=0 | 0.267 | 0.200 | −0.067 | **best** |
| 路径 2/3 expert ckpt | ≥ 0.9 | ≥ 0.9 | 0 | best 或 final 等价 |

---

## 5. 工程实施要点

### 5.1 Ckpt 文件位置与命名约定

- **远程（Drive）**：`drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5/checkpoints/arrival_v2_prototype/<benchmark>/arrival_v2/<algo>/<sensor>/seed_<N>/`
- **本地（gitignored）**：`./checkpoints/arrival_v2_prototype/...`（同结构，但通常为空 — 仅 Colab 写入）
- **每个 seed 目录里有的 ckpt 文件**：
  - `agent_best.pt` — eval 阶段 best `success_rate` 时刻的 ckpt（**收数据首选**）
  - `agent_final.pt` — 训练结束 step 的 ckpt（可能已 collapse）
  - `agent_latest.pt` — 最近一次 periodic save（与 final 通常重合）
  - `agent_step_<N>.pt` — 每 `checkpoint_every_steps=100000` 一个，共 10 个（D4RL multi-tier 收集可用）

### 5.2 Adapter 编码 spec（约 1–2 小时工作量）

在 [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) 中加：

```python
class SACCheckpointPolicy:
    """Wrap a trained SquashedGaussianActor as a baseline policy.

    Mirrors the baseline policy interface: act(env, obs) -> action.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        deterministic: bool = True,  # D4RL 默认 expert 用 deterministic
    ) -> None:
        from auv_nav.sac import SACAgent, SACConfig  # 或直接 load actor

        state = torch.load(ckpt_path, map_location=device)
        # 从 trainer_state.json 或 state["config"] 还原 obs_dim / action_dim
        self.actor = SquashedGaussianActor(...)
        self.actor.load_state_dict(state["actor"])
        self.actor.eval()
        self.device = device
        self.deterministic = deterministic

    def act(self, env: PlanarRemusEnv, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic=self.deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)
```

在 `POLICY_MAP` 旁边加 CLI 入口（不进入 `POLICY_MAP` — 因为它需要 ckpt path 参数）：

```python
parser.add_argument("--sac-ckpt", type=str, default=None,
                    help="Path to SAC checkpoint .pt (overrides --policy)")
parser.add_argument("--sac-deterministic", action="store_true",
                    help="Use mean action (default: sample from Gaussian)")
```

在 `_make_policy` 分支加 sac 路径：

```python
if args.sac_ckpt:
    policy = SACCheckpointPolicy(
        ckpt_path=args.sac_ckpt,
        device=args.device,
        deterministic=args.sac_deterministic,
    )
```

**关键 invariant**：collector 的 `--probe-layout` / `--history-length` / `--objective` / `--flow` **必须与 ckpt 训练时的 `train_config.txt` 一致**，否则 obs 维度对不上，actor 输出全废。建议加启动 sanity check：

```python
# Pseudocode
trainer_state = json.load(open(ckpt_dir / "../../trainer_state.json"))
assert trainer_state["probe_layout"] == args.probe_layout
assert trainer_state["history_length"] == args.history_length
```

### 5.3 `replay_latest.pkl` 作为 D4RL `medium-replay` 数据源（替代方案）

每个 run 的 `state/replay_latest.pkl` 是 **1M transition × 6 envs 的训练轨迹** dump（`TransitionReplay` 序列化）。这是**比 ckpt-eval-rollout 更接近真正 D4RL `medium-replay` 范式**的数据源——D4RL `medium-replay` 本质就是"训练过程中 SAC 的 replay buffer 快照"。

**优点**：
- 零额外计算成本（已经在 disk 上）
- 真实带 policy distribution shift（早期 random → 中期 medium → 后期 expert）
- 自带 `privileged_obs` / `next_privileged_obs`（AsymCritic 实验路径 — 仅 `sac_asym/` 系列有）

**缺点**：
- 需要写一个 `replay_pkl_to_npz.py` 转换脚本（pickle → D4RL-style npz schema）
- 1M transition × 13 fields ≈ 数 GB 单文件
- 不能选 quality tier — 整条 buffer 都拿，不像 ckpt 可挑 step

→ **建议同时支持两种来源**：ckpt eval rollout（quality-tier-controlled）+ replay dump（medium-replay-style）。

### 5.4 Best vs Final ckpt 的选择规则

详 §4.6（已搬到推荐子集附近）。**rev.2 摘要**：路径 1 全部用 `agent_best.pt`（4 ckpt 都 collapse）；路径 2/3 expert tier best ≈ final 等价。

### 5.5 收集时的 reset_options / manifest 选择

- **如果目标是 ReBRAC offline 训练数据**：用与 ckpt 训练时一致的 `task_geometry` / `target_speed` / `flow` 随机化（即不要 `--eval-manifest`，让 episode 在 reset 时正常采样）
- **如果目标是 fixed-manifest 收集 30-ep × N seed 的 deterministic eval rollout**（D4RL eval 标准形式）：用 `benchmarks/single_u15_cross_tgt15.json` 等 manifest + `--episodes 1000` + `--num-workers 8`

### 5.6 总预算估算（rev.2 — 路径 1 + 路径 2 primary）

| 阶段 | 工作量 |
|---|---|
| SACCheckpointPolicy adapter 编码 + sanity check（含 obs_dim / history-length 启动校验） | 1–2h |
| 路径 1：4 ckpt × 1000 ep × CPU pool 收集 | ~1h |
| 路径 2：补 SAC seed=47 / 50 训练（cross_u10 + arrival_v2 + s0_k4，600k） | ~3h L4（可并行） |
| 路径 2：3 ckpt × 1000 ep × CPU pool 收集 | ~0.5h |
| 数据集 metadata 写入（与现有 `*_s0_h4_arrival_v2_*` schema 对齐） | 0.5h |
| **数据收集小计** | **~6h** |
| FQL + ReBRAC β1∈{1,4} head-to-head on 路径 1 (4 dataset × 2 algo × 2 seed) | ~4h L4 |
| FQL + ReBRAC β1∈{1,4} head-to-head on 路径 2 (3 dataset × 2 algo × 2 seed) | ~3h L4 |
| **总计（数据 + algorithm sweep）** | **~13h L4 = 1–2 个 Colab 周** |

→ 路径 3（topology 泛化）可延后；如启动加 ~3h（3 ckpt × 1000 ep + 不额外训 SAC）。

---

## 6. 触发条件 + open questions

### 6.0 触发条件（rev.2，2026-05-24 用户声明启动后重写）

**历史**：rev.1 写过 3 路径（A 审稿人 / B asym-critic 顺势 / C 砍掉），其中：
- **路径 B（asym-critic 顺势）作废**：经独立审查（§ 与本文档同时进行），online §7.7 strong negative + §7.8 actor-side 解已锁定 actor-fundamental 结论，asym-critic on N2' 是独立 research question，不构成 SAC collector 顺势 trigger
- **路径 C 概率降级**：用户已声明启动（D4RL 范式 + FQL 再验证），C base case 不再成立

**rev.2 触发表**：

| 路径 | 触发条件 | 状态 |
|---|---|---|
| **A** | paper revision 期审稿人质疑 "为什么 behavior policy 全 rule-based" | 保留作 fallback rebuttal trigger |
| ~~B~~ | ~~Asym-critic ablation on N2' 顺势~~ | **作废（独立 research question，不是 SAC collector trigger）** |
| ~~C~~ | ~~paper revision 顺利则完全砍掉~~ | **作废（用户已选择启动）** |
| **D** | **D4RL 范式对齐**（paper revision 必备弹药；offline 论文 community 默认评测条件） | **ACTIVE — 用户驱动** |
| **E** | **FQL 在 s0_k4 SAC-trained behavior policy 数据上再验证**（FQL P2 §8.1 "BC-anchor 最优强度随目标噪声翻转" 机制在 SAC stochasticity regime 下的预测未被检验） | **ACTIVE — 用户驱动** |

→ **D + E 共同驱动 rev.2 实施**。路径 1（cross_u15 cell × 4 ckpt）+ 路径 2（cross_u10 cell × 1+2 seed）primary 双轨，详 §4。

### 6.1 其他 open questions

1. **路径 2 是否值得补 seed=47/50**：3h L4 训练成本，换 D4RL `expert` tier multi-seed 的 publishable 价值。决策点：如果 FQL P2 head-to-head 只在 cross_u10 expert tier 复现 P2 主结论（FQL ≈ ReBRAC β1=1.0），单 seed 也能定性 — 但 paper appendix 一般要求 ≥ 2 seed。**建议补**。
2. **Reward 协议**：rev.2 锁定 `arrival_v2`（与 ckpt 训练 reward 一致 + 与 offline 主线 v2 / FQL P2 §6.5 协议一致）。不再考虑 efficiency_v2 路径（efficiency_v2 主线 = paper 1 cross_u10 主对照，但 §6.5 在 arrival_v2 已是 FLOOR，efficiency_v2 重做信息价值低）。
3. **是否记录 privileged_obs**：collector 几乎零成本就能记录，即便主线不用，offline RLPD / AsymCritic ablation 可能用。**默认记录**。
4. **路径 1 是否同时收 best 和 final 两套**：路径 1 全 collapse，**只收 best**；final 可作 sanity comparison（dataset 性质对照），但默认不收。
5. **`replay_latest.pkl` 转换脚本是否值得写**：本地 `experiments/arrival_v2_prototype/<run>/state/replay_latest.pkl` 是 1M transition × 6 envs 训练轨迹 dump，天然是 D4RL `medium-replay` 范式。**rev.2 建议同时支持 ckpt-rollout + replay-dump 两种数据源**（ckpt-rollout 给 quality-tier-controlled 数据，replay-dump 给 medium-replay 范式数据）。
6. **路径 1 在 cross_u15 是否会 offline FLOOR**：高概率会（同 §4.5 sensor floor 论证）；FLOOR 本身就是 finding，**应该跑、应该报**。

---

## 7. 一句话总结（rev.2）

`arrival_v2` prototype 分支沉淀了 19 个 SAC ckpt；按用户决定与 offline 主线协议严格对齐（`s0 + k=4 + arrival_v2`），实际可采用 **8 个 ckpt**，按 D4RL tier 分为 **4 个 expert（u10_cross + 3 upstream cell）+ 1 medium + 3 failure（全在 u15_cross cell）**。**rev.2 主推双路径**：(路径 1) cross_u15 cell × 4 ckpt 收 D4RL `random/medium-replay` 数据，对话 broad val v2 N2' STRONG_NEGATIVE 与 FQL P2 §6.5 FLOOR；(路径 2) cross_u10 cell × seed=46 expert + 补 seed=47/50 训练，收 D4RL `expert` 数据，作 broad val v2 N0 平行第四轴 + FQL P2 主对照 head-to-head。**触发条件升格为 D（D4RL 范式对齐）+ E（FQL 再验证）双 active**（旧路径 B asym-critic 顺势作废）；总预算 **~13h L4 = 1–2 个 Colab 周**（含 adapter 1-2h + 路径 2 补 SAC 训练 3h + 数据收集 1.5h + algorithm head-to-head 7h）。**关键方法学发现可作 paper material**：`s0_k4 + arrival_v2 + cross_u15` 上 expert SAC ckpt 不存在 = sensor floor 的直接实证，与 §7.9.7 manifest universal floor 同源。
