# 世界模型与离线-在线 RL 研究综述

> 文档版本：2026-04-04
> 覆盖范围：Part 2（世界模型方法）+ Part 3（离线-在线 RL）
> 写作背景：本文为三阶段研究计划的后两部分提供文献基础与专业建议
> 前序：SAC 改进方法见 `docs/SAC_improvements_survey.md`

---

## 目录

**Part 2：世界模型方法**
1. [背景与动机](#part-2-世界模型方法)
2. [MBPO：奠基基线](#21-mbponeurips-2019)
3. [DreamerV3：通用 POMDP 世界模型](#22-dreamerv3arXiv-2301.04104--nature-2025)
4. [TD-MPC / TD-MPC2：隐空间规划](#23-td-mpc--td-mpc2icml-2022--iclr-2024)
5. [物理先验与混合世界模型](#24-物理先验与混合世界模型)
6. [SSM/Mamba 在 RL 世界模型中的应用](#25-ssmmamba-在-rl-世界模型中的应用)
7. [近期进展（2024–2025）](#26-近期进展20242025)
8. [方法对比与本项目评估](#27-方法对比与本项目评估)

**Part 3：离线-在线 RL**
9. [背景与动机](#part-3-离线-在线-rl)
10. [纯离线 RL 基础方法](#31-纯离线-rl-基础方法)
11. [离线转在线混合方法](#32-离线转在线混合方法)
12. [BC 正则化在在线 RL 中的使用](#33-bc-正则化在在线-rl-中的使用)
13. [数据质量与覆盖度的影响](#34-数据质量与覆盖度的影响)
14. [针对本项目的专业评估](#35-针对本项目的专业评估)

**参考文献**
15. [参考文献](#参考文献)

---

# Part 2：世界模型方法

## 2.0 背景与动机

世界模型方法（Model-Based RL，MBRL）的核心思想是：让智能体先学习一个环境的内部模型，再在该模型中进行规划或 imagination 训练，从而减少对真实环境交互的依赖。

对本任务的特殊价值：

| 问题特征 | model-free 的局限 | 世界模型的潜在优势 |
|---------|-----------------|-----------------|
| POMDP：流场部分可观测 | 仅能被动反应 | 可显式建模 belief state，预见未来流场状态 |
| 非平稳动力学：涡旋脱落 | 每 episode 流场相位不同，难以泛化 | 世界模型可学习流场的时序规律 |
| 已知 AUV 物理 | 无法利用已知物理先验 | 可将 pH-NODE 作为已知部分，只学习残差/流场 |
| 样本效率 | SAC 需大量真实交互 | 模型内 imagination 可大幅减少真实步数 |

---

## 2.1 MBPO（NeurIPS 2019）

**论文**：Janner et al., "When to Trust Your Model: Model-Based Policy Optimization"
**arXiv**：1906.08253

**核心创新：** 理论上证明了在真实回放缓冲区中的状态上分支 k 步模型 rollout、生成合成数据，再混入 SAC 训练的方案具有单调改进保证。关键在于使用**概率 ensemble 神经网络**（输出高斯分布 $\mathcal{N}(\mu, \sigma)$）建模动力学，通过集成方差控制不确定性，限制 rollout 长度（k=1–5）避免误差积累。

**核心架构：**
- N=7 个独立神经网络 ensemble（训练时随机选 5 个）
- 每个网络预测 $p(s_{t+1}, r_t | s_t, a_t)$ 为高斯分布
- 生成的合成数据与真实数据按比例混合进 SAC replay buffer

**关键数据（论文报告）：**
- Ant 任务：匹配 SAC@3M 步所需性能，仅用 300K 步（**10× 样本效率提升**）
- Walker2d：约 150K 步收敛，vs SAC 约 650K 步
- Hopper：比此前方法快约一个数量级

**主要局限：**
- Ensemble 神经网络计算代价较高（7 个网络并行前向）
- 短 rollout 限制（k≤5）：对需要长程规划的任务无效
- 无法处理 POMDP：假设完整状态可观测

**实现复杂度：中等**。作为历史基线了解即可，不建议直接用于本项目。

---

## 2.2 DreamerV3（arXiv 2301.04104 → Nature 2025）

**论文**：Hafner et al., "Mastering diverse control tasks through world models"
**发表**：arXiv 2023-01；正式发表于 **Nature 2025**

### 2.2.1 架构：RSSM

Recurrent State Space Model（RSSM）是 DreamerV3 的核心，由两个互补组件构成：

```
确定性循环状态 h_t ∈ ℝ^d  (GRU 隐状态，聚合过去的确定性历史)
随机 categorical 潜变量 z_t  (32 个独热向量，各从 32-way categorical 采样 → 1024 维)
```

**完整 world model 组件：**

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| 编码器 $q_\phi(z_t \| h_t, o_t)$ | $(h_t, o_t)$ | 后验 $z_t$ | 从观测更新 belief |
| 循环模型 $f_\phi(h_{t+1} \| h_t, z_t, a_t)$ | $(h_t, z_t, a_t)$ | $h_{t+1}$ | 确定性状态转移 |
| 先验 $p_\phi(z_t \| h_t)$ | $h_t$ | 先验 $\hat{z}_t$ | 纯靠历史预测观测 |
| 解码器 $p_\phi(o_t \| h_t, z_t)$ | $(h_t, z_t)$ | 重建 $\hat{o}_t$ | 监督信号（可选） |
| 奖励头 | $(h_t, z_t)$ | $\hat{r}_t$ | 奖励预测 |
| 继续头 | $(h_t, z_t)$ | $\hat{\gamma}_t$ | episode 终止预测 |
| Actor/Critic | $(h_t, z_t)$ | $a_t, V_t$ | 在 imagination 中训练 |

训练信号由三部分组成：重建损失 + KL 散度（后验 vs 先验，使用 KL 平衡技巧）+ 奖励/继续预测损失。Actor/Critic 完全在 world model 的 imagination rollout 中训练，不需要额外真实步。

### 2.2.2 POMDP 处理能力

这是 DreamerV3 相比 TD-MPC2 的**最核心优势**：

- $h_t$ 积累所有历史观测的确定性信息
- $z_t$ 是当前时刻的**随机 belief state**，通过后验 $q(z_t | h_t, o_t)$ 从当前观测更新
- 联合 $(h_t, z_t)$ 构成完整的信念状态，可代替真实但不可观测的马尔可夫状态
- Actor 和 Critic 以 $(h_t, z_t)$ 为输入，天然具备"历史感知"能力

对本项目的意义：流场涡旋的相位是不可观测的隐变量，恰好可由 $z_t$ 建模。

### 2.2.3 覆盖域与性能（论文报告）

**150+ 个任务，一套超参数配置**，包括：
- DMControl（本体感知 + 像素输入）
- Atari 57 游戏（首个超越人类均值的算法）
- DMLab-30（3D 第一视角导航）
- BSuite
- **Minecraft（MineDojo）：首个从零收集钻石的 RL 算法**，无人类数据或课程

vs SAC 和 TD-MPC2（在 DMControl 上）：
- DreamerV3 样本效率优于 SAC
- TD-MPC2 在精细操控任务上优于 DreamerV3（偶有数值不稳定）
- DreamerV3 在长时记忆、视觉输入、稀疏奖励任务上有独特优势

### 2.2.4 工程细节与训练技巧

DreamerV3 引入了若干训练稳定性技巧，这些细节对实际应用很重要：

- **symlog 变换**：对奖励和价值应用 $\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$，处理奖励尺度差异大的问题
- **两点分布离散化（TWO-HOT）**：将连续值的标量目标编码为离散分布，避免回归不稳定
- **自由比特（free nats）**：KL 散度低于阈值时不施加梯度，防止 KL 塌陷
- **KL 平衡**：后验 KL 的梯度分配给先验（0.8）和后验（0.2），加速先验学习

**实现复杂度：高**。官方 JAX 实现约 1.5 万行，PyTorch 非官方实现维护度不一。

---

## 2.3 TD-MPC / TD-MPC2（ICML 2022 / ICLR 2024）

**论文（TD-MPC）**：Hansen et al., "Temporal Difference Learning for Model Predictive Control," ICML 2022
**论文（TD-MPC2）**：Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control," ICLR 2024
**arXiv**：2310.16828

### 2.3.1 架构

TD-MPC2 是**无解码器（decoder-free）的隐式世界模型**，全部组件均为 MLP：

| 组件 | 符号 | 说明 |
|------|------|------|
| 编码器 | $z_t = h_\theta(o_t)$ | 观测 → 隐空间 |
| 隐动力学 | $\hat{z}_{t+1} = f_\theta(z_t, a_t)$ | 一步马尔可夫转移（**无递归**） |
| 奖励头 | $\hat{r}_t = R_\theta(z_t, a_t)$ | 奖励预测（离散回归） |
| 终止头 | $\hat{d}_t = D_\theta(z_t)$ | episode 结束预测 |
| Q 函数 ensemble | $Q_\theta(z_t, a_t)$ | 价值估计，多头 ensemble |

规划器：在隐空间上运行 **MPPI**（Model Predictive Path Integral）或 CEM，以 Q 函数引导超出规划时域的价值估计。

**稳定性技巧：**
- **SimNorm**：将隐状态分组，每组投影到单纯形（simplex）上 via softmax，防止梯度不稳定
- LayerNorm + Mish 激活
- 离散回归（离散化连续值目标）

### 2.3.2 POMDP 能力——关键局限

**TD-MPC2 本质上不处理 POMDP。** 隐动力学 $f_\theta(z_t, a_t) \to z_{t+1}$ 是单步 MLP，没有任何递归结构或历史积累机制。这意味着：

- $z_t$ 只是当前观测 $o_t$ 的非线性变换，**不包含历史信息**
- 在部分可观测环境中，多个不同的真实状态可能产生相同的 $o_t$，进而映射到相同的 $z_t$
- MPPI 规划基于马尔可夫假设，在非马尔可夫环境中会系统性地产生次优动作

后续工作（如 "Learning Contextual World Models Aids Zero-Shot Generalization," arXiv 2403.10967）尝试通过上下文向量增强 $z_t$ 来部分缓解此问题，但不是完整的 POMDP 解法。

### 2.3.3 基准性能

**104 个连续控制任务**（DMControl 39 + Meta-World + ManiSkill2 + MyoSuite）：
- 优于 SAC 和 DreamerV3 的样本效率和最终性能
- 在 DMControl 所有 39 个任务上达到 >90% 专家性能
- 在精细操控任务上优于 DreamerV3（DreamerV3 偶有数值不稳定）
- 317M 参数单一智能体在 80 个任务上多体型联合训练（多任务缩放实验）

**TD-MPC vs TD-MPC2 的关键差异：**
- TD-MPC 可能发生梯度爆炸；TD-MPC2 加入 SimNorm 解决稳定性
- TD-MPC2 支持多任务 task embedding
- TD-MPC2 改进了规划算法（更稳定的 MPPI 实现）

**实现复杂度：中等**。官方 PyTorch 实现结构清晰，约 3000 行。

---

## 2.4 物理先验与混合世界模型

这是与本项目最直接相关的方向，也是文献中最稀少的方向。

### 2.4.1 残差动力学模型（Residual Dynamics）

**核心思想：** 不从零学习完整动力学，而是在已知物理模型 $f_\text{phys}(s, a)$ 基础上学习残差：

$$s_{t+1} = f_\text{phys}(s_t, a_t, u_c) + \underbrace{\Delta f_\theta(s_t, a_t)}_{\text{学习残差}}$$

对本项目：AUV 6-DOF 动力学（pH-NODE）是已知且高精度的；需要学习的只有流场等效流速 $\hat{u}_c$ 对 AUV 运动的影响，或者流场自身的时空演化。

**相关文献：**
- **Residual Model-Based RL for Physical Dynamics**（NeurIPS 2022 Workshop）：在已知解析动力学先验基础上学习残差修正，减少真实交互需求，直接对应本项目场景。

- **Physics-Informed MBRL**（Ramesh & Ravindran, L4DC 2023, arXiv 2212.02179）：在初始条件敏感的物理环境中，比较纯数据驱动与物理约束的 MBRL。结论：物理约束版本在样本效率和平均回报上均显著更优。

### 2.4.2 Neural ODE 在 RL 世界模型中

**DyNODE**（arXiv 2009.04278）：将 Neural ODE（Runge-Kutta 积分器）替换 MBRL 中的标准 MLP 动力学模型。优势：
- 连续时间建模（$ds/dt = f_\theta(s, a)$），天然适合变步长和物理系统
- 对于已知守恒律（如能量），可通过 Hamiltonian 参数化强制满足
- 在连续控制中比标准 NN 动力学具有更好的预测精度和样本效率

**与 pH-NODE 的关系**：本项目已有预训练的 port-Hamiltonian Neural ODE，恰好是这一路线的精化版本（加入了 Hamiltonian 结构约束）。

### 2.4.3 PIN-WM：物理约束视觉世界模型

**PIN-WM**（RSS 2025, arXiv 2504.16693）："Physics-INformed World Models for Non-Prehensile Manipulation"

- 从视觉观测中识别 3D 刚体动力学参数，构建物理约束世界模型
- 两阶段：先学习物理参数，再用于 model-based RL 规划
- 代表了"物理参数识别 + 模型基 RL"的最新进展

### 2.4.4 对本项目的专业判断

**本项目的世界模型需要分离两类不确定性：**

```
AUV 动力学        → 已知（pH-NODE），精度高，无需学习
流场状态演化      → 未知且部分可观测，这才是世界模型需要学习的核心
```

因此，理想的世界模型架构是：

```
world_model(o_t, a_t) = pH-NODE(s_t, a_t, û_c_t)
                         其中 û_c_t = flow_model(probe_history_t)
```

这等价于：固定已知动力学（pH-NODE），只学习流场的 belief state 模型（对应 DreamerV3 的 RSSM 角色）。**这是文献中尚未被系统研究的新颖组合，是本研究的核心创新点之一。**

---

## 2.5 SSM/Mamba 在 RL 世界模型中的应用

State Space Models（SSM）近年成为 RNN 的重要替代方案。

### 2.5.1 R2I：记忆任务专用（ICLR 2024 Oral）

**论文**：Samsami et al., "Mastering Memory Tasks with World Models," ICLR 2024 Oral（Top 1.2%）
**arXiv**：2403.04253

**核心创新：** 将 DreamerV3 的 GRU 替换为 S4/H3 等结构化状态空间模型（S3M），同时保留 RSSM 的后验-先验双流结构。

**结果：** 在 BSuite、POPGym、Memory Maze 上达到 SOTA，Memory Maze 实现超人类性能。速度**比 DreamerV3 快 9 倍**（SSM 的并行化扫描替代了 RNN 的顺序计算）。

**关键发现：** DreamerV3 的 GRU 在 episode 超过 ~30 步时失去长程依赖；SSM 可有效处理数百步的长记忆。

**对本任务的意义：** AUV episode 长度 ~480 步，涡旋周期 ~20-40 步。若流场相位推断需要较长历史，R2I 的 SSM 世界模型可能显著优于标准 RSSM。

### 2.5.2 Drama：Mamba 世界模型（2024）

**论文**："Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient"
**arXiv**：2410.08893

**核心创新：** 将 Mamba-2（SSM 的矩阵乘法表述）替换 RSSM 中的序列模型部分：
- O(n) 内存和计算（vs Transformer 的 O(n²)）
- 无梯度消失（vs RNN）
- **仅 7M 参数**，可在普通笔记本上训练

**结果：** 在 Atari100k 上达到竞争性性能，资源占用大幅降低。

### 2.5.3 SWIM：加速 MBRL 训练（ICLR 2025 Workshop）

**论文**："Accelerating Model-Based RL with State-Space World Models"
**arXiv**：2502.20168

将 RSSM 替换为可并行化 SSM，world model 训练**加速 10 倍**，整体 MBRL 训练**加速 4 倍**。在无人机竞速（复杂动力学）上验证。

---

## 2.6 近期进展（2024–2025）

### EfficientZero V2（arXiv 2403.00564）

MuZero 系列扩展到连续动作空间。在 DMControl 上与 TD-MPC2 竞争，代表"规划 + 搜索树"路线的延续。

### WIMLE（arXiv 2602.14351，2025）

不确定性感知世界模型，在 Humanoid-run 上比最强 baseline 提升 50%+ 样本效率。

### DreamerV3 → Nature（2025）

原 arXiv 论文正式发表于 **Nature**，验证了其科学影响力。架构本身无重大变化。

---

## 2.7 方法对比与本项目评估

### 2.7.1 核心维度对比

| 方法 | POMDP 能力 | 物理先验兼容 | 样本效率 | 实现复杂度 | 规划能力 |
|------|-----------|------------|---------|-----------|---------|
| MBPO | ✗ | △（可加 ensemble） | ★★★★ vs SAC | 中 | 无（间接） |
| DreamerV3 | **✓✓（RSSM belief state）** | △（需修改架构） | ★★★★ | **高** | imagination 内 |
| TD-MPC2 | **✗（Markovian）** | **✓（可插 pH-NODE）** | ★★★★★ | 中 | MPPI 显式规划 |
| R2I（SSM）| **✓✓（更长记忆）** | △ | ★★★★ | 高 | imagination 内 |
| 混合（pH-NODE+FlowModel）| **✓（自定义）** | **✓✓** | 依实现 | 高 | MPPI 显式规划 |

### 2.7.2 针对本项目的专业判断

**两条可行路线，各有侧重：**

**路线 A：DreamerV3（或 R2I）直接应用**
- 优点：RSSM 天然处理 POMDP，belief state 可捕获流场相位隐变量；代码成熟
- 缺点：无法利用 pH-NODE 已知物理；解码器学习 AUV 动力学是浪费；实现复杂
- 适合作为纯学习基线

**路线 B：pH-NODE + 流场世界模型 + MPPI（RESEARCH_PLAN.md 方案）**
- 优点：物理先验大幅降低需学习的参数量；MPPI 规划的可解释性；安全约束可直接嵌入规划
- 缺点：需要自研 FlowWorldModel（GRU 时序编码器 + 时空查询头）；比路线 A 更多工程工作
- 适合作为本研究的**核心创新贡献**

**两条路线的消融关系：**

```
基线:    DreamerV3（全学习，POMDP-aware）
消融1:   TD-MPC2（全学习，非 POMDP，显式规划）
消融2:   pH-NODE + 常数流速 + MPPI（已知物理，无流场学习）
完整:    pH-NODE + FlowWorldModel + MPPI（已知物理 + 流场 belief state）
```

这个消融矩阵可以清楚区分：① POMDP 处理的价值；② 物理先验的价值；③ 显式规划的价值。

**R2I / SWIM 的价值：** 若流场相位推断确实需要 >30 步的历史，R2I 的 SSM world model 可作为 DreamerV3 的直接升级，且速度更快。建议在 DreamerV3 基线稳定后评估是否必要。

---

# Part 3：离线-在线 RL

## 3.0 背景与动机

离线-在线 RL（Offline-to-Online / Hybrid RL）的核心问题：**能否利用已有的历史数据（expert/sub-optimal demonstrations 或 baseline policy rollouts）加速 RL 训练或提升最终性能？**

本项目的具体数据来源：
- Baseline policy rollouts（`GoalSeekPolicy`、`CrossCurrentCompensationPolicy`、`WorldFrameCompensation`）
- 早期 SAC 训练产生的次优轨迹
- 未来 SAC/World Model 阶段产生的有价值数据

**核心判断**（先说结论，后展开论据）：

当仿真代价极低时，离线-在线 RL 对本项目的价值是**有限但真实的**。预期收益是训练初期 2–5x 更快的收敛，而非质的性能突破。

---

## 3.1 纯离线 RL 基础方法

### 3.1.1 CQL：Conservative Q-Learning（NeurIPS 2020）

**论文**：Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning"
**arXiv**：2006.04779

**核心创新：** 在标准 TD 损失基础上加罚项：最小化策略动作的 Q 值、最大化数据中动作的 Q 值，使 Q 函数提供对当前策略真实值的可证明下界。

$$\mathcal{L}_\text{CQL} = \underbrace{\alpha \left[\mathbb{E}_{a \sim \pi}[Q(s,a)] - \mathbb{E}_{a \sim \pi_\beta}[Q(s,a)]\right]}_{\text{保守性惩罚}} + \underbrace{\text{TD Loss}}_{\text{标准 Bellman}}$$

**D4RL 性能（归一化分数，近似）：**

| 任务 | CQL |
|------|-----|
| halfcheetah-medium | ~46.7 |
| hopper-medium | 59–78 |
| walker2d-medium | ~79.7 |
| antmaze-medium-diverse | ~53 |

**在线微调的已知问题：** CQL 学习的 Q 值过度保守（系统性低估）。在线微调开始时，新遇到的 OOD 动作的 Q 值相对"看起来更好"，导致策略追逐这些动作，发生 **性能崩溃（performance collapse）**。这是 Cal-QL 要解决的核心问题。

---

### 3.1.2 IQL：Implicit Q-Learning（ICLR 2022）

**论文**：Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning"
**arXiv**：2110.06169

**核心创新：** **完全避免查询分布外动作的 Q 值。** 通过分离价值学习与策略提取：

1. 用**期望值回归（expectile regression）**学习状态值函数 $V(s)$，近似数据分布中最好动作的期望值：
   $$\mathcal{L}_V = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[\mathcal{L}_\tau(Q(s,a) - V(s))\right]$$
   其中 $\mathcal{L}_\tau(u) = |\tau - \mathbf{1}(u < 0)| u^2$，$\tau \in (0.5, 1)$

2. 用 SARSA 型目标更新 Q：$Q(s,a) \leftarrow r + \gamma V(s')$

3. 优势加权 BC 提取策略：$\pi \leftarrow \arg\max_\pi \mathbb{E}[\exp(\beta(Q-V)) \log\pi(a|s)]$

**D4RL 性能（归一化分数，近似）：**

| 任务 | IQL |
|------|-----|
| halfcheetah-medium | 46.2 |
| hopper-medium | 56.8 |
| walker2d-medium | 77.0 |
| hopper-medium-expert | 90–110 |
| antmaze-medium-diverse | ~70 |
| antmaze-large-diverse | ~47 |

IQL 在 **AntMaze（需要轨迹拼接的稀疏奖励任务）** 上尤为突出，这是 TD3+BC 几乎失效的地方。

**计算效率：** 单张 GTX 1080 上 100 万次更新约 20 分钟。

**在线微调问题：** IQL 在离线→在线过渡时存在"Q 值重标定冲击"——Q 值在接触在线数据时发生显著变化，可能导致短暂性能回退。

---

### 3.1.3 TD3+BC（NeurIPS 2021）

**论文**：Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning"
**arXiv**：2106.06860

**核心创新：** 仅两处修改 TD3：

1. 策略目标加入 BC 正则化项：
   $$\pi = \arg\max_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[\lambda Q(s, \pi(s)) - (\pi(s) - a)^2\right]$$
   其中 $\lambda = \alpha / \left(\frac{1}{N}\sum |Q(s_i, a_i)|\right)$ 归一化 Q 值尺度

2. 状态归一化（零均值，单位方差）

**D4RL 性能（归一化分数，近似）：**

| 任务 | TD3+BC |
|------|--------|
| halfcheetah-medium | 48.1 |
| walker2d-medium | 82.9 |
| halfcheetah-medium-expert | 92.1 |
| walker2d-medium-expert | 111.1 |
| antmaze-medium-diverse | ~3（基本失效）|

TD3+BC 在稠密奖励 locomotion 任务上极具竞争力，但在 AntMaze（需要轨迹拼接）上几乎完全失效。

**在线微调适配性差：** BC 项天然限制策略改进幅度，在线微调时必须显式去掉或衰减 BC 项。

---

## 3.2 离线转在线混合方法

### 3.2.1 RLPD：RL with Prior Data（ICML 2023）

**论文**：Ball et al., "Efficient Online Reinforcement Learning with Offline Data"
**arXiv**：2302.02948

**核心创新：** 证明不需要任何特殊的离线 RL 算法。标准 SAC + 三个设计选择：

1. **对称采样（50/50）**：每个 batch 中 50% 来自离线数据，50% 来自在线 buffer
2. **Critic 加 LayerNorm**：防止 Q 值外推崩溃
3. **大 Q ensemble（10 个 critic）**：降低过估计偏差

无离线预训练阶段，无保守性惩罚，无策略约束。

**关键性能（论文报告）：**
- 在 21 个 benchmark（AntMaze、稀疏 Adroit、Locomotion）上比此前方法最高提升 2.5x
- AntMaze 中，约 100k–300k 在线步内解决所有任务（vs 纯 SAC 需要 1M+ 步）
- **优于 IQL 在线微调**：即使 IQL 有更好的离线初始化，RLPD 在在线阶段很快超越

**实现简洁性：极高**。本质上是 SAC + 第二个 replay buffer，约 50–100 行额外代码。

**对离线数据质量的敏感性：** RLPD 的固定 50/50 采样是数据质量无关的。若离线数据质量极差（纯随机策略），50% 的 batch 浪费在无用数据上，可能减慢早期训练。

---

### 3.2.2 Cal-QL：Calibrated Q-Learning（NeurIPS 2023）

**论文**：Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning"
**arXiv**：2303.05479

**解决的核心问题：** CQL 离线训练后，Q 值过于保守（远低于真实值）。在线微调时，新遇到的动作（即使平庸）在 Q 值尺度上"看起来更好"，导致策略崩溃。

**"Calibrated"的含义：** 在 CQL 的保守下界基础上，额外施加参考策略（行为策略）价值的上界约束，形成"三明治"：

```
V(参考策略) ≥ Q_learned(s,a) ≥ V_true(当前策略)
```

Q 值保守但不过度保守，在线微调时不会出现值函数崩溃。

**实现成本：** 在 CQL 基础上的**一行代码改动**（替换保守性下界的动态阈值）。

**结果：** 在 11 个在线微调 benchmark 中有 9 个超过 SOTA，修复了 CQL 在线微调时的性能崩溃。

---

### 3.2.3 其他 2022–2025 方法

**PROTO（arXiv 2305.15669，2023）**
- 在在线策略上施加信任域约束（trust-region），锚定到预训练策略附近
- 约束强度随在线训练逐步放松，平衡稳定性与最终性能
- 与任意离线预训练方法 + 任意在线 RL 方法兼容

**WSRL：Warm Start RL（ICLR 2025，arXiv 2412.07762）**

关键发现：**在线微调时不需要保留离线数据。** 仅用少量预训练策略的 rollout 预热在线 buffer（"warm-up phase"），桥接分布偏移，然后纯在线训练。

- 在机器人 peg 插入任务（Franka）约 18 分钟完成
- 比保留离线数据的方法收敛更快、渐近性能更高
- 对本项目的启示：若离线数据质量参差不齐，WSRL 的"只用预热而不保留离线数据"策略可能更干净

**SHARSA（NeurIPS 2025 Spotlight，arXiv 2506.04168）**

**这是纯离线 RL 的可扩展性研究，不是离线-在线方法。** 核心发现：horizon 长度是阻碍离线 RL 随数据量扩展的主要瓶颈（即使 1000x 更多数据，标准方法性能饱和）。SHARSA 用 n-step SARSA 降低有效 horizon，配合基于 flow matching 的分层 BC 策略。

- **前提：** 需要大规模离线数据集（远超标准 D4RL 规模）
- **与本项目关系：** 不适用。本项目仿真代价低，数据量由在线交互主导，不是 SHARSA 要解决的场景

---

## 3.3 BC 正则化在在线 RL 中的使用

### 3.3.1 核心方法对比

**AWR：Advantage-Weighted Regression**（Peng et al., 2019）

$$\pi \leftarrow \arg\max_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[\exp\left(\frac{A(s,a)}{\beta}\right) \log\pi(a|s)\right]$$

用优势函数 $A = Q - V$ 对 BC 项加权，高优势动作得到强约束，低优势动作被忽略。

**CRR：Critic-Regularized Regression**（Wang et al., NeurIPS 2020，arXiv 2006.15134）

使用分布式 Q 函数，优势加权 BC 的过滤函数可选：binary（阈值过滤）或 exponential（软加权）。Exponential 过滤等价于 AWAC（Nair et al.）的公式。

**与本项目的关联：** 本 repo 已实现 BC loss 衰减机制（`train_gru_residual_sac.py` 中的 `--pretrain-updates` + BC loss 参数）。在 Part 3 阶段，将此扩展为优势加权 BC（AWR 风格）可能更有效，因为它自动忽略次优数据中的低质量动作。

### 3.3.2 BC 衰减策略

文献共识：

- **线性衰减**：简单，但可能过早或过晚失去 BC 约束
- **指数衰减**：更平滑，$\lambda_\text{BC}(t) = \lambda_0 \cdot e^{-t/\tau}$
- **自适应衰减**（Zhao et al. 2022，arXiv 2210.13846）：根据 agent 性能动态调整 $\lambda_\text{BC}$，性能下降时增大，性能提升时减小

**当离线数据是次优的：** 无差别 BC 约束会设置天花板。优势加权 BC（AWR/CRR）可自动降低次优动作的权重，允许策略超越演示水平。

---

## 3.4 数据质量与覆盖度的影响

### 3.4.1 D4RL 数据质量分类

D4RL 的四种数据质量标签（Fu et al., arXiv 2004.07219）：

| 类型 | 生成方式 | 特点 |
|------|---------|------|
| `random` | 随机策略 rollout，~1M 步 | 最差，几乎无结构 |
| `medium` | SAC 训练到约 1/3 专家性能后 rollout | 次优但有结构 |
| `medium-replay` | SAC 训练过程中所有 replay buffer | 混合质量（从极差到中等） |
| `medium-expert` | 50% medium + 50% 专家数据 | 覆盖度广 |
| `expert` | 完整训练的 SAC 策略 rollout | 最高质量 |

**关键规律（跨方法共同观察）：**

- `random` 数据：几乎所有方法都效果差；BC 约束实际上有害（学习随机行为）
- `medium` 数据：离线 RL（CQL、IQL）显著优于 BC，轨迹拼接能力关键
- `medium-replay`：优势加权方法（IQL、CRR）优于均匀 BC
- 包含**低质量数据**：通常不造成灾难性损害，但浪费容量、减慢收敛；advantage-weighted 方法对此更鲁棒

**最小数据量阈值：** 文献无精确答案，但经验规律：
- Locomotion：~100k–500k 步提供实质性收益
- 稀疏奖励任务：~1M+ 步才能充分覆盖状态空间

### 3.4.2 对本项目的估计

**可用的离线数据质量分析：**

| 数据来源 | 质量类比 | 对 RLPD 的价值 |
|---------|---------|--------------|
| `GoalSeekPolicy` rollouts | `medium` 左右（部分有效） | 中等——提供合理目标导向行为 |
| `WorldFrameCompensation` rollouts | `medium` 左右 | 中等——流场补偿有用 |
| `PrivilegedCorridorPolicy` | 接近 `expert`（使用特权信息） | **高**——但分布与可部署策略不同，注意分布偏移 |
| 早期 SAC 训练数据 | `medium-replay`（混合质量） | 中等——包含从随机到中等的轨迹 |

---

## 3.5 针对本项目的专业评估

### 3.5.1 仿真代价低时的离线-在线 RL 价值判断

**支持使用离线数据的理由：**

1. **初期收敛加速：** RLPD 在稀疏奖励任务上将收敛步数从 1M+ 压缩到 100k–300k。即使仿真便宜，训练时间仍然有价值。

2. **免费数据：** 你已经有 baseline policy 的 rollout 数据，利用它的成本几乎为零（RLPD 只需约 50 行额外代码）。

3. **探索保险：** 在流场中某些区域，纯在线 SAC 的随机探索可能长期无法覆盖（例如需要精确穿越涡旋间隙的轨迹）。Baseline rollouts 提供这些区域的先验覆盖。

4. **AUV 领域的直接先例：** MAIOOS（Ocean Engineering 2025）在多 AUV 导航（有海流干扰）中验证了混合离线-在线 RL，比纯在线方法收敛更快。

**反对使用离线数据的理由：**

1. **收益边际递减：** 当在线步数可以达到 5M–10M 时，RLPD 的加速效果在后期逐渐消失。

2. **POMDP 复杂性：** RLPD 等方法设计并验证于 MDP 设定。本项目的观测堆叠（MLP SAC）是 MDP 近似，但如果未来转向真正的 POMDP 世界模型，离线数据的应用方式需要重新考虑（history-conditioned state 的离线数据收集和使用更复杂）。

3. **特权数据的分布偏移风险：** `PrivilegedCorridorPolicy` 的 rollout 使用了部署时不可用的流场信息，其行为分布与可部署策略有本质差异。将其直接加入 RLPD 的离线 buffer 可能产生负迁移。

### 3.5.2 推荐实验路线

**基于以上分析，Part 3 的建议实验结构：**

```
基线（已有）：
  pure_online_sac          - 纯在线 SAC（Part 1 最佳配置）

数据价值验证（先做）：
  rlpd_medium_quality      - RLPD + GoalSeek/WorldCompensation rollouts（中等质量数据）
  rlpd_high_quality        - RLPD + PrivilegedCorridor rollouts（高质量但分布偏移）
  rlpd_mixed               - RLPD + 所有可用离线数据（混合质量）

方法对比（选最佳 RLPD 配置后）：
  rlpd_vs_calql            - 比较 RLPD（无预训练）vs Cal-QL（有离线预训练阶段）
  bc_warmup_wsrl           - BC 预热 + 纯在线训练（WSRL 思路）

消融：
  rlpd_sampling_ratio      - 50/50 vs 25/75 vs 75/25 采样比例
  rlpd_offline_quality     - 只用高质量数据 vs 只用低质量数据 vs 混合
```

### 3.5.3 最终建议

**对 Part 3 的定位建议：**

不要将 Part 3 定位为"仿真代价低时如何用离线数据节省训练步数"——这个问题的答案几乎必然是"收益有限"。

**更有价值的定位：** 研究在不同**数据质量**和**数据来源**下，离线数据对本任务（POMDP + 非平稳流场）的影响规律，以及**分布偏移**（来自特权信息 baseline 的数据）对在线学习的正负迁移效应。这个问题在 AUV/流体动力学领域是真实且有意义的，区别于标准 D4RL 问题设置。

---

## 参考文献

### Part 2：世界模型

**核心方法**
- **MBPO**：Janner et al., "When to Trust Your Model: Model-Based Policy Optimization," NeurIPS 2019. arXiv: 1906.08253
- **DreamerV3**：Hafner et al., "Mastering diverse control tasks through world models," Nature 2025 (arXiv 2301.04104, 2023)
- **TD-MPC**：Hansen et al., "Temporal Difference Learning for Model Predictive Control," ICML 2022
- **TD-MPC2**：Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control," ICLR 2024. arXiv: 2310.16828

**物理先验与混合世界模型**
- **Physics-Informed MBRL**：Ramesh & Ravindran, L4DC 2023. arXiv: 2212.02179
- **Residual Model-Based RL**：NeurIPS 2022 Workshop. neurips.cc/virtual/2022/60675
- **DyNODE**：arXiv 2009.04278
- **PIN-WM**：RSS 2025. arXiv: 2504.16693

**SSM/Mamba 在世界模型中**
- **R2I（Mastering Memory Tasks）**：Samsami et al., ICLR 2024 Oral. arXiv: 2403.04253
- **Drama（Mamba World Model）**：arXiv 2410.08893
- **SWIM**：ICLR 2025 Workshop. arXiv: 2502.20168

**近期进展**
- **EfficientZero V2**：arXiv 2403.00564
- **WIMLE**：arXiv 2602.14351
- **Contextual TD-MPC2**：arXiv 2403.10967

**基础模型**
- **PlaNet（RSSM 奠基）**：Hafner et al., "Learning Latent Dynamics for Planning from Pixels," ICML 2019. arXiv: 1811.04551
- **RSSM 原始论文**：Hafner et al., 同上

### Part 3：离线-在线 RL

**纯离线 RL**
- **CQL**：Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS 2020. arXiv: 2006.04779
- **IQL**：Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning," ICLR 2022. arXiv: 2110.06169
- **TD3+BC**：Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning," NeurIPS 2021. arXiv: 2106.06860
- **CRR**：Wang et al., "Critic Regularized Regression," NeurIPS 2020. arXiv: 2006.15134

**离线-在线混合**
- **RLPD**：Ball et al., "Efficient Online Reinforcement Learning with Offline Data," ICML 2023. arXiv: 2302.02948
- **Cal-QL**：Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning," NeurIPS 2023. arXiv: 2303.05479
- **PROTO**：Li et al., 2023. arXiv: 2305.15669
- **WSRL**：Zhou et al., "Warm Start RL," ICLR 2025. arXiv: 2412.07762

**离线 RL 可扩展性**
- **SHARSA**：Park et al., "Horizon Reduction Makes RL Scalable," NeurIPS 2025. arXiv: 2506.04168

**BC 正则化**
- **AWR**：Peng et al., "Advantage-Weighted Regression," arXiv 2019
- **Adaptive BC Regularization**：Zhao et al., 2022. arXiv: 2210.13846

**数据与 Benchmark**
- **D4RL**：Fu et al., "D4RL: Datasets for Deep Data-Driven Reinforcement Learning," arXiv 2004.07219

**领域相关**
- **MAIOOS**：Multi-AUV navigation with offline-online RL，Ocean Engineering 2025
