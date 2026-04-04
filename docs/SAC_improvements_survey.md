# SAC 改进算法研究综述

> 文档版本：2026-04-03
> 覆盖范围：2020–2026 年 SAC 及相关 off-policy 连续控制算法
> 写作背景：本综述聚焦于 MLP SAC（含观测堆叠）的改进路径，为 AUV 尾流导航任务提供参考

---

## 目录

1. [背景与综述范围](#1-背景与综述范围)
2. [样本效率：高 UTD 与 Ensemble](#2-样本效率高-utd-与-ensemble)
3. [网络架构改进](#3-网络架构改进)
4. [Replay Buffer 与数据利用](#4-replay-buffer-与数据利用)
5. [Critic 过估计偏差改进](#5-critic-过估计偏差改进)
6. [训练稳定性与 Plasticity](#6-训练稳定性与-plasticity)
7. [探索与熵正则化](#7-探索与熵正则化)
8. [局部可观测性（无 RNN）](#8-局部可观测性无-rnn)
9. [技术演进路线图](#9-技术演进路线图)
10. [针对本项目的专业评估](#10-针对本项目的专业评估)
11. [参考文献](#11-参考文献)

---

## 1. 背景与综述范围

Soft Actor-Critic（SAC，Haarnoja et al. 2018）是连续控制领域最有影响力的 off-policy 算法之一。其核心设计包括：最大熵强化学习目标、Twin Q-network（从 TD3 引入的 Clipped Double Q 技巧）、以及自动熵系数调节。

2020 年以来，SAC 改进工作沿三条相对独立的技术脉络演进：

```
样本效率脉络：  REDQ (2021) → DroQ (2022) → CrossQ (2024) → XQC (2026)
可塑性/规模化：  Primacy Bias (2022) → BRO (2024) → SimBa (2025) → SimbaV2 (2025)
数据利用脉络：   SAC-N → EDAC (2021) → RLPD (2023)
```

本综述聚焦于**可落地的改进技术**，剔除了以下方向：

- **生成式策略**（Diffusion Policy、Flow Matching）：2D 动作空间下无实质增益，推理延迟有害
- **世界模型方法**（TD-MPC2、DreamerV3）：属于不同范式，超出本综述范围
- **纯离线 RL**（CQL、IQL）：动态流场覆盖度不足，离线数据在线化收益有限

---

## 2. 样本效率：高 UTD 与 Ensemble

### 2.1 REDQ：随机 Ensemble Double Q-Learning

**论文**：Chen et al., "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model"
**发表**：ICLR 2021

**核心创新：** 三个要素的组合：
1. 高 UTD 比（每个环境步做 20 次梯度更新，UTD=20）
2. N=10 个 Q 网络的 ensemble
3. 计算 target 时随机选取 M=2 个 Q 网络取最小值（in-target minimization）

三者协同压制高 UTD 下的过估计偏差。首次让无模型方法在样本效率上追平模型基方法（MBPO）。

**主要局限：** N=10 个网络同时训练，计算量是 SAC 的约 5–10 倍。

**实现复杂度：** 中等

---

### 2.2 DroQ：Dropout Q-Functions

**论文**：Hiraoka et al., "Dropout Q-Functions for Doubly Efficient Reinforcement Learning"
**发表**：ICLR 2022

**核心创新：** 用 Dropout + LayerNorm 替代 REDQ 的大 ensemble。仅用 2 个 Q 网络，Dropout 提供隐式多样性，实现与 REDQ 相当的样本效率（UTD=20），同时将计算开销压到远低于 REDQ 的水平，论文称之为"doubly efficient"（样本效率与计算效率均优）。

**与 SAC 的计算量比较：** DroQ 远比 REDQ 便宜，论文中将其与 SAC 相对比的描述是"comparable computational efficiency"——但 UTD=20 意味着 DroQ 仍然比 SAC（UTD=1）的计算量高约 10-15 倍，这点需注意。

**实现复杂度：低**。在现有 critic 每层加 Dropout 和 LayerNorm，约 20 行代码改动。

---

### 2.3 CrossQ：BatchRenorm + 去掉 Target Network

**论文**：Bhatt et al., "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity"
**发表**：ICLR 2024

**核心创新：** 两个关键技术组合：

1. **Batch Renormalization（BRN）**，而非标准 BatchNorm。BRN 在 warm-up 后使用运行统计量而非 mini-batch 统计量，避免了时序数据下 BatchNorm 的不稳定性。

2. **去掉 target network**。实现方式是将 (s, a) 和 (s', a') 拼接成一个大 batch 做一次 forward pass，使 BN 统计量同时覆盖两个分布——这是 CrossQ 名称的来源，也是去掉 target network 的关键机制（单纯用 BRN 不够，批次拼接是必要条件）。

**结果：** UTD=1 即可媲美 DroQ（UTD=20）的样本效率，计算量大幅下降。

**兼容性说明：** 与 MLP SAC 完全兼容。BatchNorm 对 GRU/RNN 不适用，但本项目已确认使用 MLP SAC，无此顾虑。

**实现复杂度：低**

---

### 2.4 BRO：Bigger, Regularized, Optimistic

**论文**：Nauman et al., "Bigger, Regularized, Optimistic: Scaling for Compute and Sample-Efficient Continuous Control"
**发表**：NeurIPS 2024 Spotlight

**核心创新：** 三管齐下：

1. **Bigger（BroNet 架构）**：2 个残差块、512 宽、LayerNorm、~5M 参数
2. **Regularized**：LayerNorm + weight decay + 周期性全参数 reset
3. **Optimistic**：双 actor 探索——exploration actor 优化分位数 ensemble 不确定性上界（KL 正则化的 Q 上界），exploitation actor 正常采样。加上分布式（quantile regression）critic

**关键数据（论文报告）：**
- 标准 SAC 在多个 benchmark 任务上约达到最优性能的 25%；BRO 超过 90%
- BRO Fast（UTD=2）：与 SAC wall-clock 时间相当，性能约提升 400%（4 倍）
- 在 Dog、Humanoid 等困难任务上首次接近最优解

**实现复杂度：高**。需实现 BroNet、分布式 critic、双 actor 乐观探索、reset 调度。

---

### 2.5 XQC：Well-Conditioned Optimization

**论文**：Palenicek et al., "XQC: Well-Conditioned Optimization Accelerates Deep Reinforcement Learning"
**发表**：ICLR 2026（arXiv: 2509.25174，CrossQ 团队）

**核心创新：** 将 BN + 权重归一化（WN）+ 分布式交叉熵损失三者组合，使 critic Hessian 的条件数比 baseline 小数个量级。条件数小意味着梯度方向更准确、学习率更稳定，在非平稳 RL target 下尤为重要。

**关键数据（论文报告）：** 与 SimbaV2 相比，参数量少约 4.5 倍，FLOPs 少约 5 倍，同时在 55 个本体感知任务 + 15 个视觉任务上达到相当或更高性能（UTD=1）。

**说明：** 这是截至本文写作时最新的同方向工作，实验结果尚未经过广泛独立复现，建议保持审慎态度。

**实现复杂度：中等**

---

## 3. 网络架构改进

### 3.1 LayerNorm

**背景：** 出现在几乎所有现代 SAC 变体（DroQ、BRO、SimBa、SimbaV2）中。

**机制：** 对每层的激活值归一化，控制特征幅度，提供隐式正则化效果。

**重要性：**
- 在高 UTD 下防止梯度爆炸，是高 UTD 训练的必要条件之一
- Lyle et al.（NeurIPS 2024）证明 LayerNorm 是缓解 plasticity loss 最有效的单项干预
- 大参数量网络没有 LayerNorm 时性能往往退化（MLP scaling 失效）

**实现复杂度：极低**——每层一行代码，立竿见影。

---

### 3.2 残差块（Residual Feedforward Blocks）

**SimBa（ICLR 2025，Hojoon Lee et al., Sony AI）**

三要素：运行均值观测归一化 + 残差前馈块 + LayerNorm。

残差连接提供从输入到输出的线性通路，注入"简单性偏置（simplicity bias）"，使参数量增加时性能单调提升——而标准 MLP 扩大参数时反而退化，这是 MLP 缺乏 scaling 能力的根本原因。

**实现复杂度：低**。SimBa 论文强调设计的故意简洁。

---

### 3.3 SimbaV2：超球面归一化

**论文**：Lee et al., "Hyperspherical Normalization for Scalable Deep Reinforcement Learning"
**发表**：ICML 2025 Spotlight（arXiv: 2502.15280）

**核心创新：**
- 用 **L2 归一化**将中间特征投影到单位超球面（替换 LayerNorm）
- 每次梯度更新后将权重矩阵投影到单位范数超球面
- 配合分布式值估计（quantile regression）和奖励缩放

**关键数据（论文报告）：** UTD=2 下在 57 个连续控制任务（4 个域：MuJoCo、DMC、MyoSuite、HumanoidBench）上归一化分数 **0.892**，前任 SOTA（原版 SimBa）为 0.780。UTD=1 下为 0.848。

**实现复杂度：中等**

---

### 3.4 Spectral Normalization

**核心：** 约束每层权重矩阵的最大奇异值（等价于约束 Lipschitz 常数），允许使用 6-8 层深网络而不发散。

PyTorch 内置 `torch.nn.utils.spectral_norm`，一行包装即可。

**现状评估：** 现代主流方法（BRO、SimBa、SimbaV2）更倾向用 LayerNorm + weight decay 的组合代替 spectral normalization，因为前者在 RL 非平稳环境中更稳定，且计算更简单。Spectral normalization 在分类等固定分布任务中更常见。

**实现复杂度：极低**

---

## 4. Replay Buffer 与数据利用

### 4.1 RLPD：RL with Prior Data

**论文**：Ball et al., "Efficient Online Reinforcement Learning with Offline Data"
**发表**：ICML 2023

**核心创新：** 证明标准 off-policy 算法（SAC）可以用极简方式有效利用离线数据：每个训练 batch 中 **50% 从离线数据采样，50% 从在线经验采样**。配合高 UTD + LayerNorm + Q ensemble，无需任何特殊的离线 RL 算法。

**结果：** 在 21 个 benchmark（D4RL AntMaze、Sparse Adroit、Locomotion）上比现有混合方法性能提升最高 2.5 倍，且实现最简单。

**对本项目的价值：** 若有历史仿真数据（如 baseline policy 的 rollout）或专家演示，这是利用先验数据的最低成本路径。

**实现复杂度：低**

---

### 4.2 优先经验回放（PER）变体

标准 PER（Schaul et al. 2016）按 TD error 优先采样，与 SAC 结合时有几个已知问题：熵项干扰优先级计算、actor 和 critic 的采样需求不同。主要改进方向：

- **DPER（2024）**：为 actor 和 critic 维护独立的优先级队列，避免 off-policy actor 更新的不稳定性
- **PERDP（Scientific Reports 2024）**：根据当前网络状态动态调整优先级

**说明：** PER 在 SAC 中的收益不如在 DQN 中稳定，现有文献显示它在某些连续控制任务上有益，在另一些上无益。除非有明确的样本利用率问题，否则不建议作为优先改进项。

**实现复杂度：中等**

---

## 5. Critic 过估计偏差改进

### 5.1 EDAC：Ensemble-Diversified Actor Critic

**论文**：An et al., "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble"
**发表**：NeurIPS 2021

**核心创新：** 指出 Q ensemble 的梯度方向会趋于对齐，导致对分布外（OOD）动作的惩罚不足。EDAC 在 ensemble 训练中加入**梯度多样化损失**（最小化分布内动作平均梯度的范数），强迫各 Q 网络从不同角度评估同一动作，更有效地识别 OOD 区域。

**结果：** 所需 ensemble 规模约为朴素 ensemble（SAC-N）的 1/10，在 D4RL 上达到 SOTA。

**重要说明：** EDAC 设计并验证于**离线 RL**，其梯度多样化的必要性在离线场景中远强于在线场景（在线 RL 的 OOD 问题通过持续交互自然缓解）。将其应用于在线 SAC 时，收益尚缺乏充分验证。

**实现复杂度：中等**

---

### 5.2 分布式 Critic（Quantile Regression）

**代表**：DSAC（Ma et al. 2020/2021）；BRO 和 SimbaV2 均采用此方式

**核心：** 用分位数回归建模完整 return 分布，而非仅估计均值 Q 值。优势：

1. 梯度信号更丰富（分布信息而非单点估计）
2. 天然缓解过估计（利用不同分位数作为保守/乐观估计）
3. 支持风险敏感策略（CVaR 等）

BRO 和 SimbaV2 均采用分布式 critic，逐渐成为高性能实现的新标配。

**实现复杂度：中等**

---

## 6. 训练稳定性与 Plasticity

这是 2022–2025 年最重要的 RL 稳定性研究方向，核心发现对实践影响深远。

### 6.1 Primacy Bias 与网络 Reset

**论文**：Nikishin et al., "The Primacy Bias in Deep Reinforcement Learning"
**发表**：ICML 2022

**核心发现：** 深度 RL 智能体对早期经验过拟合，随后陷入局部最优，无法从新数据中继续学习。主要原因是神经网络在非平稳 target 下逐渐失去可塑性（plasticity）。

**提出方案：** 周期性重置网络的**最后几层权重**（partial reset），同时保留 replay buffer。这是"reset trick"在 RL 中的奠基性工作。

---

### 6.2 Loss of Plasticity 的系统性研究

**关键论文：**

- **Lyle et al.**, "Normalization and Effective Learning Rates in Reinforcement Learning"，NeurIPS 2024：证明归一化层（尤其是 LayerNorm）是维持有效学习率和对抗 plasticity 损失最有效的单项技术干预。

- **Nauman et al.**, "Overestimation, Overfitting, and Plasticity in Actor-Critic: The Bitter Lesson of RL"，ICML 2024：对超过 60 种 off-policy agent 配置在 14 个任务上进行系统性测试。**核心结论：通用神经网络正则化技术（LayerNorm + weight decay + 周期性 reset）的效果显著优于大多数 RL 特定算法改进。** 合理正则化的 SAC 可以匹配模型基方法的性能。

- **Sokar et al.**, "Dormant Neuron Phenomenon in Deep Reinforcement Learning"（ReDo），ICML 2023：基于 dormancy 标准（激活值长期接近零的神经元）周期性 reset 个别神经元，代价极低。

---

### 6.3 实践建议（按投入产出排序）

| 干预项 | 实现成本 | 预期收益 |
|--------|----------|----------|
| Critic 和 Actor 加 LayerNorm | 极低 | 高——稳定训练基础 |
| Critic 加 weight decay | 极低 | 中高——减缓 Q 网络退化 |
| 周期性 reset 最后 1-2 层 | 低 | 中——适合长期训练 |
| 完整全参数 reset（BRO 方式） | 中 | 高，但需精细调参 |

---

## 7. 探索与熵正则化

### 7.1 自动熵调节的局限性

原版 SAC 将目标熵设为 $-\dim(\mathcal{A})$ 并固定，自动调节 $\alpha$。这个设定简单有效，但在以下情况可能次优：

- 训练初期需要大探索（期望熵更高）
- 任务后期精细控制（期望熵更低）

**简单改进：** 目标熵退火——训练初期用较高的目标熵（如 $-0.5 \cdot \dim(\mathcal{A})$），逐步降低到 $-\dim(\mathcal{A})$。实现成本极低，值得尝试。

---

### 7.2 BRO 的乐观双 Actor 探索

如前所述，BRO 使用基于分位数 ensemble 不确定性上界的乐观探索 actor，在困难稀疏奖励任务上显著提升探索效果。对于本项目这类稠密奖励的连续控制任务，收益相对有限。

---

### 7.3 DAC：样本感知熵正则化

**论文**：Han & Sung, "Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration"，ICML 2021

**核心创新：** 标准 SAC 的熵正则化不考虑 replay buffer 中已有样本的分布。DAC 最大化策略分布与 buffer 样本分布的加权和的熵，鼓励探索 buffer 中欠代表的状态-动作空间。

在稀疏奖励场景和 buffer 存在严重分布不均时有效。对本项目的稠密奖励任务，收益较为有限。

---

## 8. 局部可观测性（无 RNN）

### 8.1 观测堆叠（Frame Stacking）

最简单的时序上下文方案。当前项目已实现 `sac_stack4`（K=4）。

**本项目的关键问题：K 应该取多大？**

本任务的流场特征频率决定了所需历史长度。根据 `RESEARCH_PLAN.md` 中的环境参数：

- 涡旋脱落 Strouhal 数 $St \approx 0.2$，Re=150-300 时涡旋脱落周期约 **10–20 秒**
- `control_dt = 0.5 s`

因此，覆盖一个完整涡旋周期需要 $K = 10\text{s} / 0.5\text{s} = 20$ 到 $40$ 帧。

**当前 K=4 只覆盖 2 秒，约为一个涡旋周期的 10-20%。** 这意味着：

- 智能体无法从观测历史中重建当前的流场相位
- 策略只能是**纯反应式**的——它看到扰动才响应，无法预见即将到来的涡旋
- 这是 MLP SAC + 观测堆叠在本任务中的**根本性局限**，与算法本身无关

增大 K 的代价：观测维度从 11 增加到 $11 \times K$（K=20 时为 220），网络输入维度增大，batch 内存和前向时间线性增加，但对当前任务来说仍是可接受的范围。

**建议：** 在保留 K=4 作为 baseline 的同时，实验 K=16 和 K=32，观察性能变化。这是比切换算法更直接的改进路径。

---

### 8.2 非对称 Critic（Asymmetric Actor-Critic）

**核心思路：** 仿真训练时 critic 使用完整的特权状态（ground truth 流场信息），actor 只使用部署时可用的观测。训练完成后 actor 独立部署，无需 critic。

**理论依据（NeurIPS 2024）：** Peng et al. 等多篇论文证明，在 POMDP 中让 critic 使用完整状态信息能显著提升 actor 学到的策略质量，因为 critic 的价值估计更准确。

**对本项目的价值极高**：

- 本项目仿真中完全可以获得真实等效流速 $u_c$（privileged）
- Actor 观测：探针采样（部分可观测）
- Critic 观测：Actor 观测 + 真实等效流速（完整流场信息）
- 零部署成本（actor 不变），训练成本也几乎不增加

这是与观测堆叠正交的改进，可以同时使用。

**实现复杂度：低**——只需为 critic 构造更大的输入，其余训练流程不变。

---

### 8.3 Transformer 策略

理论上 Transformer 可以处理变长历史，比固定窗口堆叠更灵活。但在实践中：

- 计算代价显著高于帧堆叠（注意力机制随序列长度平方增长）
- 在标准连续控制 benchmark 上，Transformer 策略并不稳定地优于 GRU
- 需要仔细的位置编码和训练稳定性设计

目前不推荐作为优先方向。

---

## 9. 技术演进路线图

| 代际 | 方法 | 发表 | UTD | 参数量 | 核心技术 |
|------|------|------|-----|--------|---------|
| 基线 | SAC | 2018 | 1 | ~100K | Twin Q + 最大熵 |
| Gen 1 | REDQ | ICLR 2021 | 20 | ~1M | N=10 Q 网络 Ensemble |
| Gen 2 | DroQ | ICLR 2022 | 20 | ~200K | Dropout + LayerNorm |
| — | Primacy Bias | ICML 2022 | — | — | 周期性 Reset |
| Gen 3 | CrossQ | ICLR 2024 | 1 | ~200K | BatchRenorm，无 target |
| Gen 4 | BRO | NeurIPS 2024 | 2–10 | ~5M | 残差网络 + 分布式 + 乐观探索 |
| — | SimBa | ICLR 2025 | 1–2 | ~1M | 简单残差块 + LayerNorm |
| Gen 5 | SimbaV2 | ICML 2025 | 2 | ~5M | 超球面归一化 + 分布式 |
| Gen 6 | XQC | ICLR 2026 | 1 | ~1M | BN+WN+CE，低条件数 |

---

## 10. 针对本项目的专业评估

### 10.1 本任务的根本挑战（算法无关）

在讨论具体算法改进之前，必须正视一个结构性问题：

**AUV 尾流导航是一个较困难的 POMDP。** 流场的特征时间尺度（10–20 秒的涡旋脱落周期）远大于单步决策间隔（0.5 秒）。即使用帧堆叠增加 K，只要 $K \ll 20$，智能体就缺乏足够的时序上下文来做流场状态推断。

这不是 SAC 的问题，也不是实现细节的问题，而是**观测设计**和**历史长度**的问题。改进算法能提高样本效率和渐近性能，但不能从根本上弥补信息不足。

因此，**增大 K（从 4 到 16/32）和引入非对称 Critic 的优先级高于切换 SAC 变体**。

---

### 10.2 推荐改进路径（按优先级排序）

**P0：几乎零成本，应立即做**

1. **LayerNorm**：对 actor 和 critic 的所有隐藏层加 LayerNorm。无负面影响，立即稳定训练。
2. **Critic weight decay**：在 critic 优化器中加入 `weight_decay=1e-4`，缓解 Q 网络退化。
3. **熵目标退火**：训练初期提高目标熵（如 $-0.5 \cdot \dim(\mathcal{A})$），后期降低到 $-\dim(\mathcal{A})$。

**P1：低成本，预期收益高**

4. **增大观测堆叠 K**：实验 K=4 → K=16 → K=32。这是最直接针对 POMDP 的改进。注意 K=32 时 obs 维度为 352，是 K=4 时的 8 倍，可能需要相应增大网络宽度。

5. **非对称 Critic（Privileged Critic）**：在训练时给 critic 提供真实等效流速作为额外输入。实现改动极小，对本任务理论收益最高。建议作为第一个"算法实质性改进"实验。

**P2：中等成本，有良好文献支撑**

6. **DroQ（Dropout + UTD 提升）**：在现有 SAC 基础上加 Dropout，UTD 从 1 提升到 5–10。计算成本增加但可控（GPU 上 UTD=10 仍然可接受）。

7. **SimBa 残差块**：将 critic 和/或 actor 的 MLP 替换为 2–3 个残差块（512 宽），加 LayerNorm。为扩大网络容量做好准备。

8. **Primacy Bias reset**：每 N 步（如每 50k 步）重置 critic 最后一层权重，保留 replay buffer。特别适合长期训练（200k+ 步）。

**P3：高收益但实现复杂，作为后续研究**

9. **CrossQ**：去掉 target network，加 BatchRenorm。UTD=1 下达到 DroQ 样本效率。当 DroQ 的计算成本成为瓶颈时再考虑。

10. **分布式 Critic（Quantile Regression）**：按 BRO/SimbaV2 的方式将 MSE Q-loss 替换为 quantile regression loss。与残差块架构配合效果最好。

---

### 10.3 不建议的方向（在本任务中）

| 方法 | 不推荐原因 |
|------|-----------|
| PER（优先经验回放） | 在稠密奖励连续控制中收益不稳定，引入额外超参数 |
| EDAC | 离线 RL 方法，在线场景中 OOD 问题不突出 |
| DAC（多样性探索） | 稠密奖励下收益有限，增加实现复杂度 |
| Transformer 策略 | 计算代价高，不稳定，收益未经验证 |
| BRO 完整版 | 实现复杂度高，可先用 BRO 的各子组件逐步集成 |

---

### 10.4 建议的消融实验设计

基于以上分析，推荐以下消融矩阵，每组 3–5 seeds：

```
基线：
  sac_k4          - K=4 帧堆叠（当前 baseline）

观测历史实验（诊断 POMDP 瓶颈）：
  sac_k16         - K=16 帧堆叠
  sac_k32         - K=32 帧堆叠
  sac_k4_asym     - K=4 + 非对称 Critic（特权流速）

架构实验：
  sac_k4_ln       - K=4 + LayerNorm（所有改进的基础）
  sac_k4_droq     - K=4 + DroQ (UTD=10)
  sac_k4_simba    - K=4 + SimBa 残差块 + LayerNorm

最优组合（根据上述结果选择）：
  sac_k16_asym_droq  - 最优 K + 特权 Critic + DroQ
```

---

## 11. 参考文献

### 核心 SAC 改进

- **SAC**：Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," ICML 2018. arXiv: 1801.01290
- **REDQ**：Chen et al., "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model," ICLR 2021. arXiv: 2101.05982
- **DroQ**：Hiraoka et al., "Dropout Q-Functions for Doubly Efficient Reinforcement Learning," ICLR 2022. arXiv: 2110.02034
- **CrossQ**：Bhatt et al., "CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity," ICLR 2024. OpenReview: PczQtTsTIX
- **BRO**：Nauman et al., "Bigger, Regularized, Optimistic: Scaling for Compute and Sample-Efficient Continuous Control," NeurIPS 2024 Spotlight. arXiv: 2405.16158

### 网络架构

- **SimBa**：Lee et al., "SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning," ICLR 2025. arXiv: 2410.09754
- **SimbaV2**：Lee et al., "Hyperspherical Normalization for Scalable Deep Reinforcement Learning," ICML 2025 Spotlight. arXiv: 2502.15280
- **XQC**：Palenicek et al., "XQC: Well-Conditioned Optimization Accelerates Deep Reinforcement Learning," ICLR 2026. arXiv: 2509.25174
- **Spectral Norm (NeurIPS)**：Bjorck et al., "Towards Deeper Deep RL with Spectral Normalization," NeurIPS 2021
- **Spectral Norm (ICML)**：Gogianu et al., "Spectral Normalisation for Deep RL: an Optimisation Perspective," ICML 2021

### Plasticity 与训练稳定性

- **Primacy Bias**：Nikishin et al., "The Primacy Bias in Deep Reinforcement Learning," ICML 2022. arXiv: 2205.07802
- **ReDo（Dormant Neuron）**：Sokar et al., "Dormant Neuron Phenomenon in Deep Reinforcement Learning," ICML 2023
- **Normalization & Plasticity**：Lyle et al., "Normalization and Effective Learning Rates in Reinforcement Learning," NeurIPS 2024. arXiv: 2407.01800
- **Bitter Lesson of RL**：Nauman et al., "Overestimation, Overfitting, and Plasticity in Actor-Critic: The Bitter Lesson of RL," ICML 2024. arXiv: 2403.00514

### Critic 改进

- **DSAC**：Ma et al., "DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning," arXiv: 2004.14547, 2020
- **EDAC**：An et al., "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble," NeurIPS 2021. arXiv: 2110.01548

### 数据利用

- **RLPD**：Ball et al., "Efficient Online Reinforcement Learning with Offline Data," ICML 2023. arXiv: 2302.02948
- **PER**：Schaul et al., "Prioritized Experience Replay," ICLR 2016. arXiv: 1511.05952
- **DPER**：Shi et al., "Decoupled Prioritized Experience Replay for Deep Reinforcement Learning," arXiv: 2024

### 探索

- **DAC**：Han & Sung, "Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration," ICML 2021

### 局部可观测性

- **Asymmetric Actor-Critic**：Pinto et al., "Asymmetric Actor Critic for Image-Based Robot Learning," RSS 2018；Moritz et al., NeurIPS 2024 (provable POMDP with privileged info)
- **TD3**：Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods," ICML 2018（Clipped Double Q 的来源）
