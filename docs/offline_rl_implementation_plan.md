# 离线强化学习实现方案与计划

> 文档版本：2026-04-22 rev.5
> 适用范围：当前仓库的 SAC / RLPD 基础设施，以及截至 `phase0c` 已经落地的 TD3BC 离线实验主线
> 核心结论：**实现优先级仍应保持 `TD3+BC -> ReBRAC -> XQL -> FQL`；截至当前，TD3BC 主线收口已基本完成，下一阶段的重点应转向 ReBRAC，并以现有 teacher-gap 与数据支持集结论作为约束**
> 交叉验证来源：原始论文 + 官方代码仓库 + 当前 repo 现有实现接口 + 仓库内 `phase0b_v2` / `phase0c` 正式实验结果

---

## 0. 当前执行状态（截至 `phase0c`）

这份计划最初写于离线 RL 代码尚未完全落地、实验尚未形成闭环的时候。到当前仓库状态为止，需要先明确一件事：

> 计划主线没有失效，但项目已经不再处在“准备做 Phase 0”的阶段，而是已经完成了 Phase 0/1 的主体工作，并进入“收口 TD3BC、准备下一算法”的阶段。

当前状态可概括为：

| 模块 | 当前状态 | 说明 |
|---|---|---|
| Phase 0 基础设施 | **已完成** | `train_offline.py`、`evaluate_offline.py`、`evaluate_baseline_on_manifest.py`、TD3BC agent、normalizer 持久化、deployable / privileged-critic 协议均已落地 |
| Phase 1 TD3BC deployable 主线 | **已完成** | `phase0`、`phase0b_v2`、`phase0c` 已形成从协议修正到正式结果的完整链路 |
| `crosscomp` 主结论 | **已完成** | TD3BC 在 deployable 数据上已证明相对 BC 的价值；当前最优数据规模在 `1000` episodes 左右 |
| `worldcomp` teacher-gap 主线 | **已完成** | 已形成正式 screening / formal 结果与专项实验报告，证明 privileged critic 可关闭约一半 deployable teacher gap |
| 数据支持集扩展 | **部分完成** | collector 已支持动作噪声，且最小 noisy-support 诊断已完成；mixed-deployable 与更系统的数据线仍未完成 |
| ReBRAC / XQL / FQL | **未开始** | 目前仓库中还没有对应实现与结果目录 |

因此，本文档后续各章节应这样理解：

- 涉及 Phase 0 / Phase 1 的段落，属于“已经被执行并验证过的计划”；
- 涉及 ReBRAC / XQL / FQL 的段落，仍然是当前有效的下一阶段路线；
- 涉及数据支持集、teacher-gap、holdout 的段落，是理解当前 TD3BC 结果并约束后续算法设计的关键背景。

## 一、背景与目标

### 1.1 为什么当前仓库需要纯离线 RL

本仓库已具备两类方法：

| 方法 | 类型 | 训练时是否与环境交互 | 当前状态 |
|---|---|---|---|
| SAC | 在线 RL | 是 | 已实现 |
| RLPD | 离线-在线混合 | 是 | 已实现 |

但真实 AUV 场景通常不允许频繁在线探索，因此仍缺少第三类方法：

| 方法 | 类型 | 训练时是否与环境交互 | 目标 |
|---|---|---|---|
| 纯离线 RL | Offline RL | **否** | 从固定历史数据中提取策略 |

纯离线 RL 在本项目中的研究价值不是“打败在线 RL”，而是回答：

> 在完全不与环境交互的约束下，AUV 能从固定历史流场导航数据中学到多强的策略？

### 1.2 当前任务对 offline RL 的真实难点

本问题的难点并不只有“动作多模态”：

- 流场强、时变、非均匀；
- 逆流任务接近欠驱动临界；
- 观测只有局部探针，部分可观测性明显；
- 行为策略数据可能带有 teacher 的额外信息；
- 固定数据集覆盖率有限，OOD backup 风险高。

因此，完整方案必须同时处理五件事：

1. **离线算法本身是否稳。**
2. **数据协议是否锁死。**
3. **评估基础设施是否与当前 repo 兼容。**
4. **teacher 信息差是否被单独识别。**
5. **状态归一化、checkpoint schema、训练/评估接口是否真正闭环。**

---

## 二、总路线：先把问题拆对，再选算法

### 2.1 修订后的优先级

建议的实现顺序：

1. **TD3+BC**：最小 sanity baseline
2. **ReBRAC**：最强 minimalist baseline
3. **XQL**：严格 in-sample 的主离线 RL 基线
4. **FQL**：在前面三者建立基线后，再验证 expressive policy family 是否真有额外价值

### 2.2 为什么不建议直接把 FQL 设为第一主线

FQL 很强，也很新，但它不是最适合作为本仓库离线 RL 第一实现的原因有三点：

- 它实现最复杂，需要新增 flow-matching policy、one-step policy 和蒸馏逻辑。
- 它**不是**严格的 in-sample 方法，critic target 仍会评估 one-step actor 生成的动作。
- 当前任务的第一瓶颈未必是动作多模态，更可能是数据覆盖、部分可观测性和 teacher 信息差。

因此，FQL 更适合作为：

- 在简单方法建立可信基线后，
- 专门验证“flow-based behavior support + expressive policy family”是否带来额外收益

的高级实验，而不是第一块落地代码。

---

## 三、各算法的严格审查

本节结论都已经用原始论文和官方实现交叉核对。

### 3.1 TD3+BC

**来源**

- 论文：[A Minimalist Approach to Offline Reinforcement Learning](https://openreview.net/forum?id=Q32U7dzWXpc)
- 官方代码：[sfujim/TD3_BC](https://github.com/sfujim/TD3_BC)

**核心事实**

- TD3+BC 的关键变化只有两点：
  - actor loss 加 BC 正则；
  - **状态归一化**。
- 原论文明确把 “normalizing the data” 作为 TD3+BC 的组成部分，而不是可有可无的小技巧。
- 它的价值主要是：用最小改动给出一个很强的 offline RL 基线。

**对本仓库的启示**

- 如果只想做第一阶段 sanity check，TD3+BC 仍然是最合适的起点。
- 但不要把“只加 BC 项，不做输入归一化”称为“原始 TD3+BC 复现”；那是一个 AUV 版简化变体。
- 如果 checkpoint 不保存状态归一化统计量，训练和评估就不算真正闭环。

### 3.2 ReBRAC

**来源**

- 论文：[Revisiting the Minimalist Approach to Offline Reinforcement Learning](https://openreview.net/forum?id=vqGWslLeEw)
- 官方代码：[DT6A/ReBRAC](https://github.com/DT6A/ReBRAC)

**必须纠正的地方**

ReBRAC **不是** “TD3+BC + ensemble critic + observation normalization”。

论文和官方实现表明，ReBRAC 的主线是：

- 更深的网络；
- critic 中使用 LayerNorm；
- 保留 TD3+BC 的 Q normalization；
- **actor penalty + critic penalty 双通道正则**；
- 在 AntMaze 这类稀疏任务上提高 discount；
- 目标是成为更强的 **ensemble-free** minimalist offline RL 方法。

论文中的 actor / critic 目标明确包含两种 penalty，而不是“通过 `ensemble_size > 2` 变成 ReBRAC”。

**对本仓库的启示**

- 如果要实现 ReBRAC，应该把它理解成“带 actor/critic dual penalties 的强基线”。
- 在本项目里，ReBRAC 比 XQL 更接近 TD3+BC 的工程演化线，适合紧跟 TD3+BC 实现。
- 不建议把 ReBRAC 误实现成“TD3+BC 的多 Q ensemble 版本”。

### 3.3 XQL

**来源**

- 论文：[Extreme Q-Learning: MaxEnt RL without Entropy](https://openreview.net/forum?id=SJ0Lde3tRL)
- 官方代码：[Div-Infinity/XQL](https://github.com/Div-Infinity/XQL)

**核心事实**

- XQL 的理论卖点是：直接拟合 soft-optimal value / log-sum-exp，不需要对 actor 采样动作来算 backup。
- 论文明确强调它避免了 OOD action evaluation，是典型的 **in-sample** 路线。
- “不显式依赖 policy 或 entropy”是理论层面的主张。
- 但在 offline continuous control 里，实际部署仍通常需要一个 policy extraction 步骤；AWR-style actor 是合理工程选择。

**对本仓库的启示**

- 如果你想先得到一个“理论上更干净、更适合离线 setting”的主离线 RL 方法，XQL 比 FQL 更应该先落地。
- 文档里若给出 XQL actor 实现，最好明确说明：
  - `V/Q` 学习遵循 XQL；
  - actor extraction 使用 advantage-weighted BC，是工程化部署步骤，而不是 XQL 理论核心本身。

### 3.4 FQL

**来源**

- 论文：[Flow Q-Learning](https://arxiv.org/abs/2502.02538)
- 官方代码：[seohongpark/fql](https://github.com/seohongpark/fql)

**必须纠正的地方**

FQL 的强项是真实存在的，但它**不是**严格的 “no OOD action query” 方法。

更准确的描述应该是：

- BC flow policy 只负责建模数据分布；
- RL 优化的是 one-step policy；
- critic target 使用 one-step policy 在 `s'` 上生成的动作；
- 因此它仍然会对策略生成动作做 backup，只是通过 flow-based behavior support 和蒸馏显著降低 OOD 风险。

官方实现也印证了这一点：

- `next_actions = self.sample_actions(batch['next_observations'], ...)`
- 再用这些动作做 target critic 计算。

所以：

- **XQL 是严格 in-sample**
- **FQL 不是**
- FQL 的优势在于更强的行为分布建模和更稳的 policy parameterization，而不是完全消灭 OOD backup

**对本仓库的启示**

- 如果未来实验发现 `MLP BC << Flow BC`，且 XQL / ReBRAC 仍受限，那才是 FQL 最有价值的切入点。
- 在此之前，把 FQL 定位成“主算法”会高估多模态动作分布在当前任务中的主导性。

---

## 四、当前 repo 下必须先解决的基础设施问题

这部分比算法更重要，因为它决定实验结论是否可信。

### 4.1 现有 `scripts/evaluate.py` 不能直接复用于 offline agents

当前 evaluator 是 SAC 专用：

- 硬编码 `SACAgent`
- 用 `trainer_state["agent_config"]` 直接构造 `SACConfig`

因此不能再写“现有 `scripts/evaluate.py` 直接复用”。

**建议**

新增：

```text
scripts/evaluate_offline.py
scripts/evaluate_baseline_on_manifest.py
```

其中：

- `evaluate_offline.py` 负责评估离线 agent；
- `evaluate_baseline_on_manifest.py` 用同一个 manifest 评估基线 teacher；
- 可以复用的是 [`scripts/train_utils.py`](../scripts/train_utils.py) 中的 `evaluate_agent()`、`make_planar_env()` 和 manifest 工具，而不是现有 CLI。

### 4.2 holdout 机制当前没有真正闭环

当前 [`TransitionReplay.sample_batch()`](../auv_nav/replay.py) 只支持对全 buffer 均匀采样。

所以单纯生成：

```python
holdout_indices = ...
train_indices = ...
```

并不能真正把 holdout 样本排除出训练。

**建议**

- 在没有 index-aware sampler 之前，不把 `holdout_td_error` 写成主诊断指标；
- 或者新增一个 `IndexedTransitionReplay` / `ReplayView`，显式支持 train/holdout 子集采样。

### 4.3 benchmark protocol 需要真正锁死

当前 repo 的标准 manifest 会冻结：

- flow case
- start / goal / flow_time
- seeds

但不会冻结：

- `probe_layout`
- `history_length`

这点在 [`benchmarks/README.md`](../benchmarks/README.md) 中已经明确写出。

因此离线 RL 实验必须额外锁死：

- `probe_layout`
- `history_length`
- `objective`

否则“同一 benchmark”会变成多个不同协议。

### 4.4 canonical flow case 必须统一

文档和命令不能在不同 benchmark 之间来回漂移。既然当前 protocol screening 已经由 [`scripts/run_stage_a0_layout_screen.sh`](../scripts/run_stage_a0_layout_screen.sh) 固化为：

- `single_u10_cross_tgt15`
- `wake_v8_U1p00_Re150...`
- `history_length = 4`

那么 offline RL 第一阶段就应直接对齐这套协议，而不是另起一条 `Re250 upstream` 主线。

### 4.5 状态归一化统计量必须进入 checkpoint schema

这条对 TD3+BC 和 ReBRAC 都是硬要求。

如果训练时做了 observation normalization，而 checkpoint 没有保存：

- `obs_mean`
- `obs_std`
- 归一化是否启用

那么：

- 评估无法与训练口径一致；
- `alpha=0` 的 BC 基线也会被错误实现；
- “复现实验”会退化成“复现一个未知 normalizer”。

---

## 五、数据策略需要重新分类

### 5.1 不要把“teacher 强”与“数据质量高”混为一谈

当前 repo 中的数据源至少分两类：

| 类别 | 策略 | 特点 |
|---|---|---|
| Deployable teacher | `goalseek`, `crosscomp` | 只依赖可部署观测或局部启发式 |
| Privileged teacher | `worldcomp`, `privileged` | 依赖环境内部流信息，带特权先验 |

因此：

- `worldcomp-500` 不能简单称为“高质量主实验数据”；
- `privileged-500` 不能简单称为“数据上界”；
- 它们更准确的角色是：**带 teacher 信息优势的数据源**。

### 5.2 修订后的数据集角色划分

建议按下面分层：

| 数据集 | teacher 类型 | 角色 |
|---|---|---|
| `crosscomp-500` | deployable | **Phase 0 / 1 主数据集，主结论默认在这里报告** |
| `worldcomp-500` | privileged | 二级数据集，用于分析局部 teacher-gap |
| `privileged-500` | privileged | 强特权 upper bound，只能当作上界分析 |
| `mixed-deployable-500` | deployable mix | 覆盖率消融 |
| `mixed-all-500` | mixed | 探索 teacher mixture 是否有益 |

核心原则：

- **“纯 offline RL 是否对可部署策略有价值”必须先在 `crosscomp` 这类 deployable 数据上回答。**
- `worldcomp` 和 `privileged` 更适合回答“如果 teacher 有额外信息，offline agent 离 teacher 还有多远”。

### 5.3 单一确定性 teacher 数据会窄化支持集

当前 [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) 默认仍执行确定性 baseline，不做 multi-policy mixing；但与本计划初稿不同，代码中已经支持 `--action-noise-std` 与 `--action-noise-clip`。基于这些接口的最小 noisy-support 诊断已经完成；真正尚未完成的是把这些接口进一步系统地用于更大规模的 noisy / broadened-support / mixed-deployable 数据实验。

这意味着单一 teacher 数据很可能出现：

- 支持集过窄；
- recovery transition 不足；
- `TD3+BC` 与 `BC` 难以拉开；
- 一旦训练失败，很难判断是算法问题还是数据覆盖问题。

**建议**

- 第一轮 Phase 0 可以先用确定性 `crosscomp-500` 跑通链路；
- 但如果 `alpha=0` 与 `alpha>0` 难分高下，优先扩数据支持集，而不是直接升级到 FQL；
- 最小可行的扩展方向现在应改成：
  - 用已经落地的 `--action-noise-std` 生成 noisy `crosscomp` 数据；
  - 或支持 `mixed-deployable` 数据收集；
  - 或显式注入 recovery transitions。

---

## 六、`privileged_obs` 与 asymmetric critic 的协议必须先锁死

### 6.1 当前 repo 的 `privileged_obs` 到底是什么

当前环境在 [`auv_nav/env.py`](../auv_nav/env.py) 中输出的 `privileged_obs` 是：

- body-frame equivalent flow 的前两维；
- 即 `equivalent_body[:2] = [u_eq, v_eq]`。

这很重要，因为它只对应：

- 当前时刻、当前位置的等效流信息；
- 不等于完整 wake field；
- 也不等于 future corridor cost。

### 6.2 这和各类 teacher 的关系并不相同

- 对 `worldcomp` 而言，`privileged_obs` 与其使用的 `last_equivalent_current_world` 是**强相关、近似同源**的信息，只差坐标系。
- 对 `privileged` 而言，`privileged_obs` **并不覆盖**其 corridor scoring 所需的空间流场访问能力。

因此：

- asymmetric critic 可能显著缩小 `worldcomp` gap；
- 但**不能**被描述成“足以消除 `privileged` teacher 的信息优势”。

### 6.3 建议把实验口径拆成两条，且必须分开报告

#### Protocol A：Deployable mainline

- actor 只看普通 `obs`
- critic 也只看普通 `obs`
- 这是默认主结果，必须在 `crosscomp` 数据上建立

#### Protocol B：Privileged-critic ablation

- actor 只看普通 `obs`
- critic 在训练中可以读取 `privileged_obs`
- 结果必须明确标注为 `privileged-critic`

**禁止做法**

- 不加标签地把 Protocol A 和 B 的结果放在同一个主表里比较；
- 把 `worldcomp` / `privileged` 数据上的结果直接解释为“offline RL 对 deployable policy 的真实上界”；
- 把当前 repo 中 SAC 的 asymmetric critic 细节原封不动视为 offline RL 的默认语义。

换句话说，Phase 0 / 1 不应默认把“使用 privileged critic”与“纯 deployable offline RL”混为一谈。

---

## 七、修订后的算法实施路线

### 7.1 Phase 0：基础设施与协议锁定 `【已完成】`

目标：

- 新增 `scripts/train_offline.py`
- 新增 `scripts/evaluate_offline.py`
- 新增 `scripts/evaluate_baseline_on_manifest.py`
- 锁死 canonical benchmark protocol
- 明确 `trainer_state.json` schema
- 明确 normalizer schema
- 明确 privileged-critic protocol

这一层在当前仓库中已经基本完成。对应的核心落地点包括：

- [`scripts/train_offline.py`](../scripts/train_offline.py)
- [`scripts/evaluate_offline.py`](../scripts/evaluate_offline.py)
- [`scripts/evaluate_baseline_on_manifest.py`](../scripts/evaluate_baseline_on_manifest.py)
- [`auv_nav/td3bc.py`](../auv_nav/td3bc.py)

因此，后续讨论不再以“能否跑通离线训练链路”为主问题，而以“在已锁死协议上，哪类算法与数据策略更有效”为主问题。

### 7.2 Phase 1：TD3+BC `【已完成 deployable 主线】`

目标：

- 用最小方法建立 offline RL 下界
- 先确认纯离线训练链路本身是否工作

Phase 1 的主问题不是“超过 `worldcomp` 没有”，而是：

- 在 `crosscomp-500` 上是否稳定优于 pure BC；
- 是否在固定 manifest 上可复现；
- 是否已经出现明显的 OOD backup 问题；
- 状态归一化和 checkpoint 是否真正闭环。

截至 `phase0c`，这些问题已经有了明确答案：

- deployable `crosscomp` 主线已经证明 TD3BC 相对 BC 具有真实价值；
- 旧 `phase0` 中“数据越大越差”是协议伪象，已被 `phase0b_v2` 修正；
- 在更正式的 `phase0c` 中，当前最优数据规模在 `1000` episodes 左右，而不是单调偏向更大数据集。
- `worldcomp teacher-gap` 也已经完成，证明 privileged critic 可以关闭约一半 deployable teacher gap。

因此，Phase 1 现在应视为**已完成并沉淀出正式结论**，而不是待执行任务。

### 7.3 Phase 2：ReBRAC `【当前最优先下一阶段】`

目标：

- 在不引入更复杂行为模型的前提下，建立更强的 minimalist baseline
- 检查 TD3+BC 的失败是否来自 baseline 太弱，而不是来自任务本身

这里的重点依然是：

- deployable dataset 上的正增益；
- 而不是 privileged teacher 上的追分能力。

在当前项目状态下，我建议把 ReBRAC 设为**下一步最高优先级**，原因是：

- `phase0c` 已经表明问题不在“TD3BC 完全无效”，而在“更大数据集没有继续转化成收益”；
- 这更像 critic regularization / dual penalty / optimization regime 的问题，而不像必须立刻升级到更复杂 actor family 的问题；
- ReBRAC 正好是对 TD3BC 的最自然、最可解释的下一层增强。

### 7.4 Phase 3：XQL `【ReBRAC 之后的主候选】`

目标：

- 建立本项目中最强的 in-sample 离线 RL 基线
- 为后续 FQL 提供真正可比较的对照

如果 XQL 已经很好：

- 说明本问题的第一瓶颈未必是 policy multimodality；
- 那么 FQL 的收益未必能 justify 它的复杂度。

换句话说，XQL 的进入条件并不是“TD3BC 已经做完，所以顺序轮到它”，而是：

- ReBRAC 仍不能解释或改善 `1000 > 2000` 的现象；
- 我们需要一个更强的 in-sample 基线来区分“critic backup 问题”和“数据支持集/行为建模问题”。

### 7.5 Phase 4：FQL `【继续后置】`

只有在满足以下条件之一时，才建议推进 FQL：

- `Flow BC` 明显优于 `MLP BC`
- XQL / ReBRAC 在 deployable datasets 上受限明显
- 误差分析显示行为分布建模能力，而不是 value backup 稳定性，才是主要瓶颈

FQL 的问题定义应改成：

> 在行为支持受限、动作分布可能多模态的状态下，flow-based actor family 是否比 Gaussian / AWR actor 带来额外收益？

而不是：

> FQL 是否天然比 XQL 更适合作为主方法？

### 7.6 Phase 5：数据与泛化 `【优先级前移】`

建议把数据消融分成两条主线：

1. **coverage line**
   - `crosscomp`
   - `mixed-deployable`
2. **teacher-gap line**
   - `worldcomp`
   - `privileged`

这样可以把“覆盖率不足”与“teacher 特权信息”分开分析。

与本计划初稿相比，这一阶段现在应当**前移优先级**。原因是 `phase0c` 已经给出一个很强的信号：

- `2000` 比 `1000` 更大，但并没有更好；
- 当前解释最像“数据支持集更宽，但算法没有有效消化这些额外样本”。

因此，数据与泛化不再只是“后面有空再做”的延伸题，而是理解当前 TD3BC 结果的关键诊断线。

泛化实验的触发条件也不应建立在“必须显著超过 `worldcomp`”之上。

更合理的触发条件是：

- 某个 offline 方法在 canonical benchmark 上稳定优于 BC；
- 且在 deployable teacher 数据上已经显示出真实增益。

---

## 八、推荐的 canonical 实验协议

### 8.1 第一阶段推荐协议

为了尽快建立可信基线，并与当前 layout screen 主轨道保持一致，建议 Phase 0 / 1 / 2 统一采用：

- `flow = wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`
- `manifest = benchmarks/single_u10_cross_tgt15.json`
- `probe_layout = s0`
- `history_length = 4`
- `task_geometry = cross_stream`
- `target_speed = 1.5`
- `objective = efficiency_v2`

这里显式采用 `history_length = 4`，不是因为它是脚本默认值，而是因为它已经被 [`scripts/run_stage_a0_layout_screen.sh`](../scripts/run_stage_a0_layout_screen.sh) 作为 `s0/s1/s2` 比较时的标准观测协议。

因此：

- collect
- train
- evaluate

三段都必须显式写 `history_length = 4`，不能依赖默认值。

### 8.2 评估 protocol

所有可报告指标都应来自：

- 同一个 benchmark manifest
- 同一个 flow case
- 同一个 `probe_layout/history_length/objective`

建议保留：

- `eval_success_rate`
- `eval_return`
- `eval_safety_cost`
- `eval_time_s_success`
- `eval_path_efficiency`

并附带：

- 数据集标签（`crosscomp` / `worldcomp` / `privileged`）
- 协议标签（`deployable` / `privileged-critic`）

---

## 九、监控与诊断

### 9.1 主指标

| 指标 | 优先级 | 说明 |
|---|---|---|
| `eval_success_rate` | P0 | 主结论指标 |
| `eval_return` | P0 | reward 语义校验 |
| `mean_q` | P0 | Q 高估探针 |
| `critic_loss` | P1 | 训练稳定性 |
| `actor_loss` | P1 | 更新方向参考 |
| `bc_loss` | P1 | TD3+BC / ReBRAC 偏离行为策略的程度 |
| `v_loss` | P1 | 仅 XQL |
| `bc_flow_loss` / `distill_loss` | P1 | 仅 FQL |

### 9.2 暂不建议在主文档里承诺的指标

以下指标只有在补齐实现后才建议纳入主结论：

- `holdout_td_error`
- `teacher-policy gap decomposition`
- `flow multimodality metrics`

否则容易写成“文档里有，实验里其实没有”。

### 9.3 诊断顺序建议

若 `TD3+BC` 没有稳定优于 BC，优先按以下顺序排查：

1. `obs` 归一化是否正确实现并保存
2. 数据协议是否完全一致
3. 数据支持集是否过窄
4. 是否误把 privileged 结果与 deployable 结果混在一起
5. 是否真的需要更强算法

不要把“没赢 `worldcomp`”当作第一诊断入口。

---

## 十、安全成本集成的建议

当前 repo 已有：

- reward 中的 safety shaping
- replay 中的 `costs`

因此可以做两层安全信号：

1. reward 通道：通过 `batch["rewards"]` 学到
2. auxiliary weighting 通道：在 actor / distillation loss 中对高 cost transition 降权

但建议把第二层写成**可选 ablation**，而不是默认主线。原因是：

- `efficiency_v2` 已经含弱 safety shaping；
- 再叠加 cost-weighted actor regularization，容易把保守性和算法能力混在一起。

---

## 十一、最终建议

### 11.1 这份计划现在最重要的修订结论

- **FQL 不是严格 in-sample 方法**，不能与 XQL 按同一口径描述。
- **ReBRAC 不是 ensemble critic 版本的 TD3+BC**，核心是 actor/critic dual penalties 和一组已验证的设计选择。
- **现有 `scripts/evaluate.py` 不能直接复用**，必须新增 offline evaluator。
- **`crosscomp` 应该是 Phase 0 / 1 的主数据集**，`worldcomp` 只能作为二级 teacher-gap 分析。
- **当前 `privileged_obs` 只对应局部等效流，不等于完整特权流场信息**。
- **算法优先级应调整为 `TD3+BC -> ReBRAC -> XQL -> FQL`**。
- **截至 `phase0c`，Phase 0 / 1 已基本完成；当前最优 deployable TD3BC 配置出现在 `1000` episodes 左右，而不是单调偏向更大数据集。**
- **下一阶段不应直接跳到 FQL；更合适的顺序是：在现有 teacher-gap / 数据支持集结论约束下，实现 ReBRAC。**

### 11.2 如果只保留一句路线建议

> 以已经完成的 TD3BC 主线收口结果为约束，先用 ReBRAC / XQL 在同一 deployable protocol 上建立更强基线，最后才让 FQL 去回答“flow-based expressive actor 是否真的值得”。

---

## 十二、参考资料

### 论文

- TD3+BC: [A Minimalist Approach to Offline Reinforcement Learning](https://openreview.net/forum?id=Q32U7dzWXpc)
- ReBRAC: [Revisiting the Minimalist Approach to Offline Reinforcement Learning](https://openreview.net/forum?id=vqGWslLeEw)
- XQL: [Extreme Q-Learning: MaxEnt RL without Entropy](https://openreview.net/forum?id=SJ0Lde3tRL)
- FQL: [Flow Q-Learning](https://arxiv.org/abs/2502.02538)

### 官方代码

- TD3+BC: [sfujim/TD3_BC](https://github.com/sfujim/TD3_BC)
- ReBRAC: [DT6A/ReBRAC](https://github.com/DT6A/ReBRAC)
- XQL: [Div-Infinity/XQL](https://github.com/Div-Infinity/XQL)
- FQL: [seohongpark/fql](https://github.com/seohongpark/fql)
