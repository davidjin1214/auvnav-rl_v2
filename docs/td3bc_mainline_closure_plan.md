# TD3BC 主线收口执行计划

> 文档版本：2026-04-22 rev.4
> 适用范围：总结 `phase0b_v2`、`phase0c` 及其补充实验包的收口结果，为后续 ReBRAC / XQL 提供干净基线
> 当前前提：请先阅读 [td3bc_phase0b_v2_experiment_report.md](./td3bc_phase0b_v2_experiment_report.md)、[td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md) 与 [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)
> 当前定位：本文件已从“待执行计划”转为“收口档案与阶段过渡说明”；三个实验包均已完成，当前主要用于统一解释 TD3BC 主线为何可以在此处停止扩展并转向 ReBRAC。

---

## 1. 为什么现在要先收口 TD3BC

截至 `phase0c`，当前仓库已经得到三个高价值结论：

1. `crosscomp` deployable 主线上，TD3BC 已经证明相对 BC 有真实价值。
2. 旧 `phase0` 中“数据越大越差”的现象主要是协议伪象。
3. 在更正式的 `phase0c` 协议下，当前最优数据规模在 `1000` episodes 左右，而不是单调偏向更大数据集。

截至当前版本，这条主线已经完成收口。  
之所以仍保留这份文档，是为了把三组补充实验包的作用与最终结论统一起来，避免后续进入 ReBRAC 时重新引入口径混乱。

当前最稳妥的顺序已经从：

> 先用有限预算把 TD3BC 主线收口，再进入 ReBRAC。

更新为：

> TD3BC 主线已完成收口，下一阶段应进入 ReBRAC。

---

## 2. 收口目标

本轮收口工作的目标不是“再做一轮大规模 TD3BC 实验”，而是用**最小必要实验包**补齐证据链。

收口完成的判据应是：

1. `phase0c Stage C` 上至少 `1000` 与 `2000` 具备同预算 BC 正式对照。
2. `worldcomp` 数据上至少完成一条正式 `deployable vs privileged-critic` teacher-gap 对照。
3. 对 `1000 > 2000` 的现象，至少完成一轮最小 noisy-support 诊断，能够初步区分：
   - 数据支持集问题
   - critic regularization / optimization 问题
4. 补一份汇总文档，把上述三项结果与现有 `phase0c` 主结果合并解释。

换句话说，这次收口的结束条件不是“所有问题都彻底搞懂”，而是：

> 已经把下一阶段进入 ReBRAC 所需的主要混杂因素降到可接受范围。

当前状态更新：

- 实验包 A 已完成
- 实验包 B 已完成
- 实验包 C 已完成
- TD3BC 主线收口目标已达成

---

## 3. 总体执行顺序

推荐执行顺序已经完成。  
如果按信息价值排序回顾，本轮收口的最优顺序确实是：

1. 实验包 A：先补正式 BC 对照
2. 实验包 B：再做最小 noisy-support 诊断
3. 实验包 C：最后完成 `worldcomp` teacher-gap

这三步已经完成，因此当前不再建议继续扩展 TD3BC 收口矩阵。

---

## 4. 实验包 A：Stage C 同预算 BC 正式对照

> 状态：已完成

### 4.1 目标

补齐 `phase0c` 的一个核心缺口：

> 让 `TD3BC vs BC` 的正式结论不再只依赖 Stage B screening，而是具备与 Stage C 相同预算、相同 seed 数、相同 test manifest 的对照。

### 4.2 为什么它优先级最高

当前 `phase0c` 的正式主表已经有：

- `500 -> alpha=0.5`
- `1000 -> alpha=0.25`
- `2000 -> alpha=0.15`

但没有同预算 `alpha=0` 正式 run。  
这会导致一个明显的问题：你可以说“TD3BC 当前主结果最好在 `1000`”，但还不能同样强地说“TD3BC 在正式主表层面稳定优于 BC”。

### 4.3 当时采用的实验矩阵

优先补两个规模：

| 数据集 | 优先级 | 理由 |
|---|---|---|
| `crosscomp-1000` | P0 | 当前正式最优规模，必须有 BC 正式对照 |
| `crosscomp-2000` | P0 | 当前最有争议的规模，必须知道 BC 在同预算下能到哪里 |
| `crosscomp-500` | P1 | 可补，但信息价值低于 `1000/2000` |

统一协议：

- 数据集：沿用 `phase0c stage_c_final`
- `alpha = 0.0`
- seeds：`42 / 43 / 44 / 45 / 46`
- `TRAIN_EPOCHS = 96`
- `CHECKPOINT_EVERY_EPOCHS = 4`
- `VAL_MANIFEST_EPISODES = 40`
- `TEST_MANIFEST_EPISODES = 100`
- `SAMPLING_MODE = shuffle_no_replacement`
- protocol：`deployable`

### 4.4 输出要求

实际运行中，这组对照已经写入与 `stage_c_final` 平行的独立目录：

- `results/offline/td3bc/phase0c/stage_c_bc_final`
- `checkpoints/offline/td3bc/phase0c/stage_c_bc_final`

仓库中已经补了一个专门的执行入口：

- [`scripts/run_offline_td3bc_phase0c_stage_c_bc.sh`](../scripts/run_offline_td3bc_phase0c_stage_c_bc.sh)

该脚本默认：

- 复用 `phase0c/stage_c_final` 的 manifest
- 只跑 `alpha=0.0`
- 默认补 `1000` 与 `2000`
- 使用与 `stage_c_final` 对齐的 5-seed / 96-epoch / 100-test-episode 预算

### 4.5 已完成结果摘要

`stage_c_bc_final` 已经补齐，并给出如下正式结果：

| 数据集 | BC success | TD3BC success | TD3BC - BC |
|---|---:|---:|---:|
| 500 | 0.524 | 0.592 | +0.068 |
| 1000 | 0.588 | 0.672 | +0.084 |
| 2000 | 0.534 | 0.596 | +0.062 |

这说明：

1. 在与正式主表同预算的条件下，TD3BC 在三个规模上都优于 BC。
2. `1000 > 2000` 在 BC 下同样成立，因此该现象不是 TD3BC 特有问题。
3. `2000` 的问题更像数据支持集/样本利用方式层面的更一般现象，而不是单纯的 TD3BC Q 项失效。

### 4.6 当前结论

实验包 A 已经完成，而且它的核心结论在后续实验包 B、C 中都得到了更强支撑：

- `1000 > 2000` 不是 TD3BC 特有问题；
- `2000` 的一部分问题来自 deterministic 大数据支持集结构；
- 剩余的一部分 gap 则来自 `worldcomp` 下真实存在的 teacher information gap。

因此，实验包 A 现在更适合作为主线收口中的“正式 BC 对照证据”，而不再是需要继续扩展的单独分支。

---

## 5. 实验包 B：最小 noisy-support 诊断

> 状态：已完成

### 5.1 目标

这个实验包只回答一个问题：

> `2000` 之所以没有优于 `1000`，是不是因为当前确定性 `crosscomp` 数据支持集过窄，而不是因为单纯需要更强算法？

### 5.2 为什么当时需要这个实验包

在实验包 B 启动之前，`phase0c` 最关键、但尚未被实验验证的机制性推断是：

- `2000` 数据更大，但引入了更宽的状态支持集；
- 当前 TD3BC 的统一 sample weighting / current objective 没有把这些额外样本有效转化成收益；
- 最优 checkpoint 提前出现，说明问题不像“训练 epoch 不够”。

这条解释当时还只是**推断**。  
回头看，先做一轮最小支持集扩展实验，而不是立刻换算法，是正确的执行顺序。

### 5.3 当时采用的数据策略

实验包 B 实际上只做了 `crosscomp`，且只覆盖两个规模：

| 数据集 | 必做 | 理由 |
|---|---|---|
| `crosscomp-1000` | 是 | 当前最优规模，需要作为稳定参照 |
| `crosscomp-2000` | 是 | 当前最需要解释的规模 |

实际采用的温和噪声版本为：

- `action_noise_std = 0.05`
- `action_noise_clip = 0.15`

设计原则：

- 噪声要足够小，不要显著破坏 teacher 质量；
- 但要足够大，能够真实扩展支持集。

如果收集后发现 noisy dataset 的离线成功率相对 deterministic 下降超过约 `5` 个百分点，则说明噪声过大，应适当收紧；  
如果下降几乎为零且后续训练也完全无差异，再考虑第二档更强噪声，而不是一开始就开多档。

### 5.4 实际执行的最小矩阵

为了省算力，实验包 B 没有重新扫完整 alpha 网格，而是只跑了最有信息量的配置：

| 数据集 | 配置 |
|---|---|
| `crosscomp-1000 deterministic` | `alpha = 0.0`, `0.25` |
| `crosscomp-1000 noisy` | `alpha = 0.0`, `0.25` |
| `crosscomp-2000 deterministic` | `alpha = 0.0`, `0.15`, `0.2` |
| `crosscomp-2000 noisy` | `alpha = 0.0`, `0.15`, `0.2` |

实际 screening 预算为：

- seeds：`42 / 43`
- `TRAIN_EPOCHS = 64`
- `VAL_MANIFEST_EPISODES = 40`
- `TEST_MANIFEST_EPISODES = 40`

仓库中已经补了该实验包的执行入口与对照分析脚本：

- [`scripts/run_offline_td3bc_phase0c_noisy_support.sh`](../scripts/run_offline_td3bc_phase0c_noisy_support.sh)
- [`scripts/analyze_offline_td3bc_noisy_support.py`](../scripts/analyze_offline_td3bc_noisy_support.py)

结果出来后，并没有继续把 noisy finalist 升级到 Stage C 级别预算确认，因为 screening 已经足够回答机制诊断问题：noisy support 主要改善了 `2000` 下的 BC，而没有让 TD3BC 的正 `alpha` 重新稳定受益。

### 5.5 已完成结果摘要

`noisy_support_screen` 已经完成，并给出如下关键结果：

| 数据集 | 版本 | 最优 alpha | TD3BC success | BC success | TD3BC - BC |
|---|---|---:|---:|---:|---:|
| 1000 | deterministic | 0.25 | 0.7625 | 0.6750 | +0.0875 |
| 1000 | noisy | 0.0 | 0.5125 | 0.5125 | 0.0000 |
| 2000 | deterministic | 0.15 | 0.7000 | 0.6000 | +0.1000 |
| 2000 | noisy | 0.0 | 0.7375 | 0.7375 | 0.0000 |

这组结果回答了两个关键问题：

1. `2000` 的 deterministic 数据确实存在支持集结构问题，因为 noisy variant 下 `BC` 明显从 `0.6000` 提升到了 `0.7375`。
2. broadened support 并没有让当前 TD3BC 在 noisy 数据上继续受益；相反，两组 noisy 数据的最优点都退回了 `alpha=0.0`。

因此，实验包 B 的最合理结论是：

> `1000 > 2000` 不是 TD3BC 特有现象；deterministic 大数据支持集不够鲁棒，而当前 TD3BC 也不能稳定利用 broadened support。

### 5.6 结果判读逻辑

这是本实验包最重要的部分：

#### 情况 1：`2000 noisy BC` 已显著改善

说明问题在行为支持集层面就已经存在。  
这意味着：

- `2000` 的退化并不主要是 TD3BC 的 critic 目标导致；
- 更大的 deterministic 数据并没有真正增加“可学的有效支持”；
- 后续应优先推进数据线，而不是急着上更复杂算法。

#### 情况 2：`2000 noisy BC` 变化不大，但 `2000 noisy TD3BC` 改善明显

说明问题更像：

- 数据支持集与 critic/Q 引导之间存在交互；
- 支持集稍微变宽后，Q-learning 才开始发挥更稳定的价值；
- 后续 ReBRAC / XQL 可能确实有机会继续改善。

#### 情况 3：noisy 与 deterministic 基本无差别

说明 `1000 > 2000` 的主因不太像支持集窄化本身；  
下一阶段应更偏向：

- critic regularization
- dual penalty
- in-sample value learning

也就是更支持尽快进入 ReBRAC / XQL。

### 5.7 完成判据

实验包 B 完成后，应至少能回答：

- `2000` 的异常是否对支持集扩展敏感；
- BC 与 TD3BC 对 noisy support 的敏感性是否一致；
- 下一阶段更应优先走“数据线”还是“算法线”。

上述三个问题现在都已经有了初步但清晰的答案：

- `2000` 对支持集扩展显著敏感；
- BC 与 TD3BC 的敏感性并不一致，noisy support 主要先改善 BC；
- 因此，下一阶段既不应把问题简单归咎于 TD3BC，也不应把它理解成“只要加数据就行”，而应转向 `worldcomp teacher-gap` 与后续更强的 in-sample 算法。

---

## 6. 实验包 C：`worldcomp` teacher-gap 正式实验

> 状态：已完成

### 6.1 目标

这个实验包要回答的是：

> 当 teacher 具有局部等效流信息优势时，TD3BC 在 deployable protocol 下到底能逼近到什么程度？  
> 如果再给 critic `privileged_obs`，这个 gap 能缩小多少？

它的意义不在于替换 `crosscomp` 主线，而在于补齐：

- deployable offline RL 的真实边界
- local teacher information gap

### 6.2 为什么当时不建议一上来做 `worldcomp` size sweep

在实验包 C 启动之前，`crosscomp` 主线已经说明：

- 最优规模出现在 `1000`
- `2000` 的解释还不清楚

因此，`worldcomp` 当时最重要的不是再做一轮 size study，而是先在**一个代表性规模**上完成 teacher-gap 诊断。  
从资源效率和可解释性看，最终选择优先用：

- `worldcomp-1000`

理由：

- `1000` 是当前 `crosscomp` 主线最稳定、最优的规模；
- 用同一规模更容易把“teacher 信息差”与“dataset size effect”分开。

### 6.3 实际采用的协议

`worldcomp` 最终被拆成两条协议：

#### Track C1：Deployable protocol

- actor：普通 `obs`
- critic：普通 `obs`

#### Track C2：Privileged-critic ablation

- actor：普通 `obs`
- critic：训练时读取 `privileged_obs`

注意：

- Track C2 是 ablation，不是主结果替代品。
- `alpha = 0` 时 critic 不参与 actor 更新，因此 `BC` 对照只需在 deployable track 下给出即可。

### 6.4 实际执行矩阵

`worldcomp-1000` 数据准备好后，训练部分按两段推进：

#### 第一步：Deployable screening

- 数据：`worldcomp-1000`
- alphas：`0.0`, `0.1`, `0.25`, `0.5`
- seeds：`42 / 43`
- `TRAIN_EPOCHS = 64`
- `VAL_MANIFEST_EPISODES = 40`
- `TEST_MANIFEST_EPISODES = 40`

#### 第二步：Privileged-critic screening

只对第一步中最有希望的 `1-2` 个 `alpha` 做 privileged-critic ablation：

- protocol：`privileged-critic`
- seeds：`42 / 43`
- 其余预算同上

#### 第三步：Formal confirmation

只推进两个 finalist：

1. best deployable TD3BC
2. best privileged-critic TD3BC

正式预算建议与 `phase0c stage_c` 对齐：

- seeds：`42 / 43 / 44 / 45 / 46`
- `TRAIN_EPOCHS = 96`
- `VAL_MANIFEST_EPISODES = 40`
- `TEST_MANIFEST_EPISODES = 100`

同时必须在相同 manifest 上跑：

- `worldcomp` baseline
- deployable BC (`alpha=0`)

仓库中已经补了该实验包的执行入口与对照分析脚本：

- [`scripts/run_offline_td3bc_phase0c_worldcomp_teacher_gap.sh`](../scripts/run_offline_td3bc_phase0c_worldcomp_teacher_gap.sh)
- [`scripts/analyze_offline_td3bc_worldcomp_teacher_gap.py`](../scripts/analyze_offline_td3bc_worldcomp_teacher_gap.py)

### 6.5 关键判读逻辑

#### 情况 1：deployable TD3BC 已经很接近 `worldcomp`

说明 teacher 信息差并不是当前主瓶颈。  
后续更应关注算法与数据支持集。

#### 情况 2：privileged-critic 明显优于 deployable TD3BC

说明局部等效流信息差是重要因素。  
这时论文中应明确区分：

- deployable upper envelope
- privileged-critic upper envelope

#### 情况 3：deployable 与 privileged-critic 都明显低于 `worldcomp`

说明问题不只在当前 `privileged_obs` 的局部信息差，可能还包括：

- 更广的 teacher inductive bias
- 数据分布
- 算法本身

这会提高后续 ReBRAC / XQL 的研究价值。

### 6.6 已完成结果摘要

`worldcomp teacher-gap` 实验已经完成，对应专项报告见：

- [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)

正式 5-seed 结果如下：

| 协议 | best alpha | success | std | baseline gap |
|---|---:|---:|---:|---:|
| deployable | 0.0 | 0.858 | 0.080 | 0.132 |
| privileged-critic | 0.1 | 0.922 | 0.086 | 0.068 |
| `worldcomp` baseline | - | 0.990 | - | - |

与 deployable 相比，privileged-critic：

- success 提升 `+0.064`
- 将 baseline gap 从 `0.132` 缩小到 `0.068`
- 关闭约 `48.5%` 的 deployable teacher gap

终止类型聚合结果也支持这一点：

- deployable：`goal=429`, `out_of_bounds=60`, `timeout=9`, `depth_hold_failure=2`
- privileged-critic：`goal=461`, `out_of_bounds=37`, `timeout=1`, `depth_hold_failure=1`

因此，实验包 C 给出了一个非常清晰的结论：

> `worldcomp` 主线上的瓶颈并不只是数据支持集或训练预算问题；  
> 局部 teacher information gap 是一个真实且重要的性能约束。

### 6.7 当前结论

实验包 C 完成后，本轮收口计划里预设的三个核心问题都已经有了实验回答：

- `worldcomp` gap 有多大：
  - deployable 与 baseline 的正式 gap 为 `0.132`
- 这个 gap 里有多少是 `privileged_obs` 可解释的：
  - privileged-critic 关闭了约 `48.5%`
- deployable 主结果与 privileged-critic ablation 是否需要分开呈现：
  - 需要，因为二者对应不同信息约束，不能混作同一 deployable 主结果

---

## 7. 什么不要做

为了避免收口阶段失控，这里明确列出不建议立即做的事情：

1. **不要再开新的 `crosscomp size × alpha` 大扫网格。**
2. **不要优先做 holdout/index-aware sampler。**
3. **不要直接跳到 FQL。**
4. **不要把 `phase0c` 的 `500` 边界问题继续放大成主任务。**
5. **不要把“收口 TD3BC”演变成无限补实验。**

本轮收口只服务于一个目的：

> 让当前 TD3BC 主线的结论足够干净，以便后续 ReBRAC 的改进可以被正确解释。

---

## 8. 推荐的最终交付物

本轮收口完成后，建议至少形成以下输出：

1. 一份 `TD3BC closure report`
   - 正式 BC 对照
   - noisy-support 结果
   - `worldcomp` teacher-gap 结果
   - 与现有 `phase0c` 主结果的合并解释
   - 当前对应文档：
     - [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)
     - [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)

2. 一张最终主结论图或表，至少包含：
   - `crosscomp-1000`: TD3BC vs BC
   - `crosscomp-2000`: TD3BC vs BC
   - `worldcomp-1000`: deployable TD3BC vs privileged-critic vs baseline

3. 一句可供后续论文承接的结论：

> 当前 TD3BC 主线已经证明 deployable offline RL 的价值；`crosscomp` 暴露了大 deterministic 数据 regime 下的支持集/利用瓶颈，`worldcomp` 则暴露了真实的 teacher information gap，这正是后续 ReBRAC 与更强 in-sample 基线需要回答的问题。

---

## 9. 一句话执行建议

> TD3BC 主线收口已完成，下一步进入 ReBRAC；这样后续任何改进都可以更干净地解释为“算法进步”，而不是“补上了原本缺失的对照”。
