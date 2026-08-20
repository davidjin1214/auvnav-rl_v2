# TD3BC `worldcomp teacher-gap` 实验报告

> 文档定位：这是 `phase0c` 补充实验包 C 的专项报告；截至当前，其核心结论已经并入 [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md) 的统一主线解释中。
> 当前阅读建议：若需要 `phase0c` 全部实验的一体化叙事，请优先阅读 `phase0c` 总报告；若需要单独引用 `worldcomp` 下的 teacher-gap 机制与 formal 数字，则以本文为准。

## 1. 报告概述

本文档整理并分析 `results/offline/td3bc/phase0c/worldcomp_teacher_gap/` 下已经完成的 `worldcomp` teacher-gap 实验结果，主要包含：

- `deployable_screen`
- `privileged_screen`
- `deployable_final`
- `privileged_final`
- `analysis_compare`

本实验是 [td3bc_mainline_closure_plan.md](./td3bc_mainline_closure_plan.md) 中实验包 C 的正式结果，用于回答以下问题：

1. 当离线数据来自 `worldcomp` teacher 时，deployable protocol 下的 TD3BC 最多能学到什么程度？
2. 如果在训练时允许 critic 读取 `privileged_obs`，`teacher gap` 能缩小多少？
3. `worldcomp` 下的主要瓶颈更像是数据支持集问题，还是观测信息问题？
4. 这些结果对 TD3BC 主线收口，以及后续 ReBRAC 的优先级意味着什么？

本文的核心结论可以先概括为一句话：

> 在 `worldcomp-1000` 数据上，deployable TD3BC 的正式最优点退回到纯 `BC(alpha=0)`，而 privileged-critic 能将 success 从 `0.858` 提升到 `0.922`，关闭约一半 deployable-to-baseline gap。这说明 `worldcomp` 主线上的瓶颈有很大一部分来自局部 teacher information gap，而不是单纯的训练预算不足或 TD3BC 完全失效。

---

## 2. 背景与实验动机

### 2.1 为什么要做 `worldcomp teacher-gap`

在 `crosscomp` 主线上，`phase0c` 已经得到两个关键结论：

- `1000 > 2000` 不是协议伪象，而是在正式预算下稳定存在的现象；
- 这个现象在同预算 `BC` 下同样成立，因此它并不是 TD3BC 特有失败模式；
- `noisy-support` 诊断进一步表明，`2000` 的一部分问题来自 deterministic 数据支持集结构不够鲁棒。

但这些结论主要回答的是 **support structure** 问题，还没有回答另一个同样重要的问题：

> 当 teacher 本身具有 deployable policy 看不到的局部等效流信息时，当前 TD3BC 的上限究竟受什么约束？

`worldcomp` 正是为这个问题准备的：它使用更强的 teacher 行为策略，因此更适合用来测量 **teacher information gap**。

### 2.2 与 `crosscomp` 主线的关系

`worldcomp teacher-gap` 不是要替换 `crosscomp` 主线，而是要补齐其解释链：

- `crosscomp` 更适合研究 deployable offline RL 的主线行为，以及数据规模/支持集问题；
- `worldcomp` 更适合研究 teacher 比 deployable 策略多出来的局部信息，到底能带来多大增益。

因此，这组实验的意义是将两个问题拆开：

1. 数据支持集是否足够鲁棒；
2. 观测信息是否足够支撑 critic-guided improvement。

### 2.3 为什么不先做 `worldcomp` size sweep

本轮并没有对 `worldcomp` 做 `500 / 1000 / 2000` 的完整 size sweep，而是固定在 `1000` episodes。原因有两个：

- `1000` 是当前 `crosscomp` 主线中最稳定、表现最好的规模；
- 用同一规模做 teacher-gap 诊断，更容易把 dataset-size effect 与 observation-gap effect 分开。

因此，这轮 `worldcomp` 实验优先回答的是：

> 在代表性规模 `1000` 上，teacher 信息差究竟有多大？

---

## 3. 实验设计

### 3.1 任务与环境设定

本轮实验采用与 `phase0c` 主线一致的环境设定：

- benchmark：`single_u10_cross_tgt15`
- task geometry：`cross_stream`
- target speed：`1.5`
- objective：`efficiency_v2`
- probe layout：`s0`
- history length：`4`
- flow：`wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

### 3.2 离线数据集

离线数据来自 `worldcomp` teacher 行为策略，对应数据集为：

- `worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone`

数据规模为：

- `1000` episodes

baseline 对照为：

- `worldcomp` baseline

### 3.3 两条协议

本实验将 `worldcomp` 拆成两条协议。

#### Track 1：Deployable

- actor 输入：普通 deployable `obs`
- critic 输入：普通 deployable `obs`

这条协议对应真实部署约束，因此是主结果。

#### Track 2：Privileged-critic

- actor 输入：普通 deployable `obs`
- critic 在训练时额外读取 `privileged_obs`
- actor update mode：`zeros`

这条协议不是部署替代品，而是一个信息上界型 ablation，用来测量：

> 如果 critic 拥有 teacher 级别的局部信息，当前 TD3BC 能追回多少 gap？

### 3.4 Screening 协议

Screening 分为两段：

#### Deployable screening

- alphas：`0.0 / 0.1 / 0.25 / 0.5`
- seeds：`42 / 43`
- train epochs：`64`
- validation episodes：`40`
- test episodes：`40`

#### Privileged-critic screening

Privileged screening 不重新扫整张大网格，而是自动从 deployable screening 中选取前 `2` 个正 `alpha`，实际进入 privileged screening 的为：

- `0.1`
- `0.25`

其余预算与 deployable screening 对齐：

- seeds：`42 / 43`
- train epochs：`64`
- validation episodes：`40`
- test episodes：`40`

### 3.5 Formal 协议

Formal confirmation 与 `phase0c stage_c` 对齐：

- seeds：`42 / 43 / 44 / 45 / 46`
- train epochs：`96`
- checkpoint every：`4` epochs
- validation episodes：`40`
- test episodes：`100`

进入 formal 的 finalist 为：

- deployable：`alpha=0.0` 与 screening best deployable alpha
- privileged-critic：screening best privileged alpha

实际 formal 结果中：

- deployable 最优为 `alpha=0.0`
- privileged-critic 最优为 `alpha=0.1`

### 3.6 模型选择规则

各阶段统一采用如下选择规则：

- 单个 run 内 checkpoint 选择：
  - `eval_success_rate`
  - `eval_return`
  - `-eval_safety_cost`
  - `-eval_time_s`
- 跨 seed 的 `alpha` 选择：
  - `mean_val_success_rate`
  - `mean_val_return`
  - `-mean_val_safety_cost`
  - `-mean_val_time_s`

---

## 4. Screening 结果

### 4.1 Deployable screening

Deployable screening 的 validation 结果为：

| alpha | mean val success | std | mean val return |
| --- | ---: | ---: | ---: |
| 0.0 | 0.7875 | 0.0625 | -12.77 |
| 0.1 | 0.7250 | 0.0750 | -30.03 |
| 0.25 | 0.7250 | 0.1000 | -22.09 |
| 0.5 | 0.5125 | 0.0125 | -33.82 |

Deployable screening 的结论非常清楚：

- 最优点是 `alpha=0.0`
- 一旦引入正 `alpha`，validation success 就下降
- `alpha=0.5` 甚至出现明显退化

因此，在 `worldcomp` 数据上，deployable setting 下最优解退回到了纯 `BC`。

### 4.2 Privileged-critic screening

Privileged-critic screening 的 validation 结果为：

| alpha | mean val success | std | mean val return |
| --- | ---: | ---: | ---: |
| 0.1 | 0.8125 | 0.0875 | -29.83 |
| 0.25 | 0.6500 | 0.1250 | -18.40 |

与 deployable screening 相比：

- privileged-critic 在 `alpha=0.1` 上取得了最优结果；
- screening 阶段的 success 从 deployable 的 `0.7875` 提升到 privileged 的 `0.8125`；
- screening teacher-gap 改善为 `+0.025`。

这说明：

> 一旦 critic 拥有 privileged 信息，Q 引导开始重新变得有价值。

### 4.3 Screening 阶段的初步判断

只看 screening，就已经能看到一个非常有信息量的分叉：

- deployable：最优点退回 `alpha=0.0`
- privileged-critic：最优点转向 `alpha=0.1`

这意味着 deployable setting 下的问题并不是“数据太差，什么都学不出来”，而更像是：

> 在只看 deployable 观测时，critic 无法稳定学出足够好的动作排序，因此正 `alpha` 的 Q 引导帮不上忙。

---

## 5. Formal 结果

### 5.1 正式主结果

Formal 5-seed 结果如下：

| 协议 | best alpha | success | std | return | safety cost | time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deployable | 0.0 | 0.858 | 0.080 | -14.29 | 7.91 | 63.19 |
| privileged-critic | 0.1 | 0.922 | 0.086 | 16.49 | 5.73 | 54.33 |
| worldcomp baseline | - | 0.990 | - | 32.19 | - | - |

对应的 success gap 为：

- baseline - deployable = `0.132`
- baseline - privileged = `0.068`
- privileged - deployable = `+0.064`

也就是说，privileged critic 关闭了：

- `0.064 / 0.132 ≈ 48.5%`

的 deployable teacher gap。

### 5.2 相对 BC 的意义

在 formal deployable 结果里，最优点是 `alpha=0.0`。  
因此，deployable formal 下的：

- `deployable TD3BC success`
- `deployable BC success`

是相同的，均为 `0.858`。

这并不意味着 TD3BC 在 `worldcomp` 上“完全没价值”，而是更具体地说明：

> 在只给 deployable 观测时，当前 TD3BC 的最优策略退化为了纯 BC；一旦 critic 拥有 privileged 信息，正 `alpha` 才开始重新有用。

### 5.3 终止类型统计

将 5 个 seed、共 500 个 formal test episodes 聚合后：

#### Deployable

- `goal = 429`
- `out_of_bounds = 60`
- `timeout = 9`
- `depth_hold_failure = 2`

#### Privileged-critic

- `goal = 461`
- `out_of_bounds = 37`
- `timeout = 1`
- `depth_hold_failure = 1`

privileged-critic 相比 deployable：

- 多了 `32` 个成功到达
- 少了 `23` 个越界
- 少了 `8` 个超时

因此，`+0.064` 的 success 增益并不是偶然波动，而是来自显著更少的失败终止。

### 5.4 轨迹质量指标

除了 success，本轮实验在路径与效率指标上也给出了一致信号：

| 指标 | deployable | privileged-critic |
| --- | ---: | ---: |
| progress ratio | 0.735 | 0.848 |
| path efficiency | 0.701 | 0.775 |
| mean path length (m) | 56.83 | 51.23 |

这说明 privileged-critic 的收益不是“更激进地赌成功率”，而是轨迹整体更短、更快、更接近 baseline 行为。

---

## 6. 结果分析

### 6.1 `teacher gap` 是真实且重要的

Formal 结果表明：

- deployable：`0.858`
- privileged-critic：`0.922`
- baseline：`0.990`

`+0.064` 的 privileged improvement 已经足够大，不能视作边缘效应。  
这意味着：

> `worldcomp` 主线上的一部分性能差距，确实来自 deployable policy 缺少 teacher 拥有的局部信息。

### 6.2 但 `teacher gap` 不是全部解释

即使给 critic privileged 信息，privileged-critic 仍然落后 baseline：

- baseline - privileged = `0.068`

因此，当前 gap 至少由两部分组成：

1. **观测信息差**
2. **算法与策略表达差**

换句话说，privileged critic 解释了大约一半 gap，但没有完全解释全部差距。

### 6.3 为什么 deployable 下最优点会退回 `alpha=0`

这是本实验最关键的机制信号之一。

在 `crosscomp` 主线中，正 `alpha` 在部分 setting 下仍能带来收益；  
但在 `worldcomp` deployable 轨道中，最优点退回 `alpha=0.0`，说明：

> 只给 deployable 观测时，critic 很难稳定提供比行为克隆更好的动作排序，导致 Q 引导对 actor 更新没有帮助。

这与 privileged-critic 的结果形成了清晰对照：

- critic 看不到额外信息时，正 `alpha` 无益；
- critic 拥有 privileged 信息时，`alpha=0.1` 重新变成最优点。

因此，可以更明确地说：

> `worldcomp` 下 deployable TD3BC 的主要瓶颈之一，不是“TD3BC 完全失效”，而是 critic 缺少足够的信息来做可靠的 improvement step。

### 6.4 与 `crosscomp` 主线的互补关系

截至当前 TD3BC 主线收口，可以把两条数据线的结论合并为：

- `crosscomp`：主要暴露数据支持集结构与大数据 regime 的利用问题
- `worldcomp`：主要暴露局部 teacher information gap

这两个结果并不冲突，反而互补：

- `crosscomp` 告诉我们“数据怎么给”很重要；
- `worldcomp` 告诉我们“critic 能看到什么”同样重要。

因此，当前 TD3BC 的剩余差距不能再简单归结为单一原因。

---

## 7. 对主线收口与下一阶段的意义

### 7.1 对 TD3BC 主线收口的意义

这组 `worldcomp teacher-gap` 结果完成后，TD3BC 主线已经具备一条相对完整的证据链：

1. `phase0b_v2` 证明旧 `phase0` 的负 size trend 主要是协议伪象；
2. `phase0c` 证明 `1000 > 2000` 在正式预算下稳定存在；
3. Stage C 同预算 BC 对照证明 `1000 > 2000` 不是 TD3BC 特有问题；
4. noisy-support 诊断证明 deterministic 大数据支持集结构确实是部分原因；
5. `worldcomp teacher-gap` 则证明：在更强 teacher 数据上，观测信息差同样是重要瓶颈。

因此，现在已经可以较为有把握地说：

> 当前 TD3BC 主线的剩余问题，主要来自两类因素：  
> 一类是大 deterministic 数据的支持集结构问题；另一类是 deployable 观测下 critic 无法稳定利用 teacher 级局部信息的问题。

### 7.2 对 ReBRAC 的意义

这组结果直接提高了后续 ReBRAC 的研究价值。  
因为现在进入下一算法线时，问题已经被拆得更具体：

- 不是泛泛地问“ReBRAC 会不会更强”；
- 而是在问“更强的 in-sample regularization / critic coupling 能否在 deployable 信息受限时做得更稳”。

如果后续 ReBRAC 能在：

- `crosscomp-2000`
- `worldcomp deployable`

这两个 regime 中同时改善结果，那么它的贡献就会非常明确。

---

## 8. 局限性

尽管这轮实验已经足以支撑 teacher-gap 结论，但仍有几个边界条件需要说明：

1. 当前只在 `worldcomp-1000` 上做了 teacher-gap，没有做 size sweep。
2. privileged-critic 是训练期 critic 信息增强，不代表可部署策略本身可以在推理时访问 privileged 信息。
3. 本轮 formal deployable 只推进了 `alpha=0.0` 与 screening best alpha 的 finalist 逻辑，没有对更大正 `alpha` 做 5-seed 全网格确认。

这些局限并不削弱当前主结论，但意味着：

- 这组实验更适合回答“teacher gap 是否重要”
- 而不是回答“`worldcomp` 下的所有超参数最优结构”

---

## 9. 最终结论

本轮 `worldcomp teacher-gap` 实验给出如下最终结论：

1. `worldcomp` 数据上的 deployable gap 是真实存在的。  
2. 在 deployable setting 下，当前 TD3BC 的正式最优点退回到 `alpha=0.0`，说明只用 deployable 观测时，critic-guided improvement 不稳定。  
3. 一旦 critic 在训练时能够访问 `privileged_obs`，最佳 `alpha` 变为 `0.1`，success 从 `0.858` 提升到 `0.922`。  
4. privileged critic 关闭了约 `48.5%` 的 deployable teacher gap，说明局部 teacher information gap 是重要瓶颈。
   ⚠ **追注（2026-08-20）**：该 `48.5%` 算在一个与选点集嵌套的 $100$ 回合评估集上（前 $40$ 条即选 checkpoint 用的验证集，[`data_integrity_open_items.md`](./data_integrity_open_items.md) 第 ③ 条）。仅计入未参与选点的 $60$ 条时降为 **35.6%**（位移 $-12.9$ pp，同次复核中位移最大的读数之一）。**点估计不改**；引用时须连同该限定，参见 `rebrac.tex` §5.7.2/§5.7.m。  
5. 但 privileged-critic 仍未达到 baseline，因此剩余差距不只来自观测信息，也来自算法与策略层面的限制。  

一句话总结：

> `crosscomp` 主线告诉我们“数据支持集结构”是关键问题，`worldcomp` 主线则告诉我们“critic 的可见信息”同样是关键问题。TD3BC 主线收口后，下一阶段进入 ReBRAC 已具备清晰、具体且可解释的问题设定。

## 10. 这份报告在当前项目中的位置

从当前整体离线 RL 进展回看，这份专项报告的作用已经比较明确：

1. 它不是独立于 `phase0c` 之外的一条新主线，而是 `phase0c` 收口链中的 teacher-gap 证据包。
2. 它回答的是“为什么 deployable `worldcomp` 轨道会退回 pure BC”以及“privileged critic 能追回多少 gap”，而不是重新定义 deployable 主结果。
3. 它与 `crosscomp` 主线形成互补分工：
   - `crosscomp` 主要暴露 deterministic 大数据支持集结构问题；
   - `worldcomp` 主要暴露 deployable critic 的信息瓶颈问题。

因此，当前最合理的用法是：

- 在论文主结果中，把本文作为 `phase0c` 总报告的 teacher-gap 专项支撑；
- 在方法设计上，把这里得到的 “critic 可见信息不足” 结论作为后续 ReBRAC / XQL 的约束；
- 而不是继续把 `worldcomp` 扩展成新的大规模 size sweep 主线。
