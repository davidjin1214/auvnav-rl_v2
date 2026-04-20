# TD3BC `phase0c` 实验报告

## 1. 报告概述

本文档整理并分析 `results/offline/td3bc/phase0c/` 下已经完成的三阶段实验结果，包括：

- `stage_a_reval`
- `stage_b_screen`
- `stage_c_final`
- `stage_c_shadow_ep2000_a0p2`

与 [td3bc_phase0c_experiment_design.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_phase0c_experiment_design.md:1) 不同，本文档聚焦**已经实际跑出的结果**，回答以下核心问题：

1. `phase0b_v2` 之后，`phase0c` 是否进一步稳定了关于数据集规模效应的判断？
2. `TD3BC` 在不同数据集规模下的最优 `alpha` 如何变化？
3. 在更正式的 5-seed、100-episode 协议下，最佳数据集规模究竟是 `500`、`1000` 还是 `2000`？
4. 额外补跑的 `2000, alpha=0.2` shadow run 是否推翻了正式 finalist `alpha=0.15`？

本文的最终结论可以先概括为一句话：

> 在 `phase0c` 的正式协议下，TD3BC 的性能并不是随离线数据规模单调上升，而是在 **`1000` episodes 左右达到最佳**；`500` 明显偏小，`2000` 并未进一步带来收益。

---

## 2. 背景与实验动机

### 2.1 `phase0b_v2` 已经修正了什么

在旧版 `phase0` 中，曾观察到一个明显反常现象：离线数据越大，TD3BC 最优表现反而越差。后续 `phase0b_v2` 通过以下改动修正了该问题：

- 将训练预算从固定 `total_steps` 改为按 epoch 对齐
- 将 replay-style 有放回采样改为 `shuffle_no_replacement`
- 将 validation 与 test 显式分离
- 采用 checkpoint 后验选择与 `alpha` 后验选择

`phase0b_v2` 已经证明：旧版“数据越大越差”的现象主要是**实验协议伪象**。

### 2.2 为什么还需要 `phase0c`

尽管 `phase0b_v2` 修正了方向，但它仍然更像一个低成本 pilot，而不是论文正式实验：

- seed 数较少
- validation/test 规模有限
- `alpha` 搜索不够系统
- stage 之间的资源分配仍然可以更优化

因此，`phase0c` 的设计目标不是简单重复 `phase0b_v2`，而是：

- 保留论文式 ablation 所需的结构化对照
- 限制总计算开销
- 让正式结果尽可能稳定、可解释、可复现

### 2.3 `phase0c` 的方法学意义

`phase0c` 的核心思想是三阶段漏斗式协议：

- Stage A：复用已有 checkpoint，用更大 validation 降低 pilot winner 噪声
- Stage B：低成本 screening，完成 `size × alpha` 的主体 ablation
- Stage C：只推进 finalist，做正式 5-seed 确认

从实验设计角度看，`phase0c` 解决的不是“再跑更多实验”，而是“如何把算力花在最有信息量的地方”。

---

## 3. 实验设置

### 3.1 任务与环境设定

本轮实验沿用 `phase0b_v2` 的固定任务设定：

- benchmark：`single_u10_cross_tgt15`
- task geometry：`cross_stream`
- target speed：`1.5`
- objective：`efficiency_v2`
- probe layout：`s0`
- history length：`4`
- flow：`wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

### 3.2 离线数据集

离线数据来自确定性 `crosscomp` 行为策略，对应数据集为：

- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone` (`500`)
- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000` (`1000`)
- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000` (`2000`)

与 `phase0b_v2` 相同，这三组数据集的统计量约为：

| 数据集规模 | transitions | 数据集成功率 |
| --- | ---: | ---: |
| 500 episodes | 75,975 | 0.8640 |
| 1000 episodes | 152,683 | 0.8700 |
| 2000 episodes | 304,967 | 0.8855 |

因此，`phase0c` 的比较前提仍然成立：

> 更大的离线数据集在数据质量层面并没有变差，反而略有提升。

### 3.3 算法与对照

实验主体算法为 `TD3BC`，其中：

- `alpha=0` 可视为纯行为克隆（BC）
- `alpha>0` 表示引入不同强度的 Q 引导修正

本报告中使用了三类比较对象：

- `TD3BC`
- `BC (alpha=0)`
- `crosscomp baseline`

### 3.4 统一选择规则

各阶段统一采用以下规则：

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

该规则避免了人工挑选 checkpoint 或事后口径漂移。

---

## 4. `phase0c` 协议回顾

### 4.1 Stage A：大 validation 再评估

Stage A 不重新训练，只复用 `phase0b_v2` checkpoint，在更大的 validation 上重新排序：

- episodes：`500 / 1000 / 2000`
- alphas：`0.0 / 0.1 / 0.25 / 0.5`
- seeds：`42 / 43`
- validation episodes：`80`

Stage A 的定位是：**去噪与复核**，而非正式主结果。

### 4.2 Stage B：低成本 screening

Stage B 是 `phase0c` 的主体 ablation 阶段。默认配置为：

- seeds：`42 / 43`
- train epochs：`64`
- validation episodes：`40`
- test episodes：`40`

初始 alpha 网格按数据集规模分组设计：

| 数据集规模 | Stage B 初始 alpha 网格 |
| --- | --- |
| 500 | `0.1 / 0.5` |
| 1000 | `0.1 / 0.25 / 0.5 / 0.75` |
| 2000 | `0.0 / 0.05 / 0.1 / 0.15 / 0.2 / 0.25` |

后续又补充了：

- `500` 的 `alpha=0`
- `1000` 的 `alpha=0`

这样 Stage B 才具备完整的 `TD3BC vs BC` 参考。

### 4.3 Stage C：正式确认

Stage C 的作用是：只推进每个数据集规模一个 finalist，做正式确认。配置为：

- seeds：`42 / 43 / 44 / 45 / 46`
- train epochs：`96`
- validation episodes：`40`
- test episodes：`100`

finalist 来自 Stage B：

- `500 -> alpha=0.5`
- `1000 -> alpha=0.25`
- `2000 -> alpha=0.15`

### 4.4 Shadow run：`2000, alpha=0.2`

由于 Stage B 中 `2000` 的 `alpha=0.15` 与 `0.2` 非常接近，因此额外补跑了一个与 Stage C 同预算的 shadow run：

- `stage_c_shadow_ep2000_a0p2`

其目的不是重开搜索，而是验证：

> 正式 finalist `alpha=0.15` 是否被邻近点 `alpha=0.2` 明显击败？

---

## 5. Stage A 结果：pilot winner 的大 validation 复核

Stage A 只看 validation，不给正式 test 结论。

### 5.1 Stage A 验证结果

| 数据集规模 | `alpha=0` | `alpha=0.1` | `alpha=0.25` | `alpha=0.5` | Stage A winner |
| --- | ---: | ---: | ---: | ---: | --- |
| 500 | 0.2313 | 0.2500 | 0.2688 | 0.3188 | `0.5` |
| 1000 | 0.3688 | 0.5000 | 0.5063 | 0.4563 | `0.25` |
| 2000 | 0.6063 | 0.6375 | 0.5750 | 0.4750 | `0.1` |

数据来源：`stage_a_reval/analysis/alpha_validation.csv`。

### 5.2 Stage A 的关键信息

Stage A 给出了两个重要信号：

1. `1000` 的 winner 从 `phase0b_v2` pilot 中偏大的 `0.5`，被重新排序为 `0.25`。
2. `2000` 继续偏好较小 `alpha`，而不是大 `alpha`。

这说明 Stage A 的复核是有效的：

- 它没有机械重复 pilot 结果
- 而是提前识别出了 `1000 -> 0.25` 这个后续被正式结果确认的 winner

### 5.3 Stage A 的局限性

Stage A 使用的是旧 checkpoint，因此它只能回答：

- “现有 winner 稳不稳定”

而不能回答：

- “在更长训练下最终谁最好”

因此 Stage A 只是正确地扮演了**去噪层**，而不是终局层。

---

## 6. Stage B 结果：完整 screening 与 `TD3BC vs BC` 对照

### 6.1 Stage B 验证结果

补全后的 Stage B validation 结果如下：

| 数据集规模 | alpha 网格与 mean val success |
| --- | --- |
| 500 | `0.0: 0.55`, `0.1: 0.4875`, `0.5: 0.5625` |
| 1000 | `0.0: 0.675`, `0.1: 0.725`, `0.25: 0.7625`, `0.5: 0.55`, `0.75: 0.5125` |
| 2000 | `0.0: 0.60`, `0.05: 0.60`, `0.1: 0.60`, `0.15: 0.70`, `0.2: 0.6875`, `0.25: 0.6125` |

对应 winner 为：

- `500 -> alpha=0.5`
- `1000 -> alpha=0.25`
- `2000 -> alpha=0.15`

### 6.2 Stage B test 与 BC 对照

补全后的 Stage B held-out test 结果如下：

| 数据集规模 | 最优 alpha | TD3BC success | BC success | TD3BC - BC | baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.5 | 0.5625 ± 0.0625 | 0.55 ± 0.0250 | +0.0125 | 0.90 |
| 1000 | 0.25 | 0.7625 ± 0.0125 | 0.675 ± 0.0000 | +0.0875 | 0.90 |
| 2000 | 0.15 | 0.7000 ± 0.0500 | 0.60 ± 0.1500 | +0.1000 | 0.90 |

### 6.3 Stage B 的直接结论

Stage B 已经给出三个清晰结论：

1. `TD3BC` 在三个规模上都优于 `BC`，但 `500` 上优势极小。
2. `1000` 的最优点最干净、方差最小，是 screening 阶段最强的 candidate。
3. `2000` 的最优区间不是大 `alpha`，而是 `0.15 ~ 0.2` 这样的中小 `alpha`。

其中最值得强调的是：

- `500` 上，`TD3BC` 与 `BC` 几乎打平，说明在小数据 regime 下，Q 项边际收益很弱。
- `1000` 上，`TD3BC` 相对 `BC` 的提升最稳定、最可信。
- `2000` 上，Q 项仍有收益，但 `BC` 本身波动也很大，说明这一规模的学习更敏感。

### 6.4 Stage B 对 Stage C 的预测是否正确

事实证明，Stage B 的 finalist 选择是成功的：

- `500 -> 0.5`
- `1000 -> 0.25`
- `2000 -> 0.15`

这些选择全部被推进到 Stage C，并形成了正式主结果。

---

## 7. Stage C 正式结果：5-seed 主表

### 7.1 正式主结果

`stage_c_final` 使用 5 个 seeds、100 个测试 episode，得到如下正式结果：

| 数据集规模 | finalist alpha | success rate | return | safety cost | time | path efficiency | baseline success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.5 | 0.592 ± 0.119 | -155.92 ± 38.97 | 23.43 ± 4.19 | 111.53 ± 10.90 | 0.428 ± 0.074 | 0.95 |
| 1000 | 0.25 | 0.672 ± 0.045 | -126.24 ± 19.40 | 22.13 ± 4.50 | 103.99 ± 9.90 | 0.491 ± 0.025 | 0.95 |
| 2000 | 0.15 | 0.596 ± 0.036 | -162.53 ± 15.26 | 24.01 ± 4.60 | 115.33 ± 4.88 | 0.433 ± 0.031 | 0.95 |

对应相对 baseline 的 success gap 为：

- `500`: `-0.358`
- `1000`: `-0.278`
- `2000`: `-0.354`

### 7.2 Stage C 的正式排序

按正式 success rate 排序：

1. `1000 -> 0.25`: `0.672`
2. `2000 -> 0.15`: `0.596`
3. `500 -> 0.5`: `0.592`

也就是说：

> 在正式 5-seed 结果下，最佳数据规模是 `1000`，而不是 `2000`。

这里需要额外说明一点：`stage_c_final/analysis/dataset_diagnostics.csv` 中的
`best_alpha_hits_boundary=True` 不应被字面理解为“Stage C 证明最优 `alpha` 仍然撞边界”。
这是因为 Stage C 本来就只运行了每个数据集规模的单个 finalist `alpha`，因此分析脚本中的
“最大 alpha” 与 “最佳 alpha” 天然相同。真正有信息量的 `alpha` 边界判断，应以
Stage B 和 shadow 实验为准，而不是直接读 Stage C 的该字段。

### 7.3 这意味着什么

这一定义了 `phase0c` 最重要的研究结论：

- `phase0b_v2` 已经证明“更大数据不一定更差”
- `phase0c` 则进一步证明“更大数据也不一定持续更好”

当前任务、当前模型和当前 TD3BC 协议下，数据规模效应呈现出更像**倒 U 型**的关系：

- `500`：数据不足
- `1000`：最优平衡点
- `2000`：继续增大数据并未转化为更好的闭环控制

---

## 8. `2000, alpha=0.15` 与 `0.2` 的 shadow 对照

### 8.1 为什么要补 shadow run

Stage B 中，`2000` 的 `alpha=0.15` 与 `0.2` 很接近：

- `0.15`: val success `0.70`
- `0.2`: val success `0.6875`

同时 `0.2` 的 return 更好，因此需要单独验证：

> Stage C 的 `2000 -> 0.15` 是否会被 `0.2` 推翻？

### 8.2 Shadow 结果

`stage_c_shadow_ep2000_a0p2` 的正式结果为：

| 配置 | success | std | return | safety cost | time | path efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2000, alpha=0.15` | 0.596 | 0.036 | -162.53 | 24.01 | 115.33 | 0.433 |
| `2000, alpha=0.2` | 0.602 | 0.073 | -157.24 | 21.72 | 113.63 | 0.440 |

### 8.3 如何解释这组对照

`alpha=0.2` 相比 `0.15` 的结果是：

- success 均值略高：`+0.006`
- return 略好
- safety cost 略低
- 但标准差明显更大：`0.073` vs `0.036`

因此这组 shadow run 的正确解读应当是：

- `0.2` 没有明显击败 `0.15`
- `0.15` 作为 Stage C 正式 finalist 是合理的
- `2000` 附近的最优区间并不是一个尖锐点，而更像 `0.15 ~ 0.2` 的窄平坦区间

这对论文写作很有价值，因为它说明：

- `2000` 的最优 `alpha` 是局部稳定区间，而不是单点偶然噪声

---

## 9. 失败模式分析

### 9.1 Stage C 正式结果的终止类型聚合

将 Stage C 五个 seed、每个 seed 100 个 test episode 聚合后，终止统计如下：

| 数据集规模 | goal | out_of_bounds | timeout | depth_hold_failure |
| --- | ---: | ---: | ---: | ---: |
| 500 | 296 | 122 | 81 | 1 |
| 1000 | 336 | 105 | 59 | 0 |
| 2000 (`0.15`) | 298 | 116 | 81 | 5 |
| 2000 shadow (`0.2`) | 301 | 120 | 75 | 4 |

### 9.2 失败模式告诉了什么

这里最重要的现象不是 `2000` 比 `500` 更糟，而是：

- `1000` 明显比 `500` 与 `2000` 都更少 `timeout`
- `1000` 也更少 `out_of_bounds`
- `2000` 的问题并不主要体现为“严重失控”，而是“没有更高效地完成任务”

换句话说：

- `2000` 并没有带来“更稳更强”的闭环策略
- 它更像是引入了更广的状态覆盖，但当前模型并没有把这些额外数据有效转化为决策收益

Shadow `0.2` 的结果进一步说明：

- 更激进一点的 Q 修正可以略微减少 `timeout`
- 但会伴随更多 `out_of_bounds`

因此 `0.15` 与 `0.2` 的差别，更像是**保守性与激进性之间的 trade-off**，而不是谁彻底压倒谁。

---

## 10. 结果分析与解释

### 10.1 `phase0c` 的主结论不是“更大更好”，而是“存在最佳中间规模”

这是 `phase0c` 相比 `phase0b_v2` 最重要的新增发现。

`phase0b_v2` 的意义在于纠正旧版负趋势；但 `phase0c` 更进一步指出：

- 修正协议之后，数据规模效应不是简单单调上升
- 在当前任务与模型下，性能在 `1000` 附近达到最佳

这比“越大越好”更有研究价值，也更符合很多离线 RL 实际问题的经验规律。

### 10.2 最优 `alpha` 随数据规模下降

从 Stage A、Stage B、Stage C 的 winner 轨迹可以看到一个稳定趋势：

- `500`：偏大 `alpha`（`0.5`，且在 Stage B 仍是边界点）
- `1000`：中等 `alpha`（`0.25`）
- `2000`：更小的 `alpha`（`0.15 ~ 0.2`）

这说明：

> 随着数据规模增大，最优策略越来越接近“以 BC 为主、以小幅 Q 修正为辅”的 regime。

但它并没有退化到纯 BC，因为：

- `1000` 和 `2000` 的 Stage B 中，TD3BC 仍然 consistently 优于 BC。

### 10.3 `TD3BC vs BC` 的收益并不均匀

补全后的 Stage B 表明：

- `500`：TD3BC 相比 BC 几乎没有显著提升
- `1000`：TD3BC 带来稳定而可观的提升
- `2000`：TD3BC 也优于 BC，但 BC 本身方差较大

这意味着：

- 在小数据下，Q 项收益很弱
- 在中等规模数据下，Q 项最有效
- 在更大数据下，Q 项仍有用，但收益受到训练动态与数据分布宽度的影响

### 10.4 `2000` 的问题不太像“训练不够”

Stage C 的 selected checkpoint fraction 非常有信息量：

| 数据集规模 | mean selected checkpoint fraction |
| --- | ---: |
| 500 | 0.742 |
| 1000 | 0.625 |
| 2000 | 0.267 |

Shadow `2000, alpha=0.2` 的 mean fraction 也只有约 `0.273`。

这说明：

- `2000` 的最佳 checkpoint 在训练前 1/4 到 1/3 阶段就已经出现
- 后续训练并没有继续改善，反而更像是在退化

因此，`2000` 当前的核心问题并不是“训练 epoch 不够”，而更可能是：

- 更大的数据支持集引入了更宽的状态/动作分布
- 当前 TD3BC 目标与统一 sample weighting 无法持续利用这些额外数据
- 随训练进行，策略可能变得更平滑、更保守，导致超时与效率问题上升

这是一个**有根据的推断**，而不是已经被直接证明的结论。

### 10.5 Stage B 与 Stage C 的关系

Stage B 和 Stage C 的绝对数值不能直接混为一谈，因为：

- Stage B 用的是 2 seeds 和 40-episode test
- Stage C 用的是 5 seeds 和 100-episode test
- baseline success 在两阶段也不同：Stage B 为 `0.90`，Stage C 为 `0.95`

这说明两个阶段使用的是不同规模的测试 manifest，Stage B 更偏 screening，Stage C 更偏正式确认。

因此：

- Stage B 用来看趋势与 winner
- Stage C 用来写正式主表

这一分工是合理的。

---

## 11. 本轮实验的主要结论

基于 `phase0c` 已完成的所有结果，可以给出如下高置信结论：

1. `phase0b_v2` 修正了旧 `phase0` 的负趋势判断，而 `phase0c` 进一步证明：数据规模效应并非单调上升，而是在 `1000` 附近达到最佳。
2. `1000` episodes 是当前任务、当前模型、当前 TD3BC 协议下的最佳离线数据规模。
3. 最优 `alpha` 随数据规模增加而下降，从 `500` 的偏大 `alpha` 向 `1000` 的 `0.25`、再向 `2000` 的 `0.15 ~ 0.2` 迁移。
4. `TD3BC` 在 Stage B 的 held-out test 上 consistently 优于 BC，但 `500` 上优势极小，`1000` 上最稳定，`2000` 上存在但不够尖锐。
5. `2000, alpha=0.2` 没有明确击败正式 finalist `0.15`，两者更像局部平坦最优区间中的两个近邻点。

---

## 12. 局限性

尽管 `phase0c` 已经足够支撑论文主叙事，仍有几点需要诚实说明。

### 12.1 Stage C 没有同预算 BC 正式对照

Stage C 只跑 finalist，因此没有与 `TD3BC` 同预算、同 5-seed、同 100-episode 的 BC 正式对照。

这意味着：

- Stage C 的正式主表应主要用来比较不同数据集规模下的 `TD3BC`
- `TD3BC vs BC` 的直接实证证据仍主要来自补全后的 Stage B

### 12.2 `500` 上的 `alpha` 仍在边界

Stage B 中 `500` 的 winner 仍然是边界点 `0.5`，这意味着：

- 如果继续扩 `500` 的 `alpha` 上界，可能还能得到略好结果

不过由于 `500` 在整体上已经明显落后于 `1000`，这一点不影响主结论。

### 12.3 `2000` 的机制解释仍需更深入实验

目前我们知道：

- `2000` 没有继续优于 `1000`
- 最优 checkpoint 很早出现
- 最优 `alpha` 在较小区间

但更深入的问题仍未完全回答，例如：

- 是不是需要 trajectory weighting
- 是否需要更合适的 critic regularization
- 是否存在更适合大数据 regime 的 offline RL 算法

这些问题超出了本轮 `phase0c` 的范围。

---

## 13. 后续建议

如果下一步继续推进，我建议按优先级考虑：

1. 以 `1000 -> alpha=0.25` 为当前正式主配置，作为论文中的主结果点。
2. 将 `2000 -> 0.15` 与 shadow `0.2` 作为 appendix 或 supplementary，对“最优区间平坦”进行补充说明。
3. 如果继续扩展研究，不要优先继续加大 `TRAIN_EPOCHS`，而应优先考虑：
   - 大数据 regime 下的 sample weighting
   - 更适合大数据的 critic regularization
   - 不同 offline RL 算法的对照（如 IQL/CQL）
4. 若论文篇幅允许，可以把 Stage A 作为“pilot 去噪层”的方法学贡献简要写入实验设计部分。

---

## 14. 结果文件索引

本报告主要基于以下文件：

- Stage A
  - [stage_a_reval/analysis/alpha_validation.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_a_reval/analysis/alpha_validation.csv)
  - [stage_a_reval/analysis/dataset_diagnostics.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_a_reval/analysis/dataset_diagnostics.csv)
- Stage B
  - [stage_b_screen/analysis/alpha_validation.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_b_screen/analysis/alpha_validation.csv)
  - [stage_b_screen/analysis/td3bc_vs_bc_test.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_b_screen/analysis/td3bc_vs_bc_test.csv)
  - [stage_b_screen/analysis/dataset_diagnostics.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_b_screen/analysis/dataset_diagnostics.csv)
- Stage C
  - [stage_c_final/summaries/phase0b_v2_overview.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_c_final/summaries/phase0b_v2_overview.csv)
  - [stage_c_final/analysis/dataset_diagnostics.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_c_final/analysis/dataset_diagnostics.csv)
- Shadow
  - [stage_c_shadow_ep2000_a0p2 summary](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_c_shadow_ep2000_a0p2/summaries/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000_summary.json)

如需补充阅读背景，可参考：

- [td3bc_phase0b_v2_experiment_report.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_phase0b_v2_experiment_report.md:1)
- [td3bc_phase0c_experiment_design.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_phase0c_experiment_design.md:1)
