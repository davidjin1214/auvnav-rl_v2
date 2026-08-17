# TD3BC `phase0c` 实验报告

> 文档定位：TD3+BC 主线的**正式结果报告**，也是论文 §5.6 数据规模结论的数字源。`phase0b_v2`、`worldcomp teacher-gap`、`phase0c 设计文档`、`主线收口计划` 四份都把读者指到本文；本文是那条路由的终点。
> 协议范围（见 §3.1）：`efficiency_v2` + `s0` + `h4`，验证 / 终检各 `40` 回合 manifest。**与 ReBRAC 主线同奖励设定**，故 §5.7 的 ReBRAC-vs-TD3+BC 组间比较是同协议比较；其后迁到 `arrival_v2` 的是广验 v2、FQL succession 与 SAC collector 那几批（观测维度也随之由 40-D 变 48-D，见 [`rebrac_line_overview.md`](rebrac_line_overview.md) §5.2）。
> **离散度口径**：本文所有 `±` 为**总体标准差（`ddof=0`）**，见 §5.3 表下补注。
> 已知记账事项：本文的 40 回合验证集与 40 回合终检集为同一 manifest —— 即 [`data_integrity_open_items.md`](data_integrity_open_items.md) 第 ③ 条，处置已于 2026-08-16 裁决落地（走如实披露）。

## 1. 报告概述

本文档系统整理 `results/offline/td3bc/phase0c/` 下已经完成的全部关键实验与补充实验，包括：

- `stage_a_reval`
- `stage_b_screen`
- `stage_c_final`
- `stage_c_bc_final`
- `stage_c_shadow_ep2000_a0p2`
- `noisy_support_screen`
- `worldcomp_teacher_gap`

与 [td3bc_phase0c_experiment_design.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_phase0c_experiment_design.md:1) 不同，本文档聚焦**已经实际跑出的结果**，目标是把 `phase0c` 从最初的三阶段 `crosscomp` 主实验，一直到后续补做的 BC 正式对照、noisy-support 诊断和 `worldcomp teacher-gap`，统一梳理成一条完整的实验链。

本文档回答的核心问题是：

1. `phase0c` 是否进一步稳定了 `phase0b_v2` 对 size effect 的判断？
2. 在正式 5-seed 协议下，当前最优离线数据规模是 `500`、`1000` 还是 `2000`？
3. 在与 Stage C 同预算的正式协议下，TD3BC 是否仍然稳定优于 BC？
4. `1000 > 2000` 的现象究竟更像数据支持集问题，还是 TD3BC 特有问题？
5. 当离线数据来自 `worldcomp` teacher 时，deployable 上限与 teacher information gap 各有多大？

本文的核心结论可以先概括为一句话：

> 在 `phase0c` 下，`crosscomp` deployable 主线已经稳定证明 TD3BC 相对 BC 的价值，但性能并不随离线数据规模单调上升，而是在 `1000` episodes 附近达到最佳；同预算 BC 对照和 noisy-support 诊断表明，`1000 > 2000` 不是 TD3BC 特有失败，而是 deterministic 大数据支持集结构与当前 TD3BC critic/Q coupling 共同作用的结果。与此同时，`worldcomp teacher-gap` 进一步证明：当 teacher 具有局部信息优势时，deployable TD3BC 的正式最优会退回 pure BC，而 privileged critic 可以关闭约一半 teacher gap。

---

## 2. 背景与实验动机

### 2.1 `phase0b_v2` 已经修正了什么

旧版 `phase0` 曾观察到一个明显反常现象：离线数据越大，TD3BC 最优表现反而越差。  
后续 `phase0b_v2` 通过如下改动修正了这一结论：

- 将训练预算从固定 `total_steps` 改为按 epoch 对齐
- 将 replay-style 有放回采样改为 `shuffle_no_replacement`
- 将 validation 与 test 显式分离
- 采用 checkpoint 后验选择与 `alpha` 后验选择

`phase0b_v2` 已经证明：旧版“数据越大越差”的现象主要是**实验协议伪象**。

### 2.2 为什么还需要 `phase0c`

尽管 `phase0b_v2` 修正了方向，但它仍更像一个低成本 pilot，而不是论文正式实验：

- seed 数较少
- validation/test 规模有限
- `alpha` 搜索不够系统
- 缺少与正式主表同预算的 BC 对照
- 对 `2000` 的异常现象还缺少机制性诊断

因此，`phase0c` 的目标不是简单重复 `phase0b_v2`，而是进一步解决如下矛盾：

> 如何在有限算力下，既保留论文式 ablation 所需的结构化对照，又尽可能把计算资源集中在最有信息量的实验点上？

### 2.3 `phase0c` 最终演化成了什么

最初的 `phase0c` 设计只包含三阶段漏斗式协议：

- Stage A：复用已有 checkpoint，用更大 validation 降低 pilot winner 噪声
- Stage B：低成本 screening，完成 `size × alpha` 的主体 ablation
- Stage C：只推进 finalist，做正式 5-seed 确认

随着实验推进，`phase0c` 又补出了三个对主线解释非常关键的实验包：

- `stage_c_bc_final`：与 Stage C 同预算的正式 BC 对照
- `noisy_support_screen`：最小支持集扩展诊断
- `worldcomp_teacher_gap`：teacher 信息差诊断

因此，当前的 `phase0c` 不再只是“一轮 `crosscomp` size ablation”，而是：

> 以 `crosscomp` deployable 主线为中心、并用正式 BC 对照、支持集诊断和 teacher-gap 诊断补齐解释链的完整 TD3BC 收口实验。

---

## 3. 统一实验设置

### 3.1 任务与环境设定

本轮实验沿用统一任务设定：

- benchmark：`single_u10_cross_tgt15`
- task geometry：`cross_stream`
- target speed：`1.5`
- objective：`efficiency_v2`
- probe layout：`s0`
- history length：`4`
- flow：`wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

### 3.2 离线数据集

#### `crosscomp` 主线

- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone` (`500`)
- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000` (`1000`)
- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000` (`2000`)

这三组数据集的统计量约为：

| 数据集规模 | transitions | 数据集成功率 |
| --- | ---: | ---: |
| 500 episodes | 75,975 | 0.8640 |
| 1000 episodes | 152,683 | 0.8700 |
| 2000 episodes | 304,967 | 0.8855 |

#### `worldcomp` 诊断线

- `worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone` (`1000`)

### 3.3 算法与对照

本报告中涉及三类比较对象：

- `TD3BC`
- `BC (alpha=0)`
- baseline policy
  - `crosscomp` 主线中为 `crosscomp`
  - teacher-gap 诊断中为 `worldcomp`

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

## 4. 实验矩阵回顾

### 4.1 Stage A：大 validation 再评估

Stage A 不重新训练，只复用 `phase0b_v2` checkpoint，在更大的 validation 上重新排序：

- episodes：`500 / 1000 / 2000`
- alphas：`0.0 / 0.1 / 0.25 / 0.5`
- seeds：`42 / 43`
- validation episodes：`80`

### 4.2 Stage B：低成本 screening

Stage B 是 `crosscomp` 主线的主体 ablation：

- seeds：`42 / 43`
- train epochs：`64`
- validation episodes：`40`
- test episodes：`40`

初始 alpha 网格为：

| 数据集规模 | Stage B alpha 网格 |
| --- | --- |
| 500 | `0.1 / 0.5` |
| 1000 | `0.1 / 0.25 / 0.5 / 0.75` |
| 2000 | `0.0 / 0.05 / 0.1 / 0.15 / 0.2 / 0.25` |

后续又补充了：

- `500` 的 `alpha=0.0`
- `1000` 的 `alpha=0.0`

### 4.3 Stage C：正式确认

Stage C 只推进每个数据集规模一个 finalist，做正式 5-seed 确认：

- seeds：`42 / 43 / 44 / 45 / 46`
- train epochs：`96`
- validation episodes：`40`
- test episodes：`100`

finalist 来自 Stage B：

- `500 -> alpha=0.5`
- `1000 -> alpha=0.25`
- `2000 -> alpha=0.15`

### 4.4 Shadow run：`2000, alpha=0.2`

由于 Stage B 中 `2000` 的 `alpha=0.15` 与 `0.2` 很接近，因此额外补跑了一个与 Stage C 同预算的 shadow run：

- `stage_c_shadow_ep2000_a0p2`

### 4.5 Stage C 同预算 BC 正式对照

在 `crosscomp` 主实验完成后，又补充了一组与 `stage_c_final` **同预算、同 seeds、同 validation/test manifest** 的正式 BC 对照：

- `stage_c_bc_final`

### 4.6 Noisy-support 最小诊断

为回答 `1000 > 2000` 的机制问题，额外进行了一个最小 noisy-support 诊断：

- `noisy_support_screen`

screening 协议为：

- episodes：`1000 / 2000`
- seeds：`42 / 43`
- train epochs：`64`
- validation episodes：`40`
- test episodes：`40`

noisy variant 在数据采集时加入：

- `action_noise_std = 0.05`
- `action_noise_clip = 0.15`

### 4.7 `worldcomp teacher-gap` 诊断

最后，又补充了一个正式 `worldcomp` teacher-gap 实验包：

- `worldcomp_teacher_gap`

它分为：

- deployable screening
- privileged-critic screening
- deployable formal
- privileged-critic formal

formal 预算与 `phase0c stage_c` 对齐：

- 5 seeds
- 96 epochs
- 40 validation episodes
- 100 test episodes

---

## 5. `crosscomp` 主线结果

### 5.1 Stage A：pilot winner 的大 validation 复核

Stage A 只看 validation，不给正式 test 结论。

| 数据集规模 | `alpha=0` | `alpha=0.1` | `alpha=0.25` | `alpha=0.5` | winner |
| --- | ---: | ---: | ---: | ---: | --- |
| 500 | 0.2313 | 0.2500 | 0.2688 | 0.3188 | `0.5` |
| 1000 | 0.3688 | 0.5000 | 0.5063 | 0.4563 | `0.25` |
| 2000 | 0.6063 | 0.6375 | 0.5750 | 0.4750 | `0.1` |

Stage A 给出两个重要信号：

- `1000` 的 winner 从 pilot 中偏大的 `0.5` 被重新排序为 `0.25`
- `2000` 继续偏好较小 `alpha`

这说明 Stage A 的大 validation 复核是有效的，它不是机械重复 pilot 结果，而是在极低成本下先降低了后续 finalist 选择噪声。

### 5.2 Stage B：完整 screening 与 `TD3BC vs BC`

补全后的 Stage B held-out test 结果如下：

| 数据集规模 | 最优 alpha | TD3BC success | BC success | TD3BC - BC | baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.5 | 0.5625 ± 0.0625 | 0.55 ± 0.0250 | +0.0125 | 0.90 |
| 1000 | 0.25 | 0.7625 ± 0.0125 | 0.675 ± 0.0000 | +0.0875 | 0.90 |
| 2000 | 0.15 | 0.7000 ± 0.0500 | 0.60 ± 0.1500 | +0.1000 | 0.90 |

对应的 validation 赢家为：

- `500 -> alpha=0.5`
- `1000 -> alpha=0.25`
- `2000 -> alpha=0.15`

Stage B 已经给出三个清晰结论：

- `TD3BC` 在三个规模上都优于 `BC`，但 `500` 上优势极小
- `1000` 的最优点最干净、方差最小，是 screening 阶段最强的 candidate
- `2000` 的最优区间不是大 `alpha`，而是 `0.15 ~ 0.2` 这样的中小 `alpha`

### 5.3 Stage C：正式 5-seed 主表

`stage_c_final` 的正式结果如下：

| 数据集规模 | finalist alpha | success rate | return | safety cost | time | path efficiency | baseline success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.5 | 0.592 ± 0.119 | -155.92 ± 38.97 | 23.43 ± 4.19 | 111.53 ± 10.90 | 0.428 ± 0.074 | 0.95 |
| 1000 | 0.25 | 0.672 ± 0.045 | -126.24 ± 19.40 | 22.13 ± 4.50 | 103.99 ± 9.90 | 0.491 ± 0.025 | 0.95 |
| 2000 | 0.15 | 0.596 ± 0.036 | -162.53 ± 15.26 | 24.01 ± 4.60 | 115.33 ± 4.88 | 0.433 ± 0.031 | 0.95 |

> **离散度口径补注（2026-08-17，逐格回溯原始 JSON 后加）**：上表 `±` 为**总体标准差 `ddof=0`**，直接取自 `stage_c_final/analysis/analysis_summary.json` 的 `td3bc_test_success_rate_std`（如 500 格 `0.11889491…`）。**均值三格逐位吻合**原始逐种子终检值：500 = {0.74, 0.58, 0.65, 0.61, 0.38} → 0.5920；1000 = {0.68, 0.68, 0.74, 0.60, 0.66} → 0.6720；2000 = {0.61, 0.54, 0.62, 0.64, 0.57} → 0.5960（`test_selected/alpha_*/seed_4{2..6}.json`；baseline 0.95 亦与 `baselines/crosscomp_test_eval.json` 一致）。
>
> ⚠ **按样本标准差 `ddof=1` 重算则为 0.133 / 0.050 / 0.040** —— 拿本表逐种子值自行复算的读者会得到这三个数，与刊值不同，差异纯粹来自 $\sqrt{n/(n-1)}=\sqrt{5/4}$，非数据分歧。**本轮不改刊值**（改动会波及论文 §5.6 已定稿的表与相邻论证句）；此处只把口径写明。另注：广验 v2 的 §5.8 三种子 `0.878±0.051` 与 §5.5 的在线三种子 `0.46±0.39` 经复算均为 `ddof=1`——**全章存在两种离散度口径**，是否统一需另行裁决，不在本文范围。

对应相对 baseline 的 success gap 为：

- `500`: `-0.358`
- `1000`: `-0.278`
- `2000`: `-0.354`

按正式 success 排序：

1. `1000 -> 0.25`: `0.672`
2. `2000 -> 0.15`: `0.596`
3. `500 -> 0.5`: `0.592`

因此，`phase0c` 最重要的正式结论是：

> 在当前任务、模型和 TD3BC 协议下，最佳离线数据规模是 `1000`，而不是 `2000`。

### 5.4 Stage C 同预算 BC 正式对照

补充完成的 `stage_c_bc_final` 给出了与 `stage_c_final` 严格同预算的正式 BC 结果：

| 数据集规模 | BC success | TD3BC success | TD3BC - BC | BC return | TD3BC return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.524 ± 0.053 | 0.592 ± 0.119 | +0.068 | -194.71 ± 17.53 | -155.92 ± 38.97 |
| 1000 | 0.588 ± 0.070 | 0.672 ± 0.045 | +0.084 | -170.17 ± 27.15 | -126.24 ± 19.40 |
| 2000 | 0.534 ± 0.079 | 0.596 ± 0.036 | +0.062 | -183.18 ± 33.13 | -162.53 ± 15.26 |

这组结果的意义非常关键：

- 在正式 5-seed、100-episode 预算下，TD3BC 在三个规模上都优于 BC
- 但 BC 自己也在 `1000` 达到最好，并在 `2000` 回落

因此：

> `1000 > 2000` 不是 TD3BC 独有问题，而是在当前任务/数据/训练协议下更一般的现象。

### 5.5 `2000, alpha=0.15` 与 `0.2` 的 shadow 对照

`stage_c_shadow_ep2000_a0p2` 的正式结果为：

| 配置 | success | std | return | safety cost | time | path efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2000, alpha=0.15` | 0.596 | 0.036 | -162.53 | 24.01 | 115.33 | 0.433 |
| `2000, alpha=0.2` | 0.602 | 0.073 | -157.24 | 21.72 | 113.63 | 0.440 |

正确解读应当是：

- `0.2` 没有明显击败 `0.15`
- success 仅高 `0.006`
- 但方差明显更大

因此，`0.15` 作为 Stage C 正式 finalist 是合理的；`2000` 附近的最优区间更像 `0.15 ~ 0.2` 的平坦窄区间。

### 5.6 `crosscomp` 主线的失败模式

将 Stage C 五个 seed、每个 seed 100 个 test episode 聚合后，TD3BC 的终止统计为：

| 数据集规模 | goal | out_of_bounds | timeout | depth_hold_failure |
| --- | ---: | ---: | ---: | ---: |
| 500 | 296 | 122 | 81 | 1 |
| 1000 | 336 | 105 | 59 | 0 |
| 2000 (`0.15`) | 298 | 116 | 81 | 5 |
| 2000 shadow (`0.2`) | 301 | 120 | 75 | 4 |

`stage_c_bc_final` 的对应统计为：

| 数据集规模 | goal | out_of_bounds | timeout | depth_hold_failure |
| --- | ---: | ---: | ---: | ---: |
| 500 BC | 262 | 128 | 109 | 1 |
| 1000 BC | 294 | 107 | 97 | 2 |
| 2000 BC | 267 | 137 | 89 | 7 |

这些统计共同说明：

- `1000` 的优势主要体现为更少的 `timeout` 和更少的 `out_of_bounds`
- `2000` 的问题不主要是“严重失控”，而是没有更高效地完成任务
- 即使加入 Q 修正，`2000` 也没能追回 `1000` 的优势

---

## 6. `noisy_support_screen`：支持集结构诊断

`noisy_support_screen` 的跨 variant 对照结果如下：

| 数据集 | 版本 | 最优 alpha | TD3BC success | BC success | TD3BC - BC |
| --- | --- | ---: | ---: | ---: | ---: |
| 1000 | deterministic | 0.25 | 0.7625 | 0.6750 | +0.0875 |
| 1000 | noisy | 0.0 | 0.5125 | 0.5125 | 0.0000 |
| 2000 | deterministic | 0.15 | 0.7000 | 0.6000 | +0.1000 |
| 2000 | noisy | 0.0 | 0.7375 | 0.7375 | 0.0000 |

对应的跨 variant success 变化为：

- `1000`: TD3BC `-0.2500`，BC `-0.1625`
- `2000`: TD3BC `+0.0375`，BC `+0.1375`

这组结果说明：

- `1000` 上加噪声不是小扰动，而是明显破坏了数据可学性
- `2000` 上加噪声反而能明显改善闭环表现
- 但这种改善主要体现在 `BC` 上，而不是体现在正 `alpha` 的 TD3BC 上

因此，`noisy_support_screen` 给出两个高价值结论：

1. deterministic `2000` 的确存在支持集结构问题，因为 noisy variant 下 `BC` 明显从 `0.6000` 提升到了 `0.7375`
2. broadened support 并没有让当前 TD3BC 在 noisy 数据上继续受益；相反，两组 noisy 数据的最优点都退回了 `alpha=0.0`

换句话说：

> deterministic `2000` 的问题不是“数据坏了”，而是“支持集结构不够鲁棒”；而当前 TD3BC 在 broadened support 上又不能稳定利用 critic/Q 引导。

---

## 7. `worldcomp_teacher_gap`：teacher 信息差诊断

`worldcomp teacher-gap` 的详细结果另见 [td3bc_worldcomp_teacher_gap_experiment_report.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_worldcomp_teacher_gap_experiment_report.md:1)。这里给出其与 `phase0c` 主线最相关的结论。

### 7.1 Screening 结果

#### Deployable screening

| alpha | mean val success | std |
| --- | ---: | ---: |
| 0.0 | 0.7875 | 0.0625 |
| 0.1 | 0.7250 | 0.0750 |
| 0.25 | 0.7250 | 0.1000 |
| 0.5 | 0.5125 | 0.0125 |

Deployable screening 的最优点退回到了 `alpha=0.0`。

#### Privileged-critic screening

| alpha | mean val success | std |
| --- | ---: | ---: |
| 0.1 | 0.8125 | 0.0875 |
| 0.25 | 0.6500 | 0.1250 |

Privileged-critic screening 的最优点转向 `alpha=0.1`，比 deployable screening 高出 `+0.025` success。

### 7.2 Formal 结果

Formal 5-seed 结果如下：

| 协议 | best alpha | success | std | return | safety cost | time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deployable | 0.0 | 0.858 | 0.080 | -14.29 | 7.91 | 63.19 |
| privileged-critic | 0.1 | 0.922 | 0.086 | 16.49 | 5.73 | 54.33 |
| `worldcomp` baseline | - | 0.990 | - | 32.19 | - | - |

对应的 success gap 为：

- baseline - deployable = `0.132`
- baseline - privileged = `0.068`
- privileged - deployable = `+0.064`

也就是说，privileged critic 关闭了约：

- `0.064 / 0.132 ≈ 48.5%`

的 deployable teacher gap。

### 7.3 终止类型

formal 500 个 test episodes 聚合后：

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

privileged critic 相比 deployable：

- 多了 `32` 个成功到达
- 少了 `23` 个越界
- 少了 `8` 个超时

### 7.4 这组结果意味着什么

`worldcomp teacher-gap` 给出的结论非常清楚：

- 在 deployable setting 下，当前 TD3BC 的正式最优点退回到 `alpha=0.0`
- 一旦 critic 在训练期能够访问 `privileged_obs`，最佳 `alpha` 重新变成 `0.1`
- privileged critic 能显著关闭 deployable teacher gap，但仍不能完全追平 baseline

因此：

> `worldcomp` 主线上的一部分性能差距，确实来自局部 teacher information gap；但剩余差距仍然同时包含算法与策略层面的限制。

---

## 8. 综合分析

### 8.1 `phase0c` 最关键的新结论：存在最佳中间规模

`phase0b_v2` 的意义在于纠正旧版负趋势；`phase0c` 则进一步指出：

- 修正协议之后，数据规模效应不是简单单调上升
- 在当前任务、模型和 TD3BC 协议下，性能在 `1000` 附近达到最佳

这比“越大越好”更有研究价值，也更符合很多离线 RL 实际问题的经验规律。

### 8.2 `1000 > 2000` 不是 TD3BC 特有现象

这是补完 `stage_c_bc_final` 后最关键的新结论。

在与 Stage C 同预算的 BC 对照下，success 仍然是：

- `500`: `0.524`
- `1000`: `0.588`
- `2000`: `0.534`

因此，`2000` 的回落至少有一部分位于更一般的层面：

- 数据支持集结构
- 行为分布覆盖与有效可克隆性
- 当前训练协议对更大确定性数据的利用方式

把 `1000 > 2000` 简单归咎为“TD3BC 的 Q 项有问题”，已经与现有证据不符。

### 8.3 最优 `alpha` 随规模增大而下降

从 Stage A、Stage B、Stage C 以及后续诊断可以看到一个稳定趋势：

- `500`：偏大 `alpha`
- `1000`：中等 `alpha`
- `2000`：更小的 `alpha`
- `worldcomp deployable`：退回 `alpha=0.0`
- `worldcomp privileged-critic`：恢复到较小正 `alpha=0.1`

这说明：

> 随着数据规模增大，或随着 deployable 观测瓶颈增强，最优策略越来越接近“以 BC 为主、以小幅 Q 修正为辅”的 regime。

### 8.4 当前主线的两个主要瓶颈已经被拆开了

截至当前 `phase0c`，TD3BC 主线的剩余问题已经被拆成两类：

#### 瓶颈一：大 deterministic 数据的支持集结构问题

证据来自：

- `1000 > 2000` 在 BC 下同样成立
- `2000 noisy` 的 BC 从 `0.6000` 提升到 `0.7375`

这说明 deterministic `2000` 的问题不是“数据坏了”，而是新增样本没有继续转化为更鲁棒的可学习支持。

#### 瓶颈二：teacher information gap

证据来自：

- `worldcomp deployable` 最优退回 pure BC
- privileged critic 可将 success 从 `0.858` 提升到 `0.922`

这说明 deployable 观测下 critic 无法稳定获得足够好的 improvement signal，而 privileged 信息能显著缓解这一问题。

### 8.5 当前 TD3BC 主线应如何被理解

把所有结果放在一起，当前最准确的描述应当是：

> `crosscomp` 主线告诉我们：大 deterministic 数据 regime 下的支持集结构与样本利用方式是关键问题。  
> `worldcomp` 主线告诉我们：当 teacher 本身具有局部信息优势时，deployable critic 的信息瓶颈同样是关键问题。  
> 因此，当前 TD3BC 的剩余差距不是单一原因造成的，而是支持集结构问题与 teacher information gap 共同作用的结果。

---

## 9. 局限性

尽管 `phase0c` 已经足够支撑主线收口，仍有几个边界条件需要说明：

1. `noisy_support_screen` 仍然只是 `2-seed / 40-episode` screening，足够支撑机制诊断，但不应与 Stage C 主表完全同权解释。
2. `worldcomp teacher-gap` 只在 `1000` episodes 上做了正式诊断，没有再做 size sweep。
3. privileged-critic 是训练期 critic 信息增强，不代表推理期 deployable actor 能直接访问 privileged 信息。
4. `500` 上的最优 `alpha` 仍然靠近边界，但这一点不影响主结论，因为 `500` 整体上已明显落后于 `1000`。

---

## 10. 最终结论

基于 `phase0c` 已完成的全部实验，可以给出如下高置信结论：

1. `phase0c` 进一步稳定了 `phase0b_v2` 对 size effect 的判断：数据规模效应不是旧 `phase0` 的伪象，但也不是单调上升，而是在 `1000` episodes 左右达到最佳。
2. 在正式 5-seed、100-episode 协议下，TD3BC 的最优 `crosscomp` 配置是 `1000 -> alpha=0.25`。
3. 与 Stage C 同预算的正式 BC 对照证明：TD3BC 在 `500/1000/2000` 三个规模上都优于 BC，但 `1000 > 2000` 在 BC 下同样成立，因此它不是 TD3BC 特有问题。
4. `2000, alpha=0.2` 没有明确击败正式 finalist `0.15`，两者更像局部平坦最优区间中的两个近邻点。
5. `noisy_support_screen` 证明 deterministic `2000` 存在支持集结构问题，因为 broadened support 可以显著提升 `2000` 的 BC；但当前 TD3BC 在 noisy 数据上会退回 pure BC，说明 critic/Q coupling 在 broadened support 上不稳。
6. `worldcomp teacher-gap` 证明 teacher information gap 是真实且重要的：privileged critic 可将 success 从 `0.858` 提升到 `0.922`，关闭约 `48.5%` 的 deployable teacher gap。
7. 因此，当前 TD3BC 主线的剩余问题已经被拆得比较清楚：
   - `crosscomp` 暴露的是大 deterministic 数据 regime 下的支持集结构问题
   - `worldcomp` 暴露的是 deployable critic 的信息瓶颈问题

一句话总结：

> `phase0c` 已经完成了 TD3BC 主线收口：它不仅给出了当前最优 deployable 配置，也把“为什么 `2000` 没有继续变好”和“为什么更强 teacher 仍有 gap”这两个核心问题拆成了可以被后续 ReBRAC 明确回答的两个独立瓶颈。

---

## 11. 结果文件索引

本报告主要基于以下文件：

- `crosscomp` 主线
  - [stage_a_reval/analysis/alpha_validation.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_a_reval/analysis/alpha_validation.csv)
  - [stage_a_reval/analysis/dataset_diagnostics.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_a_reval/analysis/dataset_diagnostics.csv)
  - [stage_b_screen/summaries/phase0b_v2_overview.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_b_screen/summaries/phase0b_v2_overview.csv)
  - [stage_b_screen/analysis/td3bc_vs_bc_test.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_b_screen/analysis/td3bc_vs_bc_test.csv)
  - [stage_c_final/summaries/phase0b_v2_overview.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_c_final/summaries/phase0b_v2_overview.csv)
  - [stage_c_bc_final/summaries/phase0b_v2_overview.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_c_bc_final/summaries/phase0b_v2_overview.csv)
  - [stage_c_shadow_ep2000_a0p2 summary](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/stage_c_shadow_ep2000_a0p2/summaries/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000_summary.json)
- `noisy_support_screen`
  - [analysis_compare/noisy_support_comparison.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/noisy_support_screen/analysis_compare/noisy_support_comparison.csv)
- `worldcomp_teacher_gap`
  - [analysis_compare/final_comparison.csv](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/worldcomp_teacher_gap/analysis_compare/final_comparison.csv)
  - [analysis_compare/conclusions.txt](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/results/offline/td3bc/phase0c/worldcomp_teacher_gap/analysis_compare/conclusions.txt)

如需补充阅读背景，可参考：

- [td3bc_phase0b_v2_experiment_report.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_phase0b_v2_experiment_report.md:1)
- [td3bc_phase0c_experiment_design.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_phase0c_experiment_design.md:1)
- [td3bc_worldcomp_teacher_gap_experiment_report.md](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/docs/td3bc_worldcomp_teacher_gap_experiment_report.md:1)
