# TD3BC `phase0c` 实验设计文档

## 1. 文档目的

本文档用于说明 `scripts/run_offline_td3bc_phase0c.sh` 所定义的 `phase0c` 实验协议，重点覆盖以下内容：

- 为什么需要在 `phase0b_v2` 之后继续设计 `phase0c`
- `phase0c` 想回答哪些科学问题
- 实验的总体结构、各阶段作用、默认配置与选择规则
- 训练、验证、测试、checkpoint 选择、`alpha` 选择的具体方法
- 如何在有限计算预算下，同时兼顾论文式 ablation 的完整性与实验结论的可靠性

本文档**不包含实验结果本身**。待 `phase0c` 实验完成后，可在本文档预留章节中补充结果、图表和讨论。

---

## 2. 背景与设计动机

### 2.1 `phase0b_v2` 已经回答了什么

`phase0b_v2` 的主要作用，是修正旧版 `phase0` size ablation 中的关键协议问题，并重新判断“数据集规模扩大后，TD3BC 是否真的退化”。

基于已经完成的 `phase0b_v2` 实验，可以得到两个高置信判断：

1. 旧 `phase0` 中“数据越大，TD3BC 越差”的现象主要是实验协议伪象，而不是算法本质。
2. 在更公平的训练协议下，随着数据集规模扩大，TD3BC 与 BC 的表现都明显改善。

换句话说，`phase0b_v2` 已经基本完成了“纠错”工作：它告诉我们，旧结论不能成立。

### 2.2 为什么还需要 `phase0c`

尽管 `phase0b_v2` 纠正了方向，但它仍然存在两个现实问题：

1. 当前取回的 `phase0b_v2` 更接近一个低成本 pilot，而不是论文正式实验。
2. 如果直接把 `phase0b_v2` 扩展成“全数据规模 × 全 `alpha` 网格 × 5 seeds × 大 validation/test”的全因子实验，计算成本会非常高。

因此，`phase0c` 的目标不是简单重复 `phase0b_v2`，而是进一步解决如下矛盾：

> 如何在有限算力下，既保留论文式 ablation 所需的结构化对照，又尽可能把计算资源集中在真正有价值的实验点上？

### 2.3 `phase0c` 的核心设计思想

`phase0c` 采用三阶段漏斗式协议：

- **Stage A**：尽量复用已有 `phase0b_v2` checkpoint，用更大的 validation 重新排序，先降低选择噪声，不重新训练。
- **Stage B**：仅在最有希望的 `alpha` 区间做低成本 screening，得到 size-ablation 的主体图景。
- **Stage C**：每个数据集规模只晋级一个 finalist 到正式 5-seed 确认实验，用于论文主表与最终结论。

这种设计的本质是：

- 不在早期把算力浪费在明显不优的配置上；
- 不把 pilot 阶段的小样本噪声直接当成正式结论；
- 在正式结果阶段保留足够的 seed 数和 test 规模。

---

## 3. 科学问题与实验目标

`phase0c` 希望在 `phase0b_v2` 的基础上，进一步系统回答以下问题：

1. 在修正训练预算定义之后，**数据集规模**如何影响 TD3BC 的最优性能？
2. 最优 `alpha` 是否随数据集规模发生系统性变化？
3. 随着离线数据规模变大，TD3BC 相对于纯 BC 的增益是否会缩小？
4. 在给定任务上，TD3BC 是否仍然明显落后于行为基线 `crosscomp`，如果是，差距主要体现在哪些指标上？
5. 如何在有限算力下构造一个既适合论文写作、又能保持结论可靠性的实验协议？

这些问题中，前四个是算法与任务层面的科学问题，第五个是实验设计层面的工程问题。

---

## 4. 实验对象与统一任务设定

`phase0c` 沿用 `phase0b_v2` 的实验对象与任务定义，确保两轮实验具有可比性。

### 4.1 算法对象

实验主体算法为：

- **TD3BC**：以 `alpha` 控制 BC 项与 Q 引导项的相对权衡
- **BC (`alpha=0`)**：作为 TD3BC 的关键对照组
- **`crosscomp` baseline**：作为离线数据来源行为策略与参考上界之一

### 4.2 任务默认配置

`phase0c` 脚本中的默认任务配置为：

- benchmark key：`single_u10_cross_tgt15`
- flow file：`wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`
- task geometry：`cross_stream`
- target speed：`1.5`
- reward objective：`efficiency_v2`
- probe layout：`s0`
- history length：`4`

### 4.3 离线数据配置

默认离线数据来源为：

- dataset policy：`crosscomp`
- base dataset name：`crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone`
- size ablation episodes：`500 / 1000 / 2000`
- action noise std：`0.0`

因此，`phase0c` 的 size ablation 仍然围绕同一类确定性 `crosscomp` 数据集展开，只是实验协议更系统、计算资源分配更精细。

---

## 5. 总体协议概览

`phase0c` 的入口脚本是：

- [run_offline_td3bc_phase0c.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0c.sh:1)

该脚本提供四种模式：

- `MODE=stage_a`
- `MODE=stage_b`
- `MODE=stage_c`
- `MODE=all`

其中：

- `stage_a` 只做复用与再评估；
- `stage_b` 做低成本筛选；
- `stage_c` 做正式确认；
- `all` 依次串联三个阶段。

### 5.1 与 `phase0b_v2` 的关系

`phase0c` 并不重新发明训练逻辑，而是基于 `phase0b_v2` 的统一流程进行封装和调度。底层仍然复用：

- [run_offline_td3bc_phase0b_v2.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0b_v2.sh:1)
- [train_offline.py](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/train_offline.py:625)
- [analyze_offline_td3bc_phase0b_v2.py](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/analyze_offline_td3bc_phase0b_v2.py:1)

这意味着：

- `phase0c` 沿用 `phase0b_v2` 的训练、验证、测试、汇总与分析格式；
- 新增的主要是阶段性实验设计与资源调度逻辑，而不是另起一套 incompatible 的结果体系。

---

## 6. 训练与选择的统一方法学

在介绍三个阶段之前，需要先说明 `phase0c` 统一遵循的训练与模型选择原则。

### 6.1 离线训练预算定义

`phase0c` 默认使用：

- `sampling_mode=shuffle_no_replacement`
- `drop_last_batch=0`

对应底层实现位于：

- [replay.py](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/auv_nav/replay.py:286)
- [train_offline.py](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/train_offline.py:652)

其含义是：

- 每个 epoch 对整个离线数据集打乱一次；
- 然后以 mini-batch 方式无放回遍历一次数据；
- 默认保留最后一个不足整批的 batch。

因此，在默认设置下：

> 每个 epoch 都对应一次完整的数据覆盖。

如果某个数据集有 `N` 条 transitions、batch size 为 `B`，则理论上的：

- `steps_per_epoch = ceil(N / B)`

这与旧版 `phase0` 采用的“固定 gradient steps + 有放回均匀随机采样”有本质不同。`phase0c` 继承 `phase0b_v2` 的这一设计，是为了避免“大数据集在固定步数下被训稀”的问题再次出现。

### 6.2 Checkpoint 保存与后验选择

`train_offline.py` 在离线训练期间会保存：

- `agent_step_XXXXXXXX.pt`
- `agent_final.pt`

对应逻辑位于 [train_offline.py](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/train_offline.py:545)。

这样做的原因是：

- 训练期间不必频繁进行 CPU 评估；
- 可以在训练结束后对多个 checkpoint 统一做 validation；
- 再从 validation 结果中后验选择最佳 checkpoint。

### 6.3 单个 run 内的 checkpoint 选择规则

对于给定的“数据集规模 + `alpha` + seed”组合，`phase0b_v2/phase0c` 的 checkpoint 选择规则为：

1. 优先最大化 `eval_success_rate`
2. 若成功率相同，优先最大化 `eval_return`
3. 若仍相同，优先最小化 `eval_safety_cost`
4. 若仍相同，优先最小化 `eval_time_s`

该规则定义在 [run_offline_td3bc_phase0b_v2.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0b_v2.sh:471)。

这是一种明确的、可复现实验协议，而不是人工挑选“看起来最好”的 checkpoint。

### 6.4 跨 seed 的 `alpha` 选择规则

对于同一数据集规模下的多个 `alpha` 候选，`best_alpha` 的选择规则为：

1. 最大化 `mean_val_success_rate`
2. 若相同，最大化 `mean_val_return`
3. 若仍相同，最小化 `mean_val_safety_cost`
4. 若仍相同，最小化 `mean_val_time_s`

对应逻辑位于 [run_offline_td3bc_phase0b_v2.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0b_v2.sh:553)。

这一定义在 `phase0c` 中同样适用。

### 6.5 Validation 与 Test 分离

`phase0c` 保持了 `phase0b_v2` 已经建立的基本原则：

- validation 用于 checkpoint 选择与 `alpha` 选择
- test 只用于最终汇报

因此：

- Stage A 主要服务于“重新排序与稳定性检查”，并不作为最终 test 结论；
- Stage B 的 test 结果可以用于 pilot 式比较，但不作为最终正式结论；
- Stage C 才是面向正式结论的主实验阶段。

---

## 7. Stage A：复用已有结果的大 validation 再评估

### 7.1 设计目标

Stage A 的目标是：

> 在不重新训练的前提下，尽量复用 `phase0b_v2` 已有 checkpoint，用更大的 validation 集合降低 winner 选择噪声。

这是整个 `phase0c` 中最省算力的一步。它并不试图直接给出最终答案，而是先判断：

- `phase0b_v2` 中的 winner 是否稳定；
- 某些优胜配置是否只是小 validation 集导致的噪声；
- 是否有必要在后续阶段继续扩某些 `alpha` 区间。

### 7.2 默认配置

Stage A 默认参数为：

- 数据集规模：`500 / 1000 / 2000`
- `alpha`：`0.0 / 0.1 / 0.25 / 0.5`
- seeds：`42 / 43`
- validation manifest episodes：`80`

对应脚本变量位于 [run_offline_td3bc_phase0c.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0c.sh:71)。

### 7.3 数据来源与执行方式

Stage A 直接复用：

- `PHASE0B_V2_CHECKPOINT_ROOT=checkpoints/offline/td3bc/phase0b_v2`

并在新的结果目录下运行：

- validation
- alpha selection
- analysis

它**不重新训练**，也**不做正式 test**。

### 7.4 为什么 Stage A 必须存在

如果没有 Stage A，那么 `phase0b_v2` 小 validation 上的 winner 会直接进入后续重训流程，可能带来两个问题：

1. 把随机优势配置误判为真正最优；
2. 把大量算力继续投入到并不值得扩展的 `alpha` 区间。

因此，Stage A 的意义是“用很低成本提高后续实验的决策质量”。

### 7.5 预期输出

Stage A 的主要输出包括：

- 更大 validation 下的 `best checkpoint`
- 每个数据集规模的新 `best alpha`
- 对 winner 稳定性的诊断分析

对应默认结果目录为：

- `results/offline/td3bc/phase0c/stage_a_reval`

---

## 8. Stage B：低成本、面向 ablation 的定向筛选

### 8.1 设计目标

Stage B 是 `phase0c` 的主体阶段，其目标是：

> 用较低训练成本完成结构化的 size × alpha ablation，但避免把资源浪费在明显不优的网格区域上。

也就是说，Stage B 既是一个“论文式实验”，又是一个“资源敏感的筛选器”。

### 8.2 为什么不采用全因子完整网格

如果对 `500 / 1000 / 2000` 三个数据集规模都统一扫描一个完整大网格，例如：

- `alpha = 0.0 / 0.05 / 0.1 / 0.15 / 0.2 / 0.25 / 0.5 / 0.75 / 1.0`
- 5 seeds
- 较大的 validation/test

那么总实验规模会迅速膨胀，且其中很多配置从 `phase0b_v2` 的先验来看几乎不可能最优。

Stage B 的策略是：

- **保留 size ablation 结构**
- **缩小每个 size 需要扫描的 `alpha` 区间**
- **只用 2 个 seed 做 screening**

这样既保留论文中解释趋势所需的结构，又显著降低总计算量。

### 8.3 默认配置

Stage B 默认参数如下：

- seeds：`42 / 43`
- train epochs：`64`
- checkpoint every epochs：`8`
- validation manifest episodes：`40`
- test manifest episodes：`40`

分数据集规模的 `alpha` 网格为：

| 数据集规模 | Stage B `alpha` 网格 |
| --- | --- |
| 500 episodes | `0.1 / 0.5` |
| 1000 episodes | `0.1 / 0.25 / 0.5 / 0.75` |
| 2000 episodes | `0.0 / 0.05 / 0.1 / 0.15 / 0.2 / 0.25` |

对应脚本变量位于 [run_offline_td3bc_phase0c.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0c.sh:78)。

### 8.4 这些网格是如何确定的

Stage B 的 `alpha` 设计不是拍脑袋决定，而是由 `phase0b_v2` 的已有发现驱动：

- 对 `500 / 1000` 而言，pilot 结果显示 winner 接近较大 `alpha`，因此需要继续向较大区间探索；
- 对 `2000` 而言，pilot 结果显示最优点已经明显回到小 `alpha` 区间，因此应在 `0.0 ~ 0.25` 做细扫；
- `500` 的网格最窄，是因为它更像低资源下的对照组，而不是主力候选；
- `2000` 的网格最细，是因为它最有可能产生最终最佳结果。

### 8.5 Stage B 的执行流程

对每个数据集规模，Stage B 依次执行：

1. `train`
2. `validate`
3. `summarize`

所有数据集规模完成后，再统一执行：

4. `analyze`

因此，Stage B 会产出：

- checkpoint
- validation 选择结果
- selected-model test
- per-dataset summary
- 全局分析表与图

默认输出目录为：

- checkpoints：`checkpoints/offline/td3bc/phase0c/stage_b_screen`
- results：`results/offline/td3bc/phase0c/stage_b_screen`

### 8.6 Stage B 在论文中的定位

Stage B 的主要用途是：

- 生成 `dataset size × alpha` 的主体趋势图
- 支持“最优 `alpha` 随数据规模变化”的论述
- 识别每个 size 最值得进入正式确认阶段的 finalist

但是，Stage B 仍然只有 2 个 seeds，因此它更适合：

- 论文中的 ablation 图与趋势图
- 不适合作为最终主表的唯一依据

---

## 9. Stage C：面向正式结论的 finalist 确认

### 9.1 设计目标

Stage C 的目标是：

> 在每个数据集规模上，只保留一个 finalist `alpha`，并对其进行更高成本、更高可信度的正式确认。

这一步是 `phase0c` 最接近论文主实验的部分。

### 9.2 默认配置

Stage C 默认参数为：

- seeds：`42 / 43 / 44 / 45 / 46`
- train epochs：`96`
- checkpoint every epochs：`4`
- validation manifest episodes：`40`
- test manifest episodes：`100`

对应脚本变量位于 [run_offline_td3bc_phase0c.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0c.sh:91)。

### 9.3 finalist `alpha` 的来源

对于每个数据集规模，Stage C 的 finalist `alpha` 按如下优先级确定：

1. 若显式设置 `STAGE_C_ALPHA_<episodes>`，则使用人工指定值；
2. 否则读取 Stage B 的 summary 中的 `best_alpha`；
3. 若 Stage B summary 不存在，则使用 fallback 默认值。

默认 fallback 为：

| 数据集规模 | fallback `alpha` |
| --- | ---: |
| 500 episodes | 0.5 |
| 1000 episodes | 0.5 |
| 2000 episodes | 0.1 |

该逻辑位于 [run_offline_td3bc_phase0c.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0c.sh:101) 和 [run_offline_td3bc_phase0c.sh](/C:/Users/jinxiang/OneDrive/我的/Code/new_off_rl/rl_v2/scripts/run_offline_td3bc_phase0c.sh:159)。

### 9.4 为什么 Stage C 只保留一个 finalist

原因很直接：

- Stage C 的 seeds 和 test 规模都显著更大；
- 若继续保留多个 `alpha` 共同晋级，计算成本会迅速膨胀；
- 从论文写作角度看，每个数据集规模最终需要的是“正式确认后的最优配置”，而不是再次做一轮完整大网格搜索。

因此，Stage C 是一个**confirmation stage**，不是再次进行 exploratory sweep。

### 9.5 默认输出

Stage C 的输出目录为：

- checkpoints：`checkpoints/offline/td3bc/phase0c/stage_c_final`
- results：`results/offline/td3bc/phase0c/stage_c_final`

它将生成：

- 每个数据集规模的 finalist checkpoint 与 test 结果
- 5-seed 汇总统计
- 相对 BC 和 baseline 的正式比较

Stage C 的结果应该作为论文主表和最终结论的主要依据。

---

## 10. 计算资源效率与设计取舍

`phase0c` 的价值不仅在于它能跑出结果，还在于它用相对节制的预算组织结果。

### 10.1 为什么 Stage A 几乎是“白赚”的

Stage A 复用已有 `phase0b_v2` checkpoint，不重新训练，因此它主要消耗的是：

- validation 评估时间

而不是：

- GPU 训练时间

这使得它非常适合作为“低成本去噪层”。

### 10.2 为什么 Stage B 是论文式 ablation 的最佳折中

Stage B 保留了：

- 三个数据集规模
- 每个数据集规模多个 `alpha`
- 验证与测试流程

同时压缩了：

- seeds 数量
- 训练 epoch 数
- `alpha` 搜索空间

因此它更像一个“有结构的 pilot”，既能看趋势，又不至于贵到难以落地。

### 10.3 为什么 Stage C 不能再省

Stage C 是正式结论层。如果在这里继续压缩：

- seeds 太少，会导致方差不可控；
- test episode 太少，会导致最终表格不稳；
- checkpoint 过稀，会导致 best model 被漏掉。

因此，Stage C 已经是“论文可信度”与“计算预算”之间相对合理的下限。

---

## 11. 预注册分析计划

为避免实验跑完后再临时修改解释口径，`phase0c` 建议采用如下分析顺序。

### 11.1 Stage A 的解读原则

Stage A 只回答一个问题：

> 现有 winner 在更大 validation 上是否稳定？

它不承担最终性能结论，也不作为正式 test 结果引用。

### 11.2 Stage B 的解读原则

Stage B 主要用于回答：

1. 随着数据集规模变化，最优 `alpha` 如何移动？
2. `alpha`-performance 曲线是否在不同 size 下发生系统性形变？
3. 哪些区域值得进入正式确认？

Stage B 适合展示：

- 热力图
- validation 曲线
- 小规模 test 趋势

但不应把 Stage B 直接等同于最终结论。

### 11.3 Stage C 的解读原则

Stage C 是最终主结论来源，主要回答：

1. 不同数据集规模下，TD3BC 的正式表现如何？
2. 最优 `alpha` 随数据规模如何变化？
3. TD3BC 相对 BC 的真实边际收益有多大？
4. TD3BC 与 `crosscomp` baseline 还存在多大差距？

### 11.4 建议报告的指标

最终建议至少报告以下指标：

- success rate
- return
- safety cost
- episode time
- path length
- progress ratio
- path efficiency
- termination counts

其中：

- success rate 是主指标；
- return 是综合性能指标；
- safety cost 与 termination counts 用于解释失败模式；
- path efficiency 用于体现“高效航行”目标。

---

## 12. 风险点与应对策略

### 12.1 风险一：Stage B 的 winner 仍受 2-seed 噪声影响

这是 Stage B 的固有限制。对应策略是：

- 先用 Stage A 降低噪声；
- 若 Stage B 的 top-1 与 top-2 非常接近，可手动设置 `STAGE_C_ALPHA_*`；
- 不要机械地把 Stage B 自动 winner 当成绝对真理。

### 12.2 风险二：winner 落在网格边界

如果某个 size 的最优 `alpha` 落在当前搜索边界，说明网格仍未完全覆盖真正最优区间。应对策略是：

- 只对该 size 的单侧边界继续补点；
- 不要重新扩大所有 size 的完整网格。

### 12.3 风险三：最终结果仍然显著落后于 baseline

即便 `phase0c` 获得更稳定的 TD3BC 结果，也不能保证它一定接近 `crosscomp` baseline。若最终仍存在明显差距，解释时应区分：

- 是 size trend 已经转正但绝对性能仍不足；
- 还是 TD3BC 相对 BC 的增益本身已经很小；
- 抑或任务本身更接近“高质量行为克隆”而非“强 Q 修正”收益场景。

### 12.4 风险四：Stage C finalist 选择依赖 Stage B 自动结果

为避免错误 finalist 被自动推进，`phase0c` 保留了人工覆盖接口：

- `STAGE_C_ALPHA_500`
- `STAGE_C_ALPHA_1000`
- `STAGE_C_ALPHA_2000`

这使得实验者可以在必要时用人工审查结果覆盖自动选择。

---

## 13. 推荐运行方式

在 Colab 或交互式环境中，推荐最小配置如下：

```python
import os

os.environ["PYTHON_BIN"] = "python3"
os.environ["BASE_DATASET_NAME"] = "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone"
os.environ["DEVICE"] = "cuda"
os.environ["EVAL_WORKERS"] = "6"
os.environ["EVAL_WORKER_DEVICE"] = "cpu"
```

然后按阶段运行：

```python
os.environ["MODE"] = "stage_a"
!bash scripts/run_offline_td3bc_phase0c.sh
```

```python
os.environ["MODE"] = "stage_b"
!bash scripts/run_offline_td3bc_phase0c.sh
```

```python
os.environ["MODE"] = "stage_c"
!bash scripts/run_offline_td3bc_phase0c.sh
```

若需一口气运行全部阶段：

```python
os.environ["MODE"] = "all"
!bash scripts/run_offline_td3bc_phase0c.sh
```

---

## 14. 待补充的结果章节

待实验跑完后，建议在本文档后续补充如下内容。

### 14.1 Stage A 结果

- 更大 validation 下的 winner 是否稳定
- 是否有配置在小 validation 下被高估或低估

### 14.2 Stage B 结果

- `dataset size × alpha` 的 validation 热力图
- 每个数据集规模的最佳 `alpha`
- 小规模 test 趋势

### 14.3 Stage C 结果

- 5-seed 正式 test 主表
- TD3BC vs BC vs baseline 对比
- 失败模式统计

### 14.4 综合讨论

- 数据规模效应是否稳定
- 最优 `alpha` 是否随数据规模单调变化
- TD3BC 的 Q 项在大数据 regime 中是否仍然必要

---

## 15. 结论

`phase0c` 是在 `phase0b_v2` 基础上的进一步实验协议升级。它不是简单增加实验数量，而是围绕以下原则构建：

1. 先用低成本再评估减少噪声；
2. 再用定向 screening 完成论文式 ablation 主体；
3. 最后只对 finalist 做高可信度正式确认。

从实验设计角度看，`phase0c` 的核心贡献在于：

- 它继承了 `phase0b_v2` 在方法学上的修正；
- 同时进一步解决了“科学完整性”与“计算可承受性”之间的矛盾。

因此，若 `phase0c` 按计划顺利完成，它应当能够同时支撑：

- 论文中的 size-ablation 图表；
- 最优 `alpha` 与 BC/Q 权衡的讨论；
- 更可靠的正式主结果表。
