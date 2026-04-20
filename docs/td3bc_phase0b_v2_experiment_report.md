# TD3BC `phase0b_v2` 实验报告

## 1. 报告概述

本文档整理了 `results/offline/td3bc/phase0b_v2/` 下已经取回的 TD3BC 离线训练实验结果，并从研究问题、实验设计、结果、分析、局限性与后续工作几个方面，对这组实验进行系统说明。

这组实验的核心目标不是单纯比较几个数值，而是回答一个更基础的问题：

> 在 AUV 流场离线导航任务中，**数据集规模扩大后，TD3BC 的性能到底是会下降，还是此前的“下降现象”只是实验协议带来的假象？**

`phase0b_v2` 的主要价值在于：它不是简单地再跑一遍旧实验，而是针对旧协议中的关键混杂因素进行了修正。因此，这组结果更适合回答“数据规模效应”本身，而不仅仅是“某一版脚本跑出来的最终分数”。

---

## 2. 背景与实验动机

### 2.1 研究背景

本项目研究的是 AUV 在流场中的高效航行策略。当前重点关注的是基于离线数据的 TD3BC 算法在固定流场任务中的表现。

本实验使用的离线数据来自 `crosscomp` 行为策略，对应任务设定包含：

- 目标任务：`single_u10_cross_tgt15`
- 几何：`cross_stream`
- 目标速度：`1.5 m/s`
- 观测布局：`s0`
- 历史长度：`4`
- 奖励目标：`efficiency_v2`

离线数据集名称分别为：

- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone`
- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000`
- `crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000`

### 2.2 为什么需要 `phase0b_v2`

在旧版 `phase0` size ablation 中，曾经观察到一个明显的反常现象：

- 数据集从 `500` 增大到 `1000`、`2000` 后，TD3BC 最优性能反而下降。

这与直觉以及数据本身质量是冲突的。旧版 `phase0` 汇总结果如下：

| 数据集规模 | 数据集成功率 | 旧协议最优 TD3BC 成功率 | 旧协议 BC 成功率 |
| --- | ---: | ---: | ---: |
| 500 episodes | 0.8640 | 0.532 | 0.426 |
| 1000 episodes | 0.8700 | 0.498 | 0.366 |
| 2000 episodes | 0.8855 | 0.404 | 0.328 |

这些结果说明：

- 离线数据本身并没有随着规模增大而变差，反而略有变好。
- 但训练结果却呈现出“数据越大越差”的趋势。

这使得旧实验的结论很不可信。进一步分析发现，旧协议至少存在以下三个问题：

1. `total_steps=100000` 固定不变，不同数据规模下的训练强度并不公平。
2. 训练采用 replay-style 的有放回随机采样，大数据集在固定步数下会被明显“训稀”。
3. 没有显式的 validation/test 分离，无法判断“模型更差”还是“模型更慢、但没被正确选出来”。

因此，`phase0b_v2` 的实验目标可以概括为：

1. 修正 size ablation 的训练预算定义。
2. 引入 validation-based checkpoint/alpha selection。
3. 重新判断数据规模、BC、TD3BC、以及 `alpha` 的真实关系。

---

## 3. 核心研究问题

`phase0b_v2` 主要回答以下问题：

1. 在更公平的离线训练协议下，数据集规模增大是否仍会导致 TD3BC 退化？
2. 最优 `alpha` 是否会随着数据规模改变？
3. 大数据 regime 下，TD3BC 相对纯 BC 的增益是否会缩小？
4. 当前训练长度是否足够，还是模型仍然主要受训练预算限制？

其中，`alpha` 用于控制 TD3BC 中 BC 项与 Q 引导项的相对权衡。直观上：

- 较大的 `alpha` 更偏向利用 critic/Q 信息修正行为策略；
- 较小的 `alpha` 更接近纯行为克隆（BC）。

---

## 4. `phase0b_v2` 协议设计

### 4.1 设计原则

`phase0b_v2` 相比旧版 `phase0` 的核心改动有四点：

1. 使用固定 benchmark manifest，将 validation 与 test 分离。
2. 使用 epoch-aligned 训练预算，而不是所有数据规模统一固定 `100000` updates。
3. 默认使用 `shuffle_no_replacement`，即每个 epoch 对离线数据进行一次打乱后完整遍历。
4. 训练结束后先在 validation 上进行 checkpoint 选择与 alpha 选择，再仅对选中的模型运行 test。

这使得“数据规模变大”不再等同于“每个样本被看到次数变少”，从而更接近我们真正关心的 scientific question。

### 4.2 脚本默认协议

`scripts/run_offline_td3bc_phase0b_v2.sh` 的默认设计是：

- `SIZE_ABLATION_EPISODES="500 1000 2000"`
- `SIZE_ABLATION_ALPHAS="0.0 0.05 0.1 0.25 0.5 1.0"`
- `SIZE_ABLATION_SEEDS="42 43 44 45 46"`
- `TRAIN_EPOCHS=96`
- `SAMPLING_MODE=shuffle_no_replacement`
- `VAL_MANIFEST_EPISODES=20`
- `TEST_MANIFEST_EPISODES=100`

### 4.3 本次实际执行协议

需要特别强调的是：**当前取回的结果并不是上述默认协议的完整执行结果，而是一个资源受限下的 pilot 子集。**

根据结果文件本身反推，本次实际运行配置为：

| 项目 | 脚本默认值 | 本次实际运行值 |
| --- | --- | --- |
| 数据集规模 | 500 / 1000 / 2000 | 500 / 1000 / 2000 |
| `alpha` 网格 | 0.0 / 0.05 / 0.1 / 0.25 / 0.5 / 1.0 | 0.0 / 0.1 / 0.25 / 0.5 |
| seeds | 42 / 43 / 44 / 45 / 46 | 42 / 43 |
| validation 集合大小 | 20 | 12 |
| test 集合大小 | 100 | 40 |
| 训练模式 | no-replacement epoch | no-replacement epoch |
| checkpoint 频率 | 默认脚本设计为更密集保存 | 实际结果显示约每 12 个 epoch 保存一次，共 8 个中间 checkpoint + final |

也就是说，当前 `phase0b_v2` 结果更适合被视为：

- **协议正确性验证**
- **趋势验证**
- **低成本 pilot**

而不是正式的最终论文主表。

### 4.4 训练预算

尽管实际运行的 checkpoint 间隔与脚本默认值存在偏差，但训练预算本身仍然体现了 `epoch-aligned` 的思想。实际总训练步数如下：

| 数据集规模 | 总训练步数 | 等价 steps/epoch | 说明 |
| --- | ---: | ---: | --- |
| 500 episodes | 9504 | 99 | 96 个 epoch |
| 1000 episodes | 19104 | 199 | 96 个 epoch |
| 2000 episodes | 38144 | 397 | 96 个 epoch |

这与旧协议的本质区别在于：

- 旧协议：三组数据都固定 `100000` updates。
- 新协议：三组数据都尽量保持“每个样本被完整训练多轮”的公平性。

---

## 5. 数据集与离线数据统计

离线数据来自确定性的 `crosscomp` 行为策略，`action_noise_std=0.0`，因此标签噪声较低。对应数据集统计如下：

| 数据集规模 | transitions 数 | 数据集成功率 | goal rate | out-of-bounds rate |
| --- | ---: | ---: | ---: | ---: |
| 500 episodes | 75,975 | 0.8640 | 0.8640 | 0.1360 |
| 1000 episodes | 152,683 | 0.8700 | 0.8700 | 0.1300 |
| 2000 episodes | 304,967 | 0.8855 | 0.8855 | 0.1145 |

这组统计进一步支持一个事实：

> 从数据分布本身看，数据越大并没有更“脏”，反而更稳定、更完整。

---

## 6. 旧 `phase0` 与 `phase0b_v2` 的关系

### 6.1 旧协议为什么会误导

旧 `phase0` 中，不同数据规模的 effective update intensity 差异很大。以 `batch_size=256, total_steps=100000` 近似估计，每个 transition 被抽到的平均次数约为：

| 数据集规模 | 平均每条 transition 被采样次数 |
| --- | ---: |
| 500 episodes | 337 |
| 1000 episodes | 168 |
| 2000 episodes | 84 |

这意味着：旧协议中的“更大数据集”同时也意味着“更少训练覆盖”。因此旧结论不能直接解读为“数据越大越差”。

### 6.2 v2 的意义

`phase0b_v2` 的实验价值，不在于绝对分数更高，而在于它更公平地回答了 size ablation 问题：

- 当训练预算按 epoch 对齐后，性能随数据规模如何变化？
- 当使用 validation 选择 checkpoint 后，最优点是否仍然向大数据退化？

---

## 7. `phase0b_v2` 实验结果

### 7.1 Validation 结果：`alpha` 排序

validation 结果来自 `analysis/alpha_validation.csv`。按数据集规模整理如下：

| 数据集规模 | `alpha=0.0` | `alpha=0.1` | `alpha=0.25` | `alpha=0.5` | validation 最优 |
| --- | ---: | ---: | ---: | ---: | --- |
| 500 episodes | 0.375 | 0.5417 | 0.4583 | 0.5833 | `0.5` |
| 1000 episodes | 0.6250 | 0.7083 | 0.7083 | 0.7500 | `0.5` |
| 2000 episodes | 0.8333 | 0.8333 | 0.7917 | 0.6250 | `0.1`（与 `0.0` 成功率相同，但 return 更好） |

从 validation 排序可以直接看出三点：

1. `500/1000` 数据集在当前截断网格中偏好更大的 `alpha`。
2. `2000` 数据集明显不再偏好大的 `alpha`，甚至 `0.5` 已经明显变差。
3. 大数据 regime 下，最优点正在向更“小 `alpha` / 更偏 BC”的方向移动。

### 7.2 最终 test 结果

最终 test 结果来自 `analysis/dataset_diagnostics.csv` 与 `analysis/td3bc_vs_bc_test.csv`。

| 数据集规模 | 最优 `alpha` | TD3BC test success | BC test success | `crosscomp` baseline success | TD3BC - BC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 episodes | 0.5 | 0.325 ± 0.025 | 0.275 ± 0.025 | 0.900 | +0.0500 |
| 1000 episodes | 0.5 | 0.525 ± 0.175 | 0.450 ± 0.075 | 0.900 | +0.0750 |
| 2000 episodes | 0.1 | 0.700 ± 0.050 | 0.6875 ± 0.0625 | 0.900 | +0.0125 |

对应 return 也有一致趋势：

| 数据集规模 | TD3BC test return | BC test return |
| --- | ---: | ---: |
| 500 episodes | -247.82 ± 33.68 | -299.60 ± 25.77 |
| 1000 episodes | -198.75 ± 81.41 | -226.93 ± 18.58 |
| 2000 episodes | -108.44 ± 35.28 | -109.68 ± 26.62 |

### 7.3 结果最核心的事实

最值得强调的一点是：

> 在 `phase0b_v2` 协议下，TD3BC 与 BC 都随着数据集规模增大而显著变好。

这和旧 `phase0` 中“数据越大越差”的结论是相反的。

---

## 8. 结果分析

### 8.1 `phase0b_v2` 基本推翻了旧版 size-degradation 结论

在旧 `phase0` 中，最优 TD3BC success 为：

- `500`: 0.532
- `1000`: 0.498
- `2000`: 0.404

在 `phase0b_v2` 中，尽管 test manifest 不同，绝对分数不能直接横向比较，但**趋势已经完全反转**：

- `500`: 0.325
- `1000`: 0.525
- `2000`: 0.700

这里真正重要的不是绝对值，而是趋势：

- 旧版：负趋势
- v2：正趋势

这说明先前的负趋势主要来自实验协议，而不是“更多数据伤害 TD3BC”这一算法本质。

### 8.2 数据集规模扩大后，BC 本身已经显著变强

BC test success 为：

- `500`: 0.275
- `1000`: 0.450
- `2000`: 0.6875

这说明：

- 当训练协议改正后，数据规模对纯行为克隆本身就是显著有利的。
- 因此，旧 `phase0` 中“大数据更差”的解释不能归咎于数据本身，也不能简单归咎于 TD3BC 的 Q 项。

更准确的结论是：

> 旧 `phase0` 主要在比较“固定 updates 下不同规模数据集的欠训练程度”，而不是在比较“数据规模本身的价值”。

### 8.3 TD3BC 始终优于 BC，但增益在缩小

从 test 指标看，TD3BC 相对 BC 的 success 增益为：

- `500`: `+0.0500`
- `1000`: `+0.0750`
- `2000`: `+0.0125`

这意味着：

1. TD3BC 的 Q 引导项并非无用。
2. 但随着数据集扩大，BC 本身已经越来越强，TD3BC 的边际收益在缩小。
3. 对于 `2000` 集这一大数据 regime，TD3BC 与 BC 的差异已经非常小，说明大量高质量离线数据会把问题推向“BC 足够强，只需很小的 Q 修正”这一工作区间。

### 8.4 `alpha` 的最优值随数据规模变化

本次实验中：

- `500` 最优 `alpha = 0.5`
- `1000` 最优 `alpha = 0.5`
- `2000` 最优 `alpha = 0.1`

这说明最优 BC/Q 权衡并不是固定的，而是依赖于数据规模。

但这里还要做两个重要限定：

1. `500` 和 `1000` 的最优点都落在当前网格上边界 `0.5`，因此还不能说 `0.5` 就是它们的真实最优值。
2. `2000` 的最优点落在内部 `0.1`，比 `500/1000` 更可信。

换言之，本次实验已经给出一个非常清晰的信号：

- 小中数据集可能仍然需要相对更强的 Q 修正；
- 大数据集则更偏向低 `alpha`、更接近 BC 的 regime。

### 8.5 最优 checkpoint 不总在训练末端，因此“再加 epoch”不是当前首要矛盾

被最终选中的 checkpoint 相对训练终点的位置分别为：

- `500`: `0.875, 1.0`
- `1000`: `1.0, 0.5`
- `2000`: `0.875, 0.625`

这说明：

- 有些 run 的最佳点出现在训练末端；
- 但也有不少 run 的最佳点出现在中期。

因此，目前没有证据表明“统一增加训练 epoch”就是最优的下一步。相比之下，更值得优先优化的是：

- 扩展合理的 `alpha` 网格；
- 增加 seeds；
- 增大 validation/test manifest。

### 8.6 成功率提升同时体现在失败模式改善上

将两条已完成 test seed 的 termination counts 聚合后，TD3BC 的失败模式随数据规模变化如下：

| 数据集规模 | goal | timeout | out_of_bounds |
| --- | ---: | ---: | ---: |
| 500 episodes | 26 | 19 | 35 |
| 1000 episodes | 42 | 12 | 26 |
| 2000 episodes | 56 | 5 | 19 |

这说明大数据集不仅提升了最终成功率，也降低了：

- 超时失败
- 越界失败

即，性能改善并不是单纯的统计波动，而是对应到了更稳定的闭环控制行为。

### 8.7 仍然存在明显 headroom

尽管 `2000` 集时 TD3BC 已达到 `0.700` success，但参考行为基线 `crosscomp` 在当前 test manifest 上仍有 `0.900` success。

因此：

- `phase0b_v2` 证明了“更多数据有用”
- 但也说明“只靠增大数据集规模并不足以逼近行为策略上限”

换句话说，协议问题已经被部分排除，但算法本身仍然有改进空间。

---

## 9. 本次结果的局限性

为了避免过度解读，本报告特别强调以下限制：

### 9.1 这是一组 pilot 结果，不是 formal final run

当前结果只包含：

- `2` 个 seed（42, 43）
- `4` 个 alpha（0.0, 0.1, 0.25, 0.5）
- `12` 个 validation episodes
- `40` 个 test episodes

因此，统计稳定性仍然有限。

### 9.2 `500/1000` 的最优 `alpha` 仍然撞边界

由于当前网格只扫到 `0.5`，因此：

- `500` 与 `1000` 的最优点仍可能位于 `0.5` 以上；
- 不能把 `0.5` 视为最终结论。

### 9.3 `1000` 数据集的方差较大

`1000` 集最优 TD3BC 的 test success 为 `0.525 ± 0.175`，方差明显高于 `500` 和 `2000`。这提示：

- 当前 winner 可能仍受到小 validation / 小 test 的噪声影响；
- 需要更多 seed 才能稳定确认其相对排序。

### 9.4 本地只取回了结果目录，未完整取回 offline data 与 benchmark manifests

因此本报告中有少量“运行配置”是基于结果文件反推得到的，例如：

- 实际 validation/test episode 数
- 实际 checkpoint 间隔

这些信息虽然可以较高置信度恢复，但仍应与原始 notebook 配置一起交叉确认。

---

## 10. 结论

综合来看，`phase0b_v2` 已经给出以下高置信结论：

1. 旧 `phase0` 中“数据集越大，TD3BC 越差”的现象并不可靠，主要是实验协议造成的伪象。
2. 在更公平的 epoch-aligned/no-replacement 协议下，TD3BC 与 BC 都随着数据规模增大而显著提升。
3. 数据集规模越大，最优工作区间越偏向小 `alpha`，即更接近“强 BC + 弱 Q 修正”。
4. TD3BC 仍然优于 BC，但在大数据 regime 下其边际收益明显缩小。
5. 当前最优点并不总在训练末端，因此统一增加训练 epoch 不是下一阶段的首要优化方向。

如果只用一句话概括本次实验的研究结论，可以写成：

> `phase0b_v2` 证明了，先前观察到的 TD3BC 随数据规模退化主要是训练协议问题；在更合理的离线训练与模型选择流程下，更大的高质量离线数据能够显著改善 AUV 流场导航性能，但同时也会把最优 TD3BC 工作点推向更接近纯行为克隆的区域。

---

## 11. 对后续实验的建议

结合本次结果，下一轮正式实验建议如下：

1. 保持 `phase0b_v2` 的整体协议思想不变。
2. 将 `stage_b`/`stage_c` 形式的分阶段协议作为正式 ablation 的主流程。
3. 对 `500/1000` 扩展更大的 `alpha` 网格，例如继续探索 `0.75, 1.0, 1.5`。
4. 对 `2000` 细化小 `alpha` 区间，例如 `0.0, 0.05, 0.1, 0.15, 0.2, 0.25`。
5. 提升 seeds 数与 validation/test manifest 规模，将当前 pilot 结论升级为 formal 论文结果。

---

## 12. 结果文件索引

本报告主要基于以下结果文件撰写：

- `results/offline/td3bc/phase0b_v2/summaries/phase0b_v2_overview.csv`
- `results/offline/td3bc/phase0b_v2/analysis/dataset_diagnostics.csv`
- `results/offline/td3bc/phase0b_v2/analysis/alpha_validation.csv`
- `results/offline/td3bc/phase0b_v2/analysis/td3bc_vs_bc_test.csv`
- `results/offline/td3bc/phase0b_v2/analysis/conclusions.txt`

用于动机与对照的旧版 `phase0` 文件包括：

- `results/offline/td3bc/phase0/summaries/*_summary_per_dataset.csv`

