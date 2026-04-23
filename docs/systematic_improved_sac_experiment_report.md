# 改进 SAC 的系统实验报告

> 文档版本：2026-04-23  
> 对应计划：[`docs/systematic_improved_sac_experiment_plan.md`](systematic_improved_sac_experiment_plan.md)

---

## 1. 目的与维护方式

本文件用于和系统实验计划同步记录：

1. 每个阶段的实际实验设置
2. 汇总结果
3. 关键分析
4. 由结果驱动的后续决策

当前已记录阶段：

- A0：`single_u10_cross_tgt15` 观测协议可行性筛选

---

## 2. 阶段 A0：`single_u10_cross_tgt15`

### 2.1 实验目的

在更容易学习的 `cross_stream` benchmark 上，快速判断 `s0 / s1 / s2` 三种 `probe_layout` 是否具备基础可学性，并为 A1 的 upstream 主筛选提供依据。

### 2.2 实际实验设置

| 项目 | 设置 |
|------|------|
| benchmark | `single_u10_cross_tgt15` |
| flow | `wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| `task_geometry` | `cross_stream` |
| `target_speed` | `1.5` |
| 算法 | 基础 `SAC` |
| 比较对象 | `s0_k4`, `s1_k4`, `s2_k4` |
| `history_length` | `4` |
| 训练步数 | `600000` |
| `random_steps` | `5000` |
| `update_after` | `5000` |
| `eval_every` | `10000` |
| `eval_episodes` | `30` |
| `checkpoint_every` | `10000` |
| `num_envs` | `6` |
| seeds | `46, 47, 50` |
| 目标函数 | `efficiency_v2` 主线，`arrival_v1` 诊断对照 |

原始汇总文件：

- [`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/summary/ablation_summary.csv`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/summary/ablation_summary.csv)
- [`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/arrival_v1/ablation_summary.csv`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/arrival_v1/ablation_summary.csv)

说明：

- 本轮 A0 是在在线实验输出目录重构之前完成的，因此该批 run 目录仍然使用旧的扁平结构。
- 结果本身有效，不需要重跑；后续实验统一切换到新的“`experiments/` 放结果、`checkpoints/` 放权重”的镜像目录结构。
- 若后续 run 需要把大体积模型权重完全剥离到单独磁盘，可通过 `--checkpoint-dir` 或 `CHECKPOINT_ROOT` 把 `.pt` 文件外置存储。

### 2.3 汇总结果：`efficiency_v2`

| 方法 | 成功率 | `eval_return` | `eval_time_s` | `path_efficiency` |
|------|------:|------:|------:|------:|
| `s1_k4` | `0.9667 +/- 0.0272` | `57.9113 +/- 4.1264` | `37.7756 +/- 0.5784` | `0.8544 +/- 0.0235` |
| `s2_k4` | `0.8556 +/- 0.2043` | `43.2510 +/- 30.3174` | `37.3056 +/- 0.7888` | `0.8132 +/- 0.1004` |
| `s0_k4` | `0.7889 +/- 0.2114` | `31.1792 +/- 27.8114` | `39.6267 +/- 1.5740` | `0.8013 +/- 0.0567` |

### 2.4 汇总结果：`arrival_v1`

| 方法 | 成功率 | `eval_return` | `eval_time_s` | `path_efficiency` |
|------|------:|------:|------:|------:|
| `s1_k4` | `1.0000 +/- 0.0000` | `63.5795 +/- 1.7575` | `37.5311 +/- 0.8508` | `0.8746 +/- 0.0125` |
| `s0_k4` | `0.9667 +/- 0.0272` | `57.8141 +/- 5.8138` | `38.3400 +/- 1.9226` | `0.8525 +/- 0.0372` |
| `s2_k4` | `0.7111 +/- 0.2615` | `18.7582 +/- 35.8464` | `39.0689 +/- 2.5960` | `0.7083 +/- 0.1333` |

### 2.5 结果分析

#### 结论 1：`600k` 是必要预算，`100k/300k` 会明显低估方法能力

A0 的真实曲线说明，部分配置在 `300k` 之前仍接近“尚未学会”的状态。  
尤其是 `s2_k4`，若在 `100k` 或 `300k` 提前截断，会被误判为明显劣于 `s1_k4`。

这直接修正了整个系统实验计划：  
后续在线实验默认预算统一提高到 `600k`，而不是再使用 `100k` 级别的短跑协议。

#### 结论 2：`s1_k4` 是当前最稳的观测协议候选

`s1_k4` 在两个目标函数下都表现稳定：

- 在 `efficiency_v2` 下排名第一，且 seed 方差最小。
- 在 `arrival_v1` 下达到 `1.0 +/- 0.0` 的成功率。

因此，`s1_k4` 当前是最稳的主候选，而不是单纯“某个目标函数下偶然最优”。

#### 结论 3：`s2_k4` 不是失败方法，而是高方差候选

如果只看最终均值，`s2_k4` 落后于 `s1_k4`。  
但更准确的解释是：

- `s2_k4` 在 `efficiency_v2` 下有明显潜力，部分 seed 可以达到很强结果。
- 其主要问题是 seed 方差大、训练不稳定，而不是“没有信息价值”。

这意味着 A1 不应把 `s2` 直接裁掉，而应该继续观察它在 upstream 场景下是否更能体现优势。

#### 结论 4：`s0_k4` 在 A0 上不能被提前淘汰

虽然 `s0_k4` 不是 A0 第一名，但它在 `arrival_v1` 下接近 `s1_k4`，在 `efficiency_v2` 下也不是完全掉队。  
因此，A1 主实验仍然保留 `s0 / s1 / s2` 全部三组，避免过早把“较简协议”排除出 upstream 主筛选。

#### 结论 5：`arrival_v1` 适合作为诊断，不适合作为当前主线

`arrival_v1` 的主要价值是帮助判断：

- 如果 `efficiency_v2` 下学不动，到底是观测协议无效，还是目标函数把学习压得太难。

但从研究主线看，当前更关心的是高效航行，因此后续阶段仍应以 `efficiency_v2` 为主线目标函数。

### 2.6 由 A0 导出的决策

1. A1 主实验采用 `single_u10_upstream_tgt15 + efficiency_v2 + s0/s1/s2 + 600k`。
2. A1 先不并行跑 `arrival_v1`，仅在主实验结果不足以区分方法时补做。
3. 后续计划中的 `100k` 级别预算全部作废，统一按 `600k` 重写。
4. 在线实验目录结构改为 `experiments/` 与 `checkpoints/` 镜像分离，并在阶段根目录单独维护 `summary/`。

---

## 3. 下一步实验

当前已冻结的下一步是：

### A1 主实验

| 项目 | 设置 |
|------|------|
| benchmark | `single_u10_upstream_tgt15` |
| 目标函数 | `efficiency_v2` |
| 方法 | `s0_k4`, `s1_k4`, `s2_k4` |
| 训练步数 | `600k` |
| seeds | `46, 47, 50` |
| `num_envs` | `6` |

建议目录前缀：

- 结果：`experiments/protocol_screen_v2/A1_single_u10_upstream_tgt15/efficiency_v2/sac_vanilla/`
- 权重：`checkpoints/protocol_screen_v2/A1_single_u10_upstream_tgt15/efficiency_v2/sac_vanilla/`

阶段级汇总建议统一写到：

- `experiments/protocol_screen_v2/A1_single_u10_upstream_tgt15/efficiency_v2/sac_vanilla/summary/`

当前执行与汇总脚本：

- 运行：[`scripts/run_stage_a1_layout_main.sh`](../scripts/run_stage_a1_layout_main.sh)
- 汇总：[`scripts/summarize_stage_a1_layout_main.sh`](../scripts/summarize_stage_a1_layout_main.sh)

### A1 诊断线

暂不启动。只有在 `efficiency_v2` 主实验无法形成清晰结论时，才补测对应的 `arrival_v1`。
