# 改进 SAC 的系统实验方案

> 文档版本：2026-04-22  
> 对应报告：[`docs/systematic_improved_sac_experiment_report.md`](systematic_improved_sac_experiment_report.md)  
> 适用范围：当前仓库中的 `SAC / LayerNorm / Dropout / Asymmetric Critic / RLPD`

---

## 1. 当前结论与总原则

本计划已经根据 A0 阶段的真实结果更新，不再沿用之前的 `100k` 级别预算设定。

目前最重要的结论有四条：

1. `100k` 甚至 `300k` 对当前在线导航实验都明显过短，至少在 `single_u10_cross_tgt15` 上，学习通常要到 `300k+` 才开始脱离“几乎没学会”的区间。后续在线实验默认预算改为 **`600k env steps`**。
2. 阶段 A 的主任务不是“证明某个 layout 绝对最强”，而是为后续算法消融冻结一套**稳定、可学、可归因**的观测协议。
3. A0 的结果表明 `s1_k4` 是当前最稳的候选，`s2_k4` 有潜力但方差更大，`s0_k4` 在较简单的 `cross_stream` 上可行，因此 A1 不应提前淘汰 `s0`。
4. 后续主线目标函数继续固定为 `efficiency_v2`，`arrival_v1` 只作为诊断实验按需补测。

---

## 2. 统一实验协议

### 2.1 在线实验默认设置

除非某一阶段明确说明，否则在线实验统一使用：

| 项目 | 设置 |
|------|------|
| 主目标函数 | `efficiency_v2` |
| 训练步数 | `600000` |
| `random_steps` | `5000` |
| `update_after` | `5000` |
| `update_every` | `1` |
| `updates_per_step` | `1` |
| `batch_size` | `256` |
| `hidden_dim` | `256` |
| `eval_every` | `10000` |
| `eval_episodes` | `30` |
| `checkpoint_every` | `10000` |
| 默认 `num_envs` | `6` |
| 当前 pilot seeds | `46, 47, 50` |

### 2.2 关于 `num_envs`

`num_envs` 目前不再被视作可随意切换的运行细节，而是实验协议的一部分。

原因有两个：

1. 当前训练实现中，固定 `updates_per_step=1` 时，更大的 `num_envs` 会降低单位环境步的梯度更新密度。
2. 之前还暴露过两类与并行环境相关的实现问题：
   - 向量环境自动 reset 时，episode 设定可能漂回默认配置。
   - `eval_every` 采用模运算时，会受 `num_envs` 是否整除影响。

相关 bug 已修复，但**同一阶段内仍必须固定 `num_envs`**。  
如果后续换机器导致 `num_envs` 必须变化，则应把它视为新的运行协议，不能直接和旧结果混合比较。

### 2.3 输出目录规范

从本计划开始，在线训练输出按以下结构组织：

| 子目录 | 内容 |
|------|------|
| `checkpoints/` | `agent_step_*.pt`, `agent_best.pt`, `agent_latest.pt`, `agent_final.pt` |
| `results/` | `eval_log.csv`, `final_eval.json`, `train_config.txt` |
| `logs/` | `train_log.jsonl` |
| `state/` | `replay_latest.pkl`, `rng_state.pkl` |
| 根目录 | `trainer_state.json` |

这样做的目的不是改变算法行为，而是把“分析结果”和“恢复训练所需状态”明确分离，便于后续实验归档和对比。

如果需要进一步节省主实验目录的存储压力，可以在训练时使用外置 checkpoint 目录：

- 单个 run：`--checkpoint-dir <PATH>`
- A0/A1 脚本：`CHECKPOINT_ROOT=<PATH>`
- `run_suite.py`：`--checkpoint-root <PATH>`

此时 `.pt` 文件会写到外部目录，`trainer_state.json` 中会记录对应路径；`results/logs/state` 仍保留在 run 目录下，方便单独复制分析文件。

当前建议的目录层级为：

- 结果：`experiments/<study>/<benchmark>/<objective>/<algo_tag>/...`
- 权重：`checkpoints/<study>/<benchmark>/<objective>/<algo_tag>/...`

例如当前 A1 的 vanilla SAC 主实验建议使用：

- `experiments/protocol_screen_v2/A1_single_u10_upstream_tgt15/efficiency_v2/sac_vanilla/...`
- `checkpoints/protocol_screen_v2/A1_single_u10_upstream_tgt15/efficiency_v2/sac_vanilla/...`

其中阶段根目录下再单独保留一个 `summary/`，用于放阶段级聚合结果：

- `summary/ablation_runs.csv`
- `summary/ablation_summary.csv`
- `summary/ablation_summary.json`
- `summary/ablation_summary.md`
- `summary/ablation_overview.pdf`
- `summary/ablation_overview.png`

这样单个 `seed_xx/` 目录只包含该 run 自己的结果，而不会混入整阶段汇总文件。

### 2.4 指标解释顺序

主指标优先级固定为：

1. 最后一次 `final_eval.json` 的成功率
2. `results/eval_log.csv` 中后期评估曲线是否稳定上升
3. `best periodic eval success`
4. success-conditioned `time / energy / path_efficiency`
5. 终止原因分布和 seed 方差

说明：

- 阶段性筛选首先关心“是否学会”和“是否稳定”。
- 效率指标必须在成功率可接受的前提下才有解释价值。

---

## 3. A0 阶段复盘

### 3.1 实验设置

| 项目 | 设置 |
|------|------|
| benchmark | `single_u10_cross_tgt15` |
| 比较对象 | `s0_k4`, `s1_k4`, `s2_k4` |
| 目标函数 | `efficiency_v2` 主线，`arrival_v1` 诊断对照 |
| 训练步数 | `600k` |
| seeds | `46, 47, 50` |
| `num_envs` | `6` |

### 3.2 A0 对计划的直接影响

1. 后续在线实验不再使用 `100k` 或 `300k` 作为默认预算。
2. A1 主实验保留 `s0 / s1 / s2` 全部三组，不提前裁掉 `s0`。
3. `arrival_v1` 不纳入 A1 主实验，只保留为必要时补测的诊断线。
4. 阶段 A 的下一步是先完成 A1 主筛选，再决定是否进入 A2 的 `history_length` 比较。

更详细的 A0 数据与分析见：

- [`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/ablation_summary.csv`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/ablation_summary.csv)
- [`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/arrival_v1/ablation_summary.csv`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/arrival_v1/ablation_summary.csv)
- [`docs/systematic_improved_sac_experiment_report.md`](systematic_improved_sac_experiment_report.md)

---

## 4. 阶段 A：观测协议冻结

阶段 A 的最终目标仍然不变：

> 选出后续所有算法实验统一使用的 `(probe_layout, history_length)`。

### 4.1 A1：layout 主筛选

A1 是当前优先推进的主实验。

#### A1 设计

| 项目 | 设置 |
|------|------|
| benchmark | `single_u10_upstream_tgt15` |
| 主实验目标函数 | `efficiency_v2` |
| 比较对象 | `s0_k4`, `s1_k4`, `s2_k4` |
| 算法 | 基础 `SAC` |
| 训练步数 | `600k` |
| seeds | `46, 47, 50` |
| `num_envs` | `6` |

#### A1 设计理由

1. `single_u10_upstream_tgt15` 比 A0 更接近后续主任务，但还没有 `u15_upstream` 那么容易陷入“全体都没学会”的低分辨率区间。
2. A0 已经证明 `s0` 在较简单场景并非明显失败，因此 A1 仍保留三组共同进入主实验，避免过早做结构性淘汰。
3. 继续固定 `history_length=4`，让 A1 只回答 `probe_layout` 问题，不把时间上下文混进来。

#### A1 主判据

1. `final_eval.json` 的 `eval_success_rate`
2. 后期 `eval_log.csv` 是否稳定
3. seed 方差
4. success-conditioned `path_efficiency`

#### A1 之后的决策规则

- 若某一组显著落后，可直接淘汰。
- 若 `s1` 和 `s2` 接近，则优先保留更稳定、更简单的方案。
- 若 `s0` 在 upstream 上也接近 `s1/s2`，则再进入 A2 时保留 `s0` 作为候选。

### 4.2 A1 诊断实验

`arrival_v1` 在 A1 中先**只写入计划，不立即并行启动**。

启用条件：

1. `efficiency_v2` 下三组成功率都很低，难以判断是“观测协议差”还是“目标函数太难”。
2. `efficiency_v2` 下不同组的曲线都高度不稳定，无法形成有效排序。

诊断实验的设计与 A1 主实验保持一致，只把目标函数改为 `arrival_v1`。

### 4.3 A2：history length 筛选

A2 只有在 A1 完成后才启动。

#### A2 设计

| 项目 | 设置 |
|------|------|
| benchmark | `single_u10_upstream_tgt15` |
| 目标函数 | `efficiency_v2` |
| 候选 layout | A1 前 2 名 |
| 比较对象 | 每个 layout 比较 `k=1, 4, 16` |
| 训练步数 | `600k` |
| seeds | `46, 47, 50` |
| `num_envs` | `6` |

#### A2 选择原则

- 只有当更长历史带来**稳定且明确**的收益时，才升级到更大的 `k`。
- 若性能接近，优先保留 `k=4`，避免人为增加输入维度和训练不稳定性。

### 4.4 A3：hard-check

#### A3 设计

| 项目 | 设置 |
|------|------|
| benchmark | `single_u15_upstream_tgt15` |
| 目标函数 | `efficiency_v2` |
| 比较对象 | A2 前 2 名配置 |
| 训练步数 | `600k` |
| seeds | `46, 47, 50` 或新的冻结 3-seed 集合 |

#### A3 作用

A3 不是重新做冠军排序，而是排查风险：

- 某配置若只在 `u10_upstream` 有效、到 `u15_upstream` 直接崩溃，则不能作为后续固定协议。

---

## 5. 阶段 B：在线算法组件消融

只有在阶段 A 完成、观测协议被冻结后，阶段 B 才启动。

### 5.1 目标

验证当前改进项各自是否真的提供了稳定增益，而不是混合在一起“看起来更强”。

### 5.2 方法矩阵

在冻结的 `(probe_layout, history_length)` 上，比较：

1. `SAC`
2. `SAC + LayerNorm`
3. `SAC + LayerNorm + UTD4`
4. `SAC + LayerNorm + UTD4 + Dropout`
5. `SAC + LayerNorm + Asymmetric Critic`
6. `Full` 组合

说明：

- `Dropout` 不能直接和 base 比，必须先和同样 `UTD` 的非 Dropout 版本比。
- `Asymmetric Critic` 的论文表述必须清楚说明：critic 使用的是局部特权流信息，不是全状态特权观测。

### 5.3 推荐设置

| 项目 | 设置 |
|------|------|
| 主 benchmark | `single_u15_upstream_tgt15` |
| 目标函数 | `efficiency_v2` |
| 训练步数 | `600k` 起步，必要时增加到 `800k-1M` |
| 初筛 seeds | `46, 47, 50` |
| 确认性 seeds | `8-10` 个，需在启动前冻结 |

---

## 6. 阶段 C：RLPD 数据源消融

RLPD 单独成组，不和在线结构改动混合归因。

### 6.1 目标

回答的问题是：

> 哪类离线数据在当前任务里最能帮助在线学习？

### 6.2 比较方式

固定 backbone 后，比较：

1. `online only`
2. `RLPD + goalseek`
3. `RLPD + crosscomp`
4. `RLPD + worldcomp`
5. `RLPD + privileged`

### 6.3 协议要求

1. 离线数据量必须控制一致。
2. 离线数据与在线训练使用相同观测协议和目标函数。
3. 同时报告在线环境步和总 transition 使用量。

---

## 7. 阶段 D：确认性实验与拓扑泛化

最终确认性实验只针对少数最强候选开展。

### 7.1 建议 benchmark

| 类别 | benchmark |
|------|------|
| 主结论 | `single_u15_upstream_tgt15` |
| 泛化验证 | `tandem_u15_upstream_tgt15` |
| 泛化验证 | `sbs_u15_upstream_tgt15` |

### 7.2 要求

1. 使用冻结后的主协议和主方法。
2. 使用 8 到 10 个 seed。
3. 报告成功率、方差、success-conditioned 效率指标。
4. 不再继续扩展方法矩阵，重点转向确认性证据。

---

## 8. 当前立即执行项

按优先级，当前只推进以下事项：

1. 完成 A1 主实验：`single_u10_upstream_tgt15 + efficiency_v2 + s0/s1/s2 + 600k`
2. 根据 A1 结果决定是否补做 `arrival_v1` 诊断
3. 在 A1 完成后再决定 A2 的 `history_length` 候选范围

在 A1 结果出来之前，不建议提前铺开阶段 B 或阶段 C。
