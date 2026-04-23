# ReBRAC 实验计划

> 文档版本：2026-04-22 rev.1
> 适用范围：当前仓库中已完成实现的 ReBRAC 离线主线，以及它与 `phase0c / worldcomp teacher-gap` 结论之间的衔接
> 当前前提：请先阅读 [offline_rl_implementation_plan.md](./offline_rl_implementation_plan.md)、[td3bc_mainline_closure_plan.md](./td3bc_mainline_closure_plan.md)、[td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)、[td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)

---

## 0. 当前状态

当前仓库已经不再处于“讨论下一算法”的阶段，而是已经具备：

- ReBRAC agent 实现；
- `train_offline.py / evaluate_offline.py` 的 `--algo {td3bc,rebrac}` 分发；
- `next_actions` 数据链路；
- 并行评估路径对 ReBRAC 的兼容；
- 最小 screening 脚本入口。

因此，本文件的重点不是“ReBRAC 要不要做”，而是：

> 在已经完成 TD3BC 主线收口的前提下，如何用最小、可解释、可复现的实验，把 ReBRAC 变成下一个正式基线。

---

## 1. 为什么现在做 ReBRAC

### 1.1 不是因为 TD3BC 失败，而是因为问题已经被拆清楚了

截至当前，TD3BC 已经给出三个稳定结论：

1. `crosscomp` deployable 主线上，TD3BC 相比 BC 有真实增益。
2. 当前最优规模在 `1000` episodes 左右，而不是“数据越大越好”。
3. `worldcomp` 上存在真实的 deployable teacher gap，privileged critic 只能关闭大约一半。

这意味着当前问题已经从“离线 RL 能不能工作”变成了两个更具体的瓶颈：

- `crosscomp`：为什么 deterministic `2000` 没有继续转化成收益？
- `worldcomp`：为什么 critic 额外看到局部等效流后，仍然只关闭一半 teacher gap？

### 1.2 为什么 ReBRAC 是下一步，而不是 XQL 或 FQL

当前最像的解释不是“行为分布太复杂，必须立刻升级 actor family”，而是：

- critic regularization 不够；
- TD3BC 对更宽支持集的利用方式不够好；
- actor/critic 对行为分布偏离的约束还不够细。

ReBRAC 正好对应这组问题，因为它相对 TD3BC 的关键增强是：

- 更深网络；
- critic 中使用 LayerNorm；
- 保留 TD3BC 的 Q normalization；
- actor penalty 与 critic penalty 的双通道约束。

所以这一阶段做 ReBRAC，最重要的价值不是“再找一个更强算法”，而是回答：

> 当前 TD3BC 没吃下来的那部分收益，究竟是不是 critic regularization + dual penalties 就能解释。

---

## 2. 本阶段要回答的核心问题

ReBRAC 阶段建议只回答下面三个问题。

### 2.1 主问题

在与 `phase0c` 相同的 deployable canonical protocol 下，ReBRAC 是否能稳定优于 TD3BC？

### 2.2 机制问题

ReBRAC 是否能改善 `crosscomp-2000` 相对 `crosscomp-1000` 的退化，至少让 `2000` 不再明显差于 `1000`？

### 2.3 辅助问题

如果 ReBRAC 在 `worldcomp-1000` 上也有改善，这种改善更多来自：

- deployable 轨道的整体提升；
- 还是 privileged critic 轨道对 teacher gap 的进一步压缩。

---

## 3. 本阶段不做什么

为了避免范围失控，这一轮明确不做以下事情：

1. 不重开大规模 TD3BC `size × alpha` sweep。
2. 不把 noisy-support / mixed-deployable 数据线前置成主任务。
3. 不直接进入 XQL。
4. 不直接进入 FQL。
5. 不把 `worldcomp` 重新变成 size study 主线。

这一阶段只做一件事：

> 用尽量干净、尽量短的实验链路，判断 ReBRAC 是否构成对 TD3BC 的实质性下一层增强。

---

## 4. Canonical 协议

为保证结果与现有主线可直接对照，本阶段默认沿用当前 canonical protocol：

- `flow = wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`
- `manifest = benchmarks/single_u10_cross_tgt15.json`
- `probe_layout = s0`
- `history_length = 4`
- `task_geometry = cross_stream`
- `target_speed = 1.5`
- `objective = efficiency_v2`

所有 collect / train / evaluate 都必须显式写出这些条件，不依赖默认值。

---

## 5. 当前实现口径

当前仓库里的 ReBRAC 实现约定如下：

- actor：deterministic policy，与 TD3BC 保持同类接口；
- actor penalty：对行为动作的平方误差正则；
- critic penalty：在 target 端约束 next action 偏离数据中的 `next_actions`；
- Q normalization：默认开启；
- 网络：默认 `hidden_dim=256`、`num_hidden_layers=3`；
- LayerNorm：默认 critic 开启，actor 关闭；
- privileged critic：沿用现有 offline 协议，可作为 `worldcomp` 诊断轨道使用。

换句话说，这一实现要表达的是：

> 当前 ReBRAC 不是“TD3BC 多加几个开关”，而是“以 dual penalties 为核心的下一层 minimalist baseline”。

---

## 6. 实验分阶段计划

## 6.1 Stage A：实现与链路 sanity `【已完成】`

目标：

- 代码层面跑通 ReBRAC；
- 确保 checkpoint / evaluation / parallel policy worker 均闭环；
- 确保 `collect_offline_data -> train_offline -> evaluate_offline` 对 ReBRAC 可用。

当前状态：

- 已完成最小 smoke 验证；
- `mytorch1` 环境下定向测试通过；
- ReBRAC 与 TD3BC 的超小规模端到端 CLI 链路都已跑通。

这一阶段的意义不是产出可报告结果，而是先确认后续 screening 不会因为基础设施问题污染解释。

## 6.2 Stage B：最小 screening `【当前立即执行】`

### 目标

先在最有信息量的 deployable 数据上，快速判断 ReBRAC 是否有正信号。

### 为什么先做这个

因为当前最关键的问题不是“ReBRAC 在所有条件下最优超参是什么”，而是：

> 它能不能在 `crosscomp-1000/2000` 这个最核心的对照对上，给出比 TD3BC 更合理的趋势。

### 数据集

- `crosscomp-1000`
- `crosscomp-2000`

### 协议

- protocol：`deployable`
- seeds：`42, 43`
- sampling：`shuffle_no_replacement`
- train epochs：`64`
- checkpoint 选择：沿用当前离线主线标准
- 评估：固定 `single_u10_cross_tgt15` manifest

### 第一轮推荐搜索范围

为了控制预算，第一轮只做最小 2x2 screening：

- `actor_penalty_coef ∈ {1.0, 2.0}`
- `critic_penalty_coef ∈ {1.0, 2.0}`

其余默认：

- `hidden_dim = 256`
- `num_hidden_layers = 3`
- `critic_layernorm = on`
- `actor_layernorm = off`
- `normalize_q = on`

### 判据

如果出现以下任一情况，就认为 screening 有正信号：

1. `crosscomp-1000` 上 ReBRAC 稳定不弱于 TD3BC，并在 success/return 中至少一项更好。
2. `crosscomp-2000` 上 ReBRAC 明显优于当前 TD3BC 主线。
3. `2000` 至少不再比 `1000` 明显更差。

如果 2x2 screening 全部弱于 TD3BC，不建议立刻扩大搜索网格；应先看训练日志与 checkpoint 行为，再决定是：

- 小幅扩参数；
- 还是直接转向 XQL。

## 6.3 Stage C：正式确认 `【Stage B 成功后】`

### 目标

把 screening 中最好的 ReBRAC 配置升级到正式 5-seed 结果。

### 推荐矩阵

- `crosscomp-1000`
- `crosscomp-2000`

### 预算

- seeds：`42 / 43 / 44 / 45 / 46`
- train epochs：`96`
- val manifest episodes：`40`
- test manifest episodes：`100`

### 比较对象

必须与以下结果同表报告：

- TD3BC formal result
- BC formal result

### 这一阶段的作用

这一步不是为了“再刷一次最好成绩”，而是为了把 ReBRAC 的结论从“screening 观察”提升成“正式基线”。

## 6.4 Stage D：`worldcomp` teacher-gap follow-up `【只在 Stage C 有正信号时做】`

### 目标

如果 ReBRAC 在 deployable `crosscomp` 上已经证明有价值，再看它能否进一步解释 `worldcomp` 的剩余 gap。

### 数据集

- `worldcomp-1000`

### 轨道

1. deployable
2. privileged-critic

### 为什么这一步后置

因为 `worldcomp` 的价值是“做 gap 诊断”，不是“替代主线”。
只有当 ReBRAC 已经在 `crosscomp` 上证明自己是更强基线，这一步才值得做。

### 关注点

这里不只看绝对 success，还要看：

- deployable 是否提升；
- privileged critic 是否比 TD3BC 更能压缩剩余 gap；
- `out_of_bounds / timeout` 是否发生结构性变化。

## 6.5 Stage E：必要时的最小 ablation

如果 ReBRAC 有正信号，但解释还不够清楚，建议只做最小机制消融，而不是大扫网格。

优先级建议：

1. `full ReBRAC`
2. `critic penalty off`
3. `normalize_q off`

这三组足够回答两个关键问题：

- dual penalty 里的 critic 侧约束是否真的重要；
- Q normalization 是否是当前收益的重要组成部分。

不建议一开始就把 actor/critic LayerNorm、网络深度、dropout 等全部摊开做全因子实验。

---

## 7. 评价指标与选择规则

主结论仍沿用当前离线主线的标准：

- P0：`eval_success_rate`
- P0：`eval_return`
- P1：`eval_safety_cost`
- P1：`eval_time_s`
- P1：`eval_path_efficiency`

训练侧保留：

- `critic_loss`
- `actor_loss`
- `bc_loss`
- `mean_q`
- `lambda`
- `critic_penalty`

checkpoint 与超参选择规则不改，继续使用：

> `success_rate -> return -> -safety_cost -> -time`

这样 ReBRAC 与 TD3BC 的差异才可以解释为算法本身，而不是选择规则不同。

---

## 8. 结果判读逻辑

### 情况 A：`1000` 提升明显，`2000` 也被拉回

这是最强阳性结果。
它说明当前瓶颈高度符合“critic regularization + dual penalty 不足”的解释。

下一步：

- 推进 formal 5-seed；
- 再决定是否进入 `worldcomp` follow-up。

### 情况 B：`1000` 提升，但 `2000` 仍弱

说明 ReBRAC 比 TD3BC 更强，但还不足以解释 deterministic 大数据支持集问题。

下一步：

- formalize `1000`
- 谨慎决定是否需要 XQL 来进一步测试 in-sample backup 假设。

### 情况 C：`1000/2000` 都不明显提升

说明当前问题未必主要来自 TD3BC 的 minimalism 不足。
这会提高 XQL 的优先级。

下一步：

- 不扩大 ReBRAC sweep；
- 直接整理失败证据，转向 XQL。

### 情况 D：`worldcomp` privileged-critic 改善明显

说明 ReBRAC 不仅对 deployable 数据有效，也更能利用 critic 侧额外信息。
这会加强“teacher-gap 中有一部分来自 critic optimization regime”的解释。

---

## 9. 产物要求

本阶段结束后，至少应形成以下输出：

1. 一份 screening summary
2. 一份 formal summary（如果进入 Stage C）
3. 一张对比表，至少包含：
   - TD3BC vs BC vs ReBRAC on `crosscomp-1000`
   - TD3BC vs BC vs ReBRAC on `crosscomp-2000`
4. 如果进入 Stage D，再补：
   - ReBRAC deployable vs privileged-critic vs `worldcomp` baseline

---

## 10. 推荐执行顺序

如果把这一阶段压缩成最小行动清单，推荐顺序是：

1. 跑 `crosscomp-1000/2000` 的 2x2 screening。
2. 只要出现正信号，就推进 5-seed formal。
3. 只有 formal 结果成立，才进入 `worldcomp-1000` follow-up。
4. 如果 formal 不成立，就停止扩 ReBRAC，转向 XQL。

---

## 11. 一句话执行意见

> ReBRAC 这一阶段的任务不是“广泛扫参”，而是用最小、可解释的 canonical screening 去判断：当前 TD3BC 尚未解释的那部分问题，是否主要来自 critic regularization 与 dual penalties 的不足。
