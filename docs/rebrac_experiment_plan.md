# ReBRAC 实验计划

> 文档版本：2026-04-24 rev.2
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
- actor penalty：对行为动作的平方误差正则（`(pi - a).pow(2).sum(dim=-1).mean()`，**未除以 action_dim**）；
- critic penalty：在 target 端约束 next action 偏离数据中的 `next_actions`；
- Q normalization：默认开启；
- 网络：默认 `hidden_dim=256`、`num_hidden_layers=3`；
- LayerNorm：默认 critic 开启，actor 关闭；
- privileged critic：沿用现有 offline 协议，可作为 `worldcomp` 诊断轨道使用。

> **实现口径注记（Q-normalized ReBRAC 变体）**：当前 `auv_nav/rebrac.py` 的 actor loss 会将 deterministic policy gradient 除以 `|Q|.detach()`（与 TD3+BC 的 `lambda = alpha / |Q|` 同一尺度约定），而非 ReBRAC 原始论文所用的未归一化形式。因此本实验计划内的所有 `actor_penalty_coef` 数值都应在“Q-normalized ReBRAC 变体”口径下解释。与 TD3BC 的等价关系（在 `action_dim=2` 下）约为：`actor_penalty_coef ≈ 1 / (2 · alpha_TD3BC)`。

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

## 6.2 Stage B0：训练预算 probe `【已完成】`

### 目标

在正式 Stage B screening 开跑之前，先用尽量小的代价确认 `TRAIN_EPOCHS=64` 是不是合适的训练预算——这个值从 TD3BC `phase0c` 直接继承而来，没有在 ReBRAC 口径下单独验证过。

### 为什么单独做这一步

TD3BC 主线之所以最后选用 `TRAIN_EPOCHS=64`，是因为它在 Stage B screening 下已经足够拉平 winner 的排序；但这不意味着切换到 ReBRAC 后仍然成立：

- ReBRAC 的 actor 更新里多了 `β1 · (π - a)²` 正则项，早期梯度构成与 TD3BC 不完全一致；
- critic 端的 `β2 · (π_target(s') - a')²` penalty 会改变 target 的大致尺度；
- 正式 Stage B screening 涉及 36 个 run（3 × 2 × 2 × 3），预算敏感；如果 64 epoch 不够，后续所有结论都会受到 "训练没训完" 的系统性干扰。

所以这一步的目标不是 "调最优 epoch 数"，而是回答 "64 epoch 在 ReBRAC 口径下是偏紧还是有余量？"

### 方法

**不做** `TRAIN_EPOCHS ∈ {32, 64, 96, 128}` 的笛卡尔 sweep（浪费）。利用 Stage B 协议已经对齐的 `CHECKPOINT_EVERY_EPOCHS=8` + 每个 checkpoint 都走 val 的特性，**把一个 cell 训长一次（128 epoch），读中间 checkpoint 的 val 曲线**——16 个 val 点免费拿到。

等价前提：当前 `scripts/train_offline.py` 用常数 LR AdamW，`sampling_mode=shuffle_no_replacement` 每个 epoch 独立 shuffle，因此 "长训至 step N" 在优化动力学上 ≈ "只训到 N 结束"。notebook 里有一段可选 sanity check 通过权重 L2 距离交叉确认这一假设。

### 实验范围

| 轴 | 配置 | 说明 |
| --- | --- | --- |
| β1 / β2 | `2.0 / 1.0` | 对齐 TD3BC α=0.25 的 phase0c `1000` winner；也是正式 screening 网格的中心点 |
| dataset | `crosscomp-1000` + `crosscomp-2000` | 本次核心假设：`2000` 是否需要比 `1000` 更大预算 |
| seeds | `42 / 43 / 44` | 与正式 screening 同，不浪费 |
| TRAIN_EPOCHS | `128` | 产出 16 个 ckpt，每 8 epoch 一个 |
| manifest | val = 40 / test = 40 | 与 Stage B 协议一致 |

预算约为正式 screening 的 `1/6 × 128/64 ≈ 1/3`。

### 结论

详见 [rebrac_experiment_report.md](./rebrac_experiment_report.md) §5。执行要点：

1. `TRAIN_EPOCHS=64` 对两个数据集都已足够：`crosscomp-1000/2000` 上典型 seed（42、43）在 epoch 40~50 处就已经接近 val plateau，之后的增益主要来自曲线波动，不是稳定的上升趋势。
2. 真正值得警惕的信号不是 "训练不够"，而是 **seed 方差**——seed 44 在 `crosscomp-1000` 上 peak 只有 `0.775`，远低于 42、43 的 `0.95`。
3. 因此正式 Stage B 维持 `TRAIN_EPOCHS=64`，但在结果分析阶段必须把 `std_test_success_rate` 作为一等公民看待，尤其是在 `β1=2.0` 这一列。

### 输出位置

刻意与正式 screening 隔离：

- `checkpoints/offline/rebrac/screening_epoch_probe/`
- `results/offline/rebrac/screening_epoch_probe/`

执行入口：[notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb)。

## 6.3 Stage B：最小 screening `【已完成】`

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
- seeds：`42, 43, 44`（3 seeds 是用于识别 seed 方差的最低门槛；2 seeds 下 phase0c Stage A 曾在更大 val 下多次翻转赢家）
- sampling：`shuffle_no_replacement`
- train epochs：`64`
- checkpoint 策略：**与 phase0c Stage B 对齐**（即 `scripts/run_offline_td3bc_phase0b_v2.sh` 所定义）
  - 训练阶段 `--eval-every 0 --skip-final-eval`，仅按 `CHECKPOINT_EVERY_EPOCHS=8` 周期保存 `agent_step_*.pt`；
  - 训练结束后，对每个 `agent_step_*.pt` 与 `agent_final.pt` 在**独立 val manifest**（`single_u10_cross_tgt15`，40 episodes）上评估；
  - 按 `success_rate → return → -safety_cost → -time` 选择最佳 checkpoint；
  - 将选定 checkpoint 在**独立 test manifest**（40 episodes）上重跑，作为该 (β1, β2, seed) 的最终成绩。

### 第一轮推荐搜索范围

第一轮做 3×2 screening（在 Q-normalized 变体口径下，三档 BC 强度对齐 TD3BC 已验证的三个 regime）：

- `actor_penalty_coef ∈ {1.0, 2.0, 4.0}`
  - `1.0` ≈ TD3BC α=0.5（phase0c 500 winner）
  - `2.0` ≈ TD3BC α=0.25（phase0c 1000 winner）
  - `4.0` ≈ TD3BC α=0.125（覆盖 2000 所需的更弱 BC 约束）
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

如果 3×2 screening 全部弱于 TD3BC，不建议立刻扩大搜索网格；应先看训练日志与 checkpoint 行为，再决定是：

- 小幅扩参数；
- 还是直接转向 XQL。

### 训练侧诊断（与 screening summary 一同输出）

为判断“critic penalty 是否处在合理量级”，screening summary 还需要附带报告每个 (β1, β2) 组合的训练后期 25% 窗口内以下均值：

- `mean_critic_penalty`：`E[||π_target(next_obs) + noise - a'||²]`（与 β2 无关的原始量）；
- `mean_target_q`：`E[Q_target(s', a')]`（bootstrap 前端，未减 penalty）；
- `mean_critic_penalty_ratio`：`|critic_penalty| / max(|target_q|, 1e-6)`（未乘 β2，方便跨 β2 对比原始尺度；β2 加权效应请自行乘以 `critic_penalty_coef`）。

该比值提供一个粗略健康指标：当 `β2 · mean_critic_penalty_ratio` 接近 1 时，critic 会被 penalty 支配，可能出现过度悲观；当它显著低于 1 时，critic penalty 基本是“锦上添花”，主要信号来自真实 target Q。`scripts/run_offline_rebrac_screen.sh` 的 `overview.csv` 已经把这三列作为首屏列输出。

### 结论

详见 [rebrac_experiment_report.md §6](./rebrac_experiment_report.md)。要点：

1. **screening winner = `(β1=4.0, β2=2.0)`**，两个 dataset 共享。test success 分别为 `0.883 ± 0.031 (ep1000)` 与 `0.917 ± 0.012 (ep2000)`；`β2 · mean_critic_penalty_ratio` 分别为 `0.129 / 0.311`，远低于一票否决阈值。
2. **与 TD3BC phase0c 正式结果对比**：`+21.1pp (ep1000)` / `+32.1pp (ep2000)`；均值和 std 两项同时改善。
3. **机制观察**：ReBRAC 首次翻转了 TD3BC 主线 "`2000` 差于 `1000`" 的趋势（ReBRAC 下 ep2000 好于 ep1000）。
4. **优化方差的主控杠杆是 β1**：`β1 ≤ 2.0` 时 seed 44 系统性崩盘；`β1 = 4.0` 时 seed 44 恢复到与 seed 42/43 同量级。
5. **Stage C finalist 已锁定**：见 §6.4。

### 输出位置

- `checkpoints/offline/rebrac/screening/`
- `results/offline/rebrac/screening/`
- `results/offline/rebrac/screening/summaries/overview.{csv,json}`

执行入口：[notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb)（执行归档 `rebrac_screen_completed.ipynb`）。

## 6.4 Stage C：正式确认 `【当前立即执行】`

### 目标

把 Stage B screening 的 winner 升级到正式 5-seed 结果，与 TD3BC phase0c 的正式成绩同口径可比。

### Finalist（由 Stage B 锁定）

- **主 finalist**：`(β1=4.0, β2=2.0)`，在 `crosscomp-1000` 和 `crosscomp-2000` 上共用。
- **ep2000 backup finalist**：`(β1=4.0, β2=1.0)`——仅在 `crosscomp-2000` 上追加一组 5-seed，作为 "主 finalist 若在更多 seed 下方差放大" 的接盘配置。
  - 理由见 [rebrac_experiment_report.md §6.7](./rebrac_experiment_report.md)：ep2000 上它与主 finalist 的 Δmean 只有 1.7pp、Δstd 只有 0.008，再添加一组只多 5 runs 的代价，但能显著减少 Stage C 复核崩盘时的 rework 风险。
- **`crosscomp-1000` 不追加 backup**：主 finalist 相对第二名（`β1=2.0, β2=2.0`）的差距是 3.3pp，没有合适的 backup 候选。

### 推荐矩阵

- `crosscomp-1000`：1 个 finalist × 5 seeds = 5 runs
- `crosscomp-2000`：2 个 finalist × 5 seeds = 10 runs
- 合计：15 runs

### 预算

- seeds：`42 / 43 / 44 / 45 / 46`（前 3 个与 Stage B 重叠，后 2 个为新增 held-out）
- train epochs：`64`（与 Stage B 对齐；Stage B0 已经验证足够）
- val manifest episodes：`40`
- test manifest episodes：`100`

### 比较对象

必须与以下结果同表报告：

- TD3BC `phase0c` Stage C 正式 5-seed（即 `success 0.672 ± 0.045 @ crosscomp-1000, α=0.25` 与 `success 0.596 ± 0.036 @ crosscomp-2000, α=0.15`）
- BC formal result（如果 `phase0c` 留有相应 checkpoint）
- 作为 internal 对照：Stage B 的 3-seed 结果（检查新增 seed 45/46 是否改变 winner 判定）

### 这一阶段的作用

这一步不是为了"再刷一次最好成绩"，而是为了把 ReBRAC 的结论从"screening 观察"提升成"正式基线"。如果 Stage C 5-seed 下 ReBRAC 仍然稳定超越 TD3BC phase0c，则：

1. Stage B 的阳性结论正式成立；
2. ReBRAC 成为新的 deployable 主基线；
3. 才进入 Stage D `worldcomp` teacher-gap follow-up。

### 触发 Stage C 失败的条件

以下任一条件若在 Stage C 复核时出现，视为 Stage B 结论"未通过 5-seed 复核"，需要回到 Stage B 做 follow-up：

1. ep1000 上主 finalist 的 `mean_test_success_rate` 跌破 TD3BC phase0c 正式成绩 `0.672`；
2. ep2000 上两个 finalist 的 `mean_test_success_rate` 都跌破 `0.75`；
3. 任一 finalist 的 `std_test_success_rate > 0.10`（即 Stage B 的低 std 不可复现）。

如果只有 ep2000 主 finalist 掉队、backup 撑住，直接用 backup 作为 ep2000 正式成绩，不做 rework。

## 6.5 Stage D：`worldcomp` teacher-gap follow-up `【只在 Stage C 有正信号时做】`

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

## 6.6 Stage E：必要时的最小 ablation

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

1. 一份 screening summary：test 指标（success/return/safety/time 的 mean/std）+ 训练侧诊断（`mean_critic_penalty`、`mean_target_q`、`mean_critic_penalty_ratio`）。
2. 一份 formal summary（如果进入 Stage C）。
3. 一张对比表，至少包含：
   - TD3BC vs BC vs ReBRAC on `crosscomp-1000`
   - TD3BC vs BC vs ReBRAC on `crosscomp-2000`
4. 如果进入 Stage D，再补：
   - ReBRAC deployable vs privileged-critic vs `worldcomp` baseline

---

## 10. 推荐执行顺序

如果把这一阶段压缩成最小行动清单，推荐顺序是：

1. **Stage B0**：跑 `(β1=2.0, β2=1.0) × {crosscomp-1000, crosscomp-2000} × {42,43,44}, TRAIN_EPOCHS=128` 的训练预算 probe，确认 `TRAIN_EPOCHS=64` 是否足够（已完成）。
2. **Stage B**：跑 `crosscomp-1000/2000` 的 `3 × 2` screening（β1 ∈ {1.0, 2.0, 4.0} × β2 ∈ {1.0, 2.0}），3 seeds，TRAIN_EPOCHS=64。**已完成**，winner = `(β1=4.0, β2=2.0)`；详见 [rebrac_experiment_report.md §6](./rebrac_experiment_report.md)。
3. **Stage C**（当前）：跑 5-seed 正式复核，finalist 为 `crosscomp-1000: (β1=4.0, β2=2.0)` + `crosscomp-2000: (β1=4.0, β2=2.0) 与 (β1=4.0, β2=1.0)`，合计 15 runs。seeds `42/43/44/45/46`，TRAIN_EPOCHS=64，test manifest 升到 100 episodes。
4. 只有 Stage C 结果成立（详见 §6.4 的失败条件），才进入 **Stage D**（`worldcomp-1000` follow-up）。
5. 如果 Stage C 不成立，按 §6.4 末尾的条件分类分流：单 finalist 掉队用 backup 接；全员崩盘则回到 Stage B 扩网格；扩网格仍无进展则停止扩 ReBRAC，转向 XQL。

---

## 11. 一句话执行意见

> ReBRAC 这一阶段的任务不是“广泛扫参”，而是用最小、可解释的 canonical screening 去判断：当前 TD3BC 尚未解释的那部分问题，是否主要来自 critic regularization 与 dual penalties 的不足。
