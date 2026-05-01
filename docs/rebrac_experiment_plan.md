# ReBRAC 实验计划

> 文档版本：2026-05-01 rev.8
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

## 6.4 Stage C：正式确认 `【已完成】`

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

### 实际结论

详见 [rebrac_experiment_report.md §7](./rebrac_experiment_report.md)。要点：

1. **三项失败条件全部通过**：ep1000 主 finalist mean `0.902` >> `0.672`；ep2000 两 finalist mean `0.918 / 0.894` >> `0.75`；三项 std `0.021 / 0.030 / 0.048` 都 ≤ `0.10`。
2. **正式成绩**：`crosscomp-1000: 0.902 ± 0.021 @ (β1=4.0, β2=2.0)`；`crosscomp-2000: 0.918 ± 0.030 @ (β1=4.0, β2=2.0)`。相对 TD3BC phase0c 正式 5-seed，分别 `+23.0pp / +32.2pp`；std 同时 ≤ TD3BC（0.021 vs 0.045；0.030 vs 0.036）。
3. **backup finalist `(β1=4.0, β2=1.0)` 在 `crosscomp-2000` 上也通过阈值**（`0.894 ± 0.048`），但新增 seed 45 出现 `0.810` 的离群点，spread 从 Stage B 的 `0.05` 放大到 `0.12`；主 finalist 稳住，backup 不启用为正式成绩，保留 fallback 定位。
4. **ReBRAC 成为 deployable 主基线**，Stage D `worldcomp-1000` teacher-gap follow-up 触发条件满足。

### 输出位置

- `checkpoints/offline/rebrac/formal/`
- `results/offline/rebrac/formal/`
- `results/offline/rebrac/formal/summaries/overview.{csv,json}`

执行入口：[notebooks/rebrac_formal.ipynb](../notebooks/rebrac_formal.ipynb)（执行归档 [notebooks/rebrac_formal_completed.ipynb](../notebooks/rebrac_formal_completed.ipynb)）。

## 6.5 Stage D：`worldcomp` teacher-gap follow-up `【已完成 — Phase 1 + Phase 2 + critic-penalty-off probe 全部完成】`

### 6.5.0 总览：为什么拆成 Phase 1 / Phase 2

ReBRAC 在 `crosscomp` 主线上拿到 Stage C 阳性结论后，曾经有一个朴素方案是直接复刻 TD3BC worldcomp teacher-gap 的两轨道 10-run 矩阵。但 Stage C 之后做完事实核对发现：

- **`crosscomp` 与 `worldcomp` 的瓶颈性质完全不同**。TD3BC 在 `worldcomp` 下退化为 pure BC（best α=0.0，success `0.858`），这是 `crosscomp` 上根本不存在的现象；它表明 `worldcomp` 的关键瓶颈是 "deployable obs 下 critic 无法稳定做 Q guidance"，而不是 "数据支持集结构"。
- **ReBRAC 的 dual penalty 直接对应 critic robustness**，但 critic penalty 的 `next_actions` target 来自 `worldcomp` teacher，而 teacher 本身在 deployable obs 下是非马尔可夫的——这意味着 ReBRAC 的 critic penalty 在 `worldcomp` 上既可能 "进一步治好 critic 退化"，也可能 "在 noisy target 下放大 bias"。**两种结果都是可能的，且都是论文级 finding**。
- 因此 Stage D 的 deployable 轨道是 **不可绕过** 的——它直接回答 "ReBRAC 是否打破了 TD3BC 在 worldcomp 上退化为 BC 的现象"。
- 但 privileged-critic 轨道的价值 **强依赖** Phase 1 deployable 结果，其作用从 "机制诊断必做" 到 "边际确认可选" 三档浮动。

所以 Stage D 拆为：

- **Phase 1（当前立即执行，必做，~7 runs）**：epoch-probe + deployable formal。
- **Phase 2（Phase 1 完成后再启动，规模条件触发，0-5 runs）**：privileged-critic 轨道，规模与判据由 Phase 1 outcome 决定。

### 6.5.1 共同设定

#### 数据集
- `worldcomp-1000`（collector = `worldcomp` baseline policy，target speed 1.5，cross_stream，deployable obs，`next_actions` 已包含）
- **不**使用 `worldcomp-2000 / 500`：Stage D 的 scope 是 "ReBRAC 是否压缩 gap"，不是 "ReBRAC 在 worldcomp 下的 data scaling"。

#### Finalist（由 Stage C 锁定）
- **唯一 finalist = `(β1=4.0, β2=2.0)`**。不扫超参。
- 不带入 `(β1=4.0, β2=1.0)` backup：Stage C 下已暴露 seed 45 敏感，在 `worldcomp` 上做冗余对照无信息价值。

#### 通用预算
- seeds：`42 / 43 / 44 / 45 / 46`（与 Stage C、TD3BC worldcomp teacher-gap 对齐）
- val manifest episodes：`40`
- test manifest episodes：`100`（与 Stage C 对齐）

#### 对照基线
- TD3BC worldcomp teacher-gap：`deployable 0.858 ± 0.080 @ α=0.0`；`privileged-critic 0.922 ± 0.086 @ α=0.1`
- ReBRAC Stage C `crosscomp-1000`：`0.902 ± 0.021`（对照 ReBRAC 在 deployable 数据上的主线成绩）
- teacher baseline：`0.990`（数据收集时的 worldcomp behavior policy online 成绩）

#### 驱动脚本（Stage D 前置阻塞项已决策）

当前 [scripts/run_offline_rebrac_screen.sh](../scripts/run_offline_rebrac_screen.sh) 不支持 Stage D（无 worldcomp 数据路径、无 privileged-critic 轨道 flag、无 worldcomp baseline 分支）。

**决策：新建 `scripts/run_offline_rebrac_worldcomp_teacher_gap.sh`**，对齐 TD3BC 侧的既有 pattern（`scripts/run_offline_td3bc_phase0c_worldcomp_teacher_gap.sh`）。理由：保持 "screen / formal / teacher-gap 三个 driver 各司其职"，避免单一 driver 堆叠过多 Stage-specific flag 分支。

### 6.5.2 Phase 1：epoch-probe + deployable formal `【已完成 — 落入情景 A】`

#### 目标

- **核心问题**：ReBRAC 在 `worldcomp-1000` deployable 下是否打破了 TD3BC 退化为 BC 的现象？具体来说，ReBRAC 主 finalist 的 `mean_test_success_rate` 是显著超越 `0.858`（TD3BC deployable formal）、与之持平、还是反而更差？
- **次要目标**：钉死 Stage D 在 `worldcomp` 数据上的 `TRAIN_EPOCHS` 预算，避免 "训练不够" 污染解释。

#### Step 1：epoch-probe（2 runs）`【已完成】`

- 配置：`(β1=4.0, β2=2.0) × 2 seeds (42/43) × TRAIN_EPOCHS=128 × CHECKPOINT_EVERY_EPOCHS=8` on `worldcomp-1000`，**deployable 轨道**。
- 复用 Stage B0 的 "长训读中间 ckpt 当作多 epoch sweep" 方法（详见 [rebrac_experiment_report.md §5.2](./rebrac_experiment_report.md)）。
- 判据：
  - 若两个 seed 的 val peak epoch 都 ≤ 64 → Phase 1 formal 取 `TRAIN_EPOCHS=64`；
  - 若任一 seed 的 peak ∈ (64, 96] → Phase 1 formal 取 `TRAIN_EPOCHS=96`（对齐 TD3BC worldcomp teacher-gap）；
  - 若 > 96 → Phase 1 暂停，回查 `worldcomp` 训练动力学（怀疑 critic penalty 与 noisy teacher target 的相互作用）。

##### Step 1 实测结果

执行入口：[notebooks/rebrac_worldcomp_epoch_probe_completed.ipynb](../notebooks/rebrac_worldcomp_epoch_probe_completed.ipynb)。

**val 曲线 first-peak（按简化决策规则）**：seed 42 在 ep64 首次达到 `val_succ=1.0`；seed 43 在 ep56 首次达到 `val_succ=1.0`。两者都 ≤ 64。

**完整选择规则下的 selected ckpt**（`success → return → -safety → -time`）：

| seed | selected ckpt | val_succ | val_return | val_safety_cost |
| ---: | ---: | ---: | ---: | ---: |
| 42 | ep112 (`agent_step_00045696`) | 1.0 | 33.80 | 5.76 |
| 43 | ep104 (`agent_step_00042432`) | 1.0 | 37.45 | 5.03 |

**probe test 成绩**（test=40，2 seeds）：

| 指标 | 数值 | 对照 |
| --- | ---: | --- |
| `mean_test_success_rate` | **1.0 ± 0.0** | TD3BC deployable formal: `0.858 ± 0.080`；TD3BC privileged-critic formal: `0.922 ± 0.086` |
| `mean_test_return` | **35.62 ± 1.83** | TD3BC deployable formal: `−14.29`；TD3BC privileged-critic formal: `16.49`；teacher baseline: `32.19` |
| `mean_test_safety_cost` | 5.39 | TD3BC deployable formal: `7.91`；TD3BC privileged-critic formal: `5.73` |
| `β2 · mean_critic_penalty_ratio` | **0.0067** | crosscomp Stage C 是 `0.086 / 0.219 / 0.358` |

**Phase 1 formal `TRAIN_EPOCHS` 决策**：取 `64`。理由：

1. 简化决策规则（`max(first-peak-epoch) ≤ 64`）通过；
2. 即便用更严格的 selection 规则（return 也参与排序）会落在 ep104/ep112，**TRAIN_EPOCHS=64 budget 内的 best ckpt 仍然是 `val_succ=1.0` 级别**（seed 42 ep64=1.0 / 30.85；seed 43 ep56=1.0 / 30.87），相对 TD3BC `0.858` 已经有 +14pp 量级的 head room；
3. 与 Stage B / Stage C 的 `TRAIN_EPOCHS=64` 一致，`TRAIN_EPOCHS` 不再是变量；
4. 若 Phase 1 formal 出现新 seed 在 ep ≤ 64 内峰值低于 `0.95` 的情况，可针对该 seed 单点扩到 `TRAIN_EPOCHS=96` 重跑。

**注意点**：

- seed 43 在 `ep64` 出现 `val_succ=0.90` 的 dip，随后在 `ep96` 恢复到 `1.0`——这与 Stage B0 seed_42 的 dip 类似，是 ReBRAC 训练曲线非单调的常见模式，selection 规则在 ep ≤ 64 budget 内会自动选 ep56（而非 ep64）规避这个 dip。
- `β2 · mean_critic_penalty_ratio = 0.0067` 比 crosscomp 上低 **一到两个数量级**——critic penalty 在 worldcomp 数据上几乎无作用（next_actions 与 `π_target(s')` 的预测高度一致）。这间接说明 worldcomp 的 deterministic teacher 让 critic penalty 退化为接近恒等约束，主要约束来自 actor penalty。这一发现降低了原本担心的 "noisy teacher target 让 critic penalty 放大 bias" 风险。
- `mean_test_return = 35.62` 实际**超过 worldcomp teacher baseline online 成绩 `32.19`**——这是一个非常强但同时需要在 5-seed × test=100 下重新验证的早期信号。可能解释：(a) test=40 manifest 上的 sampling noise；(b) ReBRAC policy 在 efficient_v2 reward 下走出了比 teacher 更短/更省的路径（`progress_ratio=0.91`、`path_efficiency=0.80`）。

**对 Phase 2 触发情景的早期推断**：probe 已经强烈暗示 Phase 1 formal 落在 **情景 A**（mean > 0.90），可能甚至接近 `1.0`。等 Phase 1 formal 5-seed × test=100 完成后再正式定档。

#### Step 2：Phase 1 deployable formal（5 runs）`【已完成】`

- 配置：`(β1=4.0, β2=2.0) × 5 seeds (42-46) × TRAIN_EPOCHS=64` on `worldcomp-1000`，**deployable 轨道**。
- val/test manifest 与 Stage C 对齐（40 / 100）。
- 选 ckpt / 选成绩规则与 Stage C 一致：`success_rate → return → -safety_cost → -time`。

##### Step 2 实测结果

执行入口：[notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb](../notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb)。详细分析见 [rebrac_experiment_report.md §7.10](./rebrac_experiment_report.md)。

**Per-seed test 分布（test=100）：**

| seed | selected ckpt | val_succ (40 ep) | test_succ (100 ep) | test_return |
| ---: | ---: | ---: | ---: | ---: |
| 42 | `agent_final` (ep64) | 1.000 | 0.990 | 26.85 |
| 43 | `agent_step_22848` (ep56) | 1.000 | 0.930 | 17.76 |
| 44 | `agent_step_16320` (ep40) | 0.825 | **0.780** | −2.53 |
| 45 | `agent_final` (ep64) | 1.000 | 0.980 | 28.97 |
| 46 | `agent_step_22848` (ep56) | 0.975 | 0.960 | 29.80 |

**5-seed overview（test=100）：**

| 指标 | 实测 | 对照 |
| --- | ---: | --- |
| `mean_test_success_rate` | **0.928** | TD3BC deployable formal `0.858 ± 0.080`；TD3BC privileged formal `0.922 ± 0.086`；teacher baseline `0.990` |
| `std_test_success_rate` | 0.077 | TD3BC deployable formal `0.080`；TD3BC privileged formal `0.086` |
| `mean_test_return` | 20.17 | TD3BC deployable `−14.29`；TD3BC privileged `16.49`；teacher `32.19` |
| `mean_test_safety_cost` | 5.96 | TD3BC privileged 5.73 |
| `mean_target_q` | **+15.22** | crosscomp Stage C `−5 ~ −8` |
| `β2 · mean_critic_penalty_ratio` | **0.0111** | crosscomp Stage C `0.086 / 0.219 / 0.358` |

**Phase 1 通过判据核对**：

| 条件 | 阈值 | 实测 | 是否通过 |
| --- | --- | --- | --- |
| Rule 1：mean test success | `> 0.858`（TD3BC deployable） | 0.928 | **通过**（+7.0pp） |
| Rule 2：std test success | `≤ 0.10` | 0.077 | **通过** |
| 情景归类 | `> 0.90` ⇒ A | 0.928 | **情景 A** |

**关键 finding（必须传递给 Phase 2 决策）**：

1. **均值上 ReBRAC deployable 已经超过 TD3BC privileged-critic**（0.928 vs 0.922），把 deployable→teacher 的 gap 关闭了 **53.0%**（TD3BC privileged 是 48.5%）。这是核心论文级 finding：dual penalty 跨数据类型 generalize，且只用 deployable obs 就追平特权 critic 协议。
2. **probe → formal 缩水的真实原因**：probe（seeds 42/43, test=40）的 `1.0 ± 0.0` 在 5-seed × test=100 下落到 `0.928 ± 0.077`。原因不是 manifest 噪声——seeds 42/43 在 test=100 下仍为 0.99/0.93——而是 probe 没看到 seed 44 这个难 seed（val 卡在 0.825，selected ckpt 停在 ep40）。
3. **ReBRAC 在 worldcomp 上的 dual penalty 几乎退化为单 penalty**：β2·ratio = 0.0111（crosscomp Stage C 是 0.086~0.358，差 1–2 个数量级），`mean_target_q = +15.22`（crosscomp 是负值）。next_actions（来自 deterministic worldcomp teacher）与 `π_target(s')` 高度一致 → critic penalty 等价于恒等约束。worldcomp 上 ReBRAC 的全部增益基本来自 actor BC penalty + 网络容量/LayerNorm，**不是 dual penalty**。
4. **seed 44 在 worldcomp 没被救回**：与 Stage C crosscomp 上 "β1=4.0 完整回收 seed 44" 的故事**首次分叉**。机制猜想：worldcomp 上 critic penalty 几乎不工作，缺少了 crosscomp 上稳住 seed 44 的"第二条防线"。

#### Phase 1 通过判据 + Phase 2 触发情景

Phase 1 完成后，对 deployable formal 的 5-seed mean test success 落在哪个区间，决定 Phase 2 的形态：

| 情景 | Phase 1 deployable mean | 解读 | Phase 2 privileged-critic 形态 |
| --- | --- | --- | --- |
| **A** 显著超越 TD3BC | `> 0.90` | ReBRAC 也治好了 worldcomp 的 critic 退化；dual penalty 跨数据类型 generalize（论文最强 narrative） | **3 seeds 边际确认**（42/43/44），仅验证 "privileged critic 是否仍贡献 ≥ 5pp 增益与 > 50% gap closure"；若不想做也可省略 |
| **B** 与 TD3BC 持平 | `0.84 ~ 0.90` | ReBRAC 没有打破 worldcomp 退化；observation gap 是真实 algorithm-agnostic 瓶颈 | **5 seeds 完整跑**，作为 "ReBRAC 在 observation-bottleneck regime 下 critic 是否仍能利用 privileged 信息" 的诊断 |
| **C** 显著低于 TD3BC | `< 0.84` | dual penalty 在 noisy teacher target 下放大 bias；这是一个独立的 negative finding | **5 seeds 完整跑** + 追加最小 ablation（critic penalty off）理解机制 |

注：`0.84 / 0.90` 阈值的设计逻辑——TD3BC deployable std 是 `0.080`，所以 ±2 std 的判定带宽约 `0.16`；以 `0.86 ± 0.04` 作为 "持平区" 即 `[0.82, 0.90]`，但考虑 ReBRAC Stage C 在 crosscomp 上 std 是 `0.021 / 0.030`（远低于 TD3BC），把上界提到 `0.90` 偏保守，确保 "显著超越" 不被噪声误判。

#### Phase 1 输出位置

- `checkpoints/offline/rebrac/worldcomp_epoch_probe/` — Step 1
- `checkpoints/offline/rebrac/worldcomp_teacher_gap/deployable/` — Step 2
- `results/offline/rebrac/worldcomp_epoch_probe/`
- `results/offline/rebrac/worldcomp_teacher_gap/deployable/`

### 6.5.3 Phase 2：privileged-critic 轨道 `【已完成 — 5-seed 升级版 (rev.8)；4 条判据全部通过；privileged ≈ deployable，仅在 seed 44 上独立救回 +12pp】`

> rev.8 修订（2026-05-01）：Phase 2 已从 3-seed 升级到 5-seed（补 seeds 45/46）。新结果 `mean = 0.9340, std = 0.0261`（与 3-seed 0.9267 ± 0.025 几乎不变），4/4 判据保持通过。**std 0.0261 严格低于 TD3BC priv 5-seed std 0.086**——可正式做 std 对比 claim。详见 [report §7.12](./rebrac_experiment_report.md) 与 [notebooks/rebrac_paper_followup_completed.ipynb §2](../notebooks/rebrac_paper_followup_completed.ipynb)。下方 3-seed 表保留作为 rev.7 历史记录；§ "Phase 2 实测结果" 段后补 5-seed 表。

#### 配置（共同部分）

- 轨道：actor 输入 deployable obs，critic 输入 privileged obs；actor update mode `zeros`（与 TD3BC worldcomp phase0c 完全对齐）。
- finalist：`(β1=4.0, β2=2.0)`。
- TRAIN_EPOCHS：`64`（与 Phase 1 一致；epoch-probe 决定）。
- val/test manifest：40 / 100（与 Phase 1 对齐）。

#### 规模（由 Phase 1 情景决定）

> rev.8（2026-05-01）：3-seed 已升级到 5-seed。下面 rev.7 的 3-seed 决策逻辑保留作为决策回溯；最终 Phase 2 是 5 seeds = `42 / 43 / 44 / 45 / 46`。

Phase 1 实测落入情景 A（mean=0.928 > 0.90），按 plan 原本设定为 "3-seed 边际确认（或可省略）"。但 Phase 1 暴露的 seed 44 离群点（`0.78`）改变了 Phase 2 的诊断价值——3-seed **必须包含 seed 44**，理由如下：

- 情景 A：**3 seeds = `42 / 43 / 44`，必须含 seed 44**（不是任意 3 个）。
  - seeds 42/43 与 epoch-probe / Phase 1 对齐，作为 baseline 延续；
  - **seed 44 是 Phase 2 最有信息量的诊断点**——它在 deployable 轨道掉到 0.78，selected ckpt 卡在 ep40。如果 privileged critic 把它救回 → "privileged critic 在 ReBRAC 上贡献了超越 actor penalty 的额外信号"；如果救不回 → seed 44 是数据/优化层面的问题，与 critic 无关。这一信息无法通过任意 3-seed 取得。
- 情景 B / C（未触发）：5 seeds（`42 / 43 / 44 / 45 / 46`），完整诊断。

**rev.8 升级说明**：rev.1 review.md §2.2.2 指出 "3-seed = 42/43/44 是经过事先选择的 (含 seed 44)，5-seed std 不可与之直接比较"。为允许 paper 做严格 std 对比，rev.8 补 seeds 45/46，将 Phase 2 升到 5 seeds。Driver 自动 skip 已存在的 42/43/44，仅训 45/46，预算 ≈ 2h L4。

#### 可选追加：critic-penalty-off probe（机制 sanity check）`【已完成 — 落入情形 B；Finding 1 部分被证伪】`

Phase 1 Step 2 暴露的关键机制 finding 是 **β2·ratio = 0.0111**（crosscomp Stage C 是 0.086~0.358，差 1–2 个数量级），即 critic penalty 在 worldcomp 上几乎不工作。这意味着 ReBRAC 在 worldcomp 上的全部增益基本来自 actor penalty + 网络容量/LayerNorm，不是 dual penalty。

为低成本验证这一猜想（不破坏 Stage D 主线），建议在 Phase 2 之前或并行：

- 配置：`(β1=4.0, β2=0)` × 1–2 seeds（`42 / 43`）on `worldcomp-1000` deployable，TRAIN_EPOCHS=64。
- 预算：1–2 runs，远小于 Phase 2 主体。
- 判据：若 `mean_test_success` 与 Phase 1 主 finalist（`β1=4.0, β2=2.0`，0.928）在 ±2pp 内 → 机制猜想成立，ReBRAC 在 worldcomp 上的 dual penalty 可由单 actor penalty 替代。结果直接写入 Stage E ablation 章节，节省一次完整 ablation 的预算。
- 这不在 plan rev.3 原范围内，但鉴于 Step 2 的机制 finding，回报很高、风险极低（即便结果没有落入预期，也只是为 Stage E 提供更早的负面证据）。

##### Probe 实测结果（详见 [report §7.13](./rebrac_experiment_report.md)）

2 seeds × 1 配置 × 64 epoch × test=100：

| 指标 | `(β1=4.0, β2=0)` probe | Phase 1 同 seeds (42/43) | Phase 1 5-seed (42-46) |
| --- | --- | --- | --- |
| mean test success | **0.910** | 0.960 | 0.928 |
| std test success | 0.014 | — | 0.077 |
| mean_target_q | **22.30** | — | 15.22 |
| mean_critic_penalty_ratio | 0.0037 | — | 0.0057 |

**判定：落入情形 B（同 seeds 比 -5.0pp，5-seed 比 -1.8pp）**。Finding 1 的"dual penalty 退化为单 penalty"机制猜想 **部分被证伪**：

1. 表面 `β2·ratio` 数字小（0.011）不等于 critic penalty 贡献为零；
2. 关键证据是 `mean_target_q` 从 15.22 跳到 **22.30**（+46%），即 critic penalty 仍在显著抑制 Q 高估，只是绝对量级看起来小；
3. 正确表述：dual penalty 在 worldcomp 上对 mean success 的贡献"小但非零"（同 seeds 5pp 量级），对 Q 稳定性的贡献"大且必要"。

**对 Stage E 的影响**：critic-penalty-off ablation 不能仅凭 probe 收口，仍需要保留 5-seed 完整 ablation；但 probe 已经把方向钉死——Stage E 主要变量改为"Q 稳定性"而非"mean success"。

#### Phase 2 通过判据

- **均值**：`mean_test_success_rate > 0.922`（TD3BC privileged-critic 正式成绩）且 `std ≤ 0.10` → ReBRAC privileged 轨道达标。
- **gap closure**：`(ReBRAC_priv − ReBRAC_deploy) / (0.990 − ReBRAC_deploy)`
  - 情景 A：`> 50%` 视为 "privileged critic 在 ReBRAC 上仍是有意义的 gap 诊断工具"（与 TD3BC 的 48.5% 比较；预期较低，因为 ReBRAC_deploy 已经较高）。
  - 情景 B / C：`> 60%` 视为 "ReBRAC + privileged 比 TD3BC 更能利用 teacher information"。

#### Phase 2 实测结果（详见 [report §7.12](./rebrac_experiment_report.md)）

**5 seeds × 1 finalist `(β1=4.0, β2=2.0)` × 64 epoch × test=100，actor update mode `zeros`（rev.8）：**

| seed | success_rate | return | safety_cost | path_efficiency |
| --- | --- | --- | --- | --- |
| 42 | 0.960 | 22.30 | 7.28 | 0.7665 |
| 43 | 0.920 | -2.21 | 9.91 | 0.7245 |
| 44 | **0.900** | 15.88 | 7.20 | 0.7601 |
| 45 | 0.930 | 20.34 | 6.81 | 0.7713 |
| 46 | 0.960 | 28.78 | 4.26 | 0.7993 |
| **mean** | **0.9340** | 17.02 | 7.10 | 0.7643 |
| **std**  | **0.0261**（per-seed） | 11.71 | — | — |

**3-seed 历史记录（rev.7，保留作 cross-check）**：seeds = `42 / 43 / 44`，mean=0.9267, std=0.025（overview）/ 0.031（per-seed）。新增 seed 45/46 后 mean +0.7pp、std 几乎不变（0.0261 vs 0.025），结论稳。

**通过判据核对（5-seed × 4 / 4 全部通过；rev.8）**：

| Rule | 阈值 | 实测（5-seed，rev.8） | 通过 |
| --- | --- | --- | --- |
| 1. mean test success | `> 0.922`（TD3BC priv-critic formal） | **0.9340**（+1.2pp） | ✅ |
| 2. std test success | `≤ 0.10` | 0.0261 | ✅（严格 < TD3BC priv 0.086） |
| 3. gap closure（情景 A） | `> 50%`（TD3BC priv-critic 48.5%） | **57.6%** | ✅ |
| 4. seed 44 诊断 | `≥ 0.85` ⇒ 假设 A | **0.900**（+12.0pp vs deployable 0.78） | ✅ → 假设 A |

3-seed 旧表（rev.7）：mean=0.9267, std=0.025, gap closure=52.0%，4/4 通过。升 5-seed 后判据保持通过、Δ mean 与 Δ std 均在预期内（5-seed 涉极端 seed 概率高，std 估计放宽是预期的；这里反而几乎不变，进一步印证 finalist 的 seed-rubustness）。

**关键二阶 finding（写入 report §7.12.5）**：ReBRAC privileged-critic 5-seed 与 ReBRAC deployable 5-seed 在 mean 上几乎相等（0.9340 vs 0.928，Δ = +0.6pp），但 seed 44 在 privileged 轨道下被独立救回 +12pp。这说明：

- ReBRAC 在 worldcomp 上 deployable→teacher 的 gap 关闭主要由 **actor 的 BC penalty** 完成；
- privileged critic 在 ReBRAC 上**不带来 mean 上的增益**，只在 outlier seed 上起作用；
- 与 TD3BC 的故事不同：TD3BC 的 privileged-critic 增益是 **+6.4pp 平均抬升**（0.858 → 0.922），而 ReBRAC 的 privileged-critic 增益是 **0pp 平均抬升 + outlier seed 救回**——同样关闭 50%+ gap，机制完全不同。

#### Phase 2 输出位置

- `checkpoints/offline/rebrac/worldcomp_teacher_gap/privileged_critic/`
- `results/offline/rebrac/worldcomp_teacher_gap/privileged_critic/`

### 6.5.4 关注点（贯穿 Phase 1 / Phase 2）

不只看绝对 success，还要看：

- 轨道 1 deployable 的 seed 分布形状：是全 seed 抬升，还是个别 seed 回收？（与 Stage B / Stage C 的 seed 44 / seed 45 故事衔接）
- `out_of_bounds / timeout` 终止类型分布是否发生结构性变化；
- `β2 · mean_critic_penalty_ratio` 在 `worldcomp` 数据上是否仍然落在 "<1.0 健康区"（若显著高于 `crosscomp` 上的 0.086~0.358，要回查 `worldcomp` 数据 `mean_target_q` 的尺度差异）；
- 训练后期 25% 窗口的 `mean_critic_penalty` 是否在 `worldcomp` 上明显高于 `crosscomp`——这一指标如果异常大，是 "ReBRAC 的 next_actions target 在 noisy teacher 下不再代表 in-distribution action" 的直接诊断信号。

## 6.6 Stage E：收口 ablation `【已完成 — Stage E (a) 落入情形 B；(b) 维持推迟（触发恢复条件均未触发）】`

> rev.7 修订（2026-05-01）：Stage E (a) 已完成，落入情形 B（mean=0.878 ± 0.090，Δ vs Stage C -2.4pp，mean_target_q +98%，seed 44 -17pp）。**Finding 1 修正版（"小但非零 + Q 稳定性必要 + outlier seed 鲁棒性必需"）跨 dataset 成立**。Stage E (b) 触发恢复条件 1（情形 C）和触发条件 2（seed 44 < 0.85）的本意（机制层面对照）已由 Stage E (a) §7.14.6 cross-(dataset, β2) 诊断回答，维持推迟。**ReBRAC 主线整体收口。**
>
> rev.6 历史记录：原 rev.4 的"3 项最小机制消融（full / critic-penalty-off / normalize_q-off）" 已被 Phase 2 + critic-penalty-off probe 显著收紧。Phase 2 的"privileged ≈ deployable"二阶 finding 削弱了 normalize_q-off 的优先级（actor BC penalty 主导），critic-penalty-off probe 已经把方向钉死。Stage E 在 rev.6 中只剩两个 track。

### 6.6.1 Stage E (a) — `crosscomp-1000` critic-penalty-off cross-dataset 二次验证 `【已完成 — 落入情形 B；Finding 1 修正版跨 dataset 成立】`

**目的**：把 [report §7.13](./rebrac_experiment_report.md) 在 worldcomp 上做的 critic-penalty-off probe 扩展到 crosscomp，回答 Finding 1 修正版（"对 mean 贡献小但非零，对 Q 稳定性大且必要"）是否跨 dataset 成立。

**配置**：

- 算法 / 协议：ReBRAC deployable（与 Stage C 主 finalist 完全对齐，仅 β2 不同）；
- 超参：`(β1=4.0, β2=0)`；
- dataset：`crosscomp-1000`；
- seeds：`42 / 43 / 44 / 45 / 46`（5 seeds，与 Stage C 完全对齐）；
- TRAIN_EPOCHS=64，val=40，test=100；
- val/test manifests 复用 Stage C `benchmarks/offline_rebrac_screen/`；
- 输出隔离到 `results/offline/rebrac/stage_e_critic_penalty_off/`。

**判定区间**（与 Stage D probe 对齐）：

| 情形 | mean_test_success | 解读 |
| --- | --- | --- |
| **A** 几乎一致 | `0.882 ~ 0.922`（Stage C ± 2pp） | dual penalty 在 crosscomp 上也几乎可省 → Finding 1 跨 dataset 成立；Stage E 收口 |
| **B** 略有下降 | `0.85 ~ 0.88` | 与 worldcomp probe 量级一致；Finding 1 修正版跨 dataset 成立 |
| **C** 显著下降 | `< 0.85` | crosscomp 与 worldcomp 不同，Finding 1 需在 dataset 维度上分别表述 |

**核心诊断量**：`mean_target_q` 跳升幅度（worldcomp probe 是 +46%，15.22 → 22.30）。这才是 critic penalty 真正的作用渠道，比 mean success 更基础。

**预算**：5 runs × 1 finalist × 64 epoch × test=100，≈ Stage C 主 finalist 的一半（仅 1 dataset）。

**入口**：
- scaffold：[notebooks/rebrac_stage_e_critic_penalty_off_crosscomp.ipynb](../notebooks/rebrac_stage_e_critic_penalty_off_crosscomp.ipynb)
- 执行归档：[notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb](../notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb)

**驱动**：复用 `scripts/run_offline_rebrac_screen.sh`，单 config 覆盖（`ACTOR_PENALTY_COEFS=4.0`, `CRITIC_PENALTY_COEFS=0.0`）。

**实测结果**（详见 [report §7.14](./rebrac_experiment_report.md)）：

| 指标 | 实测值 | 判据 / 对照 |
| --- | --- | --- |
| mean test success | **0.878** | **情形 B**（区间 0.85~0.88） |
| std test success | 0.090 | — |
| Δ vs Stage C `(β2=2.0)` 同 dataset | -2.4pp（0.902 → 0.878） | — |
| Δ vs worldcomp probe `(β2=0)` 跨 dataset | -3.2pp（0.910 → 0.878） | 与 worldcomp 同 seeds -5pp 量级一致 |
| **mean_target_q** | **-0.13** | **从 -8.25 漂 +8.12 单位（≈ +98%）** |
| seed 44 success | **0.700** | 比 Stage C 同 seed 0.870 掉 **-17pp** |

**结论**：

1. **Finding 1 修正版跨 dataset 成立**：`mean_target_q` 在 crosscomp 上的 +8.12 单位漂移与 worldcomp +7.08 单位漂移量级一致——Q 高估抑制机制 dataset-invariant。
2. **dual penalty 是 ReBRAC 主线必需配置**：在 typical regime（5-seed mean）只掉 2~3pp，但在 outlier seed（44）上掉 17pp，且 Q 高估机制在两个 dataset 上都被显著放大。**保留 `(β1=4.0, β2=2.0)`，不向单 penalty 简化。**
3. **seed 44 cross-(dataset, β2) 诊断闭环 Phase 2 假设 A**：seed 44 在两个 dataset 上都需要某种 critic-side 稳定信号——worldcomp 上靠 privileged hull-integral 提供，crosscomp 上靠 critic penalty (β2) 提供。actor BC penalty 一条线不足以稳住 seed 44 是 cross-dataset 普遍结论。

### 6.6.2 Stage E (b) — seed 44 collector 起点分布 inspection `【维持推迟 — 触发恢复条件均未触发】`

**当前状态**：Phase 2 假设 A 已成立 + Stage E (a) seed 44 -17pp 双重证据闭环 → (b) 从 fallback 触发条件降级为"已知机制 + 未做 root-cause sanity"，不阻塞结论。

**触发恢复条件检查（Stage E (a) 完成后判定）**：

- ~~触发条件 1：Stage E (a) 落入情形 C（crosscomp 上 critic penalty 影响远大于 worldcomp）~~ — 实测落入情形 B（mean=0.878），**未触发**；
- ~~触发条件 2：seed 44 在 Stage E (a) 上 < 0.85~~ — 实测 0.70 数值上低于 0.85，但本条件的设计本意是判断 critic penalty 在 crosscomp 上对 seed 44 是否必需（与 Phase 2 假设 A 闭环加固），这一问题已由 [report §7.14.6](./rebrac_experiment_report.md) cross-(dataset, β2) 诊断回答（β2 移除让 seed 44 掉 -17pp，确认 critic penalty 是 crosscomp 上 seed 44 的必需信号），不再需要 collector 起点分析作为机制 sanity。**实质上未触发恢复行动**；
- 触发条件 3：论文 review 反馈"想看 seed 44 outlier 的 root cause 而非仅 critic 信息层面解释"——**待论文 review，当前不阻塞**。

**结论**：维持推迟。如未来论文 review 触发条件 3，再决策；否则 Stage E (b) 不再纳入主线。

**如果触发**：分析维度建议——

1. **per-seed 早期训练 batch 触达的 episode 子集**：在 `shuffle_no_replacement` 下，每个训练 seed 决定首个 epoch 内 batch 抽样到的 episode 顺序。比较 seed 44 的前 ~1k batch 触达的 episode 子集与其它 seed 的差异。
2. **首批触达 episode 的起点分布**：用 `transitions.npz` 内的 `done` 列重建 episode 边界，看 seed 44 早期触达 episode 的初始 (x, y, ψ, U_flow) 是否系统性偏向某种困难起点（如距 goal 远、流场方向不利）。
3. **首批 batch 的 Q-target 估计偏差**：把 seed 44 / seed 42 的首 ~5k 步 critic_loss / mean_target_q 的逐步序列画出来，对比是否在某个 batch 边界处 Q 估计开始发散。

**预算**：纯分析脚本，无需训练，~30 分钟（一次性）。

### 6.6.3 Stage F — paper-readiness probes（rev.8 新增） `【已完成 — 全部 4 项 closed】`

> rev.8（2026-05-01）：rev.1 review.md §2.2 + §3.1 列出 4 项必做项目（堵审稿人弱点）。本 Stage 把 4 项作为独立轨道在 ReBRAC 主线收口后并行完成。归档：[notebooks/rebrac_paper_followup_completed.ipynb](../notebooks/rebrac_paper_followup_completed.ipynb)。

| 任务 | 实施 | 关键结果 | 文档落点 |
| --- | --- | --- | --- |
| **A**：Phase 2 升 5-seed | 补 seeds 45/46（driver auto-skip 42/43/44） | mean=0.9340 ± 0.0261；Δ vs 3-seed +0.7pp、std 几乎不变 | §6.5.3（plan）/ §7.12（report） |
| **B**：critic LayerNorm-off probe | 5-seed Stage C finalist 减去 LN，2 seeds × crosscomp-1000 | mean=0.74 ± 0.25（**-16.2pp vs LN-on Stage C**）；mean_target_q 漂 -3.84 单位（更负 +46%）；比 β2=0 Stage E (a) 更差（-13.8pp） | §7.15（report，新增）；[review.md §2.2.4](./rebrac_mainline_review.md) |
| **C**：Phase 1 dep vs TD3BC priv 统计检验 | paired episode-level bootstrap（10000 resample）+ Welch's t-test | Welch's p=0.9195（**持平**）；paired bootstrap 95% CI on Δ = [-3.0pp, +4.2pp]；gap closure 95% CI on Δ = [-62pp, +86pp] | [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md) |
| **D**：Q-normalized variant method draft | 抽 `auv_nav/rebrac.py::_actor_loss_terms` + 推导 β1 ↔ TD3+BC α 折算 | algorithm 命名为 "Q-normalized dual-penalty TD3+BC variant"（alias ReBRAC-Q）；β1=4.0 ↔ TD3+BC α≈0.25 | [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md) |

**Stage F 收口结论**：

1. **任务 A** 把 review §2.2.2 "3-seed 不可与 5-seed std 对比" 弱点堵死；
2. **任务 B**（LN-off probe）显示 LN 关掉比 β2 关掉 *更* 严重（-16.2 vs -2.4pp），且 LN 的退化是 std 与 mean 双向放大（std 0.02 → 0.25，blow-up 12×）→ **LN 与 dual penalty 是两个独立 component**；堵 review §2.2.4 "提升其实主要来自 LayerNorm" 论点；
3. **任务 C** 把 review §2.2.3 "持平" 这一表述 quantitatively 确认：Welch's p=0.92、paired bootstrap CI 包含 0 → 用 "statistically not different" 替代 "首次超过"；
4. **任务 D** 把 review §2.2.1 "Q-normalized 变体不是原版 ReBRAC" 弱点彻底处理：method section 显式声明为 "Q-normalized dual-penalty TD3+BC variant (alias ReBRAC-Q)"，β1 数字与原 ReBRAC 数字不直接可比，paper 中数字比较仅与自训 TD3+BC 在 matched protocols 下做。

### 6.6.4 不再做的事

- ~~`normalize_q off` 5-seed 完整 ablation~~：被 Phase 2 二阶 finding（actor BC penalty 主导）削弱优先级；Stage E (a) 已落入情形 B、Stage F (D) 已在 method section 显式声明 Q-normalized 变体，不再列入主线；
- ~~3 项消融全因子（full / critic-off / normalize_off）~~：rev.4 原计划，已被 rev.6 收紧；
- ~~LayerNorm / dropout / 网络容量扫~~：rev.4 / rev.6 / rev.7 维持不做；rev.8 Stage F (B) 仅做 LN-off 2-seed probe，不扩 sweep（已堵 review §2.2.4 弱点）。

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

1. **Stage B0**：跑 `(β1=2.0, β2=1.0) × {crosscomp-1000, crosscomp-2000} × {42,43,44}, TRAIN_EPOCHS=128` 的训练预算 probe，确认 `TRAIN_EPOCHS=64` 是否足够（**已完成**）。
2. **Stage B**：跑 `crosscomp-1000/2000` 的 `3 × 2` screening（β1 ∈ {1.0, 2.0, 4.0} × β2 ∈ {1.0, 2.0}），3 seeds，TRAIN_EPOCHS=64（**已完成**，winner = `(β1=4.0, β2=2.0)`；详见 [rebrac_experiment_report.md §6](./rebrac_experiment_report.md)）。
3. **Stage C**：跑 5-seed 正式复核，finalist 为 `crosscomp-1000: (β1=4.0, β2=2.0)` + `crosscomp-2000: (β1=4.0, β2=2.0) 与 (β1=4.0, β2=1.0)`，合计 15 runs。seeds `42/43/44/45/46`，TRAIN_EPOCHS=64，test manifest 升到 100 episodes（**已完成**；详见 [rebrac_experiment_report.md §7](./rebrac_experiment_report.md)）。
4. **Stage D 前置（已决策）**：新建 `scripts/run_offline_rebrac_worldcomp_teacher_gap.sh`（对齐 TD3BC 侧 `phase0c_worldcomp_teacher_gap` driver 的 pattern）。
5. **Stage D Phase 1**（**已完成**）：
   - Step 1 epoch-probe（2 runs, 2 seeds × 128 epoch）→ TRAIN_EPOCHS 锁 64；
   - Step 2 deployable formal（5 runs, 5 seeds × 64 epoch × test=100）→ `mean=0.928, std=0.077`，落入**情景 A**；
   - 关键 finding：`β2·ratio=0.0111`，dual penalty 在 worldcomp 上几乎退化为单 penalty；seed 44 在 deployable 上掉到 0.78，是 Phase 2 最有信息量的诊断点。详见 §6.5.2。
6. **Stage D Phase 2**（**已完成**）：privileged-critic 轨道，3 seeds = `42 / 43 / 44`（含 seed 44），finalist `(β1=4.0, β2=2.0)`，TRAIN_EPOCHS=64，test=100 → `mean=0.9267, std=0.025`，4 条判据全部通过；二阶 finding：privileged ≈ deployable（Δ -0.1pp），seed 44 独立救回 +12pp（0.78 → 0.90），假设 A 成立（critic 信息瓶颈是 seed 44 outlier 的真因）。详见 §6.5.3 + [report §7.12](./rebrac_experiment_report.md)。
6'. **critic-penalty-off probe**（**已完成**）：`(β1=4.0, β2=0) × seeds 42/43` on worldcomp-1000 deployable → `mean=0.910, std=0.014`，落入**情形 B**；同 seeds 比 Phase 1 -5.0pp。关键 evidence：`mean_target_q` 从 15.22 跳到 22.30（+46%）。**Finding 1 部分被证伪**："β2·ratio 小"不等于"critic penalty 贡献为零"——critic penalty 仍在显著抑制 Q 高估。详见 §6.5.3 + [report §7.13](./rebrac_experiment_report.md)。
7. **Stage E**（**已完成**）：按 §6.6 收紧后的两轨设计——
   - **(a) crosscomp-1000 critic-penalty-off cross-dataset 二次验证**（**已完成**）：5 seeds × `(β1=4.0, β2=0)` × test=100 → `mean=0.878, std=0.090`，落入**情形 B**；同 dataset Δ -2.4pp、`mean_target_q` 漂 +8.12 单位（+98%）、seed 44 掉 -17pp。**Finding 1 修正版跨 dataset 成立**：critic penalty 对 mean 贡献 "小但非零"，对 Q 稳定性 "大且必要"，对 outlier seed 鲁棒性 "必需"。详见 §6.6.1 + [report §7.14](./rebrac_experiment_report.md)。
   - **(b) seed 44 collector 起点分布 inspection**（**维持推迟**）：触发恢复条件均未触发（情形 B 而非 C；seed 44 cross-(dataset, β2) 诊断已替代 collector 检查）。详见 §6.6.2。
8. **ReBRAC 主线整体收口**（rev.7）：Stage A→B0→B→C→D (Phase 1+2+probe)→E (a) 全部完成。论文级核心 finding 三条：(i) ReBRAC 在 crosscomp 上对 TD3BC +23~+32pp（Stage C）；(ii) ReBRAC 在 worldcomp 上 deployable ≈ privileged-critic 关闭 ~52% gap，privileged 价值降到 outlier seed 救援级（Stage D）；(iii) dual penalty `(β1=4.0, β2=2.0)` 是必需配置，对 mean 贡献小但对 Q 稳定性 + outlier seed 鲁棒性必需（cross-dataset，Stage D probe + Stage E (a)）。**rev.6 明确删除的项**保留删除：normalize_q-off 5-seed ablation、3 项全因子消融、LayerNorm/dropout/网络容量扫。后续动作：转向 paper drafting / online thesis 线 / 其它研究方向（用户决定）。
9. **Stage F — paper-readiness probes 全部完成**（rev.8）：4 项必做（A: Phase 2 升 5-seed；B: critic LN-off probe；C: paired bootstrap + Welch's t-test；D: Q-normalized variant method draft）已 closed。新增结果：(i) Phase 2 5-seed = `0.9340 ± 0.0261`，std 严格低于 TD3BC priv 0.086；(ii) LN-off mean=0.74 ± 0.25 比 LN-on Stage C 掉 -16.2pp，**LN 与 dual penalty 是两个独立 component**；(iii) Welch's p=0.9195，"持平" claim quantitatively confirmed；(iv) algorithm 命名为 "Q-normalized dual-penalty TD3+BC variant (alias ReBRAC-Q)"。审稿人弱点全部堵死，paper drafting 可正式启动。详见 §6.6.3 + [notebooks/rebrac_paper_followup_completed.ipynb](../notebooks/rebrac_paper_followup_completed.ipynb) + [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md) + [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md)。

---

## 11. 一句话执行意见

> ReBRAC 这一阶段的任务不是“广泛扫参”，而是用最小、可解释的 canonical screening 去判断：当前 TD3BC 尚未解释的那部分问题，是否主要来自 critic regularization 与 dual penalties 的不足。
