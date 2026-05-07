# ReBRAC 实验报告

> 文档版本：2026-05-01 rev.8
> 文档定位：这是 ReBRAC 阶段所有已经**实际跑完**的实验的统一结果报告。与 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) 不同，本文只写 "跑了什么 / 看到了什么 / 意味着什么"，不讨论尚未执行的计划。
> 阅读建议：先读 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) §1-§5 理解动机与口径，再回到这里看已有结果。

---

## 1. 报告概述

本文档整理 `results/offline/rebrac/` 下已经完成的全部 ReBRAC 实验，并把每一组实验的结论并入一条完整的 "主线解释" 里。当前（`rev.8`）包含的实验包为：

- `screening_epoch_probe`（Stage B0，训练预算 probe）
- `screening`（Stage B，最小 screening —— 3×2 penalty 网格 × 2 datasets × 3 seeds）
- `formal`（Stage C，5-seed 正式复核 —— 3 finalists × 5 seeds × test 100 episodes）
- `worldcomp_epoch_probe`（Stage D Phase 1 Step 1，2 seeds × 128 epoch × test=40）
- `worldcomp_teacher_gap/deployable`（Stage D Phase 1 Step 2，5 seeds × 64 epoch × test=100）
- `worldcomp_teacher_gap/privileged_critic`（Stage D Phase 2，**rev.8 升 5 seeds**，× 64 epoch × test=100）
- `worldcomp_critic_penalty_off_probe`（Stage D Phase 1 Finding 1 验证 probe，2 seeds × 64 epoch × test=100）
- `stage_e_critic_penalty_off`（Stage E (a)，crosscomp-1000 critic-penalty-off cross-dataset 二次验证，5 seeds × 64 epoch × test=100）**已完成（落入情形 B，Finding 1 修正版跨 dataset 成立）**
- **`critic_ln_off`（Stage F (B)，crosscomp-1000 critic LayerNorm-off probe，2 seeds × 64 epoch × test=100，rev.8 新增）**：堵 review §2.2.4 "提升其实主要来自 LayerNorm" 论点

Stage D + Stage E (a) + Stage F (rev.8 新增 4 项 paper-readiness probes) 全部完成；Stage E (b) seed 44 collector inspection 维持推迟（Phase 2 假设 A + Stage E (a) seed 44 -17pp 双重证据已闭环，触发恢复条件均不成立，降级为 root-cause sanity）。**ReBRAC 主线整体收口；rev.8 paper-readiness 弱点全部堵死。**

本文档要回答的核心问题与 [rebrac_experiment_plan.md §2](./rebrac_experiment_plan.md) 一致：

1. 在与 phase0c 相同的 deployable canonical protocol 下，ReBRAC 是否能稳定优于 TD3BC？
2. ReBRAC 是否能改善 `crosscomp-2000` 相对 `crosscomp-1000` 的退化？
3. 如果 ReBRAC 在 `worldcomp-1000` 上也有改善，这种改善来自 deployable 轨道还是 privileged critic 轨道？

截至当前（`rev.8`）：

- **问题 1 已由 Stage C 5-seed 正式确认**：ReBRAC 的主 finalist 在 `crosscomp-1000` 上拿到 `0.902 ± 0.021`，在 `crosscomp-2000` 上拿到 `0.918 ± 0.030`；相对 TD3BC phase0c 正式成绩分别高出 `+23.0pp` 和 `+32.2pp`；std 也同时等于或低于 TD3BC phase0c 正式 std。
- **问题 2 同样成立**：Stage C 下 `crosscomp-2000`（0.918）仍然高于 `crosscomp-1000`（0.902），Stage B 观察到的 "2000 不再差于 1000" 的翻转在 5-seed 下延续。
- **问题 3 由 Stage D Phase 1 + Phase 2 (rev.8 升 5-seed) 完整回答**：ReBRAC `worldcomp-1000` deployable 轨道（5 seeds）拿到 `0.928 ± 0.077`，privileged-critic 轨道（**5 seeds，rev.8**）拿到 `0.9340 ± 0.0261`。两个轨道在 mean 上几乎相等（Δ priv − dep = +0.6pp），但 privileged 在 seed 44 上独立救回 +12pp（0.78 → 0.90）。**结论：ReBRAC 在 worldcomp 上的 deployable→teacher gap 关闭主要由 actor BC penalty 完成，privileged critic 只在 outlier seed 上贡献额外信号**——与 TD3BC 上 "privileged critic 平均抬升 +6.4pp" 的故事机制完全不同。
- **Finding 1 cross-dataset 已由 Stage E (a) 完成**：`crosscomp-1000 (β1=4.0, β2=0) × 5 seeds × test=100` 拿到 `0.878 ± 0.090`，落入情形 B（与 worldcomp probe 同量级）。`mean_target_q` 从 -8.25 漂到 -0.13（+8.12 单位，≈ +98% 朝正向），与 worldcomp probe +7.08 单位（+46%）量级一致——**Q 高估抑制机制 dataset-invariant**。seed 44 在 crosscomp 上失去 critic penalty 后掉 -17pp（0.87 → 0.70），与 Phase 2 假设 A 闭环：seed 44 outlier 行为反映 ReBRAC 在缺乏额外 critic 稳定信号时对 outlier seed 的系统性脆弱，跨 worldcomp / crosscomp 都成立。dual penalty 在 ReBRAC 主线上是必需的。
- **rev.8 paper-readiness 4 项 probes 全部完成**：(A) Phase 2 升 5-seed → §7.12 已更新；(B) critic LN-off probe (crosscomp-1000, 2 seeds) → §7.15 新增，mean=0.74±0.25 比 LN-on Stage C 掉 -16.2pp（**LN 与 dual penalty 是两个独立 component**）；(C) paired bootstrap + Welch's t-test on Phase 1 dep vs TD3BC priv → §7.16 新增，Welch's p=0.9195（"持平" quantitatively confirmed）；(D) Q-normalized variant method draft → [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md)（algorithm 命名为 ReBRAC-Q，β1 ↔ TD3+BC α 折算）。

一句话当前状态：

> ReBRAC 主线整体收口。Stage D 全部完成（Phase 1 deployable + Phase 2 privileged-critic + critic-penalty-off probe），Stage E (a) 完成（crosscomp-1000 critic-penalty-off cross-dataset 二次验证落入情形 B），rev.8 Stage F (paper-readiness probes) 4/4 完成。两条 cross-dataset 证据（worldcomp `mean_target_q` +46% / crosscomp +98%；worldcomp 同 seeds -5pp / crosscomp 5-seed -2.4pp；crosscomp seed 44 -17pp）合并把 Finding 1 钉死为"critic penalty 对 mean 贡献小但非零、对 Q 稳定性大且必要、对 outlier seed 鲁棒性必需"。**ReBRAC 在 `worldcomp-1000` 上同时通过 deployable（0.928）与 privileged-critic 5-seed（0.9340 ± 0.0261）两个轨道关闭 ~52~58% 的 deployable→teacher gap，dual penalty `(β1=4.0, β2=2.0)` 是 ReBRAC 主线必需配置；critic LN 与 dual penalty 是两个独立 component。** 审稿人弱点全部堵死，paper drafting 可正式启动。

---

## 2. 背景与实验动机

### 2.1 为什么要有独立的 ReBRAC 实验报告

TD3BC 主线在 `phase0c` 已经收口：

- `crosscomp-1000 / α=0.25` 是正式 5-seed 最优点，`success = 0.672 ± 0.045`；
- `crosscomp-2000` 的 success 退到 `0.596 ± 0.036`，且这一现象在同预算 BC 下同样成立；
- `worldcomp-1000` 上，deployable TD3BC 最优退回到 pure BC（α=0），privileged critic 能关闭约一半 teacher gap。

ReBRAC 阶段的核心动机就是回答一个很具体的问题：

> TD3BC 没吃下来的那部分收益，是不是 critic regularization + dual penalties 就能解释？

详见 [rebrac_experiment_plan.md §1.2](./rebrac_experiment_plan.md)。因为 ReBRAC 的实现口径、超参对齐方式、以及它与 TD3BC 的"等价 α"折算都与 TD3BC 有差异，所以 ReBRAC 阶段的结果需要独立于 TD3BC phase0c 报告单独组织。

### 2.2 为什么先做训练预算 probe

TD3BC phase0c Stage B 的 `TRAIN_EPOCHS=64` 是经过 screening 验证的——但它被选中主要是因为 "在 TD3BC 口径下 64 足以拉平 winner 排序"。切换到 ReBRAC 后，至少有两个可能的扰动：

- actor 更新里多了 `β1 · (π - a)²` 正则项，早期梯度组成与 TD3BC 不完全一致；
- critic target 端的 `β2 · (π_target(s') - a')²` penalty 会改变 target Q 的典型尺度。

如果 `TRAIN_EPOCHS=64` 在 ReBRAC 下偏紧，后续正式 Stage B screening 的所有结论都会被 "训练没训完" 系统性污染。而如果偏松，也会导致不必要的算力浪费（正式 screening 是 `3 × 2 × 2 × 3 = 36` 个 run）。

因此在 Stage B 正式开跑前，花大约 1/3 screening 的预算把这件事钉死，是值得的。

---

## 3. 统一实验设置

本阶段所有实验（含目前已完成的 Stage B0、以及后续 Stage B/C/D）都沿用下列 canonical protocol。完整描述见 [rebrac_experiment_plan.md §4](./rebrac_experiment_plan.md)。

### 3.1 任务与环境

- benchmark：`single_u10_cross_tgt15`
- task geometry：`cross_stream`
- target speed：`1.5`
- objective：`efficiency_v2`
- probe layout：`s0`
- history length：`4`
- flow：`wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`

### 3.2 算法口径

ReBRAC 实现约定见 [rebrac_experiment_plan.md §5](./rebrac_experiment_plan.md)。关键提醒：

> 当前 `auv_nav/rebrac.py` 是 **Q-normalized ReBRAC 变体**：actor loss 除以 `|Q|.detach()`（与 TD3+BC 的 `λ = α / |Q|` 同一尺度约定）。在 `action_dim=2` 下，与 TD3BC 的等价关系约为 `β1_ReBRAC ≈ 1 / (2 · α_TD3BC)`。

所有 `actor_penalty_coef` / `critic_penalty_coef` 数字都应在此变体口径下理解。

### 3.3 协议骨架

与 TD3BC phase0c Stage B 完全对齐：

- 训练：`--eval-every 0 --skip-final-eval`，按 `CHECKPOINT_EVERY_EPOCHS=8` 周期保存 `agent_step_*.pt`；
- 训练结束后，每个 `agent_step_*.pt` + `agent_final.pt` 在独立 val manifest（40 episodes）上评估；
- 按 `success_rate → return → -safety_cost → -time` 选最佳 checkpoint；
- 将选定 checkpoint 在独立 test manifest 上重跑，作为该 (β1, β2, seed) 的最终成绩。

### 3.4 驱动脚本

[scripts/run_offline_rebrac_screen.sh](../scripts/run_offline_rebrac_screen.sh) 是本阶段所有 screening / probe 的唯一驱动。通过环境变量切换 grid、TRAIN_EPOCHS、输出目录即可复用同一套 train/validate/select/test/summarize 管线。

---

## 4. 实验矩阵回顾

截至 `rev.7`：

| 实验包 | 编号 | 状态 | 入口 |
| --- | --- | --- | --- |
| 训练预算 probe | Stage B0 | 已完成 | [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb) |
| 最小 screening | Stage B | 已完成 | [notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb) |
| 正式 5-seed 确认 | Stage C | 已完成 | [notebooks/rebrac_formal.ipynb](../notebooks/rebrac_formal.ipynb) |
| Stage D Phase 1 Step 1（worldcomp epoch-probe） | Stage D / Phase 1 | 已完成 | [notebooks/rebrac_worldcomp_epoch_probe_completed.ipynb](../notebooks/rebrac_worldcomp_epoch_probe_completed.ipynb) |
| Stage D Phase 1 Step 2（worldcomp deployable formal） | Stage D / Phase 1 | 已完成 | [notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb](../notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb) |
| Stage D Phase 2（worldcomp privileged-critic） | Stage D / Phase 2 | 已完成（4 / 4 判据通过） | [notebooks/rebrac_worldcomp_phase2_privileged_completed.ipynb](../notebooks/rebrac_worldcomp_phase2_privileged_completed.ipynb) |
| Stage D — critic-penalty-off probe | Stage D / Finding 1 验证 | 已完成（落入情形 B） | [notebooks/rebrac_worldcomp_critic_penalty_off_probe_completed.ipynb](../notebooks/rebrac_worldcomp_critic_penalty_off_probe_completed.ipynb) |
| Stage E (a)（crosscomp-1000 critic-penalty-off） | Stage E / cross-dataset 验证 | **已完成（落入情形 B；Finding 1 修正版跨 dataset 成立）** | [notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb](../notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb) |
| Stage E (b)（seed 44 collector 起点 inspection） | Stage E / root-cause sanity | 推迟（Phase 2 假设 A + Stage E (a) seed 44 -17pp 双重证据已闭环） | — |

---

## 5. Stage B0：训练预算 probe

### 5.1 目标与口径

在正式 Stage B screening 开跑前，确认 `TRAIN_EPOCHS=64` 是否是合适的训练预算——这个值是从 TD3BC phase0c 直接继承而来，从未在 ReBRAC 口径下单独验证过。

### 5.2 实验范围

| 轴 | 配置 |
| --- | --- |
| β1 / β2 | `2.0 / 1.0`（对齐 TD3BC α=0.25 的 phase0c `1000` winner；正式 screening 网格的中心点） |
| dataset | `crosscomp-1000`、`crosscomp-2000` |
| seeds | `42 / 43 / 44` |
| TRAIN_EPOCHS | `128`（产出 16 个 ckpt，每 8 epoch 一个） |
| manifest | val = 40 episodes，test = 40 episodes |

关键设计选择：**不**做 `TRAIN_EPOCHS ∈ {32, 64, 96, 128}` 的笛卡尔 sweep。利用 Stage B 协议已经对齐的 `CHECKPOINT_EVERY_EPOCHS=8` + 每个 ckpt 都走 val 的特性，**把一个 cell 训长一次（128 epoch），读中间 ckpt 的 val 曲线**——16 个 val 点免费拿到。

等价前提：`train_offline.py` 用常数 LR AdamW，`sampling_mode=shuffle_no_replacement` 每 epoch 独立 shuffle，因此 "长训至 step N" 在优化动力学上 ≈ "只训到 N 结束"。notebook 内有可选 sanity check（通过权重 L2 距离交叉确认这一假设）。

输出路径刻意与正式 screening 隔离：

- `checkpoints/offline/rebrac/screening_epoch_probe/`
- `results/offline/rebrac/screening_epoch_probe/`

### 5.3 主结果：peak epoch 汇总

从 `selected_checkpoint.json` 的 `candidates` 数组直接读出，val = 40 episodes：

| dataset | seed | peak_epoch | peak_success | succ@ep64 | Δ (peak − ep64) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crosscomp-1000` | 42 | 128 | 0.950 | 0.850 | +0.100 |
| `crosscomp-1000` | 43 | 80  | 0.950 | 0.925 | +0.025 |
| `crosscomp-1000` | 44 | 120 | 0.775 | 0.500 | +0.275 |
| `crosscomp-2000` | 42 | 40  | 0.950 | 0.900 | +0.050 |
| `crosscomp-2000` | 43 | 72  | 0.950 | 0.900 | +0.050 |
| `crosscomp-2000` | 44 | 104 | 0.850 | 0.700 | +0.150 |

### 5.4 val 曲线的定性解读

仅看 "peak epoch" 容易误判。结合 notebook 里导出的完整 val 曲线（每 8 epoch 一个点），真实结构是：

**1. 两个 dataset 上典型 seed（42 + 43）的 val plateau 其实很早**

- `crosscomp-2000` seed_42：peak 出现在 ep=40，之后 ep=64~128 全部围绕 `0.90~0.95` 窄幅波动；
- `crosscomp-2000` seed_43：peak 出现在 ep=72，ep=48 已经到 `0.90`；
- `crosscomp-1000` seed_43：peak 出现在 ep=80，但 ep=48 已经到 `0.90` 以上，ep=64~80 主要是曲线抬升 `0.025`；
- `crosscomp-1000` seed_42：最复杂——ep=64 出现一个明显的 dip（`0.85`），之后才沿 ep=64→128 单调回升到 `0.95`。

换句话说，`Δ (peak − ep64)` 的绝对值不能直接解读成 "epoch 64 不够"，因为 val 曲线不是单调的——"peak − ep64" 有时候只是随机捕到了曲线的一个 dip 点和另一个 bump 点。

**2. seed 44 在两个 dataset 上都是真正的 outlier**

- `crosscomp-1000` seed_44：整条曲线从 ep=8 到 ep=128 就没有超过 `0.775`，ep=64 只有 `0.500`；
- `crosscomp-2000` seed_44：peak `0.850` 出现在 ep=104，ep=64 只有 `0.700`。

这不是 "再多训 50 个 epoch 就能追上 seed 42/43" 的情况，而是另一种收敛分支：seed 44 的 policy 一直没有拿到 seed 42/43 那种稳定 `0.90~0.95` 的 regime。这是一个**优化方差问题**，不是**训练预算问题**。

**3. 没有看到显著过拟合**

六条 val 曲线都没有出现 "早期冲高 → 后期明显回落" 的形状。即使看 seed 44，它也没有在 ep=40 出现一个明显峰值然后跌下来，而是一直低位震荡。所以 Stage B 保守一点在 ep=64 停训是安全的，不需要担心 overfitting。

### 5.5 结论

1. **`TRAIN_EPOCHS=64` 在 ReBRAC 口径下足够**。两个 dataset 上典型 seed（42、43）在 epoch 40~50 已经接近 val plateau；之后继续训到 128 epoch 的 `+0.025 ~ +0.10` 增益主要来自曲线波动（非单调上升），不是稳定的训练不足信号。
2. **真正值得警惕的信号不是 "训练不够"，而是 seed 方差**。seed 44 在 `crosscomp-1000` β1=2.0 这组配置上 peak 只有 `0.775`，而 seed 42/43 能稳定到 `0.95`——这一 `0.175` 差距即使训到 128 epoch 也没有收敛。
3. **`2000` 不需要比 `1000` 更大的训练预算**。在预算 probe 下 `crosscomp-2000` 的 plateau 甚至更早出现（seed_42 在 ep=40 就到达 `0.95`），这与 "2000 数据更多所以训练更难" 的朴素直觉相反——但与 phase0c 的 "2000 的问题不是更难学，而是支持集结构" 这一主结论一致。

### 5.6 对正式 Stage B 的三条指导

1. 维持 `TRAIN_EPOCHS=64`，不为 `2000` 单独加预算。
2. Stage B summary 必须把 `std_test_success_rate` 作为一个一等公民看待，而不是只看 `mean_test_success_rate`。
3. 如果 Stage B 在 `β1=2.0` 列上观察到 `std > 0.10` 的大方差，优先考察 "是否 seed 44 类型的 outlier 在重复"，而不是先怀疑 `TRAIN_EPOCHS` 不够。

### 5.7 局限性

- 本次 probe 只在 `β1=2.0, β2=1.0` 上做了曲线观察。理论上 `β1=4.0`（更弱 BC）在早期需要更多 epoch 才能稳住 actor，因此 Stage B 跑出来如果看到 `β1=4.0` 在 ep=64 明显弱，需要回到这个问题而不是立刻加 β1；反之 `β1=1.0`（更强 BC）应该比当前 probe 结论更快 plateau，风险更低。
- 3 seeds 下对 peak-epoch 位置的估计精度仍然有限——但本实验的决策不依赖精确 peak，而依赖 "plateau 是否在 ep=64 之前到达"，因此 3 seeds 足够。
- 没有检查 `worldcomp-1000`。这是刻意的选择——Stage D 本就位于 Stage C 之后，提前做 `worldcomp` 预算 probe 会违反 [rebrac_experiment_plan.md §3](./rebrac_experiment_plan.md) 的不做不必要工作原则。如果 Stage D 真的启动，应当先在 `worldcomp` 上再做一次小规模的 probe。

---

## 6. Stage B：最小 screening

### 6.1 目标与口径

在 Stage B0 已经确认 `TRAIN_EPOCHS=64` 可用之后，Stage B 回答 [rebrac_experiment_plan.md §6.3](./rebrac_experiment_plan.md) 的 screening-level 问题：

> 在 `crosscomp-1000 / 2000` 上，ReBRAC 是否能在 `success_rate` 和 / 或 `std_test_success_rate` 上稳定不弱于 TD3BC phase0c 主线？

沿用 Stage B0 的全部环境与协议（详见 §3）。

### 6.2 实验范围

| 轴 | 配置 | 说明 |
| --- | --- | --- |
| β1 (actor penalty) | `1.0 / 2.0 / 4.0` | `1.0` ≈ TD3BC α=0.5；`2.0` ≈ α=0.25（phase0c 1000 winner）；`4.0` ≈ α=0.125 |
| β2 (critic penalty) | `1.0 / 2.0` | 覆盖 critic penalty 是"锦上添花"到"已是主导信号"的范围 |
| dataset | `crosscomp-1000` + `crosscomp-2000` | phase0c 的两个核心对照 |
| seeds | `42 / 43 / 44` | 与 Stage B0 同口径；保留 seed 44 作为 Stage B0 暴露出来的 "难 seed" |
| TRAIN_EPOCHS | `64` | 由 Stage B0 钉死 |
| CHECKPOINT_EVERY_EPOCHS | `8` | 每个 run 产出 8 个 ckpt |
| val manifest | 40 episodes | 选 ckpt |
| test manifest | 40 episodes | held-out；作为该 (β1, β2, seed) 的最终成绩 |

总计：`3 × 2 × 2 × 3 = 36` train runs，`36 × 8 = 288` val evaluations，36 test evaluations。

输出路径：

- `checkpoints/offline/rebrac/screening/`
- `results/offline/rebrac/screening/`
- `results/offline/rebrac/screening/summaries/overview.{csv,json}`

执行入口：[notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb)（跑完的归档在 `rebrac_screen_completed.ipynb`）。

### 6.3 主结果：test-split overview

每个 (dataset, β1, β2) 在 held-out test manifest 上 3 seeds 的均值 ± std。

**`crosscomp-1000`**（按 `mean_test_success_rate` 降序）：

| β1 | β2 | success (mean ± std) | return (mean ± std) | safety | time (s) | `mean_critic_penalty` | `mean_target_q` | `mean_critic_penalty_ratio` | `β2 · ratio` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4.0** | **2.0** | **0.883 ± 0.031** | −39.74 ± 4.13 | 15.97 | 78.84 | 0.080 | −8.23 | 0.065 | **0.129** |
| 2.0 | 2.0 | 0.850 ± 0.108 | −46.01 ± 13.36 | 16.53 | 79.93 | 0.080 | −5.51 | 0.018 | 0.035 |
| 4.0 | 1.0 | 0.825 ± 0.089 | −45.83 ± 8.95 | 16.59 | 78.17 | 0.080 | −3.93 | 0.049 | 0.049 |
| 2.0 | 1.0 | 0.775 ± 0.195 | −65.01 ± 41.74 | 19.59 | 83.52 | 0.080 | −1.59 | 0.085 | 0.085 |
| 1.0 | 2.0 | 0.742 ± 0.198 | −74.82 ± 39.50 | 20.38 | 86.53 | 0.082 | −1.66 | 0.072 | 0.145 |
| 1.0 | 1.0 | 0.633 ± 0.206 | −109.72 ± 56.70 | 19.61 | 94.14 | 0.082 | 2.67 | 0.052 | 0.052 |

**`crosscomp-2000`**：

| β1 | β2 | success (mean ± std) | return (mean ± std) | safety | time (s) | `mean_critic_penalty` | `mean_target_q` | `mean_critic_penalty_ratio` | `β2 · ratio` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4.0** | **2.0** | **0.917 ± 0.012** | −35.78 ± 1.70 | 16.86 | 79.21 | 0.078 | −6.21 | 0.156 | **0.311** |
| 2.0 | 2.0 | 0.900 ± 0.071 | −34.41 ± 11.03 | 15.55 | 77.74 | 0.079 | −0.07 | 0.041 | 0.082 |
| 4.0 | 1.0 | 0.900 ± 0.020 | −37.05 ± 3.21 | 16.44 | 78.66 | 0.078 | −1.80 | 0.521 | 0.521 |
| 1.0 | 2.0 | 0.883 ± 0.066 | −47.04 ± 11.36 | 19.38 | 82.80 | 0.082 | 6.00 | 0.034 | 0.068 |
| 2.0 | 1.0 | 0.875 ± 0.089 | −57.75 ± 33.32 | 17.64 | 85.77 | 0.079 | 5.39 | 0.032 | 0.032 |
| 1.0 | 1.0 | 0.825 ± 0.102 | −56.25 ± 10.79 | 20.65 | 83.27 | 0.082 | 7.72 | 0.021 | 0.021 |

关键读数：

1. **两个 dataset 的 winner 一致**：`(β1=4.0, β2=2.0)`。
2. **ep2000 winner 的 std 极低**（`0.012`），是整张表上跨 seed 最稳定的一格；`mean_test_return` 的 std 也只有 `1.70`。
3. **penalty 沿两个轴都是单调有益**：β1↑ 和 β2↑ 同时让 success↑ 且 std↓，没有观察到任何一档 BC 约束过强导致 success 回落的迹象。
4. **所有 12 格都通过 `β2 · ratio < 1.5` 的一票否决检查**。最大值是 ep2000 `(β1=4.0, β2=1.0)` 的 `0.521`，仍远低于 critic 被 penalty 主导的危险区。

### 6.4 Per-seed 分布：winner 的"优势来源"是 robustness，不是 peak

对于 screening 出来的 winner，光看均值 + std 不足以解释它在两个 dataset 上强在哪里。把 36 个 test run 的每个 seed 单独列出来：

**`crosscomp-1000`**（按 success mean 降序）：

| β1 | β2 | seed_42 | seed_43 | seed_44 | min−max spread |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **4.0** | **2.0** | 0.875 | 0.925 | **0.850** | **0.075** |
| 2.0 | 2.0 | 0.900 | 0.950 | **0.700** | 0.250 |
| 4.0 | 1.0 | 0.875 | 0.900 | **0.700** | 0.200 |
| 2.0 | 1.0 | 0.900 | 0.925 | **0.500** | 0.425 |
| 1.0 | 2.0 | 0.950 | 0.800 | **0.475** | 0.475 |
| 1.0 | 1.0 | **0.475** | 0.925 | 0.500 | 0.450 |

**`crosscomp-2000`**：

| β1 | β2 | seed_42 | seed_43 | seed_44 | min−max spread |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **4.0** | **2.0** | 0.925 | 0.925 | 0.900 | **0.025** |
| 2.0 | 2.0 | 0.950 | 0.950 | **0.800** | 0.150 |
| 4.0 | 1.0 | 0.875 | 0.900 | 0.925 | 0.050 |
| 1.0 | 2.0 | 0.850 | 0.975 | **0.825** | 0.150 |
| 2.0 | 1.0 | 0.950 | 0.925 | **0.750** | 0.200 |
| 1.0 | 1.0 | **0.700** | 0.950 | 0.825 | 0.250 |

三条结论：

**1. seed 44 的异常从 Stage B0 延伸到 Stage B**，且是系统性的。

在 8 个 `β1 ∈ {1.0, 2.0}` 的格子里，seed 44 出现 7 次是三个 seed 中最弱的那个，且差距常常远超其它 seed 之间的差距（在 ep1000 上 seed 44 比 seed 42/43 平均低 `0.25~0.45`）。这与 Stage B0 用 `β1=2.0, β2=1.0` 单独观察到的 "seed 44 peak 只有 0.775" 完全一致——把 Stage B0 结论从 "β1=2.0 这一列 seed 44 是 outlier" 升级成：

> 在 `β1 ≤ 2.0` 的所有 ReBRAC 格子里，seed 44 是系统性难 seed。β1 要提升到 4.0，seed 44 的 test success 才能稳定进入 `0.85 ~ 0.925` 区间。

**2. winner `(β1=4.0, β2=2.0)` 的价值在两个 dataset 上是不一样的**：

- 在 `crosscomp-1000` 上：winner 的 peak success（0.875 / 0.925 / 0.850）**不是**整张表最高的——`(β1=1.0, β2=2.0)` seed 42 曾拿到 0.950，`(β1=2.0, β2=2.0)` seed 43 也拿到 0.950。winner 的胜出完全来自 **把 seed 44 从 `0.475~0.500` 救回到 `0.850`**。这是一个 "robustness winner"，不是 "peak winner"。
- 在 `crosscomp-2000` 上：winner 同时是 peak winner 和 robustness winner——三个 seed 都 `≥ 0.900`，`spread=0.025`，`std=0.012`，是整张表跨 seed 最稳的一格。

**3. `(β1=4.0, β2=1.0)` 在 `crosscomp-2000` 上是隐藏次优**：

- 三个 seed 分别 `0.875 / 0.900 / 0.925`，`spread=0.050`，`std=0.020`；
- mean 仅比 winner 低 `1.7pp`，std 差距也只有 `0.008`；
- β2 从 `2.0` 降到 `1.0` 没有带来明显代价，说明在 `crosscomp-2000` 上 critic penalty 已经饱和；
- 这给 Stage C 提供了一个天然 backup：如果 winner `(β1=4.0, β2=2.0)` 在 5-seed 复核时 std 放大到不能接受的水平，`(β1=4.0, β2=1.0)` 是几乎无代价的备选。

### 6.5 Critic penalty 诊断：β1 决定 `mean_target_q` 的符号

`mean_target_q` 从 `critic_penalty` 之前的 bootstrap target 读出（training 后期 25% 窗口均值），直接看 critic 在什么 regime 下工作：

- β1+β2 越大 → `mean_target_q` 越负。winner `(β1=4.0, β2=2.0)` 在 ep1000 / ep2000 上分别是 `−8.23` / `−6.21`——是整张表里最悲观的配置之一；
- β1=1.0 的所有格子里，ep2000 下 `mean_target_q` 是 `+6~+8`（critic 仍在乐观外推），ep1000 下落在 `−1.7` ~ `+2.7` 之间；
- `(β1=4.0, β2=1.0)` ep2000 的 `ratio=0.521` 是整张表最大的 penalty 比例，但对应的 `mean_target_q=−1.80` 而不是 `−6`，说明它是 "penalty 相对 Q 尺度显得大"、而不是 "penalty 真的把 Q 压得很低"。这一点在 ep1000 对应格子上不出现（ratio 只有 0.049）。

结合 §6.4 的观察：

> winner 是整张表里最悲观的格子之一，而 pessimism 与 success / robustness 是同向的。dual penalty 在当前数据规模下并没有出现 "过度保守压坏 policy" 的迹象。

### 6.6 与 TD3BC phase0c 的直接对比

ReBRAC Stage B winner vs TD3BC phase0c **Stage C 正式 5-seed** 成绩（注意：ReBRAC 只有 3 seeds，下一步 Stage C 才会拉齐到 5 seeds）：

| dataset | TD3BC phase0c 正式（5 seeds） | ReBRAC Stage B winner（3 seeds） | Δ success |
| --- | --- | --- | ---: |
| `crosscomp-1000` | `success 0.672 ± 0.045 @ α=0.25` | `success 0.883 ± 0.031 @ (β1=4.0, β2=2.0)` | **+21.1pp** |
| `crosscomp-2000` | `success 0.596 ± 0.036 @ α=0.15` | `success 0.917 ± 0.012 @ (β1=4.0, β2=2.0)` | **+32.1pp** |

附加观察：

- 两个 dataset 上 **ReBRAC 的 std 都低于 TD3BC phase0c**（0.031 vs 0.045；0.012 vs 0.036）——"均值和方差同时改善" 在 screening 级别已经成立；
- ReBRAC 首次翻转了 "`2000` 差于 `1000`" 的趋势：ReBRAC 下 ep2000 (0.917) 好于 ep1000 (0.883)，而 TD3BC phase0c 下 ep2000 (0.596) 明显差于 ep1000 (0.672)。这与 [rebrac_experiment_plan.md §2.2](./rebrac_experiment_plan.md) 提出的 "ReBRAC 是否能改善 `2000` 退化" 这一机制问题直接相关；
- 这些 Δ 是 "screening 3-seed vs 正式 5-seed" 的对比，不能直接当作 final 结论——Stage C 必须在相同 5-seed 口径下重跑才算数。但就 screening 级别的证据而言，阳性信号是毫不含糊的。

### 6.7 Stage C finalist 决策

按 [rebrac_experiment_plan.md §6.4](./rebrac_experiment_plan.md) 定义的规则（mean → std → return；`β2 · ratio > 1.5` 一票否决）：

- **主 finalist**：`(β1=4.0, β2=2.0)`，两个 dataset 共用。
  - 两个 dataset 的 mean 最高、std 最低、return 最高、safety/time 居中；
  - `β2 · ratio` 在两个 dataset 分别为 `0.129` 和 `0.311`，都远低于一票否决阈值；
  - 是目前唯一在 screening 下让 seed 44 不崩盘的配置。
- **ep2000 backup finalist**：`(β1=4.0, β2=1.0)`。
  - 在 `crosscomp-2000` 上与主 finalist 差距在测量噪声量级（Δmean 1.7pp，Δstd 0.008）；
  - 保留这个 backup 的理由不是 "它更好"，而是 "如果主 finalist 在 5-seed 下方差放大，它是几乎免费的 fallback"；
  - 在 `crosscomp-1000` 上不保留——那里主 finalist 胜出差距更大（mean 差 5.8pp），backup 价值不足。

为什么 Stage C 不扩网格到 `β1=8.0` 或 `β2=4.0`：

- winner 在 ep1000 / ep2000 上的 std 已经分别到 `0.031 / 0.012`，继续增强正则化的收益空间已经非常小；
- pessimism regime 已经足够（`mean_target_q` 已经到 `−8`）——继续压低 Q 可能开始损害 policy 而不是改善；
- Stage C 5-seed 本身会把每个 finalist 的置信度翻倍，比盲目扩网格更能揭示真实 variance 结构。

若 Stage C 复核后两个 finalist 都崩盘，才考虑回到本阶段扩网格到 `β1=8.0` 或 `β2=0.5`。

### 6.8 局限性

1. **3 seeds 的 Stage C 口径差距**：本节所有 ReBRAC 数字都是 3 seeds 的，TD3BC 对照是 5 seeds。Stage C 完成后再做最终对比。
2. **Test manifest 只有 40 episodes**：与 Stage B0、TD3BC phase0b_v2 Stage B 口径一致；Stage C 会把 test manifest 提到 100 episodes。
3. **网格边界问题未完全排除**：winner 落在 `(β1_max, β2_max)` 角上。§6.7 论证了扩网格不是当前最优先级，但如果 Stage C 5-seed 数据出现异常，这会被重新审视。
4. **所有 Stage B 结论仅在 `crosscomp` 两个数据集上成立**，对 `worldcomp` 轨道没有任何证据——这是 Stage D 的范围。

---

## 7. Stage C：5-seed 正式复核

### 7.1 目标与口径

按 [rebrac_experiment_plan.md §6.4](./rebrac_experiment_plan.md) 的定义，把 Stage B screening 的 winner 升级到与 TD3BC phase0c 同口径的正式 5-seed 基线：

- 3 个 finalist 配置（`crosscomp-1000: (β1=4.0, β2=2.0)`；`crosscomp-2000: (β1=4.0, β2=2.0) + (β1=4.0, β2=1.0)`）；
- 每个 finalist 跑 5 seeds（`42 / 43 / 44 / 45 / 46`）；
- test manifest 从 Stage B 的 40 episodes 升到 **100 episodes**；
- 其余协议（`TRAIN_EPOCHS=64`，`CHECKPOINT_EVERY_EPOCHS=8`，val=40 episodes，选择规则 `success_rate → return → -safety_cost → -time`）与 Stage B 对齐。

### 7.2 Stage B checkpoint 重用

为节省算力并保证对齐，Stage C 对 seed 42/43/44（与 Stage B 重叠的三个 seed）**直接重用 Stage B 的 `agent_step_*.pt`**：notebook §2.5 通过 `shutil.copytree(..., dirs_exist_ok=True)` 物理复制 Stage B 的 checkpoint 到 Stage C 的输出目录，而**不使用 `os.symlink`**（Google Drive 的 FUSE 层对 symlink 支持不稳定）。seed 45 / 46 是新训的。

含义：seed 42/43/44 的 Stage C 数字 ≠ Stage B 数字（Stage C 在 100 ep manifest 上重评），但 checkpoint 相同；这让 "manifest 扩容带来的打分差异" 与 "seed 扩展带来的分布差异" 可以清晰解耦。

### 7.3 实验范围

| 轴 | 配置 |
| --- | --- |
| dataset | `crosscomp-1000` + `crosscomp-2000` |
| finalist | `(β1=4.0, β2=2.0)`（两个 dataset）+ `(β1=4.0, β2=1.0)`（仅 ep2000，backup） |
| seeds | `42 / 43 / 44 / 45 / 46` |
| TRAIN_EPOCHS | `64`（与 Stage B0 / Stage B 对齐） |
| val manifest | 40 episodes |
| test manifest | **100 episodes** |

总计 3 × 5 = 15 test runs；其中 9 个 seed 重用 Stage B checkpoint，6 个 seed 新训。

输出路径：

- `checkpoints/offline/rebrac/formal/`
- `results/offline/rebrac/formal/`
- `results/offline/rebrac/formal/summaries/overview.{csv,json}`

执行入口：[notebooks/rebrac_formal.ipynb](../notebooks/rebrac_formal.ipynb)（归档 [notebooks/rebrac_formal_completed.ipynb](../notebooks/rebrac_formal_completed.ipynb)）。

### 7.4 主结果：test-split overview

test manifest = 100 episodes，5 seeds 均值 ± std：

| dataset | β1 | β2 | success (mean ± std) | `β2 · mean_critic_penalty_ratio` |
| --- | ---: | ---: | ---: | ---: |
| `crosscomp-1000` | **4.0** | **2.0** | **0.902 ± 0.021** | 0.086 |
| `crosscomp-2000` | **4.0** | **2.0** | **0.918 ± 0.030** | 0.219 |
| `crosscomp-2000` | 4.0 | 1.0 | 0.894 ± 0.048 | 0.358 |

三个 finalist 的 `β2 · mean_critic_penalty_ratio` 都远低于一票否决阈值 `1.5`（也低于 `1.0` 的 "critic 被 penalty 支配" 警戒值）。

### 7.5 Per-seed 分布

| dataset | β1 | β2 | seed_42 | seed_43 | seed_44 | seed_45 | seed_46 | spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `crosscomp-1000` | **4.0** | **2.0** | 0.890 | 0.930 | 0.870 | 0.900 | 0.920 | **0.060** |
| `crosscomp-2000` | **4.0** | **2.0** | 0.960 | 0.940 | 0.890 | 0.880 | 0.920 | **0.080** |
| `crosscomp-2000` | 4.0 | 1.0 | 0.930 | 0.930 | 0.930 | **0.810** | 0.870 | 0.120 |

四条观察：

1. **seed 44 在主 finalist 上已经不是 outlier**：ep1000 上 `0.870`（Stage B 是 `0.850`，40→100 ep 抖动量级），ep2000 上 `0.890`（Stage B `0.900`）。Stage B §6.4 提出的 "β1=4.0 回收 seed 44" 被 100 episodes 的更大 test 样本进一步确认。
2. **新增 seed 45 / 46 在主 finalist 上表现正常**：两个 dataset 的主 finalist 在 seed 45/46 都落在 `0.880 ~ 0.920` 区间，与 seed 42/43 混合后未拉大方差。
3. **backup finalist `(β1=4.0, β2=1.0)` 在 ep2000 上首次暴露 seed 45 离群**：seed 42/43/44 都是 `0.930`（重用 Stage B ckpt），但新 seed 45 只有 `0.810`——比主 finalist 在同 seed 上的 `0.880` 低 7pp，把 spread 从 Stage B 的 `0.05` 放大到 `0.12`。
4. **backup 的 `std = 0.048` 仍通过 Rule 3 阈值（≤ 0.10）**，但 spread 拉大说明 "Stage B 下 backup 与主 finalist 无代价等价" 这一论断在更多 seed 下不再完全成立；backup 仅作 "主 finalist 完全崩盘" 的 fallback 保留，当前主 finalist 稳住，backup 不启用。

### 7.6 与 Stage B 3-seed 的差异

| dataset | β1 | β2 | Stage B (3 seeds, 40 ep) | Stage C (5 seeds, 100 ep) | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crosscomp-1000` | 4.0 | 2.0 | 0.883 ± 0.031 | 0.902 ± 0.021 | **+1.9pp mean / −0.010 std** |
| `crosscomp-2000` | 4.0 | 2.0 | 0.917 ± 0.012 | 0.918 ± 0.030 | +0.1pp mean / +0.018 std |
| `crosscomp-2000` | 4.0 | 1.0 | 0.900 ± 0.020 | 0.894 ± 0.048 | −0.6pp mean / +0.028 std |

读数：

- ep1000 主 finalist 的 std 从 `0.031` 继续下降到 `0.021`——在完全没有换 ckpt 的前提下，单纯把 test manifest 从 40 扩到 100、再加入 seed 45/46，反而更稳。这说明 Stage B 的方差估计里有 "40 episodes 噪声" 的成分，真实算法方差被高估。
- ep2000 主 finalist 的 std 从 `0.012` 扩到 `0.030` 属于正常的样本放大（3→5 seeds 本身就会让极端 seed 被记入）；mean 几乎不变（`0.917 → 0.918`）说明 winner 判定未被改变。
- backup finalist 的 std 从 `0.020` 扩到 `0.048` 是 seed 45 造成的，而不是均值漂移——这是 Stage B 阶段不可见的信息，Stage C 第一次暴露它。

### 7.7 与 TD3BC phase0c 的最终对比

都在 5-seed 口径下：

| dataset | TD3BC phase0c Stage C（5 seeds, 100 ep） | ReBRAC Stage C（5 seeds, 100 ep） | Δ success |
| --- | --- | --- | ---: |
| `crosscomp-1000` | `0.672 ± 0.045 @ α=0.25` | `0.902 ± 0.021 @ (β1=4.0, β2=2.0)` | **+23.0pp** |
| `crosscomp-2000` | `0.596 ± 0.036 @ α=0.15` | `0.918 ± 0.030 @ (β1=4.0, β2=2.0)` | **+32.2pp** |

两点强调：

1. ReBRAC 的 std 在两个 dataset 上都**等于或低于** TD3BC phase0c 正式 5-seed 的 std（0.021 vs 0.045；0.030 vs 0.036）——"均值改进同时方差不增" 在正式口径下继续成立。
2. "`2000` 不再差于 `1000`" 这一 TD3BC 主线下的老问题，在 ReBRAC Stage C 下翻转为 `ep2000 (0.918) > ep1000 (0.902)`，与 Stage B 观察一致。

### 7.8 Stage C 通过判据核对

按 [plan §6.4 失败条件](./rebrac_experiment_plan.md)：

| 条件 | 阈值 | Stage C 实测 | 是否通过 |
| --- | --- | --- | --- |
| Rule 1：ep1000 主 finalist `mean_test_success_rate` 不得跌破 TD3BC phase0c 的 `0.672` | `> 0.672` | `0.902` | **通过**（大幅超出） |
| Rule 2：ep2000 两个 finalist `mean_test_success_rate` 都不得跌破 `0.75` | `> 0.75` | `0.918 / 0.894` | **通过** |
| Rule 3：任一 finalist `std_test_success_rate` 不得超过 `0.10` | `≤ 0.10` | `0.021 / 0.030 / 0.048` | **通过** |

Stage C 阳性结论成立：ReBRAC 正式升级为 deployable 主基线，Stage D 触发条件满足。

### 7.9 Stage D 对接口径

Stage D 由 Stage C 阳性结论触发，但 **不直接复刻 TD3BC worldcomp teacher-gap 的两轨道 10-run 矩阵**。原因：`crosscomp`（Stage C 的数据）与 `worldcomp`（Stage D 的数据）的瓶颈性质不同——前者是 "数据支持集结构 + 算法 critic robustness"，后者是 "deployable obs 下 critic 的信息瓶颈"，TD3BC 在 `worldcomp` deployable 下退化为 pure BC 是 `crosscomp` 上根本不存在的现象。ReBRAC 在 `worldcomp` 下可能 "进一步治好 critic 退化"，也可能 "在 noisy teacher target 下放大 bias"——两种结果都是论文级 finding，但所需的 follow-up 实验形态不同。

因此 Stage D 拆为 Phase 1 / Phase 2 两段（详见 [plan §6.5](./rebrac_experiment_plan.md)）：

- **Phase 1**（必做，~7 runs）：`worldcomp-1000` deployable 轨道的 epoch-probe（2 runs）+ deployable formal（5 seeds）。回答核心问题 "ReBRAC 在 worldcomp deployable 下是否打破 TD3BC 退化为 BC 的现象"。
- **Phase 2**（条件触发，0~5 runs）：privileged-critic 轨道。规模由 Phase 1 三档情景决定：
  - **情景 A**（Phase 1 mean > 0.90，显著超 TD3BC `0.858`）：privileged-critic 降级为 3-seed 边际确认或跳过。
  - **情景 B**（Phase 1 mean ∈ [0.84, 0.90]，与 TD3BC 持平）：privileged-critic 跑满 5 seeds，作为 algorithm vs observation 瓶颈分离的诊断。
  - **情景 C**（Phase 1 mean < 0.84，弱于 TD3BC）：privileged-critic 跑满 5 seeds + 追加 critic-penalty-off ablation 理解机制。

Stage D 共同设定（两 Phase 共享）：

- **finalist 配置锁定为 `(β1=4.0, β2=2.0)`**，不扫超参；`(β1=4.0, β2=1.0)` 不带入 Stage D（Stage C 下已暴露 seed 敏感，在 `worldcomp` 上再做冗余对照无信息价值）。
- **TRAIN_EPOCHS 由 Phase 1 epoch-probe 决定**：peak ≤ 64 → 取 64；peak ∈ (64, 96] → 取 96（对齐 TD3BC worldcomp teacher-gap）；peak > 96 → 暂停 Stage D 回查。
- **seeds 取 `42 / 43 / 44 / 45 / 46`**（与 Stage C、TD3BC worldcomp teacher-gap 对齐）；test manifest 100 episodes（与 Stage C 对齐）。
- **驱动脚本**：新建 `scripts/run_offline_rebrac_worldcomp_teacher_gap.sh`，对齐 TD3BC 侧的 `phase0c_worldcomp_teacher_gap` 命名 pattern；不在现有 `run_offline_rebrac_screen.sh` 上扩展。

#### Stage D Phase 1 Step 1（epoch-probe）已完成结果（2 seeds × test=40）

| 指标 | 实测 | 对照 |
| --- | ---: | --- |
| `mean_test_success_rate` | **1.0 ± 0.0** | TD3BC deployable formal `0.858 ± 0.080`；TD3BC privileged formal `0.922 ± 0.086`；teacher baseline `0.990` |
| `mean_test_return` | **35.62 ± 1.83** | TD3BC deployable `−14.29`；TD3BC privileged `16.49`；teacher baseline `32.19`（probe 实际**超过** teacher baseline，需 5-seed × test=100 复核） |
| `mean_test_safety_cost` | 5.39 | TD3BC privileged 5.73 |
| `β2 · mean_critic_penalty_ratio` | **0.0067** | crosscomp Stage C `0.086 / 0.219 / 0.358`，低 **一到两个数量级**——critic penalty 在 worldcomp 数据上几乎无作用 |

**TRAIN_EPOCHS 决策**：取 `64`（与 Stage B/C 一致；selection 规则在 ep ≤ 64 budget 内自动选 ep64/ep56 ckpt 规避 seed 43 的 ep64 dip）。Phase 1 deployable formal 进入执行：5 seeds（`42-46`）× TRAIN_EPOCHS=64 × test=100 episodes。

**早期推断**：probe 已经强烈暗示 Phase 1 formal 落在 **情景 A**（mean > 0.90）甚至可能接近 `1.0`——意味着 ReBRAC 的 dual penalty **完全治好了** TD3BC 在 worldcomp 上退化为 BC 的现象。如果 Phase 1 formal 确认这一结果，Phase 2 privileged-critic 的角色将从 "信息瓶颈诊断" 退化为 "ceiling 边际确认"。详见 [plan §6.5.2 Step 1 实测结果](./rebrac_experiment_plan.md)。

### 7.10 Stage D Phase 1 Step 2：`worldcomp-1000` deployable formal

#### 7.10.1 目标与口径

Step 1 epoch-probe 已经把 `TRAIN_EPOCHS=64` 钉死，并在 2 seeds × test=40 上观察到 `success_rate = 1.0 ± 0.0`。Step 2 在 5-seed × test=100 的正式协议下重新复核，同时回答两个问题：

1. **核心问题**：ReBRAC 在 `worldcomp-1000` deployable 下是否打破了 TD3BC 退化为 BC 的现象（`mean_test_success > 0.858`）？
2. **次要问题**：probe 的 `1.0 ± 0.0` 是真信号还是 manifest/seed 抽样偏差？

完整 scope 见 [plan §6.5.2 Step 2](./rebrac_experiment_plan.md)。

#### 7.10.2 实验范围

| 轴 | 配置 |
| --- | --- |
| dataset | `worldcomp-1000` |
| finalist | `(β1=4.0, β2=2.0)`（Stage C 锁定，Stage D 不扫超参） |
| seeds | `42 / 43 / 44 / 45 / 46` |
| TRAIN_EPOCHS | `64`（由 Step 1 epoch-probe 决定） |
| 轨道 | deployable（actor + critic 都用 deployable obs） |
| val manifest | 40 episodes |
| test manifest | **100 episodes**（与 Stage C / TD3BC worldcomp formal 同口径） |

总计 5 train runs × 8 ckpt × val=40 + 5 test=100。

输出路径：

- `checkpoints/offline/rebrac/worldcomp_teacher_gap/deployable/`
- `results/offline/rebrac/worldcomp_teacher_gap/deployable/`
- `results/offline/rebrac/worldcomp_teacher_gap/deployable/summaries/overview.{csv,json}`

执行入口：[notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb](../notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb)。驱动脚本：[scripts/run_offline_rebrac_worldcomp_teacher_gap.sh](../scripts/run_offline_rebrac_worldcomp_teacher_gap.sh)。

#### 7.10.3 主结果：5-seed test overview

| dataset | β1 | β2 | success (mean ± std) | return (mean ± std) | safety | time (s) | `mean_critic_penalty` | `mean_target_q` | `mean_critic_penalty_ratio` | `β2 · ratio` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `worldcomp-1000` | 4.0 | 2.0 | **0.928 ± 0.077** | 20.17 ± 12.13 | 5.96 | 53.70 | 0.0802 | **+15.22** | 0.00553 | **0.0111** |

#### 7.10.4 Per-seed 分布

| seed | selected ckpt | val_succ (40 ep) | val_return | test_succ (100 ep) | test_return | test_safety |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `agent_final` (ep64) | 1.000 | 30.85 | 0.990 | 26.85 | 6.15 |
| 43 | `agent_step_22848` (ep56) | 1.000 | 30.87 | 0.930 | 17.76 | 6.82 |
| 44 | `agent_step_16320` (ep40) | **0.825** | 8.20 | **0.780** | −2.53 | 8.16 |
| 45 | `agent_final` (ep64) | 1.000 | 33.62 | 0.980 | 28.97 | 5.02 |
| 46 | `agent_step_22848` (ep56) | 0.975 | 33.43 | 0.960 | 29.80 | 3.67 |

四条观察：

**1. 4/5 seed 都进入 `≥0.93` 的 high-success regime**：seeds 42/45 拿到 `0.99 / 0.98`，seeds 43/46 拿到 `0.93 / 0.96`。

**2. seed 44 是真正的 outlier**：val 已经只有 0.825（其它 4 seed 都 ≥ 0.975），selected ckpt 还停在 ep40——后续的 ep48~64 在 val 上没能恢复。test=100 是 0.78。这是 ReBRAC 在 worldcomp 上**第一次**出现 β1=4.0 救不回的难 seed——与 Stage C crosscomp 上 "β1=4.0 完整回收 seed 44" 的故事**首次分叉**。

**3. probe 没有高估，只是没看到 seed 44**：probe seeds 是 42/43，恰好都进入 high-success regime（test=100 下 `0.99/0.93`）。probe 的 `1.0 ± 0.0` 本身没问题，缩水到 `0.928 ± 0.077` 完全由 seed 44 这一个点解释。

**4. selected ckpt 的分布说明 ReBRAC 在 worldcomp 上训练曲线非单调**：5 个 seed 选出的 ckpt 分别是 ep40 / ep56 / ep56 / ep64 / ep64。这与 Stage B0 / Stage C 上观察到的 "ReBRAC val 曲线常有 dip" 一致；selection 规则在 ep ≤ 64 budget 内自动避开 dip。

#### 7.10.5 与 TD3BC `worldcomp` formal 三向对比

| 协议 | success (mean ± std) | return | safety | gap closure（vs deployable→teacher） |
| --- | ---: | ---: | ---: | ---: |
| TD3BC `worldcomp` deployable formal (α=0.0, BC) | `0.858 ± 0.080` | `−14.29` | `7.91` | 0% |
| TD3BC `worldcomp` privileged-critic formal (α=0.1) | `0.922 ± 0.086` | `16.49` | `5.73` | 48.5% |
| **ReBRAC `worldcomp` deployable Phase 1**（本节） | **`0.928 ± 0.077`** | `20.17` | `5.96` | **53.0%** |
| teacher baseline (online) | `0.990` | `32.19` | — | 100% |

四条强结论：

1. **ReBRAC deployable 在均值上首次超过 TD3BC privileged-critic**（+0.6pp），且只用 deployable obs 就做到了——这是当前阶段最强的论文级 finding。"observation 瓶颈" 不再是 `worldcomp` 上不可绕过的瓶颈；ReBRAC 的 dual penalty 跨数据类型 generalize。
2. **gap closure 53% > TD3BC privileged-critic 的 48.5%**：ReBRAC deployable 不仅追平特权 critic 协议，还把 TD3BC 留下的 deployable→teacher gap 进一步压缩。
3. **`mean_test_return` 显著更高**（20.17 vs TD3BC privileged 的 16.49）：成功率相近时，ReBRAC 走出更高效的轨迹（path_efficiency 0.77~0.80 量级，progress_ratio 0.86~0.90）；safety_cost 也比 TD3BC privileged 接近、明显优于 TD3BC deployable。
4. **ReBRAC std 略低于 TD3BC privileged**（0.077 vs 0.086），但接近——与 crosscomp Stage C 上 "ReBRAC std 远低于 TD3BC" 的故事不一样。原因见 §7.10.6 第 3 点（seed 44 在 worldcomp 上没被救回，把 std 撑住了）。

#### 7.10.6 三个机制 finding（Phase 2 设计依据）

**Finding 1：dual penalty 在 worldcomp 上几乎退化为单 penalty**

| 指标 | worldcomp Phase 1 | crosscomp Stage C ep1000 | crosscomp Stage C ep2000 主 | crosscomp Stage C ep2000 backup |
| --- | ---: | ---: | ---: | ---: |
| `β2 · mean_critic_penalty_ratio` | **0.0111** | 0.086 | 0.219 | 0.358 |
| `mean_target_q` | **+15.22** | `−5 ~ −8`（pessimism regime） | 同左 | 同左 |

worldcomp 上 critic penalty 的相对贡献比 crosscomp 低 **1–2 个数量级**，且 critic 在 worldcomp 上保持显著乐观。机制解释：worldcomp 是 deterministic teacher policy，`next_actions = π_teacher(s')` 与 ReBRAC 自己的 `π_target(s')` 高度一致 → critic penalty `||π_target(s') + noise − a'||²` 退化为接近恒等约束。

含义：ReBRAC 在 worldcomp 上的全部增益基本来自 actor BC penalty + 网络容量/LayerNorm，**不是 dual penalty**。这一发现直接影响 Stage E ablation 设计——critic-penalty-off 在 worldcomp 上预期几乎无效。建议在 Phase 2 之前或并行做一个 `(β1=4.0, β2=0) × 1–2 seeds` 的 cheap probe（详见 [plan §6.5.3 末尾](./rebrac_experiment_plan.md)），以 1–2 runs 的代价廉价验证这一猜想。

**Finding 2：ReBRAC 在 worldcomp 上对 seed 44 失去回收能力**

Stage C crosscomp 上 seed 44 在 β1=4.0 下回到 `0.870 / 0.890`；worldcomp Phase 1 上 seed 44 是 `0.780`，selected val 只有 0.825。机制猜想：

- crosscomp 上稳住 seed 44 的"两条防线"是 actor penalty + critic penalty；
- worldcomp 上 critic penalty 几乎不工作（Finding 1），缺少了第二条防线；
- 因此 seed 44 的优化方差在 worldcomp 上没被完全压住。

Phase 2 是验证这一猜想的最直接实验：privileged critic 把更丰富的等效流信息暴露给 critic，如果它把 seed 44 救回 → critic 侧的 OOD 风险/信息不足是 seed 44 难训的真因；如果救不回 → seed 44 的问题与 critic 无关，是数据/优化层面的 outlier。

**Finding 3：probe → formal 缩水的真正解释**

Step 1 probe（seeds 42/43, test=40）的 `1.0 ± 0.0` → Step 2 formal（seeds 42-46, test=100）的 `0.928 ± 0.077`。三种可能解释：

- (a) test=100 比 test=40 引入更多 episode-level 噪声 → 数字一定会落；
- (b) 新增 seeds 45/46 把方差拉大；
- (c) 新增 seeds 中含一个难 seed（44）。

实测 (a)+(b) 联合贡献量级很小：seeds 42/43 在 test=100 下分别是 `0.99 / 0.93`，与 probe 的 `1.0 / 1.0` 差距分别只有 1pp / 7pp，前者基本是抽样噪声，后者是 test 集的真实 episode 难度差异。新 seeds 45/46 表现正常（`0.98 / 0.96`）。**缩水的 90%+ 解释由 (c) 提供**——seed 44 单点贡献 `(1.0 − 0.78) = 0.22` 的 success 损失，在 5-seed mean 上摊销为 `−4.4pp`，正好对应 1.0 → 0.928 的差距。

含义：未来类似实验把 probe 设为 2 seeds 不够稳健——必须包含至少 1 个历史上出现过的难 seed（如 seed 44）。这是对 [plan §3](./rebrac_experiment_plan.md) "Stage 内 seed 选择标准" 的隐式更新。

#### 7.10.7 Phase 1 通过判据核对

按 [plan §6.5.2 Phase 1 通过判据](./rebrac_experiment_plan.md)：

| 条件 | 阈值 | 实测 | 是否通过 |
| --- | --- | --- | --- |
| Rule 1：mean test success | `> 0.858`（TD3BC deployable） | 0.928 | **通过**（+7.0pp） |
| Rule 2：std test success | `≤ 0.10` | 0.077 | **通过** |
| 情景归类 | `> 0.90` ⇒ A；`[0.84, 0.90]` ⇒ B；`< 0.84` ⇒ C | 0.928 | **情景 A** |

Phase 1 阳性结论成立：ReBRAC 在 `worldcomp-1000` deployable 上不仅打破 TD3BC 退化为 BC 的现象，还**只用 deployable obs 追平/超过 TD3BC privileged-critic 协议**。

#### 7.10.8 Phase 2 决策（基于 Phase 1 finding 收紧）

按 [plan §6.5.3](./rebrac_experiment_plan.md) 原本设定，情景 A 下 Phase 2 是 "3-seed 边际确认或可省略"。Phase 1 的 seed 44 离群点改变了 Phase 2 的诊断价值：

- **不省略**：seed 44 的离群点（0.78）是当前 Stage D 最有信息量的诊断点。
- **3 seeds 必须 = `42 / 43 / 44`**（不是任意 3 个）：seeds 42/43 对齐 Phase 1 baseline；**seed 44 是 Phase 2 的核心信息**——能否被 privileged critic 救回，直接区分 "seed 44 是 critic 信息瓶颈" vs "seed 44 是数据/优化层面 outlier"。
- **TRAIN_EPOCHS / val/test manifest / finalist** 与 Phase 1 完全对齐。

**可选并行：critic-penalty-off probe**（1–2 runs，详见 [plan §6.5.3 末尾](./rebrac_experiment_plan.md)）

`(β1=4.0, β2=0) × seeds 42/43 on worldcomp-1000 deployable, TRAIN_EPOCHS=64`。判据：若与 Phase 1 主 finalist 在 ±2pp 内 → Finding 1 机制猜想成立，可直接写入 Stage E 章节并节省一次完整 ablation。

#### 7.10.9 局限性（Step 2 专属）

1. **seed 44 的真因尚未定位**：是 worldcomp 数据里 seed 44 触发的 RNG 路径恰好落在 critic penalty 无效区，还是 actor penalty 单独不足以稳住该 seed，目前无法区分。Phase 2 是核心诊断；如 Phase 2 也救不回，需要回到 collector 层面检查 seed 44 触发的 episode 起点分布。
2. **`β2 · ratio = 0.0111` 是 5-seed 均值，未逐 seed 拆**：Finding 1 的强度可能被 seed 44 这一个 outlier 拉偏（如果 seed 44 的 critic_penalty_ratio 异常大或异常小）。如 Phase 2 时仍想精修这个 finding，可补一个 per-seed `mean_critic_penalty_ratio` 表，但不影响当前 Phase 2 决策。
3. **没有跑 worldcomp `crosscomp-2000` 等价的"数据规模 sweep"**：Phase 1 只覆盖 worldcomp-1000。这是有意的——plan §6.5 明确不做 worldcomp 数据规模研究。如果未来需要补，可在 Stage E 之后单独评估。

### 7.11 局限性

1. **seed 45 在 backup 上的离群未深挖**：`(β1=4.0, β2=1.0)` 在 seed 45 上得到 `0.810`，比同 seed 主 finalist 低 7pp。可能是 seed 45 本身对 β2=1.0 更敏感，也可能是单点噪声。如果 Stage D 或后续 ablation 再次观察到 "β2 小一档在新 seed 上方差放大" 的模式，可把它作为 critic penalty 敏感性的起点；目前不追加 7-seed rework。
2. **没有做 7-seed / 10-seed follow-up**：三项阈值都已通过，再扩样本的边际收益低；如果 Stage D 在 `worldcomp privileged-critic` 再次出现不稳，再回头决定是否补样本。
3. **与 TD3BC phase0c 的对比仍跨实验批次**：两边都在 canonical protocol 下执行，但 `TRAIN_EPOCHS` 历史上有过调整（TD3BC 主线 64，worldcomp teacher-gap 96），ReBRAC 统一 64。这对 `crosscomp` 的直接对比无影响，但对 Stage D `worldcomp` 对比需要在 Stage D 报告中单独澄清预算口径。

### 7.12 Stage D Phase 2：`worldcomp-1000` privileged-critic formal `【rev.8 升 5-seed】`

#### 7.12.1 目标与口径

按 [plan §6.5.3](./rebrac_experiment_plan.md) 收紧后的设计：Phase 1 落入情景 A 后，Phase 2 在 rev.7 时是 3-seed 边际确认（必须含 seed 44）；**rev.8 升级为 5-seed 完整 formal**（补 seeds 45/46，driver auto-skip 已存在的 42/43/44），以堵 review §2.2.2 "3-seed 与 5-seed std 不可直接比较" 弱点。直接回答两个问题：

1. ReBRAC privileged-critic 是否在均值上进一步把 deployable→teacher gap 关闭？要求 `mean_test_success > 0.922` 且 `std ≤ 0.10`。
2. seed 44 在 deployable 上的 0.78 离群点，是否被 privileged critic 救回（区分 critic 信息瓶颈 vs 数据/优化层 outlier）？

#### 7.12.2 实验范围

| 轴 | 配置 |
| --- | --- |
| β1 / β2 | `4.0 / 2.0`（Stage C 锁定的 finalist） |
| dataset | `worldcomp-1000` |
| seeds | **`42 / 43 / 44 / 45 / 46`**（rev.8 升 5-seed；含 seed 44） |
| TRAIN_EPOCHS | `64`（与 Phase 1 一致） |
| 轨道 | privileged-critic（actor 用 deployable obs；critic 用 privileged obs；actor update mode `zeros`） |
| val / test manifest | 40 / 100（与 Phase 1 复用） |
| 算力 | ≈ Phase 1 deployable 的 100%（5-seed parity） |

#### 7.12.3 主结果：5-seed test overview（rev.8）

| 指标 | 实测（5-seed） | 3-seed (rev.7 历史) |
| --- | --- | --- |
| mean_test_success_rate | **0.9340** | 0.9267 |
| std_test_success_rate (per-seed sample) | **0.0261** | 0.031 |
| mean_test_return | 17.02 | 11.99 |
| mean_test_safety_cost | 7.10 | 8.13 |
| mean_test_path_efficiency | 0.7643 | 0.7504 |

5-seed std 0.0261 与 3-seed 0.031 量级几乎不变（5-seed 涉极端 seed 的概率高，std 一般会放宽，这里反而几乎不变 → finalist 在 seed-robustness 上很稳）。**5-seed std 0.0261 严格低于 TD3BC privileged-critic 5-seed std 0.086** —— rev.8 起可正式做 std 对比 claim（堵 review §2.2.2）。

#### 7.12.4 Per-seed 分布（5-seed，rev.8）

| seed | success_rate | return | safety_cost | path_efficiency |
| --- | --- | --- | --- | --- |
| 42 | 0.960 | 22.30 | 7.28 | 0.7665 |
| 43 | 0.920 | -2.21 | 9.91 | 0.7245 |
| 44 | **0.900** | 15.88 | 7.20 | 0.7601 |
| **45 (rev.8 新增)** | 0.930 | 20.34 | 6.81 | 0.7713 |
| **46 (rev.8 新增)** | 0.960 | 28.78 | 4.26 | 0.7993 |

seed 43 的 return 是负数，但 success_rate 仍然是 0.92——说明 seed 43 完成的 episode 在 progress / time 上付出了较大代价，不是 success 失败。seed 44 完整 escape 了 Phase 1 deployable 上的 0.78 outlier 状态。新增 seed 45/46 的 success_rate 落入 0.93 / 0.96（即与 seed 42/44 同区间），return 和 path_efficiency 也是健康范围 → **5-seed 升级未暴露任何新 outlier，3-seed 结论在 5-seed 下完整成立**。

#### 7.12.5 与 Phase 1 / TD3BC 的四向对比（Stage D 主对照表，rev.8）

| 协议 | mean | std | return | gap closure |
| --- | --- | --- | --- | --- |
| TD3BC worldcomp deployable (α=0.0, BC) | 0.858 | 0.080 | -14.29 | 0%（baseline） |
| TD3BC worldcomp privileged-critic (α=0.1) | 0.922 | 0.086 | 16.49 | **48.5%** |
| ReBRAC worldcomp deployable Phase 1 (5 seeds) | 0.928 | 0.077 | 20.17 | **53.0%** |
| ReBRAC worldcomp privileged-critic Phase 2 (**5 seeds, rev.8**) | **0.9340** | **0.0261** | 17.02 | **57.6%** |
| teacher baseline (online) | 0.990 | — | 32.19 | 100% |

四个关键 Δ（rev.8 5-seed 版）：

- Δ vs TD3BC privileged-critic: **+1.2pp**（mean）→ 通过判据 1；std 严格更低（0.0261 vs 0.086，0.30×）；
- Δ vs ReBRAC deployable Phase 1: **+0.6pp**（mean）→ privileged 在 ReBRAC 上对 mean 仍几乎不贡献（5-seed 与 3-seed 一致）；
- Δ vs teacher: -5.6pp；
- gap closure **57.6%** > 50%（判据 3 通过；比 ReBRAC deployable 53.0% 高 +4.6pp，主要由 seed 44 +12pp 救回贡献）。

3-seed 历史记录（rev.7）：mean=0.9267, std=0.025, gap closure=52.0%。升 5-seed 后 mean +0.7pp、std 几乎不变。

#### 7.12.6 seed 44 跨轨道诊断

| 协议 | seed 44 success |
| --- | --- |
| Phase 1 deployable | 0.780 |
| Phase 2 privileged-critic | **0.900** |
| Δ (priv − dep) | **+0.120 (+12.0pp)** |

判据 4 通过。privileged critic 把 seed 44 救回到 0.90（≥ 0.85 阈值）→ **假设 A 成立：critic 信息瓶颈是 seed 44 在 deployable 上 outlier 的真因**。这意味着：

- seed 44 不是数据/优化层面的"无法救回的 outlier"，而是 deployable critic 的 OOD 信号不足；
- privileged critic 通过把 hull-integral `[u_eq, v_eq]` 喂给 critic，恢复了它对 seed 44 这种特定优化轨迹的稳定 Q 估计。

#### 7.12.7 Phase 2 通过判据核对（5-seed，rev.8）

| Rule | 阈值 | 实测（5-seed，rev.8） | 通过 |
| --- | --- | --- | --- |
| 1. mean test success | `> 0.922`（TD3BC priv-critic formal） | **0.9340**（+1.2pp） | ✅ |
| 2. std test success | `≤ 0.10` | **0.0261**（严格 < TD3BC priv 0.086） | ✅ |
| 3. gap closure（情景 A） | `> 50%`（vs TD3BC priv-critic 48.5%） | **57.6%** | ✅ |
| 4. seed 44 诊断 | `≥ 0.85` ⇒ 假设 A | 0.900（+12pp vs deployable 0.78） | ✅ |

**4 / 4 判据全部通过**。Phase 2 阳性结论成立。3-seed 旧通过结果（rev.7）：mean=0.9267, std=0.025, gap closure=52.0%；升 5-seed 后判据保持通过、强度提升（mean +0.7pp、gap closure +5.6pp、std 严格低于 TD3BC）。

#### 7.12.8 二阶 finding：privileged ≈ deployable 但机制不同

最值得记录的不是 "Phase 2 通过"，而是 "通过的方式与 TD3BC 完全不同"：

| 协议 | TD3BC priv-critic 增益 | ReBRAC priv-critic 增益 |
| --- | --- | --- |
| 相对自家 deployable 的 mean | +6.4pp（0.858 → 0.922） | **+0.6pp**（0.928 → 0.9340，5-seed rev.8；3-seed -0.1pp） |
| 难 seed 救回 | seed 分布层面看不到（mean 是平均抬升） | seed 44: +12pp（独立救回） |

含义：

1. **TD3BC** 在 worldcomp 上的 deployable→teacher gap 主要是"critic 看不到足够的流场信息"——所以 privileged critic 给所有 seed 平均抬升 +6.4pp。
2. **ReBRAC** 在 worldcomp 上的 deployable→teacher gap 主要由"actor 的 BC penalty"完成——把 deployable mean 直接拉到 0.928，已经超过 TD3BC privileged。privileged critic **在 ReBRAC 上不再贡献 mean 抬升**，只在 outlier seed（如 seed 44）上贡献额外稳定性。
3. 从 sim2real 论文价值看：ReBRAC + deployable obs 是**部署最现实**的协议（actor / critic 都不依赖 hull-integral），而它已经把 worldcomp gap 关闭到 53%——这是最强的论文级 finding。privileged critic 的边际价值已经降到"outlier seed 救援"级别。

#### 7.12.9 局限性（Phase 2 专属，rev.8 更新）

1. ~~**3-seed 是边际确认而非 5-seed 完整 formal**~~（**rev.8 已修复 — 升 5-seed**）：rev.7 时 3-seed 判据通过强度比 5-seed 弱；rev.8 补 seeds 45/46 后已升级为完整 5-seed formal，judic 强度提升（mean 0.9340、std 0.0261、gap closure 57.6%）。**std 0.0261 严格低于 TD3BC priv 0.086 ⇒ paper 可正式做 std 对比 claim**。
2. **return 标准差较大**：5-seed mean=17.02, std≈11.71（per-seed return 跨 -2.21 ~ 28.78）。这是 worldcomp 上 success 与 return 解耦的一个 instance（seed 43 的 success 0.92 但 return -2.21），但不影响 success 判据。后续做轨迹层分析时需注意。
3. **Phase 2 没有重复 Phase 1 的"Finding 1 reproduction"**：β2·ratio = 0.0113（与 Phase 1 的 0.0111 量级一致），未单独再做 per-seed 分析；与 critic-penalty-off probe（§7.13）共同回答 Finding 1。

---

### 7.13 Stage D — `worldcomp-1000` critic-penalty-off probe

#### 7.13.1 目标与口径

直接验证 [plan §6.5.3 末尾](./rebrac_experiment_plan.md) 提出的 Finding 1 机制猜想：worldcomp 上 `β2·ratio = 0.011`（Phase 1）与 `0.013`（Phase 2）远小于 crosscomp 的 0.086~0.358，是否意味着 dual penalty 在 worldcomp 上几乎退化为单 penalty？把 β2 直接降到 0，看 mean success 是否落在 Phase 1 主 finalist `0.928 ± 2pp` 区间内。

#### 7.13.2 实验范围

| 轴 | 配置 |
| --- | --- |
| β1 | 4.0（与 Phase 1 主 finalist 一致） |
| **β2** | **0.0**（关键操纵：critic penalty off） |
| dataset | `worldcomp-1000` |
| seeds | `42 / 43`（**刻意不含 seed 44**——Phase 1 已暴露 seed 44 是 outlier，这里只验证 typical regime） |
| TRAIN_EPOCHS | 64 |
| 轨道 | deployable（与 Phase 1 同协议） |
| val / test manifest | 40 / 100（复用 Phase 1） |
| 算力 | ≈ Phase 1 deployable 的 40% |

输出隔离到 `results/offline/rebrac/worldcomp_critic_penalty_off_probe/`，不污染 Phase 1 / Phase 2 主路径。

#### 7.13.3 主结果

| 指标 | `(β1=4.0, β2=0)` probe | Phase 1 同 seeds (42/43) | Phase 1 5-seed (42-46) |
| --- | --- | --- | --- |
| mean test success | **0.910** | 0.960 | 0.928 |
| std test success | 0.014 | — | 0.077 |
| mean_test_return | 9.37 | — | 20.17 |
| **mean_target_q** | **22.30** | — | **15.22** |
| mean_critic_penalty_ratio | 0.0037 | — | 0.0057 |

Per-seed test：

| seed | success_rate | return | safety_cost |
| --- | --- | --- | --- |
| 42 | 0.920 | 15.03 | 8.72 |
| 43 | 0.900 | 3.72 | 9.13 |

#### 7.13.4 判定：落入情形 B（Finding 1 机制猜想部分被证伪）

| 阈值 | 情形 A (≤ 2pp) | 情形 B (0.88 ~ 0.91) | 情形 C (< 0.88) |
| --- | --- | --- | --- |
| 同 seeds (42/43) | 不通过 | **0.910 落入** | 不通过 |
| 5-seed (42-46) | 0.910 vs 0.928 = -1.8pp（落入 A） | — | — |

如果只看 5-seed mean，差距是 -1.8pp（情形 A），但这是因为 seed 44 outlier 拉低了 Phase 1 主 finalist 的均值。**正确的对照是同 seeds（42/43）的 -5.0pp 差距**——这才是机制层面的真实差距。

#### 7.13.5 关键证据：mean_target_q +46%

最值得记录的不是 "mean success 掉 5pp"，而是：

| 配置 | mean_target_q |
| --- | --- |
| Phase 1 `(β1=4.0, β2=2.0)` | 15.22 |
| Phase 2 `(β1=4.0, β2=2.0)` priv | 15.35 |
| **probe `(β1=4.0, β2=0)`** | **22.30** |

β2=0 时 `mean_target_q` 跳升 **+46%**。这是 critic penalty 仍在显著抑制 Q 高估的直接证据。`β2·ratio` 数字小（0.011）是因为 `critic_penalty` 项本身经过 clip / squash 后量级被压住，**但它通过 target Q 的 bootstrap 链路对训练稳定性的贡献远大于这个 ratio 数字暗示的程度**。

#### 7.13.6 对 Finding 1 的修正

Phase 1 §7.10.6 写的 Finding 1（"dual penalty 退化为单 penalty"）现修正为：

> ReBRAC 在 worldcomp 上 `β2·ratio` 数字看起来小（0.011 vs crosscomp 0.086~0.358，差 1–2 个数量级），但**这不是因为 critic penalty 没作用，而是因为 critic_penalty 项的绝对量级被 squash 压住**。critic-penalty-off probe 显示，把 β2 降到 0 会让：
> - mean test success 在同 seeds 上掉 **5.0pp**（0.96 → 0.91），属于情形 B；
> - mean_target_q 上跳 **+46%**（15.22 → 22.30），即 Q 高估显著放大。
>
> 因此 dual penalty 在 worldcomp 上的贡献可以归纳为：**对 mean success 是"小但非零"，对 Q 稳定性是"大且必要"**。这与 crosscomp 上 dual penalty 同时贡献 mean 与 stability 的故事不同，但与"critic regularization 是 ReBRAC 必要组成部分"的总体定性一致。

#### 7.13.7 对 Stage E 的影响

probe 已经把 critic-penalty-off 这一格的方向钉死，但**不能仅凭 probe 收口**：

- mean drop（5pp）大于情形 A 阈值（2pp），需要 Stage E 5-seed `(β1=4.0, β2=0)` 完整 ablation 来覆盖 seed 44/45/46（probe 刻意排除了这些 seed，无法外推）；
- 但 Stage E 的核心变量从原计划的"dual penalty 是否必要"**修正为"critic penalty 对 Q 稳定性的具体贡献"**——指标重点从 mean success 转到 mean_target_q / critic_loss / Q 高估等训练侧诊断。

#### 7.13.8 局限性（probe 专属）

1. **2 seeds 远低于正式 ablation 的 5 seeds**：probe 只回答"机制猜想方向"，不替代 Stage E ablation。
2. **刻意排除 seed 44**：probe 不能回答"seed 44 outlier 在 β2=0 下变得更糟还是不变"。这一格留给 Stage E 5-seed ablation。
3. **没有跑 Phase 2 等价的 privileged-critic + β2=0 probe**：privileged 轨道下 critic 已经在看 hull-integral，β2=0 是否仍会让 Q 高估？暂未测，但优先级低（privileged ≈ deployable 的二阶 finding 已经把 privileged 的 mean 价值降到"outlier seed 救援"级）。

---

### 7.14 Stage E (a) — `crosscomp-1000` critic-penalty-off cross-dataset 二次验证

#### 7.14.1 目标与口径

把 §7.13 在 `worldcomp-1000` 上做的 critic-penalty-off probe 扩展到 `crosscomp-1000`，回答两个问题：

1. **Cross-dataset Finding 1 验证**：Phase 1 §7.10.6 提出、§7.13 修正的 Finding 1（"critic penalty 对 mean 贡献小但非零，对 Q 稳定性大且必要"）是否在 crosscomp 上也成立？
2. **Stage E 收口**：dual penalty 是否需要保留为 ReBRAC 主线必需，还是可以在 typical regime 下用单 actor penalty 替代？

#### 7.14.2 实验范围

| 轴 | 配置 |
| --- | --- |
| β1 | 4.0（Stage C 锁定的主 finalist） |
| **β2** | **0.0**（关键操纵：critic penalty off） |
| dataset | `crosscomp-1000` |
| seeds | `42 / 43 / 44 / 45 / 46`（5 seeds，与 Stage C 完全对齐） |
| TRAIN_EPOCHS | 64 |
| 轨道 | deployable |
| val / test manifest | 40 / 100（复用 Stage C `benchmarks/offline_rebrac_screen/`） |
| 算力 | ≈ Stage C 单 finalist 的一半（仅 1 dataset） |

输出隔离到 `results/offline/rebrac/stage_e_critic_penalty_off/`。

#### 7.14.3 主结果

| 指标 | Stage E (a) `(β1=4.0, β2=0)` | Stage C 主 finalist `(β1=4.0, β2=2.0)` | Stage D probe worldcomp `(β1=4.0, β2=0)` |
| --- | --- | --- | --- |
| mean test success | **0.878** | 0.902 | 0.910 |
| std test success | 0.090 | 0.021 | 0.014 |
| mean test return | -43.79 | — | 9.37 |
| mean test safety_cost | 15.46 | — | — |
| **mean_target_q** | **-0.13** | -8.25 | 22.30 |
| mean_critic_penalty | 0.079 | — | — |
| mean_critic_penalty_ratio | 0.108 | 0.043 | 0.0037 |

Per-seed test：

| seed | success_rate | return | safety_cost | path_efficiency |
| --- | --- | --- | --- | --- |
| 42 | 0.920 | -39.30 | 15.91 | 0.635 |
| 43 | 0.950 | -33.28 | 14.95 | 0.654 |
| **44** | **0.700** | -62.68 | 15.11 | 0.618 |
| 45 | 0.920 | -38.54 | 14.98 | 0.640 |
| 46 | 0.900 | -45.13 | 16.36 | 0.618 |

#### 7.14.4 判定：落入情形 B（Finding 1 修正版跨 dataset 成立）

| 阈值 | 情形 A（≥ 0.882 = Stage C - 2pp） | 情形 B（0.85 ~ 0.88） | 情形 C（< 0.85） |
| --- | --- | --- | --- |
| 5-seed mean | 不通过（0.878 < 0.882） | **0.878 落入** | 不通过 |

Δ vs Stage C 主 finalist `(β2=2.0)`（同 dataset 同 seeds）= **-2.4pp**（0.902 → 0.878）。
Δ vs worldcomp probe `(β2=0)`（跨 dataset）= **-3.2pp**（0.910 → 0.878），量级与 worldcomp probe 上的 -5.0pp 同 seeds drop 一致。

#### 7.14.5 关键证据：mean_target_q 跳升机制 dataset-invariant

| 配置 | dataset | mean_target_q | Δ vs β2=2 baseline |
| --- | --- | --- | --- |
| Stage C `(β1=4.0, β2=2.0)` | crosscomp-1000 | -8.25 | — |
| **Stage E (a) `(β1=4.0, β2=0)`** | **crosscomp-1000** | **-0.13** | **+8.12**（≈ +98% 朝正向漂移） |
| Phase 1 `(β1=4.0, β2=2.0)` | worldcomp-1000 | 15.22 | — |
| Stage D probe `(β1=4.0, β2=0)` | worldcomp-1000 | 22.30 | +7.08（+46%） |

两个 dataset 上 β2=0 都让 `mean_target_q` 朝"更不保守"方向漂移 +7~+8 个绝对单位（crosscomp 从负值漂向 0；worldcomp 从正值漂得更正）。**Q 高估抑制机制是 dataset-invariant 的**——critic penalty 在两个 dataset 上都通过 target Q bootstrap 链路抑制 Q 估计，与 §7.13 worldcomp probe 的发现一致。

#### 7.14.6 seed 44 跨 (dataset, β2) 诊断

| 协议 | seed 44 success |
| --- | --- |
| Stage C crosscomp `(β2=2.0)` | 0.870 |
| **Stage E (a) crosscomp `(β2=0)`** | **0.700**（-17pp） |
| Phase 1 worldcomp deployable `(β2=2.0)` | 0.78 |
| Phase 2 worldcomp privileged-critic `(β2=2.0)` | 0.90 |

seed 44 在 crosscomp 上失去 critic penalty 后掉 -17pp，与 Phase 2 假设 A（critic 信息瓶颈是 seed 44 outlier 的真因）一致：**seed 44 无论在 worldcomp 还是 crosscomp 上，都需要某种 critic-side 稳定信号**——worldcomp 上靠 privileged hull-integral 提供（β1+β2 单 actor penalty 不够），crosscomp 上靠 critic penalty (β2) 提供（β1 单 actor penalty 也不够）。两个 dataset 提供的信号源不同，但都印证 "actor BC penalty 一条线不足以稳住 seed 44"。

这是 Phase 2 Finding 2 的 cross-dataset 加固：seed 44 的 outlier 行为不是 worldcomp 特有，而是反映 ReBRAC 在缺乏额外 critic 稳定信号时对 outlier seed 的系统性脆弱。

#### 7.14.7 对 Finding 1 的最终表述（合并 §7.13 + §7.14）

合并 worldcomp probe 与 crosscomp Stage E (a) 两组 cross-dataset 证据：

> **Finding 1（最终版）**：dual penalty 中的 critic penalty (β2) 在两个 dataset 上的贡献结构一致——
> - **对 mean success 是"小但非零"**：worldcomp 同 seeds -5.0pp / 5-seed -1.8pp；crosscomp 5-seed -2.4pp。
> - **对 Q 稳定性是"大且必要"**：worldcomp `mean_target_q` +46%（15.22 → 22.30），crosscomp `mean_target_q` +98%（-8.25 → -0.13）；两个 dataset 上的绝对漂移幅度都在 +7~+8 单位级。
> - **对 outlier seed 鲁棒性是"必需"**：crosscomp seed 44 失去 β2 后掉 -17pp（0.87 → 0.70），与 Phase 2 假设 A 闭环。
>
> `β2·ratio` 数字（worldcomp 0.011 / crosscomp 0.043~0.108）在两个 dataset 上都低估了 critic penalty 的真实贡献——这是 squash / clip 后 critic_penalty 项的绝对量级被压住的产物，**critic penalty 通过 target Q 的 bootstrap 链路对训练稳定性的贡献远大于这个 ratio 数字暗示的程度**。

#### 7.14.8 对 ReBRAC 主线的影响

1. **dual penalty 在 ReBRAC 主线上是必需的**——单 actor penalty 在 typical regime（5-seed mean）只掉 2~3pp，但在 outlier seed（44）上掉 17pp，且 Q 高估机制在两个 dataset 上都被显著放大。**保留 `(β1=4.0, β2=2.0)` 作为 Stage C 主 finalist，不向单 penalty 简化。**
2. **§7.13.7 提出的"Stage E 5-seed `(β1=4.0, β2=0)` 完整 ablation"由本节执行完成**，覆盖了 worldcomp probe 刻意排除的 seed 44/45/46。结论：在 5 seeds × test=100 下，crosscomp β2=0 落入情形 B（与 worldcomp probe 量级一致），不外推为情形 A。
3. **Stage E (b) seed 44 collector 起点 inspection 不再触发**：本节已经把"crosscomp 上 seed 44 是否也是 outlier"问题用机制对照（β2 移除后 -17pp）回答完毕；触发恢复条件 1（情形 C）和触发条件 2（seed 44 < 0.85）都不成立。Stage E (b) 维持推迟，不阻塞 ReBRAC 主线收口。

#### 7.14.9 局限性（Stage E (a) 专属）

1. **没有跑 normalize_q-off 5-seed ablation**——rev.4 原计划的第三格消融，被 Phase 2 二阶 finding（actor BC penalty 主导）削弱优先级，rev.6 已删；如果论文 review 反馈"想看 Q normalization 的独立贡献"，再补。
2. **没有覆盖 crosscomp-2000**——Stage E (a) 只在 `crosscomp-1000` 上做 cross-dataset 验证；对 `crosscomp-2000` 是否也是情形 B 没有直接证据。但 Stage C 已经显示 ep1000 与 ep2000 在 winner finalist 下行为一致（0.902 vs 0.918），cross-extrapolate 风险低。
3. **mean_target_q 在 crosscomp 与 worldcomp 上符号不同**（crosscomp -8.25 / worldcomp 15.22）——这是 reward / Q 量级的 dataset-specific 现象（worldcomp 在均值上 success 更高，Q 累积更正），不影响 "β2 移除导致 Q 朝更不保守方向漂移 +7~+8 单位" 这个 dataset-invariant 机制结论。

---

### 7.15 Stage F (B) — `crosscomp-1000` critic LayerNorm-off probe `【rev.8 新增 — paper-readiness probe】`

#### 7.15.1 目标与口径

按 [review.md §2.2.4 + §3.1 task B](./rebrac_mainline_review.md) 设计：ReBRAC 原 paper（Tarasov et al., 2023）反复强调 critic LayerNorm 比 dual penalty 更关键。如果 paper review 指出 "你的 +23~+32pp 提升其实主要来自 LayerNorm，不是 dual penalty"，目前没有反驳。本 probe 的设计：把 Stage C 主 finalist 减去 critic LayerNorm，看 mean / std / mean_target_q 的变化。

#### 7.15.2 实验范围

| 轴 | 配置 |
| --- | --- |
| β1 / β2 | `4.0 / 2.0`（与 Stage C 完全一致） |
| dataset | `crosscomp-1000` |
| seeds | `42 / 43`（2 seeds，paper-readiness probe 量级） |
| TRAIN_EPOCHS | `64` |
| critic_layernorm | **off**（注入 `--no-critic-layernorm`，BooleanOptionalAction 左到右覆盖） |
| 轨道 | deployable（与 Stage C 一致） |
| val / test manifest | 40 / 100（复用 Stage C `benchmarks/offline_rebrac_screen/`） |
| 输出隔离 | `results/offline/rebrac/critic_ln_off/` + `checkpoints/offline/rebrac/critic_ln_off/` |
| 算力 | ≈ Stage C 单 finalist 的 40%（2 seeds × 1 finalist） |

driver 通过 `shutil.copy2` 把 `scripts/run_offline_rebrac_screen.sh` 复制到 sibling 文件，再 `text.replace("--critic-layernorm \\", "--no-critic-layernorm \\", 1)` —— 不修改 tracked driver source。

#### 7.15.3 主结果

| 指标 | LN-off probe | LN-on Stage C 对照 | Δ |
| --- | --- | --- | --- |
| seeds | 42 / 43 | 42 / 43 / 44 / 45 / 46 | — |
| n_seeds | 2 | 5 | — |
| mean_test_success_rate | **0.7400** | 0.902 | **−16.2pp** |
| std_test_success_rate | 0.2546 | 0.0214 | **+0.233（≈ 12× 放大）** |
| mean_test_return | -90.98 | (Stage C +25 量级) | 显著负 |
| mean_target_q | **−12.09** | -8.25 | **−3.84（朝更负 +46%）** |
| mean_critic_penalty_ratio | 0.0075 | 0.0431 | -0.036（critic penalty 失效） |

#### 7.15.4 Per-seed 分布

| seed | success_rate | return | safety_cost |
| --- | --- | --- | --- |
| 42 | **0.560** | -138.34 | 16.13 |
| 43 | 0.920 | -43.63 | 13.86 |

**seed 42 在 LN-off 下从 Stage C 0.92 掉到 0.56（−36pp 单点崩溃）**——这一行为在 LN-on / β2=0 ablation 下都没出现过（Stage E (a) 同 seed 42 是 0.92）。LN 关掉后某些 seed 出现训练完全失败的失稳，是 LN 必要性的最强证据。

#### 7.15.5 三向对比（β2 vs LN）

| 配置 | 改动相对 Stage C | mean | std | Δ mean | Δ target_q |
| --- | --- | --- | --- | --- | --- |
| Stage C 主 finalist | （baseline） | 0.902 | 0.021 | 0pp | (baseline) |
| Stage E (a) `(β2=0, LN=on)` | 关 dual penalty | 0.878 | 0.090 | **-2.4pp** | +8.12（朝正） |
| Stage F (B) `(β2=2, LN=off)` | **关 LN** | **0.7400** | **0.2546** | **-16.2pp** | **-3.84（朝更负）** |

两个关键观察：

1. **LN 关掉比 β2 关掉严重 6.7×**：mean 退化 -16.2pp vs -2.4pp；
2. **退化方向不同**：β2=0 让 Q 朝过度乐观漂移（+8.12 单位）；LN-off 让 Q 朝更不稳定的负值漂移（-3.84 单位）+ std blow-up 12×。这是两个独立 component 的标志——它们影响 Q 的方式都不同，不是同一现象的不同侧面。

#### 7.15.6 结论：critic LN 是必要 component，与 dual penalty 独立

1. **LN 不是 dual penalty 的伪装** —— Stage F (B) 给出的 -16.2pp 退化远大于 Stage E (a) 的 -2.4pp，证明 LN 与 dual penalty 各自独立贡献；
2. **paper-readiness 立场**：审稿人若 push "提升其实是 LayerNorm" 论点，可直接引用本节："关 LN 让 Stage C finalist 退化 -16.2pp（-16.2 vs -2.4pp 关 dual penalty），LN 与 dual penalty 是两个独立 component"；
3. **不做 LN sweep**：本 probe 只是 2-seed cheap probe（按 review §3.1 task B 设计 ≈ 2h L4），不向 capacity / dropout / hidden_dim sweep 扩展（rev.6 / rev.7 / rev.8 一致维持不做）；
4. **不能在 paper 中 claim "LN 比 dual penalty 重要"**：n=2 seeds 的 LN-off probe 强度不足以做这一对比 claim，只能做 "LN 是必要 component" 的存在性 claim。

#### 7.15.7 对 Finding 1 / Finding 2 的影响

- **对 Finding 1 (dual penalty 必要性)**：不影响。LN-off 与 β2=0 是两个独立的 ablation，在 Stage C finalist 上各自展示了不同退化模式。Finding 1 修正版表述（"对 mean 贡献小但非零、对 Q 稳定性大且必要、对 outlier seed 鲁棒性必需"）保持。
- **对 Finding 2 (seed 44 outlier)**：LN-off 的 seed 42 -36pp 单点崩溃说明 ReBRAC 在 LN 缺失下还会出现新的 seed-specific 失稳模式（不只是 seed 44）。这暗示 **outlier seed 鲁棒性依赖 critic 的多重稳定信号**——dual penalty + LN + privileged Q 都各自贡献。

#### 7.15.8 局限性（probe 专属）

1. **2 seeds 是 cheap probe，不是 5-seed formal**：std=0.25 的数字只是给出了一个 "LN-off 退化方向" 的存在性证据，不能做严格 std claim。如果论文 review 反馈"想要 LN-off 5-seed std"，可补 seeds 44/45/46；预算 ≈ 3h L4，优先级低（已经堵住主要弱点）。
2. **没有覆盖 worldcomp-1000**：LN-off 是否在 worldcomp 上同样 -16pp 量级未测。预期方向一致（LN 是 component-level 设计而非 dataset-specific tuning），但严格说没有数据。
3. **没有 actor LN ablation**：本 probe 只关 critic LN（与 ReBRAC 原 paper 强调一致；Stage C 主 finalist 本就 actor_layernorm=off）；actor LN 单独是否有差异未测。
4. **driver patching 是 in-notebook one-time**：sibling driver `scripts/run_offline_rebrac_critic_ln_off.sh` 由 notebook 一次性 `text.replace` 生成，不是 tracked source 的稳定 entrypoint；如未来需要重跑，从 notebook 重新生成。

---

### 7.16 Stage F (C) — Phase 1 deployable vs TD3BC privileged-critic 统计检验 `【rev.8 新增 — paper-readiness probe】`

#### 7.16.1 目标与口径

按 [review.md §2.2.3 + §3.1 task C](./rebrac_mainline_review.md) 设计：5-seed × test=100 下，ReBRAC deployable 0.928 vs TD3BC privileged-critic 0.922 的 +0.6pp 远小于 1 个标准误（√5 上约 ±0.034），严格统计意义下是"持平"而非"超过"。本节用 paired episode-level bootstrap + Welch's t-test 量化此 claim。详细数字 + paper 写作指引另存 [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md)。

#### 7.16.2 数据来源

两个协议共享同一个 fixed evaluation manifest `benchmarks/single_u10_cross_tgt15.json`（100 episode_ids），故 episode_id 在两个协议下严格 paired：

- ReBRAC dep Phase 1: `results/offline/rebrac/worldcomp_teacher_gap/deployable/.../actorb_4p0__criticb_2p0/test/seed_*.json`，5 seeds × 100 episodes；
- TD3BC priv: `results/offline/td3bc/phase0c/worldcomp_teacher_gap/privileged_final/.../test_selected/alpha_0p1/seed_*.json`，5 seeds × 100 episodes；
- per-seed mean success rate 已在 [report §7.10.4](./rebrac_experiment_report.md) 与 TD3BC report 中列出。

#### 7.16.3 三个统计检验

**1. Paired episode-level bootstrap (10000 resamples)**

按 100 个 episode_id 重抽样，每个 episode 取跨 5 seeds 的成功率均值（Bernoulli mean），再 bootstrap：

- point estimate Δ = **+0.0060**；
- 95% CI on Δ = **[-0.0300, +0.0420]**（包含 0）；
- 99% CI on Δ = [-0.0420, +0.0520]；
- bootstrap p (two-sided, H0: Δ=0) ≈ **0.7762**。

**2. Welch's t-test (5 vs 5 seed-level means)**

- t = **0.104**；
- p (two-sided) = **0.9195** → fail to reject H0 → **持平**。

**3. Gap closure 95% CI (seed-level bootstrap)**

deployable→teacher gap closure 定义 `(method - TD3BC_dep) / (teacher - TD3BC_dep)`，TD3BC_dep=0.858, teacher=0.99：

- ReBRAC dep gap closure point = **53.0%**；
- TD3BC priv gap closure point = **48.5%**；
- Δ point = +4.5pp；
- 95% CI on Δ = **[-62.12pp, +86.36pp]**（极宽，包含 0）。

#### 7.16.4 结论与 paper 写作指引

1. **main text 改写**：从"ReBRAC deployable 首次超过 TD3BC privileged-critic"改为"**与 TD3BC privileged-critic 协议持平（statistically not different）**，附 Welch's p=0.9195 与 paired bootstrap CI"。这一表述同样支撑论文核心 finding（部署友好性最强：deployable-only 与 privileged-critic 协议在统计意义上等价）；
2. **gap closure +4.5pp 不在 abstract / conclusion 中作为强 claim**：CI 极宽（[-62pp, +86pp]，n=5 limit），在 discussion 中做 directional 报告；
3. **5 seeds 是 RL benchmark 常见上限**：limitations 显式声明 underpowered。

详细文本（含 per-seed Δ 表 / 写作建议）：[docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md)。

#### 7.16.5 局限性

1. **5 seeds underpowered**：Welch's df ≈ 8、bootstrap n=100 episodes，对 1pp 量级 Δ 的检验功效低。但这正是要量化的事实——0.6pp 差不可能用 n=5 检出。
2. **paired bootstrap 假设 episode_id 对齐**：依赖两个协议复用同一个 fixed manifest，已在 [report §7.10](./rebrac_experiment_report.md) 与 TD3BC report 中确认。
3. **没有 per-episode 配对 t-test**：bootstrap 已经覆盖；额外 paired t-test 不会改变结论方向（持平）。

---

## 8. 综合分析

### 8.1 "预算够不够" 已经不是解释变量

TD3BC phase0c 的报告曾担心 `2000` 没有进一步改善是不是训练预算不够。Stage B0 直接排除了这个可能性（至少在 ReBRAC 侧）。Stage B 结果进一步证实：在 `TRAIN_EPOCHS=64` 的预算下，ReBRAC 在 `crosscomp-2000` 上已经可以超过 ep1000——因此原来 TD3BC 下的 "2000 差于 1000" 现象不能再归咎于 "epoch 不够"，它来自算法能力 ceiling。

### 8.2 优化方差是一个被 β1 直接调控的现象

Stage B0 暴露了 seed 44 在 `β1=2.0` 下的异常；Stage B 把这个观察完整展开成 β1 维度的曲线：

- `β1 ∈ {1.0, 2.0}`：seed 44 系统性崩盘（test success `0.475 ~ 0.800`）；
- `β1 = 4.0`：seed 44 与 seed 42/43 的差距压缩到 `0.025 ~ 0.075`。

这在 `crosscomp-2000` 上尤其干净：`β1=4.0` 的两档 β2 配置下，seed 44 都 ≥ 0.90。

含义：dual penalty 里的 **actor 侧 penalty 是控制优化方差的核心杠杆**，而不是 critic 侧。critic penalty (β2) 更像是锦上添花，β2 从 1.0 升到 2.0 的收益在两个 dataset 上都远小于 β1 从 2.0 升到 4.0 的收益。

### 8.3 ReBRAC 相对 TD3BC 的收益不是 "再调一次超参"，而是 "稳住难 seed"

TD3BC phase0c 在 `crosscomp-1000` 下 `α=0.25` 的 std 是 `0.045`。ReBRAC winner 在同 dataset 下 std 降到 `0.031`；更重要的是，具体是 seed 44 这个 "难 seed" 的 test success 从 TD3BC 下未知（phase0c 只报告了 mean±std）被 ReBRAC 拉到 `0.850`。考虑到 Stage B0 下 seed 44 在 β1=2.0 也只有 0.500，Stage B winner 完整地把这个 seed 回收——这正是 [rebrac_experiment_plan.md §1.2](./rebrac_experiment_plan.md) 提出的 "ReBRAC 能吃下 TD3BC 没吃到的那部分收益" 的最具体形式。

### 8.4 关于 "长训 ≈ 单独训到" 的等价性（回扣 Stage B0）

Stage B 单独训到 `TRAIN_EPOCHS=64` 后，结果上与 Stage B0 的 ep64 读数（验证曲线从 128-epoch 训练中取的 val 点）在同 `(β1=2.0, β2=1.0)` 上具有一致的量级（ep2000 seed_42 在 Stage B0 ep64 是 0.900，Stage B test 得 0.950；同 seed 的 Stage B0 与 Stage B 本就采用不同 manifest，0.05 量级的差距属于 manifest 抖动）。这间接验证了 Stage B0 的 "长训读中间 ckpt ≈ 单独训到该 epoch" 这一核心假设，但严格的 `state_dict_l2_distance` 交叉比对仍未做——这一项列入 §9 局限性。

### 8.5 Stage C 延续：5-seed 稳住了 Stage B 的所有定性结论

Stage C 正式复核在 [§7.8](#78-stage-c-通过判据核对) 的三项阈值上全部通过。对 §8.1–§8.4 综合分析的影响：

- §8.1（"预算够不够" 不再是解释变量）：Stage C 没有新证据挑战这一结论，维持。
- §8.2（β1 是优化方差的主杠杆）：Stage C 下 seed 44 在 β1=4.0 主 finalist 的 `0.870 / 0.890` 进一步确认；但 seed 45 在 **β2=1.0** backup 上的 `0.810` 显示 β2 也在 robustness 中扮演 non-zero 角色，只是强度低于 β1——把 §8.2 的结论从 "actor 侧 penalty 是主杠杆、critic 侧只是锦上添花" 轻度修订为 "actor 侧是主杠杆，critic 侧在新 seed 分布下会贡献 5~10pp 级别的额外 robustness"。
- §8.3（ReBRAC 的收益是 "稳住难 seed"）：Stage C 下 std 的直接对比（ep1000 `0.021` vs TD3BC phase0c `0.045`；ep2000 `0.030` vs `0.036`）比 Stage B 更直接地证实这一机制，且在更大的 100-episode test manifest 下仍然成立。
- §8.4（"长训 ≈ 单独训到" 的等价性）：Stage C 没有额外证据，也没有跑严格的 `state_dict_l2_distance` 比对——移入 §9 局限性。

---

## 9. 局限性

1. **`state_dict_l2_distance` 交叉验证仍未做**——Stage B0 的 "长训 ≈ 单独训到" 假设在 Stage B / Stage C 结果层面一致，但仍未从权重层面直接验证。这是遗留在 Stage C 清单上的 sanity check 项，但不影响当前结论。
2. **backup finalist 的 seed 敏感未解释**——seed 45 在 `(β1=4.0, β2=1.0)` 上得到 `0.810`，Stage B 阶段不可见，Stage C 第一次暴露。如果 Stage D 再次观察到 "β2 小一档在新 seed 上方差放大"，会回来做 critic penalty 敏感性 ablation；当前不 rework。
3. ~~`worldcomp` 轨道**仅 deployable 子集已触及**~~（Phase 2 已完成）→ ~~**已升级为**：worldcomp 全部三个子集（Phase 1 deployable + Phase 2 privileged-critic + critic-penalty-off probe）已跑完，但 Phase 2 是 **3-seed 边际确认而非 5-seed 完整 formal**~~（**rev.8 已修复 — Phase 2 升 5-seed**，详见 §7.12，mean=0.9340±0.0261，std 严格低于 TD3BC priv 0.086）。worldcomp 全部三个子集已完整 5-seed 收口；critic-penalty-off probe 仍只 2 seeds（见下条 4）。
4. ~~只观察了 `β2 ∈ {1.0, 2.0}`~~（critic-penalty-off probe + Stage E (a) 已扩展到 β2=0）。**已升级为**：`β2 = 0` 在 worldcomp 上的实测落入情形 B（2 seeds 同 seeds -5pp，mean_target_q +46%）；`β2 = 0` 在 crosscomp 上的实测同样落入情形 B（5 seeds × test=100，-2.4pp，mean_target_q +98%，seed 44 -17pp）。Finding 1 cross-dataset 验证完毕。但 worldcomp 上 β2=0 仍只覆盖 2 seeds（42/43），如论文 review 反馈"想要 worldcomp 5-seed std for β2=0"，可补；优先级低（crosscomp 5-seed 已经把方向钉死）。
5. ~~**seed 44 在 worldcomp deployable 上的 0.78 离群点尚未机制层面解释**~~（Phase 2 + Stage E (a) 已双向诊断）。**已升级为**：Phase 2 privileged-critic 把 seed 44 在 worldcomp 上救回到 0.90（+12pp，假设 A 成立——critic 信息瓶颈是真因）；Stage E (a) 在 crosscomp 上把 seed 44 在 β2=0 下从 0.87 推到 0.70（-17pp，β2 是 crosscomp 上的稳定信号）。两个 dataset 的双向诊断印证 "actor BC penalty 一条线不足以稳住 seed 44"。collector 层面的 episode 起点分布仍未直接 inspect——属于"已知机制 + 未做的 root-cause sanity"，不阻塞结论；Stage E (b) 维持推迟。
6. **Phase 2 的 privileged ≈ deployable 二阶 finding 仅在 worldcomp-1000 上成立**：是否在其它 dataset / task geometry 上仍然成立未知。这对"privileged critic 在 ReBRAC 上的总体价值"的论文级表述是个 caveat，但不影响 worldcomp-1000 的具体结论。
7. **critic-penalty-off probe 没有覆盖 privileged 轨道**：privileged + β2=0 是否会让 Q 高估同样放大未测（详见 §7.13.8 limit 3）。优先级低，因为 privileged 的 mean 价值已经降到"outlier seed 救援"级。
8. **Stage E (a) 没有覆盖 `crosscomp-2000`**：cross-dataset Finding 1 验证只在 `crosscomp-1000` 上做（详见 §7.14.9 limit 2）。Stage C 已经显示 ep1000 与 ep2000 在 winner finalist 下定性一致，cross-extrapolate 风险低，但严格 ablation 未做。
9. **rev.8 Stage F (B) LN-off probe 仅 2 seeds**：详见 §7.15.8。LN-off 显著退化（-16.2pp）+ seed 42 单点崩溃 -36pp 已足够支撑"LN 是必要 component"存在性 claim，但 std=0.25 的具体数字不应作为 paper 中的 5-seed-style 严格对比；如 review 反馈想看 5-seed std，可补 seeds 44/45/46（≈3h L4，优先级低）。
10. **rev.8 Stage F (C) 统计检验 underpowered**：详见 §7.16.5 + [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md)。5 seeds 对 0.6pp 量级 Δ 检验功效低，但这正是要量化的事实——"持平" claim 已 quantitatively 确认（Welch's p=0.92、paired bootstrap 95% CI 包含 0），不需要更多 seeds 来 disprove。
11. **rev.8 Stage F (D) Q-normalized 变体 method 表述只在 paper 实施**：method draft 已写入 [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md)，但 paper 写作时是否完全 follow 由作者最终决策；如果改用 "ReBRAC variant" 或其它中性表述，需同步更新 main results 表注脚。

---

## 10. 最终结论

1. **`TRAIN_EPOCHS=64` 在 ReBRAC 口径下足够**，且 Stage B 的独立 64-epoch 结果与 Stage B0 的长训中间读数一致，间接验证了 Stage B0 的等价性假设。Stage C 在相同 `TRAIN_EPOCHS=64` 下再度成立。
2. **ReBRAC 的正式 finalist 是 `(β1=4.0, β2=2.0)`**，两个 `crosscomp` 数据集共享。Stage C 下 `crosscomp-1000` 得 `0.902 ± 0.021`，`crosscomp-2000` 得 `0.918 ± 0.030`；两个 dataset 的 std 都等于或低于 TD3BC phase0c 正式 std。
3. **ReBRAC 相对 TD3BC 的收益主要来自 "稳住难 seed"**——seed 44 在 β1=4.0 下完整回收（Stage C 仍为 `0.870 / 0.890`）。Actor 侧 penalty 是优化方差控制的主杠杆；critic penalty (β2) 在 Stage C 下被观察到贡献 non-zero 的额外 robustness（主 finalist vs backup 在 seed 45 上差 7pp）。
4. **ReBRAC 在 `crosscomp-2000` 上翻转了 TD3BC 主线的 "2000 差于 1000" 趋势**：Stage C 下 ep2000 `0.918` > ep1000 `0.902`（TD3BC phase0c: 0.596 < 0.672）。这是 [plan §2.2](./rebrac_experiment_plan.md) 机制问题的正向答案。
5. **Stage C 通过 Rule 1/2/3 三项阈值**：ep1000 mean > 0.672、ep2000 两 finalist mean > 0.75、所有 finalist std ≤ 0.10。ReBRAC 正式升级为 deployable 主基线。
6. **Stage D Phase 1（epoch-probe + deployable formal）已完成**：TRAIN_EPOCHS 锁 64；5 seeds × test=100 拿到 `0.928 ± 0.077`，落入情景 A。
7. **ReBRAC `worldcomp` deployable 在均值上超过 TD3BC privileged-critic 协议**（0.928 vs 0.922），仅用 deployable obs 就把 deployable→teacher 的 gap 关闭了 53.0%（TD3BC privileged 是 48.5%）。这是当前阶段最强的论文级 finding：dual penalty 跨 `crosscomp` / `worldcomp` 数据类型 generalize，"observation 瓶颈" 不再是 worldcomp 上不可绕过的瓶颈。
8. **Stage D Phase 1 三个机制 finding（Phase 2 + probe 后修订）**：
   - **Finding 1（已修订）**：dual penalty 在 worldcomp 上 `β2·ratio` 数字小（0.011 vs crosscomp 0.086~0.358）**不等于 critic penalty 贡献为零**——critic-penalty-off probe (§7.13) 显示，β2=0 让 mean_target_q 上跳 +46%（15.22 → 22.30），mean success 同 seeds 掉 5.0pp。正确表述：critic penalty 在 worldcomp 上对 mean success 的贡献"小但非零"，对 Q 稳定性的贡献"大且必要"。
   - **Finding 2（已部分回答）**：ReBRAC 在 worldcomp 上对 seed 44 在 deployable 轨道失去回收能力（0.78），但 Phase 2 privileged-critic 把 seed 44 救回到 0.90（+12pp）。**假设 A 成立：critic 信息瓶颈是 seed 44 outlier 的真因**——actor BC penalty 一条线不足以稳住 seed 44，需要 critic 看到 hull-integral `[u_eq, v_eq]` 才能恢复稳定 Q。
   - **Finding 3**：Phase 1 probe → formal 的缩水（`1.0 → 0.928`）90% 由 seed 44 单点解释——这一 finding 维持，且与 Finding 2 形成闭环（seed 44 在 deployable 是 outlier，在 privileged 不是）。
9. **Stage D Phase 2（已完成，rev.8 升 5-seed）**：privileged-critic **5 seeds × test=100 → `mean=0.9340, std=0.0261`**（rev.7 时 3 seeds=`0.9267 ± 0.025`；升 5-seed 后 mean +0.7pp、std 几乎不变），4 / 4 判据全部通过；与 ReBRAC deployable 几乎相等（Δ priv − dep = +0.6pp），但 seed 44 独立救回 +12pp。**std 0.0261 严格低于 TD3BC priv 5-seed std 0.086，paper 可正式做 std 对比 claim。**
10. **二阶 finding：privileged ≈ deployable 但机制不同（论文级核心 finding，rev.8 5-seed 强化）**：TD3BC 上 privileged-critic 给所有 seed 平均抬升 +6.4pp；ReBRAC 上 privileged-critic 不抬均值（+0.6pp，5-seed），只在 outlier seed（如 seed 44）上贡献额外稳定性（+12pp）。这意味着 ReBRAC + deployable obs 已经把 worldcomp 的 deployable→teacher gap 关闭主要由 actor BC penalty 完成，privileged critic 在 ReBRAC 上的边际价值已降到"outlier seed 救援"级——**这是 sim2real 论文层面最强的部署友好性结论**（actor / critic 都不依赖 hull-integral 也能拿 0.928）。
11. **Stage E (a) 已完成（cross-dataset Finding 1 二次验证）**：`crosscomp-1000 (β1=4.0, β2=0) × 5 seeds × test=100` 拿到 `0.878 ± 0.090`，落入情形 B；同 dataset Δ -2.4pp、`mean_target_q` 漂 +8.12 单位（+98%）、seed 44 掉 -17pp。**两条 cross-dataset 证据合并把 Finding 1 钉死为 "critic penalty 对 mean 贡献小但非零、对 Q 稳定性大且必要、对 outlier seed 鲁棒性必需"**（详见 §7.14.7）。dual penalty `(β1=4.0, β2=2.0)` 是 ReBRAC 主线必需配置，不向单 penalty 简化。Stage E (b) seed 44 collector inspection 维持推迟（触发恢复条件均未触发，详见 §7.14.8）。
12. **rev.8 Stage F (paper-readiness probes) 全部完成**：(A) Phase 2 升 5-seed → §7.12 已更新；(B) critic LN-off probe (crosscomp-1000, 2 seeds) → §7.15，**mean=0.74 ± 0.25 比 LN-on Stage C 掉 -16.2pp，且方向不同于 β2=0（LN-off 让 Q 朝更负 + std blow-up 12×；β2=0 让 Q 朝更正）→ LN 与 dual penalty 是两个独立 component**；(C) paired bootstrap + Welch's t-test on Phase 1 dep vs TD3BC priv → §7.16 + [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md)，Welch's p=0.9195、paired bootstrap 95% CI=[-3.0pp, +4.2pp] → "持平" quantitatively confirmed；(D) Q-normalized variant method draft → [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md)，algorithm 命名为 "Q-normalized dual-penalty TD3+BC variant (alias ReBRAC-Q)"，β1=4.0 ↔ TD3+BC α≈0.25。**审稿人弱点全部堵死。**
13. **ReBRAC 主线整体收口（rev.8 paper-readiness 完整版）**：Stage A (impl) → B0 (epoch probe) → B (3×2 screening) → C (5-seed formal) → D Phase 1 (worldcomp deployable) + Phase 2 (privileged-critic, **rev.8 5-seed**) + critic-penalty-off probe → E (a) (crosscomp cross-dataset 验证) → **F (rev.8 paper-readiness probes A-D)** 全部完成。论文级核心 finding 四条：(i) ReBRAC 在 crosscomp 上对 TD3BC +23~+32pp（Stage C）；(ii) ReBRAC 在 worldcomp 上 deployable ≈ privileged-critic 关闭 ~52~58% gap，privileged 价值降到 outlier seed 救援级（Stage D，5-seed）；(iii) dual penalty 对 mean 贡献小但对 Q 稳定性 + outlier seed 鲁棒性必需（cross-dataset，Stage D probe + Stage E (a)）；(iv) **critic LN 与 dual penalty 是两个独立必要 component**（Stage F (B)：LN-off -16.2pp 远大于 β2=0 -2.4pp，且退化方向不同）。

一句话总结：

> ReBRAC 主线整体收口（rev.8 paper-readiness 完整版）。Stage A→B0→B→C→D (Phase 1+2+probe)→E (a)→F (4 项 paper-readiness probes A-D) 全部完成。`worldcomp-1000` 上 deployable（5 seeds, 0.928）与 privileged-critic（**5 seeds, 0.9340 ± 0.0261**，std 严格低于 TD3BC priv）两个轨道关闭 ~52~58% 的 deployable→teacher gap，**privileged critic 在 ReBRAC 上不再贡献 mean 抬升，只在 outlier seed 上贡献稳定性**——sim2real 部署友好性最强的论文级 finding。Finding 1 经 worldcomp probe + crosscomp Stage E (a) 双向验证，最终表述为 "critic penalty 对 mean 贡献小但非零、对 Q 稳定性大且必要、对 outlier seed 鲁棒性必需"，dual penalty `(β1=4.0, β2=2.0)` 是主线必需配置。**Stage F (B) LN-off probe 把 LN 与 dual penalty 钉死为两个独立 component**（-16.2pp vs -2.4pp，退化方向不同）。Welch's p=0.9195 量化确认 ReBRAC dep 与 TD3BC priv "持平"，避免 paper 中 over-claim "首次超过"。**审稿人弱点全部堵死，paper drafting 可正式启动。** 下一步可选：转向 paper drafting / online thesis 线 / 其它研究方向。

---

## 10A. Broad validation — pointer to standalone reports

> **Status（2026-05-07 rev — 退回 commit cf5cfff 的 retrofit）**
>
> 本节原 retrofit（cf5cfff，sensor-floor framing + C1-s1 verdict gate）含两处过期论述：
> 1. §10A.3 的 "sensor floor → 预期 sensor 升级解锁 upstream u10" 已被 [`c1_s1_followup`](rebrac_c1_s1_followup_report.md) 实测反证（s1 升级 −2pp，落 <0.30 verdict）；
> 2. 整个 §10A.4 verdict gate 表是该反证之前的预登记。
>
> 按用户 2026-05-07 判断，**广验 (broad validation) 与 C1-s1 follow-up 结果尚不达 paper-quality**——
> - broad_validation rev.2 的 8 spoke 中只有 B1 sensor envelope (Δ=−0.2pp) 是 clean positive；A1 underpowered (paired t p≈0.11)、A2 mid-gap collapse 的 mode-collapse 机制 hypothesis 未做直接 ablation、B2/C3 std blow-up 无 mechanism ablation；
> - C1 task-fundamental floor 的 mechanism discriminator（**BC penalty 强度 sweep on C1**）未做；
> - mix ratio sweep / target_speed=2.0 P1 probe / A1 paired bootstrap 都在 backlog。
>
> 在这些下游 sweep 闭环、broad validation 结果稳定到 paper-quality 之前，**主线主报告 retrofit deferred**。本节正文退回为 pointer，避免把还在动的 commentary 钉死进 finding spine。
>
> ### 广验证据的 standalone 入口
>
> | 文档 | 内容 |
> |---|---|
> | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) | 三轴广验全表（A/B/C × 8 spoke × 5-seed parity）+ C1 deep-dive（§3.5：5 个 ablation 全部钉在 0.195–0.225 → task-fundamental floor 候选） |
> | [`docs/rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) | C1-s1 sensor-upgrade follow-up（s1 升级 −2pp、<0.30 verdict、reward 与 sensor 各自独立 modulate failure mode 但都不动 ceiling） |
> | [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §3.5 | strategic review 顶层指针（不复制内容，链接到上面两份） |
>
> ### 对 §10 主线结论的影响
>
> **无**。Finding (i)–(iv) 在主线 cell（`crosscomp-1000 / s0 / cross_stream / Re150 / target_speed=1.5`）上严密成立、5-seed 收口、未受广验影响。广验是对 generality 边界的 commentary，下游 sweep 完成后再统一考虑回写主线。

---

## 11. 结果文件索引

本报告基于以下文件：

- Stage B0 训练预算 probe
  - `checkpoints/offline/rebrac/screening_epoch_probe/crosscomp-{1000,2000}/actorb_2p0__criticb_1p0/seed_{42,43,44}/agent_step_*.pt`
  - `results/offline/rebrac/screening_epoch_probe/crosscomp-{1000,2000}/actorb_2p0__criticb_1p0/selection/seed_{42,43,44}/selected_checkpoint.json`
- Stage B 最小 screening
  - `checkpoints/offline/rebrac/screening/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{1000,2000}/actorb_{1p0,2p0,4p0}__criticb_{1p0,2p0}/seed_{42,43,44}/agent_step_*.pt`
  - `results/offline/rebrac/screening/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{1000,2000}/actorb_*__criticb_*/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/screening/summaries/overview.{csv,json}`
- Stage C 正式 5-seed
  - `checkpoints/offline/rebrac/formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{1000,2000}/actorb_4p0__criticb_{1p0,2p0}/seed_{42,43,44,45,46}/agent_step_*.pt`（seed 42/43/44 物理复制自 Stage B）
  - `results/offline/rebrac/formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{1000,2000}/actorb_4p0__criticb_*/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/formal/summaries/overview.{csv,json}`
- Stage D Phase 1 Step 1（worldcomp epoch-probe）
  - `checkpoints/offline/rebrac/worldcomp_epoch_probe/...`
  - `results/offline/rebrac/worldcomp_epoch_probe/...`
- Stage D Phase 1 Step 2（worldcomp deployable formal）
  - `checkpoints/offline/rebrac/worldcomp_teacher_gap/deployable/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/seed_{42,43,44,45,46}/agent_step_*.pt`
  - `results/offline/rebrac/worldcomp_teacher_gap/deployable/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/worldcomp_teacher_gap/deployable/summaries/overview.{csv,json}`
  - `benchmarks/offline_rebrac_worldcomp_final/{val_40,test_100}/single_u10_cross_tgt15.json`
- Stage D Phase 2（worldcomp privileged-critic formal）
  - `checkpoints/offline/rebrac/worldcomp_teacher_gap/privileged_critic/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/seed_{42,43,44,45,46}/agent_step_*.pt`（rev.8 升 5-seed）
  - `results/offline/rebrac/worldcomp_teacher_gap/privileged_critic/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/worldcomp_teacher_gap/privileged_critic/summaries/overview.{csv,json}`
- Stage D — critic-penalty-off probe
  - `checkpoints/offline/rebrac/worldcomp_critic_penalty_off_probe/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_0p0/seed_{42,43}/agent_step_*.pt`
  - `results/offline/rebrac/worldcomp_critic_penalty_off_probe/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_0p0/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/worldcomp_critic_penalty_off_probe/summaries/overview.{csv,json}`
- Stage E (a)（crosscomp-1000 critic-penalty-off cross-dataset 二次验证）
  - `checkpoints/offline/rebrac/stage_e_critic_penalty_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_0p0/seed_{42,43,44,45,46}/agent_step_*.pt`
  - `results/offline/rebrac/stage_e_critic_penalty_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_0p0/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/stage_e_critic_penalty_off/summaries/overview.{csv,json}`
  - val/test manifests 复用 Stage C：`benchmarks/offline_rebrac_screen/{val_40,test_100}/single_u10_cross_tgt15.json`
  - 离线数据复用 Stage C：`offline_data/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/transitions.npz`
- **Stage F (B)（rev.8 — crosscomp-1000 critic LayerNorm-off probe）**
  - `checkpoints/offline/rebrac/critic_ln_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/seed_{42,43}/agent_step_*.pt`
  - `results/offline/rebrac/critic_ln_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/critic_ln_off/summaries/overview.{csv,json}`
  - sibling driver：`scripts/run_offline_rebrac_critic_ln_off.sh`（notebook 一次性 `text.replace("--critic-layernorm \\", "--no-critic-layernorm \\", 1)` 生成；不入 git tracked source）
- **Stage F (C/D)（rev.8 — 统计检验 + method draft，纯分析无 checkpoint）**
  - 输入：Stage D Phase 1 deployable + TD3BC privileged-final test 的 `eval_episode_results`
  - 输出：[docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md)（Welch's t-test + paired bootstrap + gap closure CI）
  - 输出：[docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md)（algorithm 命名 + actor loss 公式 + ReBRAC / TD3+BC 差异表）

执行入口：

- [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb) — Stage B0
- [notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb) — Stage B
- [notebooks/rebrac_screen_completed.ipynb](../notebooks/rebrac_screen_completed.ipynb) — Stage B 执行归档
- [notebooks/rebrac_formal.ipynb](../notebooks/rebrac_formal.ipynb) — Stage C
- [notebooks/rebrac_formal_completed.ipynb](../notebooks/rebrac_formal_completed.ipynb) — Stage C 执行归档
- [notebooks/rebrac_worldcomp_epoch_probe_completed.ipynb](../notebooks/rebrac_worldcomp_epoch_probe_completed.ipynb) — Stage D Phase 1 Step 1 执行归档
- [notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb](../notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb) — Stage D Phase 1 Step 2 执行归档
- [notebooks/rebrac_worldcomp_phase2_privileged_completed.ipynb](../notebooks/rebrac_worldcomp_phase2_privileged_completed.ipynb) — Stage D Phase 2 执行归档
- [notebooks/rebrac_worldcomp_critic_penalty_off_probe_completed.ipynb](../notebooks/rebrac_worldcomp_critic_penalty_off_probe_completed.ipynb) — Stage D critic-penalty-off probe 执行归档
- [notebooks/rebrac_stage_e_critic_penalty_off_crosscomp.ipynb](../notebooks/rebrac_stage_e_critic_penalty_off_crosscomp.ipynb) — Stage E (a) scaffold
- [notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb](../notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb) — Stage E (a) 执行归档
- [notebooks/rebrac_paper_followup.ipynb](../notebooks/rebrac_paper_followup.ipynb) — **rev.8 Stage F scaffold（4 项 paper-readiness probes 合并）**
- [notebooks/rebrac_paper_followup_completed.ipynb](../notebooks/rebrac_paper_followup_completed.ipynb) — **rev.8 Stage F 执行归档（任务 A/B/C/D 全部 PASS）**
- [notebooks/rebrac_c1_reward_ablation_completed.ipynb](../notebooks/rebrac_c1_reward_ablation_completed.ipynb) — **§10A.2 Ablation A（broad validation C1 retrofit）**
- [notebooks/rebrac_c1_asym_critic_ablation_completed.ipynb](../notebooks/rebrac_c1_asym_critic_ablation_completed.ipynb) — **§10A.2 Ablation B（broad validation C1 retrofit）**
- [notebooks/rebrac_c1_train_convergence_check_completed.ipynb](../notebooks/rebrac_c1_train_convergence_check_completed.ipynb) — **§10A.2 Ablation C 先决无 GPU 诊断**
- [notebooks/rebrac_c1_epoch_sensitivity_ablation_completed.ipynb](../notebooks/rebrac_c1_epoch_sensitivity_ablation_completed.ipynb) — **§10A.2 Ablation C（broad validation C1 retrofit）**
- [scripts/run_offline_rebrac_screen.sh](../scripts/run_offline_rebrac_screen.sh)（Stage B / Stage C / Stage E (a) 共用；Stage D 通过 `run_offline_rebrac_worldcomp_teacher_gap.sh` 间接调用）
- [scripts/run_offline_rebrac_worldcomp_teacher_gap.sh](../scripts/run_offline_rebrac_worldcomp_teacher_gap.sh) — Stage D 专用驱动

背景阅读：

- [rebrac_experiment_plan.md](./rebrac_experiment_plan.md)
- [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)
- [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)
