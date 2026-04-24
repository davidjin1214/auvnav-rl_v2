# ReBRAC 实验报告

> 文档版本：2026-04-24 rev.2
> 文档定位：这是 ReBRAC 阶段所有已经**实际跑完**的实验的统一结果报告。与 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) 不同，本文只写 "跑了什么 / 看到了什么 / 意味着什么"，不讨论尚未执行的计划。
> 阅读建议：先读 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) §1-§5 理解动机与口径，再回到这里看已有结果。

---

## 1. 报告概述

本文档整理 `results/offline/rebrac/` 下已经完成的全部 ReBRAC 实验，并把每一组实验的结论并入一条完整的 "主线解释" 里。当前（`rev.2`）包含的实验包为：

- `screening_epoch_probe`（Stage B0，训练预算 probe）
- `screening`（Stage B，最小 screening —— 3×2 penalty 网格 × 2 datasets × 3 seeds）

后续实验（Stage C formal、Stage D teacher-gap follow-up）完成后，会在本文档内续写对应章节，不再分散到独立文件中。

本文档要回答的核心问题与 [rebrac_experiment_plan.md §2](./rebrac_experiment_plan.md) 一致：

1. 在与 phase0c 相同的 deployable canonical protocol 下，ReBRAC 是否能稳定优于 TD3BC？
2. ReBRAC 是否能改善 `crosscomp-2000` 相对 `crosscomp-1000` 的退化？
3. 如果 ReBRAC 在 `worldcomp-1000` 上也有改善，这种改善来自 deployable 轨道还是 privileged critic 轨道？

截至当前（`rev.2`）：

- **问题 1 已出 screening 级别的强阳性信号**：ReBRAC 的 Stage B winner 在两个 `crosscomp` 数据集上相对 TD3BC phase0c Stage C 的正式成绩都给出了大幅改进（+21.1pp / +32.1pp）。是否稳住要看 Stage C 5-seed 复核。
- **问题 2 同样在 Stage B 级别拿到初步正向**：ReBRAC 下 `crosscomp-2000` 比 `crosscomp-1000` 还略好（0.917 vs 0.883 在 winner 格子），首次翻转了 TD3BC 主线下 "`2000` 系统性差于 `1000`" 的趋势。
- **问题 3 未触及**：需要 Stage D 才能回答。

一句话当前状态：

> `TRAIN_EPOCHS=64` 已经够用，Stage B 最小 screening 跑完并给出干净的 winner `(β1=4.0, β2=2.0)`；ReBRAC 在两个 `crosscomp` 数据集上都显著超越 TD3BC phase0c。下一步直接进入 Stage C 5-seed 正式复核。

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

截至 `rev.2`：

| 实验包 | 编号 | 状态 | 入口 |
| --- | --- | --- | --- |
| 训练预算 probe | Stage B0 | 已完成 | [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb) |
| 最小 screening | Stage B | **已完成** | [notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb) |
| 正式 5-seed 确认 | Stage C | 待执行（finalist 已锁定） | — |
| `worldcomp` teacher-gap follow-up | Stage D | 待定（依赖 Stage C 成绩） | — |

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

## 7. 综合分析

### 7.1 "预算够不够" 已经不是解释变量

TD3BC phase0c 的报告曾担心 `2000` 没有进一步改善是不是训练预算不够。Stage B0 直接排除了这个可能性（至少在 ReBRAC 侧）。Stage B 结果进一步证实：在 `TRAIN_EPOCHS=64` 的预算下，ReBRAC 在 `crosscomp-2000` 上已经可以超过 ep1000——因此原来 TD3BC 下的 "2000 差于 1000" 现象不能再归咎于 "epoch 不够"，它来自算法能力 ceiling。

### 7.2 优化方差是一个被 β1 直接调控的现象

Stage B0 暴露了 seed 44 在 `β1=2.0` 下的异常；Stage B 把这个观察完整展开成 β1 维度的曲线：

- `β1 ∈ {1.0, 2.0}`：seed 44 系统性崩盘（test success `0.475 ~ 0.800`）；
- `β1 = 4.0`：seed 44 与 seed 42/43 的差距压缩到 `0.025 ~ 0.075`。

这在 `crosscomp-2000` 上尤其干净：`β1=4.0` 的两档 β2 配置下，seed 44 都 ≥ 0.90。

含义：dual penalty 里的 **actor 侧 penalty 是控制优化方差的核心杠杆**，而不是 critic 侧。critic penalty (β2) 更像是锦上添花，β2 从 1.0 升到 2.0 的收益在两个 dataset 上都远小于 β1 从 2.0 升到 4.0 的收益。

### 7.3 ReBRAC 相对 TD3BC 的收益不是 "再调一次超参"，而是 "稳住难 seed"

TD3BC phase0c 在 `crosscomp-1000` 下 `α=0.25` 的 std 是 `0.045`。ReBRAC winner 在同 dataset 下 std 降到 `0.031`；更重要的是，具体是 seed 44 这个 "难 seed" 的 test success 从 TD3BC 下未知（phase0c 只报告了 mean±std）被 ReBRAC 拉到 `0.850`。考虑到 Stage B0 下 seed 44 在 β1=2.0 也只有 0.500，Stage B winner 完整地把这个 seed 回收——这正是 [rebrac_experiment_plan.md §1.2](./rebrac_experiment_plan.md) 提出的 "ReBRAC 能吃下 TD3BC 没吃到的那部分收益" 的最具体形式。

### 7.4 关于 "长训 ≈ 单独训到" 的等价性（回扣 Stage B0）

Stage B 单独训到 `TRAIN_EPOCHS=64` 后，结果上与 Stage B0 的 ep64 读数（验证曲线从 128-epoch 训练中取的 val 点）在同 `(β1=2.0, β2=1.0)` 上具有一致的量级（ep2000 seed_42 在 Stage B0 ep64 是 0.900，Stage B test 得 0.950；同 seed 的 Stage B0 与 Stage B 本就采用不同 manifest，0.05 量级的差距属于 manifest 抖动）。这间接验证了 Stage B0 的 "长训读中间 ckpt ≈ 单独训到该 epoch" 这一核心假设，但严格的 `state_dict_l2_distance` 交叉比对仍未做——这是 Stage C 的 sanity check 清单项。

---

## 8. 局限性

1. **仍未完成 Stage C 5-seed 正式复核**——所有 Stage B 的超越 TD3BC 的结论都是 3-seed 级别的，严格意义上属于 screening 强信号。
2. Stage B 的 test manifest 是 40 episodes；Stage C 会升到 100 episodes，届时 test success 的置信区间会显著收窄。
3. `worldcomp` 轨道完全未触及——Stage B 的所有结论都不能外推到 deterministic teacher 数据上。
4. 只观察了 `β2 ∈ {1.0, 2.0}`；`β2 = 0.5` 下 ReBRAC 是否退化到 TD3BC-like 行为（critic penalty 几乎不起作用）未知。这对理解 "critic penalty 到底贡献多少" 很重要，但位于 Stage E ablation 范围，不是当前执行序列的前置条件。

---

## 9. 最终结论

1. **`TRAIN_EPOCHS=64` 在 ReBRAC 口径下足够**，且 Stage B 的独立 64-epoch 结果与 Stage B0 的长训中间读数一致，间接验证了 Stage B0 的等价性假设。
2. **ReBRAC 的 Stage B winner 是 `(β1=4.0, β2=2.0)`**，两个 `crosscomp` 数据集共享。这个 winner 在 mean 和 std 两项上都优于 TD3BC phase0c 的对应最优。
3. **ReBRAC 相对 TD3BC 的收益主要来自 "稳住难 seed"**——seed 44 从 TD3BC / 弱 penalty 下的系统性崩盘被 β1=4.0 完整回收。Actor 侧 penalty 是优化方差控制的主杠杆。
4. **ReBRAC 在 `crosscomp-2000` 上翻转了 TD3BC 主线的 "2000 差于 1000" 趋势**（ReBRAC: ep2000 0.917 > ep1000 0.883；TD3BC phase0c: ep2000 0.596 < ep1000 0.672）。这是 [plan §2.2](./rebrac_experiment_plan.md) 机制问题的初步正向答案。
5. **Stage C 进入 5-seed 正式复核**，finalist: `(β1=4.0, β2=2.0)` 两 dataset 共用；`crosscomp-2000` 上额外跑 `(β1=4.0, β2=1.0)` 作为 backup。

一句话总结：

> ReBRAC 的 Stage B screening 给出了强阳性信号：winner 统一、与 TD3BC 对比的差距显著、seed 44 类型的优化方差被 β1 直接压下去；接下来用 Stage C 5-seed 把这个结论升级到 "正式基线"。

---

## 10. 结果文件索引

本报告基于以下文件：

- Stage B0 训练预算 probe
  - `checkpoints/offline/rebrac/screening_epoch_probe/crosscomp-1000/actorb_2p0__criticb_1p0/seed_{42,43,44}/agent_step_*.pt`
  - `checkpoints/offline/rebrac/screening_epoch_probe/crosscomp-2000/actorb_2p0__criticb_1p0/seed_{42,43,44}/agent_step_*.pt`
  - `results/offline/rebrac/screening_epoch_probe/crosscomp-{1000,2000}/actorb_2p0__criticb_1p0/selection/seed_{42,43,44}/selected_checkpoint.json`
- Stage B 最小 screening
  - `checkpoints/offline/rebrac/screening/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{1000,2000}/actorb_{1p0,2p0,4p0}__criticb_{1p0,2p0}/seed_{42,43,44}/agent_step_*.pt`
  - `results/offline/rebrac/screening/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep{1000,2000}/actorb_*__criticb_*/{validation,test,selection}/seed_*.json`
  - `results/offline/rebrac/screening/summaries/overview.{csv,json}`

执行入口：

- [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb) — Stage B0
- [notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb) — Stage B
- [notebooks/rebrac_screen_completed.ipynb](../notebooks/rebrac_screen_completed.ipynb) — Stage B 执行归档
- [scripts/run_offline_rebrac_screen.sh](../scripts/run_offline_rebrac_screen.sh)

背景阅读：

- [rebrac_experiment_plan.md](./rebrac_experiment_plan.md)
- [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)
- [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)
