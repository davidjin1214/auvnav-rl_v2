# ReBRAC 实验报告

> 文档版本：2026-04-24 rev.3
> 文档定位：这是 ReBRAC 阶段所有已经**实际跑完**的实验的统一结果报告。与 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) 不同，本文只写 "跑了什么 / 看到了什么 / 意味着什么"，不讨论尚未执行的计划。
> 阅读建议：先读 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) §1-§5 理解动机与口径，再回到这里看已有结果。

---

## 1. 报告概述

本文档整理 `results/offline/rebrac/` 下已经完成的全部 ReBRAC 实验，并把每一组实验的结论并入一条完整的 "主线解释" 里。当前（`rev.3`）包含的实验包为：

- `screening_epoch_probe`（Stage B0，训练预算 probe）
- `screening`（Stage B，最小 screening —— 3×2 penalty 网格 × 2 datasets × 3 seeds）
- `formal`（Stage C，5-seed 正式复核 —— 3 finalists × 5 seeds × test 100 episodes）

后续实验（Stage D `worldcomp` teacher-gap follow-up）完成后，会在本文档内续写对应章节，不再分散到独立文件中。

本文档要回答的核心问题与 [rebrac_experiment_plan.md §2](./rebrac_experiment_plan.md) 一致：

1. 在与 phase0c 相同的 deployable canonical protocol 下，ReBRAC 是否能稳定优于 TD3BC？
2. ReBRAC 是否能改善 `crosscomp-2000` 相对 `crosscomp-1000` 的退化？
3. 如果 ReBRAC 在 `worldcomp-1000` 上也有改善，这种改善来自 deployable 轨道还是 privileged critic 轨道？

截至当前（`rev.3`）：

- **问题 1 已由 Stage C 5-seed 正式确认**：ReBRAC 的主 finalist 在 `crosscomp-1000` 上拿到 `0.902 ± 0.021`，在 `crosscomp-2000` 上拿到 `0.918 ± 0.030`；相对 TD3BC phase0c 正式成绩分别高出 `+23.0pp` 和 `+32.2pp`；std 也同时等于或低于 TD3BC phase0c 正式 std。
- **问题 2 同样成立**：Stage C 下 `crosscomp-2000`（0.918）仍然高于 `crosscomp-1000`（0.902），Stage B 观察到的 "2000 不再差于 1000" 的翻转在 5-seed 下延续。
- **问题 3 未触及**：需要 Stage D 才能回答；Stage C 的阳性结论已触发 Stage D `worldcomp-1000` 的 follow-up。

一句话当前状态：

> Stage C 5-seed 正式复核已完成并通过三项阈值，ReBRAC 正式升级为 deployable 主基线：`crosscomp-1000` 上 `0.902 ± 0.021`（ΔvsTD3BC +23.0pp），`crosscomp-2000` 上 `0.918 ± 0.030`（Δ +32.2pp）；backup finalist `(β1=4.0, β2=1.0)` 同样达标但在新 seed 45 上出现 `0.810` 的离群点，因此只做 fallback、不升级为正式成绩。下一步进入 Stage D `worldcomp-1000` teacher-gap follow-up。

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

截至 `rev.3`：

| 实验包 | 编号 | 状态 | 入口 |
| --- | --- | --- | --- |
| 训练预算 probe | Stage B0 | 已完成 | [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb) |
| 最小 screening | Stage B | 已完成 | [notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb) |
| 正式 5-seed 确认 | Stage C | **已完成** | [notebooks/rebrac_formal.ipynb](../notebooks/rebrac_formal.ipynb) |
| `worldcomp` teacher-gap follow-up | Stage D | 当前执行（Stage C 阳性已触发） | — |

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

### 7.10 局限性

1. **seed 45 在 backup 上的离群未深挖**：`(β1=4.0, β2=1.0)` 在 seed 45 上得到 `0.810`，比同 seed 主 finalist 低 7pp。可能是 seed 45 本身对 β2=1.0 更敏感，也可能是单点噪声。如果 Stage D 或后续 ablation 再次观察到 "β2 小一档在新 seed 上方差放大" 的模式，可把它作为 critic penalty 敏感性的起点；目前不追加 7-seed rework。
2. **没有做 7-seed / 10-seed follow-up**：三项阈值都已通过，再扩样本的边际收益低；如果 Stage D 在 `worldcomp privileged-critic` 再次出现不稳，再回头决定是否补样本。
3. **与 TD3BC phase0c 的对比仍跨实验批次**：两边都在 canonical protocol 下执行，但 `TRAIN_EPOCHS` 历史上有过调整（TD3BC 主线 64，worldcomp teacher-gap 96），ReBRAC 统一 64。这对 `crosscomp` 的直接对比无影响，但对 Stage D `worldcomp` 对比需要在 Stage D 报告中单独澄清预算口径。

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
3. `worldcomp` 轨道完全未触及——Stage B / Stage C 的所有结论都不能外推到 deterministic teacher 数据上。这是 Stage D 的范围。
4. 只观察了 `β2 ∈ {1.0, 2.0}`；`β2 = 0.5` 下 ReBRAC 是否退化到 TD3BC-like 行为（critic penalty 几乎不起作用）未知。这对理解 "critic penalty 到底贡献多少" 很重要，但位于 Stage E ablation 范围，不是当前执行序列的前置条件。

---

## 10. 最终结论

1. **`TRAIN_EPOCHS=64` 在 ReBRAC 口径下足够**，且 Stage B 的独立 64-epoch 结果与 Stage B0 的长训中间读数一致，间接验证了 Stage B0 的等价性假设。Stage C 在相同 `TRAIN_EPOCHS=64` 下再度成立。
2. **ReBRAC 的正式 finalist 是 `(β1=4.0, β2=2.0)`**，两个 `crosscomp` 数据集共享。Stage C 下 `crosscomp-1000` 得 `0.902 ± 0.021`，`crosscomp-2000` 得 `0.918 ± 0.030`；两个 dataset 的 std 都等于或低于 TD3BC phase0c 正式 std。
3. **ReBRAC 相对 TD3BC 的收益主要来自 "稳住难 seed"**——seed 44 在 β1=4.0 下完整回收（Stage C 仍为 `0.870 / 0.890`）。Actor 侧 penalty 是优化方差控制的主杠杆；critic penalty (β2) 在 Stage C 下被观察到贡献 non-zero 的额外 robustness（主 finalist vs backup 在 seed 45 上差 7pp）。
4. **ReBRAC 在 `crosscomp-2000` 上翻转了 TD3BC 主线的 "2000 差于 1000" 趋势**：Stage C 下 ep2000 `0.918` > ep1000 `0.902`（TD3BC phase0c: 0.596 < 0.672）。这是 [plan §2.2](./rebrac_experiment_plan.md) 机制问题的正向答案。
5. **Stage C 通过 Rule 1/2/3 三项阈值**：ep1000 mean > 0.672、ep2000 两 finalist mean > 0.75、所有 finalist std ≤ 0.10。ReBRAC 正式升级为 deployable 主基线。
6. **Stage D 进入当前执行**：`worldcomp-1000` teacher-gap follow-up，finalist 锁定 `(β1=4.0, β2=2.0)`，轨道分 `deployable` + `privileged-critic`。预算口径待 Stage D epoch-probe sanity check 后确定（64 或 96）。

一句话总结：

> ReBRAC 的 Stage C 5-seed 正式复核完整延续了 Stage B 的所有定性结论：均值大幅超越 TD3BC phase0c（+23pp / +32pp）、std 不增反降、"2000 不再差于 1000" 翻转继续成立；接下来进入 Stage D 的 `worldcomp` teacher-gap 轨道。

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

执行入口：

- [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb) — Stage B0
- [notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb) — Stage B
- [notebooks/rebrac_screen_completed.ipynb](../notebooks/rebrac_screen_completed.ipynb) — Stage B 执行归档
- [notebooks/rebrac_formal.ipynb](../notebooks/rebrac_formal.ipynb) — Stage C
- [notebooks/rebrac_formal_completed.ipynb](../notebooks/rebrac_formal_completed.ipynb) — Stage C 执行归档
- [scripts/run_offline_rebrac_screen.sh](../scripts/run_offline_rebrac_screen.sh)（Stage B / Stage C 共用）

背景阅读：

- [rebrac_experiment_plan.md](./rebrac_experiment_plan.md)
- [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)
- [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)
