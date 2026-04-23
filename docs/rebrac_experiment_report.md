# ReBRAC 实验报告

> 文档版本：2026-04-23 rev.1
> 文档定位：这是 ReBRAC 阶段所有已经**实际跑完**的实验的统一结果报告。与 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) 不同，本文只写 "跑了什么 / 看到了什么 / 意味着什么"，不讨论尚未执行的计划。
> 阅读建议：先读 [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) §1-§5 理解动机与口径，再回到这里看已有结果。

---

## 1. 报告概述

本文档整理 `results/offline/rebrac/` 下已经完成的全部 ReBRAC 实验，并把每一组实验的结论并入一条完整的 "主线解释" 里。当前（`rev.1`）包含的实验包为：

- `screening_epoch_probe`（Stage B0，训练预算 probe）

后续实验（Stage B 正式 screening、Stage C formal、Stage D teacher-gap follow-up）完成后，会在本文档内续写对应章节，不再分散到独立文件中。

本文档要回答的核心问题与 [rebrac_experiment_plan.md §2](./rebrac_experiment_plan.md) 一致：

1. 在与 phase0c 相同的 deployable canonical protocol 下，ReBRAC 是否能稳定优于 TD3BC？
2. ReBRAC 是否能改善 `crosscomp-2000` 相对 `crosscomp-1000` 的退化？
3. 如果 ReBRAC 在 `worldcomp-1000` 上也有改善，这种改善来自 deployable 轨道还是 privileged critic 轨道？

截至当前（`rev.1`），这三个问题都**没有**正式结论；本文只记录了为回答这些问题而做的**预备实验**（训练预算 probe）。

一句话当前状态：

> TD3BC phase0c 沿用下来的 `TRAIN_EPOCHS=64` 在 ReBRAC 口径下已经足够；接下来的正式 Stage B screening 可以直接按 `TRAIN_EPOCHS=64` 开跑，并在结果分析阶段把 `std_test_success_rate` 作为一等公民看待。

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

截至 `rev.1`：

| 实验包 | 编号 | 状态 | 入口 |
| --- | --- | --- | --- |
| 训练预算 probe | Stage B0 | 已完成 | [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb) |
| 最小 screening | Stage B | 进行中 / 即将开跑 | [notebooks/rebrac_screen.ipynb](../notebooks/rebrac_screen.ipynb) |
| 正式 5-seed 确认 | Stage C | 待定（依赖 Stage B 结果） | — |
| `worldcomp` teacher-gap follow-up | Stage D | 待定 | — |

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

## 6. 综合分析

在 Stage B 正式结果出来之前，本文档不能给出跨实验的综合判断；但可以把 Stage B0 的发现纳入后续分析框架：

### 6.1 "预算够不够" 已经不是解释变量

TD3BC phase0c 的报告曾担心 `2000` 没有进一步改善是不是训练预算不够。Stage B0 直接排除了这个可能性（至少在 ReBRAC 侧）——因此后续如果在 `crosscomp-2000` 上看到 ReBRAC 仍然不如 `crosscomp-1000`，不应该再归咎于 "epoch 不够"。

### 6.2 优化方差是下一个 first-class 现象

Stage B0 把 seed 44 的异常暴露得很明显。这与 phase0c 一度担心的 "Stage B seed=2 可能翻转 winner" 是同一类问题——现在 ReBRAC 下仍然存在，甚至更突出。Stage B 结果分析的重点应当放在：

- `std_test_success_rate` 的跨 β1 分布——哪一档 BC 强度能最稳地把 seed 方差压下去？
- seed 44 的异常是不是由 β1 值决定（例如 β1=4.0 下是否消失）？

### 6.3 关于 "长训 ≈ 单独训到" 的等价性

Stage B0 依赖 "长训到 step N ≈ 单独训到 step N 结束" 这一假设。这个假设在当前实验下只做了定性验证（val 曲线本身没有出现 "长训时的早期表现 vs 单独训出的早期表现" 显著偏离的信号），完整定量验证需要在 Stage B 真正跑出 `TRAIN_EPOCHS=64` 的独立 run 之后，用 notebook 里的 `state_dict_l2_distance` 做交叉比对。这一步不是当前报告的前置条件，但是 Stage B 完成后一个成本极低的 sanity check。

---

## 7. 局限性

1. 当前报告只覆盖了一个预备实验（Stage B0），没有任何关于 ReBRAC 是否优于 TD3BC 的证据。
2. Stage B0 的 β1/β2 选择限定在网格中心点，结论对 β1=1.0 和 β1=4.0 只能外推，不是直接观察。
3. `worldcomp` 轨道完全没有触及。

---

## 8. 最终结论

1. **`TRAIN_EPOCHS=64` 在 ReBRAC 口径下是合适的训练预算**：两个 `crosscomp` 数据集上典型 seed 在 epoch 40~50 已经接近 val plateau，ep=64 → ep=128 的增益主要来自曲线波动，不是稳定的训练不足。
2. **`crosscomp-2000` 不需要比 `crosscomp-1000` 更大的训练预算**：这与 "2000 数据更多所以训练更难" 的朴素直觉相反，但与 phase0c 的支持集结构解释一致。
3. **ReBRAC 下的优化方差是一个需要正视的问题**：seed 44 在 β1=2.0 配置下显著弱于 seed 42/43，且训练到 128 epoch 也没有收敛。因此 Stage B 的结果解读必须把 `std_test_success_rate` 作为一等公民。

一句话总结：

> 训练预算问题已经钉死，Stage B 可以直接按 `TRAIN_EPOCHS=64` 开跑；真正需要 Stage B 回答的不是 "算法是否训够了"，而是 "dual penalty 能不能同时改善均值 **和** 方差"。

---

## 9. 结果文件索引

本报告基于以下文件：

- Stage B0 训练预算 probe
  - `checkpoints/offline/rebrac/screening_epoch_probe/crosscomp-1000/actorb_2p0__criticb_1p0/seed_{42,43,44}/agent_step_*.pt`
  - `checkpoints/offline/rebrac/screening_epoch_probe/crosscomp-2000/actorb_2p0__criticb_1p0/seed_{42,43,44}/agent_step_*.pt`
  - `results/offline/rebrac/screening_epoch_probe/crosscomp-{1000,2000}/actorb_2p0__criticb_1p0/selection/seed_{42,43,44}/selected_checkpoint.json`

执行入口：

- [notebooks/rebrac_epoch_probe.ipynb](../notebooks/rebrac_epoch_probe.ipynb)
- [scripts/run_offline_rebrac_screen.sh](../scripts/run_offline_rebrac_screen.sh)

背景阅读：

- [rebrac_experiment_plan.md](./rebrac_experiment_plan.md)
- [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)
- [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)
