# ReBRAC 广验报告：跨数据质量 / 传感器 / 任务三轴的 Probe-then-Deepen

> 文档版本：2026-05-07 rev.2（集成 C1-s1 sensor upgrade follow-up）
> 文档定位：[rebrac_experiment_plan.md](./rebrac_experiment_plan.md) Stage C 主线已 5-seed 收口（finalist `(β1=4.0, β2=2.0)` 在 `crosscomp-1000` 上 success = 0.902 ± 0.021）、[rebrac_mainline_review.md](./rebrac_mainline_review.md) §2.2/§3.2 留下的三条 generality 弱点之后做的一轮**有限算力广验**。
> 配套文档：[broad_validation design spec](./superpowers/specs/2026-05-04-rebrac-broad-validation-design.md)、[broad_validation plan](./superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md)
> 阅读顺序：先读 §1–§3 拿主结论；§4 是 discussion 与 working hypotheses；§5 是 station headline、broad validation 找到的限定条件、future work；§6 是与 paper 主线的 cross-link 索引。
>
> ---
>
> **★ Status（2026-05-07 user direction）**：本报告**保持 standalone exploratory side study**，**不回写**主报告 [`rebrac_experiment_report.md`](rebrac_experiment_report.md) 与 strategic review [`rebrac_mainline_review.md`](rebrac_mainline_review.md) 的 finding spine。`mainline_review §3.5` 现有的顶层指针保留（不复制内容），`experiment_report §10A` 已退回为 pointer（删除 cf5cfff 的 sensor-floor framing）。
>
> **理由**：8 spoke 中只有 B1 sensor envelope (Δ=−0.2pp, std=0.028) 是 clean positive；其余 7 个 work-in-progress：A1 underpowered (paired t p≈0.11)、A2 mid-gap collapse 的 mode-collapse 机制是 hypothesis（未做直接 ablation）、A3 5-seed 反转 1-seed 决策、B2/C3 std blow-up 无 mechanism ablation、C1 task-fundamental floor 的 BC penalty 强度 sweep 未做。把还在动的 commentary 钉死进静态 finding spine 会把后续 sweep 的 churn cost 转嫁给主线。
>
> **Retrofit trigger condition**（满足后再考虑统一回写）：
> 1. **C1 BC penalty 强度 sweep**（β1 ∈ {0, 1, 2, 4, 8}）— 区分 BC floor vs task-fundamental floor，§5.3 future work #1
> 2. **A1 paired bootstrap (n_boot=10000)** 或 **mix ratio sweep on A2** — 区分 directional consistent 与 statistically robust，§3.3 + §3.2
> 3. **target_speed=2.0 P1 probe** — 测试 task-fundamental claim 的 speed-axis 边界，§5.3 future work #2
>
> 上述 trigger 闭环之前，paper §experiments / §discussion 引用本报告时仅以 cross-link 形式（不复制数字）。

---

## 0. Abstract

**动机**：ReBRAC 主线 finding 严密成立但只覆盖一个 cell（`crosscomp-1000 / s0 / cross_stream / Re150 / U=1.0 / target=1.5`）。审稿人会问的三件事——数据质量退化是否仍优？传感器更丰富是否仍优？任务几何 / 多尾流场景是否成立？——主线没有数据。

**方法**：从 Stage C finalist `(β1=4, β2=2)` 作为 anchor，在数据质量 (A) / 传感器 (B) / 任务几何 (C) 三轴上各拉 2–3 条 spoke，2-seed probe → 触发判据 → 5-seed deepen → 1-seed β refit → 5-seed parity。共 8 个 representative cell，全部 5-seed parity。

**主结论**：

1. **C 轴 upstream geometry (C1) 上 algorithm + sensor 共 4 类 5 个 ablation 全部未恢复 success → task-fundamental floor**（reward swap n=2 / asym critic n=2 / epoch ×4 single seed / **deployable sensor upgrade s0→s1 n=2** / convergence check at 64 ep）。baseline 5-seed = 0.218 ± 0.008 是 deterministic-bad 坍缩（std/anchor_std = 0.36×）。**5 个独立配置全部钉在 0.195–0.225 (3pp range)** 是强 cross-config 一致性证据；reward 与 sensor 各自独立 modulate failure mode (timeout-only 77.5/0 ↔ mixed ~53/~26)，但都不动 goal-reaching ceiling → ceiling 来自更上游的 task / data 物理上限，不是 actor 探索风格的 function。机制候选（reward 失配 / BC penalty 限制 / task-data 上限）尚未通过 BC penalty sweep 直接区分；online SAC 同 task 下也失败 → 至少部分原因在 task/reward 一侧。详见 [c1-s1 sensor upgrade follow-up](./rebrac_c1_s1_followup_report.md)。
2. **B 轴 sensor s1 (B1) robust**：Δ=−0.2pp、std=0.028 ≈ anchor std (1.35×)。**B2 (s2) mean-on-anchor 但触发 std_blow_up**（std=0.058，2.78× anchor）。Sensor envelope claim 应限定为「s0/s1 等效；s2 需要扩 seed 预算或 sensor-aware regularization」。
3. **A 轴 collector quality 敏感**：A1 goalseek 弱数据下 refit_b (4.0, 1.0) 比 anchor (4.0, 2.0) 平均高 4.0pp，**directional consistent (4/5 paired diffs 非负) 但 paired t=2.05 (df=4, p≈0.11) 未达 5% 显著线**；A3 anchor 与 refit_b 在 5-seed 下统计上不可区分（paired t=−1.23, p≈0.29）。
4. **A2 mid-gap collapse — broad validation 新发现的边界 case**：主线 crosscomp-1000 上 ReBRAC vs TD3+BC = +23pp（Stage C），广验 mix5050-1000 上 = **+0.2pp（matched 5v5）**。两个算法 std 都展开到 0.14–0.16。**这不是修订已有 paper claim，也不是反例**——是 broad validation 揭示 ReBRAC quantitative advantage 的 data-coherence-dependent 边界。机制（双峰 action / mode collapse）是 working hypothesis，未做直接 ablation。

**Future work**：C1-s2 (4 probes 16-D) backlog（s0→s1 已闭环，sensor 不是 lever 的证据已就位）；target_speed=2.0 单 seed P1 probe；BC penalty 强度 sweep（区分 BC floor vs task-fundamental floor）。

---

## 1. Introduction

### 1.1 主线只占了一格

ReBRAC 主线（[rebrac_experiment_plan.md](./rebrac_experiment_plan.md) §0–§7）的所有四条 paper-level finding 都建立在以下单一 cell 上：

| 字段 | 值 |
|---|---|
| collector | `crosscomp` (CrossCurrentCompensationPolicy) |
| dataset size | 1000 episodes |
| probe layout | `s0`（DVL 单点） |
| task geometry | `cross_stream` |
| flow | `wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| target speed | 1.5 |
| reward objective | `efficiency_v2` |

`worldcomp-1000` 是 deployable→teacher gap 的辅助验证轴，**不是** generality 覆盖。审稿人会指出：generality 是单点证据。

### 1.2 三轴广验框架

| 轴 | spoke | 唯一改动 | 其余配置 |
|---|---|---|---|
| **A** 数据质量 | A1 goalseek / A2 mix5050 / A3 privileged | collector policy | 严格沿用 anchor |
| **B** 传感器 | B1 s1 / B2 s2 | probe layout | 严格沿用 anchor |
| **C** 任务几何 | C1 upstream / C3 tandem | task geometry + flow | 严格沿用 anchor |

每条 spoke 改且仅改一个轴；anchor 本身不重跑（5-seed 已存在）。A2 额外加 TD3+BC × 5-seed 作为 ReBRAC vs BC-baseline 的 head-to-head 锚点。

### 1.3 触发判据（spec §6.1）

| 判据 | 阈值 | 触发含义 |
|---|---|---|
| `mean_shift` | \|spoke mean − anchor mean\| > 5pp | 显著差异（正负皆触发） |
| `std_blow_up` | spoke std > 2 × anchor std (= 0.042) | 不稳定 |
| `td3bc_gap_collapse` (仅 A2) | \|ReBRAC − TD3+BC\| < 5pp | 算法区分度坍塌 |

触发的 spoke 进入 P2：1-seed β refit 选 winner → winner 5-seed 扩展。

---

## 2. Experimental Protocol

### 2.1 四阶段流程

| 阶段 | 内容 | run 数 | L4 时长 |
|---|---|---:|---:|
| **S1** sanity | 7 个新 dataset 收集 + sanity_card | 7 dataset | ~6h |
| **S2 P1** probe | 7 spoke × 2 seeds + A2 TD3+BC × 2 seeds | 16 | ~8h |
| **S3.b** 1-seed β refit | 5 触发 spoke × 2 配置 (β1=2,β2=2) (β1=4,β2=1) | 10 | ~5h |
| **S3.c** 5-seed deepen | winner cell × 补 [43,45,46] | 15 | ~7.5h |
| **S3.d** 5-seed parity | A1/A2-td3bc/A3/B1/B2 anchor cell × 补 [43,45,46] | 15 | ~7.5h |
| **总计** | — | **63 run** | **~34h L4** |

实际预算（含 C1 ablation 4 子研究）约 50h L4。

### 2.2 Skip-resume 与可复现性

驱动脚本 `scripts/run_offline_rebrac_broad.sh` 通过 `trainer_state.json + agent_final.pt` 双文件存在性判断决定是否跳过；validation/selection/test 各自独立 skip。所有结果路径固定为 `results/offline/rebrac/broad_validation/<spoke>/<pair>/test/seed_*.json`，summarize 工具从磁盘扫码自动合并。

### 2.3 Anchor 与统计基线

Anchor (Stage C finalist `(β1=4.0, β2=2.0)`)：

| 指标 | 值 |
|---|---|
| mean success | 0.902 |
| std success | 0.021 |
| n seeds | 5 (42–46) |
| test episodes | 100 |

广验所有 spoke 用同一 test manifest（100 episodes）+ 同一 selection 规则（success → return → -safety → -time）。

---

## 3. Results

### 3.1 全表（5-seed parity 后）

| Axis | Spoke | Pair | n | mean succ | std succ | mean R | Δ vs anchor | Trigger |
|---|---|---|---|---|---|---|---|---|
| A | A1 | actorb_4p0__criticb_1p0 (refit_b) | 5 | 0.794 | 0.029 | −99.9 | **−10.8pp** | mean_shift |
| A | A1 | actorb_4p0__criticb_2p0 (anchor) | 5 | 0.754 | 0.047 | −115.2 | −14.8pp | — |
| A | A2 | actorb_4p0__criticb_2p0 (anchor) | 5 | 0.606 | **0.157** | −111.8 | **−29.6pp** | mean_shift, std_blow_up, td3bc_gap_collapse |
| A | A2-td3bc | alpha_0p25 | 5 | 0.604 | **0.141** | −120.9 | −29.8pp | (matched 5v5) |
| A | A3 | actorb_4p0__criticb_1p0 (refit_b) | 5 | 0.734 | 0.053 | +4.2 | **−16.8pp** | mean_shift, std_blow_up |
| A | A3 | actorb_4p0__criticb_2p0 (anchor) | 5 | 0.752 | 0.039 | +10.7 | −15.0pp | — |
| B | B1 | actorb_4p0__criticb_2p0 | 5 | **0.900** | **0.028** | −43.3 | **−0.2pp** | (untriggered) |
| B | B2 | actorb_4p0__criticb_2p0 | 5 | 0.860 | 0.058 | −58.1 | **−4.2pp** | std_blow_up |
| C | C1 | actorb_4p0__criticb_2p0 | 5 | **0.218** | **0.008** | −374.1 | **−68.4pp** | mean_shift |
| C | C3 | actorb_4p0__criticb_2p0 | 5 | 0.576 | 0.148 | −90.7 | **−32.6pp** | mean_shift, std_blow_up |

(粗体 = report headline 引用；Anchor: 0.902 ± 0.021)

#### Axis-level takeaway

- **A 轴**：collector quality 敏感，从 goalseek (mean R=−100, std=0.03) 到 privileged (mean R=+11, std=0.04 anchor / 0.05 refit_b) 单调向好；mix5050 (A2) 是 std blow-up 主体，并把 ReBRAC vs TD3+BC 的区分度拉到 0pp。
- **B 轴**：B1 (s1) 几乎完美对齐 anchor；B2 (s2) mean 接近但 std 翻倍——sensor 越丰富 BC 与 RL 信号耦合越敏感。
- **C 轴**：C1 (upstream) 是结构性失败（−68.4pp、std=0.008），C3 (tandem) 是 mild geometry shift 下的 high-variance 退化（−32.6pp、std=0.15）。

### 3.2 A2 mid-gap collapse — ReBRAC vs TD3+BC 优势的 data-coherence boundary ★

主线 Stage C 在 `crosscomp-1000` 上的正式成绩（[experiment_report.md](./rebrac_experiment_report.md) §6.x）：ReBRAC finalist `(β1=4, β2=2)` = **0.902 ± 0.021**，TD3+BC `phase0c` 在同 dataset 同 seeds = **0.672**（推算自 +23.0pp gap），即 **+23pp ReBRAC quantitative advantage**。

广验 A2 把场景换到 `mix5050`（episode-level 50% goalseek + 50% crosscomp）：

| Algorithm | n | mean | std | mean R |
|---|---|---|---|---|
| ReBRAC `(β1=4, β2=2)` | 5 | 0.606 | 0.157 | −111.8 |
| TD3+BC `α=0.25` | 5 | 0.604 | 0.141 | −120.9 |
| **Δ** | — | **+0.20pp** | — | +9.1 |

|Δ| = 0.2pp 远小于 5pp 阈值 → spec §6.1 触发 `td3bc_gap_collapse`。

**Cross-quality 对比**：

| dataset | ReBRAC | TD3+BC | Δ |
|---|---|---|---|
| crosscomp-1000 (Stage C, n=5) | 0.902 ± 0.021 | 0.672 (+23pp 主线对照) | **+23.0pp** |
| mix5050-1000 (broad val A2, n=5) | 0.606 ± 0.157 | 0.604 ± 0.141 | **+0.20pp** |

**对比 2-seed snapshot**：S2 P1 阶段 ReBRAC=0.730、TD3+BC=0.660、Δ=+7pp。但 5-seed 下两者均出现 std≈0.15 的方差展开，2-seed 看到的 +7pp 完全在 mean 估计的 sampling error 内。

**Broad validation 新发现**（既不是修订 paper rev.8 已有 claim，也不是反例，而是 generality 的 boundary 探测）：

> "ReBRAC's quantitative advantage over single-BC TD3+BC, observed at +23pp on `crosscomp-1000` (Stage C), does not extend to behaviorally noisy datasets. On `mix5050-1000` (50% goalseek + 50% crosscomp at episode level), both algorithms collapse to mean ≈ 0.60 ± 0.15, with Δ < 1pp at matched 5 seeds. We **hypothesize** (without further ablation here) that this reflects a shared BC-regularizer floor: when behavior data carries multimodal action distributions inconsistent with a single optimal policy, both single- and dual-penalty BC degenerate to similar mode-collapsed solutions. Paper-level statement should therefore be scoped as: ReBRAC outperforms TD3+BC on coherent (single-mode) behavior data; on incoherent multimodal data the relative advantage is data-quality-dependent."

> ⚠ "BC noise floor / mode collapse" 的机制解释是 working hypothesis，未做直接 ablation 验证（例如 50/50 比例改成 80/20、或在 mix dataset 上做 BC penalty 强度 sweep）。Paper §experiments 引用时应明确标记为「interpretation」。

### 3.3 A 轴 refit decisions at 5-seed parity ★

S3.b 阶段对每个触发 spoke 做了 1-seed β refit（在 seed 42 上跑 (β1=2,β2=2) 与 (β1=4,β2=1)）。winner select 阈值 = +3pp（spec §6.2）。1-seed 决策与 5-seed 实测：

| Spoke | 1-seed (S3.b) anchor | 1-seed refit_b | 1-seed winner | 5-seed anchor | 5-seed refit_b | 5-seed winner | 是否一致 |
|---|---|---|---|---|---|---|---|
| **A1** | 0.75 | 0.84 | refit_b | 0.754 ± 0.047 | **0.794 ± 0.029** | refit_b | ✓ 稳 |
| **A3** | 0.78 | 0.81 | refit_b | **0.752 ± 0.039** | 0.734 ± 0.053 | anchor (反转 1.8pp) | ✗ 反转，但在 std 内 |
| A2 | 0.69 | 0.66 | anchor (drift<3pp) | 0.606 ± 0.157 | — | anchor | ✓ 稳 |
| C1 | 0.22 | 0.24 | anchor (drift<3pp) | 0.218 ± 0.008 | — | anchor | ✓ 稳 |
| C3 | 0.72 | 0.55 | anchor | 0.576 ± 0.148 | — | anchor | ✓ 稳 |

**A1 — directional consistent 但 underpowered**：5-seed paired t-test (matched seeds 42–46)，mean_diff = +4.0pp，sd_diff = 0.044，**t = 2.05 (df=4)**，t_crit(α=0.05) = 2.78 → **p ≈ 0.11，未达 5% 显著线**。但 5/5 seeds 中有 4 个 refit_b ≥ anchor（diffs = [+0.09, +0.05, −0.01, +0.07, 0.00]），方向一致性强；refit_b std=0.029 也比 anchor std=0.047 低。**保守读法**：refit_b 在 A1 上**很可能优于 anchor 但 n=5 下未达统计显著**；做 paired bootstrap (n_boot=10000) 是后续可选 sanity，不影响报告级别 winner select。

**A3 — 方向反转且不显著**：5-seed paired t-test，mean_diff = −1.8pp，sd_diff = 0.033，t = −1.23 (df=4)，p ≈ 0.29 → 双向皆不显著。诚实读法是「A3 anchor 与 refit_b 在 5-seed 下统计上不可区分」。1-seed refit 决策在此处是 fragile——若广验只跑到 P2 1-seed，会得出错误的 refit_b 占优结论。

**Methodology limitation**（待写入 spec rev.2）：1-seed refit_b winner select 在 anchor std=0.05 量级上有 ~30% 概率反转。建议未来广验 spec 把 winner select 改为 2-seed minimum 或保留双 cell 各 5-seed 直到诚实选择。

### 3.4 B 轴 sensor envelope ★

| spoke | layout | n_probes | obs_dim | mean | std | Δ | 结论 |
|---|---|---|---|---|---|---|---|
| anchor | s0 | 1 (DVL) | 10 | 0.902 | 0.021 | — | baseline |
| **B1** | s1 | 2 (DVL+ADCP-short) | 12 | **0.900** | **0.028** | **−0.2pp** | ✅ robust |
| **B2** | s2 | 4 (DVL+ADCP-long) | 16 | 0.860 | **0.058** | −4.2pp | ⚠ variance-sensitive |

**B1 是 sensor envelope 的硬证据**：success 几乎与 anchor 相同 (Δ=−0.2pp)，std 仅 1.3 × anchor std。`untriggered` 是 spec §6.1 三条判据下的真正阴性。**这条数字可以直接进 paper §experiments**：「s1 sensor (DVL + 2 MHz short-range ADCP) 与单 DVL (s0) 在 anchor task 上 success 等效（Δ=−0.2pp，n=5）」。

**B2 是 limited claim**：mean 仍在 anchor 附近（Δ=−4.2pp，未触发 mean_shift）但 std=0.058 触发 `std_blow_up`。

> ⚠ 候选机制（hypothesis，未做直接 ablation）：s2 obs 是 16 维 `(u, v) at 4 probes`，包含 (5, 0)、(8, ±4) 三个前置探针，sensor 的「上行预报」性质（~7 步前置）使探针在 cross_stream 任务里测到的流场远早于 ego 进入该区域。**可能**导致 BC 信号（基于历史 obs-action）与 RL signal 在 16 维 obs 空间里耦合更复杂 → seed 间收敛分布展开。直接验证应是 critic LayerNorm 强度 / hidden_dim sweep on B2，未做。

**报告级别 sensor envelope claim**（保守）：「ReBRAC 在 s0/s1 上 success 等效（B1 Δ=−0.2pp，n=5）；s2 (B2) mean 接近 anchor 但 seed-level variance 翻倍（std=0.058 vs anchor 0.021）。Sensor 升级到 s2 时建议配套增加 seed 预算 / 检视 critic 正则化设置。」

### 3.5 C1 deep-dive — 5 个 ablation 全部未恢复 success → task-fundamental floor

C1 (upstream geometry) 是 5-seed 下最严重的退化（Δ=−68.4pp）。**std=0.008（ratio 0.36×anchor，5 seed 都坍缩到 [0.21, 0.23]）的零方差信号**强烈暗示这是 deterministic 的次优收敛，不是 stochastic noise。我们做了 5 个独立 ablation（4 个 algorithm-side + 1 个 sensor-side）排除常见解释假设。

> ⚠ **重要 dataset 范围说明**：§3.5.1–§3.5.4 的 4 个 ablation **均在 reward 替换后的 dataset `arrival_v2_simple` 上进行**（路径 `c1_reward_ablation/`、`c1_asym_critic_ablation/`、`c1_epoch_sensitivity/`），不是 C1 baseline 的原 `efficiency_v2`。这是因为 reward ablation 是第一个跑的，其 train_log/checkpoint 被后续 ablation 复用。结论严格上是「在更简单的 reward (`arrival_v2_simple`) 下，asym critic / epoch / convergence 都不能突破 0.22」。**§3.5.5 sensor upgrade 例外**：直接基于 `efficiency_v2` (与 P1 anchor 同 reward)，是唯一只切换 sensor 一个变量的 clean comparison。

#### 3.5.1 Reward ablation（C1 baseline → `arrival_v2_simple`）

替换 `efficiency_v2` 为更简单的 `arrival_v2_simple`（线性 progress + arrival bonus，去掉 speed-tracking penalty），其余配置不变：

| seed | success | mean R |
|---|---|---|
| 42 | 0.200 | −102.3 |
| 44 | 0.230 | −94.2 |
| **mean (n=2)** | **0.215** | **−98.3** |
| Δ vs C1 baseline 5-seed (0.218 ± 0.008) | **−0.3pp** | +5.8 |

观察：n=2 reward swap 落在 baseline 5-seed 的 0.218 ± 0.008 噪声带内。**弱阴性 flag**：reward swap 没有把 success 抬出 baseline 区间，但 n=2 不足以做强阳性「reward shaping 完全无效」的结论；不能排除 reward 还有更激进的改写空间。

#### 3.5.2 Asymmetric critic ablation（基于 `arrival_v2_simple`）

打开 `--use-asymmetric-critic`，把 hull-integral `[u_eq, v_eq]` 喂给 critic（actor 仍只看 s0 单点）：

| seed | success | mean R |
|---|---|---|
| 42 | 0.210 | −106.9 |
| 44 | 0.180 | −122.9 |
| **mean (n=2)** | **0.195** | **−114.9** |
| Δ vs reward-ablation baseline (0.215) | **−2.0pp** | −16.6 |

观察：与 reward ablation baseline (0.215, n=2) 比，asym critic mean 降低 2pp 但仍在 baseline 5-seed std=0.008 的 ±2.5σ 内 → 在 n=2 下无法区分「轻微恶化」与「噪声」。**弱阴性 flag**：privileged critic 信息没有把 success 抬到 0.30+，与「actor 信息瓶颈」假设不符。

#### 3.5.3 Epoch sensitivity（基于 `arrival_v2_simple`，single seed=42）

固定 seed 42、anchor β、`arrival_v2_simple`，scan TRAIN_EPOCHS ∈ {64, 128, 192, 256}：

| epochs | success | mean R |
|---|---|---|
| 64 | 0.200 | −102.3 |
| 128 | 0.200 | −106.3 |
| 192 | 0.220 | −96.2 |
| 256 | 0.220 | −96.2 |

观察：训练长度从 64 加到 256（4×）下 success 从 0.20 拉到 0.22。给定 baseline 5-seed std=0.008，single-seed 下 0.20→0.22 的变化处于噪声带边缘 → **延长训练对 success 改善有限**。注意这是 **single seed**，无法做总体 mean shift 检验。

#### 3.5.4 Convergence check（loss curves at 64 epochs，基于 `arrival_v2_simple`）

详见 [notebooks/rebrac_c1_train_convergence_check_completed.ipynb](../notebooks/rebrac_c1_train_convergence_check_completed.ipynb)。在 train_step 1 → 67000（即 64 epochs）的 train_log 上做 mid-epoch vs late-epoch metric 比较（threshold = 5% rel_change for losses, 3% for Q estimates）：

| metric | mean_mid | mean_late | rel_change | plateau? |
|---|---|---|---|---|
| critic_loss | +10.25 | +6.42 | −37.5% | ❌ |
| actor_loss | −0.71 | −0.82 | −15.7% | ❌ |
| bc_loss | +0.021 | +0.017 | −15.7% | ❌ |
| td_abs_error | +1.73 | +1.62 | −6.6% | ❌ |
| mean_q | +44.6 | +64.6 | **+44.8%** | ❌ |
| target_q | +44.6 | +64.7 | **+45.1%** | ❌ |

**plateau hits: 0/6** — 笔记本 verdict 是「H1 (epochs 不够) 成立：64 epochs 下 6/6 metric 仍在显著移动」。

**与 §3.5.3 联合解读**：64 epochs loss 仍在大幅移动 + 64→256 epoch 实测 success 只从 0.20 拉到 0.22 → **loss 仍在动但 success metric 已基本饱和**，loss 移动与 policy quality 在 256 epochs 之后已大幅解耦。这意味着「epochs 不够」单独不能解释 C1 失败。

#### 3.5.5 Sensor upgrade（s0 → s1，唯一保 reward 不变的 clean comparison）

升级 deployable sensor 从 s0 (1 probe DVL water-track, 10-D) 到 s1 (2 probes DVL + 短程 ADCP, 12-D, ~3 步前向流场 advance warning)，**保 `efficiency_v2` reward 与 P1 anchor 完全一致**：

| seed | success | mean R | termination (goal/timeout/oob) |
|---|---|---|---|
| 42 | 0.210 | −379.98 | 21 / 53 / 26 |
| 44 | 0.200 | −383.42 | 20 / 54 / 26 |
| **mean (n=2)** | **0.205 ± 0.005** | **−381.7** | **20.5 / 53.5 / 26.0** |
| Δ vs P1 anchor (s0, 5-seed 0.225 ± 0.005) | **−2.0pp** | — | — |

观察：sensor 升级 −2.0pp 落在 ±1.5pp single-run 噪声半径内 → **sensor 信息量增加不能突破 ceiling**。pre-committed verdict gate（≥0.50 解锁 / 0.30–0.50 部分 / <0.30 task-fundamental）触发 **<0.30** 档 → **sensor 不是 lever，边界不在 sensor 维度**。

**Termination distribution 跨 5 配置一致性**：

| config | reward | sensor | n | goal | timeout | oob |
|---|---|---|---|---|---|---|
| P1 anchor | eff_v2 | s0 | 5 | 22.5 | **77.5** | **0.0** |
| Abl A (reward swap) | arr_v2_s | s0 | 2 | 21.5 | 52.5 | 26.0 |
| Abl B (asym critic) | arr_v2_s | s0 (asym) | 2 | 19.5 | 48.5 | 32.0 |
| Abl C (epoch 256) | arr_v2_s | s0 | 1 | 22.0 | 52.0 | 26.0 |
| **C1-s1 (sensor up)** | **eff_v2** | **s1** | **2** | **20.5** | **53.5** | **26.0** |

**关键发现**：C1-s1 与 P1 **完全同 reward** (eff_v2)，仅 sensor 升级 s0→s1 就把 0 oob 翻到 26 oob、timeout 从 77.5 掉到 53.5——**sensor 与 reward 都是 failure-mode 的独立 driver**（两条不相交路径都能把 timeout-only 翻到 mixed）。但 5 个 configs 的 success 全部钉在 **0.195–0.225 (3pp range)**，比 single-run 噪声半径 ±1.5pp 还要窄 → **goal-reaching ceiling 是更上游的 task-data 物理上限，不是 actor 探索风格的 function**。

详见 [c1-s1 sensor upgrade follow-up](./rebrac_c1_s1_followup_report.md)（standalone report，含 verdict gate / per-seed 细节 / paper-narrative 升级建议）。

#### 3.5.6 综合判定 — 5 个 ablation 都未把 success 抬到 0.30+ → task-fundamental floor

| ablation (n) | success | 排除什么 | 留下什么 |
|---|---|---|---|
| C1 baseline `efficiency_v2` 5-seed | 0.218 ± 0.008 | — | basecase |
| reward swap `arrival_v2_simple` (n=2) | 0.215 | reward complexity 不是主因 | task-reward 失配的更深层 |
| + asym critic on (n=2) | 0.195 | actor 信息瓶颈不是主因 | actor / critic 共同的策略 cap |
| + epoch 64→256 (n=1, seed 42) | 0.20→0.22 | undertraining 不是主因 | loss-success 解耦 |
| + convergence check at 64 ep | plateau 0/6 | "loss 已收敛" 假设不成立 | loss 移动持续但不改善 success |
| **+ sensor s0→s1 (n=2, eff_v2)** | **0.205** | **sensor 维度不是 lever** | **task-data ceiling 不在 actor 输入信息侧** |

**5-config cross-consistency**：所有 5 个独立干预（reward / critic structure / training budget / sensor dimension / convergence）都钉在 0.195–0.225 (3pp 全幅，<2× single-run 噪声半径) → **task-fundamental floor verdict**：边界来自 `crosscomp / upstream / u10 / target_speed=1.5` 这个 task-dataset 组合的物理上限，不是任何单一 lever 的 function。

**做不出阳性的事**：5 个 ablation 都没有把 C1 success 抬到 baseline 5-seed (0.218) 之上的统计可区分水平。**没有做的事**：(1) BC penalty 强度 sweep（β1 ∈ {0, 1, 2, 4, 8}）— 关键的 mechanism discriminator；(2) `target_speed=2.0` 单 seed P1 probe；(3) C1-s2 (4 probes 16-D) sensor envelope 上限。

**Working hypothesis（机制解释，未做直接验证）**：`crosscomp/upstream` collector 在 reward `efficiency_v2` 下的最优行为可能是「全功率直行 + barely-success」，导致 dataset 内 action distribution 退化为狭窄单峰，BC penalty 把 actor 钉在该次优策略上。换 reward (3.5.1) 或放松 critic 瓶颈 (3.5.2) 或加 sensor 信息 (3.5.5) 都不动 success——若假设成立，原因是 BC penalty 项本身（actor 与 critic 双侧）。这条 hypothesis 的直接验证应是 **BC penalty 强度 sweep**，未做。

**保守级别的报告 claim**：

> "在 5 个独立 ablation（reward swap / asym critic / epoch ×4 / convergence check / sensor s0→s1）下，C1 (upstream + crosscomp + target_speed=1.5) success 都没有抬出 baseline 5-seed 的 0.218 ± 0.008 噪声带，5-config cross-consistency 钉在 0.195–0.225 (3pp range) → **task-fundamental floor at deployable sensors s0/s1 and target_speed=1.5**。**没有观察到能恢复 success 的算法 / sensor 侧调节**；机制（task-reward 失配 vs BC penalty 限制 vs 其他）需要 BC penalty 强度 sweep 直接区分。online SAC 在同 task 下也失败（[online_sac_reward_redesign.md](./online_sac_reward_redesign.md)）→ 至少部分原因在 task / reward 这一侧。"

**对 paper 的含义**：C1 是 broad validation 揭示的 **deployment-impossible task-dataset combination**——offline RL 在 `crosscomp/upstream/u10/target_speed=1.5` 上不可解，5 个常见 escape route（4 algorithm + 1 sensor）都没救回来。reward 与 sensor 各自独立 modulate failure mode 但都不动 ceiling 是更强的 task-fundamental 信号。这是有价值的 negative finding，作为 paper §experiments 的 deployability boundary case：与 Stage D Phase 2 的 `cross_stream + worldcomp` deploy-graded 区间对比，构成完整的 deployability 谱系（哪些 task algorithmic 可解 / 哪些是物理边界）。

---

## 4. Discussion

### 4.1 BC-regularizer 共同 floor 假设（A2）— working hypothesis

A2 mid-gap 5v5 显示 ReBRAC 与 TD3+BC 在 noisy mix5050 上 std 都展开到 0.14–0.16（anchor std=0.021 的 6.7–7.5×）。两个算法只差 0.2pp。

> ⚠ 本节是机制 hypothesis，**未在本广验内做直接 ablation 验证**（如 mix 比例 sweep、BC penalty 强度 sweep on mix dataset）。仅作为对观测的 candidate 解释。

候选机制：
- ReBRAC 的 dual penalty (actor BC + critic BC) 与 TD3+BC 的 single penalty (actor BC) 都假设 behavior data 提供「单峰自洽」的 policy 信号；
- mix5050 的 50% goalseek + 50% crosscomp 在 cross_stream task 下**可能**产生双峰 action distribution——同样 obs 下 goalseek 直冲目标、crosscomp 沿流偏航；
- 若假设成立，BC penalty 会把 actor 拉向两峰中点，中点既非 goalseek 最优也非 crosscomp 最优；critic-side BC penalty (ReBRAC) 在 mode collapse 中点上也学不出有意义的 advantage；
- → **若此假设成立**，两种 BC 在双峰 data 上坍缩到同一个 mode-collapsed policy 是预期的；这与 paper 主线 cell 上 ReBRAC vs TD3+BC = +23pp 的 advantage 相容（主线 collector 是单一 crosscomp，BC signal 单峰自洽）。

**直接验证路径**（未做，应在 paper revision 阶段考虑）：
1. 在 mix dataset 上对 actor/critic 输出做 KL / mode-counting，验证 mix 是否真的产生双峰 action distribution；
2. mix 比例 sweep（70/30, 90/10）观察 std blow-up 是否单调；
3. 在 mix 上 BC penalty 强度 sweep，检查能否通过减弱 BC 让 actor 跳出中点。

### 4.2 Variance asymmetry — std as scenario signature

按 std / anchor_std 比例（spec §6.1 std_blow_up 阈值 = 2.0）分组：

| ratio bucket | spoke / pair | std | mean | trigger? |
|---|---|---|---|---|
| **tight (<1×)** | C1 baseline (0.008) | 0.36× | 0.218 | — |
| **OK-anchor (1–2×)** | B1 (0.028) / A1 refit_b (0.029) / A3 anchor (0.039) | 1.35–1.84× | 0.900 / 0.794 / 0.752 | 未触发 std_blow_up |
| **mild blow-up (2–3×)** | A1 anchor (0.047) / A3 refit_b (0.053) / B2 (0.058) | 2.23–2.78× | 0.754 / 0.734 / 0.860 | 触发 |
| **severe (>5×)** | A2-td3bc (0.141) / C3 (0.148) / A2 (0.157) | 6.73–7.49× | 0.604 / 0.576 / 0.606 | 触发 |

观察（部分是 hypothesis）：

- **C1 std=0.008**（实际 ratio 0.36×anchor）— 极低 std + 低 mean 是 deterministic-bad 信号（5 seed 都坍缩到同一坏吸引子）；
- **A2 / A2-td3bc / C3 std≈0.15**（severe）— mid-quality 数据 + mild geometry shift 都进入 severe blow-up bucket；
- **B1 std=0.028**（healthy）/ A1 refit_b std=0.029（healthy）— 两个 sensor s0/s1 robust + goalseek 上 critic_β 调小后的健康收敛分布；

报告级别的实践含义：**未来 broad validation spec 应把「std < 0.5 × anchor_std」也作为 deterministic-failure flag**（即双向 std 阈值，不只看 blow-up），与 mean_shift 一起评估即可识别 C1 类的低方差坍缩。

### 4.3 Methodology limitations

#### 4.3.1 1-seed refit winner select 的脆弱性

A3 case 暴露了 spec §6.2 winner select 阈值 (+3pp) 在 std≈0.05 量级 cell 上的脆弱性：1-seed refit_b mean − 1-seed anchor mean = +3pp，处于「seed-to-seed sampling 噪声量级」内。**实测的 5-seed 反转**（refit_b 在 5-seed 下 mean 反而比 anchor 低 1.8pp）确认了这种 1-seed 决策的高错误率。

建议：未来 spec 改成 **2-seed minimum + 阈值放宽到 +5pp**，或保留 winner=anchor 直到 5-seed 数据齐全后再决策。

#### 4.3.2 Anchor 复用而非重测

广验 anchor 0.902 ± 0.021 复用自 Stage C 5-seed。所有 Δ 计算都假设 anchor 不漂——这是 spec §3.1 明确允许的「不重跑节省 5h」。但若后续有 codebase / dataset 改动可能影响 anchor，需要在那时重跑 1 seed 做 sanity。

#### 4.3.3 Single-Re150 / single-U=1.0 / single-target=1.5

广验只在 anchor 的 (Re150, U=1.0, target=1.5) 上做。Reynolds / wake speed / target speed 的 generality 是 future work——design spec §1.3 已明确不在本广验范围内。

---

## 5. Conclusions

### 5.1 站得住的 headline（直接进 paper §experiments）

1. **C1 (`crosscomp/upstream/u10/target_speed=1.5`) 上 4 类 5 个独立 ablation 全部未恢复 success → task-fundamental floor at deployable sensors s0/s1**：reward swap (n=2)、asym critic (n=2)、epoch ×4 (single seed)、sensor s0→s1 (n=2, eff_v2 clean comparison)、convergence check at 64 ep 都没有把 success 抬出 baseline 5-seed (0.218 ± 0.008) 的噪声带；**5 个 configs cross-consistency 钉在 0.195–0.225 (3pp range，<2× single-run 噪声半径)**。reward 与 sensor 各自独立 modulate failure mode (timeout-only ↔ mixed) 但都不动 goal-reaching ceiling → task-data 物理上限，不是 actor 探索风格的 function。机制（reward 失配 / BC penalty 限制 / 其他）需要 BC penalty 强度 sweep 直接区分。online SAC 在同 task 下也失败 → 至少部分原因在 task / reward 这一侧，不是 offline 特有。详见 [c1-s1 sensor upgrade follow-up](./rebrac_c1_s1_followup_report.md)。
2. **B1 sensor envelope robust**（Δ=−0.2pp、std=0.028 ≈ anchor std）。s1 (DVL + ADCP-short) 与单 DVL 在 anchor task 上 success 等效。
3. **A1 collector quality 敏感 + critic_β 可补偿（directional consistent，paired t under-powered）**：goalseek 弱数据下 refit_b (4.0, 1.0) 比 anchor (4.0, 2.0) 平均高 4.0pp、std 低 0.018，5/5 seed 4 个非负差。paired t=2.05 (df=4, p≈0.11) 未达 5% 显著线但方向稳；做 paired bootstrap n_boot=10000 是后续 sanity 选项。

### 5.2 Broad validation 找到的限定条件（既不是修订 paper claim，也不是反例）

1. **A2 mid-gap collapse**：主线 crosscomp-1000 上 ReBRAC vs TD3+BC = +23pp（Stage C），广验 mix5050-1000 上同样比较 = +0.2pp，两个 BC-regularizer 同时坍塌到 0.60 ± 0.15。**新发现**：ReBRAC 的 quantitative advantage 是 data-coherence-dependent 的；paper §experiments 应明确 scope 到 single-mode coherent behavior data，并把 mix5050 收口列为 generality boundary。机制（mode collapse hypothesis）未做直接 ablation。
2. **B 轴 sensor envelope claim 收窄**：s1 robust，s2 mean-on-anchor 但 variance-sensitive (std 2.78× anchor)。Paper 引用应限定为「s0/s1 等效；s2 需要扩 seed 预算或 sensor-aware regularization」。
3. **A3 refit decision 不显著**：5-seed paired t=−1.23 (df=4, p≈0.29)，anchor (4,2) 与 refit_b (4,1) 统计上不可区分（mean diff −1.8pp 在 sd_diff=0.033 内）。1-seed refit 决策在此处是 fragile，应作为 methodology footnote 而不是 finding。

### 5.3 Future work

1. **BC penalty 强度 sweep on C1**（β1 ∈ {0, 1, 2, 4, 8}）：是 task-fundamental floor 假设的 mechanism discriminator。若降 β1 不能恢复 success → β floor 升格为 task-fundamental（actor 即使从 BC anchor 释放也碰不到更高 success）；若降 β1 能恢复 → 当前 ceiling 是 BC penalty 的副产品。
2. **C1-s2 (4 probes 16-D) 与 target_speed=2.0 backlog**：s0→s1 已闭环（−2pp，sensor 不是 lever）；s2 升格为 16-D 进一步上探 sensor 维度上限；target_speed=2.0（顺流方向更快）测试是否解锁 deployability。两者各 ~1–2h L4，记入 spec §13 backlog（[c1-s1 sensor upgrade follow-up](./rebrac_c1_s1_followup_report.md) §7.2/§7.3）。
3. **Reward redesign for upstream**（[online_sac_reward_redesign.md](./online_sac_reward_redesign.md) 已起草）：efficiency_v2 在 upstream 上不可解，需要 task-aware reward shaping。这是 task-side 工作，不在 ReBRAC paper scope 内但是 thesis-level 必要 follow-up。
4. **Spec rev.2 — winner select 改为 2-seed minimum**（A3 case 的 lessons learned）。

---

## 6. Cross-link

### 6.1 与主线文档的关系

| 文档 | 引用关系 |
|---|---|
| [rebrac_experiment_plan.md](./rebrac_experiment_plan.md) | Stage C finalist anchor 来源 |
| [rebrac_experiment_report.md](./rebrac_experiment_report.md) | 主线 5-seed 成绩与 broad validation Δ 计算基线 |
| [rebrac_mainline_review.md](./rebrac_mainline_review.md) §2.2/§3.2 | 本广验回应的 generality 弱点列表；本报告 §3 直接对应这两节 |
| [rebrac_method_section_draft.md](./rebrac_method_section_draft.md) | paper revision 时应在 §experiments 引用 §3.2 mix5050 boundary 与 §3.5 C1 task-fundamental floor case |
| [online_sac_reward_redesign.md](./online_sac_reward_redesign.md) | C1 deep-dive 的 task-side 平行证据 |
| [rebrac_c1_s1_followup_report.md](./rebrac_c1_s1_followup_report.md) | §3.5.5 sensor upgrade 证据来源；standalone follow-up 含 verdict gate / per-seed 细节 / paper-narrative 升级建议 |
| [broad_validation design spec](./superpowers/specs/2026-05-04-rebrac-broad-validation-design.md) | 本广验的实验设计 spec（rev.1 + §13 C1-s1 follow-up；建议 rev.2 收纳 §4.3.1 winner select limitation） |

### 6.2 应回写到 mainline_review 的内容

`docs/rebrac_mainline_review.md` 的 §3.5 应新增一段，cross-link 本报告：

> "广验 (broad validation) 在三轴 (data quality / sensor / task geometry) 上做了 8 spoke × 5 seed 的覆盖 + C1 deep-dive：(1) **C1 upstream geometry 上 4 类 5 个独立 ablation (reward swap / asym critic / epoch ×4 / sensor s0→s1 / convergence check) 全部未恢复 success → task-fundamental floor at deployable sensors s0/s1 and target_speed=1.5**（baseline 0.218 ± 0.008，5-config cross-consistency 钉在 0.195–0.225 / 3pp range）；reward 与 sensor 各自独立 modulate failure mode 但都不动 ceiling；BC penalty 强度 sweep 是关键 mechanism discriminator (未做)；online SAC 同 task 下也失败；(2) B1 sensor s1 与 s0 success 等效（Δ=−0.2pp），B2 s2 mean-on-anchor 但 std_blow_up；(3) A1 goalseek 弱数据下 refit_b critic_β=1.0 比 anchor 高 4.0pp（directional consistent，paired t=2.05 df=4 p≈0.11，underpowered at n=5）；(4) **A2 mix5050 上 ReBRAC vs TD3+BC = +0.2pp**，与主线 crosscomp-1000 上的 +23pp 形成对比，揭示 ReBRAC quantitative advantage 的 data-coherence boundary。详见 [docs/rebrac_broad_validation_report.md](./rebrac_broad_validation_report.md) 与 [docs/rebrac_c1_s1_followup_report.md](./rebrac_c1_s1_followup_report.md)。"

### 6.3 已完成的 commit 序列（截至 2026-05-07）

| 顺序 | 内容 | 文件 | 状态 |
|---|---|---|---|
| 1 | 本报告 rev.2（C1-s1 集成升级） | `docs/rebrac_broad_validation_report.md` | ✓ 已提交 |
| 2 | C1-s1 follow-up 报告 §6.2/§6.3/§7.1 修正 | `docs/rebrac_c1_s1_followup_report.md` | ✓ 已提交 |
| 3 | mainline_review §3.5 cross-link（顶层指针） | `docs/rebrac_mainline_review.md` | ✓ 已提交 |
| 4 | experiment_report §10A 退回为 pointer（删 cf5cfff sensor-floor framing） | `docs/rebrac_experiment_report.md` | ✓ 2026-05-07 退回 |

**experiment_report 主线 retrofit (新增 §10B/§10C 等扩展 spoke 内容) deferred**——见文档头部 status note 列出的 trigger condition (BC penalty sweep on C1 / A1 paired bootstrap / mix ratio sweep / target_speed=2.0 P1)。

---

**END**
