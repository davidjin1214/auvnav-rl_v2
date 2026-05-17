# ReBRAC 主线总结、专家分析与下一步建议

> 文档版本：2026-05-07 rev.3
> 文档定位：ReBRAC 离线主线 rev.8 收口后的独立 review，作为 paper drafting 与战略决策的输入。
> rev.2 修订（2026-05-01）：rev.1 §2.2 + §3.1 列出的 4 项 paper-readiness 必做 (A/B/C/D) **全部完成**。本文 §2.2.1~§2.2.4 的 "建议处理" 段已改写为 "已完成 → docs/..."；§3.1 表格已 strikethrough；§4 caveats item 3 已升级；§5.1 主对照表 Phase 2 行升 5-seed。详见 [notebooks/rebrac_paper_followup_completed.ipynb](../notebooks/rebrac_paper_followup_completed.ipynb)、[docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md)、[docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md)。
> rev.3 修订（2026-05-07）：在 paper-readiness 4/4 之后做了一轮**有限算力广验**（三轴 8 spoke × 5 seed parity + C1 deep-dive 5 ablations，~30h L4）。本文新增 §3.5 broad validation 结论回写、§4 caveat (8) broad validation underpowered seed budgets。Paper claim 不需要修订，但 §experiments 应增加 broad validation 一节（A2 mid-gap collapse boundary + C1 task-fundamental floor）。详见 [docs/rebrac_broad_validation_report.md](./rebrac_broad_validation_report.md)、[docs/rebrac_c1_s1_followup_report.md](./rebrac_c1_s1_followup_report.md)。
> **⚠ 2026-05-18 update**：v1 广验全套（含 §3.5 引用的 broad_validation_report + c1_s1_followup）已 **SUPERSEDED by v2 plan** [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md)。取代原因：(1) v1 在 `efficiency_v2` 下做，与 online 线已转向 `arrival_v2` 不一致；(2) online §7.6 已独立产出更强 sensor envelope finding。v2 plan 在 `arrival_v2` 下做 cross-only spotlight (5 cell + 1 conditional sweep)，作为 paper §experiments 的 reward-bridge appendix。**本文 §3.5 / §4 caveat (8) / §5 反推内容保留作 v1 论文化摘要 archive，不重写；paper drafting 时 §experiments 应直接引用 v2 plan，不引用 v1 finding（除 B1 sensor envelope，但已被 online §7.6 取代）**。
> 配套文档：[rebrac_experiment_plan.md](./rebrac_experiment_plan.md)（计划）、[rebrac_experiment_report.md](./rebrac_experiment_report.md)（实验数据）、[td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)（baseline）、[td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)（worldcomp baseline）。
> 阅读建议：本文不重复 plan / report 的全部细节，重点是**结论的可信度评估**、**论文化时会被审稿人挑战的点**、**下一步该做什么不该做什么**。

---

## 0. 执行摘要（One Page）

### 0.1 现状

ReBRAC 主线（Stage A→B0→B→C→D Phase 1+2+probe→E (a)→**F (rev.2 paper-readiness probes A/B/C/D)**→**G (rev.3 broad validation 三轴 8 spoke + C1 deep-dive 5 ablations)**）已全部跑完，约 **70 个新训练 run** + 4 项 paper-readiness probes + 广验 ~30h L4 (8 spoke × 5 seed parity + C1 deep-dive)。**rev.2 修订**：原 §2.2.1~§2.2.4 列出的 4 项审稿人弱点全部堵死。**rev.3 修订**：广验确认 paper claim 不需要修订，但发现两条 generality 边界 (A2 mid-gap collapse / C1 task-fundamental floor at deployable sensors s0/s1)，应作为 paper §experiments 的 boundary case 引用，详见 §3.5。四条论文级 finding 已经成立：

1. **crosscomp 上 ReBRAC 显著优于 TD3BC**：crosscomp-1000 `+23.0pp`，crosscomp-2000 `+32.2pp`，且 std 同时不增反降；并翻转了 TD3BC 主线 "2000 < 1000" 的退化（[report §7](./rebrac_experiment_report.md)）。
2. **worldcomp 上 ReBRAC + deployable obs 与 TD3BC privileged-critic 协议持平**：0.928 vs 0.922（5-seed），Welch's p=0.9195 量化确认持平（[stats followup](./rebrac_statistical_test_followup.md)）；privileged critic 在 ReBRAC 上不再贡献 mean 抬升（5-seed Δ priv−dep=+0.6pp），只在 outlier seed 44 上独立救回 +12pp（[report §7.10 + §7.12](./rebrac_experiment_report.md)）。
3. **dual penalty `(β1=4.0, β2=2.0)` 是 ReBRAC 主线必需配置**：β2=0 在 typical regime 只掉 2~5pp mean，但 `mean_target_q` 跨 dataset 漂 +46% / +98%、outlier seed 44 在 crosscomp 上掉 -17pp（[report §7.13 + §7.14](./rebrac_experiment_report.md)）。
4. **critic LayerNorm 与 dual penalty 是两个独立必要 component**（rev.2 新增）：LN-off probe (crosscomp-1000, 2 seeds) 让 mean 退化 -16.2pp（远超 β2=0 的 -2.4pp），且退化方向不同（LN-off Q 朝更负 + std blow-up 12×；β2=0 Q 朝更正 + mean 微跌）→ 堵审稿人 "提升其实是 LayerNorm" 论点（[report §7.15](./rebrac_experiment_report.md)）。

### 0.2 战略判断

**rev.2 修订**：**实验本身已"够发"**且**审稿人弱点全部堵死**。原 rev.1 列出的 3 个弱点（Q-normalized 变体不是原版 ReBRAC、Phase 2 仅 3 seeds、缺 LayerNorm 单独 ablation）已在 rev.2 通过 4 项 paper-readiness probes (A/B/C/D, ~4h L4) 全部 closed。**paper drafting 可正式启动**，无需再做更多实验补强。之后可把研究力量切到 online thesis 线。

**rev.3 修订**：广验在 paper claim 之外发现两条 generality 边界 (A2 mid-gap collapse / C1 task-fundamental floor)。这些是 **限定条件而非反例**——paper §experiments 应增加 broad validation 一节作为 deployability boundary case，但不需要回退主 claim。**paper drafting 仍维持启动状态**，broad validation 段落可与正文并行起草（[docs/rebrac_broad_validation_report.md](./rebrac_broad_validation_report.md) 已是 publication-ready draft）。

继续追加 Stage E (b) 或扩 ablation 网格（capacity / dropout / hidden_dim）的边际收益仍然低于继续投入的机会成本——rev.2 维持不做。**rev.3 新增 backlog**：BC penalty 强度 sweep on C1（task-fundamental floor 假设的 mechanism discriminator）— 优先级取决于 paper review 反馈。

### 0.3 三句话答辩稿（写论文时的 narrative spine）

> 我们用一个 Q-normalized 的 dual-penalty TD3+BC 变体（实现上对应 ReBRAC 的 minimal recipe），在水下 AUV wake navigation 这个 sim2real 任务上把 deployable-only 的离线策略性能拉到了与 privileged-critic 协议持平的水平，且优于 vanilla TD3+BC 23~32pp。机制上，actor-side BC penalty 主导了 mean 性能的提升，critic-side penalty 不贡献 mean 但通过 target-Q 抑制 Q 高估、并对 outlier seed 提供 dataset-invariant 的稳定性。这一结果意味着，对水下机器人这种 deployable sensor 严重受限的场景，离线 RL 可以不依赖任何 privileged simulator 信息就把性能逼近 online teacher。

---

## 1. 实验总结

### 1.1 研究问题（plan §2）

整条 ReBRAC 主线只回答三个核心问题：

1. **主问题**：在与 phase0c 相同的 deployable canonical protocol 下，ReBRAC 是否稳定优于 TD3BC？
2. **机制问题**：ReBRAC 是否能改善 `crosscomp-2000` 相对 `crosscomp-1000` 的退化？
3. **辅助问题**：`worldcomp` 上的 teacher gap，ReBRAC 关上多少、靠什么关上？

到 rev.7 为止，三个问题分别由 Stage C / Stage C / Stage D + Stage E (a) 完整回答。**rev.8（review.md rev.2）补充**：4 项 paper-readiness probes 把 §2.2.1~§2.2.4 的审稿人弱点全部堵死，新增第 4 个 finding（critic LN 与 dual penalty 是两个独立必要 component）。

### 1.2 Canonical 协议（plan §4）

所有 stage 共用：

- benchmark `single_u10_cross_tgt15`、task `cross_stream`、target speed `1.5`、objective `efficiency_v2`、probe layout `s0`、history length `4`、flow `wake_v8 Re150 Ti5pct`。
- val=40、test=100（Stage B 和 epoch-probe 是 test=40），ckpt 选择规则 `success_rate → return → -safety_cost → -time`。
- TRAIN_EPOCHS=64（Stage B0 / Phase 1 epoch-probe 钉死），seeds `42/43/44/45/46`。
- ReBRAC 实现是 **Q-normalized 变体**（actor loss 中 deterministic policy gradient 除以 `|Q|.detach()`），与 `auv_nav/rebrac.py` 一致；与原始论文 ReBRAC 在 actor loss 形式上不同，需要在论文里显式声明（详见 §3.1）。

### 1.3 各 Stage 关键结果

| Stage | 范围 | 关键产出 | 是否通过判据 |
|---|---|---|---|
| A | 实现 + smoke | mytorch1 端到端跑通 | — |
| B0 | 训练预算 probe（6 runs，2 datasets × 3 seeds × 128 epoch） | `TRAIN_EPOCHS=64` 充足；seed 44 作为系统性难 seed 首次浮现 | — |
| B | 3×2 screening（36 runs） | winner = `(β1=4.0, β2=2.0)` 两 dataset 共用；ep1000 `+21.1pp` / ep2000 `+32.1pp` vs TD3BC | 阳性 |
| C | 5-seed formal（15 runs，9 ckpt 复用 B） | ep1000 `0.902 ± 0.021` / ep2000 `0.918 ± 0.030`；3 项失败条件全过；翻转 "2000 < 1000" | ✅ 4/4 |
| D Phase 1 Step 1 | worldcomp epoch-probe（2 runs） | TRAIN_EPOCHS 锁 64；2-seed × test=40 拿到 `1.0 ± 0.0`；早期暗示落入情景 A | — |
| D Phase 1 Step 2 | worldcomp deployable formal（5 runs） | `0.928 ± 0.077`，**首次只用 deployable obs 追平 TD3BC privileged**，gap closure 53.0% | ✅ 情景 A |
| D Phase 2 | worldcomp privileged-critic（3 runs，含 seed 44） | `0.9267 ± 0.025`，与 deployable 几乎相等；seed 44 独立救回 +12pp | ✅ 4/4 |
| D probe | critic-penalty-off worldcomp（2 runs） | `0.910` 同 seeds -5.0pp；`mean_target_q` 跳 +46% | 情形 B |
| E (a) | critic-penalty-off crosscomp 5-seed（5 runs） | `0.878 ± 0.090` 同 dataset -2.4pp；`mean_target_q` 漂 +98%；seed 44 掉 -17pp | 情形 B，Finding 1 跨 dataset 成立 |
| E (b) | seed 44 collector inspection | 维持推迟（触发恢复条件均未触发） | — |

### 1.4 总训练成本估算

| Stage | 新 run | 复用 ckpt | 备注 |
|---|---:|---:|---|
| B0 | 6 | 0 | TRAIN_EPOCHS=128 长训 |
| B | 36 | 0 | 3×2×2×3 |
| C | 6 | 9 | seed 42/43/44 物理复制 B 的 ckpt |
| D Phase 1 Step 1 | 2 | 0 | TRAIN_EPOCHS=128 长训 |
| D Phase 1 Step 2 | 5 | 0 | TRAIN_EPOCHS=64 |
| D Phase 2 | 3 | 0 | privileged critic |
| D probe | 2 | 0 | β2=0 |
| E (a) | 5 | 0 | β2=0，crosscomp |
| **小计** | **65 新 run** | 9 复用 | — |

实际算力消耗大致等价于 **~65 个 ReBRAC 64-epoch run + 8 个 128-epoch run**。Stage C 的 ckpt 复用决策（plan §6.4 / report §7.2）至少省掉 9 次重训练成本。

### 1.5 主对照表（5-seed × test=100）

| 协议 | dataset | success mean ± std | 备注 |
|---|---|---|---|
| TD3BC phase0c | crosscomp-1000 | 0.672 ± 0.045 | baseline，α=0.25 |
| TD3BC phase0c | crosscomp-2000 | 0.596 ± 0.036 | "2000 < 1000" 退化 |
| ReBRAC Stage C | crosscomp-1000 | **0.902 ± 0.021** | `+23.0pp` |
| ReBRAC Stage C | crosscomp-2000 | **0.918 ± 0.030** | `+32.2pp`，翻转退化 |
| TD3BC deployable formal | worldcomp-1000 | 0.858 ± 0.080 | 退化为 BC |
| TD3BC privileged-critic formal | worldcomp-1000 | 0.922 ± 0.086 | gap closure 48.5% |
| **ReBRAC deployable Phase 1** | worldcomp-1000 | **0.928 ± 0.077** | **gap closure 53.0%**，仅用 deployable obs |
| ReBRAC privileged Phase 2 | worldcomp-1000 | 0.9267 ± 0.025（3 seeds） | 与 deployable mean 相等 |
| ReBRAC β2=0 Stage E (a) | crosscomp-1000 | 0.878 ± 0.090 | 同 dataset -2.4pp |
| teacher baseline | worldcomp | 0.990 | 100% gap closure 上界 |

---

## 2. 专家分析

### 2.1 强项（论文应该领跑这些）

#### 2.1.1 sim2real 部署友好性是最强 narrative

"ReBRAC + deployable obs 与 TD3BC privileged-critic 持平" 是这条线最有解释力的发现。原本 asymmetric critic 的故事是 "训练时给 critic 多塞 privileged 信息可以让 deployable actor 更好用"——但你的结果反过来说：在 ReBRAC 上 critic 根本不需要 privileged，actor BC penalty 一条线就把 worldcomp 的 deployable→teacher gap 关到 53%。

对水下 AUV 这种部署受限的场景，**协议层面的简化（actor / critic 都只用 DVL water-track 单点采样）比数字上+若干 pp 重要得多**。这与项目的 sim2real 主旋律完全一致：CLAUDE.md 里写的 "deployment-realistic sensor (s0 = DVL water-track only) as the main axis" 现在有了正面证据。

#### 2.1.2 `mean_target_q` 跳升 +46% / +98% 把 Finding 1 救了回来

最初 Phase 1 看到 `β2·ratio = 0.011` 时几乎要写 "dual penalty 退化为单 penalty"——这是非常容易被审稿人一击就破的弱论断（"那为什么不直接简化算法"）。critic-penalty-off probe 强制把 β2 拨到 0、看到 Q 估计直接漂飞 +46%（worldcomp）/ +98%（crosscomp），是教科书式的 "用反事实把表面指标和真实机制分开"。Stage E (a) 在 crosscomp 上让同一机制再现一次，**dataset-invariant 的性质就立住了**——两个 dataset、两个 baseline Q 量级（worldcomp +15.22 / crosscomp -8.25）下 β2 移除都让 Q 朝更不保守方向漂 +7~+8 个绝对单位，这个量级一致性是机制论证的关键。

#### 2.1.3 seed 44 的 cross-(dataset, β2) 双向闭环

这是方法论上最干净的一段：

- worldcomp deployable：seed 44 = 0.78（崩盘）；
- worldcomp privileged：seed 44 = 0.90（救回 +12pp，靠 hull-integral 暴露给 critic）；
- crosscomp `(β2=2.0)`：seed 44 = 0.87（稳）；
- crosscomp `(β2=0)`：seed 44 = 0.70（崩 -17pp，靠 critic penalty 缺失）。

同一个 seed 在两个不同 "缺 critic 稳定信号" 的配置下都崩、加上不同形式的 critic 稳定信号都被救回——把 "seed 44 是统计噪声" 这种最朴素反驳直接堵死。这是 [report §7.14.6](./rebrac_experiment_report.md) cross-(dataset, β2) 表的真正威力。

#### 2.1.4 实验工程上的几个小但精彩的决策

- **Stage B0 的 "长训一次读中间 ckpt" 替代 TRAIN_EPOCHS sweep**：4 倍预算压成 1 倍，且保留 16 个 val 数据点（plan §6.2）。这是非常聪明的预算优化。
- **Stage C 物理复制 9 个 Stage B ckpt**：用 `shutil.copytree` 而非 `os.symlink`（Drive FUSE symlink 不稳定），干净解耦 "manifest 扩容差异" 与 "seed 扩展差异"（report §7.2）。
- **Phase 2 选 3 seeds = `42/43/44` 而非任意 3 个**：明确把 seed 44 当成最高信息量的诊断点，而不是为了省算力随便砍。
- **judgement criteria 全部事先写死**（Rule 1/2/3 + 三档情景 A/B/C），事后不调阈值。这种 pre-registration 风格在算法 RL 文献里很少见。

### 2.2 论文化前必须正面处理的弱点

按 "审稿人一定会问" 的优先级排：

#### 2.2.1 Q-normalized 变体不是原版 ReBRAC（最高优先级） `【rev.2 — 已完成】`

[plan §5 注记](./rebrac_experiment_plan.md) 已经明示：当前 `auv_nav/rebrac.py` 的 actor loss 把 deterministic policy gradient 除以 `|Q|.detach()`（继承自 TD3+BC 的 `λ = α / |Q|` 尺度约定），与 Tarasov et al. (2023) 原始 ReBRAC 的未归一化形式不同。这意味着：

- 你的 `β1, β2` 数字与原 ReBRAC paper 的 `β1, β2` **不能直接对比**；
- `+23~+32pp` 的提升如果作为 "ReBRAC vs TD3+BC" 报告，懂 ReBRAC 的审稿人会立刻 flag "这是 ReBRAC 还是 TD3+BC 的某个变体"。

~~建议处理~~ **已完成（rev.2，详见 [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md)）**：
- ✅ method section 显式命名为 **"Q-normalized dual-penalty TD3+BC variant" (alias ReBRAC-Q)**，不再简单写 "ReBRAC"；
- ✅ 与原 ReBRAC 的差别表（actor `Q` term scaling: `−E[Q]` vs `−(1/|Q|.detach())·E[Q]`）+ 与 TD3+BC 的等价关系（`β1 ≈ 1 / α_TD3BC`，β1=4.0 ↔ α≈0.25）已写入 method draft §3 / §4；
- ⏸ "non-Q-normalized vanilla ReBRAC" 1-2 seed sanity 未做（review §3.3 / report §10 limit 11 维持不做：rev.6 已删除 normalize_q-off 5-seed ablation；如审稿人 push，可补，预算低）。

**paper drafting 执行方式**：直接 import [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md) §1-§5 到 paper method section + main table 注脚。

#### 2.2.2 Phase 2 仅 3 seeds，无法支撑 "std 严格低于 TD3BC privileged" 的论断 `【rev.2 — 已完成】`

Phase 2 的 std=0.025 数字非常漂亮，但有两个结构性问题：

1. **3 seeds 与 5 seeds 的 std 不可直接对比**——5 seeds 取到极端 seed 的概率高，3 seeds 估的 std 系统性偏低；
2. **3 seeds = `42/43/44` 是经过事先选择的**，其中 seed 44 恰好在 privileged 协议下被救回。这是结构性 cherry-pick：privileged-critic 救回 worst seed → 自然把 std 压下来。如果 paper 写 "ReBRAC privileged std 0.025 << TD3BC privileged std 0.086"，审稿人会立刻指出这一点。

~~建议处理~~ **已完成（rev.2，详见 [report §7.12](./rebrac_experiment_report.md)）**：
- ✅ 补 seed 45/46 完成（driver auto-skip 42/43/44，仅训 45/46，~2h L4）；
- ✅ Phase 2 5-seed 结果：**mean=0.9340 ± 0.0261**（与 3-seed 0.9267 ± 0.025 几乎不变；mean +0.7pp、std 几乎不变 → finalist 在 seed-robustness 上很稳）；
- ✅ **std 0.0261 严格低于 TD3BC priv 5-seed std 0.086**（0.30×）→ paper 中可正式做 std 对比 claim；
- ✅ 4/4 判据保持通过，gap closure 从 52.0% 升到 57.6%（主要由 seed 44 +12pp 救回贡献）。

**paper drafting 表述更新**：从 "Phase 2 3-seed 边际确认；不 claim std 严格更低" 改为 **"Phase 2 5-seed formal；std 严格低于 TD3BC privileged"**（n=5 vs n=5 严格 parity）。

#### 2.2.3 ReBRAC deployable 0.928 vs TD3BC privileged 0.922 在统计意义上是 "持平" `【rev.2 — 已完成】`

5 seeds × test=100，跨 seed std=0.077，标准误约 0.077/√5 ≈ 0.034；TD3BC privileged 0.922 ± 0.086 标准误约 0.038。Δ +0.6pp 远小于 1 个标准误的差。

**这意味着**："ReBRAC deployable 超过 TD3BC privileged" 这种措辞在严格统计意义下站不住，正确表述是 "**与 TD3BC privileged-critic 协议持平**"。但**论文价值不变**——核心 finding 是 "**deployable-only 协议在统计意义上等价于 privileged-critic 协议**"，这本身就是非常强的部署友好性论断。

~~建议处理~~ **已完成（rev.2，详见 [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md) + [report §7.16](./rebrac_experiment_report.md)）**：
- ✅ paired episode-level bootstrap (10000 resamples)：point estimate Δ = +0.0060；**95% CI on Δ = [-3.0pp, +4.2pp]**（包含 0）；bootstrap p ≈ **0.7762**；
- ✅ Welch's t-test (5 vs 5 seed-level means)：t = 0.104，**p (two-sided) = 0.9195** → fail to reject H0 → **持平**；
- ✅ gap closure +4.5pp 95% CI = [-62.12pp, +86.36pp]（极宽，n=5 underpowered）；
- ✅ paper main results 表注脚直接拷入 [docs/rebrac_statistical_test_followup.md §5](./rebrac_statistical_test_followup.md)。

**paper drafting 表述更新**：main text 改为 **"持平（statistically not different）"**，附 Welch's p=0.9195；gap closure +4.5pp 仅在 discussion 中作 directional 报告，不在 abstract / conclusion 中作为强 claim；5 seeds 是 RL benchmark 常见上限，limitations 显式声明 underpowered。

#### 2.2.4 缺 LayerNorm / 容量单独 ablation `【rev.2 — 已完成（LN 部分）】`

当前 winner 配置是 `(β1=4.0, β2=2.0, critic_layernorm=on, hidden=256, layers=3)`。Stage E (a) 已经把 `(β2=0)` 这一格做了，但 LayerNorm 关掉会怎样、容量减半会怎样都没有数据。**ReBRAC 原 paper 反复强调 critic LayerNorm 比 dual penalty 更关键**——如果审稿人用这一点压下来 "你的提升其实主要来自 LayerNorm，不是 dual penalty"，目前没有反驳。

~~建议处理~~ **LN 部分已完成（rev.2，详见 [report §7.15](./rebrac_experiment_report.md)；容量 ablation 维持不做）**：
- ✅ `(β1=4.0, β2=2.0, critic_layernorm=off) × 2 seeds × crosscomp-1000`：**mean=0.74 ± 0.25, std blow-up 12×**；
- ✅ Δ vs LN-on Stage C: **−16.2pp**（远超阈值"5pp 退化"）；
- ✅ Δ vs β2=0 Stage E (a): **−13.8pp** (LN-off 比 dual-penalty-off 更差)；
- ✅ `mean_target_q` 漂 −3.84 单位（朝更负 +46%）+ seed 42 单点崩溃 −36pp（LN-on 同 seed 是 0.92）→ **LN 与 dual penalty 是两个独立 component**：退化方向不同（LN-off 让 Q 朝更负 + std blow-up；β2=0 让 Q 朝更正 + mean 微跌）；
- ⏸ 容量 / dropout / hidden_dim sweep 维持不做（rev.6 / rev.7 / rev.8 一致）。

**paper drafting 弱点反驳**：审稿人若 push "提升其实是 LayerNorm"，可直接引用 §7.15.5 三向对比表："关 LN 让 Stage C finalist 退化 -16.2pp（远大于关 dual penalty 的 -2.4pp），LN 与 dual penalty 是两个独立 component"。但 paper 中**不 claim "LN 比 dual penalty 重要"**——n=2 seeds 强度只支撑 "LN 是必要 component" 存在性 claim。

#### 2.2.5 probe 阶段 `mean_test_return = 35.62 > teacher 32.19` 的现象未深挖

[plan §6.5.2](./rebrac_experiment_plan.md) 提到了两种解释（test=40 manifest 噪声 / ReBRAC policy 走出更高效路径），但没继续追。formal 5-seed 后 return 落到 20.17，但 seed 42/45/46 个体仍是 26~30 量级。

如果 ReBRAC policy 真的在 efficiency_v2 reward 下走出比 worldcomp teacher 更短的轨迹，**这是另一个论文级别的 finding——offline RL 学到了 super-teacher**。当前只在 in-text 一笔带过太可惜。

**建议处理**：纯分析无需重训：
- 把 5 seeds × test=100 的轨迹在二维上画出来 vs teacher 轨迹；
- 计算每条 episode 的 `path_efficiency / progress_ratio / time` 分布；
- 用配对 t-test 比较 ReBRAC vs teacher 在同一 episode 起点上的 return。
- 这一组分析半天可以做完，可能撑起论文的一个独立 subsection。

### 2.3 容易被忽略但值得放进 discussion 的 insight

#### 2.3.1 "crosscomp-2000 反而比 crosscomp-1000 好" 的翻转

TD3BC 主线下数据越多越差（0.596 < 0.672），ReBRAC 上数据越多越好（0.918 > 0.902）。[report §8.1](./rebrac_experiment_report.md) 把这归到 "预算不再是解释变量"，但深一层是：**dual penalty + 网络容量让算法有能力 consume 更大的支持集，TD3BC 没有**。

这是离线 RL 文献里反复出现但少被讲清的 motif（"算法能不能用上更多数据" vs "数据是否更多"）。值得在 discussion 写一段——这本身就足以反驳 "离线 RL 数据越多越好" 的朴素直觉。

#### 2.3.2 β1 控制优化方差的机制

[report §6.5 + §8.2](./rebrac_experiment_report.md) 钉死了 `β1↑ → mean_target_q↓`（更悲观）→ seed 44 robustness↑。机制上很清晰：

- β1 越大，actor 越被拉向数据动作支持集；
- 评估 Q(s, π(s)) 落在 in-distribution 区域；
- in-distribution 的 Q 是真实的（更低、不外推），所以 `mean_target_q` 更悲观；
- 但 policy 不再 OOD-extrapolate，seed-to-seed 的 RNG 路径差异被压制。

"BC anchor 是 actor 优化方差的关键" 这件事在 TD3+BC 文献里其实常见，但**与 winner `(β1=4.0, β2=2.0)` 同时出现 `mean_target_q` 最负这个 specific 数据点配合，是一个比常规叙述更具说服力的实证**。

注：plan §6.4 注释里 "β1=4.0 ≈ TD3BC α=0.125 ≈ 更弱 BC 约束" 的方向性表述，依赖于 codebase 内 TD3+BC 的 α 究竟是 BC 系数还是 Q 系数（Fujimoto 原 paper 是 Q 系数，会反转方向）。**建议在 paper 的 implementation note 直接写 "actor loss 的 BC 项绝对系数 β1 = 4.0 是 grid 中最强的 anchoring"**，不依赖 α 折算。empirical fact（β1=4.0 stabilizes seed 44 + winner Q 最悲观）不受这个口径影响。

#### 2.3.3 worldcomp 与 crosscomp 上 `mean_target_q` 符号不同

worldcomp `mean_target_q = +15` vs crosscomp `mean_target_q ≈ -8`——前者 critic 整体乐观，后者整体悲观。[report §7.14.9](./rebrac_experiment_report.md) 注脚提到这是 "reward / Q 量级的 dataset-specific 现象"，但没展开。

**审稿人可能会问**："你怎么解释 worldcomp 上 critic 已经这么乐观了，actor 还能 deploy 出 0.928 success？" 答案应该是：worldcomp teacher policy 高效（mean episode return ≈ +32），所以 in-distribution Q 本来就高；ReBRAC 的 critic penalty 在这里抑制 over-optimism 的*相对*强度（+46% 漂移）与 crosscomp 上的（+98% 漂移）相当。**讨论时强调 "相对漂移幅度" 而非 "绝对 Q 符号"**，否则容易被理解为 "critic 在两个 dataset 上不稳定"。

---

## 3. 下一步建议

按 ROI 分三档。

### 3.1 必做（< 1 day L4）— paper drafting 前堵审稿人弱点 `【rev.2 — 4/4 全部完成】`

| 任务 | 预算 | 目的 | 状态 (rev.2) | 文档落点 |
|---|---|---|---|---|
| ~~A. 补 Phase 2 seed 45/46（2 runs）~~ | ~2h L4 | 把 Phase 2 升到 5-seed，可以正式做 std 对比 | ✅ **完成** mean=0.9340 ± 0.0261 | [report §7.12](./rebrac_experiment_report.md) |
| ~~B. critic LayerNorm-off probe（2 seeds × crosscomp-1000）~~ | ~2h L4 | 证明 dual penalty 不是 LayerNorm 的伪装 | ✅ **完成** -16.2pp 退化（远 > β2=0 -2.4pp） | [report §7.15](./rebrac_experiment_report.md) |
| ~~C. paired bootstrap / Welch's t-test on Phase 1 vs TD3BC priv~~ | ~30 min | 把 "持平" 的 statistical claim 量化 | ✅ **完成** Welch's p=0.9195、CI=[-3pp, +4pp] | [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md) + [report §7.16](./rebrac_experiment_report.md) |
| ~~D. 在 paper method section 显式声明 Q-normalized 变体~~ | ~1h 写作 | 堵 "这不是 ReBRAC" 的论点 | ✅ **完成** 命名为 ReBRAC-Q variant | [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md) |

~~A + B 加起来 ~4 小时 L4，能把 §2.2.1~§2.2.4 三个最显眼的弱点全部堵死。这是性价比最高的投入。~~

**rev.2 完成纪要**：4 项必做合并为单一 notebook [notebooks/rebrac_paper_followup_completed.ipynb](../notebooks/rebrac_paper_followup_completed.ipynb)，Colab 上 ~4h 跑完 + 本地分析。**§2.2.1~§2.2.4 弱点全部堵死，paper drafting 可正式启动。**

### 3.2 建议做（1 sprint）— 把潜在 finding 升级成正式 finding

| 任务 | 预算 | 目的 |
|---|---|---|
| **E. Probe → super-teacher return 现象的轨迹层分析** | ~1 day 分析 | 可能撑起论文一个独立 subsection |
| **F. crosscomp-2000 winner 的 capacity 解释**（小型 capacity sweep，例如 hidden=128 / layers=2 各 2 seeds） | ~4h L4 | 给 §2.3.1 "数据越多越好" 翻转提供机制证据 |
| **G. Phase 2 二阶 finding 的 cross-task 验证**：在 cross_stream 之外的 1 个 task geometry（如 upstream / downstream）上跑 ReBRAC deployable Phase 1 的 1-2 seeds | ~4h L4 | 增强 "privileged ≈ deployable" 的 generality；当前结论仅在 cross_stream + worldcomp-1000 上成立 |

E 是 0 算力 / 1 day 分析时间，价值高、风险低，**应当做**。F 与 G 是机会主义的，**取决于 paper 留给 ReBRAC 的篇幅**。

### 3.3 不做（rev.6 / rev.7 已正确删除，维持）

- ~~`normalize_q-off` 5-seed 完整 ablation~~：被 Phase 2 二阶 finding 削弱优先级。
- ~~3 项消融全因子（full / critic-off / normalize-off）~~：rev.6 已删，维持。
- ~~LayerNorm / dropout / 网络容量全 sweep~~：上面 §3.1.B 是 cheap probe，不是 sweep；如果 probe 证明 LN 关键，是否扩 sweep 再决策。
- ~~Stage E (b) seed 44 collector 起点 inspection~~：触发恢复条件均未触发；除非论文 review 反馈触发条件 3，不再纳入主线。

### 3.4 战略层：研究力量的下一步分配 `【rev.2 — 步骤 1 已完成】`

ReBRAC 主线已经 "够发"。继续在它上面投入的边际收益曲线已经显著下沉。**rev.1 建议 → rev.2 实际状态**：

1. ~~**本周内**：按 §3.1 完成 4 项必做~~ → ✅ **完成（rev.2，~4h L4 + 0.5h 本地分析；详见 §3.1 表）**；
2. **当前**：开始 paper drafting（method + main results + discussion），用 §0.3 的 narrative spine + 直接 import [docs/rebrac_method_section_draft.md](./rebrac_method_section_draft.md) + [docs/rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md) 的现成段落与表注脚；
3. **并行启动**：online thesis 线（[online_rl_thesis_plan.md](./online_rl_thesis_plan.md)）的 Sprint 0 / Sprint 1 实验。两条线在 paper 时间表上是互补的——offline RL paper 写作可以与 online thesis 实验并行。

### 3.5 Broad validation 结论回写 `【rev.3 新增 2026-05-07】`

继 §3.1 paper-readiness 4/4 之后，做了一轮 **有限算力广验**（Stage C finalist anchor 之外，三轴 OFAT，~30h L4）：8 spoke × 5 seed parity + C1 deep-dive (5 个独立 ablation)。完整报告见 [docs/rebrac_broad_validation_report.md](./rebrac_broad_validation_report.md)，C1-s1 sensor upgrade follow-up 见 [docs/rebrac_c1_s1_followup_report.md](./rebrac_c1_s1_followup_report.md)。

**对应 §2.2 / §2.3 留下的 generality 弱点的回应**：

1. **C1 task-fundamental floor at deployable sensors s0/s1 + target_speed=1.5**（C 轴 upstream geometry）：4 类 5 个独立 ablation（reward swap n=2 / asym critic n=2 / epoch ×4 single seed / sensor s0→s1 n=2 eff_v2 clean comparison / convergence check at 64 ep）全部未恢复 success；5-config cross-consistency 钉在 0.195–0.225 (3pp range，<2× single-run 噪声半径)。reward 与 sensor 各自独立 modulate failure mode (timeout-only 77.5/0 ↔ mixed ~53/~26) 但都不动 goal-reaching ceiling → ceiling 来自 task / data 物理上限，不是 actor 探索风格的 function。**机制候选（reward 失配 / BC penalty 限制 / 其他）需要 BC penalty 强度 sweep 直接区分（未做，记入 future work）**；online SAC 在同 task 下也失败 → 至少部分原因在 task / reward 这一侧，不是 offline 特有。
2. **B1 sensor envelope robust，B2 std_blow_up**：B1 (s0 → s1) Δ=−0.2pp、std=0.028 ≈ anchor std (1.35×) → s0/s1 success 等效。B2 (s0 → s2) mean 持平 anchor 但 std=0.058 (2.78× anchor) 触发 std_blow_up flag → paper 引用应限定为「s0/s1 等效；s2 需要扩 seed 预算或 sensor-aware regularization」。
3. **A1 collector quality 敏感，refit_b directional consistent (underpowered)**：goalseek 弱数据下 refit_b critic_β=1.0 比 anchor (4.0, 2.0) 平均高 4.0pp、std 低 0.018，5/5 seeds 4 个非负差。**paired t=2.05 (df=4, p≈0.11)** 未达 5% 显著线但方向稳；做 paired bootstrap n_boot=10000 是后续 sanity 选项。
4. **A2 mid-gap collapse — broad validation 新发现的 data-coherence boundary**：主线 crosscomp-1000 上 ReBRAC vs TD3+BC = +23pp（Stage C），广验 mix5050-1000 上同样比较 = **+0.2pp（matched 5v5）**。两个算法 std 都展开到 0.14–0.16 (anchor std 0.021 的 6.7–7.5×)。**这不是修订 paper 已有 claim，也不是反例**——是 broad validation 揭示 ReBRAC quantitative advantage 的 data-coherence-dependent 边界。paper §experiments 应明确 scope 到 single-mode coherent behavior data，并把 mix5050 收口列为 generality boundary。机制（双峰 action / mode collapse）是 working hypothesis，未做直接 ablation。

**对 paper 的含义**：
- §experiments 增加 broad validation 一节，引用 §3.2 mix5050 boundary 与 §3.5 C1 task-fundamental floor 作为 deployability boundary case；
- §discussion 把 C1 与 Stage D Phase 2 的 `cross_stream + worldcomp` deploy-graded 区间对比，构成完整的 deployability 谱系（哪些 task algorithmic 可解 / 哪些是物理边界）；
- §limitations 增加 (8) "broad validation underpowered seed budgets：A2 boundary 与 A1 paired t 在 n=5 下 underpowered；C1 deep-dive 4 algorithm-side ablations 是 n=2 / single-seed，BC penalty 强度 sweep 未做"。

**Future work backlog（按优先级）**：
1. **BC penalty 强度 sweep on C1**（β1 ∈ {0, 1, 2, 4, 8}）— 关键 mechanism discriminator；
2. **C1-s2 (16-D) 与 target_speed=2.0 backlog** — 上探 sensor / task-parameter 维度上限；
3. **Reward redesign for upstream**（[online_sac_reward_redesign.md](./online_sac_reward_redesign.md) 已起草）— task-side 工作；
4. **Spec rev.2 — winner select 改为 2-seed minimum**（A3 case lessons learned）。

---

## 4. 已知 caveats（必须写进 paper limitations）

整理 [report §9](./rebrac_experiment_report.md) 的局限性列表，挑出会被审稿人盯到的几条：

1. **`state_dict_l2_distance` 交叉验证未做**：Stage B0 "长训 ≈ 单独训到" 假设未从权重层面验证（仅从结果层面一致）。低优先级，但应在 limitations 写出来。
2. **`worldcomp` 上 β2=0 仅 2 seeds**（crosscomp 已 5 seeds 钉死）：可补，但优先级低。
3. ~~**Phase 2 仅 3 seeds**~~（**rev.2 已升 5-seed**：mean=0.9340 ± 0.0261，详见 [report §7.12](./rebrac_experiment_report.md)）。**已不再是 caveat。**
4. **Phase 2 二阶 finding 仅在 worldcomp-1000 上成立**：cross-task 未验证（见 §3.2.G）。
5. **`crosscomp-2000` 上的 critic-penalty-off 未跑**：cross-extrapolate 风险低（Stage C 已显示 ep1000 / ep2000 winner 一致），但严格说没有数据。
6. **mean_target_q 的 dataset-specific 符号差异**：见 §2.3.3。
7. **TD3BC baseline 是别人跑出来的而不是我们重跑的**：TRAIN_EPOCHS 历史上有过调整（TD3BC worldcomp teacher-gap=96，ReBRAC=64）。`crosscomp` 对比无影响，`worldcomp` 对比应在 paper 里单独澄清预算口径。
8. **Broad validation underpowered seed budgets** `【rev.3 新增 2026-05-07】`：A2 mid-gap matched 5v5 与 A1 paired t (n=5) 都 underpowered；C1 deep-dive 4 个 algorithm-side ablations 是 n=2 / single-seed (n=1) 而不是 5-seed parity；BC penalty 强度 sweep（C1 task-fundamental floor 假设的 mechanism discriminator）未做。详见 [rebrac_broad_validation_report.md §4.3](./rebrac_broad_validation_report.md) 与 [rebrac_c1_s1_followup_report.md §9](./rebrac_c1_s1_followup_report.md)。

---

## 5. 附录：核心数字快查表

### 5.1 ReBRAC vs TD3BC（5-seed × test=100）

| dataset | 协议 | TD3BC | ReBRAC | Δ |
|---|---|---|---|---:|
| crosscomp-1000 | deployable | 0.672 ± 0.045 | **0.902 ± 0.021** | +23.0pp |
| crosscomp-2000 | deployable | 0.596 ± 0.036 | **0.918 ± 0.030** | +32.2pp |
| worldcomp-1000 | deployable | 0.858 ± 0.080 | **0.928 ± 0.077** | +7.0pp |
| worldcomp-1000 | privileged-critic | 0.922 ± 0.086 | **0.9340 ± 0.0261（5 seeds，rev.2）** | +1.2pp（统计上持平；Welch's p=0.92 来自 dep vs priv 比较，详见 [stats followup](./rebrac_statistical_test_followup.md)）|

### 5.2 critic penalty 的真实贡献（β2=2 vs β2=0）

| dataset | β2=2 mean | β2=0 mean | Δ mean | β2=2 mean_target_q | β2=0 mean_target_q | Δ Q（绝对单位）|
|---|---|---|---:|---|---|---:|
| worldcomp-1000 | 0.928 (5s) / 0.960 (2s) | 0.910 (2s) | -1.8pp / -5.0pp | +15.22 | +22.30 | +7.08（+46%）|
| crosscomp-1000 | 0.902 (5s) | 0.878 (5s) | -2.4pp | -8.25 | -0.13 | +8.12（+98%）|

### 5.2b critic LayerNorm 的真实贡献（LN=on vs LN=off，rev.2 新增）

| dataset | LN=on mean | LN=off mean | Δ mean | LN=on target_q | LN=off target_q | Δ Q | std blow-up |
|---|---|---|---:|---|---|---:|---|
| crosscomp-1000 | 0.902 (5s) | **0.74 (2s)** | **-16.2pp** | -8.25 | **-12.09** | -3.84（朝更负 +46%） | std 0.021 → 0.255（**12×**） |

详见 [report §7.15](./rebrac_experiment_report.md)。**LN-off 退化方向（Q 朝更负 + std blow-up）与 β2=0 退化方向（Q 朝更正 + std 微涨）不同，证明 LN 与 dual penalty 是两个独立 component。**

### 5.3 seed 44 在四种配置下的成绩

| 协议 | seed 44 success |
|---|---:|
| crosscomp `(β2=2.0)` Stage C | 0.870 |
| crosscomp `(β2=0)` Stage E (a) | 0.700（-17pp）|
| worldcomp deployable `(β2=2.0)` | 0.780 |
| worldcomp privileged `(β2=2.0)` | 0.900（+12pp）|

### 5.4 winner 配置（Stage C 锁定，Stage D / E 全部沿用）

```
β1 = 4.0       # actor BC penalty
β2 = 2.0       # critic penalty (next-action target)
hidden_dim = 256
num_hidden_layers = 3
critic_layernorm = on
actor_layernorm = off
normalize_q = on  # actor loss divides by |Q|.detach()
TRAIN_EPOCHS = 64
CHECKPOINT_EVERY_EPOCHS = 8
val_episodes = 40
test_episodes = 100
seeds = [42, 43, 44, 45, 46]
selection_rule = success_rate → return → -safety_cost → -time
```

### 5.5 文档地图

- [rebrac_experiment_plan.md](./rebrac_experiment_plan.md)：完整计划，**rev.8**（含 Stage F paper-readiness probes A-D）
- [rebrac_experiment_report.md](./rebrac_experiment_report.md)：完整数据，**rev.8**（§7.12 升 5-seed；§7.15 LN-off probe；§7.16 stats test）
- [rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md)：**rev.2 新增**——paper main results 表注脚 + discussion 引用所需的 Welch's t-test / paired bootstrap / gap closure CI
- [rebrac_method_section_draft.md](./rebrac_method_section_draft.md)：**rev.2 新增**——Q-normalized dual-penalty TD3+BC variant (alias ReBRAC-Q) method section 草稿（直接拷入 paper）
- [notebooks/rebrac_paper_followup_completed.ipynb](../notebooks/rebrac_paper_followup_completed.ipynb)：**rev.2 新增**——4 项必做 (A/B/C/D) 合并执行归档
- [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)：crosscomp baseline
- [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)：worldcomp baseline
- [online_rl_thesis_plan.md](./online_rl_thesis_plan.md)：下一阶段（asymmetric critic + privileged hull-integral）
- 本文（rebrac_mainline_review.md）：**rev.2，paper-readiness 4 项必做全部 closed**，paper drafting 可正式启动
