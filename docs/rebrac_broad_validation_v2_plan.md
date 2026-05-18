# ReBRAC Broad Validation v2 — Cross-Only Spotlight under `arrival_v2`

> **文档版本**：2026-05-18 rev.2（精简版；rev.1 5-cell 矩阵已并入 backlog）
> **状态**：active plan（未开跑）
> **作用**：取代 v1 broad validation 全部产出（spec / plan / report），把广验从「efficiency_v2 三轴 8 spoke + C1 deep-dive」收敛成「arrival_v2 cross-only **2 core cell + 2 conditional follow-up**」。
> **取代的 v1 文档**（已加 SUPERSEDED banner）：
> - [`docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`](superpowers/specs/2026-05-04-rebrac-broad-validation-design.md)
> - [`docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`](superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md)
> - [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md)（v1 实验产物 archive，不重跑）
> - [`docs/rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md)（v1 C1 follow-up archive）
>
> **不取代的文档**（保持不动）：
> - ReBRAC 主线三层文档（plan rev.8 / report rev.8 / mainline_review rev.3）— main paper 主线维持 `efficiency_v2`，v2 广验在 paper 里作为 reward-bridge appendix
> - 既有 TD3+BC 归档文档（5 份 Phase 0c 系列）
>
> **配套参考**：
> - [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) — online 线 arrival_v2 实测（§7 strict-control + §7.6 s0 sensor envelope）
> - [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) — arrival_v2 v6 设计 spec（SHELVED 但是技术参考）
> - [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) — offline 线总览（status 同步更新）

---

## 目录

1. [TL;DR](#1-tldr)
2. [动机：为什么 v1 不够 + 为什么收敛到 cross](#2-动机)
3. [v1 → v2 关键变化（diff 表）](#3-v1--v2-关键变化)
4. [实验矩阵：2 core cell + 2 conditional](#4-实验矩阵)
5. [Pre-commit verdict gates](#5-pre-commit-verdict-gates)
6. [Stage 流程：sanity → datasets → core train → conditional](#6-stage-流程)
7. [触发判据 / std 处理](#7-触发判据)
8. [Paper narrative role](#8-paper-narrative-role)
9. [Backlog（明确不在本广验范围内）](#9-backlog)
10. [Cross-link 与文档影响](#10-cross-link)
11. [文件清单（TBD / NEW）](#11-文件清单)
12. [执行检查清单](#12-执行检查清单)

---

## 1. TL;DR

- ReBRAC paper main results 维持 `efficiency_v2` / `cross_u10` anchor 不变（rev.8 已 closed，不翻盘）
- v2 广验在 `arrival_v2` reward 下做 **2 core cell（N0 sub-critical anchor + N2 critical regime）+ 1 sanity baseline（S）+ 2 conditional follow-up（N4 teacher ceiling、M1 BC sweep）**，全部 cross-stream geometry
- **每个 cell 跑 3 seed**（rev.2 调整；rev.1 是 5 seed）— broad val 是 appendix，3 seed acceptable + 在文档中明标
- 砍掉的 v1 内容：upstream / tandem / sbs geometry、goalseek / worldcomp / mix5050 collector、C1 deep-dive 5 ablation
- 砍掉的 rev.1 内容：N1（sub-critical sensor probe）、N3（critical sensor rescue）— 都已被 v1 B1 + online §7.6 覆盖
- 增加：(U, Re, λ) 两 regime 探针（sub-critical u10/Re150/λ=0.67 vs critical u15/Re250/λ=1.0）、与 online §7.6 形成 offline ↔ online 对位、S sanity card 兼任 paper crosscomp baseline control
- paper 角色：**reward-bridge appendix + critical regime head-to-head**，不是替代 main results 而是辅助说明 main finding 对 reward 选择的稳健性
- 预算：**~6–24 h L4 + 2–3 h CPU**（最好 vs 最坏情况）。对比 v1 ~50h L4，砍 ~70%；对比 rev.1 28-40h L4，再砍 ~50%

---

## 2. 动机

### 2.1 v1 的 reward 失配嫌疑

v1 全套实验跑在 `efficiency_v2` 下，但两条独立证据线都表明 `efficiency_v2` 在 upstream / 高难度 cell 上有 reward shaping 失配：

1. [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §3 — `efficiency_v2` 在 `single_u15_upstream` 上 vanilla SAC collapse 到 `success=0`（agent 学会快速出界），arrival_v2 修复到 1.000
2. v1 broad val C1 baseline termination 分布是 `goal=22.5 / timeout=77.5 / oob=0` — 这是 `efficiency_v2` 把 agent 钉在「保守憋边界」上的 footprint；reward swap 到 `arrival_v2_simple` 后翻成 `goal=21.5 / timeout=52.5 / oob=26`

含义：v1 C1 「task-fundamental floor at deployable sensors s0/s1」claim 受 reward bias 污染，**不能在不切换 reward 的前提下被升格为 paper-quality finding**。

### 2.2 online 线已经给出更强 sensor envelope finding

online [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.6 显示：在 `arrival_v2 / cross_u15` 下 vanilla SAC + s0 catastrophic FAIL（`final=0.10 / OOB=0.667`），s1 PASS（`final=0.9`）。**s0–s1 gap = 80pp**，相比 A0 (cross_u10 + arrival_v1) 的 3pp **放大 24×**。

这条 finding 在 v1 broad val（cross_u10, efficiency_v2）里完全看不到 — v1 B1 (s1) untriggered, robust 是 weak positive；online 线的 80pp gap 才是 paper-grade sensor envelope claim。

**v2 直接把 online §7.6 作为 narrative spine**：在 cross_u15/s0/critical regime 上跑 offline ReBRAC，与 online §7.6 catastrophic FAIL 做 head-to-head。这是 v2 最强 paper hook。

### 2.3 收敛到 cross-stream 的理由

- upstream + arrival_v2 + vanilla SAC online 已 saturate 到 1.000（§7.2 single / §7.3 tandem / §7.4 sbs）— offline ReBRAC 上未必 saturate 但 ROI 低于 cross
- tandem / sbs 的几何 prior 在 cross_stream 主线 paper 里没承载 narrative 责任
- cross_stream 是 OOB-once-and-done regime，partial-observability 在这里显化 — sensor envelope 故事最干净

### 2.4 (U, Re, λ) 两 regime 的科学性 framing

v1 单档 (u10 / Re150 / λ=0.67) 是 sub-critical under-actuation；v2 增加的第二档 (u15 / Re250 / λ=1.0) 是 critical under-actuation regime。**两档是两个 physics regime 而非 continuous ladder**（涡街 Strouhal 频率 + wake 几何随 Re 变化）。v2 plan 明确按 "sub-critical vs critical regime comparison" 来 frame，paper 写作时不当 ladder 用。

如果未来需要 ladder 中间档（如 U=1.25 / Re=200），需要先用 `scripts/generate_wake.py` 跑新 LBM 设定，记入 backlog。

### 2.5 为什么不做 sensor probe（N1 / N3 砍掉）

- **N1 sub-critical s1 重测**：v1 B1（cross_u10/s1/efficiency_v2）已是 weak positive；online sub-critical 下 s0/s1 gap 本就小；arrival_v2 下重测的边际信息只是 confirmatory 一行。砍。
- **N3 critical s1 rescue**：online §7.6 已直接报告 cross_u15/s1 vanilla SAC = 0.90（PASS borderline）。offline 上重测主要是闭合 2×2 表，paper 故事并不依赖。砍 + 推 backlog。

---

## 3. v1 → v2 关键变化

| 维度 | v1 (efficiency_v2) | v2 rev.2 (arrival_v2) | 理由 |
|---|---|---|---|
| **Reward** | `efficiency_v2` | `arrival_v2`（8 参数 v6，commit `813096e`） | v1 reward shaping 在 high-difficulty cell 失配 |
| **Task geometry** | C 轴 3 spoke（main cross + C1 upstream + C3 tandem）| 只 cross_stream | upstream saturate；tandem 与主 narrative 重叠 |
| **Flow regime 轴** | 单档 `wake_v8_U1p00_Re150` | 双档：u10/Re150 (sub-critical) + u15/Re250 (critical) | online §7.6 显示 critical 是 partial-obs gap 显化 regime |
| **Collector 集合** | 4 collector + 1 mix | 2 collector：crosscomp + privileged（privileged 仅 conditional 用） | 其余 collector ROI 低或假设不成立 |
| **Sensor 轴** | s0 / s1 / s2 | 只 s0 | s1 重测信息量低（见 §2.5）；s2 在 online 故事中无角色 |
| **Cell 数** | 8 spoke + C1 ablation 5 | **2 core (N0/N2) + 1 sanity baseline (S) + 2 conditional (N4/M1)** | 砍重复 / 砍 confirmatory；只保留 paper-essential |
| **Seed/cell** | 5 seed | **3 seed** [42, 43, 44] | broad val appendix，3 seed acceptable；M1 sweep 已经是 3 seed，对齐 |
| **Mechanism discriminator** | C1 5 ablation 串联 | BC penalty sweep on **stuck cell**（conditional） | 直接、可证伪、与 v1 retrofit trigger 之一对齐 |
| **Anchor** | `efficiency_v2` 5-seed 0.902 ± 0.021 | `arrival_v2` 3-seed N0 重测 | 新 reward 必须新 anchor |
| **N0 GO 阈值** | n/a | **≥ 0.70**（不是 rev.1 的 0.50） | 严格 reward-bridge 判定，避免 "weak bridge" 进 paper |
| **std_blow_up 阈值** | 0.042 = 2 × 旧 std | **改为 warning，不进 verdict gate** | 3 seed 估 std 自由度 df=2 不稳，不作为 hard gate |
| **A2 td3bc head-to-head** | 是 | 砍 | 无 mix dataset |
| **C1 deep-dive 5 ablation** | reward/asym/epoch/sensor/conv | 砍 | 由 conditional BC sweep 替代 |
| **Paper role** | standalone exploratory | reward-bridge appendix + critical regime head-to-head | 明确 paper section |

---

## 4. 实验矩阵

**固定基底**：`cross_stream / target_speed=1.5 / arrival_v2 / probe_layout=s0 / ReBRAC (β1=4, β2=2, hidden=256, critic_LN=on) / 64 epochs / 3 seeds [42, 43, 44] / test_episodes=100`

### 4.1 主体：1 sanity baseline + 2 core cell

| Cell | 类型 | Collector | Sensor | Flow | (U, Re, λ) | 角色 | 训练？ |
|---|---|---|---|---|---|---|---|
| **S** Sanity baseline | sanity | crosscomp | s0 | `wake_v8_U1p50_Re250` | (1.5, 250, 1.0) critical | (a) 数据集可收性 GO/NO-GO；(b) **paper §discussion 必须呈现的 crosscomp baseline control**（让 N2 的 "offline beats online" claim 可证伪） | ❌（纯 rollout 评估） |
| **N0** Anchor | core | crosscomp | s0 | `wake_v8_U1p00_Re150` | (1.0, 150, 0.67) sub-critical | reward-bridge baseline；与 main line `efficiency_v2 / 0.902` 对照 | ✅ 3 seed |
| **N2** Critical regime | core | crosscomp | s0 | `wake_v8_U1p50_Re250` | (1.5, 250, 1.0) critical | 与 online §7.6 `single_cross_s0` (vanilla SAC 0.10) 对位 + crosscomp baseline (S) 做 control；offline 是否更稳 | ✅ 3 seed |

### 4.2 Conditional follow-up（视 N2 结果触发）

| Cell | 触发条件 | Collector | Sensor | Flow | 角色 | 成本 |
|---|---|---|---|---|---|---|
| **N4** Priv teacher | N2 success < 0.7 | **privileged** | s0 | u15/Re250 critical | task-data ceiling 证据：oracle teacher 在 deployable obs 下能拿多少 | 3 seed × 1h = ~3h L4 |
| **M1** BC sweep | N2 success ∈ [0.20, 0.70] | crosscomp | s0 | u15/Re250 critical | mechanism discriminator：BC penalty floor vs task-data floor；β1 ∈ {0, 1, 2, 4, 8} × 3 seed = 15 runs | ~15h L4 |

**双 conditional 协同**：
- N2 < 0.20：N4 必跑（验证 ceiling），M1 skip（β-tuning 在 catastrophic 区间无意义）
- N2 ∈ [0.20, 0.70]：N4 + M1 都跑
- N2 ≥ 0.70：都 skip，paper 引用 N0 + N2 + S 即可

---

## 5. Pre-commit Verdict Gates

**所有 gate 在跑实验之前预登记，不允许 post-hoc 调整**（v1 教训：A3 1-seed → 5-seed 反转）。

### 5.1 S Sanity baseline（GO/NO-GO + control）

| S success (crosscomp 100 ep rollout) | Verdict | 后续动作 |
|---:|---|---|
| ≥ 0.40 | dataset 可收 + 提供 paper control | proceed N2 dataset 收集 |
| 0.20–0.40 | dataset 退化但仍可用 | proceed，paper 标注 "low-quality teacher data" |
| < 0.20 | dataset 几乎全失败示范 | **暂停** N2；fallback 选项见 §6.2 |

### 5.2 N0 Anchor GO/NO-GO（最关键）

| N0 3-seed success | Verdict | 后续动作 |
|---:|---|---|
| ≥ 0.70 | reward bridge holds | N2 按计划跑 |
| 0.50–0.70 | weak bridge | N2 仍跑，paper 写作时明标 "arrival_v2 下 ReBRAC anchor 比 efficiency_v2 下退化 X pp（仍 deployable，但 reward shaping sensitivity 需在 discussion 注明）" |
| < 0.50 | **reward bridge 失败** | **暂停**所有后续 cell；review reward / collector / dataset；可能整套设计推翻 |

### 5.3 N2 Critical regime（与 online §7.6 对位）

| N2 success | vs S (crosscomp baseline) | Verdict | Paper claim 候选 |
|---:|---|---|---|
| ≥ 0.70 | typically > S | offline ReBRAC 在 critical regime 比 vanilla SAC online (§7.6 = 0.10) 显著更稳 + 比 crosscomp baseline 更强 | "offline RL 提供 stability advantage in production-difficulty regime, beyond simple imitation of baseline policy" |
| 0.40–0.70 | typically ≈ S 或略 > S | partial 一致，与 baseline policy 持平 | "offline RL partial mitigation; advantage primarily from imitation, not policy improvement" |
| 0.20–0.40 | ≤ S | offline 比 baseline 弱 | trigger N4 + M1 来诊断 |
| < 0.20 | ≪ S | offline 与 online 同样 catastrophic | trigger N4 验证 "critical regime sensor-fundamental at s0 across both online and offline RL" |

### 5.4 N4 Privileged teacher（conditional, if triggered）

| N4 success | Verdict |
|---:|---|
| ≥ 0.70 | teacher quality 在 critical regime 下能给出可学的 dataset；student (deployable obs) 能利用 |
| 0.50–0.70 | teacher 帮助有限 |
| < 0.50 | 即便 oracle data 在 critical regime 下也救不回来 → 强 "task-data fundamental ceiling" 证据 |

### 5.5 M1 BC penalty sweep（conditional, if triggered）

| β1=0 success vs β1=4 baseline | Verdict |
|---|---|
| Δ > +10pp | BC penalty 是 limiting factor（floor 来自 BC 强度，非 task） |
| Δ ∈ [−5pp, +10pp] | β1 与 ceiling 解耦（floor 来自 task / data，非 BC） |
| Δ < −5pp | β1=0 退化（BC penalty 在 stuck cell 仍提供 stabilization） |

---

## 6. Stage 流程

### 6.1 Sanity & Datasets（pre-train phase）

**Step 1: S sanity rollout**（critical regime crosscomp 评估，约 1h CPU）

```bash
# 不训练，纯 rollout 评估 crosscomp 在 cross_u15/s0 下的行为
python -m scripts.evaluate \
  --policy crosscomp \
  --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective arrival_v2 \
  --episodes 100 --seed 0 \
  --output-dir experiments/offline/rebrac/broad_validation_v2/S_sanity/
```

结果写入 `sanity_card.json` 兼 paper §discussion control 数据。**§5.1 verdict gate 判定**：< 0.20 触发 fallback。

**Step 2: N0 dataset relabel**（约 10-20 min CPU）

复用 v1 cross_u10/crosscomp/s0 dataset（已存在 `offline_data/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/transitions.npz`），只重新计算 `rewards` 列：

```bash
# 新工具：scripts/relabel_rewards.py
# 复用现有 (obs, action, next_obs, done) 不变，按 arrival_v2 公式重打 rewards
python -m scripts.relabel_rewards \
  --input  offline_data/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/transitions.npz \
  --output offline_data/crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/transitions.npz \
  --objective arrival_v2
```

`obs / actions / next_obs / dones / privileged_obs` 列原样拷贝；只 `rewards` 列重算。同步写新 `metadata.json` + `sanity_card.json` reference 到原 dataset。

**Step 3: N2 dataset 新收**（约 30-60 min CPU；仅在 S 通过 GO/NO-GO 后执行）

```bash
python -m scripts.collect_offline_data \
  --policy crosscomp \
  --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective arrival_v2 \
  --episodes 1000 --seed 0 --num-workers 8 \
  --output-dir offline_data/crosscomp_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000
```

需含 `privileged_obs` / `next_privileged_obs` 列，为未来 asym critic ablation 留口。

### 6.2 GO/NO-GO Fallback（若 S < 0.20）

- (a) 降到 u12.5 / Re~200（需要 `generate_wake` 新跑，~1h CPU）
- (b) 接受 N2 dataset 退化，但明确 paper 中说明
- (c) 砍掉 N2/N4/M1，v2 缩到只剩 N0，paper 仅讲 reward bridge 不讲 critical regime

### 6.3 Core training（N0 + N2，6 runs）

```bash
# Per cell, per seed（示例 N0；N2 同结构换 dataset + manifest）
python -m scripts.train_offline_rebrac \
  --offline-data offline_data/crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/transitions.npz \
  --actor-bc 4.0 --critic-bc 2.0 \
  --hidden-dim 256 --critic-layernorm \
  --epochs 64 --batch-size 256 \
  --probe-layout s0 --history-length 4 \
  --objective arrival_v2 \
  --eval-manifest benchmarks/single_u10_cross_tgt15.json \
  --eval-episodes 100 \
  --seed <42|43|44> \
  --device cuda \
  --save-dir checkpoints/offline/rebrac/broad_validation_v2/N0/seed_<n>
```

**执行顺序**：N0 3 seed 先跑 → §5.2 GO/NO-GO 判定 → 通过则 N2 3 seed → §5.3 判定决定是否触发 conditional。

**成本**：2 cell × 3 seed × ~1h L4 = **~6h L4**

### 6.4 Conditional follow-up（N4 / M1，0 / 3 / 15 / 18 h L4）

执行判定：

```python
if N2_mean < 0.20:
    run_N4()                  # ~3h L4
elif 0.20 <= N2_mean < 0.70:
    run_N4()                  # ~3h L4
    run_M1_bc_sweep()         # ~15h L4
else:  # N2_mean >= 0.70
    skip_all_conditional()
```

**N4 dataset 收集**（仅 N4 触发时执行，~30 min CPU）：

```bash
python -m scripts.collect_offline_data \
  --policy privileged \
  --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 ... \
  --output-dir offline_data/privileged_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000
```

**N4 训练**：3 seed × 1h L4 = ~3h L4，配置同 N2 但换 dataset。

**M1 BC sweep**：选定 N2 cell 配置不变，β1 ∈ {0, 1, 2, 4, 8} × 3 seed [42, 43, 44] = 15 runs ≈ 15h L4。

### 6.5 总预算

| 情形 | Sanity | Datasets | Core train | Conditional | Total |
|---|---|---|---|---|---|
| 最好（N2 ≥ 0.70，无 conditional） | 1h CPU | 1h CPU | 6h L4 | 0 | **~6h L4 + 2h CPU** |
| 仅 N4 (N2 < 0.20) | 1h CPU | 1.5h CPU | 6h L4 | 3h L4 | **~9h L4 + 2.5h CPU** |
| N4 + M1 (N2 ∈ [0.20, 0.70]) | 1h CPU | 1.5h CPU | 6h L4 | 18h L4 | **~24h L4 + 2.5h CPU** |

对比：v1 ~50h L4 → **砍 ~70%**；rev.1 5-cell 28-40h L4 → 再砍 ~50%。

---

## 7. 触发判据

继承 v1 spec §6.1 框架，简化：

| 判据 | 阈值 | 触发后处理 |
|---|---|---|
| `mean_shift` | \|cell mean − N0 mean\| > **5pp** | 记录方向 + 是否进入 verdict gate 下一档 |
| `std_blow_up` | cell std > **2 × N0 std** | **仅 warning，不进 verdict gate**（3 seed std 估计 df=2 太不稳；rev.1 hard gate 撤销） |
| `verdict_gate_violation` | cell success 落入 pre-commit 表的下一档 | 按 §5 verdict gate 处理 |

v1 旧 `td3bc_gap_collapse` 判据（A2 专用）**砍掉** — v2 没有 mix dataset。

---

## 8. Paper Narrative Role

v2 在 paper 里的明确角色：**§experiments 的 reward-bridge appendix + §discussion 的 critical regime head-to-head**。

### 8.1 §experiments 引用方式

```
Main results (Section X.Y) report ReBRAC's +23.0 pp gain over TD3+BC on
`crosscomp / cross_u10 / s0 / efficiency_v2`. To verify this finding is not
an artifact of the `efficiency_v2` reward design (whose limitations in
upstream geometries are documented in [arrival_v2_experiment_report]),
we re-anchor under `arrival_v2` (cell N0) and probe critical-regime
generality (cell N2 + sanity baseline S + conditional N4/M1).
[Insert main table + verdict summary.]
```

### 8.2 §discussion: critical regime head-to-head

paper §discussion 关键素材（v2 独有的 paper hook）：

| Cell | Setup | online vanilla SAC (§7) | offline (this work) | Δ |
|---|---|---|---|---|
| cross / s0 / u10 (sub-critical) | reward-bridge anchor | (待补，1 seed ~1h L4) | **N0** | reward-bridge holds? |
| cross / s0 / u15 (critical) | **stress test** | **§7.6 = 0.10** (catastrophic FAIL) | **N2** vs **S** (crosscomp baseline) | **offline stability advantage?** |

**关键 control**：N2 的 "offline beats online" claim 必须配 S（crosscomp baseline rollout）一起呈现 — paper 必须先告诉读者 "crosscomp 在这个场景下 success 是多少"，才能区分 "ReBRAC 是真有提升" 和 "ReBRAC 是平凡地模仿 crosscomp"。

**Optional online supplement**：online 线如果方便在 cross_u10 / s0 + s1 各跑 1 seed × 1M（~2h L4 total），就能闭合 sub-critical 对位。记入 §9 backlog。

### 8.3 与 main paper claim 的关系（明确不冲突）

- main paper claim "ReBRAC +23pp on crosscomp / cross_u10 / s0 / efficiency_v2" — **不变**
- v2 N0 结果用于声明 "this advantage persists under the redesigned `arrival_v2` reward (Section appendix)"
- v2 N2 + S 用于声明 "the deployable-sensor envelope identified by online SAC analysis extends (or doesn't extend) to offline ReBRAC" — 具体 claim 视 N2 数值与 §5.3 verdict gate
- v2 N4（若触发）用于声明 "even oracle teacher data fails to bridge the critical-regime ceiling under deployable obs"
- v2 M1（若触发）用于声明 "the ceiling is task-data fundamental, not a BC-penalty artifact"

---

## 9. Backlog（明确不在本广验范围内）

| Backlog item | 优先级 | 理由 |
|---|---|---|
| N1 sub-critical s1 sensor probe | 低 | v1 B1 已是 weak positive；sub-critical 下 sensor gap 本就小 |
| N3 critical s1 sensor rescue | 低 | online §7.6 已直接报告 cross_u15/s1 = 0.90 PASS；offline 重测仅闭合 2×2 表 |
| C3 tandem geometry × arrival_v2 | 低 | cross 优先级更高，且 v1 C3 std blow-up 机制未明 |
| upstream geometry × arrival_v2 × offline | 低 | online 已 saturate |
| mix dataset (cross_u15 dual-mode) | 低 | dual-mode 配方在 cross_u15 下未找到 |
| Online SAC trained collector | 待 spec | online 线 sprint 中 |
| s2 (4 probes) cells | 低 | sensor envelope 重心在 s0 vs s1 |
| Online cross_u10 s0/s1 1-seed 1 跑 | 中 | 闭合 §8.2 表的 sub-critical 行；~2h L4 |
| Flow regime intermediate (U=1.25 / Re~200) | 低 | 需 `generate_wake` 新跑 |
| asym critic ablation on stuck cell | 中（依赖 online 线） | 如先在 online 上做出来再 mirror 到 offline |
| 5-seed 补全（main cell 从 3 → 5） | 中 | 若审稿人 push back 3 seed power，对 N0/N2 各补 2 seed，约 4h L4 |

---

## 10. Cross-link 与文档影响

### 10.1 v1 文档处理（已加 SUPERSEDED banner）

| v1 文档 | 处理 |
|---|---|
| `docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md` | banner: "SUPERSEDED by v2 plan, retain as v1 design archive" |
| `docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md` | banner: 同上 |
| `docs/rebrac_broad_validation_report.md` (rev.2) | banner: "v1 实验结果 archive；v2 重启后保留作历史 reference；v1 finding 不进 paper" |
| `docs/rebrac_c1_s1_followup_report.md` | banner: 同上 |

### 10.2 main paper 文档（不变）

| 文档 | 处理 |
|---|---|
| `docs/rebrac_experiment_plan.md` (rev.8) | 不动 |
| `docs/rebrac_experiment_report.md` (rev.8) | 不动；§10A pointer 保留指向 v1 report |
| `docs/rebrac_mainline_review.md` (rev.3) | §3.5 顶层 pointer 指向本 v2 plan |
| `docs/rebrac_method_section_draft.md` | 不动 |
| `docs/rebrac_paper_writing_index.md` | 写作期 §experiments appendix 添加 v2 入口 |

### 10.3 line-level summary

| 文档 | 处理 |
|---|---|
| `docs/offline_rl_line_summary.md` | §3.3 broad validation 段落 "v1 archive + v2 active plan"；§4 backlog 更新 |

### 10.4 Online 线 cross-link

`docs/arrival_v2_experiment_report.md` §7.6 是 v2 narrative 的 online 平行证据来源；paper 写作时 §discussion 段明确引用。

---

## 11. 文件清单

### 11.1 待创建（NEW）

| Path | Role | Status |
|---|---|---|
| `scripts/relabel_rewards.py` | 复用 v1 dataset 但 reward 列重打（N0 dataset 走这条路径） | **TBD（新增）** |
| `scripts/broad_validation_v2_cell_registry.py` | N0 / N2 / S / N4 / M1 配置 single source of truth（简化版，5 个条目） | TBD |
| `scripts/run_offline_rebrac_broad_v2.sh` | core train driver（N0 / N2 / N4），skip-resume，复用 train_offline_rebrac | TBD |
| `scripts/run_broad_validation_v2_bc_sweep.sh` | M1 conditional sweep driver | TBD |
| `scripts/summarize_broad_validation_v2.py` | aggregation + verdict gate evaluation | TBD |
| `notebooks/rebrac_broad_validation_v2_pretrain.ipynb` | S sanity + N0 relabel + N2 collect（一个 notebook 串完 pre-train phase） | TBD |
| `notebooks/rebrac_broad_validation_v2_core.ipynb` | N0 + N2 训练 + verdict gate 判定 + conditional 决策 | TBD |
| `notebooks/rebrac_broad_validation_v2_conditional.ipynb` | N4 / M1（视触发） | TBD |
| `experiments/offline/rebrac/broad_validation_v2/<cell>/seed_*.json` | run results | TBD |
| `checkpoints/offline/rebrac/broad_validation_v2/<cell>/seed_*/` | checkpoints | TBD |
| `offline_data/crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` | N0 dataset（relabel 自 v1） | TBD |
| `offline_data/crosscomp_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000/` | N2 dataset（新收） | TBD |
| `offline_data/privileged_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000/` | N4 dataset（conditional 新收） | TBD（按需） |
| `benchmarks/single_u15_cross_tgt15.json` | 已存在（online §7.1 用过），复用 | EXISTS |
| `benchmarks/single_u10_cross_tgt15.json` | 已存在，复用 | EXISTS |

### 11.2 复用既有（不变）

| Path | Role |
|---|---|
| `auv_nav/rebrac.py` | ReBRAC agent；与 reward 解耦 |
| `auv_nav/reward.py` | `arrival_v2` reward preset 已 in-tree（commit `813096e`） |
| `auv_nav/baselines.py` | crosscomp + privileged policy classes |
| `scripts/train_offline_rebrac.py` | ReBRAC 训练入口 |
| `scripts/evaluate.py` | online policy rollout（S sanity 用） |
| `scripts/evaluate_offline.py` | manifest 评估 |
| `scripts/collect_offline_data.py` | offline data collection |
| `scripts/write_sanity_card.py` | dataset sanity card 工具 |
| `offline_data/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/` | v1 dataset，N0 relabel 的源 |

### 11.3 Wake data 检查（执行前必查）

执行前确认两份 wake 文件本地 + Drive 可用：
- `wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` (+ `_meta.json`) — 已用于 v1
- `wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy` (+ `_meta.json`) — online §7.6 已用，复用

如缺，按 `docs/generate_wake_usage.md` profile `navigation` / `single_u15` 重新生成。

---

## 12. 执行检查清单

执行前确认：
- [ ] 用户 review + approve 本 plan rev.2
- [ ] v1 文档已加 SUPERSEDED banner（spec / plan / report / c1_s1_followup）— 已完成（rev.1 期间）
- [ ] `offline_rl_line_summary.md` 已同步更新 §3.3 / §4
- [ ] `mainline_review.md` §3.5 pointer 已更新（指向 v2 plan + v1 archive）
- [ ] Wake data 两份文件已确认可用
- [ ] `auv_nav/reward.py` `arrival_v2` preset 在当前 branch 可用
- [ ] v1 N0-source dataset `offline_data/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/transitions.npz` 本地可用
- [ ] Colab L4 至少 2 session 预算可用（基础 6h + conditional 最坏 18h）

执行启动后逐 Stage 检查：
- [ ] **Step 1** S sanity 完成，crosscomp success rate ≥ 0.40（否则触发 §6.2 fallback）
- [ ] **Step 2** N0 dataset relabel 完成 + `sanity_card.json` 写入
- [ ] **Step 3** N2 dataset 新收完成 + `sanity_card.json` 写入
- [ ] **Core** N0 3-seed 完成 → §5.2 verdict 判定（≥ 0.70 proceed / 0.50-0.70 weak bridge / < 0.50 暂停）
- [ ] **Core** N2 3-seed 完成 → §5.3 verdict 判定 + 触发判定
- [ ] **Conditional**（若触发）N4 3-seed 完成
- [ ] **Conditional**（若触发）M1 15 runs 完成

paper 写作前最终检查：
- [ ] 所有 verdict gate 结果 documented in v2 report
- [ ] §8.2 critical regime head-to-head 表 fully populated（含 S baseline 数据）
- [ ] cross-link 到 `arrival_v2_experiment_report.md` §7.6 完整
- [ ] v1 archive cross-link 完整
- [ ] 3-seed 局限性在 v2 report 明确标注（且 backlog §9 5-seed 补全已记录）

---

**END of v2 plan rev.2**
