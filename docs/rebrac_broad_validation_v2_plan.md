# ReBRAC Broad Validation v2 — Cross-Only Spotlight under `arrival_v2`

> **文档版本**：2026-05-18 rev.1
> **状态**：active plan（未开跑）
> **作用**：取代 v1 broad validation 全部产出（spec / plan / report），把广验从「efficiency_v2 三轴 8 spoke + C1 deep-dive」收敛成「arrival_v2 cross-only 5 cell + 1 conditional mechanism sweep」。
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
4. [实验矩阵：5 cell core + 1 conditional sweep](#4-实验矩阵)
5. [Pre-commit verdict gates](#5-pre-commit-verdict-gates)
6. [Stage 流程：Stage 0 / 1 / 2](#6-stage-流程)
7. [触发判据 / std 阈值重设](#7-触发判据)
8. [Paper narrative role](#8-paper-narrative-role)
9. [Backlog（明确不在本广验范围内）](#9-backlog)
10. [Cross-link 与文档影响](#10-cross-link)
11. [文件清单（TBD / NEW）](#11-文件清单)

---

## 1. TL;DR

- ReBRAC paper main results 维持 `efficiency_v2` / `cross_u10` anchor 不变（rev.8 已 closed，不翻盘）
- v2 广验在 `arrival_v2` reward 下做 **5 cell core + 1 conditional mechanism sweep**，全部 cross-stream geometry
- 砍掉的 v1 内容：upstream / tandem / sbs geometry、goalseek / worldcomp / mix5050 collector、C1 deep-dive 5 ablation
- 增加的 v2 内容：(U, Re, λ) 两 regime 探针（sub-critical u10/Re150/λ=0.67 vs critical u15/Re250/λ=1.0）、与 online §7.6 形成 offline ↔ online 2×2×2 对位
- paper 角色：**reward-bridge appendix + deployability boundary map**，不是替代 main results 而是辅助说明 main finding 对 reward 选择的稳健性
- 预算：**~25–40h L4 + 5–6h CPU**（比 v1 ~50h L4 + 含 ablation 还省）

---

## 2. 动机

### 2.1 v1 的 reward 失配嫌疑

v1 全套实验跑在 `efficiency_v2` 下，但两条独立证据线都表明 `efficiency_v2` 在 upstream / 高难度 cell 上有 reward shaping 失配：

1. [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §3 — `efficiency_v2` 在 `single_u15_upstream` 上 vanilla SAC collapse 到 `success=0`（agent 学会快速出界），arrival_v2 修复到 1.000
2. v1 broad val C1 baseline termination 分布是 `goal=22.5 / timeout=77.5 / oob=0` — 这是 `efficiency_v2` 把 agent 钉在「保守憋边界」上的 footprint；reward swap 到 `arrival_v2_simple` 后翻成 `goal=21.5 / timeout=52.5 / oob=26`

含义：v1 C1 「task-fundamental floor at deployable sensors s0/s1」claim 受 reward bias 污染，**不能在不切换 reward 的前提下被升格为 paper-quality finding**。

### 2.2 online 线已经给出更强 sensor envelope finding

online [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.6 显示：在 `arrival_v2 / cross_u15` 下 vanilla SAC + s0 catastrophic FAIL（`final=0.10 / OOB=0.667`），s1 PASS（`final=0.9`）。**s0–s1 gap = 80pp**，相比 A0 (cross_u10 + arrival_v1) 的 3pp **放大 24×**。

这条 finding 在 v1 broad val（cross_u10, efficiency_v2）里完全看不到——v1 B1 (s1) untriggered, robust 是 weak positive；online 线的 80pp gap 才是 paper-grade sensor envelope claim。

**v2 直接把 online §7.6 作为 narrative spine**：构建一个 offline ReBRAC 在对应 cell 上的 head-to-head 表，paper §experiments 可以直接画 online vs offline 2×2 matrix（cross / s0,s1 × u10,u15）。

### 2.3 收敛到 cross-stream 的进一步理由

- upstream + arrival_v2 + vanilla SAC online 已 saturate 到 1.000（§7.2 single / §7.3 tandem / §7.4 sbs）— offline ReBRAC 上未必 saturate 但 ROI 低于 cross
- tandem / sbs 的几何 prior 在 cross_stream 主线 paper 里没承载 narrative 责任
- cross_stream 是 OOB-once-and-done regime，partial-observability 在这里显化 — sensor envelope 故事最干净

### 2.4 (U, Re, λ) 两 regime 的科学性 framing

v1 单档 (u10 / Re150 / λ=0.67) 是 sub-critical under-actuation；v2 增加的第二档 (u15 / Re250 / λ=1.0) 是 critical under-actuation regime。**两档是两个 physics regime 而非 continuous ladder**（涡街 Strouhal 频率 + wake 几何随 Re 变化）。v2 plan 明确按 "sub-critical vs critical regime comparison" 来 frame，paper 写作时不当 ladder 用。

如果未来需要 ladder 中间档（如 U=1.25 / Re=200），需要先用 `scripts/generate_wake.py` 跑新 LBM 设定，记入 backlog。

---

## 3. v1 → v2 关键变化

| 维度 | v1 (efficiency_v2) | v2 (arrival_v2) | 理由 |
|---|---|---|---|
| **Reward** | `efficiency_v2` | `arrival_v2`（8 参数 v6，commit `813096e`）— **不是** `arrival_v2_simple` (`bd37412`) | v1 §3.5.1 用 simple 版做 reward ablation，未测 full 版 |
| **Task geometry 轴** | C 轴 3 spoke（main cross + C1 upstream + C3 tandem）| 只 cross_stream | upstream saturate or low ROI；tandem 与主 narrative 重叠 |
| **Flow regime 轴** | 单档 `wake_v8_U1p00_Re150` | 双档：`wake_v8_U1p00_Re150` (sub-critical) + `wake_v8_U1p50_Re250` (critical) | online §7.6 显示 cross_u15 是 partial-obs gap 显化 regime |
| **Collector 集合** | 4 collector + 1 mix（goalseek / crosscomp / worldcomp / privileged / mix5050）| 2 collector：crosscomp + privileged | goalseek u15 cross 下失败 / worldcomp 与 main line Phase 2 重叠 / mix5050 在 u15 下假设不成立 |
| **Sensor 轴** | s0 / s1 / s2 | s0 / s1 | s2 在 online 故事中无角色，ROI 低 |
| **Mechanism discriminator** | C1 5 ablation 串联（reward swap → asym critic → epoch ×4 → sensor → convergence）| BC penalty sweep on **stuck cell**（conditional） | 直接、可证伪、与 v1 retrofit trigger condition 之一对齐 |
| **Anchor** | `efficiency_v2` 5-seed 0.902 ± 0.021（复用 main line Stage C）| `arrival_v2` 重测 5-seed（N0 cell）| 新 reward 必须新 anchor，否则 Δ 无意义 |
| **std_blow_up 阈值** | 0.042 = 2 × 旧 anchor std 0.021 | 2 × **新** N0 std（post-Stage 1 重设）| std 与 reward 紧耦合，必须 post-anchor 重设 |
| **A2 td3bc head-to-head** | 是（mix5050 上 +0.2pp finding 来源）| 砍掉 | mix5050 砍后无 head-to-head 必要 |
| **C1 deep-dive 5 ablation** | reward / asym critic / epoch / sensor / convergence | 砍掉 | 直接由 conditional BC sweep on stuck cell 替代 |
| **Paper role** | standalone exploratory，retrofit deferred | reward-bridge appendix + deployability boundary map | v2 明确 paper section，不再 "standalone TBD" |

---

## 4. 实验矩阵

**固定基底**：`cross_stream / target_speed=1.5 / arrival_v2 / probe_layout 见 cell / ReBRAC (β1=4, β2=2, hidden=256, critic_LN=on) / 64 epochs / 5 seeds [42, 43, 44, 45, 46] / test_episodes=100 / num_envs 见 cell`

**核心 5 cell**：

| Cell | Collector | Sensor | Flow file | (U, Re, λ) | 角色 |
|---|---|---|---|---|---|
| **N0** Anchor | crosscomp | s0 | `wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` | (1.0, 150, 0.67) sub-critical | reward-bridge baseline；与 main line `efficiency_v2 / 0.902` 对照 |
| **N1** Sensor probe | crosscomp | **s1** | 同 N0 | 同 N0 | s0/s1 等效性在 new reward 下是否仍立（v1 B1 finding 重测） |
| **N2** Critical regime | crosscomp | s0 | `wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy` | (1.5, 250, 1.0) critical | 与 online §7.6 `single_cross_s0` (vanilla SAC 0.10) 对位；offline 是否更稳 |
| **N3** Sensor rescue | crosscomp | **s1** | 同 N2 | 同 N2 | 与 online §7.6 `single_cross_s1` (vanilla SAC 0.90) 对位；sensor upgrade 能否 rescue offline |
| **N4** Priv teacher | **privileged** | s0 | 同 N2 | 同 N2 | teacher quality upper bound under deployable obs |

**Conditional M1 — BC penalty sweep**：

- **触发条件**：Stage 1 跑完 N0–N4 后，存在至少一个 cell 落在 `success ∈ [0.20, 0.60]`
- **配置**：选触发 cell 中 paper-priority 最高的（优先级 N2 > N4 > N3 > N1 > N0），β1 ∈ {0, 1, 2, 4, 8} × 3 seeds [42, 43, 44]，其余固定
- **目的**：区分「BC penalty floor (β-tunable)」vs「task-data fundamental floor (β-invariant)」
- **若无 stuck cell**：M1 skip，paper 写作时直接引用 N0–N4 结果，不强行做 mechanism

---

## 5. Pre-commit Verdict Gates

**所有 gate 在跑实验之前预登记，不允许 post-hoc 调整**（v1 教训：A3 1-seed → 5-seed 反转）。

### 5.1 N0 Anchor GO/NO-GO（最关键）

| N0 5-seed success | Verdict | 后续动作 |
|---:|---|---|
| ≥ 0.80 | valid anchor | N1–N4 按计划跑 |
| 0.50–0.80 | degraded anchor | N1–N4 仍跑，paper 写作时明确标 "arrival_v2 下 ReBRAC anchor 比 efficiency_v2 下退化 X pp"；不撤回 main line claim |
| < 0.50 | **reward bridge 失败** | **暂停**所有后续 cell；review reward / collector / dataset；可能整套设计推翻 |

### 5.2 N1 Sensor probe

| N1 vs N0 | Verdict |
|---|---|
| \|Δ\| < 5pp | s0/s1 等效在 new reward / sub-critical regime 下成立（v1 B1 finding robust） |
| N1 > N0 by ≥ 5pp | s1 在 sub-critical regime 下已显现 advantage（弱阳性） |
| N1 < N0 by ≥ 5pp | s1 在 sub-critical regime 下退化（unexpected，调查 sanity card） |

### 5.3 N2 Critical regime（与 online §7.6 对位）

| N2 success | Verdict | Paper claim 候选 |
|---:|---|---|
| ≥ 0.50 | offline ReBRAC 在 critical regime 比 vanilla SAC online (§7.6 = 0.10) 更稳 | "offline RL 提供 stability advantage in production-difficulty regime" |
| 0.30–0.50 | partial 一致 | "offline RL partial mitigation; both lines struggle" |
| < 0.30 | offline 与 online 同样 catastrophic | "critical regime is sensor-fundamental at s0 across both online and offline RL" |

### 5.4 N3 Sensor rescue（cross_u15 + s1）

| N3 success | Verdict |
|---:|---|
| ≥ 0.80 | sensor upgrade 在 critical regime 下 rescue |
| 0.50–0.80 | partial rescue |
| < 0.50 | sensor upgrade 不足以 rescue critical regime |

### 5.5 N4 Privileged teacher

| N4 success | Verdict |
|---:|---|
| ≥ 0.70 | teacher quality 在 critical regime 下能给出可学的 dataset；student (deployable obs) 能利用 |
| 0.50–0.70 | teacher 帮助有限 |
| < 0.50 | 即便 oracle data 在 critical regime 下也救不回来 → 强 "task-data fundamental ceiling" 证据 |

### 5.6 M1 BC penalty sweep（conditional）

| β1=0 success vs β1=4 baseline | Verdict |
|---|---|
| Δ > +10pp | BC penalty 是 limiting factor（floor 来自 BC 强度，非 task） |
| Δ ∈ [−5pp, +10pp] | β1 与 ceiling 解耦（floor 来自 task / data，非 BC） |
| Δ < −5pp | β1=0 退化（BC penalty 在 stuck cell 仍提供 stabilization） |

---

## 6. Stage 流程

### Stage 0 — Collector sanity + dataset 重收（必做）

#### Stage 0A: Collector sanity card（GO/NO-GO 关卡）

测 collector 在新 regime 下能否产生 viable dataset。每个组合跑 **online evaluation** 100 ep（不训练，直接评估 collector policy）：

| Sanity card | collector | sensor | flow | check |
|---|---|---|---|---|
| S0a-1 | crosscomp | s0 | wake_v8_U1p00_Re150 (cross) | success rate ≥ 0.4？ |
| S0a-2 | crosscomp | s1 | wake_v8_U1p00_Re150 (cross) | success rate ≥ 0.4？ |
| S0a-3 | crosscomp | s0 | wake_v8_U1p50_Re250 (cross) | **GO/NO-GO**: success rate ≥ 0.4 才能 proceed N2 |
| S0a-4 | crosscomp | s1 | wake_v8_U1p50_Re250 (cross) | success rate ≥ 0.4 才能 proceed N3 |
| S0a-5 | privileged | s0 | wake_v8_U1p50_Re250 (cross) | success rate ≥ 0.6 才能 proceed N4 |

**GO/NO-GO**：S0a-3 是关键卡。如果 crosscomp 自己在 (cross_u15, s0) 下 success < 0.4，N2 dataset 几乎全失败示范 → fallback option：
- (a) 降到 u12.5 / Re~200（需要 `generate_wake` 新跑，~1h CPU）
- (b) 接受 N2 dataset 退化，但明确 paper 中说明
- (c) 砍掉 N2/N3，v2 缩到 N0/N1/N4 三 cell（仍能讲 reward bridge + teacher gap 故事）

**成本**：5 × 100 ep × baseline policy（CPU）≈ **3–5h CPU**

#### Stage 0B: Dataset collection

通过 Stage 0A 后，收 5 个 dataset：

| # | dataset name | collector | sensor | flow | episodes | cost |
|---|---|---|---|---|---|---|
| 1 | `crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000` | crosscomp | s0 | wake_v8_U1p00_Re150 | 1000 | ~30 min CPU |
| 2 | `crosscomp_s1_h4_arrival_v2_re150_u10cross_fixdone_ep1000` | crosscomp | s1 | wake_v8_U1p00_Re150 | 1000 | ~30 min CPU |
| 3 | `crosscomp_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000` | crosscomp | s0 | wake_v8_U1p50_Re250 | 1000 | ~30 min CPU |
| 4 | `crosscomp_s1_h4_arrival_v2_re250_u15cross_fixdone_ep1000` | crosscomp | s1 | wake_v8_U1p50_Re250 | 1000 | ~30 min CPU |
| 5 | `privileged_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000` | privileged | s0 | wake_v8_U1p50_Re250 | 1000 | ~30 min CPU |

每个 dataset 必含：`transitions.npz`（含 `privileged_obs` / `next_privileged_obs` 列，为未来 asym critic ablation 留口）+ `metadata.json` + `sanity_card.json`。

**复用 v1 工具**：`scripts/collect_offline_data.py` + `scripts/write_sanity_card.py`，命令模板参考 v1 plan §Task 11A。

**成本**：5 dataset × ~30 min CPU = **~3 h CPU**（含 sanity card）

### Stage 1 — Core 5 cell × 5 seed（必做）

```bash
# 示例命令（per cell, per seed）
python -m scripts.train_offline_rebrac \
  --offline-data offline_data/<dataset_for_cell>/transitions.npz \
  --actor-bc 4.0 --critic-bc 2.0 \
  --hidden-dim 256 --critic-layernorm \
  --epochs 64 --batch-size 256 \
  --probe-layout <s0|s1> \
  --history-length 4 \
  --objective arrival_v2 \
  --eval-manifest benchmarks/<flow_geom>.json \
  --eval-episodes 100 \
  --seed <42|43|44|45|46> \
  --device cuda \
  --save-dir checkpoints/offline/rebrac/broad_validation_v2/<cell>/seed_<n>
```

**每 cell 流程**：train (64 ep) → validation (selection seeds {42–46}) → test (100 ep manifest)

**post-anchor std 重设**：N0 5-seed 跑完后，记录 `N0_std`；用 `2 × N0_std` 作为后续 N1–N4 的 `std_blow_up` 阈值（替换 v1 旧 0.042）。

**Skip-resume**：复用 v1 `[skip]` 逻辑（`trainer_state.json + agent_final.pt` 双文件存在性判断）。

**成本**：5 cell × 5 seed × ~1 h L4 = **~25 h L4**

### Stage 2 — BC penalty sweep M1（conditional, 0–15 h L4）

Stage 1 完成后判定：

```python
stuck_cells = [c for c in [N0, N1, N2, N3, N4]
               if 0.20 <= c.success_5seed_mean <= 0.60]
if not stuck_cells:
    skip M1
else:
    select_cell = max(stuck_cells, key=paper_priority)
    # priority: N2 > N4 > N3 > N1 > N0
```

**M1 配置**：选定 cell × β1 ∈ {0, 1, 2, 4, 8} × 3 seed [42, 43, 44] = **15 runs**，其余固定。

**成本**：**~15 h L4**（若触发）

### 总预算

| Stage | Cost | Runs |
|---|---|---|
| Stage 0A sanity | 3–5 h CPU | 5 evaluations |
| Stage 0B collect | 3 h CPU | 5 datasets |
| Stage 1 core | 25 h L4 | 25 train+test |
| Stage 2 (conditional) | 0–15 h L4 | 0 or 15 train+test |
| **Total** | **~28–40 h L4 + 6–8 h CPU** | **25–40 train + 5 collect + 5 sanity** |

对比 v1：~50 h L4 + ~16 train+test broad val + C1 ablation 5 runs。**v2 总预算 ~80%**，但 paper-quality finding 数量更高。

---

## 7. 触发判据

继承 v1 spec §6.1 框架，但 std 阈值动态化：

| 判据 | 阈值 | 触发后处理 |
|---|---|---|
| `mean_shift` | \|cell mean − N0 mean\| > **5pp** | 记录方向 + 是否进入 verdict gate 下一档 |
| `std_blow_up` | cell std > **2 × N0 std**（post-Stage 1 重设；不再用 v1 旧 0.042）| 记录但不撤回 cell |
| `verdict_gate_violation` | cell success 落入 pre-commit 表的下一档 | 按 §5 verdict gate 处理 |

v1 旧 `td3bc_gap_collapse` 判据（A2 专用）**砍掉**——v2 没有 mix dataset。

---

## 8. Paper Narrative Role

v2 在 paper 里的明确角色：**§experiments 的 reward-bridge appendix + §discussion 的 deployability boundary map**。

### 8.1 §experiments 引用方式

```
Main results (Section X.Y) report ReBRAC's +23.0 pp gain over TD3+BC on
`crosscomp / cross_u10 / s0 / efficiency_v2`. To verify this finding is not
an artifact of the `efficiency_v2` reward design (whose limitations in
upstream geometries are documented in [arrival_v2_experiment_report]),
we re-anchor under `arrival_v2` (cell N0) and probe sensor / flow regime
generality (cells N1–N4). [Insert main table + verdict summary.]
```

### 8.2 §discussion: online ↔ offline 对位 2×2 表

paper §discussion 关键素材（v2 独有的 paper hook）：

| Cell | sensor | flow regime | vanilla SAC online (§7) | offline ReBRAC (v2) | Δ |
|---|---|---|---|---|---|
| cross / s0 / u10 | s0 | sub-critical | (待补，1 seed ~1h L4) | **N0** | reward-bridge |
| cross / s1 / u10 | s1 | sub-critical | (待补，1 seed ~1h L4) | **N1** | sensor envelope (sub-critical) |
| cross / s0 / u15 | s0 | critical | **§7.6 = 0.10** (catastrophic FAIL) | **N2** | **offline stability advantage?** |
| cross / s1 / u15 | s1 | critical | **§7.1 = 0.90** (PASS borderline) | **N3** | sensor envelope (critical) |

**Optional online supplement**：online 线如果方便在 cross_u10 / s0 + s1 各跑 1 seed × 1M（~2h L4 total），就能闭合完整 2×2×2 表。这成本极低但 narrative 收益高。**记入 §9 backlog**。

### 8.3 与 main paper claim 的关系（明确不冲突）

- main paper claim "ReBRAC +23pp on crosscomp / cross_u10 / s0 / efficiency_v2" — **不变**
- v2 N0 结果用于声明 "this advantage persists under the redesigned `arrival_v2` reward (Section appendix)"
- v2 N2/N3 用于声明 "the deployable-sensor envelope identified by online SAC analysis extends to offline ReBRAC: critical regime stresses single-point sensors regardless of training paradigm"
- v2 N4 用于声明 "even oracle teacher data fails to bridge the critical-regime ceiling under deployable obs"
- v2 M1（若触发）用于声明 "the ceiling is task-data fundamental, not a BC-penalty artifact"

---

## 9. Backlog（明确不在本广验范围内）

| Backlog item | 优先级 | 理由 |
|---|---|---|
| C3 tandem geometry × arrival_v2 | 低 | cross 优先级更高，且 v1 C3 std blow-up 机制未明，重做边际信息低 |
| upstream geometry × arrival_v2 × offline | 低 | online 已 saturate；offline 上即便 unstable 也不在 paper main narrative 内 |
| mix dataset (cross_u15 dual-mode) | 低 | dual-mode 配方在 cross_u15 下未找到（goalseek 必败、crosscomp+priv 不是真 dual） |
| Online SAC trained collector | 待 spec | online 线 sprint 中，spec 未定 |
| s2 (4 probes, 16-D) cells | 低 | online §7.6 已表明 sensor envelope 故事重心在 s0 vs s1，s2 信息量过剩 |
| Online cross_u10 s0/s1 1-seed 各 1 跑 | 中 | 闭合 §8.2 的 2×2×2 表，~2h L4，narrative 收益高 |
| BC sweep on N3 (cross_u15 + s1) | 中（次选）| 若 M1 选了 N2，N3 是次选 sweep target |
| Flow regime intermediate probe (U=1.25 / Re~200) | 低 | 需要 `generate_wake` 新跑；如 paper 审稿人要 continuous transition 再补 |
| asym critic ablation on stuck cell | 中（依赖 online 线决定） | online 线 §8 P1#1 已列入；如先在 online 上做出来再 mirror 到 offline |

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
| `docs/rebrac_experiment_report.md` (rev.8) | 不动；§10A pointer 保留指向 v1 report（v1 是 archive） |
| `docs/rebrac_mainline_review.md` (rev.3) | §3.5 顶层 pointer 更新指向 v2 plan + v1 archive |
| `docs/rebrac_method_section_draft.md` | 不动 |
| `docs/rebrac_paper_writing_index.md` | 写作期 §experiments appendix 添加 v2 入口 |

### 10.3 line-level summary

| 文档 | 处理 |
|---|---|
| `docs/offline_rl_line_summary.md` | §3.3 broad validation 段落 重写为 "v1 archive + v2 active plan"；§4 backlog 更新 |

### 10.4 Online 线 cross-link

`docs/arrival_v2_experiment_report.md` §7.6 是 v2 narrative 的 online 平行证据来源；paper 写作时 §discussion 段明确引用。

---

## 11. 文件清单

### 11.1 待创建（NEW）

| Path | Role | Status |
|---|---|---|
| `scripts/broad_validation_v2_cell_registry.py` | 5 cell + flow + benchmark + dataset name 的 single source of truth | TBD |
| `scripts/run_offline_rebrac_broad_v2.sh` | Stage 1 driver（per cell, per seed），skip-resume，复用 train_offline_rebrac + evaluate_offline | TBD |
| `scripts/run_broad_validation_v2_bc_sweep.sh` | M1 conditional sweep driver | TBD |
| `scripts/summarize_broad_validation_v2.py` | Stage 1/2 aggregation + verdict gate evaluation | TBD |
| `notebooks/rebrac_broad_validation_v2_stage0_sanity.ipynb` | Stage 0A/0B（5 sanity + 5 collect） | TBD |
| `notebooks/rebrac_broad_validation_v2_stage1_core.ipynb` | Stage 1（5 cell × 5 seed） | TBD |
| `notebooks/rebrac_broad_validation_v2_stage2_bc_sweep.ipynb` | Stage 2 (conditional) | TBD |
| `experiments/offline/rebrac/broad_validation_v2/<cell>/<actor_critic_pair>/test/seed_*.json` | run results | TBD |
| `checkpoints/offline/rebrac/broad_validation_v2/<cell>/<actor_critic_pair>/seed_*/` | checkpoints | TBD |
| `offline_data/{crosscomp,privileged}_<sensor>_h4_arrival_v2_<flow>_fixdone_ep1000/` | 5 datasets + sanity_card.json | TBD |
| `benchmarks/single_u15_cross_tgt15.json` | 已存在（online §7.1 用过），复用 | EXISTS |
| `benchmarks/single_u10_cross_tgt15.json` | 已存在，复用 | EXISTS |

### 11.2 复用既有（不变）

| Path | Role |
|---|---|
| `auv_nav/rebrac.py` | ReBRAC agent；与 reward 解耦 |
| `auv_nav/reward.py` | `arrival_v2` reward preset 已 in-tree（commit `813096e`） |
| `auv_nav/baselines.py` | crosscomp + privileged policy classes |
| `scripts/train_offline_rebrac.py` | ReBRAC 训练入口 |
| `scripts/evaluate_offline.py` | manifest 评估 |
| `scripts/collect_offline_data.py` | offline data collection |
| `scripts/write_sanity_card.py` | dataset sanity card 工具 |

### 11.3 Wake data 检查（执行前必查）

执行前确认两份 wake 文件本地 + Drive 可用：
- `wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` (+ `_meta.json`)
- `wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy` (+ `_meta.json`)

如缺，按 `docs/generate_wake_usage.md` profile `navigation` / `single_u15` 重新生成。

---

## 12. 执行检查清单（plan owner 用）

执行前确认：
- [ ] 用户 review + approve 本 plan rev.1
- [ ] v1 文档已加 SUPERSEDED banner（spec / plan / report / c1_s1_followup）
- [ ] `offline_rl_line_summary.md` 已同步更新 §3.3 / §4
- [ ] `mainline_review.md` §3.5 pointer 已更新（指向 v2 plan + v1 archive）
- [ ] Wake data 两份文件已确认可用
- [ ] `auv_nav/reward.py` `arrival_v2` preset 在当前 branch 可用
- [ ] Colab L4 至少 3 session 预算可用

执行启动后逐 Stage 检查：
- [ ] Stage 0A 5 sanity card 写入 `offline_data/<dataset>/sanity_card.json`
- [ ] Stage 0A S0a-3 GO/NO-GO 通过（crosscomp / s0 / u15 success ≥ 0.4），否则 fallback
- [ ] Stage 0B 5 dataset 收集完成 + sanity card OK
- [ ] Stage 1 N0 跑完，重设 std_blow_up 阈值 = 2 × N0 std
- [ ] Stage 1 N0 GO/NO-GO 通过（N0 success ≥ 0.5），否则暂停后续
- [ ] Stage 1 N1–N4 完成
- [ ] Stage 1 完成后判定 M1 是否触发
- [ ] Stage 2（若触发）M1 完成

paper 写作前最终检查：
- [ ] 所有 verdict gate 结果 documented in v2 report
- [ ] online ↔ offline 2×2 表 fully populated（或 backlog 标记缺口）
- [ ] cross-link 到 `arrival_v2_experiment_report.md` §7.6 完整
- [ ] v1 archive cross-link 完整

---

**END of v2 plan rev.1**
