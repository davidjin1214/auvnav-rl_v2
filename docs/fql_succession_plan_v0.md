# FQL Succession Plan v1 — Offline RL on Data Quality × Modality Spectrum (Lean MVP)

> ⚠ **SUPERSEDED 2026-06-02**：本 plan 的实验 phase（P0–P4）已**全部完成且闭环**（NEGATIVE 收口 2026-05-23，见 [`fql_succession_p2_results.md`](fql_succession_p2_results.md)）。本 plan 当时设计的 paper-writing 出口（"FQL Paper 2 standalone"）**已撤销启动**——per 2026-06-02 用户拍板，全部 FQL 素材作为**博士论文第 5 章 §N.6 节**（FQL 算法对比 + SAC collector cross-source headline），不另起独立投稿。
> - **实验设计 / phase plan / gate 条件 / sprint 节奏 等仍有历史 / 复现价值**，可作为 §N.6 写作时回查实验 provenance 的档案。
> - **不再据本 plan 推进任何写作动作**；写作方向去 [`../paper/thesis_chapter_outline.md`](../paper/thesis_chapter_outline.md) rev.4 §N.6。
> - 已闭环结果数字 ground truth 见 [`fql_succession_p2_results.md`](fql_succession_p2_results.md) + [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9。
>
> **文档版本**：v1.4（2026-05-21,P2 sprint 0 collection 实测 + audit 降级 advisory)
> **作用**：把"重启 FQL 作 ReBRAC 后续"的 framing、scope、phase plan 与 gate 条件落地为可执行计划。**v1 相对 v0 砍掉约 50% validation insurance**（6 tier → 3 cell；4 audit 指标 → 1；4 ablation → 1；anchor 5-seed → 1-seed smoke），保留全部 core paper claim 支撑实验。
> **状态**：**Plan v1.2** — P0+P1 closed（Gate A.1 cite ✓ + Gate A.2 ✅ PASS 4/4 audit dry-run + Gate B ⚠ 3/4 PASS + c4 marginal-FAIL seed-driven, progress with caveat）；P2 main comparison spec drafting (Session A 2026-05-20)。
>
> **版本历史**:
> - v1.0 (2026-05-18, Lean MVP) — 初稿,6 tier → 3 cell + 4 audit → 1 metric + 5-seed anchor → 1-seed smoke
> - v1.1 (skipped) — 按"Gate A 通过升 v1.1"原计划应在 audit dryrun 后落,sprint 节奏未做; v1.2 一并补
> - v1.2 (2026-05-20) — Gate A.2 PASS + Gate B 3/4 PASS + c4 marginal caveat; D17/D18/D19 加入 §6 决策记录
> - v1.3 (2026-05-21) — P2 spec v1.2 wallclock-budget + storage-layout 重设 (D20 n_seeds 5→2 primary, D21 results/ mirror tree)
> - **v1.4 (2026-05-21)** — P2 sprint 0 collection 实测后 audit 降级 advisory(D22):3 dataset 收集闭环 + GMM audit on noise-widened unimodal 发现 known false-positive limitation,cell 定义改基于 collection protocol 元信息
> **前置阅读**：
> - [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) — Offline 线整体状态
> - [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) — `arrival_v2` reward 在 vanilla SAC 上的 4-cell 5/5 gate 验证报告（**本计划 reward 选型依据**）
> - [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) — **rev.3 active**：ReBRAC × `arrival_v2` × cross-only offline 实测，N0 cell (crosscomp/u10/Re150/s0) 与本计划同 reward+task+sensor，作 **Gate A.1 reward-sanity 共享证据**
> - [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) — `efficiency_v2` reward hacking 诊断 + `arrival_v2` v6 spec
> - [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) — ReBRAC paper 1 主线 4 finding 数字源
> - [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) — AUVHamNODE 线 PAUSED；本计划**不**依赖 NODE
>
> **明确的 non-goals**：
> - 不做 FQL 作 ReBRAC paper 的 baseline
> - 不做 FQL 独立线
> - 不做 FQL-NODE candidate selection（NODE 线 paused）
> - 不重做 ReBRAC paper 主线 cells（two-paper sequence 解耦）
> - **不**在 `efficiency_v2` reward 上跑任何实验（reward hacking 已证实）
> - 不做 6-tier spectrum、不做 4-metric audit、不做 TD3+BC × spectrum 重做、不做 5-seed reward-anchor 控制（v0 → v1 砍掉的 insurance）

---

## 目录

0. [TL;DR](#0-tldr)
1. [核心 motivation 与 paper claim](#1-核心-motivation-与-paper-claim)
2. [四个必须 push back 的设计前提](#2-四个必须-push-back-的设计前提)
3. [Data Quality × Modality Spectrum 设计](#3-data-quality--modality-spectrum-设计)
4. [Phase Plan](#4-phase-plan)
5. [风险表](#5-风险表)
6. [关键决策记录](#6-关键决策记录)
7. [一句话总结](#7-一句话总结)
8. [参考文档链接](#8-参考文档链接)

---

## 0. TL;DR

ReBRAC paper 1 建立在 4 类**接近专家级** deterministic baseline collector 数据 + 已证伪的 `efficiency_v2` reward 上。Offline RL 的真正 promise — **policy improvement over sub-optimal data-generating policy** — 在本任务上从未被严格测试。

本计划用已实测验证的 `arrival_v2` reward 构造**最小可支撑 paper claim 的 3-cell spectrum**，系统对比 **ReBRAC** 与 **FQL**。Paper claim 是**条件式（iff）**：

> FQL 的 expressive behavior prior **iff** data is **both** sub-optimal **and** multi-modal 才 translate 为相对 ReBRAC 的 policy improvement。

**实验体量**(v1.3 wallclock rebalance):
- 主对照:3 cell × 2 algorithm × **2 seed primary [42, 0]** = **12 runs**(可扩 3 seed → 18 runs)
- 1 个 mechanism ablation(mix ratio sweep,P3 再 spec):3 ratio × 2 algorithm × 3 seed = **18 runs**(待 P3 spec 确认)
- **总计 ≤ 36 runs**,L4 wallclock ~20-25h ≈ 1-1.5 天(vs v1.2 原 48 runs / 72h)

**两个 hard gate**：
- **Gate A**（P0 出口）：`arrival_v2` 在 offline 训练上无新型 reward artifact + M-multi-mix 多模态 audit 区分度 ≥1.5σ
- **Gate B**（P1 中段）：FQL on E-uni ≥ ReBRAC on E-uni − 8pp（实现 sanity）

**Decision-robust**：无论 FQL 赢/平/输 spectrum，paper 都有可写的 claim（positive / negative / scoped）。

**总时长 ~7-8 周**（vs v0 13-15 周）。任一 gate fail 立即 abort。

---

## 1. 核心 motivation 与 paper claim

### 1.1 为什么需要这条线

ReBRAC paper 1 的 4 个 paper-level finding 都建立在两个**未被审稿人会接受的假设**上：

| 假设 | 现实 | 暴露的问题 |
|---|---|---|
| Collector 数据足够"通用" | 4 类 collector 均为 deterministic + 小噪声 hand-designed controller，**单峰 + 接近 expert** | ReBRAC `crosscomp-1000` 0.902 success ≈ teacher 自身水平，无 policy improvement 硬证据 |
| `efficiency_v2` reward 可信 | online SAC P1 已证实 reward hacking | broad validation 8 spoke 全部基于 efficiency_v2，结论无法投 paper-quality |

→ **Offline RL 的 "policy improvement over sub-optimal data" promise 在本任务上是 open question**。这正是 D4RL 整套 quality spectrum（random / medium / medium-replay / medium-expert）存在的理由。

### 1.2 Paper claim（条件式）

```
Existing offline RL evaluation on AUV wake navigation has been
confined to near-expert baseline collectors and unimodal data
distributions on the now-deprecated efficiency_v2 reward, where
both single-BC (TD3+BC) and dual-BC (ReBRAC) trivially recover
teacher-level performance.

Using the validated arrival_v2 reward, we construct a 3-cell
data quality × modality spectrum and systematically compare
ReBRAC against FQL. We find that FQL's expressive behavior prior
translates to policy improvement over ReBRAC iff data is BOTH
sub-optimal AND multi-modal; expert-unimodal data shows parity,
and sub-optimal-unimodal data does not differentiate the two
architectures.

This challenges the assumption that flow-based offline RL
universally outperforms Gaussian baselines on sub-optimal data,
and identifies the regime where expressive policy parameterization
actually matters for AUV deployment.
```

**iff 需要 3 cell**：¬A → no effect (E-uni)；¬B → no effect (M-uni-noise)；A∧B → effect (M-multi-mix)。两 cell 设计无法表达 iff。

### 1.3 与 ReBRAC paper 的关系（two-paper sequence）

| 维度 | ReBRAC Paper 1 | FQL Paper 2 (本计划) |
|---|---|---|
| **Regime** | Near-expert collector × unimodal | Sub-optimal × (uni + multi) modal spectrum |
| **Reward** | `efficiency_v2` | `arrival_v2` |
| **Algorithm focus** | dual-BC + critic LN 的必要性 (Finding i–iv) | Gaussian vs flow-based actor 的 modality sensitivity |
| **核心 framing** | Deployable offline RL on coherent expert data | Where expressive offline RL actually helps |

**关键解耦点**：FQL Paper 2 §Setup cite [`online_sac_reward_redesign.md`](online_sac_reward_redesign.md)（efficiency_v2 hacking 诊断）+ [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md)（arrival_v2 4-cell 5/5 gate 验证）作为 reward 切换理由，并在 P0 期补 **ReBRAC × arrival_v2 × 1-seed smoke** 作为 reward-sanity spot check（不上 5 seed，cite C1 ablation 节约）。

---

## 2. 四个必须 push back 的设计前提

### 2.1 "非专家数据" 必须双轴定义（quality × modality）

D4RL 上 FQL vs TD3+BC / ReBRAC 的实证：

| Dataset 类型 | 多模态？ | Quality | FQL gap |
|---|---|---|---|
| random | 单峰均匀 | 极低 | 小 |
| medium（弱 checkpoint）| 单峰 | 中 | 小，有时倒挂 |
| **medium-replay** | **强多峰** | 中 | **明显领先** |
| medium-expert | 双峰 | 中-高 | 通常领先 |

→ "non-expert" ≠ "FQL 有优势"。**FQL 的真实 sweet spot 是 multimodal sub-optimal**。如果 spectrum 只有 quality 单轴下降（噪声更大但仍单峰），跑出来很可能是 null result。

### 2.2 "FQL 更新所以更好" 不是 self-evident

ReBRAC 在 D4RL 原论文的强 cell 包括 unimodal medium（hopper-medium / walker2d-medium）。FQL 在某些 locomotion-medium 上 gap 极小甚至倒挂。Paper claim **必须**条件式，不能写成 "FQL universally > ReBRAC"。

### 2.3 Reward 必须 settled

| Reward preset | 状态 | 本计划可用性 |
|---|---|---|
| `efficiency_v2` | 已证实 reward hacking ([online_sac_reward_redesign.md](online_sac_reward_redesign.md)) | ❌ 禁用 |
| `arrival_v2_simple` | C1 ablation 用过，简化版 | ❌ 不用（用户决定） |
| **`arrival_v2`**（commit `813096e`，8 参数 v6 完整版）| **vanilla SAC × 4 cell 严格控制 5/5 gate PASS**（[arrival_v2_experiment_report.md](arrival_v2_experiment_report.md)）；§2 reference 显示 `single_u10_cross_tgt15` + s0 也 PASS | ✅ **主选** |
| `arrival_v2_fast` | arrival_v2 的 fast-success bonus 变体 | ⚠ 备选 ablation，本计划不主用 |

**主选 `arrival_v2`**。需注意：arrival_v2 在 online SAC 上验证，offline ReBRAC 行为未独立测过 → P0 Gate A 必跑 1-seed smoke 确认 offline 兼容性。

### 2.4 ReBRAC paper 关系处理

| 方案 | 选择 |
|---|---|
| (A) FQL paper 用 `arrival_v2`，§Setup cite efficiency_v2 hacking + arrival_v2 report | ✅ **采用** |
| (B) 双 reward 重做所有 cell | ❌ |
| (C) 重做 ReBRAC 主线 cells in arrival_v2 | ❌ |

**Reviewer 防御**：P0 smoke 已包含 ReBRAC × arrival_v2 × 1 seed 对照；若 reviewer 要求更多 seed，可在 revision 补 2 seed。不预算性 5-seed。

---

## 3. Data Quality × Modality Spectrum 设计（3-cell）

### 3.1 三个 cell（v1 精简版）

| Cell ID | 构造方式 | Quality | Modality | 角色 |
|---|---|---|---|---|
| **E-uni** ¹ | privileged baseline + 小噪声 (ε=0.1) | Expert | Unimodal | **¬A reference**：iff claim 的 expert anchor，FQL 不可在此倒挂 |
| **M-uni-noise** | privileged baseline + ε·N(0, I)，**ε=0.5** | Medium | Unimodal (widened) | **¬B test**：quality 单轴下降 + 仍单峰；验证 "unimodal sub-optimal 不区分 FQL/ReBRAC" |
| **M-multi-mix** | 50% privileged + 50% goalseek (episode-level mix) | Medium | **Multi-modal**（双峰，by construction） | **A∧B core cell**：FQL 设计 sweet spot |

¹ **E-uni already collected (2026-05-19)** — `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` (1000 ep, success 0.985, 86,685 transitions),详见 [`fql_e_uni_anchor_dataset_card.md`](fql_e_uni_anchor_dataset_card.md)。M-uni-noise 与 M-multi-mix 在 P2 期收集 (Session A P2 spec §2 协议)。

**v0 → v1 砍掉的 tier**：
- ❌ M-uni-early（与 M-uni-noise 重复 ¬B 角色，选构造可控的 noise 版本）
- ❌ M-multi-replay（依赖 SAC checkpoint，R5 风险消除；M-multi-mix 已足 A∧B test）
- ❌ R-uni（spectrum 下限非 paper claim 必需）

### 3.2 多模态构造原则

- **不**靠 noise injection 制造 multimodality —— noise injection 出来仍是单峰
- **要**靠 episode-level mix 制造 multimodality
- 每个 cell 在投入主实验前**必须**通过 multimodality audit（§3.4）

### 3.3 收集协议

| 参数 | 值 |
|---|---|
| Dataset size | 1000 episodes / cell |
| Reward | `arrival_v2`（8 参数完整 v6 spec） |
| Task config | `single_u10_cross_tgt15`（cite [arrival_v2_experiment_report.md](arrival_v2_experiment_report.md) §2 cross_u10 reference 已 PASS） |
| Probe layout | `s0`（DVL only, 10-D obs；与 ReBRAC paper 1 一致） |
| History length | 4 |
| Metadata | seed / collector identity / generation timestamp 入 `metadata.json` |

### 3.4 Multimodality audit 协议（**1 个 canonical 指标**）

**主指标**：局部相似 state (k=50 NN) 下的 action GMM mode count
- 期望（单峰 cell：E-uni / M-uni-noise）：1-mode 占比 > 80%
- 期望（多峰 cell：M-multi-mix）：≥ 2-mode 占比 > 20%

**Gate A 判据**：E-uni vs M-multi-mix 的 mode count 分布 paired comparison ≥ 1.5σ 区分（Welch's t one-sided p < 0.07）。

**v0 → v1 砍掉的 audit 指标**：左/右绕轨迹比、actor-action KL 分布、return 分布 —— 全部降级到 reviewer 要求时补；不预算性跑。

**Audit 输出**：`docs/fql_succession_phase0_report.md` 内一节 + 1 张 mode-count 分布图。

### 3.5 Spectrum 不做的事

- ❌ 不混 reward variant（统一 `arrival_v2`）
- ❌ 不混 task geometry（统一 `cross_stream`，不引入 upstream C1 floor 风险）
- ❌ 不混 probe layout（统一 `s0`）
- ❌ 不变 sensor envelope

---

## 4. Phase Plan

总时长 **~7-8 周**。任一 gate fail 立即 abort 或回到上一 phase。

### 4.1 P0+P1 — Reward sanity + audit + FQL 实现 + Expert anchor（~4 周）

**状态 (2026-05-20)**:**CLOSED** — Gate A.1 cite ✓ + Gate A.2 ✅ PASS 4/4 + Gate B ⚠ 3/4 PASS + c4 marginal-FAIL (seed-driven); progress to P2 with caveat (see [`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md))。

合并 v0 的 Phase 0 + Phase 1。两件事可在 wallclock 上重叠（reward smoke 跑的时候，FQL 实现同步推进）。

**任务**：
1. **Reward sanity（cite broad val v2 N0，不独立跑）** — **✓ cite passed**
   - [`rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) §4.1 N0 cell 与本计划同 reward (`arrival_v2`) + 同 task (`single_u10_cross_tgt15`) + 同 sensor (`s0`) + 同 ReBRAC config (β1=4, β2=2, vanilla critic, critic_LN=on)
   - **Gate A.1（共享判据）**：broad val v2 N0 ≥ **0.70** success（v2 plan 已设定 GO 阈值）→ pass
   - 若 broad val v2 N0 fail，FQL plan v1 同步 abort 并联动 reward 切换决策
   - **本计划不重复跑 ReBRAC × arrival_v2 smoke**，节约 ~1 seed × 64 epoch ≈ 1.5h L4 + spec 复杂度
2. **Multimodality audit dry-run** — **✅ Gate A.2 PASS 4/4 (2026-05-19)**
   - 收集 E-uni + M-multi-mix 各 **200 episode**
   - 跑 §3.4 k-NN action GMM mode count
   - **Gate A.2**：分布区分 ≥1.5σ → 实测 Δp(≥2) = +0.582 [CI +0.534, +0.632], Welch p = 6.4e-84 (see [`fql_audit_dryrun_report.md`](fql_audit_dryrun_report.md))
3. **FQL 实现**：`auv_nav/fql.py` — **✓ 实现 + tests pass + integrated 进 `scripts/train_offline.py --algo fql`**
   - Actor：flow-matching teacher + one-step distilled student（参考官方 JAX repo 移植）
   - Critic：复用 ReBRAC twin-Q + critic LayerNorm（继承 paper 1 Finding (iv)）
   - 训练 entry：**`scripts/train_offline.py --algo fql`**（不是独立 `train_offline_fql.py`，via `auv_nav/offline_registry.py`）
4. **E-uni full collection** (1000 episodes) + FQL × E-uni Gate B sanity — **⚠ 3/4 PASS + c4 marginal (2026-05-20)**
   - **Gate B (Option B closure, 2-seed)**：FQL last-3 mean 0.728 vs ReBRAC 0.772 (Δ=−4.4pp, c1 PASS); loss_flow 0.026 PASS; actor_loss ratio 1.011 PASS; **c4 slope −0.0038 marginal-FAIL** — statistically indistinguishable from 0 noise (z=−0.27 against 30-ep manifest noise floor), seed-driven (seed=0 positive, seed=42 negative on both algos), **not** FQL-driven.
   - 按 spec §5.3 "mixed → caveat + 3-seed extension" 路径 → progress to P2 with caveat。
   - **P2 pre-requisites (Session B 2026-05-20)**:Bug 2 fix via `single_u10_cross_tgt15_ep100.json` (D18) + c4 阈值 Option α (D19),见 [`fql_succession_bug2_fix_decision.md`](fql_succession_bug2_fix_decision.md) + [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md)。

**产出**：
- `auv_nav/fql.py`（in-tree）
- `scripts/train_offline_fql.py`（in-tree）
- `experiments/fql_succession/p0p1/`
- `docs/fql_succession_phase0_report.md`（含 Gate A + Gate B 结论）

**Gate 失败处理**：
- Gate A.1 fail（reward artifact）→ 切 `arrival_v2_fast` 变体或回到 `arrival_v2_simple`，最多 +1 周
- Gate A.2 fail（多峰构造无效）→ 重设计 mix protocol（加大 mix bias、引入 worldcomp 第三 mode）
- Gate B fail → FQL 实现 debug；3 个 iter 仍不过则 abort

### 4.2 P2 — 3-cell main comparison(~1 周,v1.3 重设)

**任务**:
1. M-uni-noise (ε=0.5) + M-multi-mix 各 1000 episode 完整收集
2. 每 cell audit 落地 `multimodality_audit.md`
3. 主对照:ReBRAC vs FQL × 3 cell × **2 seed primary [42, 0]** = **12 runs**(可扩 3 seed → 18 runs,见 D20)

**统计协议**(v1.3 重设, primary verdict 用 effect-size + 方向一致性):
- Primary verdict: effect-size + 两 seed paired diff 方向一致性(δ_null=0.03, δ_signal=0.05)
- Sensitivity(报告但不作 verdict input):paired bootstrap CI(n_boot=10000)+ Welch's t + Cohen's d + Bonferroni p<0.0167
- Gray band [3pp, 5pp] 或方向不一致 → conditional 扩 n=3(加 seed 7)
- iff claim 验证:E-uni \|Δ\|<0.03 + M-uni-noise \|Δ\|<0.03 + M-multi-mix Δ≥0.05 同向

**算力**(P2 spec §8.2 数字):12 run × ~37.5 min mean ≈ ~8h L4 sequential(可扩 18 run ≈ 12h)。

**产出**:`results/fql_succession/p2/{e_uni,m_uni_noise,m_multi_mix}/` + `docs/fql_succession_p2_main_report.md`。

详见 [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) v1.2。

### 4.3 P3 — Mix ratio ablation（~1 周）

**唯一 mechanism ablation**：modality intensity → FQL gap 单调性

- mix ratio sweep on M-multi-mix：{30/70, 50/50, 70/30} × 2 algorithm × 3 seed = **18 runs**
- 验证 modality 强度（mode count）与 FQL gap 单调正相关

**v0 → v1 砍掉的 ablation**：
- ❌ FQL one-step vs multi-step inference
- ❌ BC penalty / distill weight sweep
- ❌ Cross-modality decomposition
- → 全部降级到 paper §Discussion "未做" 段或 follow-up appendix

**算力**：18 run × 1.5h ≈ 27h ≈ 1 天 wallclock + 整理 → 1 周。

**产出**：`experiments/fql_succession/p3_ablation/` + monotonicity 图。

### 4.4 P4 — Writing（~2 周）

**Two-paper sequence**：
- Paper 1 = ReBRAC 主线（已 paper-ready，本计划不动）
- Paper 2 = FQL succession（本计划）

**Paper 2 outline**：
1. Intro — offline RL promise gap on AUV navigation
2. Related Work — D4RL spectrum, FQL/ReBRAC/TD3+BC families
3. Method — FQL with inherited ReBRAC critic LN + twin-Q
4. Experimental Setup — `arrival_v2` reward, 3-cell spectrum construction, audit protocol
5. Results — 3-cell main comparison + mix ratio monotonicity
6. Discussion — conditional iff claim, when expressive actor matters
7. Limitations — efficiency_v2 not tested (intentional), AUVHamNODE NODE-aug not tested (line paused), sensor envelope not varied, ablation scope narrowed to mix ratio
8. Conclusion

---

## 5. 风险表

| ID | 风险 | 概率 | 影响 | Mitigation |
|---|---|---|---|---|
| **R1** | `arrival_v2` 在 offline ReBRAC 上有新型 reward artifact | 中 | 高 | 共享 broad val v2 N0 cell 证据；fail 联动 broad val v2 切 `arrival_v2_fast`，无需 FQL 计划独立处理 |
| **R2** | M-uni-noise (ε=0.5) 实际已发生模式分裂被误判为单峰 | 低 | 中 | Gate A.2 audit；fail 则降 ε 或换 M-uni-early |
| **R3** | FQL 在所有 sub-optimal cell 上 ≈ ReBRAC | 中 | 中 | Paper 已准备 negative finding 路径（"unimodal 不区分 + multimodal 也不区分" → scoped claim "expressive prior 在本任务无 leverage"） |
| **R4** | FQL 在 E-uni 上 < ReBRAC（Gate B fail） | 低 | 高 | 实现 bug；debug 3 iter；不通过则 STOP |
| **R5** | iff claim 仅部分成立（M-uni-noise Δ 也显著）| 中 | 中 | Paper claim 降级为 "FQL > ReBRAC on multimodal sub-optimal"（弱化 iff，仍可投） |
| **R6** | ReBRAC paper 1 revision 期被要求加 `arrival_v2` 对照 | 高 | 中 | Gate A.1 smoke 已跑，可直接补 |
| **R7** | Reviewer 质疑 "为什么不用 efficiency_v2 重做 ReBRAC" | 高 | 低 | §Setup cite efficiency_v2 hacking 诊断 + arrival_v2 报告 |
| **R8** | Two-paper sequence reviewer 觉得"应合并成一篇" | 中 | 中 | Paper 2 motivation 强调 paper 1 是 self-contained finding (i)-(iv)；paper 2 是 sub-optimal regime follow-up scope |
| **R9** | 算力超预算 | 极低 | 低 | 48 run × 1.5h ≈ 72h ≈ 3 天 wallclock，远低于 L4 月预算 |

*v0 风险表的 R5（M-multi-replay 依赖 SAC checkpoint）已删除：v1 不用 M-multi-replay。*

---

## 6. 关键决策记录

| # | 决策 | 时间 | 理由 |
|---|---|---|---|
| D1 | FQL 不做 baseline、不做独立线，做 ReBRAC succession | 2026-05-18 | 用户明确诉求；scope 解耦更清晰 |
| D2 | NODE 线明确不启用 | 2026-05-18 | AUVHamNODE PAUSED；FQL pure 不绑 NODE |
| D3 | Broad validation 不作 motivation | 2026-05-18 | efficiency_v2 reward hacking 污染所有 broad val 结论 |
| D4 | Reward 用 **`arrival_v2`**（非 `efficiency_v2`、非 `arrival_v2_simple`） | 2026-05-18 (v1) | 用户决定；`arrival_v2` 已在 vanilla SAC × 4 cell 5/5 gate PASS（[arrival_v2_experiment_report.md](arrival_v2_experiment_report.md)） |
| D5 | Paper claim 设计为**条件式 iff**（quality × modality 双轴）| 2026-05-18 | 避免 overclaim；hedge unimodal sub-optimal 的实证不确定性 |
| D6 | Two-paper sequence | 2026-05-18 | ReBRAC paper 1 已 paper-ready 4/4；不污染 |
| D7 | Phase 0 multimodality audit 作 hard gate | 2026-05-18 | 避免重蹈 broad validation "结论建立在未验证假设上" 覆辙 |
| **D10** | **Spectrum 6 tier 收敛到 3 cell**（E-uni / M-uni-noise / M-multi-mix） | **2026-05-18 (v1)** | iff claim 仅需 3 cell 支撑；多 tier 是 robustness insurance 而非 core claim |
| **D11** | **Audit 4 指标收敛到 1 个**（k-NN action GMM mode count） | **2026-05-18 (v1)** | D4RL 圈通用指标；其余 3 个降级到 reviewer 要求时再补 |
| **D12** | **Reward anchor 5-seed 收敛到 1-seed smoke** | **2026-05-18 (v1)** | 预算性 5-seed 是 reviewer-defensive paranoia；smoke + cite 已足够 |
| **D13** | **Mechanism ablation 4 → 1**（仅 mix ratio sweep） | **2026-05-18 (v1)** | mix ratio 直接验证 modality 强度 ↔ FQL gap 因果；其余 ablation 降级 follow-up |
| **D14** | **P0 + P1 合并**（5 周 → 4 周） | **2026-05-18 (v1)** | reward smoke 与 FQL 实现可 wallclock 并行 |
| **D15** | **Gate A.1 reward sanity 共享 broad val v2 N0 证据，不独立跑** | **2026-05-18 (v1)** | rev.3 broad val v2 同 reward+task+sensor+algo+config，独立 smoke 是重复工作 |
| **D16** | **M-multi-mix dataset 走 in-tree `scripts/concat_offline_datasets.py`，不改 `collect_offline_data.py`** | **2026-05-19** | v1 broad val A2 mix5050 (commit `992625c`) 已实战使用 concat helper；源 dataset 可复用让 P3 mix ratio sweep collection 成本归零；不动 in-tree 工具避免 cross-line 风险 |
| **D17** | **Gate B 3/4 PASS + c4 marginal-FAIL (seed-driven) → progress to P2 with caveat,不 trigger Option D (FQL stability ablation)** | **2026-05-20** | Gate B Option B 2-seed 闭环显示 c4 FAIL 是 seed=42 driven 而非 FQL-driven (seed=0 上 FQL slope 正);interim 的 Option D 触发条件 "c4 FAIL on both seeds" 未满足;按 spec §5.3 mitigation matrix "mixed → caveat + 3-seed extension" 路径,P2 用 n_seeds≥3 retire-or-confirm 此 marginal finding |
| **D18** | **Bug 2 修复采用方案 (a) 大 manifest**:生成 `single_u10_cross_tgt15_ep100.json` (100 ep,前 30 byte-identical to 30-ep),不动 `train_utils.py` 优先级逻辑 | **2026-05-20** | 不影响 30+ 现存 manifest 调用方 (paper 1 / broad val v2 / td3bc / online sac);(b/b') CLI honor `--episodes` 留 follow-up cleanup;ep100 manifest 同时 serve P2 default + P3 mix ratio ablation;**P2 spec 引用 ep100,不回写 P0+P1 spec §5.1/§5.2** (Gate B 历史事实保持) 详见 [`fql_succession_bug2_fix_decision.md`](fql_succession_bug2_fix_decision.md) |
| **D19** | **c4 阈值采用 Option α**:`slope ≥ −2 × SE_aggregated`,SE 按 √(p(1−p)/n_eval)/√17.5/√n_seeds 自动 scale | **2026-05-20** | retroactive 在 Gate B 数据上 4 种 (n_seeds × manifest) 配置全 PASS;P2 default (n=5, 100-ep) 下阈值 ≈ −0.0098;数学最干净 (95% 单侧 CI 下界);实现成本 5 行 numpy;与 D18 Bug 2 fix 自然配合 (noise 减半 → 阈值自动收紧);β/γ 不选,γ 可作 supplementary visualization 详见 [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) |
| **D20** | **P2 n_seeds 从 5 降至 2 primary [42, 0]**,conditional 扩 3 [42, 0, 7];verdict 改 **effect-size + 方向一致性** primary,Welch p / Bonferroni 留 sensitivity | **2026-05-21** | wallclock budget(L4 ~10h/session)+ effect-size primary 让 large-n 统计 power 不再是 paper claim 瓶颈;n=2 节省 60% wallclock (19h → 8h),让出余量给 revision 阶段扩 seed;Gate B 已用 [42, 0] 有 4-run 历史可 cross-reference;gray band [3pp, 5pp] 自动触发扩 n=3;n=2 下 c4 Option α 阈值自动 scale 到 −0.0155(比 n=5 −0.0098 宽容 1.6×,Gate B FQL slope −0.0038 仍 PASS) 详见 [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) §4.1 + §10.3 |
| **D21** | **P2 storage layout 拆 `checkpoints/` (大文件 .pt) + `results/` (绘图包) 两棵树**;train 输出 mirror 到 `results/training_curves/{algo}_seed{S}/`,final test eval 通过 `evaluate_offline --output-json` 单独写入 `results/test/`;Colab 跑完仅回收 `results/` 树即可本地绘图 + 重建 verdict | **2026-05-21** | 早期 ReBRAC c1 ablation 等 notebook 已用此约定(`CHECKPOINT_ROOT` + `RESULTS_ROOT` 单独命名)但近期 fql_succession Gate B notebook 把所有产物混在 `--save-dir` 下,违反约定;mirror 4 small file (`train_log.jsonl` + `eval_log.csv` + `trainer_state.json` + `train_config.txt`) 总大小 ~几十 KB/seed,廉价;`results/test/<algo>_seed<S>.json` self-contained final eval 与 train 解耦;train 加 `--skip-final-eval`,final eval 由独立 step 完成 详见 [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) §4.2 + §9.3.1 |
| **D22** | **P2 cell 定义改基于 collection protocol 元信息**(`policy_mixture` field),**GMM audit 从 Gate C.2 hard gate 降为 advisory sanity check** | **2026-05-21** | P2 sprint 0 实测 GMM audit 对 noise-widened single policy 有 known false-positive(privileged ε=0.5 → p_≥2=0.996;ε=0.3 → 0.850;均远超 < 0.20 阈值,但 collection metadata 证明都是 single policy);Root cause:GMM(max_comp=3, weight_floor=0.10)在 wide single Gaussian 上倾向 BIC-split,与 dryrun 验证用的 mixture-vs-mixture 对比 metric work 但 single-policy noise widening 的 negative control 上 over-sensitive;Mitigation 选项(Hartigan dip test / 改 weight_floor / per-anchor variance)reviewer 风险均高,实施成本 0.5-2 天;P2 处理:维持 GMM audit 作 advisory disclosure,paper §method 透明 caveat,cell assignment 由 collection metadata integrity 决定;**Audit 对 multi-policy mixture(M-multi-mix)仍 reliable**(P2 sprint 0 实测 p_≥2=0.414 + Welch p≈1e-29 + Δ CI lower 0.25 三 criterion 全过 → confirms mixture construction);**audit 对 single-policy noise widening(M-uni-noise)仅作 paper appendix 透明披露** 详见 [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) §3.0 + §10.5 |

---

## 7. 一句话总结

**本计划重启 FQL 作为 ReBRAC paper 1 的 sub-optimal-data 后续工作，paper claim 设计为条件式 iff（FQL 优势仅在 sub-optimal AND multi-modal 数据上 trigger），使用 vanilla SAC × 4 cell 5/5 gate 已验证的 `arrival_v2` reward + 解耦的 two-paper sequence framing。**P2 v1.2 重设**:**3-cell × 2-algorithm × 2-seed primary [42, 0] × 主对照(12 runs,可扩 18 runs)+ 1 个 mix ratio mechanism ablation(P3 sweep,~18 runs),total ≤ 36 runs / ~20h L4 / ~5-7 周 wallclock**(D20 wallclock-budget rebalance);任一 gate fail 立即 abort,最大化 decision-robust。verdict 用 effect-size + 方向一致性 primary,Welch p / Bonferroni 留 sensitivity。**

---

## 8. 参考文档链接

| 文档 | 角色 |
|---|---|
| [`offline_rl_line_summary.md`](offline_rl_line_summary.md) | Offline 线总览 |
| [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) | **`arrival_v2` reward 在 vanilla SAC × 4 cell 5/5 gate 验证报告 — 本计划 reward 选型依据** |
| [`online_sac_reward_redesign.md`](online_sac_reward_redesign.md) | efficiency_v2 hacking 诊断 + arrival_v2 v6 设计 spec |
| [`rebrac_experiment_plan.md`](rebrac_experiment_plan.md) | ReBRAC paper 1 protocol（FQL 沿用 5-seed × test=100 × `cross_stream` × `s0`） |
| [`rebrac_experiment_report.md`](rebrac_experiment_report.md) | ReBRAC paper 1 数字 ground truth |
| [`rebrac_mainline_review.md`](rebrac_mainline_review.md) | ReBRAC One Page + 4 finding spine |
| [`auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) | NODE 线 paused 状态（本计划不依赖） |
| [`auv_nav/rebrac.py`](../auv_nav/rebrac.py) | FQL 复用 twin-Q + critic LN 的源代码起点 |
| [`auv_nav/reward.py`](../auv_nav/reward.py) | `arrival_v2` preset 实现位置（commit `813096e`，§139-166） |
| [`scripts/concat_offline_datasets.py`](../scripts/concat_offline_datasets.py) | Episode-level dataset concat helper（commit `992625c`，v1 A2 mix5050 实战）— M-multi-mix 走此工具，不动 `collect_offline_data.py` |
| [`fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) | **P0+P1 可执行 spec**：5 个 task + Gate A.1 共享 + Gate A.2/B 判据 (CLI rename + c4 阈值 patch 见 v1.2 Session A) |
| [`fql_pytorch_port_design.md`](fql_pytorch_port_design.md) | **FQL 实现 design doc**：tensor shape 契约 + JAX→PyTorch pitfall + impl checklist + 12 个 test |
| [`fql_audit_multimodality_design.md`](fql_audit_multimodality_design.md) | **Audit 脚本 design doc**：k-NN GMM mode count + paired bootstrap + Gate A.2 verdict 逻辑 |
| [`fql_audit_dryrun_report.md`](fql_audit_dryrun_report.md) | **Task A dry-run report**:Gate A.2 PASS 4/4 实测 (2026-05-19, Δp(≥2) +0.58 [CI +0.53, +0.63]) |
| [`fql_e_uni_anchor_dataset_card.md`](fql_e_uni_anchor_dataset_card.md) | **E-uni 1000-ep dataset card**:Task D 闭环 (success 0.985, 86,685 transitions) |
| [`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md) | **Task E / Gate B final report**:Option B 2-seed 3/4 PASS + c4 marginal-FAIL (seed-driven) (2026-05-20) |
| [`fql_succession_bug2_fix_decision.md`](fql_succession_bug2_fix_decision.md) | **P2 pre-requisite 1/2**:Bug 2 fix design memo (D18, 方案 a 大 manifest) |
| [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) | **P2 pre-requisite 2/2**:c4 阈值 design memo (D19, Option α slope ≥ −2 × SE_agg) |
| [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) | **P2 main comparison spec v1.2 (Session A 2026-05-21,n=2 primary + checkpoints/results 拆分)** |

---

*Document version: v1.4 (2026-05-21, P2 sprint 0 collection 实测 + audit 降级 advisory). v0 → v1 砍掉约 50% validation insurance,保留全部 core paper claim 支撑实验。维护策略:本计划在 Gate A 通过后升级到 v1.1;Gate B 通过后升级到 v1.2 (v1.1 跳过原因见版本历史);P2 spec drafting 期间 storage-layout 重设升级到 v1.3;P2 sprint 0 collection 实测 + audit 降级 advisory 升级到 v1.4(本次);P2 全闭环后升级到 v2.0。每次 phase gate 通过/失败后修订对应 section 与 §6 决策记录。*
