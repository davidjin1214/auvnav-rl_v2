# Offline RL 线总结报告

> 文档版本：2026-05-07
> 作用：Offline RL 线的**总览入口**——给不熟悉本仓库的新读者一份时间轴 + 当前位置 + 下一步走向，**不重复任何 ground-truth 数字**（数字一律链接到源文档）。
> 镜像姊妹文档：[`docs/online_rl_line_summary.md`](online_rl_line_summary.md)。

---

## 目录

1. [时间轴一览](#1-时间轴一览)
2. [可用于论文写作的正式结果（链接，不复制数字）](#2-可用于论文写作的正式结果链接不复制数字)
3. [三阶段进展详解](#3-三阶段进展详解)
4. [当前活跃 backlog 与下一步](#4-当前活跃-backlog-与下一步)
5. [仓库 offline 线文件索引](#5-仓库-offline-线文件索引)
6. [给新读者的最小阅读路径](#6-给新读者的最小阅读路径)
7. [一句话总结](#7-一句话总结)

---

## 1. 时间轴一览

| 阶段 | 时间窗口 | 算法 | 状态 | 头部入口 |
|---|---|---|---|---|
| **Phase 1 — TD3+BC baseline 收口** | 2026-Q1 → ~2026-04 | TD3+BC（D4RL 经典 single-BC 作算法基线） | ✅ 已收口（不再扩展） | [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) |
| **Phase 2 — ReBRAC 主线** | ~2026-04 → 2026-05-07 | ReBRAC（dual-BC + critic-side BC penalty） | ✅ 主线 paper-ready 4/4 closed (rev.8) | [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) |
| **Phase 2.5 v1 — Broad validation (efficiency_v2)** | 2026-05-04 → 2026-05-07 | ReBRAC 跨三轴 generality 广验 + C1 deep-dive | ⚠ **SUPERSEDED 2026-05-18** by v2 plan；v1 archive 保留，不重跑、不进 paper | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md)（archive） |
| **Phase 2.5 v2 — Broad validation (arrival_v2 cross-only)** | 2026-05-18 → 2026-05-19 | cross-only spotlight + 2 flow regime + 精简 collector + conditional BC sweep | ✅ **2-seed 4-run 闭环 PASS 2026-05-19**（N0 HOLDS / N2' STRONG_NEGATIVE；M1 BC sweep 未触发） | [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) ★ |
| **Phase 3 — AUVHamNODE Offline RL(cross-domain transfer)** | 2026-05-06 → 2026-05-13 | frozen AUVHamNODE 1-step prior + ReBRAC augmentation | ⏸ **PAUSED 2026-05-13**(原状态 v2.1 locked → α 路径 Step 0-4 审计 → 4 项硬接口差异 + wake current 2-4× OOD 暴露;用户决定暂停) | [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) ★ |
| **FQL Succession — FQL vs ReBRAC（Paper 2 候选）** | 2026-05-19 → 2026-05-23 | FQL（flow-matching teacher + 1-step distill）vs ReBRAC（dual-BC） | ✅ **NEGATIVE 闭环 2026-05-23**（核心 conditional-iff 假设证伪 → 机制发现 + 诚实负面，B+A） | [`docs/fql_succession_p2_results.md`](fql_succession_p2_results.md) ★ |

**当前位置**：Phase 2 主线已 freeze；**Phase 2.5 v1 广验已 SUPERSEDED 2026-05-18**（reward 失配 + online §7.6 更强 finding）；**Phase 2.5 v2 已于 2026-05-19 PASS**（N0 HOLDS / N2' STRONG_NEGATIVE，4-run 2-seed 闭环；详见 [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)）；**Phase 3 AUVHamNODE Offline RL 已 PAUSED**(2026-05-13;不影响其他线;恢复条件见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) §6)；**FQL Succession P2 已于 2026-05-23 NEGATIVE 闭环**（"FQL > ReBRAC iff sub-optimal AND multi-modal" 证伪；机制 = BC-anchor 目标质量决定噪声鲁棒性；详见 [`fql_succession_p2_results.md`](fql_succession_p2_results.md)）。

---

## 2. 可用于论文写作的正式结果（链接，不复制数字）

### 2.0 引用规则

| 结果 | 出处（数字 ground truth） | 引用规则 |
|---|---|---|
| ✅ **ReBRAC paper-level Finding (i)–(iv)** | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §1–§10（rev.8） | 直接引用；mainline_review §0/§1 是论文化摘要 |
| ✅ **TD3+BC baseline closure** | [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) | 直接引用作 algorithm baseline |
| ✅ **Stage F (B) — critic LayerNorm 必要性** | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §7.15 | 论文 ablation 节 |
| ✅ **Stage F (C) — Phase 1 deployable vs TD3BC priv-critic 统计检验** | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §7.16 | 论文核心论点：deployable 不输 priv |
| ⚠ **Broad validation B1 sensor envelope (s1 ≈ s0)** | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3.4 | clean positive，可作 paper-quality 引用 |
| ⚠ **Broad validation 其他 spoke (A1/A2/A3/B2/C1/C3)** | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3 + [`c1_s1_followup`](rebrac_c1_s1_followup_report.md) | **standalone exploratory；不达 paper-quality** — 需要 BC penalty sweep / mix ratio sweep / paired bootstrap 闭环 |

### 2.1 Why broad validation 是 standalone 不回写主线

2026-05-07 用户判断 + 文档级落地（见 [`rebrac_experiment_report.md`](rebrac_experiment_report.md) §10A pointer + [`broad_validation_report`](rebrac_broad_validation_report.md) 头部 status note + [`c1_s1_followup`](rebrac_c1_s1_followup_report.md) 头部 status note）：

- 8 个 spoke 中只有 B1 是 clean positive；A1 underpowered (paired t p≈0.11)、A2 mid-gap collapse 的 mode-collapse 机制是 hypothesis、A3 5-seed 反转 1-seed 决策、B2/C3 std blow-up 无 mechanism ablation、C1 task-fundamental floor 的 BC penalty 强度 sweep 未做
- 把还在动的 commentary 钉死进静态 finding spine 会把后续 sweep 的 churn cost 转嫁给主线
- **Retrofit trigger condition**：BC penalty sweep on C1 / A1 paired bootstrap / mix ratio sweep / target_speed=2.0 P1 闭环之后再考虑统一回写

---

## 3. 三阶段进展详解

### 3.1 Phase 1 — TD3+BC baseline 收口

**角色**：D4RL 经典 single-BC 算法作 ReBRAC 的对照 baseline；不是论文核心算法。

**收口产出**：
- [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) — Phase 0c 总收口（含 stage_a/b/c + bc_final + noisy_support + worldcomp_teacher_gap）
- [`docs/td3bc_phase0c_experiment_design.md`](td3bc_phase0c_experiment_design.md) — 设计协议
- [`docs/td3bc_phase0b_v2_experiment_report.md`](td3bc_phase0b_v2_experiment_report.md) — 中间收口
- [`docs/td3bc_worldcomp_teacher_gap_experiment_report.md`](td3bc_worldcomp_teacher_gap_experiment_report.md) — teacher gap 子研究
- [`docs/td3bc_mainline_closure_plan.md`](td3bc_mainline_closure_plan.md) — 收口 plan

**当前角色**：
- TD3+BC 不再扩展；上面 5 份已是 final 状态
- ReBRAC paper revision 阶段仍引用 TD3+BC（作为 single-BC baseline 对照）

### 3.2 Phase 2 — ReBRAC 主线（核心算法 + paper-readiness）

**核心 plan / report / review 三层文档**（详见 [`project_offline_rl_docs_audit memory`](../../../.claude/projects/-Users-xiangjin-Library-CloudStorage-OneDrive-Personal----Code-new-off-rl-rl-v2/memory/project_offline_rl_docs_audit.md)）：

| 文档 | 角色 | 持数字 ground truth？ |
|---|---|---|
| [`docs/rebrac_experiment_plan.md`](rebrac_experiment_plan.md) (rev.8) | **计划**：stage 设计、协议定义、判据预登记 | ❌ |
| [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) (rev.8) | **实验数据库**：每 stage 完整 per-seed/per-config 数据 | ✅ 唯一 ground truth |
| [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) (rev.3) | **战略 review + 论文化输入**：One Page 摘要 + 4 finding spine + 审稿人弱点 | ❌（数字仅引用 report） |

**配套文档**：
- [`docs/rebrac_method_section_draft.md`](rebrac_method_section_draft.md) — 论文 Method 节草稿
- [`docs/rebrac_paper_writing_index.md`](rebrac_paper_writing_index.md) — 写论文期"读哪份、抄哪段"地图
- [`docs/rebrac_statistical_test_followup.md`](rebrac_statistical_test_followup.md) — Welch's p / bootstrap CI

**主线 4 paper-level findings**（详见 mainline_review §1 + experiment_report §10）：
- Finding (i) `crosscomp` 主 cell 上 ReBRAC vs TD3+BC 显著优势（dual penalty 必要性）
- Finding (ii) `worldcomp` 上 priv ≈ deployable（deployable 协议立得住）
- Finding (iii) Dual penalty cross-dataset 必要（修正自 §7.13–§7.14 critic-penalty-off probe）
- Finding (iv) Critic LN 与 dual penalty 是两个独立 necessary component（§7.15 LN-off probe）

**关键 stage**：
- Stage A–C 主线 5-seed 收口 → finalist `(β1=4.0, β2=2.0)`
- Stage D Phase 1/Phase 2（worldcomp deployable vs priv-critic）
- Stage E (a) `crosscomp-1000` cross-dataset 二次验证
- Stage F (B) critic LayerNorm-off probe + Stage F (C) 统计检验

### 3.3 Phase 2.5 — Broad validation（v1 SUPERSEDED → v2 ✅ PASS 2026-05-19）

**2026-05-19 status 升级**：v2 broad validation 4-run 闭环完成。详见 [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)。

**v2 核心结果（2 seed [42, 0], cross_stream / s0 / arrival_v2）**：
| Cell | Setup | success | verdict (plan §5) |
|---|---|---:|---|
| **N0** | crosscomp / s0 / cross_u10 / Re150 (sub-critical) | **0.850 ± 0.024** (per-seed [0.867, 0.833]) | **HOLDS** — Δ vs efficiency_v2 main-line anchor 0.902 = **−5.20pp** (paired same-direction) |
| **N2'** | privileged / s0 / cross_u15 / Re250 (critical) | **0.000 ± 0.000** (per-seed [0.000, 0.000]) | **STRONG_NEGATIVE** — recovery_of_oracle = 0%, lift_vs_online_floor = **−10pp** |
| M1 BC sweep | conditional | NOT triggered | N2' ∉ [0.15, 0.40] partial zone |

**核心 paper finding**：actor-fundamental partial-observability ceiling under deployable `s0` sensor in critical regime — 即便 oracle teacher (privileged 70% direct success) 提供 demonstrations，s0-conditioned BC 不能传递 hull-integral flow 知识；N2' 甚至 跌破 online §7.6 catastrophic floor 10pp（report §4 列三个 candidate mechanism: OOD shift / BC trap / online 10% exploration luck）。

**2026-05-18 status (v1 SUPERSEDE 历史)**：v1 全套（design spec / implementation plan / report rev.2 / c1_s1_followup）已 SUPERSEDED by [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md)。取代原因：
1. v1 跑在 `efficiency_v2` 下，但 [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §3 证 `efficiency_v2` 在 upstream / 高难度 cell 有 OOB-suicide failure mode，v1 C1 task-fundamental floor claim 受 reward bias 污染
2. online 线 §7.6 已独立产出 `cross_u15 + s0` catastrophic FAIL（s0–s1 gap 80pp，A0 的 24× 放大）— 比 v1 B1 (Δ=−0.2pp clean positive) 强得多的 sensor envelope finding
3. v1 8 spoke 中只有 B1 是 clean positive，其余 7 spoke retrofit trigger 未闭环

**v2 plan 关键变化**（详见 v2 plan §3 diff 表）：
- 收敛到 cross-stream geometry（砍 upstream / tandem / sbs）
- 增加 (U, Re, λ) 两 regime：sub-critical (u10/Re150/λ=0.67) + critical (u15/Re250/λ=1.0)
- 精简 collector 到 crosscomp + privileged（砍 goalseek / worldcomp / mix5050）
- C1 5 ablation 串联 → conditional BC penalty sweep on stuck cell
- paper role 明确为 reward-bridge appendix + deployability boundary map（不是 standalone TBD）

**v1 archive 现状（保留不动）**：

**目标**：在三轴 (data quality / sensor / task geometry) 上各拉 2–3 条 spoke 测试 ReBRAC main finding 的 generality 边界。

**完成状态（截至 2026-05-07）**：
- T1–T6 基础设施 + S1 sanity ✓
- S2 P1 8 spoke × 2-seed probe ✓
- S3.b/c/d 5 触发 spoke 1-seed β refit + 5-seed parity ✓
- C1 deep-dive 5 ablation（reward swap / asym critic / epoch ×4 / sensor s0→s1 / convergence check）✓

**结果摘要**（数字详见 [`broad_validation_report.md`](rebrac_broad_validation_report.md) §3 + [`c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) §4）：

| Spoke | Status | Paper-quality? |
|---|---|---|
| **B1** sensor s1 (DVL+ADCP-short) | ✅ untriggered, robust | ✅ clean positive 可引用 |
| A1 goalseek collector | ⚠ refit_b directional consistent 但 paired t p≈0.11 | ❌ underpowered |
| A2 mix5050 collector | ⚠ ReBRAC vs TD3+BC = +0.2pp（vs 主线 +23pp）；mode collapse 是 hypothesis | ❌ 机制未直接验证 |
| A3 privileged collector | ⚠ 5-seed 反转 1-seed 决策 | ❌ refit decision fragile |
| B2 sensor s2 (DVL+ADCP-long) | ⚠ mean-on-anchor 但 std_blow_up | ❌ mechanism 未 ablation |
| **C1** upstream geometry | ⚠ 5 个 ablation 全钉在 0.195–0.225 → task-fundamental floor candidate | ❌ BC penalty sweep 未做 |
| C3 tandem geometry | ⚠ mean shift + std blow-up | ❌ mechanism 未 ablation |

**关键决策（2026-05-07）**：广验 + c1_s1 保持 standalone，**不回写**主报告 / mainline_review。理由 + retrofit trigger condition 见 §2.1 与 [`broad_validation_report`](rebrac_broad_validation_report.md) 头部 status note。

### 3.4 Phase 3 — AUVHamNODE Offline RL（⏸ PAUSED 2026-05-13）

**Anchor 入口**:[`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) ★(pause 详情 + 累计决策 + 恢复条件 + 文件清单)

**简要时间轴**:
- 2026-05-06: v1.0 plan(批判性合并 5 份 `docs/offline_mbrl_plan/` 草案)
- 2026-05-08: v2.0 locked(升级到 cross-domain transfer framing,Plan B 算法栈撤销)
- 2026-05-09 ~ 05-10: v2.1 + amendment(RL 专家 meta-review;§11.1.0 零号前置增补)
- 2026-05-12: 战略转折——ReBRAC mainline + paper + broad_validation 暂停;主线试图转 MBRL+arrival_v2;v3 pre-notes 落地(α 路径)
- 2026-05-13: Step 0-4 廉价审计完成(详见 [`experiments/auvhamnode_spike/`](../experiments/auvhamnode_spike/));**4 项硬接口差异 + wake current 2-4× OOD 暴露**;用户决定暂停整条线

**累计决策**(pause memo §4):
- ✅ `mytorch1` env 可驱动 AUVHamNODE checkpoint
- ✅ 6/7 个条件轴对齐;唯一不对齐的是海流 v_c_n 幅度(2-4× 训练分布外)
- ✅ `vehicle.py` 不要换 `remus100_core.py`(swap 修 30% 问题,余 70% 无解)
- ✅ arrival_v2 reward 在 `auv_nav/reward.py` 已 in-tree,与本线解耦
- ⏸ Path 1B spike-lite(~5h kill-test)未执行

**未做**(pause memo §5):未跑 spike;3/6 wake 文件统计缺失;Path 2 finetune 可行性未确认;Path 4 vehicle.py oracle 未细化;AUVHamNODE 训练 dataset 未取回。

**恢复条件**(pause memo §6):AUVHamNODE 上游 pipeline 可访问 OR wake-compatible 数据生成 OR 项目转低速任务 OR 用户主动恢复。

---

### 3.5 FQL Succession — FQL vs ReBRAC（Paper 2 候选，✅ NEGATIVE 闭环 2026-05-23）

**Anchor 入口**：[`docs/fql_succession_p2_results.md`](fql_succession_p2_results.md) ★（主报告：机制 + 诚实负面）+ [`docs/fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9（权威 lab 记录）。

**问题**：表达力更强的 flow-matching 先验（FQL）是否在 sub-optimal AND multi-modal 数据上系统性超过 ReBRAC（dual-BC）？

**核心结论（假设证伪 → B+A：机制发现 + 诚实负面，NOT "FQL wins"）**：
- **2×2 矩阵**（modality × noise）：discriminator 是**噪声**而非 modality —— clean-multi cell（E-multi）NULL，modality 被排除。
- **机制三连**：Q1（critic 侧排除）→ Q1b（actor `--actor-penalty-coef` β1 4.0→1.0，+23.5pp 反超 FQL）→ Q1c（β1=1.0 在 clean 也更好）⇒ **单一固定 ReBRAC β1=1.0 在 clean+noisy 双轴 dominate FQL，worst-case-over-noise 0.910 > FQL 0.858**。原先看到的 "FQL 赢" 是 ReBRAC β1 mis-tuning 的 artifact。
- **C-1 公平复赛**（FQL 自己的 `--distill-alpha-bc` log-spaced 扫描）**RESCUE-FAIL**（clean 最高 0.858，差杆 −5.2pp；最优锚强度方向与 ReBRAC 相反）。
- **机制（可发表的 finding）**：offline-RL 对 action noise 的鲁棒性由 **BC anchor 目标质量**决定；最优锚强度随目标噪声翻转（ReBRAC 锚到 raw action，噪声下须减弱 β1；FQL 锚到 flow-denoised target，已干净但 clean 无额外 edge）。
- **统计**：固定 100-scenario manifest（`single_u10_cross_tgt15_ep100`）⇒ 配对比较，σ_train ≈ 3.8pp，两大效应 n=2 即显著；头对头 NULL。

**交付物**（commits `b1f1854` → `498d421` → `8bb9fab`）：
- [`fql_succession_p2_results.md`](fql_succession_p2_results.md) — 主报告
- [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9 — 权威 lab 记录
- [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) v1.4 — CLOSED 历史设计记录（假设性章节已逐节标 SUPERSEDED/RESOLVED）
- [`notebooks/fql_succession_p2_verdict.ipynb`](../notebooks/fql_succession_p2_verdict.ipynb)（纯分析，读 `results/` 现算所有表/图，已实跑）+ `docs/assets/fql_succession_p2/verdict_*.png`

**跨-benchmark 泛化探测（2026-05-23，FLOOR-closed）**：在更难的 `single_u15_cross`（U=1.5/Re250）跑了精简探测（地板门先行）。**实测 FLOOR** —— clean privileged collector 0.719、noisy(σ=0.5) 0.098、offline ReBRAC β1=1.0 gate 0.14（out_of_bounds 主导，in-training 仍缓升）→ **s0 observability floor，比较在此 regime 未定义**（sensor-sufficiency 边界，非算法反例）。写入 results §6.5，不弱化 u10_cross 上的 Results 1–3。

**P4 收尾（2026-05-24）**：实验线全闭环、无 pending；写作索引 [`fql_succession_paper_writing_index.md`](fql_succession_paper_writing_index.md) 已建（paper 章节→文档映射 + 复现 notebook 列表 + headline 数字 + 定位选项 + 投稿前 TODO）。剩余纯写作（Method prose / Related Work / Abstract / 定位决策）。

**对 paper 1（ReBRAC 主线）的 cross-talk caveat（2026-05-24 落地）**：FQL succession P2 Q1b/Q1c mechanism 反向暴露 paper 1 finalist β1=4.0 的 scope —— 在 (privileged × `arrival_v2` × σ=0.5 noisy) 下 **β1=1.0 反而 +23.5pp 打 β1=4.0（n=2 SIG）**。**两边不矛盾**（reward / collector / 条件动作方差 / seed pool 四轴同时变；paper 1 选 β1=4.0 是 seed 44 std-driven），但 paper 1 revision 期需准备 implementation note + 可选 5-seed 防御实验。完整论证落在 [`rebrac_mainline_review.md`](rebrac_mainline_review.md) §2.2.6（paper 1 侧）+ [`fql_succession_p2_results.md`](fql_succession_p2_results.md) §8.1（paper 2 侧）。**paper 1 finding (i)–(iv) 不受影响。**

---

## 4. 当前活跃 backlog 与下一步

### 4.1 ReBRAC paper revision (active)

paper drafting Phase 5 → revision 阶段，主要锚点：
- [`docs/rebrac_method_section_draft.md`](rebrac_method_section_draft.md) Method 节
- [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §0/§1.5 是写作期反复回看的 One Page

### 4.2 Broad validation v2（✅ PASS 2026-05-19）

详见 [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)（report） + [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md)（plan rev.3）。

**实际执行**（vs plan §6 原 4-stage 计划）：
- Stage 0A (S sanity): ✅ 完成 2026-05-18，crosscomp 0% / privileged 70% — 触发 rev.3 pivot（N2 → N2'）
- Stage 0B (dataset 收集): ✅ N0 + N2' 各 1000 ep 新 collect 完成
- Stage 1 (core training + eval): ✅ **2 seed [42, 0]**（plan 原 3 seed，缩水原因见 report §6.1）；N0 0.850 → HOLDS，N2' 0.000 → STRONG_NEGATIVE
- Stage 2 (conditional M1): ⏭ NOT triggered（N2' ∉ [0.15, 0.40]）

**剩余 follow-up**（不在主线 paper revision 关键路径上，详见 report §6.3）：
- 5-seed 补全 N0（+43, 44, 45）— Medium priority，若审稿人 push back
- Asym-critic ablation on N2'（plan §9 backlog）— **High priority** 若 paper §discussion 想区分 "actor-fundamental" vs "critic-fundamental" partial-obs ceiling
- online §7.6 candidate-C 验证（3 success episode trajectory 分析）— Medium，若 paper §discussion 想 strengthen anomaly section

**v1 retrofit trigger condition 已作废**（C1 BC sweep / A1 paired bootstrap / mix ratio / target=2.0 P1 等）— v1 finding 不进 paper。

### 4.3 Online 线交付的 SAC collector（active — rev.3 **Plan A 4 tier dataset COMPLETED** 2026-05-25）

**Spec**：[`docs/arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md) **rev.3** §4.0（rev.2 §4.1 路径 1 cross_u15 作废）。与本线协议严格对齐 = **s0 + k=4 + arrival_v2**。

**rev.3 单一路径 = Plan A**：从 `checkpoints/arrival_v2_prototype/cross_u10_regression/arrival_v2/sac_vanilla/s0_k4/seed_46/` 的训练过程切片（39 个 `agent_step_*.pt`，25k env_step cadence）里 audit pick 4 个 ckpt 构成真 D4RL 4-tier — 与 D4RL 经典 paper (CQL/IQL/FQL/ReBRAC) 一致的 tier 构造方式：

| tier | step | success | ckpt 文件名 |
|---|---:|---:|---|
| random | 25_002 | 0% | `agent_step_00025002.pt` |
| medium | 425_004 | 50% | `agent_step_00425004.pt` |
| medium_expert | 575_004 | 80% | `agent_step_00575004.pt` |
| expert | 600_000 | 100% | `agent_step_00600000.pt` |

**Audit verdict**: ✅ GO — 4 tiers cleanly separated, medium tier |Δ|=0% from D4RL target。schema 跨全 39 ckpt 一致 (obs_dim=48, priv=0)。

**两个新 finding 候选**（来自 dense sweep training curve）：
1. **Failure-mode tier drift** — random=100% OOB pure → medium=OOB/timeout mixed → expert=clean
2. **Cliff fine-tuning zone 575k→600k** — 16× learning rate vs preceding 425k→575k accelerated phase

**Adapter status**: ✅ landed（`auv_nav/sac_policy.py` + `scripts/collect_offline_data.py` 扩展，5 tests pass，commit 97d394c）。`SACCheckpointPolicy.from_checkpoint` + Layer-1/2 sanity check + once-per-worker load + metadata schema 扩展。

**Audit notebooks**: [`notebooks/sac_collector_d4rl_tier_audit.ipynb`](../notebooks/sac_collector_d4rl_tier_audit.ipynb) (self-discovering) + `_completed1.ipynb` (实验记录, commit 9f8252e)。

**Collection 决策（已落地 2026-05-25）**：mode = **A 全 stochastic**；`replay_latest.pkl` 第 5 档 = **B 4-tier 闭环后做**。

**Collection notebook**: [`notebooks/sac_collector_plan_a_collect_4tiers.ipynb`](../notebooks/sac_collector_plan_a_collect_4tiers.ipynb)（template, commit 7a82308）+ `_completed.ipynb`（实验记录, commit 032b527）。

**Actual collection results**（Colab L4 CPU pool，1000 ep × 4 tier × `--num-workers 8`，stochastic）：

| tier | step | success | mean_reward | n_trans | runtime |
|---|---:|---:|---:|---:|---:|
| random | 25_002 | 0.20% | −1.83 | 180_067 | 910 s |
| medium | 425_004 | 51.50% | +0.06 | 250_544 | 1245 s |
| mexp | 575_004 | 75.10% | +0.47 | 171_394 | 871 s |
| expert | 600_000 | 89.90% | +0.97 | 114_832 | 588 s |

→ 4 tier 清晰分离（success spread 15pp+，mean_reward 单调 -1.83→+0.97），medium tier 因 timeout 占比 20% avg_len 反转为最长（251 step > random 180 > mexp 171 > expert 115）— D4RL 范式 + failure-mode tier drift 双重证据。详 spec §4.0.7 / `_completed.ipynb`。

**下一步**（spec §4.0.5 表格末行）：FQL + ReBRAC β1∈{1.0, 4.0} head-to-head × 4 tier × 2 seed = 16 run（~4-6h L4）— 落出 paper 1 / FQL P2 SAC-collector 第四轴主结论。

**总预算 rev.3**：~5-7h L4 = **1 个 Colab 周**（adapter 2.25h + audit 0.5h + Plan A collection 1h 已完成；剩 head-to-head 4-6h）。

**关键约束**：`s0_k4 + arrival_v2 + cross_u15` 上不存在 expert SAC ckpt（rev.2 §4.5）— sensor floor 实证，与本线 v2 N2' / FQL §6.5 同物理。Plan A 选 cross_u10 而不是 cross_u15，正是基于此约束。

### 4.4 AUVHamNODE Offline RL(⏸ PAUSED 2026-05-13)

整条线已 paused,**不在当前 backlog 内**。Anchor 见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md);恢复条件见同文 §6;恢复时 first step 见 §7。

短期不需要的工作:Path 1B spike-lite 实施、Path 2 finetune 评估、Path 4 vehicle.py oracle plan 起草、AUVHamNODE 上游 dataset 取回。这些都在 pause memo §5 留作债务表。

---

## 5. 仓库 offline 线文件索引

按"想找 X 看 Y"的角度组织。所有路径相对仓库根。

### 5.1 顶层导航：当前应该读什么

| 想找... | 看这里 |
|---|---|
| Offline 线整体状态、时间轴、下一步 | **本文件**（[`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md)） |
| ReBRAC 这一条线串线总览（TD3+BC→ReBRAC→FQL，含 β1 跨线 reconciliation） | [`docs/rebrac_line_overview.md`](rebrac_line_overview.md) |
| ReBRAC One Page 摘要 + 4 finding spine | [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §0 + §1.5 |
| ReBRAC 任何具体数字 / per-seed | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §1–§10（按 stage 索引） |
| Broad validation 三轴 8 spoke 全表 | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3 |
| C1 spoke task-fundamental floor 候选证据链 | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3.5 + [`docs/rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) |
| TD3+BC baseline 详细收口 | [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) |
| AUVHamNODE Offline MBRL 线为何 paused / 累计决策 / 恢复条件 | [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) ★(pause anchor) |
| AUVHamNODE Offline MBRL 历史 plan(paused) | [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md)(v2.1, paused) + [`auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](auvhamnode_offline_mbrl_plan_v3_pre_notes.md) + [`experiments/auvhamnode_spike/`](../experiments/auvhamnode_spike/) |
| Env / sensor / reward / benchmark 规格 | [`docs/environment_design.md`](environment_design.md) |
| World model + offline RL 综述 | [`docs/world_model_and_offline_rl_survey.md`](world_model_and_offline_rl_survey.md) |
| RLPD 设计（offline-to-online，跨线复用） | [`docs/rlpd_design.md`](rlpd_design.md) |
| 论文写作期"读哪份、抄哪段" | [`docs/rebrac_paper_writing_index.md`](rebrac_paper_writing_index.md) |
| Paper 1 β1=4.0 vs paper 2 β1=1.0 为何不矛盾（cross-talk caveat）| [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §2.2.6（paper 1 侧）+ [`docs/fql_succession_p2_results.md`](fql_succession_p2_results.md) §8.1（paper 2 侧）|

### 5.2 文档（docs/）

#### A. ReBRAC 活跃主线（rev.8 收口）

| 文件 | 状态 | 角色 |
|---|---|---|
| [`rebrac_experiment_plan.md`](rebrac_experiment_plan.md) (rev.8) | active | 计划/protocol |
| [`rebrac_experiment_report.md`](rebrac_experiment_report.md) (rev.8) | active | 实验数据库（数字 ground truth；§10A 已退回为 broad_validation pointer） |
| [`rebrac_mainline_review.md`](rebrac_mainline_review.md) (rev.3) | active | 战略 review + paper One Page |
| [`rebrac_method_section_draft.md`](rebrac_method_section_draft.md) | active | 论文 Method 节草稿 |
| [`rebrac_paper_writing_index.md`](rebrac_paper_writing_index.md) | active | 写论文期 reading map |
| [`rebrac_statistical_test_followup.md`](rebrac_statistical_test_followup.md) | active | Welch's p / bootstrap CI 后续 |
| [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) ★ | **active (2026-05-19 PASS)** | v2 数字 ground truth — N0 HOLDS / N2' STRONG_NEGATIVE；paper §experiments appendix headline = actor-fundamental partial-observability ceiling under s0 |
| [`rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) | **closed plan (2026-05-19, rev.3)** | v2 cross-only spotlight under arrival_v2；5 cell core + 1 conditional sweep；rev.3 pivot N2→N2'；4-run 闭环已 PASS（详见 report）|
| [`rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) (rev.2) | **SUPERSEDED 2026-05-18 (archive)** | v1 三轴 8 spoke 数据 archive；不重跑、不进 paper |
| [`rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) | **SUPERSEDED 2026-05-18 (archive)** | v1 C1-s1 sensor upgrade follow-up archive |
| [`superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`](superpowers/specs/2026-05-04-rebrac-broad-validation-design.md) | **SUPERSEDED 2026-05-18 (archive)** | v1 design spec archive |
| [`superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`](superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md) | **SUPERSEDED 2026-05-18 (archive)** | v1 implementation plan archive |

#### B. TD3+BC 已归档（不再扩展）

| 文件 | 角色 |
|---|---|
| [`td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) ★ | TD3+BC 总收口 |
| [`td3bc_phase0c_experiment_design.md`](td3bc_phase0c_experiment_design.md) | 设计协议 |
| [`td3bc_phase0b_v2_experiment_report.md`](td3bc_phase0b_v2_experiment_report.md) | 中间收口 |
| [`td3bc_worldcomp_teacher_gap_experiment_report.md`](td3bc_worldcomp_teacher_gap_experiment_report.md) | teacher gap 子研究 |
| [`td3bc_mainline_closure_plan.md`](td3bc_mainline_closure_plan.md) | 收口 plan |

#### C. 老 offline RL 总框架（已被实际工作取代，自我声明已过期）

| 文件 | 状态 |
|---|---|
| [`offline_rl_implementation_plan.md`](offline_rl_implementation_plan.md) (rev.5) | 已被 ReBRAC 主线取代；保留作历史 |
| [`offline_rl_quick_validation.md`](offline_rl_quick_validation.md) (rev.5) | 同上 |

#### D. 综述与跨线设计（仍有效）

| 文件 | 角色 |
|---|---|
| [`world_model_and_offline_rl_survey.md`](world_model_and_offline_rl_survey.md) | Part 3 综述 |
| [`rlpd_design.md`](rlpd_design.md) | RLPD 设计（online + offline 跨线复用） |
| [`environment_design.md`](environment_design.md) | Env / sensor / reward / benchmark 规格（跨线共享） |

#### E. Offline RL 下一阶段(⏸ AUVHamNODE+MBRL 线已 paused 2026-05-13)

| 文件 | 状态 |
|---|---|
| [`auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) ★ | **pause anchor**(累计决策 + 恢复条件 + 未做的事 + 文件清单)。**任何 resume 工作必读** |
| [`auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md)(v2.1) | ⏸ paused;原 v1.0 → v2.0 → v2.1 三轮迭代 plan;顶部 banner 已加 |
| [`auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](auvhamnode_offline_mbrl_plan_v3_pre_notes.md) | ⏸ paused;α 路径交接备忘(4 项硬接口差异);顶部 banner 已加 |
| [`experiments/auvhamnode_spike/`](../experiments/auvhamnode_spike/)(Step 0-4 审计 + spike-lite 设计 + decision memo) | ⏸ Step 0-4 ✅;Path 1B spike 未执行 |
| [`offline_mbrl_plan/`](offline_mbrl_plan/) 下 6 份草案 | 已被 v2.0 批判性合并/选择性吸收;6 份均已加 deprecated banner |
| [`offline_rl_implementation_plan.md`](offline_rl_implementation_plan.md)(rev.5, 2026-04-22) | 已 deprecated(XQL/FQL 主线撤销);TD3+BC / ReBRAC 段落仍可作论文引用源 |

#### F. FQL Succession（Paper 2 候选，✅ NEGATIVE 闭环 2026-05-23）

> 注:此处 FQL 是作为 **Paper 2 的 head-to-head 对手**复活（FQL vs ReBRAC 机制研究），与上面 §C/§E "XQL/FQL 主线撤销" 是两回事——后者指 2026-Q1 老 offline 框架的算法栈撤销。

| 文件 | 状态 | 角色 |
|---|---|---|
| [`fql_succession_p2_results.md`](fql_succession_p2_results.md) ★ | **closed (2026-05-23)** | P2 主报告（机制 + 诚实负面） |
| [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9 | closed | 权威 lab 记录（Q1/Q1b/E-multi/Q1c/power/C-1）——数字 ground truth |
| [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md)(v1.4) | **CLOSED 历史设计记录** | 原 conditional-iff 假设性章节已逐节标 SUPERSEDED/RESOLVED |
| [`fql_succession_p2_collection_log.md`](fql_succession_p2_collection_log.md) | closed | dataset 收集 + audit 记录 |
| [`notebooks/fql_succession_p2_verdict.ipynb`](../notebooks/fql_succession_p2_verdict.ipynb) | 纯分析，已实跑 | 读 `results/` 现算所有表/图；builder `scripts/_build_fql_succession_p2_verdict_notebook.py` |
| [`fql_succession_p2_xbench_spec.md`](fql_succession_p2_xbench_spec.md) | **FLOOR-closed** | 跨-benchmark 泛化探测 spec + 实测 FLOOR（§6.5 依据） |
| [`fql_succession_paper_writing_index.md`](fql_succession_paper_writing_index.md) ★ | **P4 写作索引 (rev.1)** | paper 章节→文档映射 + 复现 notebook + headline 数字 + 定位选项 + 投稿前 TODO |

### 5.3 实验数据 / checkpoints / offline data

| 路径 | 角色 |
|---|---|
| `experiments/offline/rebrac/` | ReBRAC 主线 5-seed 实验数据 |
| `experiments/offline/rebrac/broad_validation/` | 广验 8 spoke + C1 deep-dive |
| `experiments/offline/rebrac/c1_s1_sensor_upgrade/` | C1-s1 follow-up 数据 |
| `experiments/offline/td3bc/` | TD3+BC Phase 0c 数据（已归档） |
| `offline_data/<collector>_<sensor>_<reward>_<flow>_ep<N>/` | 多 collector / 多 sensor offline 数据集 |
| `checkpoints/offline/rebrac/` + `checkpoints/offline/td3bc/` | actor + critic checkpoint |

### 5.4 代码组件（auv_nav/ + scripts/）

| 文件 / 类 | 角色 |
|---|---|
| [`auv_nav/rebrac.py`](../auv_nav/rebrac.py) | ReBRAC agent（dual-BC + critic penalty） |
| [`auv_nav/replay.py`](../auv_nav/replay.py) `TransitionReplay.from_npz()` | 加载 offline 数据集 |
| [`auv_nav/baselines.py`](../auv_nav/baselines.py) | 4 类 baseline collector：goalseek / crosscomp / worldcomp / privileged |
| [`auv_nav/offline_registry.py`](../auv_nav/offline_registry.py) | offline dataset 配置 registry |
| [`scripts/train_offline.py`](../scripts/train_offline.py) `--algo rebrac` | ReBRAC 训练 entry point（同脚本兼跑 td3bc / fql） |
| [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) | offline 数据收集 |
| [`scripts/run_offline_rebrac_broad.sh`](../scripts/run_offline_rebrac_broad.sh) | 广验 sweep launcher |

### 5.5 Notebooks

| Notebook | 状态 |
|---|---|
| [`notebooks/rebrac_broad_validation_completed.ipynb`](../notebooks/rebrac_broad_validation_completed.ipynb) | 广验全流程 archive |
| [`notebooks/rebrac_broad_validation_parity_completed.ipynb`](../notebooks/rebrac_broad_validation_parity_completed.ipynb) | 5-seed parity archive |
| [`notebooks/rebrac_c1_s1_sensor_upgrade_completed.ipynb`](../notebooks/rebrac_c1_s1_sensor_upgrade_completed.ipynb) | C1-s1 follow-up archive |
| [`notebooks/rebrac_c1_train_convergence_check_completed.ipynb`](../notebooks/rebrac_c1_train_convergence_check_completed.ipynb) | C1 convergence diagnostic |

---

## 6. 给新读者的最小阅读路径

| 时间预算 | 读什么 |
|---|---|
| **30 分钟** | [`rebrac_mainline_review.md`](rebrac_mainline_review.md) §0 + §1.5（One Page + finding spine） |
| **2 小时** | + [`rebrac_experiment_report.md`](rebrac_experiment_report.md) §1 + §10（主线总结） + [`rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §0 + §3（广验摘要 + 全表） |
| **1 天** | + [`rebrac_experiment_plan.md`](rebrac_experiment_plan.md) §1–§5（动机 / stage 设计） + [`rebrac_experiment_report.md`](rebrac_experiment_report.md) §7 选段（具体 stage per-seed） |
| **想看 baseline 故事** | + [`td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) §1 |
| **想看 paused 的 AUVHamNODE+MBRL 线** | [`auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md)(累计决策 + 恢复条件) |

---

## 7. 一句话总结

Offline RL 线在 2026-Q1 → 2026-05 累计完成 **TD3+BC baseline closure(Phase 0c 5 文档已归档)+ ReBRAC 主线 paper-ready 4/4 closed (rev.8) + v1 三轴 broad validation 8 spoke 5-seed parity**;主线 paper drafting 进入 revision 阶段;**v1 广验 + c1_s1 follow-up 已于 2026-05-18 SUPERSEDED**(reward 失配 + online §7.6 更强 finding)，v1 archive 保留不重跑;**v2 broad validation 已于 2026-05-19 PASS**(arrival_v2 cross-only 2-seed 4-run 闭环 — N0 HOLDS / N2' STRONG_NEGATIVE，paper §experiments appendix headline = actor-fundamental partial-observability ceiling under s0；详见 [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md));**原计划下一阶段 AUVHamNODE Offline RL 已于 2026-05-13 paused**(详见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md));**FQL Succession（FQL vs ReBRAC，Paper 2 候选）已于 2026-05-23 NEGATIVE 闭环**("FQL > ReBRAC iff sub-optimal AND multi-modal" 证伪 → 机制发现 + 诚实负面:noise 是 discriminator、ReBRAC β1=1.0 双轴 dominate、BC-anchor 目标质量决定鲁棒性;详见 [`fql_succession_p2_results.md`](fql_succession_p2_results.md))。
