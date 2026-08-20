# Offline RL 线总结报告

> ⚠ **写作出口 LOCKED 2026-06-02**：本线全部素材的写作出口 = **博士论文第 5 章**（spec [`../paper/thesis_chapter_outline.md`](../paper/thesis_chapter_outline.md) rev.4）；ReBRAC 主线 → §N.4 主干（[`../paper/archive/rebrac_standalone/main.pdf`](../paper/archive/rebrac_standalone/main.pdf) 31pp arXiv preprint draft commit `932aca1` 直接复用），FQL succession → §N.6 节，broad-val v2 + asym-critic ablation → §N.5 节 + §N.4 paper 1 §6.6/L12。**不另起任何 standalone paper**（paper 1 ReBRAC / paper 2 FQL standalone 均撤销）。本文中"paper 1 / paper 2"用法是历史 / 线索内部 shorthand，**实际写作去向是上述 §N.k 节**。
>
> 📍 **节号与版本对照（2026-07-28 全仓指针体检补注）**：本文头注写于 2026-06-02，其中 `§N.k` 是当时 8 节方案的记法、`rev.4` 是当时的 spec 版本。现行章结构为 **10 节**；spec 现行 rev **不在此写死**（写死正是本次体检查出的腐化源），以 [`CLAUDE.md`](../CLAUDE.md) 文档索引表为准。节号对照：**§N.2 → §5.5**（Online RL）、**§N.4 → §5.7**（ReBRAC-Q 主线）、**§N.5 → §5.8**（泛化边界）、**§N.6 → §5.9**（算法对比：FQL + SAC collector）。正文内 `§N.k` 一律照此读，**不逐处改写**。
>
> 🔍 **真值审计（2026-08-16）**：逐条核对本文断言 vs 源文档 / `results/` / `metadata.json` 实测。
> 数字基座核为真（A0 六个数字、SAC 4-tier 八个数字逐位吻合；19 个 commit hash 全在；36 个
> `test_result.json` 计数吻合）。**订正了六处**：v2 广验数字与「同向退化」读法停在首轮 2-seed（§1/§3.3/§4.2/§7）、
> §4.2 把两项已完成的 follow-up 挂在待办、§5.3 数据路径整张指向 `experiments/` 而实体在 `results/`、
> §4.5 漏了 2026-08-16 新增的第 ⑤ 条、§2.0 对 v1 B1 的 paper-quality 标注（用户 2026-08-16 裁定
> v1 一律不进论文）、§4.3 provenance 路径未标 Drive-only。
> **根因**：源文档后续做了 supplement，本文没跟——落点全在 2026-05 之后被追加过的条目。
> 下方「不重复任何 ground-truth 数字」的自我约束**事实上没做到**，重复的数字正是腐化处。
>
> 文档版本：2026-05-07（正文），审计订正 2026-08-16
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
| **Phase 2.5 v2 — Broad validation (arrival_v2 cross-only)** | 2026-05-18 → 2026-07-12 | cross-only spotlight + 2 flow regime + 精简 collector + conditional BC sweep | ✅ **3-seed 闭环 PASS**（首轮 2 seed 2026-05-19；asym addendum 2026-05-27；seed 43 supplement 2026-07-12 补齐第三种子）（N0 HOLDS / N2' STRONG_NEGATIVE；M1 BC sweep 未触发） | [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) ★ |
| **Phase 3 — AUVHamNODE Offline RL(cross-domain transfer)** | 2026-05-06 → 2026-05-13 | frozen AUVHamNODE 1-step prior + ReBRAC augmentation | ⏸ **PAUSED 2026-05-13**(原状态 v2.1 locked → α 路径 Step 0-4 审计 → 4 项硬接口差异 + wake current 2-4× OOD 暴露;用户决定暂停) | [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) ★ |
| **FQL Succession — FQL vs ReBRAC（Paper 2 候选）** | 2026-05-19 → 2026-05-23 | FQL（flow-matching teacher + 1-step distill）vs ReBRAC（dual-BC） | ✅ **NEGATIVE 闭环 2026-05-23**（核心 conditional-iff 假设证伪 → 机制发现 + 诚实负面，B+A） | [`docs/fql_succession_p2_results.md`](fql_succession_p2_results.md) ★ |

**当前位置**：Phase 2 主线已 freeze；**Phase 2.5 v1 广验已 SUPERSEDED 2026-05-18**（reward 失配 + online §7.6 更强 finding）；**Phase 2.5 v2 已 PASS**（N0 HOLDS / N2' STRONG_NEGATIVE；首轮 2-seed 4-run 2026-05-19，2026-07-12 seed 43 supplement 补齐为 3 seed {42, 0, 43}；详见 [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)）；**Phase 3 AUVHamNODE Offline RL 已 PAUSED**(2026-05-13;不影响其他线;恢复条件见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) §6)；**FQL Succession P2 已于 2026-05-23 NEGATIVE 闭环**（"FQL > ReBRAC iff sub-optimal AND multi-modal" 证伪；机制 = BC-anchor 目标质量决定噪声鲁棒性；详见 [`fql_succession_p2_results.md`](fql_succession_p2_results.md)）。

---

## 2. 可用于论文写作的正式结果（链接，不复制数字）

### 2.0 引用规则

| 结果 | 出处（数字 ground truth） | 引用规则 |
|---|---|---|
| ✅ **ReBRAC paper-level Finding (i)–(iv)** | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §1–§10（rev.8） | 直接引用；mainline_review §0/§1 是论文化摘要 |
| ✅ **TD3+BC baseline closure** | [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) | 直接引用作 algorithm baseline |
| ✅ **Stage F (B) — critic LayerNorm 必要性** | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §7.15 | 论文 ablation 节 |
| ✅ **Stage F (C) — Phase 1 deployable vs TD3BC priv-critic 统计检验** | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §7.16 | 论文核心论点：deployable 不输 priv |
| ❌ **v1 广验全部 8 spoke（含 B1 sensor envelope）** | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) + [`c1_s1_followup`](rebrac_c1_s1_followup_report.md) | **不进论文**（用户 2026-08-16 裁定，与 §4.2 一致）。v1 整条线作废：跑在失配的 `efficiency_v2` 下，且已被 v2 取代。**B1 曾被本表标为「clean positive 可作 paper-quality 引用」，该标注已撤销** |

### 2.1 Why broad validation 是 standalone 不回写主线（⚠ 已被 v1 作废覆盖，仅存历史决策）

> ⚠ **2026-08-16 补注**：本节写于 2026-05-07，讨论的是「v1 广验何时回写主线」。
> v1 已于 2026-05-18 被 v2 取代，其 retrofit trigger condition 亦已作废（见 §4.2 末），
> 用户 2026-08-16 裁定 **v1 finding（含 B1）一律不进论文**。故本节列出的触发条件**不再是待办**，
> 整节仅作历史决策记录读。

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

**核心 plan / report / review 三层文档**（分工另记在 Claude 侧 auto-memory `project_offline_rl_docs_audit`；原链接指向 Mac 本机 memory 目录的绝对路径，仓外且逐机不同，2026-07-28 体检时改为纯提及）：

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

**status 升级**：v2 broad validation 首轮 4-run 闭环 2026-05-19；asym-critic addendum 2026-05-27；seed 43 supplement 2026-07-12 补齐第三种子。详见 [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)。

**v2 核心结果（3 seed {42, 0, 43}, cross_stream / s0 / arrival_v2；2026-07-12 起以三种子为准）**：
| Cell | Setup | success | verdict (plan §5) |
|---|---|---:|---|
| **N0** | crosscomp / s0 / cross_u10 / Re150 (sub-critical) | **0.878 ± 0.051** (per-seed [0.867, 0.833, 0.933]) | **HOLDS** — Δ vs efficiency_v2 main-line anchor 0.902 = **−2.42pp** |
| **N2'** | privileged / s0 / cross_u15 / Re250 (critical) | **0.000 ± 0.000** (per-seed [0.000, 0.000, 0.000]) | **STRONG_NEGATIVE** — recovery_of_oracle = 0%，0/90 episode（rule-of-three 上界 ≈ 0.033） |
| M1 BC sweep | conditional | NOT triggered | N2' ∉ [0.15, 0.40] partial zone |

> ⚠ **首轮读法已撤销（勿再引用）**：首轮 2 种子给出 N0 0.850 ± 0.024 / Δ −5.20pp，并读作「两种子同向退化、非噪声」。
> 第三种子 seed 43 = 0.933 **高于** anchor，per-seed 方向变为 −3.5 / −6.9 / **+3.1** pp——同向退化的读法在
> report §2.4 已明确撤销，该差异按种子间波动解读。**本节 2026-08-16 才同步；此前引用过 −5.20pp / same-direction 的地方都需复核。**

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
- 2026-05-13: Step 0-4 廉价审计完成(详见 [`docs/auvhamnode_spike/`](auvhamnode_spike/));**4 项硬接口差异 + wake current 2-4× OOD 暴露**;用户决定暂停整条线

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

> 📍 **本节写于 2026-05，2026-08-09 复核补注**：§4.1–§4.4 **均已闭环**，本节已无「活跃 backlog」——写作出口（博士论文第 5 章）于 **2026-07-28 收口、送审就绪**，状态以 [`../paper/thesis_ch5/status.md`](../paper/thesis_ch5/status.md) 为准（**不在此复述章状态**）。唯一未结项是新增的 **§4.5 数据完整性待核项**。各小节原文保留作历史决策记录。

### 4.1 ~~ReBRAC paper revision (active)~~ ✅ 已随第 5 章收口闭环（2026-07-28）

⚠ 「revision (active)」是 2026-05 的状态。standalone paper 已撤销（见本文头注），素材写作出口 = 第 5 章 §5.7；该章十节成文、五批整改与收尾统稿轮均已闭环。**本小节两份锚点文档仍是素材源、角色不变**：
- [`docs/rebrac_method_section_draft.md`](rebrac_method_section_draft.md) Method 节
- [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §0/§1.5 是写作期反复回看的 One Page

### 4.2 Broad validation v2（✅ PASS；首轮 2026-05-19，三种子收口 2026-07-12）

详见 [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)（report） + [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md)（plan rev.3） + [`docs/rebrac_broad_validation_v2_seed43_supplement_plan.md`](rebrac_broad_validation_v2_seed43_supplement_plan.md)（seed 43 supplement 全档）。

**实际执行**（vs plan §6 原 4-stage 计划）：
- Stage 0A (S sanity): ✅ 完成 2026-05-18，crosscomp 0% / privileged 70% — 触发 rev.3 pivot（N2 → N2'）
- Stage 0B (dataset 收集): ✅ N0 + N2' 各 1000 ep 新 collect 完成
- Stage 1 (core training + eval): ✅ **3 seed {42, 0, 43}**（首轮 2 seed 缩水原因见 report §6.1；2026-07-12 supplement 补齐第三种子。注意实际种子组与预登记 {42, 43, 44} 不重合——首轮以 0 替换 44）；N0 0.878 ± 0.051 → HOLDS，N2' 0.000 → STRONG_NEGATIVE
- Stage 2 (conditional M1): ⏭ NOT triggered（N2' ∉ [0.15, 0.40]）

**已闭环的 follow-up**（本表 2026-08-16 订正——下列两项此前一直挂在「剩余」栏，实际早已完成）：
- ✅ **Asym-critic ablation on N2'** — 2026-05-27 完成，verdict `ACTOR_FUNDAMENTAL_CONFIRMED`：critic 全程拿到完美 hull-integral flow，s0 actor success 仍 0.000，天花板不在 critic 价值估计。详见 report §4.5
- ✅ **N0 第三种子** — seed 43 于 2026-07-12 补齐（原列为「5-seed 补全 +43, 44, 45」）

**剩余 follow-up**（不在关键路径上，详见 report §6.3）：
- 5-seed 补全 N0（再加 2 个种子）— Medium priority，若审稿人 push back
- online §7.6 candidate-C 验证（3 success episode trajectory 分析）— Medium，若 §discussion 想 strengthen anomaly section

**v1 retrofit trigger condition 已作废**（C1 BC sweep / A1 paired bootstrap / mix ratio / target=2.0 P1 等）— v1 finding 不进 paper。

### 4.3 Online 线交付的 SAC collector（**CLOSED 2026-05-26** — rev.3 Plan A + Sprint 1+2 + m_multi_mix supplement 39-run cross-source matrix; paper closure = **algorithm × data-quality interaction**）

**Spec**：[`docs/arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md) **rev.3** §4.0（rev.2 §4.1 路径 1 cross_u15 作废）。与本线协议严格对齐 = **s0 + k=4 + arrival_v2**。

**rev.3 单一路径 = Plan A**：从 `checkpoints/arrival_v2_prototype/cross_u10_regression/arrival_v2/sac_vanilla/s0_k4/seed_46/` 的训练过程切片（39 个 `agent_step_*.pt`，25k env_step cadence）里 audit pick 4 个 ckpt 构成真 D4RL 4-tier — 与 D4RL 经典 paper (CQL/IQL/FQL/ReBRAC) 一致的 tier 构造方式：

> ⚠ **该路径只在 Drive 侧（2026-08-16 实测标注）**：本机两个工作副本下 `checkpoints/` 均无
> `arrival_v2_prototype/` 子树，`experiments/arrival_v2_prototype/cross_u10_regression/` 下
> `agent_step_*.pt` 计数为 **0**。**四个 tier 数据集本身在本机**（`offline_data/sac_{random,medium,mexp,expert}_*`，
> 下表数字已逐位核对其 `metadata.json`），故下游 39-run 结果可从数据集复现；
> 只有「回到源 checkpoint 重采」这一步需要 Drive。**按现行决定不取回**，仅在此标明。

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

**Head-to-head sweep**（spec §4.0.8 v2 + §4.0.10 — **FQL P2 sister paired**）：共 36 run = 3 configs (FQL + ReBRAC β1=1/β2=2 + ReBRAC β1=4/β2=2) × 4 tier × 3 seed [42, 0, 7]（与 **FQL P2 v1.4 extended 3-seed** 严格 paired），seed-shard 3-session L4 并发：
- **Sprint 1 ReBRAC × 24 run** ✅ **COMPLETED 2026-05-25** — wall ~3.3h (commits `5070862` / `8ed3aeb` / `104c14c`)
- **Sprint 2 FQL × 12 run** ✅ **COMPLETED 2026-05-26** — wall ~3.5h (commits `f5608cc` / `fc5cbce`)。36 test_result.json 落 `results/offline/sac_collector_h2h/{rebrac,fql}/`。
- **M_multi_mix β1=1 supplement × 3 run** ✅ **COMPLETED 2026-05-26** — single L4 session ~1.7h, 补 FQL P2 v1.4 cross-source matrix 缺的 β1=1 cell (commits `f716a5d` / `cd56c8b`)。Joint 39-run matrix → cross-source paired bootstrap (spec §4.0.10)。

**分析脚本**: [`scripts/analyze_sac_collector_h2h_joint.py`](../scripts/analyze_sac_collector_h2h_joint.py)（36-cell stratified bootstrap）+ [`scripts/analyze_sac_collector_h2h_xsource.py`](../scripts/analyze_sac_collector_h2h_xsource.py)（cross-source paired bootstrap vs FQL P2 v1.4 m_multi_mix）。

**4 tier ↔ FQL P2 3 cell 天然对位**：expert↔E-uni clean / mexp↔M-uni-noise σ=0.5 / medium↔M-multi-mix / random=独立贡献。

**Paper finding 现状（39 run 闭环后，详 spec §4.0.10）**：

1. **β1=1 dominate β1=4 on non-saturated cells** — 跨数据源 universal weak dominance:
   - ✅ SAC mexp: Δ +0.089 stratified CI [+0.011, +0.178] 不跨 0
   - ✅ FQL P2 m_uni_noise: Δ +0.235 (FQL P2 v1.4 §6.3 Q1b/Q1c)
   - ⚠️ Saturated cells (SAC expert, m_multi_mix): inconclusive (no power), NOT falsifying
   - DESCRIPTION ERROR retired: 原 "翻转点" hypothesis 由 commit 8a9616e mis-statement 导致, 已 fix

2. **Algorithm × data-quality interaction (PAPER MAIN FINDING)** — direction FLIPS across regimes:
   - **SAC mexp (collector SR 0.751, non-saturated)**: **FQL > ReBRAC β1=1** Δ +0.123 CI [+0.022, +0.233] ✓
   - **FQL P2 m_multi_mix (collector SR 0.826 mix, saturated 0.95-0.99)**: **ReBRAC β1=1 > FQL** Δ +0.035 CI [+0.010, +0.065] ✓
   - Both CIs strictly nonzero → NOT noise, **regime-dependent algorithm ranking**
   - 高质量 saturated: ReBRAC actor BC penalty exploits clean modes
   - 中质量 non-saturated: FQL flow-matching teacher extracts policy from noisy data
   - → **FQL P2 v1.4 finding 2 ("β1=1 universal dominance over FQL") is refined to regime-dependent**

3. **SAC expert dataset-bound regime confirmed** — ReBRAC β1∈{1,4} + FQL 都 0.889-0.911 ≈ SAC src ceiling 0.899 → algorithm-agnostic dataset-bound, sprint 2 finding 2 (RESCUE-FAIL universality) 在 saturated regime 上 inconclusive。

**Paper closure 路径（locked 2026-05-26）**: cross-source contradiction → algorithm × data-quality interaction 写成主章节, 39 run 39 cell 矩阵作 §results 主表。SAC collector 作 controlled mid-quality stress-test, 揭示 privileged-collector tests 看不到的 regime-dependent ranking。

**关键约束**：`s0_k4 + arrival_v2 + cross_u15` 上不存在 expert SAC ckpt（rev.2 §4.5）— sensor floor 实证。Plan A 选 cross_u10 而不是 cross_u15，正是基于此约束。

**实际总成本**（vs spec ~21h L4 estimate）：sprint 1 ~3.3h + sprint 2 ~3.5h + supplement ~1.7h = **~8.5h L4** wall（seed-shard 3× 并发节省）。adapter + audit + collection ~3.75h 已计入。

### 4.4 AUVHamNODE Offline RL(⏸ PAUSED 2026-05-13)

整条线已 paused,**不在当前 backlog 内**。Anchor 见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md);恢复条件见同文 §6;恢复时 first step 见 §7。

短期不需要的工作:Path 1B spike-lite 实施、Path 2 finetune 评估、Path 4 vehicle.py oracle plan 起草、AUVHamNODE 上游 dataset 取回。这些都在 pause memo §5 留作债务表。

### 4.5 数据完整性问题（2026-08-02 记录 → 2026-08-16 复核，处置已落地；污染面 2026-08-21 全仓封闭）

清单与完整证据在 [`docs/data_integrity_open_items.md`](data_integrity_open_items.md)。**第 ①、③、⑤ 条已核实成立**（③ 比原怀疑更强；⑤ 为 2026-08-16 独立复核新增，本节 2026-08-16 才补录），**处置已于 2026-08-16 落地**（路线 ①-c：如实披露 + 登记敏感性读数——`setup.tex` §5.3.5/§5.3.6、`rebrac.tex` §5.7.1/§5.7.2/§5.7.m、`discussion.tex` §5.10.4）；不影响方法本身，但都会在答辩/返修时被问到。

- **① `crosscomp-2000` 训练与评估用的是同一批任务实例**（⚠ 最高优先，**已确证**）。数据集 `seed=0`、2000 回合 → 训练种子 0..1999 完整包含评估 manifest 的 1250..1349；reset RNG 重放确认 **100/100 实例逐条相同**。成因是 `collect_offline_data.py` 的 `--seed` 默认值为 0 加上规模跨过 1250。**本机 `offline_data/` 范围内仅此一个受影响**（其余 9 个止于种子 999，低于最小 `manifest_seed`=1100）。⚠ **本句原写「污染面至今未封闭」，2026-08-20 订正为两层**：本机补齐章内依赖后重跑 $29$ 个集，OVERLAP 仍只有两格（本条与 ⑤）——**第 5 章范围已封闭**；**仓库全域已于 2026-08-21 封闭**——Drive 侧实跑 $35$ 个集 × $24$ 个 manifest，零 `[ENUM MISS]`／零 `[SHADOWED ]`／退出码 $0$，账本 $41$ 个名字全部落定。此前的枚举残缺经实测确认是 `rglob` 不进符号链接目录（三个数据集在 Drive 上是软链），已由 `06e6d1f` 修掉；补回后 OVERLAP 为 $22$ 处 ＝ 两个 $2000$ 回合集（确定性与含噪）× $11$ 个 manifest，**受影响的数据集仍是那两个，没有新增第三个**。⚠ 该封闭依赖 Drive 上 $13$ 个未进 git 的评估 manifest，详见事实账本。波及 ReBRAC 主线 `cross-2000` 一格与"2000 不再劣于 1000"这条翻转论述；反向的"2000<1000"结论不受威胁（污染只会抬高 2000）。
- **② paper Table 1 的 transitions 数字与本机 metadata 对不上** —— 仅影响已撤销的 standalone paper 旧稿（`paper/archive/rebrac_standalone/sections/setup.tex`）；论文第 5 章已于 rev.3（2026-06-18）用实测值 1.5e5 / 3.0e5 / 1.0e5 纠正过，四源互证。
- **③ 选点验证集是终报测试集的前缀子集**（**已确证**，强于原怀疑）。manifest 生成器无 seed 偏移，val/test 用同一 benchmark key → `val_40` = `test_100` 的前 40 条；40+40 的单元里两份 manifest 完全相同。即报告的 100 回合测试集中有 40 条正是选 checkpoint 用的那批。
- **④ 行为策略 success_rate 0.958 vs ReBRAC-Q 0.928** —— 非缺陷，建议把两个 behaviour policy 在评估 manifest 上的成功率补成一行，纯 eval 开销。
- **⑤ 含噪 2000 回合数据集的种子同样重叠**（2026-08-16 独立复核新增，Drive 侧探针**已钉死**）。`crosscomp_..._noise0p05clip0p15_ep2000` 同为 `seed=0` / 2000 回合 → 种子 0..1999 完整包含 manifest 的 1250..1349，reset RNG 重放 **100/100 逐条相同**，与 ① 同因同病。波及第 5 章 §5.6.2 那句 noisy-support 诊断——`0.60` 出自确定性 2000（①）、`0.74` 出自本条，**两个端点都坐在污染数据上**；且该筛查是 40+40 单元，val 与 test 逐条相同（③）——**三条问题在同一句上同时命中**。减轻情节：`td3bc.tex` rev.5 已把机制归属挪到干净的一千回合反证。

**处置与复核状态**（详见 [`data_integrity_open_items.md`](data_integrity_open_items.md) 末节）：波及面评估初稿 + [独立复核](../paper/thesis_ch5/data_integrity_impact_assessment_review.md)（判「可作裁决依据但须打补丁」）；③ 已用 [`ch5_holdout_split_audit.py`](../paper/thesis_ch5/tools/ch5_holdout_split_audit.py) 零成本量化（留出 60 条上组间差全部保号且变大，唯一例外是 §5.7.2 的 53.0% → 48.9%）；① 的污染幅度 δ **已于 2026-08-16 实测**（干净 manifest `--seed 3000` 上原封不动重跑两格 × 5 种子，纯评估）：差中差 **$+0.80$ pp**（种子级 95% CI $[-3.5,+5.1]$ 跨零），两格绝对水平另同向下移 $4.0$/$4.8$ pp。详见 `rebrac.tex` §5.7.1/§5.7.m，复算 [`ch5_clean_probe_readout.py`](../paper/thesis_ch5/tools/ch5_clean_probe_readout.py)。（本行原写「δ 仍未测」，源文档 `data_integrity_open_items.md` 已于 `4295f68` 订正，本副本漏跟。）

**复核工具**：[`scripts/audit_seed_overlap.py`](../scripts/audit_seed_overlap.py)（2026-08-09 入库，2026-08-16 改为递归扫描）。比对数据集 × manifest 的种子区间相交（只在流场／几何／目标速度三项一致时才判定），`--verify DATASET MANIFEST` 重放 reset RNG 逐条比对任务实例。⚠ **它只扫所在机器的 `offline_data/` 与 `benchmarks/`**——本机跑出的「已封闭」不等于全仓封闭，且 Drive 侧枚举已被证残缺。任何**新采集的数据集或新评估 manifest 落地后都该跑一次**。

---

## 5. 仓库 offline 线文件索引

按"想找 X 看 Y"的角度组织。所有路径相对仓库根。

### 5.1 顶层导航：当前应该读什么

| 想找... | 看这里 |
|---|---|
| Offline 线整体状态、时间轴、下一步 | **本文件**（[`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md)） |
| ⚠ 已知未核的数据完整性问题（种子重叠 / 表格数字 / manifest 口径） | [`docs/data_integrity_open_items.md`](data_integrity_open_items.md)（本文 §4.5 为摘要） |
| ReBRAC 这一条线串线总览（TD3+BC→ReBRAC→FQL，含 β1 跨线 reconciliation） | [`docs/rebrac_line_overview.md`](rebrac_line_overview.md) |
| ReBRAC One Page 摘要 + 4 finding spine | [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §0 + §1.5 |
| ReBRAC 任何具体数字 / per-seed | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §1–§10（按 stage 索引） |
| 当前有效的广验结果（v2，三种子） | [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) ★ |
| ❌ v1 三轴 8 spoke 全表（**已作废，不进论文**，仅历史） | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3 |
| ❌ v1 C1 spoke task-fundamental floor 证据链（同上，作废） | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3.5 + [`docs/rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) |
| TD3+BC baseline 详细收口 | [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) |
| AUVHamNODE Offline MBRL 线为何 paused / 累计决策 / 恢复条件 | [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) ★(pause anchor) |
| AUVHamNODE Offline MBRL 历史 plan(paused) | [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md)(v2.1, paused) + [`auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](auvhamnode_offline_mbrl_plan_v3_pre_notes.md) + [`docs/auvhamnode_spike/`](auvhamnode_spike/) |
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
| [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) ★ | **active（3-seed 收口 2026-07-12）** | v2 数字 ground truth — N0 HOLDS (0.878 ± 0.051) / N2' STRONG_NEGATIVE；paper §experiments appendix headline = actor-fundamental partial-observability ceiling under s0。**数字以本文件为准，本总览曾停在首轮 2-seed** |
| [`rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) | **closed plan (2026-05-19, rev.3)** | v2 cross-only spotlight under arrival_v2；5 cell core + 1 conditional sweep；rev.3 pivot N2→N2'；闭环已 PASS（详见 report）|
| [`rebrac_broad_validation_v2_seed43_supplement_plan.md`](rebrac_broad_validation_v2_seed43_supplement_plan.md) | **closed (2026-07-12)** | seed 43 补齐三单元（N0 / N2' / asym）的 plan + 呈报 + 裁决全档 |
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
| [`docs/auvhamnode_spike/`](auvhamnode_spike/)(Step 0-4 审计 + spike-lite 设计 + decision memo) | ⏸ Step 0-4 ✅;Path 1B spike 未执行 |
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

留在 `docs/` 顶层但只被代码引用的三份（`thes=0 / entr=0` 的入链计数分不出「历史记录」和「活契约」）：

| 文件 | 状态 | 角色 |
|---|---|---|
| [`fql_pytorch_port_design.md`](fql_pytorch_port_design.md) | **IMPLEMENTED** | `auv_nav/fql.py` 的现行 design contract；模块头注 + `tests/test_fql.py` §13 指回它 |
| [`fql_audit_multimodality_design.md`](fql_audit_multimodality_design.md) | **IMPLEMENTED** | `scripts/audit_multimodality.py` 同上；`tests/test_audit_multimodality.py` §9 是其测试计划 |
| [`fql_e_uni_anchor_dataset_card.md`](fql_e_uni_anchor_dataset_card.md) | COLLECTED | E-uni 1000-ep 数据卡；P2 main spec §Status 引它作 Task D 闭环凭证 |

**P2 之前的施工记录（7 份）已于 2026-08-17 迁入
[`archive/fql_succession/`](archive/fql_succession/README.md)** —— plan v0 / P0+P1 spec /
Gate B 终报 + 中间报告 / bug2 + c4 两项 P2 前置决策 / audit dry-run 报告。
**归档只表示改变位置与标注，不含有效性判断**：这些文件的结论未被撤销，只是不再作为论文引用入口。
Gate B 终报仍是**冻结超参的来源**（flow_steps=10 / distill_alpha_bc=1.0 / β1=4 β2=2），
审稿人问「怎么调的」时回查该目录。

### 5.3 实验数据 / checkpoints / offline data

> ⚠ **2026-08-16 订正**：本表原先整张指向 `experiments/offline/...`，实测该树下**只有**
> `experiments/offline/rebrac/broad_validation_v2/`；offline 线的读数实际都在 **`results/offline/`** 下
> （v2 report 自己的 Raw outputs 行写的也是 `results/offline/rebrac/...`）。照旧表去找会误判为「数据丢失」。

| 路径 | 角色 |
|---|---|
| `results/offline/rebrac/formal/` + `screening/` | ReBRAC 主线 5-seed 实验数据 |
| `results/offline/rebrac/broad_validation/` | v1 广验 8 spoke（⚠ 已作废，不进论文） |
| `results/offline/rebrac/broad_validation_v2/` + `broad_validation_v2_n2p_asym/` | v2 三种子 + asym-critic ablation（**当前有效的广验数据**） |
| `results/offline/rebrac/c1_s1_sensor_upgrade/` + `c1_*_ablation/` | v1 C1 deep-dive 系列（⚠ 同上，作废） |
| `results/offline/rebrac/critic_ln_off/` + `stage_e_critic_penalty_off/` + `worldcomp_*/` | Stage E/F probe 数据（finding iii/iv 依据） |
| `results/offline/rebrac/clean_probe/` | 终检复现证据 |
| `results/offline/td3bc/{phase0,phase0b_v2,phase0c}/` | TD3+BC 数据（已归档） |
| `results/offline/sac_collector_h2h/{rebrac,fql}/` | SAC collector head-to-head 36 run |
| `offline_data/<collector>_<sensor>_<reward>_<flow>_ep<N>/` | 19 个 offline 数据集（含 4 个 `sac_*` tier） |
| `checkpoints/offline/td3bc/` | TD3+BC checkpoint。**注意 `checkpoints/offline/rebrac/` 本机不存在**——ReBRAC 训练产物只在 Drive 侧 |

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

Offline RL 线在 2026-Q1 → 2026-05 累计完成 **TD3+BC baseline closure(Phase 0c 5 文档已归档)+ ReBRAC 主线 paper-ready 4/4 closed (rev.8) + v1 三轴 broad validation 8 spoke 5-seed parity**;主线 paper drafting 进入 revision 阶段;**v1 广验 + c1_s1 follow-up 已于 2026-05-18 SUPERSEDED**(reward 失配 + online §7.6 更强 finding)，v1 archive 保留不重跑;**v2 broad validation 已 PASS**(arrival_v2 cross-only；首轮 2-seed 4-run 2026-05-19，2026-07-12 补齐第三种子收口为 3 seed {42, 0, 43} — N0 HOLDS 0.878 ± 0.051 / N2' STRONG_NEGATIVE 0/90，paper §experiments appendix headline = actor-fundamental partial-observability ceiling under s0；详见 [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md));**原计划下一阶段 AUVHamNODE Offline RL 已于 2026-05-13 paused**(详见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md));**FQL Succession（FQL vs ReBRAC，Paper 2 候选）已于 2026-05-23 NEGATIVE 闭环**("FQL > ReBRAC iff sub-optimal AND multi-modal" 证伪 → 机制发现 + 诚实负面:noise 是 discriminator、ReBRAC β1=1.0 双轴 dominate、BC-anchor 目标质量决定鲁棒性;详见 [`fql_succession_p2_results.md`](fql_succession_p2_results.md))。

**2026-08-09 补注（本段以上写于 2026-05-07，其中"主线 paper drafting 进入 revision 阶段"已过期）**：2026-05 之后本线**没有新增实验**，全部工作转入写作出口——博士论文第 5 章于 **2026-07-28 收口、送审就绪**（章状态唯一真相源 = [`../paper/thesis_ch5/status.md`](../paper/thesis_ch5/status.md)，**此处不复述**）。本线现存唯一未结事项是本文 **§4.5 数据完整性待核项**（清单 [`data_integrity_open_items.md`](data_integrity_open_items.md)）。
