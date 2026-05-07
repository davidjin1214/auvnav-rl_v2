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
| **Phase 2.5 — Broad validation（standalone）** | 2026-05-04 → 2026-05-07 | ReBRAC 跨三轴 generality 广验 + C1 deep-dive | ⚠ 8 spoke 5-seed parity 完成；6/8 work-in-progress，**结论尚不达 paper-quality** | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) |
| **Phase 3 — AUVHamNODE Offline MBRL** | 2026-05-06 起 | AUVHamNODE world model + offline MBRL | 📋 计划态（v1.0 plan，待 ReBRAC 收尾后启动） | [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) |

**当前位置**：Phase 2 主线已 freeze；Phase 2.5 广验保持 standalone 等下游 sweep 闭环；Phase 3 plan v1.0 待 fire。

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

### 3.3 Phase 2.5 — Broad validation（standalone exploratory）

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

### 3.4 Phase 3 — AUVHamNODE Offline MBRL（计划态）

**入口**：[`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) (v1.0, 2026-05-06)

**v1.0 status**：批判性合并了 [`docs/offline_mbrl_plan/`](offline_mbrl_plan/) 下 5 份草案；v1.0 是**唯一活跃入口**，5 份草案已被吸收。

**核心思路**（详见 plan §0–§1）：
- World model = AUVHamNODE（Hamiltonian Neural ODE 引入 AUV 物理结构先验）
- Offline MBRL 框架: world model rollout 增广 + ReBRAC actor-critic
- 跨线复用：online 线 [`auv_nav/autopilot.py`](../auv_nav/autopilot.py) `EquivalentCurrentModel` + offline 线 [`auv_nav/rebrac.py`](../auv_nav/rebrac.py) ReBRAC agent

**触发条件**：等 ReBRAC paper revision 收尾 + broad validation 下游 sweep 状态稳定后再 fire。

---

## 4. 当前活跃 backlog 与下一步

### 4.1 ReBRAC paper revision (active)

paper drafting Phase 5 → revision 阶段，主要锚点：
- [`docs/rebrac_method_section_draft.md`](rebrac_method_section_draft.md) Method 节
- [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §0/§1.5 是写作期反复回看的 One Page

### 4.2 Broad validation 下游 sweep（决定能否 retrofit 主线）

按优先级：
1. **★ BC penalty 强度 sweep on C1**（β1 ∈ {0, 1, 2, 4, 8}）— 区分 BC floor vs task-fundamental floor，是 task-fundamental claim 的关键 mechanism discriminator
2. **A1 paired bootstrap (n_boot=10000)** — 区分 directional consistent 与 statistically robust
3. **mix ratio sweep on A2**（70/30, 90/10）— 验证 mid-gap collapse 是否单调，是否真的是 mode collapse 现象
4. **target_speed=2.0 P1 probe** — 测试 task-fundamental claim 的 speed-axis 边界
5. **C1 BC penalty sweep on mix dataset** — A2 hypothesis 的直接验证

完成 1–2 项后即可考虑把广验 + c1_s1 部分回写主报告。

### 4.3 Online 线交付的 SAC collector（待 spec）

[`docs/online_rl_line_summary.md`](online_rl_line_summary.md) §4.3 列了 D4RL-style SAC collector 作为 broad validation **平行第四轴**（不替代 A2 mix5050），spec 未定。本线消费方角度：等 §4.2 下游 sweep 收敛后再决定是否真的需要 RL-trained behavior policy 数据集。

### 4.4 AUVHamNODE Offline MBRL fire 时机

ReBRAC paper revision 收尾 + 广验下游 sweep 至少 1 项闭环后启动。

---

## 5. 仓库 offline 线文件索引

按"想找 X 看 Y"的角度组织。所有路径相对仓库根。

### 5.1 顶层导航：当前应该读什么

| 想找... | 看这里 |
|---|---|
| Offline 线整体状态、时间轴、下一步 | **本文件**（[`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md)） |
| ReBRAC One Page 摘要 + 4 finding spine | [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) §0 + §1.5 |
| ReBRAC 任何具体数字 / per-seed | [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §1–§10（按 stage 索引） |
| Broad validation 三轴 8 spoke 全表 | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3 |
| C1 spoke task-fundamental floor 候选证据链 | [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) §3.5 + [`docs/rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) |
| TD3+BC baseline 详细收口 | [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) |
| AUVHamNODE Offline MBRL 计划 | [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) |
| Env / sensor / reward / benchmark 规格 | [`docs/environment_design.md`](environment_design.md) |
| World model + offline RL 综述 | [`docs/world_model_and_offline_rl_survey.md`](world_model_and_offline_rl_survey.md) |
| RLPD 设计（offline-to-online，跨线复用） | [`docs/rlpd_design.md`](rlpd_design.md) |
| 论文写作期"读哪份、抄哪段" | [`docs/rebrac_paper_writing_index.md`](rebrac_paper_writing_index.md) |

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
| [`rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) (rev.2) | **standalone exploratory** | 三轴广验 publication-ready draft；保持独立直到下游 sweep 闭环 |
| [`rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) | **standalone exploratory** | C1-s1 sensor upgrade follow-up；保持独立直到 BC penalty sweep on C1 闭环 |

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

#### E. Offline MBRL 下一阶段（计划态）

| 文件 | 状态 |
|---|---|
| [`auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) (v1.0, 2026-05-06) ★ | 唯一活跃入口 |
| [`offline_mbrl_plan/`](offline_mbrl_plan/) 下 5 份草案 | 已被 v1.0 批判性合并吸收；deprecated banner 待加 |

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
| [`scripts/train_offline_rebrac.py`](../scripts/train_offline_rebrac.py) | ReBRAC 训练 entry point |
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
| **想看未来方向** | + [`auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) §0–§1 |

---

## 7. 一句话总结

Offline RL 线在 2026-Q1 → 2026-05 累计完成 **TD3+BC baseline closure（Phase 0c 5 文档已归档）+ ReBRAC 主线 paper-ready 4/4 closed (rev.8) + 三轴 broad validation 8 spoke 5-seed parity**；主线 paper drafting 进入 revision 阶段；广验与 c1_s1 follow-up 因结果尚不达 paper-quality（mechanism discriminator 如 BC penalty sweep on C1 / mix ratio sweep / target_speed=2.0 P1 / paired bootstrap 未做）**保持 standalone 不回写主线**；下一阶段是 AUVHamNODE Offline MBRL（v1.0 plan 待 fire）。
