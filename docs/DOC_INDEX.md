# 全仓文档索引

> **自动生成，请勿手改。** 由 `python -m scripts.build_doc_index` 重建；
> `--check` 会在本文件落后于实际时报错。
>
> 只提取两样东西：H1 标题、以及文件开头 15 行内的状态横幅。
> **状态栏的 `—` 表示「该文档未自我标注」，不表示「仍然有效」**——没有横幅是证据缺失，
> 判断一份文档是否仍然作数，仍然是人的活。
>
> 研究线的叙事索引（阶段时间轴、结论路由、数字 ground truth）在
> [`offline_rl_line_summary.md`](offline_rl_line_summary.md) 与
> [`online_rl_line_summary.md`](online_rl_line_summary.md)；本文件是**文件地图**，两者角色不同。

共 119 个 markdown 文件。

## 入口与总纲（3）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`AGENTS.md`](../AGENTS.md) | AGENTS.md | — |
| [`CLAUDE.md`](../CLAUDE.md) | CLAUDE.md | — |
| [`README.md`](../README.md) | AUV Navigation in Complex Flow Fields | — |

## docs/ — 研究与实现文档（48）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) | `arrival_v2` Reward — Experiment Report | — |
| [`docs/arrival_v2_p0_variance_reduction_design.md`](arrival_v2_p0_variance_reduction_design.md) | arrival_v2 §8 P0 — SAC Variance Reduction Design | — |
| [`docs/arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md) | `arrival_v2` SAC Collector — Design & Checkpoint Inventory | — |
| [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md) | AUVHamNODE + Offline MBRL 线 — 暂停归档备忘 | — |
| [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) | AUVHamNODE-based Offline RL:Cross-Domain Transfer via Frozen Physics-Structured 1-Step … | PAUSED 2026-05-13 |
| [`docs/auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](auvhamnode_offline_mbrl_plan_v3_pre_notes.md) | AUVHamNODE Offline MBRL — v3.0 Pre-Notes(交接备忘) | PAUSED 2026-05-13 |
| [`docs/data_integrity_open_items.md`](data_integrity_open_items.md) | 数据完整性待核项（2026-08-02 记录） | — |
| [`docs/doc_cleanup_status.md`](doc_cleanup_status.md) | 文档整理 — 状态与交接 | — |
| [`docs/environment_design.md`](environment_design.md) | 复杂流场中欠驱动AUV导航的仿真环境设计 | — |
| [`docs/fql_audit_multimodality_design.md`](fql_audit_multimodality_design.md) | `scripts/audit_multimodality.py` — Detail Design Doc | — |
| [`docs/fql_e_uni_anchor_dataset_card.md`](fql_e_uni_anchor_dataset_card.md) | FQL E-uni Paper Anchor Dataset — Card | — |
| [`docs/fql_pytorch_port_design.md`](fql_pytorch_port_design.md) | `auv_nav/fql.py` — PyTorch Port Design Doc | — |
| [`docs/fql_succession_p2_collection_log.md`](fql_succession_p2_collection_log.md) | FQL Succession P2 — Sprint 0 Collection Log | — |
| [`docs/fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) | FQL Succession — P2 Main Comparison Spec | SUPERSEDED 2026-05-20 |
| [`docs/fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) | FQL Succession P2 — Mechanism Diagnostic (option C) | — |
| [`docs/fql_succession_p2_results.md`](fql_succession_p2_results.md) | FQL Succession P2 — Results & Verdict | — |
| [`docs/fql_succession_p2_xbench_spec.md`](fql_succession_p2_xbench_spec.md) | FQL Succession P2 — Cross-Benchmark Confirmation Spec (`single_u15_cross`) | — |
| [`docs/fql_succession_paper_writing_index.md`](fql_succession_paper_writing_index.md) | FQL Succession 写作文档索引（原 Paper 2 候选 → 博士论文第 5 章 §N.6） | — |
| [`docs/generate_wake_usage.md`](generate_wake_usage.md) | `scripts/generate_wake.py` 使用说明 | — |
| [`docs/offline_rl_implementation_plan.md`](offline_rl_implementation_plan.md) | 离线强化学习实现方案与计划 | DEPRECATED 2026-05-08 |
| [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) | Offline RL 线总结报告 | — |
| [`docs/offline_rl_quick_validation.md`](offline_rl_quick_validation.md) | 离线 RL 快速验证方案 | — |
| [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) | Online RL 线总结报告 | CANCELLED 2026-07-28 |
| [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) | Online RL Thesis 实验计划【DEPRECATED】 | DEPRECATED 2026-05-06 |
| [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) | Online SAC Reward Redesign Note【SHELVED / v6 Pre-Integration Spec】 | — |
| [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) | ReBRAC 广验报告：跨数据质量 / 传感器 / 任务三轴的 Probe-then-Deepen | SUPERSEDED 2026-05-18 |
| [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) | ReBRAC Broad Validation v2 — Cross-Only Spotlight under `arrival_v2` | SUPERSEDED 2026-05-04 |
| [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) | ReBRAC Broad Validation v2 — Experimental Report | — |
| [`docs/rebrac_broad_validation_v2_seed43_supplement_plan.md`](rebrac_broad_validation_v2_seed43_supplement_plan.md) | ReBRAC Broad Validation v2 — 补种子（seed 43）Supplement Plan | — |
| [`docs/rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md) | ReBRAC C1-s1 Sensor-Upgrade Follow-up — Standalone Report | SUPERSEDED 2026-05-18 |
| [`docs/rebrac_experiment_plan.md`](rebrac_experiment_plan.md) | ReBRAC 实验计划 | DEPRECATED 2026-05-08 |
| [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) | ReBRAC 实验报告 | — |
| [`docs/rebrac_line_overview.md`](rebrac_line_overview.md) | ReBRAC 线总览（贯通 TD3+BC 前置 ↔ ReBRAC 主线 ↔ FQL succession） | — |
| [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) | ReBRAC 主线总结、专家分析与下一步建议 | SUPERSEDED |
| [`docs/rebrac_method_section_draft.md`](rebrac_method_section_draft.md) | Method Section Draft — Q-normalized dual-penalty TD3+BC variant | — |
| [`docs/rebrac_paper_writing_index.md`](rebrac_paper_writing_index.md) | ReBRAC Paper 写作文档索引 | — |
| [`docs/rebrac_statistical_test_followup.md`](rebrac_statistical_test_followup.md) | ReBRAC Phase 1 deployable vs TD3BC privileged-critic — Statistical test follow-up | — |
| [`docs/rlpd_design.md`](rlpd_design.md) | RLPD：利用离线数据加速 AUV 流场导航策略学习 | DEPRECATED 2026-05-08 |
| [`docs/SAC_improvements_survey.md`](SAC_improvements_survey.md) | SAC 改进算法研究综述 | — |
| [`docs/systematic_improved_sac_experiment_plan.md`](systematic_improved_sac_experiment_plan.md) | 改进 SAC 的系统实验方案【DEPRECATED】 | DEPRECATED 2026-04-26 |
| [`docs/systematic_improved_sac_experiment_report.md`](systematic_improved_sac_experiment_report.md) | 改进 SAC 的系统实验报告【DEPRECATED】 | DEPRECATED 2026-04-26 |
| [`docs/td3bc_mainline_closure_plan.md`](td3bc_mainline_closure_plan.md) | TD3BC 主线收口执行计划 | — |
| [`docs/td3bc_phase0b_v2_experiment_report.md`](td3bc_phase0b_v2_experiment_report.md) | TD3BC `phase0b_v2` 实验报告 | — |
| [`docs/td3bc_phase0c_experiment_design.md`](td3bc_phase0c_experiment_design.md) | TD3BC `phase0c` 实验设计文档 | — |
| [`docs/td3bc_phase0c_experiment_report.md`](td3bc_phase0c_experiment_report.md) | TD3BC `phase0c` 实验报告 | — |
| [`docs/td3bc_worldcomp_teacher_gap_experiment_report.md`](td3bc_worldcomp_teacher_gap_experiment_report.md) | TD3BC `worldcomp teacher-gap` 实验报告 | — |
| [`docs/wallclock_profile_2026_05_17.md`](wallclock_profile_2026_05_17.md) | Wallclock Profile — 决定不做 GPU env 迁移的数据依据 | ARCHIVE |
| [`docs/world_model_and_offline_rl_survey.md`](world_model_and_offline_rl_survey.md) | 世界模型与离线-在线 RL 研究综述 | — |

## docs/archive/fql_succession/ — FQL succession P2 之前的施工记录（归档=移位+标注，非有效性判断）（8）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`docs/archive/fql_succession/fql_audit_dryrun_report.md`](../docs/archive/fql_succession/fql_audit_dryrun_report.md) | FQL Audit Multimodality — Task A Dry-Run Report | ARCHIVE 2026-08-17 |
| [`docs/archive/fql_succession/fql_succession_bug2_fix_decision.md`](../docs/archive/fql_succession/fql_succession_bug2_fix_decision.md) | Bug 2 修复方向决策 — `--episodes` vs manifest size | ARCHIVE 2026-08-17 |
| [`docs/archive/fql_succession/fql_succession_c4_threshold_revision.md`](../docs/archive/fql_succession/fql_succession_c4_threshold_revision.md) | c4 阈值改造决策 — slope ≥ 0 → no-major-collapse | ARCHIVE 2026-08-17 |
| [`docs/archive/fql_succession/fql_succession_gate_b_interim_report.md`](../docs/archive/fql_succession/fql_succession_gate_b_interim_report.md) | FQL Succession Gate B — Interim Report (seed=42 single-seed) | SUPERSEDED 2026-07-28 |
| [`docs/archive/fql_succession/fql_succession_gate_b_report.md`](../docs/archive/fql_succession/fql_succession_gate_b_report.md) | FQL Succession Gate B — Final Report (Option B 2-seed) | ARCHIVE 2026-08-17 |
| [`docs/archive/fql_succession/fql_succession_p0p1_spec.md`](../docs/archive/fql_succession/fql_succession_p0p1_spec.md) | FQL Succession — P0+P1 Executable Spec | CLOSED |
| [`docs/archive/fql_succession/fql_succession_plan_v0.md`](../docs/archive/fql_succession/fql_succession_plan_v0.md) | FQL Succession Plan v1 — Offline RL on Data Quality × Modality Spectrum (Lean MVP) | SUPERSEDED 2026-06-02 |
| [`docs/archive/fql_succession/README.md`](../docs/archive/fql_succession/README.md) | FQL Succession — P2 之前的施工记录（归档） | ARCHIVE 2026-08-17 |

## docs/auvhamnode_spike/ — AUVHamNODE 预 spike 审计（⏸ 线已暂停）（6）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`docs/auvhamnode_spike/00_smoke_test_log.md`](../docs/auvhamnode_spike/00_smoke_test_log.md) | Step 0 — Environment Smoke Test (AUVHamNODE phnode_full_oc_clean) | — |
| [`docs/auvhamnode_spike/01_static_distribution_audit.md`](../docs/auvhamnode_spike/01_static_distribution_audit.md) | Step 1 — Static Distribution Audit (AUVHamNODE training vs PlanarRemusEnv deployment) | — |
| [`docs/auvhamnode_spike/02_spike_lite_design.md`](../docs/auvhamnode_spike/02_spike_lite_design.md) | Spike-Lite Design (preflight kill-test, ~4-6 hours, only run if user picks Path 1B) | — |
| [`docs/auvhamnode_spike/03_dynamics_consistency_audit.md`](../docs/auvhamnode_spike/03_dynamics_consistency_audit.md) | Step 3 — Dynamics Consistency Audit (`auv_nav/vehicle.py` vs `remus100_core.py`) | — |
| [`docs/auvhamnode_spike/04_swap_vehicle_decision_memo.md`](../docs/auvhamnode_spike/04_swap_vehicle_decision_memo.md) | Step 4 — Swap `vehicle.py` → `remus100_core.py`? Decision Memo | — |
| [`docs/auvhamnode_spike/README.md`](../docs/auvhamnode_spike/README.md) | `experiments/auvhamnode_spike/` — AUVHamNODE Offline MBRL Pre-Spike Workspace | PAUSED 2026-05-13 |

## docs/offline_mbrl_plan/ — 已废弃的 MBRL 草案（未纳入 git）（6）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`docs/offline_mbrl_plan/auv_fql_flow_offline_rl_method_selection_report_zh.md`](../docs/offline_mbrl_plan/auv_fql_flow_offline_rl_method_selection_report_zh.md) | AUV复杂流场高效导航中的Flow/FQL系离线强化学习方法选择报告 | DEPRECATED 2026-05-08 |
| [`docs/offline_mbrl_plan/AUV_offline_RL_neural_ODE_report.md`](../docs/offline_mbrl_plan/AUV_offline_RL_neural_ODE_report.md) | 基于 Neural ODE 的 AUV 复杂流场 Efficient Navigation：Offline RL 方法审查与研究方案报告 | DEPRECATED 2026-05-08 |
| [`docs/offline_mbrl_plan/AUV_REBRAC_NeuralODE_OfflineRL_v2_report.md`](../docs/offline_mbrl_plan/AUV_REBRAC_NeuralODE_OfflineRL_v2_report.md) | 新版报告：在 AUV 复杂流场高效导航中，将 Neural ODE 接入 REBRAC / Offline RL 的建议 | DEPRECATED 2026-05-08 |
| [`docs/offline_mbrl_plan/AUV_REBRAC_NeuralODE_quick_route_v3.md`](../docs/offline_mbrl_plan/AUV_REBRAC_NeuralODE_quick_route_v3.md) | 快速研究路线审查报告：REBRAC + Neural ODE 用于 AUV 复杂流场高效导航 | DEPRECATED 2026-05-08 |
| [`docs/offline_mbrl_plan/NODE_IQL_FQL_SORL_revised_roadmap_v3.md`](../docs/offline_mbrl_plan/NODE_IQL_FQL_SORL_revised_roadmap_v3.md) | 面向复杂流场 AUV 离线强化学习的 NODE-IQL / NODE-FQL / NODE-SORL 研究路线报告 | DEPRECATED 2026-05-08 |
| [`docs/offline_mbrl_plan/offline_mbrl_for_auv_navigation_report.md`](../docs/offline_mbrl_plan/offline_mbrl_for_auv_navigation_report.md) | 基于 AUVHamNODE 的流场中 AUV 高效导航:Offline Model-Based RL 方案分析报告 | DEPRECATED 2026-05-08 |

## docs/superpowers/ — 早期 plan/spec 存档（5）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`docs/superpowers/plans/2026-04-04-improved-sac.md`](../docs/superpowers/plans/2026-04-04-improved-sac.md) | Improved SAC (LayerNorm + DroQ + Asymmetric Critic) — Implementation Plan | — |
| [`docs/superpowers/plans/2026-04-04-multi-cylinder-wake.md`](../docs/superpowers/plans/2026-04-04-multi-cylinder-wake.md) | Multi-Cylinder Wake Field Generation — Implementation Plan | — |
| [`docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`](../docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md) | ReBRAC Broad Validation Implementation Plan | SUPERSEDED 2026-05-18 |
| [`docs/superpowers/specs/2026-04-04-multi-cylinder-improved-sac-design.md`](../docs/superpowers/specs/2026-04-04-multi-cylinder-improved-sac-design.md) | Design: Multi-Cylinder Wake Fields + Improved SAC | — |
| [`docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`](../docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md) | ReBRAC 广验实验设计：跨数据质量 / 传感器 / 任务三轴的 Probe-then-Deepen | SUPERSEDED 2026-05-18 |

## docs/assets/ — 图注（4）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`docs/assets/auv_probe_layouts_caption.md`](../docs/assets/auv_probe_layouts_caption.md) | auv_probe_layouts_caption | — |
| [`docs/assets/fql_succession_p2/test_eval_summary.md`](../docs/assets/fql_succession_p2/test_eval_summary.md) | FQL Succession P2 — test-eval primary metric | — |
| [`docs/assets/wake_profile_real_flow_caption.md`](../docs/assets/wake_profile_real_flow_caption.md) | wake_profile_real_flow_caption | — |
| [`docs/assets/wake_profile_schematics_caption.md`](../docs/assets/wake_profile_schematics_caption.md) | wake_profile_schematics_caption | — |

## paper/thesis_ch5/ — 第 5 章活跃工作文件（5）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`paper/thesis_ch5/data_integrity_impact_assessment.md`](../paper/thesis_ch5/data_integrity_impact_assessment.md) | 数据完整性两条问题对第 5 章的波及面评估 | — |
| [`paper/thesis_ch5/data_integrity_impact_assessment_review.md`](../paper/thesis_ch5/data_integrity_impact_assessment_review.md) | 波及面评估的独立复核（2026-08-16） | — |
| [`paper/thesis_ch5/next_session_prompt.md`](../paper/thesis_ch5/next_session_prompt.md) | 第 5 章 — 轮次入口（**当前轮：数据完整性整改批的独立复核**） | CLOSED |
| [`paper/thesis_ch5/prompt_playbook.md`](../paper/thesis_ch5/prompt_playbook.md) | 第 5 章写作 Prompt Playbook（总纲） | — |
| [`paper/thesis_ch5/status.md`](../paper/thesis_ch5/status.md) | 第 5 章写作状态账本（status ledger） | — |

## paper/thesis_ch5/notes/ — 第 5 章过程存档（21）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`paper/thesis_ch5/notes/academic_paper_reviewer_prompt.md`](../paper/thesis_ch5/notes/academic_paper_reviewer_prompt.md) | §5.1/§5.2 审查 — 新对话启动 Prompt（academic-paper-reviewer） | — |
| [`paper/thesis_ch5/notes/ars_full_review_prompt.md`](../paper/thesis_ch5/notes/ars_full_review_prompt.md) | §5.1/§5.2 审查 — 新对话启动 Prompt（ars-full） | — |
| [`paper/thesis_ch5/notes/chapter_acceptance_review_5_1_5_6_findings.md`](../paper/thesis_ch5/notes/chapter_acceptance_review_5_1_5_6_findings.md) | 第 5 章 §5.1–§5.6 章级验收复审 · 审稿意见与落地核查存档 | — |
| [`paper/thesis_ch5/notes/chapter_acceptance_review_prompt.md`](../paper/thesis_ch5/notes/chapter_acceptance_review_prompt.md) | §5.1–§5.6 章级整体验收复审 — 新对话启动 Prompt | — |
| [`paper/thesis_ch5/notes/chapter_full_review_findings.md`](../paper/thesis_ch5/notes/chapter_full_review_findings.md) | 第 5 章 全章整体复审 findings（2026-07-21） | — |
| [`paper/thesis_ch5/notes/chapter_full_review_prompt.md`](../paper/thesis_ch5/notes/chapter_full_review_prompt.md) | 第 5 章全章整体复审（十节 · 58 页）— 新对话启动 Prompt | — |
| [`paper/thesis_ch5/notes/data_integrity_batch_review_findings.md`](../paper/thesis_ch5/notes/data_integrity_batch_review_findings.md) | 数据完整性整改批 —— 独立对抗复审 findings（2026-08-16） | — |
| [`paper/thesis_ch5/notes/draft_5.1_outline.md`](../paper/thesis_ch5/notes/draft_5.1_outline.md) | §5.1 章级独立引言 —— Markdown 段落级 outline | — |
| [`paper/thesis_ch5/notes/section_5_10_discussion_writing_plan.md`](../paper/thesis_ch5/notes/section_5_10_discussion_writing_plan.md) | §5.10 综合讨论与本章小结 写作方案 — 已拍板执行版 | — |
| [`paper/thesis_ch5/notes/section_5_10_review_findings.md`](../paper/thesis_ch5/notes/section_5_10_review_findings.md) | §5.10 综合讨论与本章小结（discussion.tex rev.1）独立对抗复审 findings | — |
| [`paper/thesis_ch5/notes/section_5_3_followup_review_notes.md`](../paper/thesis_ch5/notes/section_5_3_followup_review_notes.md) | Section 5.3 Follow-up Review Notes | — |
| [`paper/thesis_ch5/notes/section_5_3_review_prompt.md`](../paper/thesis_ch5/notes/section_5_3_review_prompt.md) | §5.3 节独立审查 Prompt（高水平博士论文标准 · 冷启动自足） | — |
| [`paper/thesis_ch5/notes/section_5_4_methodology_review.md`](../paper/thesis_ch5/notes/section_5_4_methodology_review.md) | 第 5 章十节结构与 §5.4 方法节审查记录 | — |
| [`paper/thesis_ch5/notes/section_5_4_paragraph_blueprint.md`](../paper/thesis_ch5/notes/section_5_4_paragraph_blueprint.md) | §5.4“强化学习方法与算法框架”段落级写作蓝图 | — |
| [`paper/thesis_ch5/notes/section_5_5_restructure_prompt.md`](../paper/thesis_ch5/notes/section_5_5_restructure_prompt.md) | 任务：系统性复审并重构博士论文第 5 章 §5.5（在线情形下的可学习性与信息瓶颈） | — |
| [`paper/thesis_ch5/notes/section_5_6_review_prompt.md`](../paper/thesis_ch5/notes/section_5_6_review_prompt.md) | 任务：按 §5.5 的标准，系统性复审博士论文第 5 章 §5.6（离线基线：行为约束方法的两个瓶颈） | — |
| [`paper/thesis_ch5/notes/section_5_7_rebrac_writing_plan.md`](../paper/thesis_ch5/notes/section_5_7_rebrac_writing_plan.md) | §5.7 离线主线（ReBRAC-Q）写作方案 — 已拍板执行版 | — |
| [`paper/thesis_ch5/notes/section_5_7_review_prompt.md`](../paper/thesis_ch5/notes/section_5_7_review_prompt.md) | 任务：独立对抗复审博士论文第 5 章 §5.7（离线主线：双侧行为约束下的可部署性能） | — |
| [`paper/thesis_ch5/notes/section_5_7_writing_organization_review_prompt.md`](../paper/thesis_ch5/notes/section_5_7_writing_organization_review_prompt.md) | §5.7 写作组织与展开度独立评估 prompt（2026-07-07 归档） | — |
| [`paper/thesis_ch5/notes/section_5_8_review_findings.md`](../paper/thesis_ch5/notes/section_5_8_review_findings.md) | §5.8 泛化边界（boundary.tex rev.1）独立对抗复审 findings | — |
| [`paper/thesis_ch5/notes/section_5_9_review_findings.md`](../paper/thesis_ch5/notes/section_5_9_review_findings.md) | §5.9 算法比较（algo_compare.tex rev.1）独立对抗复审 findings | — |

## paper/ — 其它（写作规范 / 工具说明 / 已归档论文）（4）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`paper/archive/rebrac_standalone/outline.md`](../paper/archive/rebrac_standalone/outline.md) | ReBRAC AUV Wake Navigation — 论文写作大纲 | SUPERSEDED 2026-06-02 |
| [`paper/archive/rebrac_standalone/progress.md`](../paper/archive/rebrac_standalone/progress.md) | Paper Writing Progress & Plan | — |
| [`paper/thesis_ch5/tools/README.md`](../paper/thesis_ch5/tools/README.md) | `paper/thesis_ch5/tools/` — 第 5 章机械门槛复核工具 | — |
| [`paper/thesis_chapter_outline.md`](../paper/thesis_chapter_outline.md) | 博士论文章节写作 Spec：Deployable-Sensor Offline RL for AUV Wake Navigation | — |

## .claude/ — 项目工具定义（须留在原位，Claude Code 按路径加载）（5）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`.claude/agents/docs-pointer-validator.md`](../.claude/agents/docs-pointer-validator.md) | Docs pointer validator | — |
| [`.claude/agents/sweep-script-reviewer.md`](../.claude/agents/sweep-script-reviewer.md) | Sweep-script reviewer | — |
| [`.claude/skills/experiment-summary/SKILL.md`](../.claude/skills/experiment-summary/SKILL.md) | experiment-summary | — |
| [`.claude/skills/notebook-from-template/SKILL.md`](../.claude/skills/notebook-from-template/SKILL.md) | notebook-from-template | — |
| [`.claude/skills/rl-v2-commands/SKILL.md`](../.claude/skills/rl-v2-commands/SKILL.md) | rl_v2 command reference | — |

## 其它（4）

| 文档 | 标题 | 自我标注状态 |
|---|---|---|
| [`benchmarks/README.md`](../benchmarks/README.md) | Standard Benchmarks | — |
| [`phnode_full_oc_clean/README.md`](../phnode_full_oc_clean/README.md) | phnode_full — OC clean pretrained checkpoint (seed 45) | — |
| [`phnode_full_oc_clean/reference_simulator/README.md`](../phnode_full_oc_clean/reference_simulator/README.md) | REMUS 100 reference simulator | — |
| [`reference/flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md) | 自然生物利用流场高效航行与强化学习复现研究综述 | — |
