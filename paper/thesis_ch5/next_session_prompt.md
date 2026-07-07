# 第 5 章续写 — 本轮入口 Prompt（定稿前补充验证事项盘点轮 · 候选）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **流程状态**：收尾统稿轮已闭环（2026-07-08）——包 A LOW 全清（三份复审 LOW + 章级零散项 + 机体↔船体统一为「机体」，用户拍板）、包 C 终检零违规、包 B 浮动漂移根治（fig 5.3 `[t]` 堵队列根因修复 + rebrac 三表重排 + 编号与首引序全章对齐）。58 页 / 0 undefined / bibtex 0 warning。**十节齐 + 统稿毕**：章文本层面已无登记中的未清项。
> **先决裁决（本轮动笔前向用户确认）**：是否即启动定稿前补充验证事项的盘点与执行。若用户另有优先事项（如全章通读、送审版式、与博论其余章节拼装），本文件作废重写。

---

本轮任务（获确认后执行）：**定稿前补充验证事项盘点与分道执行**——`status.md` 登记项表尚余两项非阻塞事项，性质不同、分两道处理：

## 道 1：临界传感 3-seed 补录（纯文档，可直接落地）

- **事项**：§5.5.2 临界传感比较（s0/s1/s2 @ k=4，云端三种子 seed 0/7/42，`online.tex` rev.5 已确认数字）尚未录入 ground truth 报告 `docs/arrival_v2_experiment_report.md`，使 §5.5.2 的引用链闭环。
- **来源**：章级验收 findings E·M1（答辩前，非阻塞）。
- **做法**：按 online.tex rev.5 头注所录 per-seed 数据（s0=[.4/.267/.1]→0.26±0.15、s1=[.9/.8/.9]→0.87±0.06、s2=[.8/.733/.733]→0.76±0.04）与云端结果文件核对后补录报告；**数字以云端 final_eval 为准实时回查，不从头注誊抄**（头注只作索引）。

## 道 2：临界单元 + 非对称消融补种子配对复核（需实验，先呈方案再跑）

- **事项**：§5.8 两单元与消融现为两种子（[42, 0]），所依计划预登记为三种子（[42, 43, 44]，v2 plan §6.2）；§5.8.m 已如实登记缺额并入定稿前补充验证事项。
- **来源**：spec §2 §5.8 答辩风险登记；report §6.3；`section_5_8_review_findings.md` H1。
- **做法**：先盘点出最小补种子矩阵（候选：N0 / N2′ / 消融各补 seed 43，共 3 个训练单元 + 终检；Colab L4 预算与 notebook 计划一并列出），**呈用户批准后**再起 notebook；跑完按 §5.8 预登记门槛判读，若与两种子结论一致则只改 §5.8.m 登记语与表注 n（零论证改动），若不一致按「新的事实性问题」先呈报。
- **红线**：在补种子结果落地前，§5.8 正文的两种子 caveat 与引用限定一字不动。

## 红线（本轮特有）

- 十节 `.tex` 均为 locked：道 1 不触碰 `.tex`；道 2 只在结果闭环后按上述范围定点微修；
- 实验一律 Colab（本机不跑训练，CLAUDE.md 工作流）；notebook 用 `!python` shell magic、`[skip]` 判定用 `agent_final.pt`；
- ground truth 更新遵循既有分工：broad_validation v2 report 为 §5.8 数字唯一权威、arrival_v2 report 为在线数字权威，两者互不越界。

## 按序必读

1. `status.md`（登记项表两残项）；
2. `docs/rebrac_broad_validation_v2_plan.md` §6.2（预登记种子组原文）+ `docs/rebrac_broad_validation_v2_report.md` 头部 seed-count caveat 与 §6；
3. `sections/online.tex` 头注 rev.5 块 + `docs/arrival_v2_experiment_report.md` 现状（§7.9 临界瓶颈段落）；
4. `chapter_acceptance_review_5_1_5_6_findings.md` E·M1 原条目。

## 输出与工程

- 道 1 产出：`docs/arrival_v2_experiment_report.md` 增补一节 + E·M1 追注勾销 + `status.md` 勾销；
- 道 2 产出（本轮内）：补种子矩阵方案（呈批），获批后另起执行；
- 编译不涉及（除非道 2 闭环后微修 `.tex`，则按标准 latexmk 流程 + `latexmk -c`）；
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
