# 第 5 章续写 — 本轮入口 Prompt（§5.8 独立对抗复审 · 轻流程试点检验轮）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与本节相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> 本轮双重身份：① §5.8 rev.1 的**独立对抗复审**（成稿于轻流程试点：论点骨架直落、免硬停等确认）；② **轻流程门控检验**——复审结束后统计 CRIT/HIGH 数量，与 §5.6/§5.7 独立复审的同级数量对比：若明显上升，判定「方案→落地→独立复审」三轮重流程的门控确属 load-bearing，§5.9 起退回重流程；否则 §5.9 续用轻流程。

---

你是 AUV 运动规划与强化学习交叉领域的资深专家，同时是严格的中文博士学位论文外审人。以**挑错者**身份对 `sections/boundary.tex`（§5.8 泛化边界，rev.1）做独立对抗复审：假定成稿有错，逐条找出来。**不参考成稿自述与头注作为事实依据**——一切数字与 claim 以 ground truth 文档实时回查为准。

## 复审对象与 ground truth

- 成稿：`sections/boundary.tex`（rev.1，含 2 表：tab:ch5_boundary_cells / tab:ch5_boundary_ceiling）。
- **数字唯一权威**：`docs/rebrac_broad_validation_v2_report.md` §2–§5 + §4.5；在线参照（0.26±0.15 / 0.88±0.04 / 交互步数 / 种子数）以 `sections/online.tex` §5.5.3 瓶颈表与 §5.5.m 为准；主线锚点 0.902±0.021 以 `sections/rebrac.tex` 刊值为准。
- spec：`paper/thesis_chapter_outline.md` §2 §5.8 块（claim 边界红线）、§0.4、§0.5.9、§0.5.10。

## 复审维度（按 playbook §2 惯例，对抗 + 系统性合并一轮）

1. **数字忠实性**：正文与两表每个数字逐一回查 ground truth（含区间端点、n、单位、种子归属、终止构成）。
2. **claim 边界**：✅ 只可「特权价值网络不能挽救／排除价值侧成因／加固策略侧解释」；❌ 任何滑向「proven actor-incapable / 信息论不可能」的措辞（含隐含式）。两读法（表征不足 vs 机制未转化）须完整在场。
3. **先决裁决执行**：n=2 极重 caveat 是否足重（显式 n=2 + rule-of-three 上界 + 正文登记补种子）；在线特权消融是否确未并列（仅散文指回 §5.5、无数字、线别限定）。
4. **衔接一致性**：§5.7.5 伏笔承接、§5.5.5 k=4 scope 承接（对照须用 0.26 非 legacy 10%）、§5.3.6 零计数口径归属、不嫁接回 §5.7、四态收束是否越位（应留 §5.10）。
5. **术语与语体**：§0.5.10 对照表（策略网络/价值网络/固定评估集/工况/上限…）；§0.5.9 报告体 marker；每段 3–4 句；负面结果 finding 语态非 limitation 语态。
6. **统计呈现**：预登记门槛表述与 report 原文一致；配对诊断数字与解读（1.15 SEM / 中位数 −0.016 / 46% 单回合 / 17/30）无过度引申。

## 输出

- findings 按 CRIT / HIGH / MED / LOW 分级，逐条给出：位置（行/段）、问题、ground truth 依据、建议改法；存档为 `section_5_8_review_findings.md`。
- 修订实质项落 `boundary.tex` 头注 rev 块；`status.md` §5.8 行更新。
- **轻流程判定段**：CRIT/HIGH 计数 vs §5.6/§5.7 复审基线（各存档文档可查），给出 §5.9 流程建议，交用户拍板。

## 硬护栏（不可谈判）

- 数字一律实时回查 ground truth；禁凭记忆、禁以成稿自述为二手源。
- 编译验证：`cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`，0 undefined ref / 0 undefined citation / bibtex 0 warning；成功后 `latexmk -c`。
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
