# 第 5 章写作状态账本（status ledger）

> 定位：章写作状态的**唯一真相源**——每节一行、零结果数字、零历史叙事。逐轮修订史见 git log（commit message 即变更日志）、各 `.tex` 头注 rev 块、各复审存档文档；当轮任务入口见 `next_session_prompt.md`（**每轮重写**，只含当轮所需）。
> 体例（2026-07-07 拆分自旧 next_session_prompt）：本文件只登记「现在什么状态、还欠什么、决策在哪」；不复述修订内容、不内嵌结果数字——数字一律回查 ground truth docs，本文件不充当二手数字源。

## 节状态

| 节 | 文件 | 状态 | 方案/复审存档 |
|---|---|---|---|
| §5.1 引言 | `sections/intro.tex` | ✅ 闭环（含章级验收 `238f11b`） | `chapter_acceptance_review_5_1_5_6_findings.md` |
| §5.2 研究背景与定位 | `sections/related_work.tex` | ✅ 闭环（含章级验收） | 同上 |
| §5.3 问题设定与评估协议 | `sections/setup.tex` | ✅ 闭环；**协议之家**（终检/周期命名、rule-of-three 口径在 §5.3.6） | `section_5_3_review_prompt.md` |
| §5.4 方法与算法框架 | `sections/methodology.tex` | ✅ 闭环；**算法之家**（公式口径权威文本、检查点选择规则在 §5.4.m） | `section_5_4_methodology_review.md` |
| §5.5 在线可学习性与信息瓶颈 | `sections/online.tex` | ✅ 闭环；§5.5.5 含 k=4 scope 保险（§5.8 须承接对照） | `section_5_5_restructure_prompt.md` |
| §5.6 离线基线 TD3+BC | `sections/td3bc.tex` | ✅ 闭环 | `section_5_6_review_prompt.md` |
| §5.7 离线主线 ReBRAC-Q | `sections/rebrac.tex` | ✅ 三轮闭环（rev.1 落地 `e791d2d` / rev.2 忠实性复审 `951549d` / rev.3 组织与展开度 `5034b0c`）；修订实质项全录头注 rev 块 | `section_5_7_rebrac_writing_plan.md`、`section_5_7_review_prompt.md`、`section_5_7_writing_organization_review_prompt.md` |
| §5.8 泛化边界 | `sections/boundary.tex` | ✅ rev.1 落地（**轻流程试点**：论点骨架直落、免硬停等确认）；独立复审待做（下一轮，兼判轻流程门控是否 load-bearing） | spec §2 §5.8 块；先决裁决 2026-07-07：N2′ 以极重 caveat 呈现（n=2 显式 + rule-of-three 上界 + 正文登记补种子）、在线特权消融不并列（散文指回 §5.5，四态收束留 §5.10） |
| §5.9 算法对比 FQL | `sections/algo_compare.tex`（待建） | ⏳ 待起草 | spec §2 §5.9 块 |
| §5.10 统一讨论 + 本章小结 | `sections/discussion.tex`（待建） | ⏳ 待起草；携带：贡献类型正面声明（findings G·H-2）、生物回扣一次（4 护栏，spec §8 #10） | spec §2 §5.10 块 |

## 工程状态

- 编译：47 页通过，0 undefined ref / 0 undefined citation / bibtex 0 warning（唯一 Overfull 3.1pt 系 methodology 既有公式项）。`main.tex` 已 `\input` §5.1–§5.8 八节，§5.9–§5.10 注释占位。
- ⚠ Windows 副机 MiKTeX 若报「siunitx: expl3 too old」：`miktex packages update l3kernel l3backend l3packages` + `initexmf --dump=xelatex`（2026-07-05 修复过一次）。
- 已定稿节 `.tex` 视为 locked：只 Edit 微修、不 Write 覆盖；`paper/sections/`（paper 1 素材）只读勿改。

## 待决与登记项（源：`chapter_acceptance_review_5_1_5_6_findings.md` §5 + spec §8）

| 项 | 时点 | 出处 |
|---|---|---|
| ~~N2′ 补种子 vs 极重 caveat 呈现~~ | ✅ 已决 2026-07-07：极重 caveat 呈现（§5.8.5 专段 + 表注 n=2 + 零计数上界） | spec §2 §5.8 答辩风险登记 |
| ~~online AsymCritic 是否与 §5.8 asym ablation 并列呈现~~ | ✅ 已决 2026-07-07：不并列，散文指回 §5.5（线别限定），四态收束留 §5.10 | spec §8 #4 |
| ~~online↔offline k=4 对照句~~ | ✅ 已落地（§5.8.4 对照用 0.26±0.15，非 legacy 10%；k=4/k=12 不对称承接 §5.5.5） | 承接 §5.5.5 scope 保险 |
| 临界单元 + 非对称消融补种子配对复核 | 章定稿前 / 答辩前，非阻塞（§5.8.5 正文已登记该限定） | spec §2 §5.8 答辩风险登记；report §6.3 |
| 临界传感 3-seed 补录 `docs/arrival_v2_experiment_report.md` | 答辩前，非阻塞 | findings E·M1 |
| 图浮动漂移 + 半空白页 | 全章拼装后统一处理 | findings F·M2 |
| 零散 LOW（括注清理 / refs.bib 残留等） | 收尾统稿一轮批量 | findings §3.2 |

## 已锁决策（只列指针，内容以权威所在为准、不在此复述）

| 决策 | 权威所在 |
|---|---|
| 中心命题（全章原话照用） | spec `paper/thesis_chapter_outline.md` §0.1 |
| 机制归因红线（5 条）+ 各节 claim 边界 | spec §0.4 + §2 各节块 |
| 语体（不反向宣告/不自我贬抑）与术语对照 | spec §0.5.9 / §0.5.10 |
| NO APPENDIX、内容优先于篇幅 | spec §0.5.5 / §0.5.8 |
| 生物只作引子；例外 = §5.10 一次回扣（4 护栏） | spec §8 #10（resolved 2026-07-03） |
| 公式口径（结果节引用、勿改） | `sections/methodology.tex` 正文本身 |
| §5.k.m 归属（§5.3 协议之家 / §5.4 算法之家 / 结果节只写本节增量） | findings + §5.5/§5.6 先例 |
| prompt 编写方法（会话序列 / 十要素 / 反模式） | `prompt_playbook.md` |
| git commit 授权（自行 commit，`docs:` 前缀 + Co-Authored-By trailer） | 2026-07-05 用户授权 |
