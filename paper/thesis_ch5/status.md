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
| §5.8 泛化边界 | `sections/boundary.tex` | ✅ 三轮闭环（rev.1 轻流程试点落地 / rev.2 独立对抗复审：0 CRIT + 1 HIGH + 5 MED 全落地 / rev.3 补种子落地 2026-07-12：三单元 {42, 0, 43}，表值/计数/上界/登记语全更新 + §5.8.1 锚点对照段论证级改写——「两种子同向退化」撤销，改按种子间波动解读；§5.8.4 在线参照 0.26 属道 1 裁决范围未动）；**轻流程门控判定：CRIT/HIGH=1 ≪ §5.6 基线 4、§5.7 基线 5，未触发退回条件；用户 2026-07-07 拍板采纳，轻流程转正** | `section_5_8_review_findings.md`（H1 已闭环）；先决裁决 2026-07-07：N2′ 以极重 caveat 呈现、在线特权消融不并列（散文指回 §5.5，四态收束留 §5.10）；rev.3 后 caveat 为 n=3 + 种子组与预登记不重合披露 |
| §5.9 算法对比 FQL | `sections/algo_compare.tex` | ✅ 两轮闭环（rev.1 轻流程起草 / rev.2 独立对抗复审：0 CRIT + 4 HIGH + 8 MED 全落地，三表两图数字零漂移、红线子集零踩线，4 HIGH 均为解释性散文层事实错误——预登记归属、机制前提方向、跨节指认、综合句自相矛盾）；轻流程门控：CRIT+HIGH=4 与 §5.6 基线 4 持平（§5.7 基线 5、§5.8 试点 1），触发 ≥4 呈报条件，**用户 2026-07-07 拍板采纳①——§5.10 恢复重流程**（findings §6 已录裁决） | `section_5_9_review_findings.md`；头注 rev.2 块 |
| §5.10 统一讨论 + 本章小结 | `sections/discussion.tex` | ✅ **重流程四步闭环**（方案获批 → rev.1 落地 → rev.2 独立对抗复审 2026-07-08：0 CRIT + 3 HIGH + 7 MED 全落地 + LOW 顺手 2 条，两表零漂移、收束引用零走样、伏笔兑现零扩权、红线零踩线；3 HIGH 均为收束层 scope 措辞——五重限定假全称、四态综合句分支越权、部署含义数据采集扩权）；**门控：CRIT+HIGH=3 < §5.9 基线 4，未触发追加轮条件，复审侧建议不追加，最终由用户裁决**；残留 LOW×3（含机体↔船体跨节术语分歧）记录 findings §4 留收尾统稿轮。**全章十节齐** | `section_5_10_discussion_writing_plan.md`；`section_5_10_review_findings.md`（复审存档 + 门控呈报 §6） |

## 工程状态

- **收尾统稿轮已闭环（2026-07-08）**：包 A（三份复审 LOW 全清 + 章级零散 LOW：重复英文括注 ×6、refs.bib 未引 ×3、机体↔船体统一为「机体」）＋包 C（术语终检 grep 十节零违规、跨节引用抽核通过、图表编号与首引序全章对齐）＋包 B（F·M2 浮动漂移根治：fig 5.3 `[t]`→`[tp]` 解除 figure 队列堵塞、§5.5 六图回归各小节、rebrac 三表重排、全部浮动距首引 0–2 页）。零数字/零论证改动。
- 编译：58 页通过，0 undefined ref / 0 undefined citation / 0 multiply-defined / bibtex 0 warning / 无 Overfull>10pt。`main.tex` 已 `\input` §5.1–§5.10 全部十节。
- ⚠ Windows 副机 MiKTeX 若报「siunitx: expl3 too old」：`miktex packages update l3kernel l3backend l3packages` + `initexmf --dump=xelatex`（2026-07-05 修复过一次）。
- 已定稿节 `.tex` 视为 locked：只 Edit 微修、不 Write 覆盖；`paper/sections/`（paper 1 素材）只读勿改。

## 待决与登记项（源：`chapter_acceptance_review_5_1_5_6_findings.md` §5 + spec §8）

| 项 | 时点 | 出处 |
|---|---|---|
| ~~N2′ 补种子 vs 极重 caveat 呈现~~ | ✅ 已决 2026-07-07：极重 caveat 呈现（§5.8.5 专段 + 表注 n=2 + 零计数上界） | spec §2 §5.8 答辩风险登记 |
| ~~online AsymCritic 是否与 §5.8 asym ablation 并列呈现~~ | ✅ 已决 2026-07-07：不并列，散文指回 §5.5（线别限定），四态收束留 §5.10 | spec §8 #4 |
| ~~online↔offline k=4 对照句~~ | ✅ 已落地（§5.8.4 对照用 0.26±0.15，非 legacy 10%；k=4/k=12 不对称承接 §5.5.5） | 承接 §5.5.5 scope 保险 |
| ~~临界单元 + 非对称消融补种子配对复核~~ | ✅ 全流程闭环（2026-07-12）：Colab 执行 → Drive raw 回查判读（三门槛全过）→ 呈报（N0 seed 43 越锚点、同向叙事失效、清单不自洽）→ 用户裁决 (a) 扩权落地——v2 report 三种子增补（含 §2.4 论证级改写）+ `boundary.tex` rev.3 + 编译零警告 + spec §2 §5.8 答辩风险降级 + findings H1 追注闭环 | plan 附录 B.2；v2 report 2026-07-12 Addendum |
| ~~临界传感 3-seed 补录 `docs/arrival_v2_experiment_report.md`~~ | ✅ 全流程闭环（2026-07-19）：用户批附录 C 四点（C.1 按用户修正——旧转录值定性为**引用错误、完全作废（占位数字）**，不作分批论证、不比对新旧差异；C.2 措辞照批；C.3 ⑧ 定点复审加；C.4 六份补跑 raw 从 Drive 溯源同步回本机 canonical 树）。落地：§7.10 重写（✅ 9/9 实核）→ 两图重绘（CRIT_SEEDS/CRIT_K4_S0 新值）→ `online.tex` rev.7（§5.5.2/caption/§5.5.3 补方差收缩 0.39→0.22→0.04/瓶颈表/§5.5.4 caption/§5.5.5）→ `boundary.tex` rev.4（§5.8.2「稳定支撑」限定 + §5.8.4 0.46±0.39）→ spec 五处同步 → latexmk 58 页零警告 + `-c` + 残留 grep 全清。新 ground truth：s0 0.46±0.39、s1 0.90±0.00、s2 0.87±0.03、gap 44pp；k=8/k=12 与 manifest floor 27/30、特权消融 0.267、`discussion.tex` 经核零波及 | findings E·M1（✅ 已勾销）；plan 附录 B.1（呈报）/ C（方案）；report §7.10（ground truth） |
| ~~图浮动漂移 + 半空白页~~ | ✅ 已清 2026-07-08（收尾统稿轮包 B，根因与改法见 findings F·M2 追注） | findings F·M2 |
| ~~零散 LOW（括注清理 / refs.bib 残留 / 机体↔船体术语章级统一等）~~ | ✅ 已清 2026-07-08（收尾统稿轮包 A，逐条追注在各 findings 原条目后） | findings §3.2；`section_5_8/5_9/5_10_review_findings.md` LOW 段 |

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
| 特权观测载体术语统一为「机体」（「船体」废止；「艇体」备选未采） | 2026-07-08 用户拍板；setup.tex §5.3.2/§5.3.3 定义之家 + spec §0.5.10 首现规范已同步 |
