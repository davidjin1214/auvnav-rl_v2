# 第 5 章续写 — 本轮入口 Prompt（全章收尾统稿轮 · 候选）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **流程状态**：§5.10 重流程四步已全部闭环（rev.2 复审落地 2026-07-08，0 CRIT / 3 HIGH / 7 MED 全修；门控 CRIT+HIGH=3 < §5.9 基线 4，复审侧建议不追加复审轮）——**全章十节齐**。
> **先决裁决（本轮动笔前向用户确认）**：① 用户是否采纳 §5.10 复审呈报（`section_5_10_review_findings.md` §6）、不追加一轮 §5.10 复审；② 是否即启动收尾统稿轮。若 ① 被否，本文件作废、按用户指示重写为 §5.10 追加复审轮入口。

---

本轮任务（获确认后执行）：**全章收尾统稿**——十节已各自闭环，本轮做且只做跨节层面的批量收尾，不重开任何节的论证与数字。三个工作包：

## 包 A：LOW 批量清理

各复审存档登记未修的 LOW 项集中落地（逐条回原 findings 核对原文与改法，不扩大改动面）：

1. `section_5_8_review_findings.md` §4 LOW×6（表内简称不一 / 「在线 SAC」行名 / 对照口径列举 / ≥0.40 强正面区间省略 / 数据集行指称歧义 / 「定位在策略一侧」措辞）；
2. `section_5_9_review_findings.md` LOW 残留 ×2（σ=0.5 事后加噪歧义已由 §5.9.m 澄清、图 (a) 参考线标记描述）；
3. `section_5_10_review_findings.md` §4 LOW×3（「仿真器方便提供」语感 / 表 5.10-2 态一行名「标准 SAC」精确化 / **机体↔船体章级术语统一**——以 `setup.tex` 定义之家为准裁定统一方向，涉 online/td3bc/methodology 与 setup 两族用词，改前先列出全部出现处呈用户过目）;
4. 章级验收 findings §3.2 的零散项（括注清理 / refs.bib 未引条目处置等）。

## 包 B：浮动漂移 F·M2

全章拼装后的图表浮动统一处理（§5.7 rev.2 起登记延后项）：表 5.10/图 5.12 漂移 1–2 页、半空白页、各节 `[t]`/`[htbp]` 与 `\clearpage` 布点复查。只动浮动参数与布点，**零正文改动**；处理后逐图核对图文同页/邻页可达。

## 包 C：图表编号与术语终检

1. 图表编号顺序与正文首引顺序一致性（全章过一遍）；
2. 术语终检 grep（spec §0.5.10 全表 + 各轮红线并集：teacher / 裸 ReBRAC / regime / 导航 / 「」与直引号 / 其一其二 / 信息论 / 不可能 / seed 44 具名 / 报告体 marker），十节全量；
3. 交叉引用抽查：跨节 `\S\ref` 的指向语义抽核（重点：§5.10 两表锚、§5.8↔§5.5 对照、§5.9↔§5.7 线别限定句）。

## 红线（本轮特有）

- 十节 `.tex` 均为 locked：只 Edit 定点微修，**零数字改动、零论证改动**；任何超出 LOW 清单的发现（新的事实性问题）**先呈报再动**；
- 包 A 第 3 条术语统一属跨节联动，改前呈用户确认统一方向；
- `paper/sections/`（paper 1 原稿）只读。

## 按序必读

1. `status.md`（节状态与登记项表）；
2. 三份复审存档的 LOW 段：`section_5_8_review_findings.md` §4、`section_5_9_review_findings.md` LOW 段、`section_5_10_review_findings.md` §4；
3. `chapter_acceptance_review_5_1_5_6_findings.md` §3.2/§5（零散项与登记项源头）；
4. spec `paper/thesis_chapter_outline.md` §0.5.9/§0.5.10（语体与术语全表）。

## 落地工序

1. 包 A → 包 C → 包 B 顺序执行（先定字再定版）；
2. 编译：`cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`；0 undefined / bibtex 0 warning / 无新增 Overfull>10pt；成功后 `latexmk -c`；
3. `status.md` 更新（登记项表逐条勾销）；本文件重写为下一轮入口（候选：定稿前补充验证事项盘点——临界单元补种子配对复核、临界传感 3-seed 补录，见 `status.md` 登记项表）。

## 输出与工程

- 产出：十节 `.tex` 的定点微修 + `status.md` 更新；无新文档（LOW 勾销记在各 findings 原条目后追注「已清，收尾统稿轮」）；
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
