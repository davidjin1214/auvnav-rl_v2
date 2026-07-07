# 第 5 章续写 — 本轮入口 Prompt（§5.10 统一讨论 · 重流程落地轮）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与本节相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **流程状态**：§5.10 走重流程（方案 → 硬停待批 → 落地 → 独立对抗复审）。方案轮已完成并于 **2026-07-07 获用户批准（"同意推荐"，六项裁决全按推荐锁定）**。本轮 = **落地轮**；落地完成后另起独立对抗复审轮（本文件届时再重写为复审轮入口）。

---

本次对话唯一任务：**按已拍板方案 `paper/thesis_ch5/section_5_10_discussion_writing_plan.md` 起草 `sections/discussion.tex`（§5.10 综合讨论与本章小结），完成编译与自查**。**勿重开方案讨论**——结构、图表、措辞草案方向均已锁定；若落地中发现方案层硬伤（如伏笔句与兑现句无法对齐），先呈报再动，不静默改方案。

【已锁六项裁决（2026-07-07，均按推荐）】
1. 结构 = 开篇段（不编号）+ 5 正文子节，**不设 §5.10.m**（方案 A）；spec 指定给 .m 的三资产由正文承载；
2. 节标题 =「综合讨论与本章小结」，label `sec:ch5_discussion`，文件 `sections/discussion.tex`；
3. 图表 = **两表零图**（表 5.10-1 β1 跨线协议四轴对照、表 5.10-2 特权观测四态收束），两表**零新数字**（定性差异 + 节内指针）；
4. sim2real 与 limitations 合并为 §5.10.4「对部署的含义与本章结论的适用范围」；
5. 生物回扣一句：置于 §5.10.5 末段收束层，按方案 §2.5 草案方向定稿（四护栏逐条对照）；
6. 贡献类型正面声明：置于 §5.10.5 第二段，按方案 §2.5 草案方向定稿（三类型；正向声明、禁反向宣告）。

## 按序必读

1. **方案全文** `section_5_10_discussion_writing_plan.md`：§1 节定位与三处伏笔、§2 逐子节论点骨架（含小结两段措辞草案）、§3 两表定案、**§4 claim→锚点映射表 24 条（对账基准，每段起草后逐条核）**、§5 红线与术语自查表、§6 落地工序、§7 勿改动项；
2. spec `paper/thesis_chapter_outline.md`：§0.1 中心命题原话 + §0.3 takeaway + §0.4 红线全五条 + §0.5.3（章末小结体裁）+ §0.5.9/§0.5.10 + §3 缝合点表 + §8 #10 生物回扣四护栏；
3. 九节收束接口（**收束引用逐字对齐，起草时开原文对照**）：`intro.tex` ¶6/¶7（统一句与中心命题回归的同句锚）、`online.tex` §5.5.3/§5.5.5、`td3bc.tex` 瓶颈之二与动机小节、`rebrac.tex` §5.7.5 + `tab:ch5_rebrac_screen`、`boundary.tex` §5.8.3/§5.8.5、`algo_compare.tex` §5.9.2/§5.9.5；
4. ground truth 实时回查（凡表格或正文涉及事实归属处）：`docs/rebrac_line_overview.md` §6、`docs/rebrac_mainline_review.md` §2.2.6、`docs/fql_succession_p2_results.md` §7/§8.1、`docs/rebrac_broad_validation_v2_report.md` §4.5/§6.1、`docs/arrival_v2_sac_collector_design.md` §4.0.10、paper 1 `limitations.tex` L1–L12（素材转写、只读）。

## 红线子集（方案 §5 为全表，此处只列最高频；落地每段自查）

- **零新数字**：正文与两表不出现章内未有的数字；章内已有数字原则上也不搬运，以 `\S\ref` 指回；
- 收束引用逐字对齐锁定措辞：「而非单点最高值优选」（§5.7.5）、「只宜读作机制方向上的提示」（§5.6）、质量区间口径（§5.9.5 rev.2）、两读法并存 + 两种子限定连同引用（§5.8.5）；
- §0.4 五条全适用：LN 仅一句正交定位不进清单；+23–32pp 不复述不归单一轴；三项失效方式各异；LN n=2 仅存在性；交互只方向不跨源幅度比（两表表注各声明一次）；
- ❌ 禁：信息论不可能 / proven actor-incapable / "某线调错了" / teacher / 裸 regime / 裸 ReBRAC / seed 44 具名 /「」角引号 / 报告体 marker / 序数枚举 / 跨章点名；
- 小结零数字、零新文献、不开新论点；生物回扣仅一句、贡献句无生物专名、"功能替代而非结构仿制"字样。

## 落地工序（方案 §6）

1. 起草 `sections/discussion.tex`：文件头 rev 注释块记录方案依据与红线遵守清单（沿 §5.7–§5.9 惯例）；两表按方案 §3 制作；每段对照方案 §4 映射表核锚；
2. `main.tex` 取消注释 `\input{sections/discussion}`；
3. 编译：`cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`；0 undefined ref/citation、bibtex 0 warning、无 Overfull>10pt；成功后 `latexmk -c`；
4. grep 自查 marker：teacher / 裸写 ReBRAC / 直引号与「」 / 其一其二 / 信息论 / 不可能 / regime / 导航 / seed 44 等；
5. `status.md` §5.10 行更新为"落地完成（rev.1）、待独立对抗复审"；本文件重写为复审轮入口。

## 输出与工程

- 产出：`sections/discussion.tex` + `main.tex` 一行取消注释；**不改已闭环九节 `.tex`**（发现跨节接口问题先呈报）；`paper/sections/` 只读；
- 预期规模约 4–6 页（内容优先于篇幅）；无新增 cite 预期（确需则先登记再查证）；
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
