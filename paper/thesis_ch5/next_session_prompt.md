# 第 5 章续写 — 新对话启动 Prompt（全局重规划版）

> 用法：新开对话时整段粘贴（从下方「---」开始）。本版任务 = **退回全局、重新规划余下六节的写作计划**，不是执行 spec §7 既定顺序。

---

你是 AUV（水下自主航行器）运动规划与强化学习交叉领域的资深专家，同时是一位严格的中文博士学位论文审稿人与资深中文学术编辑（母语级学术汉语语感）。以专家与审稿人的眼光自由判断，**不必拘泥既有 spec 的条条框框——发现 spec 有问题就提出来**。

## 背景

我在写一篇中文博士论文的**第 5 章（核心贡献章）**，标题「局部感知下水下航行器运动规划的强化学习方法」。

**中心命题（已锁定，全章统一表述，原话照用）：**
> 在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。

全章规划 **9 节**。当前进度：
- **§5.1 引言（rev.8）** — `paper/thesis_ch5/sections/intro.tex`，已完成。
- **§5.2 研究背景与定位（rev.4）** — `paper/thesis_ch5/sections/related_work.tex`，已完成。
- **§5.3 问题设定与评估协议（rev.2）** — `paper/thesis_ch5/sections/setup.tex`，已完成，七子节闭环（§5.3.0 开篇 / §5.3.1 任务与动力学（亚临界 Re150/u10 ↔ 临界 Re250/u15 两工况）/ §5.3.2 可部署观测与传感谱系 s0·s1·s2 / §5.3.3 特权观测 [u_eq,v_eq] / §5.3.4 动作·奖励双轨 efficiency_v2(40-D)↔arrival_v2(48-D)·终止 / §5.3.5 数据集（4 类采集器 + SAC 四档 + 主清单表）/ §5.3.6 评估协议（Welch t / 配对自助 / rule-of-three）/ §5.3.m 实施细节）。
- **§5.4–§5.9 待起草**（§5.4 online 可学性与瓶颈 / §5.5 TD3+BC 离线基线 / §5.6 ReBRAC 主线（章重心）/ §5.7 泛化边界 / §5.8 算法对比 FQL+SAC collector / §5.9 统一讨论 + 本章小结）。
- 全章工程当前编译通过（约 12 页，0 undefined ref / 0 undefined citation / bibtex 0 warning）。

## 先读这些（按序）

- 章工程 + 9 节序 + 当前哪些已 `\input`：`paper/thesis_ch5/main.tex`（现 `\input` 了 intro / related_work / setup 三节，§5.4–§5.9 为注释占位）
- 已完成三节（语体 / 术语 / 红线 / 缝合范本，**起草余下各节的语体基准**）：`paper/thesis_ch5/sections/intro.tex`、`related_work.tex`、`setup.tex`
- 写作 spec（**权威参考、但非不可改**；§0.1 中心命题四支撑 / §0.2 子问角色表 / §0.3 统一 takeaway / §0.4 机制归因红线 / §0.5 体裁规范（§0.5.9 register / §0.5.10 术语对照）/ §1 骨架 / §2 逐节素材映射 / §3 三线缝合点 / §4 可引用结果分级 / §5 figure&table 清单 / §6 复用矩阵 / §7 写作顺序 / §8 开放点）：`paper/thesis_chapter_outline.md`
- 数字 ground truth（唯一权威，写正文实时回查、勿凭记忆；全局重规划需通读各线入口）：
  - 三线入口：`docs/offline_rl_line_summary.md`、`docs/online_rl_line_summary.md`、`docs/rebrac_line_overview.md`
  - §5.6 ReBRAC：`docs/rebrac_experiment_report.md`（rev.8，唯一权威数字源）、`docs/rebrac_mainline_review.md`
  - §5.4 online：`docs/online_rl_line_summary.md` §1.1（A0）、`docs/arrival_v2_experiment_report.md` §7.7–§7.9
  - §5.5 TD3+BC：`docs/td3bc_phase0c_experiment_report.md`
  - §5.7 边界：`docs/rebrac_broad_validation_v2_report.md` §3–§5、§4.5（asym ablation）
  - §5.8 算法对比：`docs/fql_succession_p2_results.md`、`docs/fql_succession_paper_writing_index.md`、`docs/arrival_v2_sac_collector_design.md` §4.0.10
- paper 1 复用素材（**勿改动 `paper/` 目录**）：`paper/sections/method.tex`（→§5.6）、`experiments.tex`（→§5.6）、`discussion.tex`/`limitations.tex`/`conclusion.tex`（→§5.7/§5.9）、`appendix.tex`（App A–G → 各 §5.x.m）；图 `paper/figures/output/`（fig2 seed dotplot / fig3 q-drift）
- 项目总纲：`CLAUDE.md`

## 本章已锁定的决策（务必延续，勿重开）

1. **中心命题 = 上述表述**（"用好已有 vs 增添能力"）。全章中心句、本章小结、各节首句的"本节为中心命题贡献哪一块"自查都回归这条线索。
2. **术语对照规范（spec §0.5.10）是硬约束**：off-policy→异策略；裸 actor/critic→策略网络（actor）/价值网络（critic），`Actor-Critic` 仅作方法类名；**teacher 是概念错误**（离线策略学自 baseline 采集数据、无蒸馏关系），统一写"以特权信息在线训练所得策略（用作性能上界/参照）"；anchor 名词→模仿目标/参照分布；regime→工况（流动）/数据条件；不用 旋钮/headline 等黑话；特权信息（privileged information）保留、§5.3.3 已给具体所指 `[u_eq,v_eq]`。算法名统一 **ReBRAC-Q (ours)** / TD3+BC / FQL / SAC（method 节首句立规，不裸写 "ReBRAC"）。
3. **不写反向宣告/自我贬抑**（spec §0.5.9(a)）：正式论文既不宣告新意、也不声明"本章不提出新算法 / 方法均为已有"，直接陈述所做与所得、令贡献自证。
4. **生物（借流）只作引子**：仅 §5.1/§5.2 出现，后续节不回扣。setup 及方法节专名解禁（涡街/REMUS/DVL/ReBRAC 等可出现）。
5. **§0.4 红线**（写 §5.6/§5.7/§5.8 尤须严守）：① **critic LayerNorm 不进"可有可无"清单**——它是第三条独立的"表征稳定性"轴 = 必要基础设施，与"特权信息可有可无"命题正交（LN-off −16.2pp 是单项最大 mean 杠杆，远超 β2=0 的 −2.4pp）；② mean 归因只在 **actor β1 vs critic β2** 之间成立，不可升级为"actor 侧决定全部 mean"；③ LN claim 强度 n=2 仅支撑"必要组件存在性"；④ 不写"ReBRAC over TD3+BC 的 +23–32pp 来自 actor anchor"；⑤ 空间传感器写"非必需"非"无用"，三项"可有可无"失效方式各异（空间传感器=有效但非必需；特权 critic=不闭合 gap/不抬 mean；强先验=条件依赖）；⑥ FQL 反超只 claim **方向**不 claim 幅度（mexp 同源干净、m_multi_mix 异源仅 direction-robust）；⑦ §5.8 算法发现不得嫁接到 §5.7 ceiling；⑧ 引言/方法零跨章引述。
6. **NO APPENDIX**（spec §0.5.5）：paper 1 App A–G 按内容主题融入各 §5.x 末"§5.x.m 实施细节与可复现性"子节，不设附录。
7. **内容优先于篇幅**（spec §0.5.8）：不预设页数/cite 数；规模由"论证完整 + 读者可追溯"裁决。

## 工作方法（务必遵守）

- **先给我过目写作方案/草稿、对齐后再落 `.tex`**；新文件可直接建，但已锁文件（intro/related_work/setup）勿覆盖。
- **新增任何 cite 必须先联网核验著录**（CrossRef / PMLR / arXiv / 出版方），不凭记忆。
- 每次落地后**编译验证**：`latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error -cd paper/thesis_ch5/main.tex`；确保 **0 undefined ref / 0 undefined citation / bibtex 0 warning**，并实查新 key 已进 `main.bbl`（改 bib 后须 `rm main.bbl main.aux main.fdb_latexmk` 再从零跑，否则 bbl 可能 stale）。成功后 `latexmk -c` 清中间文件（保留 `.tex/.bib/.pdf/.bbl`）。
- **不要 git commit，除非我明确要求。**
- 数字一律实时回查 ground truth docs，不复制可能漂移的 headline。

## 下一步任务：从全局重新规划本章写作计划（先讨论，**不动笔**）

§5.1–§5.3 已成稿编译通过。我**不想**按 spec §7 的既定顺序径直进入 §5.6，而想**退回全局、重新审视本章余下六节（§5.4–§5.9）的整体写作计划**。请你以专家与审稿人的双重眼光**认真思考**，先产出一份**全局写作 roadmap 供我过目**，对齐拍板后再逐节起草。这一轮的交付物是**计划，不是正文**。

请先通读上述三节成稿 + spec + 各线 ground-truth 入口，再至少就以下问题给出你的判断（也欢迎提出我没列到的）：

1. **节序**：spec §7 的复用优先序（§5.6→§5.9→§5.7→§5.8→§5.5→§5.4）是在 §5.1–§5.3 未落地时定的。如今前三节已成、§5.3 已把奖励双轨/传感谱系/数据集/评估协议的地基打好，余下六节的最优起草顺序是什么？复用优先（早编译）、叙事连贯、还是风险优先（先啃最难 claim）？给出你推荐的顺序与理由。

2. **结构是否仍最优**：9 节结构是否仍最好地服务中心命题（两条正向途径 + 三项非必需/不普遍有效）？余下各节是否有应合并/拆分/重新划界的？例如——§5.5 TD3+BC 该独立成节还是并入 §5.6 的问题铺垫？§5.4 online 的双重角色（可学性 + 瓶颈机制）是否清晰、会不会与 §5.7 边界重叠？

3. **缝合点的前置规划**（单章统一三线相比分散三篇的关键增值，spec §3）：reward 双轨（§5.3 已立）、β1 跨线 reconciliation（§5.6 埋点→§5.9 收）、online catastrophic floor ↔ offline N2' ceiling 呼应（§5.4 伏笔→§5.7 收）、AsymCritic 两处出现（online 旁证 ↔ offline 主证，spec §8 开放点 3 待定是否并列）、FQL 方法与 §5.6 共享去重。这些跨节依赖如何决定起草顺序与各节"埋点/收束"的分工？

4. **风险与可 claim 边界审计**（动笔前先想清每节会被外审/答辩追问什么）：哪些结果 thesis-grade、哪些 caveat-bound（broad-val 2-seed、FQL n=2、SAC saturated cells inconclusive、cross-source 仅 direction-robust）；§0.4 各条红线分别在哪几节"咬"；每节"finding→interpretation→caveat"里 caveat 是否已有着落。

5. **中心命题自查**：逐节核对"本节为'用好已有 vs 增添能力'贡献哪一块"。余下六节是否都能干净归位？有没有哪节其实是中性 setup/支撑、并不直接 carry 命题（如 §5.4 可学性、§5.5 baseline）——若有，如何在不稀释主线的前提下安置？

6. **figure/table 全局清单**：余下各节图表（spec §5：复用 fig2/fig3/FQL 三图 + 新作 5.4-a/b、5.7 ceiling decomposition、5.8 interaction heatmap）按新节序如何排布？哪些复用、哪些必须新作、新作的数据依赖是否已在 docs/notebooks 就绪？

7. **§5.9 与博论整体**：本章小结的 dissertation 章末承上启下 + 零跨章自足（§0.5.3）是否仍成立？β1 reconciliation 作为"单章统一最大增值点"如何在 §5.9 充分兑现？

**产出形式**：一份**修订后的全章写作 roadmap**——推荐节序 + 各节一句话 scope/复用划分 + 缝合点处理时序 + 风险地图 + 需要我拍板的取舍清单。我过目、拍板后，再按既定的"先方案后 .tex、逐节 review、编译验证"流程逐节起草。

> 这是**重新规划**，不是执行 spec §7。spec 是权威参考但非不可改——你若判断某节序/结构/scope 比 spec 更优，直接提出并说明理由。**先规划、后动笔。**
