# 第 5 章续写 — 新对话启动 Prompt（§5.1–§5.6 已闭环，续写 §5.7 ReBRAC-Q 主线）

> 用法：新开对话时整段粘贴（从下方「---」开始）。本版任务 = **续写 §5.7 ReBRAC-Q 主线（本章重心，复用 paper 1）**——§5.1–§5.6 六节均已落地、经单节对抗复审 + **章级整体验收复审（七路并行子 agent）闭环、修复落地并提交（commit `238f11b`，2026-07-03）**，编译通过。**先给方案、对齐后再落 `.tex`、编译验证。**
>
> 配套：编写后续各节的新 prompt（起草 / 复审 / 插图 / 润色）前，先看 `prompt_playbook.md` —— 每节标准会话序列、prompt 十要素 checklist、分类型骨架关键句、反模式清单的元层总纲（2026-07-02 由写作期全部会话首条 prompt 回顾提炼）。

---

你是 AUV（水下自主航行器）运动规划与强化学习交叉领域的资深专家，同时是一位严格的中文博士学位论文审稿人与资深中文学术编辑（母语级学术汉语语感）。以专家与审稿人的眼光自由判断，**不必拘泥既有 spec 的条条框框——发现 spec 有问题就提出来**。

## 背景

我在写一篇中文博士论文的**第 5 章（核心贡献章）**，标题「局部感知下水下航行器运动规划的强化学习方法」。

**中心命题（已锁定，全章统一表述，原话照用）：**
> 在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。

全章规划 **10 节**（2026-06-17 由 9 节扩为 10 节：§5.3 后新增 §5.4 共同方法节，原 §5.4–§5.9 顺延为 §5.5–§5.10）。当前进度：
- **§5.1 引言（rev.9）** — `sections/intro.tex`，已完成（rev.9 仅同步末段 roadmap 到十节）。
- **§5.2 研究背景与定位（rev.2）** — `sections/related_work.tex`，已完成。
- **§5.3 问题设定与评估协议（rev.2）** — `sections/setup.tex`，已完成，七子节闭环（任务/动力学双工况 亚临界 Re150/u10 ↔ 临界 Re250/u15 / s0·s1·s2 传感谱系 / 特权观测 [u_eq,v_eq] / 动作·奖励双轨 efficiency_v2(40-D)↔arrival_v2(48-D)·终止 / 数据集 / 评估协议 Welch·配对自助·rule-of-three / §5.3.m 实施细节）。
- **§5.4 强化学习方法与算法框架** — `sections/methodology.tex`，**已落地并完成独立对抗复审**。7 子节（§5.4.1 POMDP→异策略 / §5.4.2 异策略 Actor-Critic 共同骨架 + SAC / §5.4.3 TD3+BC / §5.4.4 ReBRAC-Q / §5.4.5 FQL / §5.4.6 训练-部署信息协议 / §5.4.7 实施细节）+ 2 表（`tab:ch5_method_framework` 方法框架总表 / `tab:ch5_method_loss` 损失·信息协议汇总）+ 11 编号公式。组织 = "共同骨架 + 带动机的增量"。公式已对照 `auv_nav/rebrac.py`/`td3bc.py`/`fql.py` 源码逐行核验。复审结论 = 公式忠实度通过、§0.4 红线零踩、禁写结果零命中。
- **§5.5 在线情形下的可学习性与信息瓶颈** — `sections/online.tex`，**已落地并经系统性对抗复审加固**。5 子节发现链（亚临界可行 → 临界差距 + 三假设 → 特权/时序消融 + 机制 → 稳健性 + 经验上界 → 部署含义）+ 6 图 + 2 表。复审加固：样本效率改 k=12 立论、三假设各自收口、章级"终检/周期评估"命名上提 §5.3.6、图文一致性对齐。
- **§5.6 离线基线：行为约束方法的两个瓶颈（TD3+BC）** — `sections/td3bc.tex`，**已落地并按 §5.5 标准系统性复审**。把"数据越多越差"从协议伪象纠正为真实现象，拆出数据支持集结构与可部署 critic 信息差两个瓶颈；新增结果图 `fig:ch5_td3bc_size`（TD3+BC vs 纯 BC × 三规模 + 采集成功率参照线）；离散度硬伤已修、setup 增量内联、检查点字典序规则上提 §5.4.m。
- **§5.7–§5.10 待起草**（§5.7 ReBRAC-Q 主线（章重心，复用 paper 1）/ §5.8 泛化边界 / §5.9 算法对比 FQL+SAC collector / §5.10 统一讨论 + 本章小结）。起草这四节时须携带的登记项（贡献类型声明、生物回扣、N2′ 答辩风险、传感 3-seed 补录）见 `chapter_acceptance_review_5_1_5_6_findings.md` §5 + spec §8 开放点。
- 全章工程当前编译通过（约 33 页，0 undefined ref / 0 undefined citation / bibtex 0 warning / 0 overfull >10pt）；**§5.1–§5.6 已经章级整体验收复审 + 修复落地并提交（commit `238f11b`）**，§5.7–§5.10 待起草。

## 先读这些（按序）

- 章工程 + 10 节序 + 当前哪些已 `\input`：`paper/thesis_ch5/main.tex`（现 `\input` 了 intro / related_work / setup / methodology / online / td3bc 六节，§5.7–§5.10 为注释占位）
- 已完成六节（语体 / 术语 / 红线 / 缝合范本，**起草余下各节的语体与公式基准**）：`sections/intro.tex`、`related_work.tex`、`setup.tex`、`methodology.tex`、`online.tex`、`td3bc.tex`
- §5.4 写作参考（边界与段落级蓝图）：`paper/thesis_ch5/section_5_4_methodology_review.md`、`section_5_4_paragraph_blueprint.md`
- **§5.1–§5.6 章级验收 findings**（本轮修复依据 + §5.7–§5.10 起草须携带的登记项清单）：`paper/thesis_ch5/chapter_acceptance_review_5_1_5_6_findings.md`（§3 逐条落地核查表 / §5 剩余登记项：贡献类型声明、生物回扣、N2′ 答辩风险、传感 3-seed 补录、图浮动漂移）
- 写作 spec（**权威参考、但非不可改**；§0 中心命题/红线/体裁规范 §0.5.9 register·§0.5.10 术语 / §1 十节骨架 / §2 逐节素材映射 / §3 三线缝合点 / §4 可引用结果分级 / §5 figure&table 清单 / §6 复用矩阵 / §7 写作顺序 / §8 开放点）：`paper/thesis_chapter_outline.md` rev.12
- 数字 ground truth（唯一权威，写正文实时回查、勿凭记忆）：
  - 三线入口：`docs/offline_rl_line_summary.md`、`docs/online_rl_line_summary.md`、`docs/rebrac_line_overview.md`
  - §5.6 TD3+BC：`docs/td3bc_phase0c_experiment_report.md`、`docs/rebrac_line_overview.md` §1
  - §5.7 ReBRAC-Q：`docs/rebrac_experiment_report.md`（rev.8，唯一权威数字源）、`docs/rebrac_mainline_review.md`
  - §5.5 online：`docs/online_rl_line_summary.md` §1.1（A0）、`docs/arrival_v2_experiment_report.md` §7.7–§7.9
  - §5.8 边界：`docs/rebrac_broad_validation_v2_report.md` §3–§5、§4.5（asym ablation）
  - §5.9 算法对比：`docs/fql_succession_p2_results.md`、`docs/fql_succession_paper_writing_index.md`、`docs/arrival_v2_sac_collector_design.md` §4.0.10
- paper 1 复用素材（**勿改动 `paper/sections/` 目录**）：`paper/sections/method.tex`/`experiments.tex`（→§5.7）、`discussion.tex`/`limitations.tex`/`conclusion.tex`（→§5.8/§5.10）；图 `paper/figures/output/`（fig2 seed dotplot / fig3 q-drift）
- 项目总纲：`CLAUDE.md` / `AGENTS.md`

## 本章已锁定的决策（务必延续，勿重开）

1. **中心命题 = 上述表述**（"用好已有 vs 增添能力"）。全章中心句、本章小结、各节首句的"本节为中心命题贡献哪一块"自查都回归这条线索。
2. **术语对照规范（spec §0.5.10）是硬约束**：off-policy→异策略；裸 actor/critic→策略网络（actor）/价值网络（critic），`Actor-Critic` 仅作方法类名；**teacher 是概念错误**（离线策略学自 baseline 采集数据、无蒸馏关系），统一写"以特权信息在线训练所得策略（用作性能上界/参照）"；FQL 优先写"流匹配模型积分去噪得到的参照动作 / 蒸馏所得部署策略"；anchor 名词→模仿目标/参照分布；regime→工况（流动）/数据条件；不用 旋钮/headline 等黑话。算法名统一 **ReBRAC-Q (ours)** / TD3+BC / FQL / SAC，**ReBRAC-Q 须完整定义为 Q 归一化双正则 TD3+BC 变体、不裸写 ReBRAC 指代本文方法**。
3. **不写反向宣告/自我贬抑**（spec §0.5.9(a)）：正式论文既不宣告新意、也不声明"本章不提出新算法 / 方法均为已有"，直接陈述所做与所得、令贡献自证。
4. **生物（借流）只作引子**：仅 §5.1/§5.2 出现，**§5.7/§5.8/§5.9 不回扣**。setup 及方法节专名解禁（涡街/REMUS/DVL/ReBRAC 等可出现）。**§5.10 例外——部分解锁（2026-07-03 决，spec §8 #10 resolved）**：本章小结**允许一次"空间↔时序功能替代"回扣**（§5.5 单点时序替代空间探针 ↔ §5.2 侧线空间分布式对照的收束），四护栏 = ① 仅收束/motivation 层、一句话级；② 贡献句不带生物专名（守抽象下界）；③ 措辞锚"功能替代而非结构仿制"、与 §5.2¶4 呼应防 over-claim；④ 不改命题、不新增 claim。
5. **§5.4 已统一的公式口径（后续结果节引用、勿改）**：ReBRAC-Q actor `−λ_Q·E[Q]+β1‖π(o)−a‖²`，λ_Q=1/max(E|Q|,ε)；**ReBRAC-Q critic 侧惩罚 = ‖ã'−a'_D‖²，a'_D=同轨迹下一步数据动作（非 target smoothing 噪声项）**；TD3+BC λ=α/E|Q|、BC 权重为 1、无 critic 侧 BC、无 LayerNorm；**LayerNorm = ReBRAC-Q critic 侧表征稳定性基础设施（非可选附加能力，与"特权信息可有可无"命题正交）**；FQL teacher `‖v_θ−(a−x_0)‖²` / student `−λ_Q·Q+α_FQL‖μ−a_FM‖²`、critic 无 critic 侧 BC、不依赖特权观测。β1/β2 在 §5.4 只作系数含义，具体值与跨线 reconciliation 留 §5.7/§5.10。
6. **§0.4 红线**（写 §5.6/§5.7/§5.8 尤须严守）：① critic LayerNorm 不进"可有可无"清单（LN-off −16.2pp 是单项最大 mean 杠杆，远超 β2=0 的 −2.4pp）；② mean 归因只在 actor β1 vs critic β2 之间成立，不可升级为"actor 侧决定全部 mean"；③ LN claim 强度 n=2 仅支撑"必要组件存在性"；④ 不写"ReBRAC over TD3+BC 的 +23–32pp 来自 actor anchor"；⑤ 空间传感器写"非必需"非"无用"，三项"可有可无"失效方式各异；⑥ FQL 反超只 claim 方向不 claim 幅度（mexp 同源干净、m_multi_mix 异源仅 direction-robust）；⑦ §5.8 算法发现不得嫁接到 §5.7 ceiling；⑧ N2′ 失败不写成"s0 actor 信息论不可能"（仅 claim privileged critic 不能 rescue、排除 critic-fundamental）；⑨ 引言/方法零跨章引述。
7. **NO APPENDIX**（spec §0.5.5）：paper 1 App A–G 按内容主题融入各 §5.x 末"§5.x.m 实施细节与可复现性"子节，不设附录。**内容优先于篇幅**（spec §0.5.8）：不预设页数/cite 数；规模由"论证完整 + 读者可追溯"裁决。

## 工作方法（务必遵守）

- **先给我过目写作方案/草稿、对齐后再落 `.tex`**；新文件可直接建，已锁文件（intro/related_work/setup/methodology）勿覆盖。
- **新增任何 cite 必须先联网核验著录**（CrossRef / PMLR / arXiv / 出版方），不凭记忆。
- 每次落地后**编译验证**：`cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`；确保 **0 undefined ref / 0 undefined citation / bibtex 0 warning**，并实查新 key 已进 `main.bbl`（改 bib 后须 `rm -f main.bbl main.aux main.fdb_latexmk` 再从零跑，否则 bbl 可能 stale）。成功后 `latexmk -c` 清中间文件（保留 `.tex/.bib/.pdf`）。
- **不要 git commit，除非我明确要求。**
- 数字一律实时回查 ground truth docs，不复制可能漂移的 headline。

## 本轮任务 —— 续写 §5.7 ReBRAC-Q 主线（本章重心）

（spec §7 #3；本章分量最重的一节）§5.7 是全章重心，直接**复用 paper 1**（ReBRAC-Q arXiv preprint draft，commit `932aca1`）。在 §5.4 已统一定义 ReBRAC-Q 损失口径的基础上，本节承载主线实证：Q 归一化双正则 TD3+BC 变体如何在可部署 s0 单点观测下，把离线策略性能逼近以特权信息在线训练所得的参照上界；四条 paper-ready findings（i）–（iv）+ 消融证据。

**素材与复用：**
- 主干复用 `paper/sections/method.tex` / `experiments.tex`（**勿改动 `paper/sections/` 目录本身**；复制/改写进新建 `sections/rebrac.tex`）；图 `paper/figures/output/`（fig2 seed dotplot / fig3 q-drift）。
- 复用时须做**章级适配**：术语对齐 §0.5.10（ReBRAC-Q 完整定义为"Q 归一化双正则 TD3+BC 变体"、不裸写 ReBRAC 指代本文方法；teacher→"以特权信息在线训练所得策略（用作上界/参照）"）；记号与 §5.3 setup / §5.4 公式一致；损失基础定义已在 §5.4，本节只写主线实证增量。

**数字 ground truth（实时回查、勿凭记忆）：** `docs/rebrac_experiment_report.md`（rev.8，唯一权威数字源）+ `docs/rebrac_mainline_review.md`。

**§0.4 红线（本节尤须严守）：** ① critic LayerNorm 不进"可有可无"清单（LN-off −16.2pp 是单项最大 mean 杠杆，远超 β2=0 的 −2.4pp）；② mean 归因只在 actor β1 vs critic β2 之间成立，不可升级为"actor 侧决定全部 mean"；③ LN claim 强度 n=2 仅支撑"必要组件存在性"；④ **不写"ReBRAC-Q over TD3+BC 的 +23–32pp 来自 actor anchor"**；⑤ §5.8 算法发现不得嫁接到 §5.7 ceiling。§5.7.m 承担 ReBRAC-Q 复现细节（超参/检查点选择基础已在 §5.4，本节只留实验所需增量）。

请先通读已成六节（§5.1–§5.6）+ spec + §5.7 ground-truth 入口 + paper 1 复用素材，**先给 §5.7 写作方案（也欢迎提出我没列到的），对齐拍板后**再按"先方案后 `.tex`、编译验证"流程起草。**先规划、后动笔。**
