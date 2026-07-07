# 第 5 章续写 — 本轮入口 Prompt（§5.9 算法对比 FQL · 轻流程起草轮）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与本节相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **轻流程已转正**（2026-07-07 用户拍板；§5.8 试点检验通过：独立复审 CRIT/HIGH=1 ≪ §5.6 基线 4 / §5.7 基线 5）：论点级骨架直落、免硬停等确认，AskUserQuestion 仅留真分叉决策；落地后交 diff 摘要 + 红线条对条自查；下一轮独立对抗复审收尾。硬护栏（数字实时回查 / 红线子集 / 编译验证）不变。

---

你是 AUV 运动规划与强化学习交叉领域的资深专家，同时是严格的中文博士学位论文外审人与母语级中文学术编辑。发现 spec 或本 prompt 自身有问题就提出来，不必拘泥。本次对话唯一任务：**起草 `sections/algo_compare.tex`（§5.9 算法对比：FQL 对 ReBRAC-Q）**，这是起草轮、不是复审轮，也不是重写任何已定稿节。

【中心命题（spec §0.1 原话照用，全章只论证这一句）】在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。
【§5.9 的角色】回答子问「换更强算法会变吗」（spec §0.2 第 5 行）：更具表达力的流匹配策略先验（FQL）**不普遍取胜**，真正的算法决定因素是**模仿目标质量与数据条件的匹配**——这同时兑现中心命题正向轴第二途径（模仿目标与数据质量相适配）与反向轴第三项（更强先验不普遍有效）。上承 §5.7.5 伏笔（「当数据条件变化时这一模仿目标是否仍然恰当，属于算法与数据条件的交互问题，留待算法比较的一节考察」），为 §5.10 统一机制供料。

## 按序必读

1. spec `paper/thesis_chapter_outline.md` §2 §5.9 块（两层结论顺序 + 口径红线）+ §0.4 红线 5 + §0.5.9/§0.5.10 + §5 图表清单 (5.9) 行；
2. 邻节锚：`sections/rebrac.tex` §5.7.5（伏笔句与 β1=4.0 选择口径）、`sections/methodology.tex` §5.4.5（FQL 公式之家 subsec:ch5_method_fql，α_FQL 蒸馏项——本节零重复、只引）、`sections/setup.tex` tab:ch5_datasets（SAC 检查点四档行）与 §5.3.6（统计口径）、`sections/boundary.tex`（§5.8 刚定稿的语体基准）；
3. **数字唯一权威**（回查核对、不照抄 headline）：`docs/fql_succession_p2_results.md`（§2 2×2 矩阵 / §3 机制三连 Q1→Q1b→Q1c / §4 C-1 公平复赛 / §5 n=2 统计力 / §6 机制综合 / §6.5 u15_cross FLOOR / §7 claim 边界 / §8.1 β1 scope caveat）+ `docs/fql_succession_p2_mechanism_diagnostic.md` §9 + cheatsheet `docs/fql_succession_paper_writing_index.md` §2；跨数据源翻转 = `docs/arrival_v2_sac_collector_design.md` §4.0.10（mexp 同源 Δ+0.123 CI[+0.022,+0.233]；m_multi_mix 异源 +0.035 CI[+0.010,+0.065]）。

## 论点级骨架（子节切分与段落节奏归你裁量）

1. **承接与设问**：§5.7 的原始动作模仿目标在高质量确定性数据上工作良好；数据条件变化（质量下降、噪声、多模态）时，更具表达力的生成式先验是否更优——FQL 以流匹配行为建模 + 蒸馏策略作为对照算法入场。
2. **第一层（privileged 采集器数据条件内）**：2×2 模态×噪声矩阵证伪 conditional-iff 假设——判别因素是**噪声**而非多模态；机制三连（价值侧排除 → β1 4→1 在含噪数据 +23.5pp → β1=1 在干净数据同样更好）把唯一正格溶解；C-1 给 FQL 自身系数的公平复赛仍失败；该数据条件内单一 ReBRAC-Q β1=1.0 双轴占优。**此层是铺垫，不是本节主结论。**
3. **第二层（跨数据源，章级主结论）**：换 SAC 检查点采集器数据后排名**翻转**（mexp 上 FQL 显著更优）→ 算法排名依数据条件而异（算法×数据质量交互）；(5.9) interaction 图为章级 headline 图（强推，`_ch5_style` 新作）；verdict 3 图（matrix / noise-axis / c1-rescue）按需重绘选用。
4. **统一机制（喂 §5.10、不越位收束）**：离线性能由模仿目标质量与数据条件的匹配决定，不由策略先验表达力决定——含噪中质量数据下经流匹配去噪的参照动作更优、高质量干净数据下原始动作模仿目标更优。
5. **§5.9.m 实施细节**：FQL 相对 §5.4.5 的实验协议增量 + 2×2 协议 + 机制三连与 C-1 设置 + SAC collector 39-run 跨源设计 + 异源幅度不可比的口径计算 + n=2 统计力交代（带 caveat 引用级，report §5）。

## 红线（严守）

- **两层结论顺序不可倒**；**不把第一层「ReBRAC-Q 占优」当章级主结论**（spec §2 §5.9 加粗警告）。
- **§0.4 红线 5 口径**：mexp 反超 = 同源干净比较可报效应量；「方向翻转」的 ReBRAC>FQL 一侧落在异源 m_multi_mix——跨数据源仅 direction robust，**只 claim 方向、不 claim 幅度比**。
- u15_cross FLOOR（clean 0.719 / noisy 0.098）作 scope caveat，不弱化主工况结果；FQL n=2 属「带 caveat 引用」级，统计强度同台呈现（§0.5.6 负面/条件化结果作 finding）。
- β1=1.0 与主线 β1=4.0 的表面矛盾**不在本节 reconcile**（就近一句线别限定即可），统一 reconciliation 留 §5.10；特权 critic 四态收束同样留 §5.10。
- 术语（§0.5.10 + §5.4 先例）：FQL 专名保留；**禁 teacher 作概念主语**——写「流匹配模型积分去噪得到的参考动作 / 蒸馏所得部署策略」（FQL 内部蒸馏是真蒸馏、「蒸馏」可用）；SAC 检查点是「采集器」；数据 regime →「数据条件 / 质量区间」；每段 3–4 句；§0.5.9 报告体 marker 零出现。
- 图纪律：结果节图必须画结果数字；`_ch5_style.py` 复用 + 六条硬规范；数据手填 ground truth 精确数字。

## 输出与工程

- 新建 `sections/algo_compare.tex`（头注含 rev.1 块：依据、数字 ground truth、红线、本轮裁决）；`main.tex` 取消 §5.9 占位注释；`status.md` §5.9 行更新。
- 落地后交 diff 摘要 + 红线条对条自查；编译验证 `cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`，0 undefined ref / 0 undefined citation / bibtex 0 warning，成功后 `latexmk -c`；新增 cite（FQL 原文等）先核验著录、改 `refs.bib` 后删 `main.bbl main.aux main.fdb_latexmk` 再从零编译。
- 数字一律实时回查 ground truth；禁凭记忆、禁以任何成稿自述为二手源。可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
