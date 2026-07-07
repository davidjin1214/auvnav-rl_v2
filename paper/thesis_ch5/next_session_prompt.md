# 第 5 章续写 — 本轮入口 Prompt（§5.8 泛化边界 · 轻流程试点）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与本节相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> 本轮同时是**轻流程试点**（2026-07-07 用户决定：精简写作 prompt、扩大 AI 自由裁量）：方案降为论点级骨架、免「硬停等确认」、落地后事后复核。若本轮独立复审 CRIT/HIGH 数量相比 §5.6/§5.7 明显上升，说明门控确属 load-bearing，§5.9 起退回原「方案 → 落地 → 独立复审」三轮重流程。

---

你是 AUV（水下自主航行器）运动规划与强化学习交叉领域的资深专家，同时是严格的中文博士学位论文外审人与母语级中文学术编辑。**优先级序：事实正确 > 中心命题一致 > 术语与语体规范 > 其余一切（子节划分、详略、行文、图表形式）由你自由裁量**。发现 spec 或既有成稿有问题，直接提出。

## 任务

为第 5 章（「局部感知下水下航行器运动规划的强化学习方法」）撰写 **§5.8 泛化边界**（broad-validation v2 → actor-fundamental ceiling），落地 `sections/boundary.tex` 并在 `main.tex` 解注 `\input`。

**中心命题（全章只论证这一句，原话照用）：**
> 在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。

§5.8 为命题贡献的一块：反向轴「向价值网络注入特权观测」在临界工况下的**最强负面 finding**——特权 critic 不能挽救 s0 部署天花板，排除 critic-fundamental 解释。

## 流程（轻流程试点）

1. **先决两项**（AskUserQuestion，仅此两问）：① N2′ n=2 是补种子还是以极重 caveat 呈现（spec §2 §5.8 答辩风险登记，动笔前必决）；② 在线 AsymCritic 结果是否与本节 asym ablation 并列呈现（spec §8 #4，🟡 待拍板）。
2. 给出**论点级骨架**（每子节回答什么问题、用哪些证据、结论边界在哪；一屏以内），随后**直接落地** `.tex`，不等确认。
3. 落地后：编译验证 → 本节红线条对条自查 → 向用户交 diff 级摘要，事后复核。独立对抗复审另开一轮（铺垫节可将对抗复审与系统性复审合并，playbook §2 惯例）。

## 只读这些（按需检索，不必通读）

- spec `paper/thesis_chapter_outline.md`：§2 **§5.8 块**（写什么 / 数字源 / 答辩风险登记 / claim 边界红线）；写作时开着 §0.4（红线）、§0.5.9（语体）、§0.5.10（术语）对照。
- **数字 ground truth（唯一权威，实时回查）**：`docs/rebrac_broad_validation_v2_report.md` §3–§5 + §4.5（asym-critic ablation）。
- 邻节锚：`sections/rebrac.tex`（§5.7.5 已散文式预留的伏笔，衔接从那里接）；`sections/online.tex`（§5.5.5 k=4 scope 保险，本节须写 online↔offline 对照承接）；`sections/setup.tex` §5.3.6（rule-of-three 零计数上界口径之家）。
- 登记项：`status.md` 待决表 + `chapter_acceptance_review_5_1_5_6_findings.md` §5。
- paper 1 复用素材（只读勿改）：`paper/sections/discussion.tex` §6.6、`paper/sections/limitations.tex` L12。
- 语体与结构基准：已定稿结果节 `sections/td3bc.tex` / `sections/rebrac.tex`。

## 本节红线（spec §2 §5.8 + §0.4 子集，严守）

1. **claim 边界**：✅ 可下「privileged-flow critic 不能挽救 N2′ 天花板；排除 critic-fundamental；HARDENS actor-fundamental」；❌ 不可下「proven actor-incapable / s0 actor 信息论上不可能」（asym 零成功同时兼容「表征不可能」与「asym 机制未把信息转化给 actor」两种读法）。
2. N2′ 结果呈现显式标 n=2，按先决 ① 的裁决执行呈现强度。
3. 本节发现**不嫁接回 §5.7**（§5.7 的 ceiling 表述维持原状）。
4. 负面结果作 **finding** 同台呈现统计强度（n / verdict / 零计数上界），不降级为 limitation；机制解释与正面 finding 同等深度。
5. 生物不回扣；零跨章引述；§5.8.m 只写本节增量，协议指针回 §5.3/§5.4。

## 硬护栏（不可谈判）

- **数字一律实时回查 ground truth**：禁凭记忆，禁以任何 prompt、成稿自述或头注为二手数字源。
- 新增 cite 先联网核验著录（CrossRef / PMLR / arXiv / 出版方）。
- 编译验证：`cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`，0 undefined ref / 0 undefined citation / bibtex 0 warning；改 bib 后先 `rm -f main.bbl main.aux main.fdb_latexmk` 再从零跑；成功后 `latexmk -c` 清中间文件。
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
