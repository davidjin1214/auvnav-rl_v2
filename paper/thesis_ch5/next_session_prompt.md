# 第 5 章续写 — 本轮入口 Prompt（§5.9 算法比较 · 独立对抗复审轮）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与本节相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **轻流程已转正**（2026-07-07 用户拍板）：起草轮直落，本轮为配套的**独立对抗复审收尾轮**。硬护栏（数字实时回查 / 红线子集 / 编译验证）不变。

---

你是独立的对抗性复审人：不信任起草轮的任何自述（包括 `.tex` 头注 rev 块），一切以 ground truth docs 与 spec 原文为准。本次对话唯一任务：**对 `sections/algo_compare.tex`（§5.9 算法比较，rev.1）做独立对抗复审并落地修订（rev.2）**。参照 §5.8 复审轮先例（对抗 + 系统性合并；findings 存档 `section_5_8_review_findings.md`、成果录 boundary.tex rev.2 头注）。

【中心命题（spec §0.1 原话，全章只论证这一句）】在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。
【§5.9 的角色】回答「换更强算法会变吗」：FQL 不普遍取胜，决定因素是模仿目标质量与数据条件的匹配；第一层（privileged 采集器族内 ReBRAC-Q β1=1.0 占优）是铺垫，第二层（跨数据源排名翻转 = 算法×数据质量交互）才是章级主结论。

## 按序必读

1. `sections/algo_compare.tex` 全文 + 头注 rev.1 块（其数字清单仅作起点，**逐项重新回查**，不作二手源）；
2. spec `paper/thesis_chapter_outline.md` §2 §5.9 块 + §0.4 红线 5 + §0.5.6/§0.5.7/§0.5.9/§0.5.10 + §5 图表清单 (5.9) 行；
3. **数字唯一权威**（全量回查）：`docs/fql_succession_p2_results.md` §2/§3/§4/§5/§6/§6.5/§7/§8.1 + `docs/fql_succession_p2_mechanism_diagnostic.md` §9（任何不一致以 diagnostic 为准）+ `docs/fql_succession_p2_collection_log.md`（四单元构造/规模/采集器 SR 0.632）+ `docs/arrival_v2_sac_collector_design.md` §4.0.1/§4.0.3/§4.0.7/§4.0.10（四档协议与 39-run 联合矩阵、全部 CI）；
4. 邻节衔接锚：`sections/rebrac.tex` §5.7.5（伏笔句承接、β1=4.0 稳健性优选口径）、`sections/methodology.tex` §5.4.5（$\mathbf{a}_{\mathrm{FM}}$ / $\alpha_{\mathrm{FQL}}$ 记号一致、公式零重复）、`sections/setup.tex` §5.3.6（统计口径不被推翻）与 tab:ch5_datasets、`sections/boundary.tex`（终点检查点口径同源、FLOOR 与 §5.8 边界的呼应不越位）。

## 复审维度（对抗优先级从高到低）

1. **数字忠实性零漂移**：正文/表/图注/图脚本（`figures/scripts/fig_ch5_fql_noise_axis.py`、`fig_ch5_algo_interaction.py` 内手填数据）逐值对照 ground truth；特查：矩阵四单元 δ 与判读、机制三步（0.705→0.715 / →0.940 / 0.885→0.910）、C-1 逐种子、σ_train≈3.8pp 与全部 t 值、SAC 四档 μ±σ、六条 CI、m_multi_mix 补充 0.987±0.006、FLOOR 五数字（0.985/0.719/0.632/0.098/0.14）。
2. **红线条对条**：两层顺序；第一层不作章级主结论；红线 5（同源 mexp 可报效应量、异源仅方向、不 claim 幅度比——含图 (b) 的呈现方式是否足够防误读）；FLOOR 不弱化主工况；n=2「带 caveat 引用」级 + 统计强度同台；β1 4.0/1.0 只线别限定、不提前 reconcile；特权 critic 四态不在本节收束。
3. **协议如实性**：矩阵表「ReBRAC-Q」列=冻结 β1=4.0 是否处处不被误读为主线最优；噪声注入是采集时注入非事后加噪；跨源配对 = 两共同种子（与 0.987 的三种子均值并置是否会误导）；30 回合 vs 100 回合评估集差异的交代；「预登记判读门槛」表述与源文档 spec v1.3 §5.5 的对应。
4. **语体与术语**（§0.5.9/§0.5.10）：报告体 marker、枚举暴露、teacher/regime/headline 黑话残留、「锚定强度/锚定系数」用法与 §5.7 先例一致性、每段 3–4 句。
5. **组织与展开度**：小节切分是否服务论证；interpretation 段是否完整；与 §5.7.5 伏笔和 §5.10 供料接口是否严丝合缝；图表六条硬规范（宽≤138mm/可读/线可区分/不重叠/少字/Nature 风）。

## 输出与工程

- findings 存档 `paper/thesis_ch5/section_5_9_review_findings.md`（分级 CRIT/HIGH/MED/LOW；CRIT/HIGH 全修、MED 逐条裁决落地或记录、LOW 记录留收尾统稿）；修订直接落 `sections/algo_compare.tex`（rev.2 头注块记录实质项）。
- **轻流程门控对照**：统计本轮 CRIT+HIGH 数，与 §5.6 基线 4、§5.7 基线 5、§5.8 试点 1 并置记入 status.md §5.9 行（若 ≥4 须向用户报告轻流程退回条件是否触发）。
- 编译验证 `cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`，0 undefined ref / 0 undefined citation / bibtex 0 warning，成功后 `latexmk -c`；若改图脚本须重跑生成并目检 PNG。
- 数字一律实时回查 ground truth；禁凭记忆、禁以任何成稿自述为二手源。可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
