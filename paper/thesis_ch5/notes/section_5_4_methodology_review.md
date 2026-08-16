# 第 5 章十节结构与 §5.4 方法节审查记录

> 本文件用于固化 2026-06-17 对博士论文第 5 章修订计划的独立审查结论。它是写作参考文档，不是可直接并入论文正文的文本。

## 0. 复核结论

复核后，上一轮总体判断成立：在 §5.3 之后新增 §5.4“强化学习方法与算法框架”是必要的，位置正确，十节结构比原九节结构更适合作为中文博士论文第五章。

理由是第 5 章已经不再是单一算法实验报告，而是同时整合在线 SAC、TD3+BC 离线基线、ReBRAC-Q 离线主线、FQL 继承线、特权信息协议和泛化边界。若缺少一个独立方法入口，算法定义、损失函数、训练/部署信息边界会被迫分散在 §5.6、§5.7、§5.9 中，读者必须在多个结果节之间自行拼接方法关系。这不符合博士论文对“统一问题设定 - 方法框架 - 证据展开”的结构要求。

同时，本次复核补充一个比上一轮更精确的警示：不能直接照搬 `paper/archive/rebrac_standalone/sections/method.tex` 中 ReBRAC-Q critic-side penalty 的旧公式表述。当前实现口径以 `auv_nav/rebrac.py` 为准，critic target 中的惩罚项比较的是 target actor next action 与数据集中的 next action，而不是简单的 target smoothing 噪声项本身。§5.4 写作前必须先把公式口径统一。

## 1. 证据基础

本审查依据如下文件复核：

- `paper/thesis_chapter_outline.md` rev.11：确认十节结构、§5.4 职责、图表清单和起草顺序。
- `paper/thesis_ch5/main.tex`：确认 §5.4 已作为占位节插入，但 `sections/methodology.tex` 尚未创建。
- `paper/thesis_ch5/sections/intro.tex`：发现 roadmap 仍是旧九节结构。
- `paper/thesis_ch5/sections/setup.tex`：确认 §5.3 已定义部署观测、特权观测、奖励双轨、数据集和评估协议。
- `docs/offline_rl_line_summary.md`、`docs/online_rl_line_summary.md`、`docs/rebrac_mainline_review.md`、`docs/rebrac_broad_validation_v2_report.md`、`docs/fql_succession_p2_results.md`：复核算法关系、证据边界和不可过度 claim 的红线。
- `auv_nav/rebrac.py`、`auv_nav/td3bc.py`、`auv_nav/fql.py`、`auv_nav/sac.py`：核对 §5.4 中应采用的实际 loss 口径。

## 2. 必须修正的问题

### 2.1 节号与 roadmap 漂移

`paper/thesis_ch5/sections/intro.tex` 仍按旧九节结构写作，把 §5.4 写成在线情形，把 §5.9 写成综合讨论。这个问题必须在真正写 §5.4 前修正，否则引言与 `main.tex`、`thesis_chapter_outline.md` 的十节结构互相矛盾。

修正原则：

- §5.2：研究背景与定位。
- §5.3：问题设定与评估协议。
- §5.4：强化学习方法与算法框架。
- §5.5：Online RL 可学性与信息瓶颈。
- §5.6：TD3+BC 离线基线。
- §5.7：ReBRAC-Q 离线主线。
- §5.8：泛化边界。
- §5.9：算法比较与数据条件交互。
- §5.10：统一讨论与本章小结。

### 2.2 §5.4 不能替代结果节

§5.4 的任务是建立共同方法语言，而不是提前讲算法胜负。它可以定义 SAC、TD3+BC、ReBRAC-Q、FQL 和特权信息协议，但不能写：

- ReBRAC-Q 相对 TD3+BC 的 +23--32pp 结果。
- TD3+BC 1000 episodes 优于 2000 episodes 的退化现象。
- ReBRAC-Q deployable 与 privileged critic 持平的结果。
- FQL 在 P2 中未超过 ReBRAC-Q 的负结果。
- N2' 0% 成功率与 actor-fundamental ceiling 的证据链。

这些内容分别属于 §5.6、§5.7、§5.8 和 §5.9。

### 2.3 ReBRAC-Q 公式口径必须先统一

§5.4 若写 ReBRAC-Q，必须采用实现一致的 critic target：

```tex
a' = \mathrm{clip}\{\pi_{\bar\theta}(o') + \epsilon\},
\qquad
y = r + \gamma(1-d)
\left[
\min_j Q_{\bar\phi_j}(o', a') -
\beta_2 \|a' - a'_{\mathcal D}\|^2
\right],
```

其中 \(a'_{\mathcal D}\) 是同一条离线轨迹中下一转移对应的数据动作。该项表达的是 next action 与数据支持之间的偏离，而不是 target smoothing 噪声 \(\epsilon\) 自身。若直接沿用旧草稿中“\(\|a' - \pi_{\bar\theta}(s')\|^2\)”的写法，会把 critic-side BC penalty 写窄。

### 2.4 术语必须锁定

正文中应统一采用：

- off-policy：异策略。
- actor：策略网络(actor)。
- critic：价值网络(critic)。
- SAC：在线异策略最大熵 Actor-Critic 方法。
- TD3+BC：行为约束的离线 Actor-Critic 基线。
- ReBRAC-Q：Q 归一化双正则 TD3+BC 变体；不要裸写“ReBRAC”指代本文方法。
- FQL：基于 flow-matching 行为分布建模与蒸馏策略的离线方法。
- teacher：不作为中文论文中的概念主语。需要描述 FQL 时，优先写“flow-matching 生成的去噪参考动作”或“蒸馏参考策略”。

### 2.5 Claim 边界必须前置写清

§5.4 不能引出以下过度结论：

- 不能说 ReBRAC-Q 的 +23--32pp 提升来自单一 actor anchor。
- 不能把 LayerNorm 写成可有可无的额外能力；它是表征稳定性基础设施。
- 不能把 N2' 失败写成“证明 s0 actor 信息论不可能”。
- 不能把 FQL/SAC collector 跨数据源的幅度当作严格可比；只能说 direction robust。
- 不能把 ReBRAC-Q、FQL 或 SAC 写成全局优劣排序。

## 3. 可以保留的设计

十节结构可以保留。其优点是把“研究背景 - 问题设定 - 方法框架 - 结果线索 - 统一讨论”分层展开，使第 5 章符合博士论文章节的体裁，而不是几条实验线的机械并置。

以下设计也可以保留：

- §5.2 只做短背景与定位，不展开算法损失。
- §5.3 固定任务、观测、特权信息、奖励、数据集和评估协议。
- §5.4 集中定义方法框架与训练/部署信息边界。
- §5.5 在线线只承担可学性、信息瓶颈和 SAC collector 参照角色。
- §5.6 先立 TD3+BC 离线基线，为 §5.7 的 ReBRAC-Q 增量提供参照。
- §5.7 保持为本章离线主线与主要正面证据节。
- §5.8 将 critical regime 负结果提级为 finding，而不是降级为 limitation。
- §5.9 将 FQL 与 SAC collector 写成算法与数据质量交互，而不是单纯算法排名。
- §5.10 统一 β1、LayerNorm、特权信息、时序利用和数据质量之间的关系。

## 4. §5.4 的推荐边界

§5.4 应回答三个问题：

1. 本章使用的 RL 方法属于哪些共同框架？
2. 各算法在 actor、critic、行为约束、数据来源和训练/部署信息边界上有何差异？
3. 后文结果节中的比较为什么是同一问题下的可解释比较，而不是互不相干的算法堆叠？

§5.4 应包含：

- POMDP/MDP 表述与部署观测约束。
- 在线与离线数据来源差异。
- 异策略 Actor-Critic 的共同框架。
- SAC 作为在线可学性与 SAC collector 的方法基础。
- TD3+BC 的 Q-normalized actor loss 与 BC anchor。
- ReBRAC-Q 的 actor-side BC、critic-side BC、Q normalization、LayerNorm 和与 TD3+BC/原始 ReBRAC 的关系。
- FQL 的 flow-matching 行为建模、去噪参考动作和蒸馏策略。
- deployable 协议与 privileged-critic 协议的训练/部署信息边界。
- 一个算法框架表和一个 loss/信息协议表。

§5.4 不应包含：

- 成功率、提升百分点、p 值、置信区间和 per-seed 结果。
- 算法最终排名。
- 泛化边界结论。
- FQL 负结果或 SAC collector cross-source 结论。
- 对 §5.6--§5.9 的“后文将处理”式元话语。可以在逻辑上为后文铺垫，但不能把正文写成目录说明。

## 5. 起草顺序评估

当前建议顺序“§5.4 → §5.6 → §5.7 → §5.8 → §5.9 → §5.5 → §5.10”可以执行，但应在前面增加一个零步：

0. 同步节号、roadmap、术语表和 §5.4 公式口径。
1. 写 §5.4 方法框架。
2. 写 §5.6 TD3+BC 离线基线。
3. 写 §5.7 ReBRAC-Q 主线。
4. 写 §5.8 泛化边界。
5. 写 §5.9 算法比较与数据条件交互。
6. 回填 §5.5 Online RL 可学性与信息瓶颈。
7. 写 §5.10 统一讨论与本章小结。

这个顺序是“写作顺序”，不是最终阅读顺序。其合理性在于先固定共同方法语言，再写依赖该语言的离线主线和算法对比。唯一 caveat 是：写 §5.8 前，§5.5 的 online floor 和 history/sensor 口径至少要有段落级草图，否则 §5.8 中对 online floor 的引用会缺少前文锚点。

## 6. 风险清单

| 风险 | 严重性 | 处理方式 |
|---|---|---|
| 引言 roadmap 仍是旧九节结构 | 高 | 写 §5.4 前先同步 |
| §5.4 过度展开 ReBRAC-Q 结果 | 高 | §5.4 只写公式、协议与关系；证据留 §5.7 |
| ReBRAC-Q critic-side penalty 公式写错 | 高 | 以 `auv_nav/rebrac.py` 为准，明确 \(a'_{\mathcal D}\) |
| FQL 被写成负结果提前出现 | 中高 | §5.4 只定义方法，§5.9 再讨论结果 |
| LayerNorm 被列入“可选额外能力” | 高 | 写成表征稳定性基础设施 |
| N2' 被过度解释为信息论不可能 | 高 | 仅 claim privileged critic 不能 rescue，排除纯 critic-fundamental |
| FQL/SAC collector 跨源幅度被严格比较 | 高 | 只 claim direction robust |
| teacher 成为中文正文概念主语 | 中 | 改写为行为数据源、参考策略、去噪参考动作 |
| §5.4 表格过密 | 中 | 仅保留方法框架表和 loss/信息协议表 |
| §5.4 写成目录清单 | 中 | 每段围绕一个方法关系或信息边界展开 |

## 7. 下一步动作

最稳妥的下一步不是直接写 `sections/methodology.tex`，而是先使用 `section_5_4_paragraph_blueprint.md` 固定 §5.4 的段落级结构、公式清单、允许 claim 和禁止 claim。确认蓝图后，再执行两件事：

1. 同步 §5.1 roadmap 与 `main.tex` 注释中的状态说明。
2. 创建 `paper/thesis_ch5/sections/methodology.tex`，按蓝图写 §5.4 正文。

