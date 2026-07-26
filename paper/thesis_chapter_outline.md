# 博士论文章节写作 Spec：Deployable-Sensor Offline RL for AUV Wake Navigation

> 文档版本：rev.13（2026-07-26；rev.12 = 2026-06-17）
>
> **rev.13 修订要点（2026-07-26，全章复审整改第 5 批＝整改轮末批）**：① 新增 **§0.5.11 收口检索规程**（findings N6→N18 三轮演进的终稿：三层检索 ＋ 历轮措辞锁禁用词表 ＋ N17 文献机制核实配套）——此前该词表只存在于每轮重写的 `next_session_prompt.md`，第 1–4 批**连续四次**因此复现上轮问题；② §2 五个节块新增措辞锁：§5.1（L1 中心命题原话照用 ＋ CH2 贡献类型声明前移）、§5.3（N13 终检口径定义 ＋ N8 等价判断规则登记语）、§5.5（N16 消融检验力边界 ＋ CH3 生物回声 ＋ N15 训练计数）、§5.6（M2 采集器两套参照口径）、§5.9（N7 证伪框架 ＋ N8/N12/M10/M14/M17）；③ §8 #10 生物护栏追加解锁一处（CH3）。
> 定位：**博士论文第 5 章 单章统一三线**（online SAC + TD3+BC/ReBRAC + FQL succession）的**核心贡献章**写作 spec。
>
> **✅ 写作策略（2026-06-02 用户拍板）：本 spec = 当前写作目标，不是未来蓝图。**
> Active priority = **直接合成博士论文第 5 章**；**不另起任何 standalone paper**。
> Paper 1 ReBRAC 已有的 31pp arXiv preprint draft（commit `932aca1`）作为 §5.7 主干**直接复用**；paper 2 FQL standalone / paper 3 online SAC standalone **均撤销**，素材分别整合进 §5.9（FQL 算法对比）与 §5.5（online RL 节）。
> Spec 全部内容（§0 中心命题 / §0.4 红线 / §0.5 体裁规范 / §3 缝合点 / §6 复用矩阵 / §5.10 β1 reconciliation 等）直接驱动第 5 章逐节起草。§5.1–§5.3 已落地，rev.11 的当前任务是加入独立 **§5.4 强化学习方法与算法框架**，并据此重排后续写作计划。
>
> **rev.5 修订要点（2026-06-05 用户拍板）**：
> 1. **NO APPENDIX 锁**：本章不设 appendix；paper 1 Appendix A–G（Reproducibility / Hyperparams / Full tables / Stats / Derivations / Curves / Obs spec）全部**按节融入正文**各 §5.x 的"实施细节"子节。该约束写入 §0.5 体裁规范。
> 2. **8 节结构（vs rev.4 的 7 节；⚠ rev.8 已扩为 9 节——独立 §5.2 Background & Related Work，见 §1 节号统一 note）**：拆原 §N.1 为 §5.1 引言（dissertation 章级独立引言，与全博士论文 §1 总引言对接）+ §5.2 problem description（任务/环境/sensor/reward/数据集统一 setup）；后续 §N.2…§N.7 平移到 §5.3…§5.8，§5.8 章末加"本章小结"段。
> 3. **新增 §0.5 章级写作体裁规范（7 + 1 条）**：reader model / voice / structural depth / citation 策略（定性，**不预设条数**）/ no-appendix 实施细节融入 / 负面结果作贡献项 / 写作纪律（不预设页数）/ **内容优先于篇幅**。该节是 rev.4 缺的"how-to-write"层。
> 4. **撤销所有量化目标**（用户 2026-06-05 指示："先不要考虑预设篇幅和 cite 数，重点是先把内容写清楚，而不是控制篇幅"）：§1 章节骨架表删"估计篇幅"列；§0.5 citation 策略与 interpretation 段长改为定性原则；章节总篇幅无 hard cap。
>
> ⚠ rev 历史：rev.1（2026-05-30 初稿）→ rev.2（2026-05-31 §0 spine 收紧 + §0.4 红线）→ rev.3（2026-06-02 sync paper Phase 6/6.1 + §N.7 rev refs 校准；曾短暂含"先独立 paper"框架）→ rev.4（2026-06-02 撤销"先独立"框架 + 章号=5 lock + SAC collector 进 §N.6 正文 + 各 standalone paper 撤销）→ **rev.5（2026-06-05 加 §0.5 体裁规范 + NO APPENDIX 锁 + 8 节结构拆分 + 撤所有量化目标）** → **rev.6（2026-06-06 加 §0.5.9 报告体→学术体 register 提级 checklist；§5.1 rev.3 落地为 worked example，标题宽泛化为「局部感知下水下航行器运动规划的强化学习方法」）** → **rev.7（2026-06-07 §5.1 引言 rev.4 复审落地——去 §5.7 算法发现误嫁接到 §5.6 ceiling、cite 对齐补 SAC/FQL；spec 侧：§0.5.3 / §2 §5.1「上承下接」据 §5.1 rev.3 零跨章引述锁定回写为自足口径、§8 #8 部分 resolved、§1 新增节号待决登记 note）** → **rev.8（2026-06-07 用户拍板**独立 §5.2 Background & Related Work 永久化**：全 spec 重编号为 9 节[原 §5.2–§5.8 整体 +1 → §5.3–§5.9]；§5.2 定位"研究背景与定位"短节[领域地图 + gap 精确区分 + 贡献坐标，深度算法对比下放各方法节]；rev.7 节号待决 CLOSED；§1 骨架表/§2 新增 §5.2 RW 节定位）** → **rev.9（2026-06-07 用户拍板"生物出发点"叙事框架——§5.1 hook 提级为"自然生物借流 → RL 复现 → 本文"三段弧[聚焦鱼类主线，鸟/浮游一笔带过]、§5.2 RW 用 [`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md) 综述做骨架[生物现象 + RL 复现谱系 + 算法背景 三支汇于 gap]；先 spec 设计、intro.tex/refs.bib 后落地）** → **rev.10（2026-06-07 §5.1/§5.2 推倒重写——用户三条不满[强行分开 / 语体不规范 / 重新定位]触发，用户拍板**方案甲**[两节重新分工，不合并、9 节序不变] + **生物中剂量**。§5.1 = 纯引言漏斗[生物压一段起兴，RL 复现谱系 + gap 精确三支区分整体下放 §5.2]；§5.2 = 生物 landscape + RL 谱系 + 算法背景 + gap 的**唯一充分展开处**。语体彻底学术化[清除直译腔"游泳与飞行生物"、破折号插入语堆叠、强调引号、序数枚举、对举清单换皮]，对象/机制作主语。`intro.tex` rev.6 + `related_work.tex` rev.2 落地，5 页 0 undefined / bibtex 0 error；rev.8 独立 §5.2 + rev.9 生物叙事弧框架**保留**，仅调剂量与分工，9 节映射不变）** → **rev.11（2026-06-17 新增独立 §5.4 强化学习方法与算法框架）**：复核后确认博士论文章节需要统一 methodology 入口；§5.4 不宣称新算法，而系统介绍 MDP/POMDP、异策略 Actor-Critic、SAC、TD3+BC、ReBRAC-Q、FQL 与特权信息协议。后续 online/TD3+BC/ReBRAC/边界/算法比较/讨论整体顺延为 §5.5–§5.10；旧"每节各自深讲 method"改为"§5.4 讲共同算法框架，各结果节只讲本节增量与实验协议"。
>
> 与现有文档的关系：
> - 本文是 thesis chapter 的**总规划**；它**统摄**而非取代 [`paper/outline.md`](outline.md)（那是 ReBRAC paper 1 的 8 页 conference outline，对应本章 §5.6–§5.8 的素材来源）与 [`paper/progress.md`](progress.md)（paper 1 LaTeX 工程进度，**31 页 arXiv preprint draft**，可作本章主干直接复用）。
> - 数字 ground truth 一律链接源文档，本文**不复制可能漂移的数字**；headline 数字仅作锚点，写作时实时回查源。
> - 三条 line summary 入口：[`docs/offline_rl_line_summary.md`](../docs/offline_rl_line_summary.md)、[`docs/online_rl_line_summary.md`](../docs/online_rl_line_summary.md)、[`docs/rebrac_line_overview.md`](../docs/rebrac_line_overview.md)。

---

## 0. 章节定位与统一叙事 spine

> **rev.2（2026-05-31）修订要点**：把 spine 从"三段平铺"收紧为**单一中心命题 = 特权信息的可有可无性（privileged-information dispensability）**，并修正一处会被答辩席抓住的归因过头（见 §0.4 红线）。本节是 paper 1 自身 spine（[`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §0.3：*"离线 RL 可以不依赖任何 privileged simulator 信息就把性能逼近 online teacher"*）的**章级提升**，不是新发明。

### 0.1 中心命题（整章只论证这一句；rev.12 统一到 §0.5.10 校准锚与各落地节正文）
本章是博士论文的**核心贡献章**。中心命题（原话照用处为 `intro.tex` ¶7 首句与 `discussion.tex` §5.10.5 全句，两者与 §0.5.10 校准锚逐字一致；`setup.tex` 开篇与 `methodology.tex` 开篇不复述该句、只作指涉，不构成一致性检查点。⚠ rev.13 订正：rev.12 曾把后两处也列为「完全一致」的 locus，经实核为空判）：

> 在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。

> **轴映射（旧"特权信息可有可无 / dispensability"命题在此归位，rev.12）**：rev.2–rev.11 曾以"offline RL 不靠特权 critic、不靠更多空间传感器、不靠更强生成式先验，即可把可部署策略逼近以特权信息在线训练所得策略（性能上界/参照）"为唯一脊柱。rev.12 起，该结论**整体归入新命题反向轴"增添能力"的三项实例**——向价值网络注入特权观测、增加空间传感器、采用更具表达力的策略先验，三者**或非必需、或不普遍有效**（失效方式各异，见 §0.4 红线 3）。正向轴"用好已有信息与数据"= (i) 单点部署信号的**时序利用** + (ii) **模仿目标与数据质量的适配**。原命题"sub-critical 工况成立、critical 工况撞 actor-fundamental partial-observability ceiling"作为本命题的**适用范围与边界**保留（§5.8）。dispensability 不再是脊柱，而是反向轴的一个实例。

### 0.2 中心命题的方法入口与四个支撑（角色分明）

新增的 §5.4 是方法入口，不直接承载实验结论。它负责把 online SAC、TD3+BC、ReBRAC-Q、FQL 与特权观测协议放到同一强化学习框架中，避免后续结果节反复重讲算法基础。

| 子问 | 由谁回答（角色） | 结论（精确版，已过 §0.4 红线） |
|---|---|---|
| **方法坐标是什么？** | 强化学习方法与算法框架（**共同 methodology**，§5.4） | 统一定义 online/offline、异策略 Actor-Critic、SAC、TD3+BC、ReBRAC-Q、FQL 与 privileged-critic 协议；后续结果节只写增量机制与实验协议，不再重复算法基础。 |
| **靠什么机制？** | Online SAC（**铺垫/机制**，§5.5） | 难流场下 s0→s1 的约 44pp gap（多种子实测，2026-07-19 九宫格重排后值；旧 80pp/60pp 已弃用）**不是空间传感器瓶颈、是 actor 时序访问瓶颈**：s0+history k=12（~6s≈涡街周期 30–60%）单变量闭合到 s1 上界（§7.8/§7.9）。给 critic 喂特权流在 online 同样**不闭合 gap**（§7.7）——offline §4.5 asym ablation 的独立 echo |
| **能做到吗？** | ReBRAC 主线（**核心**，§5.7） | deployable-only ReBRAC-Q 与 privileged-critic 协议之间**未检测到统计显著差异**（worldcomp 0.928 vs 0.922, Welch p=0.92；⚠ 2026-07-22 第 3 批整改：**不写「追平／统计意义上持平」**——种子级 95% CI = [−0.127, +0.139]，本章不作等价判断，正面结论锚在可部署一侧自身水平上），并优于 vanilla TD3+BC 23–32pp。→ 特权信息喂 critic 对 **mean 不贡献**（5-seed Δ priv−dep=+0.6pp），只救 outlier seed 44（+12pp） |
| **边界在哪？** | broad-val v2（**边界**，§5.8） | critical regime（Re250/u15）下 s0 与 hull-integral 流弱相关，**oracle 示范 + 特权 critic 都救不了**（N2'=0.000；asym ablation = ACTOR_FUNDAMENTAL_CONFIRMED）→ actor-fundamental ceiling。⚠ **措辞锁（2026-07-22 第 3 批整改，N1）**：成稿 §5.8 节标题已去「上限」（改「失效与归因」）——「ceiling」暗示所观测最高值，而 §5.9.5 同工况 β1=1.0 读数 0.14 高于本节零计数上界 0.033；正文只在带 **k=4 + β1=4.0 双限定**处使用「经验上限」，且「策略侧解释得到加固」须条件于「非对称通道确能把价值侧信息传导至策略改进」这一未检验前提 |
| **换更强算法会变吗？** | FQL succession（**对照/确证**，§5.9） | 表达力更强的 flow-matching 先验（FQL）**不普遍取胜**：算法排名 regime-dependent（SAC mexp 中质量 FQL 赢、saturated 高质量 ReBRAC 赢，两 CI 各自非零）。真正的算法决定因素是 **BC-anchor 目标质量 × 数据 regime 的匹配**，不是先验表达力 |

> 侧重：§5.4 是方法入口；ReBRAC 是**唯一正面核心贡献**（正面回答"能做到吗"）；online=机制铺垫，broad-val=边界，FQL=对照确证 + 真实算法决定因素。四块都在为同一句中心命题——"用好已有的信息与数据（单点时序利用 + 模仿目标与数据质量适配）胜过为系统增添能力（特权 critic / 更多传感器 / 更强先验）"——服务。

### 0.3 统一 takeaway（章末收束）
> 对 deployable-sensor offline RL：正向杠杆是 **actor 对部署信号的时序访问** + **BC-anchor 目标质量与数据 regime 的匹配**；几个"直觉上该有用"的东西其实**非必需或不普遍有效**——给 critic 喂特权流不闭合 gap（online §7.7 + offline §4.5 两条**独立** asym ablation）、增加空间探头有效但**非必需且不可部署**（online §7.8 时序访问可替代）、更强生成式先验**不普遍取胜**（FQL regime-dependent）。这给水下机器人这类部署受限场景一条"**不依赖任何 simulator-only 信号即可逼近 online teacher**"的配方，及其在 critical regime 的失效边界。

> 该 takeaway 在 §5.10 "本章小结"段以 dissertation 章末口吻完整复述（不只是 conference paper 的一句 punchline，需展开到对全博士论文的承上启下 — 见 §0.5.3 structural depth 规范）。

### 0.4 ⚠ 机制归因红线（防答辩席反例，写作时严守）
1. **不写"actor 侧因素*决定*全部 mean 性能"**。critic **LayerNorm** 是单项最大 mean 杠杆（LN-off **−16.2pp**，远超 β2=0 的 −2.4pp；[`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §5.2b）。mean 归因只在**两个 BC penalty 之间**成立：actor β1 carry mean、critic β2 carry 跨数据集 Q-stability（review §0.3 / §5.2）。LayerNorm 是**第三条独立的"表征稳定性"轴**，与"特权信息"命题正交——属"必要基础设施"，**不进** §0.3 的"可有可无"清单。
2. **不写"ReBRAC over TD3+BC 的 +23–32pp 来自 actor anchor"**。在 β1=4.0 ↔ TD3+BC α≈0.25 actor-anchor 口径**匹配**下，recipe 差异同时含 critic-side dual penalty + critic LayerNorm；mean uplift 不能归给单一 actor 轴。⚠ Q-norm **不列入差异**：TD3+BC 的 actor loss（`eq:ch5_td3bc_actor`）本身含同一批量均值 |Q| 归一化，β1↔α 换算恰以两方法共享该写法为前提（2026-07-05 §5.7 复审修正；rebrac.tex rev.2 已同步）。
3. **不写"更多传感器无用"**。s1（多一空间探头）**确实**闭合 gap；正确表述 = "**非必需、不可部署，actor 时序访问可替代**"。三个"可有可无"项失效方式**各不同**：空间传感器=有效但非必需；特权 critic=不闭合 gap / 不抬 mean；强先验=regime-dependent——**不可混为"一律无用"**。
4. **LayerNorm claim 强度**：n=2 seeds 仅支撑 "necessary component **存在性**"，**不 claim "LN 比 dual penalty 重要"**（review §2.2.4）。
5. **FQL 反超的口径**：SAC mexp 上 FQL > ReBRAC β1=1（Δ+0.123, CI[+0.022,+0.233]）是**同源**（sprint1+2 同 collector/manifest）干净比较；"方向翻转"里 ReBRAC > FQL 一侧落在 m_multi_mix（**异源** dataset/manifest），cross-source magnitude 不可严格比较（仅 direction robust，§4.0.10 caveat）。interaction 只 claim **方向**、不 claim 幅度比。

### 0.5 章级写作体裁规范（dissertation chapter conventions）

> **rev.5 新增（2026-06-05）**。本节是 rev.4 缺失的"how-to-write"层 —— 它告诉起草者「博士论文章 ≠ 会议论文节 ≠ 实验报告 ≠ 代码文档」体裁差在哪里，应在每节起草时反复回查。
>
> 用户 2026-06-05 拍板原则：**"内容优先于篇幅"**——本节所有规范均以「论证完整性 + 读者可追溯」为裁决标准，**不预设页数 / cite 条数 / 段落长度上限**。

**0.5.1 读者模型（reader model — 决定 voice 与 prerequisite 假设）**

预期读者按可能性递减：(1) 博士论文**外审人**（同领域 RL 或 marine autonomy 教授级，知道 SAC/TD3+BC 等术语，但**不**熟悉本课题历史）；(2) **答辩席**委员（同上，且会就 §0.4 红线条目反复追问 "你能保证不是 X 解释吗"）；(3) **5-10 年后**的 cite 者（可能是博士生在做相关问题，会从本章 cite 上溯检索）。

shorthand 检验：写每一段时问"如果只看这一段，外审能否复述本段论证？" — 不能 → 加铺垫；能但赘 → 不能砍。

**0.5.2 Voice — 学术体而非汇报体**

体裁差异（rev.5 必读）：

| 体裁 | 关键句样式 | 本章避免 |
|---|---|---|
| 实验报告 / progress log | "我们跑了 X seed，结果 Y" / "TODO: Z" | ❌ |
| 代码文档 / spec | "脚本 `train_sac.py` 接受 `--probe-layout` 参数" | ❌（实施细节嵌入论证，**不**单列脚本说明段） |
| 会议论文节 | "我们提出 X，实验显示 +Y pp" | ⚠（dissertation 章需更展开 motivation 与 caveat） |
| **博士论文章 ✅** | "在 X 假设下，[mechanism Y] 决定 [outcome Z]。第 5.k.m 节给出 statistical evidence [reference]，以及它在 critical regime 失效的边界 [reference §5.8]。" | — |

具体规则：
- 主语优先**机制**（"actor 的时序访问能力决定 …"），其次**结果**（"deployable-only ReBRAC-Q 追平 …"），最后才是**操作**（"我们运行了 …"）。
- 一段一论点；每段首句 = 该段论证的 thesis sentence。
- 实验数字嵌入论证句中（"Δ priv−dep=+0.6pp, Welch p=0.92, n=5 → 特权 critic 在 mean 维度不贡献"），不单列为脚注或 callout。

**0.5.3 Structural depth — 章级 ≠ 节级**

dissertation 章相对会议论文节多两层：

- **§5.1 章级独立引言**（rev.5 拆出；**rev.7 据 §5.1 rev.3 「零跨章引述」锁定回写**）：做**自足的章级定位**——以不依赖具体邻章的抽象口径交代本课题在博论中的角色与本章独立贡献，**不点名引述前一章**（不同范式的全局路径规划；§5.1 rev.3 已锁零跨章），**不**提前披露 §5.3 problem description 的具体 setup。roadmap 以**结果语态、节作主语**收尾（§0.5.9 表(a)），**按语义标签描述节序**（问题设定与评估协议 → 强化学习方法与算法框架 → online 可学性与瓶颈 → 离线基线 → 离线主线 → 失效边界 → 算法对比 → 统一讨论），硬编号以落地 `thesis_ch5/main.tex` 实际节序为准（见 §1 节号待决登记 note）。**不是** paper 1 abstract 的扩写。**（rev.10 方案甲分工锁定，2026-06-07）**：§5.1 为**纯引言漏斗**（生物起兴 → 核心问题 → 反直觉结论 → scope → 贡献声明 → 两条主线机制 → roadmap）；生物出发点压为**一段起兴**（鱼类为主、鸟一笔带过），**RL 复现谱系与 gap 精确三支区分整体下放 §5.2**，§5.1 不重复展开（消除前数轮"两节各讲半套"的重复）。两条主线贡献仅在 §5.1 ¶6 充分展开，§5.2 对此只作 forward pointer。
- **§5.4 强化学习方法与算法框架**（rev.11 新增）：承担博士论文 methodology 功能，集中定义本章所用 RL 方法族、损失函数口径、online/offline 训练差异、特权观测使用协议与算法间关系。它不承担结果解释，不写"本章不是方法创新"之类自我降格语句；贡献性质由后续部署约束下的系统比较和机制结论自证。
- **§5.10 末段"本章小结"**（rev.7 零跨章回写，与 §5.1 对称）：除 takeaway 复述外，加一段**自足的承上启下**——以抽象口径交代本章命题在博论整体论证中的位置（如"本章为博论提供 deployable-sensor offline RL 的可行性证明、边界与算法决定因素"），**不点名引述具体后续章**（不写"下一章 §6 将…"等硬章号衔接）。

每个结果节内部仍按 conference paper 节级展开（motivation → 本节增量 setup/method → results → mechanism/discussion → 实施细节子节）。共同算法原理集中在 §5.4，后续 §5.5–§5.9 只写与本节证据直接相关的增量方法和实验协议。

**0.5.4 Citation 策略（定性，不预设条数）**

原则：每条 substantive claim（机制、数字、对比、限制）须可追溯到 (a) 本课题已有 docs（行内链接，源文档绝对权威）或 (b) 外部文献（`\cite{}`，正式 BibTeX）。**不刻意凑数，也不为压缩而省**。

具体场景：
- **机制 claim**（e.g. "BC anchor 目标质量决定 algorithm × data interaction"）：必须同时 cite 自己实验（§5.9）与原文献（FQL Park et al. 2024）。
- **方法对照**（e.g. "ReBRAC-Q vs vanilla TD3+BC vs FQL"）：算法原文献全部 cite，不假设读者已知。
- **背景概念**（e.g. "Kármán wake / Reynolds number / DVL water-track"）：cite 一篇规范综述/教科书章即可。
- **avoiding citation rot**：本课题 docs 链接用相对路径（`../docs/foo.md`），LaTeX 落地时改成 BibTeX style 内部引用或注释化。
- **不可省的 cite 类型**：(i) 任何复用算法的原文；(ii) 任何 reward / env 设计的 prior art；(iii) any "well-known" claim that is well-known **only** in 局部社区。

红线：宁可让 cite 偏多偏厚也不省。但**不为某具体数字凑 cite**（如果一段论证靠本课题独家实验闭环，"独家"本身就是结论，不需要伪外部 reference 撑场）。

**0.5.5 NO APPENDIX — 实施细节按节融入正文（rev.5 hard lock）**

**约束**：本章**不设 appendix**。paper 1 的 7 个 appendices（A Reproducibility / B Hyperparams / C Full tables / D Stats / E Derivations / F Curves / G Obs spec）全部按内容主题**融入正文各 §5.x 节末的"实施细节"子节**。

实施细节子节命名约定：**§5.k.m 实施细节与可复现性**（每节 1 个），内容覆盖：
- 数据集来源与构造（seed / episode 数 / probe layout / reward / collector）→ paper 1 App A / G 内容
- 超参数表（actor/critic lr / batch / β1, β2 / γ / τ）→ paper 1 App B 内容
- 完整 statistical tables 与 Welch / paired-mean / rule-of-three 结果 → paper 1 App C / D 内容
- 必要的方法推导（e.g. dual-penalty 损失全式）→ paper 1 App E 内容
- 学习曲线 / Q-drift 全图 → paper 1 App F 内容

这些不"放在最后"，而是各 §5.x 节内 motivation+method+results+mechanism 之后的 closing subsection，让读者读完一节即拥有复现该节实验所需全部细节，不需要翻附录。

**0.5.6 负面结果作章级贡献（不是 "limitation" 而是 "finding"）**

本章三条核心负面结果（§5.8 critical-regime actor-fundamental ceiling / §5.9 FQL conditional-iff 证伪 / §0.4 红线 3 三种"可有可无"失效方式各异）**作为正面 finding 写**，不降级为 limitation 段。这是本课题区别于"算法 paper 总宣称 SOTA"的章级骨气。

具体写法：
- 节标题不写"limitation of X"；写"Boundary: X fails when Y"或"Honest Negative: X 不普遍"。
- 负面结果与正面结果同台呈现统计 evidence（CI / p / n），不"在 discussion 末段提一句"。
- mechanism explanation 与正面 finding 同等深度（"为何会失败 = 也是机制贡献"）。

**0.5.7 写作纪律（不预设页数 / 不预设 finding 段长）**

- 每段 3–4 句封顶（沿用 paper 1 progress.md §9 R1）；超出 → 拆段或拆子节。
- 每节"finding → interpretation → caveat"三段式 — interpretation 段**不可省**（finding 与 contribution 之间靠 interpretation 缝合），**长度由论证完整性决定**，不预设上限。
- 数字实时回查源 docs（不复制可能漂移的 headline）；§0.4 红线写作时严守。
- 算法命名统一 **ReBRAC-Q**（§5.4 首次定义，§5.7 严守）/ **TD3+BC** / **FQL** / **SAC**（不裸写 "ReBRAC"）。

**0.5.8 内容优先于篇幅（rev.5 用户拍板的最高写作原则）**

本章不预设页数 / cite 条数 / 段落长度的硬性目标。所有规模决策（拆 / 合节、扩 / 缩 finding 讨论、引 / 不引文献）由「**论证是否完整、读者是否能追溯**」这一对原则裁决。

写完一节后回顾：
- "是否多说了未承担论证的内容？" → 砍。
- "是否少说了关键 caveat / mechanism alternative？" → 加。
- **不**回顾："是否超页 / 是否 cite 数够"。

章定稿后整体过一遍：是否有节扩张到掩盖中心命题 spine？是否有节缩短到读者读完仍不能复述本节贡献？再做 second-pass 调整。

**0.5.9 报告体 → 学术体 register 提级 checklist（rev.6 新增；§5.1 rev.3 为 worked example）**

> 背景：§5.1 初稿（rev.1/2）虽已去生造词，但 register 仍停在"实验报告 / 备忘录"层 —— 病不在术语，而在**汇报语态 + 列举式枚举**。rev.3 据下表逐条提级。本表是 §0.5.2 voice 规范的**可操作化**：每节起草后对照自查，凡命中左列即按右列改。

**(a) 必须消除的报告体句法（marker → 病征 → 改法）**

| 报告体 marker（避免） | 病征 | 学术化改法 |
|---|---|---|
| "本章提出一个（直接的）问题…" | 汇报语态：宣告言说行为而非执行它 | 让问题从 §gap 中自然引出（"由此引出的核心问题是…"），或直接抛出 |
| "本章的回答有三层。第一…第二…第三…" | 列举式枚举：把 outline 计数裸露成正文 | 用因果·转折·让步从属连接整合（"最…的是…；同样…的是…；然而…边界"） |
| "需要强调 / 值得注意的是…" | 空元话语 hedge | 删，直接陈述该 caveat |
| "本章的新意在于…" | 宣告 novelty 而非令其自证 | "贡献在于…"后接实质内容 |
| "本章的贡献不是新算法 / 方法均为已有成果" | 反向宣告：自我贬抑式声明贡献性质（与上一行同病的反面——正式论文既不宣告新意、也不声明无新意） | 删；直接陈述本章所做与所得（方法作工具带出 + cite），令贡献性质自证（worked example 见 `thesis_ch5/sections/intro.tex` ¶5 rev.8） |
| "本章其余部分分为两部分。先…继而…随后…最后…" | 过程流水账（procedural narration） | roadmap 改结果语态，**节作主语**（"第 5.6 节给出…"） |
| "其一…其二… / 贯穿这 N 点的是…" | 结构暴露：把论证的"形状"念给读者 | 散文对举（"一条是…另一条则…"），不预报"有几层 / 几点" |

**(b) 5 条改写原则（落地每节套用）**

1. **去 speech-act 化**：删宣告章节动作的框架句，让论点直接成句。
2. **主语降级 操作→机制**（§0.5.2 优先级）：凡"本章 + 言说动词"，改以**机制 / 对象 / 发现**作主语（"决定成败的是信息在时间上的组织方式"）。
3. **溶解枚举**：序数（第一 / 其一）、流程词（先 / 继而 / 随后 / 最后）→ 逻辑连接（因果 / 转折 / 让步 / 递进）；不向读者预报计数。
4. **去口语 editorializing**："直接的""相应地""也是本章的重点"等删除或学术化。
5. **贡献优先散文**：引言贡献宜综合为**少数主线的论证段**，优于 itemize bullet（bullet 属 §0.5.2 会议体，⚠）；正文节内"finding→interpretation→caveat"仍按散文展开。

**(c) 校准锚**：合格 register 见 paper 1 `setup.tex`（陈述句构造对象、机制作主语）；本章 worked example 见 `thesis_ch5/sections/intro.tex` rev.3。注意"我们 / 本章"本身**不是**报告体 —— `setup.tex` 亦用"我们考虑 / 我们采集"；病在**汇报语态**，不在该主语词。提级时**不**回退已锁语义（零跨章引述 / 抽象到"环境流场 + 单点局部可观测" / 已删生造词）。

**0.5.10 术语对照规范（全章用词硬约束；rev.11，2026-06-13 新增）**

> 背景：§5.1/§5.2 正文行文已基本干净，但 spec 与历史讨论里残留一批实验室黑话（旋钮 / teacher / anchor / regime / headline / BC-penalty 等）。若 §5.3–§5.9 照 spec 起草，会把这批黑话灌进正文，损害专业度。本表是 §0.5.2 voice 规范在**词汇层**的落地，作为各节起草前的硬对照。**同一概念全章只用一种译法——一致性本身就是专业度信号。**
>
> 四档原则：**A** 专名保留英文（领域内中文论文亦写英文的方法名）；**B** 有公认中译，首次括注英文一次、此后只用中文；**C** 实验室黑话 / 无精确所指的隐喻，消除改写；**D** 概念用错，必须改（与文风无关）。

**(A) 保留英文（方法专名，首次定义 + cite）**

| 英文 | 正文用语 | 备注 |
|---|---|---|
| SAC / TD3+BC / ReBRAC / FQL | 同名保留 | 首次 1 句定义 + cite；ReBRAC 全章写 **ReBRAC-Q (ours)** |
| Actor-Critic | Actor-Critic 方法 | 仅作**方法类名**；不可裸用 actor/critic 当名词（见 C） |
| asymmetric Actor-Critic | 非对称 Actor-Critic | 框架名保留；作用描述用中文 |

**(B) 用中文（首次括注英文，此后只用中文）**

| 英文 / 概念 | 正文用语 | 状态 |
|---|---|---|
| offline / online RL | 离线 / 在线强化学习 | ✓已落实 |
| off-policy | **异策略**（off-policy） | ⚠ 正文现为英文，待改 |
| behavior cloning (BC) | 行为克隆 | ✓ |
| behavior-constrained / regularization | 行为约束 / 行为正则 | — |
| BC penalty / dual penalty | 行为克隆惩罚项 / 双重惩罚 | 方法节 |
| privileged information | **特权信息**（保留，见首次定义规范） | ✓已落实；首次锚死所指 + cite |
| privileged critic | 接收特权观测的价值网络 | — |
| partial observability / POMDP | 部分可观测性 / 部分可观测马尔可夫决策过程 | — |
| distribution shift / OOD action | 分布偏移 / 分布外动作 | — |
| value overestimation | 价值高估 | ✓已落实 |
| generative policy / flow-matching | 生成式策略 / 流匹配 | ✓已落实（生成式策略） |
| ablation | 消融（实验） | — |
| sim-to-real | 仿真到现实（迁移） | — |
| ceiling / floor | 上限 / 下限 | ✓已落实（上限） |
| gap | 差距 | ✓已落实 |
| seed | 随机种子 | ✓已落实 |
| benchmark / manifest | 基准 / 固定评估集 | — |
| episode / rollout | 回合 / 轨迹采样 | — |
| collector / behavior policy | 采集器 / 行为策略 | — |
| deployable | 可部署 | ✓已落实 |
| mode collapse | 模式坍缩 | — |
| Welch / paired / bootstrap CI | Welch t 检验 / 配对（均值） / 自助置信区间 | 统计，实施细节子节 |
| Kármán wake / gait | 卡门涡街 / 卡门步态 | §5.2/§5.3 |

**(C) 消除黑话，改精确描述（最高风险，spec 污染源）**

| 黑话 | ✗ 病征 | ✓ 正文用语 |
|---|---|---|
| **旋钮 / 拧旋钮** | 口头语 | "**容量**类手段 / 设计选择"；对举见校准锚 |
| **anchor**（名词） | 无精确所指 | "**模仿目标 / 参照分布**"（§5.1 ¶6"模仿目标"✓保持）。动词"锚定"可留（"锚定于数据分布"✓） |
| **可用性与匹配** | 抽象名词并列、"可用性"指错方向（信号本可用，问题在是否用足） | **废止**；改用校准锚的动词对比表述 |
| **regime**（松用） | 一词多义裸奔 | 流动→"**亚临界 / 临界（雷诺数）工况**"；数据→"**数据条件 / 质量区间**"；regime-dependent→"**依工况而异 / 条件依赖**" |
| **headline** | 编辑口吻 | "**章级主结论 / 核心结论**" |
| **actor / critic**（裸名词） | 中英混用、浮动 | actor→"**策略网络（actor）**"后续"策略网络"；critic→"**价值网络（critic）**"后续"价值网络"。⚠ §5.2 ¶3"评价器"↔"actor-critic"混用，统一为"价值网络" |
| dispensability | 生造抽象 | "**（特权信息的）非必要性 / 可有可无**" |
| cross-source / saturated cell | — | "**跨数据源** / **饱和（区/单元）**" |

**(D) 概念纠正（硬伤，必改）**

| 词 | 为何是错 | ✓ 正文用语 |
|---|---|---|
| **teacher / online teacher** | 引进了不存在的**知识蒸馏师生关系**：离线策略学自 baseline 采集的数据，不蒸馏在线策略；该在线策略在 §5.5 是**性能上界/参照**、§5.9 是**数据采集器** | "**以特权信息在线训练所得策略（用作性能上界/参照基准）**"；不造"特权教师"。"distill/蒸馏"仅在真有蒸馏时用 |

**首次出现处理规范**

- **特权信息**（首次，§5.1 ¶2 或 §5.3）：锚死所指 + cite——"训练期可得、而部署期不可得的额外观测，即**特权信息**（privileged information）；本章中具体指对真实流场沿机体积分的等效流速估计 $[u_{eq}, v_{eq}]$~\citep{pinto2017asymmetric}。"
- **抽象下界（术语放置规则）**：引言贡献层（§5.1 ¶5/¶6）与 §5.2 **不出现** 涡街/REMUS-100/DVL/ReBRAC/FQL 等具体专名——留 §5.3 setup 与方法节；motivation 层（§5.1 ¶1、§5.2）可出现生物/流体术语。
- **任务名统一**：navigation → "**运动规划**"，全章不用"导航"（标题层已锁）。

**校准锚（中心命题定稿 — 候选一，可直接入 §5.1 / §5.10；废止旧"可用性与匹配"表述）**

> 贯穿全章的线索可以概括为一点：在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。

以本句的**动词对比**（用好已有 vs 增添能力）作为全章中心命题的统一表述；§5.1 中心句、§5.10 本章小结、各节首句的"本节贡献哪一块"自查均回归这条线索。

**0.5.11 收口检索规程（2026-07-26 第 5 批固化；findings §8 N6 → §9 → §10 N18 三轮演进的终稿）**

> 背景：全章整改轮的第 1、2、3、4 批**连续四次**因收口检索不足而复现上一轮已修的问题——第 2 批沿用原字串（漏「彼此独立」）、第 3 批漏同义表达（「消失」「所能达到的上限」）、第 4 批甚至**逐字改写了含「追平」的那一行却未触碰行首的禁用词**。规程写在此处而非每轮 prompt 内（prompt 每轮重写、会消失）。

**规则**：每轮整改落地后、编译前，收口检索须同时执行三层，**逐词报出命中数与判定，零命中也须写「零命中」**：

1. **本轮错误类别检索**——按本轮所改问题的**语义类别**（而非原字串）穷举，例如改了非劣性判定就检索全章所有非劣性判定点，改了某个口径名就检索该名的全部实例。
2. **历轮措辞锁禁用词表复检**（下表，**按语义穷举、不按词形**）——这是 N18 的核心：上一轮的锁**不会**因为本轮不碰那一节就自动保持。
3. **本轮新增/改写行自身**也在检索范围内——第 4 批正是漏在自己刚改写的那一行上。⚠ **粒度规定（2026-07-26 第 5 批复审后加，findings §11 N25）**：本层须以**改动所在的整个子节**为单位复读，覆盖其正文、**图注、表 caption 与子节标题**，而非只对全章作字串级 grep。第 5 批的四处失效（§5.7.m 定义式分母、§5.9.4 与图 caption 的跨口径并置、§5.5 图注与节导语的充分性断言、§5.9.3 子节标题与首句）**全部落在「改动行的邻近同义表述」上**，字串级检索按定义看不见它们。

| 类别（锁来源） | 须检索的词 | 合规判定 |
|---|---|---|
| 等价/非劣（第 3 批 H2，§5.7） | 追平、持平、相当、至少相当、同等水平、一致的水平、等价、无差异、都不低于／均不低于、不劣于、不低于 | 须为「未检测到统计显著差异」（立规句式）／「不可区分」（散文复述）／「未被超过」／「未见可检出的差距」 |
| 零结果的全称化（第 3 批 H2 连锁） | 不再带来、不再出现、不再有、消失、抵消殆尽 | 须带「**可检测的**」限定 |
| 上限（第 3 批 N1，§5.8） | 上限、经验上限、性能上限、天花板、ceiling、所能达到的上限 | 只可出现在带 $k{=}4$ ＋ $\beta_1{=}4.0$ 双限定处；节标题与子节标题已去「上限」 |
| 选择性呈现（第 4 批 H7，§5.5） | 单调上升、单调、代表性训练、典型训练、单次训练内部、代表性运行 | 「单调」只可用于**聚合均值**；凡「代表性」须就地说明选取方式与其余轨迹形态 |
| 口径反号（第 4 批 M1，§5.5.3） | 方向相反、反号、结论反转 | 终检口径须按**配对两种子**陈述，配对均值方向**并未相反** |
| 机制过度归属（第 4 批 H6，§5.6） | 特属于、专属于、源于……的支持集结构 | 只可写到**否定性结论**（「该受益并不普遍成立」） |
| 过度断言（第 4 批 H4，§5.2/§5.5.3） | 精确指认、指认为、证明、证实、本章发现……有用、增量不在于……而在于 | 须为「指向……物理候选」＋「表明……在**任务级上**替代」＋ scope 限定 |
| 混杂/旁证（第 4 批 M12，§5.6 开篇） | 混杂因素、排除了……混杂、旁证 | 须挂**常量轴**（不随所比较因素变化 ⇒ 非混杂 ⇒ 界定适用范围） |
| 计数与独立性（第 1、2 批 C1/C2） | 两种子、独立采样、样本不重合、彼此独立、相互印证 | 三种子；30 回合集为 100 回合集的固定前 30 条子集、任务实例嵌套；「相互印证」→「指向同一边界」 |
| 不可支撑的预测（第 2 批 M8） | 不会翻转其方向、必然收紧、终将 | 一律删除或降为条件句 |
| **参照口径混称**（第 5 批 M2） | 采集器自身成功率、自身轨迹的成功率、采集成功率参照 | 「自身轨迹的成功率／采集成功率」**专指数据集采集口径**；采集器在固定评估集上的成绩须写「在同一评估集上直接执行的成功率」，两者**不得互换、不得跨口径比较** |
| **检查点口径**（第 5 批 N13） | 终检、终检评估、最终检查点 | 「终检」只标**评估时机**；检查点身份（训练终点 ↔ 验证集字典序选点）是另一条轴，各结果节须在实施细节中标明 |
| **非劣判定的兜底**（第 5 批 N7） | 族内判定、均未被超过、未见可检出的差距 | §5.9 不作性能相近或等价判断，只作**对优越性主张的证伪**；肯定回答须锚在正向可观测量上 |

> **N17 配套（第 4 批）**：新增 `\cite` 的**存在性与著录核实不等于机制核实**——必须读摘要或正文方法段确认其机制属性与正文转述一致；凡该文献将被用来支撑本章某条轴，须显式检查它落在轴的哪一侧。

---

## 1. 章节骨架总览

> **rev.11 现行结构（2026-06-17）**：§5.1–§5.3 已落地；复核后确认本章需要独立 methodology 入口，故在 §5.3 后新增 **§5.4 强化学习方法与算法框架**。全章由 9 节调整为 **10 节**，原 §5.4–§5.9 顺延为 §5.5–§5.10。该调整不改变中心命题，只改变读者进入算法细节的方式：共同 RL 方法集中讲，结果节只讲增量机制、实验设置和证据。
>
> **节号统一（rev.11）**：
> | 落地 10 节（现行） | rev.10 旧 9 节 |
> |---|---|
> | 5.1 引言 | 5.1 引言 |
> | 5.2 Background & Related Work（研究背景与定位） | 5.2 Background & Related Work |
> | 5.3 Problem description | 5.3 Problem description |
> | **5.4 强化学习方法与算法框架（新增 methodology）** | — |
> | 5.5 Online RL | 5.4 Online RL |
> | 5.6 TD3+BC | 5.5 TD3+BC |
> | 5.7 ReBRAC 主线 | 5.6 ReBRAC 主线 |
> | 5.8 边界 | 5.7 边界 |
> | 5.9 算法对比 | 5.8 算法对比 |
> | 5.10 讨论 | 5.9 讨论 |
>
> **§5.2 RW 职责更新**：仍定位为**简短的"研究背景与定位"节，非全面文献综述**，承载领域地图、gap 精确区分与贡献坐标。算法间深度技术对比不再下放到各结果节，而集中进入 **§5.4 methodology**；§5.2 只保留行为约束、特权信息和数据驱动流场控制的背景级说明，不展开损失函数与训练细节。

| § | 小节 | 主要素材来源 | 资产状态 |
|---|---|---|---|
| 5.1 | **章级独立引言**（dissertation 体裁层，与 §1 总引言对接 + 本章 roadmap；零跨章自足） | 全新写（参考 §0.5.3 structural depth） | ✅ **rev.6 推倒重写已落地**（方案甲：纯引言漏斗 7 段；`thesis_ch5/sections/intro.tex`） |
| 5.2 | **Background & Related Work（研究背景与定位）**（生物借流现象 + RL 复现谱系 + 算法背景 三支汇于 gap；**短节非综述**，承 §5.1 生物 hook、聚焦鱼类，详上方职责限定 note + §2 骨架） | [`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md)（生物借流 + RL 复现综述骨架，22 refs）+ 引言 ¶1/¶5 cite 起步 + offline RL/POMDP·特权信息 文献 | ✅ **rev.2 推倒重写已落地**（方案甲：landscape+gap 唯一展开处，4 段约 2 页；`related_work.tex`） |
| 5.3 | **Problem description**（任务 / 环境 / s0 vs privileged / reward 双轨 / 数据集 statistics） | [`environment_design.md`](../docs/environment_design.md) + paper 1 `setup.tex` | ✅ **可直接复用** paper 1 §3 + Fig 1，需泛化到 reward 双轨 / 多 regime |
| 5.4 | **强化学习方法与算法框架**（methodology：MDP/POMDP、异策略 Actor-Critic、SAC、TD3+BC、ReBRAC-Q、FQL、特权信息协议） | paper 1 `method.tex` + `td3bc.py`/`sac.py` 概念口径 + TD3+BC/ReBRAC/FQL 原文献 + §5.3 观测协议 | 🟡 **新增统摄节**（不宣称新算法；集中讲共同算法框架和损失定义） |
| 5.5 | Online RL：可学性与信息瓶颈 | [`online_rl_line_summary.md`](../docs/online_rl_line_summary.md) §1.1 + [`arrival_v2_experiment_report.md`](../docs/arrival_v2_experiment_report.md) §7.9 | 🟡 **新写**（仅 docs，无 LaTeX） |
| 5.6 | Offline baseline：TD3+BC 拆瓶颈 | [`td3bc_phase0c_experiment_report.md`](../docs/td3bc_phase0c_experiment_report.md) + [`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §1 | 🟡 **新写/扩写**（paper 1 仅作 baseline 引用，无独立节） |
| 5.7 | ReBRAC-Q 主线：方法增量 + 四 findings（**章重心**） | paper 1 `method.tex` + `experiments.tex` + [`rebrac_experiment_report.md`](../docs/rebrac_experiment_report.md) | ✅ **大幅复用** paper 1 §4–§6 + Fig 2/3；共同算法背景移至 §5.4 |
| 5.8 | 泛化边界：broad-val v2 → actor-fundamental ceiling | [`rebrac_broad_validation_v2_report.md`](../docs/rebrac_broad_validation_v2_report.md) §3–§5 + §4.5 | 🟡 **部分复用**（paper 1 Phase 6 已折叠为 §6.6 + L12，提级为独立节） |
| 5.9 | 算法对比：FQL succession 诚实负面 + SAC collector cross-source headline | [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) + [`fql_succession_paper_writing_index.md`](../docs/fql_succession_paper_writing_index.md) + [`arrival_v2_sac_collector_design.md`](../docs/arrival_v2_sac_collector_design.md) §4.0.10 | 🟡 **新写**（撤销的 paper 2 素材整合；FQL 只讲相对 §5.4/§5.7 的方法增量） |
| 5.10 | 统一讨论 + 跨线 reconciliation + limitations + **本章小结** | paper 1 `discussion.tex`/`limitations.tex` + 各线 discussion 段 | 🟡 **整合改写**（融合三线 + β1 reconciliation + dissertation 章末承上启下段） |

**实施细节融入方式（NO APPENDIX）**：每节末附 §5.k.m "实施细节与可复现性"子节，承载 paper 1 App A–G 对应内容（§0.5.5 详）。例：
- §5.4.m 承担共同算法定义、损失函数、特权信息协议与超参数口径；外部算法原文献在此集中首次定义
- §5.7.m 承担 paper 1 App A (Repro) + B (Hyperparams) + C (Full ReBRAC tables) + D (Stats) + E (Dual-penalty derivations) + F (Q-drift curves) + G (Obs spec) 中与 ReBRAC-Q 主线直接相关的部分
- §5.3.m 承担 reward 双轨 + dataset matrix + sensor schematic 复现细节
- §5.9.m 承担 FQL collector / SAC 39-run 数据来源 + statistical method

---

## 2. 逐节素材映射

> 写作纪律（沿用 §0.5 + paper 1 progress.md §9 R1–R10）：mechanism > storytelling；每段 3–4 句封顶；数字实时回查源表；算法命名统一 **ReBRAC-Q**（不裸写 "ReBRAC"）。每节末附 §5.k.m "实施细节与可复现性"子节（NO APPENDIX，§0.5.5）。

### §5.1 章级独立引言（dissertation 体裁层 — rev.5 新增）

- **写什么（与会议节级引言显著不同，§0.5.3 structural depth；rev.7 据 §5.1 rev.3 零跨章引述锁定回写——下列各点均改为不点名引述邻章的自足口径）**：
  1. **章级定位（取代原"上承"）**：以抽象口径交代本课题在博论中的角色——自主航行器在自身无法完整观测的环境流场中做运动规划，单点局部可观测约束下 RL 能推进到何种水平；**不点名前一章**（不同范式的全局路径规划，零跨章），任务名统一"运动规划"（不用"导航/navigation"）；
  2. **本章独立贡献**：deployable-sensor offline RL 这一统一命题（§0.1 中心命题，按 §0.5.9 贡献优先散文综合为少数主线，**不** itemize、**不**借用 paper 1 abstract 的紧凑写法 — 章级独立引言可以而且应当展开 motivation 与 stakes）；
  3. **章 roadmap**：以结果语态、节作主语收尾（§0.5.9），按语义标签——问题设定与评估协议 → 强化学习方法与算法框架 → online 可学性与瓶颈 → 离线基线（TD3+BC）→ 离线主线（章重心）→ 失效边界 → 算法对比 → 统一讨论；硬编号以落地 `thesis_ch5/main.tex` 节序为准（§1 节号待决登记 note）；
  4. **章级承接（取代原"下接"）**：以抽象口径点出本章结论在博论整体论证中的位置，**不点名引述具体后续章**（与 §5.10 本章小结对称，零跨章）。
- **⚠ 措辞锁（2026-07-26 第 5 批，findings LOW L1 ＋ §5 CH2）**：
  - **中心命题原话照用**：`intro.tex` ¶7 此前为对冲式假说（"可能更多依赖于…而不是简单依赖于…"），与 §5.10.5 的断言式不齐、也与本 spec §0.1 自称的"完全一致"不符 → 已回写为 §0.1 校准锚原话，**假说身份由前置框架句"有待逐节检验的中心命题"承担**，升格依据由 §5.10.5 既有的"前九节的证据支持这一命题在本章设定内成立"自证。**后续勿再把 ¶7 改回对冲式**。
  - **贡献类型正面声明前移（CH2 已裁断采纳）**：**¶6**（贡献段；⚠ ¶5 为 scope 段，rev.13 初稿曾误记为 ¶5，经复审订正）末句承载三类型声明（经验规律与设计准则 ＋ 受控判别框架 ＋ 双侧行为约束方法这一可复用制品），与 §5.10.5 ¶2 首尾呼应而非重复。三条相容性约束须同守——① 正向类型声明，不触 §0.5.9(a) 禁反向宣告；② **不出现算法专名**（写"双侧行为约束方法"而非 ReBRAC-Q，守 §0.5.10 抽象下界，§5.10.5 亦作此处理）；③ 不提前给任何结果数字。
- **复用资产**：**无**。不复用 paper 1 abstract（abstract 是 8 页会议节级口吻，外审会立刻识别为"被搬过来"）。重新起草。
- **caveat（reader model，§0.5.1）**：假设外审人/答辩席知道 RL 标准术语（SAC / TD3+BC / Q-learning），但**不**熟悉本课题历史；术语首次出现需 1 句定义 + 1 个 cite。
- **实施细节子节**：本节无 §5.1.m（引言节不需"复现"段）。
- **落地状态（rev.4，2026-06-07）**：§5.1 已落地 LaTeX（[`thesis_ch5/sections/intro.tex`](thesis_ch5/sections/intro.tex)），6 段——环境流场 + 单点可观测的 funnel → 核心问题（online/offline 两情形，offline 重心）→ 两点反直觉 + ceiling 边界 → scope caveat → 贡献（两条主线散文：信息时序维度 / 离线内部 anchor-数据适配）→ roadmap（节作主语、结果语态）。零跨章引述、抽象下界（环境流场 + 单点局部可观测）、贡献散文（非 itemize）、online/offline 不对称权重、§0.4 红线（不提 LayerNorm / 不嫁接算法发现到 ceiling）均已守。复审记录见 commit `bf6fc12`。
- **hook 弧调整（rev.9，2026-06-07 用户拍板"生物出发点"框架；先 spec 设计、后落地 intro.tex）**：rev.4 的 ¶1 工程 hook（直接从航行器 / 环境流场切入）提级为 **"自然生物借流 → RL 复现 → 本文"三段弧**（本章最早出发点）。注意 rev.4 ¶1 已 cite `gunnarson2021ncomm` + `verma2018pnas`（即综述 [15]/[14]），生物根基已隐含，本次为**显式化、前置化**：
  - **新 ¶1（自然原则）**：自然界生物并非被动承受流场，而是**主动感知局部流动结构并转化为运动收益**，且依赖**局部、有限的感知**而非全局观测——一条普适自然原则（注：生物侧线本为空间分布式阵列，"单点"专指本文 AUV 的部署传感，二者构成反衬，勿混；详第二轮复审 B-1）。**聚焦鱼类主线**（涡街 Kármán gait / 障碍物尾流 / 侧线分布式局部感知），鸟类（热气流·动态滑翔）与浮游生物作"更广泛自然界印证"一笔带过（用户 2026-06-07 选定广度）。
  - **新 ¶2（RL 复现 + 收口）**：该借流现象启发用 RL 在复杂流场学习高效运动的研究路线（cite Colabrese 2017 / Verma 2018 / Gunnarson 2021 / Jiao 2025）→ 已有工作多假设丰富观测或允许在线交互 → 收紧到真实部署的严格约束（单点 + 离线 + 部署无特权）→ **接现有 ¶2 核心问题**。
  - **保留**：现 ¶2–¶6（核心问题 / 两点反直觉 / scope caveat / 贡献散文 / roadmap）不动；仅前置改写 ¶1 为两段弧。
  - **红线**：① 生物术语仅在 motivation 层，贡献层（¶5/¶6）仍守抽象下界（不写涡街/REMUS/DVL）；② 生物开篇放大"宽问题"，**scope 收口须更硬**（¶4 caveat + 贡献句旁诚实 scope 顶住 over-claim，§0.4 + R7/R8）；③ 收敛 ≤2 段，详细 landscape 留 §5.2，引言只取代表性 cite；④ 不破坏 rev.4 已守（零跨章 / 贡献散文 / 不对称权重 / 不提 LayerNorm / 不嫁接算法发现）。
  - **素材源**：[`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md)（自然生物借流 + RL 复现综述，22 refs 带 DOI）。
- **落地状态（rev.5 hook + 第二轮对抗复审，2026-06-07）**：生物 hook 已落地（`intro.tex` rev.5，**§5.1 现 7 段**——rev.4 funnel ¶1 拆为生物原则 ¶1 + RL 复现 ¶2；G-3 段数回写）；§5.2 独立成节落地（`related_work.tex` rev.1，4 段）。第二轮独立对抗复审（authoring/review 分离 lane）逐句核 cite 忠实 + 质疑 gap，落地——必改：**B-1**（§5.1 ¶1 侧线「单点式」→「局部、有限」，消事实错[侧线本分布式] + 与 §5.2 自相矛盾；**本 spec line 229 同源措辞已一并修正**）；**D-1**（§5.1 ¶2 jiao2025 不再错配「具身机器人控制」→「辨明可靠学习所需的局部感知量」，四 cite 对齐）；**E-1**（§5.2 ¶4 gap 第三支收敛「训练与部署均不借助特权」，去 asymmetric-AC 稻草人，对齐 §0.1）；**C-1**（§5.2 ¶4 两条主线复述删→ forward pointer，独留 §5.1 ¶6）。commit `98888be`。捎带 MEDIUM/LOW：C-2（¶4↔¶6 时序主线措辞拉开）/C-3（¶6↔§5.2¶3 生成式先验措辞）/C-4（§5.1¶1↔§5.2¶1 鱼类例子承接式去重）/D-2（flato2024 删「风切变」over-attribution）/D-3（删「同伴涡街」，群体维度留 §5.2）/E-2（gap 第一支「多于单点的观测，或在线试错」精确化）。保留不动（LOW）：A-1/A-2 register（分号对照结构）、B-2 scope（¶5 已够）、D-4 coombs cite（贴「分布式」正确 claim）、G-1 refs.bib 未引条目（BCQ/CQL 后续节备用）、G-2 pinto key（RSS 2018 正确）。编译 5 页 0 undefined / bibtex 0 error。 **（rev.6 推倒重写 SUPERSEDE，2026-06-07——用户三条不满触发，方案甲落地）**：§5.1 改为**纯引言漏斗 7 段**[生物起兴 ¶1 → 工程命题+核心问题 ¶2 → 反直觉+ceiling ¶3 → scope ¶4 → 贡献声明（五方法五 cite）¶5 → 两条主线机制 ¶6 → roadmap ¶7]；rev.4–rev.5 的生物两段弧**压为一段起兴**（鱼类涡街驻留+侧线局部有限+鸟热气流；liao/coombs/flato 三 cite）；**RL 复现谱系 + gap 精确三支区分整体下放 §5.2**（消除"两节各讲半套"重复）。语体彻底学术化：清除直译腔"游泳与飞行生物"→"水生与空中的动物"、破折号插入语全去、强调引号全去、序数枚举（其一/其二）溶解为逻辑连接、对举清单换皮（"一条是…另一条…"）溶解为论证散文。红线零踩（不带数字 / 不提 LayerNorm / 不嫁接算法发现到 ceiling / 抽象下界 / online-offline 不对称 / 零跨章 / 贡献散文）。grep 正文违规 marker clean，5 页 0 undefined / bibtex 0 error。

### §5.2 Background & Related Work（研究背景与定位 — rev.8 独立成节）

- **写什么（短节非综述，§0.5.3 + §1 职责限定 note；rev.9 用 [`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md) 做骨架，承 §5.1 生物 hook）**：**五支**汇于 gap（第 5 支为 2026-07-24 第 4 批 H4 新增，见下方措辞锁）——
  1. **自然生物借流现象**（呼应 §5.1 hook，**聚焦鱼类**）：流场的**能量场 / 信息场二分**；鱼类主线（rheotaxis 流趋性 / Kármán gait 涡街 / 障碍物尾流节能 / 鱼群涡相位匹配 / **侧线分布式局部感知**）；鸟类·浮游生物一笔带过。
  2. **RL 复现谱系**：奠基（Colabrese 2017 微游泳体 / Verma 2018 涡利用集体游泳）→ 局部感知目标导航（Gunnarson 2021 局部流速 / Jiao 2025 流场梯度必要性）→ 具身鱼形机器人穿越涡街（Zhu 2022 / Feng 2024）→ 微游泳体算法评估（Qiu 2022 对称性 / Mecanna 2025）。
  3. **算法侧背景**：offline RL 行为约束谱系（BC penalty → 更具表达力的生成式策略先验）、部分可观测下的特权信息与非对称 actor-critic。
  4. **部分可观测与记忆策略**（**2026-07-24 第 4 批 H4 新增第五支**，落地为 `related_work.tex` ¶5）：POMDP 标准框架（Kaelbling 1998）→ 循环网络编码观测序列（Hausknecht 2015 DRQN / Heess 2015 RDPG）与帧堆叠（Mnih 2015 DQN）两类记忆做法 → 流场侧「局部传感足以刻画流动类型与强度而无须全场观测」的先例，且**空间与时间两条路径各有其一**：Colvert 2017 JFM = **空间传感阵列**（同时刻速度差 → 速度梯度张量），Colvert 2018 B&B = **单点时序**（所测为**涡量**）。
  5. **本章 gap 精确区分 + 坐标**：已有 RL 借流 / offline 工作要么假设丰富·多点观测、要么允许在线交互、要么部署期可访问特权信息——"**单点局部感知 + 纯离线 + 部署无特权 + 可部署传感**"这一组合尚无系统刻画；本章科学洞见型贡献（信息时序维度 + 离线 anchor-数据适配）即定位于此。

  > ⚠ **措辞锁（2026-07-24 第 4 批，findings H4）**：整改前本节 landscape 为**四支**，全章正向轴第一途径（时序利用）**零文献坐标**——`refs.bib` 原 31 条无一涉 POMDP 记忆策略／循环网络／观测帧堆叠，§5.4.1 引入 POMDP 无 cite。已补第五支（6 条新 cite，著录经 Crossref／arXiv 逐条实检，见 `refs.bib` 该节注释）。**后续勿因本 spec 旧述「四支」而删该支**（第 1、2 批各复现过一次上轮 CRITICAL，机制均为「spec 未同步 → 下轮按 spec 灌回」）。
  > ⚠ **文献转述锁（第 4 批复审 A 判 CRITICAL、经原文实核订正）**：`colvert2017local` 是**空间**方法（"a local sensory array" + 速度梯度张量，全文无时序成分），**不得**引作时序途径的先例——它论证的恰是本章主张「可被时序窗替代」的那一方，引错即把反方证据写成正方坐标。`colvert2018wakes` 确为单点 + 时序，但所测是**涡量**、不是流速，§5.5.3 须如实写明。**教训固化**：新增 cite 的存在性与著录核实（Crossref 逐项）**不等于机制核实**，后者须读摘要或正文方法段；本批 `refs.bib` 头注曾自证「转述相符」而对 2017 篇为假。
  > **贡献定位锁**：第五支只作 landscape，**不在 §5.2 下增量断言**——帧堆叠／循环策略缓解部分可观测性是**已知通用手段**；本章增量在 §5.5.3 承载，形式为「把被时序窗恢复的信息**指向**涡街脱落相位这一物理候选 ＋ 该恢复足以在**任务级上**替代空间多点探针的受控证据」，**不得写成「本章发现观测历史有用」**（定位过宽即已知结论），**亦不得写成「精确指认」「证明」**——相位从未被直接测量，§5.5.3 机制段本身是「可能／很可能／若如此」的假说措辞，强断言与之相隔一段即构成「先撤销许可再照旧行使」（第 4 批初稿被三个维度同时命中）。
  > **mini-gap 锁**：本支 gap 的后半（**纯离线**条件下时序途径的可行性）只可写成「更少被触及」的开放面，**不得**写成本章将交付的检验——§5.5.5／§5.8.4／§5.10.1 三处均明文声明离线一侧未闭环。
  > **不为凑密度加 cite**（§0.5.4）：§5.8／§5.10 的零引用**维持不动**——引用密度是诊断依据、不是指标。
  > **CH1 未采**（用户 2026-07-24 裁断）：§5.2 **自足补一支**，不采「非硬编号一句指向全论文综述章」的写法，「零跨章」锁不松动。
- **不写什么（去重红线，避免与引言/方法节重复）**：算法间**深度技术对比**（TD3+BC α / ReBRAC dual-penalty / FQL flow-matching loss 差异）集中到 §5.4 methodology，§5.2 不展开公式；引言 ¶1/¶2 的生物 hook 与方法谱系初次点名不在 §5.2 重复，§5.2 把它们**展开为有论点的 landscape 并深化 gap**。
- **复用资产 / 素材源**：主骨架 = [`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md)（§2 能量/信息场框架 + §3 生物进展 + §4 RL 进展 + §5 自然↔RL 对应表 + §7 gap 可直接转写）；引言 ¶1/¶5 已有 cite（`gunnarson2021ncomm` / `verma2018pnas` / `levine2020offline` / `haarnoja2018sac` / `fujimoto2021td3bc` / `tarasov2023rebrac` / `park2025fql` / `pinto2017asymmetric`）作起点。**refs.bib 待补（wake-relevant 子集，落地时转 bibtex）**：核心 = Coombs 2020[1] / Liao 2022[2] / Li 2020[3] / Colabrese 2017[13] / Qiu 2022[16] / Zhu 2022[17] / Feng 2024[18] / Jiao 2025[20] / Mecanna 2025[21]；可选鱼类机制 [4][5][6][7]；广印证少量取鸟[19]·浮游[9 或 10]。约 8–12 篇，**不全搬 22 篇**（§0.5.4 定性不预设、wake-relevant 优先）。
- **voice / 篇幅**：§0.5.2 dissertation 学术体（有论点的 positioning，非 literature dump）；1.5–2.5 页量级，避免膨胀为第二个综述章（§0.5.7）。RW 用**领域层**术语，具体平台 / 传感硬件留 §5.3 setup。
- **caveat（零跨章自足）**：本节是全章自足的背景锚点，不引述博论其他章的综述章；术语首次出现 1 句定义 + 1 cite（§0.5.1 reader model）。
- **实施细节子节**：本节无 §5.2.m（背景节不需“复现”段，与 §5.1 同）。
- **落地状态（rev.2 推倒重写，2026-06-07，方案甲）**：§5.2 成为生物 landscape + RL 谱系 + 算法背景 + gap 的**唯一充分展开处**（`related_work.tex` rev.2，4 段约 2 页）：¶1 能量场/信息场二分 + 鱼类充分展开（涡街/障碍尾流/群体涡相位 + 侧线“空间分布式”阵列、反衬 AUV 单点）+ 鸟·浮游印证；¶2 RL 复现谱系（奠基 colabrese/verma → 局部感知 gunnarson/jiao → 具身 zhu/feng → 算法评估 qiu/mecanna）；¶3 offline 行为约束谱系（弱→强）+ POMDP 训练-部署不对称/特权信息（描述性、不裸写算法专名、不展开公式）；¶4 gap 三支精确区分 + 坐标 + forward pointer（第三支收敛“训练与部署阶段均不借特权”，避 asymmetric-AC 稻草人）。生物/RL 谱系在 §5.1 仅一句起兴、此处唯一展开；两条主线复述不在本节，留 §5.1 ¶6。重复已消除。

### §5.3 Problem description（任务 / 环境 / sensor / reward / 数据集）

- **写什么**：AUV/wake 任务定义 → deployable `s0`（DVL 单点）vs privileged hull-integral `[u_eq, v_eq]`（critic-only）→ **reward 双轨**（efficiency_v2 / arrival_v2）→ **数据集 statistics**（来源 / size / regime / collector policy）→ 评估协议（manifest / Welch / paired-mean / rule-of-three）。本章统一 setup，后续三线共享。
- **数字/概念源**：[`environment_design.md`](../docs/environment_design.md)；state-space 速查 [`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §5.2（obs 维度：efficiency_v2 s0/h4=40-D；arrival_v2 s0/h4=48-D；privileged_obs dim=2 不堆叠）。
- **复用资产**：paper 1 `sections/setup.tex`（rev.2，99 行，含 dataset matrix 表 + 5 子节）+ **Fig 1 sensor schematic**（已出 pdf/png）。
- **本章新增**：需把 setup 从"单一 canonical 协议（cross_u10/efficiency_v2）"**泛化**到覆盖三线用到的多个 regime（efficiency_v2 ↔ arrival_v2 reward；Re150/u10 sub-critical ↔ Re250/u15 critical；s0/s1/s2 sensor；SAC vs baseline collector）。这是统一三线的第一处缝合工作。
- **caveat**：reward 已从 `efficiency_v2`（paper 1 / TD3+BC 历史）迁到 `arrival_v2`（broad-val v2 / FQL）；两者 obs 维度不同（40 vs 48），setup 必须显式交代两套并存，否则后续表格维度对不上。
- **⚠ 措辞锁（2026-07-26 第 5 批，findings §10 N13 ＋ §9 N8/N7）——§5.3.6 是全章两条口径规则的定义之家**：
  - **"终检评估"只标评估时机**（训练后 ↔ 训练中的"周期评估"，即 rev.4 立规原意）。原定义"对**最终检查点**重新评估"与实际协议不符：GT `td3bc_phase0c_experiment_report.md` §3.4 的统一规则是**验证集字典序选点、独立测试集评估**，§5.6 与 §5.7 均按此执行，而 §5.5／§5.8／§5.9 确为训练终点口径 → 章内同名不同义。已改为：终检=训练结束后对**该次训练选定的检查点**重新评估；**检查点身份是另一条轴**（训练终点 ↔ 验证集字典序选点，§5.4.m 立规），**各结果节须在实施细节中标明所用规则**。⚠ 不采"协议之家新增两个口径名 + §5.6 全节改名"（连锁广、须造新名词），亦不采"§5.6 就地加说明"（修不好 §5.7）。⚠ **§5.4.m 必须同步**（第 5 批初稿漏此，经复审判为 HIGH）：§5.3.6 的新定义把「检查点身份」这条轴**指回 §5.4.m**，而 §5.4.m 原写「各方法训练内的检查点**统一**按……字典序选择」这一**全称规则**，被 §5.5.m／§5.8.m／§5.9.m 三节明文否定；已改为「按两种规则之一确定，由各结果节按其考察目的选取并在实施细节中标明」。
  - **等价判断规则的登记语**：§5.3.6 的容忍范围条款是**条件规则**。⚠ **第 5 批初稿曾写「本章各结果节均未作此类判断，该附加要求在全章不予触发」，经四维度复审判为 CRITICAL 级假陈述并已重做**——§5.9.1 的数据条件矩阵**确在作此类判读**（表 `tab:ch5_fql_matrix` 判读列有两行「无差异」，正文亦写「干净多模态单元恰为无差异」），且其容忍带 $|\delta|\le0.03$ **确系预登记**，即该附加要求是**已触发且已满足**，而非未触发。定稿登记语为「本章仅 \S\ref{subsec:ch5_algo_matrix} 的数据条件矩阵作此类判读，其实践容忍范围已在该节实验执行前登记；其余各结果节均只报未检测到统计显著差异，该附加要求在其上不予触发」。**教训**：为使某节的新框架自洽而在协议之家写下的**全称否定**，极易被同章别节的表格判读列当场反证；凡在定义之家写全称句，须先对全章逐处检索该行为是否存在。
- **§5.3.m 实施细节与可复现性**（NO APPENDIX 融入）：承担 paper 1 App A reproducibility + App G obs spec + dataset matrix 表；evaluation manifest JSON schema + Welch / paired-mean / rule-of-three 计算式 + seed 选择协议。

### §5.4 强化学习方法与算法框架（methodology — rev.11 新增）

- **写什么**：统一介绍本章用到的强化学习方法，而不是分散到后续结果节反复补课。建议小节结构：§5.4.1 MDP/POMDP 建模与 online/offline 区分；§5.4.2 异策略 Actor-Critic 框架与 SAC/TD3 类方法；§5.4.3 TD3+BC 的行为克隆约束；§5.4.4 ReBRAC-Q 的 Q-normalized dual penalty 与 β1/β2；§5.4.5 FQL 的 flow-matching 策略先验；§5.4.6 部署观测、特权 critic 与参照分布的统一信息协议。
- **不写什么**：不提前给主要结果数字；不判断哪种算法更优；不写"本章不是方法创新"或"只是使用已有算法"这类自我降格句。正式正文只陈述本章采用何种方法框架、每种方法约束什么信息、损失函数如何定义、后续比较为何公平。
- **复用资产**：paper 1 `method.tex` 的 ReBRAC-Q 损失定义与 Q-normalization；TD3+BC phase0c report 的 baseline loss 口径；FQL writing index / P2 report 的 flow-matching teacher 定义；`sac.py`/`td3bc.py` 仅作实现口径核对，不把代码名写入正文。
- **术语红线**：首次定义采用中文主体 + 英文括注；异策略、策略网络(actor)、价值网络(critic)、行为克隆惩罚项、参照分布、特权观测、部署观测等术语在本节统一立规。`teacher` 不作概念主语，改写为"以特权信息在线训练所得策略（用作性能上界/参照）"或"模仿目标/参照分布"。
- **叙事作用**：本节为 §5.5–§5.9 提供算法地图。§5.6 只需调用 TD3+BC 的已定义损失解释 baseline 瓶颈；§5.7 只需说明 ReBRAC-Q 相对 TD3+BC 的增量；§5.9 只需解释 FQL 相对 ReBRAC-Q 的策略先验差异。
- **§5.4.m 实施细节与可复现性**：共同符号表、损失函数汇总表、核心超参数含义（α/β1/β2/temperature）、训练/部署信息可用性表、方法原文献引用清单。具体 seed、manifest、dataset 和结果统计不放在本节，留给各结果节。

### §5.5 Online RL：可学性与信息瓶颈
- **写什么**：(1) A0 sensor screen — deployable s0 在简单 wake viable；(2) arrival_v2 prototype — H_information-bottleneck（s0+history k=12 闭合约 44pp 的 s0→s1 gap，2026-07-19 九宫格重排后值）+ manifest universal-floor 概念。
- **数字源（thesis-grade）**：[`online_rl_line_summary.md`](../docs/online_rl_line_summary.md) §1.1（A0：efficiency_v2 s1 0.967 / s0 0.789 / s2 0.856）；[`arrival_v2_experiment_report.md`](../docs/arrival_v2_experiment_report.md) §7.9 / §7.9.7（k=12 在 2/3 seed saturate manifest floor 27/30=0.900；3-seed σ_final=0.038 << target 0.10）。
- **复用资产**：无 LaTeX，**全新写**。可新作 figure（A0 三 sensor bar + arrival_v2 k-monotonicity 相位跃迁图）。
- **叙事作用**：为全章奠定"瓶颈在 actor 信息访问，不在传感器硬件"的基调，**为 §5.8 actor-fundamental ceiling 埋伏笔**（online 难流场 k=4 参照约 0.46 且跨种子剧烈分化、k=12 可稳定闭合 ↔ offline k=4 N2' 0%）。⚠ 落地正文（online.tex rev.7，2026-07-19 九宫格重排）已把在线 k=4 参照定为 0.46±0.39（2026-06-18 转录值 0.26 系引用错误、已作废；ground truth = arrival_v2 report §7.10），并在 §5.5.5 加了 k=4 scope 保险（离线只能用 k=4 既采数据、k=12 救援能否迁移属 §5.8 待答）；§5.8 起草的 online↔offline 对照须用 0.46、且承接此 k=4 scope。
- **⚠ 措辞锁（2026-07-24 第 4 批，findings H7 / M1）**：
  - **单调性的承载方式**：§5.5.4「排除初始化局部极小与优化噪声」这一论证，**重心在同一初始化下的跨历史长度对照**（seed 0：k=8 停在 $0.500$ → k=12 被推到 $0.900$，初始化与其余条件全部固定），即 GT `arrival_v2_experiment_report.md` §7.9.2 F2.3／§7.9.3 所用的原论证。**不得再挂在「单次训练内部沿 k=4→8→12 重现」上**（选择性呈现），**也不得改挂在 σ 收缩上**——第 4 批初稿曾如此改，经复审推翻：σ 只度量症状被清除、不识别病因，且末端受**天花板混杂**（k=12 三值有两个压在 0.90 上界，均值逼近硬上界时 σ 必然收缩，属均值抬升的导出量）与**评估噪声地板**（30 回合、p=0.9 的二项抽样 std $=0.055$ 已大于所报 σ $=0.04$）双重限制；σ 只可作稳健性旁证。逐种子终检实为 seed 42 `0.10/0.90/0.90`、seed 0 `0.40/0.50/0.90`、seed 7 `0.867/0.867/**0.833**`——**三次中两次单调、一次高位持平后微降**，且 k=4 最好种子（0.867）**高于** k=12 最差种子（0.833）。图 `fig:ch5_online_monotonic` 已改画**三条逐种子轨迹**，不再前置单条最强种子（原「单次代表性训练」即 seed 0，是三者中唯一严格单调者）。「单调上升」只可用于**聚合均值**（0.46→0.76→0.88，属实），不可用于单条轨迹的全称陈述。
  - **特权 critic 的评估口径**：§5.5.3「不闭合差距」结论取**周期评估**口径（曲线与峰值）。终检口径须按**配对两种子**陈述——seed 42 为 $0.100$ vs $0.167$（特权高）、seed 0 为 $0.400$ vs $0.200$（特权低），**配对均值 $0.250$ vs $0.184$、特权仍低约 $6.6$pp，方向与周期口径同向、并未相反**。⚠ **不得只取 seed 42 说「方向相反」**（第 4 批初稿如此，经复审 A/B 双命中并由本机 `final_eval.json` 实读证伪；诚实版本反而更强）。引用该结论时须携带口径限定；`discussion.tex` §5.10.3 与表 5.21 第 1 行用「收敛峰值」即周期口径，与之一致。⚠ **2026-07-26 第 5 批订正（N14）**：表 5.21 第 1 行的证据强度列写「峰值读数取周期评估口径，**终检口径下方向同向**」，**不得**写成「结论取周期评估口径」——§5.5.3 明写「差距未被实质缩小这一判断因而**不依赖于所取口径**」，整行限定为周期口径会与源头矛盾并低报证据强度。
  - **消融检验力的边界（2026-07-26 第 5 批，findings §10 N16）**：§5.5.3「差距并不出在价值网络的估计精度」由**一个**特权 critic 变体的零结果推出充分性结论，严格强度只到「**这一条**改善价值一侧信息的途径不闭合差距」→ 已改首句为「改善价值一侧信息这一途径并不闭合差距」并**删去后半的充分性断言**（「瓶颈落在策略一侧」由随后的时序消融**正面**承担）；口径 caveat 段末补**仪器效度轴**登记，与 §5.8.5 已登记的同一前提**措辞同构**（非对称通道能否把价值侧信息传导至策略改进未经检验）。三可能归宿句同步为「向价值网络补充信息这一途径由反向消融排除」。**不得改回「差距并不出在价值网络的估计精度」这类充分性表述。**
  - **生物母题中段一次回声（CH3 已裁断采纳，2026-07-26）**：§5.5.3 机制段末句以一个从句把「空间分布式感知阵列 ↔ AUV 单点」对照点破一次，四护栏全守（一句级、机制/motivation 层、不新增 claim、不带生物专名——写「沿体表分布的感知阵列」而非「侧线」）。§8 #10 的生物护栏因此为「§5.1/§5.2 引子 ＋ **§5.5.3 一个从句** ＋ §5.10.5 一次回扣」，**不再扩大**。
  - **§5.5.4 训练计数（第 5 批 N15）**：manifest 经验上界的实证基底是 GT §7.9.7 的 5 run = $k{=}12\times\{42,0,7\}$ ＋ $k{=}8\times\{7,42\}$，只涉 **3 个**随机种子 → 正文写「共五次训练（涵盖三个随机种子）」，**不得**写「五次不同初始化」。
- **caveat**：reward preset sweep（27 run×200k）**绝对数字不可引用**，仅作 `efficiency_v2` 选择的方法学依据；Sprint 0 preflight 不可作性能基线。坑（200k 假性否决 / flow 不一致 bug / efficiency_v2 OOB-suicide）可入方法论或 limitations。
- **§5.5.m 实施细节与可复现性**：A0 sensor screen 3-seed 协议 + arrival_v2 prototype k-sweep 协议 + manifest universal-floor 计算 + reward preset sweep 方法学（数字不引、仅作 viability 论证）；online §7.7 AsymCritic ablation 复现细节（§8 开放点 4 已决：cross-line 旁证留 §5.5 散文呈现，不与 §5.8 并列）。

### §5.6 Offline baseline：TD3+BC 拆瓶颈
- **写什么**：TD3+BC 把"数据越多越差"从协议伪象**纠正为真实现象**，拆出两个瓶颈：(1) 数据支持集结构（"2000<1000"退化在纯 BC 下同样成立）；(2) deployable critic 信息瓶颈（privileged-critic 关 48.5% gap）。→ 为 ReBRAC 创造问题设定。
- **数字源**：[`td3bc_phase0c_experiment_report.md`](../docs/td3bc_phase0c_experiment_report.md)；headline（[`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §1）：crosscomp 500/1000/2000 = 0.592/**0.672**/0.596；worldcomp deployable 最优退回 α≈0（纯 BC 0.858）/ privileged 0.922 / teacher 0.990。
- **复用资产**：paper 1 把 TD3+BC 仅作 baseline 行引用，**无独立节** → 本章需扩写成一个短节（dissertation 容量允许讲完整 baseline 故事，§0.5.3 structural depth）。
- **⚠ 措辞锁（2026-07-24 第 4 批，findings H6 / M12）**：
  - **noisy-support 的机制归属**：加噪展宽支持集**只在两千回合上有益**（纯 BC $0.6000\to0.7375$），**在一千回合上是强破坏**（BC $0.6750\to0.5125$、TD3+BC $0.7625\to0.5125$；GT §6 原话「`1000` 上加噪声不是小扰动，而是明显破坏了数据可学性」）。正文须两面同报，**不得**写成「展宽支持集 ⇒ 纯模仿受益」这一通用机制。⚠ 归属只可写到**否定性结论**（「展宽支持所付的代价在样本本就稀疏处压过其收益，该受益并不普遍成立」）：**不得写「特属于两千回合的支持集结构」**——第 4 批初稿如此，经复审推翻，理由有二：① 所给机制的前提（确定性采集器在相近状态给出近乎相同动作）在两个规模上**完全相同**（同一采集器），故机制**预言同号**、实测反号，「特属于」是给例外命名而非解释；② 两千侧的改善（BC $+0.1375$）**不足一个种子间标准差**（其确定性基线 BC $=0.60\pm0.15$，GT §5.2 Stage B），而一千侧的反证（$-0.1625$／$-0.2500$，基线 std 近零）远更扎实，机制归属**不可挂在弱腿上**；两千侧措辞相应由「明显上升」降为「上升」。两侧读数同属 GT §4.6 的小规模筛查口径（两随机种子 42/43、验证与测试各 40 回合），caveat 须**覆盖新补的一千回合读数**，并须声明**绝对水平不与正式主表逐格可比**（同一格 BC@1000 主表 $0.588$ vs 筛查约 $0.68$，相差约 9pp）。
  - **§5.6 开篇对 §5.5.1 的引用强度**：⚠ **改挂常量轴**——$k$ 在本节三个数据规模上**固定不变**，按混杂因素的定义（随被比较因素一同变化的第三变量）**本就不可能混杂规模间比较**；正确写法是「历史长度不随所比较的因素变化，故非混杂，而是**界定本节结论适用范围**的一重设定」，§5.5.1 只作该设定选取合理性的支持。**不得**写成「故 §5.5.1 排除了历史长度混杂」（理由错），**也不得**只把它削弱为「旁证 + 未直接排除」（第 4 批初稿如此，经复审推翻：弱化过度且方向错，还堵掉了「§5.5 说 k=4 是瓶颈、为何离线全跑 k=4」这一外审必问的现成答案）。⚠ findings M12 原病因描述「§5.5.1 是**另一奖励**」**经实核为误**——§5.5.1 与 §5.6 **同为历史效率导向奖励**。
  - **采集器参照的两套口径不得互换（2026-07-26 第 5 批，findings §3 M2 / §8 N4）**：诊断期回查钉死——`scripts/summarize_offline_phase0.py:241` 的 `reference_baseline_success_rate` 取自 `<dataset>/baselines/<policy>_final_eval.json` 的 `eval_success_rate`，故 GT Stage C 主表的 `baseline success = 0.95`（crosscomp）与 worldcomp 的 `0.990` **同属"采集器在固定评估集上直接执行"口径**；而各数据集 `metadata.json` 的 `success_rate`（crosscomp 0.864/0.870/0.886、worldcomp **0.958**）才是**数据集采集成功率**。→ 成稿此前把 worldcomp 的 $0.990$ 五处称作"采集器**自身轨迹**的成功率"**是事实错误**，已全部改为"在同一（固定）评估集上直接执行的成功率"；fig `ch5_rebrac_scale` caption 中唯一一处**跨口径**比较断言（"保持在采集成功率参照以上"）已撤下。⚠ **本章不引入 0.95 与 0.958**（用户 2026-07-26 裁定）：引入其一即须引入其二，而本章不作与采集器的性能比较，登记数值只会引出未承担的论证。⚠ 差距闭合比例 53.0%/48.5%/57.6% 的**数值不变**，改的只是分母的名称。⚠ **口径订正须覆盖 §5.9**（第 5 批初稿漏此，经三个维度同时判为 HIGH）：§5.9.4 正文「专家档三配置全部**收敛到采集器自身水平附近**」与图 `ch5_algo_interaction` caption 的「采集器参考水平」，与 §5.7 刚撤下的「保持在采集成功率参照以上」是**同型的跨口径比较**（SAC 四档 0.002/0.515/0.751/0.899 为采集口径，各配置 0.889–0.911 为 30 回合固定评估集终检）；已改为纯节内表述（「彼此之差与各自的跨种子标准差同量级」）＋ 图 caption 加口径隔离声明，全节「采集器自身成功率」统一为「数据集采集成功率」。§5.9.5 的「约 58 个百分点塌落」原为跨口径减法（$0.719-0.14$），已改用同口径的 $0.700$ 复算为**约 56 个百分点**。
- **叙事作用**：建立"为何需要 dual-BC penalty"的动机，自然过渡到 §5.7。
- **§5.6.m 实施细节与可复现性**：TD3+BC 增量设置 + α-sweep 协议 + 同预算纯 BC 平行对照 + noisy-support 筛查 + crosscomp 500/1000/2000 数据集构造 + worldcomp deployable/privileged α-optimum 报告。⚠ ground truth 无"nearest-action 距离度量"（旧条目已删）：支持集结构诊断由纯 BC 平行对照 + noisy-support 筛查支撑，落地 td3bc.tex 已据此改写。TD3+BC 损失函数基础定义已在 §5.4 给出，此处只保留本节实验所需细节。

### §5.7 ReBRAC-Q 主线：方法增量 + 四 findings（**本章重心**）
- **写什么**：在 §5.4 已定义 ReBRAC-Q 损失的基础上，说明相对 TD3+BC 的关键增量、实验协议和四条 paper-level finding。避免把共同方法框架重复写成一个长 method paper。
- **四 findings 数字锚**（实时回查 [`rebrac_experiment_report.md`](../docs/rebrac_experiment_report.md)；镜像见 paper 1 progress.md §3）：
  - (i) crosscomp +23.0pp（1000）/ +32.2pp（2000），翻转"more data hurts"，std 不增反降
  - (ii) **deployable 0.928 与 privileged-critic 0.922 之间未检测到统计显著差异**，Welch p=0.9195（最强 sim2real narrative）。⚠ **措辞锁（2026-07-22 第 3 批整改，H2）**：禁「追平／持平／相当」，种子级 95% CI = [−0.127, +0.139]（半宽 13.3pp，为 §5.6 约 6pp 抬升的两倍以上），§5.3.6 的等价判断附加要求**不予触发**；肯定回答须锚在可部署一侧自身的正向量（相对纯 BC +7.0pp、差距闭合 53.0%、点估计不低于特权参照），**不得挂在未检出结果上**
  - (iii) dual penalty cross-dataset 必需（β2=0 时 mean_target_q 漂 +46%/+98%）
  - (iv) critic LayerNorm ⊥ dual penalty（LN-off −16.2pp，远超 β2=0 的 −2.4pp，方向相反）
- **⚠ 与 §0 dispensability 命题的接口（写 Finding (iv) 时严守，防自相矛盾，cross-ref §0.4 红线 1）**：critic LayerNorm 是**第三条独立的"表征稳定性"轴 = 必要基础设施**，与 §0 中心命题反向轴中的"向价值网络注入特权观测"一项**正交**——它属"必要基础设施"，**不**进反向轴"增添能力非必需"清单。两条配套红线：(a) mean 归因只在 **actor β1 vs critic β2** 之间成立（β1 carry mean、β2 carry Q-stability），**不可**升级为"actor 侧决定全部 mean"（LN 才是单项最大 mean 杠杆）；(b) LN claim 强度 n=2 seeds 仅支撑 "necessary component 存在性"，**不 claim "LN 比 dual penalty 重要"**（review §2.2.4）。
- **复用资产**：✅ **paper 1 主干大幅可搬** — `method.tex`（rev.2，134 行）中共同算法定义前移 §5.4，本节保留 ReBRAC-Q 增量与训练口径；`experiments.tex`（rev.2，233 行，含 4 finding 子节 + 主表 + 2 ablation 表）+ **Fig 2 seed dotplot** + **Fig 3 Q-drift**。
- **关键命名红线**：全文 **ReBRAC-Q (ours)**，§5.4 首次定义 + §5.7.4 差异表（β1=4.0 ↔ TD3+BC α≈0.25）；不裸写 "ReBRAC"。
- **finalist**：`(β1=4.0, β2=2.0)`，是 **robustness winner（救 seed 44）**非 peak winner — 这是 §5.10 β1 reconciliation 的前提，§5.7 须埋点。
- **§5.7.m 实施细节与可复现性**（NO APPENDIX 重头戏 — 集中承担 paper 1 App A–G 中 ReBRAC-Q 主线部分）：
  - paper 1 App A reproducibility（5-seed 协议 / Welch / paired-mean / rule-of-three / robustness winner 选择规则）
  - paper 1 App B 超参表（actor/critic lr / batch / γ / τ / β1=4.0 β2=2.0 / Q-norm 实现 / LN 位置）
  - paper 1 App C 全表（worldcomp / crosscomp 双数据集 × 5 seed × β1×β2 sweep）
  - paper 1 App D 统计细节（Welch t-test 自由度 / paired-mean Δ priv−dep CI / Q-drift +46%/+98% 计算式）
  - paper 1 App E dual-penalty derivation（actor β1 carry mean / critic β2 carry Q-stability 的损失梯度分解）
  - paper 1 App F 学习曲线全图（5-seed 收敛轨迹 + Q-drift bar）
  - paper 1 App G obs 24-D / 40-D 详细 spec
  - 该实施细节子节是全章最长子节之一（论证完整性，§0.5.8）；不"放在最后"，紧接 §5.7 四 finding 后。

### §5.8 泛化边界：broad-val v2 → actor-fundamental ceiling
- **写什么**：broad-val v2 两 cell — N0（sub-critical）HOLDS；N2'（critical Re250）STRONG_NEGATIVE → **deployable s0 actor-fundamental partial-observability ceiling**；asym-critic ablation 排除 critic-fundamental 解释。
- **数字源**：[`rebrac_broad_validation_v2_report.md`](../docs/rebrac_broad_validation_v2_report.md) §3–§5（N0 0.878 HOLDS（3-seed，2026-07-12 补种子后；旧 2-seed 值 0.850 作废）/ N2' 0.000 STRONG_NEGATIVE，低于在线 k=4 参照水平约 0.46）+ §4.5（asym-critic ablation，verdict **ACTOR_FUNDAMENTAL_CONFIRMED**：完美 hull-integral flow 喂 critic，s0 actor 仍 0.000）。
- **复用资产**：paper 1 Phase 6 已把这部分折叠为 `discussion.tex` §6.6 + `limitations.tex` L12 → 本章**提级为独立节**（dissertation 容量允许，§0.5.3 + §0.5.6 负面结果作 finding 不降级 limitation）。
- **叙事作用（§0.5.6）**：这是本章**最强负面 finding**，与 §5.7 正面 finding 同台呈现统计 evidence（n / verdict / CI），**不**降级为 limitation 段；mechanism explanation 与正面 finding 同等深度（"为何会失败 = 也是机制贡献"）。
- **⚠ 答辩风险登记（复审维度 G/A 升级，2026-07-03）**：N2' 边界仅 **2 seed**，却是全章**最强负面结论**，且 §5.5 已为其埋了在线↔离线 floor 伏笔（放大暴露面）。原 §4 分级把它列为"带 caveat 引用"**低估**了其答辩风险——**答辩前须补种子，或以远重于常规 report §6.1 的 caveat 呈现**（显式标 n=2；asym=0.000 兼容"表征不可能"与"asym 机制未把信息转给 actor"两读法，守 claim 边界红线、不下"信息论不可能"）。此项不阻塞 §5.7 起草，但须在 §5.8 动笔前决定补种子与否。
  **✅ 风险降级（2026-07-12）**：+seed 43 × 3 单元同协议补种子完成并落稿（`boundary.tex` rev.3）——N2'/asym 三种子 0/90（零计数上界 ≈0.033），零成功结论不变；N0 三种子 0.878±0.051，第三种子（0.933）高于主线锚点、「两种子同向退化」叙事撤销（v2 report §2.4 已改写）；种子组 {42, 0, 43} 与预登记 {42, 43, 44} 不重合已在 §5.8.5/§5.8.m 如实披露。残余暴露面 = 三种子仍为全章最少 + 种子组与预登记不重合，均已成稿显式披露。判读与裁决全档：`docs/rebrac_broad_validation_v2_seed43_supplement_plan.md` 附录 B。
- **⚠ claim 边界红线**（[`fql_succession_paper_writing_index`](../docs/fql_succession_paper_writing_index.md) 同源原则，broad-val v2 report §4.5 明确）：
  - ✅ **可下**："privileged-flow critic 不能挽救 N2' 天花板；**排除 critic-fundamental**；HARDENS actor-fundamental"
  - ❌ **不可下**："proven actor-incapable / s0 actor 信息论上不可能" — asym=0 同时兼容"表征不可能"与"asym-critic 机制没把信息转化给 actor"两种读法。
- **§5.8.m 实施细节与可复现性**：broad-val v2 manifest（u15_cross / u15_upstream）+ N0/N2' 3-seed 协议（{42, 0, 43}，与预登记组不重合披露）+ asym-critic ablation 实验设置（特权流喂 critic 通道，actor 保持 s0）+ verdict ACTOR_FUNDAMENTAL_CONFIRMED 推导链 + N2' 与在线 k=4 参照水平（约 0.46，非 legacy 10%）的对照（§5.5 online ↔ §5.8 offline 比对表，须承接 §5.5.5 的 k=4 scope）。

### §5.9 算法对比：FQL vs ReBRAC-Q — 先验表达力 vs anchor 目标质量
- **写什么（两层结论，必须按此顺序，否则会过头）**：
  1. **privileged-collector regime（FQL P2）**：核心 conditional-iff 假设（"FQL > ReBRAC iff sub-optimal AND multi-modal"）**证伪** → discriminator 是**噪声**非 modality（E-multi NULL）；机制三连 Q1→Q1b→Q1c（critic 侧排除 → actor β1 4→1 noisy **+23.5pp** → β1=1 clean 也更好）；C-1 给 FQL 自己的 anchor 旋钮做公平复赛 **RESCUE-FAIL**。**该 regime 内**：单一 ReBRAC β1=1.0 双轴 dominate FQL（worst-case 0.910 > 0.858）。
  2. **跨数据源 refine（SAC collector，最新，2026-05-26）**：把 1 的 "ReBRAC dominate" 放到 RL-trained behavior policy 数据源上**会翻转** → 算法排名是 **regime-dependent（algorithm × data-quality interaction）**，不是全局。**这才是本节的章级 headline**（不是 1 的 "ReBRAC dominate"）。
- **统一机制（贯穿两层，喂 §5.10）**：offline RL 性能由 **BC-anchor 目标质量与数据 regime 的匹配**决定、**不**由策略类先验表达力决定——中质量噪声数据 FQL 的 flow-denoised teacher 更优、高质量 clean 数据 ReBRAC-Q 的 raw-action anchor 更优。
- **数字源**：FQL P2 = [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) + [`..._mechanism_diagnostic.md`](../docs/fql_succession_p2_mechanism_diagnostic.md) §9（cheatsheet [`..._paper_writing_index.md`](../docs/fql_succession_paper_writing_index.md) §2）；SAC collector interaction = [`arrival_v2_sac_collector_design.md`](../docs/arrival_v2_sac_collector_design.md) §4.0.10（SAC mexp **FQL 0.922 vs ReBRAC β1=1 0.800, Δ+0.123 CI[+0.022,+0.233]**；m_multi_mix 反向 +0.035 CI[+0.010,+0.065]）。
- **⚠ 口径红线**（§0.4 红线 5）：mexp 反超是**同源**（sprint1+2 同 collector/manifest）干净比较；"方向翻转"的 ReBRAC>FQL 一侧落在**异源** m_multi_mix（不同 dataset/manifest），cross-source 仅 **direction robust、magnitude 不可比**。interaction 只 claim 方向，不 claim 幅度比；**不要把 FQL P2 的 "ReBRAC dominate" 当章级 headline**。
- **⚠ 措辞锁（2026-07-26 第 5 批，findings §9 N7/N8 ＋ §10 N12 ＋ §3 M5/M10/M14/M17）**：
  - **族内判读是"证伪"而非"非劣判定"（N7，本批最实质）**：`fql_succession_p2_main_spec.md` §5.4 把 $\delta_{null}=0.03$／$\delta_{signal}=0.05$ 登记为**冻结配置矩阵**（ReBRAC β1=4.0 / FQL α=1.0）的 per-cell 判据，判读表只列 E-uni / M-uni-noise / M-multi-mix 三格，**不覆盖 C-1 跨配置族内对照**（GT §4 给 C-1 的判据是 "No α beats the frozen 0.858" = RESCUE-FAIL，§5 明写 "falsifying a superiority claim only requires a non-win"）。→ §5.9.3 **不作性能相近或等价判断**，只作对"生成式先验在此族上更优"这一主张的**证伪**；就地写明预登记容忍带的登记范围。**不得事后补容忍带**（本章首次引入的判定装置，且与 §5.7.2 的"不作等价判断"构成章内双标）。
  - **删「最坏情形严格更稳」**：该断言的支撑是 $+0.052$、$t\approx1.2$ 的 NULL（GT §5 第 6 行），点估计更高但统计上不可分辨 → 改"亦以 ReBRAC-Q 一侧为高，而该差异与其余头对头差异一样未过检出门槛"。
  - **饱和端的肯定回答**：⚠ **第 5 批初稿曾改锚到「跨源配对区间严格不含零」这一正向读数，经复审判为 CRITICAL 并已重做**——该区间以**回合**为重采样单位、仅两个共同种子，而 §5.3.6 明文把回合级自助降格为敏感性复核、§5.7.2 更已先例性地**拒用**同类读数改用种子级；且 GT `arrival_v2_sac_collector_design.md` §4.0.10 显示**冻结配置**（$\beta_1{=}4.0$）在同一单元同一把尺下亦给 $+0.035\,[+0.005,+0.070]$，即「灰区 ↔ 严格不含零」的分水岭是**推断单位**而非配置。定稿改为：§5.9.5／§5.10.1 写「直接模仿在高质量单元上**未被生成式先验超过**，在已然饱和的混合单元上**方向读数还偏向**直接模仿」，**不写「占优」**；正向承载交给 §5.9.4 的同源中段（$+0.123$，三种子分层配对自助）。§5.9.4 的 M5 区分句同步改为点明**推断单位**才是强度落差的来源。
  - **N8**：§5.9.4"算法差异**被数据上限吞没**"→"三者相互之差与各自的跨种子标准差同量级，算法差异在此被数据质量的上限**压平**"。
  - **N12**：全节非劣性措辞统一为"未被 FQL 超过"；ReBRAC-Q **内部** β1=1.0 对 4.0 的比较不写"不低于"——该处证据为严格为正区间（$+0.089\,[+0.011,+0.178]$），禁用词反而低报。
  - **M14 噪声注入次序**（`p2_collection_log.md` §1.3/§1.4 实核）：小噪多模态的 $\sigma{=}0.1$ 在**两个分量各自采集时**注入（privileged-500 seed100 / goalseek-500 seed200 均 `action_noise_std=0.1`），再按回合级 50/50 合成；**不是**对干净混合事后加噪。**M10**：Sprint 0 collection log 只含三集 → 干净多模态（E-multi）确为**后补**，预登记只覆盖三格，正文不得写"四项预测"。
  - **M17**：$0.985/0.719$ 为一千回合**采集时**实测成功率，与 §5.8 所报同一采集器在临界固定评估集上**直接执行**的 $0.700$ 口径不同，须就地说明。
- **复用资产**：verdict notebook 3 图已出（matrix / noise-axis / c1-rescue）；**Method prose / Related Work / Abstract 待写**（writing index §6 TODO，按 §0.5.2 dissertation voice 起草，不复用 paper 2 abstract）。
- **定位对接**：writing index §5 已预留**定位 (c) = thesis 章节 subsection**，与本方案一致 → FQL 作为本章一节，Method 与 §5.4/§5.7 共享，正文只讲 FQL 相对 ReBRAC-Q 的增量。
- **caveat**：跨-benchmark 探测在 `single_u15_cross` 撞 s0 observability **FLOOR**（clean priv 0.719 / noisy 0.098）→ scope caveat，不弱化 u10_cross 主结果。
- **§5.9.m 实施细节与可复现性**：FQL 方法增量（flow-matching teacher 损失全式，与 §5.4/§5.7 ReBRAC-Q dual-penalty 共享共有部分）+ FQL P2 2×2 modality×noise matrix 协议 + Q1/Q1b/Q1c 机制三连 ablation 设置 + C-1 RESCUE-FAIL 协议 + **SAC collector 39-run cross-source 设计**（sprint 1/2 collector / manifest / regime / 5×β1 sweep）+ §4.0.10 cross-source magnitude caveat 计算 + verdict notebook 3 图复现。

### §5.10 统一讨论 + 跨线 reconciliation + limitations + 本章小结
- **写什么**：(1) 统一机制（actor 信息 + BC-anchor 目标质量两轴）；(2) **β1 跨线 reconciliation**（paper 1 用 β1=4.0、FQL 线用 β1=1.0 不矛盾）；(3) sim2real implication；(4) limitations；(5) **本章小结**（dissertation 章末承上启下段，§0.5.3）。
- **β1 reconciliation 源**：[`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §6（四轴差异：reward / collector / 动作噪声 / seed pool；原线 β1=4.0 本是 seed-44 std-driven 选择）+ [`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §2.2.6 + [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) §8.1。**这是单章统一三线相比两篇独立 paper 的最大叙事增值点** — 在一章里能把"同一 BC-anchor 旋钮、相反最优点、由 anchor target 是否带噪决定"讲成统一机制。
- **本章小结**（§0.5.3 dissertation 体裁要求 — 不是 conference paper 的一句 punchline）：(a) 复述 §0.3 统一 takeaway；(b) 与博论整体 spine 衔接（本章为博论提供 deployable-sensor offline RL 的可行性证明 + 边界 + 算法决定因素）；(c) **自足的章级承接**（rev.7 零跨章：以抽象口径点出本章命题在博论整体论证中的位置，**不点名引述具体后续章**；与 §0.5.3 §5.10 小结、§2 §5.1 章级承接对称）；(d) **正面声明贡献类型（复审维度 G/H-2 新增，2026-07-03）**：本章"科学洞见非新算法" + §0.5.9(a) 禁反向宣告，若不加处理会在贡献栏留白，中文工科答辩"方法创新点"必被追问。§5.10 须把贡献**正面锚成类型**——部署约束下信息利用的**规律 / 设计准则** + **受控判别框架** + **双侧行为约束方法（即 ReBRAC-Q）这一可复用方法制品**。⚠ 落地措辞守 §0.5.10 抽象下界：`discussion.tex` §5.10.5 与 `intro.tex` ¶6 均写「Q 归一化双侧行为约束方法」，**不在贡献句写算法专名**。这与 §0.5.9(a) 禁"反向宣告"（说"不是 X"）不冲突，是正向的类型声明（说"贡献是 Y"）；spec 在此**显式解禁**该正向声明，防其被 §0.5.9(a) 误挡回。
- **复用资产**：paper 1 `discussion.tex`（rev.3，6 子节，含 §6.6 critical-regime boundary probe）+ `limitations.tex`（rev.4，L1–L12，含 L12 actor-fundamental ceiling 论证）+ `conclusion.tex`（rev.2，末句 scope tag 显式化 sub-critical regime + efficiency-v2 reward，指回 L12 / §6.6）。本章小结段**全新写**（paper 1 conclusion 是 8 页节级口吻，dissertation 章末需扩展承上启下，§0.5.3）。
- **限定红线**（paper 1 progress R7/R8）：anti-scaling reversal 与 sim2real implication 严格限定在 `s0 + Kármán wake + REMUS-100` 边界内，不包装为 algorithm-paper claim。
- **§5.10.m 实施细节与可复现性**：β1 reconciliation 四轴差异表（reward / collector / 动作噪声 / seed pool）+ paper 1 L1–L12 limitations 完整列表 + sim2real implication 的 scope tag 列表（s0 + Kármán wake + REMUS-100 + sub-critical regime + efficiency_v2 reward 五重限定）。

---

## 3. 三线缝合点（单章统一相比两篇 paper 的关键决策）

| 缝合点 | 问题 | 建议处理 |
|---|---|---|
| **reward 双轨** | efficiency_v2（paper 1/TD3BC）vs arrival_v2（broad-val v2/FQL），obs 40-D vs 48-D | §5.3 setup 显式交代两套 reward + 各自 obs 维度；每个 results 表标注 reward regime |
| **共同方法入口** | TD3+BC / ReBRAC-Q / FQL 若各自展开，读者会在多个结果节之间拼算法 | §5.4 集中定义方法框架、损失函数、特权信息协议；§5.6–§5.9 只讲各节增量和实验协议 |
| **β1 = 4.0 vs 1.0** | 两线最优 β1 不同会被审稿人当矛盾 | §5.7 埋点（finalist 是 robustness winner）；§5.10 统一 reconciliation（四轴差异 + 统一机制）→ 化矛盾为 finding |
| **online ↔ offline floor 呼应** | 在线 k=4 约 0.46 且跨种子剧烈分化（k=12 可稳定闭合）↔ offline k=4 N2' 0% | §5.5 埋伏笔（k=4 scope 已在 §5.5.5 注明），§5.8 收束："offline + oracle 示范反而 underperform online SAC"；⚠ 对照须用 0.46（2026-07-19 重排后值）、并承接 k=4 vs k=12 不对称 |
| **FQL Method 去重** | FQL 与 ReBRAC-Q 的 loss 公式若各写一遍会冗余 | §5.4 定义共同算法基础，§5.7 给 ReBRAC-Q 增量，§5.9 只讲 FQL distill / flow-matching 差异 |
| **AsymCritic 两处出现** | online §7.7 AsymCritic ablation ↔ offline N2' asym-critic ablation | 统一术语；§5.8 主用 offline 版（actor-fundamental），online §7.7 版在 §5.5 作 cross-line 旁证（§8 开放点 4 已决 2026-07-07：不并列独立呈现，散文指回 §5.5） |
| **特权 critic 方向四态**（复审 D-M1 新增，2026-07-03） | 在线临界=不闭合（§5.5.5）／离线亚临界 TD3+BC=方向性半程有益（§5.6.4）／§5.7 ReBRAC=不抬 mean／§5.8 临界=救不了 | §5.5.5 与 §5.6.4 已各就近作线别限定、不并列下结论；§5.10 统一 reconciliation 把四态收成"特权信息或非必需、或不普遍有效"的一致机制（对齐 §0.1 析取式命题） |

---

## 4. 可引用结果分级（写作时的"能 claim 什么"红线）

| 级别 | 结果 | 出处 |
|---|---|---|
| ✅ **thesis-grade 直接引用** | A0 sensor screen（3 seed）；arrival_v2 §7.9 cross-seed closure（σ=0.038）；ReBRAC 四 findings（5 seed）；TD3+BC closure；broad-val v2 N0/N2' + asym ablation；FQL P2 全机制链 | 各线 report |
| ⚠ **带 caveat 引用** | broad-val v2 仅 2 seed（report §6.1）；FQL n=2（writing index §6，可补 1 seed）；SAC collector saturated cells inconclusive | report 各 §caveat 段 |
| ❌ **不可作性能基线** | reward preset sweep 200k 绝对数字；Sprint 0 preflight；broad-val **v1**（SUPERSEDED 2026-05-18，不进 paper） | online summary §1.0 / offline summary §3.3 |
| 🚫 **不纳入本章** | AUVHamNODE Offline MBRL（PAUSED 2026-05-13，CLAUDE.md out-of-scope） | pause memo |

---

## 5. Figure & Table 清单

**已就位（可直接搬）**：
- Fig 1 sensor schematic（paper 1，s0 vs hull-integral 几何）→ §5.3
- Fig 2 seed dotplot（paper 1，C2 dep vs priv）→ §5.7
- Fig 3 Q-drift bar（paper 1，C3 β2=0）→ §5.7
- FQL verdict 3 图（matrix / noise-axis / c1-rescue，notebook 已出）→ §5.9

**本章需新作**：
- (5.4-a) 方法框架总表（算法 / 训练范式 / 行为约束 / actor 可用观测 / critic 可用观测 / 本章用途）
- (5.4-b) 损失函数与信息协议汇总表（TD3+BC / ReBRAC-Q / FQL / SAC）
- (5.5-a) A0 三 sensor success bar（s0/s1/s2 × efficiency_v2/arrival_v1）
- (5.5-b) arrival_v2 k-monotonicity 相位跃迁（k=4→8→12，含 seed=0 CROSS-SEED-RESCUE）
- (5.6) TD3+BC "2000<1000" 退化 + 纯 BC 对照（可选）
- (5.8) N2' ceiling decomposition（oracle 70% → offline+oracle 0%；含 asym ablation 行）
- (5.9) SAC collector 39-run algorithm×data-quality interaction heatmap（**章级 headline，强推**，§0.2 子问 4 锚）

---

## 6. 现有 paper 资产复用矩阵

| paper 1 资产 | 本章去向 | 改动量 |
|---|---|---|
| `setup.tex` (§3) | §5.3 | 中：泛化到多 reward/regime/sensor |
| `method.tex` (§4) | §5.4 + §5.7 | 中：共同算法定义前移 §5.4，ReBRAC-Q 增量留 §5.7 |
| `experiments.tex` (§5) | §5.7 | 小：照搬四 findings，按 §5.4 已定义符号删冗余方法铺垫 |
| `discussion.tex` (§6，含 §6.6) | §5.8（§6.6 提级独立节）+ §5.10 | 中：拆分 + 融入三线 |
| `limitations.tex` (§7，L1–L12) | §5.10 | 中：合并各线 limitations |
| `conclusion.tex` (§8) | §5.10/本章小结段 | 大：升级到 dissertation 章末，扩展承上启下（§0.5.3） |
| **Appendix A–G** | **按内容主题融入各 §5.x 末"实施细节"子节（NO APPENDIX，§0.5.5）** | **重组**：A/G → §5.3.m + §5.4.m + §5.7.m；B/C/D/E/F → 方法通用项进 §5.4.m，ReBRAC-Q 实验项进 §5.7.m；TD3+BC 复现 → §5.6.m；online 复现 → §5.5.m；broad-val v2 → §5.8.m；FQL + SAC collector → §5.9.m |
| `refs.bib`（12 cites） | 本章 | 增：+ FQL (Park et al.) + flow-matching policy 综述 + dissertation reader-model 所需的领域综述（§0.5.4） |

**新写内容（无现成 LaTeX）**：
- §5.4 强化学习方法与算法框架（methodology，全新统摄写作）
- §5.5 online 全节
- §5.6 TD3+BC 扩写
- §5.9 的 FQL 增量方法 / prose（按 §0.5.2 dissertation voice，不复用 paper 2 abstract）+ SAC collector 章级 headline
- §5.10 的 β1 reconciliation 段 + **本章小结**（§0.5.3 dissertation 章末承上启下，全新写）
- 各 §5.x.m 实施细节子节的重组写作（不"放在最后"，紧接各节论证后，§0.5.5）

---

## 7. 建议写作顺序（rev.11：方法入口优先 + 论证依赖优先）

> **当前状态（2026-06-17）**：§5.1 引言、§5.2 研究背景与定位、§5.3 问题设定与评估协议已经落地。下一轮不再从旧的"复用优先"顺序继续，而应先补齐 methodology 缺口，再按论证依赖写结果节。最终章节顺序仍按 §1 的 10 节结构排列；起草顺序可以不同于最终阅读顺序。

0. **全章节号与 roadmap 同步** — 轻改 §5.1 roadmap、§5.3 末尾承接句和 `main.tex` 骨架，使其承认新增 §5.4 methodology。该步只做结构衔接，不扩写正文。
1. **§5.4 强化学习方法与算法框架** — 先写共同 methodology：MDP/POMDP、online/offline、异策略 Actor-Critic、SAC、TD3+BC、ReBRAC-Q、FQL、特权信息协议。验收标准是后续 §5.6–§5.9 不需要再解释算法基础。
2. **§5.6 TD3+BC 离线基线** — 建立 baseline 证据和瓶颈，说明数据支持集结构与 deployable/privileged 信息差，为 §5.7 的 ReBRAC-Q 主线提供必要问题背景。
3. **§5.7 ReBRAC-Q 主线** — 搬运并改写 paper 1 主体，把共同算法定义前移 §5.4 后，本节聚焦 ReBRAC-Q 增量、四个 findings、LayerNorm/β1/β2 红线和 ReBRAC-Q 相对 TD3+BC 的实际贡献。
4. **§5.8 泛化边界** — 写 critical regime 负结果和 asym-critic ablation。该节要在 §5.7 后写，因为它以 ReBRAC-Q 正面结果为参照；同时必须严守"not rescued by privileged critic"，不写成 s0 actor 信息论不可能。
5. **§5.9 算法比较与数据条件交互** — 写 FQL P2 与 SAC collector 39-run interaction。该节依赖 §5.4 的 FQL 定义和 §5.7 的 ReBRAC-Q 口径，核心是算法排名随数据质量翻转，不是 FQL 或 ReBRAC-Q 的全局胜负。
6. **§5.5 Online RL 可学性与信息瓶颈** — 最后回填 online 支撑节。它在最终阅读顺序中靠前，但其表述应服务于 §5.8 的边界结论，避免过早把 online 结果写成独立 online RL paper。
7. **§5.10 统一讨论与本章小结** — 最后统一 β1=4 vs β1=1、actor 时序访问、BC-anchor 目标质量、特权信息和 LayerNorm 的边界。章末小结要回到 §0.3 中心命题，不写成结果节摘要堆叠。
8. **最终回查与编译** — 全章成稿后统一检查术语、节号、图表编号、reward/objective 标注、引用、编译和清理中间文件。

> 编译路径：`cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`；成功后立即 `latexmk -c` 清中间文件。所有正文节文件按 `main.tex` 的 `\input{sections/...}` 顺序逐节取消注释。

---

## 8. 用户后续确认的开放点 / 已决策状态（不阻塞动笔）

1. ~~**章号 N**~~ — ✅ **resolved 2026-06-02**：**章号 = 第 5 章**（用户拍板 2026-06-02）。rev.11 已全文落地为 §5.1–§5.10 十节结构；前后章衔接保持零跨章自足，不阻塞后续起草。
2. ~~**是否新增 methodology 小节**~~ — ✅ **resolved 2026-06-17（rev.11）**：新增 **§5.4 强化学习方法与算法框架**。其职责是统一方法入口，不宣称新算法；后续结果节只写增量方法与证据。
3. ~~**SAC collector 39-run** 是否进正文（算法对比节末段）还是仅 appendix~~ — ✅ **resolved 2026-06-02；rev.11 节号更新**：**进第 5 章 §5.9 正文末段**，作为 §5.9 章级 headline。锚点 = spec §5.9 layer 2 + §0.2 子问 4（"算法排名是 regime-dependent (algorithm × data-quality interaction)，不是全局"），是 §0 中心命题"特权信息可有可无 + 真杠杆是 anchor 目标质量 × 数据 regime"的关键支撑。带 §0.4 红线 5 cross-source magnitude caveat（同源 SAC mexp 干净，异源 m_multi_mix 仅 direction-robust）。
4. ~~**online §7.7 AsymCritic** 是否要与 §5.8 offline asym ablation 并列呈现（cross-line 一致性证据）还是只留 offline 版~~ — ✅ **resolved 2026-07-07（用户拍板）**：**不并列呈现**——§5.8 主用 offline 版（asym-critic ablation），online §7.7 版以散文指回 §5.5（线别限定），四态收束留 §5.10。已按此落地（§5.8/§5.10 成文；决策记录见 `paper/thesis_ch5/status.md` 已锁决策表）。
5. ~~**是否同时保留 paper 1 / paper 2 的独立投稿版**~~ — ✅ **resolved 2026-06-02；rev.11 节号更新**：**不另起任何 standalone paper**，全部素材整合进博士论文第 5 章。
   - **paper 1 ReBRAC**：31pp arXiv preprint draft（Phase 6.1 commit `932aca1`）作为第 5 章 §5.7 主干**直接复用**；不另投 CoRL/RA-L，也不单独上 arXiv（除非用户后续另议）
   - **paper 2（FQL standalone）**：**撤销启动**。素材并入 §5.9（FQL 算法对比 + SAC collector cross-source headline）；Method/RW/Abstract 不写独立版，按 §5.9 节级 prose 起草（§0.5.2 dissertation voice）
   - **paper 3（online SAC standalone）**：**撤销候选**。素材并入 §5.5（online RL：可学性与信息瓶颈）
   - thesis 第 5 章 = 当前唯一写作目标，本 spec 直接驱动其逐节起草
6. ~~**是否设 appendix**~~ — ✅ **resolved 2026-06-05（rev.5）**：**NO APPENDIX**。本章不设附录；paper 1 App A–G 全部按内容主题融入各 §5.x 末"实施细节"子节（§0.5.5）。这是 dissertation 章相对 conference paper 的体裁差之一：dissertation 章读者期待每节读完即拥有该节复现细节，不希望来回翻附录；conference paper 读者接受附录补充。
7. ~~**章节总篇幅 / cite 条数硬性目标**~~ — ✅ **resolved 2026-06-05（rev.5）**：**无硬性目标，内容优先于篇幅**（§0.5.8）。所有规模决策由「论证完整性 + 读者可追溯」裁决；写完一节后回顾"是否多说 / 少说"，不回顾"是否超页"。
8. ~~**§5.1 起草是直接起 LaTeX 还是先 Markdown outline**~~ — ✅ **resolved 2026-06-05（rev.5）**：**先 Markdown outline，对照 §0.5 体裁规范 review 通过后再起 LaTeX**。§5.1 章级独立引言是 dissertation 体裁的第一次落地实验，需双重 review（体裁是否到位 + 与 §0 中心命题是否一致），用 Markdown 比 LaTeX 迭代成本低。§5.1 outline 通过后 bootstrap `paper/thesis_ch5/` LaTeX 工程。
9. **博论整体目录与前后章衔接** — 🟢 **部分 resolved（rev.7，2026-06-07；rev.11 节号更新）**：§5.1 rev.3 已锁**零跨章引述**——引言"章级定位"段与 §5.10"本章小结"承接段均回写为**不点名引述邻章的自足口径**（§0.5.3 + §2 §5.1 已落地），故**不再依赖**具体前后章名称，原"待前后章名称拍板"的阻塞已消解。**仍 open** 的只剩整本博论目录与硬章号衔接（与开放点 1 相关），但**不影响** §5.x 各节起草（引言/小结已零跨章自足）。
10. ~~**生物借流是否在 §5.10 作一次功能替代回扣**~~ — 🟢 **resolved 2026-07-03（用户授权按复审专业意见执行）：部分解锁**。已锁决策 (b)（生物只在 §5.1/§5.2 作引子、后续不回扣）在 §5.10 **有护栏地放开一次**：§5.5 的发现"空间探针可由时序窗替代"字面上即"单点航行器功能性替代分布式侧线"的工程答案，与 §5.1/§5.2 已投入的"侧线空间分布式 ↔ AUV 单点"对照首尾呼应，故 §5.10 本章小结**允许一次空间↔时序功能替代回扣**，把生物 motif 从装饰性引子升格为立意框架。**四条护栏（§5.10 起草时严守）**：① 仅在 §5.10 本章小结的收束/motivation 层，一句话级别；② 贡献声明本身仍守抽象下界，不带生物专名（侧线/鱼类不进贡献句）；③ 措辞锚定"**功能替代而非结构仿制**"，与 §5.2¶4"非直接仿制"呼应、防 over-claim；④ 不改变中心命题、不新增 claim。此项仅 §5.10 落地，不影响 §5.7/§5.8/§5.9 起草（生物仍不在这些结果节回扣）。**🟢 追加解锁（2026-07-26 用户裁断 CH3）**：全章复审指出四护栏使生物母题在 §5.1/§5.2 建立后**消失 46 页**、§5.10.5 的一句返场读来像装饰 → 允许在 **§5.5.3 机制段**（正是"空间多设探针 ↔ 时间延长历史是获取同一相位信息的两条途径"落地处）以**一个从句**把对照点破一次。四护栏原样适用（一句级、机制/motivation 层、贡献句无生物专名、不新增 claim），且落地措辞**不写"侧线"**、只写"沿体表分布的感知阵列"。生物母题的全章分布因此固定为「§5.1/§5.2 引子 ＋ §5.5.3 一个从句 ＋ §5.10.5 一次回扣」三处，**不再扩大**；§5.6/§5.7/§5.8/§5.9 仍全部不回扣。

> **跨开放点决策（2026-06-02 + 2026-06-05 + 2026-06-17）**：用户明确 (a) "**直接合成博士论文第 5 章，不另写独立 paper**"（2026-06-02 lock）；(b) "**dissertation 章是正式学术论文，不是备忘录 / 实验报告 / 代码文档**"（2026-06-05 拍板，落入 §0.5 体裁规范）；(c) "**内容优先于篇幅，不预设页数 / cite 数**"（2026-06-05 拍板，§0.5.8 + §8 #7 lock）；(d) "**需要独立 methodology 小节**"（2026-06-17 rev.11）。本 spec 是第 5 章的写作 spec（不是"合并蓝图"）；§7 写作顺序 **#1 = §5.4 强化学习方法与算法框架** 是下一步正文动作。
