# 博士论文章节写作 Spec：Deployable-Sensor Offline RL for AUV Wake Navigation

> 文档版本：rev.5（2026-06-05）
> 定位：**博士论文第 5 章 单章统一三线**（online SAC + TD3+BC/ReBRAC + FQL succession）的**核心贡献章**写作 spec。
>
> **✅ 写作策略（2026-06-02 用户拍板）：本 spec = 当前写作目标，不是未来蓝图。**
> Active priority = **直接合成博士论文第 5 章**；**不另起任何 standalone paper**。
> Paper 1 ReBRAC 已有的 31pp arXiv preprint draft（commit `932aca1`）作为 §5.6 主干**直接复用**；paper 2 FQL standalone / paper 3 online SAC standalone **均撤销**，素材分别整合进 §5.8（FQL 算法对比）与 §5.4（online RL 节）。
> Spec 全部内容（§0 中心命题 / §0.4 红线 / §0.5 体裁规范 / §3 缝合点 / §6 复用矩阵 / §5.9 β1 reconciliation 等）直接驱动第 5 章逐节起草，§7 写作顺序 #1 = §5.1 引言起草是下一步动作。
>
> **rev.5 修订要点（2026-06-05 用户拍板）**：
> 1. **NO APPENDIX 锁**：本章不设 appendix；paper 1 Appendix A–G（Reproducibility / Hyperparams / Full tables / Stats / Derivations / Curves / Obs spec）全部**按节融入正文**各 §5.x 的"实施细节"子节。该约束写入 §0.5 体裁规范。
> 2. **8 节结构（vs rev.4 的 7 节；⚠ rev.8 已扩为 9 节——独立 §5.2 Background & Related Work，见 §1 节号统一 note）**：拆原 §N.1 为 §5.1 引言（dissertation 章级独立引言，与全博士论文 §1 总引言对接）+ §5.2 problem description（任务/环境/sensor/reward/数据集统一 setup）；后续 §N.2…§N.7 平移到 §5.3…§5.8，§5.8 章末加"本章小结"段。
> 3. **新增 §0.5 章级写作体裁规范（7 + 1 条）**：reader model / voice / structural depth / citation 策略（定性，**不预设条数**）/ no-appendix 实施细节融入 / 负面结果作贡献项 / 写作纪律（不预设页数）/ **内容优先于篇幅**。该节是 rev.4 缺的"how-to-write"层。
> 4. **撤销所有量化目标**（用户 2026-06-05 指示："先不要考虑预设篇幅和 cite 数，重点是先把内容写清楚，而不是控制篇幅"）：§1 章节骨架表删"估计篇幅"列；§0.5 citation 策略与 interpretation 段长改为定性原则；章节总篇幅无 hard cap。
>
> ⚠ rev 历史：rev.1（2026-05-30 初稿）→ rev.2（2026-05-31 §0 spine 收紧 + §0.4 红线）→ rev.3（2026-06-02 sync paper Phase 6/6.1 + §N.7 rev refs 校准；曾短暂含"先独立 paper"框架）→ rev.4（2026-06-02 撤销"先独立"框架 + 章号=5 lock + SAC collector 进 §N.6 正文 + 各 standalone paper 撤销）→ **rev.5（2026-06-05 加 §0.5 体裁规范 + NO APPENDIX 锁 + 8 节结构拆分 + 撤所有量化目标）** → **rev.6（2026-06-06 加 §0.5.9 报告体→学术体 register 提级 checklist；§5.1 rev.3 落地为 worked example，标题宽泛化为「局部感知下水下航行器运动规划的强化学习方法」）** → **rev.7（2026-06-07 §5.1 引言 rev.4 复审落地——去 §5.7 算法发现误嫁接到 §5.6 ceiling、cite 对齐补 SAC/FQL；spec 侧：§0.5.3 / §2 §5.1「上承下接」据 §5.1 rev.3 零跨章引述锁定回写为自足口径、§8 #8 部分 resolved、§1 新增节号待决登记 note）** → **rev.8（2026-06-07 用户拍板**独立 §5.2 Background & Related Work 永久化**：全 spec 重编号为 9 节[原 §5.2–§5.8 整体 +1 → §5.3–§5.9]；§5.2 定位"研究背景与定位"短节[领域地图 + gap 精确区分 + 贡献坐标，深度算法对比下放各方法节]；rev.7 节号待决 CLOSED；§1 骨架表/§2 新增 §5.2 RW 节定位）** → **rev.9（2026-06-07 用户拍板"生物出发点"叙事框架——§5.1 hook 提级为"自然生物借流 → RL 复现 → 本文"三段弧[聚焦鱼类主线，鸟/浮游一笔带过]、§5.2 RW 用 [`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md) 综述做骨架[生物现象 + RL 复现谱系 + 算法背景 三支汇于 gap]；先 spec 设计、intro.tex/refs.bib 后落地）** → **rev.10（2026-06-07 §5.1/§5.2 推倒重写——用户三条不满[强行分开 / 语体不规范 / 重新定位]触发，用户拍板**方案甲**[两节重新分工，不合并、9 节序不变] + **生物中剂量**。§5.1 = 纯引言漏斗[生物压一段起兴，RL 复现谱系 + gap 精确三支区分整体下放 §5.2]；§5.2 = 生物 landscape + RL 谱系 + 算法背景 + gap 的**唯一充分展开处**。语体彻底学术化[清除直译腔"游泳与飞行生物"、破折号插入语堆叠、强调引号、序数枚举、对举清单换皮]，对象/机制作主语。`intro.tex` rev.6 + `related_work.tex` rev.2 落地，5 页 0 undefined / bibtex 0 error；rev.8 独立 §5.2 + rev.9 生物叙事弧框架**保留**，仅调剂量与分工，9 节映射不变）**。
>
> 与现有文档的关系：
> - 本文是 thesis chapter 的**总规划**；它**统摄**而非取代 [`paper/outline.md`](outline.md)（那是 ReBRAC paper 1 的 8 页 conference outline，对应本章 §5.5–§5.7 的素材来源）与 [`paper/progress.md`](progress.md)（paper 1 LaTeX 工程进度，**31 页 arXiv preprint draft**，可作本章主干直接复用）。
> - 数字 ground truth 一律链接源文档，本文**不复制可能漂移的数字**；headline 数字仅作锚点，写作时实时回查源。
> - 三条 line summary 入口：[`docs/offline_rl_line_summary.md`](../docs/offline_rl_line_summary.md)、[`docs/online_rl_line_summary.md`](../docs/online_rl_line_summary.md)、[`docs/rebrac_line_overview.md`](../docs/rebrac_line_overview.md)。

---

## 0. 章节定位与统一叙事 spine

> **rev.2（2026-05-31）修订要点**：把 spine 从"三段平铺"收紧为**单一中心命题 = 特权信息的可有可无性（privileged-information dispensability）**，并修正一处会被答辩席抓住的归因过头（见 §0.4 红线）。本节是 paper 1 自身 spine（[`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §0.3：*"离线 RL 可以不依赖任何 privileged simulator 信息就把性能逼近 online teacher"*）的**章级提升**，不是新发明。

### 0.1 中心命题（整章只论证这一句）
本章是博士论文的**核心贡献章**。中心命题：

> 在部署受限的单点 DVL water-track 传感器（`s0`）下，offline RL 能把策略逼近用特权 hull-integral 流 `[u_eq, v_eq]` 的 online teacher——**而不需要把特权信息喂给 critic、不需要增加空间传感器、也不需要更强表达力的生成式先验**。真正决定成败的是 (i) actor 对部署信号的**时序访问**、(ii) **BC-anchor 目标质量与数据 regime 的匹配**。该命题在 sub-critical regime 成立，并在 critical regime 撞到一个 **actor-fundamental partial-observability ceiling**。

### 0.2 中心命题的四个支撑（每条对应一/两条实验线，角色分明）

| 子问 | 由谁回答（角色） | 结论（精确版，已过 §0.4 红线） |
|---|---|---|
| **能做到吗？** | ReBRAC 主线（**核心**，§5.6） | deployable-only ReBRAC-Q 在统计意义上**追平** privileged-critic 协议（worldcomp 0.928 vs 0.922, Welch p=0.92），并优于 vanilla TD3+BC 23–32pp。→ 特权信息喂 critic 对 **mean 不贡献**（5-seed Δ priv−dep=+0.6pp），只救 outlier seed 44（+12pp） |
| **靠什么机制？** | Online SAC（**铺垫/机制**，§5.4） | 难流场下 s0→s1 的 80pp gap **不是空间传感器瓶颈、是 actor 时序访问瓶颈**：s0+history k=12（~6s≈涡街周期 30–60%）单变量闭合到 s1 上界（§7.8/§7.9）。给 critic 喂特权流在 online 同样**不闭合 gap**（§7.7）——offline §4.5 asym ablation 的独立 echo |
| **边界在哪？** | broad-val v2（**边界**，§5.7） | critical regime（Re250/u15）下 s0 与 hull-integral 流弱相关，**oracle 示范 + 特权 critic 都救不了**（N2'=0.000；asym ablation = ACTOR_FUNDAMENTAL_CONFIRMED）→ actor-fundamental ceiling |
| **换更强算法会变吗？** | FQL succession（**对照/确证**，§5.8） | 表达力更强的 flow-matching 先验（FQL）**不普遍取胜**：算法排名 regime-dependent（SAC mexp 中质量 FQL 赢、saturated 高质量 ReBRAC 赢，两 CI 各自非零）。真正的算法决定因素是 **BC-anchor 目标质量 × 数据 regime 的匹配**，不是先验表达力 |

> 侧重：ReBRAC 是**唯一正面核心贡献**（正面回答"能做到吗"）；online=机制铺垫，broad-val=边界，FQL=对照确证 + 真实算法决定因素。四块都在为同一句"特权信息可有可无 + 真正杠杆是 actor 时序访问 / anchor 质量"服务。

### 0.3 统一 takeaway（章末收束）
> 对 deployable-sensor offline RL：正向杠杆是 **actor 对部署信号的时序访问** + **BC-anchor 目标质量与数据 regime 的匹配**；几个"直觉上该有用"的东西其实**非必需或不普遍有效**——给 critic 喂特权流不闭合 gap（online §7.7 + offline §4.5 两条**独立** asym ablation）、增加空间探头有效但**非必需且不可部署**（online §7.8 时序访问可替代）、更强生成式先验**不普遍取胜**（FQL regime-dependent）。这给水下机器人这类部署受限场景一条"**不依赖任何 simulator-only 信号即可逼近 online teacher**"的配方，及其在 critical regime 的失效边界。

> 该 takeaway 在 §5.9 "本章小结"段以 dissertation 章末口吻完整复述（不只是 conference paper 的一句 punchline，需展开到对全博士论文的承上启下 — 见 §0.5.3 structural depth 规范）。

### 0.4 ⚠ 机制归因红线（防答辩席反例，写作时严守）
1. **不写"actor 侧因素*决定*全部 mean 性能"**。critic **LayerNorm** 是单项最大 mean 杠杆（LN-off **−16.2pp**，远超 β2=0 的 −2.4pp；[`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §5.2b）。mean 归因只在**两个 BC penalty 之间**成立：actor β1 carry mean、critic β2 carry 跨数据集 Q-stability（review §0.3 / §5.2）。LayerNorm 是**第三条独立的"表征稳定性"轴**，与"特权信息"命题正交——属"必要基础设施"，**不进** §0.3 的"可有可无"清单。
2. **不写"ReBRAC over TD3+BC 的 +23–32pp 来自 actor anchor"**。在 β1=4.0 ↔ TD3+BC α≈0.25 actor-anchor 口径**匹配**下，recipe 差异同时含 dual penalty + critic LayerNorm + Q-norm；mean uplift 不能归给单一 actor 轴。
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
| **博士论文章 ✅** | "在 X 假设下，[mechanism Y] 决定 [outcome Z]。第 5.k.m 节给出 statistical evidence [reference]，以及它在 critical regime 失效的边界 [reference §5.7]。" | — |

具体规则：
- 主语优先**机制**（"actor 的时序访问能力决定 …"），其次**结果**（"deployable-only ReBRAC-Q 追平 …"），最后才是**操作**（"我们运行了 …"）。
- 一段一论点；每段首句 = 该段论证的 thesis sentence。
- 实验数字嵌入论证句中（"Δ priv−dep=+0.6pp, Welch p=0.92, n=5 → 特权 critic 在 mean 维度不贡献"），不单列为脚注或 callout。

**0.5.3 Structural depth — 章级 ≠ 节级**

dissertation 章相对会议论文节多两层：

- **§5.1 章级独立引言**（rev.5 拆出；**rev.7 据 §5.1 rev.3 「零跨章引述」锁定回写**）：做**自足的章级定位**——以不依赖具体邻章的抽象口径交代本课题在博论中的角色与本章独立贡献，**不点名引述前一章**（不同范式的全局路径规划；§5.1 rev.3 已锁零跨章），**不**提前披露 §5.3 problem description 的具体 setup。roadmap 以**结果语态、节作主语**收尾（§0.5.9 表(a)），**按语义标签描述节序**（问题设定与评估协议 → online 可学性与瓶颈 → 离线基线 → 离线主线 → 失效边界 → 算法对比 → 统一讨论），硬编号以落地 `thesis_ch5/main.tex` 实际节序为准（见 §1 节号待决登记 note）。**不是** paper 1 abstract 的扩写。**（rev.10 方案甲分工锁定，2026-06-07）**：§5.1 为**纯引言漏斗**（生物起兴 → 核心问题 → 反直觉结论 → scope → 贡献声明 → 两条主线机制 → roadmap）；生物出发点压为**一段起兴**（鱼类为主、鸟一笔带过），**RL 复现谱系与 gap 精确三支区分整体下放 §5.2**，§5.1 不重复展开（消除前数轮"两节各讲半套"的重复）。两条主线贡献仅在 §5.1 ¶6 充分展开，§5.2 对此只作 forward pointer。
- **§5.9 末段"本章小结"**（rev.7 零跨章回写，与 §5.1 对称）：除 takeaway 复述外，加一段**自足的承上启下**——以抽象口径交代本章命题在博论整体论证中的位置（如"本章为博论提供 deployable-sensor offline RL 的可行性证明、边界与算法决定因素"），**不点名引述具体后续章**（不写"下一章 §6 将…"等硬章号衔接）。

每节内部仍按 conference paper 节级展开（motivation → method/setup → results → mechanism/discussion → 实施细节子节），无需额外加层。

**0.5.4 Citation 策略（定性，不预设条数）**

原则：每条 substantive claim（机制、数字、对比、限制）须可追溯到 (a) 本课题已有 docs（行内链接，源文档绝对权威）或 (b) 外部文献（`\cite{}`，正式 BibTeX）。**不刻意凑数，也不为压缩而省**。

具体场景：
- **机制 claim**（e.g. "BC anchor 目标质量决定 algorithm × data interaction"）：必须同时 cite 自己实验（§5.8）与原文献（FQL Park et al. 2024）。
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

本章三条核心负面结果（§5.7 critical-regime actor-fundamental ceiling / §5.8 FQL conditional-iff 证伪 / §0.4 红线 3 三种"可有可无"失效方式各异）**作为正面 finding 写**，不降级为 limitation 段。这是本课题区别于"算法 paper 总宣称 SOTA"的章级骨气。

具体写法：
- 节标题不写"limitation of X"；写"Boundary: X fails when Y"或"Honest Negative: X 不普遍"。
- 负面结果与正面结果同台呈现统计 evidence（CI / p / n），不"在 discussion 末段提一句"。
- mechanism explanation 与正面 finding 同等深度（"为何会失败 = 也是机制贡献"）。

**0.5.7 写作纪律（不预设页数 / 不预设 finding 段长）**

- 每段 3–4 句封顶（沿用 paper 1 progress.md §9 R1）；超出 → 拆段或拆子节。
- 每节"finding → interpretation → caveat"三段式 — interpretation 段**不可省**（finding 与 contribution 之间靠 interpretation 缝合），**长度由论证完整性决定**，不预设上限。
- 数字实时回查源 docs（不复制可能漂移的 headline）；§0.4 红线写作时严守。
- 算法命名统一 **ReBRAC-Q**（§5.6 严守）/ **TD3+BC** / **FQL** / **SAC**（不裸写 "ReBRAC"，§5.6 method 首句立规）。

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
| "本章其余部分分为两部分。先…继而…随后…最后…" | 过程流水账（procedural narration） | roadmap 改结果语态，**节作主语**（"第 5.6 节给出…"） |
| "其一…其二… / 贯穿这 N 点的是…" | 结构暴露：把论证的"形状"念给读者 | 散文对举（"一条是…另一条则…"），不预报"有几层 / 几点" |

**(b) 5 条改写原则（落地每节套用）**

1. **去 speech-act 化**：删宣告章节动作的框架句，让论点直接成句。
2. **主语降级 操作→机制**（§0.5.2 优先级）：凡"本章 + 言说动词"，改以**机制 / 对象 / 发现**作主语（"决定成败的是信息在时间上的组织方式"）。
3. **溶解枚举**：序数（第一 / 其一）、流程词（先 / 继而 / 随后 / 最后）→ 逻辑连接（因果 / 转折 / 让步 / 递进）；不向读者预报计数。
4. **去口语 editorializing**："直接的""相应地""也是本章的重点"等删除或学术化。
5. **贡献优先散文**：引言贡献宜综合为**少数主线的论证段**，优于 itemize bullet（bullet 属 §0.5.2 会议体，⚠）；正文节内"finding→interpretation→caveat"仍按散文展开。

**(c) 校准锚**：合格 register 见 paper 1 `setup.tex`（陈述句构造对象、机制作主语）；本章 worked example 见 `thesis_ch5/sections/intro.tex` rev.3。注意"我们 / 本章"本身**不是**报告体 —— `setup.tex` 亦用"我们考虑 / 我们采集"；病在**汇报语态**，不在该主语词。提级时**不**回退已锁语义（零跨章引述 / 抽象到"环境流场 + 单点局部可观测" / 已删生造词）。

---

## 1. 章节骨架总览

> **rev.5 重构（2026-06-05）**：拆原 §N.1 为 §5.1 章级独立引言 + §5.2 problem description；后续节平移到 §5.3…§5.8；§5.8 末加"本章小结"段。**删除"估计篇幅"列**（NO PRESETS — §0.5.8）；**删除 Appendix 行**（NO APPENDIX — §0.5.5，paper 1 App A–G 按内容主题融入各 §5.x 末"实施细节"子节）。
>
> ✅ **节号统一（rev.8，2026-06-07，用户拍板独立 §5.2 RW 永久化）**：§5.1 引言落地（`thesis_ch5/`）引入**独立 §5.2 Background & Related Work（研究背景与定位）节**；全 spec 已据此**统一重编号为 9 节**——原 8 节基线 §5.2–§5.8 整体 **+1 → §5.3–§5.9**（rev.7 的「待决登记」CLOSED）。下表为 8→9 历史对照（仅供回溯，spec 正文已全部对齐落地 9 节，与 `main.tex` 一致）：
> | 落地 9 节（现行） | 原 8 节基线（已废止） |
> |---|---|
> | 5.1 引言 | 5.1 引言 |
> | **5.2 Background & Related Work（研究背景与定位，独立）** | —（原并入 §5.1 / §0.5.3） |
> | 5.3 Problem description | 5.2 Problem description |
> | 5.4 Online RL | 5.3 Online RL |
> | 5.5 TD3+BC | 5.4 TD3+BC |
> | 5.6 ReBRAC 主线 | 5.5 ReBRAC 主线 |
> | 5.7 边界 | 5.6 边界 |
> | 5.8 算法对比 | 5.7 算法对比 |
> | 5.9 讨论 | 5.8 讨论 |
>
> **§5.2 RW 职责限定（用户专家判断，2026-06-07）**：定位为**简短的"研究背景与定位"节，非全面文献综述**——承载 (i) 三支领域地图（offline RL 谱系 / 部分可观测·特权信息 / 数据驱动流场控制）、(ii) 本章 gap 的精确区分（已有工作要么假设丰富观测、要么允许在线交互、要么部署期可用特权信息；"单点 + 离线 + 无特权"的组合尚无系统刻画）、(iii) 科学洞见型贡献在该图上的坐标。**算法间深度技术对比**（TD3+BC α / ReBRAC dual-penalty / FQL flow-matching 差异）**下放各方法节**（§5.6/§5.8 method），§5.2 不重复；应用 hook 留引言 ¶1，方法谱系初次点名留引言 ¶5，§5.2 只深化 gap。篇幅 1.5–2.5 页量级。**反向触发**：仅当博论确有独立 RL 综述章、且放宽"零跨章自足"锁时，才回退为无独立 §5.2（与当前已锁决策冲突，非默认路径）。

| § | 小节 | 主要素材来源 | 资产状态 |
|---|---|---|---|
| 5.1 | **章级独立引言**（dissertation 体裁层，与 §1 总引言对接 + 本章 roadmap；零跨章自足） | 全新写（参考 §0.5.3 structural depth） | ✅ **rev.6 推倒重写已落地**（方案甲：纯引言漏斗 7 段；`thesis_ch5/sections/intro.tex`） |
| 5.2 | **Background & Related Work（研究背景与定位）**（生物借流现象 + RL 复现谱系 + 算法背景 三支汇于 gap；**短节非综述**，承 §5.1 生物 hook、聚焦鱼类，详上方职责限定 note + §2 骨架） | [`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md)（生物借流 + RL 复现综述骨架，22 refs）+ 引言 ¶1/¶5 cite 起步 + offline RL/POMDP·特权信息 文献 | ✅ **rev.2 推倒重写已落地**（方案甲：landscape+gap 唯一展开处，4 段约 2 页；`related_work.tex`） |
| 5.3 | **Problem description**（任务 / 环境 / s0 vs privileged / reward 双轨 / 数据集 statistics） | [`environment_design.md`](../docs/environment_design.md) + paper 1 `setup.tex` | ✅ **可直接复用** paper 1 §3 + Fig 1，需泛化到 reward 双轨 / 多 regime |
| 5.4 | Online RL：可学性与信息瓶颈 | [`online_rl_line_summary.md`](../docs/online_rl_line_summary.md) §1.1 + [`arrival_v2_experiment_report.md`](../docs/arrival_v2_experiment_report.md) §7.9 | 🟡 **新写**（仅 docs，无 LaTeX） |
| 5.5 | Offline baseline：TD3+BC 拆瓶颈 | [`td3bc_phase0c_experiment_report.md`](../docs/td3bc_phase0c_experiment_report.md) + [`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §1 | 🟡 **新写/扩写**（paper 1 仅作 baseline 引用，无独立节） |
| 5.6 | ReBRAC 主线：方法 + 四 findings（**章重心**） | paper 1 `method.tex` + `experiments.tex` + [`rebrac_experiment_report.md`](../docs/rebrac_experiment_report.md) | ✅ **几乎全文复用** paper 1 §4–§6 + Fig 2/3 |
| 5.7 | 泛化边界：broad-val v2 → actor-fundamental ceiling | [`rebrac_broad_validation_v2_report.md`](../docs/rebrac_broad_validation_v2_report.md) §3–§5 + §4.5 | 🟡 **部分复用**（paper 1 Phase 6 已折叠为 §6.6 + L12，提级为独立节） |
| 5.8 | 算法对比：FQL succession 诚实负面 + SAC collector cross-source headline | [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) + [`fql_succession_paper_writing_index.md`](../docs/fql_succession_paper_writing_index.md) + [`arrival_v2_sac_collector_design.md`](../docs/arrival_v2_sac_collector_design.md) §4.0.10 | 🟡 **新写**（撤销的 paper 2 素材整合，定位 thesis subsection；Method/RW/Abstract 按节级 prose 起草） |
| 5.9 | 统一讨论 + 跨线 reconciliation + limitations + **本章小结** | paper 1 `discussion.tex`/`limitations.tex` + 各线 discussion 段 | 🟡 **整合改写**（融合三线 + β1 reconciliation + dissertation 章末承上启下段） |

**实施细节融入方式（NO APPENDIX）**：每节末附 §5.k.m "实施细节与可复现性"子节，承载 paper 1 App A–G 对应内容（§0.5.5 详）。例：
- §5.6.m 承担 paper 1 App A (Repro) + B (Hyperparams) + C (Full ReBRAC tables) + D (Stats) + E (Dual-penalty derivations) + F (Q-drift curves) + G (Obs spec)
- §5.3.m 承担 reward 双轨 + dataset matrix + sensor schematic 复现细节
- §5.8.m 承担 FQL collector / SAC 39-run 数据来源 + statistical method

---

## 2. 逐节素材映射

> 写作纪律（沿用 §0.5 + paper 1 progress.md §9 R1–R10）：mechanism > storytelling；每段 3–4 句封顶；数字实时回查源表；算法命名统一 **ReBRAC-Q**（不裸写 "ReBRAC"）。每节末附 §5.k.m "实施细节与可复现性"子节（NO APPENDIX，§0.5.5）。

### §5.1 章级独立引言（dissertation 体裁层 — rev.5 新增）

- **写什么（与会议节级引言显著不同，§0.5.3 structural depth；rev.7 据 §5.1 rev.3 零跨章引述锁定回写——下列各点均改为不点名引述邻章的自足口径）**：
  1. **章级定位（取代原"上承"）**：以抽象口径交代本课题在博论中的角色——自主航行器在自身无法完整观测的环境流场中做运动规划，单点局部可观测约束下 RL 能推进到何种水平；**不点名前一章**（不同范式的全局路径规划，零跨章），任务名统一"运动规划"（不用"导航/navigation"）；
  2. **本章独立贡献**：deployable-sensor offline RL 这一统一命题（§0.1 中心命题，按 §0.5.9 贡献优先散文综合为少数主线，**不** itemize、**不**借用 paper 1 abstract 的紧凑写法 — 章级独立引言可以而且应当展开 motivation 与 stakes）；
  3. **章 roadmap**：以结果语态、节作主语收尾（§0.5.9），按语义标签——问题设定与评估协议 → online 可学性与瓶颈 → 离线基线（TD3+BC）→ 离线主线（章重心）→ 失效边界 → 算法对比 → 统一讨论；硬编号以落地 `thesis_ch5/main.tex` 节序为准（§1 节号待决登记 note）；
  4. **章级承接（取代原"下接"）**：以抽象口径点出本章结论在博论整体论证中的位置，**不点名引述具体后续章**（与 §5.9 本章小结对称，零跨章）。
- **复用资产**：**无**。不复用 paper 1 abstract（abstract 是 8 页会议节级口吻，外审会立刻识别为"被搬过来"）。重新起草。
- **caveat（reader model，§0.5.1）**：假设外审人/答辩席知道 RL 标准术语（SAC / TD3+BC / Q-learning），但**不**熟悉本课题历史；术语首次出现需 1 句定义 + 1 个 cite。
- **实施细节子节**：本节无 §5.1.m（引言节不需"复现"段）。
- **落地状态（rev.4，2026-06-07）**：§5.1 已落地 LaTeX（[`thesis_ch5/sections/intro.tex`](thesis_ch5/sections/intro.tex)），6 段——环境流场 + 单点可观测的 funnel → 核心问题（online/offline 两情形，offline 重心）→ 两点反直觉 + ceiling 边界 → scope caveat → 贡献（两条主线散文：信息时序维度 / 离线内部 anchor-数据适配）→ roadmap（节作主语、结果语态）。零跨章引述、抽象下界（环境流场 + 单点局部可观测）、贡献散文（非 itemize）、online/offline 不对称权重、§0.4 红线（不提 LayerNorm / 不嫁接 §5.8 算法发现到 §5.7 ceiling）均已守。复审记录见 commit `bf6fc12`。
- **hook 弧调整（rev.9，2026-06-07 用户拍板"生物出发点"框架；先 spec 设计、后落地 intro.tex）**：rev.4 的 ¶1 工程 hook（直接从航行器 / 环境流场切入）提级为 **"自然生物借流 → RL 复现 → 本文"三段弧**（本章最早出发点）。注意 rev.4 ¶1 已 cite `gunnarson2021ncomm` + `verma2018pnas`（即综述 [15]/[14]），生物根基已隐含，本次为**显式化、前置化**：
  - **新 ¶1（自然原则）**：自然界生物并非被动承受流场，而是**主动感知局部流动结构并转化为运动收益**，且依赖**局部、有限的感知**而非全局观测——一条普适自然原则（注：生物侧线本为空间分布式阵列，"单点"专指本文 AUV 的部署传感，二者构成反衬，勿混；详第二轮复审 B-1）。**聚焦鱼类主线**（涡街 Kármán gait / 障碍物尾流 / 侧线分布式局部感知），鸟类（热气流·动态滑翔）与浮游生物作"更广泛自然界印证"一笔带过（用户 2026-06-07 选定广度）。
  - **新 ¶2（RL 复现 + 收口）**：该借流现象启发用 RL 在复杂流场学习高效运动的研究路线（cite Colabrese 2017 / Verma 2018 / Gunnarson 2021 / Jiao 2025）→ 已有工作多假设丰富观测或允许在线交互 → 收紧到真实部署的严格约束（单点 + 离线 + 部署无特权）→ **接现有 ¶2 核心问题**。
  - **保留**：现 ¶2–¶6（核心问题 / 两点反直觉 / scope caveat / 贡献散文 / roadmap）不动；仅前置改写 ¶1 为两段弧。
  - **红线**：① 生物术语仅在 motivation 层，贡献层（¶5/¶6）仍守抽象下界（不写涡街/REMUS/DVL）；② 生物开篇放大"宽问题"，**scope 收口须更硬**（¶4 caveat + 贡献句旁诚实 scope 顶住 over-claim，§0.4 + R7/R8）；③ 收敛 ≤2 段，详细 landscape 留 §5.2，引言只取代表性 cite；④ 不破坏 rev.4 已守（零跨章 / 贡献散文 / 不对称权重 / 不提 LayerNorm / 不嫁接算法发现）。
  - **素材源**：[`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md)（自然生物借流 + RL 复现综述，22 refs 带 DOI）。
- **落地状态（rev.5 hook + 第二轮对抗复审，2026-06-07）**：生物 hook 已落地（`intro.tex` rev.5，**§5.1 现 7 段**——rev.4 funnel ¶1 拆为生物原则 ¶1 + RL 复现 ¶2；G-3 段数回写）；§5.2 独立成节落地（`related_work.tex` rev.1，4 段）。第二轮独立对抗复审（authoring/review 分离 lane）逐句核 cite 忠实 + 质疑 gap，落地——必改：**B-1**（§5.1 ¶1 侧线「单点式」→「局部、有限」，消事实错[侧线本分布式] + 与 §5.2 自相矛盾；**本 spec line 229 同源措辞已一并修正**）；**D-1**（§5.1 ¶2 jiao2025 不再错配「具身机器人控制」→「辨明可靠学习所需的局部感知量」，四 cite 对齐）；**E-1**（§5.2 ¶4 gap 第三支收敛「训练与部署均不借助特权」，去 asymmetric-AC 稻草人，对齐 §0.1）；**C-1**（§5.2 ¶4 两条主线复述删→ forward pointer，独留 §5.1 ¶6）。commit `98888be`。捎带 MEDIUM/LOW：C-2（¶4↔¶6 时序主线措辞拉开）/C-3（¶6↔§5.2¶3 生成式先验措辞）/C-4（§5.1¶1↔§5.2¶1 鱼类例子承接式去重）/D-2（flato2024 删「风切变」over-attribution）/D-3（删「同伴涡街」，群体维度留 §5.2）/E-2（gap 第一支「多于单点的观测，或在线试错」精确化）。保留不动（LOW）：A-1/A-2 register（分号对照结构）、B-2 scope（¶5 已够）、D-4 coombs cite（贴「分布式」正确 claim）、G-1 refs.bib 未引条目（BCQ/CQL 后续节备用）、G-2 pinto key（RSS 2018 正确）。编译 5 页 0 undefined / bibtex 0 error。 **（rev.6 推倒重写 SUPERSEDE，2026-06-07——用户三条不满触发，方案甲落地）**：§5.1 改为**纯引言漏斗 7 段**[生物起兴 ¶1 → 工程命题+核心问题 ¶2 → 反直觉+ceiling ¶3 → scope ¶4 → 贡献声明（五方法五 cite）¶5 → 两条主线机制 ¶6 → roadmap ¶7]；rev.4–rev.5 的生物两段弧**压为一段起兴**（鱼类涡街驻留+侧线局部有限+鸟热气流；liao/coombs/flato 三 cite）；**RL 复现谱系 + gap 精确三支区分整体下放 §5.2**（消除"两节各讲半套"重复）。语体彻底学术化：清除直译腔"游泳与飞行生物"→"水生与空中的动物"、破折号插入语全去、强调引号全去、序数枚举（其一/其二）溶解为逻辑连接、对举清单换皮（"一条是…另一条…"）溶解为论证散文。红线零踩（不带数字 / 不提 LayerNorm / 不嫁接 §5.8 算法发现到 §5.7 ceiling / 抽象下界 / online-offline 不对称 / 零跨章 / 贡献散文）。grep 正文违规 marker clean，5 页 0 undefined / bibtex 0 error。

### §5.2 Background & Related Work（研究背景与定位 — rev.8 独立成节）

- **写什么（短节非综述，§0.5.3 + §1 职责限定 note；rev.9 用 [`flow_navigation_rl_review_zh.md`](../reference/flow_navigation_rl_review_zh.md) 做骨架，承 §5.1 生物 hook）**：四支汇于 gap——
  1. **自然生物借流现象**（呼应 §5.1 hook，**聚焦鱼类**）：流场的**能量场 / 信息场二分**；鱼类主线（rheotaxis 流趋性 / Kármán gait 涡街 / 障碍物尾流节能 / 鱼群涡相位匹配 / **侧线分布式局部感知**）；鸟类·浮游生物一笔带过。
  2. **RL 复现谱系**：奠基（Colabrese 2017 微游泳体 / Verma 2018 涡利用集体游泳）→ 局部感知目标导航（Gunnarson 2021 局部流速 / Jiao 2025 流场梯度必要性）→ 具身鱼形机器人穿越涡街（Zhu 2022 / Feng 2024）→ 微游泳体算法评估（Qiu 2022 对称性 / Mecanna 2025）。
  3. **算法侧背景**：offline RL 行为约束谱系（BC penalty → 更具表达力的生成式策略先验）、部分可观测下的特权信息与非对称 actor-critic。
  4. **本章 gap 精确区分 + 坐标**：已有 RL 借流 / offline 工作要么假设丰富·多点观测、要么允许在线交互、要么部署期可访问特权信息——"**单点局部感知 + 纯离线 + 部署无特权 + 可部署传感**"这一组合尚无系统刻画；本章科学洞见型贡献（信息时序维度 + 离线 anchor-数据适配）即定位于此。
- **不写什么（去重红线，避免与引言/各方法节重复）**：算法间**深度技术对比**（TD3+BC α / ReBRAC dual-penalty / FQL flow-matching loss 差异）**下放各方法节**（§5.6 / §5.8 method），§5.2 不展开公式；引言 ¶1/¶2 的生物 hook 与方法谱系初次点名不在 §5.2 重复，§5.2 把它们**展开为有论点的 landscape 并深化 gap**。
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
- **§5.3.m 实施细节与可复现性**（NO APPENDIX 融入）：承担 paper 1 App A reproducibility + App G obs spec + dataset matrix 表；evaluation manifest JSON schema + Welch / paired-mean / rule-of-three 计算式 + seed 选择协议。

### §5.4 Online RL：可学性与信息瓶颈
- **写什么**：(1) A0 sensor screen — deployable s0 在简单 wake viable；(2) arrival_v2 prototype — H_information-bottleneck（s0+history k=12 闭合 80pp 的 s0→s1 gap）+ manifest universal-floor 概念。
- **数字源（thesis-grade）**：[`online_rl_line_summary.md`](../docs/online_rl_line_summary.md) §1.1（A0：efficiency_v2 s1 0.967 / s0 0.789 / s2 0.856）；[`arrival_v2_experiment_report.md`](../docs/arrival_v2_experiment_report.md) §7.9 / §7.9.7（k=12 在 2/3 seed saturate manifest floor 27/30=0.900；3-seed σ_final=0.038 << target 0.10）。
- **复用资产**：无 LaTeX，**全新写**。可新作 figure（A0 三 sensor bar + arrival_v2 k-monotonicity 相位跃迁图）。
- **叙事作用**：为全章奠定"瓶颈在 actor 信息访问，不在传感器硬件"的基调，**为 §5.7 actor-fundamental ceiling 埋伏笔**（online 难流场 catastrophic floor 10% ↔ offline N2' 跌破到 0%）。
- **caveat**：reward preset sweep（27 run×200k）**绝对数字不可引用**，仅作 `efficiency_v2` 选择的方法学依据；Sprint 0 preflight 不可作性能基线。坑（200k 假性否决 / flow 不一致 bug / efficiency_v2 OOB-suicide）可入方法论或 limitations。
- **§5.4.m 实施细节与可复现性**：A0 sensor screen 3-seed 协议 + arrival_v2 prototype k-sweep 协议 + manifest universal-floor 计算 + reward preset sweep 方法学（数字不引、仅作 viability 论证）；online §7.7 AsymCritic ablation 复现细节（若 §8 开放点 3 决定保留 cross-line 旁证）。

### §5.5 Offline baseline：TD3+BC 拆瓶颈
- **写什么**：TD3+BC 把"数据越多越差"从协议伪象**纠正为真实现象**，拆出两个瓶颈：(1) 数据支持集结构（"2000<1000"退化在纯 BC 下同样成立）；(2) deployable critic 信息瓶颈（privileged-critic 关 48.5% gap）。→ 为 ReBRAC 创造问题设定。
- **数字源**：[`td3bc_phase0c_experiment_report.md`](../docs/td3bc_phase0c_experiment_report.md)；headline（[`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §1）：crosscomp 500/1000/2000 = 0.592/**0.672**/0.596；worldcomp deployable 最优退回 α≈0（纯 BC 0.858）/ privileged 0.922 / teacher 0.990。
- **复用资产**：paper 1 把 TD3+BC 仅作 baseline 行引用，**无独立节** → 本章需扩写成一个短节（dissertation 容量允许讲完整 baseline 故事，§0.5.3 structural depth）。
- **叙事作用**：建立"为何需要 dual-BC penalty"的动机，自然过渡到 §5.6。
- **§5.5.m 实施细节与可复现性**：TD3+BC 损失全式 + α-sweep 协议 + nearest-action 距离度量 + crosscomp 500/1000/2000 数据集构造 + worldcomp deployable/privileged α-optimum 报告。

### §5.6 ReBRAC 主线：方法 + 四 findings（**本章重心**）
- **写什么**：方法（Q-normalized dual-penalty TD3+BC variant = ReBRAC-Q）+ 四条 paper-level finding。
- **四 findings 数字锚**（实时回查 [`rebrac_experiment_report.md`](../docs/rebrac_experiment_report.md)；镜像见 paper 1 progress.md §3）：
  - (i) crosscomp +23.0pp（1000）/ +32.2pp（2000），翻转"more data hurts"，std 不增反降
  - (ii) **deployable 0.928 追平 privileged-critic 0.922**，Welch p=0.9195（最强 sim2real narrative）
  - (iii) dual penalty cross-dataset 必需（β2=0 时 mean_target_q 漂 +46%/+98%）
  - (iv) critic LayerNorm ⊥ dual penalty（LN-off −16.2pp，远超 β2=0 的 −2.4pp，方向相反）
- **⚠ 与 §0 dispensability 命题的接口（写 Finding (iv) 时严守，防自相矛盾，cross-ref §0.4 红线 1）**：critic LayerNorm 是**第三条独立的"表征稳定性"轴 = 必要基础设施**，与 §0 的"特权信息可有可无"命题**正交**——它**不**进 §0.3 的"可有可无"清单。两条配套红线：(a) mean 归因只在 **actor β1 vs critic β2** 之间成立（β1 carry mean、β2 carry Q-stability），**不可**升级为"actor 侧决定全部 mean"（LN 才是单项最大 mean 杠杆）；(b) LN claim 强度 n=2 seeds 仅支撑 "necessary component 存在性"，**不 claim "LN 比 dual penalty 重要"**（review §2.2.4）。
- **复用资产**：✅ **paper 1 主干几乎全文可搬** — `method.tex`（rev.2，134 行）+ `experiments.tex`（rev.2，233 行，含 4 finding 子节 + 主表 + 2 ablation 表）+ **Fig 2 seed dotplot** + **Fig 3 Q-drift**。
- **关键命名红线**：全文 **ReBRAC-Q (ours)**，method 节首句声明 + §5.6.4 差异表（β1=4.0 ↔ TD3+BC α≈0.25）；不裸写 "ReBRAC"。
- **finalist**：`(β1=4.0, β2=2.0)`，是 **robustness winner（救 seed 44）**非 peak winner — 这是 §5.9 β1 reconciliation 的前提，§5.6 须埋点。
- **§5.6.m 实施细节与可复现性**（NO APPENDIX 重头戏 — 集中承担 paper 1 App A–G 中 ReBRAC 部分）：
  - paper 1 App A reproducibility（5-seed 协议 / Welch / paired-mean / rule-of-three / robustness winner 选择规则）
  - paper 1 App B 超参表（actor/critic lr / batch / γ / τ / β1=4.0 β2=2.0 / Q-norm 实现 / LN 位置）
  - paper 1 App C 全表（worldcomp / crosscomp 双数据集 × 5 seed × β1×β2 sweep）
  - paper 1 App D 统计细节（Welch t-test 自由度 / paired-mean Δ priv−dep CI / Q-drift +46%/+98% 计算式）
  - paper 1 App E dual-penalty derivation（actor β1 carry mean / critic β2 carry Q-stability 的损失梯度分解）
  - paper 1 App F 学习曲线全图（5-seed 收敛轨迹 + Q-drift bar）
  - paper 1 App G obs 24-D / 40-D 详细 spec
  - 该实施细节子节是全章最长子节（论证完整性，§0.5.8）；不"放在最后"，紧接 §5.6 四 finding 后。

### §5.7 泛化边界：broad-val v2 → actor-fundamental ceiling
- **写什么**：broad-val v2 两 cell — N0（sub-critical）HOLDS；N2'（critical Re250）STRONG_NEGATIVE → **deployable s0 actor-fundamental partial-observability ceiling**；asym-critic ablation 排除 critic-fundamental 解释。
- **数字源**：[`rebrac_broad_validation_v2_report.md`](../docs/rebrac_broad_validation_v2_report.md) §3–§5（N0 0.850 HOLDS / N2' 0.000 STRONG_NEGATIVE，跌破 online catastrophic floor 10pp）+ §4.5（asym-critic ablation，verdict **ACTOR_FUNDAMENTAL_CONFIRMED**：完美 hull-integral flow 喂 critic，s0 actor 仍 0.000）。
- **复用资产**：paper 1 Phase 6 已把这部分折叠为 `discussion.tex` §6.6 + `limitations.tex` L12 → 本章**提级为独立节**（dissertation 容量允许，§0.5.3 + §0.5.6 负面结果作 finding 不降级 limitation）。
- **叙事作用（§0.5.6）**：这是本章**最强负面 finding**，与 §5.6 正面 finding 同台呈现统计 evidence（n / verdict / CI），**不**降级为 limitation 段；mechanism explanation 与正面 finding 同等深度（"为何会失败 = 也是机制贡献"）。
- **⚠ claim 边界红线**（[`fql_succession_paper_writing_index`](../docs/fql_succession_paper_writing_index.md) 同源原则，broad-val v2 report §4.5 明确）：
  - ✅ **可下**："privileged-flow critic 不能挽救 N2' 天花板；**排除 critic-fundamental**；HARDENS actor-fundamental"
  - ❌ **不可下**："proven actor-incapable / s0 actor 信息论上不可能" — asym=0 同时兼容"表征不可能"与"asym-critic 机制没把信息转化给 actor"两种读法。
- **§5.7.m 实施细节与可复现性**：broad-val v2 manifest（u15_cross / u15_upstream）+ N0/N2' 2-seed 协议 + asym-critic ablation 实验设置（特权流喂 critic 通道，actor 保持 s0）+ verdict ACTOR_FUNDAMENTAL_CONFIRMED 推导链 + N2' 跌破 online catastrophic floor 的对照（§5.4 online ↔ §5.7 offline floor 比对表）。

### §5.8 算法对比：FQL vs ReBRAC — 先验表达力 vs anchor 目标质量
- **写什么（两层结论，必须按此顺序，否则会过头）**：
  1. **privileged-collector regime（FQL P2）**：核心 conditional-iff 假设（"FQL > ReBRAC iff sub-optimal AND multi-modal"）**证伪** → discriminator 是**噪声**非 modality（E-multi NULL）；机制三连 Q1→Q1b→Q1c（critic 侧排除 → actor β1 4→1 noisy **+23.5pp** → β1=1 clean 也更好）；C-1 给 FQL 自己的 anchor 旋钮做公平复赛 **RESCUE-FAIL**。**该 regime 内**：单一 ReBRAC β1=1.0 双轴 dominate FQL（worst-case 0.910 > 0.858）。
  2. **跨数据源 refine（SAC collector，最新，2026-05-26）**：把 1 的 "ReBRAC dominate" 放到 RL-trained behavior policy 数据源上**会翻转** → 算法排名是 **regime-dependent（algorithm × data-quality interaction）**，不是全局。**这才是本节的章级 headline**（不是 1 的 "ReBRAC dominate"）。
- **统一机制（贯穿两层，喂 §5.9）**：offline RL 性能由 **BC-anchor 目标质量与数据 regime 的匹配**决定、**不**由策略类先验表达力决定——中质量噪声数据 FQL 的 flow-denoised teacher 更优、高质量 clean 数据 ReBRAC 的 raw-action anchor 更优。
- **数字源**：FQL P2 = [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) + [`..._mechanism_diagnostic.md`](../docs/fql_succession_p2_mechanism_diagnostic.md) §9（cheatsheet [`..._paper_writing_index.md`](../docs/fql_succession_paper_writing_index.md) §2）；SAC collector interaction = [`arrival_v2_sac_collector_design.md`](../docs/arrival_v2_sac_collector_design.md) §4.0.10（SAC mexp **FQL 0.922 vs ReBRAC β1=1 0.800, Δ+0.123 CI[+0.022,+0.233]**；m_multi_mix 反向 +0.035 CI[+0.010,+0.065]）。
- **⚠ 口径红线**（§0.4 红线 5）：mexp 反超是**同源**（sprint1+2 同 collector/manifest）干净比较；"方向翻转"的 ReBRAC>FQL 一侧落在**异源** m_multi_mix（不同 dataset/manifest），cross-source 仅 **direction robust、magnitude 不可比**。interaction 只 claim 方向，不 claim 幅度比；**不要把 FQL P2 的 "ReBRAC dominate" 当章级 headline**。
- **复用资产**：verdict notebook 3 图已出（matrix / noise-axis / c1-rescue）；**Method prose / Related Work / Abstract 待写**（writing index §6 TODO，按 §0.5.2 dissertation voice 起草，不复用 paper 2 abstract）。
- **定位对接**：writing index §5 已预留**定位 (c) = thesis 章节 subsection**，与本方案一致 → FQL 作为本章一节，Method 与 ReBRAC 共享（§5.6）。
- **caveat**：跨-benchmark 探测在 `single_u15_cross` 撞 s0 observability **FLOOR**（clean priv 0.719 / noisy 0.098）→ scope caveat，不弱化 u10_cross 主结果。
- **§5.8.m 实施细节与可复现性**：FQL 方法增量（flow-matching teacher 损失全式，与 §5.6 ReBRAC dual-penalty 共享共有部分）+ FQL P2 2×2 modality×noise matrix 协议 + Q1/Q1b/Q1c 机制三连 ablation 设置 + C-1 RESCUE-FAIL 协议 + **SAC collector 39-run cross-source 设计**（sprint 1/2 collector / manifest / regime / 5×β1 sweep）+ §4.0.10 cross-source magnitude caveat 计算 + verdict notebook 3 图复现。

### §5.9 统一讨论 + 跨线 reconciliation + limitations + 本章小结
- **写什么**：(1) 统一机制（actor 信息 + BC-anchor 目标质量两轴）；(2) **β1 跨线 reconciliation**（paper 1 用 β1=4.0、FQL 线用 β1=1.0 不矛盾）；(3) sim2real implication；(4) limitations；(5) **本章小结**（dissertation 章末承上启下段，§0.5.3）。
- **β1 reconciliation 源**：[`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §6（四轴差异：reward / collector / 动作噪声 / seed pool；原线 β1=4.0 本是 seed-44 std-driven 选择）+ [`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §2.2.6 + [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) §8.1。**这是单章统一三线相比两篇独立 paper 的最大叙事增值点** — 在一章里能把"同一 BC-anchor 旋钮、相反最优点、由 anchor target 是否带噪决定"讲成统一机制。
- **本章小结**（§0.5.3 dissertation 体裁要求 — 不是 conference paper 的一句 punchline）：(a) 复述 §0.3 统一 takeaway；(b) 与博论整体 spine 衔接（本章为博论提供 deployable-sensor offline RL 的可行性证明 + 边界 + 算法决定因素）；(c) **自足的章级承接**（rev.7 零跨章：以抽象口径点出本章命题在博论整体论证中的位置，**不点名引述具体后续章**；与 §0.5.3 §5.9 小结、§2 §5.1 章级承接对称）。
- **复用资产**：paper 1 `discussion.tex`（rev.3，6 子节，含 §6.6 critical-regime boundary probe）+ `limitations.tex`（rev.4，L1–L12，含 L12 actor-fundamental ceiling 论证）+ `conclusion.tex`（rev.2，末句 scope tag 显式化 sub-critical regime + efficiency-v2 reward，指回 L12 / §6.6）。本章小结段**全新写**（paper 1 conclusion 是 8 页节级口吻，dissertation 章末需扩展承上启下，§0.5.3）。
- **限定红线**（paper 1 progress R7/R8）：anti-scaling reversal 与 sim2real implication 严格限定在 `s0 + Kármán wake + REMUS-100` 边界内，不包装为 algorithm-paper claim。
- **§5.9.m 实施细节与可复现性**：β1 reconciliation 四轴差异表（reward / collector / 动作噪声 / seed pool）+ paper 1 L1–L12 limitations 完整列表 + sim2real implication 的 scope tag 列表（s0 + Kármán wake + REMUS-100 + sub-critical regime + efficiency_v2 reward 五重限定）。

---

## 3. 三线缝合点（单章统一相比两篇 paper 的关键决策）

| 缝合点 | 问题 | 建议处理 |
|---|---|---|
| **reward 双轨** | efficiency_v2（paper 1/TD3BC）vs arrival_v2（broad-val v2/FQL），obs 40-D vs 48-D | §5.3 setup 显式交代两套 reward + 各自 obs 维度；每个 results 表标注 reward regime |
| **β1 = 4.0 vs 1.0** | 两线最优 β1 不同会被审稿人当矛盾 | §5.6 埋点（finalist 是 robustness winner）；§5.9 统一 reconciliation（四轴差异 + 统一机制）→ 化矛盾为 finding |
| **online ↔ offline floor 呼应** | online catastrophic floor 10% ↔ offline N2' 0% | §5.4 埋伏笔，§5.7 收束："offline + oracle 示范反而 underperform online SAC" |
| **FQL Method 去重** | FQL 与 ReBRAC 的 loss 公式若各写一遍会冗余 | Method 集中在 §5.6，§5.8 只增量讲 FQL distill 差异，共享 ReBRAC 公式 |
| **AsymCritic 两处出现** | online §7.7 AsymCritic ablation ↔ offline N2' asym-critic ablation | 统一术语；§5.7 主用 offline 版（actor-fundamental），online §7.7 版作 cross-line 旁证（§8 开放点 3 待定是否独立呈现） |

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
- Fig 2 seed dotplot（paper 1，C2 dep vs priv）→ §5.6
- Fig 3 Q-drift bar（paper 1，C3 β2=0）→ §5.6
- FQL verdict 3 图（matrix / noise-axis / c1-rescue，notebook 已出）→ §5.8

**本章需新作**：
- (5.4-a) A0 三 sensor success bar（s0/s1/s2 × efficiency_v2/arrival_v1）
- (5.4-b) arrival_v2 k-monotonicity 相位跃迁（k=4→8→12，含 seed=0 CROSS-SEED-RESCUE）
- (5.5) TD3+BC "2000<1000" 退化 + 纯 BC 对照（可选）
- (5.7) N2' ceiling decomposition（oracle 70% → offline+oracle 0%；含 asym ablation 行）
- (5.8) SAC collector 39-run algorithm×data-quality interaction heatmap（**章级 headline，强推**，§0.2 子问 4 锚）

---

## 6. 现有 paper 资产复用矩阵

| paper 1 资产 | 本章去向 | 改动量 |
|---|---|---|
| `setup.tex` (§3) | §5.3 | 中：泛化到多 reward/regime/sensor |
| `method.tex` (§4) | §5.6 | 小：基本照搬 |
| `experiments.tex` (§5) | §5.6 | 小：照搬四 findings |
| `discussion.tex` (§6，含 §6.6) | §5.7（§6.6 提级独立节）+ §5.9 | 中：拆分 + 融入三线 |
| `limitations.tex` (§7，L1–L12) | §5.9 | 中：合并各线 limitations |
| `conclusion.tex` (§8) | §5.9/本章小结段 | 大：升级到 dissertation 章末，扩展承上启下（§0.5.3） |
| **Appendix A–G** | **按内容主题融入各 §5.x 末"实施细节"子节（NO APPENDIX，§0.5.5）** | **重组**：A/G → §5.3.m + §5.6.m；B/C/D/E/F → 主要进 §5.6.m；TD3+BC 复现 → §5.5.m；online 复现 → §5.4.m；broad-val v2 → §5.7.m；FQL + SAC collector → §5.8.m |
| `refs.bib`（12 cites） | 本章 | 增：+ FQL (Park et al.) + flow-matching policy 综述 + dissertation reader-model 所需的领域综述（§0.5.4） |

**新写内容（无现成 LaTeX）**：
- §5.1 章级独立引言（全新写，dissertation 体裁，不复用 paper 1 abstract）
- §5.4 online 全节
- §5.5 TD3+BC 扩写
- §5.8 的 Method/RW/prose（按 §0.5.2 dissertation voice，不复用 paper 2 abstract）+ SAC collector 章级 headline
- §5.9 的 β1 reconciliation 段 + **本章小结**（§0.5.3 dissertation 章末承上启下，全新写）
- 各 §5.x.m 实施细节子节的重组写作（不"放在最后"，紧接各节论证后，§0.5.5）

---

## 7. 建议写作顺序（复用优先 → 新写殿后；rev.5 加入 §5.1 引言起草作为 #1）

> **rev.5 关键改动（2026-06-05）**：原 rev.4 写作顺序 #1 = "§N.1 setup 泛化"。rev.5 拆 §N.1 为 §5.1 章级独立引言 + §5.2 problem description；写作顺序新 #1 = **§5.1 章级独立引言（Markdown outline 优先于 LaTeX）**，作为 dissertation 体裁锚点 — 先在 Markdown 落地体裁、与 §0.5 体裁规范对照通过后再起 LaTeX。详见用户 2026-06-05 拍板"补 spec → 起 §5.1 outline → 再 LaTeX"步骤序。

1. **§5.1 章级独立引言（Markdown outline）** — 不直接起 LaTeX；先用 Markdown 起草段落级骨架（reader model 假设 / 与博论 §1 总引言对接句 / 本章 roadmap / 与前后章衔接），与 §0.5 体裁规范条对条 review；通过后再起 LaTeX 工程 `paper/thesis_ch5/`。这是 dissertation 体裁的第一次落地实验。
2. **§5.3 Problem description**（搬 paper 1 §3 setup.tex + 扩 reward 双轨 + dataset matrix 泛化）→ 立刻能编译，建立框架。
3. **§5.6 ReBRAC 主体**（搬 paper 1 §4–§6 + §5.6.m 实施细节子节集中承担 App A–G）→ 章重心先就位。
4. **§5.9 讨论/limitations/本章小结整合**（搬 paper 1 §6–§8 + 融三线 + 章末承上启下段全新写）→ 含 β1 reconciliation 这一最大增值点。
5. **§5.7 边界节**（提级 paper 1 §6.6 + L12，§0.5.6 负面结果作 finding 不降级）。
6. **§5.8 FQL + SAC collector 节**（writing index §2 cheatsheet 逐节填 + FQL Method 增量 + **SAC collector 39-run 章级 headline**）。
7. **§5.5 TD3+BC 短节**（baseline 故事扩写）。
8. **§5.4 online 节**（全新写 + 2 张新 figure，依赖最少、可最后补）。
9. **§5.2 Background & Related Work 短节**（研究背景与定位）→ 待主体节（§5.3 setup + §5.6 ReBRAC + §5.7 边界 + §5.8 算法对比 + §5.9 讨论）就位后写，使 gap 区分精确呼应正文论点；短节非综述、深度算法对比下放各方法节（§1 职责限定 note）。
10. **章 abstract（若博论目录要求）+ 统一 takeaway 章末收束** → 全章成稿后回写。

> 编译路径**新建** `paper/thesis_ch5/`（方案 A，paper 1 完全不动）：`cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`；成功后立即 `latexmk -c` 清中间文件（用户偏好，见 memory `feedback_latex_cleanup`）。LaTeX 工程结构（`\documentclass[11pt]{report}` + `\chapter` macro + `\input{sections/...}` 章节文件夹）在 §5.1 Markdown outline 通过后再 bootstrap。

---

## 8. 用户后续确认的开放点 / 已决策状态（不阻塞动笔）

1. ~~**章号 N**~~ — ✅ **resolved 2026-06-02**：**章号 = 第 5 章**（用户拍板 2026-06-02）。本 spec **rev.5 已全文落地** §5.1 / §5.2 / ... / §5.8 标号（rev.4 阶段曾保留 §N.k 占位以减少重编号噪音，rev.5 章号已 lock 后直接落地）。前后章衔接（§5 引言要承接多少前文）待整本博士论文目录拍板后再补，**不阻塞 §5.1 起草**。
2. ~~**SAC collector 39-run** 是否进正文（§5.8 末段）还是仅 appendix~~ — ✅ **resolved 2026-06-02**：**进第 5 章 §5.8 正文末段**，作为 §5.8 章级 headline。锚点 = spec §5.8 layer 2 + §0.2 子问 4（"算法排名是 regime-dependent (algorithm × data-quality interaction)，不是全局"），是 §0 中心命题"特权信息可有可无 + 真杠杆是 anchor 目标质量 × 数据 regime"的关键支撑。带 §0.4 红线 5 cross-source magnitude caveat（同源 SAC mexp 干净，异源 m_multi_mix 仅 direction-robust）。
3. **online §7.7 AsymCritic** 是否要与 §5.7 offline asym ablation 并列呈现（cross-line 一致性证据）还是只留 offline 版。 — 🟡 待用户拍板；可在写到 §5.7 时再定，**不阻塞 §5.1 起草**。
4. ~~**是否同时保留 paper 1 / paper 2 的独立投稿版**~~ — ✅ **resolved 2026-06-02**：**不另起任何 standalone paper**，全部素材整合进博士论文第 5 章。
   - **paper 1 ReBRAC**：31pp arXiv preprint draft（Phase 6.1 commit `932aca1`）作为第 5 章 §5.6 主干**直接复用**；不另投 CoRL/RA-L，也不单独上 arXiv（除非用户后续另议）
   - **paper 2（FQL standalone）**：**撤销启动**。素材并入 §5.8（FQL 算法对比 + SAC collector cross-source headline）；Method/RW/Abstract 不写独立版，按 §5.8 节级 prose 起草（§0.5.2 dissertation voice）
   - **paper 3（online SAC standalone）**：**撤销候选**。素材并入 §5.4（online RL：可学性与信息瓶颈）
   - thesis 第 5 章 = 当前唯一写作目标，本 spec 直接驱动其逐节起草
5. ~~**是否设 appendix**~~ — ✅ **resolved 2026-06-05（rev.5）**：**NO APPENDIX**。本章不设附录；paper 1 App A–G 全部按内容主题融入各 §5.x 末"实施细节"子节（§0.5.5）。这是 dissertation 章相对 conference paper 的体裁差之一：dissertation 章读者期待每节读完即拥有该节复现细节，不希望来回翻附录；conference paper 读者接受附录补充。
6. ~~**章节总篇幅 / cite 条数硬性目标**~~ — ✅ **resolved 2026-06-05（rev.5）**：**无硬性目标，内容优先于篇幅**（§0.5.8）。所有规模决策由「论证完整性 + 读者可追溯」裁决；写完一节后回顾"是否多说 / 少说"，不回顾"是否超页"。
7. ~~**§5.1 起草是直接起 LaTeX 还是先 Markdown outline**~~ — ✅ **resolved 2026-06-05（rev.5）**：**先 Markdown outline，对照 §0.5 体裁规范 review 通过后再起 LaTeX**。§5.1 章级独立引言是 dissertation 体裁的第一次落地实验，需双重 review（体裁是否到位 + 与 §0 中心命题是否一致），用 Markdown 比 LaTeX 迭代成本低。§5.1 outline 通过后 bootstrap `paper/thesis_ch5/` LaTeX 工程。
8. **博论整体目录与前后章衔接** — 🟢 **部分 resolved（rev.7，2026-06-07）**：§5.1 rev.3 已锁**零跨章引述**——引言"章级定位"段与 §5.9"本章小结"承接段均回写为**不点名引述邻章的自足口径**（§0.5.3 + §2 §5.1 已落地），故**不再依赖**具体前后章名称，原"待前后章名称拍板"的阻塞已消解。**仍 open** 的只剩整本博论目录与硬章号衔接（与开放点 1 相关），但**不影响** §5.x 各节起草（引言/小结已零跨章自足）。

> **跨开放点决策（2026-06-02 + 2026-06-05）**：用户明确 (a) "**直接合成博士论文第 5 章，不另写独立 paper**"（2026-06-02 lock）；(b) "**dissertation 章是正式学术论文，不是备忘录 / 实验报告 / 代码文档**"（2026-06-05 拍板，落入 §0.5 体裁规范）；(c) "**内容优先于篇幅，不预设页数 / cite 数**"（2026-06-05 拍板，§0.5.8 + §8 #6 lock）。本 spec 是第 5 章的写作 spec（不是"合并蓝图"）；§7 写作顺序 **#1 = §5.1 章级独立引言 Markdown outline** 是下一步动作。
