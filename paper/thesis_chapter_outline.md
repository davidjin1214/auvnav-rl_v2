# 博士论文章节写作 Spec：Deployable-Sensor Offline RL for AUV Wake Navigation

> 文档版本：rev.4（2026-06-02）
> 定位：**博士论文第 5 章 单章统一三线**（online SAC + TD3+BC/ReBRAC + FQL succession）的**核心贡献章**写作 spec。
>
> **✅ 写作策略（2026-06-02 用户拍板）：本 spec = 当前写作目标，不是未来蓝图。**
> Active priority = **直接合成博士论文第 5 章**；**不另起任何 standalone paper**。
> Paper 1 ReBRAC 已有的 31pp arXiv preprint draft（commit `932aca1`）作为 §N.4 主干**直接复用**；paper 2 FQL standalone / paper 3 online SAC standalone **均撤销**，素材分别整合进 §N.6（FQL 算法对比）与 §N.2（online RL 节）。
> Spec 全部内容（§0 中心命题 / §0.4 红线 / §3 缝合点 / §6 复用矩阵 / §N.7 β1 reconciliation 等）直接驱动第 5 章逐节起草，§7 写作顺序 #1 = §N.1 setup 泛化是下一步动作。
> rev.4 修订要点：撤销 rev.3 短暂存在过的"先独立 paper、后合并"框架（用户 2026-06-02 否决该框架）；恢复并明确"thesis 章 = 唯一写作目标"。
>
> ⚠ rev 历史：rev.1（2026-05-30 初稿）→ rev.2（2026-05-31 §0 spine 收紧 + §0.4 红线）→ rev.3（2026-06-02 sync paper Phase 6/6.1 + §N.7 rev refs 校准；曾短暂含"先独立 paper"框架）→ rev.4（2026-06-02 撤销"先独立"框架 + 章号=5 lock + SAC collector 进 §N.6 正文 + 各 standalone paper 撤销）。
>
> 与现有文档的关系：
> - 本文是 thesis chapter 的**总规划**；它**统摄**而非取代 [`paper/outline.md`](outline.md)（那是 ReBRAC paper 1 的 8 页 conference outline，对应本章 §N.3–N.5 的素材来源）与 [`paper/progress.md`](progress.md)（paper 1 LaTeX 工程进度，**31 页 arXiv preprint draft**，可作本章主干直接复用）。
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
| **能做到吗？** | ReBRAC 主线（**核心**，§N.4） | deployable-only ReBRAC-Q 在统计意义上**追平** privileged-critic 协议（worldcomp 0.928 vs 0.922, Welch p=0.92），并优于 vanilla TD3+BC 23–32pp。→ 特权信息喂 critic 对 **mean 不贡献**（5-seed Δ priv−dep=+0.6pp），只救 outlier seed 44（+12pp） |
| **靠什么机制？** | Online SAC（**铺垫/机制**，§N.2） | 难流场下 s0→s1 的 80pp gap **不是空间传感器瓶颈、是 actor 时序访问瓶颈**：s0+history k=12（~6s≈涡街周期 30–60%）单变量闭合到 s1 上界（§7.8/§7.9）。给 critic 喂特权流在 online 同样**不闭合 gap**（§7.7）——offline §4.5 asym ablation 的独立 echo |
| **边界在哪？** | broad-val v2（**边界**，§N.5） | critical regime（Re250/u15）下 s0 与 hull-integral 流弱相关，**oracle 示范 + 特权 critic 都救不了**（N2'=0.000；asym ablation = ACTOR_FUNDAMENTAL_CONFIRMED）→ actor-fundamental ceiling |
| **换更强算法会变吗？** | FQL succession（**对照/确证**，§N.6） | 表达力更强的 flow-matching 先验（FQL）**不普遍取胜**：算法排名 regime-dependent（SAC mexp 中质量 FQL 赢、saturated 高质量 ReBRAC 赢，两 CI 各自非零）。真正的算法决定因素是 **BC-anchor 目标质量 × 数据 regime 的匹配**，不是先验表达力 |

> 侧重：ReBRAC 是**唯一正面核心贡献**（正面回答"能做到吗"）；online=机制铺垫，broad-val=边界，FQL=对照确证 + 真实算法决定因素。四块都在为同一句"特权信息可有可无 + 真正杠杆是 actor 时序访问 / anchor 质量"服务。

### 0.3 统一 takeaway（章末收束）
> 对 deployable-sensor offline RL：正向杠杆是 **actor 对部署信号的时序访问** + **BC-anchor 目标质量与数据 regime 的匹配**；几个"直觉上该有用"的东西其实**非必需或不普遍有效**——给 critic 喂特权流不闭合 gap（online §7.7 + offline §4.5 两条**独立** asym ablation）、增加空间探头有效但**非必需且不可部署**（online §7.8 时序访问可替代）、更强生成式先验**不普遍取胜**（FQL regime-dependent）。这给水下机器人这类部署受限场景一条"**不依赖任何 simulator-only 信号即可逼近 online teacher**"的配方，及其在 critical regime 的失效边界。

### 0.4 ⚠ 机制归因红线（防答辩席反例，写作时严守）
1. **不写"actor 侧因素*决定*全部 mean 性能"**。critic **LayerNorm** 是单项最大 mean 杠杆（LN-off **−16.2pp**，远超 β2=0 的 −2.4pp；[`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §5.2b）。mean 归因只在**两个 BC penalty 之间**成立：actor β1 carry mean、critic β2 carry 跨数据集 Q-stability（review §0.3 / §5.2）。LayerNorm 是**第三条独立的"表征稳定性"轴**，与"特权信息"命题正交——属"必要基础设施"，**不进** §0.3 的"可有可无"清单。
2. **不写"ReBRAC over TD3+BC 的 +23–32pp 来自 actor anchor"**。在 β1=4.0 ↔ TD3+BC α≈0.25 actor-anchor 口径**匹配**下，recipe 差异同时含 dual penalty + critic LayerNorm + Q-norm；mean uplift 不能归给单一 actor 轴。
3. **不写"更多传感器无用"**。s1（多一空间探头）**确实**闭合 gap；正确表述 = "**非必需、不可部署，actor 时序访问可替代**"。三个"可有可无"项失效方式**各不同**：空间传感器=有效但非必需；特权 critic=不闭合 gap / 不抬 mean；强先验=regime-dependent——**不可混为"一律无用"**。
4. **LayerNorm claim 强度**：n=2 seeds 仅支撑 "necessary component **存在性**"，**不 claim "LN 比 dual penalty 重要"**（review §2.2.4）。
5. **FQL 反超的口径**：SAC mexp 上 FQL > ReBRAC β1=1（Δ+0.123, CI[+0.022,+0.233]）是**同源**（sprint1+2 同 collector/manifest）干净比较；"方向翻转"里 ReBRAC > FQL 一侧落在 m_multi_mix（**异源** dataset/manifest），cross-source magnitude 不可严格比较（仅 direction robust，§4.0.10 caveat）。interaction 只 claim **方向**、不 claim 幅度比。

---

## 1. 章节骨架总览

| § | 小节 | 估计篇幅 | 主要素材来源 | 资产状态 |
|---|---|---|---|---|
| N.1 | 引言与问题设定（任务/环境/s0 vs privileged/reward/数据集） | 中 | [`environment_design.md`](../docs/environment_design.md) + paper 1 `setup.tex` | ✅ **可直接复用** paper 1 §3 + Fig 1 |
| N.2 | Online RL：可学性与信息瓶颈 | 中 | [`online_rl_line_summary.md`](../docs/online_rl_line_summary.md) §1.1 + [`arrival_v2_experiment_report.md`](../docs/arrival_v2_experiment_report.md) §7.9 | 🟡 **新写**（仅 docs，无 LaTeX） |
| N.3 | Offline baseline：TD3+BC 拆瓶颈 | 短 | [`td3bc_phase0c_experiment_report.md`](../docs/td3bc_phase0c_experiment_report.md) + [`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §1 | 🟡 **新写/扩写**（paper 1 仅作 baseline 引用，无独立节） |
| N.4 | ReBRAC 主线：方法 + 四 findings | 长（本章重心） | paper 1 `method.tex` + `experiments.tex` + [`rebrac_experiment_report.md`](../docs/rebrac_experiment_report.md) | ✅ **几乎全文复用** paper 1 §4–§6 + Fig 2/3 |
| N.5 | 泛化边界：broad-val v2 → actor-fundamental ceiling | 中 | [`rebrac_broad_validation_v2_report.md`](../docs/rebrac_broad_validation_v2_report.md) §3–§5 + §4.5 | 🟡 **部分复用**（paper 1 Phase 6 已折叠为 §6.6 + L12，可提级为独立节） |
| N.6 | 算法对比：FQL succession 诚实负面 + BC-anchor 机制 | 中 | [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) + [`fql_succession_paper_writing_index.md`](../docs/fql_succession_paper_writing_index.md) | 🟡 **新写**（paper 2 候选，定位 (c) thesis subsection，Method/RW/Abstract 待写） |
| N.7 | 统一讨论 + 跨线 reconciliation + limitations | 中 | paper 1 `discussion.tex`/`limitations.tex` + 各线 discussion 段 | 🟡 **整合改写**（融合三线 + β1 reconciliation） |
| — | Appendix（复现/超参/全表/统计/SAC collector） | unlimited | paper 1 Appendix A–G + 各线 collection log | ✅ paper 1 Appendix 基本就位 |

---

## 2. 逐节素材映射

> 写作纪律（沿用 paper 1 progress.md §9 R1–R10）：mechanism > storytelling；每段 3–4 句封顶；数字实时回查源表；算法命名统一 **ReBRAC-Q**（不裸写 "ReBRAC"）。

### §N.1 引言与问题设定
- **写什么**：AUV/wake 任务动机 → deployable `s0`（DVL 单点）vs privileged hull-integral `[u_eq, v_eq]`（critic-only）→ reward → 数据集来源。本章统一 setup，后续三线共享。
- **数字/概念源**：[`environment_design.md`](../docs/environment_design.md)；state-space 速查 [`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §5.2（obs 维度：efficiency_v2 s0/h4=40-D；arrival_v2 s0/h4=48-D；privileged_obs dim=2 不堆叠）。
- **复用资产**：paper 1 `sections/setup.tex`（rev.2，99 行，含 dataset matrix 表 + 5 子节）+ **Fig 1 sensor schematic**（已出 pdf/png）。
- **本章新增**：需把 setup 从"单一 canonical 协议（cross_u10/efficiency_v2）"**泛化**到覆盖三线用到的多个 regime（efficiency_v2 ↔ arrival_v2 reward；Re150/u10 sub-critical ↔ Re250/u15 critical；s0/s1/s2 sensor）。这是统一三线的第一处缝合工作。
- **caveat**：reward 已从 `efficiency_v2`（paper 1 / TD3+BC 历史）迁到 `arrival_v2`（broad-val v2 / FQL）；两者 obs 维度不同（40 vs 48），setup 必须显式交代两套并存，否则后续表格维度对不上。

### §N.2 Online RL：可学性与信息瓶颈
- **写什么**：(1) A0 sensor screen — deployable s0 在简单 wake viable；(2) arrival_v2 prototype — H_information-bottleneck（s0+history k=12 闭合 80pp 的 s0→s1 gap）+ manifest universal-floor 概念。
- **数字源（thesis-grade）**：[`online_rl_line_summary.md`](../docs/online_rl_line_summary.md) §1.1（A0：efficiency_v2 s1 0.967 / s0 0.789 / s2 0.856）；[`arrival_v2_experiment_report.md`](../docs/arrival_v2_experiment_report.md) §7.9 / §7.9.7（k=12 在 2/3 seed saturate manifest floor 27/30=0.900；3-seed σ_final=0.038 << target 0.10）。
- **复用资产**：无 LaTeX，**全新写**。可新作 figure（A0 三 sensor bar + arrival_v2 k-monotonicity 相位跃迁图）。
- **叙事作用**：为全章奠定"瓶颈在 actor 信息访问，不在传感器硬件"的基调，**为 §N.5 actor-fundamental ceiling 埋伏笔**（online 难流场 catastrophic floor 10% ↔ offline N2' 跌破到 0%）。
- **caveat**：reward preset sweep（27 run×200k）**绝对数字不可引用**，仅作 `efficiency_v2` 选择的方法学依据；Sprint 0 preflight 不可作性能基线。坑（200k 假性否决 / flow 不一致 bug / efficiency_v2 OOB-suicide）可入方法论或 limitations。

### §N.3 Offline baseline：TD3+BC 拆瓶颈
- **写什么**：TD3+BC 把"数据越多越差"从协议伪象**纠正为真实现象**，拆出两个瓶颈：(1) 数据支持集结构（"2000<1000"退化在纯 BC 下同样成立）；(2) deployable critic 信息瓶颈（privileged-critic 关 48.5% gap）。→ 为 ReBRAC 创造问题设定。
- **数字源**：[`td3bc_phase0c_experiment_report.md`](../docs/td3bc_phase0c_experiment_report.md)；headline（[`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §1）：crosscomp 500/1000/2000 = 0.592/**0.672**/0.596；worldcomp deployable 最优退回 α≈0（纯 BC 0.858）/ privileged 0.922 / teacher 0.990。
- **复用资产**：paper 1 把 TD3+BC 仅作 baseline 行引用，**无独立节** → 本章需扩写成一个短节（thesis 容量允许讲完整 baseline 故事）。
- **叙事作用**：建立"为何需要 dual-BC penalty"的动机，自然过渡到 §N.4。

### §N.4 ReBRAC 主线：方法 + 四 findings（本章重心）
- **写什么**：方法（Q-normalized dual-penalty TD3+BC variant = ReBRAC-Q）+ 四条 paper-level finding。
- **四 findings 数字锚**（实时回查 [`rebrac_experiment_report.md`](../docs/rebrac_experiment_report.md)；镜像见 paper 1 progress.md §3）：
  - (i) crosscomp +23.0pp（1000）/ +32.2pp（2000），翻转"more data hurts"，std 不增反降
  - (ii) **deployable 0.928 追平 privileged-critic 0.922**，Welch p=0.9195（最强 sim2real narrative）
  - (iii) dual penalty cross-dataset 必需（β2=0 时 mean_target_q 漂 +46%/+98%）
  - (iv) critic LayerNorm ⊥ dual penalty（LN-off −16.2pp，远超 β2=0 的 −2.4pp，方向相反）
- **⚠ 与 §0 dispensability 命题的接口（写 Finding (iv) 时严守，防自相矛盾，cross-ref §0.4 红线 1）**：critic LayerNorm 是**第三条独立的"表征稳定性"轴 = 必要基础设施**，与 §0 的"特权信息可有可无"命题**正交**——它**不**进 §0.3 的"可有可无"清单。两条配套红线：(a) mean 归因只在 **actor β1 vs critic β2** 之间成立（β1 carry mean、β2 carry Q-stability），**不可**升级为"actor 侧决定全部 mean"（LN 才是单项最大 mean 杠杆）；(b) LN claim 强度 n=2 seeds 仅支撑 "necessary component 存在性"，**不 claim "LN 比 dual penalty 重要"**（review §2.2.4）。
- **复用资产**：✅ **paper 1 主干几乎全文可搬** — `method.tex`（rev.2，134 行）+ `experiments.tex`（rev.2，233 行，含 4 finding 子节 + 主表 + 2 ablation 表）+ **Fig 2 seed dotplot** + **Fig 3 Q-drift**。
- **关键命名红线**：全文 **ReBRAC-Q (ours)**，method 节首句声明 + §4.4 差异表（β1=4.0 ↔ TD3+BC α≈0.25）；不裸写 "ReBRAC"。
- **finalist**：`(β1=4.0, β2=2.0)`，是 **robustness winner（救 seed 44）**非 peak winner — 这是 §N.7 β1 reconciliation 的前提，§N.4 须埋点。

### §N.5 泛化边界：broad-val v2 → actor-fundamental ceiling
- **写什么**：broad-val v2 两 cell — N0（sub-critical）HOLDS；N2'（critical Re250）STRONG_NEGATIVE → **deployable s0 actor-fundamental partial-observability ceiling**；asym-critic ablation 排除 critic-fundamental 解释。
- **数字源**：[`rebrac_broad_validation_v2_report.md`](../docs/rebrac_broad_validation_v2_report.md) §3–§5（N0 0.850 HOLDS / N2' 0.000 STRONG_NEGATIVE，跌破 online catastrophic floor 10pp）+ §4.5（asym-critic ablation，verdict **ACTOR_FUNDAMENTAL_CONFIRMED**：完美 hull-integral flow 喂 critic，s0 actor 仍 0.000）。
- **复用资产**：paper 1 Phase 6 已把这部分折叠为 `discussion.tex` §6.6 + `limitations.tex` L12 → 本章可**提级为独立节**（thesis 容量允许；conference paper 受 8 页限折叠，thesis 不必）。
- **⚠ claim 边界红线**（[`fql_succession_paper_writing_index`](../docs/fql_succession_paper_writing_index.md) 同源原则，broad-val v2 report §4.5 明确）：
  - ✅ **可下**："privileged-flow critic 不能挽救 N2' 天花板；**排除 critic-fundamental**；HARDENS actor-fundamental"
  - ❌ **不可下**："proven actor-incapable / s0 actor 信息论上不可能" — asym=0 同时兼容"表征不可能"与"asym-critic 机制没把信息转化给 actor"两种读法。

### §N.6 算法对比：FQL vs ReBRAC — 先验表达力 vs anchor 目标质量
- **写什么（两层结论，必须按此顺序，否则会过头）**：
  1. **privileged-collector regime（FQL P2）**：核心 conditional-iff 假设（"FQL > ReBRAC iff sub-optimal AND multi-modal"）**证伪** → discriminator 是**噪声**非 modality（E-multi NULL）；机制三连 Q1→Q1b→Q1c（critic 侧排除 → actor β1 4→1 noisy **+23.5pp** → β1=1 clean 也更好）；C-1 给 FQL 自己的 anchor 旋钮做公平复赛 **RESCUE-FAIL**。**该 regime 内**：单一 ReBRAC β1=1.0 双轴 dominate FQL（worst-case 0.910 > 0.858）。
  2. **跨数据源 refine（SAC collector，最新，2026-05-26）**：把 1 的 "ReBRAC dominate" 放到 RL-trained behavior policy 数据源上**会翻转** → 算法排名是 **regime-dependent（algorithm × data-quality interaction）**，不是全局。**这才是本节的章级 headline**（不是 1 的 "ReBRAC dominate"）。
- **统一机制（贯穿两层，喂 §N.7）**：offline RL 性能由 **BC-anchor 目标质量与数据 regime 的匹配**决定、**不**由策略类先验表达力决定——中质量噪声数据 FQL 的 flow-denoised teacher 更优、高质量 clean 数据 ReBRAC 的 raw-action anchor 更优。
- **数字源**：FQL P2 = [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) + [`..._mechanism_diagnostic.md`](../docs/fql_succession_p2_mechanism_diagnostic.md) §9（cheatsheet [`..._paper_writing_index.md`](../docs/fql_succession_paper_writing_index.md) §2）；SAC collector interaction = [`arrival_v2_sac_collector_design.md`](../docs/arrival_v2_sac_collector_design.md) §4.0.10（SAC mexp **FQL 0.922 vs ReBRAC β1=1 0.800, Δ+0.123 CI[+0.022,+0.233]**；m_multi_mix 反向 +0.035 CI[+0.010,+0.065]）。
- **⚠ 口径红线**（§0.4 红线 5）：mexp 反超是**同源**（sprint1+2 同 collector/manifest）干净比较；"方向翻转"的 ReBRAC>FQL 一侧落在**异源** m_multi_mix（不同 dataset/manifest），cross-source 仅 **direction robust、magnitude 不可比**。interaction 只 claim 方向，不 claim 幅度比；**不要把 FQL P2 的 "ReBRAC dominate" 当章级 headline**。
- **复用资产**：verdict notebook 3 图已出（matrix / noise-axis / c1-rescue）；**Method prose / Related Work / Abstract 待写**（writing index §6 TODO）。
- **定位对接**：writing index §5 已预留**定位 (c) = thesis 章节 subsection**，与本方案一致 → FQL 作为本章一节，Method 与 ReBRAC 共享（§N.4）。
- **caveat**：跨-benchmark 探测在 `single_u15_cross` 撞 s0 observability **FLOOR**（clean priv 0.719 / noisy 0.098）→ scope caveat，不弱化 u10_cross 主结果。

### §N.7 统一讨论 + 跨线 reconciliation + limitations
- **写什么**：(1) 统一机制（actor 信息 + BC-anchor 目标质量两轴）；(2) **β1 跨线 reconciliation**（paper 1 用 β1=4.0、FQL 线用 β1=1.0 不矛盾）；(3) sim2real implication；(4) limitations。
- **β1 reconciliation 源**：[`rebrac_line_overview.md`](../docs/rebrac_line_overview.md) §6（四轴差异：reward / collector / 动作噪声 / seed pool；原线 β1=4.0 本是 seed-44 std-driven 选择）+ [`rebrac_mainline_review.md`](../docs/rebrac_mainline_review.md) §2.2.6 + [`fql_succession_p2_results.md`](../docs/fql_succession_p2_results.md) §8.1。**这是单章统一三线相比两篇独立 paper 的最大叙事增值点** — 在一章里能把"同一 BC-anchor 旋钮、相反最优点、由 anchor target 是否带噪决定"讲成统一机制。
- **复用资产**：paper 1 `discussion.tex`（rev.3，6 子节，含 §6.6 critical-regime boundary probe）+ `limitations.tex`（rev.4，L1–L12，含 L12 actor-fundamental ceiling 论证）+ `conclusion.tex`（rev.2，末句 scope tag 显式化 sub-critical regime + efficiency-v2 reward，指回 L12 / §6.6）。
- **限定红线**（paper 1 progress R7/R8）：anti-scaling reversal 与 sim2real implication 严格限定在 `s0 + Kármán wake + REMUS-100` 边界内，不包装为 algorithm-paper claim。

---

## 3. 三线缝合点（单章统一相比两篇 paper 的关键决策）

| 缝合点 | 问题 | 建议处理 |
|---|---|---|
| **reward 双轨** | efficiency_v2（paper 1/TD3BC）vs arrival_v2（broad-val v2/FQL），obs 40-D vs 48-D | §N.1 setup 显式交代两套 reward + 各自 obs 维度；每个 results 表标注 reward regime |
| **β1 = 4.0 vs 1.0** | 两线最优 β1 不同会被审稿人当矛盾 | §N.4 埋点（finalist 是 robustness winner）；§N.7 统一 reconciliation（四轴差异 + 统一机制）→ 化矛盾为 finding |
| **online ↔ offline floor 呼应** | online catastrophic floor 10% ↔ offline N2' 0% | §N.2 埋伏笔，§N.5 收束："offline + oracle 示范反而 underperform online SAC" |
| **FQL Method 去重** | FQL 与 ReBRAC 的 loss 公式若各写一遍会冗余 | Method 集中在 §N.4，§N.6 只增量讲 FQL distill 差异，共享 ReBRAC 公式 |
| **AsymCritic 两处出现** | online §7.7 AsymCritic ablation ↔ offline N2' asym-critic ablation | 统一术语；§N.5 主用 offline 版（actor-fundamental），online 版可作 cross-line 旁证 |

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
- Fig 1 sensor schematic（paper 1，s0 vs hull-integral 几何）→ §N.1
- Fig 2 seed dotplot（paper 1，C2 dep vs priv）→ §N.4
- Fig 3 Q-drift bar（paper 1，C3 β2=0）→ §N.4
- FQL verdict 3 图（matrix / noise-axis / c1-rescue，notebook 已出）→ §N.6

**本章需新作**：
- (N.2-a) A0 三 sensor success bar（s0/s1/s2 × efficiency_v2/arrival_v1）
- (N.2-b) arrival_v2 k-monotonicity 相位跃迁（k=4→8→12，含 seed=0 CROSS-SEED-RESCUE）
- (N.3) TD3+BC "2000<1000" 退化 + 纯 BC 对照（可选）
- (N.5) N2' ceiling decomposition（oracle 70% → offline+oracle 0%；含 asym ablation 行）
- (可选) SAC collector 39-run algorithm×data-quality interaction heatmap

---

## 6. 现有 paper 资产复用矩阵

| paper 1 资产 | 本章去向 | 改动量 |
|---|---|---|
| `setup.tex` (§3) | §N.1 | 中：泛化到多 reward/regime/sensor |
| `method.tex` (§4) | §N.4 | 小：基本照搬 |
| `experiments.tex` (§5) | §N.4 | 小：照搬四 findings |
| `discussion.tex` (§6，含 §6.6) | §N.5（§6.6 提级）+ §N.7 | 中：拆分 + 融入三线 |
| `limitations.tex` (§7，L1–L12) | §N.7 | 中：合并各线 limitations |
| `conclusion.tex` (§8) | §N.7/章末 | 中：升级到三线统一 takeaway |
| Appendix A–G | 本章 Appendix | 小：+ online/FQL/SAC collector 复现条目 |
| `refs.bib`（12 cites） | 本章 | 增：+ FQL (Park et al.) + flow-matching policy 综述 |

**新写内容（无现成 LaTeX）**：§N.2 全节、§N.3 扩写、§N.6 的 Method/RW/prose、§N.7 的 β1 reconciliation 段。

---

## 7. 建议写作顺序（复用优先 → 新写殿后）

1. **§N.1 setup 泛化**（搬 paper 1 §3 + 扩 reward 双轨）→ 立刻能编译，建立框架。
2. **§N.4 ReBRAC 主体**（搬 paper 1 §4–§6）→ 章重心先就位。
3. **§N.7 讨论/limitations 整合**（搬 paper 1 §6–§8 + 融三线）→ 含 β1 reconciliation 这一最大增值点。
4. **§N.5 边界节**（提级 paper 1 §6.6 + L12）。
5. **§N.6 FQL 节**（writing index §2 cheatsheet 逐节填 + 新写 FQL Method 增量）。
6. **§N.3 TD3+BC 短节**（baseline 故事扩写）。
7. **§N.2 online 节**（全新写 + 2 张新 figure，依赖最少、可最后补）。
8. **Abstract / 章引言 / 统一 takeaway 收束** → 全章成稿后回写。

> 编译沿用 paper 1：`cd paper && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex`；成功后立即 `latexmk -c` 清中间文件（用户偏好，见 memory `feedback_latex_cleanup`）。

---

## 8. 用户后续确认的开放点 / 已决策状态（不阻塞动笔）

1. ~~**章号 N**~~ — ✅ **resolved 2026-06-02**：**章号 = 第 5 章**（用户拍板 2026-06-02）。本 spec 中 `§N.1 / §N.2 / ...` 即未来 LaTeX 落地的 §5.1 / §5.2 / ...，spec 内部可继续用 `§N.k` 占位以减少全文重编号噪音，实际写章 LaTeX 时按章号 5 落地。前后章衔接（§5 引言要承接多少前文）待整本博士论文目录拍板后再补，**不阻塞 §N.1 起草**。
2. ~~**SAC collector 39-run** 是否进正文（§N.6 末段）还是仅 appendix~~ — ✅ **resolved 2026-06-02**：**进第 5 章 §N.6 正文末段**，作为 §N.6 章级 headline。锚点 = spec §N.6 layer 2 + §0.2 子问 4（"算法排名是 regime-dependent (algorithm × data-quality interaction)，不是全局"），是 §0 中心命题"特权信息可有可无 + 真杠杆是 anchor 目标质量 × 数据 regime"的关键支撑。带 §0.4 红线 5 cross-source magnitude caveat（同源 SAC mexp 干净，异源 m_multi_mix 仅 direction-robust）。
3. **online §7.7 AsymCritic** 是否要与 §N.5 offline asym ablation 并列呈现（cross-line 一致性证据）还是只留 offline 版。 — 🟡 待用户拍板；可在写到 §N.5 时再定，**不阻塞 §N.1 起草**。
4. ~~**是否同时保留 paper 1 / paper 2 的独立投稿版**~~ — ✅ **resolved 2026-06-02**：**不另起任何 standalone paper**，全部素材整合进博士论文第 5 章。
   - **paper 1 ReBRAC**：31pp arXiv preprint draft（Phase 6.1 commit `932aca1`）作为第 5 章 §N.4 主干**直接复用**；不另投 CoRL/RA-L，也不单独上 arXiv（除非用户后续另议）
   - **paper 2（FQL standalone）**：**撤销启动**。素材并入 §N.6（FQL 算法对比 + SAC collector cross-source headline）；Method/RW/Abstract 不写独立版，按 §N.6 节级 prose 起草
   - **paper 3（online SAC standalone）**：**撤销候选**。素材并入 §N.2（online RL：可学性与信息瓶颈）
   - thesis 第 5 章 = 当前唯一写作目标，本 spec 直接驱动其逐节起草

> **跨开放点决策（2026-06-02）**：用户明确"**直接合成博士论文第 5 章，不另写独立 paper**"。本 spec 是第 5 章的写作 spec（不是"合并蓝图"）；§7 写作顺序 #1（§N.1 setup 泛化）是下一步动作。
