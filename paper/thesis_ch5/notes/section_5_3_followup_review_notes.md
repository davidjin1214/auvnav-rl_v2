# Section 5.3 Follow-up Review Notes

Date: 2026-06-16

Scope: `paper/thesis_ch5/sections/setup.tex` after the first systematic correction pass and the follow-up discussion on reward framing, formal terminology, and setup/results boundaries. These notes are not about factual red-line errors. They define the next systematic revision plan for making Section 5.3 read as a doctoral thesis setup section rather than an experiment-tracking document.

## Updated Overall Verdict

The first correction pass has fixed the main factual problems: the historical efficiency-oriented reward now includes the soft safety penalty, the arrival-oriented reward uses the protected initial-distance scale, terminal ordering is no longer overstated, the `worldcomp` baseline is no longer described as a strict actor-observation baseline, statistical claims are less absolute, and the reproduction index has been added.

However, the section should not be revised by small local patches. The remaining issue is structural: the current wording still presents two code-named reward functions as parallel tracks and allows code-level identifiers to leak into formal prose. This creates avoidable confusion for a thesis reader. The next revision should therefore be systematic:

1. Reframe the reward subsection around the arrival-oriented reward as the main thesis setting.
2. Treat the earlier efficiency-oriented reward as historical context and design motivation, not as a co-equal "dual track".
3. Use formal Chinese academic terminology for concepts, reserving code names only for dataset paths, reproducibility tables, and, if necessary, first-use parenthetical mapping. Avoid literal mistranslations of implementation names.
4. Keep Section 5.3 as a setup section: define tasks, observations, rewards, data sources, statistical units, and reproduction coordinates without previewing which experiments succeed or fail.

## Decision Table

| Item | Decision | Rationale |
|---|---|---|
| Rewrite the reward subsection systematically | Immediate | The current "two rewards coexist" structure is the main source of confusion. The section should make the arrival-oriented reward the formal setting and use the efficiency-oriented reward only to motivate why the new terminal semantics are needed. |
| Remove code-style reward names from formal prose | Immediate | `efficiency_v2` and `arrival_v2` are implementation labels. In a formal Chinese thesis, use "效率导向奖励" and "到达导向奖励"; keep code names only in reproducibility fields or one-time mapping notes. |
| Neutralize setup/results labels | Immediate | Labels such as "成立侧", "失效侧", "teacher-gap", and "清洁参照" preview downstream conclusions or sound like lab notes. Replace them with neutral coordinates such as "亚临界到达任务", "临界到达任务", "可部署-特权协议对照", and "无动作噪声高质量参照数据". |
| Clarify statistical unit and bootstrap role | Immediate | The statistical protocol should state that random seed is the primary inferential unit. Episode-level paired bootstrap is only a fixed-manifest sensitivity check, not a replacement for seed-level inference. |
| Tighten the deployability boundary of `worldcomp` | Immediate | The policy is an idealized deployable flow-velocity compensation reference using the current effective flow estimate in world coordinates. In Chinese prose, do not translate "current" as "电流"; this chapter concerns fluid flow velocity, not electrical current. |
| Soften privileged observation register | Immediate | "真实流场" invites unnecessary realism objections. The safer phrasing is "物理含义明确的动力学有效输入". |
| Replace internal English labels, mistranslations, and experiment-management terms | Immediate | Terms such as `RL-trained`, `protocol-specific`, "历史主线", "设计验收", and "在线原型" should be translated into formal academic wording. Also replace "电流" and "船体" when they are mistranslations of flow velocity and AUV body/platform. |
| Rework reproduction table typography | Deferred until content stabilizes | `\resizebox{\linewidth}{!}{...}` is acceptable during content revision. Final typography can later use `tabularx`, shortened aliases, or split tables. |

## Priority 1: Reward Framing Must Be Rebuilt

### Problem

The current text says that the chapter contains two reward objectives and then explains them side by side. This is accurate as implementation history, but weak as thesis exposition. It gives the reader the impression that the chapter has two equally central reward landscapes, which makes later comparisons harder to interpret.

### Required Direction

The revised reward subsection should follow this logic:

1. Define the action space.
2. Introduce the historical efficiency-oriented reward only as the earlier setting used in the offline mainline.
3. Explain the design mismatch exposed by that reward: in difficult tasks, accumulated time penalty plus weak failure penalty can make early out-of-bounds termination attractive.
4. Present the arrival-oriented reward as the formal reward used for subsequent online learnability, boundary, and algorithm-comparison analyses.
5. Explain the arrival-oriented reward formula, terminal semantics, protected distance scale, elapsed-time fraction, final-distance ratio, and Markov-state context channels.
6. State the comparability boundary: historical efficiency-oriented results are retained for same-protocol comparability, but absolute returns are not compared across the two reward settings.

### Proposed Replacement Opening for the Reward Subsection

> 本章正式采用到达导向奖励作为任务成功率分析的主奖励设定。为说明其设计必要性，先回顾早期离线主线所使用的效率导向奖励。该奖励以时间惩罚和目标距离进度整形为主体，并叠加轻量安全代价与终止奖励；它适合在早期行为约束离线方法之间保持同口径比较，但在较难横流与逆流任务中暴露出目标错配：当策略难以到达目标时，持续累积的时间惩罚可能使提前出界获得比长时间接近目标更高的回报。

### Proposed Transition to the Main Reward

> 基于这一失配，后续在线可学习性、泛化边界考察与算法比较统一采用到达导向奖励。该奖励不再把"更快结束"作为隐含优化方向，而是显式区分到达、超时与硬失败，并通过早失败惩罚抑制快速出界。

### Proposed Comparability Boundary

> 早期效率导向奖励保留在离线主线中，仅用于与既有实验结果保持同口径可比；其回报数值不与到达导向奖励下的回报作直接比较。跨设定讨论只限于任务成功率、部署约束、信息利用和失效边界等机制性结论。

### Naming Rule

Use the following formal names in prose:

| Implementation label | Formal prose name |
|---|---|
| `efficiency_v2` | 效率导向奖励 / 早期效率导向奖励 / 历史效率导向奖励 |
| `arrival_v2` | 到达导向奖励 / 本文采用的到达导向奖励 |
| `worldcomp` | 世界坐标系流速补偿参照 / 理想化可部署流速补偿参照 |
| `privileged` | 特权走廊参照 / 特权示范采集器 |
| `teacher-gap` | 可部署协议与特权协议差距 |
| `RL-trained` | 在线 SAC 检查点采集器 |

Code names may remain in dataset paths, command-level reproduction entries, and optional first-use parenthetical mappings. They should not be used as conceptual labels in the running thesis prose.

### Translation and Terminology Red Lines

The next revision must explicitly avoid the following mistranslations:

| Problematic wording | Required wording | Reason |
|---|---|---|
| 电流补偿 | 流速补偿 / 来流补偿 / 等效流速补偿 | The chapter studies wake flow and local flow velocity, not electrical current. The implementation word "current" means fluid flow. |
| 当前电流 / 当前等效电流 | 当前流速 / 当前等效流速 / 当前等效来流 | Same reason; "current" should be interpreted as flow velocity. |
| 船体单点 / 船体附近 | AUV 单点 / 机体质心 / 载体附近 / AUV 附近 | The platform is an AUV, not a ship. "船体" is imprecise and changes the engineering object. |
| hull-integral flow | 沿 AUV 机体纵轴采样得到的壳积分等效流速 / 机体壳积分等效流速 | If "壳积分" is retained, it must be tied to the AUV body/vehicle, not translated as ship hull. |

Known places to check during the `setup.tex` revision include the opening paragraph of Section 5.3, the dataset paragraph describing `worldcomp`, and the Chapter 5 introduction sentence that currently refers to "船体附近局部流速".

## Priority 2: Dataset and Reproduction Tables Must Use Neutral Coordinates

### Problem

The dataset and reproduction tables currently mix setup coordinates with result-side labels. "成立侧" and "失效侧" are conclusions, not setup labels. "清洁参照" is informal. "teacher-gap" is a lab shorthand.

### Required Replacements

| Current wording | Replacement |
|---|---|
| 泛化边界（成立侧） | 泛化边界：亚临界到达任务 |
| 泛化边界（失效侧） | 泛化边界：临界到达任务 |
| 算法对比（清洁参照） | 算法对比：无动作噪声高质量参照数据 |
| 算法对比（数据质量轴） | 算法对比：SAC 检查点数据质量轴 |
| TD3+BC / ReBRAC 历史主线 | TD3+BC / ReBRAC：历史效率导向离线主线 |
| TD3+BC / ReBRAC teacher-gap | TD3+BC / ReBRAC：可部署-特权协议对照 |
| protocol-specific seeds | 固定种子集，见对应结果节 |

### Table Column Rule

In tables, rename "奖励" to "奖励设定" where possible. Use:

- 历史效率导向奖励
- 到达导向奖励

Dataset names may still contain implementation strings because they are reproduction identifiers. The conceptual columns should use formal prose names.

## Priority 3: Statistical Unit and Bootstrap Wording

### Problem

The current text still risks a pseudo-replication objection because episode-level paired bootstrap is described immediately after seed-level tests. A methods reviewer may ask whether episodes are treated as independent samples despite being nested within seeds.

### Required Direction

The setup section should state:

- Random seed is the primary inferential unit.
- Per-seed success rate is computed on a fixed manifest.
- Episode-level paired bootstrap is a sensitivity check over paired task instances within the fixed manifest.
- Strong parity or "catch-up" language must rely on seed-level effect sizes plus a pre-specified practical tolerance, not merely a non-significant p value.

### Proposed Wording

> 统计推断以随机种子为主要样本单位：每个种子先在固定评估集上得到成功率，再比较种子级均值。回合级配对自助法仅作为固定评估集下对配对任务实例的敏感性复核，用于检查结论是否依赖少数起终点或流场相位；它不替代种子级推断。涉及"追平"一类强表述时，后文同时报告种子级效应量与预设实践容忍范围。

## Priority 4: Deployability Boundary for the World-Frame Flow-Velocity Compensation Reference

### Problem

The current text correctly avoids calling `worldcomp` a strict `s0` actor-observation baseline, but it can be made more precise. This reference is stronger than a controller decoded only from the policy observation vector, yet it is still not a route-level oracle. In Chinese, this baseline must be described as a flow-velocity or effective-flow compensation reference, not as "电流补偿".

### Proposed Wording

> 世界坐标系流速补偿参照作为理想化的可部署流速补偿基线，使用导航坐标系下的当前等效流速估计修正期望对地航迹；它不读取未来流场或全场候选走廊，但其信息口径强于仅由策略观测向量直接解码的严格单点观测控制器。

### Boundary

Do not describe this baseline as:

- identical in sensor strictness to `goalseek` or `crosscomp`;
- an oracle;
- a controller that sees future flow fields;
- a baseline that directly uses the same vector available to the learned actor.

## Priority 5: Privileged Observation Register

### Problem

The phrase "驱动动力学的真实流场" is too strong. The quantity is physically meaningful and corresponds to the equivalent flow velocity used by the dynamics, but "真实流场" may invite objections about simulation realism, field measurement, or full-flow access.

### Proposed Replacement

Replace:

> 因而是驱动动力学的真实流场，而非任意的仿真隐藏状态

with:

> 因而是物理含义明确的动力学有效输入，而非与动力学无关的任意隐藏变量

The surrounding paragraph should still keep three boundaries clear:

- The privileged quantity is available only in training.
- The policy network never receives it.
- Deployment uses only the deployable observation.

## Priority 6: Formal Register and Internal Labels

### Problem

Several expressions still sound like internal experiment notes rather than dissertation prose.

### Required Replacements

| Current wording | Replacement |
|---|---|
| RL-trained 采集器族 | 由在线 SAC 策略检查点构成的采集器族 |
| online prototype / 在线原型 | 在线到达导向实验 / 在线可学习性实验 |
| 设计验收的核心约束 | 该奖励设计满足的核心约束 |
| 历史主线 | 历史效率导向离线主线 |
| 清洁参照 | 无动作噪声高质量参照数据 |
| teacher-gap | 可部署协议与特权协议差距 |
| protocol-specific seeds | 固定种子集，见对应结果节 |
| 电流补偿 | 流速补偿 / 等效流速补偿 |
| 船体单点 / 船体附近 | AUV 单点 / 机体质心 / AUV 附近 |

Avoid adding process language such as "后文将处理", "本表只是索引", "为了对齐实现", or "实验管理中". When such information is necessary, rewrite it as formal statements about scope, comparability, assumptions, or reproducibility.

## Priority 7: Reproduction Table Layout

### Decision

Defer typography until content stabilizes.

### Later Options

- Convert the table to `tabularx` with wrapped text columns.
- Split it into two tables: dataset coordinates and evaluation manifests.
- Shorten long dataset aliases in the table and define full names in prose or footnotes.

This should not block the current content revision.

## Systematic Modification Plan for `setup.tex`

### Step 1: Rewrite `\subsection{动作、奖励与终止}`

Make the subsection sequence:

1. action definition;
2. historical efficiency-oriented reward as design background;
3. failure mode of the efficiency-oriented reward;
4. arrival-oriented reward as the main thesis setting;
5. formulas and terminal semantics;
6. Markov context explanation;
7. comparability boundary with the historical setting.

The subsection should not be framed as two co-equal reward tracks.

### Step 2: Rename Conceptual Reward Labels Across Section 5.3

Replace conceptual uses of implementation labels:

- `efficiency_v2` -> 历史效率导向奖励 / 效率导向奖励;
- `arrival_v2` -> 到达导向奖励.

Keep implementation strings only in dataset paths and reproduction identifiers.

### Step 3: Clean Dataset Descriptions and Tables

Update `\subsection{数据集}` and Table `tab:ch5_datasets`:

- formalize baseline names;
- neutralize result-side labels;
- change "奖励" to "奖励设定";
- use Chinese reward names in conceptual columns.

### Step 4: Tighten Evaluation Protocol

Update `\subsection{评估协议}`:

- state seed-level inference first;
- demote episode-level bootstrap to sensitivity analysis;
- preserve Welch, bootstrap, and rule-of-three citations;
- keep "未检测到统计显著差异" and practical tolerance language.

### Step 5: Update Implementation and Reproduction Details

Update `\subsection{实施细节与可复现性}` and Table `tab:ch5_repro_index`:

- remove result-preview labels;
- replace internal English shorthand;
- correct mistranslations such as "电流" and "船体";
- keep dataset path strings as reproduction identifiers;
- use "固定种子集，见对应结果节" rather than `protocol-specific seeds`.

### Step 6: Delay Final Table Typography

Do not spend effort on `tabularx` or table splitting until content is stable. Revisit only during final layout pass.

## Non-Issues Confirmed

The following do not need to be reopened unless downstream sections change:

- Keeping Section 5.3 as the first unified setup for online SAC, offline TD3+BC/ReBRAC, and FQL remains structurally sound.
- Keeping `s0/s1/s2` in the setup section remains appropriate because this is where the sensor family is formally defined.
- Keeping the historical efficiency-oriented reward in the chapter is necessary because some completed offline experiments use it; the change is not deletion, but reframing.
- The current statistical references remain suitable; the required change is wording and inferential hierarchy, not citation replacement.
