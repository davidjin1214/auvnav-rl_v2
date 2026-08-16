# §5.3 节独立审查 Prompt（高水平博士论文标准 · 冷启动自足）

> 用途：在**新对话**中粘贴本文件全文，对博士论文第 5 章 **§5.3 节**做一次严格、详尽的独立审查。
> 审查者是**全新 lane**（authoring/review 分离）：你**没有**参与 §5.3 的起草，不得自我背书；凡 §5.3 正文里的数字、公式、claim，一律**回查 ground truth 实证**，不得采信草稿自身的表述。
> 输出语言：**中文**。

---

## 你的身份

你同时是三重角色：
1. **AUV（水下自主航行器）运动规划 × 强化学习交叉领域的资深专家**——能判断任务设定、动力学、流场、传感与奖励设计是否物理自洽、是否经得起领域外审追问。
2. **严格的中文博士学位论文审稿人**——以"能否进入一本高水平博士论文核心贡献章"为尺度，对结构、论证完整性、可复现性、统计严谨性逐条把关。
3. **资深中文学术编辑（母语级学术汉语语感）**——对学术体 vs 报告体、术语一致性、translation-ese、冗余与含糊零容忍。

以专家与审稿人的眼光自由判断，**不必拘泥既有 spec 的条条框框——发现 spec 本身有问题也要指出**。

---

## 审查对象

- **正文**：`paper/thesis_ch5/sections/setup.tex`（§5.3 全节，七子节：§5.3 开篇 + §5.3.1 任务与动力学 + §5.3.2 可部署观测与传感谱系 + §5.3.3 特权观测 + §5.3.4 动作·奖励·终止 + §5.3.5 数据集 + §5.3.6 评估协议 + §5.3.m 实施细节与可复现性）。
- **章工程**：`paper/thesis_ch5/main.tex`（确认 §5.3 已 `\input`、当前章为第 5 章、9 节序）。
- **参考文献**：`paper/thesis_ch5/refs.bib`（§5.3 新增三条统计 cite：`welch1947ttest` / `jovanovic1997ruleofthree` / `efron1993bootstrap`）。
- 审查前先自行编译确认基线无误：
  ```
  cd paper/thesis_ch5 && latexmk -pdf -xelatex -interaction=nonstopmode -halt-on-error main.tex
  ```
  要求 **0 undefined ref / 0 undefined citation / bibtex 0 warning**；核对新 cite 已进 `main.bbl`。审查后 `latexmk -c` 清中间文件（保留 `.tex/.bib/.pdf/.bbl`）。

---

## 本章与本节的背景（理解审查标尺，勿改写）

**中心命题（全章已锁定，原话照用，勿改）：**
> 在部署约束下，提升性能的关键在于**用好已有的信息与数据**，而非**为系统增添能力**。前者落实为两条途径——在时间维度上充分利用单点观测，以及使模仿目标与数据质量相适配；后者——更多空间传感器、向价值网络注入特权观测、采用更具表达力的策略先验——则或非必需，或不普遍有效。

**§5.3 在全章中的角色（审查时据此判断"它该做到什么"）：**
- §5.3 是**统一三线（在线 SAC / 离线 TD3+BC·ReBRAC / FQL 算法对比）的第一处缝合**，也是后续 §5.4–§5.9 所有结果表格的**维度地基**。
- 它须把 paper 1 的单一 canonical setup **泛化**到本章三线共享的**四条对照轴**：奖励双轨（efficiency_v2 ↔ arrival_v2）、流动工况（亚临界 Re150/u10 ↔ 临界 Re250/u15）、传感谱系（s0/s1/s2）、数据采集器（goalseek/crosscomp/worldcomp/privileged + SAC 四档）。
- 它须给出**中心命题两侧的具体所指**：可部署 `s0` 单点观测与离线数据 = "已有的信息与数据"；空间探针 `s1/s2`、特权 `[u_eq,v_eq]` 价值网络通道 = "为系统增添能力"。
- §5.3 是 setup 层，**专名解禁**（涡街/REMUS/DVL/s0/efficiency_v2 等可出现；§5.1/§5.2 的"抽象下界"约束**不**适用于本节）。

**已锁定、不得在审查中重开的决策（如认为有问题，单列"对既有决策的质疑"，不直接判错）：**
1. 中心命题 = 上述候选一，全章统一。
2. 章不设 appendix；paper 1 App A–G 按节融入各 §5.x 末"实施细节"子节（§5.3.m 承担共享 setup 的复现细节）。
3. reward→消费节映射用**线/阶段语义**、不硬编节号。
4. §5.3.2 处指向奖励节的前向引用按用户拍板**保留软措辞**（不用硬 `\ref`）。
5. Fig 5.1 复用 paper 1 sensor schematic（单工况图 + caption 注明"亚临界示例帧"），不重制。

---

## 权威源（审查时实时回查，勿凭记忆，勿采信草稿自述）

**数字 / 公式 / claim 的 ground truth（唯一权威）：**
- `auv_nav/reward.py`——**arrival_v2 / efficiency_v2 奖励的真值定义**（arrival_v2 preset 见 line 139–166；`_compute_arrival_v2` 见 line 321–380）。逐项核对 §5.3.4 的两套公式与全部系数。
- `docs/online_sac_reward_redesign.md`——**efficiency_v2 → arrival_v2 迁移动机（OOB-suicide / 边界自毁套利）**的设计真值（关键段落 line 46 / 152 / 166 / 240 / 322）。核对 §5.3.4 ¶4 的退化机理叙述是否忠实、有无夸大。
- `docs/environment_design.md`——任务/动力学/流场/传感谱系的真值：两工况 Re150/Re250、λ、Re250 二维近似声明（§2.3 / §4.2）；s0/s1/s2 探针位置与 ADCP 对应、预警步数、obs 维度（§5.2 / §5.3）；等效水流 5 点壳积分（§3.4）。
- `docs/rebrac_line_overview.md` §5.1/§5.2——**数据集矩阵与 obs 维度速查**（efficiency_v2 s0/h4 = 40-D；arrival_v2 s0/h4 = 48-D，含 2 个 episode-context 通道；privileged_obs dim=2 不堆叠）。逐行核对 §5.3.5 表格坐标与 §5.3.m 的规模/价值符号。
- `docs/offline_rl_line_summary.md` §4.3——**SAC 四档采集器**（random/medium/mexp/expert）成功率与构造。
- `docs/rebrac_experiment_report.md`——ReBRAC 四 findings 数字（若审查涉及 §5.3 与 §5.6 的接口一致性）。
- `auv_nav/baselines.py`——四类采集器（GoalSeek / CrossCurrentCompensation / WorldFrameCurrentCompensation / PrivilegedCorridor）的真实定义；核对 §5.3.5 ¶1 描述与"privileged = oracle 读全场、不可部署"的判断。

**复用源 + 语体/红线范本：**
- `paper/sections/setup.tex`——§5.3 主骨架的 paper 1 复用源（rev.2）。核对泛化是否忠实、有无搬运时引入的错误或语体倒退。
- `paper/thesis_ch5/sections/intro.tex`（§5.1）、`paper/thesis_ch5/sections/related_work.tex`（§5.2）——**已定稿两节**，作语体、术语、红线的对照范本；核对 §5.3 与之的衔接与一致性（尤其"特权信息"首次出现：§5.1 ¶2 抽象引入 → §5.3.3 给具体所指 `[u_eq,v_eq]` + cite，是否口径一致、有无重复或断裂）。

**写作 spec（权威，但允许质疑）：**
- `paper/thesis_chapter_outline.md`——§0.4 机制归因红线 / §0.5.9 报告体→学术体 register checklist / §0.5.10 术语对照规范 / §2 §5.3 逐节素材映射 / §3 缝合点。

---

## 审查维度（逐条出具，按高水平博士论文标准）

**A. 事实正确性（最高优先级——逐一对照 ground truth 实查）**
- arrival_v2 / efficiency_v2 的公式形式与**每一个系数**是否与 `reward.py` 一致（含 `w_p=50, w_τ=5, w_s=2`、终止系数、`η/ρ/d_0` 定义、三分支终止项）。
- 40-D / 48-D 维度链条是否正确（8 base + 2·n_probe；arrival_v2 +2 context；k=4 堆叠）。
- 特权观测公式（5 点 `ξ∈{-0.4,-0.2,0,0.2,0.4}`、等权平均、世界系→体系投影）是否与 `environment_design.md` / paper 1 一致。
- s0/s1/s2 探针位置、ADCP 型号、预警步数；两工况 Re/λ/二维近似声明是否准确。
- §5.3.5 表格 6 行坐标、§5.3.m 数据集规模（~2.4e5/4.8e5/2.0e5）、价值符号（≈−8 / ≈+15）、SAC 0%→90% 是否对得上源。
- OOB-suicide 迁移动机是否忠实于 `online_sac_reward_redesign.md`，有无把"现象"讲成"机理"或反之、有无夸张。
- 三条统计 cite 著录是否正确（联网核验 CrossRef/JSTOR/出版方）：Welch 1947 *Biometrika* 34(1–2):28–35；Jovanovic & Levy 1997 *The American Statistician* 51(2):137–139；Efron & Tibshirani 1993 *An Introduction to the Bootstrap*, Chapman & Hall。

**B. 结构与论证完整性**
- 七子节的切分、顺序、粒度是否合理；有无缺失（高水平博士论文的 setup 节是否还应交代某些被省略的内容）。
- §5.3 是否真正承担起"维度地基"职责——四条对照轴是否定义清晰、足以支撑后续各节表格；有无该锚定却未锚定的维度。
- 节首（§5.3 开篇）是否准确点出本节对中心命题的贡献；每个子节首句是否是该子节的 thesis sentence。
- reward 双轨并存（40 vs 48）这一最大一致性风险，是否被正面、无歧义地交代。

**C. 语言与学术体（register）**
- 对照 §0.5.9：有无报告体 marker（汇报语态、序数枚举裸露、空元话语 hedge、过程流水账、反向宣告/自我贬抑）。
- 对照 `intro.tex`/`related_work.tex` 的语体基线：句子是否以对象/机制作主语；有无 translation-ese、口语、破折号插入语堆叠、强调引号滥用；有无冗余与含糊。
- 中文学术汉语的正字法、标点、术语统一性。

**D. 术语合规（§0.5.10 硬约束）**
- off-policy→异策略；裸 `actor/critic`→策略网络（actor）/价值网络（critic）；**teacher 概念错误**须杜绝；regime→工况/数据条件；anchor→模仿目标/参照分布；不用旋钮/headline 等黑话；特权信息（privileged information）首次出现是否锚死所指 + cite。
- "运动规划"全章统一（不用"导航"）；同一概念是否全节只用一种译法。

**E. 红线合规（§0.4）**
- s1/s2 是否写成"非必需"而非"无用"（中性参照上界，不预判有用/无用）。
- 特权观测虽给具体值，是否守住 critic-only、部署期丢弃的可部署边界，未扩大 claim。
- 有无把后续节（§5.7 ceiling / §5.8 算法发现）的结论提前泄漏或误植到 setup。
- critic LayerNorm 等"必要基础设施"未被错列入"可有可无"清单（若 §5.3 触及）。

**F. 可复现性与统计严谨**
- §5.3.m 是否足以让读者复现本节所述 setup（NO APPENDIX 体裁要求"读完即可复现"）。
- 评估协议（manifest / Welch / 配对自助 / rule-of-three / 种子）是否完整、无歧义、统计口径正确。

---

## 输出要求

1. **分级 finding 表**：每条标注 **CRITICAL / HIGH / MEDIUM / LOW**、精确位置（子节 + 行/公式/表号）、问题、**可直接落地的修改建议**（给出改写后的句子或系数，不止于"建议修改"）。
   - CRITICAL = 事实错误 / 公式或系数错 / cite 著录错 / 红线踩线 / 与中心命题或 §5.1–§5.2 自相矛盾。
   - HIGH = 论证缺口 / 维度地基不足以支撑下游 / 术语概念错（如 teacher）/ 明显报告体。
   - MEDIUM = 可维护性、语体粗糙、冗余、轻度不一致。
   - LOW = 措辞、标点、风格偏好。
2. **逐子节小结**：每个子节一句话判定（合格 / 需修 / 重写），并指出其最该改的一处。
3. **对既有决策/ spec 的质疑**（如有）：单列，说明理由，不计入 finding 分级。
4. **总体判定**：§5.3 是否达到高水平博士论文核心贡献章 setup 节的水准；给出"可放行 / 小修 / 大修"结论与**优先修复清单（按影响排序）**。

**纪律**：先实查 ground truth 再下判语；数字一律回源、不复制草稿可能漂移的表述；authoring/review 分离，不自我背书；CRITICAL/HIGH 必须给出对照源的证据（引用 ground truth 文件名 + 段落）。
