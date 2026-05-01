# ReBRAC Paper 写作文档索引

> 文档版本：rev.1（2026-05-01）
> 用途：写论文时的"读哪一份、抄哪一段"地图。本身不持有任何实验数字 ground truth；
> 所有数字仍以 [`rebrac_experiment_report.md`](./rebrac_experiment_report.md) 为准。
> 维护：rebrac 主线 rev 升级（rev.8 → rev.9 …）时，同步更新本索引的"对应 rev"列。

---

## 0. TL;DR

写 paper 时只打开 4 份文档（Tier 1）。其它都是上下文或历史，等审稿人具体追问再回头查。

| 优先级 | 文档 | 角色 |
|---|---|---|
| **必读 + 可直接拷** | [rebrac_method_section_draft.md](./rebrac_method_section_draft.md) | Method 节草稿 |
| **必读 + spine** | [rebrac_mainline_review.md](./rebrac_mainline_review.md) | 4 个 finding 串联逻辑 + 数字快查表 |
| **写 Results 时打开当字典** | [rebrac_experiment_report.md](./rebrac_experiment_report.md) | 唯一 ground truth |
| **Main-table 注脚** | [rebrac_statistical_test_followup.md](./rebrac_statistical_test_followup.md) | Welch's p / bootstrap CI |

---

## 1. 文档分层

### 1.1 Tier 1 — 写作期常驻（4 份）

#### A. [`docs/rebrac_method_section_draft.md`](./rebrac_method_section_draft.md)

- **来源**：由 `notebooks/rebrac_paper_followup.ipynb` §5 自动生成（rev.1）
- **内容**：
  - §1 算法名 + positioning：**Q-normalized dual-penalty TD3+BC variant**（alias **ReBRAC-Q**）
  - §2 Actor / Critic loss 公式
  - §3 与原 ReBRAC（Tarasov et al., 2023）差异表
  - §4 与 TD3+BC（Fujimoto and Gu, 2021）差异 + β1=4.0 ↔ TD3+BC α≈0.25 等价性推导
  - §5 implementation note（推荐放进 paper 脚注）
  - §6 why-this-variant justification（discussion bullet 备选）
- **拷贝目标**：Method 节几乎逐字拷；§5 implementation note 可作为脚注或 appendix；§6 拆成 discussion 句子。

#### B. [`docs/rebrac_mainline_review.md`](./rebrac_mainline_review.md)（rev.2）

- **内容结构**：
  - §0 执行摘要 + 三句话答辩稿（**Intro / Conclusion 末段直接借用**）
  - §1 实验总结（§1.5 主对照表 — 5-seed × test=100）
  - §2 专家分析（§2.1 强项 / §2.2 paper 化前必须正面处理的弱点 / §2.3 容易被忽略的 insight）
  - §3 下一步建议（rev.2 4/4 已完成）
  - §4 caveats — **直接抄进 Limitations**
  - §5 附录核心数字快查表（§5.1 主对照 / §5.2 critic penalty 贡献 / §5.2b LN 贡献 / §5.3 seed 44 表 / §5.4 winner 配置 / §5.5 文档地图）
- **拷贝目标**：
  - **Intro 末段贡献声明** → §0.3 三句话答辩稿
  - **Discussion** 骨架 → §2 全部
  - **Limitations** → §4
  - **Results 总表** → §5.1, §5.2, §5.2b, §5.3
  - **Hyperparameters appendix** → §5.4

#### C. [`docs/rebrac_experiment_report.md`](./rebrac_experiment_report.md)（rev.8，1255 行）

- **角色**：所有 paper 数字的唯一 ground truth。其它文档（review / index / followup）都引用此处。
- **结构索引**：

  | 节 | 内容 | 拷到 paper 哪里 |
  |---|---|---|
  | §1 报告概述 | rev.8 大纲 | — |
  | §2 背景与实验动机 | TD3BC → ReBRAC 的 rationale | Intro / Method 动机段 |
  | §3 统一实验设置 | 协议、协议、benchmark | Experimental Setup |
  | §4 实验矩阵 | 全 stage overview | Setup 表 |
  | §5 Stage B0 训练预算 probe | epoch sweep | Appendix（hyperparam selection） |
  | §6 Stage B 最小 screening | β1/β2 网格 | Appendix |
  | §7 Stage C 5-seed 正式复核 | 主结果 | **Results 主体** |
  | §7.12 | Phase 2 升 5-seed → 0.9340 ± 0.0261 | finding ② 主结果 |
  | §7.13–§7.14 | Stage E (a) cross-dataset β2=0 | finding ③ |
  | §7.15 | LN-off probe → 0.74 ± 0.25（rev.8） | **finding ④** 主结果 |
  | §7.16 | stats test 数字汇总 | finding ② 注脚 |
  | §8 综合分析 | stage-to-stage 衔接逻辑 | Discussion 论证链 |
  | §9 局限性 | 11 项 | Limitations |
  | §10 最终结论 | 13 项 | Conclusion |
  | §11 结果文件索引 | 路径表 | **Reproducibility Appendix** |

#### D. [`docs/rebrac_statistical_test_followup.md`](./rebrac_statistical_test_followup.md)

- **来源**：由 `notebooks/rebrac_paper_followup.ipynb` §4 自动生成
- **内容**：Phase 1 deployable vs TD3BC privileged-critic 的 paired episode-level bootstrap（10000 resamples）+ Welch's t / 95% CI / 99% CI / gap closure CI
- **拷贝目标**：**Main results 表注脚 + Discussion 引一句 "Welch's p=0.92"**

### 1.2 Tier 2 — Setup / Related Work 期上下文（3 份，读但不拷）

| 文档 | 用途 | 何时翻 |
|---|---|---|
| [docs/rebrac_experiment_plan.md](./rebrac_experiment_plan.md)（rev.8） | 计划全文 — §4 canonical 协议 / §5 实现口径 / §6 各 stage 设计动机 | 写 **Experimental Setup**；解释 "为何 5 seeds、为何 dual penalty 必要、为何选 β1=4 β2=2" |
| [docs/environment_design.md](./environment_design.md) | 环境定义 — observation、sensor 布局、reward、benchmark | 写 **Environment / Setup**；定义 s0 / s1 / s2 与 task geometry |
| [docs/world_model_and_offline_rl_survey.md](./world_model_and_offline_rl_survey.md) | offline RL 流派对比 | 写 **Related Work** |

### 1.3 Tier 3 — 历史 / 旁路（仅审稿人追问时回查）

- **TD3+BC baseline 历史**（5 份）：
  - [td3bc_mainline_closure_plan.md](./td3bc_mainline_closure_plan.md)
  - [td3bc_phase0b_v2_experiment_report.md](./td3bc_phase0b_v2_experiment_report.md)
  - [td3bc_phase0c_experiment_design.md](./td3bc_phase0c_experiment_design.md)
  - [td3bc_phase0c_experiment_report.md](./td3bc_phase0c_experiment_report.md)（**crosscomp baseline 数字源**）
  - [td3bc_worldcomp_teacher_gap_experiment_report.md](./td3bc_worldcomp_teacher_gap_experiment_report.md)（**worldcomp baseline 数字源**）
  - 用途：所有 TD3+BC baseline 最终数字已落在 `rebrac_experiment_report.md` §7。**仅在审稿人追问 "TD3+BC 怎么调的、为什么取 α=0.25"** 时回查这 5 份。
- **与 paper 无关**：
  - [rlpd_design.md](./rlpd_design.md) — RLPD 是 online 线
  - [SAC_improvements_survey.md](./SAC_improvements_survey.md) — online 线
  - [systematic_improved_sac_experiment_*.md](./) — DEPRECATED
  - [online_rl_thesis_plan.md](./online_rl_thesis_plan.md) — 下一篇论文（asymmetric critic）

---

## 2. Paper 章节 → 文档映射（写作 cheatsheet）

| Paper section | 主要文档 | 具体节 / 拷贝点 |
|---|---|---|
| **Abstract** | review.md | §0.3 三句话答辩稿（改写） |
| **Introduction** | review.md + plan.md | review §0.1（现状）+ §0.3（贡献）；plan §1（动机） |
| **Related Work** | world_model_and_offline_rl_survey.md | offline RL 流派对比；ReBRAC / TD3+BC 原论文引用 |
| **Method** | method_section_draft.md | §1–§6 几乎全拷；§5 implementation note 进脚注 |
| **Experimental Setup** | environment_design.md + plan.md + report.md | env: environment_design.md；protocol: plan §4 §5；matrix: report §3 §4 |
| **Results — Main table** | review.md §5.1 + report §7 | 主对照 4 行（crosscomp-1000 / crosscomp-2000 / worldcomp-1000 dep / priv） |
| **Results — finding ① +23~+32pp 优势** | report §7.x（Stage C 各 dataset） | 数字 + 训练曲线 |
| **Results — finding ② priv≈dep** | report §7.12 + statistical_test_followup.md | "5-seed 0.9340 ± 0.0261；Welch's t=0.10, p=0.92；CI=[-3.0pp, +4.2pp]"；强调 seed 44 +12pp 救援 |
| **Results — finding ③ dual penalty cross-dataset** | report §7.13–§7.14 + review.md §5.2 | β2=0 在 worldcomp 退化 -1.8~-5.0pp、在 crosscomp 退化 -2.4pp；mean_target_q 飘 +46~+98% |
| **Results — finding ④ LN ⊥ dual penalty** | report §7.15 + review.md §5.2b | β2=2 LN-off → 0.74 ± 0.25（-16.2pp）；mean_target_q -12.09（朝更负 +46%）；与 β2=0 方向相反 |
| **Discussion** | review.md §2 | §2.1 强项 / §2.2 弱点回应 / §2.3 insight |
| **Limitations** | review.md §4 + report §9 | review §4 已浓缩为 paper-ready 版本；report §9 是详细版 |
| **Conclusion** | review.md §0.3 + report §10 | 三句话答辩稿 + report §10 13 项最终结论 |
| **Reproducibility Appendix** | report §11 + Notebook 列表（本文 §3） | 文件路径 + 8 个 completed `.ipynb` |
| **Hyperparameters Appendix** | review.md §5.4 + plan.md §5 | winner config + 实现口径 |

---

## 3. Reproducibility — Notebooks 索引

按时间顺序排列。每个 stage 对应一份 `*_completed.ipynb`（已含输出）。Paper appendix 中列这 8 份。

| Notebook | 阶段 | 对应 finding |
|---|---|---|
| [notebooks/rebrac_screen_completed.ipynb](../notebooks/rebrac_screen_completed.ipynb) | Stage B 最小 screening | β1/β2 网格 |
| [notebooks/rebrac_epoch_probe_completed.ipynb](../notebooks/rebrac_epoch_probe_completed.ipynb) | epoch budget probe | 训练长度选 64 epochs |
| [notebooks/rebrac_formal_completed.ipynb](../notebooks/rebrac_formal_completed.ipynb) | Stage C 5-seed 正式确认 | **finding ①** |
| [notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb](../notebooks/rebrac_worldcomp_phase1_deployable_completed.ipynb) | Stage D Phase 1 deployable | worldcomp 上 ReBRAC vs TD3BC |
| [notebooks/rebrac_worldcomp_phase2_privileged_completed.ipynb](../notebooks/rebrac_worldcomp_phase2_privileged_completed.ipynb) | Stage D Phase 2 privileged | **finding ②**（priv≈dep）3-seed 原版 |
| [notebooks/rebrac_worldcomp_critic_penalty_off_probe_completed.ipynb](../notebooks/rebrac_worldcomp_critic_penalty_off_probe_completed.ipynb) | β2=0 worldcomp probe | finding ③（部分） |
| [notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb](../notebooks/rebrac_stage_e_critic_penalty_off_crosscomp_completed.ipynb) | Stage E (a) β2=0 crosscomp | **finding ③** 闭合 |
| [notebooks/rebrac_paper_followup_completed.ipynb](../notebooks/rebrac_paper_followup_completed.ipynb) | **rev.8 paper-readiness** 4 项合并 | **finding ④** + Phase 2 升 5-seed + stats test + method draft |

---

## 4. 数字快查（写论文时不必翻 review.md / report.md）

> **数字 ground truth 在 [report rev.8](./rebrac_experiment_report.md)；本节是从 [review.md §5](./rebrac_mainline_review.md) 镜像，方便快速回查。**
> **如本节与 review / report 数字不一致，以 report 为准并立即修正本索引。**

### 4.1 主对照（5-seed × test=100）

| dataset | 协议 | TD3BC | ReBRAC | Δ |
|---|---|---|---|---:|
| crosscomp-1000 | deployable | 0.672 ± 0.045 | **0.902 ± 0.021** | +23.0pp |
| crosscomp-2000 | deployable | 0.596 ± 0.036 | **0.918 ± 0.030** | +32.2pp |
| worldcomp-1000 | deployable | 0.858 ± 0.080 | **0.928 ± 0.077** | +7.0pp |
| worldcomp-1000 | privileged-critic | 0.922 ± 0.086 | **0.9340 ± 0.0261** (5 seeds) | +1.2pp（Welch's p=0.92） |

### 4.2 critic penalty 贡献（β2=2 vs β2=0）

| dataset | β2=2 | β2=0 | Δ mean | Δ Q |
|---|---|---|---:|---:|
| worldcomp-1000 | 0.928 (5s) / 0.960 (2s) | 0.910 (2s) | -1.8pp / -5.0pp | +7.08（+46%） |
| crosscomp-1000 | 0.902 (5s) | 0.878 (5s) | -2.4pp | +8.12（+98%） |

### 4.3 LayerNorm 贡献（LN=on vs LN=off, rev.8 新增）

| dataset | LN=on | LN=off | Δ mean | Δ Q | std blow-up |
|---|---|---|---:|---:|---|
| crosscomp-1000 | 0.902 (5s) | **0.74 (2s)** | **-16.2pp** | -3.84（朝更负 +46%） | 0.021 → 0.255（**12×**） |

**关键**：LN-off 退化方向（Q 朝更负 + std blow-up）与 β2=0 退化方向（Q 朝更正）**不同**——证明 LN 与 dual penalty 是两个独立 component。

### 4.4 Winner 配置

```
β1 = 4.0       # actor BC penalty
β2 = 2.0       # critic-side BC penalty
hidden_dim = 256
num_hidden_layers = 3
critic_layernorm = on
actor_layernorm = off
normalize_q = on             # actor loss divides by |Q|.detach()
TRAIN_EPOCHS = 64
CHECKPOINT_EVERY_EPOCHS = 8
val_episodes = 40
test_episodes = 100
seeds = [42, 43, 44, 45, 46]
selection_rule = success_rate → return → -safety_cost → -time
```

### 4.5 4 个 paper-level findings

1. **TD3BC 上 +23~+32pp**（crosscomp deployable）
2. **priv ≈ dep**（worldcomp，Welch's p=0.92；seed 44 +12pp 救援）
3. **dual penalty 在 cross-dataset 上必要**（β2=0 在 worldcomp 退化 -1.8~-5pp、在 crosscomp 退化 -2.4pp）
4. **LN ⊥ dual penalty**（LN-off -16.2pp，与 β2=0 退化方向相反；capacity 解释失效）

---

## 5. 推荐写作顺序

1. **D1** — 通读 [`rebrac_mainline_review.md`](./rebrac_mainline_review.md) 全文（30 分钟）；记下 §0.3 的三句话答辩稿，确定 paper spine
2. **D1** — 拷贝 [`rebrac_method_section_draft.md`](./rebrac_method_section_draft.md) → Method 节草稿；按 paper 风格调整公式排版
3. **D2** — 写 Setup：用 [`environment_design.md`](./environment_design.md) + [`rebrac_experiment_plan.md`](./rebrac_experiment_plan.md) §3 §4 §5
4. **D3–D4** — 写 Results 主表（review §5.1）+ 4 个 finding 子节，**全部数字从 [`rebrac_experiment_report.md`](./rebrac_experiment_report.md) 实时查**
5. **D4** — 写 stats 注脚：拷 [`rebrac_statistical_test_followup.md`](./rebrac_statistical_test_followup.md) §2-§3
6. **D5** — 写 Discussion + Limitations：以 review §2 / §4 为骨架，把 4 个 finding 串起来
7. **D6** — Appendix：reproducibility（report §11 路径表 + 本索引 §3 notebook 列表）+ hyperparam（review §5.4）

**关键纪律**：写 Results 时只查 report，写 Discussion 时只查 review。这两份文档已分别为 "数据" 和 "论证" 调好分工，混读会拖慢节奏。

---

## 6. 维护说明

- 本索引随 ReBRAC 主线 rev 升级时同步更新（升级目标：rev.8 → 下次 rev.x）
- 数字 ground truth 永远在 `rebrac_experiment_report.md`；本索引 §4 是镜像
- 新 stage 落地时：(1) report 增 §7.x；(2) review §5 表新增行；(3) 本索引 §1.1.C 表 / §3 notebook 列表 / §4 数字镜像同步
- 若发现本索引与 review / report 不一致，以 report 为最高权威并立即修正索引
