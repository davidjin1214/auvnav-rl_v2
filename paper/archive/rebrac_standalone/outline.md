# ReBRAC AUV Wake Navigation — 论文写作大纲

> ⚠ **SUPERSEDED 2026-06-02**：本 outline 按 8 页 CoRL/RA-L 独立投稿设计；已被 [`thesis_chapter_outline.md`](../../thesis_chapter_outline.md) rev.4 取代。
> - paper 1 ReBRAC **不再独立投稿** — 31pp arXiv preprint draft（Phase 6.1 commit `932aca1`）作为博士论文**第 5 章 §N.4 主干直接复用**。
> - 本 outline 的 venue 分析（§0.1）、narrative spine（§0.3）、结构骨架（§1）、写作纪律（§8 R*）等**历史价值仍在**，作为 §N.4 节级写作时的参考；但**投稿口径相关段落**（venue 选择、CoRL/RA-L 8pp / IROS 6pp / 双栏切换等）请直接跳过。
> - 仅保留作存档；新动作请去 `thesis_chapter_outline.md`（rev.4）。
> - 📍 **2026-08-17 补注**：§7 建议"先拷方法节草稿到 `paper/draft.md`"——那个 markdown 中间稿 **从未创建**，写作直接落到了 LaTeX（本目录 `main.tex` + `sections/`）。不是丢失的文件。
>
> 📍 **节号与版本对照（2026-07-28 全仓指针体检补注）**：本文头注写于 2026-06-02，其中 `§N.k` 是当时 8 节方案的记法、`rev.4` 是当时的 spec 版本。现行章结构为 **10 节**；spec 现行 rev **不在此写死**（写死正是本次体检查出的腐化源），以 [`CLAUDE.md`](../../../CLAUDE.md) 文档索引表为准。节号对照：**§N.2 → §5.5**（Online RL）、**§N.4 → §5.7**（ReBRAC-Q 主线）、**§N.5 → §5.8**（泛化边界）、**§N.6 → §5.9**（算法对比：FQL + SAC collector）。正文内 `§N.k` 一律照此读，**不逐处改写**。
>
> 文档版本：rev.1（2026-05-01）
> 基于材料：[`docs/rebrac_mainline_review.md`](../../../docs/rebrac_mainline_review.md) rev.2、[`docs/rebrac_experiment_report.md`](../../../docs/rebrac_experiment_report.md) rev.8、[`docs/rebrac_method_section_draft.md`](../../../docs/rebrac_method_section_draft.md)、[`docs/rebrac_statistical_test_followup.md`](../../../docs/rebrac_statistical_test_followup.md)、[`docs/rebrac_paper_writing_index.md`](../../../docs/rebrac_paper_writing_index.md)
> 作者视角：从 offline-RL × sim2real-robotics 交叉领域审稿人 / chair 的角度反推写作骨架。
> **本文不是 paper 草稿；是让作者按图作业的写作 spec。每节给出"写什么 / 数字哪里查 / 长度上限 / 审稿风险"。**

---

## 0. 写作前的三个先决问题（动笔前先选）

### 0.1 投稿目标 venue（决定 paper 风格）

| 风格 | 候选 venue | 长度 | 优先级 | 与现有材料的契合度 |
|---|---|---|---|---|
| **Robotics-applied（推荐）** | CoRL / RA-L / IROS / IEEE Journal of Oceanic Engineering | CoRL 8pp、RA-L 8pp、IROS 6pp、JOE long-form | **第一选择** | ★★★★★（CLAUDE.md 明示 "does not chase generic algorithm-paper improvements"；narrative 与 deployable AUV sim2real 完全咬合） |
| **ML-algorithmic** | ICLR / NeurIPS / ICML offline-RL workshop | 9pp + unlimited appendix | 备选 | ★★★（method 是 minimal variant，novelty 撑不起 main conf；但 "actor anchor 主导 mean，critic penalty 只贡献跨 dataset 稳定性" 这条机制 finding 在 workshop 是足够的） |
| **Sim2real / domain-specific journal** | Ocean Engineering / Robotics and Autonomous Systems | long-form 12-20pp | 长线 | ★★★★（可作为 conf paper 的 extended journal version） |

**默认假设**：本大纲按 **8 页 CoRL / RA-L 风格** 设计。若改投 IROS（6pp），把 §6.2 / §7.2 / §8 合并；若改投 ML venue，调换 §3 与 §4 顺序、把 §6 拆成 method section。

### 0.2 paper title 三个候选

1. **`Deployable-only Offline RL Closes the Privileged-Critic Gap on Underwater Wake Navigation`**
   *评*：narrative 最强；标题就是 finding ②；推荐用于 RA-L / IROS。
2. **`Behavior-Cloning–Anchored Offline RL for Sim-to-real AUV Navigation in Wake Fields`**
   *评*：把 method（actor BC anchor）和 application（AUV / wake）一并写出；推荐用于 CoRL。
3. **`A Minimal Dual-Penalty Recipe for Offline Reinforcement Learning under Single-Sensor Deployment`**
   *评*：去 application 化；推荐用于 ML workshop。

**默认假设**：本大纲按候选 1 写。

### 0.3 narrative spine（贯穿全文的三句话；改自 [review §0.3](../../../docs/rebrac_mainline_review.md)）

> 我们用一个 Q-normalized 的 dual-penalty TD3+BC 变体（实现上对应 ReBRAC 的 minimal recipe），在水下 AUV wake navigation 这个 sim2real 任务上把 deployable-only 的离线策略性能拉到了与 privileged-critic 协议持平的水平，且优于 vanilla TD3+BC 23~32pp。机制上，actor-side BC penalty 主导了 mean 性能的提升，critic-side penalty 不贡献 mean 但通过 target-Q 抑制 Q 高估、并对 outlier seed 提供 dataset-invariant 的稳定性。这一结果意味着，对水下机器人这种 deployable sensor 严重受限的场景，离线 RL 可以不依赖任何 privileged simulator 信息就把性能逼近 online teacher。

每写一段都默问一次：**这段在为这三句话哪一句服务？** 若答不上来，就该删。

---

## 1. Paper 整体结构总览（8 页 CoRL/RA-L 风格）

| § | Section | 估计长度 | 主要数据来源 |
|---|---|---|---|
| 1 | Introduction | 0.75 pp（含 contributions list） | review §0 + plan §1 |
| 2 | Related Work | 0.75 pp | world_model_and_offline_rl_survey.md |
| 3 | Problem Setup | 0.75 pp | environment_design.md |
| 4 | Method (Q-normalized Dual-penalty TD3+BC) | 1.0 pp | method_section_draft.md |
| 5 | Experimental Protocol | 0.5 pp | plan §4 §5 + report §3 §4 |
| 6 | Main Results（4 findings） | 2.0 pp | report §7 全 + review §5 |
| 7 | Ablations | 1.0 pp | report §7.13–§7.15 + review §5.2 / §5.2b |
| 8 | Mechanistic Discussion | 0.75 pp | review §2.1 + §2.3 |
| 9 | Limitations | 0.25 pp | review §4 + report §9 |
| 10 | Conclusion | 0.25 pp | review §0.3 + report §10 |
| — | Appendix（reproducibility, hyperparams, full tables） | unlimited（移到 supp） | report §11 + paper_writing_index §3 §4.4 |

总计 main body ≈ 8 pages double column。

---

## 2. Section-by-section 写作 checklist

### §1. Introduction（0.75 pp）

#### 目标

让一个不熟悉 AUV / 不熟悉 ReBRAC 的读者在 5 句话内理解：(i) 部署受限的水下场景为何需要 offline RL；(ii) 现有 offline RL 在 deployable-only 协议下做得不够好；(iii) 我们做了什么改善了什么；(iv) 4 个 finding 概览；(v) 论文组织结构。

#### 写作段落 checklist

- [ ] **Hook 段（3–4 句）**：水下 AUV 在 wake field 中导航是真实任务；现实部署只能用 DVL water-track 单点采样（`s0`）；teacher policy 用了 hull-integral effective flow（不可部署）；offline RL 是 sim2real 的自然选择。**素材**：CLAUDE.md "deployment-realistic sensor (s0 = DVL water-track only) as the main axis"；environment_design.md sensor 节。
- [ ] **Gap 段（2–3 句）**：vanilla TD3+BC 在 crosscomp dataset 上只有 0.672 success rate；privileged-critic 协议（critic 看 hull-integral）能关一半 gap，但 deployable-only 协议留下显著 deployable→teacher gap。**素材**：review §1.5 主对照表 TD3BC 行；review §0.1 现状段落 "TD3BC 主线 '2000 < 1000' 退化"。
- [ ] **Approach 段（2 句）**：we present a **Q-normalized dual-penalty TD3+BC variant**（minimal ReBRAC recipe + TD3+BC Q-normalization），在 4 个 dataset / 协议组合上系统评测。**素材**：method_section_draft §1。
- [ ] **Contributions list（4 bullet，每个 1 行）**：
  - **C1** — On crosscomp deployable, ReBRAC-Q outperforms TD3+BC by **+23.0pp / +32.2pp** on 1000/2000 episode datasets, and **reverses** TD3+BC 的 "more data hurts" 退化。
  - **C2** — On worldcomp, **deployable-only ReBRAC-Q matches privileged-critic TD3+BC**（0.928 vs 0.922；Welch's p=0.92）— deployable-only 协议在统计意义上等价于 privileged-critic 协议。
  - **C3** — Dual penalty is **dataset-invariant necessary**：在两个 dataset、两种 baseline Q 量级（worldcomp +15 / crosscomp -8）下移除 critic penalty 都让 `mean_target_q` 漂 +46~+98%。
  - **C4** — critic LayerNorm 与 dual penalty 是 **两个独立必要 component**：LN-off 退化 -16.2pp（远超 β2=0 的 -2.4pp），且退化方向与 β2=0 相反。
- [ ] **Roadmap 段（1 句）**：列出后续节顺序。

#### 关键数字（Intro 一定要点出）

`+23.0pp / +32.2pp`（C1）、`0.928 vs 0.922 / Welch's p=0.92`（C2）、`+46~+98% Q drift`（C3）、`-16.2pp LN-off`（C4）。

#### 审稿风险 + 应对

- **R1：算法 novelty 弱**（minimal variant of existing work）。
  - 应对：Intro 末段直接承认 "method is a minimal recipe"，把 contribution 重心放在 application + finding，不在算法本身。
- **R2：数据集太小（仅 1000-2000 episodes）**。
  - 应对：在 Setup §3 中说明 "small-data regime is the realistic deployment setting for AUV"，并补一句 "scaling to larger offline corpora is left to future work"。

#### 长度

3 段 + 1 个 4-bullet list ≈ 3/4 column。

---

### §2. Related Work（0.75 pp）

#### 目标

把本文定位在三条线的交叉点：(i) offline RL methods（TD3+BC, BCQ, ReBRAC, IQL, CQL）；(ii) sim2real for underwater robotics；(iii) asymmetric / privileged learning。**关键是讲清三条线各自的局限，让审稿人接受为何要写这篇 paper。**

#### 写作段落 checklist

- [ ] **§2.1 Offline RL with behavior anchoring**（1 段）：TD3+BC（Fujimoto and Gu, 2021）/ ReBRAC（Tarasov et al., 2023）/ BCQ / IQL / CQL 列举；点出 ReBRAC 提出 dual penalty + critic LayerNorm 是 minimal-recipe；本文用其 minimal recipe 但 actor loss 不一样（保留 TD3+BC 的 Q-normalization）。**素材**：method_section_draft §3 §4 表；world_model_and_offline_rl_survey.md。
- [ ] **§2.2 Sim2real for underwater navigation**（1 段）：水下 AUV sim2real 文献概览；REMUS-100 一类 platform；wake field navigation 任务；**强调本文与现有 underwater RL 工作不同点：deployable sensor `s0` 严格等同真实部署**。**素材**：environment_design.md sensor 节；CLAUDE.md REMUS-100 描述。
- [ ] **§2.3 Asymmetric / privileged learning**（半段）：critic 看 privileged 信息 / actor 不看 是 standard recipe；**本文逆向 finding**：privileged critic 在 ReBRAC 上不再是必需。**素材**：review §2.1.1。

#### 长度

3 段 ≈ 3/4 column。每段 4-6 句，引用 3-5 篇。

#### 审稿风险

- **R3：related work 漏掉某些 offline RL 方法**。
  - 应对：至少列 TD3+BC, BCQ, ReBRAC, IQL, CQL, AWAC；diffusion policy / SfBC 等较新 method 用 1 句话带过 + 注脚 "complementary to ours"。

---

### §3. Problem Setup（0.75 pp）

#### 目标

让读者在不读 environment_design.md 的情况下能复现：环境是什么、observation / action / reward 是什么、benchmark 是什么、deployable 与 privileged 的区别在哪里。

#### 写作段落 checklist

- [ ] **§3.1 Task and dynamics**（半段）：REMUS-100 6-DOF underwater vehicle；wake field 是预生成的 `wake_v8 Re150` flow；task = cross-stream navigation, target speed 1.5 m/s。**素材**：environment_design.md。
- [ ] **§3.2 Observation: deployable s0**（1 段）：base 8 channels（surge/sway/yaw/orientation/goal）+ 1 probe @ (0,0)（DVL water-track 单点）= **10-D**；history length 4。**强调"deployable" 严格 = 真实 REMUS-100 sensor**。
- [ ] **§3.3 Privileged observation (critic-only)**（半段）：`privileged_obs ∈ ℝ²` = body-frame `[u_eq, v_eq]` from EquivalentCurrentModel（hull-integral effective flow）；只有 privileged-critic 协议下 critic 看见，actor 永远只看 deployable。**素材**：CLAUDE.md "Asymmetric Critic with privileged hull-integral flow"；environment_design.md。
- [ ] **§3.4 Reward and termination**（半段）：`efficiency_v2` reward = progress + success - safety_cost；timeout / 达标 / 失败 终止条件；evaluation = `single_u10_cross_tgt15.json` manifest（100 episodes）（2026-08-24 注：括注的 100 是起草当时的计划值，该文件实为 30 回合；100 回合的 `benchmarks/single_u10_cross_tgt15_ep100.json` 由 `ffa20cc`（2026-05-20）才生成，落差即 `docs/archive/fql_succession/fql_succession_bug2_fix_decision.md` 记的 Bug 2。原文保留）。**素材**：environment_design.md。
- [ ] **§3.5 Datasets**（半段）：worldcomp / crosscomp = baseline policies 收集；1000 / 2000 episodes 是两个 budget；明示 dataset 分布与 evaluation manifest 一致。**素材**：plan §4，CLAUDE.md。

#### 关键概念图

**Figure 1**：左图 = AUV 在 wake field 中的轨迹示意；右图 = sensor 布局（actor s0 + critic privileged hull-integral）。这张图是 paper 的"封面图"，**必须直观让人 5 秒内 get 到 deployable vs privileged 的差别**。

#### 长度

5 小段 + 1 图 ≈ 3/4 column。

#### 审稿风险

- **R4：数据集生成不透明**。
  - 应对：明示 "datasets are collected by running existing baseline policies" + appendix 给出 baseline policy 的 hyperparams。

---

### §4. Method: Q-normalized Dual-penalty TD3+BC Variant（1.0 pp）

#### 目标

公式化 + 与 TD3+BC、原 ReBRAC 三方差别表 + implementation note。**几乎逐字拷自 [`docs/rebrac_method_section_draft.md`](../../../docs/rebrac_method_section_draft.md)**。

#### 写作段落 checklist

- [ ] **§4.1 Algorithm name and positioning**（半段）：直接拷 method_section_draft §1。明示 "alias ReBRAC-Q"，明示 "β1, β2 数字与 Tarasov et al. (2023) 不可直接对比"。
- [ ] **§4.2 Actor loss**（半段+公式）：拷 method_section_draft §2 actor loss 公式。**关键公式**：
  ```
  L_actor(θ) = − (1/mean(|Q_φ(s, π_θ(s))|).detach()) · E[Q_φ(s, π_θ(s))]
              + β_1 · E[‖π_θ(s) − a‖²]
  ```
- [ ] **§4.3 Critic loss**（半段+公式）：拷 method_section_draft §2 critic loss。
  ```
  L_critic(φ) = E[(Q_φ(s, a) − y(s, a, r, s'))²] + β_2 · E[‖a' − π_θ̄(s')‖²]
  ```
- [ ] **§4.4 Difference from original ReBRAC and TD3+BC**（半段+表）：拷 method_section_draft §3 表（actor `Q` term scaling 等 6 行）；拷 §4 等价关系（β1=4.0 ↔ TD3+BC α≈0.25）。**这是 method 节的关键诚信点**。
- [ ] **§4.5 Implementation note (footnote or short paragraph)**：直接拷 method_section_draft §5。
- [ ] **§4.6 Why this variant**（可选；若空间紧改放 discussion）：拷 method_section_draft §6 三 bullet。

#### 长度

5 小段 + 2 个 displayed equation + 1 个比较表 ≈ 1 column。

#### 审稿风险

- **R5（最关键）：审稿人看到 "ReBRAC" 字样会以为是原版**。
  - 应对：method §4.1 第一句话就声明 "Q-normalized dual-penalty TD3+BC variant"；表格 §4.4 醒目放在节中段；引用时永远写 "ReBRAC-Q (ours)" 或 "our variant"，不写 "ReBRAC"。
- **R6：没有原版 ReBRAC（non-Q-normalized）的 sanity run**。
  - 应对：limitations 节写 "we leave the un-normalized variant for future work"；强调本文 contribution 是 "actor anchor 主导 + critic penalty 跨 dataset 稳定" 这一机制结论，与是否 Q-normalize 无关。

---

### §5. Experimental Protocol（0.5 pp）

#### 目标

reproducibility 必备最小集合。让读者知道：seeds 多少、test episodes 多少、checkpoint 选择规则、训练长度、benchmark 是哪个、TD3+BC baseline 的 α 是多少。

#### 写作段落 checklist

- [ ] **§5.1 Canonical protocol**（1 段）：5 seeds (42-46)、val=40 / test=100、TRAIN_EPOCHS=64、CHECKPOINT_EVERY_EPOCHS=8、selection rule `success_rate → return → -safety_cost → -time`、benchmark `single_u10_cross_tgt15.json`。**素材**：plan §4；review §5.4。
- [ ] **§5.2 Compared methods**（1 段）：
  - **TD3+BC** (Fujimoto and Gu, 2021)：α=0.25（与 ReBRAC-Q β1=4.0 等价）；baseline。
  - **ReBRAC-Q (ours)**：β1=4.0, β2=2.0, critic LayerNorm on, actor LayerNorm off, hidden=256, layers=3。
  - **Privileged-critic 协议**：critic 额外接收 hull-integral `[u_eq, v_eq]`；actor 不变。
- [ ] **§5.3 Datasets and protocols matrix**（1 段 + 表）：4 行 × 2 列 = `(crosscomp-1000, crosscomp-2000, worldcomp-1000) × (deployable, privileged)`。**素材**：review §1.5 主对照表。

#### 长度

3 小段 + 1 个 protocol matrix 表 ≈ 1/2 column。

---

### §6. Main Results（2.0 pp）

#### 目标

依次展开 4 个 finding，每个 finding 一节，每节有：标题 = finding 内容 / 1 段叙述 / 1 表 or 1 图 / 1 句机制注脚。

> **写作纪律**：所有数字从 [`docs/rebrac_experiment_report.md`](../../../docs/rebrac_experiment_report.md) 实时查；不要从 review 或本大纲拷数字（review 与本大纲是镜像，可能滞后）。

#### §6.1 Finding ①: ReBRAC-Q outperforms TD3+BC by +23.0~+32.2pp on crosscomp（半 pp）

- **段落**：开门见山报数字；强调 std 同时不增反降；highlight "2000 < 1000" 退化被翻转。
- **表 (Table 1, 主表)**：review §5.1 主对照表的 6 行（不含 worldcomp privileged）。**素材**：report §7（Stage C）。
- **关键数字**：`crosscomp-1000: 0.672 → 0.902 (+23.0pp)`；`crosscomp-2000: 0.596 → 0.918 (+32.2pp)`；std `0.045/0.036 → 0.021/0.030`。
- **机制句**：see §8 for detailed mechanistic analysis。

#### §6.2 Finding ②: Deployable-only matches privileged-critic on worldcomp（半 pp）

- **段落**：worldcomp deployable 0.928 ± 0.077 与 TD3BC privileged 0.922 ± 0.086 比较；Welch's t=0.10, p=0.92 → fail to reject；privileged-critic ReBRAC-Q 5-seed 0.9340 ± 0.0261 与 deployable 持平 (Δ=+0.6pp)。
- **表 (Table 2)**：3 行 × `(success mean ± std, gap closure %)` —— TD3BC priv / ReBRAC-Q dep / ReBRAC-Q priv。
- **图 (Figure 2)**：seed-wise success rate dot plot（5 seeds × 3 protocols），强调 seed 44 在 privileged 上被 +12pp 救回。**素材**：review §5.3 seed 44 表。
- **关键数字**：`Welch's p = 0.9195`；`paired bootstrap 95% CI = [-3.0pp, +4.2pp]`；`seed 44 dep=0.78 → priv=0.90 (+12pp)`。
- **机制句**：actor-side BC penalty 主导 mean；privileged critic 仅救 outlier seed 44 → see §8。
- **统计注脚**：拷 [`docs/rebrac_statistical_test_followup.md`](../../../docs/rebrac_statistical_test_followup.md) §2-§3 数字到 Table 2 caption。

#### §6.3 Finding ③: Dual penalty is dataset-invariant necessary（半 pp）

- **段落**：β2=2 vs β2=0 在 worldcomp 与 crosscomp 上的对比；`mean_target_q` 在两个 dataset 上分别漂 +46% / +98%；seed 44 在 crosscomp β2=0 下掉 -17pp。
- **表 (Table 3)**：4 行：`{worldcomp, crosscomp} × {β2=2, β2=0}` × `(success, mean_target_q)`。**素材**：review §5.2。
- **关键数字**：`worldcomp Δ Q = +7.08 (+46%)`；`crosscomp Δ Q = +8.12 (+98%)`；`seed 44 crosscomp β2=0 = 0.700`。
- **机制句**：β2 不抬 mean 但稳 Q → see §8 mean_target_q 跨 dataset 一致漂移。

#### §6.4 Finding ④: Critic LayerNorm and dual penalty are independent components（半 pp）

- **段落**：LN-off probe 让 mean 退化 -16.2pp（远大于 β2=0 的 -2.4pp）；std blow-up 12×；mean_target_q 朝更负 +46%；与 β2=0 退化方向相反 → 证明 LN ⊥ dual penalty。
- **表 (Table 4)**：3 行 three-way comparison：LN=on β2=2（Stage C） / LN=on β2=0（Stage E (a)） / LN=off β2=2（Stage F (B)）；列：success / std / mean_target_q。**素材**：review §5.2b。
- **关键数字**：`LN-off mean = 0.74 ± 0.25`；`Δ vs LN-on = -16.2pp`；`std blow-up 12×`；`mean_target_q -12.09 (vs Stage C -8.25, β2=0 -0.13)`。
- **机制句**：LN 抑制 critic 局部 Q 爆炸；dual penalty 抑制 actor OOD extrapolation；两者作用机制不同。

#### 长度

4 个 finding 节 × 半 pp = 2 pages。

#### 审稿风险

- **R7：表多图少**。
  - 应对：Figure 1 = sensor schematic（§3）；Figure 2 = seed-wise dot plot（§6.2）；Figure 3 = mean_target_q drift bar plot（§6.3 + §6.4）；Figure 4 = training curves of one representative seed（appendix）。
- **R8：finding ② 是 "持平" 不是 "更好" 会被认为 negative result**。
  - 应对：Intro 第一句就铺垫 "deployable-only 匹配 privileged-critic 是部署友好性论断"；conclusion 重申 "matching = success in this protocol setting"。

---

### §7. Ablations（1.0 pp）

#### 目标

把 §6 中的 finding ③ ④ 的支撑实验铺开，加上一个 actor-vs-critic 的 mechanistic ablation。

#### §7.1 critic-penalty-off ablation（拓 §6.3）（半 pp）

- 引出："we now show how the critic penalty contributes mechanistically by setting β2=0 across datasets and seeds"
- 数据：[`report §7.13–§7.14`](../../../docs/rebrac_experiment_report.md)（worldcomp probe + crosscomp Stage E (a)）。
- 关键 figure：`mean_target_q` 跨 dataset & 跨 β2 4-bar plot。

#### §7.2 LayerNorm ablation（拓 §6.4）（1/3 pp）

- 引出："since the original ReBRAC paper highlights critic LayerNorm as a key recipe component, we test its independent contribution"
- 数据：[`report §7.15`](../../../docs/rebrac_experiment_report.md) LN-off probe。
- 关键句："LN-off and β2=0 retreat in opposite Q-drift directions, indicating two independent stabilization mechanisms"。

#### §7.3 Privileged-critic ablation（拓 §6.2）（1/3 pp）

- 引出："we further verify that privileged-critic does not raise mean but only rescues outlier seeds"
- 数据：[`report §7.10–§7.12`](../../../docs/rebrac_experiment_report.md) Phase 2 5-seed。
- 关键 figure：seed-wise scatter，clearly showing seed 44 +12pp rescue, others ≈ 0。

#### 长度

3 小节 = 1 page。

---

### §8. Mechanistic Discussion（0.75 pp）

#### 目标

把 §6 / §7 的现象升级为机制论证，区分本文与原 ReBRAC 的 takeaway。

#### 写作段落 checklist

- [ ] **§8.1 actor BC penalty 主导 mean，critic penalty 不**（半段）：β2=0 在 typical regime 只掉 2~5pp mean → mean uplift 主要由 actor anchor (β1=4.0) 提供，critic anchor (β2) 只起 outlier rescue。**素材**：review §2.1.2 + §2.3.2。
- [ ] **§8.2 critic penalty's role: cross-dataset Q-magnitude invariance**（半段）：worldcomp Q ≈ +15、crosscomp Q ≈ -8，移除 β2 都让 Q 朝更不保守方向漂 +46~+98% → 论证 "critic penalty 提供 dataset-invariant 稳定性"。**素材**：review §2.1.2 + §2.3.3。
- [ ] **§8.3 seed 44 cross-(dataset, β2) closure**（半段）：seed 44 在 worldcomp deployable 崩、worldcomp privileged 救、crosscomp β2=2 稳、crosscomp β2=0 崩。**这是反驳 "seed noise" 的关键点**。**素材**：review §2.1.3 + §5.3。
- [ ] **§8.4 Implication for sim2real**（半段）：deployable-only protocol 在 ReBRAC-Q 上等价 privileged-critic protocol，意味着真实 AUV 部署不需要任何 simulator-only 信号。**素材**：review §2.1.1 + §0.3 三句话答辩稿。
- [ ] (可选) **§8.5 super-teacher hint**（1 短句 + appendix）：probe 阶段 `mean_test_return = 35.62 > teacher 32.19` 现象，列入 future work。**素材**：review §2.2.5。

#### 长度

4 小段 ≈ 3/4 column。

#### 审稿风险

- **R9：discussion 段过于啰嗦**。
  - 应对：每段 3-4 句封顶；不要重复 §6 / §7 已经报过的数字；只写 "implication"。

---

### §9. Limitations（0.25 pp）

#### 直接拷自 [review §4](../../../docs/rebrac_mainline_review.md) + [report §9](../../../docs/rebrac_experiment_report.md)。bullet 形式，5–7 项：

- [ ] L1: Q-normalized variant ≠ vanilla ReBRAC；β1 / β2 数字 not directly comparable to Tarasov et al. (2023)。
- [ ] L2: Single benchmark (`single_u10_cross_tgt15`)；evaluation manifest 与 dataset 同分布。
- [ ] L3: 5 seeds × test=100 是 RL 标准但 underpowered for tight-CI claim（n=5 Welch's t-test）；statistical claim 留 "持平 / not different"，不写 "更好"。
- [ ] L4: capacity / dropout / hidden_dim sweep 未做（仅 LN ablation 做了）。
- [ ] L5: dataset budget 1000–2000 episodes，未验证 large-scale offline corpora 行为。
- [ ] L6: hull-integral privileged signal 仅 dim=2，更高维 privileged 信号未试。
- [ ] L7: only one platform (REMUS-100)、one flow regime (Re=150, Ti=5%)。

#### 长度

3 段或 7-bullet list ≈ 1/4 column。

---

### §10. Conclusion（0.25 pp）

#### 目标

3 句话收尾，**几乎逐字拷自 review §0.3 narrative spine**。

#### 写作段落 checklist

- [ ] **第 1 句**：we propose ReBRAC-Q for AUV wake navigation。
- [ ] **第 2 句**：deployable-only matches privileged-critic; outperforms TD3+BC by 23–32pp on crosscomp。
- [ ] **第 3 句**：mechanism — actor BC penalty drives mean; critic penalty drives cross-dataset stability + outlier rescue。
- [ ] **第 4 句（implication）**：this means underwater AUV deployment does not need privileged simulator signals to approach online teacher。

#### 长度

1 段 4 句 ≈ 1/4 column。

---

### Appendix（移到 supplementary，不计 8 页内）

- **A1 Reproducibility — file paths & notebooks**：拷 [`docs/rebrac_experiment_report.md`](../../../docs/rebrac_experiment_report.md) §11 + [paper writing index §3](../../../docs/rebrac_paper_writing_index.md) Notebooks 列表。
- **A2 Hyperparameters**：winner config（[review §5.4](../../../docs/rebrac_mainline_review.md)）+ 完整 SAC / TD3+BC / ReBRAC-Q 表。
- **A3 Full results tables**：5-seed × test=100 raw numbers per stage（拷 report §7 中各表）。
- **A4 Statistical test details**：拷 [`docs/rebrac_statistical_test_followup.md`](../../../docs/rebrac_statistical_test_followup.md) §1-§5 全文。
- **A5 Method derivations**：β1 ↔ TD3+BC α 等价关系推导（拷 method_section_draft §4）。
- **A6 Training curves**：典型 seed 的 success_rate / mean_target_q / actor_loss / critic_loss vs epoch。
- **A7 Sensor / observation full spec**：拷 environment_design.md observation 节。

---

## 3. 论文图表清单（main body 用）

| Tag | 类型 | 内容 | 来源数据 | 占用空间 |
|---|---|---|---|---|
| Fig.1 | 概念图 | AUV in wake field + sensor schematic（s0 vs hull-integral） | environment_design.md + 自绘 | 1 column × 1/3 |
| Tab.1 | 主对照表 | 6 行 × `(success mean ± std)` for crosscomp & worldcomp | review §5.1 / report §7 | 2 column × 1/4 |
| Tab.2 | finding ② 表 | 3 行 worldcomp × `(mean ± std, gap closure, Welch p)` | review §1.5 + stats followup | 1 column × 1/4 |
| Fig.2 | seed dot plot | seed 42-46 × 3 protocols（worldcomp dep / TD3BC priv / RBQ priv） | report §7.10 §7.12 + review §5.3 | 1 column × 1/3 |
| Tab.3 | finding ③ 表 | 4 行 β2=0 vs β2=2 × `(success, mean_target_q)` | review §5.2 | 1 column × 1/4 |
| Tab.4 | finding ④ 表 | 3 行 LN/β2 three-way × `(success, std, target_q)` | review §5.2b | 1 column × 1/4 |
| Fig.3 | Q-drift bar | mean_target_q across `(dataset, β2, LN)` 6 bars | report §7.13 §7.14 §7.15 | 1 column × 1/3 |
| (Fig.4) | optional training curves | 1 representative seed across 4 protocols | report §7 | appendix only |

**图表总占用估算**：8 个 main 表/图 ≈ 3 columns ≈ 1.5 page；text 占 6.5 page；合计 8 page 刚好。

---

## 4. 全文统一术语表（写作时严格遵守，避免审稿人挑刺）

| 概念 | paper 中用词 | 不要用 |
|---|---|---|
| 我们的算法 | **ReBRAC-Q**（首次出现）/ **our variant** / **Q-normalized dual-penalty TD3+BC variant** | 直接写 "ReBRAC"（会与原版混淆） |
| 原版算法 | **original ReBRAC** / **ReBRAC (Tarasov et al., 2023)** | "vanilla ReBRAC" |
| 部署侧 obs | **deployable observation** / **single-sensor (s0) observation** | "raw observation" |
| privileged obs | **privileged hull-integral signal** / **privileged effective flow `[u_eq, v_eq]`** | "ground truth flow" |
| 评估指标 | **success rate**（写小数 0.928，不写 92.8%；除非在 abstract / conclusion） | 混用 |
| 主结论的描述 | "**matches**" / "**statistically not different from**" | "outperforms"（finding ② 不能用） |
| Δ 表述 | "**+23.0pp / +32.2pp**" 用 percentage points | "+23%"（会被读为 relative） |
| seed 数 | "5 random seeds (42–46)" | "5 trials" |

---

## 5. 写作风险地图（按 PR 概率排）

| 风险 | 概率 | 严重度 | 应对位置 |
|---|---|---|---|
| R1: 算法 novelty 弱 | 高 | 中 | Intro contribution list 重 application + finding |
| R5: ReBRAC 命名混淆 | 高 | 高 | Method §4.1 + 4.4 + abstract / conclusion 全文 ReBRAC-Q |
| R7: 表多图少 | 中 | 中 | 加 Fig.2 seed dot + Fig.3 Q-drift bar |
| R8: finding ② 是 "持平" | 中 | 高 | Intro / abstract 重新框架为部署友好性论断 |
| R3: related work 漏方法 | 中 | 低 | §2.1 列全 6 个 method + 1 句 diffusion policy 注脚 |
| R6: 没跑 vanilla ReBRAC | 中 | 中 | Limitations L1 显式声明；Method §4.4 实证 β1=4 ↔ α=0.25 |
| R2: 数据集太小 | 低 | 低 | Setup 强调"realistic deployment regime" |
| R4: 数据集生成不透明 | 低 | 低 | Appendix A2 给 baseline policy 描述 |
| R9: discussion 啰嗦 | 中 | 低 | 每段 3-4 句封顶 |

---

## 6. 写作时间表（D1–D6）

| 日 | 任务 | 产出 | 主要文档 |
|---|---|---|---|
| D1 | 通读 review.md + 拷 method draft → Method 节草稿 | §4 完成（first draft） | review + method_section_draft |
| D2 | 写 Setup（§3）+ Experimental Protocol（§5） | §3 §5 完成 | environment_design + plan |
| D3 | 写 Main Results §6（4 finding 节） | §6 完成 + Tab.1-4 + Fig.2 | report §7 实时查 |
| D4 | 写 Ablations §7 + stats followup 注脚 | §7 完成 + Fig.3 | report §7.13-§7.15 + stats followup |
| D5 | 写 Discussion §8 + Limitations §9 + Conclusion §10 + Intro §1 | §8 §9 §10 §1 完成 | review §2 §4 + §0.3 |
| D6 | 写 Related Work §2 + 整体 polish + figure 制作 + appendix | 全文 v1 + supp v1 | survey + 自绘 |

**第 7 天**：自审 — 用 review §2.2 / §4 反向 stress-test 全文（"审稿人在这里会问什么"）。

---

## 7. 一句话执行建议

**今天 D1**：先拷 [`docs/rebrac_method_section_draft.md`](../../../docs/rebrac_method_section_draft.md) 全文到 `paper/draft.md` §4，再通读 [`docs/rebrac_mainline_review.md`](../../../docs/rebrac_mainline_review.md)，回头按本大纲 §2.§4 顺序逐节展开。所有数字写到 `[TODO: from report §X.Y]` placeholder，最后一次性从 [report rev.8](../../../docs/rebrac_experiment_report.md) 实时核对 — 不要边写边查，会拖节奏。
