# FQL Succession (Paper 2 候选) 写作文档索引

> 文档版本：rev.1（2026-05-24，P4 收尾）
> 用途：把 FQL-vs-ReBRAC negative/mechanism 结果写成论文时的"读哪一份、抄哪一段"地图。
> **本身不持有任何实验数字 ground truth**；所有数字以 [`fql_succession_p2_results.md`](./fql_succession_p2_results.md)
> + [`fql_succession_p2_mechanism_diagnostic.md`](./fql_succession_p2_mechanism_diagnostic.md) §9 为准。
> 定位（standalone / 并入 ReBRAC paper / thesis 小节）**尚未锁定**（见 §5），本索引按 standalone 草拟、留可并入接口。

---

## 0. TL;DR

FQL Succession P2 已 **NEGATIVE 闭环（B+A：机制发现 + 诚实负面，NOT "FQL wins"）**，实验线全部完成，
**无 pending 实验**。写 paper 时只打开 3 份文档（Tier 1）。

**一句话 headline**：
> 表达力更强的 flow-matching 先验（FQL）并不系统性优于 ReBRAC（dual-BC）。原先看到的 "FQL 赢" 是
> ReBRAC 默认 β1=4.0 **mis-tuning** 的产物——单一固定 **ReBRAC β1=1.0 在 clean+noisy 双轴 dominate FQL**
> （worst-case-over-noise **0.910 vs 0.858**），且 FQL 拿到自己的 BC-anchor 扫描（C-1）仍过不了杆。存活的
> 贡献是一个干净的**机制**：*offline-RL 对 action noise 的鲁棒性由 BC anchor 的**目标质量**决定，最优锚强度随
> 目标噪声翻转*。

| 优先级 | 文档 | 角色 |
|---|---|---|
| **必读 + Results/Discussion/Limitations spine** | [fql_succession_p2_results.md](./fql_succession_p2_results.md) | 主报告（§0 TL;DR / §2-§4 结果 / §5 power / §6 机制 / §6.5 泛化 FLOOR / §7 claims / §9 provenance） |
| **写 Results 时打开当字典（ground truth）** | [fql_succession_p2_mechanism_diagnostic.md](./fql_succession_p2_mechanism_diagnostic.md) §9 | 权威 lab 记录（§9.1 Q1 / §9.4 Q1b / §9.6 E-multi / §9.7 Q1c / §9.9 power / §9.11 C-1） |
| **表/图的可执行来源** | [notebooks/fql_succession_p2_verdict.ipynb](../notebooks/fql_succession_p2_verdict.ipynb) | 纯分析，读 `results/` 现算所有表+图；图存 `docs/assets/fql_succession_p2/verdict_*.png` |

---

## 1. 文档分层

### 1.1 Tier 1 — 写作期常驻（3 份，见 §0 表）

- **results.md** 是 spine：Abstract/Intro/Discussion/Limitations 都从这里改写；§2-§4 是 Results 主体。
- **diagnostic §9** 是数字字典：任何 per-cell / per-seed / 统计值以此为准。
- **verdict notebook** 出 3 张图（`verdict_matrix.png` / `verdict_noise_axis.png` / `verdict_c1_rescue.png`）+ 所有表，重跑即更新。

### 1.2 Tier 2 — Setup / Related Work 期上下文（读但多数不拷）

| 文档 | 用途 | 何时翻 |
|---|---|---|
| [fql_succession_p2_main_spec.md](./fql_succession_p2_main_spec.md)（v1.4 CLOSED） | 2×2 设计动机 + 协议 + 原 conditional-iff 假设（已证伪，标 SUPERSEDED） | 写 **Experimental Setup / Method 动机**；解释为何 2×2、为何 noise 轴、为何冻结 β1/β2/distill_alpha_bc |
| [environment_design.md](./environment_design.md) | 环境、s0 sensor、reward、benchmark | 写 **Environment / Setup**；定义 s0、cross_stream、arrival_v2 |
| [world_model_and_offline_rl_survey.md](./world_model_and_offline_rl_survey.md) | offline RL 流派对比 | 写 **Related Work**；FQL / ReBRAC / TD3+BC 原论文引用 |
| [fql_succession_p2_collection_log.md](./fql_succession_p2_collection_log.md) | 4 个 dataset 收集 + GMM audit（advisory）+ §2.2 GMM false-positive 披露 | 写 **Setup（datasets）/ Reproducibility**；M-uni-noise 的 unimodal-by-construction 论证 |

### 1.3 Tier 3 — 历史 / 旁路（仅审稿人追问时回查）

- [fql_succession_gate_b_report.md](./fql_succession_gate_b_report.md) / [..._interim_report.md](./fql_succession_gate_b_interim_report.md) — **冻结超参的来源**（flow_steps=10 / distill_alpha_bc=1.0 / β1=4 β2=2）。仅当审稿人问 "FQL/ReBRAC 怎么调的" 时回查。
- [fql_succession_p2_xbench_spec.md](./fql_succession_p2_xbench_spec.md)（FLOOR-closed） — 跨-benchmark 泛化探测的 spec + 实测 FLOOR；**Limitations §6.5 的依据**。
- [fql_succession_p0p1_spec.md](./fql_succession_p0p1_spec.md) / [fql_succession_plan_v0.md](./fql_succession_plan_v0.md) — 早期 plan / spectrum。
- [fql_succession_bug2_fix_decision.md](./fql_succession_bug2_fix_decision.md) / [fql_succession_c4_threshold_revision.md](./fql_succession_c4_threshold_revision.md) — 方法学决策（ep100 manifest、c4 阈值）。
- **与本 paper 无关**：`auv_nav/{fql,rebrac}.py` 是冻结生产代码（只在 Method 引公式 + 行号）；online 线 / AUVHamNODE 线另属其它 paper。

---

## 2. Paper 章节 → 文档映射（写作 cheatsheet）

| Paper section | 主要文档 | 具体节 / 拷贝点 |
|---|---|---|
| **Abstract** | results §0 | TL;DR + headline + 机制 一段，改写 |
| **Introduction** | results §0 + spec §0/§1.1 | 动机 = "published-style FQL>ReBRAC iff sub-optimal AND multi-modal" 的 claim；贡献 = 证伪 + 机制 + 公平复赛 caution |
| **Related Work** | world_model_and_offline_rl_survey.md | offline RL 流派；FQL (Park et al.)、ReBRAC (Tarasov et al. 2023)、TD3+BC (Fujimoto & Gu 2021) **（TODO：需新写）** |
| **Method** | results §1.1 + §6 | §1.1 两个 BC 面（actor raw-action MSE β1 / critic BC β2；FQL student→flow-denoised teacher distill_alpha_bc）；§6 loss 公式 + 代码锚（rebrac.py L257/L284、fql.py L444/L471/L532）**（TODO：尚无 method_section_draft，需从 §1.1+§6 整理成 prose）** |
| **Experimental Setup** | environment_design.md + spec §2 + results §1.2-§1.3 + collection_log | env/sensor: environment_design；2×2 设计 + 协议: spec §2；paired metric (固定 ep100 manifest): results §1.2；datasets: collection_log |
| **Results — 2×2 matrix（noise 是 discriminator）** | results §2 + verdict §2 | `verdict_matrix.png`；E-multi NULL → modality 排除 |
| **Results — 机制三连（Q1→Q1b→Q1c）** | results §3 + diagnostic §9.1/9.4/9.7 + verdict §3 | `verdict_noise_axis.png`；Q1 critic 排除 → Q1b β1 4→1 +23.5pp → Q1c clean 也更好 |
| **Results — C-1 公平复赛（RESCUE-FAIL）** | results §4 + diagnostic §9.11 + verdict §4 | `verdict_c1_rescue.png`；FQL distill_alpha_bc 扫描 clean 最高 0.858 < 0.910 杆 |
| **Results — 统计 power** | results §5 + diagnostic §9.9 + verdict §5 | 固定 manifest ⇒ 配对；σ_train≈3.8pp；两大效应 n=2 即显著；头对头 NULL |
| **Discussion（机制综合）** | results §6 | BC-anchor 目标质量驱动鲁棒性；最优锚强度随目标噪声翻转；同 knob family 反向 optimum |
| **Limitations** | results §7 + §6.5 | §7 residual caveats（agent_final、benchmark ceiling）；**§6.5 跨-benchmark FLOOR = sensor-sufficiency 边界**（u15_cross：collector 0.719/0.098、offline 0.14） |
| **Conclusion** | results §0 + §7 | 不是 "FQL wins" paper；可发表的 falsification-grade 机制 + cautionary tale |
| **Reproducibility Appendix** | results §9 + 本文 §3 | 文件路径 + completed `.ipynb` 列表 |

---

## 3. Reproducibility — Notebooks 索引

每个实验对应一份 `*_completed.ipynb`（已含输出）。Paper appendix 列这些。

| Notebook | 阶段 | 对应结果 |
|---|---|---|
| [fql_succession_gate_b_completed.ipynb](../notebooks/fql_succession_gate_b_completed.ipynb) | Gate B | 冻结超参选择（flow_steps / distill_alpha_bc / β1 / β2） |
| [..._p2_run_cell_e_uni_completed.ipynb](../notebooks/fql_succession_p2_run_cell_e_uni_completed.ipynb) | 2×2 E-uni | clean / uni |
| [..._p2_run_cell_m_uni_noise_completed.ipynb](../notebooks/fql_succession_p2_run_cell_m_uni_noise_completed.ipynb) | 2×2 M-uni-noise | noisy / uni（原 "FQL 赢" cell） |
| [..._p2_run_cell_e_multi_completed.ipynb](../notebooks/fql_succession_p2_run_cell_e_multi_completed.ipynb) | 2×2 E-multi | clean / multi（NULL → modality 排除） |
| [..._p2_run_cell_m_multi_mix_completed.ipynb](../notebooks/fql_succession_p2_run_cell_m_multi_mix_completed.ipynb) | 2×2 M-multi-mix | noisy / multi（Null iff） |
| [..._p2_q1_critic_penalty_ablation_completed.ipynb](../notebooks/fql_succession_p2_q1_critic_penalty_ablation_completed.ipynb) | Q1 | critic 侧排除 |
| [..._p2_q1b_actor_penalty_ablation_completed.ipynb](../notebooks/fql_succession_p2_q1b_actor_penalty_ablation_completed.ipynb) | Q1b | actor β1 4→1，+23.5pp |
| [..._p2_q1c_actor_pen1_clean_completed.ipynb](../notebooks/fql_succession_p2_q1c_actor_pen1_clean_completed.ipynb) | Q1c | β1=1.0 clean 也更好 → dominate |
| [..._p2_c1_fql_alpha_sweep_completed.ipynb](../notebooks/fql_succession_p2_c1_fql_alpha_sweep_completed.ipynb) | C-1 | FQL distill_alpha_bc 公平复赛 RESCUE-FAIL |
| [..._p2_xbench_completed.ipynb](../notebooks/fql_succession_p2_xbench_completed.ipynb) | 跨-benchmark | u15_cross FLOOR（§6.5 泛化边界） |
| [..._p2_verdict.ipynb](../notebooks/fql_succession_p2_verdict.ipynb) | 聚合 | 纯分析，现算所有表+图+自动 verdict |

---

## 4. 数字快查（headline，写论文不必翻 results/diagnostic）

> **ground truth 在 [results.md](./fql_succession_p2_results.md) + [diagnostic §9](./fql_succession_p2_mechanism_diagnostic.md)；本节是镜像。如不一致，以 results/diagnostic 为准并立即修正本索引。** Per-cell / per-seed 全表见 results §2-§4。

**worst-case-over-noise（headline）**：

| config | clean | noisy | worst-case |
|---|---|---|---|
| **ReBRAC β1=1.0** | 0.910 | ~0.94 | **0.910** |
| FQL（frozen distill_alpha_bc=1.0） | 0.858 | 0.910 | 0.858 |
| ReBRAC β1=4.0（默认） | ~0.885 | 0.705 | 0.705 |

**机制三连**：Q1 critic-pen=0 noisy ≈ 0.715（vs β1=4 noisy 0.705，**+0.01 → critic 排除**）→ Q1b β1=1.0 noisy ≈ 0.94（vs 0.705，**+23.5pp，反超 FQL**）→ Q1c β1=1.0 clean ≈ 0.91（vs FQL clean 0.858，**clean 也更好 → 双轴 dominate**）。

**C-1 公平复赛**：FQL clean by distill_alpha_bc：0.3→0.74、**1.0→0.858（best, n=4）**、3.0→0.79、10.0→0.855。最高 0.858，差 0.910 杆 **−5.2pp → RESCUE-FAIL**。

**跨-benchmark 泛化（§6.5，FLOOR）**：`single_u15_cross`（U=1.5/Re250）：clean privileged collector **0.719**、noisy(σ=0.5) **0.098**、offline ReBRAC β1=1.0 gate **0.14**（in-training 仍缓升、out_of_bounds 主导）→ s0 observability floor，比较在此 regime 未定义（sensor-sufficiency 边界，非算法反例）。

---

## 5. 定位（positioning，尚未锁定）

用户 2026-05-24 决定：**先不锁定**，按 standalone 草拟、留可并入接口。三个候选：

| 候选 | framing / 长度 | 还需补什么 |
|---|---|---|
| **(a) Standalone 短文**（默认草拟方向） | negative-result / mechanism workshop-style；自带 Intro / Related Work / Abstract | Related Work 节 + Method prose + Abstract（见 §6 TODO） |
| **(b) 并入 ReBRAC 主 paper** | 作为 Discussion + Appendix（baseline-fairness / BC-anchor 机制旁证）；不单独成文 | 只需把 results §6 机制 + §4 公平复赛压成 1-2 段 Discussion + 1 个 appendix 表；最省 |
| **(c) Thesis 章节 subsection** | offline-RL 章里一个小节 | 与 ReBRAC 主线、broad-validation 串成统一叙事 |

**接口保留**：无论走哪条，Method（FQL/ReBRAC loss + 代码锚）与机制综合（results §6）都是共享的，先写它们不会浪费。

---

## 6. 投稿前 TODO（收尾清单）

实验全闭环，剩余纯写作：

- [ ] **定位决策**（§5 a/b/c）—— 决定后才好定 Abstract/Intro 篇幅。
- [ ] **Method prose** —— 尚无 `fql_method_section_draft`；从 results §1.1 + §6 整理 FQL（teacher/student/distill）与 ReBRAC（actor/critic dual-BC）的 loss 公式 + 与原论文差异。
- [ ] **Related Work** —— FQL (Park et al.)、ReBRAC (Tarasov et al. 2023)、TD3+BC (Fujimoto & Gu 2021)、flow-matching policy 综述；引用从 `world_model_and_offline_rl_survey.md` 起。
- [ ] **Abstract** —— 从 results §0 改写。
- [ ] **图** —— verdict notebook 已出 3 张（matrix / noise-axis / c1-rescue），投稿前确认 dpi/字号/配色达 camera-ready。
- [ ] **（可选）补种子** —— 仅当审稿人 push n=2；最便宜回应是 headline cell +1 seed（n=3），不是全矩阵重跑（更多 seed 只收紧 NULL，不翻盘）。

---

## 7. 收尾声明

- **P2 实验线全部闭环**（2×2 + 机制三连 + C-1 + 跨-benchmark 泛化探测），**无 pending 实验**。
- 结论 = **B+A：机制发现 + 诚实负面**，外加诚实的泛化边界（u15_cross FLOOR = sensor-sufficiency，非算法反例）。
- **publication-ready 程度**：Results / Discussion / Limitations / 统计 / 图 / 复现 notebooks 齐备；缺的只是 Method prose + Related Work + Abstract + 定位决策（§6）。
- 不动 Gate-B-frozen `auv_nav/{fql,rebrac}.py`；不重跑实验。
- 上层索引见 [`offline_rl_line_summary.md`](./offline_rl_line_summary.md) §3.5 / §5.2 F。
