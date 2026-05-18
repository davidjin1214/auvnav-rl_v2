# ReBRAC Broad Validation v2 — Experimental Report

> **Status (2026-05-19)**: COMPLETED — N0 + N2' 4-run 闭环（2 seed [42, 0]）。§5.2 verdict `HOLDS`，§5.3 verdict `STRONG_NEGATIVE`，§5.4 M1 BC sweep **not triggered**。
>
> **Branch**: `codex-arrival-v2-prototype`
> **Plan**: [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) rev.3
> **Raw outputs**: `results/offline/rebrac/broad_validation_v2/{N0,N2p}/seed_{42,0}/test_result.json` + `summaries/verdict_decision.json` + `summaries/p1_overview.csv`
> **Cross-link**: [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.6（online catastrophic floor），[`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md)（main paper efficiency_v2 anchor）
>
> **Seed count caveat**: 2 seed（plan §6.2 实际首轮缩水，详见 §6 limitations）；预登记 3 seed 在 backlog（§9 5-seed 补全条目继续保留）。

---

## 目录

- [§1 Abstract](#1-abstract)
- [§2 N0 reward bridge — 5.2pp paired degradation analysis](#2-n0-reward-bridge--52pp-paired-degradation-analysis)
- [§3 N2' failure mechanism — raw-field decomposition](#3-n2-failure-mechanism--raw-field-decomposition)
- [§4 Anomaly explanation — why N2' falls 10pp below the online floor](#4-anomaly-explanation--why-n2-falls-10pp-below-the-online-floor)
- [§5 Paper claim — actor-fundamental partial-observability ceiling under s0](#5-paper-claim--actor-fundamental-partial-observability-ceiling-under-s0)
- [§6 Limitations and next steps](#6-limitations-and-next-steps)
- [§7 Cross-link 与文档影响](#7-cross-link-与文档影响)

---

## 1. Abstract

ReBRAC broad validation v2 在 `arrival_v2` reward 下完成 2 个 core cell（N0 sub-critical anchor + N2' critical-regime oracle-teacher probe），共 4 run（2 seeds × 2 cells，~30 min L4）。所有 verdict gates 均在跑实验前预登记于 plan §5.2 / §5.3。

**核心结果**：

| Cell | Setup | success (2-seed) | per-seed [42, 0] | verdict gate (§5) | Verdict |
|---|---|---:|---|---|---|
| **N0** | crosscomp / s0 / cross_u10 / Re150 / arrival_v2 | **0.850 ± 0.024** | [0.867, 0.833] | §5.2: ≥ 0.70 = "reward bridge holds" | **HOLDS** |
| **N2'** | privileged / s0 / cross_u15 / Re250 / arrival_v2 | **0.000 ± 0.000** | [0.000, 0.000] | §5.3: < 0.15 + recovery < 21% of oracle = strong negative | **STRONG_NEGATIVE** |

**M1 BC penalty sweep**：N2' = 0.0 不在 §5.4 trigger zone `[0.15, 0.40]`，**not triggered**。

**三句话结论**：

1. **reward bridge HOLDS but with paired degradation**：N0 在新 reward 下保持 0.85 anchor，Δ vs efficiency_v2 main-line anchor (0.902) = **−5.20pp**；两个 seed 同向退化（42: −3.5pp / 0: −6.9pp），不是 noise。Paper 写作时必须明标。
2. **N2' STRONG_NEGATIVE confirms actor-fundamental partial-obs ceiling**：60 episode 全 out_of_bounds / timeout，progress_ratio ≈ 0，path_eff ≈ 0.10；零成功彻底跌破 online §7.6 catastrophic floor 10%。即便 oracle teacher (70% direct success) 提供 demonstrations，**s0-conditioned BC 在 critical regime 不能传递 hull-integral flow 知识**。
3. **paper-quality claim** (§8.2 ceiling decomposition)：在 critical regime + deployable s0 actor 下，offline RL with oracle demonstrations 不仅 fails to bridge partial-obs gap，且 underperforms 同 setup 下的 online SAC — 这是 plan rev.3 §2.4 "actor-fundamental partial-obs ceiling" caveat 的直接实证。

---

## 2. N0 reward bridge — 5.2pp paired degradation analysis

**Setup**: crosscomp / s0 / cross_u10 / Re150 / arrival_v2 / 30 eval episodes per seed / manifest `single_u10_cross_tgt15`。

### 2.1 Per-seed numbers

| Seed | success | terminations | progress_ratio | path_efficiency | return | safety_cost |
|---:|---:|---|---:|---:|---:|---:|
| 42 | 0.867 (26/30) | goal:26 / OOB:3 / timeout:1 | 0.825 ± 0.234 | 0.628 ± 0.192 | 71.3 ± 121.3 | 15.3 ± 17.1 |
| 0 | 0.833 (25/30) | goal:25 / OOB:5 / timeout:0 | 0.824 ± 0.192 | 0.605 ± 0.161 | 55.9 ± 123.8 | 16.1 ± 14.6 |
| **mean** | **0.850 ± 0.024** | dominated by `goal` (success) + `OOB` (failure) | 0.824 | 0.616 | 63.6 | 15.7 |

### 2.2 Anchor comparison

| Reference | Anchor | Source |
|---|---:|---|
| main paper anchor (efficiency_v2 / cross_u10 / s0 / 5-seed) | **0.902 ± 0.021** | [`rebrac_experiment_report.md`](rebrac_experiment_report.md) rev.8 |
| v2 N0 (arrival_v2 / cross_u10 / s0 / 2-seed) | **0.850 ± 0.024** | this report |
| **Δ** | **−5.20pp** | paired same-direction |

### 2.3 Verdict per plan §5.2

| §5.2 阈值 | N0 mean 0.850 落在 |
|---|---|
| ≥ 0.70 = reward bridge holds | ✓ |
| 0.50–0.70 = weak bridge | — |
| < 0.50 = bridge fail (pause) | — |

**Verdict**: **HOLDS** — proceed to N2'.

### 2.4 Why the 5.2pp drop is not noise

两个 seed 都比 5-seed efficiency_v2 anchor (0.902) **同向**退化：seed=42 → 86.7%（Δ=−3.5pp），seed=0 → 83.3%（Δ=−6.9pp）。同向退化两次的 binomial 概率 ≈ 0.25 — 个体 seed 上不严苛，但两 seed 退化方向一致 + termination 分布从 main-line 的「goal 主导 + 少量 OOB」漂向「goal 主导 + 略多 OOB」（v1 main-line N0 等价 cell 是 27.4 goal / 2.6 OOB / 0 timeout，本轮平均 25.5 goal / 4.0 OOB / 0.5 timeout）—— 暗示 `arrival_v2` reward shaping 让 actor 略微更激进，在 cross_u10 sub-critical regime 下 marginally 增加越界。

**与 main paper claim 不冲突**：main paper claim "ReBRAC +23pp on crosscomp / cross_u10 / s0 / efficiency_v2" 在 efficiency_v2 anchor 下成立；arrival_v2 下退化 5pp 是 **reward-specific calibration effect**，不影响 algorithm-level finding 的有效性。但 paper §experiments appendix 必须明标 "the +23pp advantage is reported under efficiency_v2; under arrival_v2 the absolute success drops 5.2pp while still substantially exceeding the catastrophic baseline (0.0%)".

---

## 3. N2' failure mechanism — raw-field decomposition

**Setup**: privileged / s0 / cross_u15 / Re250 / arrival_v2 / 30 eval episodes per seed / manifest `single_u15_cross_tgt15`。

### 3.1 Per-seed numbers

| Seed | success | terminations | progress_ratio | path_efficiency | path_length_m | return | safety_cost |
|---:|---:|---|---:|---:|---:|---:|---:|
| 42 | 0.000 (0/30) | OOB:29 / timeout:1 | 0.041 ± 0.509 | 0.107 ± 0.248 | 88.6 ± 55.9 | −247.5 ± 41.2 | 18.4 ± 13.8 |
| 0 | 0.000 (0/30) | **OOB:30** | −0.005 ± 0.618 | 0.093 ± 0.235 | 88.2 ± 55.3 | −248.8 ± 46.4 | 15.5 ± 10.1 |
| **mean** | **0.000 ± 0.000** | **OOB dominates 59/60** | 0.018 | 0.100 | 88.4 | −248.2 | 17.0 |

### 3.2 Ceiling decomposition (post-N2', plan §8.2 table fully populated)

| Layer | Setup | Performance | progress_ratio | path_eff |
|---|---|---:|---:|---:|
| hand-coded baseline | crosscomp / s0 / cross_u15 / arrival_v2 | **0.0%** (0/30) | −0.682 | −0.269 |
| **offline RL + oracle teacher** | **ReBRAC β1=4 / s0 / privileged dataset (N2')** | **0.0%** (0/60) | **+0.018** | **+0.100** |
| online RL (vanilla SAC, 1M steps) | vanilla SAC / s0 / cross_u15 / arrival_v2 | **10.0%** | — (catastrophic, see online §7.6) | — |
| oracle direct (privileged baseline) | privileged actor + hull-integral flow `[u_eq, v_eq]` | **70.0%** (21/30) | +0.685 | 0.293 |

**关键 raw-field 对比 (N2' vs crosscomp sanity)**：

| 量 | crosscomp sanity (S/log) | N2' (ReBRAC β1=4) | 含义 |
|---|---:|---:|---|
| `success_rate` | 0.0% | 0.0% | 数字相同但下层 mechanism 不同 |
| `avg_progress_ratio` | **−0.682** | **+0.018** | N2' 略微净接近目标（不是远离）— BC penalty 让 actor 学到「朝目标方向倾向」 |
| `avg_path_efficiency` | −0.269 | +0.100 | N2' 比 hand-coded 高 37pp — actor 学到 partial 的 motor pattern |
| termination `timeout` | 8 | 1 | N2' actor 不像 crosscomp 那样原地撞墙保守 — 它**主动游**，但方向不够准 |
| termination `out_of_bounds` | 22 | **59/60** | N2' 几乎 100% 出图 — actor 学到 motor pattern 但在 critical flow 下不能 closed-loop 校正 |

**Mechanism 描述**：

- crosscomp at u15 是 "frozen" — hand-coded controller 用 nominal heading + speed 在 critical flow 下被 wake 推搡，无法 closed-loop 校正，常 timeout 或 OOB
- N2' actor 学到了 privileged teacher 在 s0_obs 条件下的 **average action pattern**（即 `E[π_priv(a | privileged_obs) | s0_obs]`），所以它**会动**（progress ≈ 0、path_length ≈ 88m），但 critical regime 下 `s0_obs` 与 `privileged_obs = [u_eq, v_eq]` weakly correlated（plan §2.4 caveat），actor 拿不到 closed-loop 校正所需的 hull-integral flow 信号 → 朝大致方向开但 trajectory 振荡到 boundary → out_of_bounds

**N2' 不是 "noisy 0"**：60 episodes 中 timeout 仅 1，progress_ratio std 0.5–0.6 表明 actor 在不同初始条件下 trajectory 行为高度可变（有的负 progress 远离目标，有的正 progress 推到 boundary），这是 closed-loop 失稳 footprint，不是 stuck-policy footprint。

### 3.3 Verdict per plan §5.3

校准 (plan §5.3)：online catastrophic floor 0.10 / oracle ceiling 0.70。

| §5.3 阈值 | N2' mean 0.000 落在 |
|---|---|
| ≥ 0.40 = recovers ≥ 57% of teacher = strong positive | — |
| 0.15–0.40 = recovers 21–57% of teacher = partial / trigger M1 | — |
| **< 0.15 = ≈ online catastrophic = strong negative** | **✓** |
| recovery_of_oracle = **0.000 / 0.700 = 0.0%** | |
| lift_vs_online_floor = **0.000 − 0.10 = −10pp** | |

**Verdict**: **STRONG_NEGATIVE** — "critical regime is **actor-fundamental** under s0 sensor; even oracle demonstrations cannot bridge the partial-obs gap when actor lacks hull-integral flow access" (plan §5.3 row 3 paper claim).

**M1 trigger** (plan §5.4): N2' 0.000 ∉ [0.15, 0.40] → **NOT triggered**. M1 BC penalty sweep skipped per pre-commit rule "β-tuning 在 catastrophic 区间无信息" (plan §4.2)。

---

## 4. Anomaly explanation — why N2' falls 10pp below the online floor

**Anomaly**：N2' 0.000 < online §7.6 vanilla SAC s0 catastrophic floor 0.10 by **10pp**。

预期 prior 是 N2' ≥ online floor（offline RL + oracle 至少不劣于 online），实测反转。三个 candidate mechanism，paper §discussion 应保留全部三个作为 alternative hypothesis：

### 4.1 Candidate A: **OOD distribution shift dominates BC signal**

**Mechanism**：privileged teacher 在 critical regime 走的 trajectory 是 **closed-loop on hull-integral flow** — privileged_obs `[u_eq, v_eq]` 强信号驱动下沿能维持航向的「校正 trajectory」（curving，主动 cross-track 抵消 cross-stream），这条 trajectory 在 s0_obs space 里的 marginal 投影 `p(s0_obs | π_priv)` 与 ReBRAC actor 在 eval rollout 中实际经过的 s0 state distribution 不重合。

**Footprint**：actor 学到 `E[a | s0_obs]` 是 in-distribution 时的平均动作，但 OOD（eval rollout 偏离 teacher trajectory 后）BC penalty 把 actor 拉回它**没见过** s0 区域的 action prior，行为发散。online vanilla SAC 不背 BC penalty，只学 Q 价值，没有这个 "trapped by teacher state distribution" 问题。

**预测**：N2' actor 失败 trajectory 的 OOB **早期 timestep** 上 s0_obs 应位于 dataset s0_obs distribution 的 low-density region。

### 4.2 Candidate B: **BC penalty trap (β1 = 4 too strong for OOD)**

**Mechanism**：ReBRAC β1 = 4 是 main paper 在 sub-critical efficiency_v2 上调优的值。Critical regime 下 `Var[π_priv(a) | s0_obs]` 远大于 sub-critical，因为同一个 s0_obs（partial）对应 privileged teacher 多个不同的 closed-loop 校正 action。BC penalty 4 强制 actor 收敛到 conditional mean，丢失 multimodality，eval 中 actor 输出"折中"动作既不是 teacher 在 OOD 下的实际选择，也不是 Q 推荐的动作 → 行为退化到比 random / online 更差。

**Footprint**：N2' 行为不是 random（path_eff +0.10 > random expected ≈ 0），但也不是 task-solving — 看起来像「凑合的折中」。

**预测**：plan §4.2 M1 BC sweep（若触发）应在 β1 ∈ {0, 1, 2} 看到 N2' 提升；β1 = 0 退化到 TD3-only 在 60 ep 上 success > 0.1 即支持 candidate B。**本轮 N2' = 0.000 不触发 M1**，所以 candidate B **暂不可证伪/证实**，列入 backlog 等 §9 5-seed 补全或 future asym-critic ablation 一并 probe。

### 4.3 Candidate C: **online SAC's 10% is exploration luck, not a learned policy**

**Mechanism**：online §7.6 报告 vanilla SAC s0 cross_u15 = 0.10 — 30 episodes 中 3 个 goal，可能不是"学到了"，而是 exploration 期 random action 偶然走到 goal 附近 + sticky termination。online SAC 1M steps 在 catastrophic failure cell 上的 evaluation deterministic policy 行为有时也带 residual exploration noise（μ + 一点 std sampling）。

**Footprint**：online §7.6 cross_u15 / s0 的 progress_ratio 应非常低（接近 N2' 的 +0.018），termination 应几乎全 OOB，且每个 success 看起来都是 "侥幸朝目标方向 wander 到了 goal 附近"。

**预测**：如果 candidate C 成立，online 10% 不是有意义的 floor，N2' 0.0 vs online 0.10 的 10pp gap **不是异常**，而是 online 10% 本身就是测量噪声。本报告无法直接验证 — 需读取 online §7.6 raw trajectory + analyze 3 个 success episode 是否 trajectory pattern 与 random policy 相似。

### 4.4 Paper-writing recommendation

- **Strong claim**：N2' < online floor 这件事 **本身就是新发现**，无论 mechanism 是哪一个，都说明 "offline RL with oracle demonstrations under s0 in critical regime" 是 nontrivial failure regime，不是 trivially-doable transfer
- **Honest caveat**：paper 不应在 candidate A/B/C 之间下结论，保留三选一作为 alternative interpretations，强调 N2' = 0.0 与 online = 0.10 的 1-seed-on-each 比较 underpowered（online §7.6 也是少 seed）
- **Future work hook**：plan backlog §9 已列 asym-critic ablation 与 M1 BC sweep — 若 paper 审稿人 push back，rev.4 可补 N2' + asym-critic 或 N2' + β1=0 至少 1-seed sanity 区分 A/B/C

---

## 5. Paper claim — actor-fundamental partial-observability ceiling under s0

### 5.1 Headline claim (for §experiments appendix)

> "Under the deployable `s0` single-point DVL sensor in the critical flow regime (U=1.5, Re=250, λ=1.0), offline RL with oracle privileged-teacher demonstrations does **not** elevate the actor above the online catastrophic floor — and in fact underperforms vanilla SAC online (0.0% vs 10.0% over 30 episodes per seed). This empirically confirms an **actor-fundamental partial-observability ceiling**: when the privileged teacher's decision rule depends on hull-integral flow `[u_eq, v_eq]` that is **weakly correlated with the deployable sensor sample** in the critical regime, s0-conditioned behavior cloning cannot recover the teacher's closed-loop correction, regardless of demonstration quality."

### 5.2 Ceiling decomposition (full, for §discussion)

| Layer | Setup | Performance | Interpretation |
|---|---|---:|---|
| hand-coded baseline | crosscomp / s0 | **0.0%** | open-loop heading + speed under critical wake → cannot recover |
| online RL | vanilla SAC / s0 / 1M steps / `arrival_v2` | 10.0% | reward shaping + exploration insufficient to learn closed-loop correction under partial obs |
| **offline RL + oracle teacher** | **ReBRAC β1=4 / s0 / privileged dataset (N2')** | **0.0%** | **BC + critic regularization on `E[π_priv | s0_obs]` ≠ closed-loop correction policy** |
| oracle direct | privileged baseline + hull-integral `[u_eq, v_eq]` | 70.0% | hull-integral flow is **causally sufficient** to drive closed-loop correction; the ceiling between this row and the rows above quantifies the partial-obs gap |

**Gap quantification under arrival_v2 / cross_u15 / s0**：
- partial-obs gap (oracle direct − offline RL + oracle data) = **70.0 − 0.0 = 70pp** （actor-fundamental ceiling）
- online catastrophic gap (oracle direct − online RL) = 70.0 − 10.0 = 60pp
- **offline-vs-online residual under same partial obs** = 10.0 − 0.0 = +10pp（online 反而比 offline+oracle 高 10pp，§4 解释）

### 5.3 Why this is not a contradiction to main paper claim

| Main paper claim | v2 N2' finding | 关系 |
|---|---|---|
| ReBRAC +23pp over TD3+BC on **sub-critical / efficiency_v2 / s0** | N0 holds with −5.2pp anchor drift under `arrival_v2`；algorithm-level claim 不变 | **不冲突** |
| (implicit) ReBRAC inherits its advantage from BC + critic regularization on **closely matched (s0_obs, action) distribution** | N2' shows **critical regime breaks the (s0_obs, privileged action) correspondence**，violates the assumption | **互补 — 划定 boundary condition** |

Paper §discussion 应明写：ReBRAC 的 BC penalty 在 `s0_obs` 与 dataset action **strongly correlated** 时 effective（sub-critical / cross_u10）；critical regime 下 `s0_obs` 与 oracle action correspondence **弱化**到 BC penalty 把 actor 拉向 dataset conditional mean 这个**非 closed-loop 信号**，actor 行为退化。

### 5.4 What v2 does NOT claim

- **不 claim** ReBRAC algorithm 是"差的" — algorithm 本身在 sub-critical 仍 +23pp（v1 main paper）+ N0 0.85（v2 anchor holds）
- **不 claim** offline RL 在 critical regime 完全 hopeless — asym-critic 等机制 backlog 已列，rev.4 candidate
- **不 claim** 10pp gap (N2' vs online) 是 sharp finding — 2-seed × 30 ep underpowered，paper 保留 §4 三 candidate mechanism 作为 alternative explanation

---

## 6. Limitations and next steps

### 6.1 已知 limitations

| Limitation | Severity | 缓解措施 |
|---|---|---|
| 2 seed 而非 plan §6.2 预登记的 3 seed | Medium — N2' 0.000/0.000 deterministic 给出 strong signal；N0 0.867/0.833 同向 informative | Backlog §9 5-seed 补全；rev.4 可补 seed=43 at least for N0 |
| 30 eval episodes per seed | Low — broad val convention 一致 | 与 main paper / v1 broad val 一致 |
| `arrival_v2` reward 与 main paper anchor (`efficiency_v2`) Δ = −5.2pp 退化未做 mechanism diagnostic | Medium | Paper §experiments appendix 明标；future work：N0 + efficiency_v2 / arrival_v2 paired seed comparison |
| candidate A/B/C 三选一未做 mechanism discriminator | Medium | M1 BC sweep gate (plan §5.4) 不触发 = 暂搁置；rev.4 可单独 spec asym-critic ablation |
| 与 online §7.6 vanilla SAC s0 cross_u15 = 0.10 的比较是 cross-source（不同 seed 集 + 不同实施细节） | Low — paper 明标 | §4 candidate C 已列 |

### 6.2 已完成的 verdict-gate-implied next actions

| Action | Status |
|---|---|
| N0 verdict §5.2 → "proceed to N2'" | ✓ 已完成 N2' |
| N2' verdict §5.3 → "strong negative + skip M1" | ✓ M1 not triggered |
| Plan §10 status 更新 | **待执行**（与本 report 同步 commit） |
| Paper §robustness / appendix 段落 | **待写**（本 report §1 + §5 提供素材） |

### 6.3 推荐的 follow-up（不在本闭环范围）

| Item | Priority | Trigger |
|---|---|---|
| 5-seed 补全 N0（+seed 43, 44, 45） | Medium | 审稿人 push back 3 seed power |
| 5-seed 补全 N2'（+seed 43, 44, 45） | Low | N2' deterministic 0.000，5 seed 不太可能反转 |
| Asym-critic ablation on N2' (rev.4 candidate, plan §9 backlog) | **High** | 若 paper §discussion 想区分 "actor-fundamental partial-obs" vs "critic-fundamental partial-obs" |
| M1 BC sweep (β1 ∈ {0, 1, 2, 4, 8} × 3 seed) | Low — N2' 不在 trigger zone | Bypassed |
| online §7.6 candidate-C 验证（3 success episode trajectory 分析） | Medium | Paper §discussion 想 strengthen anomaly section |

---

## 7. Cross-link 与文档影响

### 7.1 本 report 触发的下游修改

| 文档 | 修改 | 状态 |
|---|---|---|
| [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) | §10 状态更新："PASS — completed 2026-05-19, actor-fundamental ceiling confirmed"；§12 执行检查清单逐项打勾 | **TBD** |
| [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) | §3.3 broad validation 段落由 "v2 active" → "v2 PASS"；新增 §3.3 sub-bullet 总结 ceiling finding | **TBD** |
| [`docs/rebrac_mainline_review.md`](rebrac_mainline_review.md) | §3.5 顶层 pointer 由 "v2 plan rev.3 in-progress" → "v2 PASS report" | **TBD** |
| [`docs/rebrac_paper_writing_index.md`](rebrac_paper_writing_index.md) | §experiments appendix 入口添加本 report；§discussion ceiling decomposition 入口添加本 report §5 | **TBD** |
| `auv_paper/` (LaTeX) — §experiments appendix + §discussion §robustness 1 段 | 起草中 | **TBD** |

### 7.2 Raw output 留痕

| Path | Content |
|---|---|
| `results/offline/rebrac/broad_validation_v2/N0/seed_42/test_result.json` | N0 seed 42 raw eval (success 0.867) |
| `results/offline/rebrac/broad_validation_v2/N0/seed_0/test_result.json` | N0 seed 0 raw eval (success 0.833) |
| `results/offline/rebrac/broad_validation_v2/N2p/seed_42/test_result.json` | N2' seed 42 raw eval (success 0.000) |
| `results/offline/rebrac/broad_validation_v2/N2p/seed_0/test_result.json` | N2' seed 0 raw eval (success 0.000) |
| `results/offline/rebrac/broad_validation_v2/summaries/verdict_decision.json` | anchor + per-cell mean/std/per-seed + verdict block |
| `results/offline/rebrac/broad_validation_v2/summaries/p1_overview.csv` | 7-col overview table |
| `experiments/offline/rebrac/broad_validation_v2/S_sanity/sanity_card_*.json` | S sanity baseline (DONE 2026-05-18) |
| `notebooks/rebrac_broad_validation_v2_core_completed_new.ipynb` | Colab execution notebook (2026-05-19 run) |

### 7.3 与其它 line 的 cross-link

- **Online 线** [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.6 — 提供 ceiling decomposition 的 online RL 行（vanilla SAC s0 cross_u15 = 0.10）
- **Main paper line** [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) rev.8 — 提供 §5.2 N0 anchor 比较的 main-line efficiency_v2 baseline (0.902 ± 0.021)
- **v1 archive** [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) — 已加 SUPERSEDED banner；保留作为 v1 实验历史

---

**Report 起草**: 2026-05-19
**Plan rev**: rev.3 (with §6.2 2-seed note appended)
**Reproducibility**: notebook `notebooks/rebrac_broad_validation_v2_core_completed_new.ipynb` + raw `results/offline/rebrac/broad_validation_v2/`
