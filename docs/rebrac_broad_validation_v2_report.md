# ReBRAC Broad Validation v2 — Experimental Report

> **Status (2026-05-19)**: COMPLETED — N0 + N2' 4-run 闭环（2 seed [42, 0]）。§5.2 verdict `HOLDS`，§5.3 verdict `STRONG_NEGATIVE`，§5.4 M1 BC sweep **not triggered**。
>
> **Addendum (2026-05-27)**: N2' **asym-critic ablation 完成**（2 seed [42, 0]，唯一变量 `--use-asymmetric-critic`，逐 episode 配对 vanilla N2'）。Verdict **ACTOR_FUNDAMENTAL_CONFIRMED** — 把完美 hull-integral flow `[u_eq,v_eq]` 喂给 critic 算 TD target，s0 actor success 仍 **0.000 / 0.000**（recovery 0% of oracle 0.70），部署行为与 vanilla 近乎不可区分。详见 **§4.5**。Raw: `results/offline/rebrac/broad_validation_v2_n2p_asym/{seed_42,seed_0}/test_result.json` + `summaries/asym_verdict.json`；notebook `notebooks/rebrac_broad_validation_v2_n2p_asym_critic_completed.ipynb`。
>
> **Addendum (2026-07-12)**: **+seed 43 supplement 完成**（N0 / N2' / asym 三单元全补，同协议 30-ep 终检；plan+呈报+裁决全档 [`rebrac_broad_validation_v2_seed43_supplement_plan.md`](rebrac_broad_validation_v2_seed43_supplement_plan.md)）。N2'/asym seed 43 均 **0.000**（各三种子合计 **0/90**，rule-of-three 上界 ≈ 0.033）——零成功结论不变、只更强；N0 seed 43 = **0.933**，3-seed anchor **0.878 ± 0.051**（Δ vs 0.902 = −2.42pp），**per-seed 退化方向不再一致**（−3.5 / −6.9 / **+3.1**pp）——首轮「两种子同向退化非噪声」的读法已撤销，§2.4 已按三种子改写。种子组 {42, 0, 43} 与预登记 {42, 43, 44} 不完全重合（首轮以 0 替换 44）。Raw: `results/offline/rebrac/broad_validation_v2/{N0,N2p}/seed_43/test_result.json`、`broad_validation_v2_n2p_asym/seed_43/test_result.json` + `summaries/seed43_supplement_verdict.json`。
>
> **Branch**: `codex-arrival-v2-prototype`
> **Plan**: [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) rev.3
> **Raw outputs**: `results/offline/rebrac/broad_validation_v2/{N0,N2p}/seed_{42,0,43}/test_result.json` + `summaries/verdict_decision.json` + `summaries/seed43_supplement_verdict.json` + `summaries/p1_overview.csv`
> **Cross-link**: [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.6（online catastrophic floor），[`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md)（main paper efficiency_v2 anchor）
>
> **Seed count**: 3 seed {42, 0, 43}（首轮 2 seed 缩水，2026-07-12 supplement 补齐第三种子；与预登记组 {42, 43, 44} 不完全重合——首轮以 0 替换 44，详见 §6 limitations）；5-seed 补全条目继续保留 backlog（§6.3）。

---

## 目录

- [§1 Abstract](#1-abstract)
- [§2 N0 reward bridge — anchor drift analysis（3-seed 更新 2026-07-12）](#2-n0-reward-bridge--anchor-drift-analysis3-seed-更新-2026-07-12)
- [§3 N2' failure mechanism — raw-field decomposition](#3-n2-failure-mechanism--raw-field-decomposition)
- [§4 Anomaly explanation — why N2' falls 10pp below the online floor](#4-anomaly-explanation--why-n2-falls-10pp-below-the-online-floor)
- [§5 Paper claim — actor-fundamental partial-observability ceiling under s0](#5-paper-claim--actor-fundamental-partial-observability-ceiling-under-s0)
- [§6 Limitations and next steps](#6-limitations-and-next-steps)
- [§7 Cross-link 与文档影响](#7-cross-link-与文档影响)

---

## 1. Abstract

ReBRAC broad validation v2 在 `arrival_v2` reward 下完成 2 个 core cell（N0 sub-critical anchor + N2' critical-regime oracle-teacher probe），首轮共 4 run（2 seeds × 2 cells，~30 min L4），2026-07-12 supplement 补齐第三种子（+seed 43 × 3 单元，含 §4.5 asym ablation）。所有 verdict gates 均在跑实验前预登记于 plan §5.2 / §5.3（supplement 判读门槛预登记于 supplement plan §2）。

**核心结果（3-seed，2026-07-12 更新）**：

| Cell | Setup | success (3-seed) | per-seed [42, 0, 43] | verdict gate (§5) | Verdict |
|---|---|---:|---|---|---|
| **N0** | crosscomp / s0 / cross_u10 / Re150 / arrival_v2 | **0.878 ± 0.051** | [0.867, 0.833, 0.933] | §5.2: ≥ 0.70 = "reward bridge holds" | **HOLDS** |
| **N2'** | privileged / s0 / cross_u15 / Re250 / arrival_v2 | **0.000 ± 0.000** | [0.000, 0.000, 0.000] | §5.3: < 0.15 + recovery < 21% of oracle = strong negative | **STRONG_NEGATIVE** |

**M1 BC penalty sweep**：N2' = 0.0 不在 §5.4 trigger zone `[0.15, 0.40]`，**not triggered**。

**三句话结论**：

1. **reward bridge HOLDS；退化幅度收窄且方向存疑（3-seed 改写 2026-07-12）**：N0 在新 reward 下 3-seed anchor 0.878 ± 0.051，Δ vs efficiency_v2 main-line anchor (0.902) = **−2.42pp**；per-seed 方向不一致（42: −3.5pp / 0: −6.9pp / 43: **+3.1pp**，第三种子高于 anchor）——首轮两种子的同向退化在三种子下不再成立，该差异应按种子间波动解读（§2.4）。Paper 写作时仍须明标该跨口径对照只作量级参考。
2. **N2' STRONG_NEGATIVE confirms actor-fundamental partial-obs ceiling**：90 episode 全部失败（88 out_of_bounds + 1 timeout + 1 depth_hold_failure），progress_ratio ≈ 0，path_eff ≈ 0.09；零成功彻底跌破 online §7.6 catastrophic floor 10%。即便 oracle teacher (70% direct success) 提供 demonstrations，**s0-conditioned BC 在 critical regime 不能传递 hull-integral flow 知识**。
3. **paper-quality claim** (§8.2 ceiling decomposition)：在 critical regime + deployable s0 actor 下，offline RL with oracle demonstrations 不仅 fails to bridge partial-obs gap，且 underperforms 同 setup 下的 online SAC — 这是 plan rev.3 §2.4 "actor-fundamental partial-obs ceiling" caveat 的直接实证。

**Addendum 结论 (2026-05-27，§4.5)**：N2' asym-critic ablation 把 candidate B 的 critic 侧解释**排除**——即便 critic 在训练全程拿到完美 hull-integral flow，s0 actor success 仍 0.000，部署行为与 vanilla 配对近乎相同（paired median Δprogress ≈ 0）。天花板**不在 critic 价值估计**，与 actor-fundamental 一致并 **HARDENS** §5 主 claim。

---

## 2. N0 reward bridge — anchor drift analysis（3-seed 更新 2026-07-12）

**Setup**: crosscomp / s0 / cross_u10 / Re150 / arrival_v2 / 30 eval episodes per seed / manifest `single_u10_cross_tgt15`。

### 2.1 Per-seed numbers

| Seed | success | terminations | progress_ratio | path_efficiency | return | safety_cost |
|---:|---:|---|---:|---:|---:|---:|
| 42 | 0.867 (26/30) | goal:26 / OOB:3 / timeout:1 | 0.825 ± 0.234 | 0.628 ± 0.192 | 71.3 ± 121.3 | 15.3 ± 17.1 |
| 0 | 0.833 (25/30) | goal:25 / OOB:5 / timeout:0 | 0.824 ± 0.192 | 0.605 ± 0.161 | 55.9 ± 123.8 | 16.1 ± 14.6 |
| 43 (supplement 2026-07-12) | 0.933 (28/30) | goal:28 / OOB:2 / timeout:0 | 0.869 ± 0.148 | 0.646 ± 0.139 | 89.2 ± 87.2 | 15.9 ± 10.8 |
| **mean (3-seed)** | **0.878 ± 0.051** | dominated by `goal` (success) + `OOB` (failure) | 0.839 | 0.626 | 72.1 | 15.8 |

### 2.2 Anchor comparison

| Reference | Anchor | Source |
|---|---:|---|
| main paper anchor (efficiency_v2 / cross_u10 / s0 / 5-seed) | **0.902 ± 0.021** | [`rebrac_experiment_report.md`](rebrac_experiment_report.md) rev.8 |
| v2 N0 (arrival_v2 / cross_u10 / s0 / 3-seed) | **0.878 ± 0.051** | this report |
| **Δ** | **−2.42pp** | per-seed direction NOT consistent（§2.4） |

### 2.3 Verdict per plan §5.2

| §5.2 阈值 | N0 mean 0.878 落在 |
|---|---|
| ≥ 0.70 = reward bridge holds | ✓ |
| 0.50–0.70 = weak bridge | — |
| < 0.50 = bridge fail (pause) | — |

**Verdict**: **HOLDS** — proceed to N2'.

### 2.4 Anchor drift re-read under 3 seeds — 同向性不再成立（rewritten 2026-07-12）

> 本节原题为 "Why the 5.2pp drop is not noise"，其论证依赖首轮两种子的同向退化；+seed 43 supplement 后该前提失效，按用户裁决（supplement plan 附录 B.2 (a)）如实改写。原两种子判读保留在下方作历史记录。

**3-seed 判读（现行）**：per-seed Δ vs 5-seed efficiency_v2 anchor (0.902)：seed 42 → 86.7%（−3.5pp）、seed 0 → 83.3%（−6.9pp）、seed 43 → **93.3%（+3.1pp，高于 anchor）**。三种子方向不一致，均值差 −2.42pp 小于种子间标准差（0.051）——**「系统性退化」的读法不再被支持，anchor drift 应按种子间波动解读**。termination 侧同样弱化：seed 43 为 28 goal / 2 OOB / 0 timeout，比首轮两种子更接近 main-line 形态（三种子均值 26.3 goal / 3.3 OOB / 0.3 timeout，vs v1 main-line 等价 cell 27.4 goal / 2.6 OOB / 0 timeout），「arrival_v2 使 actor 更激进、越界略增」的机制暗示同步降级为未定。

**首轮两种子判读（已撤销，仅存档）**：两 seed 同向退化 + 同向 binomial ≈ 0.25 + termination 漂移，曾被读为「非 noise 的 reward-specific calibration effect」。该读法在第三种子反向后撤销。

**与 main paper claim 不冲突（更新）**：main paper claim "ReBRAC +23pp on crosscomp / cross_u10 / s0 / efficiency_v2" 在 efficiency_v2 anchor 下成立；arrival_v2 下 3-seed anchor 0.878 ± 0.051 与 0.902 的差异方向存疑、幅度 −2.4pp，不影响 algorithm-level finding 的有效性。Paper §experiments appendix 明标语句相应更新为 "the +23pp advantage is reported under efficiency_v2; under arrival_v2 the 3-seed absolute success is 0.878 ± 0.051 (−2.4pp vs the efficiency_v2 anchor, per-seed direction not consistent), still substantially exceeding the catastrophic baseline (0.0%)".

---

## 3. N2' failure mechanism — raw-field decomposition

**Setup**: privileged / s0 / cross_u15 / Re250 / arrival_v2 / 30 eval episodes per seed / manifest `single_u15_cross_tgt15`。

### 3.1 Per-seed numbers

| Seed | success | terminations | progress_ratio | path_efficiency | path_length_m | return | safety_cost |
|---:|---:|---|---:|---:|---:|---:|---:|
| 42 | 0.000 (0/30) | OOB:29 / timeout:1 | 0.041 ± 0.509 | 0.107 ± 0.248 | 88.6 ± 55.9 | −247.5 ± 41.2 | 18.4 ± 13.8 |
| 0 | 0.000 (0/30) | **OOB:30** | −0.005 ± 0.618 | 0.093 ± 0.235 | 88.2 ± 55.3 | −248.8 ± 46.4 | 15.5 ± 10.1 |
| 43 (supplement 2026-07-12) | 0.000 (0/30) | OOB:29 / depth_hold_failure:1 | −0.021 ± 0.453 | 0.062 ± 0.222 | 94.0 ± 55.0 | −258.5 ± 46.5 | 20.3 ± 14.6 |
| **mean (3-seed)** | **0.000 ± 0.000** | **OOB dominates 88/90** | 0.005 | 0.087 | 90.3 | −251.6 | 18.1 |

### 3.2 Ceiling decomposition (post-N2', plan §8.2 table fully populated)

| Layer | Setup | Performance | progress_ratio | path_eff |
|---|---|---:|---:|---:|
| hand-coded baseline | crosscomp / s0 / cross_u15 / arrival_v2 | **0.0%** (0/30) | −0.682 | −0.269 |
| **offline RL + oracle teacher** | **ReBRAC β1=4 / s0 / privileged dataset (N2')** | **0.0%** (0/90) | **+0.005** | **+0.087** |
| online RL (vanilla SAC, 1M steps) | vanilla SAC / s0 / cross_u15 / arrival_v2 | **10.0%** | — (catastrophic, see online §7.6) | — |
| oracle direct (privileged baseline) | privileged actor + hull-integral flow `[u_eq, v_eq]` | **70.0%** (21/30) | +0.685 | 0.293 |

**关键 raw-field 对比 (N2' vs crosscomp sanity)**：

| 量 | crosscomp sanity (S/log) | N2' (ReBRAC β1=4，3-seed) | 含义 |
|---|---:|---:|---|
| `success_rate` | 0.0% | 0.0% | 数字相同但下层 mechanism 不同 |
| `avg_progress_ratio` | **−0.682** | **+0.005** | N2' 净距变化近零（不是被推离）— BC penalty 让 actor 学到「朝目标方向倾向」 |
| `avg_path_efficiency` | −0.269 | +0.087 | N2' 比 hand-coded 高约 36pp — actor 学到 partial 的 motor pattern |
| termination `timeout` | 8 | 1（另 depth_hold_failure 1） | N2' actor 不像 crosscomp 那样原地撞墙保守 — 它**主动游**，但方向不够准 |
| termination `out_of_bounds` | 22 | **88/90** | N2' 几乎 100% 出图 — actor 学到 motor pattern 但在 critical flow 下不能 closed-loop 校正 |

**Mechanism 描述**：

- crosscomp at u15 是 "frozen" — hand-coded controller 用 nominal heading + speed 在 critical flow 下被 wake 推搡，无法 closed-loop 校正，常 timeout 或 OOB
- N2' actor 学到了 privileged teacher 在 s0_obs 条件下的 **average action pattern**（即 `E[π_priv(a | privileged_obs) | s0_obs]`），所以它**会动**（progress ≈ 0、path_length ≈ 88m），但 critical regime 下 `s0_obs` 与 `privileged_obs = [u_eq, v_eq]` weakly correlated（plan §2.4 caveat），actor 拿不到 closed-loop 校正所需的 hull-integral flow 信号 → 朝大致方向开但 trajectory 振荡到 boundary → out_of_bounds

**N2' 不是 "noisy 0"**：90 episodes 中 timeout 仅 1、depth_hold_failure 仅 1，progress_ratio std 0.45–0.6 表明 actor 在不同初始条件下 trajectory 行为高度可变（有的负 progress 远离目标，有的正 progress 推到 boundary），这是 closed-loop 失稳 footprint，不是 stuck-policy footprint。

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

**预测**：plan §4.2 M1 BC sweep（若触发）应在 β1 ∈ {0, 1, 2} 看到 N2' 提升；β1 = 0 退化到 TD3-only 在 60 ep 上 success > 0.1 即支持 candidate B。**本轮 N2' = 0.000 不触发 M1**，所以 candidate B 的 **β1-tuning 轴**仍暂搁置。

**Update (2026-05-27，§4.5 asym ablation)**：candidate B 有两种可能机制——(i) **critic 侧**：critic 拿到的价值信号不足 → actor 学不到；(ii) **actor 侧**：BC penalty 把 actor 拉向 `E[π_priv | s0_obs]` conditional mean 这个非-closed-loop 信号。asym-critic ablation（§4.5）在 β1=4 不变下把完美 hull-integral flow 喂给 critic，success 仍 0.000 → **排除 (i)**：bottleneck 不是 critic 价值估计质量。若 candidate B 成立，必经 (ii) 的 actor 侧 conditional-mean 塌缩，而非 critic 侧——这与 §5 actor-fundamental 框架一致。β1 sweep 仍 bypassed（gate 不触发）。

### 4.3 Candidate C: **online SAC's 10% is exploration luck, not a learned policy**

**Mechanism**：online §7.6 报告 vanilla SAC s0 cross_u15 = 0.10 — 30 episodes 中 3 个 goal，可能不是"学到了"，而是 exploration 期 random action 偶然走到 goal 附近 + sticky termination。online SAC 1M steps 在 catastrophic failure cell 上的 evaluation deterministic policy 行为有时也带 residual exploration noise（μ + 一点 std sampling）。

**Footprint**：online §7.6 cross_u15 / s0 的 progress_ratio 应非常低（接近 N2' 的 +0.018），termination 应几乎全 OOB，且每个 success 看起来都是 "侥幸朝目标方向 wander 到了 goal 附近"。

**预测**：如果 candidate C 成立，online 10% 不是有意义的 floor，N2' 0.0 vs online 0.10 的 10pp gap **不是异常**，而是 online 10% 本身就是测量噪声。本报告无法直接验证 — 需读取 online §7.6 raw trajectory + analyze 3 个 success episode 是否 trajectory pattern 与 random policy 相似。

### 4.4 Paper-writing recommendation

- **Strong claim**：N2' < online floor 这件事 **本身就是新发现**，无论 mechanism 是哪一个，都说明 "offline RL with oracle demonstrations under s0 in critical regime" 是 nontrivial failure regime，不是 trivially-doable transfer
- **Honest caveat**：paper 不应在 candidate A/B/C 之间下结论，保留三选一作为 alternative interpretations，强调 N2' = 0.0 与 online = 0.10 的 1-seed-on-each 比较 underpowered（online §7.6 也是少 seed）
- **Future work hook**：plan backlog §9 已列 asym-critic ablation 与 M1 BC sweep — 若 paper 审稿人 push back，rev.4 可补 N2' + asym-critic 或 N2' + β1=0 至少 1-seed sanity 区分 A/B/C

### 4.5 Asym-critic ablation — actor-fundamental vs critic-fundamental 机制判别 (completed 2026-05-27)

vanilla N2' 中 actor 和 critic **都只看 s0**，0% 失败有两个无法区分的解释：(a) **actor-fundamental** — s0 actor 物理上还原不了 privileged 决策规则；(b) **critic-fundamental** — s0 critic 价值估计塌掉 → TD target 烂 → 训练失败。本 ablation 在**唯一变量** `--use-asymmetric-critic` 下隔离两者：critic 训练全程拿 hull-integral flow `privileged_obs=[u_eq,v_eq]` (dim=2) 算 TD target / critic loss，actor 改进时 priv 通道 zero-pad（`--privileged-actor-update-mode zeros`，mimic deployment，CLAUDE.md §3 默认）。dataset / manifest / β1=4 / β2=2 / seeds [42, 0]（+seed 43，2026-07-12 supplement 双侧同补）与 vanilla N2' **逐项相同**，eval 命令逐字一致（同 `--seed 123`，同 manifest，actor eval 仍只 s0）→ **逐 episode 配对**。

**主结果（配对，3 seed × 30 ep；seed 43 为 2026-07-12 supplement）：**

| | vanilla N2' (s0 critic) | **asym N2' (priv critic)** | Δ |
|---|---:|---:|---:|
| success seed 42 / seed 0 / seed 43 | 0.000 / 0.000 / 0.000 | **0.000 / 0.000 / 0.000** | +0.00pp |
| mean success | 0.000 | **0.000 ± 0.000** | +0.00pp |
| recovery of oracle 0.70 | — | **0.0%** | — |
| termination (asym) | — | seed42 OOB 30；seed0 OOB 29 + timeout 1；seed43 OOB 30 | — |

合计 **0 / 90 episodes 成功** → rule-of-three 95% 上界 ≈ 3/90 ≈ **0.033**，远低于 online floor 0.10、oracle 0.70。即便给 critic 完美 hull-integral flow，**也不能把 s0 actor 抬过 catastrophic floor**。

**配对 episode-level 诊断（关键：排除"asym 改善了行为"的假象）。** 表面上 mean progress_ratio 从 vanilla 0.041→asym 0.143 (seed 42) 似有提升，但逐 episode 配对后：

| 量 (seed 42, n=30) | 值 | 含义 |
|---|---:|---|
| paired **mean** Δprogress (asym−van) | +0.1025 | 仅 **1.15 SEM**，不显著 |
| paired **median** Δprogress | **−0.016** | 中位数甚至略负 → 无系统性改善 |
| 单 episode (ep_0006) 贡献 | **46%** of mean Δ | vanilla 该 ep 跑飞 timeout (pr −1.849)，asym 同 ep 普通 OOB (pr −0.421)；去掉它 mean Δ 掉到 +0.057 |
| \|Δprogress\| < 0.10 的 episodes | **17 / 30** | 过半 episode 配对近乎相同 |

termination 差异跨 seed **方向相反**（seed 42：vanilla 1 timeout→asym 全 OOB；seed 0：vanilla 全 OOB→asym 1 timeout），进一步证明非系统性。**结论：asym 与 vanilla 部署行为在统计上不可区分**，mean-progress 的微小正偏是 per-episode 重尾噪声 + 单个 vanilla-timeout 离群点的假象。

**Verdict（事先 commit 的 gate，§notebook §0）：mean success 0.000 ≤ 0.10 → `ACTOR_FUNDAMENTAL_CONFIRMED`。**

**诚实的因果范围（写 paper 必须遵守）：**
- ✅ **可下的强 claim**：privileged-flow critic **不能挽救** N2' 天花板；**排除了"纯 critic 价值估计失败"(critic-fundamental) 的解释**。这是项目自身 asymmetric-critic 方法（CLAUDE.md §3）在 offline critical-regime 下的直接负结果。
- ⚠ **不能下的 claim**：本实验**不证明** s0 actor 信息论上不可能学到该策略。asym=0 同时兼容 "actor 表征上不可能" 与 "asym-critic 这一机制（含 zeros actor-update）在 offline 下没把 privileged 信息有效转化给 actor" 两种读法。措辞用 **"not rescued by a privileged critic / consistent with & HARDENS actor-fundamental"**，**不写** "proven actor-incapable"。
- 统计上首轮 2 seed 已**充分**：双双 degenerate-0（强信号非 no-power），落在 ≤0.10 gate 区，**不触发**预登记的"补 seed 43 + stratified bootstrap"（该条件只在落入 MIXED [0.10,0.40] / REFRAME [≥0.40] 时触发）。**2026-07-12 update**：论文定稿前补充验证仍按同协议补跑了 seed 43（vanilla 与 asym 双侧），结果仍 degenerate-0（0/30 / 0/30，asym 全越界）——三种子合计 0/90，结论不变、只更强。配对 episode-level 诊断（上文均值/中位数/离群点分析）基于首轮两种子，未对 seed 43 重做；其终检结果与两种子完全一致，不改变诊断结论。

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
| **+ privileged critic (asym ablation, §4.5)** | **ReBRAC β1=4 / s0 actor / asym critic sees `[u_eq,v_eq]`** | **0.0%** | **给 critic 完美 hull-integral flow 仍不能抬升 s0 actor → 天花板在 actor 侧，非 critic 价值估计；排除 critic-fundamental** |
| oracle direct | privileged baseline + hull-integral `[u_eq, v_eq]` | 70.0% | hull-integral flow is **causally sufficient** to drive closed-loop correction; the ceiling between this row and the rows above quantifies the partial-obs gap |

**Gap quantification under arrival_v2 / cross_u15 / s0**：
- partial-obs gap (oracle direct − offline RL + oracle data) = **70.0 − 0.0 = 70pp** （actor-fundamental ceiling）
- online catastrophic gap (oracle direct − online RL) = 70.0 − 10.0 = 60pp
- **offline-vs-online residual under same partial obs** = 10.0 − 0.0 = +10pp（online 反而比 offline+oracle 高 10pp，§4 解释）

### 5.3 Why this is not a contradiction to main paper claim

| Main paper claim | v2 N2' finding | 关系 |
|---|---|---|
| ReBRAC +23pp over TD3+BC on **sub-critical / efficiency_v2 / s0** | N0 holds with −2.4pp anchor drift under `arrival_v2`（3-seed，per-seed 方向不一，§2.4）；algorithm-level claim 不变 | **不冲突** |
| (implicit) ReBRAC inherits its advantage from BC + critic regularization on **closely matched (s0_obs, action) distribution** | N2' shows **critical regime breaks the (s0_obs, privileged action) correspondence**，violates the assumption | **互补 — 划定 boundary condition** |

Paper §discussion 应明写：ReBRAC 的 BC penalty 在 `s0_obs` 与 dataset action **strongly correlated** 时 effective（sub-critical / cross_u10）；critical regime 下 `s0_obs` 与 oracle action correspondence **弱化**到 BC penalty 把 actor 拉向 dataset conditional mean 这个**非 closed-loop 信号**，actor 行为退化。

### 5.4 What v2 does NOT claim

- **不 claim** ReBRAC algorithm 是"差的" — algorithm 本身在 sub-critical 仍 +23pp（v1 main paper）+ N0 0.85（v2 anchor holds）
- **不 claim** offline RL 在 critical regime 完全 hopeless — 但 asym-critic ablation（§4.5，2026-05-27 完成）已证明**本项目自身的 privileged-critic 机制不能 bridge**；可探索方向退到 recurrent/history actor、显式 flow-estimation head 等 actor 侧改造，不在本闭环
- **不 claim** asym=0 证明 s0 actor 信息论上不可能（§4.5 因果范围）；只 claim 排除 critic-fundamental + HARDENS actor-fundamental
- **不 claim** 10pp gap (N2' vs online) 是 sharp finding — 3-seed × 30 ep 仍 underpowered（online 侧亦少 seed），paper 保留 §4 三 candidate mechanism 作为 alternative explanation

---

## 6. Limitations and next steps

### 6.1 已知 limitations

| Limitation | Severity | 缓解措施 |
|---|---|---|
| ~~2 seed 而非 plan §6.2 预登记的 3 seed~~ → **3 seed 已补齐（2026-07-12 supplement，+seed 43 × 三单元）**；种子组 {42, 0, 43} 与预登记 {42, 43, 44} 不完全重合（首轮以 0 替换 44） | Low（降级）— N2'/asym 三种子 deterministic 0.000；N0 第三种子方向反转（+3.1pp），见 §2.4 改写 | 5-seed 补全条目保留 backlog（§6.3）；supplement plan 附录 B 存判读与裁决全档 |
| 30 eval episodes per seed | Low — broad val convention 一致 | 与 main paper / v1 broad val 一致 |
| `arrival_v2` reward 与 main paper anchor (`efficiency_v2`) 的 Δ（3-seed 后收窄为 −2.4pp 且 per-seed 方向不一，§2.4）未做 mechanism diagnostic | Low（降级——drift 本身已不支持系统性读法） | Paper §experiments appendix 明标；future work：N0 + efficiency_v2 / arrival_v2 paired seed comparison |
| ~~candidate A/B/C 三选一未做 mechanism discriminator~~ → **actor-vs-critic 轴已判别** | Low（升级） | **asym-critic ablation 已完成（§4.5，2026-05-27）= ACTOR_FUNDAMENTAL_CONFIRMED**，排除 critic-fundamental；candidate A（OOD shift）/ B 的 β1 轴仍 open 但优先级低 |
| 与 online §7.6 vanilla SAC s0 cross_u15 = 0.10 的比较是 cross-source（不同 seed 集 + 不同实施细节） | Low — paper 明标 | §4 candidate C 已列 |

### 6.2 已完成的 verdict-gate-implied next actions

| Action | Status |
|---|---|
| N0 verdict §5.2 → "proceed to N2'" | ✓ 已完成 N2' |
| N2' verdict §5.3 → "strong negative + skip M1" | ✓ M1 not triggered |
| **Asym-critic ablation on N2'（机制判别）** | ✓ **完成 2026-05-27（§4.5）= ACTOR_FUNDAMENTAL_CONFIRMED** |
| Plan §10 status 更新 | **待执行**（与本 report 同步 commit） |
| Paper §robustness / appendix 段落 | **待写**（本 report §1 + §4.5 + §5 提供素材；落 paper 待 path B venue 决策后） |

### 6.3 推荐的 follow-up（不在本闭环范围）

| Item | Priority | Trigger |
|---|---|---|
| 5-seed 补全 N0（+seed 44, 45；~~43~~ ✓ 已补 2026-07-12） | Low（3-seed 已达预登记数；第三种子方向反转后均值差已收窄） | 审稿人 push back seed power |
| 5-seed 补全 N2'（+seed 44, 45；~~43~~ ✓ 已补 2026-07-12，仍 0.000） | Low | N2' 三种子 deterministic 0.000，5 seed 不太可能反转 |
| ~~Asym-critic ablation on N2'~~ → **✓ COMPLETED 2026-05-27 (§4.5)** | — | verdict ACTOR_FUNDAMENTAL_CONFIRMED；区分 actor-vs-critic-fundamental 已落地，喂 paper §discussion/robustness |
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
| `results/offline/rebrac/broad_validation_v2/N0/seed_43/test_result.json` | N0 seed 43 raw eval (success 0.933) — 2026-07-12 supplement |
| `results/offline/rebrac/broad_validation_v2/N2p/seed_43/test_result.json` | N2' seed 43 raw eval (success 0.000) — 2026-07-12 supplement |
| `results/offline/rebrac/broad_validation_v2_n2p_asym/seed_43/test_result.json` | asym N2' seed 43 raw eval (success 0.000) — 2026-07-12 supplement |
| `results/offline/rebrac/broad_validation_v2/summaries/seed43_supplement_verdict.json` | supplement 三门槛预登记判读 verdict（CONSISTENT ×3） |
| `results/offline/rebrac/broad_validation_v2/summaries/verdict_decision.json` | anchor + per-cell mean/std/per-seed + verdict block（首轮 2-seed） |
| `results/offline/rebrac/broad_validation_v2/summaries/p1_overview.csv` | 7-col overview table |
| `experiments/offline/rebrac/broad_validation_v2/S_sanity/sanity_card_*.json` | S sanity baseline (DONE 2026-05-18) |
| `notebooks/rebrac_broad_validation_v2_core_completed_new.ipynb` | Colab execution notebook (2026-05-19 run) |
| `notebooks/rebrac_broad_validation_v2_seed43_supplement.ipynb` | +seed 43 supplement execution notebook (2026-07-08 Colab run，回读 2026-07-12) |

### 7.3 与其它 line 的 cross-link

- **Online 线** [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.6 — 提供 ceiling decomposition 的 online RL 行（vanilla SAC s0 cross_u15 = 0.10）
- **Main paper line** [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) rev.8 — 提供 §5.2 N0 anchor 比较的 main-line efficiency_v2 baseline (0.902 ± 0.021)
- **v1 archive** [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) — 已加 SUPERSEDED banner；保留作为 v1 实验历史

---

**Report 起草**: 2026-05-19（+seed 43 supplement 增补 2026-07-12）
**Plan rev**: rev.3 (with §6.2 2-seed note appended)；supplement plan+呈报+裁决 = [`rebrac_broad_validation_v2_seed43_supplement_plan.md`](rebrac_broad_validation_v2_seed43_supplement_plan.md)
**Reproducibility**: notebooks `rebrac_broad_validation_v2_core_completed_new.ipynb` + `rebrac_broad_validation_v2_seed43_supplement.ipynb` + raw `results/offline/rebrac/broad_validation_v2*/`
