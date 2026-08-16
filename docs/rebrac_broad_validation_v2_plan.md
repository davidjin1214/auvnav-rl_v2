# ReBRAC Broad Validation v2 — Cross-Only Spotlight under `arrival_v2`

> **文档版本**：2026-05-18 rev.3（pivot 版；rev.2 的 crosscomp-based N2 被 S sanity 证伪）
> **状态**：✅ **PASS — 首轮 completed 2026-05-19，三种子收口 2026-07-12**。N0 HOLDS (**0.878 ± 0.051**，3 seed {42, 0, 43}) / N2' STRONG_NEGATIVE (0.000 ± 0.000，0/90) / M1 not triggered；actor-fundamental partial-obs ceiling 实证确认。报告见 [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)。⚠ **本行原写首轮 2-seed 的 0.850 ± 0.024，2026-08-16 订正**。
> **作用**：取代 v1 broad validation 全部产出（spec / plan / report），把广验从「efficiency_v2 三轴 8 spoke + C1 deep-dive」收敛成「arrival_v2 cross-only **2 core cell + 1 conditional sweep**」。**Rev.3 关键改动**：S sanity 实测显示 crosscomp 在 cross_u15/s0/arrival_v2 下 success=0%（与 online §7.6 vanilla SAC = 0.10 共同证伪 "simple baseline 可救 critical regime"），privileged 同 setup 下 70%。因此 N2 collector 从 crosscomp 改为 privileged，N2 与 rev.2 原 N4 合并为单一 cell N2'，narrative 重 framing 为 "**actor-fundamental partial-observability ceiling under deployable s0 sensor**"。
>
> **取代的 v1 文档**（已加 SUPERSEDED banner）：
> - [`docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`](superpowers/specs/2026-05-04-rebrac-broad-validation-design.md)
> - [`docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`](superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md)
> - [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md)（v1 实验产物 archive，不重跑）
> - [`docs/rebrac_c1_s1_followup_report.md`](rebrac_c1_s1_followup_report.md)（v1 C1 follow-up archive）
>
> **不取代的文档**（保持不动）：
> - ReBRAC 主线三层文档（plan rev.8 / report rev.8 / mainline_review rev.3）— main paper 主线维持 `efficiency_v2`，v2 广验在 paper 里作为 partial-obs ceiling appendix
> - 既有 TD3+BC 归档文档（5 份 Phase 0c 系列）
>
> **配套参考**：
> - [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) — online 线 arrival_v2 实测（§7 strict-control + §7.6 s0 sensor envelope）
> - [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) — arrival_v2 v6 设计 spec（SHELVED 但是技术参考）
> - [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) — offline 线总览（status 同步更新）

---

## 目录

1. [TL;DR](#1-tldr)
2. [动机：为什么 v1 不够 + 为什么收敛到 cross + 为什么 rev.3 pivot](#2-动机)
3. [v1 → v2 关键变化（diff 表）](#3-v1--v2-关键变化)
4. [实验矩阵：2 core cell + 1 conditional](#4-实验矩阵)
5. [Pre-commit verdict gates](#5-pre-commit-verdict-gates)
6. [Stage 流程：sanity (done) → datasets → core train → conditional](#6-stage-流程)
7. [触发判据 / std 处理](#7-触发判据)
8. [Paper narrative role](#8-paper-narrative-role)
9. [Backlog（明确不在本广验范围内）](#9-backlog)
10. [Cross-link 与文档影响](#10-cross-link)
11. [文件清单（TBD / NEW）](#11-文件清单)
12. [执行检查清单](#12-执行检查清单)

---

## 1. TL;DR

- ReBRAC paper main results 维持 `efficiency_v2` / `cross_u10` anchor 不变（rev.8 已 closed，不翻盘）
- v2 广验在 `arrival_v2` reward 下做 **2 core cell (N0 sub-critical anchor + N2' critical regime with oracle teacher) + 1 conditional follow-up (M1 BC sweep)**，全部 cross-stream geometry
- **每个 cell 跑 3 seed [42, 43, 44]** — broad val 是 appendix，3 seed acceptable + 在文档中明标
- **S sanity 已完成（rev.3 pivot trigger）**：crosscomp@critical = 0% (0/30) ↔ privileged@critical = 70% (21/30)；前者证伪 rev.2 的 crosscomp-based N2 dataset 路径，后者证明 oracle teacher 可收
- 砍掉的 v1 内容：upstream / tandem / sbs geometry、goalseek / worldcomp / mix5050 collector、C1 deep-dive 5 ablation
- 砍掉的 rev.1 内容：N1（sub-critical sensor probe）、N3（critical sensor rescue）
- 砍掉的 rev.2 内容：N4 独立 cell（已合并到 N2'）
- 增加（rev.3）：N2' 用 privileged dataset；§5.3 verdict gate 重校准到 oracle ceiling；§8 paper framing 改为 "actor-fundamental partial-obs ceiling"
- paper 角色：**partial-observability ceiling appendix** — 围绕 "在 critical regime + deployable s0 actor 下，给 offline RL 喂 oracle teacher data，能突破多少 partial-obs ceiling" 这一核心 question
- 预算：**~6–24 h L4 + ~1 h CPU**（最好 vs 最坏情况）；S sanity 已花 ~2 min CPU

---

## 2. 动机

### 2.1 v1 的 reward 失配嫌疑

v1 全套实验跑在 `efficiency_v2` 下，但两条独立证据线都表明 `efficiency_v2` 在 upstream / 高难度 cell 上有 reward shaping 失配：

1. [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §3 — `efficiency_v2` 在 `single_u15_upstream` 上 vanilla SAC collapse 到 `success=0`，arrival_v2 修复到 1.000
2. v1 broad val C1 baseline termination 分布是 `goal=22.5 / timeout=77.5 / oob=0` — 这是 `efficiency_v2` 把 agent 钉在「保守憋边界」上的 footprint

含义：v1 finding 受 reward bias 污染，**不能在不切换 reward 的前提下被升格为 paper-quality finding**。

### 2.2 online 线已经给出 sensor envelope finding

online §7.6 显示：在 `arrival_v2 / cross_u15` 下 vanilla SAC + s0 catastrophic FAIL（`final=0.10 / OOB=0.667`），s1 PASS（`final=0.9`）。**s0–s1 gap = 80pp**，相比 A0 (cross_u10 + arrival_v1) 的 3pp **放大 24×**。

### 2.3 rev.3 pivot：S sanity 揭示 crosscomp dataset 路径不可行

**实测**（2026-05-18，30 ep on `single_u15_cross_tgt15` manifest，arrival_v2 reward, probe_layout=s0, history=4）：

| Collector | success | terminations | progress_ratio |
|---|---:|---|---:|
| crosscomp | **0.0%** (0/30) | timeout 8 / OOB 22 | −0.682 |
| privileged | **70.0%** (21/30) | goal 21 / OOB 7 / timeout 2 | +0.685 |

**含义**：
- crosscomp 在 critical regime 完全无法 produce viable demonstration data（dataset 100% 失败示范，ReBRAC BC penalty 训不出有意义 policy）
- 但 critical regime 不是 "无论如何都 unsolvable" — privileged teacher（拿到 hull-integral flow `[u_eq, v_eq]`）仍能 70% success
- 这两个 sanity 数据共同给出 paper §discussion 的 **performance ceiling decomposition**（见 §8.2）

rev.2 §6.2 给出的三个 fallback (a 降流速 / b 接受退化 / c 砍 critical cell) 都不如直接 pivot：**把 N2 collector 从 crosscomp 改成 privileged**，N2 与原 N4 合并为单一 cell N2'。

### 2.4 Rev.3 paper question 的物理本质 + 关键 caveat

**Paper core question**（修订后 framing）：**「Can offline RL with oracle demonstrations elevate a deployable s0-only policy above online RL's catastrophic failure floor in the critical regime?」**

**关键 caveat（必须 honestly addressed 在 paper 中）**：

ReBRAC 默认配置是 vanilla critic（`hidden=256, critic_LN=on`，**不带 asym critic**）。因此：

| Component | Input |
|---|---|
| Actor | s0 obs (40-D stacked) only |
| Critic | s0 obs only |
| Dataset | (s0_obs, **privileged_action**, ..., privileged_obs) |

ReBRAC actor 通过 BC penalty 学的是 **E[privileged_action | s0_obs]**，**不是 privileged_action 本身**。privileged baseline 用 `[u_eq, v_eq]` (hull-integral flow) 做决策，这个 oracle 信息 actor 看不到。

**因此 N2' 的实际 ceiling 远低于 privileged 70%**，因为：
- privileged_action 在 critical regime 下对 hull-integral flow 高度敏感
- s0_obs（单点 DVL 样本，AUV 中心位置）在 critical regime 下与 hull-integral flow weakly correlated（这正是 partial-observability gap 的物理本质 — sub-critical 下 strongly correlated，critical 下 weakly correlated）
- s0-conditioned imitation 是 intrinsically lossy

这是 **actor-fundamental partial-obs ceiling**：即便给 oracle dataset，s0 actor 也不能 fully recover privileged 的 decision rule。Paper claim 必须严格围绕这个 ceiling 来 frame，不能 oversell 为 "offline RL 突破 partial-obs"。

### 2.5 (U, Re, λ) 两 regime 的科学性 framing

v1 单档 (u10 / Re150 / λ=0.67) 是 sub-critical under-actuation；v2 增加的第二档 (u15 / Re250 / λ=1.0) 是 critical under-actuation regime。**两档是两个 physics regime 而非 continuous ladder**（涡街 Strouhal 频率 + wake 几何随 Re 变化）。Paper 写作时不当 ladder 用。

### 2.6 收敛到 cross-stream + 砍 N1/N3 的理由

- upstream + arrival_v2 + vanilla SAC online 已 saturate；offline ROI 低
- tandem / sbs 几何在主 narrative 里没承载责任
- **N1**（sub-critical s1 sensor probe）：v1 B1 已是 weak positive，sub-critical 下 sensor gap 本就小，砍
- **N3**（critical s1 sensor rescue）：online §7.6 已直接报告 cross_u15/s1 vanilla SAC = 0.90 PASS，offline 重测信息量低，砍

---

## 3. v1 → v2 关键变化

| 维度 | v1 (efficiency_v2) | v2 rev.3 (arrival_v2) | 理由 |
|---|---|---|---|
| **Reward** | `efficiency_v2` | `arrival_v2`（commit `813096e`） | v1 reward shaping 在 high-difficulty cell 失配 |
| **Task geometry** | C 轴 3 spoke | 只 cross_stream | upstream saturate；tandem 与主 narrative 重叠 |
| **Flow regime 轴** | 单档 u10/Re150 | 双档：u10/Re150 (sub-critical) + u15/Re250 (critical) | online §7.6 显示 critical 是 partial-obs gap 显化 regime |
| **Collector 集合** | 4 + 1 mix | **N0=crosscomp, N2'=privileged**（rev.3：crosscomp 在 critical regime 失败被 S sanity 证伪） | S sanity 数据；其余 collector ROI 低 |
| **Sensor 轴** | s0 / s1 / s2 | 只 s0 | N1/N3 砍后 sensor 只用 s0 |
| **Cell 数** | 8 spoke + C1 ablation 5 | **2 core (N0/N2') + 1 conditional (M1)** | 砍重复 / 砍 confirmatory；rev.3 进一步合并 N4 入 N2' |
| **Seed/cell** | 5 seed | **3 seed** [42, 43, 44] | broad val appendix，3 seed acceptable |
| **Mechanism discriminator** | C1 5 ablation 串联 | M1 BC penalty sweep on stuck N2'（conditional） | 直接、可证伪 |
| **Anchor** | `efficiency_v2` 5-seed 0.902 ± 0.021 | `arrival_v2` 3-seed N0 重测 | 新 reward 必须新 anchor |
| **N0 GO 阈值** | n/a | **≥ 0.70** | 严格 reward-bridge 判定 |
| **N2' verdict 校准** | n/a | 基于 privileged ceiling 70%（≥ 0.40 strong / 0.15-0.40 partial / < 0.15 negative） | rev.3 重新校准；不再用 absolute success threshold |
| **N0 dataset 路径** | n/a | **新 collect**（不 relabel） | rev.3 改：v1 transitions.npz 缺 `distance/elapsed_time` 字段，relabel 复杂度高于预期 |
| **N2' dataset** | n/a | privileged @ cross_u15/s0（rev.3 新增；rev.2 中 N2=crosscomp 已撤回） | S sanity 证明 privileged @ critical 可收 |
| **std_blow_up 阈值** | 0.042 = 2 × 旧 std | **改为 warning，不进 verdict gate** | 3 seed std df=2 不稳 |
| **Paper role** | standalone exploratory | partial-observability ceiling appendix | 明确 paper section + 严格 caveat |

---

## 4. 实验矩阵

**固定基底**：`cross_stream / target_speed=1.5 / arrival_v2 / probe_layout=s0 / ReBRAC (β1=4, β2=2, hidden=256, critic_LN=on, vanilla critic) / 64 epochs / 3 seeds [42, 43, 44] / test_episodes=100`

### 4.1 主体：1 sanity baseline (done) + 2 core cell

| Cell | 类型 | Collector | Sensor | Flow | (U, Re, λ) | 角色 | 状态 |
|---|---|---|---|---|---|---|---|
| **S** Sanity baseline | sanity | **crosscomp + privileged** | s0 | `wake_v8_U1p50_Re250` | (1.5, 250, 1.0) critical | paper §discussion ceiling decomposition：crosscomp = 0% (lower bound), privileged = 70% (oracle upper bound) | ✅ **DONE** (2026-05-18) |
| **N0** Anchor | core | crosscomp | s0 | `wake_v8_U1p00_Re150` | (1.0, 150, 0.67) sub-critical | reward-bridge baseline；与 main line `efficiency_v2 / 0.902` 对照 | 待跑（3 seed） |
| **N2'** Critical regime + oracle teacher | core | **privileged** | s0 | `wake_v8_U1p50_Re250` | (1.5, 250, 1.0) critical | actor-fundamental partial-obs ceiling probe：oracle teacher data + s0 actor + critic 能否突破 online catastrophic FAIL (§7.6 = 0.10) | 待跑（3 seed） |

### 4.2 Conditional follow-up

| Cell | 触发条件 | Collector | Sensor | Flow | 角色 | 成本 |
|---|---|---|---|---|---|---|
| **M1** BC sweep | N2' success ∈ [0.15, 0.40]（mid-ceiling stuck region） | privileged | s0 | u15/Re250 critical | mechanism discriminator：BC penalty floor vs actor-fundamental floor；β1 ∈ {0, 1, 2, 4, 8} × 3 seed = 15 runs | ~15h L4 |

**协同规则**：
- N2' < 0.15：M1 skip（β-tuning 在 catastrophic 区间无信息）。结论已是 strong negative（actor-fundamental ceiling）
- N2' ∈ [0.15, 0.40]：M1 跑 — 区分 "卡在 partial recovery 因为 BC 太硬" vs "卡在 actor-fundamental"
- N2' ≥ 0.40：M1 skip。结论已是 strong positive（offline RL meaningfully bridges gap）

### 4.3 已删除的 cells（vs rev.2）

| Cell (rev.2) | rev.3 处理 | 理由 |
|---|---|---|
| rev.2 N2 (crosscomp / critical) | **删除并替换为 N2' (privileged)** | S sanity 证伪 — crosscomp dataset 100% 失败示范 |
| rev.2 N4 (privileged / critical, conditional) | **合并到 N2' main** | N2' 已经用 privileged，N4 独立 cell 多余 |
| rev.1 N1 (s1 / sub-critical) | 已在 rev.2 删除（保持） | v1 B1 已 cover |
| rev.1 N3 (s1 / critical) | 已在 rev.2 删除（保持） | online §7.6 cross_u15/s1 = 0.90 已 cover |

---

## 5. Pre-commit Verdict Gates

**所有 gate 在跑实验之前预登记，不允许 post-hoc 调整**（v1 教训：A3 1-seed → 5-seed 反转）。

### 5.1 S Sanity baseline（DONE）

实测结果记录见 `experiments/offline/rebrac/broad_validation_v2/S_sanity/`：

| Sanity | success | verdict |
|---|---:|---|
| crosscomp @ cross_u15/s0/arrival_v2 (30 ep) | 0.0% | hand-coded baseline lower bound — dataset 不可收 |
| privileged @ cross_u15/s0/arrival_v2 (30 ep) | 70.0% | oracle ceiling 上界 — N2' dataset 可收 |

**S sanity 直接 GO to N2' dataset collection（using privileged，不用 crosscomp）。**

### 5.2 N0 Anchor GO/NO-GO（最关键）

| N0 3-seed success | Verdict | 后续动作 |
|---:|---|---|
| ≥ 0.70 | reward bridge holds | N2' 按计划跑 |
| 0.50–0.70 | weak bridge | N2' 仍跑，paper 写作时明标 "arrival_v2 下 ReBRAC anchor 比 efficiency_v2 下退化 X pp" |
| < 0.50 | **reward bridge 失败** | **暂停**所有后续 cell；review reward / collector / dataset；可能整套设计推翻 |

### 5.3 N2' Critical regime with oracle teacher（rev.3 重校准）

校准基线：privileged oracle ceiling 70%（actor-fundamental upper bound），online §7.6 catastrophic floor 10%。

| N2' success | vs online 10% | vs oracle 70% | Verdict | Paper claim |
|---:|---|---|---|---|
| ≥ 0.40 | substantial improvement | recovers ≥ 57% of teacher ceiling | strong positive | "offline RL with oracle demonstrations meaningfully bridges partial-obs gap under deployable sensor; s0-conditioned imitation extracts ≥ half of oracle ceiling" |
| 0.15–0.40 | mild–moderate improvement | recovers 21–57% of teacher | partial / trigger M1 | "offline RL extracts some value from oracle demonstrations, but s0-conditioned imitation is intrinsically lossy in critical regime" |
| < 0.15 | ≈ online catastrophic | < 21% of teacher | strong negative | "critical regime is **actor-fundamental** under s0 sensor; even oracle demonstrations cannot bridge the partial-obs gap when actor lacks hull-integral flow access" |

三档均有 publishable narrative。No "weak / unclear" zone。

### 5.4 M1 BC penalty sweep（conditional, if triggered）

| β1=0 success vs β1=4 baseline | Verdict |
|---|---|
| Δ > +10pp | BC penalty 是 limiting factor（floor 来自 BC 强度，非 actor-fundamental） |
| Δ ∈ [−5pp, +10pp] | β1 与 ceiling 解耦（floor 来自 actor-fundamental） |
| Δ < −5pp | β1=0 退化（BC penalty 在 stuck cell 仍提供 stabilization） |

---

## 6. Stage 流程

### 6.1 Pre-train phase

**Step 1: S sanity — DONE (2026-05-18)**

完成命令（reference）：

```bash
# crosscomp
python -m scripts.evaluate_baseline_on_manifest \
  --policy crosscomp \
  --manifest benchmarks/single_u15_cross_tgt15.json \
  --probe-layout s0 --history-length 4 --objective arrival_v2 \
  --target-speed 1.5 --task-geometry cross_stream \
  --num-workers 4 --seed 123 \
  --output-json experiments/offline/rebrac/broad_validation_v2/S_sanity/sanity_card_u15_cross_s0_crosscomp.json
# privileged — 同结构，只换 --policy privileged
```

结果见 §5.1。**结论触发 rev.3 pivot**。

**Step 2: N0 dataset 新 collect**（约 30 min CPU）

> **rev.3 改路径**：rev.2 原计划 relabel v1 dataset；研判发现 `arrival_v2` reward 计算需 `previous/current/initial_distance_to_goal_m` + `elapsed_time_s` 等字段，v1 transitions.npz 未存，需从 obs 反推 + 维护 episode boundary，复杂度过高且易错。直接新 collect 更稳。

```bash
python -m scripts.collect_offline_data \
  --policy crosscomp \
  --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective arrival_v2 \
  --episodes 1000 --seed 0 --num-workers 8 \
  --output-dir offline_data/crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000
```

写 `sanity_card.json` + verify `success_rate ≥ 0.70`（crosscomp 在 sub-critical 应该稳定，v1 metadata 0.87，arrival_v2 切换不影响 collector 行为）。

**Step 3: N2' dataset 新 collect**（约 30 min CPU；privileged sanity 已 70%，可收）

```bash
python -m scripts.collect_offline_data \
  --policy privileged \
  --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective arrival_v2 \
  --episodes 1000 --seed 0 --num-workers 8 \
  --output-dir offline_data/privileged_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000
```

写 `sanity_card.json` + verify `success_rate ≈ 0.70`（与 S sanity 一致）。

### 6.2 Core training（N0 + N2'，6 runs）

```bash
# Per cell, per seed（N0 示例；N2' 同结构换 dataset + manifest）
python -m scripts.train_offline \
  --algo rebrac \
  --offline-data offline_data/crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/transitions.npz \
  --manifest benchmarks/single_u10_cross_tgt15.json \
  --probe-layout s0 --history-length 4 \
  --task-geometry cross_stream --target-speed 1.5 --objective arrival_v2 \
  --sampling-mode shuffle_no_replacement --num-epochs 64 --batch-size 256 \
  --hidden-dim 256 --num-hidden-layers 3 \
  --actor-lr 3e-4 --critic-lr 3e-4 --gamma 0.99 --tau 0.005 \
  --actor-penalty-coef 4.0 --critic-penalty-coef 2.0 \
  --policy-noise 0.2 --noise-clip 0.5 --policy-freq 2 \
  --grad-clip-norm 10.0 --normalizer-eps 1e-3 \
  --critic-layernorm --no-actor-layernorm \
  --eval-every 0 --skip-final-eval --log-every 1000 \
  --seed <42|43|44> \
  --device cuda \
  --save-dir checkpoints/offline/rebrac/broad_validation_v2/N0/seed_<n>

# N2' 改用：
#   --offline-data offline_data/privileged_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000/transitions.npz
#   --manifest benchmarks/single_u15_cross_tgt15.json
#   --save-dir checkpoints/offline/rebrac/broad_validation_v2/N2p/seed_<n>

# Evaluation (after training, separate run):
#   python -m scripts.evaluate_offline --algo rebrac --checkpoint <save-dir> \
#     --manifest <same manifest> --episodes 100 --output-json <save-dir>/test_result.json
```

> **Smoke test (2026-05-18)**: N0 / seed=42 / num-epochs=2 / device=cpu / ~1 min wallclock. All ReBRAC losses converge correctly (critic 120→27, actor 4.5→0.4, bc 0.9→0.05). Pipeline verified on new env (`obs_dim=48`) + `arrival_v2` reward + `privileged_obs` columns. Full 64-epoch run extrapolates to ~30-45 min/seed on L4.

**执行顺序**：N0 3 seed 先跑 → §5.2 GO/NO-GO 判定 → 通过则 N2' 3 seed → §5.3 判定决定是否触发 M1。

**成本**：2 cell × 3 seed × ~1h L4 = **~6h L4**

> **Update 2026-05-18 (notebook execution pass)**：实际首轮 Colab pass 缩到 **2 seed [42, 0]**（4 runs ≈ 2–3h L4），目的是先快速拿到 verdict 信号。若 N2' 落入 §5.3 partial zone (mean ∈ [0.15, 0.40]) 或需要 spec 完整 3 seed power，再补 seed 43 收尾。Driver `notebooks/rebrac_broad_validation_v2_core.ipynb`。

### 6.3 Conditional M1（0 或 15 h L4）

执行判定：

```python
if 0.15 <= N2p_mean < 0.40:
    run_M1_bc_sweep()         # ~15h L4
else:
    skip_M1()                 # 结论已 strong (positive ≥0.40 或 negative <0.15)
```

**M1 BC sweep**：N2' 配置不变（privileged dataset, s0, critical regime）×  β1 ∈ {0, 1, 2, 4, 8} × 3 seed [42, 43, 44] = 15 runs ≈ 15h L4。

### 6.4 总预算

| 情形 | Sanity | Datasets | Core train | Conditional | Total |
|---|---|---|---|---|---|
| S done + N2' ≥ 0.40 (strong positive，无 M1) | done (~2 min CPU) | 1h CPU | 6h L4 | 0 | **~6h L4 + 1h CPU** |
| S done + N2' < 0.15 (strong negative，无 M1) | done | 1h CPU | 6h L4 | 0 | **~6h L4 + 1h CPU** |
| S done + N2' ∈ [0.15, 0.40] (M1 triggered) | done | 1h CPU | 6h L4 | 15h L4 | **~21h L4 + 1h CPU** |

对比：v1 ~50h L4 → **砍 ~70%**；rev.1 5-cell 28-40h L4 → 再砍 ~50%；rev.2 6-24h L4 → 持平偏低。

---

## 7. 触发判据

| 判据 | 阈值 | 触发后处理 |
|---|---|---|
| `mean_shift` | \|cell mean − N0 mean\| > **5pp** | 记录方向 + 是否进入 verdict gate 下一档 |
| `std_blow_up` | cell std > **2 × N0 std** | **仅 warning，不进 verdict gate**（3 seed std df=2 不稳） |
| `verdict_gate_violation` | cell success 落入 pre-commit 表的下一档 | 按 §5 verdict gate 处理 |

v1 旧 `td3bc_gap_collapse` 判据（A2 专用）**砍掉**。

---

## 8. Paper Narrative Role

v2 在 paper 里的明确角色：**§experiments appendix — partial-observability ceiling probe in critical regime**。

### 8.1 §experiments 引用方式

```
Main results (Section X.Y) report ReBRAC's +23.0 pp gain over TD3+BC on
`crosscomp / cross_u10 / s0 / efficiency_v2` (sub-critical regime). To
(a) verify this finding is not a `efficiency_v2`-specific artifact, and
(b) probe the actor-fundamental partial-observability ceiling identified
by the online sensor envelope analysis (§7.6: vanilla SAC + s0 achieves
only 0.10 in critical regime), we conduct two experiments:

  N0 (reward-bridge anchor): re-train under arrival_v2 on the same
  sub-critical setup. Verifies the +23pp finding persists under the
  redesigned reward.

  N2' (critical-regime oracle-teacher probe): re-train under arrival_v2
  in the critical regime (U=1.5, Re=250, λ=1.0) using privileged-teacher
  demonstrations. The privileged teacher achieves 0.70 success directly
  but uses hull-integral flow `[u_eq, v_eq]` unavailable to the actor;
  N2' tests whether offline RL with oracle demonstrations can elevate
  a deployable s0-only policy above the online catastrophic floor.

[Insert main table + verdict summary.]
```

### 8.2 §discussion: performance ceiling decomposition（核心 paper hook）

| Layer | Setup | Performance | Source |
|---|---|---:|---|
| hand-coded baseline | crosscomp / s0 / cross_u15 / arrival_v2 | **0.0%** (0/30) | S sanity (done) |
| online RL | vanilla SAC / s0 / cross_u15 / arrival_v2 / 1M steps | **10.0%** | online §7.6 |
| **offline RL + oracle teacher** | ReBRAC β1=4 / s0 / privileged dataset / arrival_v2 | **?** | **N2'** |
| oracle direct | privileged baseline / s0 actor 但 hull-integral knowledge | **70.0%** (21/30) | S sanity (done) |

**核心 paper claim**：N2' 落在这个 ceiling decomposition 的什么位置，决定 paper §discussion 主结论方向（见 §5.3 三档）。

**关键 caveat（必须 in paper）**：privileged oracle uses hull-integral flow knowledge `[u_eq, v_eq]` that the actor cannot observe. N2' actor's ceiling is bounded above by E[privileged_action | s0_obs]，not by privileged_action itself. The gap between N2' and the 70% oracle line **is** the actor-fundamental partial-observability ceiling.

### 8.3 与 main paper claim 的关系（明确不冲突）

- main paper claim "ReBRAC +23pp on crosscomp / cross_u10 / s0 / efficiency_v2" — **不变**
- v2 N0 结果用于声明 "this advantage persists under the redesigned `arrival_v2` reward (Section appendix)"
- v2 N2' 提供 partial-observability ceiling probe：要么 strong positive（offline RL bridges gap），要么 strong negative（critical regime is actor-fundamental），要么 partial + M1 mechanism discriminator
- 所有三种 N2' outcome 都是 paper-quality finding，无 "weak / unclear" 退路

---

## 9. Backlog（明确不在本广验范围内）

| Backlog item | 优先级 | 理由 |
|---|---|---|
| N1 sub-critical s1 sensor probe | 低 | v1 B1 已是 weak positive |
| N3 critical s1 sensor rescue | 低 | online §7.6 已直接报告 cross_u15/s1 = 0.90 PASS |
| 原 N4 独立 cell | 砍 | 已合并到 N2' |
| **AsymCritic ablation on N2'** | **中** | 如果 N2' < 0.40，asym critic（critic 拿 privileged_obs，actor 仍 s0）可能 partially close gap — 与 online 线 §8 P1#1 已列 asym critic ablation 对齐；可作为 rev.4 candidate |
| sub-critical 下 privileged sanity | 低 | privileged 在 sub-critical 接近 1.0 是自明的 |
| 其他 baseline (goalseek/worldcomp) 在 critical regime 的 sanity | 低 | crosscomp 已证明 hand-coded 在 critical 失败；goalseek/worldcomp 是 strict subset of crosscomp logic |
| C3 tandem geometry × arrival_v2 | 低 | cross 优先级更高 |
| upstream geometry × arrival_v2 × offline | 低 | online 已 saturate |
| Online SAC trained collector | 待 spec | online 线 sprint 中 |
| s2 (4 probes) cells | 低 | sensor envelope 重心在 s0 |
| Online cross_u10 s0/s1 1-seed 各 1 跑 | 中 | 闭合 §8.2 ceiling decomposition 的 sub-critical 行 |
| Flow regime intermediate (U=1.25 / Re~200) | 低 | 需 `generate_wake` 新跑 |
| 5-seed 补全（main cell 从 3 → 5） | 中 | 若审稿人 push back 3 seed power，对 N0/N2' 各补 2 seed |

---

## 10. Cross-link 与文档影响

> **Status (2026-05-19 首轮 → 2026-07-12 三种子收口；本块 2026-08-16 订正)**: ✅ **PASS** — 首轮 4-run（2 seed [42, 0]），2026-07-12 seed 43 supplement 补齐第三种子；种子组 {42, 0, 43} 与预登记 {42, 43, 44} 不重合，见 report §6.1 limitations。
> - **N0** (sub-critical, crosscomp/Re150): success = **0.878 ± 0.051**, per-seed [42=0.867, 0=0.833, 43=0.933] → §5.2 verdict **HOLDS** (Δ vs efficiency_v2 anchor 0.902 = **−2.42pp**)。⚠ 原写 **0.850 ± 0.024 / −5.20pp / paired same-direction**——第三种子反向 +3.1pp，**同向退化的读法已在 report §2.4 撤销**
> - **N2'** (critical, privileged/Re250): success = **0.000 ± 0.000**, per-seed [42=0.000, 0=0.000, 43=0.000] → §5.3 verdict **STRONG_NEGATIVE** (recovery_of_oracle = 0%，0/90 episode，rule-of-three 上界 ≈ 0.033)
> - **M1** BC sweep §5.4: **NOT triggered** (N2' ∉ [0.15, 0.40] partial zone)
> - **Paper claim ready**: actor-fundamental partial-obs ceiling under s0 in critical regime — 详见 [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md)
> - **Raw outputs**: `results/offline/rebrac/broad_validation_v2/{N0,N2p}/seed_{42,0}/test_result.json` + `summaries/{verdict_decision.json, p1_overview.csv}`

### 10.1 v1 文档处理（已加 SUPERSEDED banner）

| v1 文档 | 处理 |
|---|---|
| `docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md` | banner: "SUPERSEDED by v2 plan, retain as v1 design archive" |
| `docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md` | banner: 同上 |
| `docs/rebrac_broad_validation_report.md` (rev.2) | banner: v1 实验结果 archive |
| `docs/rebrac_c1_s1_followup_report.md` | banner: 同上 |

### 10.2 main paper 文档（不变）

| 文档 | 处理 |
|---|---|
| `docs/rebrac_experiment_plan.md` (rev.8) | 不动 |
| `docs/rebrac_experiment_report.md` (rev.8) | 不动 |
| `docs/rebrac_mainline_review.md` (rev.3) | §3.5 顶层 pointer 指向本 v2 plan |
| `docs/rebrac_method_section_draft.md` | 不动 |
| `docs/rebrac_paper_writing_index.md` | 写作期 §experiments appendix 添加 v2 入口 |

### 10.3 line-level summary

| 文档 | 处理 |
|---|---|
| `docs/offline_rl_line_summary.md` | §3.3 broad validation 段落 "v1 archive + v2 active plan rev.3"；§4 backlog 更新 |

### 10.4 Online 线 cross-link

`docs/arrival_v2_experiment_report.md` §7.6 是 v2 narrative 的 online 平行证据来源；paper 写作时 §discussion §8.2 ceiling decomposition 表直接引用 §7.6 = 0.10 数据。

---

## 11. 文件清单

### 11.1 待创建（NEW）

> **📍 2026-07-28 指针体检收口**：下表 Status 列停在计划期的 `TBD`，未随落地更新，读起来像一批应存在而缺失的文件。实核结论：
> - 四个 infra 文件（`broad_validation_v2_cell_registry.py` / `run_offline_rebrac_broad_v2.sh` / `summarize_broad_validation_v2.py` / `rebrac_broad_validation_v2_pretrain.ipynb`）**未创建**——[`notebooks/rebrac_broad_validation_v2_core.ipynb`](../notebooks/rebrac_broad_validation_v2_core.ipynb) 直接调 `scripts.train_offline_rebrac` 内联完成，无 registry / shell driver / summarize 环节。
> - `run_broad_validation_v2_bc_sweep.sh` 与 `rebrac_broad_validation_v2_conditional.ipynb` 是 M1 conditional，而 M1 **not triggered**（§5.4 / v2 report）；两者不存在是设计生效，不是缺失。
> - `rebrac_broad_validation_v2_core.ipynb` 已创建并闭环；另有计划外的 `_n2p_asym_critic` 与 `_seed43_supplement` 两册。
>
> 本注只记录落地实况，不改本表任何一行，也不动本 plan 的任何设计论述或数字。

| Path | Role | Status |
|---|---|---|
| `scripts/broad_validation_v2_cell_registry.py` | N0 / N2' / S / M1 配置 single source of truth | TBD |
| `scripts/run_offline_rebrac_broad_v2.sh` | core train driver（N0 / N2'），skip-resume | TBD |
| `scripts/run_broad_validation_v2_bc_sweep.sh` | M1 conditional sweep driver | TBD |
| `scripts/summarize_broad_validation_v2.py` | aggregation + verdict gate evaluation | TBD |
| `notebooks/rebrac_broad_validation_v2_pretrain.ipynb` | N0 + N2' dataset collection（S sanity 已完成，无需 notebook） | TBD |
| `notebooks/rebrac_broad_validation_v2_core.ipynb` | N0 + N2' 训练 + verdict gate 判定 + M1 决策 | TBD |
| `notebooks/rebrac_broad_validation_v2_conditional.ipynb` | M1（视触发） | TBD |
| `experiments/offline/rebrac/broad_validation_v2/<cell>/seed_*.json` | run results | TBD |
| `checkpoints/offline/rebrac/broad_validation_v2/<cell>/seed_*/` | checkpoints | TBD |
| `offline_data/crosscomp_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` | N0 dataset（**新 collect，不 relabel**） | TBD |
| `offline_data/privileged_s0_h4_arrival_v2_re250_u15cross_fixdone_ep1000/` | N2' dataset（rev.3 升格为必收） | TBD |
| `benchmarks/single_u15_cross_tgt15.json` | 已存在，复用 | EXISTS |
| `benchmarks/single_u10_cross_tgt15.json` | 已存在，复用 | EXISTS |

### 11.2 已删除（vs rev.2 计划）

| Path | 删除理由 |
|---|---|
| `scripts/relabel_rewards.py` | rev.3 改路径：N0 dataset 改为新 collect（见 §6.1 Step 2） |
| `offline_data/<crosscomp>_re250_u15cross_*` | rev.3 删除：S sanity 证伪 crosscomp 在 critical regime 可收 |

### 11.3 复用既有（不变）

| Path | Role |
|---|---|
| `auv_nav/rebrac.py` | ReBRAC agent；与 reward 解耦 |
| `auv_nav/reward.py` | `arrival_v2` reward preset (commit `813096e`) |
| `auv_nav/baselines.py` | crosscomp + privileged policy classes |
| `scripts/train_offline.py --algo rebrac` | ReBRAC 训练入口 |
| `scripts/evaluate_baseline_on_manifest.py` | baseline policy rollout (S sanity 用过) |
| `scripts/evaluate_offline.py` | manifest 评估 |
| `scripts/collect_offline_data.py` | offline data collection |
| `scripts/write_sanity_card.py` | dataset sanity card 工具 |

### 11.4 已完成（S sanity 产物）

| Path | Role |
|---|---|
| `experiments/offline/rebrac/broad_validation_v2/S_sanity/sanity_card_u15_cross_s0_crosscomp.json` | S sanity card — crosscomp (0%) |
| `experiments/offline/rebrac/broad_validation_v2/S_sanity/sanity_card_u15_cross_s0_privileged.json` | S sanity card — privileged (70%) |
| `experiments/offline/rebrac/broad_validation_v2/S_sanity/sanity_run.log` | crosscomp run log |

### 11.5 Wake data 检查（已确认）

执行前已确认两份 wake 文件本地 + Drive 可用：
- `wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` ✓
- `wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy` ✓

---

## 12. 执行检查清单

执行前确认：
- [x] 用户 review + approve plan rev.3
- [x] v1 文档已加 SUPERSEDED banner（spec / plan / report / c1_s1_followup）
- [x] Wake data 两份文件已确认可用
- [x] `auv_nav/reward.py` `arrival_v2` preset 在当前 branch 可用
- [x] **S sanity 已完成**（crosscomp = 0% / privileged = 70%）
- [x] `offline_rl_line_summary.md` 已同步更新 §3.3 / §4（rev.3 → PASS）
- [x] `mainline_review.md` §3.5 pointer 已更新（指向 v2 PASS report，2026-05-19）
- [x] Colab L4 至少 2 session 预算可用（实际 2-seed 单 session ~30 min L4 即闭环）

逐 Step 检查：
- [x] **Step 1 (S sanity)** crosscomp + privileged sanity cards 完成 + JSON 落盘
- [x] **Step 2 (N0 dataset)** crosscomp / s0 / cross_u10 / arrival_v2 新 collect 1000 ep 完成（dataset 路径见 §11.1）
- [x] **Step 3 (N2' dataset)** privileged / s0 / cross_u15 / arrival_v2 新 collect 1000 ep 完成
- [x] **Core N0** **2-seed [42, 0]** 完成 → §5.2 verdict 判定 = **HOLDS** (0.850 ≥ 0.70)
- [x] **Core N2'** **2-seed [42, 0]** 完成 → §5.3 verdict 判定 = **STRONG_NEGATIVE** (< 0.15) + M1 NOT triggered (∉ [0.15, 0.40])
- [N/A] **Conditional M1** 不触发（N2' = 0.0 < 0.15 catastrophic 区间，β-tuning 无信息，按 §4.2 协同规则 skip）

**Seed-count 说明**：本轮执行实际跑 2 seed 而非 plan §6.2 原定 3 seed。N2' 0.000/0.000 deterministic 给出 strong signal；N0 0.867/0.833 同向退化 informative。**3rd seed 补全** 已转入 [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) §6.3 follow-up backlog。

> ⚠ **2026-08-17 补注：上段是首轮当时的登记，已被后续执行超越**。第三种子（seed 43）已于 **2026-07-12** 按同协议补齐三单元（见 [`rebrac_broad_validation_v2_seed43_supplement_plan.md`](rebrac_broad_validation_v2_seed43_supplement_plan.md)），backlog 该条已闭。且"N0 同向退化 informative"这个读法**已被撤销**——seed 43 = 0.933 反向高于 anchor，三种子 per-seed 方向为 −3.5 / −6.9 / **+3.1** pp，均值差小于种子间标准差。现行数字见本文头部状态行与 report §2.4。种子组为 {42, 0, 43}，与预登记 {42, 43, 44} 仍不完全重合（首轮以 0 替换 44）；5-seed 补全仍在 §6.3 backlog。

paper 写作前最终检查：
- [x] 所有 verdict gate 结果 documented in v2 report (§1, §2.3, §3.3)
- [x] §8.2 ceiling decomposition 表 fully populated（含 S baseline 0.0% + N2' 0.0% + online §7.6 = 10.0% + privileged 70.0%）— v2 report §3.2 + §5.2
- [x] **核心 caveat (actor-fundamental partial-obs ceiling) 明确写入 v2 report** §5.1 / §5.2 / §5.4；paper §discussion 待起草
- [x] cross-link 到 `arrival_v2_experiment_report.md` §7.6 完整（v2 report §7.3）
- [x] v1 archive cross-link 完整（v2 report §7.3）
- [x] 3-seed 局限性在 v2 report 明确标注（v2 report §6.1，本节亦标注）

---

**END of v2 plan rev.3 — completed 2026-05-19**
