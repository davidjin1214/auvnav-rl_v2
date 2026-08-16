# FQL Succession Gate B — Interim Report (seed=42 single-seed)

> **⚠ SUPERSEDED（2026-07-28 指针体检补注）** — 下面这行 Status 写于 2026-05-19，其中的「pending」**早已不 pending**：Gate B 已收口，正式结论见 [`docs/fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md)（该报告 §头注即把本文件定性为 "rev.1 single-seed interim, superseded, kept for traceability"）。整条 FQL succession 线已于 **2026-05-23 NEGATIVE 闭环**（[`docs/fql_succession_p2_results.md`](fql_succession_p2_results.md)）。本文件仅保留作可追溯性存档，**不要据此认为有待跑实验**。
>
> **Status (2026-05-19)**: INTERIM — 3/4 PASS marginal-fail on `seed=42` single seed; **Option B re-run with `seed=0` + Bug 1 fix pending** on Colab L4.
>
> **Branch**: `codex-arrival-v2-prototype`
> **Notebook**: [`notebooks/fql_succession_gate_b.ipynb`](../notebooks/fql_succession_gate_b.ipynb) (rev.1 commit `5a72587`)
> **Spec**: [`docs/fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) §5 (Task E)
> **Plan**: [`docs/fql_succession_plan_v0.md`](fql_succession_plan_v0.md) §3 Lean MVP
> **Raw outputs**:
> - `results/offline/fql_succession/gate_b/{rebrac,fql}_e_uni_seed42/{test_result.json, eval_log.csv}`
> - `results/offline/fql_succession/gate_b/summaries/{gate_b_verdict.json, gate_b_overview.csv}`
> - `notebooks/fql_succession_gate_b_completed.ipynb` (Colab pass artifact)
> **Cross-link**: [`docs/rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) (N0 anchor 0.85 reference)

---

## 目录

- [§1 Abstract](#1-abstract)
- [§2 Setup](#2-setup)
- [§3 4 criteria 数字](#3-4-criteria-数字)
- [§4 c4 failure mechanism](#4-c4-failure-mechanism)
- [§5 3 bugs identified](#5-3-bugs-identified)
- [§6 Clean c4 recompute](#6-clean-c4-recompute)
- [§7 与 broad val v2 N0 anchor 对比](#7-与-broad-val-v2-n0-anchor-对比)
- [§8 Option matrix + 选定 Option B](#8-option-matrix--选定-option-b)
- [§9 Files / artifacts](#9-files--artifacts)
- [§10 P2 spec 写作早期影响](#10-p2-spec-写作早期影响)

---

## 1. Abstract

Seed=42 paired ReBRAC + FQL × 200k step uniform sampling 在 E-uni 1000-ep paper anchor (`privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000`) 上完成 Colab L4 训练。共 **2 runs**（ReBRAC ~25 min + FQL ~49 min L4）+ 2 final eval。

**4 criteria 应用结果（seed=42 单 seed）**：

| # | 指标 | 数值 | 阈值 | Pass |
|---|---|---:|---|:---:|
| c1 | FQL last-3 eval success mean | **0.689** vs ReBRAC 0.633 (Δ=+5.6pp) | FQL ≥ ReBRAC−0.08 | ✓ |
| c2 | FQL `loss_flow` 末段 | **0.023** | < 0.05 | ✓ |
| c3 | actor_loss OOM ratio | **1.007** | ∈ [0.1, 10] | ✓ |
| c4 | FQL last-30% eval slope | **−0.0034**（contaminated）/ **−0.0305**（clean）| ≥ 0 | ✗ |
| **OVERALL** | | | | **3/4 marginal-FAIL** |

**三句话结论**：

1. **FQL implementation 健康**：loss_flow 收敛 (0.023 < 0.05)，actor_loss 量级与 ReBRAC 几乎一致 (1.007×)，last-3 mean **反超** ReBRAC +5.6pp — Track-A/B 工程实现没有结构性 bug
2. **c4 failure 由 step 200k 单点暴跌驱动** (0.767 → 0.533)，train log 显示 step 199k 出现 transient critic spike (critic_loss 320 vs 邻近 0.5, td_abs_error 1.56 vs 0.4) — 是 Q-divergence event 而非系统性 collapse
3. **单 seed × 30-ep manifest 噪声底 ±8.4pp** + **eval log 跨 run 污染（详 §5）** 双重不确定性 → **必须 Option B 双 seed clean re-run 才能给出定论**；本报告作为 interim 留痕，待 Option B 闭环后升级为 final report

---

## 2. Setup

**Run matrix**（本轮已完成）：

| Run | Algo | Dataset | Manifest | Steps | Seed | Wallclock |
|---|---|---|---|---|---|---|
| 1 | ReBRAC | `privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000` | `single_u10_cross_tgt15` | 200k | 42 | ~25 min L4 |
| 2 | FQL | 同上 | 同上 | 200k | 42 | 49.0 min L4 |

**共有 CLI** (per-algo flag 略)：
```
--algo {rebrac,fql} --offline-data <E-uni> --manifest <single_u10_cross_tgt15>
--probe-layout s0 --history-length 4 --task-geometry cross_stream --target-speed 1.5
--objective arrival_v2 --total-steps 200000 --batch-size 256 --sampling-mode uniform
--hidden-dim 256 --num-hidden-layers 3 --actor-lr 3e-4 --critic-lr 3e-4
--gamma 0.99 --tau 0.005 --policy-noise 0.2 --noise-clip 0.5 --policy-freq 2
--grad-clip-norm 10.0 --normalizer-eps 1e-3
--eval-every 10000 --eval-episodes 100 --eval-workers 4 --eval-worker-device cpu
--log-every 1000 --checkpoint-every 0 --seed 42 --device cuda
```

**Per-algo 差异**：
- ReBRAC: `--actor-penalty-coef 4.0 --critic-penalty-coef 2.0 --critic-layernorm --no-actor-layernorm`
- FQL: `--flow-steps 10 --distill-alpha-bc 1.0 --teacher-lr 3e-4 --flow-time-embed-dim 32`

**Spec CLI 字段名 vs 实际**：见 notebook `gate-b-header` cell 表格；本 report 用 actual 名称；spec patch 待 Task E 闭环统一更新。

---

## 3. 4 criteria 数字

### 3.1 c1 last-3 eval success mean

| Algo | last-3 evals (step 180k, 190k, 200k) | mean |
|---|---|---:|
| ReBRAC | 0.567 / 0.667 / 0.667 | **0.633** |
| FQL | 0.767 / 0.767 / 0.533 | **0.689** |

Δ = +5.6pp, threshold ≥ −8pp → **PASS** with margin。

### 3.2 c2 FQL `loss_flow` 末段

末 5% train log rows (step ~190k-200k) 的 `loss_flow` mean = **0.023** (< 0.05 threshold) → **PASS**。

Teacher (flow-matching network) 训练曲线后段稳定收敛，预期之内。

### 3.3 c3 actor_loss OOM parity

| Algo | actor_loss 末 5% mean |
|---|---:|
| FQL | −0.974 |
| ReBRAC | −0.967 |

`|FQL|/|ReBRAC| = 1.007`，远在 [0.1, 10] 之内 → **PASS**。

FQL 的 Q-guided 改进项（actor 试图最大化 Q）与 ReBRAC 同一量级，验证 distillation alpha + Q-normalization 工作正常。

### 3.4 c4 FQL last-30% eval monotonicity

**官方报告 slope = −0.0034**（last30pct over n_eval=38 contaminated trajectory）→ **FAIL** (< 0)。

但污染掩盖了实际 slope 严重程度，详 §5 + §6。

### 3.5 Final 100ep test_result（实际 30ep, 见 Bug 2）

| Algo | success | return | safety | path_eff | progress |
|---|---:|---:|---:|---:|---:|
| ReBRAC | **0.667** | +31.9 | 5.12 | 0.720 | 0.823 |
| FQL | **0.533** | −16.6 | 6.48 | 0.676 | 0.748 |

ReBRAC final 100ep > FQL final 100ep by 13pp，但 c1 用的是 last-3 in-training mean（避免 final-eval single-shot noise）→ verdict 走 c1 不矛盾。

---

## 4. c4 failure mechanism

### 4.1 FQL eval 轨迹（clean 第二次 run 部分）

```
step    success    Δ vs prev
10000   0.333       —
20000   0.767     +0.43   ← teacher 快速 BC 学到 demo 平均
30000   0.533     −0.23
40000   0.533      0.00
50000   0.600     +0.07
60000   0.767     +0.17
70000   0.433     −0.33
80000   0.500     +0.07
90000   0.833     +0.33
100000  0.767     −0.07
110000  0.600     −0.17
120000  0.700     +0.10
130000  0.500     −0.20
140000  0.633     +0.13
150000  0.700     +0.07
160000  0.867     +0.17   ← peak (overall max of training)
170000  0.700     −0.17
180000  0.767     +0.07
190000  0.767      0.00
200000  0.533     −0.23   ← final-eval crash
```

**特征**：
- 轨迹 high-variance（±0.2 swing 常见，30-ep eval noise floor ±8.4pp 之外仍有真实波动）
- 最大值 0.867 at step 160k
- 末三 evals (180k, 190k, 200k) = 0.767/0.767/0.533，**final-eval drop 23pp**

### 4.2 Train log 末段 critic instability

| step | critic_loss | td_abs_error | actor_loss | loss_flow |
|---:|---:|---:|---:|---:|
| 196000 | 0.31 | 0.39 | −0.99 | 0.035 |
| 197000 | 0.39 | 0.44 | −0.97 | 0.027 |
| 198000 | 0.70 | 0.50 | −0.99 | 0.009 |
| **199000** | **320.83** | **1.56** | −0.98 | 0.016 |
| 200000 | 0.42 | 0.45 | −0.97 | 0.024 |

**Step 199k critic_loss spike 至 320**（邻近 step 平均 ~0.5，~640×）。`td_abs_error` 同步 spike 4× to 1.56。Step 200k critic loss 已恢复，但权重已被推偏，正是这次 eval 的 success rate 跌至 0.533 的直接原因。

**机制假设**（不在本 report 求证）：
- A: random batch with outlier transitions 触发 transient Q-divergence
- B: target network soft update (tau=0.005) 在某 step 推 Q 出 OOD region
- C: FQL student tanh squashing 在某状态点产生数值爆炸（gradient 通过 chain rule 放大）

Option B 双 seed run 会判断这是 seed-specific noise event 还是 protocol-level 系统性问题。如系统性，留 Option D（FQL stability ablation：critic LN / tau=0.001 / actor warm-up）作为 Gate B 后续 mitigation。

### 4.3 ReBRAC 对照（c4 也会 fail 但 spec 只 apply on FQL）

ReBRAC last-30% slope = −0.0133。spec §5.3 把 c4 限定 FQL 是因为「distillation collapse」是 FQL 特异风险；但实测 ReBRAC 在 200k uniform 末段也呈轻度漂移，**两个算法都受同一 protocol-level 末段不稳影响**。

---

## 5. 3 bugs identified

### 5.1 Bug 1 — eval_log.csv 跨 run 污染（影响 verdict）

**症状**：`results/.../fql_e_uni_seed42/eval_log.csv` 有 **38 数据行**（应是 20，按 200k/10k）。
- 行 1-18 (steps 10k-180k)：**首次** FQL run（agent_final.pt 未保存即中断）
- 行 19-38 (steps 10k-200k)：**第二次** run（完整）
- Seed=42 + deterministic eval seed = `seed+10000+step` → 两次 run 在相同 step **byte-identical**，行 1-18 与行 19-36 数字完全重复

**根因**：notebook §2 train cell 的 skip-resume 只看 `trainer_state.json + agent_final.pt`；training script 用 append 模式写 `train_log.jsonl` / `eval_log.csv`。第一次 run 崩溃时已写部分 log → 第二次 run 拼后面。

**Verdict 影响**：
- c1 / c2 / c3 **不受影响**（c1 看末 3 行，c2/c3 看 train_log 末 5%，污染段在中前部）
- **c4 被错误地 mask**：spec 算 "last 30% of trajectory"，污染后 trajectory=38 → last 30% = 12 行（实际来自 clean run 2 step 90k-200k 的较长窗口）→ slope 被平滑

**Train log 也被污染**：`train_log.jsonl` 386 行（应 ~201）。影响 §5 c2/c3 if 截 "last 5%"。本轮 train log 末段是 clean run 2 的尾部，未影响 verdict，但**潜在 risk**。

**Fix**：notebook §5 verdict cell 加 dedup-by-train_step（last write wins），下 commit 落地。

### 5.2 Bug 2 — `--episodes 100` 被 manifest 静默 override 至 30

**症状**：CLI `--episodes 100` 但 `test_result.json` 报 `num_eval_episodes: 30.0`。`scripts.evaluate_offline` 当 `--manifest` 给定时走 manifest episode 数，CLI flag 失效。`single_u10_cross_tgt15` manifest 有 30 ep。

**In-training eval 同问题**：`--eval-episodes 100` 在 train script 内被 manifest size 覆盖（同 evaluate_offline 行为），实际每次 in-training eval 也是 30 ep。

**影响**：
- 30 ep × success ≈ 0.7 → 单 eval 1-σ noise = √(0.7·0.3/30) = **±8.4pp**
- 0.767 → 0.533 单 eval 跌 23.3pp = ~2.8σ → borderline，不能完全归因 noise
- **c4 slope** 噪声底比预设高 1.8× — 6 个 eval 的 slope 测量 SE 显著放大

**Fix（不在本 commit）**：option 1 = 改用更大 manifest (`_ep100` 变体 — 待生成)；option 2 = spec 改 `eval_episodes` 默认为 manifest size，c4 阈值随之调整。**留待 Option B 之后讨论**。

### 5.3 Bug 3 — notebook §5.5 字段名错误（cosmetic）

**症状**：§5.5 cell 用 `d.get('eval_avg_return', ...)`，实际字段是 `eval_return`（无 `_avg_` 前缀）。同 `eval_avg_safety_cost` / `eval_avg_progress_ratio` / `eval_avg_path_efficiency`。运行结果 4 列全 nan。

**影响**：纯打印 cosmetic，不影响 §5 verdict（用 eval_log.csv 路径）。

**Fix**：下 commit 把 `eval_avg_*` 改 `eval_*`。

---

## 6. Clean c4 recompute

### 6.1 Dedup 后 FQL clean trajectory

应用 dedup-by-train_step 后 FQL eval 数据回到 20 unique 行（即 §4.1 表的 20 个数）。

### 6.2 Clean last-30% (last 6 evals, steps 150k-200k)

```
[0.700, 0.867, 0.700, 0.767, 0.767, 0.533]
```

Linear slope：
- mean_x = 2.5, mean_y = 0.7223
- numerator = Σ(x-2.5)(y-0.7223)
  - (−2.5)(−0.0223) = +0.0558
  - (−1.5)(+0.1447) = −0.2171
  - (−0.5)(−0.0223) = +0.0111
  - (+0.5)(+0.0447) = +0.0224
  - (+1.5)(+0.0447) = +0.0670
  - (+2.5)(−0.1893) = −0.4733
  - sum = **−0.5341**
- denominator = Σ(x-2.5)² = 17.5
- **slope = −0.5341 / 17.5 = −0.0305**

### 6.3 c4 verdict 重算

| Calculation | Slope | Pass |
|---|---:|:---:|
| Contaminated last 30% of n=38 trajectory (steps 90k-200k, 12 pts) | −0.0034 | ✗ |
| **Clean last 30% of n=20 trajectory (steps 150k-200k, 6 pts)** | **−0.0305** | ✗ |

**结论**：dedup 修复 Bug 1 后 c4 slope 比官方报告**更负 9×**。c4 仍 FAIL，但更 decisive。final-eval drop 0.767 → 0.533 是真信号，非污染产物。

### 6.4 ReBRAC clean last-30% (last 6 evals, steps 150k-200k)

```
[0.667, 0.800, 0.633, 0.567, 0.667, 0.667]
```

slope = −0.0133 (matches verdict.json，ReBRAC 无污染)

**两算法 slope 比**：FQL −0.0305 vs ReBRAC −0.0133 = **2.3×** more negative。FQL 末段不稳定确实强于 ReBRAC，但绝对值都在「次级波动」量级。

---

## 7. 与 broad val v2 N0 anchor 对比

> ⚠ **追注（2026-08-16）**：本表的 v2 N0 列是**当时**的两种子值 `0.850 ± 0.024`。该 anchor 已于
> 2026-07-12 补齐第三种子，现为 **0.878 ± 0.051（3 seed {42, 0, 43}）**（report §2.4）。
> 本文档已 SUPERSEDED，表值作为当时对照的历史记录**保留不改**；引用当前 anchor 请用 0.878。

| metric | broad val v2 N0 (crosscomp, 2 seed) | Task E ReBRAC (privileged, 1 seed) | Task E FQL |
|---|---:|---:|---:|
| final test success | **0.850 ± 0.024** | 0.667 (30 ep) | 0.533 (30 ep) |
| last-3 mean | n/a | 0.633 | 0.689 |
| dataset | `crosscomp_s0_h4_arrival_v2_...` | `privileged_s0_h4_arrival_v2_...` | 同 |
| protocol | shuffle, 64 epoch (~22k steps) | uniform, 200k steps | uniform, 200k steps |

**两个核心差异解释 gap**：

1. **Dataset 差异 (crosscomp vs privileged)**：
   - crosscomp policy 用 s0 传感器输入 → dataset (s0_obs, crosscomp_action) 完美自洽，imitation 容易
   - privileged policy 用 hull-integral flow `[u_eq, v_eq]` 决策 → dataset (s0_obs, priv_action) 是 partial-obs imitation；actor 学到 `E[priv_action | s0_obs]`，**即使在 sub-critical cross_u10 也有 partial-obs gap**
   - 这本身就是 **paper-quality finding**：oracle teacher under s0 在 sub-critical 也只到 ~0.6-0.7（vs same-sensor teacher 0.85）；P2 main comparison 写作时入 §discussion

2. **Protocol 差异 (200k uniform vs 22k shuffle)**：
   - 22k shuffle = 64 epoch，每 transition 见 64 次 — 高 epoch 训练
   - 200k uniform = 每 transition 平均见 200k×256 / 86685 ≈ 590 次 — 远超 epoch 训练
   - **过训风险**：两个算法 last-30% 都呈负 slope，疑似 200k 过长。**P2 spec 该考虑**：100k step 或 best-last-k 取代 last-3

---

## 8. Option matrix + 选定 Option B

| Option | 操作 | 成本 | 信息量 |
|---|---|---|---|
| A | 删 stale logs + 重跑 FQL seed=42 1 seed | ~50 min L4 | 修 Bug 1，clean c4 = −0.0305 已知 |
| **B (选定)** | 删 stale logs + Bug 1 fix + 加 seed=0 双跑两算法 4 runs | ~3 h L4 | 双 seed 给统计 power，能区分「seed-42 是 outlier」vs「protocol-level 系统问题」 |
| C | 接受 marginal-fail，附 caveat 进 P2 | 0 | 低（若 c4 是真问题，P2 main comparison 会 surprise）|
| D | FQL stability ablation (critic LN / tau=0.001 / 100k step / actor warm-up) | ~50 min × N | Gate B 后续，**留作 Option B 之后 conditional** |

**选定 Option B 理由**：
- 单 seed × 30-ep noise 不足以 final-verdict
- 双 seed = 标准方差估计起点（df=1，但比 df=0 强）
- 一步到位 4 runs，避免反复 sync Drive
- 若 Option B 也 c4 FAIL，触发 Option D（FQL stability tuning）
- 若 Option B c4 PASS，Gate B 4/4 进 P2 main comparison spec

---

## 9. Files / artifacts

### 9.1 本轮 Colab 闭环产物

| Path | Status | Size |
|---|---|---|
| `results/offline/fql_succession/gate_b/rebrac_e_uni_seed42/test_result.json` | clean | ~30 ep eval JSON |
| `results/offline/fql_succession/gate_b/rebrac_e_uni_seed42/eval_log.csv` | clean (20 rows) | ~20 rows |
| `results/offline/fql_succession/gate_b/fql_e_uni_seed42/test_result.json` | clean | ~30 ep eval JSON |
| `results/offline/fql_succession/gate_b/fql_e_uni_seed42/eval_log.csv` | **contaminated (38 rows)** | dedup at verdict time |
| `results/offline/fql_succession/gate_b/summaries/gate_b_verdict.json` | based on contaminated logs | 重算后更新 |
| `results/offline/fql_succession/gate_b/summaries/gate_b_overview.csv` | 同上 | 同上 |
| `notebooks/fql_succession_gate_b_completed.ipynb` | Colab 跑完保存 | ~22 cells |

### 9.2 Checkpoint 状态（Drive only，gitignored）

| Path | Status |
|---|---|
| `checkpoints/.../rebrac_e_uni_seed42/{agent_final.pt, trainer_state.json, train_log.jsonl}` | 干净，Option B reuse |
| `checkpoints/.../fql_e_uni_seed42/{agent_final.pt, trainer_state.json, train_log.jsonl, eval_log.csv}` | **train_log.jsonl + eval_log.csv 污染**，需 Option B 重跑前删除 |

### 9.3 Notebook 待打补丁（本 commit 落地）

| File | Patch |
|---|---|
| `notebooks/fql_succession_gate_b.ipynb` | (a) §1 run matrix `SEEDS = [42, 0]` 4 runs; (b) §5 verdict 加 `_dedup_eval_log()` / `_dedup_train_jsonl()`; (c) §5.5 字段名 `eval_avg_*` → `eval_*`; (d) §5 aggregate per-algo across seeds |

---

## 10. P2 spec 写作早期影响

无论 Option B 结果如何，本 interim report 已暴露的 3 个问题都要在 P2 spec 起草时考虑：

1. **Eval episode 数 = manifest size (30)**（Bug 2）→ P2 bootstrap CI / Welch 显著性需按 30 ep 重新校准 power；或生成 `single_u10_cross_tgt15_ep100` 大 manifest，spec 显式用之
2. **200k uniform 末段不稳**（§4 / §6 / §7）→ P2 该考虑 100k step OR best-last-k（如 mean of best 3 evals in last 30%）取代 last-3
3. **FQL Q-instability transient**（§4.2 step 199k spike）→ 若 Option B 双 seed 看到反复发生，属系统问题；P2 ablation 候选：critic LN on FQL / tau=0.001 / actor warm-up / smaller batch
4. **Privileged dataset partial-obs ceiling under s0**（§7.1）→ Task E 实测 ~0.6-0.7 vs crosscomp 0.85；P2 §discussion 应入这条 "oracle teacher under s0 sub-critical gap" finding（独立于 c4 verdict）

---

**Report 起草**: 2026-05-19
**Notebook commit baseline**: `5a72587` (Gate B rev.1)
**Reproducibility**: notebook `fql_succession_gate_b_completed.ipynb` + raw `results/offline/fql_succession/gate_b/`
**Next action**: notebook patch (本 commit) + Drive sync + Colab Option B 4 runs
