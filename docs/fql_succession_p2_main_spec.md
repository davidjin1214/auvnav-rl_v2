# FQL Succession — P2 Main Comparison Spec

> **文档版本**：v1.2（2026-05-21，wallclock-budget + storage-layout 重设)
> **作用**：把 [`fql_succession_plan_v0.md`](fql_succession_plan_v0.md) §4.2 P2 main comparison（3-cell × 2-algo × 2-seed primary = **12 runs**, 可扩 3-seed = 18 runs）拆成可执行的 collection / audit / run / verdict 协议。
> **状态**：**Active spec**，pre-requisites 全部落地，等待 P2 sprint 0 (collection) 启动信号。
>
> **版本历史**:
> - v1.0 (2026-05-20 起草) — 10 节初稿:scope + collection + audit + run matrix + statistics + Gate C + risks + budget + notebooks + caveats
> - v1.1 (2026-05-20 patches) — Session A 4-decision refinements:(a) §2.3 M-uni-noise success 4-band contingency; (b) §2.4 M-multi-mix 2-way → 3-way conditional upgrade rule + P3 implication; (c) §5.5 Bonferroni primary + BH/uncorrected sensitivity table; (d) §9.3 `run_one()` helper 模板 + idempotent `summarize`/`verdict_preview` guards
> - **v1.2 (2026-05-21 patches)** — Session A wallclock-budget + storage-layout 重设 per user feedback:(a) §4.1 n_seeds 5 → 2 primary [42, 0],可扩 3 [42, 0, 7](L4 wallclock 从 19h → 8h,12 run vs 30 run);(b) §4.2/§9.3 helper 拆 `checkpoints/` (大文件) + `results/` (绘图包) 两棵树,加 `results/training_curves/` mirror 4 个 small file (train_log.jsonl + eval_log.csv + trainer_state.json + train_config.txt;final test 由独立 `evaluate_offline --output-json` 写入 `results/test/`),Colab 跑完仅回收 `results/` 即可本地绘图;(c) §5.4/§5.5 effect-size + 方向一致性 primary verdict(n=2 下 Welch p / Bonferroni 信息量低,移到 sensitivity);(d) §9 collection 改本机非 notebook 执行,删 collection notebook,run notebook 减为 3 个 cell(per algo × seed = 4 个 run cell)
> **范围**：仅覆盖 P2（约 1.5-2 周）。P3 mix ratio ablation 与 P4 writing 的 spec 在 P2 闭环后另写。
> **前置阅读**：
> - [`fql_succession_plan_v0.md`](fql_succession_plan_v0.md) v1.2 — plan 总览 + §3 spectrum + §6 D17/D18/D19
> - [`fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) v1.1 — P0+P1 spec（共享参数 + Gate A.1/A.2/B 历史 + Bug 2 / c4 patch）
> - [`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md) — Gate B closure (3/4 PASS + c4 marginal-FAIL seed-driven)
> - [`fql_succession_bug2_fix_decision.md`](fql_succession_bug2_fix_decision.md) — D18 ep100 manifest pre-requisite
> - [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) — D19 c4 阈值 Option α
> - [`fql_e_uni_anchor_dataset_card.md`](fql_e_uni_anchor_dataset_card.md) — E-uni 1000-ep dataset (Task D 闭环)
> - [`fql_audit_multimodality_design.md`](fql_audit_multimodality_design.md) — audit 工具 design
> - [`fql_audit_dryrun_report.md`](fql_audit_dryrun_report.md) — audit dry-run Gate A.2 PASS 4/4

---

## 目录

1. [Scope + 与 P0+P1 关系 + Gate B caveat](#1-scope--与-p0p1-关系--gate-b-caveat)
2. [3-cell spectrum collection 协议](#2-3-cell-spectrum-collection-协议)
3. [Per-cell multimodality audit](#3-per-cell-multimodality-audit)
4. [Run matrix](#4-run-matrix)
5. [统计协议](#5-统计协议)
6. [Gate C 判据](#6-gate-c-判据)
7. [Risk + mitigation](#7-risk--mitigation)
8. [算力预算](#8-算力预算)
9. [Notebook scaffold 计划](#9-notebook-scaffold-计划)
10. [Caveat 段](#10-caveat-段)

---

## 1. Scope + 与 P0+P1 关系 + Gate B caveat

### 1.1 一句话目标

> 在 3-cell spectrum (E-uni / M-uni-noise / M-multi-mix) × 2 algorithm (FQL, ReBRAC) × 2 seeds primary [42, 0]（可扩 3 [42, 0, 7]）= **12 runs** 上系统对比，量化 paper 2 的核心 conditional iff claim："FQL 的 expressive behavior prior translates to policy improvement over ReBRAC **iff** data is **both** sub-optimal **and** multi-modal"。Verdict 以 **effect-size + 两 seed 方向一致性** 为 primary 判据,Welch p / Bonferroni 留作 sensitivity。

### 1.2 与 P0+P1 的关系

P0+P1 完成 [`fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) 的 5 个 task:Gate A.1 (cite) + Gate A.2 (audit dry-run PASS 4/4) + FQL 实现 + E-uni 1000-ep collection + Gate B sanity。P2 在此基础上:

| P0+P1 产物 | P2 复用方式 |
|---|---|
| `auv_nav/fql.py` (frozen at Gate B Option B) | 不动，直接使用 |
| `auv_nav/rebrac.py` (frozen) | 不动 |
| `scripts/train_offline.py --algo {rebrac,fql}` | 不动 |
| `scripts/concat_offline_datasets.py` | M-multi-mix collection 直接调用 |
| `scripts/audit_multimodality.py` | Per-cell audit 直接调用 |
| `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` (E-uni) | P2 E-uni cell 直接 reuse (不再收集) |
| `benchmarks/single_u10_cross_tgt15_ep100.json` (D18 ep100 manifest) | P2 default eval source (取代 30-ep) |
| Gate B Option B per-algo CLI (ReBRAC β1=4/β2=2/critic_LN + FQL flow-steps=10/distill=1.0/teacher-lr=3e-4) | P2 per-algo 沿用，仅切 manifest + seed + dataset |

### 1.3 Gate B caveat 承接

Gate B Option B (2-seed) 闭环 verdict: **3/4 PASS + c4 marginal-FAIL (seed-driven)**。详见 [`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md):

- **c1/c2/c3 PASS**:FQL implementation healthy (loss_flow 0.026, actor_loss ratio 1.011, last-3 mean only −4.4pp vs ReBRAC)
- **c4 marginal-FAIL**:aggregated slope −0.0038 is **statistically indistinguishable from 0** (z=−0.27 against 30-ep manifest noise floor); **seed-driven**(both algos positive on seed=0, negative on seed=42); **not** FQL-driven
- **Mitigation activated (D17, scaled in v1.2)**:按 P0+P1 spec §5.3 "mixed → caveat + 3-seed extension" 路径,P2 用 n_seeds=2 primary [42, 0] retire-or-confirm 此 marginal finding;若 verdict 边缘扩 n_seeds=3 [42, 0, 7];**不** trigger Option D (FQL stability ablation)

P2 spec 在以下章节明确承接此 caveat:
- §4.1 n_seeds=2 primary [42, 0]、可扩 3 [42, 0, 7](vs Gate B 的 2-seed [42, 0])
- §4.2 manifest 切到 ep100 (D18)
- §5.6 c4 阈值用 Option α (D19), threshold ≈ −0.0155 at P2 default config (n=2 + ep100)
- §10.1 paper 写作 caveat 段

### 1.4 P2 出口三选一 verdict (preview, 详见 §6)

| Verdict | iff 状态 | Paper claim adjustment | P3 trigger |
|---|---|---|---|
| **完整 iff** | E-uni null + M-uni-noise null + M-multi-mix positive | Paper claim 主路径 (conditional iff hold) | 进 P3 mix ratio sweep |
| **Partial iff** | E-uni null + **M-uni-noise also positive** + M-multi-mix positive | Paper claim 降级:"FQL > ReBRAC on sub-optimal data" (drop modality 条件) | 进 P3,但 modality intensity 单调性变 secondary evidence |
| **Null iff** | M-multi-mix not positive | Paper claim 重写为 scoped negative finding ("expressive prior 在本任务无 leverage") | P3 决策点;按 R3 mitigation 写完 negative paper or pivot |

---

## 2. 3-cell spectrum collection 协议

### 2.1 共享参数 (所有 cell 强制一致)

| 参数 | 值 | 锚定 |
|---|---|---|
| Flow file | `wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` | E-uni dataset card §2 |
| Probe layout | `s0` | plan §3.3 / P0+P1 spec §0.2 |
| History length | 4 | 同上 |
| Task geometry | `cross_stream` (隐含 in benchmark) | 同上 |
| Target speed | 1.5 m/s | 同上 |
| Reward objective | `arrival_v2` | plan §2.3 (efficiency_v2 禁用) |
| Episodes per cell | **1000** | plan §3.3 |
| obs_dim / action_dim | 48 / 2 | E-uni dataset card §3.3 |
| Privileged_obs 字段 | required (`include_episode_context_obs=true`) | E-uni dataset card §3.5 |
| Num workers (本机 collection) | 6 (M-series 10-core) | E-uni dataset card §2 |

### 2.2 E-uni cell — 已就位 (reuse Task D)

| Item | Value |
|---|---|
| Path | `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` |
| Policy | `privileged` (PrivilegedCorridorPolicy) |
| Action noise | 0.1 |
| Seed | 0 |
| Episodes | 1000 |
| success_rate | 0.985 |
| num_transitions | 86,685 |
| mean_return | 132.56 ± 46.38 |
| Status | **✓ Collected (2026-05-19)** — see [`fql_e_uni_anchor_dataset_card.md`](fql_e_uni_anchor_dataset_card.md) |

**P2 不再 re-collect**。直接消费。

### 2.3 M-uni-noise cell — 待 P2 收集

| Item | Value |
|---|---|
| Path (target) | `offline_data/fql_succession/m_uni_noise_eps0p5_1000/` |
| Policy | `privileged` (与 E-uni 同 collector) |
| Action noise (key spec diff) | **ε=0.5** (vs E-uni 的 0.1) |
| Seed | 1 |
| Episodes | 1000 |

**Collection 命令**:

```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.collect_offline_data \
    --policy privileged \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --episodes 1000 \
    --seed 1 \
    --action-noise 0.5 \
    --num-workers 6 \
    --output-dir offline_data/fql_succession/m_uni_noise_eps0p5_1000
```

**预期 collection statistics** (推测,实际 collection 后写入 dataset card):

| Metric | Expected range | 备注 |
|---|---|---|
| success_rate | 0.50 – 0.75 | ε=0.5 substantially degrades vs E-uni 0.985, hits "sub-optimal" target |
| mean_return | 50 – 100 | 退化 ~30-60% vs E-uni 132 |
| num_transitions | ~120k – 180k | longer episodes due to failures |
| Wall-clock | ~3-5 min (6 workers) | scale linearly from E-uni 2.6 min |

**Success-rate 4-band contingency** (我们没有 ε=0.5 实测点,以下 band 用于 collection 后 decide-or-mitigate):

| Band | success_rate | 判定 | Action |
|---|---|---|---|
| **A (target)** | 0.50 – 0.75 | M-uni-noise 实现 "sub-optimal" 角色,paper claim 有效 | ✓ 继续 audit + run matrix |
| **B (weak)** | 0.75 – 0.90 | 仅 weakly sub-optimal,quality 退化轴不强 | 接受但 paper §discussion 标 "M-uni-noise represents weak quality degradation" |
| **C (too degraded)** | < 0.50 | data dominated by failure traj,FQL/ReBRAC 都学不到 signal | 降 ε 到 0.3 重 collect (mitigation §7.P2.R7) |
| **D (no effect)** | 0.90 – 0.98 | ε=0.5 对 privileged 无显著扰动 (signal SNR 仍 dominant) | 升 ε 到 0.7 重 collect |

**Audit 预期**: §3 audit 要求 `p_≥2 < 0.20` (单峰),证明 noise widening **没有** 制造 multimodality。如果 audit fail (p_≥2 > 0.30),则 ε=0.5 实际造成 mode split,本 cell 失效 → mitigation §7.P2.R1。

### 2.4 M-multi-mix cell — 待 P2 收集

| Item | Value |
|---|---|
| Path (target) | `offline_data/fql_succession/m_multi_mix_50priv_50goal_1000/` |
| Source A (mode A 源) | `offline_data/fql_succession/_components/privileged_500_seed100/` (新建) |
| Source B (mode B 源) | `offline_data/fql_succession/_components/goalseek_500_seed200/` (新建) |
| Mix strategy | episode-level 50/50 |
| Total episodes | 1000 (500 + 500) |

**Source A collection (privileged-500)**:

```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.collect_offline_data \
    --policy privileged \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --episodes 500 \
    --seed 100 \
    --action-noise 0.1 \
    --num-workers 6 \
    --output-dir offline_data/fql_succession/_components/privileged_500_seed100
```

**Source B collection (goalseek-500)**:

```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.collect_offline_data \
    --policy goalseek \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --episodes 500 \
    --seed 200 \
    --action-noise 0.1 \
    --num-workers 6 \
    --output-dir offline_data/fql_succession/_components/goalseek_500_seed200
```

**Mix step**:

```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.concat_offline_datasets \
    --input-dir offline_data/fql_succession/_components/privileged_500_seed100 \
    --input-dir offline_data/fql_succession/_components/goalseek_500_seed200 \
    --output-dir offline_data/fql_succession/m_multi_mix_50priv_50goal_1000 \
    --mix-strategy episode_level \
    --task-sampler anchor_distribution
```

**Audit 预期**: §3 audit 要求 `p_≥2 > 0.30` (本设计目标:multimodal by construction)。Audit dryrun (3-way mix) 实测 p_≥2 = 0.770,但 dryrun 用 3 collectors (privileged+worldcomp+crosscomp)。**本 P2 用 2-way (privileged+goalseek)** per plan §3.1 D16 — 更 conservative 测试,但可能 multimodality 弱于 dryrun。

**2-way → 3-way conditional upgrade rule** (audit 实测后 decide):

| 实测 p_≥2 (M-multi-mix) | Verdict | Action |
|---|---|---|
| **≥ 0.50** | Strong multimodal,2-way mix 设计成功 | ✓ 继续 — 用此 dataset 跑 run matrix |
| **0.35 – 0.50** | Marginal multimodal,Gate C.2 PASS by thin margin | ✓ 接受 — 用此 dataset;paper §discussion 标 "M-multi-mix marginally exceeds multimodal threshold" |
| **< 0.35** | Multimodality 不足 | ⚠ 升级 3-way:加入 worldcomp 一个 source (333 ep, seed=300),把 privileged + goalseek 各降到 333 ep,total 仍 1000 ep;重 concat + re-audit |
| **(audit otherwise fails)** | E.g. Welch p > 0.07 但 p_≥2 > 0.35 | Borderline,visual review per-anchor mode distribution 决定 |

**P3 implication (重要)**:P3 mix ratio sweep design 假设 **2-way** mix (ratio swept on simplex of 2 components: 30/70, 50/50, 70/30)。如果 P2 触发 3-way upgrade:
- **Option (a)** P3 仍走 2-way (privileged+goalseek) ratio sweep,即使 P2 主 cell 是 3-way → paper §method 解释 "P2 main cell upgraded to 3-way for multimodality strength;P3 ablation kept 2-way to isolate ratio effect from collector identity"
- **Option (b)** P3 改 3-way simplex (privileged/goalseek/worldcomp triangular ratios) → spec 复杂度 ↑,wallclock ↑;不推荐
- **本 spec 默认 Option (a)**,P3 spec 闭环时再 re-confirm

Mitigation 详见 §7.P2.R2。

### 2.5 Cell 不做的事

- ❌ 不混 reward variant (统一 arrival_v2)
- ❌ 不混 task geometry (统一 cross_stream)
- ❌ 不混 probe layout (统一 s0)
- ❌ 不变 sensor envelope
- ❌ 不变 flow file
- ❌ 不引入 R-uni / M-uni-early / M-multi-replay (plan §3.1 D10 已砍)

---

## 3. Per-cell multimodality audit

### 3.1 Audit per cell (3 个 audit, paired with E-uni)

每个 cell 与 E-uni 配对跑 `scripts/audit_multimodality.py`,共 **3 个 audit**:

| Audit ID | dataset_a (单峰参照) | dataset_b (本 cell) | 预期 |
|---|---|---|---|
| **A1** | E-uni | E-uni (self-check) | `p_≥2(E-uni) < 0.20` (sanity, single dataset call) |
| **A2** | E-uni | M-uni-noise | `p_≥2(M-uni-noise) < 0.20` (¬B test, 仍单峰); Δp(≥2) close to 0 |
| **A3** | E-uni | M-multi-mix | `p_≥2(M-multi-mix) > 0.30` (A∧B test, 双峰); Δp(≥2) > 0.10 |

**为什么不 audit M-uni-noise vs M-multi-mix**:audit 设计 anchored on E-uni,paper claim 也以 E-uni 为 reference;M-uni-noise vs M-multi-mix 不是 paper claim 直接 testable 维度。

### 3.2 Audit 命令模板 (A3 示例)

```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.audit_multimodality \
    --dataset-a offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000 \
    --dataset-b offline_data/fql_succession/m_multi_mix_50priv_50goal_1000 \
    --output-dir results/fql_succession/p2/audit/m_multi_mix_vs_e_uni \
    --label-a "E-uni" --label-b "M-multi-mix" \
    --seed 0
```

默认 args (knn_k=50, gmm_max_components=3, gmm_n_init=3, mode_weight_floor=0.10, n_anchor_states=500, n_bootstrap=1000) 已在 P0+P1 dryrun 上验证。

### 3.3 Per-cell verdict (Gate C.2 component)

| Cell | criteria | 通过条件 |
|---|---|---|
| **E-uni** | `p_≥2(E-uni)` | < 0.20 (sanity, 已在 P0+P1 dryrun 满足 0.188) |
| **M-uni-noise** | `p_≥2(M-uni-noise)` | < 0.20 (本设计要求:noise widening 不分裂) |
| **M-multi-mix** | (i) `p_≥2(M-multi-mix) > 0.30`; (ii) `Δp(≥2) CI lower > 0.10`; (iii) Welch p < 0.07 | 三条全过 (paper claim core evidence) |

### 3.4 Audit fail mitigation

| Cell fail | 处理 |
|---|---|
| E-uni p_≥2 > 0.20 | 不太可能 (P0+P1 dryrun 实测 0.188);若真发生,investigate dataset corruption |
| M-uni-noise p_≥2 > 0.20 | ε=0.5 实际造成 mode split → 降 ε 到 0.3 重新 collect (R2 mitigation) |
| M-multi-mix p_≥2 < 0.30 | 2-way mix multimodality 不足 → 升级 3-way (privileged + goalseek + worldcomp 或 crosscomp) per audit dryrun 配置 |

任一 mitigation 后 re-collect + re-audit,**不**进 run matrix 直到 audit pass。

---

## 4. Run matrix

### 4.1 主对照 12-run 设计 (可扩 18-run)

**Primary 矩阵**:3 cell × 2 algo × 2 seed = **12 run**

| Cell | FQL seeds | ReBRAC seeds | 子小计 |
|---|---|---|---:|
| E-uni | [42, 0] | [42, 0] | 4 |
| M-uni-noise | [42, 0] | [42, 0] | 4 |
| M-multi-mix | [42, 0] | [42, 0] | 4 |
| **Total (primary)** | | | **12** |

**Conditional extension** (在 primary 12 跑完后触发):

| Verdict 边缘条件 | Action | 增量 |
|---|---|---:|
| 任一 cell 两 seed paired diff 方向不一致(一正一负) | 加 seed 7 → 18 run | +6 run |
| 任一 cell Δ ∈ [+3pp, +5pp](positive 但未达 effect-size threshold) | 加 seed 7 | +6 run |
| 12 run 全部 verdict 明确(Δ ≥ 5pp 同向 / |Δ| < 3pp 双 cell)→ 不扩 | — | 0 |

**为什么 n_seeds=2 primary** (vs plan v1.2 §4.2 原值 5 / Gate B 已用 2 / 上限 7):

1. **L4 wallclock budget** — 12 run × (FQL ~50min + ReBRAC ~25min)/2 ≈ 7.5 h vs n=5 的 19h (n=2 −60% wallclock)。Colab session 单次 ~10h 内可一次跑完两个 cell
2. **Effect-size primary verdict 不依赖 large-n power** — paper claim 用 Δ ≥ 5pp 且两 seed 同方向作为 binary signal,n=2 即可判定 signal vs null;Welch p / Bonferroni 在 n=2 下信息量低(自由度 1,p<0.0167 几乎不达),改为 sensitivity 报告(§5.5)
3. **Gate B 已用 [42, 0]** — same seed pool,有 4-run 历史数据可直接 cross-reference;新 8 run + 历史 4 run = 12 row evidence
4. **Conditional extension preserves rigor** — n=2 verdict 边缘 → 自动扩 n=3,paper revision 仍可补到 n=5;不一次性 commit 28h wallclock
5. **Paper-defensible by effect size**:reviewer 关心 "is the effect real and large enough to matter",n=2 同方向 + |Δ| ≥ 5pp 是 strong indicative signal,而 n=5 + p<0.0167 是 corroborative refinement(revision 路径,非主声明)

**Seed pool** primary [42, 0] = Gate B Option B;extension [42, 0, 7] 加 seed 7(不重叠 broad val v2 + ReBRAC paper 1 的 [42-46])。

### 4.2 Per-run CLI templates

每 run = **两步**:(1) `train_offline`(写 checkpoints/) → (2) `evaluate_offline`(写 results/test/)。训练过程产物 (small files) **mirror** 到 `results/training_curves/`,Colab 跑完 _只回收 `results/` 树就能本地绘图_。

**Storage layout** (per cell × algo × seed):

```
checkpoints/fql_succession/p2/{cell_id}/{algo}_seed{S}/   ← train_offline --save-dir
  ├── agent_final.pt           ← 大,留 Drive
  ├── agent_best.pt / agent_latest.pt (optional)
  ├── trainer_state.json
  ├── train_log.jsonl          ← per-step loss
  ├── eval_log.csv             ← in-training eval curve (n_pts=20)
  └── train_config.txt

results/fql_succession/p2/{cell_id}/
  ├── test/
  │   └── {algo}_seed{S}.json  ← evaluate_offline --output-json (100-ep test)
  └── training_curves/{algo}_seed{S}/
      ├── train_log.jsonl      ← mirror copy
      ├── eval_log.csv         ← mirror copy
      ├── trainer_state.json   ← mirror copy
      └── train_config.txt     ← mirror copy
```

**ReBRAC training step** (Cell {C} × Seed {S}, `--skip-final-eval` 让 final eval 由独立 evaluate_offline 步骤负责):

```bash
python -m scripts.train_offline \
    --algo rebrac \
    --offline-data 'offline_data/fql_succession/{cell_dataset_path}' \
    --flow 'wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy' \
    --manifest 'benchmarks/single_u10_cross_tgt15_ep100.json' \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --total-steps 200000 \
    --batch-size 256 \
    --eval-every 10000 \
    --eval-episodes 100 \
    --actor-penalty-coef 4.0 \
    --critic-penalty-coef 2.0 \
    --critic-layernorm \
    --no-actor-layernorm \
    --skip-final-eval \
    --seed {S} \
    --save-dir 'checkpoints/fql_succession/p2/{cell_id}/rebrac_seed{S}' \
    --device cuda
```

**FQL training step** (Cell {C} × Seed {S}):

```bash
python -m scripts.train_offline \
    --algo fql \
    --offline-data 'offline_data/fql_succession/{cell_dataset_path}' \
    --flow 'wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy' \
    --manifest 'benchmarks/single_u10_cross_tgt15_ep100.json' \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --total-steps 200000 \
    --batch-size 256 \
    --eval-every 10000 \
    --eval-episodes 100 \
    --flow-steps 10 \
    --distill-alpha-bc 1.0 \
    --teacher-lr 3e-4 \
    --flow-time-embed-dim 32 \
    --skip-final-eval \
    --seed {S} \
    --save-dir 'checkpoints/fql_succession/p2/{cell_id}/fql_seed{S}' \
    --device cuda
```

**Final test eval step** (per algo × seed):

```bash
python -m scripts.evaluate_offline \
    --checkpoint 'checkpoints/fql_succession/p2/{cell_id}/{algo}_seed{S}' \
    --manifest 'benchmarks/single_u10_cross_tgt15_ep100.json' \
    --episodes 100 \
    --device cuda \
    --output-json 'results/fql_succession/p2/{cell_id}/test/{algo}_seed{S}.json'
```

**关键差异 vs Gate B Option B CLI**:
- `--manifest`:**`single_u10_cross_tgt15_ep100.json`** (D18, 取代 30-ep)
- `--seed`:2 个 seed [42, 0] primary(可扩 3 [42, 0, 7])
- `--save-dir`:`checkpoints/fql_succession/p2/{cell_id}/...` (per-cell namespace)
- `--skip-final-eval`:final eval 由独立 `evaluate_offline --output-json` 步骤写入 `results/test/`(分离 checkpoints / results 关注点)
- 其余 hyperparam 全部继承 Gate B Option B (per §1.2 表)

### 4.3 Skip-resume 协议

Gate B Option B 经验:rev.3.1 Colab session 经常被 idle timeout 杀掉,必须**显式 skip-resume**。

**Skip 判定 (per run = train + eval 两步)**:

| Stage | Skip 判定文件 | 含义 |
|---|---|---|
| **Train 已 done** | `checkpoints/.../{algo}_seed{S}/agent_final.pt` 存在 | 跳过 train step,直接做 mirror + evaluate |
| **整 run 已 done** | `results/.../test/{algo}_seed{S}.json` 存在 | 跳过 train + evaluate(整个 run) |
| **都不存在** | — | fresh run(train → mirror → evaluate) |

**Resume mid-training**:`trainer_state.json` 存在 + `train_step < total_steps` → `--resume <save-dir>`(train_offline 内部自动检测)。

**Per-run state files** (per Gate B observation):

| File | 位置 | 用途 |
|---|---|---|
| `agent_final.pt` | `checkpoints/.../<algo>_seed<S>/` | final weights(大,留 Drive) |
| `trainer_state.json` | `checkpoints/.../` + mirror to `results/training_curves/` | resume metadata + 绘图 config |
| `train_log.jsonl` | 同上 mirror | per-step loss(绘 loss curve) |
| `eval_log.csv` | 同上 mirror | in-training eval (dedup-by-train_step in verdict aggregation,**§5.1 primary scalar 来源**) |
| `train_config.txt` | 同上 mirror | human-readable config record |
| `{algo}_seed{S}.json` | `results/.../test/` | final canonical 100-ep test eval(`evaluate_offline --output-json` 写入) |

**Notebook cell 模板** (helper `run_one()` 见 §9.3.1)。详见 §9 notebook scaffold 设计。

### 4.4 Run 不做的事

- ❌ 不变 hyperparam between cells (`actor-penalty-coef` / `flow-steps` 等 frozen at Gate B Option B)
- ❌ 不混 sampling mode (uniform replay only, no shuffle_no_replacement)
- ❌ 不开 `--use-asymmetric-critic` (plan §3.5 vanilla critic; asym critic 是 follow-up,本 spec 不做)
- ❌ 不混 `--total-steps` (统一 200k)
- ❌ 不 enable `--num-epochs` (uniform replay, 不算 epoch)

---

## 5. 统计协议

### 5.1 Per-cell paired bootstrap CI

**Goal**:per cell 给出 `Δ_cell = mean(FQL_success_per_seed) − mean(ReBRAC_success_per_seed)` 的 95% CI,paired by seed (FQL seed=42 与 ReBRAC seed=42 算同一 paired sample)。

**算法**:

```python
def paired_bootstrap_delta(
    fql_per_seed: np.ndarray,        # [n_seeds] (2 primary, 可扩 3)
    rebrac_per_seed: np.ndarray,     # [n_seeds] (2 primary, 可扩 3)
    n_boot: int = 10000,
    rng_seed: int = 0,
) -> dict:
    """Paired bootstrap on per-seed paired difference."""
    rng = np.random.default_rng(rng_seed)
    n = len(fql_per_seed)
    assert len(rebrac_per_seed) == n
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.mean(fql_per_seed[idx] - rebrac_per_seed[idx])
    return {
        "delta_mean": float(np.mean(fql_per_seed - rebrac_per_seed)),  # 原始 paired mean
        "delta_boot_mean": float(np.mean(deltas)),
        "delta_ci_2p5": float(np.percentile(deltas, 2.5)),
        "delta_ci_97p5": float(np.percentile(deltas, 97.5)),
        "delta_boot_se": float(np.std(deltas, ddof=1)),
    }
```

**Per-seed scalar (input)**: last-3 mean from `eval_log.csv` dedup-by-train_step (per Gate B convention)。**不**用 final test_result (Gate B 经验:final 30-ep eval noise vs last-3 mean 给出 15pp vs 4.4pp gap,last-3 更稳定)。

### 5.2 Per-cell Welch's t one-sided

**H1**:per cell `mean(FQL) > mean(ReBRAC)`。报 t-statistic + one-sided p-value:

```python
from scipy import stats
t, p = stats.ttest_ind(fql_per_seed, rebrac_per_seed, equal_var=False, alternative="greater")
```

### 5.3 Per-cell Cohen's d (effect size)

```python
def cohens_d(a, b):
    pooled_std = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0
```

Per cell 报 `d = (mean(FQL) - mean(ReBRAC)) / pooled_std`。

### 5.4 iff verdict (effect-size + 方向一致性 primary)

**v1.2 决策**:n=2 primary 下 Welch p / Bonferroni 自由度过低(df=1),改为 **effect-size + 方向一致性** 作 primary 判据;Welch p / Bonferroni 留作 sensitivity (§5.5)。

**Effect-size threshold**:
- `δ_null = 0.03`(3pp,定义 null band:|Δ| < 0.03 视作 no effect)
- `δ_signal = 0.05`(5pp,定义 signal threshold:Δ ≥ 0.05 + 方向一致视作 effect 确认)
- **gray band [3pp, 5pp]** 或方向不一致 → **加 seed 7**(扩 n_seeds=3)再判

**Per-cell primary verdict**(n=2 [42, 0]):

| Cell | Null 条件 (paper claim 想要的) | Positive 条件 | Gray (加 seed 7) |
|---|---|---|---|
| **E-uni** | `\|Δ_E-uni\| < 0.03` AND 两 seed paired diff 同号或都接近 0 (\|diff_seed\| < 0.04 both) | `Δ_E-uni ≥ 0.05` 同方向 (FQL > ReBRAC,意外 positive) | 中间 |
| **M-uni-noise** | `\|Δ_M-uni-noise\| < 0.03` AND 两 seed 方向相容 | `Δ_M-uni-noise ≥ 0.05` 同方向 (意外 positive) | 中间 |
| **M-multi-mix** | `Δ_M-multi-mix < 0.03` 或两 seed 方向反 → null | `Δ_M-multi-mix ≥ 0.05` AND 两 seed paired diff 均 > 0 (方向一致) | 中间 |

**完整 iff 主判**:E-uni null ∧ M-uni-noise null ∧ M-multi-mix positive → 进 P3。

**Welch p / Bonferroni 作 sensitivity 报告**(§5.5),不作 primary binary input。

### 5.6 c4 monitoring (Option α gate, P2 default config)

承接 Gate B Option B caveat 与 D19 c4 阈值改革:

**c4 定义** (per algo × cell):
- per-seed slope = last-30% linear regression slope of in-training eval success rate (n_pts = 6 per seed,因为 200k steps + eval-every 10000 = 20 eval pts,last 30% = 6 pts)
- aggregated slope = mean over n_seeds
- threshold = `−2 × SE(slope_aggregated_n_seeds)`
- SE_per_seed = √(p(1−p)/n_eval) / √17.5 (n_eval = 100 from ep100 manifest, n_pts=6 → denom=17.5)
- SE_aggregated = SE_per_seed / √n_seeds

**P2 default 数值** (n=2 primary, per c4 revision doc §3 retroactive table):

| Quantity | n=2 primary | n=3 conditional ext. |
|---|---:|---:|
| n_eval (manifest size) | 100 | 100 |
| p (typical in-training mean success) | ≈ 0.7 | ≈ 0.7 |
| SE_eval @ p=0.7 | 0.0458 | 0.0458 |
| SE_per_seed (n_pts=6) | 0.0110 | 0.0110 |
| SE_aggregated | **0.00775** | **0.00633** |
| **c4 threshold** | **−0.01549** | **−0.01267** |

**c4 verdict (per algo × cell)**:
- aggregated slope ≥ threshold → c4 PASS
- aggregated slope < threshold → c4 FAIL (真有 late-training collapse)
- **n=2 阈值 −0.0155 比 n=5 −0.0098 更宽容 1.6×**:n=2 下 noise floor 大,Option α 自动 scale,Gate B FQL aggregated slope −0.0038 仍远高于 n=2 阈值 → retroactive PASS confirmed

**c4 在 P2 是 gate 还是 monitor**:
- **Gate** (per Session B Option α design): c4 阈值是 statistical-meaningful (95% 单侧 CI 下界);不会被噪声触发 (Gate B 验证)
- Per cell × algo × per algo gap 都有自己的 c4 verdict
- 若 **FQL 在任一 cell c4 FAIL** → flag in §6 Gate C.3 verdict,paper claim 写作时 hedge "FQL stability marginally fails in cell X"
- ReBRAC c4 FAIL 与 paper claim 无直接关系 (paper 不 claim ReBRAC stability),仅记录到 verdict report appendix

### 5.7 Per-run metric set (reference)

每 run 落地的可计算指标(从 `results/.../training_curves/<algo>_seed<S>/{eval_log.csv, train_log.jsonl}` + `results/.../test/<algo>_seed<S>.json` 提取):

| Metric | Source | Role |
|---|---|---|
| `last_3_mean` | `eval_log.csv` last 3 train_step rows (dedup) | **§5.1-5.4 primary scalar** |
| `final_test_success` | `results/.../test/<algo>_seed<S>.json` `eval_success_rate` | Secondary (Gate B 经验:more noise) |
| `last_30pct_slope` | linregress(last 6 eval pts in `eval_log.csv`) | **§5.6 c4 monitoring** |
| `final_test_return` / `safety` / `progress` / `path_eff` | `results/.../test/<algo>_seed<S>.json` | Reported in §results table appendix |
| `loss_flow_tail` | `train_log.jsonl` last 5% | FQL teacher health (per Gate B c2) |
| `loss_actor_tail` | `train_log.jsonl` last 5% | Per-algo comparison parity check |
| `q_mean_tail` | `train_log.jsonl` last 5% | Q stability monitor |

---

## 6. Gate C 判据 (P2 出口)

### 6.1 Gate C.1 — Completeness

12-run primary(或 18-run 扩展)全部完成且无系统性 error:
- 每 run 同时落地两棵树:
  - **`checkpoints/.../{cell}/{algo}_seed{S}/`**:`agent_final.pt` + `trainer_state.json` + `train_log.jsonl` + `eval_log.csv` + `train_config.txt`
  - **`results/.../{cell}/`**:`test/{algo}_seed{S}.json` (final 100-ep test) + `training_curves/{algo}_seed{S}/` (前述 4 small files 的 mirror copy)
- `eval_log.csv` ≥18 unique-step rows post-dedup(允许 ≤2 rows 缺失 from rare eval failures)
- 0 run NaN / Inf loss in `train_log.jsonl` 末段
- 0 run 因 OOM / shape mismatch / argparse error crashed

### 6.2 Gate C.2 — Audit

§3 三个 per-cell audit 全部 verdict 正确:
- A1 (E-uni self-check): p_≥2 < 0.20
- A2 (M-uni-noise vs E-uni): M-uni-noise p_≥2 < 0.20 + Δp(≥2) close to 0 (no significant 区分)
- A3 (M-multi-mix vs E-uni): M-multi-mix p_≥2 > 0.30 + Δp(≥2) CI lower > 0.10 + Welch p < 0.07

任一 cell A1/A2/A3 fail → **不进** statistical verdict (重收集 + re-audit)。

### 6.3 Gate C.3 — iff verdict 三选一

按 §5.4 effect-size primary + §5.6 c4 + §6.1/§6.2 同时 satisfied,以下三选一(`§5.5` sensitivity 表必须同时报告,但 verdict primary 不依赖 Welch p):

| Verdict | Condition (effect-size primary, n=2 [42, 0] or extended n=3) | Paper claim | P3 trigger |
|---|---|---|---|
| **完整 iff hold** | E-uni null (\|Δ\|<0.03, 两 seed 方向相容) ∧ M-uni-noise null (\|Δ\|<0.03, 方向相容) ∧ M-multi-mix positive (Δ≥0.05 AND 两 seed paired diff > 0 同号) | 主路径 conditional iff (plan §1.2) | ✓ 进 P3 mix ratio sweep |
| **Partial iff** | E-uni null ∧ M-multi-mix positive ∧ **M-uni-noise also positive (Δ≥0.05 同号)** | 降级:"FQL > ReBRAC on sub-optimal data regardless of modality" | ✓ 进 P3,但 modality intensity 单调性变 secondary evidence |
| **Null iff** | M-multi-mix not positive (Δ<0.05 或 两 seed 方向反) | 重写为 scoped negative finding:"expressive prior 在本 AUV navigation task 无 leverage" (R3 mitigation) | P3 决策点:(a) 仍跑 P3 验证 negative finding robustness; (b) skip P3 + 直接进 P4 negative paper writing |
| **Gray (扩 seed 7)** | 任一 cell 落 gray band [3pp, 5pp] 或方向不一致 | — | 先扩 n_seeds=3 跑 +6 run,然后重 evaluate |

**E-uni 上 FQL 倒挂 (Δ_E-uni < −0.05 AND 两 seed 同方向 < 0)** 是 catastrophic case:
- 与 Gate B finding (FQL on E-uni Δ=−4.4pp) 一致,marginal — 不一定 catastrophic
- 但若 P2 实测 \|Δ_E-uni\| > 0.08 (超出 Gate B 阈值),触发 R4 mitigation:debug FQL hyperparam,debug 3 iter 仍倒挂则 STOP

### 6.4 Gate C.3 verdict report 产出

`docs/fql_succession_p2_main_report.md` 应包含:
- §1 Run matrix completeness summary (Gate C.1)
- §2 Per-cell audit results table (Gate C.2)
- §3 Per-cell statistical table:Δ + paired CI + Welch p + Cohen d + c4 slope (5 metric per cell, 3 cells = 15 cells in table)
- §4 iff verdict declaration
- §5 c4 monitoring summary (per cell × algo)
- §6 Carveouts / caveats (carry forward Gate B's caveat + 新发现的)
- §7 P3 决策 (per §6.3 verdict)

---

## 7. Risk + mitigation

引用 plan §5 R1-R9 (本 spec 不重复定义),加 P2-specific 风险:

| ID | 风险 (P2-specific) | 概率 | 影响 | Mitigation |
|---|---|---|---|---|
| **P2.R1** | M-uni-noise (ε=0.5) audit fail (p_≥2 > 0.20, 已分裂为多峰) | 中 | 中 | 降 ε 到 0.3 重 collect;若仍 fail,M-uni-noise cell 整体重设计 (e.g. add gaussian only to heading dim instead of full action) — 最多 +1 周 |
| **P2.R2** | M-multi-mix 2-way audit fail (p_≥2 < 0.30) | 中 | 中 | 升级 3-way mix (privileged + goalseek + worldcomp/crosscomp per dryrun config) — concat 重组,~10 min;成本归零 |
| **P2.R3** | n_seeds=2 primary 下 verdict 落 gray band (Δ ∈ [3pp, 5pp] 或方向不一致) | 中 | 中 | conditional extension 加 seed 7 → n=3 (+6 run,~3-4h L4);若仍 gray 才进 revision 补到 n=5;**本 sprint 不 preemptive 跑 n=3** |
| **P2.R4** | FQL 在 E-uni 上 Δ > −0.08 但 < −0.05 (marginal underperform) | 中 (Gate B 实测 −4.4pp 已逼近) | 中 | 不阻塞 iff null verdict (|Δ|<0.05 严格,−0.044 已逼近边界);P4 写 hedge "FQL marginally underperforms ReBRAC on expert-uni regime" |
| **P2.R5** | c4 在 P2 multi-cell ramp 后又出现 seed-driven 不一致 | 中 | 低 | Option α 阈值已 statistical-meaningful;若仍 cell-level 不一致 (e.g. FQL c4 PASS in M-multi-mix 但 FAIL in E-uni),记入 verdict report,**不**影响 iff binary verdict |
| **P2.R6** | Effect-size primary verdict 在 n=2 下与 Welch p / Bonferroni sensitivity 给出不同方向(e.g. Δ=4.5pp 同号 = gray,但 Welch p=0.04 = signal) | 中 | 中 | primary 用 effect-size + 方向 (§5.4) 决定 binary verdict;Welch / Bonferroni sensitivity 报告在 §6.4 verdict report appendix;paper §method 显式声明 effect-size + 方向是 primary,p 值是 sensitivity 检验 |
| **P2.R7** | M-uni-noise collection success rate 太低 (<0.50),数据集 dominate by failure traj | 低 | 中 | ε=0.5 noise → 60-70% success expected;若实测 <0.50,降 ε 重 collect |
| **P2.R8** | Skip-resume 协议 在多 cell × seed runs 下出现 trainer_state corruption | 低 | 高 | Per Gate B Bug 1 经验,dedup_by_train_step 已 frozen;每 run 独立 save-dir 避免 cross-contamination |
| **P2.R9** | L4 wallclock 超预算 (Colab session 限制 ~12h) | 低 | 中 | per-cell notebook 设计支持 skip-resume,跨 session 接续;最坏需要 3-4 个 Colab session × 12h split (§8 算力预算) |

---

## 8. 算力预算

### 8.1 Collection (本机, 6 worker CPU)

| Cell | Estimated wallclock | 备注 |
|---|---|---|
| E-uni | 0 (reuse) | Task D 已 collected |
| M-uni-noise (1000 ep × ε=0.5) | ~4 min | scale linearly from E-uni 2.6 min (more failures = longer episodes ~ 1.5×) |
| M-multi-mix Source A (privileged-500) | ~1.5 min | half of E-uni 2.6 min |
| M-multi-mix Source B (goalseek-500) | ~2.5 min | goalseek 较慢 (per Gate B observation) |
| concat M-multi-mix | <1 min | shell one-liner |
| Per-cell audit (×3) | ~6 s × 3 = ~20 s | per dryrun report runtime |
| **Total collection** | **~10 min CPU wallclock** | one local session |

### 8.2 Training (Colab L4)

| Item | Per run | Primary 12 run total | Extended 18 run total |
|---|---:|---:|---:|
| ReBRAC 200k step (Gate B observed) | ~25 min | 2 seed × 3 cell = 6 run → ~2.5 h | 3 seed × 3 cell = 9 run → ~3.75 h |
| FQL 200k step (Gate B observed) | ~50 min | 2 seed × 3 cell = 6 run → ~5 h | 3 seed × 3 cell = 9 run → ~7.5 h |
| evaluate_offline 100-ep (per run) | ~2 min | 12 × 2 min = ~24 min | 18 × 2 min = ~36 min |
| Verdict aggregation + audit per cell run | trivial (~10 min) | local | local |
| **Total training wallclock** | | **~8 h L4 sequential** | **~12 h L4 sequential** |

Parallel notebook strategy: per-cell notebook (3 notebooks) → 单 Colab session ~10h 可一次跑完 2 个 cell;3 个 cell 分 2 个 session 即可(不需要 multi-account)。

### 8.3 Sprint plan (estimated, 2026-05-21 起算)

| Day | Task |
|---|---|
| Day 1 (本机) | M-uni-noise + M-multi-mix Source A/B collection + concat + 3 audit + dataset cards 起草(non-notebook, sprint 0) |
| Day 2 (本机) | Commit collection + audit log;3 run notebook scaffold finalize |
| Day 3 (Colab) | E-uni + M-uni-noise cells × 2 algo × 2 seed = 8 run (~6h, 1 session) |
| Day 4 (Colab) | M-multi-mix cell × 4 run + 若任一 cell gray band 触发 → 加 seed 7 扩 18-run (+~4h) |
| Day 5 (本机) | Verdict report (§5 + §6) + paper-2 outline finalize |

**Total wallclock**:~5 天(vs plan §4.2 estimate 1.5 周)— **-50%** ahead of plan,n=2 节省 wallclock 让出余量给 revision 阶段扩 seed。

### 8.4 算力 contingency

| Trigger | Impact |
|---|---|
| Colab L4 quota burn | 切换到 Drive sync + 本机 GPU (若有);最多 +3 天 |
| Skip-resume failure | per-run independent save-dir 防 cross-run contamination,最多 affected run 重跑 (~1h) |
| Audit per-cell fail (R1/R2) | re-collect + re-audit,+0.5-1 天 |

---

## 9. Notebook scaffold 计划

### 9.1 Notebook 划分原则

**Per-cell run notebook**(3 个 ipynb)+ **本机 collection 非 notebook**(v1.2 决策):
- Collection 一次性脚本工作流(`collect_offline_data` + `concat_offline_datasets` + 3 个 audit script call)→ 本机 bash + python 直接做,不写 notebook;产物含 dataset card + collection log
- Per-cell run notebook 优势:skip-resume 粒度细,某 cell 失败不影响其他 cell;Colab session 12h limit 可 per-cell session;debugging 路径清晰
- Notebook 风格仿 `notebooks/rebrac_c1_asym_critic_ablation.ipynb`(8-section:前传 + verdict 表 + 隔离表 / sanity / config / pre-flight / train + mirror / eval / 对比 / verdict / 报告写入),**不**仿近期 fql_succession gate b rev.3.x(反复修改不干净)

### 9.2 Notebook 清单

| Notebook | 在哪跑 | 内容 | 状态 |
|---|---|---|---|
| **本机 collection**(非 notebook) | **本机** (mytorch1 conda env) | M-uni-noise + privileged-500 + goalseek-500 collect + concat M-multi-mix-1000 + 3 audit (A1/A2/A3) + dataset card + `docs/fql_succession_p2_collection_log.md` | 本 spec sprint 0 任务 4b 执行 |
| `notebooks/fql_succession_p2_run_cell_e_uni.ipynb` | **Colab L4** | E-uni cell × FQL + ReBRAC × 2 seeds [42, 0] = 4 run | 任务 4c |
| `notebooks/fql_succession_p2_run_cell_m_uni_noise.ipynb` | **Colab L4** | M-uni-noise × 4 run | 任务 4c |
| `notebooks/fql_succession_p2_run_cell_m_multi_mix.ipynb` | **Colab L4** | M-multi-mix × 4 run | 任务 4c |
| `notebooks/fql_succession_p2_verdict.ipynb` | 本机 | aggregate 12-run results 树 + paired diff + effect-size verdict + Welch / Bonferroni sensitivity | 待 P2 闭环时另起 |

### 9.3 Per-cell run notebook 必须 cell (仿 rebrac_c1_asym_critic_ablation 8-section 风格)

每 notebook 8 节,每节 1 段 markdown + 1-2 code cell。`!python` magic 实时 stream(Bug 4 mitigation);所有路径单引号(Bug 5)。

| § | 内容 | Idempotent? |
|---|---|---|
| **§0 前传 + verdict 规则**(markdown) | Gate B 已知事实(2-seed Δ_E-uni=-4.4pp marginal)+ 本 notebook 假设 + 隔离表(vs 另两 cell)+ verdict 规则表(effect-size primary,从 §5.4 拷贝) | — |
| **§1 环境 sanity**(code) | `!nvidia-smi` + torch CUDA check + `from google.colab import drive; drive.mount(...)` + `%cd /content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5` | ✓ |
| **§2 通用配置**(code) | 定义 `CELL_ID` / `DATASET` / `FLOW` / `MANIFEST` / `CHECKPOINT_ROOT` / `RESULTS_ROOT` / `REBRAC_FLAGS` / `FQL_FLAGS` / `TRAIN_SEEDS=[42, 0]` | ✓ |
| **§3 Pre-flight**(code) | `assert (DATASET_DIR / 'transitions.npz').exists()` + npz keys + privileged_obs dim check(本 spec 用 vanilla critic 不需要 priv,但 audit 时验证 dataset 完整性) | ✓ |
| **§4 训练**(code) | `for algo in ['rebrac', 'fql']: for seed in TRAIN_SEEDS: run_one(algo, seed)` ← 内含 train + mirror + evaluate 三步(§9.3.1) | ✓ (helper 内 skip-resume check) |
| **§5 三向对比**(code) | 读取 `results/.../test/{algo}_seed{S}.json` × 4 → DataFrame:per-seed test success + termination 分布 + last-3 mean(从 `results/.../training_curves/.../eval_log.csv`) | ✓ |
| **§6 自动 verdict**(code) | per-cell `delta` + 两 seed paired diff + effect-size verdict (null/positive/gray) + c4 slope check | ✓ |
| **§7 报告写入清单**(markdown) | per-verdict 的下一步(更新 plan / spec / verdict report 哪些段;若 gray → 触发扩 seed 7 的 notebook 入口) | — |

### 9.3.1 `run_one()` helper 模板 (定义在 §2 通用配置 + §4 训练 cell)

```python
import os, shutil
from pathlib import Path

# ---- per-notebook 改这 3 个 ----
CELL_ID  = "e_uni"   # or "m_uni_noise" / "m_multi_mix"
DATASET  = "offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000"
# ↑ m_uni_noise:  "offline_data/fql_succession/m_uni_noise_eps0p5_1000"
# ↑ m_multi_mix:  "offline_data/fql_succession/m_multi_mix_50priv_50goal_1000"

# ---- 共享(3 个 notebook 一致)----
FLOW     = "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
MANIFEST = "benchmarks/single_u10_cross_tgt15_ep100.json"

CHECKPOINT_ROOT = Path(f"checkpoints/fql_succession/p2/{CELL_ID}")
RESULTS_ROOT    = Path(f"results/fql_succession/p2/{CELL_ID}")

REBRAC_FLAGS = (
    "--actor-penalty-coef 4.0 --critic-penalty-coef 2.0 "
    "--critic-layernorm --no-actor-layernorm"
)
FQL_FLAGS = (
    "--flow-steps 10 --distill-alpha-bc 1.0 "
    "--teacher-lr 3e-4 --flow-time-embed-dim 32"
)

# Small files to mirror checkpoints/ → results/training_curves/
MIRROR_FILES = ("train_log.jsonl", "eval_log.csv", "trainer_state.json", "train_config.txt")

TRAIN_SEEDS = [42, 0]   # primary;扩 [42, 0, 7] 时改这里

def _save_dir(algo, seed):
    return CHECKPOINT_ROOT / f"{algo}_seed{seed}"

def _test_json(algo, seed):
    return RESULTS_ROOT / "test" / f"{algo}_seed{seed}.json"

def _mirror_dir(algo, seed):
    return RESULTS_ROOT / "training_curves" / f"{algo}_seed{seed}"

def _mirror_curves(algo, seed):
    src_dir = _save_dir(algo, seed)
    dst_dir = _mirror_dir(algo, seed)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for fname in MIRROR_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            n_copied += 1
    print(f"[mirror] {algo} seed={seed}: {n_copied}/{len(MIRROR_FILES)} files → {dst_dir}")

def run_one(algo: str, seed: int) -> None:
    """Train + mirror small files + final evaluate. Three-stage skip-resume."""
    save_dir = _save_dir(algo, seed)
    test_json = _test_json(algo, seed)

    # Stage 0 — 整 run 已 done(results 测试 json 存在)
    if test_json.exists():
        print(f"[skip-full] {algo} seed={seed} (test json exists: {test_json})")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    algo_flags = REBRAC_FLAGS if algo == "rebrac" else FQL_FLAGS

    # Stage 1 — train (skip if agent_final.pt exists)
    if not (save_dir / "agent_final.pt").exists():
        print(f"\n========== train {algo} seed={seed} → {save_dir} ==========")
        train_cmd = (
            f"python -m scripts.train_offline "
            f"--algo {algo} "
            f"--offline-data '{DATASET}' "
            f"--flow '{FLOW}' "
            f"--manifest '{MANIFEST}' "
            f"--probe-layout s0 --history-length 4 --target-speed 1.5 "
            f"--task-geometry cross_stream --objective arrival_v2 "
            f"--total-steps 200000 --batch-size 256 "
            f"--eval-every 10000 --eval-episodes 100 "
            f"{algo_flags} "
            f"--skip-final-eval "
            f"--seed {seed} "
            f"--save-dir '{save_dir}' "
            f"--device cuda"
        )
        get_ipython().system(train_cmd)
    else:
        print(f"[skip-train] {algo} seed={seed} (agent_final.pt exists)")

    # Stage 2 — mirror training curves (overwrite OK, idempotent copy)
    _mirror_curves(algo, seed)

    # Stage 3 — final test eval → results/test/<algo>_seed<seed>.json
    test_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n========== eval {algo} seed={seed} → {test_json} ==========")
    eval_cmd = (
        f"python -m scripts.evaluate_offline "
        f"--checkpoint '{save_dir}' "
        f"--manifest '{MANIFEST}' "
        f"--episodes 100 "
        f"--device cuda "
        f"--output-json '{test_json}'"
    )
    get_ipython().system(eval_cmd)
```

### 9.3.2 §5 三向对比 + §6 verdict cell guard

§5/§6 cell 应 guard 在 `_test_json()` × 4 全部存在后再 compute(prevent partial data):

```python
ALGOS = ["rebrac", "fql"]

def all_done() -> bool:
    return all(_test_json(a, s).exists() for a in ALGOS for s in TRAIN_SEEDS)

if not all_done():
    pending = [
        f"{a}_seed{s}" for a in ALGOS for s in TRAIN_SEEDS
        if not _test_json(a, s).exists()
    ]
    raise RuntimeError(f"[not ready] {len(pending)} runs pending: {pending}")

# ... §5 读取 results/.../test/*.json + results/.../training_curves/*/eval_log.csv
#     → DataFrame → §6 effect-size verdict
```

### 9.4 状态契约(collection + run + verdict)

- **本机 collection**(sprint 0)产出:
  - `offline_data/fql_succession/m_uni_noise_eps0p5_1000/{transitions.npz, metadata.json}`
  - `offline_data/fql_succession/m_multi_mix_50priv_50goal_1000/{transitions.npz, metadata.json}`
  - Audit results 写到 `results/fql_succession/p2/audit/{A1, A2, A3}_*.json` + 文本 log 到 `docs/fql_succession_p2_collection_log.md`
  - Dataset card md 草稿(non-blocking)
- **Run notebook**(任务 4c)消费 dataset path,产出:
  - `checkpoints/fql_succession/p2/{cell_id}/{algo}_seed{S}/` (Drive-mounted, 大文件)
  - `results/fql_succession/p2/{cell_id}/{test, training_curves}/` (Drive-mounted, 小文件, 本机绘图用)
- **Verdict notebook**(P2 闭环时另起)消费 `results/` 全树,产出 `docs/fql_succession_p2_main_report.md`

---

## 10. Caveat 段

### 10.1 c4 marginal-FAIL carryforward (from Gate B)

Gate B Option B 实测 c4 marginal-FAIL (aggregated slope −0.0038,30-ep manifest noise floor 内)。原因 seed-driven (seed=0 slope 正,seed=42 slope 负,两 algo 同方向)。P2 用 Option α 阈值 (D19) + ep100 manifest (D18) + n_seeds=2 primary,redo c4 verdict per cell × algo。

**预期**:在 ep100 manifest + n=2 下 SE_agg ≈ 0.00775,Option α threshold ≈ −0.0155;Gate B FQL aggregated slope −0.0038 远高于阈值。retroactive PASS 已在 c4 revision doc §3 验证 (n=2, 100-ep 配置下 threshold = −0.0155,FQL slope PASS)。P2 mainline 配置下 c4 应稳定 PASS。

**Paper writing implication**:§experiments 段 footnote 提及 "Gate B 2-seed validation found marginal c4 fail (slope −0.0038), statistically indistinguishable from 0 under 30-episode evaluation noise. P2 evaluation upgrades to 100-episode manifest tightening the noise floor 1.8×; the auto-scaled Option α threshold restores c4 PASS verdict."

### 10.2 Noise floor 更新 (Bug 2 fix 后)

直接复用 [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) §3.1:

| 量 | 30-ep manifest (Gate B) | 100-ep manifest (P2 default) | 改善 |
|---|---:|---:|---:|
| per-eval SE @ p=0.7 | 0.0837 | 0.0458 | 1.83× |
| slope SE per-seed (n_pts=6) | 0.0200 | 0.0110 | 1.83× |
| slope SE aggregated n_seeds=2 | 0.01414 | 0.00775 | 1.83× |
| c4 threshold (Option α) at n=2 | −0.02828 | −0.01549 | (auto-scale) |
| slope SE aggregated n_seeds=3 (扩展) | 0.01155 | 0.00633 | 1.83× |
| c4 threshold (Option α) at n=3 | −0.02309 | −0.01267 | (auto-scale) |

**Per-cell × algo c4 slope 预期范围** (under healthy training):
- |aggregated slope| ≤ 2 × SE_agg ≈ 0.01 (in noise band)
- 出格意味着 真有 trend (positive = late improvement; negative = collapse)

### 10.3 n_seeds 决策 (2 primary vs 3 conditional ext.) — v1.2 重定

**v1.2 重定理由**:wallclock budget(L4 ~10h/session)+ effect-size primary verdict 让 large-n statistical power 不再是 paper claim 的瓶颈。

Gate B 暴露的 effect-size benchmark:

| Quantity | Value |
|---|---:|
| FQL vs ReBRAC last-3 gap @ seed=0 | −14.4pp |
| FQL vs ReBRAC last-3 gap @ seed=42 | +5.6pp |
| 两 seed 方向 | **不一致**(catastrophic spread Gate B 已揭示) |
| Pooled std (2-seed estimate) | ~14pp |

在 plan claim 想 detect 的 effect size (Δ_M-multi-mix ≥ 0.05) 下,**v1.2 effect-size + 方向一致性 primary**:

| n_seeds | wallclock (L4) | Effect-size detectability | Welch p detectability @ α=0.05 | Bonferroni p<0.0167 detectability |
|---:|---:|---|---|---|
| **n=2 primary** | ~8h | Δ≥5pp + 两 seed 同方向 = strong indicative signal | df=1, p<0.05 需 Δ ≥ 0.07 in same direction | df=1, p<0.0167 需 Δ ≥ 0.10 in same direction (rare) |
| **n=3 (扩 seed 7)** | ~12h | Δ≥5pp + 三 seed paired diff > 0 majority | df=2, p<0.05 需 Δ ≥ 0.05 + 3 seed 同向 | df=2, p<0.0167 仍困难 |
| n=5 (revision 阶段) | ~19h | — | df=4, p<0.0167 可达 70-80% power | full coverage |

**判断**:
1. **Paper claim 的 binary verdict 不依赖 Bonferroni p**:reviewer 关心 "effect 是否 real and large enough",effect-size + 方向一致性已 strong evidence
2. **Welch p / Bonferroni 留 sensitivity** (§5.5),paper §method 显式声明
3. **n=2 verdict 边缘 → 扩 n=3** (+~4h L4),仍是单次 Colab session 可完成
4. **若 P2 + n=3 仍 marginal**,revision 阶段补到 n=5 (+~7h Colab)

**采用 n=2 primary [42, 0]**(Gate B seed pool),conditional extension `[42, 0, 7]`,revision-stage option `[42, 0, 7, X, Y]` 待 P2 闭环再定。

### 10.4 Per-cell algo hyperparam frozen at Gate B Option B

P2 不重新 tune ReBRAC β1/β2 (frozen at 4/2 per broad val v2),不重新 tune FQL flow-steps/distill-alpha-bc (frozen at 10/1.0 per Gate B Option B)。这意味着 P2 在 **algo hyperparam 上不做 robustness check**;若 reviewer 关心 hyperparam sensitivity,P3 + revision 补 sweep。

### 10.5 P2 不做的事 (recap)

- ❌ M-uni-early cell (plan v1 §3.1 D10 砍)
- ❌ M-multi-replay cell (plan v1 §3.1 D10 砍)
- ❌ R-uni cell (同)
- ❌ Asym critic (plan §3.5 vanilla critic;follow-up)
- ❌ Hyperparam sensitivity (本 spec frozen,P3 视情况补)
- ❌ TD3+BC 重做 spectrum (D6 two-paper sequence,不重复 paper 1)
- ❌ Cross-modality decomposition (mechanism ablation 4→1,plan D13)

---

*Document version: v1.2 (2026-05-21, wallclock-budget + storage-layout 重设). 维护策略:Sprint 0 (collection) 完成后升级 v1.3 + dataset card 填写;每 cell run notebook 闭环后升级 v1.x;12-run primary 全闭环 + verdict report 完成后升级 v2.0(若进 n=3 扩展则先 v1.4)并 trigger plan v1.3 → v2.0。*
