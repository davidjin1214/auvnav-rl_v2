# FQL Succession — P2 Main Comparison Spec

> **文档版本**：v1.1（2026-05-20，Session A 起草 + 4-decision patches）
> **作用**：把 [`fql_succession_plan_v0.md`](fql_succession_plan_v0.md) §4.2 P2 main comparison（3-cell × 2-algo × 5-seed = 30 runs）拆成可执行的 collection / audit / run / verdict 协议。
> **状态**：**Active spec**，pre-requisites 全部落地，等待 P2 sprint 0 (collection) 启动信号。
>
> **版本历史**:
> - v1.0 (2026-05-20 起草) — 10 节初稿:scope + collection + audit + run matrix + statistics + Gate C + risks + budget + notebooks + caveats
> - **v1.1 (2026-05-20 patches)** — Session A 4-decision refinements:(a) §2.3 M-uni-noise success 4-band contingency; (b) §2.4 M-multi-mix 2-way → 3-way conditional upgrade rule + P3 implication; (c) §5.5 Bonferroni primary + BH/uncorrected sensitivity table; (d) §9.3 `run_one()` helper 模板 + idempotent `summarize`/`verdict_preview` guards
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

> 在 3-cell spectrum (E-uni / M-uni-noise / M-multi-mix) × 2 algorithm (FQL, ReBRAC) × 5 seeds = **30 runs** 上系统对比，量化 paper 2 的核心 conditional iff claim："FQL 的 expressive behavior prior translates to policy improvement over ReBRAC **iff** data is **both** sub-optimal **and** multi-modal"。

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
- **Mitigation activated (D17)**:按 P0+P1 spec §5.3 "mixed → caveat + 3-seed extension" 路径,P2 用 n_seeds=5 retire-or-confirm 此 marginal finding;**不** trigger Option D (FQL stability ablation)

P2 spec 在以下章节明确承接此 caveat:
- §4.1 n_seeds=5 (vs Gate B 的 2-seed)
- §4.2 manifest 切到 ep100 (D18)
- §5.6 c4 阈值用 Option α (D19), threshold ≈ −0.0098 at P2 default config
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

### 4.1 主对照 30-run 设计

**矩阵**:3 cell × 2 algo × 5 seed = **30 run**

| Cell | FQL seeds | ReBRAC seeds | 子小计 |
|---|---|---|---:|
| E-uni | [42, 43, 44, 45, 46] | [42, 43, 44, 45, 46] | 10 |
| M-uni-noise | [42, 43, 44, 45, 46] | [42, 43, 44, 45, 46] | 10 |
| M-multi-mix | [42, 43, 44, 45, 46] | [42, 43, 44, 45, 46] | 10 |
| **Total** | | | **30** |

**为什么 n_seeds=5** (vs Gate B 的 2 / plan 原值 5 / 候选 7):
1. **Gate B 暴露的 seed spread 已被 plan §4.2 n_seeds=5 default 覆盖** — 5 seed × paired bootstrap CI 在 SE_agg ≈ 0.005 level 已足够区分 ≥5pp effect size
2. **Bonferroni 多重比较 over 3 cells 在 n=5 下 critical p ≈ 0.017,仍可达**
3. **L4 wallclock budget** — 30 run × ~50min FQL or ~25min ReBRAC ≈ 1100 min ≈ 18-20h wallclock vs n=7 的 28h (+40% 边际收益不大)
4. **Paper-typical n_seeds 标准** — D4RL benchmark + ReBRAC paper 1 都用 5,reviewer 接受度高
5. **若 P2 marginal,revision 阶段补 2 seed** (升到 7) 仍是 follow-up option

**Seed pool** [42, 43, 44, 45, 46] 与 broad val v2 + ReBRAC paper 1 main spec 一致 (P0+P1 spec §0.2)。

### 4.2 Per-run CLI templates

**ReBRAC template** (Cell {C} × Seed {S}):

```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.train_offline \
    --algo rebrac \
    --offline-data offline_data/fql_succession/{cell_dataset_path} \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --manifest benchmarks/single_u10_cross_tgt15_ep100.json \
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
    --seed {S} \
    --save-dir checkpoints/fql_succession/p2/{cell_id}/rebrac_seed{S} \
    --device cuda
```

**FQL template** (Cell {C} × Seed {S}):

```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.train_offline \
    --algo fql \
    --offline-data offline_data/fql_succession/{cell_dataset_path} \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --manifest benchmarks/single_u10_cross_tgt15_ep100.json \
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
    --seed {S} \
    --save-dir checkpoints/fql_succession/p2/{cell_id}/fql_seed{S} \
    --device cuda
```

**关键差异 vs Gate B Option B CLI**:
- `--manifest`:**`single_u10_cross_tgt15_ep100.json`** (D18, 取代 30-ep)
- `--seed`:5 个 seed [42-46] 而非 2 个 [0, 42]
- `--save-dir`:`checkpoints/fql_succession/p2/{cell_id}/...` (per-cell namespace)
- 其余 hyperparam 全部继承 Gate B Option B (per §1.2 表)

### 4.3 Skip-resume 协议

Gate B Option B 经验:rev.3.1 Colab session 经常被 idle timeout 杀掉,必须**显式 skip-resume**。

**Resume 检测条件** (per run):
- `trainer_state.json` 存在 + `train_step ≥ total_steps` → skip (run 已 done)
- `trainer_state.json` 存在 + `train_step < total_steps` → resume from checkpoint (notebook cell `--resume <save-dir>` flag)
- 都不存在 → fresh run

**Per-run state files** (per Gate B observation):
- `trainer_state.json` (resume metadata)
- `agent_final.pt` (final weights)
- `train_log.jsonl` (loss metrics per log-step)
- `eval_log.csv` (in-training eval, dedup-by-train_step in verdict aggregation)
- `test_result.json` (final canonical 100-ep eval, written at end)

**Notebook cell 模板** (Bug 5 path-quote 防护):

```python
# 每个 (algo, cell, seed) 一个 cell
save_dir = f"checkpoints/fql_succession/p2/{cell_id}/{algo}_seed{S}"
test_json = f"{save_dir}/test_result.json"
if os.path.exists(test_json):
    print(f"[skip] {save_dir} already done")
else:
    # Bug 5: single-quote path with potential spaces ("Colab Notebooks" Drive path)
    cmd = f"""!python -m scripts.train_offline \\
        --algo {algo} \\
        ... \\
        --save-dir '{save_dir}'"""
    exec(cmd)  # via Jupyter magic, not os.system (Bug 4 mitigation)
```

详见 §9 notebook scaffold 设计。

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
    fql_per_seed: np.ndarray,        # [n_seeds] (5)
    rebrac_per_seed: np.ndarray,     # [n_seeds] (5)
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

### 5.4 iff verdict

**δ_null = 0.05** (5pp threshold for "no effect"):
- **E-uni**: `|Δ_E-uni| < δ_null` AND Welch p > 0.10 → ¬A no-effect 确认
- **M-uni-noise**: `|Δ_M-uni-noise| < δ_null` AND Welch p > 0.10 → ¬B no-effect 确认
- **M-multi-mix**: `Δ_M-multi-mix > 0` AND Welch p < 0.05 (after Bonferroni) AND `Δ_M-multi-mix CI lower > 0` → A∧B effect 确认

**完整 iff** = 三条全成立 → plan v1.2 → v2.0 upgrade,进 P3。

### 5.5 Multiple comparison (Bonferroni primary + BH/uncorrected sensitivity)

3 cell × 1 algo gap test = **3 tests**。本 spec **primary verdict 用 Bonferroni**:critical p = `0.05 / 3 = 0.0167`。

**理由 (Bonferroni vs BH)**:
- iff claim 是 **3-way conjunction** (E-uni null AND M-uni-noise null AND M-multi-mix positive)。一个 cell 错就破坏 iff → Type I error 控制 (FWER) 比 FDR 更对应 paper claim 逻辑
- Paper-typical 多 cell 对照采 Bonferroni 更 defensible (reviewer 接受度高)
- BH FDR @ q=0.05 在 3-test 下 critical p 阶梯式 {0.0167, 0.033, 0.05},仅在 marginal case 与 Bonferroni 给不同 verdict

**Sensitivity table** (must report in verdict report appendix per §6.4):

| Test | Welch t | Welch p (one-sided) | Bonferroni p<0.0167 verdict | BH FDR @ q=0.05 verdict | Uncorrected p<0.05 verdict |
|---|---:|---:|:---:|:---:|:---:|
| E-uni null (¬A) | (fill) | (fill) | (fill) | (fill) | (fill) |
| M-uni-noise null (¬B) | (fill) | (fill) | (fill) | (fill) | (fill) |
| M-multi-mix positive (A∧B) | (fill) | (fill) | (fill) | (fill) | (fill) |

**Verdict 判定规则**:
- **Bonferroni-primary** (paper main claim binary verdict):per §5.4 iff verdict 三选一
- **BH sensitivity** (paper §discussion robustness check):若 BH 与 Bonferroni verdict 不同,paper 写 hedge "under more permissive FDR correction at q=0.05, our M-multi-mix finding would have been retained with uncorrected p={X}; we report the more conservative Bonferroni verdict for primary claim defensibility"
- **Uncorrected** (transparency only):仅作为完整披露,**不**作为 verdict 输入

**Null verdict 校正策略**:E-uni null + M-uni-noise null 的 "null" claim 不需要 multiple-comparison 校正 (single test per cell, H1 = positive, fail to reject ≠ wrong)。仅 positive effect 判定需 Bonferroni。

### 5.6 c4 monitoring (Option α gate, P2 default config)

承接 Gate B Option B caveat 与 D19 c4 阈值改革:

**c4 定义** (per algo × cell):
- per-seed slope = last-30% linear regression slope of in-training eval success rate (n_pts = 6 per seed,因为 200k steps + eval-every 10000 = 20 eval pts,last 30% = 6 pts)
- aggregated slope = mean over 5 seeds
- threshold = `−2 × SE(slope_aggregated_5_seeds)`
- SE_per_seed = √(p(1−p)/n_eval) / √17.5 (n_eval = 100 from ep100 manifest, n_pts=6 → denom=17.5)
- SE_aggregated = SE_per_seed / √5

**P2 default 数值** (per c4 revision doc §3 retroactive table):

| Quantity | Value |
|---|---:|
| n_eval (manifest size) | 100 |
| n_seeds | 5 |
| p (typical in-training mean success) | ≈ 0.7 |
| SE_eval @ p=0.7 | 0.0458 |
| SE_per_seed (n_pts=6) | 0.0110 |
| SE_aggregated (5 seeds) | **0.00490** |
| **c4 threshold** | **−0.00980** |

**c4 verdict (per algo × cell)**:
- aggregated slope ≥ −0.0098 → c4 PASS
- aggregated slope < −0.0098 → c4 FAIL (真有 late-training collapse)

**c4 在 P2 是 gate 还是 monitor**:
- **Gate** (per Session B Option α design): c4 阈值是 statistical-meaningful (95% 单侧 CI 下界);不会被噪声触发 (Gate B 验证)
- Per cell × algo × per algo gap 都有自己的 c4 verdict
- 若 **FQL 在任一 cell c4 FAIL** → flag in §6 Gate C.3 verdict,paper claim 写作时 hedge "FQL stability marginally fails in cell X"
- ReBRAC c4 FAIL 与 paper claim 无直接关系 (paper 不 claim ReBRAC stability),仅记录到 verdict report appendix

### 5.7 Per-run metric set (reference)

每 run 落地的可计算指标 (从 `eval_log.csv` + `test_result.json` 提取):

| Metric | Source | Role |
|---|---|---|
| `last_3_mean` | eval_log.csv last 3 train_step rows (dedup) | **§5.1-5.4 primary scalar** |
| `final_test_success` | test_result.json `test_success` | Secondary (Gate B 经验:more noise) |
| `last_30pct_slope` | linregress(last 6 eval pts) | **§5.6 c4 monitoring** |
| `final_test_return` / `safety` / `progress` / `path_eff` | test_result.json | Reported in §results table appendix |
| `loss_flow_tail` | train_log.jsonl last 5% | FQL teacher health (per Gate B c2) |
| `loss_actor_tail` | train_log.jsonl last 5% | Per-algo comparison parity check |
| `q_mean_tail` | train_log.jsonl last 5% | Q stability monitor |

---

## 6. Gate C 判据 (P2 出口)

### 6.1 Gate C.1 — Completeness

30-run 全部完成且无系统性 error:
- 每 run 有 `trainer_state.json` + `agent_final.pt` + `test_result.json` + `eval_log.csv` (≥18 unique-step rows post-dedup,允许 ≤2 rows 缺失 from rare eval failures)
- 0 run NaN / Inf loss in `train_log.jsonl` 末段
- 0 run 因 OOM / shape mismatch / argparse error crashed

### 6.2 Gate C.2 — Audit

§3 三个 per-cell audit 全部 verdict 正确:
- A1 (E-uni self-check): p_≥2 < 0.20
- A2 (M-uni-noise vs E-uni): M-uni-noise p_≥2 < 0.20 + Δp(≥2) close to 0 (no significant 区分)
- A3 (M-multi-mix vs E-uni): M-multi-mix p_≥2 > 0.30 + Δp(≥2) CI lower > 0.10 + Welch p < 0.07

任一 cell A1/A2/A3 fail → **不进** statistical verdict (重收集 + re-audit)。

### 6.3 Gate C.3 — iff verdict 三选一

按 §5.4 + §5.5 Bonferroni p<0.0167 / §5.6 c4 + §6.1/§6.2 同时 satisfied,以下三选一:

| Verdict | Condition | Paper claim | P3 trigger |
|---|---|---|---|
| **完整 iff hold** | E-uni null (Δ<0.05, p>0.10) ∧ M-uni-noise null ∧ M-multi-mix positive (Δ>0, p<0.0167, CI lower>0) | 主路径 conditional iff (plan §1.2) | ✓ 进 P3 mix ratio sweep (验证 modality 强度单调性) |
| **Partial iff** | E-uni null ∧ M-multi-mix positive ∧ **M-uni-noise also positive** | 降级:"FQL > ReBRAC on sub-optimal data regardless of modality" (drop iff,只 keep quality 条件) | ✓ 进 P3,但 modality intensity 单调性变 secondary evidence |
| **Null iff** | M-multi-mix not positive (Δ≤0 or p≥0.0167 or CI lower≤0) | 重写为 scoped negative finding:"expressive prior 在本 AUV navigation task 无 leverage" (R3 mitigation) | P3 决策点:(a) 仍跑 P3 验证 negative finding robustness; (b) skip P3 + 直接进 P4 negative paper writing |

**E-uni 上 FQL 倒挂 (Δ_E-uni < −0.05 AND p < 0.10)** 是 catastrophic case:
- 与 Gate B finding (FQL on E-uni Δ=−4.4pp) 一致,marginal — 不一定 catastrophic
- 但若 P2 实测 |Δ_E-uni| > 0.08 (超出 Gate B 阈值),触发 R4 mitigation:debug FQL hyperparam,debug 3 iter 仍倒挂则 STOP

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
| **P2.R3** | n_seeds=5 仍欠 power,Δ_M-multi-mix CI lower ≈ 0 | 低-中 | 中 | revision 阶段补 2 seed (升到 7),CI 收紧 ~15% (1/√7 vs 1/√5);本 sprint 不 preemptive |
| **P2.R4** | FQL 在 E-uni 上 Δ > −0.08 但 < −0.05 (marginal underperform) | 中 (Gate B 实测 −4.4pp 已逼近) | 中 | 不阻塞 iff null verdict (|Δ|<0.05 严格,−0.044 已逼近边界);P4 写 hedge "FQL marginally underperforms ReBRAC on expert-uni regime" |
| **P2.R5** | c4 在 P2 multi-cell ramp 后又出现 seed-driven 不一致 | 中 | 低 | Option α 阈值已 statistical-meaningful;若仍 cell-level 不一致 (e.g. FQL c4 PASS in M-multi-mix 但 FAIL in E-uni),记入 verdict report,**不**影响 iff binary verdict |
| **P2.R6** | Bonferroni p<0.0167 严格,M-multi-mix marginal 显著 (p=0.03) 但 fail 阈值 | 中 | 中 | 报告 sensitivity:Bonferroni-corrected vs BH FDR vs uncorrected per-cell,paper §method 显式选 Bonferroni 但 appendix 报 sensitivity |
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

| Item | Per run | Total |
|---|---:|---:|
| ReBRAC 200k step (Gate B observed) | ~25 min | 5 seed × 3 cell = 15 run → ~6.25 h |
| FQL 200k step (Gate B observed) | ~50 min | 5 seed × 3 cell = 15 run → ~12.5 h |
| Verdict aggregation + audit per cell run | trivial (~10 min) | local |
| **Total training wallclock** | | **~19 h L4 sequential** |

Parallel notebook strategy: per-cell notebook (3 notebooks) × 2-3 concurrent Colab sessions (different Google accounts or sequential) → ~7-10 h actual elapsed if 2 concurrent sessions。

### 8.3 Sprint plan (estimated, 2026-05-21 起算)

| Day | Task |
|---|---|
| Day 1 (本机) | M-uni-noise + M-multi-mix Source A/B collection + concat + 3 audit + dataset cards 起草 |
| Day 2 (本机) | Per-cell dataset cards finalize;commit collection + audit results;notebook scaffold finalize |
| Day 3-4 (Colab) | E-uni cell × FQL + ReBRAC × 5 seeds = 10 run (1 notebook) |
| Day 5-6 (Colab) | M-uni-noise cell × 10 run |
| Day 7-8 (Colab) | M-multi-mix cell × 10 run |
| Day 9 (本机) | Verdict report (§5 + §6) + paper-2 outline finalize |

**Total wallclock**:~1.5 周 (vs plan §4.2 estimate 1.5 周) ✓ on-budget。

### 8.4 算力 contingency

| Trigger | Impact |
|---|---|
| Colab L4 quota burn | 切换到 Drive sync + 本机 GPU (若有);最多 +3 天 |
| Skip-resume failure | per-run independent save-dir 防 cross-run contamination,最多 affected run 重跑 (~1h) |
| Audit per-cell fail (R1/R2) | re-collect + re-audit,+0.5-1 天 |

---

## 9. Notebook scaffold 计划

### 9.1 Notebook 划分原则

**Per-cell notebook** (而非 all-cells single notebook):
- 优势:skip-resume 粒度更细,某 cell 失败不影响其他 cell;Colab session 12h limit 可 per-cell session;debugging 路径清晰
- 劣势:notebook 文件多 (collection 1 + run 3 + verdict 1 = 5 notebooks)
- 决定:**per-cell** (Session A decision per task prompt)

### 9.2 Notebook 清单

| Notebook | 在哪跑 | 内容 | 状态 |
|---|---|---|---|
| `notebooks/fql_succession_p2_collection.ipynb` | **本机** (mytorch1) | M-uni-noise + privileged-500 + goalseek-500 collect + concat M-multi-mix-1000 + 3 audit (A1/A2/A3) | 本 spec 起 scaffold (任务 4) |
| `notebooks/fql_succession_p2_run_cell_e_uni.ipynb` | **Colab L4** | E-uni cell × FQL + ReBRAC × 5 seeds = 10 run | 本 spec 起 scaffold (任务 4) |
| `notebooks/fql_succession_p2_run_cell_m_uni_noise.ipynb` | **Colab L4** | M-uni-noise × 10 run | 本 spec 起 scaffold (任务 4) |
| `notebooks/fql_succession_p2_run_cell_m_multi_mix.ipynb` | **Colab L4** | M-multi-mix × 10 run | 本 spec 起 scaffold (任务 4) |
| `notebooks/fql_succession_p2_verdict.ipynb` | 本机 | aggregate 30-run summaries + paired bootstrap + Welch + Cohen d + iff verdict | 待 P2 闭环时另起 |

### 9.3 Per-cell run notebook 必须 cell

仿 `notebooks/fql_succession_gate_b.ipynb` rev.3.1 风格 (real-time streaming via Jupyter `!python ...` magic;path-quote 防 Bug 5;skip-resume via `test_result.json` 存在检测)。**每 cell 都是 idempotent**:Colab session restart 后整本 re-run 不会因 partial state 出错。

| Cell ID | 内容 | Idempotent? |
|---|---|---|
| `header` | Drive mount + cd into project + Bug 5 path-quote 防护 disclaimer + CLI rename mapping 表 (cite p0p1 spec v1.1 patch) | ✓ |
| `setup` | env vars + paths + 定义 `run_one(algo, seed)` helper (见 §9.3.1) + run matrix list | ✓ |
| `rebrac_seed_42` ... `rebrac_seed_46` | **5 cell**,每 cell 一行调用:`run_one('rebrac', 42)` ... `run_one('rebrac', 46)` | ✓ (helper 内 skip-resume check) |
| `fql_seed_42` ... `fql_seed_46` | **5 cell**,每 cell 一行调用:`run_one('fql', 42)` ... `run_one('fql', 46)` | ✓ |
| `summarize` | 收集 10 run 的 metrics → CSV per cell;**guard:if not all_done(): raise** | ✓ |
| `verdict_preview` | per cell paired bootstrap CI + Welch p + Cohen d + c4 verdict;**guard:CSV 存在 + 10 row 完整再 compute** | ✓ |

### 9.3.1 `run_one()` helper 模板 (定义在 `setup` cell)

```python
import os, json

CELL_ID = "e_uni"   # or "m_uni_noise" / "m_multi_mix"
DATASET = "offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000"  # per-notebook
FLOW = "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
MANIFEST = "benchmarks/single_u10_cross_tgt15_ep100.json"
SAVE_ROOT = f"checkpoints/fql_succession/p2/{CELL_ID}"

REBRAC_FLAGS = (
    "--actor-penalty-coef 4.0 --critic-penalty-coef 2.0 "
    "--critic-layernorm --no-actor-layernorm"
)
FQL_FLAGS = (
    "--flow-steps 10 --distill-alpha-bc 1.0 "
    "--teacher-lr 3e-4 --flow-time-embed-dim 32"
)

def run_one(algo: str, seed: int) -> None:
    save_dir = f"{SAVE_ROOT}/{algo}_seed{seed}"
    test_json = f"{save_dir}/test_result.json"
    if os.path.exists(test_json):
        print(f"[skip] {save_dir} (test_result.json exists)")
        return
    algo_flags = REBRAC_FLAGS if algo == "rebrac" else FQL_FLAGS
    # Bug 5: single-quote save_dir (potential spaces in Colab Drive path)
    cmd = (
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
        f"--seed {seed} "
        f"--save-dir '{save_dir}' "
        f"--device cuda"
    )
    print(f"[run] {save_dir}")
    print(f"      cmd: {cmd[:120]}...")
    # Use IPython.get_ipython().system() for real-time stream (Bug 4 mitigation)
    get_ipython().system(cmd)
```

### 9.3.2 Idempotent `summarize` cell

```python
ALGOS = ["rebrac", "fql"]
SEEDS = [42, 43, 44, 45, 46]

def all_done() -> bool:
    return all(
        os.path.exists(f"{SAVE_ROOT}/{a}_seed{s}/test_result.json")
        for a in ALGOS for s in SEEDS
    )

if not all_done():
    pending = [
        f"{a}_seed{s}" for a in ALGOS for s in SEEDS
        if not os.path.exists(f"{SAVE_ROOT}/{a}_seed{s}/test_result.json")
    ]
    raise RuntimeError(f"[not ready] {len(pending)} runs pending: {pending}")

# ... collect last_3_mean / slope / loss_flow_tail / actor_loss_tail per run → CSV
```

`verdict_preview` cell 同样 guard:`if not os.path.exists(csv_path): raise`。

### 9.4 Collection notebook 必须 cell

| Cell ID | 内容 |
|---|---|
| `header` | local env (mytorch1 PATH) + cd into project |
| `m_uni_noise_collect` | privileged + ε=0.5 + 1000 ep collection (~4 min) |
| `m_multi_mix_source_a` | privileged-500 (~1.5 min) |
| `m_multi_mix_source_b` | goalseek-500 (~2.5 min) |
| `m_multi_mix_concat` | concat_offline_datasets call (<1 min) |
| `audit_a1_e_uni_self` | self-audit E-uni → sanity p_≥2 < 0.20 |
| `audit_a2_m_uni_noise_vs_e_uni` | A2 |
| `audit_a3_m_multi_mix_vs_e_uni` | A3 |
| `summary` | 三 audit JSON 汇总 + Gate C.2 preview verdict |
| `dataset_cards_template` | M-uni-noise + M-multi-mix dataset card 模板填写 reminder (手动 finalize) |

### 9.5 Notebook 间状态契约

- Collection notebook 产出 `offline_data/fql_succession/{m_uni_noise,m_multi_mix_50priv_50goal}_1000/` + audit results 在 `results/fql_succession/p2/audit/`
- Run notebook 消费 dataset path,产出 `checkpoints/fql_succession/p2/{cell_id}/{algo}_seed{S}/` (Drive-mounted)
- Verdict notebook 消费 checkpoints + audit results,产出 `docs/fql_succession_p2_main_report.md`

---

## 10. Caveat 段

### 10.1 c4 marginal-FAIL carryforward (from Gate B)

Gate B Option B 实测 c4 marginal-FAIL (aggregated slope −0.0038,30-ep manifest noise floor 内)。原因 seed-driven (seed=0 slope 正,seed=42 slope 负,两 algo 同方向)。P2 用 Option α 阈值 (D19) + ep100 manifest (D18) + n_seeds=5,redo c4 verdict per cell × algo。

**预期**:在 ep100 manifest 下 SE_agg ≈ 0.005,Option α threshold ≈ −0.0098;Gate B FQL aggregated slope −0.0038 远高于阈值。retroactive PASS 已在 c4 revision doc §3 验证。P2 mainline 配置下 c4 应稳定 PASS。

**Paper writing implication**:§experiments 段 footnote 提及 "Gate B 2-seed validation found marginal c4 fail (slope −0.0038), statistically indistinguishable from 0 under 30-episode evaluation noise. P2 5-seed evaluation with 100-episode manifest tightens the noise floor by 1.8×, restoring c4 statistical power."

### 10.2 Noise floor 更新 (Bug 2 fix 后)

直接复用 [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) §3.1:

| 量 | 30-ep manifest (Gate B) | 100-ep manifest (P2 default) | 改善 |
|---|---:|---:|---:|
| per-eval SE @ p=0.7 | 0.0837 | 0.0458 | 1.83× |
| slope SE per-seed (n_pts=6) | 0.0200 | 0.0110 | 1.83× |
| slope SE aggregated n_seeds=5 | 0.00894 | 0.00490 | 1.83× |
| c4 threshold (Option α) | −0.0179 | −0.0098 | (auto-scale) |

**Per-cell × algo c4 slope 预期范围** (under healthy training):
- |aggregated slope| ≤ 2 × SE_agg ≈ 0.01 (in noise band)
- 出格意味着 真有 trend (positive = late improvement; negative = collapse)

### 10.3 n_seeds 决策 (5 vs 7) 详细 power 分析

Gate B 暴露的 effect-size benchmark:

| Quantity | Value |
|---|---:|
| FQL vs ReBRAC last-3 gap @ seed=0 | −14.4pp |
| FQL vs ReBRAC last-3 gap @ seed=42 | +5.6pp |
| Pooled std (2-seed estimate, gross under-estimate) | ~14pp |

在 plan claim 想 detect 的 effect size (Δ_M-multi-mix ≈ 0.05-0.10) 下:
- **n=5,σ_pooled=0.10**: SE_paired = 0.10/√5 ≈ 0.045 → 50%-power threshold ≈ 0.058
- **n=7,σ_pooled=0.10**: SE_paired = 0.10/√7 ≈ 0.038 → 50%-power threshold ≈ 0.049

**n=5 power ≈ 70-80%** to detect 0.10 effect at α=0.0167 (Bonferroni);**n=7 power ≈ 80-90%**。差距 ~10%。

**判断**:n=5 already in "paper-defensible" power range; n=7 +40% wallclock 带 ~10% power 增益,边际收益不足。**采用 n=5**。若 P2 marginal,revision 阶段补 2 seed (升到 7) 成本 ~8h Colab。

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

*Document version: v1.0 (2026-05-20, Session A 起草). 维护策略:Sprint 0 (collection) 完成后升级 v1.1 + dataset card 填写;每 cell run notebook 闭环后升级 v1.x;30-run 全闭环 + verdict report 完成后升级 v2.0 并 trigger plan v1.2 → v2.0。*
