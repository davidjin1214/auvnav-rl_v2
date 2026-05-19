# arrival_v2 §8 P0 — SAC Variance Reduction Design

**Status**: DRAFT (2026-05-19). Awaiting k=12 seed=0 sister回流 to finalize P0 entry condition.
**Branch**: `codex-arrival-v2-prototype`
**Predecessor**: §7.8 multi-seed (3 anchors) + §8 P1#2 k=12 monotonicity (seed=42 single-anchor).
**Successor**: P0 closure → either continued history extension (k=16) or pivot to architectural variance reduction.

> **Naming note**: "§8 P0" here is the **first sub-phase of §8** in the arrival_v2 line, **not** the closed §5 P0 profiling work in `online_rl_thesis_plan.md` (which characterized num_envs / IPC bottleneck). The two share the "P0" letter but live in disjoint document namespaces.

---

## 1. Motivation — Why P0 is now thesis-relevant

### 1.1 Six-anchor evidence base on `single_u15_cross_tgt15` × `s0` × vanilla SAC × 1M

| run                            | seed | k    | final  | peak@step    | mean39 | OOB    | n_succ | Gate         |
|---                             |---:  |---:  |---:    |---           |---:    |---:    |---:    |---           |
| §7.6.4                         | 42   | 4    | 0.100  | 0.367 @ 975k | 0.221  | 0.667  | 35/39  | FAIL floor   |
| §7.7.1 sister                  | 0    | 4    | 0.400  | 0.533 @ 625k | 0.218  | 0.200  | 34/39  | FAIL         |
| §7.8 anchor                    | 42   | 8    | **0.900** | 0.900 @ 475k | **0.636** | **0.100** | 37/39 | **PASS 5/5** |
| §7.8'                          | 0    | 8    | 0.500  | 0.500 @ 925k | 0.260  | 0.133  | 32/39  | PARTIAL 2/5  |
| §7.8''                         | 7    | 8    | 0.867  | 0.900 @ 550k | 0.518  | 0.133  | 31/39  | BORDER 4/5   |
| §8P1#2 anchor                  | 42   | 12   | **0.900** | 0.900 @ 375k | 0.652 | **0.100** | 35/39 | **PASS PLATEAU** |
| §8P1#2 sister (pending回流)    | 0    | 12   | ?      | ?            | ?      | ?      | ?      | ?            |

### 1.2 Variance budget on k=8 (3 seeds)

| Metric                  | seed=42 | seed=0  | seed=7  | mean   | std    | range  |
|---                      |---:     |---:     |---:     |---:    |---:    |---:    |
| final_success_rate      | 0.900   | 0.500   | 0.867   | 0.756  | 0.181  | 0.400  |
| mean39                  | 0.636   | 0.260   | 0.518   | 0.471  | 0.157  | 0.376  |
| OOB                     | 0.100   | 0.133   | 0.133   | 0.122  | 0.016  | 0.033  |
| n_evals_with_success    | 37      | 32      | 31      | 33.3   | 2.6    | 6      |

**Observations**:

1. **`final` 维度 σ ≈ 0.18** — 比 §7.1 `s1_k4` reference 的 mean 0.497 还高，且 seed=0 是显著 outlier (final 0.500 vs 0.867/0.900).
2. **OOB 维度 cross-seed 鲁棒** — 3/3 seeds 在 0.10-0.135 (vs k=4 floor 0.667)，σ_OOB ≈ 0.016 << σ_final.
3. **n_succ 维度 33.3/39 ± 2.6** — 整个训练轨迹有 31-37 个 eval 命中过 success，但最后 final 端 seed=0 collapse 到 0.500.
4. **物理解读**：信息 capacity 充足（OOB 已 robust 压低），但 SAC 学习动力学在不同 random init 上发散出不同的 final policy.

### 1.3 Two competing hypotheses for the seed=0 stall

| Hypothesis | Evidence for | Evidence against |
|---|---|---|
| **H_info-bottleneck**: k=8 仍不够长 (seed=0 需要 k=12) | k=12 s42 mean +1.6pp vs k=8 s42, peak step 100k earlier | OOB 已 cross-seed robust 表明 information capacity sufficient |
| **H_optimization-noise**: SAC 学习动力学在 seed=0 落入 bad basin | seed=0 trajectory 全程在 0.4-0.5 高原 (last100k_mean=0.475 ≈ peak=0.500)，无 catastrophic but 持续 sub-optimal | k=12 seed=0 sister 未回流，无法直接对照 |

**P0 design 优先考虑 H_optimization-noise 路径**：即使 k=12 seed=0 (待回流) 也无法解 seed=0 stall，也仍需 variance reduction 手段把 σ_final 从 0.18 降到 thesis 可接受水平 (≤0.08-0.10).

---

## 2. Candidate Techniques

### 2.1 Survey

| Technique | Mechanism | Code complexity | Compute cost | Expected σ_final ↓ |
|---|---|---:|---:|---|
| **DroQ-lite** (LayerNorm + Dropout) | Critic regularization 抑制 overestimation | **零** (现有 flag) | 1.0x | mild (0.05-0.10) |
| **DroQ-full** (LN + Dropout + UTD=20) | + 高 update ratio | **零** | ~20x critic update | moderate-strong (0.10-0.15) |
| **N-Step Returns** (n=3 or 5) | TD bootstrap 走 n 步降 reward noise | mild (replay buffer + train loop) | ~1.0x | mild-moderate (0.05-0.10) |
| **REDQ ensemble** (M=10 critics, K=2 random subset) | Critic ensemble 降 Q variance | moderate (model + train loop) | ~10x critic | moderate-strong (0.10-0.20) |
| **SimBa residual norm** | Obs residual + lr decay + critic ensemble | heavy (架构 + scheduler) | 1.5-2x | TBD (modern, less data on SAC) |
| **CrossQ BatchNorm** | Critic BN, 取消 target net, UTD=1 仍稳 | heavy (BN handling + target net 改动) | 1.0x | TBD |
| **TQC** (Truncated Quantile Critics) | 分位 critic + truncation | very heavy (distributional) | 1.5x | moderate |

### 2.2 Recommended subset for arrival_v2 §8 P0

**P0a — DroQ-lite (zero impl cost)**:
- Flags only: `--use-layernorm --dropout-rate 0.01 --updates-per-step 1`
- Run on `s0_k8 × seed ∈ {42, 0, 7}` (3 anchors, **identical to §7.8 multi-seed**).
- Direct paired contrast vs §7.8 / §7.8' / §7.8'' (single-flag-flip discipline).

**P0b — DroQ + UTD=4 (zero impl cost)**:
- `--use-layernorm --dropout-rate 0.01 --updates-per-step 4`
- Run on `s0_k8 × seed=0` only first (focus on the unlucky seed)
- If seed=0 rescued (final ≥ 0.85) → expand to seed ∈ {42, 7}
- Note: UTD=4 means 4x critic gradient compute per env step → ~3-4h L4 per run instead of 2.5h

**P0c — N-Step Returns (mild impl cost)**:
- Add `--n-step-returns N` flag (default N=1 = current behavior)
- Replay buffer modification: sample n-step transitions
- Train loop: compute n-step TD target
- Run on `s0_k8 × seed=0 × N ∈ {3, 5}` first
- Defer until P0a/b 完成评估

**P0d — REDQ ensemble (moderate impl, optional)**:
- Add `--num-critics M --critic-subset K` flags
- Model: AsymmetricQNetwork → list[QNetwork]
- Train: random-K subset for target, mean for actor
- Defer until P0a/b 不足才考虑

**Out of scope for P0**: SimBa / CrossQ / TQC — implementation cost太高，留给 future thesis chapter if P0a-c 不足.

---

## 3. Single-Variable Discipline & Baseline Matrix

All P0 runs must isolate **exactly one variance-reduction axis** vs the §7.8 anchor (vanilla SAC k=8). The single-flag-flip discipline that worked for §7.6→§7.7→§7.8 must extend to §8.

| Run ID    | seed | k  | LN  | dropout | UTD | n-step | Comparison Anchor |
|---        |---:  |---:|---  |---:     |---: |---:    |---                |
| §7.8 anchor (baseline) | 42 | 8 | F | 0.0 | 1 | 1 | (anchor) |
| §7.8' baseline         | 0  | 8 | F | 0.0 | 1 | 1 | (anchor) |
| §7.8'' baseline        | 7  | 8 | F | 0.0 | 1 | 1 | (anchor) |
| **§8 P0a-42**         | 42 | 8 | **T** | **0.01** | 1 | 1 | vs §7.8 |
| **§8 P0a-0**          | 0  | 8 | **T** | **0.01** | 1 | 1 | vs §7.8' |
| **§8 P0a-7**          | 7  | 8 | **T** | **0.01** | 1 | 1 | vs §7.8'' |
| §8 P0b-0              | 0  | 8 | T | 0.01 | **4** | 1 | vs §8 P0a-0 |
| §8 P0c-0              | 0  | 8 | F | 0.0 | 1 | **3 or 5** | vs §7.8' |

---

## 4. Gates & Verdict Schema

### 4.1 Per-run gate (与 §7 / §7.8 同口径)

PASS 5/5:
- `final_success_rate ≥ 0.85`
- `last100k_mean ≥ 0.9 × peak`
- `final_oob_rate ≤ 0.10`
- `include_episode_context_obs == True`
- `timeout_bootstrap_semantics == 'terminal'`

BORDERLINE-PASS (从 §7.8'' seed=7 教训 codified):
- `final_success_rate ≥ 0.85` AND `0.10 < OOB ≤ 0.135` AND last100k/context/terminal PASS

### 4.2 Cross-anchor P0a verdict (3-seed aggregate)

| Verdict | Trigger | §8 主张含义 |
|---|---|---|
| **P0a-CONFIRMED-VARIANCE-REDUCTION** | σ_final on 3 seeds ↓ ≥ 0.05 vs §7.8 baseline AND mean_final ↑ ≥ 0 | DroQ-lite 是正确轴；推进 P0b (UTD=4) 看是否能进一步 |
| **P0a-NO-EFFECT** | \|Δσ_final\| < 0.03 | DroQ-lite 不够；进 P0c (N-Step) 或 P0d (REDQ) |
| **P0a-MIXED** | σ_final ↓ but mean_final ↓ ≥ 0.05 | regularization 过强，损失 lucky seed gain；考虑降 dropout 到 0.005 重测 |
| **P0a-WORSENED** | σ_final ↑ ≥ 0.03 | DroQ-lite on this task counter-productive；直接进 P0d (REDQ) |

### 4.3 P0b/c rescue verdict (single-seed=0 focus)

| Verdict | Trigger | 含义 |
|---|---|---|
| **RESCUED** | seed=0 final ≥ 0.85 (strict or borderline) | 该技术解 seed=0 stall；扩到 seed ∈ {42, 7} 验 cross-seed |
| **PARTIAL** | seed=0 final ∈ [0.5, 0.85), Δ vs §7.8' ≥ +0.10 | 部分 buy 到，但未通过 gate；考虑组合 (P0a + P0b) |
| **NO-RESCUE** | seed=0 final < 0.5 + 0.10 = 0.60 | 该技术不解；试下一个 |

### 4.4 §8 thesis claim 走向

| P0a result | P0b/c needed? | §8 thesis claim |
|---|---|---|
| CONFIRMED-VARIANCE-REDUCTION + seed=0 rescued | No | "DroQ-lite is orthogonal stabilizer on top of k=8" — clean thesis chapter |
| CONFIRMED but seed=0 still stalls | Yes (P0b/c) | "DroQ reduces variance generally; seed=0 needs higher UTD or n-step" |
| NO-EFFECT | Yes (P0b/c then P0d) | "Simple regularization insufficient; ensemble methods are required" |
| MIXED | Re-tune | "Need careful regularization tuning" |
| WORSENED | Skip to P0d | "Architectural variance reduction is the right axis" |

---

## 5. Schedule

### 5.1 Sequential plan (sequential L4 sessions on Colab Pro+)

| Step | Notebook | Duration | Cumulative |
|---   |---       |---:      |---:        |
| (pending) k=12 seed=0 sister 回流 | `sac_arrival_v2_s0_cross_k12_seed0.ipynb` | 2.5h | 2.5h |
| P0a-42 (DroQ-lite seed=42) | `sac_arrival_v2_s0_cross_p0a_seed42.ipynb` | 2.5h | 5.0h |
| P0a-0  (DroQ-lite seed=0)  | `sac_arrival_v2_s0_cross_p0a_seed0.ipynb`  | 2.5h | 7.5h |
| P0a-7  (DroQ-lite seed=7)  | `sac_arrival_v2_s0_cross_p0a_seed7.ipynb`  | 2.5h | 10.0h |
| **P0a aggregate verdict** | (analysis cell in seed=7 notebook) | — | 10.0h |
| **P0b/c (conditional)** | TBD | 2.5-5h | 12.5-15h |

### 5.2 Parallel plan (3 Colab sessions concurrent)

If user has 3 Colab Pro+ sessions available:
- Step 1 (k=12 seed=0): solo run, 2.5h
- Step 2 (P0a × 3 seeds): parallel, max(2.5h, 2.5h, 2.5h) = 2.5h
- Total wall-clock: 5h

### 5.3 Conditional decision tree (post-P0a)

```
                  P0a 3-seed aggregate
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   CONFIRMED       NO-EFFECT/MIXED     WORSENED
        │                │                │
        ▼                ▼                ▼
   stop P0,         go P0b (UTD=4)    go P0d (REDQ,
   write thesis;    or P0c (N-Step)   impl cost)
   §8 closed                          
```

---

## 6. Implementation Notes

### 6.1 P0a — zero impl cost

Existing flags in `scripts/train_sac.py`:
- `--use-layernorm` (line 887)
- `--dropout-rate` (line 891, default 0.0; docstring in sac.py:69 says "DroQ recommends 0.01")
- `--updates-per-step` (line 857, default 1)

DroQ-lite CLI delta vs §7.8 anchor:
```diff
- --use-layernorm: not set (False)
- --dropout-rate: 0.0 (default)
+ --use-layernorm
+ --dropout-rate 0.01
```

**No code change required.** Build script differential is at notebook-template level only.

### 6.2 P0b — UTD=4 zero impl

Add `--updates-per-step 4`. Note that this **4x's critic compute per env step**, so wallclock per 1M env-step run ≈ 4 × 2.5h ≈ 10h. **Too long for one Colab session** — must use checkpoint/resume across 2 sessions.

Alternative: drop `total_steps` to 600k for P0b probing (faster turnaround, accept reduced statistical power; matches earlier online_rl_thesis_plan budget convention).

### 6.3 P0c — N-Step impl sketch

```python
# auv_nav/replay.py
class TransitionReplay:
    def sample(self, batch_size: int, n_step: int = 1) -> dict:
        # ... existing code ...
        # For n_step > 1: walk forward up to n_step-1 transitions in same episode,
        # accumulate discounted rewards, set "next_obs" to obs at step+n_step,
        # set "done" = any(dones[step:step+n_step])
        ...
```

Discount factor handling: TD target becomes
```
target = sum_{i=0..n-1} γ^i * r_{t+i} + γ^n * Q_target(s_{t+n}, a_{t+n})
```

**Estimated impl effort**: 80-150 LOC + test (modify TransitionReplay + train_sac TD loss; test trajectory boundary handling). 0.5-1 day of work before any Colab time.

### 6.4 P0d — REDQ ensemble impl sketch

```python
# auv_nav/sac.py
@dataclass
class SACConfig:
    num_critics: int = 2  # current; bump to 10 for REDQ
    critic_subset_k: int = 2  # random subset size for target
```

Model changes: QNetwork → `ModuleList[QNetwork]`. Train loop: sample K indices for target computation per step.

**Estimated impl effort**: 200-400 LOC + test. 1-2 days of work.

---

## 7. Open Questions

1. **Should P0a / P0b run on k=8 or k=12 baseline?**
   - Pro k=8: matches §7.8 multi-seed comparison directly, more existing baselines.
   - Pro k=12: if k=12 seed=0 PASSes (回流后), k=12 may already partially solve the stall, in which case P0 on k=12 measures incremental gain not catastrophic rescue.
   - **Tentative decision**: pin to k=8 unless k=12 seed=0 fully rescues seed=0 (CROSS-SEED-RESCUE verdict).

2. **σ_final reduction target — what's thesis-acceptable?**
   - From §7.8 multi-seed: σ_final = 0.18 on 3 seeds.
   - Typical published SAC results on continuous control: σ_final ~ 0.05-0.10 on 5+ seeds.
   - **Tentative target**: ≤ 0.10 with 3 seeds (≤ 0.15 with 4 seeds 加 seed=11 兜底).

3. **Do we also try arrival_v2 reward smoothing as orthogonal axis?**
   - Out of P0 scope (reward redesign is §9-style separate axis).
   - If P0a-d all fail, **then** consider revisiting reward shaping.

4. **Compute budget reality check**
   - P0a (3 runs × 2.5h) + P0b (1 run × 10h or 1 run × 600k × ~6h) + P0c impl + 1 run = ~25-30h L4 + 1-2 days dev
   - Manageable in 1-2 weeks of Colab Pro+ usage if scoped tightly.

---

## 8. Decision: When Does P0 Close?

P0 closes (and §8 main writing starts) when **any one** of:

1. P0a CONFIRMED-VARIANCE-REDUCTION + seed=0 RESCUED → write "DroQ-lite is the answer"
2. P0a + P0b combination achieves σ_final ≤ 0.10 + seed=0 RESCUED → write "DroQ-lite + UTD=4"
3. P0a + P0c achieves σ_final ≤ 0.10 + seed=0 RESCUED → write "DroQ + N-Step"
4. P0d (REDQ) achieves σ_final ≤ 0.10 + seed=0 RESCUED → write "Ensemble methods"
5. All P0a-d fail → write "Vanilla SAC on cross_stream s0 has irreducible σ_final ≈ 0.18; thesis 主线 should target a different reward / sensor combination or report range-statistics instead of strict gates"

---

## 9. Predecessors & Cross-references

- §7.8 multi-seed (3 anchors) gate summaries: `experiments/arrival_v2_prototype/s0_cross_k8_seed{0,7}_summary/combined_gate_summary.json`
- §8 P1#2 k=12 monotonicity (seed=42 PASS, seed=0 pending): `experiments/arrival_v2_prototype/s0_cross_k12_seed{42,0}_summary/combined_gate_summary.json`
- DroQ flag surface: `auv_nav/sac.py` lines 36-47, 68-69, 81-82, 122-123, 153-154; `scripts/train_sac.py` lines 338-339, 857, 887, 891
- Build script archive: `/tmp/build_nb_s0_cross_asym/build_*.py`
- Verdict-bug lesson from §7.8'' seed=7: 4-tier schemas miss BORDERLINE-PASS; future P0 notebooks must use 5-tier including UNCLASSIFIED catch-all (precedent in `build_k12_seed0.py`).
