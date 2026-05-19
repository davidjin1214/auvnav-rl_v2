# arrival_v2 §8 P0 — SAC Variance Reduction Design

**Status**: **DEMOTED-TO-FUTURE-WORK / POLISH-ONLY** (2026-05-19, post §7.9.2 closure).
**Branch**: `codex-arrival-v2-prototype`
**Predecessor**: §7.8 multi-seed (3 anchors) + §8 P1#2 k=12 monotonicity (seed=42 anchor).
**Outcome of pending step (k=12 seed=0 sister)**: **CROSS-SEED-RESCUE 5/5 PASS** — final 0.500 → 0.900, OOB 0.133 → 0.100, peak @ 525k vs 925k. See [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.9.2 / §7.9.3.
**Successor**: ~~either continued history extension (k=16) or pivot to architectural variance reduction~~ — **no immediate successor**. k=16 is decided not to expand (user judgment 2026-05-19); P0 is parked as future polish work.

> **Naming note**: "§8 P0" here is the **first sub-phase of §8** in the arrival_v2 line, **not** the closed §5 P0 profiling work in `online_rl_thesis_plan.md` (which characterized num_envs / IPC bottleneck). The two share the "P0" letter but live in disjoint document namespaces.

> **⚠ 2026-05-19 STATUS UPDATE — empirical resolution of §1.3 hypothesis pair**:
>
> When this doc was drafted (2026-05-19 morning), it took §7.9.1 k=8 cross-seed σ_final = 0.181 + seed=0 stall (final=0.500) as the entry condition for P0 — the assumption being that variance reduction (DroQ / N-Step / REDQ) was needed to either rescue seed=0 or shrink σ_final. The pending step was k=12 seed=0 sister (§8 P1#2'), framed as a **prerequisite** to decide between H_info-bottleneck (k=12 fixes seed=0 → no P0 needed) and H_optimization-noise (k=12 still stalls seed=0 → P0 mandatory).
>
> **k=12 seed=0 came back as STRICT-PASS 5/5 with CROSS-SEED-RESCUE verdict**: final 0.500 → 0.900 (+40pp), OOB 0.133 → 0.100, peak @ 525k vs §7.9.1' @ 925k. Combined with §7.9.2 k=12 seed=42 (PASS-PLATEAU), the k=12 cross-seed picture is 2/2 strict 5/5 PASS + mean39 σ砍半 (0.157 → 0.064).
>
> **Hypothesis resolution**: H_information-bottleneck wins; H_optimization-noise falsified. The decisive evidence is the seed=0 cross-history trajectory k=4: 0.400 → k=8: 0.500 → k=12: 0.900 — a **phase transition between k=8 and k=12** that random-init bad-basin (H_seed-stall) or SAC dynamics noise (H_optimization-noise) cannot produce. Physical match: control_step × k=12 ≈ 6 s ≈ vortex shedding period × 30–60% (Nyquist + half-period for phase reconstruction).
>
> **P0 motivation reframe**:
> - Original: "rescue seed=0 stall + bring σ_final from 0.181 to thesis-acceptable ≤0.10"
> - Now: **"k=12 alone has done both — final=0.900 on both seeds, σ_final=0.064 already < 0.10 target"**
> - Residual: mean39 still shows a 0.652 vs 0.525 seed gap, and OOB cross-seed alignment at 0.100 has 1 ep slack on k=8 seed=0/7 (0.133); P0 would only "crisp out remaining cross-seed noise," not rescue anything.
>
> **What this doc is now**: a **design reference** retained for:
> 1. Future thesis revision / paper rebuttal if a reviewer asks "why didn't you try DroQ?"
> 2. Reuse for the offline RL line (`ReBRAC` / `AUVHamNODE` variance reduction problems share the same techniques)
> 3. Should §8 thesis chapter ever need P0 polish on top of k=12 (low priority)
>
> Most below-section ratings (priorities, schedule, "P0 closes when…" conditions) are now **superseded** by §7.9.2 RESCUE; concrete sections are marked inline.

---

## 1. Motivation — Why P0 was thesis-relevant (now resolved by §7.9.2')

> **Note 2026-05-19**: This section was authored when k=12 seed=0 was pending and §1.3 H_optimization-noise vs H_info-bottleneck was unresolved. §7.9.2' RESCUE has since resolved §1.3 in favor of H_info-bottleneck, so the motivation argument below has been **answered empirically without running P0**. Retained for archival continuity.

### 1.1 Seven-anchor evidence base on `single_u15_cross_tgt15` × `s0` × vanilla SAC × 1M (updated 2026-05-19)

| run                            | seed | k    | final  | peak@step    | mean39 | OOB    | n_succ | Gate         |
|---                             |---:  |---:  |---:    |---           |---:    |---:    |---:    |---           |
| §7.6.4                         | 42   | 4    | 0.100  | 0.367 @ 975k | 0.221  | 0.667  | 35/39  | FAIL floor   |
| §7.7.1 sister                  | 0    | 4    | 0.400  | 0.533 @ 625k | 0.218  | 0.200  | 34/39  | FAIL         |
| §7.8 anchor                    | 42   | 8    | **0.900** | 0.900 @ 475k | **0.636** | **0.100** | 37/39 | **PASS 5/5** |
| §7.9.1' (formerly §7.8')       | 0    | 8    | 0.500  | 0.500 @ 925k | 0.260  | 0.133  | 32/39  | PARTIAL 2/5  |
| §7.9.1'' (formerly §7.8'')     | 7    | 8    | 0.867  | 0.900 @ 550k | 0.518  | 0.133  | 31/39  | BORDER 4/5   |
| §7.9.2 anchor                  | 42   | 12   | **0.900** | 0.900 @ 375k | 0.652 | **0.100** | 35/39 | **PASS PLATEAU 5/5** |
| **§7.9.2' sister**             | 0    | 12   | **0.900** | 0.900 @ 525k | 0.525 | **0.100** | 32/39 | **STRICT-PASS — CROSS-SEED-RESCUE 5/5 ⭐** |

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

### 1.3 Two competing hypotheses for the seed=0 stall — **RESOLVED 2026-05-19 by §7.9.2'**

| Hypothesis | Evidence (as of 2026-05-19 morning) | Post-§7.9.2' verdict |
|---|---|---|
| **H_info-bottleneck**: k=8 仍不够长 (seed=0 需要 k=12) | k=12 s42 mean +1.6pp vs k=8 s42, peak step 100k earlier | ✅ **CONFIRMED** — §7.9.2' k=12 s0 final 0.500→0.900 (+40pp) STRICT-PASS, OOB 0.133→0.100, peak @ 525k vs 925k; seed=0 cross-history trajectory k=4: 0.4 → k=8: 0.5 → k=12: 0.9 是 phase transition 直接证据 |
| **H_optimization-noise**: SAC 学习动力学在 seed=0 落入 bad basin | seed=0 trajectory 全程在 0.4-0.5 高原 (last100k_mean=0.475 ≈ peak=0.500)，无 catastrophic but 持续 sub-optimal | ❌ **FALSIFIED** — random init bad-basin / SAC dynamics noise 不会在唯一变量从 k=8 改到 k=12 时被"解锁"出 +40pp final 单调跃迁；regularization 也不能产生 +40pp 任务级跃迁 |
| **H_seed-stall**（init-dep local min, 额外列出来对照）| seed=0 trajectory 在 k=4 + k=8 都偏低 | ❌ **FALSIFIED** — 同上：init random seed 不会因 k 增加而改变 |

**Resolution**：H_information-bottleneck wins. **物理对应**：control_step ≈ 0.5 s × k=12 history = ~6 s 时序窗 ≈ 涡街周期 (10–20 s) × 30–60%；这已显著超过 Nyquist + 半周期阈值，足以让 actor 从单点 DVL 时序节拍中**鲁棒**反演主导脉动相位（即 critic 通过 privileged hull-integral 看到的同一物理量）。**k=8 (~4s) ≈ 涡街周期 20–40% 是 phase transition 临界点 — lucky seed (=42, =7) 上够，unlucky seed (=0) 上不够**。

**P0 design implication**：原 ~~"P0 design 优先考虑 H_optimization-noise 路径...仍需 variance reduction 手段把 σ_final 从 0.18 降到 thesis 可接受水平"~~ 被 §7.9.2' 实测**直接 short-circuit**：k=12 alone 已把 σ_final 砍半到 0.064（< thesis target 0.10）+ rescue seed=0 + 任务级 final 0.900/0.900 cross-seed strict alignment。**P0 variance reduction 不再是 rescue 必要轴，只是 polish 选项**（详见 §2.2 / §4.4 / §8 closure 重审）。

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

### 2.2 Recommended subset for arrival_v2 §8 P0 — priorities **downgraded 2026-05-19**

> **Priority context update**: §7.9.2' RESCUE has resolved §1.3 in favor of H_info-bottleneck; **all P0 priorities below have been demoted from "thesis-critical" to "future polish / paper revision"**. Recommended subset retained for design-reference completeness; no immediate execution recommended.

**P0a — DroQ-lite (zero impl cost)**: ~~HIGH priority~~ → **LOW priority (polish only)**
- Flags only: `--use-layernorm --dropout-rate 0.01 --updates-per-step 1`
- Run on `s0_k8 × seed ∈ {42, 0, 7}` (3 anchors, **identical to §7.8 multi-seed**).
  - **Note 2026-05-19**: 既然 k=12 cross-seed alone 已经把 σ_final 砍半到 0.064、rescue seed=0、达到 strict 5/5 PASS 2/2，**P0a on k=8 baseline 的 motivation 从"rescue seed=0"变成"看 DroQ-lite 能否在 k=8 上用 regularization 走捷径，不必走 k=12"**。这是相对 k=12 axis 的"orthogonal axis 探索"，已不是 thesis main line。
  - 若仍想跑：建议把 baseline pin 到 **k=12 而不是 k=8**（k=12 已 thesis-grade），观察 DroQ-lite 在已 thesis-grade baseline 上能否进一步 crisp out OOB 1-ep slack（k=8 seed=0/7 OOB 0.133 vs k=12 0.100）。

**P0b — DroQ + UTD=4 (zero impl cost)**: ~~HIGH priority for seed=0 rescue~~ → **DROPPED**
- 原 motivation 是把 UTD=4 当 seed=0 rescue 工具；§7.9.2' 实测 k=12 alone 已 rescue → P0b 失去任务级 motivation
- 仅保留为 mechanism-curiosity 选项（"UTD=4 on k=12 baseline 是否 crisp out 剩余 cross-seed mean39 gap 0.652 vs 0.525？"），但 10h L4 wallclock 性价比低

**P0c — N-Step Returns (mild impl cost)**: ~~MEDIUM priority~~ → **DROPPED (defer to offline line if ever needed)**
- 80-150 LOC impl effort + N-step TD target rewrite, motivation 已消失
- 该技术对 `ReBRAC` / offline RL line 有自身 motivation（offline TD bootstrap noise），更适合在那条线评估

**P0d — REDQ ensemble (moderate impl)**: ~~LOW priority (last resort)~~ → **DROPPED**
- 200-400 LOC impl effort; 既然 k=12 alone 已解，REDQ ensemble 的边际 thesis 贡献度 ≈ 0

**Out of scope for P0**: SimBa / CrossQ / TQC — implementation cost太高，留给 future thesis chapter if 需要进一步压 cross-seed variance. **2026-05-19 status**: 与 P0c/d 同样降级，offline 线复用价值高于 online 线.

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

### 4.4 §8 thesis claim 走向 — **SUPERSEDED by §7.9 cross-seed picture**

> **2026-05-19 actual outcome (not in original table)**: k=12 history-length alone resolved both σ_final (0.181 → 0.064) and seed=0 stall (final 0.500 → 0.900) **without invoking any P0 axis**. §8 thesis claim is now written from §7.9 directly: "**k=12 (~6s ≈ 涡街周期 30–60%) is the cross-seed sweet spot; actor-side temporal information access, not critic-side variance, is the dominant bottleneck on `single_cross_s0`**". The P0a → P0b/c/d branching table below is retained for archival.

| P0a result | P0b/c needed? | §8 thesis claim (HYPOTHETICAL — not the actual closure path) |
|---|---|---|
| CONFIRMED-VARIANCE-REDUCTION + seed=0 rescued | No | "DroQ-lite is orthogonal stabilizer on top of k=8" — clean thesis chapter |
| CONFIRMED but seed=0 still stalls | Yes (P0b/c) | "DroQ reduces variance generally; seed=0 needs higher UTD or n-step" |
| NO-EFFECT | Yes (P0b/c then P0d) | "Simple regularization insufficient; ensemble methods are required" |
| MIXED | Re-tune | "Need careful regularization tuning" |
| WORSENED | Skip to P0d | "Architectural variance reduction is the right axis" |
| **(ACTUAL outcome 2026-05-19)** k=12 alone resolves it | **No P0 needed** | "**Information bottleneck (k=8 actor temporal window ≈ vortex period 20–40%) underlies seed=0 stall; k=12 (~6s ≈ 30–60%) clears it cross-seed; variance reduction not required**" |

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

> **2026-05-19 STATUS — P0 has effectively pre-closed via §7.9.2' (k=12 seed=0 CROSS-SEED-RESCUE)**.
>
> The §7.9.2' result satisfies the conditions originally written for P0 closure **without running any P0 axis**:
> - σ_final on cross-seed sample: **0.064 ≤ 0.10 target** ✓
> - seed=0 RESCUED (final 0.500 → 0.900) ✓
> - Both seeds STRICT 5/5 PASS ✓
>
> The "P0 closes when…" conditions below were authored when k=12 seed=0 was pending. Now that the pending step came back as RESCUE, **P0 closure is satisfied by `k=12 history-length alone` — no DroQ / UTD / N-Step / REDQ axis needed**.

**Effective closure condition met (no P0 axis run)**: **"k=12 alone may already satisfy thesis σ_final target"** — confirmed empirically. §8 main writing starts directly from §7.9 cross-seed picture; P0 design retained as future polish reference, not gating.

**Original closure menu (2026-05-19 morning draft, retained for archival)** — P0 would have closed when **any one** of:

1. P0a CONFIRMED-VARIANCE-REDUCTION + seed=0 RESCUED → write "DroQ-lite is the answer"
2. P0a + P0b combination achieves σ_final ≤ 0.10 + seed=0 RESCUED → write "DroQ-lite + UTD=4"
3. P0a + P0c achieves σ_final ≤ 0.10 + seed=0 RESCUED → write "DroQ + N-Step"
4. P0d (REDQ) achieves σ_final ≤ 0.10 + seed=0 RESCUED → write "Ensemble methods"
5. All P0a-d fail → write "Vanilla SAC on cross_stream s0 has irreducible σ_final ≈ 0.18; thesis 主线 should target a different reward / sensor combination or report range-statistics instead of strict gates"
6. **(post-§7.9.2', ACTUAL outcome)** k=12 history-length alone resolves σ_final + seed=0 → write **"Information capacity, not optimization variance, was the bottleneck; k=12 (~6s ≈ vortex period 30–60%) is the cross-seed sweet spot — see §7.9"**

---

## 9. Predecessors & Cross-references

- §7.8 multi-seed (3 anchors) gate summaries: `experiments/arrival_v2_prototype/s0_cross_k8_seed{0,7}_summary/combined_gate_summary.json`
- §8 P1#2 k=12 monotonicity (seed=42 PASS, seed=0 pending): `experiments/arrival_v2_prototype/s0_cross_k12_seed{42,0}_summary/combined_gate_summary.json`
- DroQ flag surface: `auv_nav/sac.py` lines 36-47, 68-69, 81-82, 122-123, 153-154; `scripts/train_sac.py` lines 338-339, 857, 887, 891
- Build script archive: `/tmp/build_nb_s0_cross_asym/build_*.py`
- Verdict-bug lesson from §7.8'' seed=7: 4-tier schemas miss BORDERLINE-PASS; future P0 notebooks must use 5-tier including UNCLASSIFIED catch-all (precedent in `build_k12_seed0.py`).
