# FQL Succession Gate B — Final Report (Option B 2-seed)

> 📦 **已归档** — P2 之前的施工记录，2026-08-17 迁入 `docs/archive/fql_succession/`；归档只改位置与标注，**不含有效性判断**。缘由与本目录清单见 [`README.md`](README.md)。

> **Status (2026-05-20)**: FINAL — **3/4 PASS + c4 marginal-FAIL (statistically indistinguishable from 0)**, seed-driven, not FQL-driven → **progress to P2 with caveat**
>
> **Branch**: `codex-arrival-v2-prototype`
> **Notebook**: [`notebooks/fql_succession_gate_b.ipynb`](../../../notebooks/fql_succession_gate_b.ipynb) (rev.3.1 commit `e6e23ec`)
> **Colab pass artifact**: [`notebooks/fql_succession_gate_b_completed_new.ipynb`](../../../notebooks/fql_succession_gate_b_completed_new.ipynb) (Option B 4-run output)
> **Interim report (rev.1 single-seed, superseded)**: [`docs/archive/fql_succession/fql_succession_gate_b_interim_report.md`](fql_succession_gate_b_interim_report.md)
> **Spec**: [`docs/archive/fql_succession/fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) §5 (Task E)
> **Plan**: [`docs/archive/fql_succession/fql_succession_plan_v0.md`](fql_succession_plan_v0.md) §3 Lean MVP
> **Raw outputs** (gitignored, local only):
> - `results/offline/fql_succession/gate_b/{rebrac,fql}_e_uni_seed{0,42}/{test_result.json, eval_log.csv}` × 4
> - `results/offline/fql_succession/gate_b/summaries/{gate_b_verdict.json, gate_b_overview.csv}`
> **Cross-link**: [`docs/rebrac_broad_validation_v2_report.md`](../../rebrac_broad_validation_v2_report.md) (N0 anchor 0.85 reference)

---

## 目录

- [§1 Executive verdict](#1-executive-verdict)
- [§2 Setup](#2-setup)
- [§3 Per-seed + aggregated 4-criteria numbers](#3-per-seed--aggregated-4-criteria-numbers)
- [§4 Seed-spread finding (key new evidence)](#4-seed-spread-finding-key-new-evidence)
- [§5 c4 statistical noise analysis](#5-c4-statistical-noise-analysis)
- [§6 Bug status](#6-bug-status)
- [§7 Comparison with interim report](#7-comparison-with-interim-report)
- [§8 Decision & next step](#8-decision--next-step)
- [§9 P2 spec implications](#9-p2-spec-implications)
- [§10 Files / artifacts](#10-files--artifacts)

---

## 1. Executive verdict

Option B closure (4 paired runs = 2 algo × 2 seed, ~75 min L4) on E-uni 1000-ep paper anchor (`privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000`).

**Pre-registered 4-criteria verdict (aggregated per-algo across seeds)**:

| # | Metric | Value | Threshold | Pass |
|---|---|---:|---|:---:|
| c1 | FQL last-3 mean vs ReBRAC | **0.7278** vs 0.7722 (Δ=−4.4pp) | ≥ −0.08 | ✓ |
| c2 | FQL `loss_flow` last 5% | **0.0264** | < 0.05 | ✓ |
| c3 | actor_loss OOM ratio | **1.011** | ∈ [0.1, 10] | ✓ |
| c4 | FQL last-30% eval slope | **−0.0038** | ≥ 0 | ✗ |
| **OVERALL** | | | | **3/4 marginal-FAIL** |

**Three-sentence conclusion**:

1. **FQL implementation is healthy** — c1/c2/c3 all PASS with comfortable margin; flow-matching teacher converges (loss_flow 0.026 < 0.05), Q-guided distillation produces actor_loss within 1.1% of ReBRAC, and aggregated last-3 success only trails ReBRAC by 4.4pp (well within −8pp threshold).
2. **c4 FAIL is statistically indistinguishable from 0 noise** (aggregated slope −0.0038, z=−0.27 against 30-ep manifest's noise floor SE ≈0.014) and is **seed-driven, not FQL-driven**: both algos show **positive** c4 slope on seed=0 (+0.014 ReBRAC, +0.023 FQL) and **negative** slope on seed=42 (−0.013 ReBRAC, −0.030 FQL); ReBRAC's aggregated slope is also essentially zero (+0.00048). The strictly-positive c4 threshold is over-sensitive at n_seeds=2 with 30-ep eval noise (Bug 2).
3. **Recommend progress to P2 main comparison with c4-as-caveat**, per interim §8 "mixed case → caveat + 3-seed extension"; do **not** trigger Option D (FQL stability ablation) — no FQL-specific instability evidence; do trigger Bug 2 fix and c4 threshold revision before P2 sign-off.

---

## 2. Setup

**Run matrix (Option B closure)**:

| # | Algo | Dataset | Manifest | Steps | Seed | Status | Sampling |
|---|---|---|---|---|---|---|---|
| 1 | ReBRAC | `privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000` | `single_u10_cross_tgt15` | 200k | 42 | rev.1 skip-resume | uniform |
| 2 | FQL | 同上 | 同上 | 200k | 42 | rev.1 skip-resume | uniform |
| 3 | ReBRAC | 同上 | 同上 | 200k | 0 | rev.3.1 fresh | uniform |
| 4 | FQL | 同上 | 同上 | 200k | 0 | rev.3.1 fresh | uniform |

**Per-algo CLI** (common flags omitted, see notebook §2):
- ReBRAC: `--actor-penalty-coef 4.0 --critic-penalty-coef 2.0 --critic-layernorm --no-actor-layernorm`
- FQL: `--flow-steps 10 --distill-alpha-bc 1.0 --teacher-lr 3e-4 --flow-time-embed-dim 32`

**Important caveat (Bug 2)**: `--eval-episodes 100` and `--episodes 100` are silently overridden to manifest size (`single_u10_cross_tgt15` has 30 ep). All eval metrics in this report are computed against **30-episode evaluations**, contributing eval-level noise SE ≈ √(p(1−p)/30) ≈ ±8.4pp for p ≈ 0.7. This is consistent across all 4 runs so does not bias comparisons, but inflates c4 slope variance (§5).

---

## 3. Per-seed + aggregated 4-criteria numbers

### 3.1 Per-seed metrics (dedup-by-train_step applied)

| Algo | Seed | last-3 mean | last-30% slope | loss_flow tail | actor_loss tail | n_eval_pts |
|---|---:|---:|---:|---:|---:|---:|
| ReBRAC | 0 | **0.9111** | **+0.01429** | n/a | −0.9670 | 20 ✓ |
| ReBRAC | 42 | 0.6333 | −0.01333 | n/a | −0.9669 | 20 ✓ |
| FQL | 0 | 0.7667 | **+0.02286** | 0.0302 | −0.9740 | 20 ✓ |
| FQL | 42 | 0.6889 | −0.03048 | 0.0227 | −0.9814 | 20 ✓ |

**Bug 1 closure check**: all 4 runs have exactly 20 unique-step rows post-dedup (matches expected `200k / 10k = 20`). FQL seed=42 raw eval_log.csv still has 38 rows (from the rev.1 append-mode contamination), but `_dedup_by_train_step()` correctly reduces it to 20 (last-write-wins). All other runs are clean at source. Bug 1 fix is operating as designed.

### 3.2 Aggregated per-algo (mean across seeds)

| Algo | last-3 mean | last-30% slope mean | loss_flow mean | actor_loss mean | seeds |
|---|---:|---:|---:|---:|---|
| ReBRAC | **0.7722** | **+0.00048** | n/a | −0.9669 | [0, 42] |
| FQL | **0.7278** | **−0.00381** | 0.0264 | −0.9777 | [0, 42] |

### 3.3 Final canonical 100-ep eval (actually 30-ep per Bug 2)

| Algo | Seed | success | return | safety | progress | path_eff |
|---|---:|---:|---:|---:|---:|---:|
| ReBRAC | 0 | **0.8667** | +97.1 | 3.97 | 0.887 | 0.815 |
| ReBRAC | 42 | 0.6667 | +31.9 | 5.12 | 0.823 | 0.720 |
| FQL | 0 | 0.7000 | +45.4 | 4.59 | 0.857 | 0.788 |
| FQL | 42 | 0.5333 | −16.6 | 6.48 | 0.748 | 0.676 |

**Aggregated final eval (2 seeds)**: ReBRAC 0.767, FQL 0.617 — gap 15pp.
**Aggregated last-3 in-training**: ReBRAC 0.772, FQL 0.728 — gap 4.4pp (this is c1's source of truth).

The discrepancy between final-eval gap (15pp) and last-3 gap (4.4pp) reflects (a) Bug 2's per-eval noise, and (b) deliberate spec choice to use last-3 mean for c1 (spec §5.3 explicitly avoids single-shot final-eval fluctuation).

---

## 4. Seed-spread finding (key new evidence)

### 4.1 The seed-by-algo c4 cross-pattern

```
                    seed=0        seed=42       across-seed mean
ReBRAC slope:    +0.01429      −0.01333         +0.00048   (essentially 0)
FQL slope:       +0.02286      −0.03048         −0.00381   (essentially 0)
```

**Both algos pass c4 on seed=0 and fail c4 on seed=42**. This is the dispositive finding: c4 failure correlates with seed, not algorithm. If c4 reflected a FQL-specific distillation instability, FQL would fail c4 on both seeds while ReBRAC passes — that is not what we observe.

### 4.2 The seed-by-algo absolute-success cross-pattern

```
                    seed=0        seed=42       Δ(0 − 42)
ReBRAC last-3:     0.9111       0.6333         +27.8pp
FQL last-3:        0.7667       0.6889          +7.8pp
ReBRAC final-30:   0.8667       0.6667         +20.0pp
FQL final-30:      0.7000       0.5333         +16.7pp
```

**Seed=0 is a "good seed" for both algos** (~+20-28pp uplift for ReBRAC, ~+8-17pp for FQL). The seed effect is enormous on this benchmark — easily larger than any algo gap — confirming that single-seed Task E results (interim report) were under-powered.

### 4.3 FQL-vs-ReBRAC gap actually FLIPS across seeds

```
                    seed=0          seed=42       aggregated
FQL last-3:       0.7667          0.6889         0.7278
ReBRAC last-3:    0.9111          0.6333         0.7722
Δ (FQL − ReBRAC): −14.4pp         +5.6pp        −4.4pp
```

- **On the good seed (0)**: ReBRAC reaches ~0.91 ceiling, FQL trails by 14pp → ReBRAC has a real ceiling advantage when conditions allow.
- **On the bad seed (42)**: ReBRAC stalls at ~0.63 plateau, FQL slightly exceeds by 5.6pp → FQL's distillation regularization may help avoid getting trapped at low-baseline plateaus.
- **Aggregated** Δ=−4.4pp averages these two regimes. The fact that FQL outperforms ReBRAC on the harder of the two trajectories is a positive datapoint for the FQL succession story.

This sign-flip is interesting but not actionable at n_seeds=2 — could be real (different optimization basins), could be noise. P2 main comparison's 3+ seeds will resolve it.

### 4.4 Why seed=42 is the harder draw

Both algos' c4 slopes are negative on seed=42 and positive on seed=0. Yet step 199k transient critic spike (interim §4.2) was observed on FQL seed=42 only (single-seed rev.1 train log). The interim's three candidate mechanisms — (A) outlier batch, (B) target-network drift, (C) tanh squashing numeric — are all consistent with **seed-dependent rare-event Q-divergence** rather than systematic FQL instability. Seed=0's training did not reproduce the step 199k spike (final-eval drop is smaller, slope still positive).

This means the interim's diagnosis "FQL has a transient Q-instability" must be downgraded to "**this seed had a Q-instability event late in training; on the other seed, FQL converged smoothly**". Not a systematic FQL property.

---

## 5. c4 statistical noise analysis

### 5.1 Eval-level noise floor

30-ep eval with success rate ≈ 0.7:
```
σ_eval = √(p(1−p)/n) = √(0.7 × 0.3 / 30) = 0.0837
```

### 5.2 Slope SE for 6-point regression (last 30% of 20 evals)

With 6 evenly-spaced x-values, denominator = Σ(x−mean_x)² = 17.5:
```
SE(slope_per_seed)  = σ_eval / √17.5 = 0.0200
SE(slope_aggregated_2seeds) = 0.0200 / √2 = 0.0141
```

### 5.3 Significance test against null hypothesis (slope = 0)

| Quantity | Slope | z-score | Significant at 95%? |
|---|---:|---:|---|
| ReBRAC seed=0 | +0.01429 | +0.71 | NOT (\|z\| < 1.96) |
| ReBRAC seed=42 | −0.01333 | −0.67 | NOT |
| FQL seed=0 | +0.02286 | +1.14 | NOT |
| FQL seed=42 | −0.03048 | −1.52 | NOT (borderline) |
| ReBRAC aggregated | +0.00048 | +0.03 | NOT |
| **FQL aggregated** | **−0.00381** | **−0.27** | **NOT** |

**Every observed slope is within 2σ of zero given the 30-ep eval noise floor**. The pre-registered c4 threshold "slope ≥ 0" with no margin is fundamentally measuring noise at this protocol's signal-to-noise ratio.

### 5.4 Implication for c4 verdict

The c4 FAIL is a **deterministic threshold being tripped by stochastic noise that the threshold cannot distinguish from zero**. This is not a finding about FQL — it is a finding about the c4 spec under Bug 2's reduced manifest size.

For P2, c4 should be revised to one of:
- **Option α (recommended)**: `slope ≥ −2 × SE(slope_aggregated)` ≈ −0.028 with n_seeds=2 — gives a meaningful "no major collapse" test
- **Option β**: Use bootstrap CI on slope and require CI's upper bound ≥ 0
- **Option γ**: Replace linear slope with peak-stability metric (e.g., `last-3 ≥ 0.85 × max-3 in trajectory`)
- **Option δ (independent of c4 design)**: Generate `single_u10_cross_tgt15_ep100` manifest to halve eval noise → halves slope SE → makes the existing "slope ≥ 0" threshold meaningful

§9 P2 implications elaborates.

---

## 6. Bug status

### 6.1 Bug closure (interim → rev.3.1)

| Bug | Description | Fixed in | Status |
|---|---|---|---|
| 1 | eval_log.csv / train_log.jsonl append-mode cross-run contamination | notebook §5 `_dedup_by_train_step()` (rev.2) | **CLOSED** — Option B confirms dedup correctly reduces 38-row FQL seed=42 log to 20 unique-step rows; other 3 runs clean at source |
| 3 | notebook §5.5 used `eval_avg_*` instead of actual `eval_*` field names | rev.2 | **CLOSED** — all 4 field values now display correctly |
| 4 | `os.system(cmd_str)` does not stream in Colab (49 min FQL train cell completely blank) | rev.3 switch to Jupyter `!python ... \` magic | **CLOSED** — Option B run confirmed real-time stream of training loss / eval rate to cell |
| 5 | `--save-dir {ckpt_dir_str}` Colab path with space "Colab Notebooks" was shell-split, argparse rejected tail | rev.3.1 single-quote `{ckpt_dir_str}` / `{test_json_str}` | **CLOSED** — Option B seed=0 fresh runs successfully wrote to `Colab Notebooks/...` paths |

### 6.2 Bug 2 (deferred, but escalated by §5 finding)

| Bug | Description | Status |
|---|---|---|
| 2 | `--eval-episodes` / `--episodes` silently overridden by `--manifest` size (30 ep) | **DEFERRED** to P2 spec writing; **escalated by §5** — eval noise floor makes c4 threshold structurally ineffective, must fix before P2 sign-off |

Fix options for Bug 2 (decide during P2 spec writing):
- **(a) Larger manifest**: Generate `single_u10_cross_tgt15_ep100` (100 ep) → halves per-eval SE, restores c4 threshold integrity; **simplest, no code change**.
- **(b) CLI honors `--episodes`**: Modify `scripts/evaluate_offline.py` / `scripts/train_offline.py` to use `--episodes` when explicitly set rather than manifest size; **most general but touches more code, requires audit of existing benchmarks for impact**.
- Recommendation: (a) for Task E P2; (b) as a follow-up cleanup item.

---

## 7. Comparison with interim report

| Metric | Interim (seed=42, single-seed) | Final (Option B, 2-seed aggregated) | Δ / Note |
|---|---:|---:|---|
| FQL last-3 mean | 0.689 | **0.728** | +3.9pp (seed=0 lifts mean) |
| ReBRAC last-3 mean | 0.633 | **0.772** | +13.9pp (seed=0 lifts ReBRAC much more) |
| c1 gap (FQL − ReBRAC) | **+5.6pp** | **−4.4pp** | **Sign flip** — seed=42 favored FQL, seed=0 favored ReBRAC, aggregated balances |
| FQL c4 slope (clean) | −0.0305 | **−0.0038** | Magnitude shrinks 8× because seed=0 cancels |
| c2 FQL loss_flow | 0.023 | **0.026** | Stable; FQL teacher consistently converges |
| c3 actor_loss ratio | 1.007 | **1.011** | Stable; FQL/ReBRAC parity preserved |
| Verdict | 3/4 marginal-FAIL | **3/4 marginal-FAIL** | Same headline; **mechanism diagnosis flipped** |

**Key narrative shift from interim to final**:

- **Interim**: "FQL has c4 instability (slope −0.030, 9× worse than ReBRAC), possible distillation collapse"
- **Final**: "c4 instability is seed-driven; aggregate FQL slope is essentially zero (−0.0038, indistinguishable from noise); both algos exhibit the same seed-driven negative slope on seed=42 and positive slope on seed=0"

The interim's Option D trigger condition ("c4 FAIL across both seeds → FQL stability ablation") **is not met**. Recommendation: skip Option D, progress to P2.

---

## 8. Decision & next step

### 8.1 Verdict-conditional branch resolution

Per interim §8 + spec §5.3 mitigation matrix:

| Branch | Trigger | Action | This run? |
|---|---|---|---|
| Aggregated PASS | 4/4 PASS | Plan v1 → P2 spec | No (c4 fail) |
| Systematic FAIL | c4 FAIL on both seeds | Option D (FQL stability ablation) | **No** — seed=0 passes c4 |
| Implementation FAIL | c1 FAIL across seeds | Debug → STOP if persists | No (c1 PASS by 3.6pp margin) |
| **Mixed** | c4 fails one seed only | **caveat + 3-seed extension during P2** | **YES** |

### 8.2 Selected branch: **Mixed → Progress to P2 with caveat**

Concretely:

1. **Plan**: Upgrade `docs/archive/fql_succession/fql_succession_plan_v0.md` to v1.1 noting Gate B 3/4 PASS + c4 marginal status; P2 spec inherits this.
2. **Spec writing (P2 main comparison)**: Start drafting `docs/fql_succession_p2_main_spec.md` (new file). Bake in:
   - **3+ seeds per algo** as minimum (P2 cannot reproduce Task E's n_seeds=2 power deficit)
   - **Bug 2 fix as pre-requisite** (either larger manifest or CLI honors `--episodes`) — must land before any P2 sweep
   - **Revised c4 threshold** (Option α / β / γ from §5.4) — empirically calibrated, not strictly-positive
3. **Caveat in P2 introduction**: explicitly note that Task E Gate B closed with seed-spread c4 marginal-fail; 3-seed P2 results will retire-or-confirm this finding.
4. **Skip Option D** (FQL stability ablation) — no FQL-specific instability evidence under aggregated-across-seeds reading.

### 8.3 What is NOT triggered

- ~~Option D (FQL stability ablation: critic LN / tau=0.001 / 100k step / actor warm-up)~~ — premise violated (seed=0 FQL slope is positive)
- ~~STOP / R4 mitigation~~ — c1/c2/c3 all PASS, FQL implementation is healthy
- ~~N0 1000-ep audit as paper appendix~~ — defer to after P2 main comparison closure; Task A audit tool can be invoked then for paper §appendix evidence

---

## 9. P2 spec implications

Carry-forward items for `docs/fql_succession_p2_main_spec.md` drafting:

| Item | Source | P2 action |
|---|---|---|
| **n_seeds ≥ 3 mandatory** | §4 huge seed spread (20-28pp) | Hard requirement; budget P2 wallclock accordingly |
| **Bug 2 fix pre-requisite** | §5 noise analysis | Generate `single_u10_cross_tgt15_ep100` manifest **before** any P2 sweep starts; OR fix CLI override |
| **c4 threshold revision** | §5.4 | Pick Option α/β/γ + sanity-check on n_seeds=3 mock data |
| **Spec CLI rename** | notebook header table | Patch `docs/archive/fql_succession/fql_succession_p0p1_spec.md` §5.1/§5.2 field names to match actual code (`--algo` not `--algorithm`, etc.) — **landed during P2 spec writing** |
| **Dataset partial-obs ceiling finding** | Interim §7 + Final §3.3 | Discussion-quality finding: privileged-policy dataset under s0 sensor caps at ~0.6-0.8 vs crosscomp 0.85 on broad val v2 N0. Belongs in P2 §discussion as "oracle teacher under s0 sub-critical gap" — independent of FQL verdict, useful framing for the sim2real story |
| **200k uniform vs 22k shuffle** | Interim §7.2 | Both algos show slight negative tendency at 200k tail (ReBRAC agg +0.00048, FQL agg −0.00381) — consider 100k step in P2 OR best-last-k metric instead of last-3 |
| **FQL Q-instability transient (step 199k spike)** | Interim §4.2, single-seed=42 observation | Re-evaluate under P2 3-seed data; if it reproduces on 2+/3 seeds, escalate to FQL stability tuning |

---

## 10. Files / artifacts

### 10.1 This-report production

| Path | Purpose | Tracked? |
|---|---|---|
| `docs/archive/fql_succession/fql_succession_gate_b_report.md` | This document (final) | **yes** |
| `docs/archive/fql_succession/fql_succession_gate_b_interim_report.md` | rev.1 single-seed interim (superseded but kept for traceability) | yes (preserved) |
| `notebooks/fql_succession_gate_b.ipynb` | rev.3.1 driver notebook | yes |
| `notebooks/fql_succession_gate_b_completed_new.ipynb` | Option B 4-run Colab pass artifact | yes (this commit) |

### 10.2 Raw results (gitignored — `results/` is in .gitignore by convention)

| Path | Size |
|---|---|
| `results/offline/fql_succession/gate_b/{rebrac,fql}_e_uni_seed{0,42}/test_result.json` | 4 files, ~1 KB each |
| `results/offline/fql_succession/gate_b/{rebrac,fql}_e_uni_seed{0,42}/eval_log.csv` | 4 files, 20 rows each (FQL seed=42 has 38 raw rows that dedup to 20) |
| `results/offline/fql_succession/gate_b/summaries/gate_b_verdict.json` | per-seed + aggregated + 4-criteria pass/fail block |
| `results/offline/fql_succession/gate_b/summaries/gate_b_overview.csv` | flat per-(algo, seed) row format |

### 10.3 Checkpoint state (Drive only, gitignored)

| Path | Purpose |
|---|---|
| `checkpoints/.../{rebrac,fql}_e_uni_seed{0,42}/agent_final.pt` | Final agent weights × 4 |
| `checkpoints/.../{rebrac,fql}_e_uni_seed{0,42}/trainer_state.json` | Resume metadata × 4 |
| `checkpoints/.../{rebrac,fql}_e_uni_seed{0,42}/train_log.jsonl` | Loss metrics per log-step × 4 |
| `checkpoints/.../{rebrac,fql}_e_uni_seed{0,42}/eval_log.csv` | In-training eval × 4 (source for `results/` copies) |

P2 may reuse these checkpoints for warm-start or anchor diagnostics; do not delete the Drive tree before P2 closure.

---

**Report finalized**: 2026-05-20
**Notebook baseline**: `e6e23ec` (rev.3.1)
**Reproducibility**: `notebooks/fql_succession_gate_b_completed_new.ipynb` (committed) + `notebooks/fql_succession_gate_b.ipynb` (rev.3.1) + `results/offline/fql_succession/gate_b/` (gitignored, on Drive)
**Next concrete action**: draft `docs/fql_succession_p2_main_spec.md` (P2 main comparison), gated on Bug 2 fix decision + c4 threshold revision design
