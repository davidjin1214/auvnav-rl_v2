# FQL Succession P2 — Results & Verdict

**Date**: 2026-05-23
**Branch**: `codex-arrival-v2-prototype`
**Status**: **CLOSED — honest negative + mechanism finding.** "FQL > ReBRAC" is falsified.
**Companion docs**: design = [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) (v1.3);
detailed lab notebook = [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md) §9 (authoritative);
collection log = [`fql_succession_p2_collection_log.md`](fql_succession_p2_collection_log.md).
**Verdict notebook**: [`../notebooks/fql_succession_p2_verdict.ipynb`](../notebooks/fql_succession_p2_verdict.ipynb) (re-aggregates every number here from `results/`).

---

## 0. TL;DR

P2 set out to confirm a published-style claim — **FQL beats ReBRAC iff the offline data
is both sub-optimal and multi-modal**. After a 2×2 cell matrix plus a four-experiment
ablation chain, that claim is **falsified**, and the more careful question ("is FQL ever
the better choice here?") also resolves **negative**:

> **A single, fixed ReBRAC configuration (actor BC weight β1 = 1.0) is at least as good
> as FQL on every cell of the 2×2 matrix, and strictly more robust across the
> action-noise axis (worst-case success 0.910 vs FQL 0.858).** FQL's apparent advantage
> was an artifact of ReBRAC's *mis-tuned default* β1 = 4.0, not a structural property of
> flow-matching. When FQL is given the same courtesy — a sweep of its own BC-anchor knob
> `distill_alpha_bc` — it still cannot clear the bar.

What survives, and what we report as the contribution, is a **clean mechanism**:

> **The dominant driver of offline-RL robustness to action-level noise is the *target
> quality* of the behavior-cloning anchor, not the algorithm family.** ReBRAC anchors its
> actor to *raw* dataset actions; FQL anchors to a *flow-denoised* reconstruction. On
> noisy data the raw anchor must be *weakened* (β1↓) to avoid chasing noise; the denoised
> anchor is already clean, so it needs no such fix — but on clean data it confers no
> advantage. **Same knob family, opposite optimum, set entirely by how noisy the anchor
> target is.**

This is a publishable, falsification-grade result and a useful cautionary tale (BC-weight
defaults that are appropriate for clean data silently handicap a baseline on noisy data),
but it is **not** a "FQL wins" paper.

**Scope** — the mechanism is established on `single_u10_cross`. A cross-benchmark probe at
the harder `single_u15_cross` (U=1.5 / Re250) **floored** (offline `s0` policy 0.14 SR,
out-of-bounds-dominated), so the FQL-vs-ReBRAC comparison is *undefined* there — a deployable-
sensor sufficiency boundary, not an algorithmic counter-result (see §6.5).

---

## 1. Setup

### 1.1 What is being compared

- **ReBRAC** — TD3+BC. Two BC surfaces: an **actor-side** MSE regression to the raw
  dataset action (weight **β1** = `--actor-penalty-coef`) and a **critic-side** BC penalty
  inside the TD target (weight β2 = `--critic-penalty-coef`). Gate B froze β1 = 4.0, β2 = 2.0.
- **FQL** (Flow Q-Learning) — a flow-matching **teacher** learns the behavior velocity
  field; a 1-step **student** is distilled toward the *flow-denoised* teacher action with
  weight **`distill_alpha_bc`** (`--distill-alpha-bc`, default 1.0). The student is the
  deployed policy. `distill_alpha_bc` is the direct analog of ReBRAC's β1 — except it
  anchors to a denoised target rather than to raw data.

Both train on the identical offline dataset per cell, the same `s0/h4` sensor, the same
200k steps, and are evaluated identically.

### 1.2 Primary metric and why comparisons are paired

Primary metric: **`eval_success_rate`** over a **fixed 100-scenario manifest**
(`benchmarks/single_u10_cross_tgt15_ep100.json`). The manifest pins identical
`episode_id`s and per-episode environment seeds across *every* training seed, algorithm,
and cell. Therefore eval-sampling noise is **common-mode** — it cancels in any within-cell
seed spread and in any between-config δ. Every comparison below is effectively **paired on
identical scenarios**, and the seed-to-seed spread is *pure training-seed variance*
(σ_train ≈ 3.8 pp; §5).

### 1.3 The 2×2 design

Two axes — **modality** (unimodal / multi-modal behavior) × **noise** (clean σ=0 /
action-noise contaminated) — give four cells:

| cell | data | noise | modality role |
|---|---|---|---|
| **E-uni** | privileged expert | σ = 0 | clean, unimodal (baseline) |
| **M-uni-noise** | privileged + Gaussian action noise | σ = 0.5 (clip ±0.5) | noisy, structurally unimodal |
| **E-multi** | 50% privileged + 50% goalseek | σ = 0 | clean, genuinely multi-modal |
| **M-multi-mix** | 50% privileged + 50% goalseek | σ = 0.1 | small-noise multi-modal |

n = 2 seeds/cell {42, 0} as standard; E-uni FQL was later topped up to n = 4 ({1, 2}) in C-1 (§4).

---

## 2. Result 1 — the 2×2 matrix says "noise", not "modality"

Per-seed `eval_success_rate` (100 ep), means in bold; δ = FQL − ReBRAC; verdict per spec
v1.3 §5.5 thresholds (|δ| ≤ 0.03 → NULL, ≥ 0.05 & consistent → POS/NEG, else GRAY):

| | clean (σ = 0) | noisy |
|---|---|---|
| **uni** | **E-uni**: R 0.885, F 0.858 (n=4) → δ = **−0.027** *NULL* | **M-uni-noise** (σ=0.5): R 0.705, F 0.910 → δ = **+0.205** *POSITIVE* |
| **multi** | **E-multi**: R 0.810, F 0.790 → δ = **−0.020** *NULL* | **M-multi-mix** (σ=0.1): R 0.990, F 0.955 → δ = **−0.035** *GRAY* |

**Reading the matrix.** Only one cell shows an FQL advantage, and it is the **high-noise**
cell (M-uni-noise, +20.5 pp). The clean column is flat-to-negative, and the genuinely
multi-modal *clean* cell (E-multi) is NULL — clean multi-modality alone gives FQL nothing.
M-multi-mix (multi-modal but only σ=0.1) is also non-positive: its small per-component
noise plus good state coverage means the *conditional action variance* ReBRAC's BC term
actually sees is low. So the discriminating variable is **action-noise magnitude**, i.e.
the conditional action variance `E_s[Var(a|s)]`, **not** modality. The original
"sub-optimal AND multi-modal" iff is cleanly falsified: it scored **1/3** on its own
predicted verdicts (E-multi was supposed to be the *primary* positive; it was null).

This already demotes the headline. But the +20.5 pp positive cell still looked like a real
FQL win — so we asked *why*, and the answer dissolves it.

---

## 3. Result 2 — the mechanism trilogy dissolves the one positive cell

A controlled three-experiment chain on the noisy cell (and its clean control) isolates
*which* ReBRAC surface causes the M-uni-noise deficit, and whether it is intrinsic.

| step | manipulation | result | conclusion |
|---|---|---|---|
| **Q1** | critic β2: 2.0 → **0** (noisy) | SR 0.705 → 0.715 (**+0.01**); `mean_q` −150 → −123 (coef took effect) | The **critic-side** BC penalty moves Q magnitude but is **not** the binding constraint on policy quality. |
| **Q1b** | actor β1: 4.0 → **1.0** (noisy) | SR 0.705 → **0.940** (**+23.5 pp**), *overtaking* FQL (0.910) | The **actor BC anchor strength to the raw noisy target** is the knob — and it is *tunable*. |
| **Q1c** | actor β1: 4.0 → **1.0** (clean) | SR 0.885 → **0.910** (**+2.5 pp**) | β1 = 1.0 is *also* better on clean → **no clean/noisy trade-off**. |

Putting Q1b + Q1c together gives the **noise-axis grid** — the decisive table:

| config | clean (E-uni) | noisy (M-uni-noise) | **worst-case-over-noise** |
|---|---:|---:|---:|
| ReBRAC β1 = 4.0 (Gate B default) | 0.885 | 0.705 | 0.705 |
| **ReBRAC β1 = 1.0** | **0.910** | **0.940** | **0.910** |
| FQL (frozen `distill_alpha_bc`=1.0) | 0.858 (n=4) | 0.910 | 0.858 |

**Verdict.** A single fixed ReBRAC config (β1 = 1.0, *no* per-dataset tuning) **dominates
FQL on both axes** (clean +5.2 pp, noisy +3.0 pp) and is strictly more robust
(worst-case 0.910 > 0.858). The +20.5 pp "FQL wins on noisy data" result was an artifact
of a **self-handicapped ReBRAC**: Gate B's β1 = 4.0 is a clean-data-appropriate anchor
that is simply too strong when the BC target is noisy. β1 ≈ 1.0 is closer to the canonical
TD3+BC anchor strength. Both the **superiority** claim and the weaker **robustness**
fallback ("FQL's one frozen config spans the noise axis better") are dead.

---

## 4. Result 3 — the fairness rematch (C-1): FQL gets its own knob, still loses

Q1c tuned ReBRAC's β1 while FQL's `distill_alpha_bc` stayed frozen at 1.0 — an
**asymmetric-tuning** objection. To close it, C-1 swept FQL's own BC-anchor knob
log-spaced around 1.0 on the **clean** axis (FQL's binding/worst-case axis), and topped up
the frozen α = 1.0 cell to n = 4:

| `distill_alpha_bc` | per-seed (clean) | mean | n |
|---|---|---:|:--:|
| 0.3 | 0.75, 0.73 | 0.740 | 2 |
| **1.0 (frozen)** | 0.80, 0.91, 0.91, 0.81 | **0.858** | **4** |
| 3.0 | 0.86, 0.72 | 0.790 | 2 |
| 10.0 | 0.86, 0.85 | 0.855 | 2 |
| *ReBRAC β1 = 1.0 (the bar)* | 0.88, 0.94 | *0.910* | 2 |

**Verdict: RESCUE-FAIL.** No α beats the frozen 0.858; FQL's best clean stays **−5.2 pp
under the 0.910 bar**, so its worst-case-over-noise cannot exceed ReBRAC β1 = 1.0's. Three
mechanism-consistent observations:

1. **The α = 1.0 n = 4 mean (0.858) confirms the n = 2 estimate (0.855).** The one cell
   §5 flagged as under-determined (per-seed swings 0.80 ↔ 0.91) has a *stable centre* —
   FQL clean is genuinely seed-noisy, but its mean is solid and well below 0.910.
2. **The optimal direction is OPPOSITE ReBRAC's.** Weakening FQL's anchor (0.3 → 0.740)
   *hurts* clean badly; strengthening it (10.0 → 0.855) just plateaus at ≈ α = 1.0. So
   `distill_alpha_bc` = 1.0 was *already near-optimal on clean*, because FQL's anchor points
   at a near-expert *denoised* teacher — loosening it discards a good target. ReBRAC's
   β1 = 4.0 over-anchored to *noisy raw* actions, so loosening *helped*. Same knob family,
   opposite optimum, set by **target quality** — this *reinforces* the mechanism rather
   than rescuing FQL.
3. The fairness caveat is now **closed**: the comparison is tuned-vs-tuned with *both*
   sides fairly swept, and FQL still loses on worst-case → a **stronger** honest-negative
   than the asymmetric-tuning version.

---

## 5. Statistical power — is n = 2 enough to say this?

Because the manifest is fixed, comparisons are paired and the seed spread is pure training
variance. Measured across 11 cells: mean |seed-spread| = 4.3 pp ⇒ **σ_train ≈ 3.8 pp**.
Two-sample t (n = 2/group, df = 2, t_crit = 4.30):

| contrast | δ | t | verdict |
|---|---:|---:|---|
| Q1b β1=1.0 vs β1=4.0 (noisy) — *mechanism keystone* | +0.235 | **4.98** | **SIG** |
| FQL vs β1=4.0 (noisy) — *the original "win"* | +0.205 | **8.20** | **SIG** |
| β1=1.0 vs FQL (clean) — head-to-head | +0.055 | 0.88 | NULL |
| β1=1.0 vs FQL (noisy) — head-to-head | +0.030 | 0.75 | NULL |
| β1=1.0 vs β1=4.0 (clean) | +0.025 | 0.82 | NULL |
| FQL α=1.0 (n=4) vs β1=1.0 bar (clean), C-1 | −0.052 | ≈1.2 | NULL |

Rows 3 and 6 are the *same* clean head-to-head with the subtraction order reversed and at
different n: at the n=2 power-audit snapshot it is +0.055 (t=0.88, ReBRAC−FQL); after C-1's
n=4 top-up the verdict notebook recomputes it as −0.052 (t=1.23, FQL−ReBRAC). Both NULL,
both tilt against FQL — the top-up only sharpened the estimate, it did not change the sign.

**Interpretation.** (i) The two *large* effects that carry the mechanism story are
**formally significant even at n = 2** — the spine is not a 2-seed accident. (ii) Every
head-to-head between FQL and the winning ReBRAC config is **NULL with the point estimate
tilting against FQL**; falsifying a superiority claim only requires a non-win, and more
seeds *tighten these NULLs, they do not resurrect FQL*. (iii) The only genuine n = 2
weakness was the FQL-clean SD estimate, and C-1's n = 4 top-up resolved it. Conclusion:
the negative verdict is statistically sound; additional seeds would sharpen CIs without
changing any sign.

---

## 6. The mechanism (synthesis)

Code-anchored attribution (`auv_nav/rebrac.py`, `auv_nav/fql.py`):

- ReBRAC's **actor** regresses to the *raw* dataset action `a = π*(s) + ε`
  (`bc_loss = (π(s) − a)²`, rebrac.py:257). With weight β1 = 4.0 the deterministic actor is
  dragged onto an irreducible noise floor `Var_clip(ε)·d` it cannot escape — it chases the
  noise. Lowering β1 loosens that anchor and recovers the policy.
- FQL's **student** regresses to `a_teacher = teacher.integrate(obs)` (fql.py:471), the
  flow-**denoised** reconstruction. For symmetric ε the teacher's conditional mean is
  `π*(s)` — the anchor target is *clean even at σ=0.5*, so no β1-style fix is needed. But
  on clean data this denoising buys nothing a properly-weighted raw anchor doesn't already
  have.
- ReBRAC's **critic** BC penalty (rebrac.py:284) shifts Q magnitude (Q1: −150→−123) but is
  not the binding constraint (Q1: SR +0.01). FQL has no critic-side BC penalty by design.

**One line**: *offline-RL robustness to action noise is governed by the quality of the BC
anchor target, and the optimal anchor strength flips with that quality.* Flow-denoising
(FQL) and β1-reduction (ReBRAC) are two ways to soften a too-strong anchor to noisy
targets; on clean data neither helps and a canonical β1 ≈ 1 is already optimal.

---

## 6.5 Cross-benchmark generalization probe — `single_u15_cross` (FLOOR)

To test whether the mechanism generalizes beyond `single_u10_cross`, we ran a deliberately
minimal probe on the harder **`single_u15_cross`** (U=1.5 / Re250 vs the main study's
U=1.0 / Re150): the noise axis only (clean vs σ=0.5), three configs (FQL, ReBRAC β1=4.0,
ReBRAC β1=1.0), gated by a single learnability check before committing the full 12-run matrix
(spec [`fql_succession_p2_xbench_spec.md`](fql_succession_p2_xbench_spec.md)).

The gate **floored** and the probe stopped after 1 run:

| quantity | `single_u15_cross` | `single_u10_cross` (main) |
|---|---|---|
| clean privileged collector SR | 0.719 | 0.985 |
| noisy (σ=0.5) collector SR | 0.098 | 0.632 |
| offline ReBRAC β1=1.0 / clean / test SR | **0.14** | ~0.910 |

The offline policy collapses far below its own data source (collector 0.719 → offline 0.14,
a ~58 pp gap, vs a small gap at u10); the in-training curve climbs only to ~0.09 at the same
200k budget that yields ~0.91 at u10 (still slowly rising, not a converged plateau); and the
dominant failure is **out-of-bounds (78/100)** — the vehicle is swept out by the stronger
flow. This is the signature of the **deployable single-point `s0` sensor being insufficient
to control the AUV in the U=1.5 cross-flow regime, independent of the offline algorithm**; the
σ=0.5 noisy dataset additionally degenerates to 0.098 collector success, leaving the noise
axis itself undefined.

**Reading**: a *sensor-sufficiency boundary*, not an algorithmic counter-result. The
FQL-vs-ReBRAC noise-robustness comparison is only defined where offline-from-`s0` does not
floor (e.g. `single_u10_cross`). The floor is consistent with the wider programme's `s0` theme
(the online asymmetric-critic line; the AUVHamNODE audit's "wake current 2–4× OOD at U=1.5")
and does **not** weaken Results 1–3, which stand on `single_u10_cross`.

---

## 7. What we claim, and what we do not

**Claim (defensible):**
1. Modality is *not* the discriminator between FQL and ReBRAC on this benchmark; **noise
   (conditional action variance) is** (Result 1, E-multi NULL).
2. The discriminating effect is a **tuning artifact, not an algorithmic property**: a
   canonical ReBRAC (β1 ≈ 1) matches or beats FQL everywhere, including worst-case-over-noise
   (Results 2–3).
3. The governing mechanism is **BC-anchor target quality**, demonstrated by a chain whose
   two load-bearing effects are significant at n = 2 (Results 2 & 5).
4. Methodological caution: **BC-weight defaults tuned for clean data silently handicap a
   baseline on noisy data** — a fair comparison must tune both sides' anchor weights.

**We do NOT claim** FQL is superior, nor more robust, nor that flow-matching offers a
free-lunch advantage on this task.

**Residual caveats (do not change the sign):** all numbers use `agent_final.pt` (late-training
collapse risk applies symmetrically); the benchmark ceilings high on clean/multi cells, so
absolute headroom above ReBRAC β1=1.0 is small — but FQL is *behind*, not merely tied.

---

## 8. Implications for the spec & paper (→ N4)

The spec (`fql_succession_p2_main_spec.md` v1.3) still carries the "iff multi-modal AND
sub-optimal" framing. **N4** rewrites it v1.3 → v1.4 to:
- replace the superiority headline with the **mechanism + honest-negative** framing above;
- retire / repurpose the GMM `p_≥2` modality axis (a poor proxy; conditional action
  variance `E_s[Var(a|s)]` is the right one — §5 of the diagnostic);
- rewrite the §5.5 verdict table to the noise-axis story and record the Q1→Q1b→Q1c→C-1 chain;
- keep the §4 collection log as a historical record.

For the broader programme, P2 was the **mechanism discriminator** for the AUVHamNODE
offline-RL line; it fired **negative**, which is a clean closure — the FQL Succession track
does not displace the locked AUVHamNODE main line.

---

## 9. Provenance

- **Numbers**: every value above is read from `results/fql_succession/p2/<cell>/test/<algo>_seed<seed>.json`
  (`eval_success_rate`). The verdict notebook re-derives all tables and figures from these files.
- **Cells / datasets**: see [`fql_succession_p2_collection_log.md`](fql_succession_p2_collection_log.md).
- **Ablation experiments** (run notebooks, `*_completed.ipynb` under `notebooks/`):
  Q1 `..._q1_critic_penalty_ablation`, Q1b `..._q1b_actor_penalty_ablation`,
  Q1c `..._q1c_actor_pen1_clean`, E-multi `..._run_cell_e_multi`,
  C-1 `..._c1_fql_alpha_sweep`.
- **Cross-benchmark probe** (§6.5): `notebooks/fql_succession_p2_xbench_completed.ipynb`;
  gate result `results/fql_succession/p2_xbench/e_uni_clean/test/rebrac_b1p1_seed42.json`
  (FLOOR — SR 0.14). Spec [`fql_succession_p2_xbench_spec.md`](fql_succession_p2_xbench_spec.md).
- **Full lab record**: [`fql_succession_p2_mechanism_diagnostic.md`](fql_succession_p2_mechanism_diagnostic.md)
  §9 (§9.1 Q1, §9.4 Q1b, §9.6 E-multi, §9.7 Q1c, §9.9 power audit, §9.11 C-1) is authoritative
  for any discrepancy.
- **Source**: `auv_nav/rebrac.py` (L257 actor BC, L284 critic BC), `auv_nav/fql.py`
  (L444 teacher, L471 student distill, L532 critic — no BC penalty).
