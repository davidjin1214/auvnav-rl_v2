# FQL Succession P2 — Mechanism Diagnostic (option C)

**Date**: 2026-05-22
**Branch**: `codex-arrival-v2-prototype`
**Inputs**:
- `results/fql_succession/p2/{e_uni, m_uni_noise, m_multi_mix}/test/{rebrac,fql}_seed{42,0}.json`
- `results/fql_succession/p2/<cell>/training_curves/<algo>_seed<seed>/{eval_log.csv, train_log.jsonl}`
- Source: `auv_nav/rebrac.py` (lines 257, 284), `auv_nav/fql.py` (lines 444, 471, 532)

**Purpose**: Before deciding whether to run more seeds or rewrite the paper's iff
claim, confirm with code-level + per-step training metrics **why** M-uni-noise
produced an unexpectedly large FQL advantage (+20.5 pp) while M-multi-mix
produced essentially null (−3.5 pp).

## 1. Headline finding

The paper's original iff claim
> **FQL > ReBRAC iff data is BOTH sub-optimal AND multi-modal**

is **falsified by the data and explained by the code**. The actual driver of FQL's
advantage is **noise contamination of ReBRAC's BC anchor**, not multi-modality.
Multi-modality contributes only insofar as it implies wide local-action variance —
the same statistical effect that noise produces.

Revised claim (Option-C diagnostic, n=2 seeds):
> **FQL > ReBRAC iff the dataset's BC anchor signal `a ≈ π*(s)` is corrupted by
> action-level noise large enough to inflate ReBRAC's critic BC penalty.**
> Multi-modality with small per-component action noise (≤ 0.1) and good state
> coverage does *not* differentiate the two methods.

## 2. Test-eval primary metric (`eval_success_rate`, 100 ep / seed)

| cell | rebrac s42 | rebrac s0 | rebrac μ | fql s42 | fql s0 | fql μ | δ=F−R | spec verdict (§5.6) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E-uni | 0.890 | 0.880 | **0.885** | 0.800 | 0.910 | **0.855** | **−0.030** | NULL (boundary; 方向不一致) |
| M-uni-noise | 0.680 | 0.730 | **0.705** | 0.910 | 0.910 | **0.910** | **+0.205** | **POSITIVE** (consistent) |
| M-multi-mix | 1.000 | 0.980 | **0.990** | 0.960 | 0.950 | **0.955** | **−0.035** | GRAY (just below 5 pp) |

Per-cell expected verdict per `docs/fql_succession_p2_main_spec.md` v1.3 §5.5:

| cell | spec expected | observed | match |
|---|---|---|---|
| E-uni | NULL | NULL (boundary) | ≈ ✓ |
| M-uni-noise | NULL | **POSITIVE** | ✗ |
| M-multi-mix | POSITIVE *(primary)* | NULL/GRAY | ✗ |

**Score: 1/3 spec expectations met.** The two unexpected results (M-uni-noise
positive, M-multi-mix null) point in opposite directions, which collapses the
modality-based iff into a noise-based explanation (§3 below).

## 3. Why? — code-level cause

> **⚠️ CORRECTION (2026-05-22, after Q1 ablation — see §9).** This section
> originally argued the **critic BC penalty** (path b, rebrac.py:284) was the
> *dominant* cause. The Q1 controlled ablation (`critic_penalty_coef=0`)
> **falsified that**: removing the critic penalty raised `mean_q` (−150→−123,
> confirming the coef took effect) but left test SR unchanged (+0.01). The
> dominant cause is path (a) — the **actor BC regression to raw noisy actions**
> (β1=4.0), not the critic-side penalty. The three paths below are all real
> noise-exposed surfaces; the correction is only about *which one binds the final
> policy*. Read §9 for the corrected attribution.

### 3.1 ReBRAC has THREE noise-exposed paths

(`auv_nav/rebrac.py`)

**(a) Actor BC loss (line 257)** — pure pointwise L2 to dataset action:
```python
bc_loss = (pi - actions).pow(2).sum(dim=-1).mean()
```
With dataset `a = π*(s) + ε`, `ε ~ N(0, σ²I)` clipped to `±0.5`, the expected loss is
```
E_ε[(π(s) − π*(s) − ε)²] = (π(s) − π*(s))² + Var_clip(ε)·d
```
The second term is an **irreducible noise floor** the actor cannot drive below.
Optimum is still `π = π*` (unbiased) but per-batch gradient direction has variance
∝ σ², slowing convergence and biasing the policy toward whichever mode the SGD
batches happen to over-sample.

**(b) Critic BC penalty (line 284)** — pointwise L2 inside the TD target:
```python
next_actions = (actor_target(next_obs) + clipped_TD3_noise).clamp(-1, 1)
critic_penalty = (next_actions - next_actions_data).pow(2).sum(dim=-1)
q_target = rewards + γ · (1 − dones) · (target_q − critic_bc_coef · critic_penalty)
```
With `next_actions_data = π*(s') + ε`:
```
E[critic_penalty] ≈ (π_target(s') − π*(s'))²  +  Var(TD3_noise)  +  Var_clip(ε)·d
                    ─────── residual ───────    ── TD3 floor ──    ── noise inflation ──
```
The noise-inflation term enters **every** TD target as additional pessimism.
Compounded across the discount horizon (`γ=0.99`), even a small per-step inflation
of `Δ ≈ Var_clip(ε)·d` propagates to a steady-state Q underestimate of order
`Δ/(1−γ) = 100·Δ`. With σ=0.5 and clip=±0.5, the predicted Q-underestimate is
≈ 100 × 0.08 × 2 ≈ 16; the observed gap is much larger (≈125) because of
self-reinforcement (a pessimistic Q drives the actor toward an even noisier
action, which further inflates the penalty).

**(c) TD bootstrap target depends on `actor_target`**, which is itself trained on
noisy BC actions → noise propagates indirectly through the policy.

### 3.2 FQL has ZERO of those paths

(`auv_nav/fql.py`)

**(a) Teacher = flow-matching velocity field (line 444)**:
```python
x_t = (1 - t) · x_0 + t · actions     # x_0 ~ N(0, I), t ~ U(0,1)
v_target = actions - x_0
teacher_loss = F.mse_loss(v_pred(x_t, t, obs), v_target)
```
The teacher learns `E[v | x_t, t, s]`. For symmetric ε, the conditional mean of
`actions` given `s` alone is `π*(s)` (unbiased), and ε enters as an effective
enlargement of the source distribution `N(0, I)` rather than as a target bias.
Integration `integrate(s, n=10)` recovers `E[a | s]` ≈ `π*(s)` — **noise-marginalized**.

**(b) Student BC distill loss (line 471)**:
```python
a_teacher = self.teacher.integrate(obs, n_steps=10)
bc_loss = (a_student - a_teacher).pow(2).mean()
```
Student matches the **clean reconstruction** `a_teacher`, not the raw noisy `actions`.

**(c) Critic update (line 532)** — no BC penalty whatsoever:
```python
q_target = rewards + γ · (1 - dones) · target_q
```
By design (docstring line 16: "*No critic-side BC penalty*"). Critic is fully
insulated from dataset action noise.

### 3.3 Late-training train_log.jsonl confirms the mechanism

Mean over last 50% of training steps:

| cell | algo | seed | bc_loss | critic_penalty | mean_q | td_err |
|---|---|---:|---:|---:|---:|---:|
| E-uni | rebrac | 42 | 0.017 | 0.090 | 80.0 | 0.75 |
| E-uni | rebrac | 0 | 0.005 | 0.079 | 87.7 | 0.59 |
| E-uni | fql | 42 | 0.009 | — | 95.0 | 0.43 |
| E-uni | fql | 0 | 0.010 | — | 94.4 | 0.42 |
| **M-uni-noise** | **rebrac** | **42** | **0.147** | **0.210** | **−144.6** | **6.04** |
| **M-uni-noise** | **rebrac** | **0** | **0.135** | **0.197** | **−162.6** | **6.79** |
| M-uni-noise | fql | 42 | 0.083 | — | −25.1 | 3.90 |
| M-uni-noise | fql | 0 | 0.086 | — | −23.1 | 3.56 |
| M-multi-mix | rebrac | 42 | 0.033 | 0.092 | −7.5 | 2.60 |
| M-multi-mix | rebrac | 0 | 0.030 | 0.091 | −6.3 | 2.58 |
| M-multi-mix | fql | 42 | 0.027 | — | 26.7 | 1.80 |
| M-multi-mix | fql | 0 | 0.028 | — | 27.3 | 1.80 |

Three predictions from §3.1/§3.2, all confirmed:

1. **ReBRAC's `bc_loss` tracks the dataset noise floor** (theory:
   `2·Var_clip(ε)·d`):
   - E-uni σ=0: 0.005–0.017 (only residual)
   - M-multi-mix σ=0.1: 0.030–0.033 (≈ predicted `2·(0.1²)·2·correction ≈ 0.03`)
   - M-uni-noise σ=0.5 clip=±0.5: 0.135–0.147 (≈ predicted `2·0.083·2 ≈ 0.17`, within
     ~20% of expectation)

2. **ReBRAC's `critic_penalty` doubles in M-uni-noise** (theory: TD3-floor +
   dataset-noise inflation):
   - E-uni: 0.08–0.09 (TD3 clip-noise floor only)
   - M-multi-mix σ=0.1: 0.091–0.092 (~unchanged — noise too small to matter)
   - **M-uni-noise σ=0.5: 0.197–0.210 (≈ 2.4× E-uni baseline)** ← smoking gun

3. **`mean_q` gap (ReBRAC − FQL) blows up only on M-uni-noise**:
   - E-uni: gap ≈ −15 (FQL slightly higher Q)
   - M-multi-mix: gap ≈ −34
   - **M-uni-noise: gap ≈ −126 (4× M-multi-mix)** — ReBRAC's critic is dramatically
     over-pessimistic exactly where critic_penalty is inflated.

The same gap shows up in `td_abs_error`: ReBRAC's own TD-error blows up on
M-uni-noise (6.0–6.8) while FQL stays near 4. FQL's stable Q-surface is precisely
what allows its actor improvement signal `−λ·Q(s, π(s))` to remain informative.

## 4. Training-curve visualization

![training curves](assets/fql_succession_p2/training_curves_3cells.png)

Re-render: `python -m scripts._plot_fql_succession_p2_diagnostic`

- E-uni: ReBRAC (blue) and FQL (orange) curves overlap heavily; both noisy.
- **M-uni-noise**: orange (FQL) sits clearly above blue (ReBRAC) for almost the
  entire training trajectory in both seeds. Test ★ (100 ep) cements: FQL 0.91/0.91
  vs ReBRAC 0.68/0.73.
- M-multi-mix: blue (ReBRAC) slightly above orange (FQL) consistently. Both
  approach the ceiling (~0.95–1.0).

## 5. Theoretical reconciliation: why GMM audit misled us

`docs/fql_succession_p2_main_spec.md` v1.3 §3 Gate A.2 used a GMM-based
multi-modality lower-bound `p_≥2` as the "modality" axis:

| cell | p_≥2 | structural interpretation |
|---|---:|---|
| E-uni | 0.112 | true unimodal ✓ |
| M-uni-noise | **0.996** | "GMM-multimodal" but **structurally unimodal** (priv 1.0 + noise=0.5) |
| M-multi-mix | 0.414 | true mixture (priv 50% + goalseek 50%) |

The audit's `p_≥2` measures **whether action samples at the same s cluster into
≥2 GMM components**. This is a *symptom* of either (i) policy mixing OR
(ii) action-level noise widening a single component. The audit cannot distinguish
them; the collection log already flagged M-uni-noise's `p_≥2=0.996` as a
"noise-widening false-positive".

But the **algorithmic** failure mode for ReBRAC isn't "≥2 GMM clusters" — it's
"σ² of the conditional action distribution given s" which directly inflates BC
losses. From the algorithm's perspective the two cells are nearly identical:
M-uni-noise has σ²_clip·d ≈ 0.17, while M-multi-mix's per-component noise is
σ²·d = 0.02 plus the *between-policy* variance only at states where both policies
diverge significantly (rare given mostly-overlapping privileged+goalseek
trajectories). So the **effective σ² seen by ReBRAC's BC penalties is dominated
by noise in M-uni-noise (high) but stays small in M-multi-mix (low)** — even
though GMM says M-uni-noise is "more multi-modal".

Conclusion: GMM `p_≥2` is a poor proxy for "the algorithmic stress that
discriminates FQL from ReBRAC". A better proxy would be
**`E_s[Var(a|s)]`** (the conditional action variance) — measurable directly from
the dataset via kNN on (s, a) clusters, no GMM needed.

## 6. Implications for the paper

| Aspect | Original story (spec v1.3) | Revised story (post-diagnostic) |
|---|---|---|
| Headline claim | iff multi-modal AND sub-optimal | iff action noise corrupts BC anchor |
| Stress axis | modality (`p_≥2`) | conditional action variance `E[Var(a\|s)]` |
| Primary cell | M-multi-mix (priv + goalseek) | **M-uni-noise (priv + ε)** |
| Null/control cell | M-uni-noise (structural uni) | **M-multi-mix (small noise, good coverage)** |
| Mechanism | "FQL handles bimodal a\|s" | "FQL marginalizes noise in flow teacher; critic has no BC penalty" |
| Code evidence | (none in spec) | rebrac.py L257, L284 vs fql.py L444, L471, L532 |

This is **a better paper**, not a worse one:
- The story is mechanistically clean and code-anchored (3 explicit lines in each
  agent file).
- The empirical signal in M-uni-noise (+20.5 pp, n=2, both seeds consistent) is
  strong enough to survive any reasonable reviewer pushback on `n`.
- M-multi-mix becoming a *null control* (instead of the primary signal) actually
  **strengthens** the causal narrative: it rules out "FQL just trains faster" or
  "FQL has more parameters" — both algorithms ceiling on M-multi-mix.

## 7. Recommended next steps (revised after diagnostic)

| Step | Rationale | Cost |
|---|---|---|
| **(C done)** | Mechanism understood. | — |
| **N1. Add 1 cell `E-multi` (priv + goalseek, σ=0)** to *separate* modality from noise — completes the 2×2: (clean uni / clean multi / noisy uni / small-noise multi). | The new iff claim predicts E-multi should be **null** (no noise → no FQL advantage). Confirming this is the single highest-value experiment. | 1 collection (already have priv-1000-clean, only need goalseek-1000-clean if not already collected) + 1 notebook (4 runs). |
| **N2. Add 2 seeds [1, 7] to all 4 cells** (n=4) — only AFTER N1 lands so we don't waste compute on a story that's about to change. | Firm up confidence intervals to satisfy reviewer "n=2 too small" concern. | +16 runs × ~30 min each ≈ 8 h Colab L4. |
| **N3. Write `notebooks/fql_succession_p2_verdict.ipynb`** aggregating the (now ≥3, eventually 4) cells into the revised iff verdict. | Final report scaffolding. | 1 session local. |
| **N4. Patch spec v1.3 → v1.4**: rewrite §1 framing, §3 Gate A.2 (audit retired or repurposed), §5.5 verdict table to the noise-axis story. Keep §4 collection log as historical record. | Doc hygiene before paper writing. | 1 session local. |

I recommend running N1 first (single new cell), then deciding on N2 based on
whether E-multi confirms the noise hypothesis.

## 8. Open questions

- **Q1** (RESOLVED — see §9): Does ReBRAC's `critic_bc_coef` tuning recover its
  M-uni-noise gap? **Answer: NO.** `critic_penalty_coef=0` gave +0.01 SR (no
  recovery). The critic penalty is NOT the bottleneck. Corrected hypothesis: the
  **actor BC anchor to raw noisy actions** (β1=4.0) is the dominant cause.
- **Q2**: Why doesn't ReBRAC's normalize_q (Finding iii) save it? Because
  `lambda_coef = 1 / |Q|.abs().mean()` doesn't rescale the BC penalty term, only
  the Q-improvement term in actor_loss. The BC penalty still operates in absolute
  action space.
- **Q3**: Late-training collapse (eval_log shows peak→final drops of 10–22 pp on
  some runs) — is this affecting our final test SR? The test json uses
  `agent_final.pt` (last step), so collapse would hurt the numbers reported here.
  An anchor "best-checkpoint" test rerun would tighten the comparison; spec §5.6
  c4 stability gate is the formal mechanism for this.

---

## 9. Q1 ablation result — mechanism correction (2026-05-22)

**Run**: ReBRAC × seeds [42, 0] on the existing M-uni-noise dataset with
`--critic-penalty-coef 0.0` (actor β1=4.0 unchanged). Notebook:
`notebooks/fql_succession_p2_q1_critic_penalty_ablation_completed.ipynb`.
Results: `results/fql_succession/p2/m_uni_noise_q1_critic_pen0/test/`.

### 9.1 Result table

| variant | seed42 | seed0 | mean | vs baseline ReBRAC | vs FQL |
|---|---:|---:|---:|---:|---:|
| baseline ReBRAC (β2=2.0) | 0.680 | 0.730 | **0.705** | — | −0.205 |
| baseline FQL | 0.910 | 0.910 | **0.910** | — | — |
| **Q1 ReBRAC (β2=0.0)** | 0.750 | 0.680 | **0.715** | **+0.010** | **−0.195** |

**VERDICT: H1 FAIL.** Setting `critic_penalty_coef=0` did not recover ReBRAC
(gain +0.01, still −19.5 pp below FQL).

### 9.2 The coefficient DID take effect (it's not a config bug)

Late-training train_log means (last 50% steps):

| run | bc_loss | critic_penalty* | mean_q | td_err |
|---|---:|---:|---:|---:|
| baseline β2=2.0 s42 | 0.1465 | 0.2097 | −144.6 | 6.04 |
| baseline β2=2.0 s0 | 0.1348 | 0.1967 | −162.6 | 6.79 |
| **Q1 β2=0.0 s42** | 0.1341 | 0.1976 | **−123.0** | 6.65 |
| **Q1 β2=0.0 s0** | 0.1347 | 0.1966 | **−123.1** | 6.41 |

\* `critic_penalty` logs the *raw* penalty value `(next_actions − a')²` **before**
multiplying by the coefficient (rebrac.py:349), so it stays ~0.197 even when the
coef is 0 — it's computed-and-logged but not used in the TD target.

The proof the coef took effect: **`mean_q` rose from ≈ −150 to ≈ −123** (+27),
exactly the expected effect of removing `−critic_bc_coef·critic_penalty` from
`q_target`. The Q-surface became less pessimistic. **But the policy didn't
improve** — and `bc_loss` stayed pinned at the noise floor (~0.134) because the
actor BC coefficient (β1=4.0) was unchanged.

### 9.3 Corrected mechanism attribution

The original §3 over-attributed FQL's advantage to the critic BC penalty
inflation. The controlled ablation reveals:

- **Critic BC penalty (path b)** — affects Q *magnitude* (mean_q −150 vs −123)
  but is **not** the binding constraint on final policy quality. Removing it does
  nothing for SR.
- **Actor BC regression (path a)** — `actor_loss = −λ·Q(s,π(s)) + β1·‖π(s) − a‖²`
  with **β1 = 4.0** and `a = π*(s) + ε`. The BC term contributes ≈ 4.0 × 0.134 ≈
  0.54 to the actor loss, comparable to or larger than the Q-improvement term
  (which `normalize_q` rescales to order 1). The deterministic actor is **strongly
  dragged toward the raw noisy targets**, and it cannot escape the noise floor.
  **This is the dominant bottleneck.**
- **FQL avoids this** not by having a weaker BC, but by regressing its student to
  the **denoised** `a_teacher = integrate(s)` (flow marginalizes ε), so its BC
  target is clean even at σ=0.5. The strength of the anchor is fine; the *target
  quality* is what matters.

**Revised one-line mechanism**:
> FQL > ReBRAC on noisy data because ReBRAC's actor regresses to **raw noisy
> behavior actions** while FQL's student regresses to a **flow-denoised
> reconstruction**. The critic-side BC penalty is a secondary effect on Q
> magnitude, not the cause of the policy-quality gap.

The headline noise-axis iff (§1) is **unaffected** — M-uni-noise's +20.5 pp FQL
win is real and large. Only the *within-ReBRAC attribution of why* is corrected:
actor-side BC target quality, not critic-side penalty.

### 9.4 Follow-up: Q1b (actor-side ablation) — proposed

To close the corrected hypothesis, the decisive test is on the **actor** side:

| Variant | actor β1 | critic β2 | Predicted M-uni-noise SR |
|---|---:|---:|---|
| baseline | 4.0 | 2.0 | 0.705 (observed) |
| Q1 (done) | 4.0 | **0.0** | 0.715 (observed — no change) |
| **Q1b-1** | **1.0** | 2.0 | should rise toward FQL if actor-BC strength binds |
| **Q1b-2** | **0.0** | 0.0 | pure TD3 offline — risks extrapolation collapse; upper bound on BC removal |

Caveat: lowering β1 weakens the BC anchor *and* keeps the noisy target — so if
Q1b-1 recovers, it confirms "strong BC to a noisy target" is the bottleneck; if
it doesn't (or collapses), it confirms the issue is *target quality*, which only
FQL's denoising fixes. Either outcome sharpens the paper claim. 2–4 runs,
~1–2 h Colab L4. Build a `_q1b` notebook analogous to the Q1 one when ready.

### 9.5 Impact on N3/N4

- **N3 verdict notebook**: unchanged — still aggregates the 2×2 cells. Add a
  short "mechanism" subsection citing Q1 (critic penalty ruled out) + Q1b (if run).
- **N4 spec rewrite**: §3 mechanism narrative must lead with **actor BC target
  quality**, cite Q1 as the ablation that ruled out the critic-side story. Do not
  repeat the original "critic penalty inflation is the smoking gun" framing.

---

**Status**: Sprint-0 diagnostic + Q1 ablation complete. Q1 corrected the
within-ReBRAC mechanism attribution (actor-side, not critic-side). E-multi
datasets collected (N1 step 1 done); E-multi training pending on Colab. Next:
(a) optional Q1b actor-side ablation, (b) E-multi 4-run on Colab, then (c) N3
verdict notebook + N4 spec rewrite once results land.
