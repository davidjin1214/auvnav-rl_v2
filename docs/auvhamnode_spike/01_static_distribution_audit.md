# Step 1 — Static Distribution Audit (AUVHamNODE training vs PlanarRemusEnv deployment)

**Date:** 2026-05-13
**Goal:** quantify the distribution shift between AUVHamNODE's training corpus and the PlanarRemusEnv deployment corpus along **every axis the model conditions on**, without writing any inference / adapter code. Produce a single go / no-go judgment for v3.0 plan structure.

**Method:** read provenance.json + config.json + wake_data metadata; run [`_wake_stats.py`](_wake_stats.py) over the 6 wake `.npy` files; cross-reference [`auv_nav/vehicle.py`](../../auv_nav/vehicle.py), [`auv_nav/env.py`](../../auv_nav/env.py), [`auv_nav/autopilot.py`](../../auv_nav/autopilot.py) for deployment-side actuator + control specs.

---

## 1. Side-by-side: 7 axes the model conditions on

| # | Axis | AUVHamNODE training (seed 45) | PlanarRemusEnv deployment | Shift verdict |
|---|---|---|---|---|
| 1 | **Inertial position `x`** | depth ∈ [0.5, 100] m; no absolute-x context (`absolute_depth_context=false`) | depth held at 0 m (z_ref_m=0, depth_fail_tol_m=1.5) by hidden autopilot; planar (x,y) anywhere in ROI | **OK — depth-translation-invariant.** Model does not condition on absolute position. |
| 2 | **Body-frame velocity `v_r` (relative to water)** | `init_surge ∈ [0.8, 2.5]` m/s; clip [4.0, 0.8, 0.8] | target cruise 1.5 m/s; max from PID never exceeds RPM 1200 ⇒ ≲ 2 m/s steady-state surge | **OK — well inside training.** |
| 3 | **Body-frame angular rate `ω`** | `init_angular_std=0.15` rad/s; clip [0.5, 0.5, 0.5] | yaw rate driven by 15° rudder + PD heading controller → bounded ≲ 0.5 rad/s | **OK — borderline; PID-bounded.** |
| 4 | **Actuator state `u_actual = [δ_r, δ_s, RPM]`** | δ_max=15°, RPM∈[400, 1400]; τ_act_init=(0.1, 0.1, 1.0) s | δ_r_max=δ_s_max=15° (exact match); max_rpm_command=1200 (in range); rpm_tau=0.80s (close to training init 1.0s); δ_tau=0.25s (slower than training init 0.1s, but learnable so unclear) | **OK — actuator ranges align; τ may have drifted during training.** |
| 5 | **Commanded actuator `u_cmd`** | Excitation mix: PRBS 40% + CHIRP 35% + OU 25% — broad temporal frequency content | Heading PD output δ_r = 1.4·err - 0.4·yawrate, smoothed at control_dt=0.5s; RPM = const cruise; stern δ_s from depth autopilot | **OK — narrower than training but in-distribution.** PID outputs are a *strict subset* of PRBS/CHIRP/OU excitation. |
| 6 | **Ocean current `v_c_n` (inertial)** | `[0.0, 0.5]` m/s magnitude; constant within each 0.2s block; vertical std 0.05 | wake_data `U_ref ∈ {1.0, 1.5}` m/s freestream; vortex regions extend higher; **see §2 stats** | **❌ FAR OUT OF DISTRIBUTION — see §2** |
| 7 | **Control horizon** | block-iid 0.2 s training; benchmark 60 s clean rollout | episode 240 s; per-step `control_dt=0.5s` ⇒ inner block must be split | **OK *iff* augmentation is strictly 1-step.** Long-horizon rollouts (>60s) not supported; v2.1 §5 already constrains to 1-step. |

---

## 2. Wake-data flow statistics vs training current range

Computed by [`_wake_stats.py`](_wake_stats.py) over all 6 wake `.npy` files (full-volume float32 percentiles on `speed = sqrt(u² + v²)`):

| File (short) | U_ref | shape (T,Nx,Ny,C) | speed_med | speed_p95 | speed_p99 | speed_max | \|u\|_med | \|v\|_p99 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| v8_U1p00_Re150 (single) | 1.00 | (1200, 320, 100, 3) | **1.030** | 1.254 | 1.312 | 1.432 | 0.998 | 0.603 |
| v8_U1p50_Re250 (single) | 1.50 | (1200, 320, 100, 3) | **1.580** | 1.963 | 2.088 | 2.378 | 1.502 | 1.157 |
| tandem_G35_U1p00 | 1.00 | (1200, 510, 120, 3) | **1.046** | 1.211 | 1.287 | 1.560 | 1.037 | 0.535 |
| tandem_G35_U1p50 | 1.50 | (not loaded — OneDrive on-demand timeout) | — | — | — | — | — | — |
| sbs_G35_U1p00 | 1.00 | (1200, ?, ?, 3) — not loaded | — | — | — | — | — | — |
| sbs_G35_U1p50 | 1.50 | (1200, ?, ?, 3) — not loaded | — | — | — | — | — | — |

**Note on missing files:** the 3 larger wake files (440 MB + 794 MB × 2 = ~2 GB total) live on OneDrive cloud storage and hit a `[Errno 60] Operation timed out` during on-demand sync. Re-run `experiments/auvhamnode_spike/_wake_stats.py` on a host with the files fully materialised (e.g., Colab Drive mount) to fill in. The 3 processed cases already establish the U=1.0 vs U=1.5 trend; sbs/tandem variants differ in spatial structure but `U_ref` dictates the magnitude scale, so the missing rows will land in the same brackets.

**Training current upper bound:** `current_speed_range = [0.0, 0.5]` m/s.

### 2.1 Shift ratios (wake / training)

All ratios use **training upper bound = 0.5 m/s** (`current_speed_range[1]`) as denominator. **Anything >1 means the wake value is outside the training support.**

| Scenario | wake median / 0.5 | wake p95 / 0.5 | wake p99 / 0.5 | wake max / 0.5 |
|---|---:|---:|---:|---:|
| v8_U1p00 (single, freestream 1.0) | **2.06×** | 2.51× | 2.62× | 2.86× |
| v8_U1p50 (single, freestream 1.5) | **3.16×** | 3.93× | **4.18×** | 4.76× |
| tandem_G35_U1p00 | **2.09×** | 2.42× | 2.57× | 3.12× |

**Reading this table:**
- Even the *median* wake speed is **2-3× above the training upper bound**. Not a fringe issue — the typical value the model sees is OOD.
- The p99 in the U=1.5 scenario is **4.2× above bound** ≈ 2.1 m/s vs 0.5 m/s training cap. The model has *literally never seen* a current value in this regime.
- max can reach **4.8× above bound** in vortex cores during transverse crossings.

For comparison, MBPO / SynthER-style 1-step augmentation under cross-domain transfer typically reports stable behavior when condition shift stays **within 1.5–2×** of training distribution (and that is for *learned* dynamics on the same environment with mild OOD; physics-structured priors give a bit more margin, but not unbounded).

### 2.2 Spatial structure

The training current `v_c_n` is **spatially uniform** within each control block (it is a single inertial-frame 3-vector held constant). The wake flow is **spatially heterogeneous**: `U_ref` is the freestream, but the wake exhibits velocity deficit regions and counter-rotating vortices. The AUVHamNODE model never sees a spatial gradient in `v_c_n`. When the AUV traverses a vortex, the *effective current* changes both magnitude and direction between successive 0.2s blocks; this is fundamentally different from the training distribution where `v_c_n` is piecewise-constant.

**Mitigating factor:** the AUV's body length is ~1.6 m and the wake spatial resolution is `dx=0.6m`. Over a 0.2s block at 1.5 m/s, the vehicle traverses 0.3 m ≈ 0.5 grid cells. Within one block, the current is *approximately* spatially uniform from the model's perspective — the spatial gradient mostly shows up *between* blocks. So the block-iid training does not directly *contradict* the spatially heterogeneous deployment, but it also gives no signal about gradient-driven dynamics.

### 2.3 Effective-current view (hull-integral, `EquivalentCurrentModel`)

`auv_nav/autopilot.py:EquivalentCurrentModel` (used in privileged-obs / asym-critic line) samples the wake at 5 hull points and forms a weighted average. This integrated `[u_eq, v_eq]` smooths the spatial variation and is the variable that **actually drives the AUV dynamics**. For typical cruise:
- `|u_eq|` is dominated by `u_mps` (streamwise) ≈ `U_ref ± deficit`. Order of magnitude: 1.0 (U=1.0) / 1.5 (U=1.5) m/s.
- `|v_eq|` is the integrated transverse component, magnitudes ≤ `|v|_p99` ≈ 0.6 (U=1.0) / 1.2 (U=1.5) m/s.

So a hull-integral `|v_c|` in the U=1.5 case sits at ~1.5–2.0 m/s, **3–4× the AUVHamNODE training upper bound**. The U=1.0 case sits at ~1.0–1.3 m/s, **2.0–2.6×** above bound.

---

## 3. Other relevant facts (cross-checked, not in the shift table)

| Fact | Source | Implication for v3.0 |
|---|---|---|
| `dt_ctrl=0.2s` (training) vs `control_dt=0.5s` (env) | `provenance.json` / `env.py:273` | Each env step = 2.5 training blocks. 1-step augmentation must call `single_step(dt=0.2)` 2–3 times with action held constant, *not* `single_step(dt=0.5)` once (which would exceed training horizon and hold `u_cmd` constant longer than training data ever did). |
| Actuator τ: trained as learnable, init=(0.1, 0.1, 1.0)s vs PlanarRemusEnv (δ_tau=0.25, rpm_tau=0.80) | `config.json` / `vehicle.py:186` | Cannot verify post-training τ without unpacking checkpoint weights. δ_τ mismatch (0.1 init vs 0.25 truth) is a candidate source of NODE → planar 1-step error; spike Check A (in-training-distribution sanity) will reveal whether the learned τ matches PlanarRemusEnv. |
| `absolute_depth_context=false` | `provenance.json:42` | Model is depth-translation-invariant. Excellent for cross-domain transfer: PlanarRemusEnv operates at z=0, AUVHamNODE trained over [0.5, 100]m, but model does not condition on absolute depth. |
| `u_dim=3` only (no roll dim) | `provenance.json:63` | REMUS 100 4-DOF (roll passive) matches PlanarRemusEnv's planar 3-DOF assumption (roll=0 fixed). No mismatch. |
| `noise_profile=clean` only | `provenance.json:60` | Model has not seen noisy IC. If we add small σ_a perturbations for v2.1 §4.2 B test, signal should be clean but the model has no training robustness margin — perturbed inputs may give large prediction errors even at small σ. |
| Eval clean 60s median pos err = 0.43m | `provenance.json:50` | Even *in-distribution* the model drifts ~0.4m over 60s = 7 mm/s positional drift rate ≈ 0.4% of 1.5 m/s cruise. This is the floor for 1-step accuracy; cross-domain will be worse. |

---

## 4. Verdict

### 4.1 Headline

> **6 of 7 conditioning axes align well. The 7th (ocean current `v_c_n`) is 2-4× outside training distribution, with the typical (median) wake value already above the training maximum. This is large enough that v2.1 §4.2 A's "整体 normalized MSE < 0.1" gate is unlikely to pass at U=1.5; possibly marginal at U=1.0.**

### 4.2 What this rules in / rules out

**Stays viable:**
- Adapter spike (Step 3) is **not** doomed by axes 1–5. Position, velocity, angular rate, actuator state, and `u_cmd` distributions all align. The adapter machinery itself (lift/project + PID + dt slicing) can be tested on near-zero current first to confirm code correctness, independent of the current-shift issue.
- The 1-step augmentation premise is still tenable *if* paper claims are scoped accordingly (see §5 below).
- Depth-translation-invariance + identical `δ_max` + `u_dim=3` mean the model is a serious cross-domain candidate **except for current magnitude**.

**Materially weakened:**
- v2.1 §4.2 A's MSE threshold "< 0.1 normalized" is **not realistic** on U=1.5 wake at the current-magnitude p99. Threshold either needs to be re-derived empirically (after spike Check B) or relaxed to "model error < known baseline error" (a relative, not absolute, criterion).
- §10.1 paper claim "frozen physics prior enables generic cross-domain transfer" must be **scoped** — the claim now reads "transfer holds for current magnitudes within ~2× training; degrades sharply outside."
- The `efficiency_v2 → arrival_v2` story implicitly assumed upstream tasks (`U=1.5`, freestream-against-target) would work. The audit shows the **upstream u15 scenario is the worst-case for AUVHamNODE** — it is exactly the highest-current scenario. The Asymmetric Critic + ReBRAC fallback (without dynamics aug) is the only credible play if upstream is required.

**Ruled out:**
- v2.1 §4.2 A test with the literal "< 0.1 normalized" gate. This number was carried over without empirical grounding; it cannot survive Step 1's evidence.
- v3.0 paper claim of broad cross-domain transfer without explicit current-magnitude scoping.

### 4.3 Confidence

| Claim | Confidence |
|---|---|
| Axes 1–5 align | **High** — direct numeric comparison; identical δ_max; RPM range subset; `absolute_depth_context=false`. |
| `v_c_n` shift ≥ 2× | **High** — direct empirical measurement on 3 wake files; remaining 3 fall in same magnitude class by `U_ref`. |
| 1-step augmentation MSE will fail v2.1 §4.2 A literal threshold at U=1.5 | **Medium-high** — extrapolated from in-distribution clean eval (0.43m / 60s); 2-4× current shift typically adds at least 1-2 orders of magnitude error growth in nonlinear dynamics models. Empirical confirmation requires Step 3 spike. |
| Adapter code itself works | **Unverified** — Step 3 needed. Independent of §1 audit. |

---

## 5. Recommendations for the user

There are 4 paths forward. They are not mutually exclusive, and the audit data lets us narrow the choice substantially.

### Path 1 — Scope v3.0 to low-current regime only (RECOMMENDED first choice)

**Premise:** the audit shows U=1.0 wakes sit at ~2× current shift (median 1.03 m/s vs 0.5 cap) — large but **not catastrophic**. U=1.5 wakes sit at 3-4× — likely catastrophic. Keep AUVHamNODE in the loop *only* for U=1.0 scenarios.

**Concrete framing:**
- Phase 1 baseline = U=1.0 wakes (`single_u10_*`, `tandem_u10_*`, `sbs_u10_*`).
- Drop U=1.5 from Phase 1.
- Paper claim: "frozen port-Hamiltonian NODE provides physically-grounded 1-step prior for offline MBRL at low-to-moderate current magnitudes (≤ 2× training cap); precision degrades sharply at high current, motivating a finetune-the-prior follow-up."
- Existing `cross_stream + u10 + efficiency_v2` evidence (A0 96.7% success per `docs/online_sac_reward_redesign.md`) shows U=1.0 is the safe regime; rebrand U=1.0 from "easy mode" to "primary regime."

**Pros:**
- Most likely to give a working spike + working Phase 1.
- arrival_v2 already de-emphasises upstream; U=1.0 is the geometry where neither efficiency_v2 nor arrival_v2 hacks.
- ReBRAC `cross_stream_u10` mainline result already exists as a reference for fair comparison.

**Cons:**
- Cannot ablate against upstream u15 task (the hardest case).
- Paper claim loses the "general" framing it had in v2.1 §10.1.

**Cost:** 1 additional spike day (Step 3 narrowed to U=1.0). Phase 1 unchanged in scope but constrained in regime.

### Path 2 — Finetune AUVHamNODE on wake-compatible flows

**Premise:** keep the AUVHamNODE structural prior but adapt to wake current statistics. Generate ~100-500 trajectories with `v_c_n` magnitudes drawn from [0.5, 2.5] m/s (covering both U=1.0 and U=1.5 wakes), finetune for 1-2 epochs with low LR.

**Pros:**
- Restores the broad-regime paper claim.
- Reuses all upstream infrastructure (training loop, eval pipeline, provenance audit).
- Cheap on Colab L4 (1-3 hours).

**Cons:**
- Requires the upstream training pipeline to be accessible. We only have the *export* (checkpoints + inference API), not the training code.
- Loses the "frozen prior" narrative — paper becomes "finetuned NODE prior" which is less novel.
- Risk of overfitting / forgetting if finetune dataset is small.

**Cost:** depends on access to upstream training repo (`g3_5_5` / `g3_5_7` per provenance). If accessible, 2-3 days including data regen + finetune + re-verify. If not, this path is closed.

### Path 3 — Constrain wake_data to ≤ 0.5 m/s magnitude

**Premise:** rescale wake `U_ref` to 0.3-0.4 m/s so all wake values stay within training distribution. This is a clean v3.0 framing but changes the underlying physics task.

**Pros:**
- Audit passes trivially.
- Spike likely succeeds.

**Cons:**
- Sample-and-hold scaling breaks Re and St dimensionless similarity — the wake becomes physically inconsistent (wake_v8 generation used CuPy DNS at specific Re; rescaling U breaks the regime).
- Re-generating wakes at a lower U_ref is possible but requires the wake generator (CFD pipeline) and is expensive (~13 min per `1200f_roi` per the `elapsed_s` field).
- Lower current speed reduces the navigational challenge — defeats the thesis premise of "challenging wake-field navigation."

**Not recommended** as a primary path. Could be a baseline ablation if the storyline supports it.

### Path 4 — Skip AUVHamNODE entirely; build MBRL with `auv_nav/vehicle.py` as the dynamics model

**Premise:** the project already has a verified Remus100 6-DOF simulator. It is **deterministic and oracle-accurate** by construction. Use it as the dynamics-augmentation source instead of a learned NODE.

**Pros:**
- Zero cross-domain transfer issue — the simulator already operates on wake flows.
- Eliminates all 4 hard interface mismatches from pre-notes §3.
- Existing planar 3-DOF abstraction maps 1:1 to vehicle.py without lift/project.
- Determinism + ground-truth eliminates the v2.1 §4.2 A MSE gate entirely.

**Cons:**
- Loses the "physics-structured NN prior" novelty — paper contribution becomes "oracle-grounded model-based offline RL," which is closer to PETS/MBPO methodology than to NODE-as-prior.
- The whole AUVHamNODE work this pre-notes was scoped around becomes irrelevant to the v3.0 plan (though it would remain available for an *online* sim2real story later).
- "Why use a simulator we already have" is harder to motivate as a contribution.

**Cost:** medium pivot — requires rewriting v2.1 §1, §3, §10 with a different narrative. But Phase 1 implementation may be cheaper (no NODE inference cost; no spike needed). 1-2 weeks to revised plan + 4-6 weeks to results.

**Not recommended** *yet* — Path 1 is the natural next step, and Path 4 remains as a fallback if Path 1 also fails (in spike Check B).

### Decision matrix

| Path | Audit-supported? | Spike still needed? | Paper claim survives? | Estimated time to v3.0 |
|---|---|---|---|---|
| 1. Scope to U=1.0 only | ✅ Yes (2× shift, marginal but plausible) | Yes, narrowed | Scoped survival | 2-3 days (spike + plan) |
| 2. Finetune NODE | ✅ Yes (restores in-distribution) | Yes | Modified ("finetuned prior") | 1 week (upstream access needed) |
| 3. Rescale wake | ✅ Yes (trivially) | Yes | Survives but framing weakens | 2-4 days (wake regen) |
| 4. Drop NODE, use vehicle.py | N/A | No | Different paper | 1-2 weeks pivot |

### My recommendation (single)

**Proceed with Path 1: scope v3.0 to U=1.0 wakes and run Step 3 (adapter spike) at U=1.0.** Reasons:

1. The audit data places U=1.0 at the **borderline**, not the catastrophic regime. Spike result will give a real number, not extrapolation.
2. The user's pre-notes §9 question 4 already flagged this as the most likely fork. Step 1 confirms the fork is real and points to U=1.0 as the safe side.
3. Path 2 (finetune) is a strict superset capability — it can be added later if Path 1 spike passes but we want broader regime claims.
4. Path 4 (drop NODE) is the safest fallback; postponing it does not damage it.

**If the user accepts:** Step 3 spike runs on a U=1.0 wake transition (1-2 days), with the spike's Check B explicitly using `wake_v8_U1p00_Re150` as the cross-domain test stimulus, not U=1.5.

**If the user wants a faster kill-test:** I can do a "spike-lite" in 4-6 hours that runs `single_step(dt=0.2)` on 100 dataset transitions from a U=1.0 episode and reports MSE distribution — no full adapter, just hand-built lift/project on a few seeds — to confirm or refute Path 1 viability before committing 1-2 days to Step 3.

---

*Step 1 complete. Awaiting user decision between Path 1 (default), Path 2 (finetune), Path 4 (drop NODE), or spike-lite kill-test.*
