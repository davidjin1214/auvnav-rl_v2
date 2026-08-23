# Step 4 — Swap `vehicle.py` → `remus100_core.py`? Decision Memo

**Date:** 2026-05-13
**Context:** the user asked, after Step 3 audit:
> *"如果用 `phnode_full_oc_clean/reference_simulator/remus100_core.py` 代替原有的 `auv_nav/vehicle.py`, 那么 NODE 是否就没有问题了呢?"*

This memo answers that question with a structured pro/con analysis, evidence pulled from Steps 1 and 3, and a recommendation.

---

## Headline

**No** — swapping `vehicle.py` for `remus100_core.py` solves the **dynamics-formula** half of the NODE deployment problem (~30% of total) but leaves the **dominant** half — **wake current OOD + spatial heterogeneity + control-period mismatch + frame-representation gap** — entirely untouched. Below is the breakdown.

---

## 1. What the swap **would** fix

All 7 of the dynamics divergences cataloged in Step 3 ([`03_dynamics_consistency_audit.md`](03_dynamics_consistency_audit.md) §14) disappear instantly:

| Step 3 § | Issue | Resolved by swap? |
|---|---|---|
| §2 | `geometry_scale=1.0096` (mass +3%, Ix/Iy +5%) | ✅ |
| §3 | Gravity 9.81 vs Somigliana 9.82 (+0.12%) | ✅ |
| §4.1 | Actuator τ (0.10/1.00 vs 0.25/0.80) | ✅ |
| §4.2 | Actuator integrator (joint RK4 vs Euler+rate-limit) | ✅ |
| §5 | Cross-flow Cd_2D (~1.20 const vs 0.25–0.80 Re/AR) | ✅ |
| §6 | Oswald e (0.7 vs 0.3, induced drag 2.33×) | ✅ |
| §7 | Propeller inflow Va (axial relative vs 3-D body) | ✅ |
| §8 | Current frame (full 3D R^T vs planar-yaw) | ✅ (numerically equivalent in planar already) |

These together give NODE physics ≡ env physics. **Necessary condition met.**

---

## 2. What the swap does **not** fix (the dominant 70%)

All four of the following are deployment-side problems that live outside `vehicle.py`. Swapping `vehicle.py` does not touch any of them.

### 2.1 Wake current magnitude is 2–4× over NODE training cap (Step 1 §2)

> source: [`01_static_distribution_audit.md`](01_static_distribution_audit.md) §2.1

NODE training distribution: `‖v_c^n‖ ∈ [0.0, 0.5] m/s`, uniform inertial.
Wake `.npy` files actually contain:

| Wake | median / 0.5 | p99 / 0.5 |
|---|---:|---:|
| v8_U1p00_Re150 (single, U=1.0) | **2.06×** | 2.62× |
| v8_U1p50_Re250 (single, U=1.5) | **3.16×** | **4.18×** |
| tandem_G35_U1p00 | 2.09× | 2.57× |

NODE is a neural network: behavior outside the training manifold is undefined.

**This is the dominant threat to NODE reliability in the wake task, and swapping `vehicle.py` does not change a single number in this table.** The wake-flow magnitudes come from `wake_data/*.npy` files, which are sampled by `FlowSampler.sample_probes_body()` (`auv_nav/flow.py`, called from the env) and fed to the NODE as a conditioning input. Whether `vehicle.py` integrates with one formula or another is irrelevant to the wake-file content.

### 2.2 Wake current is spatially heterogeneous (Step 1 §2.2)

NODE training: `v_c^n` is **block-constant in space** — one 3-vector covers the whole AUV for an entire 0.2 s block.
Wake `.npy`: spatial gradient field — vortex cores at ~3× freestream, deficit regions at ~0.7× freestream, AUV body length 1.6 m crosses ~2.7 grid cells.

NODE has *never* seen v_c^n change between two consecutive blocks by 1 m/s. This is a **structural** distribution shift, not a magnitude one. Not addressed by `vehicle.py` swap.

### 2.3 Control-period mismatch (Step 1 §3)

| Side | Period | u_cmd held constant for |
|---|---|---|
| NODE training | `dt_ctrl = 0.2 s` | 0.2 s |
| PlanarRemusEnv | `control_dt = 0.5 s` | **0.5 s — 2.5× longer** |

NODE has never seen a u_cmd plateau longer than 0.2 s. A 1-step augmentation with `single_step(dt=0.5)` extrapolates beyond training horizon for the actuator response. The fix is adapter-side (chain 2–3 `single_step(dt=0.2)` calls with held u_cmd), not `vehicle.py`-side.

### 2.4 Privileged-obs vs `v_c^n` frame gap (Step 1 §2.3)

PlanarRemusEnv emits `privileged_obs = [u_eq, v_eq]` (body-frame, hull-integral via `EquivalentCurrentModel` 5-point weighted sum).
NODE conditions on `v_c^n` (inertial frame, single 3-vector).

These are **different physical quantities**. An adapter must:
- forward: convert body-frame integrated `[u_eq, v_eq]` back to an inertial single-point `v_c^n` — non-trivial because the hull-integral discards the spatial structure
- backward: project NODE's inertial response back into body frame for the planar policy

Neither direction is in `vehicle.py`.

---

## 3. What the swap **costs**

Engineering cost and scientific cost.

### 3.1 Engineering

| Concern | Impact |
|---|---|
| State-vector interface change (12-D Euler ↔ 16-D quat + nu_r) | `env.py`, `autopilot.py`, `baselines.py`, `replay.py` (privileged_obs path) all touch `Remus100`'s public API |
| Actuator interface change (separate `step_actuators` ↔ in-state RK4) | env `step()` loop, `collect_offline_data.py` action-recording, `train_sac.py` env-rollout — all need restructure |
| Loss of `LowPassFilter`, `ALOS2D`, `IntegralSMCHeading`, `DepthController`, `LOSObserver` utilities | reference does not ship equivalents; either keep them outside Remus100 (current arrangement) or write replacements (large effort) |
| Loss of `Remus100.compute_relative_flow` (PlanarRemusEnv-specific) | env relies on this to populate observation; reference's `relative_velocity` returns `nu_r` only, no `RelativeFlow` dataclass |

Estimated rewrite cost: **3–7 working days** of focused refactor, plus 1–2 days of regression debugging on previously-green baselines.

### 3.2 Scientific (paper baselines)

Every published result was generated against the current `vehicle.py`. After swap, **none** of these remain comparable:

| Asset | Generated against | Comparability after swap |
|---|---|---|
| A0 cross_u10 96.7% success result (`docs/online_sac_reward_redesign.md`) | current `vehicle.py` | broken |
| ReBRAC paper-readiness 4/4 closed (rev.8) | current `vehicle.py` | broken |
| ReBRAC broad validation S2 P1 (running on Colab) | current `vehicle.py` | broken |
| 47 planned online thesis runs (`online_rl_thesis_plan.md`) | current `vehicle.py` | not yet run |
| All `benchmarks/*.json` fixed eval manifests | current `vehicle.py` (manifests sample IC + record reference trajectories) | likely broken; would need regeneration |

The ReBRAC mainline alone represents ~3–4 weeks of accumulated Colab GPU time and the paper revision is already in progress.

### 3.3 Hybrid mitigation

A `--dynamics-mode {legacy,node_reference}` flag in `VehicleParams` could be added so old experiments stay reproducible. This is the conservative engineering choice if a swap is ever made, but it doubles the test surface area.

---

## 4. Decision matrix

| Path | Solves dynamics gap? | Solves current OOD? | Solves control-period? | Solves frame gap? | Engineering cost | Existing-baseline blast radius |
|---|---|---|---|---|---|---|
| **A. Status quo** (do nothing; run spike-lite first) | ✗ (lives with bias) | ✗ | ✗ | ✗ | 0 | 0 |
| **B. Swap `vehicle.py`** | ✅ | ✗ | ✗ | ✗ | high (3-7 d) | **all online + offline baselines invalidated** |
| **C. Scope-down to U=1.0 only** (audit Path 1) | ✗ | partial (2× shift, marginal) | ✗ | ✗ | low (env config only) | 0 (uses existing U=1.0 results as ref) |
| **D. Finetune NODE on wake-magnitude currents** (audit Path 2) | ✗ (NODE adapts to its own dynamics) | ✅ (if data is regenerated) | partial | ✗ | medium (depends on upstream-repo access) | 0 (NODE side only) |
| **E. Drop NODE entirely; `vehicle.py` as oracle dynamics** (audit Path 4) | ✗ (no NODE involved) | ✅ (oracle on wake) | ✅ (we control dt) | ✅ (we control frame) | medium-high (re-scope paper §1, §3, §10) | medium (existing baselines OK; paper narrative changes) |
| **F. Hybrid B+D** (swap + finetune) | ✅ | ✅ | partial | ✗ | very high (1–2 weeks) | all baselines invalidated |

---

## 5. The case **against** the swap, in one sentence

> The swap solves divergence #5 out of 5 in priority order (dynamics formulae), at the cost of resetting every existing baseline, without touching divergences #1–4 (current magnitude OOD, current spatial heterogeneity, dt_ctrl mismatch, frame gap).

---

## 6. Recommendation

**Do not swap.** Run **Path A → Path C** in sequence:

1. **Spike-lite ([`02_spike_lite_design.md`](02_spike_lite_design.md))** — half day, ~5 hours.
   - Use the *current* `vehicle.py` for env transitions; feed them into NODE; measure 1-step MSE on U=1.0 wake.
   - This conflates the dynamics gap (Step 3 audit) with the current OOD (Step 1 audit), but that is OK at this stage — we want a single "is it 0.05 m or 0.5 m off" number to triage.
2. **If spike-lite MSE > thesis-acceptable threshold AND failure is dominated by current-OOD (verifiable by repeating the same MSE measurement on `current_speed=0.0` wake-free transitions)**:
   - Path C: scope v3.0 to U=1.0 only.
3. **If spike-lite MSE > threshold AND failure is dominated by dynamics gap (verifiable by §15 of the Step 3 audit — comparing NODE vs `Remus100Simulator` vs `Remus100` rollouts at zero current)**:
   - Now consider the swap (Path B), with explicit scope: "we are accepting the baseline reset to recover a NODE 1-step MSE < threshold."
4. **If spike-lite passes** (NODE 1-step MSE acceptable on U=1.0 wake): no swap needed; proceed with v3.0 Phase 1 on existing `vehicle.py`.

This ordering puts the **cheapest** evidence first (5 h spike), defers the **most expensive** action (swap + revalidate) to the case where it is actually justified, and avoids the failure mode where we pay 5–9 engineering days to fix a problem that wasn't the dominant one.

---

## 7. What this memo does **not** decide

- Whether to use NODE at all (Path E "drop NODE" is still on the table; decided after spike-lite outcome).
- Whether to finetune NODE (Path D) — gated on upstream repo access; outside scope here.
- Whether to add a `--dynamics-mode` compatibility flag — only relevant if Path B is ever chosen.

---

## 8. Cross-references

- [`01_static_distribution_audit.md`](01_static_distribution_audit.md) — Step 1, 7-axis training-vs-deployment distribution audit; §2 documents the wake current OOD (the dominant threat this memo argues the swap does not fix)
- [`02_spike_lite_design.md`](02_spike_lite_design.md) — Step 2, half-day kill-test design
- [`03_dynamics_consistency_audit.md`](03_dynamics_consistency_audit.md) — Step 3, formal physics-by-physics audit of `vehicle.py` vs `remus100_core.py`; this memo builds on its §14 severity table and §15 implication paragraphs
- [`docs/auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](../../docs/auvhamnode_offline_mbrl_plan_v3_pre_notes.md) — the α-path pre-notes that initiated this whole pre-spike sequence
- [`docs/auvhamnode_offline_mbrl_plan.md`](../../docs/auvhamnode_offline_mbrl_plan.md) v2.0 — the locked plan; section "Phase 0 fire condition" should be amended to include the spike-lite result as a gating artifact

---

*Step 4 complete. Recommendation: spike-lite first, swap only if dynamics gap is empirically isolated as dominant.*
